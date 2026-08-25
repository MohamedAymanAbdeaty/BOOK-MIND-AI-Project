import argparse
import uuid

# Configuration and Database Services
from app.config import Settings
from app.services.embedding import EmbeddingService
from app.services.vector_store import QdrantService

# Data Processing utilities
from scripts.ingestion.chunker import chunk_pages
from scripts.ingestion.loader import load_pdf


def ingest_pdf(path: str, book_id: str, title: str, author: str, batch_size: int = 64) -> int:
    """
    Reads a PDF, breaks it into chunks, converts text to vectors, and saves them to the database.
    """
    print(f"Loading configuration and connecting to database...")
    settings = Settings()
    embedding = EmbeddingService(settings.embedding_model)
    qdrant = QdrantService(
        url=settings.qdrant_url,
        collection_prefix=settings.qdrant_collection,
        embedding_service=embedding,
        api_key=settings.qdrant_api_key,
    )
    
    print(f"Reading PDF from {path}...")
    # 1. Load the PDF text page by page
    pages = load_pdf(path)
    
    # 2. Break the pages down into smaller overlapping "chunks"
    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError("No usable text was extracted from the PDF")

    print(f"Extracted {len(chunks)} chunks. Preparing database collection...")
    # 3. Make sure a database collection exists for this book
    qdrant.ensure_collection(embedding.dimension, book_id)
    
    print(f"Uploading to database in batches of {batch_size}...")
    # 4. Upload the chunks to the database in batches (to not overload memory)
    for offset in range(0, len(chunks), batch_size):
        # Get a slice of chunks for this batch
        batch = chunks[offset : offset + batch_size]
        
        # Convert the text into mathematical vectors
        vectors = embedding.embed_documents([chunk.text for chunk in batch])
        
        # Prepare the data points for the database
        points = []
        for chunk, vector in zip(batch, vectors, strict=True):
            # Create a unique, readable ID for this chunk
            chunk_id = f"{book_id}-p{chunk.page}-c{chunk.chunk_index}"
            
            # The payload contains the actual text and metadata
            payload = {
                "book_id": book_id,
                "book_title": title,
                "author": author,
                "page": chunk.page,
                "chapter": None,
                "chunk_id": chunk_id,
                "text": chunk.text,
            }
            
            # Qdrant requires UUIDs for IDs
            qdrant_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"bookmind:{chunk_id}"))
            
            points.append({
                "id": qdrant_uuid,
                "vector": vector,
                "payload": payload,
            })
            
        # Send the batch to the database
        qdrant.upsert(book_id, points)
        
    return len(chunks)


def main() -> None:
    """
    The main entry point for the script. 
    It reads command line arguments and runs the ingestion process.
    """
    parser = argparse.ArgumentParser(description="Ingest a book PDF into the database")
    parser.add_argument("path", help="Path to the PDF file")
    parser.add_argument("--book-id", required=True, help="Unique identifier for the book (e.g. 'meditations')")
    parser.add_argument("--title", required=True, help="The title of the book")
    parser.add_argument("--author", required=True, help="The author of the book")
    
    args = parser.parse_args()
    
    print("Starting ingestion process...")
    count = ingest_pdf(args.path, args.book_id, args.title, args.author)
    print(f"Successfully ingested {count} chunks for '{args.title}'!")


if __name__ == "__main__":
    main()
