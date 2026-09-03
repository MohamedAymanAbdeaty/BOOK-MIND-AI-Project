import argparse
import uuid

from app.config import Settings
from app.services.embedding import EmbeddingService
from app.services.vector_store import QdrantService
from scripts.ingestion.chunker import chunk_pages
from scripts.ingestion.loader import load_pdf


def ingest_pdf(
    path: str,
    book_id: str,
    title: str,
    author: str,
    batch_size: int = 64,
) -> int:
    print("Loading configuration...")
    settings = Settings()
    embedding = EmbeddingService(settings.embedding_model)
    qdrant = QdrantService(
        settings.qdrant_url,
        settings.qdrant_collection,
        embedding,
        settings.qdrant_api_key,
    )

    print(f"Reading {path}...")
    chunks = chunk_pages(load_pdf(path))
    if not chunks:
        raise ValueError("No usable text was extracted from the PDF")

    print(f"Found {len(chunks)} chunks.")
    qdrant.ensure_collection(embedding.dimension, book_id)

    print(f"Uploading in batches of {batch_size}...")
    for offset in range(0, len(chunks), batch_size):
        batch = chunks[offset:offset + batch_size]
        vectors = embedding.embed_documents([chunk.text for chunk in batch])
        points = [
            _make_point(book_id, title, author, chunk, vector)
            for chunk, vector in zip(batch, vectors, strict=True)
        ]
        qdrant.upsert(book_id, points)

    return len(chunks)


def _make_point(book_id: str, title: str, author: str, chunk, vector: list[float]) -> dict:
    chunk_id = f"{book_id}-p{chunk.page}-c{chunk.chunk_index}"
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"bookmind:{chunk_id}"))

    return {
        "id": point_id,
        "vector": vector,
        "payload": {
            "book_id": book_id,
            "book_title": title,
            "author": author,
            "page": chunk.page,
            "chapter": None,
            "chunk_id": chunk_id,
            "text": chunk.text,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a book PDF into Qdrant")
    parser.add_argument("path", help="Path to the PDF file")
    parser.add_argument("--book-id", required=True, help="Book ID, for example: meditations")
    parser.add_argument("--title", required=True, help="Book title")
    parser.add_argument("--author", required=True, help="Book author")
    args = parser.parse_args()

    count = ingest_pdf(args.path, args.book_id, args.title, args.author)
    print(f"Ingested {count} chunks for '{args.title}'.")


if __name__ == "__main__":
    main()
