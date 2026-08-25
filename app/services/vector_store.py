import re

# We use Pydantic models from our app.models module
from app.models.response import Source


class QdrantService:
    """
    A service to interact with Qdrant, our vector database.
    This database stores text "chunks" from books along with their mathematical representations (embeddings).
    This allows us to search for text that is conceptually similar to a user's question.
    """

    def __init__(self, url: str, collection_prefix: str, embedding_service, api_key: str = "", client=None):
        self.url = url
        self.collection_prefix = collection_prefix  # e.g., "book_chunks"
        self.embedding_service = embedding_service
        self.api_key = api_key or None
        self._client = client

    def _collection(self, book_id: str) -> str:
        """Helper to generate the collection name for a specific book."""
        return f"{self.collection_prefix}_{book_id}"

    @property
    def client(self):
        """Lazy-loads the Qdrant database client only when needed."""
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(url=self.url, api_key=self.api_key, timeout=5)
        return self._client

    def ensure_collection(self, vector_size: int, book_id: str) -> None:
        """
        Creates a new collection (like a table) in the database for a book if it doesn't exist yet.
        """
        from qdrant_client.models import (
            Distance, ScalarQuantization, ScalarQuantizationConfig, ScalarType, VectorParams,
        )
        collection = self._collection(book_id)
        
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                # INT8 quantization: Uses ~4x less memory and makes searching faster with minimal accuracy loss
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(type=ScalarType.INT8, always_ram=True)
                ),
            )

    def upsert(self, book_id: str, points: list[dict]) -> None:
        """
        Inserts or updates multiple data points (book chunks + vectors) into the database.
        """
        from qdrant_client.models import PointStruct
        
        # Convert our raw dictionaries into Qdrant PointStruct objects
        qdrant_points = [
            PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"]) 
            for p in points
        ]
        
        self.client.upsert(
            collection_name=self._collection(book_id),
            points=qdrant_points,
            wait=True,
        )

    def search(self, book_id: str, question: str, limit: int = 5, score_threshold: float = 0.36) -> list[Source]:
        """
        Searches the database for chunks of text from the book that are most relevant to the question.
        """
        if not self.collection_exists(book_id):
            return []

        from qdrant_client.models import FieldCondition, Filter, MatchValue

        # Check if the user specifically asked for a certain page number (e.g., "on page 5")
        page_match = re.search(r"\b(?:page|p\.?)\s*(\d+)\b", question, re.IGNORECASE)
        query_filter = None
        effective_threshold = score_threshold
        
        if page_match:
            # If they did ask for a specific page, we filter the database search to only that page
            target_page = int(page_match.group(1))
            query_filter = Filter(must=[FieldCondition(key="page", match=MatchValue(value=target_page))])
            effective_threshold = 0.0  # Don't filter by relevance score when the user asks for a specific page

        # Convert the user's text question into a mathematical vector
        query_vector = self.embedding_service.embed_query(question)
        
        # Perform the actual search in Qdrant
        result = self.client.query_points(
            collection_name=self._collection(book_id),
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            score_threshold=effective_threshold,
            with_payload=True, # We want the actual text data back, not just the ID
        )

        # Convert the Qdrant results into our clean Source objects
        sources = []
        for hit in result.points:
            payload = hit.payload or {}
            sources.append(Source(
                chunk_id=str(payload.get("chunk_id", hit.id)),
                page=payload.get("page"),
                chapter=payload.get("chapter"),
                text=str(payload.get("text", "")),
                # Ensure the score is between 0.0 and 1.0
                score=max(0.0, min(1.0, float(hit.score))),
            ))
            
        return sources

    def ping(self) -> bool:
        """Checks if the database is alive and reachable."""
        try:
            return bool(self.client.get_collections())
        except Exception:
            return False

    def collection_exists(self, book_id: str) -> bool:
        """Checks if a collection for a specific book already exists."""
        try:
            return self.client.collection_exists(self._collection(book_id))
        except Exception:
            return False
