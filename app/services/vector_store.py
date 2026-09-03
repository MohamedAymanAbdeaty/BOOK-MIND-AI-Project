import re

from app.models.response import Source


class QdrantService:
    def __init__(self, url: str, collection_prefix: str, embedding_service, api_key: str = "", client=None):
        self.url = url
        self.collection_prefix = collection_prefix
        self.embedding_service = embedding_service
        self.api_key = api_key or None
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(url=self.url, api_key=self.api_key, timeout=5)
        return self._client

    def collection_name(self, book_id: str) -> str:
        return f"{self.collection_prefix}_{book_id}"

    def ensure_collection(self, vector_size: int, book_id: str) -> None:
        from qdrant_client.models import (
            Distance,
            ScalarQuantization,
            ScalarQuantizationConfig,
            ScalarType,
            VectorParams,
        )

        name = self.collection_name(book_id)
        if self.client.collection_exists(name):
            return

        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(type=ScalarType.INT8, always_ram=True)
            ),
        )

    def upsert(self, book_id: str, points: list[dict]) -> None:
        from qdrant_client.models import PointStruct

        qdrant_points = [PointStruct(**point) for point in points]
        self.client.upsert(
            collection_name=self.collection_name(book_id),
            points=qdrant_points,
            wait=True,
        )

    def search(
        self,
        book_id: str,
        question: str,
        limit: int = 5,
        score_threshold: float = 0.36,
    ) -> list[Source]:
        if not self.collection_exists(book_id):
            return []

        query_filter, threshold = self._page_filter(question, score_threshold)
        result = self.client.query_points(
            collection_name=self.collection_name(book_id),
            query=self.embedding_service.embed_query(question),
            query_filter=query_filter,
            limit=limit,
            score_threshold=threshold,
            with_payload=True,
        )
        return [self._to_source(hit) for hit in result.points]

    @staticmethod
    def _page_filter(question: str, default_threshold: float):
        page_match = re.search(r"\b(?:page|p\.?)\s*(\d+)\b", question, re.IGNORECASE)
        if not page_match:
            return None, default_threshold

        from qdrant_client.models import FieldCondition, Filter, MatchValue

        page = int(page_match.group(1))
        query_filter = Filter(
            must=[FieldCondition(key="page", match=MatchValue(value=page))]
        )
        return query_filter, 0.0

    @staticmethod
    def _to_source(hit) -> Source:
        payload = hit.payload or {}
        score = max(0.0, min(1.0, float(hit.score)))
        return Source(
            chunk_id=str(payload.get("chunk_id", hit.id)),
            page=payload.get("page"),
            chapter=payload.get("chapter"),
            text=str(payload.get("text", "")),
            score=score,
        )

    def collection_exists(self, book_id: str) -> bool:
        try:
            return self.client.collection_exists(self.collection_name(book_id))
        except Exception:  # noqa: BLE001 - An unavailable collection acts as empty.
            return False

    def ping(self) -> bool:
        try:
            return bool(self.client.get_collections())
        except Exception:  # noqa: BLE001 - Health checks return False on failure.
            return False
