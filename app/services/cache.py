import hashlib
import json
import re


class RedisCache:
    def __init__(self, redis_url: str, ttl_seconds: int, corpus_version: str, client=None):
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self.corpus_version = corpus_version
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import redis

            self._client = redis.Redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=1,
            )
        return self._client

    @staticmethod
    def normalize_question(question: str) -> str:
        return re.sub(r"\s+", " ", question.strip().lower())

    def key_for(self, book_id: str, question: str) -> str:
        normalized = self.normalize_question(question)
        raw_key = f"{book_id}|{normalized}|{self.corpus_version}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        return f"bookmind:{book_id}:{key_hash}"

    def get(self, book_id: str, question: str) -> dict | None:
        try:
            value = self.client.get(self.key_for(book_id, question))
            return json.loads(value) if value else None
        except Exception:  # noqa: BLE001 - Cache failures must not break chat.
            return None

    def set_approved(self, book_id: str, question: str, payload: dict) -> bool:
        if payload.get("review_verdict") != "approved":
            return False

        try:
            self.client.setex(
                self.key_for(book_id, question),
                self.ttl_seconds,
                json.dumps(payload, ensure_ascii=False),
            )
            return True
        except Exception:  # noqa: BLE001 - Cache failures must not break chat.
            return False

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception:  # noqa: BLE001 - Health checks return False on failure.
            return False
