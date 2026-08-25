import hashlib
import json
import re


class RedisCache:
    """
    A service that talks to Redis to cache (remember) previous answers.
    If someone asks the exact same question again, we can just return the cached answer
    instead of running the whole expensive AI pipeline.
    """

    def __init__(self, redis_url: str, ttl_seconds: int, corpus_version: str, client=None):
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self.corpus_version = corpus_version
        self._client = client

    @property
    def client(self):
        """Lazy-loads the Redis connection."""
        if self._client is None:
            import redis
            # We set a short timeout because caching should be fast. If Redis is slow or down, 
            # we just skip the cache and process normally.
            self._client = redis.Redis.from_url(self.redis_url, decode_responses=True, socket_timeout=1)
        return self._client

    @staticmethod
    def normalize_question(question: str) -> str:
        """
        Cleans up the question so that minor differences like extra spaces 
        or capitalization don't prevent a cache hit.
        """
        return re.sub(r"\s+", " ", question.strip().lower())

    def key_for(self, book_id: str, question: str) -> str:
        """
        Creates a unique, secure key for this specific book and question.
        We include 'corpus_version' so that if the book data gets updated, old caches are ignored.
        """
        normalized_q = self.normalize_question(question)
        raw_string = f"{book_id}|{normalized_q}|{self.corpus_version}"
        
        # We hash the string so the key isn't extremely long if the question is long.
        hashed = hashlib.sha256(raw_string.encode("utf-8")).hexdigest()
        
        return f"bookmind:{book_id}:{hashed}"

    def get(self, book_id: str, question: str) -> dict | None:
        """Attempts to fetch a previously saved answer from the cache."""
        try:
            cache_key = self.key_for(book_id, question)
            value = self.client.get(cache_key)
            
            if value:
                return json.loads(value)
            return None
            
        except Exception:
            # If Redis fails (e.g. it's down), we just return None and act like it wasn't in the cache
            return None

    def set_approved(self, book_id: str, question: str, payload: dict) -> bool:
        """
        Saves a new answer into the cache for future use.
        We only save answers that were explicitly 'approved' by the reviewer agent.
        """
        if payload.get("review_verdict") != "approved":
            return False
            
        try:
            cache_key = self.key_for(book_id, question)
            
            # setex means "Set with Expiration" - it will automatically delete itself after ttl_seconds
            self.client.setex(
                cache_key,
                self.ttl_seconds,
                json.dumps(payload, ensure_ascii=False),
            )
            return True
        except Exception:
            return False

    def ping(self) -> bool:
        """Checks if the Redis server is alive and reachable."""
        try:
            return bool(self.client.ping())
        except Exception:
            return False
