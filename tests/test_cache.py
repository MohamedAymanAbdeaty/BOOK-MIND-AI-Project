import json

from app.services.cache import RedisCache


class FakeRedis:
    def __init__(self):
        self.values = {}

    def get(self, key):
        return self.values.get(key)

    def setex(self, key, ttl, value):
        self.values[key] = value
        self.ttl = ttl

    def ping(self):
        return True


def test_cache_key_normalizes_whitespace_and_case():
    cache = RedisCache("redis://unused", 60, "v1", client=FakeRedis())
    assert cache.key_for("book", "  What  IS this? ") == cache.key_for("book", "what is this?")


def test_cache_key_changes_with_corpus_version():
    one = RedisCache("redis://unused", 60, "v1", client=FakeRedis())
    two = RedisCache("redis://unused", 60, "v2", client=FakeRedis())
    assert one.key_for("book", "question") != two.key_for("book", "question")


def test_cache_only_writes_approved_answers():
    client = FakeRedis()
    cache = RedisCache("redis://unused", 60, "v1", client=client)
    assert not cache.set_approved("book", "q", {"review_verdict": "refused"})
    assert client.values == {}
    assert cache.set_approved("book", "q", {"review_verdict": "approved", "answer": "a"})
    assert json.loads(next(iter(client.values.values())))["answer"] == "a"


def test_cache_round_trip():
    cache = RedisCache("redis://unused", 60, "v1", client=FakeRedis())
    payload = {"review_verdict": "approved", "answer": "supported"}
    cache.set_approved("book", "q", payload)
    assert cache.get("book", "q") == payload
