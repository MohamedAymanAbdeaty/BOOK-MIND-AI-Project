import pytest

from app.config import Settings


@pytest.fixture
def settings():
    return Settings(_env_file=None, max_revision_count=1)


class NullCache:
    def __init__(self):
        self.saved = []

    def get(self, book_id, question):
        return None

    def set_approved(self, book_id, question, payload):
        self.saved.append((book_id, question, payload))
        return True


@pytest.fixture
def null_cache():
    return NullCache()
