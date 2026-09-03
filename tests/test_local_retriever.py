from app.models.response import Source
from app.services.local_retriever import FallbackRetriever, LocalPdfRetriever


def test_local_retriever_finds_evidence_for_a_book_question():
    sources = LocalPdfRetriever().search(
        "meditations", "What does the book say about character?", limit=3,
    )
    assert sources
    assert all(source.page for source in sources)
    assert all(source.text for source in sources)


def test_local_retriever_does_not_invent_evidence_for_a_greeting():
    assert LocalPdfRetriever().search("meditations", "hello", limit=3) == []


def test_fallback_retriever_handles_primary_failure():
    class BrokenRetriever:
        def search(self, *args, **kwargs):
            raise ConnectionError("offline")

    class WorkingRetriever:
        def search(self, *args, **kwargs):
            return [Source(chunk_id="local-1", text="Local evidence", score=0.8)]

    retriever = FallbackRetriever(BrokenRetriever(), WorkingRetriever())
    sources = retriever.search("meditations", "What can we control?")

    assert sources[0].chunk_id == "local-1"
