import math
import re
from collections import Counter
from pathlib import Path

from app.models.response import Source
from scripts.ingestion.chunker import chunk_pages
from scripts.ingestion.loader import load_pdf

BOOK_FILES = {
    "rich_dad_poor_dad": "rich_dad_poor_dad.pdf",
    "the_art_of_war": "The_Art_Of_War.pdf",
    "meditations": "meditationsofmar00marc.pdf",
}

STOP_WORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "book", "by", "can",
    "does", "for", "from", "how", "in", "is", "it", "of", "on", "say", "says",
    "that", "the", "this", "to", "was", "what", "when", "where", "which", "who",
    "why", "with",
}


def _term(word: str) -> str:
    word = word.lower()
    if word.startswith("prepar"):
        return "prepar"
    if word.startswith(("control", "govern", "command", "power")):
        return "control"
    for suffix in ("ingly", "edly", "ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def _terms(text: str) -> list[str]:
    return [
        _term(word)
        for word in re.findall(r"[a-zA-Z]{2,}", text)
        if word.lower() not in STOP_WORDS
    ]


class LocalPdfRetriever:
    """Read-only lexical retrieval used when an external vector DB is unavailable."""

    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir or Path(__file__).resolve().parents[2] / "data")
        self._chunks_by_book = {}

    def _chunks(self, book_id: str) -> tuple:
        if book_id in self._chunks_by_book:
            return self._chunks_by_book[book_id]

        filename = BOOK_FILES.get(book_id)
        path = self.data_dir / filename if filename else None
        chunks = tuple(chunk_pages(load_pdf(path))) if path and path.is_file() else ()
        self._chunks_by_book[book_id] = chunks
        return chunks

    def search(
        self,
        book_id: str,
        question: str,
        limit: int = 5,
        score_threshold: float = 0.36,
    ) -> list[Source]:
        query_terms = set(_terms(question))
        if not query_terms:
            return []

        page_match = re.search(r"\b(?:page|p\.?)\s*(\d+)\b", question, re.IGNORECASE)
        requested_page = int(page_match.group(1)) if page_match else None
        ranked = []
        for chunk in self._chunks(book_id):
            if requested_page is not None and chunk.page != requested_page:
                continue
            score = self._score_chunk(chunk.text, query_terms, requested_page)
            if requested_page is not None or score >= score_threshold:
                ranked.append((score, chunk))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            Source(
                chunk_id=f"{book_id}-p{chunk.page}-c{chunk.chunk_index}",
                page=chunk.page,
                chapter=None,
                text=chunk.text,
                score=score,
            )
            for score, chunk in ranked[:limit]
        ]

    @staticmethod
    def _score_chunk(text: str, query_terms: set[str], requested_page: int | None) -> float:
        if requested_page is not None:
            return 1.0

        counts = Counter(_terms(text))
        matches = [counts[term] for term in query_terms if counts[term]]
        if not matches:
            return 0.0

        coverage = len(matches) / len(query_terms)
        frequency_bonus = min(0.15, sum(math.log1p(count) for count in matches) / 30)
        phrase_bonus = 0.0
        if "control" in query_terms and re.search(
            r"\bin (?:our|your|his) power\b",
            text,
            re.IGNORECASE,
        ):
            phrase_bonus = 0.2

        return min(1.0, coverage * 0.75 + frequency_bonus + phrase_bonus)


class FallbackRetriever:
    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback

    def search(
        self,
        book_id: str,
        question: str,
        limit: int = 5,
        score_threshold: float = 0.36,
    ) -> list[Source]:
        try:
            sources = self.primary.search(
                book_id,
                question,
                limit=limit,
                score_threshold=score_threshold,
            )
        except Exception:  # noqa: BLE001 - Falling back is the purpose of this class.
            sources = []

        return sources or self.fallback.search(
            book_id,
            question,
            limit=limit,
            score_threshold=score_threshold,
        )
