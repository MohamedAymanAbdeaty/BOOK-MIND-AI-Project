from dataclasses import dataclass

from scripts.ingestion.cleaner import clean_text
from scripts.ingestion.loader import PageText


@dataclass(frozen=True)
class Chunk:
    page: int
    text: str
    chunk_index: int


def chunk_pages(pages: list[PageText], chunk_words: int = 260, overlap_words: int = 45) -> list[Chunk]:
    if chunk_words < 40 or overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("Use chunk_words >= 40 and 0 <= overlap_words < chunk_words")
    chunks: list[Chunk] = []
    chunk_index = 0
    step = chunk_words - overlap_words
    for page in pages:
        words = clean_text(page.text).split()
        for start in range(0, len(words), step):
            part = words[start : start + chunk_words]
            if len(part) < 20:
                continue
            chunks.append(Chunk(page=page.page, text=" ".join(part), chunk_index=chunk_index))
            chunk_index += 1
            if start + chunk_words >= len(words):
                break
    return chunks
