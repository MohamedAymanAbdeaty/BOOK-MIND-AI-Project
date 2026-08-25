import pytest

from scripts.ingestion.chunker import chunk_pages
from scripts.ingestion.cleaner import clean_text
from scripts.ingestion.loader import PageText


def test_cleaner_repairs_hyphenated_line_breaks():
    assert clean_text("finan-\ncial   intelligence") == "financial intelligence"


def test_chunker_preserves_page_metadata_and_overlap():
    page = PageText(page=7, text=" ".join(f"word{i}" for i in range(130)))
    chunks = chunk_pages([page], chunk_words=60, overlap_words=10)
    assert len(chunks) == 3
    assert all(chunk.page == 7 for chunk in chunks)
    assert chunks[0].text.split()[-10:] == chunks[1].text.split()[:10]


@pytest.mark.parametrize("chunk_words,overlap", [(20, 0), (60, -1), (60, 60), (60, 80)])
def test_chunker_rejects_invalid_sizes(chunk_words, overlap):
    with pytest.raises(ValueError):
        chunk_pages([], chunk_words, overlap)
