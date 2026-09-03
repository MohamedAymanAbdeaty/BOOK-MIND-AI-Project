from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PageText:
    page: int
    text: str


def load_pdf(path: str | Path) -> list[PageText]:
    import pymupdf

    pdf_path = Path(path)
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"A readable PDF file is required: {pdf_path}")
    pages: list[PageText] = []
    with pymupdf.open(pdf_path) as document:
        for index, page in enumerate(document, start=1):
            pages.append(PageText(page=index, text=page.get_text("text")))
    return pages
