from typing import TypedDict


class RAGState(TypedDict, total=False):
    question: str
    book_id: str
    retrieved_docs: list[dict]
    draft_answer: str
    citation_indexes: list[int]
    reviewer_feedback: str
    reviewer_verdict: str
    unsupported_claims: list[str]
    revision_count: int
    final_answer: str
    pipeline: list[str]
