from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ReviewVerdict(StrEnum):
    APPROVED = "approved"
    REFUSED = "refused"
    BLOCKED = "blocked"


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    page: int | None = None
    chapter: str | None = None
    text: str = Field(max_length=8000)
    score: float = Field(ge=0, le=1)


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    sources: list[Source] = Field(default_factory=list)
    review_verdict: ReviewVerdict
    review_feedback: str = ""
    cached: bool = False
    pipeline: list[str] = Field(default_factory=list)
