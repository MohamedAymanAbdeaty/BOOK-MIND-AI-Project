from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    book_id: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9_\-]+$")
    question: str = Field(min_length=2, max_length=1200)

    @field_validator("question")
    @classmethod
    def reject_control_characters(cls, value: str) -> str:
        if any(ord(char) < 32 and char not in "\n\t" for char in value):
            raise ValueError("Question contains unsupported control characters")
        return value
