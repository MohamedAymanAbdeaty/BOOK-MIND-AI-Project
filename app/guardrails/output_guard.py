from pydantic import ValidationError

from app.models.response import ChatResponse


def validate_output(payload: dict) -> ChatResponse:
    try:
        return ChatResponse.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("The generated response failed output validation") from exc
