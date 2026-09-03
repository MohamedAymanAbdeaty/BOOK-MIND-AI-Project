from flask import Blueprint, current_app, jsonify, request
from pydantic import ValidationError

from app.catalog import BOOKS
from app.models.request import ChatRequest
from app.rag.graph import build_workflow

chat_bp = Blueprint("chat", __name__)


def _workflow():
    if "bookmind_workflow" not in current_app.extensions:
        settings = current_app.config["BOOKMIND_SETTINGS"]
        current_app.extensions["bookmind_workflow"] = build_workflow(settings)
    return current_app.extensions["bookmind_workflow"]


@chat_bp.get("/books")
def list_books():
    return jsonify({"books": list(BOOKS.values())})


@chat_bp.post("/chat")
def chat():
    try:
        payload = ChatRequest.model_validate(request.get_json(silent=True) or {})
    except ValidationError as exc:
        return jsonify({"error": "invalid_request", "details": exc.errors(include_url=False)}), 400

    try:
        response = _workflow().invoke(payload.book_id, payload.question)
    except Exception:
        current_app.logger.exception("RAG request failed")
        return jsonify({
            "error": "service_unavailable",
            "message": "The answer service is temporarily unavailable.",
        }), 503

    return jsonify(response.model_dump(mode="json"))
