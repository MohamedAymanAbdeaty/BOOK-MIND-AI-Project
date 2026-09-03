from flask import Blueprint, current_app, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.get("/health")
def health():
    settings = current_app.config["BOOKMIND_SETTINGS"]
    return jsonify({
        "status": "ok",
        "service": "bookmind",
        "corpus_version": settings.corpus_version,
        "retrieval_mode": "local_pdf" if settings.demo_mode else "qdrant_with_local_fallback",
        "answer_mode": "groq" if settings.groq_api_key else "extractive",
    })
