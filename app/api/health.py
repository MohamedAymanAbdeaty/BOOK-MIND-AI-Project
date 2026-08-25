from flask import Blueprint, current_app, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.get("/health")
def health():
    settings = current_app.config["BOOKMIND_SETTINGS"]
    return jsonify({
        "status": "ok",
        "service": "bookmind",
        "corpus_version": settings.corpus_version,
    })
