import threading

from flask import Flask, render_template

from app.config import Settings
from app.api.chat import chat_bp
from app.api.health import health_bp


def create_app(settings: Settings | None = None) -> Flask:
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    s = settings or Settings()
    app.config["BOOKMIND_SETTINGS"] = s

    app.register_blueprint(chat_bp, url_prefix="/api")
    app.register_blueprint(health_bp, url_prefix="/api")

    @app.get("/")
    def index():
        return render_template("index.html", site_origin=app.config["BOOKMIND_SETTINGS"].site_origin.rstrip("/"))

    # Load the embedding model in the background so the first request isn't slow
    def _prewarm():
        try:
            from app.services.embedding_service import EmbeddingService
            EmbeddingService(s.embedding_model).embed_query("warmup")
            app.logger.info("Embedding model pre-warmed and ready.")
        except Exception as exc:
            app.logger.warning("Embedding pre-warm failed: %s", exc)

    threading.Thread(target=_prewarm, daemon=True, name="embed-prewarm").start()

    return app
