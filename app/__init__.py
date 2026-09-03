import os
import threading

from flask import Flask, render_template

from app.api.chat import chat_bp
from app.api.health import health_bp
from app.config import Settings


def create_app(settings: Settings | None = None) -> Flask:
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app_settings = settings or Settings()
    app.config["BOOKMIND_SETTINGS"] = app_settings

    app.register_blueprint(chat_bp, url_prefix="/api")
    app.register_blueprint(health_bp, url_prefix="/api")

    @app.get("/")
    def index():
        return render_template("index.html", site_origin=app_settings.site_origin.rstrip("/"))

    def prewarm_embedding():
        try:
            from app.services.embedding import EmbeddingService

            EmbeddingService(app_settings.embedding_model).embed_query("warmup")
            app.logger.info("Embedding model pre-warmed and ready.")
        except Exception as exc:  # noqa: BLE001 - Preloading must not stop the app.
            app.logger.warning("Embedding pre-warm failed: %s", exc)

    if not os.getenv("VERCEL") and not app_settings.demo_mode:
        threading.Thread(
            target=prewarm_embedding,
            daemon=True,
            name="embed-prewarm",
        ).start()

    return app
