from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    site_origin: str = "http://localhost:5000"
    secret_key: str = "change-me"
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    groq_reviewer_model: str = "llama-3.1-8b-instant"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "book_chunks"
    redis_url: str = "redis://redis:6379/0"
    cache_ttl_seconds: int = 86400
    corpus_version: str = "v1"
    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.36
    max_question_chars: int = 1200
    max_revision_count: int = 1
    demo_mode: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
