from functools import lru_cache


class EmbeddingService:
    """
    A service for turning text into vectors (lists of numbers).
    These mathematical representations let us compare how similar two pieces of text are.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    @lru_cache(maxsize=1)
    def _model(self):
        """
        Loads the Machine Learning model into memory.
        Using @lru_cache ensures we only ever load it once, as loading is slow.
        """
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(self.model_name)

    def embed_query(self, text: str) -> list[float]:
        """Converts a single question or small piece of text into a vector."""
        vector = self._model().encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Converts multiple pieces of text (like book pages) into vectors all at once."""
        vectors = self._model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [vector.tolist() for vector in vectors]

    @property
    def dimension(self) -> int:
        """Returns the size (number of dimensions) of the vectors this model creates."""
        return len(self.embed_query("dimension probe"))
