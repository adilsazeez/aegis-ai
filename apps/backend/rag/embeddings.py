from sentence_transformers import SentenceTransformer

from config import get_settings

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model. Loaded once, reused across requests."""
    global _model
    if _model is None:
        settings = get_settings()
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def embed_text(text: str) -> list[float]:
    """Embed a single text string into a 384-dimensional vector."""
    model = _get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts into 384-dimensional vectors. Uses batching internally."""
    model = _get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=64)
    return embeddings.tolist()
