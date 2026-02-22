"""
Embedding model for RAG: one model for ingestion and query.
Uses sentence-transformers/all-MiniLM-L6-v2 (384d) to match Actian RAG example (COSINE).
"""
from typing import List, Union
import rag_config

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(rag_config.EMBEDDING_MODEL)
    return _model


def encode(texts: Union[str, List[str]], batch_size: int = 32) -> List[List[float]]:
    """
    Encode one or more texts into 384-d vectors (normalized for COSINE).
    Returns list of vectors; if input was a single string, returns list of one vector.
    """
    if isinstance(texts, str):
        texts = [texts]
    model = _get_model()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    return [emb.tolist() for emb in embeddings]


def get_dimension() -> int:
    return rag_config.EMBEDDING_DIMENSION
