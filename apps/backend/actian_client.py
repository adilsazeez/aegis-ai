"""
Actian VectorAI DB client wrapper.
Uses AsyncCortexClient for runtime (FastAPI); use cortex.CortexClient in scripts for sync ingestion.
"""
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

import rag_config

# Async client for runtime (main.py, RAG search)
try:
    from cortex import AsyncCortexClient, DistanceMetric
    _ACTIAN_AVAILABLE = True
except ImportError:
    _ACTIAN_AVAILABLE = False
    AsyncCortexClient = None
    DistanceMetric = None


def is_actian_available() -> bool:
    return _ACTIAN_AVAILABLE and AsyncCortexClient is not None


@asynccontextmanager
async def get_async_client():
    """Async context manager: connect to Actian at ACTIAN_HOST."""
    if not is_actian_available():
        raise RuntimeError(
            "Actian Cortex client not installed. "
            "Install the wheel from https://github.com/hackmamba-io/actian-vectorAI-db-beta"
        )
    async with AsyncCortexClient(rag_config.ACTIAN_HOST) as client:
        yield client


async def ensure_collection(client, collection: str, dimension: int, distance_metric=DistanceMetric.COSINE):
    """Create collection if it does not exist."""
    if DistanceMetric is None:
        raise RuntimeError("Actian client not available")
    exists = await client.has_collection(collection)
    if not exists:
        await client.create_collection(
            name=collection,
            dimension=dimension,
            distance_metric=distance_metric,
            hnsw_ef_search=100,
        )


async def search_prosocial(client, query_vector: List[float], top_k: int = None) -> List[Any]:
    """
    Run semantic search on the prosocial collection.
    Returns list of results (each has .id, .score, .payload).
    """
    top_k = top_k or rag_config.RAG_TOP_K
    return await client.search(
        rag_config.PROSOCIAL_COLLECTION,
        query_vector,
        top_k=top_k,
    )


# --- Sync client for ingestion script ---
def get_sync_client():
    """Sync CortexClient for one-off ingestion. Use: with get_sync_client() as (client, DistanceMetric): ..."""
    try:
        from cortex import CortexClient, DistanceMetric as DM
        return CortexClient(rag_config.ACTIAN_HOST), DM
    except ImportError:
        raise RuntimeError(
            "Install Actian Cortex client wheel from "
            "https://github.com/hackmamba-io/actian-vectorAI-db-beta"
        )
