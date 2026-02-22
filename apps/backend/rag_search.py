"""
RAG: embed user transcript and search Actian prosocial collection.
Returns list of chunk payloads for Groq.
"""
from typing import List, Any

from embedding import encode
from actian_client import get_async_client, search_prosocial
import rag_config


async def search_prosocial_chunks(transcript: str, top_k: int = None) -> List[dict]:
    """
    Embed transcript, search Actian 'prosocial' collection, return list of payload dicts.
    If Actian is unavailable or errors, returns [].
    """
    top_k = top_k or rag_config.RAG_TOP_K
    try:
        vectors = encode(transcript)
        if not vectors:
            return []
        query_vector = vectors[0]
    except Exception:
        return []

    try:
        async with get_async_client() as client:
            results = await search_prosocial(client, query_vector, top_k=top_k)
    except Exception:
        return []

    out = []
    for r in results:
        payload = getattr(r, "payload", None) or getattr(r, "metadata", None)
        if isinstance(payload, dict):
            out.append(payload)
        elif hasattr(r, "payload") and r.payload:
            out.append(dict(r.payload))
        else:
            out.append({})
    return out
