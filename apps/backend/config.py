import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Central configuration loaded from .env file."""

    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_SERVICE_ROLE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    ASSEMBLYAI_API_KEY: str = os.getenv("ASSEMBLYAI_API_KEY", "")
    ACTIAN_DB_URL: str = os.getenv("ACTIAN_DB_URL", "localhost:50051")

    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    LEGAL_CODES_COLLECTION: str = "legal_codes"
    VECTOR_SEARCH_TOP_K: int = 5


@lru_cache()
def get_settings() -> Settings:
    return Settings()


def get_supabase_client():
    from supabase import create_client

    settings = get_settings()
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)


def get_groq_client():
    from groq import AsyncGroq

    settings = get_settings()
    return AsyncGroq(api_key=settings.GROQ_API_KEY)


_actian_client = None


async def get_actian_client():
    """Return a singleton Actian client. Created once, reused across requests."""
    global _actian_client
    if _actian_client is None:
        from cortex import AsyncCortexClient

        settings = get_settings()
        _actian_client = AsyncCortexClient(settings.ACTIAN_DB_URL)
        await _actian_client.__aenter__()
    return _actian_client
