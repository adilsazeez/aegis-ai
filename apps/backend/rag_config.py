"""
RAG pipeline configuration.
Single place for collection names, dimensions, and model names.
"""
import os
from dotenv import load_dotenv
load_dotenv()

# Actian VectorAI DB
ACTIAN_HOST = os.getenv("ACTIAN_HOST", "localhost:50051")

# Embedding model: must match between ingestion and query. all-MiniLM-L6-v2 = 384 dims, COSINE.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Actian collection for Prosocial Dialog (semantic search only; Title 18 is not in Actian)
PROSOCIAL_COLLECTION = "prosocial"

# Search
RAG_TOP_K = 10

# Groq
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
