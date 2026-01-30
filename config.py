"""
Configuration file for Pizza Interview Search Tool
Supports Windows and Linux (Carina cluster) environments
"""
import os
from pathlib import Path

# Optional dotenv support - works without it using defaults
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use defaults

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT
EXCEL_FILE = DATA_DIR / "pizza_interviews copy.xlsx"

# Search configuration
SEARCH_INDEX_DIR = PROJECT_ROOT / "search_index"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
EMBEDDINGS_CACHE_DIR = PROJECT_ROOT / "embeddings_cache"

# Create directories if they don't exist
SEARCH_INDEX_DIR.mkdir(exist_ok=True)
VECTOR_DB_DIR.mkdir(exist_ok=True)
EMBEDDINGS_CACHE_DIR.mkdir(exist_ok=True)

# Semantic search model configuration
# Using a lightweight, locally hostable model
SEMANTIC_MODEL_NAME = os.getenv(
    "SEMANTIC_MODEL_NAME", 
    "all-MiniLM-L6-v2"  # Small, fast, good quality - works on Windows
)
# Alternative models (larger, better quality):
# "all-mpnet-base-v2" - Better quality, larger
# "paraphrase-multilingual-MiniLM-L12-v2" - Multilingual support

# Full-text search configuration
FULLTEXT_INDEX_NAME = "pizza_interviews_index"

# Carina cluster configuration
CARINA_ENABLED = os.getenv("CARINA_ENABLED", "false").lower() == "true"
CARINA_HOST = os.getenv("CARINA_HOST", "")
CARINA_USER = os.getenv("CARINA_USER", "")
CARINA_KEY_PATH = os.getenv("CARINA_KEY_PATH", "")
CARINA_REMOTE_DIR = os.getenv("CARINA_REMOTE_DIR", "~/pizza_search")
CARINA_MODEL_DIR = os.getenv("CARINA_MODEL_DIR", "~/models")

# Search settings
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "50"))
SEMANTIC_SEARCH_TOP_K = int(os.getenv("SEMANTIC_SEARCH_TOP_K", "10"))
FULLTEXT_SEARCH_TOP_K = int(os.getenv("FULLTEXT_SEARCH_TOP_K", "10"))

# Windows-specific settings
WINDOWS_USE_CPU = os.getenv("WINDOWS_USE_CPU", "true").lower() == "true"

# Query expansion settings (AI-based using OpenAI)
ENABLE_QUERY_EXPANSION = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"
QUERY_EXPANSION_MAX_TERMS = int(os.getenv("QUERY_EXPANSION_MAX_TERMS", "10"))
QUERY_EXPANSION_MODEL = os.getenv("QUERY_EXPANSION_MODEL", "gpt-3.5-turbo")  # Lightweight, fast, cheap

# Multi-query retrieval: when expansion is on, embed original + expansion phrases separately
# and fuse results with Reciprocal Rank Fusion (RRF). Better than averaging embeddings.
MULTI_QUERY_RRF = os.getenv("MULTI_QUERY_RRF", "true").lower() == "true"
MULTI_QUERY_RRF_K = int(os.getenv("MULTI_QUERY_RRF_K", "40"))
MULTI_QUERY_MAX_PHRASES = int(os.getenv("MULTI_QUERY_MAX_PHRASES", "3"))

# Chunking for semantic index: smaller windows improve recall for short queries (e.g. "family", "spicy")
# window_size=1 (single-sentence chunks) helps "family" match passages that say "family meal" or "eating with family"
CHUNK_WINDOW_SIZE = int(os.getenv("CHUNK_WINDOW_SIZE", "1"))  # 1 = one sentence per chunk; 2 = sentence pairs
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "0"))  # 0 when window_size=1; use 1 for overlapping pairs when window_size=2

# Embedding batch size during index build. Lower = less peak RAM, slower build (e.g. 8 or 16 on 512MB)
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))

# E5/BGE-style query/passage prefix (set true when using intfloat/e5-* or BAAI/bge-*)
USE_QUERY_PASSAGE_PREFIX = os.getenv("USE_QUERY_PASSAGE_PREFIX", "false").lower() == "true"
QUERY_PREFIX = os.getenv("QUERY_PREFIX", "query: ")
PASSAGE_PREFIX = os.getenv("PASSAGE_PREFIX", "passage: ")
