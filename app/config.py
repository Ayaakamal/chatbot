"""
Central configuration
======================
Every path, model name, port, and tunable lives here — one place to change them.
Import `settings` anywhere; never hard-code these values in feature modules.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project root = the folder that contains this package's parent (…/Chatbot)
ROOT = Path(__file__).resolve().parent.parent


class Settings:
    # ── Paths ─────────────────────────────────────────────
    ROOT_DIR       = ROOT
    DOCUMENTS_DIR  = str(ROOT / "documents")
    CHROMA_DB_DIR  = str(ROOT / "chroma_db")
    DOCSTORE_PATH  = str(ROOT / "docstore.json")
    ANALYTICS_DB   = str(ROOT / "analytics.db")

    # ── Server ────────────────────────────────────────────
    HOST = "0.0.0.0"
    PORT = 8002

    # ── LLM (Track 4 will add vLLM/Qwen alongside these) ──
    GOOGLE_API_KEY    = os.getenv("GOOGLE_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    LLM_PROVIDER      = os.getenv("LLM_PROVIDER", "gemini")   # gemini | anthropic | vllm
    GEMINI_MODEL      = "gemini-2.5-flash"
    ANTHROPIC_MODEL   = "claude-sonnet-4-6"
    LLM_TEMPERATURE   = 0.5

    # ── Embeddings (swap to BGE-M3 for Track 1) ───────────
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    RERANKER_MODEL  = "cross-encoder/ms-marco-TinyBERT-L-6"

    # ── Chunking ──────────────────────────────────────────
    PARENT_CHUNK_SIZE    = 2000
    PARENT_CHUNK_OVERLAP = 200
    CHILD_CHUNK_SIZE     = 400
    CHILD_CHUNK_OVERLAP  = 50

    # ── Retrieval ─────────────────────────────────────────
    RETRIEVAL_K       = 8
    RETRIEVAL_FETCH_K = 40
    MMR_LAMBDA        = 0.4
    RERANK_TOP_N      = 4

    SUPPORTED_EXTENSIONS = {
        ".pdf", ".txt", ".md", ".docx", ".csv", ".xlsx", ".xls", ".parquet", ".json"
    }

    # ── Multi-tenancy / permissions (Track 3) ─────────────
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # ── SQL Agent (Track 2) ───────────────────────────────
    ERP_DATABASE_URL = os.getenv("ERP_DATABASE_URL", "")   # real ERP DB, filled by DB team


settings = Settings()
