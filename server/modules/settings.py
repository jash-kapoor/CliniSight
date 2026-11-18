"""Application settings loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

SERVER_DIR = Path(__file__).resolve().parent.parent
load_dotenv(SERVER_DIR / ".env")

# Gemini (chat + embeddings). Names come from https://ai.google.dev/gemini-api/docs/models
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.1-flash-lite")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

# Vector store: "chroma" | "faiss" | "pinecone"
VECTOR_STORE = os.getenv("VECTOR_STORE", "chroma").strip().lower()
LOCAL_VECTORSTORE_DIR = Path(
    os.getenv("LOCAL_VECTORSTORE_DIR", str(SERVER_DIR / "local_vectorstore"))
)
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "medical_assistant")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))

# Optional override. Gemini embedding models default to 3072.
_raw_dim = os.getenv("EMBEDDING_DIMENSION", "").strip()
EMBEDDING_DIMENSION: int | None = int(_raw_dim) if _raw_dim else None

# Pinecone (only required when VECTOR_STORE=pinecone)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medicalindex")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(SERVER_DIR / "uploaded_docs")))
PARSED_REPORTS_DIR = Path(
    os.getenv("PARSED_REPORTS_DIR", str(SERVER_DIR / "parsed_reports"))
)
REPORT_PAGES_DIR = UPLOAD_DIR / "report_pages"
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))


def get_report_pages_dir(report_id: str) -> Path:
    """Ensure and return the directory for a report's page images."""
    target = REPORT_PAGES_DIR / report_id
    target.mkdir(parents=True, exist_ok=True)
    return target


def require_google_api_key() -> str:
    """Return GOOGLE_API_KEY or raise a clear error."""
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy server/.env.example to server/.env and add your key."
        )
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    return GOOGLE_API_KEY
