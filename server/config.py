"""Shared configuration and paths for the server application."""

from __future__ import annotations

from modules.settings import (
    CHROMA_COLLECTION,
    EMBEDDING_DIMENSION,
    GEMINI_CHAT_MODEL,
    GEMINI_EMBED_MODEL,
    GOOGLE_API_KEY,
    LOCAL_VECTORSTORE_DIR,
    MAX_UPLOAD_SIZE_MB,
    PARSED_REPORTS_DIR,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_INDEX_NAME,
    PINECONE_REGION,
    REPORT_PAGES_DIR,
    RETRIEVAL_K,
    SERVER_DIR,
    UPLOAD_DIR,
    VECTOR_STORE,
    get_report_pages_dir,
    require_google_api_key,
)
import os

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")

__all__ = [
    "API_URL",
    "CHROMA_COLLECTION",
    "EMBEDDING_DIMENSION",
    "GEMINI_CHAT_MODEL",
    "GEMINI_EMBED_MODEL",
    "GOOGLE_API_KEY",
    "LOCAL_VECTORSTORE_DIR",
    "MAX_UPLOAD_SIZE_MB",
    "PARSED_REPORTS_DIR",
    "PINECONE_API_KEY",
    "PINECONE_CLOUD",
    "PINECONE_INDEX_NAME",
    "PINECONE_REGION",
    "REPORT_PAGES_DIR",
    "RETRIEVAL_K",
    "SERVER_DIR",
    "UPLOAD_DIR",
    "VECTOR_STORE",
    "get_report_pages_dir",
    "require_google_api_key",
]
