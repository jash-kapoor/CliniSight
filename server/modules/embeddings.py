"""Gemini embedding model factory."""

from __future__ import annotations

from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from logger import logger
from modules.settings import (
    EMBEDDING_DIMENSION,
    GEMINI_EMBED_MODEL,
    require_google_api_key,
)


def _normalize_embed_model(name: str) -> str:
    """LangChain's Generative Language client expects `models/<id>`."""
    name = name.strip()
    if name.startswith("models/"):
        return name
    return f"models/{name}"


@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Build the shared Gemini embeddings client."""
    api_key = require_google_api_key()
    kwargs: dict = {
        "model": _normalize_embed_model(GEMINI_EMBED_MODEL),
        "google_api_key": api_key,
    }
    if EMBEDDING_DIMENSION:
        kwargs["output_dimensionality"] = EMBEDDING_DIMENSION
    logger.info("Using Gemini embeddings model %s", GEMINI_EMBED_MODEL)
    return GoogleGenerativeAIEmbeddings(**kwargs)


def get_embedding_dimension() -> int:
    """Return configured dimension, or probe the model once."""
    if EMBEDDING_DIMENSION:
        return EMBEDDING_DIMENSION
    vector = get_embeddings().embed_query("dimension probe")
    dim = len(vector)
    logger.info("Probed embedding dimension: %s", dim)
    return dim
