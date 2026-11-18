"""Swappable vector store backends: Chroma, FAISS, or Pinecone with per-report isolation."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain.schema import BaseRetriever
from pydantic import Field

from logger import logger
from modules.embeddings import get_embedding_dimension, get_embeddings
from modules.settings import (
    CHROMA_COLLECTION,
    LOCAL_VECTORSTORE_DIR,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_INDEX_NAME,
    PINECONE_REGION,
    RETRIEVAL_K,
    VECTOR_STORE,
)


class SimpleRetriever(BaseRetriever):
    """Return a fixed list of documents (used after an explicit similarity search)."""

    docs: list[Document] = Field(default_factory=list)

    def _get_relevant_documents(self, query: str) -> list[Document]:
        return self.docs


def _sanitize_metadata(metadata: dict) -> dict:
    """Pinecone and vector stores only accept str/number/bool metadata values."""
    clean = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def add_texts(
    texts: list[str],
    metadatas: list[dict],
    ids: list[str],
    namespace: Optional[str] = None,
) -> None:
    """Embed and persist texts in the configured vector store with optional per-report namespace."""
    if not texts:
        return
    backend = VECTOR_STORE
    logger.info(
        "Indexing %d chunks into %s (namespace/report_id=%s)",
        len(texts),
        backend,
        namespace or "default",
    )
    if backend == "pinecone":
        _pinecone_add(texts, metadatas, ids, namespace=namespace)
    elif backend == "faiss":
        _faiss_add(texts, metadatas, ids, namespace=namespace)
    elif backend == "chroma":
        _chroma_add(texts, metadatas, ids, namespace=namespace)
    else:
        raise ValueError(
            f"Unknown VECTOR_STORE={backend!r}. Use chroma, faiss, or pinecone."
        )


def similarity_search(
    query: str,
    k: Optional[int] = None,
    namespace: Optional[str] = None,
) -> list[Document]:
    """Retrieve the top-k chunks for a query from the configured store and namespace."""
    k = k if k is not None else RETRIEVAL_K
    backend = VECTOR_STORE
    if backend == "pinecone":
        return _pinecone_search(query, k, namespace=namespace)
    if backend == "faiss":
        return _faiss_search(query, k, namespace=namespace)
    if backend == "chroma":
        return _chroma_search(query, k, namespace=namespace)
    raise ValueError(f"Unknown VECTOR_STORE={backend!r}.")


def delete_report_vectors(namespace: str) -> bool:
    """Physically remove all vectors associated with a report_id across backends."""
    backend = VECTOR_STORE
    logger.info("Deleting vectors for report namespace %s from %s", namespace, backend)
    if backend == "pinecone":
        return _pinecone_delete(namespace)
    if backend == "faiss":
        return _faiss_delete(namespace)
    if backend == "chroma":
        return _chroma_delete(namespace)
    return False


def as_retriever(docs: list[Document]) -> SimpleRetriever:
    return SimpleRetriever(docs=docs)


# --- Chroma (local) ----------------------------------------------------------

def _chroma_dir() -> Path:
    path = LOCAL_VECTORSTORE_DIR / "chroma"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _chroma_collection_name(namespace: Optional[str] = None) -> str:
    if namespace:
        clean = "".join(c for c in namespace if c.isalnum() or c in ("_", "-"))
        return f"report_{clean}"
    return CHROMA_COLLECTION


def _chroma_store(namespace: Optional[str] = None):
    from langchain_community.vectorstores import Chroma

    col_name = _chroma_collection_name(namespace)
    return Chroma(
        collection_name=col_name,
        embedding_function=get_embeddings(),
        persist_directory=str(_chroma_dir()),
    )


def _chroma_add(
    texts: list[str],
    metadatas: list[dict],
    ids: list[str],
    namespace: Optional[str] = None,
) -> None:
    store = _chroma_store(namespace=namespace)
    store.add_texts(texts=texts, metadatas=metadatas, ids=ids)


def _chroma_search(
    query: str,
    k: int,
    namespace: Optional[str] = None,
) -> list[Document]:
    store = _chroma_store(namespace=namespace)
    return store.similarity_search(query, k=k)


def _chroma_delete(namespace: str) -> bool:
    import chromadb

    col_name = _chroma_collection_name(namespace)
    try:
        client = chromadb.PersistentClient(path=str(_chroma_dir()))
        client.delete_collection(col_name)
        logger.info("Deleted Chroma collection %s", col_name)
        return True
    except Exception as exc:
        logger.warning(
            "Chroma collection delete failed or did not exist for %s: %s",
            col_name,
            exc,
        )
        return False


# --- FAISS (local) -----------------------------------------------------------

def _faiss_dir(namespace: Optional[str] = None) -> Path:
    if namespace:
        path = LOCAL_VECTORSTORE_DIR / "faiss" / namespace
    else:
        path = LOCAL_VECTORSTORE_DIR / "faiss"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _faiss_index_exists(namespace: Optional[str] = None) -> bool:
    return (_faiss_dir(namespace) / "index.faiss").exists()


def _faiss_add(
    texts: list[str],
    metadatas: list[dict],
    ids: list[str],
    namespace: Optional[str] = None,
) -> None:
    from langchain_community.vectorstores import FAISS

    embeddings = get_embeddings()
    path = str(_faiss_dir(namespace))
    if _faiss_index_exists(namespace):
        store = FAISS.load_local(
            path, embeddings, allow_dangerous_deserialization=True
        )
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    else:
        store = FAISS.from_texts(
            texts, embeddings, metadatas=metadatas, ids=ids
        )
    store.save_local(path)


def _faiss_search(
    query: str,
    k: int,
    namespace: Optional[str] = None,
) -> list[Document]:
    from langchain_community.vectorstores import FAISS

    if not _faiss_index_exists(namespace):
        logger.warning("FAISS index for %s is empty; upload a PDF first.", namespace or "default")
        return []
    store = FAISS.load_local(
        str(_faiss_dir(namespace)),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
    return store.similarity_search(query, k=k)


def _faiss_delete(namespace: str) -> bool:
    path = _faiss_dir(namespace)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
        logger.info("Deleted FAISS directory %s", path)
        return True
    return False


# --- Pinecone ----------------------------------------------------------------

def _require_pinecone_key() -> str:
    if not PINECONE_API_KEY:
        raise RuntimeError(
            "VECTOR_STORE=pinecone but PINECONE_API_KEY is not set. "
            "Switch VECTOR_STORE to chroma/faiss or add a Pinecone key."
        )
    return PINECONE_API_KEY


def _pinecone_index():
    from pinecone import Pinecone, ServerlessSpec

    api_key = _require_pinecone_key()
    pc = Pinecone(api_key=api_key)
    existing = [idx.name if hasattr(idx, "name") else idx["name"] for idx in pc.list_indexes()]
    dim = get_embedding_dimension()

    if PINECONE_INDEX_NAME not in existing:
        logger.info(
            "Creating Pinecone index %s (dimension=%s)",
            PINECONE_INDEX_NAME,
            dim,
        )
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)
    else:
        description = pc.describe_index(PINECONE_INDEX_NAME)
        existing_dim = getattr(description, "dimension", None)
        if existing_dim is None:
            existing_dim = description.get("dimension") if isinstance(description, dict) else None
        if existing_dim and int(existing_dim) != int(dim):
            raise RuntimeError(
                f"Pinecone index {PINECONE_INDEX_NAME!r} has dimension {existing_dim}, "
                f"but the embedding model produces {dim}. Create a new index or set "
                f"PINECONE_INDEX_NAME / EMBEDDING_DIMENSION to match."
            )
    return pc.Index(PINECONE_INDEX_NAME)


def _pinecone_add(
    texts: list[str],
    metadatas: list[dict],
    ids: list[str],
    namespace: Optional[str] = None,
) -> None:
    index = _pinecone_index()
    embeddings = get_embeddings().embed_documents(texts)
    vectors = []
    for item_id, values, text, metadata in zip(ids, embeddings, texts, metadatas):
        meta = _sanitize_metadata({**metadata, "text": text})
        vectors.append({"id": item_id, "values": values, "metadata": meta})
    batch_size = 50
    for start in range(0, len(vectors), batch_size):
        kwargs = {"vectors": vectors[start : start + batch_size]}
        if namespace:
            kwargs["namespace"] = namespace
        index.upsert(**kwargs)


def _pinecone_search(
    query: str,
    k: int,
    namespace: Optional[str] = None,
) -> list[Document]:
    index = _pinecone_index()
    vector = get_embeddings().embed_query(query)
    kwargs = {"vector": vector, "top_k": k, "include_metadata": True}
    if namespace:
        kwargs["namespace"] = namespace
    result = index.query(**kwargs)
    docs: list[Document] = []
    for match in result.get("matches", []):
        metadata = dict(match.get("metadata") or {})
        text = metadata.pop("text", "") or metadata.get("page_content", "")
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


def _pinecone_delete(namespace: str) -> bool:
    try:
        index = _pinecone_index()
        index.delete(delete_all=True, namespace=namespace)
        logger.info("Deleted Pinecone namespace %s", namespace)
        return True
    except Exception as exc:
        logger.warning("Pinecone namespace delete failed for %s: %s", namespace, exc)
        return False
