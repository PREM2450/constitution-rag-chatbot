"""Backward-compatible facade over the split retriever modules.

Implementation lives in ``retrieval_base`` / ``dense_retriever`` /
``bm25_retriever`` / ``hybrid_retriever``. This module re-exports the public
surface plus the ``build_retriever`` factory so existing imports keep working.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.bm25_retriever import BM25Retriever, build_bm25_retriever, tokenize
from src.dense_retriever import DenseRetriever, build_dense_retriever
from src.hybrid_retriever import HybridRetriever, build_hybrid_retriever
from src.retrieval_base import (
    DEFAULT_CHUNKS_PATH,
    DEFAULT_EMBEDDINGS_PATH,
    DEFAULT_RRF_K,
    RetrieverUnavailableError,
    _first,
    _normalize,
    _to_chunk,
    load_corpus,
)

__all__ = [
    "DEFAULT_CHUNKS_PATH",
    "DEFAULT_EMBEDDINGS_PATH",
    "DEFAULT_RRF_K",
    "RetrieverUnavailableError",
    "DenseRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "build_dense_retriever",
    "build_bm25_retriever",
    "build_hybrid_retriever",
    "build_retriever",
    "load_corpus",
    "tokenize",
]


def build_retriever(
    retrieval_mode: str,
    embeddings_path: Path | str,
    chunks_path: Path | str | None = None,
    model_name: str = "BAAI/bge-m3",
    top_k: int = 5,
    embedder: Any | None = None,
    fusion: str = "rrf",
    rrf_k: int = DEFAULT_RRF_K,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> Any:
    """Build the retriever for an explicit mode; never silently falls back.

    ``retrieval_mode`` must be one of ``dense | bm25 | hybrid`` (see
    ``AskRequest.retrieval_mode``). Unknown modes raise ``ValueError``;
    missing corpus/backend raises ``RetrieverUnavailableError``.
    """
    if retrieval_mode == "dense":
        return build_dense_retriever(
            embeddings_path, model_name=model_name, top_k=top_k, embedder=embedder
        )
    if retrieval_mode == "bm25":
        return build_bm25_retriever(
            chunks_path if chunks_path is not None else embeddings_path, top_k=top_k
        )
    if retrieval_mode == "hybrid":
        return build_hybrid_retriever(
            embeddings_path, chunks_path=chunks_path, model_name=model_name,
            top_k=top_k, embedder=embedder, fusion=fusion, rrf_k=rrf_k,
            dense_weight=dense_weight, bm25_weight=bm25_weight,
        )
    raise ValueError(
        f"retrieval_mode must be one of: dense, bm25, hybrid (got {retrieval_mode!r})"
    )
