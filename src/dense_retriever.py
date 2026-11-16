"""Dense retriever over the chunk corpus with precomputed embeddings.

Loads the embeddings JSONL produced by ``create_embeddings.py`` and answers
queries by encoding them with the same embedding model, then ranking chunks
by cosine similarity (embeddings are L2-normalized, so the dot product equals
cosine similarity).

The retriever interface itself does not hard-code the embedding model: any
object implementing ``encode(texts, **kwargs) -> numpy.ndarray`` can be
injected (unit tests use a deterministic fake embedder).

Nothing here fabricates metadata: Article/page/source values are copied from
the corpus records exactly as they were written by the ingestion pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.retrieval_base import (
    RetrieverUnavailableError,
    _normalize,
    _to_chunk,
    load_corpus,
)
from src.schemas import RetrievedPassage


def _load_embedder(model_name: str) -> Any:
    """Create the real embedder, or raise ``RetrieverUnavailableError``.

    Kept in a function so the heavy ``sentence_transformers`` import only
    happens when a real retriever is actually built.

    ``RAG_DEVICE`` selects the torch device (e.g. ``cpu``). Default is
    auto-selection; on Apple Silicon auto may pick MPS, which large models
    such as bge-m3 can exhaust, so ``RAG_DEVICE=cpu`` is the safe fallback.
    """
    import os

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RetrieverUnavailableError(
            reason="embedding_model_unavailable",
            detail=(
                "The embedding model backend (sentence-transformers) is not "
                "installed on the server, so retrieval cannot run."
            ),
        ) from exc

    device = os.getenv("RAG_DEVICE", "").strip() or None
    return SentenceTransformer(model_name, device=device)


def build_dense_retriever(
    embeddings_path: Path | str,
    model_name: str,
    top_k: int,
    embedder: Any | None = None,
) -> "DenseRetriever":
    """Build an operational retriever, raising ``RetrieverUnavailableError``
    if the corpus or embedding model cannot be loaded."""
    chunks = load_corpus(embeddings_path)
    vectors = np.array([record["embedding"] for record in chunks], dtype=np.float32)
    vectors = _normalize(vectors)

    if embedder is None:
        embedder = _load_embedder(model_name)

    return DenseRetriever(chunks=chunks, vectors=vectors, embedder=embedder, top_k=top_k)


class DenseRetriever:
    def __init__(
        self,
        chunks: list[dict[str, Any]],
        vectors: np.ndarray,
        embedder: Any,
        top_k: int,
    ) -> None:
        self._chunks = chunks
        self._vectors = vectors
        self._embedder = embedder
        self._top_k = max(1, int(top_k))

    def status(self) -> dict[str, Any]:
        return {
            "kind": "dense",
            "ready": True,
            "chunks": len(self._chunks),
            "dimension": int(self._vectors.shape[1]) if len(self._vectors) else 0,
            "top_k": self._top_k,
        }

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedPassage]:
        limit = self._top_k if top_k is None else max(1, int(top_k))
        query_vector = self._embedder.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        query_vector = _normalize(query_vector.reshape(1, -1))[0]

        scores = self._vectors @ query_vector
        top_indices = np.argsort(scores)[::-1][:limit]

        passages: list[RetrievedPassage] = []
        for rank, index in enumerate(top_indices, start=1):
            passages.append(
                RetrievedPassage(
                    chunk=_to_chunk(self._chunks[index]),
                    score=float(scores[index]),
                    rank=rank,
                    retrieval_method="dense",
                )
            )
        return passages
