"""Dense retriever over the chunk corpus with precomputed embeddings.

Loads the embeddings JSONL produced by ``create_embeddings.py`` and answers
queries by encoding them with the same embedding model, then ranking chunks
by cosine similarity (embeddings are L2-normalized, so the dot product equals
cosine similarity).

The retriever interface itself does not hard-code the embedding model: any
object implementing ``encode(texts, **kwargs) -> numpy.ndarray`` can be
injected (unit tests use a deterministic fake embedder; later, Satish and
Pooran's improved retrievers can implement the same interface).

Nothing here fabricates metadata: Article/page/source values are copied from
the corpus records exactly as they were written by the ingestion pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.schemas import DocumentChunk, RetrievedPassage

DEFAULT_EMBEDDINGS_PATH = Path("Data/processed/constitution_embeddings.jsonl")

# Field names in the corpus records written by Prem's pipeline
# (chunk_text.py / create_embeddings.py).
_TEXT_KEYS = ("text",)
_PAGE_KEYS = ("page_start", "page")
_ARTICLE_KEYS = ("article", "article_number")


class RetrieverUnavailableError(RuntimeError):
    """Raised when the retriever cannot serve queries.

    ``reason`` is a stable machine-readable tag the API maps to HTTP 503 and
    ``detail`` is a human-readable explanation safe to show users.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


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


def _first(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def load_corpus(embeddings_path: Path | str) -> list[dict[str, Any]]:
    """Load chunk records with embeddings from a JSONL file."""
    path = Path(embeddings_path)
    if not path.exists():
        raise RetrieverUnavailableError(
            reason="index_not_found",
            detail=(
                "The indexed Constitution corpus was not found on the server"
                f" ({path}). The offline preprocessing/embedding step has not"
                " been run or the artifacts are missing."
            ),
        )

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RetrieverUnavailableError(
                    reason="corrupt_index",
                    detail=(
                        "The retriever index file is malformed"
                        f" ({path}, line {line_number})."
                    ),
                ) from exc
            if not isinstance(record, dict) or not _first(record, _TEXT_KEYS):
                continue
            records.append(record)

    if not records:
        raise RetrieverUnavailableError(
            reason="empty_index",
            detail="The retriever index file contains no usable chunks.",
        )
    return records


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


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


def _to_chunk(record: dict[str, Any]) -> DocumentChunk:
    text = _first(record, _TEXT_KEYS)
    page = _first(record, _PAGE_KEYS)
    article = _first(record, _ARTICLE_KEYS)
    if page is not None:
        try:
            page = int(page)
        except (TypeError, ValueError):
            page = None
    return DocumentChunk(
        chunk_id=str(record["chunk_id"]),
        text=str(text),
        article=str(article) if article is not None else None,
        part=record.get("part"),
        clause=record.get("clause"),
        page=page,
        source_title=record.get("source_title", "Constitution of India"),
        source_url=record.get("source_url", ""),
        source_version=record.get("source_version", "unknown"),
        metadata=record.get("metadata", {}),
    )


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
