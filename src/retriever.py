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
import re
from pathlib import Path
from typing import Any

import numpy as np

from src.schemas import DocumentChunk, RetrievedPassage

DEFAULT_EMBEDDINGS_PATH = Path("Data/processed/constitution_embeddings.jsonl")
DEFAULT_CHUNKS_PATH = Path("Data/processed/constitution_chunks.jsonl")

# Standard Reciprocal Rank Fusion constant (Cormack et al.). RRF-only in
# Task 3; score-normalized weighted fusion is deferred until evaluation
# (Tasks 6-7) provides dev-set tuning data.
DEFAULT_RRF_K = 60

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


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenize for BM25: lowercase alphanumeric tokens only.

    Deliberately preserves numbers so ``Article 21`` -> ``["article", "21"]``
    and ``21A`` -> ``["21a"]``; no stop-word removal — BM25's inverse
    document frequency already down-weights very common terms.
    """
    return _TOKEN_RE.findall((text or "").lower())


def build_bm25_retriever(
    corpus_path: Path | str,
    top_k: int,
    index: Any | None = None,
) -> "BM25Retriever":
    """Build a BM25 retriever over ``corpus_path``.

    ``corpus_path`` may point to ``constitution_chunks.jsonl`` or
    ``constitution_embeddings.jsonl`` — both carry the same ``text`` field.
    Raises ``RetrieverUnavailableError`` if the corpus is missing/corrupt
    or the ``rank_bm25`` backend is not installed.
    """
    chunks = load_corpus(corpus_path)
    tokenized = [tokenize(str(_first(r, _TEXT_KEYS) or "")) for r in chunks]
    if index is None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RetrieverUnavailableError(
                reason="bm25_unavailable",
                detail=(
                    "The BM25 backend (rank_bm25) is not installed on the "
                    "server, so lexical retrieval cannot run."
                ),
            ) from exc
        index = BM25Okapi(tokenized)
    return BM25Retriever(chunks=chunks, index=index, top_k=top_k)


class BM25Retriever:
    """Lexical BM25 retriever over the same corpus records as Dense."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        index: Any,
        top_k: int,
    ) -> None:
        self._chunks = chunks
        self._index = index
        self._top_k = max(1, int(top_k))

    def status(self) -> dict[str, Any]:
        return {
            "kind": "bm25",
            "ready": True,
            "chunks": len(self._chunks),
            "top_k": self._top_k,
            "tokenizer": "alnum_lower",
            "algorithm": "BM25Okapi",
        }

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedPassage]:
        if not question or not question.strip():
            return []
        query_tokens = tokenize(question)
        if not query_tokens:
            return []
        limit = self._top_k if top_k is None else max(1, int(top_k))
        scores = np.array(self._index.get_scores(query_tokens), dtype=np.float64)
        top_indices = np.argsort(scores)[::-1][:limit]
        passages: list[RetrievedPassage] = []
        for rank, index in enumerate(top_indices, start=1):
            passages.append(
                RetrievedPassage(
                    chunk=_to_chunk(self._chunks[index]),
                    score=float(scores[index]),
                    rank=rank,
                    retrieval_method="bm25",
                )
            )
        return passages


class HybridRetriever:
    """Fuses Dense + BM25 ranked lists with Reciprocal Rank Fusion.

    RRF score is scale-free: ``w / (rrf_k + rank)`` summed per chunk.
    This avoids mixing cosine (0-1) with unbounded BM25 scores.
    ``status()`` exposes top_k / fusion / rrf_k / weights for the run
    manifest (Task 6). Never falls back silently: unknown fusion raises.
    """

    def __init__(
        self,
        dense_retriever: Any,
        bm25_retriever: Any,
        top_k: int = 5,
        rrf_k: int = DEFAULT_RRF_K,
        fusion: str = "rrf",
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> None:
        if fusion != "rrf":
            raise ValueError(
                f"unsupported hybrid_fusion={fusion!r}: only 'rrf' is implemented "
                "(weighted fusion is deferred to evaluation Tasks 6-7)"
            )
        if int(rrf_k) <= 0:
            raise ValueError("rrf_k must be positive")
        if float(dense_weight) <= 0 or float(bm25_weight) <= 0:
            raise ValueError("hybrid weights must be positive")
        self._dense = dense_retriever
        self._bm25 = bm25_retriever
        self._top_k = max(1, int(top_k))
        self._rrf_k = int(rrf_k)
        self._fusion = fusion
        self._dense_weight = float(dense_weight)
        self._bm25_weight = float(bm25_weight)

    def status(self) -> dict[str, Any]:
        dense_status = self._dense.status()
        bm25_status = self._bm25.status()
        return {
            "kind": "hybrid",
            "ready": True,
            "chunks": dense_status.get("chunks", bm25_status.get("chunks", 0)),
            "top_k": self._top_k,
            "fusion": self._fusion,
            "rrf_k": self._rrf_k,
            "dense_weight": self._dense_weight,
            "bm25_weight": self._bm25_weight,
            "dense": dense_status,
            "bm25": bm25_status,
        }

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedPassage]:
        if not question or not question.strip():
            return []
        limit = self._top_k if top_k is None else max(1, int(top_k))
        dense_passages = self._dense.retrieve(question, top_k=limit)
        bm25_passages = self._bm25.retrieve(question, top_k=limit)
        rrf_scores: dict[str, float] = {}
        chunks: dict[str, Any] = {}
        for passage in dense_passages:
            cid = passage.chunk.chunk_id
            chunks.setdefault(cid, passage.chunk)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + self._dense_weight / (self._rrf_k + passage.rank)
        for passage in bm25_passages:
            cid = passage.chunk.chunk_id
            chunks.setdefault(cid, passage.chunk)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + self._bm25_weight / (self._rrf_k + passage.rank)
        ordered = sorted(rrf_scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
        return [
            RetrievedPassage(
                chunk=chunks[cid],
                score=float(score),
                rank=rank,
                retrieval_method="hybrid",
            )
            for rank, (cid, score) in enumerate(ordered, start=1)
        ]


def build_hybrid_retriever(
    embeddings_path: Path | str,
    chunks_path: Path | str | None = None,
    model_name: str = "BAAI/bge-m3",
    top_k: int = 5,
    embedder: Any | None = None,
    fusion: str = "rrf",
    rrf_k: int = DEFAULT_RRF_K,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> HybridRetriever:
    """Build a HybridRetriever over the same corpus version.

    Dense loads ``embeddings_path``; BM25 loads ``chunks_path`` (default
    ``embeddings_path`` fallback — both carry the same ``text`` field).
    Callers must ensure both files are from the same corpus version.
    Raises ``RetrieverUnavailableError`` if either corpus/backend is missing.
    """
    dense = build_dense_retriever(
        embeddings_path, model_name=model_name, top_k=top_k, embedder=embedder
    )
    bm25_path = chunks_path if chunks_path is not None else embeddings_path
    bm25 = build_bm25_retriever(bm25_path, top_k=top_k)
    return HybridRetriever(
        dense, bm25, top_k=top_k, rrf_k=rrf_k, fusion=fusion,
        dense_weight=dense_weight, bm25_weight=bm25_weight,
    )


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
