"""Hybrid retriever fusing Dense + BM25 ranked lists with Reciprocal Rank Fusion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.bm25_retriever import build_bm25_retriever
from src.dense_retriever import build_dense_retriever
from src.retrieval_base import DEFAULT_RRF_K
from src.schemas import RetrievedPassage


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
        # Fail fast on unsupported settings instead of silently ignoring them,
        # so a misconfigured deployment is caught at build time.
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
        """Return the RRF-fused top ``top_k`` chunks for ``question``.

        Each retriever contributes ``weight / (rrf_k + rank)`` for a chunk. A
        chunk appearing in both lists accumulates two contributions, so
        agreement between dense and BM25 is rewarded. Ranking is deterministic:
        ties are broken by ``chunk_id`` so repeated runs give identical output.
        """
        if not question or not question.strip():
            return []
        limit = self._top_k if top_k is None else max(1, int(top_k))
        # Ask each sub-retriever for the same limit; fusion handles the merge.
        dense_passages = self._dense.retrieve(question, top_k=limit)
        bm25_passages = self._bm25.retrieve(question, top_k=limit)
        # chunk_id -> summed RRF score, and chunk_id -> chunk metadata.
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
        # Sort by descending score; the secondary chunk_id key makes ties stable.
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
