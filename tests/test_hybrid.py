"""Hybrid retriever tests: RRF merge, dedupe, determinism, no fallback.

Deterministic, no external model: fake dense/bm25 retrievers return
canned ranked lists so RRF ordering can be checked by hand.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.retriever import (
    DEFAULT_RRF_K,
    HybridRetriever,
    RetrieverUnavailableError,
    _to_chunk,
    build_hybrid_retriever,
    build_retriever,
)
from src.schemas import RetrievedPassage


def _passage(chunk_id, rank, method="dense"):
    chunk = _to_chunk({
        "chunk_id": chunk_id,
        "text": f"sample text for {chunk_id}",
        "page_start": 1,
        "article": "Article 21" if chunk_id == "c-1" else None,
        "source_title": "Constitution of India",
    })
    return RetrievedPassage(chunk=chunk, score=1.0, rank=rank, retrieval_method=method)


class FakeRetriever:
    """Returns canned passages regardless of query (except empty)."""

    def __init__(self, kind, passages):
        self._kind = kind
        self._passages = list(passages)

    def status(self):
        return {"kind": self._kind, "ready": True, "chunks": 3, "top_k": 5}

    def retrieve(self, question, top_k=None):
        if not question or not question.strip():
            return []
        limit = 5 if top_k is None else max(1, int(top_k))
        return list(self._passages)[:limit]


def _hybrid(dense_ids, bm25_ids, **kwargs):
    dense = FakeRetriever("dense", [_passage(cid, i, "dense") for i, cid in enumerate(dense_ids, 1)])
    bm25 = FakeRetriever("bm25", [_passage(cid, i, "bm25") for i, cid in enumerate(bm25_ids, 1)])
    return HybridRetriever(dense, bm25, **kwargs)


class StatusTests(unittest.TestCase):
    def test_status_exposes_fusion_manifest(self):
        retriever = _hybrid(["c-1", "c-2"], ["c-2", "c-3"], top_k=5)
        status = retriever.status()
        self.assertEqual(status["kind"], "hybrid")
        self.assertTrue(status["ready"])
        self.assertEqual(status["fusion"], "rrf")
        self.assertEqual(status["rrf_k"], DEFAULT_RRF_K)
        self.assertEqual(status["top_k"], 5)
        self.assertEqual(status["dense_weight"], 1.0)
        self.assertEqual(status["bm25_weight"], 1.0)
        self.assertEqual(status["dense"]["kind"], "dense")
        self.assertEqual(status["bm25"]["kind"], "bm25")

    def test_weighted_fusion_rejected_not_fallback(self):
        with self.assertRaises(ValueError):
            _hybrid(["c-1"], ["c-1"], fusion="weighted")

    def test_invalid_rrf_k_rejected(self):
        with self.assertRaises(ValueError):
            _hybrid(["c-1"], ["c-1"], rrf_k=0)

    def test_non_positive_weight_rejected(self):
        with self.assertRaises(ValueError):
            _hybrid(["c-1"], ["c-1"], dense_weight=0.0)


class MergeTests(unittest.TestCase):
    def test_merges_both_lists_and_dedupes(self):
        passages = _hybrid(["c-1", "c-2"], ["c-2", "c-3"]).retrieve("Article 21")
        ids = [p.chunk.chunk_id for p in passages]
        self.assertEqual(sorted(ids), ["c-1", "c-2", "c-3"])
        self.assertEqual(len(ids), len(set(ids)))

    def test_rrf_orders_overlap_first(self):
        passages = _hybrid(["c-1", "c-2"], ["c-2", "c-3"]).retrieve("q")
        ids = [p.chunk.chunk_id for p in passages]
        self.assertEqual(ids, ["c-2", "c-1", "c-3"])
        self.assertAlmostEqual(passages[0].score, 1 / 61 + 1 / 62, places=9)
        self.assertAlmostEqual(passages[1].score, 1 / 61, places=9)
        self.assertAlmostEqual(passages[2].score, 1 / 62, places=9)

    def test_deterministic_order_same_input(self):
        retriever = _hybrid(["c-1", "c-2"], ["c-2", "c-3"])
        first = [p.chunk.chunk_id for p in retriever.retrieve("same question")]
        second = [p.chunk.chunk_id for p in retriever.retrieve("same question")]
        self.assertEqual(first, second)

    def test_tie_break_by_chunk_id(self):
        passages = _hybrid(["c-2"], ["c-1"]).retrieve("q")
        ids = [p.chunk.chunk_id for p in passages]
        self.assertEqual(ids, ["c-1", "c-2"])

    def test_retrieval_method_is_hybrid_and_ranked(self):
        passages = _hybrid(["c-1"], ["c-2"]).retrieve("q")
        self.assertTrue(all(p.retrieval_method == "hybrid" for p in passages))
        self.assertEqual([p.rank for p in passages], [1, 2])

    def test_metadata_preserved(self):
        passages = _hybrid(["c-1"], []).retrieve("q")
        self.assertEqual(passages[0].chunk.article, "Article 21")
        self.assertEqual(passages[0].chunk.text, "sample text for c-1")

    def test_empty_query_returns_empty(self):
        retriever = _hybrid(["c-1"], ["c-1"])
        self.assertEqual(retriever.retrieve(""), [])
        self.assertEqual(retriever.retrieve("   "), [])

    def test_top_k_respected(self):
        passages = _hybrid(["c-1", "c-2", "c-3"], ["c-3", "c-2"]).retrieve("q")
        self.assertEqual(len(passages), 3)
        passages = _hybrid(["c-1", "c-2", "c-3"], ["c-3", "c-2"], top_k=2).retrieve("q")
        self.assertEqual(len(passages), 2)
        passages = _hybrid(["c-1", "c-2"], ["c-1", "c-2"]).retrieve("q", top_k=1)
        self.assertEqual(len(passages), 1)


class FakeEmbedder:
    def encode(self, texts, **kwargs):
        return np.ones((len(texts), 4), dtype=np.float32)


CORPUS = [
    {"chunk_id": "c-1", "text": "Article 21 protects life and personal liberty",
     "page_start": 9, "article": "Article 21", "embedding": [1.0, 0.0, 0.0, 0.0]},
    {"chunk_id": "c-2", "text": "Article 19 guarantees freedom of speech and expression",
     "page_start": 8, "article": "Article 19", "embedding": [0.0, 1.0, 0.0, 0.0]},
]


class BuilderFactoryTests(unittest.TestCase):
    def _corpus_path(self):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
        for row in CORPUS:
            handle.write(json.dumps(row) + "\n")
        handle.close()
        return Path(handle.name)

    def test_build_hybrid_from_files(self):
        path = self._corpus_path()
        try:
            retriever = build_hybrid_retriever(
                path, chunks_path=path, top_k=5, embedder=FakeEmbedder()
            )
            self.assertEqual(retriever.status()["kind"], "hybrid")
            self.assertEqual(retriever.status()["fusion"], "rrf")
            passages = retriever.retrieve("Article 21")
            self.assertTrue(passages)
            self.assertTrue(all(p.retrieval_method == "hybrid" for p in passages))
        finally:
            path.unlink(missing_ok=True)

    def test_missing_corpus_raises_never_dense_fallback(self):
        with self.assertRaises(RetrieverUnavailableError) as ctx:
            build_hybrid_retriever(
                Path("does/not/exist.jsonl"), top_k=5, embedder=FakeEmbedder()
            )
        self.assertEqual(ctx.exception.reason, "index_not_found")

    def test_build_retriever_respects_mode(self):
        path = self._corpus_path()
        try:
            dense = build_retriever("dense", path, top_k=5, embedder=FakeEmbedder())
            self.assertEqual(dense.status()["kind"], "dense")
            bm25 = build_retriever("bm25", path, chunks_path=path, top_k=5)
            self.assertEqual(bm25.status()["kind"], "bm25")
            hybrid = build_retriever(
                "hybrid", path, chunks_path=path, top_k=5, embedder=FakeEmbedder()
            )
            self.assertEqual(hybrid.status()["kind"], "hybrid")
        finally:
            path.unlink(missing_ok=True)

    def test_build_retriever_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            build_retriever("magic", Path("does/not/exist.jsonl"))

    def test_build_retriever_bm25_missing_corpus_not_fallback(self):
        with self.assertRaises(RetrieverUnavailableError):
            build_retriever("bm25", Path("does/not/exist.jsonl"))


if __name__ == "__main__":
    unittest.main()
