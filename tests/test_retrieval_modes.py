"""Task 4: per-request retrieval_mode dispatch via API.

All three modes must work end-to-end without silent fallback:
unknown mode -> 400, unavailable mode -> 503, retrieval_method honest.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from src.api import _build_retrievers, create_app
from src.config import Settings
from src.rag import create_orchestrator
from src.retriever import build_retriever
from src.schemas import AskRequest, DocumentChunk, RetrievedPassage

CHUNKS = [
    {"chunk_id": "c-1", "text": "Article 21 protects life and personal liberty",
     "page_start": 9, "article": "Article 21", "embedding": [1.0, 0.0, 0.0, 0.0]},
    {"chunk_id": "c-2", "text": "Article 19 guarantees freedom of speech and expression",
     "page_start": 8, "article": "Article 19", "embedding": [0.0, 1.0, 0.0, 0.0]},
    {"chunk_id": "c-3", "text": "directive principles state policy well-being arrangement",
     "page_start": 40, "article": None, "embedding": [0.0, 0.0, 1.0, 0.0]},
]


class FakeEmbedder:
    def encode(self, texts, **kwargs):
        return np.ones((len(texts), 4), dtype=np.float32)


def _client(available=("dense", "bm25", "hybrid"), retrieval_mode="dense"):
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
    for row in CHUNKS:
        handle.write(json.dumps(row) + "\n")
    handle.close()
    path = Path(handle.name)
    try:
        retrievers = {
            mode: build_retriever(mode, embeddings_path=path, chunks_path=path, top_k=5, embedder=FakeEmbedder())
            for mode in available
        }
        orchestrator = create_orchestrator(
            retrievers=retrievers,
            # Threshold 0.01 so RRF (~0.03) and BM25 clear it; calibration is Task 7.
            evidence_threshold=0.01,
            retrieval_mode="dense",
        )
        settings = Settings.from_mapping(
            {"embeddings_path": str(path), "retrieval_mode": retrieval_mode}
        )
        return TestClient(create_app(settings, orchestrator=orchestrator))
    finally:
        path.unlink(missing_ok=True)


class AskModeTests(unittest.TestCase):
    def test_ask_dense_mode(self):
        payload = _client().post("/ask", json={"question": "Article 21", "retrieval_mode": "dense"}).json()
        self.assertEqual(payload["retrieval_method"], "dense")
        self.assertTrue(all(s["retrieval_method"] == "dense" for s in payload["sources"]))

    def test_ask_bm25_mode(self):
        payload = _client().post("/ask", json={"question": "Article 21", "retrieval_mode": "bm25"}).json()
        self.assertEqual(payload["retrieval_method"], "bm25")
        self.assertTrue(all(s["retrieval_method"] == "bm25" for s in payload["sources"]))

    def test_ask_hybrid_mode(self):
        response = _client().post("/ask", json={"question": "Article 21", "retrieval_mode": "hybrid"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["retrieval_method"], "hybrid")
        self.assertTrue(all(s["retrieval_method"] == "hybrid" for s in payload["sources"]))
        chunk_ids = [s["chunk"]["chunk_id"] for s in payload["sources"]]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))

    def test_ask_default_is_dense(self):
        payload = _client().post("/ask", json={"question": "Article 21"}).json()
        self.assertEqual(payload["retrieval_method"], "dense")

    def test_ask_default_respects_configured_mode(self):
        payload = _client(retrieval_mode="bm25").post(
            "/ask", json={"question": "Article 21"}
        ).json()
        self.assertEqual(payload["retrieval_method"], "bm25")

    def test_ask_invalid_mode_400(self):
        response = _client().post("/ask", json={"question": "Article 21", "retrieval_mode": "magic"})
        self.assertEqual(response.status_code, 400)

    def test_unavailable_mode_503_never_dense_fallback(self):
        response = _client(available=("dense",)).post(
            "/ask", json={"question": "Article 21", "retrieval_mode": "bm25"}
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn("bm25", response.json()["detail"])


class HealthModeTests(unittest.TestCase):
    def test_health_reports_active_kinds(self):
        payload = _client().get("/health").json()
        self.assertEqual(payload["status"], "ok")
        retrieval = payload["retrieval"]
        self.assertEqual(retrieval["available_modes"], ["bm25", "dense", "hybrid"])
        self.assertEqual(set(retrieval["retrievers"]), {"dense", "bm25", "hybrid"})
        self.assertTrue(retrieval["retriever"]["ready"])

    def test_orchestrator_status_primary_is_default_mode(self):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
        for row in CHUNKS:
            handle.write(json.dumps(row) + "\n")
        handle.close()
        path = Path(handle.name)
        try:
            retrievers = {
                mode: build_retriever(mode, embeddings_path=path, chunks_path=path, top_k=5, embedder=FakeEmbedder())
                for mode in ("dense", "bm25", "hybrid")
            }
            orch = create_orchestrator(retrievers=retrievers, evidence_threshold=0.01, retrieval_mode="hybrid")
            self.assertEqual(orch.status()["retriever"]["kind"], "hybrid")
            for mode in ("dense", "bm25", "hybrid"):
                self.assertEqual(orch.answer(AskRequest(question="Article 21", retrieval_mode=mode)).retrieval_method, mode)
        finally:
            path.unlink(missing_ok=True)


class BuilderPartialTests(unittest.TestCase):
    def test_missing_corpus_all_modes_unavailable(self):
        settings = Settings.from_mapping(
            {
                "embeddings_path": "does/not/exist.jsonl",
                "corpus_path": "does/not",
                "retrieval_mode": "dense",
                "build_all_retrieval_modes": True,
            }
        )
        available, errors = _build_retrievers(settings)
        self.assertEqual(available, {})
        self.assertEqual(set(errors), {"dense", "bm25", "hybrid"})
        self.assertTrue(all(e.reason == "index_not_found" for e in errors.values()))


class LowMemoryRetrievalTests(unittest.TestCase):
    def test_selected_bm25_mode_skips_dense_model_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "constitution_chunks.jsonl"
            corpus.write_text(
                "\n".join(json.dumps(row) for row in CHUNKS),
                encoding="utf-8",
            )
            settings = Settings.from_mapping({
                "corpus_path": directory,
                "embeddings_path": str(Path(directory) / "missing_embeddings.jsonl"),
                "retrieval_mode": "bm25",
                "build_all_retrieval_modes": False,
            })

            available, errors = _build_retrievers(settings)

        self.assertEqual(set(available), {"bm25"})
        self.assertEqual(errors, {})

    def test_bm25_indexes_article_labels_for_direct_article_questions(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(json.dumps({
                "chunk_id": "article_14_chunk_0",
                "text": "The State shall not deny equality before the law.",
                "article": "Article 14",
                "title": "Equality before law",
            }) + "\n")
            handle.write(json.dumps({
                "chunk_id": "article_19_chunk_0",
                "text": "Citizens have freedom of speech and expression.",
                "article": "Article 19",
                "title": "Protection of certain rights",
            }) + "\n")
            handle.write(json.dumps({
                "chunk_id": "article_366_chunk_0",
                "text": "The term guarantee includes certain payment obligations.",
                "article": "Article 366",
                "title": "Definitions",
            }) + "\n")
            path = Path(handle.name)
        try:
            retriever = build_retriever(
                "bm25", embeddings_path=path, chunks_path=path, top_k=2
            )
            results = retriever.retrieve("What does Article 14 guarantee?")
            missing_article_results = retriever.retrieve("What does Article 404 guarantee?")
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(results[0].chunk.article, "Article 14")
        self.assertTrue(all(result.chunk.article == "Article 14" for result in results))
        self.assertEqual(missing_article_results, [])


class EvidenceThresholdModeTests(unittest.TestCase):
    def test_abstention_uses_the_threshold_for_the_requested_mode(self):
        class FixedRetriever:
            def __init__(self, mode):
                self.mode = mode

            def status(self):
                return {"kind": self.mode, "ready": True}

            def retrieve(self, question, top_k=None):
                return [RetrievedPassage(
                    chunk=DocumentChunk(chunk_id="c-1", text="Relevant text"),
                    score=0.2,
                    rank=1,
                    retrieval_method=self.mode,
                )]

        orchestrator = create_orchestrator(
            retrievers={
                "dense": FixedRetriever("dense"),
                "bm25": FixedRetriever("bm25"),
            },
            evidence_threshold=0.1,
            evidence_thresholds={"dense": 0.3, "bm25": 0.1},
        )

        dense = orchestrator.answer(AskRequest("a question", "dense"))
        bm25 = orchestrator.answer(AskRequest("a question", "bm25"))

        self.assertTrue(dense.insufficient_evidence)
        self.assertFalse(bm25.insufficient_evidence)


if __name__ == "__main__":
    unittest.main()
