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
from src.schemas import AskRequest

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
            {"embeddings_path": "does/not/exist.jsonl", "corpus_path": "does/not"}
        )
        available, errors = _build_retrievers(settings)
        self.assertEqual(available, {})
        self.assertEqual(set(errors), {"dense", "bm25", "hybrid"})
        self.assertTrue(all(e.reason == "index_not_found" for e in errors.values()))


if __name__ == "__main__":
    unittest.main()
