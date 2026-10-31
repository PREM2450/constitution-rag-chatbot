"""API and retriever tests using a deterministic fake embedder.

No external model, network or LLM is required: a hashing embedder converts
tokens into unit bag-of-words vectors, so identical query/chunk wording gets
a perfect score and unrelated wording gets ~0. This keeps evidence-threshold
behaviour fully deterministic.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from src.api import MAX_QUESTION_LENGTH, create_app
from src.config import Settings
from src.rag import create_orchestrator
from src.retriever import DenseRetriever, _to_chunk
from src.schemas import AskRequest, RetrievedPassage


class HashingEmbedder:
    """Deterministic unit bag-of-words embedder (no external model needed)."""

    def encode(self, texts, **kwargs):
        vectors = np.zeros((len(texts), 64), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in text.lower().split():
                vectors[i, _hash_token(token)] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return vectors / norms


def _hash_token(token: str) -> int:
    value = 2166136261
    for character in token:
        value = (value * 131) + ord(character)
        value &= 0xFFFFFFFF
    return value % 64


CHUNKS = [
    {
        "chunk_id": "c-1",
        "text": "Article 21 protects life and personal liberty",
        "page_start": 9,
        "article": "Article 21",
        "source_title": "Constitution of India",
        "source_url": "https://example.invalid/constitution",
    },
    {
        "chunk_id": "c-2",
        "text": "Article 19 guarantees freedom of speech and expression",
        "page_start": 8,
        "article": "Article 19",
        "source_title": "Constitution of India",
        "source_url": "https://example.invalid/constitution",
    },
    {
        "chunk_id": "c-3",
        "text": "directive principles state policy well-being arrangement",
        "page_start": 40,
        "article": None,
        "source_title": "Constitution of India",
        "source_url": "",
    },
]

TOP_K = 5


def _build_client():
    chunks = [dict(chunk, embedding=None) for chunk in CHUNKS]
    for chunk in chunks:
        chunk["embedding"] = HashingEmbedder().encode([chunk["text"]])[0].tolist()

    vectors = np.array([chunk["embedding"] for chunk in chunks], dtype=np.float32)
    retriever = DenseRetriever(
        chunks=chunks, vectors=vectors, embedder=HashingEmbedder(), top_k=TOP_K
    )
    orchestrator = create_orchestrator(
        retriever=retriever,
        # The hashing-embedder scores are lower than the real model's;
        # this threshold is calibrated for the fixture only.
        evidence_threshold=0.3,
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        embeddings_file = Path(handle.name)

    try:
        settings = Settings.from_mapping({"embeddings_path": str(embeddings_file)})
        app = create_app(settings, orchestrator=orchestrator)
        return TestClient(app)
    finally:
        embeddings_file.unlink(missing_ok=True)


class HealthEndpointTests(unittest.TestCase):
    def test_health_reports_ready_status(self):
        client = _build_client()
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["retrieval"]["retriever"]["ready"])

    def test_health_is_degraded_when_index_missing(self):
        settings = Settings.from_mapping(
            {"embeddings_path": "does/not/exist.jsonl"}
        )
        client = TestClient(create_app(settings))
        response = client.get("/health")
        payload = response.json()
        self.assertEqual(payload["status"], "degraded")
        self.assertFalse(payload["retrieval"]["retriever"]["ready"])
        self.assertEqual(payload["retrieval"]["retriever"]["reason"], "index_not_found")


class AskEndpointTests(unittest.TestCase):
    def test_answer_with_citations(self):
        client = _build_client()
        response = client.post(
            "/ask",
            json={"question": "What does Article 21 protect?", "retrieval_mode": "dense"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()

        self.assertFalse(payload["insufficient_evidence"])
        self.assertEqual(payload["retrieval_method"], "dense")
        self.assertTrue(payload["request_id"])

        chunk_ids = {citation["chunk_id"] for citation in payload["citations"]}
        self.assertIn("c-1", chunk_ids)
        top_citation = payload["citations"][0]
        self.assertEqual(top_citation["article"], "Article 21")
        self.assertEqual(top_citation["page"], 9)

        # Citation chunk IDs must always belong to the retrieved sources.
        source_ids = {source["chunk"]["chunk_id"] for source in payload["sources"]}
        self.assertTrue(chunk_ids.issubset(source_ids))

        # The answer must not pretend an LLM generated it.
        self.assertIn("extractive baseline", payload["answer"])

    def test_insufficient_evidence_abstains(self):
        client = _build_client()
        response = client.post(
            "/ask",
            json={"question": "quantum flux orbital subroutine zzz"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["insufficient_evidence"])
        self.assertIn("Insufficient evidence", payload["warning"])
        # Abstention answers still cite only what was retrieved.
        self.assertEqual(len(payload["citations"]), len(payload["sources"]))

    def test_empty_question_rejected(self):
        client = _build_client()
        response = client.post("/ask", json={"question": "   "})
        self.assertEqual(response.status_code, 400)
        self.assertIn("non-empty", response.json()["detail"])

    def test_missing_question_rejected(self):
        client = _build_client()
        response = client.post("/ask", json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn("required", response.json()["detail"])

    def test_non_string_question_rejected(self):
        client = _build_client()
        response = client.post("/ask", json={"question": 42})
        self.assertEqual(response.status_code, 400)

    def test_question_too_long_rejected(self):
        client = _build_client()
        response = client.post(
            "/ask", json={"question": "x" * (MAX_QUESTION_LENGTH + 1)}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("too long", response.json()["detail"])

    def test_invalid_retrieval_mode_rejected(self):
        client = _build_client()
        response = client.post(
            "/ask",
            json={"question": "What does Article 21 protect?", "retrieval_mode": "magic"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("retrieval_mode", response.json()["detail"])


class RetrieverTests(unittest.TestCase):
    def test_chunk_metadata_mapping(self):
        chunk = _to_chunk(CHUNKS[0])
        self.assertEqual(chunk.chunk_id, "c-1")
        self.assertEqual(chunk.article, "Article 21")
        self.assertEqual(chunk.page, 9)

    def test_missing_article_stays_none(self):
        chunk = _to_chunk(CHUNKS[2])
        self.assertIsNone(chunk.article)
        self.assertEqual(chunk.page, 40)

    def test_retrieve_orders_chunks_by_similarity(self):
        chunks = [dict(c, embedding=None) for c in CHUNKS]
        for chunk in chunks:
            chunk["embedding"] = HashingEmbedder().encode([chunk["text"]])[0].tolist()
        retriever = DenseRetriever(
            chunks=chunks,
            vectors=np.array([c["embedding"] for c in chunks], dtype=np.float32),
            embedder=HashingEmbedder(),
            top_k=TOP_K,
        )
        passages = retriever.retrieve("What does Article 21 protect?")
        self.assertEqual(passages[0].chunk.chunk_id, "c-1")
        self.assertTrue(all(p.rank > 0 for p in passages))


class OrchestratorTests(unittest.TestCase):
    def test_generator_seam_is_replaceable(self):
        class StubGeneration:
            name = "stub"

            def generate(self, question, passages):
                return "stubbed grounded answer"

        passage = RetrievedPassage(
            chunk=_to_chunk(CHUNKS[0]), score=0.9, rank=1, retrieval_method="dense"
        )

        class FakeRetriever:
            def status(self):
                return {"kind": "fake", "ready": True}

            def retrieve(self, question, top_k=None):
                return [passage]

        orchestrator = create_orchestrator(
            retriever=FakeRetriever(), generator=StubGeneration()
        )
        response = orchestrator.answer(
            AskRequest(question="What does Article 21 protect?")
        )
        self.assertEqual(response.answer, "stubbed grounded answer")
        self.assertFalse(response.insufficient_evidence)
        self.assertEqual(response.citations[0].chunk_id, "c-1")


if __name__ == "__main__":
    unittest.main()
