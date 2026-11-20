import os
import unittest
from unittest.mock import patch

from src.config import Settings
from src.schemas import AnswerResponse, AskRequest, Citation, DocumentChunk, RetrievedPassage


class SettingsTests(unittest.TestCase):
    def test_default_settings(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_mapping({})
        self.assertEqual(settings.retrieval_mode, "dense")
        self.assertTrue(settings.build_all_retrieval_modes)
        self.assertEqual(settings.top_k, 5)
        self.assertEqual(settings.chunk_size, 1200)
        self.assertEqual(
            settings.mode_evidence_thresholds,
            {"dense": 0.6, "bm25": 1.0, "hybrid": 0.01},
        )

    def test_env_override(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_mapping({
                "llm_model": "gpt-4.1-mini",
                "llm_base_url": "https://llm.example.invalid/v1",
                "retrieval_mode": "hybrid",
                "build_all_retrieval_modes": False,
                "top_k": 7,
                "evidence_threshold": 0.4,
                "bm25_evidence_threshold": 17.5,
            })
        self.assertEqual(settings.llm_model, "gpt-4.1-mini")
        self.assertEqual(settings.llm_base_url, "https://llm.example.invalid/v1")
        self.assertEqual(settings.retrieval_mode, "hybrid")
        self.assertFalse(settings.build_all_retrieval_modes)
        self.assertEqual(settings.top_k, 7)
        self.assertEqual(
            settings.mode_evidence_thresholds,
            {"dense": 0.4, "bm25": 17.5, "hybrid": 0.01},
        )

    def test_invalid_llm_provider_rejected(self):
        with self.assertRaisesRegex(ValueError, "llm_provider"):
            Settings.from_mapping({"llm_provider": "unknown"})

    def test_non_http_llm_base_url_rejected(self):
        with self.assertRaisesRegex(ValueError, "HTTP"):
            Settings.from_mapping({"llm_base_url": "file:///tmp/provider"})

    def test_invalid_chunk_size_rejected(self):
        with self.assertRaises(ValueError):
            Settings.from_mapping({"chunk_size": 0})


class SchemaTests(unittest.TestCase):
    def test_ask_request_requires_question(self):
        with self.assertRaises(ValueError):
            AskRequest(question="   ")

    def test_answer_response_insufficient_evidence_warning(self):
        response = AnswerResponse(
            answer="Need more evidence.",
            insufficient_evidence=True,
            request_id="r-1",
        )
        self.assertIn("Insufficient evidence", response.warning)

    def test_retrieved_passage_serialization(self):
        chunk = DocumentChunk(
            chunk_id="c-1",
            text="Sample constitutional text",
            article="Article 19",
            section="First Schedule",
            page=12,
        )
        passage = RetrievedPassage(chunk=chunk, score=0.91, rank=1, retrieval_method="dense")
        response = AnswerResponse(
            answer="Answer grounded in the source.",
            citations=[Citation(
                chunk_id="c-1",
                article="Article 19",
                section="First Schedule",
                page=12,
            )],
            sources=[passage],
            request_id="r-2",
            retrieval_method="dense",
        )
        payload = response.to_dict()
        self.assertEqual(payload["citations"][0]["chunk_id"], "c-1")
        self.assertEqual(payload["citations"][0]["section"], "First Schedule")
        self.assertEqual(payload["sources"][0]["chunk"]["section"], "First Schedule")
        self.assertEqual(payload["sources"][0]["rank"], 1)


if __name__ == "__main__":
    unittest.main()
