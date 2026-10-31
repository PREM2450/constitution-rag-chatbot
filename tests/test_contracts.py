import unittest

from src.config import Settings
from src.schemas import AnswerResponse, AskRequest, Citation, DocumentChunk, RetrievedPassage


class SettingsTests(unittest.TestCase):
    def test_default_settings(self):
        settings = Settings.from_mapping({})
        self.assertEqual(settings.retrieval_mode, "dense")
        self.assertEqual(settings.top_k, 5)
        self.assertEqual(settings.chunk_size, 1200)

    def test_env_override(self):
        settings = Settings.from_mapping({
            "llm_model": "gpt-4.1-mini",
            "retrieval_mode": "hybrid",
            "top_k": 7,
        })
        self.assertEqual(settings.llm_model, "gpt-4.1-mini")
        self.assertEqual(settings.retrieval_mode, "hybrid")
        self.assertEqual(settings.top_k, 7)

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
            page=12,
        )
        passage = RetrievedPassage(chunk=chunk, score=0.91, rank=1, retrieval_method="dense")
        response = AnswerResponse(
            answer="Answer grounded in the source.",
            citations=[Citation(chunk_id="c-1", article="Article 19", page=12)],
            sources=[passage],
            request_id="r-2",
            retrieval_method="dense",
        )
        payload = response.to_dict()
        self.assertEqual(payload["citations"][0]["chunk_id"], "c-1")
        self.assertEqual(payload["sources"][0]["rank"], 1)


if __name__ == "__main__":
    unittest.main()
