import unittest
from unittest.mock import patch

import httpx

from src.config import Settings
from src.generation import (
    ExtractiveBaselineGenerator,
    GenerationError,
    OpenAICompatibleGenerator,
    create_generation_service,
)
from src.schemas import DocumentChunk, RetrievedPassage


def _passage():
    return RetrievedPassage(
        chunk=DocumentChunk(
            chunk_id="article_21_chunk_0",
            text="No person shall be deprived of life or personal liberty.",
            article="Article 21",
            page=42,
        ),
        score=0.8,
        rank=1,
        retrieval_method="dense",
    )


class OpenAICompatibleGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.generator = OpenAICompatibleGenerator(
            api_key="test-key",
            model="test-model",
            base_url="https://llm.example.invalid/v1/",
        )

    @patch("src.generation.httpx.post")
    def test_sends_grounding_context_and_returns_provider_answer(self, post):
        post.return_value = httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Article 21 protects liberty."}}]},
        )

        answer = self.generator.generate("What does Article 21 protect?", [_passage()])

        self.assertEqual(answer, "Article 21 protects liberty.")
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://llm.example.invalid/v1/chat/completions")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")
        prompt = kwargs["json"]["messages"][1]["content"]
        self.assertIn("Article 21", prompt)
        self.assertIn("No person shall be deprived", prompt)

    @patch("src.generation.httpx.post")
    def test_provider_error_is_reported_without_response_body(self, post):
        post.return_value = httpx.Response(
            401, json={"error": {"message": "secret provider detail"}}
        )

        with self.assertRaisesRegex(GenerationError, "HTTP 401") as error:
            self.generator.generate("Question", [_passage()])

        self.assertNotIn("secret provider detail", str(error.exception))

    @patch("src.generation.httpx.post")
    def test_quota_error_returns_actionable_message_without_provider_body(self, post):
        post.return_value = httpx.Response(
            429,
            json={
                "error": {
                    "code": "insufficient_quota",
                    "message": "private provider detail",
                }
            },
        )

        with self.assertRaises(GenerationError) as error:
            self.generator.generate("Question", [_passage()])

        self.assertEqual(error.exception.http_status, 429)
        self.assertIn("billing", error.exception.user_message)
        self.assertNotIn("private provider detail", error.exception.user_message)

    @patch("src.generation.httpx.post")
    def test_rate_limit_error_suggests_retry(self, post):
        post.return_value = httpx.Response(
            429, json={"error": {"code": "rate_limit_exceeded"}}
        )

        with self.assertRaises(GenerationError) as error:
            self.generator.generate("Question", [_passage()])

        self.assertIn("Wait briefly", error.exception.user_message)

    @patch("src.generation.httpx.post")
    def test_malformed_provider_response_is_rejected(self, post):
        post.return_value = httpx.Response(200, json={"choices": []})

        with self.assertRaisesRegex(GenerationError, "no answer text"):
            self.generator.generate("Question", [_passage()])

    def test_missing_api_key_uses_transparent_extractive_baseline(self):
        generator = create_generation_service(
            Settings.from_mapping({"llm_api_key": ""})
        )

        self.assertIsInstance(generator, ExtractiveBaselineGenerator)


if __name__ == "__main__":
    unittest.main()
