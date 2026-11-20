import unittest

from experiments.remap_gold_questions import remap_questions


class GoldQuestionRemapTests(unittest.TestCase):
    def test_remaps_article_labels_to_current_chunks(self):
        rows = [
            {
                "question_id": "q1",
                "question": "What does Article 21 protect?",
                "category": "direct_article_question",
                "split": "dev",
                "answerable_from_corpus": True,
                "gold_articles": ["21"],
            },
            {
                "question_id": "q2",
                "question": "Unanswerable question",
                "category": "unanswerable_out_of_corpus",
                "split": "test",
                "answerable_from_corpus": False,
                "gold_articles": [],
            },
        ]
        chunks = [
            {"chunk_id": "article_21_chunk_1", "article_number": "21"},
            {"chunk_id": "article_21_chunk_0", "article_number": "21"},
        ]

        result = remap_questions(rows, chunks)

        self.assertEqual(
            result[0]["evidence_chunk_ids"],
            ["article_21_chunk_0", "article_21_chunk_1"],
        )
        self.assertEqual(result[0]["relevant_articles"], ["21"])
        self.assertEqual(result[1]["evidence_chunk_ids"], [])

    def test_fails_when_gold_article_is_missing_from_corpus(self):
        row = {
            "question_id": "q1",
            "answerable_from_corpus": True,
            "gold_articles": ["999"],
        }
        with self.assertRaisesRegex(ValueError, "no current corpus chunks"):
            remap_questions([row], [])


if __name__ == "__main__":
    unittest.main()
