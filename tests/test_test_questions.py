"""Task 5 dataset contract tests.

Validates schema, category coverage, split integrity, and that every
evidence_chunk_id resolves to a real corpus chunk when the corpus is present.
The corpus and dataset live under ``Data/`` (gitignored), so tests skip when
they are absent in a clean checkout.
"""

import json
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "Data" / "processed" / "test_questions.jsonl"
CHUNKS = ROOT / "Data" / "processed" / "constitution_chunks.jsonl"

CATEGORIES = {
    "direct_article_question",
    "paraphrased_question",
    "multi_article_question",
    "keyword_heavy_question",
    "ambiguous_question",
    "unanswerable_out_of_corpus",
}

REQUIRED = {
    "question_id",
    "question",
    "reference_answer",
    "relevant_articles",
    "evidence_chunk_ids",
    "answerable_from_corpus",
    "category",
    "split",
    "review_status",
}


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestQuestionDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DATASET.exists():
            raise unittest.SkipTest(f"dataset not present: {DATASET}")
        cls.rows = _load(DATASET)

    def test_row_count_and_required_fields(self):
        self.assertGreaterEqual(len(self.rows), 150)
        self.assertLessEqual(len(self.rows), 200)
        for row in self.rows:
            self.assertLessEqual(REQUIRED, set(row), row.get("question_id"))

    def test_unique_ids_and_questions(self):
        self.assertEqual(len(self.rows), len({r["question_id"] for r in self.rows}))
        self.assertEqual(len(self.rows), len({r["question"] for r in self.rows}))

    def test_categories_and_splits(self):
        self.assertEqual({r["category"] for r in self.rows}, CATEGORIES)
        self.assertLessEqual({r["split"] for r in self.rows}, {"dev", "test"})
        for category in CATEGORIES:
            splits = Counter(r["split"] for r in self.rows if r["category"] == category)
            self.assertGreaterEqual(splits["dev"], 1, category)
            self.assertGreaterEqual(splits["test"], 1, category)

    def test_answerability_invariants(self):
        for row in self.rows:
            self.assertIsInstance(row["answerable_from_corpus"], bool)
            self.assertIsInstance(row["relevant_articles"], list)
            self.assertIsInstance(row["evidence_chunk_ids"], list)
            if row["answerable_from_corpus"]:
                self.assertTrue(row["relevant_articles"], row["question_id"])
                self.assertTrue(row["evidence_chunk_ids"], row["question_id"])
            else:
                self.assertEqual(row["relevant_articles"], [], row["question_id"])
                self.assertEqual(row["evidence_chunk_ids"], [], row["question_id"])
                self.assertTrue(row.get("notes"), row["question_id"])

    def test_evidence_ids_resolve_to_corpus(self):
        if not CHUNKS.exists():
            self.skipTest(f"corpus not present: {CHUNKS}")
        chunks = {c["chunk_id"]: c for c in _load(CHUNKS)}
        for row in self.rows:
            for chunk_id in row["evidence_chunk_ids"]:
                self.assertIn(chunk_id, chunks, row["question_id"])
            if row["answerable_from_corpus"] and len(row["relevant_articles"]) == 1:
                article = row["relevant_articles"][0]
                joined = " ".join(chunks[cid]["text"] for cid in row["evidence_chunk_ids"])
                self.assertIn(f"{article}.", joined, row["question_id"])

    def test_no_duplicate_evidence_within_row(self):
        for row in self.rows:
            self.assertEqual(
                len(row["evidence_chunk_ids"]),
                len(set(row["evidence_chunk_ids"])),
                row["question_id"],
            )


if __name__ == "__main__":
    unittest.main()
