"""Task 6 evaluation harness tests.

Covers the metric primitives, the runner with a deterministic fake retriever,
manifest hashing, results rendering, and a CLI smoke test (BM25, dev, 3 rows).
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.evaluate import (
    ModeResult,
    build_manifest,
    citation_accuracy,
    compute_metrics,
    context_precision,
    evaluate_mode,
    evaluate_question,
    format_results_md,
    hit_at_k,
    reciprocal_rank,
    recall_at_k,
    run_evaluation,
)
from src.schemas import DocumentChunk, RetrievedPassage

ROOT = Path(__file__).resolve().parent.parent


class FakeRetriever:
    """Returns canned ranked passages per question."""

    def __init__(self, mapping, method="fake"):
        self.mapping = mapping
        self.method = method

    def retrieve(self, question, top_k=None):
        items = self.mapping.get(question, [])
        if top_k is not None:
            items = items[:top_k]
        return [
            RetrievedPassage(
                chunk=DocumentChunk(chunk_id=cid, text=f"text of {cid}"),
                score=score,
                rank=rank,
                retrieval_method=self.method,
            )
            for rank, (cid, score) in enumerate(items, start=1)
        ]


def _passage(chunk_id, score=1.0, rank=1):
    return RetrievedPassage(
        chunk=DocumentChunk(chunk_id=chunk_id, text="t"),
        score=score,
        rank=rank,
        retrieval_method="fake",
    )


def _question_result(question_id, gold, retrieved, answerable=True, category="direct_article_question"):
    """Build a QuestionResult by running a fake retriever for one question."""
    return evaluate_question(
        FakeRetriever({question_id: retrieved}),
        {
            "question_id": question_id,
            "question": question_id,
            "category": category,
            "split": "dev",
            "answerable_from_corpus": answerable,
            "evidence_chunk_ids": gold,
            "relevant_articles": [],
        },
        top_k=5,
    )


class MetricPrimitiveTests(unittest.TestCase):
    def test_recall_at_k_full(self):
        self.assertEqual(recall_at_k(["a", "b"], ["a", "b"], 2), 1.0)

    def test_recall_at_k_partial(self):
        self.assertEqual(recall_at_k(["a", "c"], ["a", "b"], 2), 0.5)

    def test_recall_at_k_respects_k(self):
        # Only the first k retrieved IDs count toward the metric.
        self.assertEqual(recall_at_k(["a", "b", "c"], ["c"], 1), 0.0)
        self.assertEqual(recall_at_k(["a", "b", "c"], ["c"], 3), 1.0)

    def test_recall_at_k_empty_gold(self):
        self.assertEqual(recall_at_k(["a"], [], 5), 0.0)

    def test_hit_at_k(self):
        self.assertEqual(hit_at_k(["x", "a"], ["a"], 2), 1.0)
        self.assertEqual(hit_at_k(["x", "y"], ["a"], 2), 0.0)

    def test_reciprocal_rank(self):
        self.assertEqual(reciprocal_rank(["x", "a", "b"], ["a", "b"]), 0.5)
        self.assertEqual(reciprocal_rank(["x"], ["a"]), 0.0)
        self.assertEqual(reciprocal_rank(["a"], ["a"]), 1.0)

    def test_context_precision(self):
        self.assertEqual(context_precision(["a", "x"], ["a"]), 0.5)
        self.assertEqual(context_precision([], ["a"]), 0.0)
        self.assertEqual(context_precision(["a", "b"], ["a", "b"]), 1.0)

    def test_citation_accuracy(self):
        self.assertEqual(citation_accuracy(["a", "b"], ["a", "b", "c"]), 1.0)
        self.assertEqual(citation_accuracy(["a", "z"], ["a", "b"]), 0.5)
        self.assertEqual(citation_accuracy([], ["a"]), 1.0)


class RunnerTests(unittest.TestCase):
    def test_evaluate_question_records_retrieved(self):
        result = _question_result("q1", ["c-1"], [("c-1", 0.9), ("c-2", 0.4)])
        self.assertEqual(result.question_id, "q1")
        self.assertEqual([r["chunk_id"] for r in result.retrieved], ["c-1", "c-2"])
        self.assertEqual(result.retrieved[0]["rank"], 1)
        self.assertGreaterEqual(result.latency_ms, 0.0)

    def test_evaluate_question_none_mode(self):
        result = evaluate_question(None, {
            "question_id": "q1", "question": "q1", "category": "c", "split": "dev",
            "answerable_from_corpus": True, "evidence_chunk_ids": ["c-1"],
            "relevant_articles": [],
        }, top_k=5)
        self.assertEqual(result.retrieved, [])
        self.assertEqual(result.latency_ms, 0.0)

    def test_evaluate_mode_aggregates(self):
        rows = [
            {"question_id": "q1", "question": "q1", "category": "direct_article_question",
             "split": "dev", "answerable_from_corpus": True,
             "evidence_chunk_ids": ["c-1"], "relevant_articles": ["1"]},
            {"question_id": "q2", "question": "q2", "category": "unanswerable_out_of_corpus",
             "split": "dev", "answerable_from_corpus": False,
             "evidence_chunk_ids": [], "relevant_articles": []},
        ]
        retriever = FakeRetriever({"q1": [("c-1", 0.9)]})
        result = evaluate_mode("fake", retriever, rows, top_k=5)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.metrics["n_questions"], 2.0)
        self.assertEqual(result.metrics["n_answerable"], 1.0)
        self.assertEqual(result.metrics["recall_at_k"], 1.0)
        self.assertEqual(result.metrics["mrr"], 1.0)
        self.assertEqual(result.metrics["citation_accuracy"], 1.0)

    def test_evaluate_mode_records_errors(self):
        class Broken:
            def retrieve(self, question, top_k=None):
                raise RuntimeError("boom")

        rows = [{"question_id": "q1", "question": "q1", "category": "c", "split": "dev",
                 "answerable_from_corpus": True, "evidence_chunk_ids": ["c-1"],
                 "relevant_articles": []}]
        result = evaluate_mode("broken", Broken(), rows, top_k=5)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("boom", result.errors[0])
        self.assertEqual(result.questions, [])

    def test_run_evaluation_aliases(self):
        # The retriever key stays "dense" but the reported mode is relabelled.
        rows = [{"question_id": "q1", "question": "q1", "category": "c", "split": "dev",
                 "answerable_from_corpus": True, "evidence_chunk_ids": ["c-1"],
                 "relevant_articles": []}]
        results = run_evaluation(
            rows, {"dense": FakeRetriever({"q1": [("c-1", 0.9)]})}, top_k=5,
            mode_aliases={"dense": "no_retrieval"},
        )
        self.assertIn("no_retrieval", results)
        self.assertEqual(results["no_retrieval"].metrics["recall_at_k"], 1.0)


class ManifestAndReportTests(unittest.TestCase):
    def test_build_manifest_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "d.jsonl"
            data.write_text("{}\n", encoding="utf-8")
            result = ModeResult(mode="m", top_k=5, metrics={"n_questions": 1.0})
            manifest = build_manifest(
                run_id="r1", config={"top_k": 5}, dataset_path=data,
                chunks_path=Path(tmp) / "missing.jsonl", results={"m": result},
            )
            self.assertEqual(manifest["run_id"], "r1")
            self.assertEqual(len(manifest["dataset"]["sha256"]), 64)
            self.assertIsNone(manifest["corpus"]["sha256"])
            self.assertFalse(manifest["ragas"]["enabled"])

    def test_format_results_md_contains_metrics(self):
        result = ModeResult(mode="bm25", top_k=5, metrics={
            "n_questions": 2.0, "n_answerable": 1.0, "recall_at_k": 0.5,
            "hit_at_k": 1.0, "mrr": 0.5, "context_precision": 0.5,
            "context_recall": 0.5, "citation_accuracy": 1.0,
            "latency_ms_mean": 1.0, "latency_ms_p50": 1.0, "latency_ms_p95": 1.0,
        })
        md = format_results_md({"bm25": result})
        self.assertIn("bm25", md)
        self.assertIn("recall@k", md)
        self.assertIn("0.500", md)

    def test_format_results_md_shows_error_row(self):
        md = format_results_md({"dense": ModeResult(mode="dense", top_k=5, errors=["index_not_found"])})
        self.assertIn("index_not_found", md)


class CliSmokeTests(unittest.TestCase):
    def test_cli_bm25_dev_limit(self):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "experiments" / "run_evaluation.py"),
             "--modes", "bm25", "--split", "dev", "--limit", "3", "--no-save"],
            capture_output=True, text=True, cwd=ROOT,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("recall@k", proc.stdout)
        self.assertIn("bm25", proc.stdout)


if __name__ == "__main__":
    unittest.main()
