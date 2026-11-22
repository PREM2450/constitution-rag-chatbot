"""Task 7 threshold-calibration tests.

Exercises the pure helpers behind ``experiments/calibrate_threshold.py`` on
small synthetic rows so the calibration logic is covered without needing a real
evaluation run or matplotlib.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.calibrate_threshold import (  # noqa: E402
    calibrate_mode,
    candidate_thresholds,
    evaluate_threshold,
    failure_analysis,
    render_md,
    top1_score,
)


def make_row(
    question_id,
    answerable,
    score,
    *,
    gold=(),
    category="direct_article_question",
    split="dev",
):
    """Create one raw-result row shaped like raw_<mode>.jsonl."""
    retrieved = [
        {"chunk_id": f"c{i}", "score": score, "rank": i + 1, "retrieval_method": "fake"}
        for i in range(2)
    ] if score is not None else []
    return {
        "question_id": question_id,
        "category": category,
        "split": split,
        "answerable_from_corpus": answerable,
        "gold_chunk_ids": list(gold),
        "retrieved": retrieved,
    }


class ThresholdSweepTests(unittest.TestCase):
    def test_candidates_include_observed_and_midpoints(self):
        self.assertEqual(candidate_thresholds([0.2, 0.4]), [0.2, 0.3, 0.4])

    def test_empty_scores_give_no_candidates(self):
        self.assertEqual(candidate_thresholds([]), [])

    def test_evaluate_threshold_counts_retention_and_abstention(self):
        rows = [
            make_row("a1", True, 0.9),
            make_row("a2", True, 0.7),
            make_row("u1", False, 0.1),
        ]
        stats = evaluate_threshold(rows, 0.6)
        self.assertEqual(stats["answerable_retention"], 1.0)
        self.assertEqual(stats["unanswerable_abstention"], 1.0)
        self.assertEqual(stats["youden_j"], 1.0)

        strict = evaluate_threshold(rows, 0.75)
        self.assertEqual(strict["answerable_retention"], 0.5)
        self.assertEqual(strict["youden_j"], 0.5)

    def test_unanswerable_answered_lowers_j(self):
        # Answering an unanswerable question drives Youden's J negative.
        rows = [make_row("u1", False, 0.9)]
        stats = evaluate_threshold(rows, 0.6)
        self.assertEqual(stats["unanswerable_abstention"], 0.0)
        self.assertEqual(stats["youden_j"], -1.0)


class CalibrateModeTests(unittest.TestCase):
    def test_picks_threshold_separating_dev_groups(self):
        rows = [
            make_row("a1", True, 0.9),
            make_row("a2", True, 0.8),
            make_row("u1", False, 0.1),
            make_row("u2", False, 0.2),
        ]
        cal = calibrate_mode("fake", rows)
        chosen = cal["chosen_threshold"]
        self.assertGreater(chosen, 0.2)
        self.assertLessEqual(chosen, 0.8)
        at = cal["dev"]["at_chosen"]
        self.assertEqual(at["answerable_retention"], 1.0)
        self.assertEqual(at["unanswerable_abstention"], 1.0)


class FailureAnalysisTests(unittest.TestCase):
    def test_misses_and_false_answer_risk(self):
        rows = [
            make_row("m1", True, 0.9, gold=["g1"]),
            make_row("h1", True, 0.9, gold=["c0"]),
            make_row("u1", False, 0.9),
            make_row("u2", False, 0.1),
        ]
        result = failure_analysis("fake", rows, 0.6)
        self.assertEqual(result["no_overlap_misses"], 1)
        self.assertEqual(result["full_recall_hits"], 1)
        self.assertEqual(result["unanswerable_false_answer_risk"]["count"], 1)
        self.assertEqual(
            result["miss_examples"]["would_answer_but_wrong_chunks"],
            ["m1"],
        )

    def test_top1_score_none_when_empty(self):
        self.assertIsNone(top1_score(make_row("e", True, None)))


class RenderTests(unittest.TestCase):
    def test_markdown_has_key_sections(self):
        rows = [
            make_row("a1", True, 0.9, gold=["c0"]),
            make_row("u1", False, 0.1),
        ]
        cal = calibrate_mode("dense", rows)
        failures = {"dense": failure_analysis("dense", rows, cal["chosen_threshold"])}
        md = render_md("run-x", {"dense": cal}, failures, Path("dataset.jsonl"))
        self.assertIn("# Task 7", md)
        self.assertIn("## 2. Chosen thresholds", md)
        self.assertIn("| dense |", md)


if __name__ == "__main__":
    unittest.main()
