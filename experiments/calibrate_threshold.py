"""Task 7 -- evidence-threshold calibration and failure analysis.

Reads the raw per-question outputs of an evaluation run (``raw_<mode>.jsonl``)
and, for each retrieval mode:

* reports top-1 score distributions for answerable vs unanswerable questions,
  split by dev / held-out test;
* sweeps candidate ``EVIDENCE_THRESHOLD`` values on the **dev split only** and
  picks the value that maximises Youden's J (answerable-retention minus
  unanswerable-answer-rate), then confirms it on the held-out split;
* writes ``threshold_calibration.json``, ``threshold_calibration.md`` and
  ``score_distributions.png`` next to the run.

It also produces a failure breakdown for answerable questions (no gold overlap
in top-k) bucketed by category and by whether the calibrated threshold would
have abstained, plus the unanswerable false-answer count.

Usage:
    python experiments/calibrate_threshold.py --run-id 20261119_040644
    python experiments/calibrate_threshold.py --run-dir experiments/results/<run_id>

The script never invents scores: every number comes from the run's raw files.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"


def load_raw(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def top1_score(row: dict[str, Any]) -> float | None:
    retrieved = row.get("retrieved") or []
    if not retrieved:
        return None
    return float(retrieved[0]["score"])


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


def distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": round(min(values), 4),
        "p10": round(percentile(values, 10), 4),
        "p25": round(percentile(values, 25), 4),
        "median": round(percentile(values, 50), 4),
        "p75": round(percentile(values, 75), 4),
        "p90": round(percentile(values, 90), 4),
        "max": round(max(values), 4),
        "mean": round(statistics.mean(values), 4),
    }


def candidate_thresholds(dev_scores: list[float]) -> list[float]:
    """Candidate cut-offs: every observed dev score (rounded) plus midpoints."""
    if not dev_scores:
        return []
    unique = sorted({round(v, 4) for v in dev_scores if v is not None})
    candidates = set(unique)
    for a, b in zip(unique, unique[1:]):
        candidates.add(round((a + b) / 2.0, 4))
    return sorted(candidates)


def evaluate_threshold(
    rows: list[dict[str, Any]], threshold: float
) -> dict[str, float]:
    """Abstention behaviour at a threshold, split by answerability."""
    answerable = [r for r in rows if r["answerable_from_corpus"]]
    unanswerable = [r for r in rows if not r["answerable_from_corpus"]]

    def answered(row: dict[str, Any]) -> bool:
        score = top1_score(row)
        return score is not None and score >= threshold

    n_ans = len(answerable)
    n_unans = len(unanswerable)
    ans_answered = sum(1 for r in answerable if answered(r))
    unans_answered = sum(1 for r in unanswerable if answered(r))

    retention = ans_answered / n_ans if n_ans else 0.0
    false_answer = unans_answered / n_unans if n_unans else 0.0
    return {
        "threshold": threshold,
        "answerable_total": n_ans,
        "answerable_answered": ans_answered,
        "answerable_abstained": n_ans - ans_answered,
        "answerable_retention": retention,
        "unanswerable_total": n_unans,
        "unanswerable_answered": unans_answered,
        "unanswerable_abstained": n_unans - unans_answered,
        "unanswerable_abstention": 1.0 - false_answer,
        "youden_j": retention - false_answer,
    }


def calibrate_mode(mode: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    dev = [r for r in rows if r.get("split") == "dev"]
    test = [r for r in rows if r.get("split") == "test"]

    dev_scores = [top1_score(r) for r in dev]
    candidates = candidate_thresholds(dev_scores) or [0.0]

    sweep = [evaluate_threshold(dev, t) for t in candidates]
    # Maximise Youden's J on dev; tie-break toward the lower threshold so we do
    # not over-abstain on answerable questions.
    best = max(sweep, key=lambda s: (round(s["youden_j"], 6), -s["threshold"]))
    chosen = best["threshold"]
    test_at_chosen = evaluate_threshold(test, chosen)
    dev_at_chosen = best

    return {
        "mode": mode,
        "n_dev": len(dev),
        "n_test": len(test),
        "chosen_threshold": chosen,
        "dev": {
            "answerable_top1": distribution(
                [top1_score(r) for r in dev if r["answerable_from_corpus"]]
            ),
            "unanswerable_top1": distribution(
                [top1_score(r) for r in dev if not r["answerable_from_corpus"]]
            ),
            "at_chosen": dev_at_chosen,
            "sweep": sweep,
        },
        "test": {
            "answerable_top1": distribution(
                [top1_score(r) for r in test if r["answerable_from_corpus"]]
            ),
            "unanswerable_top1": distribution(
                [top1_score(r) for r in test if not r["answerable_from_corpus"]]
            ),
            "at_chosen": test_at_chosen,
        },
    }


def failure_analysis(
    mode: str, rows: list[dict[str, Any]], threshold: float
) -> dict[str, Any]:
    def answered(row: dict[str, Any]) -> bool:
        score = top1_score(row)
        return score is not None and score >= threshold

    def overlap(row: dict[str, Any]) -> int:
        gold = set(row.get("gold_chunk_ids") or [])
        retrieved = {item["chunk_id"] for item in (row.get("retrieved") or [])}
        return len(gold & retrieved)

    answerable = [r for r in rows if r["answerable_from_corpus"]]
    unanswerable = [r for r in rows if not r["answerable_from_corpus"]]

    misses = [r for r in answerable if overlap(r) == 0]
    partial = [r for r in answerable if 0 < overlap(r) < len(set(r["gold_chunk_ids"]))]
    hits = [r for r in answerable if overlap(r) == len(set(r["gold_chunk_ids"]))]

    by_category: dict[str, dict[str, int]] = {}
    for r in misses:
        bucket = by_category.setdefault(
            r["category"], {"misses": 0, "of_which_would_abstain": 0}
        )
        bucket["misses"] += 1
        if not answered(r):
            bucket["of_which_would_abstain"] += 1

    return {
        "mode": mode,
        "threshold": threshold,
        "answerable": len(answerable),
        "full_recall_hits": len(hits),
        "partial_overlap": len(partial),
        "no_overlap_misses": len(misses),
        "misses_by_category": {
            k: v for k, v in sorted(by_category.items(), key=lambda kv: -kv[1]["misses"])
        },
        "miss_examples": {
            "would_answer_but_wrong_chunks": [
                r["question_id"] for r in misses if answered(r)
            ][:8],
            "would_abstain": [r["question_id"] for r in misses if not answered(r)][:8],
        },
        "unanswerable_false_answer_risk": {
            "count": sum(1 for r in unanswerable if answered(r)),
            "total": len(unanswerable),
            "examples": [r["question_id"] for r in unanswerable if answered(r)][:8],
        },
    }


def render_md(
    run_id: str,
    calibration: dict[str, Any],
    failures: dict[str, Any],
    dataset_path: Path,
) -> str:
    lines = [
        "# Task 7 -- Evidence Threshold Calibration + Failure Analysis",
        "",
        f"- Run ID: `{run_id}`",
        f"- Dataset: `{dataset_path}`",
        "- Method: candidate thresholds are every observed dev top-1 score plus "
        "midpoints; selection maximises Youden's J "
        "(answerable-retention - unanswerable-answer-rate) on **dev only**, then "
        "is confirmed on held-out test.",
        "- Score semantics differ per mode: dense = cosine similarity, "
        "bm25 = unbounded BM25Okapi score, hybrid = RRF fusion score "
        "(~1/(60+rank), i.e. very small). One shared threshold cannot fit all "
        "three, which is why per-mode calibration matters.",
        "",
        "## 1. Top-1 score distributions",
        "",
    ]
    for mode, cal in calibration.items():
        lines.append(f"### {mode}")
        lines.append("")
        lines.append("| split | group | n | min | p25 | median | p75 | max |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for split in ("dev", "test"):
            for group in ("answerable_top1", "unanswerable_top1"):
                d = cal[split][group]
                label = "answerable" if group.startswith("answerable") else "unanswerable"
                if d.get("n"):
                    lines.append(
                        f"| {split} | {label} | {d['n']} | {d['min']:.4f} | "
                        f"{d['p25']:.4f} | {d['median']:.4f} | {d['p75']:.4f} | {d['max']:.4f} |"
                    )
                else:
                    lines.append(f"| {split} | {label} | 0 | - | - | - | - | - |")
        lines.append("")

    lines += [
        "## 2. Chosen thresholds (dev-selected, test-confirmed)",
        "",
        "| mode | threshold | dev retention | dev unans. abstention | dev J | "
        "test retention | test unans. abstention | test J |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for mode, cal in calibration.items():
        d = cal["dev"]["at_chosen"]
        t = cal["test"]["at_chosen"]
        lines.append(
            f"| {mode} | {cal['chosen_threshold']:.4f} | "
            f"{d['answerable_retention']:.3f} ({d['answerable_answered']}/{d['answerable_total']}) | "
            f"{d['unanswerable_abstention']:.3f} ({d['unanswerable_abstained']}/{d['unanswerable_total']}) | "
            f"{d['youden_j']:.3f} | "
            f"{t['answerable_retention']:.3f} ({t['answerable_answered']}/{t['answerable_total']}) | "
            f"{t['unanswerable_abstention']:.3f} ({t['unanswerable_abstained']}/{t['unanswerable_total']}) | "
            f"{t['youden_j']:.3f} |"
        )
    lines.append("")

    lines += [
        "## 3. Failure analysis (answerable questions, top-5)",
        "",
        "| mode | answerable | full-recall hits | partial overlap | no-overlap misses | "
        "unanswerable false-answer risk |",
        "|---|---|---|---|---|---|",
    ]
    for mode, f in failures.items():
        risk = f["unanswerable_false_answer_risk"]
        lines.append(
            f"| {mode} | {f['answerable']} | {f['full_recall_hits']} | "
            f"{f['partial_overlap']} | {f['no_overlap_misses']} | "
            f"{risk['count']}/{risk['total']} |"
        )
    lines.append("")

    for mode, f in failures.items():
        lines.append(f"### {mode} misses by category")
        lines.append("")
        lines.append("| category | misses | of which threshold would abstain |")
        lines.append("|---|---|---|")
        for cat, counts in f["misses_by_category"].items():
            lines.append(
                f"| {cat} | {counts['misses']} | {counts['of_which_would_abstain']} |"
            )
        lines.append("")
        ex = f["miss_examples"]
        lines.append(
            "- Would answer but returned wrong chunks: "
            + (", ".join(ex["would_answer_but_wrong_chunks"]) or "none")
        )
        lines.append(
            "- Would abstain under calibrated threshold: "
            + (", ".join(ex["would_abstain"]) or "none")
        )
        risk = f["unanswerable_false_answer_risk"]
        lines.append(
            "- Unanswerable questions still answered (false-answer risk): "
            + (", ".join(risk["examples"]) or "none")
        )
        lines.append("")

    lines += [
        "## 4. Interpretation",
        "",
        "- The API applies a separate configured threshold per retrieval mode: "
        "`DENSE_EVIDENCE_THRESHOLD`, `BM25_EVIDENCE_THRESHOLD`, and "
        "`HYBRID_EVIDENCE_THRESHOLD`. This is necessary because cosine, BM25, "
        "and RRF scores have different scales.",
        "- Compare these dev-selected values with the active `.env` settings "
        "after changing the corpus, embedding model, or retrieval configuration.",
        "- See `score_distributions.png` for the answerable vs unanswerable top-1 "
        "score histograms with the chosen cut-off.",
        "",
    ]
    return "\n".join(lines)


def plot_distributions(
    calibration: dict[str, Any], rows_by_mode: dict[str, list[dict[str, Any]]], out_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = list(calibration)
    fig, axes = plt.subplots(1, len(modes), figsize=(5.2 * len(modes), 4.0), squeeze=False)
    for ax, mode in zip(axes[0], modes):
        dev = [r for r in rows_by_mode[mode] if r.get("split") == "dev"]
        ans = [top1_score(r) for r in dev if r["answerable_from_corpus"]]
        unans = [top1_score(r) for r in dev if not r["answerable_from_corpus"]]
        ans = [v for v in ans if v is not None]
        unans = [v for v in unans if v is not None]
        bins = 12
        ax.hist(ans, bins=bins, alpha=0.6, label=f"answerable (n={len(ans)})", color="#2b6cb0")
        ax.hist(
            unans, bins=bins, alpha=0.6, label=f"unanswerable (n={len(unans)})", color="#c53030"
        )
        threshold = calibration[mode]["chosen_threshold"]
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2, label=f"threshold={threshold:.4f}")
        ax.set_title(f"{mode} top-1 score (dev)")
        ax.set_xlabel("top-1 score")
        ax.set_ylabel("questions")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--modes", default="dense,bm25,hybrid")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else RESULTS_DIR / str(args.run_id)
    if not run_dir.exists():
        raise SystemExit(f"run directory not found: {run_dir}")
    run_id = run_dir.name
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    dataset_path = Path(manifest["dataset"]["path"])

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    calibration: dict[str, Any] = {}
    failures: dict[str, Any] = {}
    rows_by_mode: dict[str, list[dict[str, Any]]] = {}

    for mode in modes:
        raw_path = run_dir / f"raw_{mode}.jsonl"
        if not raw_path.exists():
            print(f"skip {mode}: {raw_path} missing")
            continue
        rows = load_raw(raw_path)
        rows_by_mode[mode] = rows
        cal = calibrate_mode(mode, rows)
        calibration[mode] = cal
        failures[mode] = failure_analysis(mode, rows, cal["chosen_threshold"])
        print(
            f"{mode}: chosen threshold {cal['chosen_threshold']:.4f} "
            f"(dev J={cal['dev']['at_chosen']['youden_j']:.3f}, "
            f"test J={cal['test']['at_chosen']['youden_j']:.3f})"
        )

    payload = {
        "run_id": run_id,
        "dataset": str(dataset_path),
        "method": "youden_j_on_dev",
        "calibration": calibration,
        "failures": failures,
    }
    (run_dir / "threshold_calibration.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (run_dir / "threshold_calibration.md").write_text(
        render_md(run_id, calibration, failures, dataset_path), encoding="utf-8"
    )
    if rows_by_mode:
        plot_distributions(
            calibration, rows_by_mode, run_dir / "score_distributions.png"
        )
    print(f"Wrote calibration to {run_dir}")


if __name__ == "__main__":
    main()
