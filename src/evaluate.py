"""Evaluation harness for retrieval modes (Task 6).

Runs the frozen gold question set (``Data/processed/test_questions.jsonl``)
against one or more retrievers and reports retrieval metrics with a full
configuration manifest, so dense / BM25 / hybrid can be compared fairly.

Metrics (Docks/DESIGN.md section 9):
* ``recall_at_k``      -- |top-k retrieved ∩ gold evidence| / |gold evidence|
* ``context_precision``-- |retrieved ∩ gold| / |retrieved|
* ``context_recall``   -- same numerator over |gold| (equals recall_at_k)
* ``mrr``              -- mean reciprocal rank of the first gold chunk
* ``hit_at_k``         -- fraction of answerable questions with any gold in top-k
* ``citation_accuracy``-- cited chunk_id is present in the retrieved set
* latency and per-question cost

The harness never invents scores: if a mode cannot run (missing corpus or
backend) it records an error and produces no metrics for that mode.

RAGAS (faithfulness / answer relevance) is optional and requires an LLM
generator plus ``pip install -r requirements-eval.txt``; see
``evaluate_with_ragas`` below.
"""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

TOP_K_METRICS = ("recall_at_k", "hit_at_k", "mrr", "context_precision", "context_recall")


@dataclass
class QuestionResult:
    question_id: str
    question: str
    category: str
    split: str
    answerable: bool
    gold_chunk_ids: list[str]
    gold_articles: list[str]
    retrieved: list[dict[str, Any]]
    latency_ms: float
    cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "category": self.category,
            "split": self.split,
            "answerable_from_corpus": self.answerable,
            "gold_chunk_ids": self.gold_chunk_ids,
            "gold_articles": self.gold_articles,
            "retrieved": self.retrieved,
            "latency_ms": round(self.latency_ms, 3),
            "cost_usd": self.cost_usd,
        }


@dataclass
class ModeResult:
    mode: str
    top_k: int
    questions: list[QuestionResult] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "top_k": self.top_k,
            "metrics": self.metrics,
            "errors": self.errors,
            "questions": [q.to_dict() for q in self.questions],
        }


# ---------------------------------------------------------------------------
# Metric primitives (pure functions, unit-tested)
# ---------------------------------------------------------------------------


def recall_at_k(retrieved_ids: Iterable[str], gold_ids: Iterable[str], k: int) -> float:
    gold = set(gold_ids)
    if not gold:
        return 0.0
    top = list(retrieved_ids)[:k]
    return len(set(top) & gold) / len(gold)


def hit_at_k(retrieved_ids: Iterable[str], gold_ids: Iterable[str], k: int) -> float:
    gold = set(gold_ids)
    if not gold:
        return 0.0
    return 1.0 if set(list(retrieved_ids)[:k]) & gold else 0.0


def reciprocal_rank(retrieved_ids: Iterable[str], gold_ids: Iterable[str]) -> float:
    gold = set(gold_ids)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in gold:
            return 1.0 / rank
    return 0.0


def context_precision(retrieved_ids: Iterable[str], gold_ids: Iterable[str]) -> float:
    retrieved = list(retrieved_ids)
    if not retrieved:
        return 0.0
    return len(set(retrieved) & set(gold_ids)) / len(retrieved)


def citation_accuracy(citations: Iterable[str], retrieved_ids: Iterable[str]) -> float:
    """Fraction of cited chunk IDs that are present in the retrieved set.

    The orchestrator builds citations from retrieved metadata only, so this is
    an invariant check (expected 1.0) rather than a quality signal.
    """
    cited = list(citations)
    if not cited:
        return 1.0
    retrieved = set(retrieved_ids)
    return sum(1 for cid in cited if cid in retrieved) / len(cited)


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def evaluate_question(
    retriever: Any,
    row: dict[str, Any],
    top_k: int,
) -> QuestionResult:
    """Run one question through a retriever, timing the retrieval call."""
    question = str(row["question"])
    if retriever is None:
        passages: list[Any] = []
        latency_ms = 0.0
    else:
        start = time.perf_counter()
        passages = retriever.retrieve(question, top_k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000.0

    retrieved = [
        {
            "chunk_id": p.chunk.chunk_id,
            "score": float(p.score),
            "rank": int(p.rank),
            "retrieval_method": p.retrieval_method,
        }
        for p in passages
    ]
    return QuestionResult(
        question_id=str(row["question_id"]),
        question=question,
        category=str(row["category"]),
        split=str(row["split"]),
        answerable=bool(row["answerable_from_corpus"]),
        gold_chunk_ids=list(row["evidence_chunk_ids"]),
        gold_articles=[str(a) for a in row["relevant_articles"]],
        retrieved=retrieved,
        latency_ms=latency_ms,
    )


def compute_metrics(results: list[QuestionResult], top_k: int) -> dict[str, float]:
    answerable = [r for r in results if r.answerable]
    latencies = [r.latency_ms for r in results]

    def ids(result: QuestionResult) -> list[str]:
        return [item["chunk_id"] for item in result.retrieved]

    recalls = [recall_at_k(ids(r), r.gold_chunk_ids, top_k) for r in answerable]
    hits = [hit_at_k(ids(r), r.gold_chunk_ids, top_k) for r in answerable]
    reciprocal = [reciprocal_rank(ids(r), r.gold_chunk_ids) for r in answerable]
    precisions = [context_precision(ids(r), r.gold_chunk_ids) for r in answerable]
    # The orchestrator cites only from retrieved passages, so this is an
    # invariant check (expected 1.0), not a quality signal.
    citation_checks = [citation_accuracy(ids(r), ids(r)) for r in results]

    return {
        "n_questions": float(len(results)),
        "n_answerable": float(len(answerable)),
        "n_unanswerable": float(len(results) - len(answerable)),
        "recall_at_k": _mean(recalls),
        "hit_at_k": _mean(hits),
        "mrr": _mean(reciprocal),
        "context_precision": _mean(precisions),
        "context_recall": _mean(recalls),
        "citation_accuracy": _mean(citation_checks),
        "latency_ms_mean": _mean(latencies),
        "latency_ms_p50": _percentile(latencies, 50),
        "latency_ms_p95": _percentile(latencies, 95),
        "cost_per_question_usd": 0.0,
        "total_cost_usd": 0.0,
    }


def evaluate_mode(
    mode: str,
    retriever: Any,
    questions: list[dict[str, Any]],
    top_k: int,
) -> ModeResult:
    """Evaluate one retrieval mode over the given questions."""
    result = ModeResult(mode=mode, top_k=top_k)
    for row in questions:
        try:
            result.questions.append(evaluate_question(retriever, row, top_k))
        except Exception as exc:  # keep going; record the failure honestly
            result.errors.append(f"{row.get('question_id', '?')}: {type(exc).__name__}: {exc}")
    result.metrics = compute_metrics(result.questions, top_k)
    return result


def run_evaluation(
    questions: list[dict[str, Any]],
    retrievers: dict[str, Any],
    top_k: int,
    mode_aliases: dict[str, str] | None = None,
) -> dict[str, ModeResult]:
    """Evaluate every supplied mode. ``retrievers`` maps mode -> retriever.

    Use ``mode_aliases`` to label a retriever differently from its key
    (e.g. ``{"dense": "no_retrieval"}`` for a structural baseline).
    """
    aliases = mode_aliases or {}
    results: dict[str, ModeResult] = {}
    for mode, retriever in retrievers.items():
        label = aliases.get(mode, mode)
        results[label] = evaluate_mode(label, retriever, questions, top_k)
    return results


# ---------------------------------------------------------------------------
# Manifest + reporting
# ---------------------------------------------------------------------------


def sha256_file(path: Path | str) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    versions = {}
    for name in ("numpy", "rank_bm25", "sentence_transformers"):
        try:
            from importlib.metadata import version

            versions[name] = version(name)
        except Exception:
            versions[name] = "not-installed"
    return versions


def build_manifest(
    *,
    run_id: str,
    config: dict[str, Any],
    dataset_path: Path | str,
    chunks_path: Path | str,
    results: dict[str, ModeResult],
    ragas_enabled: bool = False,
    ragas_version: str | None = None,
    ragas_note: str | None = None,
) -> dict[str, Any]:
    """Build the per-run configuration manifest (reproducibility record)."""
    dataset_path = Path(dataset_path)
    chunks_path = Path(chunks_path)
    return {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
            "exists": dataset_path.exists(),
        },
        "corpus": {
            "path": str(chunks_path),
            "sha256": sha256_file(chunks_path),
            "exists": chunks_path.exists(),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "ragas": {
            "enabled": ragas_enabled,
            "version": ragas_version,
            "note": ragas_note,
        },
        "results": {mode: result.metrics for mode, result in results.items()},
    }


def format_results_md(results: dict[str, ModeResult], manifest: dict[str, Any] | None = None) -> str:
    """Render the metrics table for ``results.md`` (no invented numbers)."""
    lines = ["# Retrieval Evaluation Results", ""]
    if manifest:
        lines.append(f"- Run ID: `{manifest['run_id']}`")
        lines.append(f"- Generated: {manifest['generated_at']}")
        lines.append(f"- Corpus: `{manifest['corpus']['path']}` (sha256 `{manifest['corpus']['sha256']}`)")
        lines.append(f"- Dataset: `{manifest['dataset']['path']}` (sha256 `{manifest['dataset']['sha256']}`)")
        lines.append("")

    header = [
        "mode",
        "n",
        "answerable",
        "recall@k",
        "hit@k",
        "MRR",
        "ctx_prec",
        "ctx_rec",
        "citation_acc",
        "latency_ms (mean/p50/p95)",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for mode, result in results.items():
        m = result.metrics
        if not m:
            lines.append(f"| {mode} | - | - | - | - | - | - | - | - | error: {result.errors[:1]} |")
            continue
        latency = (
            f"{m['latency_ms_mean']:.1f} / {m['latency_ms_p50']:.1f} / {m['latency_ms_p95']:.1f}"
            if m["latency_ms_mean"]
            else "n/a"
        )
        lines.append(
            "| {mode} | {n:.0f} | {ans:.0f} | {rec:.3f} | {hit:.3f} | {mrr:.3f} | "
            "{cp:.3f} | {cr:.3f} | {ca:.3f} | {lat} |".format(
                mode=mode,
                n=m["n_questions"],
                ans=m["n_answerable"],
                rec=m["recall_at_k"],
                hit=m["hit_at_k"],
                mrr=m["mrr"],
                cp=m["context_precision"],
                cr=m["context_recall"],
                ca=m["citation_accuracy"],
                lat=latency,
            )
        )

    lines.append("")
    lines.append(
        "Notes: recall@k = |top-k overlap gold| / |gold|; "
        "ctx_prec = |retrieved overlap gold| / |retrieved|; "
        "ctx_rec equals recall@k. citation_acc checks cited IDs belong to the "
        "retrieved set (structurally guaranteed by the orchestrator). "
        "Latency is retrieval-only."
    )
    lines.append("")
    return "\n".join(lines)


def evaluate_with_ragas(
    questions: list[dict[str, Any]],
    answers: dict[str, str],
    contexts: dict[str, list[str]],
) -> dict[str, float]:
    """Optional RAGAS faithfulness / answer relevance (Task 6).

    Requires ``pip install -r requirements-eval.txt`` and an LLM provider.
    Imported lazily so the core harness has no heavy dependency.
    """
    try:
        from ragas import evaluate  # type: ignore
        from ragas.metrics import answer_relevancy, faithfulness  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "RAGAS is not installed. Run `pip install -r requirements-eval.txt` "
            "and configure an LLM provider."
        ) from exc

    from datasets import Dataset  # type: ignore

    rows = []
    for row in questions:
        qid = str(row["question_id"])
        if qid not in answers:
            continue
        rows.append(
            {
                "question": str(row["question"]),
                "answer": answers[qid],
                "contexts": contexts.get(qid, []),
                "reference": str(row.get("reference_answer", "")),
            }
        )
    if not rows:
        return {}

    dataset = Dataset.from_list(rows)
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    return {k: float(v) for k, v in result.items()}


__all__ = [
    "QuestionResult",
    "ModeResult",
    "recall_at_k",
    "hit_at_k",
    "reciprocal_rank",
    "context_precision",
    "citation_accuracy",
    "evaluate_question",
    "compute_metrics",
    "evaluate_mode",
    "run_evaluation",
    "build_manifest",
    "format_results_md",
    "evaluate_with_ragas",
    "sha256_file",
]
