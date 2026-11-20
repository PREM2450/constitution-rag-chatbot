"""Run the Task 6 retrieval evaluation and write raw outputs + metrics.

Examples:
    python experiments/run_evaluation.py
    python experiments/run_evaluation.py --modes bm25 --split dev
    python experiments/run_evaluation.py --modes dense,bm25,hybrid --limit 20
    python experiments/run_evaluation.py --config experiments/configs/eval_defaults.json

For each mode the script:
1. builds the retriever (missing corpus/backend -> recorded error, no metrics),
2. runs every selected question and captures ranked chunk IDs, scores, latency,
3. computes Recall@k / MRR / context precision / citation accuracy,
4. writes ``manifest.json``, ``raw_<mode>.jsonl`` and ``results.md`` under
   ``experiments/results/<run_id>/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluate import (  # noqa: E402
    build_manifest,
    evaluate_mode,
    format_results_md,
    run_evaluation,
)
from src.retriever import RetrieverUnavailableError, build_retriever  # noqa: E402

DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "eval_defaults.json"


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_questions(path: Path, split: str, limit: int | None) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if split != "all":
        rows = [r for r in rows if r["split"] == split]
    if limit:
        rows = rows[:limit]
    return rows


def build_retrievers(config: dict, modes: list[str]) -> tuple[dict, dict]:
    """Build requested retrievers; return (available, errors) keyed by mode."""
    embeddings_path = ROOT / config["embeddings_path"]
    chunks_path = ROOT / config["chunks_path"]
    chunks_to_use = chunks_path if chunks_path.exists() else embeddings_path

    available: dict = {}
    errors: dict = {}
    for mode in modes:
        if mode == "none":
            available[mode] = None
            continue
        try:
            available[mode] = build_retriever(
                mode,
                embeddings_path=embeddings_path,
                chunks_path=chunks_to_use,
                model_name=config["embedding_model"],
                top_k=int(config["top_k"]),
                fusion=config["hybrid_fusion"],
                rrf_k=int(config["hybrid_rrf_k"]),
                dense_weight=float(config["hybrid_dense_weight"]),
                bm25_weight=float(config["hybrid_bm25_weight"]),
            )
        except RetrieverUnavailableError as exc:
            errors[mode] = f"{exc.reason}: {exc.detail}"
        except ValueError as exc:
            errors[mode] = str(exc)
    return available, errors


def save_run(
    output_dir: Path,
    run_id: str,
    manifest: dict,
    results: dict,
    errors: dict,
) -> Path:
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    for mode, result in results.items():
        if not result.questions:
            continue
        raw_path = run_dir / f"raw_{mode}.jsonl"
        with raw_path.open("w", encoding="utf-8") as handle:
            for question in result.questions:
                handle.write(json.dumps(question.to_dict(), ensure_ascii=False) + "\n")

    manifest["mode_errors"] = errors
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "results.md").write_text(format_results_md(results, manifest), encoding="utf-8")
    return run_dir


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--modes", default=None, help="comma-separated: dense,bm25,hybrid,none")
    parser.add_argument("--split", default=None, choices=["dev", "test", "all"])
    parser.add_argument("--limit", type=int, default=None, help="only first N questions")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--no-save", action="store_true", help="print only, do not write files")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    modes = args.modes.split(",") if args.modes else list(config["modes"])
    split = args.split or config["split"]
    output_dir = ROOT / (args.output_dir or config["output_dir"])
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    dataset_path = ROOT / config["dataset_path"]
    chunks_path = ROOT / config["chunks_path"]
    if not dataset_path.exists():
        raise SystemExit(f"dataset not found: {dataset_path}")

    questions = load_questions(dataset_path, split, args.limit)
    print(f"run_id={run_id} split={split} questions={len(questions)} modes={modes}")

    available, errors = build_retrievers(config, modes)
    for mode, message in errors.items():
        print(f"  [skip] {mode}: {message}")

    results = run_evaluation(questions, available, top_k=int(config["top_k"]))
    for mode, message in errors.items():
        from src.evaluate import ModeResult

        results[mode] = ModeResult(mode=mode, top_k=int(config["top_k"]), errors=[message])

    manifest = build_manifest(
        run_id=run_id,
        config=config,
        dataset_path=dataset_path,
        chunks_path=chunks_path,
        results=results,
        ragas_enabled=False,
        ragas_note=(
            "RAGAS not run: no LLM generator configured yet. "
            "Requires `pip install -r requirements-eval.txt` and an LLM provider."
        ),
    )

    print()
    print(format_results_md(results, manifest))

    if args.no_save:
        return
    run_dir = save_run(output_dir, run_id, manifest, results, errors)
    print(f"\nSaved: {run_dir}")


if __name__ == "__main__":
    main()
