"""Interactive human-review tool for the Task 5 gold dataset.

Shows each test question next to its cited evidence chunks so a reviewer can
verify the row against the real corpus text, then marks it verified. Progress
is saved after every action, so you can quit and resume anytime.

Usage:
    python review_dataset.py                          # review pending rows
    python review_dataset.py --reviewer Satish
    python review_dataset.py --category ambiguous_question
    python review_dataset.py --split dev
    python review_dataset.py --dump review_sheet.md   # static Markdown instead

Keys during review:
    v = verify   d = revert to draft   s = skip   b = back   q = quit   ? = help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "Data" / "processed" / "test_questions.jsonl"
CHUNKS = ROOT / "Data" / "processed" / "constitution_chunks.jsonl"
MANIFEST = ROOT / "Data" / "processed" / "dataset_manifest.md"

DRAFT = "draft_pending_human_review"
EVIDENCE_PREVIEW_CHARS = 700


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {row["chunk_id"]: row for row in load_jsonl(path)}


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def display_row(row: dict, chunks: dict[str, dict], position: int, total: int) -> None:
    print("\n" + "=" * 78)
    print(
        f"[{position}/{total}] {row['question_id']} | {row['category']} | "
        f"{row['split']} | {row['review_status']}"
    )
    print("=" * 78)
    print(f"Q: {row['question']}")
    print(f"\nReference answer:\n{row['reference_answer']}")
    print(f"\nRelevant articles: {row['relevant_articles']}")
    print(f"Evidence chunks:   {row['evidence_chunk_ids']}")
    if row.get("notes"):
        print(f"Notes:             {row['notes']}")

    if row["evidence_chunk_ids"]:
        print("\n--- EVIDENCE ---")
        for cid in row["evidence_chunk_ids"]:
            chunk = chunks.get(cid)
            if chunk is None:
                print(f"\n[{cid}] !! chunk not found in corpus")
                continue
            page = chunk.get("page_start", "?")
            print(f"\n[{cid}] (page {page}):")
            print(_truncate(chunk.get("text", ""), EVIDENCE_PREVIEW_CHARS))
    else:
        print("\n--- EVIDENCE --- (none: unanswerable row)")


def prompt() -> str:
    return input("\n[v]erify  [d]raft  [s]kip  [b]ack  [q]uit  [?]help > ").strip().lower()


def show_help() -> None:
    print(
        """
v = verify   : mark this row verified_<reviewer> (accept answer + evidence)
d = draft    : set review_status back to draft_pending_human_review
s = skip     : leave as-is and move to the next row
b = back     : go back one row
q = quit     : save and exit (resume later; verified rows are not shown again)
? = help     : show this message

Check before verifying:
  1. Is the question answerable from the cited evidence text?
  2. Is the reference answer accurate and free of outside law?
  3. Do relevant_articles match the Article(s) actually used?
  4. Do evidence_chunk_ids point to the supporting chunk(s)?
"""
    )


def update_manifest(reviewer: str, verified: int, total: int) -> None:
    if not MANIFEST.exists():
        return
    text = MANIFEST.read_text(encoding="utf-8")
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("Status:"):
            if verified == total:
                lines[i] = (
                    f"Status: **verified** by {reviewer} — all {total} rows "
                    "human-verified against the official text."
                )
            else:
                lines[i] = (
                    f"Status: **partially reviewed** — {verified}/{total} rows "
                    f"verified by {reviewer}."
                )
            break
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_review(args: argparse.Namespace) -> None:
    if not DATASET.exists():
        raise SystemExit(f"dataset not found: {DATASET}")
    all_rows = load_jsonl(DATASET)
    chunks = load_chunks(CHUNKS)

    reviewer = args.reviewer
    if not reviewer:
        reviewer = input("Your name (used as verified_<name>): ").strip() or "anonymous"

    # Start from all rows, then narrow by review status, category, and split.
    queue = list(range(len(all_rows)))
    if args.status == "pending":
        queue = [i for i in queue if all_rows[i].get("review_status") == DRAFT]
    elif args.status == "verified":
        queue = [i for i in queue if str(all_rows[i].get("review_status", "")).startswith("verified_")]
    if args.category:
        queue = [i for i in queue if all_rows[i]["category"] == args.category]
    if args.split:
        queue = [i for i in queue if all_rows[i]["split"] == args.split]

    if not queue:
        print("No rows match the current filter.")
        return

    print(f"\nReviewing {len(queue)} row(s). Reviewer: {reviewer}")
    print("Press ? for the checklist.\n")

    pos = 0
    while pos < len(queue):
        row = all_rows[queue[pos]]
        display_row(row, chunks, pos + 1, len(queue))
        choice = prompt()

        if choice == "v":
            row["review_status"] = f"verified_{reviewer}"
            dump_jsonl(DATASET, all_rows)
            pos += 1
        elif choice == "d":
            row["review_status"] = DRAFT
            dump_jsonl(DATASET, all_rows)
            pos += 1
        elif choice == "s":
            pos += 1
        elif choice == "b":
            pos = max(0, pos - 1)
        elif choice == "q":
            break
        elif choice == "?":
            show_help()
        else:
            print("unknown key — press ? for help")

    verified = sum(1 for r in all_rows if str(r.get("review_status", "")).startswith("verified_"))
    pending = sum(1 for r in all_rows if r.get("review_status") == DRAFT)
    print(f"\n{'=' * 78}")
    print(f"Progress: {verified} verified, {pending} pending, {len(all_rows)} total")
    if verified:
        update_manifest(reviewer, verified, len(all_rows))
        if verified == len(all_rows):
            print("All rows verified — manifest updated.")


def run_dump(args: argparse.Namespace) -> None:
    rows = load_jsonl(DATASET)
    chunks = load_chunks(CHUNKS)
    if args.category:
        rows = [r for r in rows if r["category"] == args.category]

    out = ["# Task 5 Review Sheet — test_questions.jsonl", ""]
    out.append(f"Rows: {len(rows)}. Evidence source: `constitution_chunks.jsonl`.")
    out.append("")
    out.append(
        "Check per row: (1) answerable from evidence, (2) reference answer accurate, "
        "(3) relevant_articles match, (4) evidence_chunk_ids correct."
    )
    out.append("")

    for i, row in enumerate(rows, start=1):
        out.append(f"## {i}. {row['question_id']} — {row['category']} ({row['split']})")
        out.append("")
        out.append(f"**Q:** {row['question']}")
        out.append("")
        out.append(f"**Reference answer:** {row['reference_answer']}")
        out.append("")
        out.append(f"**Relevant articles:** {row['relevant_articles']}")
        out.append(f"**Evidence chunks:** {row['evidence_chunk_ids']}")
        if row.get("notes"):
            out.append(f"**Notes:** {row['notes']}")
        out.append(f"**Status:** `{row['review_status']}`")
        out.append("")
        if row["evidence_chunk_ids"]:
            for cid in row["evidence_chunk_ids"]:
                chunk = chunks.get(cid)
                if chunk is None:
                    out.append(f"> **{cid}** — NOT FOUND")
                    continue
                page = chunk.get("page_start", "?")
                out.append(f"> **{cid}** (page {page}):")
                out.append(">")
                for line in chunk.get("text", "").splitlines():
                    out.append(f"> {line}")
        else:
            out.append("> _(no evidence — unanswerable row)_")
        out.append("")
        out.append("---")
        out.append("")

    Path(args.dump).write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {args.dump} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer", default=None, help="name used as verified_<name>")
    parser.add_argument("--category", default=None, help="only review this category")
    parser.add_argument("--split", default=None, help="only review dev|test")
    parser.add_argument(
        "--status",
        default="pending",
        choices=["pending", "verified", "all"],
        help="which rows to show (default: pending)",
    )
    parser.add_argument("--dump", default=None, metavar="MD_PATH", help="write a static Markdown review sheet instead")
    args = parser.parse_args()

    if args.dump:
        run_dump(args)
    else:
        run_review(args)


if __name__ == "__main__":
    main()
