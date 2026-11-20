"""Map archived human-verified Article labels onto the current corpus chunks."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = ROOT / "experiments/results/20261119_040644/raw_dense.jsonl"
CHUNKS_PATH = ROOT / "Data/processed/constitution_chunks.jsonl"
OUTPUT_PATH = ROOT / "Data/processed/test_questions.jsonl"


def remap_questions(question_rows: list[dict], chunks: list[dict]) -> list[dict]:
    chunks_by_article: dict[str, list[str]] = {}
    for chunk in chunks:
        article_number = chunk.get("article_number")
        if article_number is not None:
            chunks_by_article.setdefault(str(article_number), []).append(
                str(chunk["chunk_id"])
            )

    remapped = []
    for row in question_rows:
        articles = [str(article) for article in row.get("gold_articles", [])]
        evidence = sorted({
            chunk_id
            for article in articles
            for chunk_id in chunks_by_article.get(article, [])
        })
        answerable = bool(row["answerable_from_corpus"])
        if answerable and not evidence:
            raise ValueError(
                f"{row['question_id']} has no current corpus chunks for "
                f"gold Articles {articles}"
            )
        if not answerable and evidence:
            raise ValueError(
                f"{row['question_id']} is marked unanswerable but has gold Articles"
            )
        remapped.append({
            "question_id": str(row["question_id"]),
            "question": str(row["question"]),
            "category": str(row["category"]),
            "split": str(row["split"]),
            "answerable_from_corpus": answerable,
            "gold_articles": articles,
            "relevant_articles": articles,
            "evidence_chunk_ids": evidence,
        })
    return remapped


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    for path in (SOURCE_PATH, CHUNKS_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required input is missing: {path}")
    rows = remap_questions(_read_jsonl(SOURCE_PATH), _read_jsonl(CHUNKS_PATH))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    answerable = sum(row["answerable_from_corpus"] for row in rows)
    print(f"Questions remapped: {len(rows)} ({answerable} answerable)")
    print(f"Output file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
