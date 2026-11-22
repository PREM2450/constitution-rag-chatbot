"""BM25 lexical search CLI over the Constitution chunk corpus.

Usage:
    python search_bm25.py                 # interactive (type 'exit' to quit)
    python search_bm25.py --demo          # run 5 sample questions once and exit
    python search_bm25.py "Article 21"    # single query mode

Reads Data/processed/constitution_chunks.jsonl (or embeddings JSONL fallback)
via src.retriever.build_bm25_retriever. No embeddings/model needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.retriever import DEFAULT_CHUNKS_PATH, DEFAULT_EMBEDDINGS_PATH, build_bm25_retriever

SAMPLE_QUERIES = [
    "What does Article 21 protect?",
    "Article 19 freedom of speech and expression",
    "right to education Article 21A",
    "directive principles state policy",
    "emergency provisions Article 356",
]

TOP_K = 5


def _pick_corpus() -> Path:
    """Prefer the plain chunk corpus; fall back to the embeddings JSONL.

    Both files carry the same ``text`` field, so BM25 can use either one.
    """
    if DEFAULT_CHUNKS_PATH.exists():
        return DEFAULT_CHUNKS_PATH
    return DEFAULT_EMBEDDINGS_PATH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BM25 search over Constitution chunks")
    parser.add_argument("query", nargs="?", help="single query to run once")
    parser.add_argument("--demo", action="store_true", help="run 5 sample questions and exit")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args(argv)

    corpus_path = _pick_corpus()
    print(f"Loading BM25 index from {corpus_path} (top_k={args.top_k})...")
    try:
        retriever = build_bm25_retriever(corpus_path, top_k=args.top_k)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"BM25 ready: {retriever.status()}")

    def run_query(question: str) -> None:
        print(f"\n=== Q: {question} ===")
        passages = retriever.retrieve(question)
        if not passages:
            print("(no results)")
            return
        for p in passages:
            c = p.chunk
            # Show only a preview; the full text is in the corpus file.
            print(f"--- Rank {p.rank} | BM25 score {p.score:.2f} | {c.article or 'article unknown'} | p.{c.page} | {c.chunk_id} ---")
            print(c.text[:400].strip())
            print()

    if args.demo:
        for q in SAMPLE_QUERIES:
            run_query(q)
        return 0

    if args.query:
        run_query(args.query)
        return 0

    print("\nConstitution BM25 Lexical Search — type 'exit' to quit.")
    while True:
        try:
            question = input("\nYour question: ").strip()
        except EOFError:
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        run_query(question)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
