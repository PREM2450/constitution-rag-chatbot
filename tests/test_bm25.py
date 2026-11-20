"""BM25 retriever tests.

Deterministic, no external model: tiny in-memory corpus where exact
keyword/Article-number queries must rank the matching chunk first.
Same corpus records as the dense fake-embedder tests.
"""

import unittest
from pathlib import Path

from src.retriever import BM25Retriever, RetrieverUnavailableError, build_bm25_retriever, tokenize

CHUNKS = [
    {
        "chunk_id": "c-1",
        "text": "Article 21 protects life and personal liberty",
        "page_start": 9,
        "article": "Article 21",
        "source_title": "Constitution of India",
        "source_url": "https://example.invalid/constitution",
    },
    {
        "chunk_id": "c-2",
        "text": "Article 19 guarantees freedom of speech and expression",
        "page_start": 8,
        "article": "Article 19",
        "source_title": "Constitution of India",
        "source_url": "https://example.invalid/constitution",
    },
    {
        "chunk_id": "c-3",
        "text": "directive principles state policy well-being arrangement",
        "page_start": 40,
        "article": None,
        "source_title": "Constitution of India",
        "source_url": "",
    },
]


def _build_retriever(chunks=CHUNKS, top_k=5):
    from rank_bm25 import BM25Okapi

    tokenized = [tokenize(chunk["text"]) for chunk in chunks]
    index = BM25Okapi(tokenized)
    return BM25Retriever(chunks=[dict(c) for c in chunks], index=index, top_k=top_k)


class TokenizeTests(unittest.TestCase):
    def test_preserves_article_numbers(self):
        self.assertEqual(tokenize("Article 21A: Right to Education!"), ["article", "21a", "right", "to", "education"])
        self.assertEqual(tokenize("ARTICLE 21"), ["article", "21"])

    def test_lowercase_and_punctuation(self):
        self.assertEqual(tokenize("Life, Liberty? PERSONAL"), ["life", "liberty", "personal"])

    def test_empty(self):
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize("  "), [])


class BM25RetrieverTests(unittest.TestCase):
    def test_status(self):
        retriever = _build_retriever()
        status = retriever.status()
        self.assertEqual(status["kind"], "bm25")
        self.assertTrue(status["ready"])
        self.assertEqual(status["chunks"], 3)
        self.assertEqual(status["top_k"], 5)
        self.assertEqual(status["tokenizer"], "alnum_lower")

    def test_exact_keyword_ranks_match_first(self):
        retriever = _build_retriever()
        passages = retriever.retrieve("What does Article 21 protect life liberty?")
        self.assertEqual(passages[0].chunk.chunk_id, "c-1")
        self.assertEqual(passages[0].retrieval_method, "bm25")
        self.assertEqual(passages[0].rank, 1)

    def test_article_number_exact_match(self):
        retriever = _build_retriever()
        passages = retriever.retrieve("Article 19")
        self.assertEqual(passages[0].chunk.chunk_id, "c-2")

    def test_empty_query_returns_empty(self):
        retriever = _build_retriever()
        self.assertEqual(retriever.retrieve(""), [])
        self.assertEqual(retriever.retrieve("   "), [])
        self.assertEqual(retriever.retrieve("!!!"), [])

    def test_metadata_preserved_and_null_article(self):
        retriever = _build_retriever()
        passages = retriever.retrieve("directive principles state policy")
        self.assertEqual(passages[0].chunk.chunk_id, "c-3")
        self.assertIsNone(passages[0].chunk.article)
        self.assertEqual(passages[0].chunk.page, 40)
        self.assertEqual(passages[0].chunk.source_title, "Constitution of India")

    def test_top_k_respected(self):
        retriever = _build_retriever(top_k=2)
        passages = retriever.retrieve("Article protects freedom")
        self.assertEqual(len(passages), 2)
        passages = retriever.retrieve("Article protects freedom", top_k=1)
        self.assertEqual(len(passages), 1)


class BuilderTests(unittest.TestCase):
    def test_missing_corpus_raises_unavailable(self):
        with self.assertRaises(RetrieverUnavailableError) as ctx:
            build_bm25_retriever(Path("does/not/exist.jsonl"), top_k=5)
        self.assertEqual(ctx.exception.reason, "index_not_found")


if __name__ == "__main__":
    unittest.main()
