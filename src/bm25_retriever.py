"""BM25 lexical retriever over the same corpus records as Dense.

BM25 is the lexical (keyword) half of the hybrid system: it scores chunks by
term frequency / inverse document frequency, so it is strong on exact Article
numbers and rare keywords where dense embeddings can be vague. It implements
the same ``retrieve(question, top_k)`` interface as ``DenseRetriever`` so the
two can be swapped or fused without changing callers.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from src.retrieval_base import (
    RetrieverUnavailableError,
    _first,
    _TEXT_KEYS,
    _to_chunk,
    load_corpus,
)
from src.schemas import RetrievedPassage

# Token pattern: keep letters and digits together so "21A" stays one token.
_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Matches the word "Article"/"Articles" so we can read the number that follows.
_ARTICLE_PREFIX_RE = re.compile(r"\barticles?\s+", re.IGNORECASE)
# An Article number is 1-3 digits with an optional letter suffix (e.g. 21, 21A).
_ARTICLE_NUMBER_RE = re.compile(r"(\d{1,3}[a-z]?)", re.IGNORECASE)
# Separators that may join several Article numbers ("21, 22 and 23").
_ARTICLE_SEPARATOR_RE = re.compile(r"\s*(?:,|and|&)\s*", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Tokenize for BM25: lowercase alphanumeric tokens only.

    Deliberately preserves numbers so ``Article 21`` -> ``["article", "21"]``
    and ``21A`` -> ``["21a"]``; no stop-word removal — BM25's inverse
    document frequency already down-weights very common terms.
    """
    return _TOKEN_RE.findall((text or "").lower())


def _article_references(question: str) -> set[str]:
    """Extract Article numbers explicitly named in a question.

    Example: ``"Compare Article 21 and 22"`` -> ``{"21", "22"}``.
    Returns an empty set when the question does not mention an Article, in
    which case the caller falls back to plain BM25 ranking.
    """
    references = set()
    for prefix in _ARTICLE_PREFIX_RE.finditer(question):
        cursor = prefix.end()
        # Walk forward through "21, 22 and 23" style lists.
        while True:
            number = _ARTICLE_NUMBER_RE.match(question, cursor)
            if number is None:
                break
            references.add(number.group(1).upper())
            cursor = number.end()
            separator = _ARTICLE_SEPARATOR_RE.match(question, cursor)
            if separator is None:
                break
            cursor = separator.end()
    return references


def _chunk_article_number(record: dict[str, Any]) -> str | None:
    """Return the Article number a chunk belongs to, if the corpus records one.

    Prefers the structured ``article_number`` field; falls back to parsing the
    display ``article`` string (e.g. ``"Article 21"``). Returns ``None`` for
    chunks with no Article metadata rather than guessing.
    """
    number = record.get("article_number")
    if number is not None:
        return str(number).upper()
    match = _ARTICLE_NUMBER_RE.search(str(record.get("article", "")))
    return match.group(1).upper() if match else None


def build_bm25_retriever(
    corpus_path: Path | str,
    top_k: int,
    index: Any | None = None,
) -> "BM25Retriever":
    """Build a BM25 retriever over ``corpus_path``.

    ``corpus_path`` may point to ``constitution_chunks.jsonl`` or
    ``constitution_embeddings.jsonl`` — both carry the same ``text`` field.
    Raises ``RetrieverUnavailableError`` if the corpus is missing/corrupt
    or the ``rank_bm25`` backend is not installed.
    """
    chunks = load_corpus(corpus_path)
    # Index the Article/title label text as well as the body so that queries
    # naming an Article can match the chunk even if the body wording differs.
    tokenized = []
    for record in chunks:
        labels = (
            record.get("article"),
            record.get("title"),
            record.get("section"),
        )
        indexed_text = " ".join(
            [str(value) for value in labels if value]
            + [str(_first(record, _TEXT_KEYS) or "")]
        )
        tokenized.append(tokenize(indexed_text))
    # The backend is imported lazily so the rest of the app still starts when
    # rank_bm25 is absent; tests may also inject a prebuilt ``index``.
    if index is None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RetrieverUnavailableError(
                reason="bm25_unavailable",
                detail=(
                    "The BM25 backend (rank_bm25) is not installed on the "
                    "server, so lexical retrieval cannot run."
                ),
            ) from exc
        index = BM25Okapi(tokenized)
    return BM25Retriever(chunks=chunks, index=index, top_k=top_k)


class BM25Retriever:
    """Lexical BM25 retriever over the same corpus records as Dense."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        index: Any,
        top_k: int,
    ) -> None:
        self._chunks = chunks
        self._index = index
        self._top_k = max(1, int(top_k))

    def status(self) -> dict[str, Any]:
        return {
            "kind": "bm25",
            "ready": True,
            "chunks": len(self._chunks),
            "top_k": self._top_k,
            "tokenizer": "alnum_lower",
            "algorithm": "BM25Okapi",
        }

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedPassage]:
        """Return the top ``top_k`` chunks ranked by BM25 for ``question``.

        When the question explicitly names Article(s), only chunks belonging to
        those Articles are eligible; this prevents a keyword match on unrelated
        text from outranking the provision the user asked about.
        """
        if not question or not question.strip():
            return []
        query_tokens = tokenize(question)
        if not query_tokens:
            return []
        limit = self._top_k if top_k is None else max(1, int(top_k))
        # BM25Okapi returns one raw score per chunk; higher is more relevant.
        # Scores are unbounded (not cosine), so they are only comparable within
        # a single BM25 run — never against dense/hybrid scores.
        scores = np.array(self._index.get_scores(query_tokens), dtype=np.float64)
        requested_articles = _article_references(question)
        if requested_articles and scores.size:
            # Restrict to the Articles the question actually mentions.
            candidate_indices = [
                index
                for index, chunk in enumerate(self._chunks)
                if _chunk_article_number(chunk) in requested_articles
            ]
            if not candidate_indices:
                # Named Article(s) have no matching chunk in this corpus.
                return []
            # Add a constant larger than any BM25 score so every candidate is
            # guaranteed to outrank non-candidates, while preserving the BM25
            # ordering among the candidates themselves.
            article_boost = max(float(scores.max()), 0.0) + 1.0
            scores[candidate_indices] += article_boost
            top_indices = sorted(
                candidate_indices,
                key=lambda index: scores[index],
                reverse=True,
            )[:limit]
        else:
            # No explicit Article mention: plain global BM25 ranking.
            top_indices = np.argsort(scores)[::-1][:limit]
        passages: list[RetrievedPassage] = []
        for rank, index in enumerate(top_indices, start=1):
            passages.append(
                RetrievedPassage(
                    chunk=_to_chunk(self._chunks[index]),
                    score=float(scores[index]),
                    rank=rank,
                    retrieval_method="bm25",
                )
            )
        return passages
