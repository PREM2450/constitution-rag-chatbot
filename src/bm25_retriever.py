"""BM25 lexical retriever over the same corpus records as Dense."""

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

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_ARTICLE_PREFIX_RE = re.compile(r"\barticles?\s+", re.IGNORECASE)
_ARTICLE_NUMBER_RE = re.compile(r"(\d{1,3}[a-z]?)", re.IGNORECASE)
_ARTICLE_SEPARATOR_RE = re.compile(r"\s*(?:,|and|&)\s*", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Tokenize for BM25: lowercase alphanumeric tokens only.

    Deliberately preserves numbers so ``Article 21`` -> ``["article", "21"]``
    and ``21A`` -> ``["21a"]``; no stop-word removal — BM25's inverse
    document frequency already down-weights very common terms.
    """
    return _TOKEN_RE.findall((text or "").lower())


def _article_references(question: str) -> set[str]:
    references = set()
    for prefix in _ARTICLE_PREFIX_RE.finditer(question):
        cursor = prefix.end()
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
        if not question or not question.strip():
            return []
        query_tokens = tokenize(question)
        if not query_tokens:
            return []
        limit = self._top_k if top_k is None else max(1, int(top_k))
        scores = np.array(self._index.get_scores(query_tokens), dtype=np.float64)
        requested_articles = _article_references(question)
        if requested_articles and scores.size:
            candidate_indices = [
                index
                for index, chunk in enumerate(self._chunks)
                if _chunk_article_number(chunk) in requested_articles
            ]
            if not candidate_indices:
                return []
            article_boost = max(float(scores.max()), 0.0) + 1.0
            scores[candidate_indices] += article_boost
            top_indices = sorted(
                candidate_indices,
                key=lambda index: scores[index],
                reverse=True,
            )[:limit]
        else:
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
