"""Shared retrieval primitives: corpus loading, chunk mapping, errors.

Used by the dense, BM25, and hybrid retrievers so they all read the same
corpus records and produce the same ``RetrievedPassage`` shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.schemas import DocumentChunk

DEFAULT_EMBEDDINGS_PATH = Path("Data/processed/constitution_embeddings.jsonl")
DEFAULT_CHUNKS_PATH = Path("Data/processed/constitution_chunks.jsonl")

# Standard Reciprocal Rank Fusion constant (Cormack et al.). RRF-only in
# Task 3; score-normalized weighted fusion is deferred until evaluation
# (Tasks 6-7) provides dev-set tuning data.
DEFAULT_RRF_K = 60

# Field names in the corpus records written by Prem's pipeline
# (chunk_text.py / create_embeddings.py).
_TEXT_KEYS = ("text",)
_PAGE_KEYS = ("page_start", "page")
_ARTICLE_KEYS = ("article", "article_number")


class RetrieverUnavailableError(RuntimeError):
    """Raised when the retriever cannot serve queries.

    ``reason`` is a stable machine-readable tag the API maps to HTTP 503 and
    ``detail`` is a human-readable explanation safe to show users.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def _first(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def load_corpus(embeddings_path: Path | str) -> list[dict[str, Any]]:
    """Load chunk records with embeddings from a JSONL file."""
    path = Path(embeddings_path)
    if not path.exists():
        raise RetrieverUnavailableError(
            reason="index_not_found",
            detail=(
                "The indexed Constitution corpus was not found on the server"
                f" ({path}). The offline preprocessing/embedding step has not"
                " been run or the artifacts are missing."
            ),
        )

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RetrieverUnavailableError(
                    reason="corrupt_index",
                    detail=(
                        "The retriever index file is malformed"
                        f" ({path}, line {line_number})."
                    ),
                ) from exc
            if not isinstance(record, dict) or not _first(record, _TEXT_KEYS):
                continue
            records.append(record)

    if not records:
        raise RetrieverUnavailableError(
            reason="empty_index",
            detail="The retriever index file contains no usable chunks.",
        )
    return records


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


def _to_chunk(record: dict[str, Any]) -> DocumentChunk:
    text = _first(record, _TEXT_KEYS)
    page = _first(record, _PAGE_KEYS)
    article = _first(record, _ARTICLE_KEYS)
    if page is not None:
        try:
            page = int(page)
        except (TypeError, ValueError):
            page = None
    return DocumentChunk(
        chunk_id=str(record["chunk_id"]),
        text=str(text),
        article=str(article) if article is not None else None,
        part=record.get("part"),
        section=record.get("section"),
        clause=record.get("clause"),
        page=page,
        source_title=record.get("source_title", "Constitution of India"),
        source_url=record.get("source_url", ""),
        source_version=record.get("source_version", "unknown"),
        metadata=record.get("metadata", {}),
    )
