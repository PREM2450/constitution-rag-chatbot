from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    text: str
    article: str | None = None
    part: str | None = None
    clause: str | None = None
    page: int | None = None
    source_title: str = "Constitution of India"
    source_url: str = ""
    source_version: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "article": self.article,
            "part": self.part,
            "clause": self.clause,
            "page": self.page,
            "source_title": self.source_title,
            "source_url": self.source_url,
            "source_version": self.source_version,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass(frozen=True)
class RetrievedPassage:
    chunk: DocumentChunk
    score: float
    rank: int
    retrieval_method: str


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    article: str | None = None
    page: int | None = None
    source_title: str = "Constitution of India"
    source_url: str = ""


@dataclass(frozen=True)
class AskRequest:
    question: str
    retrieval_mode: str = "dense"

    def __post_init__(self) -> None:
        if not self.question or not self.question.strip():
            raise ValueError("question cannot be empty")
        if self.retrieval_mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError("retrieval_mode must be one of: dense, bm25, hybrid")


@dataclass
class AnswerResponse:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    sources: list[RetrievedPassage] = field(default_factory=list)
    insufficient_evidence: bool = False
    request_id: str = ""
    retrieval_method: str = "dense"
    warning: str | None = None

    def __post_init__(self) -> None:
        if not self.answer or not self.answer.strip():
            raise ValueError("answer cannot be empty")
        if self.insufficient_evidence and not self.warning:
            object.__setattr__(
                self,
                "warning",
                "Insufficient evidence found for this question.",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [
                {
                    "chunk_id": citation.chunk_id,
                    "article": citation.article,
                    "page": citation.page,
                    "source_title": citation.source_title,
                    "source_url": citation.source_url,
                }
                for citation in self.citations
            ],
            "sources": [
                {
                    "chunk": source.chunk.to_dict(),
                    "score": source.score,
                    "rank": source.rank,
                    "retrieval_method": source.retrieval_method,
                }
                for source in self.sources
            ],
            "insufficient_evidence": self.insufficient_evidence,
            "request_id": self.request_id,
            "retrieval_method": self.retrieval_method,
            "warning": self.warning,
        }
