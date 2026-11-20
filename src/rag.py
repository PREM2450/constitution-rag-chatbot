"""RAG orchestrator: connects retrieval, evidence checks and generation.

Pipeline (matches Docs/ARCHITECTURE.md, section 4):

    POST /ask -> validate -> orchestrator.answer(question)
                -> retriever.retrieve()            (ranked passages)
                -> evidence-sufficiency check      (top score vs threshold)
                -> generator.generate()            (replaceable seam)
                -> citations built from retrieved metadata ONLY
                -> AnswerResponse

The orchestrator never fabricates content: on weak evidence it abstains
instead of guessing, and citations are resolved from the retrieved chunks,
never from generator free text.
"""

from __future__ import annotations

import uuid

from src.generation import (
    ExtractiveBaselineGenerator,
    GenerationService,
    INSUFFICIENT_EVIDENCE_ANSWER,
)
from src.retriever import RetrieverUnavailableError
from src.schemas import AnswerResponse, AskRequest, Citation


VALID_MODES = {"dense", "bm25", "hybrid"}


class RagOrchestrator:
    def __init__(
        self,
        retriever=None,
        generator: GenerationService | None = None,
        evidence_threshold: float = 0.6,
        retrieval_mode: str = "dense",
        retrievers: dict | None = None,
    ) -> None:
        if retriever is not None and retrievers is not None:
            raise ValueError("pass exactly one of retriever / retrievers")
        if retrievers is not None:
            if not retrievers:
                raise ValueError("retrievers must not be empty")
            unknown = set(retrievers) - VALID_MODES
            if unknown:
                raise ValueError(f"unknown retrieval modes: {sorted(unknown)}")
            self._retrievers = dict(retrievers)
            self._retriever = None
        elif retriever is not None:
            self._retrievers = {retrieval_mode: retriever}
            self._retriever = retriever
        else:
            raise ValueError("retriever or retrievers is required")
        self._generator = generator if generator is not None else ExtractiveBaselineGenerator()
        self._evidence_threshold = float(evidence_threshold)
        self._retrieval_mode = retrieval_mode

    def status(self) -> dict:
        retrievers_status = {mode: r.status() for mode, r in self._retrievers.items()}
        primary = self._retrievers.get(self._retrieval_mode)
        if primary is None:
            primary = next(iter(self._retrievers.values()))
        return {
            "orchestrator": "ready",
            "retriever": primary.status(),
            "retrievers": retrievers_status,
            "available_modes": sorted(self._retrievers),
            "generator": {"name": self._generator.name},
            "evidence_threshold": self._evidence_threshold,
        }

    def _select_retriever(self, mode: str):
        retriever = self._retrievers.get(mode)
        if retriever is None:
            available = ", ".join(sorted(self._retrievers))
            raise RetrieverUnavailableError(
                reason="retrieval_mode_unavailable",
                detail=(
                    f"Retrieval mode '{mode}' is not available on this server "
                    f"(available: {available})."
                ),
            )
        return retriever

    def answer(self, request: AskRequest) -> AnswerResponse:
        request_id = uuid.uuid4().hex

        retriever = self._select_retriever(request.retrieval_mode)
        passages = retriever.retrieve(
            request.question,
            top_k=None,  # top_k comes from the retriever's configuration
        )

        if self._has_sufficient_evidence(passages):
            answer = self._generator.generate(request.question, passages)
            insufficient = False
            warning = None
        else:
            answer = INSUFFICIENT_EVIDENCE_ANSWER
            insufficient = True
            warning = "Insufficient evidence found for this question."

        return AnswerResponse(
            answer=answer,
            citations=self._build_citations(passages),
            sources=passages,
            insufficient_evidence=insufficient,
            request_id=request_id,
            retrieval_method=request.retrieval_mode,
            warning=warning,
        )

    def _has_sufficient_evidence(self, passages) -> bool:
        if not passages:
            return False
        # Top-rank score decides sufficiency; the threshold is a project
        # configuration value and must be calibrated on development data
        # (it is not a probability of correctness).
        return passages[0].score >= self._evidence_threshold

    @staticmethod
    def _build_citations(passages):
        # Citations come from retrieved chunk metadata only. If the corpus
        # has no verified Article label, the citation honestly shows null.
        return [
            Citation(
                chunk_id=passage.chunk.chunk_id,
                article=passage.chunk.article,
                page=passage.chunk.page,
                source_title=passage.chunk.source_title,
                source_url=passage.chunk.source_url,
            )
            for passage in passages
        ]


def create_orchestrator(
    *,
    retriever=None,
    retrievers: dict | None = None,
    generator: GenerationService | None = None,
    evidence_threshold: float = 0.6,
    retrieval_mode: str = "dense",
) -> RagOrchestrator:
    return RagOrchestrator(
        retriever=retriever,
        retrievers=retrievers,
        generator=generator,
        evidence_threshold=evidence_threshold,
        retrieval_mode=retrieval_mode,
    )


__all__ = ["RagOrchestrator", "create_orchestrator", "RetrieverUnavailableError"]
