"""FastAPI application for the Constitution RAG chatbot.

Endpoints:
* GET  /health  -> service and component status (no secrets).
* POST /ask     -> answer a constitutional question with citations and sources.

The app is built through ``create_app()`` so tests (and future integrations
by other team members) can inject a fake retriever/generator. The module
level ``app`` is the production instance used by ``uvicorn``.

Offline retrieval artifacts and the embedding model are optional at import
time: if they are missing the API still starts, reports degraded status on
/health and returns 503 on /ask instead of crashing.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.config import Settings
from src.generation import GenerationError, create_generation_service
from src.rag import RagOrchestrator
from src.retriever import RetrieverUnavailableError
from src.schemas import AskRequest

APP_VERSION = "0.1.0"
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# Documented request limit for a single question (see Docs/DESIGN.md).
MAX_QUESTION_LENGTH = 1000

DEGRADED_REASONS = {
    "index_not_found": "Retrieval index not found on the server.",
    "empty_index": "Retrieval index contains no chunks.",
    "corrupt_index": "Retrieval index is malformed.",
    "embedding_model_unavailable": "Embedding model backend unavailable on the server.",
    "bm25_unavailable": "BM25 backend unavailable on the server.",
    "hybrid_fusion_unavailable": "Requested hybrid fusion is not available on the server.",
    "retrieval_mode_unavailable": "Requested retrieval mode is not available on the server.",
}


def _parse_cors_origins(raw: str | None) -> list[str]:
    values = raw or os.getenv("CORS_ORIGINS", "*")
    return [origin.strip() for origin in values.split(",") if origin.strip()]


def _resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    # Relative settings are resolved against the repository root so the
    # server can be started from any working directory.
    return PROJECT_ROOT / path


def _build_retriever(settings: Settings):
    """Build the production dense retriever, or raise RetrieverUnavailableError."""
    from src.retriever import build_retriever

    return build_retriever(
        "dense",
        embeddings_path=_resolve(Path(settings.embeddings_path)),
        model_name=settings.embedding_model,
        top_k=settings.top_k,
    )


def _build_retrievers(settings: Settings):
    """Build all three modes independently; never silently falls back.

    Returns ``(available, errors)`` dicts keyed by mode.
    """
    from src.retriever import build_retriever

    embeddings_path = _resolve(Path(settings.embeddings_path))
    chunks_path = _resolve(Path(settings.corpus_path) / "constitution_chunks.jsonl")
    chunks_to_use = chunks_path if chunks_path.exists() else embeddings_path

    available = {}
    errors = {}
    modes = (
        ("dense", "bm25", "hybrid")
        if settings.build_all_retrieval_modes
        else (settings.retrieval_mode,)
    )
    for mode in modes:
        try:
            available[mode] = build_retriever(
                mode,
                embeddings_path=embeddings_path,
                chunks_path=chunks_to_use,
                model_name=settings.embedding_model,
                top_k=settings.top_k,
                fusion=settings.hybrid_fusion,
                rrf_k=settings.hybrid_rrf_k,
                dense_weight=settings.hybrid_dense_weight,
                bm25_weight=settings.hybrid_bm25_weight,
            )
        except RetrieverUnavailableError as exc:
            errors[mode] = exc
        except ValueError as exc:
            errors[mode] = RetrieverUnavailableError(
                reason="hybrid_fusion_unavailable", detail=str(exc)
            )
    return available, errors


def create_app(
    settings: Settings | None = None,
    *,
    orchestrator: RagOrchestrator | None = None,
    retriever_error: RetrieverUnavailableError | None = None,
) -> FastAPI:
    settings = settings if settings is not None else Settings.from_env()

    if retriever_error is None and orchestrator is None:
        retrievers, errors = _build_retrievers(settings)
        if retrievers:
            orchestrator = RagOrchestrator(
                retrievers=retrievers,
                generator=create_generation_service(settings),
                evidence_threshold=settings.evidence_threshold,
                evidence_thresholds=settings.mode_evidence_thresholds,
                retrieval_mode=settings.retrieval_mode,
            )
        else:
            retriever_error = errors.get(settings.retrieval_mode) or next(iter(errors.values()))

    app = FastAPI(
        title="Constitution RAG API",
        description=(
            "Question answering over the Constitution of India with source "
            "citations. Educational information only; not legal advice."
        ),
        version=APP_VERSION,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_parse_cors_origins(os.getenv("CORS_ORIGINS")),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health")
    def health() -> dict:
        status = {
            "status": "ok" if orchestrator is not None else "degraded",
            "version": APP_VERSION,
            "retrieval": (
                orchestrator.status() if orchestrator is not None else _degraded_status(retriever_error)
            ),
            "legal_advice_notice": "Educational information only; not legal advice.",
        }
        return status

    @app.post("/ask")
    def ask(payload: dict = Body(...)):
        # -- request validation -------------------------------------------------
        if not isinstance(payload, dict):
            return JSONResponse(status_code=400, content={"detail": "Request body must be a JSON object."})

        question = payload.get("question")
        if question is None:
            return JSONResponse(status_code=400, content={"detail": "Field 'question' is required."})
        if not isinstance(question, str) or not question.strip():
            return JSONResponse(
                status_code=400,
                content={"detail": "Field 'question' must be a non-empty string."},
            )
        if len(question) > MAX_QUESTION_LENGTH:
            return JSONResponse(
                status_code=400,
                content={
                    "detail": f"Question too long: maximum {MAX_QUESTION_LENGTH} characters."
                },
            )
        if "retrieval_mode" in payload and payload["retrieval_mode"] not in {
            "dense",
            "bm25",
            "hybrid",
        }:
            return JSONResponse(
                status_code=400,
                content={
                    "detail": "retrieval_mode must be one of: dense, bm25, hybrid."
                },
            )

        try:
            request = AskRequest(
                question=question,
                retrieval_mode=payload.get("retrieval_mode", settings.retrieval_mode),
            )
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})

        if orchestrator is None:
            assert retriever_error is not None
            return JSONResponse(status_code=503, content={"detail": retriever_error.detail})

        try:
            response = orchestrator.answer(request)
        except RetrieverUnavailableError as exc:
            return JSONResponse(status_code=503, content={"detail": exc.detail})
        except GenerationError as exc:
            logger.warning("Answer generation failed: %s", exc)
            return JSONResponse(
                status_code=exc.http_status,
                content={"detail": exc.user_message},
            )
        except Exception:
            return JSONResponse(
                status_code=500,
                content={"detail": "An unexpected error occurred while answering the question."},
            )

        return response.to_dict()

    # Serve the static frontend last so /health, /ask and /docs keep priority.
    if (FRONTEND_DIR / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

    return app


def _degraded_status(error: RetrieverUnavailableError | None) -> dict:
    if error is None:
        return {"orchestrator": "ready", "retriever": {"ready": False}}
    return {
        "orchestrator": "degraded",
        "retriever": {
            "ready": False,
            "reason": error.reason,
            "detail": DEGRADED_REASONS.get(error.reason, error.detail),
        },
    }


settings = Settings.from_env()
try:
    app = create_app(settings)
except RetrieverUnavailableError as app_error:
    app = create_app(settings, retriever_error=app_error)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)
