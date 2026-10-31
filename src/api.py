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

import os
from pathlib import Path

from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.config import Settings
from src.rag import RagOrchestrator
from src.retriever import RetrieverUnavailableError
from src.schemas import AskRequest

APP_VERSION = "0.1.0"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# Documented request limit for a single question (see Docks/DESIGN.md).
MAX_QUESTION_LENGTH = 1000

DEGRADED_REASONS = {
    "index_not_found": "Retrieval index not found on the server.",
    "empty_index": "Retrieval index contains no chunks.",
    "corrupt_index": "Retrieval index is malformed.",
    "embedding_model_unavailable": "Embedding model backend unavailable on the server.",
}


def _parse_cors_origins(raw: str | None) -> list[str]:
    values = raw or os.getenv("CORS_ORIGINS", "*")
    return [origin.strip() for origin in values.split(",") if origin.strip()]


def _build_retriever(settings: Settings):
    """Build the production dense retriever, or raise RetrieverUnavailableError."""
    from src.retriever import build_dense_retriever

    embeddings_path = Path(settings.embeddings_path)
    if not embeddings_path.is_absolute():
        # Relative settings are resolved against the repository root so the
        # server can be started from any working directory.
        embeddings_path = PROJECT_ROOT / embeddings_path

    return build_dense_retriever(
        embeddings_path=embeddings_path,
        model_name=settings.embedding_model,
        top_k=settings.top_k,
    )


def create_app(
    settings: Settings | None = None,
    *,
    orchestrator: RagOrchestrator | None = None,
    retriever_error: RetrieverUnavailableError | None = None,
) -> FastAPI:
    settings = settings if settings is not None else Settings.from_env()

    if retriever_error is None and orchestrator is None:
        try:
            retriever = _build_retriever(settings)
            orchestrator = RagOrchestrator(
                retriever=retriever,
                evidence_threshold=settings.evidence_threshold,
                retrieval_mode=settings.retrieval_mode,
            )
        except RetrieverUnavailableError as exc:
            retriever_error = exc

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
            request = AskRequest(question=question, retrieval_mode=payload.get("retrieval_mode", "dense"))
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})

        if orchestrator is None:
            assert retriever_error is not None
            return JSONResponse(status_code=503, content={"detail": retriever_error.detail})

        try:
            response = orchestrator.answer(request)
        except RetrieverUnavailableError as exc:
            return JSONResponse(status_code=503, content={"detail": exc.detail})
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
