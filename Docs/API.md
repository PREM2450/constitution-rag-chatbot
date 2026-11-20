# Backend API — Constitution RAG

Owner: **Ritik Singh** (backend API + frontend chat UI).

The API layer only validates, orchestrates and serialises. It contains no
retrieval algorithms or prompt logic; retrieval and generation are injected
behind clean interfaces so Satish and Pooran can integrate their modules
without changing the API layer.

## Endpoints

### `GET /health`

Returns service and component status. Never exposes secrets.

```json
{
  "status": "ok",                // "ok" | "degraded"
  "version": "0.1.0",
  "retrieval": {
    "orchestrator": "ready",
    "retriever": {
      "kind": "dense", "ready": true, "chunks": 4102, "dimension": 1024, "top_k": 5
    },
    "retrievers": {
      "dense":  { "kind": "dense",  "ready": true,  "chunks": 4102, "dimension": 1024, "top_k": 5 },
      "bm25":   { "kind": "bm25",   "ready": true,  "chunks": 4102, "top_k": 5 },
      "hybrid": { "kind": "hybrid", "ready": true,  "chunks": 4102, "top_k": 5,
                  "fusion": "rrf", "rrf_k": 60, "dense_weight": 1.0, "bm25_weight": 1.0 }
    },
    "available_modes": ["bm25", "dense", "hybrid"],
    "generator": { "name": "extractive_baseline" },
    "evidence_threshold": 0.6
  },
  "legal_advice_notice": "Educational information only; not legal advice."
}
```

* `retrievers` maps each mode to its own `status()`. A mode that failed to build
  is absent from `retrievers` and `available_modes`; the remaining modes stay
  usable. `retriever` is the primary (configured-default-mode) entry, kept for
  backward compatibility.
* Degraded state (`status: "degraded"`) includes `retriever.reason`:
  `index_not_found` | `empty_index` | `corrupt_index` |
  `embedding_model_unavailable` | `bm25_unavailable` |
  `hybrid_fusion_unavailable` | `retrieval_mode_unavailable`. All modes failing
  to build yields `degraded` and `503` on `/ask`.

### `POST /ask`

Request:

```json
{ "question": "What does Article 21 protect?", "retrieval_mode": "dense" }
```

* `question` — required, 1–1000 characters (`retrieval_mode` optional, one of
  `dense` | `bm25` | `hybrid`). When omitted it defaults to the configured
  `RETRIEVAL_MODE` (default `dense`); the two can differ per server.
* Each request selects its own retriever. An unavailable mode returns `503`
  with no silent fallback to another mode.

Success (200) returns the shared `AnswerResponse` contract from
`src/schemas.py::AnswerResponse.to_dict()`:

```json
{
  "answer": "...",
  "citations": [ { "chunk_id": "...", "article": "Article 21", "page": 9,
                   "source_title": "Constitution of India", "source_url": "" } ],
  "sources":   [ { "chunk": { "chunk_id": "...", "text": "...", "article": null,
                    "part": null, "clause": null, "page": 9, "source_title": "...",
                    "source_url": "", "source_version": "unknown" },
                   "score": 0.61, "rank": 1, "retrieval_method": "dense" } ],
  "insufficient_evidence": false,
  "request_id": "uuid4-hex",
  "retrieval_method": "dense",
  "warning": null
}
```

`retrieval_method` echoes the mode actually dispatched, and every
`sources[].retrieval_method` matches it. For `hybrid`, the fused list is
de-duplicated by `chunk_id`.

Rules:

* Citations are **built from retrieved chunk metadata only** — a citation's
  `chunk_id` is always present in `sources`. Missing Article labels are shown
  as `null`, never guessed.
* If the top retrieval score is below `EVIDENCE_THRESHOLD`, the response has
  `insufficient_evidence: true` with a visible warning (abstention, no
  invented answer).
* The `answer` generator is currently `extractive_baseline` (passages surfaced
  verbatim, clearly labelled). It is **not** an LLM.

Errors:

| Status | Meaning |
|---|---|
| 400 | Invalid question (empty / missing / non-string / >1000 chars) or invalid `retrieval_mode` |
| 405 | Non-POST on `/ask` |
| 422 | Malformed JSON body (FastAPI default) |
| 500 | Unexpected server error (safe generic message) |
| 503 | Retrieval unavailable (index missing/corrupt or embedding model backend not installed) |

## Static frontend

`GET /` serves the chat UI from `frontend/` (plain HTML/CSS/JS, no build
step). FastAPI docs are at `/docs` and `/openapi.json`.

## Running locally

```bash
# 1. Offline indexing (once): needs Data/raw/constitution-of-india.pdf
python3 extract_pdf.py && python3 clean_text.py && python3 chunk_text.py && python3 create_embeddings.py

# 2. Dependencies
pip3 install -r requirements.txt

# 3. Start the API + UI
python3 -m uvicorn src.api:app --host 127.0.0.1 --port 8000
# or
python3 -m src.api
```

On Apple Silicon set `RAG_DEVICE=cpu` (bge-m3 can exceed MPS memory).

`EVIDENCE_THRESHOLD` is one scalar compared against the top-1 score of every
mode (`src/rag.py`). The modes use different score scales, so a single value
cannot abstain correctly for all of them: it is meaningful for dense cosine
(~0.5–0.75), always exceeded for BM25 (unbounded scores), and never reached
for hybrid RRF (scores ≤ ~0.033). Evaluation run `20261119_040644` calibrated
per-mode dev thresholds — dense `0.5576`, bm25 `24.7565`, hybrid `0.0321`
(see `experiments/results/20261119_040644/final_report.md` and
`threshold_calibration.md`). Wiring per-mode thresholds into `Settings` is not
done yet; the scalar default remains `0.6`.

Requires `Data/processed/constitution_embeddings.jsonl` (produced by
`create_embeddings.py`) and the `sentence-transformers` package for real
retrieval. Without them the server still starts and `/ask` returns 503 with a
clear reason.

Environment variables (see `.env.example`): `EMBEDDINGS_PATH`,
`EMBEDDING_MODEL`, `TOP_K`, `EVIDENCE_THRESHOLD`, `CORS_ORIGINS`, etc.

## Integration contract for other team members

**Satish (retrieval improvements / BM25 / hybrid):** implement the
`DenseRetriever`-compatible interface in `src/retriever.py` —
`status() -> dict` and
`retrieve(question, top_k=None) -> list[RetrievedPassage]` — and return your
instance from `build_retriever(mode, ...)`; `src/api.py::_build_retrievers`
builds all modes through that factory, so no API change is needed to add one.

**Pooran / RAG generation (LLM, LoRA/QLoRA):** extend
`src/generation.py::GenerationService` (implement
`generate(question, passages) -> str`) and pass the instance to
`create_orchestrator(retriever=..., generator=...)` in `src/api.py`. Prompt
construction, API-key handling and timeouts belong in your module; the API
layer stays unchanged. Cited Articles/pages must continue to come from the
retrieved metadata — never trust free-form model citation text.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

`tests/test_api.py` covers health status, ask contracts, validation errors,
insufficient-evidence abstention, citation↔source consistency and a mocked
generator (deterministic fake embedder, no external model or LLM required).
