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
  "citations": [ { "chunk_id": "...", "article": "Article 21", "section": null, "page": 9,
                   "source_title": "Constitution of India", "source_url": "" } ],
  "sources":   [ { "chunk": { "chunk_id": "...", "text": "...", "article": null,
                    "part": null, "section": null, "clause": null, "page": 9, "source_title": "...",
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
* If the top retrieval score is below the configured evidence threshold, the
  response has `insufficient_evidence: true` with a visible warning.
* Citations always come from retrieved metadata, never model-generated text.
* With `LLM_API_KEY` configured, answers use the configured grounded
  OpenAI-compatible generator. Without a key, the service uses its
  clearly-labelled extractive baseline; it never presents that baseline as an
  LLM answer.

Errors:

| Status | Meaning |
|---|---|
| 400 | Invalid question (empty / missing / non-string / >1000 chars) or invalid `retrieval_mode` |
| 405 | Non-POST on `/ask` |
| 422 | Malformed JSON body (FastAPI default) |
| 500 | Unexpected server error (safe generic message) |
| 429 | The configured AI provider is out of quota or rate-limiting requests |
| 502 | The configured answer-generation provider failed or timed out |
| 503 | Retrieval unavailable (index missing/corrupt or embedding model backend not installed) |

## Static frontend

`GET /` serves the chat UI from `frontend/` (plain HTML/CSS/JS, no build
step). FastAPI docs are at `/docs` and `/openapi.json`.

## Running locally

```bash
# 1. Create a local environment and install dependencies
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# 2. Install and start local Ollama (macOS/Homebrew)
brew install ollama
brew services start ollama
ollama pull qwen2.5:3b

# 3. Use the local Ollama configuration
cp -n .env.example .env

# 4. Download the verified official PDF and build the Article-aware index
.venv/bin/python download_constitution.py
.venv/bin/python extract_pdf.py
.venv/bin/python clean_text.py
.venv/bin/python extract_articles.py
.venv/bin/python chunk_text.py
.venv/bin/python create_embeddings.py

# 5. Start the API + UI
.venv/bin/python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
# or
.venv/bin/python -m src.api
```

The pinned PDF is the English Constitution edition marked “as on 10 September
2026”, published in the Legislative Department's official document catalogue.
The download script checks its SHA-256 before writing it under the ignored
`Data/raw/` directory. The Article-aware chunker includes the Articles,
Schedules, and appendices with page and source-version metadata.

On Apple Silicon, `.env.example` selects `RAG_DEVICE=cpu` because `bge-m3` can
exceed MPS memory. The embedding script honors this setting.

Environment variables (see `.env.example`): `LLM_PROVIDER`, `LLM_MODEL`,
`LLM_BASE_URL`, `LLM_API_KEY`, `REQUEST_TIMEOUT_SECONDS`, `EMBEDDINGS_PATH`,
`EMBEDDING_MODEL`, `TOP_K`, and `DENSE_EVIDENCE_THRESHOLD`,
`BM25_EVIDENCE_THRESHOLD`, `HYBRID_EVIDENCE_THRESHOLD`. `EVIDENCE_THRESHOLD`
remains as a legacy dense-threshold fallback. The example thresholds are
scale-aware starting points, not correctness probabilities; calibrate them
against the current corpus using the evaluation workflow in the README.
For BM25 queries that explicitly name an Article, retrieval is restricted to
that Article's indexed chunks; an unindexed Article returns no passages so the
orchestrator can abstain instead of citing unrelated provisions.
The default provider is local Ollama and requires no API key or API billing.
For a hosted provider, set the corresponding URL, model, and secret key in
`.env`; secrets are never returned by `/health`.

## Integration contract for other team members

**Satish (retrieval improvements / BM25 / hybrid):** implement the
`DenseRetriever`-compatible interface in `src/retriever.py` —
`status() -> dict` and
`retrieve(question, top_k=None) -> list[RetrievedPassage]` — and return your
instance from `build_retriever(mode, ...)`; `src/api.py::_build_retrievers`
builds all modes through that factory, so no API change is needed to add one.

**Answer generation:** configure `LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL`,
and `LLM_API_KEY` in the local `.env`. The provider adapter lives in
`src/generation.py`; tests can inject a `GenerationService` into
`create_orchestrator`. Keep source citations sourced from retrieved metadata,
never from free-form model output.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

`tests/test_api.py` covers health status, ask contracts, validation errors,
insufficient-evidence abstention, citation↔source consistency and a mocked
generator (deterministic fake embedder, no external model or LLM required).
