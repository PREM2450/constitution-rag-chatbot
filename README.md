# constitution-rag-chatbot

Constitution of India RAG chatbot — retrieve constitutional passages, answer
with source citations, abstain when evidence is insufficient.

> Educational information only — not legal advice.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# Local answer generation (macOS/Homebrew; no paid API key)
brew install ollama
brew services start ollama
ollama pull qwen2.5:3b

# Copy the local Ollama configuration
cp -n .env.example .env

# Download the SHA-256-verified 2026 English edition from the Legislative Department.
.venv/bin/python download_constitution.py

# Build the Article-aware corpus and dense index (model download requires internet).
.venv/bin/python extract_pdf.py && \
  .venv/bin/python clean_text.py && \
  .venv/bin/python extract_articles.py && \
  .venv/bin/python chunk_text.py && \
  .venv/bin/python create_embeddings.py

# Start backend API + chat UI (http://localhost:8000)
.venv/bin/python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

- Chat UI: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- API reference / integration contract: `Docs/API.md`
- Architecture & design: `Docs/`
- Tests: `.venv/bin/python -m unittest discover -s tests -v`

The default `.env.example` uses the local Ollama model. Constitution retrieval
and answer generation run on your machine, so no OpenAI API billing is needed.
The model download uses disk space and local RAM/CPU. To use a hosted provider
instead, configure its URL, model, and API key in `.env`; never commit that
file or paste a key into chat.

## Evaluation results

Archived run `20261119_040644` (180 human-verified questions, fixed-size
chunks, previous corpus version): hybrid recall@5 **0.741** / MRR **0.598**,
dense **0.666** / **0.538**, BM25 **0.659** / **0.547**. These metrics do not
describe the current Article-aware 2026 index. Full archived metrics and
failure analysis:
[`experiments/results/20261119_040644/final_report.md`](experiments/results/20261119_040644/final_report.md).

To regenerate current-corpus evaluation results, run
`.venv/bin/python experiments/remap_gold_questions.py`, then
`.venv/bin/python experiments/run_evaluation.py --split dev` and
`.venv/bin/python experiments/calibrate_threshold.py --run-id <run_id>`.

## Team ownership

| Member | Ownership |
|---|---|
| Prem Kumar | Corpus ingestion, embeddings, dense index |
| Ritik Singh | Backend API (`src/api.py`, `src/rag.py`, `src/retriever.py`, `src/generation.py`) and chat UI (`frontend/`) |
| Satish Kumar | Retrieval experiments, BM25/hybrid, evaluation |
| Pooran Singh | QA dataset, LoRA/QLoRA fine-tuned generator |

Retrieval and generation are replaceable interfaces — see `Docs/API.md`
§Integration contract.
