# constitution-rag-chatbot

Constitution of India RAG chatbot — retrieve constitutional passages, answer
with source citations, abstain when evidence is insufficient.

> Educational information only — not legal advice.

## Quick start

```bash
pip3 install -r requirements.txt

# Offline indexing (needs the official PDF in Data/raw/constitution-of-india.pdf)
python3 extract_pdf.py && python3 clean_text.py && python3 chunk_text.py && python3 create_embeddings.py

# Start backend API + chat UI (http://localhost:8000)
python3 -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

- Chat UI: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- API reference / integration contract: `Docks/API.md`
- Architecture & design: `Docks/`
- Tests: `python3 -m unittest discover -s tests -v`

## Team ownership

| Member | Ownership |
|---|---|
| Prem Kumar | Corpus ingestion, embeddings, dense index |
| Ritik Singh | Backend API (`src/api.py`, `src/rag.py`, `src/retriever.py`, `src/generation.py`) and chat UI (`frontend/`) |
| Satish Kumar | Retrieval experiments, BM25/hybrid, evaluation |
| Pooran Singh | QA dataset, LoRA/QLoRA fine-tuned generator |

Retrieval and generation are replaceable interfaces — see `Docks/API.md`
§Integration contract.
