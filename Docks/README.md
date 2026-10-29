# Constitution RAG — Question Answering over the Constitution of India

> **Project type:** NLP / Retrieval-Augmented Generation (RAG)  
> **Team:** 12  
> **Team members:** Prem Kumar, Ritik Singh, Satish Kumar, Pooran Singh  
> **Status:** Project specification and implementation guide

## 1. Project goal

Build a chatbot that answers questions about the Constitution of India using retrieved evidence from an official Constitution source. Every substantive answer should cite the relevant Article/provision and identify the supporting source passage. When the corpus does not provide sufficient evidence, the application must say so rather than inventing an answer.

This is an educational information system, not legal advice.

## 2. Problem statement

General-purpose LLMs can answer confidently without grounding claims in the actual constitutional text. This project will build and evaluate a retrieval-augmented system and investigate which retrieval design choices improve evidence retrieval, answer faithfulness, answer relevance, and Article-citation accuracy.

## 3. Research question

**Which retrieval design choices matter most for question answering over the Constitution of India?**

The project will compare:
1. LLM without retrieval (baseline)
2. Basic dense RAG using embeddings and FAISS
3. BM25 lexical retrieval
4. Hybrid retrieval combining lexical and dense results
5. Hybrid retrieval with an optional reranker
6. Different chunk sizes and overlap settings

## 4. Scope

### In scope
- English-language first release.
- Official Constitution PDF as the primary corpus.
- PDF extraction, validation, chunking, and Article/Part/page metadata.
- Embeddings and FAISS dense retrieval.
- BM25 and hybrid retrieval; optional reranking.
- LLM answers grounded in retrieved passages.
- Article/source citations and insufficient-evidence handling.
- A reviewed 150–200 question evaluation dataset.
- RAGAS metrics, retrieval metrics, human review, latency/cost tracking.
- A simple chat interface, API, tests, reproducible experiments, and final report.

### Out of scope for the initial release
- Legal advice or predictions about case outcomes.
- Treating the chatbot as an official government service.
- Claims about current court decisions or laws outside the corpus.
- Additional languages unless separately implemented and evaluated.
- Fine-tuning a foundation model unless approved as a later extension.

## 5. Core requirements

- **R1 Source traceability:** Every chunk must point to the official source and preserve available Article/Part/clause/page metadata.
- **R2 Evidence-first answers:** The LLM receives retrieved passages and must not invent citations.
- **R3 Citation integrity:** Citations must be constructed from retrieved metadata, not generated as free-form text alone.
- **R4 Insufficient evidence:** The system must abstain or clearly communicate uncertainty when relevant evidence is not retrieved.
- **R5 Comparable experiments:** All retrieval variants use the same versioned test set and documented configurations.
- **R6 Evaluation:** Measure retrieval quality, faithfulness, answer relevance, citation accuracy, correctness, latency, and approximate cost.
- **R7 Reproducibility:** Record corpus version, models, prompts, chunking, top-k, package versions, and experiment configuration.
- **R8 Security:** No API keys or secrets committed to Git.
- **R9 Usability:** A user can submit a question and see the answer, Article citations, source passages, and error/uncertainty states.
- **R10 Documentation:** Setup, usage, tests, limitations, and research results must be documented.

## 6. Recommended implementation sequence

1. Validate the official Constitution source and record version/date.
2. Extract and inspect PDF text.
3. Build a structured corpus and metadata.
4. Prepare the question/answer/evidence dataset.
5. Implement Article-aware chunking.
6. Implement embeddings and FAISS dense retrieval.
7. Implement an LLM-without-retrieval baseline.
8. Implement basic RAG with validated citations.
9. Add BM25 and hybrid retrieval.
10. Run chunk-size and reranker experiments.
11. Evaluate using retrieval metrics, RAGAS, and human review.
12. Build and integrate API and UI.
13. Test, deploy, document, and present results.

## 7. Team ownership

| Member | Primary ownership | Expected deliverables |
|---|---|---|
| **Prem Kumar** | Corpus ingestion, cleaning, metadata, chunking, embeddings, FAISS | Validated corpus and dense retriever |
| **Ritik Singh** | LLM client, prompts, grounded generation, citations, abstention | Answer-generation module and baseline |
| **Satish Kumar** | Test set, BM25/hybrid retrieval, reranking experiments, evaluation | Dataset, experiment scripts, metrics, charts |
| **Pooran Singh** | API, frontend/chat UI, integration, deployment, docs | Usable application and end-to-end tests |

All members participate in reviews, integration, citation checks, and final presentation. Owners should agree on shared schemas and interfaces before implementation.

## 8. Suggested repository layout

```text
constitution-rag/
├── data/
│   ├── raw/                 # downloaded source; do not commit if size/licensing says otherwise
│   ├── processed/
│   └── test_questions.csv
├── artifacts/               # generated indexes; usually rebuildable
├── src/
│   ├── config.py
│   ├── schemas.py
│   ├── ingest.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── dense_retriever.py
│   ├── bm25_retriever.py
│   ├── hybrid_retriever.py
│   ├── llm_client.py
│   ├── prompts.py
│   ├── generation.py
│   ├── evaluate.py
│   └── api.py
├── frontend/
├── experiments/
│   ├── configs/
│   ├── results/
│   └── figures/
├── tests/
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
├── ARCHITECTURE.md
├── DESIGN.md
└── AGENTS.md
```

## 9. Dataset design

### Corpus chunk record

```json
{
  "chunk_id": "stable-unique-id",
  "text": "Exact extracted source passage",
  "article": "Article identifier if verified",
  "part": "Part identifier if verified",
  "clause": null,
  "page": 1,
  "source_title": "Constitution of India",
  "source_url": "official source URL",
  "source_version": "recorded source version/date"
}
```

Values must be extracted from the actual source; the example is a schema, not actual source content.

### Test question record

```json
{
  "question_id": "q001",
  "question": "Question text",
  "reference_answer": "Human-reviewed answer",
  "relevant_articles": ["verified Article identifier"],
  "evidence_chunk_ids": ["stable-chunk-id"],
  "answerable_from_corpus": true,
  "category": "direct_article_question"
}
```

The 150–200 question set should include direct Article questions, paraphrases, questions requiring multiple passages, ambiguous questions, and unanswerable/out-of-corpus questions. Human reviewers should verify reference answers and evidence labels.

## 10. Evaluation plan

| System | Purpose |
|---|---|
| LLM without retrieval | Baseline for comparison |
| Dense RAG (FAISS) | Basic RAG baseline |
| BM25 | Keyword/lexical retrieval comparison |
| Hybrid retrieval | Combined lexical + semantic retrieval |
| Hybrid + reranker | Measure reranking effect, if feasible |

Metrics:
- Retrieval Recall@k and MRR.
- Context precision/recall.
- Faithfulness and answer relevance (e.g., using RAGAS).
- Article citation accuracy.
- Human-verified answer correctness.
- Latency and approximate cost per question.

Do not invent results. Record metric definitions, package/model versions, parameters, and evaluation prompts. Automated metrics do not replace human verification of constitutional claims.

## 11. Acceptance criteria

The project is ready for final demonstration when:
- [ ] Source/version is documented and extraction quality is inspected.
- [ ] Chunks have stable IDs and traceable metadata.
- [ ] Dense retrieval returns ranked passages with metadata.
- [ ] Basic RAG produces evidence-grounded answers with citations.
- [ ] Unsupported questions trigger uncertainty/abstention.
- [ ] A reviewed 150–200 question test set exists.
- [ ] No-retrieval, dense, and hybrid systems can be compared fairly.
- [ ] Evaluation metrics and human checks are recorded.
- [ ] API and UI work end to end.
- [ ] Tests pass from a clean setup.
- [ ] README, architecture, design, agent instructions, and final report are updated.

## 12. Source and references

- Official Constitution source: https://www.legislative.gov.in/document/constitution-of-india-in-english
- Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*: https://arxiv.org/abs/2005.11401
- Es et al. (2024), *RAGAs: Automated Evaluation of Retrieval Augmented Generation*: https://aclanthology.org/2024.eacl-demo.16/
- FAISS: https://github.com/facebookresearch/faiss
- Sentence Transformers: https://www.sbert.net/
- RAGAS documentation: https://docs.ragas.io/
- FastAPI documentation: https://fastapi.tiangolo.com/

## 13. First team actions

- **Prem:** Find the official source, record version/date, inspect PDF extraction, propose chunk metadata schema.
- **Ritik:** Propose LLM provider/interface and structured answer/citation schema; keep provider configurable.
- **Satish:** Draft test-set categories, annotation template, and evaluation protocol.
- **Pooran:** Scaffold API/UI and agree on request/response contracts with the team.

Before substantial coding, agree on the corpus schema, answer schema, repository structure, and environment configuration. Keep `main` stable and use pull requests.
