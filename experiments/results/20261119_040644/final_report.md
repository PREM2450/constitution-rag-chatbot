# Constitution RAG — Retrieval Evaluation Final Report

- **Run ID:** `20261119_040644`
- **Date:** 2026-11-19
- **Corpus:** `Data/processed/constitution_chunks.jsonl` — sha256 `3fffae984ed5472321bac46b773ad11def8dcafa11dda6cdeed3c485be0017fc` (953 chunks, `CHUNK_SIZE=1200`, `OVERLAP=200`)
- **Embeddings:** `Data/processed/constitution_embeddings.jsonl` — 953 × 1024, `BAAI/bge-m3`, normalized
- **Dataset:** `Data/processed/test_questions.jsonl` — sha256 `929a78f1f39f793c336c731ec09fdaf922673e89361c7260346bfe1a4983f092` (180 rows, all human-verified; dev 48 / test 132)
- **Parameters:** `top_k=5`, `HYBRID_FUSION=rrf`, `rrf_k=60`, weights 1.0/1.0, `EVIDENCE_THRESHOLD=0.6` (untuned default), CPU-only

All numbers below come from this run's `manifest.json`, `raw_<mode>.jsonl` and
the calibration outputs (`threshold_calibration.{json,md}`,
`score_distributions.png`). No values are invented or copied from other runs.

## 1. Systems compared (`Docks/README.md` §10)

| System | Status | Notes |
|---|---|---|
| LLM without retrieval | **Not run** | No LLM generator/provider is configured. Only the structural `none` retrieval baseline was run; it is *not* the LLM-without-retrieval baseline. |
| Dense RAG | **Run** | `BAAI/bge-m3` + NumPy cosine (`src/dense_retriever.py`) |
| BM25 | **Run** | `rank_bm25==0.2.2` `BM25Okapi` (`src/bm25_retriever.py`) |
| Hybrid retrieval | **Run** | Dense + BM25 fused with RRF (`k=60`) (`src/hybrid_retriever.py`) |
| Hybrid + reranker | **Deferred** | `RERANKER_ENABLED=false`; reranker layer intentionally not implemented |

## 2. Overall metrics (180 questions, top-5)

| mode | n | answerable | recall@5 | hit@5 | MRR | ctx_prec | ctx_rec | citation_acc | latency ms (mean/p50/p95) |
|---|---|---|---|---|---|---|---|---|---|
| none | 180 | 160 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | n/a |
| dense | 180 | 160 | 0.666 | 0.675 | 0.538 | 0.143 | 0.666 | 1.000 | 1560.6 / 1451.0 / 2567.1 |
| bm25 | 180 | 160 | 0.659 | 0.675 | 0.547 | 0.139 | 0.659 | 1.000 | 19.9 / 16.7 / 36.3 |
| hybrid | 180 | 160 | **0.741** | **0.756** | **0.598** | **0.156** | 0.741 | 1.000 | 966.7 / 937.6 / 1353.7 |

`citation_acc = 1.000` is a structural invariant (citations are built from the
retrieved set only), not a semantic quality signal.

## 3. Per-category recall@5 / hit@5

| category | n | dense rec@5 | dense hit@5 | bm25 rec@5 | bm25 hit@5 | hybrid rec@5 | hybrid hit@5 |
|---|---|---|---|---|---|---|---|
| direct_article_question | 45 | 0.200 | 0.200 | 0.156 | 0.156 | 0.244 | 0.244 |
| paraphrased_question | 45 | 0.889 | 0.889 | 0.844 | 0.844 | 0.933 | 0.933 |
| keyword_heavy_question | 45 | 0.800 | 0.800 | 0.978 | 0.978 | 1.000 | 1.000 |
| multi_article_question | 15 | 0.900 | 1.000 | 0.767 | 0.933 | 0.833 | 1.000 |
| ambiguous_question | 10 | 0.800 | 0.800 | 0.500 | 0.500 | 0.800 | 0.800 |
| unanswerable_out_of_corpus | 20 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Hybrid wins overall and on every category except `multi_article_question`
recall (dense 0.900 vs hybrid 0.833; both hit@5 = 1.000).

## 4. Threshold calibration

Method (`experiments/calibrate_threshold.py`): candidate cut-offs are every
observed dev top-1 score plus midpoints; selection maximises Youden's J =
answerable-retention − unanswerable-answer-rate on **dev only**, then is
confirmed on held-out test.

| mode | chosen threshold | dev retention | dev unans. abstention | dev J | test retention | test unans. abstention | test J |
|---|---|---|---|---|---|---|---|
| dense | 0.5576 | 0.884 (38/43) | 1.000 (5/5) | 0.884 | 0.829 (97/117) | 1.000 (15/15) | 0.829 |
| bm25 | 24.7565 | 0.279 (12/43) | 1.000 (5/5) | 0.279 | 0.453 (53/117) | 1.000 (15/15) | 0.453 |
| hybrid | 0.0321 | 0.698 (30/43) | 0.600 (3/5) | 0.298 | 0.590 (69/117) | 0.733 (11/15) | 0.323 |

**Important caveat — the shipped default is not per-mode safe.** The
orchestrator compares the top-1 score against one `EVIDENCE_THRESHOLD`
(`src/rag.py`), but the three modes use incompatible score scales:

- **dense** cosine ≈ 0.53–0.75 → default `0.6` is roughly appropriate.
- **bm25** scores ≈ 9.6–75.8 → default `0.6` is always exceeded, so BM25
  effectively **never abstains**.
- **hybrid** RRF scores ≤ 0.0328 → default `0.6` is never reached, so hybrid
  **always abstains** with the shipped default.

The calibrated values above are recommendations. Wiring them into the API is a
configuration/product decision and is documented here rather than silently
applied; additionally `Settings.validate()` currently bounds
`EVIDENCE_THRESHOLD` to `[0, 1]`, which cannot express the BM25 value at all.

## 5. Failure analysis (answerable questions, top-5)

| mode | full-recall hits | partial overlap | no-overlap misses | unanswerable false-answer risk (at calibrated threshold) |
|---|---|---|---|---|
| dense | 105 | 3 | 52 | 0/20 |
| bm25 | 103 | 5 | 52 | 0/20 |
| hybrid | 116 | 5 | 39 | 6/20 |

### Dominant root cause — fixed-size chunks mix several Articles

`direct_article_question` recall is low for every mode (0.200 / 0.156 /
0.244). Chunking is by character count only, so one chunk frequently contains
several Articles, and the corpus carries no Article-level metadata. A question
about one Article is then outranked by neighbouring text or the table of
contents. Examples:

- **q001** — "What does Article 1 provide?" Gold `page_33_chunk_0` (contains
  Articles 1–3). Dense rank 1 is `page_4_chunk_0` (Contents/TOC) at score
  0.659.
- **q004** — "State the constitutional rule contained in Article 3." Same gold
  `page_33_chunk_0`; dense rank 1 is `page_37_chunk_0` ("12. Definition") at
  0.559.
- **q007** — "What is provided by Article 5?" Gold `page_35_chunk_0`; dense
  rank 1 is `page_56_chunk_0` ("51A. Fundamental duties") at 0.603.

Misses by category (no-overlap, all splits):

| category | dense misses | bm25 misses | hybrid misses |
|---|---|---|---|
| direct_article_question | 36 | 38 | 34 |
| paraphrased_question | 5 | 7 | 3 |
| keyword_heavy_question | 9 | 1 | 0 |
| ambiguous_question | 2 | 5 | 2 |
| multi_article_question | 0 | 1 | 0 |

Other observations:

- At the **calibrated** thresholds, most remaining misses would be abstained
  (e.g. dense 36 direct misses → 8 would abstain; bm25 38 → 38; hybrid 34 →
  32). Score-based abstention therefore trades recall for fewer unsupported
  answers.
- Unanswerable questions were **never** answered by dense or BM25 at their
  calibrated thresholds; hybrid still answered 6/20 (`q162, q163, q168, q170,
  q176, q178`).
- `ctx_prec` is low (0.14–0.16) mainly because gold evidence sets are small
  (often one chunk) while 5 chunks are returned; this is expected and is not a
  retrieval failure by itself.
- BM25 top-1 score separates answerable from unanswerable poorly (J ≈ 0.28),
  so its high calibrated threshold sacrifices most recall to avoid false
  answers.

## 6. Limitations

1. **Article metadata is absent.** Chunks carry page only; citations show
   `article: null`. Article-level citation accuracy cannot be measured, and
   `citation_accuracy` only proves cited IDs came from the retrieved set.
2. **Chunk granularity.** Fixed 1200-char chunks ignore Article boundaries,
   which is the dominant cause of direct-Article misses (§5).
3. **Single embedding model** (`BAAI/bge-m3`), no reranker, no alternative
   encoders compared.
4. **No generation quality measurement.** The `extractive_baseline` generator
   surfaces passages verbatim; RAGAS/faithfulness/answer-relevance were not run
   because no LLM provider is configured. Retrieval metrics do not establish
   answer correctness.
5. **Small unanswerable dev sample** (5 rows) makes threshold selection
   sensitive; test confirmation used 15.
6. **Latency is CPU-only** and retrieval-scoped (dense ~1.5 s/question, hybrid
   ~1.0 s, BM25 ~20 ms); end-to-end latency including generation is not
   measured.
7. **One human reviewer** verified the dataset; no inter-annotator agreement.
8. **Gold evidence is strict**: many direct-Article rows list one chunk, so
   recall penalises retrieving a different chunk of the same Article.

## 7. Reproduction

```bash
# 1. index (once)
python extract_pdf.py && python clean_text.py && python chunk_text.py && python create_embeddings.py

# 2. evaluation (writes experiments/results/<run_id>/)
python experiments/run_evaluation.py --split all

# 3. calibration + failure analysis for a run
python experiments/calibrate_threshold.py --run-id 20261119_040644

# 4. tests
python -m unittest discover -s tests -v
```

## 8. Recommendations / next steps

1. Make abstention **per-mode** (e.g. a `{dense, bm25, hybrid}` threshold map)
   instead of one scalar, and relax the `[0, 1]` validation for BM25.
2. Add **Article-aware chunking / metadata** so a chunk maps to exactly one
   Article; this should lift `direct_article_question` recall substantially.
3. Implement the deferred **reranker** (bounded top-20 → top-5) and re-run the
   same harness.
4. Configure an LLM generator to enable the LLM-without-retrieval baseline and
   RAGAS faithfulness/answer-relevance.
