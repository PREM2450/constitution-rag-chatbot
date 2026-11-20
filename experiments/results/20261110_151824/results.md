# Retrieval Evaluation Results

- Run ID: `20261110_151824`
- Generated: 2026-11-10T15:18:26.012523+00:00
- Corpus: `D:\Semester -5\Deep Learning\Project\constitution-rag-chatbot\Data\processed\constitution_chunks.jsonl` (sha256 `3fffae984ed5472321bac46b773ad11def8dcafa11dda6cdeed3c485be0017fc`)
- Dataset: `D:\Semester -5\Deep Learning\Project\constitution-rag-chatbot\Data\processed\test_questions.jsonl` (sha256 `9eeb920b57c0704d3d8e252e75ecb137ac8bd7128c5b205d82f4e8e5c6a120ac`)

| mode | n | answerable | recall@k | hit@k | MRR | ctx_prec | ctx_rec | citation_acc | latency_ms (mean/p50/p95) |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 180 | 160 | 0.659 | 0.675 | 0.547 | 0.139 | 0.659 | 1.000 | 5.7 / 5.1 / 10.7 |

Notes: recall@k = |top-k overlap gold| / |gold|; ctx_prec = |retrieved overlap gold| / |retrieved|; ctx_rec equals recall@k. citation_acc checks cited IDs belong to the retrieved set (structurally guaranteed by the orchestrator). Latency is retrieval-only.
