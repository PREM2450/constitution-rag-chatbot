# Retrieval Evaluation Results

- Run ID: `20261119_040644`
- Generated: 2026-11-19T04:16:31.441458+00:00
- Corpus: `D:\Semester -5\Deep Learning\Project\constitution-rag-chatbot\Data\processed\constitution_chunks.jsonl` (sha256 `3fffae984ed5472321bac46b773ad11def8dcafa11dda6cdeed3c485be0017fc`)
- Dataset: `D:\Semester -5\Deep Learning\Project\constitution-rag-chatbot\Data\processed\test_questions.jsonl` (sha256 `929a78f1f39f793c336c731ec09fdaf922673e89361c7260346bfe1a4983f092`)

| mode | n | answerable | recall@k | hit@k | MRR | ctx_prec | ctx_rec | citation_acc | latency_ms (mean/p50/p95) |
|---|---|---|---|---|---|---|---|---|---|
| none | 180 | 160 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | n/a |
| dense | 180 | 160 | 0.666 | 0.675 | 0.538 | 0.143 | 0.666 | 1.000 | 1560.6 / 1451.0 / 2567.1 |
| bm25 | 180 | 160 | 0.659 | 0.675 | 0.547 | 0.139 | 0.659 | 1.000 | 19.9 / 16.7 / 36.3 |
| hybrid | 180 | 160 | 0.741 | 0.756 | 0.598 | 0.156 | 0.741 | 1.000 | 966.7 / 937.6 / 1353.7 |

Notes: recall@k = |top-k overlap gold| / |gold|; ctx_prec = |retrieved overlap gold| / |retrieved|; ctx_rec equals recall@k. citation_acc checks cited IDs belong to the retrieved set (structurally guaranteed by the orchestrator). Latency is retrieval-only.
