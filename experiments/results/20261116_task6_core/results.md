# Retrieval Evaluation Results

- Run ID: `20261116_task6_core`
- Generated: 2026-11-16T14:03:59.209959+00:00
- Corpus: `D:\Semester -5\Deep Learning\Project\constitution-rag-chatbot\Data\processed\constitution_chunks.jsonl` (sha256 `3fffae984ed5472321bac46b773ad11def8dcafa11dda6cdeed3c485be0017fc`)
- Dataset: `D:\Semester -5\Deep Learning\Project\constitution-rag-chatbot\Data\processed\test_questions.jsonl` (sha256 `929a78f1f39f793c336c731ec09fdaf922673e89361c7260346bfe1a4983f092`)

| mode | n | answerable | recall@k | hit@k | MRR | ctx_prec | ctx_rec | citation_acc | latency_ms (mean/p50/p95) |
|---|---|---|---|---|---|---|---|---|---|
| none | 180 | 160 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | n/a |
| bm25 | 180 | 160 | 0.659 | 0.675 | 0.547 | 0.139 | 0.659 | 1.000 | 4.4 / 4.2 / 7.2 |

Notes: recall@k = |top-k overlap gold| / |gold|; ctx_prec = |retrieved overlap gold| / |retrieved|; ctx_rec equals recall@k. citation_acc checks cited IDs belong to the retrieved set (structurally guaranteed by the orchestrator). Latency is retrieval-only.
