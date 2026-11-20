# Task 7 -- Evidence Threshold Calibration + Failure Analysis

- Run ID: `20261119_040644`
- Dataset: `D:\Semester -5\Deep Learning\Project\constitution-rag-chatbot\Data\processed\test_questions.jsonl`
- Method: candidate thresholds are every observed dev top-1 score plus midpoints; selection maximises Youden's J (answerable-retention - unanswerable-answer-rate) on **dev only**, then is confirmed on held-out test.
- Score semantics differ per mode: dense = cosine similarity, bm25 = unbounded BM25Okapi score, hybrid = RRF fusion score (~1/(60+rank), i.e. very small). One shared threshold cannot fit all three, which is why per-mode calibration matters.

## 1. Top-1 score distributions

### dense

| split | group | n | min | p25 | median | p75 | max |
|---|---|---|---|---|---|---|---|
| dev | answerable | 43 | 0.5273 | 0.5714 | 0.6010 | 0.6290 | 0.7199 |
| dev | unanswerable | 5 | 0.4498 | 0.5079 | 0.5217 | 0.5549 | 0.5576 |
| test | answerable | 117 | 0.4913 | 0.5657 | 0.5968 | 0.6285 | 0.7544 |
| test | unanswerable | 15 | 0.3885 | 0.4303 | 0.5045 | 0.5221 | 0.5573 |

### bm25

| split | group | n | min | p25 | median | p75 | max |
|---|---|---|---|---|---|---|---|
| dev | answerable | 43 | 11.4560 | 16.5488 | 20.8045 | 26.2640 | 75.7865 |
| dev | unanswerable | 5 | 13.7082 | 19.2896 | 20.9509 | 22.2655 | 24.6020 |
| test | answerable | 117 | 11.6229 | 19.0385 | 23.7834 | 30.8551 | 54.7005 |
| test | unanswerable | 15 | 9.6041 | 12.0587 | 12.2201 | 15.5391 | 24.6020 |

### hybrid

| split | group | n | min | p25 | median | p75 | max |
|---|---|---|---|---|---|---|---|
| dev | answerable | 43 | 0.0164 | 0.0318 | 0.0325 | 0.0328 | 0.0328 |
| dev | unanswerable | 5 | 0.0164 | 0.0164 | 0.0320 | 0.0325 | 0.0328 |
| test | answerable | 117 | 0.0164 | 0.0164 | 0.0323 | 0.0325 | 0.0328 |
| test | unanswerable | 15 | 0.0164 | 0.0164 | 0.0315 | 0.0320 | 0.0325 |

## 2. Chosen thresholds (dev-selected, test-confirmed)

| mode | threshold | dev retention | dev unans. abstention | dev J | test retention | test unans. abstention | test J |
|---|---|---|---|---|---|---|---|
| dense | 0.5576 | 0.884 (38/43) | 1.000 (5/5) | 0.884 | 0.829 (97/117) | 1.000 (15/15) | 0.829 |
| bm25 | 24.7565 | 0.279 (12/43) | 1.000 (5/5) | 0.279 | 0.453 (53/117) | 1.000 (15/15) | 0.453 |
| hybrid | 0.0321 | 0.698 (30/43) | 0.600 (3/5) | 0.298 | 0.590 (69/117) | 0.733 (11/15) | 0.323 |

## 3. Failure analysis (answerable questions, top-5)

| mode | answerable | full-recall hits | partial overlap | no-overlap misses | unanswerable false-answer risk |
|---|---|---|---|---|---|
| dense | 160 | 105 | 3 | 52 | 0/20 |
| bm25 | 160 | 103 | 5 | 52 | 0/20 |
| hybrid | 160 | 116 | 5 | 39 | 6/20 |

### dense misses by category

| category | misses | of which threshold would abstain |
|---|---|---|
| direct_article_question | 36 | 8 |
| keyword_heavy_question | 9 | 3 |
| paraphrased_question | 5 | 0 |
| ambiguous_question | 2 | 1 |

- Would answer but returned wrong chunks: q001, q004, q007, q016, q025, q028, q037, q039
- Would abstain under calibrated threshold: q013, q031, q048, q049, q058, q069, q094, q102
- Unanswerable questions still answered (false-answer risk): none

### bm25 misses by category

| category | misses | of which threshold would abstain |
|---|---|---|
| direct_article_question | 38 | 38 |
| paraphrased_question | 7 | 6 |
| ambiguous_question | 5 | 5 |
| keyword_heavy_question | 1 | 1 |
| multi_article_question | 1 | 1 |

- Would answer but returned wrong chunks: q131
- Would abstain under calibrated threshold: q001, q004, q007, q010, q013, q016, q019, q022
- Unanswerable questions still answered (false-answer risk): none

### hybrid misses by category

| category | misses | of which threshold would abstain |
|---|---|---|
| direct_article_question | 34 | 32 |
| paraphrased_question | 3 | 2 |
| ambiguous_question | 2 | 1 |

- Would answer but returned wrong chunks: q001, q076, q080, q151
- Would abstain under calibrated threshold: q004, q007, q010, q013, q016, q025, q028, q031
- Unanswerable questions still answered (false-answer risk): q162, q163, q168, q170, q176, q178

## 4. Interpretation

- A single `EVIDENCE_THRESHOLD` is applied to the top-1 score of whichever mode served the request (`src/rag.py`). Because BM25 and RRF scores live on different scales than cosine similarity, one shared value cannot abstain correctly for all modes.
- The calibrated per-mode values above are recommendations. Wiring them into the API is a configuration decision and would change default abstention behaviour, so it is documented here rather than silently applied.
- See `score_distributions.png` for the answerable vs unanswerable top-1 score histograms with the chosen cut-off.
