# M9.1 — Hybrid Retrieval Controlled Experiment

Completed on 2026-10-02 on the unchanged local M6 cohort. M8 remains frozen; this is an offline retrieval experiment, not a default pipeline change.

## 1. Motivation

M8 Document Intelligence is frozen. This experiment tests whether lexical retrieval can retrieve evidence that exists in the current structured corpus but is absent from dense Top-20. It focuses on the four original M7.2 RETRIEVAL_FAILURE cases, while evaluating the entire unchanged M6 set and checking regressions.

## 2. BM25 and tokenization design

The existing requirements contain NumPy, sentence-transformers and PyMuPDF. A small postings-based native BM25 implementation avoids a new package, service or model dependency. It indexes the same chunks in memory. Its fixed parameters are k1=1.2 and b=0.75, with positive IDF:

```text
idf(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
score(q,d) = sum_unique_query_terms idf(t) * tf(t,d) * (k1+1)
             / (tf(t,d) + k1 * (1-b+b*length(d)/average_length))
```

These parameter defaults are documented by [Elastic](https://www.elastic.co/guide/en/elasticsearch/reference/8.19/index-modules-similarity.html); [its BM25 explanation](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables) describes term-frequency saturation and length normalization. No Elasticsearch component is used here.

The single frozen tokenizer applies NFKC, case folding, and generic dash normalization. It retains complete ASCII letter/digit identifiers with internal hyphens, slashes, underscores or decimal points. For each contiguous Chinese run it emits individual characters and adjacent two-character tokens. It uses no QA-derived vocabulary, stopword list or identifier-specific rules. Length normalization counts these emitted tokens. Repeated query tokens receive one query-term vote; document term frequencies are retained.

Synthetic tokenizer tests preserve OTN-400G, P-101, 0.82MPa, GB/T and GPON-OLT-16 after case folding. These examples are test fixtures, not private industrial evidence. No-overlap queries return no lexical candidates. This is a basic tokenizer, not a Chinese morphological analyzer; short common Chinese tokens can match many unrelated pages, and identifiers separated by OCR spaces are not repaired.

## 3. Fusion design

Dense Top-20 and BM25 Top-20 are merged by equal-weight RRF with fixed k=60:

```text
rrf_score(chunk) = sum_present_rankings 1 / (60 + rank)
```

The constant follows the [original RRF paper](https://cormack.uwaterloo.ca/cormack/cormacksigir09-rrf.pdf). It was fixed before reading M9 results; no parameter sweep was performed. Cosine and BM25 scores are never averaged.

Chunk identity includes source, page, block type, block index and chunk_id because chunk_id restarts in each Document. Duplicates receive one vote per ranking and occupy one fused slot. Ties use stable dense-first discovery order. The unique union has at most 40 chunks; only its first 20 enter the existing BGE reranker. The full union is a recall diagnostic, not a larger formal reranker input.

## 4. Evaluation protocol

The run requires the original ignored M6 input manifest, unchanged QA and M7 analysis, all eight PDFs, and matching MinerU caches. It reads cached structured documents directly and fails closed on a cache miss: no PDF parsing, OCR or vision calls are made. QA/PDF/cache hashes and controlled core-file hashes are checked before and after execution.

All arms share 1,594 structured Documents and 2,996 chunks, default max_chars=500/overlap=80, the original multilingual MiniLM embedding, cosine retrieval, and BAAI/bge-reranker-v2-m3. Query embeddings are shared between the dense-based arms. The Qwen generation pipeline is not invoked.

The file has 35 QA rows: 32 scored and three unanswerable rows excluded under the original rule. Formal metrics retain the 32-question M7 denominator. Q17's known M8 GT uncertainty is flagged, with an additional 31-question sensitivity slice; the QA and matcher are not edited.

The evaluator gains only an optional ranked-candidate provider and diagnostic limit. Its default path remains equivalent to the original dense behavior. Source/page hits, raw case-folded evidence and NFKC/whitespace-normalized evidence all reuse existing scoring code. Recall@20 means the scoped cumulative 20-candidate pool contains the required normalized evidence. It is also reported on the subset whose evidence exists in chunks.

## 5. Overall metrics

All formal counts below use the same 32 scored questions.

| Mode | Top1 source | Top3 source | Top1 page | Top3 page | Normalized evidence | Strict/raw evidence | Recall@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense | 25/32 | 30/32 | 9/32 | 15/32 | 10/32 | 8/32 | 17/32 |
| BM25 | 29/32 | 31/32 | 19/32 | 20/32 | 15/32 | 12/32 | 20/32 |
| Hybrid | 29/32 | 31/32 | 16/32 | 21/32 | 17/32 | 13/32 | 21/32 |
| Dense + BGE | 31/32 | 31/32 | 18/32 | 19/32 | 14/32 | 11/32 | 17/32 |
| Hybrid + BGE | 31/32 | 32/32 | 21/32 | 23/32 | 17/32 | 13/32 | 21/32 |

Dense and Dense+BGE reproduce M7. All 35 rows' dense and reranked evidence ranks also match the historical M7 anonymous analysis exactly. Hybrid+BGE adds four Top-3 page hits and three normalized evidence hits over Dense+BGE. Recall@20 improves from 17/32 to 21/32. Among the 21 questions whose evidence exists in the unchanged chunks, recall improves from 17/21 to 21/21; the other 11 remain unavailable under this matcher, including Q17's known GT uncertainty. This is not evidence that all 11 are confirmed parsing defects.

Removing Q17 for sensitivity, without editing QA, gives Dense+BGE Top1 page 17/31, Top3 page 18/31, evidence 14/31; Hybrid+BGE gives 20/31, 22/31, 17/31. The gains and regression conclusions are unchanged.

## 6. Original retrieval-failure cases

Ranks are normalized, source/page-scoped evidence ranks within each 20-candidate ordering; an em dash means absent. A reranked rank above 3 is a final failure.

| Anonymous ID | Dense | BM25 | Hybrid | Hybrid+BGE | Classification | Final evidence hit |
|---|---:|---:|---:|---:|---|---|
| Q01 | — | 1 | 3 | 1 | RESCUED_BY_HYBRID | Yes |
| Q13 | — | 1 | 4 | 1 | RESCUED_BY_HYBRID | Yes |
| Q21 | — | 1 | 3 | 1 | RESCUED_BY_HYBRID | Yes |
| Q24 | — | 6 | 12 | 13 | RESCUED_BY_HYBRID | No |

| Category (original four cases only) | Count |
|---|---:|
| RESCUED_BY_HYBRID | 4 |
| RESCUED_BY_BM25 | 0 |
| STILL_RETRIEVAL_FAILURE | 0 |
| PARSING_OR_GT_ISSUE | 0 |

All four have evidence in the frozen corpus. BM25 recovers all four, fusion retains all four in Top-20, and BGE brings three into final Top-3. Q24 is a candidate-recall recovery but an unresolved ranking failure; it must not be reported as a fully repaired case. For these four, the unique-union evidence ranks equal their Hybrid ranks, so trimming the union to 20 loses none.

Categories distinguish candidate recovery from final Top-3 success. RESCUED_BY_HYBRID means evidence entered the fused Top-20; RESCUED_BY_BM25 means BM25 found it but the fused pool did not retain it. STILL_RETRIEVAL_FAILURE means neither candidate pool recovered it. PARSING_OR_GT_ISSUE is used when the corpus cannot establish evidence or the ground truth is uncertain. Final reranker evidence rank and hit are listed separately.

## 7. Identifier and numeric observations

| Query feature | Questions | Dense evidence | BM25 | Hybrid | Dense+BGE | Hybrid+BGE |
|---|---:|---:|---:|---:|---:|---:|
| Identifier | 5 | 2 | 3 | 3 | 3 | 3 |
| Numeric token | 11 | 6 | 7 | 8 | 8 | 8 |
| Latin token | 7 | 2 | 3 | 3 | 3 | 3 |
| Acronym-like token | 4 | 2 | 3 | 3 | 3 | 3 |
| Chinese text | 32 | 10 | 15 | 17 | 14 | 17 |

The original four retrieval failures and the three added final successes are Chinese-only queries without numeric/Latin/identifier flags. Identifier preservation is verified by tests, but this cohort does not support attributing the final gains primarily to exact identifier matching. Chinese lexical matching is consistent with the observed recoveries; without a tokenizer ablation it is not possible to isolate the causal contribution of character versus bigram tokens. Numeric/identifier groups improve over Dense alone, but show no additional normalized-evidence gains over Dense+BGE.

Feature groups are generic, overlapping flags computed for every query: identifiers, numbers, Latin tokens, acronym-like text and Chinese text. They are post-experiment descriptive slices, not input selection rules or a causal token-ablation study.

## 8. Regressions and lexical false-positive diagnostics

Dense+BGE → Hybrid+BGE preserves all 14 normalized-evidence successes: gains Q01/Q13/Q21, REGRESSION count 0. It preserves all 19 Top-3 page successes: gains Q01/Q13/Q20/Q21, REGRESSION count 0. Joint page+evidence success gains the same three evidence cases without losses.

Pure BM25 is not a safe replacement for Dense: it gains eight normalized successes but loses Q06, Q22 and Q23 (net 10 → 15). RRF preserves all ten Dense evidence successes and gains seven (10 → 17).

| Case | QA category | Dense evidence rank | BM25 rank | Hybrid rank | Observed lexical interference |
|---|---|---:|---:|---:|---|
| Q06 | table | 2 | 8 | 1 | All three BM25 leaders are outside the expected scope, despite each sharing one numeric token and 25 Chinese tokens. |
| Q22 | multi_fact | 1 | 5 | 1 | All three BM25 leaders are outside scope; Chinese token overlap does not establish the required combined evidence. |
| Q23 | drawing_layout | 1 | 7 | 1 | Out-of-scope BM25 leaders have higher query-token coverage than the correct Dense leader; overlap alone does not recover the layout evidence. |

These observed scope errors show numeric and Chinese overlap can push wrong pages upward. They do not establish a specific similar-model or same-field cause without further semantic annotation.

| Dense Top3 evidence | BM25 Top3 evidence | Questions | Hybrid Top3 successes |
|---|---|---:|---:|
| No | Yes | 8 | 7 |
| Yes | No | 3 | 3 |
| Yes | Yes | 7 | 7 |
| No | No | 14 | 0 |

Fusion exploits seven of eight lexical-only successes and protects all three dense-only successes. When both formal Top-3 outputs fail, it creates no new Top-3 success in this run. This table concerns Top-3, not absence from the entire corpus.

Hybrid → Hybrid+BGE gains Q13 but loses Q19, leaving normalized evidence at 17/32. Q19 moves from Hybrid rank 3 to reranked rank 5; it is a regression relative to unreranked Hybrid, though not relative to the M7 Dense+BGE baseline. Thus BGE uses new recall on three original failures but does not improve Hybrid's aggregate evidence score.

Across all 96 Top-3 candidate slots, source/page scope mismatches are Dense 80, BM25 71, Hybrid 72, Dense+BGE 71, Hybrid+BGE 66. These include cases whose evidence is unavailable.

A joint success requires both Top-3 page and normalized evidence. Page-only and evidence-only changes are tracked separately. Scope-mismatch counts are relative to the frozen QA source/page; they are retrieval diagnostics, not human proof that every unmatched page is semantically irrelevant. Evidence matches also retain the existing limitations of short numeric substring matching.

## 9. Runtime

The run used the existing local CPU models (Python 3.14.4, NumPy 2.5.3, sentence-transformers 6.1.0, torch 2.14.0, PyMuPDF 1.28.2). No package or model was installed or downloaded.

| Stage | Measured cost |
|---|---:|
| BM25 index build, 2,996 chunks | 0.298 s once |
| Embedding model load + corpus embedding | 150.436 s once |
| BGE load | 8.970 s once |
| Shared query embedding | 58.98 ms/query mean |
| Dense cosine search | 10.74 ms/query mean |
| BM25 search | 17.73 ms/query mean |
| RRF | 0.50 ms/query mean |
| Hybrid recall, excluding embedding | 28.97 ms/query mean |
| Dense candidate BGE | 33.872 s/query mean |
| Hybrid candidate BGE | 33.552 s/query mean |

The added lexical+fusion search cost is approximately 18.23 ms/query in this run, small relative to CPU BGE scoring. The approximately 29 ms Hybrid value is not end-to-end RAG latency; it excludes embedding, reranking and generation. Differences between the two reranker means are not evidence of a reliable speed improvement.

These are one local CPU run's descriptive timings, without repeated trials or randomized arm order. Dense query time is cosine retrieval only; BM25 time is lexical search; Hybrid query time sums dense retrieval, lexical search and fusion and excludes the shared query embedding. Reranker times are separate. Model loading and corpus embedding are setup costs, not query latency. This is not a production throughput benchmark.

## 10. Decision

Retain this fixed, optional Hybrid strategy for the next controlled integration step: four original candidate-retrieval failures are recovered, three become final successes, overall page/evidence metrics improve, no M7 success regresses, and the added index/search complexity is small. This supports keeping the strategy; it does not justify silently changing the default RAG pipeline or claiming production accuracy from 32 reused questions.

A next M9 step should validate the unchanged strategy on independent questions and, if integration is approved, expose an explicit opt-in retrieval mode with the same budgets/provenance. Do not tune fusion parameters on this cohort. No Qdrant is needed for 2,996 in-memory chunks; database engineering remains a separate decision.

After Hybrid, four evidence-available questions still miss final Top-3: Q07 (rank 12), Q19 (5), Q24 (13), Q25 (7). Their remaining issue is ranking under this evaluator, rather than missing Top-20 recall. Inspect these ranking cases and independent-set generalization before increasing candidate budgets or changing models. The 11 unavailable-evidence/GT cases are not solved by lexical fusion, and M8 stays frozen. No vision/OCR, embedding, prompt or reranker changes are proposed here.

## Reproduction and validation

From rag-agent/, after provisioning the original ignored local M6 inputs and caches:

```bash
python evaluation/run_m9_hybrid_retrieval_experiment.py --compare-all
```

For individual offline arms use --retriever dense|bm25|hybrid and --rerank on|off. Dense/off is the default; BM25-only is unreranked in this fixed five-arm experiment. The application CLI and its default RAG behavior remain unchanged.

Detailed anonymous ranks, structural source aliases, scores, query-feature flags, transitions, timings and hash fingerprints are written only beneath the Git-ignored outputs/m9_hybrid_retrieval/ directory. No raw question, answer, evidence text, document content, PDF filename or arbitrary metadata is serialized. No real input, cache or model weight is included in Git.

Validation performed after the real run:

```text
python -m compileall src evaluation          PASS
python -m unittest discover -s tests         114 tests, PASS
git diff --check                            PASS
```

Additional checks: all 35 historical M7 dense/reranked evidence ranks match; no duplicate structural identity occurs in exported candidate pools; runtime input/cache/core fingerprints are unchanged. Changed and new files, plus the anonymous summary, contain no matching full private questions, source names, answers or evidence strings of at least 12 characters. Anonymous fields are allowlisted and covered by privacy tests; this string check complements that schema, rather than proving privacy by itself. Tracked-file inspection finds no PDF, local QA, cache, model weight or experiment output. The new files were separately checked for trailing whitespace because git diff --check does not inspect untracked files.

The single-arm rank-provenance fallback was corrected after the run and verified by a failing-then-passing regression test. It affects export metadata when other arms are absent, not compare-all rankings or measurements; the real experiment was not rerun for that metadata-only correction.

Work remains uncommitted on codex/m9-hybrid-retrieval. No merge, tag or push was performed.
