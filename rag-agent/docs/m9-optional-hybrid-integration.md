# M9.3 — Optional Hybrid Retrieval Integration

## Purpose

Expose the frozen M9.1 Dense + BM25 + RRF retrieval strategy in the PDF RAG
demo without changing the default retrieval or generation path. M9.1 and M9.2
remain controlled retrieval experiments; this integration does not retune their
parameters or make Hybrid the default.

## API and CLI

`src/hybrid_retrieval.py` is a thin orchestration layer. It requests Dense
Top-20 and BM25 Top-20 candidates, then delegates fusion to the existing
`lexical_retrieval.py` implementation. Tokenization, BM25, structural chunk
identity, deduplication and RRF remain implemented only in that module. A PDF
session builds one BM25 index only when Hybrid is selected and reuses it for
each query.

The retrieval results keep the shared `text`, `score`, `source`, `chunk_id` and
`metadata` fields. Dense `score` is cosine similarity; Hybrid `score` is an RRF
score, so those values are not directly comparable. Hybrid `dense_rank`,
`bm25_rank` and `rrf_score` are diagnostics and are not printed in ordinary
result or citation output. Source, page, block type, block index and chunk
provenance remain available to citations.

CLI behavior:

| Invocation | Behavior |
| --- | --- |
| No retrieval selector | Dense, no reranker |
| `--mode dense` | Legacy Dense, no reranker |
| `--mode reranker` | Legacy Dense + BGE |
| `--mode compare` | Legacy Dense vs. Dense + BGE comparison |
| `--retriever hybrid` | Hybrid, no reranker |
| `--retriever hybrid --rerank` | Hybrid Top-20 + BGE Top-3 |

The new selector also supports `--retriever dense` and
`--retriever dense --rerank`. Combining `--mode` with either new selector is a
CLI error. Dense default execution does not initialize BM25.

## Historical retrieval results

The frozen retrieval measurements motivating the opt-in path are:

| Dataset | Metric | Dense + BGE | Hybrid + BGE |
| --- | --- | ---: | ---: |
| M9.1, 32 questions | Top-3 page hit | 19/32 | 23/32 |
| M9.1, 32 questions | Normalized evidence hit | 14/32 | 17/32 |
| M9.2, 21 questions | Top-3 page hit | 17/21 | 17/21 |
| M9.2, 21 questions | Normalized evidence hit | 12/21 | 15/21 |
| M9.2, 21 questions | Recall@20 | 12/21 | 16/21 |

M9.2 had two page regressions. The measured changes support making Hybrid
available for controlled use; they do not show a universal improvement or
establish BGE as the only remaining bottleneck.

## Local smoke checks

The real PDF CLI was run against cached local MinerU output and local models.
Only anonymous status fields were saved under the ignored
`outputs/m9_optional_hybrid_integration/` directory. Raw questions, answers,
PDF names and document text were not written to the smoke summary or tracked
files.

| Case | Path | Evidence in Top-3 context | Citation scope | Other check |
| --- | --- | --- | --- | --- |
| Q01 | Dense + BGE | No | No | CLI completed; answer non-empty |
| Q01 | Hybrid + BGE | Yes | Yes | Page/block metadata present; diagnostics hidden |
| Q06 | Dense + BGE | Yes | Yes | Page/block metadata present |
| Q06 | Hybrid + BGE | Yes | Yes | Page/block metadata present; diagnostics hidden |
| Q33, unanswerable | Hybrid + BGE | No target evidence applies | Not applicable | Existing refusal phrase present; diagnostics hidden |

Every invoked path exited successfully, produced a non-empty answer and
retained page, block type and block index in citations. The existing refusal
phrase appeared for Q33. The MinerU cache digest was unchanged. These checks
verify application wiring, evidence entering the prompt context, citation
provenance and the existing refusal path. They are not a generation-quality
benchmark and do not establish answer correctness.

## Compatibility and limits

Focused tests cover the legacy modes, new selector combinations, mixed-selector
errors, lazy BM25 initialization, one-index-per-session reuse, BGE candidate
pool, Hybrid metadata preservation and citation score hiding. The Dense
citation's score field remains on by default. No retrieval parameters,
embedding, chunking, MinerU processing, prompt, reranker or generation model
were changed.

Dense without reranking remains the default. Hybrid is an optional controlled
path. The M9.3 smoke cases are too few to support a new retrieval or generation
quality claim.
