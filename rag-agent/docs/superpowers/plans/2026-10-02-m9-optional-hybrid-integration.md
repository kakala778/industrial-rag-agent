# M9.3 Optional Hybrid Retrieval Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the frozen Hybrid strategy as an explicit PDF RAG option while preserving Dense as the unchanged default and preserving the existing generation path.

**Architecture:** Add a thin `hybrid_retrieval.py` orchestration function that calls the existing Dense retriever and the single BM25/RRF implementation in `lexical_retrieval.py`. Separate retriever selection from optional reranking in the PDF CLI, preserve legacy `--mode` behavior, and keep retrieval diagnostics out of ordinary Hybrid citations.

**Tech Stack:** Python 3.14, NumPy, Sentence Transformers, Ollama, `unittest`, PowerShell for local smoke orchestration.

**Spec:** `rag-agent/docs/superpowers/specs/2026-10-02-m9-optional-hybrid-integration-design.md`

## Global Constraints

- BM25 stays `k1=1.2`, `b=0.75`; candidate budgets stay Dense Top-20 + BM25 Top-20; RRF stays `k=60`; Hybrid stays Top-20; BGE remains the existing model and returns Top-3.
- `lexical_retrieval.py` remains the only implementation of tokenizer, BM25, structural chunk identity, deduplication, and RRF.
- Dense and Hybrid result dictionaries share `text`, `score`, `source`, `chunk_id`, and `metadata`, but Dense `score` is cosine similarity while Hybrid `score` is RRF; never compare them as a shared relevance value. Hybrid `dense_rank`, `bm25_rank`, and `rrf_score` are diagnostics only.
- Dense is the default; do not construct a BM25 index unless Hybrid is selected, and reuse one index for that PDF session.
- Preserve `--mode dense|reranker|compare`; reject mixed legacy and new selectors.
- Do not change the embedding model, parser/MinerU pipeline, chunking, prompt, Ollama endpoint/model, refusal rules, or generation format.
- Smoke only Q01/Q06/Q33; treat them as wiring/context/citation/refusal checks, not a generation-quality benchmark.
- Keep PDFs, QA, answers, logs, outputs, caches, and model files ignored/untracked. Do not commit, push, merge, tag, or create a PR in this task.
- Document M9.1 as Dense+BGE → Hybrid+BGE Top-3 page 19/32 → 23/32 and evidence 14/32 → 17/32; document M9.2 as page 17/21 → 17/21 (two page regressions), evidence 12/21 → 15/21, and Recall@20 12/21 → 16/21.

## Review Focus

1. No lexical overlap must preserve Dense-only RRF candidates; test it alongside a truly empty corpus in Task 1 (`test_no_bm25_overlap_keeps_dense_candidates`, `test_empty_corpus_returns_no_candidates`).
2. Reused chunk IDs across pages/blocks must preserve structural identity and metadata; test duplicate fusion and metadata in Task 1 (`test_fusion_deduplicates_by_full_chunk_identity`) and citation fields in Task 2 (`test_hybrid_citation_keeps_source_page_and_block_fields`).
3. Dense cosine and RRF scores have different meanings; test that Hybrid diagnostics survive internally but do not appear in ordinary Hybrid citations in Task 2 (`test_hybrid_citation_hides_backend_scores`).
4. CLI defaults, legacy mappings, all four new combinations, conflicts, and lazy BM25/BGE initialization must be explicit; test each in Task 2 (`test_default_is_dense_without_reranker`, `test_legacy_modes_map_to_existing_behavior`, `test_new_retriever_and_rerank_combinations`, `test_mixed_selector_options_are_rejected`, `test_dense_session_skips_bm25_and_bge`, `test_hybrid_session_reuses_one_index_and_reranks_top20`).
5. The unanswerable smoke can expose weakness in the existing refusal behavior; observe Q33 without changing prompts or claiming answer accuracy in Task 3.

---

### Task 1: Reusable Hybrid orchestration

**Files:**
- Create: `rag-agent/src/hybrid_retrieval.py`
- Create: `rag-agent/tests/test_hybrid_retrieval.py`

**Interfaces:**
- Consumes: `retrieval.retrieve(query_embedding, chunks, embeddings, top_k)`, `lexical_retrieval.BM25Index.search(query, top_k)`, and `lexical_retrieval.rrf_fuse(dense_results, bm25_results, top_k)`.
- Produces: `retrieve_hybrid(query: str, query_embedding, chunks: list[dict], embeddings, bm25_index: BM25Index) -> list[dict]`; it requests 20 candidates from each retriever and returns the fixed Hybrid Top-20.

- [x] **Step 1: Write failing orchestration tests** for the 20/20/20 budgets, dense-only fallback when lexical search is empty, unique structural candidates with preserved source/page/block/chunk metadata and rank diagnostics, and empty-corpus behavior. Use real `BM25Index`/RRF with small in-memory fixtures; do not duplicate their algorithms in test helpers. Before the module exists, resolve its API through a test assertion so the expected RED is an assertion failure rather than an import error.
- [x] **Step 2: Run `python -m unittest discover -s tests -p test_hybrid_retrieval.py` from `rag-agent/`** and confirm the assertion fails because the orchestration API is absent.
- [x] **Step 3: Implement only the thin orchestration function** in `hybrid_retrieval.py`; keep the fixed candidate constant local to orchestration and delegate lexical work to `lexical_retrieval.py`.
- [x] **Step 4: Re-run the focused test command** and confirm every test passes, including no-overlap and empty-corpus behavior.

### Task 2: CLI integration, lifecycle, and citation compatibility

**Files:**
- Modify: `rag-agent/src/pdf_rag_demo.py`
- Modify: `rag-agent/src/rag_demo.py`
- Modify: `rag-agent/tests/test_rag_citations.py`
- Create: `rag-agent/tests/test_pdf_rag_demo_modes.py`

**Interfaces:**
- Consumes: Task 1's `retrieve_hybrid(query, query_embedding, chunks, embeddings, bm25_index)`.
- Produces: `resolve_retrieval_options(*, mode: str | None, retriever: str | None, rerank: bool) -> tuple[str, bool, bool]`, returning `(retriever, use_reranker, compare_mode)`; old modes map to Dense/off, Dense/on, and Dense/on/compare respectively.

- [x] **Step 1: Write failing CLI and display tests** named above. Assert these exact mappings: no flags and `--mode dense` → Dense/off; `--mode reranker` → Dense/on; `--mode compare` → Dense/on/compare; `--retriever dense` → Dense/off; `--retriever dense --rerank` → Dense/on; `--retriever hybrid` → Hybrid/off; `--retriever hybrid --rerank` → Hybrid/on. Test each conflict form (`--mode dense --retriever hybrid`, `--mode dense --rerank`) exits with parser status 2. Resolve not-yet-added helpers/options with assertions so the first RED is an assertion failure, not an import/argparse error.
- [x] **Step 2: Add failing citation/output assertions** showing Hybrid source/page/block/chunk metadata remains present while RRF/reranker diagnostics and the Hybrid `score` are hidden from ordinary Hybrid citations; Dense citation formatting remains backward compatible.
- [x] **Step 3: Run `python -m unittest discover -s tests -p 'test_pdf_rag_demo_modes.py'` and `python -m unittest discover -s tests -p test_rag_citations.py`** from `rag-agent/`; confirm the absent resolver/options, lazy index behavior, and citation behavior fail as assertions, not test import/runtime errors.
- [x] **Step 4: Implement selector resolution and wire the selected candidate pool** in `pdf_rag_demo.py`; instantiate `BM25Index` once only for Hybrid and load BGE only when selected. Keep old compare output/generation behavior.
- [x] **Step 5: Add an optional `include_score` argument to `format_sources`**, defaulting to the existing output; use it to hide Hybrid score fields in ordinary PDF citations. Show the chosen retriever for Hybrid, without adding debug ranks to regular output.
- [x] **Step 6: Re-run both focused test commands** and confirm all selector, lifecycle, reranker-pool, metadata, and citation assertions pass.

### Task 3: Documentation, real-PDF smoke, and final verification

**Files:**
- Modify: `README.md`
- Modify: `rag-agent/README.md`
- Create: `rag-agent/docs/m9-optional-hybrid-integration.md`
- Local only: `rag-agent/outputs/m9_optional_hybrid_integration/` smoke logs/driver; ensure Git ignores it.

**Interfaces:**
- Consumes: completed CLI options and stable result/citation behavior from Task 2.
- Produces: a concise implementation note with anonymous smoke outcomes and the documented optional Hybrid support/status.

- [x] **Step 1: Map the frozen local M6 rows by ordinal only** and resolve their existing local PDF/cache inputs for Q01, Q06, and Q33; keep raw question/answer/document values out of terminal summaries and tracked files.
- [x] **Step 2: Run Q01 against Dense+BGE and Hybrid+BGE**, then Q06 against Dense+BGE and Hybrid+BGE, and Q33 against Hybrid+BGE using the real PDF CLI, local MinerU caches, cached embedding/BGE models, and local Qwen3:4b. Set Hugging Face offline environment flags so no model is downloaded. Save detailed outputs only under ignored `outputs/`; inspect only non-empty output, evidence context/citation presence, expected source/page scope, and whether the existing refusal appears for Q33.
- [x] **Step 3: Update both READMEs** with optional Hybrid support and these limits: M9.1 page 19/32 → 23/32 and evidence 14/32 → 17/32; M9.2 page 17/21 → 17/21 with two regressions, evidence 12/21 → 15/21, Recall@20 12/21 → 16/21. Keep Dense the stated default and avoid universal-improvement claims.
- [x] **Step 4: Write the concise M9.3 implementation note** with why optional, architecture/API/CLI, backward compatibility, anonymized smoke statuses, and known limitations. Do not include private questions, answers, PDF names, or document text.
- [x] **Step 5: Run `python -m compileall src evaluation`, `python -m unittest discover -s tests`, and `git diff --check`** from `rag-agent/`; read all outputs and record test count/result.
- [x] **Step 6: Verify `git status`, `git ls-files` for prohibited data/model/output extensions, and ignore rules**; confirm only intended source/docs/tests are untracked/modified and no private input/output is tracked. Report the recommended commit message without committing or pushing.
