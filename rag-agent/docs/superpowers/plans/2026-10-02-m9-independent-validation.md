# M9.2 Independent Validation + Ranking Audit

The user supplied and authorized the fixed experiment design. Work stays on
codex/m9-hybrid-retrieval; no commit/push until review.

1. Select PDF pages without reading new-query retrieval results. Visually review
   facts and create 15–30 new local questions, covering all eight sources.
   Reject copied M6 queries/facts and unreliable ground truth; record exclusions.
2. Freeze QA, categories, anonymized source counts, PDF/cache inventory and M9.1
   implementation hashes before model inference. Require identical hashes after.
3. Add one offline runner reusing M9.1 BM25/RRF, dense/BGE and original evidence
   scoring. Five arms, fixed20/20/20/3, no model downloads or new dependencies.
   Add focused tests for freeze/config/privacy/audit schema before implementation.
4. Reconstruct Q07/Q19/Q24/Q25 Top20 using original corpus ordinals and anonymous
   M9.1 results; verify corpus/hash identity. Inspect private candidate text and
   original PDFs. Distinguish observed misorder from uncertain causal explanation.
   Quantify exact and near textual duplicates without altering retrieval.
5. Record metrics, complementarity, both regression directions and CPU timings.
   Report YES/NO/NEED MORE DATA from independent results, not prior preference.
6. Run compileall, complete unittest suite, whitespace/privacy/core-diff checks;
   review final report. Detailed data remains in ignored outputs only.
