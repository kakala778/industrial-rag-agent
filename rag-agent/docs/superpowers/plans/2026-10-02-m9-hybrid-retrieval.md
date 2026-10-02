# M9.1 controlled Hybrid Retrieval experiment

The supplied M9.1 specification authorizes this offline experiment. M8 is frozen.

## Fixed design

- Native in-memory BM25: k1=1.2, b=0.75, positive log IDF; no dependencies.
- NFKC/casefold; intact Latin/digit identifiers with internal separators; Chinese unigrams/bigrams; no QA-dependent vocabulary.
- Dense Top-20 + BM25 Top-20, unique structural chunk identities, equal-weight RRF k=60, truncate to 20 before the unchanged BGE.
- Reuse the M6/M7 evaluator through an optional ranked-candidate provider. Default dense behavior remains identical.
- Separate offline CLI: --retriever dense|bm25|hybrid, --rerank on|off, or --compare-all for the five fixed arms.
- Keep all 32 scored M7 rows for comparability; flag the existing Q17 GT uncertainty separately. Do not edit QA.

## Execution

1. Test BM25 formula, bilingual/identifier tokens, empty/no-overlap inputs, RRF, structural identity, metadata and default evaluator behavior.
2. Implement lexical retrieval and an optional evaluator provider; retain every evidence-matching function.
3. Implement a cache-only runner with pinned input hashes, original M7 IDs, five arms, Recall@20, regressions, timings and anonymous diagnostics.
4. Run the real eight-PDF M6 corpus once with fixed parameters; review four original retrieval failures, new gains and regression candidates locally.
5. Write the anonymous report and README status. Validate compileall, the full unittest suite, all tracked/new diffs, privacy and input integrity.

## Review focus

- chunk_id resets for each Document: identity must include source/page/block/type.
- No score-scale mixing, duplicate rank contribution or extra reranker candidates.
- Candidate-pool recovery and Top-3 evidence success are reported separately.
- Existing raw evidence can match across joined texts; normalized matching remains the original per-result matcher.
- Out-of-scope/incomplete candidates are diagnostic signals, not automatic proof of semantic irrelevance.
- No raw question, answer, evidence, text, filename, arbitrary metadata or query tokens in JSON or public reports.

Report files/results and the recommended commit message before any commit or push. No merge or tag.
