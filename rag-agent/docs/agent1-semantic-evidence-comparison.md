# Agent 1 — Semantic Evidence Comparison

**Status:** bounded experiment complete; experimental only.  
**Run date:** 2026-10-03.  
**Ground-truth authority:** AI-assisted reviewed provisional GT. It is not an independently human-verified semantic benchmark.

Agent 1 compared only evidence excerpts already accepted and rendered by the saved Agent 0.3 run. It did not search, rerender documents, or change Agent 0, retrieval, the repaired Agent 0 GT, or the DeepSeek selector. The semantic labels and source hashes were frozen before the single provider pass and verified unchanged afterward. Four of the eight tasks reached DeepSeek Flash; the other four were stopped by host preflight.

## Measured result

| Measure | Result |
| --- | ---: |
| Frozen tasks | 8 |
| Tasks sent to semantic model | 4 |
| Deterministic `INSUFFICIENT_EVIDENCE` preflight | 4/8: two retrieval-bound, one unsupported side, one unresolved scope |
| Verdict accuracy | 4/4 (100%); all four expected and returned verdicts were `NOT_COMPARABLE` |
| Object/field alignment | 4/4 (100%) |
| Value alignment | 1/4 (25%) |
| Unit alignment | 1/4 (25%) |
| Condition/applicability alignment | 3/4 (75%) |
| Invalid semantic outputs | 0 |
| Unsupported-side routing | 1/1 correct |
| Saved accepted host references retained and matched | 11/11 |

The 100% verdict score is not a useful estimate of general verdict accuracy: this cohort contains only `NOT_COMPARABLE`, so the run did not test whether the model distinguishes `EQUIVALENT` from `DIFFERENT`. It did identify all four expected `NOT_COMPARABLE` cases, but the benchmark cannot measure performance on the other verdict classes.

The clearest model failure was dimension handling. In three of four comparisons, value and unit labels disagreed with the frozen provisional GT; the model often returned `not_applicable` for those dimensions when it judged the objects or fields different. The saved error taxonomy records three value-comparison errors, three unit-comparison errors, and one condition omission. Object/field alignment had no errors. This shows why the headline verdict alone would overstate the result.

Citation accounting checked that the saved Agent 0.3 accepted references still matched their FINISH scopes, evidence IDs, excerpt hashes, and provenance. The renderer was not called again, and this was not an independent PDF authenticity or full-document review. The semantic model received no evidence IDs or citation fields and could not create citations.

## Provider usage and limits

The one non-retrying pass made four API requests and used 3,671 prompt tokens plus 701 completion tokens, 4,372 total. Of the prompt tokens, 384 were cache hits and 3,287 were cache misses. Mean request latency was 1,367.679 ms and the maximum was 1,607.2 ms.

Usage-based cost calculated from the saved provider token counts and the frozen official DeepSeek Flash rates was **¥0.01219736**. The sum of conservative per-request reservations was **¥0.077376**. These are experiment-side estimates, not a billing-console statement. The pricing snapshot used the [official DeepSeek pricing page](https://api-docs.deepseek.com/zh-cn/quick_start/pricing); rates can change.

The four API payloads contained each frozen comparison request and its accepted
evidence excerpts; those contents were sent to DeepSeek. Evidence IDs and
citations were omitted, and the API key was read from `DEEPSEEK_API_KEY` without
being written to the result files.

Two tasks remained retrieval-bound and were excluded from semantic-model accuracy. One task had an unsupported side and one had unresolved scope; both were kept out of model inference. The only unsupported-side example routed correctly. These are upstream evidence and scope limits, not semantic-model failures.

One medium-confidence provisional condition label disagreed with the model output. That is an adjudication flag for a future independent review, not grounds to alter GT after seeing the model result. No GT was changed after inference. The semantic GT remains AI-assisted and provisional.

This pass establishes that the bounded comparator, strict output validation, preflight routing, and saved-reference checks can be exercised end to end. It does **not** establish reliable semantic comparison, engineering equivalence, standards compliance, or industrial decision quality. In particular, an all-`NOT_COMPARABLE` cohort cannot validate verdict discrimination, and dimension results expose a material model weakness.

## Answers to the 16 milestone questions

1. **Tasks entering the semantic model:** 4 of 8.
2. **Tasks rejected by deterministic preflight:** 4 of 8—two retrieval-bound, one unsupported-side, and one unresolved-scope task.
3. **Verdict accuracy:** 4/4. All labels were `NOT_COMPARABLE`; this single-class result does not measure three-way discrimination.
4. **Object/field alignment:** 4/4.
5. **Value alignment:** 1/4.
6. **Unit alignment:** 1/4.
7. **Condition/applicability alignment:** 3/4.
8. **Most frequent semantic failure:** value and unit dimensions were each wrong in three cases, commonly labeled `not_applicable` after an object/field mismatch. The output contract needs dimensions to be judged independently.
9. **`NOT_COMPARABLE` detection:** 4/4 on the four expected cases. The result is encouraging only for this narrow cohort; no `EQUIVALENT` or `DIFFERENT` cases were present.
10. **Unsupported-side routing:** 1/1 correct. The other three preflight rejections were separately classified as retrieval-bound or unresolved scope.
11. **Citation grounding:** 11/11 saved accepted references passed ID, scope, hash, and provenance checks. No fresh PDF rendering or independent source audit was performed in this milestone.
12. **DeepSeek usage:** four requests; 3,671 prompt tokens, 701 completion tokens, 4,372 total; estimated usage cost ¥0.01219736 and conservative reserved cost ¥0.077376.
13. **Upstream evidence failure:** two retrieval-bound tasks remained unresolved and were not counted as semantic failures.
14. **Provisional-GT review:** one medium-confidence condition label merits independent adjudication. The pass did not establish a GT error, and labels were not changed post hoc.
15. **Does this prove Agent 1 is worth continuing?** It validates the experimental plumbing and shows that semantic comparison is testable, but it does not establish useful overall accuracy. The dimension errors argue for improving the evaluation contract before expanding capability.
16. **Next direction:** if the project continues this line, first build a small, balanced, independently adjudicated semantic benchmark covering all three verdicts and each dimension. Defer unit conversion, operator/modality semantics, and cross-document research until that benchmark shows the current contract works. Otherwise stop the semantic route; do not treat this experiment as authorization for industrial decisions.

## Reproduction

The current saved pass can be checked again from `rag-agent/` without a provider call:

```powershell
python evaluation/evaluate_agent1_semantic.py rescore
```

For a fresh local setup with the matching private frozen inputs and
`DEEPSEEK_API_KEY`, the one-pass sequence is:

```powershell
python evaluation/evaluate_agent1_semantic.py freeze-gt
python evaluation/evaluate_agent1_semantic.py run
python -m compileall src evaluation
python -m unittest discover -s tests
git diff --check
```

The runner refuses to overwrite existing frozen GT or inference results. Do not
delete the current local freeze/result to repeat this pass.

Private PDFs, QA, excerpts, semantic GT, traces, and model outputs remain under ignored local output paths. Do not commit those artifacts or API credentials.
