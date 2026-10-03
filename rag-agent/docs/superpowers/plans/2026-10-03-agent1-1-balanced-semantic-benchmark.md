# Agent 1.1 — Balanced Semantic Benchmark & Contract Validation

## Goal and boundaries

Build a separate, local-only 12–18 item benchmark from original industrial PDFs; decide and test the semantic contract before one bounded DeepSeek Flash pass. Preserve Agent 1's frozen inputs/results and all Agent runtime, retrieval, evidence, progress, coverage, budget, and RAG behavior. Do not commit or push.

## Contract decision

Use independent dimension observations before the verdict. `NOT_APPLICABLE` means the requested dimension itself is absent from one or both evidence sides; an object mismatch never makes value or unit automatically not applicable. The A/B scopes explicitly named by a comparison request are its intended comparison axis, not a comparability failure by themselves. Distinguish a comparable fact with changed value/unit/applicability (`DIFFERENT`) from different objects/fields or an incompatible scope/installation scenario (`NOT_COMPARABLE`) using a required `comparability_basis` field.

Keep the v1 comparator and its historical replay semantics unchanged by making the existing DeepSeek transport accept optional prompt/parser strategies whose defaults remain v1. Agent 1.1 supplies its own prompt, exact output validator, evaluator, fixtures, and runner.

## Implementation sequence

1. Reconfirm the canonical checkout, branch, HEAD, upstream and clean status; preserve the original eight-PDF inventory and local source hashes.
2. Render and inspect each selected original-PDF page. Curate 17 high-confidence tasks (target distribution 5/6/6), exact excerpts, four dimension labels, source-page provenance, and `ai_pdf_reviewed` authority in ignored local output only.
3. Add contract schema/prompt tests first: independent labels, N/A semantics, class consistency, incompatible installation context, malformed output, and no IDs/citations.
4. Implement Agent 1.1 contract validation and reusable comparator strategy injection, plus offline scoring for per-class/per-dimension accuracy, verdict confusion, consistency, and error taxonomy.
5. Add a fail-closed runner that checks original-PDF and excerpt hashes, freezes the new GT once, and cannot overwrite any freeze or results. Freeze GT before any provider call.
6. Run exactly one DeepSeek Flash pass with temperature 0, thinking disabled, JSON output, one request per eligible task, no retry, and the existing ¥3 soft / ¥5 hard cost controls. Bind results to the frozen hashes.
7. Offline-rescore, write the 20-question report and explicit A/B readiness decision, verify the old Agent 1 artifacts and source PDFs remain unchanged, run focused and full tests, compileall, and `git diff --check`.

## Acceptance criteria

- 12–18 reviewed original-PDF tasks with at least four high-confidence cases per verdict class.
- Original PDFs are reviewed visually; any uncertain/unsupported case is excluded and disclosed.
- GT is hash-frozen before inference and never revised from model output.
- Synthetic contract tests prove object mismatch can coexist with aligned value/unit and that N/A is not an object-mismatch fallback.
- New metrics include overall and per-class verdict accuracy, four dimension accuracies, 3×3 confusion matrix, deterministic verdict-dimension consistency, invalid output count, and error taxonomy.
- Same DeepSeek model/settings and cost limits; usage/latency/cost recorded; private artifacts remain ignored.
- Historical Agent 1 results remain unchanged; no Agent 2 implementation, commit, push, merge, or worktree.
