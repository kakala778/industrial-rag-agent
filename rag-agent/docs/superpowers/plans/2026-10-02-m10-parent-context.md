# M10.1 Parent Context Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline, task by task. No commit, push, merge or tag.

**Goal:** Test only bounded parent text in BGE input, with frozen Hybrid20 identities and early stopping.

**Architecture:** Replay stored M9 candidate ordinals after validating inputs, corpus identities and retrieval core hashes. A small helper maps children to original structured Documents without QA access. Rerank temporary expanded copies, then restore original child objects for evaluation.

**Tech Stack:** Existing Python/unittest, cached Sentence Transformers BGE, existing evaluator and MinerU cache; no installation or new models.

**Spec:** User M10.1 request; docs/m10-architecture-review-and-agent-readiness.md sections 7–8.

## Constraints and review focus

- No embedding computation, index rebuild, candidate expansion, parser/model/QA/matcher change.
- Existing review README/document edits are preserved; reuse current feature checkout for offline additions.
- Mapping must be unique at source/page/type/block and exact child occurrence, otherwise retain child and report insufficiency.
- Exact overlapping rows and verified serialized header only; no QA-dependent row selection.
- Token cap includes query and special tokens; never truncate child silently or guess an oversized row.
- Temporary expanded text must not leak into retrieved candidates, scores are attached to original child identity.
- All data outputs stay Git ignored; public report only anonymous case IDs and statistics.

## Task 1 — Read-only Q25 mapping and helper

- [x] Prove unique parent and child occurrence using frozen corpus. Stop if mapping is insufficient.
- [x] Add tests/test_parent_context.py first: deterministic mapping, restored rows/header, cap, missing/ambiguous parent and occurrence, provenance and unchanged child.
- [x] Observe expected missing-module failure; add src/parent_context.py with ParentContextIndex.expand(child, query, tokenizer, max_pair_tokens).
- [x] Token cap: single conservative 768-token total pair budget, also capped by finite model/tokenizer limit. No repeated tuning; fallback to child if required rows/child cannot fit.
- [x] Keep table rows intersecting child, not answer-selected rows; text uses bounded same-parent neighborhood. Return text/context_type/status/provenance only.

## Task 2 — Offline controlled runner

- [x] Add tests/test_m10_parent_context_experiment.py for original-child restoration, candidate replay validation, output restrictions and baseline matching.
- [x] Add evaluation/run_m10_parent_context_experiment.py. Load frozen inputs cache-only; verify M9.1 inventory/core and M9.2 manifest/corpus. Explain known M9.3 generation-only hash change separately.
- [x] Reconstruct Hybrid20 from stored ordinals, verify identities and stored evidence ranks; never load embedding model or rebuild BM25.
- [x] Load cached BGE offline with original device/config; record tokenizer/pair lengths, expansion time and A/B runtime. Execute Q25 first and require A rank7 reproduction.
- [x] If Q25 rank does not improve, save STOPPED_NO_Q25_RANK_GAIN and stop both cohorts. Otherwise rerank both 32/21 cohorts once; evaluate original children with old matcher; record transitions and all candidate rank movements.
- [x] Save private detailed contexts only under outputs/m10_parent_context, with allowlisted anonymous summary. Q17 sensitivity separate. I17 diagnosis does not imply mechanism equivalence.

## Task 3 — Evidence review and closeout

- [x] Inspect Q25 original PDF/table and contexts locally for identifier/value/unit/header association. Evaluate controls only if stop gate permits; do not invent paired results after stop.
- [x] Write docs/m10-parent-context-controlled-experiment.md with all 15 requested answers, including NOT_RUN when applicable.
- [x] Brief README completion status; no default integration. Final decision ENTER AGENT or PARENT CONTEXT ADOPTED, THEN ENTER AGENT.
- [x] Run compileall, full unittest discovery, git diff --check/status; check no PDF/QA/raw contexts/cache/outputs tracked. Suggest experiment commit only.

## Execution ledger

- Initial inventory is unchanged. M9.3 intentionally changed rag_demo.py citation display; retrieval/model input files remain frozen. Do not reject replay solely for an unused generation helper hash, but protect current core before/after.

- Q25 mapping unique; restored boundary rows instead of repeating all interior rows. Leading first three serialized lines are not automatically semantic headers; Q25/I17 headers reviewed against PDF.
- Q25 A7 -> B1; full 32/21 paired run completed, A full rankings reproduced. Stricter target-expansion gate added during review; saved Q25 output passes it without changed input/scoring.
- Evidence: original17/32 ->17/32 (Q25 gain/Q04 regression), independent15/21 ->16/21 (I17 gain). ENTER AGENT; no application adoption or follow-up RAG experiment.
- Fresh reviewer found no remaining Critical/Important issues. Intermediate output is not checkpointed; an interrupted run must restart (no claim of resume support).
