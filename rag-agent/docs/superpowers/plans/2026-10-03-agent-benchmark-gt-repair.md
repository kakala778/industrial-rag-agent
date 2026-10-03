# Agent Benchmark GT Repair and Offline Rescore Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the PDF audit's confirmed GT and candidate-label corrections to derived local inputs, then recompute Agent 0.2 and 0.3 metrics solely from saved actions and frozen observations.

**Architecture:** Leave the frozen manifest, original review sheets, candidate reviews, and run results untouched. Build ignored local derived GT/review artifacts from the original review sheet plus an explicit patch manifest; make evaluation ID correctness depend on both repaired expected IDs and actual frozen candidates. A JSON-only offline runner produces aggregate results without loading PDFs, models, or retrieval code.

**Tech Stack:** Python standard library, existing offline evaluation metrics, `unittest`.

**Spec:** User attachment `Agent Benchmark GT Repair & Offline Rescore` (2026-10-03), plus `rag-agent/docs/agent0-3-bounded-evidence-contract.md` and the WorkBuddy PDF audit report/review notes.

## Global Constraints

- Audit authority remains `ai_assisted_original_pdf_review`; no `human_verified` claims.
- Preserve all frozen source artifacts and old Agent 0.2/0.3 results byte-for-byte.
- Do not call APIs or run models, retrieval, MinerU, embedding, BM25, or BGE.
- Keep derived manifests, reviews, and rescored output under ignored `outputs/`.
- Do not change RAG/runtime harness/provider/prompt/citation/progress code; do not commit or push.

## Review Focus

- A listed expected ID can be absent from frozen SEARCH candidates; it must remain `RETRIEVAL_BOUND`.
- A semantically invalid expected ID must not count as a correct selection even if a stale candidate label says `RELEVANT`.
- A negated applicability condition must not match its affirmative counterpart.
- A `PARTIAL` candidate must not be treated as complete expected evidence.
- Rescoring the same source artifacts twice must produce identical metric content.

---

### Task 1: GT-aware evidence scoring

**Files:**
- Modify: `rag-agent/evaluation/agent02_metrics.py`
- Modify: `rag-agent/evaluation/agent03_metrics.py`
- Test: `rag-agent/tests/test_agent_benchmark_repair.py`

**Interfaces:**
- Consumes optional per-scope `expected_evidence_ids` and explicit frozen candidate IDs.
- Produces candidate-group and selected-ID results that require a relevant expected ID to be present in frozen candidates and selected.

- [x] Write regression tests for stale irrelevant expected IDs, retrieval-bound missing candidates, affirmative-vs-negated conditions, partial evidence, and repeatable scoring.
- [x] Use a red/green check for the unsupported-scope expected-ID regression; it failed before the fix and passed after it.
- [x] Implement the minimal GT-aware scoring behavior while retaining legacy behavior for old inputs without expected-ID metadata.
- [x] Re-run the focused tests and confirm all pass.

### Task 2: Derived local inputs and deterministic rescorer

**Files:**
- Create: `rag-agent/evaluation/rescore_repaired_agent_benchmark.py`
- Test: `rag-agent/tests/test_agent_benchmark_repair.py`
- Create locally (ignored): `rag-agent/outputs/agent0_3/gt-repair-manifest.local.json`, repaired GT/review inputs, and rescore output.

**Interfaces:**
- Consumes the frozen Agent 0.1 manifest/observations, Agent 0.2 original and reviewed results, Agent 0.3 results, the original Agent 0.3 review sheet, and the explicit repair manifest.
- Produces derived inputs bound to source hashes and aggregate-only deterministic Agent 0.2/0.3 metrics.

- [x] Add tests that derived expected IDs exclude `IRRELEVANT`/`PARTIAL` entries, validate against actual candidate sets, and rescore deterministically.
- [x] Verify the builder/rescorer contract with unit coverage against the repaired artifacts; a separate pre-implementation red result was not retained for every builder assertion.
- [x] Implement before/after patch validation, source-hash validation, derived artifacts, and JSON-only rescore.
- [x] Run focused tests and a local two-pass rescore; compare deterministic metric payloads.

### Task 3: Report repaired benchmark status

**Files:**
- Modify: `rag-agent/docs/agent0-2-evidence-selection-model-comparison.md`
- Modify: `rag-agent/docs/agent0-3-bounded-evidence-contract.md`
- Create: `rag-agent/docs/agent-benchmark-review-status.md`
- Modify: `rag-agent/README.md` only if needed to link the current readiness decision.

- [x] Record before/after metrics, GT contamination history, R02/R04 retrieval bounds, AI-assisted authority, R05 B's medium-confidence document-specific limit, and deferred qualifier gaps.
- [x] Verify no private industrial paths, PDF text, raw answers, candidate excerpts, or full local IDs enter tracked documentation.

### Task 4: Final verification

- [x] Run `python -m compileall src evaluation`.
- [x] Run `python -m unittest discover -s tests`.
- [x] Run `git diff --check`, verify ignored output artifacts, verify frozen source hashes, and inspect `git status` without staging or committing.
