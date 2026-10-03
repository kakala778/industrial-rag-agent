# Agent 1 Semantic Evidence Comparison Implementation Plan

> **For agentic workers:** Use inline execution in the current checkout. Steps use checkbox syntax for tracking.

**Goal:** Add an opt-in semantic comparator over saved Agent 0.3 host-accepted references, evaluate one frozen DeepSeek Flash pass, and report limits without changing retrieval or Agent 0.

**Architecture:** Keep the comparator independent from the selector and retrieval stack. A JSON-only runner validates frozen inputs, routes unsupported pairs deterministically, attaches saved host references, and calls DeepSeek only for bilateral supported evidence. Provisional semantic GT lives only under ignored `outputs/` and is hashed before inference.

**Tech Stack:** Python standard library, `unittest`, `urllib`, DeepSeek Chat Completions JSON Output.

**Spec:** User-provided Agent 1 milestone specification in this task attachment.

## Global Constraints

- Do not modify RAG, Agent 0 state/guards, evidence IDs, host renderer, repaired Agent 0 GT, or the DeepSeek selector.
- Only compare saved, host-accepted Agent 0.3 excerpts; never search, re-render, retry inference, or let the model create citations.
- Freeze provisional semantic labels and input hashes before any API call; authority remains AI-assisted provisional.
- `deepseek-flash`, non-thinking, temperature 0, JSON Output, environment-only key, one request per eligible task, soft stop ¥3 and hard ceiling ¥5.
- The exact model response is `{verdict, dimensions, reason, notes}`; no engineering recommendation, source quote, evidence ID, citation, operator score, or modality score is produced.
- Keep industrial inputs, semantic GT, traces, outputs, and credentials out of Git; do not commit, push, merge, or start a later milestone.

## Review Focus

- Malformed/duplicate-key JSON or extra fields must be rejected without repair (`test_strict_semantic_output_rejects_noncanonical_json`).
- Same value across different objects must stay `NOT_COMPARABLE` (`test_numeric_match_does_not_override_object_mismatch`).
- Decimal formatting may normalize `5` and `5.0`, but signs/operators/units cannot be guessed or converted (`test_decimal_normalization_is_conservative`).
- Unsupported, retrieval-bound, or unresolved-scope tasks must bypass the API (`test_preflight_routes_without_model_call`).
- Saved references must match accepted FINISH IDs and hashes; forged, cross-scope, or altered excerpts must stop the run (`test_only_saved_host_accepted_references_are_forwarded`).

---

### Task 1: Strict semantic comparator contract

**Files:**
- Create: `src/agent/semantic.py`
- Create: `tests/test_agent1_semantic.py`

**Interfaces:**
- Produces `parse_semantic_output(text) -> dict`, `validate_semantic_output(value) -> dict`, `normalize_decimal(value: str) -> Decimal | None`, and `numeric_values_equal(left: str, right: str) -> bool`.
- Model verdicts are `EQUIVALENT`, `DIFFERENT`, or `NOT_COMPARABLE`; host preflight alone emits `INSUFFICIENT_EVIDENCE`.
- Dimension labels are `aligned`, `different`, `not_applicable`, or `uncertain`; exact dimensions are `object_or_field`, `value`, `unit`, and `condition_or_applicability`.
- Reject invalid JSON, duplicate keys, extra keys, wrong types, unbounded reason/notes, and any model-supplied citation/ID field; never repair output.

- [x] Write failing tests for the three model verdicts, exact keys, duplicate keys, invalid JSON, forbidden IDs/citation fields, and conservative decimal normalization; Task 2 tests host-only `INSUFFICIENT_EVIDENCE`.
- [x] Run the focused test file and confirm failures are from the missing API.
- [x] Implement only the parser/schema/normalization contract.
- [x] Run the focused tests and confirm they pass.

### Task 2: Frozen-input runner, deterministic preflight, and metrics

**Files:**
- Create: `evaluation/agent1_semantic_metrics.py`
- Create: `evaluation/evaluate_agent1_semantic.py`
- Create: `tests/test_agent1_evaluation.py`

**Interfaces:**
- Produces a pure-JSON runner with separate `freeze-gt` and `run` modes; it imports no retrieval/session/selector modules.
- `preflight_pair(task_id, saved_result) -> dict` returns an `INSUFFICIENT_EVIDENCE` decision unless both A and B have accepted supported references.
- `score_semantic_run(results, ground_truth) -> dict` reports task/verdict/dimension, unsupported routing, invalid output, citation grounding, and API usage metrics.
- The runner verifies exact task IDs and SHA-256 bindings for frozen task/repaired GT/reviews/Agent 0.3 results/pricing and protected inputs; outputs remain ignored under `outputs/agent1/`.
- Error taxonomy distinguishes upstream evidence, field alignment, value, unit, condition, false-positive/false-negative `NOT_COMPARABLE`, invalid output, and uncertain GT.

- [x] Add failing tests for bilateral eligibility, unsupported routing, integrity/hash failures, accepted-ID/reference matching, one-shot API behavior, costs, unit-conversion exclusion, and all requested synthetic cases.
- [x] Run focused tests and confirm expected failures.
- [x] Implement preflight, strict API transport, frozen-input/GT guards, output persistence, and metrics without touching Agent 0 code.
- [x] Run focused tests and confirm they pass.

### Task 3: Freeze semantic GT and run one bounded provider pass

**Files:**
- Create ignored: `outputs/agent1/semantic-gt.local.json`
- Create ignored: `outputs/agent1/results.local.json`

- [x] Audit the four bilateral supported pairs from saved excerpts and task intent; write provisional verdicts/dimensions with confidence and source hashes.
- [x] Freeze and hash GT offline; verify the runner refuses missing/changed GT and the four unsupported pairs route without API calls.
- [x] Confirm `DEEPSEEK_API_KEY` availability without printing it.
- [x] Run exactly one non-retrying DeepSeek Flash request per eligible task under a ¥3 soft reservation and ¥5 hard cap; never persist the key or raw HTTP error bodies.
- [x] Re-check frozen input and GT hashes after inference; calculate all metrics from saved responses.

### Task 4: Report and current-status documentation

**Files:**
- Create: `docs/agent1-semantic-evidence-comparison.md`
- Modify: `README.md`
- Modify: `docs/agent-handoff.md`

- [x] Write aggregate-only results answering all 16 questions from the spec, separating measured results, provisional-GT judgments, and limitations.
- [x] Update current README/handoff status without rewriting historical Agent 0 reports or starting a later milestone.
- [x] Review final diff and privacy boundaries; run compileall, full unittest discovery, `git diff --check`, ignore/status checks, and read all outputs.

Final read-only review found no remaining issues. Follow-up validation rejects verdict/dimension conflicts and status/route inconsistencies; the frozen one-pass result still rescored unchanged.
