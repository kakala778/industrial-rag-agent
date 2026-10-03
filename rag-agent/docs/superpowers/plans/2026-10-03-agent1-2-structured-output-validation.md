# Agent 1.2 Structured Output Validation Implementation Plan

> **For agentic workers:** Execute inline in the current checkout; do not create a worktree, commit, or push.

**Goal:** Determine whether DeepSeek Responses `json_schema` changes Agent 1.1's 1/17 host-accepted output rate while keeping all semantic inputs and host checks frozen.

**Architecture:** Add a sibling Responses transport that reuses `semantic_messages_v2`, `parse_semantic_output_v2`, and `SemanticCostBudget`. Add an offline A/B scorer and a one-pass runner that reads Agent 1.1 as immutable Arm A and keeps raw provider bodies only in ignored Agent 1.2 outputs.

**Tech Stack:** Python standard library, `urllib`, `unittest`, DeepSeek Responses API.

**Spec:** User attachment, Agent 1.2 — Structured Output Contract Validation (2026-10-03).

## Global Constraints

- Keep the same 17 frozen tasks, GT, evidence excerpts, model, semantic prompt, verdicts, dimensions, NOT_APPLICABLE semantics, consistency rules, RAG, Agent 0, and citations.
- The only experimental variable is Chat Completions `json_object` versus Responses `text.format=json_schema`.
- Always run host schema and verdict/dimension consistency validation after provider output.
- Keep API key and all raw/private outputs out of tracked files; default outputs remain under ignored `outputs/agent1_2/`.
- Run only one Agent 1.2 pass; do not rerun paid Agent 1.1, start Agent 2, commit, or push.
- Enforce the existing ¥3 soft and ¥5 hard cost ceilings.

## Review Focus

- Incomplete Responses must be recorded as `TRUNCATED` and retain their raw body.
- A completed response without assistant `output_text` must not count as accepted.
- HTTP errors must preserve the provider body and request ID without exposing the key.
- Nested missing/extra fields and JSON duplicates must be classified without repair.
- A runner with any pre-existing result, metrics, or raw file must refuse before making a request.

## Tasks

### Task 1: Contract schema and failure taxonomy

**Files:** Modify `src/agent/semantic_contract_v2.py`; test `tests/test_agent1_2_transport.py`.

**Interface:** Add `semantic_output_json_schema_v2() -> dict`; add `failure_type` while preserving `reason_code` compatibility.

- [ ] Confirm taxonomy and JSON Schema tests fail before implementation.
- [ ] Add a closed JSON Schema matching every v2 key, enum, and nested dimension.
- [ ] Classify parser/validator failures as `INVALID_JSON`, `MISSING_FIELD`, `EXTRA_FIELD`, `INVALID_ENUM`, `WRONG_TYPE`, `NESTED_SCHEMA_ERROR`, `CONSISTENCY_ERROR`, `EMPTY_RESPONSE`, `TRUNCATED`, or `OTHER`.
- [ ] Run focused transport tests.

### Task 2: Responses transport

**Files:** Create `src/agent/semantic_responses.py`; test `tests/test_agent1_2_transport.py`.

**Interface:** `DeepSeekResponsesJsonSchemaComparator.compare(...) -> SemanticResult`; expose one event per request with usage/status/IDs and diagnostic raw data for the private runner.

- [ ] Convert the frozen v2 system/user messages into Responses `instructions` and `input`.
- [ ] Send one non-streaming `deepseek-flash` request with `reasoning.effort=none`, temperature 0, 384 output-token cap, and `text.format.type=json_schema`.
- [ ] Retain raw response bodies and provider metadata; never include Authorization or the key.
- [ ] Re-run host parsing and consistency checks; classify empty, incomplete, malformed, and invalid output cases.
- [ ] Run focused transport tests.

### Task 3: Offline A/B metrics

**Files:** Create `evaluation/agent1_2_metrics.py`; test `tests/test_agent1_2_evaluation.py`.

**Interface:** `score_agent1_2(tasks, arm_a_run, arm_b_run) -> dict`.

- [ ] Report host schema acceptance, strict all-task verdict accuracy, accepted-only verdict/class/dimension scores, N/A and consistency diagnostics, failure taxonomy, and usage.
- [ ] Mark historical Arm A schema failures unclassified where raw responses were never saved.
- [ ] Run focused evaluator tests.

### Task 4: One-pass private runner and report

**Files:** Create `evaluation/evaluate_agent1_2.py`; create `docs/agent1-2-structured-output-validation.md`; update both READMEs and `docs/agent-handoff.md`.

**Interface:** `run_agent1_2_experiment(...) -> dict`; refuse pre-existing outputs; verify frozen GT and immutable Agent 1.1 run before the first API call.

- [ ] Make one request per included task with no retries; preserve per-request raw records under `outputs/agent1_2/` only.
- [ ] Write only sanitized API events plus semantic results to private run/metrics files; prove the historical Arm A file is unchanged.
- [ ] Execute the one authorized A/B pass, rescore offline, and write aggregate-only conclusions and readiness decision.
- [ ] Run full compile, unit suite, diff/privacy/Git checks; do not start Agent 2 or commit/push.

## Verification

From `rag-agent/`: `python -m compileall src evaluation`; `python -m unittest discover -s tests`; `git diff --check`; confirm `outputs/agent1_2/` is ignored and all private result files are untracked.
