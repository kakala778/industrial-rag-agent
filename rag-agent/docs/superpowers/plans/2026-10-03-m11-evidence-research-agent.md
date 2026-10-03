# M11 Evidence Research Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing reference-action Agent path into a 2–4-scope local research demo that writes a host-rendered Markdown report.

**Architecture:** Reuse `AgentHarness`, `AgentState`, `KnowledgeBaseSession` and the active-session citation renderer. Generalize only the reference action contract, add a pure Markdown renderer and private report writer, and adapt the existing CLI while preserving legacy copied-quote behavior.

**Tech Stack:** Python standard library, existing selectors and retrieval stack, `unittest`.

**Spec:** [M11 Design Spec](../specs/2026-10-03-m11-evidence-research-agent-design.md)

## Global Constraints

- Accept exactly 2–4 unique explicit aliases for the M11 reference contract.
- Keep the copied-quote Agent 0 path limited to exactly two scopes.
- Preserve existing RAG/parser/retrieval defaults and parameters.
- Use only host-validated looked-up evidence IDs and host-rendered references.
- Write only the requested UTF-8 report under ignored `outputs/agent11/`; never overwrite.
- Persist no absolute source paths, full traces, raw provider responses, prompts, credentials, PDFs, QA or GT.
- Do not implement semantic comparison, an engineering verdict, persistence, a database, multi-agent or a new framework.
- Do not commit or push during implementation unless separately authorized.

## Review Focus

- Four scopes with the default search budget: every alias must still receive its own search before FINISH.
- Candidate evidence followed by an empty/partial lookup: do not mislabel it `no_evidence_found` or finish without per-scope lookup coverage.
- Remote selector selection: make clear that task/evidence text leaves the machine.
- Untrusted task, clarification and source excerpt text containing Markdown/HTML: prevent it from creating fake citations or report sections.
- Output filename collision or report-write failure: never overwrite and never claim a report exists when writing failed.

---

### Task 1: Generalize the M11 reference action schema

**Files:**
- Modify: `src/agent/actions.py`
- Test: `tests/test_agent_reference_actions.py`

**Interfaces:**
- Consumes: existing `validate_action(raw, contract="evidence_reference")` and `REFERENCE_ACTION_JSON_SCHEMA`.
- Produces: reference FINISH outcomes for 2–4 scopes with statuses `evidence_found`, `no_evidence_found`, `insufficient_scope`; CLARIFY questions remain bounded to 1–500 characters.

- [x] Add tests accepting 2, 3 and 4 unique outcomes; assert `evidence_found` requires 1–3 unique IDs and the other two statuses accept no extra fields.
- [x] Add rejection tests for 1/5 outcomes, duplicate scopes, unknown statuses, malformed IDs, extra claim fields and invalid CLARIFY text.
- [x] Run `python -m unittest discover -s tests -p test_agent_reference_actions.py` from `rag-agent/`; confirm new cases fail against the current fixed-two schema.
- [x] Update only the reference schema/validator to encode the spec; keep copied-quote schema and validation unchanged.
- [x] Rerun the focused test module and require all cases to pass.

### Task 2: Generalize reference harness coverage and terminal clarification

**Files:**
- Modify: `src/agent/harness.py`
- Test: `tests/test_agent_reference_harness.py`
- Regression: `tests/test_agent_harness.py`, `tests/test_agent_progress.py`

**Interfaces:**
- Consumes: Task 1 reference outcomes and existing `KnowledgeBaseSession` renderer.
- Produces: `AgentHarness.run(task, scopes)` accepts 2–4 scopes only for `evidence_reference`; a valid reference-contract CLARIFY sets terminal state `clarify`; valid FINISH stores host-rendered evidence by alias.

- [x] Add synthetic 2-, 3- and 4-scope runs and assert every alias was searched exactly once before FINISH.
- [x] Test `evidence_found` rejects unobserved, unlooked-up, wrong-scope and foreign-session IDs; test status preconditions for `no_evidence_found` and `insufficient_scope`.
- [x] Test reference-contract CLARIFY terminates without another tool call and preserves a bounded question; retain the existing copied-quote clarification gate.
- [x] Run the three focused test modules and confirm the new reference cases fail before the implementation change.
- [x] Change reference-contract preflight/FINISH validation and rendering for dynamic outcomes; preserve two-scope copied-quote comparison behavior.
- [x] Rerun focused modules and verify legacy Agent 0 benchmark assertions remain unchanged.

### Task 3: Update reference selector guidance and eligibility

**Files:**
- Modify: `src/agent/selector.py`
- Modify: `src/agent/policy.py`
- Test: `tests/test_deepseek_selector.py`
- Regression: `tests/test_agent_selector_evaluation.py`

**Interfaces:**
- Consumes: Task 1 schema and Task 2 state/coverage rules.
- Produces: state-specific reference schemas with one outcome for every resolved alias; `OllamaActionSelector(action_contract="evidence_reference")`; reference CLARIFY is selectable without a host-injected `clarification_required` value; deterministic reference policy for offline checks.

- [x] Assert the reference FINISH schema enumerates 2–4 actual aliases and offers only statuses permitted by that scope's observed candidates/lookups.
- [x] Assert the reference prompt defines statuses without treating citation as relevance or correctness, directs missing user constraints to terminal CLARIFY, and forbids semantic engineering verdicts; assert Ollama can use this contract without changing its copied-quote default.
- [x] Run focused selector tests to observe failure against the existing two-document prompt/schema.
- [x] Update reference prompt and eligibility logic, make the Ollama contract selectable, and add a deterministic reference policy; keep copied-quote selector defaults intact.
- [x] Rerun focused selector/provider tests.

### Task 4: Add host-rendered Markdown report and private writer

**Files:**
- Create: `src/agent/markdown_report.py`
- Modify: `src/agent/io.py`
- Create: `tests/test_agent_markdown_report.py`

**Interfaces:**
- Produces: `render_research_report(state, session) -> str` and `write_research_report(state, session, *, output_root=APP_ROOT, report_id=None) -> Path`, where the returned path is relative to the application root.
- The renderer consumes only `AgentState` plus active-session host references; it does not serialize trace, raw state, model prompt/response or input paths.

- [x] Test finished reports with two and four aliases; assert host-generated evidence IDs, alias/page/block, excerpt and neutral per-scope status are present.
- [x] Test clarify and budget/tool-error reports clearly identify terminal state and mark any previously gathered evidence as preliminary.
- [x] Test Markdown/HTML escaping, absent absolute paths, UTF-8 Chinese text and exclusive-create behavior when a name already exists.
- [x] Run `python -m unittest discover -s tests -p test_agent_markdown_report.py` from `rag-agent/` and confirm failure before adding the module/writer.
- [x] Implement pure rendering and unique `outputs/agent11/report_<UTC timestamp>_<unique suffix>.md` creation with UTF-8 and no overwrite.
- [x] Rerun report tests and inspect one synthetic rendered artifact in memory (do not persist test content outside a temporary directory).

### Task 5: Expose the M11 research-report CLI

**Files:**
- Modify: `src/agent_demo.py`
- Test: `tests/test_agent11_cli.py` (new)
- Regression: `tests/test_agent_selector_evaluation.py`

**Interfaces:**
- Consumes: Tasks 1–4; existing `load_corpus`, parser options and selector choices.
- Produces: `main(argv=None)` accepts `--task` and 2–4 repeated `--document ALIAS=PATH`, treats all aliases as scopes, writes one report, prints relative path/status and returns the spec-defined exit code.

- [x] Test 1- and 5-document invocations fail before `load_corpus`; test duplicate/invalid aliases likewise fail before session/model setup.
- [x] Test that three document aliases reach `AgentHarness.run` unchanged, a terminal run writes one report and stdout contains no absolute path or serialized state.
- [x] Test policy wiring selects the reference contract for local Qwen and deterministic modes; test explicit DeepSeek mode prints the data-transfer disclosure without making a network request in the test.
- [x] Test clarify/incomplete return code `2` after a report is written; initialization failure returns `1` without claiming a report exists.
- [x] Run CLI tests to observe failure before changing the entry point.
- [x] Replace the old M11-facing `--query/--scopes/--save-local` flow with `--task`/explicit scoped documents and automatic reference-report writing; default to local Qwen, retain deterministic offline mode, and disclose explicitly selected DeepSeek data transfer before its first request.
- [x] Rerun CLI and relevant legacy tests.

### Task 6: Document M11 and verify the complete implementation

**Files:**
- Modify: `README.md`
- Modify: `rag-agent/README.md`
- Modify: `rag-agent/docs/agent-handoff.md` or add a dated M11 result/handoff document, preserving historical findings

**Interfaces:**
- Documentation describes the new command, output/privacy boundary, citation limits, status meanings and local/remote selector behavior from the spec.

- [x] Update current status and quick-start documentation; do not rewrite historical Agent 0/Agent 1 experiment conclusions.
- [x] Run `python -m compileall src evaluation` from `rag-agent/`.
- [x] Run `python -m unittest discover -s tests` from `rag-agent/`.
- [x] Run `git diff --check` from the repository root.
- [x] Run one synthetic three-document end-to-end smoke through the CLI using temporary inputs; verify report path, citations, clarification/incomplete behavior and no source absolute paths.
- [x] Inspect final diff and tracked/untracked status; verify outputs remain ignored and no frozen GT, QA, private PDF or raw response was touched.
