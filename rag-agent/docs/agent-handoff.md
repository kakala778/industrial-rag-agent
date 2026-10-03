# Agent Handoff

Updated: 2026-10-04.

## Current status — M11 Evidence Research Agent

M11 adds a local-first `python -m src.agent_demo --task ... --document
ALIAS=PATH` interface for two to four explicit document scopes. The existing
bounded harness runs SEARCH / LOOKUP / CLARIFY / FINISH under host validation;
FINISH contains scope statuses and evidence IDs, not free-form findings or
engineering conclusions. A host renderer validates selected IDs against the
active session and writes a UTF-8 Markdown report to ignored
`outputs/agent11/report_<UTC timestamp>_<unique suffix>.md` without source
paths, traces, prompts, raw provider responses or credentials.

`evidence_found` records selected host-validated source IDs only;
`no_evidence_found` means the configured search returned no candidates;
`insufficient_scope` means candidates were returned and looked up but none were
selected. Those statuses do not establish relevance, correctness or the
absence of information elsewhere in a document. Non-finished runs produce a
terminal-status report and label gathered evidence preliminary. Citations
authenticate source location only; the demo produces no engineering or
compliance verdict.

Local Qwen is the default selector; deterministic reference policy is available
for offline mechanics checks. Explicit `--policy deepseek-reference` requires
`DEEPSEEK_API_KEY` and prints a data-transfer notice before the first request.
The legacy `--query/--scopes` path remains available for Agent 0's two-scope
copied-quote behavior. M11 does not change RAG/retrieval defaults or Agent 1's
semantic comparator. See the
[M11 Design Spec](superpowers/specs/2026-10-03-m11-evidence-research-agent-design.md)
and [Implementation Plan](superpowers/plans/2026-10-03-m11-evidence-research-agent.md).

M11 implementation and closeout checks are complete: `compileall`, 345 unit
tests, `git diff --check`, and a synthetic three-document CLI smoke passed.
The smoke covered FINISH with three host citations, terminal CLARIFY, budget
exhaustion, preliminary-report labeling, and absence of source absolute paths.
It did not evaluate live Qwen/DeepSeek semantic selection or industrial-PDF
evidence relevance.

## Next milestone — M12 kickoff pending scope selection

M12 has not started, and no M12 scope is recorded in the repository or current
GitHub issues. The recommended first option is a bounded demo-readiness pilot:
freeze a small set of user-approved tasks and document scopes, run M11 with the
chosen local selector, and manually audit source/page/block, excerpt fidelity,
and task completion. Record failures by parsing, retrieval, action selection,
evidence relevance, or reporting layer before considering any implementation
change. Keep RAG and the Agent 1 semantic comparator frozen unless new evidence
directly identifies one as the blocker.

This is a kickoff proposal, not an approved M12 specification. The older M10
architecture review also lists a troubleshooting evidence assistant, but it
requires a reliable operating-data source that is not currently part of the
demo. Do not let it make engineering or safety recommendations from document
retrieval alone. Confirm the M12 audience, demo surface, selected task set, and
whether the first pilot should use local Qwen before writing its formal spec.

## Previous status — Agent 1.3 ROI review

Agent 1 is frozen for a bounded evidence-first demo. Agent 1.2 used the same
frozen 17-task cohort and DeepSeek Responses `json_schema`: 15/17 passed the
full Host contract, strict verdict accuracy was 14/17, accepted-output verdict
accuracy was 14/15, and accepted dimension accuracy was object 15/15, value
13/15, unit 8/15, and condition/applicability 7/15. Two outputs were safely
rejected by the existing consistency validator. GT remains AI-reviewed, not
independently human-adjudicated.

**Decision: `AGENT1_FROZEN_FOR_DEMO`.** The recommendation at that point was to
proceed toward an end-to-end demo around
scoped evidence research, host-rendered references, and honest incomplete or
unsupported outcomes. Agent 1.2 remains evaluation-only and is not suitable
for authoritative engineering comparisons. If shown, its assessments must be
clearly experimental and paired with source evidence. This review does not
start Agent 2. RAG and Agent 0 remain frozen. No new model pass was run; see the
[Agent 1.3 ROI review](agent1-3-semantic-comparator-roi-review.md) and the
[Agent 1.2 report](agent1-2-structured-output-validation.md).

## Historical status — Agent 1.1 semantic contract validation

Agent 1.1 made 17 requests; strict host validation rejected 16 outputs as
`invalid_schema`. The single accepted result matched its verdict but marked
known value and unit dimensions `not_applicable`. Strict verdict accuracy was
1/17 (5.9%). The historical raw responses were not saved, so the exact causes
of those schema rejections remain unavailable. See the
[Agent 1.1 report](agent1-1-balanced-semantic-benchmark.md) and the
[historical Agent 1 report](agent1-semantic-evidence-comparison.md).

## Historical status — Agent 1 semantic comparison

The bounded Agent 1 experiment is complete. Four of eight frozen tasks entered
DeepSeek Flash; the other four were stopped by deterministic preflight. Verdict
accuracy was 4/4, but every expected verdict was `NOT_COMPARABLE`, so this does
not measure three-way verdict discrimination. Object/field alignment scored
4/4, value and unit alignment 1/4 each, and condition/applicability 3/4. Two
tasks remain retrieval-bound and were not counted as semantic-model failures.
The provisional semantic GT is AI-assisted, not independently human-verified.

Recommendation: if the route continues, first validate the contract against a
small, balanced, independently adjudicated semantic benchmark. Keep retrieval
and Agent 0 frozen. Defer unit conversion, operator/modality scoring,
cross-document research, and any engineering recommendation until the current
dimension-labeling weakness is resolved. No later milestone has started.
See the [Agent 1 report](agent1-semantic-evidence-comparison.md).

## Historical pre-Agent handoff — after M10.1 (2026-10-02)

Experiment checkpoint at that time: `2918f99`.

## Decision and implementation status

**ENTER AGENT.** M10.1 closes the final pre-Agent RAG experiment. No Agent
code, tool API, task-state loop or framework has been integrated yet.

Current application defaults remain PyMuPDF + Dense without BGE. MinerU
structured, Hybrid and BGE are optional PDF CLI capabilities. Visual
enhancement and parent-context reranking remain offline experiments.

## First bounded use case

Document Research Agent: the user identifies two local document scopes and
requests a field/clause comparison. The workflow selects search and original
evidence lookup, tracks covered subquestions, compares supported facts,
clarifies ambiguous scope and reports missing evidence with citations.

One-shot Top3 RAG does not manage those branches or coverage state. A fixed
RAG chain wrapped in a graph does not meet this Agent goal. Multiple documents
are multiple evidence sources, not necessarily independent fact verification.

## Reusable assets and missing contracts

| Boundary | Existing assets | Work required in the Agent stage |
|---|---|---|
| Search | Dense / lexical / Hybrid / BGE functions | Thin preloaded multi-document session and explicit backend/scope parameters |
| Evidence lookup | Structured Documents and source/page/block metadata | Validated session evidence IDs → bounded original block; no arbitrary file paths |
| Generation | Qwen context/prompt helpers | Verify action schema and parsing; existing text generation is not proven tool calling |
| State | None | Covered facts, pending questions, evidence, errors, call budget, termination |
| Evaluation | Frozen retrieval cohorts and failure audits | Freeze Agent tasks and PDF-grounded answer criteria before inference |

Preserve original queries. Do not invent equipment/document scope during
rewrite. Apply user scope consistently before Dense/BM25 candidate selection,
not as a hidden QA-derived filter. Distinguish no evidence, invalid scope/ID,
timeout and service failure. Reuse models/indexes within one session.

Keep Dense cosine, RRF and BGE score meanings distinct. Preserve source/page/
block provenance, hide ranking diagnostics from ordinary citations, and avoid
using Document-local chunk_id alone as a global evidence ID.

## Recommended first implementation boundary

Start with a small explicit Python state machine: search, lookup, clarify,
finish, with bounded calls and text length. Confirm design and tool contracts
before implementation. LangGraph becomes a separate decision only when
persistence/resume or complex branches are actual requirements.

Do not add a vector database, new embedding/reranker, Vision LLM integration,
HyDE, graph index or another pre-Agent retrieval experiment. Do not simulate
database/telemetry tools that are not available. The first task is evidence
research, not autonomous industrial operations.

## Acceptance criteria to freeze before building

- Explicit-scope comparison uses both document sources and lookup where needed.
- Missing scope triggers clarification; missing evidence produces an honest
  incomplete result/refusal rather than invented fields.
- Conflicting evidence is reported with separate provenance.
- Invalid IDs/tool actions, timeouts and call-budget exhaustion terminate or
  recover predictably without silent guessing.
- Check tool selection, arguments, task coverage, factual grounding and each
  claim's citation support. Unit tests or refusal phrases alone are insufficient.
- Real inputs, tool traces and generated content remain local and ignored.

## Limits carried forward

Q25 parent text improved BGE rank7→1, but returned child stayed incomplete.
I17 improved4→1 with already-complete target row and added table context;
Q04 regressed1→4. Original evidence did not improve in aggregate, and CPU
cost increased. Parent expansion is not enabled in Agent by this handoff.

OCR/image/layout losses, underspecified scope and matcher limitations remain.
Answer semantic correctness is not established by retrieval hit rate or source
citations. Agent must expose these limits rather than claim to repair them.

Evidence: [architecture review](m10-architecture-review-and-agent-readiness.md),
[M10.1 experiment](m10-parent-context-controlled-experiment.md),
[M9.2 validation](m9-independent-validation-and-ranking-audit.md),
[M9.3 integration](m9-optional-hybrid-integration.md).
