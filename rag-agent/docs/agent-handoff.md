# Agent Handoff — after M10.1

Updated: 2026-10-02. Experiment checkpoint: `2918f99`.

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
