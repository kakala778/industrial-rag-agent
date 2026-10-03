# M11 Design Spec — Evidence Research Agent

**Status:** Draft for review · **Date:** 2026-10-03

## 1. Background and goals

M10 provides a bounded Agent harness for scoped SEARCH / LOOKUP / CLARIFY / FINISH actions. It already tracks task-local state, evidence IDs, provenance, progress, budgets and traces. The current reference-output path lets the host validate selected evidence IDs and render excerpts from the active parsed-document session. Its main product gaps are the fixed two-scope contract, JSON-only CLI output and the absence of a user-facing research report.

M11 turns that path into a small, local-first industrial-document research demo. A user supplies one task and two to four explicit document aliases. The Agent searches only those aliases, looks up observed evidence IDs, tracks mechanical scope coverage, and terminates with either a report or a clarification request. The report presents source excerpts and provenance without claiming that a citation proves relevance, correctness, applicability or engineering compliance.

Success means a user can run one bounded task end to end and inspect a UTF-8 Markdown artifact, including honest partial results. It does not mean the system is suitable for operational or compliance decisions.

## 2. Current architecture

- `src/agent_demo.py` is the existing Agent 0 CLI. It accepts `--query`, repeated `--document ALIAS=PATH`, and exactly two `--scopes`; it prints JSON and optionally saves a private state/trace artifact.
- `src/agent/harness.py` owns the bounded action loop, preflight validation, action revalidation, tool execution, progress checks, budgets and terminal status. It currently rejects any scope count other than two.
- `src/agent/actions.py` defines strict SEARCH / LOOKUP / CLARIFY / FINISH schemas. The reference FINISH schema requires exactly two scope outcomes and uses `supported`, `insufficient_evidence` and `no_candidates` statuses.
- `src/agent/selector.py` builds state-specific action schemas and prompts. Its reference prompt and eligibility logic assume two documents. CLARIFY is currently available only when a host-supplied missing-constraint string is present.
- `src/agent/state.py` and `src/agent/progress.py` already represent scopes as lists and calculate per-scope search/lookup coverage. Coverage is mechanical; it does not establish relevance.
- `src/agent/tools.py` provides `KnowledgeBaseSession`, scoped SEARCH / LOOKUP, session evidence IDs and a host reference renderer returning a bounded excerpt, alias, page/block provenance, span and hash.
- The Agent 0 CLI uses existing local parsing/retrieval options. Retrieval algorithms, defaults and ranking parameters are outside M11.

## 3. M10-to-M11 change

M11 adopts the existing `evidence_reference` contract for the demo path and adds:

1. Reference-contract scope validation for 2–4 unique aliases. The legacy copied-quote Agent 0 contract remains two-scope and otherwise unchanged.
2. One FINISH outcome for every resolved alias, with dynamically sized schema constraints.
3. A terminal, one-shot CLARIFY action that returns a bounded question and requires the user to rerun with a clarified task. No saved session is resumed.
4. A host-side Markdown report renderer/writer and a report-oriented CLI invocation.

The M11 contract uses these scope statuses:

- `evidence_found`: the Agent selected one to three looked-up evidence IDs from this alias. This records selection only; it is not a host finding that the evidence is relevant or that a claim is true.
- `no_evidence_found`: successful search for this alias returned no candidates. The report phrases this as “the configured search returned no candidates”; it does not assert that the document contains no relevant information.
- `insufficient_scope`: search returned candidates and at least one was looked up, but the Agent selected no evidence for the requested task. This is an Agent assessment, not a semantic validation by the host.

Host checks enforce status preconditions, scope membership, observed and looked-up IDs, active-session ownership and host reference rendering. They cannot validate semantic relevance. M11 FINISH carries no free-form model claim or summary; report findings are rendered from validated status and host-rendered references. This keeps the default output from making unsupported engineering or equivalence claims.

## 4. Data flow

1. The CLI validates a nonempty task and 2–4 unique `ALIAS=PATH` document arguments before loading them. Every supplied alias is an explicit scope; there is no automatic discovery or scope expansion.
2. Existing `load_corpus` and `KnowledgeBaseSession` load the supplied local documents and construct a task-local searchable session. Existing parser/retrieval choices and defaults are passed through unchanged.
3. `AgentHarness` initializes `AgentState` with the original task and aliases. It enforces scope, action, evidence-ID, coverage, call-budget and terminal-state rules.
4. The selector chooses one of SEARCH, LOOKUP, CLARIFY or FINISH. SEARCH is limited to one resolved alias per action and preserves the original task text. LOOKUP accepts only an ID already observed in SEARCH results.
5. For `evidence_found`, the host resolves each selected ID against the active session and renders the excerpt/provenance. The model never supplies citation text, page numbers, block labels, spans or excerpts in FINISH.
6. The report renderer converts terminal state to Markdown. It uses aliases rather than input paths, escapes untrusted task/clarification text, and inserts citations and excerpts only from host-rendered references.
7. The writer creates a unique file under `outputs/agent11/` using UTF-8 and exclusive creation. The CLI prints a relative output path. The report is a local private artifact and is covered by the existing `outputs/` ignore rule.

When a remote selector is explicitly chosen, its existing provider behavior still applies. In particular, DeepSeek receives task/evidence text. The local-first demo defaults to the existing local selector; selecting `--policy deepseek-reference` prints a clear disclosure before the first provider request.

## 5. Module responsibilities

- `src/agent/actions.py`: keep strict action shapes; make only the reference outcome array dynamic from two to four scopes; define exact reference status fields and CLARIFY bounds.
- `src/agent/harness.py`: enable 2–4 aliases for the reference contract, preserve legacy two-scope copied-quote behavior, revalidate per-scope terminal outcomes, and terminate on a valid reference-contract CLARIFY.
- `src/agent/selector.py`: update the reference prompt/schema eligibility for 2–4 scopes, status semantics and one-shot clarification; keep model text and evidence as untrusted data.
- `src/agent/markdown_report.py` (new): pure rendering of `AgentState` plus host references; status-aware report sections, safe Markdown formatting and no absolute input paths.
- `src/agent/io.py`: add an exclusive-create UTF-8 report writer under `outputs/agent11/`, returning a relative path and saving no trace/raw response.
- `src/agent/policy.py`: add an explicit deterministic reference-contract policy for offline mechanics/smoke checks; it is not the default research policy and does not imply semantic relevance.
- `src/agent_demo.py`: expose the M11 task/report CLI contract using the existing entry point; `--task` plus repeated `--document ALIAS=PATH` defines the 2–4 scopes. Default to the existing local Qwen selector with the reference contract; keep DeepSeek reference selection as an explicit remote option and deterministic reference policy for offline checks. Write the report automatically. Remove the M11 need for separate `--scopes` and `--save-local` flags.
- Tests: expand reference action/harness/selector coverage and add focused report and M11 CLI tests. Existing Agent 0, Agent 1, retrieval and frozen benchmark inputs remain unchanged.

## 6. Action contract

All actions are strict JSON objects with no extra properties:

- `SEARCH`: `{ "action": "SEARCH", "query": <original task>, "scopes": [<one eligible alias>] }`.
- `LOOKUP`: `{ "action": "LOOKUP", "evidence_id": <ID previously returned by SEARCH> }`.
- `CLARIFY`: `{ "action": "CLARIFY", "question": <1–500 character question> }`. It is terminal. The question must identify a missing user constraint, not merely restate the task or disguise empty retrieval. The user starts a new run after answering.
- `FINISH`: `{ "action": "FINISH", "outcomes": [...] }`, with exactly one outcome per resolved alias and no duplicates. `evidence_found` carries one to three evidence IDs; `no_evidence_found` and `insufficient_scope` carry no IDs or generated claim text.

FINISH is eligible only after each scope has either returned no candidates or had at least one observed candidate looked up. This is per-scope mechanical coverage, not exhaustive examination of every candidate. Tool/service failures and budget exhaustion remain distinct terminal statuses and are never translated into `no_evidence_found`.

## 7. Report contract

The report contains:

```markdown
# Industrial Research Report
## Task
## Sources
## Evidence
## Findings
## Clarification
## Limitations
```

`Clarification` appears only for a `clarify` terminal state. `Evidence` lists host-rendered excerpts with alias, page when available, block type/index and host-created evidence citation ID. `Findings` lists one host-rendered status per scope and the citations selected by FINISH; for incomplete/error states it identifies the terminal reason and labels any gathered evidence as preliminary. No model-authored citation syntax, page, excerpt or free-form finding is accepted.

The report records source aliases only and never copies the filesystem paths supplied to `--document`. It excludes original PDFs, full traces, raw provider responses, prompts, API keys and benchmark/QA data. Necessary source excerpts in the requested Markdown artifact are private-derived data; the artifact remains local and ignored. The report explicitly states that provenance authenticates source location only and does not prove parser fidelity, relevance, claim support, equivalence or compliance. It makes no engineering verdict.

Output names follow `outputs/agent11/report_<UTC timestamp>_<unique suffix>.md`; creation must fail closed on collisions rather than overwrite an older report. Text is UTF-8. The writer returns a path relative to the application root, and the CLI prints only that relative path plus terminal status.

## 8. CLI design

Run from `rag-agent/`:

```powershell
python -m src.agent_demo --task "查询两个文档中的设备参数" `
  --document A=design.pdf `
  --document B=manual.pdf
```

`--document` may be supplied two to four times. Every alias must be unique and satisfy the existing safe-alias contract. M11 treats those aliases as the complete set of scopes. The command writes `outputs/agent11/report_....md` and prints its relative path and terminal status. Invalid counts/aliases fail before document/model initialization. Existing parser selection remains available with its present defaults. The local Qwen selector is the default and uses the reference contract. `--policy deepseek-reference` is explicit and prints a disclosure before sending task/evidence text to the provider. `--policy deterministic` is available for offline mechanics checks, not as a relevance-capable research mode.

Exit status is `0` for `finished`; `2` for clarification or other valid but incomplete Agent termination after writing a report; `1` for CLI/session initialization failure before a usable `AgentState` exists. No raw state/trace file is written by this entry point.

## 9. Test strategy

- Action schema: accept 2, 3 and 4 unique scope outcomes; reject 1/5, duplicate/missing/foreign scopes, invalid status payloads, unsupported fields and malformed clarification.
- Harness: exercise 2-, 3- and 4-scope reference runs; enforce one SEARCH per eligible alias, observed-ID LOOKUP, scope-local references, host rendering, status preconditions and terminal CLARIFY. Verify copied-quote Agent 0 behavior remains two-scope.
- Selector: verify dynamic per-scope FINISH schema, local/reference prompt semantics and that CLARIFY is available under the reference contract without fabricating a host constraint.
- Report: cover finished, clarify and incomplete/error states; verify real host citations/provenance, excerpt formatting, no absolute paths, escaping of untrusted content, UTF-8 and collision-safe writing.
- CLI: reject invalid document counts/aliases before loading; pass every provided alias as a scope; write and print one relative report path; never emit raw trace/provider output.
- Run the repository's established compile, unittest and diff checks from `rag-agent/`. Add one synthetic three-document end-to-end smoke test; do not use or change private PDFs, frozen GT, QA or ignored outputs as fixtures.

## 10. Out of scope

- Any change to RAG defaults, parsing, retrieval, ranking parameters, Hybrid or BGE behavior.
- Agent 1 semantic comparator changes, semantic-equivalence claims or engineering/compliance verdicts.
- More than four scopes, automatic document discovery, implicit scope expansion or search outside supplied aliases.
- LangGraph, another Agent framework, a database, multi-agent orchestration, long-term memory, pause/resume or session persistence.
- New APIs/providers or model training. Existing selectors may be reused only through the M11 evidence-reference contract.
- Benchmark/ground-truth expansion, private PDF/QA edits, raw trace/provider-output persistence, commit, push, merge or tag as part of implementation.
