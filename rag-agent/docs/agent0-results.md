# Agent 0 — implementation, verification and limits

2026-10-03; baseline `e4e3eb0`, local branch `codex/agent0-minimal-harness`.
No commits, merge, tag, PR or push. Implementation is in the attached worktree,
not copied over the original checkout. No dependencies installed; frozen RAG
source, parameters, PDF CLI defaults and historical experiments are unchanged.

## Measured outcome

Reliable bounded harness implemented. Qwen action selection remains experimental.
180 unit tests pass (143 existing + 37 Agent tests), compileall and diff checks
pass. A read-only independent review found two comparison defects; both were
reproduced with failing tests and fixed. Final review found no remaining Critical
or Important issue.

Frozen synthetic fixture SHA-256:
`41046c818a6c8cefbaeaaf3e154e57b9f9745a51a19ebe69471569723584b97e`.
Constant synthetic embeddings/reranker isolate harness mechanics. These fixtures
have at most one relevant block per scope, not industrial retrieval difficulty.

| Measurement | Deterministic | qwen3:4b, state-constrained schema |
|---|---:|---:|
| Expected task terminal/coverage/text-comparison | 10/10 | 10/10 |
| Expected tool sequence/count | 10/10 | 6/10 |
| Executed tool arguments | 7/7 applicable | 7/7 applicable |
| Search scope | 7/7 applicable | 7/7 applicable |
| Grounded findings / correct citation IDs | 4/4 evidence-bearing cases | 4/4 evidence-bearing cases |
| Executed invalid tool calls | 0 | 0 |
| Actual over-budget execution | 0 | 0 |
| Normal/conflicting/simple-finish steps | 5 | 7 |

Model selection actually ran on 7 cases, all seven met their synthetic expected
terminal/coverage criteria. Missing/invalid scope preflight and injected invalid
ID are three harness controls, not model achievements. The invalid-ID control
records one rejected action; no invalid lookup executes. The budget case
terminates at its one-step cap: expected exhaustion is not a budget violation.
Grounding/citation metrics are N/A on cases without claims, not automatically
successful claims. Tool-error injection is evaluation-only.

First static-schema Qwen run: 5/10 overall, 2/7 model-selected cases. In normal
cases it tried FINISH immediately after SEARCH, without LOOKUP; validation
blocked it. A general state-dependent schema now excludes ineligible FINISH and
constrains query/scopes/observed IDs; host validation remains mandatory. The
same frozen fixture was rerun, with no GT, retrieval or fixture edits. Final
metric refinements were applied to the saved states without further inference.
Raw model reasoning was never retained.

Even the second run repeats SEARCH in normal, missing-side, conflict and
simple-finish tasks. This explains 10/10 task termination versus 6/10 tool
sequence correctness. It does not justify claiming reliable autonomous planning.

## Real industrial cached-PDF smoke

Two distinct real document scopes selected from the existing local Q01/Q06
sources; 270 chunks. The cross-document task/query/aliases and one-sided Q01
diagnostic were frozen in a private manifest before inference. Ground truth
was never supplied to the selector or used to filter candidates/pages.

| Check | Deterministic | Qwen |
|---|---|---|
| Final status | finished | clarify; investigation incomplete |
| Steps / SEARCH / LOOKUP | 5 / 2 / 2 | 11 / 4 / 6 |
| Distinct successful lookup IDs | 2, covering A and B | 1, A only |
| Quote/citation provenance | 2 grounded excerpts | N/A: no final claims |
| Existing one-sided Q01 page+keyword diagnostic | hit | hit |
| Time, single local run | 37.463 s, includes resource initialization | 84.763 s, same session reuses resources |
| Cross-document semantic task success | unassessed | unassessed; no completed comparison |

Qwen trace: SEARCH×4 → LOOKUP×6 → CLARIFY. It searched A/B/A/A, then repeatedly
looked up the same A ID and never looked up B. This is a confirmed Agent
coverage/progress failure. The existing Q01 retrieval diagnostic hit in both
paths; it does not establish the correctness or completeness of B-side evidence.

27 protected PDF/QA/cache files have identical pre/post hashes. MinerU calls=0.
Private manifest SHA-256:
`fab3786233ced0b0d18741a90d5777f1e6941528002126be2e89feb19a046779`.
Only cached structured output and existing local MiniLM/BGE/Qwen were used.
No industrial text, QA, document names, answers or traces enter public files.
The smoke's nonzero exit reports Qwen's unfinished investigation, not a test or
hash-check failure. All observed outcomes are retained under ignored outputs.

## Contracts and the requested twenty-point handoff

1. **Agent versus RAG:** RAG retrieves once then generates. Agent tracks scope,
   searches each source, chooses lookup/clarification, checks citations and stops
   under budgets. This version compares excerpts, not engineering meaning.
2. **Modules:** `src/agent/tools.py`, `state.py`, `actions.py`, `harness.py`,
   `policy.py`, `selector.py`, `io.py`; `src/agent_demo.py`; two evaluation runners
   and a frozen synthetic JSON fixture. No runtime Agent framework.
3. **KnowledgeBaseSession:** caller maps safe aliases to Documents. Copies replace
   real source identifiers, validate parent identities, chunk once; lazy model,
   corpus embeddings and BGE reuse; BM25 caches each requested scope subset.
   Filtering occurs before both retrieval branches. No DB/server/persistence.
4. **search_knowledge:** `(query, scopes=None, retriever="hybrid", rerank=True,
   top_k=3) -> ToolResult(status, results, message)`. Results expose
   evidence_id/text/source/page/block_type/block_index/chunk_id. Top K is 1–3.
   Empty results are `no_evidence`; exceptions are `timeout/error`. Nonempty
   Dense/Hybrid candidates do not prove the query has an answer.
5. **lookup_evidence:** known session ID → up to 2,000 original characters,
   provenance/context_kind/offset/truncated. Small blocks are whole; large ones
   use bounded neighborhoods containing the child. Ambiguous repeated child
   text falls back to child-only; no guessed occurrence or arbitrary path access.
6. **Evidence identity:** full SHA-256 of canonical JSON structural tuple
   `[alias,page,block_type,block_index,chunk_id]`, prefix `ev_`. Reverse registry,
   collision check, deterministic for unchanged corpus and aliases, no local
   paths. Not durable identity across content/version changes.
7. **State:** query, requested/resolved scopes, search/lookup histories, observed
   IDs, looked-up evidence, clarification, steps/call counts/remaining budgets,
   status, findings, comparison, answer, errors and trace. Terminal statuses:
   finished/incomplete/clarify/invalid_scope/invalid_action/tool_error/timeout/
   budget_exceeded. State is task-local; no long-term memory.
8. **Actions:** SEARCH(query, one requested scope), LOOKUP(observed evidence_id),
   CLARIFY(question), FINISH(findings with scope/evidence_id/exact quote). Extra
   fields, duplicate JSON keys, unsupported actions and coerced types are rejected.
   SEARCH preserves the original query; no rewrite/HyDE tool is introduced.
9. **Loop/budget:** selector gets a copy of state; choose → validate → budget
   check → tool → observation → update → termination. Defaults 12 steps / 4
   searches / 6 lookups. Denied calls do not execute; errors/timeouts are explicit.
   Ollama has a 120 s HTTP timeout, not a whole-task wall-clock guarantee.
10. **Deterministic verification:** tests cover normal flow, missing/invalid
    scopes, no evidence, invalid ID, tool failures/timeouts, every budget,
    forged quotes, early FINISH, state mutation, repeated text, late target
    preservation and multi-finding comparison. Full suite: 180 passing.
11. **Qwen:** `/api/chat`, local qwen3:4b, `think=false`, schema-constrained JSON,
    temperature 0, bounded output. Static run failed; constrained run met 7/7
    model-selected synthetic task outcomes, but had unnecessary calls. Industrial
    selection failed to cover B. Keep optional/experimental, deterministic default.
12. **Invalid action:** terminate `invalid_action`, record a fixed validation
    reason and safe metadata; never repair JSON or execute its tool. No hidden
    reasoning is requested/stored. Unknown tool ID is an explicit invalid-ID
    observation; model cannot lookup another task's unobserved ID.
13. **Benchmark:** ten frozen cases: normal, missing side, conflicting text,
    missing scope, invalid scope, invalid ID, empty search, injected tool error,
    budget exhaustion and simple finish. Task outcome, tool count/sequence,
    arguments/scope, evidence/citations, rejected actions and budgets are separate.
14. **Industrial smoke:** details above; cache-only, unchanged inputs, zero
    MinerU. A-side historical diagnostic is not cross-document ground truth.
15. **Trace:** action/tool input/status/observation summary/state transition.
    Example below; evidence-bearing state/output remains local ignored.
16. **RAG errors/limits:** existing OCR/image/layout loss, candidate irrelevance,
    ranking/context loss and lack of a semantic no-answer detector. This stage
    does not tune or repair those components. No new RAG failure proven by smoke.
17. **Agent errors:** premature FINISH fixed by eligibility/schema validation;
    parent-prefix target loss and global multi-quote comparison fixed with
    regression tests. Repeated search/lookup and missing B coverage remain observed
    model decision failures; quoted-text authenticity does not prove relevance.
18. **LangGraph:** currently unnecessary. Consider only when durable execution,
    pause/resume, persistence or genuinely complex branches become requirements.
19. **Next suggested milestone:** Agent 0.1 progress/coverage evaluation—detect
    repeated actions and require evidence relevance plus field/condition alignment
    on a small independently reviewed cross-document set. Do not change RAG or
    add a framework in response to these observed Agent failures. Not started.
20. **Files/Git:** 19 new files plus two README edits; 37 new tests. Changes remain
    uncommitted in `codex/agent0-minimal-harness`; original checkout untouched.

Synthetic trace entry (illustrative structure, not industrial evidence):

```json
{
  "step": 1,
  "action": "SEARCH",
  "tool_input": {"query": "Compare rated pressure in A and B.", "scopes": ["A"]},
  "tool_status": "ok",
  "observation_summary": {"result_count": 1, "evidence_ids": ["ev_<sha256>"]},
  "state_transition": "running->running"
}
```

## Run from rag-agent

```powershell
python -m src.agent_demo --query "比较 A 与 B 的额定压力" `
  --document "A=<local document A.pdf>" --document "B=<local document B.pdf>" `
  --scopes A B --save-local

# Explicit optional model selector, existing MinerU cache only:
python -m src.agent_demo --query "比较 A 与 B 的额定压力" `
  --document "A=<local document A.pdf>" --document "B=<local document B.pdf>" `
  --scopes A B --parser mineru --cache-root "<local MinerU cache>" `
  --policy qwen --save-local

python evaluation/evaluate_agent0.py --save-local
python evaluation/evaluate_agent0.py --policy qwen --save-local
python evaluation/run_agent0_industrial_smoke.py `
  --data-root "<private industrial-rag-data>" --cache-root "<matching MinerU cache>"

python -m compileall src evaluation
python -m unittest discover -s tests
git diff --check
git status
```

No `--scopes` → CLARIFY before loading files/models. Unknown aliases →
invalid_scope. MinerU cache miss → initialization error, no parser rerun.
CLI returns 0 only for finished, 2 for other valid terminal outcomes; smoke
returns nonzero when investigation remains unfinished. All saved states/traces,
private manifests and model-derived excerpts go only to ignored `outputs/agent0`.

Current HTTP/schema options follow the official
[Ollama chat API](https://docs.ollama.com/api/chat) and
[structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
documentation, verified during this task. Schema formatting does not replace
the host's action/state validation.

Remaining boundaries: semantic correctness/relevance requires review; no hard
cancellation of synchronous embedding/BGE; repeated model actions are budgeted
but not yet treated as no-progress errors. Smoke hashes protect existing frozen
files; new directory entries and initialization-failure tail auditing are not
covered. Synthetic benchmark does not assess prompt-injection resistance or
real-world parameter/condition equivalence.
