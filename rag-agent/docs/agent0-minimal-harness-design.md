# Agent 0 — Minimal Industrial Agent Harness

2026-10-03, baseline `e4e3eb0`. RAG is frozen. No new dependency or runtime framework.

## Goal and boundary

One local task compares evidence in two explicit document aliases. Success means
validated actions, scoped tools, original-evidence lookup, bounded termination
and inspectable citations. It does not mean autonomous industrial diagnosis.
The user authorized implementation after a matching design; no additional gate,
commit, merge, tag, PR or push is required or performed.

## Session and tools

`KnowledgeBaseSession(corpus, model=None, reranker_model=None)` receives
`{safe_alias: [Document, ...]}`. File ingestion occurs outside tool execution.
Copies replace source names with aliases and strip other metadata. Missing block
identity is assigned a deterministic document ordinal before existing chunking.
Existing structural identities must be unique; ambiguous parents are rejected.
Models/embeddings load once; scope-specific BM25 indexes are cached. Filtering
happens before Dense/BM25/Top20/BGE, never after retrieval. Empty aliases are
valid scopes and need no models. Default Agent search is Hybrid+BGE; PDF CLI
defaults are untouched.

`search_knowledge(query, scopes=None, retriever="hybrid", rerank=True, top_k=3)`
returns `ToolResult(status, results, message)`; each result contains
`evidence_id,text,source,page,block_type,block_index,chunk_id`. `scopes=None`
means all session aliases at tool level; the comparison harness only permits
one requested alias per search. Top K is bounded 1–3. Scores are not confidence.
`no_evidence` means an empty corpus/result, not a semantic absence detector.

`lookup_evidence(evidence_id)` returns at most 2,000 original characters, plus
safe provenance, context kind, offset and truncation flag. Only registered IDs
are accepted. Whole short blocks are returned; long blocks use a neighborhood
containing the complete child. No arbitrary path access; no M10 rerank expansion.

IDs are SHA-256 of canonical JSON `[alias,page,block_type,block_index,chunk_id]`
with `ev_` prefix. Registry detects collisions and reverse maps to the exact
child/parent. IDs are session-local, deterministic for fixed corpus/aliases;
they are not permanent IDs across changed document versions.

Statuses: `ok,no_evidence,invalid_scope,invalid_evidence_id,timeout,error`.
Argument errors use `error`; exception details and paths do not enter results.
Tool-raised timeouts are distinguished; synchronous embedding/BGE execution has
no hard cancellation deadline. Call/step budgets do not imply wall-clock bounds.

## Actions, state and loop

Strict action-specific JSON, no extra keys, no coerced types or repaired JSON:

- SEARCH: `action,query,scopes` (one alias; original query preserved).
- LOOKUP: `action,evidence_id` (must have been observed in this task).
- CLARIFY: `action,question`.
- FINISH: `action,findings`; findings contain `scope,evidence_id,quote`.

FINISH quotes must be exact substrings of successfully looked-up evidence and
belong to the declared requested alias. Every scope must have been searched;
omitted/unavailable support yields an incomplete result, never an agreement.
Nonempty searched scopes require at least one successful lookup before FINISH.
The model receives remaining budgets and a state-constrained action schema;
runtime validation independently enforces the same boundaries.
The renderer compares each scope's **set of quoted texts** as same/different, explicitly reserving
semantic parameter equivalence for review. This conservative extractive v0
does not pretend string differences prove engineering contradictions or that
an authentic but irrelevant quote answers the user's question.

State: original query, requested/resolved scopes, search history, observed IDs,
lookup history/results, pending clarification, steps/calls, status, findings,
errors and trace. Terminal statuses: `finished,incomplete,clarify,invalid_scope,
invalid_action,tool_error,timeout,budget_exceeded`. No long-term memory.

Preflight checks missing/duplicate/unknown scopes before model selection.
Loop: choose → schema/state validation → budget check → execute → observation
→ update → termination. Defaults: 12 steps, 4 searches, 6 lookups; denied calls
do not execute. Invalid model output terminates `invalid_action` with zero tools
for that action. Task-local failures are visible; deterministic policy visits
each scope, looks up the first candidate, then finishes or reports tool failure.
LLM selector may choose other observed evidence or retry within these budgets.

Trace records validated action, safe tool input, status, counts/IDs summary and
state transition. It excludes hidden reasoning and raw model responses. Trace
and evidence-bearing answers are local private artifacts, only under ignored
`outputs/agent0/`; no output file is written by the tools/harness themselves.

## Implementation and evaluation

Implement compact `src/agent/{actions,state,tools,harness,policy}.py`, an explicit
CLI and one frozen synthetic benchmark runner. Deterministic tests precede
qwen3:4b strict JSON selection via Ollama `/api/chat`, `think=false`, with finite
HTTP timeout/output token limit. Never use reliable function calling as an assumption.

Freeze 10 synthetic tasks before inference: normal, missing side, different
evidence, missing scope, invalid scope, invalid ID, empty search, tool failure,
budget exhaustion and simple finish. Score terminal task success, tool sequence,
arguments/scope, grounding, citations, invalid calls, steps and actual over-budget
execution separately. Error injection is evaluation-only, not production tools.
Synthetic success is harness evidence, not industrial accuracy. Run real local
cached industrial PDF smoke separately; freeze manifest and hashes before
inference, forbid MinerU rerun, preserve private inputs. Missing data or service
must be reported as unverified, never replaced by synthetic results.

No LangGraph needed until persistence, resume or complex branching is required.
