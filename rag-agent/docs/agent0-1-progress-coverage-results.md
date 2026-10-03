# Agent 0.1 — progress, coverage and evidence relevance

2026-10-03. Baseline `e474f1e` on `codex/agent0-minimal-harness`; implemented
in the current `D:` checkout, without a worktree, dependency installation,
commit, push, PR or merge. Frozen RAG and qwen3:4b are unchanged.

**Decision: implementation/evaluation complete; full milestone acceptance NOT
met.** Real loops and omitted lookup scopes improve, but the guarded model
achieves 0/7 fully relevant evidence tasks. This is not a reliable industrial
research Agent and does not justify semantic engineering decisions.

## Observable Agent0 failure audit

The saved industrial smoke contains two `REPEATED_SEARCH` actions with identical
results, five `REPEATED_LOOKUP` actions on one A ID, and `UNCOVERED_SCOPE` B
despite three observed B candidates. The final `UNNECESSARY_CLARIFY` repeats the
original question instead of identifying a missing constraint. There is no
FINISH in this trace, so no observed industrial `PREMATURE_FINISH`. Historical
static-schema synthetic attempts were rejected before executing such a finish.

The deterministic smoke contains a genuine, grounded B quote about a different
kind of length: `IRRELEVANT_EVIDENCE`. Qwen has no final findings, hence final
grounding/relevance is N/A and the task has `INSUFFICIENT_EVIDENCE`. Initial
scope searches and first lookups are `VALID_PROGRESS`. These judgments concern
observable actions/evidence, not hidden reasoning or whole-document absence.

## Progress and coverage contract

`src/agent/progress.py` derives searched scopes, observed/lookup IDs and scope
states from task histories. Progress means a new successful searched scope,
new evidence ID, first successful lookup, explicit empty tool result, or valid
terminal action. Repeating unchanged successful input produces `no_progress`.
Trace diagnostics contain safe IDs, counts, coverage and remaining scopes; no
hidden reasoning or raw model messages are retained.

Successful SEARCH uses the exact original query and one fixed alias. The same
query/alias cannot produce a new input in this milestone and is not re-executed.
An already successful LOOKUP is similarly suppressed. Denials consume a step,
but not SEARCH/LOOKUP budget, preventing an infinite denial loop. Schema removes
duplicates; independent host checks still enforce identity, scope and grounding.
Exhausted eligible tools stop before another selector request.

Coverage transitions are UNSEARCHED → CANDIDATES_OBSERVED → LOOKED_UP_EVIDENCE,
or NO_EVIDENCE for an empty successful search, or INVALID_SCOPE at preflight.
Candidates without a lookup leave the scope outstanding. While outstanding
candidates exist, only unseen IDs in those scopes are lookup-eligible. FINISH
requires complete mechanical coverage. Missing findings still yield incomplete,
never an agreement. NO_EVIDENCE is an empty tool result, not semantic absence.

Missing/duplicate scopes clarify before tools. Unknown aliases stay invalid.
Callers can declare a missing task constraint through
`run(..., clarification_required="...")`, also handled before tools. Ordinary
unfinished work cannot CLARIFY. There is no automatic semantic ambiguity detector;
the task/ingestion boundary must supply known missing constraints.

The model still chooses search order, candidate among eligible IDs, extra unseen
lookups after coverage, exact quotes, omission of irrelevant findings and finish
timing. This does not add a deterministic answer policy, query rewriting,
engineering equivalence, unit conversion, framework or richer planner.

## Frozen evaluation protocol

Synthetic arms use the unchanged Agent0 fixture, hash
`41046c818a6c8cefbaeaaf3e154e57b9f9745a51a19ebe69471569723584b97e`.
The new industrial set was authored independently of that fixture and frozen
before industrial Qwen inference. It contains eight paired tasks over three
existing local PDFs: seven evidence investigations and one scope-preflight
control. Categories include two supported sides, different fields/devices,
applicability, different units, one side without relevant evidence and ambiguity.
It is independent task construction on the existing corpus, not external-corpus
validation. A nonempty but irrelevant candidate is not an empty corpus control.

Task/GT manifest SHA-256:
`8999684871a3b4968c99223fc0d5e5d694b0f7f3fd7415face4af5266afb4ae6`.
Ground truth authority is **assistant visual review of original PDF pages, not
independent human-reviewed GT**. Positive fields/conditions were checked on full
page renderings; the negative-side document had all 27 pages reviewed as an
overview with detailed section review. The negative label applies only to this
local document, not other referenced standards. Human confirmation is outstanding.

A uses the deterministic control on the new runtime. B loads byte-verified
Agent0 source from Git `e474f1e`; a result-type adapter passes unchanged results
from the same session. C uses qwen3:4b with progress/coverage constraints.
Both Qwen arms keep temperature 0, think=false, num_predict 1536, num_ctx 16384,
the same original query, Hybrid+BGE Top3 and lookup limits. No GT or reviewed-page
filter is passed to retrieval or selector. Raw evidence and states stay ignored.
Latency is not compared: A includes initialization, while later arms reuse models.

All seven deterministic evidence tasks first ran with the actual frozen RAG
tools. Repeated live CPU retrieval was then replaced by a separate **fixed
SEARCH-observation A/B/C experiment**: all arms replay those exact successful
query/scope results, while LOOKUP still reads original blocks from the same PDF
cache. No reranking, candidate filtering, GT input or retrieval tuning is added.
The original partially completed live run is retained, not overwritten. Its
first baseline task encountered an Ollama CUDA crash/HTTP 500; that is an
infrastructure failure, not a model-quality result. Later live baseline tasks
reproduced the repeat/clarify failure. The complete comparison below concerns
the separate replay experiment, not fresh end-to-end inference on every arm.

The frozen observation source SHA-256 is
`a542f1b5ff5e3c9adc494a82ee62a8cbfb66a12c1f17cf4c07564c6175f5b3c9`.
All 42 candidate rows were checked against the original cache registry after
inference. The source checkpoint's final audit flag was unset when interrupted;
a separate post-run audit checks all 145 protected files, frozen GT, unchanged
RAG modules, byte-identical Agent0 baseline and runtime hashes used for inference.
An incomplete source run is not represented as a completed live A/B/C run.

The offline evaluator separates mechanical completion, scope coverage, distinct
lookup scopes, repeated execution, guard denials, grounding, citation and relevance.
Relevance labels are RELEVANT/PARTIAL/IRRELEVANT/UNCERTAIN, with separate field,
unit, condition and scope alignment. A lexical screen is provisional: numeric
and unit token boundaries prevent substring errors, but cross-clause association
still needs review. `finding_reviews` supports a PDF review tied to each finding
by SHA-256, rejecting stale or incomplete reviews. No second LLM judge is used.
No findings means N/A grounding/citation, never an automatically successful claim.

## Synthetic A/B/C results

| Metric | A deterministic | B Agent0 Qwen | C guarded Qwen |
|---|---:|---:|---:|
| Expected task outcome | 10/10 | 10/10 | 10/10 |
| Model-selected cases | 0 | 7 | 7 |
| Repeated SEARCH executed | 0 | 8 | 0 |
| Repeated LOOKUP executed | 0 | 0 | 0 |
| Total steps | 25 | 33 | 25 |
| Grounding / citations, evidence-bearing cases | 4/4 | 4/4 | 4/4 |
| Invalid tool execution / actual budget violation | 0 / 0 | 0 / 0 | 0 / 0 |

The remaining three cases are harness preflight/injection controls. Sequence
diagnostics are 10/10, 6/10 and 10/10 respectively, but sequence identity is not
a success criterion. Synthetic relevance is trivial fixture evidence, not an
industrial accuracy estimate. It cannot establish this milestone's success.

## Independent industrial tasks — fixed SEARCH observations

The denominator is the **seven evidence tasks**. The eighth is a scope-preflight
control, passed by all arms without model/tool calls; exclude it from evidence
accuracy and lookup coverage. Every final quote was reviewed against original
PDF renderings after inference. Strict task success requires all supported sides,
matching field/unit/applicability, grounded quotes, and no irrelevant or partial
finding. Omitting a genuinely unsupported side is permitted.

| Metric | A deterministic | B Agent0 Qwen | C guarded Qwen |
|---|---:|---:|---:|
| Mechanical terminal completion | 7/7 | 1/7 | 3/7 |
| Fully relevant evidence task success | 1/7 | 0/7 | 0/7 |
| Successful LOOKUP scope coverage | 14/14 (100%) | 8/14 (57.1%) | 14/14 (100%) |
| Repeated SEARCH executed | 0 | 14 | 0 |
| Repeated LOOKUP executed | 0 | 30 | 0 |
| No-progress actions | 0 | 44 | 0 |
| Guard denials | 0 | 0 | 0 |
| Total steps | 35 | 73 | 35 |
| Grounding / citations, final evidence-bearing tasks | 7/7 | 1/1 | 3/3 |
| RELEVANT / PARTIAL / IRRELEVANT final findings | 7 / 2 / 5 | 1 / 0 / 1 | 3 / 1 / 2 |
| Unsupported FINISH rejected | 0 | 0 | 4 |
| Unnecessary CLARIFY | 0 | 6 | 0 |
| Invalid tool execution / actual budget violation | 0 / 0 | 0 / 0 | 0 / 0 |

Guard denials are zero because the schema already excludes duplicates; regression
tests exercise the independent denial path. Rejected FINISH is an invalid action,
not an executed invalid tool. Four C rejected tasks and six B clarification tasks
have no final findings, so grounding/citation are N/A. Accepted findings retain
the exact-substring/source/ID validator; passing percentages do not establish
that the model reliably produces accepted quotes.

| Anonymous task | B outcome | C outcome | Remaining C failure |
|---|---|---|---|
| R01 equipment grounding | Repeat, CLARIFY | Invalid FINISH | Unsupported finding; relevant candidate existed |
| R02 same unit, different device | Repeat, CLARIFY | Invalid FINISH | Unsupported finding; A target page absent from Top3 |
| R03 same value, different endpoints | Repeat, CLARIFY | Invalid FINISH | Unsupported finding despite relevant pages on both sides |
| R04 distances/conditions | Repeat, CLARIFY | FINISH, irrelevant | A uses opposite applicability; target page absent from Top3 |
| R05 one unsupported side | FINISH, irrelevant | FINISH, irrelevant | B unrelated finding should be omitted |
| R06 different distance objects | Repeat, CLARIFY | FINISH, partial | B quote omits camera-on-tower condition |
| R07 different endpoints/scenes | Repeat, CLARIFY | Invalid FINISH | Unsupported finding despite relevant pages on both sides |

R06 is A's only relevant task. A finishes mechanically by taking first candidates;
five unrelated and two partial findings show why it is not an evidence-quality
reference. The two partial A findings concern ground-bar setup/nearby text rather
than the complete equipment connection rule. C's R06 quote has the correct object,
value and unit, but omits required applicability and remains PARTIAL.

An isolated diagnostic reconstructed the pre-FINISH R03 state and called the same
Qwen selector once, outside benchmark counts. The rejection reproduced: both IDs
were looked up and scope-correct; one quote was not an exact substring although
normalized text matched. This supports a quote-fidelity/representation failure
for that diagnostic. The precise cause of every rejected benchmark action is
unknown because raw invalid payloads were not retained in safe traces. The host
check was kept; no permissive quote normalization was added.

## Acceptance and next decision

Mechanical completion rises 1/7→3/7, but relevant success stays 0/7: the full
acceptance criterion fails. Limits include a small existing corpus, seven evidence
investigations, one scope ambiguity control, one run per arm, replay rather than
complete fresh end-to-end A/B/C, and assistant-reviewed GT awaiting human review.
Semantic ambiguity of units/conditions is not automatically detected.

Two tasks lack the relevant A page in fixed Top3, a concrete retrieval constraint.
Other failures already have relevant candidates: coverage guards cannot repair
quote fidelity, selection, unsupported-side omission or condition preservation.
Evidence points to candidate availability and model extraction limits, rather
than remaining progress-state loops. RAG stays frozen; this evidence does not
authorize tuning during this milestone.

Next: **bounded model comparison with the same harness, RAG, observations and
exact-quote/condition acceptance**, reporting the candidate-available subset
separately. Human GT verification is needed before stronger accuracy claims.
Reliable evidence must precede semantic comparison. Richer planning and LangGraph
have no demonstrated need here. No next milestone was started.

### Answers to the 17 requested questions

1. Agent0 failure: repeated SEARCH/LOOKUP, uncovered B, unnecessary CLARIFY;
   grounded deterministic evidence can still be irrelevant. No real smoke FINISH.
2. Progress: new successful scope search/empty result, ID, first lookup or valid
   terminal action, derived from observable histories.
3. Guard: suppress unchanged successful input; consume a bounded step, no tool
   budget. Schema and independent host checks both enforce it.
4. Coverage: terminal states LOOKED_UP_EVIDENCE/NO_EVIDENCE/INVALID_SCOPE;
   candidates alone remain outstanding and have lookup priority.
5. Autonomy: order, eligible candidate, extra lookup, exact quote, omission and
   finish timing remain model choices. Missing constraints require explicit host
   input; automatic model discovery of ambiguity is not implemented.
6. Synthetic: 10/10 expected outcomes; B→C repeated SEARCH 8→0, steps 33→25.
7. Industrial: eight frozen tasks, seven evidence tasks; B→C mechanical 1/7→3/7,
   fully relevant 0/7→0/7 under replay controls.
8. Repeated SEARCH: 14→0.
9. Repeated LOOKUP: 30→0.
10. Successful lookup-scope coverage: 8/14→14/14, independent of relevance.
11. C relevance: 3 RELEVANT, 1 PARTIAL, 2 IRRELEVANT; no fully relevant task.
12. Grounding/citation: all accepted findings pass; absent findings N/A; four C
    FINISH attempts rejected. No validator relaxation or false acceptance.
13. Remaining: quote rejection, condition omission, irrelevant-side inclusion,
    wrong condition, two Top3 misses, limited GT and evaluation protocol.
14. Main remaining issue: candidate extraction/selection fidelity plus two
    retrieval constraints; progress/coverage loops are no longer observed.
15. Model: unchanged this round. Controlled comparison is justified next;
    no evidence that a larger model necessarily fixes these failures.
16. LangGraph: unnecessary for these demonstrated failures.
17. Next: fixed-input model comparison and human GT verification before semantic
    comparison; no richer planner/framework implementation now.

## Reproduction and verification

Run from `rag-agent/`, after providing the matching private inputs and frozen GT:

```powershell
python evaluation/evaluate_agent01.py --prepare-baseline
python evaluation/evaluate_agent01.py --synthetic
python evaluation/evaluate_agent01.py --manifest outputs/agent0_1/frozen_set.json
# Optional controlled comparison after a complete deterministic arm:
python evaluation/evaluate_agent01.py --manifest outputs/agent0_1/frozen_set.json --search-observations outputs/agent0_1/run_SOURCE/results.json
python -m compileall src evaluation
python -m unittest discover -s tests
git diff --check
git status --short
```

Baseline preparation verifies an existing snapshot instead of overwriting it.
The runner checkpoints every task and checks full frozen inventory on success
and initialization failure, including added/deleted files. It forbids MinerU.
Private PDFs, reviewed QA, renderings, evidence, states and model-derived outputs
remain under existing ignored data/cache/output directories.

Replay accepts only matching manifest/task IDs and successful, exact-query,
single-scope observations covering the requested scopes. It verifies candidate
contents against the original registry, deepcopies results, and checks the
source artifact hash through completion. Runner guards were strengthened during
the experiment and applied to the actual source in the independent post-run
audit; the Agent runtime used for inference did not change.

Current checkout verification: compileall passed; **203 unittest tests passed**;
`git diff --check` passed. All 145 protected files/GT were unchanged; the complete
replay run made zero MinerU calls. Frozen RAG and preserved Agent0 bytes match
HEAD/baseline; Agent runtime hashes match those used for inference. PDF/QA/raw
traces/model outputs/evidence/reviews remain ignored and absent from tracked and
nonignored changed files. Branch/HEAD and live remote baseline remain
`codex/agent0-minimal-harness` at `e474f1e`; changes are uncommitted and unpushed.
