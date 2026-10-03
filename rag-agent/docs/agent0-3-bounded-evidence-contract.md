# Agent 0.3 — Bounded Evidence Reference Contract

2026-10-03; implemented on `codex/agent0-minimal-harness`, based on `1ae36f3`.
The existing RAG and progress/coverage rules remain frozen. This report records
one optional evidence-reference contract and one DeepSeek Flash pass over the
unchanged Agent 0.1 tasks, SEARCH observations, cached LOOKUP evidence and
candidate reviews. No retrieval was rerun, no new provider was compared, and no
semantic comparison was started.

## Decision

The host-rendered reference contract remains suitable for **opt-in experimental
use**. After the AI-assisted original-PDF GT audit and offline rescore, DeepSeek
still has 5/5 primary candidate-available ID selection and 5/5 fully relevant
task success, compared with 4/5 for copied quotes. The repair did not change
these headline results. The benchmark is
`READY_FOR_SEMANTIC_AGENT_EXPERIMENT_WITH_PROVISIONAL_GT`; the oracle remains
AI-assisted and **not independently human-verified**. This does not justify
changing the default selector or using it for industrial decisions.

## Contract and runtime behavior

Agent 0.2's main copied-quote limitation was completeness: on R07 DeepSeek chose
the correct evidence IDs, but its copied B quote omitted the applicability
condition present in the looked-up evidence. A quote can be an exact substring
and still omit a decisive adjacent condition.

The opt-in `evidence_reference` action returns one outcome per requested scope:

```json
{"scope":"A","status":"supported","claim":"model interpretation","evidence_ids":["ev_..."]}
{"scope":"B","status":"insufficient_evidence"}
```

`no_candidates` is a separate status, allowed only after successful empty
searches. `insufficient_evidence` means candidates were observed but the model
selected none as relevant. Unsupported outcomes cannot carry evidence IDs. The
model supplies a claim as its interpretation, but has no quote field.

At FINISH, the host requires each resolved scope exactly once. For every
supported ID it verifies that the ID was observed in the current task, was
successfully looked up, belongs to the active session registry, matches the
scope/source alias, and is not duplicated. This rejects forged, unobserved,
cross-task, cross-session, wrong-scope and duplicate references. Unsupported
statuses are checked against the successful SEARCH history. FINISH can succeed
with one supported scope and one unsupported scope.

The host resolves evidence from the active parsed-corpus registry. Text spans
start from the looked-up child and include only bounded adjacent context, using
child/parent offsets and nearby sentence boundaries; the fixed excerpt ceiling
is 1,200 characters. Every excerpt is an exact text slice of the current
parsed parent and carries a span hash plus source, page, block and
chunk provenance. Span generation does not read expected answers, target
fields, candidate labels or review annotations.

For tables, a child is mapped to complete row lines only when its offsets make
that mapping reliable. A header is included only when the parsed block explicitly
declares structural header rows. The frozen MinerU cache flattens `<th>` and
`<td>` without such reliable header markers, so the two table citations in this
run contain row text without an inferred header. Unmapped or oversized rows
fall back to the exact bounded child. Images and charts use only existing
structured/OCR text; there is no VLM path.

The existing copied-quote schema remains the default. `--policy
deepseek-reference` opts into the same DeepSeek selector with the new action
contract. The experiment sends task and evidence text to the official DeepSeek
API; its parsed outputs, human review sheet and traces remain under ignored
`outputs/`.

## Frozen experiment and results

The eight frozen tasks contain five candidate-available evidence tasks, two
`RETRIEVAL_BOUND` tasks and one preflight control. The two retrieval-bound tasks
remain outside model-selection accuracy. The control clarified before any model
or tool call. DeepSeek ran once on the seven evidence tasks; successful SEARCH
observations were replayed and LOOKUP was rebuilt locally from the existing
parsed cache, with its hashes checked against the frozen candidate reviews.
Frozen inputs and all 145 protected files were unchanged, and forbidden
retrieval or embedding operations were zero.

| Candidate-available metric | Agent 0.2 DeepSeek copied quote | Agent 0.3 host reference |
|---|---:|---:|
| Correct relevant ID selection | 5/5 tasks; 9/9 IDs | 5/5 tasks; 9/9 IDs |
| Fully relevant task success | 4/5 | 5/5 |
| Field alignment | 9/9 | 9/9 |
| Unit alignment | 7/7 | 7/7 |
| Condition/applicability alignment | 6/7 | 7/7 |
| Scope alignment | 9/9 | 9/9 |

`evidence_id_selection_success` measures the model's attempted final IDs, so it
remains comparable to Agent 0.2. The evaluator separately records whether the
host accepted FINISH. A fully relevant task result and accepted unsupported-side
result require `state.status == "finished"`; a rejected FINISH cannot count as a
successful answer even when its attempted ID selection is scored separately.

For R05, A is supported and B is `insufficient_evidence`: B had candidates,
but none was selected as relevant. The task finishes without a forced bilateral
finding; semantic comparison remains `not_evaluated`. R02 and R04 remain
retrieval-bound. Their single selected supported-side references are not counted
as selector successes or failures.

The reference arm selected 11 IDs across evidence tasks: 9 on the primary
candidate-available cohort and 2 on retrieval-bound tasks. All 11 have a
RELEVANT label in the existing assistant-reviewed provisional candidate oracle;
this is not an independent human assessment. The host reproduced the current
registry's exact citation text, provenance and span bounds for 11/11 references.
All 11 citations therefore pass **host authenticity**, not a claim that parsing
matches every original PDF glyph or that each span answers the question.

The primary cohort has 7 text-context spans and 2 table-row spans. Its maximum
excerpt is 510 characters; the 9 spans add 93 context characters in total, with
a maximum addition of 73 characters to one span. Six excerpts equal their small
parent block; the other three are bounded slices. No irrelevant candidate ID
was selected, but adjacent context was not independently labeled word by word,
so semantic context inflation has not been ruled out. Table header inclusion
was verified with synthetic structural metadata, not observed in these frozen
documents.

The model returned no quote fields. Quote-copy fidelity is therefore not
applicable to this arm: the citation text is host-generated. On the applicable
primary evidence, condition alignment is 7/7 versus Agent 0.2's 6/7. The 100%
result is across a small, fixed cohort and provisional labels, not a general
reliability estimate.

The one paid pass used 35 metered requests, 125,745 input tokens and 2,887 output
tokens. The estimated metered cost was ¥0.08661524, with a conservative upper
estimate of ¥0.274586. There were no retries or unmetered requests. No API call
was repeated to tune behavior.

After inference, a narrow harness guard was added so host renderer failures
produce a sanitized invalid result without partial findings. The saved actions
were rescored offline using the current renderer; every previously reported
metric matched, all seven evidence tasks had accepted FINISH actions, and the
new accepted-result metrics passed. Raw actions, API events and usage were
unchanged. Three files differ from the hashes recorded before inference:
`src/agent/harness.py` adds the failure guard, while `evaluation/evaluate_agent03.py`
and `evaluation/agent03_metrics.py` add offline baseline-binding and
host-acceptance checks. The saved Agent 0.2 baseline passed the strengthened
hash and exact-task-set checks. No paid rerun was made.

## Fourteen requested conclusions

1. **Copied-quote defect:** an exact copied substring can omit an adjacent
   applicability condition; that happened for the correct R07 B evidence ID.
2. **Finding schema:** per-scope `supported` with claim and evidence IDs,
   `insufficient_evidence` when candidates exist but none is selected, or
   `no_candidates` after successful empty searches.
3. **ID verification:** current-task observation, successful lookup, active
   session registry, matching scope/source, uniqueness and valid scope status
   are required at FINISH.
4. **Span generation:** deterministic child/parent offsets, sentence boundaries,
   structural row mapping, fixed size cap and provenance from the active parsed
   registry; no benchmark labels or semantic judge are used.
5. **Text and tables:** text gets bounded adjacent context; tables get complete
   mapped rows and only explicitly declared headers, otherwise the exact child.
6. **Unsupported scope:** `insufficient_evidence` is distinct from `no_candidates`
   and can finish alongside supported evidence without bilateral citation.
7. **ID selection:** maintained at 5/5 primary tasks and 9/9 relevant IDs;
   fully relevant task success is 5/5 versus 4/5 for copied quotes.
8. **Condition preservation:** 7/7 applicable primary conditions versus 6/7 in
   Agent 0.2; the same fixed cohort and provisional review authority apply.
9. **Authenticity:** 11/11 selected references match host-rendered text,
   provenance and bounded spans from the current registry.
10. **Quote fidelity:** no model-authored quote exists in the reference schema;
    quote fidelity is removed from the model's responsibilities, not measured
    as a 100% model score.
11. **Irrelevant span inflation:** no selected ID was labeled irrelevant and
    spans stayed bounded; semantic relevance of every adjacent context word was
    not independently reviewed, so inflation is not fully ruled out.
12. **Main remaining issues:** provisional GT, current-parser rather than
    original-PDF authenticity, limited fixed tasks, two Top3 retrieval misses,
    and unavailable structural table headers.
13. **Adoption:** adopt the contract for opt-in experiments and retain the
    deterministic default and existing copied-quote compatibility path. This is
    not approval for production industrial decisions.
14. **Semantic comparison:** no semantic comparison was started in this
    milestone. The subsequent AI-assisted GT audit repaired the confirmed
    expected-ID and condition issues; a semantic Agent experiment may now be
    planned using the explicitly provisional oracle and its recorded limits.

## Review and reproduction

The original local sheet is
[`human-review-sheet.csv`](../outputs/agent0_3/run_23bb9cf5f607494e94ef22b25d4fdc17/human-review-sheet.csv).
Its source review status remains pending independent human PDF verification.
The later AI-assisted audit is recorded separately in
[`agent-benchmark-review-status.md`](agent-benchmark-review-status.md); it does
not overwrite that source status. The original sheet and all raw states, claims,
evidence excerpts, model actions and API events are ignored local artifacts and
must not be added to Git.

The frozen Agent 0.1/0.2 inputs, run records and pricing file are required to
reproduce the original experiment. The one-pass runner is
`evaluation/evaluate_agent03.py`; it verifies input hashes and blocks retrieval
recomputation. At the Agent 0.3 implementation checkpoint (before the GT audit),
`python -m compileall src evaluation`, `python -m unittest discover -s tests`,
`git diff --check`, frozen-input verification, credential/privacy scan and Git
status review passed; the suite had 252 tests. The post-audit validation and
rescore evidence are recorded in
[`agent-benchmark-review-status.md`](agent-benchmark-review-status.md).

## Post-run GT repair and offline rescore

The 2026-10-03 audit identified four invalid expected-ID occurrences, one
partial/redundant expected ID, a negation error in R03/A, two narrow condition
precision opportunities, and a relevant R07/B Figure 2b candidate previously
marked irrelevant. The derived repair manifest removes the invalid and partial
IDs, corrects the condition, adds three supported expected-ID occurrences, and
reclassifies six candidate labels. The Figure 2b review is grounded in the
saved original-page image and frozen candidate metadata.

| Agent 0.3 DeepSeek host-reference metric | Before repair | After repair |
|---|---:|---:|
| Candidate-available / retrieval-bound tasks | 5 / 2 | 5 / 2 |
| Candidate-available ID selection | 5/5 tasks; 9/9 IDs | 5/5 tasks; 9/9 IDs |
| Fully relevant host-reference success | 5/5 | 5/5 |
| Field / unit / condition alignment | 9/9 · 7/7 · 7/7 | 9/9 · 7/7 · 7/7 |
| Unsupported-side correctness | 1/1 | 1/1 |

The corresponding Agent 0.2 copied-quote results also remain unchanged: 5/5 ID
selection, 4/5 fully relevant success, and 9/9, 7/7, 6/7 field, unit and
condition alignment. R02 and R04 remain retrieval-bound in both saved runs
because the corrected targets are absent from frozen Top3; their outcomes are
excluded from candidate-available selector accuracy. The offline rescore used
only saved actions and frozen observations, was byte-identical across two runs,
and did not invoke inference or retrieval. Benchmark readiness is
`READY_FOR_SEMANTIC_AGENT_EXPERIMENT_WITH_PROVISIONAL_GT`, with the medium-
confidence R05/B document-specific unsupported-side label and the lack of
independent human PDF verification kept explicit. See
[`agent-benchmark-review-status.md`](agent-benchmark-review-status.md) for the
audit counts and limitations.
