# Agent 0.2 — evidence selection and DeepSeek Flash comparison

2026-10-03; baseline `7f513a5`, current checkout on
`codex/agent0-minimal-harness`. No worktree, dependency installation, commit,
push or merge. Existing RAG, progress/coverage rules and exact-quote grounding
remain fixed. Provider comparison uses local `qwen3:4b` and official API
`deepseek-flash`, non-thinking, temperature 0, 512 output tokens.

## Protocol and interpretation

Reuse Agent0.1's unchanged eight-task manifest: seven evidence investigations
and one scope-preflight control. Five evidence tasks have reviewed target IDs
available; two are `RETRIEVAL_BOUND` because the requested A target is absent
from fixed Top3. Exclude the latter from model evidence-selection success rates;
report them separately. Preflight is not a model achievement.

SEARCH replays exact successful Agent0.1 observations; LOOKUP reads the same
bounded original cache registry. Parsing, embeddings, BM25/RRF, reranking and
retrieval recomputation are forbidden by the runner. No GT or candidate review
is sent to either model. Shared messages carry the unchanged semantic selector
prompt, public state and dynamic action schema. Both actions pass the same
strict parser, eligibility checks and original harness validation.

Ollama additionally enforces the schema in its decoder; DeepSeek uses JSON
Output and receives that schema as prompt data, without native tool calls.
This is a model/provider contrast, not an isolated model-weight causal estimate.
Qwen's output cap is also 512 in this experiment (historical Agent0.1 used 1536).
Both receive the same new schema packaging; no benchmark keywords or task-specific
prompt rules were introduced.

Frozen task SHA-256:
`8999684871a3b4968c99223fc0d5e5d694b0f7f3fd7415face4af5266afb4ae6`.
Frozen observations SHA-256:
`a542f1b5ff5e3c9adc494a82ee62a8cbfb66a12c1f17cf4c07564c6175f5b3c9`.
Candidate review SHA-256:
`f6b5a1a2f3f0d7b2ffd0d2e4e27a79ad7270cc974a28c9ecb571b27b08e2ff99`.

GT remains Agent0.1's assistant original-PDF review, **not independent human GT**.
The 42 candidate annotations screen bounded LOOKUP blocks against that unchanged
oracle; positive ID blocks were checked against its requested fields/endpoints/
conditions. This is not a new independent 42-candidate human PDF audit. Frozen
page/field targets constrain the benchmark; they do not establish whole-document
absence or exhaust all valid engineering formulations.

The main selection metric evaluates the model's attempted final IDs, including
FINISH actions later rejected for nonexact quotes. All supported scopes need a
relevant observed-and-looked-up ID, with no unrelated extra finding. Successful
LOOKUP alone is not final selection. Quote fidelity is separately EXACT,
NORMALIZED_EQUIVALENT (conservative formatting normalization) or NOT_FAITHFUL.
Numeric values, case-sensitive units, comparison operators, applicability and
numeric exponents are not silently corrected. Strict runtime grounding is unchanged.

## ID-only citation prototype

An offline prototype projects the same attempted FINISH selections to
`{scope, evidence_id}` and renders the complete bounded authentic LOOKUP text
and provenance from the host registry. It rejects unseen/unlooked-up IDs,
wrong scopes and duplicate IDs. It does not repair a runtime action or change
the production copied-quote contract. It is a counterfactual on the same model
selections, not a separately inferred ID-only arm. Longer bounded context may
restore an omitted condition, but cannot repair an irrelevant selected ID.

## API and cost controls

Key is read only from `DEEPSEEK_API_KEY`, never a file, state or event. HTTPS POST
uses the standard library, official endpoint, JSON Output, disabled thinking and
no native tool calls. Private evaluation saves parsed actions and public states
to ignored outputs; API response reasoning, credentials, headers and error bodies
are not retained. HTTP errors are recorded as status codes only. Transient errors
retry at most once; auth failures/cost exhaustion stop further API tasks.

Runtime-verified [official Chinese pricing](https://api-docs.deepseek.com/zh-cn/quick_start/pricing):
Flash off-peak input cache hit ¥0.02/M, miss ¥1/M, output ¥4/M; peak miss ¥2/M
and output ¥8/M. October 3 is Saturday, hence off-peak under the published rule.
Cost reservation uses peak miss/output rates with a conservative byte-based
input bound before each request, settling on measured usage when available.
Unmetered failures retain their reservation rather than being counted as free.
The stop threshold is ¥3, within the requested ¥5 cap; no bulk repetitions.

The first partial local run stopped before any API calls because Ollama was
not running. It is retained as an infrastructure diagnostic, excluded from the
formal comparison. The existing local server was started, and the runner now
checks the local model endpoint before starting paid work.

## Measured results and decision

**DeepSeek Flash materially improves candidate-available evidence selection in
this small controlled run.** Keep it as an optional experimental policy model;
retain local Qwen and the deterministic default. This does not establish general
industrial readiness or model-weight-only causality. No further models were tried.

Primary denominator: R01/R03/R05/R06/R07, five candidate-available tasks.
Metrics include attempted final findings, even when the strict host rejects FINISH.

| Candidate-available metric | A Qwen3-4B | B DeepSeek Flash |
|---|---:|---:|
| Correct final ID selection per task | 1/5 | 5/5 |
| Correct relevant IDs / attempted final IDs | 6/10 | 9/9 |
| Fully relevant evidence task success | 0/5 | 4/5 |
| Mechanical terminal completion | 2/5 | 5/5 |
| Quote RELEVANT / PARTIAL / IRRELEVANT | 4 / 2 / 4 | 8 / 1 / 0 |
| Quote fidelity EXACT / NORMALIZED_EQUIVALENT / NOT_FAITHFUL | 5 / 5 / 0 | 9 / 0 / 0 |
| Irrelevant final IDs | 4 | 0 |
| Missing required relevant scope findings | 3 | 0 |
| FINISH rejected | 3 | 0 |
| Steps / SEARCH / LOOKUP | 25 / 10 / 10 | 25 / 10 / 10 |
| Offline ID-only host-rendered selection success | 1/5 | 5/5 |

| Attempted-quote alignment, applicable findings | A | B |
|---|---:|---:|
| Field/value | 6/10 | 9/9 |
| Unit | 5/7 | 7/7 |
| Condition/applicability | 2/7 | 6/7 |
| Source scope | 10/10 | 9/9 |

Scope alignment means the actual document alias, independently of field/value
and target clause. Unit and condition denominators exclude N/A requirements;
correct units or source alone do not establish a correct finding. Missing
required relevant scopes excludes the deliberately unsupported B side of R05.
All attempted quotes were reviewed offline against the unchanged PDF-reviewed
oracle/cache, with decisions fingerprinted to the individual finding. The
lexical screen's mistaken acceptance of the opposite R04 condition was overridden.

| Task | A outcome | B outcome |
|---|---|---|
| R01 | Wrong A ID, formatting change; FINISH rejected | Correct IDs, exact relevant quotes |
| R03 | Wrong A ID; formatting change; FINISH rejected | Correct IDs, exact relevant quotes |
| R05 | Includes unrelated B evidence | Omits unsupported B, relevant A; host status incomplete |
| R06 | Correct IDs, B condition omitted | Correct IDs, preserves B condition |
| R07 | Wrong A ID, B condition omitted; FINISH rejected | Correct IDs/exact quotes, but B applicability omitted |

R05 is a successful evidence task with `incomplete` host comparison status:
the task explicitly requires omitting the unsupported side, and the unchanged
harness honestly cannot complete a bilateral comparison. It is not a fabricated
two-sided answer. R07 is Flash's remaining primary failure: the correct ID's
bounded text has the condition, but the copied quote drops it.

The two separate `RETRIEVAL_BOUND` tasks R02/R04 have 0/2 relevant task success
for both providers. A mechanical completion is 0/2, B is 2/2, but both select
unrelated A evidence: B uses a different field on R02 and the opposite condition
on R04. They remain retrieval-bounded; acceptance of authentic irrelevant quotes
is still a system limitation. Do not count missing target IDs as provider failures.
The one preflight control passes for both with zero inference/tool calls.

Across all seven evidence tasks: mechanical completion A 2/7, B 7/7; fully
relevant success A 0/7, B 4/7; each uses 35 steps, 14 SEARCH and 14 LOOKUP calls.
A has five rejected FINISH attempts and seven EXACT/seven format-equivalent
attempted quotes; B has zero rejected FINISH and thirteen EXACT attempted quotes.
Both have zero repeated execution, invalid tool execution or budget violation.
Grounding/citations of accepted final evidence pass A 2/2 and B 7/7; A's five
rejected tasks have no final findings and are N/A, not successful claims.

**Citation decision: MOVE TOWARD HOST-RENDERED EVIDENCE.** Offline ID-only rendering
eliminates copying changes and retains the complete bounded condition context.
For Flash the same selected IDs support 5/5 task evidence versus 4/5 copied-quote
success; for Qwen it remains 1/5 because rendering cannot fix its selected IDs.
This counterfactual is promising, not evidence that an actual ID-only model prompt
would choose equally well. Whole blocks also contain nearby clauses; production
needs a bounded, auditable span/condition contract, not an unreviewed whole-block
semantic verdict. Current runtime copied-quote checks remain unchanged.

### API requests, usage and expense

| Measure | Observed |
|---|---:|
| HTTP attempts | 36 |
| Metered successful requests | 35 |
| Input tokens | 123,571 |
| Output tokens | 2,905 |
| Metered total tokens | 126,476 |
| Timeout / retry | 1 / 1 |
| Other API errors | 0 |
| Unmetered attempts | 1 |
| Metered off-peak estimated cost | ¥0.08300796 |
| Conservative total upper estimate | ¥0.296528 |

The timeout has no returned usage, so metered tokens exclude it and its cost is
not treated as zero. The upper estimate retains its full reservation and charges
all metered tokens at peak cache-miss/output rates. This is an estimate from
returned usage and official pricing, not a reconciled account bill. No additional
paid repeats were necessary; both estimates are below ¥3 and the ¥5 cap.

### Answers to the 16 requested decisions

1. Candidate-available: five evidence tasks.
2. Retrieval-bound: two evidence tasks, plus one separate preflight control.
3. Qwen selection: 1/5 tasks; 6/10 relevant final IDs; copied-quote success 0/5.
4. Flash selection: 5/5 tasks; 9/9 relevant final IDs; copied-quote success 4/5.
5. Quote field/unit/condition alignment: A 6/10, 5/7, 2/7; B 9/9, 7/7, 6/7.
6. Primary irrelevant final IDs: A four, B zero. Retrieval-bound adds two each.
7. Fidelity: primary A five EXACT/five format-equivalent, B nine EXACT; zero
   NOT_FAITHFUL after conservative offline formatting classification. This is
   not permission to accept normalized quotes in the strict runtime.
8. Mechanical completion: primary 2/5→5/5; all evidence tasks 2/7→7/7.
9. API requests: 36 attempts, 35 successful metered calls.
10. Metered tokens: 123,571 input, 2,905 output, 126,476 total.
11. RMB estimate: metered ¥0.083008; conservative total bound ¥0.296528.
12. One timeout, one successful retry; no auth/rate-limit/5xx/malformed errors.
13. Flash is worth retaining as optional experimental selector on this evidence;
    neither default replacement nor general reliability is established.
14. Move toward host-rendered evidence; current prototype is offline only.
15. Remaining bottlenecks: condition preservation and citation contract with
    available IDs; retrieval availability on two tasks. Framework/planning need
    is not demonstrated.
16. Next milestone should validate an actual bounded ID/span-only host-rendered
    citation contract and unsupported-side handling, with human GT confirmation.
    No semantic engineering comparison, further model search or next milestone
    implementation started.

Only one formal A/B pass was run on this small existing-corpus set. There is no
statistical significance claim or independent human review. The missing-candidate
subset remains separately visible; model choice alone does not solve it.

## Post-run AI-assisted GT audit and offline rescore

On 2026-10-03, an AI-assisted review against the original PDFs found expected-ID
contamination and a negation error in the provisional oracle. The source manifest,
candidate reviews, saved Agent 0.2 actions, frozen SEARCH observations and raw
results were preserved. A derived repair manifest and repaired copies were used
to rescore the saved actions offline; no model, retrieval, embedding or parser
was run. The original PDF review status remains `pending_human_pdf_review` in
the source artifacts; the derived oracle is labeled `ai_pdf_reviewed`, not
independently human-verified.

The repaired metric values match the pre-audit headline values above:

| DeepSeek metric | Before repair | After repair |
|---|---:|---:|
| Candidate-available / retrieval-bound tasks | 5 / 2 | 5 / 2 |
| Candidate-available ID selection | 5/5 tasks; 9/9 IDs | 5/5 tasks; 9/9 IDs |
| Fully relevant copied-quote success | 4/5 | 4/5 |
| Field / unit / condition alignment | 9/9 · 7/7 · 6/7 | 9/9 · 7/7 · 6/7 |
| Unsupported-side correctness | 1/1 | 1/1 |

R02 and R04 remain `RETRIEVAL_BOUND`: their newly recorded target IDs exist in
the reviewed source, but were absent from frozen Top3 observations. Qwen's
candidate-available values also remain unchanged: ID selection 1/5, fully
relevant success 0/5, field 6/10, unit 5/7, condition 2/7 and unsupported-side
correctness 0/1.

The audit removed four invalid expected-ID occurrences and one partial or
redundant occurrence, added three expected-ID occurrences (including the
second relevant R07/B figure), corrected one negated condition, and made two
narrow condition-precision updates. Six candidate labels were reclassified.
R05/B remains a medium-confidence, document-specific provisional unsupported
side. The current GT schema does not encode comparison operators or normative
modality such as `宜`; these are recorded as schema limits, not guessed into
condition groups. The full audit and readiness decision are in
[`agent-benchmark-review-status.md`](agent-benchmark-review-status.md).

The derived rescore was run twice and produced identical output bytes. The
runner confirms protected input hashes before and after scoring and records
`inference_performed=false`, `retrieval_performed=false`; all derived local
artifacts remain ignored by Git.

## Reproduction

From `rag-agent/`, with the matching private files and environment variable:

```powershell
python evaluation/evaluate_agent02.py --manifest outputs/agent0_1/frozen_set.json --observations outputs/agent0_1/run_5abe221949ec4668bcb96d13c14d1d5f/results.json --review outputs/agent0_2/candidate_reviews.json --pricing outputs/agent0_2/pricing.json
python -m compileall src evaluation
python -m unittest discover -s tests
git diff --check
git status --short
```

Inputs, review, observations and pricing are hash-checked before, after every
task, and on exit. Results checkpoint per task under ignored `outputs/agent0_2`.
Private documents, QA, candidate labels and model actions are not published.

Final verification (2026-10-03): compileall passed; all 226 unit tests passed;
`git diff --check` passed. Independent review reproduced and prompted regression
fixes for numeric and unit-exponent normalization boundaries. Offline scores were
recomputed after those fixes without repeating model inference; reported results
did not change. The final local audit verified all 145 protected files unchanged,
the original run unchanged, inference-time runtime/harness unchanged, zero
forbidden retrieval operations, ignored private artifacts, and absence of the
actual environment credential in tracked/nonignored files and stage text outputs.
Only offline metric code changed after the formal inference pass. Changes remain
local and uncommitted; no push, merge, dependency installation or next milestone
implementation was performed.
