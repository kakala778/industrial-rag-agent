# Agent 1.1 — Balanced Semantic Benchmark & Contract Validation

- **Decision:** `SEMANTIC_COMPARATOR_NOT_READY`
- **Run date:** 2026-10-03
- **Ground-truth authority:** `ai_assisted_original_pdf_review`
- **Review status:** `ai_pdf_reviewed`; this is not human verification.

Agent 1.1 introduced an independent, balanced evaluation contract and benchmark
for the existing bounded semantic comparator. The run made one DeepSeek Flash
pass over 17 original-PDF evidence pairs. The strict host parser rejected 16
responses as `invalid_schema`; the one accepted response matched the verdict,
but marked known value and unit dimensions `not_applicable`. The evidence does
not validate reliable three-way comparison. Do not start Agent 2.

Agent 1's historical 8-task result remains in
[the Agent 1 report](agent1-semantic-evidence-comparison.md) and was not
rewritten. Agent 1.1 used its own local freeze and result files under the
ignored `outputs/agent1_1/` directory.

## Contract and method

The v2 evaluation records object/field, value, unit, and condition/applicability
independently, then derives a verdict using a separate `comparability_basis`.
`NOT_APPLICABLE` means the dimension itself is absent on one or both sides; an
object mismatch alone does not make known values or units inapplicable. A
comparable field with a value or condition difference can be `DIFFERENT`, while
a different field or incompatible scenario can be `NOT_COMPARABLE`. The
original v1 comparator remains the default; the v2 prompt and parser are
injected only by the Agent 1.1 evaluation runner.

The benchmark contains 17 tasks: 5 `EQUIVALENT`, 6 `DIFFERENT`, and 6
`NOT_COMPARABLE`. Three of the eight registered original PDFs contributed
evidence; 12 source pages were visually reviewed. The local freeze re-extracted
each selected excerpt from its cited original PDF page and checked the hashes
of all eight registered PDFs. GT was frozen before inference with SHA-256
`4cc918bc32cc6cd3ef2efb0817339ecfa6b776da755a529007d7d1b09c7f3ab2`.
No task-specific expected labels were sent to the model. There were no
`UNCERTAIN` or excluded tasks. Labels remain AI-reviewed, not independently
human-verified.

Inference used the `deepseek-flash` API alias, temperature 0, thinking disabled,
JSON-object response mode, strict host schema validation, and the existing ¥3
soft / ¥5 hard cost limits. No retry was made. The official DeepSeek model
notice mapped the alias to DeepSeek V4.1 Flash on the run date; because this is
a mutable alias, the exact served model revision is not pinned by this
experiment ([model notice](https://api-docs.deepseek.com/zh-cn/news/news260910/)).
The experiment sent bilateral evidence excerpts to the provider. The API key
was read from `DEEPSEEK_API_KEY`; it was not written to experiment artifacts.
Raw provider response bodies were not retained, so the saved `invalid_schema`
code does not identify which individual schema field failed.

## Results

Invalid outputs count as failures in end-to-end verdict and dimension accuracy.
That keeps the strict host contract inside the measured system. The confusion
matrix therefore has an `INVALID` column in addition to the three verdicts.

| Measure | Result |
| --- | ---: |
| Total / headline tasks | 17 / 17 |
| Ground truth | 5 equivalent, 6 different, 6 not comparable |
| Original-PDF review | 3 PDFs, 12 pages; 8 registered PDF hashes checked |
| Uncertain / excluded | 0 / 0 |
| API requests / API errors / not run | 17 / 0 / 0 |
| Invalid semantic outputs | 16 (`invalid_schema`) |
| Overall verdict accuracy | 1/17 (5.9%) |
| EQUIVALENT accuracy | 0/5 (0%) |
| DIFFERENT accuracy | 0/6 (0%) |
| NOT_COMPARABLE accuracy | 1/6 (16.7%) |

### Verdict confusion matrix

| GT \\ Predicted | EQUIVALENT | DIFFERENT | NOT_COMPARABLE | UNCERTAIN | INVALID |
| --- | ---: | ---: | ---: | ---: | ---: |
| EQUIVALENT | 0 | 0 | 0 | 0 | 5 |
| DIFFERENT | 0 | 0 | 0 | 0 | 6 |
| NOT_COMPARABLE | 0 | 0 | 1 | 0 | 5 |

### Dimension accuracy

| Dimension | Correct / assessed | Accuracy |
| --- | ---: | ---: |
| Object / field | 1/17 | 5.9% |
| Value | 0/17 | 0% |
| Unit | 0/17 | 0% |
| Condition / applicability | 1/17 | 5.9% |

There was one schema-valid semantic result. It got the expected `NOT_COMPARABLE`
verdict, object/field label, and condition label, but used `not_applicable` for
value and unit even though both sides contained the same known value and unit.
Thus the new contract and its synthetic consistency tests define the intended
rule, but this provider pass does not show that the model follows it reliably.

## Contract consistency and error taxonomy

`verdict_dimension_consistency` passed 1/1 among schema-accepted outputs; the
other 16 outputs could not be assessed for consistency. The accepted output's
value and unit dimension mismatches were each counted once. There were no
accepted verdict errors, condition errors, or explicit verdict/dimension
inconsistencies. The 16 invalid outputs are recorded separately as
`INVALID_OUTPUT`; their exact failing fields are unavailable because raw model
responses were not retained.

The `NOT_APPLICABLE` rule is explicit in the v2 contract and tested. Its
behavioral problem is **not resolved**: on the only accepted output, both known
value and unit were still labeled `not_applicable` under an object mismatch.
That is 2 violations in 2 applicable checks. With only one accepted response,
the 1/1 consistency score is not evidence of broad consistency.

## API usage and cost

| Measure | Result |
| --- | ---: |
| Prompt tokens | 12,759 |
| Completion tokens | 3,659 |
| Total tokens | 16,418 |
| Mean / maximum latency | 1,450.532 ms / 1,826.97 ms |
| Peak-rate estimated usage cost | ¥0.04676184 |
| Conservative reserved cost | ¥0.304298 |
| Soft / hard limit | ¥3 / ¥5; neither reached |

The estimates use the official DeepSeek pricing snapshot recorded with the
run: peak input cache-hit ¥0.04, cache-miss ¥2, and output ¥8 per million
tokens. DeepSeek lists lower off-peak rates, including weekends; the figures
above are peak-rate experiment estimates, not a billing statement
([official pricing](https://api-docs.deepseek.com/zh-cn/quick_start/pricing)).

## Answers to the 20 milestone questions

1. **Benchmark size:** 17 headline tasks.
2. **Verdict class counts:** 5 `EQUIVALENT`, 6 `DIFFERENT`, 6 `NOT_COMPARABLE`.
3. **Original-PDF review:** 3 contributing PDFs and 12 visually reviewed pages; each excerpt was re-extracted from its cited original page, and all 8 registered PDF hashes were checked.
4. **Uncertain and excluded:** 0 `UNCERTAIN`, 0 excluded.
5. **Overall verdict accuracy:** 1/17 (5.9%), counting invalid outputs as failures.
6. **EQUIVALENT accuracy:** 0/5 (0%).
7. **DIFFERENT accuracy:** 0/6 (0%).
8. **NOT_COMPARABLE accuracy:** 1/6 (16.7%).
9. **Confusion matrix:** see the table above; 16 responses were invalid, and the only valid verdict was `NOT_COMPARABLE` for an expected `NOT_COMPARABLE` task.
10. **Object/field accuracy:** 1/17 (5.9%).
11. **Value accuracy:** 0/17 (0%).
12. **Unit accuracy:** 0/17 (0%).
13. **Condition/applicability accuracy:** 1/17 (5.9%).
14. **Is `NOT_APPLICABLE` solved?** The rule is defined and mechanically tested; model adherence is not. The only accepted output incorrectly marked known value and unit as `not_applicable` (2/2 violations).
15. **Verdict/dimension consistency:** 1/1 among accepted outputs; 16 invalid responses were not assessable, so this is too small to establish stable consistency.
16. **Invalid outputs:** 16/17, all recorded as `invalid_schema`; exact field-level causes are unavailable because raw responses were not saved.
17. **API usage and cost:** 17 requests; 12,759 prompt, 3,659 completion, 16,418 total tokens; peak-rate estimated cost ¥0.04676184 and reserved cost ¥0.304298. The ¥3/¥5 limits were not reached.
18. **Does the comparator show practical value?** Not yet. The run demonstrates that strict host validation rejects malformed outputs, but accepted semantic evidence is too sparse and inaccurate to establish useful comparison performance.
19. **Should the benchmark be expanded now?** No. The 17-task cohort is balanced and PDF-reviewed; first diagnose schema incompatibility and contract adherence offline. Any later inference should follow a documented decision and must not reuse or alter this frozen GT after seeing model output.
20. **Can the project enter a full cross-document research Agent?** No. Decision `SEMANTIC_COMPARATOR_NOT_READY`; do not start Agent 2 or add Agent complexity at this stage.

## Reproduction and limits

Offline rescore of the saved run:

```powershell
python evaluation/evaluate_agent1_1.py rescore
```

The pass was one-shot. The runner refuses to overwrite the frozen GT or an
existing result. Do not delete local artifacts to repeat inference. Synthetic
tests validate parser and consistency mechanics; they are not industrial
semantic-accuracy evidence. Retrieval and Agent 0 were not run or changed.
Private PDFs, review notes, GT task details, excerpts, run results, and metrics
remain in ignored local output paths.
