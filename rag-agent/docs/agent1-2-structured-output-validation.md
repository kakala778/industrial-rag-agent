# Agent 1.2 — Structured Output Contract Validation

- **Decision:** `STRUCTURE_SOLVED_SEMANTIC_COMPARATOR_NOT_READY`; do not start Agent 2.
- **Run date:** 2026-10-03.
- **Ground truth:** the same 17-task Agent 1.1 freeze, SHA-256 `4cc918bc32cc6cd3ef2efb0817339ecfa6b776da755a529007d7d1b09c7f3ab2`.
- **Model / prompt:** `deepseek-flash`, temperature 0, reasoning disabled, and the unchanged semantic v2 instructions.

## Scope and method

Agent 1.2 changed only the output transport: DeepSeek Responses API with
`text.format.type=json_schema`, using a schema generated from the existing v2
contract. The runner still applies the original Host schema and verdict /
dimension consistency checks after parsing. The request uses the documented
Responses `instructions`, `input`, `reasoning`, and `text.format` fields
([Responses API guide](https://api-docs.deepseek.com/guides/responses_api),
[Create Response reference](https://api-docs.deepseek.com/api/create-response/)).

The same GT, task texts, evidence excerpts, model alias, prompt, semantic
dimensions, verdict definitions, `NOT_APPLICABLE` semantics, and consistency
rules were used. No retrieval, Agent 0 action, or citation renderer ran. The
saved Agent 1.1 run was read-only and its bytes remained unchanged. The model
alias was `deepseek-flash`; its served revision is provider-controlled and not
pinned by this experiment.

The historical Agent 1.1 raw responses were not saved. Its 16 generic
`invalid_schema` outcomes therefore cannot be retrospectively assigned to
specific taxonomy categories. For Agent 1.2, all 17 provider responses completed
and were retained, together with parsed output, validation error, status,
usage, and IDs, under the Git-ignored `outputs/agent1_2/`. The API key was not
stored. The report contains only aggregate measurements.

## Results

Acceptance below means acceptance by the complete Host contract, including its
consistency validation. Provider response completion is reported separately.

| Measure | Agent 1.1 `json_object` | Agent 1.2 Responses `json_schema` |
| --- | ---: | ---: |
| Provider responses completed | 17/17 | 17/17 |
| Host contract accepted | 1/17 (5.9%) | 15/17 (88.2%) |
| Strict end-to-end verdict accuracy | 1/17 (5.9%) | 14/17 (82.4%) |
| Accepted-output verdict accuracy | 1/1 (100%) | 14/15 (93.3%) |
| Invalid outputs with known failure type | 0/16 | 2/2 `CONSISTENCY_ERROR` |
| API errors / retries | 0 / 0 | 0 / 0 |

Agent 1.2 host acceptance increased by 82.4 percentage points. The two rejected
responses had structurally valid output but contradicted the existing
verdict/dimension consistency rules. No output was accepted by relaxing the
contract.

| Expected verdict | Tasks | Agent 1.2 accepted | Accepted-output correct | Strict correct / all tasks |
| --- | ---: | ---: | ---: | ---: |
| `EQUIVALENT` | 5 | 5 | 5/5 (100%) | 5/5 (100%) |
| `DIFFERENT` | 6 | 6 | 5/6 (83.3%) | 5/6 (83.3%) |
| `NOT_COMPARABLE` | 6 | 4 | 4/4 (100%) | 4/6 (66.7%) |

The accepted-output scores are conditional on passing the Host contract. The
strict column counts both rejected `NOT_COMPARABLE` responses as failures.

| Dimension, among 15 accepted outputs | Correct | Accuracy |
| --- | ---: | ---: |
| Object / field | 15/15 | 100% |
| Value | 13/15 | 86.7% |
| Unit | 8/15 | 53.3% |
| Condition / applicability | 7/15 | 46.7% |

The two consistency errors bring Host consistency to 15/17 (88.2%) among
outputs that were structurally parseable. The known-dimension
`NOT_APPLICABLE` diagnostic found 2 violations in 8 applicable value/unit
checks when the expected object differed. Agent 1.1 had 2/2 such violations in
its sole accepted output; its 16 rejected outputs were not assessable.

The Agent 1.2 invalid-output taxonomy was: `CONSISTENCY_ERROR` 2;
`INVALID_JSON`, `MISSING_FIELD`, `EXTRA_FIELD`, `INVALID_ENUM`, `WRONG_TYPE`,
`NESTED_SCHEMA_ERROR`, `EMPTY_RESPONSE`, `TRUNCATED`, and `OTHER` all 0. The
Agent 1.1 historical taxonomy remains unclassified for all 16 generic
`invalid_schema` outputs because the raw bodies do not exist.

| Agent 1.2 API usage | Result |
| --- | ---: |
| Requests / errors / retries | 17 / 0 / 0 |
| Input / output / total tokens | 18,046 / 2,374 / 20,420 |
| Mean / maximum latency | 1,316.467 ms / 1,513.228 ms |
| Peak-rate estimated cost | ¥0.03651888 |
| Conservative reserved cost | ¥0.341970 |
| Soft / hard ceiling | ¥3 / ¥5; neither reached |

The cost estimate uses the existing official pricing snapshot: input cache hit
¥0.04, cache miss ¥2, and output ¥8 per million tokens
([DeepSeek pricing](https://api-docs.deepseek.com/zh-cn/quick_start/pricing)).
This is an estimate, not a billing statement.

## Answers to the milestone questions

1. **Agent 1.1 schema acceptance:** 1/17 (5.9%) passed the full Host contract.
2. **Agent 1.2 schema acceptance:** 15/17 (88.2%) passed the full Host contract; all 17 provider responses completed.
3. **Invalid output types:** Agent 1.2 had two `CONSISTENCY_ERROR` responses and zero in every other tracked category. Agent 1.1's 16 generic failures cannot be classified without saved raw responses.
4. **Did Responses / `json_schema` solve the transport issue?** It resolved the structural-output bottleneck for this pass: structural output was parseable on all 17 responses, while full Host acceptance was 15/17 because two failed consistency checks. The 15/17 acceptance versus 1/17 is strong comparative evidence that output transport was the main Agent 1.1 failure source. It does not prove that every historical failure had a transport cause because Agent 1.1 raw bodies were unavailable.
5. **Strict end-to-end verdict accuracy:** 14/17 (82.4%); the two invalid outputs and one wrong accepted verdict are failures.
6. **Accepted-output verdict accuracy:** 14/15 (93.3%).
7. **Class results:** accepted-only: `EQUIVALENT` 5/5, `DIFFERENT` 5/6, `NOT_COMPARABLE` 4/4. Counting invalid outputs as failures: 5/5, 5/6, and 4/6 respectively.
8. **Object / field accuracy:** 15/15 (100%) among accepted outputs.
9. **Value accuracy:** 13/15 (86.7%).
10. **Unit accuracy:** 8/15 (53.3%).
11. **Condition / applicability accuracy:** 7/15 (46.7%).
12. **Known-dimension N/A violations:** 2/8 in Agent 1.2; 2/2 in Agent 1.1's only accepted output.
13. **Verdict / dimension consistency:** 15/17 (88.2%); two responses were rejected by the unchanged Host consistency validator.
14. **API usage and cost:** 17 requests, 18,046 input and 2,374 output tokens, 1,316.467 ms mean latency, ¥0.03651888 estimated cost, ¥0.341970 conservative reservation, 0 API errors, and 0 retries.
15. **Current bottleneck:** output transport was the dominant Agent 1.1 problem and is substantially improved. The remaining issue is semantic contract adherence / model capability, especially unit and condition judgments and consistency. This experiment cannot distinguish model limitations from ambiguity in the frozen semantic contract. The GT remains AI-reviewed rather than independently human-verified; no specific GT defect was identified in this run.
16. **Evidence to enter Agent 2:** no. This is Case C: structure is substantially solved, but weak unit/condition accuracy, two consistency rejections, and the small AI-reviewed GT leave the semantic comparator unvalidated. Do not start Agent 2 or relax the schema to improve acceptance.

## Reproduction and limits

The one authorized paid pass has already run. Its runner refuses to overwrite
existing outputs, so do not run `run` again in this checkout. The offline
rescore can be repeated without provider calls:

```powershell
python evaluation/evaluate_agent1_2.py rescore
```

The saved metrics and run are in ignored `outputs/agent1_2/`; the full raw
provider records must remain local. This 17-task experiment does not establish
industrial reliability or production readiness.
