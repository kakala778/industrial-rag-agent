# Agent benchmark GT review status

**Status:** `READY_FOR_SEMANTIC_AGENT_EXPERIMENT_WITH_PROVISIONAL_GT`
**Review authority:** AI-assisted review against the original PDFs
**Review date:** 2026-10-03

## What changed

The pre-semantic-comparison audit found contamination in the provisional
expected-evidence IDs and one condition whose negation was reversed. A derived
patch manifest was generated from the audit report and row-level review notes;
the audit CSV's `gt_change_proposal` field was not treated as authoritative.

- Removed **4 invalid expected-ID occurrences** and **1 partial/redundant
  expected-ID occurrence**.
- Added **3 expected-ID occurrences**: the confirmed target for R02/A and R04/A,
  plus the relevant second figure for R07/B.
- Corrected **1 condition** in R03/A to preserve the PDF's negation and its
  associated copper-bar limitation.
- Made **2 narrow condition-precision updates** for a multi-strand conductor
  and an explicit video-equipment-box object.
- Reclassified **6 candidate labels**, including the R07/B Figure 2b candidate.

The source audit sheet reports 14 reviewed rows: `SUPPORTED 6`, `PARTIAL 7`,
`NOT_APPLICABLE 1`, `INCORRECT 0`, `UNCERTAIN 0`; confidence is HIGH for 13 rows
and MEDIUM for one. The derived GT is marked `ai_pdf_reviewed`; original
`pending_human_pdf_review` provenance remains recorded. This is not independent
human verification.

## Offline rescore

Saved Agent 0.2 and Agent 0.3 actions were rescored against derived copies of
the repaired oracle and candidate reviews. Frozen SEARCH observations and raw
runs were not changed. The utility reports no inference or retrieval and checks
source hashes and protected inputs. Two consecutive runs produced identical
`rescore-results.local.json` bytes.

| Metric (before → after repair) | Agent 0.2 Qwen | Agent 0.2 DeepSeek | Agent 0.3 DeepSeek reference |
|---|---:|---:|---:|
| Candidate-available / retrieval-bound tasks | 5 / 2 → 5 / 2 | 5 / 2 → 5 / 2 | 5 / 2 → 5 / 2 |
| ID-selection success | 1/5 → 1/5 | 5/5 → 5/5 | 5/5 → 5/5 |
| Correct relevant IDs / attempted IDs | 6/10 → 6/10 | 9/9 → 9/9 | 9/9 → 9/9 |
| Fully relevant task success | 0/5 → 0/5 | 4/5 → 4/5 | 5/5 → 5/5 |
| Field alignment | 6/10 → 6/10 | 9/9 → 9/9 | 9/9 → 9/9 |
| Unit alignment | 5/7 → 5/7 | 7/7 → 7/7 | 7/7 → 7/7 |
| Condition/applicability alignment | 2/7 → 2/7 | 6/7 → 6/7 | 7/7 → 7/7 |
| Unsupported-side correctness | 0/1 → 0/1 | 1/1 → 1/1 | 1/1 → 1/1 |

Every listed metric matches the original reports' headline values. R02 and R04
remain `RETRIEVAL_BOUND` for both arms and both runs: the repaired target IDs are
absent from the frozen Top3 observations. These tasks remain outside
candidate-available selector accuracy and must not be described as model
selection failures.

## Readiness and limits

The confirmed expected-ID contamination and R03/A condition error are repaired,
and offline rescoring shows no unresolved evidence or condition conflict that
blocks a semantic Agent experiment. The benchmark can proceed with the derived
GT explicitly labeled provisional. R05/B remains a **medium-confidence,
document-specific provisional unsupported-side judgment** based on AI visual
review of the current 27-page PDF; it does not establish that external standards
lack the field. Comparison operators and normative modality such as `不小于`
and `宜` are not represented by the current condition-group schema and remain
out of scope for this targeted repair.

The selected references' saved host authenticity counters were copied from the
original Agent 0.3 run and were not recomputed as part of GT rescoring. No raw
model answers, evidence excerpts, industrial PDFs, API data or candidate-level
review details are published here. Detailed experimental context remains in the
[Agent 0.2 report](agent0-2-evidence-selection-model-comparison.md) and
[Agent 0.3 report](agent0-3-bounded-evidence-contract.md); derived manifests,
repaired JSON and rescore output stay under ignored `outputs/`.
