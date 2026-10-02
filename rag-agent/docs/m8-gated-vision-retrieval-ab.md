# M8.6 — Gated Focused-VLM Retrieval A/B

**Decision: evaluated offline; not adopted into the default RAG pipeline.**

## 1. Motivation

M8.5 showed that 3× focused Qwen3-VL crops recovered strict target evidence in 3/5 OCR cases, but it did not test retrieval impact, false gate activations, or candidate interference. M8.6 tested a fixed metadata/content gate and added only gated VLM outputs as separate provenance-bearing Documents.

The experiment reused cached MinerU output. It did not reparse PDFs or change the default representation, retrieval, chunking, embedding, reranker, or generation prompt.

## 2. Q17 ground-truth review

The frozen QA source and expected page agree with the selected M8.5 input mapping. The selected page-11 image crop does not visibly establish both expected evidence fields. The cached structured block matches only one of two fields; M8.5's vision and OCR outputs do not match the complete evidence set. NFKC/case folding and whitespace removal do not resolve the mismatch, and a punctuation-insensitive diagnostic also failed to confirm it.

The evidence may belong to another visual region, or its page/block mapping may be wrong. The available evidence cannot distinguish those explanations, so Q17 is marked **GT_UNCERTAIN**. Its VLM output matches visible crop text, but that is not evidence recovery for Q17. The QA data and matcher were not changed; Q17 is excluded from formal M8.6 retrieval metrics.

## 3. Gate rule

The fixed, QA-blind rule was:

- Consider only table, image, chart, and list blocks with a valid normalized bounding box.
- Trigger when MinerU's existing structured readable text is shorter than 128 characters.
- Do not inspect the question, expected evidence, answer, or any LLM output when deciding whether to trigger.

For triggered blocks, the runner rendered a 3× crop and called the installed qwen3-vl:2b-instruct-q4_K_M with the unchanged M8.5 OCR prompt, temperature 0, and a 2,048-token output limit. Each result was kept as a separate Document with source alias, page, source block type/index, bbox, model, render scale, and fallback=true.

## 4. Positive cohort

The fixed M8.5 cohort was not reselected:

| Case | Block type | Structured text characters | Gate |
| --- | --- | ---: | --- |
| Q09 | table | 1,935 | rejected |
| Q10 | table | 804 | rejected |
| Q12 | table | 559 | rejected |
| Q17 | image | 98 | triggered; GT_UNCERTAIN |
| Q20 | list | 301 | rejected |

The four confirmed positive cases were Q09, Q10, Q12, and Q20. None triggered. Q17's trigger is reported separately and is not counted as a confirmed positive.

## 5. Negative controls

Ten unique MinerU blocks were frozen from questions that had complete evidence in the structured block and were Top-3 successes under M7.1: seven M6 baseline successes and three cases rescued by the M7 reranker. They cover seven table blocks, one list, one text block, and one image block. Selection was completed from the prior M6/M7 results before the M8.6 VLM calls.

## 6. Gate quality

| Measure | Result |
| --- | ---: |
| Confirmed positive triggers | 0/4 (0%) |
| Q17 uncertain trigger | 1/1, reported separately |
| Negative-control false activations | 3/10 (30%) |

The three false activations were NC01, NC06, and NC08. The text block in NC05 was rejected by the type filter.

A metadata-only scan of all 1,529 eligible cached blocks found 692 triggers (45.3%): 485 image, 121 table, 85 list, and 1 chart. This was not a VLM run. Trigger counts ranged from 10 to 411 per PDF, averaging 86.5. The result indicates that this threshold is not selective enough for corpus-wide inference.

## 7. Retrieval A/B

Both arms used the same 35-row M6 QA file (32 retrieval-scored cases), structured MinerU corpus, default chunking, embedding model, dense Top-20, BGE reranker, and reranker Top-3. Gate ON appended four Documents from the four triggered sample blocks. Q17 was excluded from formal metrics, leaving 31 scored questions. “Strict evidence” below is the existing evaluator's raw/case-folded evidence matcher; normalized evidence uses the existing M7 matcher.

| Metric | MinerU only | MinerU + gated VLM | Change |
| --- | ---: | ---: | ---: |
| Top-1 page | 17/31 | 17/31 | 0 |
| Top-3 page | 18/31 | 18/31 | 0 |
| Normalized evidence | 14/31 | 14/31 | 0 |
| Strict evidence | 11/31 | 11/31 | 0 |

The baseline is consistent with M7.1 after excluding Q17: its page hit is removed from both page denominators, while its evidence miss is removed from the evidence denominators.

For the four confirmed OCR positives, normalized and strict evidence remained 0/4 in both arms. Top-3 page hit remained 1/4.

## 8. Positive ranks and regressions

| Case | Gate | Baseline evidence rank | VLM candidate dense / reranker rank | Strict evidence before → after |
| --- | --- | --- | --- | --- |
| Q09 | rejected | not found in Top-20 | not generated | miss → miss |
| Q10 | rejected | not found in Top-20 | not generated | miss → miss |
| Q12 | rejected | not found in Top-20 | not generated | miss → miss |
| Q20 | rejected | not found in Top-20 | not generated | miss → miss |
| Q17 | triggered; uncertain | not found | outside Top-20 | excluded |

No confirmed positive evidence rank improved because the gate rejected all four confirmed OCR cases. Among the 11 questions that were M7.1 successes, none regressed in Top-3 page plus normalized evidence.


NC01's generated document produced two chunks containing the same fields: dense ranks 10 and 12, reranker ranks 2 and 3. Its original correct result remained rank 1, so this did not cause a measured regression. It does show that repeated Raw Text and Rows / Fields sections can create duplicate evidence candidates. The NC06 and NC08 vision candidates were outside dense Top-20.

## 9. Factual-risk review

All four generated outputs were compared with their local crops:

| Case | Review | Finding |
| --- | --- | --- |
| Q17 | SUPPORTED | The crop transcription was supported; the QA-to-crop evidence mapping remains uncertain. |
| NC01 | SUPPORTED | Visible table values and row associations were preserved. |
| NC06 | SUPPORTED | The visible identifier, specification, material, length, and quantity matched the image. |
| NC08 | PARTIAL | Visible labels and dimensions were extracted, but their association with drawing axes was not retained. |

No unsupported digit or model/code was observed. The local review file records 3 SUPPORTED, 1 PARTIAL, 0 INCORRECT, and 0 UNCERTAIN model-output reviews. Q17's separate ground-truth status remains GT_UNCERTAIN.

## 10. Runtime cost

- Actual cohort inference: 4 calls across 3 of the 8 PDFs.
- Mean request time: 3.06 seconds across those four calls; the first call took 9.15 seconds and the other three were warm calls.
- Sample average: 0.5 calls per PDF across all eight PDFs, or 1.33 per PDF touched by this small cohort.
- Corpus-wide projection under the same gate: 86.5 calls per PDF on average, with a 10–411 range. At the observed mean request time, this is roughly 4.4 minutes of inference per PDF on average, excluding rendering, embedding, and reranking.

The corpus-wide projection is derived from cached block metadata only. The experiment did not issue those 692 calls and is not a production throughput benchmark.

## 11. Decision

The VLM output was factually supported or partial on these four selected crops, but the gate missed all four confirmed OCR positives, activated on 3/10 known-success controls, and left retrieval metrics unchanged. One false activation also generated duplicate Top-3 chunks. These results do not support formal focused-VLM integration or use of a larger vision model; the current bottleneck is the gate's inability to identify incomplete blocks from available text-length/type metadata.

This experiment does not test Hybrid Search and provides no evidence for or against BM25. It also does not justify broad reranker changes.

## 12. M8 freeze recommendation

Freeze M8.1–M8.6 as offline measured experiments. Keep the default MinerU structured pipeline unchanged and do not adopt this gate. Do not scale this exact gate to the corpus: its metadata-only projection is too broad. If the Document Intelligence track resumes, first establish a reliable non-LLM quality signal for OCR completeness; only then would another focused fallback experiment be interpretable. A larger VLM is not supported by this result.

The cohort is small (four confirmed positives and ten negative controls), and Q17 remains unresolved. Rates here describe these fixed local cases, not general performance.
