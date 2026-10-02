# M8.5 — Focused OCR Comparison

## 1. Motivation

M8.1 identified five `OCR_FAILURE` cases. M8.2's representation variants did
not recover any of them. M8.3's full-page Qwen3-VL run recovered 0/3 selected
cases, while M8.4 recovered the Q10 table target from a 3× MinerU block crop.
M8.5 tests whether that crop result generalizes across the fixed five-case
OCR cohort and whether a dedicated local OCR engine is a better fallback.

This is an offline comparison. No OCR/VLM output was added to the RAG index or
default MinerU representation.

## 2. Fixed Cases and Crop Strategy

The cohort is the complete M8.1 OCR group, selected before this run. Block
indices and normalized bounding boxes were checked against the hash-matched
MinerU Middle JSON cache. All five targets had a usable block bbox; no union
crop or full-page input was needed. Table crops use their audited table blocks
and therefore include multiple rows rather than a hand-selected cell.

| Case | Page | Target block | Normalized bbox | Expected fields |
| --- | ---: | --- | --- | ---: |
| Q09 | 20 | table / 3 | `[0.934, 0.523, 0.999, 0.976]` | 1 |
| Q10 | 24 | table / 6 | `[0.932, 0.545, 0.999, 0.976]` | 1 |
| Q12 | 30 | table / 18 | `[0.762, 0.621, 0.995, 0.882]` | 2 |
| Q17 | 11 | image / 2 | `[0.535, 0.348, 0.653, 0.510]` | 2 |
| Q20 | 6 | list / 28 | `[0.089, 0.684, 0.867, 0.832]` | 1 |

Each exact bbox was rendered from the original local PDF at 2× and 3× while
preserving aspect ratio. Each rendered crop was sent unchanged to Qwen3-VL and
RapidOCR. The 3× result is primary; 2× measures resolution sensitivity. The
runner verified the existing M6 input manifest and MinerU caches and did not
invoke MinerU again.

## 3. OCR Methods

- **MinerU baseline:** current `structured` block text and the corresponding
  readable Middle JSON fields from the same cached target block. Both produced
  the same aggregate target-field scores in this cohort.
- **Qwen3-VL:** local Ollama model `qwen3-vl:2b-instruct-q4_K_M`, reported as
  2.1B parameters, Q4_K_M, 1.9 GB; Ollama 0.34.4. The fixed OCR prompt asks
  for faithful transcription and table rows/fields, with temperature 0 and a
  2048-token output limit. Ollama reported all model bytes in VRAM immediately
  after inference.
- **Dedicated OCR:** RapidOCR 3.9.2 with ONNX Runtime 1.30.0 in a separate
  ignored Python 3.12 environment. All three inference sessions used
  `CPUExecutionProvider`. The shipped ONNX model files total 31,749,509 bytes
  (31.75 MB). See the [official installation guide](https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/install/)
  and [official release](https://github.com/RapidAI/RapidOCR/releases/tag/v3.9.2).

## 4. Results

The existing normalized evidence matcher is reused. It normalizes Unicode and
case, removes whitespace, and uses substring matching. To catch a numeric or
identifier substring inside a different token, the experiment also checks
critical tokens separately. **Strict recovery** means all expected fields
match and every expected critical token matches exactly. It still does not
prove that a value is associated with the correct table row or that unrelated
transcribed fields are faithful.

| Case | MinerU fields | Qwen3-VL 3× fields; critical tokens | RapidOCR 3× fields; critical tokens |
| --- | ---: | --- | --- |
| Q09 | 0/1 | 1/1; 1/1 | 1/1; 0/1 |
| Q10 | 0/1 | 1/1; 1/1 | 1/1; 0/1 |
| Q12 | 1/2 | 1/2; 1/1 | 1/2; 0/1 |
| Q17 | 1/2 | 0/2; 0/2 | 0/2; 0/2 |
| Q20 | 0/1 | 1/1; 1/1 | 1/1; 1/1 |

The normalized evidence matcher marked Q09 and Q10 as hits for RapidOCR, but
the expected numeric token was only a substring of a different, longer
numeric token. They are not strict recoveries. Q20's critical identifier
appears in the block even when the complete field is absent from MinerU; this
is why critical-token presence by itself is not treated as evidence recovery.

| Measure | MinerU | Qwen 2× | Qwen 3× | RapidOCR 2× | RapidOCR 3× |
| --- | ---: | ---: | ---: | ---: | ---: |
| Normalized full-evidence hits | 0/5 | 3/5 | 3/5 | 3/5 | 3/5 |
| Strict evidence recoveries | 0/5 | 2/5 | 3/5 | 1/5 | 1/5 |
| Matched fields | 2/7 | 4/7 | 4/7 | 4/7 | 4/7 |
| Exact critical tokens | 3/6 | 3/6 | 4/6 | 1/6 | 1/6 |
| Critical-token errors | 3 | 3 | 2 | 5 | 5 |

The seven expected fields contain four numeric critical tokens and two
alphanumeric codes. By type, MinerU matched 1/4 numeric tokens and 1/2 codes;
Qwen3-VL at 3× matched 3/4 numeric tokens and 1/2 codes; RapidOCR matched 0/4
numeric tokens and 1/2 codes. No target evidence field in this cohort
supplied an expected unit token, so the run cannot compare unit accuracy.
Field recall is an upper bound because the existing matcher accepts
substrings; strict evidence and critical-token results are more informative
for these numeric cases.

### Case Notes

- **Q09:** Qwen3-VL recovered the complete field and exact critical token at
  both scales. RapidOCR's normalized hit was not an exact numeric match.
- **Q10:** Qwen3-VL had a normalized hit at 2× but failed the exact critical
  token check; at 3× it passed both. RapidOCR remained an inexact numeric
  substring hit at both scales.
- **Q12:** Both OCR systems matched only one of two fields. Qwen3-VL matched
  the expected critical token, but neither system recovered full evidence.
- **Q17:** Neither crop OCR method recovered either expected field from the
  image block. MinerU retained one of the two fields.
- **Q20:** Both crop OCR methods recovered the complete field and exact
  critical token; MinerU's target list block did not contain the complete
  field.

## 5. Runtime and Resource Observations

At the primary 3× scale, Qwen3-VL averaged 6.66 seconds per crop; its first
Ollama load took 4.36 seconds and later reported load times below 0.04 seconds
per request. RapidOCR averaged 10.41 seconds per crop on CPU and initialized
in 0.51 seconds. The run used a GPU for Qwen3-VL and CPU for RapidOCR, so these
latencies are observations of this local setup, not a device-matched speed
benchmark.

## 6. Visual Fact Check and Limitations

The five source crops and primary 3× outputs were compared locally. No clearly
unsupported factual field was confirmed in the reviewed target regions
(confirmed count: 0). This is not a general hallucination-rate estimate:
Q09/Q10 include handwritten signature areas that remain unverified, and
critical-token presence still does not prove a value is attached to the right
table row.

| Case | Qwen3-VL visual review | RapidOCR visual review | Note |
| --- | --- | --- | --- |
| Q09 | Partial | Partial | Table transcription is visually grounded; handwritten title/signature region remains uncertain. |
| Q10 | Partial | Partial | Table transcription is visually grounded; handwritten title/signature region remains uncertain. |
| Q12 | Partial | Partial | RapidOCR drops superscript unit markers in multiple table fields. |
| Q17 | Correct for visible crop text | Correct for visible crop text | Both transcribe the visible annotation; the existing M6 normalized matcher still reports 0/2 expected fields. It is not counted as an automated recovery. |
| Q20 | Correct | Correct | Transcribed text aligns with the visible block; differences are spacing/punctuation. |

Q17's visual transcription and normalized evidence result disagree. The
evidence-to-page mapping or its text normalization may need review before
assigning a cause; QA/ground truth was left unchanged. The local
`human_review.local.json` separates this visual assessment from automated
target-evidence scores and records the unverified signature regions.

## 7. Decision and Architecture Implications

- Focused Qwen3-VL performed better than the tested dedicated OCR on strict
  target recovery: 3/5 versus 1/5 at 3×, with 4/7 versus 4/7 matched fields
  and 4/6 versus 1/6 exact critical tokens. This is a promising small-sample
  signal, not a precision estimate.
- **Q10:** M8.3's full-page Qwen3-VL missed; M8.4's 3× block crop recovered
  the target; M8.5's OCR-oriented Qwen3-VL crop passed the exact token check
  at 3×. In this run RapidOCR did not. The 2×/3× pair shows resolution
  mattered for Q10, but the cross-stage comparison also changed prompt and
  evaluation setup, so it does not isolate crop versus prompt as the sole
  cause.
- A **gated focused-VLM OCR prototype** is worth a further offline A/B test;
  a default-pipeline integration is not justified yet. A plausible trigger is
  a table/image/list block with absent or incomplete MinerU text, but the
  trigger must be tested against ordinary blocks to measure false activations.
- If trialed, keep the OCR result as a **separate provenance-bearing Document
  candidate** (source/page/block/bbox/model/scale), not only metadata: extracted
  text must be searchable. Do not index it by default until full-output
  factual review and retrieval A/B show acceptable precision.
- Do not download a larger VLM based on this cohort. RapidOCR did not outperform
  the local VLM here. Hybrid Search is not supported or ruled out by this
  parsing-focused experiment; M8.5 did not measure retrieval failures.
- M8.5's fixed-crop comparison and visual review are recorded. The remaining
  uncertainty is localized to handwritten title/signature text and the Q17
  evidence-matcher discrepancy; it does not justify changing QA or default
  indexing.
- Before a retrieval A/B, review the Q17 evidence mapping privately and use a
  negative-control block set to measure false activations. If that check
  supports the signal, proceed with a small gated OCR-block retrieval A/B.

## 8. Reproduction and Privacy

From `rag-agent/`, with the local M6 data, matching MinerU cache, Ollama model,
and isolated RapidOCR environment available:

```powershell
..\minerU\mineru-405-poc\.venv\Scripts\python.exe evaluation\run_m8_focused_ocr_comparison.py
```

The runner validates the fixed M6 manifest and cache, uses only the five
preselected M8.1 OCR cases, and writes rendered crops, anonymous metrics, raw
responses, and a local review file under Git-ignored
`outputs/m8_focused_ocr/`. It does not store question text, answer text, target
evidence strings, source filenames, or document text in the anonymous summary.
No PDF, QA file, crop, MinerU cache, or model file is part of the experiment
diff.
