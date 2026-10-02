# M8.3 — Vision Model Feasibility Test

## Experiment Purpose

M8.1 found five OCR failures, two image-information losses, and one
layout-relation failure among the 11 M6 parsing failures. M8.2 showed that
labeling existing MinerU image text and retaining block metadata did not recover
the missing evidence. M8.3 tested whether the locally installed
`qwen3-vl:2b-instruct-q4_K_M` could add useful information from the original
page images.

This was a three-page feasibility check, not a Visual RAG implementation. The
PDFs were rendered read-only. No OCR service, additional model, RAG component,
or existing Qwen3-4B generation path was changed.

## Test Cases

Cases were chosen from the M8.1 categories before running the vision model.
Selection was based on the audited failure type and the availability of a
page-grounded evidence target, not on model results.

| Case | Type | MinerU structured and current retrieval | Vision output value | New RAG Document block? |
| --- | --- | --- | --- | --- |
| Q10 | `OCR_FAILURE` | The audited table value is absent while row/cell structure is retained. The target page and normalized evidence were both missed in reranker Top-3. | Did not reproduce the expected evidence; only 2/6 required section headings were present. | No |
| Q16 | `IMAGE_INFORMATION_LOSS` | The audited drawing annotation is absent from structured text. The target page and normalized evidence were both missed in reranker Top-3. | All 6/6 headings were present, but the expected evidence was not recovered. | No |
| Q32 | `LAYOUT_RELATION_FAILURE` | Diagram labels and paths are separate blocks without their association. Reranker Top-3 hit the page, but not normalized evidence. | Did not reproduce the expected evidence; only 3/6 headings were present and the relation section had no meaningful content. | No |

The existing evaluation's normalized evidence matcher was reused. None of the
three vision responses matched the complete expected evidence for its case.
Therefore, no case is counted as improved. For Q32, finding the page was not
enough to establish that the model recovered the missing relation.

## Model and Method

- Model tag: `qwen3-vl:2b-instruct-q4_K_M`
- Ollama-reported family: `qwen3vl`
- Parameter size: `2.1B`
- Quantization: `Q4_K_M`
- Ollama version: `0.34.4`
- Interface: local Ollama `/api/generate`
- Generation: temperature `0`, maximum `2048` generated tokens
- Image input: one full-page render per selected case, produced from the
  original PDF at 2× page scale
- Prompt: the fixed Chinese prompt specified for M8.3; the existing Qwen3-4B
  RAG prompt was not changed

The page images and unredacted model responses were kept only in the Git-ignored
local experiment output. This report contains no PDF names, source paths,
question text, expected answers, evidence strings, or document text.

## Output Quality Analysis

The model did not recover the target evidence on any of the three pages under
the evaluation matcher. Output structure was inconsistent: Q10 and Q32 omitted
several requested headings, while Q16 followed all headings but still did not
recover its target. Q32's relation section did not provide a usable relation
description.

The length or fluency of a response was not treated as evidence of correctness.
Unmatched generated descriptions were not accepted as facts or proposed for
indexing. This experiment did not perform a complete human fact-by-fact review
of every generated statement, so it cannot establish the model's precision or
hallucination rate.

## Comparison with MinerU

| Case | MinerU target information | Current retrieval | Vision target recovery | Feasibility result |
| --- | --- | --- | --- | --- |
| Q10 | Missing | Page miss; evidence miss | No | Not useful for this OCR case |
| Q16 | Missing | Page miss; evidence miss | No | Not useful for this image-information case |
| Q32 | Missing relation | Page hit; evidence miss | No; no meaningful relation section | Not useful for this layout case |

The test did not demonstrate that the 2.1B vision model can supplement the
current MinerU representation for these representative failures. It also does
not establish that a larger model would succeed: the sample contains only one
case per category, and the output assessment is limited to evidence recovery
and requested-section coverage.

## Decision

- Do not integrate this model into the RAG pipeline or add its output to the
  indexed Documents based on these results.
- M8.4 should not start as a Visual RAG implementation. If visual extraction
  remains a priority, a bounded model-comparison experiment could be justified
  with the same fixed pages, explicit relation/table criteria, and human review
  of factual correctness.
- A larger vision model is not yet proven necessary or sufficient. Do not
  download one for this result alone. If a larger model is already available
  locally, compare it under the same protocol before making an architecture
  decision.

## Reproduction and Privacy

From `rag-agent/`, run:

```powershell
python evaluation/run_m8_vision_feasibility.py
```

The runner verifies the existing local M6 input manifest, matching MinerU
caches, and M8.1/M8.2 case labels before making local Ollama calls. It selects
only Q10, Q16, and Q32. Rendered pages, raw responses, and an anonymous metric
summary stay under a Git-ignored output directory; none are included in this
report or tracked by Git.
