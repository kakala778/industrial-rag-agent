# M8.2 — Visual Information Representation Experiment

## Experiment Motivation

M8.1 identified 11 parsing failures among the original M6 failures: five OCR
failures, two image-information losses, one layout-relation failure, and three
uncertain cases. This experiment tested whether making existing MinerU image
text explicit and retaining more block metadata would recover expected
evidence or change the M7 retrieval metrics.

The run used the same local M6 QA set (35 rows, 32 scored questions), the same
eight PDFs, the same hash-matched MinerU cache, embedding model, chunking,
retrieval, and BGE reranker. All eight PDF/cache pairs passed the manifest and
content-hash check. The experiment did not rerun MinerU, invoke external OCR,
or add a model. The raw QA, PDFs, and MinerU output remain local.

## Representation Difference

| Mode | Text supplied to chunking and embedding | Metadata |
| --- | --- | --- |
| `structured` (`structured_baseline`) | Existing ordered text groups and typed table/chart/image text. Readable `image_body` text and captions are already included as plain text. | Existing source, page, block type, and block index. |
| `structured_ocr` | Same existing content, with image captions and readable image-body text labelled as `Caption` and `OCR text`. It also accepts allowlisted OCR-like child fields if present in Middle JSON. | Same metadata as `structured`. |
| `structured_full` | Same labeled text as `structured_ocr`. | Adds valid block bounding boxes, image index derived from the MinerU image block index, subtype, continuation flag, caption metadata, and per-block geometry for grouped text. |

The local cache contains `image_body`, `image_caption`, and `image_footnote`
children, but no separate `ocr_text`, `image_text`, or `extracted_text`
children. Since the current baseline already includes readable `image_body`
text, `structured_ocr` changed its labeling but did not add newly recognized
text. `structured_full` adds metadata only; metadata is carried through
chunking but is not included in embedding text.

## Results

| Mode | Dense Top-1 page | Dense Top-3 page | Dense normalized evidence | Dense + reranker Top-1 page | Dense + reranker Top-3 page | Dense + reranker normalized evidence | Original parsing failures remaining |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `structured` (`structured_baseline`) | 9/32 | 15/32 | 10/32 | 18/32 | 19/32 | 14/32 | 11/11 |
| `structured_ocr` | 8/32 | 13/32 | 9/32 | 16/32 | 18/32 | 13/32 | 11/11 |
| `structured_full` | 8/32 | 13/32 | 9/32 | 16/32 | 18/32 | 13/32 | 11/11 |

The `structured` reranker results reproduce the M7.1 Top-3 page and normalized
evidence results (19/32 and 14/32). Explicitly labeling existing image text
reduced the aggregate metrics by one to two cases. Retaining metadata in
`structured_full` did not change ranking metrics relative to
`structured_ocr`.

`structured_full` retained bounding boxes on 911/911 emitted visual-block
documents; 231/231 image documents carried an image index, 137/231 carried a
caption field, and 683 text-group documents carried block geometry. Bounding
box metadata propagated to 2,045 chunks. This demonstrates metadata retention
in the retrieval results; it does not establish improved citation usability in
a user interface.

## Failure Analysis

None of the original 11 cases gained complete expected evidence in the loaded
source/page documents under any mode. All remained parsing failures by the
existing normalized evidence matcher.

| Question ID | M8.1 category | Reranker Top-3 page hit: structured → OCR → full | Normalized evidence hit: structured → OCR → full |
| ---: | --- | --- | --- |
| 9 | OCR_FAILURE | No → No → No | No → No → No |
| 10 | OCR_FAILURE | No → No → No | No → No → No |
| 12 | OCR_FAILURE | Yes → Yes → Yes | No → No → No |
| 14 | OTHER/UNCERTAIN | Yes → Yes → Yes | No → No → No |
| 16 | IMAGE_INFORMATION_LOSS | No → No → No | No → No → No |
| 17 | OCR_FAILURE | Yes → No → No | No → No → No |
| 20 | OCR_FAILURE | No → Yes → Yes | No → No → No |
| 29 | IMAGE_INFORMATION_LOSS | No → No → No | No → No → No |
| 30 | OTHER/UNCERTAIN | No → No → No | No → No → No |
| 31 | OTHER/UNCERTAIN | No → No → No | No → No → No |
| 32 | LAYOUT_RELATION_FAILURE | Yes → Yes → Yes | No → No → No |

Question 20 gained a Top-3 page hit in the OCR-labelled runs, while question
17 lost that page hit. Neither contains the expected normalized evidence, so
the page movement is not a parsing-failure fix. The two image-information
cases remain unresolved even when their retrieved Top-3 visual blocks carry
bounding boxes in `structured_full`.

By M8.1 category, the remaining counts are OCR_FAILURE 5/5,
IMAGE_INFORMATION_LOSS 2/2, LAYOUT_RELATION_FAILURE 1/1, and
OTHER/UNCERTAIN 3/3. The experiment did not add OCR text that MinerU had
omitted from its Middle JSON.

## Decision

- Keep `structured` as the default. The OCR-labelled representation did not
  reduce parsing failures and slightly reduced the M7 metrics in this run.
- `structured_full` is useful as an opt-in representation when downstream
  citation or review code needs block geometry. This experiment did not show
  a retrieval gain from metadata alone.
- A vision model is not justified for general adoption by this three-mode
  experiment. A small feasibility test on the two image-information cases and
  the one layout-relation case could be considered; this run cannot predict
  whether a visual model would recover their missing evidence.
- A table row/column representation change is not supported: M8.1 confirmed no
  table-structure failures. Three OCR failures involve table text, so a
  separate, bounded table-cell recognition test may be useful.
- The current failure sample is small and several cases remain uncertain. No
  broad document-processing direction should be selected from this run alone.

## Reproduction and Privacy

Run from `rag-agent/` with the original local M6 inputs and matching M7
failure-analysis output available at their standard local paths. The script
uses fixed local QA/PDF/M7 paths and rejects a cohort that does not match the
35-row, 32-scored-question, eight-PDF M6 shape and the 11 audited M7 cases.
The first local setup pins input hashes and counts in an ignored manifest; if
that manifest is absent, initialize it explicitly before running the experiment:

```powershell
python evaluation/run_m8_visual_representation_experiment.py --initialize-input-manifest
python evaluation/run_m8_visual_representation_experiment.py
```

The manifest contains only content hashes and cohort counts. It does not store
source names or document/QA text, and it is not tracked by Git. Initialization
also requires all eight hash-matched MinerU caches and refuses to overwrite an
existing manifest.

The script verifies matching MinerU caches before loading the three modes and
stops without invoking MinerU if any cache is missing or stale. It writes only
anonymous aggregate and per-question flags to the Git-ignored
`outputs/m8_visual_representation_comparison.json`. The JSON contains no
question text, expected answer, evidence keyword, document text, PDF filename,
or source mapping. The experiment output, QA, PDFs, and MinerU cache are not
tracked by Git.
