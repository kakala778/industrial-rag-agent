# M8.4 — Vision Block Experiment

## 1. Experiment Purpose

M8.3 sent full-page renders to `qwen3-vl:2b-instruct-q4_K_M`. It did not
recover normalized M6 evidence for Q10, Q16, or Q32. M8.4 tests whether a
focused MinerU block crop, with and without existing MinerU text context, can
recover information that is difficult to read at full-page scale.

This is an offline comparison. No vision output was added to the default
representation, embeddings, retrieval, chunking, reranking, or RAG prompt.

## 2. Experiment Design

The fixed M6 input manifest and matching MinerU caches were verified before the
run. The caches contained Middle JSON but no MinerU ZIPs or traceable image
assets. Crops were rendered read-only from the original PDF pages using the
cached Middle JSON block bounding boxes; MinerU was not rerun. Existing
full-page renders from M8.3 were reused for Q10, Q16, and Q32.

All M8.4 calls used the specified `qwen3-vl:2b-instruct-q4_K_M` model through
local Ollama 0.34.4, with temperature 0 and a 2,048-token output limit. The
specified M8.4 prompt was sent as the system prompt. Mode B also supplied the
selected visual block's text and the explicitly selected nearby table text as
user context, without asserting that the blocks were semantically related.
The same prompt was used for all three M8.4 input modes.

M8.3 used its earlier prompt, so the M8.3-to-M8.4 full-page comparison changes
both prompt and experiment version. The A/B/C comparisons within M8.4 keep the
prompt, model, generation settings, and target page fixed.

| Case | Type | Page | MinerU block / mode B text context | MinerU evidence | M7.1 Top-3 page / evidence |
| --- | --- | ---: | --- | --- | --- |
| Q10 | `OCR_FAILURE` | 24 | table index 6 / table index 6 | Absent | Miss / miss |
| Q16 | `IMAGE_INFORMATION_LOSS` | 108 | No image/chart/table block / none | Absent | Miss / miss |
| Q29 | `IMAGE_INFORMATION_LOSS` | 8 | image index 0 / table index 3 | Absent | Miss / miss |
| Q32 | `LAYOUT_RELATION_FAILURE` | 29 | image index 2 / table index 4 | Absent | Hit / miss |
| Q5 | Known-success sanity check | 7 | Text-only page / none | Present | Hit / hit |

Q10, Q16, and Q32 are the same failure cases used in M8.3. Q29 adds a second
image-information case with an available image block. Q5 is a known M6 success:
the normalized evidence is present in MinerU's structured page and was found in
M7.1 Top-3.

The evaluation reuses the existing normalized evidence matcher. A field count
is the number of expected evidence keys matched by the vision response; a full
evidence hit requires all expected keys. The anonymous records retain case ID,
category, page, block type/index, normalized block bounds, hit flags, and
quality counts. Raw model responses and rendered images stay in the Git-ignored
local output directory.

## 3. M8.3 vs. M8.4

| Case | M8.3 full page | M8.4 full page | M8.4 block crop | M8.4 crop + MinerU text |
| --- | --- | --- | --- | --- |
| Q10 | No evidence (0/1) | No evidence (0/1) | Evidence recovered (1/1) | Evidence recovered (1/1) |
| Q16 | No evidence (0/2) | No evidence (0/2) | Not available: no visual block | Not available |
| Q32 | No evidence (0/1) | No evidence (0/1) | No evidence (0/1) | No evidence (0/1) |
| Q29 | Not tested | No evidence (0/3) | No evidence (0/3) | No evidence (0/3) |
| Q5 | Not tested | Evidence present (2/2) | Not applicable | Not applicable |

M8.3 and M8.4 full-page responses both missed the three shared target cases.
The Q10 block crop recovered its target where both full-page runs missed. The
Q16 page still failed, Q29 did not improve with a crop, and Q32 did not recover
the missing relation evidence.

## 4. Results

| Case | Type | Input | MinerU target / current retrieval | Vision fields | Sections present / 6 | Evidence recovered | Document candidate |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| Q10 | OCR | Table block crop | Absent / page miss, evidence miss | 1/1 | 5 | Yes | Yes |
| Q10 | OCR | Table crop + MinerU text | Absent / page miss, evidence miss | 1/1 | 2 | Yes | No |
| Q10 | OCR | Full page | Absent / page miss, evidence miss | 0/1 | 1 | No | No |
| Q16 | Image | Full page | Absent / page miss, evidence miss | 0/2 | 4 | No | No |
| Q29 | Image | Image block crop | Absent / page miss, evidence miss | 0/3 | 1 | No | No |
| Q29 | Image | Crop + MinerU text | Absent / page miss, evidence miss | 0/3 | 1 | No | No |
| Q29 | Image | Full page | Absent / page miss, evidence miss | 0/3 | 1 | No | No |
| Q32 | Layout | Image block crop | Absent / page hit, evidence miss | 0/1 | 1 | No | No |
| Q32 | Layout | Crop + MinerU text | Absent / page hit, evidence miss | 0/1 | 6 | No | No |
| Q32 | Layout | Full page | Absent / page hit, evidence miss | 0/1 | 2 | No | No |
| Q5 | Sanity | Full page | Present / page hit, evidence hit | 2/2 | 2 | Yes | Yes |

Counts across the four selected parsing-failure cases:

- Unique cases with normalized evidence recovered: **1/4** (Q10).
- The Q10 evidence hit occurred with both crop modes, but only the crop-only
  response met the experiment's structured Document candidate rule. Adding
  MinerU text context did not add a field and reduced the number of required
  headings returned.
- Q32's crop-plus-text response contained all six headings, but did not match
  the expected evidence. A well-formed response alone is not a recovery.
- The Q5 sanity response matched both expected fields, confirming that the
  model could extract the known-success page's text under the M8.4 prompt.

The candidate flag requires both a complete normalized evidence hit and
meaningful content in a category-relevant section. It is an offline screening
signal, not approval to index the response.

## 5. Output Quality Analysis

- **Block crop versus full page:** Q10 improved only with a focused table crop.
  The crop exposes the table at higher effective detail than the wide page
  image. The crop-only response also followed more of the requested structure
  than the context-augmented response.
- **Image information:** Q16 had no MinerU image/chart/table block to crop.
  Its full-page re-test remained a miss. Q29 had an image block, but neither
  crop nor page input recovered any of its three expected fields.
- **Layout relation:** Q32 remained an evidence miss in every mode. MinerU text
  context made the response use all requested headings, but did not restore
  the target label-to-path relation.
- **Sanity check:** Q5 recovered 2/2 fields. Thus the Q16/Q29/Q32 misses are
  not explained by a universal inability to read page text; they remain
  case-specific failures.

## 6. Architecture Recommendation

- **New Document:** Q10's crop-only output is a candidate for a separate,
  provenance-bearing Document block in a further offline retrieval A/B test.
  The evidence match is encouraging but comes from one OCR/table case; it is
  not enough to add vision output to the default index.
- **Metadata enrichment:** Q32's complete heading structure did not recover
  evidence or relations. This run does not establish reliable metadata
  enrichment for layout-heavy pages.
- **General Visual RAG:** Do not integrate the model into the default RAG path
  yet. Only one of four selected parsing failures recovered normalized
  evidence, and two image-information cases plus the layout case remain
  unresolved.
- **Next experiments:** Repeat block-crop OCR on independent table-cell and
  annotation failures. M8.1 did not find table row/column structure failures,
  so a table-structure model is not currently supported. A larger VLM is also
  not justified by this sample: the small model succeeded on one focused crop,
  while the unresolved cases involve different image and relation failures.
  If image labels or spatial relations remain a priority after additional
  crop trials, benchmark a specialized OCR or layout model separately before
  selecting an architecture.

## 7. Reproduction and Privacy

Run from `rag-agent/` with the same local M6 inputs, M7/M8.2 outputs, and
matching MinerU cache:

```powershell
python evaluation/run_m8_vision_block_experiment.py
```

The runner writes anonymous metrics, rendered inputs, and local model responses
under the Git-ignored `outputs/m8_vision_block_test/`. The report contains no
question text, expected answer, expected field text, document text, PDF name,
or private source path. No PDF, QA file, image, or MinerU cache was added to
Git.
