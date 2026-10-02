# M8.1 — Document Intelligence Failure Audit

## Audit Summary

The audit reviewed all 11 M7.2 `PARSING_FAILURE` cases. The local M7.2
analysis, M6 QA, eight original PDFs, and matching MinerU structured caches
were available; all 8 PDF-to-cache pairs matched the recorded content hashes.
The question-to-source mapping and all original page content were used locally
and are not reproduced here.

Each case was compared across the original PDF page, raw MinerU Middle JSON,
the current structured Documents, and the M7.2 expected-evidence matcher.
Classification requires evidence for the failure mechanism; ambiguous text
extraction cases remain `OTHER/UNCERTAIN`.

| Category | Count |
| --- | ---: |
| TABLE_STRUCTURE_FAILURE | 0 |
| IMAGE_INFORMATION_LOSS | 2 |
| OCR_FAILURE | 5 |
| LAYOUT_RELATION_FAILURE | 1 |
| REPRESENTATION_LIMITATION | 0 |
| OTHER/UNCERTAIN | 3 |
| **Total** | **11** |

The largest confirmed group is OCR failure (5/11). Three of those cases are
table-cell text misses. The checked Middle JSON table bodies retained row and
cell boundaries, so those cases do not establish a table row/column structure
failure. The cases also did not show complete expected evidence in raw Middle
JSON that was then lost only by the structured adapter; therefore no case is
assigned `REPRESENTATION_LIMITATION`.

## Failure Cases

Only anonymous case IDs are used. No PDF filename, question text, answer,
document text, industrial identifier, or page number is included.

### Q09

- **Category:** `OCR_FAILURE`
- **Observed issue:** A value visible in an original schedule is absent from
  both raw Middle JSON and the structured table text. The table retains its
  row/cell boundaries.
- **Evidence source:** PDF page inspection; MinerU Middle JSON and structured
  output.
- **Possible direction:** Measure numeric cell recognition on this table
  pattern before considering any table-structure change.

### Q10

- **Category:** `OCR_FAILURE`
- **Observed issue:** A visible schedule value is not recognized in the raw
  MinerU table body; the table's row/cell structure is present.
- **Evidence source:** PDF page inspection; MinerU Middle JSON and structured
  output.
- **Possible direction:** Include this case in a small numeric table-cell
  recognition comparison.

### Q12

- **Category:** `OCR_FAILURE`
- **Observed issue:** The original table row is visible and its quantity is
  represented, but part of the corresponding item text is absent from Middle
  JSON and the structured table document.
- **Evidence source:** PDF page inspection; MinerU Middle JSON and structured
  output.
- **Possible direction:** Check table OCR at the cell level while keeping the
  row/cell association fixed.

### Q14

- **Category:** `OTHER/UNCERTAIN`
- **Observed issue:** The expected text is selectable in the source PDF but is
  absent from raw Middle JSON and structured Documents. The page does not
  establish a table, image, or spatial-relation cause.
- **Evidence source:** PDF text extraction and page inspection; MinerU Middle
  JSON and structured output.
- **Possible direction:** Trace text-block coverage for this document type
  before assigning it to OCR or layout handling.

### Q16

- **Category:** `IMAGE_INFORMATION_LOSS`
- **Observed issue:** Required annotation text is visible in an engineering
  drawing but is not represented in the corresponding MinerU text or image
  body.
- **Evidence source:** PDF page inspection; MinerU Middle JSON and structured
  output.
- **Possible direction:** Test whether existing drawing annotations can be
  retained as text with source geometry; no vision model was tested.

### Q17

- **Category:** `OCR_FAILURE`
- **Observed issue:** One drawing annotation is represented, while a second
  visible dimensional annotation is absent from the image-body text.
- **Evidence source:** PDF page inspection; MinerU image-block output.
- **Possible direction:** Include dimension symbols and numeric callouts in a
  focused drawing-text recognition test.

### Q20

- **Category:** `OCR_FAILURE`
- **Observed issue:** The source page has no native text layer. MinerU emits
  text blocks, but the visibly present target clause is missing from them.
- **Evidence source:** PDF page inspection; MinerU OCR and structured output.
- **Possible direction:** Evaluate recognition of dense scanned prose on a
  small fixed sample before changing parser settings.

### Q29

- **Category:** `IMAGE_INFORMATION_LOSS`
- **Observed issue:** The original plan contains repeated labels, but its image
  body is empty in the structured output; the separate schedule text does not
  represent those plan labels.
- **Evidence source:** PDF page inspection and native text extraction; MinerU
  image/table blocks.
- **Possible direction:** Test preservation of plan labels together with
  their drawing location; the current evidence does not establish table-cell
  relationship loss.

### Q30

- **Category:** `OTHER/UNCERTAIN`
- **Observed issue:** The page contains readable prose and MinerU text blocks,
  but expected evidence is absent from both extracted text layers. Available
  evidence does not distinguish OCR error from text-block omission.
- **Evidence source:** PDF page inspection; PyMuPDF text extraction; MinerU
  Middle JSON and structured output.
- **Possible direction:** Recheck source-text coverage for this case before
  selecting an OCR or layout intervention.

### Q31

- **Category:** `OTHER/UNCERTAIN`
- **Observed issue:** Expected evidence is not found in the source text layer or
  MinerU text blocks. The checked page does not show a confirmed image, table,
  or spatial-relation cause for the miss.
- **Evidence source:** PDF page inspection and text extraction; MinerU Middle
  JSON and structured output.
- **Possible direction:** Verify the evidence-to-page mapping and text-block
  coverage before implementation.

### Q32

- **Category:** `LAYOUT_RELATION_FAILURE`
- **Observed issue:** The page's diagram labels, line paths, and allocation
  tables are emitted as separate blocks. The structured Documents do not
  encode which labels and table entries belong to each line path.
- **Evidence source:** PDF diagram inspection; MinerU image, text, and table
  blocks.
- **Possible direction:** Evaluate preserving block geometry and diagram-to-
  label associations on a small sample.

## Priority Analysis

| Priority | Failure group | Count | Cost and expected benefit |
| --- | --- | ---: | --- |
| High | OCR_FAILURE | 5 | The most frequent confirmed type; several target values are visible in the originals. A targeted recognition audit is lower cost than adding a new model and could cover both table cells and scanned prose. |
| Medium | IMAGE_INFORMATION_LOSS | 2 | Potentially useful for drawing-heavy questions, but may require preserving or extracting annotations not present in Middle JSON. The sample is small. |
| Medium-low | LAYOUT_RELATION_FAILURE | 1 | Could affect diagram questions, but a robust relation representation may cost more and only one audited case confirms it. |
| Diagnose first | OTHER/UNCERTAIN | 3 | Cause is unresolved; implementation priority should wait for source-text and evidence-mapping checks. |
| Not currently supported | TABLE_STRUCTURE_FAILURE / REPRESENTATION_LIMITATION | 0 | No audited case confirms broken table row/cell mapping or complete evidence lost only in structured projection. |

The most supported next experiment is a small OCR-focused comparison, especially
for table cell text and dense scanned prose. This does not establish that a
specific parser setting or table processor will improve retrieval. A structural
table rewrite is not supported by this audit; a targeted table OCR test is.

## Vision and Table Decisions

- **Vision model:** Not justified yet. Two cases need drawing/image information,
  and one needs diagram relationships. First test whether existing Middle JSON
  fields, readable image-body text, and source geometry can be retained and
  evaluated without adding a model. No visual model was added or run.
- **Table-specific processing:** A new row/column representation is not
  supported by these cases. Three OCR failures involve table content, so a
  bounded cell-recognition experiment is worth considering after this audit.
- **Raw block comparison:** A local, static comparison of raw Middle JSON
  content and structured output did not recover complete expected evidence for
  any of the 11 cases after table markup was interpreted as markup. This was not
  a retrieval A/B experiment and produced no new result file.

## Scope and Privacy

This audit did not add or change a model, MinerU pipeline, retrieval,
reranker, embedding, chunking, prompt, or agent. PDFs, QA data, MinerU caches,
and rendered inspection pages remain local. The report contains only anonymous
case IDs, failure categories, generic issue descriptions, and diagnostic
directions.
