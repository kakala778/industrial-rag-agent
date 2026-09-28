# M6 Document Representation Audit

This is a small diagnostic sample, not a general parser-quality benchmark. It
records where selected information was preserved or first became incomplete
along this path:

```text
Original PDF page
→ MinerU Markdown
→ Middle JSON
→ unified Document
→ chunks
→ retrieval rank
```

## Scope and method

- Reviewed nine representative pages across five anonymized source IDs. The
  sample includes drawing and table pages, two OCR pages, and a selectable-text
  contents page.
- Grounded the two supplemental table cases and two OCR cases against the
  original page images.
- Inspected page-level MinerU Markdown and cached Middle JSON, then passed the
  cached JSON through the existing PDF loader and chunking path. Retrieval
  ranks used the existing embedding model and retrieval implementation.
- Generated page-targeted MinerU Markdown for the selected pages; source PDF
  hashes were unchanged. No parser, loader, chunking, embedding, or retrieval
  code was modified.

## Findings by layer

| Case | First confirmed loss or limitation | Observed result |
| --- | --- | --- |
| Table case A (`doc_02`, page 16) | None through chunking | Row/cell evidence is present in Middle JSON, Markdown, the loaded Document, and one chunk. That chunk ranks 7 (cosine 0.5857), so it is outside Top-3 but inside Top-10. This is a Top-3 ranking symptom, not missing source content. |
| Table case B (`doc_04`, page 30) | MinerU parsing | One expected table item is absent from Middle JSON and Markdown; the other survives only partially. No chunk contains the complete expected evidence. The page first ranks 54 (0.5961); a partial-evidence chunk ranks 167 (0.5043). |
| OCR cases (`doc_03`, pages 7 and 11) | None identified in these samples | The checked evidence survives parsing and chunking. Complete evidence chunks rank 3, while the expected pages appear at rank 1. This confirms only these sampled pages. |
| Selectable-text sample (`doc_07`, page 3) | MinerU parsing | The original contents page contains more headings than appear in the parsed output. This page was not used to claim a retrieval miss. |

The table results separate two failure modes: one case has complete evidence
but ranks below the CLI's Top-3; the other is already incomplete before loading
and chunking. The latter should be investigated at the parse/output boundary,
not attributed to retrieval ranking.

## Representation and text-volume observations

- The unified loader preserves `source` and `page`, but recursively joins
  content strings into one text field. It does not keep each block's type,
  position, and table-cell relationships as independent structured items.
- For two sampled schematic pages in `doc_08`, inline image data makes up
  approximately 94.2% of the loaded text volume. The current loader copies
  these payloads into `Document.text`, after which chunking treats them as
  ordinary text.
- In the sampled `doc_04` material, chart and table serialization account for
  about 80.1% of MinerU's extracted character volume. For `doc_02`, the volume
  combines useful OCR text with drawing and table serialization, so character
  growth alone cannot distinguish useful extraction from noise.
- Across the current parsed corpus, 388 of 5,130 chunks (7.6%) are exact
  duplicates after whitespace normalization, representing 132 repeated text
  values. Some repeated schedules may be legitimate; this audit does not label
  every duplicate as a parser defect.

These counts describe representation volume, not semantic accuracy. In
particular, inline image payloads and custom chart/table markup should not be
interpreted as equivalent to readable prose.

## Sample-level first-failure summary

| Classification | Pages | Interpretation |
| --- | ---: | --- |
| MinerU parsing | 2 | Evidence is absent or incomplete in parser output for the selected sample. |
| Loader representation | 3 | Text is carried forward, but typed chart/table/image structure is flattened; this is a fidelity limitation, not three proven answer misses. |
| Serialization, chunking, or beyond-Top-10 retrieval | 0 | No first failure at these layers was confirmed in the nine-page trace. |
| No failure identified in the checked path | 4 | The sampled content was preserved sufficiently for the checks performed. |

The counts are diagnostic labels for these nine pages only. The rank-7 table
case remains outside Top-3 even though the evidence survives, so it is worth
tracking separately from the first-failure counts.

## Follow-up

1. Keep tables, charts, and images as typed blocks rather than inserting image
   data into embedding text; preserve table row and cell relationships.
2. Investigate the two sampled MinerU parse omissions against the original
   pages and their page-level Markdown/Middle JSON outputs.
3. Re-run the same fixed cases after a representation or parsing change.
   Consider retrieval changes only for complete evidence that remains poorly
   ranked; this sample does not justify a broad ranking change.

No reranker, hybrid search, vector database, new embedding model, or M7
functionality is proposed by this audit. Raw source filenames, question text,
document contents, and parser outputs remain in the local-only report.
