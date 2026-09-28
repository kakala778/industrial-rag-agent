# M6 Local MinerU Evaluation Snapshot

This report summarizes a local, exploratory evaluation of the MinerU 4.0.5
PDF path. It is a learning baseline, not an industrial-quality claim. The
PDFs, full filenames, question text, expected answers, MinerU outputs, and
generation logs remain outside this repository. `doc_01` through `doc_08` are
anonymous IDs for this local run; their mapping is not committed.

## Scope

- Compared the existing PyMuPDF path with local MinerU 4.0.5 Advanced/OCR.
- Reused the existing chunking, embedding model, cosine retrieval, and Top-3
  behavior without changes.
- Grounded 10 pilot QA cases and 2 supplemental table cases against original
  PDF page images. No case required review before scoring.
- Ran 10 local RAG questions through retrieval, context construction, and
  Ollama `qwen3:4b` generation. The saved answers are local-only.
- Parsed 7 of 8 source PDFs (357 of 662 pages). `doc_05` was skipped: it has
  305 pages and no text in the PyMuPDF baseline, so a full OCR run was
  deferred due to runtime cost.

## Page Parsing Statistics

Character growth is a volume signal only. It does not measure accuracy; large
increases can indicate useful OCR, repeated content, or noisy extraction.
Times depend on the local machine.

| ID | Pages | PyMuPDF chars | MinerU chars | MinerU chunks | Char ratio | MinerU parse (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| doc_01 | 108 | 54,007 | 117,791 | 319 | 2.18x | 442.38 |
| doc_02 | 20 | 1,786 | 52,306 | 129 | 29.29x | 166.44 |
| doc_03 | 27 | 0 | 18,068 | 57 | n/a | 148.03 |
| doc_04 | 71 | 90,021 | 515,930 | 1,245 | 5.73x | 907.60 |
| doc_06 | 30 | 27,080 | 150,549 | 371 | 5.56x | 342.90 |
| doc_07 | 72 | 62,826 | 68,458 | 178 | 1.09x | 343.33 |
| doc_08 | 29 | 34,868 | 1,184,756 | 2,831 | 33.98x | 252.84 |

All seven parsed outputs mapped continuously to 1-based page numbers, and a
second load reproduced the cached output for all seven files. A separate
before/after SHA-256 check found no change to the four additional PDFs parsed
after the pilot.

## Retrieval A/B

The 10-question pilot uses source, page, and expected evidence labels. Raw
keyword matching is shown separately from NFKC/whitespace-normalized matching;
normalized matches are checked per result so text from different chunks cannot
be joined into a false token.

| Parser | Top-1 page | Top-3 page | Top-1 source | Top-3 source | Raw evidence | Normalized evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PyMuPDF | 1/10 | 3/10 | 6/10 | 6/10 | 1/10 | 2/10 |
| MinerU | 5/10 | 6/10 | 9/10 | 10/10 | 3/10 | 5/10 |

Source hits mean the expected document appeared in the result; they do not
prove the expected page or evidence was retrieved. Error-layer counts for the
pilot were:

| Parser | Parsing | Chunking | Retrieval | Ranking | GT uncertain |
| --- | ---: | ---: | ---: | ---: | ---: |
| PyMuPDF | 6 | 0 | 2 | 1 | 0 |
| MinerU | 2 | 0 | 3 | 1 | 0 |

The 2 supplemental table cases were scored separately and are not included in
the pilot totals. Both parsers returned the expected source for both cases.
PyMuPDF placed the expected page in Top-3 for 1/2 cases; MinerU did so for
0/2. Neither parser retrieved all expected table keywords in Top-3 (0/2 each).
The diagnosed misses were parsing (2/2) for PyMuPDF and parsing (1/2) plus
ranking (1/2) for MinerU.

## Local RAG Smoke Run

All 10 questions returned a non-empty Qwen3 answer, and the CLI printed the
expected source filename in all 10 runs. The expected page appeared among the
retrieved Top-3 chunks in 7/10 runs. A strict normalized keyword check found
all expected keywords in 4/10 generated answers. This check is not a human
answer-quality score: paraphrases may fail it, and a keyword match alone does
not establish correctness. The full question-by-question outputs remain
private.

## Observations and Limits

- For `doc_03`, PyMuPDF returned no text on 27/27 pages. MinerU produced text
  on all 27 pages; the sampled OCR content was checked against original page
  images. This is evidence for the sampled pages, not a general OCR guarantee.
- `doc_07` had similar extraction volume in both paths (1.09x). MinerU improved
  the pilot page and source metrics overall, but page/evidence misses remain.
- `doc_02`, `doc_04`, and `doc_08` had large text-volume increases (5.73x to
  33.98x). The sampled originals contain dense tables or drawings. These
  increases are a noise/duplication review signal, not evidence of better
  retrieval.
- The supplemental table cases show that correct source retrieval can coexist
  with wrong-page ranking and missing table evidence.
- The 305-page scan, broader per-document QA, table reconstruction, scanned
  page completeness, and human review of generated answer correctness remain
  unverified.

## Structured Representation Experiment

The legacy `flat` representation creates one page document by recursively
collecting content. The `structured` representation emits ordered text groups
and separate table, chart, and image documents with source/page/block metadata.
Readable image-body text and captions are retained after markup and explicit
image/base64 payload cleanup; payload-only image blocks are skipped. Table rows
and cells are rendered as readable text. The parser output, fixed QA wording,
embedding model, chunking, cosine similarity, Top-K, and generation prompt
were unchanged. Both runs used the same cached Middle JSON; MinerU was not
rerun and the input PDFs' SHA-256 values stayed unchanged.
An independent scan found no `data:image/` or `base64,` marker in any
structured document or chunk.

| Metric | MinerU flat | MinerU structured |
| --- | ---: | ---: |
| Documents | 249 | 778 |
| Indexed text characters | 1,990,067 | 452,103 |
| Inline image data URI characters in documents | 1,169,359 | 0 |
| Data URI characters in chunks | 7,329 | 0 |
| Chunks | 4,811 | 1,512 |
| Exact duplicate chunk instances | 388 | 342 |
| Repeated normalized chunk values | 132 | 85 |
| Top-1 expected-page hits, all 12 cases | 5/12 | 5/12 |
| Top-3 expected-page hits, all 12 cases | 6/12 | 8/12 |
| Normalized evidence hits at Top-3, all 12 cases | 5/12 | 6/12 |

For the 10-question pilot in its original source scope, Top-1 page hits were
5/10 flat and 6/10 structured; Top-3 page hits were 6/10 and 8/10; normalized
evidence hits at Top-3 were 5/10 and 6/10. The two supplemental table cases
had 0/2 Top-1 page hits in both runs, 0/2 versus 1/2 Top-3 page hits, and 0/2
Top-3 evidence hits in both runs.

In `doc_08`, indexed characters fell from 1,184,756 to 40,927 (1,143,829
fewer, about 96.5%); chunks fell from 2,831 to 136 (2,695 fewer). Its
1,115,709 document-level inline data URI characters were reduced to zero.
There is no fixed QA question targeting this source, so its direct retrieval
accuracy cannot be compared without adding ground truth.

Table QA A's expected page rank improved from 7 to 2, while the complete
expected row evidence moved from rank 7 to rank 6 and remained outside Top-3.
For table QA B, the expected evidence was still incomplete in raw Middle JSON
for both representations; the structured page ranked 5 but complete evidence
was not retrieved. This remains a MinerU parsing limitation, not evidence of
successful table recovery.

The two sampled OCR pages retained their expected page at rank 1 in both runs.
On the first page, complete evidence improved from rank 3 to 2 and stayed in
Top-3. On the second, evidence moved from rank 3 to 4, just outside Top-3,
although the evidence remains in the ranked results. A separate pilot OCR
case whose evidence was previously present only in an image body's readable
text is now preserved by the structured adapter. Of the four pages that
initially emitted no structured document, three contained readable image-body
text and are now represented; the remaining page was visually blank. No other
loss was identified in the checked cases, but this small sample does not
establish document-wide completeness.

Eight representative fixed cases were then run through structured loading,
chunking, retrieval, the existing prompt, and local Ollama `qwen3:4b`. All 8
returned non-empty answers; retrieved context was free of data URI/base64
markers in 8/8 cases, and formatted sources carried source/page in all 8. The
expected source/page was present in Top-3 for 6/8 cases, and all expected
evidence keywords were present in Top-3 for 4/8. A strict expected-keyword
match occurred in 3/8 generated answers. These counts are diagnostic only;
there was no human answer-quality scoring, and a non-empty answer is not proof
of grounding.

Overall, the structured representation removed all measured image data URI
payload, reduced indexed text volume and chunk count, and improved Top-3 page
hits in this fixed sample. Top-1 page hits were unchanged, the complete table
row remains outside Top-3, and one OCR evidence rank moved from 3 to 4. This
supports structured representation as the default input form, but does not
justify reranker, hybrid search, or embedding changes. Broader original-PDF
QA and the deferred large scan remain outside this experiment.

## M6 Status and Follow-up

The local MinerU adapter, page mapping, cache reuse, 10-question pilot, paired
flat/structured representation evaluation, and 8-case structured RAG smoke
are verified for this local sample. M6 industrial-document validation remains
partial. Next work should expand only the original-PDF-grounded QA set and
investigate the observed table/layout extraction and text-expansion failures.
Retrieval, chunking, embedding, and generation algorithms were not changed as
part of this evaluation.
