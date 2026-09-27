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

## M6 Status and Follow-up

The local MinerU adapter, page mapping, cache reuse, 10-question pilot, and
RAG smoke run are verified. M6 industrial-document validation remains partial.
Next work should expand only the original-PDF-grounded QA set and investigate
the observed table/layout extraction and text-expansion failures. Retrieval,
chunking, embedding, and generation algorithms were not changed as part of
this evaluation.
