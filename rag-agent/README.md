# Industrial RAG Agent

## Project Overview

This is a minimal RAG prototype supporting Markdown/PDF document processing,
vector retrieval, and local LLM generation. It is a learning demo, not an
industrial-grade RAG system or production service.

## Frozen Baseline Status

The `v0.5-local-rag-generation` tag freezes the M1–M5 baseline: Markdown and
text-based PDF loading, retrieval and evaluation, local Ollama generation,
RAG context construction, and source/page citations. M6 validation is now
complete as an experimental baseline on eight anonymized local PDFs. This is
not a production-readiness claim; the expanded measurements and limitations
are recorded in the [M6 evaluation report](docs/m6-industrial-pdf-evaluation.md).

## Current Architecture

```text
PDF / Markdown
↓
Document Pipeline
↓
Chunk
↓
Embedding
↓
Retriever
↓
Prompt
↓
Ollama Qwen3-4B
↓
Answer
```

## Implemented Features

- Markdown and PDF document loading
- Markdown heading/paragraph-aware chunking
- Source and page metadata; the RAG CLI carries Markdown heading labels into
  continuation chunks for context and citation display
- Sentence Transformers embeddings
- NumPy cosine-similarity Top-K retrieval
- Markdown and PDF retrieval evaluation datasets and scripts
- Local LLM generation through Ollama with `qwen3:4b`
- Retrieved-source citations in the RAG CLI

## Quick Start

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

If Ollama is not already running, start it in a separate terminal:

```bash
ollama serve
```

Download the model:

```bash
ollama pull qwen3:4b
```

Run the Markdown RAG demo from the repository root. It reads local `.md` files
from `examples/docs/`:

```bash
python src/rag_demo.py
```

Run the test-set retrieval evaluation:

```bash
python src/evaluate_retrieval.py
```

Additional 10-question and PDF retrieval evaluations are available at
`evaluation/evaluate_retrieval.py` and `evaluation/evaluate_pdf_retrieval.py`.
The PDF RAG entry point is `python src/pdf_rag_demo.py <path-to-local-pdf>`.

## Current Limitations

This baseline does not include:

- Vector database
- Reranker
- Permission and access control
- Document version management
- Multi-tenant support
- Monitoring
- Agent workflow
- Memory
- Web API

PDF loading is text-based. Scanned documents, complex tables, and layout-heavy
documents are not handled reliably. Real project and teacher-provided data
must remain out of the repository.

## Long-term Planned Pipeline

```text
Documents
→ Parsing
→ Chunking
→ Embedding
→ Retrieval
→ RAG
→ Source Citation
→ Agent
→ Evaluation
```

## M1 Implementation

### Document Loading

- Reads Markdown files from `examples/docs/`.
- Preserves each document's text, source filename, and a null page value.

```python
{
    "text": "...",
    "metadata": {"source": "xxx.md", "page": None}
}
```

### Chunking

Three approaches were tried:

1. **Blank-line splitting**
   - Produced fewer chunks.
   - Long sections could mix multiple topics.
2. **Fixed-length sliding windows**
   - Increased the chunk count.
   - Improved retrieval for some precise queries.
3. **Markdown structure-aware splitting (current)**
   - Splits at Markdown heading and paragraph boundaries first.
   - Uses a 500-character window with 80-character overlap for longer sections.
   - Chunks retain `source`, `chunk_id`, and document metadata (`source` and
     `page`). The RAG CLI derives a section label from Markdown headings and
     carries it into continuation chunks; the retriever does not persist a
     separate section field.

Current dataset result:

```text
33 documents
→ 436 chunks
```

### Embedding

Uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` to convert document chunks and questions into vectors for semantic similarity retrieval.

### Retrieval

The retrieval path keeps embedding and similarity search as separate steps:

- Embed the query.
- Compare its embedding with chunk embeddings using NumPy cosine similarity.
- Rank scores in descending order and return the top K chunks.

Each result contains `score`, `source`, `chunk_id`, and `text`.

### Retrieval Evaluation

Run the batch evaluation from the repository root:

```bash
python src/evaluate_retrieval.py
```

The script reads `tests/retrieval_test.json` and reuses the M1 loading,
chunking, embedding, and retrieval functions. It reports exact source matches
at Top-1 and Top-3, whether all expected keywords occur in the combined Top-3
text, and the average score across all returned results. Failed cases include
their question, expected source, and retrieved sources. The optional `answer`
field is not evaluated.

## Experiment

```text
Dataset:
华东交通大学学生手册 Markdown 文档

Documents:
33

Chunks:
436
```

Test questions covered:

- Student handbook publisher and publication time
- Credit-based tuition calculation formula
- Transfer-related fee rules
- Credit tuition calculations
- TOEFL score conversion

Batch evaluation snapshot (50 questions):

```text
Top-1 source hit: 31/50
Top-3 source hit: 36/50
Keyword hit: 10/50
Average retrieved score: 0.7039
```

This is a baseline for the current documents, test set, chunking, and model; it
is not a general quality claim. Re-run the script after changing those inputs
to produce a comparable result.

## Observations

The examples below are manual observations. Use the batch snapshot above for
the measured retrieval baseline.

### Successful retrieval examples

- The publisher and publication time were retrieved accurately.
- The credit-based tuition formula was retrieved accurately.
- Transfer-related fee rules were retrieved accurately.

### Issues found

1. **Table content is difficult to retrieve.** Table relationships can be lost when they are represented as plain text. Possible later directions include better table parsing, structured representations, or multimodal processing.
2. **Some results are recalled but ranked too low.** For some questions, the correct chunk appeared in the Top-3, but was not the best Top-1 result. Reranking or hybrid search may be worth evaluating later; neither is part of M1.

## Lessons Learned

- Reliable retrieval is a foundation for RAG.
- Chunking choices directly affect retrieval quality.
- Source and chunk metadata provide a basis for later citations.
- The quality of document representation limits the quality of the knowledge base.
- Separate source and keyword metrics help identify retrieval misses and ranking issues.

## M2 — Basic RAG

Run the interactive demo from the repository root:

```bash
python src/rag_demo.py
```

The demo reuses `retrieval.py` for loading, chunking, embeddings, and Top-K
retrieval. It formats the retrieved chunks as context, combines that context
with the question and a grounding-focused prompt, then calls the local Ollama
`/api/generate` endpoint with `qwen3:4b`. It does not use a cloud API or
implement an Agent. The `来源` section lists retrieved files, section labels,
chunk IDs, and scores; these are context-level sources, not claim-level
citations.

The three-query local check answered the tuition-formula question from its
retrieved clause. The transfer-fee and handbook-publisher questions could not
be answered because their relevant source documents were absent from Top-3;
these remain retrieval coverage limits.

```text
Question
↓
Retrieval
↓
Context
↓
LLM
↓
Answer with source
```

M2 adds local answer generation. Agent capabilities remain future work.

## M3 — Citation + Retrieval Evaluation

`src/rag_demo.py` prints each unique retrieved source, section, and chunk with
its similarity score after the generated answer. These sources show which
retrieved chunks were supplied as context; they are not claim-level citations.

The retrieval-only evaluation in `evaluation/` uses 10 fixed questions from
`evaluation/qa_dataset.json`. Run it from the repository root:

```bash
python evaluation/evaluate_retrieval.py
```

It reuses `src/retrieval.py` and reports exact expected-source matches at
Top-1 and Top-3. It does not evaluate generated answers.

## M4 — Document Intelligence Pipeline

M4.1 PDF loading and M4.2 the unified document pipeline are completed. The
Markdown and PDF loaders produce documents shaped as:

```python
{
    "text": "...",
    "metadata": {"source": "example.pdf", "page": 1}
}
```

PDFs can use the existing retrieval and RAG implementation through the PDF
demo:

```bash
python src/pdf_rag_demo.py examples/pdf/w3c-dummy.pdf
```

This is a baseline text extraction path; it does not add OCR or table parsing.

### M4 PDF Retrieval Evaluation Baseline

Run the retrieval-only PDF baseline from the repository root:

```bash
python evaluation/evaluate_pdf_retrieval.py
```

It evaluates Top-1 and Top-3 source hits and expected-keyword hits using the
fixed questions in `evaluation/pdf_qa_dataset.json`. It does not call the LLM.

Supported: text-based PDFs and page metadata in source output. Current
limitations: complex tables, scanned documents, and layout-heavy documents.

## M5 — Local RAG Baseline Freeze

The `v0.5-local-rag-generation` checkpoint freezes the M1–M5 local RAG
prototype: Markdown/text-based PDF loading, Top-K retrieval, retrieved-context
prompting, and answer generation through Ollama's local `/api/generate` API
with `qwen3:4b`. The CLI displays the retrieved source information, including
PDF page metadata when available. This remains a learning baseline. Benchmark
snapshots and known limitations are recorded in
[`docs/m1-m5-summary.md`](docs/m1-m5-summary.md).

## M6 — MinerU Industrial PDF RAG

The PDF RAG CLI keeps PyMuPDF as its default parser and adds an optional local
MinerU 4.0.5 Advanced/OCR path. MinerU remains in its separate
`mineru-405-poc` environment; the adapter calls `run-mineru.ps1` through
PowerShell and reads the actual Middle JSON output. The shared document,
chunking, embedding, retrieval, and generation paths are reused.

```bash
python src/pdf_rag_demo.py path/to/document.pdf --parser pymupdf
python src/pdf_rag_demo.py path/to/document.pdf --parser mineru
```

Set `MINERU_RUNNER_PATH` or pass `--mineru-runner` if the runner is outside the
default sibling workspace location. MinerU results are cached under
`.local/mineru/`, keyed by the PDF content and parser settings. Use
`--force-parse` to replace a cached parse.

Run the same PDF QA set through either parser:

```bash
python evaluation/evaluate_pdf_retrieval.py --pdf path/to/document.pdf --dataset evaluation/industrial_pdf_qa.local.json --parser pymupdf
python evaluation/evaluate_pdf_retrieval.py --pdf path/to/document.pdf --dataset evaluation/industrial_pdf_qa.local.json --parser mineru
```

The ignored local QA file starts empty. Cases may include `question`,
`expected_source`, 1-based `expected_page`, `expected_keywords`, optional
`answer`, `category`, and `review_required`. Ground truth must be checked
against the original PDF; a page label requires `expected_source` to avoid
cross-document page-number matches. `review_required: true` cases are excluded
from evaluation. The evaluator reports parser/document/chunk stats, Top-1/Top-3
page and source hits, evidence/keyword hits, and failure categories. It does
not evaluate generated answers.

The adapter and parser selection are smoke-tested with a synthetic scanned
PDF. The initial pilot parsed 7 of 8 anonymized PDFs (357 of 662 pages),
compared PyMuPDF and MinerU on 10 manually checked questions, scored two
additional table cases separately, and ran 10 local Qwen3 RAG questions.
MinerU Top-3 page hit was 6/10 and normalized evidence hit was 5/10, compared
with 3/10 and 2/10 for PyMuPDF. These are historical pilot measurements. The
expanded validation below subsequently parsed all eight PDFs, including the
305-page scan, and reports a larger ground-truth set and known limitations.

A separate representation audit traced selected evidence through the original
pages, MinerU Markdown, Middle JSON, unified Documents, chunks, and retrieval.
It found one table case whose complete evidence survives but ranks 7 (outside
the current Top-3), and another whose expected evidence is already incomplete
in MinerU output. In the legacy flat representation, inline image data
accounted for 94.2% of one schematic-heavy source's text volume. These
observations point to parsing and representation boundaries for targeted
follow-up; they do not justify broad retrieval changes. See the anonymized
[M6 document representation audit](docs/m6-document-representation-audit.md).

The MinerU adapter now defaults to a block-aware structured representation;
`--representation flat` remains available for controlled comparison. On the
same cached MinerU outputs and fixed 10-question pilot plus two table cases,
structured representation reduced indexed text from 1,990,067 to 452,103
characters and chunks from 4,811 to 1,512; a separate scan found no image data
URI or `base64,` markers in structured documents or chunks. Top-3 page hits
improved from 6/12 to 8/12 and normalized Top-3
evidence hits from 5/12 to 6/12; Top-1 page hits stayed 5/12. The complete
evidence for one table case remains outside Top-3, and one OCR evidence rank
moved from 3 to 4. An 8-case local RAG smoke returned non-empty answers and
page metadata, but only 6/8 retrieved the expected page in Top-3 and 4/8
retrieved all expected keywords; generated-answer correctness was not
human-scored. This is a small local experiment, not a completeness or
industrial-readiness claim. These are results of the earlier fixed 12-case
experiment; the final expanded validation is recorded in the
[M6 evaluation report](docs/m6-industrial-pdf-evaluation.md).

### M6 Final Validation

- Scope: 8 anonymized PDFs, 662 pages; all raw page maps are continuous.
- Dataset: 32 answerable original-PDF-grounded QA and 3 unanswerable generation
  checks; no cases required ground-truth review.
- Structured input: 1,594 Documents, 659 non-empty represented pages, 2,996
  chunks, and 855,152 indexed characters. The three omitted pages were blank in
  the original PDFs.
- Retrieval: Top-1/Top-3 source hit 25/32 and 30/32; Top-1/Top-3 page hit
  9/32 and 15/32; normalized evidence hit 10/32.
- Failure attribution: PARSING 11, REPRESENTATION 0, CHUNKING 0, RETRIEVAL 4,
  RANKING 10, INSUFFICIENT_DATA 0, GT_UNCERTAIN 0.
- RAG: 18 local Qwen3 answers generated with source/page/block citation fields.
  The question-level review file remains local and requires human review; no
  answer-accuracy score is claimed.

The 305-page scan completed MinerU Advanced/OCR. Structured Documents and
chunks contained zero data URI characters. M6 is frozen as a measured
experimental baseline, not as a production-readiness claim. See the
[anonymized M6 evaluation report](docs/m6-industrial-pdf-evaluation.md).

## Companion Component: Industrial Preprocessor

The reusable preprocessing source is included under
[`components/industrial-preprocessor/`](components/industrial-preprocessor/)
as a standalone Python subproject. It keeps its own `src` package and should
be run from that directory in a separate Python 3.12 environment, avoiding a
module-name collision with the RAG root `src` package. See the component
README for setup.

This is a repository-level co-location only: the RAG MinerU adapter does not
import the standalone component. In the combined local workspace, the RAG
project is under `rag-agent/` and the MinerU runtime is under the sibling
`minerU/mineru-405-poc/`; the adapter detects that runner location. A standalone
clone can set `MINERU_RUNNER_PATH` or pass `--mineru-runner`.

The MinerU virtual environment, local models, PDFs, parser outputs, evaluation
datasets, and data-dependent experiment reports are local-only and must not be
committed. A fresh clone needs its own MinerU runtime and runner path.

## Roadmap

- [x] Project initialization
- [x] M1 — Minimal Retrieval Pipeline
- [x] M2 — Basic RAG
- [x] PDF ingestion
- [x] M3 — Source citation
- [x] Retrieval evaluation
- [x] M5 — Baseline freeze
- [x] M6 — MinerU parser integration infrastructure
- [x] M6 — Local pilot A/B and RAG smoke run
- [x] M6 — Structured MinerU representation experiment
- [x] M6 — Broader original-PDF-grounded QA and large-scan validation
- [ ] Agent
- [ ] Industrial document improvements

## Key Decisions

- Start minimal and add components only when needed.
- Do not introduce large RAG frameworks during initialization.
- Real industrial/project data will not be committed to this repository.
- The first technical milestone used simple Markdown documents to understand retrieval.

## Data & Security

- Real industrial documents will not be uploaded.
- Teacher-provided data will not be uploaded.
- API keys will not be uploaded.
- MinerU parsing results from real documents will not be uploaded.
- `.local/mineru/`, `.mineru-home/`, and `evaluation/industrial_pdf_qa.local.json` are local-only and ignored by Git.
- `.env` will not be uploaded.
- Future test data in this repository will be self-created or explicitly permitted for public use.
