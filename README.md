# Industrial RAG Agent

## Project Overview

This is a minimal RAG prototype supporting Markdown/PDF document processing,
vector retrieval, and local LLM generation. It is a learning demo, not an
industrial-grade RAG system or production service.

## Frozen Baseline Status

The `v0.5-local-rag-generation` tag freezes the M1–M5 baseline: Markdown and
text-based PDF loading, retrieval and evaluation, local Ollama generation,
RAG context construction, and source/page citations. M6 parser integration is
ready on the M6 development branch; industrial-PDF evaluation is still
pending.

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
PDF. Industrial-PDF A/B results and end-to-end answer validation have not yet
been measured. Complex tables, scanned-page completeness, and layout-heavy
documents remain unverified for this RAG pipeline.

## Roadmap

- [x] Project initialization
- [x] M1 — Minimal Retrieval Pipeline
- [x] M2 — Basic RAG
- [x] PDF ingestion
- [x] M3 — Source citation
- [x] Retrieval evaluation
- [x] M5 — Baseline freeze
- [x] M6 — MinerU parser integration infrastructure
- [ ] M6 — Industrial PDF A/B and end-to-end validation
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
