# Industrial RAG Agent

## Goal

This is a learning and experimentation project for understanding and building an Industrial RAG + Agent system from scratch. It is not a production system.

## Planned Pipeline

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

## Current Status

M1 — Minimal Retrieval Pipeline completed.

M1 also includes a batch evaluation script for the curated retrieval test set.

M1 retrieval logic is exposed as a reusable module and reused by M2.

M2 — Basic RAG completed. The CLI retrieves relevant chunks, builds a context
and prompt, and sends them to the locally configured Qwen3 4B model through
Ollama.

M3 — Citation + Retrieval Evaluation completed. The RAG CLI displays the
retrieved source, chunk ID, and similarity score after each answer. A separate
10-question retrieval evaluation reports Top-1 and Top-3 source accuracy.

M4.1 PDF loading completed. M4.2 Unified document pipeline completed. Markdown
and PDF now use the same Document format, and PDF documents can use the existing
chunking, retrieval, and local RAG flow.

The current retrieval flow is:

```text
Documents
↓
Chunking
↓
Embedding
↓
Cosine Similarity Retrieval
↓
Top-K Results
```

M1 retrieves and ranks document chunks. It does not generate answers.

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
   - The current implementation retains `source` and `chunk_id`; it does not yet store a separate `section` metadata field.

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
with the question and a grounding-focused system prompt, then calls the local
Ollama `/api/chat` endpoint with `qwen3:4b`. It does not use a cloud API or
implement an Agent. M3 adds a `Sources` section after the answer; it lists the
retrieved source/chunk pairs and scores rather than annotating individual
answer claims.

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

`src/rag_demo.py` prints each unique retrieved source/chunk pair and its
similarity score after the generated answer. These sources show which retrieved
chunks were supplied as context; they are not claim-level citations.

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

## Roadmap

- [x] Project initialization
- [x] M1 — Minimal Retrieval Pipeline
- [x] M2 — Basic RAG
- [x] PDF ingestion
- [x] M3 — Source citation
- [x] Retrieval evaluation
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
- `.env` will not be uploaded.
- Future test data in this repository will be self-created or explicitly permitted for public use.
