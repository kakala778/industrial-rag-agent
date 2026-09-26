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
- Preserves each document's text and source filename.

```python
{
    "text": "...",
    "source": "xxx.md"
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

## Observations

These are observations from the tested queries, not aggregate benchmark metrics.

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

## Next Step

### M2 — Basic RAG

Add answer generation to the retrieval flow:

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

M2 introduces generation. Agent capabilities remain outside this next milestone.

## Roadmap

- [x] Project initialization
- [x] M1 — Minimal Retrieval Pipeline
- [ ] M2 — Basic RAG
- [ ] PDF ingestion
- [ ] Source citation
- [ ] Retrieval evaluation
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
