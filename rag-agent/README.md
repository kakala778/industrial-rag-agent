# Industrial RAG Agent

## Project Overview

This is a minimal RAG prototype supporting Markdown/PDF document processing,
vector retrieval, and local LLM generation. It is a learning demo, not an
industrial-grade RAG system or production service.

## Current Stage — M10.1 complete; Agent next

**ENTER AGENT.** The final pre-Agent RAG experiment is complete. Parent-context
reranking remains offline and is not adopted in the application. Agent itself
is not implemented; the next work is a bounded Document Research workflow with
search, original-evidence lookup, task state and grounding checks. See the
[Agent handoff](docs/agent-handoff.md) and [M10.1 results](docs/m10-parent-context-controlled-experiment.md).

The experiment checkpoint is `2918f99` on `codex/m9-hybrid-retrieval`.

## Frozen Baseline Status

The `v0.5-local-rag-generation` tag freezes the M1–M5 baseline: Markdown and
text-based PDF loading, retrieval and evaluation, local Ollama generation,
RAG context construction, and source/page citations. M6 validation is now
complete as an experimental baseline on eight anonymized local PDFs. This is
not a production-readiness claim; the expanded measurements and limitations
are recorded in the [M6 evaluation report](docs/m6-industrial-pdf-evaluation.md).
M7.1 adds an optional reranker experiment while preserving dense retrieval as
the default, and M7.2 records a case-level failure analysis. Both are
experimental milestones; the reranker is not the default path.
M9.1 and M9.2 evaluated a fixed BM25 + RRF strategy on the original and an
independent question set. M9.3 exposes that strategy as an opt-in PDF CLI
retriever; Dense without reranking remains the default. The local smoke results
and limitations are recorded in the
[M9.3 integration note](docs/m9-optional-hybrid-integration.md).

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
- Optional MinerU structured PDF parsing, BGE reranking and BM25 + RRF Hybrid
  retrieval in the PDF CLI; default selectors are unchanged
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

Run the Markdown RAG demo from the `rag-agent/` application directory. It reads local `.md` files
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

The PDF CLI defaults to Dense retrieval without BGE reranking. The existing
`--mode dense|reranker|compare` interface remains available. To select the
optional Hybrid retriever, use `--retriever hybrid`; add `--rerank` to apply the
existing BGE reranker to its Top-20 candidates. The new selectors also support
Dense with or without reranking:

```bash
python src/pdf_rag_demo.py path/to/local.pdf
python src/pdf_rag_demo.py path/to/local.pdf --mode reranker
python src/pdf_rag_demo.py path/to/local.pdf --retriever hybrid
python src/pdf_rag_demo.py path/to/local.pdf --retriever hybrid --rerank
```

The legacy `--mode` selector cannot be combined with `--retriever` or
`--rerank`. Hybrid rank and score diagnostics are kept out of ordinary result
and citation display; Dense cosine and Hybrid RRF scores have different scales.

## Current Limitations

This baseline does not include:

- Vector database
- Production reranking; the optional M7.1 reranker is experimental, and dense
  retrieval remains the default
- Permission and access control
- Document version management
- Multi-tenant support
- Monitoring
- Agent workflow
- Memory
- Web API

The default PyMuPDF loader extracts page text without OCR. The optional MinerU
parser supports Advanced + OCR and structured blocks, but OCR accuracy, image
information and layout relationships remain limited. Real project and
teacher-provided data must remain out of the repository.

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

## M7.1 — Reranker Experiment

Added reranker experiment on top of the dense retrieval baseline. The purpose
was to test whether a cross-encoder can improve ordering when dense retrieval
has already recalled the correct evidence. The experiment keeps the existing
retrieval implementation and default mode intact:

```text
Dense retrieval Top-20
→ BAAI/bge-reranker-v2-m3
→ Top-3
```

On first use, Hugging Face downloads the model into its local user cache; model
weights are not stored in this repository.

Run the PDF RAG CLI in dense mode (default), reranker mode, or compare both for
each question:

```bash
python src/pdf_rag_demo.py path/to/local.pdf --mode dense
python src/pdf_rag_demo.py path/to/local.pdf --mode reranker
python src/pdf_rag_demo.py path/to/local.pdf --mode compare
```

Run the M6 PDF retrieval set through both modes with local PDFs and QA data:

```bash
python evaluation/evaluate_pdf_retrieval.py \
  --mode compare \
  --pdf-dir path/to/local-pdfs \
  --dataset path/to/local-m6-qa.json \
  --parser mineru
```

The comparison used the same 32 answerable M6 questions and cached structured
MinerU documents. Retrieval and evidence metrics were:

| Metric | Baseline Dense | Dense + Reranker |
| --- | ---: | ---: |
| Top-1 source hit | 25/32 | 31/32 |
| Top-3 source hit | 30/32 | 31/32 |
| Top-1 page hit | 9/32 | 18/32 |
| Top-3 page hit | 15/32 | 19/32 |
| Raw evidence hit | 8/32 | 11/32 |
| Normalized evidence hit | 10/32 | 14/32 |

The evaluator changed the failure attribution from 10 to 3 RANKING cases;
PARSING remained at 11. Four of the 10 cases originally attributed to RANKING
moved into evidence Top-3. In one case, the correct evidence moved from dense
rank 4 to reranker rank 1. These are results from one local 32-question set,
not a general accuracy claim. The run used CPU-only PyTorch. The original M7.1
summary did not preserve case rows; the M7.2 replay below reconciles the
baseline RANKING cases using the same evidence matcher.

**Decision:** keep reranking as an optional experiment for comparisons; retain
dense retrieval as the default. Reranking improved the measured ranking metrics
but did not change the PARSING count, and it cannot recover evidence absent
from its dense Top-20 candidates. Remaining failures should be audited before
adding another retrieval component.

## M7.2 — Reranker Failure Analysis

Purpose: **“Analyze which failures are solved by reranking and which require
retrieval or document processing improvements.”** This is a case-level audit;
it does not change retrieval, reranking, embedding, chunking, MinerU, or prompts.

The real M6 replay used 35 QA rows (32 scored, 3 unanswerable), 8 original
PDFs, and 8/8 matching MinerU structured caches. Its 25 original failures
classify as:

| Category | Count |
| --- | ---: |
| FIXED_BY_RERANKER | 4 |
| STILL_RANKING_FAILURE | 3 |
| RETRIEVAL_FAILURE | 4 |
| PARSING_FAILURE | 11 |
| INSUFFICIENT_DATA | 3 |

The 10 M6 RANKING labels split into 4 fixed, 3 still ranking, and 3 unassigned
cases whose evidence already matched dense and reranked Top-3. Those three are
left as `INSUFFICIENT_DATA` because they do not fit the specified causal classes.
That resolves the aggregate discrepancy without forcing a label.
The full anonymized cases and real examples are in
[`docs/m7-reranker-failure-analysis.md`](docs/m7-reranker-failure-analysis.md).

The replay command from `rag-agent/` is:

```bash
python evaluation/evaluate_pdf_retrieval.py \
  --mode compare \
  --parser mineru \
  --representation structured \
  --pdf-dir path/to/industrial-rag-data/input \
  --dataset path/to/industrial-rag-data/qa/m6_final_qa.local.json \
  --failure-analysis-output outputs/m7_failure_analysis.json
```

The generated file is Git-ignored. It contains one row per question with the
dense candidate count and Top-20 scores/ranks, reranked Top-3 scores/ranks,
rank movement, and allowlisted source/page/block metadata. Source names become
run-local aliases. Question text, expected answers, evidence keywords, and
candidate document text are omitted. Parsing failures are counted only when
the MinerU structured evaluation explicitly confirms missing evidence.

PARSING_FAILURE is the largest remaining class (11/25, 44%), so the results
support a Document Intelligence experiment before Hybrid Search. Retrieval
failures are 4/25 and still-ranking failures are 3/25; they do not justify
starting BM25 or broad reranker tuning first. See the
[full report](docs/m7-reranker-failure-analysis.md) for the classification
rules, transition counts, and anonymized examples.

## M8.2 — Visual Information Representation Experiment

M7.2 found 11 `PARSING_FAILURE` cases among the 25 original M6 failures
(44%), the largest classified group. M8.1 audited all 11 against the local
original pages, Middle JSON, and structured Documents. Five were OCR failures,
two image-information losses, one layout-relation failure, and three remain
uncertain. No case confirmed table row/column structure loss or loss introduced
only by the structured adapter. M8.2 compared the current `structured` mode,
OCR-labelled existing image text, and full block metadata on the same local M6
data. Parsing failures remained 11/11. OCR labeling reduced the M7 Top-3 page
hit from 19/32 to 18/32 and normalized evidence from 14/32 to 13/32. Full
metadata preserved block geometry without changing retrieval metrics. No
visual model or external OCR was added. Details are in the
[M8.1 failure audit](docs/m8-document-intelligence-failure-audit.md) and
[M8.2 experiment report](docs/m8-visual-representation-experiment.md).

The M8.2 script requires the fixed local M6 cohort, a one-time ignored input
manifest, and matching MinerU caches. The manifest stores only hashes and
counts; initialize it with `python evaluation/run_m8_visual_representation_experiment.py --initialize-input-manifest`
when setting up the local data. Results contain anonymous metrics only in the
ignored `outputs/` directory. The default representation remains `structured`.

## M8.3 — Vision Model Feasibility Test

Tested the locally installed `qwen3-vl:2b-instruct-q4_K_M` on one preselected
case each for OCR, image-information loss, and layout-relation failure. It did
not recover the expected evidence in any of the three cases, so the result does
not support adding vision output to the RAG index. The page images and raw
responses remain in ignored local output. See the
[M8.3 feasibility report](docs/m8-vision-model-feasibility-test.md).

## M8.4 — Vision Block Controlled Experiment

Compared the same local vision model on a MinerU block crop, block crop plus
MinerU text, and full-page input for selected visual failures. Only the Q10
3× table-block crop recovered the target; image/layout cases remained
unresolved. This remains an offline result and did not change the default
representation. See the [M8.4 report](docs/m8-vision-block-experiment.md).

## M8.5 — Focused OCR Comparison

Compared the fixed five M8.1 OCR failures using identical 2×/3× MinerU block
crops for Qwen3-VL and local RapidOCR. At 3×, strict target-evidence recovery
was 3/5 for Qwen3-VL and 1/5 for RapidOCR; RapidOCR's normalized numeric hits
included substring false positives. The run supports a further gated offline
test, not default integration. The target-crop visual review found no clearly
unsupported extra fields, but handwritten regions and the Q17 evidence-matcher
mismatch remain unresolved; this is not a general OCR precision estimate. See
the [M8.5 report](docs/m8-focused-ocr-comparison.md).

## M8.6 — Gated Focused-VLM Retrieval A/B

Analyze whether a generic MinerU metadata gate can focus OCR fallback and improve
retrieval without disturbing known successes. The fixed cohort-limited A/B
missed all four confirmed OCR positives, activated on 3/10 known-success
controls, and left the 31-question formal metrics unchanged. A corpus-wide
metadata-only scan projected 86.5 VLM calls per PDF on average. Gated VLM
fallback was evaluated and not adopted; the default RAG pipeline remains
unchanged. Q17 remains GT_UNCERTAIN and is excluded from formal metrics. See the
[M8.6 report](docs/m8-gated-vision-retrieval-ab.md).

## M9.1 — Hybrid Retrieval Controlled Experiment

The fixed offline comparison uses the same 2,996 chunks: Dense Top-20 + BM25
Top-20 → deduplicated RRF (k=60) → Hybrid Top-20 → existing BGE Top-3.
Hybrid+BGE improves Top-3 page 19/32 → 23/32 and normalized evidence
14/32 → 17/32, with no regression among the 14 baseline evidence successes.
All four original retrieval failures enter Hybrid Top-20; three reach final
Top-3, while Q24 remains a ranking failure. Pure BM25 loses three Dense
successes; RRF protects them. Retain the optional strategy for further
validation; the default application pipeline remains unchanged, and M8 stays
frozen. See the [M9.1 report](docs/m9-hybrid-retrieval-experiment.md).

Run with the original ignored local inputs and caches:

```powershell
python evaluation/run_m9_hybrid_retrieval_experiment.py --compare-all
```

Individual experiment arms support `--retriever dense|bm25|hybrid` and
`--rerank on|off`; anonymous outputs remain under ignored
`outputs/m9_hybrid_retrieval/`. No vector database or new dependency is added.

## M9.2 — Independent Validation + Ranking Audit

21 new questions cover all eight local PDFs and eight categories; original-PDF
visual review and the QA/configuration manifest were completed before inference.
Fixed Hybrid+BGE improves evidence 12/21 → 15/21 and Recall@20 12/21 → 16/21.
Top-3 page stays 17/21: two gains and two page regressions, despite zero evidence
regressions. Independent data supports optional Hybrid integration, with page
localization checks; the default pipeline remains unchanged.

Q07/Q19/Q24/Q25 audits identify notation matching, underspecified query scope,
legitimate repeated fields and table chunk-context limitations. These do not
establish BGE as the main bottleneck. See the
[M9.2 report](docs/m9-independent-validation-and-ranking-audit.md).

```powershell
# One-time freeze after authoring and reviewing the ignored local QA:
python evaluation/run_m9_independent_validation.py --freeze
# Five fixed arms; subsequent replay uses this command without --freeze:
python evaluation/run_m9_independent_validation.py
# Historical candidate audit without model loading:
python evaluation/run_m9_independent_validation.py --audit-only
```

QA and detailed outputs stay under ignored `outputs/m9_independent_validation/`
and `outputs/m9_ranking_audit/`. No BM25/RRF, chunking, representation, model or
prompt parameter was changed; no vector database was added.

## M9.3 — Optional Hybrid Retrieval Integration

The frozen M9.1 Hybrid+BGE strategy is available as an explicit option in the
PDF RAG CLI. Invocation without selectors remains Dense with no reranker and no
BM25 index. Existing `--mode dense|reranker|compare` behavior is retained;
`--retriever hybrid` selects Hybrid, and `--rerank` optionally applies the
existing BGE model to its Top-20 candidates. Mixed legacy and new selectors are
rejected.

The M9.3 local smoke used Q01, Q06 and unanswerable Q33 only to verify wiring,
context, citations and the refusal path. Q01's Dense+BGE Top-3 missed the
target evidence scope while Hybrid+BGE included it; Q06 included target evidence
with both retrievers; Q33 triggered the existing refusal phrase. These checks
are not a new generation-quality benchmark. Dense remains the default. See the
[M9.3 implementation note](docs/m9-optional-hybrid-integration.md).

## M10.1 — Parent Context Controlled Experiment

Bounded parent context showed limited benefit in table cases and remains
experimental. Q25/I17 improved, but Q04 regressed and reranking cost increased;
the strategy was not adopted in the RAG application. The next stage is Agent,
with defaults unchanged. See the [M10.1 report](docs/m10-parent-context-controlled-experiment.md).

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
- [x] M7.1 — Dense + reranker comparison experiment
- [x] M7.2 — Reranker failure analysis on the original M6 inputs
- [x] M8.1 — Document Intelligence failure audit
- [x] M8.2 — Visual information representation experiment
- [x] M8.3 — Vision model feasibility test
- [x] M8.4 — Vision block controlled experiment
- [x] M8.5 — Focused OCR comparison (offline; handwritten regions and Q17 matcher discrepancy remain open)
- [x] M8.6 — Gated focused-VLM retrieval A/B (offline; gate not adopted)
- [x] M9.1 — Hybrid retrieval controlled experiment (offline; default unchanged)
- [x] M9.2 — Independent validation and ranking audit (offline; page regressions recorded)
- [x] M9.3 — Optional Hybrid retrieval integration (Dense remains default)
- [x] M10.1 — Parent context controlled experiment (offline; not adopted)
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
