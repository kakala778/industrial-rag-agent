# Industrial RAG Agent

An experimental learning project for understanding an Industrial RAG system. It
is a minimal prototype, not a production system.

## Repository layout

- [`rag-agent/`](rag-agent/README.md) contains the RAG application and its
  M1–M10.1 experimental code, evaluation tools, and tests. M7.1 adds optional
  reranking; M7.2 audits which failures require retrieval or document
  processing improvements. M7.1 and M7.2 are complete experiments in the
  current worktree. M8.1 audits parsing failures, and M8.2 compares
  OCR-labelled text and retained visual metadata. The controlled M8.2 run did
  not reduce parsing failures; see the
  [M8.1 audit](rag-agent/docs/m8-document-intelligence-failure-audit.md),
  [M8.2 experiment](rag-agent/docs/m8-visual-representation-experiment.md),
  [M8.3 vision feasibility test](rag-agent/docs/m8-vision-model-feasibility-test.md),
  [M8.4 block-crop experiment](rag-agent/docs/m8-vision-block-experiment.md),
  and [M8.5 focused OCR comparison](rag-agent/docs/m8-focused-ocr-comparison.md).
  M8.5 found a small-cohort strict-evidence signal for focused Qwen3-VL crops.
  The target-crop review found no confirmed unsupported facts, but handwritten
  regions and one evidence-matcher discrepancy remain unresolved; this is not
  a production integration. M8.6 then measured whether a metadata gate could
  focus that OCR without changing retrieval. Gated VLM fallback was evaluated
  and not adopted; the offline gate missed all four confirmed OCR positives and
  left aggregate retrieval metrics unchanged. See the
  [M8.6 A/B report](rag-agent/docs/m8-gated-vision-retrieval-ab.md).
  M9.1 evaluates fixed in-memory BM25 + RRF without changing the default RAG
  pipeline. Hybrid+BGE improves Top-3 page 19/32 → 23/32 and normalized
  evidence 14/32 → 17/32, preserving all baseline successes. Four original
  retrieval failures enter Top-20; three reach final Top-3. See the
  [M9.1 report](rag-agent/docs/m9-hybrid-retrieval-experiment.md).
  M9.2 validates 21 new PDF-grounded questions frozen before inference.
  Hybrid+BGE increases evidence 12/21 → 15/21 and recall 12/21 → 16/21;
  Top-3 page stays 17/21 with two gains and two regressions. Historical audits
  identify matcher, scope and row-context limitations, so BGE is not established
  as the sole bottleneck. See the
  [M9.2 report](rag-agent/docs/m9-independent-validation-and-ranking-audit.md).
  M9.3 exposes the frozen Hybrid retriever as an opt-in PDF CLI path. Dense
  without reranking remains the default. Q01/Q06/Q33 local smokes verified
  context, citation metadata and the existing refusal path, but do not establish
  generation quality. See the
  [M9.3 integration note](rag-agent/docs/m9-optional-hybrid-integration.md).
  M10.1 tested bounded parent context before reranking. Q25/I17 improved, but
  one table success regressed and CPU cost increased; it remains offline and
  is not adopted in the RAG application. See the
  [M10.1 report](rag-agent/docs/m10-parent-context-controlled-experiment.md).
  The [M8 plan](rag-agent/docs/superpowers/plans/2026-10-01-m8-document-intelligence.md)
  records the broader experiment scope.
- [`rag-agent/components/industrial-preprocessor/`](rag-agent/components/industrial-preprocessor/README.md)
  contains the curated reusable preprocessing source from the local Project
  workspace. It remains a standalone component and is not imported by the RAG
  runtime.
- `minerU/mineru-405-poc/run-mineru.ps1` is the local MinerU launcher used by
  the merged workspace. Its virtual environment, models, PDFs, parser outputs,
  and experiment data stay local and are excluded from Git.

## Run the RAG demo

Run application commands from `rag-agent/`:

```powershell
cd rag-agent
python -m pip install -r requirements.txt
python src/rag_demo.py
```

The PDF demo supports PyMuPDF and the locally configured MinerU runner. A fresh
clone must install/configure its own MinerU runtime or set
`MINERU_RUNNER_PATH`.

Run the automated checks from `rag-agent/`:

```powershell
python -m unittest discover -s tests -v
```

## Data and security

Do not commit real industrial or teaching documents, PDFs, local MinerU output,
models, virtual environments, API keys, or `.env` files. The local workspace
under `minerU/` is ignored except for the portable launcher script.
