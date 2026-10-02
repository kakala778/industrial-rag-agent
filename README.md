# Industrial RAG Agent

An experimental learning project for understanding an Industrial RAG system. It
is a minimal prototype, not a production system.

## Repository layout

- [`rag-agent/`](rag-agent/README.md) contains the RAG application and its
  M1–M8.6 experimental code, evaluation tools, and tests. M7.1 adds optional
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
