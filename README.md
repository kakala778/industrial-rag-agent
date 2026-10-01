# Industrial RAG Agent

An experimental learning project for understanding an Industrial RAG system. It
is a minimal prototype, not a production system.

## Repository layout

- [`rag-agent/`](rag-agent/README.md) contains the RAG application and its
  M1–M7 experimental code, evaluation tools, and tests. M7.1 adds optional
  reranking; M7.2 audits which failures require retrieval or document
  processing improvements. M7.1 and M7.2 are complete experiments in the
  current worktree. M8 is prepared as a Document Intelligence experiment but
  has not started; see the
  [M8 plan](rag-agent/docs/superpowers/plans/2026-10-01-m8-document-intelligence.md).
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
