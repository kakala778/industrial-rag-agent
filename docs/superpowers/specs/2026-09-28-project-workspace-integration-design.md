# Project Workspace Integration Design

## Goal

Make the reusable industrial document preprocessing code available inside the
public `industrial-rag-agent` repository without exposing local PDFs, models,
MinerU outputs, evaluation datasets, or machine-specific workspace details.

## Chosen layout

Add `components/industrial-preprocessor/` as a standalone subproject. It keeps
its own `src/` package, dependency metadata, and isolated invocation context;
it is not imported into the RAG root package. The root README links to the
component and explains how to run it from its own directory.

## Included

- Tracked preprocessing source under `src/`.
- `pyproject.toml` and a tightened local-artifact `.gitignore`.
- A curated component README with portable setup guidance.

## Excluded

- Nested `.git` history and all virtual environments or model directories.
- Separate local data, experiment, and runtime workspaces.
- PDFs, MinerU outputs, cached data, model files, private QA datasets, and
  experiment-specific scripts or reports tied to those inputs.
- The original test suite, which remains in the source workspace.
- Absolute local paths and the source repository's uncommitted reports.

The original `Project` workspace is left in place as a recovery copy. The
RAG's existing MinerU adapter and retrieval/generation behavior remain
unchanged; this integration co-locates the preprocessing code but does not
wire it into the RAG runtime.

## Verification

Review the selected-file inventory, scan the new component for local absolute
paths and prohibited data/model extensions, inspect `git diff --check`, and
confirm pre-existing RAG modifications remain unstaged. Do not copy or run
tests in this task.
