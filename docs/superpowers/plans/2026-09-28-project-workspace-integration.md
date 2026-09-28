# Plan: integrate the Project workspace safely

## Scope and constraints

- Base: current `feat/m6-mineru-industrial-rag` checkout.
- Preserve the existing unstaged RAG evaluator and test modifications.
- Preserve all original files under the separate Project workspace.
- Include only the standalone preprocessing component; leave the source test
  suite and data-dependent experiments in the original workspace.
- Do not change Retrieval, RAG, or MinerU adapter behavior.
- Do not add or run tests; the user asked for integration and GitHub sync, not
  a test run.

## File responsibilities

- `components/industrial-preprocessor/src/`: reusable parser and document
  processing source copied from tracked files, including current local fixes.
- `components/industrial-preprocessor/pyproject.toml`: standalone package
  metadata.
- `components/industrial-preprocessor/.gitignore`: prevent future local data,
  model, cache, and generated-output commits.
- `components/industrial-preprocessor/README.md`: isolated setup and usage.
- `README.md`: explain the component boundary and local-only runtime/data.
- This spec and plan: record integration scope and exclusions.

## Steps

1. Copy only tracked `src/` and package metadata; omit nested Git metadata,
   tests, experiment scripts, and all local outputs.
2. Add the component README and scoped ignores. Document that commands run
   from the component directory to avoid the two projects' `src` packages
   colliding.
3. Add a concise root README section that distinguishes co-location from
   runtime integration and notes the external MinerU runtime remains local.
4. Inspect the final diff and staged-file inventory; scan for absolute local
   paths, private data/model artifacts, nested repositories, and pre-existing
   unrelated changes.
5. Commit only the integration files, then push the current feature branch
   once. If the network fails, stop and provide the manual push command.

## Completion checks

- `git diff --check` succeeds.
- The only staged files are the planned integration and documentation files.
- Existing RAG modifications remain untouched and unstaged.
- No PDF, archive, model, cache, virtual environment, or private dataset is
  staged.
- No tests are copied or run in this task.
