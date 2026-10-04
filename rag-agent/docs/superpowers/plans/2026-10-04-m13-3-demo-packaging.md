# M13.3 Demo Packaging Implementation Plan

> Implement inline in the current checkout. Do not commit or push without an explicit instruction.

**Goal:** Package the existing local Evidence Research Agent as an understandable, honest, reviewer-facing first demo.

**Architecture:** Keep the existing Streamlit-to-Agent adapter and all core contracts unchanged. Add task presets and presentation copy in the existing UI, document local setup and limitations, and include only sanitized UI/trace screenshots plus a static architecture diagram.

**Tech Stack:** Python, Streamlit, unittest/AppTest, Playwright CLI, Markdown, SVG.

**Spec:** User-provided M13.3 task attachment dated 2026-10-04.

## Global Constraints

- Do not modify retrieval algorithms, embeddings, reranking, chunking, MinerU, Agent action schemas, evidence contracts, semantic comparison, or benchmark data.
- A preexisting machine-specific audit input path may be replaced with runtime configuration to satisfy the no-local-path privacy check; do not change offline scoring logic or frozen data.
- Example tasks are suggestions; do not claim stable answer accuracy.
- Public screenshots must not contain private PDFs, source excerpts, local paths, raw outputs, or credentials.
- Preserve all M13.1 working-tree changes; use the canonical checkout and current branch.
- Do not commit or push.

## Review Focus

- Example selection must update the task field without submitting the form.
- Example buttons must work with an already-uploaded PDF and preserve normal form submission.
- Evidence status copy must distinguish host-resolved citations from relevance or correctness.
- README commands and prerequisites must match the existing local runtime.
- Public screenshots must be checked for source excerpts, document paths, and transient UI artifacts.

## Tasks

### Task 1: Reviewer-facing Streamlit presentation

**Files:** Modify `app/components/document_panel.py`, `app/components/evidence.py`, `app/streamlit_app.py`; test `tests/test_streamlit_examples.py`.

- [x] Add AppTest coverage for the three example buttons, task-field updates, and no implicit submission; run it and confirm failure before implementation.
- [x] Add the three M12 task presets outside the input form; selecting one only fills `research_task` and clears stale results.
- [x] Add project title, capability summary, and visible limitation statement.
- [x] Label action trace and host reference status separately; retain the existing correctness limitation.
- [x] Run the focused test and inspect the rendered browser UI.

### Task 2: Reviewer documentation and static architecture diagram

**Files:** Modify root `README.md`, `rag-agent/README.md`, `rag-agent/docs/agent-handoff.md`; create `rag-agent/docs/m13-3-demo-release.md` and `rag-agent/docs/assets/architecture.svg`.

- [x] Document local dependencies, launch command, one-PDF flow, the three task presets, system diagram, and explicit non-goals.
- [x] Record the M13.2 page-7 versus prior M12 page-21 observation without exposing the private source excerpt.
- [x] Keep M12 reports historical and unchanged.

### Task 3: Sanitized GitHub presentation assets

**Files:** Create `rag-agent/docs/assets/workspace-empty.png` and `rag-agent/docs/assets/example-action-trace.png`.

- [x] Capture the empty Streamlit landing UI.
- [x] Include the sanitized timeline crop from the previously successful M13.2 real browser run; omit private evidence text, document identity, evidence ID, and path.
- [x] Inspect both images and verify no private excerpts or local paths are visible.

### Task 4: Full verification and closeout

**Files:** No additional code changes unless a focused UI issue is reproducible.

- [x] Run `python -m compileall src evaluation app`.
- [x] Run `python -m unittest discover -s tests` (353 tests passed).
- [x] Verify `python -m src.agent_demo --help` and a live Streamlit launch. The fresh structured Qwen replay stopped after SEARCH with local Ollama HTTP 500; the previously successful M13.2 trace is the packaged workflow capture.
- [x] Run `git diff --check` and audit tracked paths/content for PDFs, local absolute paths, model outputs, and credentials. No PDFs are tracked; user-home/project paths were removed from text sources; Playwright artifacts are ignored.
- [x] Report the observed demo behavior, limitations, working-tree state, and that no commit/push occurred.
