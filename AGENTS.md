# Industrial RAG Agent — Coding Agent Instructions

## Project

This repository is a local learning and research prototype for industrial-document RAG and evidence-grounded document research agents. It is not a production industrial system.

Treat the repository as the source of truth.

For general project status and usage, see `README.md`.

For current Agent-stage status, active decisions, and handoff context, read `rag-agent/docs/agent-handoff.md` when the task concerns Agent development.

For retrieval, parsing, or architecture changes, consult `rag-agent/docs/m10-architecture-review-and-agent-readiness.md` and only the experiment reports relevant to the task.

Do not read all historical reports by default. Read only what the current task requires.

## Stable architectural boundaries

The established application baseline and experimental capabilities are intentionally different.

- PyMuPDF + Dense remains the ordinary PDF RAG default unless current code or documentation explicitly says otherwise.
- MinerU structured parsing is an optional document-processing path.
- Hybrid retrieval is an optional retrieval backend.
- BGE reranking is optional.
- Parent-context expansion remains experimental/offline unless a later documented decision adopts it.
- Visual/OCR enrichment beyond the established parser path remains experimental unless a later documented decision adopts it.
- Agent capabilities may have experimental implementations that are not equivalent to production-ready behavior.

Do not silently promote an experimental technique into the default runtime.

Do not add infrastructure or techniques merely because they are more advanced.

Introducing technologies such as LangGraph, a vector database, GraphRAG, HyDE, a new embedding model, a new reranker, runtime Vision LLM integration, long-term memory, or multi-agent orchestration requires a concrete observed problem and an evaluation rationale.

## Agent development principles

Prefer a small, explicit, inspectable Agent architecture.

Agent behavior should remain bounded by:

- explicit state;
- explicit action schemas;
- validated tool contracts;
- evidence provenance;
- progress and coverage checks;
- call and step budgets;
- explicit terminal states.

Prefer deterministic host guarantees for facts the program can enforce reliably.

Examples include:

- evidence identity validation;
- provenance;
- source/page/block lookup;
- citation rendering;
- scope validation;
- duplicate-action detection;
- budgets and termination.

Use the model for semantic judgment where deterministic code is insufficient.

Do not introduce an Agent framework unless requirements such as durable persistence, pause/resume, complex branching, or similar orchestration complexity justify it.

Do not expose or depend on hidden chain-of-thought. Agent observability should come from actions, tool inputs, tool results, evidence IDs, state transitions, and terminal status.

## Evidence and retrieval

Preserve source/page/block provenance through the pipeline.

Do not treat document-local `chunk_id` as a globally stable evidence identifier.

Use the repository's established evidence identity mechanism where available.

Dense cosine, BM25, RRF, and BGE scores have different semantics and scales.

Never treat retrieval or reranking scores as calibrated answer confidence.

A citation proves where retrieved evidence came from. It does not by itself prove that:

- the parser reproduced the original PDF perfectly;
- the evidence is semantically relevant to the question;
- every generated claim is supported;
- two engineering facts are equivalent.

Preserve the user's original query.

Query rewriting or decomposition must not invent document, equipment, page, drawing, entity, or other scope that the user did not provide.

Distinguish clearly between:

- no candidates;
- insufficient relevant evidence;
- ambiguous or missing scope;
- invalid evidence ID;
- timeout;
- tool or service failure;
- conflicting evidence.

Do not silently guess when evidence is missing or scope is unresolved.

## Parsing and document intelligence

Retrieval cannot recover information that parsing failed to represent.

Do not automatically reinterpret OCR, image, table, or layout failures as retrieval failures.

Do not silently normalize uncertain OCR fields, identifiers, numbers, units, signs, or model codes.

Preserve uncertainty and provenance when a correction cannot be established reliably.

Treat structured-document representation, retrieval behavior, and parser accuracy as separate failure layers.

## Engineering approach

Before significant changes:

1. inspect `git status` and the current branch;
2. inspect the relevant implementation;
3. inspect the relevant tests;
4. read only the design or experiment documents required for the task;
5. identify the smallest change that tests the current hypothesis.

Do not overwrite or discard unknown user changes.

Prefer focused changes over broad rewrites.

Reuse existing interfaces before creating new abstractions.

Do not preserve complexity merely because it already exists.

Keep experimental code distinguishable from adopted runtime behavior.

Avoid unrelated cleanup during a focused task.

Do not change multiple major variables in one controlled experiment unless the task explicitly requires it.

## Evaluation

This project is experiment-driven.

Do not claim an improvement based only on:

- a few manual examples;
- higher retrieval scores;
- passing unit tests;
- generated citations;
- fluent model output;
- qualitative impressions.

When behavior changes, define the expected behavior and evaluate both improvements and regressions.

Separate failure layers whenever possible, including:

- parsing;
- representation;
- retrieval;
- ranking;
- Agent action selection;
- evidence selection;
- citation/provenance;
- semantic interpretation;
- evaluation or ground-truth error.

Do not modify frozen ground truth, frozen search observations, candidate annotations, or evaluation inputs merely to improve metrics.

Only modify frozen evaluation data in an explicit audit or repair task, preserve the reason for each change, and recompute affected metrics without hiding the previous result.

Do not invent or substitute synthetic experimental results when required private inputs are unavailable.

Synthetic tests may validate mechanics, but they are not evidence of industrial-document accuracy.

## Verification

Run commands from `rag-agent/` unless the task requires another location.

Baseline checks:

```text
python -m compileall src evaluation
python -m unittest discover -s tests
git diff --check
```

Use narrower tests during development when appropriate.

Before treating a substantial change as complete:

1. inspect the final diff;
2. run the relevant broader verification;
3. verify privacy-sensitive artifacts remain ignored/untracked;
4. report exactly what actually ran.

If tests or evaluations cannot run because models, MinerU, private PDFs, API access, or other local resources are unavailable, state this explicitly.

Never report a test or evaluation as passing unless it actually ran successfully.

## Private data and credentials

Do not commit:

- industrial PDFs;
- private QA or GT datasets;
- private page or crop images;
- raw private model outputs;
- private Agent traces;
- MinerU caches;
- model files;
- credentials;
- `.env`;
- unintended local experiment outputs.

Check `.gitignore` before adding generated artifacts.

API credentials must be read from environment variables or another explicitly approved secret mechanism.

Never:

- hardcode API keys;
- print complete API keys;
- write credentials into traces, reports, fixtures, prompts, or generated artifacts.

## Git safety

Do not commit, push, merge, create a PR, create a tag, rewrite history, or delete branches unless the task explicitly authorizes that action.

Before changing files, inspect the working tree and preserve unrelated existing modifications.

Do not include private or ignored experimental data in commits.

## Documentation

Update documentation when a change alters:

- an architectural decision;
- CLI behavior;
- an experiment conclusion;
- an Agent/tool contract;
- an adopted/default capability.

Do not rewrite historical experiment reports merely to make them agree with later findings.

Preserve historical results and record later corrections, audits, or superseding decisions separately.

Keep transient milestone status out of this file when it can live in `README.md` or a handoff document.

## Completion

Before finishing a substantial task:

1. inspect the final diff;
2. run relevant verification;
3. summarize what changed;
4. state what was actually tested;
5. distinguish measured results from inference;
6. state remaining uncertainty and limitations;
7. report Git status and whether any commit/push occurred.