# Industrial RAG Agent V1 Release

**Release scope:** local, first-version reviewer demo  
**Status:** V1 demo development closed  
**Date:** 2026-10-04

## Project goal

Industrial requirements are often distributed across technical descriptions,
tables, and specifications. This project demonstrates how a reviewer can
search one local industrial PDF, inspect the evidence selected by a bounded
Agent, and read a report that preserves the evidence's source location. It is
a learning and demonstration project, not an industrial decision system.

## Architecture

The Streamlit demo connects the existing PDF and evidence APIs without
changing the RAG baseline:

```text
One local PDF
    ↓
PyMuPDF document loader
    ↓
KnowledgeBaseSession
    ↓
KnowledgeBaseSession SEARCH (Hybrid + BGE defaults)
    ↓
Bounded Agent loop: SEARCH / LOOKUP / CLARIFY / FINISH
    ↓
Host resolution of selected evidence IDs
    ↓
Host citation validation and rendering
    ↓
Streamlit workspace
    ↓
Markdown research report
```

The V1 Streamlit interface handles one PDF per run and uses the existing
PyMuPDF loader. Its SEARCH action calls `KnowledgeBaseSession` without a
backend override, so the existing tool defaults apply: Hybrid retrieval
followed by BGE reranking. The standalone PDF RAG CLI remains PyMuPDF plus
Dense without reranking by default; its optional MinerU, Hybrid, and BGE
selectors remain separate. M13 packaging did not change either path or its
parameters. The UI displays the trace captured by the synchronous Agent run
after it returns.

## Implemented capabilities

- Research over one locally uploaded PDF in the Streamlit demo.
- SEARCH candidates and LOOKUP of evidence IDs observed in the Agent run.
- Bounded SEARCH / LOOKUP / CLARIFY / FINISH actions with host-side contracts
  and termination limits.
- Host validation and rendering of selected evidence IDs with source, page,
  block, excerpt, and citation information.
- Neutral scope statuses and honest handling of insufficient evidence or
  clarification. An empty or insufficient result does not prove that the
  document contains no relevant information.
- An inspectable action timeline and a Markdown research report. The UI
  renders the report; the existing CLI can save reports under the ignored
  `outputs/agent11/` directory.
- Local Qwen/Ollama selection for the interactive demo.

The CLI evidence-reference interface can use one to four explicit document
aliases. The Streamlit reviewer interface remains single-PDF.

## Design decisions

**A bounded Agent instead of an autonomous Agent.** The demo uses a small,
explicit action set, validated inputs, evidence IDs, and finite execution
limits. This keeps runs inspectable and makes termination and scope behavior
part of the host contract.

**Host-generated citations instead of model-generated citations.** The model
selects evidence IDs from observed search results. The host resolves those IDs
against the active document session and renders their source locations and
excerpts. This prevents the model from inventing citation metadata; it does
not establish that a selected excerpt supports a claim.

**No semantic engineering verdict.** The Agent reports evidence and neutral
scope outcomes. The separate semantic comparator remains an experiment: its
observed unit and condition/applicability accuracies were 8/15 and 7/15, which
are insufficient grounds for engineering conclusions. It is not part of the
default demo flow.

**No uncontrolled tool execution.** The demo does not give the model a shell,
arbitrary network access, or unrestricted file tools. It reuses the explicit
search and evidence-lookup contract over the document supplied for that run.

## Known limitations

- Retrieval quality and selection of the most useful passage are not
  guaranteed. In the recorded M13.2 UI run, Qwen selected a page-7 instruction
  to provide operating-environment information rather than the target
  parameter values; an earlier M12 CLI run selected the relevant page-21
  section. Both observations are preserved, and the UI run is not counted as a
  successful retrieval of those values.
- A packaging replay on 2026-10-04 reached SEARCH, then the structured local
  Ollama action request returned HTTP 500. That run ended with `tool_error`
  and no final citation. A simple chat request returned HTTP 200, but the
  underlying runner error was not retained, so its cause is unresolved.
- A citation establishes the source location that the host rendered. It does
  not prove relevance, completeness, claim support, semantic correctness,
  engineering correctness, safety, or compliance.
- The default PyMuPDF path depends on text represented in the PDF. OCR,
  layout, table relationships, and row or cell alignment may be missing or
  imperfect. The current evidence view has page-level provenance rather than
  table-cell coordinates.
- `no_evidence_found` and `insufficient_scope` describe the bounded search and
  selection run; neither proves document-wide absence.
- The UI is local, synchronous, and single-document. It has no accounts,
  durable history, or online deployment. Dependency names in
  `requirements.txt` are not version-pinned, so future installs may resolve to
  different package versions.
- This project is not production-ready and must not be used as the sole basis
  for engineering, safety, or compliance decisions.

## Future directions

Possible V2 research directions include multimodal industrial documents,
carefully evaluated retrieval or reranking changes, CAD and image
understanding, enterprise knowledge-system integration, and bounded
engineering diagnosis workflows with explicit validation and review. These
are possible future investigations, not unfinished V1 requirements or claims
that the current system already provides those capabilities.

## Reviewer documentation

- [Demo guide](demo-guide.md)
- [V1 changelog](v1-changelog.md)
- [V1 final assessment](v1-final-assessment.md)
- [M13.3 package observations and assessment](m13-3-demo-release.md)
- [Current Agent handoff](agent-handoff.md)
- [Historical M10 architecture review](m10-architecture-review-and-agent-readiness.md)
- [M12.2 real-document case](m12-demo-case-report.md)
