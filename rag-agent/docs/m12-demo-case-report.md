# M12.2 — Industrial Evidence Research Demo Case

**Status:** Complete

**Run date:** 2026-10-04

## Purpose and boundary

This package presents the M12.1 single-document validation as a reviewer-ready
demo. It reuses the existing Evidence Research Agent and its local CLI; it adds
no Agent behavior or retrieval capability. RAG defaults and retrieval
parameters, Agent action types and loop, evidence contract, citation
validation, Agent 1, and benchmark logic remain frozen.

The demo performs document evidence research only. It does not issue an
engineering, safety, compliance, or design recommendation.

## Scenario

The local source is a 72-page procurement technical specification for
communication transmission and access equipment. The CLI receives it under
the caller-chosen alias `S1`. The demo asks the Agent to find equipment
operating-environment requirements, retrieve a quantity/configuration entry
from a structured schedule, and check a requested protocol/security field
that the retrieved evidence does not support.

The tracked report omits the private PDF filename and path, exact private task
wording, evidence excerpts, and session-specific citation IDs. The complete
generated reports remain on the local machine under the ignored
`outputs/agent11/` directory.

## Architecture flow

```text
Local industrial PDF
        |
        v
Existing PyMuPDF Document Loader
        |
        v
KnowledgeBaseSession (scope alias S1)
        |
        v
Bounded Evidence Research Agent
   SEARCH -> LOOKUP -> CLARIFY / FINISH
        |
        v
Host evidence-ID and citation validation
        |
        v
UTF-8 Markdown research report
  (local, ignored outputs/agent11/)
```

## Demo cases and observed CLI runs

Each task ran independently against the same one-document scope with the
existing local Qwen policy (`qwen3:4b`). For all three runs the trace was
`SEARCH -> LOOKUP -> FINISH`; each LOOKUP ID had first appeared in SEARCH, and
FINISH passed host validation.

| Case | Sanitized task summary | Final result | Citations and evidence location | Generated report summary |
| --- | --- | --- | --- | --- |
| A — normal evidence | Locate the operating-environment requirements. | `finished`; scope `evidence_found`. | 1 citation; PDF physical page 21, block `document:20` (printed footer page 17). | One host-rendered evidence entry under `S1`; no engineering verdict. |
| B — structured information | Locate a target equipment row's quantity and configuration. | `finished`; scope `evidence_found`. | 1 citation; PDF physical page 10, block `document:9` (printed footer page 6). | One host-rendered evidence entry under `S1`; the citation points to a page-level document block, not a table cell. |
| C — missing information | Check whether a requested protocol/security parameter is supported. | `finished`; scope `insufficient_scope`. | 0 citations; one candidate was looked up, but no evidence ID was selected. | No evidence entry or factual answer was emitted for the unsupported field. |

For A and B, the citation IDs resolved in the active session, their source and
page metadata matched the session registry, and the cited pages existed. The
rendered excerpts matched text extracted from the cited pages; visual review
also confirmed page 21 contains the operating-environment section and page 10
contains the equipment schedule. Reports were UTF-8 and contained no local
absolute PDF path. The source PDF's printed footer numbers differ from its
physical PDF page numbers because of front matter.

Case C's status describes only the searched and looked-up evidence. It does
not prove that the requested information is absent from every part of the
document, especially content not represented by the unchanged text parser.

## Reproduce locally

From `rag-agent/`, substitute the path to the same local specification. Run
each task as a separate command to create an independent report:

```powershell
python -m src.agent_demo --task "Find the operating-environment requirements in the selected specification." `
  --document S1=path/to/local-specification.pdf `
  --policy qwen

python -m src.agent_demo --task "Find a target equipment row's quantity and configuration in the structured schedule." `
  --document S1=path/to/local-specification.pdf `
  --policy qwen

python -m src.agent_demo --task "Check whether the requested protocol/security parameter is supported by the selected specification." `
  --document S1=path/to/local-specification.pdf `
  --policy qwen
```

The tracked questions above are sanitized task summaries; adapt the task text
to the local document without adding an engineering-verdict request. The CLI
prints a report path under ignored `outputs/agent11/`. Generated reports can
contain task text and source excerpts, so keep them local.

## What this demo establishes

The demo demonstrates bounded document research, evidence-backed retrieval,
citation traceability, and an unsupported-evidence outcome without a fabricated
citation. It does not demonstrate answer accuracy in general, engineering
correctness, standards compliance, safety, or autonomous industrial decision
making. A citation identifies source content and location; it does not prove
that a conclusion is relevant or correct.

The PyMuPDF path exposes page-level `document` blocks rather than table-cell
coordinates, so a reviewer must inspect the original schedule when interpreting
row and column relationships. M12.2 did not change the parser, retrieval,
ranking, Agent, or evaluation code and does not start Agent 2.

During the local runs, the model loader printed an unauthenticated Hugging Face
Hub notice while loading cached weights. Agent action selection and inference
used local Qwen; the task and evidence were not sent to a remote inference
provider. This demo did not audit unrelated dependency-level network traffic.
