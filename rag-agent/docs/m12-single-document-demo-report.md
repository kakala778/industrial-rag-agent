# M12.1 — Single PDF Industrial Evidence Research Demo

**Status:** Complete
**Run date:** 2026-10-04

## Goal and boundary

This milestone exercised the existing bounded Evidence Research Agent against
one locally available industrial specification PDF. It did not tune retrieval,
change ranking parameters, add Agent actions, use the semantic comparator, or
introduce external orchestration or storage. The default PyMuPDF loader and the
existing `KnowledgeBaseSession` search path were used without parameter
overrides. Local Qwen `qwen3:4b` selected actions.

The only behavior adjustment was to let the `evidence_reference` path accept
one through four explicit scopes. M11 required at least two and therefore could
not run a genuine one-PDF task through its CLI and harness. The legacy
`copied_quote` Agent 0 path remains exactly two-scope. No retrieval behavior or
action type changed.

## Selected document and scope

The source is one 72-page local procurement technical specification for
communication transmission and access equipment. It is represented to
the Agent by the alias `S1`. Its PDF and generated reports are local and
ignored by Git; full excerpts stay in those private reports, and the run-time
trace was checked in memory but not persisted. This tracked report does not
reproduce source text or the private file path.

Page numbers below are 1-based physical pages in the PDF file. Printed footer
page numbers differ because the file includes front matter. With the unchanged
PyMuPDF path, the host provenance identifies a page-level `document` block;
`block_index` is the zero-based position of that page document in the loaded
corpus. It is not a table-cell coordinate.

## Questions and observed runs

Each question was run in a fresh `AgentHarness` state over the same loaded
single-document session. The trace was checked in memory and not written to the
repository. All three runs selected `SEARCH → LOOKUP → FINISH`; every LOOKUP ID
had appeared in an earlier successful scoped SEARCH, and each FINISH passed
the host contract.

| Case | Research question | Result | Verified location |
| --- | --- | --- | --- |
| Q1 — normal lookup | Find the operating-environment requirements in the selected specification. | `evidence_found`; one host-rendered citation. The selected excerpt contains the requested requirement fields. | PDF page 21; `document:20` |
| Q2 — table lookup | Find a target equipment row's quantity and configuration in a structured schedule. | `evidence_found`; one host-rendered citation. The selected excerpt identifies the requested row and fields. | PDF page 10; `document:9` |
| Q3 — missing information | Find a requested protocol/security parameter using only the selected specification. | `insufficient_scope`; one candidate was looked up and no evidence ID was selected, so no citation is shown. | No final citation |

For Q3, the requested protocol/security terms were not found in
PyMuPDF-extracted document text. Search still returned a candidate, so
`insufficient_scope` is the valid M11 status; `no_evidence_found` would
contradict the observed candidate. This result means the retrieved material
did not support an answer. It does not prove that the complete source document
contains no relevant information in content the parser did not extract.

## Evidence integrity and report example

For Q1 and Q2, the report citation IDs resolved in a newly constructed session
from the same local PDF. Each host-rendered excerpt matched text on its cited
physical PDF page. The cited pages were also rendered and visually checked;
the Q2 citation points to the page containing the GPON equipment schedule.
Reports were UTF-8, contained the required limitation statement, and did not
include an absolute input path. The report writer generated unique filenames
under the ignored `outputs/agent11/` directory.

The following is a sanitized excerpt of the generated Q1 report. The actual
local report contains the host citation ID and source excerpt; they are
redacted here to keep private source material out of Git.

```markdown
# Industrial Research Report

## Task
> [Private task text omitted]

## Sources
- `S1`

## Evidence
### Source `S1`
- Citation: `[private host citation omitted]`
- Location: page 21; block document:20
> [Private source excerpt omitted]

## Findings
Terminal status: `finished`
- Source `S1`: `evidence_found` — selected host-validated source records

## Limitations
- Evidence identifies source content but does not prove engineering correctness.
- This report provides no engineering or compliance verdict.
```

To reproduce one run from `rag-agent/`, use the local PDF path supplied by the
operator:

```powershell
python -m src.agent_demo --task "Find the operating-environment requirements for the transmission system." `
  --document S1=path/to/local-specification.pdf `
  --policy qwen
```

The two other questions are listed above; run each as a separate `--task` so
each report records an independent bounded investigation.

## Limitations and decision

- This is three observed tasks on one document, not a benchmark or an estimate
  of general accuracy.
- Evidence selection is model-driven. Authentic citations establish source
  identity and location, not semantic support, completeness, or correctness.
- The PyMuPDF extraction path retains the table content in page text but does
  not create cell-level source blocks. The table citation is page-level; a
  user must inspect the original table for row/column interpretation.
- `insufficient_scope` describes the retrieved candidates and looked-up
  material only. It does not establish document-wide absence.
- No engineering, safety, standard-compliance, or recommendation judgment was
  requested or produced.
- The local retrieval-model loader emitted an unauthenticated Hugging Face Hub
  notice while loading cached model weights. The document task and evidence
  were sent only to local Qwen; no remote inference was used.

M12.1 demonstrates that the existing bounded Agent can run a single local PDF,
produce inspectable host-rendered citations for answerable questions, and
withhold a citation when retrieved evidence does not support a requested fact.
It does not establish industrial-document answer accuracy. RAG and Agent 1
remain frozen; this milestone did not start Agent 2.
