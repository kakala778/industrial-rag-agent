# Industrial RAG Agent V1 Demo Guide

This guide is for classmates, interviewers, and project reviewers who want to
run the local Streamlit demonstration and inspect what the Agent actually
did. The demo researches one local PDF per run; it is not an engineering
decision system.

## Environment requirements

- Python and `pip`, with a version supported by the packages in
  `requirements.txt`. The project does not currently pin package versions.
- Ollama installed separately, with the local `qwen3:4b` model available.
- Network access on first setup to install Python packages and, if not already
  cached, download the existing Sentence Transformers embedding and BGE
  reranker models.
- One permitted local PDF with extractable text. The default demo parser is
  PyMuPDF; scanned pages or complex layouts may not be represented reliably.

The closure checks for this V1 package use Python 3.14.4. Other Python and
dependency combinations may behave differently because the requirements are
not pinned.

## Install and start

In PowerShell, start at the repository root, enter `rag-agent/`, and create an
isolated environment there:

```powershell
cd rag-agent
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen3:4b
```

Ollama is a separate installation. If its local service is not already
running, start it in a second terminal:

```powershell
ollama serve
```

Then, from `rag-agent/`, start the demo:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Open the local address printed by Streamlit. The repository config binds the
demo to `127.0.0.1`; it is intended for local use. The first run may take
longer while the embedding and reranker models are downloaded and loaded.

The UI does not expose a retrieval selector. Its Agent SEARCH action uses the
existing `KnowledgeBaseSession` defaults: Hybrid retrieval followed by BGE
reranking. The standalone PDF RAG CLI has a separate Dense-without-reranking
default; V1 packaging did not change either path.

## Example workflow

1. Upload one permitted industrial PDF in the **Documents** panel.
2. Select an example task or type a focused research question. The example
   buttons only fill the task field; they do not submit or run it.
3. Review the task, then select **Run research**.
4. After the synchronous run returns, inspect the center timeline for the
   actual SEARCH, LOOKUP, CLARIFY, or FINISH actions and terminal status.
5. Inspect the right-hand evidence panel. Source alias, physical PDF page,
   block, excerpt, and citation are resolved and rendered by the host.
6. Read the Markdown report below the workspace, including its neutral scope
   status and limitations.

The uploaded PDF is written to a temporary file only while it is parsed; that
temporary file is removed before Agent execution. The current UI keeps results
in Streamlit session memory and does not persist them. The CLI's saved reports
are placed under the ignored `outputs/agent11/` directory and may contain
private task text and evidence excerpts; do not publish them.

## Demonstration scenarios

The three task presets are based on the M12 local demonstration. They are
examples, not benchmark cases, and do not guarantee a particular live-model
result.

| Scenario | What to inspect | Recorded observation |
| --- | --- | --- |
| Operating-environment requirements | Whether the selected excerpt contains the requested parameter fields, not just a related instruction. | An M12 CLI run cited page 21, which contained the parameter section. The M13.2 UI run cited page 7, an instruction to provide environment information; it did not contain the target values. |
| Equipment configuration | Source, page, and block provenance for a structured schedule lookup. | The M12 CLI run cited physical PDF page 10. Provenance was page-level and did not identify a table cell. |
| Unsupported information | Whether the Agent reports insufficient evidence without inventing a value. | The recorded M12 missing-field case looked up a candidate, selected no evidence ID, and finished with `insufficient_scope` and no citation. This does not establish document-wide absence. |

Live results may differ from these earlier observations. The M13.3 packaging
replay also encountered an HTTP 500 response from the structured local Ollama
action request after SEARCH; the run ended with `tool_error` and no final
citation. See the [M13.3 package report](m13-3-demo-release.md) for the
recorded details.

## Screenshots and privacy

The checked-in, sanitized presentation screenshots are in
[`docs/assets/`](assets/):

- `workspace-empty.png` shows the empty demo workspace.
- `example-action-trace.png` shows an action trace from the previously
  validated M13.2 browser run. It is cropped to omit the private document
  identity, source excerpt, evidence ID, and local path.

If preparing additional screenshots, save only reviewed, sanitized images in
`docs/assets/`. Do not include private PDFs, full task text, sensitive excerpts,
absolute local paths, raw model outputs, or credentials. Local browser
snapshots and research outputs belong in ignored directories, not the release
package.

## What the demo does and does not show

The demo shows a bounded local Agent using the existing retrieval session,
looking up observed evidence IDs, and rendering host-validated source
references in a report. Citations verify provenance, not relevance or
correctness. The system does not establish parser fidelity, complete document
coverage, engineering correctness, safety, standards compliance, diagnosis,
or recommendation quality. The Streamlit UI accepts one PDF, does not retain
durable history, and does not provide a live streamed Agent trace.

See the [V1 release summary](v1-release-summary.md) for the capability
boundary and possible future research directions.
