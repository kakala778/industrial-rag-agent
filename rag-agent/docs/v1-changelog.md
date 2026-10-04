# Industrial RAG Agent V1 Changelog

## Version

`v1.0-demo`

## Release goal

This release is **a local learning and demonstration version of an
evidence-first industrial document research agent**. It packages the existing
RAG and bounded Agent as a reviewer-facing local workflow; it does not claim
industrial decision readiness.

## Added

- A bounded Agent loop with SEARCH / LOOKUP / CLARIFY / FINISH actions.
- An evidence lookup workflow using IDs returned by the active search session.
- Host-side validation and rendering of source, page, block, excerpt, and
  citation metadata.
- A local Streamlit workspace for one-PDF research runs.
- Markdown research reports with neutral scope outcomes and limitations.
- A local PDF research workflow using the existing PyMuPDF loader and Agent
  session. Agent SEARCH uses its existing Hybrid retrieval and BGE reranking
  defaults; the standalone PDF RAG CLI retains its Dense default.

## Changed

- Organized final V1 release, setup, and reviewer guidance in dedicated
  documents.
- Packaged the demo with sanitized presentation screenshots and ignored local
  browser snapshots.
- Clarified the distinction between the RAG CLI defaults and Agent search
  defaults, and documented what citations and terminal statuses do and do not
  establish.
- Removed a machine-specific audit path from the offline rescore helper; its
  scoring logic and frozen benchmark data are unchanged.

## Not included

- Production deployment or production reliability guarantees.
- Autonomous engineering decisions or recommendations.
- Semantic compliance judgement.
- Guaranteed retrieval accuracy or complete document coverage.
- Enterprise accounts, permissions, multi-user history, or integrations.

## Verification

Release-closure checks on 2026-10-04:

- `python -m compileall src evaluation app` — passed.
- `python -m unittest discover -s tests` — passed, 353 tests.
- `git diff --check` — passed.
- Relative links and images in the release Markdown were checked. The audit
  found no PDF/model binaries, real credentials, actual workstation paths, or
  generated reports/caches among tracked and pending release files. Two
  synthetic Windows drive-root values remain only in tests that verify path
  redaction. The two release PNGs were visually reviewed; they show only the
  empty workspace and a sanitized action timeline.

The separate real Ollama observations, including a successful M13.2 UI run
with a semantically weak evidence selection and the M13.3 HTTP 500 replay, are
recorded in the [M13.3 package report](m13-3-demo-release.md). They are not
recast as retrieval accuracy results.
