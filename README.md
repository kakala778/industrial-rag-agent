# Industrial RAG Agent

A local learning prototype for document retrieval and grounded generation. This
is an experimental project, not a production industrial system.

## Current status — M11 complete; M12 scope pending

M11 provides a local-first research demo for two to four explicitly named
documents. It runs bounded SEARCH / LOOKUP / CLARIFY / FINISH actions and writes
a private Markdown report with host-rendered evidence references and neutral
scope statuses. Local Qwen is the default; deterministic mode is for offline
mechanics checks, and the explicit DeepSeek reference mode discloses that task
and selected evidence text leave the machine. The report makes no engineering
or compliance verdict. See the
[M11 design and implementation plan](rag-agent/docs/superpowers/specs/2026-10-03-m11-evidence-research-agent-design.md)
and [Agent handoff](rag-agent/docs/agent-handoff.md). The current implementation
passed 345 unit tests and a synthetic three-document CLI smoke; live model
quality and industrial-PDF evidence relevance were not evaluated in that smoke.

M12 has not started and has no approved scope. The recommended kickoff is a
small, human-reviewed demo-readiness pilot using explicitly selected documents
and tasks. The existing architecture review's troubleshooting-agent candidate
still depends on a reliable operating-data source and must not be treated as
approved scope. See the M12 section in the handoff before choosing a target.

Run from `rag-agent/`:

```powershell
python -m src.agent_demo --task "查询设备额定压力" `
  --document A=design.pdf `
  --document B=manual.pdf `
  --document C=standard.pdf
```

The report is written under ignored `outputs/agent11/`. Citations identify
source locations; they do not establish relevance, support, correctness,
applicability or compliance. The older `--query/--scopes` invocation remains
available for the two-scope Agent 0 compatibility path.

## Previous decision — Agent 1.3 ROI review

Agent 1 is frozen for a bounded evidence-first demo. Agent 1.2 accepted 15/17
outputs and achieved 14/17 strict verdict accuracy, but accepted unit and
condition/applicability accuracy remain 8/15 and 7/15. The comparator is not
ready to support engineering decisions and remains evaluation-only. At that
decision point, the next recommended milestone was an end-to-end demo centered
on scoped evidence, host-rendered references, and honest incomplete results;
any semantic verdict must remain clearly experimental. See the
[Agent 1.3 ROI review](rag-agent/docs/agent1-3-semantic-comparator-roi-review.md)
and [Agent 1.2 results](rag-agent/docs/agent1-2-structured-output-validation.md).

RAG and Agent 0 remain frozen. Agent 1.2's 17-request pass had no API errors or
retries; raw outputs remain under ignored `rag-agent/outputs/agent1_2/`.

## Historical status — Agent 1.1 semantic contract validation

Agent 1.1 used the balanced 17-task benchmark reviewed against original PDF
pages: 5 `EQUIVALENT`, 6 `DIFFERENT`, and 6 `NOT_COMPARABLE`. Its single
schema-accepted output got the verdict right but marked known value and unit
dimensions `not_applicable`; 16 other outputs were recorded as `invalid_schema`.
See the [Agent 1.1 report](rag-agent/docs/agent1-1-balanced-semantic-benchmark.md).

The original Agent 1 result remains documented separately. Agent 1.1 did not
run retrieval, and the original v1 semantic comparator remains the default.

## Historical status — Agent 0.3 bounded evidence references

M6–M10.1 cover structured PDF processing, failure analysis, controlled visual
experiments, optional Hybrid retrieval and bounded parent-context reranking.
The RAG experiments are frozen. Agent 0 adds a local multi-document session,
scoped search/lookup tools, explicit state, strict actions and bounded execution.
It compares cited excerpts; engineering semantic equivalence is not established.
See the [design](rag-agent/docs/agent0-minimal-harness-design.md) and
[results and usage](rag-agent/docs/agent0-results.md). Qwen action selection is
experimental: synthetic tasks completed, but the real industrial smoke repeated
one evidence lookup and ended in clarification without covering both sources.

Agent 0.1 adds duplicate suppression and coverage-aware action eligibility.
In a fixed-observation comparison on seven new local industrial evidence tasks,
Qwen lookup-scope coverage rises 57.1%→100% and repeated actions fall 44→0,
but fully relevant task success stays 0/7. Full milestone acceptance is unmet;
quotes and applicability still fail. See the
[Agent 0.1 results and limits](rag-agent/docs/agent0-1-progress-coverage-results.md).

Agent 0.2 compares local Qwen with optional DeepSeek Flash on the same frozen
observations. On five candidate-available tasks, final ID selection improves
1/5→5/5 and fully relevant copied-quote success 0/5→4/5. Two other tasks remain
retrieval-bound. Flash is an opt-in experimental selector; source-condition
omission and citation-contract limits remain. See the
[Agent 0.2 protocol, cost and results](rag-agent/docs/agent0-2-evidence-selection-model-comparison.md).

Agent 0.3 adds an opt-in evidence-ID contract: the host validates selected IDs
and renders bounded citations from its active parsed-document registry. On the
same five candidate-available tasks, DeepSeek retains 5/5 ID-selection success;
fully relevant task success rises to 5/5, and applicable condition alignment
to 7/7. Two tasks remain retrieval-bound. Citations are authentic to the current
parsed cache, while relevance labels remain assistant-reviewed and
**not independently human-verified**. Keep this contract experimental and
confirm the local review sheet before semantic comparison. See the
[Agent 0.3 protocol, results and limits](rag-agent/docs/agent0-3-bounded-evidence-contract.md).

Experiment checkpoint: `2918f99` on `codex/m9-hybrid-retrieval`.
`v0.5-local-rag-generation` remains the historical M1–M5 checkpoint.

## Available capabilities

| Capability | Current application status |
|---|---|
| Markdown / PDF RAG | Local Ollama `qwen3:4b`, retrieved-source citations |
| PDF parsing | PyMuPDF default; MinerU Advanced/OCR structured optional |
| Retrieval | Dense default; fixed BM25 + RRF Hybrid optional in PDF CLI |
| BGE reranking | Optional Dense/Hybrid Top20 → Top3; off by default |
| Visual enrichment / parent context | Offline experiments; not adopted in application |
| Agent | M11 evidence research demo; local Qwen default, deterministic offline mode, explicit remote DeepSeek reference mode |
| Vector database / Web API / durable execution | Not implemented |

Dense cosine, RRF and BGE scores have different meanings and scales. Hybrid
ranking diagnostics are not displayed as ordinary citations.

## What the experiments establish

These are retrieval measurements, not generated-answer accuracy. The independent
questions use the same eight-document corpus; they are not external-corpus validation.

| Frozen cohort | Dense+BGE Top3 page | Hybrid+BGE Top3 page | Dense+BGE normalized evidence | Hybrid+BGE normalized evidence |
|---|---:|---:|---:|---:|
| Original 32 questions | 19/32 | 23/32 | 14/32 | 17/32 |
| Independent 21 questions | 17/21 | 17/21 | 12/21 | 15/21 |

M10.1 changed only BGE input context over frozen Hybrid20 candidates. Q25
improved rank7→1 and I17 rank4→1, but Q04 regressed rank1→4. Original evidence
stayed 17/32; independent evidence increased 15/21→16/21. Top1 page decreased
in both cohorts. Mean input tokens rose 48.7% and CPU reranking time rose 73.5%.
**Parent expansion remains offline.** Returned child text was not expanded.
See the [full experiment and limits](rag-agent/docs/m10-parent-context-controlled-experiment.md).

The decision is **ENTER AGENT**, with OCR/image/layout losses, ambiguous scope
and claim-level citation validation carried forward as explicit limitations.

## Repository layout

- [rag-agent/](rag-agent/README.md): application, evaluation, tests and reports.
- [industrial-preprocessor](rag-agent/components/industrial-preprocessor/README.md):
  standalone curated preprocessing component; not imported by the RAG runtime.
- `minerU/mineru-405-poc/run-mineru.ps1`: portable local MinerU launcher.
  Runtime, models, industrial inputs and parser caches stay local.

## Run

Prepare your own permitted inputs and local Ollama model. Run from the application directory:

```powershell
cd rag-agent
python -m pip install -r requirements.txt
python src/rag_demo.py
python src/pdf_rag_demo.py "<local PDF>"
```

Opt-in structured Hybrid+BGE PDF path:

```powershell
python src/pdf_rag_demo.py "<local PDF>" --parser mineru --retriever hybrid --rerank
```

A fresh clone needs its own configured MinerU runtime or `MINERU_RUNNER_PATH`.
Legacy `--mode dense|reranker|compare` remains supported; mixing `--mode` with
new retrieval selectors is an error. See the [application README](rag-agent/README.md).

## Reports and next-stage preparation

| Stage | Report |
|---|---|
| M1–M5 | [Local baseline](rag-agent/docs/m1-m5-summary.md) |
| M6 | [Industrial PDF evaluation](rag-agent/docs/m6-industrial-pdf-evaluation.md), [representation audit](rag-agent/docs/m6-document-representation-audit.md) |
| M7 | [Reranker failure analysis](rag-agent/docs/m7-reranker-failure-analysis.md) |
| M8.1–M8.2 | [Failure audit](rag-agent/docs/m8-document-intelligence-failure-audit.md), [representation comparison](rag-agent/docs/m8-visual-representation-experiment.md) |
| M8.3–M8.4 | [Whole-page vision](rag-agent/docs/m8-vision-model-feasibility-test.md), [block experiment](rag-agent/docs/m8-vision-block-experiment.md) |
| M8.5–M8.6 | [Focused OCR](rag-agent/docs/m8-focused-ocr-comparison.md), [gated retrieval A/B](rag-agent/docs/m8-gated-vision-retrieval-ab.md) |
| M9.1–M9.3 | [Hybrid experiment](rag-agent/docs/m9-hybrid-retrieval-experiment.md), [independent validation](rag-agent/docs/m9-independent-validation-and-ranking-audit.md), [optional integration](rag-agent/docs/m9-optional-hybrid-integration.md) |
| M10 | [Architecture review](rag-agent/docs/m10-architecture-review-and-agent-readiness.md), [parent experiment](rag-agent/docs/m10-parent-context-controlled-experiment.md) |
| Pre-Agent | [Historical handoff and acceptance boundary](rag-agent/docs/agent-handoff.md) |
| Agent 0 | [Design](rag-agent/docs/agent0-minimal-harness-design.md), [results and usage](rag-agent/docs/agent0-results.md) |
| Agent 0.1 | [Progress, coverage and evidence relevance](rag-agent/docs/agent0-1-progress-coverage-results.md) |
| Agent 0.2 | [Evidence selection and DeepSeek comparison](rag-agent/docs/agent0-2-evidence-selection-model-comparison.md) |
| Agent 0.3 | [Bounded evidence reference contract](rag-agent/docs/agent0-3-bounded-evidence-contract.md) |
| Agent 1.2–1.3 | [Structured output validation](rag-agent/docs/agent1-2-structured-output-validation.md), [semantic comparator ROI review](rag-agent/docs/agent1-3-semantic-comparator-roi-review.md) |
| M11 | [Evidence Research Agent design and implementation plan](rag-agent/docs/superpowers/specs/2026-10-03-m11-evidence-research-agent-design.md), [Agent handoff](rag-agent/docs/agent-handoff.md) |
| M12 | Kickoff scope pending; see the [Agent handoff](rag-agent/docs/agent-handoff.md) for the current proposal and constraints |

## Verification and private data

Recommended checks from `rag-agent/`:

```powershell
python -m compileall src evaluation
python -m unittest discover -s tests
git diff --check
```

The M10.1 report records 143 tests at its checkpoint. Agent 0 verification ran
180 tests (37 new), plus a frozen 10-case synthetic benchmark and separate local
industrial smoke. These results do not establish industrial semantic accuracy.

Do not commit industrial PDFs, private QA, page/crop images, raw model answers,
MinerU caches, models, credentials or local experiment outputs. Reproducing the
industrial experiments requires the matching private frozen inputs; a fresh
clone alone is insufficient. Do not substitute synthetic results for missing data.
