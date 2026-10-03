# Industrial RAG Agent

A local learning prototype for document retrieval and grounded generation. This
is an experimental project, not a production industrial system.

## Current status — Agent 0.2 evidence selection compared

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
| Agent | Deterministic default; optional local Qwen or paid DeepSeek Flash selector |
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
