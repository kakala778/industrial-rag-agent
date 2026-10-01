# M8 Document Intelligence Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to carry out any implementation task-by-task. This document prepares the experiment; it does not authorize an implementation before the evidence-selection gate below.

**Goal:** Determine whether a targeted document-representation improvement reduces the M6 evidence misses classified as parsing failures.

**Architecture:** Freeze the M7.2 evaluation as the comparison baseline, then inspect the 11 confirmed parsing cases against local PDF pages and MinerU structured output. Select one evidence-supported representation change and compare it on the same questions while keeping retrieval, reranking, embeddings, chunking, and prompts fixed.

**Tech Stack:** Existing Python PDF evaluation harness, local MinerU 4.0.5 Advanced/OCR `middle_json` cache, existing evidence matcher, and ignored local experiment outputs.

**Spec:** [`docs/m7-reranker-failure-analysis.md`](../../m7-reranker-failure-analysis.md)

## Global Constraints

- Use the same 32 answerable M6 questions, evidence matcher, PDF set, and M7.2 baseline metrics for paired comparisons.
- Keep PDFs, QA text, answers, document text, MinerU output, and detailed per-question analysis in Git-ignored local paths.
- Public reports may contain only anonymous case IDs, aggregate counts, and allowlisted source/page/block metadata.
- Change only the document representation or parsing path selected by the evidence gate; keep retrieval, reranker, embedding, chunking, query handling, and Qwen prompt fixed.
- Do not assume a table, vision, OCR, layout, or cross-page fix until local evidence identifies that failure mechanism.
- A supported result may be that no single intervention is justified yet; record that outcome without fabricating an improvement.

## Review Focus

- Evidence visible on the original page but absent from MinerU output: record the exact anonymous page/block trace before selecting a parser or vision change.
- Table content extracted without row, column, or header relationships: distinguish extraction from relationship loss.
- Text present in structured output but omitted or fragmented in the indexed representation: separate document representation from retrieval ranking.
- Evidence spanning pages or layout regions: do not label it a parsing failure unless the existing evidence criteria confirm it.
- Unclear or conflicting source evidence: keep it unresolved and out of confirmed parsing counts.

## Files and Output Responsibilities

- `rag-agent/docs/superpowers/plans/2026-10-01-m8-document-intelligence.md` — this experiment plan.
- `rag-agent/outputs/m8_failure_cause_analysis.json` — ignored local case-level diagnosis, if needed.
- `rag-agent/outputs/m8_document_intelligence_compare.json` — ignored local paired run, if an intervention is justified.
- `rag-agent/docs/m8-document-intelligence-evaluation.md` — anonymized public findings after the experiment runs.
- Any source or evaluation code path must be selected and recorded after Task 2 identifies a concrete mechanism; no implementation file is preselected.

## Experiment Tasks

### Task 1: Freeze and reproduce the comparison baseline

- [ ] Confirm local M6 QA, PDF, and MinerU cache files are present and match their recorded hashes; do not copy them into the repository.
- [ ] Run the existing compare evaluation on the 32 answerable questions using the M7.2 settings and evidence matcher.
- [ ] Save only local/ignored detailed output; check that the reproduced baseline remains Top-3 page hit `19/32`, normalized evidence `14/32`, and the M7.2 case counts remain `4/3/4/11/3` across the five reported classes.
- [ ] Stop and report insufficient data if the original inputs or structured cache cannot be verified.

### Task 2: Diagnose the confirmed parsing cases

- [ ] For each of the 11 anonymous `PARSING_FAILURE` cases, inspect the cited original PDF page and corresponding MinerU structured output locally.
- [ ] Record one evidence-backed mechanism per case: OCR/text absent, table relationship loss, visual/image content, reading order/layout loss, cross-page evidence, or unresolved. These are diagnostic tags, not replacements for M7.2 failure classes.
- [ ] Save detailed notes only under an ignored local output path; publish only counts by mechanism and anonymous case references.
- [ ] Verify every confirmed tag has a trace to the original page and structured block; keep unresolved cases out of confirmed mechanism totals.

### Task 3: Apply the intervention selection gate

- [ ] Select one mechanism only if Task 2 shows a repeated, actionable pattern with source-page evidence.
- [ ] Write the hypothesis, the single representation change, the affected file/interface, and the expected metric before implementation.
- [ ] If causes are mixed or evidence is insufficient, do not change code; publish the diagnosis and leave M8 implementation unstarted.

### Task 4: Run one paired representation experiment

- [ ] Reuse the same 32 questions, local corpus, evidence matcher, and M7.2 baseline; change only the selected document representation or parsing path.
- [ ] Record Top-3 page hit, normalized evidence hit, per-class transition counts, and regressions against the baseline.
- [ ] Verify that retrieval, reranker, embeddings, chunking, query text, and generation prompt are unchanged.
- [ ] Keep raw QA, page content, parser output, and per-question evidence private and ignored by Git.

### Task 5: Publish findings and choose the next direction

- [ ] Create `rag-agent/docs/m8-document-intelligence-evaluation.md` with scope, paired metrics, anonymous case summaries, limitations, and reproducibility details.
- [ ] Recommend further Document Intelligence work only if the selected representation change has evidence-backed gains without hiding regressions.
- [ ] Reconsider Hybrid Search only if retrieval failures become the dominant confirmed class; reconsider reranker tuning only if ranking failures dominate after representation issues are measured.
- [ ] Mark M8 complete only when the paired evaluation and privacy checks are complete; otherwise state exactly which inputs or evidence are missing.
