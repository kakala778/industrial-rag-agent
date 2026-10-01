# M7.2 — Reranker Failure Analysis

## Experiment Summary

**Baseline:** dense retrieval, using the existing embedding and cosine ranking.

**Experiment:** dense Top-20 candidates → `BAAI/bge-reranker-v2-m3` → Top-3.
Retrieval, reranking, embeddings, chunking, MinerU, and prompt behavior are not
changed by this analysis.

| Metric | M6 dense baseline | M7.1 dense + reranker | Change |
| --- | ---: | ---: | ---: |
| Top-3 page hit | 15/32 | 19/32 | +4/32 |
| Normalized evidence | 10/32 | 14/32 | +4/32 |

The recorded M6 failure labels were PARSING 11, RANKING 10, and RETRIEVAL 4.
The M7.1 aggregate report says PARSING remained 11 and RANKING fell to 3; it
also reports that 4 of the 10 M6 RANKING cases moved into evidence Top-3. The
aggregate summary alone did not account for the full change. The case-level
replay below identifies 4 fixed, 3 still-ranking, and 3 unassigned RANKING
cases, reconciling the original 10 without inferring cases from totals.

## Case-Level Results

The replay used the local M6 QA file, all 8 PDFs under its `input/` directory,
and the matching MinerU structured cache. All 32 source-scoped QA rows matched
the PDFs. The file contained 35 rows: 32 scored questions and 3 unanswerable
questions excluded from retrieval scoring. Every PDF's cache manifest and
Middle JSON matched its content hash (8/8); the cache yielded 1,594 structured
documents spanning 659 source-page pairs.

The 25 original scored failures were classified as follows. Counts cover only
those 25 original failures; the 7 baseline successes and 3 unscored questions
remain in the anonymous per-question JSON but are excluded from this table.

| Category | Count |
| --- | ---: |
| FIXED_BY_RERANKER | 4 |
| STILL_RANKING_FAILURE | 3 |
| RETRIEVAL_FAILURE | 4 |
| PARSING_FAILURE | 11 |
| INSUFFICIENT_DATA | 3 |
| **Total original failures** | **25** |

The case-level results reconcile the M6 labels:

| M6 baseline label | M7.2 classification |
| --- | --- |
| PARSING (11) | PARSING_FAILURE (11) |
| RANKING (10) | FIXED_BY_RERANKER (4), STILL_RANKING_FAILURE (3), INSUFFICIENT_DATA (3) |
| RETRIEVAL (4) | RETRIEVAL_FAILURE (4) |

The 3 RANKING rows assigned INSUFFICIENT_DATA had normalized evidence at dense
ranks 2 and reranker ranks 1, so they do not meet the specified fixed or still
ranking definitions. Their original M6 RANKING label conflicts with the shared
evidence matcher. They are left unassigned rather than reinterpreted. Separately,
the 3 unanswerable rows are marked unscored and do not contribute to failure
counts.

Evidence ranks reuse the evaluation's existing normalized evidence matcher.
FIXED_BY_RERANKER requires evidence outside dense Top-3 but inside dense
Top-20 and reranker Top-3. STILL_RANKING_FAILURE requires evidence in dense
Top-20 but outside reranker Top-3. RETRIEVAL_FAILURE means evidence is absent
from dense Top-20. PARSING_FAILURE is assigned only when the evaluation finds
expected evidence missing from the MinerU structured documents and the run
metadata confirms `parser=mineru` and `representation=structured`.

### Anonymous Examples

| Question ID | Dense evidence rank | Reranker evidence rank | Classification |
| ---: | ---: | ---: | --- |
| 2 | 4 | 1 | FIXED_BY_RERANKER |
| 7 | 4 | 5 | STILL_RANKING_FAILURE |
| 9 | Not in dense Top-20 | Not present | PARSING_FAILURE |

No question text, answer, evidence keywords, document text, or original source
name is included in these examples. The generated per-question JSON is local at
`outputs/m7_failure_analysis.json`, which is Git-ignored. It contains candidate
count, dense Top-20 and reranked Top-3 ranks/scores, rank movement, and
allowlisted source/page/block metadata. Source names are replaced with run-local
IDs; arbitrary metadata is not serialized.

## Reproduce With Local Evaluation Inputs

Reproduce with the original private evaluation files from `rag-agent/`:

```bash
python evaluation/evaluate_pdf_retrieval.py \
  --mode compare \
  --parser mineru \
  --representation structured \
  --pdf-dir path/to/industrial-rag-data/input \
  --dataset path/to/industrial-rag-data/qa/m6_final_qa.local.json \
  --failure-analysis-output outputs/m7_failure_analysis.json
```

The JSON stays local and ignored by Git. Its `question_id` is the 1-based row
number within the run; source aliases are local to that output. The unit tests
use synthetic fixtures for logic only; all counts and examples above come from
the real M6 inputs.

## Next-Stage Recommendation

PARSING_FAILURE is the largest remaining class at 11/25 (44%), and those rows
were confirmed against the original MinerU structured representation. This
supports prioritizing a Document Intelligence experiment, especially table and
visual representation, before Hybrid Search. RETRIEVAL_FAILURE accounts for
4/25 (16%), while STILL_RANKING_FAILURE accounts for 3/25 (12%); the current
results do not justify a broad Hybrid Search or reranker retuning experiment
ahead of the parsing work. Keep the reranker optional: it promoted 4 evidence
cases into Top-3, but 3 ranking-labelled cases remain unassigned under the
specified evidence rules.
