# M9.2 — Independent Validation + Ranking Audit

## 1. Purpose

Test whether the fixed M9.1 lexical/dense complementarity persists on new PDF-grounded questions, and audit four historical candidate-recall successes that still fail the formal Top-3 matcher. M8 is frozen; no default RAG integration, model change, rewrite or database is included.

## 2. Independent Dataset

21 new questions cover all eight existing PDFs and eight categories. Three spread pages per PDF were selected before new-query retrieval. None is a page used for M6 ground truth. Original PDF pages were rendered locally and visually reviewed before QA authoring; native text or MinerU output alone was not treated as ground truth. The selected blank table, extremely wide plan and two repeated route sheets were excluded for lack of confidently distinct facts. The final set covers 20 unique source/page pairs; two separate table facts share one page.

| Category | Questions |
|---|---:|
| text | 2 |
| numeric | 1 |
| unit | 2 |
| model / identifier | 6 |
| table | 3 |
| ocr | 1 |
| multi_fact | 3 |
| drawing_layout | 3 |

Anonymous source counts: S001–S004 each 3; S005–S006 each 2; S007 3; S008 2. Lexical-sensitive questions coexist with ordinary semantic and relational controls. Authoring did not use retrieval rankings or pick winning queries. Original M6 QA was read only to avoid reused facts and queries; the runner rejects copied queries and old source/page pairs.

QA was frozen before model inference with SHA-256:

```text
8ca279513946a7bc7ed50c51e784c672e2b1d85f8f85adade2b77e36ef4116e2
```

The local manifest pins QA count/category/source distribution, PDF/cache inventory, retrieval configuration and nine protected implementation files. It refuses overwrite and requires exact pre/post equality. Private QA, PDF renders, selection notes and detailed audit candidates stay in ignored output directories.

This is independent-question validation on the same corpus, authored by the experimenter, not a random sample, a new-document benchmark or external blind evaluation. Drawing/identifier coverage is deliberately broader than M9.1. Small category counts limit subgroup conclusions.

## 3. Frozen Retrieval Configuration

All five arms reuse the M9.1 implementations: same 1,594 structured Documents, 2,996 chunks, 500-character chunks with overlap 80, multilingual MiniLM embedding, cosine similarity and BGE v2-m3. BM25 k1=1.2/b=0.75 and the identifier/Chinese unigram-bigram tokenizer remain unchanged. Dense20 + BM25 20 → deduplicated equal-weight RRF k=60 → Hybrid20 → existing BGE → formal Top3. No parameter search, MinerU parsing, OCR, VLM, generation, new dependency or model download is performed.

All seven formal metrics reuse the original evaluator. Recall@20 is the same normalized, source/page-scoped cumulative evidence test on the candidate pool. Independent I17 has no relationship to M6 Q17: the historical uncertainty/exclusion rule is not transferred to the new dataset. Formal I01–I21 are scored as frozen.

## 4. Independent Metrics

The formal run completed successfully; the post-run QA/cache/core manifest exactly matches its pre-inference freeze. All 21 frozen rows are scored.

| Mode | Top1 source | Top3 source | Top1 page | Top3 page | Normalized evidence | Strict/raw evidence | Recall@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense | 14/21 | 19/21 | 8/21 | 14/21 | 10/21 | 9/21 | 12/21 |
| BM25 | 21/21 | 21/21 | 13/21 | 18/21 | 14/21 | 13/21 | 16/21 |
| Hybrid | 18/21 | 21/21 | 13/21 | 18/21 | 11/21 | 10/21 | 16/21 |
| Dense+BGE | 17/21 | 19/21 | 13/21 | 17/21 | 12/21 | 11/21 | 12/21 |
| Hybrid+BGE | 20/21 | 21/21 | 16/21 | 17/21 | 15/21 | 14/21 | 16/21 |

The original matcher finds evidence in unchanged chunks for 16/21 questions. Within that subset, Dense Recall@20 is 12/16; BM25/Hybrid are 16/16. I05/I09/I10/I11/I15 have no complete expected-keyword match in the scoped chunks. They are matcher-unavailable cases, not automatically five confirmed parsing defects. No QA or matcher repair was performed.

## 5. Comparison with M9.1

| Paired improvement: Dense+BGE → Hybrid+BGE | M9.1 reused questions | M9.2 independent questions |
|---|---|---|
| Top3 page | 19/32 → 23/32 (+4) | 17/21 → 17/21 (0 net) |
| Normalized evidence | 14/32 → 17/32 (+3) | 12/21 → 15/21 (+3) |
| Recall@20 | 17/32 → 21/32 (+4) | 12/21 → 16/21 (+4) |
| Evidence success regressions | 0 | 0 |
| Top3 page success regressions | 0 | 2 |

Independent evidence and recall gains persist; the Top3 page net gain does not. Report these separately rather than declaring all M9.1 benefits generalized. Evidence recovery I13/I14/I16 includes unit, multi-fact and diagram-layout categories; I16 has a query identifier feature, but three examples do not establish a dominant causal tokenizer mechanism. I17 is the fourth dense candidate miss recovered to the pool, but remains BGE rank4 and final failure.

## 6. Dense/BM25 Complementarity

Success here means formal normalized evidence in Top3 before reranking.

| Dense | BM25 | Questions | Hybrid Top3 successes |
|---|---|---:|---:|
| Success | Fail | 0 | 0 |
| Fail | Success | 4 | 1 |
| Success | Success | 10 | 10 |
| Fail | Fail | 7 | 0 |

RRF protects all ten Dense evidence successes and keeps one of four lexical-only Top3 successes (I21). BM25-only successes I13/I14/I17 become Hybrid ranks5/4/6; this is a pre-reranker precision loss despite retaining evidence in Top20. BGE restores I13/I14 and separately restores the lower lexical candidate I16; I17 remains rank4. No Dense-only-success row exists in this independent set, so protection of that specific group cannot be confirmed anew; M9.1 had three such cases. BM25 alone outperforming raw Hybrid (14 vs11 evidence hits) is an observed fusion ranking limitation, not a reason to hide that arm.

## 7. Regression Analysis

| Transition | Evidence gains | Evidence regressions | Top3 page gains | Top3 page regressions |
|---|---|---|---|---|
| Dense+BGE → Hybrid+BGE | I13/I14/I16 | None (all 12 retained) | I05/I14 | I10/I11 |
| Hybrid → Hybrid+BGE | I07/I13/I14/I16 | None (all 11 retained) | I14 | I10/I11 |
| Dense → Hybrid | I21 | None (all 10 retained) | I05/I11/I13/I21 | None |

Joint page+evidence success has the same transitions as evidence above. The two page regressions are 2/21 evaluated questions (2/17 baseline page successes); they must not be folded into a claim of zero regression. I10/I11 are both model-category questions whose target keywords are unavailable in the scoped chunks. Their page-only successes are lost after Hybrid candidate reranking, while complete evidence was already missing before. This documents a citation/localization regression without guessing a representation fix or reopening M8. Pure BM25 loses no Dense evidence success on this set, unlike its three losses in M9.1.

## 8. Ranking Audit

Q denotes the original M6/M9.1 row; I denotes an independent row. Historical rankings were reconstructed from stored corpus ordinals after verifying corpus fingerprints, protected-file hashes, chunk count and every scoped evidence rank. No historical model inference or alternative scorer was used. Human classifications describe observable candidate/PDF evidence, not an explanation of BGE internals.

| Case | Dense | BM25 | RRF | BGE after Hybrid | Primary audit category |
|---|---:|---:|---:|---:|---|
| Q07 | 4 | — | 7 | 12 | OTHER: matcher/representation limitation |
| Q19 | 5 | 1 | 3 | 5 | QUERY_AMBIGUITY |
| Q24 | — | 6 | 12 | 13 | QUERY_AMBIGUITY |
| Q25 | 7 | 11 | 14 | 7 | TABLE_SERIALIZATION: chunk context limitation |

Primary counts: OTHER 1, QUERY_AMBIGUITY 2, TABLE_SERIALIZATION 1. Tags overlap and are not additional cases. The original formal scores remain unchanged; audit classifications do not rewrite ground truth.

### Q07

Observed: BGE Top1 is on the correct source/page and contains the target mathematical value in LaTeX notation. The original string matcher does not recognize it as equivalent to the Unicode form used in expected evidence; its accepted image-text block is only rank12. This is not a clean semantic ranking failure. Same-page prose/image competition and adjacent chunks exist, but the retrieved Top1 already contains the answer in an equivalent form. An independent reviewer verified the original PDF and the representation equivalence. No matcher normalization was added to formal evaluation.

Direction: retain a documented scorer limitation and later review representation-aware evaluation separately. A new reranker cannot be justified from this case alone.

### Q19

Observed: the query does not specify the component or drawing scope. Several pages have almost identical short paragraphs but genuinely different field requirements. BGE ranks an exact repeated pair first and second and another related field third; the scoped target drops 3→5. The correct target and competitors are all 55-character chunks, so target shortness alone does not explain the ranking. One third-ranked candidate contains the expected field value on a different page, yet still fails the frozen source/page rule.

Primary QUERY_AMBIGUITY; secondary SIMILAR_FIELD and DUPLICATE_CANDIDATES. RERANKER_MISORDER is an observed regression under the frozen scope, not proof of a specific model reasoning defect. Original target GT is PDF-supported; multiple plausible answers arise from an underspecified query.

Direction: review question scope and duplicate competition before replacing BGE. No QA edits, rewrite or dedup were made.

### Q24

Observed: the cover GT is clearly present in the original PDF, but the query does not identify the document. BGE puts a different document's publication block first and the correct document's imprint block second; the required cover block is rank13. The imprint has relevant publication information but violates the cover-page scope. Its text is more explicit than the cover's isolated short field. All BGE scores in this case are very low; they are ranking outputs, not calibrated correctness probabilities.

Primary QUERY_AMBIGUITY; secondary SEMANTIC_CONFUSION and SHORT_EVIDENCE. Source/page misorder is observable; internal reasons remain uncertain. GT_UNCERTAIN is not assigned because the original cover supports the frozen answer.

Direction: investigate scoped-query expectations and context sparsity in a later controlled task, without claiming a larger reranker will resolve missing query context.

### Q25

Observed: the original PDF and raw MinerU table retain the requested full row identifier. The complete row is in one chunk: BM25 rank11, RRF union rank23, then removed by the fixed Hybrid20 cutoff. Its adjacent overlapping chunk begins inside the identifier and contains the formula plus other similar rows. That truncated-context chunk passes the existing expected-keyword matcher and is the formal Hybrid rank14/BGE rank7 evidence. Competing tables share formula/field wording. No structural duplicate occurs.

Primary TABLE_SERIALIZATION (flat table chunk context); secondary SIMILAR_FIELD. This is not confirmed OCR corruption or incorrect GT. The observed BGE misorder is entangled with incomplete identifier context and fusion truncation. Formal Recall@20 counts the accepted fragment, not necessarily a complete identifier-associated row.

Direction: future row-context/evaluation diagnostics may be more informative than blindly comparing models. Chunking, serialization, RRF and matcher remain frozen here.

### Is BGE the main bottleneck?

These four cases do not support that conclusion: one formal failure is a notation-matching limitation, two queries lack disambiguating scope, and one involves table context plus candidate truncation. BGE misorder exists under the formal scorer, especially Q19, but a controlled reranker comparison is not yet the strongest first explanation. Multi-fact difficulty is not established as the main cause from these four audits.

## 9. Duplicate / Candidate Diagnostics

| Historical Hybrid20 case | Structural duplicate excess | Exact text pairs | Near text pairs | Same-block pairs |
|---|---:|---:|---:|---:|
| Q07 | 0 | 0 | 1 | 5 |
| Q19 | 0 | 1 | 5 | 2 |
| Q24 | 0 | 0 | 1 | 0 |
| Q25 | 0 | 0 | 0 | 0 |

Exact text is NFKC/casefold/whitespace-normalized equality. Near text uses SequenceMatcher ratio ≥0.90, excluding exact pairs. Counts are pairs, not unique chunks or global corpus duplicate rates. Same-block pairs can be legitimate different overlapping chunks, not duplicate identities. Reranking only permutes these pools.

Q19's exact duplicate is a legitimately repeated requirement on separate drawing pages and occupies two final leading slots. Its near matches include meaningful field differences; merging them as duplicates would risk losing distinct evidence. Q07/Q24 near pairs alone do not establish semantic equivalence or a causal Top3 problem. Duplicate competition is material in Q19, not a demonstrated corpus-wide bottleneck. No dedup optimization was applied.

## 10. Runtime

One local CPU run; averages are descriptive, not production or repeated benchmarks.

| Stage | M9.1 mean / setup | M9.2 mean / setup |
|---|---:|---:|
| BM25 build | 0.298 s | 0.937 s |
| Embedding load + corpus index | 150.436 s | 135.720 s |
| BGE load | 8.970 s | 8.733 s |
| Shared query embedding | 58.98 ms | 47.06 ms |
| Dense search | 10.74 ms | 9.03 ms |
| BM25 search | 17.73 ms | 13.46 ms |
| RRF | 0.50 ms | 0.47 ms |
| Hybrid search without embedding | 28.97 ms | 22.96 ms |
| Dense candidate BGE | 33.872 s | 33.182 s |
| Hybrid candidate BGE | 33.552 s | 34.232 s |

The added lexical+fusion mean is about 13.93 ms/query. The cost relationship is consistent: BM25/RRF are small compared with BGE CPU scoring. Hybrid search timing excludes embedding, reranker and generation. BGE scores the same maximum 20 candidates in each arm; pool content and token lengths vary. No speed improvement is claimed from small differences between means.

## 11. Decision

**YES — proceed to an explicit optional Hybrid backend integration, with the page-regression limitation recorded.** Independent recall improves by four, final evidence by three, every Dense/BGE evidence success survives, and the added search cost is small. This supports retaining the fixed strategy as an option; it does not support replacing Dense, silently enabling Hybrid by default, or claiming universal page-hit improvement.

The next implementation should expose the already frozen strategy explicitly and preserve Dense as a selectable baseline, with regression checks for evidence and citation/page localization. Do not tune BM25/RRF on either cohort. A more independent, preferably independently authored dataset would strengthen the limited same-corpus generalization evidence.

A reranker comparison is not the first priority from the four historical audits: notation matching, missing query scope, legitimate duplicates and complete row-context loss make attribution confounded. The new independent set also has one evidence-available final miss, I17 (BGE rank4), which merits a separate local row/context audit before any model replacement. Future narrowly controlled evaluation/row-context work is more justified than assuming a larger BGE alternative solves all observed failures. Query rewrite/decomposition remains unimplemented, and no dominant multi-fact failure pattern has been established.

Qdrant remains deferred: 2,996 in-memory chunks do not create a current persistence/incremental-index/filtering/scale requirement. M8 stays frozen. No new OCR/VLM or database work is proposed here.

## Reproduction and Validation

From rag-agent/ with original local inputs, existing caches, historical M9.1 anonymous summary and newly reviewed local QA:

```powershell
python evaluation/run_m9_independent_validation.py --freeze
python evaluation/run_m9_independent_validation.py
```

Freeze is a one-time command and refuses an existing manifest. For subsequent replay run the second command; --audit-only reconstructs the four historical cases without loading retrieval models. Dataset/manifest/summary paths can be supplied explicitly, but no retrieval parameter is exposed.

Detailed independent results stay beneath outputs/m9_independent_validation/; audit diagnostics, original-page images and private candidate text stay beneath outputs/m9_ranking_audit/. Only anonymous observations and aggregate counts appear in Git. Protected M9.1 implementation files and the default application flow are unchanged.

Completed local run:

```text
python evaluation/run_m9_independent_validation.py --freeze   PASS (before inference)
python evaluation/run_m9_independent_validation.py            PASS (21 questions, five arms)
python evaluation/run_m9_independent_validation.py --audit-only PASS (no models)
python -m compileall src evaluation                          PASS
python -m unittest discover -s tests                         121 tests, PASS
git diff --check                                            PASS
```

The manifest was written at 2026-10-02 11:04:13 UTC; the formal run directory was created at 11:04:27 UTC. Post-run and fresh final rechecks match the original frozen QA/cache/core manifest. MinerU call count is zero. New runner/freeze/schema/privacy tests account for seven of the 121 tests; the existing 114 baseline tests remain passing. Independent read-only reviews checked implementation, original PDF evidence for audit cases, anonymous metrics, transitions, runtime and report claims.

Final deliverable/anonymous-JSON scans find no full private question, PDF name, answer or evidence-string matches of at least 12 characters; PDF filenames were additionally checked regardless of length. Privacy relies on allowlisted fields and tests as well as this string scan. New untracked files were separately checked for trailing whitespace. Git tracks no real PDF, local QA, parser cache, model weight or experiment output, and none of the nine protected M9.1 files changed.

This milestone remains local and uncommitted on codex/m9-hybrid-retrieval pending user review. No staging, commit, push, merge or tag was performed.
