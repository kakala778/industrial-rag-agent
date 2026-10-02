# M9.3 Optional Hybrid Retrieval Integration Design

## Goal

Expose the frozen M9.1/M9.2 Dense + BM25 + RRF strategy as an explicit option
in the PDF RAG demo. Keep Dense without reranking as the unchanged default and
keep the generation prompt, Qwen model, and Ollama path fixed.

## Current Structure

- `src/retrieval.py` owns Dense search and returns chunk dictionaries with
  `text`, `score`, `source`, `chunk_id`, and `metadata`.
- `src/lexical_retrieval.py` already owns the frozen BM25 tokenizer/index,
  structural chunk identity, and RRF fusion (`k1=1.2`, `b=0.75`, `k=60`).
- `src/reranker.py` reranks only supplied candidates.
- `src/pdf_rag_demo.py` currently combines Dense selection and reranking under
  `--mode dense|reranker|compare`.
- `rag_demo.py` builds context and citations from source/page/block metadata;
  it does not need to know which retriever produced a result.

## Chosen Design

Add a small `src/hybrid_retrieval.py` orchestration module. It will compose the
existing Dense `retrieve`, `BM25Index.search`, and `rrf_fuse` functions with
fixed M9 candidate limits of 20 per retriever and Hybrid Top-20. It will not
reimplement BM25, tokenization, identity, or RRF. The BM25 index is built once
per PDF session, only when Hybrid is selected.

Expose orthogonal CLI selectors `--retriever dense|hybrid` and `--rerank`.
With neither supplied, resolve to Dense with reranking off. Retain the existing
`--mode dense|reranker|compare` as a compatibility interface; reject a command
that mixes the old and new selectors so precedence is never implicit. The old
`compare` mode keeps its existing Dense-versus-BGE behavior.

Both retrievers return the common chunk fields used by context and citation.
Hybrid may retain `dense_rank`, `bm25_rank`, and `rrf_score` for diagnostics,
but regular user output will show the selected retriever and evidence location,
not Hybrid/BGE scoring internals. Existing `build_context` and citation
provenance (source, page, block type/index, chunk) remain independent of the
retriever. No changes are made to generation, the prompt, or the Dense path.

## Compatibility and Failure Behavior

- `python src/pdf_rag_demo.py document.pdf` remains Dense, no BGE, no BM25.
- Existing `--mode` invocations preserve their behavior.
- Dense and Hybrid return at most Top-3 for generation; with `--rerank`, BGE
  receives the selected retriever's Top-20 pool and returns Top-3.
- Empty corpus or an empty fused pool returns an empty result list and uses the
  current no-results message. If BM25 has no lexical overlap but Dense has
  candidates, RRF preserves those Dense-only candidates.
- Hybrid candidates preserve the structural identity and all source/page/block
  metadata needed for citations. Dense/BM25 overlap is emitted once by the
  existing RRF deduplication.

## Validation

Add tests for default Dense selection, all four Dense/Hybrid × rerank
combinations, legacy CLI compatibility, the Hybrid Top-20 input to BGE,
metadata/citation preservation, duplicate fusion, empty inputs, and hiding
Hybrid scoring details from regular citations.

Run three local PDF smoke cases using the ignored M6 inputs: Q01 (Dense miss,
Hybrid+BGE success), Q06 (known Dense evidence success), and one existing
unanswerable row (Q33–Q35). Smoke checks establish end-to-end wiring, context,
answer/refusal behavior, and citation provenance; they are not a new answer
accuracy benchmark. Do not save raw questions, answers, or document text in
tracked files.

Run `python -m compileall src evaluation`,
`python -m unittest discover -s tests`, and `git diff --check`. Confirm PDFs,
QA, outputs, caches, and model files remain ignored and untracked. Do not rerun
or tune the M9.1/M9.2 retrieval experiment parameters.

## Out of Scope

No change to the embedding model, chunking, MinerU pipeline, retrieval
parameters, generation prompt/model/endpoint, Qwen behavior, query rewriting,
database, Agent, or RAG default. No merge, tag, PR, or push.
