"""Fixed-budget orchestration for Dense + BM25 + RRF retrieval."""

if __package__:
    from .lexical_retrieval import BM25Index, rrf_fuse
    from .retrieval import retrieve
else:
    from lexical_retrieval import BM25Index, rrf_fuse
    from retrieval import retrieve


HYBRID_CANDIDATE_K = 20


def retrieve_hybrid(query, query_embedding, chunks, embeddings, bm25_index):
    """Return the fixed Hybrid Top-20 using existing Dense/BM25/RRF code.

    The common ``score`` field is the RRF score for these results; it is not on
    the same scale as Dense cosine similarity. Component ranks and scores are
    retained as retrieval diagnostics.
    """
    dense_candidates = retrieve(
        query_embedding,
        chunks,
        embeddings,
        top_k=HYBRID_CANDIDATE_K,
    )
    lexical_candidates = bm25_index.search(query, HYBRID_CANDIDATE_K)
    return rrf_fuse(
        dense_candidates,
        lexical_candidates,
        top_k=HYBRID_CANDIDATE_K,
    )
