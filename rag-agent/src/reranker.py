"""Optional cross-encoder reranking for dense retrieval candidates."""

import numpy as np


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def load_reranker(model_name=DEFAULT_RERANKER_MODEL):
    """Load a Sentence Transformers CrossEncoder from the local/Hugging Face cache."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. "
            "Run: python -m pip install -r requirements.txt"
        ) from exc

    try:
        return CrossEncoder(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load reranker model '{model_name}'. "
            "Check Hugging Face connectivity and the local model cache. "
            f"Details: {exc}"
        ) from exc


def rerank(query, candidates, model, top_k=3):
    """Score and reorder only the supplied candidates, preserving their metadata."""
    if top_k <= 0 or not candidates:
        return []

    if not isinstance(query, str):
        raise TypeError("query must be a string")

    pairs = []
    for index, candidate in enumerate(candidates):
        text = candidate.get("text")
        if not isinstance(text, str):
            raise ValueError(f"candidate {index} must contain string text")
        pairs.append((query, text))

    scores = np.asarray(
        model.predict(pairs, show_progress_bar=False), dtype=np.float32
    ).reshape(-1)
    if len(scores) != len(candidates):
        raise ValueError(
            "reranker must return exactly one score for each candidate "
            f"(received {len(scores)} scores for {len(candidates)} candidates)"
        )

    scored_candidates = []
    for index, (candidate, score) in enumerate(zip(candidates, scores)):
        result = dict(candidate)
        result["reranker_score"] = float(score)
        scored_candidates.append((index, result))

    scored_candidates.sort(key=lambda item: (-item[1]["reranker_score"], item[0]))
    return [result for _, result in scored_candidates[:top_k]]
