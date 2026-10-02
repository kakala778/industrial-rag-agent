"""Cache-only, fixed-parameter M9.1 BM25/RRF experiment on the original M6 QA."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_pdf_retrieval import (
    CATEGORIES, TOP_K, _anonymous_candidate_metadata, _contains_keywords_in_texts,
    _evidence_rank_in_results, _expected_source, _matches_case_scope,
    _requires_retrieval_scoring, _safe_integer, _safe_score, _summarize_cases,
    evaluate_questions, load_questions,
)
from evaluation.run_m8_visual_representation_experiment import (
    DEFAULT_DATASET, DEFAULT_INPUT_MANIFEST, DEFAULT_M7_ANALYSIS, DEFAULT_PDF_DIR,
    build_local_input_manifest, require_matching_input_manifest,
)
from src.lexical_retrieval import BM25Index, BM25_B, BM25_K1, RRF_K, chunk_identity, rrf_fuse, tokenize
from src.mineru_loader import CACHE_ROOT, _cache_key, _cached_middle_json, _sha256_file
from src.retrieval import MODEL_NAME, embed_chunks, load_model, retrieve, split_documents
from src.reranker import DEFAULT_RERANKER_MODEL, load_reranker, rerank

CANDIDATE_K = 20
MODES = ("dense", "bm25", "hybrid", "dense_reranker", "hybrid_reranker")
METRICS = {
    "top1_source": ("top1_source_hits", "top1_source_total"),
    "top3_source": ("top3_source_hits", "top3_source_total"),
    "top1_page": ("top1_page_hits", "top1_page_total"),
    "top3_page": ("top3_page_hits", "top3_page_total"),
    "normalized_evidence": ("normalized_keyword_hits", "normalized_keyword_total"),
    "strict_evidence": ("keyword_hits", "keyword_total"),
}
CORE_FILES = ("src/retrieval.py", "src/reranker.py", "src/mineru_loader.py",
              "src/document_loader.py", "src/rag_demo.py", "requirements.txt")


def query_features(query):
    """Generic, overlapping feature groups; never export query tokens."""
    tokens = tokenize(query)
    latin = [token for token in tokens if re.search(r"[a-z]", token)]
    return {"identifier": any(re.search(r"[-_/]", token) or
                              (re.search(r"[a-z]", token) and re.search(r"[0-9]", token))
                              for token in tokens),
            "numeric": any(re.search(r"[0-9]", token) for token in tokens),
            "latin": bool(latin),
            "acronym_like": bool(re.search(r"\b[A-Z]{2,}\b", query)),
            "cjk": any(re.search(r"[\u3400-\u9fff]", token) for token in tokens)}


def classify_retrieval_case(evidence_available, bm25_rank, hybrid_rank):
    if not evidence_available:
        return "PARSING_OR_GT_ISSUE"
    if hybrid_rank is not None and hybrid_rank <= CANDIDATE_K:
        return "RESCUED_BY_HYBRID"
    if bm25_rank is not None and bm25_rank <= CANDIDATE_K:
        return "RESCUED_BY_BM25"
    return "STILL_RETRIEVAL_FAILURE"


def _original_retrieval_ids(analysis, count):
    ids = [case["question_id"] for case in analysis.get("cases", [])
           if case.get("classification") == "RETRIEVAL_FAILURE"]
    if len(ids) != 4 or len(set(ids)) != 4 or any(
        isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= count for value in ids
    ):
        raise ValueError("The pinned M7 analysis must identify four original retrieval failures.")
    return sorted(ids)


def _overlap_profile(query, text):
    query_tokens, result_tokens = set(tokenize(query)), set(tokenize(text))
    shared = query_tokens & result_tokens
    return {"query_token_coverage": len(shared) / len(query_tokens) if query_tokens else 0.0,
            "shared_identifier_tokens": sum(bool(re.search(r"[-_/]", token) or
                (re.search(r"[a-z]", token) and re.search(r"[0-9]", token))) for token in shared),
            "shared_numeric_tokens": sum(bool(re.search(r"[0-9]", token)) for token in shared),
            "shared_cjk_tokens": sum(bool(re.search(r"[\u3400-\u9fff]", token)) for token in shared)}


def _safe_candidate(result, rank, source_ids, query, dense_map, bm25_map):
    identity = chunk_identity(result)
    row = {"rank": rank, "metadata": _anonymous_candidate_metadata(result, source_ids),
           "chunk_id": _safe_integer(result.get("chunk_id")),
           "corpus_index": _safe_integer(result.get("corpus_index")),
           "dense_rank": dense_map.get(identity, _safe_integer(result.get("dense_rank"))),
           "bm25_rank": bm25_map.get(identity, _safe_integer(result.get("bm25_rank"))),
           "score": _safe_score(result.get("score")),
           "query_overlap": _overlap_profile(query, result.get("text", ""))}
    for key in ("rrf_score", "dense_score", "bm25_score", "reranker_score"):
        if key in result:
            row[key] = _safe_score(result[key])
    return row


def _evidence_available(item, values):
    keywords = item.get("expected_keywords", [])
    return bool(keywords) and _contains_keywords_in_texts(
        [value.get("text", "") for value in values if _matches_case_scope(item, value)], keywords
    )


def _joint_success(case):
    return case.get("top3_page_hit") is True and case.get("normalized_keyword_hit") is True


def _transition(reports, predicate, before="dense_reranker", after="hybrid_reranker"):
    if before not in reports or after not in reports:
        return {"available": False, "gained_ids": [], "regression_ids": []}
    pairs = list(zip(reports[before]["cases"], reports[after]["cases"]))
    return {"available": True,
            "baseline_success_count": sum(predicate(case) for case, _ in pairs),
            "gained_ids": [i for i, (old, new) in enumerate(pairs, 1) if not predicate(old) and predicate(new)],
            "regression_ids": [i for i, (old, new) in enumerate(pairs, 1) if predicate(old) and not predicate(new)]}


def build_anonymous_summary(questions, reports, documents, chunks, original_analysis, union_rankings=None):
    """Allowlisted traces only. All evidence judgments reuse the M6/M7 matcher."""
    if any(len(report["cases"]) != len(questions) for report in reports.values()):
        raise ValueError("Reports must preserve QA row order")
    source_ids = {source: f"S{i:03d}" for i, source in enumerate(sorted(
        {_expected_source(item) for item in questions if _expected_source(item) is not None}), 1)}
    metrics = {mode: {name: {"hits": report[hits], "total": report[total]}
                      for name, (hits, total) in METRICS.items()} for mode, report in reports.items()}
    rows = []
    for question_id, item in enumerate(questions, 1):
        scored = _requires_retrieval_scoring(item)
        dense_rows = reports.get("dense", {}).get("cases", [])
        bm25_rows = reports.get("bm25", {}).get("cases", [])
        dense_results = dense_rows[question_id - 1]["diagnostic_results"] if dense_rows else []
        bm25_results = bm25_rows[question_id - 1]["diagnostic_results"] if bm25_rows else []
        dense_map = {chunk_identity(result): rank for rank, result in enumerate(dense_results, 1)}
        bm25_map = {chunk_identity(result): rank for rank, result in enumerate(bm25_results, 1)}
        modes = {}
        for mode, report in reports.items():
            case = report["cases"][question_id - 1]
            candidates = case["diagnostic_results"][:CANDIDATE_K]
            modes[mode] = {
                "evidence_rank": _evidence_rank_in_results(case, candidates) if scored else None,
                "top1_source_hit": case["top1_source_hit"], "top3_source_hit": case["top3_source_hit"],
                "top1_page_hit": case["top1_page_hit"], "top3_page_hit": case["top3_page_hit"],
                "normalized_evidence_hit": case["normalized_keyword_hit"], "strict_evidence_hit": case["keyword_hit"],
                "out_of_scope_top3_count": sum(not _matches_case_scope(item, result) for result in candidates[:TOP_K]),
                "candidates": [_safe_candidate(result, rank, source_ids, item["question"], dense_map, bm25_map)
                               for rank, result in enumerate(candidates, 1)]}
        row = {"question_id": question_id, "scored": scored,
               "known_gt_uncertain": question_id == 17 or item.get("review_required", False),
               "qa_category": item.get("category", "text") if item.get("category", "text") in CATEGORIES else "other",
               "query_features": query_features(item["question"]),
               "evidence_in_documents": _evidence_available(item, documents) if scored else None,
               "evidence_in_chunks": _evidence_available(item, chunks) if scored else None, "modes": modes}
        if union_rankings is not None and scored:
            row["hybrid_union_evidence_rank"] = _evidence_rank_in_results(
                item, union_rankings.get(item["question"], []))
        rows.append(row)
    for mode in reports:
        scored_rows = [row for row in rows if row["scored"] and questions[row["question_id"] - 1].get("expected_keywords")]
        metrics[mode]["recall_at20"] = {"hits": sum(row["modes"][mode]["evidence_rank"] is not None for row in scored_rows),
                                          "total": len(scored_rows)}
        available = [row for row in scored_rows if row["evidence_in_chunks"]]
        metrics[mode]["recall_at20_evidence_available"] = {
            "hits": sum(row["modes"][mode]["evidence_rank"] is not None for row in available), "total": len(available)}
    originals = []
    for question_id in _original_retrieval_ids(original_analysis, len(questions)):
        row = rows[question_id - 1]
        ranks = {mode: row["modes"].get(mode, {}).get("evidence_rank") for mode in MODES}
        available = row["evidence_in_chunks"] and not row["known_gt_uncertain"]
        classification = (classify_retrieval_case(available, ranks["bm25"], ranks["hybrid"])
                          if set(MODES) <= reports.keys() else "NOT_EVALUATED")
        originals.append({"question_id": question_id, "classification": classification,
                          "evidence_in_chunks": row["evidence_in_chunks"], "ranks": ranks,
                          "hybrid_final_evidence_hit": row["modes"].get("hybrid_reranker", {}).get("normalized_evidence_hit"),
                          "hybrid_union_evidence_rank": row.get("hybrid_union_evidence_rank")})
    predicates = {"normalized_evidence": lambda case: case.get("normalized_keyword_hit") is True,
                  "top3_page": lambda case: case.get("top3_page_hit") is True, "joint_success": _joint_success}
    transitions = {name: _transition(reports, predicate) for name, predicate in predicates.items()}
    fusion_transitions = {
        "dense_to_bm25": _transition(reports, predicates["normalized_evidence"], "dense", "bm25"),
        "dense_to_hybrid": _transition(reports, predicates["normalized_evidence"], "dense", "hybrid")}
    groups = {}
    for feature in ("identifier", "numeric", "latin", "acronym_like", "cjk"):
        group = [row for row in rows if row["scored"] and row["query_features"][feature]]
        groups[feature] = {"count": len(group), "normalized_evidence_hits": {
            mode: sum(row["modes"][mode]["normalized_evidence_hit"] is True for row in group) for mode in reports}}
    # Q17's existing M8 uncertainty is a sensitivity slice, never a QA edit.
    sensitivity = {}
    for mode, report in reports.items():
        subset = _summarize_cases([case for i, case in enumerate(report["cases"], 1) if i != 17])
        sensitivity[mode] = {name: {"hits": subset[h], "total": subset[t]} for name, (h, t) in METRICS.items()}
    return {"experiment": "M9.1 Hybrid Retrieval Controlled Experiment", "question_rows": len(questions),
            "scored_questions": sum(row["scored"] for row in rows), "metrics": metrics,
            "original_retrieval_cases": originals,
            "retrieval_classification_counts": dict(Counter(case["classification"] for case in originals)),
            "transitions": transitions, "fusion_transitions": fusion_transitions,
            "query_feature_groups": groups, "excluding_q17_sensitivity": sensitivity, "cases": rows}


def safe_output_root(path):
    output = Path(path).resolve()
    root = (PROJECT_ROOT / "outputs" / "m9_hybrid_retrieval").resolve()
    if not output.is_relative_to(root):
        raise ValueError("Output must stay in the ignored outputs/m9_hybrid_retrieval directory")
    ignored = subprocess.run(["git", "check-ignore", "-q", str(output / "summary.json")],
                             cwd=PROJECT_ROOT, capture_output=True, check=False)
    if ignored.returncode != 0:
        raise ValueError("Experimental output path is not Git ignored")
    return output


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_frozen_inputs(args):
    questions = load_questions(args.dataset)
    analysis = json.loads(args.m7_analysis.read_text(encoding="utf-8"))
    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    require_matching_input_manifest(questions, analysis, pdfs, args.dataset, args.m7_analysis, args.input_manifest)
    _original_retrieval_ids(analysis, len(questions))
    documents, cache_hashes = [], []
    for pdf in pdfs:
        key, identity = _cache_key(pdf, _sha256_file(pdf))
        entry = CACHE_ROOT / key
        cached = _cached_middle_json(entry, pdf, identity, key, representation="structured")
        if cached is None:
            raise RuntimeError("A matching MinerU cache is unavailable; stopped without parsing")
        documents.extend(cached)
        cache_hashes.append(_sha256_file(entry / "output" / f"{pdf.stem}.json"))
    inventory = build_local_input_manifest(questions, analysis, pdfs, args.dataset, args.m7_analysis)
    inventory["cache_cohort_sha256"] = hashlib.sha256("\n".join(sorted(cache_hashes)).encode()).hexdigest()
    return questions, analysis, documents, inventory


def _timing_summary(values):
    return {"count": len(values), "total_seconds": sum(values),
            "mean_seconds": sum(values) / len(values) if values else None,
            "min_seconds": min(values) if values else None, "max_seconds": max(values) if values else None}


def _assert_baseline(reports):
    expected = {"dense": {"top3_page_hits": 15, "normalized_keyword_hits": 10},
                "dense_reranker": {"top1_page_hits": 18, "top3_page_hits": 19, "normalized_keyword_hits": 14}}
    for mode, checks in expected.items():
        if mode in reports and any(reports[mode][key] != value for key, value in checks.items()):
            raise RuntimeError("The original M7 baseline did not reproduce; no conclusions should be drawn")


def run_experiment(args):
    modes = selected_modes(args)
    questions, original, documents, inventory = load_frozen_inputs(args)
    core_hashes = {name: _sha256_file(PROJECT_ROOT / name) for name in CORE_FILES}
    chunks = split_documents(documents)
    identity_indices = {chunk_identity(chunk): index for index, chunk in enumerate(chunks)}
    if len(identity_indices) != len(chunks):
        raise ValueError("Chunk identities are not unique in the fixed corpus")
    root = safe_output_root(args.output_dir)
    run_dir = root / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    # Cached local models only: the experiment must not acquire new model weights.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    need_dense = any(mode != "bm25" for mode in modes)
    need_lexical = any(mode in {"bm25", "hybrid", "hybrid_reranker"} for mode in modes)
    timings = {key: [] for key in ("query_embedding", "dense_query", "bm25_query", "rrf_fusion",
                                   "hybrid_query_without_embedding", "dense_reranker", "hybrid_reranker")}
    started = time.perf_counter()
    bm25 = BM25Index(chunks) if need_lexical else None
    index_build = time.perf_counter() - started
    print(f"Verified cached corpus: {len(documents)} documents, {len(chunks)} chunks. Loading cached models.", flush=True)
    started = time.perf_counter()
    model = load_model() if need_dense else None
    embeddings = embed_chunks(chunks, model) if model is not None else []
    embedding_setup = time.perf_counter() - started
    started = time.perf_counter()
    reranker_model = load_reranker() if any(mode.endswith("reranker") for mode in modes) else None
    reranker_setup = time.perf_counter() - started
    rankings = {mode: {} for mode in modes}
    unions = {}
    for question_id, item in enumerate(questions, 1):
        if not _requires_retrieval_scoring(item):
            continue
        query = item["question"]
        started = time.perf_counter()
        vector = model.encode(query, convert_to_numpy=True, show_progress_bar=False) if model is not None else None
        if model is not None:
            timings["query_embedding"].append(time.perf_counter() - started)
        started = time.perf_counter()
        dense = retrieve(vector, chunks, embeddings, top_k=CANDIDATE_K) if model is not None else []
        dense_time = time.perf_counter() - started
        for candidate in dense:
            candidate["corpus_index"] = identity_indices[chunk_identity(candidate)]
        if model is not None:
            timings["dense_query"].append(dense_time)
        started = time.perf_counter()
        lexical = bm25.search(query, CANDIDATE_K) if bm25 is not None else []
        bm25_time = time.perf_counter() - started
        if bm25 is not None:
            timings["bm25_query"].append(bm25_time)
        hybrid = []
        if any(mode.startswith("hybrid") for mode in modes):
            started = time.perf_counter()
            union = rrf_fuse(dense, lexical, top_k=2 * CANDIDATE_K)
            rrf_time = time.perf_counter() - started
            timings["rrf_fusion"].append(rrf_time)
            timings["hybrid_query_without_embedding"].append(dense_time + bm25_time + rrf_time)
            unions[query] = union
            hybrid = union[:CANDIDATE_K]
        pools = {"dense": dense, "bm25": lexical, "hybrid": hybrid}
        for mode in modes:
            if mode.endswith("reranker"):
                pool = pools[mode.removesuffix("_reranker")]
                started = time.perf_counter()
                rankings[mode][query] = rerank(query, pool, reranker_model, top_k=len(pool))
                timings[mode].append(time.perf_counter() - started)
            else:
                rankings[mode][query] = pools[mode]
        print(f"Q{question_id:02d}: fixed retrieval arms complete", flush=True)
    reports = {}
    for mode in modes:
        report = evaluate_questions(questions, None, chunks, [], documents=documents, diagnostic_k=CANDIDATE_K,
            candidate_provider=lambda query, limit, rows=rankings[mode]: rows.get(query, [])[:limit])
        reports[mode] = report
    _assert_baseline(reports)
    summary = build_anonymous_summary(questions, reports, documents, chunks, original, unions)
    _, _, _, after_inventory = load_frozen_inputs(args)
    if inventory != after_inventory or core_hashes != {name: _sha256_file(PROJECT_ROOT / name) for name in CORE_FILES}:
        raise RuntimeError("Frozen inputs or controlled core files changed during the experiment")
    summary["configuration"] = {"parser": "mineru", "representation": "structured", "top_k": TOP_K,
        "dense_limit": CANDIDATE_K, "bm25_limit": CANDIDATE_K, "hybrid_limit": CANDIDATE_K,
        "bm25_k1": BM25_K1, "bm25_b": BM25_B, "rrf_k": RRF_K, "fusion_weights": [1, 1],
        "tokenizer": "nfkc_casefold_identifier_cjk_unigram_bigram_v1", "embedding_model": MODEL_NAME,
        "reranker_model": DEFAULT_RERANKER_MODEL, "documents": len(documents), "chunks": len(chunks),
        "mineru_calls": 0, "input_inventory": inventory, "core_sha256": core_hashes,
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "sentence-transformers", "torch", "pymupdf")},
        "python": sys.version.split()[0], "embedding_device": str(model.device) if model is not None else None,
        "reranker_device": str(reranker_model.model.device) if reranker_model is not None else None}
    summary["runtime"] = {"bm25_index_build_seconds": index_build, "embedding_load_and_index_seconds": embedding_setup,
                          "reranker_load_seconds": reranker_setup,
                          "stages": {key: _timing_summary(values) for key, values in timings.items()}}
    _write_json(run_dir / "summary.json", summary)
    print(json.dumps({"metrics": summary["metrics"], "retrieval_cases": summary["original_retrieval_cases"],
                      "transitions": summary["transitions"]}, indent=2), flush=True)
    print(f"Anonymous results: outputs/m9_hybrid_retrieval/{run_dir.name}/summary.json", flush=True)
    return summary


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-all", action="store_true", help="Run all five fixed M9 retrieval arms")
    parser.add_argument("--retriever", choices=("dense", "bm25", "hybrid"), default="dense")
    parser.add_argument("--rerank", choices=("on", "off"), default="off")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--m7-analysis", type=Path, default=DEFAULT_M7_ANALYSIS)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "m9_hybrid_retrieval")
    return parser


def selected_modes(args):
    if args.compare_all:
        if args.retriever != "dense" or args.rerank != "off":
            raise ValueError("--compare-all selects the five fixed arms; omit individual mode switches")
        return MODES
    if args.retriever == "bm25" and args.rerank == "on":
        raise ValueError("BM25-only is unreranked in this fixed five-arm experiment")
    return (args.retriever + ("_reranker" if args.rerank == "on" else ""),)


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    try:
        run_experiment(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"M9 stopped: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
