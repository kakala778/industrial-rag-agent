"""Compare PDF parser paths using retrieval-only QA metrics."""

import argparse
import json
import math
import re
import sys
import time
import unicodedata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "examples" / "pdf"
DATASET_PATH = Path(__file__).resolve().parent / "pdf_qa_dataset.json"
TOP_K = 3
DIAGNOSTIC_K = 10
RERANK_CANDIDATE_K = 20
CATEGORIES = {
    "text",
    "numeric",
    "unit",
    "model",
    "table",
    "simple_table",
    "complex_table",
    "ocr",
    "layout",
    "drawing_layout",
    "similar_field",
    "multi_fact",
    "unanswerable",
}

# Allow `python evaluation/evaluate_pdf_retrieval.py` from the project root.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import load_pdf  # noqa: E402
from src.retrieval import (  # noqa: E402
    embed_chunks,
    load_model,
    retrieve,
    split_documents,
)
from src.reranker import (  # noqa: E402
    DEFAULT_RERANKER_MODEL,
    load_reranker,
    rerank,
)


def load_questions(path=DATASET_PATH):
    """Read the legacy PDF set or the local industrial QA schema."""
    questions = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(questions, list) or not questions:
        raise ValueError("The PDF evaluation dataset must be a non-empty JSON array.")

    for index, item in enumerate(questions, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Question {index} must be a JSON object.")
        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Question {index} 'question' must be a non-empty string.")

        source = item.get("expected_source", item.get("source"))
        if source is not None and (not isinstance(source, str) or not source.strip()):
            raise ValueError(f"Question {index} source must be a non-empty string when set.")

        expected_page = item.get("expected_page")
        if expected_page is not None and (
            isinstance(expected_page, bool)
            or not isinstance(expected_page, int)
            or expected_page < 1
        ):
            raise ValueError(f"Question {index} expected_page must be a positive 1-based page.")
        if expected_page is not None and source is None:
            raise ValueError(
                f"Question {index} expected_page requires an expected_source to avoid "
                "page-number matches across different PDFs."
            )

        keywords = item.get("expected_keywords", [])
        if not isinstance(keywords, list) or not all(
            isinstance(keyword, str) and keyword.strip() for keyword in keywords
        ):
            raise ValueError(
                f"Question {index} expected_keywords must be an array of non-empty strings."
            )

        category = item.get("category", item.get("type", "text"))
        if not isinstance(category, str) or category not in CATEGORIES:
            raise ValueError(
                f"Question {index} category must be one of: {', '.join(sorted(CATEGORIES))}."
            )
        if "answer" in item and not isinstance(item["answer"], str):
            raise ValueError(f"Question {index} answer must be a string when set.")
        if "review_required" in item and not isinstance(item["review_required"], bool):
            raise ValueError(f"Question {index} review_required must be a boolean.")
        if (
            source is None
            and expected_page is None
            and not keywords
            and category != "unanswerable"
            and not item.get("review_required", False)
        ):
            raise ValueError(
                f"Question {index} needs a source, expected_page, expected_keywords, "
                "or an unanswerable/review_required label."
            )

    return questions


def _expected_source(item):
    return item.get("expected_source", item.get("source"))


def _requires_retrieval_scoring(item):
    """Share the scoring eligibility rule between loading and evaluation."""
    if item.get("review_required", False):
        return False
    return not (
        _expected_source(item) is None
        and item.get("expected_page") is None
        and not item.get("expected_keywords", [])
        and item.get("category", item.get("type", "text")) == "unanswerable"
    )


def _normalize_evidence_text(text):
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", "", normalized)


def load_pdf_documents(
    pdf_dir,
    questions,
    *,
    pdf_path=None,
    parser="pymupdf",
    force=False,
    runner_path=None,
    representation="structured",
):
    """Load the one requested PDF or legacy dataset sources through a parser."""
    expected_sources = {
        source for item in questions if (source := _expected_source(item)) is not None
    }
    unscoped_scored_count = sum(
        _expected_source(item) is None and _requires_retrieval_scoring(item)
        for item in questions
    )

    if pdf_path is not None:
        selected_pdf = Path(pdf_path)
        if not selected_pdf.is_file():
            raise FileNotFoundError(f"PDF file not found: {selected_pdf}")
        if len(expected_sources) > 1:
            raise ValueError(
                "--pdf accepts a single-source dataset, but this dataset references "
                f"multiple sources: {', '.join(sorted(expected_sources))}. "
                "Use --pdf-dir for a multi-source dataset or select a single-source dataset."
            )
        if unscoped_scored_count:
            raise ValueError(
                "--pdf requires an expected_source for every scored question. "
                f"Found {unscoped_scored_count} question(s) without one."
            )
        if expected_sources and selected_pdf.name not in expected_sources:
            raise FileNotFoundError(
                f"The dataset expects source(s) {', '.join(sorted(expected_sources))}, "
                f"but the selected PDF is {selected_pdf.name}."
            )
        return load_pdf(
            selected_pdf,
            parser=parser,
            force=force,
            runner_path=runner_path,
            representation=representation,
        )

    pdf_dir = Path(pdf_dir)
    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    available = {path.name: path for path in pdf_dir.glob("*.pdf")}
    if not available:
        raise FileNotFoundError(f"No PDF files found in: {pdf_dir}")

    missing_sources = sorted(expected_sources - available.keys())
    if missing_sources:
        raise FileNotFoundError(
            "PDF files referenced by the dataset are missing from "
            f"{pdf_dir}: {', '.join(missing_sources)}"
        )
    selected_sources = (
        set(available)
        if unscoped_scored_count or not expected_sources
        else expected_sources
    )

    documents = []
    for source in sorted(selected_sources):
        documents.extend(
            load_pdf(
                available[source],
                parser=parser,
                force=force,
                runner_path=runner_path,
                representation=representation,
            )
        )
    return documents


def _page_hit(result, expected_page, expected_source):
    metadata = result.get("metadata") or {}
    return metadata.get("page") == expected_page and (
        expected_source is None or result.get("source") == expected_source
    )


def _matches_case_scope(item, value):
    metadata = value.get("metadata") or {}
    expected_source = _expected_source(item)
    expected_page = item.get("expected_page")
    source = value.get("source", metadata.get("source"))
    if expected_source is not None and source != expected_source:
        return False
    if expected_page is not None and metadata.get("page") != expected_page:
        return False
    return True


def _contains_keywords_in_texts(texts, keywords):
    normalized_texts = [_normalize_evidence_text(text) for text in texts]
    return all(
        any(_normalize_evidence_text(keyword) in text for text in normalized_texts)
        for keyword in keywords
    )


def _diagnose_failure(
    item,
    documents,
    chunks,
    diagnostic_results,
    top_k=TOP_K,
    representation_reference_documents=None,
):
    """Classify a miss without inferring parsing failures from absent inputs."""
    keywords = item.get("expected_keywords", [])
    if item.get("review_required", False):
        return "GT_UNCERTAIN", "Ground truth requires human review."

    documents_available = documents is not None
    if documents_available:
        scoped_documents = [
            document for document in documents if _matches_case_scope(item, document)
        ]
        if not scoped_documents:
            reference_documents = [
                document
                for document in (representation_reference_documents or [])
                if _matches_case_scope(item, document)
            ]
            if reference_documents:
                return (
                    "REPRESENTATION",
                    "The source/page exists in the flat MinerU reference but not in the selected representation.",
                )
            return "PARSING", "The expected source/page is absent from loaded documents."

        if keywords and not _contains_keywords_in_texts(
            [document.get("text", "") for document in scoped_documents], keywords
        ):
            reference_documents = [
                document
                for document in (representation_reference_documents or [])
                if _matches_case_scope(item, document)
            ]
            if reference_documents and _contains_keywords_in_texts(
                [document.get("text", "") for document in reference_documents],
                keywords,
            ):
                return (
                    "REPRESENTATION",
                    "Expected evidence is present in the flat MinerU reference but missing from the selected representation.",
                )
            return "PARSING", "Expected evidence is missing from the loaded source/page text."

    scoped_chunks = [chunk for chunk in chunks if _matches_case_scope(item, chunk)]
    if keywords and not scoped_chunks:
        if not documents_available:
            return (
                "INSUFFICIENT_DATA",
                "Raw documents were not provided, so parsing and chunking cannot be distinguished.",
            )
        return "CHUNKING", "Loaded source/page evidence did not produce any chunks."
    if keywords and not _contains_keywords_in_texts(
        [chunk.get("text", "") for chunk in scoped_chunks], keywords
    ):
        if not documents_available:
            return (
                "INSUFFICIENT_DATA",
                "Raw documents were not provided, so missing chunk evidence cannot be attributed to parsing or chunking.",
            )
        return "CHUNKING", "Expected evidence was lost while splitting the source/page into chunks."
    if not documents_available and not scoped_chunks:
        return (
            "INSUFFICIENT_DATA",
            "Raw documents were not provided, so the missing source/page cannot be attributed to parsing.",
        )

    scoped_diagnostic_results = [
        result
        for result in diagnostic_results
        if _matches_case_scope(item, result)
    ]
    evidence_in_diagnostic_results = (
        _contains_keywords_in_texts(
            [result.get("text", "") for result in scoped_diagnostic_results],
            keywords,
        )
        if keywords
        else bool(scoped_diagnostic_results)
    )
    top_k_results = [
        result
        for result in diagnostic_results[:top_k]
        if _matches_case_scope(item, result)
    ]
    evidence_in_top_k = (
        _contains_keywords_in_texts(
            [result.get("text", "") for result in top_k_results], keywords
        )
        if keywords
        else bool(top_k_results)
    )

    if (
        evidence_in_diagnostic_results
        and scoped_diagnostic_results
        and diagnostic_results
        and not _matches_case_scope(item, diagnostic_results[0])
    ):
        return "RANKING", "Expected source/page evidence is in diagnostic Top-K but rank 1 is outside scope."

    if evidence_in_diagnostic_results and not evidence_in_top_k:
        return "RANKING", f"Expected evidence appears in diagnostic Top-{len(diagnostic_results)} but not Top-{top_k}."

    return "RETRIEVAL", "Expected evidence exists in chunks but was not retrieved in Top-10."


def _unscored_case(item, *, review_required):
    """Build the report row for a case excluded from retrieval scoring."""
    category = item.get("category", item.get("type", "text"))
    return {
        "question": item["question"],
        "source": _expected_source(item) if review_required else None,
        "expected_page": item.get("expected_page") if review_required else None,
        "category": category,
        "expected_keywords": (
            item.get("expected_keywords", []) if review_required else []
        ),
        "review_required": review_required,
        "not_applicable": not review_required,
        "top1_source_hit": None,
        "top3_source_hit": None,
        "top1_page_hit": None,
        "top3_page_hit": None,
        "keyword_hit": None,
        "normalized_keyword_hit": None,
        "results": [],
        "diagnostic_results": [],
        "error_layer": "GT_UNCERTAIN" if review_required else None,
        "reason": (
            "Ground truth requires human review; this case was not evaluated."
            if review_required
            else None
        ),
    }


def _evaluate_scored_question(
    item,
    *,
    model,
    chunks,
    embeddings,
    top_k,
    documents,
    representation_reference_documents,
    mode="dense",
    reranker_model=None,
):
    """Retrieve and score one question, including failure attribution."""
    expected_source = _expected_source(item)
    expected_page = item.get("expected_page")
    expected_keywords = item.get("expected_keywords", [])
    category = item.get("category", item.get("type", "text"))

    dense_candidate_results = []
    if chunks:
        query_embedding = model.encode(
            item["question"],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if mode == "dense":
            diagnostic_results = retrieve(
                query_embedding,
                chunks,
                embeddings,
                top_k=max(top_k, DIAGNOSTIC_K),
            )
        else:
            candidates = retrieve(
                query_embedding,
                chunks,
                embeddings,
                top_k=max(RERANK_CANDIDATE_K, top_k),
            )
            dense_candidate_results = candidates
            diagnostic_results = rerank(
                item["question"],
                candidates,
                reranker_model,
                top_k=len(candidates),
            )
    else:
        diagnostic_results = []
    top_results = diagnostic_results[:top_k]

    top1_source_hit = (
        None
        if expected_source is None
        else bool(
            diagnostic_results
            and diagnostic_results[0]["source"] == expected_source
        )
    )
    top3_source_hit = (
        None
        if expected_source is None
        else any(result["source"] == expected_source for result in top_results)
    )
    top1_page_hit = (
        None
        if expected_page is None
        else bool(
            diagnostic_results
            and _page_hit(
                diagnostic_results[0], expected_page, expected_source
            )
        )
    )
    top3_page_hit = (
        None
        if expected_page is None
        else any(
            _page_hit(result, expected_page, expected_source) for result in top_results
        )
    )
    evidence_results = [
        result for result in top_results if _matches_case_scope(item, result)
    ]
    retrieved_text = "\n".join(
        result["text"] for result in evidence_results
    ).casefold()
    keyword_hit = (
        None
        if not expected_keywords
        else all(keyword.casefold() in retrieved_text for keyword in expected_keywords)
    )
    normalized_keyword_hit = (
        None
        if not expected_keywords
        else _contains_keywords_in_texts(
            [result.get("text", "") for result in evidence_results],
            expected_keywords,
        )
    )

    case = {
        "question": item["question"],
        "source": expected_source,
        "expected_page": expected_page,
        "category": category,
        "review_required": False,
        "not_applicable": False,
        "expected_keywords": expected_keywords,
        "top1_source_hit": top1_source_hit,
        "top3_source_hit": top3_source_hit,
        "top1_page_hit": top1_page_hit,
        "top3_page_hit": top3_page_hit,
        "keyword_hit": keyword_hit,
        "normalized_keyword_hit": normalized_keyword_hit,
        "results": top_results,
        "diagnostic_results": diagnostic_results,
    }
    if mode == "reranker":
        case["dense_candidate_results"] = dense_candidate_results
    failed = (
        top1_source_hit is False
        or top3_source_hit is False
        or top1_page_hit is False
        or top3_page_hit is False
        or normalized_keyword_hit is False
    )
    if failed:
        case["error_layer"], case["reason"] = _diagnose_failure(
            item,
            documents,
            chunks,
            diagnostic_results,
            top_k=top_k,
            representation_reference_documents=representation_reference_documents,
        )
    else:
        case["error_layer"] = None
        case["reason"] = None
    return case


def _summarize_cases(cases):
    """Aggregate per-question rows into the existing report structure."""
    return {
        "cases": cases,
        "total": len(cases),
        "evaluated_total": sum(
            not case["review_required"] and not case["not_applicable"]
            for case in cases
        ),
        "review_required_total": sum(case["review_required"] for case in cases),
        "not_applicable_total": sum(case["not_applicable"] for case in cases),
        "top1_source_hits": sum(case["top1_source_hit"] is True for case in cases),
        "top1_source_total": sum(case["top1_source_hit"] is not None for case in cases),
        "top3_source_hits": sum(case["top3_source_hit"] is True for case in cases),
        "top3_source_total": sum(case["top3_source_hit"] is not None for case in cases),
        "top1_page_hits": sum(case["top1_page_hit"] is True for case in cases),
        "top1_page_total": sum(case["top1_page_hit"] is not None for case in cases),
        "top3_page_hits": sum(case["top3_page_hit"] is True for case in cases),
        "top3_page_total": sum(case["top3_page_hit"] is not None for case in cases),
        "keyword_hits": sum(case["keyword_hit"] is True for case in cases),
        "keyword_total": sum(case["keyword_hit"] is not None for case in cases),
        "normalized_keyword_hits": sum(
            case["normalized_keyword_hit"] is True for case in cases
        ),
        "normalized_keyword_total": sum(
            case["normalized_keyword_hit"] is not None for case in cases
        ),
        "error_layer_counts": {
            layer: sum(case["error_layer"] == layer for case in cases)
            for layer in (
                "PARSING",
                "REPRESENTATION",
                "CHUNKING",
                "RETRIEVAL",
                "RANKING",
                "INSUFFICIENT_DATA",
                "GT_UNCERTAIN",
            )
        },
    }


def evaluate_questions(
    questions,
    model,
    chunks,
    embeddings,
    top_k=TOP_K,
    documents=None,
    representation_reference_documents=None,
    mode="dense",
    reranker_model=None,
):
    """Measure PDF retrieval metrics for dense or reranked candidate ordering."""
    if mode not in {"dense", "reranker"}:
        raise ValueError("mode must be 'dense' or 'reranker'")
    if mode == "reranker" and chunks and reranker_model is None:
        raise ValueError("reranker_model is required when mode='reranker'")

    cases = []

    for item in questions:
        if item.get("review_required", False):
            cases.append(_unscored_case(item, review_required=True))
        elif not _requires_retrieval_scoring(item):
            cases.append(_unscored_case(item, review_required=False))
        else:
            cases.append(
                _evaluate_scored_question(
                    item,
                    model=model,
                    chunks=chunks,
                    embeddings=embeddings,
                    top_k=top_k,
                    documents=documents,
                    representation_reference_documents=representation_reference_documents,
                    mode=mode,
                    reranker_model=reranker_model,
                )
            )

    return _summarize_cases(cases)


def evaluate_comparison(
    questions,
    model,
    chunks,
    embeddings,
    reranker_model,
    top_k=TOP_K,
    documents=None,
    representation_reference_documents=None,
):
    """Run dense and reranked metrics over the same documents and QA cases."""
    common_arguments = {
        "top_k": top_k,
        "documents": documents,
        "representation_reference_documents": representation_reference_documents,
    }
    return {
        "dense": evaluate_questions(
            questions,
            model,
            chunks,
            embeddings,
            mode="dense",
            **common_arguments,
        ),
        "reranker": evaluate_questions(
            questions,
            model,
            chunks,
            embeddings,
            mode="reranker",
            reranker_model=reranker_model,
            **common_arguments,
        ),
    }


def _format_metric(name, hits, total):
    if not total:
        return f"{name}: n/a"
    return f"{name}: {hits}/{total}"


def _case_evidence_rank(case):
    """Return the first rank whose scoped cumulative results contain all evidence."""
    keywords = case.get("expected_keywords", [])
    if not keywords:
        return None

    scoped_texts = []
    for rank, result in enumerate(case.get("diagnostic_results", []), start=1):
        if _matches_case_scope(case, result):
            scoped_texts.append(result.get("text", ""))
            if _contains_keywords_in_texts(scoped_texts, keywords):
                return rank
    return None


def _evidence_rank_in_results(case, results):
    ranked_case = dict(case)
    ranked_case["diagnostic_results"] = results
    evidence_rank = _case_evidence_rank(ranked_case)
    if evidence_rank is not None or case.get("expected_keywords"):
        return evidence_rank
    if case.get("expected_page") is not None:
        return _case_page_rank(ranked_case)
    for rank, result in enumerate(results, start=1):
        if _matches_case_scope(case, result):
            return rank
    return None


_FAILURE_ANALYSIS_CATEGORIES = (
    "FIXED_BY_RERANKER",
    "STILL_RANKING_FAILURE",
    "RETRIEVAL_FAILURE",
    "PARSING_FAILURE",
    "INSUFFICIENT_DATA",
)
_SAFE_BLOCK_TYPES = {
    "chart",
    "equation",
    "figure",
    "image",
    "list",
    "table",
    "text",
    "text_group",
    "title",
}


def _safe_integer(value):
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _safe_score(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        score = float(value)
        if math.isfinite(score):
            return score
    return None


def _candidate_identity(result):
    """Return a transient candidate key using structural metadata only."""
    metadata = result.get("metadata") or {}
    source = metadata.get("source", result.get("source"))
    chunk_id = result.get("chunk_id")
    if source is not None and chunk_id is not None:
        return source, chunk_id
    page = metadata.get("page")
    block_index = metadata.get("block_index")
    block_type = metadata.get("block_type")
    if source is not None and (page is not None or block_index is not None):
        return source, page, block_type, block_index
    return None


def _anonymous_candidate_metadata(result, source_ids):
    metadata = result.get("metadata") or {}
    source = metadata.get("source", result.get("source"))
    if source is None:
        source_id = None
    else:
        if source not in source_ids:
            source_ids[source] = f"S{len(source_ids) + 1:03d}"
        source_id = source_ids[source]

    block_type = metadata.get("block_type")
    if not isinstance(block_type, str):
        block_type = None
    else:
        block_type = block_type.casefold()
        if block_type not in _SAFE_BLOCK_TYPES:
            block_type = "other"

    return {
        "source_id": source_id,
        "page": _safe_integer(metadata.get("page")),
        "block_type": block_type,
        "block_index": _safe_integer(metadata.get("block_index")),
    }


def _serialize_analysis_candidate(
    result, rank, source_ids, *, dense_rank=None, reranker_rank=None
):
    row = {
        "rank": rank,
        "metadata": _anonymous_candidate_metadata(result, source_ids),
    }
    chunk_id = _safe_integer(result.get("chunk_id"))
    if chunk_id is not None:
        row["chunk_id"] = chunk_id
    dense_score = _safe_score(result.get("score"))
    if dense_score is not None:
        row["dense_score"] = dense_score
    reranker_score = _safe_score(result.get("reranker_score"))
    if reranker_score is not None:
        row["reranker_score"] = reranker_score
    if dense_rank is not None:
        row["dense_rank"] = dense_rank
        row["rank_change"] = dense_rank - rank
    if reranker_rank is not None:
        row["reranker_rank"] = reranker_rank
        row["rank_change"] = rank - reranker_rank
    return row


def _confirmed_structured_parsing_failure(case, parser, representation):
    """Only accept the evaluator's explicit evidence-missing result for MinerU structured output."""
    return (
        case.get("error_layer") == "PARSING"
        and parser == "mineru"
        and representation == "structured"
        and bool(case.get("expected_keywords"))
    )


def build_failure_analysis(reports, top_k=TOP_K, candidate_k=RERANK_CANDIDATE_K):
    """Build privacy-filtered candidate traces and M6 failure classes."""
    if "dense" not in reports or "reranker" not in reports:
        raise ValueError("reports must include dense and reranker results")

    dense_cases = reports["dense"]["cases"]
    reranker_cases = reports["reranker"]["cases"]
    if len(dense_cases) != len(reranker_cases):
        raise ValueError("dense and reranker reports must have the same case count")

    counts = {category: 0 for category in _FAILURE_ANALYSIS_CATEGORIES}
    analyzed_cases = []
    source_ids = {}
    parsing_proof_available = (
        reports["dense"].get("parser") == "mineru"
        and reports["dense"].get("representation") == "structured"
        and reports["reranker"].get("parser") == "mineru"
        and reports["reranker"].get("representation") == "structured"
    )
    total_original_failures = 0
    not_baseline_failure_count = 0
    not_scored_count = 0

    for index, (dense_case, reranker_case) in enumerate(
        zip(dense_cases, reranker_cases), start=1
    ):
        original_layer = dense_case.get("error_layer")
        dense_candidates = reranker_case.get("dense_candidate_results")
        reranked_candidates = reranker_case.get("diagnostic_results")
        is_unscored = bool(
            dense_case.get("review_required")
            or dense_case.get("not_applicable")
            or original_layer == "GT_UNCERTAIN"
        )
        is_original_failure = original_layer is not None and not is_unscored
        if is_original_failure:
            total_original_failures += 1
        elif not is_unscored:
            not_baseline_failure_count += 1
        if is_unscored:
            not_scored_count += 1

        dense_rank = None
        if dense_candidates is not None:
            dense_rank = _evidence_rank_in_results(
                dense_case, dense_candidates[:candidate_k]
            )
        reranker_rank = _evidence_rank_in_results(
            reranker_case,
            reranked_candidates if reranked_candidates is not None else [],
        )

        if not is_original_failure:
            classification = (
                "INSUFFICIENT_DATA" if is_unscored else "NOT_A_BASELINE_FAILURE"
            )
        elif (
            parsing_proof_available
            and _confirmed_structured_parsing_failure(
                dense_case, "mineru", "structured"
            )
        ):
            classification = "PARSING_FAILURE"
        elif dense_candidates is None or reranked_candidates is None:
            classification = "INSUFFICIENT_DATA"
        elif original_layer not in {"PARSING", "RANKING", "RETRIEVAL"}:
            classification = "INSUFFICIENT_DATA"
        elif original_layer == "PARSING":
            classification = "INSUFFICIENT_DATA"
        elif dense_rank is None:
            classification = "RETRIEVAL_FAILURE"
        elif reranker_rank is None:
            classification = "INSUFFICIENT_DATA"
        elif reranker_rank > top_k:
            classification = "STILL_RANKING_FAILURE"
        elif dense_rank > top_k:
            classification = "FIXED_BY_RERANKER"
        else:
            # The baseline row failed for a reason not explained by the shared
            # evidence matcher; do not force it into a ranking category.
            classification = "INSUFFICIENT_DATA"

        if is_original_failure:
            counts[classification] += 1

        rank_change = (
            dense_rank - reranker_rank
            if dense_rank is not None and reranker_rank is not None
            else None
        )
        reranked_rank_by_identity = {}
        if reranked_candidates is not None:
            for rank, candidate in enumerate(reranked_candidates, start=1):
                identity = _candidate_identity(candidate)
                if identity is not None:
                    reranked_rank_by_identity.setdefault(identity, rank)

        dense_top20 = []
        if dense_candidates is not None:
            for dense_position, candidate in enumerate(
                dense_candidates[:candidate_k], start=1
            ):
                candidate_rank = reranked_rank_by_identity.get(
                    _candidate_identity(candidate)
                )
                dense_top20.append(
                    _serialize_analysis_candidate(
                        candidate,
                        dense_position,
                        source_ids,
                        reranker_rank=candidate_rank,
                    )
                )

        reranked_top3 = []
        if reranked_candidates is not None:
            dense_rank_by_identity = {
                _candidate_identity(candidate): rank
                for rank, candidate in enumerate(
                    (dense_candidates or [])[:candidate_k], start=1
                )
                if _candidate_identity(candidate) is not None
            }
            for rerank_position, candidate in enumerate(
                reranked_candidates[:top_k], start=1
            ):
                candidate_dense_rank = dense_rank_by_identity.get(
                    _candidate_identity(candidate)
                )
                reranked_top3.append(
                    _serialize_analysis_candidate(
                        candidate,
                        rerank_position,
                        source_ids,
                        dense_rank=candidate_dense_rank,
                    )
                )

        analyzed_cases.append(
            {
                "question_id": index,
                "analysis_scope": (
                    "ORIGINAL_FAILURE"
                    if is_original_failure
                    else "UNSCORED"
                    if is_unscored
                    else "BASELINE_SUCCESS"
                ),
                "classification": classification,
                "original_m6_failure_layer": original_layer,
                "dense_candidate_count": (
                    len(dense_candidates) if dense_candidates is not None else None
                ),
                "dense_top20": dense_top20,
                "reranked_top3": reranked_top3,
                "dense_evidence_rank": dense_rank,
                "reranker_evidence_rank": reranker_rank,
                "evidence_rank_change": rank_change,
            }
        )

    return {
        "experiment": {
            "baseline": "dense retrieval",
            "reranker": DEFAULT_RERANKER_MODEL,
            "candidate_limit": candidate_k,
            "reranker_output_limit": top_k,
            "question_id_policy": "1-based row number; no question text or source ID mapping is included.",
            "source_id_policy": "Source labels are run-local Snn identifiers.",
            "rank_change_policy": "Positive rank_change means movement toward rank 1.",
            "parser": reports["dense"].get("parser", "unknown"),
            "representation": reports["dense"].get("representation", "unknown"),
        },
        "total_questions": len(analyzed_cases),
        "total_original_failures": total_original_failures,
        "not_baseline_failure_count": not_baseline_failure_count,
        "not_scored_count": not_scored_count,
        "classification_counts": counts,
        "cases": analyzed_cases,
    }


def _failure_analysis_path_is_ignored(path):
    output = Path(path).resolve()
    ignored_root = (PROJECT_ROOT / "outputs").resolve()
    try:
        output.relative_to(ignored_root)
    except ValueError:
        return False
    return True


def _write_failure_analysis(path, analysis):
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _case_page_rank(case):
    expected_page = case.get("expected_page")
    if expected_page is None:
        return None
    for rank, result in enumerate(case.get("diagnostic_results", []), start=1):
        if _page_hit(result, expected_page, case.get("source")):
            return rank
    return None


def _rank_text(rank, limit):
    if rank is None:
        return f"not in Top-{limit}"
    return str(rank)


def _hit_text(hit):
    if hit is None:
        return "n/a"
    return "hit" if hit else "miss"


def print_comparison_report(reports, parser="mineru"):
    """Print side-by-side M6 metrics and question-level reranking changes."""
    dense_report = reports["dense"]
    reranker_report = reports["reranker"]
    print("PDF Retrieval Comparison")
    print(f"Parser: {parser}")
    print(f"Pages represented: {dense_report['pages']}")
    print(f"Documents: {dense_report['documents']}")
    print(f"Chunks: {dense_report['chunks']}")
    print(
        f"Questions: {dense_report['total']} "
        f"(evaluated: {dense_report['evaluated_total']}; "
        f"review required: {dense_report['review_required_total']}; "
        f"not applicable: {dense_report['not_applicable_total']})"
    )
    print(f"Reranker: dense Top-{RERANK_CANDIDATE_K} candidates -> Top-{TOP_K}")

    metric_rows = (
        ("Top-1 source hit", "top1_source_hits", "top1_source_total"),
        ("Top-3 source hit", "top3_source_hits", "top3_source_total"),
        ("Top-1 page hit", "top1_page_hits", "top1_page_total"),
        ("Top-3 page hit", "top3_page_hits", "top3_page_total"),
        ("Raw evidence hit", "keyword_hits", "keyword_total"),
        (
            "Normalized evidence hit",
            "normalized_keyword_hits",
            "normalized_keyword_total",
        ),
    )
    print("\nMetric comparison:")
    print(f"{'Metric':<28} {'Baseline Dense':>16} {'Dense + Reranker':>20}")
    for label, hits_key, total_key in metric_rows:
        dense_value = _format_metric(
            "", dense_report[hits_key], dense_report[total_key]
        ).split(": ", 1)[1]
        reranker_value = _format_metric(
            "", reranker_report[hits_key], reranker_report[total_key]
        ).split(": ", 1)[1]
        print(f"{label:<28} {dense_value:>16} {reranker_value:>20}")

    print("\nFailure attribution (baseline -> reranker):")
    for layer, dense_count in dense_report["error_layer_counts"].items():
        reranker_count = reranker_report["error_layer_counts"][layer]
        print(f"- {layer}: {dense_count} -> {reranker_count}")

    dense_cases = dense_report["cases"]
    reranker_cases = reranker_report["cases"]
    ranking_cases = [
        (dense_case, reranker_case)
        for dense_case, reranker_case in zip(dense_cases, reranker_cases)
        if dense_case["error_layer"] == "RANKING"
    ]
    ranking_rescues = sum(
        _case_evidence_rank(dense_case) is not None
        and _case_evidence_rank(dense_case) <= DIAGNOSTIC_K
        and _case_evidence_rank(dense_case) > TOP_K
        and _case_evidence_rank(reranker_case) is not None
        and _case_evidence_rank(reranker_case) <= TOP_K
        for dense_case, reranker_case in ranking_cases
    )
    print(
        f"\nOriginal dense RANKING cases rescued into evidence Top-{TOP_K}: "
        f"{ranking_rescues}/{len(ranking_cases)}"
    )

    changed_cases = []
    compared_flags = (
        ("top1_source_hit", "Top-1 source"),
        ("top3_source_hit", "Top-3 source"),
        ("top1_page_hit", "Top-1 page"),
        ("top3_page_hit", "Top-3 page"),
        ("keyword_hit", "Raw evidence"),
        ("normalized_keyword_hit", "Normalized evidence"),
    )
    for dense_case, reranker_case in zip(dense_cases, reranker_cases):
        dense_evidence_rank = _case_evidence_rank(dense_case)
        reranker_evidence_rank = _case_evidence_rank(reranker_case)
        dense_page_rank = _case_page_rank(dense_case)
        reranker_page_rank = _case_page_rank(reranker_case)
        changes = [
            f"{label}: {_hit_text(dense_case[key])} -> {_hit_text(reranker_case[key])}"
            for key, label in compared_flags
            if dense_case[key] != reranker_case[key]
        ]
        if (
            changes
            or dense_evidence_rank != reranker_evidence_rank
            or dense_page_rank != reranker_page_rank
        ):
            changed_cases.append(
                (
                    dense_case,
                    dense_evidence_rank,
                    reranker_evidence_rank,
                    dense_page_rank,
                    reranker_page_rank,
                    len(reranker_case["diagnostic_results"]),
                    changes,
                )
            )

    print("\nQuestions with ranking or hit changes:")
    if not changed_cases:
        print("No changes.")
    for index, (
        dense_case,
        dense_evidence_rank,
        reranker_evidence_rank,
        dense_page_rank,
        reranker_page_rank,
        reranker_limit,
        changes,
    ) in enumerate(changed_cases, start=1):
        dense_limit = len(dense_case["diagnostic_results"])
        print(f"\n{index}. Question: {dense_case['question']}")
        print(
            "   Evidence rank: "
            f"dense {_rank_text(dense_evidence_rank, dense_limit)} -> "
            f"reranker {_rank_text(reranker_evidence_rank, reranker_limit)}"
        )
        print(
            "   Page rank: "
            f"dense {_rank_text(dense_page_rank, dense_limit)} -> "
            f"reranker {_rank_text(reranker_page_rank, reranker_limit)}"
        )
        if changes:
            for change in changes:
                print(f"   {change}")


def print_report(report, parser="pymupdf", mode="dense"):
    """Print overall, category, and failure details for a retrieval run."""
    print("PDF Retrieval Evaluation")
    print(f"Parser: {parser}")
    if mode == "reranker":
        print(f"Retrieval mode: dense Top-{RERANK_CANDIDATE_K} + reranker Top-{TOP_K}")
    print(f"Pages represented: {report['pages']}")
    print(f"Documents: {report['documents']}")
    print(f"Chunks: {report['chunks']}")
    print(f"Load/parse time (cache included): {report['parse_seconds']:.2f}s")
    if report.get("representation_reference_loaded"):
        print(
            "Representation diagnostic reference: flat MinerU documents "
            f"({report['representation_reference_seconds']:.2f}s; no embeddings)"
        )
    print(f"Empty documents: {report['empty_documents']}")
    print(f"Representation: {report['representation']}")
    print(f"Indexed text characters: {report['text_characters']}")
    print(
        f"Questions: {report['total']} "
        f"(evaluated: {report['evaluated_total']}; "
        f"review required: {report['review_required_total']}; "
        f"not applicable to retrieval: {report['not_applicable_total']})"
    )
    print(_format_metric("Top-1 page hit", report["top1_page_hits"], report["top1_page_total"]))
    print(_format_metric("Top-3 page hit", report["top3_page_hits"], report["top3_page_total"]))
    print(_format_metric("Top-1 source hit", report["top1_source_hits"], report["top1_source_total"]))
    print(_format_metric("Top-3 source hit", report["top3_source_hits"], report["top3_source_total"]))
    print(_format_metric("Evidence/keyword hit", report["keyword_hits"], report["keyword_total"]))
    print(
        _format_metric(
            "Normalized evidence hit",
            report["normalized_keyword_hits"],
            report["normalized_keyword_total"],
        )
    )
    print("Generation: not evaluated")
    print("Error layers:")
    for layer, count in report["error_layer_counts"].items():
        print(f"- {layer}: {count}")

    category_names = sorted({case["category"] for case in report["cases"]})
    if category_names:
        print("\nBy category:")
        for category in category_names:
            cases = [case for case in report["cases"] if case["category"] == category]
            page_cases = [case for case in cases if case["top3_page_hit"] is not None]
            keyword_cases = [case for case in cases if case["keyword_hit"] is not None]
            normalized_keyword_cases = [
                case for case in cases if case["normalized_keyword_hit"] is not None
            ]
            page_hits = sum(case["top3_page_hit"] is True for case in page_cases)
            keyword_hits = sum(case["keyword_hit"] is True for case in keyword_cases)
            normalized_keyword_hits = sum(
                case["normalized_keyword_hit"] is True
                for case in normalized_keyword_cases
            )
            page_metric = f"page Top-3 {page_hits}/{len(page_cases)}" if page_cases else "page Top-3 n/a"
            keyword_metric = (
                f"keyword {keyword_hits}/{len(keyword_cases)}"
                if keyword_cases
                else "keyword n/a"
            )
            normalized_keyword_metric = (
                f"normalized {normalized_keyword_hits}/{len(normalized_keyword_cases)}"
                if normalized_keyword_cases
                else "normalized n/a"
            )
            sample_size = len(page_cases)
            sample_note = "; small sample" if sample_size < 5 else ""
            print(
                f"- {category}: n={sample_size}; {page_metric}; {keyword_metric}; "
                f"{normalized_keyword_metric}{sample_note}"
            )

    failures = [
        case
        for case in report["cases"]
        if case["error_layer"] is not None
        and not case["review_required"]
        and not case["not_applicable"]
    ]
    if failures:
        print("\nFailed cases:")
        for index, case in enumerate(failures, start=1):
            print(f"\n{index}.")
            print(f"Question:\n{case['question']}")
            if case["source"] is not None:
                print(f"Expected source:\n{case['source']}")
            if case["expected_page"] is not None:
                print(f"Expected page:\n{case['expected_page']}")
            print(f"Error layer:\n{case['error_layer']}")
            print(f"Reason:\n{case['reason']}")
            print("Top results:")
            for rank, result in enumerate(case["results"], start=1):
                page = result.get("metadata", {}).get("page")
                page_text = f", page {page}" if page is not None else ""
                print(f"{rank}. {result['source']}{page_text}")
            if not case["results"]:
                print("No results")

    review_cases = [case for case in report["cases"] if case["review_required"]]
    if review_cases:
        print("\nReview required:")
        for index, case in enumerate(review_cases, start=1):
            print(f"{index}. {case['question']}")


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate PDF retrieval with dense search or dense + reranker."
    )
    parser.add_argument(
        "--mode",
        choices=("dense", "reranker", "compare"),
        default="dense",
        help="Retrieval mode (default: dense baseline)",
    )
    parser.add_argument(
        "--failure-analysis-output",
        type=Path,
        help="Write anonymous per-case ranks under the Git-ignored outputs/ directory",
    )
    parser.add_argument(
        "--reranker-model",
        default=DEFAULT_RERANKER_MODEL,
        help=f"Hugging Face cross-encoder model (default: {DEFAULT_RERANKER_MODEL})",
    )
    parser.add_argument("--pdf", type=Path, help="Evaluate one PDF file")
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=PDF_DIR,
        help="Directory used by the legacy multi-PDF dataset",
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument(
        "--parser",
        choices=("pymupdf", "mineru"),
        default="pymupdf",
        help="Parser to evaluate (default: pymupdf baseline)",
    )
    parser.add_argument(
        "--representation",
        choices=("structured", "structured_ocr", "structured_full", "flat"),
        default="structured",
        help=(
            "MinerU representation to evaluate (default: structured; "
            "structured_ocr labels existing image text and structured_full "
            "also preserves block geometry metadata)"
        ),
    )
    parser.add_argument("--force-parse", action="store_true")
    parser.add_argument("--mineru-runner", type=Path)
    return parser


def main():
    argument_parser = build_argument_parser()
    args = argument_parser.parse_args()
    if args.failure_analysis_output is not None:
        if args.mode != "compare":
            argument_parser.error("--failure-analysis-output requires --mode compare")
        if not _failure_analysis_path_is_ignored(args.failure_analysis_output):
            argument_parser.error(
                "--failure-analysis-output must be inside the Git-ignored outputs/ directory"
            )

    try:
        questions = load_questions(args.dataset)
        parse_started = time.perf_counter()
        documents = load_pdf_documents(
            args.pdf_dir,
            questions,
            pdf_path=args.pdf,
            parser=args.parser,
            force=args.force_parse,
            runner_path=args.mineru_runner,
            representation=args.representation,
        )
        parse_seconds = time.perf_counter() - parse_started
        representation_reference_documents = None
        representation_reference_seconds = 0.0
        if args.parser == "mineru" and args.representation == "structured":
            reference_started = time.perf_counter()
            representation_reference_documents = load_pdf_documents(
                args.pdf_dir,
                questions,
                pdf_path=args.pdf,
                parser=args.parser,
                force=False,
                runner_path=args.mineru_runner,
                representation="flat",
            )
            representation_reference_seconds = time.perf_counter() - reference_started
        chunks = split_documents(documents)
        if chunks:
            model = load_model()
            embeddings = embed_chunks(chunks, model)
        else:
            model = None
            embeddings = []
        reranker_model = (
            load_reranker(args.reranker_model)
            if args.mode in {"reranker", "compare"} and chunks
            else None
        )
        common_arguments = {
            "documents": documents,
            "representation_reference_documents": representation_reference_documents,
        }
        if args.mode == "compare":
            reports = evaluate_comparison(
                questions,
                model,
                chunks,
                embeddings,
                reranker_model,
                **common_arguments,
            )
        else:
            report = evaluate_questions(
                questions,
                model,
                chunks,
                embeddings,
                mode=args.mode,
                reranker_model=reranker_model,
                **common_arguments,
            )
            reports = {args.mode: report}
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    report_statistics = {
        "pages": len(
            {
                (
                    (document.get("metadata") or {}).get("source"),
                    (document.get("metadata") or {}).get("page"),
                )
                for document in documents
            }
        ),
        "documents": len(documents),
        "chunks": len(chunks),
        "parse_seconds": parse_seconds,
        "representation_reference_loaded": representation_reference_documents is not None,
        "representation_reference_seconds": representation_reference_seconds,
        "empty_documents": sum(
            not document.get("text", "").strip() for document in documents
        ),
        "text_characters": sum(len(document.get("text", "")) for document in documents),
        "representation": args.representation if args.parser == "mineru" else "n/a",
    }
    for report in reports.values():
        report.update(report_statistics)
        report["parser"] = args.parser

    if args.mode == "compare":
        print_comparison_report(reports, parser=args.parser)
    else:
        print_report(reports[args.mode], parser=args.parser, mode=args.mode)
    if args.failure_analysis_output is not None:
        analysis = build_failure_analysis(reports)
        _write_failure_analysis(args.failure_analysis_output, analysis)
        for classification, count in analysis["classification_counts"].items():
            print(f"{classification}: {count}")
        print(f"Anonymous failure analysis saved to {args.failure_analysis_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
