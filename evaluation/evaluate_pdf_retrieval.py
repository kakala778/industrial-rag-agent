"""Compare PDF parser paths using retrieval-only QA metrics."""

import argparse
import json
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
):
    """Retrieve and score one question, including failure attribution."""
    expected_source = _expected_source(item)
    expected_page = item.get("expected_page")
    expected_keywords = item.get("expected_keywords", [])
    category = item.get("category", item.get("type", "text"))

    if chunks:
        query_embedding = model.encode(
            item["question"],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        results = retrieve(
            query_embedding,
            chunks,
            embeddings,
            top_k=max(top_k, DIAGNOSTIC_K),
        )
    else:
        results = []
    top_results = results[:top_k]
    diagnostic_results = results[:max(top_k, DIAGNOSTIC_K)]

    top1_source_hit = (
        None
        if expected_source is None
        else bool(results and results[0]["source"] == expected_source)
    )
    top3_source_hit = (
        None
        if expected_source is None
        else any(result["source"] == expected_source for result in top_results)
    )
    top1_page_hit = (
        None
        if expected_page is None
        else bool(results and _page_hit(results[0], expected_page, expected_source))
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
):
    """Measure source/page hits and raw keyword hits without evaluating generation."""
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
                )
            )

    return _summarize_cases(cases)


def _format_metric(name, hits, total):
    if not total:
        return f"{name}: n/a"
    return f"{name}: {hits}/{total}"


def print_report(report, parser="pymupdf"):
    """Print overall, category, and failure details for a retrieval run."""
    print("PDF Retrieval Evaluation")
    print(f"Parser: {parser}")
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
        description="Evaluate PDF retrieval with PyMuPDF or local MinerU."
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
        choices=("structured", "flat"),
        default="structured",
        help="MinerU representation to evaluate (default: structured)",
    )
    parser.add_argument("--force-parse", action="store_true")
    parser.add_argument("--mineru-runner", type=Path)
    return parser


def main():
    args = build_argument_parser().parse_args()

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
        report = evaluate_questions(
            questions,
            model,
            chunks,
            embeddings,
            documents=documents,
            representation_reference_documents=representation_reference_documents,
        )
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    report.update(
        {
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
    )
    print_report(report, parser=args.parser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
