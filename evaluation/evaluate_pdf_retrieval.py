"""Evaluate PDF retrieval against a small fixed question set."""

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "examples" / "pdf"
DATASET_PATH = Path(__file__).resolve().parent / "pdf_qa_dataset.json"
TOP_K = 3

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
    """Read and validate the fixed PDF question/source dataset."""
    questions = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(questions, list) or not questions:
        raise ValueError("The PDF evaluation dataset must be a non-empty JSON array.")

    for index, item in enumerate(questions, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Question {index} must be a JSON object.")
        for field in ("question", "source", "type", "expected_keywords"):
            if field not in item:
                raise ValueError(f"Question {index} is missing '{field}'.")
        for field in ("question", "source"):
            if not isinstance(item[field], str) or not item[field].strip():
                raise ValueError(f"Question {index} '{field}' must be a non-empty string.")
        if item["type"] not in ("text", "table"):
            raise ValueError(f"Question {index} type must be 'text' or 'table'.")
        keywords = item["expected_keywords"]
        if not isinstance(keywords, list) or not keywords or not all(
            isinstance(keyword, str) and keyword for keyword in keywords
        ):
            raise ValueError(
                f"Question {index} expected_keywords must be a non-empty list of strings."
            )
    return questions


def load_pdf_documents(pdf_dir, questions):
    """Load all local PDFs and require every dataset source to be present."""
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in: {pdf_dir}")

    expected_sources = {item["source"] for item in questions}
    available_sources = {path.name for path in pdf_paths}
    missing_sources = sorted(expected_sources - available_sources)
    if missing_sources:
        raise FileNotFoundError(
            "PDF files referenced by the dataset are missing from "
            f"{pdf_dir}: {', '.join(missing_sources)}"
        )

    documents = []
    for source in sorted(expected_sources):
        documents.extend(load_pdf(pdf_dir / source))
    return documents


def evaluate_questions(questions, model, chunks, embeddings, top_k=TOP_K):
    """Measure source hits and expected-keyword hits in retrieved Top-K text."""
    cases = []

    for item in questions:
        query_embedding = model.encode(
            item["question"],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        results = retrieve(query_embedding, chunks, embeddings, top_k=top_k)
        top_results = results[:top_k]
        expected_source = item["source"]
        top1_source_hit = bool(
            results and results[0]["source"] == expected_source
        )
        top3_source_hit = any(
            result["source"] == expected_source for result in top_results
        )
        retrieved_text = "\n".join(result["text"] for result in top_results).casefold()
        keyword_hit = all(
            keyword.casefold() in retrieved_text
            for keyword in item["expected_keywords"]
        )

        reasons = []
        if not top1_source_hit:
            reasons.append("expected source not ranked Top-1")
        if not top3_source_hit:
            reasons.append("expected source not found in Top-3")
        if not keyword_hit:
            if item["type"] == "table":
                reasons.append("table structure not retrieved; expected keywords are missing")
            else:
                reasons.append("expected keywords are missing from Top-3 text")

        cases.append(
            {
                "question": item["question"],
                "source": expected_source,
                "type": item["type"],
                "top1_source_hit": top1_source_hit,
                "top3_source_hit": top3_source_hit,
                "keyword_hit": keyword_hit,
                "results": top_results,
                "reason": "; ".join(reasons),
            }
        )

    total = len(cases)
    return {
        "cases": cases,
        "total": total,
        "top1_source_hits": sum(case["top1_source_hit"] for case in cases),
        "top3_source_hits": sum(case["top3_source_hit"] for case in cases),
        "keyword_hits": sum(case["keyword_hit"] for case in cases),
    }


def print_report(report):
    """Print aggregate metrics and retrieval failure details."""
    total = report["total"]
    print("PDF Retrieval Evaluation")
    print()
    print(f"Questions: {total}")
    print(f"Top-1 source hit: {report['top1_source_hits']}/{total}")
    print(f"Top-3 source hit: {report['top3_source_hits']}/{total}")
    print(f"Keyword hit: {report['keyword_hits']}/{total}")

    failures = [
        case
        for case in report["cases"]
        if not case["top1_source_hit"]
        or not case["top3_source_hit"]
        or not case["keyword_hit"]
    ]
    if failures:
        print("\nFailed cases:")
        for index, case in enumerate(failures, start=1):
            print(f"\n{index}.")
            print(f"Question:\n{case['question']}")
            print(f"Expected source:\n{case['source']}")
            print(f"Reason:\n{case['reason']}")
            print("Top results:")
            for rank, result in enumerate(case["results"], start=1):
                page = result.get("metadata", {}).get("page")
                page_text = f", page {page}" if page is not None else ""
                print(f"{rank}. {result['source']}{page_text}")
            if not case["results"]:
                print("No results")


def main():
    try:
        questions = load_questions()
        documents = load_pdf_documents(PDF_DIR, questions)
        chunks = split_documents(documents)
        if not chunks:
            raise ValueError("No non-empty PDF text chunks were found.")
        model = load_model()
        embeddings = embed_chunks(chunks, model)
        report = evaluate_questions(questions, model, chunks, embeddings)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
