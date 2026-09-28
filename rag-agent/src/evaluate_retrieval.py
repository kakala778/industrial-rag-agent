"""Batch evaluation for the M1 Markdown retrieval demo."""

import json
import sys
from pathlib import Path

if __package__:
    from .retrieval_demo import (
        embed_chunks,
        load_documents,
        load_model,
        retrieve,
        split_documents,
    )
else:
    from retrieval_demo import (
        embed_chunks,
        load_documents,
        load_model,
        retrieve,
        split_documents,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTSET_PATH = PROJECT_ROOT / "tests" / "retrieval_test.json"
TOP_K = 3


def load_test_cases(path=TESTSET_PATH):
    """Load and lightly validate the manually curated retrieval test set."""
    test_cases = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(test_cases, list):
        raise ValueError("The retrieval test set must be a JSON array.")
    if not test_cases:
        raise ValueError(f"The retrieval test set is empty: {path}")

    required = ("question", "expected_source", "expected_keywords")
    for index, case in enumerate(test_cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Test case {index} must be a JSON object.")
        for field in required:
            if field not in case:
                raise ValueError(f"Test case {index} is missing '{field}'.")
        if not isinstance(case["question"], str) or not isinstance(
            case["expected_source"], str
        ):
            raise ValueError(
                f"Test case {index} question and expected_source must be strings."
            )
        keywords = case["expected_keywords"]
        if not isinstance(keywords, list) or not all(
            isinstance(keyword, str) for keyword in keywords
        ):
            raise ValueError(
                f"Test case {index} expected_keywords must be a list of strings."
            )
    return test_cases


def evaluate_cases(test_cases, model, chunks, embeddings, top_k=TOP_K):
    """Embed each question and score it with the existing retrieval function."""
    evaluated_cases = []
    retrieved_scores = []

    for case in test_cases:
        query_embedding = model.encode(
            case["question"],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        results = retrieve(query_embedding, chunks, embeddings, top_k=top_k)
        top1_source_hit = bool(
            results and results[0]["source"] == case["expected_source"]
        )
        top3_source_hit = any(
            result["source"] == case["expected_source"] for result in results[:top_k]
        )
        top_results_text = "\n".join(result["text"] for result in results[:top_k])
        keyword_hit = all(
            keyword in top_results_text for keyword in case["expected_keywords"]
        )

        retrieved_scores.extend(result["score"] for result in results)
        evaluated_cases.append(
            {
                "question": case["question"],
                "expected_source": case["expected_source"],
                "top1_source_hit": top1_source_hit,
                "top3_source_hit": top3_source_hit,
                "keyword_hit": keyword_hit,
                "results": results,
            }
        )

    total = len(evaluated_cases)
    # Average over every result returned across all questions (up to top_k each).
    average_score = (
        sum(retrieved_scores) / len(retrieved_scores) if retrieved_scores else 0.0
    )
    return {
        "cases": evaluated_cases,
        "total": total,
        "top1_source_hits": sum(case["top1_source_hit"] for case in evaluated_cases),
        "top3_source_hits": sum(case["top3_source_hit"] for case in evaluated_cases),
        "keyword_hits": sum(case["keyword_hit"] for case in evaluated_cases),
        "average_retrieved_score": average_score,
    }


def print_report(report, top_k=TOP_K):
    """Print aggregate metrics followed by cases with a Top-1 or keyword miss."""
    total = report["total"]
    print("================================")
    print("Retrieval Evaluation Result")
    print("================================")
    print()
    print(f"Total questions: {total}")
    print(f"Top1 source hit: {report['top1_source_hits']}/{total}")
    print(f"Top3 source hit: {report['top3_source_hits']}/{total}")
    print(f"Keyword hit: {report['keyword_hits']}/{total}")
    print(f"Average retrieved score: {report['average_retrieved_score']:.4f}")

    failures = [
        case
        for case in report["cases"]
        if not case["top1_source_hit"] or not case["keyword_hit"]
    ]
    if failures:
        print("\nFailure cases (Top1 source or keyword miss):")
        for case in failures:
            print(f"\nQuestion:\n{case['question']}")
            print(f"\nExpected source:\n{case['expected_source']}")
            print("\nTop results:")
            for rank, result in enumerate(case["results"][:top_k], start=1):
                print(f"{rank}. source {result['source']}")
            if not case["results"]:
                print("No results")


def main():
    try:
        test_cases = load_test_cases()
        documents = load_documents()
        chunks = split_documents(documents)
        if not chunks:
            raise ValueError("No non-empty Markdown chunks were found.")
        model = load_model()
        embeddings = embed_chunks(chunks, model)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    report = evaluate_cases(test_cases, model, chunks, embeddings, top_k=TOP_K)
    print_report(report, top_k=TOP_K)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
