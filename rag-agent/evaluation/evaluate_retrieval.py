"""Evaluate Top-1 and Top-3 source retrieval using a fixed QA dataset."""

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = Path(__file__).resolve().parent / "qa_dataset.json"
TOP_K = 3

# Allow `python evaluation/evaluate_retrieval.py` from the project root.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import (  # noqa: E402
    embed_chunks,
    load_documents,
    load_model,
    retrieve,
    split_documents,
)


def load_questions(path=DATASET_PATH):
    """Read and validate the fixed question/source dataset."""
    questions = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(questions, list) or not questions:
        raise ValueError("The evaluation dataset must be a non-empty JSON array.")

    for index, item in enumerate(questions, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Question {index} must be a JSON object.")
        if not isinstance(item.get("question"), str) or not item["question"].strip():
            raise ValueError(f"Question {index} must include a non-empty question.")
        if not isinstance(item.get("expected_source"), str) or not item[
            "expected_source"
        ].strip():
            raise ValueError(
                f"Question {index} must include a non-empty expected_source."
            )
    return questions


def evaluate_questions(questions, model, chunks, embeddings, top_k=TOP_K):
    """Count exact expected-source hits at rank 1 and within Top-K."""
    results_by_question = []

    for item in questions:
        query_embedding = model.encode(
            item["question"],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        results = retrieve(query_embedding, chunks, embeddings, top_k=top_k)
        expected_source = item["expected_source"]
        results_by_question.append(
            {
                "question": item["question"],
                "top1_hit": bool(
                    results and results[0]["source"] == expected_source
                ),
                "topk_hit": any(
                    result["source"] == expected_source for result in results
                ),
            }
        )

    total = len(results_by_question)
    return {
        "total": total,
        "top1_hits": sum(item["top1_hit"] for item in results_by_question),
        "topk_hits": sum(item["topk_hit"] for item in results_by_question),
    }


def main():
    try:
        questions = load_questions()
        documents = load_documents()
        chunks = split_documents(documents)
        if not chunks:
            raise ValueError("No non-empty Markdown chunks were found.")
        model = load_model()
        embeddings = embed_chunks(chunks, model)
        report = evaluate_questions(questions, model, chunks, embeddings)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Evaluation Result")
    print()
    print(f"Questions: {report['total']}")
    print(f"Top-1 Accuracy: {report['top1_hits']}/{report['total']}")
    print(f"Top-3 Accuracy: {report['topk_hits']}/{report['total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
