"""Tests for the PDF retrieval evaluation's local industrial QA format."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.evaluate_pdf_retrieval import (
    build_argument_parser,
    evaluate_questions,
    load_questions,
)


class FixedModel:
    def __init__(self):
        self.calls = 0

    def encode(self, _text, **_kwargs):
        self.calls += 1
        return np.array([1.0, 0.0], dtype=np.float32)


class PdfEvaluationTests(unittest.TestCase):
    def test_page_label_requires_expected_source(self):
        dataset = [
            {
                "question": "Which value appears on page 2?",
                "expected_page": 2,
                "expected_keywords": ["0.82 MPa"],
                "category": "numeric",
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "industrial_qa.json"
            path.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "expected_source"):
                load_questions(path)

    def test_loads_local_schema_without_source_and_allows_unanswerable_keywords(self):
        dataset = [
            {
                "question": "Is the requested limit stated?",
                "expected_page": None,
                "expected_keywords": [],
                "answer": "根据现有资料无法确定",
                "category": "unanswerable",
                "review_required": False,
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "industrial_qa.json"
            path.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")

            questions = load_questions(path)

        self.assertEqual(questions[0]["category"], "unanswerable")
        self.assertEqual(questions[0]["expected_keywords"], [])

    def test_reports_page_hits_separately_from_retrieval_rank(self):
        questions = [
            {
                "question": "What pressure is specified?",
                "source": "manual.pdf",
                "expected_page": 2,
                "expected_keywords": ["0.82 MPa"],
                "category": "numeric",
            }
        ]
        chunks = [
            {
                "source": "manual.pdf",
                "chunk_id": 0,
                "text": "General introduction.",
                "metadata": {"source": "manual.pdf", "page": 1},
            },
            {
                "source": "manual.pdf",
                "chunk_id": 1,
                "text": "Operating pressure: 0.82 MPa.",
                "metadata": {"source": "manual.pdf", "page": 2},
            },
            {
                "source": "manual.pdf",
                "chunk_id": 2,
                "text": "Maintenance notes.",
                "metadata": {"source": "manual.pdf", "page": 3},
            },
        ]
        embeddings = np.array([[1.0, 0.0], [0.8, 0.6], [0.6, 0.8]], dtype=np.float32)

        report = evaluate_questions(questions, FixedModel(), chunks, embeddings)

        self.assertEqual(report["top1_page_hits"], 0)
        self.assertEqual(report["top3_page_hits"], 1)
        self.assertEqual(report["keyword_hits"], 1)

    def test_cli_accepts_pdf_dataset_and_parser_selection(self):
        args = build_argument_parser().parse_args(
            [
                "--pdf",
                "manual.pdf",
                "--dataset",
                "qa.json",
                "--parser",
                "mineru",
            ]
        )

        self.assertEqual(args.pdf, Path("manual.pdf"))
        self.assertEqual(args.dataset, Path("qa.json"))
        self.assertEqual(args.parser, "mineru")

    def test_review_required_questions_are_not_evaluated_or_counted(self):
        questions = [
            {
                "question": "Unverified question",
                "expected_page": 4,
                "expected_keywords": ["unverified"],
                "category": "numeric",
                "review_required": True,
            }
        ]
        model = FixedModel()

        report = evaluate_questions(questions, model, [], np.empty((0, 2)))

        self.assertEqual(model.calls, 0)
        self.assertEqual(report["review_required_total"], 1)
        self.assertEqual(report["evaluated_total"], 0)
        self.assertEqual(report["keyword_total"], 0)

    def test_unanswerable_cases_without_retrieval_labels_are_not_scored(self):
        questions = [
            {
                "question": "Does the manual state this unsupported condition?",
                "expected_keywords": [],
                "category": "unanswerable",
                "answer": "根据现有资料无法确定",
                "review_required": False,
            }
        ]
        model = FixedModel()

        report = evaluate_questions(questions, model, [], np.empty((0, 2)))

        self.assertEqual(model.calls, 0)
        self.assertEqual(report["not_applicable_total"], 1)
        self.assertEqual(report["evaluated_total"], 0)

    def test_empty_baseline_chunks_report_misses_without_loading_a_model(self):
        questions = [
            {
                "question": "What pressure is specified?",
                "source": "scan.pdf",
                "expected_page": 1,
                "expected_keywords": ["0.82 MPa"],
                "category": "numeric",
            }
        ]
        documents = [
            {"text": "", "metadata": {"source": "scan.pdf", "page": 1}}
        ]

        report = evaluate_questions(questions, None, [], [], documents=documents)

        self.assertEqual(report["top1_page_hits"], 0)
        self.assertEqual(report["top3_page_hits"], 0)
        self.assertEqual(report["keyword_hits"], 0)


if __name__ == "__main__":
    unittest.main()
