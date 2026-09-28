"""Tests for the PDF retrieval evaluation's local industrial QA format."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from evaluation.evaluate_pdf_retrieval import (
    _diagnose_failure,
    build_argument_parser,
    evaluate_questions,
    load_pdf_documents,
    load_questions,
)


class FixedModel:
    def __init__(self):
        self.calls = 0

    def encode(self, _text, **_kwargs):
        self.calls += 1
        return np.array([1.0, 0.0], dtype=np.float32)


class PdfEvaluationTests(unittest.TestCase):
    def test_accepts_expanded_m6_categories(self):
        dataset = [
            {
                "question": "What is shown in the drawing?",
                "expected_source": "drawing.pdf",
                "expected_page": 1,
                "expected_keywords": ["DRAWING-VALUE-Z9"],
                "category": "drawing_layout",
            },
            {
                "question": "Which material belongs to this row?",
                "expected_source": "schedule.pdf",
                "expected_page": 2,
                "expected_keywords": ["MATERIAL-VALUE-Z9"],
                "category": "complex_table",
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "industrial_qa.json"
            path.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")

            questions = load_questions(path)

        self.assertEqual(
            [item["category"] for item in questions],
            ["drawing_layout", "complex_table"],
        )

    def test_rejects_multisource_dataset_when_one_pdf_is_selected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            selected_pdf = Path(temp_dir) / "selected.pdf"
            selected_pdf.write_bytes(b"pdf placeholder")
            questions = [
                {"question": "Q1", "expected_source": "selected.pdf"},
                {"question": "Q2", "expected_source": "another.pdf"},
            ]

            with patch("evaluation.evaluate_pdf_retrieval.load_pdf") as load_pdf:
                with self.assertRaisesRegex(ValueError, "single-source dataset"):
                    load_pdf_documents(
                        temp_dir,
                        questions,
                        pdf_path=selected_pdf,
                    )

        load_pdf.assert_not_called()

    def test_missing_documents_do_not_create_parsing_attribution(self):
        item = {
            "question": "Where is the expected limit stated?",
            "expected_source": "manual.pdf",
            "expected_page": 2,
            "expected_keywords": ["0.82 MPa"],
            "category": "numeric",
        }
        chunks = [
            {
                "source": "manual.pdf",
                "text": "The specified limit is 0.82 MPa.",
                "metadata": {"source": "manual.pdf", "page": 2},
            }
        ]

        layer, _reason = _diagnose_failure(item, None, chunks, [])

        self.assertEqual(layer, "RETRIEVAL")

    def test_missing_documents_with_incomplete_scoped_chunks_are_insufficient_data(self):
        item = {
            "question": "Where is the expected limit stated?",
            "expected_source": "manual.pdf",
            "expected_page": 2,
            "expected_keywords": ["0.82 MPa"],
            "category": "numeric",
        }
        chunks = [
            {
                "source": "manual.pdf",
                "text": "A chunk exists for the expected page but has no limit value.",
                "metadata": {"source": "manual.pdf", "page": 2},
            }
        ]

        layer, _reason = _diagnose_failure(item, None, chunks, [])

        self.assertEqual(layer, "INSUFFICIENT_DATA")

    def test_structured_miss_present_in_flat_reference_is_representation(self):
        item = {
            "question": "What is the specified row value?",
            "expected_source": "manual.pdf",
            "expected_page": 2,
            "expected_keywords": ["SAMPLE-MATERIAL-Z9", "SAMPLE-VALUE-Z9"],
            "category": "complex_table",
        }
        documents = [
            {
                "text": "The table row is present but its cells were not retained.",
                "metadata": {"source": "manual.pdf", "page": 2},
            }
        ]
        flat_documents = [
            {
                "text": (
                    "ASSEMBLY-TEST-A | SECTION-TEST-B | SAMPLE-MATERIAL-Z9 | "
                    "SAMPLE-VALUE-Z9 | COUNT-TEST-3"
                ),
                "metadata": {"source": "manual.pdf", "page": 2},
            }
        ]

        layer, _reason = _diagnose_failure(
            item,
            documents,
            [],
            [],
            representation_reference_documents=flat_documents,
        )

        self.assertEqual(layer, "REPRESENTATION")

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

    def test_normalized_evidence_metric_keeps_raw_keyword_metric(self):
        questions = [
            {
                "question": "What is the discharge current and insertion loss?",
                "source": "manual.pdf",
                "expected_page": 2,
                "expected_keywords": ["1.5kA", "0.5dB"],
                "category": "multi_fact",
            }
        ]
        chunks = [
            {
                "source": "manual.pdf",
                "chunk_id": 0,
                "text": "Video protector: 1.5 kA, insertion loss 0.5 dB.",
                "metadata": {"source": "manual.pdf", "page": 2},
            }
        ]
        embeddings = np.array([[1.0, 0.0]], dtype=np.float32)

        report = evaluate_questions(questions, FixedModel(), chunks, embeddings)

        self.assertEqual(report["keyword_hits"], 0)
        self.assertEqual(report["normalized_keyword_hits"], 1)
        self.assertFalse(report["cases"][0]["keyword_hit"])
        self.assertTrue(report["cases"][0]["normalized_keyword_hit"])

    def test_evidence_keywords_from_wrong_document_do_not_count(self):
        questions = [
            {
                "question": "What is the expected value?",
                "expected_source": "target.pdf",
                "expected_page": 2,
                "expected_keywords": ["0.82 MPa"],
                "category": "unit",
            }
        ]
        chunks = [
            {
                "source": "other.pdf",
                "chunk_id": 0,
                "text": "An unrelated specification says 0.82 MPa.",
                "metadata": {"source": "other.pdf", "page": 1},
            },
            {
                "source": "target.pdf",
                "chunk_id": 1,
                "text": "The target page contains unrelated notes.",
                "metadata": {"source": "target.pdf", "page": 2},
            },
        ]
        embeddings = np.array([[1.0, 0.0], [0.8, 0.6]], dtype=np.float32)

        report = evaluate_questions(questions, FixedModel(), chunks, embeddings)

        self.assertEqual(report["top3_source_hits"], 1)
        self.assertEqual(report["top3_page_hits"], 1)
        self.assertEqual(report["keyword_hits"], 0)
        self.assertEqual(report["normalized_keyword_hits"], 0)

    def test_normalized_evidence_does_not_join_text_across_chunks(self):
        question = {
            "question": "Does the token AB appear in one result?",
            "expected_source": "target.pdf",
            "expected_keywords": ["AB"],
            "category": "text",
        }
        chunks = [
            {
                "source": "target.pdf",
                "chunk_id": 0,
                "text": "A",
                "metadata": {"source": "target.pdf", "page": 1},
            },
            {
                "source": "target.pdf",
                "chunk_id": 1,
                "text": "B",
                "metadata": {"source": "target.pdf", "page": 1},
            },
        ]
        embeddings = np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)

        report = evaluate_questions(
            [question],
            FixedModel(),
            chunks,
            embeddings,
            documents=[
                {"text": "A\nB", "metadata": {"source": "target.pdf", "page": 1}}
            ],
        )

        self.assertEqual(report["keyword_hits"], 0)
        self.assertEqual(report["normalized_keyword_hits"], 0)

        separate_keywords_report = evaluate_questions(
            [{**question, "expected_keywords": ["A", "B"]}],
            FixedModel(),
            chunks,
            embeddings,
            documents=[
                {"text": "A\nB", "metadata": {"source": "target.pdf", "page": 1}}
            ],
        )

        self.assertEqual(separate_keywords_report["normalized_keyword_hits"], 1)

    def test_failure_diagnosis_separates_parsing_chunking_retrieval_and_ranking(self):
        question = {
            "question": "Which manual values apply?",
            "expected_source": "manual.pdf",
            "expected_page": 2,
            "expected_keywords": ["alpha", "beta"],
        }
        document = {
            "text": "alpha beta",
            "metadata": {"source": "manual.pdf", "page": 2},
        }
        evidence_chunk = {
            "source": "manual.pdf",
            "text": "alpha beta",
            "metadata": {"source": "manual.pdf", "page": 2},
        }
        decoy = {
            "source": "other.pdf",
            "text": "unrelated",
            "metadata": {"source": "other.pdf", "page": 1},
        }

        parsing, _ = _diagnose_failure(
            question,
            [{"text": "no evidence", "metadata": {"source": "manual.pdf", "page": 2}}],
            [],
            [],
        )
        chunking, _ = _diagnose_failure(
            question,
            [document],
            [
                {**evidence_chunk, "text": "alpha"},
                {**evidence_chunk, "text": "unrelated"},
            ],
            [],
        )
        split_evidence_ranking, _ = _diagnose_failure(
            question,
            [document],
            [
                {**evidence_chunk, "text": "alpha"},
                {**evidence_chunk, "text": "beta"},
            ],
            [decoy] * 3
            + [
                {**evidence_chunk, "text": "alpha"},
                {**evidence_chunk, "text": "beta"},
            ]
            + [decoy] * 5,
        )
        target_page_without_evidence = {
            "source": "manual.pdf",
            "text": "general introduction",
            "metadata": {"source": "manual.pdf", "page": 2},
        }
        target_page_without_top10_evidence, _ = _diagnose_failure(
            question,
            [document],
            [evidence_chunk],
            [decoy, target_page_without_evidence] + [decoy] * 8,
        )
        retrieval, _ = _diagnose_failure(
            question,
            [document],
            [evidence_chunk],
            [decoy] * 10,
        )
        ranking, _ = _diagnose_failure(
            question,
            [document],
            [evidence_chunk],
            [decoy] * 4 + [evidence_chunk] + [decoy] * 5,
        )
        top1_ranking, _ = _diagnose_failure(
            question,
            [document],
            [evidence_chunk],
            [decoy, evidence_chunk],
        )

        self.assertEqual(parsing, "PARSING")
        self.assertEqual(chunking, "CHUNKING")
        self.assertEqual(split_evidence_ranking, "RANKING")
        self.assertEqual(target_page_without_top10_evidence, "RETRIEVAL")
        self.assertEqual(retrieval, "RETRIEVAL")
        self.assertEqual(ranking, "RANKING")
        self.assertEqual(top1_ranking, "RANKING")

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

    def test_single_pdf_rejects_a_dataset_with_multiple_sources(self):
        questions = [
            {"question": "Question for the selected PDF?", "source": "target.pdf"},
            {"question": "Question for another PDF?", "source": "other.pdf"},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            selected_pdf = Path(temp_dir) / "target.pdf"
            selected_pdf.write_bytes(b"test PDF")
            with patch("evaluation.evaluate_pdf_retrieval.load_pdf") as load_pdf:
                with self.assertRaisesRegex(ValueError, "multiple sources"):
                    load_pdf_documents(
                        temp_dir,
                        questions,
                        pdf_path=selected_pdf,
                    )
                load_pdf.assert_not_called()

    def test_missing_documents_does_not_misattribute_a_ranking_miss_to_parsing(self):
        question = {
            "question": "Where is alpha documented?",
            "expected_source": "target.pdf",
            "expected_page": 1,
            "expected_keywords": ["alpha"],
        }
        chunks = [
            {
                "source": "other.pdf",
                "chunk_id": 0,
                "text": "unrelated",
                "metadata": {"source": "other.pdf", "page": 1},
            },
            {
                "source": "target.pdf",
                "chunk_id": 1,
                "text": "alpha is documented here",
                "metadata": {"source": "target.pdf", "page": 1},
            },
        ]
        embeddings = np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)

        report = evaluate_questions(
            [question],
            FixedModel(),
            chunks,
            embeddings,
            top_k=1,
        )

        self.assertEqual(report["cases"][0]["error_layer"], "RANKING")

    def test_missing_documents_reports_insufficient_data_when_chunk_evidence_is_absent(self):
        question = {
            "question": "Where is alpha documented?",
            "expected_source": "target.pdf",
            "expected_page": 1,
            "expected_keywords": ["alpha"],
        }
        chunks = [
            {
                "source": "target.pdf",
                "chunk_id": 0,
                "text": "target page with unrelated notes",
                "metadata": {"source": "target.pdf", "page": 1},
            }
        ]
        embeddings = np.array([[1.0, 0.0]], dtype=np.float32)

        report = evaluate_questions(
            [question],
            FixedModel(),
            chunks,
            embeddings,
        )

        self.assertEqual(
            report["cases"][0]["error_layer"], "INSUFFICIENT_DATA"
        )

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
        self.assertIsNone(report["cases"][0]["error_layer"])

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
