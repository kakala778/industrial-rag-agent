"""Tests for the optional dense-candidate reranking experiment."""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np

from evaluation.evaluate_pdf_retrieval import (
    build_argument_parser,
    evaluate_comparison,
    evaluate_questions,
    print_comparison_report,
)
from src.pdf_rag_demo import build_argument_parser as build_pdf_rag_argument_parser
from src.reranker import rerank


class FakeReranker:
    def __init__(self, scores):
        self.scores = scores
        self.pairs = None

    def predict(self, pairs, *, show_progress_bar):
        self.pairs = pairs
        self.assert_progress_bar = show_progress_bar
        return np.asarray(self.scores, dtype=np.float32)


class FixedEmbeddingModel:
    def encode(self, _text, **_kwargs):
        return np.array([1.0, 0.0], dtype=np.float32)


class TargetFirstReranker:
    def predict(self, pairs, *, show_progress_bar):
        self.pairs = pairs
        self.show_progress_bar = show_progress_bar
        return np.asarray(
            [1.0 if "TARGET-EVIDENCE" in text else 0.0 for _, text in pairs],
            dtype=np.float32,
        )


class RerankerTests(unittest.TestCase):
    def test_rerank_orders_candidates_and_preserves_metadata(self):
        candidates = [
            {
                "source": "first.pdf",
                "chunk_id": 1,
                "text": "First candidate",
                "score": 0.9,
                "metadata": {"page": 1, "block_type": "text"},
            },
            {
                "source": "second.pdf",
                "chunk_id": 2,
                "text": "Second candidate",
                "score": 0.7,
                "metadata": {"page": 2, "block_type": "table"},
            },
            {
                "source": "third.pdf",
                "chunk_id": 3,
                "text": "Third candidate",
                "score": 0.8,
                "metadata": {"page": 3, "block_type": "text"},
            },
        ]
        original_candidates = [dict(candidate) for candidate in candidates]
        model = FakeReranker([0.1, 0.95, 0.5])

        results = rerank("question", candidates, model, top_k=2)

        self.assertEqual([item["source"] for item in results], ["second.pdf", "third.pdf"])
        self.assertAlmostEqual(results[0]["reranker_score"], 0.95)
        self.assertAlmostEqual(results[1]["reranker_score"], 0.5)
        self.assertEqual(results[0]["score"], 0.7)
        self.assertEqual(results[0]["metadata"], {"page": 2, "block_type": "table"})
        self.assertEqual(
            model.pairs,
            [
                ("question", "First candidate"),
                ("question", "Second candidate"),
                ("question", "Third candidate"),
            ],
        )
        self.assertFalse(model.assert_progress_bar)
        self.assertEqual(candidates, original_candidates)

    def test_empty_candidates_return_without_model_inference(self):
        model = FakeReranker([])

        self.assertEqual(rerank("question", [], model), [])
        self.assertIsNone(model.pairs)

    def test_non_positive_top_k_returns_empty_without_model_inference(self):
        model = FakeReranker([0.5])
        candidate = {"source": "manual.pdf", "chunk_id": 1, "text": "Content"}

        self.assertEqual(rerank("question", [candidate], model, top_k=0), [])
        self.assertIsNone(model.pairs)

    def test_evaluation_defaults_to_dense_and_keeps_baseline_results(self):
        parser = build_argument_parser()
        self.assertEqual(parser.parse_args([]).mode, "dense")
        self.assertEqual(parser.parse_args(["--mode", "compare"]).mode, "compare")
        pdf_parser = build_pdf_rag_argument_parser()
        self.assertEqual(
            pdf_parser.parse_args(["manual.pdf", "--mode", "reranker"]).mode,
            "reranker",
        )

        questions = [
            {
                "question": "What is the value?",
                "expected_source": "manual.pdf",
                "expected_page": 1,
                "expected_keywords": ["0.82 MPa"],
            }
        ]
        chunks = [
            {
                "source": "manual.pdf",
                "chunk_id": 0,
                "text": "The specified value is 0.82 MPa.",
                "metadata": {"source": "manual.pdf", "page": 1},
            }
        ]

        with patch("evaluation.evaluate_pdf_retrieval.rerank") as rerank_mock:
            report = evaluate_questions(
                questions,
                FixedEmbeddingModel(),
                chunks,
                np.array([[1.0, 0.0]], dtype=np.float32),
            )

        self.assertEqual(report["top1_page_hits"], 1)
        self.assertEqual(report["top3_page_hits"], 1)
        self.assertEqual(report["normalized_keyword_hits"], 1)
        rerank_mock.assert_not_called()

    def test_evaluation_reranks_only_dense_top20_and_can_promote_evidence(self):
        question = {
            "question": "Where is the target value?",
            "expected_source": "manual.pdf",
            "expected_page": 8,
            "expected_keywords": ["TARGET-EVIDENCE"],
        }
        chunks = []
        similarities = []
        for index in range(20):
            is_target = index == 7
            chunks.append(
                {
                    "source": "manual.pdf" if is_target else f"other-{index}.pdf",
                    "chunk_id": index,
                    "text": "TARGET-EVIDENCE is stated here." if is_target else f"Decoy {index}",
                    "metadata": {
                        "source": "manual.pdf" if is_target else f"other-{index}.pdf",
                        "page": 8 if is_target else index + 1,
                    },
                }
            )
            cosine = 1.0 - index * 0.01
            similarities.append([cosine, np.sqrt(1.0 - cosine**2)])

        reranker_model = TargetFirstReranker()
        report = evaluate_questions(
            [question],
            FixedEmbeddingModel(),
            chunks,
            np.asarray(similarities, dtype=np.float32),
            mode="reranker",
            reranker_model=reranker_model,
        )
        case = report["cases"][0]

        self.assertEqual(len(reranker_model.pairs), 20)
        self.assertEqual(len(case["diagnostic_results"]), 20)
        self.assertEqual(case["results"][0]["source"], "manual.pdf")
        self.assertEqual(case["top1_page_hit"], True)
        self.assertEqual(case["top3_page_hit"], True)
        self.assertEqual(case["normalized_keyword_hit"], True)

    def test_comparison_report_shows_metrics_and_evidence_rank_changes(self):
        question = {
            "question": "Where is the target value?",
            "expected_source": "manual.pdf",
            "expected_page": 8,
            "expected_keywords": ["TARGET-EVIDENCE"],
        }
        chunks = []
        embeddings = []
        for index in range(10):
            is_target = index == 7
            source = "manual.pdf" if is_target else f"other-{index}.pdf"
            chunks.append(
                {
                    "source": source,
                    "chunk_id": index,
                    "text": "TARGET-EVIDENCE is stated here." if is_target else f"Decoy {index}",
                    "metadata": {"source": source, "page": 8 if is_target else index + 1},
                }
            )
            cosine = 1.0 - index * 0.01
            embeddings.append([cosine, np.sqrt(1.0 - cosine**2)])

        reports = evaluate_comparison(
            [question],
            FixedEmbeddingModel(),
            chunks,
            np.asarray(embeddings, dtype=np.float32),
            TargetFirstReranker(),
        )
        for report in reports.values():
            report.update({"pages": 10, "documents": 10, "chunks": 10})

        output = io.StringIO()
        with redirect_stdout(output):
            print_comparison_report(reports)

        rendered = output.getvalue()
        self.assertIn("Baseline Dense", rendered)
        self.assertIn("Dense + Reranker", rendered)
        self.assertIn("Original dense RANKING cases rescued into evidence Top-3: 1/1", rendered)
        self.assertIn("dense 8 -> reranker 1", rendered)


if __name__ == "__main__":
    unittest.main()
