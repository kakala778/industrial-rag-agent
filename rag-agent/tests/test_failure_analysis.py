from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation import evaluate_pdf_retrieval as evaluation


class FixedEmbeddingModel:
    def encode(self, _text, **_kwargs):
        return np.array([1.0, 0.0], dtype=np.float32)


class TargetFirstReranker:
    def predict(self, pairs, *, show_progress_bar):
        return np.asarray(
            [1.0 if "PRIVATE-EVIDENCE" in text else 0.0 for _, text in pairs],
            dtype=np.float32,
        )


def result(text, source="private.pdf", page=7, chunk_id=0, block_index=2, score=0.5):
    value = {
        "source": source,
        "chunk_id": chunk_id,
        "score": score,
        "text": text,
        "metadata": {
            "source": source,
            "page": page,
            "block_type": "table",
            "block_index": block_index,
            "sensitive_metadata": "PRIVATE-METADATA",
        },
    }
    return value


class FailureAnalysisTests(unittest.TestCase):
    def test_comparison_retains_the_original_dense_candidate_order(self):
        chunks = [
            {
                "source": "private.pdf",
                "chunk_id": index,
                "text": "PRIVATE-EVIDENCE" if index == 3 else f"decoy-{index}",
                "metadata": {"source": "private.pdf", "page": 7},
            }
            for index in range(4)
        ]
        embeddings = np.asarray(
            [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3]],
            dtype=np.float32,
        )
        report = evaluation.evaluate_questions(
            [
                {
                    "question": "PRIVATE-QUESTION",
                    "expected_source": "private.pdf",
                    "expected_page": 7,
                    "expected_keywords": ["PRIVATE-EVIDENCE"],
                }
            ],
            FixedEmbeddingModel(),
            chunks,
            embeddings,
            mode="reranker",
            reranker_model=TargetFirstReranker(),
        )

        case = report["cases"][0]
        self.assertIn("dense_candidate_results", case)
        self.assertEqual(
            case["dense_candidate_results"][3]["text"], "PRIVATE-EVIDENCE"
        )
        self.assertEqual(case["diagnostic_results"][0]["text"], "PRIVATE-EVIDENCE")

    def test_failure_analysis_classifies_original_failures_without_raw_data(self):
        evidence = result("PRIVATE-EVIDENCE value", chunk_id=7, block_index=22)
        decoys = [
            result(f"decoy-{index}", chunk_id=index, block_index=index)
            for index in range(20)
        ]
        dense_cases = [
            {
                "question": "PRIVATE-QUESTION-FIXED",
                "answer": "PRIVATE-ANSWER",
                "source": "private.pdf",
                "expected_page": 7,
                "expected_keywords": ["PRIVATE-EVIDENCE"],
                "error_layer": "RANKING",
            },
            {
                "question": "PRIVATE-QUESTION-STILL-RANKING",
                "source": "private.pdf",
                "expected_page": 7,
                "expected_keywords": ["PRIVATE-EVIDENCE"],
                "error_layer": "RANKING",
            },
            {
                "question": "PRIVATE-QUESTION-RETRIEVAL",
                "source": "private.pdf",
                "expected_page": 7,
                "expected_keywords": ["PRIVATE-EVIDENCE"],
                "error_layer": "RETRIEVAL",
            },
            {
                "question": "PRIVATE-QUESTION-PARSING",
                "source": "private.pdf",
                "expected_page": 7,
                "expected_keywords": ["PRIVATE-EVIDENCE"],
                "error_layer": "PARSING",
            },
            {
                "question": "PRIVATE-QUESTION-SUCCESS",
                "source": "private.pdf",
                "expected_page": 7,
                "expected_keywords": ["PRIVATE-EVIDENCE"],
                "error_layer": None,
            },
        ]
        q1_candidates = decoys[:7] + [evidence] + decoys[8:20]
        q2_evidence = result("PRIVATE-EVIDENCE value", chunk_id=27, block_index=27)
        q2_candidates = decoys[:7] + [q2_evidence] + decoys[8:20]
        reranker_cases = [
            {
                "dense_candidate_results": q1_candidates,
                "diagnostic_results": [evidence] + decoys[:7] + decoys[8:20],
            },
            {
                "dense_candidate_results": q2_candidates,
                "diagnostic_results": decoys[:3] + [q2_evidence] + decoys[8:20],
            },
            {
                "dense_candidate_results": decoys[:20],
                "diagnostic_results": decoys[:20],
            },
            {
                "dense_candidate_results": decoys[:20],
                "diagnostic_results": decoys[:20],
            },
            {
                "dense_candidate_results": [evidence] + decoys[1:20],
                "diagnostic_results": [evidence] + decoys[1:20],
            },
        ]
        for case in reranker_cases:
            case.update(
                {
                    "source": "private.pdf",
                    "expected_page": 7,
                    "expected_keywords": ["PRIVATE-EVIDENCE"],
                }
            )
        reports = {
            "dense": {
                "cases": dense_cases,
                "parser": "mineru",
                "representation": "structured",
            },
            "reranker": {
                "cases": reranker_cases,
                "parser": "mineru",
                "representation": "structured",
            },
        }

        analysis = evaluation.build_failure_analysis(
            reports, top_k=3, candidate_k=20
        )

        self.assertEqual(analysis["total_questions"], 5)
        self.assertEqual(analysis["total_original_failures"], 4)
        self.assertEqual(analysis["not_baseline_failure_count"], 1)
        self.assertEqual(
            analysis["classification_counts"],
            {
                "FIXED_BY_RERANKER": 1,
                "STILL_RANKING_FAILURE": 1,
                "RETRIEVAL_FAILURE": 1,
                "PARSING_FAILURE": 1,
                "INSUFFICIENT_DATA": 0,
            },
        )
        self.assertEqual(analysis["cases"][0]["question_id"], 1)
        self.assertEqual(analysis["cases"][0]["dense_evidence_rank"], 8)
        self.assertEqual(analysis["cases"][0]["reranker_evidence_rank"], 1)
        self.assertEqual(analysis["cases"][0]["evidence_rank_change"], 7)
        self.assertEqual(
            analysis["cases"][1]["classification"], "STILL_RANKING_FAILURE"
        )
        self.assertIsNone(analysis["cases"][2]["dense_evidence_rank"])
        self.assertEqual(analysis["cases"][4]["classification"], "NOT_A_BASELINE_FAILURE")

        fixed_case = analysis["cases"][0]
        self.assertEqual(fixed_case["dense_candidate_count"], 20)
        self.assertEqual(len(fixed_case["dense_top20"]), 20)
        self.assertEqual(fixed_case["reranked_top3"][0]["rank"], 1)
        self.assertEqual(fixed_case["reranked_top3"][0]["dense_rank"], 8)
        self.assertEqual(fixed_case["reranked_top3"][0]["rank_change"], 7)
        self.assertEqual(fixed_case["dense_top20"][7]["metadata"]["page"], 7)
        self.assertEqual(fixed_case["dense_top20"][7]["metadata"]["block_index"], 22)
        self.assertEqual(fixed_case["dense_top20"][7]["metadata"]["source_id"], "S001")

        serialized = json.dumps(analysis, ensure_ascii=False)
        for private_value in (
            "PRIVATE-QUESTION",
            "PRIVATE-ANSWER",
            "PRIVATE-EVIDENCE",
            "private.pdf",
            "PRIVATE-METADATA",
            "decoy-",
            '"text"',
            '"source"',
        ):
            self.assertNotIn(private_value, serialized)

    def test_analysis_output_argument_is_limited_to_ignored_outputs(self):
        args = evaluation.build_argument_parser().parse_args(
            [
                "--mode",
                "compare",
                "--failure-analysis-output",
                "outputs/m7-analysis.local.json",
            ]
        )

        self.assertEqual(args.mode, "compare")
        self.assertTrue(
            evaluation._failure_analysis_path_is_ignored(args.failure_analysis_output)
        )
        self.assertFalse(
            evaluation._failure_analysis_path_is_ignored(
                evaluation.PROJECT_ROOT / "docs" / "m7-analysis.json"
            )
        )


    def test_anonymous_analysis_is_written_as_valid_json(self):
        analysis = {
            "total_original_failures": 1,
            "classification_counts": {"FIXED_BY_RERANKER": 1},
            "cases": [
                {
                    "question_id": "Q01",
                    "classification": "FIXED_BY_RERANKER",
                    "dense_evidence_rank": 8,
                    "reranker_evidence_rank": 1,
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "analysis.json"
            evaluation._write_failure_analysis(path, analysis)
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(saved, analysis)

    def test_parsing_failure_requires_a_mineru_structured_evidence_check(self):
        dense_case = {
            "question": "PRIVATE-QUESTION",
            "source": "private.pdf",
            "expected_page": 7,
            "expected_keywords": ["PRIVATE-EVIDENCE"],
            "error_layer": "PARSING",
        }
        reports = {
            "dense": {"cases": [dense_case], "parser": "pymupdf", "representation": "structured"},
            "reranker": {
                "cases": [{"dense_candidate_results": [], "diagnostic_results": []}],
                "parser": "pymupdf",
                "representation": "structured",
            },
        }

        analysis = evaluation.build_failure_analysis(reports)

        self.assertEqual(analysis["classification_counts"]["PARSING_FAILURE"], 0)
        self.assertEqual(analysis["classification_counts"]["INSUFFICIENT_DATA"], 1)
        self.assertEqual(analysis["cases"][0]["classification"], "INSUFFICIENT_DATA")

    def test_evidence_beyond_the_configured_candidate_limit_is_retrieval_failure(self):
        evidence = result("PRIVATE-EVIDENCE", chunk_id=9)
        decoys = [result(f"decoy-{index}", chunk_id=index) for index in range(3)]
        reports = {
            "dense": {
                "cases": [
                    {
                        "question": "PRIVATE-QUESTION",
                        "source": "private.pdf",
                        "expected_page": 7,
                        "expected_keywords": ["PRIVATE-EVIDENCE"],
                        "error_layer": "RANKING",
                    }
                ],
                "parser": "mineru",
                "representation": "structured",
            },
            "reranker": {
                "cases": [
                    {
                        "source": "private.pdf",
                        "expected_page": 7,
                        "expected_keywords": ["PRIVATE-EVIDENCE"],
                        "dense_candidate_results": decoys + [evidence],
                        "diagnostic_results": [evidence] + decoys,
                    }
                ],
                "parser": "mineru",
                "representation": "structured",
            },
        }

        analysis = evaluation.build_failure_analysis(
            reports, top_k=3, candidate_k=3
        )

        self.assertEqual(analysis["cases"][0]["dense_evidence_rank"], None)
        self.assertEqual(analysis["cases"][0]["classification"], "RETRIEVAL_FAILURE")


if __name__ == "__main__":
    unittest.main()
