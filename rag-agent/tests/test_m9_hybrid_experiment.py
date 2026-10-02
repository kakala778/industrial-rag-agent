"""Offline M9 orchestration, classification, and privacy contracts."""
import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch
from evaluation.evaluate_pdf_retrieval import evaluate_questions
from evaluation.run_m9_hybrid_retrieval_experiment import (
    MODES, build_argument_parser, selected_modes, classify_retrieval_case,
    build_anonymous_summary, query_features, safe_output_root,
)


class M9ExperimentTests(unittest.TestCase):
    def test_cli_default_is_dense_without_reranking(self):
        args = build_argument_parser().parse_args([])
        self.assertEqual(selected_modes(args), ("dense",))
        args = build_argument_parser().parse_args(["--retriever", "hybrid", "--rerank", "on"])
        self.assertEqual(selected_modes(args), ("hybrid_reranker",))
        self.assertEqual(selected_modes(build_argument_parser().parse_args(["--compare-all"])), MODES)
        with self.assertRaises(ValueError):
            selected_modes(build_argument_parser().parse_args(["--retriever", "bm25", "--rerank", "on"]))

    def test_candidate_recovery_does_not_imply_top3_success(self):
        self.assertEqual(classify_retrieval_case(True, 5, 9), "RESCUED_BY_HYBRID")
        self.assertEqual(classify_retrieval_case(True, 2, None), "RESCUED_BY_BM25")
        self.assertEqual(classify_retrieval_case(True, None, None), "STILL_RETRIEVAL_FAILURE")
        self.assertEqual(classify_retrieval_case(False, None, None), "PARSING_OR_GT_ISSUE")

    def test_features_are_generic_flags_not_tokens(self):
        features = query_features("压力 OTN-400G P-101 0.82MPa GB/T")
        self.assertTrue(features["identifier"])
        self.assertTrue(features["numeric"])
        self.assertTrue(features["cjk"])
        self.assertNotIn("otn", json.dumps(features))

    def test_output_cannot_escape_ignored_m9_directory(self):
        with self.assertRaises(ValueError):
            safe_output_root(Path.cwd() / "docs" / "m9")

    def test_anonymous_summary_and_recovery_use_original_matcher(self):
        source = "PRIVATE_SOURCE_NAME.pdf"
        questions = [{"question": "PRIVATE_QUESTION " + str(i), "answer": "PRIVATE_ANSWER",
                      "expected_source": source, "expected_page": 1,
                      "expected_keywords": ["PRIVATE_EVIDENCE"], "category": "numeric"}
                     for i in range(4)]
        candidate = {"text": "PRIVATE_EVIDENCE", "source": source, "chunk_id": 0,
                     "score": 1.0, "metadata": {"source": source, "page": 1,
                     "block_type": "table", "block_index": 0,
                     "caption": "PRIVATE_CAPTION", "secret": "PRIVATE_SECRET"}}
        chunks = [candidate]
        documents = [{"text": candidate["text"], "metadata": candidate["metadata"]}]
        reports = {}
        for mode in MODES:
            ranked = [] if mode in {"dense", "dense_reranker"} else [candidate]
            reports[mode] = evaluate_questions(questions, None, chunks, [], documents=documents,
                diagnostic_k=20, candidate_provider=lambda _q, _k, rows=ranked: rows)
        original = {"cases": [{"question_id": i + 1, "classification": "RETRIEVAL_FAILURE"} for i in range(4)]}
        summary = build_anonymous_summary(questions, reports, documents, chunks, original)
        self.assertEqual(summary["metrics"]["hybrid"]["recall_at20"], {"hits": 4, "total": 4})
        self.assertEqual(summary["transitions"]["normalized_evidence"]["gained_ids"], [1, 2, 3, 4])
        self.assertEqual(summary["transitions"]["joint_success"]["regression_ids"], [])
        self.assertEqual(summary["original_retrieval_cases"][0]["classification"], "RESCUED_BY_HYBRID")
        encoded = json.dumps(summary)
        for private in (source, "PRIVATE_QUESTION", "PRIVATE_ANSWER", "PRIVATE_EVIDENCE", "PRIVATE_CAPTION", "PRIVATE_SECRET"):
            self.assertNotIn(private, encoded)

    def test_missing_mineru_cache_stops_without_invoking_parser(self):
        from evaluation import run_m9_hybrid_retrieval_experiment as runner
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "fixture.pdf").write_bytes(b"fixture")
            analysis = root / "analysis.json"
            analysis.write_text(json.dumps({"cases": [{"question_id": i, "classification": "RETRIEVAL_FAILURE"} for i in range(1, 5)]}))
            args = build_argument_parser().parse_args([])
            args.pdf_dir = root
            args.m7_analysis = analysis
            questions = [{"question": "fixture", "source": "fixture.pdf"} for _ in range(4)]
            with patch.object(runner, "load_questions", return_value=questions), \
                 patch.object(runner, "require_matching_input_manifest"), \
                 patch.object(runner, "_cached_middle_json", return_value=None), \
                 patch("src.mineru_loader._run_mineru") as parser:
                with self.assertRaisesRegex(RuntimeError, "stopped without parsing"):
                    runner.load_frozen_inputs(args)
            parser.assert_not_called()

    def test_single_hybrid_mode_preserves_candidate_rank_provenance(self):
        from evaluation.run_m9_hybrid_retrieval_experiment import _safe_candidate
        candidate = {"text": "fixture", "source": "fixture.pdf", "chunk_id": 0,
                     "score": 0.03, "dense_rank": 8, "bm25_rank": 1,
                     "metadata": {"source": "fixture.pdf", "page": 1}}
        row = _safe_candidate(candidate, 1, {}, "fixture", {}, {})
        self.assertEqual(row["dense_rank"], 8)
        self.assertEqual(row["bm25_rank"], 1)
