"""Tests for the privacy-filtered M8.2 comparison summary."""

import json
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.run_m8_visual_representation_experiment import (
    M8_AUDIT_CATEGORIES,
    M6_EXPECTED_PDF_COUNT,
    M6_EXPECTED_QUESTION_ROWS,
    build_argument_parser,
    build_anonymous_summary,
    build_local_input_manifest,
    require_matching_input_manifest,
    _write_input_manifest,
)


class M8VisualRepresentationTests(unittest.TestCase):
    def _write_m6_fixture(self, root):
        pdf_dir = root / "input"
        pdf_dir.mkdir()
        source_names = [f"source-{index}.pdf" for index in range(1, 9)]
        for source_name in source_names:
            (pdf_dir / source_name).write_bytes(source_name.encode("ascii"))

        questions = []
        for index in range(M6_EXPECTED_QUESTION_ROWS):
            item = {
                "question": f"private-question-{index + 1}",
                "answer": f"private-answer-{index + 1}",
                "expected_source": source_names[index % len(source_names)],
                "expected_page": 1,
                "expected_keywords": [f"private-evidence-{index + 1}"],
                "category": "ocr",
            }
            if index >= 32:
                item.update(
                    expected_source=None,
                    expected_page=None,
                    expected_keywords=[],
                    category="unanswerable",
                )
            questions.append(item)
        dataset_path = root / "m6_qa.json"
        dataset_path.write_text(json.dumps(questions), encoding="utf-8")

        m7_analysis = {
            "cases": [
                {"question_id": question_id, "classification": "PARSING_FAILURE"}
                for question_id in M8_AUDIT_CATEGORIES
            ]
        }
        m7_path = root / "m7_analysis.json"
        m7_path.write_text(json.dumps(m7_analysis), encoding="utf-8")
        manifest_path = root / "outputs" / "input_manifest.json"
        manifest_path.parent.mkdir()
        pdf_paths = sorted(pdf_dir.glob("*.pdf"))
        return questions, m7_analysis, dataset_path, m7_path, pdf_paths, manifest_path

    def test_input_manifest_accepts_fixed_m6_shape_and_matching_fingerprints(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            questions, m7_analysis, dataset, m7_path, pdfs, manifest = (
                self._write_m6_fixture(root)
            )
            pinned = build_local_input_manifest(
                questions, m7_analysis, pdfs, dataset, m7_path
            )
            manifest.write_text(json.dumps(pinned), encoding="utf-8")

            self.assertEqual(
                require_matching_input_manifest(
                    questions, m7_analysis, pdfs, dataset, m7_path, manifest
                ),
                M6_EXPECTED_PDF_COUNT,
            )

    def test_input_manifest_rejects_changed_dataset_even_with_same_row_counts(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            questions, m7_analysis, dataset, m7_path, pdfs, manifest = (
                self._write_m6_fixture(root)
            )
            pinned = build_local_input_manifest(
                questions, m7_analysis, pdfs, dataset, m7_path
            )
            manifest.write_text(json.dumps(pinned), encoding="utf-8")
            changed_questions = [dict(question) for question in questions]
            changed_questions[0]["question"] = "different private question"
            dataset.write_text(json.dumps(changed_questions), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "manifest"):
                require_matching_input_manifest(
                    changed_questions,
                    m7_analysis,
                    pdfs,
                    dataset,
                    m7_path,
                    manifest,
                )

            dataset.write_text(json.dumps(questions), encoding="utf-8")
            pdfs[0].write_bytes(b"changed private source")
            with self.assertRaisesRegex(ValueError, "manifest"):
                require_matching_input_manifest(
                    questions,
                    m7_analysis,
                    pdfs,
                    dataset,
                    m7_path,
                    manifest,
                )

    def test_input_manifest_rejects_wrong_m6_row_count_or_pdf_cohort(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            questions, m7_analysis, dataset, m7_path, pdfs, manifest = (
                self._write_m6_fixture(root)
            )
            with self.assertRaisesRegex(ValueError, "35"):
                build_local_input_manifest(
                    questions[:-1], m7_analysis, pdfs, dataset, m7_path
                )
            with self.assertRaisesRegex(ValueError, "8"):
                build_local_input_manifest(
                    questions, m7_analysis, pdfs[:-1], dataset, m7_path
                )

    def test_input_manifest_allows_unscoped_unanswerable_rows_only(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            questions, m7_analysis, dataset, m7_path, pdfs, _manifest = (
                self._write_m6_fixture(root)
            )
            self.assertEqual(
                build_local_input_manifest(
                    questions, m7_analysis, pdfs, dataset, m7_path
                )["scored_questions"],
                M6_EXPECTED_QUESTION_ROWS - 3,
            )

            questions[0]["expected_source"] = None
            questions[0]["expected_page"] = None
            dataset.write_text(json.dumps(questions), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source"):
                build_local_input_manifest(
                    questions, m7_analysis, pdfs, dataset, m7_path
                )

    def test_cli_does_not_allow_replacing_the_fixed_m6_inputs(self):
        for option in ("--dataset", "--pdf-dir", "--m7-analysis"):
            with self.subTest(option=option):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        build_argument_parser().parse_args([option, "private-path"])

    def test_cli_supports_explicit_manifest_initialization(self):
        args = build_argument_parser().parse_args(["--initialize-input-manifest"])
        self.assertTrue(args.initialize_input_manifest)

    def test_input_manifest_writer_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            manifest_path = project_root / "outputs" / "input_manifest.json"
            manifest = {"schema_version": 1, "question_rows": 35}

            with patch(
                "evaluation.run_m8_visual_representation_experiment.PROJECT_ROOT",
                project_root,
            ):
                _write_input_manifest(manifest_path, manifest)
                original_content = manifest_path.read_bytes()
                with self.assertRaisesRegex(ValueError, "already exists"):
                    _write_input_manifest(
                        manifest_path,
                        {"schema_version": 1, "question_rows": 36},
                    )

            self.assertEqual(
                manifest_path.read_bytes(), original_content
            )

    def test_summary_compares_parsing_evidence_without_serializing_private_fields(self):
        secret_question = "PRIVATE question about evidence ZXQ-991"
        secret_answer = "PRIVATE answer secret pressure 87MPa"
        secret_source = "private-industrial-document.pdf"
        secret_keyword = "secret-evidence-zxq-991"
        secret_image_keyword = "secret-image-label-qwe-332"
        questions = []
        for index in range(35):
            if index == 8:
                keyword = secret_keyword
            elif index == 15:
                keyword = secret_image_keyword
            else:
                keyword = f"unused-private-evidence-{index + 1}"
            questions.append(
                {
                    "question": secret_question,
                    "answer": secret_answer,
                    "expected_source": secret_source,
                    "expected_page": 4,
                    "expected_keywords": [keyword],
                    "category": "ocr",
                }
            )

        def case(item, evidence_hit, error_layer):
            return {
                "question": item["question"],
                "source": item["expected_source"],
                "expected_page": item["expected_page"],
                "expected_keywords": item["expected_keywords"],
                "top1_page_hit": evidence_hit,
                "top3_page_hit": evidence_hit,
                "normalized_keyword_hit": evidence_hit,
                "error_layer": error_layer,
                "diagnostic_results": [
                    {
                        "metadata": {
                            "block_type": "image",
                            "bbox": [1, 2, 3, 4],
                        }
                    }
                ],
            }

        reports = {}
        for mode in ("structured", "structured_ocr", "structured_full"):
            dense_cases = [case(item, False, "PARSING") for item in questions]
            reranker_cases = [case(item, False, "PARSING") for item in questions]
            for index in (8, 15):
                dense_cases[index]["diagnostic_results"] = []
                reranker_cases[index]["diagnostic_results"] = []
            if mode != "structured":
                dense_cases[8]["normalized_keyword_hit"] = True
                reranker_cases[8]["normalized_keyword_hit"] = True
                dense_cases[8]["error_layer"] = "RETRIEVAL"
                reranker_cases[8]["error_layer"] = "RETRIEVAL"
            reports[mode] = {
                "dense": {
                    "cases": dense_cases,
                    "top1_page_hits": 2,
                    "top1_page_total": 35,
                    "top3_page_hits": 3,
                    "top3_page_total": 35,
                    "normalized_keyword_hits": 1,
                    "normalized_keyword_total": 35,
                },
                "reranker": {
                    "cases": reranker_cases,
                    "top1_page_hits": 4,
                    "top1_page_total": 35,
                    "top3_page_hits": 5,
                    "top3_page_total": 35,
                    "normalized_keyword_hits": 2,
                    "normalized_keyword_total": 35,
                },
            }

        documents = {
            mode: [
                {
                    "text": (
                        secret_keyword
                        if mode != "structured"
                        else ""
                    ),
                    "metadata": {"source": secret_source, "page": 4},
                },
                {
                    "text": "",
                    "metadata": {"source": secret_source, "page": 4},
                },
            ]
            for mode in ("structured", "structured_ocr", "structured_full")
        }
        analysis = {
            "cases": [
                {"question_id": question_id, "classification": "PARSING_FAILURE"}
                for question_id in M8_AUDIT_CATEGORIES
            ]
        }

        summary = build_anonymous_summary(questions, reports, documents, analysis)
        serialized = json.dumps(summary, ensure_ascii=False)

        self.assertEqual(
            summary["modes"]["structured"]["parsing_failures_remaining"], 11
        )
        self.assertEqual(
            summary["modes"]["structured_ocr"]["parsing_failures_remaining"], 10
        )
        self.assertEqual(
            summary["modes"]["structured_ocr"]["newly_available_evidence_cases"], [9]
        )
        for private_value in (
            secret_question,
            secret_answer,
            secret_source,
            secret_keyword,
            secret_image_keyword,
        ):
            self.assertNotIn(private_value, serialized)


if __name__ == "__main__":
    unittest.main()
