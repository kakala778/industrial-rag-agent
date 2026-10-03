import unittest
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from evaluation.agent1_1_metrics import score_agent1_1
from evaluation.evaluate_agent1_1 import (
    FrozenArtifactError,
    freeze_agent1_1_ground_truth,
    verify_frozen_agent1_1_ground_truth,
)
from src.agent.semantic import DIMENSION_KEYS


def dimensions(object_label="aligned", value="aligned", unit="aligned", condition="aligned"):
    return {
        "object_or_field": object_label,
        "value": value,
        "unit": unit,
        "condition_or_applicability": condition,
    }


def task(task_id, verdict, labels, *, include=True, exclusion_reason=None):
    return {
        "expected_verdict": verdict,
        "dimensions": labels,
        "confidence": "high" if include else "uncertain",
        "include_in_headline": include,
        "exclusion_reason": exclusion_reason,
    }


def semantic(verdict, basis, labels):
    return {
        "verdict": verdict,
        "comparability_basis": basis,
        "dimensions": labels,
        "reason": "Synthetic result for evaluator mechanics.",
        "notes": "",
    }


class Agent11MetricsTests(unittest.TestCase):
    def test_metrics_include_class_confusion_dimensions_and_independent_object_mismatch(self):
        tasks = {
            "E01": task("E01", "EQUIVALENT", dimensions(condition="not_applicable")),
            "D01": task("D01", "DIFFERENT", dimensions(value="different",
                                                         condition="not_applicable")),
            "N01": task("N01", "NOT_COMPARABLE", dimensions(
                object_label="different", condition="not_applicable")),
        }
        outcomes = {
            "E01": {"status": "compared", "semantic": semantic(
                "EQUIVALENT", "comparable_fact", dimensions(condition="not_applicable"))},
            "D01": {"status": "compared", "semantic": semantic(
                "EQUIVALENT", "comparable_fact", dimensions(condition="not_applicable"))},
            "N01": {"status": "compared", "semantic": semantic(
                "NOT_COMPARABLE", "different_object_or_field", dimensions(
                    object_label="different", condition="not_applicable"))},
        }
        metrics = score_agent1_1(tasks, outcomes, api_events=[])
        self.assertEqual(metrics["verdict_accuracy"],
                         {"passed": 2, "assessed": 3, "accuracy": 2 / 3})
        self.assertEqual(metrics["verdict_accuracy_by_class"]["DIFFERENT"],
                         {"passed": 0, "assessed": 1, "accuracy": 0.0})
        self.assertEqual(metrics["confusion_matrix"]["DIFFERENT"]["EQUIVALENT"], 1)
        self.assertEqual(metrics["dimension_accuracy"]["object_or_field"],
                         {"passed": 3, "assessed": 3, "accuracy": 1.0})
        self.assertEqual(metrics["dimension_accuracy"]["unit"],
                         {"passed": 3, "assessed": 3, "accuracy": 1.0})
        self.assertEqual(metrics["verdict_dimension_consistency"],
                         {"passed": 3, "assessed": 3, "inconsistencies": 0,
                          "consistency": 1.0})
        self.assertEqual(metrics["not_applicable_diagnostics"][
            "object_mismatch_known_value_unit_na"], {"violations": 0, "assessed": 2})
        self.assertEqual(metrics["error_taxonomy"]["VERDICT_ERROR"], 1)
        self.assertEqual(metrics["error_taxonomy"]["VALUE_ALIGNMENT_ERROR"], 1)

    def test_contract_inconsistency_is_counted_as_invalid_and_not_scored(self):
        tasks = {"N01": task("N01", "NOT_COMPARABLE", dimensions(
            object_label="different", condition="not_applicable"))}
        invalid = semantic("EQUIVALENT", "different_object_or_field", dimensions(
            object_label="different", condition="not_applicable"))
        metrics = score_agent1_1(tasks, {"N01": {
            "status": "invalid_output",
            "error_code": "verdict_dimension_inconsistency",
            "semantic": None,
        }}, api_events=[])
        self.assertEqual(metrics["invalid_semantic_outputs"], 1)
        self.assertEqual(metrics["verdict_accuracy"]["assessed"], 1)
        self.assertEqual(metrics["verdict_accuracy"]["passed"], 0)
        self.assertEqual(metrics["confusion_matrix"]["NOT_COMPARABLE"]["INVALID"], 1)
        self.assertEqual(metrics["verdict_dimension_consistency"],
                         {"passed": 0, "assessed": 1, "inconsistencies": 1,
                          "consistency": 0.0})
        self.assertEqual(metrics["error_taxonomy"]["VERDICT_DIMENSION_INCONSISTENCY"], 1)

    def test_gt_uncertain_is_excluded_and_recorded_not_relabelled_as_model_error(self):
        tasks = {
            "U01": task("U01", "NOT_COMPARABLE", dimensions(), include=False,
                         exclusion_reason="GT_UNCERTAIN"),
            "N01": task("N01", "NOT_COMPARABLE", dimensions(
                object_label="different", condition="not_applicable")),
        }
        outcomes = {"N01": {"status": "api_error", "error_code": "deepseek:timeout"}}
        metrics = score_agent1_1(tasks, outcomes, api_events=[{
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            "cost_rmb": 0.00003, "reserved_rmb": 0.01, "latency_ms": 50,
        }])
        self.assertEqual(metrics["excluded_tasks"]["GT_UNCERTAIN"], 1)
        self.assertEqual(metrics["api_errors"], 1)
        self.assertEqual(metrics["verdict_accuracy"]["assessed"], 0)
        self.assertEqual(metrics["api_usage"]["requests"], 1)
        self.assertEqual(metrics["api_usage"]["prompt_tokens"], 10)


def proposal_task(task_id, verdict, labels):
    basis = "comparable_fact" if verdict != "NOT_COMPARABLE" else "different_object_or_field"
    return {
        "case_type": "synthetic_mechanics_only",
        "comparison_request": "Compare the requested fact.",
        "expected_verdict": verdict,
        "expected_comparability_basis": basis,
        "dimensions": labels,
        "confidence": "high",
        "include_in_headline": True,
        "exclusion_reason": None,
        "review_status": "ai_pdf_reviewed",
        "review_authority": "ai_assisted_original_pdf_review",
        "review_note": "Synthetic evaluator fixture; never an industrial benchmark case.",
        "evidence": {
            side: [{
                "source_name": "source-0.pdf",
                "page_number": 1,
                "text": "PDF page evidence.",
                "excerpt_sha256": hashlib.sha256(
                    b"PDF page evidence.").hexdigest(),
            }]
            for side in ("A", "B")
        },
    }


class Agent11FreezeTests(unittest.TestCase):
    def test_freeze_binds_gt_to_pdf_hashes_and_exact_page_excerpts_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "pdfs"
            source_root.mkdir()
            files = []
            for index in range(8):
                path = source_root / f"source-{index}.pdf"
                path.write_bytes(f"protected-pdf-{index}".encode("ascii"))
                files.append({
                    "name": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "pages": 1,
                    "text_pages": 1,
                    "text_chars": 18,
                })
            manifest_path = root / "source-manifest.json"
            manifest_path.write_text(json.dumps({
                "schema_version": "agent1_1-source-manifest-v1",
                "files": files,
            }), encoding="utf-8")
            tasks = {}
            for i in range(1, 5):
                tasks[f"E{i:02}"] = proposal_task(
                    f"E{i:02}", "EQUIVALENT", dimensions(condition="not_applicable"))
                tasks[f"D{i:02}"] = proposal_task(
                    f"D{i:02}", "DIFFERENT", dimensions(value="different",
                                                         condition="not_applicable"))
                tasks[f"N{i:02}"] = proposal_task(
                    f"N{i:02}", "NOT_COMPARABLE", dimensions(
                        object_label="different", condition="not_applicable"))
            proposal_path = root / "proposal.json"
            proposal_path.write_text(json.dumps({
                "schema_version": "agent1-1-semantic-benchmark-proposal-v1",
                "authority": "ai_assisted_original_pdf_review",
                "review_status": "ai_pdf_reviewed",
                "review_authority": "ai_assisted_original_pdf_review",
                "source_review": {"visually_reviewed_pages": []},
                "tasks": tasks,
            }), encoding="utf-8")
            frozen_path = root / "frozen.json"
            results_path = root / "run.json"

            digest, class_counts = freeze_agent1_1_ground_truth(
                proposal_path=proposal_path,
                frozen_path=frozen_path,
                results_path=results_path,
                source_manifest_path=manifest_path,
                source_root=source_root,
                page_reader=lambda _path, _page: "PDF page evidence.",
            )
            frozen, verified_digest, verified_counts = verify_frozen_agent1_1_ground_truth(
                frozen_path=frozen_path,
                source_manifest_path=manifest_path,
                source_root=source_root,
                page_reader=lambda _path, _page: "PDF page evidence.",
            )
            self.assertEqual(digest, verified_digest)
            self.assertEqual(class_counts, verified_counts)
            self.assertEqual(frozen["task_count"], 12)
            self.assertEqual(class_counts, {"EQUIVALENT": 4, "DIFFERENT": 4,
                                            "NOT_COMPARABLE": 4})
            with self.assertRaises(FrozenArtifactError):
                freeze_agent1_1_ground_truth(
                    proposal_path=proposal_path,
                    frozen_path=frozen_path,
                    results_path=results_path,
                    source_manifest_path=manifest_path,
                    source_root=source_root,
                    page_reader=lambda _path, _page: "PDF page evidence.",
                )

            (source_root / "source-0.pdf").write_bytes(b"changed")
            with self.assertRaises(FrozenArtifactError):
                verify_frozen_agent1_1_ground_truth(
                    frozen_path=frozen_path,
                    source_manifest_path=manifest_path,
                    source_root=source_root,
                    page_reader=lambda _path, _page: "PDF page evidence.",
                )

    def test_runner_help_works_from_application_root_without_api_access(self):
        app_root = Path(__file__).resolve().parents[1]
        script = app_root / "evaluation" / "evaluate_agent1_1.py"
        result = subprocess.run([sys.executable, str(script), "--help"], cwd=app_root,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("freeze-gt", result.stdout)
        self.assertIn("rescore", result.stdout)


if __name__ == "__main__":
    unittest.main()
