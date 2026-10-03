"""Tests for Agent 1.2 A/B metrics and private response artifact handling."""

import importlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def _semantic(verdict, basis, object_label, value, unit, condition):
    return {
        "verdict": verdict,
        "comparability_basis": basis,
        "dimensions": {
            "object_or_field": object_label,
            "value": value,
            "unit": unit,
            "condition_or_applicability": condition,
        },
        "reason": "Evidence supports this comparison.",
        "notes": "",
    }


def _four_tasks():
    return {
        "EQ-1": {
            "include_in_headline": True,
            "expected_verdict": "EQUIVALENT",
            "dimensions": {"object_or_field": "aligned", "value": "aligned",
                           "unit": "aligned", "condition_or_applicability": "aligned"},
        },
        "DIFF-1": {
            "include_in_headline": True,
            "expected_verdict": "DIFFERENT",
            "dimensions": {"object_or_field": "aligned", "value": "different",
                           "unit": "aligned", "condition_or_applicability": "aligned"},
        },
        "NC-1": {
            "include_in_headline": True,
            "expected_verdict": "NOT_COMPARABLE",
            "dimensions": {"object_or_field": "different", "value": "aligned",
                           "unit": "aligned", "condition_or_applicability": "not_applicable"},
        },
        "EQ-2": {
            "include_in_headline": True,
            "expected_verdict": "EQUIVALENT",
            "dimensions": {"object_or_field": "aligned", "value": "aligned",
                           "unit": "aligned", "condition_or_applicability": "aligned"},
        },
    }


class Agent12EvaluationTests(unittest.TestCase):
    def _module(self, name):
        spec = importlib.util.find_spec(name)
        self.assertIsNotNone(spec, f"Agent 1.2 module {name} is missing")
        return importlib.import_module(name)

    def test_metrics_separate_acceptance_strict_and_accepted_semantic_scores(self):
        metrics_module = self._module("evaluation.agent1_2_metrics")
        tasks = _four_tasks()
        eq = _semantic("EQUIVALENT", "comparable_fact", "aligned", "aligned",
                       "aligned", "aligned")
        wrong_but_consistent = _semantic("EQUIVALENT", "comparable_fact", "aligned",
                                         "aligned", "aligned", "aligned")
        nc_with_known_dimensions_hidden = _semantic(
            "NOT_COMPARABLE", "different_object_or_field", "different",
            "not_applicable", "not_applicable", "not_applicable",
        )
        arm_a = {
            "tasks": {
                "EQ-1": {"status": "compared", "semantic": eq},
                "DIFF-1": {"status": "invalid_output", "error_code": "invalid_schema"},
                "NC-1": {"status": "invalid_output", "error_code": "invalid_schema"},
                "EQ-2": {"status": "invalid_output", "error_code": "invalid_schema"},
            },
            "api_events": [{"usage": {}} for _ in range(4)],
        }
        arm_b = {
            "tasks": {
                "EQ-1": {"status": "compared", "semantic": eq},
                "DIFF-1": {"status": "compared", "semantic": wrong_but_consistent},
                "NC-1": {"status": "compared", "semantic": nc_with_known_dimensions_hidden},
                "EQ-2": {"status": "invalid_output", "failure_type": "MISSING_FIELD"},
            },
            "api_events": [{"usage": {"prompt_tokens": 2, "completion_tokens": 3,
                                         "total_tokens": 5}} for _ in range(4)],
        }

        scored = metrics_module.score_agent1_2(tasks, arm_a, arm_b)
        self.assertEqual(scored["schema_acceptance"]["agent1_1"],
                         {"accepted": 1, "attempted": 4, "rate": 0.25})
        self.assertEqual(scored["schema_acceptance"]["agent1_2"],
                         {"accepted": 3, "attempted": 4, "rate": 0.75})
        self.assertEqual(
            scored["schema_acceptance_by_class"]["agent1_2"]["NOT_COMPARABLE"],
            {"accepted": 1, "attempted": 1, "rate": 1.0},
        )
        self.assertEqual(scored["strict_end_to_end_verdict_accuracy"]["agent1_2"],
                         {"passed": 2, "assessed": 4, "accuracy": 0.5})
        self.assertEqual(scored["accepted_output_metrics"]["agent1_2"]["verdict_accuracy"],
                         {"passed": 2, "assessed": 3, "accuracy": 2 / 3})
        self.assertEqual(
            scored["accepted_output_metrics"]["agent1_2"]["verdict_accuracy_by_class"]["DIFFERENT"],
            {"passed": 0, "assessed": 1, "accuracy": 0.0},
        )
        self.assertEqual(
            scored["accepted_output_metrics"]["agent1_2"]["not_applicable_diagnostics"][
                "object_mismatch_known_value_unit_na"],
            {"violations": 2, "assessed": 2},
        )
        self.assertEqual(scored["invalid_output_failure_taxonomy"]["agent1_2"]["MISSING_FIELD"], 1)

    def test_runner_keeps_historical_run_read_only_and_writes_private_raw_response(self):
        evaluation = self._module("evaluation.evaluate_agent1_2")
        tasks = {"EQ-1": {
            **_four_tasks()["EQ-1"],
            "comparison_request": "Compare the supported values.",
            "evidence": {"A": [{"text": "A evidence"}],
                         "B": [{"text": "B evidence"}]},
        }}
        gt_hash = "frozen-gt-hash"
        benchmark_hash = "frozen-benchmark-hash"
        frozen = {"tasks": tasks, "benchmark_sha256": benchmark_hash}
        baseline = {
            "schema_version": "agent1-1-semantic-run-v1",
            "run_status": "complete",
            "agent1_1_gt_sha256": gt_hash,
            "benchmark_sha256": benchmark_hash,
            "model": "deepseek-flash",
            "source_sha256": {"source.pdf": "source-hash"},
            "retrieval_performed": False,
            "agent0_runtime_called": False,
            "citation_renderer_called": False,
            "tasks": {"EQ-1": {"status": "invalid_output", "error_code": "invalid_schema"}},
            "api_events": [{"usage": {}}],
        }

        class FakeComparator:
            def __init__(self, **_kwargs):
                self.events = []

            def compare(self, *_args):
                self.events.append({
                    "retry": False,
                    "http_status": 200,
                    "request_id": "req-test",
                    "response_id": "resp-test",
                    "provider_status": "completed",
                    "raw_response_body": '{"id":"resp-test","status":"completed"}',
                    "raw_response_body_base64": "eyJpZCI6InJlc3AtdGVzdCIsInN0YXR1cyI6ImNvbXBsZXRlZCJ9",
                    "parsed_structured_output": _semantic(
                        "EQUIVALENT", "comparable_fact", "aligned", "aligned", "aligned", "aligned"
                    ),
                    "validation_error": None,
                    "provider_usage": {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12},
                    "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
                    "latency_ms": 1.0,
                    "cost_rmb": 0.0001,
                    "reserved_rmb": 0.001,
                    "error": None,
                })
                return _semantic("EQUIVALENT", "comparable_fact", "aligned", "aligned",
                                 "aligned", "aligned")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_path = root / "agent1_1-run.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            before = baseline_path.read_bytes()
            results_path = root / "run.local.json"
            metrics_path = root / "metrics.local.json"
            raw_path = root / "raw-responses.local.jsonl"

            def verify(**_kwargs):
                    return frozen, gt_hash, {"EQUIVALENT": 1}

            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "must-not-be-saved"}):
                result = evaluation.run_agent1_2_experiment(
                    baseline_results_path=baseline_path,
                    results_path=results_path,
                    metrics_path=metrics_path,
                    raw_responses_path=raw_path,
                    ground_truth_verifier=verify,
                    comparator_factory=FakeComparator,
                )

            self.assertEqual(baseline_path.read_bytes(), before)
            self.assertEqual(result["agent1_1_gt_sha256"], gt_hash)
            raw_record = json.loads(raw_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(raw_record["task_id"], "EQ-1")
            self.assertEqual(raw_record["provider_response_body"],
                             '{"id":"resp-test","status":"completed"}')
            self.assertEqual(raw_record["parsed_structured_output"]["verdict"], "EQUIVALENT")
            self.assertNotIn("must-not-be-saved", raw_path.read_text(encoding="utf-8"))
            self.assertNotIn("must-not-be-saved", results_path.read_text(encoding="utf-8"))
            result_text = results_path.read_text(encoding="utf-8")
            self.assertNotIn("provider_response_body", result_text)
            self.assertNotIn("raw_response_body", result_text)
            stale_metrics_run = json.loads(result_text)
            stale_metrics_run.pop("metrics")
            results_path.write_text(json.dumps(stale_metrics_run), encoding="utf-8")
            rescored = evaluation.rescore_agent1_2(
                baseline_results_path=baseline_path,
                results_path=results_path,
                metrics_path=metrics_path,
                ground_truth_verifier=verify,
            )
            self.assertIn("schema_acceptance_by_class", rescored["comparison"])
            self.assertIn("schema_acceptance_by_class",
                          json.loads(results_path.read_text(encoding="utf-8"))["metrics"])
            self.assertEqual(baseline_path.read_bytes(), before)

    def test_runner_refuses_any_preexisting_output_before_comparator_construction(self):
        evaluation = self._module("evaluation.evaluate_agent1_2")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_path = root / "run.local.json"
            metrics_path = root / "metrics.local.json"
            raw_path = root / "raw-responses.local.jsonl"
            raw_path.write_text("existing", encoding="utf-8")
            before = raw_path.read_bytes()
            constructed = []

            def comparator_factory(**kwargs):
                constructed.append(kwargs)
                return object()

            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "private-test-key"}):
                with self.assertRaises(evaluation.FrozenArtifactError):
                    evaluation.run_agent1_2_experiment(
                        baseline_results_path=root / "missing-baseline.json",
                        results_path=results_path,
                        metrics_path=metrics_path,
                        raw_responses_path=raw_path,
                        ground_truth_verifier=lambda **_kwargs: None,
                        comparator_factory=comparator_factory,
                    )
            self.assertEqual(constructed, [])
            self.assertEqual(raw_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
