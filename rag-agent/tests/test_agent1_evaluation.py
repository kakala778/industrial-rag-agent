import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from evaluation.agent1_semantic_metrics import (
    SavedReferenceError,
    evaluate_task,
    preflight_pair,
    score_semantic_run,
    validate_saved_references,
)
from evaluation.evaluate_agent1_semantic import (
    FrozenArtifactError,
    freeze_semantic_ground_truth,
    rescore_saved_record,
    verify_frozen_semantic_ground_truth,
)
from src.agent.semantic import DIMENSION_KEYS


def semantic_result(verdict, **dimension_overrides):
    dimensions = {key: "aligned" for key in DIMENSION_KEYS}
    if verdict == "NOT_COMPARABLE":
        dimensions["object_or_field"] = "different"
    elif verdict == "DIFFERENT":
        dimensions["value"] = "different"
    dimensions.update(dimension_overrides)
    return {"verdict": verdict, "dimensions": dimensions,
            "reason": "Synthetic contract response.", "notes": ""}


def reference(scope, evidence_id, excerpt=None):
    excerpt = excerpt or ("Accepted excerpt for scope " + scope)
    return {
        "evidence_id": evidence_id,
        "source": scope,
        "page": 1,
        "block_type": "text_group",
        "block_index": 0,
        "chunk_id": 0,
        "excerpt": excerpt,
        "span": {"coordinate": "child", "length": len(excerpt), "ranges": None},
        "excerpt_sha256": hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
        "context_chars": 0,
    }


def task(task_id="R01", scopes=("A", "B")):
    return {"id": task_id, "query": "Compare the requested engineering facts.",
            "scopes": list(scopes), "oracle": {}}


def saved_row(statuses=("supported", "supported"), corrupt=None):
    findings = []
    outcomes = []
    for scope, status in zip(("A", "B"), statuses):
        eid = "ev_" + ("a" if scope == "A" else "b") * 64
        refs = [reference(scope, eid)] if status == "supported" else []
        if corrupt == "wrong_source" and scope == "A" and refs:
            refs[0]["source"] = "B"
        if corrupt == "bad_hash" and scope == "A" and refs:
            refs[0]["excerpt"] += " changed"
        findings.append({"scope": scope, "status": status, "claim": "saved claim",
                         "evidence": refs})
        outcome = {"scope": scope, "status": status}
        if status == "supported":
            outcome.update(claim="saved claim", evidence_ids=[eid])
        outcomes.append(outcome)
    if corrupt == "forged_id":
        outcomes[0]["evidence_ids"] = ["ev_" + "f" * 64]
    return {
        "state": {"status": "finished", "findings": findings},
        "actions": [{"action": "FINISH", "outcomes": outcomes}],
    }


class FakeComparator:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def compare(self, request, excerpts_a, excerpts_b):
        self.calls.append((request, excerpts_a, excerpts_b))
        return self.result


class SavedReferenceAndPreflightTests(unittest.TestCase):
    def test_bilateral_saved_references_are_eligible_and_host_grounded(self):
        row = saved_row()
        refs = validate_saved_references(task(), row)
        self.assertEqual(set(refs), {"A", "B"})
        self.assertEqual(len(refs["A"]), 1)
        decision = preflight_pair(task(), row)
        self.assertTrue(decision["eligible"])
        self.assertEqual(decision["verdict"], "NOT_EVALUATED")

    def test_preflight_routes_unsupported_side_without_a_model_call(self):
        current_task = task("R05")
        row = saved_row(("supported", "insufficient_evidence"))
        decision = preflight_pair(current_task, row)
        self.assertFalse(decision["eligible"])
        self.assertEqual(decision["verdict"], "INSUFFICIENT_EVIDENCE")
        comparator = FakeComparator(semantic_result("EQUIVALENT"))
        result = evaluate_task(current_task, row, comparator)
        self.assertEqual(result["status"], "preflight_insufficient_evidence")
        self.assertIsNone(result["formal_verdict"])
        self.assertEqual(comparator.calls, [])

    def test_missing_or_unresolved_evidence_never_reaches_model(self):
        current_task = task("R08", scopes=())
        row = {"state": {"status": "clarify", "findings": []}, "actions": []}
        comparator = FakeComparator(semantic_result("EQUIVALENT"))
        result = evaluate_task(current_task, row, comparator)
        self.assertEqual(result["verdict"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(comparator.calls, [])

    def test_forged_ids_cross_scope_refs_and_changed_excerpt_hashes_are_rejected(self):
        for corruption in ("forged_id", "wrong_source", "bad_hash"):
            with self.subTest(corruption=corruption):
                with self.assertRaises(SavedReferenceError):
                    validate_saved_references(task(), saved_row(corrupt=corruption))


class SyntheticSemanticContractTests(unittest.TestCase):
    def test_identical_fact_synthetic_response_scores_equivalent(self):
        current_task = task()
        row = saved_row()
        comparator = FakeComparator(semantic_result("EQUIVALENT"))
        result = evaluate_task(current_task, row, comparator)
        self.assertEqual(result["formal_verdict"], "EQUIVALENT")
        self.assertEqual(len(comparator.calls), 1)

    def test_same_value_on_different_objects_is_not_comparable(self):
        current_task = task()
        row = saved_row()
        comparator = FakeComparator(semantic_result(
            "NOT_COMPARABLE", object_or_field="different", value="aligned", unit="aligned"))
        result = evaluate_task(current_task, row, comparator)
        self.assertEqual(result["formal_verdict"], "NOT_COMPARABLE")
        self.assertEqual(result["semantic"]["dimensions"]["object_or_field"], "different")

    def test_same_field_with_different_value_or_condition_can_be_different(self):
        for result in (
            semantic_result("DIFFERENT", value="different"),
            semantic_result("DIFFERENT", condition_or_applicability="different"),
        ):
            with self.subTest(dimensions=result["dimensions"]):
                actual = evaluate_task(task(), saved_row(), FakeComparator(result))
                self.assertEqual(actual["formal_verdict"], "DIFFERENT")

    def test_invalid_verdict_is_recorded_as_invalid_output(self):
        result = semantic_result("CONFLICT")
        actual = evaluate_task(task(), saved_row(), FakeComparator(result))
        self.assertEqual(actual["status"], "invalid_output")
        self.assertIsNone(actual["formal_verdict"])

    def test_inconsistent_verdict_dimensions_are_recorded_as_invalid_output(self):
        result = semantic_result("EQUIVALENT", object_or_field="different")
        actual = evaluate_task(task(), saved_row(), FakeComparator(result))
        self.assertEqual(actual["status"], "invalid_output")
        self.assertIsNone(actual["formal_verdict"])

    def test_unit_mismatch_is_excluded_until_a_conversion_policy_exists(self):
        result = semantic_result("DIFFERENT", unit="different")
        actual = evaluate_task(task(), saved_row(), FakeComparator(result))
        self.assertEqual(actual["status"], "unit_conversion_required")
        self.assertIsNone(actual["formal_verdict"])
        self.assertEqual(actual["semantic"]["verdict"], "DIFFERENT")

    def test_metrics_separate_not_comparable_false_negative_from_other_errors(self):
        gt = {"tasks": {"R01": {
            "expected_route": "semantic", "expected_verdict": "NOT_COMPARABLE",
            "expected_object_alignment": "different", "expected_value_alignment": "aligned",
            "expected_unit_alignment": "aligned", "expected_condition_alignment": "aligned",
            "confidence": "high",
        }}}
        outcomes = {"R01": {
            "status": "compared", "formal_verdict": "EQUIVALENT",
            "semantic": semantic_result("EQUIVALENT", object_or_field="aligned"),
            "preflight": {"eligible": True, "verdict": "NOT_EVALUATED"},
            "citation_grounding": {"passed": 2, "assessed": 2},
        }}
        metrics = score_semantic_run(outcomes, gt, api_events=[])
        self.assertEqual(metrics["verdict_accuracy"],
                         {"passed": 0, "assessed": 1, "accuracy": 0.0})
        self.assertEqual(metrics["error_taxonomy"]["NOT_COMPARABLE_FALSE_NEGATIVE"], 1)
        self.assertEqual(metrics["citation_grounding"], {"passed": 2, "assessed": 2})

    def test_metrics_reject_inconsistent_saved_semantic_output(self):
        gt = {"tasks": {"R01": {
            "expected_route": "semantic", "expected_verdict": "NOT_COMPARABLE",
            "expected_object_alignment": "different", "expected_value_alignment": "aligned",
            "expected_unit_alignment": "aligned", "expected_condition_alignment": "aligned",
            "confidence": "high",
        }}}
        outcomes = {"R01": {
            "status": "compared", "formal_verdict": "EQUIVALENT",
            "semantic": semantic_result("EQUIVALENT", object_or_field="different"),
            "preflight": {"eligible": True, "verdict": "NOT_EVALUATED"},
            "citation_grounding": {"passed": 2, "assessed": 2},
        }}
        metrics = score_semantic_run(outcomes, gt, api_events=[])
        self.assertEqual(metrics["invalid_semantic_outputs"], 1)
        self.assertEqual(metrics["verdict_accuracy"]["assessed"], 0)
        self.assertEqual(metrics["dimension_accuracy"]["object_or_field"]["assessed"], 0)

    def test_metrics_reject_inconsistent_saved_formal_verdict_or_status(self):
        gt = {"tasks": {"R01": {
            "expected_route": "semantic", "expected_verdict": "NOT_COMPARABLE",
            "expected_object_alignment": "different", "expected_value_alignment": "aligned",
            "expected_unit_alignment": "aligned", "expected_condition_alignment": "aligned",
            "confidence": "high",
        }}}
        cases = (
            {
                "status": "compared", "formal_verdict": "EQUIVALENT",
                "semantic": semantic_result("NOT_COMPARABLE"),
            },
            {
                "status": "compared", "formal_verdict": "DIFFERENT",
                "semantic": semantic_result("DIFFERENT", unit="different"),
            },
            {
                "status": "unit_conversion_required", "formal_verdict": "NOT_COMPARABLE",
                "semantic": semantic_result("NOT_COMPARABLE", unit="different"),
            },
            {
                "status": "api_error", "formal_verdict": None,
                "semantic": semantic_result("NOT_COMPARABLE"),
            },
            {
                "status": "not_run_after_stop", "formal_verdict": None,
                "semantic": semantic_result("NOT_COMPARABLE"),
            },
        )
        for actual in cases:
            actual.update({
                "preflight": {"eligible": True, "verdict": "NOT_EVALUATED"},
                "citation_grounding": {"passed": 2, "assessed": 2},
            })
            with self.subTest(status=actual["status"]):
                metrics = score_semantic_run({"R01": actual}, gt, api_events=[])
                self.assertEqual(metrics["invalid_semantic_outputs"], 1)
                self.assertEqual(metrics["verdict_accuracy"]["assessed"], 0)

    def test_valid_unit_conversion_status_excludes_verdict_without_invalidating_dimensions(self):
        gt = {"tasks": {"R01": {
            "expected_route": "semantic", "expected_verdict": "DIFFERENT",
            "expected_object_alignment": "aligned", "expected_value_alignment": "aligned",
            "expected_unit_alignment": "different", "expected_condition_alignment": "aligned",
            "confidence": "high",
        }}}
        outcomes = {"R01": {
            "status": "unit_conversion_required", "formal_verdict": None,
            "semantic": semantic_result("DIFFERENT", unit="different"),
            "preflight": {"eligible": True, "verdict": "NOT_EVALUATED"},
            "citation_grounding": {"passed": 2, "assessed": 2},
        }}
        metrics = score_semantic_run(outcomes, gt, api_events=[])
        self.assertEqual(metrics["invalid_semantic_outputs"], 0)
        self.assertEqual(metrics["verdict_accuracy"]["assessed"], 0)
        self.assertEqual(metrics["dimension_accuracy"]["unit"],
                         {"passed": 1, "assessed": 1, "accuracy": 1.0})

    def test_nonsemantic_frozen_routes_cannot_enter_model_scoring(self):
        actual = {
            "status": "compared", "formal_verdict": "NOT_COMPARABLE",
            "semantic": semantic_result("NOT_COMPARABLE"),
            "preflight": {"eligible": True, "reason": "bilateral_supported"},
            "citation_grounding": {"passed": 2, "assessed": 2},
        }
        for route in ("retrieval_bound", "unsupported_side", "unresolved_scope"):
            with self.subTest(route=route):
                metrics = score_semantic_run(
                    {"R01": actual}, {"tasks": {"R01": {"expected_route": route}}},
                    api_events=[],
                )
                self.assertEqual(metrics["invalid_semantic_outputs"], 1)
                self.assertEqual(metrics["eligible_pair_count"], 0)
                self.assertEqual(metrics["verdict_accuracy"]["assessed"], 0)
                self.assertEqual(
                    metrics["dimension_accuracy"]["object_or_field"]["assessed"], 0)
                if route == "unsupported_side":
                    self.assertEqual(
                        metrics["unsupported_side_routing_correctness"]["passed"], 0)

    def test_expected_semantic_route_with_host_preflight_rejection_is_invalid(self):
        actual = {
            "status": "preflight_insufficient_evidence",
            "verdict": "INSUFFICIENT_EVIDENCE", "formal_verdict": None,
            "semantic": None,
            "preflight": {"eligible": False, "reason": "not_bilateral_supported"},
            "citation_grounding": {"passed": 0, "assessed": 0},
        }
        metrics = score_semantic_run(
            {"R01": actual}, {"tasks": {"R01": {"expected_route": "semantic"}}},
            api_events=[],
        )
        self.assertEqual(metrics["invalid_semantic_outputs"], 1)
        self.assertEqual(metrics["eligible_pair_count"], 0)

    def test_metrics_report_confidence_cohort_and_conservative_cost_bound(self):
        gt = {"tasks": {"R01": {
            "expected_route": "semantic", "expected_verdict": "NOT_COMPARABLE",
            "expected_object_alignment": "different", "expected_value_alignment": "aligned",
            "expected_unit_alignment": "aligned", "expected_condition_alignment": "aligned",
            "confidence": "medium",
        }}}
        outcomes = {"R01": {
            "status": "compared", "formal_verdict": "NOT_COMPARABLE",
            "semantic": semantic_result("NOT_COMPARABLE"),
            "preflight": {"eligible": True, "verdict": "NOT_EVALUATED"},
            "citation_grounding": {"passed": 0, "assessed": 0},
        }}
        events = [{"usage": {"prompt_tokens": 10, "completion_tokens": 5,
                              "total_tokens": 15},
                   "cost_rmb": 0.00006, "reserved_rmb": 0.01, "latency_ms": 20}]
        metrics = score_semantic_run(outcomes, gt, api_events=events)
        self.assertEqual(metrics["eligible_pair_count"], 1)
        self.assertEqual(metrics["verdict_accuracy_by_confidence"]["medium"],
                         {"passed": 1, "assessed": 1, "accuracy": 1.0})
        self.assertEqual(metrics["api_usage"]["conservative_upper_rmb"], 0.01)


def gt_proposal():
    tasks = {}
    for task_id in ("R01", "R03", "R06", "R07"):
        tasks[task_id] = {
            "expected_route": "semantic",
            "expected_verdict": "NOT_COMPARABLE",
            "expected_object_alignment": "different",
            "expected_value_alignment": "aligned",
            "expected_unit_alignment": "aligned",
            "expected_condition_alignment": "different",
            "confidence": "high",
            "review_note": "Synthetic note.",
        }
    tasks["R02"] = {"expected_route": "retrieval_bound"}
    tasks["R04"] = {"expected_route": "retrieval_bound"}
    tasks["R05"] = {"expected_route": "unsupported_side"}
    tasks["R08"] = {"expected_route": "unresolved_scope"}
    return {"schema_version": "agent1-semantic-gt-v1",
            "authority": "AI-assisted reviewed provisional GT", "tasks": tasks}


class FrozenSemanticGroundTruthTests(unittest.TestCase):
    def test_freeze_binds_provisional_gt_to_inputs_and_refuses_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            proposal_path = root / "proposal.json"
            frozen_path = root / "semantic-gt.json"
            results_path = root / "results.json"
            source_path = root / "frozen-input.json"
            proposal_path.write_text(json.dumps(gt_proposal()), encoding="utf-8")
            source_path.write_text("frozen", encoding="utf-8")
            sources = {"frozen_input": source_path}
            frozen_sha = freeze_semantic_ground_truth(
                proposal_path, frozen_path, sources, results_path)
            loaded, verified_sha = verify_frozen_semantic_ground_truth(
                frozen_path, sources)
            self.assertEqual(frozen_sha, verified_sha)
            self.assertEqual(loaded["authority"], "AI-assisted reviewed provisional GT")
            self.assertEqual(set(loaded["tasks"]), {f"R0{i}" for i in range(1, 9)})

            source_path.write_text("changed", encoding="utf-8")
            with self.assertRaises(FrozenArtifactError):
                verify_frozen_semantic_ground_truth(frozen_path, sources)

    def test_freeze_refuses_overwrite_or_freezing_after_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            proposal_path = root / "proposal.json"
            frozen_path = root / "semantic-gt.json"
            results_path = root / "results.json"
            source_path = root / "input.json"
            proposal_path.write_text(json.dumps(gt_proposal()), encoding="utf-8")
            source_path.write_text("frozen", encoding="utf-8")
            frozen_path.write_text("already frozen", encoding="utf-8")
            with self.assertRaises(FrozenArtifactError):
                freeze_semantic_ground_truth(proposal_path, frozen_path,
                                             {"input": source_path}, results_path)

            frozen_path.unlink()
            results_path.write_text("inference already exists", encoding="utf-8")
            with self.assertRaises(FrozenArtifactError):
                freeze_semantic_ground_truth(proposal_path, frozen_path,
                                             {"input": source_path}, results_path)


class EvaluationRunnerInvocationTests(unittest.TestCase):
    def test_runner_supports_direct_script_invocation_from_application_root(self):
        app_root = Path(__file__).resolve().parents[1]
        script = app_root / "evaluation" / "evaluate_agent1_semantic.py"
        result = subprocess.run([sys.executable, str(script), "--help"], cwd=app_root,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("freeze-gt", result.stdout)

    def test_saved_outputs_can_be_rescored_offline_without_changing_gt(self):
        gt = {"tasks": {"R01": {
            "expected_route": "semantic", "expected_verdict": "NOT_COMPARABLE",
            "expected_object_alignment": "different", "expected_value_alignment": "aligned",
            "expected_unit_alignment": "aligned", "expected_condition_alignment": "aligned",
            "confidence": "medium",
        }}}
        semantic = semantic_result("NOT_COMPARABLE", object_or_field="different")
        record = {
            "semantic_gt_sha256": "frozen-sha",
            "tasks": {"R01": {
                "status": "compared", "formal_verdict": "NOT_COMPARABLE",
                "semantic": semantic,
                "preflight": {"eligible": True},
                "citation_grounding": {"passed": 0, "assessed": 0},
            }},
            "api_events": [{"usage": {"prompt_tokens": 2, "completion_tokens": 1,
                                         "total_tokens": 3},
                            "cost_rmb": 0.000016, "reserved_rmb": 0.01,
                            "latency_ms": 5}],
            "budget_reserved_rmb": 0.000016,
        }
        private_root = Path(__file__).resolve().parents[1] / "outputs" / "agent1"
        private_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=private_root) as temp:
            path = Path(temp) / "saved-results.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            updated = rescore_saved_record(path, gt, "frozen-sha")
            reread = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(updated["metrics"]["verdict_accuracy"]["accuracy"], 1.0)
        self.assertEqual(updated["metrics"]["api_usage"]["conservative_upper_rmb"], 0.01)
        self.assertEqual(updated["semantic_gt_sha256"], "frozen-sha")
        self.assertIn("rescored_at_utc", reread)


if __name__ == "__main__":
    unittest.main()
