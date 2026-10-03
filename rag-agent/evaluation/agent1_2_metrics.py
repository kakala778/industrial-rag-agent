"""Offline A/B metrics for the Agent 1.2 structured-output experiment."""

from collections import Counter
import statistics

from src.agent.semantic import DIMENSION_KEYS
from src.agent.semantic_contract_v2 import (
    InvalidSemanticOutputV2,
    OUTPUT_FAILURE_TYPES_V2,
    validate_semantic_output_v2,
)


_VERDICTS = ("EQUIVALENT", "DIFFERENT", "NOT_COMPARABLE")
_DIMENSION_ERRORS = {
    "object_or_field": "OBJECT_ALIGNMENT_ERROR",
    "value": "VALUE_ALIGNMENT_ERROR",
    "unit": "UNIT_ALIGNMENT_ERROR",
    "condition_or_applicability": "CONDITION_ALIGNMENT_ERROR",
}


def _ratio(passed, assessed):
    return passed / assessed if assessed else None


def _accuracy(passed, assessed):
    return {"passed": passed, "assessed": assessed,
            "accuracy": _ratio(passed, assessed)}


def _api_usage(events):
    prompt_tokens = completion_tokens = total_tokens = 0
    costs = []
    reservations = []
    latencies = []
    api_errors = retries = 0
    for event in events:
        if type(event) is not dict:
            continue
        usage = event.get("usage")
        if type(usage) is dict:
            for key, target in (("prompt_tokens", "prompt"),
                                ("completion_tokens", "completion"),
                                ("total_tokens", "total")):
                value = usage.get(key)
                if type(value) is int and value >= 0:
                    if target == "prompt":
                        prompt_tokens += value
                    elif target == "completion":
                        completion_tokens += value
                    else:
                        total_tokens += value
        for key, values in (("cost_rmb", costs),
                            ("reserved_rmb", reservations),
                            ("latency_ms", latencies)):
            value = event.get(key)
            if type(value) in (int, float) and value >= 0:
                values.append(float(value))
        if event.get("error") not in (None, "", False, "invalid_output"):
            api_errors += 1
        if event.get("retry") is True:
            retries += 1
    return {
        "requests": len(events),
        "api_errors": api_errors,
        "retries": retries,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_rmb": round(sum(costs), 10) if costs else None,
        "conservative_reserved_rmb": round(sum(reservations), 10),
        "mean_latency_ms": round(statistics.mean(latencies), 3) if latencies else None,
        "max_latency_ms": round(max(latencies), 3) if latencies else None,
    }


def _accepted_semantic(row):
    if type(row) is not dict or row.get("status") not in (
            "compared", "semantic_uncertain"):
        return None, None
    try:
        return validate_semantic_output_v2(row.get("semantic")), None
    except InvalidSemanticOutputV2 as exc:
        return None, exc


def _score_arm(tasks, run, *, historical=False):
    if type(run) is not dict or type(run.get("tasks")) is not dict:
        raise ValueError("invalid saved Agent 1.2 arm")
    events = run.get("api_events", [])
    if type(events) is not list:
        raise ValueError("invalid saved Agent API events")
    headline = {
        task_id: task for task_id, task in tasks.items()
        if type(task) is dict and task.get("include_in_headline") is True
    }
    class_attempts = Counter()
    for task in headline.values():
        expected = task.get("expected_verdict")
        if expected not in _VERDICTS:
            raise ValueError("included task lacks a valid Agent 1.1 verdict")
        class_attempts[expected] += 1
    accepted = {}
    strict_passed = 0
    taxonomy = Counter({kind: 0 for kind in OUTPUT_FAILURE_TYPES_V2})
    consistency_failures = 0
    legacy_unclassified = 0
    api_error_tasks = 0
    not_run = 0

    for task_id, task in headline.items():
        row = run["tasks"].get(task_id, {"status": "not_run_after_stop"})
        if type(row) is not dict:
            row = {"status": "invalid_output", "failure_type": "OTHER"}
        semantic, parser_error = _accepted_semantic(row)
        if semantic is not None:
            accepted[task_id] = (task, row, semantic)
            definite = (row.get("status") != "semantic_uncertain"
                        and semantic["comparability_basis"] != "uncertain")
            if definite and semantic["verdict"] == task.get("expected_verdict"):
                strict_passed += 1
            continue

        if row.get("status") == "api_error":
            api_error_tasks += 1
            continue
        if row.get("status") == "not_run_after_stop":
            not_run += 1
            continue

        if parser_error is not None:
            failure_type = parser_error.failure_type
        else:
            failure_type = row.get("failure_type")
            if failure_type not in OUTPUT_FAILURE_TYPES_V2:
                if historical and row.get("status") == "invalid_output" and row.get(
                        "error_code") in ("invalid_schema", "invalid_output"):
                    legacy_unclassified += 1
                    continue
                failure_type = "OTHER"
        if failure_type in OUTPUT_FAILURE_TYPES_V2:
            taxonomy[failure_type] += 1
            if failure_type == "CONSISTENCY_ERROR":
                consistency_failures += 1

    class_counts = {
        label: {"passed": 0, "assessed": 0} for label in _VERDICTS
    }
    accepted_class_counts = Counter(
        task["expected_verdict"] for task, _row, _semantic in accepted.values()
    )
    dimension_counts = {
        key: {"passed": 0, "assessed": 0} for key in DIMENSION_KEYS
    }
    confusion = {
        label: {prediction: 0 for prediction in (*_VERDICTS, "UNCERTAIN")}
        for label in _VERDICTS
    }
    na_counts = Counter()
    na_assessed = na_violations = 0
    accepted_verdict_passed = 0
    consistency_passed = len(accepted)
    for _task_id, (task, row, semantic) in accepted.items():
        expected_verdict = task.get("expected_verdict")
        expected_dimensions = task.get("dimensions")
        if expected_verdict not in class_counts or type(expected_dimensions) is not dict:
            raise ValueError("included task lacks complete Agent 1.1 ground truth")
        class_counts[expected_verdict]["assessed"] += 1
        is_uncertain = (row.get("status") == "semantic_uncertain"
                        or semantic["comparability_basis"] == "uncertain")
        if is_uncertain:
            confusion[expected_verdict]["UNCERTAIN"] += 1
        else:
            predicted = semantic["verdict"]
            confusion[expected_verdict][predicted] += 1
            if predicted == expected_verdict:
                class_counts[expected_verdict]["passed"] += 1
                accepted_verdict_passed += 1
        for dimension in DIMENSION_KEYS:
            actual = semantic["dimensions"][dimension]
            expected = expected_dimensions.get(dimension)
            dimension_counts[dimension]["assessed"] += 1
            if actual == expected:
                dimension_counts[dimension]["passed"] += 1
            if actual == "not_applicable":
                na_counts[dimension] += 1
        if (expected_dimensions.get("object_or_field") == "different"
                and expected_dimensions.get("value") != "not_applicable"):
            na_assessed += 1
            if semantic["dimensions"]["value"] == "not_applicable":
                na_violations += 1
        if (expected_dimensions.get("object_or_field") == "different"
                and expected_dimensions.get("unit") != "not_applicable"):
            na_assessed += 1
            if semantic["dimensions"]["unit"] == "not_applicable":
                na_violations += 1

    attempted = len(headline)
    metrics = {
        "schema_acceptance": {
            "accepted": len(accepted),
            "attempted": attempted,
            "rate": _ratio(len(accepted), attempted),
        },
        "schema_acceptance_by_class": {
            label: {
                "accepted": accepted_class_counts[label],
                "attempted": class_attempts[label],
                "rate": _ratio(accepted_class_counts[label], class_attempts[label]),
            }
            for label in _VERDICTS
        },
        "strict_end_to_end_verdict_accuracy": _accuracy(strict_passed, attempted),
        "accepted_output_metrics": {
            "accepted": len(accepted),
            "verdict_accuracy": _accuracy(accepted_verdict_passed, len(accepted)),
            "verdict_accuracy_by_class": {
                label: _accuracy(counts["passed"], counts["assessed"])
                for label, counts in class_counts.items()
            },
            "confusion_matrix": confusion,
            "dimension_accuracy": {
                dimension: _accuracy(counts["passed"], counts["assessed"])
                for dimension, counts in dimension_counts.items()
            },
            "verdict_dimension_consistency": {
                "passed": consistency_passed,
                "assessed": consistency_passed + consistency_failures,
                "inconsistencies": consistency_failures,
                "consistency": _ratio(
                    consistency_passed, consistency_passed + consistency_failures
                ),
            },
            "not_applicable_diagnostics": {
                "counts_by_dimension": {
                    dimension: na_counts[dimension] for dimension in DIMENSION_KEYS
                },
                "object_mismatch_known_value_unit_na": {
                    "violations": na_violations,
                    "assessed": na_assessed,
                },
            },
        },
        "invalid_output_failure_taxonomy": {
            **{kind: taxonomy[kind] for kind in sorted(OUTPUT_FAILURE_TYPES_V2)},
            "total_classified_invalid_outputs": sum(taxonomy.values()),
            "unclassified_historical_invalid_schema": legacy_unclassified,
        },
        "api_usage": _api_usage(events),
        "api_error_tasks": api_error_tasks,
        "not_run_after_stop": not_run,
    }
    return metrics


def score_agent1_2(tasks, arm_a_run, arm_b_run):
    """Compare the frozen historical json_object arm with the one json_schema arm."""
    if type(tasks) is not dict:
        raise ValueError("invalid frozen Agent 1.1 tasks")
    a = _score_arm(tasks, arm_a_run, historical=True)
    b = _score_arm(tasks, arm_b_run)
    return {
        "benchmark_tasks": len(tasks),
        "headline_tasks": sum(
            1 for task in tasks.values()
            if type(task) is dict and task.get("include_in_headline") is True
        ),
        "schema_acceptance": {
            "agent1_1": a["schema_acceptance"],
            "agent1_2": b["schema_acceptance"],
        },
        "schema_acceptance_by_class": {
            "agent1_1": a["schema_acceptance_by_class"],
            "agent1_2": b["schema_acceptance_by_class"],
        },
        "strict_end_to_end_verdict_accuracy": {
            "agent1_1": a["strict_end_to_end_verdict_accuracy"],
            "agent1_2": b["strict_end_to_end_verdict_accuracy"],
        },
        "accepted_output_metrics": {
            "agent1_1": a["accepted_output_metrics"],
            "agent1_2": b["accepted_output_metrics"],
        },
        "invalid_output_failure_taxonomy": {
            "agent1_1": a["invalid_output_failure_taxonomy"],
            "agent1_2": b["invalid_output_failure_taxonomy"],
        },
        "api_usage": {"agent1_1": a["api_usage"], "agent1_2": b["api_usage"]},
        "execution": {
            "agent1_1_api_error_tasks": a["api_error_tasks"],
            "agent1_2_api_error_tasks": b["api_error_tasks"],
            "agent1_1_not_run_after_stop": a["not_run_after_stop"],
            "agent1_2_not_run_after_stop": b["not_run_after_stop"],
        },
    }
