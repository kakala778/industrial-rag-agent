"""Offline scoring for the isolated Agent 1.1 semantic benchmark."""

from collections import Counter
import statistics

from src.agent.semantic import DIMENSION_KEYS
from src.agent.semantic_contract_v2 import (
    InvalidSemanticOutputV2,
    validate_semantic_output_v2,
)


_DIMENSION_ERRORS = {
    "object_or_field": "OBJECT_ALIGNMENT_ERROR",
    "value": "VALUE_ALIGNMENT_ERROR",
    "unit": "UNIT_ALIGNMENT_ERROR",
    "condition_or_applicability": "CONDITION_ALIGNMENT_ERROR",
}
_ERRORS = (
    "VERDICT_ERROR",
    "OBJECT_ALIGNMENT_ERROR",
    "VALUE_ALIGNMENT_ERROR",
    "UNIT_ALIGNMENT_ERROR",
    "CONDITION_ALIGNMENT_ERROR",
    "VERDICT_DIMENSION_INCONSISTENCY",
    "GT_UNCERTAIN",
    "SCHEMA_UNSUPPORTED",
    "INVALID_OUTPUT",
    "SEMANTIC_UNCERTAIN",
)
_VERDICTS = ("EQUIVALENT", "DIFFERENT", "NOT_COMPARABLE")
_CONFUSION_COLUMNS = (*_VERDICTS, "UNCERTAIN", "INVALID")


def _ratio(passed, assessed):
    return passed / assessed if assessed else None


def _accuracy(passed, assessed):
    return {"passed": passed, "assessed": assessed,
            "accuracy": _ratio(passed, assessed)}


def _api_usage(events):
    prompt_tokens = completion_tokens = total_tokens = 0
    cost_values = []
    reservation_values = []
    latencies = []
    for event in events:
        usage = event.get("usage") if type(event) is dict else None
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
        for key, values in (("cost_rmb", cost_values),
                            ("reserved_rmb", reservation_values),
                            ("latency_ms", latencies)):
            value = event.get(key) if type(event) is dict else None
            if type(value) in (int, float) and value >= 0:
                values.append(float(value))
    return {
        "requests": len(events),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_rmb": round(sum(cost_values), 10) if cost_values else None,
        "conservative_reserved_rmb": round(sum(reservation_values), 10),
        "mean_latency_ms": round(statistics.mean(latencies), 3) if latencies else None,
        "max_latency_ms": round(max(latencies), 3) if latencies else None,
    }


def score_agent1_1(tasks, outcomes, *, api_events):
    """Score model responses without changing or repairing frozen ground truth."""
    if type(tasks) is not dict or type(outcomes) is not dict or type(api_events) is not list:
        raise ValueError("invalid Agent 1.1 scoring inputs")
    verdict_passed = 0
    verdict_assessed = 0
    class_counts = {label: {"passed": 0, "assessed": 0} for label in _VERDICTS}
    dimension_passed = Counter()
    dimension_assessed = Counter()
    confusion = {expected: {predicted: 0 for predicted in _CONFUSION_COLUMNS}
                 for expected in _VERDICTS}
    errors = Counter({key: 0 for key in _ERRORS})
    excluded = Counter()
    invalid_count = semantic_responses = api_errors = not_run = uncertain = 0
    consistency_passed = consistency_assessed = inconsistencies = 0
    object_mismatch_presence_assessed = object_mismatch_na_violations = 0
    na_counts = Counter()

    for task_id, task in tasks.items():
        if type(task) is not dict:
            raise ValueError("invalid Agent 1.1 task")
        if task.get("include_in_headline") is not True:
            reason = task.get("exclusion_reason")
            if reason not in ("GT_UNCERTAIN", "SCHEMA_UNSUPPORTED"):
                raise ValueError("excluded task has no supported exclusion reason")
            excluded[reason] += 1
            errors[reason] += 1
            continue
        expected_verdict = task.get("expected_verdict")
        expected_dimensions = task.get("dimensions")
        if (expected_verdict not in _VERDICTS or type(expected_dimensions) is not dict
                or set(expected_dimensions) != set(DIMENSION_KEYS)):
            raise ValueError("included task lacks complete semantic ground truth")
        row = outcomes.get(task_id, {"status": "not_run_after_stop"})
        if type(row) is not dict:
            row = {"status": "invalid_output", "error_code": "malformed_saved_result"}
        status = row.get("status")
        if status == "api_error":
            api_errors += 1
            continue
        if status == "not_run_after_stop":
            not_run += 1
            continue
        if status not in ("compared", "semantic_uncertain", "invalid_output"):
            invalid_count += 1
            semantic_responses += 1
            verdict_assessed += 1
            class_counts[expected_verdict]["assessed"] += 1
            confusion[expected_verdict]["INVALID"] += 1
            for dimension in DIMENSION_KEYS:
                dimension_assessed[dimension] += 1
            errors["INVALID_OUTPUT"] += 1
            continue

        semantic_responses += 1
        verdict_assessed += 1
        class_counts[expected_verdict]["assessed"] += 1

        if status == "invalid_output":
            invalid_count += 1
            confusion[expected_verdict]["INVALID"] += 1
            for dimension in DIMENSION_KEYS:
                dimension_assessed[dimension] += 1
            errors["INVALID_OUTPUT"] += 1
            if row.get("error_code") == "verdict_dimension_inconsistency":
                consistency_assessed += 1
                inconsistencies += 1
                errors["VERDICT_DIMENSION_INCONSISTENCY"] += 1
            continue

        try:
            semantic = validate_semantic_output_v2(row.get("semantic"))
        except InvalidSemanticOutputV2 as exc:
            invalid_count += 1
            confusion[expected_verdict]["INVALID"] += 1
            for dimension in DIMENSION_KEYS:
                dimension_assessed[dimension] += 1
            errors["INVALID_OUTPUT"] += 1
            if exc.reason_code == "verdict_dimension_inconsistency":
                consistency_assessed += 1
                inconsistencies += 1
                errors["VERDICT_DIMENSION_INCONSISTENCY"] += 1
            continue

        consistency_assessed += 1
        consistency_passed += 1
        predicted_verdict = semantic["verdict"]
        is_uncertain = status == "semantic_uncertain" or semantic["comparability_basis"] == "uncertain"
        if is_uncertain:
            uncertain += 1
            errors["SEMANTIC_UNCERTAIN"] += 1
            confusion[expected_verdict]["UNCERTAIN"] += 1
        else:
            confusion[expected_verdict][predicted_verdict] += 1
            if predicted_verdict == expected_verdict:
                verdict_passed += 1
                class_counts[expected_verdict]["passed"] += 1
            else:
                errors["VERDICT_ERROR"] += 1

        for dimension in DIMENSION_KEYS:
            actual = semantic["dimensions"][dimension]
            dimension_assessed[dimension] += 1
            if actual == expected_dimensions[dimension]:
                dimension_passed[dimension] += 1
            else:
                errors[_DIMENSION_ERRORS[dimension]] += 1
            if actual == "not_applicable":
                na_counts[dimension] += 1

        if (expected_dimensions["object_or_field"] == "different"
                and expected_dimensions["value"] != "not_applicable"):
            object_mismatch_presence_assessed += 1
            if semantic["dimensions"]["value"] == "not_applicable":
                object_mismatch_na_violations += 1
        if (expected_dimensions["object_or_field"] == "different"
                and expected_dimensions["unit"] != "not_applicable"):
            object_mismatch_presence_assessed += 1
            if semantic["dimensions"]["unit"] == "not_applicable":
                object_mismatch_na_violations += 1

    class_metrics = {
        verdict: _accuracy(values["passed"], values["assessed"])
        for verdict, values in class_counts.items()
    }
    total_headline = sum(1 for task in tasks.values()
                         if type(task) is dict and task.get("include_in_headline") is True)
    dimension_metrics = {
        dimension: _accuracy(dimension_passed[dimension], dimension_assessed[dimension])
        for dimension in DIMENSION_KEYS
    }
    assessed_consistency = consistency_assessed
    return {
        "benchmark_tasks": len(tasks),
        "headline_tasks": total_headline,
        "excluded_tasks": dict(excluded),
        "semantic_responses": semantic_responses,
        "api_errors": api_errors,
        "not_run_after_stop": not_run,
        "semantic_uncertain_outputs": uncertain,
        "invalid_semantic_outputs": invalid_count,
        "verdict_accuracy": _accuracy(verdict_passed, verdict_assessed),
        "verdict_accuracy_by_class": class_metrics,
        "confusion_matrix": confusion,
        "dimension_accuracy": dimension_metrics,
        "verdict_dimension_consistency": {
            "passed": consistency_passed,
            "assessed": assessed_consistency,
            "inconsistencies": inconsistencies,
            "consistency": _ratio(consistency_passed, assessed_consistency),
        },
        "not_applicable_diagnostics": {
            "counts_by_dimension": {dimension: na_counts[dimension]
                                     for dimension in DIMENSION_KEYS},
            "object_mismatch_known_value_unit_na": {
                "violations": object_mismatch_na_violations,
                "assessed": object_mismatch_presence_assessed,
            },
        },
        "error_taxonomy": dict(errors),
        "api_usage": _api_usage(api_events),
    }
