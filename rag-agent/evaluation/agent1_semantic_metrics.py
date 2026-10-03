"""Offline routing, saved-reference validation, and Agent 1 metrics."""

from collections import Counter
from copy import deepcopy
import hashlib
import re
import statistics

from src.agent.semantic import InvalidSemanticOutput, validate_semantic_output


class SavedReferenceError(ValueError):
    """Saved Agent 0.3 findings do not match their accepted host FINISH."""


_EVIDENCE_ID = re.compile(r"ev_[0-9a-f]{64}\Z")
_REFERENCE_KEYS = frozenset({
    "evidence_id", "source", "page", "block_type", "block_index", "chunk_id",
    "excerpt", "span", "excerpt_sha256", "context_chars",
})
_SUPPORTED = "supported"
_UNSUPPORTED = frozenset({"insufficient_evidence", "no_candidates"})
_DIMENSION_TO_RESULT = {
    "object_or_field": "expected_object_alignment",
    "value": "expected_value_alignment",
    "unit": "expected_unit_alignment",
    "condition_or_applicability": "expected_condition_alignment",
}
_FAILURE_TYPES = (
    "EVIDENCE_UPSTREAM_FAILURE",
    "SEMANTIC_FIELD_MISALIGNMENT",
    "VALUE_COMPARISON_ERROR",
    "UNIT_COMPARISON_ERROR",
    "CONDITION_OMISSION",
    "NOT_COMPARABLE_FALSE_POSITIVE",
    "NOT_COMPARABLE_FALSE_NEGATIVE",
    "INVALID_OUTPUT",
    "GT_UNCERTAIN",
)


def _validate_reference(reference, scope):
    if type(reference) is not dict or set(reference) != _REFERENCE_KEYS:
        raise SavedReferenceError("saved host reference fields are invalid")
    evidence_id = reference.get("evidence_id")
    if type(evidence_id) is not str or not _EVIDENCE_ID.fullmatch(evidence_id):
        raise SavedReferenceError("saved evidence ID is invalid")
    if reference.get("source") != scope:
        raise SavedReferenceError("saved reference has a cross-scope source")
    if (type(reference.get("page")) is not int or reference["page"] < 1
            or type(reference.get("block_index")) is not int or reference["block_index"] < 0
            or type(reference.get("chunk_id")) is not int or reference["chunk_id"] < 0
            or type(reference.get("context_chars")) is not int or reference["context_chars"] < 0
            or type(reference.get("block_type")) is not str or not reference["block_type"]):
        raise SavedReferenceError("saved reference provenance is invalid")
    excerpt = reference.get("excerpt")
    span = reference.get("span")
    if (type(excerpt) is not str or not excerpt.strip() or len(excerpt) > 1200
            or type(span) is not dict or span.get("length") != len(excerpt)):
        raise SavedReferenceError("saved reference span is invalid")
    expected_hash = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
    if reference.get("excerpt_sha256") != expected_hash:
        raise SavedReferenceError("saved reference excerpt hash does not match")
    return reference


def validate_saved_references(task, saved_row):
    """Return copied saved citations after matching them exactly to accepted FINISH IDs."""
    if type(task) is not dict or type(saved_row) is not dict:
        raise SavedReferenceError("invalid task or saved Agent 0.3 row")
    state = saved_row.get("state")
    if type(state) is not dict:
        raise SavedReferenceError("saved Agent state is missing")
    scopes = task.get("scopes")
    if type(scopes) is not list or len(set(scopes)) != len(scopes):
        raise SavedReferenceError("task scopes are invalid")
    actions = saved_row.get("actions")
    findings = state.get("findings")
    if type(actions) is not list or type(findings) is not list:
        raise SavedReferenceError("saved actions or findings are invalid")
    finishes = [action for action in actions
                if type(action) is dict and action.get("action") == "FINISH"]
    if state.get("status") != "finished":
        if finishes or findings:
            raise SavedReferenceError("unfinished task contains accepted findings")
        return {scope: [] for scope in scopes}
    if len(finishes) != 1 or not scopes:
        raise SavedReferenceError("finished task must have one scoped FINISH")
    outcomes = finishes[0].get("outcomes")
    if type(outcomes) is not list:
        raise SavedReferenceError("FINISH outcomes are invalid")
    outcome_by_scope = {}
    for outcome in outcomes:
        if type(outcome) is not dict or outcome.get("scope") not in scopes:
            raise SavedReferenceError("FINISH contains an unexpected scope")
        scope = outcome["scope"]
        if scope in outcome_by_scope:
            raise SavedReferenceError("FINISH repeats a scope")
        outcome_by_scope[scope] = outcome
    if set(outcome_by_scope) != set(scopes):
        raise SavedReferenceError("FINISH does not cover the task scopes")
    finding_by_scope = {}
    for finding in findings:
        if type(finding) is not dict or finding.get("scope") not in scopes:
            raise SavedReferenceError("saved finding contains an unexpected scope")
        scope = finding["scope"]
        if scope in finding_by_scope:
            raise SavedReferenceError("saved findings repeat a scope")
        finding_by_scope[scope] = finding
    if set(finding_by_scope) != set(scopes):
        raise SavedReferenceError("saved findings do not cover the task scopes")

    result = {}
    for scope in scopes:
        outcome = outcome_by_scope[scope]
        finding = finding_by_scope[scope]
        status = outcome.get("status")
        if status != finding.get("status") or status != _SUPPORTED and status not in _UNSUPPORTED:
            raise SavedReferenceError("FINISH and saved finding status disagree")
        evidence_ids = outcome.get("evidence_ids", [])
        evidence = finding.get("evidence", [])
        if type(evidence_ids) is not list or type(evidence) is not list:
            raise SavedReferenceError("saved evidence references are invalid")
        if status in _UNSUPPORTED:
            if evidence_ids or evidence:
                raise SavedReferenceError("unsupported scope cannot cite evidence")
            result[scope] = []
            continue
        if not evidence_ids or len(evidence_ids) != len(evidence):
            raise SavedReferenceError("supported scope must carry every accepted reference")
        checked = []
        for expected_id, reference in zip(evidence_ids, evidence):
            if expected_id != reference.get("evidence_id"):
                raise SavedReferenceError("FINISH ID does not match host reference")
            checked.append(_validate_reference(reference, scope))
        if len({row["evidence_id"] for row in checked}) != len(checked):
            raise SavedReferenceError("saved scope repeats a reference ID")
        result[scope] = deepcopy(checked)
    return result


def preflight_pair(task, saved_row):
    """Route to the model only when both requested scopes have saved accepted references."""
    references = validate_saved_references(task, saved_row)
    scopes = task["scopes"]
    eligible = (scopes == ["A", "B"]
                and all(references.get(scope) for scope in ("A", "B")))
    if eligible:
        return {"eligible": True, "verdict": "NOT_EVALUATED",
                "reason": "bilateral_supported", "host_references": references}
    unresolved = not scopes or saved_row["state"].get("status") == "clarify"
    return {"eligible": False, "verdict": "INSUFFICIENT_EVIDENCE",
            "reason": "unresolved_scope" if unresolved else "not_bilateral_supported",
            "host_references": references}


def evaluate_task(task, saved_row, comparator):
    """Evaluate one saved task; the model receives excerpts but no IDs or citations."""
    preflight = preflight_pair(task, saved_row)
    host_references = preflight["host_references"]
    citation_count = sum(len(rows) for rows in host_references.values())
    row = {
        "task_id": task["id"],
        "status": "pending",
        "preflight": {key: value for key, value in preflight.items()
                      if key != "host_references"},
        "host_references": host_references,
        "citation_grounding": {"passed": citation_count, "assessed": citation_count},
        "semantic": None,
        "verdict": None,
        "formal_verdict": None,
        "formal_verdict_scored": False,
    }
    if not preflight["eligible"]:
        row["status"] = "preflight_insufficient_evidence"
        row["verdict"] = "INSUFFICIENT_EVIDENCE"
        return row
    excerpts_a = [item["excerpt"] for item in host_references["A"]]
    excerpts_b = [item["excerpt"] for item in host_references["B"]]
    try:
        raw_result = comparator.compare(task["query"], excerpts_a, excerpts_b)
        semantic = validate_semantic_output(raw_result)
    except InvalidSemanticOutput:
        row["status"] = "invalid_output"
        return row
    row["semantic"] = semantic
    row["verdict"] = semantic["verdict"]
    if semantic["dimensions"]["unit"] == "different":
        row["status"] = "unit_conversion_required"
        return row
    row["status"] = "compared"
    row["formal_verdict"] = semantic["verdict"]
    row["formal_verdict_scored"] = True
    return row


def _count_summary(passed, assessed):
    return {"passed": passed, "assessed": assessed,
            "accuracy": passed / assessed if assessed else None}


def score_semantic_run(task_results, ground_truth, api_events=()):
    """Compute metrics against a previously frozen provisional semantic oracle."""
    gt_tasks = ground_truth.get("tasks", {})
    if set(task_results) != set(gt_tasks):
        raise ValueError("semantic result task IDs do not match frozen ground truth")
    verdict_passed = verdict_assessed = 0
    eligible_pair_count = 0
    confidence_counts = {}
    dimension_counts = {key: [0, 0] for key in _DIMENSION_TO_RESULT}
    invalid = 0
    preflight_count = 0
    citation_passed = citation_assessed = 0
    unsupported_passed = unsupported_assessed = 0
    taxonomy = Counter({key: 0 for key in _FAILURE_TYPES})
    expected_routes = Counter()
    actual_routes = Counter()

    for task_id, expected in gt_tasks.items():
        actual = task_results[task_id]
        route = expected.get("expected_route")
        expected_routes[route] += 1
        status = actual.get("status")
        preflight = actual.get("preflight", {})
        semantic_route = route == "semantic"
        eligible = bool(preflight.get("eligible"))
        eligible_pair_count += int(semantic_route and eligible)
        if not preflight.get("eligible"):
            preflight_count += 1
        actual_routes[preflight.get("reason", "missing")] += 1
        citation = actual.get("citation_grounding", {})
        citation_passed += int(citation.get("passed", 0))
        citation_assessed += int(citation.get("assessed", 0))

        if route == "unsupported_side":
            unsupported_assessed += 1
            correct = (status == "preflight_insufficient_evidence"
                       and not preflight.get("eligible")
                       and actual.get("verdict") == "INSUFFICIENT_EVIDENCE")
            unsupported_passed += int(correct)
        if route == "retrieval_bound" and not preflight.get("eligible"):
            taxonomy["EVIDENCE_UPSTREAM_FAILURE"] += 1
        semantic = actual.get("semantic")
        if semantic_route:
            route_mismatch = (not eligible or status == "preflight_insufficient_evidence")
        else:
            route_mismatch = (
                eligible
                or status != "preflight_insufficient_evidence"
                or actual.get("verdict") != "INSUFFICIENT_EVIDENCE"
                or semantic is not None
                or actual.get("formal_verdict") is not None
            )
        invalid_output = status == "invalid_output" or route_mismatch
        if invalid_output:
            invalid += 1
            taxonomy["INVALID_OUTPUT"] += 1
        elif semantic is not None:
            try:
                semantic = validate_semantic_output(semantic)
            except InvalidSemanticOutput:
                invalid_output = True
                invalid += 1
                taxonomy["INVALID_OUTPUT"] += 1
        elif status in ("compared", "unit_conversion_required"):
            invalid_output = True
            invalid += 1
            taxonomy["INVALID_OUTPUT"] += 1
        if not invalid_output:
            formal_verdict = actual.get("formal_verdict")
            if status == "compared":
                invalid_output = (semantic is None
                                  or semantic["dimensions"]["unit"] == "different"
                                  or formal_verdict != semantic["verdict"])
            elif status == "unit_conversion_required":
                invalid_output = (
                    semantic is None
                    or semantic["dimensions"]["unit"] != "different"
                    or formal_verdict is not None
                )
            elif status == "preflight_insufficient_evidence":
                invalid_output = semantic is not None or formal_verdict is not None
            else:
                invalid_output = semantic is not None or formal_verdict is not None
            if invalid_output:
                invalid += 1
                taxonomy["INVALID_OUTPUT"] += 1

        if expected.get("confidence") == "uncertain" or route == "gt_uncertain":
            taxonomy["GT_UNCERTAIN"] += 1
            continue
        if invalid_output:
            continue
        if type(semantic) is not dict:
            continue
        dimensions = semantic.get("dimensions", {})
        for dimension, expected_key in _DIMENSION_TO_RESULT.items():
            expected_dimension = expected.get(expected_key)
            actual_dimension = dimensions.get(dimension)
            if expected_dimension is None:
                continue
            dimension_counts[dimension][1] += 1
            dimension_counts[dimension][0] += int(actual_dimension == expected_dimension)
            if actual_dimension == expected_dimension:
                continue
            error_key = {
                "object_or_field": "SEMANTIC_FIELD_MISALIGNMENT",
                "value": "VALUE_COMPARISON_ERROR",
                "unit": "UNIT_COMPARISON_ERROR",
                "condition_or_applicability": "CONDITION_OMISSION",
            }[dimension]
            taxonomy[error_key] += 1
        expected_verdict = expected.get("expected_verdict")
        actual_verdict = actual.get("formal_verdict")
        if expected_verdict and actual_verdict is not None:
            verdict_assessed += 1
            matched_verdict = actual_verdict == expected_verdict
            verdict_passed += int(matched_verdict)
            confidence = expected.get("confidence", "unclassified")
            counts = confidence_counts.setdefault(confidence, [0, 0])
            counts[1] += 1
            counts[0] += int(matched_verdict)
            if (expected_verdict == "NOT_COMPARABLE"
                    and actual_verdict in ("EQUIVALENT", "DIFFERENT")):
                taxonomy["NOT_COMPARABLE_FALSE_NEGATIVE"] += 1
            elif (actual_verdict == "NOT_COMPARABLE"
                  and expected_verdict in ("EQUIVALENT", "DIFFERENT")):
                taxonomy["NOT_COMPARABLE_FALSE_POSITIVE"] += 1

    usage = Counter()
    known_costs = []
    reserved_upper_bounds = []
    latencies = []
    for event in api_events:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens",
                    "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
            value = event.get("usage", {}).get(key, 0)
            if type(value) is int and value >= 0:
                usage[key] += value
        if isinstance(event.get("cost_rmb"), (int, float)):
            known_costs.append(float(event["cost_rmb"]))
        if isinstance(event.get("reserved_rmb"), (int, float)):
            reserved_upper_bounds.append(float(event["reserved_rmb"]))
        if isinstance(event.get("latency_ms"), (int, float)):
            latencies.append(float(event["latency_ms"]))

    dimensions = {key: _count_summary(values[0], values[1])
                  for key, values in dimension_counts.items()}
    return {
        "semantic_model_tasks": len(api_events),
        "eligible_pair_count": eligible_pair_count,
        "preflight_insufficient_tasks": preflight_count,
        "verdict_accuracy": _count_summary(verdict_passed, verdict_assessed),
        "verdict_accuracy_by_confidence": {
            confidence: _count_summary(values[0], values[1])
            for confidence, values in sorted(confidence_counts.items())
        },
        "dimension_accuracy": dimensions,
        "invalid_semantic_outputs": invalid,
        "unsupported_side_routing_correctness": _count_summary(
            unsupported_passed, unsupported_assessed),
        "citation_grounding": {"passed": citation_passed, "assessed": citation_assessed},
        "api_usage": {
            "requests": len(api_events),
            "tokens": dict(usage),
            "cost_rmb": (round(sum(known_costs), 9)
                         if len(known_costs) == len(api_events) and api_events else None),
            "conservative_upper_rmb": (round(sum(reserved_upper_bounds), 9)
                                       if len(reserved_upper_bounds) == len(api_events)
                                       and api_events else None),
            "mean_latency_ms": round(statistics.mean(latencies), 3) if latencies else None,
            "max_latency_ms": max(latencies) if latencies else None,
        },
        "expected_routes": dict(expected_routes),
        "actual_preflight_reasons": dict(actual_routes),
        "error_taxonomy": dict(taxonomy),
    }
