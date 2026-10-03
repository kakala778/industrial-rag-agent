"""Run one opt-in semantic pass over saved Agent 0.3 host references.

Usage: python evaluation/evaluate_agent1_semantic.py freeze-gt
       python evaluation/evaluate_agent1_semantic.py run
"""

import sys
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.agent1_semantic_metrics import (
    evaluate_task,
    preflight_pair,
    score_semantic_run,
)
from src.agent.semantic import (
    MODEL_VERDICTS,
    DIMENSION_LABELS,
    DeepSeekSemanticComparator,
    SemanticAPIError,
    SemanticCostBudget,
)


APP_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = APP_ROOT / "outputs"
PRIVATE_ROOT = OUTPUT_ROOT / "agent1"
EXPECTED_TASK_IDS = tuple(f"R0{i}" for i in range(1, 9))
EXPECTED_SEMANTIC_TASK_IDS = frozenset({"R01", "R03", "R06", "R07"})
EXPECTED_ROUTES = {
    "R01": "semantic", "R02": "retrieval_bound", "R03": "semantic",
    "R04": "retrieval_bound", "R05": "unsupported_side", "R06": "semantic",
    "R07": "semantic", "R08": "unresolved_scope",
}
EXPECTED_AUTHORITY = "AI-assisted reviewed provisional GT"
EXPECTED_PRICING_SOURCE = "https://api-docs.deepseek.com/zh-cn/quick_start/pricing"
DIMENSION_GT_KEYS = (
    "expected_object_alignment", "expected_value_alignment",
    "expected_unit_alignment", "expected_condition_alignment",
)


class FrozenArtifactError(RuntimeError):
    """An input, provisional GT, or previous inference pass is not frozen as expected."""


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _read_json(path):
    try:
        return json.loads(Path(path).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError("unable to read frozen JSON artifact") from exc


def _file_sha256(path):
    return _sha256(Path(path).read_bytes())


def default_input_paths(app_root=APP_ROOT):
    root = Path(app_root)
    return {
        "frozen_tasks": root / "outputs" / "agent0_1" / "frozen_set.json",
        "repaired_gt": root / "outputs" / "agent0_3" / "repaired-gt-manifest.local.json",
        "candidate_reviews": root / "outputs" / "agent0_3" / "candidate-reviews.repaired.local.json",
        "gt_repair_manifest": root / "outputs" / "agent0_3" / "gt-repair-manifest.local.json",
        "agent03_rescore": root / "outputs" / "agent0_3" / "rescore-results.local.json",
        "agent03_results": (root / "outputs" / "agent0_3" / "run_23bb9cf5f607494e94ef22b25d4fdc17"
                             / "results.json"),
        "deepseek_pricing": root / "outputs" / "agent0_2" / "pricing.json",
    }


def _protected_inventory(manifest_path):
    manifest = _read_json(manifest_path)
    roots = manifest.get("protected_roots")
    expected = manifest.get("protected_hashes")
    if type(roots) is not list or type(expected) is not dict:
        raise FrozenArtifactError("frozen protected-input manifest is invalid")
    current = {}
    for root_value in roots:
        root = Path(root_value)
        if not root.is_dir():
            raise FrozenArtifactError("frozen protected-input root is missing")
        for path in root.rglob("*"):
            if path.is_file():
                resolved = str(path.resolve())
                current[resolved] = _file_sha256(path)
    if current != expected:
        raise FrozenArtifactError("protected frozen inputs changed")
    return current


class FrozenInputGuard:
    """Hash saved artifacts and the existing protected input inventory, then recheck."""

    def __init__(self, input_paths, protected_manifest_path=None):
        self.input_paths = {name: Path(path).resolve() for name, path in input_paths.items()}
        if not self.input_paths or any(not path.is_file() for path in self.input_paths.values()):
            raise FrozenArtifactError("one or more frozen source files are missing")
        self.hashes = {name: _file_sha256(path) for name, path in self.input_paths.items()}
        self.protected_manifest_path = (Path(protected_manifest_path).resolve()
                                        if protected_manifest_path else None)
        self.protected_hashes = (_protected_inventory(self.protected_manifest_path)
                                 if self.protected_manifest_path else {})

    def verify(self):
        if any(_file_sha256(path) != self.hashes[name]
               for name, path in self.input_paths.items()):
            raise FrozenArtifactError("frozen source artifact changed during the run")
        if self.protected_manifest_path:
            current = _protected_inventory(self.protected_manifest_path)
            if current != self.protected_hashes:
                raise FrozenArtifactError("protected inputs changed during the run")


def _validate_ground_truth(proposal):
    if type(proposal) is not dict:
        raise FrozenArtifactError("semantic GT must be a JSON object")
    if proposal.get("schema_version") != "agent1-semantic-gt-v1":
        raise FrozenArtifactError("unsupported semantic GT schema")
    if proposal.get("authority") != EXPECTED_AUTHORITY:
        raise FrozenArtifactError("semantic GT must remain provisional and AI-assisted")
    tasks = proposal.get("tasks")
    if type(tasks) is not dict or set(tasks) != set(EXPECTED_TASK_IDS):
        raise FrozenArtifactError("semantic GT must cover the exact frozen task set")
    routes = {}
    for task_id, row in tasks.items():
        if type(row) is not dict:
            raise FrozenArtifactError("semantic GT task row is invalid")
        route = row.get("expected_route")
        routes[task_id] = route
        if route == "semantic":
            required = {"expected_route", "expected_verdict", *DIMENSION_GT_KEYS,
                        "confidence", "review_note"}
            if set(row) != required:
                raise FrozenArtifactError("semantic GT fields do not match schema")
            if row["expected_verdict"] not in MODEL_VERDICTS:
                raise FrozenArtifactError("semantic GT verdict is invalid")
            if row["confidence"] not in ("high", "medium", "low", "uncertain"):
                raise FrozenArtifactError("semantic GT confidence is invalid")
            if type(row["review_note"]) is not str or len(row["review_note"]) > 300:
                raise FrozenArtifactError("semantic GT review note is invalid")
            if any(type(row[key]) is not str or row[key] not in DIMENSION_LABELS
                   for key in DIMENSION_GT_KEYS):
                raise FrozenArtifactError("semantic GT dimension is invalid")
        elif route in ("retrieval_bound", "unsupported_side", "unresolved_scope"):
            if set(row) != {"expected_route"}:
                raise FrozenArtifactError("preflight GT must not contain semantic labels")
        else:
            raise FrozenArtifactError("semantic GT route is invalid")
    if frozenset(task_id for task_id, route in routes.items() if route == "semantic") != EXPECTED_SEMANTIC_TASK_IDS:
        raise FrozenArtifactError("semantic GT cohort does not match the saved bilateral cohort")
    if routes != EXPECTED_ROUTES:
        raise FrozenArtifactError("semantic GT preflight routes do not match frozen task design")
    return proposal


def _source_hashes(input_paths):
    result = {}
    for name, path in input_paths.items():
        path = Path(path)
        if not path.is_file():
            raise FrozenArtifactError("a source hash input is missing")
        result[name] = _file_sha256(path)
    return result


def _ground_truth_hash_path(frozen_path):
    return Path(str(frozen_path) + ".sha256")


def freeze_semantic_ground_truth(proposal_path, frozen_path, input_paths, results_path):
    """Freeze provisional labels and source hashes before any inference artifact exists."""
    frozen_path = Path(frozen_path)
    hash_path = _ground_truth_hash_path(frozen_path)
    if Path(results_path).exists() or frozen_path.exists() or hash_path.exists():
        raise FrozenArtifactError("refusing to overwrite GT or freeze after inference")
    proposal = _validate_ground_truth(_read_json(proposal_path))
    record = deepcopy(proposal)
    record["source_hashes"] = _source_hashes(input_paths)
    record["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    raw = (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    digest = _sha256(raw)
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with frozen_path.open("xb") as stream:
            stream.write(raw)
        with hash_path.open("x", encoding="ascii", newline="\n") as stream:
            stream.write(digest + "\n")
    except OSError as exc:
        raise FrozenArtifactError("unable to freeze semantic GT") from exc
    return digest


def verify_frozen_semantic_ground_truth(frozen_path, input_paths):
    frozen_path = Path(frozen_path)
    hash_path = _ground_truth_hash_path(frozen_path)
    try:
        raw = frozen_path.read_bytes()
        expected_digest = hash_path.read_text(encoding="ascii").strip()
    except OSError as exc:
        raise FrozenArtifactError("frozen semantic GT or hash is missing") from exc
    actual_digest = _sha256(raw)
    if expected_digest != actual_digest:
        raise FrozenArtifactError("frozen semantic GT hash changed")
    record = _read_json(frozen_path)
    _validate_ground_truth(record)
    if record.get("source_hashes") != _source_hashes(input_paths):
        raise FrozenArtifactError("semantic GT source hashes no longer match")
    if type(record.get("frozen_at_utc")) is not str or not record["frozen_at_utc"]:
        raise FrozenArtifactError("semantic GT freeze time is missing")
    return record, actual_digest


def _input_records(paths):
    return {name: _read_json(path) for name, path in paths.items()}


def validate_artifacts(paths, hashes=None):
    """Bind the Agent 0.3 accepted run to its repaired, offline-reviewed inputs."""
    records = _input_records(paths)
    hashes = hashes or {name: _file_sha256(path) for name, path in paths.items()}
    frozen = records["frozen_tasks"]
    repaired = records["repaired_gt"]
    reviews = records["candidate_reviews"]
    repair = records["gt_repair_manifest"]
    rescore = records["agent03_rescore"]
    run = records["agent03_results"]
    pricing = records["deepseek_pricing"]

    if frozen.get("version") != "agent01-industrial-v1" or repaired.get("version") != frozen.get("version"):
        raise FrozenArtifactError("the frozen Agent 0.1 benchmark is required")
    frozen_tasks = frozen.get("tasks")
    repaired_tasks = repaired.get("tasks")
    if (type(frozen_tasks) is not list or type(repaired_tasks) is not list
            or tuple(row.get("id") for row in frozen_tasks) != EXPECTED_TASK_IDS
            or tuple(row.get("id") for row in repaired_tasks) != EXPECTED_TASK_IDS):
        raise FrozenArtifactError("the exact frozen eight-task benchmark is required")
    for original, current in zip(frozen_tasks, repaired_tasks):
        if any(original.get(key) != current.get(key)
               for key in ("id", "query", "documents", "scopes", "category")):
            raise FrozenArtifactError("repaired GT changed frozen task intent or scope")
    if (frozen.get("protected_roots") != repaired.get("protected_roots")
            or frozen.get("protected_hashes") != repaired.get("protected_hashes")):
        raise FrozenArtifactError("repaired GT changed protected frozen inputs")
    if (reviews.get("manifest_sha256") != hashes["repaired_gt"]
            or reviews.get("review_status") != "ai_pdf_reviewed"
            or reviews.get("authority") != "ai_assisted_original_pdf_review"):
        raise FrozenArtifactError("repaired candidate review is not bound to the repaired manifest")
    if rescore.get("repair_manifest_sha256") != hashes["gt_repair_manifest"]:
        raise FrozenArtifactError("offline rescore is not bound to the current repair manifest")
    if (rescore.get("inference_performed") is not False
            or rescore.get("retrieval_performed") is not False
            or rescore.get("protected_inputs_unchanged") is not True
            or rescore.get("review_status") != "ai_pdf_reviewed"):
        raise FrozenArtifactError("offline repaired rescore integrity checks did not pass")
    artifact_hashes = rescore.get("source_artifact_sha256", {})
    if (artifact_hashes.get("agent03_raw") != hashes["agent03_results"]
            or artifact_hashes.get("frozen_manifest") != hashes["frozen_tasks"]):
        raise FrozenArtifactError("offline rescore does not match the saved Agent 0.3 run")
    if (run.get("contract") != "evidence_reference"
            or run.get("protected_inputs_unchanged") is not True
            or run.get("forbidden_operations") not in ({}, None)):
        raise FrozenArtifactError("saved Agent 0.3 reference run failed integrity checks")
    task_rows = run.get("arms", {}).get("B")
    rescore_rows = rescore.get("agent03", {}).get("tasks")
    if (type(task_rows) is not dict or set(task_rows) != set(EXPECTED_TASK_IDS)
            or type(rescore_rows) is not dict or set(rescore_rows) != set(EXPECTED_TASK_IDS)):
        raise FrozenArtifactError("saved Agent 0.3 task set does not match the frozen benchmark")
    if (pricing.get("model") != "deepseek-flash" or pricing.get("currency") != "CNY"
            or pricing.get("source") != EXPECTED_PRICING_SOURCE
            or pricing.get("peak_input_miss_rmb") != 2.0
            or pricing.get("peak_output_rmb") != 8.0):
        raise FrozenArtifactError("official peak-rate pricing snapshot is required")
    return records


def _validate_gt_routes(ground_truth, tasks, saved_rows, rescore_rows):
    prepared = {}
    for task_id in EXPECTED_TASK_IDS:
        task = tasks[task_id]
        row = saved_rows[task_id]
        decision = preflight_pair(task, row)
        route = ground_truth["tasks"][task_id]["expected_route"]
        if (route == "semantic") != decision["eligible"]:
            raise FrozenArtifactError("frozen semantic cohort does not match accepted references")
        if route == "retrieval_bound":
            if rescore_rows[task_id].get("candidate_group") != "RETRIEVAL_BOUND":
                raise FrozenArtifactError("retrieval-bound preflight label is not confirmed offline")
        elif route == "unsupported_side":
            oracle = tasks[task_id].get("oracle", {})
            if not any(value.get("expected_available") is False for value in oracle.values()):
                raise FrozenArtifactError("unsupported-side route lacks repaired GT support")
        elif route == "unresolved_scope":
            if row.get("state", {}).get("status") != "clarify":
                raise FrozenArtifactError("unresolved-scope route does not match saved Agent status")
        prepared[task_id] = decision
    return prepared


def _private_output_path(path):
    path = Path(path).resolve()
    root = OUTPUT_ROOT.resolve()
    if not path.is_relative_to(root):
        raise FrozenArtifactError("private Agent 1 output must stay under ignored outputs/")
    return path


def _atomic_write_json(path, value):
    path = _private_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=".agent1-",
                                     suffix=".tmp", delete=False) as stream:
        temp_path = Path(stream.name)
        stream.write((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, path)


def freeze_current_ground_truth(app_root=APP_ROOT):
    root = Path(app_root)
    paths = default_input_paths(root)
    guard = FrozenInputGuard(paths, paths["frozen_tasks"])
    records = validate_artifacts(paths, guard.hashes)
    proposal_path = root / "outputs" / "agent1" / "semantic-gt-proposal.local.json"
    frozen_path = root / "outputs" / "agent1" / "semantic-gt.local.json"
    results_path = root / "outputs" / "agent1" / "results.local.json"
    if not proposal_path.is_file():
        raise FrozenArtifactError("private semantic GT proposal is missing")
    tasks = {row["id"]: row for row in records["repaired_gt"]["tasks"]}
    _validate_gt_routes(_validate_ground_truth(_read_json(proposal_path)), tasks,
                        records["agent03_results"]["arms"]["B"],
                        records["agent03_rescore"]["agent03"]["tasks"])
    guard.verify()
    digest = freeze_semantic_ground_truth(proposal_path, frozen_path, paths, results_path)
    guard.verify()
    return digest


def _empty_task_result(task_id, decision, status="not_run"):
    references = decision["host_references"]
    citation_count = sum(len(rows) for rows in references.values())
    return {
        "task_id": task_id,
        "status": status,
        "preflight": {key: value for key, value in decision.items()
                      if key != "host_references"},
        "host_references": references,
        "citation_grounding": {"passed": citation_count, "assessed": citation_count},
        "semantic": None,
        "verdict": None,
        "formal_verdict": None,
        "formal_verdict_scored": False,
    }


def rescore_saved_record(results_path, ground_truth, expected_gt_hash):
    """Recompute metrics from one saved pass without running the comparator again."""
    results_path = Path(results_path)
    record = _read_json(results_path)
    if record.get("semantic_gt_sha256") != expected_gt_hash:
        raise FrozenArtifactError("saved inference used a different semantic GT hash")
    record["metrics"] = score_semantic_run(
        record.get("tasks", {}), ground_truth, api_events=record.get("api_events", []))
    if "budget_reserved_rmb" in record:
        record["budget_accounted_rmb"] = record.pop("budget_reserved_rmb")
    record["rescored_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(results_path, record)
    return record


def rescore_current_results(app_root=APP_ROOT):
    """Verify frozen inputs and recalculate only the metrics for the existing pass."""
    root = Path(app_root)
    paths = default_input_paths(root)
    guard = FrozenInputGuard(paths, paths["frozen_tasks"])
    records = validate_artifacts(paths, guard.hashes)
    gt_path = _private_output_path(root / "outputs" / "agent1" / "semantic-gt.local.json")
    results_path = _private_output_path(root / "outputs" / "agent1" / "results.local.json")
    ground_truth, gt_hash = verify_frozen_semantic_ground_truth(gt_path, paths)
    saved = _read_json(results_path)
    if saved.get("input_sha256") != guard.hashes:
        raise FrozenArtifactError("saved inference is bound to different frozen inputs")
    if saved.get("semantic_gt_sha256") != gt_hash:
        raise FrozenArtifactError("saved inference GT changed after the pass")
    _validate_gt_routes(
        ground_truth,
        {row["id"]: row for row in records["repaired_gt"]["tasks"]},
        records["agent03_results"]["arms"]["B"],
        records["agent03_rescore"]["agent03"]["tasks"],
    )
    guard.verify()
    updated = rescore_saved_record(results_path, ground_truth, gt_hash)
    guard.verify()
    _, verified_gt_hash = verify_frozen_semantic_ground_truth(gt_path, paths)
    if verified_gt_hash != gt_hash:
        raise FrozenArtifactError("semantic GT changed during offline rescore")
    return updated


def run_semantic_experiment(app_root=APP_ROOT):
    root = Path(app_root)
    paths = default_input_paths(root)
    output_path = _private_output_path(root / "outputs" / "agent1" / "results.local.json")
    gt_path = _private_output_path(root / "outputs" / "agent1" / "semantic-gt.local.json")
    if output_path.exists():
        raise FrozenArtifactError("a pass already exists; refusing a second inference run")
    if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        raise SemanticAPIError("deepseek:missing_key")
    guard = FrozenInputGuard(paths, paths["frozen_tasks"])
    records = validate_artifacts(paths, guard.hashes)
    ground_truth, gt_hash = verify_frozen_semantic_ground_truth(gt_path, paths)
    tasks = {row["id"]: row for row in records["repaired_gt"]["tasks"]}
    saved_rows = records["agent03_results"]["arms"]["B"]
    rescore_rows = records["agent03_rescore"]["agent03"]["tasks"]
    prepared = _validate_gt_routes(ground_truth, tasks, saved_rows, rescore_rows)
    if {tid for tid, decision in prepared.items() if decision["eligible"]} != EXPECTED_SEMANTIC_TASK_IDS:
        raise FrozenArtifactError("bilateral semantic cohort changed")
    budget = SemanticCostBudget(soft_limit_rmb=3.0, hard_limit_rmb=5.0)
    comparator = DeepSeekSemanticComparator(max_tokens=384, budget=budget)
    output = {
        "schema_version": "agent1-semantic-run-v1",
        "run_status": "in_progress",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "authority": ground_truth["authority"],
        "semantic_gt_sha256": gt_hash,
        "input_sha256": guard.hashes,
        "protected_input_files": len(guard.protected_hashes),
        "model": "deepseek-flash",
        "thinking": "disabled",
        "temperature": 0,
        "response_format": "json_object",
        "max_tokens": comparator.max_tokens,
        "cost_limits_rmb": {"soft": 3.0, "hard": 5.0},
        "pricing_snapshot": {
            "source": records["deepseek_pricing"]["source"],
            "verified_on": records["deepseek_pricing"]["verified_on"],
            "peak_input_cache_hit_rmb_per_million": SemanticCostBudget.INPUT_HIT_RMB_PER_MILLION,
            "peak_input_cache_miss_rmb_per_million": SemanticCostBudget.INPUT_MISS_RMB_PER_MILLION,
            "peak_output_rmb_per_million": SemanticCostBudget.OUTPUT_RMB_PER_MILLION,
        },
        "retrieval_performed": False,
        "citation_renderer_called": False,
        "forbidden_operations": {},
        "tasks": {},
        "api_events": [],
        "stop_reason": None,
    }
    for task_id in EXPECTED_TASK_IDS:
        decision = prepared[task_id]
        if not decision["eligible"]:
            output["tasks"][task_id] = evaluate_task(tasks[task_id], saved_rows[task_id], None)
        else:
            output["tasks"][task_id] = _empty_task_result(task_id, decision)
    _atomic_write_json(output_path, output)

    for task_id in EXPECTED_TASK_IDS:
        if not prepared[task_id]["eligible"]:
            continue
        output["tasks"][task_id]["status"] = "request_pending"
        _atomic_write_json(output_path, output)
        try:
            output["tasks"][task_id] = evaluate_task(
                tasks[task_id], saved_rows[task_id], comparator)
        except SemanticAPIError as exc:
            output["tasks"][task_id]["status"] = "api_error"
            code = str(exc)
            output["tasks"][task_id]["error_code"] = (code if code.startswith("deepseek:")
                                                        else "deepseek:api_error")
            output["stop_reason"] = output["tasks"][task_id]["error_code"]
            output["api_events"] = deepcopy(comparator.events)
            _atomic_write_json(output_path, output)
            break
        output["api_events"] = deepcopy(comparator.events)
        output["budget_accounted_rmb"] = round(budget.reserved_rmb, 9)
        _atomic_write_json(output_path, output)

    for task_id in EXPECTED_TASK_IDS:
        if output["tasks"][task_id]["status"] == "not_run":
            output["tasks"][task_id]["status"] = "not_run_after_stop"
    guard.verify()
    _, post_gt_hash = verify_frozen_semantic_ground_truth(gt_path, paths)
    if post_gt_hash != gt_hash:
        raise FrozenArtifactError("semantic GT changed during inference")
    output["protected_inputs_unchanged"] = True
    output["semantic_gt_unchanged"] = True
    output["budget_accounted_rmb"] = round(budget.reserved_rmb, 9)
    output["metrics"] = score_semantic_run(
        output["tasks"], ground_truth, api_events=output["api_events"])
    output["run_status"] = ("complete" if all(
        row["status"] in ("compared", "preflight_insufficient_evidence",
                           "invalid_output", "unit_conversion_required")
        for row in output["tasks"].values()) else "stopped")
    output["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(output_path, output)
    return output


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze-gt", "run", "rescore"))
    args = parser.parse_args()
    if args.command == "freeze-gt":
        digest = freeze_current_ground_truth()
        print("Frozen semantic GT SHA-256: " + digest)
        print("No model inference was performed.")
    elif args.command == "run":
        result = run_semantic_experiment()
        print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
        print("Private results: outputs/agent1/results.local.json")
    else:
        result = rescore_current_results()
        print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
        print("No model inference was performed.")
        print("Private results: outputs/agent1/results.local.json")


if __name__ == "__main__":
    main()
