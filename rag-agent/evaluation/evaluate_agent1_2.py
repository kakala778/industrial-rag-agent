"""Run and rescore one private Agent 1.2 Responses/json_schema pass."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.agent1_2_metrics import score_agent1_2
from evaluation.evaluate_agent1_1 import (
    FrozenArtifactError,
    SOURCE_MANIFEST_PATH,
    SOURCE_ROOT,
    _sha256_file,
    verify_frozen_agent1_1_ground_truth,
)
from src.agent.semantic import SemanticAPIError, SemanticCostBudget
from src.agent.semantic_contract_v2 import (
    InvalidSemanticOutputV2,
    parse_semantic_output_v2,
    semantic_messages_v2,
)
from src.agent.semantic_responses import DeepSeekResponsesJsonSchemaComparator


APP_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = APP_ROOT / "outputs" / "agent1_2"
FROZEN_GT_PATH = APP_ROOT / "outputs" / "agent1_1" / "benchmark-gt.local.json"
BASELINE_RESULTS_PATH = APP_ROOT / "outputs" / "agent1_1" / "run.local.json"
RESULTS_PATH = OUTPUT_ROOT / "run.local.json"
METRICS_PATH = OUTPUT_ROOT / "metrics.local.json"
RAW_RESPONSES_PATH = OUTPUT_ROOT / "raw-responses.local.jsonl"
RESULT_SCHEMA = "agent1-2-structured-output-run-v1"
METRICS_SCHEMA = "agent1-2-structured-output-metrics-v1"
OFFICIAL_PRICING_URL = "https://api-docs.deepseek.com/zh-cn/quick_start/pricing"
MAX_OUTPUT_TOKENS = 384


def _load_json(path, *, label):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError("unable to read local " + label) from exc


def _atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                        suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _reserve_output_paths(paths):
    resolved = _check_output_paths(paths)
    for path in resolved:
        path.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        for path in resolved:
            with path.open("xb"):
                pass
            created.append(path)
    except OSError:
        for path in created:
            try:
                path.unlink()
            except OSError:
                pass
        raise
    return resolved


def _check_output_paths(paths):
    resolved = [Path(path).resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise FrozenArtifactError("Agent 1.2 output paths must be distinct")
    existing = [path for path in resolved if path.exists()]
    if existing:
        raise FrozenArtifactError(
            "Agent 1.2 output already exists; refusing to overwrite or repeat inference"
        )
    return resolved


def _verify_baseline(frozen, frozen_hash, baseline_path):
    if type(frozen) is not dict or type(frozen.get("tasks")) is not dict:
        raise FrozenArtifactError("frozen Agent 1.1 task set is invalid")
    baseline_bytes = Path(baseline_path).read_bytes()
    baseline = _load_json(baseline_path, label="Agent 1.1 run")
    included_count = sum(
        1 for task in frozen["tasks"].values()
        if type(task) is dict and task.get("include_in_headline") is True
    )
    if (type(baseline) is not dict
            or baseline.get("schema_version") != "agent1-1-semantic-run-v1"
            or baseline.get("run_status") != "complete"
            or baseline.get("agent1_1_gt_sha256") != frozen_hash
            or baseline.get("benchmark_sha256") != frozen.get("benchmark_sha256")
            or baseline.get("model") != "deepseek-flash"
            or baseline.get("retrieval_performed") is not False
            or baseline.get("agent0_runtime_called") is not False
            or baseline.get("citation_renderer_called") is not False
            or type(baseline.get("tasks")) is not dict
            or set(baseline["tasks"]) != set(frozen["tasks"])
            or type(baseline.get("api_events")) is not list
            or len(baseline["api_events"]) != included_count):
        raise FrozenArtifactError("saved Agent 1.1 arm is not bound to the current freeze")
    for key in ("source_manifest_sha256", "source_sha256"):
        if key in frozen and baseline.get(key) != frozen.get(key):
            raise FrozenArtifactError("saved Agent 1.1 source hashes do not match the freeze")
    return baseline, baseline_bytes, hashlib.sha256(baseline_bytes).hexdigest()


def _sanitized_event(event):
    allowed = (
        "retry", "http_status", "request_id", "response_id", "provider_status",
        "usage", "validation_error", "response_body_truncated", "error", "latency_ms",
        "cost_rmb", "reserved_rmb",
    )
    return {key: event[key] for key in allowed if key in event}


def _redact_secret(value, secret):
    if type(value) is str:
        return (value.replace(secret, "[REDACTED]"), True) if secret and secret in value else (
            value, False
        )
    if type(value) is list:
        rows = [_redact_secret(item, secret) for item in value]
        return [row for row, _changed in rows], any(changed for _row, changed in rows)
    if type(value) is dict:
        rows = {key: _redact_secret(item, secret) for key, item in value.items()}
        return ({key: row for key, (row, _changed) in rows.items()},
                any(changed for _row, changed in rows.values()))
    return value, False


def _write_raw_event(stream, task_id, event, secret):
    body = event.get("raw_response_body")
    encoded = event.get("raw_response_body_base64")
    redacted = False
    if secret and type(body) is str and secret in body:
        body = body.replace(secret, "[REDACTED]")
        encoded = None
        redacted = True
    record = {
        "task_id": task_id,
        "request_id": event.get("request_id"),
        "response_id": event.get("response_id"),
        "http_status": event.get("http_status"),
        "provider_status": event.get("provider_status"),
        "provider_response_body": body,
        "provider_response_body_base64": encoded,
        "response_body_truncated": event.get("response_body_truncated", False),
        "parsed_structured_output": event.get("parsed_structured_output"),
        "validation_error": event.get("validation_error"),
        "provider_usage": event.get("provider_usage", {}),
        "usage": event.get("usage", {}),
        "latency_ms": event.get("latency_ms"),
        "cost_rmb": event.get("cost_rmb"),
        "reserved_rmb": event.get("reserved_rmb"),
        "error": event.get("error"),
        "raw_response_redacted_for_secret": redacted,
    }
    record, changed = _redact_secret(record, secret)
    if changed:
        record["provider_response_body_base64"] = None
        record["raw_response_redacted_for_secret"] = True
    stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                 .encode("utf-8") + b"\n")
    stream.flush()
    os.fsync(stream.fileno())


def _verify_ground_truth(verifier, frozen_path, source_manifest_path, source_root,
                         page_reader):
    return verifier(
        frozen_path=frozen_path,
        source_manifest_path=source_manifest_path,
        source_root=source_root,
        **({"page_reader": page_reader} if page_reader is not None else {}),
    )


def run_agent1_2_experiment(*, frozen_path=FROZEN_GT_PATH,
                            baseline_results_path=BASELINE_RESULTS_PATH,
                            results_path=RESULTS_PATH,
                            metrics_path=METRICS_PATH,
                            raw_responses_path=RAW_RESPONSES_PATH,
                            source_manifest_path=SOURCE_MANIFEST_PATH,
                            source_root=SOURCE_ROOT,
                            budget_factory=SemanticCostBudget,
                            comparator_factory=DeepSeekResponsesJsonSchemaComparator,
                            ground_truth_verifier=verify_frozen_agent1_1_ground_truth,
                            page_reader=None):
    """Run exactly one no-retry pass over the unchanged Agent 1.1 frozen cohort."""
    secret = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not secret:
        raise SemanticAPIError("deepseek:missing_key")
    _check_output_paths((results_path, metrics_path, raw_responses_path))
    frozen, gt_hash, class_counts = _verify_ground_truth(
        ground_truth_verifier, frozen_path, source_manifest_path, source_root, page_reader
    )
    baseline, baseline_bytes, baseline_hash = _verify_baseline(
        frozen, gt_hash, baseline_results_path
    )
    if type(frozen) is not dict or type(frozen.get("tasks")) is not dict:
        raise FrozenArtifactError("frozen Agent 1.1 task set is invalid")
    if type(class_counts) is not dict:
        raise FrozenArtifactError("frozen Agent 1.1 class counts are invalid")

    budget = budget_factory(soft_limit_rmb=3.0, hard_limit_rmb=5.0)
    comparator = comparator_factory(
        max_tokens=MAX_OUTPUT_TOKENS,
        budget=budget,
        message_builder=semantic_messages_v2,
        output_parser=parse_semantic_output_v2,
    )
    results_path, metrics_path, raw_responses_path = _reserve_output_paths(
        (results_path, metrics_path, raw_responses_path)
    )
    output = {
        "schema_version": RESULT_SCHEMA,
        "run_status": "in_progress",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "authority": frozen.get("authority"),
        "review_status": frozen.get("review_status"),
        "included_class_counts": class_counts,
        "agent1_1_gt_sha256": gt_hash,
        "benchmark_sha256": frozen["benchmark_sha256"],
        "source_manifest_sha256": frozen.get("source_manifest_sha256"),
        "source_sha256": frozen.get("source_sha256"),
        "agent1_1_run_sha256": baseline_hash,
        "model": "deepseek-flash",
        "documented_served_model": "DeepSeek V4.1 Flash",
        "thinking": "disabled",
        "temperature": 0,
        "provider_api": "Responses API",
        "response_format": "text.format=json_schema",
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "json_schema_name": "agent1_2_semantic_result",
        "cost_limits_rmb": {"soft": 3.0, "hard": 5.0},
        "pricing_snapshot": {
            "source": OFFICIAL_PRICING_URL,
            "verified_on": "2026-10-03",
            "peak_input_cache_hit_rmb_per_million": (
                SemanticCostBudget.INPUT_HIT_RMB_PER_MILLION
            ),
            "peak_input_cache_miss_rmb_per_million": (
                SemanticCostBudget.INPUT_MISS_RMB_PER_MILLION
            ),
            "peak_output_rmb_per_million": SemanticCostBudget.OUTPUT_RMB_PER_MILLION,
        },
        "retrieval_performed": False,
        "agent0_runtime_called": False,
        "citation_renderer_called": False,
        "tasks": {
            task_id: {"status": "not_run_after_stop", "semantic": None}
            for task_id in frozen["tasks"]
        },
        "api_events": [],
        "stop_reason": None,
    }
    _atomic_write_json(results_path, output)

    with raw_responses_path.open("ab") as raw_stream:
        for task_id, task in frozen["tasks"].items():
            if not task["include_in_headline"]:
                output["tasks"][task_id] = {
                    "status": "excluded",
                    "error_code": task["exclusion_reason"],
                    "semantic": None,
                }
                continue
            previous_event_count = len(comparator.events)
            caught = None
            semantic = None
            try:
                semantic = comparator.compare(
                    task["comparison_request"],
                    [row["text"] for row in task["evidence"]["A"]],
                    [row["text"] for row in task["evidence"]["B"]],
                )
            except InvalidSemanticOutputV2 as exc:
                caught = exc
            except SemanticAPIError as exc:
                caught = exc
            finally:
                new_events = comparator.events[previous_event_count:]
                if len(new_events) > 1:
                    raise FrozenArtifactError("Responses comparator made more than one request")
                if new_events:
                    event = new_events[0]
                    if event.get("retry") is True:
                        raise FrozenArtifactError("Responses comparator retried a request")
                    _write_raw_event(raw_stream, task_id, event, secret)

            if isinstance(caught, InvalidSemanticOutputV2):
                output["tasks"][task_id] = {
                    "status": "invalid_output",
                    "error_code": caught.reason_code,
                    "failure_type": caught.failure_type,
                    "semantic": None,
                }
            elif isinstance(caught, SemanticAPIError):
                code = str(caught)
                output["tasks"][task_id] = {
                    "status": "api_error",
                    "error_code": (code if code.startswith("deepseek:")
                                   else "deepseek:request_error"),
                    "semantic": None,
                }
                if "cost_limit" in code or "hard_budget_limit" in code:
                    output["stop_reason"] = code
            else:
                output["tasks"][task_id] = {
                    "status": ("semantic_uncertain"
                               if semantic["comparability_basis"] == "uncertain"
                               else "compared"),
                    "semantic": semantic,
                }

            output["api_events"] = [
                _sanitized_event(event) for event in comparator.events
            ]
            output, _redacted = _redact_secret(output, secret)
            _atomic_write_json(results_path, output)
            if output["stop_reason"] is not None:
                break

    output["api_events"] = [_sanitized_event(event) for event in comparator.events]
    output["run_status"] = "complete" if output["stop_reason"] is None else "stopped"
    output["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    comparison = score_agent1_2(frozen["tasks"], baseline, output)
    output["metrics"] = comparison
    output, _redacted = _redact_secret(output, secret)
    _atomic_write_json(results_path, output)
    metrics_record = {
        "schema_version": METRICS_SCHEMA,
        "agent1_1_gt_sha256": gt_hash,
        "benchmark_sha256": frozen["benchmark_sha256"],
        "agent1_1_run_sha256": baseline_hash,
        "comparison": comparison,
    }
    _atomic_write_json(metrics_path, metrics_record)

    frozen_after, gt_hash_after, _ = _verify_ground_truth(
        ground_truth_verifier, frozen_path, source_manifest_path, source_root, page_reader
    )
    if (gt_hash_after != gt_hash
            or frozen_after.get("benchmark_sha256") != frozen.get("benchmark_sha256")
            or Path(baseline_results_path).read_bytes() != baseline_bytes):
        raise FrozenArtifactError("Agent 1.1 frozen input changed during Agent 1.2")
    return output


def rescore_agent1_2(*, frozen_path=FROZEN_GT_PATH,
                     baseline_results_path=BASELINE_RESULTS_PATH,
                     results_path=RESULTS_PATH,
                     metrics_path=METRICS_PATH,
                     source_manifest_path=SOURCE_MANIFEST_PATH,
                     source_root=SOURCE_ROOT,
                     ground_truth_verifier=verify_frozen_agent1_1_ground_truth,
                     page_reader=None):
    frozen, gt_hash, _ = _verify_ground_truth(
        ground_truth_verifier, frozen_path, source_manifest_path, source_root, page_reader
    )
    baseline, baseline_bytes, baseline_hash = _verify_baseline(
        frozen, gt_hash, baseline_results_path
    )
    results_path = Path(results_path)
    results = _load_json(results_path, label="Agent 1.2 run")
    if (type(results) is not dict or results.get("schema_version") != RESULT_SCHEMA
            or results.get("agent1_1_gt_sha256") != gt_hash
            or results.get("benchmark_sha256") != frozen.get("benchmark_sha256")
            or results.get("agent1_1_run_sha256") != baseline_hash
            or results.get("retrieval_performed") is not False
            or results.get("agent0_runtime_called") is not False):
        raise FrozenArtifactError("saved Agent 1.2 run is not bound to the current freeze")
    result_data_before_rescore = {
        key: value for key, value in results.items() if key != "metrics"
    }
    comparison = score_agent1_2(frozen["tasks"], baseline, results)
    results["metrics"] = comparison
    _atomic_write_json(results_path, results)
    record = {
        "schema_version": METRICS_SCHEMA,
        "agent1_1_gt_sha256": gt_hash,
        "benchmark_sha256": frozen["benchmark_sha256"],
        "agent1_1_run_sha256": baseline_hash,
        "comparison": comparison,
    }
    _atomic_write_json(metrics_path, record)
    frozen_after, gt_hash_after, _ = _verify_ground_truth(
        ground_truth_verifier, frozen_path, source_manifest_path, source_root, page_reader
    )
    if (gt_hash_after != gt_hash
            or frozen_after.get("benchmark_sha256") != frozen.get("benchmark_sha256")
            or Path(baseline_results_path).read_bytes() != baseline_bytes
            or {key: value for key, value in _load_json(
                results_path, label="Agent 1.2 run after rescore"
            ).items() if key != "metrics"} != result_data_before_rescore):
        raise FrozenArtifactError("Agent 1.2 rescore changed inference data or a frozen input")
    return record


def _make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "rescore"))
    return parser


def main(argv=None):
    args = _make_parser().parse_args(argv)
    try:
        if args.command == "run":
            result = run_agent1_2_experiment()
            print(json.dumps({"run_status": result["run_status"],
                              "metrics": result["metrics"]}, ensure_ascii=False,
                             indent=2))
        else:
            result = rescore_agent1_2()
            print(json.dumps(result["comparison"], ensure_ascii=False, indent=2))
        return 0
    except (FrozenArtifactError, SemanticAPIError) as exc:
        print("Agent 1.2 stopped: " + str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
