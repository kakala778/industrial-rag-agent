"""Freeze and evaluate the private Agent 1.1 original-PDF semantic benchmark."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.agent1_1_metrics import score_agent1_1
from src.agent.semantic import (
    DeepSeekSemanticComparator,
    SemanticAPIError,
    SemanticCostBudget,
)
from src.agent.semantic_contract_v2 import (
    InvalidSemanticOutputV2,
    parse_semantic_output_v2,
    semantic_messages_v2,
    validate_semantic_output_v2,
)


APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parent
OUTPUT_ROOT = APP_ROOT / "outputs" / "agent1_1"
SOURCE_ROOT = REPO_ROOT / "minerU" / "industrial-rag-data" / "input"
PROPOSAL_PATH = OUTPUT_ROOT / "benchmark-proposal.local.json"
FROZEN_GT_PATH = OUTPUT_ROOT / "benchmark-gt.local.json"
RESULTS_PATH = OUTPUT_ROOT / "run.local.json"
METRICS_PATH = OUTPUT_ROOT / "metrics.local.json"
SOURCE_MANIFEST_PATH = OUTPUT_ROOT / "source-manifest.local.json"

EXPECTED_VERDICTS = ("EQUIVALENT", "DIFFERENT", "NOT_COMPARABLE")
EXPECTED_GT_AUTHORITY = "ai_assisted_original_pdf_review"
EXPECTED_REVIEW_STATUS = "ai_pdf_reviewed"
PROPOSAL_SCHEMA = "agent1-1-semantic-benchmark-proposal-v1"
FROZEN_SCHEMA = "agent1-1-semantic-gt-v1"
RESULT_SCHEMA = "agent1-1-semantic-run-v1"
OFFICIAL_PRICING_URL = "https://api-docs.deepseek.com/zh-cn/quick_start/pricing"
MAX_BENCHMARK_TASKS = 18


class FrozenArtifactError(ValueError):
    """A local frozen Agent 1.1 input or output failed an integrity check."""


def _canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_pdf_text(value):
    return re.sub(r"\s+", " ", value).strip()


def _read_pdf_page(path, page_number):
    try:
        import pymupdf
    except ImportError as exc:
        raise FrozenArtifactError("PyMuPDF is required to verify original-PDF evidence") from exc
    try:
        with pymupdf.open(path) as document:
            if type(page_number) is not int or not 1 <= page_number <= len(document):
                raise FrozenArtifactError("evidence page is outside the original PDF")
            return document[page_number - 1].get_text()
    except FrozenArtifactError:
        raise
    except Exception as exc:
        raise FrozenArtifactError("unable to read an original-PDF evidence page") from exc


def _load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError("unable to read a local Agent 1.1 JSON artifact") from exc


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


def _read_source_manifest(path, source_root):
    manifest = _load_json(path)
    if (type(manifest) is not dict
            or manifest.get("schema_version") != "agent1_1-source-manifest-v1"
            or type(manifest.get("files")) is not list
            or len(manifest["files"]) != 8):
        raise FrozenArtifactError("the expected eight-PDF source manifest is required")
    files = {}
    for row in manifest["files"]:
        if (type(row) is not dict or type(row.get("name")) is not str
                or type(row.get("sha256")) is not str
                or type(row.get("pages")) is not int or row["pages"] < 1):
            raise FrozenArtifactError("source manifest entry is invalid")
        name = row["name"]
        if name in files or Path(name).name != name:
            raise FrozenArtifactError("source manifest contains a duplicate or unsafe name")
        path = Path(source_root) / name
        if not path.is_file() or _sha256_file(path) != row["sha256"]:
            raise FrozenArtifactError("an original PDF no longer matches the reviewed manifest")
        files[name] = {"path": path, "sha256": row["sha256"], "pages": row["pages"]}
    return manifest, files


def _validate_proposal(proposal, source_files, page_reader=_read_pdf_page):
    if (type(proposal) is not dict
            or proposal.get("schema_version") != PROPOSAL_SCHEMA
            or proposal.get("authority") != EXPECTED_GT_AUTHORITY
            or proposal.get("review_status") != EXPECTED_REVIEW_STATUS
            or proposal.get("review_authority") != EXPECTED_GT_AUTHORITY
            or type(proposal.get("source_review")) is not dict
            or type(proposal.get("tasks")) is not dict):
        raise FrozenArtifactError("proposal provenance or schema is invalid")
    tasks = proposal["tasks"]
    if not 12 <= len(tasks) <= MAX_BENCHMARK_TASKS:
        raise FrozenArtifactError("Agent 1.1 requires 12 to 18 benchmark tasks")
    per_class = Counter()
    proposal_task_keys = {
        "case_type", "comparison_request", "expected_verdict",
        "expected_comparability_basis", "dimensions", "confidence",
        "include_in_headline", "exclusion_reason", "review_status",
        "review_authority", "review_note", "evidence",
    }
    evidence_keys = {"source_name", "page_number", "text", "excerpt_sha256"}
    page_cache = {}
    for task_id, task in tasks.items():
        if (type(task_id) is not str or not re.fullmatch(r"[A-Z]\d{2}", task_id)
                or type(task) is not dict or set(task) != proposal_task_keys):
            raise FrozenArtifactError("benchmark task schema is invalid")
        if (task.get("review_status") != EXPECTED_REVIEW_STATUS
                or task.get("review_authority") != EXPECTED_GT_AUTHORITY
                or type(task.get("review_note")) is not str
                or not task["review_note"].strip()):
            raise FrozenArtifactError("task is not bound to original-PDF review")
        verdict = task.get("expected_verdict")
        dimensions = task.get("dimensions")
        basis = task.get("expected_comparability_basis")
        included = task.get("include_in_headline")
        if verdict not in EXPECTED_VERDICTS or type(included) is not bool:
            raise FrozenArtifactError("task verdict or inclusion flag is invalid")
        if task.get("confidence") not in ("high", "medium", "uncertain"):
            raise FrozenArtifactError("task review confidence is invalid")
        if included:
            if task["confidence"] != "high" or task.get("exclusion_reason") is not None:
                raise FrozenArtifactError("headline ground truth must be high-confidence")
            per_class[verdict] += 1
        else:
            if task.get("exclusion_reason") not in ("GT_UNCERTAIN", "SCHEMA_UNSUPPORTED"):
                raise FrozenArtifactError("excluded task lacks an explicit exclusion reason")
        expected = {
            "verdict": verdict,
            "comparability_basis": basis,
            "dimensions": dimensions,
            "reason": "Frozen reviewed ground truth.",
            "notes": "",
        }
        try:
            validate_semantic_output_v2(expected)
        except InvalidSemanticOutputV2 as exc:
            raise FrozenArtifactError("ground truth violates the Agent 1.1 contract") from exc
        sides = task.get("evidence")
        if type(sides) is not dict or set(sides) != {"A", "B"}:
            raise FrozenArtifactError("task requires evidence for both sides")
        for side in ("A", "B"):
            records = sides[side]
            if type(records) is not list or not 1 <= len(records) <= 6:
                raise FrozenArtifactError("task evidence list is empty or too large")
            for record in records:
                if type(record) is not dict or set(record) != evidence_keys:
                    raise FrozenArtifactError("original-PDF evidence record is invalid")
                name = record.get("source_name")
                page = record.get("page_number")
                excerpt = record.get("text")
                if (name not in source_files or type(page) is not int
                        or page < 1 or page > source_files[name]["pages"]
                        or type(excerpt) is not str or not excerpt.strip()
                        or len(excerpt) > 1200):
                    raise FrozenArtifactError("original-PDF evidence reference is invalid")
                if record.get("excerpt_sha256") != _sha256_bytes(excerpt.encode("utf-8")):
                    raise FrozenArtifactError("evidence excerpt hash is incorrect")
                cache_key = (name, page)
                if cache_key not in page_cache:
                    page_cache[cache_key] = page_reader(source_files[name]["path"], page)
                page_text = page_cache[cache_key]
                if _normalize_pdf_text(excerpt) not in _normalize_pdf_text(page_text):
                    raise FrozenArtifactError("evidence excerpt is not present on its cited PDF page")
    if any(per_class[verdict] < 4 for verdict in EXPECTED_VERDICTS):
        raise FrozenArtifactError("each verdict class needs at least four high-confidence tasks")
    if sum(per_class.values()) > MAX_BENCHMARK_TASKS:
        raise FrozenArtifactError("benchmark inference request count exceeds the hard cap")
    return per_class


def freeze_agent1_1_ground_truth(*, proposal_path=PROPOSAL_PATH,
                                 frozen_path=FROZEN_GT_PATH,
                                 results_path=RESULTS_PATH,
                                 source_manifest_path=SOURCE_MANIFEST_PATH,
                                 source_root=SOURCE_ROOT,
                                 page_reader=_read_pdf_page):
    """Freeze a reviewed local proposal once, after validating source PDFs and spans."""
    proposal_path = Path(proposal_path)
    frozen_path = Path(frozen_path)
    results_path = Path(results_path)
    if frozen_path.exists() or results_path.exists():
        raise FrozenArtifactError("refusing to overwrite a GT freeze or post-inference result")
    manifest, sources = _read_source_manifest(source_manifest_path, source_root)
    proposal = _load_json(proposal_path)
    per_class = _validate_proposal(proposal, sources, page_reader)
    payload = {
        "schema_version": FROZEN_SCHEMA,
        "authority": EXPECTED_GT_AUTHORITY,
        "review_status": EXPECTED_REVIEW_STATUS,
        "review_authority": EXPECTED_GT_AUTHORITY,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "proposal_sha256": _sha256_file(proposal_path),
        "source_manifest_sha256": _sha256_file(source_manifest_path),
        "source_sha256": {name: row["sha256"] for name, row in sources.items()},
        "source_review": proposal["source_review"],
        "tasks": proposal["tasks"],
        "task_count": len(proposal["tasks"]),
        "included_class_counts": dict(per_class),
    }
    payload["benchmark_sha256"] = _sha256_bytes(_canonical_json(payload["tasks"]))
    payload["frozen_payload_sha256"] = _sha256_bytes(_canonical_json(payload))
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with frozen_path.open("xb") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")
    except FileExistsError as exc:
        raise FrozenArtifactError("refusing to overwrite the Agent 1.1 GT freeze") from exc
    return payload["frozen_payload_sha256"], dict(per_class)


def verify_frozen_agent1_1_ground_truth(*, frozen_path=FROZEN_GT_PATH,
                                        source_manifest_path=SOURCE_MANIFEST_PATH,
                                        source_root=SOURCE_ROOT,
                                        page_reader=_read_pdf_page):
    frozen = _load_json(frozen_path)
    if (type(frozen) is not dict or frozen.get("schema_version") != FROZEN_SCHEMA
            or frozen.get("authority") != EXPECTED_GT_AUTHORITY
            or frozen.get("review_status") != EXPECTED_REVIEW_STATUS
            or frozen.get("review_authority") != EXPECTED_GT_AUTHORITY):
        raise FrozenArtifactError("frozen Agent 1.1 GT provenance is invalid")
    frozen_digest = frozen.get("frozen_payload_sha256")
    payload = {key: value for key, value in frozen.items()
               if key != "frozen_payload_sha256"}
    if frozen_digest != _sha256_bytes(_canonical_json(payload)):
        raise FrozenArtifactError("frozen Agent 1.1 GT content hash changed")
    if frozen.get("benchmark_sha256") != _sha256_bytes(_canonical_json(frozen.get("tasks"))):
        raise FrozenArtifactError("frozen Agent 1.1 benchmark hash changed")
    manifest, sources = _read_source_manifest(source_manifest_path, source_root)
    if _sha256_file(source_manifest_path) != frozen.get("source_manifest_sha256"):
        raise FrozenArtifactError("reviewed source manifest changed after GT freeze")
    current_hashes = {name: row["sha256"] for name, row in sources.items()}
    if current_hashes != frozen.get("source_sha256"):
        raise FrozenArtifactError("original-PDF input hash changed after GT freeze")
    per_class = _validate_proposal({
        "schema_version": PROPOSAL_SCHEMA,
        "authority": frozen["authority"],
        "review_status": frozen["review_status"],
        "review_authority": frozen["review_authority"],
        "source_review": frozen.get("source_review"),
        "tasks": frozen.get("tasks"),
    }, sources, page_reader)
    return frozen, frozen_digest, dict(per_class)


def run_agent1_1_experiment(*, frozen_path=FROZEN_GT_PATH,
                            results_path=RESULTS_PATH,
                            metrics_path=METRICS_PATH,
                            source_manifest_path=SOURCE_MANIFEST_PATH,
                            source_root=SOURCE_ROOT,
                            budget_factory=SemanticCostBudget,
                            comparator_factory=DeepSeekSemanticComparator,
                            page_reader=_read_pdf_page):
    """Run exactly one no-retry pass over the frozen bilateral-evidence cohort."""
    results_path = Path(results_path)
    metrics_path = Path(metrics_path)
    if results_path.exists():
        raise FrozenArtifactError("a DeepSeek pass already exists; refusing repeat inference")
    if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        raise SemanticAPIError("deepseek:missing_key")
    frozen, gt_hash, _ = verify_frozen_agent1_1_ground_truth(
        frozen_path=frozen_path,
        source_manifest_path=source_manifest_path,
        source_root=source_root,
        page_reader=page_reader,
    )
    budget = budget_factory(soft_limit_rmb=3.0, hard_limit_rmb=5.0)
    comparator = comparator_factory(
        max_tokens=384,
        budget=budget,
        message_builder=semantic_messages_v2,
        output_parser=parse_semantic_output_v2,
    )
    output = {
        "schema_version": RESULT_SCHEMA,
        "run_status": "in_progress",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "authority": frozen["authority"],
        "review_status": frozen["review_status"],
        "agent1_1_gt_sha256": gt_hash,
        "benchmark_sha256": frozen["benchmark_sha256"],
        "source_manifest_sha256": frozen["source_manifest_sha256"],
        "source_sha256": frozen["source_sha256"],
        "model": "deepseek-flash",
        "documented_served_model": "DeepSeek V4.1 Flash",
        "thinking": "disabled",
        "temperature": 0,
        "response_format": "json_object",
        "max_tokens": comparator.max_tokens,
        "cost_limits_rmb": {"soft": 3.0, "hard": 5.0},
        "pricing_snapshot": {
            "source": OFFICIAL_PRICING_URL,
            "verified_on": "2026-10-03",
            "peak_input_cache_hit_rmb_per_million": SemanticCostBudget.INPUT_HIT_RMB_PER_MILLION,
            "peak_input_cache_miss_rmb_per_million": SemanticCostBudget.INPUT_MISS_RMB_PER_MILLION,
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
    results_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with results_path.open("xb") as stream:
            stream.write(json.dumps(output, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")
    except FileExistsError as exc:
        raise FrozenArtifactError("refusing to overwrite an Agent 1.1 run result") from exc

    for task_id, task in frozen["tasks"].items():
        if not task["include_in_headline"]:
            output["tasks"][task_id] = {
                "status": "excluded",
                "error_code": task["exclusion_reason"],
                "semantic": None,
            }
            continue
        try:
            semantic = comparator.compare(
                task["comparison_request"],
                [row["text"] for row in task["evidence"]["A"]],
                [row["text"] for row in task["evidence"]["B"]],
            )
            output["tasks"][task_id] = {
                "status": ("semantic_uncertain"
                           if semantic["comparability_basis"] == "uncertain"
                           else "compared"),
                "semantic": semantic,
            }
        except InvalidSemanticOutputV2 as exc:
            output["tasks"][task_id] = {
                "status": "invalid_output",
                "error_code": exc.reason_code,
                "semantic": None,
            }
        except SemanticAPIError as exc:
            code = str(exc)
            output["tasks"][task_id] = {
                "status": "api_error",
                "error_code": code if code.startswith("deepseek:") else "deepseek:request_error",
                "semantic": None,
            }
            if "cost_limit" in code or "hard_budget_limit" in code:
                output["stop_reason"] = code
                break
        output["api_events"] = list(comparator.events)
        _atomic_write_json(results_path, output)

    for task_id, row in output["tasks"].items():
        if row["status"] == "not_run_after_stop" and output["stop_reason"] is None:
            row["status"] = "not_run_after_stop"
        elif row["status"] == "not_run_after_stop" and output["stop_reason"]:
            row["status"] = "not_run_after_stop"
    output["api_events"] = list(comparator.events)
    output["run_status"] = "complete" if output["stop_reason"] is None else "stopped"
    output["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    output["metrics"] = score_agent1_1(frozen["tasks"], output["tasks"],
                                       api_events=output["api_events"])
    _atomic_write_json(results_path, output)
    _atomic_write_json(metrics_path, {
        "schema_version": "agent1-1-semantic-metrics-v1",
        "agent1_1_gt_sha256": gt_hash,
        "benchmark_sha256": frozen["benchmark_sha256"],
        "metrics": output["metrics"],
    })
    verify_frozen_agent1_1_ground_truth(
        frozen_path=frozen_path,
        source_manifest_path=source_manifest_path,
        source_root=source_root,
        page_reader=page_reader,
    )
    return output


def rescore_agent1_1(*, frozen_path=FROZEN_GT_PATH, results_path=RESULTS_PATH,
                     metrics_path=METRICS_PATH, source_manifest_path=SOURCE_MANIFEST_PATH,
                     source_root=SOURCE_ROOT, page_reader=_read_pdf_page):
    frozen, gt_hash, _ = verify_frozen_agent1_1_ground_truth(
        frozen_path=frozen_path,
        source_manifest_path=source_manifest_path,
        source_root=source_root,
        page_reader=page_reader,
    )
    results = _load_json(results_path)
    if (type(results) is not dict or results.get("schema_version") != RESULT_SCHEMA
            or results.get("agent1_1_gt_sha256") != gt_hash
            or results.get("benchmark_sha256") != frozen["benchmark_sha256"]
            or results.get("source_sha256") != frozen["source_sha256"]
            or results.get("retrieval_performed") is not False
            or results.get("agent0_runtime_called") is not False):
        raise FrozenArtifactError("saved run is not bound to the current Agent 1.1 freeze")
    metrics = score_agent1_1(frozen["tasks"], results.get("tasks", {}),
                             api_events=results.get("api_events", []))
    result = {
        "schema_version": "agent1-1-semantic-metrics-v1",
        "agent1_1_gt_sha256": gt_hash,
        "benchmark_sha256": frozen["benchmark_sha256"],
        "metrics": metrics,
    }
    _atomic_write_json(metrics_path, result)
    verify_frozen_agent1_1_ground_truth(
        frozen_path=frozen_path,
        source_manifest_path=source_manifest_path,
        source_root=source_root,
        page_reader=page_reader,
    )
    return result


def _make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze-gt", "run", "rescore"))
    return parser


def main(argv=None):
    args = _make_parser().parse_args(argv)
    try:
        if args.command == "freeze-gt":
            digest, classes = freeze_agent1_1_ground_truth()
            print(json.dumps({"frozen_payload_sha256": digest,
                              "included_class_counts": classes}, ensure_ascii=False))
        elif args.command == "run":
            record = run_agent1_1_experiment()
            print(json.dumps({"run_status": record["run_status"],
                              "api_requests": len(record["api_events"]),
                              "verdict_accuracy": record["metrics"]["verdict_accuracy"],
                              "api_usage": record["metrics"]["api_usage"]}, ensure_ascii=False))
        else:
            record = rescore_agent1_1()
            print(json.dumps(record["metrics"], ensure_ascii=False, indent=2))
        return 0
    except (FrozenArtifactError, SemanticAPIError) as exc:
        print(f"Agent 1.1 stopped: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
