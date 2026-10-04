"""Apply a local PDF-audit patch and rescore saved Agent 0.2/0.3 actions.

This script consumes JSON/CSV snapshots only. It does not import an inference
runner, load a PDF/cache, or execute model, API, embedding, or retrieval calls.
Derived artifacts and aggregate results must stay beneath ignored outputs/.
"""
import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.agent01_metrics import review_finding
from evaluation.agent02_metrics import score_selection
from evaluation.agent03_metrics import score_reference


APP_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = APP_ROOT / "outputs" / "agent0_3"
DEFAULT_RUN_ID = "run_23bb9cf5f607494e94ef22b25d4fdc17"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def json_bytes(payload):
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _expected_ids(value):
    if isinstance(value, list):
        result = list(value)
    elif value is None or value == "":
        result = []
    elif isinstance(value, str):
        result = [part.strip() for part in value.split(";") if part.strip()]
    else:
        raise ValueError("expected evidence IDs must be a list or semicolon-separated string")
    if len(result) != len(set(result)) or any(not isinstance(item, str) or not item for item in result):
        raise ValueError("expected evidence IDs must be nonempty unique strings")
    return result


def apply_repair_manifest(manifest, review_sheet_rows, candidate_reviews, repair_manifest):
    """Build deep-copied derived GT/review inputs and enforce every before value."""
    repaired = deepcopy(manifest)
    reviews = deepcopy(candidate_reviews)
    tasks = {task["id"]: task for task in repaired.get("tasks", [])}
    if len(tasks) != len(repaired.get("tasks", [])):
        raise ValueError("frozen manifest task IDs must be unique")

    rows = {}
    for row in review_sheet_rows:
        key = (row.get("task_id"), row.get("scope"))
        if key in rows:
            raise ValueError("baseline review sheet contains duplicate task/scope rows")
        rows[key] = row

    required = {(task_id, scope) for task_id, task in tasks.items()
                for scope, oracle in task.get("oracle", {}).items()
                if oracle.get("expected_available") is not None}
    if set(rows) != required:
        raise ValueError("baseline review sheet must cover every assessed task/scope exactly once")

    for (task_id, scope), row in rows.items():
        oracle = tasks[task_id]["oracle"][scope]
        oracle["expected_evidence_ids"] = _expected_ids(row.get("expected_evidence_ids"))
        oracle["source_review_status"] = row.get("original_review_status")
        oracle["review_status"] = ("ai_pdf_reviewed_provisional"
                                   if row.get("ai_review_confidence") == "MEDIUM"
                                   else "ai_pdf_reviewed")
        oracle["review_confidence"] = row.get("ai_review_confidence")

    for change in repair_manifest.get("expected_id_changes", []):
        oracle = tasks[change["task_id"]]["oracle"][change["scope"]]
        before = _expected_ids(change.get("before"))
        after = _expected_ids(change.get("after"))
        if oracle["expected_evidence_ids"] != before:
            raise ValueError("expected ID patch before-value does not match source review sheet")
        oracle["expected_evidence_ids"] = after

    for change in repair_manifest.get("oracle_changes", []):
        oracle = tasks[change["task_id"]]["oracle"][change["scope"]]
        field = change["field"]
        if oracle.get(field) != change.get("before"):
            raise ValueError("oracle patch before-value does not match frozen manifest")
        oracle[field] = deepcopy(change.get("after"))
        changed = set(oracle.get("_repair_changed_fields", []))
        changed.add(field)
        oracle["_repair_changed_fields"] = sorted(changed)

    for change in repair_manifest.get("candidate_review_changes", []):
        row = reviews.get("tasks", {}).get(change["task_id"], {}).get(change["evidence_id"])
        if row is None or row.get("label") != change.get("before_label"):
            raise ValueError("candidate-review patch before-label does not match source review")
        for field, value in change["after"].items():
            if field == "lookup_sha256":
                raise ValueError("repair manifest may not rewrite frozen lookup hashes")
            row[field] = deepcopy(value)
        row["review_status"] = "ai_pdf_reviewed"

    for task_id, task in tasks.items():
        task_reviews = reviews.get("tasks", {}).get(task_id, {})
        for scope, oracle in task.get("oracle", {}).items():
            if oracle.get("expected_available") is True and not oracle["expected_evidence_ids"]:
                raise ValueError("available GT scope must have at least one expected evidence ID")
            if oracle.get("expected_available") is False and oracle["expected_evidence_ids"]:
                raise ValueError("unsupported GT scope may not have expected evidence IDs")
            for evidence_id in oracle.get("expected_evidence_ids", []):
                review = task_reviews.get(evidence_id)
                if review is not None and (review.get("scope") != scope or review.get("label") != "RELEVANT"):
                    raise ValueError("expected evidence ID conflicts with repaired candidate review")
        expected = {(scope, evidence_id)
                    for scope, oracle in task.get("oracle", {}).items()
                    for evidence_id in oracle.get("expected_evidence_ids", [])}
        relevant = {(row.get("scope"), evidence_id) for evidence_id, row in task_reviews.items()
                    if row.get("label") == "RELEVANT"}
        if relevant - expected:
            raise ValueError("repaired candidate review contains a RELEVANT ID absent from GT")

    authority = repair_manifest.get("review_authority", "ai_assisted_original_pdf_review")
    repaired["gt_review"] = {
        "review_authority": authority,
        "review_status": "ai_pdf_reviewed",
        "source_manifest_sha256": repair_manifest.get("source_artifacts", {}).get(
            "frozen_manifest", {}).get("sha256"),
        "patch_summary": repair_manifest.get("summary", {}),
    }
    reviews["source_manifest_sha256"] = reviews.get("manifest_sha256")
    reviews["authority"] = authority
    reviews["review_status"] = "ai_pdf_reviewed"
    reviews["source_observations_sha256"] = reviews.get("observations_sha256")
    reviews["manifest_sha256"] = digest(json_bytes(repaired))
    return repaired, reviews


def _last_finish(actions):
    return next((action for action in reversed(actions or [])
                 if action.get("action") == "FINISH"), None)


def _alignment_counts(rows, field):
    values = [row.get(field) for row in rows if row.get(field) is not None]
    return {"passed": sum(value is True for value in values), "assessed": len(values)}


def _reviewed_findings(record, task):
    existing = record.get("finding_reviews")
    if existing is None:
        return None
    reviews = deepcopy(existing)
    finish = _last_finish(record.get("actions", []))
    findings = finish.get("findings", []) if finish else []
    if len(reviews) != len(findings):
        raise ValueError("saved Agent 0.2 finding reviews do not cover saved FINISH findings")
    for finding, review in zip(findings, reviews):
        oracle = task.get("oracle", {}).get(finding["scope"], {})
        if "condition_groups" not in oracle.get("_repair_changed_fields", []):
            continue
        evidence = record["state"].get("looked_up_evidence", {}).get(finding["evidence_id"])
        alignment = review_finding(finding, evidence, oracle)["condition_alignment"]
        if alignment is not None:
            review["condition_alignment"] = alignment
            if alignment is False and review.get("label") == "RELEVANT":
                review["label"] = "PARTIAL"
                review["reason"] = "Condition alignment was downgraded against repaired GT."
    return reviews


def _agent02_unsupported(task, state, finish):
    scopes = [scope for scope, oracle in task.get("oracle", {}).items()
              if oracle.get("expected_available") is False]
    if not scopes:
        return {"passed": 0, "assessed": 0}
    accepted = state.get("status") in ("finished", "incomplete")
    findings = finish.get("findings", []) if finish else []
    return {"passed": sum(accepted and not any(row.get("scope") == scope for row in findings)
                           for scope in scopes),
            "assessed": len(scopes)}


def _agent02_task_metrics(task, record, reviews):
    finish = _last_finish(record.get("actions", []))
    oracle = task.get("oracle") if task.get("scopes") else None
    finding_reviews = _reviewed_findings(record, task)
    selection = score_selection(record["state"], record.get("actions", []), oracle,
                                reviews, finding_reviews=finding_reviews)
    quote_rows = [row["quote_review"] for row in selection["findings"]]
    return {
        "candidate_group": selection["group"],
        "id_selection_success": selection["id_selection_success"],
        "fully_relevant_task_success": selection["fully_relevant_task_success"],
        "correct_evidence_ids": selection["correct_evidence_ids"],
        "attempted_evidence_ids": selection["attempted_evidence_ids"],
        "field_alignment": _alignment_counts(quote_rows, "field_alignment"),
        "unit_alignment": _alignment_counts(quote_rows, "unit_alignment"),
        "condition_alignment": _alignment_counts(quote_rows, "condition_alignment"),
        "scope_alignment": _alignment_counts(quote_rows, "scope_alignment"),
        "unsupported_side_correct": _agent02_unsupported(task, record["state"], finish),
        "finish_attempted": finish is not None,
    }


def _agent03_task_metrics(task, record, reviews):
    finish = _last_finish(record.get("actions", []))
    outcomes = finish.get("outcomes", []) if finish else []
    metrics = score_reference(record["state"], outcomes, task, reviews, session=None)
    return {key: metrics[key] for key in (
        "candidate_group", "expected_id_mode", "evidence_id_selection_success",
        "host_finish_accepted", "fully_relevant_task_success", "correct_evidence_ids",
        "attempted_evidence_ids", "unsupported_side_correct", "field_alignment",
        "unit_alignment", "condition_alignment", "scope_alignment", "scope_statuses",
    )} | {"citation_integrity_recomputed": False}


def _summarize(tasks, metrics_by_task, *, id_success_key):
    summaries = {}
    for group in ("CANDIDATE_AVAILABLE", "RETRIEVAL_BOUND"):
        members = [metrics for metrics in metrics_by_task.values()
                   if metrics["candidate_group"] == group]
        eligible = [metrics for metrics in members if metrics[id_success_key] is not None]
        attempted = sum(metrics["attempted_evidence_ids"] for metrics in members)
        correct = sum(metrics["correct_evidence_ids"] for metrics in members)
        summary = {
            "tasks": len(members),
            "id_selection_success": {"passed": sum(m[id_success_key] is True for m in eligible),
                                     "assessed": len(eligible)},
            "fully_relevant_task_success": {
                "passed": sum(m["fully_relevant_task_success"] is True for m in eligible),
                "assessed": sum(m["fully_relevant_task_success"] is not None for m in members),
            },
            "correct_evidence_ids": {"passed": correct, "assessed": attempted},
        }
        for field in ("field_alignment", "unit_alignment", "condition_alignment", "scope_alignment"):
            summary[field] = {
                "passed": sum(m[field]["passed"] for m in members),
                "assessed": sum(m[field]["assessed"] for m in members),
            }
        summary["unsupported_side_correct"] = {
            "passed": sum(m["unsupported_side_correct"]["passed"] for m in members),
            "assessed": sum(m["unsupported_side_correct"]["assessed"] for m in members),
        }
        summaries[group] = summary
    return summaries


def rescore_records(tasks, candidate_reviews, agent02_reviewed, agent03_results):
    """Purely rescore saved parsed actions; outputs contain no quotes or evidence text."""
    task_map = {task["id"]: task for task in tasks}
    if len(task_map) != len(tasks):
        raise ValueError("rescore task IDs must be unique")
    a02_arms, a03_arms = agent02_reviewed.get("arms", {}), agent03_results.get("arms", {})
    expected_tasks = set(task_map)
    agent02 = {"arms": {}, "summaries": {}}
    for arm, records in a02_arms.items():
        if set(records) != expected_tasks:
            raise ValueError("Agent 0.2 saved task set differs from repaired GT")
        scored = {}
        for task_id, record in records.items():
            scored[task_id] = _agent02_task_metrics(
                task_map[task_id], record,
                candidate_reviews.get("tasks", {}).get(task_id, {}))
        agent02["arms"][arm] = scored
        agent02["summaries"][arm] = _summarize(
            tasks, scored, id_success_key="id_selection_success")

    if set(a03_arms) != {"B"} or set(a03_arms["B"]) != expected_tasks:
        raise ValueError("Agent 0.3 saved task set differs from repaired GT")
    agent03_tasks = {
        task_id: _agent03_task_metrics(
            task_map[task_id], record,
            candidate_reviews.get("tasks", {}).get(task_id, {}))
        for task_id, record in a03_arms["B"].items()
    }
    agent03 = {
        "tasks": agent03_tasks,
        "summary": _summarize(tasks, agent03_tasks,
                              id_success_key="evidence_id_selection_success"),
    }
    return {"agent02": agent02, "agent03": agent03}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_review_sheet(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _hash_file(path):
    return digest(Path(path).read_bytes())


def _protected_inventory(manifest):
    expected = manifest.get("protected_hashes", {})
    roots = manifest.get("protected_roots", [])
    current = {str(path.resolve()): _hash_file(path)
               for root in roots for path in Path(root).rglob("*") if path.is_file()}
    return current == expected


def _validate_candidate_binding(tasks, observations, candidate_reviews):
    states = observations.get("arms", {}).get("A", {}).get("states", {})
    for task in tasks:
        if not task.get("scopes"):
            continue
        task_id = task["id"]
        history = states.get(task_id, {}).get("search_history", [])
        frozen = {row["evidence_id"]: row.get("source")
                  for search in history for row in search.get("results", [])}
        reviews = candidate_reviews.get("tasks", {}).get(task_id, {})
        if set(frozen) != set(reviews):
            raise ValueError("candidate reviews must cover exactly the frozen SEARCH IDs")
        if any(reviews[evidence_id].get("scope") != scope for evidence_id, scope in frozen.items()):
            raise ValueError("candidate review scope differs from frozen SEARCH evidence")


def _check_hash_bindings(inputs, repair):
    expected = repair.get("source_artifacts", {})
    for name, path in inputs.items():
        if name in ("repair_manifest", "output_root"):
            continue
        source = expected.get(name)
        if source is None or source.get("sha256") != _hash_file(path):
            raise ValueError("source artifact hash does not match the local repair manifest: " + name)
        if name in ("audit_report", "audit_sheet"):
            source_path = Path(source.get("path", "")).resolve()
            if not source_path.is_file() or source_path != Path(path).resolve():
                raise ValueError("WorkBuddy audit path differs from the local repair manifest: " + name)


def _check_run_binding(agent02_raw, agent02_reviewed, agent03_raw, paths):
    if agent02_reviewed.get("raw_results_sha256") != _hash_file(paths["agent02_raw"]):
        raise ValueError("Agent 0.2 reviewed result is not bound to the supplied raw actions")
    for source_name in ("frozen_manifest", "search_observations", "candidate_reviews"):
        path = paths[source_name].resolve()
        expected = _hash_file(path)
        for record in (agent02_raw, agent02_reviewed, agent03_raw):
            bound = record.get("frozen_hashes", {})
            if str(path) not in bound or bound[str(path)] != expected:
                raise ValueError("saved run is not bound to original frozen input: " + source_name)


def _citation_integrity_snapshot(agent03_raw):
    result = {}
    for task_id, record in agent03_raw.get("arms", {}).get("B", {}).items():
        metrics = record.get("metrics", {})
        result[task_id] = {key: metrics.get(key) for key in (
            "citation_authenticity", "citation_provenance", "span_bounds")}
    return result


def _write_json(path, value):
    target = Path(path).resolve()
    if not target.is_relative_to(PRIVATE_ROOT.resolve()):
        raise ValueError("derived evaluation artifacts must stay in ignored outputs/agent0_3")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(json_bytes(value))


def rescore_artifacts(paths):
    """Validate frozen sources, write derived local copies, and rescore snapshots."""
    repair = _read_json(paths["repair_manifest"])
    _check_hash_bindings(paths, repair)
    base_bytes = {name: Path(path).read_bytes() for name, path in paths.items()
                  if name not in ("repair_manifest", "output_root")}
    source_hashes = {name: digest(raw) for name, raw in base_bytes.items()}
    manifest = json.loads(base_bytes["frozen_manifest"])
    if not _protected_inventory(manifest):
        raise ValueError("frozen protected-file inventory does not match source manifest")
    candidate_reviews = json.loads(base_bytes["candidate_reviews"])
    observations = json.loads(base_bytes["search_observations"])
    agent02_raw = json.loads(base_bytes["agent02_raw"])
    agent02_reviewed = json.loads(base_bytes["agent02_reviewed"])
    agent03_raw = json.loads(base_bytes["agent03_raw"])
    if agent02_raw.get("protected_inputs_unchanged") is not True:
        raise ValueError("Agent 0.2 run did not attest unchanged frozen inputs")
    if agent03_raw.get("protected_inputs_unchanged") is not True:
        raise ValueError("Agent 0.3 run did not attest unchanged frozen inputs")
    if agent02_raw.get("forbidden_operations") or agent03_raw.get("forbidden_operations"):
        raise ValueError("saved baseline records forbidden operations")
    _check_run_binding(agent02_raw, agent02_reviewed, agent03_raw, paths)

    repaired, repaired_reviews = apply_repair_manifest(
        manifest, _read_review_sheet(paths["agent03_review_sheet"]), candidate_reviews, repair)
    _validate_candidate_binding(repaired["tasks"], observations, repaired_reviews)
    results = rescore_records(repaired["tasks"], repaired_reviews, agent02_reviewed, agent03_raw)
    results.update({
        "review_authority": repair.get("review_authority", "ai_assisted_original_pdf_review"),
        "review_status": "ai_pdf_reviewed",
        "review_summary": repair.get("summary", {}),
        "inference_performed": False,
        "retrieval_performed": False,
        "forbidden_operations": [],
        "source_artifact_sha256": source_hashes,
        "repair_manifest_sha256": _hash_file(paths["repair_manifest"]),
        "agent03_integrity_metrics": {
            "authority": "unchanged frozen host checks copied from the original saved run; not recalculated",
            "tasks": _citation_integrity_snapshot(agent03_raw),
        },
    })

    derived_path = Path(paths["output_root"]).resolve()
    if not derived_path.is_relative_to(PRIVATE_ROOT.resolve()):
        raise ValueError("derived artifacts must stay in ignored outputs/agent0_3")
    _write_json(derived_path / "repaired-gt-manifest.local.json", repaired)
    _write_json(derived_path / "candidate-reviews.repaired.local.json", repaired_reviews)
    _write_json(derived_path / "rescore-results.local.json", results)

    if any(_hash_file(path) != source_hashes[name]
           for name, path in paths.items() if name not in ("repair_manifest", "output_root")):
        raise RuntimeError("a frozen source artifact changed during offline rescoring")
    if not _protected_inventory(manifest):
        raise RuntimeError("a protected frozen input changed during offline rescoring")
    results["protected_inputs_unchanged"] = True
    _write_json(derived_path / "rescore-results.local.json", results)
    return repaired, repaired_reviews, results


def _default_paths(audit_root=None):
    run03 = PRIVATE_ROOT / DEFAULT_RUN_ID
    audit_root = Path(audit_root).expanduser() if audit_root is not None else None
    return {
        "frozen_manifest": APP_ROOT / "outputs" / "agent0_1" / "frozen_set.json",
        "search_observations": APP_ROOT / "outputs" / "agent0_1" / "run_5abe221949ec4668bcb96d13c14d1d5f" / "results.json",
        "candidate_reviews": APP_ROOT / "outputs" / "agent0_2" / "candidate_reviews.json",
        "agent02_raw": APP_ROOT / "outputs" / "agent0_2" / "run_5ccb38ba023b43c98d6085a5ad5eb21b" / "results.json",
        "agent02_reviewed": APP_ROOT / "outputs" / "agent0_2" / "reviewed_results.json",
        "agent03_raw": run03 / "results.json",
        "agent03_review_sheet": run03 / "human-review-sheet.csv",
        "audit_report": audit_root / "audit-report.local.md" if audit_root else None,
        "audit_sheet": audit_root / "human-review-sheet.ai-reviewed.local.csv" if audit_root else None,
        "repair_manifest": PRIVATE_ROOT / "gt-repair-manifest.local.json",
        "output_root": PRIVATE_ROOT,
    }


def main():
    defaults = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-root",
        default=os.environ.get("AGENT03_AUDIT_ROOT"),
        help="Local audit directory; may also be set with AGENT03_AUDIT_ROOT.",
    )
    parser.add_argument("--audit-report", help="Path to the local audit report.")
    parser.add_argument("--audit-sheet", help="Path to the local reviewed audit sheet.")
    for name, value in defaults.items():
        if name in ("audit_report", "audit_sheet"):
            continue
        parser.add_argument("--" + name.replace("_", "-"), default=str(value))
    args = parser.parse_args()
    if args.audit_report or args.audit_sheet:
        if not args.audit_report or not args.audit_sheet:
            parser.error("provide both --audit-report and --audit-sheet")
        audit_report, audit_sheet = Path(args.audit_report), Path(args.audit_sheet)
    elif args.audit_root:
        audit_root = Path(args.audit_root).expanduser()
        audit_report = audit_root / "audit-report.local.md"
        audit_sheet = audit_root / "human-review-sheet.ai-reviewed.local.csv"
    else:
        parser.error("provide --audit-root or AGENT03_AUDIT_ROOT for the local audit files")

    paths = {name: Path(getattr(args, name)) for name in defaults
             if name not in ("audit_report", "audit_sheet")}
    paths["audit_report"] = audit_report
    paths["audit_sheet"] = audit_sheet
    rescore_artifacts(paths)
    print("Wrote derived repaired GT, candidate reviews, and offline rescore beneath outputs/agent0_3.")


if __name__ == "__main__":
    main()
