"""Compare Agent 0.2 copied quotes with one frozen ID/reference DeepSeek pass.

SEARCH observations and LOOKUP cache entries are reused unchanged. Outputs,
actions, claims, evaluation labels and review sheets stay in ignored outputs/.
"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
from uuid import uuid4

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluate_agent01 import SearchReplaySession, inventory
from evaluation.evaluate_agent02 import usage_summary
from evaluation.agent03_metrics import score_reference
from src.agent.deepseek_selector import ApiCostBudget, DeepSeekActionSelector
from src.agent.harness import AgentHarness
from src.agent.io import APP_ROOT, load_corpus
from src.agent.tools import KnowledgeBaseSession

PRIVATE_ROOT = APP_ROOT / "outputs" / "agent0_3"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class FrozenInputs:
    def __init__(self, manifest, observations, reviews, pricing, baseline_raw, baseline_reviewed):
        self.paths = [Path(path).resolve() for path in
                      (manifest, observations, reviews, pricing, baseline_raw, baseline_reviewed)]
        self.hashes = {str(path): digest(path.read_bytes()) for path in self.paths}
        self.manifest = json.loads(self.paths[0].read_bytes())

    def verify(self):
        expected = self.manifest["protected_hashes"]
        current = {str(path.resolve()) for root in self.manifest["protected_roots"]
                   for path in Path(root).rglob("*") if path.is_file()}
        if (current != set(expected) or inventory(current) != expected
                or any(digest(path.read_bytes()) != self.hashes[str(path)] for path in self.paths)):
            raise RuntimeError("frozen inputs/observations/reviews/pricing/baseline changed")


class RecordingSelector:
    """Capture parsed actions only; never capture hidden reasoning or responses."""
    action_contract = "evidence_reference"

    def __init__(self, selector):
        self.selector = selector
        self.actions = []
        self.failures = []

    def __call__(self, state):
        try:
            action = self.selector(state)
        except Exception as exc:
            self.failures.append(str(exc) if str(exc).startswith("deepseek:")
                                 else type(exc).__name__)
            raise
        self.actions.append(deepcopy(action))
        return action


def write_human_review_sheet(path, manifest, candidate_reviews):
    """Create an ignored checklist; it is never read by the model/evaluator."""
    fields = ("task_id", "scope", "target_field", "target_unit", "target_condition",
              "expected_evidence_ids", "review_status")
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for task in manifest["tasks"]:
            reviews = candidate_reviews.get(task["id"], {})
            for scope in task.get("scopes", []):
                oracle = task.get("oracle", {}).get(scope, {})
                ids = sorted(eid for eid, row in reviews.items()
                             if row.get("scope") == scope and row.get("label") == "RELEVANT")
                writer.writerow({"task_id": task["id"], "scope": scope,
                                 "target_field": json.dumps(oracle.get("field_groups", []), ensure_ascii=False),
                                 "target_unit": json.dumps(oracle.get("unit_groups", []), ensure_ascii=False),
                                 "target_condition": json.dumps(oracle.get("condition_groups", []), ensure_ascii=False),
                                 "expected_evidence_ids": ";".join(ids),
                                 "review_status": "pending_human_pdf_review"})


def _validate_baseline_binding(baseline_raw, baseline_reviewed,
                               expected_frozen_hashes, expected_task_ids):
    """Require the Agent 0.2 comparison arm to use the same frozen inputs/tasks."""
    if (not isinstance(baseline_raw, dict) or not isinstance(baseline_reviewed, dict)
            or baseline_raw.get("frozen_hashes") != expected_frozen_hashes
            or baseline_reviewed.get("frozen_hashes") != expected_frozen_hashes):
        raise ValueError("Agent 0.2 baseline frozen inputs do not match this run")
    arms = baseline_raw.get("arms")
    baseline_tasks = arms.get("B") if isinstance(arms, dict) else None
    if not isinstance(baseline_tasks, dict) or set(baseline_tasks) != set(expected_task_ids):
        raise ValueError("Agent 0.2 baseline task set does not match this run")


def _check_artifacts(guard, manifest_path, observations_path, review_path,
                     pricing_path, baseline_raw_path, baseline_reviewed_path):
    guard.verify()
    manifest = guard.manifest
    observations = json.loads(Path(observations_path).read_bytes())
    reviews = json.loads(Path(review_path).read_bytes())
    pricing = json.loads(Path(pricing_path).read_bytes())
    baseline_raw_bytes = Path(baseline_raw_path).read_bytes()
    baseline_raw = json.loads(baseline_raw_bytes)
    baseline_reviewed = json.loads(Path(baseline_reviewed_path).read_bytes())
    manifest_sha = guard.hashes[str(Path(manifest_path).resolve())]
    observations_sha = guard.hashes[str(Path(observations_path).resolve())]
    if manifest.get("version") != "agent01-industrial-v1" or len(manifest.get("tasks", [])) != 8:
        raise ValueError("the frozen eight-task Agent 0.1 set is required")
    if (observations.get("manifest_sha256") != manifest_sha
            or reviews.get("manifest_sha256") != manifest_sha
            or reviews.get("observations_sha256") != observations_sha):
        raise ValueError("candidate review or observations mismatch frozen tasks")
    if (baseline_raw.get("contract") != "model_copied_quote"
            or baseline_reviewed.get("raw_results_sha256") != digest(baseline_raw_bytes)
            or baseline_reviewed.get("reviewed_summaries", {}).get("B") is None):
        raise ValueError("Agent 0.2 reviewed DeepSeek copied-quote baseline is required")
    baseline_input_hashes = {
        str(Path(path).resolve()): guard.hashes[str(Path(path).resolve())]
        for path in (manifest_path, observations_path, review_path, pricing_path)}
    _validate_baseline_binding(baseline_raw, baseline_reviewed,
                               baseline_input_hashes,
                               {task["id"] for task in manifest["tasks"]})
    if set(reviews.get("tasks", {})) != {task["id"] for task in manifest["tasks"]}:
        raise ValueError("candidate review must cover all frozen task IDs")
    if (pricing.get("model") != "deepseek-flash" or pricing.get("currency") != "CNY"
            or pricing.get("source") != "https://api-docs.deepseek.com/zh-cn/quick_start/pricing"):
        raise ValueError("verified official DeepSeek Flash CNY pricing is required")
    if not all(task["id"] in observations.get("arms", {}).get("A", {}).get("states", {})
               for task in manifest["tasks"]):
        raise ValueError("complete frozen SEARCH observations are required")
    return observations, reviews, pricing, baseline_reviewed


def run_comparison(manifest_path, observations_path, review_path, pricing_path,
                   baseline_raw_path, baseline_reviewed_path):
    if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        raise RuntimeError("deepseek:missing_key")
    guard = FrozenInputs(manifest_path, observations_path, review_path, pricing_path,
                         baseline_raw_path, baseline_reviewed_path)
    observations, review, pricing, baseline_reviewed = _check_artifacts(
        guard, manifest_path, observations_path, review_path, pricing_path,
        baseline_raw_path, baseline_reviewed_path)
    manifest = guard.manifest
    budget = ApiCostBudget(input_rmb_per_million=pricing["peak_input_miss_rmb"],
                           output_rmb_per_million=pricing["peak_output_rmb"], stop_rmb=3.0)
    api = DeepSeekActionSelector(action_contract="evidence_reference", budget=budget)
    output = dict(baseline="1ae36f3", started_utc=datetime.now(timezone.utc).isoformat(),
                  frozen_hashes=guard.hashes, pricing=pricing,
                  baseline_copied_quote=baseline_reviewed["reviewed_summaries"]["B"],
                  arms={"B": {}}, api_events=[], protected_inputs_unchanged=False,
                  forbidden_operations={}, contract="evidence_reference",
                  measurement="frozen_search_real_cache_lookup", max_tokens=512,
                  review_authority=review.get("authority", "assistant-reviewed provisional GT"))
    code_paths = list((APP_ROOT / "src" / "agent").glob("*.py")) + [
        Path(__file__), APP_ROOT / "evaluation" / "agent03_metrics.py"]
    output["inference_code_sha256"] = {
        str(path.relative_to(APP_ROOT)): digest(path.read_bytes()) for path in code_paths}
    run_dir = PRIVATE_ROOT / ("run_" + uuid4().hex)
    if not run_dir.resolve().is_relative_to((APP_ROOT / "outputs").resolve()):
        raise ValueError("output escapes ignored outputs directory")
    run_dir.mkdir(parents=True)
    target = run_dir / "results.json"
    sheet = run_dir / "human-review-sheet.csv"
    write_human_review_sheet(sheet, manifest, review["tasks"])
    output["human_review_sheet_sha256"] = digest(sheet.read_bytes())

    def save():
        output["api_events"] = deepcopy(api.events)
        output["api_usage"] = usage_summary(api.events, pricing)
        output["api_usage"]["conservative_upper_rmb"] = budget.reserved_rmb
        target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    def forbidden(name):
        def reject(*args, **kwargs):
            output["forbidden_operations"][name] = output["forbidden_operations"].get(name, 0) + 1
            raise RuntimeError("forbidden retrieval operation")
        return reject

    sessions = {}
    try:
        with ExitStack() as stack:
            for name in ("src.mineru_loader._run_mineru", "src.agent.tools.embed_chunks",
                         "src.agent.tools.load_model", "src.agent.tools.load_reranker",
                         "src.agent.tools.retrieve", "src.agent.tools.retrieve_hybrid",
                         "src.agent.tools.rerank_candidates", "src.agent.tools.BM25Index"):
                stack.enter_context(patch(name, side_effect=forbidden(name)))
            for task in manifest["tasks"]:
                specs = tuple(task["documents"])
                if specs not in sessions:
                    corpus = load_corpus(specs, parser="mineru", cache_root=manifest["cache_root"])
                    sessions[specs] = KnowledgeBaseSession(corpus)
                observations_for_task = observations["arms"]["A"]["states"][task["id"]]["search_history"]
                if (any(row["query"] != task["query"] or row["status"] not in ("ok", "no_evidence")
                        or len(row["scopes"]) != 1 for row in observations_for_task)
                        or {row["scopes"][0] for row in observations_for_task} != set(task.get("scopes", []))):
                    raise ValueError("incomplete or mismatched successful SEARCH observations")
                session = SearchReplaySession(sessions[specs], observations_for_task)
                candidates = {row["evidence_id"] for search in observations_for_task
                              for row in search["results"]}
                task_reviews = review["tasks"][task["id"]]
                if set(task_reviews) != candidates:
                    raise ValueError("candidate review must cover every fixed SEARCH candidate")
                for evidence_id, annotation in task_reviews.items():
                    looked = sessions[specs].lookup_evidence(evidence_id).results[0]
                    review_sha = digest(json.dumps(looked, sort_keys=True, ensure_ascii=False).encode("utf-8"))
                    if annotation.get("lookup_sha256") != review_sha:
                        raise ValueError("candidate review does not match frozen LOOKUP evidence")
                recorder = RecordingSelector(api)
                state = AgentHarness(session, recorder).run(task["query"], task.get("scopes", []))
                finish = next((action for action in reversed(recorder.actions)
                               if action.get("action") == "FINISH"), {"outcomes": []})
                metrics = score_reference(asdict(state), finish.get("outcomes", []), task,
                                          task_reviews, sessions[specs])
                output["arms"]["B"][task["id"]] = dict(
                    state=asdict(state), actions=recorder.actions,
                    selector_failures=recorder.failures, metrics=metrics)
                guard.verify()
                save()
                print(json.dumps({"task": task["id"], "status": state.status,
                                  "candidate_group": metrics["candidate_group"],
                                  "id_selection_success": metrics["evidence_id_selection_success"]},
                                 ensure_ascii=True), flush=True)
                if any(code in recorder.failures for code in
                       ("deepseek:http_401", "deepseek:http_403", "deepseek:cost_limit",
                        "deepseek:missing_key")):
                    output["stop_reason"] = recorder.failures[-1]
                    break
    finally:
        try:
            guard.verify()
            output["protected_inputs_unchanged"] = True
        finally:
            save()
            print("Private results: " + str(target), flush=True)
    if output["forbidden_operations"]:
        raise RuntimeError("frozen retrieval boundary violated")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "observations", "review", "pricing",
                 "baseline-raw", "baseline-reviewed"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    run_comparison(args.manifest, args.observations, args.review, args.pricing,
                   args.baseline_raw, args.baseline_reviewed)


if __name__ == "__main__":
    main()
