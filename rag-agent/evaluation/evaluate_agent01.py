"""A/B/C evaluation over pre-frozen local industrial tasks or synthetic controls.

Baseline is loaded from byte-identical Git source preserved before editing.
Industrial tasks and GT must be independently frozen before either Qwen arm.
Private states, PDF content and GT never enter tracked files.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
from uuid import uuid4

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.harness import AgentHarness
from src.agent.io import APP_ROOT, load_corpus
from src.agent.policy import DeterministicPolicy
from src.agent.selector import OllamaActionSelector
from src.agent.tools import KnowledgeBaseSession, ToolResult
from evaluation.agent01_metrics import aggregate, score_state

PRIVATE_ROOT = APP_ROOT / "outputs" / "agent0_1"
BASELINE_SHA = "e474f1e9afcaf1c2cdee4e92da76e2ee1bc6dca8"


def private_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(PRIVATE_ROOT.resolve()):
        raise ValueError("evaluation artifacts must be inside ignored outputs/agent0_1")
    return path


def inventory(paths):
    return {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(paths)}


def prepare_baseline():
    """Preserve Git bytes once; existing snapshots are verified, never replaced."""
    root = private_path(PRIVATE_ROOT / "baseline" / "src")
    if root.exists():
        load_baseline()
        return root
    names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", BASELINE_SHA,
                                     "--", "rag-agent/src"], cwd=APP_ROOT.parent).decode().splitlines()
    if not names:
        raise ValueError("baseline commit/source tree unavailable")
    for name in names:
        if name.endswith(".py"):
            target = root / Path(name).relative_to("rag-agent/src")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(subprocess.check_output(["git", "show", BASELINE_SHA + ":" + name], cwd=APP_ROOT.parent))
    load_baseline()
    return root


def load_baseline():
    root = private_path(PRIVATE_ROOT / "baseline" / "src")
    files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", BASELINE_SHA, "--", "rag-agent/src"], cwd=APP_ROOT.parent).decode().splitlines()
    expected = set()
    for name in files:
        if not name.endswith(".py"):
            continue
        path = root / Path(name).relative_to("rag-agent/src")
        expected.add(path)
        if path.read_bytes() != subprocess.check_output(["git", "show", BASELINE_SHA + ":" + name], cwd=APP_ROOT):
            raise ValueError("preserved baseline source differs from Git")
    if set(root.rglob("*.py")) != expected:
        raise ValueError("unexpected baseline source file")
    if "agent0_baseline" not in sys.modules:
        spec = importlib.util.spec_from_file_location("agent0_baseline", root / "__init__.py", submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return {name: importlib.import_module("agent0_baseline.agent." + name) for name in ("harness", "selector", "tools", "policy")}


class BaselineSession:
    """Type adapter only: same frozen session/search output for both Qwen arms."""
    def __init__(self, session, result_type):
        self.session, self.result_type = session, result_type

    def resolve_scopes(self, scopes):
        return self.session.resolve_scopes(scopes)

    def search_knowledge(self, **kwargs):
        r = self.session.search_knowledge(**kwargs)
        return self.result_type(r.status, r.results, r.message)

    def lookup_evidence(self, **kwargs):
        r = self.session.lookup_evidence(**kwargs)
        return self.result_type(r.status, r.results, r.message)


class SearchReplaySession:
    """Replay exact frozen observations; lookup still uses the original registry."""
    def __init__(self, session, history):
        self.session = session
        self.history = deepcopy(history)
        if hasattr(session, "registry"):
            for row in self.history:
                if any(session.registry.get(r["evidence_id"]) != r for r in row["results"]):
                    raise ValueError("search observation differs from original registry")

    def resolve_scopes(self, scopes):
        return self.session.resolve_scopes(scopes)

    def lookup_evidence(self, **kwargs):
        return self.session.lookup_evidence(**kwargs)

    def search_knowledge(self, query, scopes=None, **kwargs):
        for row in self.history:
            if row["query"] == query and row["scopes"] == list(scopes or []):
                if row.get("status") not in ("ok", "no_evidence"):
                    raise ValueError("unsuccessful frozen search observation")
                return ToolResult(row["status"], deepcopy(row["results"]))
        raise ValueError("query/scope absent from frozen search observations")


def run_industrial(manifest_path, arms=("A", "B", "C"), search_observations=None):
    path = private_path(manifest_path)
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("version") != "agent01-industrial-v1" or len(manifest["tasks"]) not in range(8, 13):
        raise ValueError("8-12 independently authored frozen tasks required")
    expected = manifest["protected_hashes"]
    roots = [Path(r) for r in manifest["protected_roots"]]
    def verify():
        current = {str(p.resolve()) for root in roots for p in root.rglob("*") if p.is_file()}
        if current != set(expected) or inventory(current) != expected or path.read_bytes() != raw:
            raise RuntimeError("frozen task/GT/protected input inventory changed")
        if search_observations and replay_hash is not None:
            if hashlib.sha256(private_path(search_observations).read_bytes()).hexdigest() != replay_hash:
                raise RuntimeError("frozen search observations changed")
    replay_hash = None
    verify()
    replay = None
    replay_hash = None
    if search_observations:
        replay_raw = private_path(search_observations).read_bytes()
        replay = json.loads(replay_raw)
        replay_hash = hashlib.sha256(replay_raw).hexdigest()
        if replay["manifest_sha256"] != hashlib.sha256(raw).hexdigest():
            raise ValueError("search observations belong to a different manifest")
        if set(replay["arms"]["A"]["states"]) != {t["id"] for t in manifest["tasks"]}:
            raise ValueError("complete deterministic search observations required")
        for task in manifest["tasks"]:
            history = replay["arms"]["A"]["states"][task["id"]]["search_history"]
            if (any(h["query"] != task["query"] or h.get("status") not in ("ok", "no_evidence")
                    or len(h["scopes"]) != 1 for h in history)
                    or {h["scopes"][0] for h in history} != set(task["scopes"])):
                raise ValueError("incomplete or mismatched frozen task searches")
    baseline = load_baseline()
    run_dir = private_path(PRIVATE_ROOT / ("run_" + uuid4().hex))
    run_dir.mkdir(parents=True)
    output = dict(manifest_sha256=hashlib.sha256(raw).hexdigest(), baseline_sha=BASELINE_SHA,
                  review_authority=manifest["review_authority"], arms={}, mineru_calls=0,
                  protected_inputs_unchanged=False)
    output["measurement_mode"] = "frozen_search_replay" if replay else "live_tools"
    output["search_observations_sha256"] = replay_hash
    def save():
        (run_dir / "results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sessions = {}
    resources = [None, None]
    calls = []
    def forbid(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("MinerU forbidden")
    try:
        with patch("src.mineru_loader._run_mineru", side_effect=forbid):
            for arm in arms:
                records, states = {}, {}
                output["arms"][arm] = dict(records=records, states=states)
                for task in manifest["tasks"]:
                    specs = tuple(task["documents"])
                    if specs not in sessions:
                        corpus = load_corpus(specs, parser="mineru", cache_root=manifest["cache_root"])
                        sessions[specs] = KnowledgeBaseSession(corpus, model=resources[0], reranker_model=resources[1])
                    session = sessions[specs]
                    runtime_session = SearchReplaySession(session, replay["arms"]["A"]["states"][task["id"]]["search_history"]) if replay else session
                    if arm == "B":
                        selector = baseline["selector"].OllamaActionSelector()
                        harness = baseline["harness"].AgentHarness(BaselineSession(runtime_session, baseline["tools"].ToolResult), selector)
                    else:
                        selector = DeterministicPolicy() if arm == "A" else OllamaActionSelector()
                        harness = AgentHarness(runtime_session, selector)
                    started = time.monotonic()
                    # The oracle/review fields are not supplied to runtime or selector.
                    state = harness.run(task["query"], task["scopes"])
                    resources[:] = [session.model, session.reranker_model]
                    record = score_state(state, task["oracle"])
                    if task.get("expected_terminal") == "clarify":
                        record["relevant_task_success"] = state.status == "clarify" and state.search_calls + state.lookup_calls == 0
                    record["elapsed_seconds"] = round(time.monotonic() - started, 3)
                    records[task["id"]], states[task["id"]] = record, asdict(state)
                    output["arms"][arm]["summary"] = aggregate(records)
                    verify()
                    save()
                    print(json.dumps(dict(arm=arm, task=task["id"], status=state.status,
                                          steps=state.step_count, mechanical=record["mechanical_completion"],
                                          relevant=record["relevant_task_success"])), flush=True)
    finally:
        output["mineru_calls"] = len(calls)
        try:
            verify()
            output["protected_inputs_unchanged"] = True
        finally:
            save()
            print("Private results: " + str(run_dir / "results.json"), flush=True)
    if calls:
        raise RuntimeError("MinerU boundary violated")
    return output


def run_synthetic():
    from evaluation import evaluate_agent0 as runner
    baseline = load_baseline()
    result = {}
    old_names = {k: getattr(runner, k) for k in ("AgentHarness", "KnowledgeBaseSession", "ToolResult", "OllamaActionSelector")}
    try:
        for arm in ("A", "B", "C"):
            if arm == "B":
                runner.AgentHarness = baseline["harness"].AgentHarness
                runner.KnowledgeBaseSession = baseline["tools"].KnowledgeBaseSession
                runner.ToolResult = baseline["tools"].ToolResult
                runner.OllamaActionSelector = baseline["selector"].OllamaActionSelector
            else:
                for key, value in old_names.items():
                    setattr(runner, key, value)
            selector = None if arm == "A" else runner.OllamaActionSelector()
            run = runner.run_benchmark(selector, include_states=True,
                                       progress=lambda name, r: print(f'{arm}/{name}: {r["status"]}', flush=True))
            metrics = {name: score_state(state) for name, state in run["states"].items()}
            result[arm] = dict(original=run, summary=aggregate(metrics), metrics=metrics)
    finally:
        for key, value in old_names.items():
            setattr(runner, key, value)
    target = private_path(PRIVATE_ROOT / ("synthetic_" + uuid4().hex + ".json"))
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Private synthetic results: " + str(target), flush=True)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest")
    p.add_argument("--search-observations")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--prepare-baseline", action="store_true")
    p.add_argument("--arms", nargs="+", choices=("A", "B", "C"), default=["A", "B", "C"])
    a = p.parse_args(argv)
    if a.prepare_baseline:
        print("Verified private baseline: " + str(prepare_baseline()))
    elif a.synthetic:
        run_synthetic()
    elif a.manifest:
        run_industrial(a.manifest, a.arms, a.search_observations)
    else:
        p.error("provide --manifest or --synthetic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
