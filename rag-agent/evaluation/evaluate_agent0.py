"""Frozen synthetic harness benchmark; not industrial retrieval/answer accuracy."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.harness import AgentHarness
from src.agent.io import write_private_result
from src.agent.policy import DeterministicPolicy
from src.agent.selector import OllamaActionSelector
from src.agent.tools import KnowledgeBaseSession, ToolResult


CASE_FILE = Path(__file__).with_name("agent0_cases.json")


class SyntheticEncoder:
    """Constant vectors isolate harness behavior from model quality."""
    def encode(self, texts, **kwargs):
        return np.array([1., 0.]) if isinstance(texts, str) else np.array([[1., 0.] for _ in texts])


class SyntheticReranker:
    def predict(self, pairs, **kwargs):
        return np.zeros(len(pairs))


class FaultSession:
    """Evaluation-only dependency fault injection, outside production tool API."""
    def __init__(self, session):
        self.session = session

    def resolve_scopes(self, scopes):
        return self.session.resolve_scopes(scopes)

    def search_knowledge(self, **kwargs):
        return ToolResult("error", message="injected benchmark fault")

    def lookup_evidence(self, **kwargs):
        return self.session.lookup_evidence(**kwargs)


def score_case(case, state, query, *, max_steps=12):
    expected = {"A": case["a"], "B": case["b"]}
    grounded = all(row["evidence_id"] in state.looked_up_evidence
                   and row["quote"] in state.looked_up_evidence[row["evidence_id"]]["text"]
                   and row["scope"] == state.looked_up_evidence[row["evidence_id"]]["source"]
                   for row in state.findings)
    relevant = all(expected.get(row["scope"], "") in row["quote"] for row in state.findings)
    covered = {row["scope"] for row in state.findings}
    success = (state.status == case["status"] and covered == set(case["covered"])
               and state.comparison == case["comparison"] and grounded and relevant)
    scopes_ok = all(h["scopes"][0] in case["scopes"] for h in state.search_history)
    arguments_ok = all(h["query"] == query and len(h["scopes"]) == 1 for h in state.search_history)
    arguments_ok &= all(h["evidence_id"] in state.evidence_ids for h in state.lookup_history)
    budget_violation = state.step_count > max_steps or state.search_calls > 4 or state.lookup_calls > 6
    terminal = case["status"]
    if terminal in ("finished", "incomplete"):
        # Order may vary, but each frozen simple case needs exactly one search
        # per scope, one lookup per supported scope and one final action.
        selection_ok = (state.search_calls == 2 and
                        {h["scopes"][0] for h in state.search_history} == {"A", "B"}
                        and state.lookup_calls == len(case["covered"])
                        and sum(t["action"] == "FINISH" for t in state.trace) == 1)
    elif terminal == "tool_error":
        selection_ok = state.search_calls == 1 and state.lookup_calls == 0
    elif terminal == "budget_exceeded":
        selection_ok = state.search_calls == 1 and state.lookup_calls == 0
    else:
        expected_action = {"clarify": "CLARIFY", "invalid_scope": "PREFLIGHT",
                           "invalid_action": "LOOKUP"}.get(terminal)
        selection_ok = (state.search_calls + state.lookup_calls == 0
                        and len(state.trace) == 1 and state.trace[0]["action"] == expected_action)
    selection_ok &= state.status == terminal
    return {"task_success": bool(success), "status": state.status,
            "comparison": state.comparison, "tool_selection_correct": selection_ok,
            "tool_arguments_correct": bool(arguments_ok) if state.search_calls + state.lookup_calls else None,
            "scope_correct": scopes_ok if state.search_calls else None,
            "grounded": grounded if state.findings else None,
            "relevant": relevant if state.findings else None,
            "citation_correct": (grounded and all(r["evidence_id"] in state.answer for r in state.findings))
            if state.findings else None,
            "finding_count": len(state.findings), "step_count": state.step_count,
            "search_calls": state.search_calls, "lookup_calls": state.lookup_calls,
            "executed_invalid_tool_calls": sum(h["evidence_id"] not in state.evidence_ids for h in state.lookup_history),
            "invalid_action_attempts": int(state.status == "invalid_action"),
            "budget_violation": bool(budget_violation)}


def run_benchmark(selector=None, *, include_states=False, progress=None):
    raw = CASE_FILE.read_bytes()
    frozen = json.loads(raw)
    selector = selector or DeterministicPolicy()
    records, states = {}, {}
    for case in frozen["cases"]:
        corpus = {alias: [{"text": case[key], "metadata": {"source": alias, "page": 1,
                    "block_type": "table", "block_index": 0}}] if case[key] else []
                  for alias, key in (("A", "a"), ("B", "b"))}
        session = KnowledgeBaseSession(corpus, model=SyntheticEncoder(),
                                       reranker_model=SyntheticReranker())
        injected = case.get("injection") == "invalid_id"
        policy = (lambda state: {"action": "LOOKUP", "evidence_id": "ev_unknown"}) if injected else selector
        if case.get("injection") == "tool_error":
            session = FaultSession(session)
        start = time.monotonic()
        state = AgentHarness(session, policy, max_steps=case.get("max_steps", 12)).run(
            frozen["query"], case["scopes"])
        record = score_case(case, state, frozen["query"], max_steps=case.get("max_steps", 12))
        record.update(injected_selector=injected, elapsed_seconds=round(time.monotonic() - start, 3),
                      model_selection_exercised=isinstance(selector, OllamaActionSelector)
                      and not injected and state.step_count > 0)
        records[case["id"]] = record
        if include_states:
            from dataclasses import asdict
            states[case["id"]] = asdict(state)
        if progress:
            progress(case["id"], record)
    if CASE_FILE.read_bytes() != raw:
        raise RuntimeError("benchmark changed during inference")
    result = {"benchmark_sha256": hashlib.sha256(raw).hexdigest(), "version": frozen["version"],
              "selector": type(selector).__name__, "case_count": len(records),
              "task_success": sum(r["task_success"] for r in records.values()),
              "budget_violations": sum(r["budget_violation"] for r in records.values()),
              "executed_invalid_tool_calls": sum(r["executed_invalid_tool_calls"] for r in records.values()),
              "case_results": records}
    if include_states:
        result["states"] = states
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("deterministic", "qwen"), default="deterministic")
    parser.add_argument("--save-local", action="store_true")
    args = parser.parse_args(argv)
    selector = OllamaActionSelector() if args.policy == "qwen" else None
    result = run_benchmark(selector, include_states=args.save_local,
                           progress=lambda name, r: print(f'{name}: {r["status"]}, success={r["task_success"]}', flush=True))
    public = {key: value for key, value in result.items() if key != "states"}
    print(json.dumps(public, indent=2))
    if args.save_local:
        print(f"Private result: {write_private_result(result, name=args.policy)}")
    return 0 if result["task_success"] == result["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
