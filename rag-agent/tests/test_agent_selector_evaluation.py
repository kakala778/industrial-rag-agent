import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class AgentSelectorEvaluationTests(unittest.TestCase):
    def test_selector_sends_action_schema_without_thinking_and_returns_validated_json(self):
        from src.agent.selector import OllamaActionSelector
        from src.agent.state import AgentState
        response = {"message": {"content": json.dumps({"action": "SEARCH", "query": "p",
                                                        "scopes": ["A"]}),
                                 "thinking": "never store me"}}
        received = {}

        def transport(request, timeout):
            received.update(json.loads(request.data))
            received["timeout"] = timeout
            return io.BytesIO(json.dumps(response).encode())

        with patch("src.agent.selector.urlopen", side_effect=transport):
            action = OllamaActionSelector(timeout=5)(AgentState("p", ["A", "B"], ["A", "B"]))
        self.assertEqual(action["action"], "SEARCH")
        self.assertFalse(received["think"])
        self.assertEqual(received["timeout"], 5)
        self.assertIn("oneOf", received["format"])
        self.assertNotIn("never store me", json.dumps(action))

    def test_selector_rejects_invalid_output_and_propagates_timeout(self):
        from src.agent.actions import InvalidAction
        from src.agent.selector import OllamaActionSelector
        from src.agent.state import AgentState
        for content in ("not JSON", '{"action":"shell"}', '{"action":"FINISH","findings":true}'):
            response = io.BytesIO(json.dumps({"message": {"content": content}}).encode())
            with patch("src.agent.selector.urlopen", return_value=response):
                with self.assertRaises(InvalidAction):
                    OllamaActionSelector()(AgentState("p", ["A", "B"]))
        with patch("src.agent.selector.urlopen", side_effect=TimeoutError):
            with self.assertRaises(TimeoutError):
                OllamaActionSelector()(AgentState("p", ["A", "B"]))

    def test_action_schema_hides_finish_before_lookup_and_constrains_observed_ids(self):
        from src.agent.selector import action_schema_for_state
        from src.agent.state import AgentState
        state = AgentState("p", ["A", "B"], ["A", "B"])
        state.evidence_ids = ["ev_seen"]
        state.search_history = [{"scopes": ["A"], "results": [{"evidence_id": "ev_seen", "source": "A"}]},
                                {"scopes": ["B"], "results": []}]
        schema = action_schema_for_state(state)
        actions = [s["properties"]["action"]["const"] for s in schema["oneOf"]]
        self.assertNotIn("FINISH", actions)
        lookup = next(s for s in schema["oneOf"] if s["properties"]["action"]["const"] == "LOOKUP")
        self.assertEqual(lookup["properties"]["evidence_id"]["enum"], ["ev_seen"])
        state.looked_up_evidence = {"ev_seen": {"source": "A"}}
        schema = action_schema_for_state(state)
        self.assertIn("FINISH", [s["properties"]["action"]["const"] for s in schema["oneOf"]])

    def test_frozen_benchmark_scores_harness_cases_separately_from_llm(self):
        from evaluation.evaluate_agent0 import run_benchmark
        result = run_benchmark()
        self.assertEqual(result["case_count"], 10)
        self.assertEqual(result["task_success"], 10)
        self.assertEqual(result["budget_violations"], 0)
        self.assertEqual(result["executed_invalid_tool_calls"], 0)
        self.assertEqual(len(result["benchmark_sha256"]), 64)
        self.assertEqual(result["case_results"]["invalid_id"]["injected_selector"], True)

    def test_benchmark_uses_na_for_no_claims_and_scores_actual_tool_counts(self):
        from evaluation.evaluate_agent0 import run_benchmark, score_case
        from src.agent.state import AgentState
        result = run_benchmark()
        self.assertIsNone(result["case_results"]["missing_scope"]["grounded"])
        self.assertIsNone(result["case_results"]["empty_search"]["citation_correct"])
        state = AgentState("p", ["A", "B"], ["A", "B"], status="incomplete",
                           search_calls=3, comparison="insufficient_evidence")
        state.search_history = [{"scopes": [s], "query": "p", "results": []}
                                for s in ("A", "A", "B")]
        state.trace = [{"action": "SEARCH", "tool_status": "no_evidence"}] * 3
        state.trace += [{"action": "FINISH", "tool_status": "ok"}]
        case = {"a": "", "b": "", "scopes": ["A", "B"], "status": "incomplete",
                "comparison": "insufficient_evidence", "covered": []}
        self.assertFalse(score_case(case, state, "p")["tool_selection_correct"])

    def test_output_boundary_rejects_paths_outside_private_directory(self):
        from src.agent.io import write_private_result
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("../escape", "D:/outside", "a/b", ""):
                with self.assertRaises(ValueError):
                    write_private_result({}, name=name, app_root=root)
            target = write_private_result({"trace": []}, name="test", app_root=root)
            self.assertTrue(target.is_relative_to(root / "outputs" / "agent0"))
            self.assertEqual(json.loads(target.read_text())["trace"], [])

    def test_cli_missing_scope_never_loads_files_or_models(self):
        from src.agent_demo import main
        with patch("src.agent_demo.load_corpus", side_effect=AssertionError("must not ingest")):
            with patch("sys.stdout", new_callable=io.StringIO) as out:
                status = main(["--query", "compare pressure"])
        self.assertEqual(status, 2)
        self.assertIn("clarify", out.getvalue())
