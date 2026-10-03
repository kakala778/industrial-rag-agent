"""Regression tests for repeated execution and incomplete comparison coverage."""
import unittest

from test_agent_tools import corpus, TinyEncoder, TinyReranker
from src.agent.harness import AgentHarness
from src.agent.policy import DeterministicPolicy
from src.agent.selector import action_schema_for_state
from src.agent.tools import KnowledgeBaseSession


class ProgressTests(unittest.TestCase):
    def session(self, data=None):
        return KnowledgeBaseSession(data or corpus(), model=TinyEncoder(),
                                    reranker_model=TinyReranker())

    def test_duplicate_search_does_not_use_tool_budget_and_can_recover(self):
        policy = DeterministicPolicy()
        def select(s):
            if s.step_count == 2:
                return dict(action="SEARCH", query=s.original_query, scopes=["A"])
            return policy(s)
        s = AgentHarness(self.session(), select, max_search_calls=2).run("p", ["A", "B"])
        self.assertEqual(s.status, "finished")
        self.assertEqual(s.search_calls, 2)
        self.assertEqual(s.step_count, 6)
        self.assertEqual(s.trace[1]["tool_status"], "no_progress")
        self.assertFalse(s.trace[1]["progress"]["progress"])

    def test_duplicate_lookup_does_not_use_tool_budget_at_limit(self):
        policy = DeterministicPolicy()
        def select(s):
            if s.step_count == 4:
                return dict(action="LOOKUP", evidence_id=next(iter(s.looked_up_evidence)))
            return policy(s)
        s = AgentHarness(self.session(), select, max_lookup_calls=2).run("p", ["A", "B"])
        self.assertEqual(s.status, "finished")
        self.assertEqual(s.lookup_calls, 2)
        self.assertEqual(s.trace[3]["tool_status"], "no_progress")

    def test_persistent_duplicates_are_step_bounded(self):
        s = AgentHarness(self.session(), lambda s: dict(action="SEARCH", query="p", scopes=["A"]),
                         max_steps=5).run("p", ["A", "B"])
        self.assertEqual(s.status, "budget_exceeded")
        self.assertEqual(s.search_calls, 1)
        self.assertEqual(sum(t["tool_status"] == "no_progress" for t in s.trace), 4)

    def test_schema_prioritizes_uncovered_scope_without_repeating_ids(self):
        saved = []
        policy = DeterministicPolicy()
        def select(s):
            if s.step_count == 4:
                saved.append((s, action_schema_for_state(s)))
            return policy(s)
        AgentHarness(self.session(), select).run("p", ["A", "B"])
        s, schema = saved[0]
        rows = {r["properties"]["action"]["const"]: r for r in schema["oneOf"]}
        self.assertNotIn("SEARCH", rows)
        self.assertNotIn("FINISH", rows)
        self.assertNotIn("CLARIFY", rows)
        ids = rows["LOOKUP"]["properties"]["evidence_id"]["enum"]
        self.assertEqual(ids, [eid for eid in s.evidence_ids if eid not in s.looked_up_evidence])
        self.assertEqual(s.progress["coverage"], {"A": "LOOKED_UP_EVIDENCE", "B": "CANDIDATES_OBSERVED"})

    def test_runtime_blocks_new_lookup_in_covered_scope_until_other_scope_covered(self):
        data = corpus()
        data["A"].append({"text": "Another pressure candidate.", "metadata": {"page": 2}})
        policy = DeterministicPolicy()
        def select(s):
            if s.step_count == 4:
                eid = next(r["evidence_id"] for h in s.search_history for r in h["results"]
                           if r["source"] == "A" and r["evidence_id"] not in s.looked_up_evidence)
                return dict(action="LOOKUP", evidence_id=eid)
            return policy(s)
        s = AgentHarness(self.session(data), select).run("p", ["A", "B"])
        self.assertEqual(s.status, "invalid_action")
        self.assertEqual(s.lookup_calls, 1)

    def test_unfinished_work_cannot_request_user_clarification(self):
        s = AgentHarness(self.session(), lambda s: dict(action="CLARIFY", question="I have not looked up B")).run("p", ["A", "B"])
        self.assertEqual(s.status, "invalid_action")
        self.assertEqual(s.search_calls + s.lookup_calls, 0)

    def test_explicit_missing_constraint_clarifies_without_tools(self):
        s = AgentHarness(self.session(), DeterministicPolicy()).run(
            "compare limits", ["A", "B"], clarification_required="Which operating condition?")
        self.assertEqual(s.status, "clarify")
        self.assertEqual(s.pending_clarification, "Which operating condition?")
        self.assertEqual(s.search_calls + s.lookup_calls, 0)

    def test_empty_scope_is_covered_but_comparison_remains_incomplete(self):
        s = AgentHarness(self.session({"A": corpus()["A"], "B": []}), DeterministicPolicy()).run("p", ["A", "B"])
        self.assertEqual(s.status, "incomplete")
        self.assertEqual(s.progress["coverage"], {"A": "LOOKED_UP_EVIDENCE", "B": "NO_EVIDENCE"})
        self.assertEqual(s.progress["remaining_scopes"], [])
        self.assertTrue(s.trace[-1]["progress"]["progress"])

    def test_malformed_scope_values_clarify_without_progress_crash(self):
        s = AgentHarness(self.session(), DeterministicPolicy()).run("p", [["A"], "B"])
        self.assertEqual(s.status, "clarify")
        self.assertEqual(s.search_calls, 0)

    def test_invalid_coverage_does_not_mark_known_scope_invalid(self):
        s = AgentHarness(self.session(), DeterministicPolicy()).run("p", ["A", "UNKNOWN"])
        self.assertEqual(s.progress["coverage"], {"A": "UNSEARCHED", "UNKNOWN": "INVALID_SCOPE"})

    def test_exhausted_eligible_tools_stop_before_calling_selector_again(self):
        calls = []
        policy = DeterministicPolicy()
        def select(s):
            calls.append(s.step_count)
            return policy(s)
        s = AgentHarness(self.session(), select, max_lookup_calls=0).run("p", ["A", "B"])
        self.assertEqual(s.status, "budget_exceeded")
        self.assertEqual(calls, [1, 2])


if __name__ == "__main__":
    unittest.main()
