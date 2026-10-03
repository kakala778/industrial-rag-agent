import json
import unittest
from unittest.mock import patch

from test_agent_tools import corpus, TinyEncoder, TinyReranker


class AgentHarnessTests(unittest.TestCase):
    def setUp(self):
        from src.agent.harness import AgentHarness
        from src.agent.policy import DeterministicPolicy
        from src.agent.tools import KnowledgeBaseSession, ToolResult
        self.Harness, self.Policy, self.Result = AgentHarness, DeterministicPolicy, ToolResult
        self.session = KnowledgeBaseSession(corpus(), model=TinyEncoder(),
                                             reranker_model=TinyReranker())

    def run_task(self, scopes=None, **kwargs):
        return self.Harness(self.session, self.Policy(), **kwargs).run(
            "Rated pressure", scopes if scopes is not None else ["A", "B"])

    def test_normal_loop_compares_and_cites_both_sides(self):
        state = self.run_task()
        self.assertEqual(state.status, "finished")
        self.assertEqual([t["action"] for t in state.trace],
                         ["SEARCH", "SEARCH", "LOOKUP", "LOOKUP", "FINISH"])
        self.assertEqual(state.comparison, "different_text")
        self.assertEqual({f["scope"] for f in state.findings}, {"A", "B"})
        self.assertEqual(len(state.looked_up_evidence), 2)
        self.assertIn("10 MPa", state.answer)
        self.assertIn("12 MPa", state.answer)

    def test_missing_scope_clarifies_without_tools(self):
        for scopes in ([], ["A"], ["A", "A"]):
            state = self.run_task(scopes)
            self.assertEqual(state.status, "clarify")
            self.assertTrue(state.pending_clarification)
            self.assertEqual(state.search_calls + state.lookup_calls, 0)

    def test_invalid_scope_is_not_merged_into_corpus(self):
        state = self.run_task(["A", "UNKNOWN"])
        self.assertEqual(state.status, "invalid_scope")
        self.assertEqual(state.search_calls, 0)

    def test_missing_side_finishes_incomplete_with_supported_side(self):
        from src.agent.tools import KnowledgeBaseSession
        self.session = KnowledgeBaseSession({"A": corpus()["A"], "B": []},
                                             model=TinyEncoder(), reranker_model=TinyReranker())
        state = self.run_task()
        self.assertEqual(state.status, "incomplete")
        self.assertEqual(state.comparison, "insufficient_evidence")
        self.assertEqual(len(state.findings), 1)
        self.assertIn("B", state.answer)

    def test_no_evidence_is_not_tool_error(self):
        with patch.object(self.session, "search_knowledge", return_value=self.Result("no_evidence")):
            state = self.run_task()
        self.assertEqual(state.status, "incomplete")
        self.assertEqual(state.errors, [])
        self.assertEqual(state.lookup_calls, 0)

    def test_tool_failure_and_timeout_propagate(self):
        for result, expected in ((self.Result("error"), "tool_error"),
                                 (self.Result("timeout"), "timeout")):
            with patch.object(self.session, "search_knowledge", return_value=result):
                state = self.run_task()
            self.assertEqual(state.status, expected)
            self.assertEqual(state.lookup_calls, 0)
            self.assertTrue(state.errors)

    def test_unknown_lookup_returns_observation_and_terminates(self):
        state = self.Harness(self.session, lambda s: {
            "action": "LOOKUP", "evidence_id": "ev_unknown"}).run("pressure", ["A", "B"])
        self.assertEqual(state.status, "invalid_action")
        self.assertEqual(state.trace[0]["tool_status"], "invalid_evidence_id")
        self.assertEqual(state.lookup_calls, 0)

    def test_all_budgets_block_calls_before_execution(self):
        for kwargs, searches, lookups in (({"max_steps": 1}, 1, 0),
                                          ({"max_search_calls": 1}, 1, 0),
                                          ({"max_lookup_calls": 1}, 2, 1)):
            state = self.run_task(**kwargs)
            self.assertEqual(state.status, "budget_exceeded")
            self.assertEqual((state.search_calls, state.lookup_calls), (searches, lookups))

    def test_invalid_model_json_never_executes_tool(self):
        bad = ('{"action":"SHELL","command":"x"}',
               '{"action":"SEARCH","query":"q","scopes":"A"}',
               '{"action":"LOOKUP","evidence_id":"x","reasoning":"hidden"}',
               '```json\n{"action":"FINISH","findings":[]}\n```',
               '{"action":"SEARCH","action":"FINISH","findings":[]}',
               '{"action":"FINISH","findings":[],"score":NaN}')
        for raw in bad:
            state = self.Harness(self.session, lambda s, raw=raw: raw).run("p", ["A", "B"])
            self.assertEqual(state.status, "invalid_action")
            self.assertEqual(state.search_calls + state.lookup_calls, 0)
            self.assertNotIn("hidden", json.dumps(state.trace))

    def test_cross_scope_or_rewritten_search_is_rejected(self):
        for query, scopes in (("pressure", ["C"]), ("rewritten", ["A"]),
                              ("pressure", ["A", "B"])):
            state = self.Harness(self.session, lambda s: {
                "action": "SEARCH", "query": query, "scopes": scopes}).run("pressure", ["A", "B"])
            self.assertEqual(state.status, "invalid_action")
            self.assertEqual(state.search_calls, 0)

    def test_early_finish_is_rejected(self):
        state = self.Harness(self.session, lambda s: {"action": "FINISH", "findings": []}).run(
            "p", ["A", "B"])
        self.assertEqual(state.status, "invalid_action")

    def test_finish_after_search_requires_lookup_of_each_nonempty_scope(self):
        policy = self.Policy()
        def early(state):
            if len(state.search_history) == 2:
                return {"action": "FINISH", "findings": []}
            return policy(state)
        state = self.Harness(self.session, early).run("p", ["A", "B"])
        self.assertEqual(state.status, "invalid_action")
        self.assertEqual(state.lookup_calls, 0)

    def test_selector_receives_actual_remaining_budgets(self):
        seen = []
        policy = self.Policy()
        def select(state):
            seen.append(state.remaining_budget.copy())
            return policy(state)
        state = self.Harness(self.session, select, max_steps=7, max_search_calls=3,
                             max_lookup_calls=4).run("p", ["A", "B"])
        self.assertEqual(state.status, "finished")
        self.assertEqual(seen[0], {"steps": 6, "search": 3, "lookup": 4})
        self.assertEqual(seen[-1], {"steps": 2, "search": 1, "lookup": 2})

    def test_forged_quote_or_scope_is_rejected(self):
        policy = self.Policy()
        for mutation in ("quote", "scope", "evidence_id"):
            def select(state):
                action = policy(state)
                if action["action"] == "FINISH":
                    action["findings"][0][mutation] = "forged"
                return action
            state = self.Harness(self.session, select).run("p", ["A", "B"])
            self.assertEqual(state.status, "invalid_action")

    def test_selector_cannot_mutate_state_to_forge_lookup(self):
        def malicious(state):
            state.resolved_scopes = ["C"]
            state.status = "finished"
            return {"action": "FINISH", "findings": []}
        state = self.Harness(self.session, malicious).run("p", ["A", "B"])
        self.assertEqual(state.resolved_scopes, ["A", "B"])
        self.assertEqual(state.status, "invalid_action")

    def test_invalid_budgets_rejected(self):
        for value in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                self.Harness(self.session, self.Policy(), max_steps=value)

    def test_deterministic_finish_preserves_retrieved_target_beyond_parent_prefix(self):
        from src.agent.tools import KnowledgeBaseSession
        data = {scope: [{"text": "Maintenance notice. " * 75 + f"Rated pressure: {value} MPa.",
                        "metadata": {"source": scope}}] for scope, value in (("A", 10), ("B", 12))}
        self.session = KnowledgeBaseSession(data, model=TinyEncoder(), reranker_model=TinyReranker())
        def target(query, scopes):
            row = next(r for r in self.session.registry.values() if r["source"] == scopes[0]
                       and "Rated pressure" in r["text"])
            return self.Result("ok", [row.copy()])
        with patch.object(self.session, "search_knowledge", side_effect=target):
            state = self.run_task()
        self.assertIn("10 MPa", state.answer)
        self.assertIn("12 MPa", state.answer)
        self.assertEqual(state.comparison, "different_text")

    def test_multiple_findings_compare_sets_per_scope(self):
        from src.agent.tools import KnowledgeBaseSession
        for value, expected in ((10, "same_text"), (12, "different_text")):
            data = {scope: [{"text": text, "metadata": {"source": scope}} for text in
                            (f"Rated pressure: {pressure} MPa.", "Rated temperature: 80 C.")]
                    for scope, pressure in (("A", 10), ("B", value))}
            session = KnowledgeBaseSession(data, model=TinyEncoder(), reranker_model=TinyReranker())
            def choose(state):
                for scope in ("A", "B"):
                    if not any(h["scopes"] == [scope] for h in state.search_history):
                        return {"action": "SEARCH", "query": state.original_query, "scopes": [scope]}
                for eid in state.evidence_ids:
                    if eid not in state.looked_up_evidence:
                        return {"action": "LOOKUP", "evidence_id": eid}
                return {"action": "FINISH", "findings": [{"scope": row["source"],
                        "evidence_id": eid, "quote": row["text"]} for eid, row in state.looked_up_evidence.items()]}
            state = self.Harness(session, choose).run("parameters", ["A", "B"])
            self.assertEqual(state.status, "finished")
            self.assertEqual(state.comparison, expected)
