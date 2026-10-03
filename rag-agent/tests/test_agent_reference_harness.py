import json
import unittest
from unittest.mock import patch

from src.agent.actions import InvalidAction
from src.agent.harness import AgentHarness
from src.agent.state import AgentState
from src.agent.tools import KnowledgeBaseSession, ToolResult
from test_agent_tools import TinyEncoder, TinyReranker, corpus


class ReferencePolicy:
    action_contract = "evidence_reference"

    def __init__(self, *, insufficient_scope=None):
        self.insufficient_scope = insufficient_scope

    def __call__(self, state):
        searched = {row["scopes"][0] for row in state.search_history}
        for scope in state.resolved_scopes:
            if scope not in searched:
                return {"action": "SEARCH", "query": state.original_query, "scopes": [scope]}
        for evidence_id in state.evidence_ids:
            if evidence_id not in state.looked_up_evidence:
                return {"action": "LOOKUP", "evidence_id": evidence_id}
        outcomes = []
        for scope in state.resolved_scopes:
            ids = [eid for eid, row in state.looked_up_evidence.items() if row["source"] == scope]
            candidates = [row for history in state.search_history
                          if history["scopes"][0] == scope for row in history["results"]]
            if ids and scope != self.insufficient_scope:
                outcomes.append({"scope": scope, "status": "evidence_found",
                                 "evidence_ids": ids[:1]})
            elif candidates:
                outcomes.append({"scope": scope, "status": "insufficient_scope"})
            else:
                outcomes.append({"scope": scope, "status": "no_evidence_found"})
        return {"action": "FINISH", "outcomes": outcomes}


class ClarifyPolicy:
    action_contract = "evidence_reference"

    def __call__(self, state):
        return {"action": "CLARIFY", "question": "请明确需要查询的设备型号。"}


class CopiedQuotePolicy:
    action_contract = "copied_quote"

    def __call__(self, state):
        return {"action": "FINISH", "findings": []}


def session_for(scopes):
    data = {scope: [{"text": f"Rated pressure for {scope}: {10 + index} MPa.",
                     "metadata": {"page": index + 1, "block_type": "paragraph",
                                  "block_index": 0}}
                    ] for index, scope in enumerate(scopes)}
    return KnowledgeBaseSession(data, model=TinyEncoder(), reranker_model=TinyReranker())


class AgentReferenceHarnessTests(unittest.TestCase):
    def setUp(self):
        self.session = KnowledgeBaseSession(corpus(), model=TinyEncoder(),
                                            reranker_model=TinyReranker())

    def test_two_three_and_four_scopes_finish_with_host_citations_and_no_claims(self):
        for scopes in (["A", "B"], ["A", "B", "C"], ["A", "B", "C", "D"]):
            session = self.session if len(scopes) == 2 else session_for(scopes)
            state = AgentHarness(session, ReferencePolicy()).run("Rated pressure", scopes)
            with self.subTest(scopes=scopes):
                self.assertEqual(state.status, "finished")
                self.assertEqual([row["scope"] for row in state.findings], scopes)
                self.assertEqual([row["status"] for row in state.findings],
                                 ["evidence_found"] * len(scopes))
                self.assertTrue(all("claim" not in row for row in state.findings))
                searches = [row for row in state.trace if row["action"] == "SEARCH"]
                self.assertEqual([row["tool_input"]["scopes"][0] for row in searches], scopes)
                self.assertEqual(len(searches), len(scopes))
                self.assertEqual(state.trace[-1]["terminal_status"], "finished")
                for finding in state.findings:
                    evidence = finding["evidence"][0]
                    self.assertEqual(evidence["source"], finding["scope"])
                    self.assertEqual(evidence["excerpt"], session.render_evidence_reference(
                        evidence["evidence_id"])["excerpt"])
                trace = json.dumps(state.trace[-1], ensure_ascii=False)
                self.assertIn("scope_statuses", trace)
                self.assertIn("host_spans", trace)
                self.assertIn("citation_provenance", trace)
                self.assertNotIn("Rated pressure for A:", trace)

    def test_empty_search_is_no_evidence_found_only_for_the_empty_scope(self):
        class OneEmptyScopeSession:
            def __init__(self, delegate):
                self.delegate = delegate
                self.registry = delegate.registry

            def resolve_scopes(self, scopes):
                return self.delegate.resolve_scopes(scopes)

            def search_knowledge(self, query, scopes):
                if scopes == ["B"]:
                    return ToolResult("no_evidence")
                return self.delegate.search_knowledge(query, scopes)

            def lookup_evidence(self, evidence_id):
                return self.delegate.lookup_evidence(evidence_id)

            def render_evidence_reference(self, evidence_id):
                return self.delegate.render_evidence_reference(evidence_id)

        state = AgentHarness(OneEmptyScopeSession(self.session), ReferencePolicy()).run(
            "Rated pressure", ["A", "B"])
        self.assertEqual(state.status, "finished")
        self.assertEqual([(row["scope"], row["status"]) for row in state.findings],
                         [("A", "evidence_found"), ("B", "no_evidence_found")])
        self.assertEqual(state.search_history[1]["status"], "no_evidence")

    def test_empty_successful_lookups_exhaust_budget_without_findings(self):
        class EmptyLookupSession:
            def __init__(self, delegate):
                self.delegate = delegate
                self.registry = delegate.registry

            def resolve_scopes(self, scopes):
                return self.delegate.resolve_scopes(scopes)

            def search_knowledge(self, query, scopes):
                return self.delegate.search_knowledge(query, scopes)

            def lookup_evidence(self, evidence_id):
                return ToolResult("ok", [])

            def render_evidence_reference(self, evidence_id):
                return self.delegate.render_evidence_reference(evidence_id)

        state = AgentHarness(EmptyLookupSession(self.session), ReferencePolicy(),
                             max_lookup_calls=2).run("Rated pressure", ["A", "B"])
        self.assertEqual(state.status, "budget_exceeded")
        self.assertEqual(state.findings, [])
        self.assertEqual(state.lookup_calls, 2)
        self.assertEqual(state.lookup_history[0]["status"], "ok")
        self.assertEqual(state.trace[-1]["terminal_status"], "budget_exceeded")

    def test_host_rejects_foreign_unobserved_unlooked_up_and_wrong_scope_ids(self):
        active = KnowledgeBaseSession({
            "A": [{"text": f"candidate{i}"} for i in range(4)],
            "B": [{"text": "B candidate"}],
        }, model=TinyEncoder(), reranker_model=TinyReranker())
        harness = AgentHarness(active, ReferencePolicy())
        own_a = next(row for row in active.registry.values() if row["source"] == "A")
        own_b = next(row for row in active.registry.values() if row["source"] == "B")
        foreign_session = KnowledgeBaseSession({"Z": [{"text": "foreign source"}]},
                                               model=TinyEncoder())
        foreign_id = next(iter(foreign_session.registry))
        observed_a_unlooked = next(row["evidence_id"] for row in active.registry.values()
                                   if row["source"] == "A" and row["evidence_id"] != own_a["evidence_id"])
        active_ids = {row["evidence_id"] for row in active.registry.values()}
        observed_ids = {own_a["evidence_id"], own_b["evidence_id"], observed_a_unlooked}
        unobserved_id = next(iter(active_ids - observed_ids))
        state = AgentState("q", ["A", "B"], ["A", "B"])
        state.evidence_ids = list(observed_ids)
        state.looked_up_evidence = {
            row["evidence_id"]: active.lookup_evidence(row["evidence_id"]).results[0]
            for row in (own_a, own_b)}
        state.search_history = [
            {"scopes": ["A"], "query": "q", "status": "ok", "results": [own_a]},
            {"scopes": ["B"], "query": "q", "status": "ok", "results": [own_b]},
        ]
        bad_ids = [foreign_id, unobserved_id, observed_a_unlooked, own_a["evidence_id"]]
        bad_scopes = ["A", "A", "A", "B"]
        for evidence_id, scope in zip(bad_ids, bad_scopes):
            action = {"action": "FINISH", "outcomes": [
                {"scope": scope, "status": "evidence_found", "evidence_ids": [evidence_id]},
                {"scope": "B" if scope == "A" else "A", "status": "insufficient_scope"},
            ]}
            with self.subTest(evidence_id=evidence_id, scope=scope), self.assertRaises(InvalidAction):
                harness._validate_state_action(state, action)

    def test_reference_statuses_require_consistent_search_and_lookup_coverage(self):
        harness = AgentHarness(self.session, ReferencePolicy())
        candidate = next(row for row in self.session.registry.values() if row["source"] == "A")
        candidate_state = AgentState("q", ["A", "B"], ["A", "B"])
        candidate_state.search_history = [
            {"scopes": ["A"], "query": "q", "status": "ok", "results": [candidate]},
            {"scopes": ["B"], "query": "q", "status": "no_evidence", "results": []},
        ]
        candidate_state.evidence_ids = [candidate["evidence_id"]]
        candidate_state.looked_up_evidence[candidate["evidence_id"]] = (
            self.session.lookup_evidence(candidate["evidence_id"]).results[0])
        invalid_status_actions = [
            {"action": "FINISH", "outcomes": [
                {"scope": "A", "status": "no_evidence_found"},
                {"scope": "B", "status": "no_evidence_found"},
            ]},
            {"action": "FINISH", "outcomes": [
                {"scope": "A", "status": "evidence_found",
                 "evidence_ids": [candidate["evidence_id"]]},
                {"scope": "B", "status": "insufficient_scope"},
            ]},
        ]
        for action in invalid_status_actions:
            with self.subTest(action=action), self.assertRaises(InvalidAction):
                harness._validate_state_action(candidate_state, action)

        not_looked_up = AgentState("q", ["A", "B"], ["A", "B"])
        not_looked_up.search_history = candidate_state.search_history
        not_looked_up.evidence_ids = [candidate["evidence_id"]]
        not_looked_up_action = {"action": "FINISH", "outcomes": [
            {"scope": "A", "status": "insufficient_scope"},
            {"scope": "B", "status": "no_evidence_found"},
        ]}
        with self.assertRaises(InvalidAction):
            harness._validate_state_action(not_looked_up, not_looked_up_action)

    def test_reference_clarify_is_terminal_without_tool_execution(self):
        state = AgentHarness(self.session, ClarifyPolicy()).run("Find the relevant rating", ["A", "B"])
        self.assertEqual(state.status, "clarify")
        self.assertEqual(state.pending_clarification, "请明确需要查询的设备型号。")
        self.assertEqual(state.search_calls, 0)
        self.assertEqual(state.lookup_calls, 0)
        self.assertEqual(state.trace[-1]["action"], "CLARIFY")
        self.assertEqual(state.trace[-1]["terminal_status"], "clarify")

    def test_reference_scope_count_boundaries_and_copied_quote_stays_two_scope(self):
        for scopes in (["A"], ["A", "B", "C", "D", "E"]):
            state = AgentHarness(self.session, ReferencePolicy()).run("q", scopes)
            with self.subTest(scopes=scopes):
                self.assertEqual(state.status, "clarify")
        for scopes in (["A"], ["A", "B", "C"]):
            state = AgentHarness(self.session, CopiedQuotePolicy()).run("q", scopes)
            with self.subTest(contract="copied_quote", scopes=scopes):
                self.assertEqual(state.status, "clarify")

    def test_reference_failure_never_mutates_source_findings(self):
        originals = self.session.registry
        before = json.dumps(originals, sort_keys=True)
        state = AgentHarness(self.session, ReferencePolicy()).run("Rated pressure", ["A", "B"])
        self.assertEqual(json.dumps(originals, sort_keys=True), before)
        self.assertTrue(all("quote" not in row for row in state.findings))

    def test_host_renderer_failure_terminates_without_crashing_or_partial_findings(self):
        with patch.object(self.session, "render_evidence_reference",
                          side_effect=ValueError("private renderer detail")):
            state = AgentHarness(self.session, ReferencePolicy()).run("Rated pressure", ["A", "B"])
        self.assertEqual(state.status, "invalid_action")
        self.assertEqual(state.findings, [])
        self.assertNotIn("private renderer detail", state.answer)
        self.assertEqual(state.trace[-1]["terminal_status"], "invalid_action")


if __name__ == "__main__":
    unittest.main()
