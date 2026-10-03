import json
import unittest
from unittest.mock import patch

from src.agent.actions import InvalidAction
from src.agent.harness import AgentHarness
from src.agent.state import AgentState
from src.agent.tools import KnowledgeBaseSession
from test_agent_tools import TinyEncoder, TinyReranker, corpus


class ReferencePolicy:
    action_contract = "evidence_reference"

    def __init__(self, *, unsupported_b=False):
        self.unsupported_b = unsupported_b

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
            if ids and not (scope == "B" and self.unsupported_b):
                outcomes.append({"scope": scope, "status": "supported", "claim": "host cites selected evidence",
                                 "evidence_ids": ids[:1]})
            elif any(h["scopes"][0] == scope and h["results"] for h in state.search_history):
                outcomes.append({"scope": scope, "status": "insufficient_evidence"})
            else:
                outcomes.append({"scope": scope, "status": "no_candidates"})
        return {"action": "FINISH", "outcomes": outcomes}


class AgentReferenceHarnessTests(unittest.TestCase):
    def setUp(self):
        self.session = KnowledgeBaseSession(corpus(), model=TinyEncoder(),
                                            reranker_model=TinyReranker())

    def test_supported_plus_unsupported_scope_finishes_with_host_citations(self):
        state = AgentHarness(self.session, ReferencePolicy(unsupported_b=True)).run(
            "Rated pressure", ["A", "B"])
        self.assertEqual(state.status, "finished")
        self.assertEqual(state.comparison, "not_evaluated")
        self.assertEqual([row["status"] for row in state.findings],
                         ["supported", "insufficient_evidence"])
        evidence = state.findings[0]["evidence"][0]
        self.assertEqual(evidence["excerpt"], self.session.render_evidence_reference(
            evidence["evidence_id"])["excerpt"])
        self.assertEqual(evidence["source"], "A")
        self.assertIn("terminal_status", state.trace[-1])
        self.assertEqual(state.trace[-1]["terminal_status"], "finished")
        summary = json.dumps(state.trace[-1], ensure_ascii=False)
        self.assertIn("scope_statuses", summary)
        self.assertIn("host_spans", summary)
        self.assertIn("citation_provenance", summary)
        self.assertNotIn("Rated pressure: 10 MPa.", summary)

    def test_empty_search_is_no_candidates_and_not_tool_no_evidence_claim(self):
        class OneEmptyScopeSession:
            def __init__(self, delegate):
                self.delegate = delegate
                self.registry = delegate.registry

            def resolve_scopes(self, scopes):
                return self.delegate.resolve_scopes(scopes)

            def search_knowledge(self, query, scopes):
                if scopes == ["B"]:
                    from src.agent.tools import ToolResult
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
                         [("A", "supported"), ("B", "no_candidates")])
        self.assertEqual(state.search_history[1]["status"], "no_evidence")

    def test_host_rejects_foreign_unobserved_and_wrong_scope_ids(self):
        active = KnowledgeBaseSession({
            "A": [{"text": f"candidate{i}"} for i in range(4)],
            "B": [{"text": "B candidate"}],
        }, model=TinyEncoder(), reranker_model=TinyReranker())
        harness = AgentHarness(active, ReferencePolicy())
        own_a = next(row for row in active.registry.values() if row["source"] == "A")
        own_b = next(row for row in active.registry.values() if row["source"] == "B")
        foreign_session = KnowledgeBaseSession({"Z": [{"text": "foreign source"}]}, model=TinyEncoder())
        foreign_id = next(iter(foreign_session.registry))
        active_ids = {row["evidence_id"] for row in active.registry.values()}
        observed_ids = {own_a["evidence_id"], own_b["evidence_id"]}
        unobserved_id = next(iter(active_ids-observed_ids))
        state = AgentState("q", ["A", "B"], ["A", "B"])
        state.evidence_ids = list(observed_ids)
        state.looked_up_evidence = {
            row["evidence_id"]: active.lookup_evidence(row["evidence_id"]).results[0]
            for row in (own_a, own_b)}
        state.search_history = [
            {"scopes": ["A"], "query": "q", "status": "ok", "results": [own_a]},
            {"scopes": ["B"], "query": "q", "status": "ok", "results": [own_b]},
        ]
        bad_ids = [foreign_id, unobserved_id, own_a["evidence_id"]]
        bad_scopes = ["A", "A", "B"]
        for evidence_id, scope in zip(bad_ids, bad_scopes):
            action = {"action": "FINISH", "outcomes": [
                {"scope": scope, "status": "supported", "claim": "claim", "evidence_ids": [evidence_id]},
                {"scope": "B" if scope == "A" else "A", "status": "insufficient_evidence"},
            ]}
            with self.subTest(evidence_id=evidence_id, scope=scope), self.assertRaises(InvalidAction):
                harness._validate_state_action(state, action)

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
