import json
import unittest

from src.agent.actions import InvalidAction
from src.agent.selector import action_schema_for_state, parse_selector_action
from src.agent.state import AgentState


def ready_state():
    state = AgentState("compare ratings", ["A", "B"])
    state.resolved_scopes = ["A", "B"]
    state.search_history = [
        {"scopes": ["A"], "query": state.original_query, "status": "ok",
         "results": [{"evidence_id": "ev_a", "source": "A"}]},
        {"scopes": ["B"], "query": state.original_query, "status": "ok",
         "results": [{"evidence_id": "ev_b", "source": "B"}]},
    ]
    state.evidence_ids = ["ev_a", "ev_b"]
    state.looked_up_evidence = {
        "ev_a": {"evidence_id": "ev_a", "source": "A", "text": "source A"},
        "ev_b": {"evidence_id": "ev_b", "source": "B", "text": "source B"},
    }
    state.remaining_budget = {"steps": 1, "search": 2, "lookup": 4}
    return state


class AgentReferenceActionTests(unittest.TestCase):
    def test_supported_and_insufficient_evidence_outcomes_parse(self):
        state = ready_state()
        action = {"action": "FINISH", "outcomes": [
            {"scope": "A", "status": "supported", "claim": "rated pressure is 10 MPa",
             "evidence_ids": ["ev_a"]},
            {"scope": "B", "status": "insufficient_evidence"},
        ]}
        self.assertEqual(parse_selector_action(json.dumps(action), state,
                                               contract="evidence_reference"), action)

    def test_no_candidates_is_a_distinct_valid_scope_outcome(self):
        state = ready_state()
        state.search_history[1] = {"scopes": ["B"], "query": state.original_query,
                                   "status": "no_evidence", "results": []}
        state.evidence_ids = ["ev_a"]
        state.looked_up_evidence.pop("ev_b")
        action = {"action": "FINISH", "outcomes": [
            {"scope": "A", "status": "supported", "claim": "present", "evidence_ids": ["ev_a"]},
            {"scope": "B", "status": "no_candidates"},
        ]}
        self.assertEqual(parse_selector_action(json.dumps(action), state,
                                               contract="evidence_reference"), action)

    def test_reference_action_schema_never_requests_model_quotes(self):
        state = ready_state()
        schema = action_schema_for_state(state, contract="evidence_reference")
        finish = next(row for row in schema["oneOf"]
                      if row["properties"]["action"]["const"] == "FINISH")
        outcome = finish["properties"]["outcomes"]["items"]["oneOf"]
        supported = next(row for row in outcome
                         if row["properties"]["status"]["const"] == "supported")
        self.assertIn("evidence_ids", supported["properties"])
        self.assertNotIn("quote", supported["properties"])

    def test_reference_finishes_require_each_scope_exactly_once(self):
        state = ready_state()
        valid_a = {"scope": "A", "status": "supported", "claim": "claim", "evidence_ids": ["ev_a"]}
        valid_b = {"scope": "B", "status": "supported", "claim": "claim", "evidence_ids": ["ev_b"]}
        for outcomes in ([valid_a], [valid_a, valid_a],
                         [valid_a, dict(valid_b, scope="C")]):
            with self.subTest(outcomes=outcomes), self.assertRaises(InvalidAction):
                parse_selector_action(json.dumps({"action": "FINISH", "outcomes": outcomes}),
                                      state, contract="evidence_reference")

    def test_unavailable_ids_and_unsupported_payloads_are_rejected(self):
        state = ready_state()
        invalid = [
            {"scope": "A", "status": "supported", "claim": "claim", "evidence_ids": ["ev_forged"]},
            {"scope": "A", "status": "insufficient_evidence", "evidence_ids": ["ev_a"]},
            {"scope": "A", "status": "no_candidates", "claim": "claim"},
            {"scope": "A", "status": "supported", "claim": "claim", "evidence_ids": ["ev_a"], "quote": "fake"},
            {"scope": "A", "status": "supported", "claim": "claim", "evidence_ids": ["ev_b"]},
        ]
        for bad in invalid:
            action = {"action": "FINISH", "outcomes": [bad,
                      {"scope": "B", "status": "insufficient_evidence"}]}
            with self.subTest(bad=bad), self.assertRaises(InvalidAction):
                parse_selector_action(json.dumps(action), state, contract="evidence_reference")

    def test_copied_quote_contract_remains_default(self):
        state = ready_state()
        self.assertEqual(action_schema_for_state(state),
                         action_schema_for_state(state, contract="copied_quote"))


if __name__ == "__main__":
    unittest.main()
