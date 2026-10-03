import unittest

from src.agent.tools import KnowledgeBaseSession
from test_agent_tools import TinyEncoder


class Agent03EvaluationTests(unittest.TestCase):
    def fixture(self, *, empty_b=False):
        from evaluation.agent03_metrics import score_reference
        session = KnowledgeBaseSession({"A": [{"text": "Rated pressure is 10 MPa indoors."}],
                                        "B": [] if empty_b else [{"text": "Other value is 12 MPa."}]},
                                       model=TinyEncoder())
        rows = {row["source"]: row for row in session.registry.values()}
        a = rows["A"]
        b = rows.get("B")
        evidence_ids = {a["evidence_id"]: a, **({b["evidence_id"]: b} if b else {})}
        state = {"status": "finished", "evidence_ids": list(evidence_ids),
                 "search_history": [
                     {"scopes": ["A"], "status": "ok", "results": [a]},
                     {"scopes": ["B"], "status": "no_evidence" if empty_b else "ok",
                      "results": [] if empty_b else [b]},
                 ],
                 "looked_up_evidence": {eid: {"source": row["source"], "page": None}
                                         for eid, row in evidence_ids.items()}}
        a_ref = session.render_evidence_reference(a["evidence_id"])
        outcomes = [{"scope": "A", "status": "supported", "claim": "supports rating",
                     "evidence_ids": [a["evidence_id"]]},
                    {"scope": "B", "status": "no_candidates" if empty_b else "insufficient_evidence"}]
        state["findings"] = [{"scope": "A", "status": "supported", "claim": "supports rating",
                              "evidence": [a_ref]},
                             {"scope": "B", "status": outcomes[1]["status"]}]
        task = {"scopes": ["A", "B"], "oracle": {
            "A": {"reviewed": True, "expected_available": True, "pages": [1],
                  "field_groups": [["Rated pressure"]], "unit_groups": [["MPa"]],
                  "condition_groups": [["indoors"]]},
            "B": {"reviewed": True, "expected_available": False, "pages": [1],
                  "field_groups": [["Rated pressure"]], "unit_groups": [["MPa"]],
                  "condition_groups": [["indoors"]]},
        }}
        reviews = {a["evidence_id"]: {"scope": "A", "label": "RELEVANT"}}
        if b:
            reviews[b["evidence_id"]] = {"scope": "B", "label": "IRRELEVANT"}
        return score_reference, session, state, outcomes, task, reviews, a

    def test_id_success_unsupported_side_and_condition_are_separate(self):
        score, session, state, outcomes, task, reviews, _ = self.fixture()
        result = score(state, outcomes, task, reviews, session)
        self.assertEqual(result["candidate_group"], "CANDIDATE_AVAILABLE")
        self.assertTrue(result["evidence_id_selection_success"])
        self.assertTrue(result["host_finish_accepted"])
        self.assertTrue(result["fully_relevant_task_success"])
        self.assertTrue(result["unsupported_side_correct"]["passed"])
        self.assertTrue(result["accepted_unsupported_side_correct"]["passed"])
        self.assertEqual(result["condition_alignment"], {"passed": 1, "assessed": 1})
        self.assertEqual(result["citation_authenticity"], {"passed": 1, "assessed": 1})
        self.assertEqual(result["span_bounds"], {"passed": 1, "assessed": 1})
        self.assertEqual(result["model_quote_fields"], 0)

    def test_rejected_finish_does_not_count_as_fully_relevant_result(self):
        score, session, state, outcomes, task, reviews, _ = self.fixture()
        state["status"] = "invalid_action"
        state["findings"] = []
        result = score(state, outcomes, task, reviews, session)
        self.assertTrue(result["evidence_id_selection_success"])
        self.assertFalse(result["host_finish_accepted"])
        self.assertFalse(result["fully_relevant_task_success"])
        self.assertEqual(result["unsupported_side_correct"], {"passed": 1, "assessed": 1})
        self.assertEqual(result["accepted_unsupported_side_correct"],
                         {"passed": 0, "assessed": 1})

    def test_no_candidates_is_distinct_from_candidates_rejected_as_irrelevant(self):
        score, session, state, outcomes, task, reviews, _ = self.fixture(empty_b=True)
        result = score(state, outcomes, task, reviews, session)
        self.assertTrue(result["unsupported_side_correct"]["passed"])
        self.assertEqual(result["scope_statuses"], {"supported": 1, "no_candidates": 1})

    def test_broken_or_unbounded_host_excerpt_fails_authenticity(self):
        score, session, state, outcomes, task, reviews, a = self.fixture()
        ref = state["findings"][0]["evidence"][0]
        ref["excerpt"] += " forged"
        ref["span"]["length"] = len(ref["excerpt"])
        result = score(state, outcomes, task, reviews, session)
        self.assertEqual(result["citation_authenticity"], {"passed": 0, "assessed": 1})
        self.assertEqual(result["span_bounds"], {"passed": 0, "assessed": 1})

    def test_retrieval_bound_group_is_not_a_model_selection_success(self):
        from evaluation.agent03_metrics import candidate_group
        task = {"scopes": ["A", "B"], "oracle": {"A": {"expected_available": True},
                           "B": {"expected_available": True}}}
        reviews = {"a": {"scope": "A", "label": "IRRELEVANT"},
                   "b": {"scope": "B", "label": "RELEVANT"}}
        self.assertEqual(candidate_group(task, reviews), "RETRIEVAL_BOUND")


if __name__ == "__main__":
    unittest.main()
