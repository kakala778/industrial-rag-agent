import unittest

from src.agent.state import AgentState


class Evaluation01Tests(unittest.TestCase):
    def oracle(self):
        return {"A": {"reviewed": True, "expected_available": True, "pages": [1],
                       "field_groups": [["rated pressure"], ["10"]],
                       "unit_groups": [["mpa"]], "condition_groups": [["pump"]]}}

    def state(self, quote, text=None):
        s = AgentState("pump rated pressure", ["A"], ["A"], status="finished")
        s.search_history = [{"query": s.original_query, "scopes": ["A"], "status": "ok",
                             "results": [{"evidence_id": "ev_a", "source": "A"}]}]
        s.evidence_ids = ["ev_a"]
        s.looked_up_evidence = {"ev_a": {"source": "A", "page": 1, "text": text or quote}}
        s.lookup_history = [{"evidence_id": "ev_a", "status": "ok", "results": []}]
        s.findings = [{"scope": "A", "evidence_id": "ev_a", "quote": quote}]
        s.answer = quote + " [ev_a]"
        return s

    def test_grounded_wrong_field_is_not_relevant_completion(self):
        from evaluation.agent01_metrics import score_state
        r = score_state(self.state("Cable length: 10 m"), self.oracle())
        self.assertTrue(r["grounding"])
        self.assertTrue(r["mechanical_completion"])
        self.assertFalse(r["relevant_task_success"])
        self.assertEqual(r["relevance_review"][0]["label"], "IRRELEVANT")

    def test_matching_value_without_applicability_is_partial(self):
        from evaluation.agent01_metrics import score_state
        r = score_state(self.state("Rated pressure: 10 MPa"), self.oracle())
        self.assertEqual(r["relevance_review"][0]["label"], "PARTIAL")
        self.assertFalse(r["relevance_review"][0]["condition_alignment"])

    def test_aligned_quote_and_scope_are_relevant(self):
        from evaluation.agent01_metrics import score_state
        r = score_state(self.state("Pump rated pressure: 10 MPa"), self.oracle())
        self.assertTrue(r["relevant_task_success"])
        self.assertEqual(r["relevance_review"][0]["label"], "RELEVANT")

    def test_no_findings_are_na_and_unknown_gt_is_not_success(self):
        from evaluation.agent01_metrics import score_state
        s = self.state("x"); s.findings = []
        oracle = self.oracle(); oracle["A"]["expected_available"] = None
        r = score_state(s, oracle)
        self.assertIsNone(r["grounding"])
        self.assertIsNone(r["citation_grounding"])
        self.assertIsNone(r["relevant_task_success"])

    def test_duplicate_metrics_distinguish_execution_from_guard_rejections(self):
        from evaluation.agent01_metrics import score_state
        s = self.state("Pump rated pressure: 10 MPa")
        s.trace = [dict(action="SEARCH", tool_input=dict(query="q", scopes=["A"]), tool_status="ok"),
                   dict(action="SEARCH", tool_input=dict(query="q", scopes=["A"]), tool_status="ok"),
                   dict(action="SEARCH", tool_input=dict(query="q", scopes=["A"]), tool_status="no_progress")]
        r = score_state(s, self.oracle())
        self.assertEqual(r["repeated_search_executed"], 1)
        self.assertEqual(r["guard_rejections"], 1)
        self.assertEqual(r["no_progress_actions"], 2)

    def test_unit_alignment_does_not_accept_mm_as_m(self):
        from evaluation.agent01_metrics import review_finding
        oracle = {"reviewed": True, "expected_available": True, "pages": [1],
                  "field_groups": [["distance"]], "unit_groups": [["m"]]}
        r = review_finding({"scope": "A", "quote": "Distance 5 mm"}, {"source": "A", "page": 1}, oracle)
        self.assertFalse(r["unit_alignment"])

    def test_numeric_alignment_does_not_accept_substring_of_decimal(self):
        from evaluation.agent01_metrics import review_finding
        oracle = {"reviewed": True, "expected_available": True, "pages": [1],
                  "field_groups": [["distance"], ["5"]], "unit_groups": [["m"]]}
        r = review_finding({"scope": "A", "quote": "Distance 0.5 m"}, {"source": "A", "page": 1}, oracle)
        self.assertEqual(r["label"], "IRRELEVANT")

    def test_pdf_review_overrides_lexical_label_and_rejects_stale_finding(self):
        from evaluation.agent01_metrics import score_state, finding_fingerprint
        s = self.state("Pump rated pressure: 10 MPa")
        rows = [{"finding_sha256": finding_fingerprint(s.findings[0]), "label": "IRRELEVANT",
                 "field_alignment": False, "unit_alignment": True,
                 "condition_alignment": False, "scope_alignment": True}]
        result = score_state(s, self.oracle(), finding_reviews=rows)
        self.assertFalse(result["relevant_task_success"])
        self.assertEqual(result["review_method"], "offline_pdf_review")
        s.findings[0]["quote"] = "Another quote"
        with self.assertRaises(ValueError):
            score_state(s, self.oracle(), finding_reviews=rows)
