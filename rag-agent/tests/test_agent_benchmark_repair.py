import unittest


class RepairedExpectedEvidenceTests(unittest.TestCase):
    def test_audit_root_is_supplied_at_runtime(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from evaluation.rescore_repaired_agent_benchmark import _default_paths

        self.assertIsNone(_default_paths()["audit_report"])
        with TemporaryDirectory() as audit_root:
            paths = _default_paths(audit_root)

        self.assertEqual(paths["audit_report"], Path(audit_root) / "audit-report.local.md")
        self.assertEqual(
            paths["audit_sheet"],
            Path(audit_root) / "human-review-sheet.ai-reviewed.local.csv",
        )

    def _oracle(self, expected_ids, *, conditions=None):
        return {
            "A": {
                "reviewed": True,
                "expected_available": True,
                "expected_evidence_ids": list(expected_ids),
                "pages": [1],
                "field_groups": [["target field"]],
                "unit_groups": [],
                "condition_groups": conditions or [],
            }
        }

    def _state_and_actions(self, selected, rows, scopes=("A",)):
        looked_up = {
            evidence_id: {"evidence_id": evidence_id, "source": scope,
                          "page": 1, "text": text}
            for evidence_id, (scope, text) in rows.items()
        }
        findings = [
            {"scope": scope, "evidence_id": evidence_id,
             "quote": looked_up[evidence_id]["text"]}
            for scope, evidence_id in selected
        ]
        search_history = [
            {"query": "frozen query", "scopes": [scope], "status": "ok",
             "results": [{"evidence_id": evidence_id} for evidence_id, (candidate_scope, _) in rows.items()
                         if candidate_scope == scope]}
            for scope in scopes
        ]
        state = {
            "status": "finished",
            "evidence_ids": list(rows),
            "looked_up_evidence": looked_up,
            "search_history": search_history,
            "findings": findings,
        }
        actions = [{"action": "FINISH", "findings": findings}]
        return state, actions

    def test_irrelevant_expected_id_is_not_counted_as_correct_selection(self):
        from evaluation.agent02_metrics import score_selection

        oracle = self._oracle(["ev_wrong"])
        state, actions = self._state_and_actions(
            [("A", "ev_wrong")], {"ev_wrong": ("A", "unrelated evidence")})
        reviews = {"ev_wrong": {"scope": "A", "label": "IRRELEVANT"}}

        result = score_selection(state, actions, oracle, reviews)

        self.assertEqual(result["group"], "RETRIEVAL_BOUND")
        self.assertFalse(result["id_selection_success"])
        self.assertFalse(result["findings"][0]["correct_expected_id"])

    def test_expected_id_missing_from_frozen_candidates_stays_retrieval_bound(self):
        from evaluation.agent02_metrics import candidate_group, score_selection

        oracle = {
            "A": {"reviewed": True, "expected_available": True,
                  "expected_evidence_ids": ["ev_target_a"]},
            "B": {"reviewed": True, "expected_available": True,
                  "expected_evidence_ids": ["ev_target_b"]},
        }
        state, actions = self._state_and_actions(
            [("A", "ev_other_a"), ("B", "ev_target_b")],
            {"ev_other_a": ("A", "other relevant-looking evidence"),
             "ev_target_b": ("B", "target field")}, scopes=("A", "B"))
        reviews = {
            "ev_other_a": {"scope": "A", "label": "RELEVANT"},
            "ev_target_b": {"scope": "B", "label": "RELEVANT"},
        }

        self.assertEqual(candidate_group(oracle, reviews), "RETRIEVAL_BOUND")
        result = score_selection(state, actions, oracle, reviews)
        self.assertEqual(result["group"], "RETRIEVAL_BOUND")
        self.assertFalse(result["id_selection_success"])

    def test_negated_applicability_does_not_match_affirmative_condition(self):
        from evaluation.agent01_metrics import review_finding

        oracle = {
            "reviewed": True,
            "expected_available": True,
            "pages": [1],
            "field_groups": [["25"], ["铜线"]],
            "unit_groups": [["mm2"]],
            "condition_groups": [["不在同一水平面"], ["30*3mm紫铜排", "无法连接"]],
        }
        evidence = {"source": "A", "page": 1}
        affirmative = review_finding(
            {"scope": "A", "quote": "target field 25 mm2 铜线在同一水平面，30*3mm紫铜排无法连接"},
            evidence, oracle)
        negative = review_finding(
            {"scope": "A", "quote": "target field 25 mm2 铜线不在同一水平面，30*3mm紫铜排无法连接"},
            evidence, oracle)

        self.assertFalse(affirmative["condition_alignment"])
        self.assertEqual(affirmative["label"], "PARTIAL")
        self.assertTrue(negative["condition_alignment"])
        self.assertEqual(negative["label"], "RELEVANT")

    def test_partial_evidence_is_not_a_complete_expected_id(self):
        from evaluation.agent02_metrics import score_selection

        oracle = self._oracle(["ev_full"])
        state, actions = self._state_and_actions(
            [("A", "ev_partial")],
            {"ev_full": ("A", "target field"),
             "ev_partial": ("A", "target field, object context is truncated")})
        reviews = {
            "ev_full": {"scope": "A", "label": "RELEVANT"},
            "ev_partial": {"scope": "A", "label": "PARTIAL"},
        }

        result = score_selection(state, actions, oracle, reviews)

        self.assertEqual(result["group"], "CANDIDATE_AVAILABLE")
        self.assertFalse(result["id_selection_success"])
        self.assertFalse(result["findings"][0]["correct_expected_id"])
        self.assertEqual(result["findings"][0]["id_label"], "PARTIAL")

    def test_unlisted_evidence_on_unsupported_scope_is_not_a_correct_expected_id(self):
        from evaluation.agent02_metrics import score_selection

        oracle = {
            "A": {"reviewed": True, "expected_available": True,
                  "expected_evidence_ids": ["ev_target"], "pages": [1],
                  "field_groups": [["target field"]], "unit_groups": [],
                  "condition_groups": []},
            "B": {"reviewed": True, "expected_available": False,
                  "pages": [1], "field_groups": [], "unit_groups": [],
                  "condition_groups": []},
        }
        state, actions = self._state_and_actions(
            [("A", "ev_target"), ("B", "ev_unsupported")],
            {"ev_target": ("A", "target field"),
             "ev_unsupported": ("B", "other field")}, scopes=("A", "B"))
        reviews = {
            "ev_target": {"scope": "A", "label": "RELEVANT"},
            "ev_unsupported": {"scope": "B", "label": "RELEVANT"},
        }

        result = score_selection(state, actions, oracle, reviews)

        self.assertFalse(result["findings"][1]["correct_expected_id"])
        self.assertFalse(result["id_selection_success"])

    def test_agent03_correct_id_count_excludes_unlisted_unsupported_scope(self):
        from evaluation.agent03_metrics import score_reference

        task = {"scopes": ["A", "B"], "oracle": {
            "A": {"reviewed": True, "expected_available": True,
                  "expected_evidence_ids": ["ev_target"], "pages": [1],
                  "field_groups": [["target field"]], "unit_groups": [],
                  "condition_groups": []},
            "B": {"reviewed": True, "expected_available": False,
                  "pages": [1], "field_groups": [], "unit_groups": [],
                  "condition_groups": []},
        }}
        state = {
            "status": "finished",
            "search_history": [
                {"scopes": ["A"], "status": "ok", "results": [{"evidence_id": "ev_target"}]},
                {"scopes": ["B"], "status": "ok", "results": [{"evidence_id": "ev_unsupported"}]},
            ],
            "looked_up_evidence": {
                "ev_target": {"source": "A", "page": 1, "text": "target field"},
                "ev_unsupported": {"source": "B", "page": 1, "text": "other field"},
            },
            "findings": [
                {"scope": "A", "status": "supported", "evidence": [{
                    "evidence_id": "ev_target", "excerpt": "target field",
                    "span": {"coordinate": "child", "length": 12}}]},
                {"scope": "B", "status": "supported", "evidence": [{
                    "evidence_id": "ev_unsupported", "excerpt": "other field",
                    "span": {"coordinate": "child", "length": 11}}]},
            ],
        }
        outcomes = [
            {"scope": "A", "status": "supported", "evidence_ids": ["ev_target"]},
            {"scope": "B", "status": "supported", "evidence_ids": ["ev_unsupported"]},
        ]
        reviews = {
            "ev_target": {"scope": "A", "label": "RELEVANT"},
            "ev_unsupported": {"scope": "B", "label": "RELEVANT"},
        }

        result = score_reference(state, outcomes, task, reviews)

        self.assertEqual(result["correct_evidence_ids"], 1)
        self.assertEqual(result["unsupported_side_correct"], {"passed": 0, "assessed": 1})

    def test_agent03_host_reference_does_not_accept_a_partial_id_as_complete(self):
        from evaluation.agent03_metrics import score_reference

        task = {
            "scopes": ["A"],
            "oracle": self._oracle(["ev_full"]),
        }
        state = {
            "status": "finished",
            "search_history": [{"scopes": ["A"], "status": "ok", "results": [
                {"evidence_id": "ev_full"}, {"evidence_id": "ev_partial"}]}],
            "looked_up_evidence": {
                "ev_full": {"source": "A", "page": 1, "text": "target field"},
                "ev_partial": {"source": "A", "page": 1,
                               "text": "target field with truncated object context"},
            },
            "findings": [{"scope": "A", "status": "supported", "evidence": [{
                "evidence_id": "ev_partial", "excerpt": "target field with truncated object context",
                "span": {"coordinate": "child", "length": 41},
            }]}],
        }
        outcomes = [{"scope": "A", "status": "supported", "evidence_ids": ["ev_partial"]}]
        reviews = {
            "ev_full": {"scope": "A", "label": "RELEVANT"},
            "ev_partial": {"scope": "A", "label": "PARTIAL"},
        }

        result = score_reference(state, outcomes, task, reviews)

        self.assertEqual(result["candidate_group"], "CANDIDATE_AVAILABLE")
        self.assertFalse(result["evidence_id_selection_success"])
        self.assertFalse(result["fully_relevant_task_success"])
        self.assertEqual(result["correct_evidence_ids"], 0)
        self.assertEqual(result["citation_authenticity"]["status"], "not_recomputed")

    def test_agent03_expected_id_absent_from_saved_search_is_retrieval_bound(self):
        from evaluation.agent03_metrics import candidate_group

        task = {"scopes": ["A"], "oracle": self._oracle(["ev_target"])}
        reviews = {"ev_other": {"scope": "A", "label": "RELEVANT"}}

        self.assertEqual(candidate_group(task, reviews, {"ev_other"}), "RETRIEVAL_BOUND")

    def test_repaired_rescore_is_deterministic_for_saved_actions(self):
        from evaluation.rescore_repaired_agent_benchmark import rescore_records

        task = {"id": "R01", "scopes": ["A"], "oracle": self._oracle(["ev_a"])}
        reviews = {"tasks": {"R01": {"ev_a": {"scope": "A", "label": "RELEVANT"}}}}
        state, actions = self._state_and_actions(
            [("A", "ev_a")], {"ev_a": ("A", "target field" )})
        agent02 = {"arms": {"B": {"R01": {"state": state, "actions": actions}}}}
        reference = {"evidence_id": "ev_a", "excerpt": "target field",
                     "span": {"coordinate": "child", "length": len("target field")}}
        state03 = dict(state, findings=[{"scope": "A", "status": "supported",
                                         "evidence": [reference]}])
        agent03 = {"arms": {"B": {"R01": {
            "state": state03,
            "actions": [{"action": "FINISH", "outcomes": [
                {"scope": "A", "status": "supported", "evidence_ids": ["ev_a"]}]}],
        }}}}

        first = rescore_records([task], reviews, agent02, agent03)
        second = rescore_records([task], reviews, agent02, agent03)

        self.assertEqual(first, second)
        self.assertTrue(first["agent02"]["arms"]["B"]["R01"]["id_selection_success"])
        self.assertTrue(first["agent03"]["tasks"]["R01"]["evidence_id_selection_success"])

    def test_repair_builder_changes_only_derived_inputs_and_keeps_review_provenance(self):
        from copy import deepcopy
        from evaluation.rescore_repaired_agent_benchmark import apply_repair_manifest

        manifest = {"tasks": [{"id": "R02", "scopes": ["A", "B"], "oracle": {
            "A": {"expected_available": False}, "B": {"expected_available": True}}}]}
        reviews = {"authority": "old authority", "manifest_sha256": "old-manifest",
                   "observations_sha256": "frozen-observations", "tasks": {"R02": {
                       "ev_bad": {"scope": "B", "label": "RELEVANT", "lookup_sha256": "looked"},
                       "ev_good": {"scope": "B", "label": "RELEVANT", "lookup_sha256": "also-looked"}}}}
        original_manifest, original_reviews = deepcopy(manifest), deepcopy(reviews)
        sheet_rows = [
            {"task_id": "R02", "scope": "A", "expected_evidence_ids": "",
             "original_review_status": "pending_human_pdf_review", "ai_review_confidence": "HIGH"},
            {"task_id": "R02", "scope": "B", "expected_evidence_ids": "ev_bad;ev_good",
             "original_review_status": "pending_human_pdf_review", "ai_review_confidence": "HIGH"},
        ]
        repair = {
            "review_authority": "ai_assisted_original_pdf_review",
            "expected_id_changes": [{"task_id": "R02", "scope": "B",
                                     "before": ["ev_bad", "ev_good"], "after": ["ev_good"]}],
            "oracle_changes": [],
            "candidate_review_changes": [{"task_id": "R02", "evidence_id": "ev_bad",
                                          "before_label": "RELEVANT",
                                          "after": {"label": "IRRELEVANT",
                                                    "field_alignment": False,
                                                    "scope_alignment": True}}],
        }

        repaired_manifest, repaired_reviews = apply_repair_manifest(
            manifest, sheet_rows, reviews, repair)

        self.assertEqual(manifest, original_manifest)
        self.assertEqual(reviews, original_reviews)
        self.assertEqual(repaired_manifest["tasks"][0]["oracle"]["B"]["expected_evidence_ids"],
                         ["ev_good"])
        self.assertEqual(repaired_manifest["tasks"][0]["oracle"]["B"]["review_status"],
                         "ai_pdf_reviewed")
        self.assertEqual(repaired_manifest["tasks"][0]["oracle"]["B"]["source_review_status"],
                         "pending_human_pdf_review")
        self.assertEqual(repaired_reviews["tasks"]["R02"]["ev_bad"]["label"], "IRRELEVANT")
        self.assertEqual(repaired_reviews["tasks"]["R02"]["ev_bad"]["lookup_sha256"], "looked")
        self.assertEqual(repaired_reviews["authority"], "ai_assisted_original_pdf_review")


if __name__ == "__main__":
    unittest.main()
