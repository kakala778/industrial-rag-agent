import unittest

from src.agent.actions import (
    ACTION_JSON_SCHEMA,
    REFERENCE_ACTION_JSON_SCHEMA,
    InvalidAction,
    validate_action,
)


def no_evidence_outcome(scope):
    return {"scope": scope, "status": "no_evidence_found"}


class AgentReferenceActionTests(unittest.TestCase):
    def test_reference_finish_accepts_two_three_and_four_unique_scope_outcomes(self):
        for count in (2, 3, 4):
            outcomes = [no_evidence_outcome(f"S{index}") for index in range(count)]
            action = {"action": "FINISH", "outcomes": outcomes}
            with self.subTest(count=count):
                self.assertEqual(validate_action(action, contract="evidence_reference"), action)

    def test_reference_finish_requires_two_to_four_unique_scope_outcomes(self):
        valid_a = no_evidence_outcome("A")
        valid_b = no_evidence_outcome("B")
        valid_c = no_evidence_outcome("C")
        valid_d = no_evidence_outcome("D")
        for outcomes in ([valid_a], [valid_a, valid_b, valid_c, valid_d,
                                     no_evidence_outcome("E")],
                         [valid_a, no_evidence_outcome("A")]):
            with self.subTest(outcomes=outcomes), self.assertRaises(InvalidAction):
                validate_action({"action": "FINISH", "outcomes": outcomes},
                                contract="evidence_reference")

    def test_evidence_found_requires_one_to_three_unique_nonempty_ids(self):
        valid = {"scope": "A", "status": "evidence_found",
                 "evidence_ids": ["ev_a", "ev_b", "ev_c"]}
        self.assertEqual(validate_action({"action": "FINISH", "outcomes": [
            valid, no_evidence_outcome("B")
        ]}, contract="evidence_reference")["outcomes"][0], valid)
        invalid_rows = [
            {"scope": "A", "status": "evidence_found", "evidence_ids": []},
            {"scope": "A", "status": "evidence_found", "evidence_ids": ["ev_a"] * 4},
            {"scope": "A", "status": "evidence_found", "evidence_ids": ["ev_a", "ev_a"]},
            {"scope": "A", "status": "evidence_found", "evidence_ids": [""]},
            {"scope": "A", "status": "evidence_found", "evidence_ids": ["ev_a"],
             "claim": "unsupported free-form conclusion"},
        ]
        for row in invalid_rows:
            action = {"action": "FINISH", "outcomes": [row, no_evidence_outcome("B")]}
            with self.subTest(row=row), self.assertRaises(InvalidAction):
                validate_action(action, contract="evidence_reference")

    def test_non_evidence_statuses_carry_no_ids_or_extra_fields(self):
        for status in ("no_evidence_found", "insufficient_scope"):
            valid = {"scope": "A", "status": status}
            action = {"action": "FINISH", "outcomes": [valid, no_evidence_outcome("B")]}
            self.assertEqual(validate_action(action, contract="evidence_reference"), action)
            invalid = dict(valid, evidence_ids=["ev_a"])
            with self.subTest(status=status), self.assertRaises(InvalidAction):
                validate_action({"action": "FINISH", "outcomes": [
                    invalid, no_evidence_outcome("B")
                ]}, contract="evidence_reference")

    def test_reference_finish_rejects_unknown_status_and_legacy_claim(self):
        invalid_rows = [
            {"scope": "A", "status": "unknown"},
            {"scope": "A", "status": "supported", "claim": "claim",
             "evidence_ids": ["ev_a"]},
        ]
        for row in invalid_rows:
            with self.subTest(row=row), self.assertRaises(InvalidAction):
                validate_action({"action": "FINISH", "outcomes": [
                    row, no_evidence_outcome("B")
                ]}, contract="evidence_reference")

    def test_reference_schema_has_dynamic_two_to_four_scope_cardinality(self):
        finish = next(row for row in REFERENCE_ACTION_JSON_SCHEMA["oneOf"]
                      if row["properties"]["action"]["const"] == "FINISH")
        outcomes = finish["properties"]["outcomes"]
        self.assertEqual((outcomes["minItems"], outcomes["maxItems"]), (2, 4))

    def test_clarify_question_remains_bounded_and_nonempty(self):
        valid = {"action": "CLARIFY", "question": "请说明设备型号。"}
        self.assertEqual(validate_action(valid, contract="evidence_reference"), valid)
        for question in ("", " " * 501):
            with self.subTest(length=len(question)), self.assertRaises(InvalidAction):
                validate_action({"action": "CLARIFY", "question": question},
                                contract="evidence_reference")

    def test_copied_quote_schema_keeps_its_existing_finish_contract(self):
        finish = next(row for row in ACTION_JSON_SCHEMA["oneOf"]
                      if row["properties"]["action"]["const"] == "FINISH")
        self.assertIn("findings", finish["properties"])
        self.assertNotIn("outcomes", finish["properties"])
        action = {"action": "FINISH", "findings": []}
        self.assertEqual(validate_action(action), action)


if __name__ == "__main__":
    unittest.main()
