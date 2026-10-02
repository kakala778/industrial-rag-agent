import json
import unittest

from evaluation.run_m8_focused_ocr_comparison import (
    EXPERIMENT_CASES,
    _analyze_output,
    _anonymous_case_record,
)


class FocusedOcrComparisonTests(unittest.TestCase):
    def test_uses_the_fixed_five_ocr_failure_cases_and_target_blocks(self):
        actual = {
            question_id: (case["block_type"], case["block_index"])
            for question_id, case in EXPERIMENT_CASES.items()
        }

        self.assertEqual(
            actual,
            {
                9: ("table", 3),
                10: ("table", 6),
                12: ("table", 18),
                17: ("image", 2),
                20: ("list", 28),
            },
        )

    def test_analyzes_full_fields_and_exact_critical_tokens(self):
        result = _analyze_output(
            "A01; value 0.82 MPa",
            ["A01", "0.82 MPa"],
        )

        self.assertTrue(result["normalized_evidence_hit"])
        self.assertEqual(result["matched_field_count"], 2)
        self.assertEqual(result["expected_field_count"], 2)
        self.assertEqual(result["matched_critical_token_count"], 3)
        self.assertEqual(result["expected_critical_token_count"], 3)
        self.assertTrue(result["strict_evidence_hit"])

    def test_rejects_a_wrong_numeric_token_even_when_other_fields_match(self):
        result = _analyze_output(
            "A01; value 0.32 MPa",
            ["A01", "0.82 MPa"],
        )

        self.assertFalse(result["normalized_evidence_hit"])
        self.assertEqual(result["matched_field_count"], 1)
        self.assertEqual(result["expected_field_count"], 2)
        self.assertEqual(result["matched_critical_token_count"], 2)
        self.assertEqual(result["expected_critical_token_count"], 3)

    def test_critical_tokens_reject_numbers_that_only_match_as_substrings(self):
        result = _analyze_output("166", ["66"])

        # The established evidence matcher intentionally uses substring
        # matching; the separate critical-token metric must expose this case.
        self.assertTrue(result["normalized_evidence_hit"])
        self.assertEqual(result["matched_field_count"], 1)
        self.assertEqual(result["matched_critical_token_count"], 0)
        self.assertEqual(result["critical_token_error_count"], 1)
        self.assertFalse(result["strict_evidence_hit"])

    def test_dimension_like_mixed_identifier_is_checked_as_one_critical_token(self):
        correct = _analyze_output("2L45x4", ["2L45x4"])
        wrong = _analyze_output("2L45x6", ["2L45x4"])

        self.assertEqual(correct["expected_critical_token_count"], 1)
        self.assertEqual(correct["matched_critical_token_count"], 1)
        self.assertTrue(correct["strict_evidence_hit"])
        self.assertFalse(wrong["normalized_evidence_hit"])
        self.assertEqual(wrong["matched_critical_token_count"], 0)

    def test_number_and_attached_unit_are_checked_independently(self):
        result = _analyze_output("0.82MPa", ["0.82 MPa"])

        self.assertEqual(result["expected_critical_token_count"], 2)
        self.assertEqual(result["matched_critical_token_count"], 2)
        self.assertTrue(result["strict_evidence_hit"])

    def test_critical_number_check_preserves_sign(self):
        correct = _analyze_output("-0.82MPa", ["-0.82 MPa"])
        wrong_sign = _analyze_output("0.82MPa", ["-0.82 MPa"])

        self.assertEqual(correct["expected_critical_token_count"], 2)
        self.assertEqual(correct["matched_critical_token_count"], 2)
        self.assertTrue(correct["strict_evidence_hit"])
        self.assertFalse(wrong_sign["normalized_evidence_hit"])
        self.assertEqual(wrong_sign["critical_token_error_count"], 1)

    def test_anonymous_record_excludes_all_document_and_response_text(self):
        case = {
            "question_id": 9,
            "page_number": 20,
            "block_type": "table",
            "block_index": 3,
            "bbox_normalized": [0.1, 0.2, 0.3, 0.4],
            "expected_fields": ["PRIVATE-EVIDENCE-SENTINEL"],
            "question": "PRIVATE-QUESTION-SENTINEL",
            "answer": "PRIVATE-ANSWER-SENTINEL",
            "source": "PRIVATE-DOCUMENT-SENTINEL.pdf",
            "document_text": "PRIVATE-DOCUMENT-TEXT-SENTINEL",
        }
        outputs = {
            "mineru": {"raw_text": "PRIVATE-MINERU-SENTINEL", "matched_field_count": 0},
            "qwen3vl": {"3x": {"raw_text": "PRIVATE-VLM-SENTINEL", "matched_field_count": 1}},
            "rapidocr": {"3x": {"raw_text": "PRIVATE-OCR-SENTINEL", "matched_field_count": 1}},
        }

        serialized = json.dumps(_anonymous_case_record(case, outputs))

        for sentinel in (
            "PRIVATE-EVIDENCE-SENTINEL",
            "PRIVATE-QUESTION-SENTINEL",
            "PRIVATE-ANSWER-SENTINEL",
            "PRIVATE-DOCUMENT-SENTINEL.pdf",
            "PRIVATE-DOCUMENT-TEXT-SENTINEL",
            "PRIVATE-MINERU-SENTINEL",
            "PRIVATE-VLM-SENTINEL",
            "PRIVATE-OCR-SENTINEL",
        ):
            self.assertNotIn(sentinel, serialized)
        self.assertIn('"question_id": 9', serialized)
        self.assertIn('"block_index": 3', serialized)
        self.assertIn('"expected_field_count": 1', serialized)


if __name__ == "__main__":
    unittest.main()
