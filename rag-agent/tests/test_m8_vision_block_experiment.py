import unittest

from evaluation.run_m8_vision_block_experiment import (
    _analyze_vision_output,
    _anonymous_case_record,
    _normalized_bbox_to_rect,
)


class VisionOutputAnalysisTests(unittest.TestCase):
    def test_matches_expected_fields_and_marks_document_candidate(self):
        output = """## Text
Visible label
## Entities
Valve
## Parameters
Pressure 20 MPa
## Table
Row A | 20 MPa
## Relations
Label points to valve
## Uncertainty
None
"""

        analysis = _analyze_vision_output(
            output,
            ["Pressure", "20 MPa"],
            "OCR_FAILURE",
        )

        self.assertTrue(analysis["vision_expected_evidence_hit"])
        self.assertEqual(analysis["matched_expected_field_count"], 2)
        self.assertEqual(analysis["expected_field_count"], 2)
        self.assertTrue(analysis["structured_document_candidate"])

    def test_anonymous_record_does_not_copy_private_input_or_output(self):
        secret_question = "private question sentinel"
        secret_answer = "private answer sentinel"
        secret_source = "private-document-name.pdf"
        secret_output = "private model response sentinel"
        case = {
            "question_id": 29,
            "category": "IMAGE_INFORMATION_LOSS",
            "page_number": 18,
            "expected_keywords": [secret_answer],
            "question": secret_question,
            "answer": secret_answer,
            "source": secret_source,
            "mineru_case": {
                "evidence_in_representation": False,
                "reranker_top3_page_hit": True,
                "reranker_normalized_evidence_hit": False,
                "document_text": secret_answer,
            },
        }
        analysis = _analyze_vision_output(
            secret_output,
            case["expected_keywords"],
            case["category"],
        )

        record = _anonymous_case_record(
            case,
            "block_crop",
            analysis,
            block_type="image",
            block_index=0,
            context_block_count=2,
        )
        serialized = repr(record)

        for secret in (
            secret_question,
            secret_answer,
            secret_source,
            secret_output,
        ):
            self.assertNotIn(secret, serialized)
        self.assertEqual(record["question_id"], 29)
        self.assertEqual(record["block_type"], "image")
        self.assertEqual(record["block_index"], 0)
        self.assertEqual(record["context_block_count"], 2)

    def test_normalized_bbox_maps_to_page_coordinates(self):
        rect = _normalized_bbox_to_rect(
            [0.1, 0.2, 0.4, 0.6],
            (0.0, 0.0, 1000.0, 2000.0),
        )

        self.assertEqual(tuple(rect), (100.0, 400.0, 400.0, 1200.0))


if __name__ == "__main__":
    unittest.main()
