"""Tests for the privacy-safe M8.3 feasibility summary."""

import json
import unittest

from evaluation.run_m8_vision_feasibility import analyze_case, build_argument_parser


class M8VisionFeasibilityTests(unittest.TestCase):
    def test_runner_has_a_safe_help_only_cli(self):
        args = build_argument_parser().parse_args([])
        self.assertEqual(vars(args), {})

    def test_analyzer_records_recovery_without_serializing_model_text_or_evidence(self):
        secret_evidence = "PRIVATE-TAG-8142"
        secret_output = (
            "## 页面摘要\nPRIVATE summary\n"
            "## 文本信息\nPRIVATE-TAG-8142\n"
            "## 参数信息\n不确定\n"
            "## 表格信息\n无\n"
            "## 图形/布局关系\nPRIVATE relationship\n"
            "## 不确定信息\n无"
        )
        result = analyze_case(
            question_id=16,
            category="IMAGE_INFORMATION_LOSS",
            mineru_case={
                "evidence_in_representation": False,
                "reranker_top3_page_hit": False,
                "reranker_normalized_evidence_hit": False,
            },
            expected_keywords=[secret_evidence],
            vision_output=secret_output,
        )
        serialized = json.dumps(result, ensure_ascii=False)

        self.assertTrue(result["vision_expected_evidence_hit"])
        self.assertTrue(result["vision_added_evidence"])
        self.assertTrue(result["vision_relation_section_nonempty"])
        self.assertTrue(result["candidate_document_block"])
        self.assertNotIn(secret_evidence, serialized)
        self.assertNotIn("PRIVATE", serialized)

    def test_analyzer_does_not_claim_a_recovery_when_evidence_is_absent(self):
        result = analyze_case(
            question_id=32,
            category="LAYOUT_RELATION_FAILURE",
            mineru_case={
                "evidence_in_representation": False,
                "reranker_top3_page_hit": True,
                "reranker_normalized_evidence_hit": False,
            },
            expected_keywords=["PRIVATE-RELATION-2391"],
            vision_output=(
                "## 页面摘要\n摘要\n## 文本信息\n无\n"
                "## 参数信息\n无\n## 表格信息\n无\n"
                "## 图形/布局关系\n未发现\n## 不确定信息\n无"
            ),
        )

        self.assertFalse(result["vision_expected_evidence_hit"])
        self.assertFalse(result["vision_added_evidence"])
        self.assertFalse(result["candidate_document_block"])
        self.assertFalse(result["vision_relation_section_nonempty"])
        self.assertFalse(result["current_retrieval_evidence_hit"])
        self.assertTrue(result["current_retrieval_page_hit"])

    def test_ocr_candidate_needs_target_evidence_inside_a_table_section(self):
        result = analyze_case(
            question_id=10,
            category="OCR_FAILURE",
            mineru_case={
                "evidence_in_representation": False,
                "reranker_top3_page_hit": False,
                "reranker_normalized_evidence_hit": False,
            },
            expected_keywords=["PRIVATE-CELL-VALUE-6821"],
            vision_output=(
                "## 页面摘要\n摘要\n## 文本信息\nPRIVATE-CELL-VALUE-6821\n"
                "## 参数信息\n无\n## 表格信息\n无\n"
                "## 图形/布局关系\n无\n## 不确定信息\n无"
            ),
        )

        self.assertTrue(result["vision_expected_evidence_hit"])
        self.assertFalse(result["candidate_document_block"])


if __name__ == "__main__":
    unittest.main()
