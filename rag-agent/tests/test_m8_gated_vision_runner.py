import unittest

from evaluation.run_m8_gated_vision_retrieval_ab import (
    _is_matching_m8_5_summary,
    _validate_fixed_positive_cases,
)


class GatedVisionRunnerTests(unittest.TestCase):
    def test_accepts_the_existing_m85_summary_schema_for_the_fixed_cohort(self):
        summary = {
            "vision_model": {"tag": "qwen3-vl:2b-instruct-q4_K_M"},
            "primary_scale": "3x",
            "cases": [
                {"question_id": 9},
                {"question_id": 10},
                {"question_id": 12},
                {"question_id": 17},
                {"question_id": 20},
            ],
        }

        self.assertTrue(_is_matching_m8_5_summary(summary))

    def test_rejects_the_m83_case_set_when_m85_fixed_positives_are_required(self):
        wrong_m83_cases = [{"question_id": qid} for qid in (10, 16, 29, 32)]

        with self.assertRaises(ValueError):
            _validate_fixed_positive_cases(wrong_m83_cases)

    def test_accepts_only_the_fixed_m85_positive_ids(self):
        fixed_cases = [{"question_id": qid} for qid in (9, 10, 12, 17, 20)]

        self.assertEqual(_validate_fixed_positive_cases(fixed_cases), fixed_cases)

    def test_rejects_a_different_model_or_case_cohort(self):
        wrong_model = {
            "vision_model": {"tag": "other-model"},
            "primary_scale": "3x",
            "cases": [{"question_id": qid} for qid in (9, 10, 12, 17, 20)],
        }
        wrong_cohort = {
            "vision_model": {"tag": "qwen3-vl:2b-instruct-q4_K_M"},
            "primary_scale": "3x",
            "cases": [{"question_id": qid} for qid in (9, 10, 12, 17, 21)],
        }

        self.assertFalse(_is_matching_m8_5_summary(wrong_model))
        self.assertFalse(_is_matching_m8_5_summary(wrong_cohort))


if __name__ == "__main__":
    unittest.main()
