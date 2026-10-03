import json
import os
import unittest
from decimal import Decimal
from unittest.mock import patch

from src.agent.semantic import (
    DeepSeekSemanticComparator,
    DIMENSION_KEYS,
    InvalidSemanticOutput,
    MODEL_VERDICTS,
    SemanticAPIError,
    SemanticCostBudget,
    normalize_decimal,
    numeric_values_equal,
    parse_semantic_output,
    semantic_messages,
    validate_semantic_output,
)


def semantic_result(verdict="NOT_COMPARABLE", **dimension_overrides):
    dimensions = {key: "aligned" for key in DIMENSION_KEYS}
    if verdict == "NOT_COMPARABLE":
        dimensions["object_or_field"] = "different"
    elif verdict == "DIFFERENT":
        dimensions["value"] = "different"
    dimensions.update(dimension_overrides)
    return {
        "verdict": verdict,
        "dimensions": dimensions,
        "reason": "The evidence refers to different engineering facts.",
        "notes": "",
    }


class SemanticOutputContractTests(unittest.TestCase):
    def test_accepts_only_the_three_model_verdicts(self):
        for verdict in MODEL_VERDICTS:
            with self.subTest(verdict=verdict):
                self.assertEqual(parse_semantic_output(json.dumps(semantic_result(verdict))),
                                 semantic_result(verdict))

    def test_host_only_insufficient_evidence_is_not_a_model_verdict(self):
        with self.assertRaises(InvalidSemanticOutput):
            validate_semantic_output(semantic_result("INSUFFICIENT_EVIDENCE"))

    def test_rejects_verdicts_that_contradict_the_dimension_labels(self):
        inconsistent = (
            semantic_result("EQUIVALENT", object_or_field="different"),
            semantic_result("EQUIVALENT", value="different"),
            semantic_result("EQUIVALENT", condition_or_applicability="different"),
            semantic_result("DIFFERENT", value="aligned"),
            semantic_result("DIFFERENT", object_or_field="different", value="different"),
            semantic_result("NOT_COMPARABLE", object_or_field="aligned"),
        )
        for value in inconsistent:
            with self.subTest(verdict=value["verdict"], dimensions=value["dimensions"]):
                with self.assertRaises(InvalidSemanticOutput):
                    validate_semantic_output(value)

    def test_accepts_comparable_verdicts_with_consistent_independent_dimensions(self):
        equivalent = semantic_result(
            "EQUIVALENT", condition_or_applicability="not_applicable")
        different_value = semantic_result("DIFFERENT", value="different")
        different_condition = semantic_result(
            "DIFFERENT", condition_or_applicability="different")
        not_comparable = semantic_result("NOT_COMPARABLE", object_or_field="different")
        not_comparable_condition = semantic_result(
            "NOT_COMPARABLE", object_or_field="aligned",
            condition_or_applicability="different")
        for value in (equivalent, different_value, different_condition,
                      not_comparable, not_comparable_condition):
            with self.subTest(verdict=value["verdict"], dimensions=value["dimensions"]):
                self.assertEqual(validate_semantic_output(value), value)

    def test_rejects_extra_fields_and_model_supplied_citations(self):
        value = semantic_result()
        value["evidence_ids"] = ["ev_forged"]
        with self.assertRaises(InvalidSemanticOutput):
            validate_semantic_output(value)

    def test_rejects_extra_dimension_and_invalid_dimension_label(self):
        value = semantic_result()
        value["dimensions"]["operator"] = "aligned"
        with self.assertRaises(InvalidSemanticOutput):
            validate_semantic_output(value)

        value = semantic_result()
        value["dimensions"]["unit"] = "converted"
        with self.assertRaises(InvalidSemanticOutput):
            validate_semantic_output(value)

    def test_rejects_invalid_json_duplicate_keys_and_unbounded_text(self):
        with self.assertRaises(InvalidSemanticOutput):
            parse_semantic_output("not JSON")

        duplicate = '{"verdict":"EQUIVALENT","verdict":"DIFFERENT"}'
        with self.assertRaises(InvalidSemanticOutput):
            parse_semantic_output(duplicate)

        value = semantic_result()
        value["reason"] = "x" * 1001
        with self.assertRaises(InvalidSemanticOutput):
            validate_semantic_output(value)

    def test_normalizes_only_plain_decimal_numbers_and_preserves_sign(self):
        self.assertEqual(normalize_decimal("5.0"), Decimal("5"))
        self.assertTrue(numeric_values_equal("5.0", "5"))
        self.assertTrue(numeric_values_equal("-5.00", "-5"))
        self.assertFalse(numeric_values_equal("-5", "5"))
        self.assertFalse(numeric_values_equal("-0", "0"))
        for value in (">=5", "≤5", "5 m", "5,000", "1e3", " 5 ", ""):
            with self.subTest(value=value):
                self.assertIsNone(normalize_decimal(value))

    def test_same_numeric_value_does_not_override_object_mismatch(self):
        result = semantic_result(
            "NOT_COMPARABLE",
            object_or_field="different",
            value="aligned",
            unit="aligned",
        )
        parsed = validate_semantic_output(result)
        self.assertEqual(parsed["verdict"], "NOT_COMPARABLE")
        self.assertEqual(parsed["dimensions"]["value"], "aligned")

    def test_dimension_keys_are_exact_and_stable(self):
        self.assertEqual(
            DIMENSION_KEYS,
            ("object_or_field", "value", "unit", "condition_or_applicability"),
        )

    def test_prompt_compares_only_excerpts_and_never_sends_evidence_ids(self):
        messages = semantic_messages("Compare the stated facts.", ["A accepted excerpt"],
                                     ["B accepted excerpt"])
        self.assertEqual([row["role"] for row in messages], ["system", "user"])
        prompt = "\n".join(row["content"] for row in messages)
        self.assertIn("same value or unit does not make different objects comparable", prompt)
        self.assertIn("condition_or_applicability", prompt)
        self.assertNotIn("evidence_id", prompt)
        self.assertNotIn("ev_", prompt)

    def test_prompt_rejects_unbounded_or_empty_excerpt_inputs(self):
        with self.assertRaises(ValueError):
            semantic_messages("request", [], ["B"])
        with self.assertRaises(ValueError):
            semantic_messages("request", ["x" * 1201], ["B"])


class SemanticAPIClientTests(unittest.TestCase):
    class Response:
        def __init__(self, value):
            self.body = json.dumps(value).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit=-1):
            return self.body

    def test_sends_one_nonthinking_json_request_without_ids(self):
        calls = []
        result = semantic_result(
            "NOT_COMPARABLE", object_or_field="different", value="aligned", unit="aligned",
        )

        def transport(request, timeout):
            calls.append((request, timeout))
            return self.Response({
                "choices": [{"message": {"content": json.dumps(result)},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 40,
                          "total_tokens": 60},
            })

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-secret"}):
            client = DeepSeekSemanticComparator(transport=transport)
            actual = client.compare("Compare these facts.", ["A accepted text"], ["B accepted text"])

        self.assertEqual(actual["verdict"], "NOT_COMPARABLE")
        self.assertEqual(len(calls), 1)
        request, timeout = calls[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "deepseek-flash")
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertNotIn("evidence_id", json.dumps(payload))
        self.assertNotIn("test-secret", request.data.decode("utf-8"))
        self.assertGreater(timeout, 0)
        self.assertEqual(client.events[0]["usage"]["prompt_tokens"], 20)

    def test_invalid_model_json_is_not_retried(self):
        calls = []

        def transport(_request, timeout):
            calls.append(True)
            return self.Response({
                "choices": [{"message": {"content": "not JSON"},
                             "finish_reason": "stop"}],
                "usage": {},
            })

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-secret"}):
            client = DeepSeekSemanticComparator(transport=transport)
            with self.assertRaises(InvalidSemanticOutput):
                client.compare("request", ["A"], ["B"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(client.events[0]["error"], "invalid_output")


class SemanticCostBudgetTests(unittest.TestCase):
    def test_reserves_at_peak_rates_then_settles_metered_usage(self):
        budget = SemanticCostBudget(soft_limit_rmb=3.0, hard_limit_rmb=5.0)
        reservation = budget.reserve(1000, 100)
        self.assertGreater(reservation, 0)
        actual = budget.settle(reservation, {
            "prompt_tokens": 10, "completion_tokens": 5,
            "prompt_cache_hit_tokens": 2, "prompt_cache_miss_tokens": 8,
        })
        self.assertAlmostEqual(actual, (2 * 0.04 + 8 * 2.0 + 5 * 8.0) / 1_000_000)
        self.assertAlmostEqual(budget.reserved_rmb, actual)

    def test_cost_ceiling_is_enforced_before_the_request(self):
        budget = SemanticCostBudget(soft_limit_rmb=0.001, hard_limit_rmb=0.002)
        with self.assertRaises(SemanticAPIError):
            budget.reserve(10_000, 512)
        self.assertEqual(budget.reserved_rmb, 0)
        with self.assertRaises(ValueError):
            SemanticCostBudget(soft_limit_rmb=3.01, hard_limit_rmb=5.0)
        with self.assertRaises(ValueError):
            SemanticCostBudget(soft_limit_rmb=3.0, hard_limit_rmb=5.01)


if __name__ == "__main__":
    unittest.main()
