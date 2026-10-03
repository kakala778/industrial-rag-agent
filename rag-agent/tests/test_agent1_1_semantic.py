import json
import os
import unittest
from unittest.mock import patch

from src.agent.semantic import DeepSeekSemanticComparator, DIMENSION_KEYS
from src.agent.semantic_contract_v2 import (
    InvalidSemanticOutputV2,
    parse_semantic_output_v2,
    semantic_messages_v2,
    validate_semantic_output_v2,
)


def result(verdict="EQUIVALENT", basis=None, **overrides):
    dimensions = {key: "aligned" for key in DIMENSION_KEYS}
    if verdict == "DIFFERENT":
        dimensions["value"] = "different"
    elif verdict == "NOT_COMPARABLE":
        dimensions["object_or_field"] = "different"
    if basis is None:
        basis = ("different_object_or_field" if verdict == "NOT_COMPARABLE"
                 else "comparable_fact")
    dimensions.update(overrides)
    return {
        "verdict": verdict,
        "comparability_basis": basis,
        "dimensions": dimensions,
        "reason": "The requested facts have been compared.",
        "notes": "",
    }


class Agent11SemanticContractTests(unittest.TestCase):
    def test_object_mismatch_keeps_observed_value_and_unit(self):
        value = result(
            "NOT_COMPARABLE",
            object_or_field="different",
            value="aligned",
            unit="aligned",
            condition_or_applicability="not_applicable",
        )
        self.assertEqual(validate_semantic_output_v2(value), value)

    def test_condition_difference_can_be_a_comparable_difference(self):
        value = result(
            "DIFFERENT",
            value="different",
            condition_or_applicability="different",
        )
        self.assertEqual(validate_semantic_output_v2(value), value)

    def test_incompatible_installation_scenario_is_not_comparable(self):
        value = result(
            "NOT_COMPARABLE",
            basis="incompatible_scope_or_installation_scenario",
            object_or_field="aligned",
            value="aligned",
            unit="aligned",
            condition_or_applicability="different",
        )
        self.assertEqual(validate_semantic_output_v2(value), value)

    def test_uncertain_dimension_is_a_valid_abstention_not_a_guessed_verdict(self):
        value = result(
            "NOT_COMPARABLE",
            basis="uncertain",
            object_or_field="aligned",
            value="uncertain",
            unit="aligned",
            condition_or_applicability="aligned",
        )
        self.assertEqual(validate_semantic_output_v2(value), value)

    def test_rejects_verdict_dimension_and_basis_inconsistencies(self):
        invalid = (
            result("EQUIVALENT", value="different"),
            result("DIFFERENT", value="aligned"),
            result("NOT_COMPARABLE", basis="different_object_or_field",
                   object_or_field="aligned"),
            result("NOT_COMPARABLE", basis="incompatible_scope_or_installation_scenario",
                   object_or_field="aligned", condition_or_applicability="aligned"),
            result("EQUIVALENT", basis="different_object_or_field",
                   object_or_field="different"),
            result("NOT_COMPARABLE", basis="uncertain", object_or_field="aligned"),
        )
        for value in invalid:
            with self.subTest(verdict=value["verdict"], basis=value["comparability_basis"]):
                with self.assertRaises(InvalidSemanticOutputV2):
                    validate_semantic_output_v2(value)

    def test_rejects_extra_fields_bad_labels_and_duplicate_json_keys(self):
        value = result()
        value["citations"] = []
        with self.assertRaises(InvalidSemanticOutputV2):
            validate_semantic_output_v2(value)

        value = result()
        value["dimensions"]["unit"] = "converted"
        with self.assertRaises(InvalidSemanticOutputV2):
            validate_semantic_output_v2(value)

        with self.assertRaises(InvalidSemanticOutputV2):
            parse_semantic_output_v2('{"verdict":"EQUIVALENT","verdict":"DIFFERENT"}')

    def test_prompt_requires_independent_dimensions_before_verdict_and_limits_na(self):
        messages = semantic_messages_v2(
            "Compare the equipment configuration in the named A and B scopes.",
            ["Scope A evidence"],
            ["Scope B evidence"],
        )
        prompt = "\n".join(row["content"] for row in messages).lower()
        self.assertIn("independently classify all four dimensions first", prompt)
        self.assertIn("object/field mismatch is never a reason", prompt)
        self.assertIn("explicitly named a/b scopes are the intended comparison axis", prompt)
        self.assertIn("not_applicable only when that dimension itself is absent", prompt)
        self.assertNotIn("evidence_id", prompt)
        self.assertNotIn("ev_", prompt)

    def test_v2_prompt_and_parser_can_be_injected_without_changing_transport_settings(self):
        calls = []
        expected = result(
            "NOT_COMPARABLE", object_or_field="different",
            value="aligned", unit="aligned",
            condition_or_applicability="not_applicable",
        )

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit=-1):
                return json.dumps({
                    "choices": [{"message": {"content": json.dumps(expected)},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                              "total_tokens": 15},
                }).encode("utf-8")

        def builder(request, a, b):
            return [{"role": "system", "content": "v2 prompt"},
                    {"role": "user", "content": request + a[0] + b[0]}]

        def transport(request, timeout):
            calls.append((request, timeout))
            return Response()

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-secret"}):
            client = DeepSeekSemanticComparator(
                transport=transport,
                message_builder=builder,
                output_parser=parse_semantic_output_v2,
            )
            actual = client.compare("compare", ["A"], ["B"])

        self.assertEqual(actual, expected)
        self.assertEqual(len(calls), 1)
        payload = json.loads(calls[0][0].data.decode("utf-8"))
        self.assertEqual(payload["model"], "deepseek-flash")
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["messages"][0]["content"], "v2 prompt")
        self.assertNotIn("test-secret", calls[0][0].data.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
