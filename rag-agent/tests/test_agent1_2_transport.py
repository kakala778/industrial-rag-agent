"""Tests for the isolated Agent 1.2 Responses/json_schema transport."""

import json
import importlib
import importlib.util
import io
import os
import unittest
from urllib.error import HTTPError
from unittest.mock import patch

from src.agent import semantic_contract_v2 as contract
from src.agent.semantic_contract_v2 import InvalidSemanticOutputV2
from src.agent.semantic import SemanticAPIError


def _result(*, verdict="EQUIVALENT", basis="comparable_fact", dimensions=None):
    labels = {
        "object_or_field": "aligned",
        "value": "aligned",
        "unit": "aligned",
        "condition_or_applicability": "aligned",
    }
    if dimensions:
        labels.update(dimensions)
    return {
        "verdict": verdict,
        "comparability_basis": basis,
        "dimensions": labels,
        "reason": "The supplied facts match.",
        "notes": "",
    }


def _response_body(output_text, *, status="completed", usage=None, output=None,
                   response_id="resp_agent12"):
    if output is None:
        output = [{
            "type": "message",
            "id": "msg_agent12",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": output_text,
                         "annotations": []}],
        }]
    return {
        "id": response_id,
        "object": "response",
        "created_at": 1791050400,
        "status": status,
        "model": "deepseek-flash",
        "output": output,
        "usage": usage or {
            "input_tokens": 42,
            "input_tokens_details": {"cached_tokens": 2},
            "output_tokens": 31,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 73,
        },
        "store": False,
        "error": None,
        "incomplete_details": None,
    }


class _Response:
    def __init__(self, body, *, status=200, headers=None):
        self.body = body if isinstance(body, bytes) else json.dumps(
            body, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        self.status = status
        self.headers = headers or {"x-request-id": "req_agent12"}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit=-1):
        return self.body


class Agent12JsonSchemaTests(unittest.TestCase):
    def _schema_builder(self):
        builder = getattr(contract, "semantic_output_json_schema_v2", None)
        self.assertTrue(callable(builder), "Agent 1.2 JSON Schema generator is missing")
        return builder

    def _responses_comparator(self):
        spec = importlib.util.find_spec("src.agent.semantic_responses")
        self.assertIsNotNone(spec, "Responses/json_schema module is missing")
        module = importlib.import_module("src.agent.semantic_responses")
        comparator = getattr(module, "DeepSeekResponsesJsonSchemaComparator", None)
        self.assertTrue(callable(comparator), "Responses/json_schema comparator is missing")
        return comparator

    def test_json_schema_closes_both_objects_and_requires_every_v2_field(self):
        schema = self._schema_builder()()
        self.assertEqual(schema["type"], "object")
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            set(schema["required"]),
            {"verdict", "comparability_basis", "dimensions", "reason", "notes"},
        )
        self.assertEqual(
            set(schema["properties"]["verdict"]["enum"]),
            {"EQUIVALENT", "DIFFERENT", "NOT_COMPARABLE"},
        )
        nested = schema["properties"]["dimensions"]
        self.assertEqual(nested["type"], "object")
        self.assertFalse(nested["additionalProperties"])
        self.assertEqual(
            set(nested["required"]),
            {"object_or_field", "value", "unit", "condition_or_applicability"},
        )
        self.assertTrue(all(
            set(nested["properties"][key]["enum"])
            == {"aligned", "different", "not_applicable", "uncertain"}
            for key in nested["required"]
        ))

    def test_response_request_maps_frozen_messages_and_revalidates_schema_output(self):
        output = _result()
        raw_body = json.dumps(_response_body(json.dumps(output)), ensure_ascii=False,
                              separators=(",", ":"))
        sent = []

        def transport(request, timeout):
            sent.append((request, timeout))
            return _Response(raw_body.encode("utf-8"))

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "private-test-key"}):
            client = self._responses_comparator()(transport=transport)
            result = client.compare("Compare A with B.", ["A evidence"], ["B evidence"])

        self.assertEqual(result, output)
        self.assertEqual(len(sent), 1)
        request, timeout = sent[0]
        self.assertEqual(request.full_url, "https://api.deepseek.com/responses")
        self.assertGreater(timeout, 0)
        payload = json.loads(request.data.decode("utf-8"))
        messages = contract.semantic_messages_v2(
            "Compare A with B.", ["A evidence"], ["B evidence"]
        )
        self.assertEqual(payload["model"], "deepseek-flash")
        self.assertEqual(payload["reasoning"], {"effort": "none"})
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["max_output_tokens"], 384)
        self.assertIs(payload["stream"], False)
        self.assertEqual(payload["instructions"], messages[0]["content"])
        self.assertEqual(payload["input"], [{"role": "user", "content": messages[1]["content"]}])
        output_format = payload["text"]["format"]
        self.assertEqual(output_format["type"], "json_schema")
        self.assertEqual(output_format["name"], "agent1_2_semantic_result")
        self.assertNotIn("strict", output_format)
        self.assertNotIn("private-test-key", request.data.decode("utf-8"))

        event = client.events[0]
        self.assertEqual(event["request_id"], "req_agent12")
        self.assertEqual(event["response_id"], "resp_agent12")
        self.assertEqual(event["provider_status"], "completed")
        self.assertEqual(event["http_status"], 200)
        self.assertEqual(event["raw_response_body"], raw_body)
        self.assertEqual(event["parsed_structured_output"], output)
        self.assertEqual(event["usage"]["prompt_tokens"], 42)
        self.assertEqual(event["usage"]["completion_tokens"], 31)
        self.assertEqual(event["usage"]["prompt_cache_hit_tokens"], 2)
        self.assertNotIn("private-test-key", json.dumps(event))

    def test_host_consistency_validation_runs_after_provider_schema_output(self):
        inconsistent = _result(dimensions={"value": "different"})
        calls = []

        def transport(request, timeout):
            calls.append(request)
            return _Response(_response_body(json.dumps(inconsistent)))

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "private-test-key"}):
            client = self._responses_comparator()(transport=transport)
            with self.assertRaises(InvalidSemanticOutputV2) as caught:
                client.compare("Compare A with B.", ["A evidence"], ["B evidence"])

        self.assertEqual(len(calls), 1)
        self.assertEqual(caught.exception.failure_type, "CONSISTENCY_ERROR")
        self.assertEqual(client.events[0]["validation_error"]["failure_type"],
                         "CONSISTENCY_ERROR")
        self.assertEqual(client.events[0]["parsed_structured_output"], inconsistent)

    def test_schema_failures_are_classified_by_contract_failure(self):
        missing = _result()
        missing.pop("notes")
        extra = dict(_result(), unexpected="value")
        invalid_enum = _result(verdict="SAME")
        wrong_type = _result()
        wrong_type["reason"] = 17
        nested = _result()
        nested["dimensions"].pop("unit")
        inconsistent = _result(dimensions={"value": "different"})
        cases = (
            ("not json", "INVALID_JSON"),
            ('{"verdict":"EQUIVALENT","verdict":"DIFFERENT"}', "INVALID_JSON"),
            ("", "EMPTY_RESPONSE"),
            (json.dumps(missing), "MISSING_FIELD"),
            (json.dumps(extra), "EXTRA_FIELD"),
            (json.dumps(invalid_enum), "INVALID_ENUM"),
            (json.dumps(wrong_type), "WRONG_TYPE"),
            (json.dumps(nested), "NESTED_SCHEMA_ERROR"),
            (json.dumps(inconsistent), "CONSISTENCY_ERROR"),
        )
        for text, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaises(InvalidSemanticOutputV2) as caught:
                    contract.parse_semantic_output_v2(text)
                self.assertEqual(caught.exception.failure_type, expected)

    def test_incomplete_response_is_truncated_and_raw_body_is_retained(self):
        response = _response_body(
            "", status="incomplete", output=[],
        )
        response["incomplete_details"] = {"reason": "max_output_tokens"}
        raw_body = json.dumps(response, separators=(",", ":"))

        def transport(_request, timeout):
            return _Response(raw_body.encode("utf-8"))

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "private-test-key"}):
            client = self._responses_comparator()(transport=transport)
            with self.assertRaises(InvalidSemanticOutputV2) as caught:
                client.compare("Compare A with B.", ["A evidence"], ["B evidence"])
        self.assertEqual(caught.exception.failure_type, "TRUNCATED")
        self.assertEqual(client.events[0]["raw_response_body"], raw_body)
        self.assertEqual(client.events[0]["provider_status"], "incomplete")

    def test_completed_response_without_output_text_is_not_accepted(self):
        raw_body = json.dumps(_response_body("", output=[{
            "type": "message", "status": "completed", "role": "assistant",
            "content": [],
        }]))

        def transport(_request, timeout):
            return _Response(raw_body.encode("utf-8"))

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "private-test-key"}):
            client = self._responses_comparator()(transport=transport)
            with self.assertRaises(InvalidSemanticOutputV2) as caught:
                client.compare("Compare A with B.", ["A evidence"], ["B evidence"])
        self.assertEqual(caught.exception.failure_type, "EMPTY_RESPONSE")
        self.assertEqual(client.events[0]["raw_response_body"], raw_body)

    def test_http_error_retains_response_body_and_request_id_without_key(self):
        error_body = b'{"error":{"message":"rate limited"}}'

        def transport(request, timeout):
            raise HTTPError(request.full_url, 429, "Too Many Requests",
                            {"x-request-id": "req_rate_limit"}, io.BytesIO(error_body))

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "private-test-key"}):
            client = self._responses_comparator()(transport=transport)
            with self.assertRaises(SemanticAPIError):
                client.compare("Compare A with B.", ["A evidence"], ["B evidence"])
        event = client.events[0]
        self.assertEqual(event["request_id"], "req_rate_limit")
        self.assertEqual(event["raw_response_body"], error_body.decode("utf-8"))
        self.assertNotIn("private-test-key", json.dumps(event))


if __name__ == "__main__":
    unittest.main()
