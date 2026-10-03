"""DeepSeek Responses transport with a provider-enforced semantic JSON Schema."""

import base64
import json
import math
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .semantic import SemanticAPIError, SemanticCostBudget
from .semantic_contract_v2 import (
    InvalidSemanticOutputV2,
    parse_semantic_output_v2,
    semantic_messages_v2,
    semantic_output_json_schema_v2,
)


_MAX_PROVIDER_BODY_BYTES = 1_000_000


class DeepSeekResponsesJsonSchemaComparator:
    """Make one Responses/json_schema call and revalidate its output on the host."""

    URL = "https://api.deepseek.com/responses"

    def __init__(self, *, timeout=90, max_tokens=384, budget=None, transport=None,
                 message_builder=semantic_messages_v2,
                 output_parser=parse_semantic_output_v2):
        if (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0
                or type(max_tokens) is not int or not 1 <= max_tokens <= 512
                or not callable(message_builder) or not callable(output_parser)):
            raise ValueError("invalid semantic Responses API limits")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.budget = budget
        self.transport = transport or urlopen
        self.message_builder = message_builder
        self.output_parser = output_parser
        self.events = []

    @staticmethod
    def _request_id(headers):
        getter = getattr(headers, "get", None)
        if not callable(getter):
            return None
        for name in ("x-request-id", "request-id", "x-deepseek-request-id"):
            value = getter(name)
            if type(value) is str and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _response_status(response):
        status = getattr(response, "status", None)
        if type(status) is int:
            return status
        getcode = getattr(response, "getcode", None)
        try:
            status = getcode() if callable(getcode) else None
        except Exception:
            return None
        return status if type(status) is int else None

    @staticmethod
    def _retain_body(event, raw):
        truncated = len(raw) > _MAX_PROVIDER_BODY_BYTES
        retained = raw[:_MAX_PROVIDER_BODY_BYTES]
        event["raw_response_body"] = retained.decode("utf-8", errors="replace")
        event["raw_response_body_base64"] = base64.b64encode(retained).decode("ascii")
        event["response_body_truncated"] = truncated

    @staticmethod
    def _record_usage(event, provider_usage):
        event["provider_usage"] = provider_usage if type(provider_usage) is dict else {}
        canonical = {}
        if type(provider_usage) is dict:
            for source, target in (("input_tokens", "prompt_tokens"),
                                   ("output_tokens", "completion_tokens"),
                                   ("total_tokens", "total_tokens")):
                value = provider_usage.get(source)
                if type(value) is int and value >= 0:
                    canonical[target] = value
            details = provider_usage.get("input_tokens_details")
            cached = details.get("cached_tokens") if type(details) is dict else None
            if type(cached) is int and cached >= 0:
                canonical["prompt_cache_hit_tokens"] = cached
                prompt = canonical.get("prompt_tokens")
                if prompt is not None and cached <= prompt:
                    canonical["prompt_cache_miss_tokens"] = prompt - cached
        event["usage"] = canonical

    def _request_payload(self, comparison_request, excerpts_a, excerpts_b):
        messages = self.message_builder(comparison_request, excerpts_a, excerpts_b)
        if (type(messages) is not list or not messages or type(messages[0]) is not dict
                or messages[0].get("role") != "system"
                or type(messages[0].get("content")) is not str):
            raise ValueError("semantic message builder must start with system instructions")
        if any(type(message) is not dict or message.get("role") == "system"
               or message.get("role") not in ("user", "assistant", "developer")
               or type(message.get("content")) is not str
               for message in messages[1:]):
            raise ValueError("semantic message builder returned unsupported input messages")
        return {
            "model": "deepseek-flash",
            "instructions": messages[0]["content"],
            "input": [
                {"role": message["role"], "content": message["content"]}
                for message in messages[1:]
            ],
            "reasoning": {"effort": "none"},
            "temperature": 0,
            "max_output_tokens": self.max_tokens,
            "stream": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "agent1_2_semantic_result",
                    "schema": semantic_output_json_schema_v2(),
                }
            },
        }

    @staticmethod
    def _assistant_output_text(body):
        output = body.get("output")
        if type(output) is not list:
            raise SemanticAPIError("deepseek:malformed_response")
        text_parts = []
        saw_refusal = False
        for item in output:
            if (type(item) is not dict or item.get("type") != "message"
                    or item.get("role") != "assistant"):
                continue
            content = item.get("content")
            if type(content) is not list:
                continue
            for part in content:
                if type(part) is not dict:
                    continue
                if part.get("type") == "output_text":
                    text = part.get("text")
                    if type(text) is not str:
                        raise InvalidSemanticOutputV2(
                            "Responses output_text is not text", failure_type="WRONG_TYPE"
                        )
                    text_parts.append(text)
                elif part.get("type") == "refusal":
                    saw_refusal = True
        if not text_parts:
            failure_type = "OTHER" if saw_refusal else "EMPTY_RESPONSE"
            raise InvalidSemanticOutputV2(
                "Responses API returned no assistant output_text",
                failure_type=failure_type,
            )
        content = "".join(text_parts)
        if not content:
            raise InvalidSemanticOutputV2("empty semantic response",
                                          failure_type="EMPTY_RESPONSE")
        return content

    def _read_http_error(self, event, exc):
        event["http_status"] = exc.code
        event["request_id"] = self._request_id(exc.headers)
        try:
            self._retain_body(event, exc.read(_MAX_PROVIDER_BODY_BYTES + 1))
        except Exception:
            event["raw_response_body"] = None
            event["raw_response_body_base64"] = None
            event["response_body_truncated"] = False

    def compare(self, comparison_request, excerpts_a, excerpts_b):
        key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not key:
            raise SemanticAPIError("deepseek:missing_key")
        payload = self._request_payload(comparison_request, excerpts_a, excerpts_b)
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        reservation = self.budget.reserve(len(encoded), self.max_tokens) if self.budget else 0.0
        event = {
            "retry": False,
            "http_status": None,
            "request_id": None,
            "response_id": None,
            "provider_status": None,
            "provider_usage": {},
            "usage": {},
            "raw_response_body": None,
            "raw_response_body_base64": None,
            "response_body_truncated": False,
            "parsed_structured_output": None,
            "validation_error": None,
            "error": None,
            "latency_ms": None,
            "cost_rmb": None,
            "reserved_rmb": reservation,
        }
        self.events.append(event)
        request = Request(
            self.URL,
            data=encoded,
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + key},
            method="POST",
        )
        started = time.monotonic()
        try:
            with self.transport(request, timeout=self.timeout) as response:
                event["http_status"] = self._response_status(response)
                event["request_id"] = self._request_id(getattr(response, "headers", None))
                raw = response.read(_MAX_PROVIDER_BODY_BYTES + 1)
            self._retain_body(event, raw)
            event["latency_ms"] = round((time.monotonic() - started) * 1000, 3)
            if event["http_status"] is not None and not 200 <= event["http_status"] < 300:
                event["error"] = "http_" + str(event["http_status"])
                raise SemanticAPIError("deepseek:" + event["error"])
            if event["response_body_truncated"]:
                raise InvalidSemanticOutputV2(
                    "Responses API body exceeds the diagnostic size limit",
                    failure_type="TRUNCATED",
                )
            try:
                body = json.loads(event["raw_response_body"])
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise InvalidSemanticOutputV2(
                    "invalid Responses API JSON body", failure_type="INVALID_JSON"
                ) from exc
            if type(body) is not dict:
                raise SemanticAPIError("deepseek:malformed_response")
            event["response_id"] = body.get("id") if type(body.get("id")) is str else None
            event["provider_status"] = (
                body.get("status") if type(body.get("status")) is str else None
            )
            self._record_usage(event, body.get("usage"))
            if self.budget:
                event["cost_rmb"] = self.budget.settle(reservation, event["usage"])
            status = event["provider_status"]
            if status == "incomplete":
                details = body.get("incomplete_details")
                reason = details.get("reason") if type(details) is dict else None
                failure_type = "TRUNCATED" if reason == "max_output_tokens" else "OTHER"
                raise InvalidSemanticOutputV2(
                    "Responses API did not complete the semantic output",
                    failure_type=failure_type,
                )
            if status != "completed":
                event["error"] = "provider_status_" + (status or "missing")
                raise SemanticAPIError("deepseek:provider_response_failed")
            content = self._assistant_output_text(body)
            try:
                event["parsed_structured_output"] = json.loads(content)
            except (json.JSONDecodeError, TypeError, ValueError):
                event["parsed_structured_output"] = None
            return self.output_parser(content)
        except HTTPError as exc:
            self._read_http_error(event, exc)
            event["error"] = "http_" + str(exc.code)
            event["latency_ms"] = round((time.monotonic() - started) * 1000, 3)
            exc.close()
            raise SemanticAPIError("deepseek:" + event["error"]) from None
        except (TimeoutError, URLError, OSError) as exc:
            event["latency_ms"] = round((time.monotonic() - started) * 1000, 3)
            timed_out = (isinstance(exc, TimeoutError)
                         or isinstance(getattr(exc, "reason", None), TimeoutError))
            event["error"] = "timeout" if timed_out else "network_error"
            raise SemanticAPIError("deepseek:" + event["error"]) from None
        except InvalidSemanticOutputV2 as exc:
            if event["latency_ms"] is None:
                event["latency_ms"] = round((time.monotonic() - started) * 1000, 3)
            event["error"] = "invalid_output"
            event["validation_error"] = {
                "failure_type": exc.failure_type,
                "reason_code": exc.reason_code,
                "message": str(exc),
            }
            raise
        except SemanticAPIError:
            if event["latency_ms"] is None:
                event["latency_ms"] = round((time.monotonic() - started) * 1000, 3)
            if event["error"] is None:
                event["error"] = "malformed_response"
            raise
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            if event["latency_ms"] is None:
                event["latency_ms"] = round((time.monotonic() - started) * 1000, 3)
            event["error"] = "malformed_response"
            raise SemanticAPIError("deepseek:malformed_response") from None
