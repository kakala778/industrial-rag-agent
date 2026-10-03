"""Strict, bounded semantic-comparison contract for saved evidence references."""

import json
import math
import os
import re
import time
from decimal import Decimal, InvalidOperation
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MODEL_VERDICTS = frozenset({"EQUIVALENT", "DIFFERENT", "NOT_COMPARABLE"})
DIMENSION_KEYS = (
    "object_or_field",
    "value",
    "unit",
    "condition_or_applicability",
)
DIMENSION_LABELS = frozenset({"aligned", "different", "not_applicable", "uncertain"})
_OUTPUT_KEYS = frozenset({"verdict", "dimensions", "reason", "notes"})
_NUMBER_PATTERN = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")
_MAX_OUTPUT_CHARS = 8192
_MAX_REASON_CHARS = 500
_MAX_NOTES_CHARS = 300


class InvalidSemanticOutput(ValueError):
    """The model did not return the exact semantic result contract."""


class SemanticAPIError(RuntimeError):
    """Sanitized API failure; provider response bodies and credentials are omitted."""


class SemanticCostBudget:
    """Peak-rate reservation with explicit soft and hard RMB ceilings."""

    INPUT_HIT_RMB_PER_MILLION = 0.04
    INPUT_MISS_RMB_PER_MILLION = 2.0
    OUTPUT_RMB_PER_MILLION = 8.0

    def __init__(self, *, soft_limit_rmb=3.0, hard_limit_rmb=5.0):
        values = (soft_limit_rmb, hard_limit_rmb)
        if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
               for value in values):
            raise ValueError("invalid semantic API budget")
        if soft_limit_rmb > 3.0 or hard_limit_rmb > 5.0 or soft_limit_rmb > hard_limit_rmb:
            raise ValueError("semantic API budget exceeds the milestone ceiling")
        self.soft_limit_rmb = float(soft_limit_rmb)
        self.hard_limit_rmb = float(hard_limit_rmb)
        self.reserved_rmb = 0.0

    def reserve(self, request_bytes, max_tokens):
        if type(request_bytes) is not int or request_bytes < 0:
            raise ValueError("invalid request size")
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("invalid output token cap")
        # Treat every UTF-8 byte as an input token and reserve output at peak rate.
        estimate = ((request_bytes + 4096) * self.INPUT_MISS_RMB_PER_MILLION
                    + max_tokens * self.OUTPUT_RMB_PER_MILLION) / 1_000_000
        if self.reserved_rmb + estimate > self.soft_limit_rmb:
            raise SemanticAPIError("deepseek:cost_limit")
        if self.reserved_rmb + estimate > self.hard_limit_rmb:
            raise SemanticAPIError("deepseek:hard_budget_limit")
        self.reserved_rmb += estimate
        return estimate

    def settle(self, reservation, usage):
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if type(prompt) is not int or prompt < 0 or type(completion) is not int or completion < 0:
            return None
        hit = usage.get("prompt_cache_hit_tokens")
        miss = usage.get("prompt_cache_miss_tokens")
        if (type(hit) is int and hit >= 0 and type(miss) is int and miss >= 0
                and hit + miss == prompt):
            input_cost = (hit * self.INPUT_HIT_RMB_PER_MILLION
                          + miss * self.INPUT_MISS_RMB_PER_MILLION) / 1_000_000
        else:
            input_cost = prompt * self.INPUT_MISS_RMB_PER_MILLION / 1_000_000
        actual = input_cost + completion * self.OUTPUT_RMB_PER_MILLION / 1_000_000
        self.reserved_rmb = max(0.0, self.reserved_rmb + actual - reservation)
        return actual


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidSemanticOutput("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(_value):
    raise InvalidSemanticOutput("non-finite JSON number")


def parse_semantic_output(text):
    """Parse one JSON object and reject duplicate keys without repairing it."""
    if not isinstance(text, str) or not text or len(text) > _MAX_OUTPUT_CHARS:
        raise InvalidSemanticOutput("invalid JSON text size")
    try:
        value = json.loads(text, object_pairs_hook=_unique_object,
                           parse_constant=_reject_constant)
    except InvalidSemanticOutput:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise InvalidSemanticOutput("invalid semantic JSON") from exc
    return validate_semantic_output(value)


def validate_semantic_output(value):
    """Validate exact keys, enums, and bounded text; return a detached value."""
    if type(value) is not dict or set(value) != _OUTPUT_KEYS:
        raise InvalidSemanticOutput("semantic output keys do not match schema")
    verdict = value["verdict"]
    dimensions = value["dimensions"]
    reason = value["reason"]
    notes = value["notes"]
    if type(verdict) is not str or verdict not in MODEL_VERDICTS:
        raise InvalidSemanticOutput("invalid semantic verdict")
    if type(dimensions) is not dict or set(dimensions) != set(DIMENSION_KEYS):
        raise InvalidSemanticOutput("semantic dimensions do not match schema")
    if any(type(dimensions[key]) is not str or dimensions[key] not in DIMENSION_LABELS
           for key in DIMENSION_KEYS):
        raise InvalidSemanticOutput("invalid semantic dimension label")
    object_alignment = dimensions["object_or_field"]
    comparison_dimensions = (
        dimensions["value"],
        dimensions["unit"],
        dimensions["condition_or_applicability"],
    )
    if verdict == "EQUIVALENT":
        consistent = (
            object_alignment == "aligned"
            and dimensions["value"] == "aligned"
            and dimensions["unit"] == "aligned"
            and dimensions["condition_or_applicability"] in ("aligned", "not_applicable")
        )
    elif verdict == "DIFFERENT":
        consistent = (
            object_alignment == "aligned"
            and "different" in comparison_dimensions
        )
    else:
        consistent = (
            object_alignment == "different"
            or dimensions["condition_or_applicability"] == "different"
        )
    if not consistent:
        raise InvalidSemanticOutput("semantic verdict conflicts with dimension labels")
    if (type(reason) is not str or not reason.strip()
            or len(reason) > _MAX_REASON_CHARS):
        raise InvalidSemanticOutput("invalid semantic reason")
    if type(notes) is not str or len(notes) > _MAX_NOTES_CHARS:
        raise InvalidSemanticOutput("invalid semantic notes")
    return {
        "verdict": verdict,
        "dimensions": {key: dimensions[key] for key in DIMENSION_KEYS},
        "reason": reason.strip(),
        "notes": notes.strip(),
    }


def normalize_decimal(value):
    """Parse plain signed decimal notation only; do not parse operators or units."""
    if type(value) is not str or not _NUMBER_PATTERN.fullmatch(value):
        return None
    try:
        number = Decimal(value)
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def numeric_values_equal(left, right):
    """Compare plain decimals while retaining the sign, including signed zero."""
    left_number = normalize_decimal(left)
    right_number = normalize_decimal(right)
    if left_number is None or right_number is None:
        return False
    return left_number == right_number and left_number.is_signed() == right_number.is_signed()


def semantic_messages(comparison_request, excerpts_a, excerpts_b):
    """Build a comparison prompt from accepted excerpts only, without IDs/citations."""
    if type(comparison_request) is not str or not comparison_request.strip():
        raise ValueError("comparison request is required")
    sides = (excerpts_a, excerpts_b)
    if any(type(rows) is not list or not rows or len(rows) > 6 for rows in sides):
        raise ValueError("each side requires one to six accepted excerpts")
    if any(type(text) is not str or not text.strip() or len(text) > 1200
           for rows in sides for text in rows):
        raise ValueError("invalid accepted excerpt")
    user_payload = {
        "comparison_request": comparison_request.strip(),
        "scope_A_evidence": excerpts_a,
        "scope_B_evidence": excerpts_b,
    }
    system = (
        "Compare only the user's request and the supplied Scope A and Scope B evidence. "
        "Assume the host already accepted both sides as supported. Do not search, add "
        "facts, infer omitted values, make recommendations, or assess safety/compliance. "
        "Decide comparability by engineering object and field before comparing numbers: "
        "the same value or unit does not make different objects comparable. Use EQUIVALENT "
        "only for the same or equivalent fact with aligned value and unit and no conflicting "
        "applicability. Use DIFFERENT only for the same comparable fact with a different "
        "value, unit, or applicability. Use NOT_COMPARABLE when the objects, fields, "
        "endpoints, scope, or installation scenario make them different comparands. Never "
        "convert units or guess OCR, signs, operators, or missing values. Plain decimal "
        "formatting such as 5.0 versus 5 may align; preserve explicit signs and comparison "
        "operators. If both sides state "
        "different unit expressions, mark the unit dimension different and say "
        "unit_conversion_required in notes. Use not_applicable when a side does not state a "
        "dimension and uncertain when the supplied evidence cannot resolve it. Use only "
        "these four dimensions: object_or_field, value, unit, and condition_or_applicability. "
        "Operators and normative modality may be mentioned in notes only and are not scored "
        "dimensions. Do not output evidence IDs, citations, source quotes, or copied excerpts. "
        "Give a brief explanation in reason. Return one JSON object with exactly these keys: "
        "verdict, dimensions, reason, notes. Model verdict must be EQUIVALENT, DIFFERENT, or "
        "NOT_COMPARABLE; host code handles insufficient evidence before you are called."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


class DeepSeekSemanticComparator:
    """One non-retrying DeepSeek Flash call per already-validated evidence pair."""

    URL = "https://api.deepseek.com/chat/completions"

    def __init__(self, *, timeout=90, max_tokens=384, budget=None, transport=None):
        if (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0
                or type(max_tokens) is not int or not 1 <= max_tokens <= 512):
            raise ValueError("invalid semantic API limits")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.budget = budget
        self.transport = transport or urlopen
        self.events = []

    def compare(self, comparison_request, excerpts_a, excerpts_b):
        key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not key:
            raise SemanticAPIError("deepseek:missing_key")
        payload = {
            "model": "deepseek-flash",
            "messages": semantic_messages(comparison_request, excerpts_a, excerpts_b),
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
            "stream": False,
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        reservation = self.budget.reserve(len(encoded), self.max_tokens) if self.budget else 0.0
        event = {"retry": False, "usage": {}, "error": None,
                 "latency_ms": None, "cost_rmb": None,
                 "reserved_rmb": reservation}
        self.events.append(event)
        request = Request(self.URL, data=encoded,
                          headers={"Content-Type": "application/json",
                                   "Authorization": "Bearer " + key},
                          method="POST")
        started = time.monotonic()
        try:
            with self.transport(request, timeout=self.timeout) as response:
                raw = response.read(128001)
            event["latency_ms"] = round((time.monotonic() - started) * 1000, 3)
            if len(raw) > 128000:
                raise InvalidSemanticOutput("response too large")
            body = json.loads(raw)
            if type(body) is not dict:
                raise ValueError("invalid response envelope")
            raw_usage = body.get("usage")
            if type(raw_usage) is dict:
                event["usage"] = {
                    key_name: value for key_name, value in raw_usage.items()
                    if key_name in ("prompt_tokens", "completion_tokens", "total_tokens",
                                    "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")
                    and type(value) is int and value >= 0
                }
            if self.budget:
                event["cost_rmb"] = self.budget.settle(reservation, event["usage"])
            choice = body["choices"][0]
            if choice.get("finish_reason") == "length":
                raise InvalidSemanticOutput("truncated semantic output")
            content = choice["message"]["content"]
            if type(content) is not str:
                raise InvalidSemanticOutput("semantic content is not text")
            result = parse_semantic_output(content)
            return result
        except HTTPError as exc:
            event["latency_ms"] = round((time.monotonic() - started) * 1000, 3)
            event["error"] = "http_" + str(exc.code)
            exc.close()
            raise SemanticAPIError("deepseek:" + event["error"]) from None
        except (TimeoutError, URLError) as exc:
            event["latency_ms"] = round((time.monotonic() - started) * 1000, 3)
            timed_out = (isinstance(exc, TimeoutError)
                         or isinstance(getattr(exc, "reason", None), TimeoutError))
            event["error"] = "timeout" if timed_out else "network_error"
            raise SemanticAPIError("deepseek:" + event["error"]) from None
        except InvalidSemanticOutput:
            event["latency_ms"] = (event["latency_ms"] if event["latency_ms"] is not None
                                    else round((time.monotonic() - started) * 1000, 3))
            event["error"] = "invalid_output"
            raise
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            event["latency_ms"] = (event["latency_ms"] if event["latency_ms"] is not None
                                    else round((time.monotonic() - started) * 1000, 3))
            event["error"] = "malformed_response"
            raise SemanticAPIError("deepseek:malformed_response") from None
