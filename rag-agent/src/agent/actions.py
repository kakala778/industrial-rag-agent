"""Strict action-specific JSON validation; never repair model output."""

import json

from .progress import snapshot


class InvalidAction(ValueError):
    pass


def finish_ready(state):
    return bool(state.resolved_scopes) and not snapshot(state)["remaining_scopes"]


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise InvalidAction("duplicate JSON key")
        value[key] = item
    return value


def _nonfinite(value):
    raise InvalidAction("nonfinite JSON number")


def _text(value, maximum):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= maximum


def validate_action(raw):
    if isinstance(raw, str):
        if len(raw) > 32000:
            raise InvalidAction("action too long")
        try:
            raw = json.loads(raw, object_pairs_hook=_object, parse_constant=_nonfinite)
        except (ValueError, RecursionError) as exc:
            raise InvalidAction("invalid JSON action") from exc
    if not isinstance(raw, dict):
        raise InvalidAction("action must be an object")
    action = raw.get("action")
    keys = {"SEARCH": {"action", "query", "scopes"},
            "LOOKUP": {"action", "evidence_id"},
            "CLARIFY": {"action", "question"},
            "FINISH": {"action", "findings"}}
    if not isinstance(action, str) or action not in keys or set(raw) != keys[action]:
        raise InvalidAction("unknown action or unexpected fields")
    if action == "SEARCH":
        scopes = raw["scopes"]
        if (not _text(raw["query"], 4000) or not isinstance(scopes, list)
                or len(scopes) != 1 or not _text(scopes[0], 64)):
            raise InvalidAction("invalid SEARCH arguments")
    elif action == "LOOKUP":
        if not _text(raw["evidence_id"], 100):
            raise InvalidAction("invalid LOOKUP arguments")
    elif action == "CLARIFY":
        if not _text(raw["question"], 500):
            raise InvalidAction("invalid CLARIFY arguments")
    else:
        findings = raw["findings"]
        if not isinstance(findings, list) or len(findings) > 6:
            raise InvalidAction("invalid findings")
        for row in findings:
            if (not isinstance(row, dict) or set(row) != {"scope", "evidence_id", "quote"}
                    or not _text(row["scope"], 64) or not _text(row["evidence_id"], 100)
                    or not _text(row["quote"], 1000)):
                raise InvalidAction("invalid finding")
    return raw


ACTION_JSON_SCHEMA = {
    "oneOf": [
        {"type": "object", "properties": {
            "action": {"const": "SEARCH"}, "query": {"type": "string"},
            "scopes": {"type": "array", "items": {"type": "string"},
                       "minItems": 1, "maxItems": 1}},
         "required": ["action", "query", "scopes"], "additionalProperties": False},
        {"type": "object", "properties": {
            "action": {"const": "LOOKUP"}, "evidence_id": {"type": "string"}},
         "required": ["action", "evidence_id"], "additionalProperties": False},
        {"type": "object", "properties": {
            "action": {"const": "CLARIFY"}, "question": {"type": "string"}},
         "required": ["action", "question"], "additionalProperties": False},
        {"type": "object", "properties": {
            "action": {"const": "FINISH"}, "findings": {"type": "array", "maxItems": 6,
                "items": {"type": "object", "properties": {
                    "scope": {"type": "string"}, "evidence_id": {"type": "string"},
                    "quote": {"type": "string"}},
                    "required": ["scope", "evidence_id", "quote"],
                    "additionalProperties": False}}},
         "required": ["action", "findings"], "additionalProperties": False},
    ]}
