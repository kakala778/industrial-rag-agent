"""Agent 1.1 semantic contract with independent dimension observations."""

import json

from .semantic import DIMENSION_KEYS, InvalidSemanticOutput


MODEL_VERDICTS_V2 = frozenset({"EQUIVALENT", "DIFFERENT", "NOT_COMPARABLE"})
DIMENSION_LABELS_V2 = frozenset({"aligned", "different", "not_applicable", "uncertain"})
COMPARABILITY_BASES = frozenset({
    "comparable_fact",
    "different_object_or_field",
    "incompatible_scope_or_installation_scenario",
    "uncertain",
})
OUTPUT_KEYS_V2 = frozenset({
    "verdict", "comparability_basis", "dimensions", "reason", "notes",
})
MAX_OUTPUT_CHARS = 8192
MAX_REASON_CHARS = 500
MAX_NOTES_CHARS = 300


class InvalidSemanticOutputV2(InvalidSemanticOutput):
    """Agent 1.1 result rejected by its stricter, versioned output contract."""

    def __init__(self, message, *, reason_code="invalid_schema"):
        super().__init__(message)
        self.reason_code = reason_code


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise InvalidSemanticOutputV2("duplicate JSON key")
        value[key] = item
    return value


def _reject_constant(_value):
    raise InvalidSemanticOutputV2("non-finite JSON number")


def parse_semantic_output_v2(text):
    """Parse a single JSON object without duplicate keys or schema repair."""
    if type(text) is not str or not text or len(text) > MAX_OUTPUT_CHARS:
        raise InvalidSemanticOutputV2("invalid JSON text size")
    try:
        result = json.loads(text, object_pairs_hook=_unique_object,
                            parse_constant=_reject_constant)
    except InvalidSemanticOutputV2:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise InvalidSemanticOutputV2("invalid semantic JSON") from exc
    return validate_semantic_output_v2(result)


def validate_semantic_output_v2(value):
    """Validate exact fields and deterministic verdict/dimension consistency."""
    if type(value) is not dict or set(value) != OUTPUT_KEYS_V2:
        raise InvalidSemanticOutputV2("semantic output keys do not match Agent 1.1 schema")
    verdict = value["verdict"]
    basis = value["comparability_basis"]
    dimensions = value["dimensions"]
    reason = value["reason"]
    notes = value["notes"]
    if type(verdict) is not str or verdict not in MODEL_VERDICTS_V2:
        raise InvalidSemanticOutputV2("invalid semantic verdict")
    if type(basis) is not str or basis not in COMPARABILITY_BASES:
        raise InvalidSemanticOutputV2("invalid comparability basis")
    if type(dimensions) is not dict or set(dimensions) != set(DIMENSION_KEYS):
        raise InvalidSemanticOutputV2("semantic dimensions do not match Agent 1.1 schema")
    if any(type(dimensions[key]) is not str or dimensions[key] not in DIMENSION_LABELS_V2
           for key in DIMENSION_KEYS):
        raise InvalidSemanticOutputV2("invalid semantic dimension label")
    if (type(reason) is not str or not reason.strip()
            or len(reason) > MAX_REASON_CHARS):
        raise InvalidSemanticOutputV2("invalid semantic reason")
    if type(notes) is not str or len(notes) > MAX_NOTES_CHARS:
        raise InvalidSemanticOutputV2("invalid semantic notes")

    object_label = dimensions["object_or_field"]
    other_labels = (
        dimensions["value"],
        dimensions["unit"],
        dimensions["condition_or_applicability"],
    )
    has_uncertainty = any(label == "uncertain" for label in dimensions.values())
    consistent = False
    if basis == "uncertain":
        consistent = verdict == "NOT_COMPARABLE" and has_uncertainty
    elif has_uncertainty:
        consistent = False
    elif verdict == "EQUIVALENT":
        consistent = (
            basis == "comparable_fact"
            and object_label == "aligned"
            and dimensions["value"] == "aligned"
            and dimensions["unit"] in ("aligned", "not_applicable")
            and dimensions["condition_or_applicability"] in ("aligned", "not_applicable")
        )
    elif verdict == "DIFFERENT":
        consistent = (
            basis == "comparable_fact"
            and object_label == "aligned"
            and "different" in other_labels
        )
    elif verdict == "NOT_COMPARABLE":
        if basis == "different_object_or_field":
            consistent = object_label == "different"
        elif basis == "incompatible_scope_or_installation_scenario":
            consistent = (
                object_label == "aligned"
                and dimensions["condition_or_applicability"] == "different"
            )
    if not consistent:
        raise InvalidSemanticOutputV2(
            "verdict conflicts with independent dimensions or comparability basis",
            reason_code="verdict_dimension_inconsistency",
        )
    return {
        "verdict": verdict,
        "comparability_basis": basis,
        "dimensions": {key: dimensions[key] for key in DIMENSION_KEYS},
        "reason": reason.strip(),
        "notes": notes.strip(),
    }


def semantic_messages_v2(comparison_request, excerpts_a, excerpts_b):
    """Build the Agent 1.1 prompt from a request and source-page excerpts only."""
    if type(comparison_request) is not str or not comparison_request.strip():
        raise ValueError("comparison request is required")
    sides = (excerpts_a, excerpts_b)
    if any(type(rows) is not list or not rows or len(rows) > 6 for rows in sides):
        raise ValueError("each side requires one to six original-PDF excerpts")
    if any(type(text) is not str or not text.strip() or len(text) > 1200
           for rows in sides for text in rows):
        raise ValueError("invalid original-PDF excerpt")
    user_payload = {
        "comparison_request": comparison_request.strip(),
        "scope_A_evidence": excerpts_a,
        "scope_B_evidence": excerpts_b,
    }
    system = (
        "Compare only the request and the supplied original-PDF evidence. Assume both "
        "sides passed deterministic host preflight. Independently classify all four "
        "dimensions first: object_or_field, value, unit, condition_or_applicability. Do this "
        "before deciding a verdict. Dimension labels record what the evidence states; an "
        "object/field mismatch is never a reason to mark value or unit not_applicable. "
        "Use not_applicable only when that dimension itself is absent in one or both "
        "evidence sides, not as a shortcut for ambiguity or object mismatch. Use uncertain "
        "when the evidence cannot resolve a dimension. The explicitly named A/B scopes "
        "are the intended comparison axis in the request; their names alone do not make "
        "facts incomparable. A condition is a qualification or applicability constraint "
        "on the requested fact, not merely the A/B label. Then decide comparability. Use "
        "EQUIVALENT only for the same object/field and aligned value, with aligned or "
        "not_applicable unit and condition. Use DIFFERENT when the same requested fact "
        "remains comparable but its value, unit, or applicable condition differs. Use "
        "NOT_COMPARABLE when the object/field differs, or when an incompatible scope or "
        "installation scenario changes the comparand rather than just qualifying the same "
        "fact. Set comparability_basis to comparable_fact, different_object_or_field, "
        "incompatible_scope_or_installation_scenario, or uncertain, consistently with "
        "the verdict. Never convert units or guess OCR, signs, operators, or missing values. "
        "If units differ and a conversion would be needed, mark unit different and put "
        "unit_conversion_required in notes. Operators and normative modality may be noted "
        "but are not separate scored dimensions. Do not search, add facts, infer omitted "
        "values, make recommendations, assess safety/compliance, or return evidence IDs, "
        "citations, or copied excerpts. Give a brief reason. Return one JSON object with "
        "exactly these keys: verdict, comparability_basis, dimensions, reason, notes. "
        "The three verdicts are EQUIVALENT, DIFFERENT, and NOT_COMPARABLE. Host code "
        "handles insufficient evidence before you are called."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
