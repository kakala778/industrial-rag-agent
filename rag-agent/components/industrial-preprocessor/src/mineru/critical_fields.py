"""Conservative cross-checks for declared industrial critical fields."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Mapping, Sequence


R1_CANONICAL_LIST_SEPARATOR = "R1_CANONICAL_LIST_SEPARATOR"
R2_STANDALONE_SLASH_COLLAPSE = "R2_STANDALONE_SLASH_COLLAPSE"


class CriticalFieldType(str, Enum):
    """Critical field categories supported by the first validation version."""

    MODEL = "model"
    NUMBER = "number"
    UNIT = "unit"
    PORT_CONFIG = "port_config"
    QUANTITY = "quantity"


class ConflictType(str, Enum):
    """Explain why parser and native values differ."""

    LETTER_DIGIT_CONFUSION = "LETTER_DIGIT_CONFUSION"
    CRITICAL_FIELD_CONFLICT = "CRITICAL_FIELD_CONFLICT"


@dataclass(frozen=True)
class CriticalConflict:
    """One auditable difference between parser and native field values."""

    conflict_type: ConflictType
    parser_value: str
    native_value: str | None
    normalized_parser_value: str
    normalized_native_value: str
    differences: tuple[tuple[int, str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "conflict_type": self.conflict_type.value,
            "parser_value": self.parser_value,
            "native_value": self.native_value,
            "normalized_parser_value": self.normalized_parser_value,
            "normalized_native_value": self.normalized_native_value,
            "differences": [
                {
                    "position": position,
                    "parser_char": parser_char,
                    "native_char": native_char,
                }
                for position, parser_char, native_char in self.differences
            ],
        }


@dataclass(frozen=True)
class CriticalFieldCheckResult:
    """Result and provenance of one critical-field validation."""

    original_parser_value: str
    native_value: str | None
    field_type: CriticalFieldType | str | None
    passed: bool
    conflicts: tuple[CriticalConflict, ...]
    resolved_value: str | None
    review_required: bool
    evidence: dict[str, object]
    resolution_source: str | None = None
    resolution_reason: str | None = None
    rule_id: str = "critical-field-letter-digit-arbitration-v1"

    def to_dict(self) -> dict[str, object]:
        field_type = self.field_type
        if isinstance(field_type, Enum):
            field_type = field_type.value
        return {
            "original_parser_value": self.original_parser_value,
            "native_value": self.native_value,
            "resolved_value": self.resolved_value,
            "field_type": field_type,
            "passed": self.passed,
            "review_required": self.review_required,
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
            "resolution_source": self.resolution_source,
            "resolution_reason": self.resolution_reason,
            "rule_id": self.rule_id,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class CriticalFieldInput:
    """A declared critical field for the optional validation stage."""

    field_name: str
    parser_value: str
    native_pdf_text: str | None
    field_type: CriticalFieldType | str | None = None
    context: Mapping[str, object] = field(default_factory=dict)


class CriticalFieldValidator:
    """Compare parser values with native PDF text without guessing."""

    _LIST_FIELD_TYPES = frozenset({CriticalFieldType.PORT_CONFIG})
    _SLASH_FIELD_TYPES = frozenset(
        {CriticalFieldType.MODEL, CriticalFieldType.PORT_CONFIG}
    )
    _LIST_SEPARATOR_PATTERN = re.compile(r"[；、;,，]")
    _DIAMETER_MEASUREMENT_PATTERN = re.compile(
        r"^\s*\$?\s*(?:¢|φ|Φ|Ø|\\phi|\\Phi)\s*"
        r"(\d+(?:\.\d+)?)\s*(mm|cm|m)\s*\$?\s*$",
        re.IGNORECASE,
    )

    _PUNCTUATION_TRANSLATION = str.maketrans(
        {
            "（": "(",
            "）": ")",
            "［": "[",
            "］": "]",
            "｛": "{",
            "｝": "}",
            "，": ",",
            "、": ",",
            "；": ";",
            "：": ":",
        }
    )
    _CONFUSABLE_PAIRS = {
        frozenset(("O", "0")),
        frozenset(("I", "1")),
        frozenset(("l", "1")),
        frozenset(("B", "8")),
        frozenset(("S", "5")),
    }

    def validate(
        self,
        parser_value: str,
        native_pdf_text: str | None,
        field_type: CriticalFieldType | str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> CriticalFieldCheckResult:
        """Validate one declared critical field conservatively."""

        context = context or {}
        normalized_parser = self._normalize(parser_value, field_type)
        normalized_native = self._normalize(native_pdf_text or "", field_type)
        native_reliable = bool(context.get("native_text_reliable", False))
        native_candidate_unique = bool(
            context.get("native_candidate_unique", False)
        )
        evidence = {
            "normalized_parser_value": normalized_parser,
            "normalized_native_value": normalized_native,
            "native_text_reliable": native_reliable,
            "native_candidate_unique": native_candidate_unique,
            "field_type": _field_type_value(field_type),
            "normalization": [
                "whitespace and punctuation/full-width delimiters only",
                f"{R1_CANONICAL_LIST_SEPARATOR} for port_config",
                f"{R2_STANDALONE_SLASH_COLLAPSE} at whitespace boundaries",
            ],
        }

        if not normalized_native:
            conflict = CriticalConflict(
                conflict_type=ConflictType.CRITICAL_FIELD_CONFLICT,
                parser_value=parser_value,
                native_value=native_pdf_text,
                normalized_parser_value=normalized_parser,
                normalized_native_value=normalized_native,
                differences=(),
            )
            evidence["reason"] = "native_text_missing"
            return CriticalFieldCheckResult(
                original_parser_value=parser_value,
                native_value=native_pdf_text,
                field_type=_coerce_field_type(field_type),
                passed=False,
                conflicts=(conflict,),
                resolved_value=None,
                review_required=True,
                evidence=evidence,
            )

        if normalized_parser == normalized_native:
            evidence["comparison"] = "normalized_equal"
            return CriticalFieldCheckResult(
                original_parser_value=parser_value,
                native_value=native_pdf_text,
                field_type=_coerce_field_type(field_type),
                passed=True,
                conflicts=(),
                resolved_value=parser_value,
                review_required=False,
                evidence=evidence,
            )

        diameter_parser = _diameter_measurement(parser_value)
        diameter_native = _diameter_measurement(native_pdf_text)
        if (
            _is_model_field_type(field_type)
            and diameter_parser is not None
            and diameter_parser == diameter_native
        ):
            evidence["comparison"] = "contextual_diameter_symbol_equivalence"
            evidence["diameter_symbol_rule"] = "DIAMETER_SYMBOL_EQUIVALENCE"
            return CriticalFieldCheckResult(
                original_parser_value=parser_value,
                native_value=native_pdf_text,
                field_type=_coerce_field_type(field_type),
                passed=True,
                conflicts=(),
                resolved_value=parser_value,
                review_required=False,
                evidence=evidence,
            )

        differences = _differences(normalized_parser, normalized_native)
        only_confusables = bool(differences) and all(
            frozenset((parser_char, native_char))
            in self._CONFUSABLE_PAIRS
            for _, parser_char, native_char in differences
        )
        conflict_type = (
            ConflictType.LETTER_DIGIT_CONFUSION
            if only_confusables
            else ConflictType.CRITICAL_FIELD_CONFLICT
        )
        conflict = CriticalConflict(
            conflict_type=conflict_type,
            parser_value=parser_value,
            native_value=native_pdf_text,
            normalized_parser_value=normalized_parser,
            normalized_native_value=normalized_native,
            differences=differences,
        )
        evidence["differences"] = [
            {
                "position": position,
                "parser_char": parser_char,
                "native_char": native_char,
            }
            for position, parser_char, native_char in differences
        ]

        if (
            only_confusables
            and native_reliable
            and native_candidate_unique
            and _is_auto_resolvable_identifier_field_type(field_type)
            and all(
                parser_char == "0" and native_char == "O"
                for _, parser_char, native_char in differences
            )
        ):
            evidence["comparison"] = "unique_native_o_zero_candidate"
            evidence["native_candidate_count"] = 1
            return CriticalFieldCheckResult(
                original_parser_value=parser_value,
                native_value=native_pdf_text,
                field_type=_coerce_field_type(field_type),
                passed=True,
                conflicts=(conflict,),
                resolved_value=native_pdf_text,
                review_required=False,
                evidence=evidence,
                resolution_source="NATIVE_TEXT",
                resolution_reason="LETTER_DIGIT_CONFUSION",
            )

        evidence["comparison"] = (
            "confusable_without_deterministic_resolution"
            if only_confusables
            else "critical_difference"
        )
        return CriticalFieldCheckResult(
            original_parser_value=parser_value,
            native_value=native_pdf_text,
            field_type=_coerce_field_type(field_type),
            passed=False,
            conflicts=(conflict,),
            resolved_value=None,
            review_required=True,
            evidence=evidence,
        )

    def validate_fields(
        self, fields: Sequence[CriticalFieldInput]
    ) -> dict[str, CriticalFieldCheckResult]:
        """Run the optional stage only for explicitly declared fields."""

        return {
            item.field_name: self.validate(
                parser_value=item.parser_value,
                native_pdf_text=item.native_pdf_text,
                field_type=item.field_type,
                context=item.context,
            )
            for item in fields
        }

    @classmethod
    def _normalize(
        cls,
        value: str,
        field_type: CriticalFieldType | str | None = None,
    ) -> str:
        translated = value.translate(cls._PUNCTUATION_TRANSLATION)
        if _is_list_field_type(field_type):
            translated = cls._LIST_SEPARATOR_PATTERN.sub(";", translated)
        # Apply R2 before whitespace removal.  A slash is collapsible only
        # when it is an independent token, never when it is inside 10Gb/s,
        # A/B, or GYTAH-48B/GYTAH-12B.
        if _is_slash_field_type(field_type):
            translated = re.sub(r"\s+/\s+", " ", translated)
        return "".join(character for character in translated if not character.isspace())


def _differences(
    parser_value: str, native_value: str
) -> tuple[tuple[int, str, str], ...]:
    differences: list[tuple[int, str, str]] = []
    for position in range(max(len(parser_value), len(native_value))):
        parser_char = parser_value[position] if position < len(parser_value) else ""
        native_char = native_value[position] if position < len(native_value) else ""
        if parser_char != native_char:
            differences.append((position, parser_char, native_char))
    return tuple(differences)


def _coerce_field_type(
    field_type: CriticalFieldType | str | None,
) -> CriticalFieldType | str | None:
    if field_type is None or isinstance(field_type, CriticalFieldType):
        return field_type
    try:
        return CriticalFieldType(field_type)
    except ValueError:
        return field_type


def _field_type_value(
    field_type: CriticalFieldType | str | None,
) -> str | None:
    if isinstance(field_type, Enum):
        return str(field_type.value)
    return field_type


def _is_list_field_type(
    field_type: CriticalFieldType | str | None,
) -> bool:
    if isinstance(field_type, CriticalFieldType):
        resolved = field_type
    else:
        try:
            resolved = CriticalFieldType(field_type) if field_type else None
        except ValueError:
            resolved = None
    return resolved in CriticalFieldValidator._LIST_FIELD_TYPES


def _is_slash_field_type(
    field_type: CriticalFieldType | str | None,
) -> bool:
    if isinstance(field_type, CriticalFieldType):
        resolved = field_type
    else:
        try:
            resolved = CriticalFieldType(field_type) if field_type else None
        except ValueError:
            resolved = None
    return resolved in CriticalFieldValidator._SLASH_FIELD_TYPES


def _is_model_field_type(
    field_type: CriticalFieldType | str | None,
) -> bool:
    if isinstance(field_type, CriticalFieldType):
        return field_type is CriticalFieldType.MODEL
    return field_type == CriticalFieldType.MODEL.value


def _is_auto_resolvable_identifier_field_type(
    field_type: CriticalFieldType | str | None,
) -> bool:
    value = _field_type_value(field_type)
    return value in {
        CriticalFieldType.MODEL.value,
        CriticalFieldType.PORT_CONFIG.value,
    }


def _diameter_measurement(value: str | None) -> tuple[str, str] | None:
    if not value:
        return None
    match = CriticalFieldValidator._DIAMETER_MEASUREMENT_PATTERN.fullmatch(value)
    if match is None:
        return None
    number, unit = match.groups()
    return number, unit.casefold()


__all__ = [
    "ConflictType",
    "CriticalConflict",
    "CriticalFieldCheckResult",
    "CriticalFieldInput",
    "CriticalFieldType",
    "CriticalFieldValidator",
    "R1_CANONICAL_LIST_SEPARATOR",
    "R2_STANDALONE_SLASH_COLLAPSE",
]
