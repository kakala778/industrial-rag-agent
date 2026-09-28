"""Configurable quality signals and gate decisions for one parsed page."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Sequence

from .output_reader import MinerUPage
from .probe import PageProbeResult


@dataclass(frozen=True)
class QualityThresholds:
    """Provisional thresholds; they are configuration, not industry standards."""

    min_char_ratio: float = 0.5
    min_numeric_ratio: float = 0.5
    min_token_coverage: float = 0.5
    min_order_similarity: float = 0.8
    reordering_min_coverage: float = 0.8
    max_char_ratio: float = 3.0
    max_duplicate_token_ratio: float = 0.4
    max_repeated_line_ratio: float = 0.4
    near_empty_char_count: int = 1
    status: str = "provisional"

    def to_dict(self) -> dict[str, object]:
        return {
            "min_char_ratio": self.min_char_ratio,
            "min_numeric_ratio": self.min_numeric_ratio,
            "min_token_coverage": self.min_token_coverage,
            "min_order_similarity": self.min_order_similarity,
            "reordering_min_coverage": self.reordering_min_coverage,
            "max_char_ratio": self.max_char_ratio,
            "max_duplicate_token_ratio": self.max_duplicate_token_ratio,
            "max_repeated_line_ratio": self.max_repeated_line_ratio,
            "near_empty_char_count": self.near_empty_char_count,
            "status": self.status,
        }


@dataclass(frozen=True)
class QualityMetrics:
    """Raw quality signals, deliberately not an accuracy/confidence score."""

    char_ratio: float | None
    numeric_ratio: float | None
    token_coverage: float
    token_order_similarity: float
    duplicate_token_ratio: float
    repeated_line_ratio: float

    def to_dict(self) -> dict[str, float | None]:
        return {
            "char_ratio": self.char_ratio,
            "numeric_ratio": self.numeric_ratio,
            "token_coverage": self.token_coverage,
            "token_order_similarity": self.token_order_similarity,
            "duplicate_token_ratio": self.duplicate_token_ratio,
            "repeated_line_ratio": self.repeated_line_ratio,
        }


@dataclass(frozen=True)
class QualityResult:
    """Quality gate output for one page attempt."""

    passed: bool
    flags: tuple[str, ...] = field(default_factory=tuple)
    metrics: QualityMetrics = field(
        default_factory=lambda: QualityMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "flags": list(self.flags),
            "metrics": self.metrics.to_dict(),
        }


class QualityGate:
    """Evaluate configured quality signals without claiming accuracy."""

    def __init__(self, thresholds: QualityThresholds | None = None) -> None:
        self.thresholds = thresholds or QualityThresholds()

    def evaluate(
        self, probe: PageProbeResult, output_page: MinerUPage
    ) -> QualityResult:
        output_text = output_page.text
        native_tokens = _tokenize_sequence(probe.probe_word_sequence)
        output_tokens = _tokenize_text(output_text)
        output_word_tokens = _tokenize_words(output_text)
        metrics = QualityMetrics(
            char_ratio=_ratio(
                _char_count(output_text), probe.native_char_count
            ),
            numeric_ratio=_ratio(
                len(_NUMBER_PATTERN.findall(output_text)),
                probe.native_numeric_token_count,
            ),
            token_coverage=_token_coverage(native_tokens, output_tokens),
            token_order_similarity=_order_similarity(
                native_tokens, output_tokens
            ),
            duplicate_token_ratio=_duplicate_ratio(output_word_tokens),
            repeated_line_ratio=_repeated_line_ratio(output_text),
        )

        flags: list[str] = []
        if (
            probe.native_char_count > 0
            and _char_count(output_text) <= self.thresholds.near_empty_char_count
        ):
            flags.append("near_empty")
        if probe.native_char_count > 0 and (
            metrics.char_ratio is None
            or metrics.char_ratio < self.thresholds.min_char_ratio
            or metrics.token_coverage < self.thresholds.min_token_coverage
        ):
            flags.append("text_coverage_drop")
        if (
            probe.native_numeric_token_count > 0
            and (
                metrics.numeric_ratio is None
                or metrics.numeric_ratio < self.thresholds.min_numeric_ratio
            )
        ):
            flags.append("numeric_coverage_drop")
        if (
            metrics.char_ratio is not None
            and metrics.char_ratio > self.thresholds.max_char_ratio
        ):
            flags.append("abnormal_expansion")
        if (
            metrics.duplicate_token_ratio > self.thresholds.max_duplicate_token_ratio
            or metrics.repeated_line_ratio > self.thresholds.max_repeated_line_ratio
        ):
            flags.append("suspicious_repetition")
        if (
            metrics.token_coverage >= self.thresholds.reordering_min_coverage
            and metrics.token_order_similarity < self.thresholds.min_order_similarity
        ):
            flags.append("suspected_reordering")

        return QualityResult(
            passed=not flags,
            flags=tuple(flags),
            metrics=metrics,
        )


_TOKEN_PATTERN = re.compile(r"[a-z0-9]|[\u4e00-\u9fff]", re.IGNORECASE)
_WORD_TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]", re.IGNORECASE)
_NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)?")


def _tokenize_text(value: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_PATTERN.findall(value)]


def _tokenize_sequence(values: Sequence[str]) -> list[str]:
    return _tokenize_text(" ".join(values))


def _tokenize_words(value: str) -> list[str]:
    return [token.casefold() for token in _WORD_TOKEN_PATTERN.findall(value)]


def _char_count(value: str) -> int:
    return len("".join(value.split()))


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _token_coverage(native: Sequence[str], output: Sequence[str]) -> float:
    if not native:
        return 1.0 if not output else 0.0
    native_counts = Counter(native)
    output_counts = Counter(output)
    matched = sum(
        min(count, output_counts[token]) for token, count in native_counts.items()
    )
    return matched / len(native)


def _order_similarity(native: Sequence[str], output: Sequence[str]) -> float:
    if not native and not output:
        return 1.0
    if not native or not output:
        return 0.0
    return SequenceMatcher(None, native, output, autojunk=False).ratio()


def _duplicate_ratio(tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    return (len(tokens) - len(set(tokens))) / len(tokens)


def _repeated_line_ratio(value: str) -> float:
    lines = [line.strip().casefold() for line in value.splitlines() if line.strip()]
    if not lines:
        return 0.0
    counts = Counter(lines)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(lines)


__all__ = [
    "QualityGate",
    "QualityMetrics",
    "QualityResult",
    "QualityThresholds",
]
