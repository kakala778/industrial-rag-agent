"""Occurrence identities and outcome types for native-text recovery."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class NativeRecoveryStatus(str, Enum):
    RECOVERY_COMPLETE = "RECOVERY_COMPLETE"
    PARTIAL_REVIEW_REQUIRED = "PARTIAL_REVIEW_REQUIRED"
    NO_RECOVERY = "NO_RECOVERY"


@dataclass(frozen=True)
class EvidenceOccurrence:
    """One token occurrence from native PDF text and its source location."""

    evidence_id: str
    page_number: int
    block_id: str
    occurrence_index: int
    word_index: int | None
    char_start: int | None
    char_end: int | None
    token: str
    bbox: tuple[float, float, float, float] | None
    provenance_reliable: bool = False


@dataclass(frozen=True)
class RecoveryThresholds:
    global_min_tokens: int = 30
    global_min_chars: int = 80
    global_min_ratio: float = 0.35
    image_min_tokens: int = 20
    image_min_chars: int = 60
    image_min_ratio: float = 0.50

    def to_dict(self) -> dict[str, dict[str, int | float]]:
        return {
            "global": {
                "tokens": self.global_min_tokens,
                "chars": self.global_min_chars,
                "ratio": self.global_min_ratio,
            },
            "image": {
                "tokens": self.image_min_tokens,
                "chars": self.image_min_chars,
                "ratio": self.image_min_ratio,
            },
        }


@dataclass(frozen=True)
class CoverageCounts:
    tokens: int
    chars: int
    ratio: float
    denominator_tokens: int
    significant: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return {
            "tokens": self.tokens,
            "chars": self.chars,
            "ratio": self.ratio,
            "denominator_tokens": self.denominator_tokens,
            "significant": self.significant,
        }


@dataclass(frozen=True)
class NativeRecoveryAssessment:
    salvage_triggered: bool
    recovery_status: NativeRecoveryStatus | None
    initial_strong_loss: bool | None
    significant_residual: bool | None
    threshold_status: str
    thresholds: RecoveryThresholds
    global_initial: CoverageCounts | None
    global_residual: CoverageCounts | None
    image_initial: CoverageCounts | None
    image_residual: CoverageCounts | None
    consumed_evidence_ids: tuple[str, ...]
    assessment_available: bool
    auto_restored_count: int
    evidence_link_ambiguity: bool
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "salvage_triggered": self.salvage_triggered,
            "recovery_status": (
                self.recovery_status.value if self.recovery_status is not None else None
            ),
            "initial_strong_loss": self.initial_strong_loss,
            "significant_residual": self.significant_residual,
            "threshold_status": self.threshold_status,
            "thresholds": self.thresholds.to_dict(),
            "global_initial": (
                self.global_initial.to_dict() if self.global_initial else None
            ),
            "global_residual": (
                self.global_residual.to_dict() if self.global_residual else None
            ),
            "image_initial": self.image_initial.to_dict() if self.image_initial else None,
            "image_residual": (
                self.image_residual.to_dict() if self.image_residual else None
            ),
            "consumed_evidence_ids": list(self.consumed_evidence_ids),
            "assessment_available": self.assessment_available,
            "auto_restored_count": self.auto_restored_count,
            "evidence_link_ambiguity": self.evidence_link_ambiguity,
            "reason_codes": list(self.reason_codes),
        }


def _coverage_counts(
    occurrences: Sequence[EvidenceOccurrence],
    *,
    denominator_tokens: int,
    thresholds: RecoveryThresholds,
    image: bool,
) -> CoverageCounts:
    token_count = len(occurrences)
    char_count = sum(len(item.token) for item in occurrences)
    ratio = token_count / denominator_tokens if denominator_tokens else 0.0
    if image:
        significant = (
            token_count >= thresholds.image_min_tokens
            and char_count >= thresholds.image_min_chars
            and ratio >= thresholds.image_min_ratio
        )
    else:
        significant = (
            token_count >= thresholds.global_min_tokens
            and char_count >= thresholds.global_min_chars
            and ratio >= thresholds.global_min_ratio
        )
    return CoverageCounts(
        tokens=token_count,
        chars=char_count,
        ratio=ratio,
        denominator_tokens=denominator_tokens,
        significant=significant,
    )


def assess_native_recovery(
    *,
    salvage_triggered: bool,
    initial_strong_loss: bool | None,
    native_occurrences: Sequence[EvidenceOccurrence],
    mineru_tokens: Sequence[str],
    image_occurrence_ids: frozenset[str],
    auto_restored_count: int,
    candidate_consumed_evidence_ids: Sequence[str],
    assessment_available: bool = True,
    assessment_failure_reason_code: str = "NATIVE_TEXT_EXTRACTION_FAILED",
    evidence_link_ambiguity: bool = False,
    thresholds: RecoveryThresholds = RecoveryThresholds(),
) -> NativeRecoveryAssessment:
    """Assess uncovered native token occurrences without value-wide removal."""

    if not salvage_triggered:
        return NativeRecoveryAssessment(
            salvage_triggered=False,
            recovery_status=None,
            initial_strong_loss=initial_strong_loss,
            significant_residual=None,
            threshold_status="provisional",
            thresholds=thresholds,
            global_initial=None,
            global_residual=None,
            image_initial=None,
            image_residual=None,
            consumed_evidence_ids=(),
            assessment_available=assessment_available,
            auto_restored_count=auto_restored_count,
            evidence_link_ambiguity=False,
            reason_codes=(),
        )

    if not assessment_available:
        failure_reasons = [assessment_failure_reason_code]
        if evidence_link_ambiguity:
            failure_reasons.append("NATIVE_EVIDENCE_LINK_AMBIGUITY")
        return NativeRecoveryAssessment(
            salvage_triggered=True,
            recovery_status=NativeRecoveryStatus.NO_RECOVERY,
            initial_strong_loss=initial_strong_loss,
            significant_residual=None,
            threshold_status="provisional",
            thresholds=thresholds,
            global_initial=None,
            global_residual=None,
            image_initial=None,
            image_residual=None,
            consumed_evidence_ids=(),
            assessment_available=False,
            auto_restored_count=auto_restored_count,
            evidence_link_ambiguity=evidence_link_ambiguity,
            reason_codes=tuple(failure_reasons),
        )

    mineru_counts = Counter(mineru_tokens)
    initial_uncovered: list[EvidenceOccurrence] = []
    for occurrence in native_occurrences:
        if mineru_counts[occurrence.token] > 0:
            mineru_counts[occurrence.token] -= 1
        else:
            initial_uncovered.append(occurrence)

    initial_uncovered_ids = {item.evidence_id for item in initial_uncovered}
    occurrence_by_id: dict[str, list[EvidenceOccurrence]] = {}
    for occurrence in native_occurrences:
        occurrence_by_id.setdefault(occurrence.evidence_id, []).append(occurrence)

    candidate_counts = Counter(candidate_consumed_evidence_ids)
    link_ambiguous = evidence_link_ambiguity
    consumable_ids: set[str] = set()
    if auto_restored_count > 0:
        for evidence_id, candidate_count in candidate_counts.items():
            matches = occurrence_by_id.get(evidence_id, [])
            if candidate_count != 1 or len(matches) != 1:
                link_ambiguous = True
                continue
            occurrence = matches[0]
            if not occurrence.provenance_reliable:
                link_ambiguous = True
                continue
            if evidence_id in initial_uncovered_ids:
                consumable_ids.add(evidence_id)

    consumed_evidence_ids = tuple(
        item.evidence_id
        for item in initial_uncovered
        if item.evidence_id in consumable_ids
    )
    residual_global_occurrences = [
        item
        for item in initial_uncovered
        if item.evidence_id not in consumable_ids
    ]
    initial_image_occurrences = [
        item for item in native_occurrences if item.evidence_id in image_occurrence_ids
    ]
    image_mineru_counts = Counter(mineru_tokens)
    initial_uncovered_image: list[EvidenceOccurrence] = []
    for occurrence in initial_image_occurrences:
        if image_mineru_counts[occurrence.token] > 0:
            image_mineru_counts[occurrence.token] -= 1
        else:
            initial_uncovered_image.append(occurrence)
    residual_image_occurrences = [
        item
        for item in initial_uncovered_image
        if item.evidence_id not in consumable_ids
    ]

    global_initial = _coverage_counts(
        initial_uncovered,
        denominator_tokens=len(native_occurrences),
        thresholds=thresholds,
        image=False,
    )
    global_residual = _coverage_counts(
        residual_global_occurrences,
        denominator_tokens=len(native_occurrences),
        thresholds=thresholds,
        image=False,
    )
    image_initial = _coverage_counts(
        initial_uncovered_image,
        denominator_tokens=len(initial_image_occurrences),
        thresholds=thresholds,
        image=True,
    )
    image_residual = _coverage_counts(
        residual_image_occurrences,
        denominator_tokens=len(initial_image_occurrences),
        thresholds=thresholds,
        image=True,
    )

    significant_residual = (
        None
        if initial_strong_loss is None
        else initial_strong_loss
        and (global_residual.significant or image_residual.significant)
    )
    if auto_restored_count <= 0 or significant_residual is None:
        recovery_status = NativeRecoveryStatus.NO_RECOVERY
    elif significant_residual:
        recovery_status = NativeRecoveryStatus.PARTIAL_REVIEW_REQUIRED
    else:
        recovery_status = NativeRecoveryStatus.RECOVERY_COMPLETE

    reason_codes: list[str] = []
    if significant_residual:
        reason_codes.append("UNRESOLVED_NATIVE_CONTENT")
    if link_ambiguous:
        reason_codes.append("NATIVE_EVIDENCE_LINK_AMBIGUITY")
    if initial_strong_loss is None:
        reason_codes.append("NATIVE_COVERAGE_ASSESSMENT_UNAVAILABLE")

    return NativeRecoveryAssessment(
        salvage_triggered=True,
        recovery_status=recovery_status,
        initial_strong_loss=initial_strong_loss,
        significant_residual=significant_residual,
        threshold_status="provisional",
        thresholds=thresholds,
        global_initial=global_initial,
        global_residual=global_residual,
        image_initial=image_initial,
        image_residual=image_residual,
        consumed_evidence_ids=consumed_evidence_ids,
        assessment_available=assessment_available,
        auto_restored_count=auto_restored_count,
        evidence_link_ambiguity=link_ambiguous,
        reason_codes=tuple(reason_codes),
    )
