"""Conservative detection of text whose table-row ownership is ambiguous."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias


BoundingBoxTuple: TypeAlias = tuple[float, float, float, float]


@dataclass(frozen=True)
class TableRowAnchor:
    """A positioned text anchor known to belong to one table row."""

    row_number: int
    bbox: BoundingBoxTuple


@dataclass(frozen=True)
class TableTextCandidate:
    """Text not yet assigned to a cell, plus neighboring row anchors."""

    table_id: str
    text: str
    bbox: BoundingBoxTuple
    neighboring_rows: tuple[TableRowAnchor, ...]


@dataclass(frozen=True)
class TableLayoutFinding:
    """An auditable signal that row ownership needs human review."""

    table_id: str
    text: str
    bbox: BoundingBoxTuple
    affected_rows: tuple[int, int]
    reason: str
    distances: dict[str, float]
    review_required: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "table_id": self.table_id,
            "text": self.text,
            "bbox": list(self.bbox),
            "affected_rows": list(self.affected_rows),
            "reason": self.reason,
            "distances": self.distances,
            "review_required": self.review_required,
        }


class TableStructureStatus(str, Enum):
    """Table-structure result, distinct from individual field accuracy."""

    PASS = "PASS"
    FAIL = "FAIL"
    UNCERTAIN = "UNCERTAIN"


def assess_table_structure(
    *,
    has_complete_ground_truth: bool,
    confirmed_mismatch: bool,
) -> TableStructureStatus:
    """Require complete structure ground truth for PASS, retain confirmed FAILs."""

    if confirmed_mismatch:
        return TableStructureStatus.FAIL
    if not has_complete_ground_truth:
        return TableStructureStatus.UNCERTAIN
    return TableStructureStatus.PASS


class TableLayoutAmbiguityDetector:
    """Flag an unassigned span sitting in the narrow corridor between rows."""

    def __init__(
        self,
        *,
        max_vertical_gap: float = 10.0,
        max_horizontal_gap: float = 12.0,
    ) -> None:
        if max_vertical_gap < 0 or max_horizontal_gap < 0:
            raise ValueError("ambiguity gaps must not be negative")
        self.max_vertical_gap = max_vertical_gap
        self.max_horizontal_gap = max_horizontal_gap

    def detect(
        self, candidate: TableTextCandidate
    ) -> TableLayoutFinding | None:
        """Return a review finding without assigning or moving the candidate."""

        rows = sorted(candidate.neighboring_rows, key=lambda row: row.row_number)
        for upper, lower in zip(rows, rows[1:]):
            if lower.row_number != upper.row_number + 1:
                continue
            upper_bottom = upper.bbox[3]
            lower_top = lower.bbox[1]
            candidate_top, candidate_bottom = candidate.bbox[1], candidate.bbox[3]
            if not (upper_bottom <= candidate_top <= candidate_bottom <= lower_top):
                continue

            upper_gap = candidate_top - upper_bottom
            lower_gap = lower_top - candidate_bottom
            horizontal_gap = min(
                _horizontal_gap(candidate.bbox, upper.bbox),
                _horizontal_gap(candidate.bbox, lower.bbox),
            )
            if (
                upper_gap <= self.max_vertical_gap
                and lower_gap <= self.max_vertical_gap
                and horizontal_gap <= self.max_horizontal_gap
            ):
                return TableLayoutFinding(
                    table_id=candidate.table_id,
                    text=candidate.text,
                    bbox=candidate.bbox,
                    affected_rows=(upper.row_number, lower.row_number),
                    reason="INTERLEAVED_TEXT_NEAR_ADJACENT_ROWS",
                    distances={
                        "upper_row_vertical_gap": upper_gap,
                        "lower_row_vertical_gap": lower_gap,
                        "nearest_row_horizontal_gap": horizontal_gap,
                    },
                )
        return None

    def detect_many(
        self, candidates: tuple[TableTextCandidate, ...]
    ) -> tuple[TableLayoutFinding, ...]:
        return tuple(
            finding
            for candidate in candidates
            if (finding := self.detect(candidate)) is not None
        )


def _horizontal_gap(
    first: BoundingBoxTuple,
    second: BoundingBoxTuple,
) -> float:
    if first[0] <= second[2] and second[0] <= first[2]:
        return 0.0
    return max(second[0] - first[2], first[0] - second[2], 0.0)


__all__ = [
    "TableLayoutAmbiguityDetector",
    "TableLayoutFinding",
    "TableRowAnchor",
    "TableStructureStatus",
    "TableTextCandidate",
    "assess_table_structure",
]
