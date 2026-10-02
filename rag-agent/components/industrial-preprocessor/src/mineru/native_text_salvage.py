"""Conservative native-text recovery for image-dominant PDF pages."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Callable, Iterable, Sequence

from src.models.block import Block, BlockType
from src.models.metadata import BoundingBox, JSONValue, SourceReference

from .native_recovery import (
    EvidenceOccurrence,
    NativeRecoveryAssessment,
    NativeRecoveryStatus,
    assess_native_recovery,
)
from .output_reader import MinerUPage


PdfBBox = tuple[float, float, float, float]
_MILEAGE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:D1|D)?K\d{2,3}\+\d{3,4}(?!\d)", re.IGNORECASE
)
_LOCATION_NAME_PATTERN = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9·（）()]{1,32}?"
    r"(?:大桥|隧道|中继站|基站|车站|站|桥)"
)
_LENGTH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])[-–—−]?\s*\d+(?:\.\d+)?\s*m\b", re.IGNORECASE
)
_COVERAGE_TOKEN_PATTERN = re.compile(
    r"[\u3400-\u9fff]|(?:[^\W_]|_)+(?:[.+/\-−–—](?:[^\W_]|_)+)*"
    r"|[+\-−–—]|[<>≤≥=×*%‰°℃μµ¢ΦφØΩ²³]",
    re.UNICODE,
)
_FULLWIDTH_ASCII_TRANSLATION = str.maketrans(
    {
        codepoint: chr(codepoint - 0xFEE0)
        for codepoint in range(0xFF01, 0xFF5F)
    }
    | {0x3000: " "}
)


class SalvageStatus(str, Enum):
    AUTO_RESTORED = "AUTO_RESTORED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


@dataclass(frozen=True)
class NativeWordGeometry:
    """A word from PyMuPDF with offsets in its reconstructed source line."""

    word_index: int
    text: str
    bbox: PdfBBox
    char_start: int
    char_end: int
    provenance_reliable: bool = False


@dataclass(frozen=True)
class NativeTextBlock:
    """Text extracted from the PDF native layer with page-point geometry."""

    text: str
    bbox: PdfBBox
    block_id: str
    word_geometry: tuple[NativeWordGeometry, ...] = ()


@dataclass(frozen=True)
class NativeTextCoverageAssessment:
    """Auditable native-vs-MinerU token coverage and spatial gap metrics."""

    native_char_count: int
    mineru_char_count: int
    native_word_count: int
    mineru_word_count: int
    native_token_coverage: float
    missing_native_word_count: int
    missing_native_char_count: int
    image_block_count: int
    spatial_image_block_count: int
    image_bbox: PdfBBox | None
    image_bboxes: tuple[PdfBBox, ...]
    native_words_inside_image_bbox: int
    uncovered_native_words_inside_image_bbox: int
    uncovered_native_characters_inside_image_bbox: int
    uncovered_native_ratio_inside_image_bbox: float
    strong_native_coverage_gap: bool
    strong_uncovered_text_inside_image: bool
    salvage_eligible: bool
    eligibility_reasons: tuple[str, ...]
    provisional_thresholds: tuple[tuple[str, float | int], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "native_char_count": self.native_char_count,
            "mineru_char_count": self.mineru_char_count,
            "native_word_count": self.native_word_count,
            "mineru_word_count": self.mineru_word_count,
            "native_token_coverage": self.native_token_coverage,
            "missing_native_word_count": self.missing_native_word_count,
            "missing_native_char_count": self.missing_native_char_count,
            "image_block_count": self.image_block_count,
            "spatial_image_block_count": self.spatial_image_block_count,
            "image_bbox": list(self.image_bbox) if self.image_bbox else None,
            "image_bboxes": [list(bbox) for bbox in self.image_bboxes],
            "native_words_inside_image_bbox": self.native_words_inside_image_bbox,
            "uncovered_native_words_inside_image_bbox": (
                self.uncovered_native_words_inside_image_bbox
            ),
            "uncovered_native_characters_inside_image_bbox": (
                self.uncovered_native_characters_inside_image_bbox
            ),
            "uncovered_native_ratio_inside_image_bbox": (
                self.uncovered_native_ratio_inside_image_bbox
            ),
            "strong_native_coverage_gap": self.strong_native_coverage_gap,
            "strong_uncovered_text_inside_image": (
                self.strong_uncovered_text_inside_image
            ),
            "salvage_eligible": self.salvage_eligible,
            "eligibility_reasons": list(self.eligibility_reasons),
            "threshold_status": "provisional",
            "provisional_thresholds": dict(self.provisional_thresholds),
        }


@dataclass(frozen=True)
class NativeSalvageItem:
    """A recovered location record or a reviewable table-header sequence."""

    kind: str
    text: str
    bbox: PdfBBox
    status: SalvageStatus
    reason: str
    source_block_ids: tuple[str, ...]
    evidence: dict[str, object]
    header_cells: tuple[str, ...] = ()
    consumed_evidence_ids: tuple[str, ...] | None = None

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "kind": self.kind,
            "text": self.text,
            "bbox": list(self.bbox),
            "status": self.status.value,
            "reason": self.reason,
            "source_block_ids": list(self.source_block_ids),
            "evidence": self.evidence,
            "header_cells": list(self.header_cells),
        }
        if (
            self.status is SalvageStatus.AUTO_RESTORED
            and self.consumed_evidence_ids is not None
        ):
            value["consumed_evidence_ids"] = list(self.consumed_evidence_ids)
        return value


@dataclass(frozen=True)
class NativeTextSalvageResult:
    """Page-local supplement; it never replaces MinerU output or table data."""

    page_number: int
    triggered: bool
    high_res_visual_required: bool
    items: tuple[NativeSalvageItem, ...]
    native_block_count: int
    reason: str
    error_message: str | None = None
    coverage_assessment: NativeTextCoverageAssessment | None = None
    recovery_assessment: NativeRecoveryAssessment | None = None

    @property
    def salvage_triggered(self) -> bool:
        return self.triggered

    @property
    def recovery_status(self) -> NativeRecoveryStatus | None:
        return (
            self.recovery_assessment.recovery_status
            if self.recovery_assessment is not None
            else None
        )

    @property
    def auto_restored_items(self) -> tuple[NativeSalvageItem, ...]:
        return tuple(
            item for item in self.items if item.status is SalvageStatus.AUTO_RESTORED
        )

    @property
    def review_items(self) -> tuple[NativeSalvageItem, ...]:
        return tuple(
            item for item in self.items if item.status is SalvageStatus.REVIEW_REQUIRED
        )

    @property
    def requires_review(self) -> bool:
        return bool(
            self.review_items
            or self.error_message
            or (self.triggered and self.native_block_count == 0)
            or (
                self.recovery_assessment is not None
                and self.recovery_assessment.reason_codes
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "page_number": self.page_number,
            "salvage_triggered": self.triggered,
            "triggered": self.triggered,
            "recovery_status": (
                self.recovery_status.value if self.recovery_status is not None else None
            ),
            "recovery_assessment": (
                self.recovery_assessment.to_dict()
                if self.recovery_assessment is not None
                else None
            ),
            "high_res_visual_required": self.high_res_visual_required,
            "native_block_count": self.native_block_count,
            "reason": self.reason,
            "error_message": self.error_message,
            "coverage_assessment": (
                self.coverage_assessment.to_dict()
                if self.coverage_assessment is not None
                else None
            ),
            "auto_restored_count": len(self.auto_restored_items),
            "review_required_count": len(self.review_items),
            "items": [item.to_dict() for item in self.items],
        }

    def to_document_blocks(self, document_id: str) -> list[Block]:
        """Project evidence into the existing flexible unified block model."""

        blocks: list[Block] = []
        for index, item in enumerate(self.items, start=1):
            x0, y0, x1, y1 = item.bbox
            if item.kind == "TABLE_HEADER_ORDER":
                block_type = BlockType.TABLE
                content: JSONValue = {
                    "headers": list(item.header_cells),
                    "status": item.status.value,
                    "reason": item.reason,
                }
            else:
                block_type = BlockType.TEXT
                content = item.text
            blocks.append(
                Block(
                    block_id=f"native-p{self.page_number}-supplement-{index}",
                    type=block_type,
                    content=content,
                    page_number=self.page_number,
                    source=SourceReference(
                        document_id=document_id,
                        page_number=self.page_number,
                        source_id="pdf-native-text",
                        locator="pymupdf-word-bbox",
                        parser_block_id=",".join(item.source_block_ids),
                        extra={
                            "source_type": "PDF_NATIVE_TEXT",
                            "salvage_status": item.status.value,
                            "review_required": (
                                item.status is SalvageStatus.REVIEW_REQUIRED
                            ),
                            "reason": item.reason,
                            "evidence": item.evidence,
                        },
                    ),
                    bbox=BoundingBox(
                        x=x0,
                        y=y0,
                        width=max(0.0, x1 - x0),
                        height=max(0.0, y1 - y0),
                    ),
                )
            )
        return blocks

    def to_markdown_addendum(self) -> str:
        """Render an explicitly separate supplement without changing exporters."""

        if not self.items:
            return ""
        lines = ["## 原 PDF 原生文字补充", ""]
        for item in self.items:
            label = item.status.value
            lines.append(f"- [{label}] {item.text}")
            if item.status is SalvageStatus.REVIEW_REQUIRED:
                lines.append(f"  - reason: `{item.reason}`")
        return "\n".join(lines) + "\n"


class NativeTextSalvager:
    """Recover only unique location callouts and repeated header sequences."""

    def __init__(
        self,
        *,
        min_image_area_ratio: float = 0.55,
        min_missing_native_words_for_global_gap: int = 30,
        min_missing_native_characters_for_global_gap: int = 80,
        min_missing_native_ratio_for_global_gap: float = 0.35,
        min_native_words_inside_image_bbox: int = 25,
        min_uncovered_native_words_inside_image_bbox: int = 20,
        min_uncovered_native_characters_inside_image_bbox: int = 60,
        min_uncovered_native_ratio_inside_image_bbox: float = 0.50,
        max_neighbor_distance: float = 32.0,
        header_band_tolerance: float = 28.0,
        header_alignment_tolerance: float = 20.0,
        block_extractor: Callable[[str | Path, int], Sequence[NativeTextBlock]] | None = None,
    ) -> None:
        if not 0.0 <= min_image_area_ratio <= 1.0:
            raise ValueError("min_image_area_ratio must be between 0 and 1")
        count_thresholds = (
            min_missing_native_words_for_global_gap,
            min_missing_native_characters_for_global_gap,
            min_native_words_inside_image_bbox,
            min_uncovered_native_words_inside_image_bbox,
            min_uncovered_native_characters_inside_image_bbox,
        )
        if any(value < 0 for value in count_thresholds) or max_neighbor_distance < 0:
            raise ValueError("coverage counts and neighbor distance must not be negative")
        ratio_thresholds = (
            min_missing_native_ratio_for_global_gap,
            min_uncovered_native_ratio_inside_image_bbox,
        )
        if any(not 0.0 <= value <= 1.0 for value in ratio_thresholds):
            raise ValueError("coverage ratio thresholds must be between 0 and 1")
        self.min_image_area_ratio = min_image_area_ratio
        self.min_missing_native_words_for_global_gap = (
            min_missing_native_words_for_global_gap
        )
        self.min_missing_native_characters_for_global_gap = (
            min_missing_native_characters_for_global_gap
        )
        self.min_missing_native_ratio_for_global_gap = (
            min_missing_native_ratio_for_global_gap
        )
        self.min_native_words_inside_image_bbox = min_native_words_inside_image_bbox
        self.min_uncovered_native_words_inside_image_bbox = (
            min_uncovered_native_words_inside_image_bbox
        )
        self.min_uncovered_native_characters_inside_image_bbox = (
            min_uncovered_native_characters_inside_image_bbox
        )
        self.min_uncovered_native_ratio_inside_image_bbox = (
            min_uncovered_native_ratio_inside_image_bbox
        )
        self.max_neighbor_distance = max_neighbor_distance
        self.header_band_tolerance = header_band_tolerance
        self.header_alignment_tolerance = header_alignment_tolerance
        self._block_extractor = block_extractor or self.extract_pdf_page_blocks

    def salvage_pdf_page(
        self,
        pdf_path: str | Path,
        page_number: int,
        mineru_page: MinerUPage,
        *,
        page_width: float,
        page_height: float,
    ) -> NativeTextSalvageResult:
        """Extract native text and assess coverage before entering salvage rules."""

        try:
            native_blocks = self._block_extractor(pdf_path, page_number)
        except Exception as exc:
            has_large_image_region = self._has_large_image_region(
                mineru_page,
                page_width=page_width,
                page_height=page_height,
            )
            return NativeTextSalvageResult(
                page_number=page_number,
                triggered=has_large_image_region,
                high_res_visual_required=has_large_image_region,
                items=(),
                native_block_count=0,
                reason=(
                    "native_text_extraction_failed"
                    if has_large_image_region
                    else "native_text_extraction_unavailable_without_image_signal"
                ),
                error_message=(
                    f"{type(exc).__name__}: {exc}"
                    if has_large_image_region
                    else None
                ),
                recovery_assessment=assess_native_recovery(
                    salvage_triggered=has_large_image_region,
                    initial_strong_loss=None,
                    native_occurrences=(),
                    mineru_tokens=(),
                    image_occurrence_ids=frozenset(),
                    auto_restored_count=0,
                    candidate_consumed_evidence_ids=(),
                    assessment_available=False,
                ),
            )
        return self.salvage_blocks(
            page_number=page_number,
            blocks=native_blocks,
            mineru_page=mineru_page,
            page_width=page_width,
            page_height=page_height,
        )

    def salvage_blocks(
        self,
        *,
        page_number: int,
        blocks: Sequence[NativeTextBlock],
        mineru_page: MinerUPage,
        page_width: float,
        page_height: float,
    ) -> NativeTextSalvageResult:
        """Pure recovery path for testable native word/block evidence."""

        native_blocks = tuple(block for block in blocks if block.text.strip())
        try:
            assessment = self.assess_coverage(
                native_blocks,
                mineru_page,
                page_width=page_width,
                page_height=page_height,
            )
        except Exception as exc:
            if not self._has_large_image_region(
                mineru_page,
                page_width=page_width,
                page_height=page_height,
            ):
                raise
            return NativeTextSalvageResult(
                page_number=page_number,
                triggered=True,
                high_res_visual_required=True,
                items=(),
                native_block_count=len(native_blocks),
                reason="native_coverage_assessment_failed",
                error_message=f"{type(exc).__name__}: {exc}",
                recovery_assessment=assess_native_recovery(
                    salvage_triggered=True,
                    initial_strong_loss=None,
                    native_occurrences=(),
                    mineru_tokens=(),
                    image_occurrence_ids=frozenset(),
                    auto_restored_count=0,
                    candidate_consumed_evidence_ids=(),
                    assessment_available=False,
                    assessment_failure_reason_code=(
                        "NATIVE_COVERAGE_ASSESSMENT_FAILED"
                    ),
                ),
            )
        evidence_occurrences = build_evidence_occurrences(page_number, native_blocks)
        mineru_tokens = tuple(
            token
            for block in mineru_page.blocks
            if block.block_type.casefold() != "image"
            for token in _coverage_tokens(block.text)
        )
        image_occurrence_ids = self._image_occurrence_ids(
            evidence_occurrences,
            native_blocks,
            assessment,
            page_width=page_width,
            page_height=page_height,
        )
        initial_strong_loss = (
            assessment.strong_native_coverage_gap
            or assessment.strong_uncovered_text_inside_image
        )
        if not assessment.salvage_eligible:
            return NativeTextSalvageResult(
                page_number=page_number,
                triggered=False,
                high_res_visual_required=False,
                items=(),
                native_block_count=len(native_blocks),
                reason=(
                    "native_text_layer_empty"
                    if assessment.native_word_count == 0
                    else "native_text_coverage_sufficient"
                ),
                coverage_assessment=assessment,
                recovery_assessment=assess_native_recovery(
                    salvage_triggered=False,
                    initial_strong_loss=initial_strong_loss,
                    native_occurrences=evidence_occurrences,
                    mineru_tokens=mineru_tokens,
                    image_occurrence_ids=image_occurrence_ids,
                    auto_restored_count=0,
                    candidate_consumed_evidence_ids=(),
                ),
            )

        items = [
            *self._location_items(
                native_blocks,
                page_number=page_number,
                evidence_occurrences=evidence_occurrences,
            ),
            *self._header_items(native_blocks),
        ]
        items.sort(key=lambda item: (item.bbox[1], item.bbox[0], item.kind))
        auto_restored_count = sum(
            item.status is SalvageStatus.AUTO_RESTORED for item in items
        )
        candidate_ids = tuple(
            evidence_id
            for item in items
            if item.status is SalvageStatus.AUTO_RESTORED
            and item.consumed_evidence_ids is not None
            for evidence_id in item.consumed_evidence_ids
        )
        recovery_assessment = assess_native_recovery(
            salvage_triggered=True,
            initial_strong_loss=initial_strong_loss,
            native_occurrences=evidence_occurrences,
            mineru_tokens=mineru_tokens,
            image_occurrence_ids=image_occurrence_ids,
            auto_restored_count=auto_restored_count,
            candidate_consumed_evidence_ids=candidate_ids,
        )
        consumed_ids = set(recovery_assessment.consumed_evidence_ids)
        items = [
            replace(
                item,
                consumed_evidence_ids=tuple(
                    evidence_id
                    for evidence_id in (item.consumed_evidence_ids or ())
                    if evidence_id in consumed_ids
                ),
            )
            if item.status is SalvageStatus.AUTO_RESTORED
            else item
            for item in items
        ]
        return NativeTextSalvageResult(
            page_number=page_number,
            triggered=True,
            high_res_visual_required=assessment.strong_uncovered_text_inside_image,
            items=tuple(items),
            native_block_count=len(native_blocks),
            reason="+".join(assessment.eligibility_reasons),
            coverage_assessment=assessment,
            recovery_assessment=recovery_assessment,
        )

    def _image_occurrence_ids(
        self,
        occurrences: Sequence[EvidenceOccurrence],
        blocks: Sequence[NativeTextBlock],
        assessment: NativeTextCoverageAssessment,
        *,
        page_width: float,
        page_height: float,
    ) -> frozenset[str]:
        large_image_bboxes = tuple(
            bbox
            for bbox in assessment.image_bboxes
            if self._image_area_ratio(bbox, page_width, page_height)
            >= self.min_image_area_ratio
        )
        blocks_by_id: dict[str, list[NativeTextBlock]] = {}
        for block in blocks:
            blocks_by_id.setdefault(block.block_id, []).append(block)
        return frozenset(
            occurrence.evidence_id
            for occurrence in occurrences
            if len(blocks_by_id.get(occurrence.block_id, ())) == 1
            and any(
                _bbox_center_inside(
                    blocks_by_id[occurrence.block_id][0].bbox,
                    image_bbox,
                    page_width=page_width,
                    page_height=page_height,
                )
                for image_bbox in large_image_bboxes
            )
        )

    def assess_coverage(
        self,
        blocks: Sequence[NativeTextBlock],
        mineru_page: MinerUPage,
        *,
        page_width: float,
        page_height: float,
    ) -> NativeTextCoverageAssessment:
        """Compare native text tokens to MinerU text/table tokens and image regions."""

        native_blocks = tuple(block for block in blocks if block.text.strip())
        native_tokens_by_block = tuple(
            (block, _coverage_tokens(block.text)) for block in native_blocks
        )
        native_tokens = [
            token for _, tokens in native_tokens_by_block for token in tokens
        ]
        mineru_text_blocks = tuple(
            block
            for block in mineru_page.blocks
            if block.block_type.casefold() != "image"
        )
        mineru_text = "\n".join(block.text for block in mineru_text_blocks)
        mineru_tokens = _coverage_tokens(mineru_text)
        missing_tokens = _uncovered_tokens(native_tokens, mineru_tokens)
        native_word_count = len(native_tokens)
        missing_word_count = len(missing_tokens)
        missing_char_count = sum(len(token) for token in missing_tokens)
        missing_ratio = missing_word_count / native_word_count if native_word_count else 0.0

        valid_image_bboxes = tuple(
            block.bbox
            for block in mineru_page.image_blocks
            if block.bbox is not None
        )
        large_image_bboxes = tuple(
            bbox
            for bbox in valid_image_bboxes
            if self._image_area_ratio(bbox, page_width, page_height)
            >= self.min_image_area_ratio
        )
        image_bbox = max(
            valid_image_bboxes,
            key=lambda bbox: self._image_area_ratio(bbox, page_width, page_height),
            default=None,
        )
        inside_tokens: list[str] = []
        for block, tokens in native_tokens_by_block:
            if tokens and any(
                _bbox_center_inside(
                    block.bbox,
                    image_bbox,
                    page_width=page_width,
                    page_height=page_height,
                )
                for image_bbox in large_image_bboxes
            ):
                inside_tokens.extend(tokens)
        uncovered_inside_tokens = _uncovered_tokens(inside_tokens, mineru_tokens)
        inside_word_count = len(inside_tokens)
        uncovered_inside_word_count = len(uncovered_inside_tokens)
        uncovered_inside_char_count = sum(len(token) for token in uncovered_inside_tokens)
        uncovered_inside_ratio = (
            uncovered_inside_word_count / inside_word_count
            if inside_word_count
            else 0.0
        )
        strong_global_gap = (
            missing_word_count >= self.min_missing_native_words_for_global_gap
            and missing_char_count >= self.min_missing_native_characters_for_global_gap
            and missing_ratio >= self.min_missing_native_ratio_for_global_gap
        )
        strong_image_gap = (
            bool(large_image_bboxes)
            and inside_word_count >= self.min_native_words_inside_image_bbox
            and uncovered_inside_word_count
            >= self.min_uncovered_native_words_inside_image_bbox
            and uncovered_inside_char_count
            >= self.min_uncovered_native_characters_inside_image_bbox
            and uncovered_inside_ratio
            >= self.min_uncovered_native_ratio_inside_image_bbox
        )
        reasons = tuple(
            reason
            for condition, reason in (
                (strong_global_gap, "STRONG_NATIVE_COVERAGE_GAP"),
                (
                    strong_image_gap,
                    "STRONG_UNCOVERED_NATIVE_TEXT_INSIDE_IMAGE",
                ),
            )
            if condition
        )
        threshold_values: tuple[tuple[str, float | int], ...] = (
            ("min_image_area_ratio", self.min_image_area_ratio),
            (
                "min_missing_native_words_for_global_gap",
                self.min_missing_native_words_for_global_gap,
            ),
            (
                "min_missing_native_characters_for_global_gap",
                self.min_missing_native_characters_for_global_gap,
            ),
            (
                "min_missing_native_ratio_for_global_gap",
                self.min_missing_native_ratio_for_global_gap,
            ),
            (
                "min_native_words_inside_image_bbox",
                self.min_native_words_inside_image_bbox,
            ),
            (
                "min_uncovered_native_words_inside_image_bbox",
                self.min_uncovered_native_words_inside_image_bbox,
            ),
            (
                "min_uncovered_native_characters_inside_image_bbox",
                self.min_uncovered_native_characters_inside_image_bbox,
            ),
            (
                "min_uncovered_native_ratio_inside_image_bbox",
                self.min_uncovered_native_ratio_inside_image_bbox,
            ),
        )
        return NativeTextCoverageAssessment(
            native_char_count=sum(_non_whitespace_count(block.text) for block in native_blocks),
            mineru_char_count=sum(
                _non_whitespace_count(block.text) for block in mineru_text_blocks
            ),
            native_word_count=native_word_count,
            mineru_word_count=len(mineru_tokens),
            native_token_coverage=(
                1.0 - missing_ratio if native_word_count else 0.0
            ),
            missing_native_word_count=missing_word_count,
            missing_native_char_count=missing_char_count,
            image_block_count=len(mineru_page.image_blocks),
            spatial_image_block_count=len(large_image_bboxes),
            image_bbox=image_bbox,
            image_bboxes=valid_image_bboxes,
            native_words_inside_image_bbox=inside_word_count,
            uncovered_native_words_inside_image_bbox=uncovered_inside_word_count,
            uncovered_native_characters_inside_image_bbox=uncovered_inside_char_count,
            uncovered_native_ratio_inside_image_bbox=uncovered_inside_ratio,
            strong_native_coverage_gap=strong_global_gap,
            strong_uncovered_text_inside_image=strong_image_gap,
            salvage_eligible=bool(native_word_count and reasons),
            eligibility_reasons=reasons,
            provisional_thresholds=threshold_values,
        )

    def _has_large_image_region(
        self,
        mineru_page: MinerUPage,
        *,
        page_width: float,
        page_height: float,
    ) -> bool:
        return any(
            block.bbox is not None
            and self._image_area_ratio(block.bbox, page_width, page_height)
            >= self.min_image_area_ratio
            for block in mineru_page.image_blocks
        )

    @staticmethod
    def _image_area_ratio(
        bbox: PdfBBox,
        page_width: float,
        page_height: float,
    ) -> float:
        x0, y0, x1, y1 = bbox
        if all(-0.05 <= value <= 1.05 for value in bbox):
            width, height = max(0.0, x1 - x0), max(0.0, y1 - y0)
        elif page_width > 0 and page_height > 0:
            width = max(0.0, x1 - x0) / page_width
            height = max(0.0, y1 - y0) / page_height
        else:
            return 0.0
        return min(1.0, width * height)

    def _location_items(
        self,
        blocks: Sequence[NativeTextBlock],
        *,
        page_number: int,
        evidence_occurrences: Sequence[EvidenceOccurrence],
    ) -> list[NativeSalvageItem]:
        occurrences: list[tuple[NativeTextBlock, re.Match[str]]] = [
            (block, match)
            for block in blocks
            for match in _MILEAGE_PATTERN.finditer(block.text)
        ]
        counts: dict[str, int] = {}
        for _, match in occurrences:
            key = match.group(0).casefold()
            counts[key] = counts.get(key, 0) + 1

        items: list[NativeSalvageItem] = []
        for anchor_block, mileage_match in occurrences:
            anchor = mileage_match.group(0)
            nearby_names: list[tuple[NativeTextBlock, str, float]] = []
            for block in blocks:
                distance = _bbox_distance(anchor_block.bbox, block.bbox)
                if distance > self.max_neighbor_distance:
                    continue
                nearby_names.extend(
                    (block, match.group(0), distance)
                    for match in _LOCATION_NAME_PATTERN.finditer(block.text)
                )
            # Deduplicate an identical name repeated through overlapping spans.
            nearby_names = _unique_name_matches(nearby_names)
            if not nearby_names:
                continue

            nearby_lengths: list[tuple[NativeTextBlock, str, float]] = []
            for block in blocks:
                distance = _bbox_distance(anchor_block.bbox, block.bbox)
                if distance <= self.max_neighbor_distance:
                    nearby_lengths.extend(
                        (block, match.group(0).strip(), distance)
                        for match in _LENGTH_PATTERN.finditer(block.text)
                    )
            nearby_lengths = _unique_text_matches(nearby_lengths)
            if anchor.casefold() in counts and counts[anchor.casefold()] == 1:
                anchor_unique = True
            else:
                anchor_unique = False
            is_unique = anchor_unique and len(nearby_names) == 1 and len(nearby_lengths) <= 1
            name_block, name, name_distance = min(
                nearby_names,
                key=lambda item: (item[2], item[0].bbox[1], item[0].bbox[0]),
            )
            length_item = min(
                nearby_lengths,
                key=lambda item: item[2],
                default=None,
            )
            native_spans: list[dict[str, object]] = []
            candidate_evidence_ids: list[str] = []
            name_matches = [
                match
                for match in _LOCATION_NAME_PATTERN.finditer(name_block.text)
                if match.group(0) == name
            ]
            if len(name_matches) == 1:
                match = name_matches[0]
                native_spans.append(
                    {
                        "role": "location_name",
                        "block_id": name_block.block_id,
                        "char_start": match.start(),
                        "char_end": match.end(),
                    }
                )
                candidate_evidence_ids.extend(
                    _evidence_ids_for_span(
                        evidence_occurrences,
                        page_number=page_number,
                        block_id=name_block.block_id,
                        char_start=match.start(),
                        char_end=match.end(),
                    )
                )
            native_spans.append(
                {
                    "role": "mileage_anchor",
                    "block_id": anchor_block.block_id,
                    "char_start": mileage_match.start(),
                    "char_end": mileage_match.end(),
                }
            )
            candidate_evidence_ids.extend(
                _evidence_ids_for_span(
                    evidence_occurrences,
                    page_number=page_number,
                    block_id=anchor_block.block_id,
                    char_start=mileage_match.start(),
                    char_end=mileage_match.end(),
                )
            )
            length_added = False
            value_parts = [name, anchor]
            selected_blocks = [name_block, anchor_block]
            if length_item is not None:
                length_block, length, _ = length_item
                if length not in anchor:
                    value_parts.append(length)
                    length_added = True
                selected_blocks.append(length_block)
            if length_item is not None and length_added:
                length_block, length, _ = length_item
                length_matches = [
                    match
                    for match in _LENGTH_PATTERN.finditer(length_block.text)
                    if match.group(0).strip() == length
                ]
                if len(length_matches) == 1:
                    match = length_matches[0]
                    native_spans.append(
                        {
                            "role": "optional_length",
                            "block_id": length_block.block_id,
                            "char_start": match.start(),
                            "char_end": match.end(),
                        }
                    )
                    candidate_evidence_ids.extend(
                        _evidence_ids_for_span(
                            evidence_occurrences,
                            page_number=page_number,
                            block_id=length_block.block_id,
                            char_start=match.start(),
                            char_end=match.end(),
                        )
                    )
            bbox = _union_bbox(block.bbox for block in selected_blocks)
            reason = (
                "UNIQUE_LOCATION_ANCHOR_AND_NEARBY_LABEL"
                if is_unique
                else "non_unique_location_evidence"
            )
            status = (
                SalvageStatus.AUTO_RESTORED
                if is_unique
                else SalvageStatus.REVIEW_REQUIRED
            )
            items.append(
                NativeSalvageItem(
                    kind="LOCATION_RECORD",
                    text=" ".join(value_parts),
                    bbox=bbox,
                    status=status,
                    reason=reason,
                    source_block_ids=tuple(
                        dict.fromkeys(block.block_id for block in selected_blocks)
                    ),
                    evidence={
                        "mileage_anchor": anchor,
                        "anchor_occurrences": counts[anchor.casefold()],
                        "nearby_location_candidates": [
                            {"text": value, "distance": distance, "bbox": list(block.bbox)}
                            for block, value, distance in nearby_names
                        ],
                        "nearby_length_candidates": [
                            {"text": value, "distance": distance, "bbox": list(block.bbox)}
                            for block, value, distance in nearby_lengths
                        ],
                        "selected_name_distance": name_distance,
                        "native_geometry_unit": "pdf_points",
                        "native_text_spans": native_spans,
                    },
                    consumed_evidence_ids=(
                        tuple(dict.fromkeys(candidate_evidence_ids))
                        if status is SalvageStatus.AUTO_RESTORED
                        else None
                    ),
                )
            )
        return items

    def _header_items(
        self,
        blocks: Sequence[NativeTextBlock],
    ) -> list[NativeSalvageItem]:
        bands = _text_bands(blocks, self.header_band_tolerance)
        candidates = [
            band
            for band in bands
            if len(band) >= 3
            and all(not any(character.isdigit() for character in block.text) for block in band)
            and any(re.search(r"[A-Za-z\u4e00-\u9fff]", block.text) for block in band)
        ]
        for first_index, first in enumerate(candidates):
            first_cells = tuple(block.text.strip() for block in first)
            for second in candidates[first_index + 1 :]:
                second_cells = tuple(block.text.strip() for block in second)
                if first_cells != second_cells:
                    continue
                aligned = all(
                    abs(_x_center(left.bbox) - _x_center(right.bbox))
                    <= self.header_alignment_tolerance
                    for left, right in zip(first, second)
                )
                if not aligned:
                    continue
                all_blocks = (*first, *second)
                return [
                    NativeSalvageItem(
                        kind="TABLE_HEADER_ORDER",
                        text="；".join(first_cells),
                        bbox=_union_bbox(block.bbox for block in all_blocks),
                        status=SalvageStatus.REVIEW_REQUIRED,
                        reason="multiple_spatial_instances_require_table_assignment",
                        source_block_ids=tuple(
                            block.block_id for block in all_blocks
                        ),
                        evidence={
                            "instances": [
                                {
                                    "bbox": list(_union_bbox(block.bbox for block in band)),
                                    "cells": [block.text for block in band],
                                }
                                for band in (first, second)
                            ],
                            "native_geometry_unit": "pdf_points",
                            "table_numeric_data_salvaged": False,
                        },
                        header_cells=first_cells,
                    )
                ]
                break
        return []

    @staticmethod
    def extract_pdf_page_blocks(
        pdf_path: str | Path,
        page_number: int,
    ) -> tuple[NativeTextBlock, ...]:
        """Read native PDF words and retain their source geometry."""

        try:
            import pymupdf
        except ImportError:
            import fitz as pymupdf

        document = pymupdf.open(str(pdf_path))
        try:
            if page_number < 1 or page_number > len(document):
                raise IndexError(f"PDF page number out of range: {page_number}")
            page = document[page_number - 1]
            words = page.get_text("words", sort=False)
            grouped: dict[tuple[int, int], list[tuple[int, str, PdfBBox]]] = {}
            for word in words:
                x0, y0, x1, y1, text, block_no, line_no, word_no = word[:8]
                grouped.setdefault((int(block_no), int(line_no)), []).append(
                    (int(word_no), str(text), (float(x0), float(y0), float(x1), float(y1)))
                )
            blocks: list[NativeTextBlock] = []
            for (block_no, line_no), line_words in sorted(grouped.items()):
                line_words.sort(key=lambda item: item[0])
                boxes = [item[2] for item in line_words]
                word_geometry: list[NativeWordGeometry] = []
                char_position = 0
                for position, (word_index, text, bbox) in enumerate(line_words):
                    if position:
                        char_position += 1
                    char_start = char_position
                    char_position += len(text)
                    word_geometry.append(
                        NativeWordGeometry(
                            word_index=word_index,
                            text=text,
                            bbox=bbox,
                            char_start=char_start,
                            char_end=char_position,
                            provenance_reliable=True,
                        )
                    )
                blocks.append(
                    NativeTextBlock(
                        text=" ".join(item[1] for item in line_words),
                        bbox=_union_bbox(boxes),
                        block_id=f"p{page_number}-b{block_no}-l{line_no}",
                        word_geometry=tuple(word_geometry),
                    )
                )
            return tuple(blocks)
        finally:
            document.close()


def _unique_name_matches(
    matches: Sequence[tuple[NativeTextBlock, str, float]],
) -> list[tuple[NativeTextBlock, str, float]]:
    unique: dict[tuple[str, str], tuple[NativeTextBlock, str, float]] = {}
    for block, text, distance in matches:
        unique.setdefault((block.block_id, text), (block, text, distance))
    return list(unique.values())


def _coverage_tokens(value: str) -> tuple[str, ...]:
    """Keep semantic letters, numbers, signs, and units; normalize only fullwidth ASCII."""

    return tuple(_COVERAGE_TOKEN_PATTERN.findall(value.translate(_FULLWIDTH_ASCII_TRANSLATION)))


def _evidence_ids_for_span(
    occurrences: Sequence[EvidenceOccurrence],
    *,
    page_number: int,
    block_id: str,
    char_start: int,
    char_end: int,
) -> tuple[str, ...]:
    id_counts = Counter(item.evidence_id for item in occurrences)
    return tuple(
        dict.fromkeys(
            item.evidence_id
            for item in occurrences
            if item.page_number == page_number
            and item.block_id == block_id
            and item.char_start is not None
            and item.char_end is not None
            and char_start <= item.char_start
            and item.char_end <= char_end
            and item.word_index is not None
            and item.bbox is not None
            and item.provenance_reliable
            and id_counts[item.evidence_id] == 1
        )
    )


def build_evidence_occurrences(
    page_number: int,
    blocks: Sequence[NativeTextBlock],
) -> tuple[EvidenceOccurrence, ...]:
    """Build stable token identities without treating identity as provenance."""

    occurrences: list[EvidenceOccurrence] = []
    for block in blocks:
        normalized_text = block.text.translate(_FULLWIDTH_ASCII_TRANSLATION)
        for occurrence_index, match in enumerate(
            _COVERAGE_TOKEN_PATTERN.finditer(normalized_text)
        ):
            token = match.group(0)
            char_start, char_end = match.span()
            matching_words = [
                word
                for word in block.word_geometry
                if word.char_start <= char_start and char_end <= word.char_end
            ]
            word = matching_words[0] if len(matching_words) == 1 else None
            bbox = word.bbox if word is not None else block.bbox
            word_index = word.word_index if word is not None else None
            provenance_reliable = bool(
                word is not None
                and word.provenance_reliable
                and normalized_text[word.char_start : word.char_end]
                == word.text.translate(_FULLWIDTH_ASCII_TRANSLATION)
            )
            identity = json.dumps(
                [
                    page_number,
                    block.block_id,
                    occurrence_index,
                    word_index,
                    char_start,
                    char_end,
                    token,
                    bbox,
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            evidence_id = hashlib.sha256(identity).hexdigest()
            occurrences.append(
                EvidenceOccurrence(
                    evidence_id=evidence_id,
                    page_number=page_number,
                    block_id=block.block_id,
                    occurrence_index=occurrence_index,
                    word_index=word_index,
                    char_start=char_start,
                    char_end=char_end,
                    token=token,
                    bbox=bbox,
                    provenance_reliable=provenance_reliable,
                )
            )
    return tuple(occurrences)


def _uncovered_tokens(
    native_tokens: Sequence[str],
    mineru_tokens: Sequence[str],
) -> tuple[str, ...]:
    available = Counter(mineru_tokens)
    missing: list[str] = []
    for token in native_tokens:
        if available[token] > 0:
            available[token] -= 1
        else:
            missing.append(token)
    return tuple(missing)


def _non_whitespace_count(value: str) -> int:
    return sum(not character.isspace() for character in value)


def _bbox_center_inside(
    native_bbox: PdfBBox,
    image_bbox: PdfBBox,
    *,
    page_width: float,
    page_height: float,
) -> bool:
    image_points = _bbox_to_page_points(
        image_bbox,
        page_width=page_width,
        page_height=page_height,
    )
    if image_points is None:
        return False
    center_x = (native_bbox[0] + native_bbox[2]) / 2.0
    center_y = (native_bbox[1] + native_bbox[3]) / 2.0
    return (
        image_points[0] <= center_x <= image_points[2]
        and image_points[1] <= center_y <= image_points[3]
    )


def _bbox_to_page_points(
    bbox: PdfBBox,
    *,
    page_width: float,
    page_height: float,
) -> PdfBBox | None:
    if all(-0.05 <= value <= 1.05 for value in bbox):
        if page_width <= 0 or page_height <= 0:
            return None
        return (
            bbox[0] * page_width,
            bbox[1] * page_height,
            bbox[2] * page_width,
            bbox[3] * page_height,
        )
    return bbox


def _unique_text_matches(
    matches: Sequence[tuple[NativeTextBlock, str, float]],
) -> list[tuple[NativeTextBlock, str, float]]:
    unique: dict[tuple[str, str], tuple[NativeTextBlock, str, float]] = {}
    for block, text, distance in matches:
        unique.setdefault((block.block_id, text), (block, text, distance))
    return list(unique.values())


def _bbox_distance(first: PdfBBox, second: PdfBBox) -> float:
    horizontal = max(first[0] - second[2], second[0] - first[2], 0.0)
    vertical = max(first[1] - second[3], second[1] - first[3], 0.0)
    return math.hypot(horizontal, vertical)


def _union_bbox(boxes: Iterable[PdfBBox]) -> PdfBBox:
    materialized = tuple(boxes)
    if not materialized:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(box[0] for box in materialized),
        min(box[1] for box in materialized),
        max(box[2] for box in materialized),
        max(box[3] for box in materialized),
    )


def _text_bands(
    blocks: Sequence[NativeTextBlock],
    tolerance: float,
) -> list[tuple[NativeTextBlock, ...]]:
    ordered = sorted(blocks, key=lambda block: (_y_center(block.bbox), block.bbox[0]))
    bands: list[list[NativeTextBlock]] = []
    for block in ordered:
        center = _y_center(block.bbox)
        if not bands or center - _y_center(_union_bbox(b.bbox for b in bands[-1])) > tolerance:
            bands.append([block])
        else:
            bands[-1].append(block)
    return [
        tuple(sorted(band, key=lambda block: (block.bbox[0], block.bbox[1])))
        for band in bands
    ]


def _x_center(bbox: PdfBBox) -> float:
    return (bbox[0] + bbox[2]) / 2.0


def _y_center(bbox: PdfBBox) -> float:
    return (bbox[1] + bbox[3]) / 2.0


__all__ = [
    "NativeSalvageItem",
    "NativeTextCoverageAssessment",
    "NativeTextBlock",
    "NativeTextSalvageResult",
    "NativeTextSalvager",
    "SalvageStatus",
]
