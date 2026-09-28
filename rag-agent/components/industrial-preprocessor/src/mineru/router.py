"""Bounded, configurable fallback routing for page-level MinerU runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol, Sequence

from .output_reader import MinerUOutputReader, MinerUPage, OutputSchemaError
from .native_text_salvage import (
    NativeTextSalvageResult,
    NativeTextSalvager,
)
from .probe import PageProbe, PageProbeResult
from .profiles import ParseProfile, STANDARD_AUTO
from .quality import QualityGate, QualityMetrics, QualityResult
from .runner import MinerURunner, RunResult
from .table_ambiguity import (
    TableLayoutAmbiguityDetector,
    TableLayoutFinding,
    TableTextCandidate,
)
from .visual_assets import PageRenderer, VisualAssetMetadata


class RunnerLike(Protocol):
    def run(
        self,
        input_pdf: str | Path,
        page_selector: str | int | None,
        output_dir: str | Path,
        profile: ParseProfile,
        timeout: float | None = 600.0,
    ) -> RunResult: ...


class ProbeLike(Protocol):
    def probe(self, input_pdf: str | Path, page_number: int) -> PageProbeResult: ...


class PageRendererLike(Protocol):
    def render_page(
        self,
        input_pdf: str | Path,
        page_number: int,
        output_dir: str | Path,
    ) -> VisualAssetMetadata: ...


class NativeTextSalvagerLike(Protocol):
    def salvage_pdf_page(
        self,
        pdf_path: str | Path,
        page_number: int,
        mineru_page: MinerUPage,
        *,
        page_width: float,
        page_height: float,
    ) -> NativeTextSalvageResult: ...


@dataclass(frozen=True)
class RouteAttempt:
    """A serializable record of one profile attempt."""

    attempt: int
    profile: ParseProfile
    quality_passed: bool
    quality_flags: tuple[str, ...]
    quality_metrics: QualityMetrics
    decision: str
    run_result: RunResult

    def to_dict(self) -> dict[str, object]:
        return {
            "attempt": self.attempt,
            "profile": self.profile.to_dict(),
            "quality_passed": self.quality_passed,
            "quality_flags": list(self.quality_flags),
            "quality_metrics": self.quality_metrics.to_dict(),
            "decision": self.decision,
            "run": self.run_result.to_dict(include_logs=False),
        }


@dataclass(frozen=True)
class RouteResult:
    """Final decision and audit paths for one page."""

    page_number: int
    decision: str
    attempts: tuple[RouteAttempt, ...]
    page_quality_path: Path
    route_log_path: Path
    review_required_path: Path
    visual_asset: dict[str, object]
    table_layout_findings: tuple[TableLayoutFinding, ...] = ()
    native_salvage_path: Path | None = None
    native_markdown_path: Path | None = None
    final_markdown_path: Path | None = None
    native_recovery: dict[str, object] = field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.decision == "ACCEPT"


class FallbackRouter:
    """Run, gate, and switch through an explicitly configured profile list."""

    def __init__(
        self,
        *,
        runner: RunnerLike | None = None,
        probe: ProbeLike | None = None,
        output_reader: MinerUOutputReader | None = None,
        quality_gate: QualityGate | None = None,
        page_renderer: PageRendererLike | None = None,
        native_text_salvager: NativeTextSalvagerLike | None = None,
        table_ambiguity_detector: TableLayoutAmbiguityDetector | None = None,
        initial_profile: ParseProfile = STANDARD_AUTO,
        fallback_profiles: Sequence[ParseProfile] = (),
        max_fallbacks: int = 1,
    ) -> None:
        if max_fallbacks < 0:
            raise ValueError("max_fallbacks must not be negative")
        self.runner = runner or MinerURunner()
        self.probe = probe or PageProbe()
        self.output_reader = output_reader or MinerUOutputReader()
        self.quality_gate = quality_gate or QualityGate()
        self.page_renderer = page_renderer or PageRenderer()
        self.native_text_salvager = native_text_salvager or NativeTextSalvager()
        self.table_ambiguity_detector = (
            table_ambiguity_detector or TableLayoutAmbiguityDetector()
        )
        self.initial_profile = initial_profile
        self.fallback_profiles = tuple(fallback_profiles)[:max_fallbacks]
        self.max_fallbacks = max_fallbacks

    def route(
        self,
        input_pdf: str | Path,
        page_number: int,
        output_dir: str | Path,
        timeout: float | None = 600.0,
        preserve_high_res_visual: bool = False,
        table_text_candidates: Sequence[TableTextCandidate] = (),
        document_id: str | None = None,
    ) -> RouteResult:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        probe_result = self.probe.probe(input_pdf, page_number)
        high_res_visual_required = (
            preserve_high_res_visual or probe_result.is_ultrawide
        )
        visual_asset = _initial_visual_asset(high_res_visual_required)
        table_layout_findings = self.table_ambiguity_detector.detect_many(
            tuple(table_text_candidates)
        )
        if high_res_visual_required:
            rendered = self.page_renderer.render_page(
                input_pdf=input_pdf,
                page_number=page_number,
                output_dir=root,
            )
            visual_asset.update(
                {
                    "original_highres_path": str(rendered.output_path),
                    **rendered.to_dict(),
                }
            )
            visual_asset.pop("output_path", None)
        profiles = (self.initial_profile, *self.fallback_profiles)
        attempts: list[RouteAttempt] = []

        for position, profile in enumerate(profiles, start=1):
            attempt_dir = root / f"attempt-{position}-{profile.name}"
            run_result = self.runner.run(
                input_pdf=input_pdf,
                page_selector=page_number,
                output_dir=attempt_dir,
                profile=profile,
                timeout=timeout,
            )
            quality = self._evaluate_attempt(run_result, page_number, probe_result)
            quality_flags = list(quality.flags)
            if (
                high_res_visual_required
                and "high_res_visual_required" not in quality_flags
            ):
                quality_flags.append("high_res_visual_required")
            if table_layout_findings and "table_layout_ambiguity" not in quality_flags:
                quality_flags.append("table_layout_ambiguity")
            visual_asset["mineru_asset_path"] = _find_mineru_asset_path(
                self.output_reader, run_result, page_number
            )
            is_last = position == len(profiles)
            decision = (
                "ACCEPT"
                if run_result.success and quality.passed and not table_layout_findings
                else "REVIEW_REQUIRED"
                if is_last
                else "FALLBACK"
            )
            attempt = RouteAttempt(
                attempt=position,
                profile=profile,
                quality_passed=quality.passed,
                quality_flags=tuple(quality_flags),
                quality_metrics=quality.metrics,
                decision=decision,
                run_result=run_result,
            )
            attempts.append(attempt)
            if decision == "ACCEPT" or decision == "REVIEW_REQUIRED":
                break

        native_salvage: NativeTextSalvageResult | None = None
        native_salvage_path: Path | None = None
        native_markdown_path: Path | None = None
        final_markdown_path: Path | None = None
        final_output = None
        final_run = attempts[-1].run_result
        if final_run.success:
            try:
                final_output = self.output_reader.read(final_run.output_dir)
                final_page = final_output.get_page(page_number)
            except OutputSchemaError:
                final_page = None
            if final_page is not None:
                native_salvage = self.native_text_salvager.salvage_pdf_page(
                    pdf_path=input_pdf,
                    page_number=page_number,
                    mineru_page=final_page,
                    page_width=probe_result.page_width,
                    page_height=probe_result.page_height,
                )
                if native_salvage.triggered:
                    if native_salvage.high_res_visual_required:
                        visual_asset["high_res_visual_required"] = True
                        if visual_asset.get("original_highres_path") is None:
                            try:
                                rendered = self.page_renderer.render_page(
                                    input_pdf=input_pdf,
                                    page_number=page_number,
                                    output_dir=root,
                                )
                                visual_asset.update(
                                    {
                                        "original_highres_path": str(rendered.output_path),
                                        **rendered.to_dict(),
                                    }
                                )
                                visual_asset.pop("output_path", None)
                            except Exception as exc:
                                visual_asset["render_error"] = (
                                    f"{type(exc).__name__}: {exc}"
                                )
                    native_salvage_path = root / "native_salvage.json"
                    native_markdown_path = root / "native_salvage.md"
                    final_markdown_path = root / "final_markdown.md"
                    resolved_document_id = document_id or Path(input_pdf).stem
                    recovered_blocks = native_salvage.to_document_blocks(
                        resolved_document_id
                    )
                    _write_json(
                        native_salvage_path,
                        {
                            "schema_version": "native-salvage-v2",
                            "document_id": resolved_document_id,
                            "source_pdf": str(input_pdf),
                            "page_number": page_number,
                            "salvage_triggered": native_salvage.salvage_triggered,
                            "triggered": native_salvage.triggered,
                            "salvage": native_salvage.to_dict(),
                            "blocks": [block.to_dict() for block in recovered_blocks],
                        },
                    )
                    markdown_addendum = native_salvage.to_markdown_addendum()
                    native_markdown_path.write_text(markdown_addendum, encoding="utf-8")
                    base_markdown = final_output.markdown if final_output is not None else ""
                    combined_markdown = base_markdown.rstrip()
                    if markdown_addendum:
                        combined_markdown = (
                            f"{combined_markdown}\n\n{markdown_addendum}"
                            if combined_markdown
                            else markdown_addendum
                        )
                    final_markdown_path.write_text(
                        combined_markdown.rstrip() + "\n",
                        encoding="utf-8",
                    )
                    final_attempt = attempts[-1]
                    flags = list(final_attempt.quality_flags)
                    if native_salvage.auto_restored_items and (
                        "native_text_salvage_applied" not in flags
                    ):
                        flags.append("native_text_salvage_applied")
                    if native_salvage.high_res_visual_required and (
                        "high_res_visual_required" not in flags
                    ):
                        flags.append("high_res_visual_required")
                    must_review = (
                        native_salvage.requires_review
                        or bool(visual_asset.get("render_error"))
                    )
                    if must_review and "native_text_salvage_review" not in flags:
                        flags.append("native_text_salvage_review")
                    if flags != list(final_attempt.quality_flags) or must_review:
                        attempts[-1] = replace(
                            final_attempt,
                            decision=(
                                "REVIEW_REQUIRED" if must_review else final_attempt.decision
                            ),
                            quality_flags=tuple(flags),
                        )

        final_decision = attempts[-1].decision
        native_recovery, reason_codes = _native_recovery_summary(
            native_salvage,
            native_salvage_path,
            table_layout_findings,
        )
        result = RouteResult(
            page_number=page_number,
            decision=final_decision,
            attempts=tuple(attempts),
            page_quality_path=root / "page_quality.json",
            route_log_path=root / "route_log.json",
            review_required_path=root / "review_required.json",
            visual_asset=visual_asset,
            table_layout_findings=table_layout_findings,
            native_salvage_path=native_salvage_path,
            native_markdown_path=native_markdown_path,
            final_markdown_path=final_markdown_path,
            native_recovery=native_recovery,
            reason_codes=reason_codes,
        )
        self._write_audits(result, probe_result)
        return result

    def _evaluate_attempt(
        self,
        run_result: RunResult,
        page_number: int,
        probe_result: PageProbeResult,
    ) -> QualityResult:
        if not run_result.success:
            return QualityResult(
                passed=False,
                flags=("runner_failed",),
                metrics=_empty_metrics(),
            )
        try:
            output = self.output_reader.read(run_result.output_dir)
            output_page = output.get_page(page_number)
            if output_page is None:
                raise OutputSchemaError(
                    f"output does not contain page {page_number}"
                )
        except OutputSchemaError:
            return QualityResult(
                passed=False,
                flags=("output_schema_error",),
                metrics=_empty_metrics(),
            )
        return self.quality_gate.evaluate(probe_result, output_page)

    def _write_audits(
        self, result: RouteResult, probe_result: PageProbeResult
    ) -> None:
        page_quality = {
            "page": result.page_number,
            "probe": {
                "has_text_layer": probe_result.has_text_layer,
                "native_char_count": probe_result.native_char_count,
                "native_numeric_token_count": probe_result.native_numeric_token_count,
                "native_word_count": probe_result.native_word_count,
                "page_width": probe_result.page_width,
                "page_height": probe_result.page_height,
                "aspect_ratio": probe_result.aspect_ratio,
                "is_ultrawide": probe_result.is_ultrawide,
            },
            "thresholds": self.quality_gate.thresholds.to_dict(),
            "visual_asset": result.visual_asset,
            "native_recovery": result.native_recovery,
            "reason_codes": list(result.reason_codes),
            "table_layout_findings": [
                finding.to_dict() for finding in result.table_layout_findings
            ],
            "native_text_salvage": {
                "structured_path": (
                    str(result.native_salvage_path)
                    if result.native_salvage_path is not None
                    else None
                ),
                "markdown_addendum_path": (
                    str(result.native_markdown_path)
                    if result.native_markdown_path is not None
                    else None
                ),
                "final_markdown_path": (
                    str(result.final_markdown_path)
                    if result.final_markdown_path is not None
                    else None
                ),
            },
            "attempts": [
                {
                    "attempt": attempt.attempt,
                    "profile": attempt.profile.to_dict(),
                    "flags": list(attempt.quality_flags),
                    "metrics": attempt.quality_metrics.to_dict(),
                    "passed": attempt.quality_passed,
                }
                for attempt in result.attempts
            ],
        }
        route_log = {
            "page": result.page_number,
            "decision": result.decision,
            "visual_asset": result.visual_asset,
            "native_recovery": result.native_recovery,
            "reason_codes": list(result.reason_codes),
            "attempts": [attempt.to_dict() for attempt in result.attempts],
            "table_layout_findings": [
                finding.to_dict() for finding in result.table_layout_findings
            ],
            "native_text_salvage": {
                "structured_path": (
                    str(result.native_salvage_path)
                    if result.native_salvage_path is not None
                    else None
                ),
                "markdown_addendum_path": (
                    str(result.native_markdown_path)
                    if result.native_markdown_path is not None
                    else None
                ),
                "final_markdown_path": (
                    str(result.final_markdown_path)
                    if result.final_markdown_path is not None
                    else None
                ),
            },
        }
        review_required = {
            "review_required": not result.accepted,
            "page": result.page_number,
            "native_recovery": result.native_recovery,
            "reason_codes": list(result.reason_codes),
            "reasons": [
                flag
                for flag in result.attempts[-1].quality_flags
                if flag != "high_res_visual_required"
            ],
            "table_layout_findings": [
                finding.to_dict() for finding in result.table_layout_findings
            ],
            "native_text_salvage": {
                "structured_path": (
                    str(result.native_salvage_path)
                    if result.native_salvage_path is not None
                    else None
                ),
                "markdown_addendum_path": (
                    str(result.native_markdown_path)
                    if result.native_markdown_path is not None
                    else None
                ),
                "final_markdown_path": (
                    str(result.final_markdown_path)
                    if result.final_markdown_path is not None
                    else None
                ),
            },
        }
        _write_json(result.page_quality_path, page_quality)
        _write_json(result.route_log_path, route_log)
        _write_json(result.review_required_path, review_required)


def _native_recovery_summary(
    salvage: NativeTextSalvageResult | None,
    assessment_path: Path | None,
    table_layout_findings: Sequence[TableLayoutFinding],
) -> tuple[dict[str, object], tuple[str, ...]]:
    assessment = salvage.recovery_assessment if salvage is not None else None
    recovery_reasons = tuple(assessment.reason_codes) if assessment is not None else ()
    combined_reasons = list(recovery_reasons)
    if table_layout_findings:
        combined_reasons.append("TABLE_ROW_RELATION_AMBIGUITY")
    reason_codes = tuple(dict.fromkeys(combined_reasons))
    summary: dict[str, object] = {
        "salvage_triggered": (
            salvage.salvage_triggered if salvage is not None else False
        ),
        "recovery_status": (
            salvage.recovery_status.value
            if salvage is not None and salvage.recovery_status is not None
            else None
        ),
        "review_required": salvage.requires_review if salvage is not None else False,
        "reason_codes": list(recovery_reasons),
        "assessment_path": str(assessment_path) if assessment_path is not None else None,
    }
    return summary, reason_codes


def _empty_metrics() -> QualityMetrics:
    return QualityMetrics(
        char_ratio=None,
        numeric_ratio=None,
        token_coverage=0.0,
        token_order_similarity=0.0,
        duplicate_token_ratio=0.0,
        repeated_line_ratio=0.0,
    )


def _initial_visual_asset(required: bool) -> dict[str, object]:
    return {
        "original_highres_path": None,
        "mineru_asset_path": None,
        "high_res_visual_required": required,
    }


def _find_mineru_asset_path(
    reader: MinerUOutputReader,
    run_result: RunResult,
    page_number: int,
) -> str | None:
    if not run_result.success:
        return None
    try:
        output = reader.read(run_result.output_dir)
        page = output.get_page(page_number)
    except OutputSchemaError:
        return None
    if page is None:
        return None
    for block in page.blocks:
        if block.source_path is not None:
            return str(block.source_path)
    return None


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


__all__ = ["FallbackRouter", "RouteAttempt", "RouteResult"]
