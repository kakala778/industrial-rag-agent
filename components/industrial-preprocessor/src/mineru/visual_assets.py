"""High-resolution visual assets rendered directly from source PDFs."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VisualAssetMetadata:
    """Metadata for one source-page PNG rendered without MinerU re-sampling."""

    source_pdf: str
    page_number: int
    requested_dpi: float
    actual_dpi: float
    width: int
    height: int
    sha256: str
    output_path: Path
    downscale_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "source_pdf": self.source_pdf,
            "page_number": self.page_number,
            "requested_dpi": self.requested_dpi,
            "actual_dpi": self.actual_dpi,
            "width": self.width,
            "height": self.height,
            "sha256": self.sha256,
            "output_path": str(self.output_path),
            "downscale_reason": self.downscale_reason,
        }


class PageRenderer:
    """Render a PDF page directly to PNG with explicit size protection."""

    points_per_inch = 72.0

    def __init__(
        self,
        *,
        default_dpi: float = 150.0,
        max_pixels: int | None = 50_000_000,
        max_dimension: int | None = 30_000,
    ) -> None:
        self._validate_dpi(default_dpi)
        self._validate_limit(max_pixels, "max_pixels")
        self._validate_limit(max_dimension, "max_dimension")
        self.default_dpi = float(default_dpi)
        self.max_pixels = max_pixels
        self.max_dimension = max_dimension

    def render_page(
        self,
        input_pdf: str | Path,
        page_number: int,
        output_dir: str | Path,
    ) -> VisualAssetMetadata:
        if page_number < 1:
            raise ValueError("page_number must be one-based and positive")

        input_path = Path(input_pdf)
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        output_path = output_root / "original_page.png"
        pymupdf = _load_pymupdf()
        document = pymupdf.open(str(input_path))
        try:
            if page_number > document.page_count:
                raise ValueError(
                    f"page_number {page_number} exceeds {document.page_count} pages"
                )
            page = document[page_number - 1]
            page_width = float(page.rect.width)
            page_height = float(page.rect.height)
            actual_dpi, downscale_reason = self._safe_dpi(
                page_width, page_height, self.default_dpi
            )
            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(
                    actual_dpi / self.points_per_inch,
                    actual_dpi / self.points_per_inch,
                ),
                alpha=False,
            )
            pixmap.save(str(output_path))
            width = int(pixmap.width)
            height = int(pixmap.height)
        finally:
            document.close()

        return VisualAssetMetadata(
            source_pdf=str(input_path.resolve()),
            page_number=page_number,
            requested_dpi=self.default_dpi,
            actual_dpi=actual_dpi,
            width=width,
            height=height,
            sha256=_sha256(output_path),
            output_path=output_path.resolve(),
            downscale_reason=downscale_reason,
        )

    def _safe_dpi(
        self,
        page_width: float,
        page_height: float,
        requested_dpi: float,
    ) -> tuple[float, str | None]:
        requested_width, requested_height = _pixel_dimensions(
            page_width, page_height, requested_dpi
        )
        reasons: list[str] = []
        if (
            self.max_pixels is not None
            and requested_width * requested_height > self.max_pixels
        ):
            reasons.append(f"max_pixels={self.max_pixels}")
        if (
            self.max_dimension is not None
            and max(requested_width, requested_height) > self.max_dimension
        ):
            reasons.append(f"max_dimension={self.max_dimension}")

        if not reasons:
            return requested_dpi, None

        lower = 0.0
        upper = requested_dpi
        for _ in range(50):
            candidate = (lower + upper) / 2
            width, height = _pixel_dimensions(
                page_width, page_height, candidate
            )
            within_pixels = (
                self.max_pixels is None
                or width * height <= self.max_pixels
            )
            within_dimension = (
                self.max_dimension is None
                or max(width, height) <= self.max_dimension
            )
            if within_pixels and within_dimension:
                lower = candidate
            else:
                upper = candidate

        return lower, "; ".join(reasons)

    @staticmethod
    def _validate_dpi(value: float) -> None:
        if value <= 0:
            raise ValueError("dpi must be positive")

    @staticmethod
    def _validate_limit(value: int | None, name: str) -> None:
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive when configured")


def _pixel_dimensions(
    page_width: float, page_height: float, dpi: float
) -> tuple[int, int]:
    return (
        max(1, math.ceil(page_width * dpi / PageRenderer.points_per_inch)),
        max(1, math.ceil(page_height * dpi / PageRenderer.points_per_inch)),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pymupdf():
    try:
        import pymupdf

        return pymupdf
    except ImportError as pymupdf_error:
        try:
            import fitz as pymupdf

            return pymupdf
        except ImportError as fitz_error:
            raise RuntimeError(
                "PyMuPDF is required for PageRenderer; install project dependencies"
            ) from fitz_error


__all__ = ["PageRenderer", "VisualAssetMetadata"]
