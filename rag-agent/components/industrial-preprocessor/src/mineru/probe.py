"""Independent native-text probes backed by PyMuPDF."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PageProbeResult:
    """Facts extracted from the original PDF page, not a Ground Truth label."""

    page_number: int
    has_text_layer: bool
    native_char_count: int
    native_numeric_token_count: int
    native_word_count: int
    page_width: float
    page_height: float
    probe_word_sequence: tuple[str, ...]
    aspect_ratio: float = 0.0
    is_ultrawide: bool = False


class PageProbe:
    """Probe one PDF page independently from MinerU."""

    def __init__(self, ultrawide_ratio_threshold: float = 2.0) -> None:
        if ultrawide_ratio_threshold <= 0:
            raise ValueError("ultrawide_ratio_threshold must be positive")
        self.ultrawide_ratio_threshold = ultrawide_ratio_threshold

    def probe(self, input_pdf: str | Path, page_number: int) -> PageProbeResult:
        if page_number < 1:
            raise ValueError("page_number must be one-based and positive")

        pymupdf = _load_pymupdf()
        document = pymupdf.open(str(input_pdf))
        try:
            if page_number > document.page_count:
                raise ValueError(
                    f"page_number {page_number} exceeds {document.page_count} pages"
                )
            page = document[page_number - 1]
            page_width = float(page.rect.width)
            page_height = float(page.rect.height)
            native_text = page.get_text("text", sort=True)
            words = page.get_text("words", sort=True)
            word_sequence = tuple(
                str(word[4]) for word in words if len(word) > 4 and str(word[4]).strip()
            )
            return PageProbeResult(
                page_number=page_number,
                has_text_layer=bool(native_text.strip()),
                native_char_count=len(_without_whitespace(native_text)),
                native_numeric_token_count=len(_NUMBER_PATTERN.findall(native_text)),
                native_word_count=len(word_sequence),
                page_width=page_width,
                page_height=page_height,
                probe_word_sequence=word_sequence,
                aspect_ratio=page_width / page_height,
                is_ultrawide=(
                    page_width / page_height
                    >= self.ultrawide_ratio_threshold
                ),
            )
        finally:
            document.close()


_NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)?")


def _without_whitespace(value: str) -> str:
    return "".join(value.split())


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
                "PyMuPDF is required for PageProbe; install project dependencies"
            ) from fitz_error


__all__ = ["PageProbe", "PageProbeResult"]
