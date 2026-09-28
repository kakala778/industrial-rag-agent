"""Explicit MinerU parse profiles used by the configurable router."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ParseTier(str, Enum):
    """MinerU processing tier."""

    STANDARD = "standard"
    ADVANCED = "advanced"


class OCRMode(str, Enum):
    """Whether the MinerU CLI should force OCR."""

    AUTO = "auto"
    OCR = "ocr"


@dataclass(frozen=True)
class ParseProfile:
    """A named, serializable MinerU command profile."""

    name: str
    tier: ParseTier
    ocr_mode: OCRMode

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("profile name must not be blank")
        object.__setattr__(self, "tier", ParseTier(self.tier))
        object.__setattr__(self, "ocr_mode", OCRMode(self.ocr_mode))

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "tier": self.tier.value,
            "ocr_mode": self.ocr_mode.value,
        }


STANDARD_AUTO = ParseProfile(
    name="standard_auto",
    tier=ParseTier.STANDARD,
    ocr_mode=OCRMode.AUTO,
)
STANDARD_OCR = ParseProfile(
    name="standard_ocr",
    tier=ParseTier.STANDARD,
    ocr_mode=OCRMode.OCR,
)
ADVANCED_AUTO = ParseProfile(
    name="advanced_auto",
    tier=ParseTier.ADVANCED,
    ocr_mode=OCRMode.AUTO,
)
ADVANCED_OCR = ParseProfile(
    name="advanced_ocr",
    tier=ParseTier.ADVANCED,
    ocr_mode=OCRMode.OCR,
)


__all__ = [
    "ADVANCED_AUTO",
    "ADVANCED_OCR",
    "OCRMode",
    "ParseProfile",
    "ParseTier",
    "STANDARD_AUTO",
    "STANDARD_OCR",
]
