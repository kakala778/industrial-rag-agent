"""Metadata and source-location models for the unified document model."""

from dataclasses import dataclass, field
from typing import TypeAlias, cast


JSONValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | list["JSONValue"]
    | dict[str, "JSONValue"]
)


@dataclass
class Metadata:
    """Document-level metadata independent of any concrete parser."""

    original_filename: str | None = None
    file_hash: str | None = None
    created_at: str | None = None
    parser_name: str | None = None
    parser_version: str | None = None
    extra: dict[str, JSONValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "original_filename": self.original_filename,
            "file_hash": self.file_hash,
            "created_at": self.created_at,
            "parser_name": self.parser_name,
            "parser_version": self.parser_version,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Metadata":
        return cls(
            original_filename=cast(str | None, data.get("original_filename")),
            file_hash=cast(str | None, data.get("file_hash")),
            created_at=cast(str | None, data.get("created_at")),
            parser_name=cast(str | None, data.get("parser_name")),
            parser_version=cast(str | None, data.get("parser_version")),
            extra=cast(dict[str, JSONValue], data.get("extra", {})),
        )


@dataclass
class BoundingBox:
    """A block's position within its page."""

    x: float
    y: float
    width: float
    height: float

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "BoundingBox":
        return cls(
            x=float(cast(float, data["x"])),
            y=float(cast(float, data["y"])),
            width=float(cast(float, data["width"])),
            height=float(cast(float, data["height"])),
        )


@dataclass
class SourceReference:
    """Traceability information connecting a block to its source document."""

    document_id: str
    page_number: int
    source_id: str | None = None
    locator: str | None = None
    parser_block_id: str | None = None
    extra: dict[str, JSONValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "document_id": self.document_id,
            "page_number": self.page_number,
            "source_id": self.source_id,
            "locator": self.locator,
            "parser_block_id": self.parser_block_id,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "SourceReference":
        return cls(
            document_id=cast(str, data["document_id"]),
            page_number=int(cast(int, data["page_number"])),
            source_id=cast(str | None, data.get("source_id")),
            locator=cast(str | None, data.get("locator")),
            parser_block_id=cast(str | None, data.get("parser_block_id")),
            extra=cast(dict[str, JSONValue], data.get("extra", {})),
        )
