"""Content blocks used by the unified document model."""

from dataclasses import dataclass
from enum import Enum
from typing import cast

from .metadata import BoundingBox, JSONValue, SourceReference


class BlockType(str, Enum):
    """Supported first-version content block types."""

    TEXT = "TEXT"
    TITLE = "TITLE"
    TABLE = "TABLE"
    IMAGE = "IMAGE"
    FORMULA = "FORMULA"

    def to_dict(self) -> str:
        return self.value

    @classmethod
    def from_dict(cls, value: object) -> "BlockType":
        return cls(cast(str, value))


@dataclass
class Block:
    """A page-level content unit with flexible JSON-compatible content."""

    block_id: str
    type: BlockType
    content: JSONValue
    page_number: int
    source: SourceReference
    bbox: BoundingBox | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "block_id": self.block_id,
            "type": self.type.to_dict(),
            "content": self.content,
            "page_number": self.page_number,
            "source": self.source.to_dict(),
            "bbox": self.bbox.to_dict() if self.bbox is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Block":
        raw_bbox = data.get("bbox")
        bbox = (
            BoundingBox.from_dict(cast(dict[str, object], raw_bbox))
            if raw_bbox is not None
            else None
        )
        return cls(
            block_id=cast(str, data["block_id"]),
            type=BlockType.from_dict(data["type"]),
            content=cast(JSONValue, data.get("content")),
            page_number=int(cast(int, data["page_number"])),
            source=SourceReference.from_dict(
                cast(dict[str, object], data["source"])
            ),
            bbox=bbox,
        )


__all__ = ["Block", "BlockType", "BoundingBox", "SourceReference"]
