"""Page model for the unified document model."""

from dataclasses import dataclass, field
from typing import cast

from .block import Block


@dataclass
class Page:
    """A page containing ordered content blocks."""

    page_number: int
    blocks: list[Block] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "page_number": self.page_number,
            "blocks": [block.to_dict() for block in self.blocks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Page":
        raw_blocks = cast(list[dict[str, object]], data.get("blocks", []))
        return cls(
            page_number=int(cast(int, data["page_number"])),
            blocks=[Block.from_dict(block) for block in raw_blocks],
        )
