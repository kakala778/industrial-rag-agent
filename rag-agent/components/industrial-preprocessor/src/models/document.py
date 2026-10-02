"""Top-level document model and JSON serialization boundary."""

import json
from dataclasses import dataclass, field
from typing import cast

from .metadata import JSONValue, Metadata
from .page import Page


@dataclass
class Document:
    """A parser-independent structured document."""

    document_id: str
    filename: str
    file_type: str
    metadata: Metadata
    pages: list[Page] = field(default_factory=list)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "metadata": self.metadata.to_dict(),
            "pages": [page.to_dict() for page in self.pages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Document":
        raw_pages = cast(list[dict[str, object]], data.get("pages", []))
        return cls(
            document_id=cast(str, data["document_id"]),
            filename=cast(str, data["filename"]),
            file_type=cast(str, data["file_type"]),
            metadata=Metadata.from_dict(
                cast(dict[str, object], data["metadata"])
            ),
            pages=[Page.from_dict(page) for page in raw_pages],
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, value: str) -> "Document":
        return cls.from_dict(cast(dict[str, object], json.loads(value)))
