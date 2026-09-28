"""Minimal, read-only access to MinerU structured output packages."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class OutputSchemaError(ValueError):
    """Raised when a MinerU output package is present but malformed."""


@dataclass(frozen=True)
class MinerUBlock:
    """A raw MinerU block exposed to quality checks without model conversion."""

    block_id: str
    block_type: str
    text: str
    bbox: tuple[float, float, float, float] | None
    source_path: Path | str | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class MinerUPage:
    """One output page using a one-based public page number."""

    page_number: int
    page_idx: int
    blocks: tuple[MinerUBlock, ...]

    @property
    def text(self) -> str:
        return "\n".join(block.text for block in self.blocks if block.text)

    @property
    def text_blocks(self) -> tuple[MinerUBlock, ...]:
        return tuple(block for block in self.blocks if block.block_type == "text")

    @property
    def table_blocks(self) -> tuple[MinerUBlock, ...]:
        return tuple(block for block in self.blocks if block.block_type == "table")

    @property
    def image_blocks(self) -> tuple[MinerUBlock, ...]:
        return tuple(block for block in self.blocks if block.block_type == "image")


@dataclass(frozen=True)
class MinerUOutput:
    """Read-only output data needed by the quality layer."""

    output_path: Path
    pages: tuple[MinerUPage, ...]
    markdown: str
    structured_content: dict[str, Any] | None
    middle_json: dict[str, Any] | None
    source_format: str

    def get_page(self, page_number: int) -> MinerUPage | None:
        return next(
            (page for page in self.pages if page.page_number == page_number),
            None,
        )


class MinerUOutputReader:
    """Read a directory or ZIP without interpreting it as a Document."""

    def read(self, output_path: str | Path) -> MinerUOutput:
        path = Path(output_path)
        if path.is_dir():
            archive = _find_nested_archive(path)
            source = _ZipSource(archive) if archive is not None else _DirectorySource(path)
        elif path.is_file() and path.suffix.lower() == ".zip":
            source = _ZipSource(path)
        else:
            raise OutputSchemaError(f"MinerU output is not a directory or ZIP: {path}")

        structured = self._read_json_if_present(source, "structured_content.json")
        middle = self._read_json_if_present(source, "middle_json.json")
        if structured is None and middle is None:
            raise OutputSchemaError(
                "MinerU output has neither structured_content.json nor middle_json.json"
            )

        if structured is not None:
            self._validate_pages(structured, "structured_content.json")
        if middle is not None:
            self._validate_pages(middle, "middle_json.json")

        primary = structured if structured is not None else middle
        assert primary is not None
        primary_name = "structured_content" if structured is not None else "middle_json"
        pages = tuple(self._read_pages(primary["pages"], source))
        markdown = self._read_text_if_present(source, "markdown.md")
        return MinerUOutput(
            output_path=path,
            pages=pages,
            markdown=markdown,
            structured_content=structured,
            middle_json=middle,
            source_format=primary_name,
        )

    @staticmethod
    def _read_json_if_present(
        source: "_OutputSource", filename: str
    ) -> dict[str, Any] | None:
        raw = source.read_optional(filename)
        if raw is None:
            return None
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OutputSchemaError(f"invalid JSON in {filename}: {exc}") from exc
        if not isinstance(value, dict):
            raise OutputSchemaError(f"{filename} must contain a JSON object")
        return value

    @staticmethod
    def _read_text_if_present(source: "_OutputSource", filename: str) -> str:
        raw = source.read_optional(filename)
        if raw is None:
            return ""
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise OutputSchemaError(f"invalid UTF-8 in {filename}: {exc}") from exc

    @staticmethod
    def _validate_pages(data: dict[str, Any], filename: str) -> None:
        pages = data.get("pages")
        if not isinstance(pages, list):
            raise OutputSchemaError(f"{filename}.pages must be a list")

    def _read_pages(
        self, pages_data: list[Any], source: "_OutputSource"
    ) -> Iterable[MinerUPage]:
        for page_position, raw_page in enumerate(pages_data):
            if not isinstance(raw_page, dict):
                raise OutputSchemaError(
                    f"page {page_position} must be a JSON object"
                )
            page_idx = raw_page.get("page_idx")
            blocks_data = raw_page.get("blocks")
            if (
                not isinstance(page_idx, int)
                or isinstance(page_idx, bool)
                or page_idx < 0
            ):
                raise OutputSchemaError(
                    f"page {page_position} has no non-negative integer page_idx"
                )
            if not isinstance(blocks_data, list):
                raise OutputSchemaError(
                    f"page {page_idx} blocks must be a list"
                )
            blocks = tuple(
                self._read_block(page_idx, block_position, raw_block, source)
                for block_position, raw_block in enumerate(blocks_data)
            )
            yield MinerUPage(
                page_number=page_idx + 1,
                page_idx=page_idx,
                blocks=blocks,
            )

    @staticmethod
    def _read_block(
        page_idx: int,
        block_position: int,
        raw_block: Any,
        source: "_OutputSource",
    ) -> MinerUBlock:
        if not isinstance(raw_block, dict):
            raise OutputSchemaError(
                f"page {page_idx} block {block_position} must be an object"
            )
        block_type = raw_block.get("type")
        if not isinstance(block_type, str) or not block_type:
            raise OutputSchemaError(
                f"page {page_idx} block {block_position} has no type"
            )
        bbox = _read_bbox(raw_block.get("bbox"), page_idx, block_position)
        image_reference = _find_reference(raw_block)
        return MinerUBlock(
            block_id=f"page-{page_idx}-block-{block_position}",
            block_type=block_type,
            text=_content_to_text(raw_block.get("content")),
            bbox=bbox,
            source_path=(
                source.resolve(image_reference)
                if image_reference is not None
                else None
            ),
            raw=raw_block,
        )


class _OutputSource:
    def read_optional(self, filename: str) -> bytes | None:
        raise NotImplementedError

    def resolve(self, relative_path: str) -> Path | str:
        raise NotImplementedError


class _DirectorySource(_OutputSource):
    def __init__(self, root: Path) -> None:
        self.root = root

    def read_optional(self, filename: str) -> bytes | None:
        candidates = [self.root / filename, *self.root.rglob(filename)]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.read_bytes()
        return None

    def resolve(self, relative_path: str) -> Path:
        return self.root / Path(relative_path)


class _ZipSource(_OutputSource):
    def __init__(self, archive: Path) -> None:
        self.archive = archive

    def read_optional(self, filename: str) -> bytes | None:
        with zipfile.ZipFile(self.archive) as package:
            member = _find_zip_member(package.namelist(), filename)
            return package.read(member) if member is not None else None

    def resolve(self, relative_path: str) -> str:
        return f"{self.archive}!/{relative_path}"


def _find_nested_archive(root: Path) -> Path | None:
    archives = tuple(root.rglob("*.zip"))
    if len(archives) == 1:
        return archives[0]
    return None


def _find_zip_member(names: list[str], filename: str) -> str | None:
    for name in names:
        if name == filename or name.replace("\\", "/").endswith(
            f"/{filename}"
        ):
            return name
    return None


def _read_bbox(
    raw_bbox: Any, page_idx: int, block_position: int
) -> tuple[float, float, float, float] | None:
    if raw_bbox is None:
        return None
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        raise OutputSchemaError(
            f"page {page_idx} block {block_position} bbox must have four values"
        )
    try:
        return tuple(float(value) for value in raw_bbox)  # type: ignore[return-value]
    except (TypeError, ValueError) as exc:
        raise OutputSchemaError(
            f"page {page_idx} block {block_position} bbox is not numeric"
        ) from exc


def _find_reference(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in ("image_source", "image_path"):
            reference = value.get(key)
            if isinstance(reference, str):
                return reference
        for child in value.values():
            reference = _find_reference(child)
            if reference is not None:
                return reference
    elif isinstance(value, list):
        for child in value:
            reference = _find_reference(child)
            if reference is not None:
                return reference
    return None


def _content_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        if "content" in value:
            return _content_to_text(value["content"])
        return "\n".join(
            text for child in value.values() if (text := _content_to_text(child))
        )
    if isinstance(value, list):
        return "\n".join(
            text for child in value if (text := _content_to_text(child))
        )
    return ""


__all__ = [
    "MinerUBlock",
    "MinerUOutput",
    "MinerUOutputReader",
    "MinerUPage",
    "OutputSchemaError",
]
