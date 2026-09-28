"""Load PDF pages from the local MinerU 4.0.5 Middle JSON output."""

import hashlib
from html.parser import HTMLParser
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_ROOT = PROJECT_ROOT / ".local" / "mineru"
MINERU_VERSION = "4.0.5"
MINERU_TIER = "advanced"
MINERU_OCR_MODE = "ocr"
CACHE_SCHEMA_VERSION = 1
REPRESENTATIONS = {"flat", "structured"}
TEXT_BLOCK_TYPES = {
    "text",
    "paragraph_title",
    "doc_title",
    "list",
    "header",
    "footer",
    "page_number",
    "ref_text",
    "aside_text",
}
VISUAL_BODY_TYPES = {"image_body", "chart_body", "table_body"}
TEXT_PAYLOAD_RE = re.compile(r"(?i)data:image/[^\s<>\"']+|base64,[A-Za-z0-9+/=_-]+")


def _default_runner_path():
    configured_path = os.environ.get("MINERU_RUNNER_PATH")
    if configured_path:
        return Path(configured_path).expanduser()

    # Keep support for the original sibling-workspace layout.
    if PROJECT_ROOT.name == "rag-agent":
        integrated_runner = (
            PROJECT_ROOT.parent
            / "minerU"
            / "mineru-405-poc"
            / "run-mineru.ps1"
        )
        if integrated_runner.is_file():
            return integrated_runner

    projects_root = PROJECT_ROOT.parent.parent.parent
    return projects_root / "Project" / "mineru-405-poc" / "run-mineru.ps1"


def _resolve_runner_path(runner_path=None):
    path = Path(runner_path).expanduser() if runner_path else _default_runner_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"MinerU runner script not found: {path}. "
            "Set MINERU_RUNNER_PATH or pass runner_path explicitly."
        )
    return path.resolve()


def _resolve_powershell(executable=None):
    if executable:
        return str(executable)

    configured = os.environ.get("MINERU_POWERSHELL_EXECUTABLE")
    if configured:
        return configured

    for candidate in ("pwsh.exe", "pwsh", "powershell.exe", "powershell"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    raise FileNotFoundError(
        "PowerShell was not found. Install PowerShell or set "
        "MINERU_POWERSHELL_EXECUTABLE."
    )


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as pdf_file:
        for block in iter(lambda: pdf_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_key(pdf_path, pdf_sha256):
    identity = {
        "cache_schema": CACHE_SCHEMA_VERSION,
        "source_name": pdf_path.name,
        "pdf_sha256": pdf_sha256,
        "mineru_version": MINERU_VERSION,
        "format": "middle_json",
        "tier": MINERU_TIER,
        "ocr_mode": MINERU_OCR_MODE,
    }
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), identity


def _extract_content(value):
    """Read only MinerU content fields, including the current nested block shape."""
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        return _extract_content(value.get("content"))
    if isinstance(value, list):
        parts = []
        for item in value:
            parts.extend(_extract_content(item))
        return parts
    return []


def _clean_readable_text(value):
    """Remove explicit image payloads and non-printable data from text values."""
    if not isinstance(value, str):
        return ""
    text = TEXT_PAYLOAD_RE.sub(" ", value)
    text = "".join(
        character
        for character in text
        if character in "\n\r\t" or character.isprintable()
    )
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\f\v ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _content_strings(value):
    if isinstance(value, str):
        text = _clean_readable_text(value)
        return [text] if text else []
    if isinstance(value, dict):
        if value.get("type") in VISUAL_BODY_TYPES:
            return []
        return _content_strings(value.get("content"))
    if isinstance(value, list):
        parts = []
        for item in value:
            parts.extend(_content_strings(item))
        return parts
    return []


class _ReadableMarkupParser(HTMLParser):
    """Extract text from MinerU chart markup without retaining control tags."""

    def __init__(self, *, chart_cells=False):
        super().__init__(convert_charrefs=True)
        self.chart_cells = chart_cells
        self.parts = []
        self.skip_depth = 0

    def handle_starttag(self, tag, _attrs):
        tag = tag.lower()
        if tag in {"script", "style"}:
            self.skip_depth += 1
            return
        if not self.chart_cells:
            return
        if tag in {"nl", "br", "tr"}:
            self.parts.append("\n")
        elif tag.endswith("cel"):
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if tag.lower() in {"script", "style"} and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data):
        if not self.skip_depth:
            self.parts.append(data)


class _TableMarkupParser(HTMLParser):
    """Read HTML table rows and cells into a simple, stable text form."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.row = None
        self.cell = None
        self.outside_text = []

    def _finish_cell(self):
        if self.cell is not None:
            self.row.append(_clean_readable_text("".join(self.cell)))
            self.cell = None

    def _finish_row(self):
        if self.row is not None:
            self._finish_cell()
            if any(self.row):
                self.rows.append(self.row)
            self.row = None

    def handle_starttag(self, tag, _attrs):
        tag = tag.lower()
        if tag == "tr":
            self._finish_row()
            self.row = []
        elif tag in {"td", "th"}:
            if self.row is None:
                self.row = []
            self._finish_cell()
            self.cell = []
        elif tag == "br":
            if self.cell is not None:
                self.cell.append(" ")
            elif self.row is None:
                self.outside_text.append(" ")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in {"td", "th"}:
            self._finish_cell()
        elif tag == "tr":
            self._finish_row()

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)
        elif self.row is None:
            self.outside_text.append(data)


def _table_body_text(value):
    strings = _content_strings(value)
    rendered = []
    for raw in strings:
        parser = _TableMarkupParser()
        parser.feed(raw)
        parser.close()
        rows = [
            " | ".join(cell for cell in row)
            for row in parser.rows
            if any(cell for cell in row)
        ]
        if rows:
            rendered.extend(rows)
        else:
            text_parser = _ReadableMarkupParser()
            text_parser.feed(raw)
            text_parser.close()
            plain = _clean_readable_text("".join(text_parser.parts))
            if plain:
                rendered.append(plain)
    return "\n".join(rendered).strip()


def _chart_body_text(value):
    strings = _content_strings(value)
    rendered = []
    for raw in strings:
        parser = _ReadableMarkupParser(chart_cells=True)
        parser.feed(raw)
        parser.close()
        text = _clean_readable_text("".join(parser.parts))
        lines = []
        for line in text.splitlines():
            cells = [cell.strip() for cell in line.split("|") if cell.strip()]
            if cells:
                lines.append(" | ".join(cells))
        text = "\n".join(lines)
        if text:
            rendered.append(text)
    return "\n".join(rendered)


def _image_body_text(value):
    """Keep readable image-body text while stripping inline payloads and markup."""
    rendered = []
    for raw in _extract_content(value):
        clean = _clean_readable_text(raw)
        parser = _ReadableMarkupParser()
        parser.feed(clean)
        parser.close()
        text = _clean_readable_text("".join(parser.parts))
        if text:
            rendered.append(text)
    return "\n".join(rendered)


def _block_children(block):
    content = block.get("content")
    if isinstance(content, list):
        return content
    if content is None:
        return []
    return [content]


def _typed_text_for_visual_block(block, block_type):
    if block_type == "table":
        body_type = "table_body"
        text_types = {"table_caption", "table_footnote", "text", "ref_text"}
        body_renderer = _table_body_text
    elif block_type == "chart":
        body_type = "chart_body"
        text_types = {"chart_caption", "text", "ref_text"}
        body_renderer = _chart_body_text
    else:
        body_type = "image_body"
        text_types = {"image_caption", "image_footnote", "text", "ref_text"}
        body_renderer = _image_body_text

    parts = []
    for child in _block_children(block):
        if not isinstance(child, dict):
            if isinstance(child, str):
                text = body_renderer(child) if body_renderer else _clean_readable_text(child)
                if text:
                    parts.append(text)
            continue
        child_type = child.get("type")
        if child_type == body_type:
            text = body_renderer(child.get("content"))
        elif child_type in text_types:
            text = "\n".join(_content_strings(child.get("content")))
        else:
            text = ""
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _block_index(block, fallback):
    value = block.get("index")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return fallback


def _documents_from_middle_json(middle_json, source, representation="structured"):
    if not isinstance(middle_json, dict) or not isinstance(middle_json.get("pages"), list):
        raise ValueError("Middle JSON must contain a top-level 'pages' array.")
    if representation not in REPRESENTATIONS:
        raise ValueError(
            f"Unsupported MinerU representation: {representation!r}. "
            "Choose 'flat' or 'structured'."
        )

    pages = middle_json["pages"]
    if not pages:
        raise ValueError("Middle JSON contains no pages.")

    documents = []
    for page_position, page in enumerate(pages):
        if not isinstance(page, dict):
            raise ValueError(f"Middle JSON page {page_position} must be an object.")

        page_idx = page.get("page_idx")
        if isinstance(page_idx, bool) or not isinstance(page_idx, int) or page_idx < 0:
            raise ValueError(
                f"Middle JSON page {page_position} has an invalid page_idx: {page_idx!r}."
            )

        blocks = page.get("blocks", [])
        if not isinstance(blocks, list):
            raise ValueError(f"Middle JSON page_idx {page_idx} 'blocks' must be an array.")

        if representation == "flat":
            content_parts = []
            block_types = []
            for block_position, block in enumerate(blocks):
                if not isinstance(block, dict):
                    raise ValueError(
                        f"MinerU block {block_position} on page_idx {page_idx} must be an object."
                    )
                block_type = block.get("type")
                if isinstance(block_type, str) and block_type not in block_types:
                    block_types.append(block_type)
                content_parts.extend(_extract_content(block.get("content")))
            documents.append(
                {
                    "text": "\n".join(content_parts),
                    "metadata": {
                        "source": source,
                        "page": page_idx + 1,
                        "parser": "mineru",
                        "block_types": block_types,
                    },
                }
            )
            continue

        text_group = []

        def flush_text_group():
            if not text_group:
                return
            text = "\n".join(part["text"] for part in text_group if part["text"]).strip()
            if text:
                group_types = list(dict.fromkeys(part["type"] for part in text_group))
                documents.append(
                    {
                        "text": text,
                        "metadata": {
                            "source": source,
                            "page": page_idx + 1,
                            "parser": "mineru",
                            "representation": "structured",
                            "block_type": "text_group",
                            "block_index": text_group[0]["index"],
                            "block_types": group_types,
                        },
                    }
                )
            text_group.clear()

        for block_position, block in enumerate(blocks):
            if not isinstance(block, dict):
                raise ValueError(
                    f"Middle JSON block {block_position} on page_idx {page_idx} "
                    "must be an object."
                )
            block_type = block.get("type")
            block_index = _block_index(block, block_position)
            if block_type in TEXT_BLOCK_TYPES:
                text = "\n".join(_content_strings(block.get("content")))
                if text:
                    text_group.append(
                        {"text": text, "type": block_type, "index": block_index}
                    )
                continue

            flush_text_group()
            if block_type in {"table", "chart", "image"}:
                text = _typed_text_for_visual_block(block, block_type)
            else:
                text = "\n".join(_content_strings(block.get("content")))
            if text:
                documents.append(
                    {
                        "text": text,
                        "metadata": {
                            "source": source,
                            "page": page_idx + 1,
                            "parser": "mineru",
                            "representation": "structured",
                            "block_type": block_type or "unknown",
                            "block_index": block_index,
                        },
                    }
                )

        flush_text_group()

    return documents


def _read_middle_json(path, source, representation="structured"):
    try:
        middle_json = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Middle JSON could not be read or parsed: {path}: {exc}") from exc
    return _documents_from_middle_json(middle_json, source, representation=representation)


def _cached_middle_json(cache_entry, pdf_path, identity, key, representation="structured"):
    manifest_path = cache_entry / "manifest.json"
    middle_json_path = cache_entry / "output" / f"{pdf_path.stem}.json"
    if not manifest_path.is_file() or not middle_json_path.is_file():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None

    if (
        manifest.get("cache_key") != key
        or manifest.get("identity") != identity
        or manifest.get("middle_json") != f"output/{pdf_path.stem}.json"
    ):
        return None
    try:
        return _read_middle_json(
            middle_json_path,
            pdf_path.name,
            representation=representation,
        )
    except ValueError:
        # Treat damaged or outdated cached Middle JSON as a miss so it can be
        # regenerated from the source PDF instead of requiring manual cleanup.
        return None


def _run_mineru(pdf_path, output_dir, runner_path, powershell_executable):
    command = [
        powershell_executable,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(runner_path),
        "parse",
        str(pdf_path),
        "--output",
        str(output_dir),
        "--format",
        "middle_json",
        "--tier",
        MINERU_TIER,
        "--ocr-mode",
        MINERU_OCR_MODE,
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=3600,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("MinerU parsing timed out after 3600 seconds.") from exc
    except OSError as exc:
        raise RuntimeError(f"Could not start MinerU through PowerShell: {exc}") from exc

    if result.returncode != 0:
        details = []
        if result.stdout and result.stdout.strip():
            details.append(f"stdout: {result.stdout.strip()[-1500:]}")
        if result.stderr and result.stderr.strip():
            details.append(f"stderr: {result.stderr.strip()[-1500:]}")
        suffix = "\n" + "\n".join(details) if details else ""
        raise RuntimeError(f"MinerU parsing failed with exit code {result.returncode}.{suffix}")


def load_pdf_with_mineru(
    path,
    *,
    force=False,
    runner_path=None,
    cache_root=None,
    powershell_executable=None,
    representation="structured",
):
    """Load cached or parsed Middle JSON as flat pages or structured block documents."""
    if representation not in REPRESENTATIONS:
        raise ValueError(
            f"Unsupported MinerU representation: {representation!r}. "
            "Choose 'flat' or 'structured'."
        )
    pdf_path = Path(path).expanduser()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    resolved_cache_root = Path(cache_root) if cache_root else CACHE_ROOT
    resolved_cache_root.mkdir(parents=True, exist_ok=True)

    pdf_sha256 = _sha256_file(pdf_path)
    key, identity = _cache_key(pdf_path, pdf_sha256)
    cache_entry = resolved_cache_root / key

    if not force:
        cached_documents = _cached_middle_json(
            cache_entry,
            pdf_path,
            identity,
            key,
            representation=representation,
        )
        if cached_documents is not None:
            return cached_documents

    resolved_runner = _resolve_runner_path(runner_path)
    resolved_powershell = _resolve_powershell(powershell_executable)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{key[:12]}-", dir=resolved_cache_root)
    )
    try:
        output_dir = staging_dir / "output"
        output_dir.mkdir()
        _run_mineru(pdf_path, output_dir, resolved_runner, resolved_powershell)

        # MinerU 4.0.5 was smoke-tested to write <input-stem>.json directly
        # under the requested --output directory for --format middle_json.
        middle_json_path = output_dir / f"{pdf_path.stem}.json"
        if not middle_json_path.is_file():
            raise FileNotFoundError(
                f"MinerU completed but the expected Middle JSON is missing: "
                f"{middle_json_path}"
            )

        documents = _read_middle_json(
            middle_json_path,
            pdf_path.name,
            representation=representation,
        )
        manifest = {
            "cache_key": key,
            "identity": identity,
            "middle_json": f"output/{pdf_path.stem}.json",
        }
        (staging_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        if cache_entry.exists():
            shutil.rmtree(cache_entry)
        os.replace(staging_dir, cache_entry)
        return documents
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
