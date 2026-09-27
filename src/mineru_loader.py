"""Load PDF pages from the local MinerU 4.0.5 Middle JSON output."""

import hashlib
import json
import os
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


def _default_runner_path():
    configured_path = os.environ.get("MINERU_RUNNER_PATH")
    if configured_path:
        return Path(configured_path).expanduser()

    # The local workspace layout is D:\Dev\Projects\AI\<repo> and
    # D:\Dev\Projects\Project\mineru-405-poc.
    projects_root = PROJECT_ROOT.parent.parent
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


def _documents_from_middle_json(middle_json, source):
    if not isinstance(middle_json, dict) or not isinstance(middle_json.get("pages"), list):
        raise ValueError("Middle JSON must contain a top-level 'pages' array.")

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

        content_parts = []
        block_types = []
        for block_position, block in enumerate(blocks):
            if not isinstance(block, dict):
                raise ValueError(
                    f"Middle JSON block {block_position} on page_idx {page_idx} "
                    "must be an object."
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

    return documents


def _read_middle_json(path, source):
    try:
        middle_json = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Middle JSON could not be read or parsed: {path}: {exc}") from exc
    return _documents_from_middle_json(middle_json, source)


def _cached_middle_json(cache_entry, pdf_path, identity, key):
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
        return _read_middle_json(middle_json_path, pdf_path.name)
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
):
    """Parse a PDF once with local MinerU and return one unified document per page."""
    pdf_path = Path(path).expanduser()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    resolved_cache_root = Path(cache_root) if cache_root else CACHE_ROOT
    resolved_cache_root.mkdir(parents=True, exist_ok=True)

    pdf_sha256 = _sha256_file(pdf_path)
    key, identity = _cache_key(pdf_path, pdf_sha256)
    cache_entry = resolved_cache_root / key

    if not force:
        cached_documents = _cached_middle_json(cache_entry, pdf_path, identity, key)
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

        documents = _read_middle_json(middle_json_path, pdf_path.name)
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
