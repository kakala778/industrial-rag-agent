"""Ingestion and private artifact I/O, outside the Agent tool action surface."""

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from uuid import uuid4

from .tools import SAFE_ALIAS


APP_ROOT = Path(__file__).resolve().parents[2]


def load_corpus(specifications, *, parser="pymupdf", cache_root=None):
    from ..document_loader import load_pdf
    from .. import mineru_loader
    corpus = {}
    for specification in specifications:
        alias, separator, value = specification.partition("=")
        if not separator or not SAFE_ALIAS.fullmatch(alias) or alias in corpus:
            raise ValueError("documents require unique ALIAS=PATH specifications")
        path = Path(value).expanduser()
        if path.suffix.lower() == ".md":
            documents = [{"text": path.read_text(encoding="utf-8"),
                          "metadata": {"source": alias, "page": None}}]
        elif path.suffix.lower() == ".pdf":
            if parser == "mineru":
                # Strictly cache-only: do not enter loader's parse/mutation path.
                sha = mineru_loader._sha256_file(path)
                key, identity = mineru_loader._cache_key(path, sha)
                root = Path(cache_root) if cache_root else mineru_loader.CACHE_ROOT
                documents = mineru_loader._cached_middle_json(
                    root / key, path, identity, key, representation="structured")
                if documents is None:
                    raise ValueError("matching MinerU cache unavailable; no parsing attempted")
            elif parser == "pymupdf":
                documents = load_pdf(path, parser=parser)
            else:
                raise ValueError("unsupported parser")
        else:
            raise ValueError("only explicit local Markdown/PDF inputs are supported")
        corpus[alias] = documents
    return corpus


def write_private_result(value, *, name="run", app_root=APP_ROOT):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
        raise ValueError("output name must be a safe identifier")
    root = Path(app_root).resolve()
    directory = root / "outputs" / "agent0"
    if not directory.resolve().is_relative_to(root):
        raise ValueError("private output directory escapes application root")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{name}_{uuid4().hex}.json"
    if is_dataclass(value):
        value = asdict(value)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return target


def write_research_report(state, session, *, output_root=APP_ROOT, report_id=None):
    """Write one local UTF-8 M11 report with exclusive creation."""
    from .markdown_report import render_research_report

    if report_id is None:
        report_id = uuid4().hex[:12]
    if not isinstance(report_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", report_id):
        raise ValueError("report ID must be a safe identifier")
    report = render_research_report(state, session)
    root = Path(output_root).resolve()
    directory = root / "outputs" / "agent11"
    if not directory.resolve().is_relative_to(root):
        raise ValueError("private report directory escapes output root")
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = directory / f"report_{timestamp}_{report_id}.md"
    stream = target.open("x", encoding="utf-8", newline="\n")
    try:
        with stream:
            stream.write(report)
    except Exception:
        try:
            target.unlink()
        except OSError:
            pass
        raise
    return target.relative_to(root)
