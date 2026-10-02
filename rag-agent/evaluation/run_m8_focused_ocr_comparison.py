"""Run a local, fixed-cohort comparison of MinerU, Qwen3-VL, and RapidOCR."""

import argparse
import importlib.metadata
import json
import math
import re
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "m8_focused_ocr"
LOCAL_DATA_ROOT = PROJECT_ROOT.parent / "minerU" / "industrial-rag-data"
PDF_DIR = LOCAL_DATA_ROOT / "input"
DATASET_PATH = LOCAL_DATA_ROOT / "qa" / "m6_final_qa.local.json"
M7_ANALYSIS_PATH = PROJECT_ROOT / "outputs" / "m7_failure_analysis.json"
M8_INPUT_MANIFEST_PATH = PROJECT_ROOT / "outputs" / "m8_m6_input_manifest.json"
M8_COMPARISON_PATH = PROJECT_ROOT / "outputs" / "m8_visual_representation_comparison.json"
DEFAULT_OCR_PYTHON = OUTPUT_ROOT / "rapidocr-env" / "Scripts" / "python.exe"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
VISION_MODEL = "qwen3-vl:2b-instruct-q4_K_M"
SCALES = (2, 3)
PRIMARY_SCALE = 3

PROMPT = """你是工业文档视觉文字提取器。

请逐字读取图片区域中的可见内容。

要求：

1. 优先保持原始文字、数字、型号、单位，不做总结。
2. 如果是表格，请按行输出，并保持字段之间的对应关系。
3. 不要根据上下文补全看不清的内容。
4. 看不清的位置输出 `[不确定]`。
5. 不要解释图片，只提取可见事实。

输出：

## Raw Text

## Rows / Fields

## Uncertain"""

# Fixed from the M8.1 audit and the hash-verified Middle JSON block indices.
# These are not reselected using M8.5 outcomes.
EXPERIMENT_CASES = {
    9: {"category": "OCR_FAILURE", "block_type": "table", "block_index": 3},
    10: {"category": "OCR_FAILURE", "block_type": "table", "block_index": 6},
    12: {"category": "OCR_FAILURE", "block_type": "table", "block_index": 18},
    17: {"category": "OCR_FAILURE", "block_type": "image", "block_index": 2},
    20: {"category": "OCR_FAILURE", "block_type": "list", "block_index": 28},
}

_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9._-])[+-]?(?:\d+(?:[.,]\d+)?|\.\d+)(?![0-9.])")
_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*(?![A-Za-z0-9])")
_NUMBER_LITERAL_RE = re.compile(r"[+-]?(?:\d+(?:[.,]\d+)?|\.\d+)\Z")
_MULTIPLIED_NUMBERS_RE = re.compile(r"\d+(?:x\d+)+\Z", re.IGNORECASE)
_UNIT_NAMES = (
    "MPa", "kPa", "Pa", "bar", "psi", "µm", "μm", "um", "mm²", "m²", "mm", "cm", "km",
    "m³", "kg", "mg", "g", "t", "MW", "kW", "W", "kV", "mV",
    "V", "kA", "mA", "A", "Hz", "rpm", "mL", "L", "N·m", "Nm", "°C", "℃",
    "°F", "%", "dB", "DN", "PN", "SN", "IP",
)
_NORMALIZED_UNITS = tuple(
    sorted(
        {unicodedata.normalize("NFKC", unit).casefold() for unit in _UNIT_NAMES},
        key=len,
        reverse=True,
    )
)
_SYMBOL_UNIT_RE = re.compile(r"%|℃|°[CF]|µm|μm|m²|mm²|m³", re.IGNORECASE)


def _normalize(value):
    from evaluation.evaluate_pdf_retrieval import _normalize_evidence_text

    return _normalize_evidence_text(value)


def _critical_tokens(expected_fields):
    tokens = []
    seen = set()

    def add(kind, token):
        key = (kind, _normalize(token))
        if key not in seen:
            seen.add(key)
            tokens.append((kind, token))

    for field in expected_fields:
        text = unicodedata.normalize("NFKC", str(field))
        code_spans = []
        for match in _TOKEN_RE.finditer(text):
            token = match.group(0)
            normalized = _normalize(token)
            has_letters = bool(re.search(r"[a-z]", normalized))
            has_digits = bool(re.search(r"\d", normalized))

            if normalized in _NORMALIZED_UNITS:
                add("unit", token)
                continue

            suffix_unit = next(
                (
                    unit
                    for unit in _NORMALIZED_UNITS
                    if normalized.endswith(unit) and len(normalized) > len(unit)
                ),
                None,
            )
            if suffix_unit:
                numeric_prefix = normalized[: -len(suffix_unit)]
                if _NUMBER_LITERAL_RE.fullmatch(numeric_prefix):
                    if match.start() > 0 and text[match.start() - 1] in "+-":
                        if match.start() == 1 or not text[match.start() - 2].isalnum():
                            numeric_prefix = text[match.start() - 1] + numeric_prefix
                    add("number", numeric_prefix)
                    add("unit", suffix_unit)
                    continue
                if _MULTIPLIED_NUMBERS_RE.fullmatch(numeric_prefix):
                    for number in re.findall(r"\d+", numeric_prefix):
                        add("number", number)
                    add("unit", suffix_unit)
                    continue

            if has_letters and has_digits:
                code_spans.append(match.span())
                add("code", token)

        for match in _NUMBER_RE.finditer(text):
            if any(
                match.start() < code_end and code_start < match.end()
                for code_start, code_end in code_spans
            ):
                continue
            add("number", match.group(0))

        for match in _SYMBOL_UNIT_RE.finditer(text):
            add("unit", match.group(0))
    return tokens


def _critical_token_present(text, kind, token):
    normalized_text = _normalize(text)
    normalized_token = _normalize(token)
    if kind == "number":
        pattern = re.compile(
            r"(?<![0-9.])" + re.escape(normalized_token) + r"(?![0-9.])"
        )
        return bool(pattern.search(normalized_text))
    if kind == "code":
        pattern = re.compile(
            r"(?<![a-z0-9])" + re.escape(normalized_token) + r"(?![a-z0-9])"
        )
        return bool(pattern.search(normalized_text))
    pattern = re.compile(
        r"(?<![a-z])" + re.escape(normalized_token) + r"(?![a-z])"
    )
    return bool(pattern.search(normalized_text))


def _analyze_output(output, expected_fields):
    """Return counts only; expected evidence and OCR text are never serialized."""
    from evaluation.evaluate_pdf_retrieval import _contains_keywords_in_texts

    output = output if isinstance(output, str) else ""
    fields = [field for field in expected_fields if isinstance(field, str) and field.strip()]
    matched_fields = sum(
        _contains_keywords_in_texts([output], [field]) for field in fields
    )
    critical_tokens = _critical_tokens(fields)
    matched_tokens = sum(
        _critical_token_present(output, kind, token)
        for kind, token in critical_tokens
    )
    expected_token_count = len(critical_tokens)
    evidence_hit = bool(fields) and matched_fields == len(fields)
    all_critical_tokens_exact = matched_tokens == expected_token_count
    return {
        "normalized_evidence_hit": evidence_hit,
        "strict_evidence_hit": evidence_hit and all_critical_tokens_exact,
        "matched_field_count": matched_fields,
        "expected_field_count": len(fields),
        "field_recall": matched_fields / len(fields) if fields else 0.0,
        "matched_critical_token_count": matched_tokens,
        "expected_critical_token_count": expected_token_count,
        "critical_token_error_count": expected_token_count - matched_tokens,
        "exact_critical_token_accuracy": (
            matched_tokens / expected_token_count if expected_token_count else None
        ),
    }


def _safe_metric_record(metrics):
    allowed = {
        "normalized_evidence_hit",
        "strict_evidence_hit",
        "matched_field_count",
        "expected_field_count",
        "field_recall",
        "matched_critical_token_count",
        "expected_critical_token_count",
        "critical_token_error_count",
        "exact_critical_token_accuracy",
        "inference_seconds",
        "model_load_seconds",
        "width",
        "height",
    }
    safe = {}
    for key in allowed:
        value = metrics.get(key)
        if isinstance(value, bool) or value is None:
            safe[key] = value
        elif isinstance(value, (int, float)) and math.isfinite(value):
            safe[key] = value
    return safe


def _anonymous_case_record(case, outputs):
    """Build a strict allowlist record containing no document or model text."""
    expected_fields = case.get("expected_fields", [])
    record = {
        "question_id": int(case["question_id"]),
        "category": str(case.get("category", "OCR_FAILURE")),
        "page": int(case["page_number"]),
        "block_type": str(case["block_type"]),
        "block_index": int(case["block_index"]),
        "bbox_normalized": [float(value) for value in case["bbox_normalized"]],
        "expected_field_count": len(expected_fields),
    }
    for method in ("mineru", "qwen3vl", "rapidocr"):
        method_output = outputs.get(method, {})
        if method == "mineru":
            record[method] = {
                name: _safe_metric_record(method_output.get(name, {}))
                for name in ("structured", "middle_json")
            }
        else:
            record[method] = {
                f"{scale}x": _safe_metric_record(method_output.get(f"{scale}x", {}))
                for scale in SCALES
            }
    return record


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _assert_git_ignored(path):
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", str(path)],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("The experiment output path is not Git-ignored.")


def _safe_output_directory(path):
    requested = Path(path).expanduser().resolve()
    root = OUTPUT_ROOT.resolve()
    if requested != root and root not in requested.parents:
        raise ValueError("M8.5 images and local responses must remain under outputs/.")
    requested.mkdir(parents=True, exist_ok=True)
    _assert_git_ignored(requested)
    return requested


def _request_json(path, payload=None, timeout=30):
    from urllib.request import Request, urlopen

    url = f"{OLLAMA_BASE_URL}{path}"
    if payload is None:
        request = Request(url, method="GET")
    else:
        request = Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _vision_model_info():
    version = _request_json("/api/version").get("version", "unknown")
    details = _request_json("/api/show", {"model": VISION_MODEL}).get("details", {})
    ollama = shutil_which("ollama")
    model_size = None
    if ollama:
        listing = subprocess.run(
            [ollama, "list"], capture_output=True, text=True, check=False
        ).stdout
        for line in listing.splitlines():
            if line.startswith(VISION_MODEL + " "):
                pieces = line.split()
                if len(pieces) >= 3:
                    model_size = " ".join(pieces[2:4]) if len(pieces) >= 4 else pieces[2]
                break
    return {
        "tag": VISION_MODEL,
        "family": details.get("family"),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
        "reported_size": model_size,
        "ollama_version": version,
    }


def shutil_which(command):
    from shutil import which

    return which(command)


def _normalized_bbox_to_rect(bbox, page_rect, pymupdf):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("MinerU block bbox must contain four normalized values.")
    try:
        x0, y0, x1, y1 = (float(value) for value in bbox)
    except (TypeError, ValueError) as exc:
        raise ValueError("MinerU block bbox must be numeric.") from exc
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("MinerU block bbox is not ordered in [0, 1].")
    rect = pymupdf.Rect(page_rect)
    return pymupdf.Rect(
        rect.x0 + x0 * rect.width,
        rect.y0 + y0 * rect.height,
        rect.x0 + x1 * rect.width,
        rect.y0 + y1 * rect.height,
    )


def _render_crop(pdf_path, page_number, bbox, scale, image_path):
    import pymupdf

    with pymupdf.open(pdf_path) as document:
        if page_number < 1 or page_number > document.page_count:
            raise ValueError("A selected page number is outside the local M6 PDF.")
        page = document.load_page(page_number - 1)
        clip = _normalized_bbox_to_rect(bbox, page.rect, pymupdf)
        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(scale, scale), clip=clip, alpha=False
        )
        pixmap.save(image_path)
        return {"width": int(pixmap.width), "height": int(pixmap.height)}


def _load_fixed_cases():
    """Verify fixed M6/M7/M8 inputs and load exact target blocks from cache only."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from evaluation.evaluate_pdf_retrieval import _expected_source, load_questions
    from evaluation.run_m8_vision_feasibility import _load_experiment_inputs
    from src.mineru_loader import (
        CACHE_ROOT,
        TEXT_BLOCK_TYPES,
        _block_index,
        _cache_key,
        _content_strings,
        _cached_middle_json,
        _sha256_file,
        _typed_text_for_visual_block,
    )

    # Reuse the existing M6 manifest/cache preflight. It fails closed before
    # model calls and verifies every local PDF cache without invoking MinerU.
    _m83_inputs, comparison = _load_experiment_inputs()
    questions = load_questions(DATASET_PATH)
    m7 = json.loads(M7_ANALYSIS_PATH.read_text(encoding="utf-8"))
    m7_by_id = {case["question_id"]: case for case in m7.get("cases", [])}
    m8_by_id = {
        case["question_id"]: case
        for case in comparison["modes"]["structured"]["parsing_cases"]
    }
    pdf_by_name = {path.name: path for path in PDF_DIR.glob("*.pdf")}
    prepared = []

    for question_id, config in EXPERIMENT_CASES.items():
        if question_id > len(questions):
            raise ValueError("A fixed M8.1 case is missing from the local M6 QA set.")
        item = questions[question_id - 1]
        m7_case = m7_by_id.get(question_id)
        m8_case = m8_by_id.get(question_id)
        if not m7_case or m7_case.get("classification") != "PARSING_FAILURE":
            raise ValueError("A selected case is no longer an M7 parsing failure.")
        if not m8_case or m8_case.get("audit_category") != config["category"]:
            raise ValueError("A selected case no longer matches its fixed M8.1 type.")
        if m8_case.get("evidence_in_representation"):
            raise ValueError("A selected OCR failure now has evidence in M8.2 baseline.")

        source_name = _expected_source(item)
        pdf_path = pdf_by_name.get(source_name)
        page_number = item.get("expected_page")
        expected_fields = item.get("expected_keywords", [])
        if pdf_path is None or not isinstance(page_number, int) or not expected_fields:
            raise ValueError("A fixed M6 case lacks a verified page or evidence fields.")

        pdf_digest = _sha256_file(pdf_path)
        cache_key, identity = _cache_key(pdf_path, pdf_digest)
        cache_entry = CACHE_ROOT / cache_key
        cached = _cached_middle_json(
            cache_entry,
            pdf_path,
            identity,
            cache_key,
            representation="structured_full",
        )
        if cached is None:
            raise RuntimeError("A selected hash-matched MinerU cache is missing.")
        middle_path = cache_entry / "output" / f"{pdf_path.stem}.json"
        middle_json = json.loads(middle_path.read_text(encoding="utf-8"))
        page = next(
            (
                page
                for page in middle_json.get("pages", [])
                if page.get("page_idx") == page_number - 1
            ),
            None,
        )
        if page is None:
            raise ValueError("A fixed M8.1 target page is missing from MinerU cache.")

        target = None
        for block_position, block in enumerate(page.get("blocks", [])):
            if (
                block.get("type") == config["block_type"]
                and _block_index(block, block_position) == config["block_index"]
            ):
                if target is not None:
                    raise ValueError("A fixed MinerU target block is ambiguous.")
                target = block
        if target is None:
            raise ValueError("A fixed MinerU target block is missing from cache.")
        bbox = target.get("bbox")
        if (
            not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox)
            or not (0 <= bbox[0] < bbox[2] <= 1 and 0 <= bbox[1] < bbox[3] <= 1)
        ):
            raise ValueError("A fixed MinerU target block has no normalized bbox.")

        block_type = config["block_type"]
        if block_type in TEXT_BLOCK_TYPES:
            structured_text = "\n".join(_content_strings(target.get("content")))
            middle_text = structured_text
        elif block_type in {"table", "chart", "image"}:
            structured_text = _typed_text_for_visual_block(
                target, block_type, representation="structured"
            )
            middle_text = _typed_text_for_visual_block(
                target, block_type, representation="structured_full"
            )
        else:
            structured_text = "\n".join(_content_strings(target.get("content")))
            middle_text = structured_text

        prepared.append(
            {
                "question_id": question_id,
                "category": config["category"],
                "page_number": page_number,
                "block_type": config["block_type"],
                "block_index": config["block_index"],
                "bbox_normalized": [float(value) for value in bbox],
                "expected_fields": expected_fields,
                "pdf_path": pdf_path,
                "mineru_structured_text": structured_text,
                "mineru_middle_json_text": middle_text,
            }
        )
    return prepared


def _call_vision_model(image_path):
    import base64

    image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "model": VISION_MODEL,
        "prompt": PROMPT,
        "images": [image_data],
        "stream": False,
        "keep_alive": "5m",
        "options": {"temperature": 0, "num_predict": 2048},
    }
    started = time.perf_counter()
    response = _request_json("/api/generate", payload, timeout=900)
    wall_seconds = time.perf_counter() - started
    text = response.get("response")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Ollama returned an empty local vision response.")
    return {
        "raw_text": text,
        "inference_seconds": wall_seconds,
        "ollama_total_seconds": _nanoseconds_to_seconds(response.get("total_duration")),
        "ollama_load_seconds": _nanoseconds_to_seconds(response.get("load_duration")),
    }


def _nanoseconds_to_seconds(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return round(value / 1_000_000_000, 4)
    return None


def _rapidocr_worker_provider(engine):
    found = set()
    visited = set()

    def walk(obj, depth=0):
        if obj is None or depth > 6 or id(obj) in visited:
            return
        visited.add(id(obj))
        get_providers = getattr(obj, "get_providers", None)
        if callable(get_providers):
            try:
                found.update(get_providers())
            except Exception:
                pass
        if depth >= 6 or not hasattr(obj, "__dict__"):
            return
        for name, value in vars(obj).items():
            if name.startswith("_"):
                continue
            if isinstance(value, (list, tuple)):
                for child in value[:10]:
                    walk(child, depth + 1)
            else:
                walk(value, depth + 1)

    for name in ("text_det", "text_cls", "text_rec"):
        walk(getattr(engine, name, None))
    return sorted(found)


def _run_rapidocr_worker(input_path, result_path):
    """Entry point for the isolated Python 3.12 OCR environment."""
    import importlib.metadata
    import rapidocr
    from rapidocr import RapidOCR

    inputs = json.loads(Path(input_path).read_text(encoding="utf-8"))
    started = time.perf_counter()
    engine = RapidOCR()
    load_seconds = time.perf_counter() - started
    records = []
    for item in inputs:
        call_started = time.perf_counter()
        result = engine(str(item["image_path"]))
        inference_seconds = time.perf_counter() - call_started
        recognized = getattr(result, "txts", None) if result is not None else None
        if recognized is None:
            recognized_text = ""
        else:
            recognized_text = "\n".join(
                str(line) for line in recognized if line is not None and str(line).strip()
            )
        records.append(
            {
                "question_id": int(item["question_id"]),
                "scale": str(item["scale"]),
                "raw_text": recognized_text,
                "inference_seconds": round(inference_seconds, 4),
            }
        )
    model_dir = Path(rapidocr.__file__).resolve().parent / "models"
    model_files = sorted(model_dir.glob("*.onnx"))
    _write_json(
        Path(result_path),
        {
            "engine": "RapidOCR",
            "version": importlib.metadata.version("rapidocr"),
            "onnxruntime_version": importlib.metadata.version("onnxruntime"),
            "providers": _rapidocr_worker_provider(engine),
            "model_file_count": len(model_files),
            "model_size_bytes": sum(path.stat().st_size for path in model_files),
            "engine_load_seconds": round(load_seconds, 4),
            "cases": records,
        },
    )
    return 0


def _launch_rapidocr(python_path, image_records, run_dir):
    python_path = Path(python_path).expanduser().resolve()
    if not python_path.is_file():
        raise FileNotFoundError("The isolated RapidOCR Python interpreter is missing.")
    request_path = run_dir / "rapidocr_input.local.json"
    response_path = run_dir / "rapidocr_responses.local.json"
    worker_log_path = run_dir / "rapidocr_worker.local.log"
    _write_json(request_path, image_records)
    completed = subprocess.run(
        [str(python_path), str(Path(__file__).resolve()), "--rapidocr-worker", str(request_path), str(response_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    worker_log_path.write_text(
        (completed.stdout or "") + (completed.stderr or ""), encoding="utf-8"
    )
    if completed.returncode != 0 or not response_path.is_file():
        raise RuntimeError("The isolated RapidOCR worker failed; see ignored local log.")
    return json.loads(response_path.read_text(encoding="utf-8"))


def _device_observation():
    try:
        running = _request_json("/api/ps").get("models", [])
    except (OSError, ValueError):
        return "not reported by Ollama"
    for model in running:
        if model.get("name") == VISION_MODEL or model.get("model") == VISION_MODEL:
            size = model.get("size")
            vram = model.get("size_vram")
            if isinstance(size, (int, float)) and isinstance(vram, (int, float)):
                if vram >= size and size > 0:
                    return "GPU (Ollama reports all model bytes in VRAM)"
                if vram == 0:
                    return "CPU (Ollama reports zero model bytes in VRAM)"
                return "mixed/partial GPU (Ollama reports partial model bytes in VRAM)"
    return "not reported by Ollama after inference"


def _metric_rollup(cases, selector):
    selected = [selector(case) for case in cases]
    selected = [value for value in selected if value is not None]
    if not selected:
        return {
            "strict_evidence_hits": 0,
            "full_evidence_hits": 0,
            "cases": 0,
            "matched_fields": 0,
            "expected_fields": 0,
            "field_recall": None,
            "matched_critical_tokens": 0,
            "expected_critical_tokens": 0,
            "critical_token_errors": 0,
            "exact_critical_token_accuracy": None,
        }
    matched_fields = sum(item["matched_field_count"] for item in selected)
    expected_fields = sum(item["expected_field_count"] for item in selected)
    matched_tokens = sum(item["matched_critical_token_count"] for item in selected)
    expected_tokens = sum(item["expected_critical_token_count"] for item in selected)
    return {
        "strict_evidence_hits": sum(bool(item.get("strict_evidence_hit")) for item in selected),
        "full_evidence_hits": sum(bool(item["normalized_evidence_hit"]) for item in selected),
        "cases": len(selected),
        "matched_fields": matched_fields,
        "expected_fields": expected_fields,
        "field_recall": matched_fields / expected_fields if expected_fields else 0.0,
        "matched_critical_tokens": matched_tokens,
        "expected_critical_tokens": expected_tokens,
        "critical_token_errors": expected_tokens - matched_tokens,
        "exact_critical_token_accuracy": matched_tokens / expected_tokens if expected_tokens else None,
    }


def _new_human_review(cases, anonymous_cases=None):
    methods = {
        "mineru_structured": ("mineru", "structured"),
        "mineru_middle_json": ("mineru", "middle_json"),
        "qwen3vl_3x": ("qwen3vl", "3x"),
        "rapidocr_3x": ("rapidocr", "3x"),
    }
    results_by_id = {
        case["question_id"]: case for case in (anonymous_cases or [])
    }
    return {
        "note": (
            "Automated target-evidence metrics are separate from the manual "
            "visual review. Keep uncertain handwriting unverified."
        ),
        "cases": [
            {
                "case_id": f"Q{case['question_id']:02d}",
                "checks": {
                    method: _human_review_item(
                        results_by_id.get(case["question_id"]), target
                    )
                    for method, target in methods.items()
                },
            }
            for case in cases
        ],
    }


def _human_review_item(anonymous_case, target):
    if anonymous_case is None:
        status = "pending"
    else:
        metrics = anonymous_case[target[0]][target[1]]
        if metrics.get("strict_evidence_hit"):
            status = "correct_target_evidence"
        elif (
            metrics.get("matched_field_count", 0) > 0
            and metrics.get("matched_critical_token_count", 0) > 0
        ):
            status = "partial_target_evidence"
        else:
            status = "target_evidence_not_recovered"
    return {
        "automated_target_evidence_status": status,
        "manual_visual_assessment": "pending",
        "confirmed_unsupported_field_count": None,
        "unverified_region_count": None,
    }


def run_experiment(output_dir=None, ocr_python=DEFAULT_OCR_PYTHON):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import pymupdf

    cases = _load_fixed_cases()
    model_info = _vision_model_info()
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = _safe_output_directory(output_dir or OUTPUT_ROOT)
    run_dir = output_root / f"run_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    _assert_git_ignored(run_dir)

    raw_responses = []
    anonymous_cases = []
    image_records = []
    per_case_work = []
    for case in cases:
        crop_map = {}
        for scale in SCALES:
            image_name = f"case_{case['question_id']}_{scale}x.png"
            image_path = run_dir / image_name
            size = _render_crop(
                case["pdf_path"], case["page_number"], case["bbox_normalized"],
                float(scale), image_path,
            )
            crop_map[f"{scale}x"] = {
                "path": image_path,
                "width": size["width"],
                "height": size["height"],
            }
            image_records.append(
                {
                    "question_id": case["question_id"],
                    "scale": f"{scale}x",
                    "image_path": str(image_path),
                }
            )
        per_case_work.append((case, crop_map))

    _write_json(run_dir / "human_review.local.json", _new_human_review(cases))
    _write_json(run_dir / "model_responses.local.json", raw_responses)

    for case, crop_map in per_case_work:
        outputs = {
            "mineru": {
                "structured": _analyze_output(
                    case["mineru_structured_text"], case["expected_fields"]
                ),
                "middle_json": _analyze_output(
                    case["mineru_middle_json_text"], case["expected_fields"]
                ),
            },
            "qwen3vl": {},
            "rapidocr": {},
        }
        for scale in SCALES:
            model_result = _call_vision_model(crop_map[f"{scale}x"]["path"])
            metrics = _analyze_output(model_result["raw_text"], case["expected_fields"])
            metrics.update(
                inference_seconds=round(model_result["inference_seconds"], 4),
                model_load_seconds=model_result["ollama_load_seconds"],
                width=crop_map[f"{scale}x"]["width"],
                height=crop_map[f"{scale}x"]["height"],
            )
            outputs["qwen3vl"][f"{scale}x"] = metrics
            raw_responses.append(
                {
                    "question_id": case["question_id"],
                    "method": "qwen3vl",
                    "scale": f"{scale}x",
                    "raw_text": model_result["raw_text"],
                    "wall_seconds": round(model_result["inference_seconds"], 4),
                    "ollama_total_seconds": model_result["ollama_total_seconds"],
                    "ollama_load_seconds": model_result["ollama_load_seconds"],
                }
            )
            _write_json(run_dir / "model_responses.local.json", raw_responses)
        anonymous_cases.append(_anonymous_case_record(case, outputs))
        print(f"Case Q{case['question_id']:02d}: local crops processed by Qwen3-VL.", flush=True)

    rapidocr_result = _launch_rapidocr(ocr_python, image_records, run_dir)
    rapidocr_by_key = {
        (record["question_id"], record["scale"]): record
        for record in rapidocr_result.get("cases", [])
    }
    if len(rapidocr_by_key) != len(cases) * len(SCALES):
        raise RuntimeError("RapidOCR did not return every fixed case/scale pair.")

    for case, record in zip(cases, anonymous_cases):
        for scale in SCALES:
            result = rapidocr_by_key[(case["question_id"], f"{scale}x")]
            metrics = _analyze_output(result["raw_text"], case["expected_fields"])
            metrics.update(
                inference_seconds=round(float(result["inference_seconds"]), 4),
                model_load_seconds=(
                    rapidocr_result.get("engine_load_seconds") if scale == SCALES[0] else 0.0
                ),
                width=record["qwen3vl"][f"{scale}x"].get("width"),
                height=record["qwen3vl"][f"{scale}x"].get("height"),
            )
            record["rapidocr"][f"{scale}x"] = _safe_metric_record(metrics)
            raw_responses.append(
                {
                    "question_id": case["question_id"],
                    "method": "rapidocr",
                    "scale": f"{scale}x",
                    "raw_text": result["raw_text"],
                    "inference_seconds": result["inference_seconds"],
                }
            )
        _write_json(run_dir / "model_responses.local.json", raw_responses)

    qwen_summary = {
        f"{scale}x": _metric_rollup(
            anonymous_cases,
            lambda case: case["qwen3vl"][f"{scale}x"],
        )
        for scale in SCALES
    }
    rapid_summary = {
        f"{scale}x": _metric_rollup(
            anonymous_cases,
            lambda case: case["rapidocr"][f"{scale}x"],
        )
        for scale in SCALES
    }
    totals = {
        "mineru_structured": _metric_rollup(
            anonymous_cases, lambda case: case["mineru"]["structured"]
        ),
        "mineru_middle_json": _metric_rollup(
            anonymous_cases, lambda case: case["mineru"]["middle_json"]
        ),
        "qwen3vl": qwen_summary,
        "rapidocr": rapid_summary,
    }
    average_qwen_time = sum(
        case["qwen3vl"][f"{PRIMARY_SCALE}x"]["inference_seconds"]
        for case in anonymous_cases
    ) / len(anonymous_cases)
    average_ocr_time = sum(
        case["rapidocr"][f"{PRIMARY_SCALE}x"]["inference_seconds"]
        for case in anonymous_cases
    ) / len(anonymous_cases)
    summary = {
        "experiment": "M8.5 Focused OCR Comparison",
        "cohort": "fixed five M8.1 OCR_FAILURE cases",
        "mineru_was_reparsed": False,
        "primary_scale": f"{PRIMARY_SCALE}x",
        "scales": [f"{scale}x" for scale in SCALES],
        "vision_model": model_info,
        "dedicated_ocr": {
            "name": rapidocr_result.get("engine"),
            "version": rapidocr_result.get("version"),
            "onnxruntime_version": rapidocr_result.get("onnxruntime_version"),
            "providers": rapidocr_result.get("providers", []),
            "model_file_count": rapidocr_result.get("model_file_count"),
            "model_size_bytes": rapidocr_result.get("model_size_bytes"),
            "engine_load_seconds": rapidocr_result.get("engine_load_seconds"),
        },
        "vision_inference_device": _device_observation(),
        "average_primary_scale_inference_seconds": {
            "qwen3vl": round(average_qwen_time, 4),
            "rapidocr": round(average_ocr_time, 4),
        },
        "totals": totals,
        "cases": anonymous_cases,
    }
    _write_json(run_dir / "anonymous_summary.json", summary)
    _write_json(
        run_dir / "human_review.local.json",
        _new_human_review(cases, anonymous_cases),
    )
    print(f"M8.5 complete; anonymous metrics and local artifacts are under outputs/m8_focused_ocr/{run_dir.name}.")
    return run_dir, summary


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Compare current MinerU evidence with same-region local OCR methods."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--ocr-python", type=Path, default=DEFAULT_OCR_PYTHON)
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if len(sys.argv) > 1 and sys.argv[1] == "--rapidocr-worker":
        if len(sys.argv) != 4:
            return 2
        return _run_rapidocr_worker(sys.argv[2], sys.argv[3])
    try:
        run_dir, summary = run_experiment(args.output_dir, args.ocr_python)
    except (OSError, RuntimeError, ValueError, KeyError, ImportError, json.JSONDecodeError) as exc:
        print(
            f"M8.5 stopped: {type(exc).__name__}; private source/model text was not printed.",
            file=sys.stderr,
        )
        return 1
    print(f"Cases: {len(summary['cases'])}; primary scale: {summary['primary_scale']}.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--rapidocr-worker":
        if len(sys.argv) != 4:
            raise SystemExit(2)
        raise SystemExit(_run_rapidocr_worker(sys.argv[2], sys.argv[3]))
    raise SystemExit(main())
