"""Run the local M8.4 block-crop versus full-page vision experiment."""

import argparse
import base64
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pymupdf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DATA_ROOT = PROJECT_ROOT.parent / "minerU" / "industrial-rag-data"
PDF_DIR = LOCAL_DATA_ROOT / "input"
DATASET_PATH = LOCAL_DATA_ROOT / "qa" / "m6_final_qa.local.json"
M7_ANALYSIS_PATH = PROJECT_ROOT / "outputs" / "m7_failure_analysis.json"
M8_COMPARISON_PATH = PROJECT_ROOT / "outputs" / "m8_visual_representation_comparison.json"
M8_INPUT_MANIFEST_PATH = PROJECT_ROOT / "outputs" / "m8_m6_input_manifest.json"
M8_3_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "m8_vision_test"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "m8_vision_block_test"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
MODEL = "qwen3-vl:2b-instruct-q4_K_M"
OUTPUT_HEADINGS = (
    "## Text",
    "## Entities",
    "## Parameters",
    "## Table",
    "## Relations",
    "## Uncertainty",
)
PROMPT = """你是工业文档视觉解析专家。

请分析这个图片区域，并生成可以用于工业知识库检索的信息。

要求：

1. 提取图片中的明确文字。
2. 提取设备、部件、编号、参数。
3. 如果存在表格，恢复行列关系。
4. 如果存在图纸、流程图、结构图：
   - 描述对象
   - 描述连接关系
   - 描述标注对应关系

5. 不要生成图片中不存在的信息。
6. 不确定请标记。

输出：

## Text

## Entities

## Parameters

## Table

## Relations

## Uncertainty"""

# The indices refer to the hash-verified M6 Middle JSON pages. Q10 is the
# audited table OCR miss; Q29 and Q32 use visual blocks and adjacent text.
EXPERIMENT_CASES = (
    {
        "question_id": 10,
        "category": "OCR_FAILURE",
        "block_type": "table",
        "block_index": 6,
        "context_block_indices": (6,),
        "modes": ("block_crop", "block_crop_plus_mineru_text", "full_page"),
    },
    {
        "question_id": 16,
        "category": "IMAGE_INFORMATION_LOSS",
        "block_type": None,
        "block_index": None,
        "context_block_indices": (),
        "modes": ("full_page",),
    },
    {
        "question_id": 29,
        "category": "IMAGE_INFORMATION_LOSS",
        "block_type": "image",
        "block_index": 0,
        "context_block_indices": (0, 3),
        "modes": ("block_crop", "block_crop_plus_mineru_text", "full_page"),
    },
    {
        "question_id": 32,
        "category": "LAYOUT_RELATION_FAILURE",
        "block_type": "image",
        "block_index": 2,
        "context_block_indices": (2, 4),
        "modes": ("block_crop", "block_crop_plus_mineru_text", "full_page"),
    },
    {
        "question_id": 5,
        "category": "SANITY_SUCCESS",
        "block_type": None,
        "block_index": None,
        "context_block_indices": (),
        "modes": ("full_page",),
    },
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_pdf_retrieval import (  # noqa: E402
    _contains_keywords_in_texts,
    _expected_source,
    _matches_case_scope,
    load_questions,
)
from evaluation.run_m8_vision_feasibility import (  # noqa: E402
    _load_experiment_inputs,
    _model_info,
)
from src.document_loader import load_pdf  # noqa: E402


def _section_content(output, heading):
    lines = output.splitlines()
    for index, line in enumerate(lines):
        if line.strip().casefold() == heading.casefold():
            content = []
            for following in lines[index + 1 :]:
                if following.strip().startswith("## "):
                    break
                content.append(following.strip())
            return "\n".join(line for line in content if line)
    return ""


def _meaningful_section(output, heading):
    content = _section_content(output, heading).strip().casefold()
    return content not in {"", "-", "none", "n/a", "not applicable", "无", "未提供"}


def _analyze_vision_output(output, expected_keywords, category):
    meaningful = {
        heading: _meaningful_section(output, heading) for heading in OUTPUT_HEADINGS
    }
    matched_fields = sum(
        _contains_keywords_in_texts([output], [keyword])
        for keyword in expected_keywords
    )
    if category == "OCR_FAILURE":
        relevant = meaningful["## Table"] or meaningful["## Parameters"]
    elif category == "IMAGE_INFORMATION_LOSS":
        relevant = (
            meaningful["## Text"]
            or meaningful["## Entities"]
            or meaningful["## Relations"]
        )
    elif category == "LAYOUT_RELATION_FAILURE":
        relevant = meaningful["## Relations"]
    else:
        relevant = any(meaningful.values())
    evidence_hit = bool(expected_keywords) and _contains_keywords_in_texts(
        [output], expected_keywords
    )
    return {
        "vision_expected_evidence_hit": bool(evidence_hit),
        "matched_expected_field_count": matched_fields,
        "expected_field_count": len(expected_keywords),
        "required_sections_present": sum(
            heading in output for heading in OUTPUT_HEADINGS
        ),
        "meaningful_sections": sum(meaningful.values()),
        "structured_document_candidate": bool(evidence_hit and relevant),
    }


def _anonymous_case_record(
    case,
    input_mode,
    analysis,
    *,
    block_type=None,
    block_index=None,
    block_bbox=None,
    context_block_count=0,
):
    mineru_case = case.get("mineru_case", {})
    bbox = None
    if isinstance(block_bbox, (list, tuple)) and len(block_bbox) == 4:
        bbox = [float(value) for value in block_bbox]
    return {
        "question_id": int(case["question_id"]),
        "category": case["category"],
        "page": int(case["page_number"]),
        "input_mode": input_mode,
        "block_type": block_type,
        "block_index": block_index,
        "block_bbox_normalized": bbox,
        "context_block_count": int(context_block_count),
        "mineru_evidence_present": bool(
            mineru_case.get("evidence_in_representation", False)
        ),
        "mineru_reranker_top3_page_hit": mineru_case.get(
            "reranker_top3_page_hit"
        ),
        "mineru_reranker_evidence_hit": mineru_case.get(
            "reranker_normalized_evidence_hit"
        ),
        **analysis,
    }


def _normalized_bbox_to_rect(bbox, page_rect):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("MinerU block bbox must contain four normalized numbers.")
    try:
        x0, y0, x1, y1 = (float(value) for value in bbox)
    except (TypeError, ValueError) as exc:
        raise ValueError("MinerU block bbox must be numeric.") from exc
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("MinerU block bbox must be ordered and normalized to [0, 1].")

    rect = pymupdf.Rect(page_rect)
    width, height = rect.width, rect.height
    return pymupdf.Rect(
        rect.x0 + x0 * width,
        rect.y0 + y0 * height,
        rect.x0 + x1 * width,
        rect.y0 + y1 * height,
    )


def _render_page(pdf_path, page_number, output_path, scale=2.0, clip=None):
    with pymupdf.open(pdf_path) as document:
        if page_number < 1 or page_number > document.page_count:
            raise ValueError("Selected page is outside the local M6 PDF.")
        page = document.load_page(page_number - 1)
        page_clip = _normalized_bbox_to_rect(clip, page.rect) if clip else None
        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(scale, scale),
            clip=page_clip,
            alpha=False,
        )
        pixmap.save(output_path)


def _request_json(path, payload=None, timeout=30):
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


def _call_vision_model(image_path, mineru_context=""):
    user_content = "待分析图片见附件。"
    if mineru_context:
        user_content += "\n\nMinerU 已有文本上下文（不代表已确认语义关联）：\n"
        user_content += mineru_context
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": PROMPT},
            {
                "role": "user",
                "content": user_content,
                "images": [base64.b64encode(image_path.read_bytes()).decode("ascii")],
            },
        ],
        "stream": False,
        "keep_alive": "5m",
        "options": {"temperature": 0, "num_predict": 2048},
    }
    result = _request_json("/api/chat", payload, timeout=900)
    output = (result.get("message") or {}).get("content")
    if not isinstance(output, str) or not output.strip():
        raise RuntimeError("Ollama returned an empty vision response.")
    return output


def _find_block(documents, block_type, block_index):
    matches = []
    for document in documents:
        metadata = document.get("metadata") or {}
        if (
            metadata.get("block_type") == block_type
            and metadata.get("block_index") == block_index
        ):
            matches.append(document)
    if len(matches) != 1:
        raise ValueError("A selected MinerU block is missing or ambiguous.")
    return matches[0]


def _context_for_blocks(documents, block_indices):
    selected = []
    for block_index in block_indices:
        matches = [
            document
            for document in documents
            if (document.get("metadata") or {}).get("block_index") == block_index
        ]
        if len(matches) != 1:
            raise ValueError("A selected MinerU context block is missing or ambiguous.")
        text = str(matches[0].get("text", "")).strip()
        if text:
            selected.append(text)
    return "\n\n".join(selected), len(selected)


def _m83_baseline():
    summaries = sorted(M8_3_OUTPUT_ROOT.glob("run_*/anonymous_summary.json"))
    if not summaries:
        raise FileNotFoundError("The prior anonymous M8.3 result is missing.")
    summary = json.loads(summaries[-1].read_text(encoding="utf-8"))
    return {
        case["question_id"]: bool(case.get("vision_expected_evidence_hit"))
        for case in summary.get("cases", [])
    }


def _prepare_cases():
    selected_m83, m8_comparison = _load_experiment_inputs()
    questions = load_questions(DATASET_PATH)
    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    pdf_by_name = {path.name: path for path in pdf_paths}
    m8_by_id = {
        case["question_id"]: case
        for case in m8_comparison["modes"]["structured"]["parsing_cases"]
    }
    m83_inputs = {case["question_id"]: case for case in selected_m83}
    m7 = json.loads(M7_ANALYSIS_PATH.read_text(encoding="utf-8"))
    m7_by_id = {case["question_id"]: case for case in m7.get("cases", [])}

    prepared = []
    documents_by_source = {}
    for config in EXPERIMENT_CASES:
        question_id = config["question_id"]
        item = questions[question_id - 1]
        source = _expected_source(item)
        pdf_path = pdf_by_name.get(source)
        page_number = item.get("expected_page")
        if pdf_path is None or not isinstance(page_number, int):
            raise ValueError("A selected case lacks a verified local PDF/page mapping.")
        if not item.get("expected_keywords"):
            raise ValueError("A selected case lacks M6 normalized evidence keys.")

        documents = documents_by_source.get(source)
        if documents is None:
            # parser='mineru' reuses the hash-matched Middle JSON cache verified
            # by the M8.3 loader above; the experiment never invokes MinerU.
            documents = load_pdf(
                pdf_path,
                parser="mineru",
                force=False,
                representation="structured_full",
            )
            documents_by_source[source] = documents
        page_documents = [
            document
            for document in documents
            if _matches_case_scope(item, document)
        ]
        mineru_evidence = _contains_keywords_in_texts(
            [document.get("text", "") for document in page_documents],
            item["expected_keywords"],
        )

        if question_id == 5:
            m7_case = m7_by_id.get(question_id, {})
            if (
                m7_case.get("classification") != "NOT_A_BASELINE_FAILURE"
                or not mineru_evidence
            ):
                raise ValueError("The known-success sanity case is no longer valid.")
            mineru_case = {
                "evidence_in_representation": True,
                "reranker_top3_page_hit": True,
                "reranker_normalized_evidence_hit": True,
            }
        else:
            mineru_case = m8_by_id.get(question_id)
            if not mineru_case or mineru_case.get("audit_category") != config["category"]:
                raise ValueError("An experiment case no longer matches its M8.1 type.")
            if bool(mineru_case.get("evidence_in_representation")) != bool(mineru_evidence):
                raise ValueError("Cached MinerU evidence no longer matches M8.2 analysis.")

        actual_m83 = m83_inputs.get(question_id)
        if question_id in {10, 16, 32} and actual_m83 is None:
            raise ValueError("An M8.3 comparison case is missing its verified PDF mapping.")
        prepared.append(
            {
                **config,
                "page_number": page_number,
                "pdf_path": pdf_path,
                "expected_keywords": item["expected_keywords"],
                "mineru_case": mineru_case,
                "mineru_evidence_present": bool(mineru_evidence),
                "page_documents": page_documents,
                "m83_evidence_hit": None,
            }
        )

    m83 = _m83_baseline()
    for case in prepared:
        case["m83_evidence_hit"] = m83.get(case["question_id"])
    if any(prepared_case["question_id"] not in m83 for prepared_case in prepared if prepared_case["question_id"] in {10, 16, 32}):
        raise ValueError("M8.3 does not contain all prior full-page comparison cases.")
    return prepared


def _copy_m83_full_page(question_id, destination):
    run_dirs = sorted(path for path in M8_3_OUTPUT_ROOT.glob("run_*") if path.is_dir())
    for run_dir in reversed(run_dirs):
        candidate = run_dir / f"case_{question_id}_page.png"
        if candidate.is_file():
            shutil.copyfile(candidate, destination)
            return True
    return False


def _safe_output_directory(path):
    requested = Path(path).expanduser().resolve()
    output_root = OUTPUT_ROOT.resolve()
    if requested != output_root and output_root not in requested.parents:
        raise ValueError("M8.4 images and model output must stay under ignored outputs/.")
    requested.mkdir(parents=True, exist_ok=True)
    return requested


def _write_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_experiment(output_dir=None):
    cases = _prepare_cases()
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = _safe_output_directory(output_dir or OUTPUT_ROOT)
    run_dir = output_root / f"run_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    model_info = _model_info()
    anonymous_records = []
    raw_responses = []

    for case in cases:
        question_id = case["question_id"]
        block_path = None
        block = None
        context, context_block_count = "", 0
        if case["block_type"] is not None:
            block = _find_block(
                case["page_documents"],
                case["block_type"],
                case["block_index"],
            )
            block_metadata = block.get("metadata") or {}
            block_bbox = block_metadata.get("bbox")
            if not block_bbox:
                raise ValueError("A selected MinerU block has no cached bbox.")
            block_path = run_dir / f"case_{question_id}_block.png"
            _render_page(
                case["pdf_path"],
                case["page_number"],
                block_path,
                scale=3.0,
                clip=block_bbox,
            )
            context, context_block_count = _context_for_blocks(
                case["page_documents"],
                case["context_block_indices"],
            )

        page_path = run_dir / f"case_{question_id}_page.png"
        if question_id in {10, 16, 32} and _copy_m83_full_page(question_id, page_path):
            pass
        else:
            _render_page(case["pdf_path"], case["page_number"], page_path, scale=2.0)

        for input_mode in case["modes"]:
            if input_mode == "block_crop":
                image_path, context_text, used_blocks = block_path, "", 0
            elif input_mode == "block_crop_plus_mineru_text":
                image_path, context_text, used_blocks = (
                    block_path,
                    context,
                    context_block_count,
                )
            else:
                image_path, context_text, used_blocks = page_path, "", 0

            response_text = _call_vision_model(image_path, context_text)
            analysis = _analyze_vision_output(
                response_text,
                case["expected_keywords"],
                case["category"],
            )
            metadata = block.get("metadata", {}) if block else {}
            record = _anonymous_case_record(
                case,
                input_mode,
                analysis,
                block_type=case["block_type"] if input_mode != "full_page" else None,
                block_index=case["block_index"] if input_mode != "full_page" else None,
                block_bbox=metadata.get("bbox") if input_mode != "full_page" else None,
                context_block_count=used_blocks,
            )
            record["m8_3_full_page_evidence_hit"] = (
                case["m83_evidence_hit"] if question_id in {10, 16, 32} else None
            )
            anonymous_records.append(record)
            raw_responses.append(
                {
                    "question_id": question_id,
                    "input_mode": input_mode,
                    "response": response_text,
                }
            )
            _write_json(run_dir / "model_responses.local.json", raw_responses)
            print(
                f"Case {question_id} {input_mode}: evidence="
                f"{analysis['vision_expected_evidence_hit']}, "
                f"fields={analysis['matched_expected_field_count']}/"
                f"{analysis['expected_field_count']}",
                flush=True,
            )

    summary = {
        "experiment": "M8.4 Vision block controlled experiment",
        "model_info": model_info,
        "input_asset_method": "Cached Middle JSON bbox rendered from local PDF; no MinerU ZIP image assets were present.",
        "mineru_was_reparsed": False,
        "anonymous_cases": anonymous_records,
    }
    _write_json(run_dir / "anonymous_summary.json", summary)
    print(f"M8.4 complete; anonymous results saved under outputs/m8_vision_block_test/{run_dir.name}.")
    return summary


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Compare MinerU block crops, MinerU text context, and full-page vision input."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="Git-ignored output directory (must remain under outputs/).",
    )
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    run_experiment(args.output_dir)


if __name__ == "__main__":
    main()
