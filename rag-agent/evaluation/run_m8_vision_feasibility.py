"""Run the fixed M8.3 local Ollama vision feasibility experiment."""

import argparse
import base64
import json
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
VISION_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "m8_vision_test"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
MODEL = "qwen3-vl:2b-instruct-q4_K_M"
SELECTED_CASES = {
    10: "OCR_FAILURE",
    16: "IMAGE_INFORMATION_LOSS",
    32: "LAYOUT_RELATION_FAILURE",
}
REQUIRED_SECTIONS = (
    "## 页面摘要",
    "## 文本信息",
    "## 参数信息",
    "## 表格信息",
    "## 图形/布局关系",
    "## 不确定信息",
)
PROMPT = """你是一个工业文档解析专家。

请分析这张 PDF 页面图片，并生成可用于知识库检索的结构化文本。

要求：

1. 提取页面中的所有可读文字。
2. 提取设备名称、型号、编号、参数、单位。
3. 如果存在表格，请恢复表格的行列关系。
4. 如果存在工程图、流程图、示意图，请描述：
   - 图中对象
   - 对象之间的连接关系
   - 标注信息
5. 只输出图片中明确存在的信息。
6. 不确定的信息请标记“不确定”，不要猜测。


输出：

## 页面摘要

## 文本信息

## 参数信息

## 表格信息

## 图形/布局关系

## 不确定信息"""

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_pdf_retrieval import (  # noqa: E402
    _contains_keywords_in_texts,
    _expected_source,
    load_questions,
)
from evaluation.run_m8_visual_representation_experiment import (  # noqa: E402
    _require_matching_mineru_cache,
    require_matching_input_manifest,
)


def _section_content(output, heading):
    lines = output.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == heading:
            content = []
            for candidate in lines[index + 1 :]:
                if candidate.strip().startswith("## "):
                    break
                content.append(candidate)
            return "\n".join(content).strip()
    return ""


def _has_meaningful_section(output, heading):
    content = _section_content(output, heading)
    compact = "".join(content.split()).strip("。．,，；;")
    return bool(compact) and not (
        compact in {"无", "不确定", "未发现", "未识别", "没有"}
        or compact.startswith("未发现")
    )


def analyze_case(question_id, category, mineru_case, expected_keywords, vision_output):
    """Return an anonymous signal summary; never copy model or evidence text."""
    vision_hit = _contains_keywords_in_texts([vision_output], expected_keywords)
    relation_content = _has_meaningful_section(vision_output, "## 图形/布局关系")
    table_content = _has_meaningful_section(vision_output, "## 表格信息")
    text_content = _has_meaningful_section(vision_output, "## 文本信息")
    if category == "OCR_FAILURE":
        relevant_content = table_content
    elif category == "IMAGE_INFORMATION_LOSS":
        relevant_content = text_content or relation_content
    elif category == "LAYOUT_RELATION_FAILURE":
        relevant_content = relation_content
    else:
        relevant_content = False
    candidate_block = vision_hit and relevant_content
    return {
        "question_id": question_id,
        "category": category,
        "mineru_target_info_present": bool(
            mineru_case.get("evidence_in_representation")
        ),
        "current_retrieval_page_hit": mineru_case.get(
            "reranker_top3_page_hit"
        ),
        "current_retrieval_evidence_hit": mineru_case.get(
            "reranker_normalized_evidence_hit"
        ),
        "vision_expected_evidence_hit": vision_hit,
        "vision_added_evidence": (
            vision_hit and not mineru_case.get("evidence_in_representation", False)
        ),
        "vision_required_sections_present": sum(
            heading in vision_output for heading in REQUIRED_SECTIONS
        ),
        "vision_relation_section_nonempty": bool(relation_content),
        "vision_output_characters": len(vision_output),
        "candidate_document_block": bool(candidate_block),
    }


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


def _model_info():
    version = _request_json("/api/version").get("version", "unknown")
    shown = _request_json("/api/show", {"model": MODEL})
    details = shown.get("details", {})
    return {
        "model": MODEL,
        "family": details.get("family"),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
        "ollama_version": version,
    }


def _render_page(pdf_path, page_number, image_path):
    with pymupdf.open(pdf_path) as document:
        if page_number < 1 or page_number > document.page_count:
            raise ValueError("A selected M6 page number is outside its PDF.")
        page = document.load_page(page_number - 1)
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
        pixmap.save(image_path)


def _call_vision_model(image_path):
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "images": [base64.b64encode(image_path.read_bytes()).decode("ascii")],
        "stream": False,
        "keep_alive": "5m",
        "options": {"temperature": 0, "num_predict": 2048},
    }
    result = _request_json("/api/generate", payload, timeout=900)
    output = result.get("response")
    if not isinstance(output, str) or not output.strip():
        raise RuntimeError("Ollama returned an empty vision response.")
    return output


def _write_local_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _load_experiment_inputs():
    questions = load_questions(DATASET_PATH)
    m7_analysis = json.loads(M7_ANALYSIS_PATH.read_text(encoding="utf-8"))
    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    require_matching_input_manifest(
        questions,
        m7_analysis,
        pdf_paths,
        DATASET_PATH,
        M7_ANALYSIS_PATH,
        M8_INPUT_MANIFEST_PATH,
    )
    _require_matching_mineru_cache(PDF_DIR)

    m8_comparison = json.loads(M8_COMPARISON_PATH.read_text(encoding="utf-8"))
    mineru_cases = {
        case["question_id"]: case
        for case in m8_comparison["modes"]["structured"]["parsing_cases"]
    }
    m7_parsing_ids = {
        case["question_id"]
        for case in m7_analysis.get("cases", [])
        if case.get("classification") == "PARSING_FAILURE"
    }
    if not set(SELECTED_CASES).issubset(m7_parsing_ids):
        raise ValueError("Selected M8.3 cases are not in the M7 parsing-failure set.")

    pdf_by_name = {path.name: path for path in pdf_paths}
    prepared = []
    for question_id, category in SELECTED_CASES.items():
        item = questions[question_id - 1]
        mineru_case = mineru_cases.get(question_id)
        if not mineru_case or mineru_case.get("audit_category") != category:
            raise ValueError("A selected M8.3 case does not match its M8.1 category.")
        source = _expected_source(item)
        pdf_path = pdf_by_name.get(source)
        page_number = item.get("expected_page")
        if pdf_path is None or not isinstance(page_number, int):
            raise ValueError("A selected M8.3 case lacks a local PDF page mapping.")
        if not item.get("expected_keywords"):
            raise ValueError("A selected M8.3 case lacks evaluation evidence keys.")
        prepared.append(
            {
                "question_id": question_id,
                "category": category,
                "page_number": page_number,
                "pdf_path": pdf_path,
                "expected_keywords": item["expected_keywords"],
                "mineru_case": mineru_case,
            }
        )
    return prepared, m8_comparison


def run_experiment():
    prepared, _m8_comparison = _load_experiment_inputs()
    model_info = _model_info()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = VISION_OUTPUT_ROOT / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    raw_records = []
    public_summary = {
        "experiment": "M8.3 Vision Model Feasibility Experiment",
        "model_info": model_info,
        "cases": [],
    }

    for case in prepared:
        question_id = case["question_id"]
        image_name = f"case_{question_id}_page.png"
        image_path = run_dir / image_name
        _render_page(case["pdf_path"], case["page_number"], image_path)
        print(f"Rendered anonymous case {question_id}.", flush=True)
        vision_output = _call_vision_model(image_path)
        safe_case = analyze_case(
            question_id,
            case["category"],
            case["mineru_case"],
            case["expected_keywords"],
            vision_output,
        )
        public_summary["cases"].append(safe_case)
        raw_records.append(
            {
                "question_id": question_id,
                "category": case["category"],
                "page_number": case["page_number"],
                "image_file": image_name,
                "response": vision_output,
            }
        )
        _write_local_json(run_dir / "model_responses.local.json", raw_records)
        _write_local_json(run_dir / "anonymous_summary.json", public_summary)
        print(
            f"Case {question_id}: target evidence="
            f"{safe_case['vision_expected_evidence_hit']}; "
            f"candidate block={safe_case['candidate_document_block']}; "
            f"response characters={safe_case['vision_output_characters']}",
            flush=True,
        )

    return run_dir, public_summary


def build_argument_parser():
    return argparse.ArgumentParser(
        description=(
            "Run the fixed three-case M8.3 experiment against the local Ollama model."
        )
    )


def main(argv=None):
    build_argument_parser().parse_args(argv)
    try:
        run_dir, summary = run_experiment()
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(
            f"M8.3 stopped: {type(exc).__name__}; "
            "no local document text was printed.",
            file=sys.stderr,
        )
        return 1
    print(f"M8.3 complete; local outputs are in {run_dir.relative_to(PROJECT_ROOT)}.")
    print(f"Model: {summary['model_info']['model']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
