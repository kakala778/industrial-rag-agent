"""Offline, cohort-limited M8.6 gated focused-VLM retrieval A/B."""

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import run_m8_focused_ocr_comparison as focused
from evaluation import run_m8_vision_feasibility as feasibility
from evaluation.evaluate_pdf_retrieval import (
    RERANK_CANDIDATE_K,
    TOP_K,
    _case_evidence_rank,
    _contains_keywords_in_texts,
    _expected_source,
    _matches_case_scope,
    _normalize_evidence_text,
    _requires_retrieval_scoring,
    evaluate_questions,
    load_questions,
)
from evaluation.m8_gated_vision import (
    MAX_STRUCTURED_TEXT_CHARS,
    RENDER_SCALE,
    VISION_MODEL,
    anonymous_candidate_record,
    build_vision_document,
    gate_decision,
)
from src.mineru_loader import (
    CACHE_ROOT,
    _block_index,
    _cache_key,
    _cached_middle_json,
    _content_strings,
    _documents_from_middle_json,
    _sha256_file,
    _typed_text_for_visual_block,
)
from src.reranker import load_reranker
from src.retrieval import embed_chunks, load_model, split_documents


POSITIVE_IDS = (9, 10, 12, 17, 20)
CONFIRMED_POSITIVE_IDS = (9, 10, 12, 20)
NEGATIVE_CONTROL_IDS = (2, 3, 4, 5, 8, 11, 22, 23, 26, 27)
GT_UNCERTAIN_IDS = frozenset({17})
VISION_TYPE = "vision_ocr"


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_output_directory(path):
    requested = Path(path).expanduser().resolve()
    root = (PROJECT_ROOT / "outputs" / "m8_gated_vision").resolve()
    if requested != root and root not in requested.parents:
        raise ValueError("M8.6 local artifacts must remain under outputs/m8_gated_vision/.")
    requested.mkdir(parents=True, exist_ok=True)
    focused._assert_git_ignored(requested)
    return requested


def _source_aliases(pdf_paths):
    return {path.name: f"S{index:03d}" for index, path in enumerate(sorted(pdf_paths), start=1)}


def _load_verified_cache(pdf_path):
    digest = _sha256_file(pdf_path)
    key, identity = _cache_key(pdf_path, digest)
    cache_entry = CACHE_ROOT / key
    cached_documents = _cached_middle_json(
        cache_entry,
        pdf_path,
        identity,
        key,
        representation="structured",
    )
    if cached_documents is None:
        raise RuntimeError("A matching structured MinerU cache is missing.")
    middle_path = cache_entry / "output" / f"{pdf_path.stem}.json"
    if not middle_path.is_file():
        raise RuntimeError("A matching MinerU Middle JSON file is missing.")
    middle_json = json.loads(middle_path.read_text(encoding="utf-8"))
    return middle_json, cached_documents, digest


def _block_text(block):
    block_type = block.get("type")
    if block_type in {"table", "chart", "image"}:
        return _typed_text_for_visual_block(
            block,
            block_type,
            representation="structured",
        )
    return "\n".join(_content_strings(block.get("content")))


def _case_from_question(question_id, item, pdf_path, middle_json, *, cohort, case_ref):
    page_number = item.get("expected_page")
    if not isinstance(page_number, int):
        raise ValueError("A selected M8.6 case must have a verified 1-based page.")
    page = next(
        (
            page
            for page in middle_json.get("pages", [])
            if page.get("page_idx") == page_number - 1
        ),
        None,
    )
    if page is None:
        raise ValueError("A selected M8.6 page is absent from its verified cache.")

    expected_fields = item.get("expected_keywords", [])
    matches = []
    for position, block in enumerate(page.get("blocks", [])):
        text = _block_text(block)
        if expected_fields and _contains_keywords_in_texts([text], expected_fields):
            matches.append(
                {
                    "source": pdf_path.name,
                    "page": page_number,
                    "block_type": block.get("type") or "unknown",
                    "block_index": _block_index(block, position),
                    "bbox": block.get("bbox"),
                    "structured_text": text,
                    "middle_block": block,
                    "question_id": question_id,
                    "expected_fields": expected_fields,
                    "cohort": cohort,
                    "case_ref": case_ref,
                }
            )
    if not matches:
        raise ValueError("A selected negative control has no single complete-evidence block.")
    matches.sort(key=lambda row: (row["block_index"], row["block_type"]))
    return matches[0]


def _is_matching_m8_5_summary(summary):
    case_ids = tuple(sorted(case.get("question_id") for case in summary.get("cases", [])))
    return (
        summary.get("vision_model", {}).get("tag") == VISION_MODEL
        and summary.get("primary_scale") == "3x"
        and case_ids == tuple(sorted(POSITIVE_IDS))
    )


def _validate_fixed_positive_cases(cases):
    case_ids = tuple(sorted(case.get("question_id") for case in cases))
    if case_ids != tuple(sorted(POSITIVE_IDS)):
        raise ValueError("The fixed M8.5 five-case positive cohort changed.")
    return cases


def _load_m8_5_summary():
    run_dirs = sorted((PROJECT_ROOT / "outputs" / "m8_focused_ocr").glob("run_*"))
    for run_dir in reversed(run_dirs):
        summary_path = run_dir / "anonymous_summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if _is_matching_m8_5_summary(summary):
                return run_dir, summary
    raise RuntimeError("A matching completed M8.5 five-case summary is missing.")


def _prepare_cohorts():
    """Validate the frozen input set, fixed positives, and M7-success controls."""

    positive_prepared = _validate_fixed_positive_cases(focused._load_fixed_cases())
    questions = load_questions(feasibility.DATASET_PATH)
    m7 = json.loads(feasibility.M7_ANALYSIS_PATH.read_text(encoding="utf-8"))
    m7_by_id = {case["question_id"]: case for case in m7.get("cases", [])}
    pdf_paths = sorted(feasibility.PDF_DIR.glob("*.pdf"))
    pdf_by_name = {path.name: path for path in pdf_paths}
    aliases = _source_aliases(pdf_paths)
    middle_by_source = {}
    structured_documents = []
    flat_documents = []
    input_hashes = {}

    expected_sources = {
        source
        for item in questions
        if (source := _expected_source(item)) is not None
    }
    if expected_sources != set(pdf_by_name):
        raise ValueError("The local M6 PDF cohort no longer matches the QA source set.")

    for source in sorted(expected_sources):
        pdf_path = pdf_by_name[source]
        middle_json, cached_documents, digest = _load_verified_cache(pdf_path)
        middle_by_source[source] = middle_json
        input_hashes[aliases[source]] = digest
        structured_documents.extend(cached_documents)
        flat_documents.extend(
            _documents_from_middle_json(middle_json, source, representation="flat")
        )

    m8_5_run, m8_5_summary = _load_m8_5_summary()
    positive_rows = []
    for question_id in POSITIVE_IDS:
        item = questions[question_id - 1]
        prepared = next((row for row in positive_prepared if row["question_id"] == question_id), None)
        if prepared is None:
            raise ValueError("A fixed M8.5 positive case is missing.")
        source = _expected_source(item)
        positive_rows.append(
            {
                "question_id": question_id,
                "cohort": "positive_ocr",
                "case_ref": f"Q{question_id:02d}",
                "source": source,
                "source_id": aliases[source],
                "page": prepared["page_number"],
                "block_type": prepared["block_type"],
                "block_index": prepared["block_index"],
                "bbox": prepared["bbox_normalized"],
                "structured_text": prepared["mineru_structured_text"],
                "expected_fields": item.get("expected_keywords", []),
                "gt_status": "GT_UNCERTAIN" if question_id in GT_UNCERTAIN_IDS else "CONFIRMED",
            }
        )

    negative_rows = []
    for control_number, question_id in enumerate(NEGATIVE_CONTROL_IDS, start=1):
        item = questions[question_id - 1]
        m7_case = m7_by_id.get(question_id)
        if (
            m7_case is None
            or m7_case.get("classification") not in {"NOT_A_BASELINE_FAILURE", "FIXED_BY_RERANKER"}
            or not isinstance(m7_case.get("reranker_evidence_rank"), int)
            or m7_case["reranker_evidence_rank"] > TOP_K
        ):
            raise ValueError("A frozen negative control is not an M7.1 Top-3 success.")
        source = _expected_source(item)
        matched = _case_from_question(
            question_id,
            item,
            pdf_by_name[source],
            middle_by_source[source],
            cohort="negative_control",
            case_ref=f"NC{control_number:02d}",
        )
        matched["source_id"] = aliases[source]
        matched["gt_status"] = "CONFIRMED"
        negative_rows.append(matched)

    if tuple(row["question_id"] for row in positive_rows) != POSITIVE_IDS:
        raise ValueError("The fixed M8.5 positive set changed.")
    if len({(row["source"], row["page"], row["block_index"]) for row in negative_rows}) != len(negative_rows):
        raise ValueError("Negative controls must be unique MinerU blocks.")

    return {
        "questions": questions,
        "m7_cases": m7_by_id,
        "pdf_paths": [pdf_by_name[source] for source in sorted(expected_sources)],
        "source_aliases": aliases,
        "middle_by_source": middle_by_source,
        "structured_documents": structured_documents,
        "flat_documents": flat_documents,
        "input_hashes": input_hashes,
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
        "m8_5_run": m8_5_run,
        "m8_5_summary": m8_5_summary,
    }


def _full_corpus_gate_scan(pdf_paths, middle_by_source, source_aliases):
    eligible = 0
    triggered = 0
    trigger_by_type = Counter()
    eligible_by_type = Counter()
    per_source = Counter()
    for pdf_path in pdf_paths:
        for page in middle_by_source[pdf_path.name].get("pages", []):
            for block in page.get("blocks", []):
                block_type = block.get("type")
                if block_type not in {"table", "image", "chart", "list"}:
                    continue
                decision = gate_decision(block_type, _block_text(block), block.get("bbox"))
                if decision["reason"] == "invalid_bbox":
                    continue
                eligible += 1
                eligible_by_type[block_type] += 1
                if decision["trigger"]:
                    triggered += 1
                    trigger_by_type[block_type] += 1
                    per_source[source_aliases[pdf_path.name]] += 1
    return {
        "eligible_blocks": eligible,
        "triggered_blocks": triggered,
        "triggered_by_type": dict(sorted(trigger_by_type.items())),
        "eligible_by_type": dict(sorted(eligible_by_type.items())),
        "per_source_id_trigger_counts": dict(sorted(per_source.items())),
    }


def _selected_gate_records(cohorts):
    rows = cohorts["positive_rows"] + cohorts["negative_rows"]
    records = []
    for row in rows:
        decision = gate_decision(row["block_type"], row["structured_text"], row["bbox"])
        records.append(
            {
                "case_ref": row["case_ref"],
                "question_id": row["question_id"],

                "cohort": row["cohort"],
                "gt_status": row["gt_status"],
                "source_id": row["source_id"],
                "page": row["page"],
                "block_type": row["block_type"],
                "block_index": row["block_index"],
                "bbox": row["bbox"],
                "text_chars": decision["text_chars"],
                "trigger": decision["trigger"],
                "reason": decision["reason"],
            }
        )
    return records


def _render_triggered_blocks(gate_records, cohorts, run_dir):
    by_ref = {row["case_ref"]: row for row in cohorts["positive_rows"] + cohorts["negative_rows"]}
    crops_dir = run_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    focused._assert_git_ignored(crops_dir)
    pdf_by_name = {path.name: path for path in cohorts["pdf_paths"]}
    rendered = []
    for gate_record in gate_records:
        if not gate_record["trigger"]:
            continue
        case = by_ref[gate_record["case_ref"]]
        image_path = crops_dir / f"{case['case_ref']}_3x.png"
        dimensions = focused._render_crop(
            pdf_by_name[case["source"]],
            case["page"],
            case["bbox"],
            float(RENDER_SCALE),
            image_path,
        )
        rendered.append(
            {
                "case": case,
                "gate_record": gate_record,
                "image_path": image_path,
                "dimensions": dimensions,
            }
        )
    return rendered


def _run_vision_calls(rendered, run_dir):
    model_info = focused._vision_model_info()
    responses = []
    documents = []
    for entry in rendered:
        started = time.perf_counter()
        result = focused._call_vision_model(entry["image_path"])
        wall_seconds = time.perf_counter() - started
        case = entry["case"]
        documents.append(
            build_vision_document(
                result["raw_text"],
                source=case["source"],
                page=case["page"],
                source_block_type=case["block_type"],
                block_index=case["block_index"],
                bbox=case["bbox"],
            )
        )
        responses.append(
            {
                "case_ref": case["case_ref"],
                "question_id": case["question_id"],
                "source_id": case["source_id"],
                "page": case["page"],
                "block_type": case["block_type"],
                "block_index": case["block_index"],
                "bbox": case["bbox"],
                "model": VISION_MODEL,
                "render_scale": RENDER_SCALE,
                "raw_text": result["raw_text"],
                "wall_seconds": round(wall_seconds, 4),
                "ollama_total_seconds": result["ollama_total_seconds"],
                "ollama_load_seconds": result["ollama_load_seconds"],
            }
        )
        _write_json(run_dir / "model_responses.local.json", responses)
        print(f"Vision OCR completed for {case['case_ref']}.", flush=True)
    return model_info, responses, documents


def _strict_evidence_rank(case):
    keywords = case.get("expected_keywords", [])
    if not keywords:
        return None
    texts = []
    for rank, result in enumerate(case.get("diagnostic_results", []), start=1):
        if _matches_case_scope(case, result):
            texts.append(result.get("text", ""))
            combined = "\n".join(texts).casefold()
            if all(keyword.casefold() in combined for keyword in keywords):
                return rank
    return None


def _candidate_record(candidate, rank, item, source_aliases):
    source = candidate.get("source")
    record = anonymous_candidate_record(
        candidate,
        rank=rank,
        source_alias=source_aliases.get(source, "S_UNKNOWN"),
    )
    metadata = candidate.get("metadata") or {}
    text = candidate.get("text", "")
    keywords = item.get("expected_keywords", [])
    normalized_text = _normalize_evidence_text(text)
    raw_text = text.casefold()
    record["in_expected_scope"] = _matches_case_scope(item, candidate)
    record["normalized_evidence_fields_hit"] = sum(
        _normalize_evidence_text(keyword) in normalized_text for keyword in keywords
    )
    record["strict_evidence_fields_hit"] = sum(
        keyword.casefold() in raw_text for keyword in keywords
    )
    record["vision_candidate"] = metadata.get("block_type") == VISION_TYPE
    return record


def _serialize_case(question_id, item, case, source_aliases):
    return {
        "question_id": question_id,
        "metrics": {
            "top1_page_hit": case.get("top1_page_hit"),
            "top3_page_hit": case.get("top3_page_hit"),
            "normalized_evidence_hit": case.get("normalized_keyword_hit"),
            "strict_evidence_hit": case.get("keyword_hit"),
            "normalized_evidence_rank": _case_evidence_rank(case),
            "strict_evidence_rank": _strict_evidence_rank(case),
        },
        "dense_top20": [
            _candidate_record(candidate, rank, item, source_aliases)
            for rank, candidate in enumerate(case.get("dense_candidate_results", []), start=1)
        ],
        "reranked_top20": [
            _candidate_record(candidate, rank, item, source_aliases)
            for rank, candidate in enumerate(case.get("diagnostic_results", []), start=1)
        ],
    }


def _metric_summary(serialized_cases, question_ids):
    chosen = [row for row in serialized_cases if row["question_id"] in question_ids]
    result = {"cases": len(chosen)}
    for key, label in (
        ("top1_page_hit", "top1_page"),
        ("top3_page_hit", "top3_page"),
        ("normalized_evidence_hit", "normalized_evidence"),

        ("strict_evidence_hit", "strict_evidence"),
    ):
        values = [
            row["metrics"][key]
            for row in chosen
            if row["metrics"][key] is not None
        ]
        result[label] = {
            "hits": sum(value is True for value in values),
            "total": len(values),
        }
    return result


def _candidate_rank_for_block(serialized_case, case):
    def find_rank(name):
        for record in serialized_case[name]:
            if (
                record["vision_candidate"]
                and record["source_id"] == case["source_id"]
                and record["page"] == case["page"]
                and record["block_index"] == case["block_index"]
            ):
                return record["rank"], record
        return None, None

    dense_rank, dense_record = find_rank("dense_top20")
    reranker_rank, reranker_record = find_rank("reranked_top20")
    return dense_rank, reranker_rank, dense_record or reranker_record


def _rank_case_summary(case, baseline_serialized, gated_serialized):
    base_metrics = baseline_serialized["metrics"]
    gate_metrics = gated_serialized["metrics"]
    dense_rank, rerank_rank, candidate = _candidate_rank_for_block(gated_serialized, case)
    return {
        "case_ref": case["case_ref"],
        "question_id": case["question_id"],
        "gt_status": case["gt_status"],
        "baseline_normalized_evidence_rank": base_metrics["normalized_evidence_rank"],
        "baseline_strict_evidence_rank": base_metrics["strict_evidence_rank"],
        "vision_candidate_dense_rank": dense_rank,
        "vision_candidate_reranker_rank": rerank_rank,
        "vision_candidate_normalized_fields_hit": (
            candidate.get("normalized_evidence_fields_hit") if candidate else None
        ),
        "vision_candidate_strict_fields_hit": (
            candidate.get("strict_evidence_fields_hit") if candidate else None
        ),
        "gated_normalized_evidence_rank": gate_metrics["normalized_evidence_rank"],
        "gated_strict_evidence_rank": gate_metrics["strict_evidence_rank"],
        "baseline_normalized_evidence_hit": base_metrics["normalized_evidence_hit"],
        "gated_normalized_evidence_hit": gate_metrics["normalized_evidence_hit"],
        "baseline_strict_evidence_hit": base_metrics["strict_evidence_hit"],
        "gated_strict_evidence_hit": gate_metrics["strict_evidence_hit"],
    }


def _review_template(rendered):
    return {
        "note": "Local review only. Compare each candidate against its crop; do not copy document text here.",
        "cases": [
            {
                "case_ref": entry["case"]["case_ref"],
                "question_id": entry["case"]["question_id"],
                "source_id": entry["case"]["source_id"],
                "page": entry["case"]["page"],
                "block_type": entry["case"]["block_type"],
                "block_index": entry["case"]["block_index"],
                "review_status": "UNREVIEWED",
                "digits": "UNREVIEWED",
                "codes_and_models": "UNREVIEWED",
                "table_row_relations": "UNREVIEWED",
                "notes": "",
            }
            for entry in rendered
        ],
    }


def run_experiment(output_dir=None):
    cohorts = _prepare_cohorts()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = _safe_output_directory(
        output_dir or (PROJECT_ROOT / "outputs" / "m8_gated_vision")
    )
    run_dir = output_root / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    focused._assert_git_ignored(run_dir)

    print("Loading only hash-verified cached MinerU documents.", flush=True)
    baseline_chunks = split_documents(cohorts["structured_documents"])
    embedding_model = load_model()
    baseline_embeddings = embed_chunks(baseline_chunks, embedding_model)
    reranker_model = load_reranker()
    common = {
        "top_k": TOP_K,
        "documents": cohorts["structured_documents"],
        "representation_reference_documents": cohorts["flat_documents"],
        "mode": "reranker",
        "reranker_model": reranker_model,
    }
    baseline_report = evaluate_questions(
        cohorts["questions"],
        embedding_model,
        baseline_chunks,
        baseline_embeddings,
        **common,
    )

    baseline_cases = baseline_report["cases"]
    baseline_by_id = {index: case for index, case in enumerate(baseline_cases, start=1)}
    for case in cohorts["negative_rows"]:
        baseline = baseline_by_id[case["question_id"]]
        if baseline.get("top3_page_hit") is not True or baseline.get("normalized_keyword_hit") is not True:
            raise RuntimeError("A frozen negative control did not reproduce its M7.1 success.")
    for question_id in CONFIRMED_POSITIVE_IDS:
        baseline = baseline_by_id[question_id]
        if baseline.get("normalized_keyword_hit") is True:
            raise RuntimeError("A fixed positive case no longer reproduces its MinerU evidence failure.")

    full_gate_scan = _full_corpus_gate_scan(
        cohorts["pdf_paths"],
        cohorts["middle_by_source"],
        cohorts["source_aliases"],
    )
    gate_records = _selected_gate_records(cohorts)
    positive_records = [row for row in gate_records if row["cohort"] == "positive_ocr"]
    confirmed_positive_records = [
        row for row in positive_records if row["gt_status"] == "CONFIRMED"
    ]
    uncertain_records = [row for row in positive_records if row["gt_status"] == "GT_UNCERTAIN"]
    negative_records = [row for row in gate_records if row["cohort"] == "negative_control"]
    triggered_positive = sum(row["trigger"] for row in confirmed_positive_records)
    triggered_uncertain = sum(row["trigger"] for row in uncertain_records)

    false_activations = sum(row["trigger"] for row in negative_records)
    if len(confirmed_positive_records) != len(CONFIRMED_POSITIVE_IDS):
        raise RuntimeError("The confirmed M8.5 positive denominator changed.")
    if len(negative_records) != len(NEGATIVE_CONTROL_IDS):
        raise RuntimeError("The fixed negative-control denominator changed.")

    _write_json(run_dir / "gate_decisions.local.json", gate_records)
    _write_json(
        run_dir / "cohort_manifest.local.json",
        {
            "positive_question_ids": list(POSITIVE_IDS),
            "negative_question_ids": list(NEGATIVE_CONTROL_IDS),
            "gt_uncertain_question_ids": sorted(GT_UNCERTAIN_IDS),
            "source_sha256_by_alias": cohorts["input_hashes"],
            "m8_5_run_id": cohorts["m8_5_run"].name,
            "gate_full_corpus_counts": full_gate_scan,
        },
    )

    rendered = _render_triggered_blocks(gate_records, cohorts, run_dir)
    model_info, responses, vision_documents = _run_vision_calls(rendered, run_dir)
    _write_json(run_dir / "human_review.local.json", _review_template(rendered))

    vision_chunks = split_documents(vision_documents)
    vision_embeddings = embed_chunks(vision_chunks, embedding_model)
    if len(vision_chunks):
        gated_chunks = baseline_chunks + vision_chunks
        gated_embeddings = np.concatenate([baseline_embeddings, vision_embeddings], axis=0)
    else:
        gated_chunks = baseline_chunks
        gated_embeddings = baseline_embeddings
    gated_documents = cohorts["structured_documents"] + vision_documents
    gated_common = dict(common)
    gated_common["documents"] = gated_documents
    gated_report = evaluate_questions(
        cohorts["questions"],
        embedding_model,
        gated_chunks,
        gated_embeddings,
        **gated_common,
    )

    baseline_serialized = []
    gated_serialized = []
    case_records = []
    cohort_by_id = {
        case["question_id"]: case
        for case in cohorts["positive_rows"] + cohorts["negative_rows"]
    }
    gate_by_id = {record["question_id"]: record for record in gate_records}
    m7_success_ids = {2, 3, 4, 5, 8, 11, 22, 23, 26, 27, 28}
    for question_id, (item, baseline, gated) in enumerate(
        zip(cohorts["questions"], baseline_cases, gated_report["cases"]),
        start=1,
    ):
        base_row = _serialize_case(question_id, item, baseline, cohorts["source_aliases"])
        gated_row = _serialize_case(question_id, item, gated, cohorts["source_aliases"])
        case_info = cohort_by_id.get(question_id)
        label = case_info["cohort"] if case_info else ("M7_SUCCESS" if question_id in m7_success_ids else "M6")
        gate_record = gate_by_id.get(question_id)
        gated_row["gate_triggered"] = gate_record["trigger"] if gate_record else None
        gated_row["gate_reason"] = gate_record["reason"] if gate_record else None
        case_records.append(
            {
                "question_id": question_id,
                "cohort": label,
                "formal_metric_included": question_id not in GT_UNCERTAIN_IDS,
                "baseline": base_row,
                "gated": gated_row,
            }
        )
        baseline_serialized.append(base_row)
        gated_serialized.append(gated_row)

    formal_ids = [
        question_id
        for question_id, item in enumerate(cohorts["questions"], start=1)
        if _requires_retrieval_scoring(item) and question_id not in GT_UNCERTAIN_IDS
    ]
    positive_ids = list(CONFIRMED_POSITIVE_IDS)
    control_ids = list(NEGATIVE_CONTROL_IDS)
    original_success_ids = sorted(
        question_id
        for question_id, m7_case in cohorts["m7_cases"].items()
        if m7_case.get("analysis_scope") == "BASELINE_SUCCESS"
        or m7_case.get("classification") == "FIXED_BY_RERANKER"
    )
    baseline_summary = _metric_summary(baseline_serialized, formal_ids)
    gated_summary = _metric_summary(gated_serialized, formal_ids)
    positive_baseline = _metric_summary(baseline_serialized, positive_ids)
    positive_gated = _metric_summary(gated_serialized, positive_ids)
    control_baseline = _metric_summary(baseline_serialized, control_ids)
    control_gated = _metric_summary(gated_serialized, control_ids)

    regressions = []
    for question_id in original_success_ids:
        before = baseline_serialized[question_id - 1]["metrics"]
        after = gated_serialized[question_id - 1]["metrics"]
        before_success = before["top3_page_hit"] is True and before["normalized_evidence_hit"] is True
        after_success = after["top3_page_hit"] is True and after["normalized_evidence_hit"] is True
        if before_success and not after_success:
            regressions.append(question_id)

    positive_case_summaries = []
    for case in cohorts["positive_rows"]:
        question_id = case["question_id"]
        record = _rank_case_summary(
            case,
            baseline_serialized[question_id - 1],
            gated_serialized[question_id - 1],
        )
        record["gate_triggered"] = gate_by_id[question_id]["trigger"]
        record["gate_reason"] = gate_by_id[question_id]["reason"]
        positive_case_summaries.append(record)

    average_inference = (
        round(sum(response["wall_seconds"] for response in responses) / len(responses), 4)
        if responses
        else None
    )
    calls_per_pdf = round(len(responses) / len(cohorts["pdf_paths"]), 4)
    full_corpus_average = round(full_gate_scan["triggered_blocks"] / len(cohorts["pdf_paths"]), 4)
    summary = {
        "experiment": "M8.6 Gated Focused-VLM Retrieval A/B",
        "scope": "cohort-limited VLM calls; full-corpus gate scan was metadata-only",
        "mineru_reparsed": False,
        "m8_5_run_id": cohorts["m8_5_run"].name,

        "model": model_info,
        "gate": {
            "eligible_block_types": ["chart", "image", "list", "table"],
            "rule": f"valid normalized bbox and structured text length < {MAX_STRUCTURED_TEXT_CHARS} characters",
            "positive_confirmed": {
                "triggered": triggered_positive,
                "total": len(confirmed_positive_records),
                "recall": triggered_positive / len(confirmed_positive_records),
            },
            "positive_gt_uncertain": {
                "triggered": triggered_uncertain,
                "total": len(uncertain_records),
            },
            "negative_controls": {
                "false_activations": false_activations,
                "total": len(negative_records),
                "false_activation_rate": false_activations / len(negative_records),
            },
            "full_corpus_metadata_scan": full_gate_scan,
        },
        "retrieval": {
            "metric_definitions": {
                "normalized_evidence": "existing M7 evaluator normalized evidence matcher",
                "strict_evidence": "existing M7 evaluator raw/case-folded evidence matcher",
                "retrieval": f"dense Top-{RERANK_CANDIDATE_K} + existing BGE reranker Top-{TOP_K}",
                "chunking": {"max_chars": 500, "overlap": 80},
            },
            "formal_question_ids": formal_ids,
            "baseline": baseline_summary,
            "gated": gated_summary,
            "positive_ocr_baseline": positive_baseline,
            "positive_ocr_gated": positive_gated,
            "negative_control_baseline": control_baseline,
            "negative_control_gated": control_gated,
            "m7_success_question_ids": original_success_ids,
            "baseline_success_regressions": regressions,
            "positive_case_ranks": positive_case_summaries,
        },
        "cost": {
            "cohort_gate_triggers": sum(row["trigger"] for row in gate_records),
            "cohort_vlm_calls": len(responses),
            "average_inference_seconds": average_inference,
            "cohort_calls_per_pdf": calls_per_pdf,
            "full_corpus_estimated_calls_per_pdf_if_all_triggers_inferred": full_corpus_average,
        },
        "negative_control_ids": [row["case_ref"] for row in cohorts["negative_rows"]],
        "gate_records": gate_records,
        "case_records": case_records,
    }
    _write_json(run_dir / "anonymous_summary.json", summary)
    _write_json(run_dir / "human_review.local.json", _review_template(rendered))
    print(f"M8.6 local experiment artifacts: {run_dir}")
    print(json.dumps(
        {
            "positive_gate_recall": summary["gate"]["positive_confirmed"],
            "negative_false_activation": summary["gate"]["negative_controls"],
            "baseline": baseline_summary,
            "gated": gated_summary,
            "positive_ocr_baseline": positive_baseline,
            "positive_ocr_gated": positive_gated,
            "regressions": regressions,
            "vlm_calls": len(responses),
            "average_inference_seconds": average_inference,
            "estimated_full_corpus_calls_per_pdf": full_corpus_average,
        },
        ensure_ascii=False,
        indent=2,
    ))
    return run_dir, summary


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Run the offline M8.6 gated focused-VLM retrieval A/B."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Ignored local output directory under outputs/m8_gated_vision/.",
    )
    return parser


def main():
    args = build_argument_parser().parse_args()
    try:
        run_experiment(output_dir=args.output_dir)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"M8.6 experiment error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
