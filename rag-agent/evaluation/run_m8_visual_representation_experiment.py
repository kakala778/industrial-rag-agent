"""Run the fixed M6 QA set against three MinerU representations."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DATA_ROOT = PROJECT_ROOT.parent / "minerU" / "industrial-rag-data"
DEFAULT_PDF_DIR = LOCAL_DATA_ROOT / "input"
DEFAULT_DATASET = LOCAL_DATA_ROOT / "qa" / "m6_final_qa.local.json"
DEFAULT_M7_ANALYSIS = PROJECT_ROOT / "outputs" / "m7_failure_analysis.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "m8_visual_representation_comparison.json"
DEFAULT_INPUT_MANIFEST = PROJECT_ROOT / "outputs" / "m8_m6_input_manifest.json"
REPRESENTATION_MODES = ("structured", "structured_ocr", "structured_full")
M6_EXPECTED_QUESTION_ROWS = 35
M6_EXPECTED_SCORED_QUESTIONS = 32
M6_EXPECTED_PDF_COUNT = 8
M8_AUDIT_CATEGORIES = {
    9: "OCR_FAILURE",
    10: "OCR_FAILURE",
    12: "OCR_FAILURE",
    14: "OTHER/UNCERTAIN",
    16: "IMAGE_INFORMATION_LOSS",
    17: "OCR_FAILURE",
    20: "OCR_FAILURE",
    29: "IMAGE_INFORMATION_LOSS",
    30: "OTHER/UNCERTAIN",
    31: "OTHER/UNCERTAIN",
    32: "LAYOUT_RELATION_FAILURE",
}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_pdf_retrieval import (  # noqa: E402
    _contains_keywords_in_texts,
    _expected_source,
    _matches_case_scope,
    _requires_retrieval_scoring,
    evaluate_comparison,
    load_pdf_documents,
    load_questions,
)
from src.mineru_loader import (  # noqa: E402
    CACHE_ROOT,
    _cache_key,
    _cached_middle_json,
    _sha256_file,
)
from src.retrieval import embed_chunks, load_model, split_documents  # noqa: E402
from src.reranker import DEFAULT_RERANKER_MODEL, load_reranker  # noqa: E402


def _require_matching_mineru_cache(pdf_dir):
    """Fail closed unless every local PDF has a hash-matched cache entry."""
    pdf_paths = sorted(Path(pdf_dir).glob("*.pdf"))
    if not pdf_paths:
        raise RuntimeError("No local PDFs were found; M8.2 was not run.")

    missing = 0
    for pdf_path in pdf_paths:
        digest = _sha256_file(pdf_path)
        key, identity = _cache_key(pdf_path, digest)
        cached = _cached_middle_json(
            CACHE_ROOT / key,
            pdf_path,
            identity,
            key,
            representation="structured",
        )
        if cached is None:
            missing += 1
    if missing:
        raise RuntimeError(
            f"{missing} local PDF(s) lack a matching MinerU cache; "
            "the experiment stopped without running MinerU."
        )
    return len(pdf_paths)


def _metric(report, hits_key, total_key):
    return {
        "hits": report[hits_key],
        "total": report[total_key],
    }


def _evidence_in_documents(item, documents):
    keywords = item.get("expected_keywords", [])
    if not keywords:
        return None
    scoped = [document for document in documents if _matches_case_scope(item, document)]
    return _contains_keywords_in_texts(
        [document.get("text", "") for document in scoped], keywords
    )


def _metadata_coverage(documents, chunks):
    visual = [
        document
        for document in documents
        if (document.get("metadata") or {}).get("block_type")
        in {"image", "chart", "table"}
    ]
    images = [
        document
        for document in visual
        if (document.get("metadata") or {}).get("block_type") == "image"
    ]
    text_groups = [
        document
        for document in documents
        if (document.get("metadata") or {}).get("block_type") == "text_group"
    ]
    return {
        "visual_documents": len(visual),
        "visual_documents_with_bbox": sum(
            isinstance((document.get("metadata") or {}).get("bbox"), list)
            for document in visual
        ),
        "image_documents": len(images),
        "image_documents_with_index": sum(
            "image_index" in (document.get("metadata") or {})
            for document in images
        ),
        "image_documents_with_caption": sum(
            bool((document.get("metadata") or {}).get("caption"))
            for document in images
        ),
        "text_group_documents_with_block_geometry": sum(
            bool((document.get("metadata") or {}).get("block_geometry"))
            for document in text_groups
        ),
        "chunks_with_bbox": sum(
            isinstance((chunk.get("metadata") or {}).get("bbox"), list)
            for chunk in chunks
        ),
    }


def _representation_metrics(reports):
    return {
        "dense": {
            "top1_page_hit": _metric(
                reports["dense"], "top1_page_hits", "top1_page_total"
            ),
            "top3_page_hit": _metric(
                reports["dense"], "top3_page_hits", "top3_page_total"
            ),
            "normalized_evidence_hit": _metric(
                reports["dense"],
                "normalized_keyword_hits",
                "normalized_keyword_total",
            ),
        },
        "dense_plus_reranker": {
            "top1_page_hit": _metric(
                reports["reranker"], "top1_page_hits", "top1_page_total"
            ),
            "top3_page_hit": _metric(
                reports["reranker"], "top3_page_hits", "top3_page_total"
            ),
            "normalized_evidence_hit": _metric(
                reports["reranker"],
                "normalized_keyword_hits",
                "normalized_keyword_total",
            ),
        },
    }


def _original_parsing_ids(failure_analysis, question_count):
    ids = sorted(
        case["question_id"]
        for case in failure_analysis.get("cases", [])
        if case.get("classification") == "PARSING_FAILURE"
    )
    if ids != sorted(M8_AUDIT_CATEGORIES):
        raise ValueError(
            "The local M7 analysis does not match the 11 cases audited in M8.1."
        )
    if any(
        not isinstance(question_id, int) or not 1 <= question_id <= question_count
        for question_id in ids
    ):
        raise ValueError("M7 question IDs do not map to the local M6 QA rows.")
    return ids


def build_local_input_manifest(
    questions, failure_analysis, pdf_paths, dataset_path, m7_analysis_path
):
    """Fingerprint the fixed private M6 inputs without storing their contents."""
    pdf_paths = sorted(Path(path) for path in pdf_paths)
    if len(questions) != M6_EXPECTED_QUESTION_ROWS:
        raise ValueError(
            f"The fixed M6 cohort must contain {M6_EXPECTED_QUESTION_ROWS} QA rows."
        )
    scored_count = sum(_requires_retrieval_scoring(item) for item in questions)
    if scored_count != M6_EXPECTED_SCORED_QUESTIONS:
        raise ValueError(
            "The fixed M6 cohort does not have the expected number of scored questions."
        )
    if len(pdf_paths) != M6_EXPECTED_PDF_COUNT:
        raise ValueError(
            f"The fixed M6 cohort must contain {M6_EXPECTED_PDF_COUNT} PDFs."
        )

    expected_sources = {
        source
        for item in questions
        if (source := _expected_source(item)) is not None
    }
    unscoped_scored_count = sum(
        _expected_source(item) is None and _requires_retrieval_scoring(item)
        for item in questions
    )
    if unscoped_scored_count:
        raise ValueError("Every scored fixed M6 QA row must identify its source.")
    pdf_names = {path.name for path in pdf_paths}
    if expected_sources != pdf_names:
        raise ValueError("The fixed M6 QA and PDF source cohorts do not match.")

    parsing_ids = _original_parsing_ids(failure_analysis, len(questions))
    pdf_hashes = sorted(_sha256_file(path) for path in pdf_paths)
    pdf_cohort_hash = hashlib.sha256("\n".join(pdf_hashes).encode("ascii")).hexdigest()
    return {
        "schema_version": 1,
        "question_rows": len(questions),
        "scored_questions": scored_count,
        "pdf_count": len(pdf_paths),
        "m7_parsing_failure_count": len(parsing_ids),
        "dataset_sha256": _sha256_file(dataset_path),
        "m7_analysis_sha256": _sha256_file(m7_analysis_path),
        "pdf_cohort_sha256": pdf_cohort_hash,
    }


def require_matching_input_manifest(
    questions,
    failure_analysis,
    pdf_paths,
    dataset_path,
    m7_analysis_path,
    manifest_path,
):
    """Require the ignored local fingerprint from the completed M6 run."""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise RuntimeError(
            "The ignored M8.2 M6 input manifest is missing; refusing to run an "
            "unverified cohort."
        )
    expected = build_local_input_manifest(
        questions, failure_analysis, pdf_paths, dataset_path, m7_analysis_path
    )
    observed = json.loads(manifest_path.read_text(encoding="utf-8"))
    if observed != expected:
        raise ValueError("The local M6 inputs do not match the pinned M8.2 manifest.")
    return expected["pdf_count"]


def _write_input_manifest(path, manifest):
    output = Path(path).resolve()
    ignored_root = (PROJECT_ROOT / "outputs").resolve()
    try:
        output.relative_to(ignored_root)
    except ValueError as exc:
        raise ValueError(
            "The M8.2 input manifest must stay inside the Git-ignored outputs directory"
        ) from exc
    if output.exists():
        raise ValueError("The local M8.2 input manifest already exists; refusing overwrite.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def build_anonymous_summary(
    questions,
    reports_by_mode,
    documents_by_mode,
    failure_analysis,
    chunks_by_mode=None,
):
    """Return aggregate metrics and anonymous case flags without source content."""
    if set(reports_by_mode) != set(REPRESENTATION_MODES):
        raise ValueError("reports_by_mode must contain all three M8.2 modes")
    if set(documents_by_mode) != set(REPRESENTATION_MODES):
        raise ValueError("documents_by_mode must contain all three M8.2 modes")
    chunks_by_mode = chunks_by_mode or {mode: [] for mode in REPRESENTATION_MODES}
    original_ids = _original_parsing_ids(failure_analysis, len(questions))

    baseline_presence = {
        question_id: _evidence_in_documents(
            questions[question_id - 1], documents_by_mode["structured"]
        )
        for question_id in original_ids
    }
    mode_summaries = {}
    for mode in REPRESENTATION_MODES:
        reports = reports_by_mode[mode]
        dense_cases = reports["dense"]["cases"]
        reranker_cases = reports["reranker"]["cases"]
        if len(dense_cases) != len(questions) or len(reranker_cases) != len(questions):
            raise ValueError("Evaluation reports must preserve the M6 QA row order")

        case_summaries = []
        for question_id in original_ids:
            item = questions[question_id - 1]
            dense_case = dense_cases[question_id - 1]
            reranker_case = reranker_cases[question_id - 1]
            evidence_present = _evidence_in_documents(
                item, documents_by_mode[mode]
            )
            if evidence_present is None:
                raise ValueError("An audited M7 case lacks expected evidence keywords")
            retrieved_top3 = reranker_case.get("diagnostic_results", [])[:3]
            case_summaries.append(
                {
                    "question_id": question_id,
                    "audit_category": M8_AUDIT_CATEGORIES[question_id],
                    "evidence_in_representation": evidence_present,
                    "newly_available_vs_structured": (
                        baseline_presence[question_id] is False and evidence_present
                    ),
                    "dense_top1_page_hit": dense_case.get("top1_page_hit"),
                    "dense_top3_page_hit": dense_case.get("top3_page_hit"),
                    "dense_normalized_evidence_hit": dense_case.get(
                        "normalized_keyword_hit"
                    ),
                    "reranker_top1_page_hit": reranker_case.get("top1_page_hit"),
                    "reranker_top3_page_hit": reranker_case.get("top3_page_hit"),
                    "reranker_normalized_evidence_hit": reranker_case.get(
                        "normalized_keyword_hit"
                    ),
                    "reranker_error_layer": reranker_case.get("error_layer"),
                    "reranker_top3_visual_blocks_with_bbox": sum(
                        result.get("metadata", {}).get("block_type")
                        in {"image", "chart", "table"}
                        and isinstance(result.get("metadata", {}).get("bbox"), list)
                        for result in retrieved_top3
                    ),
                }
            )

        mode_summaries[mode] = {
            "metrics": _representation_metrics(reports),
            "parsing_failures_remaining": sum(
                not case["evidence_in_representation"] for case in case_summaries
            ),
            "newly_available_evidence_cases": [
                case["question_id"]
                for case in case_summaries
                if case["newly_available_vs_structured"]
            ],
            "metadata_coverage": _metadata_coverage(
                documents_by_mode[mode], chunks_by_mode[mode]
            ),
            "parsing_cases": case_summaries,
        }

    scored_count = sum(_requires_retrieval_scoring(item) for item in questions)
    return {
        "experiment": "M8.2 Visual Information Representation Experiment",
        "parser": "mineru",
        "question_rows": len(questions),
        "scored_questions": scored_count,
        "original_m7_parsing_failure_count": len(original_ids),
        "modes": mode_summaries,
    }


def _write_summary(path, summary):
    output = Path(path).resolve()
    ignored_root = (PROJECT_ROOT / "outputs").resolve()
    try:
        output.relative_to(ignored_root)
    except ValueError as exc:
        raise ValueError(
            "M8.2 output must stay inside the Git-ignored outputs directory"
        ) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Compare fixed M6 metrics across three MinerU representations."
    )
    parser.add_argument(
        "--initialize-input-manifest",
        action="store_true",
        help="Pin hashes and counts for the fixed local M6 cohort, without evaluating it.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    try:
        output = args.output.resolve()
        output.relative_to((PROJECT_ROOT / "outputs").resolve())
        questions = load_questions(DEFAULT_DATASET)
        failure_analysis = json.loads(
            DEFAULT_M7_ANALYSIS.read_text(encoding="utf-8")
        )
        _original_parsing_ids(failure_analysis, len(questions))
        pdf_paths = sorted(DEFAULT_PDF_DIR.glob("*.pdf"))
        if args.initialize_input_manifest:
            if DEFAULT_INPUT_MANIFEST.exists():
                raise ValueError(
                    "The local M8.2 input manifest already exists; refusing overwrite."
                )
            manifest = build_local_input_manifest(
                questions,
                failure_analysis,
                pdf_paths,
                DEFAULT_DATASET,
                DEFAULT_M7_ANALYSIS,
            )
            cached_pdf_count = _require_matching_mineru_cache(DEFAULT_PDF_DIR)
            _write_input_manifest(DEFAULT_INPUT_MANIFEST, manifest)
            print(
                "Local M8.2 input manifest initialized for "
                f"{manifest['question_rows']} QA rows, "
                f"{manifest['scored_questions']} scored questions, and "
                f"{cached_pdf_count} hash-matched PDF caches."
            )
            return 0

        pinned_pdf_count = require_matching_input_manifest(
            questions,
            failure_analysis,
            pdf_paths,
            DEFAULT_DATASET,
            DEFAULT_M7_ANALYSIS,
            DEFAULT_INPUT_MANIFEST,
        )
        cached_pdf_count = _require_matching_mineru_cache(DEFAULT_PDF_DIR)
        if cached_pdf_count != pinned_pdf_count:
            raise ValueError("The pinned M6 PDF count differs from the cache count.")

        model = load_model()
        reranker_model = load_reranker(args.reranker_model)
        reports_by_mode = {}
        documents_by_mode = {}
        chunks_by_mode = {}

        for mode in REPRESENTATION_MODES:
            documents = load_pdf_documents(
                DEFAULT_PDF_DIR,
                questions,
                parser="mineru",
                force=False,
                representation=mode,
            )
            chunks = split_documents(documents)
            embeddings = embed_chunks(chunks, model) if chunks else []
            reports = evaluate_comparison(
                questions,
                model,
                chunks,
                embeddings,
                reranker_model,
                documents=documents,
            )
            reports_by_mode[mode] = reports
            documents_by_mode[mode] = documents
            chunks_by_mode[mode] = chunks

        summary = build_anonymous_summary(
            questions,
            reports_by_mode,
            documents_by_mode,
            failure_analysis,
            chunks_by_mode=chunks_by_mode,
        )
        summary["matching_local_pdf_caches"] = cached_pdf_count
        _write_summary(args.output, summary)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(
            f"M8.2 stopped: {type(exc).__name__}; "
            "no local source text was printed.",
            file=sys.stderr,
        )
        return 1

    print("M8.2 comparison complete.")
    print(f"Scored questions: {summary['scored_questions']}")
    print(f"Matching local PDF caches: {summary['matching_local_pdf_caches']}")
    for mode, report in summary["modes"].items():
        reranker = report["metrics"]["dense_plus_reranker"]
        print(
            f"{mode}: Top-1 page {reranker['top1_page_hit']['hits']}/"
            f"{reranker['top1_page_hit']['total']}; Top-3 page "
            f"{reranker['top3_page_hit']['hits']}/{reranker['top3_page_hit']['total']}; "
            f"normalized evidence {reranker['normalized_evidence_hit']['hits']}/"
            f"{reranker['normalized_evidence_hit']['total']}; remaining parsing "
            f"{report['parsing_failures_remaining']}/"
            f"{summary['original_m7_parsing_failure_count']}"
        )
    print(f"Anonymous summary: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
