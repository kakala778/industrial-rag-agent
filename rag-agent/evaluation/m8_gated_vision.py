"""Pure helpers for the offline M8.6 gated focused-VLM experiment."""

import math


ELIGIBLE_BLOCK_TYPES = frozenset({"table", "image", "chart", "list"})
MAX_STRUCTURED_TEXT_CHARS = 128
VISION_MODEL = "qwen3-vl:2b-instruct-q4_K_M"
RENDER_SCALE = 3


def _valid_bbox(bbox):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox):
        return False
    x0, y0, x1, y1 = bbox
    return (
        all(math.isfinite(float(value)) for value in bbox)
        and 0 <= x0 < x1 <= 1
        and 0 <= y0 < y1 <= 1
    )


def gate_decision(block_type, structured_text, bbox):
    """Return a deterministic, QA-blind decision from block metadata and text."""
    text = structured_text if isinstance(structured_text, str) else ""
    text_chars = len(text.strip())
    if block_type not in ELIGIBLE_BLOCK_TYPES:
        return {
            "trigger": False,
            "reason": "unsupported_block_type",
            "text_chars": text_chars,
        }
    if not _valid_bbox(bbox):
        return {
            "trigger": False,
            "reason": "invalid_bbox",
            "text_chars": text_chars,
        }
    if text_chars < MAX_STRUCTURED_TEXT_CHARS:
        return {
            "trigger": True,
            "reason": "short_structured_text",
            "text_chars": text_chars,
        }
    return {
        "trigger": False,
        "reason": "structured_text_not_short",
        "text_chars": text_chars,
    }


def build_vision_document(
    text,
    *,
    source,
    page,
    source_block_type,
    block_index,
    bbox,
):
    """Create a separate OCR candidate with traceable source-block provenance."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Vision candidate text must be a non-empty string.")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Vision candidate source must be a non-empty string.")
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        raise ValueError("Vision candidate page must be a positive 1-based integer.")
    if source_block_type not in ELIGIBLE_BLOCK_TYPES:
        raise ValueError("Vision candidate source block type is not eligible.")
    if isinstance(block_index, bool) or not isinstance(block_index, int) or block_index < 0:
        raise ValueError("Vision candidate block index must be a non-negative integer.")
    if not _valid_bbox(bbox):
        raise ValueError("Vision candidate bbox must be normalized and valid.")

    return {
        "text": text,
        "metadata": {
            "source": source,
            "page": page,
            "block_type": "vision_ocr",
            "source_block_type": source_block_type,
            "block_index": block_index,
            "bbox": [float(value) for value in bbox],
            "generated_by": VISION_MODEL,
            "render_scale": RENDER_SCALE,
            "fallback": True,
        },
    }


def anonymous_candidate_record(candidate, *, rank, source_alias):
    """Allowlist rank/provenance fields; never serialize candidate text."""
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("Candidate rank must be a positive integer.")
    if not isinstance(source_alias, str) or not source_alias:
        raise ValueError("A non-empty anonymous source alias is required.")

    metadata = candidate.get("metadata") or {}
    record = {
        "rank": rank,
        "source_id": source_alias,
        "page": metadata.get("page"),
        "block_type": metadata.get("block_type"),
        "block_index": metadata.get("block_index"),
    }
    bbox = metadata.get("bbox")
    if _valid_bbox(bbox):
        record["bbox"] = [float(value) for value in bbox]

    for key in ("score", "reranker_score"):
        score = candidate.get(key)
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            score = float(score)
            if math.isfinite(score):
                record[key] = score
    return record
