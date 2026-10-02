import json
import unittest

from evaluation.m8_gated_vision import (
    anonymous_candidate_record,
    build_vision_document,
    gate_decision,
)


BBOX = [0.1, 0.2, 0.7, 0.8]


class GatedVisionTests(unittest.TestCase):
    def test_gate_triggers_for_short_supported_block_with_geometry(self):
        result = gate_decision("table", "x" * 127, BBOX)

        self.assertEqual(result["trigger"], True)
        self.assertEqual(result["reason"], "short_structured_text")
        self.assertEqual(result["text_chars"], 127)

    def test_gate_rejects_long_text_unsupported_types_and_missing_geometry(self):
        self.assertFalse(gate_decision("table", "x" * 128, BBOX)["trigger"])
        self.assertFalse(gate_decision("text", "short", BBOX)["trigger"])
        self.assertFalse(gate_decision("image", "short", None)["trigger"])

    def test_vision_document_keeps_separate_block_provenance(self):
        source = "private-source.pdf"
        text = "locally generated OCR result"
        document = build_vision_document(
            text,
            source=source,
            page=11,
            source_block_type="image",
            block_index=2,
            bbox=BBOX,
        )

        self.assertEqual(document["text"], text)
        self.assertEqual(
            document["metadata"],
            {
                "source": source,
                "page": 11,
                "block_type": "vision_ocr",
                "source_block_type": "image",
                "block_index": 2,
                "bbox": BBOX,
                "generated_by": "qwen3-vl:2b-instruct-q4_K_M",
                "render_scale": 3,
                "fallback": True,
            },
        )

    def test_anonymous_candidate_record_drops_document_text_and_source_name(self):
        candidate = {
            "text": "SECRET INDUSTRIAL TEXT",
            "source": "private-source.pdf",
            "score": 0.8125,
            "reranker_score": 3.25,
            "metadata": {
                "source": "private-source.pdf",
                "page": 11,
                "block_type": "vision_ocr",
                "block_index": 2,
                "bbox": BBOX,
            },
        }

        record = anonymous_candidate_record(candidate, rank=1, source_alias="S004")
        serialized = json.dumps(record, ensure_ascii=False)

        self.assertEqual(record["source_id"], "S004")
        self.assertEqual(record["page"], 11)
        self.assertEqual(record["block_type"], "vision_ocr")
        self.assertEqual(record["rank"], 1)
        self.assertNotIn("SECRET INDUSTRIAL TEXT", serialized)
        self.assertNotIn("private-source.pdf", serialized)


if __name__ == "__main__":
    unittest.main()
