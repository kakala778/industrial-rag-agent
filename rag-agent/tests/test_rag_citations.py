import inspect
import unittest

from src.rag_demo import format_sources


class RagCitationTests(unittest.TestCase):
    def test_structured_blocks_with_reused_chunk_ids_keep_distinct_citations(self):
        results = [
            {
                "score": 0.91,
                "source": "manual.pdf",
                "chunk_id": 0,
                "text": "Pressure limits",
                "metadata": {
                    "source": "manual.pdf",
                    "page": 4,
                    "block_type": "text_group",
                    "block_index": 2,
                },
            },
            {
                "score": 0.88,
                "source": "manual.pdf",
                "chunk_id": 0,
                "text": "Pressure | 0.82 MPa",
                "metadata": {
                    "source": "manual.pdf",
                    "page": 4,
                    "block_type": "table",
                    "block_index": 5,
                },
            },
        ]

        citations = format_sources(results)

        self.assertIn("[1]", citations)
        self.assertIn("[2]", citations)
        self.assertIn("Block type:\ntext_group", citations)
        self.assertIn("Block index:\n2", citations)
        self.assertIn("Block type:\ntable", citations)
        self.assertIn("Block index:\n5", citations)
        self.assertEqual(citations.count("Page:\n4"), 2)

    def test_hybrid_citation_keeps_source_page_and_block_fields(self):
        self.assertIn("include_score", inspect.signature(format_sources).parameters)
        hybrid_result = {
            "score": 0.0328,
            "rrf_score": 0.0328,
            "dense_rank": 8,
            "bm25_rank": 1,
            "reranker_score": 4.2,
            "source": "manual.pdf",
            "chunk_id": 3,
            "text": "Pressure limit",
            "metadata": {
                "source": "manual.pdf",
                "page": 7,
                "block_type": "table",
                "block_index": 5,
            },
        }

        citations = format_sources([hybrid_result], include_score=False)

        self.assertIn("File:\nmanual.pdf", citations)
        self.assertIn("Page:\n7", citations)
        self.assertIn("Block type:\ntable", citations)
        self.assertIn("Block index:\n5", citations)
        self.assertIn("Chunk:\n3", citations)
        self.assertNotIn("Score:", citations)
        self.assertNotIn("0.0328", citations)
        self.assertNotIn("rrf_score", citations)
        self.assertNotIn("dense_rank", citations)
        self.assertNotIn("bm25_rank", citations)
        self.assertNotIn("reranker_score", citations)

    def test_dense_citation_keeps_existing_score_by_default(self):
        result = {
            "score": 0.91,
            "source": "manual.pdf",
            "chunk_id": 0,
            "text": "Dense evidence",
            "metadata": {"source": "manual.pdf", "page": 4},
        }

        citations = format_sources([result])

        self.assertIn("Score:\n0.9100", citations)


if __name__ == "__main__":
    unittest.main()
