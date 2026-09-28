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


if __name__ == "__main__":
    unittest.main()
