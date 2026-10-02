import importlib
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from src.lexical_retrieval import BM25Index, chunk_identity


class HybridRetrievalTests(unittest.TestCase):
    def _retrieve_hybrid(self):
        module_spec = importlib.util.find_spec("src.hybrid_retrieval")
        self.assertIsNotNone(module_spec, "src.hybrid_retrieval must be added")
        module = importlib.import_module("src.hybrid_retrieval")
        function = getattr(module, "retrieve_hybrid", None)
        self.assertTrue(callable(function), "retrieve_hybrid must be public")
        return module, function

    @staticmethod
    def _fixture(count=25, text_prefix="shared"):
        chunks = []
        for index in range(count):
            text = f"{text_prefix} row {index}"
            if index == 24:
                text += " needle"
            chunks.append(
                {
                    "text": text,
                    "source": "manual.pdf",
                    "chunk_id": index % 5,
                    "metadata": {
                        "source": "manual.pdf",
                        "page": index // 5 + 1,
                        "block_type": "table" if index == 24 else "text_group",
                        "block_index": index % 5,
                    },
                }
            )
        embeddings = np.asarray(
            [[1.0, index / 100] for index in range(count)], dtype=np.float32
        )
        return chunks, embeddings

    def test_fusion_uses_fixed_candidate_budgets_and_preserves_metadata(self):
        module, retrieve_hybrid = self._retrieve_hybrid()
        chunks, embeddings = self._fixture()
        bm25_index = BM25Index(chunks)

        with patch.object(module, "retrieve", wraps=module.retrieve) as dense_search:
            with patch.object(
                bm25_index, "search", wraps=bm25_index.search
            ) as lexical_search:
                results = retrieve_hybrid(
                    "shared needle",
                    np.asarray([1.0, 0.0], dtype=np.float32),
                    chunks,
                    embeddings,
                    bm25_index,
                )

        dense_search.assert_called_once()
        self.assertEqual(dense_search.call_args.kwargs["top_k"], 20)
        lexical_search.assert_called_once_with("shared needle", 20)
        self.assertEqual(len(results), 20)
        identities = [chunk_identity(result) for result in results]
        self.assertEqual(len(identities), len(set(identities)))

        lexical_only = next(result for result in results if result["chunk_id"] == 4 and result["metadata"]["page"] == 5)
        self.assertIsNone(lexical_only["dense_rank"])
        self.assertEqual(lexical_only["bm25_rank"], 1)
        self.assertEqual(lexical_only["metadata"]["block_type"], "table")
        self.assertEqual(lexical_only["metadata"]["block_index"], 4)
        self.assertEqual(lexical_only["score"], lexical_only["rrf_score"])
        self.assertTrue(all(result["source"] == "manual.pdf" for result in results))

    def test_no_bm25_overlap_keeps_dense_candidates(self):
        _, retrieve_hybrid = self._retrieve_hybrid()
        chunks, embeddings = self._fixture(text_prefix="ordinary")

        results = retrieve_hybrid(
            "unmatched query",
            np.asarray([1.0, 0.0], dtype=np.float32),
            chunks,
            embeddings,
            BM25Index(chunks),
        )

        self.assertEqual(len(results), 20)
        self.assertTrue(all(result["dense_rank"] is not None for result in results))
        self.assertTrue(all(result["bm25_rank"] is None for result in results))
        self.assertEqual(results[0]["metadata"]["page"], 1)

    def test_empty_corpus_returns_no_candidates(self):
        _, retrieve_hybrid = self._retrieve_hybrid()

        results = retrieve_hybrid(
            "any query",
            np.asarray([1.0, 0.0], dtype=np.float32),
            [],
            np.empty((0, 0), dtype=np.float32),
            BM25Index([]),
        )

        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
