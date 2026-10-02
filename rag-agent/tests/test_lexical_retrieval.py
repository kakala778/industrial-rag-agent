"""Behavior tests for fixed BM25 and reciprocal rank fusion."""
import math
import unittest
from src.lexical_retrieval import BM25Index, tokenize, chunk_identity, rrf_fuse


def chunk(text, index=0, page=1, block=0):
    return {"text": text, "source": "fixture.pdf", "chunk_id": index,
            "metadata": {"source": "fixture.pdf", "page": page,
                         "block_type": "table", "block_index": block}}


class LexicalRetrievalTests(unittest.TestCase):
    def test_identifiers_preserved_and_normalized(self):
        tokens = tokenize("OTN-400G P-101 0.82MPa GB/T GPON-OLT-16 ABC_12")
        self.assertEqual(tokens, ["otn-400g", "p-101", "0.82mpa", "gb/t", "gpon-olt-16", "abc_12"])
        self.assertEqual(tokenize("Ｐ－１０１ OTN–400G"), ["p-101", "otn-400g"])

    def test_chinese_unigrams_and_bigrams(self):
        self.assertEqual(tokenize("压力阀"), ["压", "力", "阀", "压力", "力阀"])
        results = BM25Index([chunk("压力阀 P-101"), chunk("温度计", 1)]).search("压力", 3)
        self.assertEqual([row["chunk_id"] for row in results], [0])

    def test_identifier_does_not_match_longer_identifier(self):
        index = BM25Index([chunk("P-1010"), chunk("P-101", 1)])
        self.assertEqual([x["chunk_id"] for x in index.search("P-101")], [1])

    def test_bm25_matches_analytic_score(self):
        index = BM25Index([chunk("a a b"), chunk("b", 1)])
        expected = math.log(2) * (2 * 2.2) / (2 + 1.2 * (0.25 + 0.75 * 3 / 2))
        self.assertAlmostEqual(index.search("a")[0]["score"], expected)
        self.assertEqual(index.search("a a"), index.search("a"))

    def test_empty_no_overlap_and_nonpositive_limit(self):
        self.assertEqual(BM25Index([]).search("query"), [])
        index = BM25Index([chunk(""), chunk("known", 1)])
        for query in ("", "   !!!", "unknown"):
            self.assertEqual(index.search(query), [])
        self.assertEqual(index.search("known", 0), [])

    def test_metadata_preserved_and_corpus_order_breaks_ties(self):
        originals = [chunk("known", 0, block=1), chunk("known", 0, block=2)]
        results = BM25Index(originals).search("known", 20)
        self.assertEqual([r["corpus_index"] for r in results], [0, 1])
        self.assertEqual(results[0]["metadata"], originals[0]["metadata"])
        results[0]["metadata"]["page"] = 99
        self.assertEqual(originals[0]["metadata"]["page"], 1)

    def test_identity_includes_document_location_not_just_chunk_id(self):
        self.assertNotEqual(chunk_identity(chunk("a", block=1)), chunk_identity(chunk("a", block=2)))
        self.assertNotEqual(chunk_identity(chunk("a", page=1)), chunk_identity(chunk("a", page=2)))
        with self.assertRaises(ValueError):
            chunk_identity({"text": "anonymous"})

    def test_rrf_formula_dedup_provenance_and_metadata(self):
        a, b = chunk("a"), chunk("b", 1)
        a["score"], b["score"] = 0.9, 0.8
        lexical_a = dict(a, score=4.2)
        results = rrf_fuse([a, a, b], [lexical_a], top_k=20)
        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[0]["rrf_score"], 2 / 61)
        self.assertAlmostEqual(results[1]["rrf_score"], 1 / 62)
        self.assertEqual(results[0]["dense_rank"], 1)
        self.assertEqual(results[0]["bm25_rank"], 1)
        self.assertEqual(results[0]["dense_score"], 0.9)
        self.assertEqual(results[0]["bm25_score"], 4.2)
        self.assertEqual(results[0]["metadata"], a["metadata"])
        self.assertNotIn("rrf_score", a)

    def test_rrf_keeps_same_local_id_on_different_blocks(self):
        a, b = chunk("a", block=1), chunk("b", block=2)
        self.assertEqual(len(rrf_fuse([a], [b])), 2)
        self.assertEqual(rrf_fuse([], []), [])
        self.assertEqual(rrf_fuse([a], [b], top_k=0), [])
        with self.assertRaises(ValueError):
            rrf_fuse([a], [], k=-1)
