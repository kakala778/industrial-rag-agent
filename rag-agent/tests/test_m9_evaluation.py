"""Tests that experimental retrieval uses the original evidence scorer."""
import unittest
import numpy as np
from src.retrieval import retrieve
from evaluation.evaluate_pdf_retrieval import evaluate_questions


class QueryModel:
    def encode(self, *_args, **_kwargs):
        return np.array([1.0, 0.0])


class M9EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.questions = [{"question": "fixture question", "expected_source": "fixture.pdf",
                           "expected_page": 1, "expected_keywords": ["P-101"]}]
        self.chunks = [{"text": "P-101", "source": "fixture.pdf", "chunk_id": 0,
                        "metadata": {"source": "fixture.pdf", "page": 1}}]
        self.embeddings = np.array([[1.0, 0.0]])

    def test_provider_dense_matches_default_report_exactly(self):
        baseline = evaluate_questions(self.questions, QueryModel(), self.chunks, self.embeddings)
        provider = lambda _q, limit: retrieve([1, 0], self.chunks, self.embeddings, top_k=limit)
        provided = evaluate_questions(self.questions, None, self.chunks, [], candidate_provider=provider)
        self.assertEqual(provided, baseline)

    def test_provider_uses_shared_normalized_and_raw_matching(self):
        ranks = [dict(self.chunks[0], text="Ｐ－１０１")]
        report = evaluate_questions(self.questions, None, self.chunks, [],
                                    candidate_provider=lambda _q, _limit: ranks, diagnostic_k=20)
        self.assertTrue(report["cases"][0]["normalized_keyword_hit"])
        self.assertFalse(report["cases"][0]["keyword_hit"])

    def test_provider_and_existing_reranker_use_same_candidate_limit(self):
        limits = []
        def provider(query, limit):
            self.assertEqual(query, "fixture question")
            limits.append(limit)
            return self.chunks
        class Reranker:
            def predict(self, pairs, **_kwargs):
                return [0.5 for _ in pairs]
        report = evaluate_questions(self.questions, None, self.chunks, [], mode="reranker",
                     reranker_model=Reranker(), candidate_provider=provider)
        self.assertEqual(limits, [20])
        self.assertTrue(report["cases"][0]["normalized_keyword_hit"])

    def test_unscored_questions_do_not_call_provider(self):
        questions=[{"question":"unknown", "category":"unanswerable"}]
        def provider(*_args):
            self.fail("unscored row reached retrieval")
        report=evaluate_questions(questions,None,self.chunks,[],candidate_provider=provider)
        self.assertEqual(report["not_applicable_total"],1)

    def test_hybrid_recalls_lexical_evidence_absent_from_dense_candidates(self):
        from src.lexical_retrieval import BM25Index, rrf_fuse
        irrelevant = {"text": "Other parameter", "source": "other.pdf", "chunk_id": 0,
                      "score": 0.99, "metadata": {"source": "other.pdf", "page": 9}}
        lexical = BM25Index(self.chunks).search("P-101", 20)
        fused = rrf_fuse([irrelevant], lexical)
        dense = evaluate_questions(self.questions, None, self.chunks, [],
                                  candidate_provider=lambda _q, _k: [irrelevant])
        hybrid = evaluate_questions(self.questions, None, self.chunks, [],
                                   candidate_provider=lambda _q, _k: fused)
        self.assertFalse(dense["cases"][0]["normalized_keyword_hit"])
        self.assertTrue(hybrid["cases"][0]["normalized_keyword_hit"])
        self.assertEqual(fused[1]["bm25_rank"], 1)
        self.assertIsNone(fused[1]["dense_rank"])
