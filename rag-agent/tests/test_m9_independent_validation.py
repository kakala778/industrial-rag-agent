"""M9.2 runner, freeze, privacy and audit contracts; no real model calls."""
import copy
import json
import unittest
from pathlib import Path
try:
    from evaluation import run_m9_independent_validation as runner
except ImportError:
    runner = None
from evaluation.evaluate_pdf_retrieval import evaluate_questions


class IndependentValidationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runner, "M9.2 runner not implemented")
        self.item = {"question": "PRIVATE_QUESTION", "answer": "PRIVATE_ANSWER",
                     "expected_source": "PRIVATE_FILE.pdf", "expected_page": 1,
                     "expected_keywords": ["PRIVATE_EVIDENCE"], "category": "text",
                     "original_pdf_page_visual_reviewed": True}
        self.candidate = {"text": "PRIVATE_EVIDENCE", "source": "PRIVATE_FILE.pdf", "chunk_id": 0,
                          "score": 1., "metadata": {"source": "PRIVATE_FILE.pdf", "page": 1,
                          "block_type": "text", "block_index": 0, "secret": "PRIVATE_SECRET"}}

    def test_frozen_config_keeps_m9_constants(self):
        config = runner.frozen_configuration()
        self.assertEqual((config["dense_limit"], config["bm25_limit"], config["hybrid_limit"], config["top_k"]), (20,20,20,3))
        self.assertEqual((config["bm25_k1"],config["bm25_b"],config["rrf_k"]), (1.2,.75,60))
        self.assertEqual((config["max_chars"],config["overlap"]),(500,80))

    def test_manifest_is_anonymous_and_detects_qa_edit(self):
        original=[dict(self.item, question="old question",expected_page=2)]
        manifest=runner.build_manifest([self.item],original,"a"*64,{"cache":"b"}, {"core":"c"})
        self.assertEqual(manifest["question_count"],1)
        self.assertEqual(manifest["source_distribution"],{"S001":1})
        self.assertNotIn("PRIVATE",json.dumps(manifest))
        runner.require_manifest(manifest,copy.deepcopy(manifest))
        edited=copy.deepcopy(manifest); edited["qa_sha256"]="x"
        with self.assertRaises(ValueError): runner.require_manifest(manifest,edited)

    def test_copied_question_or_old_fact_page_is_rejected(self):
        with self.assertRaises(ValueError): runner.build_manifest([self.item],[self.item],"a",{}, {})
        old=dict(self.item,question="different wording")
        with self.assertRaises(ValueError): runner.build_manifest([self.item],[old],"a",{}, {})
        bad=dict(self.item,original_pdf_page_visual_reviewed=False)
        with self.assertRaises(ValueError): runner.build_manifest([bad],[],"a",{}, {})

    def test_summary_reuses_matcher_without_m6_q17_rule_or_raw_data(self):
        questions=[dict(self.item,question="PRIVATE_QUESTION_"+str(i)) for i in range(21)]
        reports={m:evaluate_questions(questions,None,[self.candidate],[],documents=[self.candidate],
                 diagnostic_k=20,candidate_provider=lambda q,k:[self.candidate]) for m in runner.MODES}
        summary=runner.build_summary(questions,reports,[self.candidate],[self.candidate])
        self.assertEqual(summary["metrics"]["hybrid"]["recall_at20"],{"hits":21,"total":21})
        self.assertFalse(summary["cases"][16]["known_gt_uncertain"])
        self.assertEqual(summary["complementarity"]["dense_yes_bm25_yes"]["count"],21)
        self.assertNotIn("PRIVATE",json.dumps(summary))

    def test_audit_distinguishes_structural_identity_from_text_duplicates(self):
        a=self.candidate; b=dict(a,chunk_id=1); c=dict(a,text="different text",chunk_id=2)
        stats=runner.duplicate_diagnostics([a,b,c])
        self.assertEqual(stats["structural_duplicate_count"],0)
        self.assertEqual(stats["exact_text_duplicate_pairs"],[[1,2]])
        self.assertEqual(stats["near_text_duplicate_pairs"],[])
        row=runner.audit_case(self.item,[a,b,c],7)
        self.assertEqual(row["question_id"],7)
        self.assertEqual(row["evidence_rank"],1)
        self.assertNotIn("PRIVATE",json.dumps(row))
        self.assertNotIn("classification",row) # Human audit must not be guessed by code.

    def test_private_output_cannot_escape_ignored_directories(self):
        with self.assertRaises(ValueError): runner.safe_root(Path.cwd()/"docs")
        with self.assertRaises(ValueError): runner.safe_root(Path.cwd()/"outputs"/"other")

    def test_runner_has_no_tunable_retrieval_parameters(self):
        parser=runner.build_argument_parser()
        destinations={a.dest for a in parser._actions}
        self.assertTrue({"freeze","audit_only","dataset","manifest"} <= destinations)
        self.assertTrue(destinations.isdisjoint({"k1","b","rrf_k","top_k","reranker_model","chunk_size"}))
