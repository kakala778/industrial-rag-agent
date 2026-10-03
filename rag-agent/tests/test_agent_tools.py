import copy
import unittest
from unittest.mock import patch

import numpy as np


class TinyEncoder:
    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return np.array([1., 0.], dtype=np.float32)
        return np.array([[1., 0.] for _ in texts], dtype=np.float32)


class TinyReranker:
    def predict(self, pairs, **kwargs):
        return np.array([len(text) for _, text in pairs], dtype=np.float32)


def corpus():
    return {alias: [{"text": text, "metadata": {
        "source": "D:/private/same.pdf", "page": 1,
        "block_type": "table", "block_index": 0, "private_path": "secret"}}]
        for alias, text in (("A", "Rated pressure: 10 MPa."),
                            ("B", "Rated pressure: 12 MPa."))}


class AgentToolsTests(unittest.TestCase):
    def setUp(self):
        from src.agent.tools import KnowledgeBaseSession
        self.Session = KnowledgeBaseSession
        self.session = self.Session(corpus(), model=TinyEncoder(),
                                    reranker_model=TinyReranker())

    def test_scopes_filter_before_dense_hybrid_and_rerank(self):
        for backend in ("dense", "hybrid"):
            result = self.session.search_knowledge("Rated pressure", ["B"], backend)
            self.assertEqual(result.status, "ok")
            self.assertEqual({r["source"] for r in result.results}, {"B"})
            self.assertEqual(result.results[0]["text"], "Rated pressure: 12 MPa.")
            self.assertNotIn("secret", str(result))
            self.assertNotIn("private", str(result))
        self.assertEqual(len(self.session.scope_indexes), 1)

    def test_ids_are_unique_stable_and_reverse_lookup(self):
        first = self.session.search_knowledge("pressure", rerank=False)
        other = self.Session(corpus(), model=TinyEncoder())
        second = other.search_knowledge("pressure", rerank=False)
        ids = [r["evidence_id"] for r in first.results]
        self.assertEqual(ids, [r["evidence_id"] for r in second.results])
        self.assertEqual(len(set(ids)), 2)
        self.assertEqual(self.session.lookup_evidence(ids[0]).results[0]["text"],
                         "Rated pressure: 10 MPa.")

    def test_lookup_cannot_read_path_or_unknown_id(self):
        for bad in ("D:/private/same.pdf", "ev_missing", None, [], {}):
            self.assertEqual(self.session.lookup_evidence(bad).status,
                             "invalid_evidence_id")

    def test_lookup_is_bounded_and_contains_original_child(self):
        text = " ".join(f"row{i:04d}=value{i:04d}" for i in range(600))
        session = self.Session({"A": [{"text": text, "metadata": {"source": "x"}}]},
                               model=TinyEncoder())
        # Every registered child, including those at the end, must be recoverable.
        for child in session.registry.values():
            row = session.lookup_evidence(child["evidence_id"]).results[0]
            self.assertLessEqual(len(row["text"]), 2000)
            self.assertIn(child["text"], row["text"])
            self.assertTrue(row["truncated"])
            start = row["offset"]
            self.assertEqual(row["text"], text[start:start + len(row["text"])])

    def test_repeated_child_falls_back_without_guessing_offset(self):
        session = self.Session({"A": [{"text": "repeat " * 800,
                                       "metadata": {"source": "x"}}]},
                               model=TinyEncoder())
        row = session.lookup_evidence(next(iter(session.registry))).results[0]
        self.assertEqual(row["context_kind"], "child_only")
        self.assertIsNone(row["offset"])

    def test_invalid_scope_and_empty_are_distinct(self):
        self.assertEqual(self.session.search_knowledge("p", ["missing"]).status,
                         "invalid_scope")
        empty = self.Session({"A": []})
        self.assertEqual(empty.search_knowledge("p", ["A"]).status, "no_evidence")
        self.assertEqual(empty.search_knowledge("p", []).status, "invalid_scope")

    def test_invalid_arguments_return_status(self):
        for kwargs in ({"top_k": True}, {"top_k": 4}, {"query": ""},
                       {"retriever": "bm25"}, {"rerank": "yes"}):
            args = dict(query="pressure", rerank=False)
            args.update(kwargs)
            self.assertEqual(self.session.search_knowledge(**args).status, "error")

    def test_duplicate_parent_and_unsafe_alias_fail_closed(self):
        duplicated = corpus()
        duplicated["A"].append(copy.deepcopy(duplicated["A"][0]))
        for value in (duplicated, {"D:/private": []}, {"../A": []}):
            with self.assertRaises(ValueError):
                self.Session(value, model=TinyEncoder())

    def test_timeout_and_error_do_not_leak_exception_details(self):
        for exception, status in ((TimeoutError("private"), "timeout"),
                                  (RuntimeError("secret path"), "error")):
            with patch.object(self.session.model, "encode", side_effect=exception):
                result = self.session.search_knowledge("p", ["A"], rerank=False)
            self.assertEqual(result.status, status)
            self.assertNotIn("private", result.message)
            self.assertNotIn("secret", result.message)

    def test_documents_and_returned_evidence_are_not_mutated(self):
        original = corpus()
        snapshot = copy.deepcopy(original)
        session = self.Session(original, model=TinyEncoder())
        row = session.search_knowledge("p", rerank=False).results[0]
        row["text"] = "forged"
        self.assertEqual(original, snapshot)
        self.assertNotEqual(session.lookup_evidence(row["evidence_id"]).results[0]["text"],
                            "forged")
