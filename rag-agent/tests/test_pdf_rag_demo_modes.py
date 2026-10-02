import contextlib
import io
import sys
import unittest
from unittest.mock import patch

import numpy as np

from src import pdf_rag_demo
from src.lexical_retrieval import BM25Index
from src.retrieval import retrieve
from src.reranker import rerank


class _EmbeddingModel:
    def encode(self, *_args, **_kwargs):
        return np.asarray([1.0, 0.0], dtype=np.float32)


class _RerankerModel:
    def predict(self, pairs, **_kwargs):
        return np.arange(len(pairs), dtype=np.float32)


def _session_fixture():
    chunks = []
    for index in range(24):
        text = f"common passage {index}"
        if index == 23:
            text += " needle"
        chunks.append(
            {
                "text": text,
                "source": "fixture.pdf",
                "chunk_id": index % 6,
                "metadata": {
                    "source": "fixture.pdf",
                    "page": index // 6 + 1,
                    "block_type": "table" if index == 23 else "text_group",
                    "block_index": index % 6,
                },
            }
        )
    embeddings = np.asarray(
        [[1.0, index / 100] for index in range(24)], dtype=np.float32
    )
    documents = [{"text": chunks[0]["text"], "metadata": chunks[0]["metadata"]}]
    return documents, chunks, _EmbeddingModel(), embeddings


class PdfRagDemoModeTests(unittest.TestCase):
    def _resolve(self, options):
        if "--retriever" in options or "--rerank" in options:
            registered = {
                option
                for action in pdf_rag_demo.build_argument_parser()._actions
                for option in action.option_strings
            }
            for option in ("--retriever", "--rerank"):
                if option in options and option not in registered:
                    self.fail(f"CLI option {option} must be available")

        resolver = getattr(pdf_rag_demo, "resolve_retrieval_options", None)
        self.assertTrue(callable(resolver), "resolve_retrieval_options must exist")
        args = pdf_rag_demo.build_argument_parser().parse_args(
            ["fixture.pdf", *options]
        )
        return resolver(mode=args.mode, retriever=args.retriever, rerank=args.rerank)

    def test_default_is_dense_without_reranker(self):
        args = pdf_rag_demo.build_argument_parser().parse_args(["fixture.pdf"])
        resolver = getattr(pdf_rag_demo, "resolve_retrieval_options", None)
        self.assertTrue(callable(resolver), "resolve_retrieval_options must exist")
        self.assertEqual(
            resolver(mode=args.mode, retriever=args.retriever, rerank=args.rerank),
            ("dense", False, False),
        )

    def test_legacy_modes_map_to_existing_behavior(self):
        expected = {
            "dense": ("dense", False, False),
            "reranker": ("dense", True, False),
            "compare": ("dense", True, True),
        }
        for mode, wanted in expected.items():
            with self.subTest(mode=mode):
                self.assertEqual(self._resolve(["--mode", mode]), wanted)

    def test_new_retriever_and_rerank_combinations(self):
        expected = {
            ("--retriever", "dense"): ("dense", False, False),
            ("--retriever", "dense", "--rerank"): ("dense", True, False),
            ("--retriever", "hybrid"): ("hybrid", False, False),
            ("--retriever", "hybrid", "--rerank"): ("hybrid", True, False),
        }
        for options, wanted in expected.items():
            with self.subTest(options=options):
                self.assertEqual(self._resolve(list(options)), wanted)

    def test_mixed_selector_options_are_rejected(self):
        for options in (
            ["--mode", "dense", "--retriever", "hybrid"],
            ["--mode", "dense", "--rerank"],
        ):
            with self.subTest(options=options):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    with patch.object(
                        sys, "argv", ["pdf_rag_demo.py", "fixture.pdf", *options]
                    ):
                        with self.assertRaises(SystemExit) as raised:
                            pdf_rag_demo.main()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("cannot be combined", stderr.getvalue().lower())

    def test_default_session_skips_bm25_and_bge(self):
        registered = {
            option
            for action in pdf_rag_demo.build_argument_parser()._actions
            for option in action.option_strings
        }
        self.assertIn("--retriever", registered, "retriever selector must be available")
        documents, chunks, model, embeddings = _session_fixture()
        with (
            patch.object(
                pdf_rag_demo,
                "initialize_pdf_retriever",
                return_value=(documents, chunks, model, embeddings),
            ),
            patch.object(pdf_rag_demo, "BM25Index", wraps=BM25Index, create=True) as bm25,
            patch.object(pdf_rag_demo, "load_reranker", wraps=pdf_rag_demo.load_reranker) as load_bge,
            patch.object(pdf_rag_demo, "generate_answer", return_value="fixture answer"),
            patch("builtins.input", side_effect=["common", ""]),
            patch.object(sys, "argv", ["pdf_rag_demo.py", "fixture.pdf"]),
        ):
            self.assertEqual(pdf_rag_demo.main(), 0)

        bm25.assert_not_called()
        load_bge.assert_not_called()

    def test_hybrid_session_reuses_one_index_and_reranks_top20(self):
        registered = {
            option
            for action in pdf_rag_demo.build_argument_parser()._actions
            for option in action.option_strings
        }
        self.assertIn("--retriever", registered, "retriever selector must be available")
        documents, chunks, model, embeddings = _session_fixture()
        constructed_indexes = []

        def build_index(indexed_chunks):
            index = BM25Index(indexed_chunks)
            constructed_indexes.append(index)
            return index

        with (
            patch.object(
                pdf_rag_demo,
                "initialize_pdf_retriever",
                return_value=(documents, chunks, model, embeddings),
            ),
            patch.object(pdf_rag_demo, "BM25Index", side_effect=build_index, create=True) as bm25,
            patch.object(pdf_rag_demo, "load_reranker", return_value=_RerankerModel()) as load_bge,
            patch.object(pdf_rag_demo, "rerank", wraps=rerank) as rerank_spy,
            patch.object(pdf_rag_demo, "generate_answer", return_value="fixture answer"),
            patch("builtins.input", side_effect=["common needle", "common", ""]),
            patch.object(
                sys,
                "argv",
                ["pdf_rag_demo.py", "fixture.pdf", "--retriever", "hybrid", "--rerank"],
            ),
        ):
            self.assertEqual(pdf_rag_demo.main(), 0)

        bm25.assert_called_once_with(chunks)
        self.assertEqual(len(constructed_indexes), 1)
        load_bge.assert_called_once()
        self.assertEqual(rerank_spy.call_count, 2)
        for call in rerank_spy.call_args_list:
            self.assertEqual(len(call.args[1]), 20)
            self.assertEqual(call.kwargs["top_k"], 3)

    def test_compare_mode_still_generates_from_reranked_dense_top3(self):
        documents, chunks, model, embeddings = _session_fixture()
        stdout = io.StringIO()
        with (
            patch.object(
                pdf_rag_demo,
                "initialize_pdf_retriever",
                return_value=(documents, chunks, model, embeddings),
            ),
            patch.object(pdf_rag_demo, "load_reranker", return_value=_RerankerModel()),
            patch.object(pdf_rag_demo, "generate_answer", return_value="fixture answer"),
            patch("builtins.input", side_effect=["common", ""]),
            patch.object(sys, "argv", ["pdf_rag_demo.py", "fixture.pdf", "--mode", "compare"]),
            contextlib.redirect_stdout(stdout),
        ):
            self.assertEqual(pdf_rag_demo.main(), 0)

        self.assertIn("Dense results:", stdout.getvalue())
        self.assertIn("Reranker results:", stdout.getvalue())
        self.assertIn("Generating answer from reranker Top-3 context", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
