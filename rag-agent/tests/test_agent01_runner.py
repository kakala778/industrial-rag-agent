"""Freeze and baseline integrity checks without PDFs, models or network calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation import evaluate_agent01 as runner


class Runner01Tests(unittest.TestCase):
    def test_frozen_search_replay_validates_query_and_copies_observations(self):
        from evaluation.evaluate_agent01 import SearchReplaySession
        from src.agent.tools import ToolResult
        class Session:
            def resolve_scopes(self, scopes): return tuple(scopes)
            def lookup_evidence(self, **kwargs): return ToolResult("no_evidence")
            def search_knowledge(self, **kwargs): raise AssertionError("must not recalculate frozen RAG")
        history = [{"query": "q", "scopes": ["A"], "status": "ok",
                    "results": [{"evidence_id": "ev_a", "source": "A", "text": "real cached evidence"}]}]
        replay = SearchReplaySession(Session(), history)
        result = replay.search_knowledge(query="q", scopes=["A"])
        result.results[0]["text"] = "mutated"
        self.assertEqual(replay.search_knowledge(query="q", scopes=["A"]).results[0]["text"], "real cached evidence")
        with self.assertRaises(ValueError): replay.search_knowledge(query="changed", scopes=["A"])
        with self.assertRaises(ValueError): replay.search_knowledge(query="q", scopes=["B"])

    def test_changed_or_added_input_is_rejected_before_loading_baseline(self):
        for mutation in ("change", "add"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); protected = root / "inputs"; protected.mkdir()
                source = protected / "input.txt"; source.write_text("frozen")
                manifest = dict(version="agent01-industrial-v1", tasks=[{}] * 8,
                                protected_hashes=runner.inventory([source]), protected_roots=[str(protected)])
                target = root / "frozen.json"; target.write_text(json.dumps(manifest))
                if mutation == "change": source.write_text("changed")
                else: (protected / "new.txt").write_text("new")
                with patch.object(runner, "PRIVATE_ROOT", root), patch.object(runner, "load_baseline") as baseline:
                    with self.assertRaisesRegex(RuntimeError, "inventory changed"):
                        runner.run_industrial(target)
                    baseline.assert_not_called()

    def test_search_replay_rejects_evidence_that_differs_from_cache(self):
        from evaluation.evaluate_agent01 import SearchReplaySession
        class Session:
            registry = {"ev_a": {"evidence_id": "ev_a", "text": "original"}}
        history = [{"query": "q", "scopes": ["A"], "status": "ok",
                    "results": [{"evidence_id": "ev_a", "text": "changed"}]}]
        with self.assertRaisesRegex(ValueError, "original registry"):
            SearchReplaySession(Session(), history)

    def test_initialization_failure_preserves_protected_audit_and_failure_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); protected = root / "inputs"; protected.mkdir()
            source = protected / "input.txt"; source.write_text("frozen")
            task = dict(documents=["A=missing.pdf", "B=missing.pdf"], query="q", scopes=["A", "B"], oracle={})
            manifest = dict(version="agent01-industrial-v1", tasks=[task] * 8,
                            protected_hashes=runner.inventory([source]), protected_roots=[str(protected)],
                            review_authority="test", cache_root=str(root / "cache"))
            target = root / "frozen.json"; target.write_text(json.dumps(manifest))
            with patch.object(runner, "PRIVATE_ROOT", root), patch.object(runner, "load_baseline", return_value={}):
                with patch.object(runner, "load_corpus", side_effect=ValueError("cache unavailable")):
                    with self.assertRaisesRegex(ValueError, "cache unavailable"):
                        runner.run_industrial(target, arms=("A",))
            result = json.loads(next(root.glob("run_*/results.json")).read_text())
            self.assertTrue(result["protected_inputs_unchanged"])
            self.assertEqual(result["mineru_calls"], 0)
            self.assertEqual(source.read_text(), "frozen")
