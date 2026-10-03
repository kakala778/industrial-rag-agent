import io
import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.agent.state import AgentState


class Agent11CliTests(unittest.TestCase):
    def documents(self, count=3):
        return [f"--document={chr(65 + index)}=D:/private/{index}.md" for index in range(count)]

    def test_invalid_document_count_and_aliases_fail_before_loading_or_session_setup(self):
        from src.agent_demo import main

        cases = [self.documents(1), self.documents(5),
                 ["--document=A=one.md", "--document=A=two.md"],
                 ["--document=../A=one.md", "--document=B=two.md"],
                 ["--document=A", "--document=B=two.md"]]
        for documents in cases:
            with self.subTest(documents=documents), \
                 patch("src.agent_demo.load_corpus") as load, \
                 patch("src.agent_demo.KnowledgeBaseSession") as session, \
                 patch("src.agent_demo.OllamaActionSelector") as selector, \
                 patch("sys.stdout", new_callable=io.StringIO) as output:
                status = main(["--task", "check pressure", *documents])
                self.assertEqual(status, 2)
                load.assert_not_called()
                session.assert_not_called()
                selector.assert_not_called()
                self.assertNotIn("Report:", output.getvalue())

    def test_three_explicit_aliases_reach_harness_and_only_relative_report_is_printed(self):
        from src.agent_demo import main

        session = object()
        state = AgentState("check pressure", ["A", "B", "C"], ["A", "B", "C"],
                           status="finished", answer="private serialized answer")
        harness = Mock()
        harness.run.return_value = state
        report_path = Path("outputs/agent11/report_20261003T120000Z_demo.md")
        task = "check pressure D:\\private\\manual.pdf"
        argv = ["--task", task, *self.documents(3), "--policy", "deterministic"]
        with patch("src.agent_demo.load_corpus", return_value={"A": [], "B": [], "C": []}), \
             patch("src.agent_demo.KnowledgeBaseSession", return_value=session), \
             patch("src.agent_demo.AgentHarness", return_value=harness) as host, \
             patch("src.agent_demo.write_research_report", return_value=report_path) as writer, \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            status = main(argv)

        self.assertEqual(status, 0)
        self.assertEqual(host.call_args.args[1].__class__.__name__, "DeterministicReferencePolicy")
        harness.run.assert_called_once_with(task, ["A", "B", "C"])
        writer.assert_called_once_with(state, session)
        self.assertIn("Status: finished", output.getvalue())
        self.assertIn(report_path.as_posix(), output.getvalue())
        self.assertNotIn("D:\\private", output.getvalue())
        self.assertNotIn("private serialized answer", output.getvalue())
        self.assertNotIn('"trace"', output.getvalue())

    def test_local_qwen_and_deterministic_modes_use_reference_contract(self):
        from src.agent_demo import main

        for policy_name in ("qwen", "deterministic", None):
            with self.subTest(policy=policy_name):
                session, selector, harness = object(), object(), Mock()
                harness.run.return_value = AgentState("q", ["A", "B"], ["A", "B"], status="finished")
                with patch("src.agent_demo.load_corpus", return_value={"A": [], "B": []}), \
                     patch("src.agent_demo.KnowledgeBaseSession", return_value=session), \
                     patch("src.agent_demo.AgentHarness", return_value=harness), \
                     patch("src.agent_demo.write_research_report", return_value=Path("outputs/agent11/r.md")), \
                     patch("src.agent_demo.OllamaActionSelector", return_value=selector) as ollama, \
                     patch("src.agent_demo.DeterministicReferencePolicy", return_value=selector) as deterministic, \
                     patch("sys.stdout", new_callable=io.StringIO):
                    arguments = ["--task", "q", *self.documents(2)]
                    if policy_name is not None:
                        arguments += ["--policy", policy_name]
                    self.assertEqual(main(arguments), 0)
                if policy_name in ("qwen", None):
                    ollama.assert_called_once_with(action_contract="evidence_reference")
                    deterministic.assert_not_called()
                else:
                    deterministic.assert_called_once_with()
                    ollama.assert_not_called()

    def test_deepseek_reference_discloses_data_transfer_without_network_in_test(self):
        from src.agent_demo import main

        session, policy, harness = object(), object(), Mock()
        harness.run.return_value = AgentState("q", ["A", "B"], ["A", "B"], status="clarify")
        def assert_disclosed_before_run(*_args):
            self.assertIn("task and selected evidence excerpts leave this machine", output.getvalue())
            return harness.run.return_value
        harness.run.side_effect = assert_disclosed_before_run
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only-token"}), \
             patch("src.agent_demo.load_corpus", return_value={"A": [], "B": []}), \
             patch("src.agent_demo.KnowledgeBaseSession", return_value=session), \
             patch("src.agent_demo.AgentHarness", return_value=harness), \
             patch("src.agent_demo.DeepSeekActionSelector", return_value=policy) as deepseek, \
             patch("src.agent.deepseek_selector.urlopen") as network, \
             patch("src.agent_demo.write_research_report", return_value=Path("outputs/agent11/r.md")), \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            status = main(["--task", "q", *self.documents(2), "--policy", "deepseek-reference"])
        self.assertEqual(status, 2)
        deepseek.assert_called_once_with(action_contract="evidence_reference")
        network.assert_not_called()
        self.assertIn("task and selected evidence excerpts leave this machine", output.getvalue())
        self.assertIn("Status: clarify", output.getvalue())
        self.assertIn("Report: outputs/agent11/r.md", output.getvalue())

    def test_deepseek_without_key_stops_before_ingestion(self):
        from src.agent_demo import main

        with patch.dict(os.environ, {}, clear=True), \
             patch("src.agent_demo.load_corpus") as load, \
             patch("src.agent_demo.DeepSeekActionSelector") as selector, \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(main(["--task", "q", *self.documents(2), "--policy", "deepseek-reference"]), 1)
        load.assert_not_called()
        selector.assert_not_called()
        self.assertNotIn("Report:", output.getvalue())

    def test_incomplete_writes_report_before_returning_two_and_failures_do_not_claim_report(self):
        from src.agent_demo import main

        arguments = ["--task", "q", *self.documents(2), "--policy", "deterministic"]
        session, harness = object(), Mock()
        for terminal in ("clarify", "budget_exceeded"):
            with self.subTest(terminal=terminal):
                state = AgentState("q", ["A", "B"], ["A", "B"], status=terminal)
                harness.run.return_value = state
                with patch("src.agent_demo.load_corpus", return_value={"A": [], "B": []}), \
                     patch("src.agent_demo.KnowledgeBaseSession", return_value=session), \
                     patch("src.agent_demo.AgentHarness", return_value=harness), \
                     patch("src.agent_demo.write_research_report", return_value=Path("outputs/agent11/r.md")) as writer, \
                     patch("sys.stdout", new_callable=io.StringIO) as output:
                    self.assertEqual(main(arguments), 2)
                writer.assert_called_once_with(state, session)
                self.assertIn("Report: outputs/agent11/r.md", output.getvalue())

        with patch("src.agent_demo.load_corpus", side_effect=OSError("private input detail")), \
             patch("src.agent_demo.write_research_report") as writer, \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(main(arguments), 1)
        writer.assert_not_called()
        self.assertNotIn("Report:", output.getvalue())

        with patch("src.agent_demo.load_corpus", return_value={"A": [], "B": []}), \
             patch("src.agent_demo.KnowledgeBaseSession", return_value=session), \
             patch("src.agent_demo.AgentHarness", return_value=harness), \
             patch("src.agent_demo.write_research_report", side_effect=OSError("disk full")), \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(main(arguments), 1)
        self.assertNotIn("Report:", output.getvalue())


if __name__ == "__main__":
    unittest.main()
