import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf

from app.runner import (
    AgentExecutionError,
    DocumentLoadError,
    run_uploaded_pdf_research,
)
from src.agent.io import load_corpus as real_load_corpus


class ClarifyPolicy:
    action_contract = "evidence_reference"

    def __call__(self, state):
        return {"action": "CLARIFY", "question": "Which parameter should be located?"}


class StreamlitRunnerTests(unittest.TestCase):
    @staticmethod
    def one_page_pdf():
        document = pymupdf.open()
        page = document.new_page()
        page.insert_text((72, 72), "TX-03 transmission system technical specification")
        content = document.tobytes()
        document.close()
        return content

    def test_uploaded_pdf_runs_existing_harness_and_removes_temporary_file(self):
        captured = {}

        def record_load(specifications, **kwargs):
            alias, path = specifications[0].split("=", 1)
            captured["alias"] = alias
            captured["path"] = Path(path)
            captured["existed_during_load"] = Path(path).is_file()
            return real_load_corpus(specifications, **kwargs)

        with patch("app.runner.load_corpus", side_effect=record_load), \
                patch("app.runner.OllamaActionSelector", return_value=ClarifyPolicy()):
            result = run_uploaded_pdf_research(
                self.one_page_pdf(), "Find the TX-03 bitrate requirement."
            )

        self.assertEqual(captured["alias"], "S1")
        self.assertTrue(captured["existed_during_load"])
        self.assertFalse(captured["path"].exists())
        self.assertEqual(result.page_count, 1)
        self.assertEqual(result.state.status, "clarify")
        self.assertEqual(result.state.trace[0]["action"], "CLARIFY")
        self.assertIn(r"Find the TX\-03 bitrate requirement\.", result.report)
        self.assertNotIn(str(captured["path"]), result.report)

    def test_empty_task_is_rejected_before_pdf_loading(self):
        with patch("app.runner.load_corpus") as load:
            with self.assertRaisesRegex(ValueError, "task"):
                run_uploaded_pdf_research(self.one_page_pdf(), "  ")

        load.assert_not_called()

    def test_pdf_parse_failure_is_reported_as_a_document_error(self):
        with patch("app.runner.load_corpus", side_effect=ValueError("malformed PDF")):
            with self.assertRaisesRegex(DocumentLoadError, "malformed PDF"):
                run_uploaded_pdf_research(self.one_page_pdf(), "Find the bitrate.")

    def test_agent_failure_is_reported_separately_from_pdf_loading(self):
        corpus = {"S1": [{"text": "TX-03 specification", "metadata": {"page": 1}}]}
        with patch("app.runner.load_corpus", return_value=corpus), \
                patch("app.runner.AgentHarness") as harness:
            harness.return_value.run.side_effect = RuntimeError("selector unavailable")
            with self.assertRaisesRegex(AgentExecutionError, "selector unavailable") as caught:
                run_uploaded_pdf_research(self.one_page_pdf(), "Find the bitrate.")
        self.assertEqual(caught.exception.page_count, 1)


if __name__ == "__main__":
    unittest.main()
