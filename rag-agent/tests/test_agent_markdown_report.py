import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.agent.state import AgentState
from src.agent.tools import KnowledgeBaseSession
from test_agent_tools import TinyEncoder


def report_fixture(scopes, *, empty_scopes=(), task="核对设备额定参数"):
    corpus = {}
    for index, scope in enumerate(scopes):
        corpus[scope] = ([] if scope in empty_scopes else [{
            "text": f"设备 {scope} 的额定压力为 {10 + index} MPa。",
            "metadata": {"page": index + 1, "block_type": "paragraph", "block_index": index},
        }])
    session = KnowledgeBaseSession(corpus, model=TinyEncoder())
    state = AgentState(task, list(scopes), list(scopes), status="finished")
    state.findings = []
    for scope in scopes:
        ids = [eid for eid, row in session.registry.items() if row["source"] == scope]
        if not ids:
            state.search_history.append({"scopes": [scope], "query": task,
                                         "status": "no_evidence", "results": []})
            state.findings.append({"scope": scope, "status": "no_evidence_found"})
            continue
        evidence_id = ids[0]
        state.search_history.append({"scopes": [scope], "query": task, "status": "ok",
                                     "results": [session.registry[evidence_id]]})
        state.evidence_ids.append(evidence_id)
        state.looked_up_evidence[evidence_id] = session.lookup_evidence(evidence_id).results[0]
        reference = session.render_evidence_reference(evidence_id)
        state.findings.append({"scope": scope, "status": "evidence_found", "evidence": [reference]})
    return state, session


class AgentMarkdownReportTests(unittest.TestCase):
    def test_finished_reports_render_two_and_four_scopes_from_host_references(self):
        from src.agent.markdown_report import render_research_report

        for scopes, empty in ((["A", "B"], ["B"]), (["A", "B", "C", "D"], [])):
            state, session = report_fixture(scopes, empty_scopes=empty)
            report = render_research_report(state, session)
            with self.subTest(scopes=scopes):
                self.assertTrue(report.startswith("# Industrial Research Report"))
                self.assertIn("## Task", report)
                self.assertIn("## Sources", report)
                self.assertIn("## Evidence", report)
                self.assertIn("## Findings", report)
                self.assertIn("## Limitations", report)
                for scope in scopes:
                    self.assertIn(f"Source `{scope}`", report)
                    self.assertIn(f"`{scope}`", report)
                for scope in set(scopes) - set(empty):
                    evidence = next(row for row in state.findings if row["scope"] == scope)["evidence"][0]
                    self.assertIn(evidence["evidence_id"], report)
                    self.assertIn(f"page {evidence['page']}", report)
                    self.assertIn(f"{evidence['block_type']}:{evidence['block_index']}", report)
                    self.assertIn(evidence["excerpt"], report)
                    self.assertIn("evidence_found", report)
                if empty:
                    self.assertIn("no_evidence_found", report)
                    self.assertIn("does not establish that the document contains no relevant information", report)
                self.assertIn("does not establish relevance", report)
                self.assertNotIn("same_text", report)
                self.assertIn("no engineering or compliance verdict", report.lower())

    def test_terminal_clarify_budget_and_tool_error_reports_mark_prior_evidence_preliminary(self):
        from src.agent.markdown_report import render_research_report

        for status in ("clarify", "budget_exceeded", "tool_error"):
            state, session = report_fixture(["A", "B"])
            evidence_id = state.findings[0]["evidence"][0]["evidence_id"]
            state.status = status
            state.findings = []
            state.pending_clarification = "请明确设备型号。" if status == "clarify" else ""
            state.answer = "private raw state must not be printed"
            state.errors = ["provider body must not be printed"]
            state.trace = [{"raw_response": "secret response"}]
            report = render_research_report(state, session)
            with self.subTest(status=status):
                self.assertIn(f"Terminal status: `{status}`", report)
                self.assertIn("preliminary", report.lower())
                self.assertIn(evidence_id, report)
                self.assertNotIn("private raw state", report)
                self.assertNotIn("provider body", report)
                self.assertNotIn("secret response", report)
                if status == "clarify":
                    self.assertIn("## Clarification", report)
                    self.assertIn("请明确设备型号", report)
                else:
                    self.assertNotIn("## Clarification", report)

        state, session = report_fixture(["A", "B"])
        state.status = "tool_error"
        state.findings = []
        state.search_history = []
        with self.assertRaises(ValueError):
            render_research_report(state, session)

    def test_untrusted_markdown_html_and_absolute_paths_are_safely_rendered(self):
        from src.agent.markdown_report import render_research_report

        task = "核对参数\n# Injected heading\n<script>alert(1)</script> D:\\Private\\design.pdf"
        state, session = report_fixture(["A", "B"], task=task)
        row = next(row for row in session.registry.values() if row["source"] == "A")
        row["text"] = "设备 A 参数。 ## Findings <script>bad()</script> `code` [fake](url) D:\\Private\\design.pdf"
        # Construct a fresh session so the hostile source text and its host ID agree.
        session = KnowledgeBaseSession({
            "A": [{"text": row["text"], "metadata": {"page": 1, "block_type": "paragraph", "block_index": 0}}],
            "B": [{"text": "设备 B 参数正常。", "metadata": {"page": 2, "block_type": "paragraph", "block_index": 0}}],
        }, model=TinyEncoder())
        state = AgentState(task, ["A", "B"], ["A", "B"], status="finished")
        for scope in ("A", "B"):
            evidence_id = next(eid for eid, evidence in session.registry.items() if evidence["source"] == scope)
            state.evidence_ids.append(evidence_id)
            state.search_history.append({"scopes": [scope], "query": task, "status": "ok",
                                         "results": [session.registry[evidence_id]]})
            state.looked_up_evidence[evidence_id] = session.lookup_evidence(evidence_id).results[0]
            state.findings.append({"scope": scope, "status": "evidence_found",
                                   "evidence": [session.render_evidence_reference(evidence_id)]})

        report = render_research_report(state, session)
        self.assertIn("\\# Injected heading", report)
        self.assertIn("&lt;script&gt;alert\\(1\\)&lt;/script&gt;", report)
        self.assertIn("\\#\\# Findings", report)
        self.assertIn("\\`code\\`", report)
        self.assertNotIn("<script>", report)
        self.assertNotIn("D:\\Private\\design.pdf", report)
        self.assertIn("\n> \\# Injected heading", report)

    def test_private_writer_is_utf8_relative_exclusive_and_rejects_unsafe_ids(self):
        from src.agent.io import write_research_report

        state, session = report_fixture(["A", "B"])
        fixed_time = datetime(2026, 10, 3, 12, 30, 0, tzinfo=timezone.utc)

        class FixedDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed_time

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("src.agent.io.datetime", FixedDateTime), \
                 patch("src.agent.io.uuid4", return_value=SimpleNamespace(hex="fixed-suffix")):
                path = write_research_report(state, session, output_root=root, report_id="fixed-suffix")
                self.assertEqual(path, Path("outputs/agent11/report_20261003T123000Z_fixed-suffix.md"))
                target = root / path
                content = target.read_bytes()
                self.assertIn("额定参数".encode("utf-8"), content)
                self.assertEqual(content.decode("utf-8"), content.decode())
                with self.assertRaises(FileExistsError):
                    write_research_report(state, session, output_root=root, report_id="fixed-suffix")
                self.assertEqual(target.read_bytes(), content)
            for report_id in ("../escape", "D:/outside", "contains space"):
                with self.subTest(report_id=report_id), self.assertRaises(ValueError):
                    write_research_report(state, session, output_root=root, report_id=report_id)
            self.assertFalse((root / "escape.md").exists())


if __name__ == "__main__":
    unittest.main()
