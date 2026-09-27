"""Focused tests for the local MinerU adapter and PDF parser selector."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf

from src.document_loader import load_pdf
from src.mineru_loader import load_pdf_with_mineru
from src.pdf_rag_demo import build_argument_parser


MIDDLE_JSON = {
    "schema": "docvortex.middle",
    "schema_version": "2.0",
    "metadata": {},
    "pages": [
        {
            "page_idx": 0,
            "blocks": [
                {
                    "type": "paragraph_title",
                    "bbox": [1, 2, 3, 4],
                    "content": [{"type": "text", "content": "Pump manual"}],
                },
                {
                    "type": "text",
                    "bbox": [1, 5, 3, 8],
                    "content": [{"type": "text", "content": "Pressure: 0.82 MPa."}],
                },
                {
                    "type": "table",
                    "bbox": [1, 9, 3, 12],
                    "content": [
                        {
                            "type": "table",
                            "content": "<table><tr><td>TEST-PUMP-01</td></tr></table>",
                            "bbox": [1, 9, 3, 12],
                            "image_path": "images/table_0.jpg",
                        }
                    ],
                },
            ],
        },
        {"page_idx": 1, "blocks": []},
    ],
}


class MineruLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.pdf_path = self.root / "pump manual.pdf"
        self.pdf_path.write_bytes(b"test pdf bytes")
        self.runner_path = self.root / "run-mineru.ps1"
        self.runner_path.write_text("# mock runner path", encoding="utf-8")
        self.cache_root = self.root / ".local" / "mineru"

    def _fake_mineru_success(self, command, **_kwargs):
        command = list(command)
        output_dir = Path(command[command.index("--output") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        middle_json = output_dir / f"{self.pdf_path.stem}.json"
        middle_json.write_text(json.dumps(MIDDLE_JSON), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "parsed", "")

    @patch("src.mineru_loader.subprocess.run")
    def test_middle_json_pages_map_to_one_based_documents(self, run):
        run.side_effect = self._fake_mineru_success

        documents = load_pdf_with_mineru(
            self.pdf_path,
            runner_path=self.runner_path,
            cache_root=self.cache_root,
            powershell_executable="pwsh",
        )

        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0]["metadata"]["page"], 1)
        self.assertEqual(documents[1]["metadata"]["page"], 2)
        self.assertEqual(documents[0]["metadata"]["source"], "pump manual.pdf")
        self.assertEqual(documents[0]["metadata"]["parser"], "mineru")
        self.assertIn("Pump manual", documents[0]["text"])
        self.assertIn("0.82 MPa", documents[0]["text"])
        self.assertIn("<table>", documents[0]["text"])

    @patch("src.mineru_loader.subprocess.run")
    def test_cached_middle_json_is_reused_until_force_is_requested(self, run):
        run.side_effect = self._fake_mineru_success
        options = {
            "runner_path": self.runner_path,
            "cache_root": self.cache_root,
            "powershell_executable": "pwsh",
        }

        load_pdf_with_mineru(self.pdf_path, **options)
        load_pdf_with_mineru(self.pdf_path, **options)
        self.assertEqual(run.call_count, 1)

        load_pdf_with_mineru(self.pdf_path, force=True, **options)
        self.assertEqual(run.call_count, 2)

    @patch("src.mineru_loader.subprocess.run")
    def test_cache_hit_does_not_require_mineru_runner(self, run):
        run.side_effect = self._fake_mineru_success
        load_pdf_with_mineru(
            self.pdf_path,
            runner_path=self.runner_path,
            cache_root=self.cache_root,
            powershell_executable="pwsh",
        )
        self.runner_path.unlink()

        with patch.dict(
            "os.environ",
            {"MINERU_RUNNER_PATH": str(self.root / "missing" / "run-mineru.ps1")},
        ):
            documents = load_pdf_with_mineru(
                self.pdf_path,
                cache_root=self.cache_root,
            )

        self.assertEqual(documents[0]["metadata"]["parser"], "mineru")
        self.assertEqual(run.call_count, 1)

    @patch("src.mineru_loader.subprocess.run")
    def test_corrupt_cached_middle_json_is_reparsed(self, run):
        run.side_effect = self._fake_mineru_success
        options = {
            "runner_path": self.runner_path,
            "cache_root": self.cache_root,
            "powershell_executable": "pwsh",
        }
        load_pdf_with_mineru(self.pdf_path, **options)
        cache_entry = next(self.cache_root.iterdir())
        middle_json = cache_entry / "output" / f"{self.pdf_path.stem}.json"
        middle_json.write_text("{broken json", encoding="utf-8")

        documents = load_pdf_with_mineru(self.pdf_path, **options)

        self.assertEqual(run.call_count, 2)
        self.assertEqual(documents[0]["metadata"]["parser"], "mineru")

    @patch("src.mineru_loader.subprocess.run")
    def test_successful_process_without_expected_json_has_clear_error(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, "parsed", "")

        with self.assertRaisesRegex(FileNotFoundError, "Middle JSON"):
            load_pdf_with_mineru(
                self.pdf_path,
                runner_path=self.runner_path,
                cache_root=self.cache_root,
                powershell_executable="pwsh",
            )

    def test_missing_pdf_fails_before_calling_mineru(self):
        with self.assertRaisesRegex(FileNotFoundError, "PDF file not found"):
            load_pdf_with_mineru(
                self.root / "missing.pdf",
                runner_path=self.runner_path,
                cache_root=self.cache_root,
                powershell_executable="pwsh",
            )

    def test_missing_mineru_runner_has_clear_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "MinerU.*runner"):
            load_pdf_with_mineru(
                self.pdf_path,
                runner_path=self.root / "missing" / "run-mineru.ps1",
                cache_root=self.cache_root,
                powershell_executable="pwsh",
            )

    @patch("src.mineru_loader.load_pdf_with_mineru")
    def test_pdf_loader_dispatches_to_selected_parser(self, mineru_loader):
        mineru_loader.return_value = [{"text": "parsed", "metadata": {"page": 1}}]

        documents = load_pdf(
            self.pdf_path,
            parser="mineru",
            force=True,
            runner_path=self.runner_path,
            cache_root=self.cache_root,
        )

        self.assertEqual(documents[0]["text"], "parsed")
        mineru_loader.assert_called_once_with(
            self.pdf_path,
            force=True,
            runner_path=self.runner_path,
            cache_root=self.cache_root,
            powershell_executable=None,
        )

    def test_pdf_loader_keeps_pymupdf_as_the_default_parser(self):
        pdf_path = self.root / "baseline.pdf"
        pdf = pymupdf.open()
        page = pdf.new_page()
        page.insert_text((72, 72), "Baseline PDF text")
        pdf.save(pdf_path)
        pdf.close()

        documents = load_pdf(pdf_path)

        self.assertEqual(documents[0]["text"].strip(), "Baseline PDF text")
        self.assertEqual(
            documents[0]["metadata"], {"source": "baseline.pdf", "page": 1}
        )

    def test_pdf_loader_rejects_unknown_parser(self):
        with self.assertRaisesRegex(ValueError, "Unsupported PDF parser"):
            load_pdf(self.pdf_path, parser="unknown")

    def test_pdf_rag_cli_defaults_to_pymupdf_and_accepts_mineru(self):
        parser = build_argument_parser()

        baseline_args = parser.parse_args(["manual.pdf"])
        mineru_args = parser.parse_args(["manual.pdf", "--parser", "mineru"])

        self.assertEqual(baseline_args.parser, "pymupdf")
        self.assertEqual(mineru_args.parser, "mineru")


if __name__ == "__main__":
    unittest.main()
