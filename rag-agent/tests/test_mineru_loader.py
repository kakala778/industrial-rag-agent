"""Focused tests for the local MinerU adapter and PDF parser selector."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf

from src.document_loader import load_pdf
from src.mineru_loader import (
    _chart_body_text,
    _default_runner_path,
    _documents_from_middle_json,
    load_pdf_with_mineru,
)
from src.pdf_rag_demo import build_argument_parser
from src.retrieval import split_documents


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
                            "type": "table_body",
                            "content": "<table><tr><td>TEST-PUMP-01</td></tr></table>",
                            "bbox": [1, 9, 3, 12],
                            "image_path": "images/table_0.jpg",
                        }
                    ],
                },
            ],
        },
        {
            "page_idx": 1,
            "blocks": [
                {"type": "text", "content": [{"type": "text", "content": "Second page text."}]}
            ],
        },
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

    def test_default_runner_uses_integrated_mineru_directory(self):
        repo_root = self.root / "industrial-rag-agent"
        project_root = repo_root / "rag-agent"
        expected_runner = repo_root / "minerU" / "mineru-405-poc" / "run-mineru.ps1"
        expected_runner.parent.mkdir(parents=True)
        expected_runner.write_text("# local runner", encoding="utf-8")

        with (
            patch("src.mineru_loader.PROJECT_ROOT", project_root),
            patch.dict("src.mineru_loader.os.environ", {"MINERU_RUNNER_PATH": ""}),
        ):
            self.assertEqual(_default_runner_path(), expected_runner)

    def test_default_runner_preserves_sibling_project_fallback(self):
        projects_root = self.root / "Projects"
        repo_root = projects_root / "AI" / "industrial-rag-agent"
        project_root = repo_root / "rag-agent"
        expected_runner = (
            projects_root / "Project" / "mineru-405-poc" / "run-mineru.ps1"
        )
        expected_runner.parent.mkdir(parents=True)
        expected_runner.write_text("# sibling runner", encoding="utf-8")

        with (
            patch("src.mineru_loader.PROJECT_ROOT", project_root),
            patch.dict("src.mineru_loader.os.environ", {"MINERU_RUNNER_PATH": ""}),
        ):
            self.assertEqual(_default_runner_path(), expected_runner)

    @staticmethod
    def _text_block(block_type, text):
        return {
            "type": block_type,
            "content": [{"type": "text", "content": text}],
        }

    def test_structured_mode_groups_text_blocks_and_preserves_block_order(self):
        middle_json = {
            "pages": [
                {
                    "page_idx": 4,
                    "blocks": [
                        self._text_block("paragraph_title", "Pump manual"),
                        self._text_block("text", "Pressure: 0.82 MPa."),
                        self._text_block("list", "Inspect the valve."),
                        {
                            "type": "table",
                            "content": [
                                {
                                    "type": "table_body",
                                    "content": (
                                        "<table><tr><th>Part</th><th>Size</th></tr>"
                                        "<tr><td>Valve</td><td>DN50</td></tr></table>"
                                    ),
                                }
                            ],
                        },
                        self._text_block("text", "After the table."),
                    ],
                }
            ]
        }

        documents = _documents_from_middle_json(middle_json, "manual.pdf")

        self.assertEqual(
            [document["metadata"]["block_type"] for document in documents],
            ["text_group", "table", "text_group"],
        )
        self.assertEqual(
            [document["metadata"]["block_index"] for document in documents],
            [0, 3, 4],
        )
        self.assertIn(
            "Pump manual\nPressure: 0.82 MPa.\nInspect the valve.",
            documents[0]["text"],
        )
        self.assertIn("Part | Size\nValve | DN50", documents[1]["text"])
        self.assertEqual(documents[2]["text"], "After the table.")
        self.assertTrue(
            all(
                document["metadata"]["page"] == 5
                and document["metadata"]["source"] == "manual.pdf"
                and document["metadata"]["parser"] == "mineru"
                for document in documents
            )
        )

    def test_image_data_uri_and_base64_payload_never_enter_structured_text(self):
        middle_json = {
            "pages": [
                {
                    "page_idx": 0,
                    "blocks": [
                        self._text_block(
                            "text",
                            "Readable text data:image/png;base64,AAAAABBB== end.",
                        ),
                        {
                            "type": "image",
                            "content": [
                                {
                                    "type": "image_body",
                                    "content": "data:image/jpeg;base64,CCCCDDDD==",
                                },
                                {
                                    "type": "image_caption",
                                    "content": "Motor assembly overview",
                                },
                            ],
                        },
                    ],
                }
            ]
        }

        documents = _documents_from_middle_json(middle_json, "manual.pdf")
        all_text = "\n".join(document["text"] for document in documents)

        self.assertIn("Readable text", all_text)
        self.assertIn("Motor assembly overview", all_text)
        self.assertNotIn("data:image/", all_text.lower())
        self.assertNotIn("AAAAABBB", all_text)
        self.assertNotIn("CCCCDDDD", all_text)

    def test_image_body_keeps_readable_ocr_text_and_removes_inline_payload(self):
        middle_json = {
            "pages": [
                {
                    "page_idx": 0,
                    "blocks": [
                        {
                            "type": "image",
                            "content": [
                                {
                                    "type": "image_body",
                                    "content": (
                                        "Readable OCR marker 417 "
                                        "data:image/png;base64,SECRET_PAYLOAD"
                                    ),
                                }
                            ],
                        }
                    ],
                }
            ]
        }

        documents = _documents_from_middle_json(middle_json, "manual.pdf")

        self.assertEqual(len(documents), 1)
        self.assertIn("Readable OCR marker 417", documents[0]["text"])
        self.assertNotIn("data:image/", documents[0]["text"].lower())
        self.assertNotIn("SECRET_PAYLOAD", documents[0]["text"])
        self.assertEqual(documents[0]["metadata"]["block_type"], "image")

    def test_table_image_payload_is_removed_while_cell_relationships_remain(self):
        middle_json = {
            "pages": [
                {
                    "page_idx": 2,
                    "blocks": [
                        {
                            "type": "table",
                            "content": [
                                {
                                    "type": "table_caption",
                                    "content": "Parts schedule",
                                },
                                {
                                    "type": "table_body",
                                    "content": (
                                        '<table><tr><th>Part</th><th>Quantity</th></tr>'
                                        '<tr><td>Valve</td><td>2</td></tr>'
                                        '<tr><td><img src="data:image/png;base64,AAAA"/>'
                                        '</td><td>3</td></tr></table>'
                                    ),
                                },
                            ],
                        }
                    ],
                }
            ]
        }

        documents = _documents_from_middle_json(middle_json, "manual.pdf")

        self.assertEqual(len(documents), 1)
        self.assertIn("Parts schedule", documents[0]["text"])
        self.assertIn("Part | Quantity", documents[0]["text"])
        self.assertIn("Valve | 2", documents[0]["text"])
        self.assertIn("3", documents[0]["text"])
        self.assertNotIn("<table", documents[0]["text"].lower())
        self.assertNotIn("data:image/", documents[0]["text"].lower())
        self.assertEqual(documents[0]["metadata"]["block_type"], "table")
        self.assertEqual(documents[0]["metadata"]["block_index"], 0)
        self.assertEqual(documents[0]["metadata"]["page"], 3)

    def test_chart_keeps_readable_labels_but_drops_custom_cell_markers(self):
        middle_json = {
            "pages": [
                {
                    "page_idx": 0,
                    "blocks": [
                        {
                            "type": "chart",
                            "content": [
                                {
                                    "type": "chart_body",
                                    "content": (
                                        "<lcel>Speed</lcel><fcel>10</fcel><nl>"
                                        "<lcel>Flow</lcel><fcel>20</fcel>"
                                    ),
                                }
                            ],
                        }
                    ],
                }
            ]
        }

        documents = _documents_from_middle_json(middle_json, "manual.pdf")

        self.assertEqual(len(documents), 1)
        self.assertIn("Speed", documents[0]["text"])
        self.assertIn("10", documents[0]["text"])
        self.assertIn("Flow", documents[0]["text"])
        self.assertIn("20", documents[0]["text"])
        self.assertNotIn("lcel", documents[0]["text"].lower())
        self.assertNotIn("fcel", documents[0]["text"].lower())
        self.assertEqual(documents[0]["metadata"]["block_type"], "chart")

    def test_chart_normalization_handles_a_long_run_of_empty_cell_markers(self):
        chart = "<ucel></ucel>" * 24 + "<fcel>Reading</fcel>"

        self.assertEqual(_chart_body_text(chart), "Reading")

    def test_empty_visual_payloads_and_text_blocks_do_not_create_documents(self):
        middle_json = {
            "pages": [
                {
                    "page_idx": 0,
                    "blocks": [
                        self._text_block("text", "  \n "),
                        {
                            "type": "image",
                            "content": [
                                {
                                    "type": "image_body",
                                    "content": "data:image/png;base64,AAAA",
                                }
                            ],
                        },
                        {
                            "type": "table",
                            "content": [
                                {
                                    "type": "table_body",
                                    "content": "data:image/png;base64,BBBB",
                                }
                            ],
                        },
                    ],
                }
            ]
        }

        self.assertEqual(_documents_from_middle_json(middle_json, "manual.pdf"), [])

    def test_flat_representation_preserves_legacy_page_documents(self):
        documents = _documents_from_middle_json(
            MIDDLE_JSON,
            "pump manual.pdf",
            representation="flat",
        )

        self.assertEqual(len(documents), 2)
        self.assertIn("<table>", documents[0]["text"])
        self.assertEqual(documents[0]["metadata"]["page"], 1)
        self.assertEqual(documents[1]["metadata"]["page"], 2)

    def test_block_metadata_survives_existing_chunking_for_citations(self):
        middle_json = {
            "pages": [
                {
                    "page_idx": 6,
                    "blocks": [
                        {
                            "index": 12,
                            "type": "table",
                            "content": [
                                {
                                    "type": "table_body",
                                    "content": (
                                        "<table><tr><td>Part</td><td>Valve</td></tr>"
                                        "</table>"
                                    ),
                                }
                            ],
                        }
                    ],
                }
            ]
        }

        documents = _documents_from_middle_json(middle_json, "manual.pdf")
        chunks = split_documents(documents)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["source"], "manual.pdf")
        self.assertEqual(chunks[0]["metadata"]["page"], 7)
        self.assertEqual(chunks[0]["metadata"]["block_type"], "table")
        self.assertEqual(chunks[0]["metadata"]["block_index"], 12)

    @patch("src.mineru_loader.subprocess.run")
    def test_flat_and_structured_representations_reuse_the_same_parse_cache(self, run):
        run.side_effect = self._fake_mineru_success
        options = {
            "runner_path": self.runner_path,
            "cache_root": self.cache_root,
            "powershell_executable": "pwsh",
        }

        flat_documents = load_pdf_with_mineru(
            self.pdf_path,
            representation="flat",
            **options,
        )
        structured_documents = load_pdf_with_mineru(
            self.pdf_path,
            representation="structured",
            **options,
        )

        self.assertEqual(run.call_count, 1)
        self.assertEqual(len(flat_documents), 2)
        self.assertEqual(len(structured_documents), 3)
        self.assertNotEqual(flat_documents[0]["text"], structured_documents[0]["text"])

    @patch("src.mineru_loader.subprocess.run")
    def test_middle_json_pages_map_to_one_based_documents(self, run):
        run.side_effect = self._fake_mineru_success

        documents = load_pdf_with_mineru(
            self.pdf_path,
            runner_path=self.runner_path,
            cache_root=self.cache_root,
            powershell_executable="pwsh",
        )

        self.assertEqual(len(documents), 3)
        self.assertEqual(
            [document["metadata"]["page"] for document in documents], [1, 1, 2]
        )
        self.assertEqual(documents[0]["metadata"]["source"], "pump manual.pdf")
        self.assertEqual(documents[0]["metadata"]["parser"], "mineru")
        self.assertIn("Pump manual", documents[0]["text"])
        self.assertIn("0.82 MPa", documents[0]["text"])
        self.assertIn("TEST-PUMP-01", documents[1]["text"])
        self.assertNotIn("<table>", documents[1]["text"])

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
            representation="structured",
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
        flat_args = parser.parse_args(
            ["manual.pdf", "--parser", "mineru", "--representation", "flat"]
        )

        self.assertEqual(baseline_args.parser, "pymupdf")
        self.assertEqual(baseline_args.representation, "structured")
        self.assertEqual(mineru_args.parser, "mineru")
        self.assertEqual(mineru_args.representation, "structured")
        self.assertEqual(flat_args.representation, "flat")


if __name__ == "__main__":
    unittest.main()
