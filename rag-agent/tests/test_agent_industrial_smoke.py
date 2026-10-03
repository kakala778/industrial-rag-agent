import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class IndustrialSmokeBoundaryTests(unittest.TestCase):
    def test_inventory_detects_content_changes_and_missing_files(self):
        from evaluation.run_agent0_industrial_smoke import inventory
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            file = root / "input.pdf"
            file.write_bytes(b"fixture")
            before = inventory([file])
            file.write_bytes(b"changed")
            self.assertNotEqual(before, inventory([file]))
            with self.assertRaises(FileNotFoundError):
                inventory([root / "absent.pdf"])

    def test_cache_miss_never_starts_mineru(self):
        from src.agent.io import load_corpus
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            pdf = root / "fixture.pdf"
            pdf.write_bytes(b"synthetic-not-a-real-pdf")
            with patch("src.mineru_loader._run_mineru", side_effect=AssertionError("forbidden")):
                with self.assertRaisesRegex(ValueError, "cache unavailable"):
                    load_corpus([f"A={pdf}"], parser="mineru", cache_root=root / "cache")
            self.assertFalse((root / "cache").exists())
