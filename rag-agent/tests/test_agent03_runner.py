import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path


class Agent03RunnerTests(unittest.TestCase):
    def test_baseline_must_match_all_frozen_inputs_and_exact_task_set(self):
        from evaluation.evaluate_agent03 import _validate_baseline_binding
        frozen_hashes = {"manifest": "m1", "observations": "o1",
                         "reviews": "r1", "pricing": "p1"}
        task_ids = {"R01", "R02"}
        baseline_raw = {"frozen_hashes": dict(frozen_hashes),
                        "arms": {"B": {task_id: {} for task_id in task_ids}}}
        baseline_reviewed = {"frozen_hashes": dict(frozen_hashes)}
        _validate_baseline_binding(baseline_raw, baseline_reviewed,
                                   frozen_hashes, task_ids)

        mismatched_raw = json.loads(json.dumps(baseline_raw))
        mismatched_raw["frozen_hashes"]["observations"] = "other"
        with self.assertRaisesRegex(ValueError, "frozen inputs"):
            _validate_baseline_binding(mismatched_raw, baseline_reviewed,
                                       frozen_hashes, task_ids)

        mismatched_reviewed = json.loads(json.dumps(baseline_reviewed))
        mismatched_reviewed["frozen_hashes"]["pricing"] = "other"
        with self.assertRaisesRegex(ValueError, "frozen inputs"):
            _validate_baseline_binding(baseline_raw, mismatched_reviewed,
                                       frozen_hashes, task_ids)

        missing_task = json.loads(json.dumps(baseline_raw))
        missing_task["arms"]["B"].pop("R02")
        with self.assertRaisesRegex(ValueError, "task set"):
            _validate_baseline_binding(missing_task, baseline_reviewed,
                                       frozen_hashes, task_ids)

    def test_frozen_inputs_detect_any_protected_file_change(self):
        from evaluation.evaluate_agent03 import FrozenInputs
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protected = root / "frozen.bin"
            protected.write_bytes(b"frozen")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"protected_roots": [str(root / "inputs")],
                                            "protected_hashes": {}}), encoding="utf-8")
            review = root / "review.json"
            review.write_text("{}", encoding="utf-8")
            (root / "inputs").mkdir()
            protected.replace(root / "inputs" / "frozen.bin")
            manifest.write_text(json.dumps({"protected_roots": [str(root / "inputs")],
                                            "protected_hashes": {
                                                str((root / "inputs" / "frozen.bin").resolve()):
                                                hashlib.sha256(b"frozen").hexdigest()}}), encoding="utf-8")
            extras = []
            for index in range(4):
                artifact = root / f"artifact-{index}.json"
                artifact.write_text("{}", encoding="utf-8")
                extras.append(artifact)
            guard = FrozenInputs(manifest, review, *extras)
            guard.verify()
            (root / "inputs" / "frozen.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "frozen"):
                guard.verify()

    def test_human_review_sheet_has_target_fields_and_pending_status(self):
        from evaluation.evaluate_agent03 import write_human_review_sheet
        manifest = {"tasks": [{"id": "R01", "scopes": ["A", "B"], "oracle": {
            "A": {"field_groups": [["field"]], "unit_groups": [["unit"]],
                  "condition_groups": [["condition"]], "expected_available": True},
            "B": {"field_groups": [["field"]], "unit_groups": [],
                  "condition_groups": [], "expected_available": False}}}]}
        reviews = {"R01": {"ev_a": {"scope": "A", "label": "RELEVANT"},
                           "ev_b": {"scope": "B", "label": "IRRELEVANT"}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review.csv"
            write_human_review_sheet(path, manifest, reviews)
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual([row["task_id"] for row in rows], ["R01", "R01"])
        self.assertEqual(rows[0]["target_field"], '[["field"]]')
        self.assertEqual(rows[0]["target_unit"], '[["unit"]]')
        self.assertEqual(rows[0]["expected_evidence_ids"], "ev_a")
        self.assertEqual(rows[1]["expected_evidence_ids"], "")
        self.assertTrue(all(row["review_status"] == "pending_human_pdf_review" for row in rows))

    def test_recording_selector_keeps_parsed_action_but_sanitizes_failure(self):
        from evaluation.evaluate_agent03 import RecordingSelector
        selector = RecordingSelector(lambda state: {"action": "SEARCH", "query": "q", "scopes": ["A"]})
        action = selector(object())
        self.assertEqual(action["action"], "SEARCH")
        self.assertEqual(selector.actions, [action])
        failed = RecordingSelector(lambda state: (_ for _ in ()).throw(RuntimeError("secret key contents")))
        with self.assertRaises(RuntimeError):
            failed(object())
        self.assertEqual(failed.failures, ["RuntimeError"])
        self.assertNotIn("secret", json.dumps(failed.failures))


if __name__ == "__main__":
    unittest.main()
