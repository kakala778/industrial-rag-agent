import unittest

from src.agent.tools import KnowledgeBaseSession


class AgentReferenceSpanTests(unittest.TestCase):
    def session(self, text, block_type="text_group", metadata=None):
        values = {"source": "private.pdf", "page": 4, "block_type": block_type,
                  "block_index": 7}
        values.update(metadata or {})
        return KnowledgeBaseSession({"A": [{"text": text, "metadata": values}]})

    @staticmethod
    def by_chunk(session, chunk_id):
        return next(row["evidence_id"] for row in session.registry.values()
                    if row["chunk_id"] == chunk_id)

    def test_text_span_keeps_adjacent_condition_and_is_exact_parent_slice(self):
        condition = "仅限室内安装。"
        text = "背景说明。" + ("x" * 395) + condition + " " + " ".join(
            f"detail{i}" for i in range(180))
        session = self.session(text)
        reference = session.render_evidence_reference(self.by_chunk(session, 1))
        span = reference["span"]
        self.assertLessEqual(len(reference["excerpt"]), 1200)
        self.assertIn(condition, reference["excerpt"])
        self.assertEqual(reference["excerpt"], text[span["ranges"][0]["start"]:
                                                         span["ranges"][0]["end"]])
        self.assertEqual(reference["source"], "A")
        self.assertEqual(reference["page"], 4)
        self.assertEqual(reference["block_type"], "text_group")
        self.assertEqual(reference["block_index"], 7)

    def test_table_span_contains_complete_rows_and_declared_header(self):
        rows = ["Column | Unit"] + [f"row{i:02} | value{i:02}" for i in range(80)]
        text = "\n".join(rows)
        session = self.session(text, "table", {"table_header_rows": [0]})
        reference = session.render_evidence_reference(self.by_chunk(session, 1))
        ranges = reference["span"]["ranges"]
        self.assertEqual(reference["excerpt"], "".join(text[r["start"]:r["end"]] for r in ranges))
        self.assertIn("Column | Unit", reference["excerpt"])
        selected = [line for line in reference["excerpt"].splitlines() if line.startswith("row")]
        self.assertTrue(selected)
        for line in selected:
            self.assertIn(line, rows)
        self.assertNotIn("row79 | value79", reference["excerpt"])
        self.assertLessEqual(len(reference["excerpt"]), 1200)
        self.assertEqual(reference["span"]["kind"], "table_rows_with_header")

    def test_ambiguous_table_or_oversized_row_falls_back_to_exact_child(self):
        repeated = "repeat " * 600
        session = self.session(repeated, "table")
        child_id = self.by_chunk(session, 1)
        child = session.registry[child_id]["text"]
        reference = session.render_evidence_reference(child_id)
        self.assertEqual(reference["excerpt"], child)
        self.assertEqual(reference["span"]["kind"], "child_only_ambiguous")
        self.assertIsNone(reference["span"]["ranges"])

        long_row = "A | " + ("x" * 1400)
        session = self.session(long_row, "table")
        child_id = self.by_chunk(session, 0)
        reference = session.render_evidence_reference(child_id)
        self.assertEqual(reference["span"]["kind"], "child_only_row_too_long")
        self.assertLessEqual(len(reference["excerpt"]), 1200)

    def test_reference_provenance_and_digest_are_host_derived(self):
        from hashlib import sha256
        session = self.session("rated pressure is 10 MPa")
        evidence_id = next(iter(session.registry))
        reference = session.render_evidence_reference(evidence_id)
        self.assertEqual(reference["evidence_id"], evidence_id)
        self.assertEqual(reference["evidence_id"], evidence_id)
        self.assertEqual(reference["excerpt_sha256"],
                         sha256(reference["excerpt"].encode("utf-8")).hexdigest())
        self.assertEqual(reference["chunk_id"], session.registry[evidence_id]["chunk_id"])
        with self.assertRaises(ValueError):
            session.render_evidence_reference("ev_forged")


if __name__ == "__main__":
    unittest.main()
