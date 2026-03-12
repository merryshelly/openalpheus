"""Tests for MatrixBot._split_message()."""

import unittest


class FakeMatrixBot:
    """Minimal stand-in so we can test the static method without nio."""

    MAX_MESSAGE_CHARS = 25_000

    @staticmethod
    def _split_message(text: str, limit: int | None = None) -> list[str]:
        if limit is None:
            limit = FakeMatrixBot.MAX_MESSAGE_CHARS

        if len(text) <= limit:
            return [text]

        chunks: list[str] = []
        remaining = text
        marker_budget = len("\n\n[\u2026continued]")
        effective = limit - marker_budget

        while remaining:
            if len(remaining) <= limit:
                chunks.append(remaining)
                break

            candidate = remaining[:effective]
            split_pos = candidate.rfind("\n\n")

            if split_pos < effective // 4:
                split_pos = candidate.rfind("\n")

            if split_pos < effective // 4:
                split_pos = effective

            chunk = remaining[:split_pos].rstrip()
            remaining = remaining[split_pos:].lstrip("\n")
            chunks.append(chunk)

        if len(chunks) > 1:
            for i in range(len(chunks)):
                if i < len(chunks) - 1:
                    chunks[i] += "\n\n[\u2026continued]"
                if i > 0:
                    chunks[i] = "[\u2026continued]\n\n" + chunks[i]

        return chunks


split = FakeMatrixBot._split_message


class TestSplitMessage(unittest.TestCase):
    def test_short_message_not_split(self):
        text = "Hello, world!"
        result = split(text)
        self.assertEqual(result, [text])

    def test_exactly_at_limit(self):
        text = "x" * 100
        result = split(text, limit=100)
        self.assertEqual(result, [text])

    def test_splits_on_paragraph_boundary(self):
        para1 = "A" * 60
        para2 = "B" * 60
        text = para1 + "\n\n" + para2
        result = split(text, limit=100)
        self.assertEqual(len(result), 2)
        self.assertTrue(result[0].startswith("A"))
        self.assertTrue(result[1].endswith("B" * 60))

    def test_splits_on_newline_fallback(self):
        line1 = "A" * 60
        line2 = "B" * 60
        text = line1 + "\n" + line2
        result = split(text, limit=100)
        self.assertEqual(len(result), 2)

    def test_hard_cut_when_no_breaks(self):
        text = "A" * 200
        result = split(text, limit=100)
        self.assertGreater(len(result), 1)
        joined = "".join(
            c.replace("\n\n[\u2026continued]", "").replace("[\u2026continued]\n\n", "")
            for c in result
        )
        self.assertEqual(joined, text)

    def test_continuation_markers_present(self):
        para1 = "A" * 60
        para2 = "B" * 60
        text = para1 + "\n\n" + para2
        result = split(text, limit=100)
        self.assertIn("[\u2026continued]", result[0])
        self.assertTrue(result[1].startswith("[\u2026continued]"))

    def test_first_chunk_no_header(self):
        text = ("word " * 100 + "\n\n") * 5
        result = split(text, limit=300)
        self.assertFalse(result[0].startswith("[\u2026continued]"))

    def test_last_chunk_no_footer(self):
        text = ("word " * 100 + "\n\n") * 5
        result = split(text, limit=300)
        self.assertFalse(result[-1].endswith("[\u2026continued]"))

    def test_preserves_all_content(self):
        """No content lost after stripping markers."""
        paragraphs = [f"Paragraph {i}: " + "x" * 200 for i in range(20)]
        text = "\n\n".join(paragraphs)
        result = split(text, limit=500)
        cleaned = []
        for chunk in result:
            c = chunk.replace("\n\n[\u2026continued]", "").replace("[\u2026continued]\n\n", "")
            cleaned.append(c)
        rejoined = "\n\n".join(cleaned)
        for p in paragraphs:
            self.assertIn(p, rejoined)

    def test_real_world_size(self):
        """39K message (like SAW's feed output) splits to chunks under 25K."""
        text = "\n\n".join(f"## Item {i}\n" + "x" * 500 for i in range(78))
        self.assertGreater(len(text), 39_000)
        result = split(text)
        for chunk in result:
            self.assertLessEqual(len(chunk), 25_000)

    def test_three_way_split(self):
        """A message 3x the limit produces 3 chunks."""
        text = "\n\n".join(["A" * 80] * 100)
        result = split(text, limit=3000)
        self.assertGreaterEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main()
