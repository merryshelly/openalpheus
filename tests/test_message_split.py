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

            chunks = FakeMatrixBot._repair_fences(chunks)

        return chunks

    @staticmethod
    def _repair_fences(chunks):
        if not chunks:
            return chunks

        CONT_END = "\n\n[\u2026continued]"
        CONT_START = "[\u2026continued]\n\n"

        result = []
        fence_opener_for_next = None

        for chunk in chunks:
            end_marker = ""
            start_marker = ""
            content = chunk

            if content.endswith(CONT_END):
                end_marker = CONT_END
                content = content[:-len(end_marker)]

            if content.startswith(CONT_START):
                start_marker = CONT_START
                content = content[len(start_marker):]

            if fence_opener_for_next:
                content = fence_opener_for_next + content
                fence_opener_for_next = None

            lines = content.split("\n")
            fence_count = 0
            last_opener = "```"

            for line in lines:
                stripped = line.strip()
                if stripped.startswith("```"):
                    fence_count += 1
                    if fence_count % 2 == 1:
                        last_opener = stripped

            if fence_count % 2 == 1:
                content += "\n```"
                fence_opener_for_next = last_opener + "\n"

            result.append(start_marker + content + end_marker)

        return result


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


    def test_no_content_lost_with_fences(self):
        """Content inside code fences is preserved across splits."""
        code = "```python\n" + "\n".join(f"line_{i} = {i}" for i in range(50)) + "\n```"
        text = "Before code:\n\n" + code + "\n\nAfter code."
        result = split(text, limit=200)
        # Strip markers and fences, verify all lines present
        full = "\n".join(
            c.replace("\n\n[\u2026continued]", "").replace("[\u2026continued]\n\n", "")
            for c in result
        )
        for i in range(50):
            self.assertIn(f"line_{i} = {i}", full)


class TestFenceRepair(unittest.TestCase):
    """Tests for code fence repair across chunked messages."""

    def test_split_inside_code_block_closes_and_reopens(self):
        """A code fence split across chunks gets closed and reopened."""
        code_lines = "\n".join(f"x = {i}" for i in range(30))
        text = f"Before:\n\n```python\n{code_lines}\n```\n\nAfter."
        result = split(text, limit=200)
        self.assertGreater(len(result), 1)

        # First chunk should end with ``` (closed fence) before continuation marker
        first_content = result[0].replace("\n\n[\u2026continued]", "")
        self.assertTrue(first_content.rstrip().endswith("```"),
                        f"First chunk should close fence:\n{result[0]}")

        # Second chunk should reopen with ```python after continuation marker
        second_content = result[1].replace("[\u2026continued]\n\n", "", 1)
        self.assertTrue(second_content.lstrip().startswith("```python"),
                        f"Second chunk should reopen fence:\n{result[1]}")

    def test_no_repair_needed_when_fences_balanced(self):
        """Chunks with balanced fences are not modified."""
        chunk1 = "```\ncode\n```\n\nText"
        chunk2 = "More text\n\n```\ncode2\n```"
        text = chunk1 + "\n\n" + chunk2
        result = split(text, limit=len(chunk1) + 20)
        # Both chunks should have balanced fences — no extra ``` added
        for chunk in result:
            content = chunk.replace("\n\n[\u2026continued]", "").replace("[\u2026continued]\n\n", "")
            fence_count = sum(1 for line in content.split("\n") if line.strip().startswith("```"))
            self.assertEqual(fence_count % 2, 0,
                             f"Chunk should have balanced fences:\n{chunk}")

    def test_language_tag_preserved(self):
        """The language tag (```python, ```bash) is preserved on reopening."""
        text = "Start\n\n```bash\n" + "\n".join(f"echo {i}" for i in range(30)) + "\n```\n\nEnd"
        result = split(text, limit=200)
        if len(result) > 1:
            second_content = result[1].replace("[\u2026continued]\n\n", "", 1)
            self.assertIn("```bash", second_content)

    def test_multiple_code_blocks_only_last_matters(self):
        """Multiple code blocks — only unclosed last one triggers repair."""
        text = "```\nblock1\n```\n\nBetween\n\n```python\n" + "x\n" * 50 + "```\n\nEnd"
        result = split(text, limit=150)
        # Should still have all content and valid fences
        full = "\n".join(
            c.replace("\n\n[\u2026continued]", "").replace("[\u2026continued]\n\n", "")
            for c in result
        )
        self.assertIn("block1", full)
        self.assertIn("Between", full)

    def test_inline_backticks_not_counted(self):
        """Inline `code` backticks don't affect fence tracking."""
        text = "Use `foo` and `bar`\n\n" + "More `inline` text\n\n" * 20
        result = split(text, limit=200)
        # No fence repair needed — no actual code fences
        for chunk in result:
            content = chunk.replace("\n\n[\u2026continued]", "").replace("[\u2026continued]\n\n", "")
            # Should not contain standalone ``` (repair artifacts)
            lines = content.split("\n")
            fence_lines = [line for line in lines if line.strip() == "```"]
            self.assertEqual(len(fence_lines), 0,
                             f"Should have no fence artifacts:\n{chunk}")

    def test_three_way_split_through_code_block(self):
        """A code block spanning 3 chunks gets properly repaired at each boundary."""
        code_lines = "\n".join(f"line_{i}" for i in range(100))
        text = f"```\n{code_lines}\n```"
        result = split(text, limit=400)
        self.assertGreaterEqual(len(result), 3)

        # Every chunk should have balanced fences
        for i, chunk in enumerate(result):
            content = chunk.replace("\n\n[\u2026continued]", "").replace("[\u2026continued]\n\n", "")
            fence_count = sum(1 for line in content.split("\n") if line.strip().startswith("```"))
            self.assertEqual(fence_count % 2, 0,
                             f"Chunk {i} should have balanced fences:\n{chunk}")


if __name__ == "__main__":
    unittest.main()
