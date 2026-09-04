"""Tests for markdown-aware text chunking.

Interface contract:
    chunk_file(text: str, path: str) -> list[Chunk]

Chunk: path (str), start_line (int), end_line (int), text (str)

Chunking strategy:
    1. Markdown: split on ## headers. Each section includes its header.
    2. Size guard: sections > 1500 chars split on ### headers, then paragraph boundaries.
    3. Minimum chunk size: 50 chars. Trivially small fragments are skipped.
    4. Non-markdown files: paragraph splitting (double newline), then fixed 1000-char
       windows with 200-char overlap.
"""

from openalph.memory.chunker import chunk_file, Chunk


class TestChunkDataclass:

    def test_chunk_has_required_fields(self):
        c = Chunk(path="test.md", start_line=1, end_line=5, text="hello")
        assert c.path == "test.md"
        assert c.start_line == 1
        assert c.end_line == 5
        assert c.text == "hello"


class TestMarkdownSplitting:

    def test_splits_on_h2_headers(self):
        text = "## Section One\nContent one with enough text to pass the minimum chunk size threshold for testing.\n\n## Section Two\nContent two that is long enough to pass the minimum chunk size threshold of fifty characters."
        chunks = chunk_file(text, "test.md")
        assert len(chunks) >= 2
        texts = " ".join(c.text for c in chunks)
        assert "Section One" in texts
        assert "Section Two" in texts

    def test_h2_header_included_in_chunk(self):
        text = "## My Header\nBody text here that is long enough to pass the minimum chunk size threshold of fifty characters."
        chunks = chunk_file(text, "test.md")
        assert len(chunks) == 1
        assert chunks[0].text.startswith("## My Header")

    def test_preamble_before_first_h2_becomes_chunk(self):
        text = "Preamble text that is long enough to pass the minimum chunk size.\n\n## First Section\nBody text also long enough to pass minimum chunk size threshold."
        chunks = chunk_file(text, "test.md")
        assert len(chunks) == 2
        assert "Preamble text" in chunks[0].text

    def test_single_section_no_split(self):
        text = "## Only Section\nSome content here that meets the minimum size requirement for chunks."
        chunks = chunk_file(text, "test.md")
        assert len(chunks) == 1

    def test_h1_headers_do_not_split(self):
        """Only ## (h2) triggers a split, not # (h1)."""
        text = "# Title\nIntro text long enough to pass the minimum.\n\n## Section\nBody text also long enough for minimum."
        chunks = chunk_file(text, "test.md")
        assert len(chunks) == 2
        assert "# Title" in chunks[0].text

    def test_preserves_line_numbers(self):
        text = "## A\nLine 2 with enough content to pass minimum chunk size threshold for this test.\n\n## B\nLine 5 with enough content to pass minimum chunk size threshold for this test.\n"
        chunks = chunk_file(text, "test.md")
        assert chunks[0].start_line == 1
        # ## B starts after the blank line
        assert chunks[1].start_line >= 4

    def test_end_line_is_inclusive(self):
        text = "## A\nLine 2 with enough content to pass minimum chunk.\n\n## B\nLine 5 with enough content to pass minimum chunk.\n"
        chunks = chunk_file(text, "test.md")
        assert chunks[0].end_line >= chunks[0].start_line
        assert chunks[1].end_line >= chunks[1].start_line


class TestSizeGuard:

    def test_large_section_splits_on_h3(self):
        """Sections > 1500 chars split on ### sub-headers."""
        body_a = "A" * 800
        body_b = "B" * 800
        text = f"## Big Section\n### Sub A\n{body_a}\n\n### Sub B\n{body_b}"
        chunks = chunk_file(text, "test.md")
        assert len(chunks) >= 2
        texts = [c.text for c in chunks]
        assert any("Sub A" in t for t in texts)
        assert any("Sub B" in t for t in texts)

    def test_large_section_without_h3_splits_on_paragraphs(self):
        """If no ### headers, split on double-newline paragraph boundaries."""
        para1 = "First paragraph. " * 60  # ~1000 chars
        para2 = "Second paragraph. " * 60
        text = f"## Big Section\n{para1}\n\n{para2}"
        chunks = chunk_file(text, "test.md")
        assert len(chunks) >= 2

    def test_small_section_not_split(self):
        """Sections under 1500 chars remain intact."""
        text = "## Small\n" + "x" * 100
        chunks = chunk_file(text, "test.md")
        assert len(chunks) == 1


class TestMinimumChunkSize:

    def test_tiny_chunks_skipped(self):
        """Chunks under 50 chars are dropped."""
        text = "## A\nOk.\n\n## B\nThis section has enough content to be meaningful and should be kept by the chunker."
        chunks = chunk_file(text, "test.md")
        # Only B should survive (A is < 50 chars)
        assert all(len(c.text) >= 50 for c in chunks)

    def test_empty_sections_skipped(self):
        text = "## A\n\n\n## B\nReal content here that is long enough to pass the minimum threshold."
        chunks = chunk_file(text, "test.md")
        assert all(len(c.text.strip()) >= 50 for c in chunks)


class TestNonMarkdown:

    def test_plain_text_splits_on_paragraphs(self):
        para1 = "First paragraph. " * 40
        para2 = "Second paragraph. " * 40
        text = f"{para1}\n\n{para2}"
        chunks = chunk_file(text, "plain.txt")
        assert len(chunks) >= 1

    def test_long_plain_text_uses_fixed_windows(self):
        """Very long text without paragraph breaks gets fixed-size windows."""
        text = "word " * 500  # ~2500 chars, no paragraph breaks
        chunks = chunk_file(text, "notes.txt")
        assert len(chunks) >= 2

    def test_md_extension_uses_markdown_splitting(self):
        text = "## A\nContent A long enough to meet minimum.\n\n## B\nContent B also long enough to meet minimum chunk size."
        chunks = chunk_file(text, "readme.md")
        assert len(chunks) >= 1


class TestEdgeCases:

    def test_empty_string(self):
        chunks = chunk_file("", "empty.md")
        assert chunks == []

    def test_whitespace_only(self):
        chunks = chunk_file("   \n\n  \n", "ws.md")
        assert chunks == []

    def test_no_headers_markdown(self):
        """Markdown file with no ## headers treated as single chunk if long enough."""
        text = "Just a paragraph of text that is long enough to meet the minimum threshold for chunking to work."
        chunks = chunk_file(text, "no-headers.md")
        assert len(chunks) == 1

    def test_path_preserved_in_all_chunks(self):
        text = "## A\nContent A is long enough to meet minimum size.\n\n## B\nContent B is also long enough for the minimum size."
        chunks = chunk_file(text, "my/path.md")
        assert all(c.path == "my/path.md" for c in chunks)

    def test_consecutive_headers(self):
        """Two headers with no content between them."""
        text = "## A\n## B\nContent for B is sufficiently long to pass minimum chunk threshold for testing."
        chunks = chunk_file(text, "test.md")
        assert any("Content for B" in c.text for c in chunks)

    def test_unicode_content(self):
        text = "## セクション\nこれはテスト内容です。十分な長さのコンテンツが必要です。このセクションは最小チャンクサイズを超える必要があります。追加テキスト。"
        chunks = chunk_file(text, "unicode.md")
        assert len(chunks) >= 1
        assert "セクション" in chunks[0].text


# ===========================================================================
# BUG-8 — _split_large_section must not emit the final subsection twice
# BUG-15 — paragraph line numbers must survive separators with 3+ newlines
# ===========================================================================

class TestChunkerRegressions:

    def _oversized_h3_section(self):
        body = "## Big Section\n\n"
        for name, word in (("Sub A", "alpha"), ("Sub B", "bravo"), ("Sub C", "charlie")):
            body += f"### {name}\n" + (f"{word} line of text padded out here\n" * 20) + "\n"
        assert len(body) > 1500  # ensure chunk_file routes into _split_large_section
        return body

    def test_no_duplicate_final_subsection(self):
        """BUG-8: the last ### subsection was emitted twice (sentinel + trailing)."""
        chunks = chunk_file(self._oversized_h3_section(), "notes.md")
        texts = [c.text for c in chunks]
        assert len(texts) == len(set(texts)), "a chunk was emitted more than once"

    def test_final_subsection_present_exactly_once(self):
        chunks = chunk_file(self._oversized_h3_section(), "notes.md")
        charlie = [c for c in chunks if "charlie" in c.text]
        assert len(charlie) == 1

    def test_paragraph_line_numbers_survive_wide_separators(self):
        """BUG-15: citations must point at the paragraph's real lines."""
        txt = "para one is here\n\n\n\npara two is here\n\n\n\npara three is here"
        lines = txt.split("\n")
        for c in chunk_file(txt, "plain.txt"):
            actual = "\n".join(lines[c.start_line - 1:c.end_line]).strip()
            assert actual == c.text.strip(), (
                f"L{c.start_line}-{c.end_line} cites {c.text!r} but file has {actual!r}"
            )

    def test_paragraph_line_numbers_with_leading_blank_lines(self):
        txt = "\n\nfirst para\n\n\nsecond para"
        lines = txt.split("\n")
        for c in chunk_file(txt, "plain.txt"):
            actual = "\n".join(lines[c.start_line - 1:c.end_line]).strip()
            assert actual == c.text.strip()
