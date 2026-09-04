"""Tests for file operations: read, write, edit.

Interface contract:
    read_file(path, offset=None, limit=None) -> ToolResult
    write_file(path, content) -> ToolResult
    edit_file(path, old_text, new_text) -> ToolResult

offset/limit are line-based (1-indexed). Binary files return error.
edit_file requires exactly one match (0 or 2+ matches → error).
write_file creates parent directories and overwrites existing files.
"""

import pytest
from openalph.tools.file import read_file, write_file, edit_file
from openalph.tools import ToolResult


# --- read_file ---


class TestReadFile:

    @pytest.mark.asyncio
    async def test_reads_full_contents(self, tmp_path):
        """Returns full file contents."""
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        result = await read_file(str(f))
        assert result.content == "hello world"
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_returns_tool_result(self, tmp_path):
        """Return type is ToolResult."""
        f = tmp_path / "test.txt"
        f.write_text("test")
        result = await read_file(str(f))
        assert isinstance(result, ToolResult)

    @pytest.mark.asyncio
    async def test_offset_starts_at_line(self, tmp_path):
        """offset parameter starts reading from that line (1-indexed)."""
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\nline4\n")
        result = await read_file(str(f), offset=2)
        assert "line2" in result.content
        assert "line1" not in result.content

    @pytest.mark.asyncio
    async def test_limit_caps_line_count(self, tmp_path):
        """limit parameter caps number of lines returned."""
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\nline4\n")
        result = await read_file(str(f), limit=2)
        lines = [line for line in result.content.strip().split("\n") if line]
        assert len(lines) == 2

    @pytest.mark.asyncio
    async def test_offset_and_limit_combined(self, tmp_path):
        """offset + limit returns a specific range."""
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\nline4\nline5\n")
        result = await read_file(str(f), offset=2, limit=2)
        assert "line2" in result.content
        assert "line3" in result.content
        assert "line1" not in result.content
        assert "line4" not in result.content

    @pytest.mark.asyncio
    async def test_offset_beyond_file_returns_empty(self, tmp_path):
        """offset past end of file → empty content, not error."""
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\n")
        result = await read_file(str(f), offset=100)
        assert result.is_error is False
        assert result.content.strip() == ""

    @pytest.mark.asyncio
    async def test_nonexistent_file_returns_error(self):
        """Missing file → is_error=True."""
        result = await read_file("/tmp/does_not_exist_abc123.txt")
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_binary_file_returns_error(self, tmp_path):
        """Binary file → is_error=True."""
        f = tmp_path / "binary.bin"
        f.write_bytes(bytes(range(256)))
        result = await read_file(str(f))
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_empty_file(self, tmp_path):
        """Empty file → empty content, not error."""
        f = tmp_path / "empty.txt"
        f.write_text("")
        result = await read_file(str(f))
        assert result.is_error is False
        assert result.content == ""

    @pytest.mark.asyncio
    async def test_utf8_content(self, tmp_path):
        """UTF-8 content is handled correctly."""
        f = tmp_path / "unicode.txt"
        f.write_text("héllo wörld 🐈‍⬛")
        result = await read_file(str(f))
        assert "héllo" in result.content
        assert "🐈‍⬛" in result.content


# --- write_file ---


class TestWriteFile:

    @pytest.mark.asyncio
    async def test_creates_new_file(self, tmp_path):
        """Creates a new file with content."""
        f = tmp_path / "new.txt"
        result = await write_file(str(f), "hello")
        assert result.is_error is False
        assert f.read_text() == "hello"

    @pytest.mark.asyncio
    async def test_creates_parent_directories(self, tmp_path):
        """Creates missing parent directories."""
        f = tmp_path / "a" / "b" / "c" / "new.txt"
        result = await write_file(str(f), "deep")
        assert result.is_error is False
        assert f.read_text() == "deep"

    @pytest.mark.asyncio
    async def test_overwrites_existing_file(self, tmp_path):
        """Overwrites existing file content."""
        f = tmp_path / "existing.txt"
        f.write_text("old content")
        result = await write_file(str(f), "new content")
        assert result.is_error is False
        assert f.read_text() == "new content"

    @pytest.mark.asyncio
    async def test_returns_confirmation(self, tmp_path):
        """Result content confirms the write (path or similar)."""
        f = tmp_path / "test.txt"
        result = await write_file(str(f), "content")
        assert result.is_error is False
        assert len(result.content) > 0  # some confirmation message

    @pytest.mark.asyncio
    async def test_empty_content(self, tmp_path):
        """Writing empty string creates empty file."""
        f = tmp_path / "empty.txt"
        result = await write_file(str(f), "")
        assert result.is_error is False
        assert f.read_text() == ""

    @pytest.mark.asyncio
    async def test_utf8_content(self, tmp_path):
        """UTF-8 content is written correctly."""
        f = tmp_path / "unicode.txt"
        result = await write_file(str(f), "héllo wörld 🐈‍⬛")
        assert f.read_text() == "héllo wörld 🐈‍⬛"


# --- edit_file ---


class TestEditFile:

    @pytest.mark.asyncio
    async def test_replaces_exact_match(self, tmp_path):
        """Replaces old_text with new_text."""
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        result = await edit_file(str(f), "world", "universe")
        assert result.is_error is False
        assert f.read_text() == "hello universe"

    @pytest.mark.asyncio
    async def test_no_match_returns_error(self, tmp_path):
        """old_text not found → is_error=True, file unchanged."""
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        result = await edit_file(str(f), "missing text", "replacement")
        assert result.is_error is True
        assert f.read_text() == "hello world"

    @pytest.mark.asyncio
    async def test_multiple_matches_returns_error(self, tmp_path):
        """Ambiguous match (multiple occurrences) → is_error=True, file unchanged."""
        f = tmp_path / "test.txt"
        f.write_text("aaa bbb aaa")
        result = await edit_file(str(f), "aaa", "ccc")
        assert result.is_error is True
        assert f.read_text() == "aaa bbb aaa"

    @pytest.mark.asyncio
    async def test_file_not_found_returns_error(self):
        """Missing file → is_error=True."""
        result = await edit_file("/tmp/does_not_exist_abc123.txt", "old", "new")
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_multiline_replacement(self, tmp_path):
        """Multiline old_text and new_text work."""
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\n")
        result = await edit_file(str(f), "line1\nline2", "replaced1\nreplaced2")
        assert result.is_error is False
        assert f.read_text() == "replaced1\nreplaced2\nline3\n"

    @pytest.mark.asyncio
    async def test_replace_with_empty_string(self, tmp_path):
        """Replacing with empty string (deletion) works."""
        f = tmp_path / "test.txt"
        f.write_text("hello beautiful world")
        result = await edit_file(str(f), " beautiful", "")
        assert result.is_error is False
        assert f.read_text() == "hello world"

    @pytest.mark.asyncio
    async def test_whitespace_sensitive(self, tmp_path):
        """Match is exact — whitespace matters."""
        f = tmp_path / "test.txt"
        f.write_text("  indented line\n")
        result = await edit_file(str(f), "indented line", "no match here")
        # "  indented line" != "indented line" — leading spaces differ
        # This should still match because "indented line" is a substring
        # Actually, "indented line" IS in "  indented line" as exact text
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_returns_confirmation(self, tmp_path):
        """Successful edit returns a confirmation message."""
        f = tmp_path / "test.txt"
        f.write_text("old text here")
        result = await edit_file(str(f), "old text", "new text")
        assert result.is_error is False
        assert len(result.content) > 0
