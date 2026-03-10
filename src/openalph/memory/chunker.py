"""Markdown-aware text chunking for memory search indexing."""

import re
from dataclasses import dataclass

# Thresholds
MAX_SECTION_CHARS = 1500
MIN_CHUNK_CHARS = 50
FIXED_WINDOW_SIZE = 1000
FIXED_WINDOW_OVERLAP = 200

H2_RE = re.compile(r"^## ", re.MULTILINE)
H3_RE = re.compile(r"^### ", re.MULTILINE)


@dataclass
class Chunk:
    path: str
    start_line: int
    end_line: int
    text: str


def chunk_file(text: str, path: str) -> list[Chunk]:
    """Split text into chunks for indexing.

    Markdown files: split on ## headers, with size guards.
    Non-markdown: paragraph splitting, then fixed windows.
    """
    if not text or not text.strip():
        return []

    is_markdown = path.lower().endswith(".md")

    if is_markdown:
        sections = _split_on_h2(text)
    else:
        sections = _split_plain_text(text)

    # Apply size guard and minimum filter
    result = []
    for start_line, section_text in sections:
        if len(section_text) > MAX_SECTION_CHARS and is_markdown:
            sub_chunks = _split_large_section(section_text, start_line)
        else:
            end_line = start_line + section_text.count("\n")
            sub_chunks = [(start_line, end_line, section_text)]

        for s, e, t in sub_chunks:
            if len(t) >= MIN_CHUNK_CHARS:
                result.append(Chunk(path=path, start_line=s, end_line=e, text=t))

    return result


def _split_on_h2(text: str) -> list[tuple[int, str]]:
    """Split markdown on ## headers. Returns (start_line, text) pairs."""
    lines = text.split("\n")
    sections: list[tuple[int, str]] = []
    current_start = 0
    current_lines: list[str] = []

    for i, line in enumerate(lines):
        if H2_RE.match(line) and current_lines:
            # Emit previous section
            section_text = "\n".join(current_lines).rstrip()
            if section_text:
                sections.append((current_start + 1, section_text))  # 1-indexed
            current_lines = [line]
            current_start = i
        else:
            current_lines.append(line)

    # Emit last section
    if current_lines:
        section_text = "\n".join(current_lines).rstrip()
        if section_text:
            sections.append((current_start + 1, section_text))

    return sections


def _split_large_section(text: str, base_start_line: int) -> list[tuple[int, int, str]]:
    """Split an oversized section on ### headers, then paragraphs."""
    # Try splitting on ### headers first
    lines = text.split("\n")
    h3_indices = [i for i, line in enumerate(lines) if H3_RE.match(line)]

    if h3_indices:
        chunks = []
        # If there's content before the first ###, include it
        boundaries = h3_indices + [len(lines)]
        prev = 0
        for idx in boundaries:
            if idx == prev and idx in h3_indices:
                prev = idx
                continue
            if idx > prev:
                chunk_text = "\n".join(lines[prev:idx]).rstrip()
                if chunk_text:
                    start = base_start_line + prev
                    end = base_start_line + idx - 1
                    chunks.append((start, end, chunk_text))
            if idx in h3_indices:
                prev = idx

        # Handle last segment
        if prev < len(lines):
            chunk_text = "\n".join(lines[prev:]).rstrip()
            if chunk_text:
                start = base_start_line + prev
                end = base_start_line + len(lines) - 1
                chunks.append((start, end, chunk_text))

        if len(chunks) > 1:
            return chunks

    # Fall back to paragraph splitting
    paragraphs = _split_paragraphs(text, base_start_line)
    if len(paragraphs) > 1:
        return paragraphs

    # Last resort: fixed windows
    return _fixed_windows(text, base_start_line)


def _split_paragraphs(text: str, base_start_line: int) -> list[tuple[int, int, str]]:
    """Split text on double-newline paragraph boundaries."""
    parts = re.split(r"\n\n+", text)
    if len(parts) <= 1:
        end_line = base_start_line + text.count("\n")
        return [(base_start_line, end_line, text)]

    chunks = []
    current_line = base_start_line
    for part in parts:
        part = part.strip()
        if not part:
            continue
        line_count = part.count("\n")
        chunks.append((current_line, current_line + line_count, part))
        # +2 for the double newline separator
        current_line += line_count + 2

    return chunks


def _split_plain_text(text: str) -> list[tuple[int, str]]:
    """Split non-markdown text on paragraph boundaries, then fixed windows."""
    paragraphs = re.split(r"\n\n+", text)
    if len(paragraphs) > 1:
        sections = []
        current_line = 0
        for para in paragraphs:
            para = para.strip()
            if para:
                sections.append((current_line + 1, para))  # 1-indexed
            current_line += para.count("\n") + 2  # +2 for separator
        return sections

    # No paragraph breaks — use fixed windows
    chunks = []
    start = 0
    while start < len(text):
        end = start + FIXED_WINDOW_SIZE
        chunk = text[start:end]
        line_offset = text[:start].count("\n")
        chunks.append((line_offset + 1, chunk))
        start += FIXED_WINDOW_SIZE - FIXED_WINDOW_OVERLAP

    return chunks


def _fixed_windows(text: str, base_start_line: int) -> list[tuple[int, int, str]]:
    """Split text into fixed-size overlapping windows."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + FIXED_WINDOW_SIZE
        chunk = text[start:end]
        char_offset_lines = text[:start].count("\n")
        chunk_lines = chunk.count("\n")
        s = base_start_line + char_offset_lines
        e = s + chunk_lines
        chunks.append((s, e, chunk))
        start += FIXED_WINDOW_SIZE - FIXED_WINDOW_OVERLAP

    return chunks
