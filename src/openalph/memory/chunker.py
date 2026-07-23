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
        # BUG-8: iterate over the ### boundaries only. The previous version
        # appended a `len(lines)` sentinel to `boundaries` AND kept a separate
        # "handle last segment" block below -- so the final subsection was
        # emitted TWICE (once when the loop reached the sentinel, once by the
        # trailing block, whose `prev` still pointed at the last ###). Both
        # copies got distinct sha256 ids, were embedded and inserted -- doubling
        # that chunk's embedding cost and skewing BM25 corpus stats -- while the
        # search-side dedup hid it from query output (mis-attributing the cause
        # to "re-index churn"). The sentinel is gone; the trailing block alone
        # closes the final segment, emitted exactly once.
        prev = 0
        for idx in h3_indices:
            if idx > prev:
                chunk_text = "\n".join(lines[prev:idx]).rstrip()
                if chunk_text:
                    start = base_start_line + prev
                    end = base_start_line + idx - 1
                    chunks.append((start, end, chunk_text))
            prev = idx

        # Final segment: from the last ### to the end (or from 0 if the section
        # opened with content before any ###).
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
    """Split text on blank-line paragraph boundaries, tracking real offsets.

    BUG-15: the previous version advanced the running line number by
    `part.count("\n") + 2`, hard-coding a two-newline separator even though it
    split on `\n\n+`. Any separator with 3+ newlines (and the leading blank
    lines that `.strip()` discarded) shifted every subsequent chunk's recorded
    start/end -- and those numbers are surfaced to the agent as `path:start-end`
    citations, so a drifted citation points the operator at the wrong lines.
    Line numbers are now derived from each paragraph's actual character span via
    `re.finditer`, so any separator width is exact.
    """
    # Walk the non-empty segments between blank-line separators, recovering
    # each paragraph's exact character offset (and thus line number).
    chunks: list[tuple[int, int, str]] = []
    pos = 0
    seps = list(re.finditer(r"\n\n+", text))
    boundaries = [(m.start(), m.end()) for m in seps]
    segments: list[tuple[int, int]] = []
    for start, end in boundaries:
        segments.append((pos, start))
        pos = end
    segments.append((pos, len(text)))

    if len([seg for seg in segments if text[seg[0]:seg[1]].strip()]) <= 1:
        end_line = base_start_line + text.count("\n")
        return [(base_start_line, end_line, text.strip() or text)]

    for seg_start, seg_end in segments:
        raw = text[seg_start:seg_end]
        part = raw.strip()
        if not part:
            continue
        # Offset of the stripped paragraph within `text`, in lines.
        lead_ws = len(raw) - len(raw.lstrip())
        start_off = seg_start + lead_ws
        start_line = base_start_line + text.count("\n", 0, start_off)
        end_line = start_line + part.count("\n")
        chunks.append((start_line, end_line, part))

    return chunks


def _split_plain_text(text: str) -> list[tuple[int, str]]:
    """Split non-markdown text on paragraph boundaries, then fixed windows."""
    # BUG-15: derive line numbers from real character spans rather than
    # assuming a 2-newline separator (see `_split_paragraphs`).
    seps = list(re.finditer(r"\n\n+", text))
    if seps:
        sections = []
        pos = 0
        spans = [(m.start(), m.end()) for m in seps]
        segments = []
        for start, end in spans:
            segments.append((pos, start))
            pos = end
        segments.append((pos, len(text)))
        for seg_start, seg_end in segments:
            raw = text[seg_start:seg_end]
            para = raw.strip()
            if not para:
                continue
            lead_ws = len(raw) - len(raw.lstrip())
            start_off = seg_start + lead_ws
            start_line = text.count("\n", 0, start_off) + 1  # 1-indexed
            sections.append((start_line, para))
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
