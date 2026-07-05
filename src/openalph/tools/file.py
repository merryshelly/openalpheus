"""File operations: read, write, edit, patch."""

import os
import re
from pathlib import Path

from openalph.tools import ToolResult


def _is_binary(filepath: str) -> bool:
    """Check if a file is binary by reading the first chunk."""
    try:
        with open(filepath, 'rb') as f:
            chunk = f.read(8192)
            if not chunk:
                return False
            # Check for null bytes or high ratio of non-printable chars
            if b'\x00' in chunk:
                return True
            # Use a simple heuristic: if >30% non-printable, consider binary
            non_printable = sum(1 for b in chunk if b < 32 and b not in (9, 10, 13))
            return non_printable / len(chunk) > 0.3
    except Exception:
        return False


async def read_file(path: str, offset: int | None = None, limit: int | None = None) -> ToolResult:
    """Read file contents.
    
    Args:
        path: Path to the file
        offset: 1-indexed line number to start from
        limit: Maximum number of lines to return
    
    Returns:
        ToolResult with content or error
    """
    try:
        if not os.path.exists(path):
            return ToolResult(content=f"Error: File not found: {path}", is_error=True)
        
        if not os.path.isfile(path):
            return ToolResult(content=f"Error: Path is not a file: {path}", is_error=True)
        
        if _is_binary(path):
            return ToolResult(content=f"Error: Binary file cannot be read: {path}", is_error=True)
        
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Handle offset (1-indexed)
        if offset is not None:
            if offset < 1:
                return ToolResult(content=f"Error: offset must be >= 1", is_error=True)
            start_idx = offset - 1
        else:
            start_idx = 0
        
        # Handle limit
        if limit is not None:
            if limit < 0:
                return ToolResult(content=f"Error: limit must be >= 0", is_error=True)
            end_idx = start_idx + limit
        else:
            end_idx = len(lines)
        
        # Slice the lines
        selected_lines = lines[start_idx:end_idx]
        content = ''.join(selected_lines)
        
        return ToolResult(content=content, is_error=False)
    
    except UnicodeDecodeError:
        return ToolResult(content=f"Error: Binary file cannot be read: {path}", is_error=True)
    except Exception as e:
        return ToolResult(content=f"Error reading file: {e}", is_error=True)


async def write_file(path: str, content: str, tool_config: dict | None = None) -> ToolResult:
    """Write content to a file, creating parent directories if needed.

    The actual disk write is routed through the shared validate-on-edit
    seam (``_validated_write``, tools/validate.py, Component C) so that
    known code types are syntax-checked before a clean file is allowed to
    become broken. See _validated_write's docstring for the full decision
    matrix; a rejection returns its ToolResult as-is (file left untouched),
    a warning/fail-open note is appended to this function's own success
    message.

    Args:
        path: Path to the file
        content: Content to write
        tool_config: Tool config dict (may be None); validate_on_edit
            (default True) is the validation kill-switch.

    Returns:
        ToolResult with confirmation or error
    """
    try:
        # Create parent directories
        parent = Path(path).parent
        if parent:
            parent.mkdir(parents=True, exist_ok=True)

        from .validate import _validated_write
        vresult = await _validated_write(path, content, tool_config)
        if vresult.is_error:
            return vresult

        message = f"Successfully wrote to {path}"
        if vresult.content:
            message = f"{message} {vresult.content}"
        return ToolResult(content=message, is_error=False)

    except Exception as e:
        return ToolResult(content=f"Error writing file: {e}", is_error=True)


async def edit_file(
    path: str, old_text: str, new_text: str, replace_all: bool = False,
    tool_config: dict | None = None,
) -> ToolResult:
    """Replace occurrence(s) of old_text with new_text.
    
    Default (replace_all=False): requires exactly one match. 0 or 2+ matches
    results in error. File is unchanged on error.
    replace_all=True: 0 matches still errors (same steering message); 1 or
    more matches are ALL replaced, and the result reports the count.

    The actual disk write (either branch) is routed through the shared
    validate-on-edit seam (``_validated_write``, tools/validate.py,
    Component C) — see its docstring for the full decision matrix. A
    validation rejection returns that ToolResult as-is (file untouched).

    Args:
        path: Path to the file
        old_text: Text to find
        new_text: Text to replace with
        replace_all: If True, replace every occurrence instead of requiring
            exactly one (use for renaming a symbol/string across the file)
        tool_config: Tool config dict (may be None); validate_on_edit
            (default True) is the validation kill-switch.
    
    Returns:
        ToolResult with confirmation or error
    """
    try:
        if not os.path.exists(path):
            return ToolResult(content=f"Error: File not found: {path}", is_error=True)
        
        if not os.path.isfile(path):
            return ToolResult(content=f"Error: Path is not a file: {path}", is_error=True)

        if _is_binary(path):
            return ToolResult(content=f"Error: Binary file cannot be edited: {path}", is_error=True)
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count occurrences
        count = content.count(old_text)
        
        if count == 0:
            return ToolResult(
                content=(
                    f"Error: old_text not found in {path}. "
                    "Read the file first — its content may differ from what you expect. "
                    "Check spacing, indentation, and line endings exactly."
                ),
                is_error=True,
            )
        
        if not replace_all and count > 1:
            return ToolResult(
                content=(
                    f"Error: old_text appears {count} times in {path} (ambiguous). "
                    "Add more surrounding context lines to old_text to disambiguate "
                    "and uniquely identify the target occurrence."
                ),
                is_error=True,
            )

        from .validate import _validated_write

        if replace_all:
            # Replace every occurrence (count already established as >= 1 above).
            new_content = content.replace(old_text, new_text)

            vresult = await _validated_write(path, new_content, tool_config)
            if vresult.is_error:
                return vresult

            message = f"Replaced {count} occurrence(s) in {path}"
            if vresult.content:
                message = f"{message} {vresult.content}"
            return ToolResult(content=message, is_error=False)

        # Exactly one match - perform replacement
        new_content = content.replace(old_text, new_text, 1)

        vresult = await _validated_write(path, new_content, tool_config)
        if vresult.is_error:
            return vresult

        message = f"Successfully edited {path}"
        if vresult.content:
            message = f"{message} {vresult.content}"
        return ToolResult(content=message, is_error=False)
    
    except Exception as e:
        return ToolResult(content=f"Error editing file: {e}", is_error=True)


# ---------------------------------------------------------------------------
# file_patch: multi-hunk SEARCH/REPLACE editing (Aider diff-fence syntax)
# ---------------------------------------------------------------------------

# Fence regexes (line-anchored, case-sensitive keywords, trailing-whitespace
# tolerant). Widths 5-9 accepted, matching the design's fence-marker range.
_SEARCH_FENCE_RE = re.compile(r'^<{5,9} SEARCH\s*$')
_DIVIDER_FENCE_RE = re.compile(r'^={5,9}\s*$')
_REPLACE_FENCE_RE = re.compile(r'^>{5,9} REPLACE\s*$')

_PATCH_FORMAT_EXAMPLE = (
    "Expected patch format (one or more hunks; text outside blocks is ignored):\n"
    "\n"
    "<<<<<<< SEARCH\n"
    "exact existing lines to find\n"
    "=======\n"
    "replacement lines\n"
    ">>>>>>> REPLACE\n"
    "\n"
    "Worked example — to change 'foo = 1' to 'foo = 2' in a file containing that line:\n"
    "\n"
    "<<<<<<< SEARCH\n"
    "foo = 1\n"
    "=======\n"
    "foo = 2\n"
    ">>>>>>> REPLACE\n"
    "\n"
    "Fence markers may repeat 5-9 times (e.g. <<<<< SEARCH ... <<<<<<<<< SEARCH); "
    "the SEARCH/REPLACE keywords are case-sensitive."
)


def _parse_patch_hunks(patch: str) -> tuple[list[tuple[str, str]] | None, str | None]:
    """Parse one or more SEARCH/REPLACE hunks out of a patch string.

    Walks the patch line-by-line through a simple three-state machine
    (outside -> search -> replace -> outside), collecting the lines
    between fences into a search/replace text pair per hunk. Lines outside
    any block (prose) are ignored, tolerating weak-model chatter around
    the fenced blocks.

    Args:
        patch: Raw patch text containing one or more fenced hunks

    Returns:
        (hunks, None) on success, where hunks is a non-empty list of
        (search_text, replace_text) tuples in patch order; or
        (None, error_message) if no complete hunk was found, or a hunk's
        SEARCH section is empty.
    """
    lines = patch.splitlines()
    hunks: list[tuple[str, str]] = []
    state = "outside"  # outside -> search -> replace -> outside
    search_lines: list[str] = []
    replace_lines: list[str] = []

    for line in lines:
        if state == "outside":
            if _SEARCH_FENCE_RE.match(line):
                state = "search"
                search_lines = []
            # else: prose outside a block — ignored (weak-model tolerance)
        elif state == "search":
            if _DIVIDER_FENCE_RE.match(line):
                state = "replace"
                replace_lines = []
            else:
                search_lines.append(line)
        elif state == "replace":
            if _REPLACE_FENCE_RE.match(line):
                search_text = "\n".join(search_lines)
                replace_text = "\n".join(replace_lines)
                if search_text == "":
                    return None, (
                        "Error: SEARCH must not be empty — use file_write to create new files. "
                        "file_patch requires exact existing text to match and replace; it "
                        "cannot be used to create brand-new content from nothing."
                    )
                hunks.append((search_text, replace_text))
                state = "outside"
            else:
                replace_lines.append(line)
        # (a block left open at end-of-input — never reaching its closing
        # fence — is simply dropped; it contributes to a "zero blocks"
        # outcome only if no other hunk in the patch parsed successfully)

    if not hunks:
        return None, (
            "Error: no valid SEARCH/REPLACE blocks found in patch.\n\n" + _PATCH_FORMAT_EXAMPLE
        )

    return hunks, None


async def patch_file(path: str, patch: str, tool_config: dict | None = None) -> ToolResult:
    """Apply one or more SEARCH/REPLACE hunks to an existing file.

    Parses fenced hunks (Aider diff-fence syntax; see _PATCH_FORMAT_EXAMPLE),
    then applies them SEQUENTIALLY against the evolving in-memory buffer —
    hunk N may match text produced by hunk N-1. Each hunk's SEARCH text must
    match exactly once in the buffer as it stands when that hunk is reached
    (0 matches or >1 matches is an error naming the hunk). ALL-OR-NOTHING:
    hunks are validated and applied only in memory; the file on disk is
    written once, only after every hunk has succeeded — any failure leaves
    the file byte-identical to before the call.

    The final disk write is routed through the shared validate-on-edit seam
    (``_validated_write``, tools/validate.py, Component C) — see its
    docstring for the full decision matrix. A validation rejection returns
    that ToolResult as-is (file untouched, byte-identical — same V1
    atomicity guarantee as a failed hunk match).

    Args:
        path: Path to the file to patch (must already exist; non-binary)
        patch: One or more SEARCH/REPLACE blocks (text outside blocks, e.g.
            conversational prose, is ignored)
        tool_config: Tool config dict (may be None); validate_on_edit
            (default True) is the validation kill-switch.

    Returns:
        ToolResult with "Applied N hunk(s) to <path>" on success, or an
        error (file untouched) naming the parse/apply failure
    """
    try:
        if not os.path.exists(path):
            return ToolResult(content=f"Error: File not found: {path}", is_error=True)

        if not os.path.isfile(path):
            return ToolResult(content=f"Error: Path is not a file: {path}", is_error=True)

        if _is_binary(path):
            return ToolResult(content=f"Error: Binary file cannot be patched: {path}", is_error=True)

        hunks, parse_error = _parse_patch_hunks(patch)
        if parse_error is not None:
            return ToolResult(content=parse_error, is_error=True)

        with open(path, 'r', encoding='utf-8') as f:
            buffer = f.read()

        # Apply hunks sequentially against the evolving buffer. Nothing is
        # written to disk until every hunk below has succeeded in memory —
        # this loop is the sole enforcement point for atomicity (V1).
        for i, (search, replace) in enumerate(hunks, start=1):
            count = buffer.count(search)
            if count == 0:
                return ToolResult(
                    content=(
                        f"Error: hunk {i} SEARCH not found in {path} "
                        "(checked against the file as it stands after any earlier hunks "
                        f"in this patch). First 80 chars of hunk {i} SEARCH: "
                        f"{search[:80]!r}. Read the file first — its content may differ "
                        "from what you expect. No changes were written (all-or-nothing)."
                    ),
                    is_error=True,
                )
            if count > 1:
                return ToolResult(
                    content=(
                        f"Error: hunk {i} SEARCH matches {count} times in {path} "
                        "(ambiguous). Add more surrounding context lines to hunk "
                        f"{i}'s SEARCH to uniquely identify the target occurrence. "
                        "No changes were written (all-or-nothing)."
                    ),
                    is_error=True,
                )
            buffer = buffer.replace(search, replace, 1)

        from .validate import _validated_write
        vresult = await _validated_write(path, buffer, tool_config)
        if vresult.is_error:
            return vresult

        message = f"Applied {len(hunks)} hunk(s) to {path}"
        if vresult.content:
            message = f"{message} {vresult.content}"
        return ToolResult(content=message, is_error=False)

    except Exception as e:
        return ToolResult(content=f"Error patching file: {e}", is_error=True)
