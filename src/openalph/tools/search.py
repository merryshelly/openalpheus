"""grep and glob tools: bounded, pure-stdlib file search over a workspace.

Component D (grep) + Component E (glob), design §6/§7, invariant V3
("Bounded returns"). Both tools share a single walker (module-private
``_walk``) so the skip-list, symlink policy, and scan-count bound are
defined exactly once.

Walker contract (V3):
  * ``os.walk(followlinks=False)`` — never descends into a symlinked
    directory (avoids symlink-cycle infinite loops).
  * Directory names in ``SKIP_DIRS`` are pruned from the walk entirely —
    never descended into, never returned as a candidate (grep or glob).
  * Hard bound ``max_scan_files`` on the total number of entries (files +
    directories combined) visited in one call; hitting it stops the walk
    early and adds a footer note steering the caller to narrow path/glob.
  * Deterministic final ordering: mtime desc, path asc tiebreak.

grep additionally applies a per-file content-readability filter (binary
detection via ``_is_binary``, and a ``max_file_bytes`` size cap) before
reading a candidate file's lines — glob never reads file content at all
(it is a name/listing search, "ls replacement"), so that filter does not
apply to it.
"""

import fnmatch
import os
import re
from dataclasses import dataclass
from pathlib import Path

from openalph.tools import ToolResult

from .file import _is_binary

# Directory names pruned from the walk entirely (design §6, shared by
# grep + glob). Never descended into; never returned as a match.
SKIP_DIRS = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".memory-index", ".cache",
})

# Config defaults (overridable per-call via tool_config / TOML [config]).
DEFAULT_MAX_SCAN_FILES = 10000
DEFAULT_MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MiB = 5242880

# Exact overflow steering text (design §6/§7, V3). The em-dash is literal —
# do not "fix" it to a hyphen; tests pin these exact bytes.
_OVERFLOW = ('[truncated: showing first {n} of {m} — '
             'narrow with path/glob, or use output_mode="count"]')

# Shared "the walk itself was cut short" footer (distinct from the
# head_limit/output overflow steering above — this one means the SCAN
# stopped early, not that results were merely trimmed for display).
_SCAN_BOUND_NOTE = (
    "(scan limit of {max_scan_files} files reached before the walk "
    "completed — results may be incomplete; narrow your path/glob to "
    "scan fewer files)"
)


@dataclass
class _Entry:
    """One walked filesystem entry (file or directory)."""
    path: Path
    is_dir: bool
    mtime: float
    size: int


def _resolve_root(path: str | None, workspace: Path) -> Path:
    """Resolve the search root.

    ``execute_tool``'s own path-resolution tuple already workspace-joins a
    relative ``path`` before this module ever sees it (grep/glob are added
    to that tuple, not the read-registry tuple — a match is not a file
    read). This second resolution is defensive only, so these functions
    behave correctly if ever called directly (e.g. future direct/unit use)
    bypassing that dispatcher step.
    """
    workspace = Path(workspace)
    if path is None:
        return workspace
    root = Path(path)
    if not root.is_absolute():
        root = workspace / root
    return root


def _display_path(p: Path, workspace: Path) -> str:
    """Render a path workspace-relative when possible, else absolute (V7)."""
    try:
        return p.resolve().relative_to(Path(workspace).resolve()).as_posix()
    except Exception:
        return str(p)


def _walk(root: Path, max_scan_files: int) -> tuple[list[_Entry], bool]:
    """Walk ``root`` (no symlink following), pruning SKIP_DIRS, collecting
    up to ``max_scan_files`` total entries (files and directories combined).

    Returns ``(entries, bound_hit)`` where ``bound_hit`` is True if the walk
    was stopped early because the bound was reached (V3 footer steering).
    Per-directory names are processed in sorted order so an early stop
    always drops the same trailing entries call-to-call (determinism).
    """
    entries: list[_Entry] = []
    bound_hit = False

    for dirpath, dirnames, filenames in os.walk(str(root), topdown=True, followlinks=False):
        # Prune in place: removes skip-list dirs from further traversal AND
        # from being emitted as directory candidates themselves.
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        dirpath_p = Path(dirpath)

        for d in dirnames:
            if len(entries) >= max_scan_files:
                bound_hit = True
                break
            p = dirpath_p / d
            try:
                st = p.stat()
            except OSError:
                continue
            entries.append(_Entry(path=p, is_dir=True, mtime=st.st_mtime, size=0))
        if bound_hit:
            break

        for f in sorted(filenames):
            if len(entries) >= max_scan_files:
                bound_hit = True
                break
            p = dirpath_p / f
            try:
                st = p.stat()
            except OSError:
                continue
            entries.append(_Entry(path=p, is_dir=False, mtime=st.st_mtime, size=st.st_size))
        if bound_hit:
            break

    return entries, bound_hit


# ---------------------------------------------------------------------------
# grep (Component D, design §6)
# ---------------------------------------------------------------------------

async def run_grep(
    pattern: str,
    path: str | None,
    glob: str | None,
    output_mode: str,
    head_limit: int | None,
    case_insensitive: bool,
    config: dict,
    workspace: Path,
) -> ToolResult:
    """Bounded, pure-Python line-oriented regex search.

    Args:
        pattern: Python ``re`` regex, matched against each line individually.
        path: File or directory root to search (optional; default workspace).
        glob: Optional fnmatch filename filter (e.g. ``"*.py"``) narrowing
            which files are searched.
        output_mode: "files_with_matches" (default) | "content" | "count".
        head_limit: Cap on returned entries (mode-specific default applies
            when None: 50 / 100 / 50 respectively).
        case_insensitive: Match case-insensitively when True (default False).
        config: Tool config dict (``max_scan_files``, ``max_file_bytes``;
            sane module defaults apply for any missing key).
        workspace: Agent workspace root, for default rooting + display.

    Returns:
        ToolResult. Zero matches is a hint, not an error. Invalid regex is
        an error naming the ``re.error`` message plus Python-re steering.
    """
    try:
        workspace = Path(workspace)
        max_scan_files = config.get("max_scan_files", DEFAULT_MAX_SCAN_FILES)
        max_file_bytes = config.get("max_file_bytes", DEFAULT_MAX_FILE_BYTES)

        root = _resolve_root(path, workspace)
        if not root.exists():
            return ToolResult(
                content=f"Error: path does not exist: {root}",
                is_error=True,
            )

        try:
            flags = re.IGNORECASE if case_insensitive else 0
            rx = re.compile(pattern, flags)
        except re.error as e:
            return ToolResult(
                content=(
                    f"Error: invalid regex pattern {pattern!r}: {e}. "
                    "Note: pattern uses Python re syntax (not shell globs "
                    "or PCRE) — check backslash-escaping and special "
                    "characters, then retry."
                ),
                is_error=True,
            )

        # Candidate file set. grep's `path` may name a single file directly
        # (design §6 schema: "file or directory root") — bypass the walker
        # in that case rather than treating it as a (non-existent) subtree.
        bound_hit = False
        if root.is_file():
            candidates = [root]
        else:
            entries, bound_hit = _walk(root, max_scan_files)
            candidates = [e.path for e in entries if not e.is_dir]
            if glob:
                candidates = [p for p in candidates if fnmatch.fnmatch(p.name, glob)]

        skipped = 0
        # (path, mtime, [(lineno, line_text), ...]) per file with >=1 match.
        file_matches: list[tuple[Path, float, list[tuple[int, str]]]] = []

        for p in candidates:
            try:
                st = p.stat()
            except OSError:
                continue
            if st.st_size > max_file_bytes:
                skipped += 1
                continue
            if _is_binary(str(p)):
                skipped += 1
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    text = f.read()
            except (UnicodeDecodeError, OSError):
                skipped += 1
                continue

            hits: list[tuple[int, str]] = []
            for lineno, line in enumerate(text.splitlines(), start=1):
                if rx.search(line) is not None:
                    hits.append((lineno, line))
            if hits:
                file_matches.append((p, st.st_mtime, hits))

        # Deterministic order: mtime desc, path asc tiebreak (V3).
        file_matches.sort(key=lambda t: (-t[1], _display_path(t[0], workspace)))

        if not file_matches:
            return ToolResult(
                content=(
                    f"No matches for pattern {pattern!r}. Try broadening "
                    "the search, check the path/glob filters, or verify "
                    "the regex is correct (Python re syntax)."
                ),
                is_error=False,
            )

        footer_parts: list[str] = []

        if output_mode == "content":
            limit = head_limit if head_limit is not None else 100
            rows: list[str] = []
            for p, _mtime, hits in file_matches:
                disp = _display_path(p, workspace)
                for lineno, line in hits:
                    rows.append(f"{disp}:{lineno}: {line[:500]}")
            total = len(rows)
            body = "\n".join(rows[:limit])
            if total > limit:
                footer_parts.append(_OVERFLOW.format(n=limit, m=total))

        elif output_mode == "count":
            limit = head_limit if head_limit is not None else 50
            per_file = [(p, len(hits)) for p, _mtime, hits in file_matches]
            total_matches = sum(c for _p, c in per_file)
            shown = per_file[:limit]
            lines = [f"{_display_path(p, workspace)}: {c}" for p, c in shown]
            lines.append(f"total: {total_matches}")
            body = "\n".join(lines)
            if len(per_file) > limit:
                footer_parts.append(_OVERFLOW.format(n=limit, m=len(per_file)))

        else:  # "files_with_matches" (default)
            limit = head_limit if head_limit is not None else 50
            paths = [p for p, _mtime, _hits in file_matches]
            total = len(paths)
            body = "\n".join(_display_path(p, workspace) for p in paths[:limit])
            if total > limit:
                footer_parts.append(_OVERFLOW.format(n=limit, m=total))

        if skipped > 0:
            footer_parts.append(
                f"({skipped} file(s) skipped: binary or exceeds size limit)"
            )
        if bound_hit:
            footer_parts.append(_SCAN_BOUND_NOTE.format(max_scan_files=max_scan_files))

        content = body
        if footer_parts:
            content = content + "\n\n" + "\n".join(footer_parts)

        return ToolResult(content=content, is_error=False)

    except Exception as e:
        return ToolResult(content=f"Error running grep: {e}", is_error=True)


# ---------------------------------------------------------------------------
# glob (Component E, design §7)
# ---------------------------------------------------------------------------

def _translate_glob_segment(seg: str) -> str:
    """Translate ONE path segment's shell-wildcard syntax (``*``, ``?``,
    ``[seq]``/``[!seq]``) to a regex fragment matching within a single
    segment (never crosses ``/``). The segment must not itself be ``*`` or
    ``**`` (the caller special-cases those for cross-segment semantics).

    Hand-rolled rather than ``glob.translate`` (stdlib 3.13+ only — this
    project targets >=3.11 per pyproject.toml) or ``fnmatch``'s private
    ``_translate`` helper, so it works on any supported Python version
    without depending on private API surface.
    """
    out = []
    i, n = 0, len(seg)
    while i < n:
        c = seg[i]
        if c == '*':
            out.append('[^/]*')
            i += 1
        elif c == '?':
            out.append('[^/]')
            i += 1
        elif c == '[':
            j = i + 1
            if j < n and seg[j] in ('!', '^'):
                j += 1
            if j < n and seg[j] == ']':
                j += 1
            while j < n and seg[j] != ']':
                j += 1
            if j >= n:
                # No closing bracket found -- treat '[' as a literal.
                out.append(re.escape('['))
                i += 1
            else:
                inner = seg[i + 1:j]
                if inner.startswith('!'):
                    inner = '^' + inner[1:]
                inner = inner.replace('\\', '\\\\')
                out.append('[' + inner + ']')
                i = j + 1
        else:
            out.append(re.escape(c))
            i += 1
    return ''.join(out)


def _glob_pattern_to_regex(pattern: str) -> re.Pattern:
    """Translate a pathlib-style glob pattern (including ``**``) into a
    regex matching a ``/``-joined RELATIVE path string (no leading or
    trailing slash) — mirrors ``pathlib.Path.glob`` semantics exactly
    (validated against it directly; dotfiles are matched by ``*`` with no
    bash-style hidden-file exclusion, matching pathlib's own default).

    A trailing ``**`` also matches its own prefix with zero additional
    segments (``"dir/**"`` matches ``"dir"`` itself, as pathlib does).
    Consecutive ``**`` segments collapse to a single "any segments" group.
    """
    segments = pattern.split('/')
    n = len(segments)
    last_idx = n - 1

    one_last_segment = '[^/]+'
    one_segment = one_last_segment + '/'
    any_segments = f'(?:{one_segment})*'
    any_last_segments = f'{any_segments}(?:{one_last_segment})?'

    parts: list[str] = []
    emitted_real = False  # True once a non-collapsed, non-"**" segment (or
                          # a non-collapsed "**" group) has been emitted.
    i = 0
    while i < n:
        seg = segments[i]
        is_last = (i == last_idx)
        if seg == '**':
            if is_last:
                if not emitted_real:
                    parts.append(any_last_segments)
                else:
                    # Trailing "**" after a real prefix also matches the
                    # prefix itself (pathlib: "dir/**" includes "dir").
                    parts.append(f'(?:/{any_last_segments})?')
            else:
                nxt = segments[i + 1]
                if nxt != '**':
                    parts.append(any_segments)
                    emitted_real = True
                # else: consecutive "**" collapse -- this one contributes
                # nothing and does not itself count as "emitted_real".
            i += 1
        elif seg == '*':
            parts.append(one_last_segment if is_last else one_segment)
            emitted_real = True
            i += 1
        else:
            parts.append(_translate_glob_segment(seg))
            emitted_real = True
            if not is_last:
                nxt = segments[i + 1]
                if nxt == '**' and (i + 1) == last_idx:
                    pass  # separator folded into the trailing "**" tail group
                else:
                    parts.append('/')
            i += 1

    return re.compile(r'\A(?:' + ''.join(parts) + r')\Z', re.DOTALL)


async def run_glob(
    pattern: str,
    path: str | None,
    head_limit: int | None,
    config: dict,
    workspace: Path,
) -> ToolResult:
    """Bounded pathlib-style file/directory name search ("ls replacement").

    Args:
        pattern: Glob pattern (pathlib syntax, including ``**``), matched
            against paths relative to the search root.
        path: Directory root to search (optional; default workspace).
        head_limit: Cap on returned entries (default 100 when None).
        config: Tool config dict (``max_scan_files``; ``max_file_bytes`` is
            accepted for schema symmetry with grep but unused here — glob
            never reads file content, so size/binary skip does not apply).
        workspace: Agent workspace root, for default rooting + display.

    Returns:
        ToolResult listing matched paths, directories marked with a
        trailing "/", sorted newest-first (mtime desc, path asc tiebreak).
        Zero matches is a hint, not an error. A nonexistent root is an error.
    """
    try:
        workspace = Path(workspace)
        max_scan_files = config.get("max_scan_files", DEFAULT_MAX_SCAN_FILES)

        root = _resolve_root(path, workspace)
        if not root.exists():
            return ToolResult(
                content=f"Error: path does not exist: {root}",
                is_error=True,
            )
        if not root.is_dir():
            return ToolResult(
                content=f"Error: path is not a directory: {root}",
                is_error=True,
            )

        try:
            rx = _glob_pattern_to_regex(pattern)
        except Exception as e:
            return ToolResult(
                content=f"Error: invalid glob pattern {pattern!r}: {e}",
                is_error=True,
            )

        entries, bound_hit = _walk(root, max_scan_files)

        matched: list[_Entry] = []
        for e in entries:
            rel = e.path.relative_to(root).as_posix()
            if rx.match(rel) is not None:
                matched.append(e)

        # Deterministic order: mtime desc, path asc tiebreak (V3).
        matched.sort(key=lambda e: (-e.mtime, _display_path(e.path, workspace)))

        if not matched:
            return ToolResult(
                content=(
                    f"No matches for pattern {pattern!r} under {root}. "
                    "Try a broader pattern or a different path root."
                ),
                is_error=False,
            )

        limit = head_limit if head_limit is not None else 100
        total = len(matched)
        shown = matched[:limit]
        lines = [
            _display_path(e.path, workspace) + ("/" if e.is_dir else "")
            for e in shown
        ]
        body = "\n".join(lines)

        footer_parts: list[str] = []
        if total > limit:
            footer_parts.append(_OVERFLOW.format(n=limit, m=total))
        if bound_hit:
            footer_parts.append(_SCAN_BOUND_NOTE.format(max_scan_files=max_scan_files))

        content = body
        if footer_parts:
            content = content + "\n\n" + "\n".join(footer_parts)

        return ToolResult(content=content, is_error=False)

    except Exception as e:
        return ToolResult(content=f"Error running glob: {e}", is_error=True)
