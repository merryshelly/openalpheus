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

import contextlib
import fnmatch
import logging
import os
import re
import signal
import time
from dataclasses import dataclass
from pathlib import Path

from openalph.tools import ToolResult

from .file import _is_binary

logger = logging.getLogger(__name__)

# Directory names pruned from the walk entirely (design §6, shared by
# grep + glob). Never descended into; never returned as a match.
SKIP_DIRS = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".memory-index", ".cache",
})

# Config defaults (overridable per-call via tool_config / TOML [config]).
DEFAULT_MAX_SCAN_FILES = 10000
DEFAULT_MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MiB = 5242880
DEFAULT_TIME_BUDGET_SECONDS = 10.0

# Regex input is capped per line so a single re call is bounded in principle;
# this is a secondary defense, NOT the primary ReDoS bound (see
# _time_budget_guard below) -- catastrophic backtracking is superlinear in
# input length, but even a short capped line can still blow the wall-clock
# budget, which is why every match is also wrapped by the deadline guard.
_MAX_REGEX_INPUT_CHARS = 10_000

# R11: valid grep output modes (schema + validation share this one set).
_VALID_OUTPUT_MODES = ("files_with_matches", "content", "count")

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

# Footer note for skipped symlink files (R7), parallel in shape to
# _SCAN_BOUND_NOTE / the binary-skip footer text.
_SYMLINK_SKIP_NOTE = "({n} symlink(s) skipped: not followed)"


class _TimeBudgetExceeded(Exception):
    """Raised internally when a scan exceeds its wall-clock time budget.

    Never escapes run_grep/run_glob directly -- always caught and turned
    into an is_error ToolResult naming the budget.
    """


def _alarm_handler(signum, frame):
    raise _TimeBudgetExceeded()


@contextlib.contextmanager
def _time_budget_guard(deadline: float):
    """Bound the wall-clock time of the code inside the ``with`` block.

    R1 (CRITICAL): a single catastrophic ``re`` match holds the GIL for its
    entire duration -- on slow/throttled hardware a single pathological
    line (e.g. ``(a+)+$`` on ~28 'a's) can take tens of seconds, which is
    LONGER than any reasonable per-file/per-line "check the clock between
    units" granularity could bound (the match itself is the unit that
    overruns). A between-units monotonic check (also present in the scan
    loops below, as a fast path and a defense for the no-SIGALRM case)
    therefore cannot be the only enforcement -- it only helps for future
    units, not the one currently backtracking.

    The one stdlib mechanism that CAN interrupt a single in-flight ``re``
    match is a signal: CPython's regex engine periodically checks for
    pending signals during backtracking and raises the handler's exception
    from inside the match call. This requires running on the main thread of
    the main interpreter (``signal.signal``/``setitimer`` raise ValueError
    off-main-thread) -- which holds here because run_grep/run_glob's sync
    scan core runs inline (see module docstring below on why NOT
    asyncio.to_thread for this one call).

    If signals are unavailable (no SIGALRM on this platform, or we are not
    on the main thread for some reason), this degrades to a no-op: the
    caller's between-units monotonic checks remain as the fallback bound
    (coarser, but still eventually terminates). N1: unlike before, this
    degradation is now LOUD rather than silent -- when SIGALRM is expected
    to be available (``can_signal`` True) but arming fails at the syscall
    (``ValueError``/``OSError``, the observed signature of "not the main
    thread"), a ``logger.warning`` fires once for this scan entry stating
    that the scan is running WITHOUT the primary ReDoS bound and is relying
    on between-unit checks only. Grep/glob currently dispatch on the
    asyncio main thread in production (verified: no ``Thread`` /
    ``to_thread`` / worker-loop wraps the tool-execution path), so this
    should never fire today; it exists so a future refactor that moves the
    scan off-thread (e.g. ``asyncio.to_thread``, a ``ThreadPoolExecutor``)
    is caught by an operator instead of silently losing the CRITICAL bound.
    The dispatch-level ``asyncio.wait_for`` "belt" in tools/__init__.py does
    NOT provide backup here or in general -- it can only cancel at an
    ``await`` point, and the scan body below is entirely synchronous, so it
    is inert against a single hung/backtracking match regardless of thread.

    CORRECTNESS PRECONDITION (N5): the code inside the ``with`` block MUST
    remain entirely ``await``-free and this guard must not be entered
    re-entrantly (nested) or concurrently on the same thread. The signal
    handler and itimer deadline are PROCESS-GLOBAL state that this context
    manager saves on ``__enter__`` and restores on ``__exit__``; if the
    guarded region ever yields control (an ``await``) while armed, or if two
    guards are active at once on one thread, the second ``__enter__``
    clobbers the first's itimer deadline and the second's ``__exit__``
    restores the WRONG saved handler (the first guard's ``_alarm_handler``,
    not the original), corrupting the save-restore chain. This is currently
    safe only because run_grep/run_glob's scan bodies contain zero awaits
    and ``asyncio.gather`` never actually interleaves two synchronous,
    await-free coroutines on one thread -- both must remain true for this
    guard to stay correct.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _TimeBudgetExceeded()

    can_signal = hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")
    armed = False
    old_handler = None
    if can_signal:
        try:
            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, remaining)
            armed = True
        except (ValueError, OSError):
            armed = False  # e.g. not main thread -- fall back to no-op
            # N1: loud degradation -- SIGALRM was available but could not be
            # armed (the ValueError/OSError signature of "not running on the
            # main thread of the main interpreter"). This scan proceeds with
            # NO primary ReDoS bound; only the coarser between-unit
            # time.monotonic() checks remain. Logged once per scan entry
            # (this function is entered exactly once per run_grep/run_glob
            # call), never per scanned unit.
            logger.warning(
                "grep/glob time-budget guard could not arm SIGALRM (not "
                "running on the main thread of the main interpreter) -- "
                "this scan is running WITHOUT the primary ReDoS time bound "
                "and is relying on between-unit checks only, which cannot "
                "interrupt a single in-flight catastrophic regex match."
            )

    try:
        yield
    finally:
        if armed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)


def _format_time_budget_error(budget: float) -> str:
    return (
        f"Search exceeded time budget ({budget:g}s) — narrow with "
        "path/glob or simplify the pattern."
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
    """Render a path workspace-relative when possible, else absolute (V7).

    R15: computed LEXICALLY via ``os.path.relpath`` -- never ``.resolve()``
    (which follows symlinks). A symlink INSIDE the workspace pointing at a
    file OUTSIDE it must still display at its in-workspace name (e.g.
    "link.txt"), never the resolved external absolute target: resolving
    would leak the external path and would make grep/glob spell the same
    in-workspace entry differently depending on incidental symlink targets.
    A purely lexical relpath naturally keeps grep and glob spelling the same
    file identically too (R15 forward-guard), since it never depends on
    what (if anything) a path component points at.
    """
    try:
        rel = os.path.relpath(str(p), str(workspace))
    except Exception:
        return str(p)
    rel_path = Path(rel)
    if rel == os.curdir or rel_path.parts[:1] == (os.pardir,):
        # Escapes upward out of the workspace (or IS the workspace root) --
        # not really "within" it for display purposes; fall back to
        # absolute rather than leak a ".."-relative escape (V7). Checked by
        # PARTS, not a string prefix, so a real dirname like "..hidden"
        # (which merely starts with the two characters ".." but is not the
        # parent-dir token) is not misclassified as an escape.
        return str(p)
    return rel_path.as_posix()


def _walk(
    root: Path, max_scan_files: int, deadline: float | None = None
) -> tuple[list[_Entry], bool, int]:
    """Walk ``root`` (no symlink following), pruning SKIP_DIRS, collecting
    up to ``max_scan_files`` total entries (files and directories combined).

    Returns ``(entries, bound_hit, symlinks_skipped)`` where ``bound_hit`` is
    True if the walk was stopped early because the bound was reached (V3
    footer steering), and ``symlinks_skipped`` is the count of symlinked
    FILE entries excluded from ``entries`` (R7). Per-directory names are
    processed in sorted order so an early stop always drops the same
    trailing entries call-to-call (determinism).

    R7: symlinked FILE entries are never followed -- checked via
    ``os.path.islink`` BEFORE the ``stat()``/read that would otherwise
    silently resolve through the link to outside-workspace content. Counted
    so the caller can report "N symlink(s) skipped" (an honest footer, not a
    silent omission). Symlinked DIRECTORIES are already excluded from
    descent by ``os.walk(..., followlinks=False)`` above (pre-existing,
    unchanged) -- this walker never recurses into one, so no file living
    only behind a symlinked directory can ever reach ``entries`` regardless.

    R1: if ``deadline`` is given, a ``time.monotonic()`` check runs between
    each directory entry processed (the "per glob directory entry" between-
    unit bound) -- this is the degraded-mode bound only; the primary
    interruption for a single pathological op is the caller's SIGALRM guard.
    Raises ``_TimeBudgetExceeded`` if the deadline has passed.
    """
    entries: list[_Entry] = []
    bound_hit = False
    symlinks_skipped = 0

    for dirpath, dirnames, filenames in os.walk(str(root), topdown=True, followlinks=False):
        # Prune in place: removes skip-list dirs from further traversal AND
        # from being emitted as directory candidates themselves.
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        dirpath_p = Path(dirpath)

        for d in dirnames:
            if deadline is not None and time.monotonic() >= deadline:
                raise _TimeBudgetExceeded()
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
            if deadline is not None and time.monotonic() >= deadline:
                raise _TimeBudgetExceeded()
            if len(entries) >= max_scan_files:
                bound_hit = True
                break
            p = dirpath_p / f
            # R7: check BEFORE stat()/read -- a symlinked file must never be
            # followed (no-follow policy matches the directory case above).
            if os.path.islink(p):
                symlinks_skipped += 1
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            entries.append(_Entry(path=p, is_dir=False, mtime=st.st_mtime, size=st.st_size))
        if bound_hit:
            break

    return entries, bound_hit, symlinks_skipped


def _skip_bound_footer_parts(
    skipped: int, symlinks_skipped: int, bound_hit: bool, max_scan_files: int
) -> list[str]:
    """Build the shared skip/bound footer fragments (R5/R7), shared by both
    grep's zero-match and non-zero-match paths, and by glob's.

    R5: this MUST be computed and consulted before any zero-match early
    return decides its wording -- a zero-match result must not claim
    definitive absence ("No matches for pattern ...") when some candidates
    were never scanned (skipped as binary/oversize/symlink, or the walk
    itself was bounded before completion): the needle could be sitting
    unread in one of those. Returning a non-empty list here is the signal
    the caller uses to swap the confident zero-match wording for the
    honest "incomplete" one.
    """
    parts: list[str] = []
    if skipped > 0:
        parts.append(f"({skipped} file(s) skipped: binary or exceeds size limit)")
    if symlinks_skipped > 0:
        parts.append(_SYMLINK_SKIP_NOTE.format(n=symlinks_skipped))
    if bound_hit:
        parts.append(_SCAN_BOUND_NOTE.format(max_scan_files=max_scan_files))
    return parts


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
    budget = DEFAULT_TIME_BUDGET_SECONDS
    try:
        # R11: validate params before doing any I/O -- an unknown output_mode
        # must not silently fall through to files_with_matches, and a
        # head_limit < 1 must not silently negative-slice or produce an
        # empty body (mirrors read_file's offset/limit posture).
        if output_mode not in _VALID_OUTPUT_MODES:
            return ToolResult(
                content=(
                    f"Error: invalid output_mode {output_mode!r}. "
                    f"Must be one of: {', '.join(_VALID_OUTPUT_MODES)}."
                ),
                is_error=True,
            )
        if head_limit is not None and head_limit < 1:
            return ToolResult(
                content=(
                    f"Error: head_limit must be >= 1 (got {head_limit}). "
                    "Omit head_limit to use the mode's default cap."
                ),
                is_error=True,
            )

        workspace = Path(workspace)
        max_scan_files = config.get("max_scan_files", DEFAULT_MAX_SCAN_FILES)
        max_file_bytes = config.get("max_file_bytes", DEFAULT_MAX_FILE_BYTES)
        budget = float((config or {}).get("time_budget_seconds", DEFAULT_TIME_BUDGET_SECONDS))

        root = _resolve_root(path, workspace)
        # N2: refuse a symlink AS THE SEARCH ROOT itself (narrow fix -- the
        # walker (R7) already skips symlinks discovered INSIDE a scan, but
        # never checked the root the caller named). os.path.islink() lstats
        # (never follows), so this also catches a broken symlink before
        # root.exists()/is_file() would otherwise silently follow it. No
        # general workspace-containment/realpath-escape check is added here
        # (filed separately) -- this refuses the literal symlink-root case
        # only, matching the no-follow policy the rest of the walker enforces.
        if os.path.islink(root):
            return ToolResult(
                content=(
                    f"Error: path is a symlink: {root} — search tools do "
                    "not follow symlinks; pass the resolved target "
                    "explicitly if intended."
                ),
                is_error=True,
            )
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

        # R1: deadline captured at scan entry; the SIGALRM guard below is the
        # primary interruption for a single catastrophic re.search call (a
        # match holds the GIL for its whole duration -- no between-unit check
        # can interrupt the unit currently backtracking). Between-unit
        # time.monotonic() checks (per file below, per entry inside _walk)
        # remain as the degraded-mode bound (no SIGALRM / non-main-thread)
        # and as a cheap early-out once the budget is already spent.
        deadline = time.monotonic() + budget
        bound_hit = False
        skipped = 0
        symlinks_skipped = 0
        # (path, mtime, [(lineno, line_text), ...]) per file with >=1 match.
        file_matches: list[tuple[Path, float, list[tuple[int, str]]]] = []

        with _time_budget_guard(deadline):
            # Candidate file set. grep's `path` may name a single file
            # directly (design §6 schema: "file or directory root") --
            # bypass the walker in that case rather than treating it as a
            # (non-existent) subtree.
            if root.is_file():
                candidates = [root]
            else:
                entries, bound_hit, symlinks_skipped = _walk(
                    root, max_scan_files, deadline=deadline)
                candidates = [e.path for e in entries if not e.is_dir]
                if glob:
                    candidates = [p for p in candidates if fnmatch.fnmatch(p.name, glob)]

            for p in candidates:
                if time.monotonic() >= deadline:
                    raise _TimeBudgetExceeded()
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
                    # Between-unit (per-line-batch) deadline check -- the
                    # degraded-mode bound; granularity of 1 line is cheap
                    # (a single time.monotonic() call) and matches the R1
                    # fixtures' per-line cost distribution.
                    if time.monotonic() >= deadline:
                        raise _TimeBudgetExceeded()
                    if rx.search(line[:_MAX_REGEX_INPUT_CHARS]) is not None:
                        hits.append((lineno, line))
                if hits:
                    file_matches.append((p, st.st_mtime, hits))

        # Deterministic order: mtime desc, path asc tiebreak (V3).
        file_matches.sort(key=lambda t: (-t[1], _display_path(t[0], workspace)))

        # R5: the skip/bound footer is computed BEFORE the zero-match early
        # return decides its wording -- a zero-match result must not claim
        # definitive absence when some candidates were never scanned (the
        # needle could be sitting unread in a skipped/unreached file).
        footer_parts = _skip_bound_footer_parts(
            skipped, symlinks_skipped, bound_hit, max_scan_files)

        if not file_matches:
            if footer_parts:
                # Skips/bound-hit exist: do NOT claim definitive absence.
                # NOTE: the pattern is deliberately NOT echoed here — when a
                # zero-match is caused by skipped entries (e.g. symlinks,
                # R7), echoing the needle back would reproduce the very
                # string the no-follow policy refused to read (pinned by
                # TestR7Symlinks::test_symlinked_file_content_not_returned).
                content = (
                    "No matches in scanned files. Try broadening the "
                    "search, check the path/glob filters, or verify the "
                    "regex is correct (Python re syntax).\n\n"
                    + "\n".join(footer_parts)
                )
                return ToolResult(content=content, is_error=False)
            # PRESERVATION CONSTRAINT: zero skips and no bound hit -- keep
            # the exact original text (test_search_tools.py pins this
            # substring in test_zero_match_is_hint_not_error and
            # test_case_insensitive_flag).
            return ToolResult(
                content=(
                    f"No matches for pattern {pattern!r}. Try broadening "
                    "the search, check the path/glob filters, or verify "
                    "the regex is correct (Python re syntax)."
                ),
                is_error=False,
            )

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

        # skip/symlink/bound-hit fragments were already appended to
        # footer_parts above (R5, before the zero-match decision) -- only
        # the mode-specific OVERFLOW fragment (if any) was added since.

        content = body
        if footer_parts:
            content = content + "\n\n" + "\n".join(footer_parts)

        return ToolResult(content=content, is_error=False)

    except _TimeBudgetExceeded:
        return ToolResult(content=_format_time_budget_error(budget), is_error=True)
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
    budget = DEFAULT_TIME_BUDGET_SECONDS
    try:
        # R11: glob shares grep's head_limit posture -- < 1 must error, not
        # silently negative-slice or produce an empty body.
        if head_limit is not None and head_limit < 1:
            return ToolResult(
                content=(
                    f"Error: head_limit must be >= 1 (got {head_limit}). "
                    "Omit head_limit to use the default cap (100)."
                ),
                is_error=True,
            )

        workspace = Path(workspace)
        max_scan_files = config.get("max_scan_files", DEFAULT_MAX_SCAN_FILES)
        budget = float((config or {}).get("time_budget_seconds", DEFAULT_TIME_BUDGET_SECONDS))

        root = _resolve_root(path, workspace)
        # N2: refuse a symlink AS THE SEARCH ROOT itself (see run_grep's
        # identical check above for the full rationale — narrow fix only,
        # no general workspace-containment/realpath-escape check).
        if os.path.islink(root):
            return ToolResult(
                content=(
                    f"Error: path is a symlink: {root} — search tools do "
                    "not follow symlinks; pass the resolved target "
                    "explicitly if intended."
                ),
                is_error=True,
            )
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

        # R1: same deadline + SIGALRM design as run_grep -- a single
        # backtracking rx.match(rel) call below holds the GIL for its whole
        # duration, so the SIGALRM guard is the primary interruption; the
        # per-entry time.monotonic() check is the degraded-mode bound.
        deadline = time.monotonic() + budget

        with _time_budget_guard(deadline):
            entries, bound_hit, symlinks_skipped = _walk(
                root, max_scan_files, deadline=deadline)

            matched: list[_Entry] = []
            for e in entries:
                if time.monotonic() >= deadline:
                    raise _TimeBudgetExceeded()
                rel = e.path.relative_to(root).as_posix()
                if rx.match(rel) is not None:
                    matched.append(e)

        # Deterministic order: mtime desc, path asc tiebreak (V3).
        matched.sort(key=lambda e: (-e.mtime, _display_path(e.path, workspace)))

        # R5: the skip/bound footer is computed BEFORE the zero-match early
        # return decides its wording (mirrors run_grep) -- a zero-match
        # result must not claim definitive absence when the walk was cut
        # short or symlink entries were skipped (the target could be one of
        # them). glob has no binary/oversize skip class, hence skipped=0.
        zero_footer_parts = _skip_bound_footer_parts(
            0, symlinks_skipped, bound_hit, max_scan_files)

        if not matched:
            if zero_footer_parts:
                # Skips/bound-hit exist: do NOT claim definitive absence.
                # Pattern deliberately not echoed (mirrors run_grep's
                # honest path; see R7 no-follow note there).
                return ToolResult(
                    content=(
                        f"No matches in scanned files under {root}. The "
                        "scan was incomplete -- try a broader pattern or "
                        "a different path root.\n\n"
                        + "\n".join(zero_footer_parts)
                    ),
                    is_error=False,
                )
            # PRESERVATION: zero skips and no bound hit -- keep the original
            # confident wording.
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
        if symlinks_skipped > 0:
            footer_parts.append(_SYMLINK_SKIP_NOTE.format(n=symlinks_skipped))
        if bound_hit:
            footer_parts.append(_SCAN_BOUND_NOTE.format(max_scan_files=max_scan_files))

        content = body
        if footer_parts:
            content = content + "\n\n" + "\n".join(footer_parts)

        return ToolResult(content=content, is_error=False)

    except _TimeBudgetExceeded:
        return ToolResult(content=_format_time_budget_error(budget), is_error=True)
    except Exception as e:
        return ToolResult(content=f"Error running glob: {e}", is_error=True)
