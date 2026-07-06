"""Validate-on-edit subsystem (Component C, bundle-2 workspace-kdsn.195).

Design: specs/file-search-refresh-design.md §5. Invariants V1 (atomic
mutation) and V2 (never-worse validation — an agent can always fix a
broken file; checker unavailability/timeout/crash always fails OPEN).

Checker registry (module-level, one line to extend):
    .py                -> compile(source, path, "exec")   in-process
    .json               -> json.loads(source)               in-process
    .toml               -> tomllib.loads(source)             in-process
    .sh / .bash         -> bash -n <tmpfile>                 subprocess, gated on shutil.which("bash")
    .js / .mjs / .cjs    -> node --check <tmpfile>            subprocess, gated on shutil.which("node")
    anything else       -> no checker, skip silently

In-process checkers never touch disk (no subprocess, no __pycache__ side
effects — compile() alone never writes bytecode; that only happens via
`import`/py_compile). Subprocess checkers run against a NamedTemporaryFile
holding the CANDIDATE content (which is not yet on disk at the real path),
with the file's real extension as the tempfile suffix (so e.g. .mjs content
is checked under ESM parsing rules), a minimal env (PATH only — never
executes the file, only -n/--check flags), a 5s timeout, and are always
cleaned up in `finally` regardless of outcome (no orphan tempfiles).

Missing subprocess binary (shutil.which -> falsy) is a SILENT skip —
behaviorally identical to "no checker for this extension" (best-effort
gating, V4). This is distinct from a checker that IS available but then
times out or crashes when invoked (an infrastructure failure, not a syntax
verdict) — that case fails open too, but appends a "(validation skipped:
checker unavailable)" note so the write result says why.

Shared seam: _validated_write(path, new_content, tool_config) is used by
ALL THREE mutating file tools (write_file / edit_file / patch_file in
file.py). It is the SOLE place any of the three ever calls open(path, 'w')
for a validated write, which is what makes V1 (atomicity — a rejected edit
leaves the file byte-identical) tractable: the disk write simply never
happens on the reject path.

Decision matrix (design §5 pt 5, V2):
    post_ok                    -> write, unadorned (regardless of pre_ok —
                                   fixing a broken file is never penalized)
    not post_ok AND pre_ok     -> REJECT: no write, file untouched,
                                   is_error=True, checker output capped at
                                   2000 chars, "file unchanged" language,
                                   fix-and-retry steering
    not post_ok AND not pre_ok -> write proceeds (V2: never block a broken
                                   file from being touched) + a "pre-existing
                                   errors" warning appended to the result

Each of pre_ok/post_ok invokes its checker AT MOST ONCE per _validated_write
call (never re-run on the same content) — pre_ok is skipped entirely
(vacuously True, no checker invocation) when the target file does not yet
exist, since "new files must be born clean" reduces to just checking
post_ok in that case.
"""

import json
import logging
import os
import shutil
import stat
import subprocess
import tempfile
import tomllib
from pathlib import Path
from typing import Callable

from . import ToolResult

logger = logging.getLogger(__name__)

_SUBPROCESS_TIMEOUT = 5.0
_CHECKER_OUTPUT_CAP = 2000


# ---------------------------------------------------------------------------
# In-process checkers: content, path -> (ok, detail)
# ---------------------------------------------------------------------------

def _check_python(content: str, path: str) -> tuple[bool, str]:
    """compile() only — never imports, so never writes __pycache__."""
    try:
        compile(content, path, "exec")
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _check_json(content: str, path: str) -> tuple[bool, str]:
    try:
        json.loads(content)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _check_toml(content: str, path: str) -> tuple[bool, str]:
    try:
        tomllib.loads(content)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# extension -> checker(content, path) -> (ok, detail)
_INPROCESS_CHECKERS: dict[str, Callable[[str, str], tuple[bool, str]]] = {
    ".py": _check_python,
    ".json": _check_json,
    ".toml": _check_toml,
}

# extension -> (binary, check_args) for `binary *check_args tmpfile`
_SUBPROCESS_CHECKERS: dict[str, tuple[str, tuple[str, ...]]] = {
    ".sh": ("bash", ("-n",)),
    ".bash": ("bash", ("-n",)),
    ".js": ("node", ("--check",)),
    ".mjs": ("node", ("--check",)),
    ".cjs": ("node", ("--check",)),
}


def _cap(text: str) -> str:
    """Cap checker output relayed to the model at _CHECKER_OUTPUT_CAP chars."""
    if len(text) <= _CHECKER_OUTPUT_CAP:
        return text
    return text[:_CHECKER_OUTPUT_CAP] + " \u2026[checker output truncated]"


def _scrub_tmp_path(detail: str, tmp_path: str, real_path: str) -> str:
    """Rewrite the NamedTemporaryFile path a subprocess checker printed into
    its output with the REAL target path, before that detail is relayed to
    the model (R14). A checker like `bash -n` names the file it checked
    (the ephemeral tempfile) in its error text; left as-is, the rejection
    detail would reference a meaningless /tmp basename instead of the file
    the model actually asked to write. Replaces the full tmp path first,
    then falls back to a basename-only replace in case only the basename
    survived some other transformation upstream."""
    if not detail:
        return detail
    scrubbed = detail.replace(tmp_path, real_path)
    tmp_base = os.path.basename(tmp_path)
    real_base = os.path.basename(real_path)
    if tmp_base and tmp_base != real_base:
        scrubbed = scrubbed.replace(tmp_base, real_base)
    return scrubbed


def _run_subprocess_checker(
    binary: str, args: tuple[str, ...], content: str, suffix: str, real_path: str,
) -> tuple[str, str]:
    """Run one subprocess syntax check against candidate `content`.

    Returns (status, detail):
        "ok"          - checker ran, exit 0
        "fail"        - checker ran, nonzero exit; detail = checker output,
                        with any ephemeral tempfile path rewritten to
                        `real_path` (R14) before being relayed to the model
        "unavailable" - the checker invocation itself failed (timeout or
                        crash) — an infrastructure failure, not a verdict
                        (fail-open, V2/V4); detail explains why

    Args:
        real_path: the REAL target path (not yet on disk, or the file being
            overwritten) — used only to scrub the tempfile path out of a
            "fail" detail before it is relayed.

    The tempfile is always removed in `finally`, regardless of outcome —
    no orphan tempfiles survive a call, success or failure.
    """
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False, encoding="utf-8",
        ) as tf:
            tf.write(content)
            tmp_path = tf.name

        minimal_env = {"PATH": os.environ.get("PATH", "")}
        proc = subprocess.run(
            [binary, *args, tmp_path],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT,
            env=minimal_env,
        )
        if proc.returncode == 0:
            return "ok", ""
        detail = (proc.stderr or proc.stdout or "").strip()
        if not detail:
            detail = f"{binary} exited with code {proc.returncode}"
        detail = _scrub_tmp_path(detail, tmp_path, real_path)
        return "fail", detail
    except subprocess.TimeoutExpired:
        logger.warning(
            "Validation checker %s timed out after %ss; failing open",
            binary, _SUBPROCESS_TIMEOUT,
        )
        return "unavailable", f"{binary} check timed out"
    except Exception as e:
        logger.warning("Validation checker %s crashed: %s; failing open", binary, e)
        return "unavailable", f"{binary} check crashed: {e}"
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _get_checker(path: str):
    """Resolve the checker plan for `path`'s extension, or None.

    None covers BOTH "no checker registered for this extension" and
    "checker registered but its binary is not on PATH" — both are a
    SILENT skip (plain write, no note at all; best-effort gating per V4).
    This is intentionally the SAME outcome as an unregistered extension —
    a missing `bash`/`node` is not treated as a "checker unavailable"
    infrastructure failure (that note is reserved for a checker that WAS
    available but then timed out or crashed when actually invoked).

    Returns one of:
        ("inprocess", checker_fn)
        ("subprocess", binary, args)
        None
    """
    ext = Path(path).suffix
    if ext in _INPROCESS_CHECKERS:
        return ("inprocess", _INPROCESS_CHECKERS[ext])
    if ext in _SUBPROCESS_CHECKERS:
        binary, args = _SUBPROCESS_CHECKERS[ext]
        if shutil.which(binary):
            return ("subprocess", binary, args)
        return None
    return None


def _run_checker(plan, content: str, path: str) -> tuple[str, str]:
    """Run one checker invocation per `plan` (see _get_checker). (status, detail)."""
    if plan[0] == "inprocess":
        fn = plan[1]
        ok, detail = fn(content, path)
        return ("ok" if ok else "fail"), detail
    _, binary, args = plan
    suffix = Path(path).suffix
    return _run_subprocess_checker(binary, args, content, suffix=suffix, real_path=path)


def _write_now(path: str, content: str) -> None:
    """The ONE place a validated write reaches disk (single point of truth).

    Writes ATOMICALLY (R6): content lands in a ``tempfile.NamedTemporaryFile``
    created in the TARGET'S PARENT DIRECTORY (same filesystem as the target,
    so the final ``os.replace`` is an atomic rename, never a cross-device
    copy), then flushed and ``os.fsync``'d. When overwriting an EXISTING
    file, the tmp is ``os.chmod``'d to that file's PRIOR mode before the
    replace — a fresh ``NamedTemporaryFile`` defaults to 0o600, which would
    otherwise silently reset an existing file's permissions on every write
    that goes through this seam. The tmp is unlinked on ANY failure path
    (tempfile creation, write, fsync, chmod, or the replace itself), so a
    failed write never leaves an orphan tmp file, and the real path — if it
    already existed — is left byte-identical.

    N6 (atomicity claim, scoped): the guarantee above is about
    READER-VISIBILITY only — no reader of ``path`` can ever observe a torn
    or partially-written file, because the swap is a single ``os.replace``
    rename that either lands in full or not at all. This is NOT a claim
    about cross-crash durability of the rename itself: ``os.replace`` is not
    followed by an ``fsync`` on the parent directory's fd here, so on a
    hard crash/power-loss in the narrow window right after this function
    returns, a journaling filesystem could in principle revert the
    directory entry to the prior inode on reboot even though the tempfile's
    *data* was fsync'd. That cross-crash durability gap is deliberately out
    of scope for this function (it would need a parent-dir fsync to close);
    what V1 actually promises, and what this function delivers, is that a
    reader never sees a torn/partial file, not that the rename itself
    durably survives a power-loss immediately after the call.

    N3 (symlink targets): if ``path`` is a symlink, the OLD pre-R6 code
    (``open(path, 'w')``) wrote THROUGH the link (mutating the link's
    target in place); naively ``os.replace``-ing onto ``path`` itself would
    instead replace the symlink's own directory entry with a regular file —
    silently detaching the link (the target file would keep its original,
    now-stale content, and the agent's edit would appear to succeed while
    never reaching the real file). To preserve the pre-existing write-
    through-symlink semantics while keeping R6's atomicity, this function
    resolves ``path`` via ``os.path.realpath()`` ONCE when ``path`` is a
    symlink, and performs the ENTIRE tempfile+chmod+replace sequence against
    that REAL target instead: the tempfile is created in the real target's
    parent directory, the prior mode is read from the real target, and
    ``os.replace`` lands on the real target's path. ``path`` itself (the
    symlink) is never touched, so it survives pointing at the same (now
    updated) real file. Non-symlink ``path`` values are completely unaffected
    (``target_path`` degenerates to ``path`` and every step below is
    byte-for-byte the same as before this fix).

    Hardlink divergence (documented, no code change — see N3 finding): if
    ``path`` has hardlink siblings (``os.stat(path).st_nlink > 1``),
    ``os.replace`` still gives ``path`` a NEW inode, so sibling hardlinks
    keep the OLD content and now point at a different inode than ``path``.
    An in-place fallback (truncate + write) would avoid that divergence but
    would reopen R6's original atomicity hole (a reader could observe a
    torn/partial file mid-write, and a mid-write crash could leave a
    corrupted file instead of the original) — that tradeoff is deliberately
    rejected here. Hardlinked targets are therefore expected to diverge from
    their siblings on write, same as a plain ``os.replace``-based editor
    would behave; there is no special-case handling for this in the code.

    ``os.replace`` is called as a plain module-level attribute access (this
    module does ``import os`` at the top, never ``from os import replace``)
    so that tests which monkeypatch ``openalph.tools.validate.os.replace``
    observe the substitution.

    R10: parent-directory creation happens HERE — immediately before the
    write actually lands — rather than upfront in the calling tool. A
    rejected write (validation reject) never reaches this function, so it
    leaves no orphan parent directories for a brand-new nested path.
    """
    # N3: resolve a symlinked path ONCE to its real target and perform the
    # whole tempfile/chmod/replace sequence there, so `path` (the symlink)
    # is never itself replaced and keeps pointing at the (now-updated) real
    # file. Non-symlink paths are unaffected: target_path == path.
    target_path = os.path.realpath(path) if os.path.islink(path) else path

    parent = os.path.dirname(os.path.abspath(target_path)) or os.sep
    os.makedirs(parent, exist_ok=True)

    prior_mode: int | None = None
    if os.path.exists(target_path):
        prior_mode = stat.S_IMODE(os.stat(target_path).st_mode)

    suffix = Path(target_path).suffix or None
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=parent, prefix=".tmp-", suffix=suffix,
            delete=False, encoding="utf-8",
        ) as tf:
            tmp_path = tf.name
            tf.write(content)
            tf.flush()
            os.fsync(tf.fileno())

        if prior_mode is not None:
            os.chmod(tmp_path, prior_mode)

        os.replace(tmp_path, target_path)
    except Exception:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise


async def _validated_write(
    path: str, new_content: str, tool_config: dict | None,
) -> ToolResult:
    """Shared validated-write seam for write_file / edit_file / patch_file.

    Performs the actual disk write when validation allows it (this is the
    ONLY place any of the three tools writes to disk for a validated call —
    see module docstring for why that matters for V1 atomicity) and returns
    a ToolResult:
        is_error=False, content=""      -> clean write, no note
        is_error=False, content=<note>  -> write proceeded but carries a
                                            note (pre-existing errors warning,
                                            or a checker-unavailable
                                            fail-open note) — callers should
                                            append this to their own
                                            success message
        is_error=True,  content=<msg>   -> write REFUSED; file is untouched;
                                            callers should return this
                                            ToolResult AS-IS (it is already
                                            the complete error message)

    Args:
        path: Target file path (real, resolved or not — used only for
            extension detection, compile()'s filename arg, and an
            existence check for pre_ok).
        new_content: The full candidate content that would be written.
        tool_config: Tool config dict (may be None); validate_on_edit
            (default True) is the kill-switch.
    """
    tool_config = tool_config or {}

    if not tool_config.get("validate_on_edit", True):
        _write_now(path, new_content)
        return ToolResult(content="", is_error=False)

    plan = _get_checker(path)
    if plan is None:
        _write_now(path, new_content)
        return ToolResult(content="", is_error=False)

    # pre_ok: checker(original) if the file exists, else vacuously True —
    # "new files must be born clean" reduces to checking post_ok alone.
    pre_status = "ok"
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                original = f.read()
        except FileNotFoundError:
            # R13 TOCTOU: the file existed at the os.path.exists() check
            # above but is gone by the time we actually open it (e.g. a
            # concurrent delete). Treat this exactly like a new file —
            # pre_ok stays vacuously True (no checker invocation on
            # content we can no longer read) — and fall through to the
            # normal post_ok gate below. This is deliberately NOT the
            # generic fail-open path just below: that path skips
            # validation ENTIRELY (writes unconditionally), which would
            # let a broken candidate land with no check at all merely
            # because of an unlucky race.
            pass
        except Exception:
            # Can't even read the original to validate it for some OTHER
            # reason (e.g. a binary file — write_file has no _is_binary
            # guard for existing files). Validation cannot proceed
            # meaningfully; skip it entirely rather than guess at a verdict.
            _write_now(path, new_content)
            return ToolResult(content="", is_error=False)
        else:
            pre_status, _pre_detail = _run_checker(plan, original, path)

    if pre_status == "unavailable":
        # Infra failure on the pre-check: never learned pre_ok, so we can't
        # trust any verdict. Fail open immediately without invoking the
        # checker again on the new content (at most one invocation per side).
        _write_now(path, new_content)
        return ToolResult(
            content="(validation skipped: checker unavailable)", is_error=False,
        )

    post_status, post_detail = _run_checker(plan, new_content, path)

    if post_status == "unavailable":
        _write_now(path, new_content)
        return ToolResult(
            content="(validation skipped: checker unavailable)", is_error=False,
        )

    if post_status == "ok":
        # Fixing a broken file (pre_status == "fail") is never penalized —
        # V2 always allows an agent to repair a pre-existing problem.
        _write_now(path, new_content)
        return ToolResult(content="", is_error=False)

    # post_status == "fail" from here on.
    if pre_status == "ok":
        # Clean -> broken: REJECT. No write. File stays byte-identical (V1).
        return ToolResult(
            content=(
                "Edit rejected: it introduces syntax errors (file unchanged). "
                f"{_cap(post_detail)} Fix the edit and retry."
            ),
            is_error=True,
        )

    # pre_status == "fail": broken -> still broken. Write proceeds (V2)
    # with a warning so the agent knows the file is still not clean.
    _write_now(path, new_content)
    return ToolResult(
        content=(
            "Note: file had pre-existing syntax errors and still fails "
            f"validation: {_cap(post_detail)}."
        ),
        is_error=False,
    )
