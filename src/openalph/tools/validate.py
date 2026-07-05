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


def _run_subprocess_checker(
    binary: str, args: tuple[str, ...], content: str, suffix: str,
) -> tuple[str, str]:
    """Run one subprocess syntax check against candidate `content`.

    Returns (status, detail):
        "ok"          - checker ran, exit 0
        "fail"        - checker ran, nonzero exit; detail = checker output
        "unavailable" - the checker invocation itself failed (timeout or
                        crash) — an infrastructure failure, not a verdict
                        (fail-open, V2/V4); detail explains why

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
    return _run_subprocess_checker(binary, args, content, suffix=suffix)


def _write_now(path: str, content: str) -> None:
    """The ONE place a validated write reaches disk (single point of truth)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


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
        except Exception:
            # Can't even read the original to validate it (e.g. a binary
            # file — write_file has no _is_binary guard for existing
            # files). Validation cannot proceed meaningfully; skip it
            # entirely rather than guess at a verdict.
            _write_now(path, new_content)
            return ToolResult(content="", is_error=False)
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
