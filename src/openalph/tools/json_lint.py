"""json_lint — read-only JSON/JSONL validation for stations (bead
workspace-e2uh.168, Decision 18 "proper tools" follow-up).

Stations authoring JSON/JSONL (the decomposer's ticket manifests) used to
get syntax errors only at the DETERMINISTIC VALIDATOR's gate 1 — after the
episode ended, costing a full repair round for a missing brace. A read-only
lint tool lets the station self-check BEFORE submitting, converting a
post-hoc repair into an in-episode fix.

Read-only by construction: opens the given path (or takes inline ``text``),
parses, returns a bounded structured verdict; never writes anything.

Formats:
- ``jsonl`` (default when the target path ends with ``.jsonl``): every
  NON-BLANK line is parsed INDEPENDENTLY and ALL defects are reported with
  1-based line numbers and positions — line-precise defects are exactly
  what an edit-in-place repair round needs.
- ``json``: one ``json.loads`` over the whole content; success reports the
  parsed shape (object/array/scalar, top-level keys or item count).

Output is a compact JSON string (bounded: max 20 defects, excerpts capped),
so the station can read the verdict without another tool call.
"""

from __future__ import annotations

import json
import os

from openalph.tools import ToolResult

# Bounded-tool discipline (kdsn.163 class): refuse absurd inputs rather
# than parsing them into the event/stream plane.
_MAX_CONTENT_BYTES = 2 * 1024 * 1024
_MAX_DEFECTS = 20
_EXCERPT_CAP = 120

_JSONL_SUFFIXES = (".jsonl", ".ndjson")


def _excerpt(line: str) -> str:
    return line if len(line) <= _EXCERPT_CAP else line[:_EXCERPT_CAP] + "…"


def _lint_jsonl(content: str) -> dict:
    """Parse every non-blank line independently; report ALL defects with
    1-based line numbers (never fail fast — the repair loop wants the full
    defect list)."""
    errors: list[dict] = []
    total = 0
    parsed = 0
    for lineno, raw in enumerate(content.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        total += 1
        try:
            json.loads(line)
            parsed += 1
        except json.JSONDecodeError as exc:
            if len(errors) < _MAX_DEFECTS:
                errors.append(
                    {
                        "line": lineno,
                        "col": exc.colno,
                        "msg": exc.msg,
                        "excerpt": _excerpt(line),
                    }
                )
    if total == 0:
        return {
            "ok": False,
            "format": "jsonl",
            "lines": 0,
            "parsed": 0,
            "errors": [{"msg": "no non-blank lines to parse"}],
        }
    # Defects beyond the cap are summarized, never silently dropped.
    suppressed = total - parsed - len(errors)
    if suppressed > 0:
        errors.append(
            {"msg": f"… {suppressed} further defect(s) not shown (cap {_MAX_DEFECTS})"}
        )
    return {
        "ok": parsed == total,
        "format": "jsonl",
        "lines": total,
        "parsed": parsed,
        "errors": errors,
    }


def _lint_json(content: str) -> dict:
    try:
        obj = json.loads(content)
    except json.JSONDecodeError as exc:
        lines = content.splitlines()
        excerpt = lines[exc.lineno - 1] if 0 < exc.lineno <= len(lines) else ""
        return {
            "ok": False,
            "format": "json",
            "errors": [
                {
                    "line": exc.lineno,
                    "col": exc.colno,
                    "msg": exc.msg,
                    "excerpt": _excerpt(excerpt),
                }
            ],
        }
    summary: dict
    if isinstance(obj, dict):
        summary = {"kind": "object", "top_level_keys": sorted(map(str, obj.keys()))[:64]}
    elif isinstance(obj, list):
        summary = {"kind": "array", "items": len(obj)}
    else:
        summary = {"kind": type(obj).__name__}
    return {"ok": True, "format": "json", **summary}


async def run_json_lint(
    *,
    path: str | None = None,
    text: str | None = None,
    format: str | None = None,
) -> ToolResult:
    """Validate JSON/JSONL from a file path or inline text (exactly one
    source). Returns a bounded JSON verdict as the result content."""
    if (path is None) == (text is None):
        return ToolResult(
            content="Error: provide exactly one of `path` or `text`.",
            is_error=True,
        )
    if format is not None and format not in ("json", "jsonl"):
        return ToolResult(
            content=f"Error: format must be 'json' or 'jsonl' (got {format!r}).",
            is_error=True,
        )

    if path is not None:
        if not os.path.exists(path):
            return ToolResult(content=f"Error: File not found: {path}", is_error=True)
        if not os.path.isfile(path):
            return ToolResult(content=f"Error: Path is not a file: {path}", is_error=True)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read(_MAX_CONTENT_BYTES + 1)
        except UnicodeDecodeError:
            return ToolResult(
                content=f"Error: Binary file cannot be linted: {path}", is_error=True
            )
        except Exception as e:  # pragma: no cover - defensive, mirrors read_file
            return ToolResult(content=f"Error reading file: {e}", is_error=True)
        if "\x00" in content:
            # Parity with read_file's binary signal: a NUL byte means the
            # "JSON file" is really binary corruption — say so plainly
            # instead of emitting a confusing parse error.
            return ToolResult(
                content=f"Error: Binary file cannot be linted: {path}", is_error=True
            )
        if len(content.encode("utf-8")) > _MAX_CONTENT_BYTES:
            return ToolResult(
                content=(
                    f"Error: file exceeds the {_MAX_CONTENT_BYTES // (1024 * 1024)}MB "
                    "lint cap — lint a narrower file or pass inline `text`."
                ),
                is_error=True,
            )
        effective_format = format or (
            "jsonl" if str(path).lower().endswith(_JSONL_SUFFIXES) else "json"
        )
    else:
        content = text or ""
        if len(content.encode("utf-8")) > _MAX_CONTENT_BYTES:
            return ToolResult(
                content=(
                    f"Error: inline text exceeds the "
                    f"{_MAX_CONTENT_BYTES // (1024 * 1024)}MB lint cap."
                ),
                is_error=True,
            )
        effective_format = format or "json"

    verdict = _lint_jsonl(content) if effective_format == "jsonl" else _lint_json(content)
    return ToolResult(content=json.dumps(verdict), is_error=not verdict["ok"])
