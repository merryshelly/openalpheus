"""Bead workspace-e2uh.168: the json_lint builtin — read-only JSON/JSONL
validation for stations (Decision 18 "proper tools" follow-up).

Contract under test:
- exactly one of `path`/`text`; `format` json|jsonl (jsonl default for
  *.jsonl/*.ndjson paths, else json);
- jsonl: every non-blank line parsed independently, ALL defects reported
  with 1-based line numbers (never fail fast), blank lines skipped, an
  empty result is a defect, defects beyond the cap are SUMMARIZED (never
  silently dropped);
- json: one parse; success carries the parsed shape;
- read-only: never writes; missing/not-a-file/binary are errors;
- bounded: the verdict JSON stays small (capped defects, capped excerpts).
"""

from __future__ import annotations

import asyncio
import json

from openalph.tools import BUILTIN_TOOLS
from openalph.tools.json_lint import run_json_lint


def lint(**kwargs):
    return asyncio.run(run_json_lint(**kwargs))


def test_registered_in_builtin_tools():
    entry = BUILTIN_TOOLS["json_lint"]
    assert "read-only" in entry["description"]
    assert entry["parameters"]["properties"]["format"]["enum"] == ["json", "jsonl"]


def test_requires_exactly_one_source():
    r = lint()
    assert r.is_error and "exactly one" in r.content
    r = lint(path="a", text="b")
    assert r.is_error and "exactly one" in r.content


def test_format_validation():
    r = lint(text="{}", format="yaml")
    assert r.is_error and "format must be" in r.content


def test_json_happy_object():
    r = lint(text='{"a": 1, "b": [2, 3]}')
    assert not r.is_error
    verdict = json.loads(r.content)
    assert verdict["ok"] is True
    assert verdict["kind"] == "object"
    assert verdict["top_level_keys"] == ["a", "b"]


def test_json_happy_array_and_scalar():
    verdict = json.loads(lint(text="[1, 2]").content)
    assert verdict == {"ok": True, "format": "json", "kind": "array", "items": 2}
    verdict = json.loads(lint(text="42").content)
    assert verdict["kind"] == "int"


def test_json_error_carries_line_col():
    r = lint(text='{\n  "a": 1,\n  "b": ,\n}')
    assert r.is_error
    verdict = json.loads(r.content)
    assert verdict["ok"] is False
    err = verdict["errors"][0]
    assert err["line"] == 3
    assert err["col"] > 0
    assert err["msg"]
    assert err["excerpt"].strip() == '"b": ,'


def test_jsonl_happy_skips_blank_lines():
    r = lint(text='{"n": 1}\n\n   \n{"n": 2}\n', format="jsonl")
    assert not r.is_error
    verdict = json.loads(r.content)
    assert verdict == {"ok": True, "format": "jsonl", "lines": 2, "parsed": 2, "errors": []}


def test_jsonl_reports_all_defects_with_line_numbers():
    content = "\n".join(
        [
            '{"ok": 1}',
            '{"bad": [}',
            '{"ok": 2}',
            'not json at all',
            '{"ok": 3}',
        ]
    )
    r = lint(text=content, format="jsonl")
    assert r.is_error
    verdict = json.loads(r.content)
    assert verdict["lines"] == 5
    assert verdict["parsed"] == 3
    assert [e["line"] for e in verdict["errors"]] == [2, 4]
    assert verdict["errors"][0]["col"] > 0


def test_jsonl_empty_content_is_a_defect():
    r = lint(text="\n  \n", format="jsonl")
    assert r.is_error
    verdict = json.loads(r.content)
    assert verdict["lines"] == 0
    assert "no non-blank lines" in verdict["errors"][0]["msg"]


def test_jsonl_defect_cap_summarized_not_dropped():
    # 25 defective lines; the verdict shows 20 + a summary line
    content = "\n".join("{bad}" for _ in range(25))
    r = lint(text=content, format="jsonl")
    verdict = json.loads(r.content)
    assert len(verdict["errors"]) == 21  # _MAX_DEFECTS + the summary entry
    assert "20" in verdict["errors"][-1]["msg"]


def test_jsonl_default_format_for_jsonl_path(tmp_path):
    p = tmp_path / "manifest.jsonl"
    p.write_text('{"a": 1}\n{"b": 2}\n')
    r = lint(path=str(p))
    assert not r.is_error
    assert json.loads(r.content)["format"] == "jsonl"
    # .json defaults to json — a JSONL body is a single-parse defect there
    p2 = tmp_path / "thing.json"
    p2.write_text('{"a": 1}\n{"b": 2}\n')
    r2 = lint(path=str(p2))
    assert r2.is_error
    assert json.loads(r2.content)["format"] == "json"


def test_ndjson_suffix_defaults_to_jsonl(tmp_path):
    p = tmp_path / "rows.ndjson"
    p.write_text('1\n2\n')
    r = lint(path=str(p))
    assert not r.is_error
    assert json.loads(r.content)["format"] == "jsonl"


def test_path_errors(tmp_path):
    r = lint(path=str(tmp_path / "missing.jsonl"))
    assert r.is_error and "not found" in r.content
    d = tmp_path / "dir"
    d.mkdir()
    r = lint(path=str(d))
    assert r.is_error and "not a file" in r.content


def test_path_binary_is_error(tmp_path):
    p = tmp_path / "blob.jsonl"
    p.write_bytes(b'{"a": 1}\n\x00\x00\n')
    r = lint(path=str(p))
    assert r.is_error and "Binary" in r.content


def test_excerpt_capped():
    long_bad = "x" * 500
    r = lint(text=long_bad, format="jsonl")
    verdict = json.loads(r.content)
    assert len(verdict["errors"][0]["excerpt"]) <= 121  # cap + ellipsis


def test_never_writes(tmp_path):
    p = tmp_path / "w.json"
    p.write_text('{"a": 1}')
    before = p.read_bytes()
    lint(path=str(p))
    assert p.read_bytes() == before  # read-only by construction
