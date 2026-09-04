"""Shared helpers for the context-handoff red suites (kdsn.322 TDD build).

One helpers module for all five slice suites (test_context_handoff_{config,
strip,checkpoint,exec,toolslash}.py) — shared fixtures instead of
copy-pasted per-file fixture code, so interface drift between serialized
slices (T0→T1→T2→T3→T4) dies at the source. Not collected by pytest (no
test_ prefix).

The suites these helpers serve are ORCHESTRATOR-AUTHORED red suites
(tdd-orchestration skill): they are the specification. Implementor
sub-agents make them green; they never weaken assertions here.
"""

from pathlib import Path

import pytest

from openalph.config import ConfigError, load_config

# Repo layout anchor: tests/ -> repo root -> src/openalph
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "openalph"

# Minimal valid agent TOML — the same shape the kdsn.305 integration suite
# uses for [context] parsing tests.
BASE_TOML = '''
[agent]
name = "gc-agent"
default_model = "anthropic/claude-sonnet-4-20250514"
max_tokens = 8192

[providers.anthropic]
type = "anthropic"
api_key = "sk-test"

[workspace]
path = "/tmp/test"
'''


def write_and_load(tmp_path, body):
    """Write `body` as an agent TOML and load it through the public loader."""
    p = tmp_path / "agent.toml"
    p.write_text(body)
    return load_config(p)


def expect_context_error(tmp_path, context_block, needle=None):
    """Assert a [context] block raises ConfigError (fail-loud discipline).

    When `needle` is given, the error message must contain it — steering
    assertions use substrings of the NEW key names, never full sentences
    (steering wording is allowed to churn; the new key name is the contract).
    Returns the ConfigError for further assertion.
    """
    with pytest.raises(ConfigError) as ei:
        write_and_load(tmp_path, BASE_TOML + context_block)
    if needle is not None:
        assert needle in str(ei.value), (
            f"error must steer to {needle!r}, got: {ei.value}")
    return ei.value


def scan_src_for_tokens(tokens, suffix=".py"):
    """Find occurrences of `tokens` under src/openalph.

    Returns a list of (relative_path, lineno, line) hits — empty means the
    hard-epoch absence criterion holds. Used by per-slice absence pins;
    each slice pins only the spellings its own fiber retires (see the
    suite docstrings for which slice pins what).
    """
    hits = []
    for p in sorted(SRC_ROOT.rglob(f"*{suffix}")):
        if not p.is_file():
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            for tok in tokens:
                if tok in line:
                    hits.append(
                        (str(p.relative_to(SRC_ROOT)), i, line.strip()))
                    break
    return hits


# ---------------------------------------------------------------------------
# Session/entry builders shared by the strip + checkpoint + toolslash suites
# (compact port of the kdsn.305 suites' builder style)
# ---------------------------------------------------------------------------

ROOM = "!handoff:matrix.local"
AGENT_ID = "@handoff-agent:matrix.local"


def user(content, source=None, **kw):
    d = {"role": "user", "content": content}
    if source is not None:
        d["source"] = source
    d.update(kw)
    return d


def assistant(content="", tool_calls=None, thinking=None, **kw):
    d = {"role": "assistant", "content": content}
    if tool_calls is not None:
        d["tool_calls"] = tool_calls
    if thinking is not None:
        d["thinking"] = thinking
    d.update(kw)
    return d


def tc(call_id, name, input=None):
    return {"call_id": call_id, "name": name,
            "input": input if input is not None else {}}


def tool(call_id, name, output, is_error=False, **kw):
    d = {"role": "tool", "call_id": call_id, "name": name,
         "output": output}
    if is_error:
        d["is_error"] = True
    d.update(kw)
    return d


def make_log(tmp_path, handoff_default=True):
    """SessionLog wired for the handoff render (kwarg renamed at T1)."""
    from openalph.session import SessionLog
    return SessionLog(tmp_path, AGENT_ID, handoff_default=handoff_default)


def append_all(log, entries, room=ROOM):
    for e in entries:
        role = e["role"]
        kw = {k: v for k, v in e.items() if k not in ("role",)}
        log.append(role=role, sender=AGENT_ID, room=room, **kw)


def marker(log, event, entry_index, detail=None, room=ROOM):
    """Append a boundary-style system marker (old or new event name)."""
    kw = {"event": event, "entry_index": entry_index}
    if detail is not None:
        kw["detail"] = detail
    log.append(role="system", sender=AGENT_ID, room=room, **kw)


def parse_ts(ts):
    """Parse a session ts ("...Z" ISO 8601 UTC) to an epoch float."""
    from datetime import datetime, timezone
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc).timestamp()
