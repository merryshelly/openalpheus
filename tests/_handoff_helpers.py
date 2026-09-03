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
