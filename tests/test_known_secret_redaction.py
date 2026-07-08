"""RED suite — L1 value-based known-secret redaction (op-handling design).

Authored from op-handling-test-plan.md (the spec), tests 1–18.  TESTS ONLY —
no source is touched.  The feature under test does NOT exist yet:

    tools/security.py :: redact_known_secrets(text, known_values, min_length=12)
    tools/__init__.py :: _collect_known_secrets(agent_config) -> set[str]

RED honesty: the two new symbols are imported INSIDE each test body (via the
_load_* helpers), never at module top — a module-level import of a missing
symbol is a collection error that kills the whole module (an invalid RED).
Import-inside-test yields a legible per-test ImportError, the correct RED
signal.  Existing symbols (execute_tool, ToolResult, RedactionEvent,
AgentConfig, ProviderConfig) ARE imported at top; they exist today.

Value pass semantics (spec §L1):
  * redacts every literal occurrence of each known value with
    len(value) >= min_length, replacing with label [REDACTED:known_secret];
  * runs AFTER the pattern pass, on already-pattern-redacted text (so shaped
    keys are already [REDACTED:api_key] and the value pass is the backstop
    for secrets the patterns MISS, e.g. an arbitrary password);
  * never logs the value; RedactionEvent records char_count/position only;
  * no-op fast paths: empty text -> (text, []); empty known_values -> (text, []).

Remediation round 1 (see tmp/op-build/remediation-spec.md §B): L1 was
reordered to run the value pass BEFORE the pattern pass (so a known secret is
redacted whole before the pattern pass could fragment it).  Two tests below
are changed accordingly (explicitly authorized spec refinement):
  * test_execute_tool_pattern_still_wins_for_shaped_keys was renamed/repurposed
    to test_execute_tool_value_pass_redacts_known_shaped_key (value-first now
    labels a KNOWN shaped key as known_secret, not api_key).
  * test_execute_tool_pattern_catches_unknown_shaped_secret was added (proves
    the pattern pass still catches an UNKNOWN shaped secret not in the
    known-set, i.e. value-first did not weaken pattern coverage).
"""

import dataclasses
import logging
import time

import pytest

from openalph.config import AgentConfig, ProviderConfig
from openalph.tools import ToolResult, execute_tool
from openalph.tools.security import RedactionEvent


# --- New-symbol loaders (import-inside-test → legible per-test RED) ---------

def _load_redact_known_secrets():
    from openalph.tools.security import redact_known_secrets
    return redact_known_secrets


def _load_collect_known_secrets():
    from openalph.tools import _collect_known_secrets
    return _collect_known_secrets


# --- Fixtures / helpers -----------------------------------------------------

# A long, high-entropy value that matches NONE of the CREDENTIAL_PATTERNS
# (no sk-/ghp_/ops_/Bearer/0x-hex shape) — so the pattern pass leaves it
# untouched and only the L1 value pass can redact it.  Fake, not a real secret.
FAKE_SECRET = "Zx9-not-a-pattern-sudo-like-secret-8823"  # 39 chars


@pytest.fixture(autouse=True)
def _clean_api_key_cache():
    """Snapshot + restore the module-global _api_key_cache around every test.

    _api_key_cache leaking a seeded value into other test modules is a real
    regression class (spec flags it); this guarantees teardown emptiness.
    """
    from openalph.tools import _api_key_cache
    snapshot = dict(_api_key_cache)
    try:
        yield
    finally:
        _api_key_cache.clear()
        _api_key_cache.update(snapshot)


def _make_config(tmp_path, providers):
    """Minimal AgentConfig (mirrors test_credential_redaction.py fixture)."""
    return AgentConfig(
        name="test-known-secret",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers=providers,
        workspace=tmp_path,
    )


def _provider(api_key, key="anthropic", type="anthropic"):
    return ProviderConfig(key=key, type=type, api_key=api_key,
                          base_url=None, quirks=[])


# ===========================================================================
# Unit — pure redact_known_secrets  (tests 1–11)
# ===========================================================================

class TestRedactKnownSecretsUnit:

    def test_redacts_exact_known_value(self):
        """1: a known value present once → replaced with [REDACTED:known_secret]; 1 event."""
        redact_known_secrets = _load_redact_known_secrets()
        text = f"the resolved value is {FAKE_SECRET} in the output"
        out, events = redact_known_secrets(text, {FAKE_SECRET})
        assert FAKE_SECRET not in out
        assert "[REDACTED:known_secret]" in out
        assert len(events) == 1

    def test_redacts_all_occurrences(self):
        """2: value present 3× → all replaced (assert 3 label substrings)."""
        redact_known_secrets = _load_redact_known_secrets()
        text = f"{FAKE_SECRET} then {FAKE_SECRET} then {FAKE_SECRET}"
        out, events = redact_known_secrets(text, {FAKE_SECRET})
        assert FAKE_SECRET not in out
        assert out.count("[REDACTED:known_secret]") == 3

    def test_redacts_multiple_distinct_secrets(self):
        """3: two distinct known values both present → both redacted."""
        redact_known_secrets = _load_redact_known_secrets()
        s1 = FAKE_SECRET
        s2 = "Qw7-another-distinct-long-secret-value-4410"
        text = f"first {s1} and second {s2} at end"
        out, events = redact_known_secrets(text, {s1, s2})
        assert s1 not in out
        assert s2 not in out
        assert out.count("[REDACTED:known_secret]") == 2

    def test_redacts_value_as_substring(self):
        """4: value embedded inside a larger token → still redacted (backstop)."""
        redact_known_secrets = _load_redact_known_secrets()
        text = f"PREFIX{FAKE_SECRET}SUFFIX"
        out, events = redact_known_secrets(text, {FAKE_SECRET})
        assert FAKE_SECRET not in out
        assert "[REDACTED:known_secret]" in out
        assert "PREFIX" in out and "SUFFIX" in out

    def test_skips_below_min_length(self):
        """5: a known value shorter than min_length (default 12) → NOT redacted."""
        redact_known_secrets = _load_redact_known_secrets()
        short = "short"  # 5 chars < 12
        text = f"the value is {short} here"
        out, events = redact_known_secrets(text, {short})
        assert out == text
        assert events == []

    def test_empty_known_set_noop(self):
        """6: empty known_values → text unchanged, no events (works-without-1Password)."""
        redact_known_secrets = _load_redact_known_secrets()
        text = "some tool output with nothing sensitive"
        out, events = redact_known_secrets(text, set())
        assert out == text
        assert events == []

    def test_empty_text_noop(self):
        """7: empty text → ('', [])."""
        redact_known_secrets = _load_redact_known_secrets()
        out, events = redact_known_secrets("", {FAKE_SECRET})
        assert out == ""
        assert events == []

    def test_no_known_value_present(self):
        """8: text contains none of the known values → unchanged, no events."""
        redact_known_secrets = _load_redact_known_secrets()
        text = "totally unrelated benign output with no secrets"
        out, events = redact_known_secrets(text, {FAKE_SECRET})
        assert out == text
        assert events == []

    def test_never_logs_value(self, caplog):
        """9: the secret string must not appear in any log record; a warning IS emitted."""
        redact_known_secrets = _load_redact_known_secrets()
        text = f"value {FAKE_SECRET} present"
        with caplog.at_level(logging.WARNING, logger="openalph.security"):
            redact_known_secrets(text, {FAKE_SECRET})
        for record in caplog.records:
            assert FAKE_SECRET not in record.getMessage(), \
                "secret value leaked into a log message"
            assert FAKE_SECRET not in str(record.args), \
                "secret value leaked into log args"
        security_records = [r for r in caplog.records if r.name == "openalph.security"]
        assert security_records, "a redaction WARNING must be emitted"
        assert any("chars" in r.getMessage() for r in security_records), \
            "log line must record the char count (…chars…), never the value"

    def test_event_shape_no_value_field(self):
        """10: RedactionEvent carries char_count/position/label; the value is absent."""
        redact_known_secrets = _load_redact_known_secrets()
        text = f"xx {FAKE_SECRET} yy"
        out, events = redact_known_secrets(text, {FAKE_SECRET})
        assert len(events) >= 1
        ev = events[0]
        assert isinstance(ev, RedactionEvent)
        assert isinstance(ev.char_count, int)
        assert isinstance(ev.position, int)
        assert isinstance(ev.redaction_label, str)
        for fld in dataclasses.fields(ev):
            assert FAKE_SECRET not in str(getattr(ev, fld.name)), \
                f"secret leaked into RedactionEvent.{fld.name}"

    def test_custom_min_length(self):
        """11: min_length=20 → a 15-char known value kept, a 25-char one redacted."""
        redact_known_secrets = _load_redact_known_secrets()
        short_secret = "Short-val-12345"          # 15 chars
        long_secret = "Longer-secret-value-33221"  # 25 chars
        assert len(short_secret) == 15 and len(long_secret) == 25
        text = f"one {short_secret} two {long_secret} three"
        out, events = redact_known_secrets(
            text, {short_secret, long_secret}, min_length=20)
        assert short_secret in out          # below threshold → kept
        assert long_secret not in out       # above threshold → redacted
        assert out.count("[REDACTED:known_secret]") == 1


# ===========================================================================
# Collector — _collect_known_secrets  (tests 12–16)
# ===========================================================================

class TestCollectKnownSecrets:

    def test_collects_provider_keys(self, tmp_path):
        """12: agent_config with 2 providers (long api_keys) → both in set."""
        _collect = _load_collect_known_secrets()
        k1 = "provider-one-long-secret-aaaa1111"
        k2 = "provider-two-long-secret-bbbb2222"
        cfg = _make_config(tmp_path, {
            "anthropic": _provider(k1, key="anthropic", type="anthropic"),
            "openai": _provider(k2, key="openai", type="openai"),
        })
        known = _collect(cfg)
        assert k1 in known
        assert k2 in known

    def test_collects_api_key_cache(self, tmp_path):
        """13: a resolved value seeded into _api_key_cache → present in the set."""
        _collect = _load_collect_known_secrets()
        from openalph.tools import _api_key_cache
        cached = "cached-resolved-secret-value-7777"
        _api_key_cache["op read op://x/item/field"] = (cached, time.monotonic() + 3600)
        cfg = _make_config(tmp_path, {})
        known = _collect(cfg)
        assert cached in known

    def test_unions_and_dedups(self, tmp_path):
        """14: same value in both sources → appears once (set-typed union)."""
        _collect = _load_collect_known_secrets()
        from openalph.tools import _api_key_cache
        shared = "shared-secret-in-both-sources-9999"
        _api_key_cache["op read op://a/b/c"] = (shared, time.monotonic() + 3600)
        cfg = _make_config(tmp_path, {"anthropic": _provider(shared)})
        known = _collect(cfg)
        assert isinstance(known, set)
        assert shared in known

    def test_skips_empty_provider_keys(self, tmp_path):
        """15: a provider with api_key='' → empty string not in the set."""
        _collect = _load_collect_known_secrets()
        real = "real-provider-secret-value-5555"
        cfg = _make_config(tmp_path, {
            "empty": _provider("", key="empty"),
            "real": _provider(real, key="real"),
        })
        known = _collect(cfg)
        assert "" not in known
        assert real in known

    def test_empty_when_no_sources(self, tmp_path):
        """16: no providers + empty cache → set()."""
        _collect = _load_collect_known_secrets()
        cfg = _make_config(tmp_path, {})
        known = _collect(cfg)
        assert known == set()


# ===========================================================================
# Integration — real execute_tool redaction pipeline  (tests 17–19)
# NOTE: execute_tool is NOT mocked — it is the code under test.  echo is a
# real subprocess; the secrets are fake.
# ===========================================================================

class TestExecuteToolKnownSecretRedaction:

    @pytest.mark.asyncio
    async def test_execute_tool_redacts_known_secret_in_output(self, tmp_path):
        """17: provider api_key is a long non-pattern secret; echo it via a real
        shell subprocess; execute_tool output must be [REDACTED:known_secret]."""
        cfg = _make_config(tmp_path, {"anthropic": _provider(FAKE_SECRET)})
        result = await execute_tool(
            name="shell",
            input={"command": f"echo '{FAKE_SECRET}'"},
            tool_config={"default_timeout": 30, "max_output": 50000},
            agent_config=cfg,
        )
        assert isinstance(result, ToolResult)
        assert FAKE_SECRET not in result.content, \
            "raw known secret must not survive into the tool result"
        assert "[REDACTED:known_secret]" in result.content

    @pytest.mark.asyncio
    async def test_execute_tool_value_pass_redacts_known_shaped_key(self, tmp_path):
        """Renamed/repurposed from test_execute_tool_pattern_still_wins_for_shaped_keys
        (remediation spec §B): a provider key with a recognised SHAPE (sk-ant-…) that
        is ALSO a known value (≥12 chars) — under the new value-FIRST ordering, the
        value pass now labels it [REDACTED:known_secret] (not [REDACTED:api_key]).
        Fully redacted either way; this test tracks the new label."""
        _collect = _load_collect_known_secrets()
        shaped = "sk-ant-" + "fixtureAAAA1111bbbb2222"  # sk-ant- shape, well over 12 chars
        cfg = _make_config(tmp_path, {"anthropic": _provider(shaped)})
        # The shaped key IS a known value…
        assert shaped in _collect(cfg)
        result = await execute_tool(
            name="shell",
            input={"command": f"echo '{shaped}'"},
            tool_config={"default_timeout": 30, "max_output": 50000},
            agent_config=cfg,
        )
        # …and under value-first ordering, the value pass wins: labeled
        # known_secret (the raw key is absent from result.content either way).
        assert shaped not in result.content
        assert "[REDACTED:known_secret]" in result.content

    @pytest.mark.asyncio
    async def test_execute_tool_pattern_catches_unknown_shaped_secret(self, tmp_path):
        """New (remediation spec §B): provider api_key = FAKE_SECRET (non-shaped,
        known). Tool output ALSO contains a DIFFERENT shaped secret (sk-ant- + 24
        chars) that is NOT in the known-set — the pattern pass must still redact
        it as [REDACTED:api_key], proving value-first did not weaken pattern
        coverage for secrets the known-set doesn't contain."""
        unknown_shaped = "sk-ant-" + "z9y8x7w6v5u4t3s2r1q0p9o8"  # 24 chars, NOT the provider key
        assert len(unknown_shaped) - len("sk-ant-") == 24
        assert unknown_shaped != FAKE_SECRET
        cfg = _make_config(tmp_path, {"anthropic": _provider(FAKE_SECRET)})
        result = await execute_tool(
            name="shell",
            input={"command": f"echo '{unknown_shaped}'"},
            tool_config={"default_timeout": 30, "max_output": 50000},
            agent_config=cfg,
        )
        assert unknown_shaped not in result.content, \
            "the unknown shaped secret must not survive into the tool result"
        assert "[REDACTED:api_key]" in result.content, \
            "the pattern pass must still catch a shaped secret NOT in the known-set"
