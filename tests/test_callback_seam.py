"""Characterization + spec tests for the callback seam extraction (kdsn.237 Phase 0).

CHARACTERIZATION (pin current behavior — must pass before AND after refactor):
  TestCallbacksDictKeys — exact key set + value types from _build_agent_callbacks.
  The broader characterization harness is the existing suite: test_context_status.py,
  test_advisor_integration.py, test_advisor_remediation.py, test_cache_keepalive.py —
  all call _build_agent_callbacks / _build_context_status and exercise the wiring.

SPEC (new shared functions — RED until openalph.callbacks is implemented):
  TestBuildContextStatusHeadless — build_context_status works without MatrixBot.
  TestBuildCallbacksHeadless — build_callbacks works without MatrixBot.
  TestCommsSinksProtocol — MatrixSinks satisfies the CommsSinks protocol.

Design spec: memory/projects/openalph/specs/cli-firstclass-and-matrix-decoupling.md
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from openalph.matrix import MatrixBot
from openalph.config import MatrixConfig, AgentConfig, ProviderConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_matrix_config(**kwargs):
    defaults = dict(
        homeserver="https://matrix.local",
        user_id="@merry:matrix.local",
        device_id="TEST",
        password="test-password",
        access_token=None,
        context_reserve=16384,
        sync_timeout=30000,
        retry_base=1,
        retry_max=10,
    )
    defaults.update(kwargs)
    return MatrixConfig(**defaults)


def _make_agent(**overrides):
    """Mock agent with the attributes build_callbacks reads."""
    agent = MagicMock()
    agent.system_prompt = "test system prompt"
    agent.history = MagicMock(return_value=[{"role": "user", "content": "hi"}])
    agent.status = MagicMock(return_value={
        "name": "test-agent",
        "model": "anthropic/claude-sonnet-4",
        "turns": 3,
        "context_tokens": 5000,
        "context_max": 200000,
        "context_pct": 2,
    })
    agent._read_registries = {}
    agent._advisor_uses = {}
    for key, val in overrides.items():
        setattr(agent, key, val)
    return agent


def _make_bot(agent=None, **overrides):
    """Construct a MatrixBot via __new__ with minimal attributes for callback tests."""
    bot = MatrixBot.__new__(MatrixBot)
    bot.config = _make_matrix_config()
    bot.agent = agent or _make_agent()
    bot.client = MagicMock()
    bot.session_log = None
    bot.heartbeat = None
    bot.umbral = None
    bot._advisor_results = {}
    bot._subagent_results = {}
    for key, val in overrides.items():
        setattr(bot, key, val)
    return bot


# The exact key set the tools layer expects (the wire format contract).
# Pinning this proves the refactor doesn't change what tools see.
_EXPECTED_CALLBACK_KEYS = frozenset({
    "send_media",
    "on_redaction",
    "on_keepalive_miss",
    "on_degenerate",
    "context_status",
    "send_notice",
    "log_reminder",
    "turn_source",
    "read_registry",
    "room_id",
    "get_transcript",
    "advisor_uses",
    "advisor_results",
    "subagent_results",
})

_SIDE_EFFECT_KEYS = frozenset({
    "send_media", "on_redaction", "on_keepalive_miss", "on_degenerate",
    "send_notice", "log_reminder",
})

_AGENT_STATE_KEYS = frozenset({
    "context_status", "turn_source", "read_registry", "room_id",
    "get_transcript", "advisor_uses", "advisor_results", "subagent_results",
})


# ---------------------------------------------------------------------------
# CHARACTERIZATION: callbacks dict key set + value types
# ---------------------------------------------------------------------------

class TestCallbacksDictKeys:
    """Pin the exact wire format of _build_agent_callbacks.

    After the refactor, MatrixBot._build_agent_callbacks delegates to
    callbacks.build_callbacks — this test proves the returned dict is
    structurally identical (same keys, same value types).
    """

    def test_exact_key_set(self):
        """The callbacks dict has exactly the expected 14 keys — no more, no less."""
        bot = _make_bot()
        cb = bot._build_agent_callbacks("!room:server", None)
        assert set(cb.keys()) == set(_EXPECTED_CALLBACK_KEYS)

    def test_side_effect_keys_are_callable(self):
        """All 6 side-effect keys are async callables (closures delegating to sinks)."""
        bot = _make_bot()
        cb = bot._build_agent_callbacks("!room:server", None)
        for key in _SIDE_EFFECT_KEYS:
            assert callable(cb[key]), f"side-effect key {key!r} must be callable"

    def test_get_transcript_is_callable(self):
        bot = _make_bot()
        cb = bot._build_agent_callbacks("!room:server", None)
        assert callable(cb["get_transcript"])

    def test_context_status_is_callable(self):
        bot = _make_bot()
        cb = bot._build_agent_callbacks("!room:server", None)
        assert callable(cb["context_status"])

    def test_turn_source_passes_through(self):
        """turn_source is the raw value passed to the builder (None or 'heartbeat'/'umbral')."""
        bot = _make_bot()
        cb_none = bot._build_agent_callbacks("!room:server", None)
        assert cb_none["turn_source"] is None

        cb_hb = bot._build_agent_callbacks("!room:server", "heartbeat")
        assert cb_hb["turn_source"] == "heartbeat"

    def test_room_id_passes_through(self):
        bot = _make_bot()
        cb = bot._build_agent_callbacks("!myroom:server", None)
        assert cb["room_id"] == "!myroom:server"

    def test_read_registry_is_dict(self):
        bot = _make_bot()
        cb = bot._build_agent_callbacks("!room:server", None)
        assert isinstance(cb["read_registry"], dict)

    def test_advisor_uses_is_dict(self):
        bot = _make_bot()
        cb = bot._build_agent_callbacks("!room:server", None)
        assert isinstance(cb["advisor_uses"], dict)

    def test_advisor_results_is_dict(self):
        bot = _make_bot()
        cb = bot._build_agent_callbacks("!room:server", None)
        assert isinstance(cb["advisor_results"], dict)

    def test_subagent_results_is_dict(self):
        bot = _make_bot()
        cb = bot._build_agent_callbacks("!room:server", None)
        assert isinstance(cb["subagent_results"], dict)

    def test_get_transcript_returns_prompt_and_history_copy(self):
        """get_transcript returns (system_prompt, list(history)) — a copy, not the live list."""
        bot = _make_bot()
        cb = bot._build_agent_callbacks("!room:server", None)
        prompt, history = cb["get_transcript"]()
        assert prompt == "test system prompt"
        assert history == [{"role": "user", "content": "hi"}]
        # Must be a copy — mutating it must not affect agent.history()
        history.append({"role": "assistant", "content": "new"})
        assert bot.agent.history.call_args is not None  # history() was called
        assert len(bot.agent.history.return_value) == 1  # original unchanged

    def test_read_registry_is_per_room(self):
        """Two different rooms get different read_registry dicts."""
        bot = _make_bot()
        cb_a = bot._build_agent_callbacks("!roomA:server", None)
        cb_b = bot._build_agent_callbacks("!roomB:server", None)
        assert cb_a["read_registry"] is not cb_b["read_registry"]

    def test_read_registry_is_stable_for_same_room(self):
        """Same room gets the same read_registry dict across calls (setdefault semantics)."""
        bot = _make_bot()
        cb1 = bot._build_agent_callbacks("!room:server", None)
        cb2 = bot._build_agent_callbacks("!room:server", None)
        assert cb1["read_registry"] is cb2["read_registry"]


# ---------------------------------------------------------------------------
# SPEC: build_context_status works headless (RED until callbacks.py exists)
# ---------------------------------------------------------------------------

class TestBuildContextStatusHeadless:
    """build_context_status(agent, room_id, ...) works without MatrixBot or nio.

    This is the hoisted function — reads agent state + session_log + timer
    managers. room_name is injected by the caller (no nio client lookup).
    """

    def test_importable_from_callbacks(self):
        """build_context_status is importable from openalph.callbacks."""
        from openalph.callbacks import build_context_status
        assert callable(build_context_status)

    def test_works_with_no_optional_args(self):
        """Functions with only agent + room_id — all optional args default to absent."""
        from openalph.callbacks import build_context_status
        agent = _make_agent()
        data = build_context_status(agent, "!room:server")
        assert data["name"] == "test-agent"
        assert data["room_id"] == "!room:server"
        assert data["room_name"] is None
        assert data["session_age_minutes"] is None
        assert data["heartbeat_active"] is False
        assert data["umbral_active"] is False

    def test_room_name_injected_by_caller(self):
        """room_name is a caller-supplied parameter, not resolved from a Matrix client."""
        from openalph.callbacks import build_context_status
        agent = _make_agent()
        data = build_context_status(agent, "!room:server", room_name="my-label")
        assert data["room_name"] == "my-label"

    def test_room_name_none_by_default(self):
        from openalph.callbacks import build_context_status
        agent = _make_agent()
        data = build_context_status(agent, "!room:server")
        assert data["room_name"] is None

    def test_heartbeat_fields_when_active(self):
        from openalph.callbacks import build_context_status
        agent = _make_agent()
        mock_entry = MagicMock()
        mock_entry.room_id = "!room:server"
        mock_entry.interval_seconds = 300
        mock_entry.seconds_until_next = 120
        mock_hb = MagicMock()
        mock_hb.is_active.return_value = True
        mock_hb.status.return_value = [mock_entry]
        data = build_context_status(agent, "!room:server", heartbeat=mock_hb)
        assert data["heartbeat_active"] is True
        assert data["heartbeat_interval_minutes"] == 5  # 300/60
        assert data["heartbeat_next_minutes"] == 2  # round(120/60)

    def test_umbral_fields_when_active(self):
        from openalph.callbacks import build_context_status
        agent = _make_agent()
        mock_entry = MagicMock()
        mock_entry.room_id = "!room:server"
        mock_entry.interval_seconds = 14400
        mock_entry.seconds_until_next = 10800
        mock_um = MagicMock()
        mock_um.is_active.return_value = True
        mock_um.status.return_value = [mock_entry]
        data = build_context_status(agent, "!room:server", umbral=mock_um)
        assert data["umbral_active"] is True
        assert data["umbral_interval_minutes"] == 240
        assert data["umbral_next_minutes"] == 180

    def test_session_age_from_session_log(self):
        """session_age_minutes is computed from the first entry in session_log."""
        from openalph.callbacks import build_context_status
        from datetime import datetime, timezone, timedelta
        agent = _make_agent()
        # First entry 10 minutes ago
        first_ts = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        mock_sl = MagicMock()
        mock_sl.build_context.return_value = None
        mock_sl.read.return_value = [{"ts": first_ts}]
        data = build_context_status(agent, "!room:server", session_log=mock_sl)
        assert data["session_age_minutes"] is not None
        assert 9 <= data["session_age_minutes"] <= 11  # ~10 min (allow test latency)

    def test_session_age_none_when_no_entries(self):
        from openalph.callbacks import build_context_status
        agent = _make_agent()
        mock_sl = MagicMock()
        mock_sl.build_context.return_value = None
        mock_sl.read.return_value = []
        data = build_context_status(agent, "!room:server", session_log=mock_sl)
        assert data["session_age_minutes"] is None

    def test_matches_matrixbot_output_for_same_inputs(self):
        """The hoisted function produces the same dict as MatrixBot._build_context_status
        for the same agent + session_log + heartbeat + umbral + room_name.

        This is the core behavior-preservation assertion: MatrixBot delegates to
        build_context_status, so their outputs must be identical.
        """
        from openalph.callbacks import build_context_status
        agent = _make_agent()

        # MatrixBot path (room not in client → room_name=None)
        bot = _make_bot(agent=agent)
        bot.client.rooms.get = MagicMock(return_value=None)
        bot_result = bot._build_context_status("!room:server")

        # Shared function path (room_name=None, same agent)
        shared_result = build_context_status(
            agent, "!room:server",
            session_log=None, room_name=None, heartbeat=None, umbral=None,
        )

        assert bot_result == shared_result


# ---------------------------------------------------------------------------
# SPEC: build_callbacks works headless (RED until callbacks.py exists)
# ---------------------------------------------------------------------------

class TestBuildCallbacksHeadless:
    """build_callbacks(agent, room_id, sinks, ...) works without MatrixBot.

    Agent-state callbacks read from agent. Side-effects route through sinks.
    Returns the same-keyed dict the tools expect (wire format unchanged).
    """

    def test_importable_from_callbacks(self):
        from openalph.callbacks import build_callbacks
        assert callable(build_callbacks)

    def test_returns_expected_key_set(self):
        """The headless builder returns the exact same key set as MatrixBot."""
        from openalph.callbacks import build_callbacks
        agent = _make_agent()
        sinks = MagicMock()  # duck-typed — any object with the sink methods
        cb = build_callbacks(agent, "!room:server", sinks, turn_source=None)
        assert set(cb.keys()) == set(_EXPECTED_CALLBACK_KEYS)

    def test_turn_source_passes_through(self):
        from openalph.callbacks import build_callbacks
        agent = _make_agent()
        sinks = MagicMock()
        cb = build_callbacks(agent, "!room:server", sinks, turn_source="heartbeat")
        assert cb["turn_source"] == "heartbeat"

    def test_room_id_passes_through(self):
        from openalph.callbacks import build_callbacks
        agent = _make_agent()
        sinks = MagicMock()
        cb = build_callbacks(agent, "!myroom:server", sinks, turn_source=None)
        assert cb["room_id"] == "!myroom:server"

    def test_get_transcript_works(self):
        from openalph.callbacks import build_callbacks
        agent = _make_agent()
        sinks = MagicMock()
        cb = build_callbacks(agent, "!room:server", sinks, turn_source=None)
        prompt, history = cb["get_transcript"]()
        assert prompt == "test system prompt"
        assert history == [{"role": "user", "content": "hi"}]

    def test_read_registry_is_dict(self):
        from openalph.callbacks import build_callbacks
        agent = _make_agent()
        sinks = MagicMock()
        cb = build_callbacks(agent, "!room:server", sinks, turn_source=None)
        assert isinstance(cb["read_registry"], dict)

    def test_advisor_uses_is_dict(self):
        from openalph.callbacks import build_callbacks
        agent = _make_agent()
        sinks = MagicMock()
        cb = build_callbacks(agent, "!room:server", sinks, turn_source=None)
        assert isinstance(cb["advisor_uses"], dict)

    def test_advisor_results_is_dict(self):
        from openalph.callbacks import build_callbacks
        agent = _make_agent()
        sinks = MagicMock()
        cb = build_callbacks(agent, "!room:server", sinks, turn_source=None,
                             advisor_results={})
        assert isinstance(cb["advisor_results"], dict)

    def test_subagent_results_is_dict(self):
        from openalph.callbacks import build_callbacks
        agent = _make_agent()
        sinks = MagicMock()
        cb = build_callbacks(agent, "!room:server", sinks, turn_source=None,
                             subagent_results={})
        assert isinstance(cb["subagent_results"], dict)

    def test_context_status_callback_works_headless(self):
        """The context_status callback returns a dict without a MatrixBot."""
        from openalph.callbacks import build_callbacks
        agent = _make_agent()
        sinks = MagicMock()
        cb = build_callbacks(agent, "!room:server", sinks, turn_source=None)
        result = asyncio.run(cb["context_status"]())
        assert isinstance(result, dict)
        assert result["room_id"] == "!room:server"
        assert result["name"] == "test-agent"

    def test_side_effects_route_through_sinks(self):
        """Calling a side-effect callback delegates to the corresponding sinks method."""
        from openalph.callbacks import build_callbacks
        agent = _make_agent()
        sinks = MagicMock()
        sinks.send_notice = AsyncMock()
        cb = build_callbacks(agent, "!room:server", sinks, turn_source=None)
        asyncio.run(cb["send_notice"]("!room:server", "test notice"))
        sinks.send_notice.assert_awaited_once_with("!room:server", "test notice")

    def test_send_media_routes_through_sinks(self):
        from openalph.callbacks import build_callbacks
        agent = _make_agent()
        sinks = MagicMock()
        sinks.send_media = AsyncMock()
        cb = build_callbacks(agent, "!room:server", sinks, turn_source=None)
        asyncio.run(cb["send_media"]("/path/file.txt", "text/plain", "file.txt"))
        sinks.send_media.assert_awaited_once()


# ---------------------------------------------------------------------------
# SPEC: CommsSinks protocol + MatrixSinks (RED until callbacks.py exists)
# ---------------------------------------------------------------------------

class TestCommsSinksProtocol:
    """The CommsSinks protocol defines the side-effect contract.

    MatrixSinks implements it via real Matrix delivery.
    HeadlessSinks (Phase 1) implements it via stdout/no-op.
    """

    def test_comms_sinks_importable(self):
        from openalph.callbacks import CommsSinks
        assert CommsSinks is not None

    def test_matrix_sinks_satisfies_protocol(self):
        """MatrixSinks structurally satisfies CommsSinks (has all required methods)."""
        from openalph.callbacks import CommsSinks, MatrixSinks
        # Structural check: MatrixSinks must have all the protocol's methods
        protocol_methods = {
            attr for attr in dir(CommsSinks)
            if not attr.startswith("_") and callable(getattr(CommsSinks, attr, None))
        }
        # Filter to just the async method names we defined
        expected_methods = {
            "send_notice", "log_reminder", "send_media",
            "on_redaction", "on_keepalive_miss", "on_degenerate",
        }
        for method in expected_methods:
            assert hasattr(MatrixSinks, method), f"MatrixSinks missing {method}"


# ---------------------------------------------------------------------------
# SPEC: MatrixBot delegates to shared functions (proves the refactor)
# ---------------------------------------------------------------------------

class TestMatrixBotDelegation:
    """After the refactor, MatrixBot._build_agent_callbacks and _build_context_status
    delegate to the shared functions. This test proves the delegation is wired.

    RED until the refactor is done — then green because the methods delegate.
    """

    def test_build_agent_callbacks_delegates_to_shared(self):
        """MatrixBot._build_agent_callbacks calls callbacks.build_callbacks under the hood."""
        from openalph.callbacks import build_callbacks
        bot = _make_bot()
        with patch("openalph.matrix.build_callbacks", wraps=build_callbacks) as mock_bc:
            cb = bot._build_agent_callbacks("!room:server", "heartbeat")
            assert mock_bc.called
            assert set(cb.keys()) == set(_EXPECTED_CALLBACK_KEYS)

    def test_build_context_status_delegates_to_shared(self):
        """MatrixBot._build_context_status calls callbacks.build_context_status under the hood."""
        from openalph.callbacks import build_context_status
        bot = _make_bot()
        bot.client.rooms.get = MagicMock(return_value=None)
        with patch("openalph.matrix.build_context_status", wraps=build_context_status) as mock_bcs:
            data = bot._build_context_status("!room:server")
            assert mock_bcs.called
            assert data["room_id"] == "!room:server"
