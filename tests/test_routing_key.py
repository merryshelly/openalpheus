"""Tests for the routing-key session-affinity header stamp (workspace-kdsn.353).

Contract under test — SB-ratified 2026-09-24 (default-on):

  1. Any openai-compatible provider may carry a routing-key header stamp:
     the existing salted sha256(room_id) derivation (the Fireworks
     x-session-affinity hint, design #5) sent as an extra HTTP header on
     /v1/chat/completions. Header-only — never a body field — so it is
     invisible to the model (D1-clean: never logged at info level either).
  2. Blackwell is DEFAULT-ON: a ``[providers.blackwell]`` block stamps
     ``X-SMG-Routing-Key: <affinity>`` on every call without any
     configuration. Kill flag: ``routing_key = false``. The header name is
     config-default ``X-SMG-Routing-Key`` but overridable via
     ``routing_key_header``.
  3. Any other provider (macstudio etc.) opts in with ``routing_key = true``
     — no code changes. All other providers stay off by default.
  4. Fireworks keeps its existing x-session-affinity path (user field +
     x-session-affinity header) byte-for-byte unchanged.
  5. room_id is the identity source: room_id=None -> NO header (never
     invent identity). Same room_id -> same key across calls/restarts
     (hardcoded SALT); different room_id -> different key.

  NOTE (subagent-room claim, verified 2026-09-24 by reading
  tools/subagent.py): FALSE. run_subagent() does not run subagents in
  distinct rooms — it drives complete() inline with NO room_id (main loop
  and circuit-breaker paths alike), so sub LLM calls carry room_id=None and
  get no stamp. There is no per-sub room identity to key on; per rule 5
  that is correct (no invented identity), and the fanout-distinctness claim
  is reported back to the SB rather than papered over.
"""

import hashlib
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openalph.config import AgentConfig, ProviderConfig, load_config
from openalph.provider import (
    DEFAULT_ROUTING_KEY_HEADER,
    SALT,
    complete,
    session_affinity_key,
    stream,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers (mirrors test_provider_stream.py conventions)
# ---------------------------------------------------------------------------

ROOM = "!room:tuwunel.local"
ROOM_B = "!other:tuwunel.local"


def make_provider(key="blackwell", type="openai", api_key="sk-test",
                  base_url="http://10.0.20.111:8000/v1", **extra):
    kwargs = dict(key=key, type=type, api_key=api_key, base_url=base_url)
    kwargs.update(extra)
    return ProviderConfig(**kwargs)


def make_config(**kwargs):
    defaults = {
        "name": "test-agent",
        "default_model": "blackwell/qwen38-27b-fp8",
        "max_tokens": 8192,
        "providers": {"blackwell": make_provider(key="blackwell")},
        "workspace": Path("/tmp/test"),
    }
    defaults.update(kwargs)
    return AgentConfig(**defaults)


class MockOpenAIStream:
    """Mock async-iterable for client.chat.completions.create results."""

    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._iter_impl()

    async def _iter_impl(self):
        for c in self._chunks:
            yield c


def _openai_text_chunk(content, finish_reason=None):
    choice = MagicMock()
    delta = MagicMock()
    delta.content = content
    delta.reasoning = None
    delta.reasoning_content = None
    delta.tool_calls = None
    choice.delta = delta
    choice.finish_reason = finish_reason
    chunk = MagicMock()
    chunk.id = "gen-test"
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


def _openai_usage_chunk(prompt_tokens=100, completion_tokens=50):
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens
    chunk = MagicMock()
    chunk.id = None
    chunk.choices = []
    chunk.usage = usage
    return chunk


async def collect_events(gen):
    async for event in gen:
        if event.type == "done":
            return event
    return None


def _chunks():
    return [
        _openai_text_chunk("ok", finish_reason="stop"),
        _openai_usage_chunk(),
    ]


def _openai_config(provider, default_model):
    """Single-provider openai config around `provider`."""
    return make_config(providers={provider.key: provider},
                       default_model=default_model)


def _mock_client_stream():
    """Fresh mock OpenAI client whose create() yields a 2-chunk stream."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=MockOpenAIStream(_chunks()),
    )
    return client


async def _drive_stream(config, room_id=None):
    """Drive stream() against a fresh mock client; return the mock."""
    client = _mock_client_stream()
    with patch("openalph.provider._get_client", return_value=client):
        await collect_events(stream(
            config=config, system="Test",
            messages=[{"role": "user", "content": "Hi"}],
            room_id=room_id,
        ))
    return client


# ---------------------------------------------------------------------------
# (a) Blackwell stamps the header BY DEFAULT — stream and complete
# ---------------------------------------------------------------------------


class TestBlackwellDefaultOn:

    @pytest.mark.asyncio
    async def test_blackwell_stream_stamps_header_by_default(self):
        """No routing_key config at all: blackwell stamps X-SMG-Routing-Key."""
        config = make_config()
        client = await _drive_stream(config, room_id=ROOM)

        kw = client.chat.completions.create.call_args.kwargs
        expected = session_affinity_key(ROOM)
        assert kw["extra_headers"][DEFAULT_ROUTING_KEY_HEADER] == expected
        # Header-only: never a body field, and the Fireworks `user` hint is
        # blackwell-foreign.
        assert "user" not in kw
        assert "x-session-affinity" not in kw["extra_headers"]

    @pytest.mark.asyncio
    async def test_blackwell_complete_stamps_header_by_default(self):
        """complete() delegates to stream() — the stamp rides along too."""
        config = make_config()
        client = _mock_client_stream()
        with patch("openalph.provider._get_client", return_value=client):
            await complete(
                config=config, system="Test",
                messages=[{"role": "user", "content": "Hi"}],
                room_id=ROOM,
            )

        kw = client.chat.completions.create.call_args.kwargs
        assert kw["extra_headers"][DEFAULT_ROUTING_KEY_HEADER] == \
            session_affinity_key(ROOM)


# ---------------------------------------------------------------------------
# (b) Kill flag
# ---------------------------------------------------------------------------


class TestKillFlag:

    @pytest.mark.asyncio
    async def test_routing_key_false_disables_blackwell_default(self):
        """Explicit routing_key=false is the kill flag (beats the default).
        Asserts the ROUTING-KEY header is absent — extra_headers itself may
        legitimately exist (the im7t.36.38 metrics-labels stamp is a
        separate default-on knob for blackwell)."""
        config = make_config(providers={
            "blackwell": make_provider(key="blackwell", routing_key=False),
        })
        client = await _drive_stream(config, room_id=ROOM)

        kw = client.chat.completions.create.call_args.kwargs
        assert DEFAULT_ROUTING_KEY_HEADER not in kw.get("extra_headers", {})

    @pytest.mark.asyncio
    async def test_kill_flag_beats_header_override(self):
        """routing_key=false wins even when a custom header name is set —
        the kill flag is about the STAMP, not the name."""
        config = _openai_config(
            make_provider(key="macstudio", routing_key=False,
                          routing_key_header="X-Custom"),
            "macstudio/glm-5.3-flash",
        )
        client = await _drive_stream(config, room_id=ROOM)

        kw = client.chat.completions.create.call_args.kwargs
        assert "extra_headers" not in kw


# ---------------------------------------------------------------------------
# (c) / (d) Deterministic per room
# ---------------------------------------------------------------------------


class TestKeyStability:

    def test_same_room_same_key(self):
        """Derivation is deterministic: same room_id -> same key."""
        assert session_affinity_key(ROOM) == session_affinity_key(ROOM)

    @pytest.mark.asyncio
    async def test_same_room_same_key_across_calls(self):
        """Two full stream() drives in the same room stamp identical keys."""
        config = make_config()
        values = []
        for _ in range(2):
            client = await _drive_stream(config, room_id=ROOM)
            values.append(
                client.chat.completions.create.call_args.kwargs
                ["extra_headers"][DEFAULT_ROUTING_KEY_HEADER],
            )
        assert values[0] == values[1]

    @pytest.mark.asyncio
    async def test_different_room_different_key(self):
        """Distinct rooms get distinct keys (sha256-collision odds)."""
        config = make_config()
        values = []
        for room in (ROOM, ROOM_B):
            client = await _drive_stream(config, room_id=room)
            values.append(
                client.chat.completions.create.call_args.kwargs
                ["extra_headers"][DEFAULT_ROUTING_KEY_HEADER],
            )
        assert values[0] != values[1]

    def test_key_is_32_lowercase_hex_of_salted_room(self):
        """The value IS the existing salted derivation (reuse, not a fork)."""
        expected = hashlib.sha256((SALT + ROOM).encode()).hexdigest()[:32]
        assert session_affinity_key(ROOM) == expected
        assert len(expected) == 32
        assert all(c in "0123456789abcdef" for c in expected)


# ---------------------------------------------------------------------------
# (e) room_id None -> no header (never invent identity)
# ---------------------------------------------------------------------------


class TestNoRoomNoHeader:

    @pytest.mark.asyncio
    async def test_room_id_none_no_header(self):
        config = make_config()
        client = await _drive_stream(config, room_id=None)

        kw = client.chat.completions.create.call_args.kwargs
        assert "extra_headers" not in kw
        assert "user" not in kw

    def test_helper_none_returns_none(self):
        assert session_affinity_key(None) is None


# ---------------------------------------------------------------------------
# Opt-in + header-name override (no code changes for other providers)
# ---------------------------------------------------------------------------


class TestOptInAndOverride:

    @pytest.mark.asyncio
    async def test_non_blackwell_off_by_default(self):
        """macstudio-shaped provider without the knob: no stamp."""
        config = _openai_config(
            make_provider(key="macstudio"), "macstudio/glm-5.3-flash")
        client = await _drive_stream(config, room_id=ROOM)

        kw = client.chat.completions.create.call_args.kwargs
        assert "extra_headers" not in kw

    @pytest.mark.asyncio
    async def test_non_blackwell_opt_in_stamps(self):
        """routing_key=true on any provider opts in — config, not code."""
        config = _openai_config(
            make_provider(key="macstudio", routing_key=True),
            "macstudio/glm-5.3-flash",
        )
        client = await _drive_stream(config, room_id=ROOM)

        kw = client.chat.completions.create.call_args.kwargs
        assert kw["extra_headers"][DEFAULT_ROUTING_KEY_HEADER] == \
            session_affinity_key(ROOM)

    @pytest.mark.asyncio
    async def test_header_name_override(self):
        """routing_key_header renames the stamp; value derivation is the same."""
        config = make_config(providers={
            "blackwell": make_provider(
                key="blackwell",
                routing_key=True,
                routing_key_header="X-Custom-Affinity"),
        })
        client = await _drive_stream(config, room_id=ROOM)

        kw = client.chat.completions.create.call_args.kwargs
        assert kw["extra_headers"]["X-Custom-Affinity"] == \
            session_affinity_key(ROOM)
        assert DEFAULT_ROUTING_KEY_HEADER not in kw["extra_headers"]


# ---------------------------------------------------------------------------
# (f) Fireworks behavior unchanged
# ---------------------------------------------------------------------------


class TestFireworksUnregressed:

    @pytest.mark.asyncio
    async def test_fireworks_keeps_own_path(self):
        """Fireworks: user + x-session-affinity as before, and NO
        X-SMG-Routing-Key (it is not blackwell; opt-in would be explicit)."""
        config = make_config(
            providers={"fireworks": make_provider(
                key="fireworks", type="openai",
                base_url="https://api.fireworks.ai/inference/v1")},
            default_model="fireworks/accounts/fireworks/models/glm-5p2",
        )
        client = await _drive_stream(config, room_id=ROOM)

        kw = client.chat.completions.create.call_args.kwargs
        expected = session_affinity_key(ROOM)
        assert kw["user"] == expected
        assert kw["extra_headers"]["x-session-affinity"] == expected
        assert DEFAULT_ROUTING_KEY_HEADER not in kw["extra_headers"]


# ---------------------------------------------------------------------------
# D1-clean: the header is invisible to the model and to info-level logs
# ---------------------------------------------------------------------------


class TestNotLogged:

    @pytest.mark.asyncio
    async def test_routing_key_never_logged(self, caplog):
        """No log record at any level carries the header name or value."""
        config = make_config()
        with caplog.at_level(logging.DEBUG, logger="openalph.provider"):
            await _drive_stream(config, room_id=ROOM)

        value = session_affinity_key(ROOM)
        for record in caplog.records:
            assert DEFAULT_ROUTING_KEY_HEADER not in record.getMessage()
            assert value not in record.getMessage()


# ---------------------------------------------------------------------------
# Resolver unit tests (the blackwell-default-on decision lives here)
# ---------------------------------------------------------------------------


class TestResolver:

    def test_blackwell_default_on(self):
        from openalph.provider import _resolved_routing_key_header
        assert _resolved_routing_key_header(make_provider(key="blackwell")) == \
            "X-SMG-Routing-Key"

    def test_other_provider_default_off(self):
        from openalph.provider import _resolved_routing_key_header
        assert _resolved_routing_key_header(
            make_provider(key="macstudio")) is None
        assert _resolved_routing_key_header(
            make_provider(key="fireworks")) is None

    def test_kill_flag_beats_default(self):
        from openalph.provider import _resolved_routing_key_header
        assert _resolved_routing_key_header(
            make_provider(key="blackwell", routing_key=False)) is None

    def test_opt_in_and_override(self):
        from openalph.provider import _resolved_routing_key_header
        assert _resolved_routing_key_header(
            make_provider(key="macstudio", routing_key=True)) == \
            "X-SMG-Routing-Key"
        assert _resolved_routing_key_header(make_provider(
            key="macstudio", routing_key=True,
            routing_key_header="X-Other")) == "X-Other"


# ---------------------------------------------------------------------------
# Config parsing (fail-loud, same discipline as the neighboring knobs)
# ---------------------------------------------------------------------------

TOML_BASE = """
[agent]
name = "test"
default_model = "blackwell/qwen38-27b-fp8"

[providers.blackwell]
type = "openai"
api_key = "sk-test"
base_url = "http://10.0.20.111:8000/v1"
{knobs}
[workspace]
path = "/tmp/test"
"""


class TestConfigParsing:

    def test_absent_defaults_none(self, tmp_path):
        (tmp_path / "agent.toml").write_text(TOML_BASE.format(knobs=""))
        config = load_config(tmp_path / "agent.toml")
        p = config.providers["blackwell"]
        assert p.routing_key is None
        assert p.routing_key_header is None

    def test_parsed(self, tmp_path):
        (tmp_path / "agent.toml").write_text(TOML_BASE.format(
            knobs='routing_key = true\nrouting_key_header = "X-Custom"\n'))
        config = load_config(tmp_path / "agent.toml")
        p = config.providers["blackwell"]
        assert p.routing_key is True
        assert p.routing_key_header == "X-Custom"

    def test_kill_flag_parsed(self, tmp_path):
        (tmp_path / "agent.toml").write_text(TOML_BASE.format(
            knobs="routing_key = false\n"))
        config = load_config(tmp_path / "agent.toml")
        assert config.providers["blackwell"].routing_key is False

    def test_invalid_routing_key_skips_provider(self, tmp_path):
        (tmp_path / "agent.toml").write_text(TOML_BASE.format(
            knobs='routing_key = "yes"\n'))
        config = load_config(tmp_path / "agent.toml")
        assert config.providers == {}
        assert "routing_key must be a boolean" in \
            config.skipped_providers["blackwell"]

    def test_invalid_routing_key_header_skips_provider(self, tmp_path):
        (tmp_path / "agent.toml").write_text(TOML_BASE.format(
            knobs='routing_key_header = 42\n'))
        config = load_config(tmp_path / "agent.toml")
        assert config.providers == {}
        assert "routing_key_header must be a non-empty string" in \
            config.skipped_providers["blackwell"]

    def test_empty_routing_key_header_skips_provider(self, tmp_path):
        """An empty header name is a config error (fail-loud), not an
        implicit kill flag — ``routing_key = false`` is the kill flag."""
        (tmp_path / "agent.toml").write_text(TOML_BASE.format(
            knobs='routing_key_header = ""\n'))
        config = load_config(tmp_path / "agent.toml")
        assert config.providers == {}
        assert "routing_key_header must be a non-empty string" in \
            config.skipped_providers["blackwell"]
