"""Tests for cache bust notice feature.

Covers:
- Config parsing: cache_bust_notices defaults, TOML parsing, type validation
- Agent callback: on_cache_status fires on cache miss, not on cache hit/below threshold
- Transport callback logic: provider opt-in/opt-out gating
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from openalph.config import AgentConfig, ProviderConfig, load_config, ConfigError
from openalph.provider import Usage, Response, StreamEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_provider(key="anthropic", type="anthropic", api_key="sk-test",
                  base_url=None, quirks=None, cache_bust_notices=False):
    return ProviderConfig(
        key=key, type=type, api_key=api_key, base_url=base_url,
        quirks=quirks or [], cache_bust_notices=cache_bust_notices,
    )


def make_config(workspace, cache_bust_notices=False, **kwargs):
    defaults = dict(
        name="test",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": make_provider(
            key="anthropic", cache_bust_notices=cache_bust_notices,
        )},
    )
    defaults.update(kwargs)
    defaults["workspace"] = workspace
    return AgentConfig(**defaults)


def make_stream_events(content="Hello", input_tokens=100, output_tokens=50,
                       cache_read_tokens=None, cache_creation_tokens=None):
    """Create a mock async generator yielding stream events."""
    async def _stream(*args, **kwargs):
        yield StreamEvent(type="text", content=content)
        yield StreamEvent(
            type="done",
            response=Response(
                content=content,
                model="claude-sonnet-4-20250514",
                usage=Usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=cache_read_tokens,
                    cache_creation_tokens=cache_creation_tokens,
                ),
                stop_reason="end_turn",
            ),
            stop_reason="end_turn",
            model="claude-sonnet-4-20250514",
        )
    return _stream


# ===========================================================================
# A. Config parsing tests
# ===========================================================================


class TestCacheBustNoticesConfig:

    def test_defaults_to_false(self):
        """cache_bust_notices defaults to False on ProviderConfig."""
        p = ProviderConfig(key="test", type="anthropic", api_key="sk-test")
        assert p.cache_bust_notices is False

    def test_explicit_true(self):
        """cache_bust_notices=True is stored correctly."""
        p = ProviderConfig(key="test", type="anthropic", api_key="sk-test",
                           cache_bust_notices=True)
        assert p.cache_bust_notices is True

    def test_toml_parses_true(self, tmp_path):
        """cache_bust_notices = true in TOML is parsed correctly."""
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "anthropic/test-model"

[providers.anthropic]
type = "anthropic"
api_key = "sk-test"
cache_bust_notices = true

[workspace]
path = "/tmp/test-workspace"
""")
        config = load_config(tmp_path / "agent.toml")
        assert config.providers["anthropic"].cache_bust_notices is True

    def test_toml_defaults_to_false(self, tmp_path):
        """cache_bust_notices omitted from TOML defaults to False."""
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "anthropic/test-model"

[providers.anthropic]
type = "anthropic"
api_key = "sk-test"

[workspace]
path = "/tmp/test-workspace"
""")
        config = load_config(tmp_path / "agent.toml")
        assert config.providers["anthropic"].cache_bust_notices is False

    def test_toml_rejects_non_boolean(self, tmp_path):
        """cache_bust_notices = "yes" in TOML raises ConfigError."""
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "anthropic/test-model"

[providers.anthropic]
type = "anthropic"
api_key = "sk-test"
cache_bust_notices = "yes"

[workspace]
path = "/tmp/test-workspace"
""")
        with pytest.raises(ConfigError, match="cache_bust_notices must be a boolean"):
            load_config(tmp_path / "agent.toml")


# ===========================================================================
# B. Agent on_cache_status callback tests
# ===========================================================================


class TestAgentCacheStatusCallback:

    @pytest.mark.asyncio
    async def test_callback_fires_on_cache_miss(self, tmp_path):
        """on_cache_status callback receives Usage with cache miss data."""
        (tmp_path / "SOUL.md").write_text("Test soul.")
        config = make_config(tmp_path)

        from openalph.agent import Agent
        agent = Agent(config)

        callback = AsyncMock()

        with patch("openalph.agent.stream", make_stream_events(
            cache_read_tokens=0, cache_creation_tokens=50000,
        )):
            await agent.handle_input("hello", "room1", on_cache_status=callback)

        callback.assert_awaited_once()
        usage_arg = callback.call_args[0][0]
        assert usage_arg.cache_read_tokens == 0
        assert usage_arg.cache_creation_tokens == 50000

    @pytest.mark.asyncio
    async def test_callback_fires_on_cache_hit(self, tmp_path):
        """on_cache_status callback fires even on cache hits (filtering is transport's job)."""
        (tmp_path / "SOUL.md").write_text("Test soul.")
        config = make_config(tmp_path)

        from openalph.agent import Agent
        agent = Agent(config)

        callback = AsyncMock()

        with patch("openalph.agent.stream", make_stream_events(
            cache_read_tokens=50000, cache_creation_tokens=500,
        )):
            await agent.handle_input("hello", "room1", on_cache_status=callback)

        # Agent always fires the callback — the transport decides whether to act
        callback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_crash_agent(self, tmp_path):
        """on_cache_status exception is swallowed — agent returns normally."""
        (tmp_path / "SOUL.md").write_text("Test soul.")
        config = make_config(tmp_path)

        from openalph.agent import Agent
        agent = Agent(config)

        callback = AsyncMock(side_effect=RuntimeError("boom"))

        with patch("openalph.agent.stream", make_stream_events(
            cache_read_tokens=0, cache_creation_tokens=50000,
        )):
            result = await agent.handle_input("hello", "room1", on_cache_status=callback)

        assert result == "Hello"  # Agent returns normally despite callback error

    @pytest.mark.asyncio
    async def test_no_callback_no_error(self, tmp_path):
        """Omitting on_cache_status doesn't cause errors (backward compat)."""
        (tmp_path / "SOUL.md").write_text("Test soul.")
        config = make_config(tmp_path)

        from openalph.agent import Agent
        agent = Agent(config)

        with patch("openalph.agent.stream", make_stream_events(
            cache_read_tokens=0, cache_creation_tokens=50000,
        )):
            result = await agent.handle_input("hello", "room1")

        assert result == "Hello"


# ===========================================================================
# C. Transport-layer callback logic tests
# ===========================================================================


class TestCacheStatusTransportLogic:
    """Test the cache status callback logic that would be defined in matrix.py.

    We test the callback function in isolation to verify filtering logic.
    The callback fires when cache_creation_tokens >= 10k regardless of
    cache_read_tokens (catches both full and partial cache misses).
    """

    @staticmethod
    def _make_callback(send_notice, session_log, providers):
        """Build a _cache_status callback matching the matrix.py implementation."""
        async def _cache_status(usage, model_str):
            cr = usage.cache_read_tokens or 0
            cc = usage.cache_creation_tokens or 0
            if cc < 10000:
                return
            try:
                from openalph.config import resolve_model
                provider_cfg, _ = resolve_model(model_str, providers)
                if not provider_cfg.cache_bust_notices:
                    return
            except Exception:
                return
            total = cr + cc
            miss_pct = (cc / total * 100) if total > 0 else 100
            notice = f"⚠️ Cache warning — {cc:,} tokens written, {cr:,} read ({miss_pct:.0f}% uncached)"
            await send_notice(notice)
            if session_log:
                session_log.append(
                    role="system",
                    sender="@test:matrix.local",
                    room="!room:test",
                    event_id=None,
                    event="cache_warning",
                    detail=f"cache_read={cr} cache_creation={cc} miss_pct={miss_pct:.0f}",
                )
        return _cache_status

    @pytest.mark.asyncio
    async def test_notice_on_full_cache_miss(self):
        """Full cache miss (0 read, >=10k creation) sends notice with 100% missed."""
        send_notice = AsyncMock()
        session_log = MagicMock()
        providers = {"anthropic": make_provider(key="anthropic", cache_bust_notices=True)}
        cb = self._make_callback(send_notice, session_log, providers)

        usage = Usage(input_tokens=100, output_tokens=50,
                      cache_read_tokens=0, cache_creation_tokens=50000)
        await cb(usage, "anthropic/claude-sonnet-4-20250514")

        send_notice.assert_awaited_once()
        msg = send_notice.call_args[0][0]
        assert "50,000" in msg
        assert "100% uncached" in msg
        session_log.append.assert_called_once()
        assert session_log.append.call_args[1]["event"] == "cache_warning"

    @pytest.mark.asyncio
    async def test_notice_on_partial_cache_miss(self):
        """Partial cache miss (read > 0, creation >= 10k) sends notice with correct %."""
        send_notice = AsyncMock()
        providers = {"anthropic": make_provider(key="anthropic", cache_bust_notices=True)}
        cb = self._make_callback(send_notice, None, providers)

        usage = Usage(input_tokens=100, output_tokens=50,
                      cache_read_tokens=40000, cache_creation_tokens=20000)
        await cb(usage, "anthropic/claude-sonnet-4-20250514")

        send_notice.assert_awaited_once()
        msg = send_notice.call_args[0][0]
        assert "20,000 tokens written" in msg
        assert "40,000 read" in msg
        assert "33% uncached" in msg

    @pytest.mark.asyncio
    async def test_no_notice_on_cache_hit(self):
        """Cache hit (small creation < 10k) does not send notice regardless of read."""
        send_notice = AsyncMock()
        providers = {"anthropic": make_provider(key="anthropic", cache_bust_notices=True)}
        cb = self._make_callback(send_notice, None, providers)

        usage = Usage(input_tokens=100, output_tokens=50,
                      cache_read_tokens=50000, cache_creation_tokens=500)
        await cb(usage, "anthropic/claude-sonnet-4-20250514")

        send_notice.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_notice_below_threshold(self):
        """Cache creation below 10k does not send notice (even with 0 read)."""
        send_notice = AsyncMock()
        providers = {"anthropic": make_provider(key="anthropic", cache_bust_notices=True)}
        cb = self._make_callback(send_notice, None, providers)

        usage = Usage(input_tokens=100, output_tokens=50,
                      cache_read_tokens=0, cache_creation_tokens=5000)
        await cb(usage, "anthropic/claude-sonnet-4-20250514")

        send_notice.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_notice_when_provider_opt_out(self):
        """cache_bust_notices=False suppresses notice even on full cache miss."""
        send_notice = AsyncMock()
        providers = {"anthropic": make_provider(key="anthropic", cache_bust_notices=False)}
        cb = self._make_callback(send_notice, None, providers)

        usage = Usage(input_tokens=100, output_tokens=50,
                      cache_read_tokens=0, cache_creation_tokens=50000)
        await cb(usage, "anthropic/claude-sonnet-4-20250514")

        send_notice.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_notice_at_exact_threshold(self):
        """Exactly 10000 cache_creation_tokens triggers notice."""
        send_notice = AsyncMock()
        providers = {"anthropic": make_provider(key="anthropic", cache_bust_notices=True)}
        cb = self._make_callback(send_notice, None, providers)

        usage = Usage(input_tokens=100, output_tokens=50,
                      cache_read_tokens=0, cache_creation_tokens=10000)
        await cb(usage, "anthropic/claude-sonnet-4-20250514")

        send_notice.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_none_cache_tokens_treated_as_zero(self):
        """None cache tokens (non-Anthropic provider) treated as 0 — no notice."""
        send_notice = AsyncMock()
        providers = {"anthropic": make_provider(key="anthropic", cache_bust_notices=True)}
        cb = self._make_callback(send_notice, None, providers)

        usage = Usage(input_tokens=100, output_tokens=50,
                      cache_read_tokens=None, cache_creation_tokens=None)
        await cb(usage, "anthropic/claude-sonnet-4-20250514")

        send_notice.assert_not_awaited()
