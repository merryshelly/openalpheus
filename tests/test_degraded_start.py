"""Acceptance tests: graceful degraded-start + anti-flap (workspace-kdsn.292).

SB ruling (2026-08-26, recorded on kdsn.164): one stray config in a provider
block must never crash-loop an agent or throttle 1Password fleet-wide.

Spec: memory/projects/openalph/specs/kdsn.292-degraded-start-spec.md

Semantics under test:
- ANY per-provider problem (bad type, key-resolution failure, missing
  base_url, bad timeout) skips that provider WITH REASON into
  AgentConfig.skipped_providers — never fatal at config load.
- Zero providers / default provider skipped / default references undeclared
  provider → degraded start, NOT ConfigError.
- Structural errors (bad TOML, missing [agent]/default_model, no
  [providers.*] section at all) remain ConfigError.
- LLM invocations against an unavailable provider raise
  ProviderUnavailableError (subclass of ProviderError) — pre-network.
- Matrix layer: warn-once notice latch, startup broadcast, /model both ways.
- ntfy [notifications]: POST only when the DEFAULT provider is degraded.
- Shipped systemd template carries anti-flap directives.

NOTE to implementor: test ASSERTIONS are law (spec contract). Fixture
scaffolding may be adapted to match real seams, but behavior expectations
may not be weakened. Keep the suite fast (monkeypatch subprocess; no real
sleep/timeout waits; no network).
"""

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openalph.config import (
    AgentConfig,
    ConfigError,
    ProviderConfig,
    load_config,
    resolve_model,
)
from openalph.provider import ProviderError


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _write_config(tmp_path: Path, toml_content: str) -> Path:
    config_path = tmp_path / "agent.toml"
    config_path.write_text(toml_content)
    ws = tmp_path / "workspace"
    ws.mkdir(exist_ok=True)
    return config_path


BASE_HEADER = """
[agent]
name = "test-agent"
default_model = "{default}"

[workspace]
path = "{workspace}"

"""

OK_ANTHROPIC = """
[providers.anthropic]
type = "anthropic"
api_key = "sk-test-inline"
"""

OK_FIREWORKS = """
[providers.fireworks]
type = "openai"
api_key = "fw-test-inline"
base_url = "https://api.fireworks.ai/inference/v1"
"""

BAD_CMD_ANTHROPIC = """
[providers.anthropic]
type = "anthropic"
api_key_cmd = "false"
"""

BAD_CMD_FIREWORKS = """
[providers.fireworks]
type = "openai"
api_key_cmd = "false"
base_url = "https://api.fireworks.ai/inference/v1"
"""


def _toml(tmp_path: Path, default: str, *blocks: str) -> Path:
    head = BASE_HEADER.format(default=default, workspace=tmp_path / "workspace")
    (tmp_path / "workspace").mkdir(exist_ok=True)
    return _write_config(tmp_path, head + "".join(blocks))


def _make_agent_config(**kwargs) -> AgentConfig:
    """Minimal AgentConfig for runtime tests (mirrors test_matrix.py style)."""
    defaults = dict(
        name="test-agent",
        default_model="anthropic/claude-test",
        max_tokens=8192,
        providers={
            "anthropic": ProviderConfig(
                key="anthropic", type="anthropic", api_key="sk-test",
            )
        },
        workspace=Path("/tmp/test"),
        max_iterations=25,
        truncation_limit=50000,
        model_max_tokens=200000,
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def _make_bot(config=None, **overrides):
    """Minimal MatrixBot test double (mirrors test_matrix.py make_bot)."""
    from openalph.matrix import MatrixBot

    bot = MatrixBot.__new__(MatrixBot)
    bot.config = config or MagicMock()
    bot.agent = MagicMock()
    bot.client = MagicMock()
    bot.client.room_send = AsyncMock()
    bot.client.room_typing = AsyncMock()
    bot.send = AsyncMock()
    bot.send_notice = AsyncMock()
    bot._unavailable_noticed = set()
    for key, val in overrides.items():
        setattr(bot, key, val)
    return bot


# ==========================================================================
# §2 — config.py degraded-start semantics
# ==========================================================================

class TestConfigDegradedStart:
    """Per-provider problems skip-with-reason; never fatal at load."""

    def test_single_key_failure_skipped_with_reason(self, tmp_path):
        cfg = _toml(tmp_path, "anthropic/claude-test", OK_ANTHROPIC, BAD_CMD_FIREWORKS)
        config = load_config(cfg)
        assert "anthropic" in config.providers
        assert "fireworks" not in config.providers
        assert "fireworks" in config.skipped_providers
        assert "exit code" in config.skipped_providers["fireworks"]

    def test_default_provider_key_failure_is_degraded_not_fatal(self, tmp_path):
        """THE incident case: default_model's provider fails key resolution.
        Was ConfigError + exit 1 + crash-loop; must now load degraded."""
        cfg = _toml(tmp_path, "anthropic/claude-test", BAD_CMD_ANTHROPIC, OK_FIREWORKS)
        config = load_config(cfg)  # must NOT raise
        assert config.default_model == "anthropic/claude-test"
        assert "anthropic" not in config.providers
        assert "anthropic" in config.skipped_providers
        assert "fireworks" in config.providers  # healthy half still usable

    def test_all_providers_fail_is_degraded_start(self, tmp_path):
        cfg = _toml(tmp_path, "anthropic/claude-test", BAD_CMD_ANTHROPIC, BAD_CMD_FIREWORKS)
        config = load_config(cfg)  # must NOT raise
        assert config.providers == {}
        assert set(config.skipped_providers) == {"anthropic", "fireworks"}

    def test_default_references_undeclared_provider_is_degraded(self, tmp_path):
        """default_model pointing at a provider that was never even declared
        (the blackwell-with-no-api_key incident shape, further corrupted)."""
        cfg = _toml(tmp_path, "blackwell/qwen-test", OK_ANTHROPIC)
        config = load_config(cfg)  # must NOT raise
        assert "blackwell" in config.skipped_providers
        assert "not configured" in config.skipped_providers["blackwell"]

    def test_invalid_provider_type_skipped(self, tmp_path):
        bad_type = (
            '\n[providers.broken]\ntype = "antropic"\napi_key = "sk-x"\n'
        )
        cfg = _toml(tmp_path, "anthropic/claude-test", OK_ANTHROPIC, bad_type)
        config = load_config(cfg)  # must NOT raise
        assert "broken" in config.skipped_providers
        assert "type" in config.skipped_providers["broken"].lower()
        assert "anthropic" in config.providers

    def test_openai_missing_base_url_skipped(self, tmp_path):
        no_base = (
            '\n[providers.localapi]\ntype = "openai"\napi_key = "none"\n'
        )
        cfg = _toml(tmp_path, "anthropic/claude-test", OK_ANTHROPIC, no_base)
        config = load_config(cfg)  # must NOT raise
        assert "localapi" in config.skipped_providers
        assert "base_url" in config.skipped_providers["localapi"]
        assert "anthropic" in config.providers

    def test_key_cmd_timeout_skipped(self, tmp_path, monkeypatch):
        import subprocess

        def _boom(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="sleep 999", timeout=10)

        monkeypatch.setattr(subprocess, "run", _boom)
        cfg = _toml(tmp_path, "anthropic/claude-test", OK_FIREWORKS, BAD_CMD_ANTHROPIC)
        config = load_config(cfg)  # must NOT raise
        assert "anthropic" in config.skipped_providers
        assert "timed out" in config.skipped_providers["anthropic"].lower()

    def test_api_key_cmd_single_attempt_no_retry(self, tmp_path, monkeypatch):
        """Startup must never re-hammer op: exactly ONE subprocess attempt
        per provider per config load (work item 3)."""
        import subprocess

        calls = {"n": 0}

        def _count(*args, **kwargs):
            calls["n"] += 1
            raise subprocess.CalledProcessError(1, "op read")

        monkeypatch.setattr(subprocess, "run", _count)
        cfg = _toml(tmp_path, "anthropic/claude-test", BAD_CMD_ANTHROPIC, OK_FIREWORKS)
        load_config(cfg)
        assert calls["n"] == 1

    def test_healthy_config_has_empty_skipped_map(self, tmp_path):
        cfg = _toml(tmp_path, "anthropic/claude-test", OK_ANTHROPIC, OK_FIREWORKS)
        config = load_config(cfg)
        assert config.skipped_providers == {}
        assert set(config.providers) == {"anthropic", "fireworks"}

    def test_skipped_providers_defaults_empty(self):
        """Backward compat: AgentConfig constructed directly (every existing
        call site) gets an empty skipped map."""
        config = _make_agent_config()
        assert config.skipped_providers == {}


class TestStructuralFatalsUnchanged:
    """Structural config errors remain fatal — they fire zero op reads."""

    def test_bad_toml_still_fatal(self, tmp_path):
        cfg = _write_config(tmp_path, "not [valid toml")
        with pytest.raises(ConfigError):
            load_config(cfg)

    def test_missing_default_model_still_fatal(self, tmp_path):
        body = (
            '[agent]\nname = "t"\n\n'
            f'[workspace]\npath = "{tmp_path / "workspace"}"\n\n'
        ) + OK_ANTHROPIC
        cfg = _write_config(tmp_path, body)
        with pytest.raises(ConfigError):
            load_config(cfg)

    def test_no_providers_sections_still_fatal(self, tmp_path):
        """[providers.*] section ABSENT entirely = misconfig, not throttling."""
        body = (
            '[agent]\nname = "t"\ndefault_model = "anthropic/x"\n\n'
            f'[workspace]\npath = "{tmp_path / "workspace"}"\n'
        )
        cfg = _write_config(tmp_path, body)
        with pytest.raises(ConfigError):
            load_config(cfg)


# ==========================================================================
# §3 — runtime typed error
# ==========================================================================

class TestProviderUnavailableError:
    """resolve_model_checked converts missing-provider to a typed error."""

    def test_class_hierarchy(self):
        from openalph.provider import ProviderUnavailableError

        assert issubclass(ProviderUnavailableError, ProviderError)

    def test_healthy_passthrough_matches_resolve_model(self):
        from openalph.provider import resolve_model_checked

        config = _make_agent_config()
        cfg, api_model = resolve_model_checked(
            "anthropic/claude-test", config.providers
        )
        expected = resolve_model("anthropic/claude-test", config.providers)
        assert (cfg, api_model) == expected

    def test_skipped_provider_raises_with_reason(self):
        from openalph.provider import ProviderUnavailableError, resolve_model_checked

        config = _make_agent_config(
            providers={},
            skipped_providers={"anthropic": "api_key_cmd failed with exit code 1"},
        )
        with pytest.raises(ProviderUnavailableError) as exc_info:
            resolve_model_checked(
                "anthropic/claude-test", config.providers,
                skipped_providers=config.skipped_providers,
            )
        e = exc_info.value
        assert e.provider_key == "anthropic"
        assert "unavailable" in str(e)
        assert "exit code 1" in str(e)

    def test_never_configured_provider_raises(self):
        from openalph.provider import ProviderUnavailableError, resolve_model_checked

        config = _make_agent_config()
        with pytest.raises(ProviderUnavailableError) as exc_info:
            resolve_model_checked(
                "blackwell/qwen38-27b-fp8", config.providers,
                skipped_providers=getattr(config, "skipped_providers", {}),
            )
        assert exc_info.value.provider_key == "blackwell"

    def test_unknown_alias_stays_valueerror(self):
        """Misspelled alias = user typo at runtime; keep the existing
        ValueError listing available aliases (do NOT convert)."""
        from openalph.provider import ProviderUnavailableError, resolve_model_checked

        config = _make_agent_config(model_aliases={"kimi": "fireworks/kimi-k3"})
        with pytest.raises(ValueError) as exc_info:
            resolve_model_checked(
                "kini", config.providers, aliases=config.model_aliases
            )
        assert not isinstance(exc_info.value, ProviderUnavailableError)
        assert "kimi" in str(exc_info.value)

    def test_alias_expanding_to_skipped_provider_raises_typed(self):
        """Review F1 (kimi + glm, both HIGH): a BARE ALIAS that expands to a
        skipped provider must produce the typed error WITH the skip reason —
        not a raw ValueError that bypasses the warn-once latch and in-room
        notice path. This is the default_model-alias incident shape."""
        from openalph.provider import ProviderUnavailableError, resolve_model_checked

        config = _make_agent_config(
            providers={},
            model_aliases={"myalias": "broken/some-model"},
            skipped_providers={"broken": "api_key_cmd failed with exit code 1"},
        )
        with pytest.raises(ProviderUnavailableError) as exc_info:
            resolve_model_checked(
                "myalias", config.providers,
                aliases=config.model_aliases,
                skipped_providers=config.skipped_providers,
            )
        e = exc_info.value
        assert e.provider_key == "broken"
        assert "exit code 1" in str(e)

    def test_alias_expanding_to_never_configured_raises_typed(self):
        from openalph.provider import ProviderUnavailableError, resolve_model_checked

        config = _make_agent_config(
            providers={},
            model_aliases={"myalias": "blackwell/qwen"},
        )
        with pytest.raises(ProviderUnavailableError) as exc_info:
            resolve_model_checked(
                "myalias", config.providers,
                aliases=config.model_aliases,
                skipped_providers=config.skipped_providers,
            )
        assert exc_info.value.provider_key == "blackwell"


class TestDegradedAtProviderSeam:
    """Integration: api_key_cmd failure on the DEFAULT provider yields an
    alive-but-degraded agent — config loads, and any LLM invocation fails
    with the typed error BEFORE touching the network (work items 1+2+5)."""

    def _degraded_config(self, tmp_path):
        cfg = _toml(tmp_path, "anthropic/claude-test", BAD_CMD_ANTHROPIC)
        return load_config(cfg)  # alive: must not raise

    def test_stream_raises_typed_pre_network(self, tmp_path):
        import openalph.provider as provider_mod

        config = self._degraded_config(tmp_path)
        gen = provider_mod.stream(config, "sys", [{"role": "user", "content": "hi"}])
        loop = asyncio.new_event_loop()
        try:
            with pytest.raises(provider_mod.ProviderUnavailableError):
                loop.run_until_complete(gen.__anext__())
        finally:
            loop.close()

    def test_complete_raises_typed_pre_network(self, tmp_path):
        import openalph.provider as provider_mod

        config = self._degraded_config(tmp_path)
        loop = asyncio.new_event_loop()
        try:
            with pytest.raises(provider_mod.ProviderUnavailableError):
                loop.run_until_complete(
                    provider_mod.complete(config, "sys", [{"role": "user", "content": "hi"}])
                )
        finally:
            loop.close()

    def test_alias_default_with_dead_provider_full_loop(self, tmp_path):
        """End-to-end incident shape: default_model is a bare ALIAS whose
        target provider fails key resolution. Config loads degraded (not
        ConfigError) AND the runtime turn raises the typed error — both
        sides of the path, not just one (review F1)."""
        import openalph.provider as provider_mod

        body = (
            '[agent]\nname = "t"\ndefault_model = "myalias"\n\n'
            f'[workspace]\npath = "{tmp_path / "workspace"}"\n\n'
            '[model_aliases]\nmyalias = "broken/some-model"\n\n'
            '[providers.broken]\ntype = "anthropic"\napi_key_cmd = "false"\n'
        )
        cfg = _write_config(tmp_path, body)
        config = load_config(cfg)  # alive — must not raise
        assert config.providers == {}
        assert config.skipped_providers["broken"]
        gen = provider_mod.stream(config, "sys", [{"role": "user", "content": "hi"}])
        loop = asyncio.new_event_loop()
        try:
            with pytest.raises(provider_mod.ProviderUnavailableError):
                loop.run_until_complete(gen.__anext__())
        finally:
            loop.close()


# ==========================================================================
# §3a — warn-once latch
# ==========================================================================

class TestWarnOnceLatch:
    """A degraded default + 5m heartbeat must not notice-spam the room."""

    def _err(self, provider="anthropic", reason="api_key_cmd failed with exit code 1"):
        from openalph.provider import ProviderUnavailableError

        return ProviderUnavailableError(
            f"provider '{provider}' unavailable: {reason}",
            provider_key=provider, reason=reason,
        )

    @pytest.mark.asyncio
    async def test_first_notice_sent_second_suppressed(self):
        bot = _make_bot()
        await bot._emit_provider_notice("!r:s", self._err())
        assert bot.send.await_count == 1
        await bot._emit_provider_notice("!r:s", self._err())  # same room+provider
        assert bot.send.await_count == 1  # still one — latched

    @pytest.mark.asyncio
    async def test_latch_is_per_room_and_per_provider(self):
        bot = _make_bot()
        await bot._emit_provider_notice("!r:s", self._err("anthropic"))
        await bot._emit_provider_notice("!r2:s", self._err("anthropic"))  # new room
        await bot._emit_provider_notice("!r:s", self._err("fireworks"))   # new provider
        assert bot.send.await_count == 3

    @pytest.mark.asyncio
    async def test_transient_provider_error_never_latched(self):
        bot = _make_bot()
        await bot._emit_provider_notice("!r:s", ProviderError("Provider timed out"))
        await bot._emit_provider_notice("!r:s", ProviderError("Provider timed out"))
        assert bot.send.await_count == 2

    @pytest.mark.asyncio
    async def test_notice_names_provider_and_reason(self):
        bot = _make_bot()
        await bot._emit_provider_notice("!r:s", self._err())
        text = bot.send.await_args[0][1]
        assert "anthropic" in text
        assert "exit code 1" in text


# ==========================================================================
# §3c — startup broadcast
# ==========================================================================

class TestStartupBroadcast:
    """One m.notice per joined room when the process came up degraded."""

    @pytest.mark.asyncio
    async def test_broadcast_one_notice_per_room(self):
        cfg = _make_agent_config(
            providers={},
            skipped_providers={
                "anthropic": "api_key_cmd failed with exit code 1",
                "fireworks": "api_key_cmd timed out after 10 seconds",
            },
        )
        bot = _make_bot()
        bot.agent.config = cfg
        bot.client.rooms = {"!a:s": MagicMock(), "!b:s": MagicMock()}
        await bot._broadcast_degraded_start()
        assert bot.send_notice.await_count == 2
        rooms = {c[0][0] for c in bot.send_notice.await_args_list}
        assert rooms == {"!a:s", "!b:s"}
        body = bot.send_notice.await_args_list[0][0][1]
        assert "anthropic" in body
        assert "exit code 1" in body
        assert "degraded" in body.lower() or "DEGRADED" in body

    @pytest.mark.asyncio
    async def test_no_broadcast_when_healthy(self):
        cfg = _make_agent_config()  # skipped_providers == {}
        bot = _make_bot()
        bot.agent.config = cfg
        bot.client.rooms = {"!a:s": MagicMock()}
        await bot._broadcast_degraded_start()
        assert bot.send_notice.await_count == 0

    @pytest.mark.asyncio
    async def test_broadcast_reasons_are_redacted(self):
        secret = "sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"
        cfg = _make_agent_config(
            providers={},
            skipped_providers={"anthropic": f"command output looked like key {secret} tail"},
        )
        bot = _make_bot()
        bot.agent.config = cfg
        bot.client.rooms = {"!a:s": MagicMock()}
        await bot._broadcast_degraded_start()
        body = bot.send_notice.await_args_list[0][0][1]
        assert secret not in body

    @pytest.mark.asyncio
    async def test_broadcast_send_failure_never_raises(self):
        cfg = _make_agent_config(
            providers={}, skipped_providers={"anthropic": "boom"}
        )
        bot = _make_bot()
        bot.agent.config = cfg
        bot.client.rooms = {"!a:s": MagicMock()}
        bot.send_notice = AsyncMock(side_effect=RuntimeError("matrix down"))
        await bot._broadcast_degraded_start()  # must not raise


# ==========================================================================
# §3b — /model bidirectional pin
# ==========================================================================

class TestModelSwitchFromDegraded:
    """switch_model: healthy restores WITHOUT restart; dead is loud + keeps state."""

    def _agent(self, tmp_path):
        from openalph.agent import Agent

        cfg = _make_agent_config(
            workspace=tmp_path,  # real dir: Agent assembles prompt from it
            providers={
                "fireworks": ProviderConfig(
                    key="fireworks", type="openai", api_key="fw-test",
                    base_url="https://api.fireworks.ai/inference/v1",
                )
            },
            default_model="anthropic/claude-test",
            skipped_providers={"anthropic": "api_key_cmd failed with exit code 1"},
        )
        return Agent(cfg)

    def test_switch_to_healthy_restores(self, tmp_path):
        agent = self._agent(tmp_path)
        assert agent.switch_model("fireworks/kimi-k3", "!r:s") is None
        assert agent.get_model("!r:s") == "fireworks/kimi-k3"

    def test_switch_to_dead_is_loud_and_keeps_state(self, tmp_path):
        agent = self._agent(tmp_path)
        err = agent.switch_model("anthropic/claude-test", "!r:s")
        assert err is not None
        assert "exit code 1" in err  # names the skip reason
        assert agent.get_model("!r:s") == "anthropic/claude-test"  # unchanged


# ==========================================================================
# §4 — ntfy [notifications]
# ==========================================================================

class TestNotifyModule:
    def _cfg(self, **kw):
        return _make_agent_config(**kw)

    def test_degraded_summary_names_providers_reasons_default(self):
        from openalph.notify import degraded_summary

        cfg = self._cfg(
            providers={},
            skipped_providers={"anthropic": "api_key_cmd failed with exit code 1"},
        )
        text = degraded_summary(cfg)
        assert "anthropic" in text
        assert "exit code 1" in text
        assert "claude-test" in text  # default model visible for context

    def test_sends_when_default_degraded(self):
        from openalph.notify import maybe_notify_default_degraded

        notif = MagicMock()
        notif.ntfy_url = "http://127.0.0.1:8090/ops"
        notif.ntfy_token = "tk-test"
        cfg = self._cfg(
            providers={},
            skipped_providers={"anthropic": "boom"},
            notifications=notif,
        )
        with patch("openalph.notify.urlopen") as mock_open:
            assert maybe_notify_default_degraded(cfg) is True
            req = mock_open.call_args[0][0]
            assert req.full_url.startswith("http://127.0.0.1:8090/ops")
            assert "Bearer tk-test" in (req.headers.get("Authorization") or "")

    def test_silent_when_default_healthy(self):
        from openalph.notify import maybe_notify_default_degraded

        notif = MagicMock()
        cfg = self._cfg(
            skipped_providers={"fireworks": "boom"},  # NON-default skip
            notifications=notif,
        )
        with patch("openalph.notify.urlopen") as mock_open:
            assert maybe_notify_default_degraded(cfg) is False
            mock_open.assert_not_called()

    def test_silent_when_notifications_not_configured(self):
        from openalph.notify import maybe_notify_default_degraded

        cfg = self._cfg(providers={}, skipped_providers={"anthropic": "boom"})
        with patch("openalph.notify.urlopen") as mock_open:
            assert maybe_notify_default_degraded(cfg) is False
            mock_open.assert_not_called()

    def test_network_failure_is_fail_soft(self):
        from openalph.notify import maybe_notify_default_degraded

        notif = MagicMock()
        notif.ntfy_url = "http://127.0.0.1:8090/ops"
        notif.ntfy_token = None
        cfg = self._cfg(
            providers={}, skipped_providers={"anthropic": "boom"},
            notifications=notif,
        )
        with patch("openalph.notify.urlopen", side_effect=OSError("down")):
            assert maybe_notify_default_degraded(cfg) is False  # never raises


class TestNotificationsConfigParse:
    def test_notifications_section_parsed(self, tmp_path):
        notif = (
            '\n[notifications]\n'
            'ntfy_url = "http://127.0.0.1:8090/ops"\n'
            'ntfy_token_cmd = "cat /dev/null"\n'
        )
        cfg = _write_config(
            tmp_path,
            ('[agent]\nname = "t"\ndefault_model = "anthropic/x"\n\n'
             f'[workspace]\npath = "{tmp_path / "workspace"}"\n')
            + OK_ANTHROPIC + notif,
        )
        config = load_config(cfg)
        assert config.notifications is not None
        assert config.notifications.ntfy_url == "http://127.0.0.1:8090/ops"
        # token_cmd produced empty output → fail-soft: token None, no raise
        assert config.notifications.ntfy_token is None

    def test_notifications_absent_is_none(self, tmp_path):
        cfg = _toml(tmp_path, "anthropic/claude-test", OK_ANTHROPIC)
        config = load_config(cfg)
        assert config.notifications is None

    def test_notifications_token_cmd_failure_fails_soft(self, tmp_path):
        notif = (
            '\n[notifications]\n'
            'ntfy_url = "http://127.0.0.1:8090/ops"\n'
            'ntfy_token_cmd = "false"\n'
        )
        cfg = _write_config(
            tmp_path,
            ('[agent]\nname = "t"\ndefault_model = "anthropic/x"\n\n'
             f'[workspace]\npath = "{tmp_path / "workspace"}"\n')
            + OK_ANTHROPIC + notif,
        )
        config = load_config(cfg)  # must NOT raise — the alerting path must
        assert config.notifications is not None          # never recreate the
        assert config.notifications.ntfy_token is None   # bug it reports

    def test_notifications_bad_url_is_fatal(self, tmp_path):
        notif = '\n[notifications]\nntfy_url = 42\n'
        cfg = _write_config(
            tmp_path,
            ('[agent]\nname = "t"\ndefault_model = "anthropic/x"\n\n'
             f'[workspace]\npath = "{tmp_path / "workspace"}"\n')
            + OK_ANTHROPIC + notif,
        )
        with pytest.raises(ConfigError):
            load_config(cfg)


# ==========================================================================
# §5 — anti-flap template parity
# ==========================================================================

class TestAntiFlapUnitTemplate:
    """Shipped openalph@.service carries the anti-flap directives so fresh
    installs inherit bounded-restart behavior (fleet drop-in deployed
    separately as an ops artifact)."""

    UNIT = Path(__file__).resolve().parent.parent / "src" / "openalph" / "data" / "openalph@.service"

    def test_template_has_start_limits_in_unit_section(self):
        import configparser

        cp = configparser.ConfigParser(strict=True)
        cp.read(self.UNIT)
        assert cp.get("Unit", "StartLimitIntervalSec") == "600"
        assert cp.get("Unit", "StartLimitBurst") == "5"

    def test_template_restart_sec_slowed(self):
        import configparser

        cp = configparser.ConfigParser(strict=True)
        cp.read(self.UNIT)
        assert cp.get("Service", "RestartSec") == "30"
