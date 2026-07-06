"""RED test suite — Subagent cache keepalive.

Refreshes the parent's Anthropic prompt-cache TTL while a blocking `subagent`
tool call is in flight, so a subagent that outlives the cache TTL does not
force a 2x cache-write bust on return.

This suite is intentionally RED. It specifies the acceptance criteria before
implementation. All feature imports/attrs are guarded so the suite still
COLLECTS; individual tests fail (not error) until the feature is built.

Spec: memory/projects/openalph/specs/subagent-cache-keepalive.md
Bead: workspace-kdsn.190

Interface locked by this suite (re-anchored against post-.186 source):

  provider.py
    ping_cache(config, *, system, messages, tools, cache_ttl, thinking_level, model=None)
        -> Usage | None    # None (no-op) for non-Anthropic providers
    KEEPALIVE_MAX_OUTPUT_TOKENS == 1

  agent.py  (module-level, patchable)
    KEEPALIVE_REFRESH_FRACTION == 0.8
    KEEPALIVE_MIN_INTERVAL_S   == 60
    KEEPALIVE_MIN_CACHE_READ   == 1000
    _keepalive_ttl_seconds(cache_ttl) -> int      # "1h"->3600, "5m"->300, else 3600
    _keepalive_interval(cache_ttl)    -> float     # max(ttl*FRACTION, MIN_INTERVAL)
    _keepalive_is_hit(read, write)    -> bool      # read > write and read >= MIN_CACHE_READ
    ping_cache                                     # imported into agent namespace
    Agent._cache_keepalive(*, system, messages, tools, cache_ttl, model, thinking,
                           room_id, on_miss, stop)  # background coroutine

  config.py
    ProviderConfig.subagent_cache_keepalive: bool = False

  matrix.py
    MatrixBot._build_agent_callbacks(...) -> dict includes key "on_keepalive_miss"
"""

import asyncio
import logging
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Guarded imports — feature may not exist yet; keep collection green.
# ---------------------------------------------------------------------------
try:
    from openalph.agent import Agent
    import openalph.agent as agent_mod
    _agent_imported = True
except Exception:  # pragma: no cover
    Agent = None  # type: ignore
    agent_mod = None  # type: ignore
    _agent_imported = False

try:
    from openalph.config import AgentConfig, ProviderConfig, load_config, ConfigError
    _config_imported = True
except Exception:  # pragma: no cover
    AgentConfig = ProviderConfig = load_config = ConfigError = None  # type: ignore
    _config_imported = False

try:
    from openalph.provider import Usage, Response, StreamEvent, ToolCall
    import openalph.provider as provider_mod
    _provider_imported = True
except Exception:  # pragma: no cover
    Usage = Response = StreamEvent = ToolCall = provider_mod = None  # type: ignore
    _provider_imported = False

try:
    from openalph.tools import ToolResult
    _tools_imported = True
except Exception:  # pragma: no cover
    ToolResult = None  # type: ignore
    _tools_imported = False

try:
    from openalph.matrix import MatrixBot
    _matrix_imported = True
except Exception:  # pragma: no cover
    MatrixBot = None  # type: ignore
    _matrix_imported = False


pytestmark = pytest.mark.skipif(
    not (_agent_imported and _config_imported and _provider_imported and _tools_imported),
    reason="core openalph modules failed to import",
)

ROOM = "!room-a:matrix.local"
MODEL_ANTHROPIC = "anthropic/claude-sonnet-4-20250514"
MODEL_OPUS = "anthropic/claude-opus-4-8"  # adaptive-thinking model (merry's real config)
MODEL_OPENAI = "openai/gpt-x"

# A cache-hit Usage: huge read, ~zero write (prefix was warm).
HIT = dict(input_tokens=8, output_tokens=1, cache_read_tokens=150_000, cache_creation_tokens=4)
# A cache-miss/write Usage: ~zero read, huge write (prefix drifted / expired).
MISS = dict(input_tokens=8, output_tokens=1, cache_read_tokens=0, cache_creation_tokens=150_000)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _provider(keepalive=False, ptype="anthropic", key="anthropic"):
    kw = dict(key=key, type=ptype, api_key="sk-test", base_url=None, quirks=[])
    if ptype == "openai":
        kw["base_url"] = "http://local"
    # Always pass the flag so the suite is RED until the field exists.
    kw["subagent_cache_keepalive"] = keepalive
    return ProviderConfig(**kw)


def _make_agent_config(workspace, *, keepalive=False, anthropic=True, with_openai=False, **kwargs):
    providers = {}
    if anthropic:
        providers["anthropic"] = _provider(keepalive=keepalive, ptype="anthropic", key="anthropic")
    if with_openai or not anthropic:
        providers["openai"] = _provider(keepalive=keepalive, ptype="openai", key="openai")
    defaults = dict(
        name="test-agent",
        default_model=MODEL_ANTHROPIC if anthropic else MODEL_OPENAI,
        max_tokens=8192,
        providers=providers,
        workspace=workspace,
        max_iterations=5,
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=None,
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def _usage(**kw):
    return Usage(**kw)


def _mock_client(cache_read=150_000, cache_creation=4, in_tok=8, out_tok=1):
    """A MagicMock anthropic client whose messages.create returns a usage-bearing msg."""
    msg = MagicMock()
    msg.usage.input_tokens = in_tok
    msg.usage.output_tokens = out_tok
    msg.usage.cache_read_input_tokens = cache_read
    msg.usage.cache_creation_input_tokens = cache_creation
    msg.content = []
    msg.model = "claude-sonnet-4-20250514"
    msg.stop_reason = "end_turn"
    msg.id = "msg_ping"
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=msg)
    return client


def _subagent_stream(sub_names=("subagent",), final_text="all done"):
    """Stream factory: iter 1 yields tool_use(s) (incl. a subagent); iter 2 yields text."""
    tcs = [ToolCall(id=f"sub{i}", name=n, input={"task": "deep research"})
           for i, n in enumerate(sub_names)]
    responses = [
        Response(content="", tool_calls=tcs, model="claude-sonnet-4-20250514",
                 usage=Usage(input_tokens=10, output_tokens=5), stop_reason="tool_use"),
        Response(content=final_text, tool_calls=[], model="claude-sonnet-4-20250514",
                 usage=Usage(input_tokens=10, output_tokens=5), stop_reason="end_turn"),
    ]
    it = iter(responses)

    async def _stream(*args, **kwargs):
        r = next(it)
        if r.content:
            yield StreamEvent(type="text", content=r.content)
        for i, t in enumerate(r.tool_calls):
            yield StreamEvent(type="tool_done", tool_index=i, tool_call=t)
        yield StreamEvent(type="done", response=r, stop_reason=r.stop_reason, model=r.model)

    return _stream


def _nonsub_stream(tool_names=("shell",), final_text="done"):
    """Stream factory: iter 1 yields non-subagent tool_use; iter 2 yields text."""
    return _subagent_stream(sub_names=tool_names, final_text=final_text)


def _blocking_exec(release: asyncio.Event, seen=None):
    """execute_tool replacement: blocks until `release` is set (holds the gather open)."""
    async def _exec(*args, **kwargs):
        if seen is not None:
            seen.append(kwargs.get("name"))
        await release.wait()
        return ToolResult(content="sub result", is_error=False)
    return _exec


async def _wait_for(predicate, timeout=2.0, poll=0.005):
    """Poll until predicate() is truthy or timeout. Returns final predicate value."""
    loops = int(timeout / poll)
    for _ in range(loops):
        if predicate():
            return True
        await asyncio.sleep(poll)
    return predicate()


# ===========================================================================
# A. Config surface — mirror cache_bust_notices exactly.
# ===========================================================================
@pytest.mark.skipif(not _config_imported, reason="config import failed")
class TestConfig:

    def test_default_false(self):
        p = ProviderConfig(key="anthropic", type="anthropic", api_key="sk-test")
        assert p.subagent_cache_keepalive is False

    def test_toml_parse_true(self, tmp_path):
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "anthropic/test-model"

[providers.anthropic]
type = "anthropic"
api_key = "sk-test"
subagent_cache_keepalive = true

[workspace]
path = "/tmp/test-workspace"
""")
        config = load_config(tmp_path / "agent.toml")
        assert config.providers["anthropic"].subagent_cache_keepalive is True

    def test_toml_defaults_false_when_absent(self, tmp_path):
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
        assert config.providers["anthropic"].subagent_cache_keepalive is False

    def test_type_validation_non_bool_raises(self, tmp_path):
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "anthropic/test-model"

[providers.anthropic]
type = "anthropic"
api_key = "sk-test"
subagent_cache_keepalive = "yes"

[workspace]
path = "/tmp/test-workspace"
""")
        with pytest.raises(ConfigError, match="subagent_cache_keepalive must be a boolean"):
            load_config(tmp_path / "agent.toml")


# ===========================================================================
# B. Pure helpers — interval math + hit detection.  (Test plan #9, supports #4)
# ===========================================================================
class TestHelpers:

    def test_ttl_seconds_1h(self):
        assert agent_mod._keepalive_ttl_seconds("1h") == 3600

    def test_ttl_seconds_5m(self):
        assert agent_mod._keepalive_ttl_seconds("5m") == 300

    def test_ttl_seconds_none_defaults_1h(self):
        assert agent_mod._keepalive_ttl_seconds(None) == 3600

    def test_ttl_seconds_unknown_defaults_1h(self):
        assert agent_mod._keepalive_ttl_seconds("off") == 3600

    def test_interval_1h_is_2880(self):
        assert agent_mod._keepalive_interval("1h") == pytest.approx(2880.0)

    def test_interval_5m_is_240(self):
        assert agent_mod._keepalive_interval("5m") == pytest.approx(240.0)

    def test_interval_floor_60(self):
        # A hypothetical tiny TTL must be floored at KEEPALIVE_MIN_INTERVAL_S.
        with patch.object(agent_mod, "_keepalive_ttl_seconds", return_value=10):
            assert agent_mod._keepalive_interval("whatever") == float(agent_mod.KEEPALIVE_MIN_INTERVAL_S)

    def test_constants(self):
        assert agent_mod.KEEPALIVE_REFRESH_FRACTION == pytest.approx(0.8)
        assert agent_mod.KEEPALIVE_MIN_INTERVAL_S == 60
        assert agent_mod.KEEPALIVE_MIN_CACHE_READ == 1000
        assert provider_mod.KEEPALIVE_MAX_OUTPUT_TOKENS == 1

    def test_is_hit_true_when_read_dominates(self):
        assert agent_mod._keepalive_is_hit(150_000, 4) is True

    def test_is_hit_false_on_write_signature(self):
        assert agent_mod._keepalive_is_hit(0, 150_000) is False

    def test_is_hit_false_below_floor(self):
        # read > write but tiny read is not a real prefix hit.
        assert agent_mod._keepalive_is_hit(500, 0) is False

    def test_is_hit_handles_none(self):
        assert agent_mod._keepalive_is_hit(None, None) is False


# ===========================================================================
# C. ping_cache — the ping request.  (Test plan #5)
# ===========================================================================
@pytest.mark.asyncio
class TestPingCache:

    async def test_ping_replays_parent_thinking_adaptive(self, tmp_path):
        # workspace-kdsn.199: the ping MUST replay the parent's EXACT thinking mode.
        # Anthropic keys the prompt cache on the extended-thinking mode AND effort, so
        # a thinking-off (or wrong-effort) ping is a guaranteed total miss + rewrite.
        # Verified live 2026-07-05: adaptive + max_tokens=1 is accepted and HITS.
        cfg = _make_agent_config(tmp_path, keepalive=True)
        client = _mock_client()
        with patch.object(provider_mod, "_get_client", return_value=client):
            await provider_mod.ping_cache(
                cfg, system="SYS", messages=[{"role": "user", "content": "hi"}],
                tools=None, cache_ttl="1h", model=MODEL_OPUS, thinking_level="max",
            )
        client.messages.create.assert_awaited_once()
        kw = client.messages.create.call_args.kwargs
        assert kw["max_tokens"] == 1                       # stays cheap even with thinking on
        assert kw["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert kw["output_config"]["effort"] == "max"      # effort matches parent (part of cache key)

    async def test_ping_thinking_off_stays_off(self, tmp_path):
        # A non-thinking parent -> the ping omits thinking too (matches the off prefix).
        cfg = _make_agent_config(tmp_path, keepalive=True)
        client = _mock_client()
        with patch.object(provider_mod, "_get_client", return_value=client):
            await provider_mod.ping_cache(
                cfg, system="SYS", messages=[{"role": "user", "content": "hi"}],
                tools=None, cache_ttl="1h", model=MODEL_OPUS, thinking_level="off",
            )
        kw = client.messages.create.call_args.kwargs
        assert kw["max_tokens"] == 1
        assert "thinking" not in kw
        assert "output_config" not in kw

    async def test_ping_cache_key_fields_match_parent_stream(self, tmp_path):
        # Guard for the .199 bug CLASS: the ping's cache-key inputs (system, tools,
        # messages, thinking, output_config) must be IDENTICAL to the parent stream's
        # for the same thinking_level -- only max_tokens may differ. Any divergence
        # silently busts the cache; this is the mock-free unit proxy for that.
        from openalph.provider import _build_anthropic_kwargs
        common = dict(api_model="claude-opus-4-8", system="SYS",
                      provider_messages=[{"role": "user", "content": "hi"}],
                      provider_tools=None, thinking_level="max", model_max_tokens=200000,
                      temperature=None, top_p=None, cache_ttl="1h")
        parent = _build_anthropic_kwargs(max_tokens=2048, **common)
        ping = _build_anthropic_kwargs(max_tokens=1, **common)
        for field in ("system", "tools", "messages", "thinking", "output_config"):
            assert parent.get(field) == ping.get(field), f"cache-key field {field!r} diverged parent vs ping"
        assert ping["max_tokens"] == 1 and parent["max_tokens"] == 2048

    async def test_returns_usage_with_cache_counters(self, tmp_path):
        cfg = _make_agent_config(tmp_path, keepalive=True)
        client = _mock_client(cache_read=123_456, cache_creation=7)
        with patch.object(provider_mod, "_get_client", return_value=client):
            usage = await provider_mod.ping_cache(
                cfg, system="SYS", messages=[{"role": "user", "content": "hi"}],
                tools=None, cache_ttl="1h", thinking_level="off",
            )
        assert usage.cache_read_tokens == 123_456
        assert usage.cache_creation_tokens == 7

    async def test_replays_same_prefix_inputs(self, tmp_path):
        cfg = _make_agent_config(tmp_path, keepalive=True)
        client = _mock_client()
        sys = "ORCHESTRATOR SYSTEM PROMPT"
        msgs = [{"role": "user", "content": "long history here"}]
        with patch.object(provider_mod, "_get_client", return_value=client):
            await provider_mod.ping_cache(
                cfg, system=sys, messages=msgs, tools=None, cache_ttl="1h", thinking_level="off",
            )
        kw = client.messages.create.call_args.kwargs
        # system is wrapped in a cache_control text block by _build_anthropic_kwargs
        assert kw["system"][0]["text"] == sys
        assert kw["system"][0]["cache_control"]["ttl"] == "1h"
        assert kw["messages"][-1]["role"] == "user"

    async def test_non_anthropic_is_noop(self, tmp_path):
        cfg = _make_agent_config(tmp_path, anthropic=False, keepalive=True)
        # Should not attempt any Anthropic client call; returns None.
        with patch.object(provider_mod, "_get_client") as gc:
            usage = await provider_mod.ping_cache(
                cfg, system="SYS", messages=[{"role": "user", "content": "hi"}],
                tools=None, cache_ttl="1h", model=MODEL_OPENAI, thinking_level="off",
            )
        assert usage is None
        gc.assert_not_called()


# ===========================================================================
# D. Arm gating — arm iff (subagent in batch) ∧ Anthropic ∧ flag.  (#1, #10)
# ===========================================================================
@pytest.mark.asyncio
class TestArmGating:

    async def _run_until_pings_or_timeout(self, agent, callbacks, stream_factory, exec_fn,
                                          release, expect_armed):
        with patch.object(agent_mod, "stream", side_effect=stream_factory), \
             patch.object(agent_mod, "execute_tool", side_effect=exec_fn), \
             patch.object(agent_mod, "ping_cache", new=AsyncMock(return_value=_usage(**HIT))) as ping, \
             patch.object(agent_mod, "_keepalive_interval", return_value=0.01):
            task = asyncio.create_task(agent.handle_input("go", room_id=ROOM, callbacks=callbacks))
            if expect_armed:
                armed = await _wait_for(lambda: ping.call_count > 0, timeout=1.0)
            else:
                # give several intervals; assert nothing fired
                await asyncio.sleep(0.15)
                armed = ping.call_count > 0
            count = ping.call_count
            release.set()
            await task
        return armed, count, ping

    async def test_armed_when_subagent_anthropic_flag_on(self, tmp_path):
        cfg = _make_agent_config(tmp_path, keepalive=True)
        agent = Agent(cfg)
        release = asyncio.Event()
        armed, count, _ = await self._run_until_pings_or_timeout(
            agent, {}, _subagent_stream(), _blocking_exec(release), release, expect_armed=True)
        assert armed and count >= 1, "keepalive must arm for a subagent batch on Anthropic + flag"

    async def test_not_armed_when_no_subagent_in_batch(self, tmp_path):
        cfg = _make_agent_config(tmp_path, keepalive=True)
        agent = Agent(cfg)
        release = asyncio.Event()
        armed, count, _ = await self._run_until_pings_or_timeout(
            agent, {}, _nonsub_stream(("shell",)), _blocking_exec(release), release, expect_armed=False)
        assert not armed and count == 0, "no subagent in batch → no keepalive"

    async def test_not_armed_when_flag_off(self, tmp_path):
        cfg = _make_agent_config(tmp_path, keepalive=False)
        agent = Agent(cfg)
        release = asyncio.Event()
        armed, count, _ = await self._run_until_pings_or_timeout(
            agent, {}, _subagent_stream(), _blocking_exec(release), release, expect_armed=False)
        assert not armed and count == 0, "flag off → no keepalive"

    async def test_not_armed_on_non_anthropic_provider(self, tmp_path):
        cfg = _make_agent_config(tmp_path, anthropic=False, keepalive=True)
        agent = Agent(cfg)
        release = asyncio.Event()
        armed, count, _ = await self._run_until_pings_or_timeout(
            agent, {}, _subagent_stream(), _blocking_exec(release), release, expect_armed=False)
        assert not armed and count == 0, "non-Anthropic provider → no keepalive"


# ===========================================================================
# E. Short subagent → zero pings.  (#2)
# ===========================================================================
@pytest.mark.asyncio
class TestShortSubagent:

    async def test_fast_subagent_no_pings(self, tmp_path):
        cfg = _make_agent_config(tmp_path, keepalive=True)
        agent = Agent(cfg)
        # execute_tool returns immediately; interval is much longer than the run.
        with patch.object(agent_mod, "stream", side_effect=_subagent_stream()), \
             patch.object(agent_mod, "execute_tool", new=AsyncMock(return_value=ToolResult(content="x", is_error=False))), \
             patch.object(agent_mod, "ping_cache", new=AsyncMock(return_value=_usage(**HIT))) as ping, \
             patch.object(agent_mod, "_keepalive_interval", return_value=0.5):
            result = await agent.handle_input("go", room_id=ROOM, callbacks={})
        assert ping.call_count == 0, "short subagent returns before first fire → zero pings"
        assert "all done" in result


# ===========================================================================
# F. Long subagent → periodic pings that replay the captured request.  (#3)
# ===========================================================================
@pytest.mark.asyncio
class TestLongSubagent:

    async def test_periodic_pings_then_disarm(self, tmp_path):
        cfg = _make_agent_config(tmp_path, keepalive=True)
        agent = Agent(cfg)
        release = asyncio.Event()
        with patch.object(agent_mod, "stream", side_effect=_subagent_stream()), \
             patch.object(agent_mod, "execute_tool", side_effect=_blocking_exec(release)), \
             patch.object(agent_mod, "ping_cache", new=AsyncMock(return_value=_usage(**HIT))) as ping, \
             patch.object(agent_mod, "_keepalive_interval", return_value=0.01):
            task = asyncio.create_task(agent.handle_input("go", room_id=ROOM, callbacks={}))
            await _wait_for(lambda: ping.call_count >= 3, timeout=1.0)
            count_at_release = ping.call_count
            release.set()
            await task
            after = ping.call_count
            await asyncio.sleep(0.1)  # several intervals with no active subagent
            assert ping.call_count == after, "pings must stop after the gather completes (disarm)"
        assert count_at_release >= 3, "a long subagent must trigger periodic pings"

    async def test_ping_replays_captured_request(self, tmp_path):
        cfg = _make_agent_config(tmp_path, keepalive=True)
        agent = Agent(cfg)
        release = asyncio.Event()
        with patch.object(agent_mod, "stream", side_effect=_subagent_stream()) as mock_stream, \
             patch.object(agent_mod, "execute_tool", side_effect=_blocking_exec(release)), \
             patch.object(agent_mod, "ping_cache", new=AsyncMock(return_value=_usage(**HIT))) as ping, \
             patch.object(agent_mod, "_keepalive_interval", return_value=0.01):
            task = asyncio.create_task(agent.handle_input("go", room_id=ROOM, callbacks={}))
            await _wait_for(lambda: ping.call_count >= 1, timeout=1.0)
            ping_kw = ping.call_args.kwargs
            release.set()
            await task
        first_stream_kw = mock_stream.call_args_list[0].kwargs
        # The ping must replay the SAME model/system/tools/cache_ttl the spawning request
        # used — all four are part of the Anthropic cache key.
        assert ping_kw["model"] == first_stream_kw["model"]
        assert ping_kw["system"] == first_stream_kw["system"]
        assert ping_kw["tools"] == first_stream_kw["tools"]
        assert ping_kw["cache_ttl"] == first_stream_kw["cache_ttl"]


# ===========================================================================
# G. LOAD-BEARING — cache-hit verification / miss abort.  (#4)
# ===========================================================================
@pytest.mark.asyncio
class TestHitVerification:

    async def test_hit_continues_looping(self, tmp_path):
        cfg = _make_agent_config(tmp_path, keepalive=True)
        agent = Agent(cfg)
        release = asyncio.Event()
        with patch.object(agent_mod, "stream", side_effect=_subagent_stream()), \
             patch.object(agent_mod, "execute_tool", side_effect=_blocking_exec(release)), \
             patch.object(agent_mod, "ping_cache", new=AsyncMock(return_value=_usage(**HIT))) as ping, \
             patch.object(agent_mod, "_keepalive_interval", return_value=0.01):
            task = asyncio.create_task(agent.handle_input("go", room_id=ROOM, callbacks={}))
            ok = await _wait_for(lambda: ping.call_count >= 3, timeout=1.0)
            release.set()
            await task
        assert ok, "healthy cache-read pings must keep the loop alive (multiple pings)"

    async def test_miss_aborts_loop_and_notifies(self, tmp_path, caplog):
        cfg = _make_agent_config(tmp_path, keepalive=True)
        agent = Agent(cfg)
        release = asyncio.Event()
        miss_notice = AsyncMock()
        callbacks = {"on_keepalive_miss": miss_notice}
        with patch.object(agent_mod, "stream", side_effect=_subagent_stream()), \
             patch.object(agent_mod, "execute_tool", side_effect=_blocking_exec(release)), \
             patch.object(agent_mod, "ping_cache", new=AsyncMock(return_value=_usage(**MISS))) as ping, \
             patch.object(agent_mod, "_keepalive_interval", return_value=0.01), \
             caplog.at_level(logging.WARNING):
            task = asyncio.create_task(agent.handle_input("go", room_id=ROOM, callbacks=callbacks))
            # First ping is a MISS → keepalive must abort. Wait for the notice.
            await _wait_for(lambda: miss_notice.await_count > 0, timeout=1.0)
            await asyncio.sleep(0.1)  # ensure no further pings after the miss
            count_after_abort = ping.call_count
            await asyncio.sleep(0.1)
            release.set()
            await task
        assert count_after_abort == 1, "on a write-signature miss, abort after exactly one ping"
        miss_notice.assert_awaited()  # operator notice emitted
        assert any("keepalive" in r.message.lower() and "miss" in r.message.lower()
                   for r in caplog.records), "miss must log at WARNING"

    async def test_miss_without_notice_callback_still_aborts(self, tmp_path):
        # on_keepalive_miss absent → must still abort (no crash).
        cfg = _make_agent_config(tmp_path, keepalive=True)
        agent = Agent(cfg)
        release = asyncio.Event()
        with patch.object(agent_mod, "stream", side_effect=_subagent_stream()), \
             patch.object(agent_mod, "execute_tool", side_effect=_blocking_exec(release)), \
             patch.object(agent_mod, "ping_cache", new=AsyncMock(return_value=_usage(**MISS))) as ping, \
             patch.object(agent_mod, "_keepalive_interval", return_value=0.01):
            task = asyncio.create_task(agent.handle_input("go", room_id=ROOM, callbacks={}))
            await asyncio.sleep(0.15)
            count = ping.call_count
            release.set()
            await task
        assert count == 1, "miss aborts after one ping even without a notice callback"


# ===========================================================================
# H. Prefix identity — replay the PRE-assistant-turn snapshot.  (#5, load-bearing)
# ===========================================================================
@pytest.mark.asyncio
class TestPrefixIdentity:

    async def test_ping_messages_exclude_assistant_tooluse_turn(self, tmp_path):
        """The captured messages must be the request suffix BEFORE the assistant
        tool_use turn was appended — i.e. it must NOT end on an assistant/tool_use
        message (that is an invalid request and a different cache breakpoint)."""
        cfg = _make_agent_config(tmp_path, keepalive=True)
        agent = Agent(cfg)
        release = asyncio.Event()
        with patch.object(agent_mod, "stream", side_effect=_subagent_stream()) as mock_stream, \
             patch.object(agent_mod, "execute_tool", side_effect=_blocking_exec(release)), \
             patch.object(agent_mod, "ping_cache", new=AsyncMock(return_value=_usage(**HIT))) as ping, \
             patch.object(agent_mod, "_keepalive_interval", return_value=0.01):
            task = asyncio.create_task(agent.handle_input("go", room_id=ROOM, callbacks={}))
            await _wait_for(lambda: ping.call_count >= 1, timeout=1.0)
            ping_msgs = ping.call_args.kwargs["messages"]
            release.set()
            await task
        # Same list content the FIRST stream() call used (pre-append snapshot).
        assert ping_msgs == mock_stream.call_args_list[0].kwargs["messages"]
        # And crucially not ending on an assistant tool_use turn.
        assert ping_msgs, "captured messages must be non-empty"
        last = ping_msgs[-1]
        assert last.get("role") != "assistant", "must not replay the assistant tool_use turn"
        assert "tool_calls" not in last


# ===========================================================================
# I. Cancellation — /stop cancels keepalive; no orphan task.  (#6)
# ===========================================================================
@pytest.mark.asyncio
class TestCancellation:

    async def test_cancel_stops_keepalive_no_orphan(self, tmp_path):
        cfg = _make_agent_config(tmp_path, keepalive=True)
        agent = Agent(cfg)
        release = asyncio.Event()
        with patch.object(agent_mod, "stream", side_effect=_subagent_stream()), \
             patch.object(agent_mod, "execute_tool", side_effect=_blocking_exec(release)), \
             patch.object(agent_mod, "ping_cache", new=AsyncMock(return_value=_usage(**HIT))) as ping, \
             patch.object(agent_mod, "_keepalive_interval", return_value=0.01):
            task = asyncio.create_task(agent.handle_input("go", room_id=ROOM, callbacks={}))
            await _wait_for(lambda: ping.call_count >= 1, timeout=1.0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            count_at_cancel = ping.call_count
            await asyncio.sleep(0.1)  # keepalive must be gone — no more pings
            assert ping.call_count == count_at_cancel, "no pings after cancellation"
        # No lingering keepalive coroutine tasks after cancellation.
        leftover = [t for t in asyncio.all_tasks()
                    if not t.done() and "_cache_keepalive" in str(t.get_coro())]
        assert leftover == [], f"orphaned keepalive tasks after cancel: {leftover}"


# ===========================================================================
# J. Multiple parallel subagents → ONE keepalive task.  (#7)
# ===========================================================================
@pytest.mark.asyncio
class TestMultipleSubagents:

    async def test_two_subagents_single_keepalive(self, tmp_path):
        cfg = _make_agent_config(tmp_path, keepalive=True)
        agent = Agent(cfg)
        release = asyncio.Event()
        seen = []

        # Count how many keepalive coroutines are spawned for the batch.
        spawns = {"n": 0}
        real_keepalive = Agent._cache_keepalive

        async def counting_keepalive(self, **kwargs):
            spawns["n"] += 1
            await real_keepalive(self, **kwargs)

        with patch.object(agent_mod, "stream", side_effect=_subagent_stream(("subagent", "subagent"))), \
             patch.object(agent_mod, "execute_tool", side_effect=_blocking_exec(release, seen)), \
             patch.object(agent_mod, "ping_cache", new=AsyncMock(return_value=_usage(**HIT))), \
             patch.object(Agent, "_cache_keepalive", counting_keepalive), \
             patch.object(agent_mod, "_keepalive_interval", return_value=0.02):
            task = asyncio.create_task(agent.handle_input("go", room_id=ROOM, callbacks={}))
            await _wait_for(lambda: spawns["n"] >= 1, timeout=1.0)
            await asyncio.sleep(0.05)  # give any erroneous second keepalive time to spawn
            release.set()
            await task
        assert seen.count("subagent") == 2, "both subagents dispatched"
        assert spawns["n"] == 1, "exactly ONE keepalive task for a batch of N subagents"


# ===========================================================================
# K. Ping error isolation — a ping failure never breaks the parent turn.  (#8)
# ===========================================================================
@pytest.mark.asyncio
class TestPingErrorIsolation:

    async def test_ping_raises_does_not_break_turn(self, tmp_path, caplog):
        cfg = _make_agent_config(tmp_path, keepalive=True)
        agent = Agent(cfg)
        release = asyncio.Event()

        calls = {"n": 0}

        async def flaky_ping(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient network blip")
            return _usage(**HIT)

        with patch.object(agent_mod, "stream", side_effect=_subagent_stream()), \
             patch.object(agent_mod, "execute_tool", side_effect=_blocking_exec(release)), \
             patch.object(agent_mod, "ping_cache", side_effect=flaky_ping), \
             patch.object(agent_mod, "_keepalive_interval", return_value=0.01), \
             caplog.at_level(logging.WARNING):
            task = asyncio.create_task(agent.handle_input("go", room_id=ROOM, callbacks={}))
            # After the raising ping, the loop must retry and land a healthy ping.
            await _wait_for(lambda: calls["n"] >= 2, timeout=1.0)
            release.set()
            result = await task
        assert "all done" in result, "parent turn completes normally despite a ping error"
        assert calls["n"] >= 2, "keepalive retried after a transient ping error"

    async def test_persistent_errors_abort_with_notice(self, tmp_path):
        """Audit F2/Gemini: persistent ping errors must abort after
        KEEPALIVE_MAX_CONSECUTIVE_ERRORS (with an operator notice), not retry forever."""
        cfg = _make_agent_config(tmp_path, keepalive=True)
        agent = Agent(cfg)
        release = asyncio.Event()
        miss = AsyncMock()
        attempts = {"n": 0}

        async def always_fail(*args, **kwargs):
            attempts["n"] += 1
            raise RuntimeError("API down")

        with patch.object(agent_mod, "stream", side_effect=_subagent_stream()), \
             patch.object(agent_mod, "execute_tool", side_effect=_blocking_exec(release)), \
             patch.object(agent_mod, "ping_cache", side_effect=always_fail), \
             patch.object(agent_mod, "_keepalive_interval", return_value=0.01):
            task = asyncio.create_task(
                agent.handle_input("go", room_id=ROOM, callbacks={"on_keepalive_miss": miss}))
            await _wait_for(lambda: miss.await_count > 0, timeout=2.0)
            await asyncio.sleep(0.05)  # ensure it stopped retrying after abort
            settled = attempts["n"]
            release.set()
            await task
        assert settled == agent_mod.KEEPALIVE_MAX_CONSECUTIVE_ERRORS, \
            "aborts after exactly MAX_CONSECUTIVE_ERRORS ping attempts"
        assert attempts["n"] == settled, "no further ping attempts after abort"
        miss.assert_awaited()  # operator gets a notice on persistent-failure abort


# ===========================================================================
# L. Miss notice plumbing — matrix callback exists + sends + logs system.
# ===========================================================================
@pytest.mark.skipif(not _matrix_imported, reason="matrix import failed")
@pytest.mark.asyncio
class TestMissNoticeWiring:

    def _make_bot(self):
        bot = MatrixBot.__new__(MatrixBot)
        bot.config = MagicMock()
        bot.config.user_id = "@agent:m.local"
        bot.agent = MagicMock()
        bot.agent.config = MagicMock()
        bot.agent.config.providers = {}
        bot.agent.config.model_aliases = {}
        bot.agent._read_registries = {}
        bot.send_notice = AsyncMock()
        bot._room_send_with_retry = AsyncMock()
        bot.session_log = MagicMock()
        bot.session_log.append = MagicMock()
        return bot

    async def test_callbacks_include_on_keepalive_miss(self):
        bot = self._make_bot()
        cbs = bot._build_agent_callbacks(ROOM, None)
        assert "on_keepalive_miss" in cbs
        assert callable(cbs["on_keepalive_miss"])

    async def test_on_keepalive_miss_sends_notice_and_logs_system(self):
        bot = self._make_bot()
        cbs = bot._build_agent_callbacks(ROOM, None)
        await cbs["on_keepalive_miss"](ROOM)
        bot.send_notice.assert_awaited()
        text = bot.send_notice.await_args.args[1] if len(bot.send_notice.await_args.args) > 1 \
            else bot.send_notice.await_args.kwargs.get("text", "")
        assert "keepalive" in text.lower()
        # Logged as a system entry (excluded from LLM context), mirroring redaction/cache-bust.
        bot.session_log.append.assert_called()
        kw = bot.session_log.append.call_args.kwargs
        assert kw.get("role") == "system"



# ===========================================================================
# M. Real-path wiring (workspace-kdsn.199) — the ARM SITE must thread the
#    parent's effective thinking level into the ping. Drives a REAL Agent
#    through handle_input (only stream + ping_cache mocked), per the
#    tool-management "one lesson": a mocked-seam test can't catch a wiring gap.
# ===========================================================================
@pytest.mark.asyncio
class TestKeepaliveThreadsParentThinking:

    async def _run_and_capture(self, cfg):
        agent = Agent(cfg)
        release = asyncio.Event()
        with patch.object(agent_mod, "stream", side_effect=_subagent_stream()), \
             patch.object(agent_mod, "execute_tool", side_effect=_blocking_exec(release)), \
             patch.object(agent_mod, "ping_cache", new=AsyncMock(return_value=_usage(**HIT))) as ping, \
             patch.object(agent_mod, "_keepalive_interval", return_value=0.01):
            task = asyncio.create_task(agent.handle_input("go", room_id=ROOM, callbacks={}))
            await _wait_for(lambda: ping.call_count > 0, timeout=1.0)
            release.set()
            await task
        return ping

    async def test_ping_receives_parent_effective_thinking(self, tmp_path):
        cfg = _make_agent_config(tmp_path, keepalive=True, thinking="high")
        ping = await self._run_and_capture(cfg)
        assert ping.call_count >= 1
        assert ping.call_args.kwargs.get("thinking_level") == "high", \
            "arm site must thread the parent's effective thinking level into ping_cache"

    async def test_ping_thinking_off_when_parent_off(self, tmp_path):
        # Default config.thinking == "off" -> ping must also be off (matches the prefix).
        cfg = _make_agent_config(tmp_path, keepalive=True)
        ping = await self._run_and_capture(cfg)
        assert ping.call_args.kwargs.get("thinking_level") == "off"


# ===========================================================================
# N. Live-API cache-hit smoke (workspace-kdsn.199 acceptance criterion).
#    Mocks cannot exercise Anthropic's cache-key semantics, so this asserts a
#    real parent-mode ping HITS a thinking-ON prefix. Opt-in: set
#    OPENALPH_LIVE_ANTHROPIC_KEY to run; skipped otherwise (keeps CI green).
# ===========================================================================
_LIVE_KEY = os.environ.get("OPENALPH_LIVE_ANTHROPIC_KEY")


@pytest.mark.skipif(not _LIVE_KEY, reason="set OPENALPH_LIVE_ANTHROPIC_KEY to run the live cache-hit smoke test")
@pytest.mark.asyncio
class TestLiveCacheHit:

    async def test_parent_mode_ping_hits_thinking_on_prefix(self):
        import time as _t
        import anthropic
        from openalph.provider import _build_anthropic_kwargs
        api_model = "claude-opus-4-8"
        marker = f"kdsn199-{int(_t.time())}"
        system = ("Cache smoke harness. Answer in one word. "
                  + ("The quick brown fox jumps over the lazy dog. " * 500)
                  + f" marker:{marker}")
        user = [{"role": "user", "content": "Reply with exactly: OK"}]
        client = anthropic.AsyncAnthropic(api_key=_LIVE_KEY)

        def _kw(level, mt):
            return _build_anthropic_kwargs(
                api_model=api_model, system=system,
                provider_messages=[dict(m) for m in user], provider_tools=None,
                max_tokens=mt, thinking_level=level, model_max_tokens=200000,
                temperature=None, top_p=None, cache_ttl="1h")

        w = await client.messages.create(**_kw("max", 2048))   # WRITE the thinking-ON prefix
        assert w.usage.cache_creation_input_tokens > 0
        await asyncio.sleep(2)
        p = await client.messages.create(**_kw("max", 1))       # PING in parent mode, mt=1
        assert p.usage.cache_read_input_tokens > 0, "parent-mode ping must HIT the thinking-ON prefix"
        assert p.usage.cache_creation_input_tokens == 0, "parent-mode ping must not rewrite the prefix"
