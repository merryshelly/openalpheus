"""Shared fixtures for the kdsn.330 async-subagent red suite.

Real-path discipline (tool-management "one lesson"): REAL Agent, REAL
SessionLog, REAL MatrixBot callback construction; ONLY the provider
(main stream + sub complete) and the nio client are mocked.

The async-subagent implementation does not exist yet — the seam names the
red suite expects are those ruled in specs/kdsn.330-p1-test-plan.md
(R1-R16). Red-verify must show these files failing for the right reasons
(missing feature), with sync-mode and unrelated tests staying GREEN.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openalph.agent import Agent
from openalph.config import AgentConfig, MatrixConfig, ProviderConfig
from openalph.provider import Response, Usage, StreamEvent, ToolCall
from openalph.session import SessionLog
from openalph.matrix import MatrixBot

ROOM = "!async-room:matrix.local"
ROOM_B = "!async-room-b:matrix.local"
AGENT_USER = "@agent:matrix.local"
OP_USER = "@op:matrix.local"
SUB_SENTINEL = "__sub__"

DEFAULT_TASK = "Summarize the florbisticate report"


# --- config helpers -------------------------------------------------------

def make_matrix_config(**kwargs):
    defaults = dict(
        homeserver="https://matrix.local",
        user_id=AGENT_USER,
        device_id="TEST",
        password="test-pw",
        access_token=None,
        context_reserve=16384,
        sync_timeout=30000,
        retry_base=1,
        retry_max=10,
    )
    defaults.update(kwargs)
    return MatrixConfig(**defaults)


def make_provider(key="anthropic", type="anthropic", api_key="sk-test", **kw):
    defaults = dict(key=key, type=type, api_key=api_key)
    defaults.update(kw)
    return ProviderConfig(**defaults)


def make_agent_config(workspace, tools=("subagent", "shell"), **kwargs):
    _setup_workspace(workspace, tools=tools)
    defaults = dict(
        name="test-async",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": make_provider()},
        max_iterations=25,
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=make_matrix_config(),
    )
    defaults.update(kwargs)
    defaults["workspace"] = workspace  # Path — prompt.py does workspace / "skills"
    return AgentConfig(**defaults)


def _setup_workspace(workspace, tools=("subagent", "shell")):
    tools_dir = workspace / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    for name in tools:
        (tools_dir / f"{name}.toml").write_text("[config]\n")
    (workspace / "sessions").mkdir(exist_ok=True)
    return workspace


# --- provider fakes -------------------------------------------------------

def make_sub_response(content="Sub report: all done.", model="anthropic/claude-sonnet-4-20250514",
                      usage=None, **kw):
    if usage is None:
        usage = Usage(input_tokens=100, output_tokens=50)
    return Response(content=content, model=model, usage=usage,
                    stop_reason="end_turn", **kw)


def sub_complete_factory(content="Sub report: all done.", delay=0.0, exc=None, usage=None):
    """Async stand-in for openalph.tools.subagent.complete."""
    async def _complete(**kwargs):
        if delay:
            await asyncio.sleep(delay)
        if exc is not None:
            raise exc
        return make_sub_response(content=content, usage=usage)
    return _complete


def make_main_stream(tool_calls=None, final_text="Done"):
    """Main-loop stream factory: emits the given tool_calls on call 1, then text.

    Returns (stream_fn, payloads) — payloads captures the wire messages of
    every call (for A05/A06 pins).
    """
    payloads = []
    call_idx = [0]

    async def _stream(*, config=None, system=None, messages=None, tools=None,
                      model="test", thinking=None, cache_ttl=None, **kw):
        payloads.append(list(messages))
        call_idx[0] += 1
        if tool_calls is not None and call_idx[0] == 1:
            yield StreamEvent(
                type="done",
                response=Response(content="", tool_calls=list(tool_calls), model=model,
                                  usage=Usage(input_tokens=10, output_tokens=5),
                                  stop_reason="tool_use"),
                stop_reason="tool_use", model=model)
        else:
            yield StreamEvent(type="text", content=final_text)
            yield StreamEvent(
                type="done",
                response=Response(content=final_text, model=model,
                                  usage=Usage(input_tokens=10, output_tokens=5),
                                  stop_reason="end_turn"),
                stop_reason="end_turn", model=model)

    return _stream, payloads


def sub_tool_call(task=DEFAULT_TASK, background=True, **extra):
    """Tool INPUT dict (execute_tool's `input=` shape). Dispatch id flows via
    callbacks["call_id"] on the real path — set it in the callbacks dict."""
    inp = {"task": task, "background": background}
    inp.update(extra)
    return inp


def sub_tool_call_tc(call_id="tc_async_1", task=DEFAULT_TASK, background=True, **extra):
    """Provider-wire ToolCall (for make_main_stream tool_calls)."""
    return ToolCall(id=call_id, name="subagent",
                    input=sub_tool_call(task=task, background=background, **extra))


# --- bot/agent construction (real path) -----------------------------------

def build_bot(tmp_path, agent=None, tools=("subagent", "shell"), **agent_kw):
    """REAL Agent + REAL MatrixBot (AsyncClient patched) + REAL SessionLog."""
    if agent is None:
        config = make_agent_config(tmp_path, tools=tools, **agent_kw)
        agent = Agent(config)
    matrix_config = make_matrix_config()
    with patch("openalph.matrix.AsyncClient"):
        bot = MatrixBot(agent, matrix_config)
    bot.client = MagicMock()
    bot.client.room_send = AsyncMock(return_value=MagicMock(event_id="$resp1"))
    bot.client.room_typing = AsyncMock()
    bot._room_send_with_retry = AsyncMock(return_value=MagicMock(event_id="$evt1"))
    bot.send = AsyncMock()
    bot.send_notice = AsyncMock()
    bot._synced = True
    if not hasattr(bot, "session_log") or bot.session_log is None:
        bot.session_log = SessionLog(tmp_path, AGENT_USER)
    bot._active_rooms = {ROOM}
    return bot, agent


def real_callbacks(bot, room=ROOM, turn_source="live"):
    """The REAL callback-construction path (MatrixBot._build_agent_callbacks)."""
    return bot._build_agent_callbacks(room, turn_source)


# --- async waiters / inspectors -------------------------------------------

async def await_terminal(agent, room, dispatch_id, timeout=5.0):
    """Wait until the dispatch leaves `running` (any terminal state)."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while True:
        rec = agent._dispatch_ledger.get(room, {}).get(dispatch_id)
        if rec is not None and rec.state != "running":
            return rec
        if loop.time() > deadline:
            raise TimeoutError(
                f"dispatch {dispatch_id} still running after {timeout}s")
        await asyncio.sleep(0.01)


async def settle(timeout=0.3):
    """Let pending background tasks (terminal handlers, fires) run."""
    await asyncio.sleep(timeout)
    await asyncio.sleep(0)


def ledger_entries(session_log, room, event=None):
    out = []
    for e in session_log.read(room):
        if e.get("role") != "system":
            continue
        ev = e.get("event", "")
        if event is None:
            if ev.startswith("subagent"):
                out.append(e)
        elif ev == event:
            out.append(e)
    return out


def pending_events(agent, room):
    return list(getattr(agent, "_pending_events", {}).get(room, []))


def history_user_messages(agent, room):
    return [m for m in agent.history(room) if m.get("role") == "user"]


# --- keepalive lifecycle recorder -----------------------------------------

class KeepaliveRecorder:
    """Wraps Agent._maybe_arm_cache_keepalive, recording (task, stop) pairs.

    Lets the red suite pin the disarm re-gate (R9) without patching ping
    internals: stop-event state + task liveness ARE the observable.
    """

    def __init__(self):
        self.arms = []          # list of (task-or-None, stop-or-None)

    def wrap(self, agent):
        original = agent._maybe_arm_cache_keepalive
        # Stub the ping loop body: the armed task must stay ALIVE (so the
        # re-gate is observable) without doing live HTTP. Cancellation still
        # interrupts Event().wait(), so disarm semantics are preserved.
        async def sleeping_loop(*a, **k):
            await asyncio.Event().wait()

        agent._cache_keepalive = sleeping_loop

        async def recorder(*args, **kwargs):
            task, stop = await original(*args, **kwargs)
            self.arms.append((task, stop))
            return task, stop

        return recorder


@pytest.fixture
def keepalive_recorder():
    return KeepaliveRecorder()


# --- default main-loop stream ----------------------------------------------

def _plain_stream(*, config=None, system=None, messages=None, tools=None,
                  model="test", thinking=None, cache_ttl=None, **kw):
    async def _gen():
        yield StreamEvent(type="text", content="Done.")
        yield StreamEvent(type="done",
                          response=Response(content="Done.", model=model,
                                            usage=Usage(input_tokens=10, output_tokens=5),
                                            stop_reason="end_turn"),
                          stop_reason="end_turn", model=model)
    return _gen()


@pytest.fixture
def default_main_stream():
    """Autouse-per-module default: the main loop never touches a real provider.

    Tests needing tool_calls or specific main-loop behavior apply their own
    inner patch("openalph.agent.stream", ...) which overrides this for the
    duration of its block.
    """
    with patch("openalph.agent.stream",
               side_effect=lambda **kw: _plain_stream(**kw)):
        yield
