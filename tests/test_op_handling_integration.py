"""RED suite — op-handling real-path integration (design "the one lesson").

Authored from op-handling-test-plan.md: L1 real-path #19, A05/A06 op-path
copies #20/#21, and L2 real-path #26 + A05 op-guard copy #27.  TESTS ONLY.

Golden rule: at least one test per layer drives a REAL Agent through the REAL
callback path (MatrixBot._build_agent_callbacks), mocking ONLY the provider
(LLM stream) and the nio client.  **execute_tool is NEVER mocked** — it is the
code under test (L1 redaction tail / L2 shell guard live inside it).

RED honesty (all tests here are BEHAVIORAL RED — no new-symbol imports needed):
  * #19/#20/#21: with L1 absent, execute_tool's redaction tail does not add the
    value pass, so the long non-pattern FAKE_SECRET survives verbatim into the
    tool-result history/replay entry → the "raw secret absent" assertions fail.
  * #26/#27: with the L2 guard absent, the real `op read` subprocess runs (auth
    error — no OP_SERVICE_ACCOUNT_TOKEN in the test env, fake refs, fast-fail)
    and its output — NOT the redirect message — lands in the tool entry → the
    "op-run present" assertions fail.
  Neither is a collection/typo error; both are "feature absent" failures.

Deviations from the plain A05/A06 originals (documented in red-report.md):
  * The originals mock execute_tool and are green-by-construction.  #20/#21/#27
    do NOT mock execute_tool (forbidden for op-path) and additionally assert
    the redaction/guard happened — so they are RED now and faithful once green.
  * #20/#27 stay faithful to test_guidance_injection.py::A05 call style
    (handle_input with no callbacks; reminders still inject).  #19/#26 use the
    full production callbacks from bot._build_agent_callbacks (plan's explicit
    "real callback path" requirement for the one-lesson test per layer).

Helpers are copied from test_guidance_integration.py /
test_file_search_integration.py (no cross-test-module imports in this suite).
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openalph.agent import Agent
from openalph.config import AgentConfig, ProviderConfig, MatrixConfig
from openalph.provider import Response, Usage, StreamEvent, ToolCall
from openalph.matrix import MatrixBot
from openalph.session import SessionLog
from openalph.tools import ToolResult, execute_tool


ROOM = "!op-integ:matrix.local"
AGENT_USER = "@agent:matrix.local"
REMINDER_TAG_OPEN = "<system-reminder>"

# Long, high-entropy value matching NONE of the CREDENTIAL_PATTERNS — only the
# L1 value pass can redact it.  Fake, not a real secret.  No shell-hostile chars.
FAKE_SECRET = "Zx9-not-a-pattern-sudo-like-secret-8823"


# --- Helpers (copied real-path pattern) ------------------------------------

def _cfg(workspace, **kw):
    defaults = dict(
        name="test-op-integ",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": ProviderConfig(
            key="anthropic", type="anthropic", api_key="sk-test",
            base_url=None, quirks=[],
        )},
        workspace=workspace,
        max_iterations=100,
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=None,
        reminders=True,
    )
    defaults.update(kw)
    return AgentConfig(**defaults)


def _setup_workspace(tmp_path, tools=("shell", "file_read", "file_write",
                                       "file_edit", "todo_write", "memory_search")):
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir(exist_ok=True)
    for name in tools:
        (tools_dir / f"{name}.toml").write_text("[config]\n")
    return tmp_path


def _make_capturing_stream(tool_iterations=6, final_text="Done", tool_name="shell",
                           tool_input=None):
    """Stream factory: tool_use N times then text.  Captures wire payloads.

    Parameterized on tool_name/tool_input so op-path turns can script an
    `echo <secret>` or `op read` shell call while keeping the exact
    cache-safety structure of the original A05 stream.
    """
    payloads = []
    call_idx = [0]

    async def _stream(*, config=None, system=None, messages=None,
                      tools=None, model="test", thinking=None,
                      cache_ttl=None, **kw):
        payloads.append(list(messages))
        call_idx[0] += 1
        if tools is not None and call_idx[0] <= tool_iterations:
            inp = tool_input if tool_input is not None else {"command": f"echo {call_idx[0]}"}
            tc = ToolCall(id=f"tc_{call_idx[0]}", name=tool_name, input=inp)
            yield StreamEvent(type="tool_done", tool_index=0, tool_call=tc)
            yield StreamEvent(
                type="done",
                response=Response(
                    content="", tool_calls=[tc], model=model,
                    usage=Usage(input_tokens=10, output_tokens=5),
                    stop_reason="tool_use"),
                stop_reason="tool_use", model=model)
        else:
            yield StreamEvent(type="text", content=final_text)
            yield StreamEvent(
                type="done",
                response=Response(
                    content=final_text, model=model,
                    usage=Usage(input_tokens=10, output_tokens=5),
                    stop_reason="end_turn"),
                stop_reason="end_turn", model=model)

    return _stream, payloads


def _make_bot_with_real_agent(tmp_path, agent=None, tools_list=None, **agent_kw):
    ws = _setup_workspace(tmp_path, tools=tools_list) if tools_list \
        else _setup_workspace(tmp_path)
    config = _cfg(ws, **agent_kw)
    if agent is None:
        agent = Agent(config)

    matrix_config = MatrixConfig(
        homeserver="https://matrix.local",
        user_id=AGENT_USER,
        device_id="TEST",
        password="test-password",
        access_token=None,
        context_reserve=16384,
        sync_timeout=30000,
        retry_base=1,
        retry_max=10,
    )
    bot = MatrixBot.__new__(MatrixBot)
    bot.config = matrix_config
    bot.agent = agent
    bot.client = MagicMock()
    bot.client.room_send = AsyncMock(return_value=MagicMock(event_id="$resp1"))
    bot.client.room_typing = AsyncMock()
    bot._current_room = None
    bot._synced = True
    bot._active_rooms = set()
    bot._room_effort = {}
    bot._room_cache_ttl = {}
    bot._room_timesense = {}
    bot._halted_rooms = set()
    bot._background_tasks = set()
    bot._session_locks = {}
    bot.session_log = MagicMock()
    bot.session_log.append = MagicMock()
    bot.session_log.build_context = MagicMock(return_value=[])
    bot.session_log.read = MagicMock(return_value=[])
    bot.session_log.last_event_id = MagicMock(return_value=None)
    bot.session_log.usage_totals = MagicMock(return_value={})
    bot.heartbeat = MagicMock()
    bot.heartbeat.is_active = MagicMock(return_value=False)
    bot.umbral = MagicMock()
    bot.umbral.is_active = MagicMock(return_value=False)
    bot._steering_inbox = {}
    bot._active_turns = set()
    return bot, agent


@pytest.fixture(autouse=True)
def _clean_api_key_cache():
    """Snapshot + restore _api_key_cache so no real turn can leak module state."""
    from openalph.tools import _api_key_cache
    snapshot = dict(_api_key_cache)
    try:
        yield
    finally:
        _api_key_cache.clear()
        _api_key_cache.update(snapshot)


def _tool_entries(history):
    return [m for m in history if m.get("role") == "tool"]


# ===========================================================================
# L1 real-path (#19) — the one lesson for the redaction layer
# ===========================================================================

class TestRealPathKnownSecretRedaction:

    @pytest.mark.asyncio
    async def test_realpath_known_secret_redacted_in_history(self, tmp_path):
        """#19: REAL Agent + REAL callbacks; provider api_key = long fake secret;
        the mocked provider scripts a single `echo <secret>` shell call; driving
        a real turn must land a REDACTED tool-result entry in agent history.

        execute_tool is NOT mocked.  RED (L1 absent): the non-pattern secret
        survives the pattern-only redaction → raw value present in history.
        """
        bot, agent = _make_bot_with_real_agent(
            tmp_path,
            providers={"anthropic": ProviderConfig(
                key="anthropic", type="anthropic", api_key=FAKE_SECRET,
                base_url=None, quirks=[])},
            max_iterations=10,
        )
        cb = bot._build_agent_callbacks(ROOM, None)
        stream_fn, _ = _make_capturing_stream(
            tool_iterations=1, tool_name="shell",
            tool_input={"command": f"echo '{FAKE_SECRET}'"})
        with patch("openalph.agent.stream", side_effect=stream_fn):
            await agent.handle_input("run it", room_id=ROOM, callbacks=cb)

        tools = _tool_entries(agent.history(ROOM))
        assert tools, "expected at least one tool-result entry in history"
        joined = " ".join(str(m.get("content", "")) for m in tools)
        assert FAKE_SECRET not in joined, \
            "raw known secret must NOT survive into the tool-result history entry"
        assert "[REDACTED:known_secret]" in joined, \
            "the value pass must redact the known secret in the real pipeline"


# ===========================================================================
# L1 cache-safety op-path copies (#20 A05, #21 A06)
# ===========================================================================

class TestOpRedactionCacheGuardrails:

    @pytest.mark.asyncio
    async def test_A05_cache_safety_strict_prefix_op_redaction_turn(self, tmp_path):
        """#20 (A05 copy): call N messages are a strict prefix of call N+1 — on a
        turn whose tool results are L1-redacted.

        Faithful copy of test_guidance_injection.py::test_A05_cache_safety_strict_prefix
        (:302), driving REAL `echo <secret>` shell calls (execute_tool NOT mocked)
        with provider api_key = long fake secret.  The extra redaction assertion
        makes it RED now (L1 absent) and keeps it a faithful cache-safety guard.
        """
        ws = _setup_workspace(tmp_path)
        config = _cfg(ws, max_iterations=10, providers={"anthropic": ProviderConfig(
            key="anthropic", type="anthropic", api_key=FAKE_SECRET,
            base_url=None, quirks=[])})
        agent = Agent(config)
        stream_fn, payloads = _make_capturing_stream(
            tool_iterations=7, tool_name="shell",
            tool_input={"command": f"echo '{FAKE_SECRET}'"})
        with patch("openalph.agent.stream", side_effect=stream_fn):
            await agent.handle_input("work", room_id=ROOM)

        # Phase 1: strict-prefix property (object identity across calls).
        for i in range(len(payloads) - 1):
            cur = payloads[i]
            nxt = payloads[i + 1]
            assert len(nxt) >= len(cur), (
                f"Call {i+1} has fewer messages ({len(nxt)}) than call {i} ({len(cur)})")
            for j in range(len(cur)):
                assert cur[j] is nxt[j], (
                    f"Message {j} is a DIFFERENT object in call {i+1} vs {i} — cache bust risk.")

        # Phase 2: a mid-loop reminder MUST have been injected (test meaningfulness).
        history = agent.history(ROOM)
        reminders = [m for m in history
                     if REMINDER_TAG_OPEN in str(m.get("content", ""))]
        assert reminders, (
            "Cache-safety test requires a mid-loop reminder injection; "
            "no <system-reminder> found — engine integration missing.")

        # Feature assertion (RED now): every tool-result entry is L1-redacted.
        tools = _tool_entries(history)
        assert tools, "expected tool-result entries in history"
        joined = " ".join(str(m.get("content", "")) for m in tools)
        assert FAKE_SECRET not in joined, \
            "raw known secret must NOT survive into any tool-result entry"
        assert "[REDACTED:known_secret]" in joined, \
            "L1 value pass must redact the known secret on this turn"

    @pytest.mark.asyncio
    async def test_A06_replay_identity_op_redacted_transcript(self, tmp_path):
        """#21 (A06 copy): build_context reproduces the redacted tool-result entry
        byte-identically, and replays the reminder entry verbatim.

        Faithful copy of test_guidance_injection.py::test_A06_replay_identity (:335).
        The tool-result content is produced by the REAL execute_tool path (secret
        in agent_config.providers), so RED (L1 absent) = raw secret present in the
        replayed content; GREEN = verbatim replay of the redacted string.
        """
        config = _cfg(tmp_path, providers={"anthropic": ProviderConfig(
            key="anthropic", type="anthropic", api_key=FAKE_SECRET,
            base_url=None, quirks=[])})
        # Produce the tool output through the REAL pipeline (execute_tool NOT mocked).
        tool_result = await execute_tool(
            name="shell",
            input={"command": f"echo '{FAKE_SECRET}'"},
            tool_config={"default_timeout": 30, "max_output": 50000},
            agent_config=config,
        )
        tool_output = tool_result.content

        sl = SessionLog(workspace=tmp_path, agent_user_id=AGENT_USER)
        framed = (f"{REMINDER_TAG_OPEN}\n"
                  "You have not consulted memory this session.\n"
                  "</system-reminder>")
        sl.append(role="user", sender="@op:x", room=ROOM, event_id="$e1",
                  content="run the command")
        sl.append(role="assistant", sender=AGENT_USER, room=ROOM, event_id=None,
                  content="", tool_calls=[
                      {"name": "shell", "id": "tc_1",
                       "input": {"command": f"echo '{FAKE_SECRET}'"}}])
        sl.append(role="tool", sender=AGENT_USER, room=ROOM, event_id=None,
                  call_id="tc_1", name="shell", output=tool_output)
        sl.append(role="user", sender=AGENT_USER, room=ROOM, event_id=None,
                  content=framed, source="reminder", trigger="memory-salience")
        sl.append(role="assistant", sender=AGENT_USER, room=ROOM, event_id=None,
                  content="OK, searching memory")

        context = sl.build_context(ROOM)

        # A06 core: the reminder entry replays verbatim, appears once.
        reminder_msgs = [m for m in context
                         if REMINDER_TAG_OPEN in str(m.get("content", ""))]
        assert len(reminder_msgs) == 1, "reminder must appear once in rebuilt context"
        assert reminder_msgs[0]["content"] == framed, \
            "build_context must replay reminder content VERBATIM (A06)"
        assert reminder_msgs[0]["role"] == "user"

        # Feature assertion (RED now): the replayed tool entry is L1-redacted.
        tool_msgs = [m for m in context if m.get("role") == "tool"]
        assert tool_msgs, "the tool-result entry must survive build_context (matched tool_call)"
        tool_content = str(tool_msgs[0].get("content", ""))
        assert FAKE_SECRET not in tool_content, \
            "raw known secret must NOT be present in the replayed tool entry"
        assert "[REDACTED:known_secret]" in tool_content, \
            "the replayed tool entry must carry the L1-redacted label"


# ===========================================================================
# L2 real-path (#26) — the one lesson for the guard layer
# ===========================================================================

class TestRealPathOpReadBlocked:

    @pytest.mark.asyncio
    async def test_realpath_op_read_blocked_in_history(self, tmp_path):
        """#26: REAL Agent + REAL callbacks; the mocked provider scripts a single
        `op read "op://x"` shell call; driving a real turn must land the redirect
        message + is_error=True in history, and no `op` process output.

        execute_tool is NOT mocked.  RED (L2 guard absent): the real `op read`
        subprocess runs and its auth-failure text — NOT the redirect — lands in
        the tool entry → the "op-run present" assertion fails.
        """
        bot, agent = _make_bot_with_real_agent(tmp_path, max_iterations=10)
        cb = bot._build_agent_callbacks(ROOM, None)
        stream_fn, _ = _make_capturing_stream(
            tool_iterations=1, tool_name="shell",
            tool_input={"command": 'op read "op://x/y"'})
        with patch("openalph.agent.stream", side_effect=stream_fn):
            await agent.handle_input("read the secret", room_id=ROOM, callbacks=cb)

        tools = _tool_entries(agent.history(ROOM))
        assert tools, "expected a tool-result entry in history"
        entry = tools[0]
        content = str(entry.get("content", ""))
        assert entry.get("is_error") is True, \
            "a blocked op read must be surfaced as an error result"
        assert "op-run" in content, \
            "the tool entry must be the redirect message (guard ran before subprocess)"
        assert "op inject" in content
        # The op subprocess must NOT have run: its auth-failure text must be absent.
        assert "No accounts configured" not in content
        assert "OP_SERVICE_ACCOUNT_TOKEN" not in content


# ===========================================================================
# L2 cache-safety op-guard copy (#27 A05)
# ===========================================================================

class TestOpGuardCacheGuardrails:

    @pytest.mark.asyncio
    async def test_A05_cache_safety_strict_prefix_op_guard_turn(self, tmp_path):
        """#27 (A05 copy): call N messages are a strict prefix of call N+1 — on a
        turn whose `op read` shell calls are blocked+redirected by the L2 guard.

        Faithful copy of test_guidance_injection.py::test_A05_cache_safety_strict_prefix
        (:302), driving REAL `op read` shell calls (execute_tool NOT mocked).  The
        extra redirect assertion makes it RED now (L2 guard absent) and keeps it a
        faithful cache-safety guard once green.
        """
        ws = _setup_workspace(tmp_path)
        config = _cfg(ws, max_iterations=10)
        agent = Agent(config)
        stream_fn, payloads = _make_capturing_stream(
            tool_iterations=7, tool_name="shell",
            tool_input={"command": 'op read "op://x/y"'})
        with patch("openalph.agent.stream", side_effect=stream_fn):
            await agent.handle_input("work", room_id=ROOM)

        # Phase 1: strict-prefix property (object identity across calls).
        for i in range(len(payloads) - 1):
            cur = payloads[i]
            nxt = payloads[i + 1]
            assert len(nxt) >= len(cur), (
                f"Call {i+1} has fewer messages ({len(nxt)}) than call {i} ({len(cur)})")
            for j in range(len(cur)):
                assert cur[j] is nxt[j], (
                    f"Message {j} is a DIFFERENT object in call {i+1} vs {i} — cache bust risk.")

        # Phase 2: a mid-loop reminder MUST have been injected (test meaningfulness).
        history = agent.history(ROOM)
        reminders = [m for m in history
                     if REMINDER_TAG_OPEN in str(m.get("content", ""))]
        assert reminders, (
            "Cache-safety test requires a mid-loop reminder injection; "
            "no <system-reminder> found — engine integration missing.")

        # Feature assertion (RED now): every op-read tool entry is the redirect.
        tools = _tool_entries(history)
        assert tools, "expected tool-result entries in history"
        for entry in tools:
            content = str(entry.get("content", ""))
            assert "op-run" in content, \
                "each blocked op read must be redirected (guard ran before subprocess)"
            assert entry.get("is_error") is True
            assert "No accounts configured" not in content, \
                "the op subprocess must not have run"
