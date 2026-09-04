"""RED suite — Bundle-2 integration (design §10-bullet-5).

Real-path per the tool-management "one lesson": a REAL Agent + REAL
MatrixBot._build_agent_callbacks; ONLY the provider stream and the nio
client are mocked.  These tests exercise the new tools through the SAME
pipeline production uses (execute_tool → per-room read registry;
redact → truncate → wrap, V6) and pin the cross-cutting invariants.

RED honesty: file_patch/grep/glob are unregistered today, so
``discover_tools`` RAISES ``ToolError`` on a workspace whose tools/ contains
their TOMLs.  Constructing a real Agent from such a workspace would crash at
COLLECTION-adjacent runtime — forbidden.  We therefore:
  * drive execute_tool-level tests from a STANDARD workspace (construction
    succeeds; the new-tool call returns "Unknown tool" → honest red);
  * wrap discovery/inheritance Agent construction and convert any ToolError
    into a clean ``pytest.fail`` (a behavior failure, never an import error).

The A05 strict-prefix and A06 replay-identity guardrails are faithful COPIES
of test_guidance_injection.py:302/:335 (that file is FORBIDDEN to modify),
here exercised on turns whose stream/transcript name the NEW tools — pinning
V5 ("static tool surface; A05/A06 stay green").

Helpers copied from test_guidance_integration.py (no cross-test-module
imports exist in this suite → copy per convention).
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openalph.agent import Agent
from openalph.config import AgentConfig, ProviderConfig, MatrixConfig
from openalph.provider import Response, Usage, StreamEvent, ToolCall
from openalph.matrix import MatrixBot
from openalph.session import SessionLog
from openalph.tools import (
    execute_tool, tool_schemas,
    ToolResult, ToolError,
)


ROOM = "!fs-integ:matrix.local"
AGENT_USER = "@agent:matrix.local"
REMINDER_TAG_OPEN = "<system-reminder>"

FAKE_SECRET = "sk-ant-api03-KKKKLLLLMMMMNNNNOOOOPPPPQQQQRRRRSSSSTTTT"

# The three new tools this bundle adds.
NEW_TOOLS = ("file_patch", "grep", "glob")


# --- Helpers (copied real-path pattern) ------------------------------------

def _cfg(workspace, **kw):
    defaults = dict(
        name="test-fs-integ",
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

    Parameterized on tool_name so A05/A06 can drive turns that name the NEW
    tools while keeping the exact cache-safety structure of the original.
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


# ===========================================================================
# file_patch through execute_tool updates the REAL per-room read registry
# ===========================================================================

class TestRealRegistryUpdate:
    @pytest.mark.asyncio
    async def test_file_patch_updates_real_read_registry(self, tmp_path):
        """file_patch success refreshes the REAL per-room registry from _build_agent_callbacks."""
        bot, agent = _make_bot_with_real_agent(tmp_path)
        cb = bot._build_agent_callbacks(ROOM, None)
        assert cb["read_registry"] is agent._read_registries[ROOM], \
            "sanity: callbacks registry must be the agent's real per-room dict"

        f = tmp_path / "reg.txt"
        f.write_text("before\n")
        patch = "<<<<<<< SEARCH\nbefore\n=======\nafter\n>>>>>>> REPLACE"
        res = await execute_tool(
            name="file_patch",
            input={"path": str(f), "patch": patch},
            tool_config={},
            agent_config=agent.config,
            tools=agent.tools,
            callbacks={"read_registry": cb["read_registry"], "call_id": "tc"},
        )
        assert not res.is_error, f"file_patch must succeed through execute_tool: {res.content}"
        resolved = str(Path(str(f)).resolve())
        assert resolved in cb["read_registry"], \
            "successful file_patch must refresh the REAL per-room read registry (V6)"


# ===========================================================================
# file_write guard fires BEFORE validation — two distinct error paths
# ===========================================================================

class TestGuardBeforeValidation:
    @pytest.mark.asyncio
    async def test_unread_refusal_distinct_from_syntax_rejection(self, tmp_path):
        """Guard (unread file) fires BEFORE validation; the two error paths differ (§10)."""
        bot, agent = _make_bot_with_real_agent(tmp_path)
        cb = bot._build_agent_callbacks(ROOM, None)

        f = tmp_path / "guarded.py"
        f.write_text("x = 1\n")  # exists, NOT read this session

        # (A) Unread existing file → GUARD refusal, regardless of syntax.
        res_guard = await execute_tool(
            name="file_write",
            input={"path": str(f), "content": "x = = 1\n"},  # broken syntax too
            tool_config={"require_read_before_write": True},
            agent_config=agent.config,
            tools=agent.tools,
            callbacks={"read_registry": cb["read_registry"], "call_id": "g1"},
        )
        assert res_guard.is_error, "unread existing file must be refused"
        assert "not read" in res_guard.content.lower(), \
            "guard refusal must mention the file was not read (guard, not validator)"
        assert "syntax" not in res_guard.content.lower(), \
            "guard must fire BEFORE validation — refusal must not mention syntax"

        # (B) Read it, then write broken syntax → VALIDATION rejection (distinct path).
        await execute_tool(
            name="file_read", input={"path": str(f)}, tool_config={},
            agent_config=agent.config, tools=agent.tools,
            callbacks={"read_registry": cb["read_registry"], "call_id": "r1"},
        )
        before = f.read_bytes()
        res_val = await execute_tool(
            name="file_write",
            input={"path": str(f), "content": "x = = 1\n"},
            tool_config={"require_read_before_write": True},
            agent_config=agent.config,
            tools=agent.tools,
            callbacks={"read_registry": cb["read_registry"], "call_id": "w1"},
        )
        assert res_val.is_error, "clean→broken write of a READ file must be validation-rejected"
        assert "not read" not in res_val.content.lower(), \
            "validation rejection must be a DISTINCT path from the guard refusal"
        assert f.read_bytes() == before, "V1: validation-rejected write leaves file byte-identical"


# ===========================================================================
# Validation rejection arrives as a normal wrapped <tool_result> error
# ===========================================================================

class TestValidationThroughPipeline:
    @pytest.mark.asyncio
    async def test_validation_rejection_wraps_as_tool_result_error(self, tmp_path):
        """A validation rejection flows the normal truncate→wrap pipeline as an error (§10/V6)."""
        bot, agent = _make_bot_with_real_agent(tmp_path)
        cb = bot._build_agent_callbacks(ROOM, None)

        f = tmp_path / "v.py"
        f.write_text("x = 1\n")
        await execute_tool(
            name="file_read", input={"path": str(f)}, tool_config={},
            agent_config=agent.config, tools=agent.tools,
            callbacks={"read_registry": cb["read_registry"], "call_id": "r"},
        )
        result = await execute_tool(
            name="file_write",
            input={"path": str(f), "content": "def broken(:\n"},
            tool_config={"require_read_before_write": True},
            agent_config=agent.config,
            tools=agent.tools,
            callbacks={"read_registry": cb["read_registry"], "call_id": "w"},
        )
        assert result.is_error, "validation must reject broken syntax"

        # Emulate the exact call-site pipeline (agent.py:988-989): truncate then wrap.
        from openalph.tools import truncate_result, wrap_tool_result
        wrapped = wrap_tool_result(
            truncate_result(result.content, agent.config.truncation_limit),
            "file_write", "w")
        assert wrapped.startswith('<tool_result tool="file_write" id="w">'), \
            "rejection must wrap in a standard <tool_result> envelope"
        assert "</tool_result>" in wrapped


# ===========================================================================
# grep output containing a planted credential is redacted by the pipeline
# ===========================================================================

class TestGrepRedaction:
    @pytest.mark.asyncio
    async def test_grep_output_credential_redacted(self, tmp_path):
        """grep results carrying a planted secret are redacted inside execute_tool (V6)."""
        bot, agent = _make_bot_with_real_agent(tmp_path)
        cb = bot._build_agent_callbacks(ROOM, None)

        secret_file = tmp_path / "leak.txt"
        secret_file.write_text(f"config token = {FAKE_SECRET}\n")

        res = await execute_tool(
            name="grep",
            input={"pattern": "token", "output_mode": "content"},
            tool_config={},
            agent_config=agent.config,
            tools=agent.tools,
            callbacks={"read_registry": cb["read_registry"], "call_id": "grep1"},
        )
        assert not res.is_error, f"grep must succeed and find the line: {res.content}"
        assert FAKE_SECRET not in res.content, \
            "grep output must be redacted by the execute_tool pipeline (V6)"
        assert "[REDACTED" in res.content, \
            "redacted grep output must carry a [REDACTED:*] marker"


# ===========================================================================
# TOML discovery — present → in self.tools + tool_schemas; absent → absent
# ===========================================================================

class TestToolDiscovery:
    def _agent_with_tools(self, tmp_path, tool_names):
        """Build a real Agent whose workspace enables tool_names; ToolError → fail (not raise)."""
        ws = _setup_workspace(tmp_path, tools=tool_names)
        try:
            return Agent(_cfg(ws))
        except ToolError as e:
            pytest.fail(
                f"discover_tools rejected {tool_names!r} — the new tools are not yet "
                f"registered in BUILTIN_TOOLS (expected RED): {e}")

    def test_new_tools_present_when_toml_present(self, tmp_path):
        """file_patch/grep/glob TOMLs present → tools appear in self.tools (design §12)."""
        agent = self._agent_with_tools(
            tmp_path, ("shell", "file_read", "file_patch", "grep", "glob"))
        names = {t.name for t in agent.tools}
        for n in NEW_TOOLS:
            assert n in names, f"{n}.toml present must yield {n} in self.tools"

    def test_tool_schemas_emit_exact_shape(self, tmp_path):
        """tool_schemas emits {name, description, input_schema} for new tools (§12)."""
        agent = self._agent_with_tools(
            tmp_path, ("shell", "file_patch", "grep", "glob"))
        schemas = {s["name"]: s for s in tool_schemas(agent.tools)}
        for n in NEW_TOOLS:
            assert n in schemas, f"{n} must be emitted by tool_schemas"
            s = schemas[n]
            assert set(s.keys()) == {"name", "description", "input_schema"}, \
                f"{n} schema must be exactly {{name, description, input_schema}}"
            assert isinstance(s["description"], str) and s["description"], \
                f"{n} must have a non-empty description"
            assert isinstance(s["input_schema"], dict) and "properties" in s["input_schema"], \
                f"{n} input_schema must be a JSON-schema dict"

    def test_new_tools_absent_when_toml_absent(self, tmp_path):
        """No new-tool TOMLs → new tools absent from self.tools (design §12)."""
        # Standard workspace (no new-tool TOMLs) always constructs cleanly.
        agent = Agent(_cfg(_setup_workspace(tmp_path)))
        names = {t.name for t in agent.tools}
        for n in NEW_TOOLS:
            assert n not in names, f"{n} must be absent when {n}.toml is absent"


# ===========================================================================
# Sub-agent inherits new tools (minus subagent)
# ===========================================================================

class TestSubAgentInheritance:
    @pytest.mark.asyncio
    async def test_subagent_inherits_new_tools_minus_subagent(self, tmp_path):
        """Sub-agent inherits parent's new tools; only 'subagent' is filtered (anchors §2)."""
        ws = _setup_workspace(
            tmp_path,
            tools=("shell", "subagent", "file_patch", "grep", "glob"))
        try:
            parent = Agent(_cfg(ws))
        except ToolError as e:
            pytest.fail(
                f"discover_tools rejected new tools — not yet registered (expected RED): {e}")

        from openalph.tools import subagent as subagent_mod
        captured = {}

        async def _fake_complete(*, config=None, system=None, messages=None,
                                 tools=None, max_tokens=None, **kw):
            captured["tools"] = tools
            return Response(content="done", tool_calls=[], model="test",
                            usage=Usage(input_tokens=1, output_tokens=1),
                            stop_reason="end_turn")

        with patch.object(subagent_mod, "complete", _fake_complete):
            await subagent_mod.run_subagent(
                task="inherit check", config=parent.config,
                tools=parent.tools, max_iterations=2)

        inherited = {t.name for t in (captured.get("tools") or [])}
        for n in NEW_TOOLS:
            assert n in inherited, f"sub-agent must inherit {n}"
        assert "subagent" not in inherited, "sub-agent must NOT inherit the subagent tool"


# ===========================================================================
# A05 / A06 guardrails (copies) exercised on turns that name the NEW tools (V5)
# ===========================================================================

class TestCacheGuardrailsWithNewTools:
    @pytest.mark.asyncio
    async def test_A05_cache_safety_strict_prefix_new_tool_turn(self, tmp_path):
        """A05 (copy): call N messages are a strict prefix of call N+1 — on a grep turn.

        Faithful copy of test_guidance_injection.py::test_A05_cache_safety_strict_prefix
        (:302), here driving a turn whose stream issues 'grep' tool calls.  V5: the
        static tool surface must not disturb the cache-safety prefix property.
        """
        ws = _setup_workspace(tmp_path)
        config = _cfg(ws, max_iterations=10)
        agent = Agent(config)
        stream_fn, payloads = _make_capturing_stream(
            tool_iterations=7, tool_name="grep",
            tool_input={"pattern": "needle"})
        mock_exec = AsyncMock(return_value=ToolResult(content="ok", is_error=False))
        with patch("openalph.agent.stream", side_effect=stream_fn), \
             patch("openalph.agent.execute_tool", mock_exec):
            await agent.handle_input("search please", room_id=ROOM)

        # Phase 1: strict prefix property.
        for i in range(len(payloads) - 1):
            cur = payloads[i]
            nxt = payloads[i + 1]
            assert len(nxt) >= len(cur), (
                f"Call {i+1} has fewer messages ({len(nxt)}) than call {i} ({len(cur)})")
            for j in range(len(cur)):
                assert cur[j] is nxt[j], (
                    f"Message {j} is a DIFFERENT object in call {i+1} vs {i} — cache bust risk.")

        # Phase 2: a reminder MUST have been injected for the test to be meaningful.
        history = agent.history(ROOM)
        reminders = [m for m in history
                     if REMINDER_TAG_OPEN in str(m.get("content", ""))]
        assert reminders, (
            "Cache-safety test requires a mid-loop reminder injection; "
            "no <system-reminder> found — engine integration missing.")

    def test_A06_replay_identity_new_tool_transcript(self, tmp_path):
        """A06 (copy): build_context reproduces reminder entries verbatim in a file_patch transcript.

        Faithful copy of test_guidance_injection.py::test_A06_replay_identity (:335),
        with the assistant turn issuing a file_patch tool call.  V5: new tools do not
        alter the replay-identity property of reminder entries.
        """
        sl = SessionLog(workspace=tmp_path, agent_user_id=AGENT_USER)
        framed = (f"{REMINDER_TAG_OPEN}\n"
                  "You have not consulted memory this session.\n"
                  "</system-reminder>")
        sl.append(role="user", sender="@op:x", room=ROOM, event_id="$e1",
                  content="patch the file")
        sl.append(role="assistant", sender=AGENT_USER, room=ROOM, event_id=None,
                  content="", tool_calls=[
                      {"name": "file_patch", "id": "tc_1",
                       "input": {"path": "a.py", "patch": "<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE"}}])
        sl.append(role="tool", sender=AGENT_USER, room=ROOM, event_id=None,
                  call_id="tc_1", name="file_patch", output="Applied 1 hunk(s) to a.py")
        sl.append(role="user", sender=AGENT_USER, room=ROOM, event_id=None,
                  content=framed, source="reminder", trigger="memory-salience")
        sl.append(role="assistant", sender=AGENT_USER, room=ROOM, event_id=None,
                  content="OK, searching memory")

        context = sl.build_context(ROOM)
        reminder_msgs = [m for m in context
                         if REMINDER_TAG_OPEN in str(m.get("content", ""))]
        assert len(reminder_msgs) == 1, "reminder must appear once in rebuilt context"
        assert reminder_msgs[0]["content"] == framed, \
            "build_context must replay reminder content VERBATIM (A06), new tools notwithstanding"
        assert reminder_msgs[0]["role"] == "user"
