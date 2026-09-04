"""RED suite — Component G (tool-error detail notices, matrix.py).

Bundle-2 (workspace-kdsn.195), design §8b + §10-bullet-4.

Real-path: a REAL MatrixBot (via the canonical _make_bot_with_real_agent
pattern) with its REAL _make_tool_callbacks → _tool_notice closure.  Only
the nio client and provider are mocked.  All notice paths funnel through
``bot.client.room_send(room_id, "m.room.message", content)`` — the content
dict is the capture point.

Today (pre-Component-G) an errored non-subagent/non-todo tool emits a PLAIN
``{"msgtype":"m.notice","body":"🔧 {name}{detail} ❌ error"}`` with NO
``formatted_body`` and NO ``<details>`` block — so the detail-bearing
assertions fail RED cleanly.  Non-error and todo_write notices are pinned
BYTE-IDENTICAL to today (GREEN guardrails that Component G must not disturb).

Helpers copied from test_guidance_integration.py (no cross-test-module
imports exist in this suite → copy per convention, never modify an existing
test file).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openalph.agent import Agent
from openalph.config import AgentConfig, ProviderConfig, MatrixConfig
from openalph.provider import Response, Usage, ToolCall
from openalph.matrix import MatrixBot
from openalph.tools import ToolResult, wrap_tool_result
from openalph.tools.security import redact_credentials as _redact


ROOM = "!notices-room:matrix.local"
AGENT_USER = "@agent:matrix.local"

# Fake credential in the sk-ant-<alnum> pattern (redactor: sk-ant-[a-zA-Z0-9_-]{8,}).
FAKE_SECRET = "sk-ant-api03-AAAABBBBCCCCDDDDEEEEFFFFGGGGHHHHIIIIJJJJ"


# --- Helpers (copied real-path pattern) ------------------------------------

def _cfg(workspace, **kw):
    defaults = dict(
        name="test-notices",
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
    bot.heartbeat = MagicMock()
    bot.heartbeat.is_active = MagicMock(return_value=False)
    bot.umbral = MagicMock()
    bot.umbral.is_active = MagicMock(return_value=False)
    bot._steering_inbox = {}
    bot._active_turns = set()
    return bot, agent


def _notice_contents(bot):
    """All notice content dicts sent through client.room_send this test."""
    out = []
    for call in bot.client.room_send.call_args_list:
        # _room_send_with_retry calls: room_send(room_id, "m.room.message", content)
        if len(call.args) >= 3:
            out.append(call.args[2])
    return out


# ===========================================================================
# Success notices: abbreviated body preserved, full call+result now furled
# (workspace-kdsn.247.1 — generalizes the collapsed-<details> disclosure to
# EVERY generic tool notice, not just errors)
# ===========================================================================

class TestNoticeInvariantsPinned:
    @pytest.mark.asyncio
    async def test_non_error_body_stays_abbreviated_line(self, tmp_path):
        """The default/unfurled body stays the short abbreviated line, unchanged
        from today's plain format — only formatted_body gains the fold (247.1)."""
        bot, agent = _make_bot_with_real_agent(tmp_path)
        _tool_notice, _tool_intent = bot._make_tool_callbacks(ROOM)

        await _tool_notice("tc1", "file_read", {"path": "/x/y.txt"},
                           "some result content", False)

        contents = _notice_contents(bot)
        assert len(contents) == 1, f"expected exactly one notice, got {len(contents)}"
        c = contents[0]
        assert c["body"] == "🔧 file_read `/x/y.txt` ✅", \
            f"abbreviated body must be unchanged from today's format; got {c['body']!r}"
        assert c["msgtype"] == "m.notice"

    @pytest.mark.asyncio
    async def test_non_error_notice_furls_full_call_and_result(self, tmp_path):
        """Non-error notice now carries a collapsed <details> with the FULL call
        (all params) and FULL result — the 247.1 disclosure, closed by default."""
        bot, agent = _make_bot_with_real_agent(tmp_path)
        _tool_notice, _ = bot._make_tool_callbacks(ROOM)
        await _tool_notice("tc1", "shell", {"command": "echo hi", "cwd": "/tmp"},
                           "ok output", False)
        c = _notice_contents(bot)[0]
        fb = c.get("formatted_body", "")
        assert c.get("format") == "org.matrix.custom.html"
        assert "<details>" in fb, "success notice must now furl a <details> block"
        assert "📥 Full call" in fb and "📤 Full result" in fb, \
            "success notice must furl both the full call and the full result"
        # Full input = ALL params, not just the single abbreviated field.
        assert "command" in fb and "cwd" in fb, \
            "furled call must contain every input param, not just the summary field"
        assert "ok output" in fb, "furled result must contain the result content"

    @pytest.mark.asyncio
    async def test_todo_write_notice_path_unaffected(self, tmp_path):
        """todo_write notice keeps its 📋 special path, unaffected by Component G (§8b)."""
        bot, agent = _make_bot_with_real_agent(tmp_path)
        _tool_notice, _ = bot._make_tool_callbacks(ROOM)
        await _tool_notice("tc1", "todo_write", {"todos": []},
                           "Todo list updated (1 item)", False)
        c = _notice_contents(bot)[0]
        assert c.get("body", "").startswith("📋"), \
            "todo_write notice must retain its 📋 body prefix"
        assert "<details>" in c.get("formatted_body", ""), \
            "todo_write notice must retain its collapsed <details> list"


# ===========================================================================
# Error notices gain a detail block (RED — not implemented yet)
# ===========================================================================

class TestErrorNoticeDetail:
    @pytest.mark.asyncio
    async def test_error_notice_has_details_block_with_tool_name(self, tmp_path):
        """Error notice: tool name + detail inside a collapsed <details> block (§8b)."""
        bot, agent = _make_bot_with_real_agent(tmp_path)
        _tool_notice, _ = bot._make_tool_callbacks(ROOM)

        err = "Error: File not found: /x/missing.txt"
        await _tool_notice("tc1", "file_read", {"path": "/x/missing.txt"}, err, True)

        c = _notice_contents(bot)[0]
        blob = c.get("formatted_body", "") + c.get("body", "")
        assert "file_read" in blob, "error notice must name the tool"
        assert "<details>" in c.get("formatted_body", ""), \
            "error notice must add a collapsed <details> block (Component G)"
        assert "missing.txt" in c.get("formatted_body", ""), \
            "error notice <details> must contain the error detail"

    @pytest.mark.asyncio
    async def test_error_notice_html_escapes_detail(self, tmp_path):
        """<, >, & in error detail are html-escaped inside <details> (§8b)."""
        bot, agent = _make_bot_with_real_agent(tmp_path)
        _tool_notice, _ = bot._make_tool_callbacks(ROOM)

        err = "Error: bad token <script> && </script> in <config>"
        await _tool_notice("tc1", "shell", {"command": "run"}, err, True)

        c = _notice_contents(bot)[0]
        fb = c.get("formatted_body", "")
        assert "&lt;script&gt;" in fb, "'<' and '>' in detail must be html-escaped"
        assert "&amp;&amp;" in fb, "'&' in detail must be html-escaped"
        assert "<script>" not in fb, "raw '<script>' must not appear unescaped in detail"

    @pytest.mark.asyncio
    async def test_error_notice_redacted_credential_not_raw(self, tmp_path):
        """Planted credential arrives post-redaction → notice shows [REDACTED], not raw (§8b/V6)."""
        bot, agent = _make_bot_with_real_agent(tmp_path)
        _tool_notice, _ = bot._make_tool_callbacks(ROOM)

        # Reproduce the real pipeline: redact THEN wrap (V6 order) — exactly what
        # agent.py hands to on_tool_call as wrapped_content.
        raw = f"api error, key was {FAKE_SECRET}"
        redacted, events = _redact(raw)
        assert events, "sanity: planted secret must match a redaction pattern"
        wrapped = wrap_tool_result(redacted, "shell", "tc1")

        await _tool_notice("tc1", "shell", {"command": "curl"}, wrapped, True)

        c = _notice_contents(bot)[0]
        blob = c.get("formatted_body", "") + c.get("body", "")
        assert "[REDACTED" in blob, \
            "error notice detail must surface the post-redaction [REDACTED:*] marker"
        assert FAKE_SECRET not in blob, \
            "raw secret must NEVER appear in the notice (post-redaction guarantee, V6)"

    @pytest.mark.asyncio
    async def test_error_detail_head_tail_truncated_not_hard_2000(self, tmp_path):
        """Over-long detail is head+tail truncated (truncate_result marker), NOT
        hard-cut at 2000 — the 247.1 point is seeing MORE than the abbreviated
        line, while a per-notice cap keeps it under the Matrix PDU limit."""
        bot, agent = _make_bot_with_real_agent(tmp_path)
        _tool_notice, _ = bot._make_tool_callbacks(ROOM)

        big = "Z" * 40000
        await _tool_notice("tc1", "shell", {"command": "run"}, big, True)

        c = _notice_contents(bot)[0]
        fb = c.get("formatted_body", "")
        # kdsn.247.2: the marker is head+tail elision via truncate_result, but
        # operator-addressed (this render path's reader cannot re-run anything).
        assert "[truncated:" not in fb, \
            "agent-facing re-run marker must not appear in operator notices"
        assert "chars elided from this notice" in fb, \
            "over-long detail must carry the operator-facing elision marker"
        assert "full content in session JSONL" in fb
        # Head+tail, not hard-cut: material from BOTH ends survives, and the
        # middle elision means fewer than the full 40000 chars pass through.
        assert fb.count("Z") < 40000, \
            "elided notice must not carry the entire 40000-char detail"
        # Shows MORE than the old 2000 hard cap...
        assert fb.count("Z") > 2000, \
            "furled detail must surface more than the old 2000-char hard cut"
        # ...but the whole notice stays well under the ~25K Matrix PDU limit.
        assert len(fb) < 25000, \
            f"notice must stay under the Matrix PDU limit; got {len(fb)} chars"


# ===========================================================================
# Field-picker + furled-detail (workspace-kdsn.247.1)
# ===========================================================================

class TestAbbreviatedLineFieldPicker:
    @pytest.mark.asyncio
    async def test_grep_abbreviated_line_shows_pattern(self, tmp_path):
        """grep's REQUIRED param is 'pattern' (path is optional/omitted); the
        abbreviated line must show it instead of a context-free '🔧 grep ✅'."""
        bot, agent = _make_bot_with_real_agent(tmp_path)
        _tool_notice, _ = bot._make_tool_callbacks(ROOM)
        await _tool_notice("tc1", "grep", {"pattern": "TODO|FIXME"},
                           "3 matches", False)
        c = _notice_contents(bot)[0]
        assert c["body"] == "🔧 grep `TODO|FIXME` ✅", \
            f"grep abbreviated line must surface the pattern; got {c['body']!r}"

    @pytest.mark.asyncio
    async def test_glob_abbreviated_line_shows_pattern(self, tmp_path):
        """glob likewise keys on 'pattern'."""
        bot, agent = _make_bot_with_real_agent(tmp_path)
        _tool_notice, _ = bot._make_tool_callbacks(ROOM)
        await _tool_notice("tc1", "glob", {"pattern": "**/*.py"}, "12 files", False)
        c = _notice_contents(bot)[0]
        assert c["body"] == "🔧 glob `**/*.py` ✅", \
            f"glob abbreviated line must surface the pattern; got {c['body']!r}"

    @pytest.mark.asyncio
    async def test_grep_pattern_wins_over_optional_path(self, tmp_path):
        """When grep supplies BOTH pattern and the optional path, the meaningful
        field (pattern) wins, and the optional glob filter is appended."""
        bot, agent = _make_bot_with_real_agent(tmp_path)
        _tool_notice, _ = bot._make_tool_callbacks(ROOM)
        await _tool_notice("tc1", "grep",
                           {"pattern": "needle", "path": "/src", "glob": "*.py"},
                           "1 match", False)
        c = _notice_contents(bot)[0]
        assert c["body"] == "🔧 grep `needle` (glob: `*.py`) ✅", \
            f"pattern must win over path, with glob filter appended; got {c['body']!r}"


class TestFurledResultRedaction:
    @pytest.mark.asyncio
    async def test_success_furled_result_is_post_redaction(self, tmp_path):
        """The furled FULL result on SUCCESS must surface only post-redaction
        content — the whole reason the old success path hid the result. A
        planted secret arrives already redacted (execute_tool → on_tool_call)."""
        bot, agent = _make_bot_with_real_agent(tmp_path)
        _tool_notice, _ = bot._make_tool_callbacks(ROOM)

        raw = f"config loaded, token={FAKE_SECRET}"
        redacted, events = _redact(raw)
        assert events, "sanity: planted secret must match a redaction pattern"
        wrapped = wrap_tool_result(redacted, "shell", "tc1")

        await _tool_notice("tc1", "shell", {"command": "cat cfg"}, wrapped, False)

        c = _notice_contents(bot)[0]
        fb = c.get("formatted_body", "")
        assert "[REDACTED" in fb, "furled result must show the [REDACTED:*] marker"
        assert FAKE_SECRET not in fb, \
            "raw secret must NEVER appear in the furled result (post-redaction)"

    @pytest.mark.asyncio
    async def test_furled_result_strips_tool_result_envelope(self, tmp_path):
        """The furled result unwraps the <tool_result tool/id> provenance
        envelope — its call id is noise in an operator notice."""
        bot, agent = _make_bot_with_real_agent(tmp_path)
        _tool_notice, _ = bot._make_tool_callbacks(ROOM)
        wrapped = wrap_tool_result("clean output", "shell", "toolu_ENVELOPE_ID")
        await _tool_notice("tc1", "shell", {"command": "echo"}, wrapped, False)
        c = _notice_contents(bot)[0]
        fb = c.get("formatted_body", "")
        assert "clean output" in fb
        assert "toolu_ENVELOPE_ID" not in fb, "envelope call id must be stripped"


# ===========================================================================
# Sub-agent scope: internal tool errors produce NO notices (GREEN guardrail)
# ===========================================================================

class TestSubAgentNoticeScope:
    @pytest.mark.asyncio
    async def test_subagent_internal_tool_calls_have_no_on_tool_call(self, tmp_path):
        """run_subagent dispatches tools WITHOUT an on_tool_call callback (§8b scope)."""
        from openalph.tools import subagent as subagent_mod

        cfg = _cfg(_setup_workspace(tmp_path))

        captured = {}

        async def _capturing_execute(*, name, input, tool_config,
                                     agent_config, tools=None, callbacks=None):
            captured["callbacks"] = callbacks
            return ToolResult(content="Error: File not found", is_error=True)

        # complete(): first turn issues an (internally-erroring) tool call, then text.
        calls = {"n": 0}

        async def _fake_complete(*, config=None, system=None, messages=None,
                                 tools=None, max_tokens=None, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                tc = ToolCall(id="s1", name="file_read",
                              input={"path": "/nope.txt"})
                return Response(content="", tool_calls=[tc],
                                model="test", usage=Usage(input_tokens=1, output_tokens=1),
                                stop_reason="tool_use")
            return Response(content="done", tool_calls=[],
                            model="test", usage=Usage(input_tokens=1, output_tokens=1),
                            stop_reason="end_turn")

        with patch.object(subagent_mod, "complete", _fake_complete), \
             patch("openalph.tools.execute_tool", _capturing_execute):
            result = await subagent_mod.run_subagent(
                task="do a thing",
                config=cfg,
                tools=None,
                max_iterations=3,
            )

        assert not result.is_error, "sub-agent must complete normally"
        assert "callbacks" in captured, "sub-agent must have dispatched a tool"
        cb = captured["callbacks"] or {}
        assert "on_tool_call" not in cb, \
            "sub-agent internal tool calls must have NO on_tool_call → no notices (§8b scope)"


class TestFurledTruncationMarkerAudience:
    """kdsn.247.2 — an elided furl block must address its actual reader: the
    operator, who cannot 're-run' anything. The agent-facing steering marker
    ('[truncated: N chars removed — re-run with a smaller limit …]') is the
    right instruction inside a <tool_result> but wrong in a Matrix notice
    render, where it reads like the operator should go retry something.
    Surfaced when a 13.1K template read confused the operator into thinking
    the file itself was truncated; the session JSONL was complete."""

    @pytest.mark.asyncio
    async def test_elided_furled_result_uses_operator_facing_marker(self, tmp_path):
        bot, agent = _make_bot_with_real_agent(tmp_path)
        _tool_notice, _ = bot._make_tool_callbacks(ROOM)

        big = "x" * 13000  # > _FURL_RESULT_RAW_CAP (12000) → head+tail elision
        await _tool_notice("tc1", "file_read", {"path": "/x/big.md"}, big, False)

        fb = _notice_contents(bot)[0].get("formatted_body", "")
        assert "chars elided from this notice" in fb, (
            "elided furl block must speak to the operator, not the agent")
        assert "full content in session JSONL" in fb
        assert "re-run with a smaller" not in fb, (
            "agent-facing re-run instruction must not appear in "
            "operator-facing Matrix notices")

    @pytest.mark.asyncio
    async def test_under_cap_result_notice_has_no_marker(self, tmp_path):
        """Results under the raw cap render whole — no elision marker at all."""
        bot, agent = _make_bot_with_real_agent(tmp_path)
        _tool_notice, _ = bot._make_tool_callbacks(ROOM)

        await _tool_notice("tc1", "file_read", {"path": "/x/small.md"},
                           "tiny", False)

        fb = _notice_contents(bot)[0].get("formatted_body", "")
        assert "elided from this notice" not in fb
        assert "re-run with a smaller" not in fb
