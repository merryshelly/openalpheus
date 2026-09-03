"""kdsn.319 — Room-visible shell tool dispatch notices.

Contract (memory/projects/openalph/specs/kdsn.319-shell-dispatch-notice-spec.md):
  1. _tool_intent gains a shell branch: dispatch-time m.notice (redacted
     command preview + collapsed full command) + start-time recording.
  2. _subagent_start_times renamed _tool_start_times (shared subagent+shell).
  3. Shell completion notice gains elapsed time (subagent's format).
  4. Generic completion detail (input_data fields, e.g. shell command) is
     redacted before display — closing the raw-command leak in room notices.
  5. Non-shell tools: NO dispatch-time notice. No flags, no rate-limiting.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from openalph.matrix import MatrixBot
from openalph.config import AgentConfig, MatrixConfig, ProviderConfig
from openalph.provider import ToolCall
from openalph.session import SessionLog


ROOM = "!shellnotice:matrix.local"
AGENT_USER = "@bot:matrix.local"

SECRET = "sk-live-abcdef1234567890abcdef1234567890"


def make_provider_config():
    return ProviderConfig(key="anthropic", type="anthropic", api_key="sk-test",
                          base_url=None, quirks=[])


def make_agent_config(workspace: Path, **kwargs):
    defaults = dict(
        name="test-agent",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": make_provider_config()},
        workspace=workspace,
        max_iterations=25,
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=None,
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def make_matrix_config():
    return MatrixConfig(
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


def make_bot(tmp_path):
    """MatrixBot with MagicMock agent + captured room_send (mirrors
    test_per_room_usage.make_bot)."""
    config = make_matrix_config()
    agent = MagicMock()
    agent.handle_input = AsyncMock(return_value="Response")
    agent.history = MagicMock(return_value=[])
    agent.last_turn_usage = MagicMock(return_value=None)
    agent.config = make_agent_config(tmp_path)

    bot = MatrixBot.__new__(MatrixBot)
    bot.config = config
    bot.agent = agent
    bot.client = MagicMock()
    bot.client.room_send = AsyncMock()
    bot.client.room_typing = AsyncMock()
    bot.client.room_messages = AsyncMock(return_value=MagicMock(chunk=[], end=None))
    bot._set_typing = AsyncMock()
    bot.send = AsyncMock()
    bot.send_notice = AsyncMock()
    bot._current_room = None
    bot._synced = True
    bot._active_rooms = set()
    bot._halted_rooms = set()
    bot._room_effort = {}
    bot._background_tasks = set()
    bot._session_locks = {}
    bot.session_log = SessionLog(tmp_path, AGENT_USER)
    return bot, agent


def notices(bot):
    """All m.notice sends captured from the mocked client, as (body, formatted).
    _room_send_with_retry calls client.room_send(room_id, "m.room.message", content)
    — content is the THIRD positional arg."""
    out = []
    for c in bot.client.room_send.await_args_list:
        msg = c.args[2] if len(c.args) >= 3 else c.kwargs.get("content")
        if isinstance(msg, dict) and msg.get("msgtype") == "m.notice":
            out.append((msg.get("body", ""), msg.get("formatted_body", "")))
    return out


# ---------------------------------------------------------------------------
# Dispatch-time notices (_tool_intent)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_t1_shell_intent_emits_one_notice(tmp_path):
    bot, _ = make_bot(tmp_path)
    _, tool_intent = bot._make_tool_callbacks(ROOM)
    tc = ToolCall(id="c1", name="shell", input={"command": "systemctl status nginx"})
    await tool_intent([tc], "content")
    ns = notices(bot)
    assert len(ns) == 1, f"expected exactly 1 dispatch notice, got {len(ns)}"
    body, formatted = ns[0]
    assert "shell ▶" in body
    assert "systemctl status nginx" in body


@pytest.mark.asyncio
async def test_t2_dispatch_preview_redacted(tmp_path):
    bot, _ = make_bot(tmp_path)
    _, tool_intent = bot._make_tool_callbacks(ROOM)
    cmd = f"curl -H 'Authorization: Bearer {SECRET}' https://api.example.com/x"
    tc = ToolCall(id="c2", name="shell", input={"command": cmd})
    await tool_intent([tc], "content")
    ns = notices(bot)
    assert len(ns) == 1
    body, formatted = ns[0]
    assert SECRET not in body, "raw secret leaked into dispatch body"
    assert SECRET not in formatted, "raw secret leaked into dispatch HTML"
    assert "[REDACTED:" in body or "[REDACTED:" in formatted


@pytest.mark.asyncio
async def test_t3_full_command_collapsed_and_escaped(tmp_path):
    bot, _ = make_bot(tmp_path)
    _, tool_intent = bot._make_tool_callbacks(ROOM)
    cmd = "echo '<script>alert(1)</script>' && echo done"
    tc = ToolCall(id="c3", name="shell", input={"command": cmd})
    await tool_intent([tc], "content")
    ns = notices(bot)
    assert len(ns) == 1
    _, formatted = ns[0]
    assert "<details>" in formatted and "full command" in formatted
    assert "<script>" not in formatted, "raw HTML tag leaked into notice"
    assert "&lt;script&gt;" in formatted


@pytest.mark.asyncio
async def test_t4_non_shell_intent_emits_nothing(tmp_path):
    bot, _ = make_bot(tmp_path)
    _, tool_intent = bot._make_tool_callbacks(ROOM)
    tc = ToolCall(id="c4", name="file_read", input={"path": "memory/foo.md"})
    await tool_intent([tc], "content")
    assert notices(bot) == []


@pytest.mark.asyncio
async def test_t5_subagent_branch_unaffected_by_rename(tmp_path):
    bot, _ = make_bot(tmp_path)
    tool_notice, tool_intent = bot._make_tool_callbacks(ROOM)
    tc = ToolCall(id="c5", name="subagent",
                  input={"task": "do a thing", "model": "qwen38blackwell"})
    await tool_intent([tc], "content")
    ns = notices(bot)
    assert len(ns) == 1
    body, _ = ns[0]
    assert "sub-agent" in body
    # start time recorded under the renamed dict: a subagent completion notice
    # carries elapsed (proves _tool_start_times is wired across both closures)
    import asyncio
    await asyncio.sleep(0.01)
    await tool_notice("c5", "subagent", tc.input, "sub result", is_error=False)
    import re
    completion = [n for n in notices(bot) if "sub-agent" not in n[0]]
    assert len(completion) == 1
    assert re.search(r"— \d", completion[0][0]), completion[0][0]


@pytest.mark.asyncio
async def test_t10_parallel_shell_calls_two_notices(tmp_path):
    bot, _ = make_bot(tmp_path)
    _, tool_intent = bot._make_tool_callbacks(ROOM)
    tcs = [ToolCall(id="p1", name="shell", input={"command": "echo one"}),
           ToolCall(id="p2", name="shell", input={"command": "echo two"})]
    await tool_intent(tcs, "content")
    assert len(notices(bot)) == 2


@pytest.mark.asyncio
async def test_t11_shell_intent_missing_command_no_crash(tmp_path):
    bot, _ = make_bot(tmp_path)
    _, tool_intent = bot._make_tool_callbacks(ROOM)
    tc = ToolCall(id="c11", name="shell", input={})
    await tool_intent([tc], "content")  # must not raise
    ns = notices(bot)
    assert len(ns) == 1
    body, _ = ns[0]
    assert "shell ▶" in body


@pytest.mark.asyncio
async def test_t12_long_command_truncated_120(tmp_path):
    bot, _ = make_bot(tmp_path)
    _, tool_intent = bot._make_tool_callbacks(ROOM)
    cmd = "echo " + "x" * 300
    tc = ToolCall(id="c12", name="shell", input={"command": cmd})
    await tool_intent([tc], "content")
    body, _ = notices(bot)[0]
    # head-truncation contract: first 120 chars of the flattened command
    # ("echo " + 115 x's) — identifies the call; never beyond 120
    assert ("x" * 300) not in body
    assert ("echo " + "x" * 115) in body
    assert ("x" * 116) not in body


@pytest.mark.asyncio
async def test_t13_room_send_failure_swallowed(tmp_path):
    bot, _ = make_bot(tmp_path)
    with __import__("unittest").mock.patch.object(
            bot, "_room_send_with_retry", AsyncMock(side_effect=RuntimeError("matrix down"))):
        _, tool_intent = bot._make_tool_callbacks(ROOM)
        tc = ToolCall(id="c13", name="shell", input={"command": "echo hi"})
        await tool_intent([tc], "content")  # must not raise


# ---------------------------------------------------------------------------
# Completion notices (_tool_notice) — elapsed + redacted detail
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_t6_shell_completion_has_elapsed(tmp_path):
    import asyncio
    bot, _ = make_bot(tmp_path)
    tool_notice, tool_intent = bot._make_tool_callbacks(ROOM)
    tc = ToolCall(id="c6", name="shell", input={"command": "sleep 0.01 && echo ok"})
    await tool_intent([tc], "content")
    await asyncio.sleep(0.02)
    import re
    await tool_notice("c6", "shell", tc.input, "ok", is_error=False)
    ns = notices(bot)
    completion = [n for n in ns if "shell ▶" not in n[0]]
    assert len(completion) == 1, f"expected 1 completion notice, got {len(completion)}"
    body, _ = completion[0]
    assert re.search(r"— \d", body), f"no elapsed suffix in completion notice: {body!r}"


@pytest.mark.asyncio
async def test_t7_completion_without_start_time_no_crash_no_elapsed(tmp_path):
    import re
    bot, _ = make_bot(tmp_path)
    tool_notice, _ = bot._make_tool_callbacks(ROOM)
    await tool_notice("never-dispatched", "shell", {"command": "echo hi"}, "ok", is_error=False)
    ns = notices(bot)
    assert len(ns) == 1
    body, _ = ns[0]
    assert "✅" in body
    assert not re.search(r"— \d", body), f"unexpected elapsed suffix: {body!r}"


@pytest.mark.asyncio
async def test_t8_completion_detail_redacted_all_tools(tmp_path):
    bot, _ = make_bot(tmp_path)
    tool_notice, _ = bot._make_tool_callbacks(ROOM)
    cmd = f"deploy.sh --token {SECRET}"
    await tool_notice("c8", "shell", {"command": cmd}, "deployed", is_error=False)
    body, formatted = notices(bot)[0]
    assert SECRET not in body, "raw secret leaked into completion body"
    assert SECRET not in formatted, "raw secret leaked into completion HTML"
    assert "[REDACTED:" in body or "[REDACTED:" in formatted


# ---------------------------------------------------------------------------
# kdsn.319 audit pins (H1 layering + M2 canonical-pass discrimination)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_t14_shape_pass_pinned_not_just_egress(tmp_path):
    """--key is NOT in the egress regex list, so only the canonical shape
    pass can redact an sk-ant-shaped key here. If redact_credentials is
    dropped from this seam, this test goes red (T2/T8 alone wouldn't —
    their Bearer/--token commands are caught by the egress layer)."""
    bot, _ = make_bot(tmp_path)
    _, tool_intent = bot._make_tool_callbacks(ROOM)
    secret = "sk-ant-api03-" + "ab12cd34" * 5
    cmd = f"validate-key.py --key {secret}"
    tc = ToolCall(id="c14", name="shell", input={"command": cmd})
    await tool_intent([tc], "content")
    body, formatted = notices(bot)[0]
    assert secret not in body and secret not in formatted
    assert "[REDACTED:" in body or "[REDACTED:" in formatted


@pytest.mark.asyncio
async def test_t15_env_assignment_value_masked_name_kept(tmp_path):
    bot, _ = make_bot(tmp_path)
    _, tool_intent = bot._make_tool_callbacks(ROOM)
    tc = ToolCall(id="c15", name="shell",
                  input={"command": "export DB_PASSWORD=hunter2pass123 && ./migrate"})
    await tool_intent([tc], "content")
    body, _ = notices(bot)[0]
    assert "hunter2pass123" not in body
    assert "DB_PASSWORD=" in body
    assert "[REDACTED:" in body


@pytest.mark.asyncio
async def test_t17_env_secret_word_at_front_of_name(tmp_path):
    """Audit regression pin: the secret word at position 0 of the var name
    (PASSWORD=, SECRET_VALUE=) must redact — a naive [A-Za-z_] first-char
    consumption makes the front-anchored alternation unreachable."""
    bot, _ = make_bot(tmp_path)
    _, tool_intent = bot._make_tool_callbacks(ROOM)
    tcs = [ToolCall(id="f1", name="shell", input={"command": "PASSWORD=letmein1234 ./login"}),
           ToolCall(id="f2", name="shell", input={"command": "SECRET_VALUE=s3cr3t-hunter2 db-push"}),
           ToolCall(id="f3", name="shell", input={"command": "HOME=/home/oa-merry ls"})]
    await tool_intent(tcs, "content")
    bodies = [n[0] for n in notices(bot)]
    assert "letmein1234" not in bodies[0] and "[REDACTED:" in bodies[0]
    assert "s3cr3t-hunter2" not in bodies[1] and "SECRET_VALUE=" in bodies[1]
    assert bodies[2] == "🔧 shell ▶ — HOME=/home/oa-merry ls"  # benign env survives


@pytest.mark.asyncio
async def test_t16_url_userinfo_masked(tmp_path):
    bot, _ = make_bot(tmp_path)
    _, tool_intent = bot._make_tool_callbacks(ROOM)
    tc = ToolCall(id="c16", name="shell",
                  input={"command": "curl https://admin:hunter2pass@api.example.com/v1"})
    await tool_intent([tc], "content")
    body, _ = notices(bot)[0]
    assert "hunter2pass" not in body
    assert "api.example.com" in body  # host survives; only userinfo masked


@pytest.mark.asyncio
async def test_t9_error_completion_status_and_elapsed(tmp_path):
    import asyncio
    import re
    bot, _ = make_bot(tmp_path)
    tool_notice, tool_intent = bot._make_tool_callbacks(ROOM)
    tc = ToolCall(id="c9", name="shell", input={"command": "false"})
    await tool_intent([tc], "content")
    await asyncio.sleep(0.01)
    await tool_notice("c9", "shell", tc.input, "EXIT=1", is_error=True)
    ns = notices(bot)
    completion = [n for n in ns if "shell ▶" not in n[0]]
    assert len(completion) == 1
    body, _ = completion[0]
    assert "❌" in body
    assert re.search(r"— \d", body), f"no elapsed suffix on error completion: {body!r}"
