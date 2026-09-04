"""Consistency gate test (req-4) — Workstream D.

Asserts that all four context-accounting surfaces derive IDENTICAL
context_tokens AND context_max values from the same JSONL session,
including after a /cache toolstrip and a /model override.

Surfaces under test:
  1. /status path  — agent.status(room, history=sl.build_context(room))
  2. context_status tool — bot._build_context_status(room)
  3. 80%-alert path — agent.status(room, history=sl.build_context(room))  (same as /status)
  4. overflow-guard view — agent._estimate_context_tokens + _resolve_model_limit
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from openalph.agent import Agent
from openalph.session import SessionLog
from openalph.matrix import MatrixBot
from openalph.config import AgentConfig, MatrixConfig, ProviderConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOM = "!consistency:matrix.local"
AGENT_USER = "@bot:matrix.local"
USER = "@sb:matrix.local"
OPUS_MODEL = "anthropic/claude-opus-4-8"
OPUS_WINDOW = 1_048_576


def make_provider_config():
    return ProviderConfig(
        key="anthropic",
        type="anthropic",
        api_key="sk-test",
        base_url=None,
        quirks=[],
    )


def make_agent_config(workspace: Path):
    return AgentConfig(
        name="test-agent",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": make_provider_config()},
        workspace=workspace,
        max_iterations=25,
        truncation_limit=50000,
        model_max_tokens=200_000,
        matrix=None,
    )


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


def _append_toolstrip(sl, entry_index):
    """Write a /cache toolstrip marker entry (mirrors test_toolstrip.py convention)."""
    sl.append(
        role="system",
        sender=USER,
        room=ROOM,
        event="toolstrip",
        entry_index=entry_index,
    )


def _build_session(tmp_path, sl):
    """Build a realistic session in JSONL: user + several tool-use turns + text turn.

    Returns the list of JSONL entry positions for the toolstrip boundary
    (we want the strip to cover the first round of tool results).

    Session layout (JSONL positions):
      0: user
      1: assistant (tool-use, thinking)
      2: tool result
      3: assistant (tool-use, thinking)
      4: tool result
      5: assistant (tool-use, thinking)
      6: tool result
      7: assistant (final text)
    """
    thinking_a = [{"thinking": "Let me check the filesystem.", "signature": "sigABC111"}]
    thinking_b = [{"thinking": "Now let me check memory.", "signature": "sigDEF222"}]
    thinking_c = [{"thinking": "One more check.", "signature": "sigGHI333"}]

    # 0: user
    sl.append(
        role="user",
        sender=USER,
        room=ROOM,
        event_id="$ev0",
        content="What is the system state?",
    )

    # 1: assistant tool-use turn with thinking
    sl.append(
        role="assistant",
        sender=AGENT_USER,
        room=ROOM,
        event_id=None,
        content="",
        tool_calls=[{"call_id": "tc1", "name": "shell", "input": {"command": "df -h"}}],
        thinking=thinking_a,
        usage={
            "input_tokens": 200,
            "output_tokens": 40,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "tool_calls": 1,
        },
    )

    # 2: tool result
    sl.append(
        role="tool",
        sender=AGENT_USER,
        room=ROOM,
        event_id=None,
        call_id="tc1",
        name="shell",
        output="Filesystem  100G  50G  50G  50% /\n" * 20,
        is_error=False,
    )

    # 3: assistant tool-use turn with thinking
    sl.append(
        role="assistant",
        sender=AGENT_USER,
        room=ROOM,
        event_id=None,
        content="",
        tool_calls=[{"call_id": "tc2", "name": "shell", "input": {"command": "free -h"}}],
        thinking=thinking_b,
        usage={
            "input_tokens": 350,
            "output_tokens": 45,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "tool_calls": 1,
        },
    )

    # 4: tool result
    sl.append(
        role="tool",
        sender=AGENT_USER,
        room=ROOM,
        event_id=None,
        call_id="tc2",
        name="shell",
        output="              total  used  free\nMem:          32G   16G   16G\n" * 20,
        is_error=False,
    )

    # 5: assistant tool-use turn with thinking
    sl.append(
        role="assistant",
        sender=AGENT_USER,
        room=ROOM,
        event_id=None,
        content="",
        tool_calls=[{"call_id": "tc3", "name": "shell", "input": {"command": "uptime"}}],
        thinking=thinking_c,
        usage={
            "input_tokens": 500,
            "output_tokens": 50,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "tool_calls": 1,
        },
    )

    # 6: tool result
    sl.append(
        role="tool",
        sender=AGENT_USER,
        room=ROOM,
        event_id=None,
        call_id="tc3",
        name="shell",
        output=" 10:00:00 up 5 days, load average: 0.1 0.2 0.1\n" * 20,
        is_error=False,
    )

    # 7: final text turn
    sl.append(
        role="assistant",
        sender=AGENT_USER,
        room=ROOM,
        event_id=None,
        content="The system is healthy: 50% disk, 50% memory, 5 days uptime.",
        usage={
            "input_tokens": 600,
            "output_tokens": 30,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "tool_calls": 0,
        },
    )

    # Return the first entry_index where toolstrip starts (covers positions 0..3)
    # Positions 0-3 include the first two tool results (entries 2 and 4 are at
    # positions 2 and 4). We strip at boundary=5 so positions < 5 get placeholders.
    return 5  # strip_boundary index


class TestStatusConsistency:
    """All four accounting surfaces must agree on context_tokens and context_max."""

    def test_all_four_surfaces_agree(self, tmp_path):
        """Core consistency gate: /status, context_status, 80%-alert, overflow-guard
        must all return the same context_tokens and context_max (opus-4-8 window)."""
        # --- Real Agent + real SessionLog ---
        with patch("openalph.agent.assemble_prompt", return_value="system prompt"):
            agent = Agent(make_agent_config(tmp_path))

        sl = SessionLog(workspace=tmp_path, agent_user_id=AGENT_USER)

        # Build the session in JSONL
        strip_idx = _build_session(tmp_path, sl)

        # Mirror in-memory history from JSONL
        history = agent.history(ROOM)
        history.extend(sl.build_context(ROOM))

        # Set /model override → opus-4-8, window = 1_048_576
        agent._room_models[ROOM] = OPUS_MODEL

        # Apply /cache toolstrip marker — strips positions 0..(strip_idx-1)
        _append_toolstrip(sl, strip_idx)

        # Rebuild in-memory from JSONL after strip (mirrors matrix.py behaviour)
        history.clear()
        history.extend(sl.build_context(ROOM))

        # --- Build minimal bot for _build_context_status ---
        bot = MatrixBot.__new__(MatrixBot)
        bot.agent = agent
        bot.session_log = sl
        bot.config = make_matrix_config()
        bot.client = MagicMock()
        bot.client.rooms = {}
        bot.heartbeat = None
        bot.umbral = None

        # --- Surface 1: /status path ---
        s1 = agent.status(ROOM, history=sl.build_context(ROOM))

        # --- Surface 2: context_status tool (bot._build_context_status) ---
        cs = bot._build_context_status(ROOM)

        # --- Surface 3: 80%-alert path (same as /status — same call site) ---
        s_alert = agent.status(ROOM, history=sl.build_context(ROOM))

        # --- Surface 4: overflow-guard raw values ---
        count = agent._estimate_context_tokens(ROOM, history=sl.build_context(ROOM))
        ceiling = agent._resolve_model_limit(ROOM)

        # All context_tokens values must be equal
        assert s1["context_tokens"] == cs["context_tokens"] == s_alert["context_tokens"] == count, (
            f"context_tokens diverged: "
            f"/status={s1['context_tokens']}, "
            f"context_status={cs['context_tokens']}, "
            f"80%-alert={s_alert['context_tokens']}, "
            f"overflow-guard={count}"
        )

        # All context_max values must equal the opus-4-8 window
        assert s1["context_max"] == cs["context_max"] == ceiling == OPUS_WINDOW, (
            f"context_max diverged: "
            f"/status={s1['context_max']}, "
            f"context_status={cs['context_max']}, "
            f"ceiling={ceiling}, "
            f"expected={OPUS_WINDOW}"
        )

        # context_remaining correctness
        assert cs["context_remaining"] == cs["context_max"] - cs["context_tokens"], (
            f"context_remaining incorrect: "
            f"{cs['context_remaining']} != {cs['context_max']} - {cs['context_tokens']}"
        )

