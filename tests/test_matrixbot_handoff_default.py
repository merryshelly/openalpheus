"""Red suite: transport SessionLog handoff_default wiring (kdsn.322.13).

REGRESSION: matrix.py constructed the transport SessionLog with
``handoff_default=getattr(config, "context", ...) is not None and ...``
where ``config`` is the MatrixConfig the caller passes (cli.py:
``MatrixBot(agent, config.matrix)``). MatrixConfig has NO ``context``
field, so getattr returned None and handoff_default was ALWAYS False.

Fleet impact: every DEFAULT build_context render (the /status context
line, restart hydration, gated-room hydration) ignored handoff
boundaries and rendered full pre-boundary history. Found live on the
kdsn.322 canary (SB: /status showed 57,815 tok post-boundary; the
stripped render estimates ~21K).

The fix wires from ``self.agent.config.context.handoff_enabled``
(AgentConfig.context always materializes — field(default_factory)).

These tests go through the REAL MatrixBot.__init__ with a REAL
MatrixConfig — the exact seam the existing real-path pattern bypasses
with ``MatrixBot.__new__`` + a MagicMock session_log, which is why the
bug shipped invisible.
"""

from unittest.mock import MagicMock, patch


from openalph.agent import Agent
from openalph.config import (
    AgentConfig,
    ContextHandoffConfig,
    MatrixConfig,
    ProviderConfig,
)
from openalph.handoff import apply_boundary_and_rebuild
from openalph.matrix import MatrixBot
from openalph.session import SessionLog

ROOM = "!wiring-test:matrix.local"
AGENT_USER = "@agent:matrix.local"


def _cfg(workspace, handoff_enabled=True):
    return AgentConfig(
        name="wiring-test",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8_192,
        providers={"anthropic": ProviderConfig(
            key="anthropic", type="anthropic", api_key="sk-test",
            base_url=None, quirks=[])},
        workspace=workspace,
        max_iterations=10,
        truncation_limit=50_000,
        model_max_tokens=200_000,
        matrix=None,
        reminders=False,
        context=ContextHandoffConfig(handoff_enabled=handoff_enabled),
    )


def _matrix_config():
    # REAL MatrixConfig — the type the production caller passes
    # (cli.py: MatrixBot(agent, config.matrix)). The bug lived in reading
    # .context off THIS object.
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


def _real_bot(tmp_path, handoff_enabled=True):
    """MatrixBot through the REAL __init__ — mocked nio client only."""
    ws = tmp_path
    (ws / "tools").mkdir(exist_ok=True)
    (ws / "tools" / "shell.toml").write_text("[config]\n")
    agent = Agent(_cfg(ws, handoff_enabled))
    with patch("openalph.matrix.AsyncClient", MagicMock()):
        bot = MatrixBot(agent, _matrix_config())
    return bot, agent


class TestTransportHandoffDefault:
    def test_handoff_enabled_wired_from_agent_config(self, tmp_path):
        """THE regression: the transport SessionLog must default to the
        AGENT's context.handoff_enabled, not read .context off the
        MatrixConfig (which never has it -> always False)."""
        bot, agent = _real_bot(tmp_path, handoff_enabled=True)
        assert isinstance(bot.session_log, SessionLog), (
            "real __init__ must build a real SessionLog")
        assert bot.session_log.handoff_default is True, (
            "handoff_default must follow agent.config.context."
            "handoff_enabled (default True) — reading it off MatrixConfig "
            "yields None -> False, the fleet-wide canary bug")

    def test_kill_switch_propagates(self, tmp_path):
        bot, _ = _real_bot(tmp_path, handoff_enabled=False)
        assert bot.session_log.handoff_default is False

    def test_status_render_respects_boundary(self, tmp_path):
        """End-to-end on the transport log: with a boundary applied,
        build_context (what /status consumes) must strip pre-boundary
        entries instead of rendering full history."""
        bot, agent = _real_bot(tmp_path, handoff_enabled=True)
        sl = bot.session_log
        for i in range(5):
            sl.append(role="user", content="u" * 400, room=ROOM,
                      sender="@op:matrix.local")
            sl.append(role="assistant", content="a" * 200, room=ROOM,
                      sender=AGENT_USER)
        res = apply_boundary_and_rebuild(
            agent, sl, ROOM, trigger="tool", exclude_inflight=False)
        assert res["applied"] is True
        rendered = sl.build_context(ROOM)
        raw = sl.read(ROOM)
        assert len(rendered) < len(raw), (
            f"post-boundary render must strip: rendered {len(rendered)} of "
            f"{len(raw)} entries — handoff_default wiring is broken")
        # and the /status estimate path sees the stripped size — measurably
        # smaller than the full render of the same file (legacy behavior).
        # (Legacy renders skip system entries — the marker — so full is
        # len(raw) - 1; the meaningful pin is CONTENT: pre-boundary turns
        # render in legacy mode and vanish under the boundary.)
        full_sl = SessionLog(tmp_path, AGENT_USER, handoff_default=False)
        full_render = full_sl.build_context(ROOM)
        assert len(full_render) > len(rendered), (
            "legacy render must carry more messages than the stripped one")
        assert sum(1 for mm in full_render
                   if mm.get("content") == "u" * 400) == 5, (
            "legacy render must carry all pre-boundary turns")
        assert not [mm for mm in rendered
                    if mm.get("content") == "u" * 400], (
            "stripped render must drop every pre-boundary turn")
        stripped_est = agent.status(ROOM, history=rendered)["context_tokens"]
        full_est = agent.status(ROOM, history=full_render)["context_tokens"]
        assert stripped_est < full_est, (
            "/status estimate must reflect the strip: stripped "
            f"{stripped_est} must be < full {full_est}")
