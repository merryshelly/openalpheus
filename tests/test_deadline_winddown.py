"""Deadline-winddown reminder (cairn-eval .61.22; cairn fork .61.19's OA seam).

TDD suite for the wall-clock axis of the two-phase task timer. Spec decisions
(D-numbered, inline):

  D1  New trigger id ``deadline-winddown``; tool_loop_boundary only.
  D2  ReminderState gains ``elapsed_seconds: float = 0`` and
      ``deadline_seconds: float = 0`` — defaults keep every existing
      construction valid; 0 deadline = silent skip (the available_tokens
      convention: fewer nudges, never false urgency).
  D3  Predicate: deadline_seconds > 0 AND elapsed_seconds >= deadline_seconds
      (inclusive at the exact deadline).
  D4  Once per turn: ``_deadline_fired_this_turn`` latch, cleared by
      reset_turn() (the T4 seam); cleared by reset() too.
  D5  Text names both numbers (int seconds) and the hard-kill consequence.
  D6  AgentConfig gains ``soft_deadline_seconds: float = 0``; the agent's
      tool-loop-boundary ReminderState construction populates elapsed from a
      per-turn monotonic baseline and deadline from config. Turn-start site
      unchanged (trigger is boundary-only; dataclass defaults keep it valid).
  D7  ``openalph exec --soft-deadline N`` (int seconds, optional) → cmd_exec
      replaces config like --max-turns. Chat paths stay 0 → fleet-silent.
  D8  Evaluation order: after the context ladder, before T4 (both may fire at
      the same boundary; reminders dilute but never conflict).
  D9  No persistence: no rehydrate branch — a serialized deadline-winddown
      entry must NOT latch the per-turn flag across sessions.

Pure-engine style mirrors test_context_nudge_ladder.py.
"""

from openalph.cli import parse_args
from openalph.config import AgentConfig, ProviderConfig
from openalph.reminders import ReminderEngine, ReminderState

TRIGGER = "deadline-winddown"
TAG_OPEN = chr(60) + "system-reminder" + chr(62)
TAG_CLOSE = chr(60) + "/system-reminder" + chr(62)


def _cfg(tmp_path, **kw):
    defaults = dict(
        name="test-agent",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": ProviderConfig(
            key="anthropic", type="anthropic", api_key="sk-test",
            base_url=None, quirks=[],
        )},
        workspace=tmp_path,
        max_iterations=100,
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=None,
    )
    defaults.update(kw)
    return AgentConfig(**defaults)


def _state(**kw):
    defaults = dict(
        evaluation_point="tool_loop_boundary",
        iteration=0,
        max_iterations=100,
        context_tokens=10000,
        context_limit=200000,
        completed_turns=0,
        turn_source=None,
        tool_calls_this_turn={},
        tool_calls_session={},
        todo_list=[],
        enabled_tools=set(),
    )
    defaults.update(kw)
    return ReminderState(**defaults)


def _engine(tmp_path, **kw):
    return ReminderEngine(_cfg(tmp_path, **kw))


def _dw(results):
    return [r for r in results if r.trigger == TRIGGER]


class TestPredicate:
    def test_fires_at_exact_deadline_inclusive(self, tmp_path):
        eng = _engine(tmp_path)
        out = _dw(eng.evaluate(_state(elapsed_seconds=1500, deadline_seconds=1500)))
        assert len(out) == 1

    def test_fires_past_deadline(self, tmp_path):
        eng = _engine(tmp_path)
        out = _dw(eng.evaluate(_state(elapsed_seconds=1600.5, deadline_seconds=1500)))
        assert len(out) == 1

    def test_silent_before_deadline(self, tmp_path):
        eng = _engine(tmp_path)
        assert _dw(eng.evaluate(_state(elapsed_seconds=1499, deadline_seconds=1500))) == []

    def test_silent_when_deadline_default_zero(self, tmp_path):
        eng = _engine(tmp_path)
        # Huge elapsed, no deadline known — never fire (D2 fail-safe).
        assert _dw(eng.evaluate(_state(elapsed_seconds=999999))) == []

    def test_boundary_only_not_turn_start(self, tmp_path):
        eng = _engine(tmp_path)
        assert _dw(eng.evaluate(_state(
            evaluation_point="turn_start",
            elapsed_seconds=2000, deadline_seconds=1500,
        ))) == []


class TestLatchSemantics:
    def test_once_per_turn_no_duplicate(self, tmp_path):
        eng = _engine(tmp_path)
        first = eng.evaluate(_state(elapsed_seconds=1500, deadline_seconds=1500))
        second = eng.evaluate(_state(elapsed_seconds=1510, deadline_seconds=1500))
        assert len(_dw(first)) == 1
        assert _dw(second) == []

    def test_rearm_after_reset_turn(self, tmp_path):
        eng = _engine(tmp_path)
        eng.evaluate(_state(elapsed_seconds=1500, deadline_seconds=1500))
        eng.reset_turn()
        again = eng.evaluate(_state(elapsed_seconds=1520, deadline_seconds=1500))
        assert len(_dw(again)) == 1

    def test_reset_clears_latch(self, tmp_path):
        eng = _engine(tmp_path)
        eng.evaluate(_state(elapsed_seconds=1500, deadline_seconds=1500))
        eng.reset()
        again = eng.evaluate(_state(elapsed_seconds=1500, deadline_seconds=1500))
        assert len(_dw(again)) == 1

    def test_rehydrate_does_not_latch(self, tmp_path):
        # D9: a serialized entry from a previous session is informational only.
        eng = _engine(tmp_path)
        eng.rehydrate([{"source": "reminder", "trigger": TRIGGER, "text": "x"}])
        out = eng.evaluate(_state(elapsed_seconds=1500, deadline_seconds=1500))
        assert len(_dw(out)) == 1


class TestTextContract:
    def test_text_names_both_numbers_and_consequence(self, tmp_path):
        eng = _engine(tmp_path)
        (rem,) = _dw(eng.evaluate(_state(elapsed_seconds=1623, deadline_seconds=1500)))
        assert "1623" in rem.text
        assert "1500" in rem.text
        assert "hard kill" in rem.text

    def test_content_is_framed_reminder(self, tmp_path):
        eng = _engine(tmp_path)
        (rem,) = _dw(eng.evaluate(_state(elapsed_seconds=1500, deadline_seconds=1500)))
        assert rem.content.startswith(TAG_OPEN)
        assert rem.content.endswith(TAG_CLOSE)


class TestPlumbing:
    def test_agent_config_accepts_soft_deadline_default_zero(self, tmp_path):
        cfg = _cfg(tmp_path)
        assert getattr(cfg, "soft_deadline_seconds", None) == 0

    def test_exec_soft_deadline_flag_parses(self):
        args = parse_args(
            ["exec", "--config", "/tmp/agent.toml", "--task-file", "t.md",
             "--soft-deadline", "1500"]
        )
        assert args.soft_deadline == 1500

    def test_exec_soft_deadline_optional_default_none(self):
        args = parse_args(["exec", "--config", "/tmp/agent.toml",
                           "--task-file", "t.md"])
        assert args.soft_deadline is None
