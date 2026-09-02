"""Tests for the `openalph exec` one-shot subcommand (bead workspace-e2uh.149).

`exec` is the OA half of the worker-migration driver: a fresh Agent, a single
`handle_input` call, and EXACTLY ONE JSON line on stdout (everything else,
including all logging, goes to stderr). The Stigmergy driver parses that stdout
as JSON, so stdout discipline is load-bearing — these tests pin it.

Written RED-first (before the implementation exists in cli.py). cmd_exec is a
sync function that runs asyncio.run() internally, mirroring cmd_chat. Tests
stub the Agent symbol (patch openalph.agent.Agent) and load_agent_config,
matching the existing test_cli_chat.py fixtures/patterns.

Exit-code contract (spec §3.9): 0 done / 1 failed / 2 infra / 3 wedged.
"""

import json
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai

import pytest

from openalph.config import AgentConfig, ProviderConfig
from openalph.prompt import INJECTION_DEFENSE
from openalph.tools import BUILTIN_TOOLS


# The exact iteration-cap sentinel appended to history by Agent.handle_input
# on max_iterations exhaustion (agent.py). Pinned verbatim so the structural
# ceiling detection cannot drift silently from the agent's own message.
CAP_SENTINEL = "[SYSTEM: Tool call limit reached."

# The four keys every exec usage object must carry.
USAGE_KEYS = ("in", "cached", "out", "reasoning")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_config(tmp_path, *, name="exec-agent", default_model="p/model",
                max_iterations=100) -> AgentConfig:
    """Build a REAL AgentConfig (not a MagicMock) so dataclasses.replace()
    exercises the real --model/--max-turns pre-construction override."""
    provider = ProviderConfig(key="p", type="openai", api_key="sk-test",
                              base_url="http://127.0.0.1:18081")
    return AgentConfig(
        name=name,
        default_model=default_model,
        max_tokens=8192,
        providers={"p": provider},
        workspace=Path(tmp_path),
        max_iterations=max_iterations,
    )


def make_agent_stub(*, history=None, last_turn_usage=None, last_stop_reason=None,
                    response_text="FINAL TEXT", tools=(), handle_input=None):
    """Build a MagicMock shaped like the real Agent for the bits exec reads.

    `history` is the list returned by agent.history(room_id); `last_turn_usage`
    is the dict returned by agent.last_turn_usage(room_id); `last_stop_reason`
    the value from agent.last_stop_reason(room_id); `tools` becomes
    agent.tools (post-construction inventory, what a BUILTIN-resolved set
    should equal); `handle_input` overrides the default async return.
    """
    agent = MagicMock()
    agent.last_turn_usage = MagicMock(return_value=last_turn_usage)
    agent.last_stop_reason = MagicMock(return_value=last_stop_reason)
    agent.history = MagicMock(return_value=history if history is not None else [])
    agent.tools = list(tools)
    if handle_input is None:
        agent.handle_input = AsyncMock(return_value=response_text)
    else:
        agent.handle_input = AsyncMock(side_effect=handle_input)
    return agent


def run_exec(argv, *, config, agent, stdin=None):
    """Run main() over `argv` with load_agent_config + Agent patched.

    Returns (stdout_str, stderr_str, exit_code). The exec surface must sys.exit
    with the numeric code, so SystemExit is expected here.
    """
    from openalph.cli import main

    out, err = StringIO(), StringIO()
    stdin_cm = (patch("sys.stdin", StringIO(stdin)) if stdin is not None
                else _nullcontext())
    with (
        patch("openalph.cli.load_agent_config", return_value=config),
        patch("openalph.agent.Agent", return_value=agent),
        stdin_cm,
        redirect_stdout(out),
        redirect_stderr(err),
    ):
        with pytest.raises(SystemExit) as exc_info:
            main(argv)
    return out.getvalue(), err.getvalue(), exc_info.value.code


class _nullcontext:
    """Minimal no-op context manager for the stdin=None case."""
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def parse_single_json(stdout):
    """Assert `stdout` is EXACTLY one JSON object (nothing else) and return it.

    This is the load-bearing stdout-discipline check: Stigmergy parses the raw
    stdout as one JSON object, so any stray byte (a log line, a banner) breaks
    classification. Whitespace outside the single object is tolerated.
    """
    stripped = stdout.strip()
    assert stripped, "stdout must carry exactly one JSON line (got empty)"
    obj, end = json.JSONDecoder().raw_decode(stripped)
    assert stripped[end:].strip() == "", (
        f"stdout carries content AFTER the JSON object: {stripped[end:]!r}"
    )
    assert isinstance(obj, dict), f"stdout is not a JSON object: {type(obj)!r}"
    return obj


def assert_usage_four_keys(usage):
    assert list(usage.keys()) == list(USAGE_KEYS), f"usage keys {list(usage.keys())}"
    for k in USAGE_KEYS:
        assert isinstance(usage[k], int), f"usage[{k}] not int: {usage[k]!r}"
        assert usage[k] >= 0, f"usage[{k}] negative: {usage[k]}"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class TestExecParseArgs:
    def test_command_and_required_agent(self):
        from openalph.cli import parse_args

        args = parse_args(["exec", "--agent", "stigmergy-worker",
                           "--task-file", "/task/prompt.md"])
        assert args.command == "exec"
        assert args.agent == "stigmergy-worker"
        assert args.task_file == "/task/prompt.md"

    def test_defaults(self):
        from openalph.cli import parse_args

        args = parse_args(["exec", "--agent", "w", "--task-file", "p"])
        assert args.model is None
        assert args.effort is None
        assert args.max_turns is None
        assert args.tools is None
        assert args.room is None

    def test_all_flags(self):
        from openalph.cli import parse_args

        args = parse_args([
            "exec", "--agent", "w", "--task-file", "-",
            "--model", "blackwell/qwen38-27b-fp8",
            "--effort", "medium", "--max-turns", "40",
            "--tools", "shell,file_read", "--room", "d1",
        ])
        assert args.task_file == "-"
        assert args.model == "blackwell/qwen38-27b-fp8"
        assert args.effort == "medium"
        assert args.max_turns == 40
        assert args.tools == "shell,file_read"
        assert args.room == "d1"

    def test_task_file_required(self):
        from openalph.cli import parse_args

        with pytest.raises(SystemExit):
            parse_args(["exec", "--agent", "w"])

    def test_agent_required(self):
        from openalph.cli import parse_args

        with pytest.raises(SystemExit):
            parse_args(["exec", "--task-file", "p"])

    def test_effort_choice_validated(self):
        from openalph.cli import parse_args

        with pytest.raises(SystemExit):
            parse_args(["exec", "--agent", "w", "--task-file", "p",
                        "--effort", "banana"])

    def test_exec_registered_in_main_dispatch(self):
        """main() must route the `exec` command to a handler (not KeyError)."""
        from openalph.cli import main

        with patch("openalph.cli.cmd_exec") as mock_exec:
            main(["exec", "--agent", "w", "--task-file", "p"])
            mock_exec.assert_called_once()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestExecHappyPath:
    def test_one_json_line_exit0_usage_and_content(self, tmp_path):
        config = make_config(tmp_path)
        agent = make_agent_stub(
            last_turn_usage={"input_tokens": 5, "cache_read_tokens": 2,
                             "output_tokens": 3, "cache_creation_tokens": 0},
            last_stop_reason="end_turn",
            response_text="FINAL TEXT",
        )
        task = tmp_path / "prompt.md"
        task.write_text("do the thing")

        stdout, stderr, code = run_exec(
            ["exec", "--agent", "w", "--task-file", str(task)],
            config=config, agent=agent,
        )

        assert code == 0
        obj = parse_single_json(stdout)
        assert obj["status"] == "done"
        assert obj["content"] == "FINAL TEXT"
        assert_usage_four_keys(obj["usage"])
        assert obj["usage"] == {"in": 5, "cached": 2, "out": 3, "reasoning": 0}
        assert obj["stop_reason"] == "end_turn"
        assert obj["ceiling_trip"] is None
        assert obj["deny_reason"] is None
        assert obj["tool_trace"] == []
        assert obj["detail"] == ""

    def test_task_text_passed_verbatim(self, tmp_path):
        """The rendered prompt is the single user message, verbatim (no wrap)."""
        config = make_config(tmp_path)
        task = tmp_path / "prompt.md"
        body = "line one\nline two [SYSTEM: whatever]\n  indented"
        task.write_text(body)
        agent = make_agent_stub()

        run_exec(["exec", "--agent", "w", "--task-file", str(task)],
                 config=config, agent=agent)

        agent.handle_input.assert_awaited_once()
        assert agent.handle_input.call_args[0][0] == body

    def test_handle_input_called_with_room_id_and_no_callbacks(self, tmp_path):
        config = make_config(tmp_path)
        agent = make_agent_stub()
        task = tmp_path / "p.md"
        task.write_text("t")

        run_exec(["exec", "--agent", "w", "--task-file", str(task)],
                 config=config, agent=agent)

        kw = agent.handle_input.call_args.kwargs
        # default room label is `_exec`
        assert agent.handle_input.call_args.args[1] == "_exec"
        # no chat plumbing leaks in
        assert kw.get("on_tool_intent") is None
        assert kw.get("callbacks") is None

    def test_room_flag_used(self, tmp_path):
        config = make_config(tmp_path)
        agent = make_agent_stub()
        task = tmp_path / "p.md"
        task.write_text("t")

        run_exec(["exec", "--agent", "w", "--task-file", str(task),
                  "--room", "dispatch-42"], config=config, agent=agent)

        assert agent.handle_input.call_args.args[1] == "dispatch-42"

    def test_nothing_else_on_stdout(self, tmp_path):
        """Stdout must be ONLY the JSON object (no banner/log)."""
        config = make_config(tmp_path)
        agent = make_agent_stub(response_text="hi")
        task = tmp_path / "p.md"
        task.write_text("t")

        stdout, stderr, code = run_exec(
            ["exec", "--agent", "w", "--task-file", str(task)],
            config=config, agent=agent,
        )
        # stdout is a single object and nothing else (raw_decode + tail check)
        parse_single_json(stdout)
        assert code == 0


# ---------------------------------------------------------------------------
# Effort mapping
# ---------------------------------------------------------------------------


class TestExecEffort:
    def test_none_maps_to_off(self, tmp_path):
        config = make_config(tmp_path)
        agent = make_agent_stub()
        task = tmp_path / "p.md"
        task.write_text("t")

        run_exec(["exec", "--agent", "w", "--task-file", str(task),
                  "--effort", "none"], config=config, agent=agent)

        assert agent.handle_input.call_args.kwargs.get("thinking") == "off"

    def test_medium_maps_to_medium(self, tmp_path):
        config = make_config(tmp_path)
        agent = make_agent_stub()
        task = tmp_path / "p.md"
        task.write_text("t")

        run_exec(["exec", "--agent", "w", "--task-file", str(task),
                  "--effort", "medium"], config=config, agent=agent)

        assert agent.handle_input.call_args.kwargs.get("thinking") == "medium"

    def test_low_maps_to_low(self, tmp_path):
        config = make_config(tmp_path)
        agent = make_agent_stub()
        task = tmp_path / "p.md"
        task.write_text("t")

        run_exec(["exec", "--agent", "w", "--task-file", str(task),
                  "--effort", "low"], config=config, agent=agent)

        assert agent.handle_input.call_args.kwargs.get("thinking") == "low"

    def test_xhigh_maps_to_xhigh(self, tmp_path):
        config = make_config(tmp_path)
        agent = make_agent_stub()
        task = tmp_path / "p.md"
        task.write_text("t")

        run_exec(["exec", "--agent", "w", "--task-file", str(task),
                  "--effort", "xhigh"], config=config, agent=agent)

        assert agent.handle_input.call_args.kwargs.get("thinking") == "xhigh"

    def test_high_maps_to_high(self, tmp_path):
        """SB ruling 2026-09-02: 'high' is card-native on Synthetic GLM/Kimi
        (lh3c.11 wire probe: 1:1 pass-through) — a legal charter-native
        effort value for the subscription worker rungs."""
        config = make_config(tmp_path)
        agent = make_agent_stub()
        task = tmp_path / "p.md"
        task.write_text("t")

        run_exec(["exec", "--agent", "w", "--task-file", str(task),
                  "--effort", "high"], config=config, agent=agent)

        assert agent.handle_input.call_args.kwargs.get("thinking") == "high"

    def test_no_effort_sends_none(self, tmp_path):
        """When --effort is absent, thinking is not forced (None)."""
        config = make_config(tmp_path)
        agent = make_agent_stub()
        task = tmp_path / "p.md"
        task.write_text("t")

        run_exec(["exec", "--agent", "w", "--task-file", str(task)],
                 config=config, agent=agent)

        assert agent.handle_input.call_args.kwargs.get("thinking") is None


# ---------------------------------------------------------------------------
# --model / --max-turns config replacement (pre-construction)
# ---------------------------------------------------------------------------


class TestExecConfigOverrides:
    def test_model_override_replaces_default_model(self, tmp_path):
        """--model must replace config.default_model BEFORE Agent(config)."""
        config = make_config(tmp_path, default_model="p/orig")
        built = []

        def _capture(cfg):
            built.append(cfg)
            return make_agent_stub()

        task = tmp_path / "p.md"
        task.write_text("t")
        out, err = StringIO(), StringIO()
        with (
            patch("openalph.cli.load_agent_config", return_value=config),
            patch("openalph.agent.Agent", side_effect=_capture),
            redirect_stdout(out), redirect_stderr(err),
        ):
            with pytest.raises(SystemExit) as exc_info:
                from openalph.cli import main
                main(["exec", "--agent", "w", "--task-file", str(task),
                      "--model", "p/override"])
        assert exc_info.value.code == 0
        assert len(built) == 1
        assert built[0].default_model == "p/override"

    def test_max_turns_replaces_max_iterations(self, tmp_path):
        config = make_config(tmp_path, max_iterations=100)
        built = []

        def _capture(cfg):
            built.append(cfg)
            return make_agent_stub()

        task = tmp_path / "p.md"
        task.write_text("t")
        out, err = StringIO(), StringIO()
        with (
            patch("openalph.cli.load_agent_config", return_value=config),
            patch("openalph.agent.Agent", side_effect=_capture),
            redirect_stdout(out), redirect_stderr(err),
        ):
            with pytest.raises(SystemExit) as exc_info:
                from openalph.cli import main
                main(["exec", "--agent", "w", "--task-file", str(task),
                      "--max-turns", "40"])
        assert exc_info.value.code == 0
        assert built[0].max_iterations == 40

    def test_model_and_max_turns_combined(self, tmp_path):
        config = make_config(tmp_path, default_model="p/orig", max_iterations=100)
        built = []

        def _capture(cfg):
            built.append(cfg)
            return make_agent_stub()

        task = tmp_path / "p.md"
        task.write_text("t")
        with (
            patch("openalph.cli.load_agent_config", return_value=config),
            patch("openalph.agent.Agent", side_effect=_capture),
            redirect_stdout(StringIO()), redirect_stderr(StringIO()),
        ):
            with pytest.raises(SystemExit):
                from openalph.cli import main
                main(["exec", "--agent", "w", "--task-file", str(task),
                      "--model", "p/x", "--max-turns", "7"])
        assert built[0].default_model == "p/x"
        assert built[0].max_iterations == 7


# ---------------------------------------------------------------------------
# --tools resolution
# ---------------------------------------------------------------------------


class TestExecTools:
    def test_valid_tools_resolved_and_installed(self, tmp_path):
        """--tools names resolve to ToolDefs and are installed on the agent."""
        from openalph.tools import ToolDef

        config = make_config(tmp_path)
        built = {}

        def _capture(cfg):
            built["agent"] = make_agent_stub(tools=[
                ToolDef(name="shell", description="s", parameters={}, config={}),
                ToolDef(name="file_read", description="f", parameters={}, config={}),
            ])
            return built["agent"]

        task = tmp_path / "p.md"
        task.write_text("t")
        with (
            patch("openalph.cli.load_agent_config", return_value=config),
            patch("openalph.agent.Agent", side_effect=_capture),
            redirect_stdout(StringIO()), redirect_stderr(StringIO()),
        ):
            with pytest.raises(SystemExit) as exc_info:
                from openalph.cli import main
                main(["exec", "--agent", "w", "--task-file", str(task),
                      "--tools", "shell,file_read"])
        assert exc_info.value.code == 0
        # exec must have RESOLVED these names against BUILTIN_TOOLS (no error)
        for name in ("shell", "file_read"):
            assert name in BUILTIN_TOOLS

    def test_unknown_tool_exits1_empty_stdout(self, tmp_path):
        config = make_config(tmp_path)
        agent = make_agent_stub()
        task = tmp_path / "p.md"
        task.write_text("t")

        stdout, stderr, code = run_exec(
            ["exec", "--agent", "w", "--task-file", str(task),
             "--tools", "shell,does_not_exist"],
            config=config, agent=agent,
        )

        assert code == 1
        assert stdout.strip() == ""  # no partial JSON on failure
        assert "does_not_exist" in stderr
        # handle_input must never run
        agent.handle_input.assert_not_awaited()

    def test_all_builtin_names_resolve(self):
        """Every BUILTIN_TOOLS key is a valid --tools value (no silent reject).

        Calls the resolver directly so the registry contract is pinned without
        needing a full exec run.
        """
        from openalph.cli import _resolve_exec_tools

        for name in BUILTIN_TOOLS:
            defs = _resolve_exec_tools([name])
            assert [d.name for d in defs] == [name]


# ---------------------------------------------------------------------------
# --task-file
# ---------------------------------------------------------------------------


class TestExecTaskFile:
    def test_reads_file_content(self, tmp_path):
        config = make_config(tmp_path)
        task = tmp_path / "prompt.md"
        task.write_text("the task body")
        agent = make_agent_stub()

        run_exec(["exec", "--agent", "w", "--task-file", str(task)],
                 config=config, agent=agent)
        assert agent.handle_input.call_args[0][0] == "the task body"

    def test_missing_file_exits1_stderr_empty_stdout(self, tmp_path):
        config = make_config(tmp_path)
        agent = make_agent_stub()

        stdout, stderr, code = run_exec(
            ["exec", "--agent", "w", "--task-file", str(tmp_path / "nope.md")],
            config=config, agent=agent,
        )

        assert code == 1
        assert stdout.strip() == ""
        assert "nope.md" in stderr
        agent.handle_input.assert_not_awaited()

    def test_stdin_dash_reads_stdin(self, tmp_path):
        config = make_config(tmp_path)
        agent = make_agent_stub()

        stdout, stderr, code = run_exec(
            ["exec", "--agent", "w", "--task-file", "-"],
            config=config, agent=agent, stdin="piped task content",
        )

        assert code == 0
        assert agent.handle_input.call_args[0][0] == "piped task content"


# ---------------------------------------------------------------------------
# Iteration-cap (structural ceiling detection)
# ---------------------------------------------------------------------------


class TestExecIterationCap:
    def test_sentinel_in_history_trips_ceiling(self, tmp_path):
        """Sentinel in the room history -> ceiling_trip=driver_turns, failed, 1."""
        config = make_config(tmp_path)
        history = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "user", "content": CAP_SENTINEL + " rest of message"},
            {"role": "assistant", "content": "summary text"},
        ]
        agent = make_agent_stub(history=history, response_text="summary text")

        stdout, stderr, code = run_exec(
            ["exec", "--agent", "w", "--task-file", "-"],
            config=config, agent=agent, stdin="task",
        )

        assert code == 1
        obj = parse_single_json(stdout)
        assert obj["status"] == "failed"
        assert obj["ceiling_trip"] == "driver_turns"
        assert obj["deny_reason"] is None
        assert obj["content"] == "summary text"

    def test_no_sentinel_no_ceiling(self, tmp_path):
        config = make_config(tmp_path)
        history = [{"role": "user", "content": "task"},
                   {"role": "assistant", "content": "ok"}]
        agent = make_agent_stub(history=history, response_text="ok")

        stdout, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", "-"],
            config=config, agent=agent, stdin="task",
        )
        assert code == 0
        obj = parse_single_json(stdout)
        assert obj["status"] == "done"
        assert obj["ceiling_trip"] is None


# ---------------------------------------------------------------------------
# Exception classification (spec §3.8)
# ---------------------------------------------------------------------------


def _make_http_error(status_code, *, header=None, body=None):
    """Build an openai.APIStatusError (with an httpx.Response) wrapped in a
    ProviderError via `from e`, the way provider.py actually re-raises."""
    request = httpx.Request("POST", "http://127.0.0.1:18081/chat/completions")
    headers = {"content-type": "application/json"}
    if header is not None:
        headers[header[0]] = header[1]
    response = httpx.Response(status_code, request=request,
                              headers=headers,
                              content=(body or "{}").encode())
    sdk_err = openai.APIStatusError("boom", response=response,
                                    body={"error": {"x": 1}})
    wrapper = Exception(f"Provider error: {status_code}")
    wrapper.__cause__ = sdk_err
    return wrapper


class TestExecDenyMarker:
    def test_deny_header_classifies_failed(self, tmp_path):
        """x-stigmergy-deny-reason header -> failed + deny_reason, exit 1."""
        config = make_config(tmp_path)
        err = _make_http_error(402, header=("x-stigmergy-deny-reason", "quota-calls"))
        agent = make_agent_stub(handle_input=AsyncMock(side_effect=err))

        stdout, stderr, code = run_exec(
            ["exec", "--agent", "w", "--task-file", "-"],
            config=config, agent=agent, stdin="task",
        )

        assert code == 1
        obj = parse_single_json(stdout)
        assert obj["status"] == "failed"
        assert obj["deny_reason"] == "quota-calls"
        assert "quota-calls" in obj["detail"]  # relay-deny:<reason> provenance

    def test_deny_body_classifies_failed(self, tmp_path):
        """Deny body {error:{type: stigmergy_relay_deny, reason}} also works."""
        config = make_config(tmp_path)
        body = json.dumps({"error": {"type": "stigmergy_relay_deny",
                                     "reason": "quota-tokens"}})
        err = _make_http_error(402, body=body)  # no header, body only
        agent = make_agent_stub(handle_input=AsyncMock(side_effect=err))

        stdout, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", "-"],
            config=config, agent=agent, stdin="task",
        )

        assert code == 1
        obj = parse_single_json(stdout)
        assert obj["status"] == "failed"
        assert obj["deny_reason"] == "quota-tokens"

    def test_http_error_without_marker_is_infra(self, tmp_path):
        """HTTP error WITHOUT the deny marker -> infra, exit 2 (never failed)."""
        config = make_config(tmp_path)
        err = _make_http_error(503)  # plain upstream error, no marker
        agent = make_agent_stub(handle_input=AsyncMock(side_effect=err))

        stdout, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", "-"],
            config=config, agent=agent, stdin="task",
        )

        assert code == 2
        obj = parse_single_json(stdout)
        assert obj["status"] == "infra"
        assert obj["deny_reason"] is None


# ---------------------------------------------------------------------------
# Transport / infra errors
# ---------------------------------------------------------------------------


class TestExecTransport:
    def test_remote_protocol_error_is_infra(self, tmp_path):
        config = make_config(tmp_path)
        err = Exception("transport broken")
        err.__cause__ = httpx.RemoteProtocolError("peer closed connection")
        agent = make_agent_stub(handle_input=AsyncMock(side_effect=err))

        stdout, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", "-"],
            config=config, agent=agent, stdin="task",
        )

        assert code == 2
        obj = parse_single_json(stdout)
        assert obj["status"] == "infra"

    def test_connect_error_is_infra(self, tmp_path):
        config = make_config(tmp_path)
        err = Exception("unreachable")
        err.__cause__ = httpx.ConnectError("connection refused")
        agent = make_agent_stub(handle_input=AsyncMock(side_effect=err))

        stdout, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", "-"],
            config=config, agent=agent, stdin="task",
        )

        assert code == 2
        assert parse_single_json(stdout)["status"] == "infra"

    def test_api_timeout_is_infra(self, tmp_path):
        config = make_config(tmp_path)
        err = Exception("timed out")
        err.__cause__ = openai.APITimeoutError(
            request=httpx.Request("POST", "http://127.0.0.1:18081/chat/completions"))
        agent = make_agent_stub(handle_input=AsyncMock(side_effect=err))

        stdout, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", "-"],
            config=config, agent=agent, stdin="task",
        )

        assert code == 2
        assert parse_single_json(stdout)["status"] == "infra"

    def test_marker_wins_over_infra_in_chain(self, tmp_path):
        """A deny marker on any object in the chain classifies as failed even
        though a transport-type error is also present (marker takes
        precedence — a deny is Stigmergy's own budget, not infra)."""
        config = make_config(tmp_path)
        top = Exception("top-level")
        top.response = type("_R", (), {"headers":
                                       {"x-stigmergy-deny-reason": "quota-calls"}})
        # also put a transport error in the chain
        top.__cause__ = httpx.RemoteProtocolError("stream cut")
        agent = make_agent_stub(handle_input=AsyncMock(side_effect=top))

        stdout, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", "-"],
            config=config, agent=agent, stdin="task",
        )

        assert code == 1
        obj = parse_single_json(stdout)
        assert obj["status"] == "failed"
        assert obj["deny_reason"] == "quota-calls"


# ---------------------------------------------------------------------------
# Generic failure (no infra marker) + bounded detail
# ---------------------------------------------------------------------------


class TestExecGenericFailure:
    def test_generic_exception_failed_exit1(self, tmp_path):
        config = make_config(tmp_path)
        agent = make_agent_stub(handle_input=AsyncMock(
            side_effect=RuntimeError("something unexpected happened")))

        stdout, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", "-"],
            config=config, agent=agent, stdin="task",
        )

        assert code == 1
        obj = parse_single_json(stdout)
        assert obj["status"] == "failed"
        assert obj["deny_reason"] is None
        assert "something unexpected" in obj["detail"]

    def test_detail_bounded_500(self, tmp_path):
        config = make_config(tmp_path)
        big = "x" * 5000
        agent = make_agent_stub(handle_input=AsyncMock(
            side_effect=RuntimeError(big)))

        stdout, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", "-"],
            config=config, agent=agent, stdin="task",
        )

        assert code == 1
        obj = parse_single_json(stdout)
        assert len(obj["detail"]) <= 500
        assert obj["detail"] == big[:500]


# ---------------------------------------------------------------------------
# Partial usage on failure (best-effort, zeros if unavailable)
# ---------------------------------------------------------------------------


class TestExecFailureUsage:
    def test_usage_present_with_values_on_failure(self, tmp_path):
        """A failed turn still reports the partial usage it accumulated."""
        config = make_config(tmp_path)
        agent = make_agent_stub(
            handle_input=AsyncMock(side_effect=RuntimeError("boom")),
            last_turn_usage={"input_tokens": 11, "cache_read_tokens": 4,
                             "output_tokens": 7, "cache_creation_tokens": 0},
        )

        stdout, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", "-"],
            config=config, agent=agent, stdin="task",
        )

        assert code == 1
        obj = parse_single_json(stdout)
        assert_usage_four_keys(obj["usage"])
        assert obj["usage"] == {"in": 11, "cached": 4, "out": 7, "reasoning": 0}

    def test_usage_zeroed_when_unavailable(self, tmp_path):
        """No usage recorded -> all four keys zero (never fabricated)."""
        config = make_config(tmp_path)
        agent = make_agent_stub(
            handle_input=AsyncMock(side_effect=RuntimeError("boom")),
            last_turn_usage=None,  # no usage ever recorded
        )

        stdout, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", "-"],
            config=config, agent=agent, stdin="task",
        )

        assert code == 1
        obj = parse_single_json(stdout)
        assert obj["usage"] == {"in": 0, "cached": 0, "out": 0, "reasoning": 0}

    def test_usage_present_on_infra_failure(self, tmp_path):
        config = make_config(tmp_path)
        err = Exception("down")
        err.__cause__ = httpx.ConnectError("refused")
        agent = make_agent_stub(
            handle_input=AsyncMock(side_effect=err),
            last_turn_usage={"input_tokens": 3, "cache_read_tokens": 0,
                             "output_tokens": 1, "cache_creation_tokens": 0},
        )

        stdout, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", "-"],
            config=config, agent=agent, stdin="task",
        )

        assert code == 2
        obj = parse_single_json(stdout)
        assert obj["usage"]["out"] == 1
        assert obj["usage"]["in"] == 3


# ---------------------------------------------------------------------------
# stop_reason emission
# ---------------------------------------------------------------------------


class TestExecStopReason:
    def test_stop_reason_from_last_response(self, tmp_path):
        config = make_config(tmp_path)
        agent = make_agent_stub(last_stop_reason="end_turn")
        task = tmp_path / "p.md"
        task.write_text("t")

        stdout, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", str(task)],
            config=config, agent=agent,
        )
        assert code == 0
        assert parse_single_json(stdout)["stop_reason"] == "end_turn"

    def test_stop_reason_none_when_unavailable(self, tmp_path):
        config = make_config(tmp_path)
        agent = make_agent_stub(last_stop_reason=None)
        task = tmp_path / "p.md"
        task.write_text("t")

        stdout, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", str(task)],
            config=config, agent=agent,
        )
        assert code == 0
        assert parse_single_json(stdout)["stop_reason"] is None


# ---------------------------------------------------------------------------
# Tool trace (bounded, names + is_error only)
# ---------------------------------------------------------------------------


class TestExecToolTrace:
    def test_tool_trace_records_names_and_errors(self, tmp_path):
        config = make_config(tmp_path)

        async def hi(text, room_id, *, on_tool_call=None, thinking=None,
                     on_tool_intent=None, callbacks=None, **kw):
            if on_tool_call is not None:
                await on_tool_call("c1", "shell", {}, "ok", False)
                await on_tool_call("c2", "file_read", {}, "no such file", True)
                await on_tool_call("c3", "grep", {}, "hit", False)
            return "done"

        agent = make_agent_stub(handle_input=hi)
        task = tmp_path / "p.md"
        task.write_text("t")

        stdout, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", str(task)],
            config=config, agent=agent,
        )

        assert code == 0
        obj = parse_single_json(stdout)
        names = [e["name"] for e in obj["tool_trace"]]
        assert names == ["shell", "file_read", "grep"]
        errors = [e["is_error"] for e in obj["tool_trace"]]
        assert errors == [False, True, False]

    def test_tool_trace_bounded_at_50(self, tmp_path):
        config = make_config(tmp_path)

        async def hi(text, room_id, *, on_tool_call=None, **kw):
            if on_tool_call is not None:
                for i in range(70):
                    await on_tool_call(f"c{i}", "shell", {}, "ok", False)
            return "done"

        agent = make_agent_stub(handle_input=hi)
        task = tmp_path / "p.md"
        task.write_text("t")

        stdout, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", str(task)],
            config=config, agent=agent,
        )

        assert code == 0
        obj = parse_single_json(stdout)
        assert len(obj["tool_trace"]) == 50


# ---------------------------------------------------------------------------
# Prompt assembly: empty workspace (injection-defense fallback pin)
# ---------------------------------------------------------------------------


class TestExecPromptAssembly:
    def test_empty_workspace_no_identity_leak(self, tmp_path):
        """A workspace with NO identity files must not leak SOUL-family or
        SAFETY content into the worker system prompt. It may only carry
        framework-owned fallback text (the built-in injection-defense footer
        and, at most, the mechanical runtime note) — never operator-owned or
        behavioural content the operator did not place.
        """
        ws = tmp_path / "empty-ws"
        ws.mkdir()
        from openalph.prompt import assemble_prompt

        prompt = assemble_prompt(ws, injection_defense=True)

        # No operator identity files -> none of their section markers appear.
        for marker in ("## SOUL.md", "## SAFETY.md", "## OPERATOR.md",
                       "## WAKE.md", "## ENVIRONMENT.md", "## OPERATIONS.md",
                       "## Skills"):
            assert marker not in prompt, f"leaked section {marker!r}"
        # The only acceptable non-empty content is the built-in footer.
        assert INJECTION_DEFENSE in prompt
        # Strip the known framework-owned fallbacks; nothing operator-owned or
        # behavioural may remain.
        residual = prompt.replace(INJECTION_DEFENSE, "").replace("## Runtime", "")
        residual = residual.replace(str(ws.resolve()), "").strip()
        for token in ("SOUL", "SAFETY", "OPERATOR", "WAKE", "ENVIRONMENT",
                      "OPERATIONS", "Skills"):
            assert token not in residual, f"leaked token {token!r}: {residual!r}"

    def test_empty_workspace_defense_off_leaks_no_identity(self, tmp_path):
        """With injection_defense off and no files, the prompt carries NO
        injection-defense footer and NO identity/behavioural section — only
        at most the mechanical runtime note (which names no agent)."""
        ws = tmp_path / "empty-ws2"
        ws.mkdir()
        from openalph.prompt import assemble_prompt

        prompt = assemble_prompt(ws, injection_defense=False)
        assert INJECTION_DEFENSE not in prompt
        for marker in ("## SOUL.md", "## SAFETY.md", "## OPERATOR.md",
                       "## WAKE.md", "## ENVIRONMENT.md", "## OPERATIONS.md",
                       "## Skills"):
            assert marker not in prompt, f"leaked section {marker!r}"
        # Nothing but the mechanical runtime block (if any).
        residual = prompt.replace("## Runtime", "").replace(str(ws.resolve()), "").strip()
        for token in ("SOUL", "SAFETY", "OPERATOR", "WAKE", "ENVIRONMENT",
                      "OPERATIONS", "Skills", "INJECTION_DEFENSE"):
            assert token not in residual, f"leaked token {token!r}: {residual!r}"
