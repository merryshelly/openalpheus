"""Tests for `openalph exec --system-prompt-file` (station prompt artifact
delivery).

Extends the `exec` one-shot subcommand with `--system-prompt-file PATH`:
the file's contents become the worker's system prompt, BYTE-FAITHFUL —
exactly the file text, nothing prepended (an unhashed preamble silently
breaks prompt-artifact hash provenance) — plus the same injection-defense
footer the empty-workspace prompt assembly path gets today (the built-in
`INJECTION_DEFENSE` fallback; a workspace `SECURITY_FOOTER.md` would win if
present), unless the agent config has `injection_defense = false` (no
footer at all).

Stubs the Agent symbol exactly like test_cli_exec.py (patch
openalph.agent.Agent) and inspects the system_prompt attribute that
cmd_exec assigns post-construction — the same seam the real Agent reads
(self.system_prompt) when it builds the chat `system` field.

Exit codes: 0 done / 1 failed / 2 infra / 3 wedged. Stdout is EXACTLY one
JSON line; everything else goes to stderr.
"""

import json
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from openalph.config import AgentConfig, ProviderConfig
from openalph.prompt import INJECTION_DEFENSE, SECURITY_FOOTER_FILENAME


# ---------------------------------------------------------------------------
# Helpers (mirroring tests/test_cli_exec.py)
# ---------------------------------------------------------------------------


def make_config(tmp_path, *, name="exec-sysprompt-agent",
                default_model="p/model", max_iterations=100,
                injection_defense=True) -> AgentConfig:
    """Build a REAL AgentConfig (not a MagicMock) so the post-construction
    override and the config's injection_defense flag are exercised for real."""
    provider = ProviderConfig(key="p", type="openai", api_key="sk-test",
                              base_url="http://127.0.0.1:18081")
    return AgentConfig(
        name=name,
        default_model=default_model,
        max_tokens=8192,
        providers={"p": provider},
        workspace=Path(tmp_path),
        max_iterations=max_iterations,
        injection_defense=injection_defense,
    )


def make_agent_stub(*, response_text="FINAL TEXT",
                    last_turn_usage=None, last_stop_reason=None):
    """Build a MagicMock shaped like the real Agent for the bits exec reads.

    `system_prompt` is a plain settable attribute, exactly like the real
    Agent's (agent.py sets it in __init__; cmd_exec overrides it when
    --system-prompt-file is given). The test reads it back after run_exec.
    """
    from unittest.mock import AsyncMock, MagicMock

    agent = MagicMock()
    agent.last_turn_usage = MagicMock(return_value=last_turn_usage)
    agent.last_stop_reason = MagicMock(return_value=last_stop_reason)
    agent.history = MagicMock(return_value=[])
    agent.tools = []
    agent.system_prompt = "sentinel: built-in agent system prompt"
    agent.handle_input = AsyncMock(return_value=response_text)
    return agent


def run_exec(argv, *, config, agent, stdin=None):
    """Run main() over `argv` with load_agent_config + Agent patched.

    Returns (stdout_str, stderr_str, exit_code). The exec surface must
    sys.exit with the numeric code, so SystemExit is expected here.
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
    """Assert `stdout` is EXACTLY one JSON object (nothing else) and return it."""
    stripped = stdout.strip()
    assert stripped, "stdout must carry exactly one JSON line (got empty)"
    obj, end = json.JSONDecoder().raw_decode(stripped)
    assert stripped[end:].strip() == "", (
        f"stdout carries content AFTER the JSON object: {stripped[end:]!r}"
    )
    assert isinstance(obj, dict), f"stdout is not a JSON object: {type(obj)!r}"
    return obj


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class TestExecSystemPromptParseArgs:
    def test_flag_parses_to_path(self):
        from openalph.cli import parse_args

        args = parse_args([
            "exec", "--agent", "w", "--task-file", "p",
            "--system-prompt-file", "/artifacts/station-prompt.md",
        ])
        assert args.system_prompt_file == "/artifacts/station-prompt.md"

    def test_flag_defaults_to_none(self):
        from openalph.cli import parse_args

        args = parse_args(["exec", "--agent", "w", "--task-file", "p"])
        assert args.system_prompt_file is None


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestExecSystemPromptHappyPath:
    def test_artifact_plus_fallback_footer_exit0_one_json_line(self, tmp_path):
        """--system-prompt-file → exit 0, one JSON line, and the agent's
        system prompt is EXACTLY the artifact text (byte-for-byte prefix,
        nothing prepended) plus the built-in injection-defense footer —
        what the empty-workspace assembly returns today."""
        ws = tmp_path / "ws"
        ws.mkdir()
        config = make_config(ws, name="exec-sysprompt-agent")

        artifact = "STATION PROMPT: dispatch unit 7\nact on task, report back."
        spf = tmp_path / "station-prompt.md"
        spf.write_text(artifact)
        task = tmp_path / "task.md"
        task.write_text("the task")
        agent = make_agent_stub(
            last_turn_usage={"input_tokens": 1, "cache_read_tokens": 0,
                             "output_tokens": 1, "cache_creation_tokens": 0},
            last_stop_reason="end_turn",
        )

        stdout, stderr, code = run_exec(
            ["exec", "--agent", "w", "--task-file", str(task),
             "--system-prompt-file", str(spf)],
            config=config, agent=agent,
        )

        assert code == 0
        obj = parse_single_json(stdout)
        assert obj["status"] == "done"
        # The load-bearing assertions: byte-faithful prefix + the SAME footer
        # the empty-workspace path appends (built-in fallback).
        assert agent.system_prompt.startswith(artifact)
        assert agent.system_prompt == artifact + INJECTION_DEFENSE
        agent.handle_input.assert_awaited_once()

    def test_no_flag_leaves_built_prompt_untouched(self, tmp_path):
        """Without --system-prompt-file the agent keeps the system prompt
        Agent.__init__ assembled (the stub's sentinel stands in for it)."""
        config = make_config(tmp_path)
        task = tmp_path / "t.md"
        task.write_text("t")
        agent = make_agent_stub()

        stdout, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", str(task)],
            config=config, agent=agent,
        )

        assert code == 0
        assert agent.system_prompt == "sentinel: built-in agent system prompt"


# ---------------------------------------------------------------------------
# Byte-faithfulness
# ---------------------------------------------------------------------------


class TestExecSystemPromptByteFaithful:
    def test_distinctive_whitespace_and_unicode_survive(self, tmp_path):
        """No stripped trailing newline, no added preamble, no encoding
        mangling: the artifact is a byte-for-byte prefix of the result."""
        config = make_config(tmp_path)
        artifact = ("STATION PROMPT\n\n"
                    "  indented rule\t(tab kept)\n"
                    "unicode: é ü ñ 中文 🚀\r\n"
                    "trailing blank lines below\n"
                    "\n"
                    "\n")
        spf = tmp_path / "artifact.md"
        spf.write_text(artifact)
        task = tmp_path / "t.md"
        task.write_text("t")
        agent = make_agent_stub()

        stdout, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", str(task),
             "--system-prompt-file", str(spf)],
            config=config, agent=agent,
        )

        assert code == 0
        assert agent.system_prompt.startswith(artifact), (
            "artifact text is not a byte-for-byte prefix — "
            "stripped newline or injected preamble"
        )
        # No preamble: the very first character of the system prompt is the
        # first character of the artifact.
        assert agent.system_prompt[0] == "S"
        assert agent.system_prompt == artifact + INJECTION_DEFENSE

    def test_empty_file_yields_footer_only(self, tmp_path):
        """A zero-byte artifact is still byte-faithful: the system prompt is
        exactly the footer (and starts with the empty artifact trivially)."""
        config = make_config(tmp_path)
        spf = tmp_path / "empty-artifact.md"
        spf.write_text("")
        task = tmp_path / "t.md"
        task.write_text("t")
        agent = make_agent_stub()

        stdout, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", str(task),
             "--system-prompt-file", str(spf)],
            config=config, agent=agent,
        )

        assert code == 0
        assert agent.system_prompt == INJECTION_DEFENSE


# ---------------------------------------------------------------------------
# Footer resolution (injection_defense flag + workspace file seam)
# ---------------------------------------------------------------------------


class TestExecSystemPromptFooter:
    def test_injection_defense_false_no_footer(self, tmp_path):
        """[agent] injection_defense = false → NO footer appended: the
        system prompt is exactly the artifact text, byte-for-byte."""
        config = make_config(tmp_path, injection_defense=False)
        artifact = "STATION PROMPT (defense off)\nno footer expected."
        spf = tmp_path / "artifact.md"
        spf.write_text(artifact)
        task = tmp_path / "t.md"
        task.write_text("t")
        agent = make_agent_stub()

        stdout, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", str(task),
             "--system-prompt-file", str(spf)],
            config=config, agent=agent,
        )

        assert code == 0
        assert agent.system_prompt == artifact

    def test_workspace_security_footer_file_wins_over_fallback(self, tmp_path):
        """If the agent workspace carries SECURITY_FOOTER.md, that file's
        text is the footer (the same workspace-file-first resolution
        assemble_prompt uses), not the built-in constant."""
        config = make_config(tmp_path)
        custom_footer = ("## CUSTOM WORKSPACE SECURITY FOOTER\n"
                         "operator-edited rules.")
        (tmp_path / SECURITY_FOOTER_FILENAME).write_text(custom_footer)
        artifact = "STATION PROMPT (custom footer ws)"
        spf = tmp_path / "artifact.md"
        spf.write_text(artifact)
        task = tmp_path / "t.md"
        task.write_text("t")
        agent = make_agent_stub()

        stdout, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", str(task),
             "--system-prompt-file", str(spf)],
            config=config, agent=agent,
        )

        assert code == 0
        assert agent.system_prompt.startswith(artifact)
        assert agent.system_prompt == artifact + custom_footer
        assert INJECTION_DEFENSE not in agent.system_prompt


# ---------------------------------------------------------------------------
# Missing / unreadable file → fail loud
# ---------------------------------------------------------------------------


class TestExecSystemPromptMissingFile:
    def test_missing_file_exits1_stderr_empty_stdout(self, tmp_path):
        config = make_config(tmp_path)
        agent = make_agent_stub()
        task = tmp_path / "t.md"
        task.write_text("t")
        spf = tmp_path / "no-such-prompt.md"

        stdout, stderr, code = run_exec(
            ["exec", "--agent", "w", "--task-file", str(task),
             "--system-prompt-file", str(spf)],
            config=config, agent=agent,
        )

        assert code == 1
        assert stdout.strip() == ""  # no partial JSON on failure
        assert "no-such-prompt.md" in stderr
        # Agent is never constructed; handle_input never runs.
        agent.handle_input.assert_not_awaited()
        # The built-in (stub) prompt must be untouched — no partial override.
        assert agent.system_prompt == "sentinel: built-in agent system prompt"

    def test_path_is_directory_exits1_empty_stdout(self, tmp_path):
        """Unreadable-as-file (a directory) also fails loud."""
        config = make_config(tmp_path)
        agent = make_agent_stub()
        task = tmp_path / "t.md"
        task.write_text("t")
        spf = tmp_path / "dir-not-file"
        spf.mkdir()

        stdout, stderr, code = run_exec(
            ["exec", "--agent", "w", "--task-file", str(task),
             "--system-prompt-file", str(spf)],
            config=config, agent=agent,
        )

        assert code == 1
        assert stdout.strip() == ""
        assert "dir-not-file" in stderr
        agent.handle_input.assert_not_awaited()

    def test_non_utf8_exits1_stderr_empty_stdout(self, tmp_path):
        """An undecodable (non-UTF-8) artifact is 'unreadable' as a text
        prompt → fail loud, exit 1, empty stdout (NOT misclassified as a
        failed turn with JSON on stdout)."""
        config = make_config(tmp_path)
        agent = make_agent_stub()
        task = tmp_path / "t.md"
        task.write_text("t")
        spf = tmp_path / "bad-encoding.bin"
        # 0xFF 0xFE 0x01 is not valid UTF-8.
        spf.write_bytes(b"\xff\xfe\x01")

        stdout, stderr, code = run_exec(
            ["exec", "--agent", "w", "--task-file", str(task),
             "--system-prompt-file", str(spf)],
            config=config, agent=agent,
        )

        assert code == 1
        assert stdout.strip() == ""  # no partial JSON on failure
        assert "bad-encoding.bin" in stderr
        agent.handle_input.assert_not_awaited()
        assert agent.system_prompt == "sentinel: built-in agent system prompt"


# ---------------------------------------------------------------------------
# Interaction with existing flags
# ---------------------------------------------------------------------------


class TestExecSystemPromptWithOtherFlags:
    def test_artifact_with_tools_effort_room_maxturns(self, tmp_path):
        """--system-prompt-file + --tools + --effort + --room + --max-turns:
        the artifact still lands byte-for-byte (+footer), and every other
        flag keeps its existing behaviour (no interference)."""
        from openalph.tools import ToolDef

        config = make_config(tmp_path, max_iterations=100)

        agent = make_agent_stub()
        agent.tools = [
            ToolDef(name="shell", description="s", parameters={}, config={}),
            ToolDef(name="file_read", description="f", parameters={}, config={}),
        ]
        built = {}

        def _capture(cfg):
            built["cfg"] = cfg
            return agent

        artifact = "STATION PROMPT (flags combined)"
        spf = tmp_path / "artifact.md"
        spf.write_text(artifact)
        task = tmp_path / "t.md"
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
                      "--system-prompt-file", str(spf),
                      "--tools", "shell,file_read",
                      "--effort", "medium", "--max-turns", "7",
                      "--room", "dispatch-7"])

        assert exc_info.value.code == 0
        obj = parse_single_json(out.getvalue())
        assert obj["status"] == "done"
        # Artifact + footer, byte-for-byte prefix.
        assert agent.system_prompt == artifact + INJECTION_DEFENSE
        # --max-turns replaced max_iterations on the PRE-construction config.
        assert built["cfg"].max_iterations == 7
        # --room used as the room_id arg.
        assert agent.handle_input.call_args.args[1] == "dispatch-7"
        # --effort mapped to thinking.
        assert agent.handle_input.call_args.kwargs.get("thinking") == "medium"
        # --tools installed the resolved ToolDefs (exec's own seam).
        assert [t.name for t in agent.tools] == ["shell", "file_read"]
