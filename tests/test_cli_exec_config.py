"""kdsn.320 — `openalph exec --config <path>` + `list --all` headless section.

Context: headless/station agent TOMLs (Stigmergy decomposer/worker,
eval-harness eval agent) live in per-project SUBDIRS of CONFIG_DIR
(/etc/openalph/agents/stigmergy/..., /etc/openalph/agents/eval-harness/...).
The top-level *.toml glob is the systemd blast radius of
`openalph list` / `start|stop|restart all` and must stay top-level-only
(the retired/ convention, 2026-08-28). Headless callers (stigmergy
stations, the eval-harness Harbor adapter) address their configs by
explicit path: `openalph exec --config <path>`, mutually exclusive with
`--agent` (exactly one required). `openalph list --all` surfaces the
headless/station configs for visibility; DEFAULT `list` output is
byte-identical to before (agent-retirement.md pipes
`openalph list | grep -c <name>`).

Written RED-first (kdsn.320, 2026-09-03). The suite below is the spec;
the implementation lives in src/openalph/cli.py (the only file it may
touch).
"""

from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from openalph.cli import cmd_list, cmd_restart, list_agents, main, parse_args
from test_cli_exec import make_agent_stub, make_config, parse_single_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_task(tmp_path: Path, text: str = "Do the thing.") -> Path:
    task = tmp_path / "task.md"
    task.write_text(text, encoding="utf-8")
    return task


def run_exec(argv, *, config, agent, real_loader=False):
    """Run main() over `argv` (an exec invocation) with the loader and Agent
    stubbed. Returns (stdout, stderr, exit_code, name_resolver_calls,
    path_loader_calls).

    - real_loader=True: the REAL openalph.config.load_config is used
      (for the missing-file fail-loud test); otherwise it is stubbed to
      return `config` and record its path argument.
    - name_resolver_calls: records every load_agent_config(name) call —
      must be empty for --config invocations.
    - path_loader_calls: records every load_config(path) call — must be
      empty for --agent invocations.
    """
    out, err = StringIO(), StringIO()
    name_resolver_calls, path_loader_calls = [], []

    if real_loader:
        load_config_cm = _nullcontext()
    else:
        load_config_cm = patch(
            "openalph.cli.load_config",
            side_effect=lambda p: path_loader_calls.append(p) or config,
            create=True,
        )

    with (
        load_config_cm,
        patch(
            "openalph.cli.load_agent_config",
            side_effect=lambda name: name_resolver_calls.append(name) or config,
        ),
        patch("openalph.agent.Agent", return_value=agent),
        redirect_stdout(out),
        redirect_stderr(err),
    ):
        with pytest.raises(SystemExit) as exc_info:
            main(argv)
    return out.getvalue(), err.getvalue(), exc_info.value.code, (
        name_resolver_calls,
        path_loader_calls,
    )


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _make_config_dir(tmp_path: Path, monkeypatch, top=()) -> Path:
    """A CONFIG_DIR stand-in with the given top-level toml stems."""
    config_dir = tmp_path / "agents"
    config_dir.mkdir()
    for name in top:
        (config_dir / f"{name}.toml").write_text("")
    monkeypatch.setattr("openalph.cli.CONFIG_DIR", config_dir)
    return config_dir


# ---------------------------------------------------------------------------
# argparse surface
# ---------------------------------------------------------------------------


class TestExecConfigArgparse:
    def test_config_flag_parses(self):
        args = parse_args(
            ["exec",
             "--config",
             "/etc/openalph/agents/stigmergy/stigmergy-decomposer.toml",
             "--task-file", "t.md"],
        )
        assert args.config == (
            "/etc/openalph/agents/stigmergy/stigmergy-decomposer.toml"
        )
        assert args.agent is None

    def test_agent_flag_parses(self):
        args = parse_args(["exec", "--agent", "watson", "--task-file", "t.md"])
        assert args.agent == "watson"
        assert args.config is None

    def test_both_flags_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(
                ["exec", "--agent", "watson", "--config", "/x.toml",
                 "--task-file", "t.md"],
            )

    def test_neither_flag_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(["exec", "--task-file", "t.md"])


# ---------------------------------------------------------------------------
# cmd_exec config resolution
# ---------------------------------------------------------------------------


class TestExecConfigResolution:
    def test_config_loads_via_path_not_name(self, tmp_path):
        cfg_dir = tmp_path / "agents" / "stigmergy"
        cfg_dir.mkdir(parents=True)
        cfg_path = cfg_dir / "stigmergy-decomposer.toml"
        cfg_path.write_text("")
        task = _write_task(tmp_path)
        config = make_config(tmp_path)
        agent = make_agent_stub()

        out, err, code, (name_calls, path_calls) = run_exec(
            ["exec", "--config", str(cfg_path), "--task-file", str(task)],
            config=config,
            agent=agent,
        )
        assert code == 0
        assert name_calls == [], "--config must not route via the name resolver"
        assert [Path(p) for p in path_calls] == [cfg_path]
        result = parse_single_json(out)
        assert result["status"] == "done"

    def test_config_missing_file_fails_loud(self, tmp_path):
        # REAL loader: missing path -> ConfigError -> exit 1, stderr names
        # the path, stdout stays empty (stdout discipline: the Stigmergy
        # driver parses stdout as JSON, a traceback byte would corrupt it).
        task = _write_task(tmp_path)
        missing = tmp_path / "nope" / "agent.toml"
        agent = make_agent_stub()

        out, err, code, _ = run_exec(
            ["exec", "--config", str(missing), "--task-file", str(task)],
            config=make_config(tmp_path),
            agent=agent,
            real_loader=True,
        )
        assert code == 1
        assert "Config error" in err
        assert str(missing) in err
        assert out == ""

    def test_config_relative_path_passthrough(self, tmp_path):
        # The path is passed to the loader AS GIVEN (relative stays
        # relative; CWD resolution is the loader's job, not the CLI's).
        task = _write_task(tmp_path)
        config = make_config(tmp_path)
        agent = make_agent_stub()

        out, err, code, (name_calls, path_calls) = run_exec(
            ["exec", "--config", "sub/agent.toml", "--task-file", str(task)],
            config=config,
            agent=agent,
        )
        assert code == 0
        assert name_calls == []
        assert [str(p) for p in path_calls] == ["sub/agent.toml"]

    def test_agent_flag_does_not_use_path_loader(self, tmp_path):
        # Symmetric pin: --agent routes through the name resolver ONLY —
        # the pre-kdsn.320 behavior, preserved byte-for-byte for the
        # in-cage worker driver (`--agent stigmergy-worker`).
        task = _write_task(tmp_path)
        config = make_config(tmp_path)
        agent = make_agent_stub()

        out, err, code, (name_calls, path_calls) = run_exec(
            ["exec", "--agent", "watson", "--task-file", str(task)],
            config=config,
            agent=agent,
        )
        assert code == 0
        assert name_calls == ["watson"]
        assert path_calls == []


# ---------------------------------------------------------------------------
# list --all (headless/station section)
# ---------------------------------------------------------------------------


class TestListHeadlessSection:
    def test_default_list_byte_unchanged_with_subdirs(self, tmp_path, monkeypatch, capsys):
        d = _make_config_dir(tmp_path, monkeypatch, top=("watson", "babson"))
        (d / "stigmergy").mkdir()
        (d / "stigmergy" / "stigmergy-decomposer.toml").write_text("")
        cmd_list(Namespace(all=False))
        # Byte-exact pin: subdir tomls never leak into default output
        # (agent-retirement.md pipes `openalph list | grep -c <name>`).
        assert capsys.readouterr().out == "babson\nwatson\n"

    def test_list_all_shows_headless_section(self, tmp_path, monkeypatch, capsys):
        d = _make_config_dir(tmp_path, monkeypatch, top=("watson", "babson"))
        (d / "stigmergy").mkdir()
        (d / "stigmergy" / "stigmergy-decomposer.toml").write_text("")
        (d / "stigmergy" / "stigmergy-worker.toml").write_text("")
        (d / "eval-harness").mkdir()
        (d / "eval-harness" / "evals.toml").write_text("")
        cmd_list(Namespace(all=True))
        lines = capsys.readouterr().out.splitlines()
        # Top section first, byte-identical to default output.
        assert lines[:2] == ["babson", "watson"]
        # Then: blank, a #-prefixed header, then sorted subdir/name lines.
        assert lines[2] == ""
        assert lines[3].startswith("#")
        assert lines[4:] == [
            "eval-harness/evals",
            "stigmergy/stigmergy-decomposer",
            "stigmergy/stigmergy-worker",
        ]

    def test_list_all_excludes_retired(self, tmp_path, monkeypatch, capsys):
        d = _make_config_dir(tmp_path, monkeypatch, top=("watson",))
        (d / "retired").mkdir()
        (d / "retired" / "oldagent.toml").write_text("")
        cmd_list(Namespace(all=True))
        # retired/ is the retirement convention — dead agents, not active
        # headless configs. No section at all when nothing else qualifies.
        assert capsys.readouterr().out == "watson\n"

    def test_list_all_one_level_only(self, tmp_path, monkeypatch, capsys):
        d = _make_config_dir(tmp_path, monkeypatch, top=("watson",))
        (d / "a" / "b").mkdir(parents=True)
        (d / "a" / "b" / "deep.toml").write_text("")
        cmd_list(Namespace(all=True))
        assert capsys.readouterr().out == "watson\n"

    def test_list_all_empty_subdir_no_section(self, tmp_path, monkeypatch, capsys):
        d = _make_config_dir(tmp_path, monkeypatch, top=("watson",))
        (d / "stigmergy").mkdir()  # empty subdir
        (d / "eval-harness").mkdir()
        (d / "eval-harness" / "notes.txt").write_text("")  # non-toml
        cmd_list(Namespace(all=True))
        assert capsys.readouterr().out == "watson\n"

    def test_list_all_no_top_level_agents(self, tmp_path, monkeypatch, capsys):
        d = _make_config_dir(tmp_path, monkeypatch, top=())
        (d / "eval-harness").mkdir()
        (d / "eval-harness" / "evals.toml").write_text("")
        cmd_list(Namespace(all=True))
        out = capsys.readouterr().out
        assert "No agents configured." in out
        assert "eval-harness/evals" in out

    def test_list_agents_excludes_subdirs(self, tmp_path, monkeypatch):
        # BLAST-RADIUS PIN: list_agents() — the enumeration behind
        # start/stop/restart 'all' — stays top-level-only no matter what
        # subdirs exist.
        d = _make_config_dir(tmp_path, monkeypatch, top=("watson", "babson"))
        (d / "stigmergy").mkdir()
        (d / "stigmergy" / "stigmergy-decomposer.toml").write_text("")
        (d / "eval-harness").mkdir()
        (d / "eval-harness" / "evals.toml").write_text("")
        assert set(list_agents()) == {"watson", "babson"}

    def test_restart_all_excludes_subdir_tomls(self, tmp_path, monkeypatch):
        # BLAST-RADIUS PIN, end-to-end: `restart all` with headless tomls
        # present restarts ONLY top-level agents.
        d = _make_config_dir(tmp_path, monkeypatch, top=("watson", "babson"))
        (d / "eval-harness").mkdir()
        (d / "eval-harness" / "evals.toml").write_text("")
        with patch("openalph.cli.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            cmd_restart(Namespace(agent="all"))
        services = [
            c.args[0][-1] for c in mock_run.call_args_list
            if c.args and c.args[0][:1] == ["systemctl"]
        ]
        assert sorted(services) == [
            "openalph@babson.service",
            "openalph@watson.service",
        ]
