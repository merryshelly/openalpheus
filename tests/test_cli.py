"""Tests for openalph.cli module.

CLI subcommands: start, stop, restart, status, list, logs, new-agent, run, monitor.
Systemctl/journalctl calls are mocked. Config discovery uses monkeypatched CONFIG_DIR.
"""

import subprocess
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

from openalph.cli import (
    parse_args,
    main,
    cmd_start,
    cmd_stop,
    cmd_restart,
    cmd_status,
    cmd_list,
    cmd_logs,
    cmd_new_agent,
    cmd_run,
    list_agents,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_start(self):
        args = parse_args(["start", "watson"])
        assert args.command == "start"
        assert args.agent == "watson"

    def test_stop(self):
        args = parse_args(["stop", "watson"])
        assert args.command == "stop"
        assert args.agent == "watson"

    def test_restart(self):
        args = parse_args(["restart", "watson"])
        assert args.command == "restart"
        assert args.agent == "watson"

    def test_status_no_agent(self):
        args = parse_args(["status"])
        assert args.command == "status"
        assert args.agent is None

    def test_status_with_agent(self):
        args = parse_args(["status", "watson"])
        assert args.command == "status"
        assert args.agent == "watson"

    def test_list(self):
        args = parse_args(["list"])
        assert args.command == "list"

    def test_logs(self):
        args = parse_args(["logs", "watson"])
        assert args.command == "logs"
        assert args.agent == "watson"

    def test_logs_follow(self):
        args = parse_args(["logs", "watson", "-f"])
        assert args.command == "logs"
        assert args.follow is True

    def test_logs_follow_default_false(self):
        args = parse_args(["logs", "watson"])
        assert args.follow is False

    def test_new_agent(self):
        args = parse_args(["new-agent", "watson"])
        assert args.command == "new-agent"
        assert args.name == "watson"

    def test_new_agent_dry_run(self):
        args = parse_args(["new-agent", "watson", "--dry-run"])
        assert args.dry_run is True

    def test_new_agent_dry_run_default_false(self):
        args = parse_args(["new-agent", "watson"])
        assert args.dry_run is False

    def test_run(self):
        args = parse_args(["run", "watson"])
        assert args.command == "run"
        assert args.agent == "watson"

    def test_run_verbose(self):
        args = parse_args(["run", "watson", "-v"])
        assert args.verbose is True

    def test_monitor(self):
        args = parse_args(["monitor", "watson"])
        assert args.command == "monitor"
        assert args.agent == "watson"

    def test_no_args_exits(self):
        with pytest.raises(SystemExit):
            parse_args([])

    def test_global_verbose_flag(self):
        args = parse_args(["-v", "list"])
        assert args.verbose is True

    def test_verbose_default_false(self):
        args = parse_args(["list"])
        assert args.verbose is False


# ---------------------------------------------------------------------------
# cmd_start / cmd_stop / cmd_restart
# ---------------------------------------------------------------------------


class TestCmdStart:
    @patch("openalph.cli.subprocess.run")
    def test_calls_systemctl(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        cmd_start(Namespace(agent="watson"))
        mock_run.assert_called_once_with(
            ["systemctl", "start", "openalph@watson.service"],
            check=True,
        )


class TestCmdStop:
    @patch("openalph.cli.subprocess.run")
    def test_calls_systemctl(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        cmd_stop(Namespace(agent="watson"))
        mock_run.assert_called_once_with(
            ["systemctl", "stop", "openalph@watson.service"],
            check=True,
        )


class TestCmdRestart:
    @patch("openalph.cli.subprocess.run")
    def test_calls_systemctl(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        cmd_restart(Namespace(agent="watson"))
        mock_run.assert_called_once_with(
            ["systemctl", "restart", "openalph@watson.service"],
            check=True,
        )


# ---------------------------------------------------------------------------
# cmd_logs
# ---------------------------------------------------------------------------


class TestCmdLogs:
    @patch("openalph.cli.subprocess.run")
    def test_calls_journalctl(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        cmd_logs(Namespace(agent="watson", follow=False))
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "journalctl" in args
        assert "openalph@watson.service" in " ".join(args)

    @patch("openalph.cli.subprocess.run")
    def test_follow_flag(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        cmd_logs(Namespace(agent="watson", follow=True))
        args = mock_run.call_args[0][0]
        assert "-f" in args


# ---------------------------------------------------------------------------
# cmd_new_agent
# ---------------------------------------------------------------------------


class TestCmdNewAgent:
    @patch("openalph.cli.create_agent")
    def test_calls_create_agent(self, mock_create):
        mock_create.return_value = []
        cmd_new_agent(Namespace(name="watson", dry_run=False, force=False))
        mock_create.assert_called_once_with("watson", dry_run=False, force=False)

    @patch("openalph.cli.create_agent")
    def test_dry_run(self, mock_create):
        mock_create.return_value = []
        cmd_new_agent(Namespace(name="watson", dry_run=True, force=False))
        mock_create.assert_called_once_with("watson", dry_run=True, force=False)


# ---------------------------------------------------------------------------
# cmd_status
# ---------------------------------------------------------------------------


class TestCmdStatus:
    @patch("openalph.cli.subprocess.run")
    def test_with_agent(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="active")
        cmd_status(Namespace(agent="watson"))
        args = mock_run.call_args[0][0]
        assert "systemctl" in args
        assert "openalph@watson.service" in " ".join(args)

    @patch("openalph.cli.list_agents", return_value=["watson", "babson"])
    @patch("openalph.cli.subprocess.run")
    def test_without_agent_shows_all(self, mock_run, mock_list):
        mock_run.return_value = MagicMock(returncode=0, stdout="active")
        cmd_status(Namespace(agent=None))
        # Should query status for each agent
        assert mock_run.call_count >= 2


# ---------------------------------------------------------------------------
# list_agents
# ---------------------------------------------------------------------------


class TestListAgents:
    def test_lists_toml_files(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "agents"
        config_dir.mkdir()
        (config_dir / "watson.toml").write_text("")
        (config_dir / "babson.toml").write_text("")
        (config_dir / "notes.txt").write_text("")
        monkeypatch.setattr("openalph.cli.CONFIG_DIR", config_dir)
        agents = list_agents()
        assert set(agents) == {"watson", "babson"}

    def test_empty_dir(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "agents"
        config_dir.mkdir()
        monkeypatch.setattr("openalph.cli.CONFIG_DIR", config_dir)
        assert list_agents() == []

    def test_missing_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("openalph.cli.CONFIG_DIR", tmp_path / "nonexistent")
        assert list_agents() == []


# ---------------------------------------------------------------------------
# cmd_list
# ---------------------------------------------------------------------------


class TestCmdList:
    @patch("openalph.cli.list_agents", return_value=["watson", "babson"])
    def test_prints_agents(self, mock_list, capsys):
        cmd_list(Namespace())
        output = capsys.readouterr().out
        assert "watson" in output
        assert "babson" in output

    @patch("openalph.cli.list_agents", return_value=[])
    def test_empty(self, mock_list, capsys):
        cmd_list(Namespace())
        output = capsys.readouterr().out
        assert "no agents" in output.lower() or output.strip() == ""


# ---------------------------------------------------------------------------
# main() dispatch
# ---------------------------------------------------------------------------


class TestMainDispatch:
    @patch("openalph.cli.cmd_start")
    def test_dispatches_start(self, mock_cmd):
        main(["start", "watson"])
        mock_cmd.assert_called_once()

    @patch("openalph.cli.cmd_stop")
    def test_dispatches_stop(self, mock_cmd):
        main(["stop", "watson"])
        mock_cmd.assert_called_once()

    @patch("openalph.cli.cmd_list")
    def test_dispatches_list(self, mock_cmd):
        main(["list"])
        mock_cmd.assert_called_once()

    @patch("openalph.cli.cmd_new_agent")
    def test_dispatches_new_agent(self, mock_cmd):
        main(["new-agent", "test"])
        mock_cmd.assert_called_once()

    @patch("openalph.cli.cmd_run")
    def test_dispatches_run(self, mock_cmd):
        main(["run", "watson"])
        mock_cmd.assert_called_once()

    def test_returns_zero_on_success(self):
        with patch("openalph.cli.cmd_list"):
            result = main(["list"])
            assert result == 0
