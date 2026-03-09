"""Tests for the systemd template unit file.

Verifies the openalph@.service template has correct directives for
multi-agent isolation and hardening.
"""

import configparser
import shutil
import subprocess
from pathlib import Path

import pytest

UNIT_PATH = Path(__file__).parent.parent / "etc" / "openalph@.service"


@pytest.fixture
def unit_text():
    return UNIT_PATH.read_text()


@pytest.fixture
def unit_config(unit_text):
    """Parse the unit file as INI (systemd units are INI-like)."""
    cp = configparser.ConfigParser(interpolation=None)
    cp.read_string(unit_text)
    return cp


class TestUnitFileExists:
    def test_file_exists(self):
        assert UNIT_PATH.exists(), f"Missing: {UNIT_PATH}"

    def test_not_empty(self, unit_text):
        assert len(unit_text.strip()) > 50


class TestUnitSections:
    def test_has_unit_section(self, unit_config):
        assert unit_config.has_section("Unit")

    def test_has_service_section(self, unit_config):
        assert unit_config.has_section("Service")

    def test_has_install_section(self, unit_config):
        assert unit_config.has_section("Install")


class TestServiceDirectives:
    def test_user_is_template(self, unit_config):
        assert unit_config["Service"]["User"] == "oa-%i"

    def test_group_is_openalph(self, unit_config):
        assert unit_config["Service"]["Group"] == "openalph"

    def test_working_directory(self, unit_config):
        assert unit_config["Service"]["WorkingDirectory"] == "/home/oa-%i"

    def test_exec_start_references_config(self, unit_config):
        exec_start = unit_config["Service"]["ExecStart"]
        assert "/etc/openalph/agents/%i.toml" in exec_start

    def test_restart_on_failure(self, unit_config):
        assert unit_config["Service"]["Restart"] == "on-failure"

    def test_type_simple(self, unit_config):
        assert unit_config["Service"]["Type"] == "simple"


class TestHardening:
    def test_no_new_privileges(self, unit_config):
        assert unit_config["Service"]["NoNewPrivileges"] == "yes"

    def test_protect_system_strict(self, unit_config):
        assert unit_config["Service"]["ProtectSystem"] == "strict"

    def test_protect_home_tmpfs(self, unit_config):
        """ProtectHome=tmpfs hides all /home, then BindPaths exposes only this agent's."""
        assert unit_config["Service"]["ProtectHome"] == "tmpfs"

    def test_bind_paths_agent_home(self, unit_config):
        bind = unit_config["Service"]["BindPaths"]
        assert "/home/oa-%i" in bind

    def test_bind_readonly_shared(self, unit_config):
        readonly = unit_config["Service"]["BindReadOnlyPaths"]
        assert "/srv/openalph/shared" in readonly

    def test_bind_readonly_config(self, unit_config):
        readonly = unit_config["Service"]["BindReadOnlyPaths"]
        assert "/etc/openalph" in readonly

    def test_private_tmp(self, unit_config):
        assert unit_config["Service"]["PrivateTmp"] == "yes"


class TestInstall:
    def test_wanted_by_multi_user(self, unit_config):
        assert unit_config["Install"]["WantedBy"] == "multi-user.target"


class TestSystemdAnalyze:
    """Run systemd-analyze verify if available (not required in CI)."""

    @pytest.mark.skipif(
        not shutil.which("systemd-analyze"),
        reason="systemd-analyze not available",
    )
    def test_unit_verifies(self):
        # systemd-analyze verify exits 0 for valid units
        # Note: template units with %i may produce warnings but should not error
        result = subprocess.run(
            ["systemd-analyze", "verify", "--man=no", str(UNIT_PATH)],
            capture_output=True,
            text=True,
        )
        # Allow warnings (exit code 0 or specific known warnings about %i)
        # The key check: no hard parse errors
        assert "Failed to parse" not in result.stderr
