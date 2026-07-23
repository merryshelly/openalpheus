"""Tests for the systemd template unit file.

Verifies the openalph@.service template has correct directives for
multi-agent isolation and hardening.

ARCH-3/BUG-1: this file used to validate `etc/openalph@.service` while
`install.sh` deployed a DIFFERENT unit from an inline heredoc -- so the unit
actually installed on operator machines was untested, and the one under test
was broken (`test_exec_start_references_config` asserted the crash-looping
ExecStart, locking the bug in). There is now a single canonical unit shipped
as package data; `etc/openalph@.service` is a symlink to it and `install.sh`
copies it from the installed package. `TestSingleSourceOfTruth` below guards
against the three-way divergence coming back.
"""

import configparser
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
UNIT_PATH = REPO_ROOT / "etc" / "openalph@.service"
CANONICAL_UNIT_PATH = REPO_ROOT / "src" / "openalph" / "data" / "openalph@.service"
INSTALL_SH = REPO_ROOT / "install.sh"


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

    def test_exec_start_invokes_run_subcommand(self, unit_config):
        """BUG-1: the unit must call the CLI with its required subcommand.

        `ExecStart=/usr/local/bin/openalph /etc/openalph/agents/%i.toml`
        passes a bare path and no subcommand. `cli.py` sets
        `sub.required = True`, so argparse exits 2 before doing any work and
        `Restart=on-failure` turns that into a permanent restart loop. The
        previous version of this test asserted the broken form, so the suite
        enforced the bug.
        """
        exec_start = unit_config["Service"]["ExecStart"]
        assert exec_start == "/usr/local/bin/openalph run %i", exec_start

    def test_exec_start_does_not_pass_bare_config_path(self, unit_config):
        """Regression guard for the exact crash-looping form."""
        exec_start = unit_config["Service"]["ExecStart"]
        assert "/etc/openalph/agents/%i.toml" not in exec_start

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

    def test_shared_dir_is_writable(self, unit_config):
        """ARCH-3: the shared dir is created 2770 group-writable by design.

        The repo unit bound it READ-ONLY while the installed unit bound it
        read-write -- one of the three-way divergences. Read-only contradicts
        the design, so the canonical unit binds it read-write.
        """
        assert "/srv/openalph/shared" in unit_config["Service"]["BindPaths"]
        assert "/srv/openalph/shared" not in unit_config["Service"].get(
            "BindReadOnlyPaths", ""
        )

    def test_binds_only_own_config(self, unit_config):
        """SEC-11: the agent must see ONLY its own config, not the whole dir.

        Binding all of /etc/openalph let one agent read another's config (and
        any inline api_key). The agents dir is now masked with a tmpfs and only
        this instance's %i.toml is bound in.
        """
        readonly = unit_config["Service"]["BindReadOnlyPaths"]
        assert "/etc/openalph/agents/%i.toml" in readonly
        # the whole-dir bind must be gone
        assert "/etc/openalph " not in readonly and not readonly.strip().startswith("/etc/openalph ")

    def test_masks_sibling_configs_with_tmpfs(self, unit_config):
        assert unit_config["Service"]["TemporaryFileSystem"] == "/etc/openalph/agents"

    def test_bind_readonly_venv(self, unit_config):
        """The venv the ExecStart binary resolves into must be mounted."""
        assert "/opt/openalph-venv" in unit_config["Service"]["BindReadOnlyPaths"]

    def test_sec10_hardening_directives_present(self, unit_config):
        """SEC-10: the isolation set a shell-running service should carry."""
        svc = unit_config["Service"]
        assert svc["CapabilityBoundingSet"] == ""
        assert svc["RestrictNamespaces"] == "yes"
        assert svc["RestrictSUIDSGID"] == "yes"
        assert svc["LockPersonality"] == "yes"
        assert svc["PrivateDevices"] == "yes"
        assert svc["ProtectProc"] == "invisible"
        assert "@system-service" in svc["SystemCallFilter"]
        assert "AF_INET" in svc["RestrictAddressFamilies"]

    def test_memory_deny_write_execute_deliberately_absent(self, unit_config):
        """MDWX breaks the agent's Node/Python tools (JIT); must stay unset."""
        assert "MemoryDenyWriteExecute" not in unit_config["Service"]

    def test_private_tmp(self, unit_config):
        assert unit_config["Service"]["PrivateTmp"] == "yes"

    def test_environment_has_shared_bin_on_path(self, unit_config):
        env = unit_config["Service"]["Environment"]
        assert "/srv/openalph/shared/bin" in env

    def test_environment_has_bd_actor(self, unit_config):
        """Guards the single-directive form: a second `Environment=` line
        would be a DuplicateOptionError here, and dropping it silently would
        unset BD_ACTOR for every agent."""
        assert "BD_ACTOR=oa-%i" in unit_config["Service"]["Environment"]

    def test_umask_not_world_readable(self, unit_config):
        """ARCH-3: present in the installed unit, absent from the repo copy."""
        assert unit_config["Service"]["UMask"] == "0027"


class TestSingleSourceOfTruth:
    """ARCH-3: one unit file, installed and tested, with no second copy."""

    def test_canonical_unit_exists(self):
        assert CANONICAL_UNIT_PATH.is_file(), f"Missing: {CANONICAL_UNIT_PATH}"

    def test_etc_copy_is_a_symlink_to_canonical(self):
        assert UNIT_PATH.is_symlink(), (
            "etc/openalph@.service must be a symlink to the packaged unit, "
            "not a second copy that can drift"
        )
        assert UNIT_PATH.resolve() == CANONICAL_UNIT_PATH.resolve()

    def test_installer_does_not_inline_a_second_unit(self):
        """The heredoc that produced the divergent installed unit is gone.

        `install.sh` must COPY the packaged unit, not re-emit one. Matching on
        the section headers catches any reintroduced heredoc regardless of
        its delimiter.
        """
        text = INSTALL_SH.read_text()
        assert "Description=OpenAlph Agent - %i" not in text, (
            "install.sh contains an inline unit body again -- it must install "
            "the packaged data/openalph@.service instead"
        )

    def test_installer_installs_the_packaged_unit(self):
        text = INSTALL_SH.read_text()
        assert "data" in text and "openalph@.service" in text
        assert 'install -m 644 -o root -g root "${UNIT_SRC}" "${UNIT_PATH}"' in text


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
