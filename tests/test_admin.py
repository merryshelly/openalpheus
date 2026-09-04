"""Tests for openalph.admin module.

Agent administration: name validation, path computation, config skeleton
generation, and agent creation / shared-dir setup.

System operations use a plan/execute pattern:
- plan_*() return Operation lists (testable without root or mocking)
- execute_plan() runs the operations (integration tests need root)
- create_agent() / setup_shared_dir() are high-level wrappers
"""

import subprocess
import tomllib
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from openalph.admin import (
    AdminError,
    Operation,
    validate_agent_name,
    agent_username,
    agent_home,
    agent_config_path,
    generate_config_skeleton,
    plan_create_agent,
    plan_setup_shared_dir,
    create_agent,
    setup_shared_dir,
    execute_plan,
    SHARED_DIR,
    OPENALPH_GROUP,
    CONFIG_DIR,
)


# ---------------------------------------------------------------------------
# Name validation
# ---------------------------------------------------------------------------


class TestValidateAgentName:
    """Rules: lowercase letter start, [a-z0-9-], no trailing hyphen, 1-32 chars."""

    @pytest.mark.parametrize(
        "name",
        ["watson", "babson", "my-agent", "agent1", "watson-2b", "a", "a" * 32],
    )
    def test_valid_names_accepted(self, name):
        assert validate_agent_name(name) == name

    @pytest.mark.parametrize(
        "name",
        [
            "",          # empty
            "  ",        # whitespace only
            "Watson",    # uppercase
            "my_agent",  # underscore
            "my agent",  # space
            "agent@1",   # special char
            "1agent",    # starts with number
            "-agent",    # starts with hyphen
            "agent-",    # ends with hyphen
            "a" * 33,    # too long
        ],
    )
    def test_invalid_names_rejected(self, name):
        with pytest.raises(ValueError):
            validate_agent_name(name)

    @pytest.mark.parametrize("name", ["root", "nobody", "daemon", "bin", "sys"])
    def test_reserved_names_rejected(self, name):
        with pytest.raises(ValueError):
            validate_agent_name(name)


# ---------------------------------------------------------------------------
# Path computation (pure, no side effects)
# ---------------------------------------------------------------------------


class TestPathComputation:
    def test_username(self):
        assert agent_username("watson") == "oa-watson"

    def test_home(self):
        assert agent_home("watson") == Path("/home/oa-watson")

    def test_config_path(self):
        assert agent_config_path("watson") == Path("/etc/openalph/agents/watson.toml")

    def test_username_validates(self):
        with pytest.raises(ValueError):
            agent_username("INVALID")

    def test_home_validates(self):
        with pytest.raises(ValueError):
            agent_home("INVALID")

    def test_config_path_validates(self):
        with pytest.raises(ValueError):
            agent_config_path("INVALID")


# ---------------------------------------------------------------------------
# Config skeleton
# ---------------------------------------------------------------------------


class TestConfigSkeleton:
    def test_valid_toml(self):
        skeleton = generate_config_skeleton("watson")
        parsed = tomllib.loads(skeleton)
        assert isinstance(parsed, dict)

    def test_required_sections(self):
        parsed = tomllib.loads(generate_config_skeleton("watson"))
        for section in ("agent", "providers", "workspace", "matrix"):
            assert section in parsed, f"Missing section: {section}"
        assert "anthropic" in parsed["providers"], "Missing providers.anthropic sub-table"

    def test_agent_name_matches(self):
        parsed = tomllib.loads(generate_config_skeleton("watson"))
        assert parsed["agent"]["name"] == "watson"

    def test_workspace_path(self):
        parsed = tomllib.loads(generate_config_skeleton("watson"))
        assert parsed["workspace"]["path"] == "/home/oa-watson/workspace"

    def test_matrix_user_id(self):
        parsed = tomllib.loads(generate_config_skeleton("babson"))
        assert parsed["matrix"]["user_id"] == "@babson:CHANGE_ME"

    def test_api_key_cmd_in_agent_home(self):
        parsed = tomllib.loads(generate_config_skeleton("watson"))
        assert "/home/oa-watson/" in parsed["providers"]["anthropic"]["api_key_cmd"]

    def test_access_token_cmd_in_agent_home(self):
        parsed = tomllib.loads(generate_config_skeleton("watson"))
        assert "/home/oa-watson/" in parsed["matrix"]["access_token_cmd"]

    def test_model_is_placeholder(self):
        parsed = tomllib.loads(generate_config_skeleton("watson"))
        assert parsed["agent"]["default_model"] == "CHANGE_ME"

    def test_validates_name(self):
        with pytest.raises(ValueError):
            generate_config_skeleton("INVALID")

    def test_has_homeserver(self):
        parsed = tomllib.loads(generate_config_skeleton("watson"))
        assert parsed["matrix"]["homeserver"] == "http://localhost:4269"

    def test_provider_type(self):
        parsed = tomllib.loads(generate_config_skeleton("watson"))
        assert parsed["providers"]["anthropic"]["type"] == "anthropic"

    def test_has_max_tokens_defaults(self):
        parsed = tomllib.loads(generate_config_skeleton("watson"))
        assert "max_tokens" in parsed["agent"]
        assert "model_max_tokens" in parsed["agent"]


# ---------------------------------------------------------------------------
# Plan: setup shared directory
# ---------------------------------------------------------------------------


class TestPlanSetupSharedDir:
    @pytest.fixture
    def ops(self):
        return plan_setup_shared_dir()

    def test_returns_operation_list(self, ops):
        assert isinstance(ops, list)
        assert all(isinstance(op, Operation) for op in ops)
        assert len(ops) > 0

    def test_creates_main_dir(self, ops):
        mkdir_ops = [op for op in ops if op.kind == "mkdir"]
        paths = {op.path for op in mkdir_ops}
        assert SHARED_DIR in paths

    def test_creates_beads_subdir(self, ops):
        mkdir_ops = [op for op in ops if op.kind == "mkdir"]
        paths = {op.path for op in mkdir_ops}
        assert SHARED_DIR / "beads" in paths

    def test_creates_docs_subdir(self, ops):
        mkdir_ops = [op for op in ops if op.kind == "mkdir"]
        paths = {op.path for op in mkdir_ops}
        assert SHARED_DIR / "docs" in paths

    def test_sets_mode_2770(self, ops):
        chmod_ops = [op for op in ops if op.kind == "chmod" and op.path == SHARED_DIR]
        assert len(chmod_ops) >= 1
        assert chmod_ops[0].mode == "2770"

    def test_sets_group_openalph(self, ops):
        chgrp_ops = [op for op in ops if op.kind == "chgrp"]
        main_chgrp = [op for op in chgrp_ops if op.path == SHARED_DIR]
        assert len(main_chgrp) >= 1
        assert main_chgrp[0].group == OPENALPH_GROUP

    def test_subdirs_mode_2770(self, ops):
        """beads/ and docs/ must also be 2770 for group write."""
        chmod_ops = [op for op in ops if op.kind == "chmod"]
        beads_chmod = [op for op in chmod_ops if op.path == SHARED_DIR / "beads"]
        docs_chmod = [op for op in chmod_ops if op.path == SHARED_DIR / "docs"]
        assert len(beads_chmod) >= 1 and beads_chmod[0].mode == "2770"
        assert len(docs_chmod) >= 1 and docs_chmod[0].mode == "2770"

    def test_every_op_has_description(self, ops):
        for op in ops:
            assert op.description, f"Missing description on {op.kind} op"


# ---------------------------------------------------------------------------
# Plan: create agent
# ---------------------------------------------------------------------------


class TestPlanCreateAgent:
    @pytest.fixture
    def ops(self):
        return plan_create_agent("watson")

    def test_returns_operation_list(self, ops):
        assert isinstance(ops, list)
        assert all(isinstance(op, Operation) for op in ops)
        assert len(ops) > 0

    def test_validates_name_first(self):
        with pytest.raises(ValueError):
            plan_create_agent("INVALID")

    # --- User creation ---

    def test_creates_user(self, ops):
        useradd_ops = [op for op in ops if op.kind == "useradd"]
        assert len(useradd_ops) == 1

    def test_username_is_oa_prefixed(self, ops):
        useradd = [op for op in ops if op.kind == "useradd"][0]
        assert useradd.username == "oa-watson"

    def test_user_group_is_openalph(self, ops):
        useradd = [op for op in ops if op.kind == "useradd"][0]
        assert useradd.group == OPENALPH_GROUP

    def test_user_shell_is_nologin(self, ops):
        useradd = [op for op in ops if op.kind == "useradd"][0]
        assert useradd.shell == "/usr/sbin/nologin"

    def test_user_home_dir(self, ops):
        useradd = [op for op in ops if op.kind == "useradd"][0]
        assert useradd.home == Path("/home/oa-watson")

    # --- Home directory ---

    def test_home_mode_750(self, ops):
        home = Path("/home/oa-watson")
        chmod_ops = [op for op in ops if op.kind == "chmod" and op.path == home]
        assert len(chmod_ops) >= 1
        assert chmod_ops[0].mode == "750"

    # --- Workspace scaffold ---

    def test_scaffolds_workspace(self, ops):
        mkdir_ops = [op for op in ops if op.kind == "mkdir"]
        paths = {op.path for op in mkdir_ops}
        home = Path("/home/oa-watson")
        assert home / "workspace" in paths

    def test_scaffolds_workspace_memory(self, ops):
        mkdir_ops = [op for op in ops if op.kind == "mkdir"]
        paths = {op.path for op in mkdir_ops}
        assert Path("/home/oa-watson/workspace/memory") in paths

    def test_scaffolds_workspace_skills(self, ops):
        mkdir_ops = [op for op in ops if op.kind == "mkdir"]
        paths = {op.path for op in mkdir_ops}
        assert Path("/home/oa-watson/workspace/skills") in paths

    def test_scaffolds_dotconfig(self, ops):
        mkdir_ops = [op for op in ops if op.kind == "mkdir"]
        paths = {op.path for op in mkdir_ops}
        assert Path("/home/oa-watson/.config") in paths

    def test_scaffolds_cache(self, ops):
        mkdir_ops = [op for op in ops if op.kind == "mkdir"]
        paths = {op.path for op in mkdir_ops}
        assert Path("/home/oa-watson/.cache") in paths

    # --- Config skeleton ---

    def test_writes_config_skeleton(self, ops):
        write_ops = [op for op in ops if op.kind == "write_file"]
        config_writes = [
            op for op in write_ops
            if op.path == Path("/etc/openalph/agents/watson.toml")
        ]
        assert len(config_writes) == 1

    def test_config_skeleton_is_valid_toml(self, ops):
        write_ops = [op for op in ops if op.kind == "write_file"]
        config_write = [
            op for op in write_ops
            if op.path == Path("/etc/openalph/agents/watson.toml")
        ][0]
        parsed = tomllib.loads(config_write.content)
        assert parsed["agent"]["name"] == "watson"

    # --- systemd ---

    def test_enables_systemd_service(self, ops):
        systemctl_ops = [op for op in ops if op.kind == "systemctl"]
        enable_ops = [op for op in systemctl_ops if op.action == "enable"]
        assert len(enable_ops) == 1
        assert enable_ops[0].unit == "openalph@watson.service"

    def test_does_not_start_service(self, ops):
        """new-agent enables but does NOT start the service."""
        systemctl_ops = [op for op in ops if op.kind == "systemctl"]
        start_ops = [op for op in systemctl_ops if op.action == "start"]
        assert len(start_ops) == 0

    # --- Config dir ---

    def test_creates_config_dir(self, ops):
        mkdir_ops = [op for op in ops if op.kind == "mkdir"]
        paths = {op.path for op in mkdir_ops}
        assert CONFIG_DIR in paths

    # --- Ownership ---

    def test_sets_home_ownership(self, ops):
        chown_ops = [op for op in ops if op.kind == "chown"]
        home_chown = [
            op for op in chown_ops
            if op.path == Path("/home/oa-watson")
        ]
        assert len(home_chown) >= 1
        assert home_chown[0].user == "oa-watson"
        assert home_chown[0].group == OPENALPH_GROUP

    def test_chown_is_recursive(self, ops):
        """chown must be recursive so scaffold dirs get correct ownership."""
        chown_ops = [op for op in ops if op.kind == "chown" and op.path == Path("/home/oa-watson")]
        assert len(chown_ops) >= 1
        assert chown_ops[0].recursive is True

    def test_chown_after_mkdirs(self, ops):
        """chown must come after all mkdirs to cover scaffold dirs."""
        chown_idx = next(i for i, op in enumerate(ops) if op.kind == "chown")
        last_mkdir_idx = max(i for i, op in enumerate(ops) if op.kind == "mkdir" and (Path("/home/oa-watson") in op.path.parents or op.path == Path("/home/oa-watson")))
        assert chown_idx > last_mkdir_idx

    # --- Description ---

    def test_every_op_has_description(self, ops):
        for op in ops:
            assert op.description, f"Missing description on {op.kind} op"


# ---------------------------------------------------------------------------
# Execute plan (mocked subprocess)
# ---------------------------------------------------------------------------


class TestExecutePlan:
    @patch("openalph.admin.subprocess.run")
    def test_executes_operations(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        ops = [
            Operation(kind="mkdir", path=Path("/tmp/test"), description="test"),
        ]
        execute_plan(ops)
        assert mock_run.call_count >= 1

    @patch("openalph.admin.subprocess.run")
    def test_raises_admin_error_on_failure(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, "mkdir")
        ops = [
            Operation(kind="mkdir", path=Path("/tmp/test"), description="test"),
        ]
        with pytest.raises(AdminError):
            execute_plan(ops)

    @patch("openalph.admin.subprocess.run")
    def test_useradd_returns_9_is_not_error(self, mock_run):
        """Exit code 9 from useradd means user already exists — idempotent."""
        result = MagicMock(returncode=9)
        mock_run.return_value = result
        ops = [
            Operation(
                kind="useradd",
                username="oa-test",
                group="openalph",
                shell="/usr/sbin/nologin",
                home=Path("/home/oa-test"),
                description="create user",
            ),
        ]
        # Should not raise
        execute_plan(ops)

    def test_write_file_creates_parent_dirs(self, tmp_path):
        """write_file operations should create parent dirs if needed."""
        target = tmp_path / "sub" / "dir" / "test.toml"
        ops = [
            Operation(
                kind="write_file",
                path=target,
                content="[agent]\nname = \"test\"\n",
                description="write config",
            ),
        ]
        execute_plan(ops)
        assert target.exists()
        assert "[agent]" in target.read_text()


# ---------------------------------------------------------------------------
# High-level wrappers
# ---------------------------------------------------------------------------


class TestCreateAgentWrapper:
    @patch("openalph.admin.execute_plan")
    def test_dry_run_returns_plan_without_executing(self, mock_exec):
        result = create_agent("watson", dry_run=True)
        mock_exec.assert_not_called()
        assert isinstance(result, list)
        assert len(result) > 0

    @patch("openalph.admin.execute_plan")
    def test_executes_when_not_dry_run(self, mock_exec):
        create_agent("watson")
        mock_exec.assert_called_once()


class TestSetupSharedDirWrapper:
    @patch("openalph.admin.execute_plan")
    def test_dry_run_returns_plan_without_executing(self, mock_exec):
        result = setup_shared_dir(dry_run=True)
        mock_exec.assert_not_called()
        assert isinstance(result, list)

    @patch("openalph.admin.execute_plan")
    def test_executes_when_not_dry_run(self, mock_exec):
        setup_shared_dir()
        mock_exec.assert_called_once()


# ===========================================================================
# BUG-2 — `new-agent` must not clobber a live agent
#
# `execute_plan` deliberately tolerates useradd's exit 9 ("user exists") for
# idempotency, but then unconditionally rewrote /etc/openalph/agents/<name>.toml
# with the CHANGE_ME skeleton and overwrote the agent's customized
# OPERATIONS.md. README and INSTALL present `sudo openalph new-agent <name>`
# as the normal flow, so an accidental re-run destroyed a running agent's
# wired-up provider and Matrix config with no prompt -- unlike install.sh
# step8, which guards on `id oa-<name>` and requires --force.
# ===========================================================================

class TestNewAgentDoesNotClobber:

    def test_existing_file_is_not_overwritten(self, tmp_path):
        live = tmp_path / "agent.toml"
        live.write_text('LIVE CONFIG\napi_key = "sk-real"\n')

        execute_plan([Operation(
            kind="write_file", path=live, content="CHANGE_ME skeleton",
            overwrite=False, description="write config",
        )])

        assert live.read_text().startswith("LIVE CONFIG")

    def test_force_overwrites(self, tmp_path):
        live = tmp_path / "agent.toml"
        live.write_text("LIVE CONFIG\n")

        execute_plan([Operation(
            kind="write_file", path=live, content="CHANGE_ME skeleton",
            overwrite=True, description="write config",
        )])

        assert live.read_text() == "CHANGE_ME skeleton"

    def test_missing_file_is_still_created(self, tmp_path):
        """The guard must not break the first-run path."""
        new = tmp_path / "nested" / "agent.toml"

        execute_plan([Operation(
            kind="write_file", path=new, content="skeleton",
            overwrite=False, description="write config",
        )])

        assert new.read_text() == "skeleton"

    def test_default_plan_never_overwrites(self):
        write_ops = [o for o in plan_create_agent("demo") if o.kind == "write_file"]
        assert write_ops, "expected the plan to write config and OPERATIONS.md"
        assert all(not o.overwrite for o in write_ops)

    def test_force_plan_overwrites(self):
        write_ops = [
            o for o in plan_create_agent("demo", force=True) if o.kind == "write_file"
        ]
        assert all(o.overwrite for o in write_ops)

    def test_config_is_written_640_not_default_umask(self):
        """Was root:root under the default umask, unlike the installer's 640."""
        write_ops = [o for o in plan_create_agent("demo") if o.kind == "write_file"]
        assert any(o.file_mode == "640" for o in write_ops)

    def test_plan_writes_security_footer(self):
        """PHIL-1 rollout gap (found in pre-merge review): direct `new-agent`
        did not write SECURITY_FOOTER.md at all -- only install.sh's bootstrap
        population step did, so a directly-created agent got the correct
        byte-identical FALLBACK behavior but no visible, editable file. The
        plan must now include it, with the same skip-existing-unless-force
        contract as OPERATIONS.md and the config skeleton, and its content
        must be exactly the INJECTION_DEFENSE constant prompt.py falls back
        to (so a freshly-created workspace's footer and an upgraded
        workspace's fallback are identical from turn one)."""
        from openalph.prompt import INJECTION_DEFENSE

        write_ops = [o for o in plan_create_agent("demo") if o.kind == "write_file"]
        footer_ops = [o for o in write_ops if o.path.name == "SECURITY_FOOTER.md"]
        assert len(footer_ops) == 1, "expected exactly one SECURITY_FOOTER.md write op"
        op = footer_ops[0]
        assert op.path.parent.name == "workspace"
        assert op.content == INJECTION_DEFENSE
        assert op.overwrite is False

    def test_plan_writes_security_footer_force(self):
        write_ops = [o for o in plan_create_agent("demo", force=True) if o.kind == "write_file"]
        footer_ops = [o for o in write_ops if o.path.name == "SECURITY_FOOTER.md"]
        assert len(footer_ops) == 1
        assert footer_ops[0].overwrite is True
