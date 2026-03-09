"""OpenAlph admin module: agent setup using plan/execute pattern."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

from openalph.config import CONFIG_DIR  # canonical definition in config.py

SHARED_DIR = Path("/srv/openalph/shared")
OPENALPH_GROUP = "openalph"

_RESERVED_NAMES = {"root", "nobody", "daemon", "bin", "sys"}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class AdminError(Exception):
    """Raised when an admin operation fails."""


# ---------------------------------------------------------------------------
# Operation dataclass
# ---------------------------------------------------------------------------

@dataclass
class Operation:
    kind: str
    description: str
    # useradd
    username: Optional[str] = None
    group: Optional[str] = None
    shell: Optional[str] = None
    home: Optional[Path] = None
    # mkdir / chmod / chgrp / chown / write_file
    path: Optional[Path] = None
    mode: Optional[str] = None
    # chown
    user: Optional[str] = None
    # write_file
    content: Optional[str] = None
    # systemctl
    action: Optional[str] = None
    unit: Optional[str] = None
    # chown recursive
    recursive: Optional[bool] = None


# ---------------------------------------------------------------------------
# Name validation and path helpers
# ---------------------------------------------------------------------------

def validate_agent_name(name: str) -> str:
    if not name or not name.strip():
        raise ValueError(f"Agent name must not be empty: {name!r}")
    if name in _RESERVED_NAMES:
        raise ValueError(f"Agent name is reserved: {name!r}")
    if len(name) > 32:
        raise ValueError(f"Agent name too long (max 32 chars): {name!r}")
    if not re.match(r'^[a-z][a-z0-9-]*$', name):
        raise ValueError(f"Agent name must start with lowercase letter and contain only [a-z0-9-]: {name!r}")
    if name.endswith('-'):
        raise ValueError(f"Agent name must not end with hyphen: {name!r}")
    return name


def agent_username(name: str) -> str:
    validate_agent_name(name)
    return f"oa-{name}"


def agent_home(name: str) -> Path:
    validate_agent_name(name)
    return Path(f"/home/oa-{name}")


def agent_config_path(name: str) -> Path:
    validate_agent_name(name)
    return CONFIG_DIR / f"{name}.toml"


# ---------------------------------------------------------------------------
# Config skeleton
# ---------------------------------------------------------------------------

def generate_config_skeleton(name: str) -> str:
    validate_agent_name(name)
    home = f"/home/oa-{name}"
    return f"""\
[agent]
name = "{name}"
model = "CHANGE_ME"

[provider]
api_key_cmd = "{home}/secrets/api-key.sh"

[workspace]
path = "{home}/workspace"

[matrix]
user_id = "@{name}:matrix.local"
access_token_cmd = "{home}/secrets/access-token.sh"
"""


# ---------------------------------------------------------------------------
# Plan functions
# ---------------------------------------------------------------------------

def plan_setup_shared_dir() -> list[Operation]:
    ops = [
        Operation(kind="mkdir", path=SHARED_DIR, description=f"Create shared dir {SHARED_DIR}"),
        Operation(kind="chmod", path=SHARED_DIR, mode="2770", description=f"Set mode 2770 on {SHARED_DIR}"),
        Operation(kind="chgrp", path=SHARED_DIR, group=OPENALPH_GROUP, description=f"Set group {OPENALPH_GROUP} on {SHARED_DIR}"),
        Operation(kind="mkdir", path=SHARED_DIR / "beads", description="Create beads subdir"),
        Operation(kind="chmod", path=SHARED_DIR / "beads", mode="2770", description="Set mode 2770 on beads subdir"),
        Operation(kind="mkdir", path=SHARED_DIR / "docs", description="Create docs subdir"),
        Operation(kind="chmod", path=SHARED_DIR / "docs", mode="2770", description="Set mode 2770 on docs subdir"),
    ]
    return ops


def plan_create_agent(name: str) -> list[Operation]:
    validate_agent_name(name)
    username = agent_username(name)
    home = agent_home(name)
    config_path = agent_config_path(name)

    ops: list[Operation] = []

    # Create Unix user
    ops.append(Operation(
        kind="useradd",
        username=username,
        group=OPENALPH_GROUP,
        shell="/usr/sbin/nologin",
        home=home,
        description=f"Create user {username}",
    ))

    # Scaffold workspace dirs (before chown so recursive chown covers them)
    for subdir in ["workspace", "workspace/memory", "workspace/skills", ".config", ".cache"]:
        ops.append(Operation(kind="mkdir", path=home / subdir, description=f"Create {home / subdir}"))

    # Set home permissions + recursive ownership (after mkdirs)
    ops.append(Operation(kind="chmod", path=home, mode="750", description=f"chmod 750 {home}"))
    ops.append(Operation(kind="chown", path=home, user=username, group=OPENALPH_GROUP, recursive=True, description=f"chown -R {username}:{OPENALPH_GROUP} {home}"))

    # Config dir
    ops.append(Operation(kind="mkdir", path=CONFIG_DIR, description=f"Create config dir {CONFIG_DIR}"))

    # Write config skeleton
    ops.append(Operation(
        kind="write_file",
        path=config_path,
        content=generate_config_skeleton(name),
        description=f"Write config skeleton to {config_path}",
    ))

    # Enable systemd service (do NOT start)
    ops.append(Operation(
        kind="systemctl",
        action="enable",
        unit=f"openalph@{name}.service",
        description=f"Enable openalph@{name}.service",
    ))

    return ops


# ---------------------------------------------------------------------------
# Execute plan
# ---------------------------------------------------------------------------

def execute_plan(ops: list[Operation]) -> None:
    for op in ops:
        try:
            if op.kind == "mkdir":
                subprocess.run(["mkdir", "-p", str(op.path)], check=True)
            elif op.kind == "chmod":
                subprocess.run(["chmod", op.mode, str(op.path)], check=True)
            elif op.kind == "chgrp":
                subprocess.run(["chgrp", op.group, str(op.path)], check=True)
            elif op.kind == "chown":
                cmd = ["chown"]
                if op.recursive:
                    cmd.append("-R")
                cmd.extend([f"{op.user}:{op.group}", str(op.path)])
                subprocess.run(cmd, check=True)
            elif op.kind == "useradd":
                result = subprocess.run(
                    ["useradd", "-g", op.group, "-s", op.shell, "-d", str(op.home), "-m", op.username],
                )
                if result.returncode not in (0, 9):
                    raise AdminError(f"useradd failed with exit code {result.returncode} for {op.username}")
            elif op.kind == "write_file":
                op.path.parent.mkdir(parents=True, exist_ok=True)
                op.path.write_text(op.content)
            elif op.kind == "systemctl":
                subprocess.run(["systemctl", op.action, op.unit], check=True)
            else:
                raise AdminError(f"Unknown operation kind: {op.kind}")
        except subprocess.CalledProcessError as e:
            raise AdminError(f"Operation {op.kind} failed: {e}") from e


# ---------------------------------------------------------------------------
# High-level wrappers
# ---------------------------------------------------------------------------

def create_agent(name: str, *, dry_run: bool = False) -> list[Operation]:
    ops = plan_create_agent(name)
    if not dry_run:
        execute_plan(ops)
    return ops


def setup_shared_dir(*, dry_run: bool = False) -> list[Operation]:
    ops = plan_setup_shared_dir()
    if not dry_run:
        execute_plan(ops)
    return ops
