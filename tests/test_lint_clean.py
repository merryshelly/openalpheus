"""Mechanical lint ratchet (workspace-kdsn.326).

The suite itself enforces lint-zero: `ruff check .` under the pyproject
ruleset (E4/E7/E9/F) must exit 0 — host-side AND in-cage. Ruff runs as a
subprocess binary (NOT `sys.executable -m ruff` — ruff is a Rust binary on
PATH, not an importable module). `--no-cache` keeps this safe on read-only
mounts (ruff otherwise writes .ruff_cache/ into the cwd). Skips loudly when
ruff is not installed; fails with the rule statistics tail when the tree
carries findings.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUFF_TIMEOUT = 120  # seconds; generous headroom for slow rigs


def test_full_tree_lint_zero():
    """`ruff check .` exits 0 on the whole tree (mechanical ratchet)."""
    ruff = shutil.which("ruff")
    if ruff is None:
        pytest.skip("ruff not available")

    result = subprocess.run(
        [ruff, "check", ".", "--no-cache"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=RUFF_TIMEOUT,
    )
    if result.returncode != 0:
        stats = subprocess.run(
            [ruff, "check", ".", "--no-cache", "--statistics"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=RUFF_TIMEOUT,
        )
        pytest.fail(
            "lint ratchet tripped — `ruff check .` must exit 0\n"
            + stats.stdout[-4000:]
        )
