"""Tests: CancelledError kills the shell subprocess process group (kdsn.144).

run_shell() spawns subprocesses with start_new_session=True (separate process
group). The timeout handler kills the process group, but CancelledError
(/stop or systemctl stop) bypassed it — CancelledError is a BaseException,
not caught by ``except Exception``, so the subprocess was orphaned and ran
to completion, surviving both /stop and systemctl stop.

The fix: a ``finally`` block that kills the process group if the subprocess
is still running (proc.returncode is None). On normal/timeout completion
paths, returncode is set and the finally is a no-op.
"""

import asyncio
import os
import pytest

from openalph.tools.shell import run_shell


class TestCancelledKillsProcessGroup:

    @pytest.mark.asyncio
    async def test_cancelled_kills_subprocess(self, tmp_path):
        """CancelledError during run_shell kills the subprocess process group.

        The subprocess writes its PID to a file, then sleeps. After cancelling
        the task, the PID should be dead (SIGKILL'd by the finally block).
        """
        pid_file = tmp_path / "child.pid"
        # echo $$ writes the shell PID; exec sleep replaces the shell so
        # the PID persists as the sleep process. start_new_session=True
        # means this is also the process group leader.
        cmd = f"echo $$ > {pid_file} && exec sleep 30"
        task = asyncio.create_task(run_shell(cmd, timeout=60))

        # Wait for the subprocess to start and write its PID
        for _ in range(20):
            await asyncio.sleep(0.1)
            if pid_file.exists():
                break
        assert pid_file.exists(), "subprocess did not start — PID file not written"
        pid = int(pid_file.read_text().strip())

        # Verify the process is alive before cancellation
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pytest.skip("subprocess exited before we could cancel — racy, retry")

        # Cancel the task (simulates /stop)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # The finally block should have SIGKILL'd the process group.
        # Give the OS a moment to process the signal + reap.
        await asyncio.sleep(0.3)

        # Process should be dead
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)

    @pytest.mark.asyncio
    async def test_cancelled_kills_child_processes_too(self, tmp_path):
        """Cancelling kills the ENTIRE process group, not just the leader.

        A shell command that spawns a child sleep process — both should be
        killed when the parent task is cancelled.
        """
        parent_pid_file = tmp_path / "parent.pid"
        child_pid_file = tmp_path / "child.pid"
        # The shell spawns a background child sleep, writes both PIDs, then
        # sleeps itself. Both are in the same process group.
        cmd = (
            f"echo $$ > {parent_pid_file} && "
            f"sleep 30 & echo $! > {child_pid_file} && "
            f"sleep 30"
        )
        task = asyncio.create_task(run_shell(cmd, timeout=60))

        # Wait for both PIDs to be written
        for _ in range(30):
            await asyncio.sleep(0.1)
            if parent_pid_file.exists() and child_pid_file.exists():
                break
        assert parent_pid_file.exists(), "parent PID not written"
        assert child_pid_file.exists(), "child PID not written"
        parent_pid = int(parent_pid_file.read_text().strip())
        child_pid = int(child_pid_file.read_text().strip())

        # Cancel the task
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        await asyncio.sleep(0.3)

        # Both parent and child should be dead (process group kill)
        with pytest.raises(ProcessLookupError):
            os.kill(parent_pid, 0)
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)

    @pytest.mark.asyncio
    async def test_normal_completion_not_affected(self):
        """Normal completion: the finally block is a no-op (returncode is set)."""
        result = await run_shell("echo hello")
        assert result.is_error is False
        assert "hello" in result.content

    @pytest.mark.asyncio
    async def test_timeout_still_works(self):
        """Timeout path still kills the process (existing behavior preserved)."""
        result = await run_shell("sleep 60", timeout=1)
        assert result.is_error is True
        assert "timeout" in result.content.lower()
