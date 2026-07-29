"""Shell command executor for OpenAlph.

Stateless subprocess execution. Each call is independent.
"""

import asyncio
import os
import signal
from typing import Any

from . import ToolResult, truncate_result
from .security import op_egress_block_reason


async def run_shell(
    command: str,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 30,
    max_output: int = 50000,
) -> ToolResult:
    """Run a shell command via subprocess.

    Args:
        command: Shell command to execute
        cwd: Working directory (None = current process directory)
        env: Environment variables to merge with existing environment
        timeout: Timeout in seconds (default 30)
        max_output: Maximum output size in characters (default 50000)

    Returns:
        ToolResult with stdout on success, stderr+exit_code on failure.
        Timeout returns error with "timeout" in content.
    """
    # L2 op-egress guard: block+redirect commands that would print a
    # 1Password secret to stdout (op read / opread / op document get /
    # op item get --fields|--format|--reveal), BEFORE any subprocess is
    # spawned and before any state (process_env, etc.) is built. Pure
    # predicate — no I/O, no mutation. Covers sub-agents too, since they
    # route shell calls through this same run_shell.
    block_reason = op_egress_block_reason(command)
    if block_reason:
        return ToolResult(content=block_reason, is_error=True)

    # Build environment: merge provided env with existing
    process_env = os.environ.copy()
    if env is not None:
        process_env.update(env)

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=process_env,
            start_new_session=True,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            # Kill the entire process group (shell + children) on timeout.
            # Without this, create_subprocess_shell children (e.g. sleep)
            # survive as orphans and hold pipes open, hanging the event loop.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            try:
                await proc.wait()
            except Exception:
                pass
            return ToolResult(
                content="Command exceeded timeout",
                is_error=True,
            )
        finally:
            # CancelledError cleanup (kdsn.144): if the task is cancelled
            # (/stop or systemctl stop) while the subprocess is still running,
            # kill its process group before letting CancelledError propagate.
            # CancelledError is a BaseException, so the outer ``except Exception``
            # does NOT catch it — without this finally, the subprocess would be
            # orphaned in its own process group (start_new_session=True) and
            # run to completion, surviving both /stop and systemctl stop.
            # On normal/timeout paths, returncode is set and this is a no-op.
            if proc.returncode is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                try:
                    await proc.wait()
                except (asyncio.CancelledError, Exception):
                    pass

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            # Non-zero exit: error with stderr and exit code
            error_parts = []
            if stderr:
                error_parts.append(stderr.rstrip())
            if stdout:
                error_parts.append(stdout.rstrip())
            error_parts.append(f"exit code: {proc.returncode}")
            error_content = "\n".join(error_parts)
            error_content = truncate_result(error_content, max_output)
            return ToolResult(content=error_content, is_error=True)

        # Success: include stdout, and stderr if present
        if stderr:
            output = f"{stdout.rstrip()}\n[stderr]\n{stderr.rstrip()}"
        else:
            output = stdout.rstrip()

        output = truncate_result(output, max_output)
        return ToolResult(content=output, is_error=False)

    except FileNotFoundError as e:
        # Invalid cwd
        return ToolResult(
            content=f"Error: {e}",
            is_error=True,
        )
    except Exception as e:
        return ToolResult(
            content=f"Error executing command: {e}",
            is_error=True,
        )
