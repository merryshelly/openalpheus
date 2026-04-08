"""Tests for cancel-awaits-task-completion fix (Code Review #10)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openalph.agent import Agent


# ---------------------------------------------------------------------------
# Agent.cancel() unit tests
# ---------------------------------------------------------------------------


def test_agent_cancel_returns_task():
    """Agent.cancel() should return the current task and call .cancel() on it."""
    agent = Agent.__new__(Agent)
    mock_task = MagicMock(spec=asyncio.Task)
    agent._current_task = mock_task

    result = agent.cancel()

    assert result is mock_task
    mock_task.cancel.assert_called_once()


def test_agent_cancel_returns_none_when_no_task():
    """Agent.cancel() should return None when there is no current task."""
    agent = Agent.__new__(Agent)
    agent._current_task = None

    result = agent.cancel()

    assert result is None


# ---------------------------------------------------------------------------
# MatrixHandler._cancel_current() integration tests
# ---------------------------------------------------------------------------


def _make_handler():
    """Return a MatrixHandler with mocked internals."""
    from openalph.matrix import MatrixBot as MatrixHandler

    handler = MatrixHandler.__new__(MatrixHandler)
    handler._current_room = "!room:example.org"
    handler.send = AsyncMock()
    handler._set_typing = AsyncMock()
    return handler


@pytest.mark.asyncio
async def test_cancel_awaits_task_completion():
    """_cancel_current() must wait for the task to finish before sending 'Cancelled.'."""
    handler = _make_handler()

    completed = False

    async def slow_work():
        nonlocal completed
        await asyncio.sleep(0.05)
        completed = True

    # Create a real running task
    loop = asyncio.get_event_loop()
    task = loop.create_task(slow_work())
    task.cancel()  # request cancellation — but it hasn't yielded yet

    mock_agent = MagicMock()
    mock_agent.cancel.return_value = task
    handler.agent = mock_agent

    # Before _cancel_current the task is not yet done
    assert not task.done()

    await handler._cancel_current()

    # After _cancel_current the task must be done
    assert task.done()
    handler.send.assert_awaited_once_with("!room:example.org", "Cancelled.")


@pytest.mark.asyncio
async def test_cancel_awaits_task_that_finishes_normally():
    """_cancel_current() should not error when awaiting a task that ends cleanly."""
    handler = _make_handler()

    async def already_done():
        return 42

    task = asyncio.ensure_future(already_done())
    await asyncio.sleep(0)  # let it complete

    mock_agent = MagicMock()
    mock_agent.cancel.return_value = task
    handler.agent = mock_agent

    await handler._cancel_current()

    handler.send.assert_awaited_once_with("!room:example.org", "Cancelled.")


@pytest.mark.asyncio
async def test_cancel_handles_already_finished_task():
    """_cancel_current() should not error when cancel() returns an already-done task."""
    handler = _make_handler()

    async def noop():
        pass

    task = asyncio.ensure_future(noop())
    await asyncio.sleep(0)  # let it finish

    mock_agent = MagicMock()
    mock_agent.cancel.return_value = task
    handler.agent = mock_agent

    await handler._cancel_current()  # must not raise

    handler.send.assert_awaited_once_with("!room:example.org", "Cancelled.")


@pytest.mark.asyncio
async def test_cancel_handles_no_task():
    """_cancel_current() should still send 'Cancelled.' when cancel() returns None."""
    handler = _make_handler()

    mock_agent = MagicMock()
    mock_agent.cancel.return_value = None
    handler.agent = mock_agent

    await handler._cancel_current()

    handler.send.assert_awaited_once_with("!room:example.org", "Cancelled.")


@pytest.mark.asyncio
async def test_cancel_handles_no_cancel_attr():
    """_cancel_current() should work when agent has no cancel() method."""
    handler = _make_handler()
    handler.agent = object()  # no cancel attribute

    await handler._cancel_current()

    handler.send.assert_awaited_once_with("!room:example.org", "Cancelled.")


@pytest.mark.asyncio
async def test_cancel_handles_task_raising_exception():
    """_cancel_current() should swallow non-CancelledError exceptions from the task."""
    handler = _make_handler()

    async def failing():
        raise ValueError("boom")

    task = asyncio.ensure_future(failing())
    await asyncio.sleep(0)  # let it raise

    mock_agent = MagicMock()
    mock_agent.cancel.return_value = task
    handler.agent = mock_agent

    await handler._cancel_current()  # must not propagate ValueError

    handler.send.assert_awaited_once_with("!room:example.org", "Cancelled.")


@pytest.mark.asyncio
async def test_cancel_does_not_block_on_hung_task():
    """_cancel_current() should return within timeout even if task is stuck.

    Reproduces the unresponsive-agent failure mode: a sub-agent is stuck in
    an httpx call that ignores cancellation. Without the timeout, _cancel_current
    blocks the event loop indefinitely, making the agent completely dead.
    """
    handler = _make_handler()

    # Gate that the test controls — lets us kill the hung task cleanly
    kill_switch = asyncio.Event()

    # Simulate a task stuck in a network call that ignores cancellation
    async def hung_task():
        try:
            await asyncio.sleep(999)
        except asyncio.CancelledError:
            # Simulate stuck: catch the cancel and keep blocking
            # until the test signals us to stop
            await kill_switch.wait()

    task = asyncio.ensure_future(hung_task())
    await asyncio.sleep(0)  # let it start

    mock_agent = MagicMock()
    mock_agent.cancel.return_value = task
    handler.agent = mock_agent
    handler._current_room = "!room:example.org"

    # Set a short timeout for the test
    handler._CANCEL_TIMEOUT = 0.1

    import time
    start = time.monotonic()
    await handler._cancel_current()
    elapsed = time.monotonic() - start

    # Should have returned quickly (within timeout + small margin), not hung
    assert elapsed < 1.0, f"_cancel_current blocked for {elapsed:.1f}s"

    # Should still have sent "Cancelled." despite timeout
    handler.send.assert_awaited_once_with("!room:example.org", "Cancelled.")

    # Clean up: release the hung task so it completes
    kill_switch.set()
    await asyncio.sleep(0)
