"""Per-tool executor override seam — THE SPECIFICATION (bead workspace-kdsn.317).

Platform seam letting a caller redirect individual tool executions (e.g. `shell`
into a Harbor task container) WITHOUT monkey-patching. Default path is
byte-identical to today: no map → every built-in branch, guard, and registry
update runs exactly as before.

Contract
--------
1. Agent.__init__(config, tool_executors=None):
   - tool_executors: dict[str, ToolExecutor] | None.
     ToolExecutor = async callable, invoked as:
         await executor(name=<tool name>, input=<normalized dict>,
                        tool_config=<dict>, agent_config=<as dispatched>)
     and returning ToolResult.
   - Stored DEFENSIVELY (dict copy — later mutation of the caller's dict must
     not affect the agent). Default None → self.tool_executors is None.
   - Validation at construction: non-dict → ValueError; any non-callable value
     → ValueError.
2. execute_tool(..., tool_executors=None): new keyword parameter with default
   None, forwarded to _execute_tool_inner. The existing positional-compat shim
   (kdsn.305.2: dict in the agent_config slot = callbacks) MUST keep working.
3. Seam consult in _execute_tool_inner — placed AFTER name validation, required-
   param validation, GC-sentinel rejection, input copy, workspace path-join
   normalization, and resolved-path computation; BEFORE the built-in if/elif
   chain. If tool_executors contains an entry for the incoming tool NAME, the
   executor is awaited and its ToolResult returned; the built-in branch does
   NOT run.
4. Bridged-path semantics:
   - Branch-specific normalization does NOT apply (e.g. the shell branch's
     cwd→workspace default is branch-local; bridged executors see input as
     normalized by the GENERIC blocks only).
   - read-registry auto-update (_update_read_registry) does NOT run on the
     bridged path — registry bookkeeping for bridged tools is the executor's
     responsibility.
   - The credential-redaction tail in execute_tool (R9 invariant: every return
     path) STILL applies to bridged results.
   - Executor exceptions propagate unchanged — identical to how built-in
     executor exceptions behave today.
   - GC-sentinel, unknown-tool, and missing-param refusals fire BEFORE the
     seam: the executor never sees marker inputs, unknown names, or invalid
     params.
5. Propagation (explicit kwargs, no ambient state):
   - agent.py's dispatch passes tool_executors=self.tool_executors.
   - The "subagent" tool branch forwards tool_executors into run_subagent
     (always, as an explicit kwarg — None when the map is absent).
   - run_subagent(..., tool_executors=None) passes it into its internal
     execute_tool calls (A9: subs inherit the parent's bridge automatically).
   - The spotter call site is unchanged (default None → local execution).
"""

import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from openalph.config import AgentConfig, ProviderConfig
from openalph.provider import Response, ToolCall, Usage
from openalph.tools import ToolResult, _GC_PLACEHOLDER_RE, execute_tool
from openalph.tools.subagent import run_subagent


# ------------------------------------------------------------------ helpers


def make_provider(key="default", type="anthropic", api_key="sk-test", base_url=None, quirks=None):
    return ProviderConfig(key=key, type=type, api_key=api_key, base_url=base_url, quirks=quirks or [])


def make_agent_config(workspace, **kwargs):
    """Mirror the AgentConfig construction used by tests/test_agent_tools.py.
    (Scaffolding — if the real required fields differ, fix THIS helper, never
    the assertions.)"""
    defaults = dict(
        name="test-agent",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": make_provider(key="anthropic")},
        workspace=Path(workspace),
        max_iterations=25,
        truncation_limit=50000,
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def make_ns(workspace):
    """Minimal agent_config stand-in for direct execute_tool calls (the only
    attribute the generic dispatch path requires is .workspace)."""
    return SimpleNamespace(workspace=Path(workspace))


def make_recorder(result=None):
    """Executor double: records every invocation, returns a preset ToolResult."""
    calls = []

    async def executor(**kwargs):
        calls.append(kwargs)
        if result is not None:
            return result
        return ToolResult(content="bridged-ok", is_error=False)

    executor.calls = calls
    return executor


def text_response(text, input_tokens=100, output_tokens=50):
    """Mirrors tests/test_agent_tools.py — normal text response, no tool calls."""
    return Response(
        content=text,
        tool_calls=[],
        model="claude-sonnet-4-20250514",
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        stop_reason="end_turn",
    )


def tool_use_response(tool_calls, text="", input_tokens=100, output_tokens=50):
    """Mirrors tests/test_agent_tools.py — response carrying tool_use blocks."""
    return Response(
        content=text,
        tool_calls=tool_calls,
        model="claude-sonnet-4-20250514",
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        stop_reason="tool_use",
    )


def gc_marker_string():
    """A string the platform's GC-sentinel regex will flag."""
    # Literal adjusted to the live _GC_PLACEHOLDER_RE (this helper's
    # self-check below pins the match — see the helper's original note).
    marker = "[stripped: 12345 chars]"
    assert _GC_PLACEHOLDER_RE.match(marker.strip()), (
        "marker string must match the live _GC_PLACEHOLDER_RE — adjust the "
        "literal to the current pattern"
    )
    return marker


# ------------------------------------------------------------------ constructor surface


async def test_agent_accepts_tool_executors(tmp_path):
    from openalph.agent import Agent

    with patch("openalph.agent.assemble_prompt", return_value="system prompt"):
        fn = make_recorder()
        cfg = make_agent_config(tmp_path / "ws")
        source_map = {"shell": fn}
        agent = Agent(cfg, tool_executors=source_map)
        assert agent.tool_executors == {"shell": fn}
        source_map["shell"] = "mutated-after-construction"
        assert agent.tool_executors["shell"] is fn  # defensive copy


async def test_agent_tool_executors_default_none(tmp_path):
    from openalph.agent import Agent

    with patch("openalph.agent.assemble_prompt", return_value="system prompt"):
        agent = Agent(make_agent_config(tmp_path / "ws"))
        assert agent.tool_executors is None


async def test_agent_rejects_non_callable_executor(tmp_path):
    from openalph.agent import Agent

    with patch("openalph.agent.assemble_prompt", return_value="system prompt"):
        with pytest.raises(ValueError):
            Agent(make_agent_config(tmp_path / "ws"), tool_executors={"shell": "not-callable"})


async def test_agent_rejects_non_dict_tool_executors(tmp_path):
    from openalph.agent import Agent

    with patch("openalph.agent.assemble_prompt", return_value="system prompt"):
        with pytest.raises(ValueError):
            Agent(make_agent_config(tmp_path / "ws"), tool_executors=["shell"])


# ------------------------------------------------------------------ seam dispatch (execute_tool level)


async def test_default_shell_path_unchanged(tmp_path):
    """No map → byte-identical built-in behavior: local subprocess runs IN the
    workspace (cwd default intact), file lands on the host FS."""
    out = tmp_path / "out.txt"
    result = await execute_tool(
        name="shell",
        input={"command": f"printf local-ran > {out}"},
        tool_config={},
        agent_config=make_ns(tmp_path),
    )
    assert result.is_error is False
    assert out.exists() and out.read_text() == "local-ran"


async def test_bridged_shell_executor_receives_call(tmp_path):
    import os

    marker = tmp_path / "should-not-run.marker"
    rec = make_recorder()
    result = await execute_tool(
        name="shell",
        input={"command": f"touch {marker}"},
        tool_config={},
        agent_config=make_ns(tmp_path),
        tool_executors={"shell": rec},
    )
    assert result.content == "bridged-ok"
    assert len(rec.calls) == 1
    call = rec.calls[0]
    assert call["name"] == "shell"
    assert call["input"]["command"] == f"touch {marker}"
    assert call["tool_config"] == {}
    assert not os.path.exists(marker)  # local subprocess never ran


async def test_bridged_result_passes_redaction_tail(tmp_path):
    secret = "sk-ant-api03-" + "abcdefghij0123456789"
    rec = make_recorder(result=ToolResult(content=f"the key is {secret}", is_error=False))
    result = await execute_tool(
        name="shell",
        input={"command": "irrelevant"},
        tool_config={},
        agent_config=make_ns(tmp_path),
        tool_executors={"shell": rec},
    )
    assert secret not in result.content
    assert "[REDACTED" in result.content


async def test_bridged_write_skips_host_fs_and_registry(tmp_path):
    target = tmp_path / "f.txt"
    reg = {}
    rec = make_recorder()
    result = await execute_tool(
        name="file_write",
        input={"path": str(target), "content": "bridged"},
        tool_config={},
        agent_config=make_ns(tmp_path),
        callbacks={"read_registry": reg},
        tool_executors={"file_write": rec},
    )
    assert result.is_error is False
    assert len(rec.calls) == 1
    assert not target.exists()  # host FS untouched
    assert reg == {}  # no registry auto-update on the bridged path


async def test_bridged_edit_leaves_host_file_untouched(tmp_path):
    host_file = tmp_path / "edit-target.txt"
    host_file.write_text("local-content")
    rec = make_recorder()
    result = await execute_tool(
        name="file_edit",
        input={"path": str(host_file), "old_text": "local", "new_text": "BRIDGED"},
        tool_config={},
        agent_config=make_ns(tmp_path),
        tool_executors={"file_edit": rec},
    )
    assert result.is_error is False
    assert host_file.read_text() == "local-content"  # built-in edit never ran


async def test_registry_auto_update_only_on_builtin_path(tmp_path):
    real_file = tmp_path / "readable.txt"
    real_file.write_text("hello")
    reg_builtin = {}
    await execute_tool(
        name="file_read",
        input={"path": str(real_file)},
        tool_config={},
        agent_config=make_ns(tmp_path),
        callbacks={"read_registry": reg_builtin},
    )
    assert str(real_file.resolve()) in reg_builtin  # built-in path updates registry

    reg_bridged = {}
    rec = make_recorder(result=ToolResult(content="hello", is_error=False))
    await execute_tool(
        name="file_read",
        input={"path": str(real_file)},
        tool_config={},
        agent_config=make_ns(tmp_path),
        callbacks={"read_registry": reg_bridged},
        tool_executors={"file_read": rec},
    )
    assert len(rec.calls) == 1
    assert reg_bridged == {}  # bridged path leaves registry alone


async def test_gc_sentinel_blocks_before_seam(tmp_path):
    rec = make_recorder()
    result = await execute_tool(
        name="shell",
        input={"command": gc_marker_string()},
        tool_config={},
        agent_config=make_ns(tmp_path),
        tool_executors={"shell": rec},
    )
    assert result.is_error is True
    assert "context-GC elision marker" in result.content  # the pinned refusal message (log line says "GC placeholder sentry" — content does not)
    assert rec.calls == []  # executor never sees marker input


async def test_unknown_tool_rejected_before_seam(tmp_path):
    rec = make_recorder()
    result = await execute_tool(
        name="definitely_not_a_tool",
        input={},
        tool_config={},
        agent_config=make_ns(tmp_path),
        tool_executors={"definitely_not_a_tool": rec},
    )
    assert result.is_error is True
    assert "Unknown tool" in result.content
    assert rec.calls == []


async def test_missing_param_rejected_before_seam(tmp_path):
    rec = make_recorder()
    result = await execute_tool(
        name="shell",
        input={},
        tool_config={},
        agent_config=make_ns(tmp_path),
        tool_executors={"shell": rec},
    )
    assert result.is_error is True
    assert "Missing required parameter" in result.content
    assert rec.calls == []


async def test_executor_exception_propagates(tmp_path):
    async def boom(**kwargs):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await execute_tool(
            name="shell",
            input={"command": "x"},
            tool_config={},
            agent_config=make_ns(tmp_path),
            tool_executors={"shell": boom},
        )


async def test_shell_cwd_default_is_branch_local(tmp_path):
    """Built-in: cwd defaults to workspace. Bridged: NO branch-local defaults —
    the executor sees input exactly as the generic blocks left it."""
    builtin = await execute_tool(
        name="shell",
        input={"command": "pwd"},
        tool_config={},
        agent_config=make_ns(tmp_path),
    )
    assert str(tmp_path) in builtin.content  # workspace cwd injected on built-in path

    rec = make_recorder()
    await execute_tool(
        name="shell",
        input={"command": "pwd"},
        tool_config={},
        agent_config=make_ns(tmp_path),
        tool_executors={"shell": rec},
    )
    assert "cwd" not in rec.calls[0]["input"]  # no branch-local default injected


async def test_map_missing_entry_falls_through_to_builtin(tmp_path):
    real_file = tmp_path / "plain.txt"
    real_file.write_text("plain-content")
    rec = make_recorder()
    result = await execute_tool(
        name="file_read",
        input={"path": str(real_file)},
        tool_config={},
        agent_config=make_ns(tmp_path),
        tool_executors={"file_write": rec},  # entry for a DIFFERENT tool
    )
    assert result.is_error is False
    assert "plain-content" in result.content
    assert rec.calls == []  # built-in local read ran


async def test_tool_config_forwarded_to_executor(tmp_path):
    rec = make_recorder()
    await execute_tool(
        name="shell",
        input={"command": "x"},
        tool_config={"default_timeout": 99},
        agent_config=make_ns(tmp_path),
        tool_executors={"shell": rec},
    )
    assert rec.calls[0]["tool_config"] == {"default_timeout": 99}


async def test_agent_config_forwarded_to_executor(tmp_path):
    sentinel = object()
    rec = make_recorder()
    await execute_tool(
        name="shell",
        input={"command": "x"},
        tool_config={},
        agent_config=sentinel,
        tool_executors={"shell": rec},
    )
    assert rec.calls[0]["agent_config"] is sentinel


async def test_positional_callbacks_compat_still_works(tmp_path):
    """kdsn.305.2: a dict in the agent_config slot is a callbacks mapping.
    The new kwarg must not break that shim."""
    reg = {}
    real_file = tmp_path / "cb.txt"
    real_file.write_text("cb-content")
    result = await execute_tool(
        "file_read",
        {"path": str(real_file)},
        {},
        {"read_registry": reg},  # fourth positional = callbacks mapping
    )
    assert result.is_error is False
    assert "cb-content" in result.content


# ------------------------------------------------------------------ propagation (A9)


def test_run_subagent_signature_has_tool_executors():
    sig = inspect.signature(run_subagent)
    assert "tool_executors" in sig.parameters
    assert sig.parameters["tool_executors"].default is None


async def test_subagent_branch_forwards_tool_executors(tmp_path):
    """The subagent tool branch must ALWAYS pass tool_executors through to
    run_subagent — the map when present, None when absent (explicit kwarg)."""
    rec = make_recorder()
    with patch("openalph.tools.subagent.run_subagent", new=AsyncMock()) as mock_run:
        mock_run.return_value = ToolResult(content="sub-done", is_error=False)
        await execute_tool(
            name="subagent",
            input={"task": "do a thing"},
            tool_config={},
            agent_config=make_ns(tmp_path),
            tool_executors={"shell": rec},
        )
        assert mock_run.await_count == 1
        assert mock_run.await_args.kwargs["tool_executors"] == {"shell": rec}

        await execute_tool(
            name="subagent",
            input={"task": "do a thing"},
            tool_config={},
            agent_config=make_ns(tmp_path),
        )
        assert mock_run.await_args.kwargs["tool_executors"] is None


async def test_sub_dispatch_uses_parent_executors(tmp_path):
    """THE A9 TEST: a subagent's tool call routes through the parent-provided
    executor (sub `shell hostname` would see the container's hostname, not the
    eval host's). Scripted provider: first response issues a shell tool call,
    second returns text — same pattern as tests/test_executor_subagent.py."""
    rec = make_recorder(result=ToolResult(content="bridged-shell-out", is_error=False))
    scripted = [
        tool_use_response([ToolCall(id="c1", name="shell", input={"command": "echo hi"})]),
        text_response("sub-done"),
    ]
    with patch("openalph.tools.subagent.complete", new_callable=AsyncMock, side_effect=scripted):
        result = await run_subagent(
            "run a shell command",
            make_agent_config(tmp_path / "ws"),
            tool_executors={"shell": rec},
        )
    assert len(rec.calls) == 1
    assert rec.calls[0]["name"] == "shell"
    assert rec.calls[0]["input"]["command"] == "echo hi"
    assert result.is_error is False
    assert "sub-done" in result.content


async def test_agent_dispatch_passes_tool_executors(tmp_path):
    """Agent.handle_input's execute_tool dispatch carries the agent's map."""
    from openalph.agent import Agent

    rec = make_recorder(result=ToolResult(content="bridged-ok", is_error=False))
    agent = Agent(make_agent_config(tmp_path / "ws"), tool_executors={"shell": rec})
    scripted = [
        tool_use_response([ToolCall(id="t1", name="shell", input={"command": "echo hi"})]),
        text_response("done"),
    ]
    with patch("openalph.agent.assemble_prompt", return_value="system prompt"):
        agent2 = Agent(make_agent_config(tmp_path / "ws"), tool_executors={"shell": rec})
    with patch("openalph.agent.stream", side_effect=_make_stream_responses(scripted)):
        result = await agent2.handle_input("run a shell command")
    assert len(rec.calls) == 1
    assert rec.calls[0]["name"] == "shell"
    assert "done" in result  # handle_input returns the reply TEXT (str), not a ToolResult


def _make_stream_responses(responses):
    """Mirrors tests/test_agent_tools.py make_stream_responses."""
    call_iter = iter(responses)

    async def _stream(*args, **kwargs):
        from openalph.provider import StreamEvent

        response = next(call_iter)
        if response.content:
            yield StreamEvent(type="text", content=response.content)
        for i, tc in enumerate(response.tool_calls):
            yield StreamEvent(type="tool_done", tool_index=i, tool_call=tc)
        yield StreamEvent(
            type="done",
            response=response,
            stop_reason=response.stop_reason,
            model=response.model,
        )

    return _stream
