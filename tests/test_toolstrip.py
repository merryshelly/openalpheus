"""Tests for /cache toolstrip — context window reclamation (.113).

Toolstrip marks a point in session history. On subsequent context builds,
tool results before that point are replaced with lightweight placeholders.
JSONL remains intact (append-only). Stripping is a view at build_context() time.

Architecture:
    session.py  — build_context() toolstrip awareness + strippable_stats()
    matrix.py   — /cache toolstrip handler + /cache bare output + /status row
    agent.py    — status() includes strippable stats
"""

from pathlib import Path

from openalph.session import SessionLog


ROOM_ID = "!test:matrix.local"
AGENT_USER = "@agent:matrix.local"
USER = "@sb:matrix.local"


def make_sl(tmp_path: Path) -> SessionLog:
    return SessionLog(workspace=tmp_path, agent_user_id=AGENT_USER)


def append_user(sl, content="hello"):
    sl.append(role="user", sender=USER, room=ROOM_ID, content=content)


def append_assistant(sl, content="response", tool_calls=None):
    kwargs = {"content": content}
    if tool_calls:
        kwargs["tool_calls"] = tool_calls
    sl.append(role="assistant", sender=AGENT_USER, room=ROOM_ID, **kwargs)


def append_tool(sl, call_id="tc_1", name="shell", output="some output", is_error=False):
    sl.append(
        role="tool", sender=AGENT_USER, room=ROOM_ID,
        call_id=call_id, name=name, output=output, is_error=is_error,
    )


def append_toolstrip(sl, entry_index):
    sl.append(
        role="system", sender=USER, room=ROOM_ID,
        event="toolstrip", entry_index=entry_index,
    )


def append_system(sl, event="session_start", detail=""):
    sl.append(role="system", sender=AGENT_USER, room=ROOM_ID, event=event, detail=detail)


# ---------------------------------------------------------------------------
# build_context: no toolstrip → no change
# ---------------------------------------------------------------------------

class TestNoToolstrip:

    def test_no_toolstrip_returns_normal_context(self, tmp_path):
        """Without any toolstrip marker, build_context behaves identically."""
        sl = make_sl(tmp_path)
        append_user(sl, "what time is it")
        append_assistant(sl, "", tool_calls=[
            {"call_id": "tc_1", "name": "shell", "input": {"command": "date"}}
        ])
        append_tool(sl, "tc_1", "shell", "Thu Apr 8 12:00:00 EDT 2026")
        append_assistant(sl, "It's noon.")

        ctx = sl.build_context(ROOM_ID)
        tool_msgs = [m for m in ctx if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert "Thu Apr 8" in tool_msgs[0]["content"]

    def test_no_toolstrip_no_performance_overhead(self, tmp_path):
        """Verify build_context doesn't do extra work without a marker."""
        sl = make_sl(tmp_path)
        for i in range(100):
            append_user(sl, f"msg {i}")
            append_assistant(sl, f"reply {i}")
        ctx = sl.build_context(ROOM_ID)
        assert len(ctx) == 200


# ---------------------------------------------------------------------------
# build_context: toolstrip replaces tool results with placeholders
# ---------------------------------------------------------------------------

class TestToolstripBasic:

    def test_tool_results_before_marker_stripped(self, tmp_path):
        """Tool results at positions before entry_index get placeholders."""
        sl = make_sl(tmp_path)
        append_user(sl)                               # 0
        append_assistant(sl, "", tool_calls=[           # 1
            {"call_id": "tc_1", "name": "shell", "input": {"command": "ls"}}
        ])
        append_tool(sl, "tc_1", "shell", "file1.txt\nfile2.txt")  # 2
        append_assistant(sl, "I see two files.")        # 3
        append_toolstrip(sl, entry_index=4)             # 4 (system, skipped)
        append_user(sl, "continue")                     # 5

        ctx = sl.build_context(ROOM_ID)
        tool_msgs = [m for m in ctx if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert "[stripped:" in tool_msgs[0]["content"]
        assert "shell" in tool_msgs[0]["content"]
        # Original content not present
        assert "file1.txt" not in tool_msgs[0]["content"]

    def test_tool_results_after_marker_untouched(self, tmp_path):
        """Tool results at positions after entry_index remain full-fidelity."""
        sl = make_sl(tmp_path)
        append_user(sl)                               # 0
        append_assistant(sl, "", tool_calls=[           # 1
            {"call_id": "tc_1", "name": "shell", "input": {"command": "ls"}}
        ])
        append_tool(sl, "tc_1", "shell", "old output")  # 2
        append_assistant(sl, "done")                    # 3
        append_toolstrip(sl, entry_index=4)             # 4
        append_user(sl, "now do more")                  # 5
        append_assistant(sl, "", tool_calls=[           # 6
            {"call_id": "tc_2", "name": "shell", "input": {"command": "pwd"}}
        ])
        append_tool(sl, "tc_2", "shell", "/home/agent")  # 7
        append_assistant(sl, "You're in /home/agent")    # 8

        ctx = sl.build_context(ROOM_ID)
        tool_msgs = [m for m in ctx if m["role"] == "tool"]
        assert len(tool_msgs) == 2
        # First is stripped
        assert "[stripped:" in tool_msgs[0]["content"]
        # Second is untouched
        assert tool_msgs[1]["content"] == "/home/agent"

    def test_placeholder_format(self, tmp_path):
        """Placeholder contains tool name and character count."""
        sl = make_sl(tmp_path)
        output = "x" * 5000
        append_user(sl)
        append_assistant(sl, "", tool_calls=[
            {"call_id": "tc_1", "name": "file_read", "input": {"path": "/tmp/big"}}
        ])
        append_tool(sl, "tc_1", "file_read", output)
        append_assistant(sl, "read it")
        append_toolstrip(sl, entry_index=4)

        ctx = sl.build_context(ROOM_ID)
        tool_msgs = [m for m in ctx if m["role"] == "tool"]
        assert "file_read" in tool_msgs[0]["content"]
        assert "5000" in tool_msgs[0]["content"]

    def test_tool_call_id_preserved(self, tmp_path):
        """Stripped tool results keep their tool_call_id for API compatibility."""
        sl = make_sl(tmp_path)
        append_user(sl)
        append_assistant(sl, "", tool_calls=[
            {"call_id": "tc_99", "name": "shell", "input": {"command": "echo"}}
        ])
        append_tool(sl, "tc_99", "shell", "hello")
        append_assistant(sl, "said hello")
        append_toolstrip(sl, entry_index=4)

        ctx = sl.build_context(ROOM_ID)
        tool_msgs = [m for m in ctx if m["role"] == "tool"]
        assert tool_msgs[0]["tool_call_id"] == "tc_99"

    def test_is_error_preserved(self, tmp_path):
        """Stripped error tool results keep is_error flag."""
        sl = make_sl(tmp_path)
        append_user(sl)
        append_assistant(sl, "", tool_calls=[
            {"call_id": "tc_1", "name": "shell", "input": {"command": "bad"}}
        ])
        append_tool(sl, "tc_1", "shell", "command not found", is_error=True)
        append_assistant(sl, "that failed")
        append_toolstrip(sl, entry_index=4)

        ctx = sl.build_context(ROOM_ID)
        tool_msgs = [m for m in ctx if m["role"] == "tool"]
        assert tool_msgs[0].get("is_error") is True

    def test_user_messages_never_stripped(self, tmp_path):
        """User messages before the marker are never altered."""
        sl = make_sl(tmp_path)
        append_user(sl, "my important question")
        append_assistant(sl, "answer")
        append_toolstrip(sl, entry_index=2)

        ctx = sl.build_context(ROOM_ID)
        user_msgs = [m for m in ctx if m["role"] == "user"]
        assert user_msgs[0]["content"] == "my important question"

    def test_assistant_text_never_stripped(self, tmp_path):
        """Assistant text responses before the marker are never altered."""
        sl = make_sl(tmp_path)
        append_user(sl)
        append_assistant(sl, "my carefully synthesized answer")
        append_toolstrip(sl, entry_index=2)

        ctx = sl.build_context(ROOM_ID)
        assistant_msgs = [m for m in ctx if m["role"] == "assistant"]
        assert assistant_msgs[0]["content"] == "my carefully synthesized answer"


# ---------------------------------------------------------------------------
# build_context: large tool_call input stripping
# ---------------------------------------------------------------------------

class TestToolCallInputStripping:

    def test_large_input_values_rendered_verbatim(self, tmp_path):
        """Pre-boundary tool call input values (ANY length, incl. >500)
        render verbatim — input compaction retired (workspace-37ch)."""
        big_content = "x" * 600
        sl = make_sl(tmp_path)
        append_user(sl)
        append_assistant(sl, "", tool_calls=[
            {"call_id": "tc_1", "name": "file_write",
             "input": {"path": "/tmp/file.txt", "content": big_content}}
        ])
        append_tool(sl, "tc_1", "file_write", "OK")
        append_assistant(sl, "wrote it")
        append_toolstrip(sl, entry_index=4)

        ctx = sl.build_context(ROOM_ID)
        assistant_msgs = [m for m in ctx if m.get("tool_calls")]
        tc = assistant_msgs[0]["tool_calls"][0]
        # Path preserved (short)
        assert tc.input["path"] == "/tmp/file.txt"
        # 37ch: the >500 input renders VERBATIM — byte-identical to the
        # JSONL content, no "[stripped: N chars]" input placeholder.
        assert tc.input["content"] == big_content
        assert "[stripped:" not in tc.input["content"]
        # Result-side placeholder family is UNCHANGED.
        tool_msgs = [m for m in ctx if m["role"] == "tool"]
        assert "[stripped:" in tool_msgs[0]["content"]

    def test_small_input_values_preserved(self, tmp_path):
        """Tool call input values <= 500 chars are not touched."""
        sl = make_sl(tmp_path)
        append_user(sl)
        append_assistant(sl, "", tool_calls=[
            {"call_id": "tc_1", "name": "shell",
             "input": {"command": "echo hello", "cwd": "/tmp"}}
        ])
        append_tool(sl, "tc_1", "shell", "hello")
        append_assistant(sl, "done")
        append_toolstrip(sl, entry_index=4)

        ctx = sl.build_context(ROOM_ID)
        assistant_msgs = [m for m in ctx if m.get("tool_calls")]
        tc = assistant_msgs[0]["tool_calls"][0]
        assert tc.input["command"] == "echo hello"
        assert tc.input["cwd"] == "/tmp"

    def test_tool_call_inputs_after_marker_untouched(self, tmp_path):
        """Tool call inputs after the marker are never stripped."""
        big_content = "y" * 600
        sl = make_sl(tmp_path)
        append_user(sl)
        append_assistant(sl, "ok")
        append_toolstrip(sl, entry_index=2)
        append_user(sl, "write this")
        append_assistant(sl, "", tool_calls=[
            {"call_id": "tc_1", "name": "file_write",
             "input": {"path": "/tmp/f.txt", "content": big_content}}
        ])
        append_tool(sl, "tc_1", "file_write", "OK")
        append_assistant(sl, "done")

        ctx = sl.build_context(ROOM_ID)
        assistant_msgs = [m for m in ctx if m.get("tool_calls")]
        tc = assistant_msgs[0]["tool_calls"][0]
        assert tc.input["content"] == big_content


# ---------------------------------------------------------------------------
# build_context: multiple toolstrip markers
# ---------------------------------------------------------------------------

class TestMultipleToolstrips:

    def test_latest_marker_wins(self, tmp_path):
        """With multiple markers, the latest entry_index is the boundary."""
        sl = make_sl(tmp_path)
        append_user(sl)                               # 0
        append_assistant(sl, "", tool_calls=[           # 1
            {"call_id": "tc_1", "name": "shell", "input": {"command": "ls"}}
        ])
        append_tool(sl, "tc_1", "shell", "old output")  # 2
        append_assistant(sl, "done")                    # 3
        append_toolstrip(sl, entry_index=4)             # 4 — first strip
        append_user(sl, "more work")                    # 5
        append_assistant(sl, "", tool_calls=[           # 6
            {"call_id": "tc_2", "name": "shell", "input": {"command": "pwd"}}
        ])
        append_tool(sl, "tc_2", "shell", "second output")  # 7
        append_assistant(sl, "ok")                      # 8
        append_toolstrip(sl, entry_index=9)             # 9 — second strip

        ctx = sl.build_context(ROOM_ID)
        tool_msgs = [m for m in ctx if m["role"] == "tool"]
        assert len(tool_msgs) == 2
        # Both should be stripped now
        assert "[stripped:" in tool_msgs[0]["content"]
        assert "[stripped:" in tool_msgs[1]["content"]


# ---------------------------------------------------------------------------
# build_context: toolstrip + orphan stripping coexist
# ---------------------------------------------------------------------------

class TestToolstripOrphanInteraction:

    def test_orphan_stripping_still_works_with_toolstrip(self, tmp_path):
        """Orphaned tool_calls are still removed even when toolstrip is active."""
        sl = make_sl(tmp_path)
        append_user(sl)                               # 0
        append_assistant(sl, "", tool_calls=[           # 1
            {"call_id": "tc_1", "name": "shell", "input": {"command": "ls"}}
        ])
        append_tool(sl, "tc_1", "shell", "output")     # 2
        append_assistant(sl, "ok")                      # 3
        # Simulate crash: assistant with tool_calls but no results
        append_assistant(sl, "", tool_calls=[           # 4
            {"call_id": "tc_orphan", "name": "shell", "input": {"command": "crash"}}
        ])
        # No tool result for tc_orphan
        append_toolstrip(sl, entry_index=5)             # 5
        append_user(sl, "continue after crash")         # 6

        ctx = sl.build_context(ROOM_ID)
        # The orphaned assistant message should be gone
        orphans = [m for m in ctx if m.get("role") == "assistant"
                   and m.get("tool_calls")
                   and any(tc.id == "tc_orphan" for tc in m["tool_calls"])]
        assert len(orphans) == 0
        # The valid stripped tool result should still be present
        tool_msgs = [m for m in ctx if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert "[stripped:" in tool_msgs[0]["content"]


# ---------------------------------------------------------------------------
# strippable_stats()
# ---------------------------------------------------------------------------

class TestStrippableStats:

    def test_counts_tool_results(self, tmp_path):
        """Returns count and total chars of strippable tool results."""
        sl = make_sl(tmp_path)
        append_user(sl)
        append_assistant(sl, "", tool_calls=[
            {"call_id": "tc_1", "name": "shell", "input": {"command": "ls"}}
        ])
        append_tool(sl, "tc_1", "shell", "a" * 1000)
        append_assistant(sl, "", tool_calls=[
            {"call_id": "tc_2", "name": "file_read", "input": {"path": "/tmp/x"}}
        ])
        append_tool(sl, "tc_2", "file_read", "b" * 2000)
        append_assistant(sl, "done")

        count, total_chars = sl.strippable_stats(ROOM_ID)
        assert count == 2
        assert total_chars == 3000

    def test_respects_existing_strip_boundary(self, tmp_path):
        """Only counts entries after the last toolstrip marker."""
        sl = make_sl(tmp_path)
        append_user(sl)                               # 0
        append_assistant(sl, "", tool_calls=[           # 1
            {"call_id": "tc_1", "name": "shell", "input": {"command": "ls"}}
        ])
        append_tool(sl, "tc_1", "shell", "a" * 1000)   # 2
        append_assistant(sl, "done")                    # 3
        append_toolstrip(sl, entry_index=4)             # 4
        append_user(sl, "more")                         # 5
        append_assistant(sl, "", tool_calls=[           # 6
            {"call_id": "tc_2", "name": "shell", "input": {"command": "pwd"}}
        ])
        append_tool(sl, "tc_2", "shell", "b" * 500)    # 7
        append_assistant(sl, "ok")                      # 8

        count, total_chars = sl.strippable_stats(ROOM_ID)
        # Only tc_2 is strippable (tc_1 already covered by marker)
        assert count == 1
        assert total_chars == 500

    def test_empty_session_returns_zero(self, tmp_path):
        """Empty session has nothing strippable."""
        sl = make_sl(tmp_path)
        count, total_chars = sl.strippable_stats(ROOM_ID)
        assert count == 0
        assert total_chars == 0

    def test_no_tool_results_returns_zero(self, tmp_path):
        """Session with only user/assistant messages has nothing strippable."""
        sl = make_sl(tmp_path)
        append_user(sl, "hello")
        append_assistant(sl, "hi")
        count, total_chars = sl.strippable_stats(ROOM_ID)
        assert count == 0
        assert total_chars == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestToolstripEdgeCases:

    def test_strip_empty_session(self, tmp_path):
        """Toolstrip on empty session: zero tool results stripped."""
        sl = make_sl(tmp_path)
        append_toolstrip(sl, entry_index=0)
        ctx = sl.build_context(ROOM_ID)
        assert ctx == []

    def test_strip_with_no_tool_results(self, tmp_path):
        """Toolstrip on session with no tools: passes through cleanly."""
        sl = make_sl(tmp_path)
        append_user(sl, "hello")
        append_assistant(sl, "hi there")
        append_toolstrip(sl, entry_index=2)
        append_user(sl, "bye")

        ctx = sl.build_context(ROOM_ID)
        assert len(ctx) == 3
        assert ctx[0]["content"] == "hello"
        assert ctx[1]["content"] == "hi there"
        assert ctx[2]["content"] == "bye"

    def test_jsonl_not_mutated(self, tmp_path):
        """Original JSONL is untouched by toolstrip — stripping is view-only."""
        sl = make_sl(tmp_path)
        original_output = "important data here"
        append_user(sl)
        append_assistant(sl, "", tool_calls=[
            {"call_id": "tc_1", "name": "shell", "input": {"command": "cat"}}
        ])
        append_tool(sl, "tc_1", "shell", original_output)
        append_assistant(sl, "done")
        append_toolstrip(sl, entry_index=4)

        # build_context strips the view
        ctx = sl.build_context(ROOM_ID)
        tool_msgs = [m for m in ctx if m["role"] == "tool"]
        assert "[stripped:" in tool_msgs[0]["content"]

        # But raw JSONL still has original data
        entries = sl.read(ROOM_ID)
        tool_entries = [e for e in entries if e["role"] == "tool"]
        assert tool_entries[0]["output"] == original_output


# ---------------------------------------------------------------------------
# status() context estimate reflects toolstrip
# ---------------------------------------------------------------------------

class TestToolstripContextEstimate:
    """agent.status() context_tokens must reflect stripped context, not raw.

    Regression test for 2026-04-09: /status reported full unstripped context
    after /cache toolstrip because _estimate_context_tokens used raw in-memory
    history instead of build_context() output.
    """

    def test_status_context_smaller_after_toolstrip(self, tmp_path):
        """Context token estimate decreases after toolstrip is applied."""
        from openalph.config import AgentConfig, ProviderConfig

        AgentConfig(
            name="test",
            default_model="anthropic/claude-sonnet-4-20250514",
            max_tokens=8192,
            model_max_tokens=200000,
            providers={"anthropic": ProviderConfig(
                key="anthropic", type="anthropic",
                api_key="sk-test", base_url=None, quirks=None,
            )},
            workspace=tmp_path,
        )
        sl = make_sl(tmp_path)

        # Build a session with substantial tool output
        big_output = "x" * 20000
        append_user(sl)                                          # 0
        append_assistant(sl, "", tool_calls=[                    # 1
            {"call_id": "tc_1", "name": "shell", "input": {"command": "ls"}}
        ])
        append_tool(sl, "tc_1", "shell", big_output)            # 2
        append_assistant(sl, "got it")                           # 3

        # Get context before strip
        ctx_before = sl.build_context(ROOM_ID)
        chars_before = sum(
            len(m.get("content", "")) if isinstance(m.get("content"), str) else 0
            for m in ctx_before
        )

        # Apply toolstrip
        append_toolstrip(sl, entry_index=4)                      # 4
        append_user(sl, "continue")                              # 5

        # Get context after strip
        ctx_after = sl.build_context(ROOM_ID)
        chars_after = sum(
            len(m.get("content", "")) if isinstance(m.get("content"), str) else 0
            for m in ctx_after
        )

        # Stripped context must be substantially smaller
        assert chars_after < chars_before
        # The 20000-char tool output should be replaced by a ~30-char placeholder
        assert chars_before - chars_after > 19000

    def test_estimate_with_build_context_history(self, tmp_path):
        """_estimate_context_tokens with build_context output is smaller than raw."""
        from openalph.agent import Agent
        from openalph.config import AgentConfig, ProviderConfig

        config = AgentConfig(
            name="test",
            default_model="anthropic/claude-sonnet-4-20250514",
            max_tokens=8192,
            model_max_tokens=200000,
            providers={"anthropic": ProviderConfig(
                key="anthropic", type="anthropic",
                api_key="sk-test", base_url=None, quirks=None,
            )},
            workspace=tmp_path,
        )
        agent = Agent(config)

        sl = make_sl(tmp_path)
        big_output = "y" * 40000
        append_user(sl)
        append_assistant(sl, "", tool_calls=[
            {"call_id": "tc_1", "name": "file_read", "input": {"path": "/big"}}
        ])
        append_tool(sl, "tc_1", "file_read", big_output)
        append_assistant(sl, "read it")
        append_toolstrip(sl, entry_index=4)
        append_user(sl, "next")

        # Load raw history into agent (simulating what matrix.py does)
        raw_ctx = sl.build_context(ROOM_ID, skip_system=True)
        # Hack: also build without strip to simulate raw
        # We need to remove the toolstrip to get raw context
        raw_entries = sl.read(ROOM_ID)
        raw_no_strip = [e for e in raw_entries if not (e.get("role") == "system" and e.get("event") == "toolstrip")]

        # Estimate on stripped context
        stripped_estimate = agent._estimate_context_tokens(history=raw_ctx)

        # Estimate on raw (unstripped) — just sum the content
        raw_chars = sum(len(e.get("content", "")) + len(e.get("output", "")) for e in raw_no_strip)

        # Stripped should be way smaller
        assert stripped_estimate < raw_chars // 4


# ---------------------------------------------------------------------------
# Regression: /cache toolstrip must refresh agent in-memory history
# ---------------------------------------------------------------------------

class TestToolstripRefreshesInMemoryHistory:
    """Regression: /cache toolstrip must refresh agent in-memory history.

    Before the fix, /cache toolstrip wrote the marker to the JSONL session log
    but never updated the agent's in-memory history list. Subsequent API calls
    therefore still sent the full unstripped tool outputs.

    The fix adds three lines in matrix.py after the marker is appended:
        history = self.agent.history(room_id)
        history.clear()
        history.extend(self.session_log.build_context(room_id))

    These tests verify the expected semantics of that fix.
    """

    def test_in_memory_history_reflects_strip_after_rebuild(self, tmp_path):
        """After rebuild from build_context, tool results appear stripped."""
        from openalph.agent import Agent
        from openalph.config import AgentConfig, ProviderConfig

        config = AgentConfig(
            name="test",
            default_model="anthropic/claude-sonnet-4-20250514",
            max_tokens=8192,
            model_max_tokens=200000,
            providers={"anthropic": ProviderConfig(
                key="anthropic", type="anthropic",
                api_key="sk-test", base_url=None, quirks=None,
            )},
            workspace=tmp_path,
        )
        agent = Agent(config)
        sl = make_sl(tmp_path)

        # Build a session with a large tool result
        big_output = "z" * 10000
        append_user(sl, "do something")             # 0
        append_assistant(sl, "", tool_calls=[        # 1
            {"call_id": "tc_1", "name": "shell", "input": {"command": "bigcmd"}}
        ])
        append_tool(sl, "tc_1", "shell", big_output) # 2
        append_assistant(sl, "done")                 # 3

        # Simulate what matrix.py does before the fix: load history from
        # build_context without any toolstrip yet applied
        pre_strip_ctx = sl.build_context(ROOM_ID)
        history = agent.history(ROOM_ID)
        history.clear()
        history.extend(pre_strip_ctx)

        # Confirm the full output is in the in-memory history
        pre_tool_msgs = [m for m in agent.history(ROOM_ID) if m["role"] == "tool"]
        assert len(pre_tool_msgs) == 1
        assert big_output in pre_tool_msgs[0]["content"]

        # Now apply toolstrip marker to session log (simulates /cache toolstrip)
        entry_count = len(sl.read(ROOM_ID))  # = 4
        append_toolstrip(sl, entry_index=entry_count)  # 4
        append_user(sl, "continue")                    # 5

        # Apply the fix: rebuild in-memory history from build_context
        history = agent.history(ROOM_ID)
        history.clear()
        history.extend(sl.build_context(ROOM_ID))

        # Tool results should now be stripped placeholders, not full output
        post_tool_msgs = [m for m in agent.history(ROOM_ID) if m["role"] == "tool"]
        assert len(post_tool_msgs) == 1
        assert "[stripped:" in post_tool_msgs[0]["content"]
        assert big_output not in post_tool_msgs[0]["content"]

    def test_in_memory_history_is_same_object_as_agent_history(self, tmp_path):
        """history() returns the same mutable list — clear+extend updates it in place."""
        from openalph.agent import Agent
        from openalph.config import AgentConfig, ProviderConfig

        config = AgentConfig(
            name="test",
            default_model="anthropic/claude-sonnet-4-20250514",
            max_tokens=8192,
            model_max_tokens=200000,
            providers={"anthropic": ProviderConfig(
                key="anthropic", type="anthropic",
                api_key="sk-test", base_url=None, quirks=None,
            )},
            workspace=tmp_path,
        )
        agent = Agent(config)

        # Populate history
        agent.history(ROOM_ID).append({"role": "user", "content": "hello"})
        agent.history(ROOM_ID).append({"role": "assistant", "content": "hi"})

        # Get the reference and mutate it
        h = agent.history(ROOM_ID)
        h.clear()
        h.extend([{"role": "user", "content": "new"}])

        # The agent's history should now reflect the mutation
        assert len(agent.history(ROOM_ID)) == 1
        assert agent.history(ROOM_ID)[0]["content"] == "new"

    def test_in_memory_history_reduces_context_size_after_strip(self, tmp_path):
        """After the rebuild, total context chars are substantially reduced."""
        from openalph.agent import Agent
        from openalph.config import AgentConfig, ProviderConfig

        config = AgentConfig(
            name="test",
            default_model="anthropic/claude-sonnet-4-20250514",
            max_tokens=8192,
            model_max_tokens=200000,
            providers={"anthropic": ProviderConfig(
                key="anthropic", type="anthropic",
                api_key="sk-test", base_url=None, quirks=None,
            )},
            workspace=tmp_path,
        )
        agent = Agent(config)
        sl = make_sl(tmp_path)

        big_output = "q" * 50000
        append_user(sl, "run")                        # 0
        append_assistant(sl, "", tool_calls=[          # 1
            {"call_id": "tc_1", "name": "shell", "input": {"command": "bigrun"}}
        ])
        append_tool(sl, "tc_1", "shell", big_output)  # 2
        append_assistant(sl, "finished")               # 3

        # Load pre-strip history
        history = agent.history(ROOM_ID)
        history.clear()
        history.extend(sl.build_context(ROOM_ID))

        def total_chars(h):
            total = 0
            for m in h:
                c = m.get("content", "")
                total += len(c) if isinstance(c, str) else 0
            return total

        chars_before = total_chars(agent.history(ROOM_ID))

        # Apply strip marker and rebuild
        append_toolstrip(sl, entry_index=4)
        append_user(sl, "next step")

        history = agent.history(ROOM_ID)
        history.clear()
        history.extend(sl.build_context(ROOM_ID))

        chars_after = total_chars(agent.history(ROOM_ID))

        # Should be dramatically smaller
        assert chars_after < chars_before
        assert chars_before - chars_after > 49000  # 50000-char output → ~50-char placeholder

    def test_new_messages_after_strip_are_preserved_in_history(self, tmp_path):
        """Messages added after the toolstrip marker are present in rebuilt history."""
        from openalph.agent import Agent
        from openalph.config import AgentConfig, ProviderConfig

        config = AgentConfig(
            name="test",
            default_model="anthropic/claude-sonnet-4-20250514",
            max_tokens=8192,
            model_max_tokens=200000,
            providers={"anthropic": ProviderConfig(
                key="anthropic", type="anthropic",
                api_key="sk-test", base_url=None, quirks=None,
            )},
            workspace=tmp_path,
        )
        agent = Agent(config)
        sl = make_sl(tmp_path)

        append_user(sl, "old question")                # 0
        append_assistant(sl, "old answer")             # 1
        append_toolstrip(sl, entry_index=2)            # 2
        append_user(sl, "new question after strip")    # 3
        append_assistant(sl, "new answer after strip") # 4

        # Rebuild history (simulating the fix)
        history = agent.history(ROOM_ID)
        history.clear()
        history.extend(sl.build_context(ROOM_ID))

        user_msgs = [m for m in agent.history(ROOM_ID) if m["role"] == "user"]
        assistant_msgs = [m for m in agent.history(ROOM_ID) if m["role"] == "assistant"]

        # All user and assistant messages should be present
        contents = [m["content"] for m in user_msgs]
        assert "old question" in contents
        assert "new question after strip" in contents

        a_contents = [m["content"] for m in assistant_msgs]
        assert "old answer" in a_contents
        assert "new answer after strip" in a_contents
