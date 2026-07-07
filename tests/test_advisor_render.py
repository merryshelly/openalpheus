"""RED suite — advisor transcript serializer (`render_transcript`).

Spec: specs/advisor-design.md §4 (Component B — transcript serializer) + §2 A3
(deterministic, append-only rendering) + §10 test plan. Recon shapes:
specs/advisor-anchors.md §5 (history block shapes: ToolCall objects on
tool_calls, pre-wrapped tool results, thinking-as-dict-list, image list-content).

`render_transcript(system_prompt: str | None, messages: list) -> str` is a PURE
function: same input → byte-identical output, no timestamps/counts/summaries.

These tests are the SPECIFICATION. Advisor code does not exist yet, so the import
is guarded and every test fails with a clean "advisor module not implemented"
message (a FAILED line, never a collection ERROR).
"""

import re
import pytest

try:
    from openalph.tools.advisor import render_transcript, run_advisor
except ImportError:
    render_transcript = None
    run_advisor = None

from openalph.provider import ToolCall


# --- Helpers -------------------------------------------------------------

def _tc(tid, name, inp):
    """Live-history tool_calls carry ToolCall objects (anchors §5)."""
    return ToolCall(id=tid, name=name, input=inp)


def _wrapped(tool, tid, body):
    """A tool_result history entry's content is ALREADY <tool_result>-wrapped
    (anchors §5 — wrap happens before append). The renderer emits it verbatim."""
    return f'<tool_result tool="{tool}" id="{tid}">\n{body}\n</tool_result>'


def _mixed_history():
    """A history exercising every shape the serializer must handle, built as ONE
    master list so that slices [:k] extend element-wise (identity-shared prefix)."""
    return [
        {"role": "user", "content": "Please help me design feature X"},
        {"role": "assistant", "content": "Sure, let me look around first."},
        {"role": "assistant", "content": "",
         "tool_calls": [_tc("t1", "shell", {"command": "ls -la"})]},
        {"role": "tool", "tool_call_id": "t1",
         "content": _wrapped("shell", "t1", "total 0\nfile_one.py"),
         "is_error": False},
        {"role": "assistant", "content": "Now I'll read two files at once.",
         "tool_calls": [
             _tc("t2", "file_read", {"path": "a.txt"}),
             _tc("t3", "file_read", {"path": "b.txt"}),
         ]},
        {"role": "tool", "tool_call_id": "t2",
         "content": _wrapped("file_read", "t2", "alpha contents"),
         "is_error": False},
        {"role": "tool", "tool_call_id": "t3",
         "content": _wrapped("file_read", "t3", "beta contents"),
         "is_error": False},
    ]


SYS = "You are Babson. Follow OPERATIONS.md. Consult before non-trivial designs."


# --- Determinism (§2 A3) -------------------------------------------------

class TestDeterminism:

    def test_same_object_twice_byte_identical(self):
        """render(x) == render(x): repeated call on identical input is byte-stable."""
        assert render_transcript is not None, "advisor module not implemented"
        msgs = _mixed_history()
        a = render_transcript(SYS, msgs)
        b = render_transcript(SYS, msgs)
        assert a == b, "render_transcript must be deterministic (byte-identical)"

    def test_equal_value_inputs_identical(self):
        """Two independently-built EQUAL inputs → identical output (no hidden state)."""
        assert render_transcript is not None, "advisor module not implemented"
        a = render_transcript(SYS, _mixed_history())
        b = render_transcript(SYS, _mixed_history())
        assert a == b, "Equal-value inputs must render identically (no cross-call state)"

    def test_no_wallclock_or_counts(self):
        """A3: rendering must contain no timestamps/turn-counts (stability across time).

        Weak-but-meaningful pin: two renders separated in the test are identical AND
        the output does not contain an obvious ISO timestamp fragment.
        """
        assert render_transcript is not None, "advisor module not implemented"
        out = render_transcript(SYS, _mixed_history())
        # No ISO-8601-ish date (YYYY-MM-DDTHH) — would break byte-reproducibility.
        assert not re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:", out), \
            "Rendering must not embed wall-clock timestamps (A3)"


# --- A3 APPEND-ONLY PREFIX PROPERTY (critical) ---------------------------

class TestPrefixProperty:

    def test_growing_history_is_byte_prefix(self):
        """A3 (critical): for h' extending h element-wise, render(h) is a byte PREFIX
        of render(h'). Checked across ALL pairs and both system-prompt modes, over a
        history that includes plain text, tool_use, tool_result, and PARALLEL tool_use.
        """
        assert render_transcript is not None, "advisor module not implemented"
        msgs = _mixed_history()
        for sys in (None, SYS):
            renders = [render_transcript(sys, msgs[:k]) for k in range(1, len(msgs) + 1)]
            for i in range(len(renders)):
                for j in range(i, len(renders)):
                    assert renders[j].startswith(renders[i]), (
                        f"A3 VIOLATION (sys={'set' if sys else 'None'}): "
                        f"render(msgs[:{i+1}]) is NOT a byte-prefix of render(msgs[:{j+1}]). "
                        "Append-only rendering is required for advisor-side cache hits."
                    )

    def test_prefix_holds_when_last_entry_is_parallel_tool_use(self):
        """A3 specifically across the parallel-tool_use frontier (multiple tool_calls
        in one assistant message) — the trickiest growth step."""
        assert render_transcript is not None, "advisor module not implemented"
        msgs = _mixed_history()
        before = render_transcript(SYS, msgs[:4])   # ends at first tool_result
        after = render_transcript(SYS, msgs[:5])    # adds parallel-tool_use assistant msg
        assert after.startswith(before), \
            "Adding a parallel-tool_use assistant message must only APPEND to the render"


# --- Thinking excluded (§4) ---------------------------------------------

class TestThinkingExcluded:

    def test_thinking_text_never_rendered(self):
        """An assistant entry with a 'thinking' key renders NO thinking text; the
        visible content still renders."""
        assert render_transcript is not None, "advisor module not implemented"
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "VISIBLE_ANSWER_TEXT",
             "thinking": [{"thinking": "SECRET_REASONING_ZZZ", "signature": "sigABC=="}]},
        ]
        out = render_transcript(None, msgs)
        assert "SECRET_REASONING_ZZZ" not in out, "Thinking content must be excluded (§4)"
        assert "sigABC==" not in out, "Thinking signature must be excluded (§4)"
        assert "VISIBLE_ANSWER_TEXT" in out, "Visible assistant content must still render"

    def test_thinking_on_tool_use_message_excluded(self):
        """Thinking key on a tool_use-bearing assistant message is also excluded."""
        assert render_transcript is not None, "advisor module not implemented"
        msgs = [
            {"role": "assistant", "content": "acting",
             "tool_calls": [_tc("t1", "shell", {"command": "id"})],
             "thinking": [{"thinking": "HIDDEN_TU_THOUGHT", "signature": "s"}]},
        ]
        out = render_transcript(None, msgs)
        assert "HIDDEN_TU_THOUGHT" not in out, "Thinking on tool_use msg must be excluded"
        assert "id" in out, "The tool_use input must still render"


# --- Image placeholder (§4) ---------------------------------------------

class TestImagePlaceholder:

    def test_image_becomes_placeholder_with_media_type_and_kb(self):
        """Vision list-content → '[image omitted: <media_type>, ~N KB]'.
        N ≈ len(base64)*3/4 bytes. media_type + an approx-KB number appear; NO base64.
        """
        assert render_transcript is not None, "advisor module not implemented"
        # 40960 base64 chars → ~30720 bytes → ~30 KB (distinctive, clean number).
        blob = "Q" * 40960
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "please review this diagram"},
            {"type": "image", "media_type": "image/png", "data": blob},
        ]}]
        out = render_transcript(None, msgs)
        assert blob not in out, "Base64 image data must NEVER appear in the rendering"
        assert "please review this diagram" in out, "Text block alongside image must render"
        assert "image/png" in out, "Image placeholder must name the media_type"
        m = re.search(r"image omitted:\s*image/png,\s*~?\s*(\d+(?:\.\d+)?)\s*KB", out)
        assert m, f"Expected '[image omitted: image/png, ~N KB]' placeholder, got: {out!r}"
        kb = float(m.group(1))
        assert 29.0 <= kb <= 31.0, f"KB estimate (~30) off: got {kb} (len(b64)*3/4/1024)"

    def test_image_jpeg_media_type_reported(self):
        """A different media_type is reported faithfully."""
        assert render_transcript is not None, "advisor module not implemented"
        msgs = [{"role": "user", "content": [
            {"type": "image", "media_type": "image/jpeg", "data": "A" * 4096},
        ]}]
        out = render_transcript(None, msgs)
        assert "image/jpeg" in out, "Placeholder must report the actual media_type"
        assert "A" * 4096 not in out, "No base64 in output"


# --- Tool blocks render input dict + stored wrapped bytes (§4) -----------

class TestToolBlocks:

    def test_tool_use_input_dict_rendered(self):
        """A tool_use block renders its tool name, id, and input dict."""
        assert render_transcript is not None, "advisor module not implemented"
        msgs = [{"role": "assistant", "content": "",
                 "tool_calls": [_tc("toolu_abc", "shell",
                                     {"command": "grep -r NEEDLE ."})]}]
        out = render_transcript(None, msgs)
        assert "shell" in out, "tool_use must render the tool name"
        assert "toolu_abc" in out, "tool_use must render the tool call id"
        assert "grep -r NEEDLE ." in out, "tool_use must render the input dict value"
        assert "command" in out, "tool_use must render the input dict keys"

    def test_tool_result_stored_bytes_verbatim(self):
        """A tool_result renders the stored (already-<tool_result>-wrapped) bytes
        VERBATIM — the advisor sees exactly what the executor saw, no re-wrapping."""
        assert render_transcript is not None, "advisor module not implemented"
        wrapped = _wrapped("shell", "toolu_abc", "STDOUT_LINE_1\nSTDOUT_LINE_2")
        msgs = [{"role": "tool", "tool_call_id": "toolu_abc",
                 "content": wrapped, "is_error": False}]
        out = render_transcript(None, msgs)
        assert wrapped in out, \
            "Stored <tool_result>-wrapped bytes must appear verbatim (no re-wrap, no strip)"

    def test_parallel_tool_use_both_calls_rendered(self):
        """Parallel tool_use: multiple tool_calls in one assistant msg all render."""
        assert render_transcript is not None, "advisor module not implemented"
        msgs = [{"role": "assistant", "content": "batch",
                 "tool_calls": [
                     _tc("p1", "file_read", {"path": "aaa.txt"}),
                     _tc("p2", "web_search", {"query": "bbb topic"}),
                 ]}]
        out = render_transcript(None, msgs)
        assert "aaa.txt" in out and "file_read" in out, "First parallel call must render"
        assert "bbb topic" in out and "web_search" in out, "Second parallel call must render"
        assert "p1" in out and "p2" in out, "Both parallel tool call ids must render"


# --- include_system_prompt toggle (§3 config / §4) ----------------------

class TestSystemPromptToggle:

    def test_system_prompt_present_when_given(self):
        """system_prompt given → its verbatim text is in the rendering."""
        assert render_transcript is not None, "advisor module not implemented"
        out = render_transcript("SYSTEM_PROMPT_MARKER_9Z", [{"role": "user", "content": "hi"}])
        assert "SYSTEM_PROMPT_MARKER_9Z" in out, "System prompt must render when provided"

    def test_system_prompt_absent_when_none(self):
        """system_prompt None → the section is absent/empty; no leaked prompt text.

        The handler passes None when include_system_prompt=false; the pure function
        must then omit the system section entirely."""
        assert render_transcript is not None, "advisor module not implemented"
        out = render_transcript(None, [{"role": "user", "content": "hi"}])
        assert "SYSTEM_PROMPT_MARKER_9Z" not in out, "No system text when None"
        assert "hi" in out, "Transcript body still renders when system prompt omitted"


# --- transcript_max_chars head-drop (§3 config) -------------------------

class TestTranscriptMaxChars:

    def _long_history(self):
        msgs = [{"role": "user", "content": "HEAD_MARKER_EARLIEST first request"}]
        for i in range(30):
            msgs.append({"role": "assistant", "content": f"filler assistant line {i} " + "x" * 80})
            msgs.append({"role": "user", "content": f"filler user line {i} " + "y" * 80})
        msgs.append({"role": "assistant", "content": "TAIL_MARKER_LATEST final answer"})
        return msgs

    def test_head_drop_keeps_tail_with_marker(self):
        """transcript_max_chars set → keep the TAIL, drop the HEAD, add an explicit
        truncation marker; total length bounded near the budget."""
        assert render_transcript is not None, "advisor module not implemented"
        msgs = self._long_history()
        budget = 600
        out = render_transcript(None, msgs, transcript_max_chars=budget)
        assert "TAIL_MARKER_LATEST final answer" in out, "Tail (most recent) must be kept"
        assert "HEAD_MARKER_EARLIEST" not in out, "Head (earliest) must be dropped"
        assert "truncat" in out.lower(), "An explicit truncation marker must be present"
        assert len(out) <= budget + 300, \
            f"Output length {len(out)} should be bounded near budget {budget} (+marker/header)"

    def test_unlimited_zero_keeps_everything(self):
        """transcript_max_chars=0 (unlimited) → head retained, no truncation marker."""
        assert render_transcript is not None, "advisor module not implemented"
        msgs = self._long_history()
        out = render_transcript(None, msgs, transcript_max_chars=0)
        assert "HEAD_MARKER_EARLIEST" in out, "Unlimited must keep the earliest content"
        assert "TAIL_MARKER_LATEST final answer" in out, "Unlimited must keep the latest content"
        assert "truncat" not in out.lower(), "Unlimited must not add a truncation marker"

    def test_default_is_unlimited(self):
        """Omitting transcript_max_chars behaves as unlimited (default 0)."""
        assert render_transcript is not None, "advisor module not implemented"
        msgs = self._long_history()
        out = render_transcript(None, msgs)
        assert "HEAD_MARKER_EARLIEST" in out, "Default (no arg) must keep everything"


# --- Edge cases ----------------------------------------------------------

class TestEdgeCases:

    def test_empty_history_no_crash(self):
        """Empty history renders to a str without error, both prompt modes."""
        assert render_transcript is not None, "advisor module not implemented"
        assert isinstance(render_transcript(None, []), str)
        out = render_transcript(SYS, [])
        assert isinstance(out, str)
        assert SYS in out, "System prompt still renders even with empty transcript"

    def test_unicode_round_trips(self):
        """Non-ASCII content survives verbatim (no mojibake, no escaping)."""
        assert render_transcript is not None, "advisor module not implemented"
        s = "café ☕ 日本語 Ω — naïve façade — 🔮"
        msgs = [{"role": "user", "content": s},
                {"role": "assistant", "content": "réponse: " + s}]
        out = render_transcript(None, msgs)
        assert s in out, "Unicode user content must round-trip verbatim"
        assert "réponse: " + s in out, "Unicode assistant content must round-trip verbatim"
