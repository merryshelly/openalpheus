"""Tests for the streaming DegenerationMonitor (kdsn.241.4).

The design doc's §2 false-positive / true-positive table is the spec. The
detector's whole value depends on NOT firing on legitimate bounded repetition
(pytest dots, tables, base64) while catching genuine generative loops. A
false-positive that truncates real output is worse than no detector, so the
false-positive suite is as important as the true-positive suite.
"""
import pytest

from openalph.degen import DegenerationMonitor, _WORD_MAX


def _varied_prose(nchars, seed=1):
    """Deterministic non-repeating prose (random word order) — high entropy, no
    phrase loops. Used where a test needs 'normal' output that must not trip."""
    import random
    r = random.Random(seed)
    pool = ("the model produced a detailed and varied analysis of many distinct "
            "topics including sampling penalties reasoning replay compression ratios "
            "vendor guidance token budgets streaming aborts and threshold calibration "
            "without repeating any phrase verbatim across the whole passage").split()
    out = []
    while sum(len(w) + 1 for w in out) < nchars:
        out.append(r.choice(pool))
    return " ".join(out)


def _feed_all(text, mode="warn", chunk=None, **kw):
    """Feed `text` to a fresh monitor. chunk=None -> one delta; else N-char deltas.
    Returns (monitor, tripped_bool)."""
    mon = DegenerationMonitor(mode=mode, **kw)
    tripped = False
    if chunk is None:
        tripped = mon.feed(text)
    else:
        for i in range(0, len(text), chunk):
            if mon.feed(text[i:i + chunk]):
                tripped = True
    return mon, tripped


# --------------------------------------------------------------------------
# True positives — must trip
# --------------------------------------------------------------------------
class TestTruePositives:

    def test_verbatim_phrase_loop_ngram(self):
        text = "the build is completely broken now " * 40
        mon, tripped = _feed_all(text)
        assert tripped
        assert mon.trigger_layer == "ngram"

    def test_phrase_loop_detected_when_streamed_in_small_chunks(self):
        """Delta-boundary independence: same detection whether fed whole or split."""
        text = "the build is completely broken now " * 40
        mon, tripped = _feed_all(text, chunk=7)
        assert tripped
        assert mon.trigger_layer == "ngram"

    def test_single_char_spam_char_run_backstop(self):
        text = "!" * 5000
        mon, tripped = _feed_all(text)
        assert tripped
        assert mon.trigger_layer == "char_run"
        assert mon.trigger_pos == 0

    def test_char_run_backstop_position_after_prose(self):
        prefix = _varied_prose(300) + " "   # non-repeating, so only the char-run fires
        text = prefix + "x" * 5000
        mon, tripped = _feed_all(text)
        assert tripped
        assert mon.trigger_layer == "char_run"
        assert mon.trigger_pos == len(prefix)

    def test_repeated_multiline_block_zlib(self):
        """A repeated identical multi-line block defeats the n-gram newline-reset
        but is caught by the compression layer (this IS real degeneration)."""
        block = "processing item alpha\nprocessing item beta\n"
        text = block * 200  # ~8600 chars, sustained low-ratio, >=8 distinct chars
        mon, tripped = _feed_all(text)
        assert tripped
        assert mon.trigger_layer == "zlib"


# --------------------------------------------------------------------------
# False positives — must NOT trip (the FP table from the design doc §2)
# --------------------------------------------------------------------------
class TestFalsePositives:

    def test_pytest_dot_output(self):
        # 2000 dots (below 4500 backstop; 1 distinct char blocks the zlib layer)
        text = "running tests " + "." * 2000 + " done"
        mon, tripped = _feed_all(text)
        assert not tripped

    def test_pytest_dots_with_newlines(self):
        text = "\n".join("." * 70 for _ in range(40))  # dot grid, 40 lines
        mon, tripped = _feed_all(text)
        assert not tripped

    def test_separator_line(self):
        text = "Section\n" + "=" * 200 + "\nmore content here that is normal prose."
        mon, tripped = _feed_all(text)
        assert not tripped

    def test_markdown_table_varied_rows(self):
        rows = [f"| item {i} | value {i*7 % 13} | status ok |" for i in range(60)]
        text = "| a | b | c |\n|---|---|---|\n" + "\n".join(rows)
        mon, tripped = _feed_all(text)
        assert not tripped

    def test_base64_blob(self):
        import base64, os
        blob = base64.b64encode(os.urandom(6000)).decode()
        text = "Here is the encoded payload:\n" + blob + "\nEnd."
        mon, tripped = _feed_all(text)
        assert not tripped

    def test_normal_long_prose(self):
        text = _varied_prose(6000)
        mon, tripped = _feed_all(text)
        assert not tripped

    def test_json_payload_not_flagged(self):
        """A legitimately-varied JSON array (distinct fields per object) stays
        above the compression threshold and must not fire."""
        import json
        notes = ["off-by-one in the loop bound", "missing null check on input",
                 "race between capture and refund", "unvalidated amount field",
                 "idempotency key not enforced", "timezone bug in scheduler",
                 "leaks a file handle on error", "sql built by string concat",
                 "retry storm under contention", "float used for currency"]
        obj = {"reviews": [
            {"id": i, "file": f"module_{i}.py", "line": 40 + i * 3,
             "severity": ["low", "med", "high"][i % 3], "notes": notes[i % len(notes)]}
            for i in range(20)]}
        text = json.dumps(obj, indent=2)
        mon, tripped = _feed_all(text)
        assert not tripped

    def test_short_output_below_arm_threshold(self):
        text = "ok. " * 10  # short + bounded; below arm_at
        mon, tripped = _feed_all(text)
        assert not tripped

    def test_char_run_just_below_backstop(self):
        text = "x" * 4499  # one below the backstop
        mon, tripped = _feed_all(text)
        assert not tripped

    def test_uniform_numbered_list_not_flagged(self):
        """Highly-uniform-but-legit structured output (a long numbered list) must
        not fire. This shape compresses to ~0.055 and false-fired at the old 0.15
        threshold — regression guard for the audit recalibration to 0.04."""
        text = "\n".join(
            f"{i}. Review the finding and confirm the fix is applied correctly."
            for i in range(200)
        )
        mon, tripped = _feed_all(text)
        assert not tripped

    def test_yaml_and_code_shapes_not_flagged(self):
        yaml = "\n".join(f"service_{i}:\n  enabled: true\n  port: {8000+i}" for i in range(120))
        code = "\n".join(f"    def handler_{i}(self, req):\n        return self.run(req, {i%4})"
                          for i in range(120))
        for text in (yaml, code):
            mon, tripped = _feed_all(text)
            assert not tripped


class TestBoundedState:
    """The monitor must hold bounded state regardless of input shape (kdsn.241.4
    audit: an unbounded in-progress word was O(n^2) and could block the event loop)."""

    def test_cur_word_is_capped_on_whitespace_free_stream(self):
        # A long whitespace-free blob (base64/minified). Must complete and keep
        # the in-progress word bounded — no O(n^2), no unbounded memory.
        import base64, os
        blob = base64.b64encode(os.urandom(200_000)).decode()  # ~266K chars, no spaces
        mon, tripped = _feed_all(blob, chunk=1024)
        assert len(mon._cur_word) <= _WORD_MAX
        assert not tripped  # random base64 is high-entropy — must not false-fire


# --------------------------------------------------------------------------
# Persistence gate (layer 2)
# --------------------------------------------------------------------------
class TestPersistenceGate:

    def test_single_window_dip_does_not_fire(self):
        """A brief low-ratio burst that then returns to normal prose must not
        fire the compression layer (persistence requires several ticks)."""
        # Short repeated block, then lots of varied prose so the window recovers.
        burst = "processing item alpha\nprocessing item beta\n" * 8  # < persist window
        recovery = _varied_prose(9000)  # long, non-repeating; window recovers
        mon, tripped = _feed_all(burst + recovery)
        assert not tripped


# --------------------------------------------------------------------------
# Modes and result state
# --------------------------------------------------------------------------
class TestModes:

    def test_off_mode_never_trips(self):
        text = "!" * 20000
        mon, tripped = _feed_all(text, mode="off")
        assert not tripped
        assert mon.tripped is False

    def test_warn_mode_detects(self):
        mon, tripped = _feed_all("loop loop loop " * 60, mode="warn")
        assert tripped
        assert mon.mode == "warn"

    def test_abort_mode_detects(self):
        mon, tripped = _feed_all("loop loop loop " * 60, mode="abort")
        assert tripped
        assert mon.mode == "abort"

    def test_invalid_mode_coerced_to_warn(self):
        mon = DegenerationMonitor(mode="banana")
        assert mon.mode == "warn"

    def test_feed_after_trip_is_idempotent(self):
        mon = DegenerationMonitor(mode="warn")
        first = None
        for _ in range(200):
            r = mon.feed("broken output here ")
            if r and first is None:
                first = True
            elif mon.tripped:
                # after the trip, further feeds report False
                assert r is False
        assert mon.tripped is True

    def test_empty_delta_noop(self):
        mon = DegenerationMonitor(mode="warn")
        assert mon.feed("") is False
        assert mon.tripped is False

    def test_trigger_pos_and_layer_set_on_trip(self):
        mon, tripped = _feed_all("x" * 5000)
        assert tripped
        assert mon.trigger_layer is not None
        assert mon.trigger_pos is not None
        assert 0 <= mon.trigger_pos <= 5000
