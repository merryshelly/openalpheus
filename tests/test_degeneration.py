"""Tests for degeneration detection in provider.py."""
import pytest
from openalph.provider import (
    _detect_and_truncate_degeneration,
    _DEGEN_CHAR_THRESHOLD,
    _DEGEN_WARNING,
)


class TestDetectAndTruncateDegeneration:
    """Tests for _detect_and_truncate_degeneration."""

    def test_normal_text_unchanged(self):
        text = "This is a perfectly normal response with some punctuation! Great."
        result, degenerate = _detect_and_truncate_degeneration(text)
        assert result == text
        assert degenerate is False

    def test_empty_string(self):
        result, degenerate = _detect_and_truncate_degeneration("")
        assert result == ""
        assert degenerate is False

    def test_none_input(self):
        result, degenerate = _detect_and_truncate_degeneration(None)
        assert result is None
        assert degenerate is False

    def test_short_text_below_threshold(self):
        text = "!" * (_DEGEN_CHAR_THRESHOLD - 1)
        result, degenerate = _detect_and_truncate_degeneration(text)
        assert result == text
        assert degenerate is False

    def test_exact_threshold_triggers(self):
        """Exactly threshold-length run of identical chars triggers detection."""
        text = "Hello world" + "!" * _DEGEN_CHAR_THRESHOLD
        result, degenerate = _detect_and_truncate_degeneration(text)
        assert degenerate is True
        assert "Hello world" in result
        assert _DEGEN_WARNING in result
        assert "!" * _DEGEN_CHAR_THRESHOLD not in result

    def test_watson_exclamation_pattern(self):
        """Reproduces the actual Watson bug: coherent text then exclamation flood."""
        coherent = "**Status Update**\n\n| Component | Status |\n|---|---|\n| UV | ✅ |\n| Air Quality | ✅ |"
        degenerate_suffix = "!" * 32000
        text = coherent + degenerate_suffix
        result, degenerate = _detect_and_truncate_degeneration(text)
        assert degenerate is True
        assert "Status Update" in result
        assert _DEGEN_WARNING in result
        # Should not contain the exclamation flood
        assert "!" * 100 not in result

    def test_entirely_degenerate_output(self):
        """When the entire output is degenerate, return just the warning."""
        text = "!" * 500
        result, degenerate = _detect_and_truncate_degeneration(text)
        assert degenerate is True
        assert "repetition collapse" in result.lower()

    def test_preserves_text_before_degeneration(self):
        """Text before the degenerate run is preserved."""
        prefix = "Here is my analysis:\n\n1. First point\n2. Second point"
        text = prefix + "a" * _DEGEN_CHAR_THRESHOLD
        result, degenerate = _detect_and_truncate_degeneration(text)
        assert degenerate is True
        assert "First point" in result
        assert "Second point" in result

    def test_different_degenerate_characters(self):
        """Detection works for any repeated character, not just '!'."""
        for char in [".", "-", "a", " ", "\n", "x", "0"]:
            text = f"Some text{char * _DEGEN_CHAR_THRESHOLD}"
            result, degenerate = _detect_and_truncate_degeneration(text)
            assert degenerate is True, f"Failed to detect run of '{repr(char)}'"

    def test_multiple_short_runs_no_trigger(self):
        """Multiple runs below threshold don't trigger."""
        # 30 exclamations, then text, then 30 more — each below threshold
        text = "!" * 30 + " hello " + "!" * 30
        result, degenerate = _detect_and_truncate_degeneration(text)
        assert degenerate is False
        assert result == text

    def test_trailing_whitespace_stripped(self):
        """Trailing whitespace before degenerate run is stripped."""
        text = "Clean text.   " + "!" * _DEGEN_CHAR_THRESHOLD
        result, degenerate = _detect_and_truncate_degeneration(text)
        assert degenerate is True
        assert result.startswith("Clean text.")
        assert not result.startswith("Clean text.   \n")

    def test_legitimate_code_blocks_not_triggered(self):
        """Code with repeated chars (e.g., separators) below threshold is safe."""
        text = "```\n" + "=" * 40 + "\n```"  # 40 < 50 threshold
        result, degenerate = _detect_and_truncate_degeneration(text)
        assert degenerate is False
        assert result == text
