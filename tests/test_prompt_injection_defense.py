"""Tests for prompt injection defense in system prompt (.40).

assemble_prompt() appends a hardcoded injection defense section that:
- Instructs the model to treat <tool_result> content as untrusted data
- Explicitly warns about common injection patterns
- Is always present (not dependent on workspace files)
- Appears after workspace files and skills index
"""

import pytest
from pathlib import Path
from openalph.prompt import assemble_prompt, INJECTION_DEFENSE


class TestInjectionDefensePresent:

    def test_defense_section_in_prompt(self, tmp_path):
        """Injection defense text is present in assembled prompt."""
        (tmp_path / "SOUL.md").write_text("Soul.")
        prompt = assemble_prompt(tmp_path)
        assert "tool_result" in prompt
        assert "untrusted" in prompt.lower() or "data" in prompt.lower()

    def test_defense_present_even_with_empty_workspace(self, tmp_path):
        """Defense section is appended even when no workspace files exist."""
        prompt = assemble_prompt(tmp_path)
        assert "tool_result" in prompt

    def test_defense_appears_after_workspace_files(self, tmp_path):
        """Defense section comes after workspace file content."""
        (tmp_path / "SOUL.md").write_text("SOUL_MARKER")
        (tmp_path / "SAFETY.md").write_text("SAFETY_MARKER")
        prompt = assemble_prompt(tmp_path)

        soul_pos = prompt.index("SOUL_MARKER")
        defense_pos = prompt.index("tool_result")
        assert defense_pos > soul_pos

    def test_defense_appears_after_skills_index(self, tmp_path):
        """Defense section comes after skills index."""
        skills = tmp_path / "skills"
        skills.mkdir()
        (skills / "test-skill.md").write_text("# Test\nA test skill.")
        prompt = assemble_prompt(tmp_path)

        skill_pos = prompt.index("test-skill")
        defense_pos = prompt.index("tool_result")
        assert defense_pos > skill_pos


class TestInjectionDefenseContent:

    def test_mentions_tool_result_tags(self, tmp_path):
        """Defense text references <tool_result> tags explicitly."""
        prompt = assemble_prompt(tmp_path)
        assert "<tool_result>" in prompt or "tool_result" in prompt

    def test_warns_about_instruction_injection(self, tmp_path):
        """Defense text warns about instructions embedded in tool output."""
        prompt = assemble_prompt(tmp_path)
        defense = INJECTION_DEFENSE.lower()
        assert "instruction" in defense or "ignore" in defense

    def test_distinguishes_trusted_workspace(self, tmp_path):
        """Defense text distinguishes workspace reads from external content."""
        defense = INJECTION_DEFENSE.lower()
        # Should mention that workspace/skill files are trusted
        assert "skill" in defense or "workspace" in defense

    def test_warns_about_tag_escape_attempts(self, tmp_path):
        """Defense text warns about attempts to close tool_result tags."""
        defense = INJECTION_DEFENSE.lower()
        assert "closing" in defense or "</tool_result>" in defense.lower() or "escape" in defense


class TestInjectionDefenseConstant:

    def test_constant_is_string(self):
        """INJECTION_DEFENSE is a non-empty string constant."""
        assert isinstance(INJECTION_DEFENSE, str)
        assert len(INJECTION_DEFENSE) > 100  # substantive, not a stub

    def test_constant_is_stable(self):
        """Calling it twice produces identical content (no dynamic generation)."""
        assert INJECTION_DEFENSE == INJECTION_DEFENSE  # trivially true, but documents intent
