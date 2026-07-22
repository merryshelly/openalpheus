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


# ===========================================================================
# PHIL-1 — the security footer is operator-owned, not a hardcoded string
#
# Until v0.1.3 `assemble_prompt` appended ~44 lines of INJECTION_DEFENSE prose
# unconditionally: no config flag, no workspace override, and not one of the
# operator-owned markdown files. The README's headline promise -- "your agent
# doesn't read a single character you didn't put there" -- was therefore
# false, and the operator could neither see nor edit the text without reading
# the source. The text may well be desirable; shipping it as an immutable
# string is what made the claim untrue.
# ===========================================================================

import pathlib

import openalph.templates
from openalph.prompt import (
    INJECTION_DEFENSE,
    SECURITY_FOOTER_FILENAME,
    assemble_prompt,
)

TEMPLATE_FOOTER = (
    pathlib.Path(openalph.templates.__file__).parent / SECURITY_FOOTER_FILENAME
)


class TestSecurityFooterIsOperatorOwned:

    def test_template_ships_with_the_package(self):
        assert TEMPLATE_FOOTER.is_file(), f"Missing: {TEMPLATE_FOOTER}"

    def test_template_matches_the_fallback_constant(self):
        """The extracted file must be byte-identical to the constant.

        Otherwise upgrading an existing workspace would silently change the
        agent's instructions.
        """
        assert TEMPLATE_FOOTER.read_text().strip() == INJECTION_DEFENSE.strip()

    def test_workspace_file_is_used_when_present(self, tmp_path):
        (tmp_path / "SOUL.md").write_text("test agent\n")
        (tmp_path / SECURITY_FOOTER_FILENAME).write_text("## My Own Rules\nBe careful.\n")

        prompt = assemble_prompt(tmp_path)

        assert "My Own Rules" in prompt
        assert "Tool Result Security" not in prompt, (
            "the hardcoded constant overrode the operator's own file"
        )

    def test_falls_back_when_workspace_predates_the_file(self, tmp_path):
        """Upgrading must not silently drop the instruction."""
        (tmp_path / "SOUL.md").write_text("test agent\n")

        prompt = assemble_prompt(tmp_path)

        assert "Tool Result Security" in prompt

    def test_can_be_switched_off(self, tmp_path):
        """`[agent] injection_defense = false` appends nothing at all."""
        (tmp_path / "SOUL.md").write_text("test agent\n")
        (tmp_path / SECURITY_FOOTER_FILENAME).write_text("## My Own Rules\n")

        prompt = assemble_prompt(tmp_path, injection_defense=False)

        assert "My Own Rules" not in prompt
        assert "Tool Result Security" not in prompt

    def test_no_behavioural_text_beyond_operator_files(self, tmp_path):
        """With the footer off, the only framework addition is mechanical.

        The `## Runtime` block names the workspace path; it carries no
        behavioural instruction. The README names it explicitly rather than
        claiming nothing is added.
        """
        (tmp_path / "SOUL.md").write_text("ONLY THIS.\n")

        prompt = assemble_prompt(tmp_path, injection_defense=False)

        assert prompt.startswith("## SOUL.md\nONLY THIS.")
        remainder = prompt[len("## SOUL.md\nONLY THIS.\n"):].strip()
        assert remainder == "" or remainder.startswith("## Runtime"), remainder
