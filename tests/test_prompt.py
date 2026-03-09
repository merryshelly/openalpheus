"""Tests for system prompt assembly.

Interface contract:
    assemble_prompt(workspace: Path) -> str

Reads known workspace files in a defined order, adds headers,
scans skills/ directory to build a brief index.
Only known files are included — no arbitrary .md files.

File order: SOUL.md, OPERATOR.md, SAFETY.md, OPERATIONS.md, ENVIRONMENT.md, WAKE.md
"""

import pytest
from pathlib import Path
from openalph.prompt import assemble_prompt


# --- Core assembly ---


class TestAssemblePrompt:

    def test_reads_standard_files(self, tmp_path):
        (tmp_path / "SOUL.md").write_text("I am a test agent.")
        (tmp_path / "SAFETY.md").write_text("Safety rules here.")
        (tmp_path / "OPERATOR.md").write_text("About the operator.")

        prompt = assemble_prompt(tmp_path)

        assert "I am a test agent." in prompt
        assert "Safety rules here." in prompt
        assert "About the operator." in prompt

    def test_missing_files_skipped(self, tmp_path):
        """Only files that exist are included. No errors for missing ones."""
        (tmp_path / "SOUL.md").write_text("Just the soul.")

        prompt = assemble_prompt(tmp_path)

        assert "Just the soul." in prompt

    def test_file_order(self, tmp_path):
        """Files appear in a defined, stable order."""
        (tmp_path / "SOUL.md").write_text("SOUL_MARKER")
        (tmp_path / "OPERATOR.md").write_text("OPERATOR_MARKER")
        (tmp_path / "SAFETY.md").write_text("SAFETY_MARKER")
        (tmp_path / "OPERATIONS.md").write_text("OPERATIONS_MARKER")
        (tmp_path / "ENVIRONMENT.md").write_text("ENVIRONMENT_MARKER")
        (tmp_path / "WAKE.md").write_text("WAKE_MARKER")

        prompt = assemble_prompt(tmp_path)

        soul = prompt.index("SOUL_MARKER")
        operator = prompt.index("OPERATOR_MARKER")
        safety = prompt.index("SAFETY_MARKER")
        operations = prompt.index("OPERATIONS_MARKER")
        environment = prompt.index("ENVIRONMENT_MARKER")
        wake = prompt.index("WAKE_MARKER")

        assert safety < soul < operator < wake < environment < operations

    def test_files_have_headers(self, tmp_path):
        """Each file's content is preceded by a header identifying it."""
        (tmp_path / "SOUL.md").write_text("Soul content here.")

        prompt = assemble_prompt(tmp_path)

        assert "SOUL.md" in prompt
        assert "Soul content here." in prompt

    def test_empty_workspace(self, tmp_path):
        """Empty workspace produces empty or minimal prompt."""
        prompt = assemble_prompt(tmp_path)
        assert isinstance(prompt, str)

    def test_non_workspace_md_files_ignored(self, tmp_path):
        """Only known workspace files are included — not arbitrary .md files."""
        (tmp_path / "SOUL.md").write_text("Soul.")
        (tmp_path / "random.txt").write_text("Should not appear.")
        (tmp_path / "NOTES.md").write_text("Also should not appear.")
        (tmp_path / "README.md").write_text("Nope.")
        (tmp_path / "MEMORY.md").write_text("Not a known file anymore.")

        prompt = assemble_prompt(tmp_path)

        assert "Should not appear" not in prompt
        assert "Also should not appear" not in prompt
        assert "Nope" not in prompt
        assert "Not a known file anymore" not in prompt

    def test_all_six_files(self, tmp_path):
        """All 6 workspace files are included when present."""
        files = ["SOUL.md", "OPERATOR.md", "SAFETY.md",
                 "OPERATIONS.md", "ENVIRONMENT.md", "WAKE.md"]
        for f in files:
            (tmp_path / f).write_text(f"Content of {f}")

        prompt = assemble_prompt(tmp_path)

        for f in files:
            assert f"Content of {f}" in prompt


# --- Skills index ---


class TestSkillsIndex:

    def test_skills_indexed(self, tmp_path):
        """Skills directory is scanned and an index is included."""
        skills = tmp_path / "skills"
        skills.mkdir()
        (skills / "email.md").write_text("# Email\nSend and receive email via SMTP.")
        (skills / "wallet.md").write_text("# Wallet\nEthereum wallet operations.")

        prompt = assemble_prompt(tmp_path)

        assert "email" in prompt.lower()
        assert "wallet" in prompt.lower()

    def test_skill_name_from_filename(self, tmp_path):
        """Skill name derives from filename (minus .md extension)."""
        skills = tmp_path / "skills"
        skills.mkdir()
        (skills / "garmin-connect.md").write_text("# Garmin Connect\nHealth data.")

        prompt = assemble_prompt(tmp_path)

        assert "garmin-connect" in prompt.lower()

    def test_empty_skills_directory(self, tmp_path):
        """Empty skills directory doesn't crash or add noise to the prompt."""
        (tmp_path / "skills").mkdir()
        (tmp_path / "SOUL.md").write_text("Soul.")

        prompt = assemble_prompt(tmp_path)
        assert "Soul." in prompt

    def test_no_skills_directory(self, tmp_path):
        """Missing skills directory is fine."""
        (tmp_path / "SOUL.md").write_text("Soul.")

        prompt = assemble_prompt(tmp_path)
        assert "Soul." in prompt

    def test_non_md_files_in_skills_ignored(self, tmp_path):
        """Non-.md files in skills directory are not indexed."""
        skills = tmp_path / "skills"
        skills.mkdir()
        (skills / "helper.py").write_text("print('not a skill')")
        (skills / "email.md").write_text("# Email\nSend email.")

        prompt = assemble_prompt(tmp_path)

        assert "helper.py" not in prompt
        assert "email" in prompt.lower()

    def test_skills_appear_after_workspace_files(self, tmp_path):
        """Skills index appears after the workspace files."""
        (tmp_path / "SOUL.md").write_text("SOUL_MARKER")
        skills = tmp_path / "skills"
        skills.mkdir()
        (skills / "test-skill.md").write_text("# Test\nA test skill.")

        prompt = assemble_prompt(tmp_path)

        soul_pos = prompt.index("SOUL_MARKER")
        skill_pos = prompt.lower().index("test-skill")
        assert soul_pos < skill_pos
