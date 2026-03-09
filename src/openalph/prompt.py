from pathlib import Path


def assemble_prompt(workspace: Path) -> str:
    """
    Assemble the system prompt from workspace files and skills index.
    
    Reads 6 specific files in order, adds headers, and appends a skills index.
    Missing files are silently skipped. Returns empty string if workspace is empty.
    """
    # Define the 6 files to read, in order.
    # Safety and identity first — most important, least likely to be lost to context.
    files_to_read = [
        "SAFETY.md",
        "SOUL.md",
        "OPERATOR.md",
        "WAKE.md",
        "ENVIRONMENT.md",
        "OPERATIONS.md",
    ]
    
    prompt_parts = []
    
    # Read each file in order, adding headers
    for filename in files_to_read:
        file_path = workspace / filename
        if file_path.exists():
            # Add header identifying the file
            prompt_parts.append(f"## {filename}")
            prompt_parts.append(file_path.read_text())
    
    # Build skills index from .md files in skills/ directory
    skills_dir = workspace / "skills"
    if skills_dir.exists():
        # Only include .md files, ignore others
        skill_files = [f.stem for f in skills_dir.iterdir() if f.suffix == ".md"]
        if skill_files:
            prompt_parts.append("## Skills")
            for skill_name in sorted(skill_files):
                prompt_parts.append(f"- {skill_name}")
    
    # Inject runtime workspace path only if not already mentioned in loaded files
    workspace_str = str(workspace.resolve())
    already_mentioned = any(workspace_str in part for part in prompt_parts)
    if not already_mentioned:
        prompt_parts.append("## Runtime")
        prompt_parts.append(f"Workspace: {workspace_str}")
        prompt_parts.append("Use this as the working directory for shell commands (pass as cwd).")

    # Join all parts with newlines and return
    return "\n".join(prompt_parts) if prompt_parts else ""
