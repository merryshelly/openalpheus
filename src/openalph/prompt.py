from pathlib import Path


def assemble_prompt(workspace: Path) -> str:
    """
    Assemble the system prompt from workspace files and skills index.
    
    Reads 6 specific files in order, adds headers, and appends a skills index.
    Missing files are silently skipped. Returns empty string if workspace is empty.
    """
    # Define the 6 files to read, in order
    files_to_read = [
        "SOUL.md",
        "OPERATOR.md", 
        "SAFETY.md",
        "OPERATIONS.md",
        "ENVIRONMENT.md",
        "WAKE.md"
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
    
    # Inject tool usage note if tools directory exists with any .toml files
    tools_dir = workspace / "tools"
    if tools_dir.exists() and any(tools_dir.glob("*.toml")):
        prompt_parts.append("## Tools")
        prompt_parts.append(
            "Your operator sees a brief notice when you call a tool "
            "(tool name, success/failure, result size) — but **not** the actual "
            "content returned. Tool results are only visible to you. When you read "
            "a file, run a command, or get any tool result that the operator needs "
            "to see, include the relevant content in your response."
        )

    # Shared-room behavior guidance (harmless in DMs, essential in shared rooms)
    prompt_parts.append("## Shared Room Behavior")
    prompt_parts.append(
        "IMPORTANT: You share rooms with other agents. Each agent is a separate entity.\n\n"
        "Rules:\n"
        "1. You are ONE agent. Only write YOUR OWN words. NEVER write words, dialogue, or responses attributed to another agent or user.\n"
        "2. If asked to interact with another agent (e.g. 'give each other a compliment'), only provide YOUR part. The other agent will provide theirs.\n"
        "3. Do not simulate, predict, or write both sides of a conversation.\n"
        "4. If the message doesn't need your input, say nothing — silence is valid.\n"
        "5. Keep responses focused on what was asked of you specifically."
    )

    # Inject runtime workspace path only if not already mentioned in loaded files
    workspace_str = str(workspace.resolve())
    already_mentioned = any(workspace_str in part for part in prompt_parts)
    if not already_mentioned:
        prompt_parts.append("## Runtime")
        prompt_parts.append(f"Workspace: {workspace_str}")
        prompt_parts.append("Use this as the working directory for shell commands (pass as cwd).")

    # Join all parts with newlines and return
    return "\n".join(prompt_parts) if prompt_parts else ""
