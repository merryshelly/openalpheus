#!/usr/bin/env python3
"""Minimal interactive chat — Phase 1 smoke test.

Usage:
    python example/chat.py example/agent.toml

Type messages, get responses. Ctrl-C to quit.
Shows token usage after each turn.
"""

import asyncio
import sys
from pathlib import Path

# Allow running from repo root without pip install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from openalph.config import load_config
from openalph.agent import Agent


async def main():
    if len(sys.argv) < 2:
        print("Usage: python example/chat.py <config.toml>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    config = load_config(config_path)

    print(f"Loaded agent: {config.name} ({config.model} via {config.provider})")
    print(f"Workspace: {config.workspace}")
    print()

    agent = Agent(config)

    if agent.system_prompt:
        # Show how many workspace files were loaded
        file_count = agent.system_prompt.count("## ") 
        print(f"System prompt assembled ({file_count} sections, {len(agent.system_prompt)} chars)")
    else:
        print("No workspace files found — system prompt is empty.")
    print()
    print("Type a message (Ctrl-C to quit):")
    print("-" * 40)

    while True:
        try:
            user_input = input("\nyou> ")
        except (KeyboardInterrupt, EOFError):
            print("\n")
            break

        if not user_input.strip():
            continue

        try:
            response = await agent.handle_input(user_input)
            print(f"\n{config.name}> {response}")

            status = agent.status()
            print(f"\n  [{status['turns']} turns | "
                  f"in: {status['total_input_tokens']} "
                  f"out: {status['total_output_tokens']} tokens]")
        except Exception as e:
            print(f"\nError: {e}")

    # Final status
    status = agent.status()
    print(f"Session: {status['turns']} turns, "
          f"{status['total_input_tokens']} in / "
          f"{status['total_output_tokens']} out tokens")


if __name__ == "__main__":
    asyncio.run(main())
