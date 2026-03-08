# OpenAlph

Purpose-built multi-agent AI platform. Matrix rooms as sessions, one process per agent.

Named for the pistol shrimp (genus *Alpheus*) — tiny crustacean, outsized impact.

## Status

Phase 1: In Development

## Architecture

- **Process model:** One process per agent, managed by systemd
- **Interface:** Matrix (room = session = context = persistence)
- **Providers:** Anthropic SDK + OpenAI SDK (OpenRouter, Ollama)
- **Isolation:** Unix users per agent
- **Language:** Python
- **License:** MIT
