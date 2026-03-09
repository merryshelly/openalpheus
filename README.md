# OpenAlph

Purpose-built multi-agent AI platform. Matrix as transport, local state as persistence, one process per agent.

Named for the pistol shrimp (genus *Alpheus*) — tiny crustacean, outsized impact.

## Status

Phase 3.6: Session Persistence (in progress)

## Architecture

- **Process model:** One process per agent, managed by systemd
- **Communication:** Matrix (conduwuit) — rooms as sessions, membership as ACL
- **Persistence:** Local JSONL per session — canonical source of agent context state
- **Providers:** Anthropic SDK + OpenAI SDK (OpenRouter, Ollama)
- **Isolation:** Unix users per agent (Phase 4)
- **Language:** Python
- **License:** MIT
