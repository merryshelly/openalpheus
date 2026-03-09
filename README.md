# OpenAlph

Purpose-built multi-agent AI platform. Matrix as transport, local JSONL as canonical session state, one process per agent.

Named for the pistol shrimp (genus *Alpheus*) — tiny crustacean, outsized impact.

## Status

**Phase 3.6 complete.** 276 tests passing. Local JSONL session persistence shipped; journal rooms removed. Phase 4 (multi-agent: Unix users, systemd, CLI) is next.

Live on conduwuit (`@merry-dev:matrix.local`).

## Architecture

- **Process model:** One process per agent, managed by systemd (Phase 4)
- **Communication:** Matrix (conduwuit) — rooms as sessions, membership as ACL
- **Persistence:** Local JSONL per session (`<workspace>/sessions/<room-id-safe>.jsonl`) — canonical source of agent context state. Matrix is transport only.
- **Context reconstruction:** Read local JSONL on wake; gap-fill from Matrix for any messages missed while offline
- **Providers:** Anthropic SDK + OpenAI SDK (OpenRouter, Ollama)
- **Isolation:** Unix users per agent (Phase 4)
- **Language:** Python 3.11+
- **License:** AGPL-3.0

## Source Tree

```
src/openalph/
├── config.py         # TOML loader, AgentConfig + MatrixConfig, API key resolution
├── provider.py       # Anthropic + OpenAI routing, response normalization, tool schema conversion
├── prompt.py         # System prompt assembly from workspace files + skills index
├── agent.py          # Agent loop, tool dispatch, cancellation, token tracking
├── session.py        # SessionLog: local JSONL append/read/gap-fill/context-build
├── matrix.py         # Matrix client, sync loop, lazy room activation, /stop, /status
└── tools/
    ├── __init__.py   # Registry, discovery (workspace/tools/*.toml), dispatch, truncation
    ├── shell.py      # Stateless subprocess (explicit cwd/env/timeout, defaults to agent home)
    ├── file.py       # read/write/edit with path resolution
    ├── web.py        # search (Brave) + fetch (HTML→text)
    └── subagent.py   # Multi-turn sub-agent with parent's tools (minus subagent, preventing recursion)
```

## How to Run

```bash
# Install dependencies
pip install -e .

# Run agent (foreground / dev mode)
python -m openalph --config path/to/agent.toml
```

Minimal config (`agent.toml`):

```toml
[agent]
name = "merry"
model = "claude-opus-4-6"

[provider]
type = "anthropic"
api_key_cmd = "cat ~/.config/anthropic-key"

[workspace]
path = "/home/merryshelly/.openclaw/workspace"

[matrix]
homeserver = "http://matrix.local:6167"
user_id = "@merry:matrix.local"
device_id = "OPENALPH"
access_token_cmd = "cat ~/.config/matrix-token"
```

## How to Test

```bash
pytest                        # all 276 tests
pytest tests/test_session.py  # session persistence unit tests
pytest tests/test_matrix.py   # Matrix integration tests
pytest -x                     # stop on first failure
```

## Repo

- **Codeberg:** https://codeberg.org/merryshelly/openalph (private)
- **Planning docs:** `~/.openclaw/workspace/memory/projects/openalph/`
