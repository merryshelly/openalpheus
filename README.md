# OpenAlph

Purpose-built multi-agent AI platform. Matrix as transport, local JSONL as canonical session state, one process per agent.

Named for the pistol shrimp (genus *Alpheus*) — tiny crustacean, outsized impact.

## Status

**Phase 5 complete.** 511 tests passing, ~3,300 LOC. Two test agents deployed on conduwuit.

### What's Shipped

| Phase | Feature | Status |
|-------|---------|--------|
| 1–3 | Core agent loop, providers, session persistence, tools | ✅ |
| 4 | Multi-agent: Unix users, systemd, CLI (`new-agent`, `run`, `start/stop/restart`, `status`, `list`, `logs`) | ✅ |
| 5.0 | Shared rooms: @mention gating, context hydration, per-room TOML overrides | ✅ |
| 5.1 | Emergency fallback: `openalph chat <agent>`, `openalph showprompt <agent>`, prompt refactor | ✅ |
| — | Per-room heartbeat timers (`/heartbeat start/stop/status`) | 🚧 In progress |

## Architecture

- **Process model:** One process per agent, managed by systemd (`openalph@<agent>.service`)
- **Communication:** Matrix (conduwuit) — rooms as sessions, membership as ACL
- **Persistence:** Local JSONL per session (`<workspace>/sessions/<room-id-safe>.jsonl`) — canonical source of agent context state. Matrix is transport only.
- **Context reconstruction:** Read local JSONL on wake; gap-fill from Matrix for any messages missed while offline
- **Providers:** Anthropic SDK + OpenAI SDK (OpenRouter, Ollama)
- **Isolation:** Unix users per agent, shared `openalph` group, per-agent workspaces
- **Prompt assembly:** Workspace files injected in safety-first order (SAFETY → SOUL → OPERATOR → WAKE → ENVIRONMENT → OPERATIONS). No hardcoded content — all behavior defined by workspace files.
- **Language:** Python 3.11+
- **License:** AGPL-3.0

## Source Tree

```
src/openalph/
├── admin.py          # new-agent scaffolding: user, workspace, config, systemd
├── agent.py          # Agent loop, tool dispatch, cancellation, circuit breaker (25 iter)
├── cli.py            # CLI: new-agent, run, start/stop/restart, status, list, logs, chat, showprompt
├── config.py         # TOML loader, AgentConfig + MatrixConfig, API key resolution
├── matrix.py         # Matrix client, sync loop, lazy wake, mention gating, commands, typing
├── mention.py        # Pure mention detection + room gating (no async/nio dependencies)
├── prompt.py         # System prompt assembly from workspace files + skills index
├── provider.py       # Anthropic + OpenAI routing, response normalization, tool schema conversion
├── session.py        # SessionLog: append-only JSONL, context-build, overflow handling
└── tools/
    ├── __init__.py   # Registry, discovery (workspace/tools/*.toml), dispatch, truncation
    ├── shell.py      # Stateless subprocess (explicit cwd/env/timeout, defaults to agent home)
    ├── file.py       # read/write/edit with path resolution
    ├── web.py        # search (Brave) + fetch (HTML→text)
    └── subagent.py   # Multi-turn sub-agent with parent's tools (minus subagent)
```

## How to Run

```bash
# Install
pip install -e .

# Create a new agent (interactive — creates Unix user, workspace, config, systemd unit)
sudo openalph new-agent myagent

# Start via systemd
sudo systemctl start openalph@myagent

# Or run in foreground (dev mode)
openalph run myagent

# Emergency CLI chat (no Matrix dependency)
openalph chat myagent

# View assembled system prompt + tools
openalph showprompt myagent
```

Minimal config (`/etc/openalph/agents/myagent.toml`):

```toml
[agent]
name = "myagent"
model = "claude-haiku-4-5-20251001"

[provider]
type = "anthropic"
api_key_cmd = "cat /home/oa-myagent/.config/anthropic-key"

[workspace]
path = "/home/oa-myagent/workspace"

[matrix]
homeserver = "http://matrix.local:6167"
user_id = "@myagent:matrix.local"
device_id = "OPENALPH"
access_token_cmd = "cat /home/oa-myagent/.config/matrix-token"
```

## Matrix Commands

| Command | Description |
|---------|-------------|
| `/stop` | Cancel in-flight agent work |
| `/status` | Show agent model, context usage, turn count |
| `/showprompt` | Display full assembled system prompt + tool list |
| `/reset` | Clear agent context for this room |
| `/heartbeat start <interval>` | Start recurring heartbeat (e.g. `6h`, `15m`) |
| `/heartbeat stop` | Stop heartbeat in current room |
| `/heartbeat status` | List all active heartbeats |

All slash commands bypass @mention gating in shared rooms.

## How to Test

```bash
pytest                        # all 511 tests
pytest tests/test_matrix.py   # Matrix integration
pytest tests/test_gating.py   # mention gating unit tests
pytest tests/test_matrix_gating.py  # gating integration
pytest -x                     # stop on first failure
```

## Repo

- **Codeberg:** https://codeberg.org/merryshelly/openalph (private)
- **Planning docs:** `~/.openclaw/workspace/memory/projects/openalph/`
