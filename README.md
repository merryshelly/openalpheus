# OpenAlpheus

Self-hosted multi-agent harness for solo operators and small teams.

## What It Is

Named after the pistol shrimp (*Alpheus*) — a crustacean smaller than your thumb that generates shockwaves louder than a gunshot. 

Tiny, self-contained, disproportionately effective.

OpenAlpheus runs AI agents as isolated Unix processes on your hardware. Matrix provides the transport layer. Local JSONL files hold session state. systemd manages the lifecycle. Four direct Python dependencies. No cloud except the ones you explicitly choose.

**Status: v0.1.2 — early release.** Core is stable and tested (1,435+ tests). The interface may evolve.

## Why

All agent frameworks make tradeoffs. We optimized for:

- **Control.** Full control of the system prompt. Your agent doesn't read a single character you didn't put there. Behavior is configured by editing markdown files — no code required.
- **Ease of use.** `systemctl`, `journalctl`, `grep`, `nano` — operate agents with the same Linux tools people have used for decades.
- **Simplicity.** Each Matrix room is a session with your agent. One messaging protocol. Nine tools. For anything that's not a native tool, there's `shell`. No arcane message routing, no opaque session spawning.
- **Visibility.** All agent actions — tool calls, subagent dispatches, thinking blocks — surface in the chat history.
- **Maintainability.** ~9,000 LOC. Full test coverage. Four direct dependencies: `anthropic`, `openai`, `matrix-nio`, `mistune`.
- **Resilience.** Each agent runs as an isolated Unix process with its own filesystem. One agent can crash out, trash its workspace, and the others are unaffected.
- **Focus.** Matrix is a mature protocol with an array of clients for mobile, desktop, web. No bespoke UI, no custom views to maintain.
- **Transparency.** Session state is append-only text in JSONL, not a database. `grep` works. `cat` works. No migrations, no schema, no query language needed.

## Quick Start

One command takes a fresh Linux machine to a running agent:

```bash
curl -fsSL https://codeberg.org/merryshelly/openalpheus/raw/branch/main/install.sh | sudo bash
```

Non-interactive:

```bash
sudo OPENALPH_AGENT_NAME=myagent \
     OPENALPH_PROVIDER=anthropic \
     OPENALPH_API_KEY_FILE=/path/to/key \
     OPENALPH_DOMAIN=myhost.tailnet.ts.net \
     OPENALPH_TLS_MODE=tailscale \
     bash install.sh
```

The installer handles Python venv, tuwunel (Matrix homeserver) in Docker, TLS via Caddy, Matrix account creation, agent Unix user + workspace + systemd unit, and Cinny web client.

Connect with Cinny (web), Element (mobile), or any Matrix client. Create a room, invite the agent, start talking.

## What You Get

| Component | Detail |
|-----------|--------|
| Matrix homeserver | Tuwunel (Conduwuit fork), Docker-managed, registration locked after bootstrap |
| Operator account | Your Matrix identity on the homeserver |
| AI agent | Sandboxed Unix process, systemd-managed, dedicated user and workspace |
| TLS | Caddy — Tailscale (homelab), Let's Encrypt (VPS), or BYO cert |
| Web client | Cinny at your domain |
| Workspace | `/home/oa-<name>/workspace/` with template prompt files, skills, and tools |

## Architecture

### Process Model

One Unix process per agent. Each runs as a dedicated user in the `openalph` group, managed by a systemd template unit (`openalph@<name>.service`). Kernel-enforced isolation between agents.

### Communication

Matrix rooms are sessions. Room membership is the ACL — any room member can message the agent. Multiple agents can share a room; they respond only when @mentioned (context hydration on mention).

### Session State

Local append-only JSONL files are the source of truth. Matrix is transport only. On wake, the agent gap-fills from Matrix for messages missed while offline.

### Prompt Assembly

Six workspace markdown files assembled at process start:

**SAFETY → SOUL → OPERATOR → WAKE → ENVIRONMENT → OPERATIONS**

Skills are listed by name in the prompt; the agent reads their content on demand. No hardcoded behavior — everything lives in workspace files the operator controls.

### Tools

Enabled by placing `.toml` files in `workspace/tools/`. Empty file = tool enabled with defaults.

Built-in tools: `shell`, `file_read`, `file_write`, `file_edit`, `web_search`, `web_fetch`, `subagent`, `memory_search`, `send_media`.

### Providers

Anthropic (native SDK) and OpenAI-compatible (OpenRouter, vLLM, llama.cpp, etc). Multiple providers per agent. Switch models at runtime via `/model` or configure aliases in TOML.

### Memory

Hybrid semantic + keyword search over workspace files. Nomic-embed-text embeddings combined with BM25 ranking.

### Security

Credential redaction (10 pattern types) applied to all tool output before it enters agent context. Tool results wrapped with injection defense. Unix user isolation is kernel-enforced. Homeserver registration locked after bootstrap.

## Source Tree

```
src/openalph/
├── admin.py           Agent scaffolding (user, workspace, config, systemd)
├── agent.py           Agent loop, tool dispatch, streaming, circuit breaker
├── cli.py             CLI entry points
├── config.py          TOML config, API key resolution
├── heartbeat.py       Per-room recurring timers
├── matrix.py          Matrix client, sync, mention gating, slash commands
├── mention.py         Mention detection + room gating
├── prompt.py          System prompt assembly from workspace files
├── provider.py        Anthropic + OpenAI routing, streaming, error handling
├── session.py         Append-only JSONL, context rebuild, overflow
├── umbral.py          Recurring context rotation (archive + wipe + reset)
├── memory/
│   ├── chunker.py     Document chunking
│   ├── embeddings.py  Embedding generation
│   ├── indexer.py     Index construction
│   └── search.py      Hybrid semantic + keyword search
└── tools/
    ├── __init__.py    Registry, discovery, dispatch, truncation
    ├── file.py        file_read, file_write, file_edit
    ├── media.py       Send files to Matrix rooms
    ├── memory_search.py  Hybrid search tool
    ├── security.py    Credential redaction (10 patterns)
    ├── shell.py       Subprocess execution
    ├── subagent.py    Multi-turn sub-agent with tool access
    └── web.py         Web search (Brave) + fetch (HTML→text)
```

~9,000 LOC source. 1,435+ tests. 4 direct dependencies: `anthropic`, `openai`, `matrix-nio`, `mistune`.

## Prerequisites

- Debian 12+ or Ubuntu 22.04+ (amd64 or arm64)
- Python 3.11+ (pre-installed on Debian 12+ and Ubuntu 24.04+)
- An LLM API key (Anthropic recommended, or any OpenAI-compatible provider)
- TLS: Tailscale (homelab), a public domain (VPS), or your own cert/key pair

Docker, Caddy, and other system dependencies are auto-installed if missing.

See [INSTALL.md](INSTALL.md) for detailed requirements.

## Post-Bootstrap

| Task | How |
|------|-----|
| Customize identity | Edit `SOUL.md` in agent workspace |
| Configure behavior | Edit `OPERATIONS.md`, `ENVIRONMENT.md`, etc. |
| Add skills | Place markdown files in `workspace/skills/` |
| Add tools | Place `.toml` files in `workspace/tools/` |
| Add another agent | `sudo openalph new-agent <name>` |

All prompt changes require a restart: `sudo systemctl restart openalph@<name>`

See [INSTALL.md](INSTALL.md) for detailed post-bootstrap configuration.

## In-Room Commands

| Command | Effect |
|---------|--------|
| `/status` | Model, context usage, token counts |
| `/model <provider/model>` | Switch model for this room |
| `/thinking <off\|low\|medium\|high>` | Set extended thinking level |
| `/heartbeat start <interval>` | Start recurring timer (e.g., `5m`, `1h`) |
| `/heartbeat stop` | Stop heartbeat |
| `/umbral start <interval>` | Start recurring context rotation (min 30m) |
| `/umbral stop` | Stop context rotation |
| `/cache 1h` | Extended Anthropic prompt cache TTL |
| `/stop` | Cancel current processing |
| `/resume` | Re-enable after `/stop` |

> **Client note:** Some Matrix clients intercept `/` commands for their own features. If a slash command isn't reaching the agent:
>
> | Client | How to send |
> |--------|-------------|
> | **Cinny** | `/command` — works directly |
> | **Element Desktop** | `//command` — the first `/` escapes Element's own command handler |
> | **Element X** (mobile) | `/command` — works directly |

## CLI

```
openalph new-agent <name>    Create agent (user, workspace, config, systemd)
openalph run <name>          Run agent in foreground (debug)
openalph restart <name>      Restart agent
openalph status <name>       Show agent status
openalph logs <name>         Follow agent logs
openalph showprompt <name>   Display assembled system prompt
```

## Security

- **Process isolation:** Each agent runs as a dedicated Unix user. Kernel-enforced boundaries.
- **Credential redaction:** 10 pattern types (API keys, tokens, passwords, etc.) scrubbed from all tool output before reaching agent context.
- **Tool result wrapping:** Injection defense on all tool returns.
- **Locked registration:** Homeserver registration disabled after bootstrap. No open federation by default.
- **No telemetry:** Nothing leaves your machine except LLM API calls and Matrix federation (if you enable it).

## License

[AGPL-3.0](LICENSE)

**Repository:** [codeberg.org/merryshelly/openalpheus](https://codeberg.org/merryshelly/openalpheus)
