# OpenAlpheus

Self-hosted multi-agent harness for solo operators and small teams.

## What It Is

Named after the pistol shrimp (*Alpheus*) — a crustacean smaller than your thumb that generates shockwaves louder than a gunshot. 

Tiny, self-contained, disproportionately effective.

OpenAlpheus runs AI agents as isolated Unix processes on your hardware. Matrix provides the transport layer. Local JSONL files hold session state. systemd manages the lifecycle. Five direct Python dependencies. No cloud except the ones you explicitly choose.

**Status: v0.1.2 — early release.** Core is stable and tested (2,650+ tests). The interface may evolve.

## Why

All agent frameworks make tradeoffs. We optimized for:

- **Control.** Full control of the system prompt. Every instruction your agent receives lives in a markdown file you can read and edit — including the tool-result security footer (`SECURITY_FOOTER.md`), which you can rewrite or switch off entirely with `injection_defense = false`. Behavior is configured by editing markdown files — no code required. The framework adds exactly two things beyond your files, both mechanical rather than behavioural: a `## Runtime` line naming the workspace path, and a `## Model Aliases` table generated from the aliases in your own config.
- **Ease of use.** `systemctl`, `journalctl`, `grep`, `nano` — operate agents with the same Linux tools people have used for decades.
- **Simplicity.** Each Matrix room is a session with your agent. One messaging protocol. Sixteen tools. For anything that's not a native tool, there's `shell`. No arcane message routing, no opaque session spawning.
- **Visibility.** All agent actions — tool calls, subagent dispatches, thinking blocks — surface in the chat history.
- **Maintainability.** ~16,900 LOC source. ~85% line coverage (2,650+ tests). Five direct dependencies: `anthropic`, `openai`, `matrix-nio`, `mistune`, `httpx`. Semantic memory search adds two more (`llama-cpp-python`, `sqlite-vec`) as the optional `[memory]` extra, installed by default by `install.sh`.
- **Resilience.** Each agent runs as an isolated Unix process with its own filesystem. One agent can crash out, trash its workspace, and the others are unaffected.
- **Focus.** Matrix is a mature protocol with an array of clients for mobile, desktop, web. No bespoke UI, no custom views to maintain.
- **Transparency.** Session state is append-only text in JSONL, not a database. `grep` works. `cat` works. No migrations, no schema, no query language needed.

## Quick Start

One command takes a fresh Linux machine to a running agent:

```bash
curl -fsSL https://github.com/merryshelly/openalpheus/raw/refs/heads/main/install.sh | sudo bash
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

Skills are listed by name in the prompt; the agent reads their content on demand. No hardcoded behavior — every behavioural instruction lives in one of the seven workspace files the operator controls (`SAFETY.md`, `SOUL.md`, `OPERATOR.md`, `WAKE.md`, `ENVIRONMENT.md`, `OPERATIONS.md`, `SECURITY_FOOTER.md`). Run `openalph prompt <agent>` to see exactly what the agent receives.

### Tools

Enabled by placing `.toml` files in `workspace/tools/`. Empty file = tool enabled with defaults.

Built-in tools: `shell`, `file_read`, `file_write`, `file_edit`, `file_patch`, `web_search`\*, `web_fetch`, `web_fetch_js`\*\*, `grep`, `glob`, `subagent`, `advisor`, `memory_search`, `send_media`, `view_image`, `context_status`, `todo_write`, `heartbeat`.

\*`web_search` requires a [Brave Search API key](https://brave.com/search/api/) configured in `workspace/tools/web_search.toml`. Without it, the tool is available but returns an error. `web_fetch` (direct URL fetching) works without any API key.

\*\*`web_fetch_js` renders JavaScript-heavy pages (SPAs, dashboards, infinite-scroll) via [Tabstack](https://tabstack.ai)'s cloud browser — opt-in via `api_key` in `workspace/tools/web_fetch_js.toml` (unconfigured = tool unavailable). Unlike `web_fetch`, the target URL and page content transit a third-party cloud, so treat it as a deliberate exception to "no cloud except the ones you explicitly choose." `web_fetch` nudges toward it at the point of need when a fetch looks unrendered.

### Vision

Vision is a **model-level capability**, not an agent-level one. When someone posts an image to a room, the harness records a `[media: path (mime, size)]` tag; whether that tag expands into base64 image blocks is decided per room, per *active model*, by a three-layer resolver (`model_supports_vision` in `provider.py`):

1. `[model_vision]` TOML override — exact per-model `true`/`false` (operator escape hatch in both directions),
2. a curated capabilities table in `provider.py` (same fragment convention as context-window resolution),
3. **fail-closed `false`** with a one-time warning — uncharacterized models never receive image data.

This makes `/model` switches safe in both directions: posting an image to a room running a blind model passes the tag through as plain text, and switching models mid-session is blocked only when the history contains images *and* the target model can't see. Supported image types: JPEG, PNG, GIF, WebP. Images count toward context at roughly 1 token per 750 raw bytes.

Agents can also self-serve images with the `view_image` tool: it validates a workspace image (containment, type, 5 MB default cap, active-model vision gate), then stages it for injection as a framed user message at the next tool-loop boundary — never as a tool-result image block, which only Anthropic accepts. Subagents get their own vision inbox; images land in the sub's context, not the parent's.

### Guidance injection

Optional, config-aware in-stream guidance that helps agents stay on track during long turns — without touching the system prompt, and fully visible.

- **System reminders.** A per-room engine evaluates deterministic triggers against harness-observable state (iteration count, context depth, memory-search history, todo state) and injects `<system-reminder>` messages at tool-call boundaries. Six triggers: `todo-nudge` (5 tool calls, no active plan), `context-pressure` (≥80% of the window — converge), `memory-salience` (turn ≥2, zero `memory_search` calls), `iteration-budget` (80% of the tool-call budget), `memory-salience-deep` (≥50K absolute context tokens, zero `memory_search` calls — catches deep single-turn sessions the turn-based trigger structurally cannot see), `advisor-salience` (≥75K absolute tokens, zero `advisor` consults, interactive sessions only). No NLP, no model calls. Every injected byte is durable in the session JSONL and surfaces as a collapsed Matrix notice. Disable per agent with `reminders = false` in `[agent]`.
- **`todo_write` tool.** A session-scoped working-memory task list. The harness stays tracker-agnostic — promoting todos to a durable tracker is an operator convention, not a dependency.
- **Read-before-write guard.** `file_write` refuses to overwrite an existing file not read this session (or changed on disk since) — blind overwrites are structurally prevented, not just discouraged. Opt out with `require_read_before_write = false` in `workspace/tools/file_write.toml`.
- **Rich tool descriptions.** All built-in tools carry prompt-engineered descriptions (purpose, constraints, when-not) plus steering in error/truncation returns — passed via the API `tools` parameter, never injected into the system prompt.
- **Multi-hunk patching + validated edits.** `file_patch` applies several SEARCH/REPLACE hunks to a file in one atomic, all-or-nothing call. `file_edit`, `file_write`, and `file_patch` run a zero-dependency syntax check (Python / JSON / TOML) before writing and reject an edit that would turn a clean file broken; writes commit atomically (temp file + rename).
- **Bounded search.** `grep` (regex over file contents) and `glob` (filename patterns) are pure-Python and rooted at the workspace by default, with file-count and byte caps, a per-scan time budget that defends against catastrophic-backtracking (ReDoS) patterns, and no-follow handling of symlinks. Filesystem containment is OS-enforced (Unix user + systemd sandbox — see SECURITY.md §3), not an app-level path check.
- **`advisor` tool.** A built-in second opinion: the calling agent can hand its own transcript to a separate (often stronger) model mid-task and get judgment back — no tools of its own, capped uses per room, useful before a non-obvious design decision, a first substantive write, or declaring complex work done. Advice is guidance, not a directive; the calling agent stays responsible for the outcome.

### Providers

Anthropic (native SDK) and OpenAI-compatible (OpenRouter, vLLM, llama.cpp, etc). Multiple providers per agent. Switch models at runtime via `/model` or configure aliases in TOML.

Any OpenAI-compatible endpoint works with no core changes, so community shims can bridge other backends — e.g. codex-sidecar adapts a subscription-backed model into a provider.

### Memory

Hybrid semantic + keyword search over workspace files. Nomic-embed-text embeddings combined with BM25 ranking. The semantic half needs the `[memory]` extra and a local GGUF model; without them search degrades to keyword-only (BM25) and says so in-room on the first search, rather than degrading silently.

### Security

Credential redaction (10 pattern types) applied to all tool output before it enters agent context. Tool results wrapped with injection defense. Literal `<system-reminder>` markup in untrusted tool and user content is escaped so it cannot forge harness guidance. Unix user isolation is kernel-enforced. Homeserver registration locked after bootstrap.

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
├── reminders.py       Deterministic system-reminder engine (state-triggered guidance)
├── session.py         Append-only JSONL, context rebuild, overflow
├── umbral.py          Recurring context rotation (archive + wipe + reset)
├── memory/
│   ├── chunker.py     Document chunking
│   ├── embeddings.py  Embedding generation
│   ├── indexer.py     Index construction
│   ├── schema.py      SQLite schema + sqlite-vec loading
│   └── search.py      Hybrid semantic + keyword search
└── tools/
    ├── __init__.py    Registry, discovery, dispatch, truncation
    ├── advisor.py     Second-model consult (client-side, no tools of its own)
    ├── file.py        file_read, file_write, file_edit, file_patch
    ├── media.py       Send files to Matrix rooms
    ├── memory_search.py  Hybrid search tool
    ├── search.py      grep + glob (bounded, pure-stdlib file search)
    ├── security.py    Credential redaction (10 patterns)
    ├── shell.py       Subprocess execution
    ├── subagent.py    Multi-turn sub-agent with tool access
    ├── validate.py    Syntax check on file_edit/file_write/file_patch (fails open)
    └── web.py         web_search (Brave) + web_fetch (HTML→text) + web_fetch_js (Tabstack)
```

~16,900 LOC source. 2,650+ tests. 5 direct dependencies: `anthropic`, `openai`, `matrix-nio`, `mistune`, `httpx` — plus the optional `[memory]` extra (`llama-cpp-python`, `sqlite-vec`) for semantic search.

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
| `/model <provider/model\|list>` | Switch model for this room, or list configured aliases |
| `/effort <off\|low\|medium\|high\|xhigh\|max>` | Set reasoning effort level for this room (`xhigh`/`max` require supported models; no argument shows the current level) |
| `/heartbeat start <interval> [directive]` | Start recurring timer (e.g., `5m`, `1h`); optional trailing directive becomes the turn content on each fire (else a WAKE pointer) |
| `/heartbeat schedule "<cron>" [directive]` | Start a calendar-scheduled heartbeat (5-field cron in quotes, e.g., `"30 6 * * 1-5"` = 06:30 weekdays; host-local timezone shown in status; min gap 5m); optional trailing directive as above |
| `/heartbeat stop` | Stop heartbeat |
| `/heartbeat status` | List active heartbeats across rooms |
| `/umbral start <interval> [directive]` | Start recurring context rotation (min 30m); optional trailing directive is re-injected as the turn content each cycle |
| `/umbral schedule "<cron>" [directive]` | Start a calendar-scheduled context rotation (5-field cron in quotes, e.g., `"0 20 * * 0"` = Sundays 20:00; host-local timezone; min gap 30m); directive re-injected each cycle |
| `/umbral stop` | Stop context rotation |
| `/umbral status` | List active umbral timers across rooms |
| `/cache <1h\|5m\|off>` | Anthropic prompt cache TTL (default `1h`; no argument shows current TTL + toolstrip state) |
| `/cache toolstrip` | Reclaim context by replacing old tool outputs with placeholders |
| `/timesense <on\|off>` | Prepend timestamp to every user message in LLM context (off by default; no argument shows current state) |
| `/steer <message>` | Inject a mid-turn steering note into the **active** turn (real-time steering). Logged + delivered to the agent at the next tool-call boundary as a user message. Requires an active turn; deposits without interrupting. |
| `/showprompt` | Display the assembled system prompt + tool list, delivered as a Markdown file attachment |
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
openalph new-agent <name>         Create agent (user, workspace, config, systemd)
openalph start <name|all>         Start agent(s)
openalph stop <name|all>          Stop agent(s)
openalph restart <name|all>       Restart agent(s)
openalph status [name]            Show systemd status (all agents if omitted)
openalph list                     List configured agent names
openalph logs <name> [-f]         Follow agent logs (journalctl)
openalph run <name>               Run agent in foreground (debug)
openalph monitor <name>           Live-tail an agent's JSONL session log, formatted
openalph chat <name> [--room ID]  Local interactive session, no Matrix — in-memory
                                   only, history lost on exit; quick debugging
openalph showprompt <name>        Display the assembled system prompt + tool list
```

## Security

- **Process isolation:** Each agent runs as a dedicated Unix user. Kernel-enforced boundaries.
- **Credential redaction:** 10 pattern types (API keys, tokens, passwords, etc.) scrubbed from all tool output before reaching agent context.
- **Tool result wrapping:** Injection defense on all tool returns.
- **Locked registration:** Homeserver registration disabled after bootstrap. No open federation by default.
- **No telemetry:** Nothing leaves your machine except LLM API calls and Matrix federation (if you enable it).

## License

[AGPL-3.0](LICENSE)

**Repository:** [github.com/merryshelly/openalpheus](https://github.com/merryshelly/openalpheus) (public mirror)

**Canonical:** [forgejo-proxmox.quetzal-moth.ts.net/merryshelly/openalpheus](https://forgejo-proxmox.quetzal-moth.ts.net/merryshelly/openalpheus) (self-hosted Forgejo, tailnet-only)
