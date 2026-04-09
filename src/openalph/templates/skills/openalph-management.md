<!-- Shipped with OpenAlph. Customize for your setup. -->
# OpenAlph Management

Manage OpenAlph agents and the platform. Use when: starting, stopping, or restarting agents; modifying agent configs or prompts; deploying new agents; working on OpenAlph source code; debugging agent behavior; or any operational task involving the OpenAlph platform.

## Contents

- [Architecture First Principles](#architecture-first-principles) — Room=session, JSONL state, isolation, orchestrator privileges
- [System Prompt Lifecycle](#system-prompt-lifecycle) — Assembly, change workflow, restart requirements
- [Tool Discovery](#tool-discovery) — Enabling tools via workspace `.toml` files
- [Heartbeat System](#heartbeat-system) — Start/stop, orphaned heartbeats
- [Configuration](#configuration) — Agent TOML structure, change workflow
- [Agent Management Commands](#agent-management-commands) — Status, start/stop, logs, showprompt, dev mode, new-agent
- [Common Operations](#common-operations) — Model changes, adding skills, prompt edits, health checks
- [File Permissions](#file-permissions)
- [Beads Task Tracker](#beads-task-tracker)
- [Troubleshooting](#troubleshooting) — ContextOverflow, not responding, tool failures, orphaned heartbeats
- [Quick Reference](#quick-reference) — Command cheat sheet
- [Deep Reference](#deep-reference) — Links to architecture docs and project files

## Architecture First Principles

### Room = Session

Each Matrix room corresponds to exactly one session. Sessions are persisted to JSONL files in `workspace/sessions/`. Context grows until it hits the model limit, at which point a `ContextOverflowError` fires (hard error, not silent truncation). To start fresh: create a new Matrix room.

### JSONL = Canonical State

The local JSONL file (`workspace/sessions/<room-id-safe>.jsonl`) is the source of truth for session history. Matrix is used for message delivery and room membership, but the JSONL is what the agent actually loads on wake and appends to on every turn.

### Room Membership = ACL

Room membership IS the access control list. Anyone in the room can trigger the agent. Be mindful of who you invite to agent rooms.

### Agent Isolation via Unix Users

Each agent runs as a dedicated Unix user (`oa-<name>`) in the `openalph` group. Home directories are mode 750. Agents cannot read each other's workspaces, credentials, or sessions. This is kernel-enforced sandboxing—not a bug, the security model working correctly.

**What agents can access:**
- Their own home: `/home/oa-<name>/`
- Their workspace: `/home/oa-<name>/workspace/`
- Shared resources: `/srv/openalph/shared/` (mode 2770, setgid `openalph` group)
- Standard system paths

**What agents cannot access:**
- Their own config at `/etc/openalph/agents/<name>.toml` (root-owned)
- Other agents' workspaces
- The OpenAlph source at `/opt/openalph/`

### Orchestrator Privileges (Merry Only)

The orchestrator has a differentiated security profile from domain agents. The orchestrator's systemd unit override (`/etc/systemd/system/openalph@<orchestrator>.service.d/override.conf`) disables the hardening restrictions that domain agents keep:

- `NoNewPrivileges=no` — allows sudo escalation
- `ProtectHome=no` — allows access to all home directories
- `ProtectSystem=no` — allows writes to system paths
- `PrivateTmp=no` — shares host `/tmp`

**Sudo access:** Each agent with sudo has it configured via `/etc/sudoers.d/oa-<agent>`. Password stored in 1Password (`op://<your-vault>/oa-<agent> sudo/password` — check ENVIRONMENT.md), resolved at runtime — never cached on disk.

```bash
export OP_SERVICE_ACCOUNT_TOKEN=$(cat ~/.config/op/service-account-token)
op read "op://<your-vault>/oa-<agent> sudo/password" | sudo -S <command>
```

**Revocation:** Operator can revoke sudo remotely by removing the password item from the 1Password vault. No host access or sudoers edit required.

**Domain agents keep full hardening.** This override applies only to the orchestrator unit.

## System Prompt Lifecycle

System prompts are assembled **once at process start** from 6 workspace files and held in memory:

1. `SAFETY.md` — Hard safety rules
2. `SOUL.md` — Identity, persona, tone
3. `OPERATOR.md` — About the operator (you)
4. `WAKE.md` — Heartbeat/wake instructions
5. `ENVIRONMENT.md` — Paths, credentials, capabilities
6. `OPERATIONS.md` — Procedures, shared room behavior

Missing files are silently skipped. Skills are listed by name only (not injected). The agent must `file_read` a skill to load its content.

**Critical:** Prompt changes require a process restart. New rooms alone are NOT sufficient—the same process holds the same `self.system_prompt` in memory.

### Prompt Change Workflow

1. Edit the workspace file(s) in `/home/oa-<name>/workspace/`
2. Restart the agent process: `sudo systemctl restart openalph@<name>`
3. Verify: `openalph showprompt <name>` or `/showprompt` in Matrix

What restart does:
- ✅ Reassembles system prompt from current workspace files
- ✅ All rooms (existing and new) get the new prompt
- ⚠️ Existing rooms retain old conversation history shaped by the old prompt
- ⚠️ A new room without restart still uses the old prompt (prompt lives in process memory, not per-room)

Best practice: Restart is mechanically sufficient. A new room after restart gives a clean slate (no old history influenced by old prompt) but is optional.

## Tool Discovery

Tools are enabled by placing `.toml` files in `workspace/tools/`:

```
workspace/tools/
├── shell.toml
├── file_read.toml
├── file_write.toml
├── file_edit.toml
├── web_search.toml
├── web_fetch.toml
└── subagent.toml
```

Empty files are valid—they enable the tool with defaults. No tools directory (or empty directory) means the agent has NO tools and will hallucinate XML tool calls.

Override config as needed:
```toml
# workspace/tools/shell.toml
[config]
default_timeout = 60
```

## Heartbeat System

Heartbeats are NOT configured in TOML. They are started in-room and persisted to JSON.

- Start: `/heartbeat start <interval>` in a Matrix room
- Stop: `/heartbeat stop`
- Status: `/heartbeat status`
- Minimum interval: 5 minutes
- Persisted to: `workspace/heartbeats.json`
- Auto-resumed on process restart

On context overflow, heartbeats auto-stop for that room with a warning.

**Orphaned heartbeat prevention:** Create agent rooms as public so the operator can rejoin after accidental departure. Always `/heartbeat stop` before leaving a room with an active heartbeat.

### Fixing Orphaned Heartbeats

If you leave a room with an active heartbeat and can't rejoin:

1. Stop the agent: `sudo systemctl stop openalph@<name>`
2. Edit: `sudo file_edit /home/oa-<name>/workspace/heartbeats.json`
3. Remove the orphaned room entry (or set to `[]` to clear all)
4. Restart: `sudo systemctl start openalph@<name>`

## Configuration

Agent configs live at `/etc/openalph/agents/<name>.toml`. They are **operator-managed, not agent-managed.**

- Owned by root, readable by the orchestrator user
- Agents cannot read or write their own configs
- All config changes require `sudo`

### Config Structure

```toml
[agent]
name = "<name>"
default_model = "anthropic/claude-sonnet-4"
max_tokens = 8192
model_max_tokens = 200000
max_iterations = 50
truncation_limit = 50000
vision = false

[providers.anthropic]
type = "anthropic"
api_key_cmd = "cat /home/oa-<name>/.config/anthropic/key"
cache_bust_notices = true    # optional, default false

[providers.openrouter]
type = "openai"
base_url = "https://openrouter.ai/api/v1"
api_key_cmd = "cat /home/oa-<name>/.config/openrouter/key"

[matrix]
homeserver = "https://matrix.example.com"
user_id = "@<name>:example.com"
access_token_cmd = "cat /home/oa-<name>/.config/matrix/token"

[matrix.rooms."!abc:server"]
require_mention = true
```

Key points:
- `api_key_cmd` runs via shell with 10s timeout
- `[matrix.rooms]` is for overrides only—NOT a room allowlist
- Agents join any room they're invited to
- `cache_bust_notices` (optional, default `false`): when `true`, emits an `m.notice` to the room when `cache_creation_tokens >= 10k` on any API call through that provider. Useful for Anthropic (dollar cost) and local inference (TTFT cost). Notice is logged as `role: "system"` in JSONL — never enters agent context.

### Config Change Workflow

1. Read current config:
   ```bash
   sudo file_read /etc/openalph/agents/<name>.toml
   ```
2. Edit the config (requires sudo):
   ```bash
   sudo file_edit /etc/openalph/agents/<name>.toml
   ```
3. Restart the agent:
   ```bash
   sudo systemctl restart openalph@<name>
   ```
4. Verify:
   ```bash
   sudo systemctl status openalph@<name>
   ```

## Agent Management Commands

### Status
```bash
sudo systemctl status openalph@<name>          # Single agent
sudo systemctl list-units 'openalph-*'          # All agent units
systemctl --user list-units 'openalph-*'        # Wrong—units are system-level
```

### Start / Stop / Restart
```bash
sudo systemctl restart openalph@<name>     # Direct
openalph restart <name>                     # CLI wrapper (same thing)
```

The `openalph` CLI is a thin wrapper around `systemctl`. Both are equivalent.

⚠️ Restart produces a cosmetic `CancelledError` traceback in the journal. This is normal (SIGTERM cancels sync_forever). No data loss.

### Logs
```bash
sudo journalctl -u openalph@<name> -f                   # Follow live
sudo journalctl -u openalph@<name> --since "5 min ago"  # Recent
openalph logs <name>                                     # CLI wrapper
```

### Show Assembled Prompt
```bash
openalph showprompt <name>
```
Or in Matrix: `/showprompt`

### Dev Mode (Foreground)
```bash
openalph run <name>    # Foreground, no systemd, for debugging
```

### Create New Agent
```bash
sudo openalph new-agent <name>
```

This creates:
- Unix user `oa-<name>` in `openalph` group
- Home directory with workspace structure
- Default `OPERATIONS.md` with tool visibility notes
- Config skeleton at `/etc/openalph/agents/<name>.toml` (edit the CHANGE_ME placeholders)
- Systemd unit (enabled but NOT started)

After running `new-agent`:
1. Edit the config: `sudo file_edit /etc/openalph/agents/<name>.toml`
2. Set API keys (usually via `api_key_cmd` pointing to files)
3. Start the agent: `sudo systemctl start openalph@<name>`
4. Invite to Matrix room and verify

## Common Operations

### Changing an Agent's Model

1. Read model roster:
   ```bash
   file_read /srv/openalph/shared/docs/model-roster.md
   ```
2. Update TOML: change `default_model` AND `model_max_tokens` together (they must match)
3. Restart: `sudo systemctl restart openalph@<name>`
4. Verify: `/status` in Matrix shows new model

### Adding a Skill to an Agent

1. Write the skill file:
   ```bash
   sudo file_write /home/oa-<name>/workspace/skills/<skill-name>.md
   ```
2. Restart agent or start new room (prompt reassembly picks up skill name)
3. Agent must `file_read` the skill to load content (only names are in prompt)

### Modifying Agent Prompts

1. Edit files in `/home/oa-<name>/workspace/` (SOUL.md, OPERATIONS.md, etc.)
2. Restart the agent: `sudo systemctl restart openalph@<name>`
3. Verify: `openalph showprompt <name>` or `/showprompt` in Matrix

### Checking Agent Health

1. `sudo systemctl status openalph@<name>` — process running?
2. `sudo journalctl -u openalph@<name> --since "5 min ago"` — any errors?
3. `/status` in Matrix — model, context %, token counts
4. Check session JSONL: `file_read /home/oa-<name>/workspace/sessions/<room-id-safe>.jsonl`

## File Permissions

When editing agent workspace files as the orchestrator:

```bash
# Files created by the orchestrator may need group write for the agent:
sudo chown oa-<name>:openalph /home/oa-<name>/workspace/<file>
sudo chmod 640 /home/oa-<name>/workspace/<file>

# Or write as the agent user:
sudo -u oa-<name> tee /home/oa-<name>/workspace/<file> << 'EOF'
content
EOF
```

Shared resources at `/srv/openalph/shared/` are setgid `openalph` (mode 2770). Files created there inherit the group.

## Beads Task Tracker

Beads (`bd` CLI) tracks tasks. Database at `/srv/openalph/shared/beads/`.

```bash
/srv/openalph/shared/bin/bd ready              # Actionable work
/srv/openalph/shared/bin/bd list               # Everything open
/srv/openalph/shared/bin/bd show <id>          # Full details
/srv/openalph/shared/bin/bd create "title" -p P2 -t task --parent <epic>
/srv/openalph/shared/bin/bd update <id> -s in_progress
/srv/openalph/shared/bin/bd close <id>
```

Priorities: P1=urgent, P2=normal, P3=low, P4=icebox.

Every task must have a parent epic. Query children with `bd children <id> --pretty`.

## Troubleshooting

### ContextOverflowError

The agent's context exceeded `model_max_tokens`. Solutions:
1. Start a new Matrix room (fresh session)
2. Increase `model_max_tokens` in config (if model supports it)
3. Reduce context by using subagents for large tasks

### Agent Not Responding

1. Check if process is running: `sudo systemctl status openalph@<name>`
2. Check recent logs: `sudo journalctl -u openalph@<name> --since "5 min ago"`
3. Check if room requires mention (gated rooms): look for `[matrix.rooms."<room-id>"]` with `require_mention = true`
4. Verify the agent is in the room (Matrix member list)

### Tool Calls Failing

1. Check tools directory exists: `file_read /home/oa-<name>/workspace/tools/`
2. Verify tool TOML files exist (even empty ones enable the tool)
3. Check for syntax errors in custom tool configs

### Cache Warnings

If `cache_bust_notices = true` on a provider, the agent emits in-room notices like:

> ⚠️ Cache warning — 72,198 tokens written, 0 read (100% uncached)

**Common causes:**
- **TTL expiry (>5 min gap):** The Anthropic prompt cache has a 5-minute TTL. Gaps between turns (idle time, long subagent runs) evict the cache. The TTL refreshes on each hit, but once it expires, the entire prefix must be re-cached.
- **Room/session change:** Each room has its own conversation history, so switching rooms always means a cold cache.
- **Post-restart prompt change:** After a process restart with modified system prompts, the prefix hash changes and nothing matches.
- **Large content ingestion:** Reading a big file or getting a large tool result adds a lot of *new* content to cache — this is normal operation, not a cache failure. Use the % and your judgment.
- **20-block lookback exceeded:** Very long tool loops (>10 consecutive tool iterations) can push the prior cache write outside Anthropic's 20-block lookback window.

**Mitigation:** For build sessions with expected long subagent runs or pauses, use `/cache 1h` to extend the Anthropic cache TTL to 1 hour (2x base input cost for writes, but avoids repeated full re-caches). Breakeven: ~0.33 busts per session — if there's even a 33% chance of one bust, the 1h TTL pays for itself. Use `/cache 5m` or `/cache off` to revert.

### Orphaned Heartbeat

See "Fixing Orphaned Heartbeats" in the Heartbeat System section above.

## Quick Reference

| Task | Command |
|------|---------|
| Create agent | `sudo openalph new-agent <name>` |
| Start agent | `sudo systemctl start openalph@<name>` |
| Stop agent | `sudo systemctl stop openalph@<name>` |
| Restart agent | `sudo systemctl restart openalph@<name>` |
| Agent status | `sudo systemctl status openalph@<name>` |
| View logs | `sudo journalctl -u openalph@<name> -f` |
| Show prompt | `openalph showprompt <name>` |
| Run foreground | `openalph run <name>` |
| Edit config | `sudo file_edit /etc/openalph/agents/<name>.toml` |
| Model roster | `file_read /srv/openalph/shared/docs/model-roster.md` |
| Beads tasks | `/srv/openalph/shared/bin/bd ready` |

### In-Room Slash Commands

| Command | Effect |
|---------|--------|
| `/status` | Show model, context usage, token counts |
| `/showprompt` | Display assembled system prompt |
| `/model <provider/model>` | Switch model for this room (persisted) |
| `/thinking <off\|low\|medium\|high>` | Set thinking level for this room (persisted) |
| `/cache` | Show current cache TTL for this room |
| `/cache 1h` | Set 1-hour Anthropic prompt cache TTL (persisted, room-scoped) |
| `/cache 5m` or `/cache off` | Revert to default 5-minute TTL |
| `/stop` | Cancel current processing and halt the room |
| `/resume` | Re-enable a halted room |
| `/heartbeat start <interval>` | Start periodic heartbeat (e.g., `5m`, `1h`) |
| `/heartbeat stop` | Stop heartbeat for this room |

All overrides (`/model`, `/thinking`, `/cache`) are room-scoped, persisted to JSONL, and restored on session resume. They self-clean when the session ends.

## Deep Reference

- Architecture: `file_read memory/projects/openalph/architecture-summary.md`
- Deployment gotchas: `file_read memory/projects/openalph/first-contact-gotchas.md`
- Model configs: `file_read /srv/openalph/shared/docs/model-roster.md`
- Project overview: `file_read memory/projects/openalph/README.md`
