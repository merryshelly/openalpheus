<!-- Shipped with OpenAlph. Customize for your setup. -->
# Agent Deployment

Procedure for standing up a new agent on OpenAlph. Use this skill when deploying, setting up, or standing up a new agent from a finalized spec.

## Contents

- [Bead-Tracked Deployment](#bead-tracked-deployment) — Create epic + child beads, work through sequentially
- [Reference: Credential Provisioning](#credential-provisioning) — 1Password vault, SA token, API keys
- [Reference: Matrix Account](#matrix-account) — Tuwunel registration, access token
- [Reference: Create Agent + Configure](#create-agent--configure) — `new-agent`, TOML config
- [Reference: Prompt Files](#prompt-files) — 6 system prompt files
- [Reference: Tools + Skills](#tools--skills) — Tool discovery, shared skill symlinks
- [Reference: Auth Wiring](#auth-wiring) — 1Password token, Matrix credentials in TOML
- [Reference: Memory Structure](#memory-structure) — Workspace directories
- [Reference: Launch & Verify](#launch--verify) — Start service, verification checks
- [Reference: Fleet Integration](#fleet-integration) — Warden, guardrails, archivist
- [Model Roster](#model-roster)
- [Critical Gotchas](#critical-gotchas)
- [Agent Lifecycle](#agent-lifecycle) — Opus buildout → shadow testing → stepdown
- [Cleanup](#cleanup)

---

## Bead-Tracked Deployment

Agent deployment has ~10 discrete steps spanning multiple HITL boundaries. Rather than relying on inline checkboxes and attention, **create a bead epic with child tasks at the start**. Close each bead as it's verified complete.

### Create the Beads

```bash
BD=/srv/openalph/shared/bin/bd

# Create the epic
$BD create "Deploy agent: <name>" -t epic -p P2 -d "Full deployment of <name> agent"

# Child beads — use the returned epic ID as --parent for all of these
$BD create "Prerequisites: spec finalized, model selected" \
  -t task -p P2 --parent <epic-id> \
  -d "Verify: agent spec finalized (all HITL items resolved), model selection confirmed. See §Model Roster."

$BD create "1Password: vault + SA token + API keys" \
  -t task -p P2 --parent <epic-id> -l hitl \
  -d "HITL: Operator creates 1Password vault + service account. Provision API keys (Anthropic, OpenRouter, Brave Search, ElevenLabs). See §Credential Provisioning."
# Then: $BD update <bead-id> -s blocked

$BD create "Matrix: create account + place token" \
  -t task -p P2 --parent <epic-id> \
  -d "Register account on tuwunel via API, place access token at ~/.config/tuwunel-token, back up to 1Password. See §Matrix Account."

$BD create "Create agent + configure TOML" \
  -t task -p P2 --parent <epic-id> \
  -d "Run 'sudo openalph new-agent <name>', edit TOML with providers + matrix. See §Create Agent + Configure."

$BD create "Write prompt files" \
  -t task -p P2 --parent <epic-id> \
  -d "Write SAFETY.md, SOUL.md, OPERATOR.md, WAKE.md, ENVIRONMENT.md, OPERATIONS.md. chmod 444. See §Prompt Files."

$BD create "Enable tools + link skills" \
  -t task -p P2 --parent <epic-id> \
  -d "Create tool .toml files in workspace/tools/. Symlink shared skills, create agent-specific skills. See §Tools + Skills."

$BD create "Wire 1Password + Matrix auth" \
  -t task -p P2 --parent <epic-id> \
  -d "Write SA token to ~/.config/op-token. Verify api_key_cmd and access_token_cmd resolve. See §Auth Wiring."

$BD create "Create memory structure" \
  -t task -p P2 --parent <epic-id> \
  -d "Create memory/{daily,projects,ad-hoc-research}, tmp/, scripts/. See §Memory Structure."

$BD create "Launch + verify" \
  -t task -p P2 --parent <epic-id> \
  -d "systemctl enable+start, verify: service active, first contact in Matrix, tool access, isolation, 1Password access. See §Launch & Verify."

$BD create "Fleet integration" \
  -t task -p P2 --parent <epic-id> \
  -d "Add to: system-warden agent list, checksum-watchdog, archivist AGENTS list, escape-pod backup script, agent roster. See §Fleet Integration."
```

### Working the Beads

1. Create all beads upfront before starting any work.
2. Work sequentially — dependencies flow downward (auth wiring needs 1Password + Matrix done first, etc.).
3. HITL beads start `blocked` — unblock and close them as the operator completes each.
4. Close each bead only after verifying its acceptance criteria.
5. If deployment spans sessions, the next session picks up from `bd children <epic-id>`.

---

## Credential Provisioning

Each agent gets its own **1Password vault and service account** — no shared vaults between agents.

### 1Password Setup (HITL)

Operator actions:
1. Create vault named for the agent (e.g., `agent_wonmun`, `fitness_personal`)
2. Create service account scoped to that vault only
3. Provide the raw SA token to Merry for §Auth Wiring

### Required API Keys

Provision in the agent's 1Password vault. Even if not all are used at launch, having them ready avoids blocking later:

| Credential | 1Password Item Name | Used By |
|-----------|-------------------|---------|
| Anthropic API key | `API Credentials - anthropic <name>` | Claude models |
| OpenRouter API key | `API Credentials - openrouter <name>` | Kimi, GLM, etc. |
| Brave Search API key | `API Credentials - brave search <name>` | `web_search` tool |
| ElevenLabs API key | `API Credentials - elevenlabs <name>` | TTS/STT skills |

---

## Matrix Account

### Account Creation

Account creation is **not** HITL — Merry can register accounts via the Tuwunel API using the registration token.

**Homeserver:** `http://localhost:4269` | **server_name:** `<your-server-name>`

#### Step 1 — Initiate registration (get session token)

```bash
curl -s -X POST http://localhost:4269/_matrix/client/v3/register \
  -H 'Content-Type: application/json' \
  -d '{
    "username": "<name>",
    "password": "<password>",
    "kind": "user"
  }'
```

The server returns HTTP 401 with a JSON body containing a `session` field — this is expected UIAA behavior. Note the `session` value.

#### Step 2 — Complete registration with token

```bash
curl -s -X POST http://localhost:4269/_matrix/client/v3/register \
  -H 'Content-Type: application/json' \
  -d '{
    "username": "<name>",
    "password": "<password>",
    "kind": "user",
    "auth": {
      "type": "m.login.registration_token",
      "token": "<registration-token>",
      "session": "<session-from-step-1>"
    }
  }'
```

Account `@<name>:<your-server-name>` is now created.

### Get Access Token

```bash
curl -s -X POST http://localhost:4269/_matrix/client/v3/login \
  -H 'Content-Type: application/json' \
  -d '{
    "type": "m.login.password",
    "identifier": {"type": "m.id.user", "user": "<name>"},
    "password": "<password>",
    "initial_device_display_name": "OPENALPH"
  }' | jq '{access_token, device_id}'
```

Save `access_token`.

### Place Token File

```bash
sudo sh -c 'echo -n "<access_token>" > /home/oa-<name>/.config/tuwunel-token'
sudo chown oa-<name>:openalph /home/oa-<name>/.config/tuwunel-token
sudo chmod 600 /home/oa-<name>/.config/tuwunel-token
```

### Back Up to 1Password

All matrix credentials are backed up centrally in a single 1Password item (`op://<your-vault>/<your-server-name> matrix credentials`):

```bash
export OP_SERVICE_ACCOUNT_TOKEN="$(cat ~/.config/op/service-account-token)"
op item edit "<your-server-name> matrix credentials" --vault <your-vault> \
  "<name>.username[text]=<name>" \
  "<name>.password[password]=<password>" \
  "<name>.access_token[password]=<access_token>"
```

> **Note:** If the item does not yet exist, create it first:
> `op item create --vault <your-vault> --category login --title "<your-server-name> matrix credentials"`

---

## Create Agent + Configure

### Create

```bash
sudo openalph new-agent <name>
```

Creates: Unix user `oa-<name>`, workspace scaffold, config skeleton at `/etc/openalph/agents/<name>.toml`, systemd unit (enabled but not started).

### Configure TOML

Edit `/etc/openalph/agents/<name>.toml`. The `api_key_cmd` pattern is consistent across all agents:

```
OP_SERVICE_ACCOUNT_TOKEN=$(cat /home/oa-<name>/.config/op-token) /usr/local/bin/op read 'op://<vault>/<item>/<field>'
```

Full example:

```toml
[agent]
name = "<name>"
default_model = "openrouter/z-ai/glm-5"
max_tokens = 16384
model_max_tokens = 131072
vision = false
thinking = "high"

[providers.openrouter]
type = "openai"
base_url = "https://openrouter.ai/api/v1"
api_key_cmd = "OP_SERVICE_ACCOUNT_TOKEN=$(cat /home/oa-<name>/.config/op-token) /usr/local/bin/op read 'op://<vault>/API Credentials - openrouter <name>/credential'"

[providers.anthropic]
type = "anthropic"
api_key_cmd = "OP_SERVICE_ACCOUNT_TOKEN=$(cat /home/oa-<name>/.config/op-token) /usr/local/bin/op read 'op://<vault>/API Credentials - anthropic <name>/credential'"

[providers.local]
type = "openai"
base_url = "<your-local-inference-url>"
api_key = "sk-local-no-auth"
timeout = 1800

[workspace]
path = "/home/oa-<name>/workspace"

[matrix]
homeserver = "http://localhost:4269"
user_id = "@<name>:<your-server-name>"
device_id = "OPENALPH"
access_token_cmd = "cat /home/oa-<name>/.config/tuwunel-token"
```

**Note:** `default_model` and `model_max_tokens` must match. Check §Model Roster. Include all providers the agent may need — even if `default_model` only uses one, sub-agents or model switches may use others.

---

## Prompt Files

Write to `/home/oa-<name>/workspace/`:

| File | Purpose |
|------|---------|
| `SAFETY.md` | Hard rules, protected files, Ask First policies |
| `SOUL.md` | Persona, values, tone, continuity model |
| `OPERATOR.md` | About the operator, preferences, communication style |
| `WAKE.md` | Startup behavior, heartbeat instructions |
| `ENVIRONMENT.md` | Workspace layout, tools, credentials, runtime |
| `OPERATIONS.md` | Procedures, checklists, operational patterns |

Set read-only after writing:

```bash
sudo chmod 444 /home/oa-<name>/workspace/{SAFETY,SOUL,OPERATOR,WAKE,ENVIRONMENT,OPERATIONS}.md
```

**Prompt assembly happens once at process start.** Changes require a service restart. See the openalph-management skill for the prompt change workflow.

---

## Tools + Skills

### Tools

Create `.toml` files in `/home/oa-<name>/workspace/tools/`. Empty files enable with defaults.

```bash
# Baseline (every agent)
for tool in shell file_read file_write file_edit; do
    sudo -u oa-<name> touch /home/oa-<name>/workspace/tools/${tool}.toml
done

# Common additions (per agent needs)
for tool in web_search web_fetch subagent memory_search; do
    sudo -u oa-<name> touch /home/oa-<name>/workspace/tools/${tool}.toml
done
```

### Shared Skills (Symlinks)

Canonical skills live in `/srv/openalph/shared/skills/` (mode 444). Agents access via symlinks — no drift from manual copying.

```bash
sudo mkdir -p /home/oa-<name>/workspace/skills

# Baseline (every agent)
for skill in 1password beads cognitive-pipelines; do
    sudo ln -s /srv/openalph/shared/skills/${skill}.md /home/oa-<name>/workspace/skills/${skill}.md
    sudo chown -h oa-<name>:openalph /home/oa-<name>/workspace/skills/${skill}.md
done

# Additional shared skills as needed
# See: ls /srv/openalph/shared/skills/
```

### Agent-Specific Skills

Domain skills that don't belong in the shared library go as regular files:

```bash
sudo tee /home/oa-<name>/workspace/skills/domain-skill.md << 'EOF'
# Domain Skill
...
EOF
sudo chown oa-<name>:openalph /home/oa-<name>/workspace/skills/domain-skill.md
sudo chmod 640 /home/oa-<name>/workspace/skills/domain-skill.md
```

### ⚠️ 1Password Skill Tailoring

The shared `1password.md` skill may contain vault names and token paths from the orchestrator. If the agent needs its own 1Password commands, create a **local copy** instead of a symlink and update:
- Token path: `~/.config/op/service-account-token` → `~/.config/op-token`
- Vault name: orchestrator's vault → agent's vault
- All example commands referencing the above

---

## Auth Wiring

### 1Password Service Account Token

```bash
# Write the raw SA token (NOT KEY=value format)
sudo sh -c 'echo "<raw-token>" > /home/oa-<name>/.config/op-token'
sudo chown oa-<name>:openalph /home/oa-<name>/.config/op-token
sudo chmod 600 /home/oa-<name>/.config/op-token
```

**Canonical path:** `/home/oa-<name>/.config/op-token` — all agent TOMLs reference this location.

### Matrix Access Token

The access token is stored as a local file — no 1Password round-trip at startup.

```bash
# Place token (created during §Matrix Account)
sudo sh -c 'echo -n "<token>" > /home/oa-<name>/.config/tuwunel-token'
sudo chown oa-<name>:openalph /home/oa-<name>/.config/tuwunel-token
sudo chmod 600 /home/oa-<name>/.config/tuwunel-token
```

**Canonical path:** `/home/oa-<name>/.config/tuwunel-token` — all agent TOMLs use `access_token_cmd = "cat /home/oa-<name>/.config/tuwunel-token"`.

**Backup:** All matrix tokens are backed up centrally in `op://<your-vault>/<your-server-name> matrix credentials` (one item, per-agent sections).

### Verify Credentials Resolve

Before starting the agent, test that `api_key_cmd` and `access_token_cmd` actually work:

```bash
# Test as the agent user
sudo -u oa-<name> bash -c 'OP_SERVICE_ACCOUNT_TOKEN=$(cat ~/.config/op-token) /usr/local/bin/op read "op://<vault>/API Credentials - openrouter <name>/credential"' | head -c 10 && echo '...(ok)'
```

---

## Memory Structure

```bash
sudo -u oa-<name> mkdir -p /home/oa-<name>/workspace/memory/{daily,projects,ad-hoc-research}
sudo -u oa-<name> mkdir -p /home/oa-<name>/workspace/{tmp,scripts}
```

---

## Launch & Verify

### Start

```bash
sudo systemctl enable openalph@<name>
sudo systemctl start openalph@<name>
sudo systemctl status openalph@<name>
```

### Verification

All of these must pass before closing the launch bead:

| Check | How | Pass Criteria |
|-------|-----|---------------|
| Service running | `sudo systemctl status openalph@<name>` | Active (running) |
| First contact | Send message in Matrix room | Agent responds in character |
| Tool access | Ask agent to run `date` (shell), read a file | Tools work |
| Isolation | Ask agent to `cat /etc/openalph/agents/<name>.toml` | Permission denied |
| 1Password | Ask agent to list its vault | Returns correct vault, no others |
| Persona | Review first response | Matches SOUL.md |

---

## Fleet Integration

Five systems need to know about the new agent. **All five must be updated** before closing the fleet integration bead.

### 1. System Warden — Agent Health Check

File: `scripts/warden-check.sh`, line ~161

```bash
local agents=(agent1 agent2 <name>)
```

### 2. Checksum Watchdog — SAFETY.md Monitoring

File: `scripts/checksum-watchdog.sh`, `PROTECTED_ABSOLUTE` array

Add two entries:
```bash
"/etc/openalph/agents/<name>.toml"
"/home/oa-<name>/workspace/SAFETY.md"
```

Then rebaseline: `bash scripts/checksum-watchdog.sh --rebaseline`

### 3. Archivist — Session Memory Extraction

File: `/srv/openalph/shared/bin/archivist.py`, line 32

```python
AGENTS = ["agent1", "agent2", "<name>"]
```

### 4. Escape Pod Backup

File: `scripts/backup-openalph`, `tar -czf` command

Add the agent's workspace to the backup list:

```bash
home/oa-<name>/workspace \
```

### 5. Agent Roster

File: `/srv/openalph/shared/ROSTER.md`

Add the new agent to the live agents table with pronouns, domain, and default model. Remove from "Pending Creation" if listed there.

---

## Model Roster

Available models: `/srv/openalph/shared/docs/model-roster.md`

Common choices:

| Model | `default_model` | `model_max_tokens` | Use Case |
|-------|-----------------|-------------------|----------|
| Claude Opus 4 | `anthropic/claude-opus-4-6` | 200000 | Highest capability |
| Claude Sonnet 4 | `anthropic/claude-sonnet-4-6` | 200000 | Balanced |
| Claude Haiku 4 | `anthropic/claude-haiku-4-5-20251001` | 200000 | Fast and cheap |
| GLM-5 | `openrouter/z-ai/glm-5` | 131072 | Default for domain agents |
| Local model | `local/your-model` | 131072 | Zero cost, local inference |

---

## Critical Gotchas

| Issue | Fix |
|-------|-----|
| Prompt files not loaded | Must be in `/home/oa-<name>/workspace/`, not subdirs |
| Tool not available | Create `.toml` in `workspace/tools/` |
| 1Password access fails | Check `~/.config/op-token` exists, permissions 600, owned by agent |
| Matrix auth fails | Verify `/home/oa-<name>/.config/tuwunel-token` exists, permissions 600, owned by agent. Test: `sudo -u oa-<name> cat ~/.config/tuwunel-token` |
| Config changes ignored | Restart service after TOML edits |
| Wrong file ownership | Files created by the orchestrator need `chown oa-<name>:openalph` |

---

## Agent Lifecycle

Agents typically progress through three phases. See the openalph-management skill for operational commands (restart, logs, config changes).

1. **Opus Buildout** — Run on highest-capability model. Operator reviews each interaction. Verify persona, access controls, tool restrictions. Goal: confirm correctness.
2. **Shadow Testing** — Evaluate lower-cost model responses. Watch for persona drift or reasoning degradation.
3. **Stepdown** — Switch to production model. Monitor first sessions. Document stepdown date in agent's OPERATIONS.md. Revert if quality regresses.

---

## Cleanup (Emergency Only)

```bash
sudo systemctl stop openalph@<name>
sudo systemctl disable openalph@<name>
sudo rm /etc/openalph/agents/<name>.toml
sudo userdel -r oa-<name>    # CAUTION: deletes workspace

# Also remove from fleet integration files (§Fleet Integration)
```
