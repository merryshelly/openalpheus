# INSTALL.md — OpenAlpheus

Detailed installation, configuration, and operation guide. For a quick overview, see [README.md](README.md).

## 1. Prerequisites

The installer auto-installs most system dependencies (curl, jq, python3-venv, Docker, Caddy). You only need to ensure the following before running `install.sh`:

### Operating System

Debian 12+ or Ubuntu 22.04+, amd64 or arm64, systemd-based. Root or sudo access required throughout.

```bash
cat /etc/os-release | grep -E '^(ID|VERSION_ID)='   # distro
dpkg --print-architecture                             # arch
systemctl --version | head -1                         # systemd (need 249+)
```

Debian 12 ships systemd 252, Ubuntu 22.04 ships 249. Below 249 → need a newer OS.

### Python 3.11+

Python is the one dependency the installer **will not** auto-install — version upgrades are complex and distro-specific. Debian 12+ and Ubuntu 24.04+ ship with Python 3.11+ out of the box.

```bash
python3 --version
```

| Distro | Default | Action |
|--------|---------|--------|
| Debian 12+ | 3.11+ | None |
| Ubuntu 24.04+ | 3.12+ | None |
| Ubuntu 22.04 | 3.10 | **Need deadsnakes PPA** (see below) |

**Ubuntu 22.04 only:**
```bash
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
```

### Auto-Installed Dependencies

The following are installed automatically by the installer if missing. No manual action needed:

- **curl** — installed via `apt-get`
- **jq** — installed via `apt-get`
- **python3-venv** — installed via `apt-get` (versioned package, e.g. `python3.12-venv`)
- **Docker 24+ with Compose v2** — installed via Docker's official convenience script ([get.docker.com](https://get.docker.com))
- **Caddy** — installed from the official Caddy APT repository

> **Note:** If Docker is already installed but below version 24, the installer will **not** auto-upgrade it — upgrading Docker is left to the user. See https://docs.docker.com/engine/install/

### TLS — Choose One

| Mode | Best for | Requirements |
|------|----------|-------------|
| **Tailscale** (recommended) | Homelab | Tailscale installed + connected (`tailscale status`). Enable HTTPS in Tailscale admin console → DNS. |
| **Let's Encrypt** | VPS with public domain | DNS A/AAAA → this machine. Port 80 reachable from internet. |
| **BYO certificate** | Everything else | Cert + key files on disk. |

For Tailscale: install from https://tailscale.com/download/linux, then `tailscale up`.

### Ports

| Port | Purpose | Check with |
|------|---------|-----------|
| 443 | HTTPS (Caddy) | `ss -tlnp \| grep :443` |
| 4269 | Tuwunel (loopback) | `ss -tlnp \| grep :4269` |
| 80 | Let's Encrypt only | `ss -tlnp \| grep :80` |

If 443 is occupied (nginx, apache, etc.), stop the service before running the installer.

### LLM API Key

| Provider | Get a key | Notes |
|----------|----------|-------|
| **Anthropic** (recommended) | https://console.anthropic.com/ | Native SDK, best tool-use support |
| **OpenRouter** | https://openrouter.ai/ | Multi-model gateway |
| **Local inference** | — | vLLM, llama.cpp, Ollama — no key, you provide the base URL |

Have the key ready. For automated installs, write it to a file and use `OPENALPH_API_KEY_FILE`.

### Brave Search API Key (optional)

The `web_search` tool uses the [Brave Search API](https://brave.com/search/api/) to perform web searches. A free tier is available. Without it, your agent can still fetch and read specific URLs via `web_fetch`, but cannot perform general web searches.

The installer will prompt for this key. You can skip it and add one later by editing `workspace/tools/web_search.toml`.

### Disk Space

Minimum 500 MB free (hard fail). Recommended 2 GB+ (Docker images ~500 MB, workspaces grow). Check: `df -h /var/lib/docker`

---

## 2. Quick Install

```bash
curl -fsSL https://codeberg.org/merryshelly/openalpheus/raw/branch/main/install.sh | sudo bash
```

What it does: (1) auto-installs missing system dependencies (curl, jq, Docker, Caddy, etc.) and verifies prerequisites, (2) installs OpenAlpheus into `/opt/openalph-venv/`, (3) deploys tuwunel via Docker, (4) configures TLS via Caddy, (5) creates Matrix accounts, (6) creates your agent (Unix user + workspace + systemd unit), (7) serves Cinny web client.

Total time: 2–5 minutes (Docker image pull is the bottleneck). Log written to `/tmp/openalph-bootstrap-<timestamp>.log`.

---

## 3. What the Installer Will Ask

Every prompt, in order. All can be skipped via environment variables (§4).

**Confirmation** — `Proceed with installation? [y/N]` — last chance before system changes.

**TLS mode** — `[1] Tailscale  [2] Let's Encrypt  [3] BYO cert` — default 1.

**Domain** — auto-detected for Tailscale; prompted for others. ⚠️ **This becomes the Matrix `server_name` and is PERMANENT.** It's baked into every user ID and room ID. Cannot be changed without destroying the database.

**Operator username** — your Matrix login (e.g. `alice`). Lowercase, digits, dots, hyphens.

**Operator password** — hidden input, minimum 8 chars, must confirm.

**Agent name** — becomes Unix user `oa-<name>`, Matrix user `@<name>:domain`, unit `openalph@<name>`. Starts with lowercase letter, only `[a-z0-9-]`, max 32 chars.

**LLM provider** — `[1] Anthropic  [2] OpenRouter  [3] Local` — default 1.

**API key** — hidden input. Not prompted for local providers.

**Base URL** — local provider only (e.g. `http://localhost:11434/v1`).

**Model** — press Enter for default (`claude-sonnet-4-6` for Anthropic). Changeable later.

**Brave Search API key** — optional. Enables the `web_search` tool. Press Enter to skip. Get a free key at https://brave.com/search/api/.

**If it fails:** fix the issue and re-run. Use `--force` to overwrite existing agent resources: `sudo bash install.sh --force`

---

## 4. Non-Interactive Installation

| Variable | Purpose | Valid values | Default |
|----------|---------|-------------|---------|
| `OPENALPH_YES` | Skip confirmation | `true` | unset |
| `OPENALPH_TLS_MODE` | TLS method | `tailscale`, `letsencrypt`, `custom` | unset |
| `OPENALPH_DOMAIN` | Matrix FQDN | FQDN | auto for Tailscale |
| `OPENALPH_TLS_CERT` | Cert path (custom mode) | file path | unset |
| `OPENALPH_TLS_KEY` | Key path (custom mode) | file path | unset |
| `OPENALPH_OP_USER` | Operator username | localpart | unset |
| `OPENALPH_OP_PASS` | Operator password | 8+ chars | unset |
| `OPENALPH_AGENT_NAME` | Agent name | `[a-z][a-z0-9-]{0,31}` | unset |
| `OPENALPH_PROVIDER` | LLM provider | `anthropic`, `openrouter`, `local` | unset |
| `OPENALPH_API_KEY_FILE` | Path to API key file | file path | unset |
| `OPENALPH_API_KEY` | API key directly ⚠️ | string | unset |
| `OPENALPH_BASE_URL` | Local inference URL | URL | unset |
| `OPENALPH_MODEL` | Model name | string | provider default |
| `OPENALPH_BRAVE_API_KEY_FILE` | Path to Brave Search key file | file path | unset |
| `OPENALPH_BRAVE_API_KEY` | Brave Search key directly | string | unset |
| `OPENALPH_VERSION` | Version to install | git tag | `v0.1.2` |
| `OPENALPH_FORCE` | Overwrite existing | `true` | `false` |

⚠️ `OPENALPH_API_KEY` is visible in `/proc/<pid>/environ`. Prefer `OPENALPH_API_KEY_FILE` — write key to a file (mode 600) and pass the path.

### Example: Tailscale
```bash
echo "sk-ant-..." > /root/.openalph-key && chmod 600 /root/.openalph-key
sudo OPENALPH_YES=true OPENALPH_TLS_MODE=tailscale \
     OPENALPH_OP_USER=alice OPENALPH_OP_PASS=changeme123 \
     OPENALPH_AGENT_NAME=myagent OPENALPH_PROVIDER=anthropic \
     OPENALPH_API_KEY_FILE=/root/.openalph-key bash install.sh
```

### Example: Let's Encrypt
```bash
sudo OPENALPH_YES=true OPENALPH_TLS_MODE=letsencrypt \
     OPENALPH_DOMAIN=matrix.example.com \
     OPENALPH_OP_USER=alice OPENALPH_OP_PASS=changeme123 \
     OPENALPH_AGENT_NAME=myagent OPENALPH_PROVIDER=anthropic \
     OPENALPH_API_KEY_FILE=/root/.openalph-key bash install.sh
```

### Example: BYO Certificate
```bash
sudo OPENALPH_YES=true OPENALPH_TLS_MODE=custom \
     OPENALPH_DOMAIN=matrix.example.com \
     OPENALPH_TLS_CERT=/etc/ssl/certs/matrix.pem \
     OPENALPH_TLS_KEY=/etc/ssl/private/matrix.key \
     OPENALPH_OP_USER=alice OPENALPH_OP_PASS=changeme123 \
     OPENALPH_AGENT_NAME=myagent OPENALPH_PROVIDER=openrouter \
     OPENALPH_API_KEY_FILE=/root/.openalph-key bash install.sh
```

---

## 5. What Gets Installed

| Path | Purpose |
|------|---------|
| `/opt/openalph-venv/` | Python venv (OpenAlpheus + dependencies) |
| `/usr/local/bin/openalph` | CLI symlink |
| `/etc/openalph/agents/<name>.toml` | Agent config (provider, model, Matrix creds) |
| `/etc/openalph/reg-token` | Registration token for future accounts (root-only) |
| `/srv/openalph/shared/` | Shared resources across agents |
| `/opt/tuwunel/docker-compose.yml` | Tuwunel Docker Compose file |
| `/home/oa-<name>/workspace/` | Agent workspace (prompts, skills, tools, sessions) |
| `/home/oa-<name>/.config/provider-key` | LLM API key (mode 600) |
| `/home/oa-<name>/.config/matrix-token` | Matrix access token (mode 600) |
| `/opt/openalph/web/` | Cinny web client |
| `/etc/caddy/Caddyfile` | TLS reverse proxy config |
| `/etc/systemd/system/openalph@.service` | Systemd template unit |

---

## 6. Connecting to Your Agent

### Web (Cinny)
1. Open `https://your-domain` in a browser
2. Log in with your operator username and password
3. Create a new room (Private)
4. Invite `@agentname:your-domain`
5. Wait for auto-join, then send a message

### Mobile (Element)
Install Element (https://element.io/download). Sign in → Other homeserver → `https://your-domain`. Create a room, invite the agent.

### Any Matrix Client
Point it at your homeserver URL, log in with operator credentials. FluffyChat, Nheko, iamb, etc. all work.

### Room Behavior
**1:1 rooms:** agent responds to every message. **3+ members:** agent only responds when @mentioned.

---

## 7. Customizing Your Agent

Six workspace markdown files assembled into the system prompt at process start:

| File | Purpose | Customize? |
|------|---------|------------|
| `SAFETY.md` | Hard behavioral constraints | Review, keep defaults |
| `SOUL.md` | Identity, personality, tone | **Yes — start here** |
| `OPERATOR.md` | About you — name, prefs, timezone | **Yes** |
| `WAKE.md` | Heartbeat behavior | When using heartbeats |
| `ENVIRONMENT.md` | Infrastructure docs, paths | When adding infra |
| `OPERATIONS.md` | Procedures, runbooks | As needed |

```bash
sudo -u oa-<name> nano /home/oa-<name>/workspace/SOUL.md
```

⚠️ **Prompt changes require restart:** `sudo systemctl restart openalph@<name>`

---

## 8. Skills and Tools

### Skills
Markdown files in `workspace/skills/`. Listed in the prompt by name; agent reads content on demand. Updating an existing skill takes effect immediately (no restart). Adding a new skill file requires a restart for it to appear in the prompt's skill index.
```bash
sudo -u oa-<name> nano /home/oa-<name>/workspace/skills/my-skill.md
```

### Tools
Enabled by `.toml` files in `workspace/tools/`. Empty file = enabled with defaults. All 14 built-in tools enabled by default:

| Tool | Purpose |
|------|---------|
| `shell` | Execute shell commands |
| `file_read` | Read files |
| `file_write` | Write/create files |
| `file_edit` | Find-and-replace in files |
| `file_patch` | Apply multiple SEARCH/REPLACE hunks to a file atomically |
| `web_search` | Web search (requires Brave Search API key in TOML) |
| `web_fetch` | Fetch + extract text from URLs |
| `grep` | Search file contents by regex (bounded, workspace-rooted) |
| `glob` | Find files by name pattern (bounded, workspace-rooted) |
| `subagent` | Spawn sub-agents for complex tasks |
| `memory_search` | Semantic + keyword search over workspace |
| `send_media` | Send files to Matrix room |
| `context_status` | Agent self-monitoring (context %, tokens, session age) |
| `todo_write` | Session-scoped working-memory task list |

Disable a tool by removing its TOML: `rm workspace/tools/shell.toml` then restart.

**`web_search` note:** requires a [Brave Search API key](https://brave.com/search/api/) in the TOML. Without it, `web_fetch` still works for reading specific URLs.

---

## 9. Adding More Agents

The bootstrap created one agent with all credentials wired up. Additional agents need manual setup.

```bash
sudo openalph new-agent <name>           # creates Unix user, workspace, config skeleton
sudo nano /etc/openalph/agents/<name>.toml   # fill in CHANGE_ME placeholders
```

**Register a Matrix account** — temporarily re-enable registration, register, then close it again. The registration API is a multi-step UIAA flow. If you have a running agent, ask it to handle this — it can execute the curl commands and parse the JSON responses. The short version:

1. Write a temporary Docker Compose override enabling registration with the saved token:
```bash
REG_TOKEN=$(sudo cat /etc/openalph/reg-token)
cat > /opt/tuwunel/docker-compose.override.yml <<EOF
services:
  homeserver:
    environment:
      TUWUNEL_ALLOW_REGISTRATION: "true"
      TUWUNEL_REGISTRATION_TOKEN: "${REG_TOKEN}"
EOF
sudo chmod 600 /opt/tuwunel/docker-compose.override.yml
cd /opt/tuwunel && sudo docker compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

2. Register the account (two-phase: initiate to get a session, then complete with token + credentials). Save the returned `access_token`.

3. Close registration immediately:
```bash
sudo rm /opt/tuwunel/docker-compose.override.yml
cd /opt/tuwunel && sudo docker compose up -d
```

4. Verify registration is closed: `curl -s http://127.0.0.1:4269/_matrix/client/v3/register -d '{}' | jq .` should return an error, not a session.

**Store credentials and start:**
```bash
echo -n "<api-key>" > /home/oa-<name>/.config/provider-key
echo -n "<access_token>" > /home/oa-<name>/.config/matrix-token
chown oa-<name>:openalph /home/oa-<name>/.config/{provider-key,matrix-token}
chmod 600 /home/oa-<name>/.config/{provider-key,matrix-token}
sudo systemctl enable --now openalph@<name>
```

Invite `@<name>:your-domain` to a room from your Matrix client.

---

## 10. Managing Agents

```bash
sudo systemctl status openalph@<name>        # status
sudo systemctl start openalph@<name>         # start
sudo systemctl stop openalph@<name>          # stop
sudo systemctl restart openalph@<name>       # restart
sudo journalctl -fu openalph@<name>          # follow logs
sudo journalctl -u openalph@<name> --since "5 min ago"   # recent logs
openalph showprompt <name>                    # view assembled prompt
```

**Config:** `/etc/openalph/agents/<name>.toml` — **Workspace:** `/home/oa-<name>/workspace/`

**Change model:** edit `default_model` in the TOML → restart. Or use `/model <provider/model>` in-room (per-room, no restart).

**In-room commands:** `/status`, `/model`, `/effort`, `/heartbeat`, `/umbral`, `/cache`, `/stop`, `/resume` — see [README.md](README.md) for full reference.

---

## 11. Troubleshooting

### Preflight Failures

| Error | Fix |
|-------|-----|
| Python < 3.11 | Install 3.11+ (deadsnakes PPA on Ubuntu 22.04, see §1) |
| Docker not found / < 24 | Install from https://docs.docker.com/engine/install/ |
| Compose v2 not found | Comes with official Docker; or `apt install docker-compose-plugin` |
| systemd < 249 | Need Debian 12+ or Ubuntu 22.04+ |
| Low disk space | `docker system prune` or free space manually |

### TLS / Caddy

**Tailscale cert fails:** check HTTPS enabled in Tailscale admin console → DNS. Verify Tailscale is connected: `tailscale status`. Try manually: `sudo tailscale cert $(tailscale status --json | jq -r '.Self.DNSName' | sed 's/\.$//')`

**Let's Encrypt fails:** verify DNS resolves (`dig +short your-domain`), port 80 is internet-reachable, and check `sudo journalctl -xeu caddy`.

**Cert errors in browser:** domain mismatch — ensure you're accessing the exact domain configured during install.

### Tuwunel Won't Start

```bash
sudo systemctl status docker                  # Docker running?
cd /opt/tuwunel && sudo docker compose logs   # container logs
sudo ss -tlnp | grep :4269                    # port conflict?
```

Nuclear option: `cd /opt/tuwunel && sudo docker compose down && sudo docker compose up -d`

### Agent Won't Start

Check `sudo journalctl -u openalph@<name> --since "5 min ago"` and look for:

- **401/403 from LLM API** → wrong API key. Check `/home/oa-<name>/.config/provider-key`.
- **M_UNKNOWN_TOKEN** → bad Matrix token. Check `/home/oa-<name>/.config/matrix-token`.
- **TOML parse error** → config syntax. Validate: `python3 -c "import tomllib; tomllib.load(open('/etc/openalph/agents/<name>.toml','rb'))"`
- **CancelledError on restart** → cosmetic. If the agent starts successfully after, ignore it.

### Agent Not Responding

1. Process running? `sudo systemctl status openalph@<name>`
2. Agent in the room? Check member list in Cinny/Element.
3. Room gated? 3+ members → must @mention. Try `@agentname: hello`
4. Watch logs live: `sudo journalctl -fu openalph@<name>` then send a message.

### Context Overflow

Agent hit the model's context limit. **Fix:** create a new room (fresh session). Or increase `model_max_tokens` in the TOML config if the model supports it.

### Re-running the Installer

Mostly idempotent. Use `--force` to overwrite existing agent resources. The domain is permanent — changing it requires `cd /opt/tuwunel && docker compose down -v` (destroys all Matrix data).

---

## 12. Uninstalling

```bash
# 1. Stop and disable all agents
systemctl list-units 'openalph@*'
sudo systemctl stop openalph@<name>
sudo systemctl disable openalph@<name>

# 2. Remove agent Unix users (and home directories)
sudo userdel -r oa-<name>

# 3. Remove tuwunel (including database)
cd /opt/tuwunel && sudo docker compose down -v
sudo rm -rf /opt/tuwunel

# 4. Remove Caddy config (optionally uninstall Caddy)
sudo rm /etc/caddy/Caddyfile
sudo systemctl stop caddy

# 5. Remove OpenAlpheus files
sudo rm -rf /opt/openalph-venv /opt/openalph/web /etc/openalph /srv/openalph
sudo rm -f /usr/local/bin/openalph /etc/systemd/system/openalph@.service
sudo systemctl daemon-reload

# 6. Remove system group
sudo groupdel openalph
```
