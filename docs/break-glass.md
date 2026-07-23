# Break Glass: Revert OpenAlph to Known Good State

**When to use:** Something is broken after a code change. Agent behavior is wrong, services won't start, or you suspect a bad deploy.

**Time to complete:** ~5 minutes

---

## Step 0: Stop the Bleeding

Stop the affected agent(s). This is safe — agents reconnect cleanly on restart.

```bash
# Stop one agent
sudo systemctl stop openalph@<name>

# Stop all agents
sudo systemctl stop openalph@agent1 openalph@agent2 openalph@agent3

# Nuclear: stop everything
sudo systemctl stop 'openalph@*'
```

Agents are now down. Nothing is getting worse. Take a breath.

---

## Step 1: Identify the Last Known Good Version

ARCH-7: `install.sh` pip-installs OpenAlph into `/opt/openalph-venv` from the
Codeberg repo (`git+https://codeberg.org/merryshelly/openalph.git@<version>`).
There is **no git checkout, no source tree, and no test suite on disk** — so
recovery is a pip reinstall of a known-good ref, not a local `git checkout`.

Find the version currently installed:

```bash
/opt/openalph-venv/bin/openalph --version
/opt/openalph-venv/bin/pip show openalph | grep -i version
```

Pick the ref to roll back to (a tag or commit that predates the bad change).
Browse history on the remote — https://codeberg.org/merryshelly/openalph/commits —
or, if you keep a checkout on a workstation, `git log --oneline` there. Note the
tag (e.g. `v0.1.2`) or commit SHA.

---

## Step 2: Reinstall a Known-Good Version

Reinstall the package at the chosen ref into the existing venv. `--force-reinstall`
replaces the current code; `--no-deps` keeps it fast and avoids churning
dependencies (add it only if deps are unchanged between the two refs).

```bash
sudo /opt/openalph-venv/bin/pip install --force-reinstall \
    "git+https://codeberg.org/merryshelly/openalph.git@<good-ref>"
```

Replace `<good-ref>` with the tag or SHA from Step 1 (e.g. `v0.1.2`).

If pip itself is failing (network, build), and you took a venv snapshot before
upgrading (recommended: `cp -a /opt/openalph-venv /opt/openalph-venv.bak-<date>`
before any upgrade), restore it instead:

```bash
sudo systemctl stop 'openalph@*'
sudo rm -rf /opt/openalph-venv
sudo mv /opt/openalph-venv.bak-<date> /opt/openalph-venv
```

---

## Step 3: Smoke-Test the Reinstall

The wheel does not ship the test suite, so verify the install is sane rather
than running pytest:

```bash
# Right version, and the CLI imports/starts cleanly
/opt/openalph-venv/bin/openalph --version
# Assemble a prompt without touching credentials — catches import/config breakage
sudo /opt/openalph-venv/bin/openalph prompt <name>
```

If both succeed you're ready to restart. If you want the full test suite, run it
from a git checkout on a workstation (not on the box):
`git clone https://codeberg.org/merryshelly/openalph.git && cd openalph && pip install -e ".[dev]" && pytest -q`.

---

## Step 4: Restart Services

```bash
# Start one agent
sudo systemctl start openalph@<name>

# Start all agents
sudo systemctl start openalph@agent1 openalph@agent2 openalph@agent3

# Verify they're running
systemctl status 'openalph@*' --no-pager
```

---

## Step 5: Verify Agent Is Working

Send a test message in Matrix. Check the journal for errors:

```bash
journalctl -u openalph@<name> -n 50 --no-pager
```

---

## Reference: Architecture

| Component | Path | Notes |
|-----------|------|-------|
| Installed package | `/opt/openalph-venv/lib/python*/site-packages/openalph/` | The running code (pip-installed; no separate src tree) |
| Virtualenv | `/opt/openalph-venv/` | The venv the service runs from; bound read-only into it |
| Tests | not on disk | Ship only in the git repo; run from a workstation checkout |
| Agent configs | `/etc/openalph/agents/*.toml` | Per-agent config (not in git) |
| Service unit | `/etc/systemd/system/openalph@.service` | Template unit for all agents |
| Agent workspaces | `/home/oa-<name>/workspace/` | Per-agent prompts, tools, memory |
| Shared data | `/srv/openalph/shared/` | Beads DB, shared docs |
| Git remote | `codeberg.org/merryshelly/openalph` | Full history |

**Key fact:** The venv at `/opt/openalph-venv` is bind-mounted read-only into the running services via `BindReadOnlyPaths`. A reinstall changes the code on disk, but running services keep the old code until they are **restarted** — so a stopped service cannot make things worse, and your fix does not take effect until Step 4.

---

## Emergency Contacts

If the orchestrator is the one misbehaving, it can't help you debug itself. Options:

- Read the journal directly: `journalctl -u openalph@<name> -f`
- Talk to another agent to investigate
- SSH to the host and work manually
