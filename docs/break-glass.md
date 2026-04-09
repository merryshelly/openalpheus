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

## Step 1: Identify the Last Known Good Commit

```bash
cd /opt/openalph
git log --oneline -20
```

Look for the last commit before the bad change. Commits have descriptive messages.

If you know which file is broken but not which commit:

```bash
# Show commit history for a specific file
git log --oneline -10 -- src/openalph/agent.py

# Show what changed in a specific commit
git show <commit> --stat
```

---

## Step 2: Revert the Files

**Option A: Revert specific files to a known good commit**

```bash
cd /opt/openalph
git checkout <good-commit> -- src/openalph/agent.py src/openalph/tools/__init__.py
```

This restores those files to their state at `<good-commit>` without affecting anything else.

**Option B: Revert an entire commit (undo the last change)**

```bash
cd /opt/openalph
git revert --no-commit HEAD
```

This stages the inverse of the last commit. Review with `git diff --cached`, then `git commit -m "revert: <reason>"`.

**Option C: Hard reset to a known good commit (last resort)**

```bash
cd /opt/openalph
git reset --hard <good-commit>
```

⚠️ This discards all changes after that commit. Only use if you want to throw away everything since then.

---

## Step 3: Verify Tests Pass

```bash
cd /opt/openalph
.venv/bin/python -m pytest tests/ -q
```

If tests pass, you're ready to restart. If tests fail, you may need to go further back — repeat Step 1-2.

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
| Source code | `/opt/openalph/src/openalph/` | Bind-mounted read-only into service |
| Tests | `/opt/openalph/tests/` | Run from repo root |
| Virtualenv | `/opt/openalph/.venv/` | Pytest and all deps live here |
| Agent configs | `/etc/openalph/agents/*.toml` | Per-agent config (not in git) |
| Service unit | `/etc/systemd/system/openalph@.service` | Template unit for all agents |
| Agent workspaces | `/home/oa-<name>/workspace/` | Per-agent prompts, tools, memory |
| Shared data | `/srv/openalph/shared/` | Beads DB, shared docs |
| Git remote | `codeberg.org/merryshelly/openalph` | Full history |

**Key fact:** The source at `/opt/openalph/src/` is bind-mounted read-only into the running services via `BindReadOnlyPaths`. Changes to source files take effect on **next service restart** — not immediately. A stopped service cannot make things worse.

---

## Emergency Contacts

If the orchestrator is the one misbehaving, it can't help you debug itself. Options:

- Read the journal directly: `journalctl -u openalph@<name> -f`
- Talk to another agent to investigate
- SSH to the host and work manually
