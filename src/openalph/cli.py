"""CLI for OpenAlph — manage agents via systemctl and run in dev mode."""

import argparse
import asyncio
import getpass
import json
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

from openalph.config import CONFIG_DIR, load_agent_config, ConfigError
from openalph.admin import create_agent

logger = logging.getLogger("openalph")


def list_agents() -> list[str]:
    """Return sorted list of agent names from TOML files in CONFIG_DIR."""
    if not CONFIG_DIR.exists():
        return []
    return sorted(p.stem for p in CONFIG_DIR.glob("*.toml"))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="openalph", description="Manage OpenAlph agents.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    parser.set_defaults(command=None)

    sub = parser.add_subparsers(dest="command")
    sub.required = True

    # start / stop / restart
    for cmd in ("start", "stop", "restart"):
        p = sub.add_parser(cmd)
        p.add_argument("agent")

    # status
    p = sub.add_parser("status")
    p.add_argument("agent", nargs="?", default=None)

    # list
    sub.add_parser("list")

    # logs
    p = sub.add_parser("logs")
    p.add_argument("agent")
    p.add_argument("-f", "--follow", action="store_true", default=False)

    # new-agent
    p = sub.add_parser("new-agent")
    p.add_argument("name")
    p.add_argument("--dry-run", action="store_true", default=False)
    # BUG-2: without this, a re-run silently replaced a live agent's config
    # and OPERATIONS.md with the CHANGE_ME skeleton.
    p.add_argument(
        "--force", action="store_true", default=False,
        help="Overwrite an existing agent's config and OPERATIONS.md "
             "(default: existing files are left alone)",
    )

    # run
    p = sub.add_parser("run")
    p.add_argument("agent")
    p.add_argument("-v", "--verbose", action="store_true", dest="verbose")

    # monitor
    p = sub.add_parser("monitor")
    p.add_argument("agent")

    # showprompt
    p = sub.add_parser("showprompt", help="Display the full assembled system prompt")
    p.add_argument("agent")

    # chat
    p = sub.add_parser("chat", help="Interactive CLI session with an agent")
    p.add_argument("agent")
    p.add_argument("--room", default=None, help="Custom room ID for session isolation")
    p.add_argument("--truncate", type=int, default=None,
                     help="Tool-result preview character limit (default 200, 0=unlimited)")

    # exec — one-shot headless turn (Stigmergy worker driver, bead workspace-e2uh.149).
    # No Matrix, no SessionLog, no chat loop: a fresh Agent, a single
    # handle_input, and EXACTLY ONE JSON line on stdout (everything else —
    # including all logging — goes to stderr).
    p = sub.add_parser("exec", help="Run one headless agent turn and emit one JSON result line")
    p.add_argument("--agent", required=True, help="Agent name (config in CONFIG_DIR)")
    p.add_argument("--task-file", required=True,
                   help="Path to the task prompt file, or - for stdin")
    p.add_argument("--model", default=None,
                   help="Override the agent's default model (provider/model)")
    p.add_argument("--effort", choices=("none", "low", "medium", "xhigh"),
                   default=None, help="Reasoning effort (card-native values)")
    p.add_argument("--max-turns", type=int, default=None,
                   help="Override max tool-call iterations for this run")
    p.add_argument("--tools", default=None,
                   help="Comma-separated builtin tool names (bypasses workspace discovery)")
    p.add_argument("--room", default=None,
                   help="Room label for history isolation (default: _exec)")

    return parser.parse_args(argv)


def _systemctl(action: str, agent: str):
    """Run a systemctl command with helpful error on permission failure."""
    try:
        subprocess.run(
            ["systemctl", action, f"openalph@{agent}.service"],
            check=True,
        )
    except subprocess.CalledProcessError:
        import os
        if os.geteuid() != 0:
            print(f"Failed to {action} openalph@{agent}. Try: sudo openalph {action} {agent}",
                  file=sys.stderr)
        sys.exit(1)


def cmd_start(args):
    if args.agent == "all":
        agents = list_agents()
        if not agents:
            print("No agents found.", file=sys.stderr)
            sys.exit(1)
        for agent in agents:
            print(f"Starting {agent}...")
            _systemctl("start", agent)
        print(f"Started {len(agents)} agent(s).")
    else:
        _systemctl("start", args.agent)


def cmd_stop(args):
    if args.agent == "all":
        agents = list_agents()
        if not agents:
            print("No agents found.", file=sys.stderr)
            sys.exit(1)
        for agent in agents:
            print(f"Stopping {agent}...")
            _systemctl("stop", agent)
        print(f"Stopped {len(agents)} agent(s).")
    else:
        _systemctl("stop", args.agent)


def cmd_restart(args):
    if args.agent == "all":
        agents = list_agents()
        if not agents:
            print("No agents found.", file=sys.stderr)
            sys.exit(1)
        for agent in agents:
            print(f"Restarting {agent}...")
            _systemctl("restart", agent)
        print(f"Restarted {len(agents)} agent(s).")
    else:
        _systemctl("restart", args.agent)


def cmd_status(args):
    agents = [args.agent] if args.agent else list_agents()
    for agent in agents:
        subprocess.run(
            ["systemctl", "status", f"openalph@{agent}.service"],
            check=False,
        )


def cmd_list(args):
    agents = list_agents()
    if not agents:
        print("No agents configured.")
    else:
        for agent in agents:
            print(agent)


def cmd_logs(args):
    cmd = ["journalctl", "-u", f"openalph@{args.agent}.service"]
    if args.follow:
        cmd.append("-f")
    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        pass


def cmd_new_agent(args):
    ops = create_agent(
        args.name, dry_run=args.dry_run, force=getattr(args, "force", False)
    )
    if args.dry_run:
        print("Dry run — planned operations:")
        for i, op in enumerate(ops, 1):
            print(f"  {i}. [{op.kind}] {op.description}")
        print(f"\n{len(ops)} operations planned. Run without --dry-run to execute.")
    else:
        print(f"Agent '{args.name}' created successfully ({len(ops)} operations).")


def cmd_run(args):
    from openalph.agent import Agent
    from openalph.matrix import MatrixBot

    try:
        config = load_agent_config(args.agent)
    except ConfigError as e:
        logger.error("Config error: %s", e)
        sys.exit(1)

    if config.matrix is None:
        print("Error: No [matrix] section in config", file=sys.stderr)
        sys.exit(1)

    # kdsn.292: best-effort ntfy alert when the DEFAULT provider is degraded.
    # Runs before Matrix connect (alerts even if the homeserver is down) and
    # is DEFENSIVELY wrapped — a startup alert must never take the agent down;
    # maybe_notify_default_degraded is itself fail-soft, this is belt+braces.
    try:
        from openalph.notify import maybe_notify_default_degraded
        maybe_notify_default_degraded(config)
    except Exception:
        logger.exception("degraded-start ntfy alert path failed (ignored)")

    agent = Agent(config)
    bot = MatrixBot(agent, config.matrix)

    async def run():
        loop = asyncio.get_running_loop()
        stop = loop.create_future()

        def _signal_handler():
            if not stop.done():
                stop.set_result(None)

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _signal_handler)

        bot_task = asyncio.create_task(bot.start())
        logger.info("Agent '%s' running (model: %s)", config.name, config.default_model)
        await stop
        logger.info("Shutting down...")
        await bot.stop()
        try:
            await asyncio.wait_for(bot_task, timeout=5.0)
        except asyncio.TimeoutError:
            bot_task.cancel()
            try:
                await bot_task
            except asyncio.CancelledError:
                pass

    asyncio.run(run())


def cmd_monitor(args):
    """Live tail of agent JSONL logs with formatted output."""
    import json
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    # Read workspace path directly from TOML — monitor never needs API keys,
    # so skip load_agent_config which triggers api_key_cmd (op CLI prompts).
    import tomllib
    config_path = CONFIG_DIR / f"{args.agent}.toml"
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    try:
        with config_path.open("rb") as f:
            toml_data = tomllib.load(f)
        workspace = Path(toml_data.get("workspace", {}).get("path", ""))
        agent_name = toml_data.get("agent", {}).get("name", args.agent)
    except Exception as e:
        print(f"Failed to read config: {e}", file=sys.stderr)
        sys.exit(1)

    logs_dir = workspace / "logs"
    if not logs_dir.is_dir():
        print(f"No logs directory: {logs_dir}", file=sys.stderr)
        sys.exit(1)

    # ANSI colors
    DIM = "\033[2m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    MAGENTA = "\033[35m"
    RESET = "\033[0m"

    def format_entry(entry: dict) -> str:
        ts = entry.get("ts", "")
        try:
            dt = datetime.fromisoformat(ts).astimezone()
            time_str = dt.strftime("%H:%M:%S")
        except (ValueError, TypeError):
            time_str = ts[:8] if ts else "??:??:??"

        model = entry.get("model", "?")
        # Shorten model name: "anthropic/claude-opus-4-6" → "opus-4-6"
        short_model = model.split("/")[-1] if "/" in model else model

        in_tok = entry.get("input_tokens", 0)
        out_tok = entry.get("output_tokens", 0)
        latency = entry.get("latency_ms", 0)
        latency_s = latency / 1000 if latency else 0

        tool_calls = entry.get("tool_calls", [])
        preview = entry.get("content_preview", "").replace("\n", " ")[:120]

        # Room ID → short form: "!ABC...XYZ:server" → "ABC..XYZ"
        room = entry.get("room_id", "")
        if room.startswith("!") and ":" in room:
            room_short = room[1:room.index(":")]
            if len(room_short) > 8:
                room_short = room_short[:4] + ".." + room_short[-4:]
        else:
            room_short = room[:10]

        lines = []

        # Header line
        header = (
            f"{DIM}{time_str}{RESET} "
            f"{CYAN}{short_model}{RESET} "
            f"{DIM}[{room_short}]{RESET} "
            f"{GREEN}{in_tok:,}→{out_tok:,}tok{RESET} "
            f"{DIM}{latency_s:.1f}s{RESET}"
        )
        lines.append(header)

        # Tool calls
        for tc in tool_calls:
            name = tc.get("name", "?")
            tc_input = tc.get("input", {})
            is_err = tc.get("is_error", False)
            color = RED if is_err else YELLOW
            status = "x" if is_err else "->"

            # Compact input preview
            if name == "shell":
                detail = tc_input.get("command", "")[:80]
            elif name == "file_read":
                detail = tc_input.get("path", "")
            elif name == "file_write":
                detail = tc_input.get("path", "")
            elif name == "file_edit":
                detail = tc_input.get("path", "")
            elif name == "web_search":
                detail = tc_input.get("query", "")[:60]
            elif name == "web_fetch":
                detail = tc_input.get("url", "")[:60]
            elif name == "subagent":
                task = tc_input.get("task", "")[:60]
                model_override = tc_input.get("model", "")
                detail = f"{model_override + ': ' if model_override else ''}{task}"
            elif name == "memory_search":
                detail = tc_input.get("query", "")[:60]
            else:
                detail = str(tc_input)[:60]

            lines.append(f"  {color}{status} {name}{RESET} {DIM}{detail}{RESET}")

        # Content preview (if any, and not just tool calls)
        if preview:
            lines.append(f"  {MAGENTA}>{RESET} {DIM}{preview}{RESET}")

        return "\n".join(lines)

    def current_log_path():
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return logs_dir / f"{agent_name}-{date_str}.jsonl"

    def follow(get_path):
        """Yield new lines from the CURRENT per-day log file, following the
        rotation across UTC midnight and waiting for the file to appear.

        BUG-11: the old tail_file() opened one fd and looped it forever, so
        after midnight -- when the agent starts a new <agent>-<date>.jsonl --
        the monitor went permanently silent (the rotation check had an empty
        body). And if today's file didn't exist yet at start (agent idle since
        midnight), it returned immediately and the command exited 0 after the
        banner, appearing to work while monitoring nothing. This re-resolves
        the path, reopens on date change, and polls for a missing file instead
        of giving up.
        """
        fh = None
        cur = None
        first_open = True
        try:
            while True:
                path = get_path()
                if fh is None or path != cur:
                    if fh is not None:
                        fh.close()
                        first_open = False  # a rotation, not the initial open
                    # Wait for the file to exist rather than exiting.
                    while not path.exists():
                        time.sleep(0.5)
                        path = get_path()
                    fh = open(path, "r")
                    # Initial open: start at end (only new lines). Rotation:
                    # start at the beginning so the new day's early entries
                    # aren't missed.
                    fh.seek(0, 2) if first_open else fh.seek(0, 0)
                    cur = path
                line = fh.readline()
                if line:
                    yield line
                elif get_path() != cur:
                    continue  # date rolled over — reopen on next loop
                else:
                    time.sleep(0.3)
        finally:
            if fh is not None:
                fh.close()

    print(f"{BOLD}Monitoring {agent_name}{RESET} — {logs_dir}", file=sys.stderr)
    print(f"{DIM}Ctrl+C to stop{RESET}\n", file=sys.stderr)

    try:
        for line in follow(current_log_path):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                print(format_entry(entry))
            except json.JSONDecodeError:
                print(f"{DIM}(malformed: {line[:80]}){RESET}")
    except KeyboardInterrupt:
        print(f"\n{DIM}Monitor stopped.{RESET}", file=sys.stderr)


def cmd_showprompt(args):
    """Display the full assembled system prompt and available tools for an agent.

    Only needs the workspace path — does not resolve API keys, so it works
    without the agent's credentials (e.g. when run as a different user).
    """
    import tomllib
    from pathlib import Path
    from openalph.prompt import assemble_prompt
    from openalph.tools import discover_tools

    config_path = CONFIG_DIR / f"{args.agent}.toml"
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with config_path.open("rb") as f:
            toml_data = tomllib.load(f)
        workspace = Path(toml_data.get("workspace", {}).get("path", ""))
        if not workspace.is_dir():
            print(f"Workspace not found: {workspace}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Failed to read config: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract model aliases for prompt assembly
    aliases_section = toml_data.get("model_aliases", {})
    model_aliases = {k: v for k, v in aliases_section.items() if isinstance(v, str)}

    # PHIL-1: honour the agent's own injection_defense setting, so this
    # command shows what the agent ACTUALLY receives. A prompt inspector that
    # ignores a prompt-affecting flag is worse than none -- it is the tool an
    # operator would use to check the claim.
    injection_defense = toml_data.get("agent", {}).get("injection_defense", True)
    if not isinstance(injection_defense, bool):
        injection_defense = True

    # workspace-kdsn.305: CONTINUITY.md (the 8th operator file) is gated on
    # [context] gc_enabled — pass the same flag the live agent passes
    # (agent.py assembles with config.context.gc_enabled). A prompt inspector
    # that ignores a prompt-affecting flag is worse than none (PHIL-1, same
    # principle as injection_defense above). The fail-loud parser raises
    # ConfigError on a bad [context] value rather than previewing a prompt
    # the agent would refuse to build.
    from openalph.config import _parse_context_gc_config
    gc_enabled = _parse_context_gc_config(toml_data).gc_enabled

    prompt = assemble_prompt(
        workspace,
        model_aliases=model_aliases,
        injection_defense=injection_defense,
        gc_enabled=gc_enabled,
    )
    if prompt:
        print(prompt)
    else:
        print("(empty prompt — no workspace files found)", file=sys.stderr)

    # Show tools available via API tool parameter
    tools = discover_tools(workspace)
    if tools:
        print("\n## Available Tools (passed via API, not in prompt)\n")
        for tool in tools:
            params = ", ".join(tool.parameters.get("properties", {}).keys())
            print(f"- **{tool.name}**: {tool.description}")
            if params:
                print(f"  Parameters: {params}")
    else:
        print("\n(no tools configured)", file=sys.stderr)


# ---------------------------------------------------------------------------
# exec — one-shot headless turn (Stigmergy worker driver, bead .149)
# ---------------------------------------------------------------------------
#
# Design invariants (bead149-build-spec.md §3):
#   * EXACTLY ONE json.dumps line on stdout. Every log line, warning, tool
#     notice, and error message goes to stderr. The Stigmergy driver parses
#     the raw stdout as one JSON object — any stray byte breaks
#     classification, so stdout discipline is load-bearing.
#   * No SessionLog, no HeadlessSinks, no chat plumbing: a fresh Agent, a
#     single handle_input, then post-turn introspection (usage, iteration
#     cap sentinel, stop_reason, tool trace).
#   * --model / --max-turns are config-level replacements done BEFORE Agent
#     construction (run_subagent's `replace(config, default_model=model)`
#     pattern).
#   * Exit codes: 0 done / 1 failed / 2 infra / 3 wedged (the JSON is
#     authoritative; the code is a coarse mirror).

# Structural iteration-cap sentinel appended to history by
# Agent.handle_input when max_iterations is exhausted (agent.py). The
# substring (up to the first '.') is what the cap message always starts
# with; matching the prefix keeps detection robust to tail edits.
_EXEC_ITERATION_CAP_SENTINEL = "[SYSTEM: Tool call limit reached."

# Charter-native effort surface → OA `thinking=` kwarg (spec §2.8).
_EXEC_EFFORT_MAP = {"none": "off", "low": "low", "medium": "medium",
                    "xhigh": "xhigh"}

# Stigmergy relay deny marker (bead .147/.134): a machine-readable
# x-stigmergy-deny-reason header and/or a JSON body
# {"error":{"type":"stigmergy_relay_deny","reason":...}}.
_EXEC_DENY_REASON_HEADER = "x-stigmergy-deny-reason"
_EXEC_DENY_TYPE_MARKER = "stigmergy_relay_deny"

# Bound for the result detail string (spec §3.8: 500 chars).
_EXEC_DETAIL_MAX = 500

# Bound for the tool trace (spec §3.7: names + is_error only, cap 50).
_EXEC_TOOL_TRACE_CAP = 50

# Transport / infra error classes: genuinely-forwarded upstream failures
# (spec §3.8). Checked by type-name across the exception + __cause__ chain
# so the check does not hard-import httpx/openai here (they are already
# process deps via openalph.provider, but name-matching keeps the seam
# import-free and resilient to provider re-raises).
_EXEC_INFRA_EXC_NAMES = {
    "RemoteProtocolError",   # httpx.RemoteProtocolError
    "APITimeoutError",       # openai.APITimeoutError (httpx.TimeoutException)
    "ConnectError",          # httpx.ConnectError
    "APIConnectionError",    # openai.APIConnectionError (wraps httpx connect)
    "TimeoutException",      # httpx.TimeoutException base
}
# HTTP-error marker classes: any of these in the chain is an upstream/relay
# HTTP failure; the deny marker then decides failed-vs-infra.
_EXEC_HTTP_EXC_NAMES = {
    "APIStatusError",        # openai.APIStatusError
    "APIError",              # openai.APIError (base; has .response)
    "HTTPStatusError",       # httpx.HTTPStatusError
}


def _exec_bounded_detail(detail: str) -> str:
    """Bound the result detail to _EXEC_DETAIL_MAX chars (spec §3.8)."""
    if len(detail) <= _EXEC_DETAIL_MAX:
        return detail
    return detail[:_EXEC_DETAIL_MAX]


def _resolve_exec_tools(names: list[str]) -> list:
    """Resolve --tools names against the BUILTIN_TOOLS registry.

    Bypasses discover_tools entirely (the worker workspace has no tools/
    dir). Raises ValueError on any unknown name; the caller maps that to
    stderr + exit 1.
    """
    from openalph.tools import BUILTIN_TOOLS, ToolDef

    defs = []
    for name in names:
        if not name:
            continue
        if name not in BUILTIN_TOOLS:
            raise ValueError(
                f"Unknown tool '{name}'. Available tools: "
                f"{', '.join(sorted(BUILTIN_TOOLS.keys()))}"
            )
        builtin = BUILTIN_TOOLS[name]
        defs.append(ToolDef(
            name=name,
            description=builtin["description"],
            parameters=builtin["parameters"].copy(),
            config=builtin["config"].copy(),
        ))
    return defs


def _exec_ceil_to_int(v) -> int:
    """Coerce a usage value to a non-negative int (zeros on failure)."""
    try:
        n = int(v)
    except (TypeError, ValueError):
        return 0
    return n if n >= 0 else 0


def _exec_partial_usage(agent, room_id: str) -> dict:
    """Best-effort 4-key usage {in, cached, out, reasoning} for a room.

    Reads agent.last_turn_usage(room_id) (the real Agent's per-turn delta:
    input/output/cache_read/cache_creation tokens). Zeros when unavailable —
    never fabricated, never negative. The relay JSONL is the authoritative
    meter; this is only the driver's view of what the turn reported.
    """
    ltu = None
    try:
        ltu = agent.last_turn_usage(room_id)
    except Exception:
        ltu = None
    if not isinstance(ltu, dict):
        return {"in": 0, "cached": 0, "out": 0, "reasoning": 0}
    return {
        "in": _exec_ceil_to_int(ltu.get("input_tokens", 0)),
        "cached": _exec_ceil_to_int(ltu.get("cache_read_tokens", 0)),
        "out": _exec_ceil_to_int(ltu.get("output_tokens", 0)),
        # OA does not surface a separate reasoning-token count; reasoning
        # tokens are folded into output_tokens by the provider adapters.
        # Never fabricated — fixed 0 (spec §3.7: never fabricated).
        "reasoning": 0,
    }


def _exec_iteration_cap_reached(agent, room_id: str) -> bool:
    """Structural check: is the iteration-cap sentinel in this room's history?

    On max_iterations exhaustion Agent.handle_input appends the exact
    sentinel to history (agent.py) and forces a no-tools summary call. This
    check finds it deterministically.
    """
    try:
        history = agent.history(room_id)
    except Exception:
        return False
    for m in history:
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, str) and _EXEC_ITERATION_CAP_SENTINEL in content:
            return True
    return False


def _exec_collect_tool_trace() -> list:
    """Return (trace_list, async_on_tool_call) — trace holds {name, is_error}
    entries bounded to _EXEC_TOOL_TRACE_CAP, names-only (post-mortem aid)."""
    trace = []

    async def _on_tool_call(call_id, name, input_data, result, is_error):
        if len(trace) < _EXEC_TOOL_TRACE_CAP:
            trace.append({"name": name, "is_error": bool(is_error)})

    return trace, _on_tool_call


def _exec_find_deny_reason(exc: BaseException) -> str | None:
    """Walk the exception + __cause__ chain for a Stigmergy relay deny.

    Returns the reason string if found, else None. Detects BOTH the
    x-stigmergy-deny-reason header (any object carrying .response.headers or
    .headers) and the machine-readable body
    {"error":{"type":"stigmergy_relay_deny","reason":...}} (any object
    carrying .response with a readable .text/.body or .body — the body may
    be a str/bytes or an already-parsed JSON dict).
    """
    seen = set()
    chain = []
    cur = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        chain.append(cur)
        cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)

    for obj in chain:
        # Header form: .response.headers or a bare .headers mapping.
        for headers in _exec_candidate_headers(obj):
            try:
                reason = headers.get(_EXEC_DENY_REASON_HEADER)
            except Exception:
                reason = None
            if isinstance(reason, str) and reason:
                return reason

        # Body form: every candidate body location (the SDK's .body may be an
        # already-parsed dict that is NOT the deny body, so scan ALL of them
        # and return the first one carrying the marker).
        for parsed in _exec_candidate_bodies(obj):
            err = parsed.get("error")
            if isinstance(err, dict) and \
                    err.get("type") == _EXEC_DENY_TYPE_MARKER:
                reason = err.get("reason")
                if isinstance(reason, str) and reason:
                    return reason
    return None


def _exec_candidate_headers(obj) -> list:
    """Return header mappings reachable from an exception object (best-effort)."""
    out = []
    resp = getattr(obj, "response", None)
    for h in (getattr(obj, "headers", None),
              getattr(resp, "headers", None)):
        if isinstance(h, dict):
            out.append(h)
        elif h is not None and hasattr(h, "get"):
            out.append(h)
    return out


def _exec_candidate_bodies(obj) -> list:
    """Best-effort extraction of error bodies from an exception object.

    Yields a parsed JSON dict for EVERY candidate location that parses as a
    JSON object: the SDK's pre-parsed .body (dict), the httpx response
    .text/.content (str/bytes of JSON), and a bare .message. The caller
    returns the first candidate carrying the deny marker, so a non-marker
    body at one location never shadows the marker at another.
    """
    resp = getattr(obj, "response", None)
    cands = [getattr(obj, "body", None),
             getattr(resp, "text", None),
             getattr(resp, "content", None),
             getattr(obj, "message", None)]
    out = []
    for cand in cands:
        if isinstance(cand, dict):
            out.append(cand)
        elif isinstance(cand, str) and cand:
            try:
                parsed = json.loads(cand)
            except (ValueError, TypeError):
                continue
            if isinstance(parsed, dict):
                out.append(parsed)
        elif isinstance(cand, (bytes, bytearray)):
            try:
                parsed = json.loads(bytes(cand).decode("utf-8", "replace"))
            except (ValueError, TypeError, UnicodeDecodeError):
                continue
            if isinstance(parsed, dict):
                out.append(parsed)
    return out


def _exec_chain_has_infra_transport(exc: BaseException) -> bool:
    """True if the exception + __cause__/__context__ chain carries a
    transport-level error (RemoteProtocolError / APITimeout / ConnectError /
    APIConnectionError / TimeoutException)."""
    seen = set()
    cur = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if type(cur).__name__ in _EXEC_INFRA_EXC_NAMES:
            return True
        cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
    return False


def _exec_chain_has_http_error(exc: BaseException) -> bool:
    """True if the chain carries an HTTP-status error (openai.APIStatusError /
    httpx.HTTPStatusError / openai.APIError)."""
    seen = set()
    cur = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if type(cur).__name__ in _EXEC_HTTP_EXC_NAMES:
            return True
        cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
    return False


def _exec_classify_exception(exc: BaseException) -> tuple[str, str | None, str]:
    """Classify a handle_input exception per spec §3.8.

    Returns (status, deny_reason, detail):
      * deny marker found        -> ("failed", reason, "relay-deny:<reason>")
      * HTTP error, no marker    -> ("infra", None, <bounded str(exc)>)
      * transport error          -> ("infra", None, <bounded str(exc)>)
      * anything else            -> ("failed", None, <bounded str(exc)>)
    """
    reason = _exec_find_deny_reason(exc)
    if reason is not None:
        return "failed", reason, f"relay-deny:{reason}"
    if _exec_chain_has_http_error(exc) or _exec_chain_has_infra_transport(exc):
        return "infra", None, _exec_bounded_detail(str(exc))
    return "failed", None, _exec_bounded_detail(str(exc))


def _exec_read_task(args) -> str:
    """Read the task from --task-file (or stdin for -). Raises on missing
    file; the caller maps that to stderr + exit 1."""
    if args.task_file == "-":
        return sys.stdin.read()
    path = Path(args.task_file)
    if not path.is_file():
        raise FileNotFoundError(f"Task file not found: {args.task_file}")
    return path.read_text()


def _exec_sanitize_room(label: str | None) -> str:
    """Resolve --room to a room id; default _exec."""
    if not label:
        return "_exec"
    return _sanitize_room_label(label)


def cmd_exec(args):
    """One-shot headless turn: fresh Agent, single handle_input, one JSON line.

    See the module section above for the load-bearing stdout/stderr and
    classification invariants. Exits 0 done / 1 failed / 2 infra / 3 wedged.
    """
    # stdout discipline: every log line goes to stderr, never stdout.
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    def _fail(msg: str, code: int = 1) -> None:
        print(f"exec: {msg}", file=sys.stderr)
        sys.exit(code)

    from dataclasses import replace
    from openalph.agent import Agent

    # 1. Config (fail loud, exit 1, stderr).
    try:
        config = load_agent_config(args.agent)
    except ConfigError as e:
        _fail(f"Config error: {e}", 1)

    # 2. Tool resolution (bypass discover_tools; unknown -> exit 1).
    if args.tools:
        names = [n.strip() for n in args.tools.split(",") if n.strip()]
        try:
            resolved_tools = _resolve_exec_tools(names)
        except ValueError as e:
            _fail(str(e), 1)
    else:
        resolved_tools = None

    # 3. Pre-construction config overrides (run_subagent replace pattern).
    if args.model is not None:
        config = replace(config, default_model=args.model)
    if args.max_turns is not None:
        config = replace(config, max_iterations=args.max_turns)
    # The Spotter is an autonomous monitor that fires its own complete()
    # calls (spotter_model) around turns. A worker dispatch is ONE
    # deterministic handle_input metered against the capability's
    # max_calls=driver_turns — the spotter would consume that budget and add
    # nondeterminism the driver does not account for. Disable it for exec
    # (spec §3: "fewer failure modes"). Matrix/chat agents are unaffected.
    config = replace(config, spotter_enabled=False)

    # 4. Effort -> thinking kwarg.
    thinking = _EXEC_EFFORT_MAP.get(args.effort) if args.effort else None

    # 5. Task text (missing file -> exit 1, stderr, empty stdout).
    try:
        task = _exec_read_task(args)
    except (FileNotFoundError, OSError) as e:
        _fail(str(e), 1)

    # Room label (default _exec).
    try:
        room_id = _exec_sanitize_room(args.room)
    except ValueError as e:
        _fail(str(e), 1)

    agent = Agent(config)

    # --tools: install the resolved ToolDefs on the agent (fresh agent, so
    # this fully determines the tool set; discovery is bypassed by design).
    if resolved_tools is not None:
        agent.tools = resolved_tools

    # Tool trace capture (names + is_error only, bounded).
    tool_trace, on_tool_call = _exec_collect_tool_trace()

    content = None
    status = "done"
    deny_reason = None
    ceiling_trip = None
    detail = ""

    async def _run():
        return await agent.handle_input(
            task, room_id,
            on_tool_call=on_tool_call,
            thinking=thinking,
        )

    try:
        content = asyncio.run(_run())
    except SystemExit:
        raise
    except KeyboardInterrupt:
        _fail("interrupted", 130)
    except Exception as exc:
        status, deny_reason, detail = _exec_classify_exception(exc)
        logger.warning("exec handle_input failed: %s", exc)

    # Post-turn introspection.
    usage = _exec_partial_usage(agent, room_id)
    stop_reason = None
    try:
        stop_reason = agent.last_stop_reason(room_id)
    except Exception:
        stop_reason = None

    # Iteration cap -> structural ceiling trip (failed, not infra: this is
    # Stigmergy's own per-dispatch driver-turn budget speaking, the exact
    # analog of claude-code's error_max_turns).
    if _exec_iteration_cap_reached(agent, room_id):
        ceiling_trip = "driver_turns"
        status = "failed"

    result = {
        "status": status,
        "content": content,
        "usage": usage,
        "stop_reason": stop_reason,
        "ceiling_trip": ceiling_trip,
        "deny_reason": deny_reason,
        "tool_trace": tool_trace,
        "detail": detail,
    }
    print(json.dumps(result), flush=True)

    if status == "done":
        sys.exit(0)
    if status == "infra":
        sys.exit(2)
    if status == "wedged":
        sys.exit(3)
    sys.exit(1)


# ---------------------------------------------------------------------------
# CLI helpers — kdsn.237 Phase 1 (headless session support)
# ---------------------------------------------------------------------------

def _generate_session_name() -> str:
    """Generate a random 3-word bip39 mnemonic as a session name."""
    import secrets
    from pathlib import Path
    wordlist_path = Path(__file__).parent / "data" / "bip39-english.txt"
    words = [w.strip() for w in wordlist_path.read_text().splitlines() if w.strip()]
    return " ".join(secrets.choice(words) for _ in range(3))


def _sanitize_room_label(label: str) -> str:
    """Sanitize a --room label for use as a session key. Rejects path traversal."""
    if not label:
        raise ValueError("Room label cannot be empty")
    if any(c in label for c in ("/", "\\", "..")):
        raise ValueError(f"Room label contains forbidden characters: {label!r}")
    import re as _re
    if not _re.fullmatch(r'[a-zA-Z0-9_\-.]+', label):
        raise ValueError(f"Room label contains invalid characters: {label!r}")
    return label


def _setup_cli_session(config, agent, room_id, *, explicit_room=False):
    """Create or resume a CLI session. Returns (session_log, room_name, is_new)."""
    from openalph.session import SessionLog
    # Resolve user_id: Phase 1 agents have config.user_id; legacy agents use matrix.user_id
    uid = getattr(config, "user_id", None) or (config.matrix.user_id if config.matrix else "cli")
    sl = SessionLog(config.workspace, uid)
    entries = sl.read(room_id)

    if entries:
        # Resume: find stored room_name from session_start
        room_name = None
        for e in entries:
            if e.get("role") == "system" and e.get("event") == "session_start":
                room_name = e.get("room_name")
                break
        # Rehydrate history + usage + reminders
        history = agent.history(room_id)
        history.clear()
        history.extend(sl.build_context(room_id))
        agent.restore_usage(room_id, sl.usage_totals(room_id))
        try:
            agent.rehydrate_reminders(room_id, entries)
        except AttributeError:
            pass
        # Log resume
        sl.append(
            role="system", sender=uid, room=room_id,
            event="session_resume", detail="CLI session resumed",
        )
        return (sl, room_name, False)

    # New session
    room_name = room_id if explicit_room else _generate_session_name()
    sl.append(
        role="system", sender=uid, room=room_id,
        event="session_start", room_name=room_name, detail="CLI session started",
    )
    return (sl, room_name, True)


def _check_workspace_writable(config) -> bool:
    """Check if the workspace is writable by the current user."""
    ws = Path(config.workspace)
    if not ws.exists():
        print(
            f"⚠️ Workspace {ws} is not writable by current user ({getpass.getuser()}).\n"
            f"   Run as the agent user: sudo -u oa-{config.name} openalph chat {config.name}",
            file=sys.stderr,
        )
        return False
    # os.access() is bypassed by root; also check stat bits for any write permission
    st = ws.stat()
    if not os.access(str(ws), os.W_OK) or not (st.st_mode & 0o222):
        print(
            f"⚠️ Workspace {ws} is not writable by current user ({getpass.getuser()}).\n"
            f"   Run as the agent user: sudo -u oa-{config.name} openalph chat {config.name}",
            file=sys.stderr,
        )
        return False
    return True


def _get_truncate_limit(*, args=None) -> int | None:
    """Get the tool-result preview truncation limit.

    Precedence: --truncate arg > OPENALPH_TRUNCATE env > default 200.
    Returns None for 0 (no truncation).
    """
    # CLI arg takes precedence
    if args is not None and getattr(args, 'truncate', None) is not None:
        val = args.truncate
        if val == 0:
            return None
        return val
    # Env var
    env_val = os.environ.get("OPENALPH_TRUNCATE")
    if env_val:
        try:
            val = int(env_val)
            if val == 0:
                return None
            return val
        except ValueError:
            pass  # fall through to default
    # Default
    return 200


async def _process_cli_line(agent, session_log, callbacks, room_id, config, line, *, truncate_limit=200):
    """Process one line: slash command or regular message. Returns (response, should_exit)."""
    from openalph.session import persist_assistant_turn
    stripped = line.strip()
    if not stripped:
        return (None, False)

    if stripped in ("/quit", "/exit"):
        return (None, True)

    if stripped == "/help":
        print("/status    - agent status", file=sys.stderr)
        print("/quit      - exit", file=sys.stderr)
        print("/showprompt - display system prompt", file=sys.stderr)
        print("/model <name> - switch model", file=sys.stderr)
        print("--room <name>  - use at launch for separate sessions", file=sys.stderr)
        print("", file=sys.stderr)
        print("Sessions are persisted to local JSONL.", file=sys.stderr)
        return (None, False)

    if stripped == "/status":
        s = agent.status(room_id)
        parts = [f"{s.get('name', config.name)} ({s.get('model', config.default_model)})"]
        if "turns" in s:
            parts.append(f"{s['turns']} turns")
        if "context_pct" in s:
            parts.append(f"context {s['context_pct']}%")
        if "context_tokens" in s:
            parts.append(f"~{s['context_tokens']:,} tokens")
        if "context_remaining" in s:
            parts.append(f"{s['context_remaining']:,} remaining")
        if "total_tool_calls" in s:
            parts.append(f"{s['total_tool_calls']} tool calls")
        print(" - ".join(parts), file=sys.stderr)
        return (None, False)

    if stripped == "/showprompt":
        print(agent.system_prompt, file=sys.stderr)
        return (None, False)

    if stripped.startswith("/model"):
        parts = stripped.split(None, 1)
        if len(parts) == 1:
            print(f"Current model: {config.default_model}", file=sys.stderr)
        else:
            result = agent.switch_model(parts[1], room_id)
            print(f"Model set: {result}", file=sys.stderr)
            session_log.append(
                role="system", sender=session_log.agent_user_id, room=room_id,
                event="model_override", detail=parts[1],
            )
        return (None, False)

    if stripped.startswith("/effort"):
        print("Effort override not supported in CLI mode.", file=sys.stderr)
        return (None, False)

    # Regular message
    session_log.append(
        role="user", sender="operator", room=room_id, content=line,
    )

    async def _tool_notice(call_id, name, input_data, result, is_error):
        status = "error" if is_error else "ok"
        if truncate_limit is not None:
            preview = str(result)[:truncate_limit].replace('\n', ' ')
        else:
            preview = str(result).replace('\n', ' ')
        print(f"🔧 {name}: ({status}) {preview}", file=sys.stderr, flush=True)
        session_log.append(
            role="tool", sender=session_log.agent_user_id, room=room_id,
            call_id=call_id, name=name, output=result, is_error=is_error,
        )

    async def _tool_intent(tool_calls, content_text):
        from openalph.session import persist_assistant_turn
        persist_assistant_turn(agent, session_log, room_id, content=content_text, tool_calls=tool_calls)

    try:
        response = await agent.handle_input(
            line, room_id,
            on_tool_call=_tool_notice,
            on_tool_intent=_tool_intent,
            callbacks=callbacks,
        )
    except Exception as exc:
        from openalph.agent import ContextOverflowError
        if isinstance(exc, ContextOverflowError):
            print(f"\n! Context overflow - ~{exc.current_tokens:,} / {exc.max_tokens:,} tokens. "
                  f"Restart with --room for a fresh session.", file=sys.stderr)
        elif isinstance(exc, asyncio.CancelledError):
            print("\nCancelled.", file=sys.stderr)
            raise
        else:
            print(f"\nError: {exc}", file=sys.stderr)
        return (None, False)

    persist_assistant_turn(agent, session_log, room_id, content=response)
    return (response, False)


def cmd_chat(args):
    """Interactive CLI session with an agent (Phase 1: persisted sessions)."""
    from openalph.agent import Agent
    from openalph.callbacks import HeadlessSinks, build_callbacks

    # Suppress library noise (httpx, anthropic, etc.) - tool notices handle
    # user-facing feedback. Only show errors unless -v was passed.
    if not args.verbose:
        logging.getLogger().setLevel(logging.ERROR)

    try:
        config = load_agent_config(args.agent)
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)

    if not _check_workspace_writable(config):
        sys.exit(1)

    agent = Agent(config)

    explicit_room = bool(args.room)
    room_id = _sanitize_room_label(args.room) if args.room else "_cli"

    session_log, room_name, is_new = _setup_cli_session(
        config, agent, room_id, explicit_room=explicit_room,
    )

    uid = getattr(config, "user_id", None) or (config.matrix.user_id if config.matrix else "cli")
    sinks = HeadlessSinks(session_log=session_log, agent_user_id=uid)
    callbacks = build_callbacks(
        agent, room_id, sinks,
        turn_source=None, session_log=session_log, room_name=room_name,
    )

    # Ready banner
    status = "new" if is_new else "resumed"
    print(f"{config.name} ready (model: {config.default_model})  [{room_name}] ({status}). "
          f"Type /help for commands, Ctrl+D to exit.", file=sys.stderr)

    async def chat_loop():
        import readline  # noqa: F401  -- enables arrow keys, history

        truncate_limit = _get_truncate_limit(args=args)

        while True:
            try:
                prompt = f"\n{config.name}> "
                line = input(prompt)
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.", file=sys.stderr)
                break

            response, should_exit = await _process_cli_line(
                agent, session_log, callbacks, room_id, config, line,
                truncate_limit=truncate_limit,
            )
            if should_exit:
                print("Goodbye.", file=sys.stderr)
                break
            if response is not None:
                print(f"\n{response}")

    asyncio.run(chat_loop())


def main(argv=None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    dispatch = {
        "start": cmd_start,
        "stop": cmd_stop,
        "restart": cmd_restart,
        "status": cmd_status,
        "list": cmd_list,
        "logs": cmd_logs,
        "new-agent": cmd_new_agent,
        "run": cmd_run,
        "monitor": cmd_monitor,
        "chat": cmd_chat,
        "showprompt": cmd_showprompt,
        "exec": cmd_exec,
    }

    handler = dispatch[args.command]
    try:
        handler(args)
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0
