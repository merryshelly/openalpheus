"""CLI for OpenAlph — manage agents via systemctl and run in dev mode."""

import argparse
import asyncio
import logging
import signal
import subprocess
import sys

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
            status = "✗" if is_err else "→"

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
            lines.append(f"  {MAGENTA}▸{RESET} {DIM}{preview}{RESET}")

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

    prompt = assemble_prompt(
        workspace,
        model_aliases=model_aliases,
        injection_defense=injection_defense,
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


def cmd_chat(args):
    """Interactive CLI session with an agent."""
    from openalph.agent import Agent, ContextOverflowError

    # Suppress library noise (httpx, anthropic, etc.) — tool notices handle
    # user-facing feedback. Only show errors unless -v was passed.
    if not args.verbose:
        logging.getLogger().setLevel(logging.ERROR)

    try:
        config = load_agent_config(args.agent)
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)

    agent = Agent(config)
    room_id = args.room or "_cli"

    # Wire tool notices to stderr
    async def tool_notice(call_id, name, input_data, result, is_error):
        status = "error" if is_error else "ok"
        preview = str(result)[:80].replace('\n', ' ')
        print(f"🔧 {name}: ({status}) {preview}", file=sys.stderr, flush=True)

    async def chat_loop():
        import readline  # noqa: F401  -- enables arrow keys, history

        print(f"{config.name} ready (model: {config.default_model}). "
              f"Type /help for commands, Ctrl+D to exit.", file=sys.stderr)

        while True:
            try:
                line = input(f"\n{config.name}> ")
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.", file=sys.stderr)
                break

            line = line.strip()
            if not line:
                continue

            # Commands
            if line == "/quit" or line == "/exit":
                print("Goodbye.", file=sys.stderr)
                break

            if line == "/help":
                print("/status        — agent status", file=sys.stderr)
                print("/quit          — exit", file=sys.stderr)
                print("--room <name>  — use at launch for separate sessions", file=sys.stderr)
                print("", file=sys.stderr)
                print("Sessions are in-memory only — history is lost on exit.", file=sys.stderr)
                continue

            if line == "/status":
                s = agent.status(room_id)
                print(f"{s['name']} ({s['model']}) — "
                      f"{s['turns']} turns, ~{s['context_tokens']:,} tokens "
                      f"({s['context_pct']}%), "
                      f"{s['total_tool_calls']} tool calls", file=sys.stderr)
                continue

            # Regular message
            try:
                response = await agent.handle_input(line, room_id, on_tool_call=tool_notice)
                print(f"\n{response}")
            except ContextOverflowError as e:
                print(f"\n⚠️ Context overflow — ~{e.current_tokens:,} / "
                      f"{e.max_tokens:,} tokens. Restart with --room for a fresh session.",
                      file=sys.stderr)
            except asyncio.CancelledError:
                print("\nCancelled.", file=sys.stderr)
            except Exception as e:
                print(f"\nError: {e}", file=sys.stderr)
            finally:
                pass

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
