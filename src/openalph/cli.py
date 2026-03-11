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


def cmd_start(args):
    subprocess.run(["systemctl", "start", f"openalph@{args.agent}.service"], check=True)


def cmd_stop(args):
    subprocess.run(["systemctl", "stop", f"openalph@{args.agent}.service"], check=True)


def cmd_restart(args):
    subprocess.run(["systemctl", "restart", f"openalph@{args.agent}.service"], check=True)


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
    ops = create_agent(args.name, dry_run=args.dry_run)
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
    print("not yet implemented")


def cmd_showprompt(args):
    """Display the full assembled system prompt and available tools for an agent."""
    from openalph.prompt import assemble_prompt
    from openalph.tools import discover_tools

    try:
        config = load_agent_config(args.agent)
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)

    prompt = assemble_prompt(config.workspace)
    if prompt:
        print(prompt)
    else:
        print("(empty prompt — no workspace files found)", file=sys.stderr)

    # Show tools available via API tool parameter
    tools = discover_tools(config.workspace)
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
        import readline  # enables arrow keys, history

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
    handler(args)
    return 0
