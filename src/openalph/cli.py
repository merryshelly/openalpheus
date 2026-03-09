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
            check=True,
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
    subprocess.run(cmd, check=True)


def cmd_new_agent(args):
    create_agent(args.name, dry_run=args.dry_run)


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
        logger.info("Agent '%s' running (model: %s)", config.name, config.model)
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
    }

    handler = dispatch[args.command]
    handler(args)
    return 0
