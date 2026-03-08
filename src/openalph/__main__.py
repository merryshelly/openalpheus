"""CLI entry point for OpenAlph.

Usage: python -m openalph <config.toml>

Loads agent config, connects to Matrix, runs until SIGINT/SIGTERM.
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

from openalph.agent import Agent
from openalph.config import ConfigError, load_config
from openalph.matrix import MatrixBot

logger = logging.getLogger("openalph")


def main():
    parser = argparse.ArgumentParser(
        prog="openalph",
        description="Run an OpenAlph agent.",
    )
    parser.add_argument("config", type=Path, help="Path to agent TOML config file")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    try:
        config = load_config(args.config)
    except ConfigError as e:
        logger.error("Config error: %s", e)
        sys.exit(1)
    except FileNotFoundError:
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)

    if config.matrix is None:
        logger.error("No [matrix] section in config — Matrix is required")
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

        # Start bot in background, wait for shutdown signal
        bot_task = asyncio.create_task(bot.start())
        logger.info("Agent '%s' running (model: %s)", config.name, config.model)

        await stop  # Block until signal

        logger.info("Shutting down...")
        await bot.stop()

        # Give the sync loop a moment to exit after stop flag
        try:
            await asyncio.wait_for(bot_task, timeout=5.0)
        except asyncio.TimeoutError:
            bot_task.cancel()
            try:
                await bot_task
            except asyncio.CancelledError:
                pass

    asyncio.run(run())


if __name__ == "__main__":
    main()
