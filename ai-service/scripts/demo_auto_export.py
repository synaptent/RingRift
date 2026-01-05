#!/usr/bin/env python3
"""Demo script for auto-cascade NPZ export.

This script demonstrates how the AutoExportDaemon automatically triggers
NPZ export when selfplay completes with enough games.

Usage:
    python scripts/demo_auto_export.py
"""

import asyncio
import logging
from pathlib import Path

from app.coordination.auto_export_daemon import (
    AutoExportDaemon,
    AutoExportConfig,
    get_auto_export_daemon,
)
from app.coordination.event_emission_helpers import safe_emit_event_async

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def simulate_selfplay_events():
    """Simulate selfplay completion events to trigger auto-export."""
    logger.info("Simulating selfplay events...")

    # Simulate multiple selfplay batches
    configs = [
        ("hex8", 2, 50),    # 50 games
        ("hex8", 2, 30),    # 30 more games (total: 80)
        ("hex8", 2, 25),    # 25 more games (total: 105 - triggers export!)
        ("square8", 2, 120), # 120 games (triggers export immediately)
    ]

    for board_type, num_players, games in configs:
        logger.info(f"\nEmitting SELFPLAY_COMPLETE: {board_type}_{num_players}p with {games} games")

        await safe_emit_event_async(
            "SELFPLAY_COMPLETE",
            {
                "task_id": f"demo_{board_type}_{num_players}p",
                "board_type": board_type,
                "num_players": num_players,
                "games_generated": games,
                "success": True,
                "node_id": "demo_node",
                "duration_seconds": 30.0,
                "selfplay_type": "standard",
            },
            context="demo_auto_export",
        )

        # Wait to observe the daemon's response
        await asyncio.sleep(2)


async def monitor_daemon_status(daemon, duration=30):
    """Monitor daemon status for a duration."""
    logger.info(f"\nMonitoring daemon status for {duration} seconds...")

    for i in range(duration):
        status = daemon.get_status()
        if status["configs_tracked"] > 0:
            logger.info(f"\n=== Daemon Status (t={i}s) ===")
            for config, state in status["states"].items():
                logger.info(
                    f"{config}: {state['games_pending']} games pending, "
                    f"in_progress={state['in_progress']}, "
                    f"failures={state['failures']}"
                )
        await asyncio.sleep(1)


async def demo_basic_usage():
    """Demo: Basic usage with default config."""
    logger.info("=" * 60)
    logger.info("DEMO 1: Basic Usage with Default Config")
    logger.info("=" * 60)

    # Get singleton daemon (uses default config)
    daemon = get_auto_export_daemon()
    await daemon.start()

    try:
        # Simulate selfplay events
        await simulate_selfplay_events()

        # Monitor daemon status
        await monitor_daemon_status(daemon, duration=15)

    finally:
        await daemon.stop()
        logger.info("\nDaemon stopped")


async def demo_custom_config():
    """Demo: Custom configuration with lower threshold."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 2: Custom Config with Lower Threshold (50 games)")
    logger.info("=" * 60)

    # Create custom config
    config = AutoExportConfig(
        min_games_threshold=50,        # Lower threshold for demo
        export_cooldown_seconds=5,     # Shorter cooldown
        max_concurrent_exports=1,      # Single export at a time
        output_dir=Path("data/demo_exports"),
        persist_state=False,           # Disable state persistence for demo
    )

    daemon = AutoExportDaemon(config)
    await daemon.start()

    try:
        # Emit event that should trigger immediately
        logger.info("\nEmitting SELFPLAY_COMPLETE with 75 games (> 50 threshold)")
        await safe_emit_event_async(
            "SELFPLAY_COMPLETE",
            {
                "task_id": "demo_hex8_4p",
                "board_type": "hex8",
                "num_players": 4,
                "games_generated": 75,
                "success": True,
                "node_id": "demo_node",
                "duration_seconds": 40.0,
            },
            context="demo_auto_export",
        )

        # Monitor for response
        await monitor_daemon_status(daemon, duration=10)

    finally:
        await daemon.stop()
        logger.info("\nDaemon stopped")


async def demo_status_monitoring():
    """Demo: Monitoring daemon status."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 3: Status Monitoring")
    logger.info("=" * 60)

    daemon = get_auto_export_daemon()
    await daemon.start()

    try:
        # Emit some events
        for i in range(3):
            await safe_emit_event_async(
                "SELFPLAY_COMPLETE",
                {
                    "task_id": f"demo_batch_{i}",
                    "board_type": "square19",
                    "num_players": 2,
                    "games_generated": 40,
                    "success": True,
                },
                context="demo_auto_export",
            )
            await asyncio.sleep(1)

        # Get detailed status
        status = daemon.get_status()
        logger.info(f"\nDaemon Status:")
        logger.info(f"  Running: {status['running']}")
        logger.info(f"  Configs tracked: {status['configs_tracked']}")
        logger.info(f"\nPer-Config States:")
        for config, state in status["states"].items():
            logger.info(f"\n  {config}:")
            logger.info(f"    Pending games: {state['games_pending']}")
            logger.info(f"    Last export: {state['last_export']}")
            logger.info(f"    Total samples: {state['total_samples']}")
            logger.info(f"    In progress: {state['in_progress']}")

    finally:
        await daemon.stop()


async def main():
    """Run all demos."""
    try:
        # Run demos sequentially
        await demo_basic_usage()
        await demo_custom_config()
        await demo_status_monitoring()

        logger.info("\n" + "=" * 60)
        logger.info("All demos completed successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
