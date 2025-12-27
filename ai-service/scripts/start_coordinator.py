#!/usr/bin/env python3
"""Start coordinator with all required daemons.

December 27, 2025: Created as part of infrastructure fix plan.

This script starts the coordinator-specific daemons on the mac-studio
or any designated coordinator node. It's a thin wrapper around master_loop.py
that ensures:
1. The correct daemon profile is used for coordinators
2. Intensive daemons (selfplay, training, etc.) are filtered out
3. Sync daemons run with PULL strategy for data recovery
4. Health monitoring is enabled

Usage:
    # Run coordinator daemons
    python scripts/start_coordinator.py

    # Watch mode (show status only)
    python scripts/start_coordinator.py --watch

    # Dry run (show what would start)
    python scripts/start_coordinator.py --dry-run

    # As a systemd/launchd service:
    nohup python scripts/start_coordinator.py > logs/coordinator.log 2>&1 &

Environment Variables:
    RINGRIFT_IS_COORDINATOR=true  (set automatically by this script)
    RINGRIFT_SYNC_STRATEGY=pull   (use pull strategy for coordinator)
    RINGRIFT_NODE_ID=<hostname>   (auto-detected if not set)
"""

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

# Ensure ai-service is in Python path
ai_service_root = Path(__file__).parent.parent
sys.path.insert(0, str(ai_service_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_coordinator_environment() -> None:
    """Set up environment variables for coordinator mode."""
    # Force coordinator mode
    os.environ["RINGRIFT_IS_COORDINATOR"] = "true"

    # Use PULL sync strategy for coordinator
    os.environ.setdefault("RINGRIFT_SYNC_STRATEGY", "pull")

    # Set node ID if not already set
    import socket
    if not os.environ.get("RINGRIFT_NODE_ID"):
        hostname = socket.gethostname().split(".")[0].lower()
        # Normalize common coordinator hostnames
        if "mac-studio" in hostname or "armand" in hostname:
            os.environ["RINGRIFT_NODE_ID"] = "mac-studio"
        elif "coordinator" in hostname:
            os.environ["RINGRIFT_NODE_ID"] = "coordinator"
        else:
            os.environ["RINGRIFT_NODE_ID"] = hostname

    logger.info(f"Coordinator environment configured:")
    logger.info(f"  RINGRIFT_IS_COORDINATOR: {os.environ['RINGRIFT_IS_COORDINATOR']}")
    logger.info(f"  RINGRIFT_SYNC_STRATEGY: {os.environ.get('RINGRIFT_SYNC_STRATEGY')}")
    logger.info(f"  RINGRIFT_NODE_ID: {os.environ.get('RINGRIFT_NODE_ID')}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Start coordinator daemons for RingRift AI service"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch mode - show status without running daemons",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would start without actually starting",
    )
    parser.add_argument(
        "--daemon-profile",
        type=str,
        default="standard",
        choices=["minimal", "standard", "full", "sync-only"],
        help="Daemon profile to use (default: standard)",
    )
    args = parser.parse_args()

    # Set up coordinator environment
    setup_coordinator_environment()

    # Verify we're on a coordinator node
    from app.config.env import env
    if not env.is_coordinator:
        logger.warning(
            "This script is intended for coordinator nodes. "
            "RINGRIFT_IS_COORDINATOR was set to true to override."
        )

    # Show what daemons would run
    if args.dry_run:
        from app.coordination.daemon_types import DaemonType

        # Get the daemons that would be started
        # Coordinator should NOT run intensive daemons
        intensive = {
            DaemonType.IDLE_RESOURCE,
            DaemonType.TRAINING_NODE_WATCHER,
            DaemonType.AUTO_EXPORT,
            DaemonType.TOURNAMENT_DAEMON,
            DaemonType.EVALUATION,
            DaemonType.AUTO_PROMOTION,
            DaemonType.QUEUE_POPULATOR,
            DaemonType.UTILIZATION_OPTIMIZER,
        }

        # Standard profile daemons minus intensive ones
        coordinator_daemons = [
            DaemonType.EVENT_ROUTER,
            DaemonType.HEALTH_SERVER,
            DaemonType.NODE_HEALTH_MONITOR,
            DaemonType.FEEDBACK_LOOP,
            DaemonType.DATA_PIPELINE,
            DaemonType.AUTO_SYNC,
            DaemonType.ELO_SYNC,
            DaemonType.MODEL_DISTRIBUTION,
            DaemonType.DAEMON_WATCHDOG,
            DaemonType.QUALITY_MONITOR,
        ]

        logger.info("=" * 60)
        logger.info("DRY RUN - Coordinator would start these daemons:")
        logger.info("=" * 60)
        for d in coordinator_daemons:
            if d not in intensive:
                logger.info(f"  [+] {d.name}")
            else:
                logger.info(f"  [-] {d.name} (skipped - intensive)")
        logger.info("=" * 60)
        return

    # Run master_loop with coordinator settings
    try:
        # Import after env setup
        from scripts.master_loop import MasterLoop

        loop = MasterLoop(
            daemon_profile=args.daemon_profile,
            watch_only=args.watch,
        )

        # Handle signals for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            loop.stop()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # Run
        logger.info("Starting coordinator daemons via MasterLoop...")
        asyncio.run(loop.run())

    except KeyboardInterrupt:
        logger.info("Coordinator shutdown requested")
    except Exception as e:
        logger.error(f"Coordinator failed: {e}")
        raise


if __name__ == "__main__":
    main()
