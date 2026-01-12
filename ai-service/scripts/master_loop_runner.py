#!/usr/bin/env python3
"""Master Loop Runner - Cluster-integrated wrapper for master_loop.py.

This script provides convenient defaults and cluster integration for running
the master loop on production nodes.

Features:
- Auto-detects node role (coordinator vs worker)
- Registers with P2P orchestrator for visibility
- Sets up proper logging with rotation
- Handles graceful shutdown on termination signals
- Provides health check endpoint

Usage:
    # Run as coordinator (full automation)
    python scripts/master_loop_runner.py

    # Run as worker (selfplay-only mode)
    python scripts/master_loop_runner.py --worker

    # Quick status check
    python scripts/master_loop_runner.py --status

    # Dry run to see what would happen
    python scripts/master_loop_runner.py --dry-run

December 2025: Created as part of cluster integration improvements.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

# Setup PYTHONPATH
ai_service_root = Path(__file__).parent.parent
sys.path.insert(0, str(ai_service_root))

from app.config.ports import get_local_p2p_status_url, get_local_p2p_url

# Configure logging with rotation
LOG_DIR = ai_service_root / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_file = LOG_DIR / f"master_loop_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file),
    ],
)
logger = logging.getLogger(__name__)


def detect_node_role() -> str:
    """Detect if this node is a coordinator or worker.

    Returns 'coordinator' if this appears to be the main orchestration node,
    'worker' otherwise.
    """
    hostname = socket.gethostname().lower()

    # Coordinator indicators
    coordinator_patterns = ["coordinator", "master", "leader", "control"]
    for pattern in coordinator_patterns:
        if pattern in hostname:
            return "coordinator"

    # Check for P2P leader status
    try:
        import urllib.request
        import json

        with urllib.request.urlopen(get_local_p2p_status_url(), timeout=2) as resp:
            status = json.loads(resp.read().decode())
            if status.get("is_leader"):
                return "coordinator"
    except (OSError, TimeoutError, json.JSONDecodeError, ValueError):
        pass

    # Default to worker
    return "worker"


def get_default_configs(role: str) -> list[str] | None:
    """Get default configs based on node role.

    Coordinators manage all configs, workers focus on smaller boards.
    """
    if role == "coordinator":
        return None  # All configs

    # Workers focus on smaller boards (faster selfplay)
    return [
        "hex8_2p", "hex8_3p", "hex8_4p",
        "square8_2p", "square8_3p", "square8_4p",
    ]


async def register_with_p2p(role: str) -> bool:
    """Register this master loop instance with the P2P orchestrator."""
    try:
        import urllib.request
        import json

        hostname = socket.gethostname()
        data = json.dumps({
            "service": "master_loop",
            "role": role,
            "hostname": hostname,
            "started_at": time.time(),
        }).encode()

        req = urllib.request.Request(
            f"{get_local_p2p_url()}/register_service",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                logger.info(f"[Runner] Registered with P2P as {role}")
                return True

    except Exception as e:
        logger.debug(f"[Runner] Could not register with P2P: {e}")

    return False


async def run_master_loop(args: argparse.Namespace) -> None:
    """Run the master loop with cluster integration."""
    from scripts.master_loop import MasterLoopController, watch_mode
    import signal

    # Detect role if not specified
    role = args.role or detect_node_role()
    logger.info(f"[Runner] Node role: {role}")

    # Get configs
    configs = None
    if args.configs:
        configs = [c.strip() for c in args.configs.split(",")]
    elif role == "worker":
        configs = get_default_configs(role)

    logger.info(f"[Runner] Configs: {configs or 'all'}")

    # Create controller
    controller = MasterLoopController(
        configs=configs,
        dry_run=args.dry_run,
        skip_daemons=args.skip_daemons or role == "worker",
    )

    # Handle signals
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(controller.stop()))

    # Register with P2P
    if not args.dry_run:
        await register_with_p2p(role)

    # Run
    if args.status:
        await controller._initialize_state()
        status = controller.get_status()
        import json
        print(json.dumps(status, indent=2, default=str))
        return

    if args.watch:
        await controller.start()
        await watch_mode(controller, interval=args.interval)
    else:
        logger.info("[Runner] Starting master loop...")
        await controller.run()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Master Loop Runner - Cluster-integrated automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--role",
        choices=["coordinator", "worker"],
        help="Node role (auto-detected if not specified)",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Run as worker (selfplay-only, smaller boards)",
    )
    parser.add_argument(
        "--configs",
        type=str,
        help="Comma-separated list of configs to manage",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without executing",
    )
    parser.add_argument(
        "--skip-daemons",
        action="store_true",
        help="Don't start/stop daemons",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch mode - display live status",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Watch mode update interval in seconds",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current status and exit",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # --worker is shorthand for --role worker
    if args.worker:
        args.role = "worker"

    return args


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info("=" * 60)
    logger.info("RingRift Master Loop Runner")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 60)

    try:
        asyncio.run(run_master_loop(args))
    except KeyboardInterrupt:
        logger.info("[Runner] Interrupted by user")
    except Exception as e:
        logger.error(f"[Runner] Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
