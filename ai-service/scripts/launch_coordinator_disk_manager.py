#!/usr/bin/env python3
"""Launch Coordinator Disk Manager Daemon.

December 27, 2025: Simple launcher for coordinator-only nodes.

This script starts the CoordinatorDiskManager daemon which:
- Monitors disk usage and triggers cleanup at 50% (vs 60% for regular nodes)
- Syncs data to remote storage (OWC on mac-studio) before cleanup
- Removes synced training/game files after 24 hours
- Keeps canonical databases locally for quick access

Usage:
    # Run in foreground (for testing)
    python scripts/launch_coordinator_disk_manager.py

    # Run as background service
    python scripts/launch_coordinator_disk_manager.py &

    # With custom remote host
    RINGRIFT_COORDINATOR_REMOTE_HOST=mac-studio python scripts/launch_coordinator_disk_manager.py

    # With custom remote path
    RINGRIFT_COORDINATOR_REMOTE_PATH=/Volumes/RingRift-Data python scripts/launch_coordinator_disk_manager.py

Environment Variables:
    RINGRIFT_COORDINATOR_REMOTE_HOST - Remote host for sync (default: mac-studio)
    RINGRIFT_COORDINATOR_REMOTE_PATH - Remote path for sync (default: /Volumes/RingRift-Data)
    RINGRIFT_DISK_SPACE_CHECK_INTERVAL - Check interval in seconds (default: 1800 = 30 min)
    RINGRIFT_DISK_SPACE_PROACTIVE_THRESHOLD - Cleanup threshold % (default: 50)
    RINGRIFT_DISK_SPACE_TARGET_USAGE - Target usage % after cleanup (default: 40)
"""

import asyncio
import logging
import os
import signal
import sys

# Add ai-service to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Global daemon reference for signal handling
_daemon = None


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _daemon
    logger.info(f"Received signal {signum}, initiating shutdown...")
    if _daemon is not None:
        asyncio.get_event_loop().create_task(_daemon.stop())
    else:
        sys.exit(0)


async def main():
    """Main entry point."""
    global _daemon

    # Import after path setup
    from app.coordination.disk_space_manager_daemon import (
        CoordinatorDiskManager,
        CoordinatorDiskConfig,
    )

    # Create configuration
    config = CoordinatorDiskConfig.for_coordinator()

    # Log configuration
    logger.info("=== Coordinator Disk Manager ===")
    logger.info(f"Remote host: {config.remote_host}")
    logger.info(f"Remote path: {config.remote_base_path}")
    logger.info(f"Proactive threshold: {config.proactive_cleanup_threshold}%")
    logger.info(f"Target usage: {config.target_disk_usage}%")
    logger.info(f"Check interval: {config.check_interval_seconds}s")
    logger.info(f"Sync before cleanup: {config.sync_before_cleanup}")
    logger.info("================================")

    # Create and start daemon
    _daemon = CoordinatorDiskManager(config)

    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        logger.info("Starting CoordinatorDiskManager daemon...")
        await _daemon.start()

        # Get initial status
        status = _daemon.get_disk_status()
        if status:
            logger.info(
                f"Initial disk status: {status.usage_percent:.1f}% used, "
                f"{status.free_gb:.1f}GB free"
            )

        # Wait for daemon to complete (runs forever)
        while _daemon._running:
            await asyncio.sleep(60)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    except Exception as e:
        logger.error(f"Error running daemon: {e}", exc_info=True)
        raise
    finally:
        if _daemon._running:
            await _daemon.stop()
        logger.info("CoordinatorDiskManager shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown complete")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
