#!/usr/bin/env python3
"""Vast.ai Termination Guard - Emergency data sync before instance termination.

This script runs on Vast.ai instances to detect impending termination and
trigger emergency data sync before data in /dev/shm is lost.

Detection methods:
1. Poll Vast.ai API for instance status changes
2. Watch for SIGTERM/SIGINT signals
3. Monitor system load and memory pressure
4. Heartbeat with central collector

Usage:
    # Run as a daemon on Vast.ai instances
    python scripts/vastai_termination_guard.py --instance-id 12345

    # With custom sync target
    python scripts/vastai_termination_guard.py --instance-id 12345 \\
        --sync-target user@central-host:~/ringrift/ai-service/data/games/

    # Test mode (simulate termination)
    python scripts/vastai_termination_guard.py --test
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

# Ensure ai-service root on path for scripts/lib imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("vastai_termination_guard")

@dataclass
class GuardConfig:
    """Configuration for the termination guard."""
    instance_id: str = ""
    data_dir: str = "/dev/shm/games"  # RAM storage on Vast.ai
    sync_target: str = ""  # user@host:path or local path
    heartbeat_interval: int = 30  # Seconds between heartbeats
    sync_interval: int = 300  # Regular sync interval (5 min)
    emergency_sync_timeout: int = 120  # Max seconds for emergency sync
    vastai_api_key: str = ""  # Optional: for API status polling
    central_collector_url: str = ""  # Optional: HTTP endpoint for heartbeat


@dataclass
class SyncStatus:
    """Status of data sync operations."""
    last_sync_time: float = 0.0
    last_sync_games: int = 0
    total_synced: int = 0
    pending_games: int = 0
    last_error: str = ""
    emergency_sync_triggered: bool = False


class VastaiTerminationGuard:
    """Guards against data loss on Vast.ai instance termination."""

    def __init__(self, config: GuardConfig):
        self.config = config
        self.status = SyncStatus()
        self._running = False
        self._shutdown_requested = False
        self._emergency_sync_in_progress = False

        # Track games that have been synced
        self._synced_games: set = set()

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        # Try SIGUSR1 for manual emergency sync trigger
        try:
            signal.signal(signal.SIGUSR1, self._handle_emergency_signal)
        except (AttributeError, ValueError):
            pass  # Not available on all platforms

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Handle termination signals - trigger emergency sync."""
        sig_name = signal.Signals(signum).name
        logger.warning(f"Received {sig_name} - triggering emergency sync")
        self._shutdown_requested = True

        # Run emergency sync synchronously (blocking)
        if not self._emergency_sync_in_progress:
            self._emergency_sync_in_progress = True
            try:
                # Use subprocess to avoid async issues in signal handler
                self._sync_emergency_blocking()
            finally:
                self._emergency_sync_in_progress = False

    def _handle_emergency_signal(self, signum: int, frame: Any) -> None:
        """Handle SIGUSR1 - manual emergency sync trigger."""
        logger.info("Received SIGUSR1 - triggering manual emergency sync")
        if not self._emergency_sync_in_progress:
            self._emergency_sync_in_progress = True
            try:
                self._sync_emergency_blocking()
            finally:
                self._emergency_sync_in_progress = False

    def _get_local_game_dbs(self) -> List[Path]:
        """Get list of local game databases."""
        data_dir = Path(self.config.data_dir)
        if not data_dir.exists():
            return []
        return list(data_dir.glob("*.db"))

    def _count_local_games(self) -> int:
        """Count total games in local databases."""
        total = 0
        for db_path in self._get_local_game_dbs():
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM games")
                total += cursor.fetchone()[0]
                conn.close()
            except Exception as e:
                logger.debug(f"Error counting games in {db_path}: {e}")
        return total

    def _get_pending_games(self) -> int:
        """Get count of games not yet synced."""
        total = self._count_local_games()
        return max(0, total - len(self._synced_games))

    def _sync_emergency_blocking(self) -> bool:
        """Perform emergency sync (blocking, for signal handlers).

        Returns True on success.
        """
        logger.info("Starting emergency data sync...")
        self.status.emergency_sync_triggered = True

        data_dir = Path(self.config.data_dir)
        if not data_dir.exists():
            logger.warning(f"Data directory does not exist: {data_dir}")
            return False

        db_files = self._get_local_game_dbs()
        if not db_files:
            logger.info("No database files to sync")
            return True

        if not self.config.sync_target:
            logger.error("No sync target configured!")
            return False

        synced = 0
        errors = 0

        for db_path in db_files:
            try:
                # Use rsync for robust transfer
                cmd = [
                    "rsync", "-avz", "--progress", "--timeout=60",
                    str(db_path),
                    self.config.sync_target,
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.config.emergency_sync_timeout,
                )

                if result.returncode == 0:
                    logger.info(f"Synced {db_path.name}")
                    synced += 1
                else:
                    logger.error(f"Failed to sync {db_path.name}: {result.stderr}")
                    errors += 1

            except subprocess.TimeoutExpired:
                logger.error(f"Timeout syncing {db_path.name}")
                errors += 1
            except Exception as e:
                logger.error(f"Error syncing {db_path.name}: {e}")
                errors += 1

        # Log summary
        total = len(db_files)
        if errors == 0:
            logger.info(f"Emergency sync complete: {synced}/{total} files synced")
        else:
            logger.warning(f"Emergency sync partial: {synced}/{total} synced, {errors} errors")

        return errors == 0

    async def _sync_incremental(self) -> int:
        """Perform incremental sync of new games.

        Returns number of games synced.
        """
        if not self.config.sync_target:
            return 0

        data_dir = Path(self.config.data_dir)
        if not data_dir.exists():
            return 0

        db_files = self._get_local_game_dbs()
        if not db_files:
            return 0

        total_synced = 0

        for db_path in db_files:
            try:
                # Use rsync with checksum for incremental sync
                cmd = f'rsync -avz --checksum --timeout=60 {db_path} {self.config.sync_target}'

                process = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=120,
                )

                if process.returncode == 0:
                    # Count games in this DB
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT game_id FROM games")
                    game_ids = [row[0] for row in cursor.fetchall()]
                    conn.close()

                    new_games = [gid for gid in game_ids if gid not in self._synced_games]
                    self._synced_games.update(new_games)
                    total_synced += len(new_games)

            except asyncio.TimeoutError:
                logger.warning(f"Timeout syncing {db_path.name}")
            except Exception as e:
                logger.warning(f"Error syncing {db_path.name}: {e}")

        if total_synced > 0:
            self.status.last_sync_time = time.time()
            self.status.last_sync_games = total_synced
            self.status.total_synced += total_synced
            logger.info(f"Incremental sync: {total_synced} new games")

        return total_synced

    async def _send_heartbeat(self) -> bool:
        """Send heartbeat to central collector.

        Returns True on success.
        """
        if not self.config.central_collector_url:
            return True  # No collector configured, skip

        try:
            import aiohttp

            payload = {
                "instance_id": self.config.instance_id,
                "timestamp": time.time(),
                "pending_games": self._get_pending_games(),
                "total_synced": self.status.total_synced,
                "data_dir": self.config.data_dir,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.central_collector_url}/heartbeat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    return resp.status == 200

        except ImportError:
            # aiohttp not available, skip heartbeat
            return True
        except Exception as e:
            logger.debug(f"Heartbeat failed: {e}")
            return False

    async def _check_vastai_status(self) -> Optional[str]:
        """Check Vast.ai instance status via API.

        Returns status string or None if unavailable.
        """
        if not self.config.vastai_api_key or not self.config.instance_id:
            return None

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.config.vastai_api_key}"}
                async with session.get(
                    f"https://console.vast.ai/api/v0/instances/{self.config.instance_id}/",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("actual_status", "unknown")

        except Exception as e:
            logger.debug(f"Vast.ai API check failed: {e}")

        return None

    async def run(self) -> None:
        """Main guard loop."""
        self._running = True
        logger.info(f"Starting termination guard for instance {self.config.instance_id}")
        logger.info(f"Data directory: {self.config.data_dir}")
        logger.info(f"Sync target: {self.config.sync_target}")

        last_sync = time.time()
        last_heartbeat = time.time()

        while self._running and not self._shutdown_requested:
            try:
                now = time.time()

                # Update pending games count
                self.status.pending_games = self._get_pending_games()

                # Send heartbeat
                if now - last_heartbeat >= self.config.heartbeat_interval:
                    await self._send_heartbeat()
                    last_heartbeat = now

                # Check Vast.ai status (if API key configured)
                status = await self._check_vastai_status()
                if status and status not in ("running", "loading"):
                    logger.warning(f"Instance status changed to '{status}' - triggering emergency sync")
                    self._sync_emergency_blocking()
                    break

                # Regular incremental sync
                if now - last_sync >= self.config.sync_interval:
                    await self._sync_incremental()
                    last_sync = now

                # Sleep
                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Guard loop error: {e}")
                await asyncio.sleep(30)

        # Final sync on shutdown
        if self._shutdown_requested:
            logger.info("Shutdown requested - performing final sync")
            self._sync_emergency_blocking()

        logger.info("Termination guard stopped")

    def stop(self) -> None:
        """Request graceful shutdown."""
        self._running = False
        self._shutdown_requested = True


def create_systemd_service(instance_id: str, sync_target: str) -> str:
    """Generate a systemd service file for the termination guard.

    This can be installed on Vast.ai instances for automatic startup.
    """
    return f"""[Unit]
Description=Vast.ai Termination Guard
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/ringrift/ai-service
ExecStart=/usr/bin/python3 scripts/vastai_termination_guard.py \\
    --instance-id {instance_id} \\
    --sync-target {sync_target}
Restart=always
RestartSec=10
KillSignal=SIGTERM
TimeoutStopSec=180

[Install]
WantedBy=multi-user.target
"""


def main():
    parser = argparse.ArgumentParser(description="Vast.ai Termination Guard")
    parser.add_argument("--instance-id", type=str, default="", help="Vast.ai instance ID")
    parser.add_argument("--data-dir", type=str, default="/dev/shm/games", help="Local data directory")
    parser.add_argument("--sync-target", type=str, default="", help="Sync target (user@host:path)")
    parser.add_argument("--sync-interval", type=int, default=300, help="Regular sync interval (seconds)")
    parser.add_argument("--heartbeat-interval", type=int, default=30, help="Heartbeat interval (seconds)")
    parser.add_argument("--vastai-api-key", type=str, default="", help="Vast.ai API key (optional)")
    parser.add_argument("--collector-url", type=str, default="", help="Central collector URL (optional)")
    parser.add_argument("--generate-systemd", action="store_true", help="Generate systemd service file")
    parser.add_argument("--test", action="store_true", help="Test mode (simulate termination)")

    args = parser.parse_args()

    # Get instance ID from environment if not specified
    instance_id = args.instance_id or os.environ.get("VAST_CONTAINERLABEL", "unknown")

    # Get sync target from environment if not specified
    sync_target = args.sync_target or os.environ.get("RINGRIFT_SYNC_TARGET", "")

    if args.generate_systemd:
        print(create_systemd_service(instance_id, sync_target))
        return

    if not sync_target:
        logger.error("No sync target specified. Use --sync-target or set RINGRIFT_SYNC_TARGET")
        sys.exit(1)

    config = GuardConfig(
        instance_id=instance_id,
        data_dir=args.data_dir,
        sync_target=sync_target,
        sync_interval=args.sync_interval,
        heartbeat_interval=args.heartbeat_interval,
        vastai_api_key=args.vastai_api_key or os.environ.get("VASTAI_API_KEY", ""),
        central_collector_url=args.collector_url,
    )

    guard = VastaiTerminationGuard(config)

    if args.test:
        logger.info("Test mode - simulating termination signal")
        # Trigger emergency sync
        guard._sync_emergency_blocking()
        return

    # Run the guard
    try:
        asyncio.run(guard.run())
    except KeyboardInterrupt:
        logger.info("Interrupted - triggering final sync")
        guard._sync_emergency_blocking()


if __name__ == "__main__":
    main()
