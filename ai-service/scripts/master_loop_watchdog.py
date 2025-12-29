#!/usr/bin/env python3
"""Master Loop Watchdog - Monitors and restarts master_loop.py if hung or crashed.

This daemon monitors the master loop health by checking:
1. Heartbeat staleness in SQLite database
2. PID file validity (process actually running)
3. Lock file status

If the master loop is detected as unhealthy, the watchdog will:
1. Kill the stale process (if any)
2. Clean up lock files
3. Restart master_loop.py
4. Apply exponential backoff on repeated failures

Usage:
    # Start watchdog (runs as daemon)
    python scripts/master_loop_watchdog.py

    # Run once (check and exit)
    python scripts/master_loop_watchdog.py --once

    # Test mode (don't restart, just report)
    python scripts/master_loop_watchdog.py --test

    # Custom heartbeat threshold
    python scripts/master_loop_watchdog.py --heartbeat-threshold 120

December 2025: Created as part of 48-hour autonomous operation plan.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.lib.process import (
    SingletonLock,
    is_process_running,
    kill_process,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration (Environment Variable Overrides)
# =============================================================================

# How often to check master loop health (seconds)
CHECK_INTERVAL = float(os.environ.get("RINGRIFT_WATCHDOG_INTERVAL", "30"))

# Heartbeat considered stale after this many seconds
HEARTBEAT_STALE_THRESHOLD = float(
    os.environ.get("RINGRIFT_HEARTBEAT_STALE_THRESHOLD", "90")
)

# Maximum restarts per hour before backing off
MAX_RESTARTS_PER_HOUR = int(os.environ.get("RINGRIFT_MAX_RESTARTS_PER_HOUR", "3"))

# Backoff multiplier for repeated failures
BACKOFF_BASE = 60  # 1 minute
BACKOFF_MAX = 1800  # 30 minutes

# Paths
STATE_DB_PATH = Path(__file__).parent.parent / "data" / "coordination" / "master_loop_state.db"
PID_FILE_PATH = Path(__file__).parent.parent / "data" / "coordination" / "master_loop.pid"
LOCK_DIR = Path(__file__).parent.parent / "data" / "coordination"
WATCHDOG_STATE_PATH = LOCK_DIR / "watchdog_state.json"

# Master loop script path
MASTER_LOOP_SCRIPT = Path(__file__).parent / "master_loop.py"


@dataclass
class WatchdogState:
    """Tracks watchdog state for restart limiting."""
    restarts_this_hour: int = 0
    last_restart_time: float = 0.0
    hour_start: float = field(default_factory=time.time)
    consecutive_failures: int = 0
    current_backoff: float = 0.0
    last_healthy_time: float = field(default_factory=time.time)

    def reset_hourly_if_needed(self) -> None:
        """Reset hourly counter if an hour has passed."""
        if time.time() - self.hour_start >= 3600:
            self.hour_start = time.time()
            self.restarts_this_hour = 0

    def record_restart(self) -> None:
        """Record a restart attempt."""
        self.reset_hourly_if_needed()
        self.restarts_this_hour += 1
        self.last_restart_time = time.time()
        self.consecutive_failures += 1
        # Calculate backoff: 60s, 120s, 240s, ... up to 30 minutes
        self.current_backoff = min(
            BACKOFF_BASE * (2 ** (self.consecutive_failures - 1)),
            BACKOFF_MAX,
        )

    def record_healthy(self) -> None:
        """Record that master loop is healthy."""
        self.consecutive_failures = 0
        self.current_backoff = 0.0
        self.last_healthy_time = time.time()

    def can_restart(self) -> tuple[bool, str]:
        """Check if we can attempt a restart.

        Returns:
            Tuple of (can_restart, reason_if_not)
        """
        self.reset_hourly_if_needed()

        # Check hourly limit
        if self.restarts_this_hour >= MAX_RESTARTS_PER_HOUR:
            remaining = 3600 - (time.time() - self.hour_start)
            return False, f"Hourly limit reached ({MAX_RESTARTS_PER_HOUR}). Retry in {int(remaining)}s"

        # Check backoff
        if self.current_backoff > 0:
            time_since_restart = time.time() - self.last_restart_time
            if time_since_restart < self.current_backoff:
                remaining = self.current_backoff - time_since_restart
                return False, f"Backoff active. Retry in {int(remaining)}s"

        return True, ""


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    healthy: bool
    pid: int | None = None
    heartbeat_age: float | None = None
    loop_iteration: int | None = None
    status: str = "unknown"
    reason: str = ""


class MasterLoopWatchdog:
    """Monitors master loop health and restarts if needed."""

    def __init__(
        self,
        test_mode: bool = False,
        heartbeat_threshold: float = HEARTBEAT_STALE_THRESHOLD,
        check_interval: float = CHECK_INTERVAL,
    ):
        self.test_mode = test_mode
        self.heartbeat_threshold = heartbeat_threshold
        self.check_interval = check_interval
        self.state = WatchdogState()
        self._running = False
        self._lock: SingletonLock | None = None

    def _get_heartbeat_info(self) -> tuple[float | None, int | None, str | None]:
        """Get heartbeat info from SQLite database.

        Returns:
            Tuple of (last_beat, loop_iteration, status) or (None, None, None)
        """
        if not STATE_DB_PATH.exists():
            return None, None, None

        try:
            conn = sqlite3.connect(STATE_DB_PATH, timeout=5.0)
            row = conn.execute(
                "SELECT last_beat, loop_iteration, status FROM heartbeat WHERE id = 1"
            ).fetchone()
            conn.close()

            if row:
                return row[0], row[1], row[2]
        except (sqlite3.Error, OSError) as e:
            logger.warning(f"[Watchdog] Failed to read heartbeat: {e}")

        return None, None, None

    def _get_pid_from_file(self) -> int | None:
        """Get PID from PID file."""
        if not PID_FILE_PATH.exists():
            return None

        try:
            content = PID_FILE_PATH.read_text().strip()
            return int(content) if content else None
        except (ValueError, OSError):
            return None

    def _get_pid_from_lock(self) -> int | None:
        """Get PID from lock file."""
        lock = SingletonLock("master_loop", lock_dir=LOCK_DIR)
        return lock.get_holder_pid()

    def check_health(self) -> HealthCheckResult:
        """Check master loop health.

        Returns:
            HealthCheckResult with health status and details
        """
        # 1. Check if master loop PID is running
        pid = self._get_pid_from_lock() or self._get_pid_from_file()
        pid_alive = pid is not None and is_process_running(pid)

        # 2. Check heartbeat freshness
        last_beat, loop_iteration, status = self._get_heartbeat_info()

        if last_beat is not None:
            heartbeat_age = time.time() - last_beat
        else:
            heartbeat_age = None

        # 3. Determine health status
        if not pid_alive:
            if pid is not None:
                return HealthCheckResult(
                    healthy=False,
                    pid=pid,
                    heartbeat_age=heartbeat_age,
                    loop_iteration=loop_iteration,
                    status="dead",
                    reason=f"Process {pid} not running",
                )
            else:
                return HealthCheckResult(
                    healthy=False,
                    pid=None,
                    heartbeat_age=heartbeat_age,
                    loop_iteration=loop_iteration,
                    status="not_started",
                    reason="No PID found - master loop not running",
                )

        # PID is alive - check heartbeat
        if heartbeat_age is None:
            return HealthCheckResult(
                healthy=False,
                pid=pid,
                heartbeat_age=None,
                loop_iteration=loop_iteration,
                status="no_heartbeat",
                reason="No heartbeat data - possible startup or DB issue",
            )

        if heartbeat_age > self.heartbeat_threshold:
            return HealthCheckResult(
                healthy=False,
                pid=pid,
                heartbeat_age=heartbeat_age,
                loop_iteration=loop_iteration,
                status="stale",
                reason=f"Heartbeat stale ({heartbeat_age:.1f}s > {self.heartbeat_threshold}s threshold)",
            )

        # All checks passed
        return HealthCheckResult(
            healthy=True,
            pid=pid,
            heartbeat_age=heartbeat_age,
            loop_iteration=loop_iteration,
            status=status or "running",
            reason="Healthy",
        )

    def _kill_stale_process(self, pid: int) -> bool:
        """Kill a stale master loop process.

        Args:
            pid: Process ID to kill

        Returns:
            True if process was killed or already dead
        """
        logger.warning(f"[Watchdog] Killing stale master loop process (PID {pid})")

        # Send SIGTERM first
        success = kill_process(pid, signal.SIGTERM, wait=True, timeout=30.0)

        if not success:
            logger.error(f"[Watchdog] Failed to kill PID {pid}")
            return False

        # Clean up lock file
        lock = SingletonLock("master_loop", lock_dir=LOCK_DIR)
        lock.force_release(kill_holder=False)

        # Clean up PID file
        try:
            if PID_FILE_PATH.exists():
                PID_FILE_PATH.unlink()
        except OSError:
            pass

        return True

    def _start_master_loop(self) -> bool:
        """Start the master loop process.

        Returns:
            True if process started successfully
        """
        if not MASTER_LOOP_SCRIPT.exists():
            logger.error(f"[Watchdog] Master loop script not found: {MASTER_LOOP_SCRIPT}")
            return False

        logger.info("[Watchdog] Starting master loop...")

        try:
            # Start as background process with log redirection
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(__file__).parent.parent)

            # Ensure logs directory exists
            logs_dir = MASTER_LOOP_SCRIPT.parent.parent / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)

            log_file = logs_dir / "master_loop.log"
            error_log = logs_dir / "master_loop.error.log"

            # Open log files for appending
            stdout_file = open(log_file, "a")
            stderr_file = open(error_log, "a")

            process = subprocess.Popen(
                [sys.executable, str(MASTER_LOOP_SCRIPT)],
                cwd=str(MASTER_LOOP_SCRIPT.parent.parent),
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,  # Detach from watchdog
            )

            # Wait a moment to check if it started
            time.sleep(2.0)

            if process.poll() is not None:
                logger.error(f"[Watchdog] Master loop exited immediately (code {process.returncode})")
                return False

            logger.info(f"[Watchdog] Master loop started (PID {process.pid})")
            return True

        except Exception as e:
            logger.error(f"[Watchdog] Failed to start master loop: {e}")
            return False

    def _restart_master_loop(self, health: HealthCheckResult) -> bool:
        """Restart the master loop.

        Args:
            health: Health check result with status details

        Returns:
            True if restart was successful
        """
        # Check restart limits
        can_restart, reason = self.state.can_restart()
        if not can_restart:
            logger.warning(f"[Watchdog] Cannot restart: {reason}")
            return False

        if self.test_mode:
            logger.info("[Watchdog] TEST MODE: Would restart master loop")
            return True

        # Kill stale process if running
        if health.pid is not None and is_process_running(health.pid):
            if not self._kill_stale_process(health.pid):
                return False

            # Brief pause after kill
            time.sleep(1.0)

        # Start new instance
        if not self._start_master_loop():
            self.state.record_restart()  # Record failure for backoff
            return False

        # Record successful restart
        self.state.record_restart()
        logger.info(
            f"[Watchdog] Restart successful (attempt {self.state.restarts_this_hour}/{MAX_RESTARTS_PER_HOUR} this hour)"
        )

        return True

    def run_once(self) -> bool:
        """Run a single health check and restart if needed.

        Returns:
            True if master loop is healthy after check
        """
        health = self.check_health()

        if health.healthy:
            self.state.record_healthy()
            logger.debug(
                f"[Watchdog] Healthy: PID={health.pid}, "
                f"heartbeat_age={health.heartbeat_age:.1f}s, "
                f"iteration={health.loop_iteration}"
            )
            return True

        # Unhealthy - log and potentially restart
        logger.warning(f"[Watchdog] Unhealthy: {health.reason}")

        # Attempt restart
        if self._restart_master_loop(health):
            # Wait for startup and recheck
            time.sleep(10.0)
            post_health = self.check_health()
            if post_health.healthy:
                logger.info("[Watchdog] Master loop recovered after restart")
                return True
            else:
                logger.error(f"[Watchdog] Master loop still unhealthy after restart: {post_health.reason}")
                return False

        return False

    def run_forever(self) -> None:
        """Run watchdog loop indefinitely."""
        # Acquire singleton lock
        self._lock = SingletonLock("master_loop_watchdog", lock_dir=LOCK_DIR)
        if not self._lock.acquire():
            holder_pid = self._lock.get_holder_pid()
            logger.error(f"[Watchdog] Another watchdog is already running (PID {holder_pid})")
            sys.exit(1)

        logger.info(
            f"[Watchdog] Starting (interval={self.check_interval}s, "
            f"threshold={self.heartbeat_threshold}s, "
            f"max_restarts={MAX_RESTARTS_PER_HOUR}/hr)"
        )

        self._running = True

        # Handle shutdown signals
        def on_shutdown(signum: int, frame: Any) -> None:
            logger.info(f"[Watchdog] Received signal {signum}, shutting down")
            self._running = False

        signal.signal(signal.SIGTERM, on_shutdown)
        signal.signal(signal.SIGINT, on_shutdown)

        try:
            while self._running:
                try:
                    self.run_once()
                except Exception as e:
                    logger.error(f"[Watchdog] Error in health check: {e}")

                # Sleep with interrupt support
                for _ in range(int(self.check_interval)):
                    if not self._running:
                        break
                    time.sleep(1.0)

        finally:
            if self._lock:
                self._lock.release()
            logger.info("[Watchdog] Shutdown complete")

    def get_status(self) -> dict:
        """Get watchdog status for monitoring."""
        health = self.check_health()
        return {
            "watchdog_running": self._running,
            "master_loop_healthy": health.healthy,
            "master_loop_pid": health.pid,
            "master_loop_status": health.status,
            "heartbeat_age_seconds": health.heartbeat_age,
            "loop_iteration": health.loop_iteration,
            "restarts_this_hour": self.state.restarts_this_hour,
            "consecutive_failures": self.state.consecutive_failures,
            "current_backoff_seconds": self.state.current_backoff,
            "last_restart_time": self.state.last_restart_time,
            "last_healthy_time": self.state.last_healthy_time,
            "heartbeat_threshold": self.heartbeat_threshold,
            "check_interval": self.check_interval,
        }


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Master Loop Watchdog - monitors and restarts master_loop.py"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run single check and exit",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode - don't actually restart",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show status and exit",
    )
    parser.add_argument(
        "--heartbeat-threshold",
        type=float,
        default=HEARTBEAT_STALE_THRESHOLD,
        help=f"Heartbeat staleness threshold in seconds (default: {HEARTBEAT_STALE_THRESHOLD})",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=CHECK_INTERVAL,
        help=f"Check interval in seconds (default: {CHECK_INTERVAL})",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    watchdog = MasterLoopWatchdog(
        test_mode=args.test,
        heartbeat_threshold=args.heartbeat_threshold,
        check_interval=args.interval,
    )

    if args.status:
        import json
        status = watchdog.get_status()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.once:
        healthy = watchdog.run_once()
        sys.exit(0 if healthy else 1)

    # Run forever
    watchdog.run_forever()


if __name__ == "__main__":
    main()
