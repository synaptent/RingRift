"""Training Watchdog Daemon - Monitors stuck training processes.

January 4, 2026 (Sprint 17): Created to prevent training lock deadlocks.
Monitors training process heartbeats and kills stale processes.

Problem Solved:
- Training processes that crash while holding locks leave locks orphaned
- No visibility into stuck training processes
- Manual intervention required to clear stuck locks

Features:
1. Tracks training process heartbeats via TRAINING_HEARTBEAT events
2. Monitors for stale processes (no heartbeat for configurable timeout)
3. Kills stale processes (SIGTERM, then SIGKILL after grace period)
4. Releases associated training locks
5. Emits TRAINING_PROCESS_KILLED event for monitoring

Usage:
    from app.coordination.training_watchdog_daemon import (
        TrainingWatchdogDaemon,
        get_training_watchdog_daemon,
    )

    # Start the daemon
    daemon = get_training_watchdog_daemon()
    await daemon.start()

    # Register a training process (called by train.py)
    daemon.register_training_process(
        config_key="hex8_2p",
        pid=12345,
        node_id="worker-1",
    )

    # Send heartbeat (called periodically during training)
    daemon.heartbeat(config_key="hex8_2p", pid=12345)
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.coordination.event_emission_helpers import safe_emit_event
from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.protocols import CoordinatorStatus

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Maximum time without heartbeat before killing (2 hours)
DEFAULT_STALE_THRESHOLD_SECONDS = 2 * 60 * 60  # 7200 seconds

# Grace period after SIGTERM before SIGKILL
DEFAULT_KILL_GRACE_PERIOD_SECONDS = 30

# How often to check for stale processes
DEFAULT_CHECK_INTERVAL_SECONDS = 60

# Database path for heartbeat tracking
DEFAULT_DB_PATH = Path("data/coordination/training_heartbeats.db")


@dataclass
class TrainingWatchdogConfig:
    """Configuration for training watchdog daemon."""

    # Interval between stale process checks
    check_interval_seconds: int = DEFAULT_CHECK_INTERVAL_SECONDS

    # Maximum time without heartbeat before process is considered stale
    stale_threshold_seconds: int = DEFAULT_STALE_THRESHOLD_SECONDS

    # Grace period after SIGTERM before SIGKILL
    kill_grace_period_seconds: int = DEFAULT_KILL_GRACE_PERIOD_SECONDS

    # Database path for heartbeat tracking
    db_path: Path = field(default_factory=lambda: DEFAULT_DB_PATH)

    # Whether to actually kill processes (False for testing)
    enable_process_kill: bool = True

    # Whether to release locks after killing
    release_locks_on_kill: bool = True

    @classmethod
    def from_env(cls, prefix: str = "RINGRIFT_TRAINING_WATCHDOG") -> "TrainingWatchdogConfig":
        """Load configuration from environment variables."""
        config = cls()

        if os.environ.get(f"{prefix}_INTERVAL"):
            try:
                config.check_interval_seconds = int(os.environ[f"{prefix}_INTERVAL"])
            except ValueError:
                pass

        if os.environ.get(f"{prefix}_STALE_THRESHOLD"):
            try:
                config.stale_threshold_seconds = int(os.environ[f"{prefix}_STALE_THRESHOLD"])
            except ValueError:
                pass

        if os.environ.get(f"{prefix}_KILL_GRACE"):
            try:
                config.kill_grace_period_seconds = int(os.environ[f"{prefix}_KILL_GRACE"])
            except ValueError:
                pass

        if os.environ.get(f"{prefix}_DB_PATH"):
            config.db_path = Path(os.environ[f"{prefix}_DB_PATH"])

        if os.environ.get(f"{prefix}_ENABLE_KILL"):
            config.enable_process_kill = os.environ[f"{prefix}_ENABLE_KILL"].lower() == "true"

        return config


# =============================================================================
# Training Process Info
# =============================================================================


@dataclass
class TrainingProcessInfo:
    """Information about a tracked training process."""

    config_key: str
    pid: int
    node_id: str
    started_at: float
    last_heartbeat: float
    status: str = "running"  # running, killed, completed


# =============================================================================
# Training Watchdog Daemon
# =============================================================================


class TrainingWatchdogDaemon(HandlerBase):
    """Monitors training processes and kills stale ones.

    January 4, 2026: Created as part of Sprint 17 stability improvements.

    Features:
    - Tracks training process heartbeats in SQLite
    - Subscribes to TRAINING_HEARTBEAT and TRAINING_LOCK_ACQUIRED events
    - Detects stale processes (no heartbeat for 2+ hours)
    - Kills stale processes with SIGTERM/SIGKILL
    - Releases associated training locks
    - Emits TRAINING_PROCESS_KILLED event
    """

    def __init__(self, config: TrainingWatchdogConfig | None = None):
        """Initialize the training watchdog daemon.

        Args:
            config: Optional configuration. Defaults to from_env().
        """
        self._daemon_config = config or TrainingWatchdogConfig.from_env()

        super().__init__(
            name="TrainingWatchdogDaemon",
            config=self._daemon_config,
            cycle_interval=float(self._daemon_config.check_interval_seconds),
        )

        # Node identification
        self._node_id = socket.gethostname()

        # Statistics
        self._processes_registered: int = 0
        self._processes_killed: int = 0
        self._heartbeats_received: int = 0
        self._locks_released: int = 0

        # Database connection (lazily initialized)
        self._db_path = self._daemon_config.db_path
        self._db_initialized = False

    @property
    def config(self) -> TrainingWatchdogConfig:
        """Get daemon configuration."""
        return self._daemon_config

    # =========================================================================
    # Database Management
    # =========================================================================

    def _ensure_db_initialized(self) -> None:
        """Ensure database schema is created."""
        if self._db_initialized:
            return

        # Create directory if needed
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS training_processes (
                        config_key TEXT NOT NULL,
                        pid INTEGER NOT NULL,
                        node_id TEXT NOT NULL,
                        started_at REAL NOT NULL,
                        last_heartbeat REAL NOT NULL,
                        status TEXT NOT NULL DEFAULT 'running',
                        PRIMARY KEY (config_key, node_id)
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_training_processes_status
                    ON training_processes (status)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_training_processes_heartbeat
                    ON training_processes (last_heartbeat)
                """)
                conn.commit()

            self._db_initialized = True
            logger.debug(f"[{self.name}] Database initialized at {self._db_path}")

        except sqlite3.Error as e:
            logger.error(f"[{self.name}] Database initialization failed: {e}")
            raise

    # =========================================================================
    # Process Registration and Heartbeat
    # =========================================================================

    def register_training_process(
        self,
        config_key: str,
        pid: int,
        node_id: str | None = None,
    ) -> None:
        """Register a training process for monitoring.

        Args:
            config_key: Training configuration (e.g., "hex8_2p")
            pid: Process ID of the training process
            node_id: Node where the process is running (defaults to this node)
        """
        self._ensure_db_initialized()
        node_id = node_id or self._node_id
        now = time.time()

        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO training_processes
                    (config_key, pid, node_id, started_at, last_heartbeat, status)
                    VALUES (?, ?, ?, ?, ?, 'running')
                    """,
                    (config_key, pid, node_id, now, now),
                )
                conn.commit()

            self._processes_registered += 1
            logger.info(
                f"[{self.name}] Registered training process: "
                f"config={config_key}, pid={pid}, node={node_id}"
            )

        except sqlite3.Error as e:
            logger.error(f"[{self.name}] Failed to register process: {e}")

    def heartbeat(self, config_key: str, pid: int, node_id: str | None = None) -> bool:
        """Record a heartbeat for a training process.

        Args:
            config_key: Training configuration
            pid: Process ID
            node_id: Node ID (defaults to this node)

        Returns:
            True if heartbeat recorded, False if process not found
        """
        self._ensure_db_initialized()
        node_id = node_id or self._node_id
        now = time.time()

        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                cursor = conn.execute(
                    """
                    UPDATE training_processes
                    SET last_heartbeat = ?, pid = ?
                    WHERE config_key = ? AND node_id = ? AND status = 'running'
                    """,
                    (now, pid, config_key, node_id),
                )
                conn.commit()

                if cursor.rowcount > 0:
                    self._heartbeats_received += 1
                    logger.debug(f"[{self.name}] Heartbeat: config={config_key}, pid={pid}")
                    return True

            # Process not found, auto-register
            logger.debug(
                f"[{self.name}] Process not found for heartbeat, auto-registering: "
                f"config={config_key}, pid={pid}"
            )
            self.register_training_process(config_key, pid, node_id)
            return True

        except sqlite3.Error as e:
            logger.error(f"[{self.name}] Heartbeat failed: {e}")
            return False

    def mark_completed(self, config_key: str, node_id: str | None = None) -> None:
        """Mark a training process as completed.

        Args:
            config_key: Training configuration
            node_id: Node ID (defaults to this node)
        """
        self._ensure_db_initialized()
        node_id = node_id or self._node_id

        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute(
                    """
                    UPDATE training_processes
                    SET status = 'completed'
                    WHERE config_key = ? AND node_id = ?
                    """,
                    (config_key, node_id),
                )
                conn.commit()

            logger.info(f"[{self.name}] Marked completed: config={config_key}, node={node_id}")

        except sqlite3.Error as e:
            logger.error(f"[{self.name}] Failed to mark completed: {e}")

    # =========================================================================
    # Event Subscriptions
    # =========================================================================

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """Subscribe to training-related events."""
        return {
            "training_heartbeat": self._on_training_heartbeat,
            "training_lock_acquired": self._on_training_lock_acquired,
            "training_completed": self._on_training_completed,
        }

    async def _on_training_heartbeat(self, event: dict[str, Any]) -> None:
        """Handle TRAINING_HEARTBEAT event."""
        config_key = event.get("config_key", "")
        pid = event.get("pid", 0)
        node_id = event.get("node_id", self._node_id)

        if config_key and pid:
            # Run in thread to avoid blocking event loop
            await asyncio.to_thread(self.heartbeat, config_key, pid, node_id)

    async def _on_training_lock_acquired(self, event: dict[str, Any]) -> None:
        """Handle TRAINING_LOCK_ACQUIRED event - register process."""
        config_key = event.get("config_key", "")
        pid = event.get("pid", os.getpid())
        node_id = event.get("node_id", self._node_id)

        if config_key:
            await asyncio.to_thread(self.register_training_process, config_key, pid, node_id)

    async def _on_training_completed(self, event: dict[str, Any]) -> None:
        """Handle TRAINING_COMPLETED event - mark as completed."""
        config_key = event.get("config_key", "")
        node_id = event.get("node_id", self._node_id)

        if config_key:
            await asyncio.to_thread(self.mark_completed, config_key, node_id)

    # =========================================================================
    # Main Watchdog Cycle
    # =========================================================================

    async def _run_cycle(self) -> None:
        """Check for and kill stale training processes."""
        stale_processes = await asyncio.to_thread(self._find_stale_processes)

        if not stale_processes:
            logger.debug(f"[{self.name}] No stale processes found")
            return

        logger.warning(
            f"[{self.name}] Found {len(stale_processes)} stale training process(es)"
        )

        for proc in stale_processes:
            await self._handle_stale_process(proc)

    def _find_stale_processes(self) -> list[TrainingProcessInfo]:
        """Find processes that haven't sent a heartbeat recently.

        Returns:
            List of stale process info
        """
        self._ensure_db_initialized()
        stale_threshold = time.time() - self._daemon_config.stale_threshold_seconds

        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT config_key, pid, node_id, started_at, last_heartbeat, status
                    FROM training_processes
                    WHERE status = 'running'
                    AND last_heartbeat < ?
                    """,
                    (stale_threshold,),
                )
                rows = cursor.fetchall()

            return [
                TrainingProcessInfo(
                    config_key=row["config_key"],
                    pid=row["pid"],
                    node_id=row["node_id"],
                    started_at=row["started_at"],
                    last_heartbeat=row["last_heartbeat"],
                    status=row["status"],
                )
                for row in rows
            ]

        except sqlite3.Error as e:
            logger.error(f"[{self.name}] Failed to find stale processes: {e}")
            return []

    async def _handle_stale_process(self, proc: TrainingProcessInfo) -> None:
        """Handle a stale training process.

        Args:
            proc: Stale process information
        """
        stale_duration = time.time() - proc.last_heartbeat
        logger.warning(
            f"[{self.name}] Stale process detected: "
            f"config={proc.config_key}, pid={proc.pid}, node={proc.node_id}, "
            f"stale_for={stale_duration:.1f}s"
        )

        # Only kill processes on the local node
        if proc.node_id == self._node_id:
            await self._kill_local_process(proc)
        else:
            # For remote processes, just mark as killed in database
            logger.info(
                f"[{self.name}] Remote stale process - marking killed: "
                f"config={proc.config_key}, node={proc.node_id}"
            )
            await asyncio.to_thread(self._mark_process_killed, proc)

        # Release associated lock
        if self._daemon_config.release_locks_on_kill:
            await self._release_training_lock(proc.config_key)

        # Emit event
        await self._emit_process_killed_event(proc, stale_duration)

    async def _kill_local_process(self, proc: TrainingProcessInfo) -> None:
        """Kill a stale process on the local node.

        Args:
            proc: Process to kill
        """
        if not self._daemon_config.enable_process_kill:
            logger.info(f"[{self.name}] Process kill disabled, would kill pid={proc.pid}")
            await asyncio.to_thread(self._mark_process_killed, proc)
            return

        pid = proc.pid

        # Check if process still exists
        if not self._process_exists(pid):
            logger.info(f"[{self.name}] Process {pid} no longer exists")
            await asyncio.to_thread(self._mark_process_killed, proc)
            return

        # Send SIGTERM first
        try:
            logger.warning(f"[{self.name}] Sending SIGTERM to pid={pid}")
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            logger.info(f"[{self.name}] Process {pid} already gone")
            await asyncio.to_thread(self._mark_process_killed, proc)
            return
        except PermissionError:
            logger.error(f"[{self.name}] No permission to kill pid={pid}")
            await asyncio.to_thread(self._mark_process_killed, proc)
            return

        # Wait for grace period
        await asyncio.sleep(self._daemon_config.kill_grace_period_seconds)

        # Check if still alive, send SIGKILL
        if self._process_exists(pid):
            try:
                logger.warning(f"[{self.name}] Process {pid} still alive, sending SIGKILL")
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                logger.error(f"[{self.name}] No permission to SIGKILL pid={pid}")

        await asyncio.to_thread(self._mark_process_killed, proc)
        self._processes_killed += 1
        logger.info(f"[{self.name}] Killed stale process: pid={pid}, config={proc.config_key}")

    def _process_exists(self, pid: int) -> bool:
        """Check if a process exists.

        Args:
            pid: Process ID

        Returns:
            True if process exists
        """
        try:
            os.kill(pid, 0)  # Signal 0 doesn't kill, just checks existence
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def _mark_process_killed(self, proc: TrainingProcessInfo) -> None:
        """Mark a process as killed in the database.

        Args:
            proc: Process to mark
        """
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute(
                    """
                    UPDATE training_processes
                    SET status = 'killed'
                    WHERE config_key = ? AND node_id = ?
                    """,
                    (proc.config_key, proc.node_id),
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"[{self.name}] Failed to mark process killed: {e}")

    async def _release_training_lock(self, config_key: str) -> None:
        """Release the training lock for a config.

        Args:
            config_key: Training configuration
        """
        try:
            from app.coordination.distributed_lock import DistributedLock

            lock_name = f"training:{config_key}"
            lock = DistributedLock(lock_name, lock_timeout=1)

            # Force-release by acquiring and immediately releasing
            # This works because locks have TTL and the original holder is dead
            if await asyncio.to_thread(lock.acquire, 1):
                await asyncio.to_thread(lock.release)
                self._locks_released += 1
                logger.info(f"[{self.name}] Released training lock: {lock_name}")
            else:
                logger.debug(f"[{self.name}] Lock already released: {lock_name}")

        except ImportError:
            logger.debug(f"[{self.name}] DistributedLock not available")
        except Exception as e:
            logger.error(f"[{self.name}] Failed to release lock: {e}")

    async def _emit_process_killed_event(
        self, proc: TrainingProcessInfo, stale_duration: float
    ) -> None:
        """Emit TRAINING_PROCESS_KILLED event.

        Args:
            proc: Killed process info
            stale_duration: How long the process was stale
        """
        safe_emit_event(
            "TRAINING_PROCESS_KILLED",
            {
                "config_key": proc.config_key,
                "pid": proc.pid,
                "node_id": proc.node_id,
                "started_at": proc.started_at,
                "last_heartbeat": proc.last_heartbeat,
                "stale_duration_seconds": stale_duration,
                "killed_at": time.time(),
                "killed_by": self._node_id,
            },
            source="training_watchdog_daemon",
            context="process_killed",
        )

    # =========================================================================
    # Health Check
    # =========================================================================

    def health_check(self) -> HealthCheckResult:
        """Return health check result."""
        details = {
            "running": self._running,
            "processes_registered": self._processes_registered,
            "processes_killed": self._processes_killed,
            "heartbeats_received": self._heartbeats_received,
            "locks_released": self._locks_released,
            "uptime_seconds": self.uptime_seconds,
            "cycles_completed": self._stats.cycles_completed,
            "errors_count": self._stats.errors_count,
            "db_path": str(self._db_path),
            "stale_threshold_hours": self._daemon_config.stale_threshold_seconds / 3600,
        }

        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message=f"{self.name} is not running",
                details=details,
            )

        # Check for high kill rate (might indicate broader issues)
        if self._processes_killed > 10:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.DEGRADED,
                message=f"{self.name} has killed {self._processes_killed} processes",
                details=details,
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"{self.name} healthy, {self._heartbeats_received} heartbeats received",
            details=details,
        )

    def get_status(self) -> dict[str, Any]:
        """Get daemon status for monitoring."""
        health = self.health_check()
        return {
            "name": self.name,
            "running": self._running,
            "uptime_seconds": self.uptime_seconds,
            "config": {
                "check_interval": self._daemon_config.check_interval_seconds,
                "stale_threshold": self._daemon_config.stale_threshold_seconds,
                "kill_grace_period": self._daemon_config.kill_grace_period_seconds,
            },
            "health": {
                "healthy": health.healthy,
                "status": health.status.value if hasattr(health.status, "value") else str(health.status),
                "message": health.message,
            },
            **health.details,
        }

    def get_active_training_processes(self) -> list[TrainingProcessInfo]:
        """Get list of currently active training processes.

        Returns:
            List of active training process info
        """
        self._ensure_db_initialized()

        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT config_key, pid, node_id, started_at, last_heartbeat, status
                    FROM training_processes
                    WHERE status = 'running'
                    """,
                )
                rows = cursor.fetchall()

            return [
                TrainingProcessInfo(
                    config_key=row["config_key"],
                    pid=row["pid"],
                    node_id=row["node_id"],
                    started_at=row["started_at"],
                    last_heartbeat=row["last_heartbeat"],
                    status=row["status"],
                )
                for row in rows
            ]

        except sqlite3.Error as e:
            logger.error(f"[{self.name}] Failed to get active processes: {e}")
            return []


# =============================================================================
# Singleton Access
# =============================================================================


def get_training_watchdog_daemon() -> TrainingWatchdogDaemon:
    """Get the singleton TrainingWatchdogDaemon instance."""
    return TrainingWatchdogDaemon.get_instance()


def reset_training_watchdog_daemon() -> None:
    """Reset the singleton (for testing)."""
    TrainingWatchdogDaemon.reset_instance()


# =============================================================================
# Heartbeat Helper Functions
# =============================================================================


def send_training_heartbeat(config_key: str, pid: int | None = None) -> None:
    """Send a training heartbeat from the training process.

    This is a convenience function for train.py to call periodically.

    Args:
        config_key: Training configuration (e.g., "hex8_2p")
        pid: Process ID (defaults to current process)
    """
    pid = pid or os.getpid()

    success = safe_emit_event(
        "TRAINING_HEARTBEAT",
        {
            "config_key": config_key,
            "pid": pid,
            "node_id": socket.gethostname(),
            "timestamp": time.time(),
        },
        source="send_training_heartbeat",
        context="heartbeat",
    )

    if not success:
        # Fallback: directly update the watchdog daemon
        try:
            daemon = TrainingWatchdogDaemon.get_instance()
            if daemon._running:
                daemon.heartbeat(config_key, pid)
        except Exception:
            pass  # Best effort
