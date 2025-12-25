"""Daemon Health Watchdog - Active monitoring with automatic recovery.

This module provides proactive health monitoring for daemons managed by DaemonManager.
Unlike passive monitoring (waiting for failures), it actively detects stuck daemons,
auto-restarts failed daemons with appropriate cooldowns, and emits alerts for issues
requiring manual intervention.

Key Features:
    - Periodic health checks every 30s
    - Detects stuck RUNNING daemons (task.done() but state still RUNNING)
    - Auto-restarts failed daemons with exponential backoff
    - Emits alerts for IMPORT_FAILED daemons (require manual fix)
    - Integrates with EventRouter for alerting

Usage:
    from app.coordination.daemon_watchdog import DaemonWatchdog, start_watchdog

    # Start watchdog (auto-integrates with daemon manager)
    watchdog = await start_watchdog()

    # Or manual control
    watchdog = DaemonWatchdog()
    await watchdog.start()
    ...
    await watchdog.stop()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.coordination.daemon_manager import DaemonManager

logger = logging.getLogger(__name__)


class WatchdogAlert(str, Enum):
    """Alert types emitted by the watchdog."""

    DAEMON_STUCK = "daemon_stuck"  # Task done but state RUNNING
    DAEMON_CRASHED = "daemon_crashed"  # Unexpected failure
    DAEMON_IMPORT_FAILED = "daemon_import_failed"  # Import error, needs manual fix
    DAEMON_RESTART_EXHAUSTED = "daemon_restart_exhausted"  # Max restarts exceeded
    DAEMON_AUTO_RESTARTED = "daemon_auto_restarted"  # Successfully auto-restarted
    WATCHDOG_STARTED = "watchdog_started"
    WATCHDOG_STOPPED = "watchdog_stopped"


@dataclass
class WatchdogConfig:
    """Configuration for daemon watchdog."""

    # Health check interval
    check_interval_seconds: float = 30.0

    # Minimum time between auto-restart attempts for same daemon
    auto_restart_cooldown_seconds: float = 60.0

    # Maximum auto-restarts per daemon before requiring manual intervention
    max_auto_restarts: int = 3

    # Enable auto-restart functionality
    auto_restart_enabled: bool = True

    # Time window for counting auto-restarts (resets after this)
    auto_restart_window_seconds: float = 3600.0  # 1 hour


@dataclass
class DaemonHealthRecord:
    """Health tracking for a single daemon."""

    last_check_time: float = 0.0
    last_restart_time: float = 0.0
    auto_restart_count: int = 0
    first_auto_restart_time: float = 0.0
    last_alert_time: float = 0.0
    consecutive_healthy_checks: int = 0


class DaemonWatchdog:
    """Active health monitoring for daemons with automatic recovery.

    The watchdog runs periodic health checks to detect:
    1. Stuck daemons - task completed but state still RUNNING
    2. Crashed daemons - unexpected FAILED state
    3. Import failures - IMPORT_FAILED state requiring manual fix

    For recoverable failures, it attempts auto-restart with cooldown.
    For non-recoverable issues, it emits alerts for manual intervention.
    """

    def __init__(
        self,
        config: WatchdogConfig | None = None,
        manager: DaemonManager | None = None,
    ):
        self.config = config or WatchdogConfig()
        self._manager = manager
        self._running = False
        self._task: asyncio.Task | None = None
        self._health_records: dict[str, DaemonHealthRecord] = {}
        self._alert_callbacks: list[callable] = []
        self._lock = asyncio.Lock()

    @property
    def manager(self) -> DaemonManager:
        """Get the daemon manager, lazily loading if needed."""
        if self._manager is None:
            from app.coordination.daemon_manager import get_daemon_manager

            self._manager = get_daemon_manager()
        return self._manager

    def add_alert_callback(self, callback: callable) -> None:
        """Add a callback for alerts. Callback receives (alert_type, daemon_name, details)."""
        self._alert_callbacks.append(callback)

    async def _emit_alert(
        self,
        alert_type: WatchdogAlert,
        daemon_name: str,
        details: dict | None = None,
    ) -> None:
        """Emit an alert through all registered callbacks and event router."""
        details = details or {}
        details["daemon_name"] = daemon_name
        details["alert_type"] = alert_type.value
        details["timestamp"] = time.time()

        # Call registered callbacks
        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert_type, daemon_name, details)
                else:
                    callback(alert_type, daemon_name, details)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

        # Emit through event router if available
        try:
            from app.coordination.event_router import get_router
            from app.distributed.data_events import DataEventType

            router = get_router()
            if router:
                await router.publish(
                    event_type=DataEventType.DAEMON_STATUS_CHANGED,
                    payload={
                        "watchdog_alert": alert_type.value,
                        **details,
                    },
                    source="DaemonWatchdog",
                )
        except Exception as e:
            logger.debug(f"Event router not available for alert: {e}")

        # Log the alert
        log_level = (
            logging.WARNING
            if alert_type
            in (
                WatchdogAlert.DAEMON_IMPORT_FAILED,
                WatchdogAlert.DAEMON_RESTART_EXHAUSTED,
            )
            else logging.INFO
        )
        logger.log(log_level, f"Watchdog alert: {alert_type.value} for {daemon_name}")

    def _get_health_record(self, daemon_name: str) -> DaemonHealthRecord:
        """Get or create health record for a daemon."""
        if daemon_name not in self._health_records:
            self._health_records[daemon_name] = DaemonHealthRecord()
        return self._health_records[daemon_name]

    def _reset_auto_restart_window(
        self, record: DaemonHealthRecord, now: float
    ) -> None:
        """Reset auto-restart count if outside the window."""
        if record.first_auto_restart_time > 0:
            elapsed = now - record.first_auto_restart_time
            if elapsed > self.config.auto_restart_window_seconds:
                record.auto_restart_count = 0
                record.first_auto_restart_time = 0.0

    async def _check_daemon_health(self, daemon_name: str) -> None:
        """Check health of a single daemon and take action if needed."""
        from app.coordination.daemon_manager import DaemonState, DaemonType

        try:
            daemon_type = DaemonType(daemon_name)
        except ValueError:
            return  # Unknown daemon type

        info = self.manager.get_daemon_info(daemon_type)
        if info is None:
            return  # Daemon not registered

        record = self._get_health_record(daemon_name)
        now = time.time()
        record.last_check_time = now

        # Reset auto-restart window if needed
        self._reset_auto_restart_window(record, now)

        # Check for stuck daemon (task done but state RUNNING)
        if info.state == DaemonState.RUNNING and info.task is not None:
            if info.task.done():
                logger.warning(f"Detected stuck daemon: {daemon_name}")
                await self._emit_alert(
                    WatchdogAlert.DAEMON_STUCK,
                    daemon_name,
                    {"state": info.state.value, "task_done": True},
                )
                # Force state update
                try:
                    exc = info.task.exception()
                    if exc:
                        info.state = DaemonState.FAILED
                        info.last_error = str(exc)
                    else:
                        info.state = DaemonState.STOPPED
                except Exception:
                    info.state = DaemonState.FAILED
                    info.last_error = "Task completed unexpectedly"

        # Check for import failure (requires manual fix)
        if info.state == DaemonState.IMPORT_FAILED:
            # Only alert once per hour
            if now - record.last_alert_time > 3600:
                record.last_alert_time = now
                await self._emit_alert(
                    WatchdogAlert.DAEMON_IMPORT_FAILED,
                    daemon_name,
                    {
                        "import_error": info.import_error,
                        "action_required": "Fix import error and restart manually",
                    },
                )
            return  # Don't auto-restart import failures

        # Check for failed daemon (auto-restart if enabled)
        if info.state == DaemonState.FAILED:
            if not self.config.auto_restart_enabled:
                return

            # Check cooldown
            if now - record.last_restart_time < self.config.auto_restart_cooldown_seconds:
                return  # Still in cooldown

            # Check max restarts
            if record.auto_restart_count >= self.config.max_auto_restarts:
                # Only alert once per hour
                if now - record.last_alert_time > 3600:
                    record.last_alert_time = now
                    await self._emit_alert(
                        WatchdogAlert.DAEMON_RESTART_EXHAUSTED,
                        daemon_name,
                        {
                            "auto_restart_count": record.auto_restart_count,
                            "action_required": "Manual restart required",
                        },
                    )
                return

            # Attempt auto-restart
            logger.info(f"Auto-restarting failed daemon: {daemon_name}")
            try:
                success = await self.manager.restart_failed_daemon(
                    daemon_type, force=True
                )
                if success:
                    record.last_restart_time = now
                    record.auto_restart_count += 1
                    if record.first_auto_restart_time == 0:
                        record.first_auto_restart_time = now
                    record.consecutive_healthy_checks = 0

                    await self._emit_alert(
                        WatchdogAlert.DAEMON_AUTO_RESTARTED,
                        daemon_name,
                        {
                            "attempt": record.auto_restart_count,
                            "max_attempts": self.config.max_auto_restarts,
                        },
                    )
                else:
                    logger.error(f"Failed to auto-restart daemon: {daemon_name}")
            except Exception as e:
                logger.error(f"Error auto-restarting {daemon_name}: {e}")

        # Track healthy checks (for stability monitoring)
        if info.state == DaemonState.RUNNING:
            record.consecutive_healthy_checks += 1

    async def _health_check_loop(self) -> None:
        """Main health check loop."""
        logger.info("Daemon watchdog started")
        await self._emit_alert(WatchdogAlert.WATCHDOG_STARTED, "watchdog", {})

        while self._running:
            try:
                # Get all registered daemons
                status = self.manager.get_status()
                daemons = status.get("daemons", {})

                # Check each daemon
                async with self._lock:
                    for daemon_name in daemons:
                        await self._check_daemon_health(daemon_name)

            except Exception as e:
                logger.error(f"Watchdog health check error: {e}")

            # Wait for next check interval
            try:
                await asyncio.sleep(self.config.check_interval_seconds)
            except asyncio.CancelledError:
                break

        logger.info("Daemon watchdog stopped")
        await self._emit_alert(WatchdogAlert.WATCHDOG_STOPPED, "watchdog", {})

    async def start(self) -> None:
        """Start the watchdog."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._health_check_loop())
        self._task.add_done_callback(self._handle_task_error)

    def _handle_task_error(self, task: asyncio.Task) -> None:
        """Handle errors from the watchdog task."""
        try:
            if task.cancelled():
                return
            exc = task.exception()
            if exc:
                logger.error(f"DaemonWatchdog task failed: {exc}")
        except asyncio.InvalidStateError:
            pass

    async def stop(self) -> None:
        """Stop the watchdog."""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def get_health_status(self) -> dict:
        """Get health status for all monitored daemons."""
        return {
            "running": self._running,
            "check_interval": self.config.check_interval_seconds,
            "auto_restart_enabled": self.config.auto_restart_enabled,
            "daemons": {
                name: {
                    "last_check": record.last_check_time,
                    "auto_restart_count": record.auto_restart_count,
                    "consecutive_healthy": record.consecutive_healthy_checks,
                }
                for name, record in self._health_records.items()
            },
        }


# Module-level singleton
_watchdog: DaemonWatchdog | None = None


def get_watchdog(config: WatchdogConfig | None = None) -> DaemonWatchdog:
    """Get the singleton watchdog instance."""
    global _watchdog
    if _watchdog is None:
        _watchdog = DaemonWatchdog(config=config)
    return _watchdog


async def start_watchdog(config: WatchdogConfig | None = None) -> DaemonWatchdog:
    """Start the singleton watchdog."""
    watchdog = get_watchdog(config)
    await watchdog.start()
    return watchdog


async def stop_watchdog() -> None:
    """Stop the singleton watchdog."""
    global _watchdog
    if _watchdog:
        await _watchdog.stop()
