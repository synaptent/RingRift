"""Base Daemon Class for RingRift Coordination.

This module provides a base class for all cluster daemons, consolidating
common patterns for:
- Lifecycle management (start/stop)
- Coordinator protocol registration
- Error handling and resilience
- Metrics and health reporting

Dec 2025: Extracted from 6+ daemon implementations to reduce ~650 LOC duplication.

Usage:
    from app.coordination.base_daemon import BaseDaemon, DaemonConfig

    @dataclass
    class MyDaemonConfig(DaemonConfig):
        my_setting: str = "default"

    class MyDaemon(BaseDaemon[MyDaemonConfig]):
        async def _run_cycle(self) -> None:
            # Implement daemon-specific logic
            pass
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Generic, TypeVar

from app.coordination.protocols import (
    CoordinatorStatus,
    HealthCheckResult,
    register_coordinator,
    unregister_coordinator,
)
from app.core.async_context import safe_create_task

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Base Class
# =============================================================================


@dataclass
class DaemonConfig:
    """Base configuration for all daemons.

    Subclass this to add daemon-specific configuration.
    All daemons share:
    - enabled: Whether the daemon should run
    - check_interval_seconds: How often to run the main cycle
    """

    enabled: bool = True
    check_interval_seconds: int = 300  # 5 minutes default

    @classmethod
    def from_env(cls, prefix: str = "RINGRIFT") -> "DaemonConfig":
        """Load configuration from environment variables.

        Override in subclass for daemon-specific env vars.

        Args:
            prefix: Environment variable prefix (e.g., RINGRIFT_WATCHDOG_)
        """
        config = cls()
        env_prefix = f"{prefix}_"

        # Common environment variables
        enabled_key = f"{env_prefix}ENABLED"
        interval_key = f"{env_prefix}INTERVAL"

        if os.environ.get(enabled_key):
            config.enabled = os.environ.get(enabled_key, "1") == "1"
        if os.environ.get(interval_key):
            try:
                config.check_interval_seconds = int(os.environ.get(interval_key, "300"))
            except ValueError:
                pass

        return config


ConfigT = TypeVar("ConfigT", bound=DaemonConfig)


# =============================================================================
# Base Daemon Class
# =============================================================================


class BaseDaemon(ABC, Generic[ConfigT]):
    """Abstract base class for all cluster daemons.

    Provides common functionality:
    - Lifecycle management (start, stop, is_running)
    - Coordinator protocol registration
    - Protected main loop with error handling
    - Metrics and status reporting
    - Health check interface

    Subclasses must implement:
    - _run_cycle(): The daemon's main work function
    - _get_default_config(): Return default config instance

    Optional overrides:
    - _on_start(): Called after daemon starts
    - _on_stop(): Called before daemon stops
    - _get_daemon_name(): Return custom daemon name
    - get_status(): Extend with daemon-specific status
    """

    def __init__(self, config: ConfigT | None = None):
        """Initialize the daemon.

        Args:
            config: Daemon configuration. If None, uses default from _get_default_config().
        """
        self.config: ConfigT = config or self._get_default_config()
        self._running = False
        self._task: asyncio.Task | None = None
        self._start_time: float = 0.0

        # Coordinator protocol fields
        self._coordinator_status = CoordinatorStatus.INITIALIZING
        self._events_processed: int = 0
        self._errors_count: int = 0
        self._last_error: str = ""
        self._cycles_completed: int = 0

        # Node identification
        self.node_id = socket.gethostname()
        self._hostname = socket.gethostname()

        logger.info(f"[{self._get_daemon_name()}] Initialized on {self.node_id}")

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    async def start(self) -> None:
        """Start the daemon.

        Registers with coordinator protocol and starts the main loop.
        """
        if self._running:
            logger.warning(f"[{self._get_daemon_name()}] Already running")
            return

        if not self.config.enabled:
            logger.info(f"[{self._get_daemon_name()}] Disabled by configuration")
            return

        self._running = True
        self._start_time = time.time()
        self._coordinator_status = CoordinatorStatus.RUNNING

        # Register with coordinator protocol
        self._coordinator_register()

        # Allow subclass initialization
        await self._on_start()

        # Start main loop
        self._task = safe_create_task(
            self._protected_loop(),
            name=f"{self._get_daemon_name()}_main_loop",
        )

        logger.info(f"[{self._get_daemon_name()}] Started")

    async def stop(self) -> None:
        """Stop the daemon.

        Cancels the main loop and unregisters from coordinator protocol.
        """
        if not self._running:
            return

        logger.info(f"[{self._get_daemon_name()}] Stopping...")
        self._running = False
        self._coordinator_status = CoordinatorStatus.STOPPED

        # Allow subclass cleanup
        await self._on_stop()

        # Cancel main loop
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        # Unregister from coordinator
        self._coordinator_unregister()

        logger.info(f"[{self._get_daemon_name()}] Stopped after {self.uptime_seconds:.1f}s")

    @property
    def is_running(self) -> bool:
        """Check if daemon is currently running."""
        return self._running

    @property
    def uptime_seconds(self) -> float:
        """Get daemon uptime in seconds."""
        if self._start_time > 0:
            return time.time() - self._start_time
        return 0.0

    # =========================================================================
    # Main Loop
    # =========================================================================

    async def _protected_loop(self) -> None:
        """Protected main loop with error handling.

        Wraps _run_cycle() with:
        - Exception handling and logging
        - Error counting
        - Configurable sleep interval
        """
        try:
            while self._running:
                try:
                    await self._run_cycle()
                    self._cycles_completed += 1
                except asyncio.CancelledError:
                    raise  # Re-raise to exit loop
                except Exception as e:
                    self._errors_count += 1
                    self._last_error = str(e)
                    self._coordinator_status = CoordinatorStatus.ERROR
                    logger.error(f"[{self._get_daemon_name()}] Cycle failed: {e}")

                    # Reset to running after error (will try again next cycle)
                    if self._running:
                        self._coordinator_status = CoordinatorStatus.RUNNING

                # Wait for next cycle
                await asyncio.sleep(self.config.check_interval_seconds)

        except asyncio.CancelledError:
            logger.debug(f"[{self._get_daemon_name()}] Main loop cancelled")
        except Exception as e:
            logger.error(f"[{self._get_daemon_name()}] Main loop crashed: {e}")
            self._running = False
            self._coordinator_status = CoordinatorStatus.ERROR
            self._last_error = str(e)

    # =========================================================================
    # Abstract Methods (Must Implement)
    # =========================================================================

    @abstractmethod
    async def _run_cycle(self) -> None:
        """Run one daemon cycle.

        This is the main work function. Called repeatedly with
        config.check_interval_seconds delay between calls.

        Implementations should:
        - Perform the daemon's primary function
        - Be idempotent (safe to retry on failure)
        - Not catch exceptions (base class handles them)
        """
        pass

    @staticmethod
    @abstractmethod
    def _get_default_config() -> ConfigT:
        """Return default configuration.

        Override to return daemon-specific config class instance.
        """
        pass

    # =========================================================================
    # Optional Overrides
    # =========================================================================

    async def _on_start(self) -> None:
        """Called after daemon starts.

        Override to add daemon-specific startup logic.
        """
        pass

    async def _on_stop(self) -> None:
        """Called before daemon stops.

        Override to add daemon-specific cleanup logic.
        """
        pass

    def _get_daemon_name(self) -> str:
        """Return daemon name for logging and registration.

        Override to customize the daemon name.
        """
        return self.__class__.__name__

    # =========================================================================
    # Status and Health
    # =========================================================================

    def get_status(self) -> dict[str, Any]:
        """Get daemon status for monitoring.

        Override to add daemon-specific status fields.
        """
        return {
            "daemon": self._get_daemon_name(),
            "running": self._running,
            "uptime_seconds": self.uptime_seconds,
            "node_id": self.node_id,
            "config": {
                "enabled": self.config.enabled,
                "interval": self.config.check_interval_seconds,
            },
            "stats": {
                "cycles_completed": self._cycles_completed,
                "events_processed": self._events_processed,
                "errors": self._errors_count,
                "last_error": self._last_error,
            },
            "coordinator_status": self._coordinator_status.value
            if hasattr(self._coordinator_status, "value")
            else str(self._coordinator_status),
        }

    def health_check(self) -> HealthCheckResult:
        """Perform health check.

        Override to add daemon-specific health criteria.
        """
        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message=f"{self._get_daemon_name()} is not running",
            )

        # Check error rate (unhealthy if >50% failure rate with sufficient cycles)
        if self._cycles_completed > 10:
            error_rate = self._errors_count / self._cycles_completed
            if error_rate > 0.5:
                return HealthCheckResult(
                    healthy=False,
                    status=CoordinatorStatus.ERROR,
                    message=f"High error rate: {error_rate:.1%}",
                    details=self.get_status(),
                )

        return HealthCheckResult(
            healthy=True,
            status=self._coordinator_status,
            message=f"{self._get_daemon_name()} healthy",
            details=self.get_status(),
        )

    # =========================================================================
    # Coordinator Protocol
    # =========================================================================

    def _coordinator_register(self) -> None:
        """Register with coordinator protocol."""
        try:
            register_coordinator(
                self._get_daemon_name(),
                CoordinatorStatus(
                    coordinator_type=self._get_daemon_name(),
                    is_running=True,
                    host=self._hostname,
                    start_time=self._start_time,
                ),
            )
        except Exception as e:
            logger.debug(f"[{self._get_daemon_name()}] Failed to register coordinator: {e}")

    def _coordinator_unregister(self) -> None:
        """Unregister from coordinator protocol."""
        try:
            unregister_coordinator(self._get_daemon_name())
        except Exception as e:
            logger.debug(f"[{self._get_daemon_name()}] Failed to unregister coordinator: {e}")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def record_event_processed(self, count: int = 1) -> None:
        """Record that events were processed.

        Args:
            count: Number of events processed
        """
        self._events_processed += count

    def record_error(self, error: str | Exception) -> None:
        """Record an error.

        Args:
            error: Error message or exception
        """
        self._errors_count += 1
        self._last_error = str(error)


# =============================================================================
# Convenience Exports
# =============================================================================

__all__ = [
    "BaseDaemon",
    "DaemonConfig",
]
