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
import signal
import socket
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Generic, TypeVar

from app.coordination.health_check_helper import HealthCheckHelper
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


# Note: kw_only=True removed for Python 3.9 compatibility (requires 3.10+)
@dataclass
class DaemonConfig:
    """Base configuration for all daemons.

    Subclass this to add daemon-specific configuration.
    All daemons share:
    - enabled: Whether the daemon should run
    - check_interval_seconds: How often to run the main cycle
    - handle_signals: Whether to register SIGTERM/SIGINT handlers

    Note: Uses kw_only=True for safe dataclass inheritance.
    """

    enabled: bool = True
    check_interval_seconds: int = 300  # 5 minutes default
    handle_signals: bool = False  # Enable for daemons that need graceful shutdown

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

        # Signal handling
        signals_key = f"{env_prefix}HANDLE_SIGNALS"
        if os.environ.get(signals_key):
            config.handle_signals = os.environ.get(signals_key, "0") == "1"

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

        # Signal handling (December 2025: graceful shutdown support)
        self._original_sigterm_handler: Callable | int | None = None
        self._original_sigint_handler: Callable | int | None = None
        self._shutdown_in_progress = False

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

        # Register signal handlers if enabled
        if self.config.handle_signals:
            self._register_signal_handlers()

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

        # Unregister signal handlers
        if self.config.handle_signals:
            self._unregister_signal_handlers()

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
                except (KeyboardInterrupt, SystemExit):
                    raise  # Re-raise system signals
                except Exception as e:  # noqa: BLE001 - daemon must stay running
                    # Catches operational errors; programming errors would crash before here
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

    @property
    def name(self) -> str:
        """Return daemon name (CoordinatorProtocol compliance)."""
        return self._get_daemon_name()

    @property
    def status(self) -> CoordinatorStatus:
        """Return current status (CoordinatorProtocol compliance)."""
        return self._coordinator_status

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

        # Check error rate using HealthCheckHelper (unhealthy if >50% failure rate)
        is_healthy, msg = HealthCheckHelper.check_error_rate(
            errors=self._errors_count,
            cycles=self._cycles_completed,
            threshold=0.5,
        )
        if not is_healthy:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=msg,
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
            # Register self - BaseDaemon now implements CoordinatorProtocol
            # via name and status properties
            register_coordinator(self)
        except Exception as e:
            logger.debug(f"[{self._get_daemon_name()}] Failed to register coordinator: {e}")

    def _coordinator_unregister(self) -> None:
        """Unregister from coordinator protocol."""
        try:
            unregister_coordinator(self._get_daemon_name())
        except Exception as e:
            logger.debug(f"[{self._get_daemon_name()}] Failed to unregister coordinator: {e}")

    # =========================================================================
    # Signal Handling (December 2025: graceful shutdown support)
    # =========================================================================

    def _register_signal_handlers(self) -> None:
        """Register SIGTERM/SIGINT handlers for graceful shutdown.

        Called during start() when config.handle_signals=True.
        """
        try:
            loop = asyncio.get_running_loop()

            # Save original handlers
            self._original_sigterm_handler = signal.getsignal(signal.SIGTERM)
            self._original_sigint_handler = signal.getsignal(signal.SIGINT)

            # Register our handlers
            loop.add_signal_handler(signal.SIGTERM, self._handle_termination_signal)
            loop.add_signal_handler(signal.SIGINT, self._handle_termination_signal)

            logger.debug(f"[{self._get_daemon_name()}] Signal handlers registered")
        except (NotImplementedError, RuntimeError) as e:
            # Signal handlers not supported (e.g., Windows, non-main thread)
            logger.debug(f"[{self._get_daemon_name()}] Could not register signal handlers: {e}")

    def _unregister_signal_handlers(self) -> None:
        """Unregister signal handlers during shutdown."""
        try:
            loop = asyncio.get_running_loop()
            loop.remove_signal_handler(signal.SIGTERM)
            loop.remove_signal_handler(signal.SIGINT)
        except (NotImplementedError, RuntimeError, ValueError):
            pass

    def _handle_termination_signal(self) -> None:
        """Handle SIGTERM/SIGINT for graceful shutdown.

        This is called synchronously by the event loop. It schedules
        the async shutdown for execution.

        Override _on_graceful_shutdown() for custom pre-shutdown logic.
        """
        if self._shutdown_in_progress:
            logger.warning(f"[{self._get_daemon_name()}] Shutdown already in progress")
            return

        self._shutdown_in_progress = True
        logger.info(f"[{self._get_daemon_name()}] Received termination signal, initiating graceful shutdown")

        # Import here to avoid circular imports
        from app.core.async_context import fire_and_forget

        fire_and_forget(
            self._graceful_shutdown(),
            on_error=lambda e: logger.error(f"[{self._get_daemon_name()}] Graceful shutdown error: {e}"),
        )

    async def _graceful_shutdown(self) -> None:
        """Perform graceful shutdown sequence.

        Called when SIGTERM/SIGINT is received. Subclasses should override
        _on_graceful_shutdown() for custom pre-shutdown logic (e.g., final sync).
        """
        try:
            # Allow subclass to perform final actions (e.g., sync, flush)
            await self._on_graceful_shutdown()
        except Exception as e:
            logger.error(f"[{self._get_daemon_name()}] Error during graceful shutdown: {e}")
        finally:
            await self.stop()

    async def _on_graceful_shutdown(self) -> None:
        """Hook for subclass custom graceful shutdown logic.

        Override this to perform final actions before the daemon stops,
        such as:
        - Triggering a final data sync
        - Flushing buffers
        - Completing in-progress work
        - Emitting shutdown events

        This is called BEFORE stop() is called.
        """
        pass

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
