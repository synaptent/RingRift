"""Coordination Protocols - Unified interfaces for all coordinators.

This module defines common protocols (interfaces) that all coordinators should
implement. This enables consistent lifecycle management, health checking, and
metrics collection across the 134+ coordinator classes in the system.

Protocols:
- CoordinatorProtocol: Base protocol for all coordinators
- DaemonProtocol: Protocol for long-running daemon coordinators
- EventDrivenProtocol: Protocol for event-subscribing coordinators

Usage:
    from app.coordination.protocols import (
        CoordinatorProtocol,
        CoordinatorStatus,
        HealthCheckResult,
    )

    class MyCoordinator(CoordinatorProtocol):
        @property
        def name(self) -> str:
            return "MyCoordinator"

        @property
        def status(self) -> CoordinatorStatus:
            return CoordinatorStatus.RUNNING if self._running else CoordinatorStatus.STOPPED

        async def start(self) -> None:
            self._running = True

        async def stop(self) -> None:
            self._running = False

        def get_metrics(self) -> dict[str, Any]:
            return {"items_processed": self._count}

        def health_check(self) -> HealthCheckResult:
            return HealthCheckResult(healthy=True)

Based on the architecture review identifying 134 coordinator classes with no common interface.
Created: December 2025
Purpose: Unified interface for coordinators (Phase 14)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

__all__ = [
    # Data classes and enums
    "CoordinatorMetrics",
    "CoordinatorStatus",
    "HealthCheckResult",
    # Protocols
    "ConfigurableProtocol",
    "CoordinatorProtocol",
    "DaemonProtocol",
    "EventDrivenProtocol",
    # Base classes
    "BaseCoordinator",
    "BaseDaemon",
    # Registry functions
    "get_all_metrics",
    "get_coordinator",
    "get_registered_coordinators",
    "health_check_all",
    "is_coordinator",
    "is_daemon",
    "is_event_driven",
    "register_coordinator",
    "unregister_coordinator",
    "validate_coordinator",
]


# =============================================================================
# Enums and Data Classes
# =============================================================================


class CoordinatorStatus(str, Enum):
    """Status of a coordinator."""

    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    DEGRADED = "degraded"  # Running but with reduced functionality


@dataclass
class HealthCheckResult:
    """Result of a health check.

    Provides a consistent structure for health check responses
    across all coordinators.
    """

    healthy: bool
    status: CoordinatorStatus = CoordinatorStatus.RUNNING
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "healthy": self.healthy,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "details": self.details,
        }

    @classmethod
    def unhealthy(cls, message: str, **details: Any) -> HealthCheckResult:
        """Create an unhealthy result."""
        return cls(
            healthy=False,
            status=CoordinatorStatus.ERROR,
            message=message,
            details=details,
        )

    @classmethod
    def degraded(cls, message: str, **details: Any) -> HealthCheckResult:
        """Create a degraded health result."""
        return cls(
            healthy=True,  # Still operational
            status=CoordinatorStatus.DEGRADED,
            message=message,
            details=details,
        )


@dataclass
class CoordinatorMetrics:
    """Standard metrics structure for coordinators.

    Provides a common format for metrics across coordinators.
    """

    name: str
    status: CoordinatorStatus
    uptime_seconds: float = 0.0
    start_time: float = 0.0
    events_processed: int = 0
    errors_count: int = 0
    last_error: str = ""
    last_activity_time: float = 0.0
    custom_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "uptime_seconds": self.uptime_seconds,
            "start_time": self.start_time,
            "events_processed": self.events_processed,
            "errors_count": self.errors_count,
            "last_error": self.last_error,
            "last_activity_time": self.last_activity_time,
            **self.custom_metrics,
        }


# =============================================================================
# Core Protocol
# =============================================================================


@runtime_checkable
class CoordinatorProtocol(Protocol):
    """Base protocol for all coordinators.

    All coordinator classes should implement this protocol to enable
    consistent lifecycle management and observability.

    This is a structural protocol - classes don't need to explicitly
    inherit from it, they just need to implement the required methods.
    """

    @property
    def name(self) -> str:
        """Unique name identifying this coordinator.

        Returns:
            Coordinator name (e.g., "TrainingCoordinator", "SyncDaemon")
        """
        ...

    @property
    def status(self) -> CoordinatorStatus:
        """Current status of the coordinator.

        Returns:
            Current status enum value
        """
        ...

    async def start(self) -> None:
        """Start the coordinator.

        Should be idempotent - calling on an already running coordinator
        should be a no-op.
        """
        ...

    async def stop(self) -> None:
        """Stop the coordinator gracefully.

        Should complete any in-progress work and clean up resources.
        Should be idempotent.
        """
        ...

    def get_metrics(self) -> dict[str, Any]:
        """Get coordinator metrics.

        Returns:
            Dictionary of metrics (events processed, error counts, etc.)
        """
        ...

    def health_check(self) -> HealthCheckResult:
        """Check coordinator health.

        Returns:
            Health check result with status and details
        """
        ...


# =============================================================================
# Extended Protocols
# =============================================================================


@runtime_checkable
class DaemonProtocol(CoordinatorProtocol, Protocol):
    """Protocol for long-running daemon coordinators.

    Extends CoordinatorProtocol with daemon-specific methods for
    restart, pause/resume, and background loop management.
    """

    @property
    def is_running(self) -> bool:
        """Whether the daemon is currently running.

        Returns:
            True if running
        """
        ...

    @property
    def uptime_seconds(self) -> float:
        """Time since daemon started, in seconds.

        Returns:
            Uptime in seconds
        """
        ...

    async def restart(self) -> None:
        """Restart the daemon.

        Should stop and then start the daemon.
        """
        ...

    async def pause(self) -> None:
        """Pause daemon processing without stopping.

        The daemon remains running but doesn't process new work.
        """
        ...

    async def resume(self) -> None:
        """Resume daemon processing after pause.

        Resumes normal operation.
        """
        ...


@runtime_checkable
class EventDrivenProtocol(CoordinatorProtocol, Protocol):
    """Protocol for event-driven coordinators.

    Extends CoordinatorProtocol with event subscription management.
    """

    def subscribe_to_events(self) -> bool:
        """Subscribe to relevant events.

        Should set up event subscriptions for the coordinator.

        Returns:
            True if successfully subscribed
        """
        ...

    def unsubscribe_from_events(self) -> None:
        """Unsubscribe from all events.

        Should clean up event subscriptions.
        """
        ...

    @property
    def subscribed_events(self) -> list[str]:
        """List of event types this coordinator is subscribed to.

        Returns:
            List of event type names
        """
        ...


@runtime_checkable
class ConfigurableProtocol(Protocol):
    """Protocol for coordinators with runtime configuration.

    Enables dynamic configuration updates without restart.
    """

    def get_config(self) -> dict[str, Any]:
        """Get current configuration.

        Returns:
            Dictionary of configuration values
        """
        ...

    def update_config(self, config: dict[str, Any]) -> bool:
        """Update configuration at runtime.

        Args:
            config: Dictionary of configuration values to update

        Returns:
            True if configuration was updated successfully
        """
        ...


# =============================================================================
# Abstract Base Classes
# =============================================================================


class BaseCoordinator(ABC):
    """Abstract base class for coordinators.

    Provides default implementations for common functionality.
    Coordinators can inherit from this for convenience.

    Note: Using ABC instead of Protocol here because we provide
    concrete implementations of some methods.
    """

    def __init__(self):
        self._status = CoordinatorStatus.INITIALIZING
        self._start_time: float = 0.0
        self._events_processed: int = 0
        self._errors_count: int = 0
        self._last_error: str = ""
        self._last_activity: float = 0.0

    @property
    @abstractmethod
    def name(self) -> str:
        """Return coordinator name."""
        ...

    @property
    def status(self) -> CoordinatorStatus:
        """Return current status."""
        return self._status

    @property
    def uptime_seconds(self) -> float:
        """Return uptime in seconds."""
        if self._start_time <= 0:
            return 0.0
        return time.time() - self._start_time

    async def start(self) -> None:
        """Start the coordinator."""
        if self._status == CoordinatorStatus.RUNNING:
            return  # Already running
        self._status = CoordinatorStatus.RUNNING
        self._start_time = time.time()
        await self._on_start()

    async def stop(self) -> None:
        """Stop the coordinator."""
        if self._status == CoordinatorStatus.STOPPED:
            return  # Already stopped
        self._status = CoordinatorStatus.STOPPING
        await self._on_stop()
        self._status = CoordinatorStatus.STOPPED

    @abstractmethod
    async def _on_start(self) -> None:
        """Hook for subclass start logic."""
        ...

    @abstractmethod
    async def _on_stop(self) -> None:
        """Hook for subclass stop logic."""
        ...

    def get_metrics(self) -> dict[str, Any]:
        """Get coordinator metrics."""
        return CoordinatorMetrics(
            name=self.name,
            status=self._status,
            uptime_seconds=self.uptime_seconds,
            start_time=self._start_time,
            events_processed=self._events_processed,
            errors_count=self._errors_count,
            last_error=self._last_error,
            last_activity_time=self._last_activity,
            custom_metrics=self._get_custom_metrics(),
        ).to_dict()

    def _get_custom_metrics(self) -> dict[str, Any]:
        """Override to provide custom metrics."""
        return {}

    def health_check(self) -> HealthCheckResult:
        """Check coordinator health."""
        if self._status == CoordinatorStatus.ERROR:
            return HealthCheckResult.unhealthy(
                f"Coordinator in error state: {self._last_error}"
            )
        if self._status == CoordinatorStatus.STOPPED:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="Coordinator is stopped",
            )
        if self._status == CoordinatorStatus.DEGRADED:
            return HealthCheckResult.degraded("Coordinator running in degraded mode")

        return HealthCheckResult(
            healthy=True,
            status=self._status,
            details={
                "uptime_seconds": self.uptime_seconds,
                "events_processed": self._events_processed,
            },
        )

    def _record_event(self) -> None:
        """Record that an event was processed."""
        self._events_processed += 1
        self._last_activity = time.time()

    def _record_error(self, error: str) -> None:
        """Record an error."""
        self._errors_count += 1
        self._last_error = error


class BaseDaemon(BaseCoordinator):
    """Abstract base class for daemon coordinators.

    Extends BaseCoordinator with daemon-specific functionality
    like pause/resume and restart.
    """

    def __init__(self):
        super().__init__()
        self._paused = False

    @property
    def is_running(self) -> bool:
        """Whether daemon is running."""
        return self._status == CoordinatorStatus.RUNNING

    async def restart(self) -> None:
        """Restart the daemon."""
        await self.stop()
        await self.start()

    async def pause(self) -> None:
        """Pause the daemon."""
        if self._status == CoordinatorStatus.RUNNING:
            self._paused = True
            self._status = CoordinatorStatus.PAUSED

    async def resume(self) -> None:
        """Resume the daemon."""
        if self._status == CoordinatorStatus.PAUSED:
            self._paused = False
            self._status = CoordinatorStatus.RUNNING


# =============================================================================
# Validation Helpers
# =============================================================================


def validate_coordinator(coordinator: Any) -> list[str]:
    """Validate that an object implements CoordinatorProtocol.

    Args:
        coordinator: Object to validate

    Returns:
        List of missing methods/properties (empty if valid)
    """
    required = ["name", "status", "start", "stop", "get_metrics", "health_check"]
    missing = []

    for attr in required:
        if not hasattr(coordinator, attr):
            missing.append(attr)
        elif attr in ("name", "status"):
            # Check if it's a property
            if not isinstance(getattr(type(coordinator), attr, None), property):
                # Could be a regular attribute, that's ok
                pass

    return missing


def is_coordinator(obj: Any) -> bool:
    """Check if an object implements CoordinatorProtocol.

    Args:
        obj: Object to check

    Returns:
        True if object implements the protocol
    """
    return isinstance(obj, CoordinatorProtocol)


def is_daemon(obj: Any) -> bool:
    """Check if an object implements DaemonProtocol.

    Args:
        obj: Object to check

    Returns:
        True if object implements the protocol
    """
    return isinstance(obj, DaemonProtocol)


def is_event_driven(obj: Any) -> bool:
    """Check if an object implements EventDrivenProtocol.

    Args:
        obj: Object to check

    Returns:
        True if object implements the protocol
    """
    return isinstance(obj, EventDrivenProtocol)


# =============================================================================
# Registry for Runtime Discovery
# =============================================================================


_coordinator_registry: dict[str, CoordinatorProtocol] = {}


def register_coordinator(coordinator: CoordinatorProtocol) -> None:
    """Register a coordinator for runtime discovery.

    Args:
        coordinator: Coordinator to register
    """
    _coordinator_registry[coordinator.name] = coordinator


def unregister_coordinator(name: str) -> None:
    """Unregister a coordinator.

    Args:
        name: Coordinator name to unregister
    """
    _coordinator_registry.pop(name, None)


def get_registered_coordinators() -> dict[str, CoordinatorProtocol]:
    """Get all registered coordinators.

    Returns:
        Dictionary mapping names to coordinators
    """
    return dict(_coordinator_registry)


def get_coordinator(name: str) -> CoordinatorProtocol | None:
    """Get a coordinator by name.

    Args:
        name: Coordinator name

    Returns:
        Coordinator or None if not found
    """
    return _coordinator_registry.get(name)


async def start_all_coordinators() -> dict[str, bool]:
    """Start all registered coordinators.

    Returns:
        Dictionary mapping names to success status
    """
    results = {}
    for name, coordinator in _coordinator_registry.items():
        try:
            await coordinator.start()
            results[name] = True
        except Exception as e:
            results[name] = False
            import logging
            logging.getLogger(__name__).error(f"Failed to start {name}: {e}")
    return results


async def stop_all_coordinators() -> dict[str, bool]:
    """Stop all registered coordinators.

    Returns:
        Dictionary mapping names to success status
    """
    results = {}
    for name, coordinator in _coordinator_registry.items():
        try:
            await coordinator.stop()
            results[name] = True
        except Exception as e:
            results[name] = False
            import logging
            logging.getLogger(__name__).error(f"Failed to stop {name}: {e}")
    return results


def health_check_all() -> dict[str, HealthCheckResult]:
    """Run health checks on all registered coordinators.

    Returns:
        Dictionary mapping names to health check results
    """
    results = {}
    for name, coordinator in _coordinator_registry.items():
        try:
            results[name] = coordinator.health_check()
        except Exception as e:
            results[name] = HealthCheckResult.unhealthy(f"Health check failed: {e}")
    return results


def get_all_metrics() -> dict[str, dict[str, Any]]:
    """Get metrics from all registered coordinators.

    Returns:
        Dictionary mapping names to metrics dictionaries
    """
    results = {}
    for name, coordinator in _coordinator_registry.items():
        try:
            results[name] = coordinator.get_metrics()
        except Exception as e:
            results[name] = {"error": str(e)}
    return results
