"""Pure Coordination Contracts - Zero-dependency protocols and types.

This module contains pure protocol definitions and data classes that have
NO dependencies on other coordination modules. This breaks circular import
chains by providing a single source of truth for interface definitions.

CRITICAL: Do NOT add any imports from app.coordination.* or other ai-service
modules. This file must only use standard library imports.

Usage:
    from app.coordination.contracts import (
        CoordinatorProtocol,
        DaemonProtocol,
        CoordinatorStatus,
        HealthCheckResult,
    )

Created: December 2025
Purpose: Break 8-cycle circular dependency chain in coordination modules
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol, runtime_checkable

__all__ = [
    # Enums
    "CoordinatorStatus",
    # Data classes
    "CoordinatorMetrics",
    "HealthCheckResult",
    # Protocols
    "ConfigurableProtocol",
    "CoordinatorProtocol",
    "DaemonProtocol",
    "EventDrivenProtocol",
    "HealthCheckable",
    "SyncManagerProtocol",
    # Health check helpers (December 29, 2025)
    "invoke_health_check",
    "is_health_checkable",
    # Registry functions
    "get_coordinator",
    "get_registered_coordinators",
    "is_coordinator",
    "is_daemon",
    "is_event_driven",
    "register_coordinator",
    "unregister_coordinator",
]


# =============================================================================
# Enums
# =============================================================================


class CoordinatorStatus(str, Enum):
    """Status of a coordinator.

    Canonical enum combining values from protocols.py, coordinator_base.py,
    and handler_base.py. This is the single source of truth.

    December 2025: Added STARTING for handler lifecycle compatibility.
    """

    INITIALIZING = "initializing"
    STARTING = "starting"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    DRAINING = "draining"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    DEGRADED = "degraded"


# =============================================================================
# Data Classes
# =============================================================================


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
    def healthy(cls, message: str = "", **details: Any) -> HealthCheckResult:
        """Create a healthy result.

        Args:
            message: Optional status message
            **details: Additional details to include

        Returns:
            HealthCheckResult with healthy=True
        """
        return cls(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=message,
            details=details,
        )

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

    @classmethod
    def from_metrics(
        cls,
        uptime_seconds: float,
        events_processed: int = 0,
        errors_count: int = 0,
        error_rate: float | None = None,
        last_activity_ago: float | None = None,
        max_inactivity_seconds: float = 300.0,
        max_error_rate: float = 0.1,
        **extra_details: Any,
    ) -> HealthCheckResult:
        """Create a health check result from metrics.

        December 2025: Added to standardize health check creation from metrics.

        Args:
            uptime_seconds: How long the component has been running
            events_processed: Number of events processed
            errors_count: Number of errors encountered
            error_rate: Error rate (errors/events), computed if not provided
            last_activity_ago: Seconds since last activity
            max_inactivity_seconds: Max allowed inactivity before degraded
            max_error_rate: Max error rate before unhealthy
            **extra_details: Additional details to include

        Returns:
            HealthCheckResult with computed status
        """
        # Compute error rate if not provided
        if error_rate is None and events_processed > 0:
            error_rate = errors_count / events_processed
        elif error_rate is None:
            error_rate = 0.0

        # Build details
        details = {
            "uptime_seconds": uptime_seconds,
            "events_processed": events_processed,
            "errors_count": errors_count,
            "error_rate": round(error_rate, 4),
            **extra_details,
        }

        if last_activity_ago is not None:
            details["last_activity_ago"] = last_activity_ago

        # Determine health status
        if error_rate > max_error_rate:
            return cls(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"High error rate: {error_rate:.1%} > {max_error_rate:.1%}",
                details=details,
            )

        if last_activity_ago is not None and last_activity_ago > max_inactivity_seconds:
            return cls(
                healthy=True,  # Still running but degraded
                status=CoordinatorStatus.DEGRADED,
                message=f"Inactive for {last_activity_ago:.0f}s (max: {max_inactivity_seconds:.0f}s)",
                details=details,
            )

        if error_rate > max_error_rate / 2:
            return cls(
                healthy=True,
                status=CoordinatorStatus.DEGRADED,
                message=f"Elevated error rate: {error_rate:.1%}",
                details=details,
            )

        return cls(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message="OK",
            details=details,
        )

    def health_score(self) -> float:
        """Compute a health score from 0.0 (dead) to 1.0 (perfect health).

        December 2025: Added for aggregate health scoring.

        Returns:
            Float between 0.0 and 1.0
        """
        if not self.healthy:
            return 0.0

        if self.status == CoordinatorStatus.ERROR:
            return 0.1
        elif self.status == CoordinatorStatus.DEGRADED:
            return 0.7
        elif self.status in (CoordinatorStatus.RUNNING, CoordinatorStatus.READY):
            return 1.0
        elif self.status == CoordinatorStatus.PAUSED:
            return 0.5
        elif self.status == CoordinatorStatus.DRAINING:
            return 0.6
        elif self.status == CoordinatorStatus.STOPPING:
            return 0.3
        elif self.status == CoordinatorStatus.STOPPED:
            return 0.0
        elif self.status == CoordinatorStatus.INITIALIZING:
            return 0.8

        return 0.5  # Unknown status

    def with_details(self, **extra_details: Any) -> HealthCheckResult:
        """Return a new HealthCheckResult with additional details.

        Args:
            **extra_details: Additional details to merge

        Returns:
            New HealthCheckResult with merged details
        """
        merged = {**self.details, **extra_details}
        return HealthCheckResult(
            healthy=self.healthy,
            status=self.status,
            message=self.message,
            timestamp=self.timestamp,
            details=merged,
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
# Health Check Protocol and Helpers
# =============================================================================


@runtime_checkable
class HealthCheckable(Protocol):
    """Protocol for components that support health checking.

    December 29, 2025: Added to unify health check signatures across 51+ coordinators.

    Supports both sync and async health_check() methods. Use invoke_health_check()
    helper to call health_check() regardless of whether it's sync or async.

    Example:
        class MyDaemon(HealthCheckable):
            def health_check(self) -> HealthCheckResult:
                return HealthCheckResult(healthy=True)

        # Works with both sync and async implementations:
        result = await invoke_health_check(my_daemon)
    """

    def health_check(self) -> HealthCheckResult:
        """Return current health status.

        Returns:
            HealthCheckResult with healthy, status, message, and details.
        """
        ...


async def invoke_health_check(
    component: Any,
    default_on_error: bool = False,
) -> HealthCheckResult:
    """Invoke health_check() on a component, handling both sync and async.

    December 29, 2025: Added to handle the mix of sync (30+) and async (5)
    health_check implementations across the coordination infrastructure.

    Args:
        component: Object with a health_check() method (sync or async)
        default_on_error: If True, return unhealthy result on error instead of raising

    Returns:
        HealthCheckResult from the component

    Raises:
        AttributeError: If component has no health_check method (and default_on_error=False)
        Exception: Any exception from health_check (and default_on_error=False)

    Example:
        # Works with sync implementations:
        result = await invoke_health_check(sync_daemon)

        # Works with async implementations:
        result = await invoke_health_check(async_daemon)

        # Safe mode returns unhealthy on errors:
        result = await invoke_health_check(unknown_component, default_on_error=True)
    """
    import asyncio
    import inspect

    try:
        if not hasattr(component, "health_check"):
            if default_on_error:
                return HealthCheckResult(
                    healthy=False,
                    status=CoordinatorStatus.ERROR,
                    message="Component has no health_check method",
                    details={"component_type": type(component).__name__},
                )
            raise AttributeError(f"{type(component).__name__} has no health_check method")

        method = component.health_check

        # Check if it's a coroutine function (async def)
        if asyncio.iscoroutinefunction(method):
            return await method()

        # Check if calling it returns a coroutine (e.g., decorated async)
        result = method()
        if asyncio.iscoroutine(result):
            return await result

        # It's a sync method, result is already the HealthCheckResult
        return result

    except Exception as e:
        if default_on_error:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"Health check failed: {e}",
                details={
                    "component_type": type(component).__name__,
                    "error_type": type(e).__name__,
                },
            )
        raise


def is_health_checkable(component: Any) -> bool:
    """Check if a component has a health_check method.

    Args:
        component: Object to check

    Returns:
        True if component has a health_check method
    """
    return hasattr(component, "health_check") and callable(getattr(component, "health_check"))


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
        """Unique name identifying this coordinator."""
        ...

    @property
    def status(self) -> CoordinatorStatus:
        """Current status of the coordinator."""
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
        """Whether the daemon is currently running."""
        ...

    @property
    def uptime_seconds(self) -> float:
        """Time since daemon started, in seconds."""
        ...

    async def restart(self) -> None:
        """Restart the daemon. Should stop and then start."""
        ...

    async def pause(self) -> None:
        """Pause daemon processing without stopping."""
        ...

    async def resume(self) -> None:
        """Resume daemon processing after pause."""
        ...


@runtime_checkable
class EventDrivenProtocol(CoordinatorProtocol, Protocol):
    """Protocol for event-driven coordinators.

    Extends CoordinatorProtocol with event subscription management.
    """

    def subscribe_to_events(self) -> bool:
        """Subscribe to relevant events.

        Returns:
            True if successfully subscribed
        """
        ...

    def unsubscribe_from_events(self) -> None:
        """Unsubscribe from all events."""
        ...

    @property
    def subscribed_events(self) -> list[str]:
        """List of event types this coordinator is subscribed to."""
        ...


@runtime_checkable
class ConfigurableProtocol(Protocol):
    """Protocol for coordinators with runtime configuration.

    Enables dynamic configuration updates without restart.
    """

    def get_config(self) -> dict[str, Any]:
        """Get current configuration."""
        ...

    def update_config(self, config: dict[str, Any]) -> bool:
        """Update configuration at runtime.

        Returns:
            True if configuration was updated successfully
        """
        ...


@runtime_checkable
class SyncManagerProtocol(Protocol):
    """Protocol for sync managers that coordinate data synchronization.

    Sync managers handle replicating data (databases, models, NPZ files)
    across cluster nodes. This protocol defines the minimal interface
    required for sync manager integration with the coordination infrastructure.

    Created: December 29, 2025
    See: app/coordination/sync_base.py for base class implementation
    """

    @property
    def is_running(self) -> bool:
        """Check if sync manager is currently running."""
        ...

    async def sync_with_node(self, node: str) -> bool:
        """Sync with a specific node.

        Args:
            node: Node identifier to sync with

        Returns:
            True if sync succeeded, False otherwise
        """
        ...

    async def sync_with_cluster(self) -> dict[str, bool]:
        """Sync with all nodes in the cluster.

        Returns:
            Dict mapping node -> success status
        """
        ...

    async def start(self) -> None:
        """Start the sync manager background loop."""
        ...

    async def stop(self) -> None:
        """Stop the sync manager."""
        ...

    def get_status(self) -> dict[str, Any]:
        """Get current sync status.

        Returns:
            Status dict with sync progress, failed nodes, etc.
        """
        ...

    def health_check(self) -> HealthCheckResult:
        """Check health of the sync manager.

        Returns:
            HealthCheckResult indicating current health
        """
        ...


# =============================================================================
# Coordinator Registry (Pure Functions)
# =============================================================================

# Global registry - simple dict, no external dependencies
_COORDINATOR_REGISTRY: dict[str, Any] = {}


def register_coordinator(coordinator: Any, name: str | None = None) -> None:
    """Register a coordinator instance in the global registry.

    Args:
        coordinator: Coordinator instance to register
        name: Optional name override (defaults to coordinator.name)
    """
    if name is None:
        if hasattr(coordinator, "name"):
            name = coordinator.name
        else:
            name = type(coordinator).__name__

    _COORDINATOR_REGISTRY[name] = coordinator


def unregister_coordinator(name: str) -> None:
    """Unregister a coordinator from the global registry.

    Args:
        name: Name of coordinator to unregister
    """
    _COORDINATOR_REGISTRY.pop(name, None)


def get_coordinator(name: str) -> Any | None:
    """Retrieve a registered coordinator by name.

    Args:
        name: Coordinator name

    Returns:
        Coordinator instance or None if not found
    """
    return _COORDINATOR_REGISTRY.get(name)


def get_registered_coordinators() -> dict[str, Any]:
    """Get all registered coordinators.

    Returns:
        Dictionary of name -> coordinator mappings
    """
    return dict(_COORDINATOR_REGISTRY)


def is_coordinator(obj: Any) -> bool:
    """Check if object implements CoordinatorProtocol.

    Args:
        obj: Object to check

    Returns:
        True if object implements CoordinatorProtocol
    """
    return isinstance(obj, CoordinatorProtocol)


def is_daemon(obj: Any) -> bool:
    """Check if object implements DaemonProtocol.

    Args:
        obj: Object to check

    Returns:
        True if object implements DaemonProtocol
    """
    return isinstance(obj, DaemonProtocol)


def is_event_driven(obj: Any) -> bool:
    """Check if object implements EventDrivenProtocol.

    Args:
        obj: Object to check

    Returns:
        True if object implements EventDrivenProtocol
    """
    return isinstance(obj, EventDrivenProtocol)
