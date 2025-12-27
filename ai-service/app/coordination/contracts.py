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

    Canonical enum combining values from protocols.py and coordinator_base.py.
    """

    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    DRAINING = "draining"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    DEGRADED = "degraded"  # Running but with reduced functionality


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
