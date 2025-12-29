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

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

# Import pure contracts - these have no dependencies on coordination modules
# and break circular import chains
from app.coordination.contracts import (
    ConfigurableProtocol,
    CoordinatorMetrics,
    CoordinatorProtocol,
    CoordinatorStatus,
    DaemonProtocol,
    EventDrivenProtocol,
    HealthCheckResult,
    SyncManagerProtocol,
    get_coordinator,
    get_registered_coordinators,
    is_coordinator,
    is_daemon,
    is_event_driven,
    register_coordinator,
    unregister_coordinator,
)

logger = logging.getLogger(__name__)

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
    "SyncManagerProtocol",
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
# Enums, Data Classes, and Protocols
# =============================================================================
# NOTE: The following are now imported from app.coordination.contracts to break
# circular dependency chains:
# - CoordinatorStatus, HealthCheckResult, CoordinatorMetrics
# - CoordinatorProtocol, DaemonProtocol, EventDrivenProtocol, ConfigurableProtocol
# - Registry functions: register_coordinator, unregister_coordinator, etc.
#
# This module now only contains implementation classes (BaseCoordinator, BaseDaemon)
# and additional utility functions.
# =============================================================================


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

    DEPRECATED (December 2025 Phase 10): This minimal implementation is deprecated.
    Use the feature-rich BaseDaemon from app.coordination.base_daemon instead:

        from app.coordination.base_daemon import BaseDaemon

    The base_daemon.py version provides:
    - Generic[ConfigT] for typed configuration
    - Full lifecycle management with protected main loop
    - Automatic health check interface
    - Metrics and status reporting
    - Coordinator protocol registration

    This class is retained for backward compatibility and will be removed Q2 2026.
    """

    def __init__(self):
        import warnings
        warnings.warn(
            "protocols.BaseDaemon is deprecated. Use app.coordination.base_daemon.BaseDaemon instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
            logger.error(f"Failed to start {name}: {e}")
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
            logger.error(f"Failed to stop {name}: {e}")
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
