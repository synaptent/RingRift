"""Health Probe Interface for RingRift AI Service.

Provides Kubernetes-style health probes for components:
- Liveness: Is the component alive? (restart if not)
- Readiness: Is the component ready to accept traffic?
- Startup: Has the component finished starting?

Usage:
    from app.core.health import HealthCheck, HealthStatus, HealthRegistry

    class DatabasePool(HealthCheck):
        async def check_health(self) -> HealthStatus:
            try:
                await self._pool.execute("SELECT 1")
                return HealthStatus.healthy("Database responding")
            except Exception as e:
                return HealthStatus.unhealthy(f"Database error: {e}")

    # Register and aggregate
    registry = HealthRegistry()
    registry.register("database", db_pool)
    registry.register("cache", cache_manager)

    # Check all at once
    result = await registry.check_all()
    print(result.is_healthy)  # True if all healthy
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "HealthCheck",
    "HealthRegistry",
    "HealthResult",
    "HealthState",
    "HealthStatus",
    "ProbeType",
    "health_check",
]


class HealthState(Enum):
    """Health states for components.

    DEPRECATED (January 2026): Use app.coordination.health.HealthStatus instead.
    This enum will be removed in Q2 2026.

    Migration:
        from app.coordination.health import HealthStatus, from_legacy_health_state

        # Convert existing HealthState to HealthStatus
        status = from_legacy_health_state(HealthState.HEALTHY)  # Returns HealthStatus.HEALTHY

        # Or use HealthStatus directly
        status = HealthStatus.HEALTHY

    Mapping:
        HEALTHY -> HealthStatus.HEALTHY
        DEGRADED -> HealthStatus.DEGRADED
        UNHEALTHY -> HealthStatus.UNHEALTHY
        UNKNOWN -> HealthStatus.UNKNOWN
    """
    HEALTHY = "healthy"
    DEGRADED = "degraded"  # Working but not optimal
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

    def to_health_status(self):
        """Convert to canonical HealthStatus.

        Returns:
            HealthStatus equivalent of this HealthState.
        """
        from app.coordination.health import from_legacy_health_state
        return from_legacy_health_state(self)


class ProbeType(Enum):
    """Types of health probes (Kubernetes-style)."""
    LIVENESS = "liveness"    # Is it alive? (restart if fails)
    READINESS = "readiness"  # Can it accept traffic?
    STARTUP = "startup"      # Has it finished starting?


@dataclass
class HealthStatus:
    """Result of a health check.

    Attributes:
        state: The health state
        message: Human-readable status message
        details: Additional diagnostic details
        latency_ms: Time taken to check health
        timestamp: When the check was performed
    """
    state: HealthState
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def is_healthy(self) -> bool:
        """Check if status indicates health."""
        return self.state == HealthState.HEALTHY

    @property
    def is_degraded(self) -> bool:
        """Check if status indicates degraded state."""
        return self.state == HealthState.DEGRADED

    @property
    def is_unhealthy(self) -> bool:
        """Check if status indicates unhealthy state."""
        return self.state == HealthState.UNHEALTHY

    @classmethod
    def healthy(cls, message: str = "OK", **details: Any) -> HealthStatus:
        """Create a healthy status."""
        return cls(state=HealthState.HEALTHY, message=message, details=details)

    @classmethod
    def degraded(cls, message: str, **details: Any) -> HealthStatus:
        """Create a degraded status."""
        return cls(state=HealthState.DEGRADED, message=message, details=details)

    @classmethod
    def unhealthy(cls, message: str, **details: Any) -> HealthStatus:
        """Create an unhealthy status."""
        return cls(state=HealthState.UNHEALTHY, message=message, details=details)

    @classmethod
    def unknown(cls, message: str = "Check not performed", **details: Any) -> HealthStatus:
        """Create an unknown status."""
        return cls(state=HealthState.UNKNOWN, message=message, details=details)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "state": self.state.value,
            "message": self.message,
            "details": self.details,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
        }


class HealthCheck(ABC):
    """Abstract base class for health-checkable components.

    Implement this interface to make a component health-checkable.
    Can support multiple probe types for Kubernetes deployments.

    Example:
        class CacheManager(HealthCheck):
            async def check_health(self) -> HealthStatus:
                if not self._connected:
                    return HealthStatus.unhealthy("Not connected")
                if self._hit_rate < 0.5:
                    return HealthStatus.degraded("Low hit rate",
                        hit_rate=self._hit_rate)
                return HealthStatus.healthy("Cache operational",
                    hit_rate=self._hit_rate,
                    size=len(self._cache))
    """

    @abstractmethod
    async def check_health(self) -> HealthStatus:
        """Perform health check.

        Returns:
            HealthStatus indicating component health
        """

    async def check_liveness(self) -> HealthStatus:
        """Check if component is alive.

        Override for custom liveness logic.
        Default delegates to check_health().
        """
        return await self.check_health()

    async def check_readiness(self) -> HealthStatus:
        """Check if component is ready for traffic.

        Override for custom readiness logic.
        Default delegates to check_health().
        """
        return await self.check_health()

    async def check_startup(self) -> HealthStatus:
        """Check if component has finished starting.

        Override for custom startup logic.
        Default delegates to check_health().
        """
        return await self.check_health()

    async def check(self, probe_type: ProbeType = ProbeType.READINESS) -> HealthStatus:
        """Check health for specific probe type.

        Args:
            probe_type: Type of probe to check

        Returns:
            HealthStatus for the probe type
        """
        start = time.time()

        try:
            if probe_type == ProbeType.LIVENESS:
                status = await self.check_liveness()
            elif probe_type == ProbeType.STARTUP:
                status = await self.check_startup()
            else:
                status = await self.check_readiness()

            status.latency_ms = (time.time() - start) * 1000
            return status

        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.warning(f"Health check failed: {e}")
            return HealthStatus(
                state=HealthState.UNHEALTHY,
                message=f"Check failed: {e}",
                latency_ms=latency,
            )


@dataclass
class HealthResult:
    """Aggregated health result from multiple components.

    Attributes:
        state: Overall health state
        components: Individual component statuses
        total_latency_ms: Total time to check all components
        timestamp: When the check was performed
    """
    state: HealthState
    components: dict[str, HealthStatus]
    total_latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def is_healthy(self) -> bool:
        """Check if overall state is healthy."""
        return self.state == HealthState.HEALTHY

    @property
    def healthy_count(self) -> int:
        """Count of healthy components."""
        return sum(1 for s in self.components.values() if s.is_healthy)

    @property
    def unhealthy_count(self) -> int:
        """Count of unhealthy components."""
        return sum(1 for s in self.components.values() if s.is_unhealthy)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "state": self.state.value,
            "is_healthy": self.is_healthy,
            "components": {
                name: status.to_dict()
                for name, status in self.components.items()
            },
            "healthy_count": self.healthy_count,
            "unhealthy_count": self.unhealthy_count,
            "total_latency_ms": self.total_latency_ms,
            "timestamp": self.timestamp,
        }


class HealthRegistry:
    """Registry for aggregating health checks across components.

    Manages multiple health-checkable components and provides
    aggregated health status.

    Example:
        registry = HealthRegistry()
        registry.register("database", db_pool)
        registry.register("cache", cache_manager)
        registry.register("queue", message_queue)

        # Check all components
        result = await registry.check_all()
        if not result.is_healthy:
            print(f"Unhealthy components: {result.unhealthy_count}")

        # Check specific probe
        liveness = await registry.check_all(probe_type=ProbeType.LIVENESS)
    """

    def __init__(self, timeout: float = 5.0):
        """Initialize the registry.

        Args:
            timeout: Default timeout for health checks in seconds
        """
        self._components: dict[str, HealthCheck] = {}
        self._timeout = timeout
        self._last_result: HealthResult | None = None
        self._critical_components: set[str] = set()

    def register(
        self,
        name: str,
        component: HealthCheck,
        critical: bool = True,
    ) -> None:
        """Register a component for health checking.

        Args:
            name: Unique name for the component
            component: The health-checkable component
            critical: If True, component failure makes overall status unhealthy
        """
        self._components[name] = component
        if critical:
            self._critical_components.add(name)
        logger.debug(f"Registered health check: {name} (critical={critical})")

    def unregister(self, name: str) -> None:
        """Unregister a component."""
        self._components.pop(name, None)
        self._critical_components.discard(name)

    @property
    def component_names(self) -> list[str]:
        """Get registered component names."""
        return list(self._components.keys())

    @property
    def last_result(self) -> HealthResult | None:
        """Get the last health check result."""
        return self._last_result

    async def check(
        self,
        name: str,
        probe_type: ProbeType = ProbeType.READINESS,
        timeout: float | None = None,
    ) -> HealthStatus:
        """Check health of a specific component.

        Args:
            name: Component name
            probe_type: Type of probe to check
            timeout: Timeout in seconds (uses default if None)

        Returns:
            HealthStatus for the component
        """
        component = self._components.get(name)
        if component is None:
            return HealthStatus.unknown(f"Component '{name}' not registered")

        timeout = timeout or self._timeout

        try:
            status = await asyncio.wait_for(
                component.check(probe_type),
                timeout=timeout,
            )
            return status
        except asyncio.TimeoutError:
            return HealthStatus.unhealthy(
                f"Health check timed out after {timeout}s"
            )
        except Exception as e:
            return HealthStatus.unhealthy(f"Health check error: {e}")

    async def check_all(
        self,
        probe_type: ProbeType = ProbeType.READINESS,
        parallel: bool = True,
        timeout: float | None = None,
    ) -> HealthResult:
        """Check health of all registered components.

        Args:
            probe_type: Type of probe to check
            parallel: Run checks in parallel
            timeout: Timeout per check in seconds

        Returns:
            Aggregated HealthResult
        """
        start = time.time()
        timeout = timeout or self._timeout

        if parallel:
            statuses = await self._check_parallel(probe_type, timeout)
        else:
            statuses = await self._check_sequential(probe_type, timeout)

        # Determine overall state
        overall_state = self._compute_overall_state(statuses)

        result = HealthResult(
            state=overall_state,
            components=statuses,
            total_latency_ms=(time.time() - start) * 1000,
        )

        self._last_result = result
        return result

    async def _check_parallel(
        self,
        probe_type: ProbeType,
        timeout: float,
    ) -> dict[str, HealthStatus]:
        """Check all components in parallel."""
        async def check_one(name: str) -> tuple[str, HealthStatus]:
            status = await self.check(name, probe_type, timeout)
            return name, status

        results = await asyncio.gather(
            *[check_one(name) for name in self._components],
            return_exceptions=True,
        )

        statuses = {}
        for result in results:
            if isinstance(result, Exception):
                continue
            name, status = result
            statuses[name] = status

        return statuses

    async def _check_sequential(
        self,
        probe_type: ProbeType,
        timeout: float,
    ) -> dict[str, HealthStatus]:
        """Check all components sequentially."""
        statuses = {}
        for name in self._components:
            statuses[name] = await self.check(name, probe_type, timeout)
        return statuses

    def _compute_overall_state(
        self,
        statuses: dict[str, HealthStatus],
    ) -> HealthState:
        """Compute overall health state from component statuses."""
        if not statuses:
            return HealthState.UNKNOWN

        # Check critical components first
        for name in self._critical_components:
            status = statuses.get(name)
            if status and status.is_unhealthy:
                return HealthState.UNHEALTHY

        # Count states
        unhealthy_count = sum(1 for s in statuses.values() if s.is_unhealthy)
        degraded_count = sum(1 for s in statuses.values() if s.is_degraded)

        if unhealthy_count > 0:
            return HealthState.UNHEALTHY
        if degraded_count > 0:
            return HealthState.DEGRADED

        return HealthState.HEALTHY


# Decorator for simple health checks
def health_check(
    name: str | None = None,
    cache_seconds: float = 0,
) -> Callable:
    """Decorator to create a simple health check function.

    Usage:
        @health_check("database")
        async def check_database() -> HealthStatus:
            # Check database health
            return HealthStatus.healthy()

    Args:
        name: Name for the health check (uses function name if None)
        cache_seconds: Cache result for this many seconds
    """
    def decorator(func: Callable[[], HealthStatus]) -> Callable:
        _cache: dict[str, Any] = {"status": None, "expires": 0}
        check_name = name or func.__name__

        async def wrapper() -> HealthStatus:
            now = time.time()

            # Return cached if valid
            if cache_seconds > 0 and _cache["expires"] > now:
                return _cache["status"]

            # Perform check
            if asyncio.iscoroutinefunction(func):
                status = await func()
            else:
                status = func()

            # Cache result
            if cache_seconds > 0:
                _cache["status"] = status
                _cache["expires"] = now + cache_seconds

            return status

        wrapper.__name__ = check_name
        wrapper._is_health_check = True
        return wrapper

    return decorator


# Simple health check implementations
class FunctionHealthCheck(HealthCheck):
    """Health check that wraps a function."""

    def __init__(
        self,
        check_func: Callable[[], HealthStatus | bool],
        name: str = "function_check",
    ):
        self._check_func = check_func
        self._name = name

    async def check_health(self) -> HealthStatus:
        try:
            if asyncio.iscoroutinefunction(self._check_func):
                result = await self._check_func()
            else:
                result = self._check_func()

            if isinstance(result, HealthStatus):
                return result
            elif isinstance(result, bool):
                return HealthStatus.healthy() if result else HealthStatus.unhealthy("Check returned False")
            else:
                return HealthStatus.healthy(str(result))
        except Exception as e:
            return HealthStatus.unhealthy(f"Check raised: {e}")


class CompositeHealthCheck(HealthCheck):
    """Health check that combines multiple checks."""

    def __init__(self, checks: dict[str, HealthCheck]):
        self._checks = checks

    async def check_health(self) -> HealthStatus:
        statuses = {}
        for name, check in self._checks.items():
            statuses[name] = await check.check_health()

        unhealthy = [n for n, s in statuses.items() if s.is_unhealthy]
        if unhealthy:
            return HealthStatus.unhealthy(
                f"Unhealthy: {', '.join(unhealthy)}",
                components=statuses,
            )

        degraded = [n for n, s in statuses.items() if s.is_degraded]
        if degraded:
            return HealthStatus.degraded(
                f"Degraded: {', '.join(degraded)}",
                components=statuses,
            )

        return HealthStatus.healthy("All components healthy", components=statuses)
