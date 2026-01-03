"""Canonical health status types for RingRift.

January 2026: Phase 4.1 - Unified Health State System.

This module consolidates 5 different health state enums into one canonical enum:
- app.core.health.HealthState (HEALTHY, DEGRADED, UNHEALTHY, UNKNOWN)
- app.distributed.health_registry.HealthLevel (OK, WARNING, ERROR, UNKNOWN)
- app.coordination.unified_health_manager.SystemHealthLevel (HEALTHY, DEGRADED, UNHEALTHY, CRITICAL)
- app.coordination.node_status.NodeHealthState (HEALTHY, DEGRADED, UNHEALTHY, EVICTED, UNKNOWN, OFFLINE, PROVIDER_DOWN, RETIRED)
- app.monitoring.node_health_orchestrator.NodeHealthState (HEALTHY, UNHEALTHY, RECOVERING, UNKNOWN)

The canonical HealthStatus enum covers all use cases with consistent naming.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class HealthStatus(str, Enum):
    """Canonical health status for all subsystems.

    States (in order of severity, healthy to worst):
        HEALTHY: Operating normally, no issues
        DEGRADED: Working but not optimal (warnings, minor issues)
        UNHEALTHY: Needs attention (errors, failures)
        CRITICAL: Immediate attention needed (severe, cascading failures)
        EVICTED: Removed from active duty (too many failures)
        OFFLINE: Confirmed not reachable
        PROVIDER_DOWN: Cloud provider reports the node as down
        RETIRED: Manually decommissioned
        RECOVERING: Currently in recovery process
        UNKNOWN: Initial state, health not yet determined
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    EVICTED = "evicted"
    OFFLINE = "offline"
    PROVIDER_DOWN = "provider_down"
    RETIRED = "retired"
    RECOVERING = "recovering"
    UNKNOWN = "unknown"

    @property
    def is_operational(self) -> bool:
        """Check if this status indicates the component can do work."""
        return self in (HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.RECOVERING)

    @property
    def is_healthy(self) -> bool:
        """Check if this status indicates full health."""
        return self == HealthStatus.HEALTHY

    @property
    def is_degraded(self) -> bool:
        """Check if this status indicates degraded operation."""
        return self == HealthStatus.DEGRADED

    @property
    def is_unhealthy(self) -> bool:
        """Check if this status indicates unhealthy state."""
        return self in (HealthStatus.UNHEALTHY, HealthStatus.CRITICAL)

    @property
    def is_offline(self) -> bool:
        """Check if this status indicates the node is not reachable."""
        return self in (HealthStatus.OFFLINE, HealthStatus.PROVIDER_DOWN, HealthStatus.RETIRED, HealthStatus.EVICTED)

    @property
    def severity(self) -> int:
        """Return numeric severity (0=healthy, higher=worse)."""
        severity_map = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 1,
            HealthStatus.RECOVERING: 2,
            HealthStatus.UNHEALTHY: 3,
            HealthStatus.CRITICAL: 4,
            HealthStatus.EVICTED: 5,
            HealthStatus.OFFLINE: 6,
            HealthStatus.PROVIDER_DOWN: 7,
            HealthStatus.RETIRED: 8,
            HealthStatus.UNKNOWN: 9,
        }
        return severity_map.get(self, 9)

    def __lt__(self, other: HealthStatus) -> bool:
        if not isinstance(other, HealthStatus):
            return NotImplemented
        return self.severity < other.severity

    def __le__(self, other: HealthStatus) -> bool:
        if not isinstance(other, HealthStatus):
            return NotImplemented
        return self.severity <= other.severity

    def __gt__(self, other: HealthStatus) -> bool:
        if not isinstance(other, HealthStatus):
            return NotImplemented
        return self.severity > other.severity

    def __ge__(self, other: HealthStatus) -> bool:
        if not isinstance(other, HealthStatus):
            return NotImplemented
        return self.severity >= other.severity

    @classmethod
    def worst(cls, *statuses: HealthStatus) -> HealthStatus:
        """Return the worst (highest severity) status."""
        if not statuses:
            return cls.UNKNOWN
        return max(statuses, key=lambda s: s.severity)

    @classmethod
    def best(cls, *statuses: HealthStatus) -> HealthStatus:
        """Return the best (lowest severity) status."""
        if not statuses:
            return cls.UNKNOWN
        return min(statuses, key=lambda s: s.severity)


@dataclass
class HealthStatusInfo:
    """Extended health status with additional context."""

    status: HealthStatus
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    @property
    def is_operational(self) -> bool:
        return self.status.is_operational

    @property
    def is_healthy(self) -> bool:
        return self.status.is_healthy

    @classmethod
    def healthy(cls, message: str = "Operating normally", **details: Any) -> HealthStatusInfo:
        import time
        return cls(HealthStatus.HEALTHY, message, details, time.time())

    @classmethod
    def degraded(cls, message: str, **details: Any) -> HealthStatusInfo:
        import time
        return cls(HealthStatus.DEGRADED, message, details, time.time())

    @classmethod
    def unhealthy(cls, message: str, **details: Any) -> HealthStatusInfo:
        import time
        return cls(HealthStatus.UNHEALTHY, message, details, time.time())

    @classmethod
    def critical(cls, message: str, **details: Any) -> HealthStatusInfo:
        import time
        return cls(HealthStatus.CRITICAL, message, details, time.time())

    @classmethod
    def unknown(cls, message: str = "Status unknown", **details: Any) -> HealthStatusInfo:
        import time
        return cls(HealthStatus.UNKNOWN, message, details, time.time())


def to_health_status(value: Any) -> HealthStatus:
    """Convert any value to HealthStatus."""
    if value is None:
        return HealthStatus.UNKNOWN

    if isinstance(value, HealthStatus):
        return value

    if hasattr(value, "value"):
        value = value.value

    if isinstance(value, str):
        normalized = value.lower().strip()
        try:
            return HealthStatus(normalized)
        except ValueError:
            pass

        legacy_mapping = {
            "ok": HealthStatus.HEALTHY,
            "warning": HealthStatus.DEGRADED,
            "error": HealthStatus.UNHEALTHY,
            "good": HealthStatus.HEALTHY,
            "bad": HealthStatus.UNHEALTHY,
            "up": HealthStatus.HEALTHY,
            "down": HealthStatus.OFFLINE,
            "active": HealthStatus.HEALTHY,
            "inactive": HealthStatus.OFFLINE,
            "available": HealthStatus.HEALTHY,
            "unavailable": HealthStatus.OFFLINE,
        }

        if normalized in legacy_mapping:
            return legacy_mapping[normalized]

    return HealthStatus.UNKNOWN


def from_legacy_health_state(state: Any) -> HealthStatus:
    """Convert app.core.health.HealthState to HealthStatus."""
    if state is None:
        return HealthStatus.UNKNOWN
    value = state.value if hasattr(state, "value") else str(state)
    mapping = {
        "healthy": HealthStatus.HEALTHY,
        "degraded": HealthStatus.DEGRADED,
        "unhealthy": HealthStatus.UNHEALTHY,
        "unknown": HealthStatus.UNKNOWN,
    }
    return mapping.get(value.lower(), HealthStatus.UNKNOWN)


def from_legacy_health_level(level: Any) -> HealthStatus:
    """Convert app.distributed.health_registry.HealthLevel to HealthStatus."""
    if level is None:
        return HealthStatus.UNKNOWN
    value = level.value if hasattr(level, "value") else str(level)
    mapping = {
        "ok": HealthStatus.HEALTHY,
        "warning": HealthStatus.DEGRADED,
        "error": HealthStatus.UNHEALTHY,
        "unknown": HealthStatus.UNKNOWN,
    }
    return mapping.get(value.lower(), HealthStatus.UNKNOWN)


def from_legacy_system_health_level(level: Any) -> HealthStatus:
    """Convert app.coordination.unified_health_manager.SystemHealthLevel to HealthStatus."""
    if level is None:
        return HealthStatus.UNKNOWN
    value = level.value if hasattr(level, "value") else str(level)
    mapping = {
        "healthy": HealthStatus.HEALTHY,
        "degraded": HealthStatus.DEGRADED,
        "unhealthy": HealthStatus.UNHEALTHY,
        "critical": HealthStatus.CRITICAL,
    }
    return mapping.get(value.lower(), HealthStatus.UNKNOWN)


def from_legacy_node_health_state(state: Any) -> HealthStatus:
    """Convert app.coordination.node_status.NodeHealthState to HealthStatus."""
    if state is None:
        return HealthStatus.UNKNOWN
    value = state.value if hasattr(state, "value") else str(state)
    mapping = {
        "healthy": HealthStatus.HEALTHY,
        "degraded": HealthStatus.DEGRADED,
        "unhealthy": HealthStatus.UNHEALTHY,
        "evicted": HealthStatus.EVICTED,
        "unknown": HealthStatus.UNKNOWN,
        "offline": HealthStatus.OFFLINE,
        "provider_down": HealthStatus.PROVIDER_DOWN,
        "retired": HealthStatus.RETIRED,
        "recovering": HealthStatus.RECOVERING,
    }
    return mapping.get(value.lower(), HealthStatus.UNKNOWN)


def get_health_score(status: HealthStatus) -> float:
    """Convert HealthStatus to a 0.0-1.0 score."""
    score_map = {
        HealthStatus.HEALTHY: 1.0,
        HealthStatus.DEGRADED: 0.8,
        HealthStatus.RECOVERING: 0.6,
        HealthStatus.UNHEALTHY: 0.4,
        HealthStatus.CRITICAL: 0.2,
        HealthStatus.EVICTED: 0.0,
        HealthStatus.OFFLINE: 0.0,
        HealthStatus.PROVIDER_DOWN: 0.0,
        HealthStatus.RETIRED: 0.0,
        HealthStatus.UNKNOWN: 0.0,
    }
    return score_map.get(status, 0.0)


def from_health_score(score: float) -> HealthStatus:
    """Convert a 0.0-1.0 score to HealthStatus."""
    if score >= 0.9:
        return HealthStatus.HEALTHY
    elif score >= 0.7:
        return HealthStatus.DEGRADED
    elif score >= 0.5:
        return HealthStatus.RECOVERING
    elif score >= 0.3:
        return HealthStatus.UNHEALTHY
    elif score >= 0.1:
        return HealthStatus.CRITICAL
    else:
        return HealthStatus.OFFLINE
