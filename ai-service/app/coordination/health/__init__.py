"""Unified health state types for RingRift coordination.

January 2026: Created as part of Phase 4.1 health state unification.

This package provides the canonical health status enum and related types.
All health-related modules should import from here.

Usage:
    from app.coordination.health import (
        HealthStatus,
        HealthStatusInfo,
        to_health_status,
    )

    # Convert from any legacy enum
    status = to_health_status("ok")  # Returns HealthStatus.HEALTHY
    status = to_health_status(OldHealthLevel.WARNING)  # Returns HealthStatus.DEGRADED

Migration:
    # Old imports (deprecated Q2 2026)
    from app.core.health import HealthState
    from app.distributed.health_registry import HealthLevel
    from app.coordination.unified_health_manager import SystemHealthLevel
    from app.coordination.node_status import NodeHealthState

    # New canonical import
    from app.coordination.health import HealthStatus, to_health_status
"""

from app.coordination.health.types import (
    HealthStatus,
    HealthStatusInfo,
    to_health_status,
    from_legacy_health_state,
    from_legacy_health_level,
    from_legacy_system_health_level,
    from_legacy_node_health_state,
    get_health_score,
    from_health_score,
)

__all__ = [
    "HealthStatus",
    "HealthStatusInfo",
    "to_health_status",
    "from_legacy_health_state",
    "from_legacy_health_level",
    "from_legacy_system_health_level",
    "from_legacy_node_health_state",
    "get_health_score",
    "from_health_score",
]
