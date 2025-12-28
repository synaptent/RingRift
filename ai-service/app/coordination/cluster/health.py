"""Unified cluster health module (December 2025).

Consolidates health-related functionality from:
- unified_health_manager.py (error recovery, circuit breakers)
- host_health_policy.py (pre-spawn SSH health checks)
- node_health_monitor.py (async node monitoring, eviction)

This module re-exports all health-related APIs for unified access.

Usage:
    from app.coordination.cluster.health import (
        # Health manager
        UnifiedHealthManager,
        get_health_manager,
        wire_health_events,

        # Host health (pre-spawn checks)
        check_host_health,
        is_host_healthy,
        get_healthy_hosts,

        # Node monitoring
        NodeHealthMonitor,
        get_node_health_monitor,
        NodeStatus,
    )
"""

from __future__ import annotations

# Re-export from unified_health_manager
from app.coordination.unified_health_manager import (
    UnifiedHealthManager,
    get_health_manager,
    wire_health_events,
    ErrorSeverity,
    RecoveryStatus,
)

# Re-export from host_health_policy
from app.coordination.host_health_policy import (
    HealthStatus,
    check_host_health,
    is_host_healthy,
    get_healthy_hosts,
    clear_health_cache,
    get_health_summary,
    is_cluster_healthy,
    check_cluster_health,
)

# Re-export from health_facade (PREFERRED - unified interface, December 2025)
from app.coordination.health_facade import (
    # Node-level health (replaces node_health_monitor)
    get_health_orchestrator,
    HealthCheckOrchestrator,
    NodeHealthState,
    NodeHealthDetails,
    get_node_health,
    get_healthy_nodes,
    get_unhealthy_nodes,
    get_degraded_nodes,
    get_offline_nodes,
    mark_node_retired,
    get_cluster_health_summary,
    # System-level health
    get_system_health_score,
    get_system_health_level,
    should_pause_pipeline,
    SystemHealthLevel,
    SystemHealthScore,
)

# Re-export aliases from canonical modules (December 2025)
# These are backward-compatible aliases - prefer the health_facade imports above
from app.coordination.health_check_orchestrator import (
    HealthCheckOrchestrator as NodeHealthMonitor,
    get_health_orchestrator as get_node_health_monitor,
)
from app.coordination.node_status import (
    NodeHealthState as NodeStatus,
    NodeMonitoringStatus as NodeHealth,
)

__all__ = [
    # From unified_health_manager
    "UnifiedHealthManager",
    "get_health_manager",
    "wire_health_events",
    "ErrorSeverity",
    "RecoveryStatus",
    # From host_health_policy
    "HealthStatus",
    "check_host_health",
    "is_host_healthy",
    "get_healthy_hosts",
    "clear_health_cache",
    "get_health_summary",
    "is_cluster_healthy",
    "check_cluster_health",
    # From health_facade (PREFERRED - December 2025)
    "get_health_orchestrator",
    "HealthCheckOrchestrator",
    "NodeHealthState",
    "NodeHealthDetails",
    "get_node_health",
    "get_healthy_nodes",
    "get_unhealthy_nodes",
    "get_degraded_nodes",
    "get_offline_nodes",
    "mark_node_retired",
    "get_cluster_health_summary",
    "get_system_health_score",
    "get_system_health_level",
    "should_pause_pipeline",
    "SystemHealthLevel",
    "SystemHealthScore",
    # From node_health_monitor (DEPRECATED - Q2 2026 removal)
    "NodeHealthMonitor",
    "get_node_health_monitor",
    "NodeStatus",
    "NodeHealth",
]
