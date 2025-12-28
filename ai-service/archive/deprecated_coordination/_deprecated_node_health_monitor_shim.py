"""Backward compatibility stub for deprecated node_health_monitor.

.. deprecated:: December 2025
    This module has been archived. Use health_check_orchestrator instead.

    Migration:
        # OLD (deprecated)
        from app.coordination.node_health_monitor import get_node_health_monitor

        # NEW (recommended)
        from app.coordination.health_check_orchestrator import get_health_orchestrator

    Or use the unified health_facade:
        from app.coordination.health_facade import get_node_health

    For NodeStatus/NodeHealth:
        from app.coordination.node_status import NodeHealthState, NodeMonitoringStatus
"""

from app.coordination.health_check_orchestrator import (
    HealthCheckOrchestrator,
    get_health_orchestrator,
)
from app.coordination.node_status import (
    NodeHealthState,
    NodeMonitoringStatus,
)

# Aliases for backward compatibility
NodeHealthMonitor = HealthCheckOrchestrator
get_node_health_monitor = get_health_orchestrator
NodeStatus = NodeHealthState  # Alias for legacy callers
NodeHealth = NodeMonitoringStatus  # Alias for legacy callers

__all__ = [
    "NodeHealthMonitor",
    "get_node_health_monitor",
    "HealthCheckOrchestrator",
    "get_health_orchestrator",
    "NodeStatus",
    "NodeHealth",
    "NodeHealthState",
    "NodeMonitoringStatus",
]
