"""Unified Health Facade - Single entry point for all health operations.

This module consolidates health functionality from:
- unified_health_manager.py (system-level health, pipeline pause)
- health_check_orchestrator.py (node-level health, recovery escalation)

Replaces deprecated modules (Q2 2026 removal):
- system_health_monitor.py
- node_health_monitor.py

Usage:
    from app.coordination.health_facade import (
        # System-level health
        get_system_health_score,
        get_system_health_level,
        should_pause_pipeline,

        # Node-level health
        get_node_health,
        get_healthy_nodes,
        get_unhealthy_nodes,

        # Managers (for advanced use)
        get_health_manager,
        get_health_orchestrator,
    )

    # Check if pipeline should pause
    should_pause, reasons = should_pause_pipeline()

    # Get individual node health
    node = get_node_health("runpod-h100")
    if node and node.state == NodeHealthState.HEALTHY:
        # Node is healthy
        pass

Created: December 2025
Purpose: Unified health interface (consolidation phase)
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

# Re-export system-level health from unified_health_manager
from app.coordination.unified_health_manager import (
    # Core manager
    get_health_manager,
    UnifiedHealthManager,
    # Health levels and config
    SystemHealthLevel,
    SystemHealthConfig,
    SystemHealthScore,
    # Convenience functions
    get_system_health_score,
    get_system_health_level,
    should_pause_pipeline,
    is_pipeline_paused,  # Deprecated but still available
)

# Re-export node-level health from health_check_orchestrator
from app.coordination.health_check_orchestrator import (
    # Core orchestrator
    get_health_orchestrator,
    HealthCheckOrchestrator,
    # Node health types
    NodeHealthState,
    NodeHealthDetails,
)

if TYPE_CHECKING:
    from typing import Any


# =============================================================================
# Convenience Functions (unified interface)
# =============================================================================

def get_node_health(node_id: str) -> NodeHealthDetails | None:
    """Get health details for a specific node.

    Args:
        node_id: The node identifier

    Returns:
        NodeHealthDetails or None if node not tracked
    """
    return get_health_orchestrator().get_node_health(node_id)


def get_healthy_nodes() -> list[str]:
    """Get list of healthy node IDs.

    Returns:
        List of node IDs in HEALTHY state
    """
    orchestrator = get_health_orchestrator()
    return [
        node_id for node_id, details in orchestrator.node_health.items()
        if details.state == NodeHealthState.HEALTHY
    ]


def get_unhealthy_nodes() -> list[str]:
    """Get list of unhealthy node IDs.

    Returns:
        List of node IDs NOT in HEALTHY state
    """
    orchestrator = get_health_orchestrator()
    return [
        node_id for node_id, details in orchestrator.node_health.items()
        if details.state != NodeHealthState.HEALTHY
    ]


def get_degraded_nodes() -> list[str]:
    """Get list of degraded node IDs.

    Returns:
        List of node IDs in DEGRADED state
    """
    orchestrator = get_health_orchestrator()
    return [
        node_id for node_id, details in orchestrator.node_health.items()
        if details.state == NodeHealthState.DEGRADED
    ]


def get_offline_nodes() -> list[str]:
    """Get list of offline node IDs.

    Returns:
        List of node IDs in OFFLINE or RETIRED state
    """
    orchestrator = get_health_orchestrator()
    return [
        node_id for node_id, details in orchestrator.node_health.items()
        if details.state in (NodeHealthState.OFFLINE, NodeHealthState.RETIRED)
    ]


def mark_node_retired(node_id: str) -> bool:
    """Mark a node as retired (removed from active use).

    Args:
        node_id: The node to retire

    Returns:
        True if marked, False if node not found
    """
    return get_health_orchestrator().mark_retired(node_id)


def get_cluster_health_summary() -> dict[str, Any]:
    """Get a summary of cluster health.

    Returns:
        Dict with health counts and overall status
    """
    orchestrator = get_health_orchestrator()
    manager = get_health_manager()

    health_counts = {
        NodeHealthState.HEALTHY.value: 0,
        NodeHealthState.DEGRADED.value: 0,
        NodeHealthState.UNHEALTHY.value: 0,
        NodeHealthState.OFFLINE.value: 0,
        NodeHealthState.RETIRED.value: 0,
    }

    for details in orchestrator.node_health.values():
        state_value = details.state.value
        if state_value in health_counts:
            health_counts[state_value] += 1

    system_score = manager.calculate_system_health_score()
    should_pause, pause_reasons = should_pause_pipeline()

    return {
        "total_nodes": len(orchestrator.node_health),
        "node_counts": health_counts,
        "system_score": system_score.score,
        "system_level": system_score.level.value,
        "pipeline_paused": should_pause,
        "pause_reasons": pause_reasons,
    }


# =============================================================================
# Backward Compatibility (deprecated functions)
# =============================================================================

def get_node_health_monitor():
    """DEPRECATED: Use get_health_orchestrator() instead.

    Returns the HealthCheckOrchestrator for backward compatibility.
    """
    warnings.warn(
        "get_node_health_monitor() is deprecated. Use get_health_orchestrator() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_health_orchestrator()


def get_system_health():
    """DEPRECATED: Use get_health_manager() instead.

    Returns the UnifiedHealthManager for backward compatibility.
    """
    warnings.warn(
        "get_system_health() is deprecated. Use get_health_manager() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_health_manager()


__all__ = [
    # System-level health
    "get_health_manager",
    "UnifiedHealthManager",
    "SystemHealthLevel",
    "SystemHealthConfig",
    "SystemHealthScore",
    "get_system_health_score",
    "get_system_health_level",
    "should_pause_pipeline",
    "is_pipeline_paused",
    # Node-level health
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
    # Cluster summary
    "get_cluster_health_summary",
    # Backward compat (deprecated)
    "get_node_health_monitor",
    "get_system_health",
]
