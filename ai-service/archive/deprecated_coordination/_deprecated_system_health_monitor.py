"""DEPRECATED: System Health Monitor - Use cluster.health instead.

.. deprecated:: December 2025
   This module is deprecated as part of the coordination module consolidation
   (67 modules → 15). All health functionality has been unified.

Migration Guide:
   OLD:
      from app.coordination.system_health_monitor import (
          SystemHealthMonitor,
          check_system_health,
      )

   NEW:
      from app.coordination.cluster.health import (
          check_cluster_health,
          is_cluster_healthy,
          get_health_summary,
      )

For node-specific health monitoring:
   NEW:
      from app.coordination.cluster.health import (
          NodeHealthMonitor,
          get_node_health_monitor,
      )

The unified health module provides better integration with eviction policies,
circuit breakers, and error recovery.

This wrapper will be removed in Q2 2026.
"""

import warnings

warnings.warn(
    "system_health_monitor is deprecated and will be removed in Q2 2026. "
    "Use 'from app.coordination.cluster.health import check_cluster_health' instead. "
    "See module docstring for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from cluster health
from app.coordination.host_health_policy import (
    check_cluster_health,
    is_cluster_healthy,
    get_health_summary,
)

# Alias for backward compatibility
SystemHealthMonitor = None  # Class-based monitor deprecated

def check_system_health():
    """DEPRECATED: Use check_cluster_health() instead."""
    warnings.warn(
        "check_system_health() is deprecated, use check_cluster_health()",
        DeprecationWarning,
        stacklevel=2,
    )
    return check_cluster_health()

__all__ = [
    "SystemHealthMonitor",
    "check_system_health",
    "check_cluster_health",
    "is_cluster_healthy",
    "get_health_summary",
]
