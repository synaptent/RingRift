"""DEPRECATED: Host Health Policy - Use cluster.health instead.

.. deprecated:: December 2025
   This module is deprecated as part of the coordination module consolidation
   (67 modules → 15). Health policy functionality is now in cluster.health.

Migration Guide:
   OLD:
      from app.coordination.host_health_policy import (
          check_host_health,
          is_host_healthy,
          get_healthy_hosts,
      )

   NEW:
      from app.coordination.cluster.health import (
          check_host_health,
          is_host_healthy,
          get_healthy_hosts,
      )

   # The imports are the same, but go through the unified health module now

The unified cluster.health module is the canonical source for all health
checks, providing better integration with circuit breakers and recovery.

This wrapper will be removed in Q2 2026.
"""

import warnings

warnings.warn(
    "Importing from host_health_policy is deprecated and will be removed in Q2 2026. "
    "Use 'from app.coordination.cluster.health import check_host_health' instead. "
    "The API is identical but the unified module provides better integration.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from cluster health (which itself imports from host_health_policy)
# This creates a proper deprecation path while maintaining functionality
from app.coordination.cluster.health import (
    HealthStatus,
    check_host_health,
    is_host_healthy,
    get_healthy_hosts,
    clear_health_cache,
    get_health_summary,
    is_cluster_healthy,
    check_cluster_health,
)

__all__ = [
    "HealthStatus",
    "check_host_health",
    "is_host_healthy",
    "get_healthy_hosts",
    "clear_health_cache",
    "get_health_summary",
    "is_cluster_healthy",
    "check_cluster_health",
]
