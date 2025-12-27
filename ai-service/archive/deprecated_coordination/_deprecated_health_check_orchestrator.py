"""DEPRECATED: Health Check Orchestrator - Use cluster.health instead.

.. deprecated:: December 2025
   This module is deprecated as part of the coordination module consolidation
   (67 modules → 15). All health functionality has been unified.

Migration Guide:
   OLD:
      from app.coordination.health_check_orchestrator import (
          HealthCheckOrchestrator,
          get_health_orchestrator,
      )

   NEW:
      from app.coordination.cluster.health import (
          UnifiedHealthManager,
          get_health_manager,
      )

The unified health manager provides the same functionality with better
integration across health monitoring, error recovery, and circuit breakers.

This wrapper will be removed in Q2 2026.
"""

import warnings

warnings.warn(
    "health_check_orchestrator is deprecated and will be removed in Q2 2026. "
    "Use 'from app.coordination.cluster.health import UnifiedHealthManager, get_health_manager' instead. "
    "See module docstring for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from unified health manager
from app.coordination.unified_health_manager import (
    UnifiedHealthManager as HealthCheckOrchestrator,
    get_health_manager as get_health_orchestrator,
    wire_health_events,
)

__all__ = [
    "HealthCheckOrchestrator",
    "get_health_orchestrator",
    "wire_health_events",
]
