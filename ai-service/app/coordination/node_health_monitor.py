"""Backward compatibility re-export for deprecated node_health_monitor.

.. deprecated:: December 2025
    This module is deprecated. Use :mod:`app.coordination.health_check_orchestrator`
    instead. This file will be removed in Q2 2026.

Migration:
    # OLD (deprecated)
    from app.coordination.node_health_monitor import get_node_health_monitor

    # NEW (recommended)
    from app.coordination.health_check_orchestrator import get_health_orchestrator
    orchestrator = get_health_orchestrator()
    health = orchestrator.get_node_health("node-1")
"""

from __future__ import annotations

import warnings

warnings.warn(
    "node_health_monitor module is deprecated as of December 2025. "
    "Use health_check_orchestrator module instead. "
    "This module will be removed in Q2 2026.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from health_check_orchestrator for backward compatibility
from app.coordination.health_check_orchestrator import (
    HealthCheckOrchestrator,
    get_health_orchestrator,
)

# Alias for backward compatibility
NodeHealthMonitor = HealthCheckOrchestrator
get_node_health_monitor = get_health_orchestrator

__all__ = [
    "NodeHealthMonitor",
    "get_node_health_monitor",
    "HealthCheckOrchestrator",
    "get_health_orchestrator",
]
