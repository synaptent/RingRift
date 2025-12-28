"""DEPRECATED: System Health Monitor - Use unified_health_manager instead.

.. deprecated:: December 2025
   This module is deprecated as part of the coordination module consolidation.
   All health functionality has been unified into unified_health_manager.py.

Migration Guide:
   OLD:
      from app.coordination.system_health_monitor import (
          SystemHealthMonitorDaemon,
          get_system_health,
          is_pipeline_paused,
      )

   NEW:
      from app.coordination.unified_health_manager import (
          UnifiedHealthManager,
          get_health_manager,
          is_pipeline_paused,
      )

   For scoring functions:
      from app.coordination.unified_health_manager import (
          get_system_health_score,
          get_system_health_level,
          should_pause_pipeline,
      )

This wrapper will be removed in Q2 2026.
Full implementation preserved in archive/deprecated_coordination/_deprecated_system_health_monitor_full.py
"""

from __future__ import annotations

import warnings

warnings.warn(
    "system_health_monitor is deprecated and will be removed in Q2 2026. "
    "Use 'from app.coordination.unified_health_manager import get_health_manager' instead. "
    "See module docstring for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from unified_health_manager for backward compatibility
from app.coordination.unified_health_manager import (
    # Enums
    PipelineState,
    SystemHealthLevel,
    # Data classes
    SystemHealthConfig,
    SystemHealthScore,
    # Main class (alias)
    UnifiedHealthManager as SystemHealthMonitorDaemon,
    # Functions
    get_health_manager as get_system_health,
    get_system_health_level,
    get_system_health_score,
    get_system_health_score as get_health_score,
    is_pipeline_paused,
    reset_health_manager as reset_system_health_monitor,
    should_pause_pipeline,
)

__all__ = [
    # Enums
    "PipelineState",
    "SystemHealthLevel",
    # Data classes
    "SystemHealthConfig",
    "SystemHealthScore",
    # Main class
    "SystemHealthMonitorDaemon",
    # Functions
    "get_health_score",
    "get_system_health",
    "get_system_health_level",
    "get_system_health_score",
    "is_pipeline_paused",
    "reset_system_health_monitor",
    "should_pause_pipeline",
]
