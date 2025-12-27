"""Consolidated Coordinator Base Classes (December 2025).

This module consolidates coordinator base classes and dependency management:
- Base classes: CoordinatorBase, mixins (SQLitePersistence, Singleton, Callback, EventDriven)
- Registry: CoordinatorRegistry for tracking and graceful shutdown
- Dependencies: CoordinatorDependencyGraph for initialization ordering

This is part of the 157→15 module consolidation (Phase 5).

Migration Guide:
    # Old imports (deprecated, still work):
    from app.coordination.coordinator_base import (
        CoordinatorBase, CoordinatorStatus, CoordinatorProtocol,
        SQLitePersistenceMixin, SingletonMixin, CallbackMixin,
        CoordinatorRegistry, get_coordinator_registry,
    )
    from app.coordination.coordinator_dependencies import (
        CoordinatorDependencyGraph, validate_dependencies,
        get_initialization_order, COORDINATOR_REGISTRY,
    )

    # New imports (preferred):
    from app.coordination.core_base import (
        # Base classes
        CoordinatorBase, CoordinatorStats,

        # Protocols and enums
        CoordinatorProtocol, CoordinatorStatus,

        # Mixins
        SQLitePersistenceMixin, SingletonMixin,
        CallbackMixin, EventDrivenMonitorMixin,

        # Registry
        CoordinatorRegistry, get_coordinator_registry,
        get_all_coordinators, get_coordinator_statuses,
        shutdown_all_coordinators,

        # Dependencies
        CoordinatorDependency, CoordinatorDependencyGraph,
        COORDINATOR_REGISTRY as DEPENDENCY_REGISTRY,
        get_dependency_graph, reset_dependency_graph,
        validate_dependencies, get_initialization_order,
        check_event_chain_cycles,

        # Utility
        is_coordinator,
    )
"""

from __future__ import annotations

# =============================================================================
# Re-exports from coordinator_base.py
# =============================================================================

from app.coordination.coordinator_base import (
    # Mixins
    CallbackMixin,
    # Base classes
    CoordinatorBase,
    # Protocols (re-exported from protocols.py via coordinator_base)
    CoordinatorProtocol,
    # Registry
    CoordinatorRegistry,
    CoordinatorStats,
    # Enums (re-exported from protocols.py via coordinator_base)
    CoordinatorStatus,
    EventDrivenMonitorMixin,
    SQLitePersistenceMixin,
    SingletonMixin,
    get_all_coordinators,
    get_coordinator_registry,
    get_coordinator_statuses,
    # Utility functions
    is_coordinator,
    shutdown_all_coordinators,
)

# Also re-export HealthCheckResult from protocols (commonly used with health_check())
from app.coordination.protocols import HealthCheckResult

# =============================================================================
# Re-exports from coordinator_dependencies.py
# =============================================================================

from app.coordination.coordinator_dependencies import (
    COORDINATOR_REGISTRY,
    CoordinatorDependency,
    CoordinatorDependencyGraph,
    check_event_chain_cycles,
    get_dependency_graph,
    get_initialization_order,
    reset_dependency_graph,
    validate_dependencies,
)

# =============================================================================
# Alias for clarity
# =============================================================================

# COORDINATOR_REGISTRY in coordinator_dependencies.py contains dependency info,
# while CoordinatorRegistry is the runtime registry. Use this alias to distinguish.
DEPENDENCY_REGISTRY = COORDINATOR_REGISTRY

# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Base classes
    "CoordinatorBase",
    "CoordinatorStats",
    # Protocols and enums
    "CoordinatorProtocol",
    "CoordinatorStatus",
    "HealthCheckResult",
    # Mixins
    "SQLitePersistenceMixin",
    "SingletonMixin",
    "CallbackMixin",
    "EventDrivenMonitorMixin",
    # Runtime Registry
    "CoordinatorRegistry",
    "get_coordinator_registry",
    "get_all_coordinators",
    "get_coordinator_statuses",
    "shutdown_all_coordinators",
    # Dependency Management
    "CoordinatorDependency",
    "CoordinatorDependencyGraph",
    "COORDINATOR_REGISTRY",  # Raw dependency registry dict
    "DEPENDENCY_REGISTRY",  # Alias for clarity
    "get_dependency_graph",
    "reset_dependency_graph",
    "validate_dependencies",
    "get_initialization_order",
    "check_event_chain_cycles",
    # Utility
    "is_coordinator",
]
