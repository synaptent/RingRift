"""Core daemon infrastructure - consolidated re-exports.

Provides unified access to daemon management infrastructure:
- DaemonManager - lifecycle management, health monitoring, auto-restart
- DaemonFactory - daemon creation with lazy loading
- DaemonRegistry - declarative daemon configuration
- BaseDaemon - base class for all daemons
- DaemonType - enum of all daemon types

Usage:
    from app.coordination.core_daemons import (
        # Manager
        DaemonManager, get_daemon_manager, reset_daemon_manager,
        DaemonManagerConfig,

        # Types
        DaemonType, DaemonState, DaemonInfo,
        CRITICAL_DAEMONS, DAEMON_DEPENDENCIES, DAEMON_STARTUP_ORDER,

        # Base class
        BaseDaemon, DaemonConfig,

        # Factory
        DaemonFactory, get_daemon_factory, DaemonImportSpec,

        # Registry
        DaemonSpec, DAEMON_REGISTRY,
        get_daemons_by_category, get_categories, validate_registry,
    )

This module consolidates:
- daemon_types.py - DaemonType enum, DaemonState, DaemonInfo
- daemon_manager.py - DaemonManager, get_daemon_manager
- base_daemon.py - BaseDaemon, DaemonConfig
- daemon_factory.py - DaemonFactory, get_daemon_factory
- daemon_registry.py - DaemonSpec, DAEMON_REGISTRY
- daemon_lifecycle.py - Lifecycle utilities

Created December 2025 as part of 157->15 module consolidation.
"""

from __future__ import annotations

# =============================================================================
# Daemon Types (from daemon_types.py)
# =============================================================================
from app.coordination.daemon_types import (
    # Core types
    DaemonType,
    DaemonState,
    DaemonInfo,
    DaemonManagerConfig,
    # Constants
    CRITICAL_DAEMONS,
    DAEMON_DEPENDENCIES,
    DAEMON_STARTUP_ORDER,
    MAX_RESTART_DELAY,
    DAEMON_RESTART_RESET_AFTER,
    # Utilities
    get_daemon_startup_position,
    mark_daemon_ready,
    register_mark_ready_callback,
    validate_daemon_dependencies,
    validate_startup_order_consistency,
    validate_startup_order_or_raise,
)

# =============================================================================
# Base Daemon Class (from base_daemon.py)
# =============================================================================
from app.coordination.base_daemon import (
    BaseDaemon,
    DaemonConfig,
)

# =============================================================================
# Daemon Manager (from daemon_manager.py)
# =============================================================================
from app.coordination.daemon_manager import (
    DaemonManager,
    get_daemon_manager,
    reset_daemon_manager,
    setup_signal_handlers,
)

# =============================================================================
# Daemon Factory (from daemon_factory.py)
# =============================================================================
from app.coordination.daemon_factory import (
    DaemonFactory,
    DaemonImportSpec,
    get_daemon_factory,
    reset_daemon_factory,
)

# =============================================================================
# Daemon Registry (from daemon_registry.py)
# =============================================================================
from app.coordination.daemon_registry import (
    DaemonSpec,
    DAEMON_REGISTRY,
    get_daemons_by_category,
    get_categories,
    get_deprecated_daemons,
    is_daemon_deprecated,
    validate_registry,
)

# =============================================================================
# Daemon Lifecycle (from daemon_lifecycle.py)
# =============================================================================
from app.coordination.daemon_lifecycle import (
    DaemonLifecycleManager,
    DependencyValidationError,
    StateUpdateCallback,
)

# =============================================================================
# Daemon Runners (from daemon_runners.py)
# =============================================================================
from app.coordination.daemon_runners import (
    get_runner,
    get_all_runners,
)

# =============================================================================
# Daemon Adapters (from daemon_adapters.py) - for wrapping existing daemons
# =============================================================================
from app.coordination.daemon_adapters import (
    DaemonAdapter,
    DaemonAdapterConfig,
)

# =============================================================================
# Exports
# =============================================================================
__all__ = [
    # === Types ===
    "DaemonType",
    "DaemonState",
    "DaemonInfo",
    "DaemonManagerConfig",
    # === Constants ===
    "CRITICAL_DAEMONS",
    "DAEMON_DEPENDENCIES",
    "DAEMON_STARTUP_ORDER",
    "MAX_RESTART_DELAY",
    "DAEMON_RESTART_RESET_AFTER",
    # === Type utilities ===
    "get_daemon_startup_position",
    "mark_daemon_ready",
    "register_mark_ready_callback",
    "validate_daemon_dependencies",
    "validate_startup_order_consistency",
    "validate_startup_order_or_raise",
    # === Base class ===
    "BaseDaemon",
    "DaemonConfig",
    # === Manager ===
    "DaemonManager",
    "get_daemon_manager",
    "reset_daemon_manager",
    "setup_signal_handlers",
    # === Factory ===
    "DaemonFactory",
    "DaemonImportSpec",
    "get_daemon_factory",
    "reset_daemon_factory",
    # === Registry ===
    "DaemonSpec",
    "DAEMON_REGISTRY",
    "get_daemons_by_category",
    "get_categories",
    "get_deprecated_daemons",
    "is_daemon_deprecated",
    "validate_registry",
    # === Lifecycle ===
    "DaemonLifecycleManager",
    "DependencyValidationError",
    "StateUpdateCallback",
    # === Runners ===
    "get_runner",
    "get_all_runners",
    # === Adapters ===
    "DaemonAdapter",
    "DaemonAdapterConfig",
]
