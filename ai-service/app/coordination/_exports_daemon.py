"""Daemon management exports for coordination package.

December 2025: Extracted from __init__.py to improve maintainability.
This module consolidates all daemon lifecycle and management imports.
"""

# DaemonManager - unified lifecycle management for all background services
from app.coordination.daemon_manager import (
    DaemonInfo,
    DaemonManager,
    DaemonManagerConfig,
    DaemonState,
    DaemonType,
    get_daemon_manager,
    reset_daemon_manager,
    setup_signal_handlers,
)

# DaemonLifecycle - lifecycle operations and dependency validation
from app.coordination.daemon_lifecycle import (
    DaemonLifecycleManager,
    DependencyValidationError,
)

# DaemonFactory - centralized lazy daemon creation
from app.coordination.daemon_factory import (
    DaemonFactory,
    DaemonSpec,
    get_daemon_factory,
    reset_daemon_factory,
)

# BaseDaemon - abstract base class for all cluster daemons
from app.coordination.base_daemon import (
    BaseDaemon,
    DaemonConfig,
)

# UnifiedIdleShutdownDaemon - provider-agnostic idle shutdown (December 2025)
# Consolidates lambda_idle_daemon and vast_idle_daemon
from app.coordination.unified_idle_shutdown_daemon import (
    IdleShutdownConfig,
    NodeIdleStatus,
    UnifiedIdleShutdownDaemon,
    create_lambda_idle_daemon,
    create_runpod_idle_daemon,
    create_vast_idle_daemon,
)

# Backward compatibility aliases for idle daemons
from app.coordination.unified_idle_shutdown_daemon import (
    IdleShutdownConfig as LambdaIdleConfig,
    IdleShutdownConfig as VastIdleConfig,
    NodeIdleStatus as LambdaNodeStatus,
    NodeIdleStatus as VastNodeStatus,
    UnifiedIdleShutdownDaemon as LambdaIdleDaemon,
    UnifiedIdleShutdownDaemon as VastIdleDaemon,
)

__all__ = [
    # Core Daemon Management
    "BaseDaemon",
    "DaemonConfig",
    "DaemonFactory",
    "DaemonInfo",
    "DaemonLifecycleManager",
    "DaemonManager",
    "DaemonManagerConfig",
    "DaemonSpec",
    "DaemonState",
    "DaemonType",
    "DependencyValidationError",
    "get_daemon_factory",
    "get_daemon_manager",
    "reset_daemon_factory",
    "reset_daemon_manager",
    "setup_signal_handlers",
    # Unified Idle Shutdown
    "IdleShutdownConfig",
    "NodeIdleStatus",
    "UnifiedIdleShutdownDaemon",
    "create_lambda_idle_daemon",
    "create_runpod_idle_daemon",
    "create_vast_idle_daemon",
    # Backward Compatibility Aliases
    "LambdaIdleConfig",
    "LambdaIdleDaemon",
    "LambdaNodeStatus",
    "VastIdleConfig",
    "VastIdleDaemon",
    "VastNodeStatus",
]
