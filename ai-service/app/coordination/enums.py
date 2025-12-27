"""Centralized enum definitions for coordination module.

This module provides a single import path for all coordination enums,
ensuring consistent usage across the codebase and avoiding naming collisions.

Usage:
    from app.coordination.enums import (
        # Leadership/cluster role enums
        LeadershipRole,        # Raft-like roles: LEADER, FOLLOWER, CANDIDATE, OFFLINE
        ClusterNodeRole,       # Job roles: TRAINING, SELFPLAY, COORDINATOR, IDLE, OFFLINE

        # Recovery action enums (different scopes)
        JobRecoveryAction,     # Job-level: RESTART_JOB, KILL_JOB, etc.
        SystemRecoveryAction,  # System-level: RESTART_P2P, SOFT_REBOOT, etc.
        NodeRecoveryAction,    # Node-level: RESTART, FAILOVER, etc.

        # Daemon management
        DaemonType,            # All daemon types (60+)
        DaemonState,           # Daemon states: STOPPED, STARTING, RUNNING, etc.

        # Health and status
        ErrorSeverity,         # ERROR, WARNING, CRITICAL, etc.
        RecoveryStatus,        # PENDING, IN_PROGRESS, COMPLETED, FAILED
        RecoveryResult,        # SUCCESS, FAILED, ESCALATED, SKIPPED

        # Data events
        DataEventType,         # All data pipeline events
    )

NOTE (Dec 2025): This module was created during consolidation to avoid
the naming collision bugs where NodeRole and RecoveryAction had multiple
conflicting definitions across different modules.
"""

# Leadership/cluster role enums
from app.coordination.leadership_coordinator import LeadershipRole
from app.coordination.multi_provider_orchestrator import ClusterNodeRole

# Recovery action enums - three different scopes
from app.coordination.unified_health_manager import (
    JobRecoveryAction,
    ErrorSeverity,
    RecoveryStatus,
    RecoveryResult,
)
from app.coordination.recovery_orchestrator import SystemRecoveryAction
from app.coordination.node_recovery_daemon import NodeRecoveryAction

# Daemon management
from app.coordination.daemon_types import DaemonType, DaemonState

# Health states (canonical location: node_status.py)
from app.coordination.node_status import NodeHealthState

# Data events
from app.distributed.data_events import DataEventType

# Backward-compat aliases with deprecation warnings
# These aliases allow existing code to import from enums.py but will
# emit warnings to encourage migration to specific names

import warnings as _warnings


def _deprecated_alias(name: str, replacement: str):
    """Create a deprecated alias property."""
    _warnings.warn(
        f"{name} is deprecated and ambiguous. Use {replacement} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


# Re-export the deprecated alias names for backward compatibility
# NOTE: Import directly from source modules for the aliases
from app.coordination.leadership_coordinator import NodeRole as _LeadershipNodeRole
from app.coordination.multi_provider_orchestrator import NodeRole as _ClusterNodeRole

# Document all exports
__all__ = [
    # Primary exports (use these)
    "LeadershipRole",
    "ClusterNodeRole",
    "JobRecoveryAction",
    "SystemRecoveryAction",
    "NodeRecoveryAction",
    "DaemonType",
    "DaemonState",
    "NodeHealthState",
    "ErrorSeverity",
    "RecoveryStatus",
    "RecoveryResult",
    "DataEventType",
]
