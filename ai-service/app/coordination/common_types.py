"""Canonical type definitions for coordination modules.

This module provides a unified import point for commonly-used types
across the coordination layer. It re-exports types from their canonical
locations to provide a single import path.

All coordination modules should import shared types from here to avoid
circular dependencies and duplication.

Created: December 29, 2025

Usage:
    from app.coordination.common_types import (
        HealthCheckResult,
        CoordinatorStatus,
        SyncPriority,
        SyncResult,
        TransportState,
        TransportError,
        # Config classes
        TimeoutConfig,       # Circuit breaker timeouts
        TransportConfig,     # File transfer settings
        SyncProtocolConfig,  # Sync protocol enablement
    )

Config Classes (December 29, 2025 - Consolidated with unique names):

    - TimeoutConfig (cluster_transport.py):
      Connection and circuit breaker timeouts.
      Use for: connect_timeout, operation_timeout, failure_threshold.

    - TransportConfig (transport_manager.py):
      File transfer settings including bandwidth, SSH keys, retries.
      Use for: rsync_bandwidth_limit, ssh_key_path, max_retries.

    - SyncProtocolConfig (storage_provider.py):
      Which sync protocols to enable and their specific settings.
      Use for: enable_aria2, enable_gossip, enable_bittorrent, fallback_chain.
"""

from __future__ import annotations

# Re-export from contracts.py (coordinator types)
from app.coordination.contracts import (
    CoordinatorStatus,
    HealthCheckResult,
)

# Re-export from sync_constants.py (sync types)
from app.coordination.sync_constants import (
    SyncPriority,
    SyncResult,
)

# Re-export from transport_base.py (transport types)
from app.coordination.transport_base import (
    TransportError,
    TransportResult,
    TransportState,
)

# Re-export config classes with distinct names (December 29, 2025)
from app.coordination.cluster_transport import TimeoutConfig
from app.coordination.transport_manager import TransportConfig
from app.distributed.storage_provider import SyncProtocolConfig

__all__ = [
    # From contracts.py
    "CoordinatorStatus",
    "HealthCheckResult",
    # From sync_constants.py
    "SyncPriority",
    "SyncResult",
    # From transport_base.py
    "TransportState",
    "TransportResult",
    "TransportError",
    # Config classes (December 29, 2025 consolidation)
    "TimeoutConfig",  # Circuit breaker timeouts (cluster_transport)
    "TransportConfig",  # File transfer settings (transport_manager)
    "SyncProtocolConfig",  # Sync protocol enablement (storage_provider)
]


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# These aliases ensure existing code continues to work during migration
# They can be removed after Q2 2026

# For code that uses string status instead of enum
RUNNING = CoordinatorStatus.RUNNING
STOPPED = CoordinatorStatus.STOPPED
ERROR = CoordinatorStatus.ERROR
