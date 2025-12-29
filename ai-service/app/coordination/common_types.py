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
    )

Note on TransportConfig:
    There are multiple TransportConfig classes for different purposes:
    - cluster_transport.TransportConfig: SSH/HTTP circuit breaker config
    - transport_manager.TransportConfig: File transfer routing config
    - storage_provider.TransportConfig: Multi-protocol sync config

    Import from the specific module that matches your use case.
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
