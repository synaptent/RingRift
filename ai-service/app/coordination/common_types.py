"""Canonical type definitions for coordination modules.

This module provides the single source of truth for commonly-used types
across the coordination layer. It re-exports canonical types from contracts.py
and adds additional commonly-used types.

All coordination modules should import shared types from here to avoid
circular dependencies and duplication.

Created: December 29, 2025

Usage:
    from app.coordination.common_types import (
        HealthCheckResult,
        CoordinatorStatus,
        SyncResult,
        TransportError,
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Re-export canonical types from contracts.py (the primary source)
# This provides a consistent import path while maintaining backward compatibility
from app.coordination.contracts import (
    CoordinatorStatus,
    HealthCheckResult,
)

__all__ = [
    # From contracts.py
    "CoordinatorStatus",
    "HealthCheckResult",
    # Additional types defined here
    "SyncPriority",
    "SyncResult",
    "TransportState",
    "TransportConfig",
    "TransportError",
]


# =============================================================================
# SYNC PRIORITY ENUM
# =============================================================================


class SyncPriority(Enum):
    """Priority level for sync operations."""

    LOW = 1
    NORMAL = 5
    HIGH = 10
    URGENT = 20
    CRITICAL = 50


# =============================================================================
# SYNC TYPES
# =============================================================================


@dataclass
class SyncResult:
    """Result of a sync operation.

    Attributes:
        success: Whether the sync completed successfully.
        files_synced: Number of files transferred.
        bytes_transferred: Total bytes transferred.
        duration_seconds: How long the sync took.
        errors: List of error messages if any failures occurred.
        source: Source of the sync (node name or path).
        destination: Destination of the sync.
    """

    success: bool
    files_synced: int = 0
    bytes_transferred: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    source: str = ""
    destination: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "files_synced": self.files_synced,
            "bytes_transferred": self.bytes_transferred,
            "duration_seconds": self.duration_seconds,
            "errors": self.errors,
            "source": self.source,
            "destination": self.destination,
        }

    @property
    def has_errors(self) -> bool:
        """Check if any errors occurred."""
        return len(self.errors) > 0


# =============================================================================
# TRANSPORT TYPES
# =============================================================================


class TransportState(Enum):
    """State of a transport layer (for circuit breaker pattern)."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit tripped, fast-fail
    HALF_OPEN = "half_open"  # Testing if recovery is possible


@dataclass
class TransportConfig:
    """Configuration for transport layer.

    Attributes:
        timeout_seconds: Request timeout.
        max_retries: Maximum retry attempts.
        retry_delay_seconds: Delay between retries.
        circuit_breaker_threshold: Failures before tripping circuit.
        circuit_breaker_timeout_seconds: Time before attempting reset.
    """

    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout_seconds: float = 60.0


class TransportError(Exception):
    """Error during transport operation.

    Attributes:
        message: Error description.
        transport_type: Type of transport that failed.
        is_retryable: Whether the operation can be retried.
        details: Additional error context.
    """

    def __init__(
        self,
        message: str,
        transport_type: str = "unknown",
        is_retryable: bool = True,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.transport_type = transport_type
        self.is_retryable = is_retryable
        self.details = details or {}

    def __str__(self) -> str:
        return f"TransportError({self.transport_type}): {self.message}"


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# These aliases ensure existing code continues to work during migration
# They can be removed after Q2 2026

# For code that uses string status instead of enum
RUNNING = CoordinatorStatus.RUNNING
STOPPED = CoordinatorStatus.STOPPED
ERROR = CoordinatorStatus.ERROR
