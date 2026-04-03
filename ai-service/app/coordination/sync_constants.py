"""Shared Sync Constants Module (Phase 5 - December 2025).

Single source of truth for sync-related enums and dataclasses.
Previously duplicated across 6-9 files:
- SyncState: sync_coordination_core, sync_base, gossip_sync, registry_sync_manager, elo_sync_manager
- SyncPriority: sync_coordination_core, sync_coordinator
- SyncResult: cluster_data_sync, sync_bandwidth, p2p_sync_client, aria2_transport, sync_orchestrator

Usage:
    from app.coordination.sync_constants import SyncState, SyncPriority, SyncResult

Migration:
    Files should import from this module instead of defining their own.
    Existing definitions will be deprecated with runtime warnings in future.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any

__all__ = [
    "SyncState",
    "SyncPriority",
    "SyncResult",
    "SyncTarget",
    "SyncDirection",
]


class SyncState(Enum):
    """State of a sync operation.

    Standard lifecycle: PENDING -> IN_PROGRESS -> COMPLETED/FAILED/CANCELLED
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"  # For operations intentionally not performed

    @classmethod
    def terminal_states(cls) -> set["SyncState"]:
        """Return set of terminal states (no further transitions possible)."""
        return {cls.COMPLETED, cls.FAILED, cls.CANCELLED, cls.SKIPPED}

    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self in self.terminal_states()


class SyncPriority(IntEnum):
    """Priority levels for sync operations.

    Higher values = higher priority. Used for queue ordering.
    """

    CRITICAL = 100  # Training about to start, blocking
    HIGH = 75  # Training node needs data urgently
    NORMAL = 50  # Regular replication
    LOW = 25  # Background replication
    BACKGROUND = 10  # Opportunistic sync, lowest priority


class SyncDirection(Enum):
    """Direction of data flow in a sync operation."""

    PUSH = "push"  # Local -> Remote
    PULL = "pull"  # Remote -> Local
    BIDIRECTIONAL = "bidirectional"  # Both directions


@dataclass
class SyncTarget:
    """Target specification for a sync operation."""

    host: str
    path: str
    port: int = 22  # SSH port
    user: str = "ubuntu"
    ssh_key: str | None = None

    @property
    def ssh_spec(self) -> str:
        """Return SSH connection spec (user@host)."""
        return f"{self.user}@{self.host}"

    @property
    def rsync_spec(self) -> str:
        """Return rsync-compatible destination spec."""
        return f"{self.ssh_spec}:{self.path}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "host": self.host,
            "path": self.path,
            "port": self.port,
            "user": self.user,
            "ssh_key": self.ssh_key,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SyncTarget":
        """Create from dictionary."""
        return cls(
            host=data.get("host", ""),
            path=data.get("path", ""),
            port=data.get("port", 22),
            user=data.get("user", "ubuntu"),
            ssh_key=data.get("ssh_key"),
        )


@dataclass
class SyncResult:
    """Result of a sync operation.

    Comprehensive result type capturing all sync outcomes.
    Can be serialized for event emission and logging.
    """

    success: bool
    source: str = ""
    dest: str = ""
    host: str = ""

    # Transfer metrics
    bytes_transferred: int = 0
    files_synced: int = 0
    duration_seconds: float = 0.0

    # Bandwidth info
    bwlimit_kbps: int = 0
    effective_rate_kbps: float = 0.0

    # Error handling
    error: str | None = None
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""

    # State tracking
    state: SyncState = SyncState.COMPLETED
    priority: SyncPriority = SyncPriority.NORMAL
    request_id: str = ""

    # Timing
    started_at: float = 0.0
    completed_at: float = 0.0

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "source": self.source,
            "dest": self.dest,
            "host": self.host,
            "bytes_transferred": self.bytes_transferred,
            "files_synced": self.files_synced,
            "duration_seconds": self.duration_seconds,
            "bwlimit_kbps": self.bwlimit_kbps,
            "effective_rate_kbps": self.effective_rate_kbps,
            "error": self.error,
            "exit_code": self.exit_code,
            "state": self.state.value if isinstance(self.state, SyncState) else self.state,
            "priority": self.priority.value if isinstance(self.priority, SyncPriority) else self.priority,
            "request_id": self.request_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SyncResult":
        """Create from dictionary."""
        state = data.get("state", "completed")
        if isinstance(state, str):
            state = SyncState(state)

        priority = data.get("priority", 50)
        if isinstance(priority, int):
            # Find matching priority by value
            for p in SyncPriority:
                if p.value == priority:
                    priority = p
                    break
            else:
                priority = SyncPriority.NORMAL

        return cls(
            success=data.get("success", False),
            source=data.get("source", ""),
            dest=data.get("dest", ""),
            host=data.get("host", ""),
            bytes_transferred=data.get("bytes_transferred", 0),
            files_synced=data.get("files_synced", 0),
            duration_seconds=data.get("duration_seconds", 0.0),
            bwlimit_kbps=data.get("bwlimit_kbps", 0),
            effective_rate_kbps=data.get("effective_rate_kbps", 0.0),
            error=data.get("error"),
            exit_code=data.get("exit_code", 0),
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            state=state,
            priority=priority,
            request_id=data.get("request_id", ""),
            started_at=data.get("started_at", 0.0),
            completed_at=data.get("completed_at", 0.0),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def success_result(
        cls,
        source: str,
        dest: str,
        bytes_transferred: int = 0,
        files_synced: int = 0,
        duration_seconds: float = 0.0,
        **kwargs: Any,
    ) -> "SyncResult":
        """Create a successful sync result."""
        return cls(
            success=True,
            source=source,
            dest=dest,
            bytes_transferred=bytes_transferred,
            files_synced=files_synced,
            duration_seconds=duration_seconds,
            state=SyncState.COMPLETED,
            **kwargs,
        )

    @classmethod
    def failure_result(
        cls,
        source: str,
        dest: str,
        error: str,
        exit_code: int = 1,
        **kwargs: Any,
    ) -> "SyncResult":
        """Create a failed sync result."""
        return cls(
            success=False,
            source=source,
            dest=dest,
            error=error,
            exit_code=exit_code,
            state=SyncState.FAILED,
            **kwargs,
        )
