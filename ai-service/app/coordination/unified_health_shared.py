"""Shared enums, dataclasses, and event hooks for health coordination."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.coordination.types import ErrorSeverity
from app.distributed.circuit_breaker import CircuitState

try:
    from app.coordination.event_router import emit_node_overloaded

    HAS_NODE_EVENTS = True
except ImportError:
    emit_node_overloaded = None
    HAS_NODE_EVENTS = False


class SystemHealthLevel(Enum):
    """System health levels for aggregate scoring."""

    HEALTHY = "healthy"  # 80-100
    DEGRADED = "degraded"  # 60-79
    UNHEALTHY = "unhealthy"  # 40-59
    CRITICAL = "critical"  # 0-39


class RecoveryStatus(Enum):
    """Recovery attempt status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class JobRecoveryAction(str, Enum):
    """Types of job-level recovery actions."""

    RESTART_JOB = "restart_job"
    KILL_JOB = "kill_job"
    RESTART_NODE_SERVICES = "restart_node_services"
    REBOOT_NODE = "reboot_node"
    REMOVE_NODE = "remove_node"
    ESCALATE_HUMAN = "escalate_human"
    NONE = "none"


RecoveryAction = JobRecoveryAction


class RecoveryResult(str, Enum):
    """Result of a recovery attempt."""

    SUCCESS = "success"
    FAILED = "failed"
    ESCALATED = "escalated"
    SKIPPED = "skipped"


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""

    error_id: str
    component: str
    error_type: str
    message: str
    node_id: str = ""
    severity: ErrorSeverity = ErrorSeverity.ERROR
    timestamp: float = field(default_factory=time.time)
    context: dict[str, Any] = field(default_factory=dict)
    recovered: bool = False
    recovery_time: float = 0.0


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt."""

    recovery_id: str
    error_id: str
    component: str
    node_id: str
    strategy: str
    status: RecoveryStatus = RecoveryStatus.PENDING
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    success: bool = False
    message: str = ""
    attempt_number: int = 1

    @property
    def duration(self) -> float:
        """Get recovery duration in seconds."""
        if self.completed_at > 0:
            return self.completed_at - self.started_at
        return time.time() - self.started_at


@dataclass
class NodeRecoveryState:
    """Track recovery state for a node."""

    node_id: str
    is_online: bool = True
    recovery_attempts: int = 0
    last_attempt_time: float = 0.0
    consecutive_failures: int = 0
    is_escalated: bool = False
    last_escalation_time: float = 0.0
    offline_since: float = 0.0
    last_heartbeat: float = 0.0
    last_health_update: float = 0.0

    @property
    def is_healthy(self) -> bool:
        """Alias for is_online (backward compat)."""
        return self.is_online

    @is_healthy.setter
    def is_healthy(self, value: bool) -> None:
        self.is_online = value

    @property
    def is_responsive(self) -> bool:
        """Node is responsive if online and recently sent heartbeat."""
        if not self.is_online:
            return False
        if self.last_heartbeat == 0.0:
            return True
        return (time.time() - self.last_heartbeat) < 120.0

    @is_responsive.setter
    def is_responsive(self, value: bool) -> None:
        if not value:
            self.is_online = False

    @property
    def failure_count(self) -> int:
        """Alias for consecutive_failures (backward compat)."""
        return self.consecutive_failures

    @failure_count.setter
    def failure_count(self, value: int) -> None:
        self.consecutive_failures = value


NodeHealthState = NodeRecoveryState


@dataclass
class SystemHealthConfig:
    """Configuration for system health monitoring."""

    check_interval_seconds: int = 30
    healthy_threshold: int = 80
    degraded_threshold: int = 60
    unhealthy_threshold: int = 40
    pause_health_threshold: int = 40
    pause_node_offline_percent: float = 0.5
    pause_error_burst_count: int = 10
    pause_error_burst_window: int = 300
    critical_circuits: list[str] = field(
        default_factory=lambda: ["training", "evaluation", "promotion"]
    )
    resume_health_threshold: int = 60
    resume_delay_seconds: int = 120
    expected_nodes: int = 0
    node_weight: float = 0.40
    circuit_weight: float = 0.25
    error_weight: float = 0.20
    recovery_weight: float = 0.15


@dataclass
class SystemHealthScore:
    """Aggregate system health score."""

    score: int
    level: SystemHealthLevel
    components: dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    node_availability: float = 100.0
    circuit_health: float = 100.0
    error_rate: float = 100.0
    recovery_success: float = 100.0
    pause_triggers: list[str] = field(default_factory=list)


@dataclass
class HealthStats:
    """Aggregate health statistics."""

    total_errors: int = 0
    errors_by_severity: dict[str, int] = field(default_factory=dict)
    errors_by_component: dict[str, int] = field(default_factory=dict)
    errors_by_node: dict[str, int] = field(default_factory=dict)
    recovery_attempts: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    recovery_rate: float = 0.0
    avg_recovery_time: float = 0.0
    circuit_breakers_open: int = 0
    open_circuits: list[str] = field(default_factory=list)
    nodes_tracked: int = 0
    nodes_online: int = 0
    nodes_offline: int = 0
    escalated_nodes: list[str] = field(default_factory=list)
    jobs_tracked: int = 0


__all__ = [
    "CircuitState",
    "ErrorRecord",
    "ErrorSeverity",
    "HAS_NODE_EVENTS",
    "HealthStats",
    "JobRecoveryAction",
    "NodeHealthState",
    "NodeRecoveryState",
    "RecoveryAction",
    "RecoveryAttempt",
    "RecoveryResult",
    "RecoveryStatus",
    "SystemHealthConfig",
    "SystemHealthLevel",
    "SystemHealthScore",
    "emit_node_overloaded",
]
