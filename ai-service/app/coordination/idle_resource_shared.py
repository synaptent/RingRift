"""Shared idle-resource daemon types and optional integrations."""

from __future__ import annotations

from dataclasses import dataclass

# SSH fallback for node discovery when P2P is unavailable (Dec 2025)
try:
    from app.config.cluster_config import ClusterNode, get_cluster_nodes as get_configured_hosts
    from app.execution.executor import SSHExecutor

    HAS_SSH_FALLBACK = True
except ImportError:
    HAS_SSH_FALLBACK = False
    get_configured_hosts = None
    ClusterNode = None
    SSHExecutor = None

# Job scheduler integration (Phase 21.2 - Dec 2025)
try:
    from app.coordination.job_scheduler import (
        JobPriority,
        PriorityJobScheduler,
        ScheduledJob,
        get_scheduler,
    )

    HAS_JOB_SCHEDULER = True
except ImportError:
    HAS_JOB_SCHEDULER = False
    get_scheduler = None
    PriorityJobScheduler = None
    ScheduledJob = None
    JobPriority = None

# Circuit breaker integration (Phase 4 - December 2025)
try:
    from app.distributed.circuit_breaker import get_operation_breaker

    HAS_CIRCUIT_BREAKER = True
except ImportError:
    HAS_CIRCUIT_BREAKER = False
    get_operation_breaker = None

# Unified backpressure monitoring (Phase 21.5 - December 2025)
try:
    from app.coordination.backpressure import BackpressureMonitor, get_backpressure_monitor

    HAS_BACKPRESSURE = True
except ImportError:
    HAS_BACKPRESSURE = False
    BackpressureMonitor = None
    get_backpressure_monitor = None

# Job stall detection (Phase 21.5 - December 2025)
try:
    from app.coordination.stall_detection import JobStallDetector, get_stall_detector

    HAS_STALL_DETECTION = True
except ImportError:
    HAS_STALL_DETECTION = False
    JobStallDetector = None
    get_stall_detector = None

# Event emission for node incompatibility
try:
    from app.distributed.data_events import emit_node_incompatible_with_workload

    HAS_INCOMPATIBILITY_EVENTS = True
except ImportError:
    HAS_INCOMPATIBILITY_EVENTS = False
    emit_node_incompatible_with_workload = None


@dataclass
class NodeStatus:
    """Status of a cluster node for idle-resource tracking."""

    node_id: str
    host: str
    gpu_utilization: float = 0.0
    gpu_memory_total_gb: float = 0.0
    gpu_memory_used_gb: float = 0.0
    last_seen: float = 0.0
    idle_since: float = 0.0
    active_jobs: int = 0
    provider: str = "unknown"


__all__ = [
    "BackpressureMonitor",
    "ClusterNode",
    "HAS_BACKPRESSURE",
    "HAS_CIRCUIT_BREAKER",
    "HAS_INCOMPATIBILITY_EVENTS",
    "HAS_JOB_SCHEDULER",
    "HAS_SSH_FALLBACK",
    "HAS_STALL_DETECTION",
    "JobPriority",
    "JobStallDetector",
    "NodeStatus",
    "PriorityJobScheduler",
    "SSHExecutor",
    "ScheduledJob",
    "emit_node_incompatible_with_workload",
    "get_backpressure_monitor",
    "get_configured_hosts",
    "get_operation_breaker",
    "get_scheduler",
    "get_stall_detector",
]
