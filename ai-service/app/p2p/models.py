"""P2P Orchestrator Data Models.

Core data structures for node and job management in the P2P cluster.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class NodeRole(str, Enum):
    """Role a node plays in the cluster."""
    LEADER = "leader"
    FOLLOWER = "follower"
    CANDIDATE = "candidate"


class JobType(str, Enum):
    """Types of jobs nodes can run."""
    SELFPLAY = "selfplay"
    GPU_SELFPLAY = "gpu_selfplay"
    HYBRID_SELFPLAY = "hybrid_selfplay"
    TRAINING = "training"
    CMAES = "cmaes"
    DISTRIBUTED_CMAES_COORDINATOR = "distributed_cmaes_coordinator"
    DISTRIBUTED_CMAES_WORKER = "distributed_cmaes_worker"
    DISTRIBUTED_TOURNAMENT_COORDINATOR = "distributed_tournament_coordinator"
    DISTRIBUTED_TOURNAMENT_WORKER = "distributed_tournament_worker"
    IMPROVEMENT_LOOP = "improvement_loop"


class JobStatus(str, Enum):
    """Status of a running job."""
    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    COMPLETED = "completed"


class NodeHealth(str, Enum):
    """Health status of a node."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    RETIRED = "retired"


@dataclass
class ResourceMetrics:
    """Resource utilization metrics for a node."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_gb_available: float = 0.0
    disk_percent: float = 0.0
    disk_gb_free: float = 0.0
    gpu_utilization: float = 0.0
    gpu_memory_percent: float = 0.0
    load_average_1m: float = 0.0
    load_average_5m: float = 0.0
    load_average_15m: float = 0.0
    timestamp: float = 0.0

    @property
    def load_score(self) -> float:
        """Calculate composite load score (0-100)."""
        return max(
            self.cpu_percent,
            self.memory_percent,
            self.load_average_1m * 10,  # Normalize load average
        )

    @property
    def is_overloaded(self) -> bool:
        """Check if node is overloaded (80% max utilization enforced 2025-12-16)."""
        return self.load_score > 80 or self.memory_percent > 80

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_gb_available": self.memory_gb_available,
            "disk_percent": self.disk_percent,
            "disk_gb_free": self.disk_gb_free,
            "gpu_utilization": self.gpu_utilization,
            "gpu_memory_percent": self.gpu_memory_percent,
            "load_average_1m": self.load_average_1m,
            "load_score": self.load_score,
            "is_overloaded": self.is_overloaded,
        }


@dataclass
class NodeSummary:
    """Summary information about a cluster node."""
    node_id: str
    hostname: str
    ip_address: str
    port: int = 8770
    role: NodeRole = NodeRole.FOLLOWER
    health: NodeHealth = NodeHealth.HEALTHY
    resources: ResourceMetrics = field(default_factory=ResourceMetrics)

    # GPU info
    gpu_type: str = "Unknown"
    gpu_count: int = 0
    gpu_priority: int = 0

    # Job tracking
    active_jobs: int = 0
    selfplay_count: int = 0
    training_active: bool = False

    # Timing
    last_heartbeat: float = 0.0
    uptime_seconds: float = 0.0
    joined_at: float = 0.0

    @property
    def endpoint(self) -> str:
        """HTTP endpoint for this node."""
        return f"http://{self.ip_address}:{self.port}"

    @property
    def is_online(self) -> bool:
        """Check if node is online."""
        return self.health in (NodeHealth.HEALTHY, NodeHealth.DEGRADED)

    @property
    def can_accept_jobs(self) -> bool:
        """Check if node can accept new jobs."""
        return (
            self.is_online
            and not self.resources.is_overloaded
            and self.health != NodeHealth.DEGRADED
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "ip_address": self.ip_address,
            "port": self.port,
            "endpoint": self.endpoint,
            "role": self.role.value,
            "health": self.health.value,
            "resources": self.resources.to_dict(),
            "gpu_type": self.gpu_type,
            "gpu_count": self.gpu_count,
            "gpu_priority": self.gpu_priority,
            "active_jobs": self.active_jobs,
            "selfplay_count": self.selfplay_count,
            "training_active": self.training_active,
            "last_heartbeat": self.last_heartbeat,
            "uptime_seconds": self.uptime_seconds,
            "is_online": self.is_online,
            "can_accept_jobs": self.can_accept_jobs,
        }


@dataclass
class JobInfo:
    """Information about a running job."""
    job_id: str
    job_type: JobType
    status: JobStatus = JobStatus.PENDING
    node_id: Optional[str] = None

    # Configuration
    board_type: str = "square8"
    num_players: int = 2
    model_path: Optional[str] = None

    # Progress
    games_completed: int = 0
    games_target: int = 0
    progress_percent: float = 0.0

    # Timing
    created_at: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0

    # Error tracking
    error_message: Optional[str] = None
    retry_count: int = 0

    @property
    def is_active(self) -> bool:
        """Check if job is currently active."""
        return self.status in (JobStatus.PENDING, JobStatus.STARTING, JobStatus.RUNNING)

    @property
    def duration_seconds(self) -> float:
        """Duration of the job in seconds."""
        if self.started_at == 0:
            return 0.0
        end_time = self.completed_at if self.completed_at > 0 else datetime.now().timestamp()
        return end_time - self.started_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "job_type": self.job_type.value,
            "status": self.status.value,
            "node_id": self.node_id,
            "board_type": self.board_type,
            "num_players": self.num_players,
            "model_path": self.model_path,
            "games_completed": self.games_completed,
            "games_target": self.games_target,
            "progress_percent": self.progress_percent,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "duration_seconds": self.duration_seconds,
            "is_active": self.is_active,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
        }


@dataclass
class ClusterStatus:
    """Overall cluster status summary."""
    leader_id: Optional[str] = None
    total_nodes: int = 0
    online_nodes: int = 0
    healthy_nodes: int = 0

    total_jobs: int = 0
    active_jobs: int = 0
    pending_jobs: int = 0
    failed_jobs: int = 0

    total_selfplay: int = 0
    training_nodes: int = 0

    cluster_health: NodeHealth = NodeHealth.HEALTHY
    last_update: float = 0.0

    nodes: List[NodeSummary] = field(default_factory=list)
    jobs: List[JobInfo] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "leader_id": self.leader_id,
            "total_nodes": self.total_nodes,
            "online_nodes": self.online_nodes,
            "healthy_nodes": self.healthy_nodes,
            "total_jobs": self.total_jobs,
            "active_jobs": self.active_jobs,
            "pending_jobs": self.pending_jobs,
            "failed_jobs": self.failed_jobs,
            "total_selfplay": self.total_selfplay,
            "training_nodes": self.training_nodes,
            "cluster_health": self.cluster_health.value,
            "last_update": self.last_update,
        }
