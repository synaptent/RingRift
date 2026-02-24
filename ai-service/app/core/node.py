"""Unified NodeInfo dataclass for RingRift AI Service.

This module consolidates the various node information structures used across
the codebase into a single, comprehensive dataclass that supports:

- All fields from P2P orchestrator, cluster monitor, idle resource daemon
- Factory methods for creating from P2P status, SSH discovery, provider APIs
- Serialization to/from dict/JSON
- Nested dataclasses for GPU info, resources, health status, connection info

Usage:
    from app.core.node import NodeInfo, NodeHealth, GPUInfo

    # Create from P2P status response
    node = NodeInfo.from_p2p_status(peer_data)

    # Create from SSH discovery
    node = NodeInfo.from_ssh_discovery(hostname, nvidia_smi_output)

    # Serialize
    data = node.to_dict()

    # Deserialize
    node = NodeInfo.from_dict(data)

Migration from existing modules:
    - scripts/p2p/models.NodeInfo -> app.core.node.NodeInfo
    - app/coordination/idle_resource_daemon.NodeStatus -> app.core.node.NodeInfo
    - app/distributed/cluster_monitor.NodeStatus -> app.core.node.NodeInfo
    - app/p2p/models.NodeSummary -> app.core.node.NodeInfo
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar

# December 2025: Use centralized port constants
from app.config.ports import P2P_DEFAULT_PORT
from app.config.thresholds import DISK_SYNC_TARGET_PERCENT

# Jan 22, 2026: Import canonical PEER_TIMEOUT to prevent mismatch
# Previously hardcoded 90s here vs 120s in app/p2p/constants.py
try:
    from app.p2p.constants import PEER_TIMEOUT as _CANONICAL_PEER_TIMEOUT
except ImportError:
    _CANONICAL_PEER_TIMEOUT = 120.0

__all__ = [
    "NodeInfo",
    "NodeRole",
    "NodeHealth",
    "NodeState",
    "Provider",
    "GPUInfo",
    "ResourceMetrics",
    "ConnectionInfo",
    "HealthStatus",
    "ProviderInfo",
    "JobStatus",
    # P2P leader detection utilities
    "get_this_node_id",
    "check_p2p_leader_status",
    "get_is_leader_sync",
]


# =============================================================================
# Enums
# =============================================================================

class NodeRole(str, Enum):
    """Role a node plays in the cluster."""
    LEADER = "leader"
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    OFFLINE = "offline"


class NodeHealth(str, Enum):
    """Health status of a node."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    RETIRED = "retired"
    UNKNOWN = "unknown"


class NodeState(str, Enum):
    """Connection state of a node."""
    ONLINE = "online"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class Provider(str, Enum):
    """Cloud provider/infrastructure type."""
    LAMBDA = "lambda"
    VAST = "vast"
    RUNPOD = "runpod"
    NEBIUS = "nebius"
    VULTR = "vultr"
    HETZNER = "hetzner"
    AWS = "aws"
    GCP = "gcp"
    LOCAL = "local"
    UNKNOWN = "unknown"


# =============================================================================
# Nested Dataclasses
# =============================================================================

@dataclass
class GPUInfo:
    """GPU information for a node."""
    has_gpu: bool = False
    gpu_name: str = ""
    gpu_count: int = 0
    gpu_type: str = ""

    memory_total_gb: float = 0.0
    memory_used_gb: float = 0.0
    memory_free_gb: float = 0.0

    utilization_percent: float = 0.0
    memory_percent: float = 0.0
    power_score: int = 0

    GPU_POWER_RANKINGS: ClassVar[dict[str, int]] = {
        "H200": 2500, "H100": 2000, "GH200": 2000, "A100": 624,
        "L40": 362, "5090": 419, "4090": 330, "4080": 242,
        "3090": 142, "3080": 119, "A10G": 250, "V100": 125,
        "Apple M3": 30, "Apple M2": 25, "Unknown": 10,
    }

    def __post_init__(self):
        if self.memory_total_gb > 0 and self.memory_used_gb > 0:
            self.memory_free_gb = self.memory_total_gb - self.memory_used_gb
            if self.memory_percent == 0:
                self.memory_percent = (self.memory_used_gb / self.memory_total_gb) * 100
        if self.power_score == 0 and self.gpu_name:
            self.power_score = self._calculate_power_score()

    def _calculate_power_score(self) -> int:
        if not self.gpu_name:
            return 0
        gpu_upper = self.gpu_name.upper()
        for key, score in self.GPU_POWER_RANKINGS.items():
            if key.upper() in gpu_upper:
                return score * max(1, self.gpu_count)
        return self.GPU_POWER_RANKINGS["Unknown"]

    @property
    def is_cuda_gpu(self) -> bool:
        gpu_upper = (self.gpu_name or "").upper()
        return self.has_gpu and "MPS" not in gpu_upper and "APPLE" not in gpu_upper

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if k != "GPU_POWER_RANKINGS"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GPUInfo:
        return cls(**{k: v for k, v in data.items() if k != "GPU_POWER_RANKINGS"})

    @classmethod
    def from_nvidia_smi(cls, output: str) -> GPUInfo:
        """Create from nvidia-smi output: 'util%, mem_used_mb, mem_total_mb'."""
        try:
            parts = [p.strip() for p in output.strip().split(",")]
            if len(parts) >= 3:
                return cls(
                    has_gpu=True,
                    utilization_percent=float(parts[0]),
                    memory_used_gb=float(parts[1]) / 1024,
                    memory_total_gb=float(parts[2]) / 1024,
                )
        except (ValueError, IndexError):
            pass
        return cls()


@dataclass
class ResourceMetrics:
    """Resource utilization metrics for a node."""
    cpu_count: int = 0
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_gb_total: float = 0.0
    memory_gb_available: float = 0.0
    disk_percent: float = 0.0
    disk_gb_free: float = 0.0
    disk_gb_total: float = 0.0
    load_average_1m: float = 0.0
    load_average_5m: float = 0.0
    timestamp: float = 0.0

    DISK_WARNING_THRESHOLD: ClassVar[float] = 65.0
    DISK_CRITICAL_THRESHOLD: ClassVar[float] = float(DISK_SYNC_TARGET_PERCENT)
    MEMORY_WARNING_THRESHOLD: ClassVar[float] = 85.0

    @property
    def load_score(self) -> float:
        return max(self.cpu_percent, self.memory_percent)

    @property
    def is_overloaded(self) -> bool:
        return self.load_score > 80 or self.memory_percent > 80

    @property
    def is_disk_warning(self) -> bool:
        return self.disk_percent >= self.DISK_WARNING_THRESHOLD

    @property
    def is_disk_critical(self) -> bool:
        return self.disk_percent >= self.DISK_CRITICAL_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["load_score"] = self.load_score
        d["is_overloaded"] = self.is_overloaded
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResourceMetrics:
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid_keys})


@dataclass
class ConnectionInfo:
    """Connection information for reaching a node."""
    host: str = ""
    port: int = P2P_DEFAULT_PORT
    scheme: str = "http"
    tailscale_ip: str | None = None
    ssh_host: str | None = None
    ssh_user: str = "ubuntu"
    ssh_port: int = 22
    ssh_key: str | None = None
    nat_blocked: bool = False
    nat_blocked_since: float | None = None  # Timestamp when NAT block started
    relay_via: str = ""
    data_server_port: int = 8766

    @property
    def endpoint(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}"

    @property
    def best_ip(self) -> str | None:
        for candidate in (self.tailscale_ip, self.ssh_host, self.host):
            if candidate:
                host = str(candidate).strip()
                if host and "@" in host:
                    host = host.split("@", 1)[1]
                if host:
                    return host
        return None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConnectionInfo:
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid_keys})


@dataclass
class NodeHealthStatus:
    """Health status and failure tracking for a node.

    Dec 2025: Renamed from HealthStatus to avoid confusion with
    app.monitoring.base.HealthStatus (the simple enum).

    This dataclass tracks detailed node health over time, while the
    monitoring.base.HealthStatus enum is for simple status values.
    """
    health: NodeHealth = NodeHealth.UNKNOWN
    state: NodeState = NodeState.UNKNOWN
    last_heartbeat: float = 0.0
    last_seen: float = 0.0
    uptime_seconds: float = 0.0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: float = 0.0
    retired: bool = False
    retired_at: float | None = None  # Timestamp when node was retired
    nfs_accessible: bool = True
    errors_last_hour: int = 0

    # Jan 22, 2026: Use canonical timeout from app.p2p.constants (was hardcoded 90s)
    PEER_TIMEOUT: ClassVar[float] = _CANONICAL_PEER_TIMEOUT
    MAX_CONSECUTIVE_FAILURES: ClassVar[int] = 3

    @property
    def is_alive(self) -> bool:
        heartbeat_time = self.last_heartbeat or self.last_seen
        return time.time() - heartbeat_time < self.PEER_TIMEOUT

    @property
    def is_healthy(self) -> bool:
        if not self.is_alive or self.retired:
            return False
        if self.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
            return False
        return self.health in (NodeHealth.HEALTHY, NodeHealth.DEGRADED)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["health"] = self.health.value
        d["state"] = self.state.value
        d["is_alive"] = self.is_alive
        d["is_healthy"] = self.is_healthy
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeHealthStatus:
        data = data.copy()
        if "health" in data and isinstance(data["health"], str):
            data["health"] = NodeHealth(data["health"])
        if "state" in data and isinstance(data["state"], str):
            data["state"] = NodeState(data["state"])
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid_keys})


# Backwards compatibility alias
HealthStatus = NodeHealthStatus


@dataclass
class ProviderInfo:
    """Cloud provider-specific information."""
    provider: Provider = Provider.UNKNOWN
    instance_id: str | None = None
    vast_instance_id: str | None = None
    ringrift_path: str = "~/ringrift/ai-service"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["provider"] = self.provider.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProviderInfo:
        data = data.copy()
        if "provider" in data and isinstance(data["provider"], str):
            try:
                data["provider"] = Provider(data["provider"])
            except ValueError:
                data["provider"] = Provider.UNKNOWN
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid_keys})

    @classmethod
    def detect_from_node_id(cls, node_id: str) -> ProviderInfo:
        name_lower = node_id.lower()
        provider = Provider.UNKNOWN
        if "vast" in name_lower:
            provider = Provider.VAST
        elif "runpod" in name_lower:
            provider = Provider.RUNPOD
        elif "nebius" in name_lower:
            provider = Provider.NEBIUS
        elif "vultr" in name_lower:
            provider = Provider.VULTR
        elif "hetzner" in name_lower:
            provider = Provider.HETZNER
        elif "lambda" in name_lower or any(name_lower.startswith(f"{c}-") for c in "abcdefghijklmnopqrst"):
            provider = Provider.LAMBDA
        return cls(provider=provider)


@dataclass
class JobStatus:
    """Job status for a node."""
    active_jobs: int = 0
    selfplay_jobs: int = 0
    training_jobs: int = 0
    training_active: bool = False
    cmaes_running: bool = False
    gauntlet_running: bool = False
    tournament_running: bool = False
    data_merge_running: bool = False  # Whether data merge/export is in progress
    game_counts: dict[str, int] = field(default_factory=dict)
    total_games: int = 0

    @property
    def has_external_work(self) -> bool:
        return self.cmaes_running or self.gauntlet_running or self.tournament_running

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["has_external_work"] = self.has_external_work
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobStatus:
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid_keys})


# =============================================================================
# Unified NodeInfo
# =============================================================================

@dataclass
class NodeInfo:
    """Unified node information dataclass.

    Consolidates all node information structures used across the codebase.
    """
    node_id: str
    hostname: str = ""
    role: NodeRole = NodeRole.FOLLOWER
    leader_id: str | None = None  # Current cluster leader node_id
    capabilities: list[str] = field(default_factory=list)
    version: str = "1.0.0"
    code_version: str = ""

    gpu: GPUInfo = field(default_factory=GPUInfo)
    resources: ResourceMetrics = field(default_factory=ResourceMetrics)
    connection: ConnectionInfo = field(default_factory=ConnectionInfo)
    health: HealthStatus = field(default_factory=HealthStatus)
    provider: ProviderInfo = field(default_factory=ProviderInfo)
    jobs: JobStatus = field(default_factory=JobStatus)

    last_sync_time: datetime | None = None
    sync_lag_seconds: float = 0.0

    def __post_init__(self):
        if not self.hostname:
            self.hostname = self.node_id
        if self.provider.provider == Provider.UNKNOWN:
            self.provider = ProviderInfo.detect_from_node_id(self.node_id)

    @property
    def is_alive(self) -> bool:
        return self.health.is_alive

    @property
    def is_healthy(self) -> bool:
        return self.health.is_healthy and not self.resources.is_disk_warning

    @property
    def is_online(self) -> bool:
        return self.health.health in (NodeHealth.HEALTHY, NodeHealth.DEGRADED)

    @property
    def can_accept_jobs(self) -> bool:
        return self.is_online and not self.resources.is_overloaded

    @property
    def is_gpu_node(self) -> bool:
        return self.gpu.is_cuda_gpu

    @property
    def gpu_power_score(self) -> int:
        return self.gpu.power_score

    @property
    def endpoint(self) -> str:
        return self.connection.endpoint

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "role": self.role.value,
            "leader_id": self.leader_id,
            "capabilities": self.capabilities,
            "version": self.version,
            "code_version": self.code_version,
            "gpu": self.gpu.to_dict(),
            "resources": self.resources.to_dict(),
            "connection": self.connection.to_dict(),
            "health": self.health.to_dict(),
            "provider": self.provider.to_dict(),
            "jobs": self.jobs.to_dict(),
            "last_sync_time": self.last_sync_time.isoformat() if self.last_sync_time else None,
            "sync_lag_seconds": self.sync_lag_seconds,
            "is_alive": self.is_alive,
            "is_healthy": self.is_healthy,
            "is_online": self.is_online,
            "can_accept_jobs": self.can_accept_jobs,
            "is_gpu_node": self.is_gpu_node,
            "gpu_power_score": self.gpu_power_score,
            "endpoint": self.endpoint,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeInfo:
        data = data.copy()
        if "role" in data and isinstance(data["role"], str):
            try:
                data["role"] = NodeRole(data["role"])
            except ValueError:
                data["role"] = NodeRole.FOLLOWER

        data["gpu"] = GPUInfo.from_dict(data.get("gpu", {}))
        data["resources"] = ResourceMetrics.from_dict(data.get("resources", {}))
        data["connection"] = ConnectionInfo.from_dict(data.get("connection", {}))
        data["health"] = HealthStatus.from_dict(data.get("health", {}))
        data["provider"] = ProviderInfo.from_dict(data.get("provider", {}))
        data["jobs"] = JobStatus.from_dict(data.get("jobs", {}))

        if "last_sync_time" in data and isinstance(data["last_sync_time"], str):
            try:
                data["last_sync_time"] = datetime.fromisoformat(data["last_sync_time"])
            except ValueError:
                data["last_sync_time"] = None

        valid_keys = {"node_id", "hostname", "role", "leader_id", "capabilities", "version",
                      "code_version", "gpu", "resources", "connection", "health",
                      "provider", "jobs", "last_sync_time", "sync_lag_seconds"}
        return cls(**{k: v for k, v in data.items() if k in valid_keys})

    @classmethod
    def from_json(cls, json_str: str) -> NodeInfo:
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_p2p_status(cls, data: dict[str, Any]) -> NodeInfo:
        """Create from P2P orchestrator status response."""
        node_id = data.get("node_id", "")
        role_str = data.get("role", "follower")
        try:
            role = NodeRole(role_str)
        except ValueError:
            role = NodeRole.FOLLOWER

        gpu = GPUInfo(
            has_gpu=data.get("has_gpu", False),
            gpu_name=data.get("gpu_name", ""),
            gpu_count=data.get("gpu_count", 1 if data.get("has_gpu") else 0),
            memory_total_gb=data.get("gpu_memory_total", 0.0),
            memory_used_gb=data.get("gpu_memory_used", 0.0),
            utilization_percent=data.get("gpu_percent", 0.0),
        )

        resources = ResourceMetrics(
            cpu_count=data.get("cpu_count", 0),
            cpu_percent=data.get("cpu_percent", 0.0),
            memory_percent=data.get("memory_percent", 0.0),
            disk_percent=data.get("disk_percent", 0.0),
            disk_gb_free=data.get("disk_free_gb", 0.0),
        )

        connection = ConnectionInfo(
            host=data.get("host", ""),
            port=data.get("port", P2P_DEFAULT_PORT),
            tailscale_ip=data.get("tailscale_ip"),
            nat_blocked=data.get("nat_blocked", False),
            nat_blocked_since=data.get("nat_blocked_since"),
            relay_via=data.get("relay_via", ""),
        )

        health_state = data.get("health", "unknown")
        try:
            node_health = NodeHealth(health_state)
        except ValueError:
            node_health = NodeHealth.UNKNOWN

        health = HealthStatus(
            health=node_health,
            last_heartbeat=data.get("last_heartbeat", 0.0),
            last_seen=data.get("last_seen", time.time()),
            consecutive_failures=data.get("consecutive_failures", 0),
            retired=data.get("retired", False),
            retired_at=data.get("retired_at"),
        )

        jobs = JobStatus(
            active_jobs=data.get("active_jobs", 0),
            selfplay_jobs=data.get("selfplay_jobs", 0),
            training_jobs=data.get("training_jobs", 0),
            training_active=data.get("training_active", False),
            data_merge_running=data.get("data_merge_running", False),
        )

        return cls(
            node_id=node_id,
            hostname=data.get("hostname", node_id),
            role=role,
            leader_id=data.get("leader_id"),
            capabilities=data.get("capabilities", []),
            gpu=gpu,
            resources=resources,
            connection=connection,
            health=health,
            provider=ProviderInfo.detect_from_node_id(node_id),
            jobs=jobs,
        )

    @classmethod
    def from_ssh_discovery(
        cls,
        node_id: str,
        host: str,
        nvidia_smi_output: str | None = None,
        **kwargs: Any,
    ) -> NodeInfo:
        """Create from SSH discovery."""
        gpu = GPUInfo()
        if nvidia_smi_output and "no-gpu" not in nvidia_smi_output:
            gpu = GPUInfo.from_nvidia_smi(nvidia_smi_output)

        connection = ConnectionInfo(
            host=host,
            ssh_host=host,
            ssh_user=kwargs.get("ssh_user", "ubuntu"),
            ssh_port=kwargs.get("ssh_port", 22),
            ssh_key=kwargs.get("ssh_key"),
        )

        health = HealthStatus(
            health=NodeHealth.HEALTHY,
            state=NodeState.ONLINE,
            last_seen=time.time(),
        )

        return cls(
            node_id=node_id,
            hostname=node_id,
            gpu=gpu,
            connection=connection,
            health=health,
            provider=ProviderInfo.detect_from_node_id(node_id),
        )

    @classmethod
    def from_cluster_config(cls, name: str, config: dict[str, Any]) -> NodeInfo:
        """Create from distributed_hosts.yaml configuration."""
        connection = ConnectionInfo(
            host=config.get("ssh_host", ""),
            tailscale_ip=config.get("tailscale_ip"),
            ssh_host=config.get("ssh_host"),
            ssh_user=config.get("ssh_user", "ubuntu"),
            ssh_port=config.get("ssh_port", 22),
            ssh_key=config.get("ssh_key"),
        )

        gpu = GPUInfo(
            has_gpu=bool(config.get("gpu")),
            gpu_name=config.get("gpu", ""),
        )

        resources = ResourceMetrics(
            memory_gb_total=config.get("memory_gb", 0),
            cpu_count=config.get("cpus", 0),
        )

        provider = ProviderInfo.detect_from_node_id(name)
        provider.ringrift_path = config.get("ringrift_path", "~/ringrift/ai-service")

        status = config.get("status", "unknown")
        if status in ("ready", "active"):
            health = HealthStatus(health=NodeHealth.HEALTHY, state=NodeState.ONLINE)
        elif status in ("offline", "terminated", "archived"):
            health = HealthStatus(health=NodeHealth.OFFLINE, state=NodeState.OFFLINE)
        else:
            health = HealthStatus(health=NodeHealth.UNKNOWN, state=NodeState.UNKNOWN)

        role_str = config.get("role", "follower")
        try:
            role = NodeRole(role_str)
        except ValueError:
            role = NodeRole.FOLLOWER

        return cls(
            node_id=name,
            hostname=name,
            role=role,
            gpu=gpu,
            resources=resources,
            connection=connection,
            health=health,
            provider=provider,
        )

    @classmethod
    def create_local(cls, node_id: str | None = None) -> NodeInfo:
        """Create NodeInfo for the local machine."""
        hostname = socket.gethostname()
        node_id = node_id or hostname

        resources = ResourceMetrics(cpu_count=os.cpu_count() or 1)

        try:
            import psutil
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            resources.cpu_percent = psutil.cpu_percent()
            resources.memory_percent = memory.percent
            resources.memory_gb_total = memory.total / (1024**3)
            resources.disk_percent = disk.percent
            resources.disk_gb_free = disk.free / (1024**3)
        except ImportError:
            pass

        gpu = GPUInfo()
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = [p.strip() for p in result.stdout.strip().split(",")]
                if len(parts) >= 4:
                    gpu = GPUInfo(
                        has_gpu=True,
                        gpu_name=parts[0],
                        gpu_count=1,
                        utilization_percent=float(parts[1]),
                        memory_used_gb=float(parts[2]) / 1024,
                        memory_total_gb=float(parts[3]) / 1024,
                    )
        except (FileNotFoundError, OSError, ValueError, IndexError, TypeError):
            pass

        return cls(
            node_id=node_id,
            hostname=hostname,
            gpu=gpu,
            resources=resources,
            health=HealthStatus(health=NodeHealth.HEALTHY, state=NodeState.ONLINE),
            provider=ProviderInfo.detect_from_node_id(node_id),
        )


# =============================================================================
# P2P Leader Detection Utilities (December 2025)
# Consolidated from job_reaper.py and other modules
# =============================================================================

def get_this_node_id() -> str:
    """Get this node's ID for leader comparison.

    Returns the node ID from:
    1. RINGRIFT_NODE_ID environment variable (if set)
    2. Hostname as fallback

    Returns:
        Node identifier string
    """
    node_id = os.environ.get("RINGRIFT_NODE_ID")
    if node_id:
        return node_id
    return socket.gethostname()


async def check_p2p_leader_status(timeout: float = 5.0) -> tuple[bool, str | None]:
    """Check if this node is the P2P cluster leader.

    December 2025: Consolidated from job_reaper.py for reuse across modules.

    Args:
        timeout: Timeout in seconds for the P2P status check

    Returns:
        Tuple of (is_leader, leader_id):
        - (True, node_id) if this node is the leader
        - (False, leader_id) if another node is the leader
        - (False, None) if P2P is unavailable or error
    """
    try:
        import aiohttp
    except ImportError:
        return False, None

    this_node = get_this_node_id()

    # Get P2P URL from centralized config
    try:
        from app.config.ports import get_local_p2p_url
        p2p_base = get_local_p2p_url()
    except ImportError:
        p2p_base = os.environ.get("RINGRIFT_P2P_URL", f"http://localhost:{P2P_DEFAULT_PORT}")

    p2p_url = f"{p2p_base}/status"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                p2p_url, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    leader_id = data.get("leader_id")
                    is_leader = data.get("is_leader", False)

                    # Check by is_leader flag first
                    if is_leader:
                        return True, leader_id

                    # Check by comparing node IDs
                    if leader_id and leader_id == this_node:
                        return True, leader_id

                    return False, leader_id

    except (
        aiohttp.ClientError,
        asyncio.TimeoutError,
        OSError,
        json.JSONDecodeError,
        KeyError,
    ):
        # Network/timeout/parse errors - assume not leader (safer default)
        return False, None


def get_is_leader_sync(timeout: float = 5.0) -> bool:
    """Synchronous wrapper for leader check.

    Args:
        timeout: Timeout in seconds for the P2P status check

    Returns:
        True if this node is the leader, False otherwise
    """
    import asyncio

    try:
        loop = asyncio.get_running_loop()
        # Can't run sync in running loop, return False (safe default)
        return False
    except RuntimeError:
        # No running loop, create one
        is_leader, _ = asyncio.run(check_p2p_leader_status(timeout))
        return is_leader


# Compatibility aliases
NodeStatus = NodeInfo
NodeSummary = NodeInfo
ClusterNode = NodeInfo
