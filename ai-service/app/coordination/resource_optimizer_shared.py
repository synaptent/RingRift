"""Shared constants and resource-state models for resource optimization."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from app.config.env import env

try:
    from app.config.coordination_defaults import PIDDefaults, UtilizationDefaults

    PID_KP = PIDDefaults.KP
    PID_KI = PIDDefaults.KI
    PID_KD = PIDDefaults.KD
    UTILIZATION_UPDATE_INTERVAL = UtilizationDefaults.UPDATE_INTERVAL
    OPTIMIZATION_INTERVAL = UtilizationDefaults.OPTIMIZATION_INTERVAL
except ImportError:
    PID_KP = env.pid_kp
    PID_KI = env.pid_ki
    PID_KD = env.pid_kd
    UTILIZATION_UPDATE_INTERVAL = 10
    OPTIMIZATION_INTERVAL = 30

TARGET_UTIL_MIN = env.target_util_min
TARGET_UTIL_MAX = env.target_util_max
TARGET_UTIL_OPTIMAL = (TARGET_UTIL_MIN + TARGET_UTIL_MAX) / 2
SCALE_UP_THRESHOLD = env.scale_up_threshold
SCALE_DOWN_THRESHOLD = env.scale_down_threshold


@dataclass
class NodeResources:
    """Resource state for a single node."""

    node_id: str
    cpu_percent: float = 0.0
    gpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    cpu_count: int = 0
    gpu_count: int = 0
    memory_gb: float = 0.0
    has_gpu: bool = False
    gpu_name: str = ""
    active_jobs: int = 0
    selfplay_jobs: int = 0
    training_jobs: int = 0
    updated_at: float = 0.0
    orchestrator: str = ""

    def get_max_gpu_jobs(self) -> int:
        if not self.has_gpu or self.gpu_count == 0:
            return 0
        if any(g in self.gpu_name.upper() for g in ["H100", "H200", "A100", "L40"]):
            jobs_per_gpu = 4
        elif any(g in self.gpu_name.upper() for g in ["A10", "4090", "5090", "3090"]):
            jobs_per_gpu = 3
        else:
            jobs_per_gpu = 2
        return self.gpu_count * jobs_per_gpu

    def get_max_cpu_jobs(self) -> int:
        if self.cpu_count == 0:
            return 8
        cpu_based = max(1, self.cpu_count // 2)
        memory_based = max(1, int(self.memory_gb / 2)) if self.memory_gb > 0 else 32
        return min(cpu_based, memory_based, 48)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeResources:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ClusterState:
    """Aggregated cluster resource state."""

    nodes: list[NodeResources]
    total_cpu_util: float = 0.0
    total_gpu_util: float = 0.0
    total_memory_util: float = 0.0
    total_gpu_memory_util: float = 0.0
    gpu_node_count: int = 0
    cpu_node_count: int = 0
    total_jobs: int = 0
    updated_at: float = 0.0

    GPU_MEMORY_WARNING: float = 80.0
    GPU_MEMORY_CRITICAL: float = 90.0

    def compute_aggregates(self) -> None:
        if not self.nodes:
            return

        cpu_utils = [n.cpu_percent for n in self.nodes if n.cpu_percent > 0]
        gpu_utils = [n.gpu_percent for n in self.nodes if n.has_gpu and n.gpu_percent > 0]
        mem_utils = [n.memory_percent for n in self.nodes if n.memory_percent > 0]
        gpu_mem_utils = [n.gpu_memory_percent for n in self.nodes if n.has_gpu and n.gpu_memory_percent > 0]

        self.total_cpu_util = sum(cpu_utils) / len(cpu_utils) if cpu_utils else 0.0
        self.total_gpu_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0.0
        self.total_memory_util = sum(mem_utils) / len(mem_utils) if mem_utils else 0.0
        self.total_gpu_memory_util = sum(gpu_mem_utils) / len(gpu_mem_utils) if gpu_mem_utils else 0.0

        self.gpu_node_count = len([n for n in self.nodes if n.has_gpu])
        self.cpu_node_count = len(self.nodes)
        self.total_jobs = sum(n.active_jobs for n in self.nodes)
        self.updated_at = time.time()

    def is_gpu_memory_constrained(self) -> bool:
        return any(node.has_gpu and node.gpu_memory_percent > self.GPU_MEMORY_WARNING for node in self.nodes)

    def is_gpu_memory_critical(self) -> bool:
        return any(node.has_gpu and node.gpu_memory_percent > self.GPU_MEMORY_CRITICAL for node in self.nodes)

    def get_gpu_memory_status(self) -> str:
        if self.is_gpu_memory_critical():
            return "critical"
        if self.is_gpu_memory_constrained():
            return "warning"
        return "ok"


__all__ = [
    "ClusterState",
    "NodeResources",
    "OPTIMIZATION_INTERVAL",
    "PID_KD",
    "PID_KI",
    "PID_KP",
    "SCALE_DOWN_THRESHOLD",
    "SCALE_UP_THRESHOLD",
    "TARGET_UTIL_MAX",
    "TARGET_UTIL_MIN",
    "TARGET_UTIL_OPTIMAL",
    "UTILIZATION_UPDATE_INTERVAL",
]
