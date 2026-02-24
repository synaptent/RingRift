"""Unified resource utilization targets for cluster orchestration.

This module provides consistent utilization targets across all orchestrators
(unified_ai_loop, p2p_orchestrator, cluster_orchestrator) to achieve stable
60-80% CPU/GPU utilization for optimal AI training throughput.

Key principles:
1. Single source of truth for utilization targets
2. Host-specific adjustments based on capability
3. Adaptive targets based on training pipeline health
4. Backpressure-aware throttling

Usage:
    from app.coordination import (
        get_resource_targets,
        get_host_targets,
        should_scale_up,
        should_scale_down,
        get_utilization_score,
    )

    targets = get_resource_targets()
    if should_scale_up("runpod-h100", current_gpu=45):
        # Add more jobs
        pass
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from app.utils.yaml_utils import safe_load_yaml
from app.coordination.singleton_mixin import SingletonMixin
from app.config.thresholds import (
    DISK_CRITICAL_PERCENT,
    DISK_PRODUCTION_HALT_PERCENT,
    SQLITE_CONNECT_TIMEOUT,
)

# Import interfaces for type hints (no circular dependency)
# December 2025: IResourceTargets/IResourceTargetManager enable protocol-based typing
from app.coordination.interfaces import IResourceTargets, IResourceTargetManager

logger = logging.getLogger(__name__)

# Import hardware-aware selfplay limits from resource_optimizer (single source of truth)
# Lazy import to avoid circular dependency
_get_max_selfplay_for_node = None


def _get_hw_max_selfplay(node_id: str, gpu_name: str = "", cpu_count: int = 0,
                          memory_gb: float = 0, gpu_count: int = 0, has_gpu: bool = False) -> int:
    """Get hardware-aware max selfplay (lazy import to avoid circular dependency)."""
    global _get_max_selfplay_for_node
    if _get_max_selfplay_for_node is None:
        try:
            from app.coordination.resource_optimizer import get_max_selfplay_for_node
            _get_max_selfplay_for_node = get_max_selfplay_for_node
        except ImportError:
            # Fallback if resource_optimizer not available
            return 8  # Conservative default
    return _get_max_selfplay_for_node(
        node_id=node_id, gpu_count=gpu_count, gpu_name=gpu_name,
        cpu_count=cpu_count, memory_gb=memory_gb, has_gpu=has_gpu
    )


# Default coordination DB path
_DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "coordination.db"
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "unified_loop.yaml"


def _load_config_targets(config_path: Path | None = None) -> dict[str, Any]:
    """Load resource_targets from config file."""
    path = config_path or _DEFAULT_CONFIG_PATH
    config = safe_load_yaml(path, default={}, log_errors=True)
    return config.get("resource_targets", {})


class HostTier(Enum):
    """Host capability tiers for target adjustment."""

    HIGH_END = "high_end"       # H100, H200, GH200, Mac Studio
    MID_TIER = "mid_tier"       # RTX 4090, A6000, Mac Pro
    LOW_TIER = "low_tier"       # Consumer GPUs, older Macs
    CPU_ONLY = "cpu_only"       # No GPU acceleration


@dataclass
class UtilizationTargets:
    """Target utilization ranges for resource optimization.

    Targets 60-80% utilization as the optimal operating range:
    - Below 60%: Scale up (underutilized, wasting capacity)
    - 60-80%: Optimal range (stable, efficient)
    - Above 80%: Scale down (risk of throttling/OOM)
    """

    # CPU targets (percentage) - max 80% enforced 2025-12-16
    cpu_min: float = 50.0           # Scale up below this
    cpu_target: float = 65.0        # Ideal operating point
    cpu_max: float = 80.0           # Scale down above this (HARD LIMIT)
    cpu_critical: float = 80.0      # Emergency stop (same as max)

    # GPU targets (percentage) - max 80% enforced 2025-12-16
    gpu_min: float = 50.0           # Scale up below this
    gpu_target: float = 65.0        # Ideal operating point
    gpu_max: float = 80.0           # Scale down above this (HARD LIMIT)
    gpu_critical: float = 80.0      # Emergency stop (same as max)

    # Memory targets (percentage) - max 80% enforced 2025-12-16
    memory_warn: float = 70.0       # Reduce jobs
    memory_critical: float = 80.0   # Stop spawning (HARD LIMIT)

    # Disk targets (percentage) - from app.config.thresholds (canonical source)
    disk_warn: float = float(DISK_PRODUCTION_HALT_PERCENT)   # Trigger cleanup / pause production
    disk_critical: float = float(DISK_CRITICAL_PERCENT)      # Stop all data-producing tasks

    # Job concurrency targets
    jobs_per_core: float = 0.5      # Target 50% core utilization from jobs
    max_jobs_per_node: int = 48     # Hard cap per node
    max_selfplay_cluster: int = 500 # Hard cap cluster-wide

    # Throughput targets (games per hour)
    throughput_min: int = 500       # Scale up below this
    throughput_target: int = 1000   # Cluster-wide target
    throughput_max: int = 2000      # Scale down above this


@dataclass
class HostTargets:
    """Per-host adjusted targets based on capability tier."""

    host: str
    tier: HostTier

    # Adjusted targets
    cpu_min: float
    cpu_target: float
    cpu_max: float

    gpu_min: float
    gpu_target: float
    gpu_max: float

    max_jobs: int
    max_selfplay: int
    max_training: int

    # Current state tracking
    last_cpu: float = 0.0
    last_gpu: float = 0.0
    last_memory: float = 0.0
    last_jobs: int = 0
    last_update: float = 0.0


# Host tier classification based on known hosts
HOST_TIER_MAP: dict[str, HostTier] = {
    # High-end (>= 80GB VRAM, modern architecture)
    "runpod-h100": HostTier.HIGH_END,
    "nebius-h100": HostTier.HIGH_END,
    "gh200": HostTier.HIGH_END,
    "mac-studio": HostTier.HIGH_END,
    "mac-studio-ultra": HostTier.HIGH_END,

    # Mid-tier (24-48GB VRAM)
    "rtx4090": HostTier.MID_TIER,
    "a6000": HostTier.MID_TIER,
    "mac-pro": HostTier.MID_TIER,
    "aws-g5": HostTier.MID_TIER,

    # Low-tier (<24GB VRAM)
    "rtx3090": HostTier.LOW_TIER,
    "m1-pro": HostTier.LOW_TIER,
    "m1-max": HostTier.LOW_TIER,
    "vast-rtx3090": HostTier.LOW_TIER,

    # CPU only
    "aws-c5": HostTier.CPU_ONLY,
    "local": HostTier.CPU_ONLY,
}

# Tier-specific target adjustments
TIER_ADJUSTMENTS: dict[HostTier, dict[str, float]] = {
    HostTier.HIGH_END: {
        "cpu_boost": 5.0,      # Can run 5% higher CPU
        "gpu_boost": 5.0,      # Can run 5% higher GPU
        "job_multiplier": 1.5, # 50% more jobs
    },
    HostTier.MID_TIER: {
        "cpu_boost": 0.0,
        "gpu_boost": 0.0,
        "job_multiplier": 1.0,
    },
    HostTier.LOW_TIER: {
        "cpu_boost": -5.0,     # Run 5% lower CPU
        "gpu_boost": -5.0,     # Run 5% lower GPU
        "job_multiplier": 0.7, # 30% fewer jobs
    },
    HostTier.CPU_ONLY: {
        "cpu_boost": 0.0,
        "gpu_boost": 0.0,
        "job_multiplier": 0.5, # Half as many jobs (no GPU acceleration)
    },
}


class ResourceTargetManager(SingletonMixin):
    """Manages unified resource utilization targets across the cluster.

    Thread-safe singleton that provides:
    - Consistent targets for all orchestrators
    - Host-specific adjustments
    - Adaptive targets based on pipeline health
    - Utilization history tracking

    December 27, 2025: Migrated to SingletonMixin (Wave 4 Phase 1).
    """

    def __init__(self, db_path: Path | None = None, config_path: Path | None = None):
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._config_path = config_path or _DEFAULT_CONFIG_PATH
        self._targets = self._load_targets_from_config()
        self._tier_overrides = self._load_tier_overrides_from_config()
        self._host_targets: dict[str, HostTargets] = {}
        self._utilization_history: dict[str, list[tuple[float, float, float]]] = {}
        self._backpressure_factor: float = 1.0
        self._last_adaptive_update: float = 0.0
        # Lazy DB initialization - don't call _init_db() here to avoid
        # import-time writes that fail on readonly filesystems
        self._db_initialized: bool = False
        self._readonly_mode: bool = False

    def _ensure_db(self) -> bool:
        """Lazily initialize database, returns True if writable.

        This method is called before any database operation to ensure
        the database is initialized. On readonly filesystems, it sets
        _readonly_mode=True and returns False.
        """
        if self._db_initialized:
            return not self._readonly_mode

        # Check if parent directory is writable
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.debug(f"Cannot create DB directory: {self._db_path.parent}")
            self._readonly_mode = True
            self._db_initialized = True
            return False

        if not os.access(self._db_path.parent, os.W_OK):
            logger.debug(f"DB directory not writable: {self._db_path.parent}")
            self._readonly_mode = True
            self._db_initialized = True
            return False

        try:
            self._init_db()
            self._db_initialized = True
            return True
        except sqlite3.OperationalError as e:
            if "readonly" in str(e).lower():
                logger.debug(f"Database is readonly: {e}")
                self._readonly_mode = True
                self._db_initialized = True
                return False
            raise

    def _load_targets_from_config(self) -> UtilizationTargets:
        """Load utilization targets from config file."""
        config = _load_config_targets(self._config_path)

        # Build targets from config, with defaults
        return UtilizationTargets(
            cpu_min=config.get("cpu_min", 60.0),
            cpu_target=config.get("cpu_target", 70.0),
            cpu_max=config.get("cpu_max", 80.0),
            cpu_critical=config.get("cpu_critical", 90.0),
            gpu_min=config.get("gpu_min", 60.0),
            gpu_target=config.get("gpu_target", 75.0),
            gpu_max=config.get("gpu_max", 85.0),
            gpu_critical=config.get("gpu_critical", 95.0),
            memory_warn=config.get("memory_max", 75.0),
            memory_critical=config.get("memory_critical", 85.0),
            disk_warn=config.get("disk_warn", 80.0),
            disk_critical=config.get("disk_critical", 90.0),
            jobs_per_core=config.get("jobs_per_core", 0.5),
            max_jobs_per_node=config.get("max_jobs_per_node", 48),
            max_selfplay_cluster=config.get("max_selfplay_cluster", 500),
            throughput_min=config.get("throughput_min", 500),
            throughput_target=config.get("throughput_target", 1000),
            throughput_max=config.get("throughput_max", 2000),
        )

    def _load_tier_overrides_from_config(self) -> dict[HostTier, dict[str, Any]]:
        """Load tier-specific overrides from config file."""
        config = _load_config_targets(self._config_path)
        tier_config = config.get("tier_overrides", {})

        # Merge with defaults
        overrides = dict(TIER_ADJUSTMENTS)  # Start with defaults

        for tier_name, tier_values in tier_config.items():
            try:
                tier = HostTier[tier_name.upper()]
                if tier in overrides:
                    # Merge values
                    for key, value in tier_values.items():
                        overrides[tier][key] = value
            except KeyError:
                logger.warning(f"Unknown tier: {tier_name}")

        return overrides

    def _init_db(self) -> None:
        """Initialize database tables for target persistence."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS resource_targets (
                    id INTEGER PRIMARY KEY,
                    targets_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS utilization_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    host TEXT NOT NULL,
                    cpu_percent REAL NOT NULL,
                    gpu_percent REAL NOT NULL,
                    memory_percent REAL NOT NULL,
                    job_count INTEGER NOT NULL,
                    recorded_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_util_host_time
                ON utilization_history(host, recorded_at)
            """)
            conn.commit()

    def get_targets(self) -> UtilizationTargets:
        """Get current utilization targets."""
        return self._targets

    def get_host_targets(self, host: str) -> HostTargets:
        """Get adjusted targets for a specific host."""
        if host not in self._host_targets:
            self._host_targets[host] = self._compute_host_targets(host)
        return self._host_targets[host]

    def _get_node_hardware(self, host: str) -> dict[str, Any] | None:
        """Get hardware info for a node from the coordination DB."""
        # Ensure DB is available (graceful degradation on readonly)
        if not self._ensure_db():
            return None
        try:
            # December 27, 2025: Use context manager to prevent connection leaks
            with sqlite3.connect(self._db_path, timeout=SQLITE_CONNECT_TIMEOUT) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT gpu_count, gpu_name, cpu_count, memory_gb, has_gpu FROM node_resources WHERE node_id = ?",
                    (host,)
                ).fetchone()
                if row:
                    return {
                        "gpu_count": row["gpu_count"] or 0,
                        "gpu_name": row["gpu_name"] or "",
                        "cpu_count": row["cpu_count"] or 0,
                        "memory_gb": row["memory_gb"] or 0,
                        "has_gpu": bool(row["has_gpu"]),
                    }
        except (sqlite3.Error, OSError, KeyError):
            pass
        return None

    def _compute_host_targets(self, host: str) -> HostTargets:
        """Compute tier-adjusted targets for a host.

        Uses hardware-aware limits from resource_optimizer when available,
        falling back to tier-based defaults if hardware info not found.
        """
        # Determine host tier
        tier = HOST_TIER_MAP.get(host, HostTier.LOW_TIER)

        # Check for tier patterns in host name
        if tier == HostTier.LOW_TIER:
            host_lower = host.lower()
            if any(x in host_lower for x in ["h100", "h200", "gh200", "studio-ultra"]):
                tier = HostTier.HIGH_END
            elif any(x in host_lower for x in ["4090", "a6000", "mac-pro"]):
                tier = HostTier.MID_TIER

        # Get tier adjustments (from config or defaults)
        adj = self._tier_overrides.get(tier, TIER_ADJUSTMENTS[tier])
        base = self._targets

        # Apply backpressure reduction
        bp_factor = self._backpressure_factor

        # Config can override specific targets per tier
        cpu_target = adj.get("cpu_target", base.cpu_target + adj.get("cpu_boost", 0.0))
        gpu_target = adj.get("gpu_target", base.gpu_target + adj.get("gpu_boost", 0.0))
        max_jobs = adj.get("max_jobs_per_node", int(base.max_jobs_per_node * adj.get("job_multiplier", 1.0)))

        # Handle CPU-only tier (no GPU targets)
        gpu_min = adj.get("gpu_min", base.gpu_min + adj.get("gpu_boost", 0.0))
        gpu_max = adj.get("gpu_max", base.gpu_max + adj.get("gpu_boost", 0.0))

        # Get hardware-aware max_selfplay from resource_optimizer (single source of truth)
        hw = self._get_node_hardware(host)
        if hw is not None:
            # Use hardware-aware calculation
            max_selfplay = _get_hw_max_selfplay(
                node_id=host,
                gpu_name=hw["gpu_name"],
                cpu_count=hw["cpu_count"],
                memory_gb=hw["memory_gb"],
                gpu_count=hw["gpu_count"],
                has_gpu=hw["has_gpu"],
            )
        else:
            # Fallback to tier-based calculation (when hardware info not available)
            # Values calibrated from observed workloads (GH200: 48 jobs at 70% GPU util)
            if tier == HostTier.HIGH_END:
                # GH200/H100 class - very high capacity
                max_selfplay = 48
            elif tier == HostTier.MID_TIER:
                # A10/4090 class
                max_selfplay = 16
            elif tier == HostTier.CPU_ONLY:
                # CPU-only nodes
                max_selfplay = 12
            else:
                max_selfplay = 8  # LOW_TIER/consumer GPUs

        return HostTargets(
            host=host,
            tier=tier,
            cpu_min=(base.cpu_min + adj.get("cpu_boost", 0.0)) * bp_factor,
            cpu_target=cpu_target * bp_factor,
            cpu_max=base.cpu_max + adj.get("cpu_boost", 0.0),
            gpu_min=gpu_min * bp_factor,
            gpu_target=gpu_target * bp_factor,
            gpu_max=gpu_max,
            max_jobs=max_jobs,
            max_selfplay=max_selfplay,
            max_training=1 if tier in (HostTier.HIGH_END, HostTier.MID_TIER) else 0,
        )

    def should_scale_up(
        self,
        host: str,
        current_cpu: float,
        current_gpu: float = 0.0,
        current_jobs: int = 0,
    ) -> tuple[bool, str]:
        """Check if a host should scale up (add jobs).

        Returns: (should_scale_up, reason)
        """
        targets = self.get_host_targets(host)

        # Check GPU first (if available)
        if current_gpu > 0 and current_gpu < targets.gpu_min:
            gpu_gap = targets.gpu_target - current_gpu
            return True, f"GPU underutilized: {current_gpu:.1f}% < {targets.gpu_min:.1f}% (gap: {gpu_gap:.1f}%)"

        # Check CPU
        if current_cpu < targets.cpu_min:
            cpu_gap = targets.cpu_target - current_cpu
            return True, f"CPU underutilized: {current_cpu:.1f}% < {targets.cpu_min:.1f}% (gap: {cpu_gap:.1f}%)"

        # Check job count headroom
        if current_jobs < targets.max_jobs * 0.8:
            return True, f"Job headroom available: {current_jobs} < {int(targets.max_jobs * 0.8)}"

        return False, "Within target range"

    def should_scale_down(
        self,
        host: str,
        current_cpu: float,
        current_gpu: float = 0.0,
        current_memory: float = 0.0,
    ) -> tuple[bool, int, str]:
        """Check if a host should scale down (remove jobs).

        Returns: (should_scale_down, reduction_count, reason)
        """
        targets = self.get_host_targets(host)
        base = self._targets

        # Critical memory - emergency reduction
        if current_memory >= base.memory_critical:
            return True, 10, f"CRITICAL memory: {current_memory:.1f}% >= {base.memory_critical:.1f}%"

        # Critical GPU - emergency reduction
        if current_gpu >= base.gpu_critical:
            return True, 6, f"CRITICAL GPU: {current_gpu:.1f}% >= {base.gpu_critical:.1f}%"

        # Critical CPU - emergency reduction
        if current_cpu >= base.cpu_critical:
            return True, 4, f"CRITICAL CPU: {current_cpu:.1f}% >= {base.cpu_critical:.1f}%"

        # High memory - moderate reduction
        if current_memory >= base.memory_warn:
            return True, 3, f"High memory: {current_memory:.1f}% >= {base.memory_warn:.1f}%"

        # High GPU - moderate reduction
        if current_gpu > targets.gpu_max:
            reduction = max(1, int((current_gpu - targets.gpu_target) / 10))
            return True, reduction, f"GPU overloaded: {current_gpu:.1f}% > {targets.gpu_max:.1f}%"

        # High CPU - moderate reduction
        if current_cpu > targets.cpu_max:
            reduction = max(1, int((current_cpu - targets.cpu_target) / 10))
            return True, reduction, f"CPU overloaded: {current_cpu:.1f}% > {targets.cpu_max:.1f}%"

        return False, 0, "Within target range"

    def get_target_job_count(
        self,
        host: str,
        cpu_cores: int,
        current_cpu: float,
        current_gpu: float = 0.0,
    ) -> int:
        """Calculate the target number of jobs for a host.

        Uses a feedback-based approach to converge on optimal job count.
        """
        targets = self.get_host_targets(host)
        base = self._targets

        # Base job count from core count
        base_jobs = int(cpu_cores * base.jobs_per_core)

        # Adjust based on current utilization
        if current_gpu > 0:
            # GPU node: target GPU utilization
            gpu_gap = targets.gpu_target - current_gpu
            adjustment = int(gpu_gap / 15)  # ~15% GPU per job
        else:
            # CPU node: target CPU utilization
            cpu_gap = targets.cpu_target - current_cpu
            adjustment = int(cpu_gap / 10)  # ~10% CPU per job

        target_jobs = base_jobs + adjustment

        # Apply limits
        target_jobs = max(1, min(target_jobs, targets.max_jobs))

        # Apply backpressure reduction
        target_jobs = int(target_jobs * self._backpressure_factor)

        return max(1, target_jobs)

    def get_utilization_score(
        self,
        host: str,
        current_cpu: float,
        current_gpu: float = 0.0,
    ) -> float:
        """Calculate a 0-100 utilization score for load balancing.

        Score interpretation:
        - 0-40: Underutilized (should receive more work)
        - 40-70: Optimal (balanced load)
        - 70-100: Overloaded (should shed load)
        """
        targets = self.get_host_targets(host)

        # Calculate CPU score
        if current_cpu <= targets.cpu_min:
            cpu_score = (current_cpu / targets.cpu_min) * 40
        elif current_cpu <= targets.cpu_max:
            cpu_score = 40 + ((current_cpu - targets.cpu_min) / (targets.cpu_max - targets.cpu_min)) * 30
        else:
            cpu_score = 70 + min(30, (current_cpu - targets.cpu_max) / 10 * 30)

        # Calculate GPU score (if available)
        if current_gpu > 0:
            if current_gpu <= targets.gpu_min:
                gpu_score = (current_gpu / targets.gpu_min) * 40
            elif current_gpu <= targets.gpu_max:
                gpu_score = 40 + ((current_gpu - targets.gpu_min) / (targets.gpu_max - targets.gpu_min)) * 30
            else:
                gpu_score = 70 + min(30, (current_gpu - targets.gpu_max) / 10 * 30)

            # Weighted average (GPU-heavy for GPU nodes)
            return cpu_score * 0.3 + gpu_score * 0.7

        return cpu_score

    def record_utilization(
        self,
        host: str,
        cpu_percent: float,
        gpu_percent: float,
        memory_percent: float,
        job_count: int,
    ) -> None:
        """Record utilization metrics for adaptive target tuning."""
        now = time.time()

        # Update host targets state
        if host in self._host_targets:
            ht = self._host_targets[host]
            ht.last_cpu = cpu_percent
            ht.last_gpu = gpu_percent
            ht.last_memory = memory_percent
            ht.last_jobs = job_count
            ht.last_update = now

        # Record to history (in memory, periodic DB flush)
        if host not in self._utilization_history:
            self._utilization_history[host] = []

        history = self._utilization_history[host]
        history.append((now, cpu_percent, gpu_percent))

        # Keep last hour in memory
        cutoff = now - 3600
        self._utilization_history[host] = [(t, c, g) for t, c, g in history if t > cutoff]

        # Periodic DB flush (every 5 minutes)
        if now - self._last_adaptive_update > 300:
            self._flush_history_to_db()
            self._update_adaptive_targets()
            self._last_adaptive_update = now

    def _flush_history_to_db(self) -> None:
        """Flush utilization history to database."""
        # Skip flush on readonly filesystem
        if not self._ensure_db():
            return
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                # Delete old records (keep 24 hours)
                cutoff = time.time() - 86400
                conn.execute(
                    "DELETE FROM utilization_history WHERE recorded_at < ?",
                    (cutoff,)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"DB flush error: {e}")

    def _update_adaptive_targets(self) -> None:
        """Update targets based on historical utilization patterns."""
        # Check training pipeline backpressure
        try:
            from app.coordination.queue_monitor import QueueType, get_throttle_factor
            self._backpressure_factor = get_throttle_factor(QueueType.TRAINING_DATA)
        except (ImportError, AttributeError, TypeError):
            self._backpressure_factor = 1.0

        # Invalidate cached host targets to pick up new backpressure
        self._host_targets.clear()

    def set_backpressure(self, factor: float) -> None:
        """Manually set backpressure factor (0.0-1.0)."""
        self._backpressure_factor = max(0.0, min(1.0, factor))
        self._host_targets.clear()

    def get_cluster_summary(self) -> dict:
        """Get summary of cluster utilization state."""
        total_cpu = 0.0
        total_gpu = 0.0
        total_memory = 0.0
        total_jobs = 0
        host_count = 0

        for _host, targets in self._host_targets.items():
            if targets.last_update > time.time() - 120:  # Updated in last 2 min
                total_cpu += targets.last_cpu
                total_gpu += targets.last_gpu
                total_memory += targets.last_memory
                total_jobs += targets.last_jobs
                host_count += 1

        if host_count == 0:
            return {
                "active_hosts": 0,
                "avg_cpu": 0.0,
                "avg_gpu": 0.0,
                "avg_memory": 0.0,
                "total_jobs": 0,
                "backpressure_factor": self._backpressure_factor,
                "targets": asdict(self._targets),
            }

        return {
            "active_hosts": host_count,
            "avg_cpu": total_cpu / host_count,
            "avg_gpu": total_gpu / host_count,
            "avg_memory": total_memory / host_count,
            "total_jobs": total_jobs,
            "backpressure_factor": self._backpressure_factor,
            "targets": asdict(self._targets),
        }

    def health_check(self) -> "HealthCheckResult":
        """Check resource target manager health for DaemonManager integration.

        Returns:
            HealthCheckResult with status and metrics
        """
        from app.coordination.contracts import HealthCheckResult
        from app.coordination.protocols import CoordinatorStatus

        try:
            summary = self.get_cluster_summary()
            active_hosts = summary.get("active_hosts", 0)
            avg_cpu = summary.get("avg_cpu", 0.0)
            avg_gpu = summary.get("avg_gpu", 0.0)
            backpressure = summary.get("backpressure_factor", 1.0)

            # Health checks:
            # 1. At least one active host reporting
            # 2. Backpressure not too severe (factor > 0.3)
            # 3. CPU not critically high (< 95%)
            issues = []
            if active_hosts == 0:
                issues.append("No active hosts reporting utilization")
            if backpressure < 0.3:
                issues.append(f"Severe backpressure: {backpressure:.2f}")
            if avg_cpu > 95:
                issues.append(f"Critical CPU utilization: {avg_cpu:.1f}%")

            is_healthy = len(issues) == 0

            return HealthCheckResult(
                healthy=is_healthy,
                status=CoordinatorStatus.RUNNING if is_healthy else CoordinatorStatus.DEGRADED,
                message="; ".join(issues) if issues else "",
                details={
                    "active_hosts": active_hosts,
                    "avg_cpu": round(avg_cpu, 1),
                    "avg_gpu": round(avg_gpu, 1),
                    "backpressure_factor": round(backpressure, 2),
                    "total_jobs": summary.get("total_jobs", 0),
                },
            )

        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"Health check failed: {e}",
                details={"error": str(e)},
            )

    # ==========================================
    # Singleton Reset (for testing)
    # ==========================================

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance and clear all internal state.

        This method clears all internal state before resetting the singleton,
        ensuring clean state for tests. Internal state cleared:
        - Host targets cache
        - Utilization history
        - Backpressure factor
        - Adaptive update timestamp
        - Database initialization flags

        December 2025: Added for singleton registry test cleanup.
        """
        from app.coordination.singleton_mixin import SingletonMixin

        with cls._get_lock():
            if cls in SingletonMixin._instances:
                instance = SingletonMixin._instances[cls]

                # Clear host targets cache
                if hasattr(instance, "_host_targets"):
                    instance._host_targets.clear()

                # Clear utilization history
                if hasattr(instance, "_utilization_history"):
                    instance._utilization_history.clear()

                # Reset backpressure factor
                if hasattr(instance, "_backpressure_factor"):
                    instance._backpressure_factor = 1.0

                # Reset adaptive update timestamp
                if hasattr(instance, "_last_adaptive_update"):
                    instance._last_adaptive_update = 0.0

                # Reset DB initialization flags (allow reinit on next access)
                if hasattr(instance, "_db_initialized"):
                    instance._db_initialized = False
                if hasattr(instance, "_readonly_mode"):
                    instance._readonly_mode = False

            # Call parent reset
            super().reset_instance()


# Module-level singleton accessors
_manager: ResourceTargetManager | None = None
_manager_lock = threading.RLock()


def get_resource_targets() -> UtilizationTargets:
    """Get current utilization targets."""
    return ResourceTargetManager.get_instance().get_targets()


def get_host_targets(host: str) -> HostTargets:
    """Get adjusted targets for a specific host."""
    return ResourceTargetManager.get_instance().get_host_targets(host)


def should_scale_up(
    host: str,
    current_cpu: float,
    current_gpu: float = 0.0,
    current_jobs: int = 0,
) -> tuple[bool, str]:
    """Check if a host should scale up."""
    return ResourceTargetManager.get_instance().should_scale_up(
        host, current_cpu, current_gpu, current_jobs
    )


def should_scale_down(
    host: str,
    current_cpu: float,
    current_gpu: float = 0.0,
    current_memory: float = 0.0,
) -> tuple[bool, int, str]:
    """Check if a host should scale down."""
    return ResourceTargetManager.get_instance().should_scale_down(
        host, current_cpu, current_gpu, current_memory
    )


def get_target_job_count(
    host: str,
    cpu_cores: int,
    current_cpu: float,
    current_gpu: float = 0.0,
) -> int:
    """Calculate target job count for a host."""
    return ResourceTargetManager.get_instance().get_target_job_count(
        host, cpu_cores, current_cpu, current_gpu
    )


def get_utilization_score(host: str, current_cpu: float, current_gpu: float = 0.0) -> float:
    """Calculate utilization score for load balancing."""
    return ResourceTargetManager.get_instance().get_utilization_score(
        host, current_cpu, current_gpu
    )


def record_utilization(
    host: str,
    cpu_percent: float,
    gpu_percent: float,
    memory_percent: float,
    job_count: int,
) -> None:
    """Record utilization metrics."""
    ResourceTargetManager.get_instance().record_utilization(
        host, cpu_percent, gpu_percent, memory_percent, job_count
    )


def get_cluster_summary() -> dict:
    """Get cluster utilization summary."""
    return ResourceTargetManager.get_instance().get_cluster_summary()


def set_backpressure(factor: float) -> None:
    """Set backpressure factor (0.0-1.0)."""
    ResourceTargetManager.get_instance().set_backpressure(factor)


def reset_resource_targets() -> None:
    """Reset singleton for testing."""
    ResourceTargetManager.reset_instance()


def select_host_for_task(
    available_hosts: list[str],
    task_resource_type: str,
    host_metrics: dict[str, dict[str, float]] | None = None,
) -> str | None:
    """Select the best host for a task based on its resource requirements.

    This enables CPU and GPU utilization targets to be pursued independently:
    - GPU tasks are assigned to hosts with low GPU utilization
    - CPU tasks are assigned to hosts with low CPU utilization
    - Neither blocks the other

    Args:
        available_hosts: List of host names to consider
        task_resource_type: One of "cpu", "gpu", "hybrid", "io"
        host_metrics: Optional dict of {host: {"cpu_util": %, "gpu_util": %}}
                     If not provided, uses cached metrics from record_utilization()

    Returns:
        Best host name, or None if no suitable host found
    """
    if not available_hosts:
        return None

    manager = ResourceTargetManager.get_instance()
    best_host = None
    best_score = float('inf')  # Lower is better (more available capacity)

    for host in available_hosts:
        # Get current utilization
        if host_metrics and host in host_metrics:
            cpu_util = host_metrics[host].get("cpu_util", 50)
            gpu_util = host_metrics[host].get("gpu_util", 50)
        else:
            # Use cached metrics from manager
            summary = manager.get_cluster_summary()
            host_data = summary.get("hosts", {}).get(host, {})
            cpu_util = host_data.get("cpu_util", 50)
            gpu_util = host_data.get("gpu_util", 50)

        # Calculate score based on task resource type
        if task_resource_type == "gpu":
            # GPU tasks: only consider GPU utilization
            score = gpu_util
        elif task_resource_type == "cpu":
            # CPU tasks: only consider CPU utilization
            score = cpu_util
        elif task_resource_type == "hybrid":
            # Hybrid tasks: consider both, weighted average
            score = (cpu_util + gpu_util) / 2
        else:  # "io" or unknown
            # I/O tasks: prefer hosts with lower overall load
            score = (cpu_util + gpu_util) / 2

        # Prefer hosts below target range (60-80%)
        targets = manager.get_host_targets(host)
        if task_resource_type == "gpu" and gpu_util < targets.gpu_min:
            score -= 20  # Bonus for underutilized GPU
        elif task_resource_type == "cpu" and cpu_util < targets.cpu_min:
            score -= 20  # Bonus for underutilized CPU

        if score < best_score:
            best_score = score
            best_host = host

    return best_host


def get_hosts_for_gpu_tasks(
    available_hosts: list[str],
    max_gpu_util: float = 70.0,
) -> list[str]:
    """Get hosts suitable for GPU tasks (low GPU utilization).

    Returns hosts sorted by GPU availability (lowest utilization first).
    """
    manager = ResourceTargetManager.get_instance()
    summary = manager.get_cluster_summary()
    hosts_data = summary.get("hosts", {})

    suitable = []
    for host in available_hosts:
        host_data = hosts_data.get(host, {})
        gpu_util = host_data.get("gpu_util", 100)  # Assume busy if unknown
        if gpu_util < max_gpu_util:
            suitable.append((host, gpu_util))

    # Sort by GPU utilization (lowest first)
    suitable.sort(key=lambda x: x[1])
    return [h for h, _ in suitable]


def get_hosts_for_cpu_tasks(
    available_hosts: list[str],
    max_cpu_util: float = 70.0,
) -> list[str]:
    """Get hosts suitable for CPU tasks (low CPU utilization).

    Returns hosts sorted by CPU availability (lowest utilization first).
    """
    manager = ResourceTargetManager.get_instance()
    summary = manager.get_cluster_summary()
    hosts_data = summary.get("hosts", {})

    suitable = []
    for host in available_hosts:
        host_data = hosts_data.get(host, {})
        cpu_util = host_data.get("cpu_util", 100)  # Assume busy if unknown
        if cpu_util < max_cpu_util:
            suitable.append((host, cpu_util))

    # Sort by CPU utilization (lowest first)
    suitable.sort(key=lambda x: x[1])
    return [h for h, _ in suitable]


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    "HostTargets",
    # Enums
    "HostTier",
    # Main class
    "ResourceTargetManager",
    # Data classes
    "UtilizationTargets",
    "get_cluster_summary",
    "get_host_targets",
    "get_hosts_for_cpu_tasks",
    "get_hosts_for_gpu_tasks",
    # Functions
    "get_resource_targets",
    "get_target_job_count",
    "get_utilization_score",
    "record_utilization",
    "reset_resource_targets",
    "select_host_for_task",
    "set_backpressure",
    "should_scale_down",
    "should_scale_up",
]
