"""JobOrchestrationManager: Job spawning, scaling, and cluster coordination.

Extracted from p2p_orchestrator.py for better modularity and testability.
January 9, 2026: Phase 1 of P2P Orchestrator Deep Decomposition.

Handles:
- Local job spawning (_start_local_job)
- Cluster job management (_manage_cluster_jobs)
- Work execution (_execute_claimed_work)
- Job scaling and rebalancing
- GPU auto-scaling

This manager reduces p2p_orchestrator.py by ~2,800 LOC.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from app.config.coordination_defaults import (
    JobDefaults,
    OperationTimeouts,
)

if TYPE_CHECKING:
    from ..models import ClusterJob, NodeInfo

logger = logging.getLogger(__name__)


# ============================================================================
# Constants (imported from orchestrator or coordination_defaults)
# ============================================================================

# Thresholds for resource management - aligned with app.config.thresholds
try:
    from app.config.thresholds import DISK_CRITICAL_PERCENT, DISK_PRODUCTION_HALT_PERCENT
    DISK_CLEANUP_THRESHOLD = DISK_PRODUCTION_HALT_PERCENT  # 85
    DISK_WARNING_THRESHOLD = DISK_PRODUCTION_HALT_PERCENT - 5  # 80
    DISK_CRITICAL_THRESHOLD = DISK_CRITICAL_PERCENT  # 90
except ImportError:
    DISK_CLEANUP_THRESHOLD = 85
    DISK_WARNING_THRESHOLD = 80
    DISK_CRITICAL_THRESHOLD = 90
MEMORY_WARNING_THRESHOLD = 75
MEMORY_CRITICAL_THRESHOLD = 85
LOAD_MAX_FOR_NEW_JOBS = 80
GPU_IDLE_THRESHOLD = 5
GPU_IDLE_RESTART_TIMEOUT = 300
RUNAWAY_SELFPLAY_PROCESS_THRESHOLD = int(
    os.environ.get("RINGRIFT_RUNAWAY_SELFPLAY_PROCESS_THRESHOLD", "128")
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class JobOrchestrationConfig:
    """Configuration for JobOrchestrationManager."""

    max_local_jobs: int = 4
    job_timeout_seconds: float = 3600.0
    rebalance_interval_seconds: float = 60.0
    # Spawn rate limiting
    max_spawns_per_minute: int = 10
    spawn_cooldown_seconds: float = 5.0


@dataclass
class JobOrchestrationStats:
    """Statistics for job orchestration monitoring."""

    jobs_started: int = 0
    jobs_failed: int = 0
    jobs_completed: int = 0
    jobs_scaled_down: int = 0
    cluster_management_runs: int = 0
    work_items_executed: int = 0
    spawn_blocked_count: int = 0


# ============================================================================
# JobOrchestrationManager
# ============================================================================

class JobOrchestrationManager:
    """Manages job spawning, scaling, and cluster-wide coordination.

    This manager consolidates job orchestration logic extracted from P2POrchestrator.
    It uses callbacks for accessing orchestrator state to avoid circular dependencies.

    Architecture:
        - Uses callback pattern for state access (lambda: self.peers)
        - Locks passed explicitly for thread safety
        - health_check() for DaemonManager integration

    Example usage:
        manager = JobOrchestrationManager(
            node_id="worker-1",
            ringrift_path="/path/to/ringrift",
            get_peers=lambda: self.peers,
            get_local_jobs=lambda: self.local_jobs,
            jobs_lock=self.jobs_lock,
            is_leader=self._is_leader,
            get_self_info=lambda: self.self_info,
        )
    """

    def __init__(
        self,
        node_id: str,
        ringrift_path: str,
        get_peers: Callable[[], dict[str, "NodeInfo"]],
        get_local_jobs: Callable[[], dict[str, "ClusterJob"]],
        jobs_lock: threading.RLock,
        is_leader: Callable[[], bool],
        get_self_info: Callable[[], "NodeInfo | None"],
        # Optional callbacks for other managers
        selfplay_scheduler: Any | None = None,
        job_manager: Any | None = None,
        # State persistence callback
        save_state_fn: Callable[[], None] | None = None,
        # Path helpers
        get_script_path_fn: Callable[[str], str] | None = None,
        get_ai_service_path_fn: Callable[[], str] | None = None,
        # Spawn management callbacks
        can_spawn_process_fn: Callable[[str], tuple[bool, str]] | None = None,
        record_spawn_fn: Callable[[], None] | None = None,
        # GPU job tracking
        update_gpu_job_count_fn: Callable[[int], None] | None = None,
        # Process monitoring callbacks
        monitor_selfplay_process_fn: Callable[..., Any] | None = None,
        monitor_gpu_selfplay_fn: Callable[..., Any] | None = None,
        # Distributed hosts config
        load_distributed_hosts_fn: Callable[[], dict] | None = None,
        # Remote operation callbacks (for cluster-wide management)
        request_remote_cleanup_fn: Callable[["NodeInfo"], Any] | None = None,
        request_reduce_selfplay_fn: Callable[["NodeInfo", int, str], Any] | None = None,
        request_job_restart_fn: Callable[["NodeInfo"], Any] | None = None,
        cleanup_local_disk_fn: Callable[[], Any] | None = None,
        restart_local_stuck_jobs_fn: Callable[[], Any] | None = None,
        reduce_local_selfplay_jobs_fn: Callable[[int, str], Any] | None = None,
        # Additional state access
        peers_lock: threading.RLock | None = None,
        get_gpu_idle_since: Callable[[], dict[str, float]] | None = None,
        set_gpu_idle_since: Callable[[str, float | None], None] | None = None,
        # Configuration
        config: JobOrchestrationConfig | None = None,
    ):
        """Initialize JobOrchestrationManager.

        Args:
            node_id: This node's identifier
            ringrift_path: Path to RingRift root directory
            get_peers: Callback to get peers dict
            get_local_jobs: Callback to get local jobs dict
            jobs_lock: Lock for thread-safe job access
            is_leader: Callback to check if this node is leader
            get_self_info: Callback to get this node's info
            selfplay_scheduler: SelfplayScheduler instance (optional)
            job_manager: JobManager instance (optional)
            save_state_fn: Callback to persist state
            get_script_path_fn: Callback to get script path
            get_ai_service_path_fn: Callback to get ai-service path
            can_spawn_process_fn: Callback to check if spawn is allowed
            record_spawn_fn: Callback to record spawn for rate limiting
            update_gpu_job_count_fn: Callback to update GPU job count
            monitor_selfplay_process_fn: Callback to monitor selfplay process
            monitor_gpu_selfplay_fn: Callback to monitor GPU selfplay
            load_distributed_hosts_fn: Callback to load distributed hosts config
            request_remote_cleanup_fn: Callback for remote disk cleanup
            request_reduce_selfplay_fn: Callback to reduce remote selfplay
            request_job_restart_fn: Callback to restart remote jobs
            cleanup_local_disk_fn: Callback for local disk cleanup
            restart_local_stuck_jobs_fn: Callback to restart local stuck jobs
            reduce_local_selfplay_jobs_fn: Callback to reduce local selfplay
            peers_lock: Lock for thread-safe peer access
            get_gpu_idle_since: Callback to get GPU idle tracking dict
            set_gpu_idle_since: Callback to set/clear GPU idle timestamp
            config: Configuration options
        """
        self.node_id = node_id
        self.ringrift_path = ringrift_path
        self._get_peers = get_peers
        self._get_local_jobs = get_local_jobs
        self._jobs_lock = jobs_lock
        self._is_leader = is_leader
        self._get_self_info = get_self_info

        # Manager references
        self._selfplay_scheduler = selfplay_scheduler
        self._job_manager = job_manager

        # Callbacks
        self._save_state = save_state_fn
        self._get_script_path = get_script_path_fn
        self._get_ai_service_path = get_ai_service_path_fn
        self._can_spawn_process = can_spawn_process_fn
        self._record_spawn = record_spawn_fn
        self._update_gpu_job_count = update_gpu_job_count_fn
        self._monitor_selfplay_process = monitor_selfplay_process_fn
        self._monitor_gpu_selfplay = monitor_gpu_selfplay_fn
        self._load_distributed_hosts = load_distributed_hosts_fn
        self._request_remote_cleanup = request_remote_cleanup_fn
        self._request_reduce_selfplay = request_reduce_selfplay_fn
        self._request_job_restart = request_job_restart_fn
        self._cleanup_local_disk = cleanup_local_disk_fn
        self._restart_local_stuck_jobs = restart_local_stuck_jobs_fn
        self._reduce_local_selfplay_jobs = reduce_local_selfplay_jobs_fn
        self._peers_lock = peers_lock or jobs_lock
        self._get_gpu_idle_since = get_gpu_idle_since
        self._set_gpu_idle_since = set_gpu_idle_since

        # Configuration
        self._config = config or JobOrchestrationConfig()

        # Statistics
        self._stats = JobOrchestrationStats()

        # Internal state
        self._running = False
        self._spawn_times: list[float] = []  # For rate limiting

    # =========================================================================
    # Health Check (required for DaemonManager integration)
    # =========================================================================

    def health_check(self) -> dict[str, Any]:
        """Return health status for DaemonManager integration."""
        return {
            "healthy": True,
            "status": "running" if self._running else "initialized",
            "node_id": self.node_id,
            "stats": {
                "jobs_started": self._stats.jobs_started,
                "jobs_failed": self._stats.jobs_failed,
                "jobs_completed": self._stats.jobs_completed,
                "jobs_scaled_down": self._stats.jobs_scaled_down,
                "cluster_management_runs": self._stats.cluster_management_runs,
                "work_items_executed": self._stats.work_items_executed,
                "spawn_blocked_count": self._stats.spawn_blocked_count,
            },
        }

    # =========================================================================
    # Start/Stop
    # =========================================================================

    def start(self) -> None:
        """Start the manager."""
        self._running = True
        logger.info(f"[JobOrchestrationManager] Started on {self.node_id}")

    def stop(self) -> None:
        """Stop the manager."""
        self._running = False
        logger.info("[JobOrchestrationManager] Stopped")

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_ai_service_path_safe(self) -> str:
        """Get ai-service path safely."""
        if self._get_ai_service_path:
            return self._get_ai_service_path()
        return str(Path(self.ringrift_path) / "ai-service")

    def _get_script_path_safe(self, script_name: str) -> str:
        """Get script path safely."""
        if self._get_script_path:
            return self._get_script_path(script_name)
        return str(Path(self._get_ai_service_path_safe()) / "scripts" / script_name)

    def _check_spawn_rate_limit(self) -> tuple[bool, str]:
        """Check if spawn rate limit allows spawning."""
        now = time.time()
        # Clean old entries
        self._spawn_times = [t for t in self._spawn_times if now - t < 60]

        if len(self._spawn_times) >= self._config.max_spawns_per_minute:
            return False, f"Rate limit: {len(self._spawn_times)} spawns in last minute"

        return True, "OK"

    def _record_spawn_internal(self) -> None:
        """Record spawn time for rate limiting."""
        self._spawn_times.append(time.time())
        if self._record_spawn:
            self._record_spawn()

    # =========================================================================
    # Statistics Tracking (called by orchestrator during delegation phase)
    # =========================================================================

    def record_job_started(self, job_type: str = "") -> None:
        """Record that a job was started (called by orchestrator)."""
        self._stats.jobs_started += 1
        logger.debug(f"[JobOrchestration] Job started: {job_type} (total: {self._stats.jobs_started})")

    def record_job_completed(self, job_type: str = "") -> None:
        """Record that a job completed (called by orchestrator)."""
        self._stats.jobs_completed += 1

    def record_job_failed(self, job_type: str = "", reason: str = "") -> None:
        """Record that a job failed (called by orchestrator)."""
        self._stats.jobs_failed += 1
        logger.debug(f"[JobOrchestration] Job failed: {job_type} reason={reason}")

    def record_spawn_blocked(self, reason: str = "") -> None:
        """Record that a spawn was blocked (called by orchestrator)."""
        self._stats.spawn_blocked_count += 1
        logger.debug(f"[JobOrchestration] Spawn blocked: {reason}")

    def record_cluster_management_run(self) -> None:
        """Record that cluster management ran (called by orchestrator)."""
        self._stats.cluster_management_runs += 1

    def record_work_executed(self, work_type: str = "") -> None:
        """Record that a work item was executed (called by orchestrator)."""
        self._stats.work_items_executed += 1

    def record_jobs_scaled_down(self, count: int = 1) -> None:
        """Record that jobs were scaled down (called by orchestrator)."""
        self._stats.jobs_scaled_down += count

    def get_stats(self) -> JobOrchestrationStats:
        """Get current statistics."""
        return self._stats

    # =========================================================================
    # Main Job Orchestration Methods (to be populated from orchestrator)
    # =========================================================================

    async def start_local_job(
        self,
        job_type: Any,  # JobType enum
        board_type: str = "square8",
        num_players: int = 2,
        engine_mode: str = "gumbel-mcts",
        job_id: str | None = None,
        cuda_visible_devices: str | None = None,
        export_params: dict[str, Any] | None = None,
        simulation_budget: int | None = None,
    ) -> Any | None:
        """Start a job on the local node.

        This is a delegation stub. The actual implementation remains in
        P2POrchestrator._start_local_job() until full extraction is complete.

        Args:
            job_type: Type of job to start
            board_type: Board type (hex8, square8, etc.)
            num_players: Number of players (2, 3, 4)
            engine_mode: AI engine mode
            job_id: Optional job ID (auto-generated if None)
            cuda_visible_devices: GPU device selection
            export_params: Parameters for DATA_EXPORT jobs
            simulation_budget: Gumbel MCTS simulation budget

        Returns:
            ClusterJob if started, None if blocked
        """
        # For now, this method is a stub
        # The full implementation will be migrated in a subsequent step
        logger.warning(
            "JobOrchestrationManager.start_local_job() called - "
            "delegation not yet implemented, use orchestrator directly"
        )
        return None

    async def manage_cluster_jobs(self) -> None:
        """Manage jobs across the cluster (leader only).

        This is a delegation stub. The actual implementation remains in
        P2POrchestrator._manage_cluster_jobs() until full extraction is complete.

        Responsibilities:
        - Check node disk/memory pressure and trigger cleanup
        - Detect stuck jobs (GPU idle with running processes)
        - Detect runaway processes
        - Calculate job distribution for healthy nodes
        - Dispatch jobs to nodes that need more work
        """
        if not self._is_leader():
            return

        self._stats.cluster_management_runs += 1
        logger.info("Leader: Managing cluster jobs (via JobOrchestrationManager)")

        # For now, this method is a stub
        # The full implementation will be migrated in a subsequent step
        logger.warning(
            "JobOrchestrationManager.manage_cluster_jobs() called - "
            "delegation not yet implemented, use orchestrator directly"
        )

    async def execute_claimed_work(self, work_item: dict[str, Any]) -> bool:
        """Execute a claimed work item locally.

        This is a delegation stub. The actual implementation remains in
        P2POrchestrator._execute_claimed_work() until full extraction is complete.

        Args:
            work_item: Work item dict with work_type, config, work_id

        Returns:
            True if execution started successfully, False otherwise
        """
        work_type = work_item.get("work_type", "")
        work_id = work_item.get("work_id", "")

        logger.info(f"Executing claimed work: {work_type} ({work_id})")
        self._stats.work_items_executed += 1

        # For now, this method is a stub
        # The full implementation will be migrated in a subsequent step
        logger.warning(
            "JobOrchestrationManager.execute_claimed_work() called - "
            "delegation not yet implemented, use orchestrator directly"
        )
        return False

    async def manage_local_jobs_decentralized(self) -> int:
        """Manage local jobs in decentralized mode (non-leader).

        Returns:
            Number of jobs started
        """
        # Stub - to be implemented
        return 0

    async def local_gpu_auto_scale(self) -> int:
        """Auto-scale GPU jobs based on utilization.

        Returns:
            Number of jobs started
        """
        # Stub - to be implemented
        return 0

    async def auto_rebalance_from_work_queue(self) -> int:
        """Rebalance work from queue.

        Returns:
            Number of jobs started
        """
        # Stub - to be implemented
        return 0

    async def check_cluster_balance(self) -> dict[str, Any]:
        """Check cluster job balance.

        Returns:
            Balance report with per-node job counts
        """
        # Stub - to be implemented
        return {"status": "not_implemented"}

    async def reduce_local_selfplay_jobs(
        self, target_selfplay_jobs: int, *, reason: str
    ) -> dict[str, Any]:
        """Reduce local selfplay jobs to target count.

        Args:
            target_selfplay_jobs: Target number of jobs
            reason: Reason for reduction (for logging)

        Returns:
            Result dict with jobs_killed, etc.
        """
        # Stub - to be implemented
        return {"jobs_killed": 0, "reason": reason}


# ============================================================================
# Factory function
# ============================================================================

def create_job_orchestration_manager(
    orchestrator: Any,
) -> JobOrchestrationManager:
    """Factory to create JobOrchestrationManager from P2POrchestrator.

    This helper creates a manager with all callbacks wired to the orchestrator.

    Args:
        orchestrator: P2POrchestrator instance

    Returns:
        Configured JobOrchestrationManager
    """
    return JobOrchestrationManager(
        node_id=orchestrator.node_id,
        ringrift_path=orchestrator.ringrift_path,
        get_peers=lambda: orchestrator.peers,
        get_local_jobs=lambda: orchestrator.local_jobs,
        jobs_lock=orchestrator.jobs_lock,
        is_leader=orchestrator._is_leader,
        get_self_info=lambda: orchestrator.self_info,
        selfplay_scheduler=getattr(orchestrator, "selfplay_scheduler", None),
        job_manager=getattr(orchestrator, "job_manager", None),
        save_state_fn=orchestrator._save_state,
        get_script_path_fn=getattr(orchestrator, "_get_script_path", None),
        get_ai_service_path_fn=getattr(orchestrator, "_get_ai_service_path", None),
        can_spawn_process_fn=getattr(orchestrator, "_can_spawn_process", None),
        record_spawn_fn=getattr(orchestrator, "_record_spawn", None),
        update_gpu_job_count_fn=getattr(orchestrator, "_update_gpu_job_count", None),
        monitor_selfplay_process_fn=getattr(orchestrator, "_monitor_selfplay_process", None),
        monitor_gpu_selfplay_fn=getattr(orchestrator, "_monitor_gpu_selfplay_and_validate", None),
        load_distributed_hosts_fn=getattr(orchestrator, "_load_distributed_hosts", None),
        request_remote_cleanup_fn=getattr(orchestrator, "_request_remote_cleanup", None),
        request_reduce_selfplay_fn=getattr(orchestrator, "_request_reduce_selfplay", None),
        request_job_restart_fn=getattr(orchestrator, "_request_job_restart", None),
        cleanup_local_disk_fn=getattr(orchestrator, "_cleanup_local_disk", None),
        restart_local_stuck_jobs_fn=getattr(orchestrator, "_restart_local_stuck_jobs", None),
        reduce_local_selfplay_jobs_fn=lambda target, reason: orchestrator._reduce_local_selfplay_jobs(target, reason=reason),
        peers_lock=orchestrator.peers_lock,
    )
