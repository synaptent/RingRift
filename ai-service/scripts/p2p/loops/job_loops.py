"""Job Management Loops for P2P Orchestrator.

December 2025: Background loops for job lifecycle management.

Loops:
- JobReaperLoop: Cleans up stale/stuck jobs
- IdleDetectionLoop: Detects idle nodes for potential shutdown
- WorkerPullLoop: Workers poll leader for work (pull model)
- WorkQueueMaintenanceLoop: Leader maintains work queue (cleanup, timeouts)

Usage:
    from scripts.p2p.loops import JobReaperLoop, IdleDetectionLoop

    reaper = JobReaperLoop(
        get_active_jobs=lambda: orchestrator.active_jobs,
        cancel_job=orchestrator.cancel_job,
    )
    await reaper.run_forever()
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from .base import BaseLoop

# Lazy import for event emission (avoid circular imports)
_emit_event = None


def _get_emit_event():
    """Lazy load event emission to avoid circular imports."""
    global _emit_event
    if _emit_event is None:
        try:
            from app.coordination.event_router import emit_event
            _emit_event = emit_event
        except ImportError:
            # Fallback: no-op if event router not available
            _emit_event = lambda *args, **kwargs: None
    return _emit_event


logger = logging.getLogger(__name__)


# Job-type specific stale thresholds (Sprint 17.9, Jan 2026)
# Increased timeouts to prevent killing legitimate long-running jobs
# GPU jobs can take 20-40 min for large batches, training even longer
DEFAULT_STALE_THRESHOLDS: dict[str, float] = {
    "gpu_gumbel": 2400.0,     # 40 min (was 10 min) - large batches take time
    "gpu_policy": 1200.0,     # 20 min (was 10 min) - inference can stall
    "gpu_selfplay": 1800.0,   # 30 min (was 10 min) - general GPU selfplay
    "training": 3600.0,       # 60 min (was 30 min) - training with data loading
    "evaluation": 1200.0,     # 20 min (was 15 min) - gauntlet evaluation
    "cpu_heuristic": 1800.0,  # 30 min - unchanged
    "cpu_gumbel": 1200.0,     # 20 min - unchanged
    "selfplay": 1200.0,       # 20 min (was 15 min) - generic selfplay
    "default": 2400.0,        # 40 min fallback (was 30 min)
}


@dataclass
class JobReaperConfig:
    """Configuration for job reaper loop.

    Supports job-type-specific thresholds via stale_thresholds_by_type.
    GPU jobs use faster thresholds (10-15 min) since issues surface quickly.
    CPU jobs can wait longer (30 min) since they're cheaper.
    """

    # Default fallback threshold (used if job type not in stale_thresholds_by_type)
    stale_job_threshold_seconds: float = 1800.0  # 30 min (reduced from 1 hour)
    stuck_job_threshold_seconds: float = 7200.0  # 2 hours
    max_jobs_to_reap_per_cycle: int = 10
    check_interval_seconds: float = 300.0  # 5 minutes

    # Job-type-specific stale thresholds (P1 - Sprint 6, Jan 2026)
    # Keys: gpu_gumbel, gpu_policy, gpu_selfplay, training, evaluation,
    #       cpu_heuristic, cpu_gumbel, selfplay, default
    stale_thresholds_by_type: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_STALE_THRESHOLDS)
    )

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.stale_job_threshold_seconds <= 0:
            raise ValueError("stale_job_threshold_seconds must be > 0")
        if self.stuck_job_threshold_seconds <= 0:
            raise ValueError("stuck_job_threshold_seconds must be > 0")
        # Removed: stale < stuck validation since stale thresholds are now per-type
        if self.max_jobs_to_reap_per_cycle <= 0:
            raise ValueError("max_jobs_to_reap_per_cycle must be > 0")
        if self.check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be > 0")

    def get_stale_threshold(self, job_type: str) -> float:
        """Get the stale threshold for a specific job type.

        Args:
            job_type: Type of job (e.g., 'gpu_gumbel', 'cpu_heuristic')

        Returns:
            Stale threshold in seconds for this job type.
        """
        # Check for exact match first
        if job_type in self.stale_thresholds_by_type:
            return self.stale_thresholds_by_type[job_type]

        # Check for prefix matches (e.g., "gpu_gumbel_hex8" matches "gpu_gumbel")
        for known_type in self.stale_thresholds_by_type:
            if job_type.startswith(known_type):
                return self.stale_thresholds_by_type[known_type]

        # Check if it's a GPU or CPU job for category fallback
        if "gpu" in job_type.lower():
            return self.stale_thresholds_by_type.get("gpu_selfplay", 600.0)
        if "cpu" in job_type.lower():
            return self.stale_thresholds_by_type.get("cpu_heuristic", 1800.0)

        # Use default
        return self.stale_thresholds_by_type.get(
            "default", self.stale_job_threshold_seconds
        )


class JobReaperLoop(BaseLoop):
    """Background loop that cleans up stale and stuck jobs.

    Identifies jobs that:
    - Have been running too long (stuck)
    - Were claimed but never started (stale)
    - Have no heartbeat for too long (abandoned)

    And either cancels them or marks them for retry.
    """

    def __init__(
        self,
        get_active_jobs: Callable[[], dict[str, Any]],
        cancel_job: Callable[[str], Coroutine[Any, Any, bool]],
        get_job_heartbeats: Callable[[], dict[str, float]] | None = None,
        config: JobReaperConfig | None = None,
    ):
        """Initialize job reaper loop.

        Args:
            get_active_jobs: Callback returning dict of job_id -> job_info
            cancel_job: Async callback to cancel a job by ID
            get_job_heartbeats: Optional callback returning job_id -> last_heartbeat_time
            config: Reaper configuration
        """
        self.config = config or JobReaperConfig()
        super().__init__(
            name="job_reaper",
            interval=self.config.check_interval_seconds,
        )
        self._get_active_jobs = get_active_jobs
        self._cancel_job = cancel_job
        self._get_job_heartbeats = get_job_heartbeats
        self._reap_stats = {
            "stale_jobs_reaped": 0,
            "stuck_jobs_reaped": 0,
            "abandoned_jobs_reaped": 0,
        }

    async def _run_once(self) -> None:
        """Check for and clean up problematic jobs.

        Uses job-type-specific thresholds (P1 - Sprint 6, Jan 2026):
        - GPU jobs (gpu_gumbel, gpu_policy): 10 min threshold
        - Evaluation jobs: 15 min threshold
        - CPU jobs: 30 min threshold
        """
        active_jobs = self._get_active_jobs()
        if not active_jobs:
            return

        now = time.time()
        heartbeats = self._get_job_heartbeats() if self._get_job_heartbeats else {}
        jobs_to_reap: list[tuple[str, str]] = []  # (job_id, reason)

        for job_id, job_info in active_jobs.items():
            # Get job type for threshold lookup
            job_type = job_info.get("job_type", job_info.get("type", "default"))
            stale_threshold = self.config.get_stale_threshold(job_type)

            # Check for stale jobs (claimed but not started)
            started_at = job_info.get("started_at", 0)
            claimed_at = job_info.get("claimed_at", 0)
            status = job_info.get("status", "")

            if status == "claimed" and not started_at:
                age = now - claimed_at if claimed_at else now
                if age > stale_threshold:
                    jobs_to_reap.append((job_id, "stale"))
                    logger.debug(
                        f"[JobReaper] {job_id} stale: age={age:.0f}s > threshold={stale_threshold:.0f}s "
                        f"(type={job_type})"
                    )
                    continue

            # Check for stuck jobs (running too long)
            if started_at and status in ("running", "started"):
                runtime = now - started_at
                if runtime > self.config.stuck_job_threshold_seconds:
                    jobs_to_reap.append((job_id, "stuck"))
                    continue

            # Check for abandoned jobs (no heartbeat) - uses job-type threshold
            if job_id in heartbeats:
                last_heartbeat = heartbeats[job_id]
                silence = now - last_heartbeat
                if silence > stale_threshold:
                    jobs_to_reap.append((job_id, "abandoned"))
                    logger.debug(
                        f"[JobReaper] {job_id} abandoned: silence={silence:.0f}s > threshold={stale_threshold:.0f}s "
                        f"(type={job_type})"
                    )

        # Reap jobs up to limit
        reaped_count = 0
        for job_id, reason in jobs_to_reap[:self.config.max_jobs_to_reap_per_cycle]:
            try:
                success = await self._cancel_job(job_id)
                if success:
                    reaped_count += 1
                    self._reap_stats[f"{reason}_jobs_reaped"] += 1
                    logger.info(f"[JobReaper] Reaped {reason} job: {job_id}")
            except Exception as e:
                logger.warning(f"[JobReaper] Failed to reap job {job_id}: {e}")

        if reaped_count > 0:
            logger.info(f"[JobReaper] Reaped {reaped_count} jobs this cycle")

    def get_reap_stats(self) -> dict[str, Any]:
        """Get reaping statistics."""
        return {
            **self._reap_stats,
            **self.stats.to_dict(),
        }

    def health_check(self) -> Any:
        """Check loop health with job reaper-specific status.

        Jan 2026: Added for DaemonManager integration.
        Reports reap statistics and identifies potential issues with stuck jobs.

        Returns:
            HealthCheckResult with job reaper status
        """
        try:
            from app.coordination.protocols import CoordinatorStatus, HealthCheckResult
        except ImportError:
            # Fallback if protocols not available
            return {
                "healthy": self.running,
                "status": "running" if self.running else "stopped",
                "message": f"JobReaperLoop {'running' if self.running else 'stopped'}",
                "details": self.get_reap_stats(),
            }

        # Not running
        if not self.running:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="JobReaperLoop is stopped",
                details={"running": False},
            )

        # Calculate total reaped
        total_reaped = (
            self._reap_stats.get("stale_jobs_reaped", 0)
            + self._reap_stats.get("stuck_jobs_reaped", 0)
            + self._reap_stats.get("abandoned_jobs_reaped", 0)
        )

        # Check active jobs for current state
        active_jobs = {}
        try:
            active_jobs = self._get_active_jobs()
        except Exception:
            pass

        # Healthy - report stats
        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"JobReaperLoop healthy ({total_reaped} jobs reaped total)",
            details={
                "stale_jobs_reaped": self._reap_stats.get("stale_jobs_reaped", 0),
                "stuck_jobs_reaped": self._reap_stats.get("stuck_jobs_reaped", 0),
                "abandoned_jobs_reaped": self._reap_stats.get("abandoned_jobs_reaped", 0),
                "active_jobs_count": len(active_jobs),
                "check_interval": self.interval,
            },
        )


@dataclass
class IdleDetectionConfig:
    """Configuration for idle detection loop."""

    gpu_idle_threshold_percent: float = 10.0  # GPU utilization below this = idle
    idle_duration_threshold_seconds: float = 60.0  # 1 minute (reduced from 15 min for faster dispatch)
    check_interval_seconds: float = 30.0  # Check every 30 seconds
    min_nodes_to_keep: int = 2  # Never flag last N nodes as idle

    # Zombie detection: nodes with jobs but 0% GPU (stuck processes)
    zombie_gpu_threshold_percent: float = 5.0  # GPU below this with jobs = zombie
    zombie_duration_threshold_seconds: float = 600.0  # 10 minutes of zombie state

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.gpu_idle_threshold_percent <= 0:
            raise ValueError("gpu_idle_threshold_percent must be > 0")
        if self.gpu_idle_threshold_percent > 100:
            raise ValueError("gpu_idle_threshold_percent must be <= 100")
        if self.idle_duration_threshold_seconds <= 0:
            raise ValueError("idle_duration_threshold_seconds must be > 0")
        if self.check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be > 0")
        if self.min_nodes_to_keep < 0:
            raise ValueError("min_nodes_to_keep must be >= 0")
        if self.zombie_gpu_threshold_percent <= 0:
            raise ValueError("zombie_gpu_threshold_percent must be > 0")
        if self.zombie_gpu_threshold_percent > 100:
            raise ValueError("zombie_gpu_threshold_percent must be <= 100")
        if self.zombie_duration_threshold_seconds <= 0:
            raise ValueError("zombie_duration_threshold_seconds must be > 0")


class IdleDetectionLoop(BaseLoop):
    """Background loop that detects idle nodes and triggers selfplay on them.

    Monitors GPU utilization and starts selfplay on nodes that have been
    idle for too long. Only runs on the cluster leader.
    """

    def __init__(
        self,
        get_role: Callable[[], str] | None = None,
        get_peers: Callable[[], dict[str, Any]] | None = None,
        get_work_queue: Callable[[], Any] | None = None,
        on_idle_detected: Callable[[Any, float], Coroutine[Any, Any, None]] | None = None,
        on_zombie_detected: Callable[[Any, float], Coroutine[Any, Any, None]] | None = None,
        config: IdleDetectionConfig | None = None,
        # Legacy parameters for backward compatibility
        get_node_metrics: Callable[[], dict[str, dict[str, Any]]] | None = None,
    ):
        """Initialize idle detection loop.

        Args:
            get_role: Callback returning node role ("leader", "follower", etc.)
            get_peers: Callback returning dict of node_id -> peer info
            get_work_queue: Callback returning work queue (to check for pending work)
            on_idle_detected: Optional async callback (peer, idle_duration) - auto-start selfplay
            on_zombie_detected: Optional async callback (peer, zombie_duration) - handle stuck processes
            config: Detection configuration
            get_node_metrics: Legacy param - if provided, used instead of get_peers
        """
        self.config = config or IdleDetectionConfig()
        super().__init__(
            name="idle_detection",
            interval=self.config.check_interval_seconds,
        )
        self._get_role = get_role
        self._get_peers = get_peers
        self._get_work_queue = get_work_queue
        self._on_idle_detected = on_idle_detected
        self._on_zombie_detected = on_zombie_detected
        # Legacy support
        self._get_node_metrics = get_node_metrics
        self._idle_since: dict[str, float] = {}  # node_id -> timestamp when became idle
        self._zombie_since: dict[str, float] = {}  # node_id -> timestamp when became zombie
        self._detected_count = 0
        self._zombie_detected_count = 0
        self._skipped_not_leader = 0

    async def _run_once(self) -> None:
        """Check for idle nodes and trigger selfplay."""
        # Only run on leader
        if self._get_role:
            role = self._get_role()
            if role != "leader":
                self._skipped_not_leader += 1
                return

        # Get peer metrics
        if self._get_node_metrics:
            # Legacy path
            peers = self._get_node_metrics()
        elif self._get_peers:
            peers = self._get_peers()
        else:
            return

        if not peers:
            return

        now = time.time()
        gpu_peers = {}

        # Filter to GPU peers and extract metrics
        for node_id, peer_info in peers.items():
            # Handle both dict and object forms
            if hasattr(peer_info, "has_gpu"):
                has_gpu = peer_info.has_gpu
                gpu_util = getattr(peer_info, "gpu_percent", 0) or 0
                selfplay_jobs = getattr(peer_info, "selfplay_jobs", 0) or 0
            else:
                has_gpu = peer_info.get("has_gpu", False)
                gpu_util = peer_info.get("gpu_percent", 0) or peer_info.get("gpu_utilization", 0) or 0
                selfplay_jobs = peer_info.get("selfplay_jobs", 0) or 0

            if has_gpu:
                gpu_peers[node_id] = {
                    "peer": peer_info,
                    "gpu_utilization": gpu_util,
                    "selfplay_jobs": selfplay_jobs,
                }

        active_nodes = len(gpu_peers)
        if active_nodes == 0:
            return

        for node_id, metrics in gpu_peers.items():
            gpu_util = metrics["gpu_utilization"]
            selfplay_jobs = metrics["selfplay_jobs"]

            # Node is idle if GPU < threshold AND no selfplay jobs
            is_idle = gpu_util < self.config.gpu_idle_threshold_percent and selfplay_jobs == 0

            if is_idle:
                if node_id not in self._idle_since:
                    self._idle_since[node_id] = now
                    logger.debug(f"[IdleDetection] Node {node_id} became idle (GPU: {gpu_util}%, jobs: {selfplay_jobs})")

                # Check if idle long enough
                idle_duration = now - self._idle_since[node_id]
                if idle_duration >= self.config.idle_duration_threshold_seconds:
                    # Don't flag if we're at minimum nodes
                    non_idle_count = active_nodes - len([
                        n for n in self._idle_since
                        if now - self._idle_since[n] >= self.config.idle_duration_threshold_seconds
                    ])
                    if non_idle_count >= self.config.min_nodes_to_keep:
                        peer = metrics["peer"]
                        if self._on_idle_detected:
                            try:
                                await self._on_idle_detected(peer, idle_duration)
                                self._detected_count += 1
                                logger.info(
                                    f"[IdleDetection] Triggered selfplay on {node_id} (idle for {idle_duration:.0f}s)"
                                )
                                # Remove from idle tracking after triggering
                                del self._idle_since[node_id]
                            except Exception as e:
                                logger.warning(f"[IdleDetection] Callback failed for {node_id}: {e}")
                        else:
                            logger.info(
                                f"[IdleDetection] Node {node_id} idle for {idle_duration:.0f}s (no callback configured)"
                            )
            else:
                # Node is active, remove from idle tracking
                if node_id in self._idle_since:
                    del self._idle_since[node_id]
                    logger.debug(f"[IdleDetection] Node {node_id} became active (GPU: {gpu_util}%)")

            # Zombie detection: node has jobs but GPU is nearly idle (stuck processes)
            is_zombie = (
                selfplay_jobs > 0
                and gpu_util < self.config.zombie_gpu_threshold_percent
            )

            if is_zombie:
                if node_id not in self._zombie_since:
                    self._zombie_since[node_id] = now
                    logger.warning(
                        f"[ZombieDetection] Node {node_id} may have zombie processes "
                        f"(jobs: {selfplay_jobs}, GPU: {gpu_util}%)"
                    )

                zombie_duration = now - self._zombie_since[node_id]
                if zombie_duration >= self.config.zombie_duration_threshold_seconds:
                    peer = metrics["peer"]
                    if self._on_zombie_detected:
                        try:
                            await self._on_zombie_detected(peer, zombie_duration)
                            self._zombie_detected_count += 1
                            logger.warning(
                                f"[ZombieDetection] Triggered zombie handler on {node_id} "
                                f"(zombie for {zombie_duration:.0f}s, jobs: {selfplay_jobs})"
                            )
                            # Don't remove from tracking - keep monitoring until state changes
                        except Exception as e:
                            logger.warning(f"[ZombieDetection] Callback failed for {node_id}: {e}")
                    else:
                        logger.warning(
                            f"[ZombieDetection] Node {node_id} has zombie processes for {zombie_duration:.0f}s "
                            f"(jobs: {selfplay_jobs}, GPU: {gpu_util}%) - no callback configured"
                        )
            else:
                # Node is not a zombie, remove from tracking
                if node_id in self._zombie_since:
                    del self._zombie_since[node_id]
                    logger.info(f"[ZombieDetection] Node {node_id} recovered from zombie state")

    def get_idle_nodes(self) -> dict[str, float]:
        """Get currently tracked idle nodes and their idle duration."""
        now = time.time()
        return {
            node_id: now - idle_since
            for node_id, idle_since in self._idle_since.items()
        }

    def get_zombie_nodes(self) -> dict[str, float]:
        """Get currently tracked zombie nodes and their zombie duration."""
        now = time.time()
        return {
            node_id: now - zombie_since
            for node_id, zombie_since in self._zombie_since.items()
        }

    def get_detection_stats(self) -> dict[str, Any]:
        """Get detection statistics."""
        return {
            "currently_idle": len(self._idle_since),
            "currently_zombie": len(self._zombie_since),
            "total_detections": self._detected_count,
            "total_zombie_detections": self._zombie_detected_count,
            "skipped_not_leader": self._skipped_not_leader,
            "idle_nodes": list(self._idle_since.keys()),
            "zombie_nodes": list(self._zombie_since.keys()),
            **self.stats.to_dict(),
        }


@dataclass
class PredictiveScalingConfig:
    """Configuration for predictive idle scaling loop.

    January 2026 Sprint 6: Spawns jobs BEFORE nodes become idle to minimize launch latency.
    Addresses 5-10 min launch lag from reactive-only idle detection.
    """

    check_interval_seconds: float = 30.0  # Check every 30 seconds
    queue_depth_threshold: int = 50  # Start preemptive spawning when queue > this
    approaching_idle_threshold_percent: float = 20.0  # GPU util below this = approaching idle
    approaching_idle_duration_seconds: float = 30.0  # How long at low util before considered approaching
    min_jobs_to_spawn_preemptively: int = 1  # Minimum jobs to spawn per cycle
    max_jobs_to_spawn_preemptively: int = 5  # Maximum jobs to spawn per cycle
    skip_nodes_with_pending_jobs: bool = True  # Don't spawn to nodes with pending work

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be > 0")
        if self.queue_depth_threshold < 0:
            raise ValueError("queue_depth_threshold must be >= 0")
        if self.approaching_idle_threshold_percent <= 0:
            raise ValueError("approaching_idle_threshold_percent must be > 0")
        if self.approaching_idle_threshold_percent > 100:
            raise ValueError("approaching_idle_threshold_percent must be <= 100")
        if self.approaching_idle_duration_seconds <= 0:
            raise ValueError("approaching_idle_duration_seconds must be > 0")
        if self.min_jobs_to_spawn_preemptively <= 0:
            raise ValueError("min_jobs_to_spawn_preemptively must be > 0")
        if self.max_jobs_to_spawn_preemptively < self.min_jobs_to_spawn_preemptively:
            raise ValueError("max_jobs_to_spawn_preemptively must be >= min_jobs_to_spawn_preemptively")


class PredictiveScalingLoop(BaseLoop):
    """Background loop that spawns jobs preemptively before nodes become idle.

    January 2026 Sprint 6: Part of the Predictive Idle Scaling system.

    Unlike IdleDetectionLoop which reacts AFTER nodes are idle, this loop:
    1. Monitors queue depth to see if there's enough work
    2. Identifies nodes "approaching idle" (low GPU, no pending work)
    3. Preemptively spawns jobs to those nodes to minimize launch latency

    Expected impact: -5-10 min launch latency reduction.
    """

    def __init__(
        self,
        get_role: Callable[[], str] | None = None,
        get_peers: Callable[[], dict[str, Any]] | None = None,
        get_queue_depth: Callable[[], int] | None = None,
        get_pending_jobs_for_node: Callable[[str], int] | None = None,
        spawn_preemptive_job: Callable[[Any], Coroutine[Any, Any, bool]] | None = None,
        config: PredictiveScalingConfig | None = None,
    ):
        """Initialize predictive scaling loop.

        Args:
            get_role: Callback returning node role ("leader", "follower", etc.)
            get_peers: Callback returning dict of node_id -> peer info
            get_queue_depth: Callback returning current work queue depth
            get_pending_jobs_for_node: Callback returning pending job count for a node
            spawn_preemptive_job: Async callback to spawn a preemptive job on a node
            config: Scaling configuration
        """
        self.config = config or PredictiveScalingConfig()
        super().__init__(
            name="predictive_scaling",
            interval=self.config.check_interval_seconds,
        )
        self._get_role = get_role
        self._get_peers = get_peers
        self._get_queue_depth = get_queue_depth
        self._get_pending_jobs_for_node = get_pending_jobs_for_node
        self._spawn_preemptive_job = spawn_preemptive_job

        # Track nodes approaching idle state
        self._approaching_idle_since: dict[str, float] = {}

        # Statistics
        self._preemptive_spawns = 0
        self._skipped_low_queue = 0
        self._skipped_not_leader = 0

    async def _run_once(self) -> None:
        """Check for nodes approaching idle and spawn jobs preemptively."""
        # Only run on leader
        if self._get_role:
            role = self._get_role()
            if role != "leader":
                self._skipped_not_leader += 1
                return

        # Check queue depth - only preemptively spawn if queue has work
        queue_depth = 0
        if self._get_queue_depth:
            try:
                queue_depth = self._get_queue_depth()
            except Exception as e:
                logger.debug(f"[PredictiveScaling] Queue depth check failed: {e}")
                return

        if queue_depth < self.config.queue_depth_threshold:
            self._skipped_low_queue += 1
            return

        # Get peer metrics
        if not self._get_peers:
            return

        peers = self._get_peers()
        if not peers:
            return

        now = time.time()
        nodes_approaching_idle: list[tuple[str, Any, float]] = []

        # Find nodes approaching idle state
        for node_id, peer_info in peers.items():
            # Extract GPU metrics
            if hasattr(peer_info, "has_gpu"):
                has_gpu = peer_info.has_gpu
                gpu_util = getattr(peer_info, "gpu_percent", 0) or 0
                selfplay_jobs = getattr(peer_info, "selfplay_jobs", 0) or 0
            else:
                has_gpu = peer_info.get("has_gpu", False)
                gpu_util = peer_info.get("gpu_percent", 0) or peer_info.get("gpu_utilization", 0) or 0
                selfplay_jobs = peer_info.get("selfplay_jobs", 0) or 0

            if not has_gpu:
                continue

            # Check for pending jobs (jobs dispatched but not yet started)
            pending_jobs = 0
            if self.config.skip_nodes_with_pending_jobs and self._get_pending_jobs_for_node:
                try:
                    pending_jobs = self._get_pending_jobs_for_node(node_id)
                except Exception:
                    pending_jobs = 0

            # Node is approaching idle if:
            # 1. GPU util is low (but not zero, which means idle)
            # 2. Currently running a job (selfplay_jobs > 0)
            # 3. No pending jobs waiting to be started
            is_approaching_idle = (
                0 < gpu_util < self.config.approaching_idle_threshold_percent
                and selfplay_jobs > 0
                and pending_jobs == 0
            )

            if is_approaching_idle:
                if node_id not in self._approaching_idle_since:
                    self._approaching_idle_since[node_id] = now
                    logger.debug(
                        f"[PredictiveScaling] Node {node_id} approaching idle "
                        f"(GPU: {gpu_util}%, jobs: {selfplay_jobs})"
                    )

                approaching_duration = now - self._approaching_idle_since[node_id]
                if approaching_duration >= self.config.approaching_idle_duration_seconds:
                    nodes_approaching_idle.append((node_id, peer_info, approaching_duration))
            else:
                # Clear tracking if no longer approaching idle
                if node_id in self._approaching_idle_since:
                    del self._approaching_idle_since[node_id]

        # Spawn preemptive jobs to nodes approaching idle
        if nodes_approaching_idle and self._spawn_preemptive_job:
            # Sort by how long they've been approaching idle (longest first)
            nodes_approaching_idle.sort(key=lambda x: x[2], reverse=True)

            # Limit spawns per cycle
            max_spawns = min(
                self.config.max_jobs_to_spawn_preemptively,
                len(nodes_approaching_idle),
                queue_depth // 10 + 1,  # Scale with queue depth
            )

            spawned_count = 0
            for node_id, peer, duration in nodes_approaching_idle[:max_spawns]:
                try:
                    success = await self._spawn_preemptive_job(peer)
                    if success:
                        spawned_count += 1
                        self._preemptive_spawns += 1
                        # Clear from approaching tracking since we spawned
                        if node_id in self._approaching_idle_since:
                            del self._approaching_idle_since[node_id]
                        logger.info(
                            f"[PredictiveScaling] Preemptively spawned job on {node_id} "
                            f"(approaching idle for {duration:.0f}s)"
                        )
                except Exception as e:
                    logger.warning(f"[PredictiveScaling] Failed to spawn on {node_id}: {e}")

            if spawned_count > 0:
                logger.info(
                    f"[PredictiveScaling] Spawned {spawned_count} preemptive jobs "
                    f"(queue depth: {queue_depth})"
                )

    def get_scaling_stats(self) -> dict[str, Any]:
        """Get predictive scaling statistics."""
        return {
            "preemptive_spawns": self._preemptive_spawns,
            "skipped_low_queue": self._skipped_low_queue,
            "skipped_not_leader": self._skipped_not_leader,
            "nodes_approaching_idle": len(self._approaching_idle_since),
            "approaching_idle_nodes": list(self._approaching_idle_since.keys()),
            **self.stats.to_dict(),
        }


@dataclass
class WorkerPullConfig:
    """Configuration for worker pull loop."""

    pull_interval_seconds: float = 30.0
    # Jan 2, 2026: Raised from 15% to 90% - now used as GPU overload guard,
    # not primary capacity limiter. Slot-based claiming is the primary mechanism.
    gpu_idle_threshold_percent: float = 90.0
    cpu_idle_threshold_percent: float = 30.0
    initial_delay_seconds: float = 30.0

    # Jan 2, 2026: Slot-based capacity management
    # Allows work queue claiming to coexist with legacy selfplay processes
    enable_slot_based_claiming: bool = True
    default_max_selfplay_slots: int = 8
    min_available_slots_to_claim: int = 1

    # Jan 4, 2026: Autonomous queue fallback (Phase 2 P2P Resilience)
    # When enabled, workers will try autonomous queue when leader is unavailable
    enable_autonomous_fallback: bool = True

    # Jan 4, 2026: WorkDiscoveryManager integration (Phase 5 P2P Resilience)
    # When enabled, uses multi-channel work discovery instead of simple leader check
    enable_work_discovery_manager: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.pull_interval_seconds <= 0:
            raise ValueError("pull_interval_seconds must be > 0")
        if self.gpu_idle_threshold_percent <= 0:
            raise ValueError("gpu_idle_threshold_percent must be > 0")
        if self.gpu_idle_threshold_percent > 100:
            raise ValueError("gpu_idle_threshold_percent must be <= 100")
        if self.cpu_idle_threshold_percent <= 0:
            raise ValueError("cpu_idle_threshold_percent must be > 0")
        if self.cpu_idle_threshold_percent > 100:
            raise ValueError("cpu_idle_threshold_percent must be <= 100")
        if self.initial_delay_seconds < 0:
            raise ValueError("initial_delay_seconds must be >= 0")
        if self.default_max_selfplay_slots <= 0:
            raise ValueError("default_max_selfplay_slots must be > 0")
        if self.min_available_slots_to_claim <= 0:
            raise ValueError("min_available_slots_to_claim must be > 0")


class WorkerPullLoop(BaseLoop):
    """Background loop for workers to poll leader for work (pull model).

    This implements a worker pull model where nodes periodically check
    if they are idle and pull work from the leader's work queue.

    Benefits:
    - Workers claim work at their own pace
    - Naturally load balances across the cluster
    - Works with NAT-blocked nodes (they initiate connections)
    - No need to track worker connectivity for pushing

    Only runs on non-leader nodes.
    """

    def __init__(
        self,
        is_leader: Callable[[], bool],
        get_leader_id: Callable[[], str | None],
        get_self_metrics: Callable[[], dict[str, Any]],
        claim_work_from_leader: Callable[[list[str]], Coroutine[Any, Any, dict[str, Any] | None]],
        execute_work: Callable[[dict[str, Any]], Coroutine[Any, Any, bool]],
        report_work_result: Callable[[dict[str, Any], bool], Coroutine[Any, Any, None]],
        get_allowed_work_types: Callable[[], list[str]] | None = None,
        pop_autonomous_work: Callable[[], Coroutine[Any, Any, dict[str, Any] | None]] | None = None,
        get_work_discovery_manager: Callable[[], Any] | None = None,  # Phase 5: Multi-channel discovery
        config: WorkerPullConfig | None = None,
    ):
        """Initialize worker pull loop.

        Args:
            is_leader: Callback returning True if this node is leader
            get_leader_id: Callback returning current leader node ID
            get_self_metrics: Callback returning self node metrics (gpu_percent, cpu_percent, etc.)
            claim_work_from_leader: Async callback to claim work from leader
            execute_work: Async callback to execute claimed work
            report_work_result: Async callback to report work completion/failure
            get_allowed_work_types: Optional callback returning allowed work types
            pop_autonomous_work: Optional async callback to get work from autonomous queue (Phase 2)
            get_work_discovery_manager: Optional callback to get WorkDiscoveryManager instance (Phase 5)
            config: Loop configuration
        """
        self.config = config or WorkerPullConfig()
        super().__init__(
            name="worker_pull",
            interval=self.config.pull_interval_seconds,
        )
        self._is_leader = is_leader
        self._get_leader_id = get_leader_id
        self._get_self_metrics = get_self_metrics
        self._claim_work = claim_work_from_leader
        self._execute_work = execute_work
        self._report_result = report_work_result
        self._get_allowed_work_types = get_allowed_work_types
        self._pop_autonomous_work = pop_autonomous_work
        self._get_work_discovery_manager = get_work_discovery_manager  # Phase 5

        # Statistics
        self._work_claimed = 0
        self._work_completed = 0
        self._work_failed = 0
        self._skipped_leader = 0
        self._skipped_busy = 0
        self._autonomous_work_claimed = 0  # Jan 4, 2026: Track autonomous queue usage
        # Jan 4, 2026: Phase 5 - Track work discovery channel usage
        self._work_discovery_stats: dict[str, int] = {
            "leader": 0,
            "peer": 0,
            "autonomous": 0,
            "direct": 0,
        }
        self._last_discovery_channel: str = "unknown"

    async def _on_start(self) -> None:
        """Initial delay for cluster stabilization."""
        logger.info("Worker pull loop starting...")
        await asyncio.sleep(self.config.initial_delay_seconds)
        logger.info("Worker pull loop started")

    async def _run_once(self) -> None:
        """Check if idle and pull work from leader or autonomous queue."""
        # Skip if we are the leader (leader pushes, doesn't pull)
        if self._is_leader():
            self._skipped_leader += 1
            return

        # Jan 4, 2026: Phase 5 - Try WorkDiscoveryManager first if enabled
        # This provides multi-channel work discovery (leader → peer → autonomous → direct)
        discovery_manager = None
        use_work_discovery = False
        if self.config.enable_work_discovery_manager and self._get_work_discovery_manager:
            try:
                discovery_manager = self._get_work_discovery_manager()
                if discovery_manager:
                    use_work_discovery = True
            except Exception:
                pass  # Fall back to legacy behavior

        # Check if leader is available (for legacy path)
        leader_id = self._get_leader_id()
        use_autonomous_fallback = False

        if not use_work_discovery and not leader_id:
            # Legacy: No leader and no discovery manager - try autonomous queue fallback
            if self.config.enable_autonomous_fallback and self._pop_autonomous_work:
                use_autonomous_fallback = True
                logger.debug("[WorkerPull] No leader, trying autonomous queue fallback")
            else:
                return  # No fallback available

        # Check if we're idle enough to take on work
        metrics = self._get_self_metrics()
        gpu_percent = float(metrics.get("gpu_percent", 0) or 0)
        cpu_percent = float(metrics.get("cpu_percent", 0) or 0)
        training_jobs = int(metrics.get("training_jobs", 0) or 0)
        has_gpu = bool(metrics.get("has_gpu", False))

        # Jan 2, 2026: Get slot-based capacity metrics
        # selfplay_jobs counts ALL processes (both work queue and legacy) via pgrep
        selfplay_jobs = int(metrics.get("selfplay_jobs", 0) or 0)
        max_slots = int(
            metrics.get("max_selfplay_slots", self.config.default_max_selfplay_slots)
            or self.config.default_max_selfplay_slots
        )
        available_slots = max(0, max_slots - selfplay_jobs)

        # Don't pull work if already running training
        if training_jobs > 0:
            self._skipped_busy += 1
            return

        # Jan 2, 2026: Slot-based capacity check with GPU guard rail
        # This allows work queue claiming to coexist with legacy selfplay processes
        if self.config.enable_slot_based_claiming and has_gpu:
            has_capacity = available_slots >= self.config.min_available_slots_to_claim
            gpu_safe = gpu_percent < self.config.gpu_idle_threshold_percent

            if not has_capacity:
                self._skipped_busy += 1
                logger.debug(
                    f"[WorkerPull] No slots available: {selfplay_jobs}/{max_slots} "
                    f"(GPU: {gpu_percent:.1f}%)"
                )
                return

            if not gpu_safe:
                self._skipped_busy += 1
                logger.debug(
                    f"[WorkerPull] GPU overloaded ({gpu_percent:.1f}% >= "
                    f"{self.config.gpu_idle_threshold_percent}%), "
                    f"slots: {available_slots}/{max_slots}"
                )
                return
        else:
            # Fallback to legacy CPU% check for non-GPU nodes
            if cpu_percent >= self.config.cpu_idle_threshold_percent:
                self._skipped_busy += 1
                return

        # Get allowed work types
        capabilities = ["selfplay", "training", "gpu_cmaes", "tournament"]
        if self._get_allowed_work_types:
            try:
                capabilities = self._get_allowed_work_types()
            except (RuntimeError, ValueError, TypeError, AttributeError):
                pass  # Use default capabilities on callback failure

        # Try to get work from appropriate source
        work_item: dict[str, Any] | None = None
        work_source = "leader"

        # Jan 4, 2026: Phase 5 - Use WorkDiscoveryManager for multi-channel discovery
        if use_work_discovery and discovery_manager:
            try:
                result = await discovery_manager.discover_work(capabilities)
                if result.work_item:
                    work_item = result.work_item
                    work_source = result.channel.value  # "leader", "peer", "autonomous", "direct"
                    self._work_discovery_stats[work_source] = (
                        self._work_discovery_stats.get(work_source, 0) + 1
                    )
                    self._last_discovery_channel = work_source
                    self._work_claimed += 1
                    if work_source == "autonomous":
                        self._autonomous_work_claimed += 1
            except Exception as e:
                logger.debug(f"[WorkerPull] WorkDiscoveryManager error: {e}")
                # Fall through to legacy behavior

        # Legacy fallback if discovery manager didn't find work
        if not work_item:
            if use_autonomous_fallback:
                # Jan 4, 2026: Use autonomous queue when leader unavailable
                try:
                    work_item = await self._pop_autonomous_work()
                    if work_item:
                        work_source = "autonomous"
                        self._autonomous_work_claimed += 1
                except Exception as e:
                    logger.debug(f"[WorkerPull] Autonomous queue error: {e}")
            elif not use_work_discovery:
                # Normal path: claim from leader (skip if already tried via discovery)
                work_item = await self._claim_work(capabilities)
                if work_item:
                    self._work_claimed += 1
                elif self.config.enable_autonomous_fallback and self._pop_autonomous_work:
                    # Jan 4, 2026: Leader available but no work - try autonomous fallback
                    try:
                        work_item = await self._pop_autonomous_work()
                        if work_item:
                            work_source = "autonomous"
                            self._autonomous_work_claimed += 1
                    except Exception as e:
                        logger.debug(f"[WorkerPull] Autonomous fallback error: {e}")

        if work_item:
            work_id = work_item.get("work_id", "unknown")
            work_type = work_item.get("work_type", "unknown")
            logger.info(f"[WorkerPull] Claimed work {work_id}: {work_type} (source: {work_source})")

            # Execute the work
            try:
                success = await self._execute_work(work_item)
                if success:
                    self._work_completed += 1
                else:
                    self._work_failed += 1

                # Report completion/failure (only to leader if we have one)
                if work_source == "leader" or leader_id:
                    await self._report_result(work_item, success)
            except Exception as e:
                self._work_failed += 1
                logger.error(f"[WorkerPull] Error executing work {work_id}: {e}")
                if work_source == "leader" or leader_id:
                    await self._report_result(work_item, False)

    def get_pull_stats(self) -> dict[str, Any]:
        """Get worker pull statistics."""
        return {
            "work_claimed": self._work_claimed,
            "work_completed": self._work_completed,
            "work_failed": self._work_failed,
            "skipped_leader": self._skipped_leader,
            "skipped_busy": self._skipped_busy,
            "autonomous_work_claimed": self._autonomous_work_claimed,  # Jan 4, 2026
            # Jan 4, 2026: Phase 5 - Work discovery channel stats
            "work_discovery_stats": self._work_discovery_stats.copy(),
            "last_discovery_channel": self._last_discovery_channel,
            **self.stats.to_dict(),
        }

    def health_check(self) -> Any:
        """Check loop health with worker pull-specific status.

        Jan 2026: Added for DaemonManager integration.
        Reports work claiming and completion rates for worker health.

        Returns:
            HealthCheckResult with worker pull status
        """
        try:
            from app.coordination.protocols import CoordinatorStatus, HealthCheckResult
        except ImportError:
            # Fallback if protocols not available
            return {
                "healthy": self.running,
                "status": "running" if self.running else "stopped",
                "message": f"WorkerPullLoop {'running' if self.running else 'stopped'}",
                "details": self.get_pull_stats(),
            }

        # Not running
        if not self.running:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="WorkerPullLoop is stopped",
                details={"running": False},
            )

        # Workers skip if they're the leader - that's expected
        if self._is_leader():
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message="WorkerPullLoop: This node is leader (pull disabled)",
                details={"is_leader": True, "skipped_leader": self._skipped_leader},
            )

        # Calculate success rate
        total_work = self._work_completed + self._work_failed
        success_rate = (self._work_completed / total_work * 100) if total_work > 0 else 100.0

        # Degraded if failure rate > 30%
        if total_work > 10 and success_rate < 70.0:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.DEGRADED,
                message=f"WorkerPullLoop degraded: {success_rate:.1f}% success rate",
                details={
                    "work_claimed": self._work_claimed,
                    "work_completed": self._work_completed,
                    "work_failed": self._work_failed,
                    "success_rate": success_rate,
                    "last_discovery_channel": self._last_discovery_channel,
                },
            )

        # Healthy
        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"WorkerPullLoop healthy ({self._work_claimed} claimed, {self._work_completed} completed)",
            details={
                "work_claimed": self._work_claimed,
                "work_completed": self._work_completed,
                "work_failed": self._work_failed,
                "skipped_busy": self._skipped_busy,
                "autonomous_work_claimed": self._autonomous_work_claimed,
                "work_discovery_stats": self._work_discovery_stats.copy(),
            },
        )


@dataclass
class WorkQueueMaintenanceConfig:
    """Configuration for work queue maintenance loop."""

    maintenance_interval_seconds: float = 300.0  # 5 minutes
    cleanup_age_seconds: float = 86400.0  # 24 hours
    initial_delay_seconds: float = 60.0
    # Orphan cleanup thresholds (Dec 2025)
    max_pending_age_hours: float = 24.0  # Remove stale pending items
    max_claimed_age_hours: float = 2.0  # Reset claimed items without heartbeat
    # Jan 2026: Work queue stall detection for 48h autonomous operation
    stall_threshold_seconds: float = float(
        os.environ.get("RINGRIFT_WORK_QUEUE_STALL_THRESHOLD", "300")
    )  # 5 minutes without dispatched work = stall
    stall_recovery_threshold_seconds: float = float(
        os.environ.get("RINGRIFT_WORK_QUEUE_STALL_RECOVERY_THRESHOLD", "60")
    )  # 1 minute of activity to clear stall

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.maintenance_interval_seconds <= 0:
            raise ValueError("maintenance_interval_seconds must be > 0")
        if self.cleanup_age_seconds <= 0:
            raise ValueError("cleanup_age_seconds must be > 0")
        if self.initial_delay_seconds < 0:
            raise ValueError("initial_delay_seconds must be >= 0")
        if self.max_pending_age_hours <= 0:
            raise ValueError("max_pending_age_hours must be > 0")
        if self.max_claimed_age_hours <= 0:
            raise ValueError("max_claimed_age_hours must be > 0")
        if self.stall_threshold_seconds <= 0:
            raise ValueError("stall_threshold_seconds must be > 0")


class WorkQueueMaintenanceLoop(BaseLoop):
    """Background loop for leader to maintain the work queue.

    Runs periodically to:
    - Check for timed out work items
    - Clean up old completed items from the database
    - Detect and alert on work queue stalls (Jan 2026)

    Only runs on the leader node.
    """

    def __init__(
        self,
        is_leader: Callable[[], bool],
        get_work_queue: Callable[[], Any],
        config: WorkQueueMaintenanceConfig | None = None,
    ):
        """Initialize work queue maintenance loop.

        Args:
            is_leader: Callback returning True if this node is leader
            get_work_queue: Callback returning work queue instance
            config: Loop configuration
        """
        self.config = config or WorkQueueMaintenanceConfig()
        super().__init__(
            name="work_queue_maintenance",
            interval=self.config.maintenance_interval_seconds,
        )
        self._is_leader = is_leader
        self._get_work_queue = get_work_queue

        # Statistics
        self._timeouts_processed = 0
        self._items_cleaned = 0
        self._stale_items_handled = 0

        # Jan 2026: Work queue stall detection
        self._stall_detected = False
        self._stall_detected_at: float = 0.0
        self._last_work_completed_time: float = time.time()
        self._stall_events_emitted = 0
        self._recovery_events_emitted = 0

    async def _on_start(self) -> None:
        """Initial delay before starting maintenance."""
        logger.info("Work queue maintenance loop starting...")
        await asyncio.sleep(self.config.initial_delay_seconds)
        logger.info("Work queue maintenance loop started")

    async def _run_once(self) -> None:
        """Perform work queue maintenance."""
        # Only leader performs maintenance
        if not self._is_leader():
            return

        wq = self._get_work_queue()
        if wq is None:
            return

        # Check for timeouts
        timed_out = wq.check_timeouts()
        if timed_out:
            self._timeouts_processed += len(timed_out)
            logger.info(f"[WorkQueueMaintenance] {len(timed_out)} items timed out")

        # Cleanup old completed/failed items
        removed = wq.cleanup_old_items(max_age_seconds=self.config.cleanup_age_seconds)
        if removed:
            self._items_cleaned += removed
            logger.info(f"[WorkQueueMaintenance] Cleaned up {removed} old items")

        # Cleanup stale/orphaned items (Dec 2025)
        # - Stale PENDING items (never claimed, config may be invalid)
        # - Orphaned CLAIMED items (claimer crashed without timeout)
        if hasattr(wq, 'cleanup_stale_items'):
            stale_stats = wq.cleanup_stale_items(
                max_pending_age_hours=self.config.max_pending_age_hours,
                max_claimed_age_hours=self.config.max_claimed_age_hours,
            )
            handled = stale_stats.get("pending_removed", 0) + stale_stats.get("claimed_reset", 0)
            if handled:
                self._stale_items_handled += handled
                logger.info(
                    f"[WorkQueueMaintenance] Stale items: {stale_stats.get('pending_removed', 0)} removed, "
                    f"{stale_stats.get('claimed_reset', 0)} reset to pending"
                )

        # Jan 2026: Work queue stall detection for 48h autonomous operation
        await self._check_work_queue_stall(wq)

    async def _check_work_queue_stall(self, wq: Any) -> None:
        """Check for work queue stall and emit events.

        A stall is detected when no work has been completed for longer than
        the configured threshold. This enables alerting and automatic recovery
        for 48-hour autonomous operation.
        """
        now = time.time()

        # Check if any work was completed recently
        # Use work queue stats if available, otherwise track via timeouts processed
        work_completed_recently = False

        if hasattr(wq, 'get_stats'):
            try:
                stats = wq.get_stats()
                last_completed = stats.get("last_work_completed_time", 0)
                if last_completed > self._last_work_completed_time:
                    self._last_work_completed_time = last_completed
                    work_completed_recently = True
            except Exception:
                pass

        # Also count timeouts processed as "activity" (work is happening)
        if self._timeouts_processed > 0:
            work_completed_recently = True
            self._last_work_completed_time = now

        # Calculate idle duration
        idle_duration = now - self._last_work_completed_time

        if not self._stall_detected:
            # Check for new stall
            if idle_duration > self.config.stall_threshold_seconds:
                self._stall_detected = True
                self._stall_detected_at = now
                self._stall_events_emitted += 1

                # Gather context for the event
                blocked_configs = []
                if hasattr(wq, 'get_blocked_configs'):
                    try:
                        blocked_configs = list(wq.get_blocked_configs())
                    except Exception:
                        pass

                pending_count = 0
                if hasattr(wq, 'get_pending_count'):
                    try:
                        pending_count = wq.get_pending_count()
                    except Exception:
                        pass

                logger.warning(
                    f"[WorkQueueMaintenance] Work queue STALLED: no work dispatched in "
                    f"{idle_duration:.0f}s (threshold: {self.config.stall_threshold_seconds}s). "
                    f"Pending items: {pending_count}, blocked configs: {len(blocked_configs)}"
                )

                # Emit stall event
                try:
                    emit = _get_emit_event()
                    emit(
                        "work_queue_stalled",
                        {
                            "idle_seconds": idle_duration,
                            "threshold_seconds": self.config.stall_threshold_seconds,
                            "pending_count": pending_count,
                            "blocked_configs": blocked_configs[:10],  # Limit to 10
                            "stall_detected_at": self._stall_detected_at,
                        },
                    )
                except Exception as e:
                    logger.debug(f"[WorkQueueMaintenance] Failed to emit stall event: {e}")
        else:
            # Already in stall state - check for recovery
            if work_completed_recently:
                # Work resumed - check if sustained
                recovery_duration = now - self._last_work_completed_time
                if recovery_duration < self.config.stall_recovery_threshold_seconds:
                    # Sustained activity - consider recovered
                    stall_duration = now - self._stall_detected_at
                    self._stall_detected = False
                    self._recovery_events_emitted += 1

                    logger.info(
                        f"[WorkQueueMaintenance] Work queue RECOVERED after {stall_duration:.0f}s stall"
                    )

                    # Emit recovery event
                    try:
                        emit = _get_emit_event()
                        emit(
                            "work_queue_recovered",
                            {
                                "stall_duration_seconds": stall_duration,
                                "recovery_time": now,
                            },
                        )
                    except Exception as e:
                        logger.debug(f"[WorkQueueMaintenance] Failed to emit recovery event: {e}")

    def get_maintenance_stats(self) -> dict[str, Any]:
        """Get maintenance statistics."""
        now = time.time()
        return {
            "timeouts_processed": self._timeouts_processed,
            "items_cleaned": self._items_cleaned,
            "stale_items_handled": self._stale_items_handled,
            # Jan 2026: Stall detection stats
            "stall_detected": self._stall_detected,
            "stall_events_emitted": self._stall_events_emitted,
            "recovery_events_emitted": self._recovery_events_emitted,
            "idle_duration_seconds": now - self._last_work_completed_time,
            "stall_threshold_seconds": self.config.stall_threshold_seconds,
            **self.stats.to_dict(),
        }

    def health_check(self) -> Any:
        """Check loop health with work queue maintenance-specific status.

        Jan 2026: Added for DaemonManager integration and 48h autonomous operation.
        Reports stall detection status, which is critical for autonomous recovery.

        Returns:
            HealthCheckResult with work queue maintenance status
        """
        try:
            from app.coordination.protocols import CoordinatorStatus, HealthCheckResult
        except ImportError:
            # Fallback if protocols not available
            return {
                "healthy": self.running and not self._stall_detected,
                "status": "stalled" if self._stall_detected else ("running" if self.running else "stopped"),
                "message": f"WorkQueueMaintenanceLoop {'STALLED' if self._stall_detected else 'healthy'}",
                "details": self.get_maintenance_stats(),
            }

        # Not running
        if not self.running:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="WorkQueueMaintenanceLoop is stopped",
                details={"running": False},
            )

        now = time.time()
        idle_duration = now - self._last_work_completed_time

        # Stall detected - CRITICAL for 48h autonomous operation
        if self._stall_detected:
            stall_duration = now - self._stall_detected_at if self._stall_detected_at else 0
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"Work queue STALLED for {stall_duration:.0f}s",
                details={
                    "stall_detected": True,
                    "stall_duration_seconds": stall_duration,
                    "stall_events_emitted": self._stall_events_emitted,
                    "timeouts_processed": self._timeouts_processed,
                    "items_cleaned": self._items_cleaned,
                },
            )

        # Approaching stall threshold - warning state
        approaching_stall = idle_duration > (self.config.stall_threshold_seconds * 0.7)
        if approaching_stall:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.DEGRADED,
                message=f"Work queue idle for {idle_duration:.0f}s (approaching stall threshold)",
                details={
                    "idle_duration_seconds": idle_duration,
                    "stall_threshold_seconds": self.config.stall_threshold_seconds,
                    "timeouts_processed": self._timeouts_processed,
                },
            )

        # Healthy
        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"WorkQueueMaintenanceLoop healthy ({self._items_cleaned} items cleaned)",
            details={
                "timeouts_processed": self._timeouts_processed,
                "items_cleaned": self._items_cleaned,
                "stale_items_handled": self._stale_items_handled,
                "idle_duration_seconds": idle_duration,
                "recovery_events_emitted": self._recovery_events_emitted,
            },
        )


@dataclass
class SpawnVerificationConfig:
    """Configuration for spawn verification loop.

    January 2026 Sprint 6: Verifies that dispatched jobs actually start running.
    Addresses 10-15% wasted capacity from unconfirmed job spawns.
    """

    check_interval_seconds: float = 5.0  # Fast checks for quick verification
    verification_timeout_seconds: float = 30.0  # Default per-job timeout
    log_stats_interval_runs: int = 12  # Log stats every N runs (~1 minute at 5s interval)
    min_spawns_for_rate_calc: int = 10  # Minimum spawns before reporting success rate

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be > 0")
        if self.verification_timeout_seconds <= 0:
            raise ValueError("verification_timeout_seconds must be > 0")
        if self.log_stats_interval_runs <= 0:
            raise ValueError("log_stats_interval_runs must be > 0")


class SpawnVerificationLoop(BaseLoop):
    """Background loop that verifies job spawns completed successfully.

    January 2026 Sprint 6: Part of the Job Spawn Verification system.

    This loop:
    1. Periodically calls SelfplayScheduler.verify_pending_spawns()
    2. Tracks spawn success/failure rates per node
    3. Emits JOB_SPAWN_VERIFIED / JOB_SPAWN_FAILED events
    4. Logs statistics for capacity estimation

    The verification data helps the scheduler:
    - Avoid dispatching to nodes with high spawn failure rates
    - Adjust capacity estimates based on actual spawn success
    - Identify nodes with resource or configuration issues
    """

    def __init__(
        self,
        verify_pending_spawns: Callable[[], Coroutine[Any, Any, dict[str, int]]],
        get_spawn_stats: Callable[[], dict[str, Any]] | None = None,
        config: SpawnVerificationConfig | None = None,
    ):
        """Initialize spawn verification loop.

        Args:
            verify_pending_spawns: Async callback that verifies pending spawns.
                Returns dict with 'verified', 'failed', 'pending' counts.
            get_spawn_stats: Optional callback returning spawn statistics.
            config: Loop configuration
        """
        self.config = config or SpawnVerificationConfig()
        super().__init__(
            name="spawn_verification",
            interval=self.config.check_interval_seconds,
        )
        self._verify_pending_spawns = verify_pending_spawns
        self._get_spawn_stats = get_spawn_stats

        # Statistics
        self._total_verified = 0
        self._total_failed = 0
        self._runs_since_stats_log = 0

    async def _run_once(self) -> None:
        """Verify pending job spawns and track results."""
        try:
            result = await self._verify_pending_spawns()
        except Exception as e:
            logger.warning(f"[SpawnVerification] Error verifying spawns: {e}")
            return

        verified = result.get("verified", 0)
        failed = result.get("failed", 0)
        pending = result.get("pending", 0)

        self._total_verified += verified
        self._total_failed += failed
        self._runs_since_stats_log += 1

        if verified > 0 or failed > 0:
            logger.info(
                f"[SpawnVerification] Cycle: verified={verified}, failed={failed}, pending={pending}"
            )

        # Periodically log aggregate statistics
        if self._runs_since_stats_log >= self.config.log_stats_interval_runs:
            self._log_aggregate_stats()
            self._runs_since_stats_log = 0

    def _log_aggregate_stats(self) -> None:
        """Log aggregate spawn verification statistics."""
        total = self._total_verified + self._total_failed
        if total == 0:
            return

        success_rate = (self._total_verified / total) * 100.0
        logger.info(
            f"[SpawnVerification] Aggregate: verified={self._total_verified}, "
            f"failed={self._total_failed}, success_rate={success_rate:.1f}%"
        )

        # Log per-node stats if available
        if self._get_spawn_stats:
            try:
                stats = self._get_spawn_stats()
                if stats.get("per_node"):
                    low_success_nodes = [
                        node_id for node_id, rate in stats.get("per_node", {}).items()
                        if rate < 0.7  # Less than 70% success rate
                    ]
                    if low_success_nodes:
                        logger.warning(
                            f"[SpawnVerification] Low success rate nodes: {low_success_nodes}"
                        )
            except Exception as e:
                logger.debug(f"[SpawnVerification] Could not get per-node stats: {e}")

    def get_verification_stats(self) -> dict[str, Any]:
        """Get spawn verification statistics."""
        total = self._total_verified + self._total_failed
        success_rate = (self._total_verified / total) * 100.0 if total > 0 else 100.0

        return {
            "total_verified": self._total_verified,
            "total_failed": self._total_failed,
            "success_rate": success_rate,
            **self.stats.to_dict(),
        }


# =============================================================================
# Job Reassignment Loop (P1 - Sprint 6, Jan 2026)
# =============================================================================


@dataclass
class JobReassignmentConfig:
    """Configuration for job reassignment loop.

    Automatically reassigns orphaned jobs to healthy nodes.
    This loop detects jobs that haven't received heartbeats and
    reassigns them to other available nodes.
    """

    # How often to check for orphaned jobs (default: 60 seconds)
    check_interval_seconds: float = 60.0

    # How long a job can go without heartbeat before reassignment (default: 5 min)
    # This is faster than the 1-hour reaper threshold
    orphan_threshold_seconds: float = 300.0

    # Max jobs to reassign per cycle (prevent thundering herd)
    max_reassignments_per_cycle: int = 5

    # Only leaders should run reassignment
    leader_only: bool = True

    # Initial delay before first check (allow cluster to stabilize)
    initial_delay_seconds: float = 60.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be > 0")
        if self.orphan_threshold_seconds <= 0:
            raise ValueError("orphan_threshold_seconds must be > 0")
        if self.max_reassignments_per_cycle <= 0:
            raise ValueError("max_reassignments_per_cycle must be > 0")
        if self.initial_delay_seconds < 0:
            raise ValueError("initial_delay_seconds must be >= 0")


class JobReassignmentLoop(BaseLoop):
    """Background loop that automatically reassigns orphaned jobs.

    This loop detects jobs that have stopped sending heartbeats and
    attempts to reassign them to healthy nodes. This is more aggressive
    than the JobReaperLoop (which only cleans up after 1 hour).

    Key differences from JobReaperLoop:
    - Faster detection (5 min vs 1 hour)
    - Reassigns to healthy nodes instead of just cancelling
    - Tracks reassignment success/failure rates
    - Only runs on leader node

    Integration with JobManager:
    - Uses process_stale_jobs() for the actual reassignment
    - Respects exponential backoff for retry limits
    - Emits JOB_REASSIGNED events for monitoring
    """

    def __init__(
        self,
        get_role: Callable[[], Any],
        check_and_reassign: Callable[[], Coroutine[Any, Any, int]],
        get_healthy_nodes: Callable[[], list[str]] | None = None,
        config: JobReassignmentConfig | None = None,
    ):
        """Initialize job reassignment loop.

        Args:
            get_role: Callback returning current node role (must have .is_leader)
            check_and_reassign: Async callback to check and reassign stale jobs
                               (from JobManager.process_stale_jobs)
            get_healthy_nodes: Optional callback returning list of healthy node IDs
            config: Loop configuration
        """
        self.config = config or JobReassignmentConfig()
        super().__init__(
            name="job_reassignment",
            interval=self.config.check_interval_seconds,
        )
        self._get_role = get_role
        self._check_and_reassign = check_and_reassign
        self._get_healthy_nodes = get_healthy_nodes

        # Statistics
        self._total_reassigned = 0
        self._cycles_run = 0
        self._last_reassignment_time: float = 0.0
        self._skipped_not_leader = 0
        self._skipped_no_healthy_nodes = 0

        # Initial delay flag
        self._initial_delay_done = False
        self._start_time = time.time()

    async def _run_once(self) -> None:
        """Check for orphaned jobs and reassign them."""
        self._cycles_run += 1

        # Apply initial delay
        if not self._initial_delay_done:
            elapsed = time.time() - self._start_time
            if elapsed < self.config.initial_delay_seconds:
                logger.debug(
                    f"[JobReassignment] Waiting for initial delay: "
                    f"{self.config.initial_delay_seconds - elapsed:.0f}s remaining"
                )
                return
            self._initial_delay_done = True
            logger.info("[JobReassignment] Initial delay complete, starting checks")

        # Only leader runs reassignment
        if self.config.leader_only:
            role = self._get_role()
            is_leader = getattr(role, "is_leader", False) if role else False
            if hasattr(role, "name"):
                is_leader = role.name == "LEADER"

            if not is_leader:
                self._skipped_not_leader += 1
                return

        # Optional: Check if we have healthy nodes to reassign to
        if self._get_healthy_nodes:
            try:
                healthy_nodes = self._get_healthy_nodes()
                if len(healthy_nodes) < 2:  # Need at least 2 nodes
                    self._skipped_no_healthy_nodes += 1
                    logger.debug(
                        f"[JobReassignment] Skipping: only {len(healthy_nodes)} healthy nodes"
                    )
                    return
            except Exception as e:
                logger.debug(f"[JobReassignment] Could not get healthy nodes: {e}")

        # Run the actual reassignment check
        try:
            reassigned = await self._check_and_reassign()
            if reassigned > 0:
                self._total_reassigned += reassigned
                self._last_reassignment_time = time.time()
                logger.info(
                    f"[JobReassignment] Reassigned {reassigned} jobs "
                    f"(total: {self._total_reassigned})"
                )
        except Exception as e:
            logger.warning(f"[JobReassignment] Error during reassignment check: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get reassignment statistics."""
        return {
            "total_reassigned": self._total_reassigned,
            "cycles_run": self._cycles_run,
            "last_reassignment_time": self._last_reassignment_time,
            "skipped_not_leader": self._skipped_not_leader,
            "skipped_no_healthy_nodes": self._skipped_no_healthy_nodes,
            **self.stats.to_dict(),
        }


__all__ = [
    "IdleDetectionConfig",
    "IdleDetectionLoop",
    "JobReaperConfig",
    "JobReaperLoop",
    "JobReassignmentConfig",
    "JobReassignmentLoop",
    "PredictiveScalingConfig",
    "PredictiveScalingLoop",
    "SpawnVerificationConfig",
    "SpawnVerificationLoop",
    "WorkerPullConfig",
    "WorkerPullLoop",
    "WorkQueueMaintenanceConfig",
    "WorkQueueMaintenanceLoop",
]
