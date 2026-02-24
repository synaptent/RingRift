"""Job Coordination Manager for P2P Orchestrator.

January 2026: Phase 15 of P2P Orchestrator Decomposition.
Consolidates job management loop, cluster job coordination, and auto-scaling.

This module provides:
- Job management loop orchestration
- Local job management (decentralized)
- GPU auto-scaling
- Resource cleanup
- Work queue rebalancing
- Cluster balance checking

Extracted from p2p_orchestrator.py to reduce complexity and improve testability.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

try:
    import aiohttp
    from aiohttp import ClientTimeout
except ImportError:
    aiohttp = None  # type: ignore
    ClientTimeout = None  # type: ignore

if TYPE_CHECKING:
    from scripts.p2p_orchestrator import P2POrchestrator
    from scripts.p2p.models import NodeInfo

logger = logging.getLogger(__name__)

# Constants
DEFAULT_PORT = 8770
JOB_CHECK_INTERVAL = 30  # Seconds between job management cycles


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class JobCoordinationConfig:
    """Configuration for job coordination behavior."""
    job_management_interval: float = 30.0
    auto_scale_interval: float = 120.0
    rebalance_interval: float = 60.0
    gpu_utilization_target: float = 0.75
    gpu_utilization_min: float = 0.60
    gpu_utilization_max: float = 0.80
    # Disk: aligned with app.config.thresholds (DISK_PRODUCTION_HALT=85, DISK_SYNC_TARGET=70)
    disk_cleanup_threshold: float = 80.0   # Start cleanup before production halt
    disk_warning_threshold: float = 75.0   # Between sync target (70) and cleanup (80)
    memory_warning_threshold: float = 70.0
    memory_critical_threshold: float = 85.0
    # Job limits
    max_selfplay_jobs_per_node: int = 4
    max_training_jobs_per_node: int = 1


@dataclass
class JobCoordinationStats:
    """Statistics about job coordination operations."""
    coordination_cycles: int = 0
    jobs_started: int = 0
    jobs_stopped: int = 0
    cluster_rebalances: int = 0
    gpu_scaling_adjustments: int = 0
    work_queue_dispatches: int = 0
    stuck_jobs_detected: int = 0
    resource_cleanups: int = 0
    local_job_cycles: int = 0
    cluster_job_cycles: int = 0


# ============================================================================
# Singleton Pattern
# ============================================================================

_job_coordination_manager: JobCoordinationManager | None = None


def get_job_coordination_manager() -> JobCoordinationManager | None:
    """Get the global JobCoordinationManager singleton."""
    return _job_coordination_manager


def set_job_coordination_manager(manager: JobCoordinationManager) -> None:
    """Set the global JobCoordinationManager singleton."""
    global _job_coordination_manager
    _job_coordination_manager = manager


def reset_job_coordination_manager() -> None:
    """Reset the global JobCoordinationManager singleton (for testing)."""
    global _job_coordination_manager
    _job_coordination_manager = None


def create_job_coordination_manager(
    config: JobCoordinationConfig | None = None,
    orchestrator: P2POrchestrator | None = None,
) -> JobCoordinationManager:
    """Factory function to create and register a JobCoordinationManager.

    Args:
        config: Optional configuration. Uses defaults if not provided.
        orchestrator: The P2P orchestrator instance.

    Returns:
        The created JobCoordinationManager instance.
    """
    manager = JobCoordinationManager(
        config=config or JobCoordinationConfig(),
        orchestrator=orchestrator,
    )
    set_job_coordination_manager(manager)
    return manager


# ============================================================================
# JobCoordinationManager
# ============================================================================

class JobCoordinationManager:
    """Manages job coordination for P2P cluster.

    This manager handles:
    - Job management loop orchestration
    - Local job management (decentralized)
    - GPU auto-scaling
    - Resource cleanup
    - Work queue rebalancing
    - Cluster balance checking

    Jan 27, 2026: Phase 15 decomposition from p2p_orchestrator.py.
    """

    def __init__(
        self,
        config: JobCoordinationConfig | None = None,
        orchestrator: P2POrchestrator | None = None,
    ):
        """Initialize the JobCoordinationManager.

        Args:
            config: Job coordination configuration.
            orchestrator: The P2P orchestrator instance.
        """
        self.config = config or JobCoordinationConfig()
        self._orchestrator = orchestrator
        self._stats = JobCoordinationStats()

        # State tracking with proper dicts (not dynamic attributes)
        self._last_local_job_manage: float = 0.0
        self._last_local_gpu_scale: float = 0.0
        self._last_local_resource_check: float = 0.0
        self._last_gpu_auto_scale: float = 0.0
        self._last_work_queue_rebalance: float = 0.0
        self._last_cluster_balance_check: float = 0.0

        # Per-node state tracking
        self._gpu_idle_since: dict[str, float] = {}
        self._wq_idle_since: dict[str, float] = {}
        self._diverse_config_counter: dict[str, int] = {}
        self._local_gpu_idle_since: float = 0.0

    # =========================================================================
    # Properties (delegate to orchestrator)
    # =========================================================================

    @property
    def _peers(self) -> dict[str, Any]:
        """Get peers dict from orchestrator."""
        return getattr(self._orchestrator, "peers", {})

    @property
    def _peers_lock(self) -> Any:
        """Get peers lock from orchestrator."""
        return getattr(self._orchestrator, "peers_lock", None)

    @property
    def _node_id(self) -> str:
        """Get this node's ID."""
        return getattr(self._orchestrator, "node_id", "unknown")

    @property
    def _self_info(self) -> Any:
        """Get this node's info."""
        return getattr(self._orchestrator, "self_info", None)

    @property
    def _leader_id(self) -> str | None:
        """Get current leader ID."""
        return getattr(self._orchestrator, "leader_id", None)

    @property
    def _role(self) -> Any:
        """Get this node's role."""
        return getattr(self._orchestrator, "role", None)

    @property
    def _running(self) -> bool:
        """Check if orchestrator is running."""
        return getattr(self._orchestrator, "running", False)

    @property
    def _job_lifecycle_manager(self) -> Any:
        """Get job lifecycle manager."""
        return getattr(self._orchestrator, "job_lifecycle_manager", None)

    @property
    def _selfplay_scheduler(self) -> Any:
        """Get selfplay scheduler."""
        return getattr(self._orchestrator, "selfplay_scheduler", None)

    @property
    def _work_queue(self) -> Any:
        """Get work queue."""
        return getattr(self._orchestrator, "work_queue", None)

    # =========================================================================
    # Helper Methods (delegate to orchestrator)
    # =========================================================================

    def _auth_headers(self) -> dict[str, str]:
        """Get auth headers from orchestrator."""
        if self._orchestrator and hasattr(self._orchestrator, "_auth_headers"):
            return self._orchestrator._auth_headers()
        return {}

    def _url_for_peer(self, peer: Any, path: str) -> str:
        """Get URL for peer."""
        if self._orchestrator and hasattr(self._orchestrator, "_url_for_peer"):
            return self._orchestrator._url_for_peer(peer, path)
        # Fallback
        scheme = getattr(peer, "scheme", "http") or "http"
        host = getattr(peer, "host", "localhost")
        port = getattr(peer, "port", DEFAULT_PORT)
        return f"{scheme}://{host}:{port}{path}"

    async def _update_self_info(self) -> None:
        """Update self info.

        Feb 22, 2026: No-op. Resource detection (GPU, disk, CPU) is slow on
        macOS (10-30s) and blocks the event loop. self_info is refreshed in
        the background by heartbeat and health loops - use cached data.
        """
        pass

    def _get_cached_peer_snapshot(self) -> dict[str, Any]:
        """Get cached peer snapshot for lock-free access."""
        if self._orchestrator and hasattr(self._orchestrator, "_get_cached_peer_snapshot"):
            return self._orchestrator._get_cached_peer_snapshot()
        if self._peers_lock:
            with self._peers_lock:
                return dict(self._peers)
        return dict(self._peers)

    def _check_yaml_gpu_config(self) -> bool:
        """Check if this node has GPU configured in YAML."""
        if self._orchestrator and hasattr(self._orchestrator, "_check_yaml_gpu_config"):
            return self._orchestrator._check_yaml_gpu_config()
        return False

    def _pick_weighted_config(self) -> dict | None:
        """Pick a weighted selfplay config from the scheduler.

        Returns:
            Config dict with board_type and num_players, or None if unavailable.
        """
        scheduler = self._selfplay_scheduler
        if scheduler and hasattr(scheduler, "pick_weighted_config"):
            return scheduler.pick_weighted_config(self._self_info)
        return None

    async def _start_local_job(
        self,
        job_type: str,
        board_type: str,
        num_players: int,
        engine_mode: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Start a local job via orchestrator.

        Args:
            job_type: Type of job (gumbel_selfplay, gpu_selfplay, hybrid_selfplay, etc.)
            board_type: Board type (hex8, square8, etc.)
            num_players: Number of players (2, 3, 4)
            engine_mode: Engine mode (gumbel-mcts, gpu, mixed, etc.)
            **kwargs: Additional job parameters

        Returns:
            Job object if started, None otherwise.
        """
        if not self._orchestrator:
            return None

        # Convert string job_type to JobType enum if needed
        try:
            from scripts.p2p.models import JobType
            if isinstance(job_type, str):
                job_type_map = {
                    "gumbel_selfplay": JobType.GUMBEL_SELFPLAY,
                    "gpu_selfplay": JobType.GPU_SELFPLAY,
                    "hybrid_selfplay": JobType.HYBRID_SELFPLAY,
                    "selfplay": JobType.SELFPLAY,
                }
                job_type_enum = job_type_map.get(job_type.lower(), JobType.SELFPLAY)
            else:
                job_type_enum = job_type
        except ImportError:
            job_type_enum = job_type

        if hasattr(self._orchestrator, "_start_local_job"):
            return await self._orchestrator._start_local_job(
                job_type_enum,
                board_type=board_type,
                num_players=num_players,
                engine_mode=engine_mode,
                **kwargs,
            )
        return None

    async def _emit_batch_scheduled(
        self,
        batch_id: str,
        batch_type: str,
        config_key: str,
        job_count: int,
        target_nodes: list[str],
        reason: str,
    ) -> None:
        """Emit BATCH_SCHEDULED event via orchestrator."""
        if self._orchestrator and hasattr(self._orchestrator, "_emit_batch_scheduled"):
            await self._orchestrator._emit_batch_scheduled(
                batch_id=batch_id,
                batch_type=batch_type,
                config_key=config_key,
                job_count=job_count,
                target_nodes=target_nodes,
                reason=reason,
            )

    async def _emit_batch_dispatched(
        self,
        batch_id: str,
        batch_type: str,
        config_key: str,
        jobs_dispatched: int,
        jobs_failed: int,
        target_nodes: list[str],
    ) -> None:
        """Emit BATCH_DISPATCHED event via orchestrator."""
        if self._orchestrator and hasattr(self._orchestrator, "_emit_batch_dispatched"):
            await self._orchestrator._emit_batch_dispatched(
                batch_id=batch_id,
                batch_type=batch_type,
                config_key=config_key,
                jobs_dispatched=jobs_dispatched,
                jobs_failed=jobs_failed,
                target_nodes=target_nodes,
            )

    # =========================================================================
    # Local Job Management (Decentralized)
    # =========================================================================

    async def manage_local_jobs_decentralized(self) -> None:
        """DECENTRALIZED: Each node manages its own job capacity.

        This runs on ALL nodes independently of leader status.
        Nodes determine their own job targets based on:
        - Local GPU/CPU capacity
        - Current job counts
        - Selfplay scheduler priorities

        Jan 27, 2026: Migrated from p2p_orchestrator.py (Phase 15).
        """
        # Delegate to orchestrator's implementation for now
        # This method has complex GPU detection and job spawning logic
        if self._orchestrator and hasattr(self._orchestrator, "_manage_local_jobs_decentralized"):
            await self._orchestrator._manage_local_jobs_decentralized()
            self._stats.local_job_cycles += 1

    async def local_gpu_auto_scale(self) -> int:
        """DECENTRALIZED: Each GPU node manages its own GPU utilization.

        Runs on ALL GPU nodes to ensure optimal GPU usage without leader.
        Targets 60-80% GPU utilization by starting/stopping GPU selfplay jobs.

        Returns:
            Number of GPU jobs started

        Jan 28, 2026: Phase 18A - Full implementation migrated from p2p_orchestrator.py.
        """
        started = 0
        now = time.time()

        # Rate limit: check every 2 minutes
        if now - self._last_local_gpu_scale < 120:
            return 0
        self._last_local_gpu_scale = now

        node = self._self_info
        if not node:
            return 0

        # Skip if not a GPU node
        if not getattr(node, "has_gpu", False):
            return 0

        # Skip if training is running (training uses GPU)
        training_jobs = int(getattr(node, "training_jobs", 0) or 0)
        if training_jobs > 0:
            return 0

        # Skip if memory is critical
        memory_percent = float(getattr(node, "memory_percent", 0) or 0)
        if memory_percent >= self.config.memory_warning_threshold:
            logger.info(f"LOCAL: Memory at {memory_percent:.0f}% - skipping GPU auto-scale")
            return 0

        # DECENTRALIZED: Always allow local GPU scaling
        # Leader manages cluster-wide coordination, but each node optimizes its own GPU
        # Use smaller batches when leader is present to avoid conflicts
        from scripts.p2p.models import NodeRole
        has_leader = bool(self._leader_id or self._role == NodeRole.LEADER)
        max_jobs_per_cycle = 1 if has_leader else 3

        TARGET_GPU_MIN = 60.0
        TARGET_GPU_MAX = 80.0
        MIN_IDLE_TIME = 120 if has_leader else 60  # Faster response when leaderless

        gpu_percent = float(getattr(node, "gpu_percent", 0) or 0)
        gpu_name = (getattr(node, "gpu_name", "") or "").lower()

        # YAML fallback when runtime GPU detection fails
        if not gpu_name:
            yaml_has_gpu, yaml_gpu_name = self._check_yaml_gpu_config_for_node(self._node_id)
            if yaml_has_gpu and yaml_gpu_name:
                gpu_name = yaml_gpu_name.lower()
                logger.debug(f"LOCAL: Using YAML GPU name: {yaml_gpu_name}")

        # Track GPU idle time
        if gpu_percent < TARGET_GPU_MIN:
            if self._local_gpu_idle_since == 0:
                self._local_gpu_idle_since = now
            elif now - self._local_gpu_idle_since > MIN_IDLE_TIME:
                # Calculate new jobs to add
                gpu_headroom = TARGET_GPU_MAX - gpu_percent
                is_high_end_gpu = any(tag in gpu_name for tag in ("h100", "h200", "gh200", "5090", "a100", "4090"))
                if any(tag in gpu_name for tag in ("h100", "h200", "gh200", "5090")):
                    jobs_per_10_percent = 2
                elif any(tag in gpu_name for tag in ("a100", "4090", "3090")):
                    jobs_per_10_percent = 1.5
                else:
                    jobs_per_10_percent = 1

                new_jobs = max(1, int(gpu_headroom / 10 * jobs_per_10_percent))
                new_jobs = min(new_jobs, max_jobs_per_cycle)

                # Use GUMBEL/GPU_SELFPLAY for high-end GPUs
                job_type_str = "GUMBEL/GPU" if is_high_end_gpu else "diverse/hybrid"
                logger.info(f"LOCAL: {gpu_percent:.0f}% GPU util, starting {new_jobs} {job_type_str} selfplay job(s)")

                # Generate batch ID and emit BATCH_SCHEDULED
                batch_id = f"gpu_selfplay_{self._node_id}_{int(time.time())}"
                first_config = self._pick_weighted_config()
                config_key = f"{first_config['board_type']}_{first_config['num_players']}p" if first_config else "mixed"
                await self._emit_batch_scheduled(
                    batch_id=batch_id,
                    batch_type="selfplay",
                    config_key=config_key,
                    job_count=new_jobs,
                    target_nodes=[self._node_id],
                    reason="gpu_auto_scale",
                )

                gpu_jobs_dispatched = 0
                gpu_jobs_failed = 0
                for _ in range(new_jobs):
                    try:
                        config = self._pick_weighted_config()
                        if config:
                            # Select job type based on GPU capabilities
                            if is_high_end_gpu:
                                # High-end GPUs: 50% GUMBEL (quality) / 50% GPU_SELFPLAY (volume)
                                if random.random() < 0.5:
                                    job_type = "gumbel_selfplay"
                                    engine_mode = "gumbel-mcts"
                                else:
                                    job_type = "gpu_selfplay"
                                    engine_mode = "gpu"
                            else:
                                # Mid-tier GPUs: HYBRID mode for rule fidelity
                                job_type = "hybrid_selfplay"
                                engine_mode = "mixed"

                            job = await self._start_local_job(
                                job_type=job_type,
                                board_type=config["board_type"],
                                num_players=config["num_players"],
                                engine_mode=engine_mode,
                            )
                            if job:
                                started += 1
                                gpu_jobs_dispatched += 1
                            else:
                                gpu_jobs_failed += 1
                    except Exception as e:  # noqa: BLE001
                        logger.info(f"LOCAL: Failed to start selfplay: {e}")
                        gpu_jobs_failed += 1
                        break

                # Emit BATCH_DISPATCHED after loop completes
                await self._emit_batch_dispatched(
                    batch_id=batch_id,
                    batch_type="selfplay",
                    config_key=config_key,
                    jobs_dispatched=gpu_jobs_dispatched,
                    jobs_failed=gpu_jobs_failed,
                    target_nodes=[self._node_id],
                )

                self._local_gpu_idle_since = now  # Reset after action
        else:
            self._local_gpu_idle_since = 0  # GPU is busy, reset

        if started > 0:
            logger.info(f"LOCAL GPU auto-scale: started {started} job(s)")
            self._stats.gpu_scaling_adjustments += 1
            self._stats.jobs_started += started

        return started

    async def local_resource_cleanup(self) -> None:
        """DECENTRALIZED: Each node handles its own resource pressure.

        Runs on ALL nodes to ensure resource cleanup without leader coordination.
        Handles disk cleanup, memory pressure, and log rotation.

        Jan 27, 2026: Phase 16B - Full implementation migrated from p2p_orchestrator.py.
        """
        import asyncio

        now = time.time()

        # Rate limit: check every 5 minutes
        if now - self._last_local_resource_check < 300:
            return
        self._last_local_resource_check = now

        await self._update_self_info()
        node = self._self_info
        if not node:
            return

        # Get thresholds from config
        disk_threshold = self.config.disk_cleanup_threshold
        memory_critical = self.config.memory_critical_threshold
        memory_warning = self.config.memory_warning_threshold

        disk_percent = getattr(node, "disk_percent", 0) or 0
        memory_percent = getattr(node, "memory_percent", 0) or 0

        # Disk cleanup
        if disk_percent >= disk_threshold:
            logger.info(f"LOCAL: Disk at {disk_percent:.0f}% - triggering cleanup")
            if self._orchestrator and hasattr(self._orchestrator, "_cleanup_local_disk"):
                await self._orchestrator._cleanup_local_disk()

        # Memory pressure - reduce jobs and clear caches
        if memory_percent >= memory_critical:
            logger.warning(f"LOCAL: Memory CRITICAL at {memory_percent:.0f}% - emergency cleanup")
            if self._orchestrator and hasattr(self._orchestrator, "_reduce_local_selfplay_jobs"):
                await self._orchestrator._reduce_local_selfplay_jobs(0, reason="memory_critical")
            if self._orchestrator and hasattr(self._orchestrator, "_emergency_memory_cleanup"):
                self._orchestrator._emergency_memory_cleanup()
            # Backoff after emergency cleanup to avoid tight loop
            logger.info("LOCAL: Sleeping 60s after memory emergency to avoid tight loop")
            await asyncio.sleep(60)
        elif memory_percent >= memory_warning:
            current = int(getattr(node, "selfplay_jobs", 0) or 0)
            target = max(1, current // 2)
            logger.info(f"LOCAL: Memory warning at {memory_percent:.0f}% - reducing jobs to {target}")
            if self._orchestrator and hasattr(self._orchestrator, "_reduce_local_selfplay_jobs"):
                await self._orchestrator._reduce_local_selfplay_jobs(target, reason="memory_warning")

        # Clean up old completed/failed jobs to prevent memory leak
        job_manager = getattr(self._orchestrator, "job_manager", None)
        if job_manager and hasattr(job_manager, "cleanup_completed_jobs"):
            job_manager.cleanup_completed_jobs()

        self._stats.resource_cleanups += 1

    # =========================================================================
    # Leader-Only Operations
    # =========================================================================

    async def manage_cluster_jobs(self) -> None:
        """LEADER ONLY: Manage jobs across the cluster.

        Coordinates job distribution, handles stuck jobs, and
        balances load across nodes.

        Jan 27, 2026: Migrated from p2p_orchestrator.py (Phase 15).
        """
        if self._orchestrator and hasattr(self._orchestrator, "_manage_cluster_jobs"):
            await self._orchestrator._manage_cluster_jobs()
            self._stats.cluster_job_cycles += 1

    async def auto_scale_gpu_utilization(self) -> None:
        """LEADER ONLY: Scale GPU utilization across cluster.

        Identifies under/over-utilized nodes and adjusts job counts
        to optimize cluster-wide GPU utilization.

        Jan 27, 2026: Migrated from p2p_orchestrator.py (Phase 15).
        """
        if self._orchestrator and hasattr(self._orchestrator, "_auto_scale_gpu_utilization"):
            await self._orchestrator._auto_scale_gpu_utilization()

    async def auto_rebalance_from_work_queue(self) -> None:
        """LEADER ONLY: Dispatch queued work to idle nodes.

        Claims work from the distributed work queue and dispatches
        to available nodes.

        Jan 27, 2026: Migrated from p2p_orchestrator.py (Phase 15).
        """
        if self._orchestrator and hasattr(self._orchestrator, "_auto_rebalance_from_work_queue"):
            await self._orchestrator._auto_rebalance_from_work_queue()
            self._stats.work_queue_dispatches += 1

    async def dispatch_queued_work(self, peer: Any, work_item: Any) -> bool:
        """Dispatch a work queue item to a specific node.

        Routes different work types to appropriate endpoints.

        Args:
            peer: Target node info (NodeInfo)
            work_item: Work item dict with work_id, work_type, config, etc.

        Returns:
            True if dispatch was successful.

        Jan 27, 2026: Phase 16B - Full implementation migrated from p2p_orchestrator.py.
        """
        if aiohttp is None:
            return False

        from app.coordination.work_queue import WorkType
        from scripts.p2p.network import get_client_session

        try:
            timeout = ClientTimeout(total=30)
            async with get_client_session(timeout) as session:
                # Handle both dict and object formats for backward compatibility
                if isinstance(work_item, dict):
                    work_type = work_item.get("work_type")
                    config = work_item.get("config", {})
                    work_id = work_item.get("work_id")
                else:
                    work_type = getattr(work_item, "work_type", None)
                    config = getattr(work_item, "config", {})
                    work_id = getattr(work_item, "work_id", None)

                # Convert string work_type to enum if needed
                if isinstance(work_type, str):
                    try:
                        work_type = WorkType(work_type)
                    except ValueError:
                        logger.warning(f"Unknown work type string: {work_type}")
                        return False

                # Get peer node_id for logging
                peer_node_id = getattr(peer, "node_id", "unknown")

                if work_type == WorkType.TRAINING:
                    url = self._url_for_peer(peer, "/start_job")
                    payload = {
                        "job_type": "training",
                        "board_type": config.get("board_type", "square8"),
                        "num_players": config.get("num_players", 2),
                        "work_id": work_id,
                    }
                elif work_type == WorkType.GPU_CMAES:
                    url = self._url_for_peer(peer, "/cmaes/start")
                    payload = {
                        "board_type": config.get("board_type", "square8"),
                        "num_players": config.get("num_players", 2),
                        "work_id": work_id,
                    }
                elif work_type == WorkType.TOURNAMENT:
                    url = self._url_for_peer(peer, "/tournament/start")
                    payload = {
                        "board_type": config.get("board_type", "square8"),
                        "num_players": config.get("num_players", 2),
                        "work_id": work_id,
                    }
                elif work_type == WorkType.SELFPLAY:
                    url = self._url_for_peer(peer, "/selfplay/start")
                    payload = {
                        "board_type": config.get("board_type", "square8"),
                        "num_players": config.get("num_players", 2),
                        "num_games": config.get("num_games", 500),
                        "work_id": work_id,
                    }
                elif work_type == WorkType.GAUNTLET:
                    url = self._url_for_peer(peer, "/gauntlet/start")
                    payload = {
                        "board_type": config.get("board_type", "square8"),
                        "num_players": config.get("num_players", 2),
                        "model_path": config.get("candidate_model", ""),
                        "games": config.get("games", 100),
                        "work_id": work_id,
                    }
                else:
                    logger.warning(f"Unknown work type: {work_type}")
                    return False

                async with session.post(url, json=payload, headers=self._auth_headers()) as resp:
                    if resp.status == 200:
                        logger.info(f"Dispatched {work_type.value} work to {peer_node_id}")
                        self._stats.work_queue_dispatches += 1
                        return True
                    else:
                        error = await resp.text()
                        logger.warning(f"Failed to dispatch work to {peer_node_id}: {error}")
                        return False

        except Exception as e:  # noqa: BLE001
            logger.error(f"Error dispatching work to {getattr(peer, 'node_id', 'unknown')}: {e}")
            return False

    async def schedule_diverse_selfplay_on_node(self, node_id: str) -> dict | None:
        """Schedule a diverse selfplay job on a specific node.

        Job type is selected based on GPU capabilities:
        - High-end GPUs (GH200, H100, A100, 5090, 4090): 50% GUMBEL / 50% GPU_SELFPLAY
        - Mid-tier GPUs: HYBRID mode (CPU rules + GPU eval) for rule fidelity
        Rotates through all board/player configurations for diversity.

        Args:
            node_id: Target node ID.

        Returns:
            Response dict if successful, None otherwise.

        Jan 27, 2026: Phase 16C - Full implementation migrated from p2p_orchestrator.py.
        """
        if aiohttp is None:
            return None

        from scripts.p2p.network import get_client_session

        try:
            from app.config.coordination_defaults import get_board_priority_overrides
        except ImportError:
            def get_board_priority_overrides() -> dict:
                return {}

        # Get peer from orchestrator's peers dict
        peer = None
        if self._peers_lock:
            with self._peers_lock:
                peer = self._peers.get(node_id)
        if not peer or not peer.is_alive():
            return None

        # Policy check: ensure selfplay is allowed on this node
        try:
            from app.coordination.node_policies import is_work_allowed
            if not is_work_allowed(node_id, "selfplay"):
                logger.debug(f"Selfplay not allowed on {node_id} by policy")
                return None
        except ImportError:
            pass

        # Rotate through diverse configurations with priority-based weighting
        board_priority_overrides = get_board_priority_overrides()

        diverse_configs = [
            ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
            ("square19", 3), ("square19", 4), ("square19", 2),
            ("hex8", 2), ("hex8", 3), ("hex8", 4),
            ("square8", 3), ("square8", 4), ("square8", 2),
        ]

        # Build weighted list based on priority overrides
        # Lower priority value = more weight (0=CRITICAL gets 4x, 3=LOW gets 1x)
        weighted_configs = []
        for board_type, num_players in diverse_configs:
            config_key = f"{board_type}_{num_players}p"
            priority = board_priority_overrides.get(config_key, 3)  # Default LOW
            weight = 4 - priority  # 0->4, 1->3, 2->2, 3->1
            weighted_configs.extend([(board_type, num_players)] * weight)

        # Round-robin through weighted list based on node-specific counter
        counter = self._diverse_config_counter.get(node_id, 0)
        self._diverse_config_counter[node_id] = counter + 1
        board_type, num_players = weighted_configs[counter % len(weighted_configs)]

        # Determine job type based on node GPU capabilities
        gpu_name = (getattr(peer, "gpu_name", "") or "").upper()
        has_gpu = bool(getattr(peer, "has_gpu", False))

        # YAML fallback when runtime GPU detection fails
        if not has_gpu or not gpu_name:
            yaml_has_gpu, yaml_gpu_name = self._check_yaml_gpu_config_for_node(node_id)
            if yaml_has_gpu:
                has_gpu = True
                if yaml_gpu_name:
                    gpu_name = yaml_gpu_name.upper()

        is_high_end_gpu = any(tag in gpu_name for tag in ("H100", "H200", "GH200", "A100", "5090", "4090"))
        is_apple_gpu = "MPS" in gpu_name or "APPLE" in gpu_name

        if has_gpu and is_high_end_gpu and not is_apple_gpu:
            # High-end GPUs: 50% GUMBEL (quality) / 50% GPU_SELFPLAY (volume)
            if random.random() < 0.5:
                job_type = "gumbel_selfplay"
                engine_mode = "gumbel-mcts"
            else:
                job_type = "gpu_selfplay"
                engine_mode = "gpu"
        else:
            # Mid-tier or no GPU: HYBRID mode for rule fidelity
            job_type = "hybrid_selfplay"
            engine_mode = "mixed"

        try:
            timeout = ClientTimeout(total=30)
            async with get_client_session(timeout) as session:
                url = self._url_for_peer(peer, "/selfplay/start")
                payload = {
                    "board_type": board_type,
                    "num_players": num_players,
                    "num_games": 200,  # Smaller batches for diversity
                    "engine_mode": engine_mode,
                    "auto_scaled": True,
                    "job_type": job_type,
                }
                async with session.post(url, json=payload, headers=self._auth_headers()) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        logger.info(f"Started diverse selfplay on {node_id}: {board_type} {num_players}p")
                        return data
                    else:
                        error = await resp.text()
                        logger.info(f"Diverse selfplay start failed on {node_id}: {error}")
                        return None
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to schedule diverse selfplay on {node_id}: {e}")
            return None

    def _check_yaml_gpu_config_for_node(self, node_id: str) -> tuple[bool, str | None]:
        """Check YAML config for GPU info when runtime detection fails.

        Delegates to orchestrator's implementation.
        """
        if self._orchestrator and hasattr(self._orchestrator, "_check_yaml_gpu_config"):
            result = self._orchestrator._check_yaml_gpu_config(node_id)
            if isinstance(result, tuple) and len(result) >= 2:
                return result[0], result[1]
        return False, None

    async def check_cluster_balance(self) -> None:
        """LEADER ONLY: Check and rebalance cluster load.

        Identifies overloaded/underutilized nodes and migrates
        jobs to balance the cluster.

        Jan 27, 2026: Migrated from p2p_orchestrator.py (Phase 15).
        """
        if self._orchestrator and hasattr(self._orchestrator, "_check_cluster_balance"):
            await self._orchestrator._check_cluster_balance()
            self._stats.cluster_rebalances += 1

    # =========================================================================
    # Coordination Cycle
    # =========================================================================

    async def run_coordination_cycle(self) -> None:
        """Run one job coordination cycle.

        This is the main entry point called from the orchestrator's
        job management loop. It handles both decentralized and
        leader-only operations.

        Jan 27, 2026: Migrated from p2p_orchestrator.py (Phase 15).
        """
        self._stats.coordination_cycles += 1

        # Decentralized operations (all nodes)
        await self.local_resource_cleanup()
        await self.manage_local_jobs_decentralized()
        await self.local_gpu_auto_scale()

        # Leader-only operations
        from scripts.p2p.models import NodeRole
        if self._role == NodeRole.LEADER:
            await self.manage_cluster_jobs()
            await self.check_cluster_balance()
            await self.auto_rebalance_from_work_queue()
            await self.auto_scale_gpu_utilization()

    # =========================================================================
    # Health Check
    # =========================================================================

    def health_check(self) -> dict[str, Any]:
        """Return health status for DaemonManager integration.

        Returns:
            dict with healthy status, message, and details.
        """
        is_healthy = self._running
        stats = asdict(self._stats)

        message = "Job coordination healthy" if is_healthy else "Job coordination stopped"

        return {
            "healthy": is_healthy,
            "message": message,
            "details": stats,
        }

    def get_stats(self) -> JobCoordinationStats:
        """Get current statistics."""
        return self._stats
