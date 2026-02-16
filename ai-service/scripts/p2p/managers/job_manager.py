"""JobManager: Job spawning and lifecycle management for P2P orchestrator.

Extracted from p2p_orchestrator.py for better modularity.
Handles job spawning, execution, monitoring, and cleanup across different job types.

December 28, 2025: Updated to use EventSubscriptionMixin for consolidated event handling.
This eliminates ~50 LOC of duplicate subscription boilerplate.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import shutil

from app.config.coordination_defaults import (
    DaemonHealthDefaults,
    JobDefaults,
    JobReaperDefaults,
    OperationTimeouts,
    SSHDefaults,
    TaskLifecycleDefaults,
)

# Sprint 10 (Jan 3, 2026): Adaptive timeout for SSH operations
try:
    from app.distributed.circuit_breaker import get_adaptive_timeout
except ImportError:
    def get_adaptive_timeout(operation_type: str, host: str, default: float) -> float:
        """Fallback when circuit_breaker not available."""
        return default

# Sprint 15.1.1 (Jan 3, 2026): Per-transport circuit breakers for HTTP calls
try:
    from scripts.p2p.network import (
        check_peer_transport_circuit,
        record_peer_transport_success,
        record_peer_transport_failure,
    )
    HAS_TRANSPORT_CB = True
except ImportError:
    HAS_TRANSPORT_CB = False
    def check_peer_transport_circuit(host: str, transport: str = "http") -> bool:
        return True  # Fallback: allow all
    def record_peer_transport_success(host: str, transport: str = "http") -> None:
        pass
    def record_peer_transport_failure(host: str, transport: str = "http", error: Exception | None = None) -> None:
        pass

# Sprint 17.9 (Jan 2026): SSH fallback for job dispatch when HTTP circuit is open
try:
    from app.core.ssh import run_ssh_command_async
    HAS_SSH_FALLBACK = True
except ImportError:
    HAS_SSH_FALLBACK = False
    async def run_ssh_command_async(host: str, command: str, timeout: float = 30.0) -> Any:
        """Fallback when SSH module not available."""
        return None

# Dec 31, 2025: Import batch size calculator for GPU utilization optimization
try:
    from app.ai.gpu_parallel_games import get_optimal_batch_size
except ImportError:
    # Fallback if gpu_parallel_games not available
    def get_optimal_batch_size(board_type=None, num_players=2, **kwargs):
        # Conservative defaults based on board complexity
        if board_type in ("hexagonal", "square19"):
            return 128  # Large boards need smaller batches
        return 256  # Small boards can use larger batches

# Session 17.24 (Jan 2026): Node queue tracking for load balancing
# Session 17.38 (Jan 2026): NodeConfigTracker for multi-config per node
try:
    from app.coordination.node_allocator import (
        get_node_queue_tracker,
        get_node_config_tracker,
    )
    HAS_QUEUE_TRACKER = True
    HAS_CONFIG_TRACKER = True
except ImportError:
    HAS_QUEUE_TRACKER = False
    HAS_CONFIG_TRACKER = False
    def get_node_queue_tracker():
        """Fallback when node_allocator not available."""
        return None
    def get_node_config_tracker():
        """Fallback when node_allocator not available."""
        return None

# Import mixin for consolidated event handling (Dec 28, 2025)
from scripts.p2p.p2p_mixin_base import EventSubscriptionMixin

if TYPE_CHECKING:
    from ..models import ClusterJob, ImprovementLoopState, NodeInfo

logger = logging.getLogger(__name__)


@dataclass
class JobManagerStats:
    """Statistics for job manager monitoring.

    December 27, 2025: Added to track job lifecycle events for observability.
    December 29, 2025: Added reassignment stats for Phase 15.1.9.
    """

    jobs_spawned: int = 0
    jobs_completed: int = 0
    jobs_failed: int = 0
    jobs_cancelled: int = 0
    nodes_recovered: int = 0
    hosts_offline: int = 0
    hosts_online: int = 0
    # Phase 15.1.9: Reassignment tracking
    jobs_reassigned: int = 0
    jobs_orphaned: int = 0
    reassignment_failures: int = 0
    # Phase 15.1.7: Dispatch retry tracking
    dispatch_retries: int = 0
    dispatch_escalations: int = 0

# Event emission helper - imported lazily to avoid circular imports
# Dec 2025: Added thread-safe initialization to prevent race conditions
_emit_event: Callable[[str, dict], None] | None = None
_event_emitter_lock = threading.Lock()

# ============================================
# Raft Job Assignment Integration (Dec 30, 2025 - P5.3)
# ============================================

# Cached Raft job assignments
_raft_job_assignments: Any | None = None
_raft_job_assignments_available: bool | None = None


def _check_raft_job_assignments_available() -> bool:
    """Check if Raft job assignments are available.

    Returns True if P2P orchestrator is running with Raft enabled
    and ReplicatedJobAssignments is accessible.

    Dec 30, 2025 (P5.3): Added for Job Assignment Consistency.
    """
    global _raft_job_assignments_available, _raft_job_assignments

    if _raft_job_assignments_available is not None:
        return _raft_job_assignments_available

    try:
        from app.p2p.constants import RAFT_ENABLED
        from app.p2p.raft_state import PYSYNCOBJ_AVAILABLE

        if not RAFT_ENABLED or not PYSYNCOBJ_AVAILABLE:
            logger.debug("Raft job assignments disabled: RAFT_ENABLED=%s", RAFT_ENABLED)
            _raft_job_assignments_available = False
            return False

        # Try to get job assignments from P2P orchestrator
        try:
            from scripts.p2p_orchestrator import P2POrchestrator

            orchestrator = getattr(P2POrchestrator, "_instance", None)
            if orchestrator is None:
                logger.debug("Raft job assignments: P2P orchestrator not running")
                _raft_job_assignments_available = False
                return False

            raft_initialized = getattr(orchestrator, "_raft_initialized", False)
            if not raft_initialized:
                logger.debug("Raft job assignments: Raft not initialized")
                _raft_job_assignments_available = False
                return False

            raft_ja = getattr(orchestrator, "_raft_job_assignments", None)
            if raft_ja is None:
                logger.debug("Raft job assignments: ReplicatedJobAssignments not available")
                _raft_job_assignments_available = False
                return False

            if not getattr(raft_ja, "is_ready", False):
                logger.debug("Raft job assignments: ReplicatedJobAssignments not ready")
                _raft_job_assignments_available = False
                return False

            _raft_job_assignments = raft_ja
            _raft_job_assignments_available = True
            logger.info(
                "Raft job assignments available (leader: %s)",
                getattr(raft_ja, "leader_address", "unknown"),
            )
            return True

        except ImportError:
            _raft_job_assignments_available = False
            return False

    except ImportError:
        _raft_job_assignments_available = False
        return False
    except Exception as e:
        logger.warning("Error checking Raft job assignments: %s", e)
        _raft_job_assignments_available = False
        return False


def reset_raft_job_assignments_cache() -> None:
    """Reset the Raft job assignments cache."""
    global _raft_job_assignments_available, _raft_job_assignments
    _raft_job_assignments_available = None
    _raft_job_assignments = None


def _record_raft_job_assignment(
    job_id: str,
    node_id: str,
    job_type: str,
    board_type: str = "",
    num_players: int = 0,
    **extra_data: Any,
) -> bool:
    """Record a job assignment in Raft (Dec 30, 2025 - P5.3).

    This provides cluster-wide visibility of job assignments.

    Args:
        job_id: Unique job identifier
        node_id: Node the job is assigned to
        job_type: Type of job (selfplay, training, etc.)
        board_type: Board type if applicable
        num_players: Number of players if applicable
        **extra_data: Additional job metadata

    Returns:
        True if recorded successfully, False otherwise
    """
    if not _check_raft_job_assignments_available():
        return False

    try:
        job_data = {
            "job_type": job_type,
            "board_type": board_type,
            "num_players": num_players,
            **extra_data,
        }
        return _raft_job_assignments.assign_job(job_id, node_id, job_data)
    except Exception as e:
        logger.warning("Failed to record Raft job assignment for %s: %s", job_id, e)
        return False


def _start_raft_job(job_id: str) -> bool:
    """Mark a Raft job as started/running."""
    if not _check_raft_job_assignments_available():
        return False

    try:
        return _raft_job_assignments.start_job(job_id)
    except Exception as e:
        logger.warning("Failed to start Raft job %s: %s", job_id, e)
        return False


def _complete_raft_job(job_id: str, result: dict[str, Any] | None = None) -> bool:
    """Mark a Raft job as completed."""
    if not _check_raft_job_assignments_available():
        return False

    try:
        return _raft_job_assignments.complete_job(job_id, result)
    except Exception as e:
        logger.warning("Failed to complete Raft job %s: %s", job_id, e)
        return False


def _fail_raft_job(job_id: str, error: str = "") -> bool:
    """Mark a Raft job as failed."""
    if not _check_raft_job_assignments_available():
        return False

    try:
        return _raft_job_assignments.fail_job(job_id, error)
    except Exception as e:
        logger.warning("Failed to fail Raft job %s: %s", job_id, e)
        return False

# Default P2P port - cached to avoid repeated imports
_default_p2p_port: int | None = None


def _get_default_port() -> int:
    """Get the default P2P port from centralized config.

    Dec 2025: Replaced hardcoded 8770 with centralized constant.
    Falls back to 8770 if constants not available.
    """
    global _default_p2p_port
    if _default_p2p_port is None:
        try:
            from scripts.p2p.constants import DEFAULT_PORT
            _default_p2p_port = DEFAULT_PORT
        except ImportError:
            _default_p2p_port = 8770  # Fallback if constants unavailable
    return _default_p2p_port


def _get_event_emitter() -> Callable[[str, dict], None] | None:
    """Get the event emitter function, initializing if needed (thread-safe).

    Dec 2025: Fixed race condition where multiple threads could race to
    initialize _emit_event. Now uses double-check locking pattern.
    """
    global _emit_event
    # Fast path - already initialized
    if _emit_event is not None:
        return _emit_event

    # Slow path with lock
    with _event_emitter_lock:
        if _emit_event is None:
            try:
                from app.coordination.event_router import emit_sync
                _emit_event = emit_sync
            except ImportError:
                # Event system not available
                logger.debug("Event router not available, job events will not be emitted")
    return _emit_event


class JobManager(EventSubscriptionMixin):
    """Manages job spawning and lifecycle for P2P orchestrator.

    Inherits from EventSubscriptionMixin for consolidated event handling (Dec 28, 2025).

    Responsibilities:
    - Spawn selfplay jobs (GPU, hybrid, heuristic)
    - Spawn training jobs (local and distributed)
    - Spawn export/tournament jobs
    - Track running jobs per node
    - Monitor job status and cleanup
    - Provide job metrics
    - Handle HOST_OFFLINE, HOST_ONLINE, NODE_RECOVERED events

    Usage:
        job_mgr = JobManager(
            ringrift_path="/path/to/ringrift",
            node_id="node1",
            peers={},
            peers_lock=threading.Lock(),
            active_jobs={},
            jobs_lock=threading.Lock(),
        )

        # Subscribe to events
        job_mgr.subscribe_to_events()

        # Spawn a selfplay job (default: GPU Gumbel MCTS for high quality)
        await job_mgr.run_gpu_selfplay_job(
            job_id="job123",
            board_type="hex8",
            num_players=2,
            num_games=100,
            engine_mode="gumbel-mcts"  # High-quality GPU-accelerated selfplay
        )

        # Get job status
        job_count = job_mgr.get_job_count_for_node("node1")
    """

    # Engine modes that require search (use generate_gumbel_selfplay.py)
    # January 2026: Added missing variants (gumbel-mcts-only, mcts-only, etc.)
    # that were falling through to CPU-only run_self_play_soak.py
    SEARCH_ENGINE_MODES = {
        "maxn", "brs", "mcts", "mcts-only",
        "gumbel-mcts", "gumbel-mcts-only", "gumbel", "gpu-gumbel",
        "policy-only", "nn-descent", "descent-only", "nn-minimax",
        "diverse",  # These resolve to GPU modes
    }

    # GPU-required engine modes (require CUDA or MPS) - December 2025
    # These modes use neural network inference and require GPU acceleration
    GPU_REQUIRED_ENGINE_MODES = {
        "gumbel-mcts", "mcts", "nnue-guided", "policy-only",
        "nn-minimax", "nn-descent", "gnn", "hybrid",
        "gmo", "ebmo", "ig-gmo", "cage",
        "multigame-gumbel",  # Feb 2026: batched multi-game Gumbel MCTS
    }

    # CPU-compatible engine modes (can run on any node)
    CPU_COMPATIBLE_ENGINE_MODES = {
        "heuristic-only", "heuristic", "random", "random-only",
        "descent-only", "maxn", "brs",
    }

    # Weak GPU models that should prefer CPU selfplay over Gumbel MCTS
    # Jan 18, 2026: These GPUs have limited VRAM (<12GB) and slower compute,
    # making Gumbel MCTS (800 simulations) too slow for efficient selfplay.
    # Route these to heuristic/CPU selfplay instead.
    WEAK_GPU_MODELS = {
        "3060", "3060 ti", "3060ti", "rtx 3060",  # 12GB VRAM, slow
        "4060", "4060 ti", "4060ti", "rtx 4060",  # 8GB VRAM, slow
        "2060", "2070", "2080",  # Older Turing, limited VRAM
        "1080", "1080 ti", "1070", "1660",  # Pascal/Turing consumer
    }

    # Mixin type identifier (required by EventSubscriptionMixin)
    MIXIN_TYPE = "job_manager"

    # Phase 15.1.9: Job heartbeat timeout and reassignment
    # Jobs without heartbeat for this duration are considered abandoned
    JOB_HEARTBEAT_TIMEOUT: float = float(os.environ.get(
        "RINGRIFT_JOB_HEARTBEAT_TIMEOUT",
        str(TaskLifecycleDefaults.HEARTBEAT_TIMEOUT * 5)  # 5 minutes default (5 * 60s)
    ))

    # Maximum reassignment attempts before job is considered permanently failed
    MAX_REASSIGNMENT_ATTEMPTS: int = int(os.environ.get(
        "RINGRIFT_MAX_JOB_REASSIGNMENT_ATTEMPTS", "3"
    ))

    def __init__(
        self,
        ringrift_path: str,
        node_id: str,
        peers: dict[str, Any],
        peers_lock: threading.Lock,
        active_jobs: dict[str, dict[str, Any]],
        jobs_lock: threading.Lock,
        improvement_loop_state: dict[str, Any] | None = None,
        distributed_tournament_state: dict[str, Any] | None = None,
    ):
        """Initialize the JobManager.

        Args:
            ringrift_path: Path to RingRift repository root
            node_id: Current node's ID
            peers: Dict of node_id -> NodeInfo
            peers_lock: Lock for thread-safe peer access
            active_jobs: Dict tracking running jobs by type
            jobs_lock: Lock for thread-safe job access
            improvement_loop_state: Optional improvement loop state dict
            distributed_tournament_state: Optional tournament state dict
        """
        self.ringrift_path = ringrift_path
        self.node_id = node_id
        self.peers = peers
        self.peers_lock = peers_lock
        self.active_jobs = active_jobs
        self.jobs_lock = jobs_lock
        # CRITICAL: Use `is None` check, NOT `or {}`, because empty dicts are falsy
        # and `{} or {}` returns a NEW dict instead of the original empty dict!
        # This caused a bug where the tournament state wasn't shared between handler and manager.
        self.improvement_loop_state = improvement_loop_state if improvement_loop_state is not None else {}
        self.distributed_tournament_state = distributed_tournament_state if distributed_tournament_state is not None else {}

        # Event subscription state (December 2025)
        # Dec 28, 2025: Added _subscription_lock to prevent race conditions during subscribe_to_events
        self._subscribed = False
        self._subscription_lock = threading.Lock()

        # Statistics tracking (December 27, 2025)
        self.stats = JobManagerStats()

        # December 28, 2025: Track active subprocess handles for graceful shutdown
        # Maps job_id -> asyncio.subprocess.Process
        self._active_processes: dict[str, asyncio.subprocess.Process] = {}
        self._processes_lock = threading.Lock()

        # Phase 15.1.9: Job heartbeat tracking for abandoned job detection
        # Maps job_id -> (last_heartbeat_time, reassignment_count, next_reassignment_time)
        # January 2026: Added next_reassignment_time for exponential backoff
        self._job_heartbeats: dict[str, tuple[float, int, float]] = {}
        self._heartbeats_lock = threading.Lock()

        # January 2026 Sprint 6: Spawn verification callback for SelfplayScheduler integration
        # Called when a job is spawned to register it for verification tracking
        self._spawn_registration_callback: Callable[[str, str, str], None] | None = None

    # =========================================================================
    # Spawn Verification (January 2026 Sprint 6)
    # =========================================================================

    def set_spawn_registration_callback(
        self, callback: Callable[[str, str, str], None] | None
    ) -> None:
        """Set the callback for registering pending job spawns.

        January 2026 Sprint 6: Part of the Job Spawn Verification system.

        The callback is called when a job is successfully spawned with:
        - job_id: The unique job identifier
        - node_id: The node the job was dispatched to
        - config_key: The config key (e.g., "hex8_2p")

        Args:
            callback: The callback function, or None to disable registration
        """
        self._spawn_registration_callback = callback

    def _register_spawn(self, job_id: str, node_id: str, config_key: str) -> None:
        """Register a pending job spawn for verification tracking.

        January 2026 Sprint 6: Called when a job is spawned to register it
        with the SelfplayScheduler for verification tracking.

        Args:
            job_id: The unique job identifier
            node_id: The node the job was dispatched to
            config_key: The config key (e.g., "hex8_2p")
        """
        if self._spawn_registration_callback is not None:
            try:
                self._spawn_registration_callback(job_id, node_id, config_key)
            except Exception as e:
                logger.warning(f"Failed to register spawn for job {job_id}: {e}")

    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Get the status of a job by ID.

        January 2026 Sprint 6: Used by spawn verification to check if jobs are running.

        Args:
            job_id: The job ID to look up

        Returns:
            Job info dict with 'status' field, or None if not found
        """
        with self.jobs_lock:
            for job_type, jobs in self.active_jobs.items():
                if job_id in jobs:
                    return jobs[job_id]
        return None

    def get_jobs_for_node(self, node_id: str) -> list[dict[str, Any]]:
        """Get all jobs assigned to a specific node.

        January 2026 Sprint 6: Used by PredictiveScalingLoop to check if a node
        has pending work before spawning preemptive jobs.

        Args:
            node_id: The node identifier to filter by

        Returns:
            List of job info dicts assigned to this node
        """
        result = []
        with self.jobs_lock:
            for job_type, jobs in self.active_jobs.items():
                for job_id, job_info in jobs.items():
                    job_node = job_info.get("node_id") or job_info.get("worker_id")
                    if job_node == node_id:
                        result.append({
                            **job_info,
                            "job_id": job_id,
                            "job_type": job_type,
                        })
        return result

    def get_active_configs_for_node(self, node_id: str) -> list[str]:
        """Get the config keys of selfplay jobs running on a node.

        Session 17.34: Used for multi-config per node support.
        Returns config keys like ["hex8_2p", "square8_3p"] for selfplay jobs.

        Args:
            node_id: The node identifier

        Returns:
            List of config keys (e.g., "hex8_2p") for selfplay jobs on this node
        """
        config_keys = []
        with self.jobs_lock:
            selfplay_jobs = self.active_jobs.get("selfplay", {})
            for job_id, job_info in selfplay_jobs.items():
                job_node = job_info.get("node_id") or job_info.get("worker_id")
                if job_node == node_id:
                    board_type = job_info.get("board_type", "")
                    num_players = job_info.get("num_players", 2)
                    if board_type:
                        config_keys.append(f"{board_type}_{num_players}p")
        return config_keys

    async def dispatch_selfplay_job(
        self,
        node_id: str,
        job_id: str,
        board_type: str,
        num_players: int,
        num_games: int,
        preemptive: bool = False,
        model_version: str | None = None,
        engine_mode: str | None = None,
    ) -> dict[str, Any]:
        """Dispatch a selfplay job to a specific node.

        January 2026 Sprint 6: Used by PredictiveScalingLoop for preemptive job spawning.
        Session 17.22: Added model_version for architecture selection feedback loop.
        Jan 12, 2026: Added engine_mode for harness diversity across cluster.

        Args:
            node_id: The target node identifier
            job_id: Unique job identifier
            board_type: Board type (hex8, square8, etc.)
            num_players: Number of players (2, 3, or 4)
            num_games: Number of games to generate
            preemptive: Whether this is a preemptive spawn (for tracking)
            model_version: Architecture version (e.g., "v5", "v2", "v5-heavy").
                          If None, uses default "v5".
            engine_mode: Engine mode (e.g., "mixed", "gumbel-mcts", "nnue-guided").
                        If None, uses default "mixed" for harness diversity.

        Returns:
            Dict with 'success' bool and optional 'error' message
        """
        try:
            # Find peer info for the node
            with self.peers_lock:
                peer = self.peers.get(node_id)

            if peer is None:
                return {"success": False, "error": f"Node {node_id} not found in peers"}

            # Construct the selfplay request
            # Session 17.22: Include model_version for architecture selection feedback loop
            # Jan 21, 2026: Default to v2 (most canonical models are v2)
            effective_version = model_version or "v2"
            model_path = self._get_model_path_for_version(
                board_type, num_players, effective_version
            )

            # Sprint 17.9: Handle model not found - prevent cascade failures
            if model_path is None:
                config_key = f"{board_type}_{num_players}p"
                logger.warning(
                    f"[JobManager] Cannot dispatch selfplay for {config_key}: model not found"
                )
                return {
                    "success": False,
                    "error": "model_not_found",
                    "config_key": config_key,
                    "model_version": effective_version,
                }

            # Jan 12, 2026: Default to "mixed" for harness diversity across cluster
            # Jan 18, 2026: Route weak GPUs to heuristic selfplay instead of Gumbel
            requested_engine_mode = engine_mode or "mixed"
            effective_engine_mode = self._get_effective_engine_mode(
                requested_engine_mode, peer
            )

            config = {
                "board_type": board_type,
                "num_players": num_players,
                "num_games": num_games,
                "auto_assigned": True,
                "preemptive": preemptive,
                "job_id": job_id,
                "reason": f"preemptive_spawn_{board_type}_{num_players}p",
                "model_version": effective_version,
                "model_path": model_path,
                "engine_mode": effective_engine_mode,
            }

            # Use HTTP to dispatch to the node's selfplay endpoint
            from aiohttp import ClientSession, ClientTimeout

            peer_ip = getattr(peer, "tailscale_ip", None) or getattr(peer, "public_ip", None)
            peer_port = getattr(peer, "port", 8770)
            if not peer_ip:
                return {"success": False, "error": f"No IP for node {node_id}"}

            # Sprint 15.1.1: Check per-transport circuit breaker before attempting
            if not check_peer_transport_circuit(peer_ip, "http"):
                logger.debug(f"Skipping preemptive spawn to {node_id}: HTTP circuit breaker is OPEN")
                return {"success": False, "error": "circuit_open", "node_id": node_id}

            url = f"http://{peer_ip}:{peer_port}/selfplay/start"
            timeout = ClientTimeout(total=30)

            async with ClientSession(timeout=timeout) as session:
                async with session.post(url, json=config) as resp:
                    if resp.status == 200:
                        # Sprint 15.1.1: Record success for per-transport CB
                        record_peer_transport_success(peer_ip, "http")

                        # Register the spawn for verification tracking
                        config_key = f"{board_type}_{num_players}p"
                        self._register_spawn(job_id, node_id, config_key)

                        # Session 17.24: Track job dispatch for load balancing
                        if HAS_QUEUE_TRACKER:
                            try:
                                tracker = get_node_queue_tracker()
                                if tracker:
                                    tracker.on_job_dispatched(node_id)
                            except Exception as e:
                                # Non-critical tracking, log at debug level
                                logger.debug(f"[JobManager] Queue tracker dispatch failed: {e}")

                        # Session 17.38: Track config on node for multi-config support
                        if HAS_CONFIG_TRACKER:
                            try:
                                config_tracker = get_node_config_tracker()
                                if config_tracker:
                                    # Get GPU VRAM from peer info (fallback to conservative default)
                                    gpu_vram = int(
                                        getattr(peer, "gpu_vram_gb", 0)
                                        or getattr(peer, "memory_gb", 0)
                                        or 24  # Conservative default
                                    )
                                    config_tracker.add_config(node_id, config_key)
                                    logger.debug(
                                        f"[JobManager] Added config {config_key} to {node_id} "
                                        f"(vram={gpu_vram}GB)"
                                    )
                            except Exception as e:
                                # Non-critical tracking, log at debug level
                                logger.debug(f"[JobManager] Config tracker dispatch failed: {e}")

                        # Track the job
                        # Session 17.22: Include model_version for architecture feedback loop
                        with self.jobs_lock:
                            if "selfplay" not in self.active_jobs:
                                self.active_jobs["selfplay"] = {}
                            self.active_jobs["selfplay"][job_id] = {
                                "node_id": node_id,
                                "board_type": board_type,
                                "num_players": num_players,
                                "num_games": num_games,
                                "status": "running",
                                "started_at": time.time(),
                                "preemptive": preemptive,
                                "model_version": effective_version,
                            }

                        return {"success": True}
                    else:
                        # Sprint 15.1.1: Record failure for per-transport CB
                        record_peer_transport_failure(peer_ip, "http")
                        body = await resp.text()
                        return {"success": False, "error": f"HTTP {resp.status}: {body[:100]}"}

        except Exception as e:
            # Sprint 15.1.1: Record failure on exception (extract IP if available)
            if "peer_ip" in dir() and peer_ip:
                record_peer_transport_failure(peer_ip, "http")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Path Helpers (December 2025)
    # =========================================================================

    def _get_ai_service_path(self) -> str:
        """Get the path to the ai-service directory.

        Handles both cases:
        - ringrift_path = /path/to/RingRift (root directory)
        - ringrift_path = /path/to/RingRift/ai-service (already ai-service)

        Returns:
            Path to ai-service directory.
        """
        if self.ringrift_path.rstrip("/").endswith("ai-service"):
            return self.ringrift_path
        return os.path.join(self.ringrift_path, "ai-service")

    def _get_script_path(self, script_name: str) -> str:
        """Get the full path to a script in ai-service/scripts/.

        Args:
            script_name: Name of the script (e.g., "run_gpu_selfplay.py")

        Returns:
            Full path to the script.
        """
        return os.path.join(self._get_ai_service_path(), "scripts", script_name)

    def _get_data_path(self, *subpath: str) -> str:
        """Get a path within ai-service/data/.

        Args:
            *subpath: Path components under data/ (e.g., "training", "hex8_2p.npz")

        Returns:
            Full path to the data file/directory.
        """
        return os.path.join(self._get_ai_service_path(), "data", *subpath)

    def _get_model_path_for_version(
        self,
        board_type: str,
        num_players: int,
        model_version: str = "v2",
    ) -> str | None:
        """Get the model path for a specific architecture version.

        Session 17.22: Added for architecture selection feedback loop.
        Jan 21, 2026: Changed default from v5 to v2 (most canonical models are v2).
        This allows selfplay to use different architecture versions based
        on their Elo performance (tracked by ArchitectureTracker).

        Sprint 17.9 (Jan 5, 2026): Added pre-flight model existence check.
        Returns None if model not found, emits MODEL_NOT_FOUND event to
        trigger model sync. Prevents cascade failures when stale model
        is deleted.

        Args:
            board_type: Board type (hex8, square8, etc.)
            num_players: Number of players (2, 3, or 4)
            model_version: Architecture version (e.g., "v5", "v2", "v5-heavy")

        Returns:
            Path to the canonical model file for the given configuration,
            or None if no model exists. Falls back to default (no version
            suffix) if version-specific model doesn't exist.
        """
        models_dir = os.path.join(self._get_ai_service_path(), "models")
        config_key = f"{board_type}_{num_players}p"

        # Jan 21, 2026: Fixed version lookup logic
        # v2 models have no suffix (canonical_hex8_2p.pth)
        # Other versions have suffix (canonical_hex8_2p_v5-heavy.pth)
        if model_version and model_version not in ("v2", "default", ""):
            versioned_name = f"canonical_{board_type}_{num_players}p_{model_version}.pth"
            versioned_path = os.path.join(models_dir, versioned_name)
            if os.path.exists(versioned_path):
                logger.debug(
                    f"Using versioned model for {config_key}: {versioned_name}"
                )
                return versioned_path
            # If versioned model doesn't exist, fall back to default

        # Default: canonical_hex8_2p.pth (v2 architecture)
        default_name = f"canonical_{board_type}_{num_players}p.pth"
        default_path = os.path.join(models_dir, default_name)

        # Sprint 17.9: Pre-flight existence check - emit event if missing
        if not os.path.exists(default_path):
            logger.warning(
                f"[JobManager] Model not found for {config_key}: {default_path}"
            )
            self._emit_model_not_found(
                board_type=board_type,
                num_players=num_players,
                model_version=model_version,
                expected_path=default_path,
            )
            return None

        return default_path

    def _emit_model_not_found(
        self,
        board_type: str,
        num_players: int,
        model_version: str,
        expected_path: str,
    ) -> None:
        """Emit MODEL_NOT_FOUND event to trigger model sync.

        Sprint 17.9 (Jan 5, 2026): Added for stale model detection.
        Subscribers (e.g., sync_router, model_lifecycle_coordinator) can
        react by initiating model sync from other nodes.

        Args:
            board_type: Board type (hex8, square8, etc.)
            num_players: Number of players (2, 3, or 4)
            model_version: Architecture version requested
            expected_path: Path where model was expected
        """
        config_key = f"{board_type}_{num_players}p"
        self._safe_emit_event(
            "MODEL_NOT_FOUND",
            {
                "config_key": config_key,
                "board_type": board_type,
                "num_players": num_players,
                "model_version": model_version,
                "expected_path": expected_path,
                "node_id": self.node_id,
                "timestamp": time.time(),
            },
        )

    def _should_use_gpu_tree(self) -> bool:
        """Check if GPU tree mode should be enabled for this node.

        Checks the distributed_hosts.yaml config for this node's disable_gpu_tree setting.
        GPU tree is enabled by default unless explicitly disabled for the node (e.g., vGPU nodes).

        Returns:
            True if GPU tree should be used, False if it should be disabled.
        """
        try:
            import yaml
            config_path = os.path.join(self._get_ai_service_path(), "config", "distributed_hosts.yaml")
            if not os.path.exists(config_path):
                # No config file, default to GPU tree enabled
                return True

            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            hosts = config.get("hosts", {})
            node_config = hosts.get(self.node_id, {})

            # disable_gpu_tree: true means DON'T use GPU tree
            # Default is False (use GPU tree)
            return not node_config.get("disable_gpu_tree", False)
        except Exception as e:
            logger.debug(f"Could not load node config for GPU tree check: {e}")
            # On error, default to GPU tree enabled
            return True

    def _check_yaml_gpu_config(self) -> bool:
        """Check if YAML config indicates this node has a GPU.

        Used as fallback when runtime GPU detection fails (e.g., vGPU, containers).
        Returns True if node has gpu, gpu_vram_gb, or role containing 'gpu'.

        Session 17.50 (Jan 2026): Added to fix GPU nodes running CPU selfplay
        when torch.cuda.is_available() returns False due to driver issues.
        """
        try:
            from app.config.cluster_config import get_config_cache
            config = get_config_cache().get_config()
            host_cfg = config.hosts_raw.get(self.node_id, {})

            # Check multiple indicators
            gpu_name = host_cfg.get("gpu", "")
            gpu_vram = host_cfg.get("gpu_vram_gb", 0)
            role = host_cfg.get("role", "")

            has_gpu = bool(gpu_name) or gpu_vram > 0 or "gpu" in role.lower()

            if has_gpu:
                logger.debug(
                    f"[YAML GPU] Node {self.node_id}: gpu={gpu_name}, "
                    f"vram={gpu_vram}GB, role={role}"
                )
            return has_gpu
        except Exception as e:
            logger.debug(f"Could not check YAML GPU config: {e}")
            return False

    # =========================================================================
    # GPU Capability Helpers (December 2025)
    # =========================================================================

    def _engine_mode_requires_gpu(self, engine_mode: str) -> bool:
        """Check if an engine mode requires GPU acceleration.

        Args:
            engine_mode: The engine mode string (e.g., "gumbel-mcts", "heuristic-only")

        Returns:
            True if the engine mode requires GPU (CUDA or MPS), False otherwise.

        December 2025: Added to prevent dispatching GPU-required jobs to CPU-only nodes.
        """
        if not engine_mode:
            return False
        mode_lower = engine_mode.lower().strip()
        return mode_lower in self.GPU_REQUIRED_ENGINE_MODES

    def _worker_has_gpu(self, worker: Any) -> bool:
        """Check if a worker node has GPU capability.

        Args:
            worker: Worker node info object (NodeInfo or similar)

        Returns:
            True if the worker has GPU (CUDA-capable), False otherwise.

        December 2025: Added for GPU-aware job dispatch.
        """
        # Try has_cuda_gpu() method first (NodeInfo from scripts/p2p/models.py)
        if hasattr(worker, "has_cuda_gpu"):
            return worker.has_cuda_gpu()

        # Try is_gpu_node() method (also on NodeInfo)
        if hasattr(worker, "is_gpu_node"):
            return worker.is_gpu_node()

        # Try gpu_info attribute directly
        if hasattr(worker, "gpu_info"):
            gpu_info = worker.gpu_info
            if gpu_info is not None:
                gpu_count = getattr(gpu_info, "gpu_count", 0)
                return gpu_count > 0

        # Try node_type or capabilities attribute
        if hasattr(worker, "node_type"):
            return worker.node_type in ("gpu", "training", "h100", "a100", "gh200")

        if hasattr(worker, "capabilities"):
            caps = worker.capabilities or {}
            return caps.get("gpu", False) or caps.get("cuda", False)

        # Fallback: assume no GPU if we can't determine
        logger.debug(f"Cannot determine GPU capability for worker: {worker}")
        return False

    def _is_weak_gpu(self, worker: Any) -> bool:
        """Check if a worker has a weak GPU that should prefer CPU selfplay.

        Weak GPUs (3060Ti, 4060Ti, etc.) have limited VRAM and slower compute,
        making Gumbel MCTS (800 simulations) too slow for efficient selfplay.
        These nodes should be routed to heuristic/CPU selfplay instead.

        Args:
            worker: Worker node info object (NodeInfo or similar)

        Returns:
            True if the worker has a weak GPU that should use CPU selfplay.

        Jan 18, 2026: Added to route weak GPUs to CPU selfplay for better throughput.
        """
        # Get GPU name from various possible attributes
        gpu_name = ""
        if hasattr(worker, "gpu"):
            gpu_name = str(getattr(worker, "gpu", "") or "")
        elif hasattr(worker, "gpu_info"):
            gpu_info = worker.gpu_info
            if gpu_info:
                gpu_name = str(getattr(gpu_info, "name", "") or "")
        elif hasattr(worker, "capabilities"):
            caps = worker.capabilities or {}
            gpu_name = str(caps.get("gpu_name", "") or caps.get("gpu", "") or "")

        if not gpu_name:
            return False

        # Check if GPU model is in weak GPU list
        gpu_lower = gpu_name.lower()
        for weak_model in self.WEAK_GPU_MODELS:
            if weak_model in gpu_lower:
                logger.debug(f"Weak GPU detected: {gpu_name} on worker {worker}")
                return True

        return False

    def _get_effective_engine_mode(
        self, engine_mode: str, worker: Any
    ) -> str:
        """Get effective engine mode based on worker GPU capability.

        Routes weak GPU nodes to CPU selfplay (heuristic) instead of
        GPU-intensive modes like Gumbel MCTS.

        Args:
            engine_mode: Requested engine mode (e.g., "gumbel-mcts", "mixed")
            worker: Worker node info object

        Returns:
            Effective engine mode - may be changed to "heuristic" for weak GPUs.

        Jan 18, 2026: Added to improve cluster efficiency by routing weak GPUs
        to faster CPU selfplay instead of slow Gumbel MCTS.
        """
        # If requested mode requires GPU and worker has weak GPU, use heuristic
        if self._engine_mode_requires_gpu(engine_mode) and self._is_weak_gpu(worker):
            logger.info(
                f"Routing weak GPU worker to heuristic selfplay "
                f"(requested: {engine_mode})"
            )
            return "heuristic"

        return engine_mode

    # =========================================================================
    # Phase 15: Fair Node Selection (Session 17.33, Jan 5, 2026)
    # =========================================================================

    def _sort_workers_by_load(self, workers: list[Any]) -> list[Any]:
        """Sort workers by pending job count for fair distribution.

        Session 17.33 Phase 15: Implements fair node selection to prevent fast
        nodes from getting disproportionate work. Workers are sorted ascending
        by their pending job count so that underutilized nodes get jobs first.

        Uses NodeQueueTracker from app.coordination.node_allocator to track
        pending jobs per node.

        Args:
            workers: List of worker node info objects

        Returns:
            Workers sorted by pending job count (least loaded first)

        Example:
            >>> workers = [node1, node2, node3]
            >>> sorted_workers = self._sort_workers_by_load(workers)
            # node with fewest pending jobs is first
        """
        if not HAS_QUEUE_TRACKER:
            # No tracker available - return original order
            return workers

        if not workers:
            return workers

        tracker = get_node_queue_tracker()
        if tracker is None:
            return workers

        def get_worker_load(worker: Any) -> int:
            """Get pending job count for a worker."""
            worker_id = getattr(worker, "node_id", None) or str(worker)
            return tracker.get_pending_jobs(worker_id)

        # Sort by pending jobs ascending (least loaded first)
        sorted_workers = sorted(workers, key=get_worker_load)

        # Log fairness info
        if len(sorted_workers) > 1:
            first_id = getattr(sorted_workers[0], "node_id", "?")
            last_id = getattr(sorted_workers[-1], "node_id", "?")
            first_load = get_worker_load(sorted_workers[0])
            last_load = get_worker_load(sorted_workers[-1])
            if first_load != last_load:
                logger.debug(
                    f"[FairSelection] Sorted {len(sorted_workers)} workers by load: "
                    f"{first_id} ({first_load} jobs) → {last_id} ({last_load} jobs)"
                )

        return sorted_workers

    # =========================================================================
    # Subprocess Environment Setup (December 2025)
    # =========================================================================

    def _get_subprocess_env(self, include_shadow_skip: bool = True) -> dict[str, str]:
        """Create standard environment for subprocess spawning.

        This helper ensures consistent environment setup across all subprocess calls,
        including:
        - PYTHONPATH pointing to ai-service
        - RINGRIFT_SKIP_SHADOW_CONTRACTS to skip validation overhead
        - RINGRIFT_ALLOW_PENDING_GATE to bypass parity gate on cluster nodes

        December 29, 2025: Added to fix parity gate blocking on cluster nodes.
        Cluster nodes lack Node.js runtime, so TypeScript parity gates fail with
        "pending_gate" status. This environment variable allows selfplay to proceed
        without TS validation on these nodes.

        Args:
            include_shadow_skip: Whether to include RINGRIFT_SKIP_SHADOW_CONTRACTS
                                (default: True for selfplay, may want False for validation)

        Returns:
            Environment dict with standard RingRift settings.
        """
        env = os.environ.copy()
        env["PYTHONPATH"] = self._get_ai_service_path()

        if include_shadow_skip:
            env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

        # December 29, 2025: Allow pending parity gate status on cluster nodes
        # This enables selfplay on nodes without Node.js (Vast.ai, RunPod, Nebius)
        env["RINGRIFT_ALLOW_PENDING_GATE"] = "1"

        return env

    # =========================================================================
    # Pre-flight Node Validation (December 2025)
    # =========================================================================

    async def _preflight_check_node(
        self, node_id: str, timeout: float | None = None
    ) -> tuple[bool, str]:
        """Quick health check before dispatching job to node.

        Performs a fast SSH probe and checks P2P alive status to avoid
        dispatching jobs to dead or unreachable nodes.

        Args:
            node_id: The node ID to check
            timeout: Maximum time to wait for probe response (uses adaptive timeout if None)

        Returns:
            Tuple of (is_available, reason).
            - (True, "ok") if node is available
            - (False, reason) if node is unavailable with reason string

        December 2025: Added as part of cluster availability fix to prevent
        dispatching jobs to unavailable nodes.
        Sprint 10 (Jan 3, 2026): Uses adaptive timeout based on host history.
        """
        # Check 1: Node in alive peers
        with self.peers_lock:
            if node_id not in self.peers:
                return False, "not_in_peers"
            peer = self.peers.get(node_id)
            if peer and hasattr(peer, "status"):
                if peer.status in ("offline", "dead", "retired"):
                    return False, f"peer_status_{peer.status}"

        # Sprint 10 (Jan 3, 2026): Use adaptive timeout based on host performance history
        effective_timeout = timeout if timeout is not None else get_adaptive_timeout(
            "ssh", node_id, default=5.0
        )

        # Check 2: Quick SSH probe
        try:
            from app.core.ssh import run_ssh_command_async

            result = await asyncio.wait_for(
                run_ssh_command_async(node_id, "echo ok", timeout=effective_timeout),
                timeout=effective_timeout + 1
            )
            if not result or not result.success:
                stderr_preview = (result.stderr[:100] if result and result.stderr else "no result")
                return False, f"ssh_probe_failed: {stderr_preview}"

            return True, "ok"
        except asyncio.TimeoutError:
            return False, "ssh_timeout"
        except ImportError:
            # SSH module not available, skip check
            logger.debug(f"SSH module not available for preflight check of {node_id}")
            return True, "ssh_unavailable_skipped"
        except Exception as e:
            return False, f"preflight_error: {e}"

    # =========================================================================
    # Disk Space Check (December 2025)
    # =========================================================================

    def _check_disk_space(
        self,
        output_dir: str | Path,
        min_free_gb: float | None = None,
    ) -> tuple[bool, str]:
        """Check if there is sufficient disk space before spawning a job.

        Args:
            output_dir: Directory where job output will be written
            min_free_gb: Minimum required free space in GB. If None, reads from
                RINGRIFT_MIN_DISK_FREE_GB env var (default: 10.0 GB)

        Returns:
            Tuple of (has_space, reason).
            - (True, "ok") if sufficient disk space
            - (False, reason) if insufficient disk space

        December 2025: Added as part of autonomous loop fixes to prevent jobs
        from crashing silently when disk fills up.
        """
        # Get minimum free space threshold from env or default
        if min_free_gb is None:
            try:
                min_free_gb = float(os.environ.get("RINGRIFT_MIN_DISK_FREE_GB", "10.0"))
            except (ValueError, TypeError):
                min_free_gb = 10.0

        try:
            output_path = Path(output_dir)

            # Find existing parent directory to check
            check_path = output_path
            while not check_path.exists() and check_path.parent != check_path:
                check_path = check_path.parent

            if not check_path.exists():
                # Can't determine disk space if no parent exists
                logger.warning(f"Cannot check disk space: no existing parent for {output_dir}")
                return True, "path_not_found_skipped"

            # Get disk usage statistics
            disk_usage = shutil.disk_usage(check_path)
            free_gb = disk_usage.free / (1024 ** 3)  # Convert bytes to GB

            if free_gb < min_free_gb:
                return False, f"insufficient_disk_space: {free_gb:.1f}GB free < {min_free_gb:.1f}GB required"

            logger.debug(
                f"Disk space check passed for {output_dir}: {free_gb:.1f}GB free "
                f"(min: {min_free_gb:.1f}GB)"
            )
            return True, "ok"

        except OSError as e:
            logger.warning(f"Disk space check failed for {output_dir}: {e}")
            # Return True on error to avoid blocking jobs due to check failures
            return True, f"disk_check_error_skipped: {e}"

    async def _check_disk_space_async(
        self,
        node_id: str,
        output_dir: str,
        min_free_gb: float = 10.0,
        timeout: float = 10.0,
    ) -> tuple[bool, str]:
        """Check disk space on a remote node via SSH.

        Args:
            node_id: The node ID to check
            output_dir: Directory where job output will be written
            min_free_gb: Minimum required free space in GB
            timeout: Maximum time to wait for SSH command

        Returns:
            Tuple of (has_space, reason).
            - (True, "ok") if sufficient disk space
            - (False, reason) if insufficient disk space

        December 2025: Added for remote job dispatch disk validation.
        """
        try:
            from app.core.ssh import run_ssh_command_async

            # df -BG outputs in gigabytes, get available space for output dir
            cmd = f"df -BG {output_dir} 2>/dev/null | tail -1 | awk '{{print $4}}' | tr -d 'G'"

            result = await asyncio.wait_for(
                run_ssh_command_async(node_id, cmd, timeout=timeout),
                timeout=timeout + 1
            )

            if not result or not result.success:
                # Fallback: check root partition if specific dir doesn't exist
                cmd = "df -BG / | tail -1 | awk '{print $4}' | tr -d 'G'"
                result = await asyncio.wait_for(
                    run_ssh_command_async(node_id, cmd, timeout=timeout),
                    timeout=timeout + 1
                )

            if not result or not result.success:
                logger.debug(f"Disk space check failed for {node_id}: SSH command failed")
                return True, "ssh_check_failed_skipped"

            try:
                free_gb = float(result.stdout.strip())
            except (ValueError, TypeError):
                logger.debug(f"Could not parse disk space for {node_id}: {result.stdout}")
                return True, "parse_error_skipped"

            if free_gb < min_free_gb:
                logger.warning(
                    f"Insufficient disk space on {node_id}: {free_gb:.1f}GB free < {min_free_gb:.1f}GB required"
                )
                return False, f"insufficient_disk_space: {free_gb:.1f}GB free < {min_free_gb:.1f}GB required"

            logger.debug(f"Disk space check passed for {node_id}: {free_gb:.1f}GB free")
            return True, "ok"

        except asyncio.TimeoutError:
            return True, "disk_check_timeout_skipped"
        except ImportError:
            # SSH module not available
            return True, "ssh_unavailable_skipped"
        except Exception as e:
            logger.debug(f"Disk space check error for {node_id}: {e}")
            return True, f"disk_check_error_skipped: {e}"

    async def _check_gpu_health(self, node_id: str, timeout: float | None = None) -> tuple[bool, str]:
        """Verify GPU is available and not in error state.

        Runs nvidia-smi on the target node to check for CUDA errors,
        busy GPUs, or other GPU-related issues before dispatching GPU jobs.

        Args:
            node_id: The node ID to check
            timeout: Maximum time to wait for nvidia-smi (uses adaptive timeout if None)

        Returns:
            Tuple of (is_healthy, reason).
            - (True, "ok") if GPU is healthy
            - (True, "no_gpu_on_node") if node doesn't have GPU (skipped)
            - (False, reason) if GPU has issues

        December 2025: Added as part of cluster availability fix to prevent
        dispatching to nodes with CUDA errors.

        January 2026: Skip nvidia-smi on non-GPU nodes (macOS, Hetzner CPU)
        to prevent crashes and false failures.
        Sprint 10 (Jan 3, 2026): Uses adaptive timeout based on host history.
        """
        # Check if target node has GPU capability before running nvidia-smi
        # This prevents crashes on macOS coordinators and CPU-only nodes
        with self.peers_lock:
            node_info = self.peers.get(node_id)

        if node_info is not None:
            has_gpu = self._worker_has_gpu(node_info)
            if not has_gpu:
                logger.debug(f"Skipping GPU health check for non-GPU node {node_id}")
                return True, "no_gpu_on_node"
        elif node_id == self.node_id:
            # Current node - check via platform detection
            import platform
            if platform.system() == "Darwin":
                logger.debug(f"Skipping GPU health check for macOS node {node_id}")
                return True, "macos_no_nvidia_gpu"

        # Sprint 10 (Jan 3, 2026): Use adaptive timeout based on host performance history
        effective_timeout = timeout if timeout is not None else get_adaptive_timeout(
            "ssh", node_id, default=10.0
        )

        try:
            from app.core.ssh import run_ssh_command_async

            result = await asyncio.wait_for(
                run_ssh_command_async(
                    node_id,
                    "nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>&1",
                    timeout=effective_timeout
                ),
                timeout=effective_timeout + 1
            )
            if not result or not result.success:
                stderr = result.stderr if result else "no result"
                return False, f"nvidia_smi_failed: {stderr[:100]}"

            output = (result.stdout or "").lower()
            stderr = (result.stderr or "").lower()

            # Check for CUDA errors
            if "cuda" in stderr and "error" in stderr:
                return False, f"cuda_error: {result.stderr[:100]}"
            if "no devices" in output or "no devices" in stderr:
                return False, "no_gpu_devices"
            if "busy" in stderr or "unavailable" in stderr:
                return False, "gpu_busy_or_unavailable"
            if "failed" in stderr:
                return False, f"nvidia_smi_error: {stderr[:100]}"

            return True, "ok"
        except asyncio.TimeoutError:
            return False, "gpu_check_timeout"
        except ImportError:
            logger.debug(f"SSH module not available for GPU health check of {node_id}")
            return True, "ssh_unavailable_skipped"
        except Exception as e:
            return False, f"gpu_check_error: {e}"

    async def validate_node_for_job(
        self,
        node_id: str,
        requires_gpu: bool = False,
        preflight_timeout: float | None = None,
        gpu_check_timeout: float | None = None,
    ) -> tuple[bool, str]:
        """Validate that a node is suitable for running a job.

        Combines preflight check and optional GPU health check into a single
        validation call that can be used before job dispatch.

        Args:
            node_id: The node ID to validate
            requires_gpu: Whether the job requires GPU (triggers GPU health check)
            preflight_timeout: Timeout for SSH probe (uses adaptive timeout if None)
            gpu_check_timeout: Timeout for GPU health check (uses adaptive timeout if None)

        Returns:
            Tuple of (is_valid, reason).
            - (True, "ok") if node is valid for the job
            - (False, reason) if node cannot accept the job

        December 2025: Convenience method combining preflight and GPU checks.
        Updated Dec 2025: Now checks availability cache first for fast rejection.
        Sprint 10 (Jan 3, 2026): Uses adaptive timeouts based on host history when None.
        """
        # Step 0: Check availability cache first (fast path for known-unavailable nodes)
        try:
            from app.coordination.node_availability_cache import get_availability_cache

            cache = get_availability_cache()
            if not cache.is_available(node_id):
                entry = cache.get_entry(node_id)
                reason = entry.reason.value if entry else "cached_unavailable"
                logger.debug(f"Node {node_id} rejected by availability cache: {reason}")
                return False, f"cached_{reason}"
        except ImportError:
            pass  # Cache not available, continue with full checks

        # Step 1: Basic preflight check (uses adaptive timeout when preflight_timeout is None)
        is_available, reason = await self._preflight_check_node(node_id, preflight_timeout)
        if not is_available:
            logger.warning(f"Node {node_id} failed preflight check: {reason}")
            # Update cache with failure
            self._update_availability_cache(node_id, available=False, reason=reason)
            return False, f"preflight_{reason}"

        # Step 2: GPU health check if required (uses adaptive timeout when gpu_check_timeout is None)
        if requires_gpu:
            is_healthy, gpu_reason = await self._check_gpu_health(node_id, gpu_check_timeout)
            if not is_healthy:
                logger.warning(f"Node {node_id} failed GPU health check: {gpu_reason}")
                # Update cache with GPU failure
                self._update_availability_cache(node_id, available=False, reason=gpu_reason)
                return False, f"gpu_{gpu_reason}"

        # Node is valid - update cache with success
        self._update_availability_cache(node_id, available=True)
        return True, "ok"

    def _update_availability_cache(
        self, node_id: str, available: bool, reason: str | None = None
    ) -> None:
        """Update the availability cache based on validation result.

        Args:
            node_id: The node ID
            available: Whether the node is available
            reason: Reason for unavailability (if not available)
        """
        try:
            from app.coordination.node_availability_cache import (
                AvailabilityReason,
                get_availability_cache,
            )

            cache = get_availability_cache()
            if available:
                cache.mark_available(node_id, source="job_validation")
            else:
                # Map reason string to AvailabilityReason
                reason_map = {
                    "ssh_timeout": AvailabilityReason.SSH_TIMEOUT,
                    "ssh_probe_failed": AvailabilityReason.SSH_FAILED,
                    "gpu_busy_or_unavailable": AvailabilityReason.GPU_ERROR,
                    "cuda_error": AvailabilityReason.GPU_ERROR,
                    "no_gpu_devices": AvailabilityReason.GPU_ERROR,
                }
                availability_reason = reason_map.get(
                    reason or "", AvailabilityReason.HEALTH_CHECK_FAILED
                )
                cache.mark_unavailable(
                    node_id,
                    availability_reason,
                    source="job_validation",
                    error_message=reason,
                )
        except ImportError:
            pass  # Cache not available

    # Event Subscriptions (December 2025)
    # =========================================================================

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """Define event subscriptions for JobManager.

        Returns:
            Dict mapping event names to handler methods.

        December 28, 2025: Uses EventSubscriptionMixin pattern for consolidated
        event handling. This replaces ~35 LOC of duplicate subscription code.
        """
        return {
            "HOST_OFFLINE": self._on_host_offline,
            "HOST_ONLINE": self._on_host_online,
            "NODE_RECOVERED": self._on_node_recovered,
        }

    def subscribe_to_events(self) -> None:
        """Subscribe to job-relevant events.

        Subscribes to:
        - HOST_OFFLINE: Cancel jobs on offline hosts
        - HOST_ONLINE: Potentially reschedule cancelled jobs
        - NODE_RECOVERED: Re-enable job dispatch to recovered nodes

        Dec 28, 2025: Refactored to use EventSubscriptionMixin for consolidated
        subscription logic with thread-safe double-check locking.
        """
        # Use mixin's consolidated subscription method (inherited from EventSubscriptionMixin)
        EventSubscriptionMixin.subscribe_to_events(self)

    async def _on_host_offline(self, event) -> None:
        """Handle HOST_OFFLINE events - cancel jobs on offline host.

        December 28, 2025: Uses EventSubscriptionMixin helpers for logging and event emission.
        """
        # Use mixin helper to extract payload safely
        payload = self._extract_event_payload(event)
        node_id = payload.get("node_id", "")

        if not node_id:
            return

        self._log_info(f"HOST_OFFLINE: {node_id}, cancelling jobs")

        # Cancel all jobs running on offline host
        cancelled = 0
        cancelled_jobs: list[tuple[str, str]] = []  # (job_type, job_id) tuples
        with self.jobs_lock:
            for job_type, jobs in self.active_jobs.items():
                for job_id, job in list(jobs.items()):
                    job_node = job.get("node_id") if isinstance(job, dict) else getattr(job, "node_id", None)
                    if job_node == node_id:
                        job_status = job.get("status") if isinstance(job, dict) else getattr(job, "status", "running")
                        if job_status == "running":
                            if isinstance(job, dict):
                                job["status"] = "cancelled"
                            else:
                                job.status = "cancelled"
                            cancelled += 1
                            cancelled_jobs.append((job_type, job_id))
                            self._log_info(f"Cancelled {job_type} job {job_id} on offline node {node_id}")

        # December 27, 2025: Emit TASK_ABANDONED events for cancelled jobs
        # This allows SelfplayOrchestrator and other subscribers to track abandoned work
        for job_type, job_id in cancelled_jobs:
            # Use mixin's safe event emission helper
            self._safe_emit_event(
                "TASK_ABANDONED",
                {
                    "job_id": job_id,
                    "job_type": job_type,
                    "reason": "host_offline",
                    "offline_node": node_id,
                    "node_id": self.node_id,
                }
            )
            self.stats.jobs_cancelled += 1

        if cancelled > 0:
            self.stats.hosts_offline += 1
            self._log_info(f"Cancelled {cancelled} jobs on offline node {node_id}")

    async def _on_host_online(self, event) -> None:
        """Handle HOST_ONLINE events - log for potential job rescheduling.

        December 28, 2025: Uses EventSubscriptionMixin helpers for payload extraction.
        """
        # Use mixin helper to extract payload safely
        payload = self._extract_event_payload(event)
        node_id = payload.get("node_id", "")

        if not node_id:
            return

        self._log_info(f"HOST_ONLINE: {node_id}, node available for jobs")

        # Track for observability
        self.stats.hosts_online += 1

        # Note: Job rescheduling is handled by the scheduler, not the job manager
        # This is logged for observability

    async def _on_node_recovered(self, event) -> None:
        """Handle NODE_RECOVERED events - re-enable job dispatch to recovered node.

        December 27, 2025: Added to complete health event integration.
        December 28, 2025: Uses EventSubscriptionMixin helpers for payload extraction.

        When a node recovers from an unhealthy state, we can consider it for new jobs again.
        Unlike HOST_ONLINE (which is for new connections), NODE_RECOVERED indicates
        a node that was degraded/unhealthy is now healthy again.
        """
        # Use mixin helper to extract payload safely
        payload = self._extract_event_payload(event)
        node_id = payload.get("node_id", "") or payload.get("host", "")
        recovery_reason = payload.get("reason", "unknown")

        if not node_id:
            return

        self._log_info(
            f"NODE_RECOVERED: {node_id} recovered (reason: {recovery_reason}), "
            f"available for new jobs"
        )

        # Track recovery for scheduling decisions
        self.stats.nodes_recovered += 1

        # Note: Actual job dispatch decisions are made by SelfplayScheduler
        # and TrainingCoordinator. This event is tracked for observability
        # and to inform scheduling heuristics.

    # =========================================================================
    # Subprocess Lifecycle Management (December 28, 2025)
    # =========================================================================

    def _register_process(self, job_id: str, proc: asyncio.subprocess.Process) -> None:
        """Register a subprocess handle for tracking.

        Args:
            job_id: Job identifier
            proc: asyncio subprocess handle
        """
        with self._processes_lock:
            self._active_processes[job_id] = proc

    def _unregister_process(self, job_id: str) -> None:
        """Unregister a subprocess handle after completion.

        Args:
            job_id: Job identifier
        """
        with self._processes_lock:
            self._active_processes.pop(job_id, None)

    async def _kill_process(self, job_id: str, proc: asyncio.subprocess.Process | None = None) -> None:
        """Kill a subprocess and wait for it to terminate.

        December 28, 2025: Centralized subprocess cleanup to prevent zombie processes.
        Tries SIGKILL first for immediate termination, falls back to SIGTERM.

        Args:
            job_id: Job identifier for logging
            proc: Optional process handle. If None, looks up from _active_processes.
        """
        if proc is None:
            with self._processes_lock:
                proc = self._active_processes.get(job_id)

        if proc is None:
            logger.debug(f"No process found for job {job_id}, nothing to kill")
            return

        try:
            # Check if process is still running
            if proc.returncode is not None:
                logger.debug(f"Process for job {job_id} already terminated (rc={proc.returncode})")
                self._unregister_process(job_id)
                return

            # Try SIGKILL first for immediate termination
            logger.info(f"Killing process for job {job_id} (pid={proc.pid})")
            proc.kill()
            try:
                await asyncio.wait_for(proc.wait(), timeout=OperationTimeouts.THREAD_JOIN)
                logger.debug(f"Process for job {job_id} killed successfully")
            except asyncio.TimeoutError:
                # SIGKILL didn't work, try SIGTERM as fallback
                logger.warning(f"Process for job {job_id} didn't respond to SIGKILL, trying SIGTERM")
                try:
                    proc.terminate()
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    logger.error(f"Process for job {job_id} (pid={proc.pid}) is stuck, may become zombie")
                except ProcessLookupError:
                    pass  # Process already dead
        except ProcessLookupError:
            logger.debug(f"Process for job {job_id} already dead (ProcessLookupError)")
        except OSError as e:
            logger.debug(f"OSError killing process for job {job_id}: {e}")
        finally:
            self._unregister_process(job_id)

    async def cleanup_active_processes(self) -> int:
        """Kill all active subprocesses during shutdown.

        December 28, 2025: Added for graceful shutdown to prevent zombie processes.

        Returns:
            Number of processes killed
        """
        with self._processes_lock:
            job_ids = list(self._active_processes.keys())

        if not job_ids:
            return 0

        logger.info(f"Cleaning up {len(job_ids)} active processes during shutdown")
        killed = 0

        for job_id in job_ids:
            try:
                await self._kill_process(job_id)
                killed += 1
            except (OSError, ProcessLookupError, asyncio.CancelledError) as e:
                # Dec 2025: Narrowed from broad Exception - these are expected process cleanup errors
                logger.warning(f"Error killing process for job {job_id}: {e}")

        return killed

    # =========================================================================
    # Job Heartbeat Tracking (Phase 15.1.9 - December 29, 2025)
    # =========================================================================

    def update_job_heartbeat(self, job_id: str) -> None:
        """Update the heartbeat timestamp for a job.

        Should be called periodically by running jobs to signal they're still alive.

        Args:
            job_id: The job identifier
        """
        with self._heartbeats_lock:
            current = self._job_heartbeats.get(job_id)
            reassignment_count = current[1] if current else 0
            next_reassign_time = current[2] if current else 0.0
            self._job_heartbeats[job_id] = (time.time(), reassignment_count, next_reassign_time)

    def get_job_heartbeats(self) -> dict[str, float]:
        """Get all job heartbeat timestamps.

        Returns:
            Dict mapping job_id -> last_heartbeat_timestamp
        """
        with self._heartbeats_lock:
            return {job_id: hb[0] for job_id, hb in self._job_heartbeats.items()}

    def _register_job_heartbeat(self, job_id: str, reassignment_count: int = 0) -> None:
        """Register a new job for heartbeat tracking.

        Called when a job is spawned to start heartbeat tracking.

        Args:
            job_id: The job identifier
            reassignment_count: Number of times this job has been reassigned
        """
        with self._heartbeats_lock:
            # next_reassignment_time = 0.0 means eligible for immediate reassignment if needed
            self._job_heartbeats[job_id] = (time.time(), reassignment_count, 0.0)

    def _unregister_job_heartbeat(self, job_id: str) -> None:
        """Remove a job from heartbeat tracking.

        Called when a job completes or is cancelled.

        Args:
            job_id: The job identifier
        """
        with self._heartbeats_lock:
            self._job_heartbeats.pop(job_id, None)

    async def check_stale_jobs(self) -> list[tuple[str, str]]:
        """Check for jobs that have exceeded heartbeat timeout.

        Phase 15.1.9: Detects jobs that haven't sent a heartbeat within
        JOB_HEARTBEAT_TIMEOUT and marks them for reassignment.

        Returns:
            List of (job_id, job_type) tuples for stale jobs
        """
        stale_jobs: list[tuple[str, str]] = []
        now = time.time()

        with self._heartbeats_lock:
            stale_job_ids = [
                job_id for job_id, (last_hb, _, _) in self._job_heartbeats.items()
                if now - last_hb > self.JOB_HEARTBEAT_TIMEOUT
            ]

        if not stale_job_ids:
            return stale_jobs

        # Find job info for stale jobs
        with self.jobs_lock:
            for job_type, jobs in self.active_jobs.items():
                for job_id, job in jobs.items():
                    if job_id in stale_job_ids:
                        status = job.get("status") if isinstance(job, dict) else getattr(job, "status", "")
                        if status == "running":
                            stale_jobs.append((job_id, job_type))

        return stale_jobs

    async def reassign_stale_job(
        self,
        job_id: str,
        job_type: str,
        preferred_nodes: list[str] | None = None,
    ) -> bool:
        """Attempt to reassign a stale job to another node.

        Phase 15.1.9: When a job exceeds heartbeat timeout, this method
        attempts to reassign it to a different node.

        Args:
            job_id: The stale job's identifier
            job_type: Type of job (selfplay, training, etc.)
            preferred_nodes: Optional list of preferred node IDs for reassignment

        Returns:
            True if job was successfully queued for reassignment, False otherwise
        """
        now = time.time()

        # Get reassignment count and backoff info
        with self._heartbeats_lock:
            current = self._job_heartbeats.get(job_id)
            reassignment_count = current[1] if current else 0
            next_reassignment_time = current[2] if current else 0.0

        # January 2026: Check exponential backoff before allowing reassignment
        # This prevents rapid retry cycles when jobs consistently fail
        if now < next_reassignment_time:
            backoff_remaining = next_reassignment_time - now
            logger.debug(
                f"Job {job_id} in backoff period, {backoff_remaining:.1f}s remaining "
                f"(attempt {reassignment_count})"
            )
            return False

        # Check if we've exceeded max reassignment attempts
        if reassignment_count >= self.MAX_REASSIGNMENT_ATTEMPTS:
            logger.warning(
                f"Job {job_id} exceeded max reassignment attempts ({self.MAX_REASSIGNMENT_ATTEMPTS}), "
                f"marking as permanently failed"
            )
            self._handle_permanent_job_failure(job_id, job_type)
            return False

        # Get original job info
        job_info: dict[str, Any] | None = None
        original_node: str | None = None
        with self.jobs_lock:
            if job_type in self.active_jobs and job_id in self.active_jobs[job_type]:
                job = self.active_jobs[job_type][job_id]
                if isinstance(job, dict):
                    job_info = job.copy()
                    original_node = job.get("node_id")
                    # Mark original as cancelled
                    job["status"] = "reassigned"
                else:
                    job_info = {
                        "job_id": job_id,
                        "board_type": getattr(job, "board_type", ""),
                        "num_players": getattr(job, "num_players", 2),
                        "num_games": getattr(job, "num_games", 100),
                    }
                    original_node = getattr(job, "node_id", None)
                    job.status = "reassigned"

        if not job_info:
            logger.warning(f"Cannot reassign job {job_id}: job info not found")
            return False

        # Emit TASK_ORPHANED event for the original job
        self._emit_task_event(
            "TASK_ORPHANED",
            job_id,
            job_type,
            original_node=original_node,
            reassignment_count=reassignment_count + 1,
            board_type=job_info.get("board_type", ""),
        )

        # Increment reassignment count and calculate exponential backoff
        new_reassignment_count = reassignment_count + 1
        backoff_delay = min(
            JobReaperDefaults.REASSIGN_BACKOFF_BASE * (
                JobReaperDefaults.REASSIGN_BACKOFF_MULTIPLIER ** reassignment_count
            ),
            JobReaperDefaults.REASSIGN_BACKOFF_MAX,
        )
        next_reassign_time = now + backoff_delay

        with self._heartbeats_lock:
            self._job_heartbeats[job_id] = (now, new_reassignment_count, next_reassign_time)

        self.stats.jobs_orphaned += 1
        self.stats.jobs_reassigned += 1

        # Log the reassignment with backoff info for monitoring
        logger.info(
            f"Reassigning orphaned job {job_id} (type={job_type}, "
            f"original_node={original_node}, attempt={new_reassignment_count}, "
            f"backoff_delay={backoff_delay:.0f}s)"
        )

        # Note: The actual re-dispatch is handled by SelfplayScheduler which
        # subscribes to TASK_ORPHANED events. We just emit the event here.
        return True

    def _handle_permanent_job_failure(self, job_id: str, job_type: str) -> None:
        """Handle a job that has permanently failed after max reassignment attempts.

        Args:
            job_id: The failed job's identifier
            job_type: Type of job
        """
        # Remove from active jobs
        with self.jobs_lock:
            if job_type in self.active_jobs and job_id in self.active_jobs[job_type]:
                job = self.active_jobs[job_type][job_id]
                if isinstance(job, dict):
                    job["status"] = "permanently_failed"
                else:
                    job.status = "permanently_failed"

        # Remove from heartbeat tracking
        self._unregister_job_heartbeat(job_id)

        # Emit permanent failure event
        self._emit_task_event(
            "TASK_FAILED",
            job_id,
            job_type,
            error="exceeded_max_reassignment_attempts",
            permanent=True,
        )

        self.stats.reassignment_failures += 1
        logger.error(
            f"Job {job_id} permanently failed after {self.MAX_REASSIGNMENT_ATTEMPTS} "
            f"reassignment attempts"
        )

    async def process_stale_jobs(self) -> int:
        """Check for and reassign all stale jobs.

        Phase 15.1.9: Convenience method that combines stale job detection
        and reassignment. Should be called periodically (e.g., every 60s).

        Returns:
            Number of jobs reassigned
        """
        stale_jobs = await self.check_stale_jobs()
        if not stale_jobs:
            return 0

        reassigned = 0
        for job_id, job_type in stale_jobs:
            try:
                if await self.reassign_stale_job(job_id, job_type):
                    reassigned += 1
            except Exception as e:
                logger.warning(f"Error reassigning job {job_id}: {e}")

        if reassigned > 0:
            logger.info(f"Reassigned {reassigned}/{len(stale_jobs)} stale jobs")

        return reassigned

    def _emit_task_event(self, event_type: str, job_id: str, job_type: str, **kwargs) -> None:
        """Emit a task lifecycle event if the event system is available.

        Args:
            event_type: One of TASK_SPAWNED, TASK_COMPLETED, TASK_FAILED
            job_id: Unique job identifier
            job_type: Type of job (selfplay, training, etc.)
            **kwargs: Additional event data
        """
        emitter = _get_event_emitter()
        if emitter is None:
            return

        payload = {
            "job_id": job_id,
            "job_type": job_type,
            "node_id": self.node_id,
            "timestamp": time.time(),
            **kwargs,
        }

        try:
            emitter(event_type, payload)
            logger.debug(f"Emitted {event_type} for job {job_id}")
        except (OSError, ConnectionError, RuntimeError, TypeError) as e:
            # Dec 2025: Narrowed from broad Exception - event emission errors
            # OSError/ConnectionError: Network issues
            # RuntimeError: Event bus state errors
            # TypeError: Payload serialization errors
            logger.debug(f"Failed to emit {event_type}: {e}")

    # =========================================================================
    # Selfplay Job Methods
    # =========================================================================

    async def run_gpu_selfplay_job(
        self,
        job_id: str,
        board_type: str,
        num_players: int,
        num_games: int,
        engine_mode: str,
        engine_extra_args: dict[str, Any] | None = None,
        model_version: str = "v5",
        cuda_device: int | None = None,
    ) -> None:
        """Run selfplay job with appropriate script based on engine mode.

        For simple modes (random, heuristic, nnue-guided): use run_gpu_selfplay.py (GPU-optimized)
        For search modes (maxn, brs, mcts, gumbel-mcts): use generate_gumbel_selfplay.py (supports search)

        Args:
            job_id: Unique job identifier
            board_type: Board type (hex8, square8, etc.)
            num_players: Number of players (2-4)
            num_games: Number of games to play
            engine_mode: Engine mode (heuristic-only, gumbel-mcts, etc.)
            engine_extra_args: Extra engine arguments (e.g., {"budget": 64} for gumbel-mcts)
            cuda_device: Optional GPU device index for multi-GPU nodes (sets CUDA_VISIBLE_DEVICES)
        """
        board_norm = board_type.replace("hexagonal", "hex")
        # Session 17.35: Use _get_data_path() to avoid doubled ai-service/ path
        output_dir = Path(self._get_data_path("selfplay", "p2p_gpu", f"{board_norm}_{num_players}p", job_id))
        output_dir.mkdir(parents=True, exist_ok=True)

        # December 2025: Check disk space before spawning job to avoid silent failures
        has_space, space_reason = self._check_disk_space(output_dir)
        if not has_space:
            logger.warning(
                f"Skipping selfplay job {job_id} due to insufficient disk space: {space_reason}"
            )
            self._emit_task_event(
                "TASK_FAILED", job_id, "selfplay",
                error=f"disk_space_check_failed: {space_reason}",
                board_type=board_type, num_players=num_players,
            )
            return

        # Jan 2026: Default to gumbel-mcts for high-quality training data
        effective_mode = engine_mode or "gumbel-mcts"

        # December 2025: GPU availability check using GPU_REQUIRED_ENGINE_MODES
        # This prevents wasting compute on GPU-required modes when no GPU is available
        if self._engine_mode_requires_gpu(effective_mode):
            try:
                import torch
                has_cuda = torch.cuda.is_available()
                has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

                if not (has_cuda or has_mps):
                    # Jan 2026 (Session 17.50): Check YAML config as authoritative fallback
                    # Runtime GPU detection can fail on some nodes (vGPU, containers)
                    yaml_has_gpu = self._check_yaml_gpu_config()

                    if yaml_has_gpu:
                        logger.warning(
                            f"[GPU Detection] Runtime failed (CUDA={has_cuda}, MPS={has_mps}) "
                            f"but YAML shows GPU for {self.node_id}. Continuing with GPU mode for job {job_id}"
                        )
                        # Continue with GPU mode - don't fall back to CPU
                    else:
                        # No GPU via runtime OR YAML: use CPU mode
                        try:
                            from scripts.p2p.managers.selfplay_scheduler import SelfplayScheduler
                            effective_mode, _ = SelfplayScheduler._select_board_engine(
                                has_gpu=False,  # CPU-only selection
                                board_type=board_type,
                                num_players=num_players,
                            )
                            logger.warning(
                                f"GPU-required mode requested but no GPU available "
                                f"(CUDA={has_cuda}, MPS={has_mps}), using CPU mode '{effective_mode}' for job {job_id}"
                            )
                        except (ImportError, AttributeError, TypeError):
                            logger.warning(
                                f"GPU-required mode requested but no GPU available, falling back to maxn for job {job_id}"
                            )
                            effective_mode = "maxn"  # High-quality CPU mode
                else:
                    device_type = "CUDA" if has_cuda else "MPS"
                    logger.debug(f"GPU available ({device_type}) for mode '{effective_mode}' job {job_id}")
            except ImportError:
                logger.warning(
                    f"PyTorch not available for GPU check, falling back to maxn for job {job_id}"
                )
                effective_mode = "maxn"  # High-quality CPU mode

        # January 2026: Mixed opponent mode uses scripts/selfplay.py with --engine mixed
        # This provides actual diverse opponents (random, heuristic, mcts, minimax, etc.)
        # to break weak-vs-weak training cycles for starved configs
        if effective_mode in ("mixed", "mixed-opponents"):
            script_path = self._get_script_path("selfplay.py")
            if not os.path.exists(script_path):
                logger.warning(f"Selfplay script not found: {script_path}")
                return

            cmd = [
                sys.executable,
                script_path,
                "--board", board_norm,
                "--num-players", str(num_players),
                "--num-games", str(num_games),
                "--engine", "mixed",  # Use MixedOpponentSelfplayRunner
                "--output", str(output_dir / "games.db"),
                "--seed", str(int(time.time() * 1000) % 2**31),
            ]
            logger.info(
                f"Using mixed opponent selfplay for {board_type}_{num_players}p "
                f"(job {job_id}) - diverse opponents for better training signal"
            )

        elif effective_mode == "multigame-gumbel":
            # February 2026: Batched multi-game Gumbel MCTS for 10-20x throughput
            script_path = self._get_script_path("run_multigame_gumbel_selfplay.py")
            if not os.path.exists(script_path):
                logger.warning(f"Multigame Gumbel script not found: {script_path}")
                return

            simulation_budget = 800
            if engine_extra_args and "budget" in engine_extra_args:
                simulation_budget = engine_extra_args["budget"]

            batch_size = engine_extra_args.get("batch_size", 64) if engine_extra_args else 64
            model_elo = engine_extra_args.get("model_elo", 0) if engine_extra_args else 0

            cmd = [
                sys.executable,
                script_path,
                "--board", board_norm,
                "--num-players", str(num_players),
                "--num-games", str(num_games),
                "--batch-size", str(batch_size),
                "--simulation-budget", str(simulation_budget),
                "--db", str(output_dir / "games.db"),
                "--model-version", model_version,
                "--seed", str(int(time.time() * 1000) % 2**31),
            ]
            if model_elo > 0:
                cmd.extend(["--model-elo", str(model_elo)])
            logger.info(
                f"Using multigame Gumbel selfplay for {board_type}_{num_players}p "
                f"(job {job_id}) - batch={batch_size}, budget={simulation_budget}"
                f"{f', model_elo={model_elo}' if model_elo > 0 else ''}"
            )

        elif effective_mode in self.SEARCH_ENGINE_MODES:
            # December 29, 2025: Use generate_gumbel_selfplay.py for search-based modes
            # (run_hybrid_selfplay.py was archived, generate_gumbel_selfplay.py is the active replacement)
            script_path = self._get_script_path("generate_gumbel_selfplay.py")
            if not os.path.exists(script_path):
                logger.warning(f"Gumbel selfplay script not found: {script_path}")
                return

            # December 2025: Get budget from engine_extra_args if provided (for large board mix)
            # Jan 2026: Increased to 800 - AlphaZero minimum for quality training data
            simulation_budget = 800  # Quality tier - matches AlphaZero minimum
            if engine_extra_args and "budget" in engine_extra_args:
                simulation_budget = engine_extra_args["budget"]
                logger.debug(f"Using custom budget {simulation_budget} from engine_extra_args for job {job_id}")

            cmd = [
                sys.executable,
                script_path,
                "--board", board_norm,  # generate_gumbel_selfplay uses --board not --board-type
                "--num-players", str(num_players),
                "--num-games", str(num_games),
                "--db", str(output_dir / "games.db"),  # uses --db not --record-db
                "--seed", str(int(time.time() * 1000) % 2**31),
                "--simulation-budget", str(simulation_budget),
                "--model-version", model_version,  # Jan 5, 2026: Architecture selection feedback loop
            ]
            # Jan 1, 2026: Use per-node config to decide GPU tree mode
            # GPU tree gives 170x speedup but hangs on some nodes (vGPU, large boards)
            # Check distributed_hosts.yaml for disable_gpu_tree setting
            if self._should_use_gpu_tree():
                cmd.append("--use-gpu-tree")
                logger.debug(f"GPU tree enabled for node {self.node_id} job {job_id}")
            else:
                cmd.append("--no-gpu-tree")
                logger.debug(f"GPU tree disabled for node {self.node_id} job {job_id}")
        else:
            # Use run_gpu_selfplay.py for GPU-optimized modes
            script_path = self._get_script_path("run_gpu_selfplay.py")
            if not os.path.exists(script_path):
                logger.warning(f"GPU selfplay script not found: {script_path}")
                return

            # Map engine modes: run_gpu_selfplay.py supports: random-only, heuristic-only, nnue-guided
            # For GPU-accelerated modes (gumbel-mcts, mcts, etc.), we use heuristic-only base mode
            # but enable --use-policy to activate neural network guidance
            mode_map = {
                "mixed": "heuristic-only",
                "gpu": "heuristic-only",
                "descent-only": "heuristic-only",
                "heuristic-only": "heuristic-only",
                "nnue-guided": "nnue-guided",
                "random-only": "random-only",
            }
            gpu_engine_mode = mode_map.get(effective_mode, "heuristic-only")

            # Jan 2026: Modes that require neural network policy guidance
            # These modes need --use-policy --use-heuristic to enable GPU-accelerated NN inference
            POLICY_GUIDED_MODES = {
                "gumbel-mcts", "gumbel", "gpu-gumbel", "mcts", "policy-only", "nn-descent",
                "nn-minimax", "gnn", "hybrid", "gmo", "ebmo", "ig-gmo", "cage",
            }
            use_policy = effective_mode in POLICY_GUIDED_MODES

            # Dec 31, 2025: Calculate optimal batch size for GPU utilization
            # Higher batch sizes improve GPU utilization significantly
            optimal_batch = get_optimal_batch_size(board_type=board_type, num_players=num_players)
            logger.debug(f"Computed optimal batch size {optimal_batch} for {board_type}_{num_players}p")

            cmd = [
                sys.executable,
                script_path,
                "--board", board_norm,  # Note: --board not --board-type
                "--num-players", str(num_players),
                "--num-games", str(num_games),
                "--output-dir", str(output_dir),
                "--output-db", str(output_dir / "games.db"),
                "--engine-mode", gpu_engine_mode,
                "--seed", str(int(time.time() * 1000) % 2**31),
                "--batch-size", str(optimal_batch),  # Dec 31, 2025: Pass batch size for GPU utilization
            ]

            # Jan 2026: Enable policy network for GPU-accelerated modes
            if use_policy:
                cmd.extend(["--use-policy", "--use-heuristic"])
                logger.info(f"GPU selfplay job {job_id}: enabling policy network guidance for mode '{effective_mode}'")

        # December 29, 2025: Use helper for consistent env setup (includes RINGRIFT_ALLOW_PENDING_GATE)
        env = self._get_subprocess_env()

        # Jan 18, 2026: Set CUDA_VISIBLE_DEVICES for multi-GPU nodes
        # This routes the subprocess to a specific GPU when running parallel workers
        if cuda_device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
            logger.info(f"[Multi-GPU] Job {job_id} routed to GPU {cuda_device}")

        # December 28, 2025: Initialize proc before try block for proper cleanup in except handlers
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            # December 28, 2025: Register process for tracking (enables graceful shutdown)
            self._register_process(job_id, proc)

            # Track the job
            with self.jobs_lock:
                if "selfplay" not in self.active_jobs:
                    self.active_jobs["selfplay"] = {}
                self.active_jobs["selfplay"][job_id] = {
                    "job_id": job_id,
                    "status": "running",
                    "board_type": board_type,
                    "num_players": num_players,
                    "num_games": num_games,
                    "started_at": time.time(),
                    "pid": proc.pid,
                    "node_id": self.node_id,  # Phase 15.1.9: Track which node is running the job
                }

            # Dec 30, 2025 (P5.3): Record job assignment in Raft for cluster-wide visibility
            _record_raft_job_assignment(
                job_id=job_id,
                node_id=self.node_id,
                job_type="selfplay",
                board_type=board_type,
                num_players=num_players,
                num_games=num_games,
                engine_mode=effective_mode,
            )
            _start_raft_job(job_id)

            # Phase 15.1.9: Register for heartbeat tracking
            self._register_job_heartbeat(job_id)

            # Emit TASK_SPAWNED event for pipeline coordination
            self._emit_task_event(
                "TASK_SPAWNED",
                job_id,
                "selfplay",
                board_type=board_type,
                num_players=num_players,
                num_games=num_games,
                engine_mode=effective_mode,
            )

            # Dec 28, 2025: Use centralized JobDefaults.SELFPLAY_TIMEOUT (7200s = 2 hours)
            _stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=JobDefaults.SELFPLAY_TIMEOUT)

            # December 28, 2025: Unregister process after successful completion
            self._unregister_process(job_id)

            # Phase 15.1.9: Unregister from heartbeat tracking
            self._unregister_job_heartbeat(job_id)

            # Update job status and emit completion event
            duration = time.time() - self.active_jobs.get("selfplay", {}).get(job_id, {}).get("started_at", time.time())
            with self.jobs_lock:
                if job_id in self.active_jobs.get("selfplay", {}):
                    if proc.returncode == 0:
                        self.active_jobs["selfplay"][job_id]["status"] = "completed"
                        # Dec 30, 2025 (P5.3): Mark job as completed in Raft
                        _complete_raft_job(job_id, result={
                            "duration_seconds": duration,
                            "num_games": num_games,
                        })
                        self._emit_task_event(
                            "TASK_COMPLETED",
                            job_id,
                            "selfplay",
                            board_type=board_type,
                            num_players=num_players,
                            num_games=num_games,
                            duration_seconds=duration,
                            engine_mode=effective_mode,  # Dec 29 2025: For bandit feedback
                        )
                    else:
                        error_msg = stderr.decode()[:500]
                        logger.warning(f"Selfplay job {job_id} failed: {error_msg}")
                        self.active_jobs["selfplay"][job_id]["status"] = "failed"
                        # Dec 30, 2025 (P5.3): Mark job as failed in Raft
                        _fail_raft_job(job_id, error=error_msg)
                        self._emit_task_event(
                            "TASK_FAILED",
                            job_id,
                            "selfplay",
                            board_type=board_type,
                            num_players=num_players,
                            error=error_msg,
                            duration_seconds=duration,
                        )

                    # Session 17.24: Track job completion for load balancing
                    job_node_id = self.active_jobs["selfplay"][job_id].get("node_id")
                    if HAS_QUEUE_TRACKER and job_node_id:
                        try:
                            tracker = get_node_queue_tracker()
                            if tracker:
                                tracker.on_job_completed(job_node_id)
                        except (AttributeError, TypeError, KeyError):
                            pass  # Non-critical

                    # Session 17.38: Track config completion for multi-config support
                    if HAS_CONFIG_TRACKER and job_node_id:
                        try:
                            config_tracker = get_node_config_tracker()
                            job_config_key = self.active_jobs["selfplay"][job_id].get(
                                "config_key"
                            )
                            if config_tracker and job_config_key:
                                config_tracker.remove_config(job_node_id, job_config_key)
                        except (AttributeError, TypeError, KeyError):
                            pass  # Non-critical

                    # Remove from active jobs
                    del self.active_jobs["selfplay"][job_id]

        except asyncio.TimeoutError:
            # December 28, 2025: Use centralized process cleanup
            await self._kill_process(job_id, proc)
            # Phase 15.1.9: Unregister from heartbeat tracking on timeout
            self._unregister_job_heartbeat(job_id)
            with self.jobs_lock:
                if job_id in self.active_jobs.get("selfplay", {}):
                    # Session 17.24: Track job completion for load balancing
                    job_node_id = self.active_jobs["selfplay"][job_id].get("node_id")
                    if HAS_QUEUE_TRACKER and job_node_id:
                        try:
                            tracker = get_node_queue_tracker()
                            if tracker:
                                tracker.on_job_completed(job_node_id)
                        except Exception:
                            pass  # Non-critical
                    # Session 17.38: Track config completion for multi-config support
                    if HAS_CONFIG_TRACKER and job_node_id:
                        try:
                            config_tracker = get_node_config_tracker()
                            job_config_key = self.active_jobs["selfplay"][job_id].get(
                                "config_key"
                            )
                            if config_tracker and job_config_key:
                                config_tracker.remove_config(job_node_id, job_config_key)
                        except Exception:
                            pass  # Non-critical
                    self.active_jobs["selfplay"][job_id]["status"] = "timeout"
                    del self.active_jobs["selfplay"][job_id]
            # Dec 30, 2025 (P5.3): Mark job as failed in Raft
            _fail_raft_job(job_id, error="timeout")
            logger.warning(f"Selfplay job {job_id} timed out and was killed")
            self._emit_task_event("TASK_FAILED", job_id, "selfplay", error="timeout", board_type=board_type)
        except (OSError, ValueError, RuntimeError) as e:
            # Dec 2025: Narrowed from broad Exception - subprocess execution errors
            # OSError: File not found, permission denied, process creation failed
            # ValueError: Invalid arguments to subprocess
            # RuntimeError: Subprocess state errors
            await self._kill_process(job_id, proc)
            # Phase 15.1.9: Unregister from heartbeat tracking on error
            self._unregister_job_heartbeat(job_id)
            with self.jobs_lock:
                if job_id in self.active_jobs.get("selfplay", {}):
                    # Session 17.24: Track job completion for load balancing
                    job_node_id = self.active_jobs["selfplay"][job_id].get("node_id")
                    if HAS_QUEUE_TRACKER and job_node_id:
                        try:
                            tracker = get_node_queue_tracker()
                            if tracker:
                                tracker.on_job_completed(job_node_id)
                        except Exception:
                            pass  # Non-critical
                    # Session 17.38: Track config completion for multi-config support
                    if HAS_CONFIG_TRACKER and job_node_id:
                        try:
                            config_tracker = get_node_config_tracker()
                            job_config_key = self.active_jobs["selfplay"][job_id].get(
                                "config_key"
                            )
                            if config_tracker and job_config_key:
                                config_tracker.remove_config(job_node_id, job_config_key)
                        except Exception:
                            pass  # Non-critical
                    self.active_jobs["selfplay"][job_id]["status"] = "error"
                    del self.active_jobs["selfplay"][job_id]
            # Dec 30, 2025 (P5.3): Mark job as failed in Raft
            _fail_raft_job(job_id, error=str(e))
            logger.error(f"Error running selfplay job {job_id}: {e}")
            self._emit_task_event("TASK_FAILED", job_id, "selfplay", error=str(e), board_type=board_type)
        except (asyncio.CancelledError, ChildProcessError, BrokenPipeError) as e:
            # Dec 2025: Narrowed catch-all to subprocess cancellation/pipe errors
            # asyncio.CancelledError: Task was cancelled
            # ChildProcessError: Child process state error
            # BrokenPipeError: Process pipe communication failed
            await self._kill_process(job_id, proc)
            self._unregister_job_heartbeat(job_id)
            with self.jobs_lock:
                if job_id in self.active_jobs.get("selfplay", {}):
                    # Session 17.24: Track job completion for load balancing
                    job_node_id = self.active_jobs["selfplay"][job_id].get("node_id")
                    if HAS_QUEUE_TRACKER and job_node_id:
                        try:
                            tracker = get_node_queue_tracker()
                            if tracker:
                                tracker.on_job_completed(job_node_id)
                        except Exception:
                            pass  # Non-critical
                    # Session 17.38: Track config completion for multi-config support
                    if HAS_CONFIG_TRACKER and job_node_id:
                        try:
                            config_tracker = get_node_config_tracker()
                            job_config_key = self.active_jobs["selfplay"][job_id].get(
                                "config_key"
                            )
                            if config_tracker and job_config_key:
                                config_tracker.remove_config(job_node_id, job_config_key)
                        except Exception:
                            pass  # Non-critical
                    self.active_jobs["selfplay"][job_id]["status"] = "error"
                    del self.active_jobs["selfplay"][job_id]
            # Dec 30, 2025 (P5.3): Mark job as failed in Raft
            _fail_raft_job(job_id, error=f"{type(e).__name__}: {e}")
            logger.warning(f"Selfplay job {job_id} interrupted: {type(e).__name__}: {e}")
            self._emit_task_event("TASK_FAILED", job_id, "selfplay", error=str(e), board_type=board_type)
            raise

    async def run_distributed_selfplay(self, job_id: str) -> None:
        """Coordinate distributed selfplay for improvement loop.

        Distributes selfplay games across all available workers.
        Each worker runs selfplay using the current best model and reports
        progress back to the coordinator.

        Args:
            job_id: Job ID for the improvement loop
        """
        state = self.improvement_loop_state.get(job_id)
        if not state:
            return

        # Distribute selfplay across workers
        num_workers = max(len(state.worker_nodes), 1)
        games_per_worker = state.games_per_iteration // num_workers
        remainder = state.games_per_iteration % num_workers

        logger.info(f"Starting distributed selfplay: {games_per_worker} games/worker, {num_workers} workers")

        # Create output directory for this iteration
        iteration_dir = self._get_data_path(
            "selfplay", f"improve_{job_id}", f"iter_{state.current_iteration}"
        )
        os.makedirs(iteration_dir, exist_ok=True)

        # Dispatch selfplay tasks to workers via HTTP
        if state.worker_nodes:
            dispatch_results = await self._dispatch_selfplay_to_workers(
                job_id=job_id,
                workers=state.worker_nodes,
                games_per_worker=games_per_worker,
                remainder=remainder,
                board_type=state.board_type,
                num_players=state.num_players,
                model_path=state.best_model_path,
                output_dir=iteration_dir,
            )
            logger.info(
                f"Dispatched selfplay to {len(dispatch_results)} workers: "
                f"{sum(1 for r in dispatch_results.values() if r.get('success'))} succeeded"
            )
        else:
            # If no workers available, run locally
            logger.info("No workers available, running selfplay locally")
            await self.run_local_selfplay(
                job_id, state.games_per_iteration,
                state.board_type, state.num_players,
                state.best_model_path, iteration_dir
            )

    # =========================================================================
    # Phase 15.1.7: Dispatch Retry with Exponential Backoff
    # =========================================================================

    async def _dispatch_single_worker_with_retry(
        self,
        session: Any,  # aiohttp.ClientSession
        worker: Any,
        job_id: str,
        payload: dict[str, Any],
        timeout: Any,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Dispatch selfplay to a single worker with retry logic.

        Phase 15.1.7 (Dec 2025): Implements exponential backoff retry before
        returning failure. Previously, a single timeout or connection error
        would immediately mark the worker as failed.

        Args:
            session: aiohttp client session
            worker: Worker node info object
            job_id: Parent job ID
            payload: HTTP payload for /run-selfplay
            timeout: Request timeout
            max_retries: Maximum retry attempts (default: 3)

        Returns:
            Dict with success status, games dispatched, or error info
        """
        worker_id = getattr(worker, "node_id", str(worker))
        worker_ip = getattr(worker, "best_ip", None) or getattr(worker, "ip", None)
        worker_port = getattr(worker, "port", None) or _get_default_port()

        if not worker_ip:
            return {"success": False, "error": "no_ip", "worker_id": worker_id}

        # Sprint 15.1.1 (Jan 3, 2026): Check per-transport circuit breaker before attempting
        # This enables failover and prevents hammering unreachable workers
        # Sprint 17.9 (Jan 2026): Try SSH fallback when HTTP circuit is open
        http_circuit_open = not check_peer_transport_circuit(worker_ip, "http")

        if http_circuit_open:
            logger.debug(
                f"HTTP circuit breaker is OPEN for {worker_id}, trying SSH fallback"
            )

            # Try SSH fallback if available and SSH circuit is not open
            if HAS_SSH_FALLBACK and check_peer_transport_circuit(worker_ip, "ssh"):
                ssh_result = await self._dispatch_via_ssh_fallback(
                    worker_id=worker_id,
                    worker_ip=worker_ip,
                    payload=payload,
                )
                if ssh_result.get("success"):
                    return ssh_result

                # SSH also failed - log and return circuit_open
                logger.debug(
                    f"SSH fallback also failed for {worker_id}: {ssh_result.get('error')}"
                )

            return {
                "success": False,
                "error": "circuit_open",
                "worker_id": worker_id,
                "attempts": 0,
                "http_circuit_open": True,
                "ssh_attempted": HAS_SSH_FALLBACK,
            }

        url = f"http://{worker_ip}:{worker_port}/run-selfplay"

        for attempt in range(max_retries):
            try:
                async with session.post(url, json=payload, timeout=timeout) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        # Sprint 15.1.1: Record success to per-transport circuit breaker
                        record_peer_transport_success(worker_ip, "http")
                        return {
                            "success": result.get("success", True),
                            "games": payload.get("num_games", 0),
                            "worker_id": worker_id,
                            "attempts": attempt + 1,
                        }
                    elif resp.status in (502, 503, 504):
                        # Retryable server errors - gateway/service unavailable
                        wait_time = min(30, 2 ** attempt)
                        logger.warning(
                            f"Worker {worker_id} returned {resp.status}, "
                            f"retry {attempt + 1}/{max_retries} in {wait_time}s"
                        )
                        if attempt < max_retries - 1:
                            self.stats.dispatch_retries += 1
                            await asyncio.sleep(wait_time)
                        continue
                    else:
                        # Non-retryable HTTP error - record failure
                        record_peer_transport_failure(worker_ip, "http")
                        return {
                            "success": False,
                            "error": f"http_{resp.status}",
                            "worker_id": worker_id,
                            "attempts": attempt + 1,
                        }
            except asyncio.TimeoutError:
                wait_time = min(30, 2 ** attempt)
                logger.warning(
                    f"Dispatch to {worker_id} timed out, "
                    f"retry {attempt + 1}/{max_retries} in {wait_time}s"
                )
                if attempt < max_retries - 1:
                    self.stats.dispatch_retries += 1
                    await asyncio.sleep(wait_time)
            except (OSError, ConnectionError) as e:
                wait_time = min(30, 2 ** attempt)
                logger.warning(
                    f"Connection error to {worker_id}: {e}, "
                    f"retry {attempt + 1}/{max_retries} in {wait_time}s"
                )
                if attempt < max_retries - 1:
                    self.stats.dispatch_retries += 1
                    await asyncio.sleep(wait_time)

        # All retries exhausted - record failure to trip circuit breaker
        record_peer_transport_failure(worker_ip, "http")
        return {
            "success": False,
            "error": "max_retries_exceeded",
            "worker_id": worker_id,
            "attempts": max_retries,
        }

    async def _dispatch_via_ssh_fallback(
        self,
        worker_id: str,
        worker_ip: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Dispatch selfplay job via SSH when HTTP circuit is open.

        Sprint 17.9 (Jan 2026): SSH fallback for transport failover.
        When HTTP circuit breaker is open for a node, try SSH as fallback
        before giving up entirely.

        Args:
            worker_id: Worker node identifier
            worker_ip: Worker IP address
            payload: Selfplay job payload (board_type, num_players, num_games, etc.)

        Returns:
            Dict with success status, games dispatched, worker_id, and error info
        """
        if not HAS_SSH_FALLBACK:
            return {
                "success": False,
                "error": "ssh_module_unavailable",
                "worker_id": worker_id,
            }

        try:
            # Build SSH command to run selfplay on remote node
            board_type = payload.get("board_type", "hex8")
            num_players = payload.get("num_players", 2)
            num_games = payload.get("num_games", 10)
            model_path = payload.get("model_path", "")

            # Construct remote selfplay command
            cmd_parts = [
                "cd ~/ringrift/ai-service &&",
                "PYTHONPATH=.",
                "python scripts/selfplay.py",
                f"--board {board_type}",
                f"--num-players {num_players}",
                f"--num-games {num_games}",
                "--engine gumbel",
            ]

            if model_path:
                cmd_parts.append(f"--model-path {model_path}")

            ssh_command = " ".join(cmd_parts)

            # Execute via SSH with timeout
            result = await run_ssh_command_async(
                host=worker_ip,
                command=ssh_command,
                timeout=300.0,  # 5 minute timeout for selfplay
            )

            if result is not None:
                # SSH succeeded - record success and reset circuit
                record_peer_transport_success(worker_ip, "ssh")
                logger.info(
                    f"SSH fallback succeeded for {worker_id}: {num_games} games dispatched"
                )
                return {
                    "success": True,
                    "games": num_games,
                    "worker_id": worker_id,
                    "via_ssh": True,
                }
            else:
                # SSH returned None (failure)
                record_peer_transport_failure(worker_ip, "ssh")
                return {
                    "success": False,
                    "error": "ssh_command_failed",
                    "worker_id": worker_id,
                }

        except asyncio.TimeoutError:
            record_peer_transport_failure(worker_ip, "ssh")
            logger.warning(f"SSH fallback to {worker_id} timed out")
            return {
                "success": False,
                "error": "ssh_timeout",
                "worker_id": worker_id,
            }

        except Exception as e:
            record_peer_transport_failure(worker_ip, "ssh")
            logger.warning(f"SSH fallback to {worker_id} failed: {e}")
            return {
                "success": False,
                "error": f"ssh_exception: {type(e).__name__}",
                "worker_id": worker_id,
            }

    async def _escalate_failed_dispatch(
        self,
        job_id: str,
        failed_workers: list[str],
        board_type: str,
        num_players: int,
        total_failed_games: int,
    ) -> None:
        """Escalate permanently failed dispatches for recovery.

        Phase 15.1.7 (Dec 2025): When dispatch fails to multiple workers
        after all retries, emit an event for cluster-level recovery.

        Args:
            job_id: Job ID
            failed_workers: List of worker IDs that failed
            board_type: Board type for the job
            num_players: Number of players
            total_failed_games: Total games that couldn't be dispatched
        """
        self.stats.dispatch_escalations += 1

        logger.error(
            f"[{job_id}] Dispatch failed to {len(failed_workers)} workers "
            f"({total_failed_games} games). Escalating for recovery."
        )

        # Emit event for recovery orchestration
        self._emit_task_event(
            "DISPATCH_PERMANENTLY_FAILED",
            job_id,
            "selfplay",
            error="max_retries_exceeded",
            board_type=board_type,
            num_players=num_players,
            failed_workers=failed_workers,
            failed_games=total_failed_games,
        )

        # Also emit as TASK_FAILED for general monitoring
        self._emit_task_event(
            "TASK_FAILED",
            job_id,
            "selfplay",
            error=f"dispatch_failed_to_{len(failed_workers)}_workers",
            board_type=board_type,
            num_players=num_players,
        )

    async def _dispatch_selfplay_to_workers(
        self,
        job_id: str,
        workers: list[Any],
        games_per_worker: int,
        remainder: int,
        board_type: str,
        num_players: int,
        model_path: str | None,
        output_dir: str,
    ) -> dict[str, dict[str, Any]]:
        """Dispatch selfplay tasks to worker nodes via HTTP.

        Args:
            job_id: Job ID for tracking
            workers: List of worker node info objects
            games_per_worker: Base games per worker
            remainder: Extra games to distribute to first workers
            board_type: Board type
            num_players: Number of players
            model_path: Path to model file (or None for heuristic)
            output_dir: Directory for output files

        Returns:
            Dict of worker_id -> dispatch result
        """
        results: dict[str, dict[str, Any]] = {}

        try:
            from scripts.p2p.network import get_client_session
            from aiohttp import ClientTimeout
        except ImportError:
            logger.warning("aiohttp not available for distributed dispatch")
            return results

        # Dec 28, 2025: Use centralized JobDefaults.HEALTH_CHECK_TIMEOUT (30s)
        timeout = ClientTimeout(total=JobDefaults.HEALTH_CHECK_TIMEOUT)

        # Jan 2026: Use weighted engine mix for high-quality diverse training data
        # Priority: GPU Gumbel MCTS (highest quality) > Bandit selection > Weighted mix
        effective_engine_mode = "gumbel-mcts"  # Default to high-quality GPU mode
        engine_extra_args: dict[str, Any] | None = None

        # Try bandit selection first (learns optimal engine per config)
        if model_path:
            try:
                from app.coordination.selfplay_engine_bandit import get_selfplay_engine_bandit
                bandit = get_selfplay_engine_bandit()
                config_key = f"{board_type}_{num_players}p"
                effective_engine_mode = bandit.select_engine(config_key)
                logger.info(
                    f"[EngineBandit] Selected engine '{effective_engine_mode}' for {config_key}"
                )
            except (ImportError, AttributeError) as e:
                # Jan 2026: Use SelfplayScheduler's weighted engine mix for diversity
                # This ensures GPU nodes get GPU engines, CPU nodes get CPU engines
                try:
                    from scripts.p2p.managers.selfplay_scheduler import SelfplayScheduler
                    # Assume GPU availability (workers will be filtered below)
                    effective_engine_mode, engine_extra_args = SelfplayScheduler._select_board_engine(
                        has_gpu=True,  # Prefer GPU engines, filtered below
                        board_type=board_type,
                        num_players=num_players,
                    )
                    logger.info(
                        f"[EngineSelect] Selected '{effective_engine_mode}' for {board_type}_{num_players}p "
                        f"via weighted mix (extra_args={engine_extra_args})"
                    )
                except Exception as mix_err:
                    # Ultimate fallback: gumbel-mcts for quality
                    logger.debug(f"[EngineSelect] Fallback to gumbel-mcts: {e}, {mix_err}")
                    effective_engine_mode = "gumbel-mcts"
                    engine_extra_args = {"budget": 150}
        else:
            # No model: use weighted engine mix for diverse training data
            try:
                from scripts.p2p.managers.selfplay_scheduler import SelfplayScheduler
                effective_engine_mode, engine_extra_args = SelfplayScheduler._select_board_engine(
                    has_gpu=True,  # Prefer GPU engines, filtered below
                    board_type=board_type,
                    num_players=num_players,
                )
                logger.info(
                    f"[EngineSelect] Selected '{effective_engine_mode}' (no model) via weighted mix"
                )
            except Exception as e:
                logger.debug(f"[EngineSelect] Fallback to gumbel-mcts (no model): {e}")
                effective_engine_mode = "gumbel-mcts"
                engine_extra_args = {"budget": 150}
        gpu_required = self._engine_mode_requires_gpu(effective_engine_mode)

        if gpu_required:
            logger.info(
                f"GPU-required engine mode '{effective_engine_mode}' requested, "
                f"filtering workers with GPU capability"
            )

        # Session 17.33 Phase 15: Sort workers by load for fair distribution
        # This ensures underutilized nodes get jobs first, improving fairness
        # across heterogeneous GPU nodes (+8-12% estimated fairness improvement)
        sorted_workers = self._sort_workers_by_load(workers)

        async with get_client_session(timeout) as session:
            for i, worker in enumerate(sorted_workers):
                worker_id = getattr(worker, "node_id", str(worker))
                games = games_per_worker + (1 if i < remainder else 0)

                # Get worker endpoint
                worker_ip = getattr(worker, "best_ip", None) or getattr(worker, "ip", None)
                worker_port = getattr(worker, "port", None) or _get_default_port()

                if not worker_ip:
                    results[worker_id] = {"success": False, "error": "no_ip"}
                    continue

                # December 2025: GPU capability validation before dispatch
                # Skip CPU-only workers for GPU-required engine modes
                if gpu_required and not self._worker_has_gpu(worker):
                    logger.warning(
                        f"Skipping worker {worker_id} for GPU-required mode "
                        f"'{effective_engine_mode}': worker lacks GPU capability"
                    )
                    results[worker_id] = {
                        "success": False,
                        "error": "gpu_required_but_no_gpu",
                        "skipped": True,
                    }
                    continue

                # December 2025: Pre-flight validation before dispatch
                # Verify node is reachable and GPU is healthy (if required)
                is_valid, validation_reason = await self.validate_node_for_job(
                    worker_id, requires_gpu=gpu_required
                )
                if not is_valid:
                    logger.warning(
                        f"Skipping worker {worker_id}: failed validation - {validation_reason}"
                    )
                    results[worker_id] = {
                        "success": False,
                        "error": f"validation_failed: {validation_reason}",
                        "skipped": True,
                    }
                    continue

                # December 2025: Disk space check before dispatch
                has_space, space_reason = await self._check_disk_space_async(
                    worker_id, output_dir, min_free_gb=10.0
                )
                if not has_space:
                    logger.warning(
                        f"Skipping worker {worker_id}: insufficient disk space - {space_reason}"
                    )
                    results[worker_id] = {
                        "success": False,
                        "error": f"disk_space: {space_reason}",
                        "skipped": True,
                    }
                    continue

                # Session 17.39 (Jan 2026): Multi-config capacity check BEFORE dispatch
                # This prevents OOM on GH200/H100 nodes by respecting VRAM-based limits
                config_key = f"{board_type}_{num_players}p"
                if HAS_CONFIG_TRACKER:
                    try:
                        config_tracker = get_node_config_tracker()
                        if config_tracker:
                            # Get GPU VRAM from worker info (fallback to conservative 24GB)
                            gpu_vram = int(
                                getattr(worker, "gpu_vram_gb", 0)
                                or getattr(worker, "memory_gb", 0)
                                or 24  # Conservative default
                            )
                            if not config_tracker.can_add_config(worker_id, gpu_vram, config_key):
                                current_configs = config_tracker.get_active_configs(worker_id)
                                max_configs = config_tracker._get_max_for_vram(gpu_vram)
                                logger.info(
                                    f"[ConfigTracker] Skipping worker {worker_id} for {config_key}: "
                                    f"at max capacity ({len(current_configs)}/{max_configs} configs, "
                                    f"vram={gpu_vram}GB, running={current_configs})"
                                )
                                results[worker_id] = {
                                    "success": False,
                                    "error": f"at_max_configs: {len(current_configs)}/{max_configs}",
                                    "skipped": True,
                                }
                                continue
                    except Exception:
                        pass  # Non-critical, allow dispatch if tracker fails

                # Phase 15.1.7: Build payload for retry-enabled dispatch
                # Jan 18, 2026: Apply weak GPU routing per worker
                worker_engine_mode = self._get_effective_engine_mode(
                    effective_engine_mode, worker
                )
                payload = {
                    "job_id": f"{job_id}_{worker_id}",
                    "parent_job_id": job_id,
                    "board_type": board_type,
                    "num_players": num_players,
                    "num_games": games,
                    "model_path": model_path,
                    "output_dir": output_dir,
                    "engine_mode": worker_engine_mode,
                }
                # Jan 2026: Include engine extra args (e.g., budget for gumbel-mcts)
                if engine_extra_args:
                    payload["engine_extra_args"] = engine_extra_args

                # Phase 15.1.7: Use retry-enabled dispatch method
                result = await self._dispatch_single_worker_with_retry(
                    session=session,
                    worker=worker,
                    job_id=job_id,
                    payload=payload,
                    timeout=timeout,
                    max_retries=3,  # Up to 3 attempts per worker
                )
                results[worker_id] = result
                if result.get("success"):
                    logger.debug(
                        f"Dispatched {games} games to {worker_id} "
                        f"(attempts: {result.get('attempts', 1)})"
                    )
                    # January 2026 Sprint 6: Register spawn for verification tracking
                    config_key = f"{board_type}_{num_players}p"
                    worker_job_id = f"{job_id}_{worker_id}"
                    self._register_spawn(worker_job_id, worker_id, config_key)

                    # Session 17.24: Track job dispatch for load balancing
                    if HAS_QUEUE_TRACKER:
                        try:
                            tracker = get_node_queue_tracker()
                            if tracker:
                                tracker.on_job_dispatched(worker_id)
                        except Exception:
                            pass  # Non-critical

        # Phase 15.1.7: Check for failed dispatches and escalate if needed
        failed_workers = [
            wid for wid, r in results.items()
            if not r.get("success") and not r.get("skipped")
        ]
        if failed_workers:
            total_failed_games = sum(
                games_per_worker + (1 if i < remainder else 0)
                for i, w in enumerate(workers)
                if getattr(w, "node_id", str(w)) in failed_workers
            )
            if len(failed_workers) > len(workers) // 2:
                # Majority failed - escalate for cluster-level recovery
                await self._escalate_failed_dispatch(
                    job_id=job_id,
                    failed_workers=failed_workers,
                    board_type=board_type,
                    num_players=num_players,
                    total_failed_games=total_failed_games,
                )

        return results

    async def run_local_selfplay(
        self,
        job_id: str,
        num_games: int,
        board_type: str,
        num_players: int,
        model_path: str | None,
        output_dir: str,
    ) -> None:
        """Run selfplay locally using subprocess.

        Args:
            job_id: Job ID
            num_games: Number of games to play
            board_type: Board type
            num_players: Number of players
            model_path: Path to model (or None for heuristic)
            output_dir: Output directory for games
        """
        # December 2025: Check disk space before spawning local selfplay
        has_space, space_reason = self._check_disk_space(output_dir)
        if not has_space:
            logger.warning(
                f"Skipping local selfplay job {job_id} due to insufficient disk space: {space_reason}"
            )
            self._emit_task_event(
                "TASK_FAILED", job_id, "selfplay",
                error=f"disk_space_check_failed: {space_reason}",
                board_type=board_type,
            )
            return

        output_file = os.path.join(output_dir, f"{self.node_id}_games.jsonl")

        # Jan 2026 (Session 17.50): Use GPU script on GPU nodes for better utilization
        yaml_has_gpu = self._check_yaml_gpu_config()

        if yaml_has_gpu:
            # GPU node: use generate_gumbel_selfplay.py for GPU-accelerated selfplay
            script_path = self._get_script_path("generate_gumbel_selfplay.py")
            board_norm = board_type.replace("hexagonal", "hex")  # Normalize board type
            cmd = [
                sys.executable,
                script_path,
                "--board", board_norm,
                "--num-players", str(num_players),
                "--num-games", str(num_games),
                "--db", str(Path(output_dir) / "games.db"),
                "--seed", str(int(time.time() * 1000) % 2**31),
                "--simulation-budget", "150",  # Standard budget
            ]
            # Use GPU tree for maximum speedup
            if self._should_use_gpu_tree():
                cmd.append("--use-gpu-tree")
                logger.info(
                    f"[Local Selfplay] Using GPU script with --use-gpu-tree for {self.node_id}"
                )
            else:
                cmd.append("--no-gpu-tree")
                logger.info(
                    f"[Local Selfplay] Using GPU script (no GPU tree) for {self.node_id}"
                )
        else:
            # CPU node: use run_self_play_soak.py for CPU selfplay
            cmd = [
                sys.executable,
                self._get_script_path("run_self_play_soak.py"),
                "--num-games", str(num_games),
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--engine-mode", "gumbel-mcts" if model_path else "maxn",  # High-quality modes
                "--max-moves", "10000",  # Avoid draws due to move limit
                "--log-jsonl", output_file,
            ]
            logger.debug(f"[Local Selfplay] Using CPU script for {self.node_id}")

        # December 29, 2025: Use helper for consistent env setup (includes RINGRIFT_ALLOW_PENDING_GATE)
        env = self._get_subprocess_env()

        # December 28, 2025: Initialize proc before try block for proper cleanup
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            # December 28, 2025: Register process for tracking
            self._register_process(job_id, proc)

            _stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=3600  # 1 hour max
            )

            # December 28, 2025: Unregister process after completion
            self._unregister_process(job_id)

            if proc.returncode == 0:
                logger.info(f"Local selfplay completed: {num_games} games")
                # Update progress
                if job_id in self.improvement_loop_state:
                    self.improvement_loop_state[job_id].selfplay_progress[self.node_id] = num_games
            else:
                error_msg = stderr.decode()[:500] if stderr else "Unknown error"
                logger.warning(f"Local selfplay failed: {error_msg}")
                # Dec 2025: Emit TASK_FAILED for local selfplay failures
                self._emit_task_event(
                    "TASK_FAILED", job_id, "selfplay",
                    error=f"returncode={proc.returncode}: {error_msg}",
                    board_type=board_type,
                )

        except asyncio.TimeoutError:
            # December 28, 2025: Kill subprocess on timeout to prevent zombies
            await self._kill_process(job_id, proc)
            logger.warning("Local selfplay timed out")
            # Dec 2025: Emit TASK_FAILED for timeout
            self._emit_task_event("TASK_FAILED", job_id, "selfplay", error="timeout", board_type=board_type)
        except (OSError, ValueError, RuntimeError) as e:
            # Dec 2025: Narrowed from broad Exception - subprocess errors
            await self._kill_process(job_id, proc)
            logger.error(f"Local selfplay error: {e}")
            self._emit_task_event("TASK_FAILED", job_id, "selfplay", error=str(e), board_type=board_type)
        except (asyncio.CancelledError, ChildProcessError, BrokenPipeError) as e:
            # Dec 2025: Narrowed to subprocess cancellation/pipe errors
            await self._kill_process(job_id, proc)
            logger.warning(f"Local selfplay interrupted: {type(e).__name__}: {e}")
            self._emit_task_event("TASK_FAILED", job_id, "selfplay", error=str(e), board_type=board_type)
            raise

    # =========================================================================
    # Training Job Methods
    # =========================================================================

    async def export_training_data(self, job_id: str) -> None:
        """Export training data from selfplay games.

        Converts JSONL game records to training format (HDF5 or NPZ).

        Args:
            job_id: Job ID for the improvement loop
        """
        state = self.improvement_loop_state.get(job_id)
        if not state:
            return

        logger.info(f"Exporting training data for job {job_id}, iteration {state.current_iteration}")

        iteration_dir = self._get_data_path(
            "selfplay", f"improve_{job_id}", f"iter_{state.current_iteration}"
        )
        output_file = self._get_data_path(
            "training", f"improve_{job_id}", f"iter_{state.current_iteration}.npz"
        )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # December 2025: Check disk space before export job
        has_space, space_reason = self._check_disk_space(os.path.dirname(output_file))
        if not has_space:
            logger.warning(
                f"Skipping export job {job_id} due to insufficient disk space: {space_reason}"
            )
            self._emit_task_event(
                "TASK_FAILED", job_id, "export",
                error=f"disk_space_check_failed: {space_reason}",
            )
            return

        jsonl_files = list(Path(iteration_dir).glob("*.jsonl"))
        if not jsonl_files:
            logger.warning(f"No JSONL files found for export in {iteration_dir}")
            return

        cmd = [
            sys.executable,
            self._get_script_path("jsonl_to_npz.py"),
            "--input-dir",
            iteration_dir,
            "--output",
            output_file,
            "--board-type",
            state.board_type,
            "--num-players",
            str(state.num_players),
        ]

        # December 29, 2025: Use helper for consistent env setup (includes RINGRIFT_ALLOW_PENDING_GATE)
        # Note: Export doesn't skip shadow contracts (include_shadow_skip=False)
        env = self._get_subprocess_env(include_shadow_skip=False)

        # December 28, 2025: Initialize proc before try block for proper cleanup
        export_job_id = f"{job_id}_export"  # Unique ID for process tracking
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            # December 28, 2025: Register process for tracking
            self._register_process(export_job_id, proc)

            _stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=600  # 10 minutes max
            )

            # December 28, 2025: Unregister process after completion
            self._unregister_process(export_job_id)

            if proc.returncode == 0:
                logger.info(f"Exported training data to {output_file}")
                state.training_data_path = output_file
            else:
                error_msg = stderr.decode()[:500] if stderr else "Unknown error"
                logger.warning(f"Training data export failed: {error_msg}")
                # Dec 2025: Emit TASK_FAILED for export failures
                self._emit_task_event(
                    "TASK_FAILED", job_id, "export",
                    error=f"returncode={proc.returncode}: {error_msg}",
                )

        except asyncio.TimeoutError:
            # December 28, 2025: Kill subprocess on timeout to prevent zombies
            await self._kill_process(export_job_id, proc)
            logger.warning("Training data export timed out")
            # Dec 2025: Emit TASK_FAILED for timeout
            self._emit_task_event("TASK_FAILED", job_id, "export", error="timeout")
        except (OSError, ValueError, RuntimeError) as e:
            # Dec 2025: Narrowed from broad Exception - subprocess errors
            await self._kill_process(export_job_id, proc)
            logger.error(f"Training data export error: {e}")
            self._emit_task_event("TASK_FAILED", job_id, "export", error=str(e))
        except (asyncio.CancelledError, ChildProcessError, BrokenPipeError) as e:
            # Dec 2025: Narrowed to subprocess cancellation/pipe errors
            await self._kill_process(export_job_id, proc)
            logger.warning(f"Training export interrupted: {type(e).__name__}: {e}")
            self._emit_task_event("TASK_FAILED", job_id, "export", error=str(e))
            raise

    async def run_training(self, job_id: str) -> None:
        """Run neural network training on GPU node.

        Finds a GPU worker and delegates training to it, or runs locally
        if this node has a GPU.

        CRITICAL FIX (Jan 2026): Now uses --init-weights to load from canonical
        model instead of training from scratch. This enables incremental learning
        where each iteration builds on previous knowledge.

        Args:
            job_id: Job ID for the improvement loop
        """
        state = self.improvement_loop_state.get(job_id)
        if not state:
            return

        logger.info(f"Running training for job {job_id}, iteration {state.current_iteration}")

        # Find GPU worker
        gpu_worker = None
        with self.peers_lock:
            for peer in self.peers.values():
                if hasattr(peer, 'has_gpu') and peer.has_gpu and hasattr(peer, 'is_healthy') and peer.is_healthy():
                    gpu_worker = peer
                    break

        # Model output path
        new_model_path = os.path.join(
            self.ringrift_path, "ai-service", "models",
            f"improve_{job_id}", f"iter_{state.current_iteration}.pt"
        )
        os.makedirs(os.path.dirname(new_model_path), exist_ok=True)

        # Jan 2026: Find canonical model for warm start (incremental learning)
        # This prevents catastrophic forgetting by starting from best known weights
        config_key = f"{state.board_type}_{state.num_players}p"
        canonical_model_path = os.path.join(
            self.ringrift_path, "ai-service", "models",
            f"canonical_{config_key}.pth"
        )
        init_weights = canonical_model_path if os.path.exists(canonical_model_path) else None

        # Jan 2026: Detect model version from canonical checkpoint to ensure architecture match
        model_version = "v2"  # Default
        if init_weights:
            try:
                import torch
                checkpoint = torch.load(init_weights, map_location="cpu", weights_only=True)
                metadata = checkpoint.get("_versioning_metadata", {})
                model_version = metadata.get("model_version", "v2")
                logger.info(f"Using warm start from canonical model: {init_weights} (version: {model_version})")
            except Exception as e:
                logger.warning(f"Could not detect model version from {init_weights}: {e}, using default v2")
                logger.info(f"Using warm start from canonical model: {init_weights}")
        else:
            logger.warning(f"No canonical model found for {config_key}, training from scratch")

        training_config = {
            "job_id": job_id,
            "iteration": state.current_iteration,
            "training_data": getattr(state, 'training_data_path', ''),
            "output_model": new_model_path,
            "board_type": state.board_type,
            "num_players": state.num_players,
            "epochs": 10,
            "batch_size": 256,
            "learning_rate": 0.001,
            "init_weights": init_weights,  # Jan 2026: Warm start from canonical
            "model_version": model_version,  # Jan 2026: Match canonical model architecture
        }

        # For distributed training, would delegate to GPU worker here
        # For now, run locally
        await self.run_local_training(training_config)
        state.candidate_model_path = new_model_path

    async def run_local_training(self, config: dict) -> None:
        """Run training locally using subprocess.

        CRITICAL FIX (Jan 2026): Uses proper train.py script with --init-weights
        for incremental learning instead of creating a fresh model each time.

        Args:
            config: Training configuration dict containing:
                - job_id: Job identifier for event tracking
                - training_data: Path to training data
                - output_model: Path to save trained model
                - board_type, num_players, epochs, batch_size, learning_rate
                - init_weights: Path to canonical model for warm start (Jan 2026)
        """
        # Dec 2025: Extract job_id from config for event emission
        job_id = config.get("job_id", "unknown")
        logger.info(f"Running local training for job {job_id}")

        # Jan 2026: Build proper training command using train.py
        # This enables incremental learning with --init-weights
        cmd = [
            sys.executable, "-m", "app.training.train",
            "--board-type", config.get("board_type", "square8"),
            "--num-players", str(config.get("num_players", 2)),
            "--data-path", config.get("training_data", ""),
            "--save-path", config.get("output_model", "/tmp/model.pt"),
            "--epochs", str(config.get("epochs", 10)),
            "--batch-size", str(config.get("batch_size", 256)),
            "--learning-rate", str(config.get("learning_rate", 0.001)),
            "--model-version", config.get("model_version", "v2"),  # Jan 2026: Use detected version
        ]

        # Jan 2026: Add init_weights for warm start (incremental learning)
        init_weights = config.get("init_weights")
        if init_weights and os.path.exists(init_weights):
            cmd.extend(["--init-weights", init_weights])
            logger.info(f"Training with warm start from: {init_weights}")
        # December 29, 2025: Use helper for consistent env setup (includes RINGRIFT_ALLOW_PENDING_GATE)
        # Note: Training doesn't skip shadow contracts (include_shadow_skip=False)
        env = self._get_subprocess_env(include_shadow_skip=False)

        # December 28, 2025: Initialize proc before try block for proper cleanup
        training_job_id = f"{job_id}_training"  # Unique ID for process tracking
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            # December 28, 2025: Register process for tracking
            self._register_process(training_job_id, proc)

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=3600  # 1 hour max
            )

            # December 28, 2025: Unregister process after completion
            self._unregister_process(training_job_id)

            logger.info(f"Training output: {stdout.decode()}")
            if proc.returncode != 0:
                error_msg = stderr.decode()[:500] if stderr else "Unknown error"
                logger.warning(f"Training stderr: {error_msg}")
                # Dec 2025: Emit TASK_FAILED for training failures
                self._emit_task_event(
                    "TASK_FAILED", job_id, "training",
                    error=f"returncode={proc.returncode}: {error_msg}",
                )

        except asyncio.TimeoutError:
            # December 28, 2025: Kill subprocess on timeout to prevent zombies
            await self._kill_process(training_job_id, proc)
            logger.warning("Local training timed out")
            # Dec 2025: Emit TASK_FAILED for timeout
            self._emit_task_event("TASK_FAILED", job_id, "training", error="timeout")
        except (OSError, ValueError, RuntimeError) as e:
            # Dec 2025: Narrowed from broad Exception - subprocess errors
            await self._kill_process(training_job_id, proc)
            logger.error(f"Local training error: {e}")
            self._emit_task_event("TASK_FAILED", job_id, "training", error=str(e))
        except (asyncio.CancelledError, ChildProcessError, BrokenPipeError) as e:
            # Dec 2025: Narrowed to subprocess cancellation/pipe errors
            await self._kill_process(training_job_id, proc)
            logger.warning(f"Local training interrupted: {type(e).__name__}: {e}")
            self._emit_task_event("TASK_FAILED", job_id, "training", error=str(e))
            raise

    # =========================================================================
    # Tournament Job Methods
    # =========================================================================

    async def run_distributed_tournament(self, job_id: str) -> None:
        """Main coordinator loop for distributed tournament.

        Distributes matches across workers, collects results, and calculates
        Elo rating updates.

        Args:
            job_id: Tournament job ID
        """
        try:
            state = self.distributed_tournament_state.get(job_id)
            if not state:
                logger.warning(f"Tournament {job_id} not found in state dict (keys: {list(self.distributed_tournament_state.keys())})")
                return

            logger.info(f"Tournament coordinator started for job {job_id}")
            state.status = "running"

            # Get tournament configuration
            # Support both old (models/games_per_pair) and new (agent_ids/games_per_pairing) attribute names
            agent_ids = getattr(state, "agent_ids", None) or getattr(state, "models", [])
            games_per_pairing = getattr(state, "games_per_pairing", None) or getattr(state, "games_per_pair", 10)
            board_type = getattr(state, "board_type", "hex8")
            num_players = getattr(state, "num_players", 2)

            if len(agent_ids) < 2:
                logger.warning(f"Tournament {job_id} needs at least 2 agents")
                state.status = "error: not enough agents"
                return

            # Generate match pairings (round-robin)
            matches = self._generate_tournament_matches(agent_ids, games_per_pairing)
            state.total_matches = len(matches)
            state.completed_matches = 0

            logger.info(f"Tournament {job_id}: {len(matches)} matches for {len(agent_ids)} agents")

            # Get available workers
            workers = self._get_tournament_workers()
            if not workers:
                # Run locally if no workers
                logger.info("No workers available, running tournament locally")
                results = await self._run_tournament_matches_locally(
                    job_id, matches, board_type, num_players
                )
            else:
                # Distribute matches to workers
                results = await self._dispatch_tournament_matches(
                    job_id, matches, workers, board_type, num_players
                )

            # Calculate Elo updates
            # Dec 28, 2025: Fixed undefined 'models' -> 'agent_ids'
            elo_updates = self._calculate_elo_updates(agent_ids, results)
            state.elo_updates = elo_updates

            # Update state
            state.results = results
            state.status = "completed"

            logger.info(
                f"Tournament {job_id} completed: {state.completed_matches} matches, "
                f"Elo updates: {elo_updates}"
            )

            # Emit tournament completion event
            self._emit_task_event(
                "TASK_COMPLETED",
                job_id,
                "tournament",
                models=agent_ids,  # Dec 28, 2025: Fixed undefined 'models' -> 'agent_ids'
                total_matches=state.total_matches,
                elo_updates=elo_updates,
            )

        except (KeyError, AttributeError, ValueError, TypeError) as e:
            # Dec 2025: Narrowed from broad Exception - tournament state/data errors
            # KeyError: Missing state key
            # AttributeError: Missing state attribute
            # ValueError/TypeError: Invalid match data
            logger.error(f"Tournament coordinator error: {e}")
            if job_id in self.distributed_tournament_state:
                self.distributed_tournament_state[job_id].status = f"error: {e}"
            self._emit_task_event("TASK_FAILED", job_id, "tournament", error=str(e))
        except (asyncio.TimeoutError, asyncio.CancelledError, ConnectionError) as e:
            # Dec 28, 2025: Added narrower handler for async/network errors
            # TimeoutError: Matches didn't complete in time
            # CancelledError: Tournament cancelled by caller
            # ConnectionError: Worker communication failure
            logger.exception(f"Tournament {job_id} communication error: {e}")
            if job_id in self.distributed_tournament_state:
                self.distributed_tournament_state[job_id].status = f"error: {type(e).__name__}"
            self._emit_task_event("TASK_FAILED", job_id, "tournament", error=str(e))
        except (aiohttp.ClientError, aiohttp.ServerTimeoutError) as e:
            # Dec 2025: Narrowed to network client errors not caught above
            logger.exception(f"Tournament {job_id} network error: {e}")
            if job_id in self.distributed_tournament_state:
                self.distributed_tournament_state[job_id].status = f"error: {type(e).__name__}"
            self._emit_task_event("TASK_FAILED", job_id, "tournament", error=str(e))
            raise

    def _generate_tournament_matches(
        self,
        models: list[str],
        games_per_pair: int,
    ) -> list[dict[str, Any]]:
        """Generate round-robin match pairings.

        Args:
            models: List of model paths
            games_per_pair: Number of games per model pair

        Returns:
            List of match configurations
        """
        matches = []
        for i, model_a in enumerate(models):
            for model_b in models[i + 1:]:
                for game_num in range(games_per_pair):
                    # Alternate starting player
                    first_player = model_a if game_num % 2 == 0 else model_b
                    second_player = model_b if game_num % 2 == 0 else model_a
                    matches.append({
                        "match_id": f"{i}_{len(matches)}",
                        "player1_model": first_player,
                        "player2_model": second_player,
                        "game_num": game_num,
                    })
        return matches

    def _get_tournament_workers(self) -> list[Any]:
        """Get available workers for tournament matches.

        Dec 29, 2025: Filter out NAT-blocked workers since direct HTTP won't work.

        Returns:
            List of worker node info objects (only non-NAT-blocked)
        """
        workers = []
        with self.peers_lock:
            for peer in self.peers.values():
                if hasattr(peer, "is_healthy") and peer.is_healthy():
                    # Skip NAT-blocked workers - can't reach them directly
                    if getattr(peer, "nat_blocked", False):
                        continue
                    # Prefer GPU nodes for neural net matches
                    if hasattr(peer, "has_gpu") and peer.has_gpu:
                        workers.insert(0, peer)
                    else:
                        workers.append(peer)
        return workers

    async def _dispatch_tournament_matches(
        self,
        job_id: str,
        matches: list[dict[str, Any]],
        workers: list[Any],
        board_type: str,
        num_players: int,
    ) -> list[dict[str, Any]]:
        """Dispatch tournament matches to workers.

        Args:
            job_id: Tournament job ID
            matches: List of match configurations
            workers: List of worker nodes
            board_type: Board type
            num_players: Number of players

        Returns:
            List of match results
        """
        results = []

        try:
            from scripts.p2p.network import get_client_session
            from aiohttp import ClientTimeout
        except ImportError:
            logger.warning("aiohttp not available for distributed tournament")
            return await self._run_tournament_matches_locally(
                job_id, matches, board_type, num_players
            )

        timeout = ClientTimeout(total=120)  # 2 min per match
        state = self.distributed_tournament_state.get(job_id)

        # Distribute matches across workers (round-robin)
        worker_assignments: dict[str, list[dict]] = {
            getattr(w, "node_id", str(w)): [] for w in workers
        }

        for i, match in enumerate(matches):
            worker = workers[i % len(workers)]
            worker_id = getattr(worker, "node_id", str(worker))
            worker_assignments[worker_id].append(match)

        # Dispatch individual matches to workers via /tournament/match endpoint
        # Dec 28, 2025: Fixed to use correct endpoint (was /run-tournament-matches)
        async with get_client_session(timeout) as session:
            dispatch_tasks = []

            for worker in workers:
                worker_id = getattr(worker, "node_id", str(worker))
                worker_matches = worker_assignments.get(worker_id, [])
                if not worker_matches:
                    continue

                # Dec 28, 2025: Peer info uses 'host' attribute, not 'ip' or 'best_ip'
                worker_ip = getattr(worker, "best_ip", None) or getattr(worker, "host", None) or getattr(worker, "ip", None)
                worker_port = getattr(worker, "port", None) or _get_default_port()

                if not worker_ip:
                    logger.debug(f"Worker {worker_id} has no IP address, skipping")
                    continue

                # Dec 29, 2025: Skip NAT-blocked workers - direct connection will fail
                if getattr(worker, "nat_blocked", False):
                    logger.debug(f"Worker {worker_id} is NAT-blocked, skipping")
                    continue

                # Send each match individually to /tournament/match endpoint
                for match in worker_matches:
                    # Enrich match with board info for the worker
                    match["board_type"] = board_type
                    match["num_players"] = num_players

                    url = f"http://{worker_ip}:{worker_port}/tournament/match"
                    payload = {
                        "job_id": job_id,
                        "match": match,
                    }

                    dispatch_tasks.append(
                        self._send_tournament_request(session, url, payload, worker_id, timeout)
                    )

            # Wait for all workers to complete
            worker_results = await asyncio.gather(*dispatch_tasks, return_exceptions=True)

            for result in worker_results:
                if isinstance(result, Exception):
                    logger.warning(f"Tournament worker error: {result}")
                elif isinstance(result, list):
                    results.extend(result)
                    if state:
                        state.completed_matches += len(result)

        return results

    async def _send_tournament_request(
        self,
        session: Any,
        url: str,
        payload: dict,
        worker_id: str,
        timeout: Any,
    ) -> list[dict[str, Any]]:
        """Send tournament match request to a worker.

        Args:
            session: aiohttp session
            url: Worker endpoint URL
            payload: Match request payload
            worker_id: Worker node ID
            timeout: Request timeout

        Returns:
            List of match results from worker
        """
        # Sprint 15.1.1: Extract host from URL for per-transport CB
        from urllib.parse import urlparse
        parsed = urlparse(url)
        host = parsed.hostname or ""

        # Sprint 15.1.1: Check per-transport circuit breaker before attempting
        if host and not check_peer_transport_circuit(host, "http"):
            logger.debug(f"Skipping tournament request to {worker_id}: HTTP circuit breaker is OPEN")
            return []

        try:
            async with session.post(url, json=payload, timeout=timeout) as resp:
                if resp.status == 200:
                    # Sprint 15.1.1: Record success for per-transport CB
                    if host:
                        record_peer_transport_success(host, "http")
                    result = await resp.json()
                    match_results = result.get("results", [])
                    logger.debug(
                        f"Worker {worker_id} completed {len(match_results)} matches"
                    )
                    return match_results
                else:
                    # Sprint 15.1.1: Record failure for per-transport CB
                    if host:
                        record_peer_transport_failure(host, "http")
                    logger.warning(f"Worker {worker_id} returned status {resp.status}")
                    return []
        except asyncio.TimeoutError:
            # Sprint 15.1.1: Record failure for per-transport CB
            if host:
                record_peer_transport_failure(host, "http")
            logger.warning(f"Worker {worker_id} timed out")
            return []
        except (OSError, ConnectionError) as e:
            # Dec 2025: Narrowed from broad Exception - network/connection errors
            # Sprint 15.1.1: Record failure for per-transport CB
            if host:
                record_peer_transport_failure(host, "http")
            logger.warning(f"Worker {worker_id} error: {e}")
            return []

    async def _run_tournament_matches_locally(
        self,
        job_id: str,
        matches: list[dict[str, Any]],
        board_type: str,
        num_players: int,
    ) -> list[dict[str, Any]]:
        """Run tournament matches locally when no workers available.

        Args:
            job_id: Tournament job ID
            matches: List of match configurations
            board_type: Board type
            num_players: Number of players

        Returns:
            List of match results
        """
        results = []
        state = self.distributed_tournament_state.get(job_id)

        for match in matches:
            try:
                # Import game runner (lazy to avoid circular imports)
                from app.ai.heuristic_ai import HeuristicAI

                # For now, run heuristic-only matches for testing
                # Full implementation would load neural net models
                result = {
                    "match_id": match["match_id"],
                    "player1_model": match["player1_model"],
                    "player2_model": match["player2_model"],
                    "winner": None,  # Would be determined by actual game
                    "moves": 0,
                }
                results.append(result)

                if state:
                    state.completed_matches += 1

            except (KeyError, ImportError, RuntimeError, ValueError) as e:
                # Dec 2025: Narrowed from broad Exception - match execution errors
                # KeyError: Missing match data
                # ImportError: AI module not available
                # RuntimeError: Game state error
                # ValueError: Invalid game parameters
                logger.warning(f"Local match error: {e}")

        return results

    def _calculate_elo_updates(
        self,
        models: list[str],
        results: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Calculate Elo rating updates from match results.

        Uses standard Elo formula with K=32.

        Args:
            models: List of model paths
            results: List of match results

        Returns:
            Dict of model -> Elo delta
        """
        K = 32  # Elo K-factor
        elo_deltas: dict[str, float] = {m: 0.0 for m in models}
        game_counts: dict[str, int] = {m: 0 for m in models}

        # Current ratings (start at 1500)
        ratings = {m: 1500.0 for m in models}

        for result in results:
            p1 = result.get("player1_model")
            p2 = result.get("player2_model")
            winner = result.get("winner")

            if p1 not in ratings or p2 not in ratings:
                continue

            # Calculate expected scores
            r1, r2 = ratings[p1], ratings[p2]
            e1 = 1.0 / (1.0 + 10 ** ((r2 - r1) / 400))
            e2 = 1.0 - e1

            # Actual scores
            if winner == p1:
                s1, s2 = 1.0, 0.0
            elif winner == p2:
                s1, s2 = 0.0, 1.0
            else:  # Draw
                s1, s2 = 0.5, 0.5

            # Update ratings
            delta1 = K * (s1 - e1)
            delta2 = K * (s2 - e2)

            elo_deltas[p1] += delta1
            elo_deltas[p2] += delta2
            game_counts[p1] += 1
            game_counts[p2] += 1

        return elo_deltas

    async def run_model_comparison_tournament(self, config: dict) -> None:
        """Run a model comparison tournament.

        Args:
            config: Tournament configuration
        """
        logger.info(f"Running model comparison tournament: {config}")
        # Placeholder for tournament logic

    # =========================================================================
    # Gauntlet and Validation Methods
    # =========================================================================

    async def run_post_training_gauntlet(self, job: Any) -> bool:
        """Run gauntlet evaluation after training.

        Args:
            job: TrainingJob object

        Returns:
            True if gauntlet passed, False otherwise
        """
        logger.info(f"Running post-training gauntlet for job {job.job_id}")
        # Placeholder for gauntlet logic
        return True

    # =========================================================================
    # Job Lifecycle and Metrics
    # =========================================================================
    def cleanup_completed_jobs(self) -> int:
        """Clean up completed jobs from tracking.

        Returns:
            Number of jobs cleaned up
        """
        cleaned = 0
        with self.jobs_lock:
            for job_type in list(self.active_jobs.keys()):
                for job_id in list(self.active_jobs[job_type].keys()):
                    job = self.active_jobs[job_type][job_id]
                    status = job.get("status") if isinstance(job, dict) else getattr(job, "status", "running")
                    if status in ("completed", "failed", "timeout", "error"):
                        del self.active_jobs[job_type][job_id]
                        cleaned += 1
        return cleaned

    # =========================================================================
    # Health Check (December 2025)
    # =========================================================================

    def health_check(self):
        """Check health status of JobManager.

        Returns:
            HealthCheckResult with status, job counts, and error info
        """
        # Import from contracts (zero dependencies)
        from app.coordination.contracts import HealthCheckResult, CoordinatorStatus

        status = CoordinatorStatus.RUNNING
        is_healthy = True
        errors_count = 0
        last_error: str | None = None
        total_jobs = 0
        failed_jobs = 0
        running_jobs = 0

        with self.jobs_lock:
            for job_type, jobs in self.active_jobs.items():
                for job_id, job in jobs.items():
                    total_jobs += 1
                    job_status = job.get("status") if isinstance(job, dict) else getattr(job, "status", "running")
                    if job_status == "running":
                        running_jobs += 1
                    elif job_status in ("failed", "error", "timeout"):
                        failed_jobs += 1

        # Degrade status if high failure rate
        if total_jobs > 0:
            failure_rate = failed_jobs / total_jobs
            if failure_rate > 0.5:
                status = CoordinatorStatus.ERROR
                is_healthy = False
                last_error = f"High job failure rate: {failure_rate:.0%}"
                errors_count = failed_jobs
            elif failure_rate > 0.2:
                status = CoordinatorStatus.DEGRADED
                last_error = f"Elevated job failure rate: {failure_rate:.0%}"
                errors_count = failed_jobs

        # Check if subscribed to events
        if not self._subscribed:
            if is_healthy:
                status = CoordinatorStatus.DEGRADED
                last_error = "Not subscribed to events"

        return HealthCheckResult(
            healthy=is_healthy,
            status=status if isinstance(status, str) else status,
            message=last_error or "JobManager healthy",
            details={
                "operations_count": total_jobs,
                "errors_count": errors_count,
                "running_jobs": running_jobs,
                "failed_jobs": failed_jobs,
                "job_types": list(self.active_jobs.keys()),
                "subscribed": self._subscribed,
            },
        )
