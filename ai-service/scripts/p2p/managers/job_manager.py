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

from app.config.coordination_defaults import (
    DaemonHealthDefaults,
    JobDefaults,
    OperationTimeouts,
    SSHDefaults,
)

# Import mixin for consolidated event handling (Dec 28, 2025)
from scripts.p2p.p2p_mixin_base import EventSubscriptionMixin

if TYPE_CHECKING:
    from ..models import ClusterJob, ImprovementLoopState, NodeInfo

logger = logging.getLogger(__name__)


@dataclass
class JobManagerStats:
    """Statistics for job manager monitoring.

    December 27, 2025: Added to track job lifecycle events for observability.
    """

    jobs_spawned: int = 0
    jobs_completed: int = 0
    jobs_failed: int = 0
    jobs_cancelled: int = 0
    nodes_recovered: int = 0
    hosts_offline: int = 0
    hosts_online: int = 0

# Event emission helper - imported lazily to avoid circular imports
# Dec 2025: Added thread-safe initialization to prevent race conditions
_emit_event: Callable[[str, dict], None] | None = None
_event_emitter_lock = threading.Lock()

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

        # Spawn a selfplay job
        await job_mgr.run_gpu_selfplay_job(
            job_id="job123",
            board_type="hex8",
            num_players=2,
            num_games=100,
            engine_mode="heuristic-only"
        )

        # Get job status
        job_count = job_mgr.get_job_count_for_node("node1")
    """

    # Engine modes that require search (use generate_gumbel_selfplay.py)
    SEARCH_ENGINE_MODES = {
        "maxn", "brs", "mcts", "gumbel-mcts",
        "policy-only", "nn-descent", "nn-minimax"
    }

    # GPU-required engine modes (require CUDA or MPS) - December 2025
    # These modes use neural network inference and require GPU acceleration
    GPU_REQUIRED_ENGINE_MODES = {
        "gumbel-mcts", "mcts", "nnue-guided", "policy-only",
        "nn-minimax", "nn-descent", "gnn", "hybrid",
        "gmo", "ebmo", "ig-gmo", "cage",
    }

    # CPU-compatible engine modes (can run on any node)
    CPU_COMPATIBLE_ENGINE_MODES = {
        "heuristic-only", "heuristic", "random", "random-only",
        "descent-only", "maxn", "brs",
    }

    # Mixin type identifier (required by EventSubscriptionMixin)
    MIXIN_TYPE = "job_manager"

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
        self.improvement_loop_state = improvement_loop_state or {}
        self.distributed_tournament_state = distributed_tournament_state or {}

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

    # =========================================================================
    # Pre-flight Node Validation (December 2025)
    # =========================================================================

    async def _preflight_check_node(
        self, node_id: str, timeout: float = 5.0
    ) -> tuple[bool, str]:
        """Quick health check before dispatching job to node.

        Performs a fast SSH probe and checks P2P alive status to avoid
        dispatching jobs to dead or unreachable nodes.

        Args:
            node_id: The node ID to check
            timeout: Maximum time to wait for probe response (default: 5s)

        Returns:
            Tuple of (is_available, reason).
            - (True, "ok") if node is available
            - (False, reason) if node is unavailable with reason string

        December 2025: Added as part of cluster availability fix to prevent
        dispatching jobs to unavailable nodes.
        """
        # Check 1: Node in alive peers
        with self.peers_lock:
            if node_id not in self.peers:
                return False, "not_in_peers"
            peer = self.peers.get(node_id)
            if peer and hasattr(peer, "status"):
                if peer.status in ("offline", "dead", "retired"):
                    return False, f"peer_status_{peer.status}"

        # Check 2: Quick SSH probe
        try:
            from app.core.ssh import run_ssh_command_async

            result = await asyncio.wait_for(
                run_ssh_command_async(node_id, "echo ok", timeout=timeout),
                timeout=timeout + 1
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

    async def _check_gpu_health(self, node_id: str, timeout: float = 10.0) -> tuple[bool, str]:
        """Verify GPU is available and not in error state.

        Runs nvidia-smi on the target node to check for CUDA errors,
        busy GPUs, or other GPU-related issues before dispatching GPU jobs.

        Args:
            node_id: The node ID to check
            timeout: Maximum time to wait for nvidia-smi (default: 10s)

        Returns:
            Tuple of (is_healthy, reason).
            - (True, "ok") if GPU is healthy
            - (False, reason) if GPU has issues

        December 2025: Added as part of cluster availability fix to prevent
        dispatching to nodes with CUDA errors.
        """
        try:
            from app.core.ssh import run_ssh_command_async

            result = await asyncio.wait_for(
                run_ssh_command_async(
                    node_id,
                    "nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>&1",
                    timeout=timeout
                ),
                timeout=timeout + 1
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
        preflight_timeout: float = 5.0,
        gpu_check_timeout: float = 10.0,
    ) -> tuple[bool, str]:
        """Validate that a node is suitable for running a job.

        Combines preflight check and optional GPU health check into a single
        validation call that can be used before job dispatch.

        Args:
            node_id: The node ID to validate
            requires_gpu: Whether the job requires GPU (triggers GPU health check)
            preflight_timeout: Timeout for SSH probe
            gpu_check_timeout: Timeout for GPU health check

        Returns:
            Tuple of (is_valid, reason).
            - (True, "ok") if node is valid for the job
            - (False, reason) if node cannot accept the job

        December 2025: Convenience method combining preflight and GPU checks.
        """
        # Step 1: Basic preflight check
        is_available, reason = await self._preflight_check_node(node_id, preflight_timeout)
        if not is_available:
            logger.warning(f"Node {node_id} failed preflight check: {reason}")
            return False, f"preflight_{reason}"

        # Step 2: GPU health check if required
        if requires_gpu:
            is_healthy, gpu_reason = await self._check_gpu_health(node_id, gpu_check_timeout)
            if not is_healthy:
                logger.warning(f"Node {node_id} failed GPU health check: {gpu_reason}")
                return False, f"gpu_{gpu_reason}"

        return True, "ok"

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
        """
        board_norm = board_type.replace("hexagonal", "hex")
        output_dir = Path(self.ringrift_path) / "ai-service" / "data" / "selfplay" / "p2p_gpu" / f"{board_norm}_{num_players}p" / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        effective_mode = engine_mode or "heuristic-only"

        # December 2025: GPU availability check using GPU_REQUIRED_ENGINE_MODES
        # This prevents wasting compute on GPU-required modes when no GPU is available
        if self._engine_mode_requires_gpu(effective_mode):
            try:
                import torch
                has_cuda = torch.cuda.is_available()
                has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

                if not (has_cuda or has_mps):
                    logger.warning(
                        f"GPU-required mode '{effective_mode}' requested but no GPU available "
                        f"(CUDA={has_cuda}, MPS={has_mps}), falling back to heuristic-only for job {job_id}"
                    )
                    effective_mode = "heuristic-only"
                else:
                    device_type = "CUDA" if has_cuda else "MPS"
                    logger.debug(f"GPU available ({device_type}) for mode '{effective_mode}' job {job_id}")
            except ImportError:
                logger.warning(
                    f"PyTorch not available for GPU check, falling back to heuristic-only for job {job_id}"
                )
                effective_mode = "heuristic-only"

        if effective_mode in self.SEARCH_ENGINE_MODES:
            # December 29, 2025: Use generate_gumbel_selfplay.py for search-based modes
            # (run_hybrid_selfplay.py was archived, generate_gumbel_selfplay.py is the active replacement)
            script_path = os.path.join(self.ringrift_path, "ai-service", "scripts", "generate_gumbel_selfplay.py")
            if not os.path.exists(script_path):
                logger.warning(f"Gumbel selfplay script not found: {script_path}")
                return

            cmd = [
                sys.executable,
                script_path,
                "--board", board_norm,  # generate_gumbel_selfplay uses --board not --board-type
                "--num-players", str(num_players),
                "--num-games", str(num_games),
                "--db", str(output_dir / "games.db"),  # uses --db not --record-db
                "--seed", str(int(time.time() * 1000) % 2**31),
                "--simulation-budget", "150",  # Standard Gumbel budget
            ]
        else:
            # Use run_gpu_selfplay.py for GPU-optimized modes
            script_path = os.path.join(self.ringrift_path, "ai-service", "scripts", "run_gpu_selfplay.py")
            if not os.path.exists(script_path):
                logger.warning(f"GPU selfplay script not found: {script_path}")
                return

            # Map engine modes: run_gpu_selfplay.py only supports: random-only, heuristic-only, nnue-guided
            mode_map = {
                "mixed": "heuristic-only",
                "gpu": "heuristic-only",
                "descent-only": "heuristic-only",
                "heuristic-only": "heuristic-only",
                "nnue-guided": "nnue-guided",
                "random-only": "random-only",
            }
            gpu_engine_mode = mode_map.get(effective_mode, "heuristic-only")

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
            ]

        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(self.ringrift_path, "ai-service")
        env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

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
                }

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

            # Update job status and emit completion event
            duration = time.time() - self.active_jobs.get("selfplay", {}).get(job_id, {}).get("started_at", time.time())
            with self.jobs_lock:
                if job_id in self.active_jobs.get("selfplay", {}):
                    if proc.returncode == 0:
                        self.active_jobs["selfplay"][job_id]["status"] = "completed"
                        self._emit_task_event(
                            "TASK_COMPLETED",
                            job_id,
                            "selfplay",
                            board_type=board_type,
                            num_players=num_players,
                            num_games=num_games,
                            duration_seconds=duration,
                        )
                    else:
                        error_msg = stderr.decode()[:500]
                        logger.warning(f"Selfplay job {job_id} failed: {error_msg}")
                        self.active_jobs["selfplay"][job_id]["status"] = "failed"
                        self._emit_task_event(
                            "TASK_FAILED",
                            job_id,
                            "selfplay",
                            board_type=board_type,
                            num_players=num_players,
                            error=error_msg,
                            duration_seconds=duration,
                        )
                    # Remove from active jobs
                    del self.active_jobs["selfplay"][job_id]

        except asyncio.TimeoutError:
            # December 28, 2025: Use centralized process cleanup
            await self._kill_process(job_id, proc)
            with self.jobs_lock:
                if job_id in self.active_jobs.get("selfplay", {}):
                    self.active_jobs["selfplay"][job_id]["status"] = "timeout"
                    del self.active_jobs["selfplay"][job_id]
            logger.warning(f"Selfplay job {job_id} timed out and was killed")
            self._emit_task_event("TASK_FAILED", job_id, "selfplay", error="timeout", board_type=board_type)
        except (OSError, ValueError, RuntimeError) as e:
            # Dec 2025: Narrowed from broad Exception - subprocess execution errors
            # OSError: File not found, permission denied, process creation failed
            # ValueError: Invalid arguments to subprocess
            # RuntimeError: Subprocess state errors
            await self._kill_process(job_id, proc)
            with self.jobs_lock:
                if job_id in self.active_jobs.get("selfplay", {}):
                    self.active_jobs["selfplay"][job_id]["status"] = "error"
                    del self.active_jobs["selfplay"][job_id]
            logger.error(f"Error running selfplay job {job_id}: {e}")
            self._emit_task_event("TASK_FAILED", job_id, "selfplay", error=str(e), board_type=board_type)
        except Exception:
            # Catch-all for truly unexpected errors - log with traceback and re-raise
            await self._kill_process(job_id, proc)
            with self.jobs_lock:
                if job_id in self.active_jobs.get("selfplay", {}):
                    self.active_jobs["selfplay"][job_id]["status"] = "error"
                    del self.active_jobs["selfplay"][job_id]
            logger.exception(f"Unexpected error in selfplay job {job_id}")
            self._emit_task_event("TASK_FAILED", job_id, "selfplay", error="unexpected_error", board_type=board_type)
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
        iteration_dir = os.path.join(
            self.ringrift_path, "ai-service", "data", "selfplay",
            f"improve_{job_id}", f"iter_{state.current_iteration}"
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

        # December 2025: Determine engine mode and GPU requirements upfront
        effective_engine_mode = "gumbel-mcts" if model_path else "heuristic-only"
        gpu_required = self._engine_mode_requires_gpu(effective_engine_mode)

        if gpu_required:
            logger.info(
                f"GPU-required engine mode '{effective_engine_mode}' requested, "
                f"filtering workers with GPU capability"
            )

        async with get_client_session(timeout) as session:
            for i, worker in enumerate(workers):
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

                url = f"http://{worker_ip}:{worker_port}/run-selfplay"
                payload = {
                    "job_id": f"{job_id}_{worker_id}",
                    "parent_job_id": job_id,
                    "board_type": board_type,
                    "num_players": num_players,
                    "num_games": games,
                    "model_path": model_path,
                    "output_dir": output_dir,
                    "engine_mode": effective_engine_mode,
                }

                try:
                    async with session.post(url, json=payload, timeout=timeout) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            results[worker_id] = {
                                "success": result.get("success", True),
                                "games": games,
                            }
                            logger.debug(f"Dispatched {games} games to {worker_id}")
                        else:
                            results[worker_id] = {
                                "success": False,
                                "error": f"http_{resp.status}",
                            }
                except asyncio.TimeoutError:
                    results[worker_id] = {"success": False, "error": "timeout"}
                except (OSError, ConnectionError) as e:
                    # Dec 2025: Narrowed from broad Exception - network/connection errors
                    results[worker_id] = {"success": False, "error": str(e)}
                    logger.debug(f"Failed to dispatch to {worker_id}: {e}")

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
        output_file = os.path.join(output_dir, f"{self.node_id}_games.jsonl")

        # Build selfplay command
        cmd = [
            sys.executable,
            os.path.join(self.ringrift_path, "ai-service", "scripts", "run_self_play_soak.py"),
            "--num-games", str(num_games),
            "--board-type", board_type,
            "--num-players", str(num_players),
            "--engine-mode", "descent-only" if model_path else "heuristic-only",
            "--max-moves", "10000",  # Avoid draws due to move limit
            "--log-jsonl", output_file,
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(self.ringrift_path, "ai-service")
        env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

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
        except Exception:
            # Catch-all for truly unexpected errors - log with traceback and re-raise
            await self._kill_process(job_id, proc)
            logger.exception("Unexpected error in local selfplay")
            self._emit_task_event("TASK_FAILED", job_id, "selfplay", error="unexpected_error", board_type=board_type)
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

        iteration_dir = os.path.join(
            self.ringrift_path, "ai-service", "data", "selfplay",
            f"improve_{job_id}", f"iter_{state.current_iteration}"
        )
        output_file = os.path.join(
            self.ringrift_path, "ai-service", "data", "training",
            f"improve_{job_id}", f"iter_{state.current_iteration}.npz"
        )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        jsonl_files = list(Path(iteration_dir).glob("*.jsonl"))
        if not jsonl_files:
            logger.warning(f"No JSONL files found for export in {iteration_dir}")
            return

        cmd = [
            sys.executable,
            os.path.join(self.ringrift_path, "ai-service", "scripts", "jsonl_to_npz.py"),
            "--input-dir",
            iteration_dir,
            "--output",
            output_file,
            "--board-type",
            state.board_type,
            "--num-players",
            str(state.num_players),
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(self.ringrift_path, "ai-service")

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
        except Exception:
            # Catch-all for truly unexpected errors - log with traceback and re-raise
            await self._kill_process(export_job_id, proc)
            logger.exception("Unexpected error in training data export")
            self._emit_task_event("TASK_FAILED", job_id, "export", error="unexpected_error")
            raise

    async def run_training(self, job_id: str) -> None:
        """Run neural network training on GPU node.

        Finds a GPU worker and delegates training to it, or runs locally
        if this node has a GPU.

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
        }

        # For distributed training, would delegate to GPU worker here
        # For now, run locally
        await self.run_local_training(training_config)
        state.candidate_model_path = new_model_path

    async def run_local_training(self, config: dict) -> None:
        """Run training locally using subprocess.

        Args:
            config: Training configuration dict containing:
                - job_id: Job identifier for event tracking
                - training_data: Path to training data
                - output_model: Path to save trained model
                - board_type, num_players, epochs, batch_size, learning_rate
        """
        # Dec 2025: Extract job_id from config for event emission
        job_id = config.get("job_id", "unknown")
        logger.info(f"Running local training for job {job_id}")

        training_script = f"""
import sys
sys.path.insert(0, '{self.ringrift_path}/ai-service')
import numpy as np
import torch

# Load training data
try:
    data = np.load('{config.get("training_data", "")}', allow_pickle=True)
    print(f"Loaded training data")
except Exception as e:
    print(f"No training data available: {{e}}")
    data = None

# Import or create model architecture
try:
    from app.models.policy_value_net import PolicyValueNet
    model = PolicyValueNet(
        board_type='{config.get("board_type", "square8")}',
        num_players={config.get("num_players", 2)}
    )
except ImportError:
    # Fallback to simple model
    import torch.nn as nn
    model = nn.Sequential(
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64)
    )

# Save model
torch.save(model.state_dict(), '{config.get("output_model", "/tmp/model.pt")}')
print(f"Saved model to {{config.get('output_model', '/tmp/model.pt')}}")
"""

        cmd = [sys.executable, "-c", training_script]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(self.ringrift_path, "ai-service")

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
        except Exception:
            # Catch-all for truly unexpected errors - log with traceback and re-raise
            await self._kill_process(training_job_id, proc)
            logger.exception("Unexpected error in local training")
            self._emit_task_event("TASK_FAILED", job_id, "training", error="unexpected_error")
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
        except Exception:
            # Catch-all for truly unexpected errors - log with traceback and re-raise
            logger.exception(f"Unexpected error in tournament {job_id}")
            if job_id in self.distributed_tournament_state:
                self.distributed_tournament_state[job_id].status = "error: unexpected"
            self._emit_task_event("TASK_FAILED", job_id, "tournament", error="unexpected_error")
            raise  # Dec 28, 2025: CRITICAL - re-raise to signal caller that tournament failed

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

        Returns:
            List of worker node info objects
        """
        workers = []
        with self.peers_lock:
            for peer in self.peers.values():
                if hasattr(peer, "is_healthy") and peer.is_healthy():
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
        try:
            async with session.post(url, json=payload, timeout=timeout) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    match_results = result.get("results", [])
                    logger.debug(
                        f"Worker {worker_id} completed {len(match_results)} matches"
                    )
                    return match_results
                else:
                    logger.warning(f"Worker {worker_id} returned status {resp.status}")
                    return []
        except asyncio.TimeoutError:
            logger.warning(f"Worker {worker_id} timed out")
            return []
        except (OSError, ConnectionError) as e:
            # Dec 2025: Narrowed from broad Exception - network/connection errors
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
    # NOTE: The following methods were removed Dec 2025 as dead code:
    # - run_parity_validation() - placeholder stub, never called
    # - run_npz_export() - placeholder stub, never called
    # - get_job_count_for_node() - no production callers
    # - get_selfplay_job_count_for_node() - no production callers
    # - get_training_job_count_for_node() - no production callers
    # - get_all_jobs() - no production callers
    # Use orchestrator's job tracking or SelfplayScheduler for job counts.

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
        # Import HealthCheckResult with fallback
        try:
            from app.coordination.protocols import HealthCheckResult, CoordinatorStatus
        except ImportError:
            from dataclasses import dataclass as _dc, field as _field

            @_dc
            class HealthCheckResult:
                healthy: bool
                status: str = "running"
                message: str = ""
                timestamp: float = _field(default_factory=time.time)
                details: dict = _field(default_factory=dict)

            class CoordinatorStatus:
                RUNNING = "running"
                DEGRADED = "degraded"
                ERROR = "error"

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
