"""Job Orchestrator - Handles job spawning and process management.

January 2026: Created as part of P2POrchestrator decomposition.

Responsibilities:
- Job spawning with rate limiting
- Task isolation and subprocess management
- Spawn gating (resource checks)
- GPU job tracking
- Job result recording
- Process lifecycle management
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import time
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from scripts.p2p.orchestrators.base_orchestrator import BaseOrchestrator, HealthCheckResult, get_job_attr

if TYPE_CHECKING:
    from scripts.p2p_orchestrator import P2POrchestrator

logger = logging.getLogger(__name__)


class JobOrchestrator(BaseOrchestrator):
    """Orchestrator for job spawning and process management.

    This orchestrator handles all aspects of job management in the P2P cluster:
    - Rate-limited job spawning
    - Process spawn gating based on resources
    - GPU job tracking and result recording
    - Safe task creation with error isolation

    The actual job execution is delegated to JobManager, but this orchestrator
    provides spawn control, resource gating, and health monitoring.

    Usage:
        # In P2POrchestrator.__init__:
        self.jobs = JobOrchestrator(self)

        # Check if spawn allowed:
        can_spawn, reason = self.jobs.can_spawn_process("selfplay")

        # Record job result:
        self.jobs.record_gpu_job_result(success=True)
    """

    # Rate limiting constants
    SPAWN_RATE_LIMIT_PER_MINUTE = 30

    def __init__(self, p2p: "P2POrchestrator") -> None:
        """Initialize the job orchestrator.

        Args:
            p2p: The parent P2POrchestrator instance.
        """
        super().__init__(p2p)

        # Rate limiting state
        self._spawn_timestamps: list[float] = []

        # Job statistics
        self._jobs_spawned: int = 0
        self._jobs_completed: int = 0
        self._jobs_failed: int = 0
        self._gpu_jobs_active: int = 0

    @property
    def name(self) -> str:
        """Return the name of this orchestrator."""
        return "jobs"

    def health_check(self) -> HealthCheckResult:
        """Check the health of job orchestrator.

        Returns:
            HealthCheckResult with job status details.
        """
        try:
            issues = []

            # Check job manager availability
            job_manager = getattr(self._p2p, "job_manager", None)
            if job_manager is None:
                issues.append("JobManager not available")

            # Check job queue health
            if job_manager is not None and hasattr(job_manager, "get_queue_size"):
                queue_size = job_manager.get_queue_size()
                if queue_size > 100:
                    issues.append(f"Large job queue: {queue_size}")

            # Check failure rate
            total_completed = self._jobs_completed + self._jobs_failed
            if total_completed > 10:
                failure_rate = self._jobs_failed / total_completed
                if failure_rate > 0.5:
                    issues.append(f"High job failure rate: {failure_rate:.0%}")

            # Check rate limiting
            can_spawn, reason = self.check_spawn_rate_limit()
            if not can_spawn:
                issues.append(f"Rate limited: {reason}")

            healthy = len(issues) == 0
            message = "Jobs healthy" if healthy else "; ".join(issues)

            return HealthCheckResult(
                healthy=healthy,
                message=message,
                details={
                    "jobs_spawned": self._jobs_spawned,
                    "jobs_completed": self._jobs_completed,
                    "jobs_failed": self._jobs_failed,
                    "gpu_jobs_active": self._gpu_jobs_active,
                    "rate_limit_status": reason,
                    "issues": issues,
                },
            )
        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                message=f"Health check failed: {e}",
                details={"error": str(e)},
            )

    # =========================================================================
    # Rate Limiting
    # =========================================================================

    def check_spawn_rate_limit(self) -> tuple[bool, str]:
        """Check if we're within the spawn rate limit.

        Jan 29, 2026: Implementation moved from P2POrchestrator.

        Returns:
            (can_spawn, reason) - True if within rate limit
        """
        now = time.time()
        # Clean old timestamps (older than 60 seconds)
        self._spawn_timestamps = [t for t in self._spawn_timestamps if now - t < 60]

        current = len(self._spawn_timestamps)
        limit = self.SPAWN_RATE_LIMIT_PER_MINUTE

        if current >= limit:
            return False, f"Rate limit: {current}/{limit} spawns in last minute"

        return True, f"Rate OK: {current}/{limit}"

    def record_spawn(self) -> None:
        """Record a process spawn for rate limiting."""
        self._spawn_timestamps.append(time.time())
        self._jobs_spawned += 1

    # =========================================================================
    # Spawn Gating
    # =========================================================================

    def can_spawn_process(self, reason: str = "job") -> tuple[bool, str]:
        """Combined safeguard check before spawning any process.

        Jan 29, 2026: Implementation moved from P2POrchestrator._can_spawn_process().

        SAFEGUARD: Checks load average, rate limit, agent mode, backpressure,
        and graceful degradation.

        Args:
            reason: Description of why we want to spawn (for logging)

        Returns:
            (can_spawn, explanation) - True if all checks pass
        """
        # Check 1: Load average
        self_info = getattr(self._p2p, "self_info", None)
        if self_info is not None and hasattr(self_info, "check_load_average_safe"):
            load_ok, load_reason = self_info.check_load_average_safe()
            if not load_ok:
                self._log_info(f"BLOCKED spawn ({reason}): {load_reason}")
                return False, load_reason

        # Check 2: Rate limit
        rate_ok, rate_reason = self.check_spawn_rate_limit()
        if not rate_ok:
            self._log_info(f"BLOCKED spawn ({reason}): {rate_reason}")
            return False, rate_reason

        # Check 3: Agent mode - if coordinator is available and we're in agent mode,
        # we should not autonomously spawn jobs (let coordinator decide)
        agent_mode = getattr(self._p2p, "agent_mode", False)
        coordinator_available = getattr(self._p2p, "coordinator_available", False)
        if agent_mode and coordinator_available:
            msg = "Agent mode: deferring to coordinator"
            self._log_info(f"BLOCKED spawn ({reason}): {msg}")
            return False, msg

        # Check 4: Backpressure (new coordination) - if training queue is saturated,
        # don't spawn more selfplay jobs that would produce more data
        try:
            from app.coordination.backpressure import (
                QueueType,
                should_stop_production,
                should_throttle_production,
                get_throttle_factor,
            )
            HAS_NEW_COORDINATION = True
        except ImportError:
            HAS_NEW_COORDINATION = False

        if HAS_NEW_COORDINATION and "selfplay" in reason.lower():
            if should_stop_production(QueueType.TRAINING_DATA):
                msg = "Backpressure: training queue at STOP level"
                self._log_info(f"BLOCKED spawn ({reason}): {msg}")
                return False, msg
            if should_throttle_production(QueueType.TRAINING_DATA):
                throttle = get_throttle_factor(QueueType.TRAINING_DATA)
                import random
                if random.random() > throttle:
                    msg = f"Backpressure: throttled (factor={throttle:.2f})"
                    self._log_info(f"BLOCKED spawn ({reason}): {msg}")
                    return False, msg

        # Check 5: Graceful degradation - don't spawn under heavy resource pressure
        try:
            from app.coordination.resource_guard import (
                get_degradation_level,
                should_proceed_with_priority,
                OperationPriority,
            )
            HAS_RESOURCE_GUARD = True
        except ImportError:
            HAS_RESOURCE_GUARD = False
            get_degradation_level = None
            should_proceed_with_priority = None
            OperationPriority = None

        if HAS_RESOURCE_GUARD and get_degradation_level is not None:
            degradation = get_degradation_level()
            if degradation >= 4:  # CRITICAL - resources at/above limits
                msg = f"Graceful degradation: critical resource pressure (level {degradation})"
                self._log_info(f"BLOCKED spawn ({reason}): {msg}")
                return False, msg
            elif degradation >= 3:  # HEAVY - only critical ops proceed
                # Selfplay is NORMAL priority, blocked under heavy pressure
                if should_proceed_with_priority is not None and not should_proceed_with_priority(OperationPriority.NORMAL):
                    msg = f"Graceful degradation: heavy resource pressure (level {degradation})"
                    self._log_info(f"BLOCKED spawn ({reason}): {msg}")
                    return False, msg

        return True, "All safeguards passed"

    # =========================================================================
    # Task Management
    # =========================================================================

    def create_safe_task(
        self,
        coro: Coroutine,
        name: str,
        factory: Callable[[], Coroutine] | None = None,
    ) -> asyncio.Task:
        """Create a task wrapped with exception isolation and restart support.

        Jan 29, 2026: Wrapper for P2POrchestrator._create_safe_task().

        Args:
            coro: The coroutine to run
            name: Task name for logging
            factory: Optional callable that returns a new coroutine for restarts

        Returns:
            asyncio.Task wrapped with safe error handling
        """
        if hasattr(self._p2p, "_create_safe_task"):
            return self._p2p._create_safe_task(coro, name, factory)

        # Fallback: create task directly
        return asyncio.create_task(coro, name=name)

    # =========================================================================
    # GPU Job Tracking
    # =========================================================================

    def record_gpu_job_result(self, success: bool) -> None:
        """Record a GPU job result.

        Jan 29, 2026: Wrapper for P2POrchestrator._record_gpu_job_result().

        Args:
            success: Whether the job completed successfully
        """
        if success:
            self._jobs_completed += 1
        else:
            self._jobs_failed += 1

        if hasattr(self._p2p, "_record_gpu_job_result"):
            self._p2p._record_gpu_job_result(success)

    def update_gpu_job_count(self, delta: int) -> None:
        """Update the active GPU job count.

        Jan 29, 2026: Wrapper for P2POrchestrator._update_gpu_job_count().

        Args:
            delta: Change in job count (+1 for start, -1 for end)
        """
        self._gpu_jobs_active += delta

        if hasattr(self._p2p, "_update_gpu_job_count"):
            self._p2p._update_gpu_job_count(delta)

    def get_gpu_job_count(self) -> int:
        """Get the current active GPU job count.

        Returns:
            Number of active GPU jobs
        """
        return self._gpu_jobs_active

    # =========================================================================
    # Job Preferences
    # =========================================================================

    def get_node_job_preference(self, node_id: str) -> str:
        """Get the job type preference for a specific node.

        Jan 29, 2026: Wrapper for P2POrchestrator._get_node_job_preference().

        Args:
            node_id: The node ID to check

        Returns:
            Preferred job type (e.g., "selfplay", "training", "any")
        """
        if hasattr(self._p2p, "_get_node_job_preference"):
            return self._p2p._get_node_job_preference(node_id)
        return "any"

    # =========================================================================
    # Job Status
    # =========================================================================

    def get_job_status(self) -> dict[str, Any]:
        """Get current job status.

        Returns:
            Dict with job statistics and state.
        """
        job_manager = getattr(self._p2p, "job_manager", None)

        status = {
            "jobs_spawned": self._jobs_spawned,
            "jobs_completed": self._jobs_completed,
            "jobs_failed": self._jobs_failed,
            "gpu_jobs_active": self._gpu_jobs_active,
            "job_manager_available": job_manager is not None,
        }

        if job_manager is not None:
            if hasattr(job_manager, "get_active_jobs"):
                status["active_jobs"] = len(job_manager.get_active_jobs())
            if hasattr(job_manager, "get_queue_size"):
                status["queue_size"] = job_manager.get_queue_size()

        return status

    # =========================================================================
    # Work Discovery
    # =========================================================================

    def initialize_work_discovery_manager(self) -> bool:
        """Initialize WorkDiscoveryManager for multi-channel work discovery.

        Jan 29, 2026: Implementation moved from P2POrchestrator._initialize_work_discovery_manager().

        January 4, 2026: Phase 5 of P2P Cluster Resilience.
        Enables workers to find work through multiple channels:
        1. Leader work queue (fastest)
        2. Peer discovery (query other peers)
        3. Autonomous queue (from AutonomousQueueLoop)
        4. Direct selfplay (last resort)

        Returns:
            True if initialization succeeded, False otherwise.
        """
        try:
            from scripts.p2p.managers.work_discovery_manager import (
                WorkDiscoveryManager,
                WorkDiscoveryConfig,
                set_work_discovery_manager,
            )

            # Get callback functions from P2P orchestrator
            get_leader_id = lambda: getattr(self._p2p, "leader_id", None)

            claim_from_leader = getattr(self._p2p, "_claim_work_from_leader", None)

            # Jan 22, 2026: Use lock-free snapshot to prevent race conditions
            def get_alive_peers() -> list[str]:
                peer_snapshot = getattr(self._p2p, "_peer_snapshot", None)
                if peer_snapshot is None:
                    return []
                return [
                    p.node_id for p in peer_snapshot.get_snapshot().values() if p.is_alive()
                ]

            query_peer_work = getattr(self._p2p, "_query_peer_for_work", None)
            pop_autonomous_work = getattr(self._p2p, "_pop_autonomous_queue_work", None)
            create_direct_selfplay_work = getattr(self._p2p, "_create_direct_selfplay_work", None)

            # Create manager with callbacks to the orchestrator
            manager = WorkDiscoveryManager(
                # Channel 1: Leader
                get_leader_id=get_leader_id,
                claim_from_leader=claim_from_leader,
                # Channel 2: Peer discovery
                get_alive_peers=get_alive_peers,
                query_peer_work=query_peer_work,
                # Channel 3: Autonomous queue
                pop_autonomous_work=pop_autonomous_work,
                # Channel 4: Direct selfplay
                create_direct_selfplay_work=create_direct_selfplay_work,
                # Config from environment
                config=WorkDiscoveryConfig.from_env(),
            )

            # Set as singleton for WorkerPullLoop access
            set_work_discovery_manager(manager)
            self._log_info("WorkDiscoveryManager: initialized with 4 discovery channels")
            return True

        except ImportError as e:
            self._log_debug(f"WorkDiscoveryManager: not available: {e}")
            return False
        except Exception as e:  # noqa: BLE001
            self._log_warning(f"WorkDiscoveryManager: initialization failed: {e}")
            return False

    # =========================================================================
    # Job Queries
    # =========================================================================

    def get_pending_jobs_for_node(self, node_id: str) -> int:
        """Get count of pending/running jobs assigned to a specific node.

        Jan 29, 2026: Implementation moved from P2POrchestrator._get_pending_jobs_for_node().

        Used by PredictiveScalingLoop to skip nodes that already have
        work pending, avoiding over-allocation.

        Args:
            node_id: The node identifier to check.

        Returns:
            Number of pending/running jobs for this node.
        """
        try:
            job_manager = getattr(self._p2p, "job_manager", None)
            if job_manager is None:
                return 0
            # Count jobs that are pending or running for this node
            if hasattr(job_manager, "get_jobs_for_node"):
                jobs = job_manager.get_jobs_for_node(node_id)
                return len([j for j in jobs if get_job_attr(j, "status") in ("pending", "running", "claimed")])
            return 0
        except Exception as e:  # noqa: BLE001
            self._log_debug(f"Failed to get pending jobs for {node_id}: {e}")
            return 0

    # =========================================================================
    # Job Spawning
    # =========================================================================

    def spawn_and_track_job(
        self,
        job_id: str,
        job_type: Any,
        board_type: str,
        num_players: int,
        engine_mode: str,
        cmd: list[str],
        output_dir: Any,
        log_filename: str = "run.log",
        cuda_visible_devices: str | None = None,
        extra_env: dict[str, str] | None = None,
        safeguard_reason: str | None = None,
    ) -> tuple[Any, Any] | None:
        """Spawn a subprocess job and track it in local_jobs.

        Jan 29, 2026: Implementation moved from P2POrchestrator._spawn_and_track_job().

        Args:
            job_id: Unique job identifier
            job_type: Type of job (SELFPLAY, GPU_SELFPLAY, etc.)
            board_type: Board type (hex8, square8, etc.)
            num_players: Number of players
            engine_mode: Engine mode for the job
            cmd: Command to execute
            output_dir: Directory for output files
            log_filename: Name of log file in output_dir
            cuda_visible_devices: CUDA_VISIBLE_DEVICES value (None = inherit, "" = disable)
            extra_env: Additional environment variables
            safeguard_reason: Reason for safeguard check (default: job_type-board_type-Np)

        Returns:
            Tuple of (ClusterJob, Popen) if successful, None if blocked or failed
        """
        import os
        import subprocess

        # Get job_type value (handle enum or string)
        job_type_val = job_type.value if hasattr(job_type, "value") else str(job_type)

        # Build safeguard check reason
        if safeguard_reason is None:
            safeguard_reason = f"{job_type_val}-{board_type}-{num_players}p"

        # SAFEGUARD: Final check before spawning
        can_spawn, spawn_reason = self.can_spawn_process(safeguard_reason)
        if not can_spawn:
            self._log_info(f"BLOCKED {job_type_val} spawn: {spawn_reason}")
            return None

        # Build environment
        env = os.environ.copy()

        # Get AI service path from P2P
        ai_service_path = ""
        if hasattr(self._p2p, "_get_ai_service_path"):
            ai_service_path = self._p2p._get_ai_service_path()
        elif hasattr(self._p2p, "ringrift_path"):
            ai_service_path = str(self._p2p.ringrift_path / "ai-service")

        env["PYTHONPATH"] = ai_service_path
        env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"
        env["RINGRIFT_JOB_ORIGIN"] = "p2p_orchestrator"

        # Handle CUDA_VISIBLE_DEVICES
        if cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices).strip()

        # Apply extra environment variables
        if extra_env:
            env.update(extra_env)

        # Ensure output directory exists
        from pathlib import Path
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / log_filename

        # Get ringrift_path from P2P
        ringrift_path = getattr(self._p2p, "ringrift_path", None)

        # Spawn subprocess
        try:
            log_handle = open(log_path, "a")  # noqa: SIM115
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=ringrift_path,
                )
                self.record_spawn()
            finally:
                log_handle.close()
        except (OSError, subprocess.SubprocessError) as e:
            self._log_error(f"Failed to spawn {job_type_val}: {e}")
            return None

        # Create ClusterJob - get class from P2P
        node_id = getattr(self._p2p, "node_id", "unknown")

        # Try to get ClusterJob class
        ClusterJob = None
        if hasattr(self._p2p, "ClusterJob"):
            ClusterJob = self._p2p.ClusterJob
        else:
            # Import from scripts.p2p if available
            try:
                from scripts.p2p.job_types import ClusterJob
            except ImportError:
                # Fallback: create a simple namedtuple-like dict
                pass

        if ClusterJob is not None:
            job = ClusterJob(
                job_id=job_id,
                job_type=job_type,
                node_id=node_id,
                board_type=board_type,
                num_players=num_players,
                engine_mode=engine_mode,
                pid=proc.pid,
                started_at=time.time(),
                status="running",
            )
        else:
            # Simple dict fallback
            job = {
                "job_id": job_id,
                "job_type": job_type_val,
                "node_id": node_id,
                "board_type": board_type,
                "num_players": num_players,
                "engine_mode": engine_mode,
                "pid": proc.pid,
                "started_at": time.time(),
                "status": "running",
            }

        # Track in local_jobs
        jobs_lock = getattr(self._p2p, "jobs_lock", None)
        local_jobs = getattr(self._p2p, "local_jobs", {})

        if jobs_lock is not None:
            with jobs_lock:
                local_jobs[job_id] = job
        else:
            local_jobs[job_id] = job

        self._log_info(f"Started {job_type_val} job {job_id} (PID {proc.pid})")

        # Save state
        if hasattr(self._p2p, "_save_state"):
            self._p2p._save_state()

        # Track via JobOrchestrationManager
        job_orchestration = getattr(self._p2p, "job_orchestration", None)
        if job_orchestration is not None:
            if hasattr(job_orchestration, "record_job_started"):
                job_orchestration.record_job_started(job_type_val)

        return job, proc

    # =========================================================================
    # Local Job Counting
    # =========================================================================

    def count_local_jobs(self) -> tuple[int, int]:
        """Count running selfplay and training jobs on this node.

        Jan 29, 2026: Implementation moved from P2POrchestrator._count_local_jobs().

        Returns:
            Tuple of (selfplay_count, training_count).
        """
        def _pid_alive(pid: int) -> bool:
            try:
                os.kill(pid, 0)
                return True
            except ProcessLookupError:
                return False
            except PermissionError:
                return True
            except AttributeError:
                return False

        # Primary source of truth: jobs we started and are tracking.
        selfplay_pids: set[str] = set()
        training_pids: set[str] = set()

        stale_job_ids: list[str] = []

        # Get JobType enum - try from P2P or import directly
        JobType = None
        try:
            from scripts.p2p.job_types import JobType
        except ImportError:
            pass

        try:
            # Jan 31, 2026: Use lock-free JobSnapshot for reads to avoid blocking event loop
            # JobSnapshot uses copy-on-write pattern - get_snapshot() returns instantly
            job_snapshot_obj = getattr(self._p2p, "_job_snapshot", None)
            jobs_lock = None  # Only used for stale job cleanup in fallback path
            local_jobs = {}

            if job_snapshot_obj is not None:
                # Lock-free path: get immutable snapshot
                snapshot_dict = job_snapshot_obj.get_snapshot()
                # Convert dict values to (job_id, job) tuples for processing
                jobs_snapshot = [(job_id, job_dict) for job_id, job_dict in snapshot_dict.items()]
            else:
                # Fallback: use old lock-based approach if JobSnapshot not available
                jobs_lock = getattr(self._p2p, "jobs_lock", None)
                local_jobs = getattr(self._p2p, "local_jobs", {})
                if jobs_lock is not None:
                    with jobs_lock:
                        jobs_snapshot = list(local_jobs.items())
                else:
                    jobs_snapshot = list(local_jobs.items())

            for job_id, job in jobs_snapshot:
                if get_job_attr(job, "status") != "running":
                    continue
                pid = int(get_job_attr(job, "pid", 0) or 0)
                if pid <= 0:
                    continue
                if not _pid_alive(pid):
                    stale_job_ids.append(job_id)
                    continue

                job_type = get_job_attr(job, "job_type")
                if JobType is not None:
                    if job_type in (JobType.SELFPLAY, JobType.GPU_SELFPLAY, JobType.HYBRID_SELFPLAY, JobType.CPU_SELFPLAY, JobType.GUMBEL_SELFPLAY):
                        selfplay_pids.add(str(pid))
                    elif job_type == JobType.TRAINING:
                        training_pids.add(str(pid))
                else:
                    # Fallback: check string representation
                    job_type_str = str(job_type).lower()
                    if "selfplay" in job_type_str:
                        selfplay_pids.add(str(pid))
                    elif "training" in job_type_str:
                        training_pids.add(str(pid))

            if stale_job_ids and jobs_lock is not None:
                with jobs_lock:
                    for job_id in stale_job_ids:
                        local_jobs.pop(job_id, None)
        except (ValueError, AttributeError):
            pass

        # Secondary check: best-effort process scan for untracked jobs (e.g. manual runs).
        # IMPORTANT: never return (0,0) just because `pgrep` is missing or fails;
        # that can cause the leader to spawn runaway selfplay processes until disk fills.
        try:
            if shutil.which("pgrep"):
                # Jan 12, 2026: Helper to filter out non-Python processes
                # SSH processes and shell wrappers (zsh, bash) with "selfplay" in their args
                # were being counted as local jobs - only count actual Python processes
                def _get_excluded_pids() -> set[str]:
                    """Get PIDs of SSH and shell processes (to exclude from local job counts)."""
                    excluded_pids: set[str] = set()
                    # Exclude SSH processes (dispatchers to remote nodes)
                    for pattern in ("^ssh", "ssh "):
                        try:
                            out = subprocess.run(
                                ["pgrep", "-f", pattern],
                                capture_output=True,
                                text=True,
                                timeout=5,
                            )
                            if out.returncode == 0 and out.stdout.strip():
                                excluded_pids.update(out.stdout.strip().split())
                        except (subprocess.SubprocessError, subprocess.TimeoutExpired, OSError):
                            pass
                    # Exclude shell processes (Claude wrappers that contain "selfplay" in args)
                    for shell_pattern in ("/bin/zsh", "/bin/bash", "/bin/sh"):
                        try:
                            out = subprocess.run(
                                ["pgrep", "-f", shell_pattern],
                                capture_output=True,
                                text=True,
                                timeout=5,
                            )
                            if out.returncode == 0 and out.stdout.strip():
                                excluded_pids.update(out.stdout.strip().split())
                        except (subprocess.SubprocessError, subprocess.TimeoutExpired, OSError):
                            pass
                    return excluded_pids

                excluded_pids = _get_excluded_pids()

                # December 2025: Added selfplay.py pattern - the current unified selfplay entry point
                # December 2025: Added gumbel_selfplay and SelfplayRunner patterns for module invocations
                for pattern in (
                    "selfplay.py",
                    "run_self_play_soak.py",
                    "run_gpu_selfplay.py",
                    "run_hybrid_selfplay.py",
                    "gumbel_selfplay",  # screen session name
                    "SelfplayRunner",   # class-based invocation
                    "selfplay_runner",  # module invocation
                    "-m app.training.selfplay",  # module mode
                ):
                    out = subprocess.run(
                        ["pgrep", "-f", pattern],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if out.returncode == 0 and out.stdout.strip():
                        # Jan 12, 2026: Filter out excluded PIDs (SSH, shells) - not local jobs
                        pids = [p for p in out.stdout.strip().split() if p and p not in excluded_pids]
                        selfplay_pids.update(pids)

                for pattern in ("train_", "train.py", "-m app.training.train"):
                    out = subprocess.run(
                        ["pgrep", "-f", pattern],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if out.returncode == 0 and out.stdout.strip():
                        # Jan 12, 2026: Filter out excluded PIDs (SSH, shells)
                        pids = [p for p in out.stdout.strip().split() if p and p not in excluded_pids]
                        training_pids.update(pids)
        except (subprocess.SubprocessError, subprocess.TimeoutExpired, OSError, KeyError, IndexError, AttributeError, ImportError):
            pass

        return len(selfplay_pids), len(training_pids)

    # =========================================================================
    # Stale Process Cleanup
    # =========================================================================

    def cleanup_stale_processes(self) -> int:
        """Kill processes that have been running too long.

        Jan 29, 2026: Implementation moved from P2POrchestrator._cleanup_stale_processes().

        Scans for known process patterns (tournaments, gauntlets, selfplay)
        and kills any that exceed their maximum runtime threshold.

        Returns:
            Number of processes killed.
        """
        if not shutil.which("pgrep") or not shutil.which("ps"):
            return 0

        killed_count = 0

        # Runtime thresholds (in seconds)
        MAX_TOURNAMENT_RUNTIME = 4 * 3600  # 4 hours
        MAX_GAUNTLET_RUNTIME = 6 * 3600    # 6 hours
        MAX_SELFPLAY_RUNTIME = 8 * 3600    # 8 hours
        MAX_TRAINING_RUNTIME = 24 * 3600   # 24 hours

        # Map patterns to their max runtimes
        # December 2025: Added selfplay.py - the current unified selfplay entry point
        pattern_max_runtime = {
            "run_model_elo_tournament.py": MAX_TOURNAMENT_RUNTIME,
            "run_gauntlet.py": MAX_GAUNTLET_RUNTIME,
            "selfplay.py": MAX_SELFPLAY_RUNTIME,  # Unified selfplay script
            "run_self_play_soak.py": MAX_SELFPLAY_RUNTIME,
            "run_gpu_selfplay.py": MAX_SELFPLAY_RUNTIME,
            "run_hybrid_selfplay.py": MAX_SELFPLAY_RUNTIME,
            "train_nnue.py": MAX_TRAINING_RUNTIME,
            "train.py": MAX_TRAINING_RUNTIME,
        }

        for pattern, max_runtime in pattern_max_runtime.items():
            try:
                # Get PIDs matching the pattern
                pgrep_result = subprocess.run(
                    ["pgrep", "-f", pattern],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if pgrep_result.returncode != 0 or not pgrep_result.stdout.strip():
                    continue

                pids = [p.strip() for p in pgrep_result.stdout.strip().split() if p.strip()]

                for pid in pids:
                    try:
                        # Get process start time using ps
                        ps_result = subprocess.run(
                            ["ps", "-o", "etimes=", "-p", pid],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        if ps_result.returncode != 0:
                            continue

                        elapsed_seconds = int(ps_result.stdout.strip())

                        if elapsed_seconds > max_runtime:
                            # Process has exceeded max runtime - kill it
                            self._log_warning(
                                f"Killing stale process {pid} ({pattern}): "
                                f"running for {elapsed_seconds/3600:.1f}h, max={max_runtime/3600:.1f}h"
                            )
                            subprocess.run(
                                ["kill", "-9", pid],
                                capture_output=True,
                                timeout=5,
                            )
                            killed_count += 1

                            # Send alert via notifier if available
                            notifier = getattr(self._p2p, "notifier", None)
                            if notifier is not None:
                                asyncio.create_task(
                                    notifier.send(
                                        title="Stale Process Killed",
                                        message=f"Killed {pattern} (PID {pid}) after {elapsed_seconds/3600:.1f} hours",
                                        level="warning",
                                        node_id=self.node_id,
                                    )
                                )

                    except (ValueError, subprocess.TimeoutExpired):
                        continue

            except Exception as e:  # noqa: BLE001
                self._log_debug(f"Error checking pattern {pattern}: {e}")
                continue

        if killed_count > 0:
            self._log_info(f"Stale process cleanup: killed {killed_count} processes")

        return killed_count

    async def auto_rebalance_from_work_queue(self) -> int:
        """Auto-rebalance: assign queued work to idle GPU nodes.

        Jan 29, 2026: Moved from P2POrchestrator._auto_rebalance_from_work_queue().

        When idle GPU-heavy nodes are detected, check the work queue for pending
        high-priority work and dispatch it. This ensures queued work gets done
        before falling back to selfplay auto-scaling.

        Returns:
            Number of work items dispatched
        """
        import time
        from app.coordination.work_queue import get_work_queue

        GPU_IDLE_THRESHOLD = 10.0  # Node is idle if GPU < 10%
        MIN_IDLE_TIME = 60  # Seconds of idle before assigning work
        GPU_HEAVY_TAGS = ['gh200', 'h100', 'h200', 'a100', '4090', '5090']

        p2p = self._p2p
        dispatched = 0
        now = time.time()

        # Rate limit rebalancing (once per minute)
        last_rebalance = getattr(p2p, "_last_work_queue_rebalance", 0)
        if now - last_rebalance < 60:
            return 0

        # Check if work queue is available
        wq = get_work_queue()
        if wq is None:
            return 0

        # Get queue status
        queue_status = wq.get_queue_status()
        pending_count = queue_status.get("by_status", {}).get("pending", 0)
        if pending_count == 0:
            return 0  # No work to dispatch

        # Find idle GPU-heavy nodes
        idle_nodes = []

        with p2p.peers_lock:
            peers_snapshot = list(p2p.peers.values())

        for peer in peers_snapshot:
            if not peer.is_alive() or peer.retired:
                continue

            has_gpu = bool(getattr(peer, "has_gpu", False))
            if not has_gpu:
                continue

            gpu_name = (getattr(peer, "gpu_name", "") or "").upper()
            is_gpu_heavy = any(tag.upper() in gpu_name for tag in GPU_HEAVY_TAGS)
            if not is_gpu_heavy:
                continue

            gpu_percent = float(getattr(peer, "gpu_percent", 0) or 0)
            training_jobs = int(getattr(peer, "training_jobs", 0) or 0)
            selfplay_jobs = int(getattr(peer, "selfplay_jobs", 0) or 0)

            # Skip if already busy
            if training_jobs > 0:
                continue

            # Check if truly idle
            if gpu_percent < GPU_IDLE_THRESHOLD:
                # Track how long it's been idle
                idle_key = f"_wq_idle_since_{peer.node_id}"
                idle_since = getattr(p2p, idle_key, 0)
                if idle_since == 0:
                    setattr(p2p, idle_key, now)
                elif now - idle_since > MIN_IDLE_TIME:
                    # Get allowed work types for this node
                    try:
                        from app.coordination.node_policies import get_policy_manager
                        pm = get_policy_manager()
                        allowed = list(pm.get_allowed_work_types(peer.node_id))
                    except ImportError:
                        allowed = ["training", "gpu_cmaes", "tournament", "selfplay"]

                    idle_nodes.append({
                        "node_id": peer.node_id,
                        "peer": peer,
                        "gpu_percent": gpu_percent,
                        "gpu_name": gpu_name,
                        "allowed": allowed,
                        "selfplay_jobs": selfplay_jobs,
                    })
            else:
                # Not idle, reset timer
                idle_key = f"_wq_idle_since_{peer.node_id}"
                setattr(p2p, idle_key, 0)

        # Dispatch work to idle nodes
        for node_info in idle_nodes[:5]:  # Max 5 nodes per cycle
            node_id = node_info["node_id"]
            allowed = node_info["allowed"]

            # Try to claim work for this node using Raft or SQLite based on consensus mode
            work_item = p2p.claim_work_distributed(node_id, allowed)
            if work_item is None:
                continue

            # Get work_type - may be string or WorkType enum
            work_type_str = work_item.get("work_type", "unknown")
            if hasattr(work_type_str, "value"):
                work_type_str = work_type_str.value
            work_id = work_item.get("work_id", "unknown")

            print(
                f"[P2P] Work queue rebalance: {node_id} idle at {node_info['gpu_percent']:.0f}% GPU, "
                f"assigning {work_type_str} work ({work_id})"
            )

            # Dispatch work to the node
            success = await p2p.job_coordination_manager.dispatch_queued_work(node_info["peer"], work_item)
            if success:
                # Mark work as started using distributed method for Raft consistency
                p2p.start_work_distributed(work_id)
                dispatched += 1
                # Reset idle timer since we assigned work
                idle_key = f"_wq_idle_since_{node_id}"
                setattr(p2p, idle_key, 0)
            else:
                # Failed to dispatch, reset work status for retry
                p2p.fail_work_distributed(work_id, "dispatch_failed")

        if dispatched > 0:
            p2p._last_work_queue_rebalance = now
            self._log_info(f"Work queue rebalance: dispatched {dispatched} work item(s) to idle nodes")

        return dispatched
