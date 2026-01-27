"""JobLifecycleManager: Stuck job detection and termination.

January 2026: Extracted from p2p_orchestrator.py for better modularity.
Handles detection and termination of stuck training/selfplay jobs.

This manager handles:
- Detecting stuck training jobs with no progress
- Detecting stuck GPU selfplay jobs with 0% GPU utilization
- Remote killing of stuck jobs via P2P
- Local stuck process cleanup
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from aiohttp import ClientTimeout
    from scripts.p2p.models import PeerInfo

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class JobLifecycleConfig:
    """Configuration for JobLifecycleManager.

    Attributes:
        training_stuck_threshold: Seconds before training job considered stuck
        selfplay_stuck_threshold: Seconds before selfplay job considered stuck
        local_check_interval: Seconds between local stuck job checks
        orphan_check_interval: Seconds between orphan process checks
        kill_timeout: Timeout for kill commands (seconds)
    """

    training_stuck_threshold: float = 600.0  # 10 minutes
    selfplay_stuck_threshold: float = 900.0  # 15 minutes
    local_check_interval: float = 300.0  # 5 minutes
    orphan_check_interval: float = 3600.0  # 1 hour
    kill_timeout: float = 10.0


@dataclass
class JobLifecycleStats:
    """Statistics for JobLifecycleManager operations."""

    stuck_jobs_detected: int = 0
    stuck_jobs_killed: int = 0
    local_stuck_processes_killed: int = 0
    remote_kills_attempted: int = 0
    remote_kills_succeeded: int = 0
    orphans_detected: int = 0
    last_check_time: float = 0.0
    last_local_stuck_check: float = 0.0
    last_orphan_check: float = 0.0


# ============================================================================
# Singleton management
# ============================================================================

_instance: JobLifecycleManager | None = None


def get_job_lifecycle_manager() -> JobLifecycleManager | None:
    """Get the singleton JobLifecycleManager instance."""
    return _instance


def set_job_lifecycle_manager(manager: JobLifecycleManager) -> None:
    """Set the singleton JobLifecycleManager instance."""
    global _instance
    _instance = manager


def reset_job_lifecycle_manager() -> None:
    """Reset the singleton JobLifecycleManager instance (for testing)."""
    global _instance
    _instance = None


def create_job_lifecycle_manager(
    config: JobLifecycleConfig | None = None,
    orchestrator: Any | None = None,
) -> JobLifecycleManager:
    """Create and register a JobLifecycleManager instance.

    Args:
        config: Optional configuration
        orchestrator: P2P orchestrator reference (for callbacks)

    Returns:
        The created JobLifecycleManager instance
    """
    manager = JobLifecycleManager(config=config, orchestrator=orchestrator)
    set_job_lifecycle_manager(manager)
    return manager


# ============================================================================
# JobLifecycleManager
# ============================================================================


class JobLifecycleManager:
    """Manager for stuck job detection and termination.

    This class handles:
    - Detecting stuck training jobs with no progress
    - Detecting stuck GPU selfplay jobs with 0% GPU utilization
    - Remote killing of stuck jobs via P2P
    - Local stuck process cleanup
    """

    def __init__(
        self,
        config: JobLifecycleConfig | None = None,
        orchestrator: Any | None = None,
    ):
        """Initialize JobLifecycleManager.

        Args:
            config: Configuration for the manager
            orchestrator: P2P orchestrator reference (for callbacks)
        """
        self.config = config or JobLifecycleConfig()
        self._orchestrator = orchestrator
        self._stats = JobLifecycleStats()
        self._local_last_gpu_active = 0.0

    @property
    def stats(self) -> JobLifecycleStats:
        """Get current statistics."""
        return self._stats

    def set_orchestrator(self, orchestrator: Any) -> None:
        """Set the P2P orchestrator reference.

        Called during orchestrator initialization.
        """
        self._orchestrator = orchestrator

    def _get_node_id(self) -> str:
        """Get this node's ID from orchestrator."""
        return getattr(self._orchestrator, "node_id", "unknown")

    def _get_training_jobs_snapshot(self) -> list[Any]:
        """Get a snapshot of training jobs under lock."""
        training_lock = getattr(self._orchestrator, "training_lock", None)
        training_jobs = getattr(self._orchestrator, "training_jobs", {})

        if training_lock:
            with training_lock:
                return list(training_jobs.values())
        return list(training_jobs.values())

    def _get_peers_snapshot(self) -> list[Any]:
        """Get a snapshot of peers under lock."""
        peers_lock = getattr(self._orchestrator, "peers_lock", None)
        peers = getattr(self._orchestrator, "peers", {})

        if peers_lock:
            with peers_lock:
                return list(peers.values())
        return list(peers.values())

    def _get_peer_by_id(self, peer_id: str) -> Any | None:
        """Get a peer by ID under lock."""
        peers_lock = getattr(self._orchestrator, "peers_lock", None)
        peers = getattr(self._orchestrator, "peers", {})

        if peers_lock:
            with peers_lock:
                return peers.get(peer_id)
        return peers.get(peer_id)

    def _get_self_info(self) -> Any | None:
        """Get self info from orchestrator."""
        return getattr(self._orchestrator, "self_info", None)

    def _get_local_jobs(self) -> dict:
        """Get local jobs dict."""
        return getattr(self._orchestrator, "local_jobs", {})

    def _get_jobs_lock(self) -> Any | None:
        """Get jobs lock from orchestrator."""
        return getattr(self._orchestrator, "jobs_lock", None)

    def _url_for_peer(self, peer: Any, endpoint: str) -> str:
        """Build URL for a peer endpoint."""
        if hasattr(self._orchestrator, "_url_for_peer"):
            return self._orchestrator._url_for_peer(peer, endpoint)
        # Fallback
        host = getattr(peer, "host", "localhost")
        port = getattr(peer, "port", 8770)
        return f"http://{host}:{port}{endpoint}"

    def _auth_headers(self) -> dict[str, str]:
        """Get authentication headers from orchestrator."""
        if hasattr(self._orchestrator, "_auth_headers"):
            return self._orchestrator._auth_headers()
        return {}

    async def _get_notifier(self) -> Any | None:
        """Get notifier from orchestrator."""
        return getattr(self._orchestrator, "notifier", None)

    def _run_subprocess_sync(
        self, cmd: list[str], timeout: float
    ) -> tuple[int, str, str]:
        """Run a subprocess synchronously.

        Returns:
            Tuple of (returncode, stdout, stderr)
        """
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "timeout"
        except Exception as e:
            return -1, "", str(e)

    # ========================================================================
    # Stuck job detection and killing
    # ========================================================================

    async def check_and_kill_stuck_jobs(self) -> int:
        """Detect and terminate stuck training/selfplay jobs.

        A job is considered stuck if:
        - Training: No log output for 10+ minutes while process still running
        - Selfplay: No new games generated for 15+ minutes

        Returns:
            Number of stuck jobs terminated
        """
        killed = 0
        now = time.time()
        config = self.config
        node_id = self._get_node_id()

        # Check training jobs
        training_snapshot = self._get_training_jobs_snapshot()

        for job in training_snapshot:
            if job.status != "running":
                continue
            started = getattr(job, "started_at", 0) or 0
            last_progress = getattr(job, "last_progress_time", started) or started
            if (
                now - last_progress > config.training_stuck_threshold
                and now - started > config.training_stuck_threshold
            ):
                self._stats.stuck_jobs_detected += 1
                logger.info(
                    f"STUCK DETECTED: Training job {job.job_id} on {job.target_node} "
                    f"- no progress for {int((now - last_progress)/60)}min"
                )

                # Try to kill the process on the target node
                target_node = job.target_node
                if target_node and target_node != node_id:
                    await self._remote_kill_stuck_job(target_node, job.job_id, "training")
                else:
                    # Local kill
                    try:
                        subprocess.run(
                            ["pkill", "-9", "-f", f"train.*{job.job_id}"],
                            timeout=5,
                            capture_output=True,
                        )
                    except (
                        subprocess.SubprocessError,
                        subprocess.TimeoutExpired,
                        OSError,
                    ):
                        pass

                job.status = "failed"
                job.error_message = "Killed: no progress detected"
                job.completed_at = now
                killed += 1
                self._stats.stuck_jobs_killed += 1
                logger.info(f"Killed stuck training job {job.job_id}")

                # ALERTING: Notify when stuck job is killed
                notifier = await self._get_notifier()
                if notifier:
                    asyncio.create_task(
                        notifier.send(
                            title="Stuck Job Killed",
                            message=f"Training job {job.job_id} killed after no progress "
                            f"for {int((now - last_progress)/60)}min",
                            level="warning",
                            fields={
                                "Job ID": job.job_id,
                                "Type": job.job_type,
                                "Node": job.target_node or "local",
                                "Config": f"{job.board_type}_{job.num_players}p",
                                "Stuck For": f"{int((now - last_progress)/60)} minutes",
                            },
                            node_id=node_id,
                        )
                    )

        # Check for GPU nodes with 0% GPU but running GPU jobs
        peers_snapshot = self._get_peers_snapshot()

        for peer in peers_snapshot:
            if not peer.is_alive():
                continue
            gpu_percent = float(getattr(peer, "gpu_percent", 0) or 0)
            selfplay_jobs = int(getattr(peer, "selfplay_jobs", 0) or 0)
            has_gpu = bool(getattr(peer, "has_gpu", False))

            # Check for stuck GPU selfplay (has GPU, jobs running, but 0% GPU util)
            if has_gpu and selfplay_jobs > 0 and gpu_percent == 0:
                last_gpu_active = getattr(peer, "_last_gpu_active_time", 0)
                if last_gpu_active == 0:
                    peer._last_gpu_active_time = now
                elif now - last_gpu_active > config.selfplay_stuck_threshold:
                    logger.info(
                        f"STUCK DETECTED: {peer.node_id} has {selfplay_jobs} jobs "
                        f"but 0% GPU for {int((now - last_gpu_active)/60)}min"
                    )
                    # Don't auto-kill selfplay, just log - might be CPU selfplay
            elif has_gpu and gpu_percent > 5:
                peer._last_gpu_active_time = now

        self._stats.last_check_time = now
        if killed > 0:
            logger.info(f"Self-healing: killed {killed} stuck job(s)")
        return killed

    async def check_local_stuck_jobs(self) -> int:
        """DECENTRALIZED: Detect and kill stuck processes on THIS node only.

        Runs on ALL nodes (not just leader) to ensure each node can self-heal
        even when there's no functioning leader in the cluster.

        Detects:
        - GPU selfplay processes with 0% GPU utilization for too long
        - Training processes that haven't made progress
        - Orphaned processes that aren't tracked in local_jobs

        Returns:
            Number of stuck processes terminated
        """
        killed = 0
        now = time.time()
        config = self.config

        # Only check periodically to avoid excessive process scanning
        if now - self._stats.last_local_stuck_check < config.local_check_interval:
            return 0
        self._stats.last_local_stuck_check = now

        # Check if local GPU is at 0% but we have running GPU selfplay jobs
        self_info = self._get_self_info()
        if not self_info:
            return 0

        gpu_percent = float(getattr(self_info, "gpu_percent", 0) or 0)
        selfplay_jobs = int(getattr(self_info, "selfplay_jobs", 0) or 0)
        has_gpu = bool(getattr(self_info, "has_gpu", False))

        if has_gpu and selfplay_jobs > 0 and gpu_percent < 5:
            if self._local_last_gpu_active == 0:
                self._local_last_gpu_active = now
            elif now - self._local_last_gpu_active > config.selfplay_stuck_threshold:
                logger.info(
                    f"LOCAL STUCK: {selfplay_jobs} selfplay jobs but "
                    f"{gpu_percent:.0f}% GPU for "
                    f"{int((now - self._local_last_gpu_active)/60)}min"
                )
                # Kill all local GPU selfplay processes and let them restart
                try:
                    returncode, _, _ = await asyncio.to_thread(
                        self._run_subprocess_sync,
                        ["pkill", "-9", "-f", "gpu_selfplay"],
                        10,
                    )
                    if returncode == 0:
                        killed += 1
                        self._stats.local_stuck_processes_killed += 1
                        logger.info("LOCAL: Killed stuck GPU selfplay processes")
                        # Clear job tracking so they restart
                        jobs_lock = self._get_jobs_lock()
                        local_jobs = self._get_local_jobs()
                        if jobs_lock and local_jobs:
                            with jobs_lock:
                                gpu_jobs = [
                                    jid
                                    for jid, job in local_jobs.items()
                                    if "gpu" in str(getattr(job, "job_type", "")).lower()
                                ]
                                for jid in gpu_jobs:
                                    del local_jobs[jid]
                        self._local_last_gpu_active = now
                except Exception as e:  # noqa: BLE001
                    logger.info(f"LOCAL: Failed to kill stuck processes: {e}")
        elif has_gpu and gpu_percent >= 5:
            self._local_last_gpu_active = now

        # Check for orphaned selfplay processes
        await self._check_orphaned_processes(now)

        if killed > 0:
            logger.info(f"LOCAL self-healing: terminated {killed} stuck process(es)")
        return killed

    async def _check_orphaned_processes(self, now: float) -> None:
        """Check for orphaned selfplay processes (processes running but not tracked)."""
        config = self.config

        if now - self._stats.last_orphan_check < config.orphan_check_interval:
            return

        try:

            def _count_local_selfplay_processes() -> int:
                """Count local selfplay processes, excluding SSH dispatchers and shells."""
                try:
                    # Get all selfplay-related PIDs
                    result = subprocess.run(
                        ["pgrep", "-f", "selfplay|gpu_selfplay"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode != 0 or not result.stdout.strip():
                        return 0
                    all_pids = set(result.stdout.strip().split())

                    # Get PIDs to exclude (SSH and shell processes)
                    excluded_pids: set[str] = set()
                    for pattern in (
                        "^ssh",
                        "ssh ",
                        "/bin/zsh",
                        "/bin/bash",
                        "/bin/sh",
                    ):
                        try:
                            exclude_result = subprocess.run(
                                ["pgrep", "-f", pattern],
                                capture_output=True,
                                text=True,
                                timeout=5,
                            )
                            if (
                                exclude_result.returncode == 0
                                and exclude_result.stdout.strip()
                            ):
                                excluded_pids.update(exclude_result.stdout.strip().split())
                        except (
                            subprocess.SubprocessError,
                            subprocess.TimeoutExpired,
                            OSError,
                        ):
                            pass

                    # Return count excluding non-Python processes
                    return len(all_pids - excluded_pids)
                except (subprocess.SubprocessError, subprocess.TimeoutExpired, OSError):
                    return 0

            actual_processes = await asyncio.to_thread(_count_local_selfplay_processes)

            jobs_lock = self._get_jobs_lock()
            local_jobs = self._get_local_jobs()
            tracked_jobs = 0
            if jobs_lock and local_jobs:
                with jobs_lock:
                    tracked_jobs = len(local_jobs)
            elif local_jobs:
                tracked_jobs = len(local_jobs)

            # If we have way more processes than tracked jobs, we have orphans
            if actual_processes > tracked_jobs + 10:
                self._stats.orphans_detected += actual_processes - tracked_jobs
                logger.info(
                    f"LOCAL: Orphan detection: {actual_processes} processes vs "
                    f"{tracked_jobs} tracked"
                )
                # Don't auto-kill orphans yet, just warn
                self._stats.last_orphan_check = now
        except (
            subprocess.SubprocessError,
            subprocess.TimeoutExpired,
            OSError,
            ValueError,
        ):
            pass

    async def remote_kill_stuck_job(
        self, target_node: str, job_id: str, job_type: str
    ) -> bool:
        """Send kill command to remote node for stuck job.

        This is the public method that can be called from outside the class.

        Args:
            target_node: The node ID to send kill command to
            job_id: The job ID to kill
            job_type: The type of job (training, selfplay, etc.)

        Returns:
            True if kill command succeeded, False otherwise
        """
        return await self._remote_kill_stuck_job(target_node, job_id, job_type)

    async def _remote_kill_stuck_job(
        self, target_node: str, job_id: str, job_type: str
    ) -> bool:
        """Send kill command to remote node for stuck job."""
        from scripts.p2p.connection_pool import get_client_session

        self._stats.remote_kills_attempted += 1

        peer = self._get_peer_by_id(target_node)
        if not peer or not peer.is_alive():
            return False

        try:
            from aiohttp import ClientTimeout

            timeout = ClientTimeout(total=self.config.kill_timeout)
            async with get_client_session(timeout) as session:
                url = self._url_for_peer(peer, "/job/kill")
                payload = {"job_id": job_id, "job_type": job_type, "reason": "stuck"}
                async with session.post(
                    url, json=payload, headers=self._auth_headers()
                ) as resp:
                    if resp.status == 200:
                        self._stats.remote_kills_succeeded += 1
                        return True
                    return False
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to kill stuck job on {target_node}: {e}")
            return False

    # ========================================================================
    # Health check
    # ========================================================================

    def health_check(self) -> dict[str, Any]:
        """Return health check information for DaemonManager integration.

        Returns:
            Dict with health status and statistics
        """
        now = time.time()

        # Check if we've been checking recently
        time_since_last_check = (
            now - self._stats.last_check_time
            if self._stats.last_check_time > 0
            else float("inf")
        )

        # Calculate kill success rate
        kill_success_rate = (
            self._stats.remote_kills_succeeded / self._stats.remote_kills_attempted
            if self._stats.remote_kills_attempted > 0
            else 1.0
        )

        # Determine overall health
        if kill_success_rate < 0.5:
            status = "DEGRADED"
        elif time_since_last_check > 1800:  # 30 minutes
            status = "STALE"
        else:
            status = "HEALTHY"

        return {
            "status": status,
            "stats": {
                "stuck_jobs_detected": self._stats.stuck_jobs_detected,
                "stuck_jobs_killed": self._stats.stuck_jobs_killed,
                "local_stuck_processes_killed": self._stats.local_stuck_processes_killed,
                "remote_kills_attempted": self._stats.remote_kills_attempted,
                "remote_kills_succeeded": self._stats.remote_kills_succeeded,
                "orphans_detected": self._stats.orphans_detected,
            },
            "last_check_time": self._stats.last_check_time,
            "last_local_stuck_check": self._stats.last_local_stuck_check,
            "time_since_last_check": (
                time_since_last_check if self._stats.last_check_time > 0 else None
            ),
            "kill_success_rate": kill_success_rate,
        }
