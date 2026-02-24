"""Memory and Disk Manager for P2P Orchestrator.

January 2026: Phase 10 Aggressive Decomposition - Extracts memory pressure handling
and disk cleanup operations from the monolithic p2p_orchestrator.py.

This manager handles:
- Emergency memory cleanup (cache clearing, GC)
- Local disk cleanup (old logs, temp files)
- Remote cleanup requests
- Selfplay job reduction for load shedding

Dependencies:
- Orchestrator reference for path helpers and job tracking
- subprocess for disk monitor script
- asyncio.to_thread for non-blocking I/O
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scripts.p2p_orchestrator import P2POrchestrator

logger = logging.getLogger(__name__)

# Singleton instance
_memory_disk_manager: "MemoryDiskManager | None" = None

# Constants - aligned with app.config.thresholds (canonical source)
try:
    from app.config.thresholds import DISK_CRITICAL_PERCENT, DISK_PRODUCTION_HALT_PERCENT
    DISK_CLEANUP_THRESHOLD = DISK_PRODUCTION_HALT_PERCENT - 5  # Start cleanup at 80% (before production halt)
    DISK_CRITICAL_THRESHOLD = DISK_CRITICAL_PERCENT  # Force cleanup at 90%
except ImportError:
    DISK_CLEANUP_THRESHOLD = 80
    DISK_CRITICAL_THRESHOLD = 90
MEMORY_WARNING_THRESHOLD = 0.7  # 70% RAM
MEMORY_CRITICAL_THRESHOLD = 0.85  # 85% RAM
HTTP_TOTAL_TIMEOUT = 30  # seconds


@dataclass
class MemoryDiskConfig:
    """Configuration for MemoryDiskManager."""

    memory_warning_threshold: float = MEMORY_WARNING_THRESHOLD
    memory_critical_threshold: float = MEMORY_CRITICAL_THRESHOLD
    disk_cleanup_threshold: int = DISK_CLEANUP_THRESHOLD
    disk_critical_threshold: int = DISK_CRITICAL_THRESHOLD
    cleanup_request_timeout: float = HTTP_TOTAL_TIMEOUT
    log_retention_days: int = 7


@dataclass
class MemoryDiskStats:
    """Statistics tracked by MemoryDiskManager."""

    emergency_cleanups: int = 0
    disk_cleanups: int = 0
    remote_cleanup_requests: int = 0
    selfplay_jobs_reduced: int = 0
    bytes_cleaned: int = 0
    gc_collections: int = 0


class MemoryDiskManager:
    """Manages memory pressure and disk cleanup operations.

    Extracted from P2POrchestrator in January 2026 (Phase 10) to improve
    modularity and testability.
    """

    def __init__(
        self,
        config: MemoryDiskConfig | None = None,
        orchestrator: "P2POrchestrator | None" = None,
    ):
        """Initialize MemoryDiskManager.

        Args:
            config: Configuration options
            orchestrator: Reference to P2P orchestrator for state access
        """
        self.config = config or MemoryDiskConfig()
        self._orchestrator = orchestrator
        self._stats = MemoryDiskStats()

    def _get_ai_service_path(self) -> str:
        """Get path to ai-service directory."""
        if self._orchestrator and hasattr(self._orchestrator, "_get_ai_service_path"):
            return self._orchestrator._get_ai_service_path()
        # Fallback: derive from this file's location
        return str(Path(__file__).parent.parent.parent.parent)

    def _get_ringrift_path(self) -> Path:
        """Get path to RingRift root directory."""
        if self._orchestrator and hasattr(self._orchestrator, "ringrift_path"):
            return self._orchestrator.ringrift_path
        # Fallback: derive from ai-service path
        return Path(self._get_ai_service_path()).parent

    def _get_resource_usage(self) -> dict[str, float]:
        """Get current resource usage.

        Returns:
            Dict with cpu_percent, memory_percent, disk_percent
        """
        if self._orchestrator and hasattr(self._orchestrator, "_get_resource_usage"):
            return self._orchestrator._get_resource_usage()

        # Fallback implementation
        try:
            import psutil

            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory().percent
            disk = psutil.disk_usage("/").percent
            return {"cpu_percent": cpu, "memory_percent": mem, "disk_percent": disk}
        except ImportError:
            return {"cpu_percent": 0.0, "memory_percent": 0.0, "disk_percent": 0.0}

    def emergency_memory_cleanup(self) -> dict[str, Any]:
        """Emergency memory cleanup when memory is critical.

        Clears gossip caches and triggers garbage collection
        to free memory when above critical threshold.

        Returns:
            Dict with cleanup stats
        """
        self._stats.emergency_cleanups += 1

        states_cleared = 0
        manifests_cleared = 0

        # Clear gossip state caches if orchestrator available
        if self._orchestrator:
            gossip_states = getattr(self._orchestrator, "_gossip_peer_states", None)
            gossip_manifests = getattr(self._orchestrator, "_gossip_peer_manifests", None)

            if gossip_states:
                states_cleared = len(gossip_states)
                gossip_states.clear()

            if gossip_manifests:
                manifests_cleared = len(gossip_manifests)
                gossip_manifests.clear()

        # Force garbage collection
        gc.collect()
        self._stats.gc_collections += 1

        logger.info(
            f"[MemoryDisk] Emergency memory cleanup: cleared {states_cleared} gossip states, "
            f"{manifests_cleared} manifests, ran gc.collect()"
        )

        return {
            "states_cleared": states_cleared,
            "manifests_cleared": manifests_cleared,
            "gc_ran": True,
        }

    async def cleanup_local_disk(self) -> dict[str, Any]:
        """Clean up disk space on local node.

        Automatically archives old data:
        - Remove deprecated selfplay databases
        - Compress and archive old logs
        - Clear /tmp files older than 24h

        Returns:
            Dict with cleanup results
        """
        self._stats.disk_cleanups += 1
        logger.info("[MemoryDisk] Running local disk cleanup...")

        result: dict[str, Any] = {
            "success": False,
            "method": "none",
            "cleaned_files": [],
        }

        try:
            ai_service_path = self._get_ai_service_path()
            disk_monitor = Path(ai_service_path) / "scripts" / "disk_monitor.py"

            if disk_monitor.exists():
                # Use shared disk monitor for consistent cleanup policy
                usage = await asyncio.to_thread(self._get_resource_usage)
                disk_percent = float(usage.get("disk_percent", 0.0) or 0.0)

                cmd = [
                    sys.executable,
                    str(disk_monitor),
                    "--threshold",
                    str(self.config.disk_cleanup_threshold),
                    "--ringrift-path",
                    str(self._get_ringrift_path()),
                    "--aggressive",
                ]
                if disk_percent >= self.config.disk_critical_threshold:
                    cmd.append("--force")

                def _run_cleanup() -> subprocess.CompletedProcess:
                    return subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=300,
                        cwd=ai_service_path,
                    )

                out = await asyncio.to_thread(_run_cleanup)
                if out.returncode == 0:
                    logger.info("[MemoryDisk] Disk monitor cleanup completed")
                    result["success"] = True
                    result["method"] = "disk_monitor"
                else:
                    logger.info(f"[MemoryDisk] Disk monitor cleanup failed: {out.stderr[:200]}")
                    result["error"] = out.stderr[:200]
            else:
                # Minimal fallback: clear old logs
                def _cleanup_old_logs() -> list[str]:
                    log_dir = Path(ai_service_path) / "logs"
                    cleaned = []
                    if log_dir.exists():
                        cutoff = time.time() - (self.config.log_retention_days * 86400)
                        for logfile in log_dir.rglob("*.log"):
                            try:
                                if logfile.stat().st_mtime < cutoff:
                                    logfile.unlink()
                                    cleaned.append(str(logfile))
                            except (OSError, IOError):
                                continue
                    return cleaned

                cleaned = await asyncio.to_thread(_cleanup_old_logs)
                result["success"] = True
                result["method"] = "fallback_logs"
                result["cleaned_files"] = cleaned
                for logfile in cleaned:
                    logger.info(f"[MemoryDisk] Cleaned old log: {logfile}")

        except Exception as e:  # noqa: BLE001
            logger.info(f"[MemoryDisk] Disk cleanup error: {e}")
            result["error"] = str(e)

        return result

    async def request_remote_cleanup(self, node_id: str, host: str, port: int = 8770) -> bool:
        """Request a remote node to clean up disk space.

        Args:
            node_id: ID of the remote node
            host: Hostname or IP of the remote node
            port: HTTP port of the remote node

        Returns:
            True if request was successful
        """
        self._stats.remote_cleanup_requests += 1

        try:
            from aiohttp import ClientSession, ClientTimeout

            timeout = ClientTimeout(total=self.config.cleanup_request_timeout)

            # Try direct HTTP request
            url = f"http://{host}:{port}/cleanup"
            async with ClientSession(timeout=timeout) as session:
                async with session.post(url, json={}) as resp:
                    if resp.status == 200:
                        logger.info(f"[MemoryDisk] Cleanup requested on {node_id}")
                        return True
                    else:
                        logger.info(f"[MemoryDisk] Cleanup request failed on {node_id}: HTTP {resp.status}")
                        return False

        except Exception as e:  # noqa: BLE001
            logger.error(f"[MemoryDisk] Failed to request cleanup from {node_id}: {e}")
            return False

    async def request_remote_cleanup_via_orchestrator(self, node: Any) -> bool:
        """Request remote cleanup using orchestrator's node info.

        Args:
            node: NodeInfo object from orchestrator

        Returns:
            True if request was successful
        """
        if not self._orchestrator:
            return False

        self._stats.remote_cleanup_requests += 1
        node_id = getattr(node, "node_id", "unknown")

        try:
            # Check if NAT-blocked, use relay queue
            if getattr(node, "nat_blocked", False):
                if hasattr(self._orchestrator, "_enqueue_relay_command_for_peer"):
                    cmd_id = await self._orchestrator._enqueue_relay_command_for_peer(node, "cleanup", {})
                    if cmd_id:
                        logger.info(f"[MemoryDisk] Enqueued relay cleanup for {node_id}")
                        return True
                    else:
                        logger.info(f"[MemoryDisk] Relay queue full; skipping cleanup enqueue for {node_id}")
                return False

            # Direct HTTP request
            from aiohttp import ClientTimeout

            if hasattr(self._orchestrator, "get_client_session"):
                get_client_session = self._orchestrator.get_client_session
            else:
                from scripts.p2p_orchestrator import get_client_session

            timeout = ClientTimeout(total=self.config.cleanup_request_timeout)
            async with get_client_session(timeout) as session:
                last_err: str | None = None

                # Try multiple URLs for the peer
                if hasattr(self._orchestrator, "_urls_for_peer"):
                    urls = self._orchestrator._urls_for_peer(node, "/cleanup")
                else:
                    # Fallback: construct URL directly
                    host = getattr(node, "host", None) or getattr(node, "tailscale_ip", None)
                    port = getattr(node, "port", 8770)
                    urls = [f"http://{host}:{port}/cleanup"] if host else []

                for url in urls:
                    try:
                        headers = {}
                        if hasattr(self._orchestrator, "_auth_headers"):
                            headers = self._orchestrator._auth_headers()

                        async with session.post(url, json={}, headers=headers) as resp:
                            if resp.status == 200:
                                logger.info(f"[MemoryDisk] Cleanup requested on {node_id}")
                                return True
                            last_err = f"http_{resp.status}"
                    except Exception as e:  # noqa: BLE001
                        last_err = str(e)
                        continue

                if last_err:
                    logger.info(f"[MemoryDisk] Cleanup request failed on {node_id}: {last_err}")

        except Exception as e:  # noqa: BLE001
            logger.error(f"[MemoryDisk] Failed to request cleanup from {node_id}: {e}")

        return False

    async def reduce_local_selfplay_jobs(self, target_selfplay_jobs: int, *, reason: str) -> dict[str, Any]:
        """Best-effort: stop excess selfplay jobs on this node (load shedding).

        Used when disk/memory pressure is high to avoid OOM/disk-full scenarios.

        Args:
            target_selfplay_jobs: Target number of selfplay jobs to keep
            reason: Reason for reduction (logged)

        Returns:
            Dict with before/after counts and stopped count
        """
        try:
            target = max(0, int(target_selfplay_jobs))
        except ValueError:
            target = 0

        if not self._orchestrator:
            return {
                "running_before": 0,
                "running_after": 0,
                "stopped": 0,
                "target": target,
                "reason": reason,
                "error": "No orchestrator reference",
            }

        # Get initial count using orchestrator's method
        try:
            if hasattr(self._orchestrator, "_count_local_jobs"):
                selfplay_before, _training_before = await asyncio.to_thread(
                    self._orchestrator._count_local_jobs
                )
            else:
                selfplay_before = 0
        except (AttributeError, TypeError):
            selfplay_before = 0

        # Hard shedding (target=0): use existing restart sweep
        if target <= 0:
            if hasattr(self._orchestrator, "_restart_local_stuck_jobs"):
                await self._orchestrator._restart_local_stuck_jobs()

            try:
                if hasattr(self._orchestrator, "_count_local_jobs"):
                    selfplay_after, _training_after = await asyncio.to_thread(
                        self._orchestrator._count_local_jobs
                    )
                else:
                    selfplay_after = 0
            except (AttributeError, TypeError):
                selfplay_after = 0

            stopped = max(0, int(selfplay_before) - int(selfplay_after))
            self._stats.selfplay_jobs_reduced += stopped
            return {
                "running_before": int(selfplay_before),
                "running_after": int(selfplay_after),
                "stopped": stopped,
                "target": 0,
                "reason": reason,
            }

        # Get running selfplay jobs from orchestrator
        jobs_lock = getattr(self._orchestrator, "jobs_lock", None)
        local_jobs = getattr(self._orchestrator, "local_jobs", {})

        running: list[tuple[str, Any]] = []
        if jobs_lock:
            with jobs_lock:
                running = self._get_running_selfplay_jobs(local_jobs)
        else:
            running = self._get_running_selfplay_jobs(local_jobs)

        if selfplay_before <= target and len(running) <= target:
            return {
                "running_before": int(selfplay_before),
                "running_after": int(selfplay_before),
                "stopped": 0,
                "target": target,
                "reason": reason,
            }

        # Stop newest-first to avoid killing long-running jobs near completion
        running.sort(key=lambda pair: float(getattr(pair[1], "started_at", 0.0) or 0.0), reverse=True)
        to_stop = running[target:]

        stopped = 0
        if jobs_lock:
            with jobs_lock:
                stopped = self._stop_jobs(to_stop)
        else:
            stopped = self._stop_jobs(to_stop)

        # Kill untracked processes if still over target
        selfplay_mid = max(0, int(selfplay_before) - stopped)
        try:
            if hasattr(self._orchestrator, "_count_local_jobs"):
                selfplay_mid, _training_mid = await asyncio.to_thread(
                    self._orchestrator._count_local_jobs
                )
        except (AttributeError, TypeError):
            pass

        if selfplay_mid > target:
            additional_killed = await self._kill_untracked_selfplay_processes(
                selfplay_mid - target
            )
            stopped += additional_killed

        if stopped:
            self._stats.selfplay_jobs_reduced += stopped
            if hasattr(self._orchestrator, "_save_state"):
                self._orchestrator._save_state()

        # Get final count
        try:
            if hasattr(self._orchestrator, "_count_local_jobs"):
                selfplay_after, _training_after = await asyncio.to_thread(
                    self._orchestrator._count_local_jobs
                )
            else:
                selfplay_after = max(0, int(selfplay_before) - stopped)
        except (AttributeError, TypeError):
            selfplay_after = max(0, int(selfplay_before) - stopped)

        return {
            "running_before": int(selfplay_before),
            "running_after": int(selfplay_after),
            "stopped": int(max(0, int(selfplay_before) - int(selfplay_after))),
            "target": target,
            "reason": reason,
        }

    def _get_running_selfplay_jobs(self, local_jobs: dict) -> list[tuple[str, Any]]:
        """Get list of running selfplay jobs.

        Args:
            local_jobs: Dict of job_id -> ClusterJob

        Returns:
            List of (job_id, job) tuples for running selfplay jobs
        """
        # Import JobType if available
        try:
            from scripts.p2p_orchestrator import JobType

            selfplay_types = (
                JobType.SELFPLAY,
                JobType.GPU_SELFPLAY,
                JobType.HYBRID_SELFPLAY,
                JobType.CPU_SELFPLAY,
                JobType.GUMBEL_SELFPLAY,
            )
        except ImportError:
            # Fallback: check string names
            selfplay_types = None

        running = []
        for job_id, job in local_jobs.items():
            if getattr(job, "status", None) != "running":
                continue

            job_type = getattr(job, "job_type", None)
            if selfplay_types and job_type in selfplay_types:
                running.append((job_id, job))
            elif job_type and "selfplay" in str(job_type).lower():
                running.append((job_id, job))

        return running

    def _stop_jobs(self, jobs: list[tuple[str, Any]]) -> int:
        """Stop a list of jobs.

        Args:
            jobs: List of (job_id, job) tuples

        Returns:
            Number of jobs stopped
        """
        stopped = 0
        for _job_id, job in jobs:
            try:
                pid = getattr(job, "pid", None)
                if pid:
                    os.kill(int(pid), signal.SIGTERM)
                job.status = "stopped"
                stopped += 1
            except (ValueError, AttributeError, OSError):
                continue
        return stopped

    async def _kill_untracked_selfplay_processes(self, excess: int) -> int:
        """Kill untracked selfplay processes.

        Args:
            excess: Number of excess processes to kill

        Returns:
            Number of processes killed
        """
        import shutil

        if not shutil.which("pgrep"):
            return 0

        killed = 0
        try:
            pids: list[int] = []
            patterns = (
                "selfplay.py",
                "run_self_play_soak.py",
                "run_gpu_selfplay.py",
                "run_hybrid_selfplay.py",
                "run_random_selfplay.py",
            )

            # Use orchestrator's async subprocess helper if available
            run_subprocess = None
            if self._orchestrator and hasattr(self._orchestrator, "_run_subprocess_async"):
                run_subprocess = self._orchestrator._run_subprocess_async
            else:
                # Fallback implementation
                async def _run_subprocess_async(
                    cmd: list[str], timeout: float = 5
                ) -> tuple[int, str, str]:
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    try:
                        stdout, stderr = await asyncio.wait_for(
                            proc.communicate(), timeout=timeout
                        )
                        return proc.returncode or 0, stdout.decode(), stderr.decode()
                    except asyncio.TimeoutError:
                        proc.kill()
                        return -1, "", "timeout"

                run_subprocess = _run_subprocess_async

            for pattern in patterns:
                returncode, stdout, _stderr = await run_subprocess(
                    ["pgrep", "-f", pattern], timeout=5
                )
                if returncode == 0 and stdout.strip():
                    for token in stdout.strip().split():
                        try:
                            pids.append(int(token))
                        except ValueError:
                            continue

            # Kill newest-ish (highest PID) first
            pids = sorted(set(pids), reverse=True)
            for pid in pids:
                if killed >= excess:
                    break
                try:
                    os.kill(pid, signal.SIGTERM)
                    killed += 1
                except (OSError, ProcessLookupError):
                    continue

        except Exception as e:  # noqa: BLE001
            logger.debug(f"[MemoryDisk] Error killing untracked processes: {e}")

        return killed

    async def request_reduce_selfplay(
        self, node: Any, target_selfplay_jobs: int, *, reason: str
    ) -> bool:
        """Ask a node to shed excess selfplay (for memory/disk pressure).

        Args:
            node: NodeInfo object from orchestrator
            target_selfplay_jobs: Target number of selfplay jobs
            reason: Reason for reduction

        Returns:
            True if request was successful
        """
        if not self._orchestrator:
            return False

        try:
            target = max(0, int(target_selfplay_jobs))
        except ValueError:
            target = 0

        node_id = getattr(node, "node_id", "unknown")

        # Check if NAT-blocked, use relay queue
        if getattr(node, "nat_blocked", False):
            if hasattr(self._orchestrator, "_enqueue_relay_command_for_peer"):
                payload = {"target_selfplay_jobs": target, "reason": reason}
                cmd_id = await self._orchestrator._enqueue_relay_command_for_peer(
                    node, "reduce_selfplay", payload
                )
                if cmd_id:
                    logger.info(
                        f"[MemoryDisk] Enqueued relay reduce_selfplay for {node_id} "
                        f"(target={target}, reason={reason})"
                    )
                    return True
                else:
                    logger.info(f"[MemoryDisk] Relay queue full for {node_id}; skipping reduce_selfplay enqueue")
            return False

        # Direct HTTP request
        try:
            from aiohttp import ClientTimeout

            if hasattr(self._orchestrator, "get_client_session"):
                get_client_session = self._orchestrator.get_client_session
            else:
                from scripts.p2p_orchestrator import get_client_session

            timeout = ClientTimeout(total=self.config.cleanup_request_timeout)
            async with get_client_session(timeout) as session:
                last_err: str | None = None
                payload = {"target_selfplay_jobs": target, "reason": reason}

                # Get URLs for the peer
                if hasattr(self._orchestrator, "_urls_for_peer"):
                    urls = self._orchestrator._urls_for_peer(node, "/reduce_selfplay")
                else:
                    host = getattr(node, "host", None) or getattr(node, "tailscale_ip", None)
                    port = getattr(node, "port", 8770)
                    urls = [f"http://{host}:{port}/reduce_selfplay"] if host else []

                for url in urls:
                    try:
                        headers = {}
                        if hasattr(self._orchestrator, "_auth_headers"):
                            headers = self._orchestrator._auth_headers()

                        async with session.post(url, json=payload, headers=headers) as resp:
                            if resp.status == 200:
                                logger.info(
                                    f"[MemoryDisk] Requested load shedding on {node_id} "
                                    f"(target={target}, reason={reason})"
                                )
                                return True
                            last_err = f"http_{resp.status}"
                    except Exception as e:  # noqa: BLE001
                        last_err = str(e)
                        continue

                if last_err:
                    logger.info(f"[MemoryDisk] reduce_selfplay request failed on {node_id}: {last_err}")

        except Exception as e:  # noqa: BLE001
            logger.error(f"[MemoryDisk] Failed to request reduce_selfplay from {node_id}: {e}")

        return False

    def health_check(self) -> dict[str, Any]:
        """Return health check result for daemon protocol compliance.

        Returns:
            Dict with health status and statistics
        """
        usage = self._get_resource_usage()
        memory_percent = usage.get("memory_percent", 0.0)
        disk_percent = usage.get("disk_percent", 0.0)

        # Determine health status
        if memory_percent >= self.config.memory_critical_threshold * 100:
            status = "critical"
            is_healthy = False
            message = f"Memory critical: {memory_percent:.1f}%"
        elif disk_percent >= self.config.disk_critical_threshold:
            status = "critical"
            is_healthy = False
            message = f"Disk critical: {disk_percent:.1f}%"
        elif memory_percent >= self.config.memory_warning_threshold * 100:
            status = "warning"
            is_healthy = True
            message = f"Memory warning: {memory_percent:.1f}%"
        elif disk_percent >= self.config.disk_cleanup_threshold:
            status = "warning"
            is_healthy = True
            message = f"Disk warning: {disk_percent:.1f}%"
        else:
            status = "healthy"
            is_healthy = True
            message = f"Resources OK (memory: {memory_percent:.1f}%, disk: {disk_percent:.1f}%)"

        return {
            "healthy": is_healthy,
            "status": status,
            "message": message,
            "details": {
                "memory_percent": memory_percent,
                "disk_percent": disk_percent,
                "stats": {
                    "emergency_cleanups": self._stats.emergency_cleanups,
                    "disk_cleanups": self._stats.disk_cleanups,
                    "remote_cleanup_requests": self._stats.remote_cleanup_requests,
                    "selfplay_jobs_reduced": self._stats.selfplay_jobs_reduced,
                    "gc_collections": self._stats.gc_collections,
                },
            },
        }


# Singleton accessors
def create_memory_disk_manager(
    config: MemoryDiskConfig | None = None,
    orchestrator: "P2POrchestrator | None" = None,
) -> MemoryDiskManager:
    """Create a new MemoryDiskManager instance.

    Args:
        config: Configuration options
        orchestrator: Reference to P2P orchestrator

    Returns:
        New MemoryDiskManager instance
    """
    return MemoryDiskManager(config=config, orchestrator=orchestrator)


def get_memory_disk_manager() -> MemoryDiskManager | None:
    """Get the singleton MemoryDiskManager instance.

    Returns:
        The singleton instance, or None if not set
    """
    return _memory_disk_manager


def set_memory_disk_manager(manager: MemoryDiskManager) -> None:
    """Set the singleton MemoryDiskManager instance.

    Args:
        manager: The manager instance to set as singleton
    """
    global _memory_disk_manager
    _memory_disk_manager = manager


def reset_memory_disk_manager() -> None:
    """Reset the singleton MemoryDiskManager instance."""
    global _memory_disk_manager
    _memory_disk_manager = None
