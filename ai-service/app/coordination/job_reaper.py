"""Job Reaper Daemon - Automatic timeout enforcement and work reassignment.

This daemon runs on the P2P leader and:
1. Detects jobs that have exceeded their timeout
2. Kills stuck processes via SSH if needed
3. Marks timed-out work as TIMEOUT in the work queue
4. Automatically reassigns failed work to other nodes
5. Maintains a temporary blacklist for failing nodes

Usage:
    # Typically started by P2P orchestrator leader
    from app.coordination.job_reaper import JobReaperDaemon

    reaper = JobReaperDaemon(work_queue, cluster_config)
    asyncio.create_task(reaper.run())
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from app.coordination.work_queue import WorkQueue

logger = logging.getLogger(__name__)

# ============================================
# Configuration
# ============================================

CHECK_INTERVAL = 30  # seconds between checks
DEFAULT_JOB_TIMEOUT = 3600  # 1 hour default timeout
MAX_REASSIGN_ATTEMPTS = 3  # Maximum times to reassign a failed job
NODE_BLACKLIST_DURATION = 600  # 10 minutes blacklist for failing nodes
SSH_TIMEOUT = 30  # seconds for SSH commands


class ReaperAction(str, Enum):
    """Actions taken by the reaper."""
    TIMEOUT = "timeout"
    REASSIGN = "reassign"
    BLACKLIST = "blacklist"
    KILL = "kill"


@dataclass
class BlacklistedNode:
    """A temporarily blacklisted node."""
    node_id: str
    reason: str
    blacklisted_at: float
    expires_at: float
    failure_count: int = 1


@dataclass
class ReaperStats:
    """Statistics for the job reaper."""
    jobs_reaped: int = 0
    jobs_reassigned: int = 0
    processes_killed: int = 0
    nodes_blacklisted: int = 0
    last_check: Optional[float] = None
    errors: int = 0


class JobReaperDaemon:
    """Background daemon that enforces job timeouts and reassigns work."""

    def __init__(
        self,
        work_queue: "WorkQueue",
        ssh_config: Optional[dict[str, Any]] = None,
        check_interval: float = CHECK_INTERVAL,
    ):
        self.work_queue = work_queue
        self.ssh_config = ssh_config or {}
        self.check_interval = check_interval

        self.running = False
        self.blacklisted_nodes: dict[str, BlacklistedNode] = {}
        self.stats = ReaperStats()

        # Job type specific timeouts (in seconds)
        self.job_timeouts = {
            "selfplay": 7200,       # 2 hours
            "gpu_selfplay": 3600,   # 1 hour
            "hybrid_selfplay": 5400, # 1.5 hours
            "gumbel_selfplay": 10800, # 3 hours (MCTS is slow)
            "training": 14400,      # 4 hours
            "cmaes": 7200,          # 2 hours
            "tournament": 3600,     # 1 hour
            "gauntlet": 7200,       # 2 hours
            "data_export": 1800,    # 30 minutes
        }

    def get_timeout_for_job(self, job_type: str) -> float:
        """Get the timeout for a specific job type."""
        return self.job_timeouts.get(job_type.lower(), DEFAULT_JOB_TIMEOUT)

    def is_node_blacklisted(self, node_id: str) -> bool:
        """Check if a node is currently blacklisted."""
        if node_id not in self.blacklisted_nodes:
            return False

        bl = self.blacklisted_nodes[node_id]
        if time.time() > bl.expires_at:
            # Expired, remove from blacklist
            del self.blacklisted_nodes[node_id]
            logger.info(f"Node {node_id} removed from blacklist (expired)")
            return False

        return True

    def blacklist_node(self, node_id: str, reason: str, duration: float = NODE_BLACKLIST_DURATION):
        """Add a node to the temporary blacklist."""
        now = time.time()

        if node_id in self.blacklisted_nodes:
            # Extend blacklist and increment count
            bl = self.blacklisted_nodes[node_id]
            bl.failure_count += 1
            bl.expires_at = now + duration * bl.failure_count  # Longer blacklist for repeat offenders
            bl.reason = reason
            logger.warning(
                f"Node {node_id} blacklist extended (failure #{bl.failure_count}): {reason}"
            )
        else:
            self.blacklisted_nodes[node_id] = BlacklistedNode(
                node_id=node_id,
                reason=reason,
                blacklisted_at=now,
                expires_at=now + duration,
            )
            self.stats.nodes_blacklisted += 1
            logger.warning(f"Node {node_id} blacklisted for {duration}s: {reason}")

    async def _get_timed_out_jobs(self) -> list[dict[str, Any]]:
        """Query work queue for jobs that have exceeded their timeout."""
        timed_out = []

        try:
            # Get all running jobs
            running_jobs = self.work_queue.get_running_items()

            now = time.time()
            for job in running_jobs:
                job_type = job.get("work_type", "unknown")
                timeout = self.get_timeout_for_job(job_type)
                started_at = job.get("started_at", 0)

                if started_at and (now - started_at) > timeout:
                    job["timeout_duration"] = now - started_at
                    job["expected_timeout"] = timeout
                    timed_out.append(job)

        except Exception as e:
            logger.error(f"Error getting timed out jobs: {e}")
            self.stats.errors += 1

        return timed_out

    async def _kill_remote_process(self, node_id: str, pid: int) -> bool:
        """Kill a process on a remote node via SSH."""
        if not pid:
            return False

        # Get SSH config for node
        ssh_host = self.ssh_config.get(node_id, {}).get("host", node_id)
        ssh_user = self.ssh_config.get(node_id, {}).get("user", "ubuntu")
        ssh_key = self.ssh_config.get(node_id, {}).get("key")

        ssh_cmd = ["ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no"]
        if ssh_key:
            ssh_cmd.extend(["-i", ssh_key])

        ssh_cmd.extend([f"{ssh_user}@{ssh_host}", f"kill -9 {pid} 2>/dev/null || true"])

        try:
            proc = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=SSH_TIMEOUT)

            if proc.returncode == 0:
                logger.info(f"Killed process {pid} on {node_id}")
                self.stats.processes_killed += 1
                return True
            else:
                logger.warning(f"Failed to kill process {pid} on {node_id}")
                return False

        except asyncio.TimeoutError:
            logger.error(f"Timeout killing process {pid} on {node_id}")
            return False
        except Exception as e:
            logger.error(f"Error killing process {pid} on {node_id}: {e}")
            return False

    async def _reap_stuck_jobs(self):
        """Find and terminate stuck jobs."""
        timed_out = await self._get_timed_out_jobs()

        for job in timed_out:
            job_id = job.get("work_id")
            node_id = job.get("claimed_by")
            pid = job.get("pid")
            job_type = job.get("work_type", "unknown")
            duration = job.get("timeout_duration", 0)

            logger.warning(
                f"Job {job_id} ({job_type}) on {node_id} timed out after {duration:.0f}s "
                f"(expected: {job.get('expected_timeout', 0):.0f}s)"
            )

            # Kill the process if we have a PID
            if pid and node_id:
                await self._kill_remote_process(node_id, pid)

            # Mark as timed out in work queue
            try:
                self.work_queue.timeout_work(job_id)
                self.stats.jobs_reaped += 1
                logger.info(f"Marked job {job_id} as TIMEOUT")
            except Exception as e:
                logger.error(f"Error marking job {job_id} as timeout: {e}")
                self.stats.errors += 1

            # Blacklist the node if it's a repeat offender
            if node_id:
                # Check how many recent timeouts this node has
                recent_timeouts = sum(
                    1 for bl in self.blacklisted_nodes.values()
                    if bl.node_id == node_id and time.time() - bl.blacklisted_at < 3600
                )
                if recent_timeouts >= 2:
                    self.blacklist_node(node_id, f"Multiple job timeouts ({recent_timeouts + 1})")

    async def _reassign_failed_work(self):
        """Find failed/timed out work and reassign to other nodes."""
        try:
            # Get failed items that can be retried
            failed_items = self.work_queue.get_retriable_items(max_attempts=MAX_REASSIGN_ATTEMPTS)

            for item in failed_items:
                work_id = item.get("work_id")
                attempts = item.get("attempts", 0)
                failed_node = item.get("claimed_by")

                # Don't reassign to the same node or blacklisted nodes
                excluded_nodes = {failed_node} if failed_node else set()
                excluded_nodes.update(self.blacklisted_nodes.keys())

                # Reset to pending for reassignment
                try:
                    self.work_queue.reset_for_retry(
                        work_id,
                        excluded_nodes=list(excluded_nodes),
                    )
                    self.stats.jobs_reassigned += 1
                    logger.info(
                        f"Reassigned job {work_id} for retry (attempt {attempts + 1}/{MAX_REASSIGN_ATTEMPTS}), "
                        f"excluding nodes: {excluded_nodes}"
                    )
                except Exception as e:
                    logger.error(f"Error reassigning job {work_id}: {e}")
                    self.stats.errors += 1

        except Exception as e:
            logger.error(f"Error in reassignment loop: {e}")
            self.stats.errors += 1

    async def _cleanup_expired_blacklists(self):
        """Remove expired entries from the blacklist."""
        now = time.time()
        expired = [
            node_id for node_id, bl in self.blacklisted_nodes.items()
            if now > bl.expires_at
        ]
        for node_id in expired:
            del self.blacklisted_nodes[node_id]
            logger.info(f"Node {node_id} removed from blacklist (expired)")

    async def run(self):
        """Main daemon loop."""
        self.running = True
        logger.info("Job Reaper Daemon starting")

        while self.running:
            try:
                self.stats.last_check = time.time()

                # Cleanup expired blacklists
                await self._cleanup_expired_blacklists()

                # Reap stuck jobs
                await self._reap_stuck_jobs()

                # Reassign failed work
                await self._reassign_failed_work()

            except Exception as e:
                logger.error(f"Error in reaper loop: {e}")
                self.stats.errors += 1

            await asyncio.sleep(self.check_interval)

        logger.info("Job Reaper Daemon stopped")

    def stop(self):
        """Stop the daemon."""
        self.running = False

    def get_stats(self) -> dict[str, Any]:
        """Get daemon statistics."""
        return {
            "jobs_reaped": self.stats.jobs_reaped,
            "jobs_reassigned": self.stats.jobs_reassigned,
            "processes_killed": self.stats.processes_killed,
            "nodes_blacklisted": self.stats.nodes_blacklisted,
            "currently_blacklisted": len(self.blacklisted_nodes),
            "blacklisted_nodes": [
                {
                    "node_id": bl.node_id,
                    "reason": bl.reason,
                    "expires_in": bl.expires_at - time.time(),
                    "failure_count": bl.failure_count,
                }
                for bl in self.blacklisted_nodes.values()
            ],
            "last_check": self.stats.last_check,
            "errors": self.stats.errors,
            "running": self.running,
        }
