"""Job Reaper Daemon - Automatic timeout enforcement and work reassignment.

This daemon runs ONLY on the P2P leader and:
1. Detects jobs that have exceeded their timeout
2. Kills stuck processes via SSH if needed
3. Marks timed-out work as TIMEOUT in the work queue
4. Automatically reassigns failed work to other nodes
5. Maintains a temporary blacklist for failing nodes

IMPORTANT: The daemon automatically checks P2P leader status and will:
- Only perform actions when this node is the cluster leader
- Gracefully stop if leadership is lost
- Re-check leadership before each reap cycle

December 2025: Added P2P leader awareness to prevent duplicate actions
when multiple nodes try to run the reaper simultaneously.

Usage:
    # Typically started by P2P orchestrator leader
    from app.coordination.job_reaper import JobReaperDaemon

    reaper = JobReaperDaemon(work_queue, cluster_config)
    asyncio.create_task(reaper.run())
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

# December 2025: Use centralized P2P leader detection from app.core.node
from app.core.node import check_p2p_leader_status, get_this_node_id

# December 2025: Use consolidated daemon stats base class
from app.coordination.daemon_stats import JobDaemonStats

if TYPE_CHECKING:
    from app.coordination.work_queue import WorkQueue

logger = logging.getLogger(__name__)

# ============================================
# Configuration (December 27, 2025: Centralized in coordination_defaults.py)
# ============================================

from app.config.coordination_defaults import JobReaperDefaults, WorkQueueCleanupDefaults

CHECK_INTERVAL = JobReaperDefaults.CHECK_INTERVAL
DEFAULT_JOB_TIMEOUT = JobReaperDefaults.DEFAULT_JOB_TIMEOUT
MAX_REASSIGN_ATTEMPTS = JobReaperDefaults.MAX_REASSIGN_ATTEMPTS
NODE_BLACKLIST_DURATION = JobReaperDefaults.NODE_BLACKLIST_DURATION
SSH_TIMEOUT = JobReaperDefaults.SSH_TIMEOUT
LEADER_CHECK_TIMEOUT = JobReaperDefaults.LEADER_CHECK_TIMEOUT
LEADER_RETRY_DELAY = JobReaperDefaults.LEADER_RETRY_DELAY


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
class ReaperStats(JobDaemonStats):
    """Statistics for the job reaper.

    December 2025: Now extends JobDaemonStats for consistent tracking.
    Inherits: jobs_processed, jobs_succeeded, jobs_failed, jobs_timed_out,
              jobs_reassigned, errors_count, consecutive_failures, etc.
    """

    # Reaper-specific fields (not in base class)
    processes_killed: int = 0
    nodes_blacklisted: int = 0
    leader_checks: int = 0
    not_leader_skips: int = 0

    # Backward compatibility aliases
    @property
    def jobs_reaped(self) -> int:
        """Alias for jobs_timed_out (backward compatibility)."""
        return self.jobs_timed_out

    @property
    def last_check(self) -> float | None:
        """Alias for last_check_time (backward compatibility)."""
        return self.last_check_time if self.last_check_time > 0 else None

    @property
    def errors(self) -> int:
        """Alias for errors_count (backward compatibility)."""
        return self.errors_count

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary with reaper-specific fields."""
        base = super().to_dict()
        base.update({
            # Reaper-specific
            "processes_killed": self.processes_killed,
            "nodes_blacklisted": self.nodes_blacklisted,
            "leader_checks": self.leader_checks,
            "not_leader_skips": self.not_leader_skips,
            # Backward compat aliases
            "jobs_reaped": self.jobs_reaped,
            "last_check": self.last_check,
            "errors": self.errors,
        })
        return base


class JobReaperDaemon:
    """Background daemon that enforces job timeouts and reassigns work."""

    # P0.4 Dec 2025: Blacklist persistence path
    _BLACKLIST_DB_PATH = Path("data/coordination/.blacklist.db")

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

        # P0.4 Dec 2025: Initialize SQLite persistence for blacklist
        self._init_blacklist_db()
        self._load_blacklist_from_db()

        # Job type specific timeouts (in seconds)
        # Dec 2025: Increased gpu_selfplay from 1hr to 2hr - jobs taking 1.5+ hours were killed prematurely
        self.job_timeouts = {
            "selfplay": 7200,       # 2 hours
            "gpu_selfplay": 7200,   # 2 hours (was 1hr, increased Dec 2025)
            "hybrid_selfplay": 7200, # 2 hours (was 1.5hr, aligned Dec 2025)
            "gumbel_selfplay": 10800, # 3 hours (MCTS is slow)
            "training": 14400,      # 4 hours
            "cmaes": 7200,          # 2 hours
            "tournament": 3600,     # 1 hour
            "gauntlet": 7200,       # 2 hours
            "data_export": 1800,    # 30 minutes
        }

    def get_timeout_for_job(self, job_type: str, job_params: dict[str, Any] | None = None) -> float:
        """Get the timeout for a specific job type with adaptive scaling.

        December 29, 2025: Added adaptive timeout calculation based on job parameters.
        This prevents premature kills for large jobs while keeping short timeouts
        for small jobs.

        Args:
            job_type: The type of job (selfplay, training, etc.)
            job_params: Optional job parameters for adaptive scaling:
                - num_games: Number of games (for selfplay)
                - num_samples: Number of training samples
                - board_type: Board configuration (larger boards = more time)
                - num_players: Player count (more players = more time)

        Returns:
            Timeout in seconds, scaled by job size.
        """
        base_timeout = self.job_timeouts.get(job_type.lower(), DEFAULT_JOB_TIMEOUT)

        if not job_params:
            return base_timeout

        multiplier = 1.0

        # Scale by number of games (selfplay jobs)
        num_games = job_params.get("num_games", 0)
        if num_games > 0:
            # Base: 100 games = 1x, 500 games = 2x, 1000 games = 3x
            game_multiplier = 1.0 + (num_games / 500)
            multiplier *= min(5.0, game_multiplier)  # Cap at 5x

        # Scale by number of samples (training jobs)
        num_samples = job_params.get("num_samples", 0)
        if num_samples > 0:
            # Base: 100K samples = 1x, 500K samples = 2x, 1M samples = 3x
            sample_multiplier = 1.0 + (num_samples / 500000)
            multiplier *= min(5.0, sample_multiplier)  # Cap at 5x

        # Scale by board type (larger boards are slower)
        board_type = job_params.get("board_type", "")
        if board_type:
            board_multipliers = {
                "hex8": 1.0,
                "square8": 1.0,
                "square19": 2.5,    # ~6x more cells than square8
                "hexagonal": 3.0,   # ~7x more cells than hex8
            }
            multiplier *= board_multipliers.get(board_type, 1.0)

        # Scale by player count (more players = longer games)
        num_players = job_params.get("num_players", 2)
        if num_players > 2:
            # Each additional player adds ~25% time
            player_multiplier = 1.0 + (num_players - 2) * 0.25
            multiplier *= player_multiplier

        # Apply multiplier with reasonable bounds
        scaled_timeout = base_timeout * multiplier
        max_timeout = base_timeout * 10  # Never exceed 10x base
        min_timeout = base_timeout * 0.5  # Never go below 0.5x base

        return max(min_timeout, min(max_timeout, scaled_timeout))

    def get_timeout_for_job_simple(self, job_type: str) -> float:
        """Get the base timeout for a specific job type (no adaptive scaling).

        December 29, 2025: Kept for backward compatibility with callers that
        don't have job parameters.
        """
        return self.job_timeouts.get(job_type.lower(), DEFAULT_JOB_TIMEOUT)

    # =========================================================================
    # P0.4 Dec 2025: Blacklist Persistence
    # =========================================================================

    def _init_blacklist_db(self) -> None:
        """Initialize SQLite table for blacklist persistence.

        P0.4 Dec 2025: Prevents failed nodes from immediately getting new jobs
        after job_reaper restart.
        """
        try:
            self._BLACKLIST_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self._BLACKLIST_DB_PATH) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS blacklisted_nodes (
                        node_id TEXT PRIMARY KEY,
                        reason TEXT NOT NULL,
                        blacklisted_at REAL NOT NULL,
                        expires_at REAL NOT NULL,
                        failure_count INTEGER DEFAULT 1
                    )
                """)
                # Clean up expired entries on startup
                conn.execute(
                    "DELETE FROM blacklisted_nodes WHERE expires_at < ?",
                    (time.time(),)
                )
                conn.commit()
        except (sqlite3.Error, OSError) as e:
            # Dec 2025: Narrowed to database and filesystem errors
            logger.warning(f"Failed to initialize blacklist DB: {type(e).__name__}: {e}")

    def _load_blacklist_from_db(self) -> None:
        """Load non-expired blacklist entries from SQLite."""
        try:
            with sqlite3.connect(self._BLACKLIST_DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM blacklisted_nodes WHERE expires_at > ?",
                    (time.time(),)
                ).fetchall()

                for row in rows:
                    self.blacklisted_nodes[row["node_id"]] = BlacklistedNode(
                        node_id=row["node_id"],
                        reason=row["reason"],
                        blacklisted_at=row["blacklisted_at"],
                        expires_at=row["expires_at"],
                        failure_count=row["failure_count"],
                    )

                if rows:
                    logger.info(f"Loaded {len(rows)} blacklisted nodes from persistence")
        except (sqlite3.Error, OSError, KeyError) as e:
            # Dec 2025: Narrowed to database, filesystem, and row access errors
            logger.warning(f"Failed to load blacklist from DB: {type(e).__name__}: {e}")

    def _persist_blacklist_entry(self, bl: BlacklistedNode) -> None:
        """Persist a single blacklist entry to SQLite."""
        try:
            with sqlite3.connect(self._BLACKLIST_DB_PATH) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO blacklisted_nodes
                    (node_id, reason, blacklisted_at, expires_at, failure_count)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    bl.node_id,
                    bl.reason,
                    bl.blacklisted_at,
                    bl.expires_at,
                    bl.failure_count,
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to persist blacklist entry: {e}")

    def _remove_blacklist_entry(self, node_id: str) -> None:
        """Remove a blacklist entry from SQLite."""
        try:
            with sqlite3.connect(self._BLACKLIST_DB_PATH) as conn:
                conn.execute(
                    "DELETE FROM blacklisted_nodes WHERE node_id = ?",
                    (node_id,)
                )
                conn.commit()
        except Exception as e:
            logger.debug(f"Failed to remove blacklist entry: {e}")

    def is_node_blacklisted(self, node_id: str) -> bool:
        """Check if a node is currently blacklisted."""
        if node_id not in self.blacklisted_nodes:
            return False

        bl = self.blacklisted_nodes[node_id]
        if time.time() > bl.expires_at:
            # Expired, remove from blacklist
            del self.blacklisted_nodes[node_id]
            # P0.4: Also remove from persistence
            self._remove_blacklist_entry(node_id)
            logger.info(f"Node {node_id} removed from blacklist (expired)")
            return False

        return True

    def blacklist_node(self, node_id: str, reason: str, duration: float = NODE_BLACKLIST_DURATION) -> None:
        """Add a node to the temporary blacklist.

        P0.4 Dec 2025: Now persists to SQLite so blacklist survives restart.
        """
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
            bl = BlacklistedNode(
                node_id=node_id,
                reason=reason,
                blacklisted_at=now,
                expires_at=now + duration,
            )
            self.blacklisted_nodes[node_id] = bl
            self.stats.nodes_blacklisted += 1
            logger.warning(f"Node {node_id} blacklisted for {duration}s: {reason}")

        # P0.4: Persist to SQLite
        self._persist_blacklist_entry(self.blacklisted_nodes[node_id])

    async def _get_timed_out_jobs(self) -> list[dict[str, Any]]:
        """Query work queue for jobs that have exceeded their timeout.

        December 29, 2025: Now uses adaptive timeouts based on job parameters.
        """
        timed_out = []

        try:
            # Get all running jobs
            running_jobs = self.work_queue.get_running_items()

            now = time.time()
            for job in running_jobs:
                job_type = job.get("work_type", "unknown")

                # December 29, 2025: Extract job parameters for adaptive timeout
                job_params = {
                    "num_games": job.get("num_games", job.get("games", 0)),
                    "num_samples": job.get("num_samples", job.get("samples", 0)),
                    "board_type": job.get("board_type", job.get("config", {}).get("board_type", "")),
                    "num_players": job.get("num_players", job.get("config", {}).get("num_players", 2)),
                }

                timeout = self.get_timeout_for_job(job_type, job_params)
                started_at = job.get("started_at", 0)

                if started_at and (now - started_at) > timeout:
                    job["timeout_duration"] = now - started_at
                    job["expected_timeout"] = timeout
                    job["adaptive_params"] = job_params  # For debugging
                    timed_out.append(job)

        except Exception as e:
            logger.error(f"Error getting timed out jobs: {e}")
            self.stats.errors_count += 1

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

            # P1.4 Dec 2025: FIRST mark as timeout in DB, THEN kill process
            # Previous order (kill then update) caused jobs to stay RUNNING forever
            # if reaper crashed between kill and DB update
            try:
                self.work_queue.timeout_work(job_id)
                self.stats.record_job_timeout()
                logger.info(f"Marked job {job_id} as TIMEOUT")

                # Emit WORK_TIMEOUT event for pipeline coordination
                from app.coordination.event_emission_helpers import safe_emit_event_async

                await safe_emit_event_async(
                    "WORK_TIMEOUT",
                    {
                        "work_id": job_id,
                        "work_type": job_type,
                        "node_id": node_id,
                        "duration_seconds": duration,
                        "expected_timeout": job.get("expected_timeout", 0),
                    },
                    source="JobReaperDaemon",
                    context="JobReaperDaemon.handle_timeout",
                )

            except Exception as e:
                logger.error(f"Error marking job {job_id} as timeout: {e}")
                self.stats.record_failure(e)

            # P1.4 Dec 2025: NOW kill the process (after DB is updated)
            # Even if kill fails, the job is already marked as TIMEOUT
            if pid and node_id:
                await self._kill_remote_process(node_id, pid)

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
                    self.stats.record_job_reassigned()
                    logger.info(
                        f"Reassigned job {work_id} for retry (attempt {attempts + 1}/{MAX_REASSIGN_ATTEMPTS}), "
                        f"excluding nodes: {excluded_nodes}"
                    )
                except Exception as e:
                    logger.error(f"Error reassigning job {work_id}: {e}")
                    self.stats.record_failure(e)

        except Exception as e:
            logger.error(f"Error in reassignment loop: {e}")
            self.stats.record_failure(e)

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

    async def run(self) -> None:
        """Main daemon loop.

        December 2025: Now includes P2P leader awareness. The daemon will:
        - Check leadership status before each reap cycle
        - Skip actions if not the leader
        - Wait shorter intervals when not leader to detect leadership changes
        """
        self.running = True
        logger.info("Job Reaper Daemon starting")

        while self.running:
            try:
                # Check if this node is the P2P leader (December 2025)
                is_leader, leader_id = await check_p2p_leader_status(timeout=LEADER_CHECK_TIMEOUT)
                self.stats.leader_checks += 1

                if not is_leader:
                    # Not the leader, skip this cycle
                    self.stats.not_leader_skips += 1
                    if self.stats.not_leader_skips % 10 == 1:
                        # Log occasionally to avoid spam
                        logger.debug(
                            f"Job Reaper skipping - not leader (leader: {leader_id}, "
                            f"skips: {self.stats.not_leader_skips})"
                        )
                    # Wait shorter interval to detect leadership changes faster
                    await asyncio.sleep(LEADER_RETRY_DELAY)
                    continue

                # We are the leader - proceed with reaping
                if self.stats.not_leader_skips > 0:
                    logger.info(
                        f"Job Reaper resuming as leader (was not leader for "
                        f"{self.stats.not_leader_skips} cycles)"
                    )
                    self.stats.not_leader_skips = 0

                self.stats.record_attempt()  # Updates last_check_time

                # Cleanup expired blacklists
                await self._cleanup_expired_blacklists()

                # Reap stuck jobs
                await self._reap_stuck_jobs()

                # Reassign failed work
                await self._reassign_failed_work()

                # December 29, 2025: Clean up orphaned work items (stale pending + claimed)
                # Jobs claimed >2h without progress are likely from crashed nodes
                # January 2026: Use centralized WorkQueueCleanupDefaults
                try:
                    cleanup_result = self.work_queue.cleanup_stale_items(
                        max_pending_age_hours=WorkQueueCleanupDefaults.MAX_PENDING_AGE_HOURS,
                        max_claimed_age_hours=WorkQueueCleanupDefaults.MAX_CLAIMED_AGE_HOURS,
                    )
                    if cleanup_result.get("removed_stale_pending", 0) > 0:
                        logger.info(
                            f"Cleaned up {cleanup_result['removed_stale_pending']} stale pending items"
                        )
                    if cleanup_result.get("reset_stale_claimed", 0) > 0:
                        logger.info(
                            f"Reset {cleanup_result['reset_stale_claimed']} stale claimed items"
                        )
                except Exception as cleanup_err:
                    logger.debug(f"Stale item cleanup failed: {cleanup_err}")

            except Exception as e:
                logger.error(f"Error in reaper loop: {e}")
                self.stats.record_failure(e)

            await asyncio.sleep(self.check_interval)

        logger.info("Job Reaper Daemon stopped")

    def stop(self) -> None:
        """Stop the daemon."""
        self.running = False

    def get_stats(self) -> dict[str, Any]:
        """Get daemon statistics.

        December 2025: Now uses base class to_dict() for consistency.
        """
        stats = self.stats.to_dict()
        # Add daemon-specific runtime info
        stats.update({
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
            "running": self.running,
        })
        return stats

    def health_check(self) -> "HealthCheckResult":
        """Check daemon health status.

        December 2025: Added to satisfy CoordinatorProtocol for unified health monitoring.
        """
        from app.coordination.protocols import HealthCheckResult, CoordinatorStatus

        if not self.running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="Job reaper daemon not running",
            )

        # Check error rate
        total_ops = self.stats.jobs_timed_out + self.stats.jobs_reassigned + self.stats.errors_count
        if total_ops > 0:
            error_rate = self.stats.errors_count / total_ops
            if error_rate > 0.5:
                return HealthCheckResult(
                    healthy=False,
                    status=CoordinatorStatus.DEGRADED,
                    message=f"Job reaper has high error rate: {error_rate:.1%}",
                    details=self.get_stats(),
                )

        # Check consecutive failures
        if self.stats.consecutive_failures > 5:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"Job reaper has {self.stats.consecutive_failures} consecutive failures",
                details=self.get_stats(),
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"Job reaper running (reaped: {self.stats.jobs_reaped}, reassigned: {self.stats.jobs_reassigned})",
            details=self.get_stats(),
        )


# =============================================================================
# Public Helpers
# =============================================================================

async def is_p2p_leader() -> bool:
    """Check if this node is the P2P cluster leader.

    This is a convenience function for other daemons that need to check
    leader status before performing leader-only operations.

    Returns:
        True if this node is the P2P leader, False otherwise.

    Usage:
        from app.coordination.job_reaper import is_p2p_leader

        if await is_p2p_leader():
            # Perform leader-only action
            ...

    December 2025: Now uses centralized check_p2p_leader_status from app.core.node
    """
    is_leader, _ = await check_p2p_leader_status()
    return is_leader


def get_node_id() -> str:
    """Get this node's ID (from RINGRIFT_NODE_ID or hostname).

    Returns:
        The node ID string.

    December 2025: Now uses centralized get_this_node_id from app.core.node
    """
    return get_this_node_id()


# =============================================================================
# HandlerBase Integration (January 2026)
# =============================================================================

try:
    from app.coordination.handler_base import HandlerBase
    from app.coordination.contracts import HealthCheckResult as HBHealthCheckResult

    HAS_HANDLER_BASE = True
except ImportError:
    HAS_HANDLER_BASE = False

if HAS_HANDLER_BASE:

    class JobReaperHandler(HandlerBase):
        """HandlerBase wrapper for JobReaperDaemon.

        January 2026: Added for unified daemon lifecycle management.
        The handler wraps the existing daemon and delegates to its methods.

        Note: This daemon only operates on the P2P leader node.
        """

        def __init__(self, work_queue: "WorkQueue | None" = None):
            super().__init__(
                name="job_reaper",
                cycle_interval=CHECK_INTERVAL,
            )
            # Lazy-load work queue if not provided
            if work_queue is None:
                try:
                    from app.coordination.work_queue import WorkQueue
                    work_queue = WorkQueue.get_instance()
                except (ImportError, RuntimeError):
                    work_queue = None  # Will fail gracefully in _run_cycle

            self._daemon = JobReaperDaemon(work_queue=work_queue) if work_queue else None

        async def _run_cycle(self) -> None:
            """Run one reaper cycle (only on P2P leader)."""
            if not self._daemon:
                logger.warning("[JobReaperHandler] No work queue available, skipping")
                return

            # Check if this node is the P2P leader
            is_leader, leader_id = await check_p2p_leader_status(timeout=LEADER_CHECK_TIMEOUT)
            self._daemon.stats.leader_checks += 1

            if not is_leader:
                self._daemon.stats.not_leader_skips += 1
                if self._daemon.stats.not_leader_skips % 10 == 1:
                    logger.debug(f"[JobReaperHandler] Skipping - not leader (leader: {leader_id})")
                return

            # We are the leader - proceed with reaping
            if self._daemon.stats.not_leader_skips > 0:
                logger.info(f"[JobReaperHandler] Resuming as leader")
                self._daemon.stats.not_leader_skips = 0

            self._daemon.stats.record_attempt()

            # Cleanup expired blacklists
            await self._daemon._cleanup_expired_blacklists()

            # Reap stuck jobs
            await self._daemon._reap_stuck_jobs()

            # Reassign failed work
            await self._daemon._reassign_failed_work()

            # Clean up orphaned work items
            try:
                cleanup_result = self._daemon.work_queue.cleanup_stale_items(
                    max_pending_age_hours=24.0,
                    max_claimed_age_hours=4.0,
                )
                if cleanup_result.get("removed_stale_pending", 0) > 0:
                    logger.info(f"Cleaned up {cleanup_result['removed_stale_pending']} stale pending items")
                if cleanup_result.get("reset_stale_claimed", 0) > 0:
                    logger.info(f"Reset {cleanup_result['reset_stale_claimed']} stale claimed items")
            except Exception as cleanup_err:
                logger.debug(f"Stale item cleanup failed: {cleanup_err}")

        def _get_event_subscriptions(self) -> dict:
            """Get event subscriptions for job reaper."""
            return {
                "JOB_STUCK": self._on_job_stuck,
                "NODE_FAILED": self._on_node_failed,
            }

        async def _on_job_stuck(self, event: dict) -> None:
            """Handle job stuck event - trigger immediate reap check."""
            if self._daemon:
                # Feb 2026: Extract payload from RouterEvent
                from app.coordination.event_router import get_event_payload
                payload = get_event_payload(event)
                job_id = payload.get("job_id", "unknown")
                logger.info(f"[JobReaperHandler] Triggered by JOB_STUCK event: {job_id}")
                # Run cycle will handle it

        async def _on_node_failed(self, event: dict) -> None:
            """Handle node failed event - blacklist the node."""
            if self._daemon:
                # Feb 2026: Extract payload from RouterEvent
                from app.coordination.event_router import get_event_payload
                payload = get_event_payload(event)
                node_id = payload.get("node_id", "")
                reason = payload.get("reason", "NODE_FAILED event")
                if node_id:
                    self._daemon.blacklist_node(node_id, reason)

        def health_check(self) -> HBHealthCheckResult:
            """Return health check result."""
            if not self._daemon:
                return HBHealthCheckResult(
                    healthy=False,
                    status="unavailable",
                    details={"error": "No work queue available"},
                )

            result = self._daemon.health_check()
            return HBHealthCheckResult(
                healthy=result.healthy,
                status=result.status.value if hasattr(result.status, 'value') else str(result.status),
                details=result.details or {},
            )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main class
    "JobReaperDaemon",
    # Data classes
    "ReaperStats",
    "BlacklistedNode",
    "ReaperAction",
    # Public helpers (December 2025)
    "is_p2p_leader",
    "get_node_id",
    # Constants
    "CHECK_INTERVAL",
    "DEFAULT_JOB_TIMEOUT",
    "MAX_REASSIGN_ATTEMPTS",
    "NODE_BLACKLIST_DURATION",
]

# Add HandlerBase wrapper to exports if available
if HAS_HANDLER_BASE:
    __all__.append("JobReaperHandler")
