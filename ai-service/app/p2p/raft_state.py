"""PySyncObj-based Replicated State Machines for P2P Coordination.

This module provides Raft-based replicated state machines for distributed
work queue and job assignment management. Uses PySyncObj for sub-second
leader failover, distributed locks, and automatic log compaction.

Key Classes:
- ReplicatedWorkQueue: Distributed work queue with atomic claim/complete
- ReplicatedJobAssignments: Tracks job-to-node assignments across cluster

Usage:
    from app.p2p.raft_state import ReplicatedWorkQueue, ReplicatedJobAssignments

    # Create work queue with Raft consensus
    queue = ReplicatedWorkQueue(
        node_id="nebius-h100-1",
        self_address="89.169.111.139:4321",
        partner_addresses=["89.169.112.47:4321", "208.167.249.164:4321"]
    )
    await queue.wait_ready()

    # Add work (replicated across cluster)
    queue.add_work("work-001", {"work_type": "selfplay", "board_type": "hex8"})

    # Claim work atomically
    success = queue.claim_work("work-001", "nebius-h100-1")

Phase 2.3 - Dec 26, 2025
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, ItemsView, KeysView, ValuesView

import yaml

logger = logging.getLogger(__name__)

# Dec 2025: Use cluster_config helpers instead of inline YAML parsing
try:
    from app.config.cluster_config import get_cluster_nodes, get_p2p_voters
    HAS_CLUSTER_CONFIG = True
except ImportError:
    HAS_CLUSTER_CONFIG = False
    get_cluster_nodes = None
    get_p2p_voters = None

# ============================================
# Import PySyncObj with graceful fallback
# ============================================

try:
    from pysyncobj import SyncObj, SyncObjConf, replicated
    from pysyncobj.batteries import ReplDict, ReplLockManager

    PYSYNCOBJ_AVAILABLE = True
except ImportError:
    PYSYNCOBJ_AVAILABLE = False
    logger.warning(
        "pysyncobj not installed. Install with: pip install pysyncobj\n"
        "Raft-based state replication will be unavailable."
    )

    # Define stub classes for type hints when PySyncObj is not available
    class SyncObj:  # type: ignore[no-redef]
        """Stub SyncObj when pysyncobj is not installed."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError("pysyncobj not installed")

    class SyncObjConf:  # type: ignore[no-redef]
        """Stub SyncObjConf when pysyncobj is not installed."""

        def __init__(self, *args, **kwargs):
            pass

    class ReplDict:  # type: ignore[no-redef]
        """Stub ReplDict when pysyncobj is not installed."""

        def __init__(self) -> None:
            self._data: dict[str, Any] = {}

        def __getitem__(self, key: str) -> Any:
            return self._data[key]

        def __setitem__(self, key: str, value: Any) -> None:
            self._data[key] = value

        def __contains__(self, key: str) -> bool:
            return key in self._data

        def get(self, key: str, default: Any = None) -> Any:
            return self._data.get(key, default)

        def items(self) -> ItemsView[str, Any]:
            return self._data.items()

        def keys(self) -> KeysView[str]:
            return self._data.keys()

        def values(self) -> ValuesView[Any]:
            return self._data.values()

    class ReplLockManager:  # type: ignore[no-redef]
        """Stub ReplLockManager when pysyncobj is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    def replicated(func: Callable[..., Any]) -> Callable[..., Any]:  # type: ignore[no-redef]
        """Stub replicated decorator."""
        return func


# ============================================
# Import constants from P2P
# ============================================

from app.p2p.constants import (
    RAFT_AUTO_UNLOCK_TIME,
    RAFT_BIND_PORT,
    RAFT_COMPACTION_MIN_ENTRIES,
    RAFT_ENABLED,
)


# ============================================
# Data Classes
# ============================================


@dataclass
class WorkItem:
    """A unit of work in the distributed queue.

    Mirrors app.coordination.work_queue.WorkItem structure for compatibility.
    """

    work_id: str
    work_type: str = "selfplay"
    priority: int = 50
    config: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    claimed_at: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0
    status: str = "pending"  # pending, claimed, running, completed, failed
    claimed_by: str = ""
    attempts: int = 0
    max_attempts: int = 3
    timeout_seconds: float = 3600.0
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkItem":
        """Create from dictionary."""
        return cls(**data)

    def is_claimable(self) -> bool:
        """Check if work can be claimed."""
        return self.status == "pending" and self.attempts < self.max_attempts

    def is_timed_out(self) -> bool:
        """Check if work has timed out."""
        if self.status not in ("claimed", "running"):
            return False
        if self.claimed_at == 0:
            return False
        return time.time() - self.claimed_at > self.timeout_seconds


@dataclass
class JobAssignment:
    """Job assignment record tracking which node runs which job."""

    job_id: str
    node_id: str
    job_type: str = "selfplay"
    board_type: str = "square8"
    num_players: int = 2
    assigned_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    status: str = "assigned"  # assigned, running, completed, failed
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobAssignment":
        """Create from dictionary."""
        return cls(**data)


# ============================================
# Configuration Helpers
# ============================================


def load_raft_partner_addresses(
    node_id: str,
    config_path: Path | None = None,
    bind_port: int = RAFT_BIND_PORT,
) -> list[str]:
    """Load Raft partner addresses from distributed_hosts.yaml.

    Uses voter nodes as Raft cluster members for consistency.

    Args:
        node_id: This node's ID (excluded from partners)
        config_path: Path to distributed_hosts.yaml
        bind_port: Raft bind port (default from constants)

    Returns:
        List of partner addresses in "host:port" format
    """
    if config_path is None:
        config_path = (
            Path(__file__).parent.parent.parent / "config" / "distributed_hosts.yaml"
        )

    partners: list[str] = []

    if not config_path.exists():
        logger.warning(f"Raft config not found: {config_path}")
        return partners

    try:
        with open(config_path) as f:
            hosts_config = yaml.safe_load(f)

        # Get voter list from p2p_voters key (single source of truth)
        voter_list = hosts_config.get("p2p_voters", [])
        hosts = hosts_config.get("hosts", {})

        for voter_name in voter_list:
            if voter_name == node_id:
                continue

            host_config = hosts.get(voter_name, {})
            ssh_host = host_config.get("ssh_host", "")

            if ssh_host and ssh_host not in ("localhost", "127.0.0.1"):
                partners.append(f"{ssh_host}:{bind_port}")

        logger.info(f"Raft partners from config: {partners}")

    except Exception as e:
        logger.warning(f"Could not load Raft partners from config: {e}")

    return partners


def get_self_raft_address(
    config_path: Path | None = None,
    bind_port: int = RAFT_BIND_PORT,
) -> str | None:
    """Get this node's Raft address from config.

    Args:
        config_path: Path to distributed_hosts.yaml
        bind_port: Raft bind port

    Returns:
        Self address in "host:port" format, or None if not determinable
    """
    import socket

    # Try to get hostname-based address first
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if ip and ip not in ("127.0.0.1", "localhost"):
            return f"{ip}:{bind_port}"
    except socket.error:
        pass

    # Fall back to first non-localhost interface
    try:
        import subprocess

        result = subprocess.run(
            ["hostname", "-I"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            ips = result.stdout.strip().split()
            for ip in ips:
                if ip and not ip.startswith("127."):
                    return f"{ip}:{bind_port}"
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return None


# ============================================
# ReplicatedWorkQueue
# ============================================


class ReplicatedWorkQueue(SyncObj):
    """Distributed work queue using Raft consensus.

    Features:
    - Atomic work claiming with distributed locks
    - Automatic leader election and failover
    - Log compaction for long-running clusters
    - Priority-based work scheduling

    All mutating operations (@replicated) are automatically replicated
    across the Raft cluster with strong consistency.
    """

    def __init__(
        self,
        node_id: str,
        self_address: str,
        partner_addresses: list[str],
        auto_unlock_time: float = RAFT_AUTO_UNLOCK_TIME,
        compaction_min_entries: int = RAFT_COMPACTION_MIN_ENTRIES,
        on_ready: Callable[[], None] | None = None,
        on_leader_change: Callable[[str | None], None] | None = None,
    ):
        """Initialize replicated work queue.

        Args:
            node_id: Unique identifier for this node
            self_address: This node's Raft address (host:port)
            partner_addresses: List of partner Raft addresses
            auto_unlock_time: Seconds before auto-releasing locks
            compaction_min_entries: Min log entries before compaction
            on_ready: Callback when cluster is ready
            on_leader_change: Callback on leader changes (receives leader address)
        """
        if not PYSYNCOBJ_AVAILABLE:
            raise RuntimeError(
                "pysyncobj not installed. Install with: pip install pysyncobj"
            )

        self.node_id = node_id
        self._on_ready_callback = on_ready
        self._on_leader_change_callback = on_leader_change
        self._is_ready = False

        # Replicated state
        self.__work_items = ReplDict()
        self.__claimed = ReplDict()  # work_id -> claimer_node_id
        self.__lock_manager = ReplLockManager(autoUnlockTime=auto_unlock_time)

        # Configure PySyncObj
        conf = SyncObjConf(
            autoTick=True,
            appendEntriesUseBatch=True,
            raftMinTimeout=0.5,
            raftMaxTimeout=2.0,
            commandsWaitLeader=True,
            compactionMinEntries=compaction_min_entries,
            fullDumpFile=f"/tmp/raft_work_queue_{node_id}.bin",
            journalFile=f"/tmp/raft_work_queue_{node_id}.journal",
            onReady=self._handle_ready,
            onLeaderChanged=self._handle_leader_change,
        )

        super().__init__(
            self_address,
            partner_addresses,
            conf,
            consumers=[self.__lock_manager],
        )

        logger.info(
            f"ReplicatedWorkQueue initialized: {self_address} -> {partner_addresses}"
        )

    def _handle_ready(self) -> None:
        """Handle cluster ready event."""
        self._is_ready = True
        leader = self.getLeader()
        logger.info(f"Raft cluster ready. Leader: {leader}")
        if self._on_ready_callback:
            try:
                self._on_ready_callback()
            except Exception as e:
                logger.error(f"Error in on_ready callback: {e}")

    def _handle_leader_change(self) -> None:
        """Handle leader change event."""
        leader = self.getLeader()
        logger.info(f"Raft leader changed to: {leader}")
        if self._on_leader_change_callback:
            try:
                self._on_leader_change_callback(leader)
            except Exception as e:
                logger.error(f"Error in on_leader_change callback: {e}")

    @property
    def is_ready(self) -> bool:
        """Check if cluster is ready for operations."""
        return self._is_ready

    @property
    def is_leader(self) -> bool:
        """Check if this node is the current leader."""
        return self._isLeader()

    @property
    def leader_address(self) -> str | None:
        """Get current leader's address."""
        return self.getLeader()

    # ========================================
    # Replicated (mutating) methods
    # ========================================

    @replicated
    def add_work(
        self,
        work_id: str,
        work_data: dict[str, Any],
    ) -> bool:
        """Add work item to the queue.

        Args:
            work_id: Unique work identifier
            work_data: Work item data (must include work_type, priority, etc.)

        Returns:
            True if added, False if work_id already exists
        """
        if work_id in self.__work_items:
            return False

        work_data["work_id"] = work_id
        work_data.setdefault("created_at", time.time())
        work_data.setdefault("status", "pending")
        work_data.setdefault("priority", 50)
        work_data.setdefault("attempts", 0)
        work_data.setdefault("max_attempts", 3)
        work_data.setdefault("timeout_seconds", 3600.0)

        self.__work_items[work_id] = work_data
        logger.debug(f"Raft: Added work {work_id}")
        return True

    @replicated
    def claim_work(
        self,
        work_id: str,
        claimer_node_id: str,
    ) -> bool:
        """Atomically claim a work item.

        Uses distributed lock to prevent race conditions.

        Args:
            work_id: Work item to claim
            claimer_node_id: Node claiming the work

        Returns:
            True if claimed successfully, False otherwise
        """
        # Check if work exists and is claimable
        work = self.__work_items.get(work_id)
        if not work:
            return False

        if work.get("status") != "pending":
            return False

        if work.get("attempts", 0) >= work.get("max_attempts", 3):
            return False

        # Acquire lock
        lock_name = f"work:{work_id}"
        if not self.__lock_manager.tryAcquire(lock_name, sync=True):
            return False

        try:
            # Double-check status under lock
            work = self.__work_items.get(work_id)
            if not work or work.get("status") != "pending":
                return False

            # Update work item
            work["status"] = "claimed"
            work["claimed_by"] = claimer_node_id
            work["claimed_at"] = time.time()
            work["attempts"] = work.get("attempts", 0) + 1
            self.__work_items[work_id] = work

            # Track claim
            self.__claimed[work_id] = claimer_node_id

            logger.debug(f"Raft: Work {work_id} claimed by {claimer_node_id}")
            return True

        finally:
            self.__lock_manager.release(lock_name)

    @replicated
    def complete_work(
        self,
        work_id: str,
        result: dict[str, Any] | None = None,
    ) -> bool:
        """Mark work as completed.

        Args:
            work_id: Work item to complete
            result: Optional result data

        Returns:
            True if completed, False if work not found or not claimed
        """
        work = self.__work_items.get(work_id)
        if not work:
            return False

        if work.get("status") not in ("claimed", "running"):
            return False

        work["status"] = "completed"
        work["completed_at"] = time.time()
        work["result"] = result or {}
        self.__work_items[work_id] = work

        # Clean up claim tracking
        if work_id in self.__claimed:
            del self.__claimed._data[work_id]

        logger.debug(f"Raft: Work {work_id} completed")
        return True

    @replicated
    def fail_work(
        self,
        work_id: str,
        error: str = "",
    ) -> bool:
        """Mark work as failed.

        If attempts < max_attempts, resets to pending for retry.

        Args:
            work_id: Work item that failed
            error: Error message

        Returns:
            True if processed, False if work not found
        """
        work = self.__work_items.get(work_id)
        if not work:
            return False

        attempts = work.get("attempts", 0)
        max_attempts = work.get("max_attempts", 3)

        if attempts < max_attempts:
            # Reset for retry
            work["status"] = "pending"
            work["claimed_by"] = ""
            work["claimed_at"] = 0.0
            work["error"] = error
        else:
            # Permanent failure
            work["status"] = "failed"
            work["completed_at"] = time.time()
            work["error"] = error

        self.__work_items[work_id] = work

        # Clean up claim tracking
        if work_id in self.__claimed:
            del self.__claimed._data[work_id]

        logger.debug(f"Raft: Work {work_id} failed (attempt {attempts}/{max_attempts})")
        return True

    @replicated
    def start_work(
        self,
        work_id: str,
    ) -> bool:
        """Mark claimed work as running.

        Args:
            work_id: Work item to start

        Returns:
            True if started, False if not found or not claimed
        """
        work = self.__work_items.get(work_id)
        if not work:
            return False

        if work.get("status") != "claimed":
            return False

        work["status"] = "running"
        work["started_at"] = time.time()
        self.__work_items[work_id] = work

        logger.debug(f"Raft: Work {work_id} started")
        return True

    @replicated
    def _remove_work(self, work_id: str) -> bool:
        """Remove a work item (internal use).

        Args:
            work_id: Work to remove

        Returns:
            True if removed, False if not found
        """
        if work_id not in self.__work_items:
            return False

        del self.__work_items._data[work_id]
        if work_id in self.__claimed:
            del self.__claimed._data[work_id]

        return True

    # ========================================
    # Local read methods (no replication needed)
    # ========================================

    def get_pending_work(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get pending work items sorted by priority.

        Args:
            limit: Maximum items to return

        Returns:
            List of pending work items (highest priority first)
        """
        pending = []
        for work_id, work in self.__work_items.items():
            if work.get("status") == "pending":
                pending.append(dict(work))

        # Sort by priority (descending)
        pending.sort(key=lambda x: -x.get("priority", 50))
        return pending[:limit]

    def get_claimed_work(self, node_id: str | None = None) -> list[dict[str, Any]]:
        """Get claimed/running work items.

        Args:
            node_id: Filter by claiming node (None for all)

        Returns:
            List of claimed/running work items
        """
        claimed = []
        for work_id, work in self.__work_items.items():
            if work.get("status") in ("claimed", "running"):
                if node_id is None or work.get("claimed_by") == node_id:
                    claimed.append(dict(work))
        return claimed

    def get_work(self, work_id: str) -> dict[str, Any] | None:
        """Get a specific work item.

        Args:
            work_id: Work item ID

        Returns:
            Work item data or None
        """
        work = self.__work_items.get(work_id)
        return dict(work) if work else None

    def get_queue_stats(self) -> dict[str, Any]:
        """Get queue statistics.

        Returns:
            Dict with pending, claimed, completed, failed counts
        """
        stats = {
            "pending": 0,
            "claimed": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "total": 0,
        }

        for work in self.__work_items.values():
            status = work.get("status", "pending")
            if status in stats:
                stats[status] += 1
            stats["total"] += 1

        stats["is_leader"] = self.is_leader
        stats["leader_address"] = self.leader_address
        stats["is_ready"] = self.is_ready

        return stats


# ============================================
# ReplicatedJobAssignments
# ============================================


class ReplicatedJobAssignments(SyncObj):
    """Distributed job assignments using Raft consensus.

    Tracks which jobs are assigned to which nodes, enabling:
    - Consistent view of cluster workload
    - Automatic reassignment on node failure
    - Load balancing decisions
    """

    def __init__(
        self,
        node_id: str,
        self_address: str,
        partner_addresses: list[str],
        compaction_min_entries: int = RAFT_COMPACTION_MIN_ENTRIES,
        on_ready: Callable[[], None] | None = None,
        on_leader_change: Callable[[str | None], None] | None = None,
    ):
        """Initialize replicated job assignments.

        Args:
            node_id: Unique identifier for this node
            self_address: This node's Raft address (host:port)
            partner_addresses: List of partner Raft addresses
            compaction_min_entries: Min log entries before compaction
            on_ready: Callback when cluster is ready
            on_leader_change: Callback on leader changes
        """
        if not PYSYNCOBJ_AVAILABLE:
            raise RuntimeError(
                "pysyncobj not installed. Install with: pip install pysyncobj"
            )

        self.node_id = node_id
        self._on_ready_callback = on_ready
        self._on_leader_change_callback = on_leader_change
        self._is_ready = False

        # Replicated state
        self.__assignments = ReplDict()  # job_id -> JobAssignment dict
        self.__node_jobs = ReplDict()  # node_id -> list[job_id]

        # Configure PySyncObj
        conf = SyncObjConf(
            autoTick=True,
            appendEntriesUseBatch=True,
            raftMinTimeout=0.5,
            raftMaxTimeout=2.0,
            commandsWaitLeader=True,
            compactionMinEntries=compaction_min_entries,
            fullDumpFile=f"/tmp/raft_job_assignments_{node_id}.bin",
            journalFile=f"/tmp/raft_job_assignments_{node_id}.journal",
            onReady=self._handle_ready,
            onLeaderChanged=self._handle_leader_change,
        )

        super().__init__(
            self_address,
            partner_addresses,
            conf,
        )

        logger.info(
            f"ReplicatedJobAssignments initialized: {self_address} -> {partner_addresses}"
        )

    def _handle_ready(self) -> None:
        """Handle cluster ready event."""
        self._is_ready = True
        leader = self.getLeader()
        logger.info(f"Job assignments Raft cluster ready. Leader: {leader}")
        if self._on_ready_callback:
            try:
                self._on_ready_callback()
            except Exception as e:
                logger.error(f"Error in on_ready callback: {e}")

    def _handle_leader_change(self) -> None:
        """Handle leader change event."""
        leader = self.getLeader()
        logger.info(f"Job assignments Raft leader changed to: {leader}")
        if self._on_leader_change_callback:
            try:
                self._on_leader_change_callback(leader)
            except Exception as e:
                logger.error(f"Error in on_leader_change callback: {e}")

    @property
    def is_ready(self) -> bool:
        """Check if cluster is ready."""
        return self._is_ready

    @property
    def is_leader(self) -> bool:
        """Check if this node is leader."""
        return self._isLeader()

    @property
    def leader_address(self) -> str | None:
        """Get current leader's address."""
        return self.getLeader()

    # ========================================
    # Replicated (mutating) methods
    # ========================================

    @replicated
    def assign_job(
        self,
        job_id: str,
        node_id: str,
        job_data: dict[str, Any],
    ) -> bool:
        """Assign a job to a node.

        Args:
            job_id: Unique job identifier
            node_id: Node to assign the job to
            job_data: Job configuration (job_type, board_type, etc.)

        Returns:
            True if assigned, False if job_id already exists
        """
        if job_id in self.__assignments:
            return False

        assignment = {
            "job_id": job_id,
            "node_id": node_id,
            "assigned_at": time.time(),
            "status": "assigned",
            **job_data,
        }
        self.__assignments[job_id] = assignment

        # Track by node
        node_jobs = list(self.__node_jobs.get(node_id, []))
        node_jobs.append(job_id)
        self.__node_jobs[node_id] = node_jobs

        logger.debug(f"Raft: Job {job_id} assigned to {node_id}")
        return True

    @replicated
    def start_job(self, job_id: str) -> bool:
        """Mark job as running.

        Args:
            job_id: Job to start

        Returns:
            True if started, False if not found or not assigned
        """
        assignment = self.__assignments.get(job_id)
        if not assignment:
            return False

        if assignment.get("status") != "assigned":
            return False

        assignment["status"] = "running"
        assignment["started_at"] = time.time()
        self.__assignments[job_id] = assignment

        logger.debug(f"Raft: Job {job_id} started")
        return True

    @replicated
    def complete_job(
        self,
        job_id: str,
        result: dict[str, Any] | None = None,
    ) -> bool:
        """Mark job as completed.

        Args:
            job_id: Job to complete
            result: Optional result data

        Returns:
            True if completed, False if not found
        """
        assignment = self.__assignments.get(job_id)
        if not assignment:
            return False

        assignment["status"] = "completed"
        assignment["completed_at"] = time.time()
        assignment["result"] = result or {}
        self.__assignments[job_id] = assignment

        # Remove from node's active jobs
        node_id = assignment.get("node_id")
        if node_id and node_id in self.__node_jobs:
            node_jobs = list(self.__node_jobs.get(node_id, []))
            if job_id in node_jobs:
                node_jobs.remove(job_id)
                self.__node_jobs[node_id] = node_jobs

        logger.debug(f"Raft: Job {job_id} completed")
        return True

    @replicated
    def fail_job(
        self,
        job_id: str,
        error: str = "",
    ) -> bool:
        """Mark job as failed.

        Args:
            job_id: Job that failed
            error: Error message

        Returns:
            True if marked failed, False if not found
        """
        assignment = self.__assignments.get(job_id)
        if not assignment:
            return False

        assignment["status"] = "failed"
        assignment["completed_at"] = time.time()
        assignment["error"] = error
        self.__assignments[job_id] = assignment

        # Remove from node's active jobs
        node_id = assignment.get("node_id")
        if node_id and node_id in self.__node_jobs:
            node_jobs = list(self.__node_jobs.get(node_id, []))
            if job_id in node_jobs:
                node_jobs.remove(job_id)
                self.__node_jobs[node_id] = node_jobs

        logger.debug(f"Raft: Job {job_id} failed: {error}")
        return True

    @replicated
    def _remove_assignment(self, job_id: str) -> bool:
        """Remove a job assignment (internal use).

        Args:
            job_id: Job to remove

        Returns:
            True if removed, False if not found
        """
        assignment = self.__assignments.get(job_id)
        if not assignment:
            return False

        # Remove from node tracking
        node_id = assignment.get("node_id")
        if node_id and node_id in self.__node_jobs:
            node_jobs = list(self.__node_jobs.get(node_id, []))
            if job_id in node_jobs:
                node_jobs.remove(job_id)
                self.__node_jobs[node_id] = node_jobs

        del self.__assignments._data[job_id]
        return True

    # ========================================
    # Local read methods (no replication needed)
    # ========================================

    def get_node_jobs(self, node_id: str) -> list[dict[str, Any]]:
        """Get all jobs assigned to a node.

        Args:
            node_id: Node to query

        Returns:
            List of job assignments for the node
        """
        job_ids = self.__node_jobs.get(node_id, [])
        jobs = []
        for job_id in job_ids:
            assignment = self.__assignments.get(job_id)
            if assignment:
                jobs.append(dict(assignment))
        return jobs

    def get_all_assignments(self) -> dict[str, dict[str, Any]]:
        """Get all job assignments.

        Returns:
            Dict mapping job_id to assignment data
        """
        return {job_id: dict(data) for job_id, data in self.__assignments.items()}

    def get_assignment(self, job_id: str) -> dict[str, Any] | None:
        """Get a specific job assignment.

        Args:
            job_id: Job ID to query

        Returns:
            Assignment data or None
        """
        assignment = self.__assignments.get(job_id)
        return dict(assignment) if assignment else None

    def get_active_jobs_count(self, node_id: str | None = None) -> int:
        """Get count of active (assigned/running) jobs.

        Args:
            node_id: Filter by node (None for all nodes)

        Returns:
            Count of active jobs
        """
        count = 0
        for assignment in self.__assignments.values():
            if assignment.get("status") in ("assigned", "running"):
                if node_id is None or assignment.get("node_id") == node_id:
                    count += 1
        return count

    def get_assignment_stats(self) -> dict[str, Any]:
        """Get assignment statistics.

        Returns:
            Dict with counts by status and per-node breakdowns
        """
        stats = {
            "assigned": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "total": 0,
            "by_node": {},
        }

        for assignment in self.__assignments.values():
            status = assignment.get("status", "assigned")
            if status in stats:
                stats[status] += 1
            stats["total"] += 1

            node_id = assignment.get("node_id", "unknown")
            if node_id not in stats["by_node"]:
                stats["by_node"][node_id] = {"active": 0, "total": 0}
            stats["by_node"][node_id]["total"] += 1
            if status in ("assigned", "running"):
                stats["by_node"][node_id]["active"] += 1

        stats["is_leader"] = self.is_leader
        stats["leader_address"] = self.leader_address
        stats["is_ready"] = self.is_ready

        return stats


# ============================================
# Factory Functions
# ============================================


def create_replicated_work_queue(
    node_id: str,
    config_path: Path | None = None,
    bind_port: int = RAFT_BIND_PORT,
    on_ready: Callable[[], None] | None = None,
    on_leader_change: Callable[[str | None], None] | None = None,
) -> ReplicatedWorkQueue | None:
    """Create a ReplicatedWorkQueue with auto-configured addresses.

    Args:
        node_id: This node's unique identifier
        config_path: Path to distributed_hosts.yaml
        bind_port: Raft bind port
        on_ready: Callback when cluster is ready
        on_leader_change: Callback on leader changes

    Returns:
        ReplicatedWorkQueue instance, or None if Raft not available/enabled
    """
    if not PYSYNCOBJ_AVAILABLE:
        logger.warning("Cannot create ReplicatedWorkQueue: pysyncobj not installed")
        return None

    if not RAFT_ENABLED:
        logger.info("Raft is disabled (RINGRIFT_RAFT_ENABLED=false)")
        return None

    self_address = get_self_raft_address(config_path, bind_port)
    if not self_address:
        logger.error("Cannot determine self Raft address")
        return None

    partners = load_raft_partner_addresses(node_id, config_path, bind_port)
    if not partners:
        logger.warning("No Raft partners configured - running in single-node mode")

    return ReplicatedWorkQueue(
        node_id=node_id,
        self_address=self_address,
        partner_addresses=partners,
        on_ready=on_ready,
        on_leader_change=on_leader_change,
    )


def create_replicated_job_assignments(
    node_id: str,
    config_path: Path | None = None,
    bind_port: int = RAFT_BIND_PORT,
    on_ready: Callable[[], None] | None = None,
    on_leader_change: Callable[[str | None], None] | None = None,
) -> ReplicatedJobAssignments | None:
    """Create a ReplicatedJobAssignments with auto-configured addresses.

    Args:
        node_id: This node's unique identifier
        config_path: Path to distributed_hosts.yaml
        bind_port: Raft bind port (uses different offset for job assignments)
        on_ready: Callback when cluster is ready
        on_leader_change: Callback on leader changes

    Returns:
        ReplicatedJobAssignments instance, or None if Raft not available/enabled
    """
    if not PYSYNCOBJ_AVAILABLE:
        logger.warning(
            "Cannot create ReplicatedJobAssignments: pysyncobj not installed"
        )
        return None

    if not RAFT_ENABLED:
        logger.info("Raft is disabled (RINGRIFT_RAFT_ENABLED=false)")
        return None

    # Use port offset to avoid conflict with work queue
    assignment_port = bind_port + 1

    self_address = get_self_raft_address(config_path, assignment_port)
    if not self_address:
        logger.error("Cannot determine self Raft address for job assignments")
        return None

    partners = load_raft_partner_addresses(node_id, config_path, assignment_port)
    if not partners:
        logger.warning(
            "No Raft partners configured for job assignments - single-node mode"
        )

    return ReplicatedJobAssignments(
        node_id=node_id,
        self_address=self_address,
        partner_addresses=partners,
        on_ready=on_ready,
        on_leader_change=on_leader_change,
    )


# ============================================
# Module exports
# ============================================

__all__ = [
    # Availability flag
    "PYSYNCOBJ_AVAILABLE",
    # Constants
    "RAFT_ENABLED",
    "RAFT_BIND_PORT",
    "RAFT_AUTO_UNLOCK_TIME",
    "RAFT_COMPACTION_MIN_ENTRIES",
    # Data classes
    "WorkItem",
    "JobAssignment",
    # Main classes
    "ReplicatedWorkQueue",
    "ReplicatedJobAssignments",
    # Factory functions
    "create_replicated_work_queue",
    "create_replicated_job_assignments",
    # Configuration helpers
    "load_raft_partner_addresses",
    "get_self_raft_address",
]
