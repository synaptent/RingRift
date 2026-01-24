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
    """Load Raft partner addresses from cluster configuration.

    Uses voter nodes as Raft cluster members for consistency.

    Args:
        node_id: This node's ID (excluded from partners)
        config_path: Path to distributed_hosts.yaml (fallback only)
        bind_port: Raft bind port (default from constants)

    Returns:
        List of partner addresses in "host:port" format
    """
    partners: list[str] = []

    # Dec 2025: Use cluster_config helpers as primary method
    if HAS_CLUSTER_CONFIG and get_p2p_voters is not None:
        try:
            voters = get_p2p_voters()
            nodes = get_cluster_nodes()

            for voter_name in voters:
                if voter_name == node_id:
                    continue

                if voter_name in nodes:
                    node = nodes[voter_name]
                    ssh_host = node.best_ip
                    if ssh_host and ssh_host not in ("localhost", "127.0.0.1"):
                        partners.append(f"{ssh_host}:{bind_port}")

            if partners:
                logger.info(f"Raft partners from cluster_config: {partners}")
                return partners

        except Exception as e:
            logger.warning(f"Could not load Raft partners from cluster_config: {e}")

    # Fallback: Load from YAML directly if cluster_config unavailable
    if config_path is None:
        config_path = (
            Path(__file__).parent.parent.parent / "config" / "distributed_hosts.yaml"
        )

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

        logger.info(f"Raft partners from YAML fallback: {partners}")

    except Exception as e:
        logger.warning(f"Could not load Raft partners from YAML: {e}")

    return partners


def get_self_raft_address(
    config_path: Path | None = None,
    bind_port: int = RAFT_BIND_PORT,
) -> str | None:
    """Get this node's Raft address from config.

    Jan 24, 2026: BLOCKING SUBPROCESS CALLS REMOVED for P2P stability.
    Previous subprocess calls (hostname -I, ifconfig) could block the event
    loop for up to 15 seconds when network interfaces were slow to respond.

    Now uses only non-blocking methods:
    1. RINGRIFT_ADVERTISE_HOST environment variable (RECOMMENDED for Raft)
    2. cluster_config tailscale_ip lookup
    3. hostname-based socket lookup (usually fast)

    For Raft to work reliably, set RINGRIFT_ADVERTISE_HOST in your environment
    or ensure your node is configured in distributed_hosts.yaml with tailscale_ip.

    Args:
        config_path: Path to distributed_hosts.yaml
        bind_port: Raft bind port

    Returns:
        Self address in "host:port" format, or None if not determinable.
        If None is returned, Raft should NOT be started as it won't be able
        to communicate with other nodes.
    """
    import os
    import socket

    # Method 1: Check environment variable (PREFERRED - non-blocking)
    # This is the most reliable method and should be set for Raft nodes
    advertise_host = os.environ.get("RINGRIFT_ADVERTISE_HOST")
    if advertise_host and advertise_host not in ("127.0.0.1", "localhost"):
        logger.debug(f"Using RINGRIFT_ADVERTISE_HOST for Raft: {advertise_host}")
        return f"{advertise_host}:{bind_port}"

    # Method 2: Get from cluster_config using node_id (non-blocking)
    if HAS_CLUSTER_CONFIG and get_cluster_nodes is not None:
        try:
            node_id = os.environ.get("RINGRIFT_NODE_ID")
            if not node_id:
                node_id = socket.gethostname().lower().replace(".", "-")

            nodes = get_cluster_nodes()
            for node in nodes:
                if node.name == node_id:
                    ip = node.tailscale_ip or node.ssh_host
                    if ip and ip not in ("127.0.0.1", "localhost"):
                        logger.debug(f"Using cluster_config for Raft: {ip}")
                        return f"{ip}:{bind_port}"
        except Exception as e:
            logger.debug(f"cluster_config lookup failed: {e}")

    # Method 3: Try to get hostname-based address (usually fast, non-blocking)
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if ip and ip not in ("127.0.0.1", "localhost"):
            logger.debug(f"Using hostname-based lookup for Raft: {ip}")
            return f"{ip}:{bind_port}"
    except socket.error:
        pass

    # REMOVED: Blocking subprocess calls (hostname -I, ifconfig)
    # These could block for 5-15 seconds and cause event loop stalls.
    # If you need Raft, set RINGRIFT_ADVERTISE_HOST or configure your node
    # in distributed_hosts.yaml with a tailscale_ip.

    logger.warning(
        "Could not determine Raft address. Raft will be disabled. "
        "To enable Raft, set RINGRIFT_ADVERTISE_HOST environment variable "
        "or configure tailscale_ip in distributed_hosts.yaml for this node."
    )
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

        # NOTE: Do NOT pass lock_manager as consumer - it contains threading.Lock
        # objects that cannot be pickled during Raft log compaction.
        # Each node will have its own local lock manager instance.
        # January 2026 fix: Removed consumers=[self.__lock_manager] to prevent
        # TypeError: cannot pickle '_thread.lock' object
        super().__init__(
            self_address,
            partner_addresses,
            conf,
        )

        logger.info(
            f"ReplicatedWorkQueue initialized: {self_address} -> {partner_addresses}"
        )

        # Store config for recreation after unpickling (January 2026 fix)
        self._auto_unlock_time = auto_unlock_time

    def __getstate__(self) -> dict:
        """Custom pickling to exclude unpicklable objects.

        The ReplLockManager contains threading objects that cannot be pickled.
        We exclude it and recreate it in __setstate__.

        January 2026: Fix for 'cannot pickle _thread.lock' error.
        """
        state = self.__dict__.copy()
        # Remove unpicklable attributes
        state.pop('_ReplicatedWorkQueue__lock_manager', None)
        state.pop('_on_ready_callback', None)
        state.pop('_on_leader_change_callback', None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Custom unpickling to recreate the lock manager.

        January 2026: Fix for 'cannot pickle _thread.lock' error.
        """
        self.__dict__.update(state)
        # Recreate the lock manager
        auto_unlock_time = getattr(self, '_auto_unlock_time', RAFT_AUTO_UNLOCK_TIME)
        self._ReplicatedWorkQueue__lock_manager = ReplLockManager(
            autoUnlockTime=auto_unlock_time
        )
        # Reset callbacks to None - they'll need to be set again if needed
        self._on_ready_callback = None
        self._on_leader_change_callback = None

    def _handle_ready(self) -> None:
        """Handle cluster ready event."""
        self._is_ready = True
        leader = self._getLeader()
        logger.info(f"Raft cluster ready. Leader: {leader}")
        if self._on_ready_callback:
            try:
                self._on_ready_callback()
            except Exception as e:
                logger.error(f"Error in on_ready callback: {e}")

    def _handle_leader_change(self) -> None:
        """Handle leader change event."""
        leader = self._getLeader()
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
        return self._getLeader()

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

    def __getstate__(self) -> dict:
        """Custom pickling to exclude unpicklable callback objects.

        January 2026: Fix for potential pickle errors with callbacks.
        """
        state = self.__dict__.copy()
        # Remove unpicklable callback attributes
        state.pop('_on_ready_callback', None)
        state.pop('_on_leader_change_callback', None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Custom unpickling to restore default callback state.

        January 2026: Fix for potential pickle errors with callbacks.
        """
        self.__dict__.update(state)
        # Reset callbacks to None - they'll need to be set again if needed
        self._on_ready_callback = None
        self._on_leader_change_callback = None

    def _handle_ready(self) -> None:
        """Handle cluster ready event."""
        self._is_ready = True
        leader = self._getLeader()
        logger.info(f"Job assignments Raft cluster ready. Leader: {leader}")
        if self._on_ready_callback:
            try:
                self._on_ready_callback()
            except Exception as e:
                logger.error(f"Error in on_ready callback: {e}")

    def _handle_leader_change(self) -> None:
        """Handle leader change event."""
        leader = self._getLeader()
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
        return self._getLeader()

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
# ReplicatedEloStore
# ============================================


@dataclass
class EloRatingEntry:
    """Elo rating entry for Raft replication.

    Mirrors app.training.elo_service.EloRating structure for compatibility.
    """

    participant_id: str
    board_type: str
    num_players: int
    rating: float = 1500.0
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    peak_rating: float = 1500.0
    last_update: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EloRatingEntry":
        """Create from dictionary."""
        return cls(**data)

    @property
    def key(self) -> str:
        """Get the composite key for this rating."""
        return f"{self.participant_id}:{self.board_type}:{self.num_players}"


@dataclass
class EloMatchEntry:
    """Match result entry for Raft replication."""

    match_id: str
    participant_ids: list[str]
    winner_id: str | None  # None for draw
    board_type: str
    num_players: int
    game_length: int = 0
    duration_sec: float = 0.0
    timestamp: float = field(default_factory=time.time)
    elo_before: dict[str, float] = field(default_factory=dict)
    elo_after: dict[str, float] = field(default_factory=dict)
    elo_changes: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EloMatchEntry":
        """Create from dictionary."""
        return cls(**data)


class ReplicatedEloStore(SyncObj):
    """Distributed Elo rating store using Raft consensus.

    Provides strong consistency for Elo ratings across the cluster.
    Features:
    - Atomic rating updates
    - Match history replication
    - Leader-based writes with follower reads
    - Automatic log compaction

    December 30, 2025 - P5.2: ELO Strong Consistency
    """

    # Baseline Elo rating (anchored to prevent rating inflation)
    BASELINE_ELO_RANDOM = 400.0
    INITIAL_ELO = 1500.0
    K_FACTOR = 32.0

    def __init__(
        self,
        node_id: str,
        self_address: str,
        partner_addresses: list[str],
        compaction_min_entries: int = RAFT_COMPACTION_MIN_ENTRIES,
        on_ready: Callable[[], None] | None = None,
        on_leader_change: Callable[[str | None], None] | None = None,
    ):
        """Initialize replicated Elo store.

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
        # Key: "{participant_id}:{board_type}:{num_players}"
        self.__ratings = ReplDict()
        # Match history: match_id -> match data (limited to recent N matches)
        self.__recent_matches = ReplDict()
        self.__match_count = 0
        self._max_recent_matches = 10000  # Keep last 10K matches in Raft

        # Configure PySyncObj
        conf = SyncObjConf(
            autoTick=True,
            appendEntriesUseBatch=True,
            raftMinTimeout=0.5,
            raftMaxTimeout=2.0,
            commandsWaitLeader=True,
            compactionMinEntries=compaction_min_entries,
            fullDumpFile=f"/tmp/raft_elo_store_{node_id}.bin",
            journalFile=f"/tmp/raft_elo_store_{node_id}.journal",
            onReady=self._handle_ready,
            onLeaderChanged=self._handle_leader_change,
        )

        super().__init__(
            self_address,
            partner_addresses,
            conf,
        )

        logger.info(
            f"ReplicatedEloStore initialized: {self_address} -> {partner_addresses}"
        )

    def __getstate__(self) -> dict:
        """Custom pickling to exclude unpicklable callback objects.

        January 2026: Fix for potential pickle errors with callbacks.
        """
        state = self.__dict__.copy()
        # Remove unpicklable callback attributes
        state.pop('_on_ready_callback', None)
        state.pop('_on_leader_change_callback', None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Custom unpickling to restore default callback state.

        January 2026: Fix for potential pickle errors with callbacks.
        """
        self.__dict__.update(state)
        # Reset callbacks to None - they'll need to be set again if needed
        self._on_ready_callback = None
        self._on_leader_change_callback = None

    def _handle_ready(self) -> None:
        """Handle cluster ready event."""
        self._is_ready = True
        leader = self._getLeader()
        logger.info(f"Elo store Raft cluster ready. Leader: {leader}")
        if self._on_ready_callback:
            try:
                self._on_ready_callback()
            except Exception as e:
                logger.error(f"Error in on_ready callback: {e}")

    def _handle_leader_change(self) -> None:
        """Handle leader change event."""
        leader = self._getLeader()
        logger.info(f"Elo store Raft leader changed to: {leader}")
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
        return self._getLeader()

    def _make_key(
        self, participant_id: str, board_type: str, num_players: int
    ) -> str:
        """Create composite key for rating lookup."""
        return f"{participant_id}:{board_type}:{num_players}"

    def _is_random_participant(self, participant_id: str) -> bool:
        """Check if participant is a random baseline that should be anchored."""
        pid_lower = participant_id.lower()
        if pid_lower.startswith("none:random"):
            return True
        if pid_lower in ("random", "baseline_random", "tier1_random"):
            return True
        if "random" in pid_lower and not any(
            x in pid_lower for x in ("heuristic", "minimax", "mcts", "descent", "neural")
        ):
            return True
        return False

    # ========================================
    # Replicated (mutating) methods
    # ========================================

    @replicated
    def update_rating(
        self,
        participant_id: str,
        board_type: str,
        num_players: int,
        rating: float,
        games_played: int,
        wins: int,
        losses: int,
        draws: int,
        peak_rating: float | None = None,
    ) -> bool:
        """Update or create a rating entry.

        Args:
            participant_id: Participant ID
            board_type: Board type (e.g., 'square8', 'hex8')
            num_players: Number of players (2, 3, or 4)
            rating: Current Elo rating
            games_played: Total games played
            wins: Total wins
            losses: Total losses
            draws: Total draws
            peak_rating: Peak rating (optional, defaults to current rating)

        Returns:
            True on success
        """
        key = self._make_key(participant_id, board_type, num_players)

        # Anchor random participants at fixed Elo
        if self._is_random_participant(participant_id):
            rating = self.BASELINE_ELO_RANDOM

        # Get existing peak rating
        existing = self.__ratings.get(key)
        if existing and peak_rating is None:
            peak_rating = max(existing.get("peak_rating", rating), rating)
        elif peak_rating is None:
            peak_rating = rating

        entry = {
            "participant_id": participant_id,
            "board_type": board_type,
            "num_players": num_players,
            "rating": rating,
            "games_played": games_played,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "peak_rating": peak_rating,
            "last_update": time.time(),
        }

        self.__ratings[key] = entry
        logger.debug(f"Raft: Updated rating for {key}: {rating:.1f}")
        return True

    @replicated
    def record_match(
        self,
        match_id: str,
        participant_a: str,
        participant_b: str,
        winner_id: str | None,
        board_type: str,
        num_players: int,
        game_length: int = 0,
        duration_sec: float = 0.0,
        k_factor: float = 32.0,
    ) -> dict[str, Any]:
        """Record a match and update ratings atomically.

        Args:
            match_id: Unique match identifier
            participant_a: First participant ID
            participant_b: Second participant ID
            winner_id: Winner ID (None for draw)
            board_type: Board type
            num_players: Number of players
            game_length: Number of moves
            duration_sec: Duration in seconds
            k_factor: K-factor for rating update

        Returns:
            Dict with match_id, elo_before, elo_after, elo_changes
        """
        import math

        # Get current ratings (create defaults if needed)
        key_a = self._make_key(participant_a, board_type, num_players)
        key_b = self._make_key(participant_b, board_type, num_players)

        rating_a = self.__ratings.get(key_a, {
            "participant_id": participant_a,
            "board_type": board_type,
            "num_players": num_players,
            "rating": self.INITIAL_ELO,
            "games_played": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "peak_rating": self.INITIAL_ELO,
            "last_update": time.time(),
        })
        rating_b = self.__ratings.get(key_b, {
            "participant_id": participant_b,
            "board_type": board_type,
            "num_players": num_players,
            "rating": self.INITIAL_ELO,
            "games_played": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "peak_rating": self.INITIAL_ELO,
            "last_update": time.time(),
        })

        old_elo_a = rating_a["rating"]
        old_elo_b = rating_b["rating"]
        elo_before = {participant_a: old_elo_a, participant_b: old_elo_b}

        # Calculate expected scores
        exp_a = 1.0 / (1.0 + math.pow(10, (old_elo_b - old_elo_a) / 400))
        exp_b = 1.0 - exp_a

        # Actual scores
        if winner_id == participant_a:
            score_a, score_b = 1.0, 0.0
            win_a, loss_a, draw_a = 1, 0, 0
            win_b, loss_b, draw_b = 0, 1, 0
        elif winner_id == participant_b:
            score_a, score_b = 0.0, 1.0
            win_a, loss_a, draw_a = 0, 1, 0
            win_b, loss_b, draw_b = 1, 0, 0
        else:
            score_a, score_b = 0.5, 0.5
            win_a, loss_a, draw_a = 0, 0, 1
            win_b, loss_b, draw_b = 0, 0, 1

        # Scale K-factor for multiplayer
        base_k = k_factor / (num_players - 1) if num_players > 2 else k_factor

        # Calculate changes
        change_a = base_k * (score_a - exp_a)
        change_b = base_k * (score_b - exp_b)

        new_elo_a = old_elo_a + change_a
        new_elo_b = old_elo_b + change_b

        # Anchor random participants
        if self._is_random_participant(participant_a):
            new_elo_a = self.BASELINE_ELO_RANDOM
            change_a = 0.0
        if self._is_random_participant(participant_b):
            new_elo_b = self.BASELINE_ELO_RANDOM
            change_b = 0.0

        elo_after = {participant_a: new_elo_a, participant_b: new_elo_b}
        elo_changes = {participant_a: change_a, participant_b: change_b}

        # Update ratings
        now = time.time()
        rating_a.update({
            "rating": new_elo_a,
            "games_played": rating_a.get("games_played", 0) + 1,
            "wins": rating_a.get("wins", 0) + win_a,
            "losses": rating_a.get("losses", 0) + loss_a,
            "draws": rating_a.get("draws", 0) + draw_a,
            "peak_rating": max(rating_a.get("peak_rating", new_elo_a), new_elo_a),
            "last_update": now,
        })
        rating_b.update({
            "rating": new_elo_b,
            "games_played": rating_b.get("games_played", 0) + 1,
            "wins": rating_b.get("wins", 0) + win_b,
            "losses": rating_b.get("losses", 0) + loss_b,
            "draws": rating_b.get("draws", 0) + draw_b,
            "peak_rating": max(rating_b.get("peak_rating", new_elo_b), new_elo_b),
            "last_update": now,
        })

        self.__ratings[key_a] = rating_a
        self.__ratings[key_b] = rating_b

        # Record match (limited history)
        match_entry = {
            "match_id": match_id,
            "participant_ids": [participant_a, participant_b],
            "winner_id": winner_id,
            "board_type": board_type,
            "num_players": num_players,
            "game_length": game_length,
            "duration_sec": duration_sec,
            "timestamp": now,
            "elo_before": elo_before,
            "elo_after": elo_after,
            "elo_changes": elo_changes,
        }
        self.__recent_matches[match_id] = match_entry
        self.__match_count += 1

        # Clean up old matches if exceeding limit
        if self.__match_count > self._max_recent_matches * 1.5:
            self._cleanup_old_matches()

        logger.debug(f"Raft: Recorded match {match_id}, changes: {elo_changes}")
        return {
            "match_id": match_id,
            "elo_before": elo_before,
            "elo_after": elo_after,
            "elo_changes": elo_changes,
        }

    def _cleanup_old_matches(self) -> None:
        """Remove oldest matches to stay within limit (internal, not replicated)."""
        # Sort matches by timestamp
        matches = list(self.__recent_matches.items())
        if len(matches) <= self._max_recent_matches:
            return

        matches.sort(key=lambda x: x[1].get("timestamp", 0))
        remove_count = len(matches) - self._max_recent_matches

        for i in range(remove_count):
            match_id = matches[i][0]
            if match_id in self.__recent_matches._data:
                del self.__recent_matches._data[match_id]

        self.__match_count = len(self.__recent_matches._data)

    # ========================================
    # Local read methods (no replication needed)
    # ========================================

    def get_rating(
        self,
        participant_id: str,
        board_type: str,
        num_players: int,
    ) -> dict[str, Any] | None:
        """Get rating for a participant.

        Args:
            participant_id: Participant ID
            board_type: Board type
            num_players: Number of players

        Returns:
            Rating data dict or None if not found
        """
        key = self._make_key(participant_id, board_type, num_players)
        rating = self.__ratings.get(key)
        if rating:
            # Always return anchored rating for random participants
            if self._is_random_participant(participant_id):
                rating = dict(rating)
                rating["rating"] = self.BASELINE_ELO_RANDOM
            return dict(rating)
        return None

    def get_ratings_batch(
        self,
        participant_ids: list[str],
        board_type: str,
        num_players: int,
    ) -> dict[str, dict[str, Any]]:
        """Get ratings for multiple participants.

        Args:
            participant_ids: List of participant IDs
            board_type: Board type
            num_players: Number of players

        Returns:
            Dict mapping participant_id to rating data
        """
        result = {}
        for pid in participant_ids:
            rating = self.get_rating(pid, board_type, num_players)
            if rating:
                result[pid] = rating
        return result

    def get_leaderboard(
        self,
        board_type: str,
        num_players: int,
        limit: int = 50,
        min_games: int = 0,
    ) -> list[dict[str, Any]]:
        """Get leaderboard for a configuration.

        Args:
            board_type: Board type
            num_players: Number of players
            limit: Maximum entries
            min_games: Minimum games required

        Returns:
            List of rating entries sorted by rating (descending)
        """
        entries = []
        for key, rating in self.__ratings.items():
            if (
                rating.get("board_type") == board_type
                and rating.get("num_players") == num_players
                and rating.get("games_played", 0) >= min_games
            ):
                entry = dict(rating)
                # Anchor random participants
                if self._is_random_participant(rating.get("participant_id", "")):
                    entry["rating"] = self.BASELINE_ELO_RANDOM
                entries.append(entry)

        entries.sort(key=lambda x: -x.get("rating", 0))
        return entries[:limit]

    def get_recent_matches(
        self,
        board_type: str | None = None,
        num_players: int | None = None,
        participant_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get recent matches with optional filters.

        Args:
            board_type: Filter by board type
            num_players: Filter by player count
            participant_id: Filter by participant
            limit: Maximum matches

        Returns:
            List of match entries (newest first)
        """
        matches = []
        for match in self.__recent_matches.values():
            if board_type and match.get("board_type") != board_type:
                continue
            if num_players and match.get("num_players") != num_players:
                continue
            if participant_id and participant_id not in match.get("participant_ids", []):
                continue
            matches.append(dict(match))

        matches.sort(key=lambda x: -x.get("timestamp", 0))
        return matches[:limit]

    def get_stats(self) -> dict[str, Any]:
        """Get store statistics.

        Returns:
            Dict with rating count, match count, and cluster status
        """
        return {
            "total_ratings": len(self.__ratings._data),
            "total_matches": len(self.__recent_matches._data),
            "is_leader": self.is_leader,
            "leader_address": self.leader_address,
            "is_ready": self.is_ready,
        }


# ============================================
# Factory Functions
# ============================================


def create_replicated_elo_store(
    node_id: str,
    config_path: Path | None = None,
    bind_port: int = RAFT_BIND_PORT,
    on_ready: Callable[[], None] | None = None,
    on_leader_change: Callable[[str | None], None] | None = None,
) -> ReplicatedEloStore | None:
    """Create a ReplicatedEloStore with auto-configured addresses.

    Args:
        node_id: This node's unique identifier
        config_path: Path to distributed_hosts.yaml
        bind_port: Raft bind port (uses different offset for Elo store)
        on_ready: Callback when cluster is ready
        on_leader_change: Callback on leader changes

    Returns:
        ReplicatedEloStore instance, or None if Raft not available/enabled
    """
    if not PYSYNCOBJ_AVAILABLE:
        logger.warning("Cannot create ReplicatedEloStore: pysyncobj not installed")
        return None

    if not RAFT_ENABLED:
        logger.info("Raft is disabled (RINGRIFT_RAFT_ENABLED=false)")
        return None

    # Use port offset to avoid conflict with work queue and job assignments
    elo_port = bind_port + 2

    self_address = get_self_raft_address(config_path, elo_port)
    if not self_address:
        logger.error("Cannot determine self Raft address for Elo store")
        return None

    partners = load_raft_partner_addresses(node_id, config_path, elo_port)
    if not partners:
        logger.warning("No Raft partners configured for Elo store - single-node mode")

    return ReplicatedEloStore(
        node_id=node_id,
        self_address=self_address,
        partner_addresses=partners,
        on_ready=on_ready,
        on_leader_change=on_leader_change,
    )


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
    "EloRatingEntry",
    "EloMatchEntry",
    # Main classes
    "ReplicatedWorkQueue",
    "ReplicatedJobAssignments",
    "ReplicatedEloStore",
    # Factory functions
    "create_replicated_work_queue",
    "create_replicated_job_assignments",
    "create_replicated_elo_store",
    # Configuration helpers
    "load_raft_partner_addresses",
    "get_self_raft_address",
]
