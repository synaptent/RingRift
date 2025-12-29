"""Consensus Mixin for PySyncObj Raft Integration.

Extracted from p2p_orchestrator.py for modularity.
This mixin provides Raft-based consensus integration with P2POrchestrator.

Usage:
    class P2POrchestrator(ConsensusMixin, ...):
        pass

Phase 2 extraction - Dec 26, 2025
Refactored to use P2PMixinBase - Dec 27, 2025

Features:
- Integrates PySyncObj Raft for replicated state machines
- Provides work queue operations via Raft when enabled
- Supports instant fallback to SQLite when Raft unavailable
- Routes operations based on CONSENSUS_MODE (bully/raft/hybrid)

Requirements:
- PySyncObj must be installed: pip install pysyncobj
- RINGRIFT_RAFT_ENABLED=true to enable Raft consensus
- RINGRIFT_CONSENSUS_MODE controls routing (bully/raft/hybrid)
"""

from __future__ import annotations

import asyncio
import logging
import time
from threading import RLock
from typing import TYPE_CHECKING, Any

from scripts.p2p.p2p_mixin_base import P2PMixinBase

if TYPE_CHECKING:
    from scripts.p2p.models import NodeInfo
    from scripts.p2p.types import NodeRole

logger = logging.getLogger(__name__)

# Import constants with fallbacks using protocol_utils
from scripts.p2p.protocol_utils import load_constants, safe_import, safe_import_from

_consts = load_constants(
    "CONSENSUS_MODE", "DEFAULT_PORT", "RAFT_AUTO_UNLOCK_TIME",
    "RAFT_BIND_PORT", "RAFT_COMPACTION_MIN_ENTRIES", "RAFT_ENABLED",
    defaults={
        "RAFT_ENABLED": False,
        "RAFT_BIND_PORT": 4321,
        "RAFT_COMPACTION_MIN_ENTRIES": 1000,
        "RAFT_AUTO_UNLOCK_TIME": 300.0,
        "CONSENSUS_MODE": "bully",
        "DEFAULT_PORT": 8770,
    },
)
RAFT_ENABLED = _consts["RAFT_ENABLED"]
RAFT_BIND_PORT = _consts["RAFT_BIND_PORT"]
RAFT_COMPACTION_MIN_ENTRIES = _consts["RAFT_COMPACTION_MIN_ENTRIES"]
RAFT_AUTO_UNLOCK_TIME = _consts["RAFT_AUTO_UNLOCK_TIME"]
CONSENSUS_MODE = _consts["CONSENSUS_MODE"]
DEFAULT_PORT = _consts["DEFAULT_PORT"]

# Try to import pysyncobj - graceful fallback if unavailable using protocol_utils
_, _SyncObj, _SyncObjConf, _replicated, PYSYNCOBJ_AVAILABLE = safe_import(
    "pysyncobj", "SyncObj", "SyncObjConf", "replicated"
)
_ReplDict, _ = safe_import_from("pysyncobj.batteries", "ReplDict")
_ReplLockManager, _ = safe_import_from("pysyncobj.batteries", "ReplLockManager")


def get_work_queue():
    """Get the work queue singleton (lazy load).

    This provides fallback to SQLite-based work queue when Raft is unavailable.
    """
    try:
        from scripts.p2p.handlers.work_queue import get_work_queue as _get_wq

        return _get_wq()
    except ImportError:
        return None


# Type alias for optional Raft classes
ReplicatedWorkQueue = None
ReplicatedJobAssignments = None

# Only define Raft classes if pysyncobj is available
if PYSYNCOBJ_AVAILABLE:

    class ReplicatedWorkQueue(_SyncObj):
        """Raft-replicated work queue for distributed work distribution.

        This provides strongly consistent work queue operations across the cluster.
        Work items are replicated via Raft log entries before being acknowledged.

        Features:
        - Atomic work claim with distributed locking
        - Automatic log compaction after RAFT_COMPACTION_MIN_ENTRIES
        - Timeout-based lock release for dead workers
        """

        def __init__(
            self,
            self_addr: str,
            partner_addrs: list[str],
            auto_unlock_time: float = RAFT_AUTO_UNLOCK_TIME,
            compaction_min_entries: int = RAFT_COMPACTION_MIN_ENTRIES,
        ):
            # Raft configuration
            conf = _SyncObjConf(
                autoTick=True,
                appendEntriesUseBatch=True,
                fullDumpFile=None,  # In-memory only for now
                logCompactionMinEntries=compaction_min_entries,
                dynamicMembershipChange=True,  # Allow adding/removing nodes
            )

            super().__init__(self_addr, partner_addrs, conf=conf)

            # Replicated data structures
            self._work_items: _ReplDict[str, dict[str, Any]] = _ReplDict()
            self._claimed_work: _ReplDict[str, str] = _ReplDict()  # work_id -> node_id
            self._lock_manager = _ReplLockManager(autoUnlockTime=auto_unlock_time)

            # Local tracking
            self._local_start_time = time.time()

        @_replicated
        def add_work(
            self,
            work_id: str,
            work_type: str,
            priority: int,
            config: dict[str, Any],
            timeout_seconds: float = 3600.0,
        ) -> bool:
            """Add work to the replicated queue.

            Returns:
                True if work was added successfully
            """
            if work_id in self._work_items:
                return False  # Already exists

            self._work_items[work_id] = {
                "work_id": work_id,
                "work_type": work_type,
                "priority": priority,
                "config": config,
                "timeout_seconds": timeout_seconds,
                "status": "pending",
                "created_at": time.time(),
                "claimed_by": None,
                "claimed_at": None,
            }
            return True

        @_replicated
        def claim_work(self, node_id: str, work_id: str) -> bool:
            """Atomically claim work for a node.

            Uses distributed locking to ensure only one node claims each work item.

            Returns:
                True if claim succeeded, False otherwise
            """
            if work_id not in self._work_items:
                return False

            item = self._work_items[work_id]
            if item["status"] != "pending":
                return False

            # Try to acquire distributed lock
            if not self._lock_manager.tryAcquire(f"work:{work_id}", sync=True):
                return False

            # Update item
            item["status"] = "claimed"
            item["claimed_by"] = node_id
            item["claimed_at"] = time.time()
            self._work_items[work_id] = item
            self._claimed_work[work_id] = node_id

            return True

        @_replicated
        def complete_work(self, work_id: str, result: dict[str, Any] | None = None) -> bool:
            """Mark work as completed.

            Returns:
                True if work was completed successfully
            """
            if work_id not in self._work_items:
                return False

            item = self._work_items[work_id]
            item["status"] = "completed"
            item["completed_at"] = time.time()
            item["result"] = result or {}
            self._work_items[work_id] = item

            # Release lock
            self._lock_manager.release(f"work:{work_id}")

            # Clean up claimed_work
            if work_id in self._claimed_work:
                del self._claimed_work[work_id]

            return True

        @_replicated
        def fail_work(self, work_id: str, error: str) -> bool:
            """Mark work as failed (may be retried).

            Returns:
                True if work was marked as failed
            """
            if work_id not in self._work_items:
                return False

            item = self._work_items[work_id]
            item["status"] = "failed"
            item["failed_at"] = time.time()
            item["error"] = error
            self._work_items[work_id] = item

            # Release lock
            self._lock_manager.release(f"work:{work_id}")

            # Clean up claimed_work
            if work_id in self._claimed_work:
                del self._claimed_work[work_id]

            return True

        def get_pending_work(
            self, capabilities: list[str] | None = None
        ) -> list[dict[str, Any]]:
            """Get all pending work items, optionally filtered by capabilities.

            Args:
                capabilities: Optional list of work types the node can handle

            Returns:
                List of pending work items sorted by priority (highest first)
            """
            pending = []
            for work_id, item in self._work_items.items():
                if item["status"] != "pending":
                    continue
                if capabilities and item["work_type"] not in capabilities:
                    continue
                pending.append(dict(item))

            # Sort by priority (descending)
            pending.sort(key=lambda x: x["priority"], reverse=True)
            return pending

        def get_queue_status(self) -> dict[str, Any]:
            """Get overall queue status."""
            status_counts = {"pending": 0, "claimed": 0, "completed": 0, "failed": 0}
            for item in self._work_items.values():
                status = item.get("status", "unknown")
                if status in status_counts:
                    status_counts[status] += 1

            return {
                "total_items": len(self._work_items),
                "by_status": status_counts,
                "raft_leader": self._getLeader() is not None,
                "raft_term": self._getTerm() if hasattr(self, "_getTerm") else None,
            }

    class ReplicatedJobAssignments(_SyncObj):
        """Raft-replicated job assignments for cluster coordination.

        Tracks which nodes are assigned to which jobs for distributed coordination.
        Separate from work queue for lighter-weight tracking without full work metadata.
        """

        def __init__(
            self,
            self_addr: str,
            partner_addrs: list[str],
            auto_unlock_time: float = RAFT_AUTO_UNLOCK_TIME,
        ):
            conf = _SyncObjConf(
                autoTick=True,
                appendEntriesUseBatch=True,
            )
            super().__init__(self_addr, partner_addrs, conf=conf)

            # node_id -> list of job_ids
            self._node_jobs: _ReplDict[str, list[str]] = _ReplDict()
            # job_id -> node_id
            self._job_node: _ReplDict[str, str] = _ReplDict()

        @_replicated
        def assign_job(self, node_id: str, job_id: str) -> bool:
            """Assign a job to a node."""
            if job_id in self._job_node:
                return False  # Already assigned

            self._job_node[job_id] = node_id

            if node_id not in self._node_jobs:
                self._node_jobs[node_id] = []
            jobs = list(self._node_jobs[node_id])
            jobs.append(job_id)
            self._node_jobs[node_id] = jobs

            return True

        @_replicated
        def unassign_job(self, job_id: str) -> bool:
            """Remove job assignment."""
            if job_id not in self._job_node:
                return False

            node_id = self._job_node[job_id]
            del self._job_node[job_id]

            if node_id in self._node_jobs:
                jobs = list(self._node_jobs[node_id])
                if job_id in jobs:
                    jobs.remove(job_id)
                    self._node_jobs[node_id] = jobs

            return True

        def get_node_jobs(self, node_id: str) -> list[str]:
            """Get all jobs assigned to a node."""
            return list(self._node_jobs.get(node_id, []))

        def get_job_node(self, job_id: str) -> str | None:
            """Get the node assigned to a job."""
            return self._job_node.get(job_id)


class ConsensusMixin(P2PMixinBase):
    """Mixin integrating PySyncObj Raft with P2POrchestrator.

    Inherits from P2PMixinBase for shared logging helpers.

    Provides Raft-based consensus for:
    - Work queue operations (add, claim, complete)
    - Job assignments
    - Distributed locking

    Falls back to SQLite-based operations when Raft is unavailable.

    Requires the implementing class to have:
    State:
    - node_id: str - This node's ID
    - role: NodeRole - Current node role
    - voter_node_ids: list[str] - Configured voters
    - peers: dict[str, NodeInfo] - Active peers
    - peers_lock: RLock - Lock for peers dict
    - advertise_host: str - This node's advertised host
    - advertise_port: int - This node's advertised port

    Methods:
    - _save_state() - Persist state changes
    """

    MIXIN_TYPE = "consensus"

    # Type hints for IDE support (implemented by P2POrchestrator)
    node_id: str
    role: Any  # NodeRole
    voter_node_ids: list[str]
    peers: dict[str, Any]  # dict[str, NodeInfo]
    peers_lock: "RLock"
    advertise_host: str
    advertise_port: int

    def _init_raft_consensus(self) -> bool:
        """Initialize Raft consensus if enabled.

        Returns:
            True if Raft was initialized successfully, False otherwise
        """
        # Initialize instance attributes
        self._raft_work_queue = None
        self._raft_job_assignments = None
        self._raft_initialized = False
        self._raft_init_error: str | None = None

        if not RAFT_ENABLED:
            logger.debug("Raft consensus disabled (RINGRIFT_RAFT_ENABLED != true)")
            return False

        if not PYSYNCOBJ_AVAILABLE:
            self._raft_init_error = "pysyncobj not installed"
            logger.warning(
                f"Raft consensus unavailable: {self._raft_init_error}. "
                "Install with: pip install pysyncobj"
            )
            return False

        try:
            # Get Raft partner addresses
            partners = self._get_raft_partners()
            if not partners:
                self._raft_init_error = "no Raft partners available"
                logger.warning(f"Raft consensus unavailable: {self._raft_init_error}")
                return False

            # Build self address for Raft
            self_addr = f"{self.advertise_host}:{RAFT_BIND_PORT}"

            logger.info(
                f"Initializing Raft consensus: self={self_addr}, "
                f"partners={partners}, mode={CONSENSUS_MODE}"
            )

            # Initialize replicated work queue
            self._raft_work_queue = ReplicatedWorkQueue(
                self_addr=self_addr,
                partner_addrs=partners,
            )

            # Initialize replicated job assignments
            self._raft_job_assignments = ReplicatedJobAssignments(
                self_addr=self_addr,
                partner_addrs=partners,
            )

            self._raft_initialized = True
            logger.info("Raft consensus initialized successfully")

            # Emit leader change event if we're the Raft leader
            if self.is_raft_leader():
                self._emit_raft_leader_event(is_leader=True)

            return True

        except Exception as e:
            self._raft_init_error = str(e)
            logger.error(f"Failed to initialize Raft consensus: {e}")
            return False

    def _get_raft_partners(self) -> list[str]:
        """Get Raft partner addresses from voter nodes.

        Returns:
            List of partner addresses in format "host:port"
        """
        partners = []

        # Use voter nodes as Raft partners (they should be stable)
        for voter_id in self.voter_node_ids:
            if voter_id == self.node_id:
                continue  # Skip self

            # Try to get peer address
            with self.peers_lock:
                peer = self.peers.get(voter_id)

            if peer and hasattr(peer, "endpoint") and peer.endpoint:
                # Extract host from endpoint, use Raft port
                host = peer.endpoint.split(":")[0]
                partners.append(f"{host}:{RAFT_BIND_PORT}")

        return partners

    def _should_use_raft(self) -> bool:
        """Check if Raft should be used for consensus operations.

        Returns:
            True if Raft should be used, False to fall back to SQLite
        """
        if not RAFT_ENABLED or not getattr(self, "_raft_initialized", False):
            return False

        if CONSENSUS_MODE == "bully":
            return False  # Bully mode uses SQLite only

        if CONSENSUS_MODE == "raft":
            return True  # Raft mode always uses Raft

        if CONSENSUS_MODE == "hybrid":
            # Hybrid mode: use Raft for work queue, Bully for leader election
            return True

        return False

    def claim_work_distributed(
        self,
        node_id: str,
        capabilities: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Claim available work from the queue using appropriate consensus.

        Routes to Raft or SQLite based on CONSENSUS_MODE.

        Args:
            node_id: ID of the node claiming work
            capabilities: Optional list of work types the node can handle

        Returns:
            Work item dict if claim succeeded, None otherwise
        """
        if self._should_use_raft() and getattr(self, "_raft_work_queue", None) is not None:
            return self._claim_work_raft(node_id, capabilities)
        else:
            return self._claim_work_sqlite(node_id, capabilities)

    def _claim_work_raft(
        self,
        node_id: str,
        capabilities: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Claim work using Raft-replicated work queue.

        Args:
            node_id: ID of the node claiming work
            capabilities: Optional list of work types the node can handle

        Returns:
            Work item dict if claim succeeded, None otherwise
        """
        raft_wq = getattr(self, "_raft_work_queue", None)
        if raft_wq is None:
            logger.warning("Raft work queue not initialized, falling back to SQLite")
            return self._claim_work_sqlite(node_id, capabilities)

        try:
            # Get pending work
            pending = raft_wq.get_pending_work(capabilities)
            if not pending:
                return None

            # Try to claim the highest priority item
            for item in pending:
                work_id = item["work_id"]
                if raft_wq.claim_work(node_id, work_id):
                    logger.info(
                        f"[Raft] Node {node_id} claimed work {work_id} "
                        f"(type={item['work_type']}, priority={item['priority']})"
                    )
                    return item

            return None

        except Exception as e:
            logger.error(f"Raft claim_work failed: {e}, falling back to SQLite")
            return self._claim_work_sqlite(node_id, capabilities)

    def _claim_work_sqlite(
        self,
        node_id: str,
        capabilities: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Claim work using SQLite-based work queue.

        Args:
            node_id: ID of the node claiming work
            capabilities: Optional list of work types the node can handle

        Returns:
            Work item dict if claim succeeded, None otherwise
        """
        wq = get_work_queue()
        if wq is None:
            logger.warning("Work queue not available")
            return None

        try:
            item = wq.claim_work(node_id, capabilities)
            if item is not None:
                return item.to_dict() if hasattr(item, "to_dict") else item
            return None
        except Exception as e:
            logger.error(f"SQLite claim_work failed: {e}")
            return None

    def start_work_distributed(self, work_id: str) -> bool:
        """Mark claimed work as started using appropriate consensus.

        Routes to Raft or SQLite based on CONSENSUS_MODE.
        For Raft: claim already implies start, so this is a no-op.
        For SQLite: transitions from CLAIMED to RUNNING.

        Args:
            work_id: ID of the work item to start

        Returns:
            True if start succeeded, False otherwise
        """
        if self._should_use_raft() and getattr(self, "_raft_work_queue", None) is not None:
            # Raft: claim implies start, no additional action needed
            # The work is already in "claimed" status which means running
            logger.debug(f"[Raft] Work {work_id} already started via claim")
            return True
        else:
            return self._start_work_sqlite(work_id)

    def _start_work_sqlite(self, work_id: str) -> bool:
        """Mark work as started using SQLite work queue.

        Args:
            work_id: ID of the work item to start

        Returns:
            True if start succeeded, False otherwise
        """
        wq = get_work_queue()
        if wq is None:
            logger.warning("Work queue not available")
            return False

        try:
            return wq.start_work(work_id)
        except Exception as e:
            logger.error(f"SQLite start_work failed: {e}")
            return False

    def fail_work_distributed(
        self,
        work_id: str,
        error: str = "",
    ) -> bool:
        """Mark work as failed using appropriate consensus.

        Routes to Raft or SQLite based on CONSENSUS_MODE.

        Args:
            work_id: ID of the work item that failed
            error: Error message describing the failure

        Returns:
            True if fail succeeded, False otherwise
        """
        if self._should_use_raft() and getattr(self, "_raft_work_queue", None) is not None:
            return self._fail_work_raft(work_id, error)
        else:
            return self._fail_work_sqlite(work_id, error)

    def _fail_work_raft(self, work_id: str, error: str) -> bool:
        """Mark work as failed using Raft-replicated work queue.

        Args:
            work_id: ID of the work item that failed
            error: Error message describing the failure

        Returns:
            True if fail succeeded, False otherwise
        """
        raft_wq = getattr(self, "_raft_work_queue", None)
        if raft_wq is None:
            logger.warning("Raft work queue not initialized, falling back to SQLite")
            return self._fail_work_sqlite(work_id, error)

        try:
            success = raft_wq.fail_work(work_id, error)
            if success:
                logger.info(f"[Raft] Work {work_id} marked as failed: {error}")
            return success
        except Exception as e:
            logger.error(f"Raft fail_work failed: {e}, falling back to SQLite")
            return self._fail_work_sqlite(work_id, error)

    def _fail_work_sqlite(self, work_id: str, error: str) -> bool:
        """Mark work as failed using SQLite work queue.

        Args:
            work_id: ID of the work item that failed
            error: Error message describing the failure

        Returns:
            True if fail succeeded, False otherwise
        """
        wq = get_work_queue()
        if wq is None:
            logger.warning("Work queue not available")
            return False

        try:
            return wq.fail_work(work_id, error)
        except Exception as e:
            logger.error(f"SQLite fail_work failed: {e}")
            return False

    def complete_work_distributed(
        self,
        work_id: str,
        result: dict[str, Any] | None = None,
    ) -> bool:
        """Mark work as completed using appropriate consensus.

        Routes to Raft or SQLite based on CONSENSUS_MODE.

        Args:
            work_id: ID of the work item that completed
            result: Optional result data

        Returns:
            True if completion succeeded, False otherwise
        """
        if self._should_use_raft() and getattr(self, "_raft_work_queue", None) is not None:
            return self._complete_work_raft(work_id, result)
        else:
            return self._complete_work_sqlite(work_id, result)

    def _complete_work_raft(
        self,
        work_id: str,
        result: dict[str, Any] | None = None,
    ) -> bool:
        """Mark work as completed using Raft-replicated work queue.

        Args:
            work_id: ID of the work item that completed
            result: Optional result data

        Returns:
            True if completion succeeded, False otherwise
        """
        raft_wq = getattr(self, "_raft_work_queue", None)
        if raft_wq is None:
            logger.warning("Raft work queue not initialized, falling back to SQLite")
            return self._complete_work_sqlite(work_id, result)

        try:
            success = raft_wq.complete_work(work_id, result)
            if success:
                logger.info(f"[Raft] Work {work_id} marked as completed")
            return success
        except Exception as e:
            logger.error(f"Raft complete_work failed: {e}, falling back to SQLite")
            return self._complete_work_sqlite(work_id, result)

    def _complete_work_sqlite(
        self,
        work_id: str,
        result: dict[str, Any] | None = None,
    ) -> bool:
        """Mark work as completed using SQLite work queue.

        Args:
            work_id: ID of the work item that completed
            result: Optional result data

        Returns:
            True if completion succeeded, False otherwise
        """
        wq = get_work_queue()
        if wq is None:
            logger.warning("Work queue not available")
            return False

        try:
            return wq.complete_work(work_id, result)
        except Exception as e:
            logger.error(f"SQLite complete_work failed: {e}")
            return False

    def is_raft_leader(self) -> bool:
        """Check if this node is the Raft leader.

        Returns:
            True if this node is the Raft leader, False otherwise
        """
        if not getattr(self, "_raft_initialized", False):
            return False

        raft_wq = getattr(self, "_raft_work_queue", None)
        if raft_wq is None:
            return False

        try:
            leader = raft_wq._getLeader()
            if leader is None:
                return False
            # Compare with our address
            self_addr = f"{self.advertise_host}:{RAFT_BIND_PORT}"
            return str(leader) == self_addr
        except Exception as e:
            logger.debug(f"Error checking Raft leader status: {e}")
            return False

    def get_raft_status(self) -> dict[str, Any]:
        """Get Raft consensus status for monitoring.

        Returns:
            Dict with Raft status information
        """
        raft_initialized = getattr(self, "_raft_initialized", False)
        raft_init_error = getattr(self, "_raft_init_error", None)

        status = {
            "raft_enabled": RAFT_ENABLED,
            "raft_available": PYSYNCOBJ_AVAILABLE,
            "raft_initialized": raft_initialized,
            "raft_init_error": raft_init_error,
            "consensus_mode": CONSENSUS_MODE,
            "should_use_raft": self._should_use_raft(),
        }

        if raft_initialized:
            raft_wq = getattr(self, "_raft_work_queue", None)
            if raft_wq is not None:
                try:
                    status.update(
                        {
                            "is_raft_leader": self.is_raft_leader(),
                            "raft_leader": str(raft_wq._getLeader() or "none"),
                            "work_queue_status": raft_wq.get_queue_status(),
                        }
                    )
                except Exception as e:
                    status["raft_error"] = str(e)

        return status

    def _emit_raft_leader_event(self, is_leader: bool) -> None:
        """Emit an event when Raft leadership changes.

        Args:
            is_leader: True if this node became leader, False if it lost leadership
        """
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            if router is None:
                return

            event_type = "RAFT_LEADER_ELECTED" if is_leader else "RAFT_LEADER_LOST"
            payload = {
                "node_id": self.node_id,
                "is_leader": is_leader,
                "consensus_mode": CONSENSUS_MODE,
                "timestamp": time.time(),
            }

            # Fire and forget - don't block on event publishing
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None:
                from app.core.async_context import fire_and_forget

                fire_and_forget(
                    router.publish(event_type, payload=payload, source="consensus_mixin")
                )
            else:
                # Sync context - skip for now to avoid blocking
                logger.debug(f"Would emit {event_type} but no event loop running")

        except ImportError:
            # Event router not available
            pass
        except Exception as e:
            logger.debug(f"Failed to emit Raft leader event: {e}")

    def consensus_health_check(self) -> dict[str, Any]:
        """Return health status for Raft consensus subsystem.

        Returns:
            dict with is_healthy, raft_enabled, raft_initialized details
        """
        raft_init = getattr(self, "_raft_initialized", False)
        raft_error = getattr(self, "_raft_init_error", None)
        # Healthy if Raft is disabled or initialized successfully
        is_healthy = (not RAFT_ENABLED) or raft_init
        return {
            "is_healthy": is_healthy,
            "raft_enabled": RAFT_ENABLED,
            "raft_available": PYSYNCOBJ_AVAILABLE,
            "raft_initialized": raft_init,
            "raft_init_error": raft_error,
            "consensus_mode": CONSENSUS_MODE,
        }

    def health_check(self) -> dict[str, Any]:
        """Return health status for consensus mixin (DaemonManager integration).

        December 2025: Added for unified health check interface.
        Uses base class helper for standardized response format.

        Returns:
            dict with healthy status, message, and details
        """
        status = self.consensus_health_check()
        is_healthy = status.get("is_healthy", False)
        error = status.get("raft_init_error", "unknown")
        message = "Consensus healthy" if is_healthy else f"Consensus unhealthy: {error}"
        return self._build_health_response(is_healthy, message, status)


# Export public API
__all__ = [
    "ConsensusMixin",
    "ReplicatedJobAssignments",
    "ReplicatedWorkQueue",
    "PYSYNCOBJ_AVAILABLE",
    "RAFT_ENABLED",
    "CONSENSUS_MODE",
]
