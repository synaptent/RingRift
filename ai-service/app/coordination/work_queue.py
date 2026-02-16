"""Centralized Work Queue for Cluster Work Distribution.

The leader maintains a prioritized work queue. Workers pull appropriate work
based on their capabilities and policies.

Architecture:
- Leader: Maintains work queue, assigns work to workers
- Workers: Poll for work, report completion/failure
- Work items: Typed (training, cmaes, tournament, etc.) with priorities

Backend Priority (Dec 30, 2025 - P5.1 Raft Integration):
1. **Raft** - Cluster-wide strongly consistent queue via PySyncObj
2. **SQLite** - Local persistence with file-based locking (fallback)

When Raft is available (P2P orchestrator running with Raft enabled), work queue
operations use the replicated state machine for cluster-wide consistency. This
eliminates duplicate job assignments and provides atomic claiming.

Usage:
    # On leader
    queue = WorkQueue()
    queue.add_work(WorkItem(work_type="training", config={"board": "square8"}))

    # On worker (via API)
    work = queue.claim_work(node_id="gpu-node-1", capabilities=["training", "gpu_cmaes"])
    # ... do work ...
    queue.complete_work(work.work_id)

    # Check which backend is being used
    print(f"Backend: {queue.backend}")  # "raft" or "sqlite"
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# December 2025: Import WorkStatus from canonical source
from app.coordination.types import WorkStatus  # noqa: E402
from app.coordination.contracts import CoordinatorStatus, HealthCheckResult  # noqa: E402
# Jan 6, 2026: Import node circuit breaker for claim filtering
from app.coordination.node_circuit_breaker import get_node_circuit_breaker  # noqa: E402
from app.utils.retry import RetryConfig  # noqa: E402
from app.utils.disk_utils import is_enospc_error, handle_enospc_error
from app.coordination.event_utils import parse_config_key
from app.config.thresholds import SQLITE_CONNECT_TIMEOUT, SQLITE_SHORT_TIMEOUT
from app.config.coordination_defaults import WorkQueueCleanupDefaults

# Jan 2, 2026: Strategy pattern for Raft/SQLite backends
from app.coordination.work_queue_backends import (
    BackendResult,
    BackendType,
    RaftBackend,
    SQLiteBackend,
    WorkQueueBackend,
    create_backend,
)

logger = logging.getLogger(__name__)


# ============================================
# Work Queue Backend Selection (Dec 30, 2025 - P5.1)
# ============================================


class WorkQueueBackendType(str, Enum):
    """Available work queue backend types."""

    RAFT = "raft"  # Cluster-wide via Raft consensus
    SQLITE = "sqlite"  # Local SQLite database


# Raft work queue availability check (cached)
_raft_wq_available: bool | None = None
_raft_work_queue: Any = None  # ReplicatedWorkQueue instance
_raft_node_id: str | None = None


def _check_raft_work_queue_available() -> bool:
    """Check if Raft work queue is available.

    Returns True if:
    1. pysyncobj is installed
    2. RAFT_ENABLED is True
    3. P2P orchestrator is running with initialized Raft
    4. ReplicatedWorkQueue is accessible

    Result is cached for performance.
    """
    global _raft_wq_available, _raft_work_queue, _raft_node_id

    if _raft_wq_available is not None:
        return _raft_wq_available

    try:
        # Check if Raft is enabled
        from app.p2p.raft_state import PYSYNCOBJ_AVAILABLE
        from app.p2p.constants import RAFT_ENABLED

        if not RAFT_ENABLED or not PYSYNCOBJ_AVAILABLE:
            logger.debug(
                "Raft work queue disabled: RAFT_ENABLED=%s, PYSYNCOBJ=%s",
                RAFT_ENABLED, PYSYNCOBJ_AVAILABLE
            )
            _raft_wq_available = False
            return False

        # Try to get work queue from P2P orchestrator
        try:
            from scripts.p2p_orchestrator import P2POrchestrator

            # Check for singleton instance
            orchestrator = getattr(P2POrchestrator, "_instance", None)
            if orchestrator is None:
                logger.debug("Raft work queue: P2P orchestrator not running")
                _raft_wq_available = False
                return False

            # Check if Raft is initialized
            raft_initialized = getattr(orchestrator, "_raft_initialized", False)
            if not raft_initialized:
                logger.debug("Raft work queue: Raft not initialized on orchestrator")
                _raft_wq_available = False
                return False

            # Get the replicated work queue
            raft_wq = getattr(orchestrator, "_raft_work_queue", None)
            if raft_wq is None:
                logger.debug("Raft work queue: ReplicatedWorkQueue not available")
                _raft_wq_available = False
                return False

            # Check if it's ready
            if not getattr(raft_wq, "is_ready", False):
                logger.debug("Raft work queue: ReplicatedWorkQueue not ready")
                _raft_wq_available = False
                return False

            # Success - cache the work queue
            _raft_work_queue = raft_wq
            _raft_node_id = getattr(orchestrator, "node_id", "unknown")
            _raft_wq_available = True
            logger.info(
                "Raft work queue available via P2P orchestrator (node: %s, leader: %s)",
                _raft_node_id,
                getattr(raft_wq, "leader_address", "unknown"),
            )
            return True

        except ImportError:
            logger.debug("Raft work queue: Could not import P2P orchestrator")
            _raft_wq_available = False
            return False

    except ImportError:
        logger.debug("Raft work queue: pysyncobj or raft_state not available")
        _raft_wq_available = False
        return False
    except Exception as e:
        logger.warning("Raft work queue: Unexpected error checking availability: %s", e)
        _raft_wq_available = False
        return False


def reset_raft_work_queue_cache() -> None:
    """Reset the Raft work queue availability cache.

    Call this if P2P orchestrator state changes (e.g., Raft initialization).
    """
    global _raft_wq_available, _raft_work_queue, _raft_node_id
    _raft_wq_available = None
    _raft_work_queue = None
    _raft_node_id = None


def get_raft_work_queue() -> Any:
    """Get the cached Raft work queue instance.

    Returns:
        ReplicatedWorkQueue instance or None if not available
    """
    if _check_raft_work_queue_available():
        return _raft_work_queue
    return None


# Default path for work queue database
# Respect RINGRIFT_WORK_QUEUE_DB environment variable for consistency across all components
_DEFAULT_DB_DIR = Path(__file__).parent.parent.parent / "data"
DEFAULT_DB_PATH = Path(os.environ.get("RINGRIFT_WORK_QUEUE_DB", str(_DEFAULT_DB_DIR / "work_queue.db")))

# Dec 28, 2025: Backpressure thresholds to prevent unbounded queue growth
# Jan 5, 2026: Doubled thresholds to support 30+ node cluster throughput
# Jan 6, 2026 (Session 17.47): Doubled again to 4000 hard limit.
# Root cause: 3-player selfplay jobs were rejected due to full queue (2000/2000)
# despite 500x priority multiplier. Starving configs need queue capacity.
# Jan 25, 2026: Increased to 10000 hard limit for 20+ node cluster stability.
# Jan 27, 2026: Increased to 15000 hard limit to provide recovery headroom.
# With 20 nodes × 25 cores × 2 items/min = 1000 items/min capacity.
# 15000 limit provides ~15 min buffer for backpressure handling.
# Soft limit: Emit BACKPRESSURE_ACTIVATED event, warn callers
# Hard limit: Reject new items, force callers to wait
BACKPRESSURE_SOFT_LIMIT = int(os.environ.get("RINGRIFT_WORK_QUEUE_SOFT_LIMIT", "7500"))
BACKPRESSURE_HARD_LIMIT = int(os.environ.get("RINGRIFT_WORK_QUEUE_HARD_LIMIT", "15000"))
# Recovery threshold: Emit BACKPRESSURE_RELEASED when queue drops below this
BACKPRESSURE_RECOVERY_THRESHOLD = int(os.environ.get("RINGRIFT_WORK_QUEUE_RECOVERY", "1200"))


class SlackWorkQueueNotifier:
    """Simple Slack notifier for work queue events."""

    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
        self.enabled = bool(self.webhook_url)
        if self.enabled:
            logger.info("Slack work queue notifications enabled")

    def _send(self, text: str, color: str = "#36a64f") -> bool:
        """Send a Slack message."""
        if not self.enabled:
            return False

        try:
            import urllib.request
            payload = json.dumps({
                "attachments": [{
                    "color": color,
                    "text": text,
                    "footer": "RingRift Work Queue",
                    "ts": int(time.time())
                }]
            }).encode("utf-8")

            req = urllib.request.Request(
                self.webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)
            return True
        except (OSError, TimeoutError, ValueError) as e:
            # URLError inherits from OSError; ValueError for malformed URLs
            logger.warning(f"Failed to send Slack notification: {e}")
            return False

    def on_work_added(self, item: WorkItem) -> None:
        """Notify on high-priority work added."""
        if item.priority >= 90:
            self._send(
                f":inbox_tray: *High-priority work added*\n"
                f"Type: `{item.work_type.value}` | Priority: {item.priority}\n"
                f"ID: `{item.work_id}` | Config: {json.dumps(item.config)}",
                color="#f2c744"
            )

    def on_work_completed(self, item: WorkItem) -> None:
        """Notify on work completion (high-priority only to reduce noise)."""
        if item.priority < 80:
            return  # Skip low-priority completions
        duration = item.completed_at - item.created_at if item.completed_at and item.created_at else 0
        self._send(
            f":white_check_mark: *Work completed*\n"
            f"Type: `{item.work_type.value}` | ID: `{item.work_id}`\n"
            f"Node: `{item.claimed_by}` | Duration: {duration:.1f}s",
            color="#36a64f"
        )

    def on_work_failed(self, item: WorkItem, permanent: bool = False) -> None:
        """Notify on work failure."""
        status = "permanently failed" if permanent else f"failed (attempt {item.attempts}/{item.max_attempts})"
        self._send(
            f":x: *Work {status}*\n"
            f"Type: `{item.work_type.value}` | ID: `{item.work_id}`\n"
            f"Node: `{item.claimed_by}` | Error: {item.error or 'unknown'}",
            color="#e01e5a" if permanent else "#f2c744"
        )

    def on_work_timeout(self, item: WorkItem, permanent: bool = False) -> None:
        """Notify on work timeout."""
        status = "permanently timed out" if permanent else f"timed out (attempt {item.attempts}/{item.max_attempts})"
        self._send(
            f":hourglass: *Work {status}*\n"
            f"Type: `{item.work_type.value}` | ID: `{item.work_id}`\n"
            f"Node: `{item.claimed_by}` | Timeout: {item.timeout_seconds}s",
            color="#e01e5a" if permanent else "#f2c744"
        )


class WorkType(str, Enum):
    """Types of work that can be queued."""
    TRAINING = "training"
    GPU_CMAES = "gpu_cmaes"
    CPU_CMAES = "cpu_cmaes"
    TOURNAMENT = "tournament"
    GAUNTLET = "gauntlet"
    SELFPLAY = "selfplay"
    DATA_MERGE = "data_merge"
    DATA_SYNC = "data_sync"
    VALIDATION = "validation"  # Model validation against baselines
    HYPERPARAM_SWEEP = "hyperparam_sweep"  # Hyperparameter tuning trials


# WorkStatus is now imported from app.coordination.types
# Canonical values: PENDING, CLAIMED, RUNNING, COMPLETED, FAILED, CANCELLED, TIMEOUT


@dataclass
class WorkItem:
    """A unit of work to be executed."""
    work_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    work_type: WorkType = WorkType.SELFPLAY
    priority: int = 50  # Higher = more urgent (0-100)
    config: dict[str, Any] = field(default_factory=dict)

    # Scheduling
    created_at: float = field(default_factory=time.time)
    claimed_at: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0

    # Assignment
    status: WorkStatus = WorkStatus.PENDING
    claimed_by: str = ""  # node_id
    attempts: int = 0
    max_attempts: int = 3
    timeout_seconds: float = 3600.0  # 1 hour default

    # Results
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    # Dependencies - list of work_ids that must complete before this can run
    depends_on: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["work_type"] = self.work_type.value
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorkItem:
        d = d.copy()
        d["work_type"] = WorkType(d.get("work_type", "selfplay"))
        d["status"] = WorkStatus(d.get("status", "pending"))
        # Handle depends_on - ensure it's a list
        if "depends_on" in d and isinstance(d["depends_on"], str):
            import json
            try:
                d["depends_on"] = json.loads(d["depends_on"]) if d["depends_on"] else []
            except (json.JSONDecodeError, TypeError):
                d["depends_on"] = []
        # Feb 2026: Cast numeric fields to prevent type errors from JSON/gossip
        for float_field in ("created_at", "claimed_at", "started_at", "completed_at", "timeout_seconds"):
            if float_field in d and not isinstance(d[float_field], (int, float)):
                try:
                    d[float_field] = float(d[float_field])
                except (ValueError, TypeError):
                    d[float_field] = 0.0
        for int_field in ("priority", "attempts", "max_attempts"):
            if int_field in d and not isinstance(d[int_field], int):
                try:
                    d[int_field] = int(d[int_field])
                except (ValueError, TypeError):
                    d[int_field] = 0
        return cls(**d)

    def is_claimable(self) -> bool:
        """Check if this work can be claimed (doesn't check dependencies)."""
        if self.status != WorkStatus.PENDING:
            return False
        return not self.attempts >= self.max_attempts

    def has_pending_dependencies(self, completed_ids: set[str]) -> bool:
        """Check if any dependencies are not yet completed.

        Args:
            completed_ids: Set of work_ids that are completed

        Returns:
            True if there are unmet dependencies, False if all deps are met
        """
        if not self.depends_on:
            return False
        return any(dep_id not in completed_ids for dep_id in self.depends_on)

    def is_timed_out(self) -> bool:
        """Check if this work has timed out."""
        if self.status not in (WorkStatus.CLAIMED, WorkStatus.RUNNING):
            return False
        if self.claimed_at == 0:
            return False
        return time.time() - self.claimed_at > self.timeout_seconds


@dataclass
class ClaimRejectionStats:
    """Track why jobs are not being dispatched to improve observability.

    Jan 2, 2026: Added to diagnose GPU node idle issues where jobs queue
    but never dispatch. Jobs pass through 7 filtering gates in claim_work()
    that can silently reject them.
    """

    total_claim_attempts: int = 0
    rejected_by_circuit_breaker: int = 0  # Jan 6, 2026: Node circuit is OPEN
    rejected_by_capability: int = 0
    rejected_by_exclusion: int = 0
    rejected_by_target_node: int = 0
    rejected_by_target_node_expired: int = 0  # target_node was cleared due to expiration
    rejected_by_requires_gpu: int = 0
    rejected_by_policy: int = 0
    rejected_by_already_claimed: int = 0
    successful_claims: int = 0

    # Track which target_nodes are being rejected most often
    target_node_rejections: dict[str, int] = field(default_factory=dict)

    # Timestamp of last reset (for rate calculations)
    last_reset_at: float = field(default_factory=time.time)

    def increment_target_node_rejection(self, target_node: str) -> None:
        """Track rejection by specific target_node."""
        self.rejected_by_target_node += 1
        self.target_node_rejections[target_node] = (
            self.target_node_rejections.get(target_node, 0) + 1
        )

    def reset(self) -> None:
        """Reset all counters."""
        self.total_claim_attempts = 0
        self.rejected_by_circuit_breaker = 0
        self.rejected_by_capability = 0
        self.rejected_by_exclusion = 0
        self.rejected_by_target_node = 0
        self.rejected_by_target_node_expired = 0
        self.rejected_by_requires_gpu = 0
        self.rejected_by_policy = 0
        self.rejected_by_already_claimed = 0
        self.successful_claims = 0
        self.target_node_rejections.clear()
        self.last_reset_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "total_claim_attempts": self.total_claim_attempts,
            "rejected_by_circuit_breaker": self.rejected_by_circuit_breaker,
            "rejected_by_capability": self.rejected_by_capability,
            "rejected_by_exclusion": self.rejected_by_exclusion,
            "rejected_by_target_node": self.rejected_by_target_node,
            "rejected_by_target_node_expired": self.rejected_by_target_node_expired,
            "rejected_by_requires_gpu": self.rejected_by_requires_gpu,
            "rejected_by_policy": self.rejected_by_policy,
            "rejected_by_already_claimed": self.rejected_by_already_claimed,
            "successful_claims": self.successful_claims,
            "target_node_rejections": self.target_node_rejections.copy(),
            "last_reset_at": self.last_reset_at,
            "elapsed_seconds": time.time() - self.last_reset_at,
        }


class WorkQueue:
    """Centralized work queue managed by the leader.

    Features:
    - Priority-based scheduling
    - Capability-based work matching
    - Policy enforcement
    - Timeout handling
    - Retry logic
    - Dual-backend support: Raft (cluster-wide) or SQLite (local)

    Backend Selection (Dec 30, 2025 - P5.1):
    - **Raft**: Used when P2P orchestrator is running with Raft enabled.
      Provides cluster-wide atomic claiming and strong consistency.
    - **SQLite**: Fallback for local persistence when Raft unavailable.
    """

    def __init__(
        self,
        policy_manager=None,
        db_path: Path | None = None,
        slack_webhook: str | None = None,
        use_raft: bool = True,
    ):
        """Initialize work queue.

        Args:
            policy_manager: Optional policy manager for work assignment
            db_path: Path to SQLite database (fallback backend)
            slack_webhook: Optional Slack webhook URL for notifications
            use_raft: Whether to try Raft backend first (default: True)
        """
        self._items: dict[str, WorkItem] = {}  # work_id -> WorkItem
        self.lock = threading.RLock()
        self.db_path = db_path or DEFAULT_DB_PATH
        self._use_raft = use_raft
        self._backend: WorkQueueBackendType = WorkQueueBackendType.SQLITE

        # Dec 30, 2025 (P5.1): Try Raft backend first
        if self._use_raft and _check_raft_work_queue_available():
            self._backend = WorkQueueBackendType.RAFT
            logger.info("WorkQueue using Raft backend (cluster-wide consistency)")
        else:
            logger.debug("WorkQueue using SQLite backend (local persistence)")

        # Try to get policy manager
        try:
            if policy_manager is None:
                from app.coordination.node_policies import get_policy_manager
                self.policy_manager = get_policy_manager()
            else:
                self.policy_manager = policy_manager
        except ImportError:
            self.policy_manager = None

        # Slack notifier
        self.notifier = SlackWorkQueueNotifier(webhook_url=slack_webhook)

        # Statistics (local tracking, even with Raft backend)
        self.stats = {
            "total_added": 0,
            "total_completed": 0,
            "total_failed": 0,
            "total_timeout": 0,
        }

        # Track initialization state (December 2025: Lazy initialization)
        self._db_initialized = False
        self._readonly_mode = False

        # Dec 28, 2025: Backpressure state tracking
        self._backpressure_active = False
        self._backpressure_stats = {
            "activations": 0,
            "rejections": 0,
            "last_activation_at": 0.0,
            "last_rejection_at": 0.0,
        }

        # Jan 2, 2026: Claim rejection tracking for dispatch observability
        self._claim_rejection_stats = ClaimRejectionStats()

        # Jan 2, 2026: Strategy pattern backend (lazy initialized)
        # Backend is created on first use to allow _get_connection to be ready
        self._backend_impl: WorkQueueBackend | None = None

        # Database initialization is now lazy - deferred to first use
        # This allows importing the module on read-only filesystems

    @property
    def items(self) -> dict[str, WorkItem]:
        """Access work items, triggering lazy database loading if needed.

        Dec 28, 2025: This property ensures that items are loaded from the
        database when a new WorkQueue instance is created pointing to an
        existing database file. This fixes the persistence bug where a new
        instance would have empty items until a method like add_work() was called.

        Dec 30, 2025 (P5.1): When using Raft backend, this returns a view
        of the in-memory cache which is synced from the replicated state.
        """
        if self._backend == WorkQueueBackendType.RAFT:
            # With Raft, items are managed by ReplicatedWorkQueue
            # Return local cache for compatibility
            return self._items
        if not self._db_initialized:
            self._ensure_db()
        return self._items

    @property
    def backend(self) -> str:
        """Get the active backend type.

        Returns:
            "raft" or "sqlite"
        """
        return self._backend.value

    def is_using_raft(self) -> bool:
        """Check if currently using Raft backend.

        Returns:
            True if Raft backend is active, False for SQLite
        """
        return self._backend == WorkQueueBackendType.RAFT

    def _get_backend_impl(self) -> WorkQueueBackend:
        """Get or create the backend implementation (lazy initialization).

        Jan 2, 2026: Strategy pattern backend creation. The backend is created
        on first use to ensure _get_connection is available.

        Returns:
            WorkQueueBackend implementation (Raft or SQLite)
        """
        if self._backend_impl is None:
            # Ensure database is initialized first
            self._ensure_db()

            self._backend_impl = create_backend(
                db_path=self.db_path,
                get_connection=self._get_connection,
                use_raft=self._use_raft,
                readonly_mode=self._readonly_mode,
            )

            # Sync backend type for backward compatibility
            if isinstance(self._backend_impl, RaftBackend):
                self._backend = WorkQueueBackendType.RAFT
            else:
                self._backend = WorkQueueBackendType.SQLITE

            logger.debug(f"[WorkQueue] Backend initialized: {self._backend_impl.backend_type.value}")

        return self._backend_impl

    def get_queue_stats(self) -> dict[str, Any]:
        """Get queue statistics for health monitoring.

        Dec 29, 2025: Added for master_loop.py health validation.
        Dec 30, 2025 (P5.1): Also queries Raft backend when available.
        Jan 2, 2026: Refactored to use Strategy pattern backend.

        Returns:
            Dictionary with queue health statistics.
        """
        # Jan 2, 2026: Use Strategy pattern - backend handles Raft/SQLite transparently
        backend = self._get_backend_impl()
        backend_stats = backend.get_stats()

        # Merge backend stats with WorkQueue-level stats
        result = {
            "total_items": backend_stats.get("total", 0),
            "pending": backend_stats.get("pending", 0),
            "claimed": backend_stats.get("claimed", 0),
            "running": backend_stats.get("running", 0),
            "completed": backend_stats.get("completed", 0),
            "failed": backend_stats.get("failed", 0),
            "total_added": self.stats.get("total_added", 0),
            "total_completed": self.stats.get("total_completed", 0),
            "total_failed": self.stats.get("total_failed", 0),
            "total_timeout": self.stats.get("total_timeout", 0),
            "backpressure_active": self._backpressure_active,
            "db_initialized": self._db_initialized,
            "readonly_mode": self._readonly_mode,
            "backend": backend.backend_type.value,
        }

        # Add Raft-specific fields if using Raft
        if backend.backend_type == BackendType.RAFT:
            result["raft_is_leader"] = backend_stats.get("is_leader", False)
            result["raft_leader_address"] = backend_stats.get("leader_address")
            result["raft_is_ready"] = backend_stats.get("is_ready", False)
        if backend_stats.get("fallback_active"):
            result["raft_fallback_active"] = True

        return result

    def _get_queue_stats_raft(self) -> dict[str, Any]:
        """Get queue statistics from Raft backend (Dec 30, 2025 - P5.1).

        DEPRECATED: Jan 2, 2026 - Use get_queue_stats() which now uses
        Strategy pattern backend transparently.
        """
        # Delegate to main method which handles backend selection
        return self.get_queue_stats()

    def get_claim_rejection_stats(self) -> dict[str, Any]:
        """Get claim rejection statistics for debugging job dispatch issues.

        Jan 2, 2026: Added for /dispatch/stats endpoint to diagnose why
        GPU nodes are idle despite jobs being queued.

        Returns:
            Dictionary with claim rejection breakdown by filter type.
        """
        return self._claim_rejection_stats.to_dict()

    def get_claim_rejection_stats_dict(self) -> dict[str, Any]:
        """Get enhanced claim rejection statistics for monitoring.

        January 13, 2026: Added for /work_queue/claim_stats endpoint.
        Includes computed fields like success_rate and top_rejection_reason.

        Returns:
            Dictionary with enhanced claim stats including:
            - total_attempts, successful_claims, success_rate
            - rejections breakdown by reason
            - top_rejection_reason
        """
        stats = self._claim_rejection_stats
        total = stats.total_claim_attempts or 1  # Avoid division by zero

        # Build rejections breakdown
        rejections = {
            "circuit_breaker": stats.rejected_by_circuit_breaker,
            "capability": stats.rejected_by_capability,
            "exclusion": stats.rejected_by_exclusion,
            "target_node": stats.rejected_by_target_node,
            "target_node_expired": stats.rejected_by_target_node_expired,
            "requires_gpu": stats.rejected_by_requires_gpu,
            "policy": stats.rejected_by_policy,
            "already_claimed": stats.rejected_by_already_claimed,
        }

        # Find top rejection reason
        top_reason = "none"
        top_count = 0
        for reason, count in rejections.items():
            if count > top_count:
                top_reason = reason
                top_count = count

        return {
            "total_attempts": stats.total_claim_attempts,
            "successful_claims": stats.successful_claims,
            "success_rate": stats.successful_claims / total,
            "rejections": rejections,
            "top_rejection_reason": top_reason,
            "top_rejection_count": top_count,
            "target_node_rejections": stats.target_node_rejections.copy(),
            "elapsed_seconds": time.time() - stats.last_reset_at,
            "last_reset_at": stats.last_reset_at,
        }

    def reset_claim_rejection_stats(self) -> None:
        """Reset claim rejection statistics.

        Jan 2, 2026: Call periodically to get fresh rate data.
        """
        self._claim_rejection_stats.reset()

    def clear_stale_target_nodes(self, valid_node_ids: set[str]) -> int:
        """Clear target_node from jobs targeted at non-existent nodes.

        Jan 2, 2026: Added to fix jobs stuck on old/renamed node targets.

        When nodes are renamed or removed, jobs with target_node set to those
        old names will never be claimed. This method clears target_node for
        all pending jobs where the target doesn't exist in valid_node_ids.

        Args:
            valid_node_ids: Set of currently valid node IDs in the cluster.

        Returns:
            Number of jobs that had their target_node cleared.
        """
        if not getattr(self, '_db_initialized', False):
            self._ensure_db()
        if not self._db_initialized or self._readonly_mode:
            logger.warning("Cannot clear stale targets: database not initialized or readonly")
            return 0

        cleared_count = 0
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Get all pending items with target_node set
            cursor.execute("""
                SELECT work_id, config FROM work_items
                WHERE status = 'pending'
            """)

            updates = []
            for row in cursor.fetchall():
                work_id, config_json = row
                try:
                    config = json.loads(config_json) if config_json else {}
                    target_node = config.get("target_node")
                    if target_node and target_node not in valid_node_ids:
                        # Clear the stale target_node
                        config.pop("target_node", None)
                        config.pop("target_node_expires_at", None)
                        updates.append((json.dumps(config), work_id))
                        logger.info(f"Clearing stale target_node {target_node} from work {work_id}")
                except json.JSONDecodeError:
                    continue

            # Batch update
            if updates:
                cursor.executemany("""
                    UPDATE work_items SET config = ? WHERE work_id = ?
                """, updates)
                conn.commit()
                cleared_count = len(updates)
                logger.info(f"Cleared stale target_node from {cleared_count} pending jobs")

        except sqlite3.Error as e:
            logger.error(f"Error clearing stale target nodes: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

        return cleared_count

    def _init_db(self) -> None:
        """Initialize SQLite database for work queue persistence.

        Uses context manager to ensure connection is properly closed even if
        exceptions occur during initialization (December 2025 resource leak fix).

        Jan 2026: Migrated to RetryConfig for centralized retry behavior.
        """
        # Jan 2026: Use RetryConfig for centralized retry pattern
        retry_config = RetryConfig(max_attempts=3, base_delay=0.5, max_delay=4.0)

        for attempt in retry_config.attempts():
            try:
                os.makedirs(self.db_path.parent, exist_ok=True)

                # Use context manager to ensure connection is closed on any exception
                with sqlite3.connect(str(self.db_path), timeout=SQLITE_SHORT_TIMEOUT) as conn:
                    cursor = conn.cursor()

                    # Enable WAL mode for better crash recovery and concurrent access
                    # WAL (Write-Ahead Logging) ensures data integrity on crash
                    cursor.execute("PRAGMA journal_mode=WAL")
                    cursor.execute("PRAGMA synchronous=NORMAL")  # Good balance of safety/performance
                    cursor.execute("PRAGMA wal_autocheckpoint=1000")  # Checkpoint every 1000 pages
                    cursor.execute("PRAGMA busy_timeout=10000")  # 10s timeout for locked db

                    # Work items table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS work_items (
                            work_id TEXT PRIMARY KEY,
                            work_type TEXT NOT NULL,
                            priority INTEGER NOT NULL DEFAULT 50,
                            config TEXT NOT NULL DEFAULT '{}',
                            created_at REAL NOT NULL,
                            claimed_at REAL NOT NULL DEFAULT 0.0,
                            started_at REAL NOT NULL DEFAULT 0.0,
                            completed_at REAL NOT NULL DEFAULT 0.0,
                            status TEXT NOT NULL DEFAULT 'pending',
                            claimed_by TEXT NOT NULL DEFAULT '',
                            attempts INTEGER NOT NULL DEFAULT 0,
                            max_attempts INTEGER NOT NULL DEFAULT 3,
                            timeout_seconds REAL NOT NULL DEFAULT 3600.0,
                            result TEXT NOT NULL DEFAULT '{}',
                            error TEXT NOT NULL DEFAULT '',
                            depends_on TEXT NOT NULL DEFAULT '[]'
                        )
                    """)

                    # Add depends_on column if missing (migration for existing databases)
                    try:
                        cursor.execute("SELECT depends_on FROM work_items LIMIT 1")
                    except sqlite3.OperationalError:
                        cursor.execute("ALTER TABLE work_items ADD COLUMN depends_on TEXT NOT NULL DEFAULT '[]'")

                    # Stats table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS work_stats (
                            key TEXT PRIMARY KEY,
                            value INTEGER NOT NULL DEFAULT 0
                        )
                    """)

                    # Initialize stats if not present
                    for key in ["total_added", "total_completed", "total_failed", "total_timeout"]:
                        cursor.execute(
                            "INSERT OR IGNORE INTO work_stats (key, value) VALUES (?, 0)",
                            (key,)
                        )

                    # Dec 30, 2025: Backpressure state persistence table
                    # Ensures backpressure state survives restarts
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS backpressure_state (
                            id INTEGER PRIMARY KEY CHECK (id = 1),
                            active INTEGER NOT NULL DEFAULT 0,
                            activations INTEGER NOT NULL DEFAULT 0,
                            rejections INTEGER NOT NULL DEFAULT 0,
                            last_activation_at REAL NOT NULL DEFAULT 0.0,
                            last_rejection_at REAL NOT NULL DEFAULT 0.0,
                            updated_at REAL NOT NULL DEFAULT 0.0
                        )
                    """)
                    # Insert default row if not exists
                    cursor.execute("""
                        INSERT OR IGNORE INTO backpressure_state
                        (id, active, activations, rejections, last_activation_at, last_rejection_at, updated_at)
                        VALUES (1, 0, 0, 0, 0.0, 0.0, 0.0)
                    """)

                    # Create indexes for common queries
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON work_items(status)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_priority ON work_items(priority DESC)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_claimed_by ON work_items(claimed_by)")

                    conn.commit()

                # Connection closed by context manager exit
                self._db_initialized = True
                logger.info(f"Work queue database initialized at {self.db_path}")
                return

            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt.should_retry:
                    logger.warning(
                        f"Database locked, retrying in {attempt.delay:.1f}s "
                        f"(attempt {attempt.number}/{retry_config.max_attempts})"
                    )
                    attempt.wait()
                else:
                    logger.error(f"Failed to initialize work queue database after {attempt.number} attempts: {e}")
            except (sqlite3.Error, OSError, PermissionError) as e:
                logger.error(f"Failed to initialize work queue database: {e}")
                break

        self._db_initialized = False

    def _ensure_db(self) -> bool:
        """Lazily initialize database, returns True if writable.

        December 2025: This enables import on read-only filesystems.
        Database initialization is deferred until first actual use.

        Returns:
            True if database is writable, False if readonly or unavailable
        """
        if self._db_initialized:
            return not self._readonly_mode

        try:
            self._init_db()
            if self._db_initialized:
                self._load_items()
            return self._db_initialized
        except sqlite3.OperationalError as e:
            if "readonly" in str(e).lower() or "read-only" in str(e).lower():
                self._readonly_mode = True
                self._db_initialized = True  # Mark as "initialized" in readonly mode
                logger.warning(f"[WorkQueue] Readonly mode enabled: {e}")
                return False
            # Re-raise other operational errors
            raise
        except PermissionError as e:
            self._readonly_mode = True
            self._db_initialized = True
            logger.warning(f"[WorkQueue] Readonly mode (permission denied): {e}")
            return False
        except OSError as e:
            # Handle other filesystem errors (e.g., read-only filesystem)
            if "Read-only file system" in str(e):
                self._readonly_mode = True
                self._db_initialized = True
                logger.warning(f"[WorkQueue] Readonly mode (filesystem): {e}")
                return False
            raise

    def _get_connection(self, timeout: float = 10.0) -> sqlite3.Connection:
        """Get a SQLite connection with WAL mode and proper settings.

        WAL (Write-Ahead Logging) provides:
        - Better crash recovery (uncommitted transactions can be rolled back)
        - Concurrent read access during writes
        - Better performance for mixed read/write workloads

        Returns:
            A configured sqlite3.Connection

        Raises:
            RuntimeError: If database is not initialized or in readonly mode
        """
        # Ensure database is initialized (lazy init)
        self._ensure_db()
        if not self._db_initialized:
            raise RuntimeError("WorkQueue database not initialized")
        conn = sqlite3.connect(str(self.db_path), timeout=timeout)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    @contextmanager
    def _db_connection(self, timeout: float = 10.0):
        """Context manager for safe database operations.

        Ensures connection is always closed, even if operations fail.
        Provides automatic rollback on exception and commit on success.

        December 2025: Added to fix connection leak issues.

        Usage:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM work_items")
                conn.commit()
        """
        conn = None
        try:
            conn = self._get_connection(timeout)
            conn.row_factory = sqlite3.Row
            yield conn
        except (sqlite3.Error, OSError, RuntimeError):
            if conn is not None:
                try:
                    conn.rollback()
                except sqlite3.Error:
                    pass  # Ignore rollback errors
            raise
        finally:
            if conn is not None:
                try:
                    conn.close()
                except (sqlite3.Error, OSError):
                    pass  # Suppress cleanup errors

    def _load_items(self) -> None:
        """Load work items from database on startup."""
        conn = None
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Load all non-terminal work items (pending, claimed, running)
            cursor.execute("""
                SELECT * FROM work_items
                WHERE status IN ('pending', 'claimed', 'running')
            """)

            for row in cursor.fetchall():
                # Parse depends_on safely (column may be missing in old databases)
                depends_on_raw = row["depends_on"] if "depends_on" in row.keys() else "[]"
                try:
                    depends_on = json.loads(depends_on_raw) if depends_on_raw else []
                except (json.JSONDecodeError, TypeError):
                    depends_on = []

                # Feb 2026: Cast numeric fields to prevent type errors from
                # manually inserted rows with string timestamps or priorities.
                item = WorkItem(
                    work_id=row["work_id"],
                    work_type=WorkType(row["work_type"]),
                    priority=int(row["priority"]),
                    config=json.loads(row["config"]),
                    created_at=float(row["created_at"]),
                    claimed_at=float(row["claimed_at"]),
                    started_at=float(row["started_at"]),
                    completed_at=float(row["completed_at"]),
                    status=WorkStatus(row["status"]),
                    claimed_by=row["claimed_by"],
                    attempts=int(row["attempts"]),
                    max_attempts=int(row["max_attempts"]),
                    timeout_seconds=float(row["timeout_seconds"]),
                    result=json.loads(row["result"]),
                    error=row["error"],
                    depends_on=depends_on,
                )
                self._items[item.work_id] = item

            # Load stats
            cursor.execute("SELECT key, value FROM work_stats")
            for row in cursor.fetchall():
                if row["key"] in self.stats:
                    self.stats[row["key"]] = row["value"]

            logger.info(f"Loaded {len(self._items)} work items from database")

            # Dec 30, 2025: Load backpressure state after items loaded
            # This validates state against current queue depth
            self._load_backpressure_state()
        except (sqlite3.OperationalError, sqlite3.IntegrityError) as e:
            logger.error(f"Database error loading work items: {e}")
        except (sqlite3.Error, OSError) as e:
            # Catch remaining DB errors and file system issues
            logger.error(f"Failed to load work items from database: {e}")
        finally:
            # Dec 2025: Ensure connection is closed even on error
            if conn is not None:
                try:
                    conn.close()
                except (sqlite3.Error, OSError):
                    pass  # Suppress cleanup errors to avoid masking original error

    def _save_item(self, item: WorkItem) -> None:
        """Save a work item to the database."""
        # Skip write if in readonly mode (December 2025: Lazy init)
        if self._readonly_mode:
            logger.debug(f"[WorkQueue] Skipping save for {item.work_id} (readonly mode)")
            return
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO work_items
                (work_id, work_type, priority, config, created_at, claimed_at,
                 started_at, completed_at, status, claimed_by, attempts,
                 max_attempts, timeout_seconds, result, error, depends_on)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item.work_id,
                item.work_type.value,
                item.priority,
                json.dumps(item.config),
                item.created_at,
                item.claimed_at,
                item.started_at,
                item.completed_at,
                item.status.value,
                item.claimed_by,
                item.attempts,
                item.max_attempts,
                item.timeout_seconds,
                json.dumps(item.result),
                item.error,
                json.dumps(item.depends_on),
            ))

            conn.commit()
        except sqlite3.OperationalError as e:
            # Dec 28, 2025: Check for ENOSPC and emit DISK_FULL event
            if is_enospc_error(e):
                handle_enospc_error(e, self.db_path, operation="save work item")
            logger.error(f"Database error saving work item {item.work_id}: {e}")
        except sqlite3.IntegrityError as e:
            logger.error(f"Database integrity error saving work item {item.work_id}: {e}")
        except (OSError, sqlite3.Error) as e:
            logger.error(f"Failed to save work item {item.work_id}: {e}")
        finally:
            # Dec 2025: Ensure connection is closed even on error
            if conn is not None:
                try:
                    conn.close()
                except (sqlite3.Error, OSError):
                    pass  # Suppress cleanup errors to avoid masking original error

    def _save_stats(self) -> None:
        """Save stats to the database.

        December 2025: Refactored to use context manager for safe cleanup.
        """
        # Skip write if in readonly mode (December 2025: Lazy init)
        if self._readonly_mode:
            return
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                for key, value in self.stats.items():
                    cursor.execute(
                        "INSERT OR REPLACE INTO work_stats (key, value) VALUES (?, ?)",
                        (key, value)
                    )
                conn.commit()
        except sqlite3.OperationalError as e:
            # Dec 28, 2025: Check for ENOSPC and emit DISK_FULL event
            if is_enospc_error(e):
                handle_enospc_error(e, self.db_path, operation="save work stats")
            logger.error(f"Database error saving work stats: {e}")
        except sqlite3.IntegrityError as e:
            logger.error(f"Database integrity error saving work stats: {e}")
        except (sqlite3.Error, OSError) as e:
            # Catch remaining DB errors and file system issues
            logger.error(f"Failed to save work stats: {e}")

    def _delete_item(self, work_id: str) -> None:
        """Delete a work item from the database.

        December 2025: Refactored to use context manager for safe cleanup.
        """
        # Skip write if in readonly mode (December 2025: Lazy init)
        if self._readonly_mode:
            return
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM work_items WHERE work_id = ?", (work_id,))
                conn.commit()
        except sqlite3.OperationalError as e:
            # Dec 28, 2025: Check for ENOSPC and emit DISK_FULL event
            if is_enospc_error(e):
                handle_enospc_error(e, self.db_path, operation="delete work item")
            logger.error(f"Database error deleting work item {work_id}: {e}")
        except sqlite3.IntegrityError as e:
            logger.error(f"Database integrity error deleting work item {work_id}: {e}")
        except (TypeError, ValueError, KeyError, AttributeError) as e:
            # Dec 2025: Narrowed from bare Exception - these indicate programming errors
            logger.error(f"Data error deleting work item {work_id}: {type(e).__name__}: {e}")

    def add_work(self, item: WorkItem, force: bool = False) -> str:
        """Add work to the queue. Returns work_id.

        Args:
            item: The work item to add
            force: If True, bypass backpressure limits (use for critical work)

        Returns:
            work_id on success

        Raises:
            RuntimeError: If queue is at hard limit and force=False
            RuntimeError: If cluster is in critical state and force=False

        Jan 2, 2026: Refactored to use Strategy pattern backend.
        Jan 3, 2026: Added ClusterCircuitBreaker check for cascade prevention.
        """
        # Jan 3, 2026: Check cluster health before adding work
        if not force:
            try:
                from app.coordination.node_circuit_breaker import get_cluster_circuit_breaker
                cluster_cb = get_cluster_circuit_breaker()
                if cluster_cb.should_pause_new_work():
                    status = cluster_cb.get_status()
                    raise RuntimeError(
                        f"[CLUSTER_CRITICAL] Cluster in critical state ({status.failure_ratio:.0%} nodes failing). "
                        f"Work item {item.work_id} rejected. Wait for cluster recovery or use force=True."
                    )
            except ImportError:
                pass  # ClusterCircuitBreaker not available

        with self.lock:
            # Dec 28, 2025: Check backpressure before adding
            pending = sum(1 for i in self.items.values() if i.status == WorkStatus.PENDING)
            should_reject = self._check_and_update_backpressure(pending)

            if should_reject and not force:
                self._backpressure_stats["rejections"] += 1
                self._backpressure_stats["last_rejection_at"] = time.time()
                raise RuntimeError(
                    f"[BACKPRESSURE] Queue at hard limit ({pending}/{BACKPRESSURE_HARD_LIMIT}). "
                    f"Work item {item.work_id} rejected. Wait for queue to drain or use force=True."
                )

            # Jan 2, 2026: Use Strategy pattern - backend handles Raft/SQLite transparently
            backend = self._get_backend_impl()
            result = backend.add_item(item.work_id, item.to_dict())

            if result.success:
                # Update local cache and stats
                self._items[item.work_id] = item
                self.stats["total_added"] += 1
                self._save_stats()

                backend_label = "[Raft]" if result.fallback_used is False and backend.backend_type == BackendType.RAFT else ""
                if should_reject and force:
                    logger.warning(
                        f"{backend_label} Added work {item.work_id} despite backpressure (force=True): "
                        f"{item.work_type.value} (priority: {item.priority})"
                    )
                else:
                    logger.info(f"{backend_label} Added work {item.work_id}: {item.work_type.value} (priority: {item.priority})")
            else:
                logger.warning(f"Failed to add work {item.work_id}: {result.error}")
                # Still cache locally for consistency
                self._items[item.work_id] = item

        # Notify (outside lock to avoid blocking)
        self.notifier.on_work_added(item)
        # Emit event to unified coordination (December 2025)
        self._emit_work_event("WORK_QUEUED", item)
        return item.work_id

    def _add_work_raft(self, item: WorkItem, force: bool = False) -> str:
        """Add work via Raft backend (Dec 30, 2025 - P5.1).

        DEPRECATED: Jan 2, 2026 - Use add_work() which now uses
        Strategy pattern backend transparently.
        """
        # Delegate to main method which handles backend selection
        return self.add_work(item, force)

    def add_work_batch(self, items: list[WorkItem], force: bool = False) -> list[str]:
        """Add multiple work items to the queue efficiently.

        December 29, 2025: Added for batch performance optimization.
        Uses executemany() for efficient bulk inserts instead of individual writes.
        Jan 3, 2026: Added ClusterCircuitBreaker check for cascade prevention.

        Args:
            items: List of work items to add
            force: If True, bypass backpressure limits

        Returns:
            List of work_ids for successfully added items

        Raises:
            RuntimeError: If queue is at hard limit and force=False
            RuntimeError: If cluster is in critical state and force=False
        """
        if not items:
            return []

        # Jan 3, 2026: Check cluster health before adding batch
        if not force:
            try:
                from app.coordination.node_circuit_breaker import get_cluster_circuit_breaker
                cluster_cb = get_cluster_circuit_breaker()
                if cluster_cb.should_pause_new_work():
                    status = cluster_cb.get_status()
                    raise RuntimeError(
                        f"[CLUSTER_CRITICAL] Cluster in critical state ({status.failure_ratio:.0%} nodes failing). "
                        f"Batch of {len(items)} work items rejected. Wait for cluster recovery or use force=True."
                    )
            except ImportError:
                pass  # ClusterCircuitBreaker not available

        added_ids: list[str] = []

        with self.lock:
            # Check backpressure once for the entire batch
            pending = sum(1 for i in self.items.values() if i.status == WorkStatus.PENDING)
            batch_size = len(items)

            if pending + batch_size > BACKPRESSURE_HARD_LIMIT and not force:
                self._backpressure_stats["rejections"] += batch_size
                self._backpressure_stats["last_rejection_at"] = time.time()
                raise RuntimeError(
                    f"[BACKPRESSURE] Batch of {batch_size} items would exceed hard limit "
                    f"({pending + batch_size}/{BACKPRESSURE_HARD_LIMIT}). "
                    f"Use force=True to override."
                )

            # Update in-memory state first
            for item in items:
                self.items[item.work_id] = item
                added_ids.append(item.work_id)

            self.stats["total_added"] += batch_size

            # Batch save to database using executemany
            self._save_items_batch(items)
            self._save_stats()

            if pending + batch_size > BACKPRESSURE_SOFT_LIMIT:
                self._check_and_update_backpressure(pending + batch_size)

            logger.info(f"Added batch of {batch_size} work items")

        # Emit events outside lock
        for item in items:
            self.notifier.on_work_added(item)
            self._emit_work_event("WORK_QUEUED", item)

        return added_ids

    def _save_items_batch(self, items: list[WorkItem]) -> None:
        """Save multiple work items to the database efficiently.

        December 29, 2025: Uses executemany() for O(1) database round trips
        instead of O(n) individual inserts.
        """
        if self._readonly_mode or not items:
            return

        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                # Prepare batch data
                batch_data = [
                    (
                        item.work_id,
                        item.work_type.value,
                        item.priority,
                        json.dumps(item.config),
                        item.created_at,
                        item.claimed_at,
                        item.started_at,
                        item.completed_at,
                        item.status.value,
                        item.claimed_by,
                        item.attempts,
                        item.max_attempts,
                        item.timeout_seconds,
                        json.dumps(item.result) if item.result else None,
                        item.error,
                        json.dumps(item.depends_on),
                    )
                    for item in items
                ]
                cursor.executemany("""
                    INSERT OR REPLACE INTO work_items
                    (work_id, work_type, priority, config, created_at, claimed_at,
                     started_at, completed_at, status, claimed_by, attempts,
                     max_attempts, timeout_seconds, result, error, depends_on)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, batch_data)
                conn.commit()
                logger.debug(f"Batch saved {len(items)} work items")
        except sqlite3.OperationalError as e:
            if is_enospc_error(e):
                handle_enospc_error(e, self.db_path, operation="batch save work items")
            logger.error(f"Database error in batch save: {e}")
        except sqlite3.IntegrityError as e:
            logger.error(f"Database integrity error in batch save: {e}")

    def add_training(self, board_type: str, num_players: int, priority: int = 100) -> str:
        """Convenience method to add training work."""
        item = WorkItem(
            work_type=WorkType.TRAINING,
            priority=priority,
            config={"board_type": board_type, "num_players": num_players, "model_type": "nnue"},
            timeout_seconds=7200.0,  # 2 hours for training
        )
        return self.add_work(item)

    def add_gpu_cmaes(self, board_type: str, num_players: int, priority: int = 90) -> str:
        """Convenience method to add GPU CMA-ES work."""
        item = WorkItem(
            work_type=WorkType.GPU_CMAES,
            priority=priority,
            config={"board_type": board_type, "num_players": num_players, "generations": 50},
            timeout_seconds=3600.0,
        )
        return self.add_work(item)

    def add_cpu_cmaes(self, board_type: str, num_players: int, priority: int = 60) -> str:
        """Convenience method to add CPU CMA-ES work."""
        item = WorkItem(
            work_type=WorkType.CPU_CMAES,
            priority=priority,
            config={"board_type": board_type, "num_players": num_players},
            timeout_seconds=7200.0,  # CPU CMA-ES is slower
        )
        return self.add_work(item)

    def claim_work(self, node_id: str, capabilities: list[str] | None = None) -> WorkItem | None:
        """Claim work for a node based on capabilities, policies, and dependencies.

        Uses atomic operations to prevent TOCTOU race conditions where multiple
        workers could claim the same work item.

        Dec 30, 2025 (P5.1): When Raft backend is active, uses Raft's atomic
        claim which provides cluster-wide consistency. This eliminates duplicate
        job assignments across nodes.

        Jan 2, 2026: Refactored to use Strategy pattern backend. The filtering
        logic (capabilities, exclusions, target_node, requires_gpu, policy)
        stays in WorkQueue, while atomic claim is delegated to backend.

        Args:
            node_id: The node claiming work
            capabilities: Work types this node can handle (if None, check all)

        Returns:
            WorkItem if work was claimed, None otherwise
        """
        with self.lock:
            # Jan 2, 2026: Track claim attempts for observability
            self._claim_rejection_stats.total_claim_attempts += 1

            # Jan 6, 2026: Check if claiming node's circuit is open (unhealthy)
            # This prevents assigning work to nodes that are known to be failing,
            # avoiding cascade failures where work is assigned but never completes.
            try:
                node_breaker = get_node_circuit_breaker()
                if not node_breaker.can_check(node_id):
                    self._claim_rejection_stats.rejected_by_circuit_breaker += 1
                    logger.debug(f"Node {node_id} circuit is OPEN, rejecting work claim")
                    return None
            except Exception as e:
                # Don't block claims if circuit breaker check fails
                logger.warning(f"Circuit breaker check failed for {node_id}: {e}")

            # Get set of completed work_ids for dependency checking
            completed_ids = {
                item.work_id for item in self.items.values()
                if item.status == WorkStatus.COMPLETED
            }

            # Get claimable items sorted by priority (highest first)
            claimable = [
                item for item in self.items.values()
                if item.is_claimable() and not item.has_pending_dependencies(completed_ids)
            ]

            if not claimable:
                return None

            # Sort by priority (descending)
            claimable.sort(key=lambda x: -x.priority)

            # Feb 2026: Build set of (board_type, num_players, work_type) tuples
            # already claimed by this node to prevent duplicate assignments.
            # This fixes the bug where a node could claim multiple training or
            # selfplay jobs for the same config, wasting GPU capacity.
            node_active_configs: set[tuple[str, int, str]] = set()
            for existing in self.items.values():
                if (
                    existing.status == WorkStatus.CLAIMED
                    and existing.claimed_by == node_id
                ):
                    bt = existing.config.get("board_type", "")
                    np_ = existing.config.get("num_players", 0)
                    wt = existing.work_type.value
                    if bt and np_:
                        node_active_configs.add((bt, np_, wt))

            # Find work matching capabilities and policies
            for item in claimable:
                work_type = item.work_type.value

                # Check capabilities
                if capabilities and work_type not in capabilities:
                    self._claim_rejection_stats.rejected_by_capability += 1
                    continue

                # Feb 2026: Prevent duplicate config+type assignments per node.
                # Skip if this node already has a claimed item for the same
                # (board_type, num_players, work_type) combination.
                item_bt = item.config.get("board_type", "")
                item_np = item.config.get("num_players", 0)
                if item_bt and item_np:
                    config_key = (item_bt, item_np, work_type)
                    if config_key in node_active_configs:
                        logger.debug(
                            f"Node {node_id} already has {work_type} for "
                            f"{item_bt}_{item_np}p, skipping {item.work_id}"
                        )
                        continue

                # Check if this node is excluded (set by JobReaperDaemon for failed nodes)
                excluded_nodes = item.config.get("_excluded_nodes", [])
                if node_id in excluded_nodes:
                    self._claim_rejection_stats.rejected_by_exclusion += 1
                    logger.debug(f"Node {node_id} excluded from {item.work_id}")
                    continue

                # Jan 2, 2026: Check target_node with expiration support
                # If work was queued for a specific node, only that node can claim it
                # BUT if target_node_expires_at is set and expired, clear the target_node
                target_node = item.config.get("target_node")
                target_node_expires_at = item.config.get("target_node_expires_at", 0)

                if target_node:
                    now = time.time()
                    if target_node_expires_at > 0 and now > target_node_expires_at:
                        # Target node assignment has expired - clear it so any node can claim
                        logger.info(
                            f"Work {item.work_id} target_node {target_node} expired "
                            f"(expired {now - target_node_expires_at:.0f}s ago), clearing"
                        )
                        item.config.pop("target_node", None)
                        item.config.pop("target_node_expires_at", None)
                        self._claim_rejection_stats.rejected_by_target_node_expired += 1
                        # Continue to let this or any node claim it now
                        self._save_item(item)
                    elif target_node != node_id:
                        # Not expired and wrong node - reject
                        self._claim_rejection_stats.increment_target_node_rejection(target_node)
                        logger.debug(f"Work {item.work_id} targeted for {target_node}, not {node_id}")
                        continue

                # Dec 30, 2025: Check requires_gpu flag to prevent CPU-only/coordinator nodes
                # from claiming GPU-intensive work (selfplay should run on cluster GPU nodes)
                # Jan 5, 2026: Extended to include Hetzner CPU nodes (Phase 6 - CPU Node Integration)
                requires_gpu = item.config.get("requires_gpu", False)
                if requires_gpu:
                    # Check if this is a coordinator node (no GPU, shouldn't run selfplay)
                    # Coordinator nodes are identified by known prefixes
                    coordinator_prefixes = ("mac-studio", "local-mac", "macbook", "mbp-")
                    # Jan 5, 2026: CPU-only nodes that participate in P2P but can't run GPU work
                    cpu_only_prefixes = ("hetzner-cpu",)
                    is_coordinator = any(
                        node_id.lower().startswith(prefix) for prefix in coordinator_prefixes
                    )
                    is_cpu_only = any(
                        node_id.lower().startswith(prefix) for prefix in cpu_only_prefixes
                    )
                    if is_coordinator or is_cpu_only:
                        self._claim_rejection_stats.rejected_by_requires_gpu += 1
                        node_type = "coordinator" if is_coordinator else "CPU-only node"
                        logger.debug(
                            f"Work {item.work_id} requires GPU, skipping {node_type} {node_id}"
                        )
                        continue

                # Check policy
                if self.policy_manager and not self.policy_manager.is_work_allowed(node_id, work_type):
                    self._claim_rejection_stats.rejected_by_policy += 1
                    logger.debug(f"Policy denies {work_type} on {node_id}")
                    continue

                # Jan 2, 2026: Use Strategy pattern - backend handles atomic claim
                claimed_at = time.time()
                backend = self._get_backend_impl()
                backend_result = backend.claim_item(item.work_id, node_id, claimed_at)

                if backend_result.success:
                    # Update in-memory state
                    item.status = WorkStatus.CLAIMED
                    item.claimed_by = node_id
                    item.claimed_at = claimed_at
                    item.attempts += 1
                    self._claim_rejection_stats.successful_claims += 1
                    logger.info(f"Work {item.work_id} claimed by {node_id}: {work_type}")
                    return item
                else:
                    # Another worker claimed it first, skip to next
                    self._claim_rejection_stats.rejected_by_already_claimed += 1
                    logger.debug(f"Work {item.work_id} already claimed, skipping")
                    continue

            return None

    def _atomic_claim(self, work_id: str, node_id: str, claimed_at: float) -> bool:
        """Atomically claim a work item in the database.

        DEPRECATED: Jan 2, 2026 - Use backend.claim_item() via the Strategy pattern.
        This method now delegates to the backend for backward compatibility.

        Args:
            work_id: ID of work item to claim
            node_id: Node claiming the work
            claimed_at: Timestamp of claim

        Returns:
            True if claim succeeded, False if item was already claimed
        """
        backend = self._get_backend_impl()
        result = backend.claim_item(work_id, node_id, claimed_at)
        return result.success

    def _claim_work_raft(
        self, node_id: str, capabilities: list[str] | None = None
    ) -> WorkItem | None:
        """Claim work via Raft backend (Dec 30, 2025 - P5.1).

        DEPRECATED: Jan 2, 2026 - Use claim_work() which now uses
        Strategy pattern backend transparently.
        """
        return self.claim_work(node_id, capabilities)

    def claim_work_batch(
        self,
        node_id: str,
        max_items: int = 5,
        capabilities: list[str] | None = None,
    ) -> list[WorkItem]:
        """Claim multiple work items in a single call for better utilization.

        Session 17.34 (Jan 5, 2026): Added batch claiming to reduce round-trip
        overhead and improve GPU utilization by +30-40%.

        Instead of claiming one job per request, this allows nodes to claim
        multiple jobs at once (up to their available slot capacity). This:
        - Reduces HTTP round-trips from ~100 to ~10-20 per batch
        - Allows nodes to queue work locally for immediate execution
        - Improves cluster-wide throughput by reducing claiming latency

        Args:
            node_id: The node claiming work
            max_items: Maximum number of items to claim (default: 5, max: 10)
            capabilities: Work types this node can handle (if None, check all)

        Returns:
            List of claimed WorkItems (may be empty if no work available)
        """
        max_items = min(max_items, 10)  # Hard cap at 10 to prevent hoarding
        claimed_items: list[WorkItem] = []

        with self.lock:
            # Get set of completed work_ids for dependency checking
            completed_ids = {
                item.work_id for item in self.items.values()
                if item.status == WorkStatus.COMPLETED
            }

            # Get claimable items sorted by priority (highest first)
            claimable = [
                item for item in self.items.values()
                if item.is_claimable() and not item.has_pending_dependencies(completed_ids)
            ]

            if not claimable:
                return []

            # Sort by priority (descending)
            claimable.sort(key=lambda x: -x.priority)

            # Session 17.50 (Jan 30, 2026): Optimized batch claiming
            # Feb 2026: Build set of already-claimed configs for this node
            node_active_configs: set[tuple[str, int, str]] = set()
            for existing in self.items.values():
                if (
                    existing.status == WorkStatus.CLAIMED
                    and existing.claimed_by == node_id
                ):
                    bt = existing.config.get("board_type", "")
                    np_ = existing.config.get("num_players", 0)
                    wt = existing.work_type.value
                    if bt and np_:
                        node_active_configs.add((bt, np_, wt))

            # First pass: filter candidates (no DB operations)
            candidates: list[str] = []
            candidate_items: dict[str, WorkItem] = {}

            for item in claimable:
                if len(candidates) >= max_items:
                    break

                if item.work_id in candidate_items:
                    continue

                work_type = item.work_type.value

                # Check capabilities
                if capabilities and work_type not in capabilities:
                    continue

                # Feb 2026: Prevent duplicate config+type per node in batch
                item_bt = item.config.get("board_type", "")
                item_np = item.config.get("num_players", 0)
                if item_bt and item_np:
                    config_key = (item_bt, item_np, work_type)
                    if config_key in node_active_configs:
                        continue
                    # Also prevent duplicates within this batch
                    node_active_configs.add(config_key)

                # Check if this node is excluded
                excluded_nodes = item.config.get("_excluded_nodes", [])
                if node_id in excluded_nodes:
                    continue

                # Check target_node with expiration
                target_node = item.config.get("target_node")
                target_node_expires_at = item.config.get("target_node_expires_at", 0)

                if target_node:
                    now = time.time()
                    if target_node_expires_at > 0 and now > target_node_expires_at:
                        item.config.pop("target_node", None)
                        item.config.pop("target_node_expires_at", None)
                        self._save_item(item)
                    elif target_node != node_id:
                        continue

                # Check requires_gpu
                requires_gpu = item.config.get("requires_gpu", False)
                if requires_gpu:
                    coordinator_prefixes = ("mac-studio", "local-mac", "macbook", "mbp-")
                    cpu_only_prefixes = ("hetzner-cpu",)
                    is_coordinator = any(
                        node_id.lower().startswith(prefix) for prefix in coordinator_prefixes
                    )
                    is_cpu_only = any(
                        node_id.lower().startswith(prefix) for prefix in cpu_only_prefixes
                    )
                    if is_coordinator or is_cpu_only:
                        continue

                # Check policy
                if self.policy_manager and not self.policy_manager.is_work_allowed(node_id, work_type):
                    continue

                # Item passes all filters - add to batch
                candidates.append(item.work_id)
                candidate_items[item.work_id] = item

            # Second pass: batch claim via backend (single transaction)
            if candidates:
                claimed_at = time.time()
                backend = self._get_backend_impl()
                backend_result = backend.claim_items_batch(candidates, node_id, claimed_at)

                if backend_result.success:
                    # Update in-memory state for successfully claimed items
                    claimed_ids = backend_result.data.get("claimed_ids", [])
                    for work_id in claimed_ids:
                        item = candidate_items[work_id]
                        item.status = WorkStatus.CLAIMED
                        item.claimed_by = node_id
                        item.claimed_at = claimed_at
                        item.attempts += 1
                        claimed_items.append(item)
                        logger.debug(f"Batch claim: {work_id} claimed by {node_id}")

            if claimed_items:
                logger.info(
                    f"Batch claimed {len(claimed_items)} items for {node_id}: "
                    f"{[i.work_id for i in claimed_items]}"
                )

        return claimed_items

    async def claim_work_batch_async(
        self,
        node_id: str,
        max_items: int = 5,
        capabilities: list[str] | None = None,
    ) -> list[WorkItem]:
        """Async wrapper for claim_work_batch().

        Session 17.34 (Jan 5, 2026): Added for async-safe batch claiming.
        """
        import asyncio
        return await asyncio.to_thread(
            self.claim_work_batch, node_id, max_items, capabilities
        )

    def start_work(self, work_id: str) -> bool:
        """Mark work as started (running).

        Jan 2, 2026: Refactored to use Strategy pattern backend.
        """
        with self.lock:
            # Jan 2, 2026: Use Strategy pattern - backend handles Raft/SQLite transparently
            backend = self._get_backend_impl()
            started_at = time.time()
            result = backend.start_item(work_id, started_at)

            if result.success:
                # Update local cache
                item = self._items.get(work_id)
                if item:
                    item.status = WorkStatus.RUNNING
                    item.started_at = started_at
                backend_label = "[Raft]" if backend.backend_type == BackendType.RAFT else ""
                logger.debug(f"{backend_label} Work {work_id} started")

            return result.success

    def _start_work_raft(self, work_id: str) -> bool:
        """Start work via Raft backend (Dec 30, 2025 - P5.1).

        DEPRECATED: Jan 2, 2026 - Use start_work() which now uses
        Strategy pattern backend transparently.
        """
        return self.start_work(work_id)

    def complete_work(self, work_id: str, result: dict[str, Any] | None = None) -> bool:
        """Mark work as completed successfully.

        P0.3 Dec 2025: Event emission moved inside lock for atomicity.
        This prevents work being marked COMPLETED but event never emitted
        if crash occurs between DB write and event emission.

        Jan 2, 2026: Refactored to use Strategy pattern backend.
        """
        with self.lock:
            # Jan 2, 2026: Use Strategy pattern - backend handles Raft/SQLite transparently
            backend = self._get_backend_impl()
            completed_at = time.time()
            backend_result = backend.complete_item(work_id, result, completed_at)

            if not backend_result.success:
                return False

            # Update local cache
            item = self._items.get(work_id)
            if item:
                item.status = WorkStatus.COMPLETED
                item.completed_at = completed_at
                item.result = result or {}

            self.stats["total_completed"] += 1
            self._save_stats()

            # Dec 28, 2025: Check if backpressure should be released
            pending = sum(1 for i in self.items.values() if i.status == WorkStatus.PENDING)
            self._check_and_update_backpressure(pending)

            backend_label = "[Raft]" if backend.backend_type == BackendType.RAFT else ""
            claimed_by = item.claimed_by if item else "unknown"
            logger.info(f"{backend_label} Work {work_id} completed by {claimed_by}")

            # P0.3 Dec 2025: Event emission now atomic with state change
            # Notify and emit inside lock to prevent crash window
            if item:
                try:
                    self.notifier.on_work_completed(item)
                    self._emit_work_event("WORK_COMPLETED", item, result=result or {})
                except (ImportError, RuntimeError, AttributeError) as e:
                    # Event emission failure should not break work completion
                    logger.warning(f"Failed to emit WORK_COMPLETED event: {e}")

        return True

    def _complete_work_raft(self, work_id: str, result: dict[str, Any] | None = None) -> bool:
        """Complete work via Raft backend (Dec 30, 2025 - P5.1).

        DEPRECATED: Jan 2, 2026 - Use complete_work() which now uses
        Strategy pattern backend transparently.
        """
        return self.complete_work(work_id, result)

    def fail_work(self, work_id: str, error: str = "") -> bool:
        """Mark work as failed. May be retried if attempts < max_attempts.

        P0.3 Dec 2025: Event emission moved inside lock for atomicity.
        Dec 30, 2025 (P5.1): Routes to Raft backend when available.
        Jan 2, 2026: Refactored to use Strategy pattern backend.
        """
        permanent = False
        with self.lock:
            # Get item from cache to determine if permanent
            item = self._items.get(work_id)
            if not item:
                return False

            # Determine if permanent failure
            permanent = item.attempts >= item.max_attempts
            completed_at = time.time() if permanent else None

            # Use Strategy pattern - backend handles Raft/SQLite transparently
            backend = self._get_backend_impl()
            backend_result = backend.fail_item(work_id, error, permanent, completed_at)

            if not backend_result.success:
                logger.error(f"Backend failed to mark {work_id} as failed: {backend_result.error}")
                return False

            # Update local cache
            if permanent:
                item.status = WorkStatus.FAILED
                item.completed_at = completed_at
                item.error = error
                self.stats["total_failed"] += 1
                self._save_stats()
                logger.error(f"Work {work_id} permanently failed: {error}")
            else:
                # Reset for retry
                item.status = WorkStatus.PENDING
                item.claimed_by = ""
                item.claimed_at = 0.0
                item.error = error
                logger.warning(f"Work {work_id} failed (attempt {item.attempts}), will retry: {error}")

            # Dec 28, 2025: Check if backpressure should be released (if permanent failure)
            if permanent:
                pending = sum(1 for i in self._items.values() if i.status == WorkStatus.PENDING)
                self._check_and_update_backpressure(pending)

            # P0.3 Dec 2025: Event emission now atomic with state change
            try:
                self._emit_work_event(
                    "WORK_FAILED" if permanent else "WORK_RETRY",
                    item,
                    error=error,
                    permanent=permanent,
                )
                self.notifier.on_work_failed(item, permanent=permanent)
            except Exception as e:
                logger.warning(f"Failed to emit WORK_FAILED event: {e}")

        return True

    def _fail_work_raft(self, work_id: str, error: str = "") -> bool:
        """Fail work via Raft backend (Dec 30, 2025 - P5.1).

        DEPRECATED: Jan 2, 2026 - Use fail_work() which now uses
        Strategy pattern backend transparently.
        """
        return self.fail_work(work_id, error)

    def cancel_work(self, work_id: str) -> bool:
        """Cancel pending or claimed work."""
        with self.lock:
            item = self.items.get(work_id)
            if not item or item.status in (WorkStatus.COMPLETED, WorkStatus.FAILED):
                return False

            item.status = WorkStatus.CANCELLED
            item.completed_at = time.time()
            self._save_item(item)
            logger.info(f"Work {work_id} cancelled")

            # Dec 28, 2025: Check if backpressure should be released
            pending = sum(1 for i in self.items.values() if i.status == WorkStatus.PENDING)
            self._check_and_update_backpressure(pending)

        # Dec 2025: Emit WORK_CANCELLED event for unified coordination (outside lock)
        self._emit_work_event("WORK_CANCELLED", item)
        return True

    def check_timeouts(self) -> list[str]:
        """Check for timed out work and reset for retry. Returns list of timed out work_ids."""
        timed_out = []
        to_notify = []  # (item, permanent)
        with self.lock:
            for item in self.items.values():
                if item.is_timed_out():
                    timed_out.append(item.work_id)
                    if item.attempts < item.max_attempts:
                        item.status = WorkStatus.PENDING
                        item.claimed_by = ""
                        item.claimed_at = 0.0
                        item.error = "timeout"
                        self._save_item(item)
                        to_notify.append((item, False))
                        logger.warning(f"Work {item.work_id} timed out, will retry")
                    else:
                        item.status = WorkStatus.TIMEOUT
                        item.completed_at = time.time()
                        self.stats["total_timeout"] += 1
                        self._save_item(item)
                        self._save_stats()
                        to_notify.append((item, True))
                        logger.error(f"Work {item.work_id} timed out permanently")

        # Notify (outside lock)
        for item, permanent in to_notify:
            self.notifier.on_work_timeout(item, permanent=permanent)
            # Dec 2025: Emit WORK_TIMEOUT event for unified coordination
            self._emit_work_event(
                "WORK_TIMEOUT",
                item,
                permanent=permanent,
                error="timeout",
            )

        return timed_out

    def get_queue_status(self) -> dict[str, Any]:
        """Get current queue status."""
        with self.lock:
            by_status = {}
            by_type = {}

            for item in self.items.values():
                status = item.status.value
                work_type = item.work_type.value

                by_status[status] = by_status.get(status, 0) + 1
                by_type[work_type] = by_type.get(work_type, 0) + 1

            pending = [
                item.to_dict() for item in self.items.values()
                if item.status == WorkStatus.PENDING
            ]
            running = [
                item.to_dict() for item in self.items.values()
                if item.status in (WorkStatus.CLAIMED, WorkStatus.RUNNING)
            ]

            return {
                "total_items": len(self.items),
                "by_status": by_status,
                "by_type": by_type,
                "pending": pending[:10],  # Show first 10
                "running": running,
                "stats": self.stats.copy(),
            }

    def get_work_for_node(self, node_id: str) -> list[dict[str, Any]]:
        """Get all work assigned to a node."""
        with self.lock:
            return [
                item.to_dict() for item in self.items.values()
                if item.claimed_by == node_id and item.status in (WorkStatus.CLAIMED, WorkStatus.RUNNING)
            ]

    def cleanup_old_items(self, max_age_seconds: float = 86400.0) -> int:
        """Remove completed/failed items older than max_age. Returns count removed."""
        removed = 0
        cutoff = time.time() - max_age_seconds

        with self.lock:
            to_remove = [
                work_id for work_id, item in self.items.items()
                if item.status in (WorkStatus.COMPLETED, WorkStatus.FAILED, WorkStatus.CANCELLED, WorkStatus.TIMEOUT)
                and item.completed_at > 0 and item.completed_at < cutoff
            ]
            for work_id in to_remove:
                del self.items[work_id]
                self._delete_item(work_id)
                removed += 1

        if removed:
            logger.info(f"Cleaned up {removed} old work items")
        return removed

    def cleanup_stale_items(
        self,
        max_pending_age_hours: float = WorkQueueCleanupDefaults.MAX_PENDING_AGE_HOURS,
        max_claimed_age_hours: float = WorkQueueCleanupDefaults.MAX_CLAIMED_AGE_HOURS,
    ) -> dict[str, int]:
        """Remove stale items that were never executed (December 2025).

        Handles:
        1. PENDING items older than max_pending_age - never claimed, should be removed
        2. CLAIMED items older than max_claimed_age - claimer crashed, reset to pending

        This prevents the queue from filling with items that will never execute,
        which can happen if:
        - Item config is invalid
        - All eligible workers are offline
        - Workers crash after claiming

        Args:
            max_pending_age_hours: Remove PENDING items older than this
            max_claimed_age_hours: Reset CLAIMED items older than this

        Returns:
            Dict with counts: {"removed_stale_pending": N, "reset_stale_claimed": M}
        """
        now = time.time()
        pending_cutoff = now - (max_pending_age_hours * 3600)
        claimed_cutoff = now - (max_claimed_age_hours * 3600)

        removed_pending = 0
        reset_claimed = 0

        with self.lock:
            # Find stale pending items
            to_remove = []
            for work_id, item in self.items.items():
                if (item.status == WorkStatus.PENDING
                    and item.created_at > 0
                    and item.created_at < pending_cutoff):
                    to_remove.append(work_id)
                    logger.warning(
                        f"Removing stale PENDING item: {work_id} "
                        f"(age: {(now - item.created_at) / 3600:.1f}h)"
                    )

            # Remove stale pending
            for work_id in to_remove:
                del self.items[work_id]
                self._delete_item(work_id)
                removed_pending += 1

            # Find and reset stale claimed items
            for item in self.items.values():
                if (item.status == WorkStatus.CLAIMED
                    and item.claimed_at > 0
                    and item.claimed_at < claimed_cutoff
                    and item.started_at == 0):
                    # Reset to pending for re-claim
                    logger.warning(
                        f"Resetting stale CLAIMED item: {item.work_id} "
                        f"(claimer: {item.claimed_by}, claimed {(now - item.claimed_at) / 3600:.1f}h ago)"
                    )
                    item.status = WorkStatus.PENDING
                    item.claimed_by = ""
                    item.claimed_at = 0.0
                    item.error = "reset_stale_claimed"
                    self._save_item(item)
                    reset_claimed += 1

        result = {
            "removed_stale_pending": removed_pending,
            "reset_stale_claimed": reset_claimed,
        }

        if removed_pending or reset_claimed:
            logger.info(
                f"Stale item cleanup: removed {removed_pending} pending, "
                f"reset {reset_claimed} claimed"
            )

        return result

    def get_history(self, limit: int = 50, status_filter: str | None = None) -> list[dict[str, Any]]:
        """Get work history from the database.

        Args:
            limit: Maximum number of items to return
            status_filter: Optional status to filter by (e.g., "completed", "failed")

        Returns:
            List of work items as dicts, most recent first
        """
        # Dec 28, 2025 (Wave 7 Phase 3.1): Proper connection cleanup with finally block
        conn = None
        items: list[dict[str, Any]] = []
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if status_filter:
                cursor.execute("""
                    SELECT * FROM work_items
                    WHERE status = ?
                    ORDER BY completed_at DESC, created_at DESC
                    LIMIT ?
                """, (status_filter, limit))
            else:
                cursor.execute("""
                    SELECT * FROM work_items
                    ORDER BY completed_at DESC, created_at DESC
                    LIMIT ?
                """, (limit,))

            for row in cursor.fetchall():
                items.append({
                    "work_id": row["work_id"],
                    "work_type": row["work_type"],
                    "priority": row["priority"],
                    "config": json.loads(row["config"]),
                    "created_at": row["created_at"],
                    "claimed_at": row["claimed_at"],
                    "started_at": row["started_at"],
                    "completed_at": row["completed_at"],
                    "status": row["status"],
                    "claimed_by": row["claimed_by"],
                    "attempts": row["attempts"],
                    "error": row["error"],
                })

        except (sqlite3.OperationalError, sqlite3.IntegrityError) as e:
            logger.error(f"Database error getting work history: {e}")
        except (sqlite3.Error, OSError) as e:
            # Catch remaining DB errors and file system issues
            logger.error(f"Failed to get work history: {e}")
        finally:
            if conn is not None:
                try:
                    conn.close()
                except (sqlite3.Error, OSError):
                    pass

        return items

    def get_pending_count(self) -> int:
        """Get number of pending work items."""
        with self.lock:
            return sum(1 for item in self.items.values() if item.status == WorkStatus.PENDING)

    def get_running_count(self) -> int:
        """Get number of running work items."""
        with self.lock:
            return sum(
                1 for item in self.items.values()
                if item.status in (WorkStatus.CLAIMED, WorkStatus.RUNNING)
            )

    # =========================================================================
    # Backpressure Management (Dec 28, 2025)
    # =========================================================================

    def is_backpressure_active(self) -> bool:
        """Check if backpressure is currently active."""
        return self._backpressure_active

    def get_backpressure_status(self) -> dict[str, Any]:
        """Get detailed backpressure status."""
        pending = self.get_pending_count()
        return {
            "active": self._backpressure_active,
            "pending_count": pending,
            "soft_limit": BACKPRESSURE_SOFT_LIMIT,
            "hard_limit": BACKPRESSURE_HARD_LIMIT,
            "recovery_threshold": BACKPRESSURE_RECOVERY_THRESHOLD,
            "utilization_pct": round(100.0 * pending / BACKPRESSURE_HARD_LIMIT, 1) if BACKPRESSURE_HARD_LIMIT > 0 else 0.0,
            "stats": dict(self._backpressure_stats),
        }

    def _check_and_update_backpressure(self, pending_count: int) -> bool:
        """Check backpressure state and emit events on state changes.

        Returns True if new items should be rejected (hard limit reached).
        """
        was_active = self._backpressure_active

        if pending_count >= BACKPRESSURE_HARD_LIMIT:
            # Hard limit - reject new items
            if not was_active:
                self._activate_backpressure(pending_count, "hard_limit")
            return True

        if pending_count >= BACKPRESSURE_SOFT_LIMIT:
            # Soft limit - warn but accept
            if not was_active:
                self._activate_backpressure(pending_count, "soft_limit")
            return False

        if pending_count <= BACKPRESSURE_RECOVERY_THRESHOLD:
            # Below recovery threshold - deactivate if active
            if was_active:
                self._deactivate_backpressure(pending_count)

        return False

    def _activate_backpressure(self, pending_count: int, trigger: str) -> None:
        """Activate backpressure and emit event."""
        self._backpressure_active = True
        self._backpressure_stats["activations"] += 1
        self._backpressure_stats["last_activation_at"] = time.time()

        logger.warning(
            f"[BACKPRESSURE ACTIVATED] Queue at {pending_count}/{BACKPRESSURE_HARD_LIMIT} items "
            f"(trigger: {trigger}). New job submissions may be delayed."
        )

        # Dec 30, 2025: Persist state for crash recovery
        self._persist_backpressure_state()

        # Emit event for coordination layer
        try:
            from app.coordination.event_router import get_event_bus
            from app.core.async_context import fire_and_forget
            bus = get_event_bus()
            coro = bus.publish("BACKPRESSURE_ACTIVATED", {
                "pending_count": pending_count,
                "trigger": trigger,
                "soft_limit": BACKPRESSURE_SOFT_LIMIT,
                "hard_limit": BACKPRESSURE_HARD_LIMIT,
                "timestamp": time.time(),
            })
            try:
                fire_and_forget(coro)
            except (RuntimeError, TypeError, AttributeError):
                coro.close()  # Event loop not available
        except ImportError:
            pass  # Event system not available

    def _deactivate_backpressure(self, pending_count: int) -> None:
        """Deactivate backpressure and emit event."""
        self._backpressure_active = False

        logger.info(
            f"[BACKPRESSURE RELEASED] Queue at {pending_count}/{BACKPRESSURE_HARD_LIMIT} items. "
            f"Normal job submission resumed."
        )

        # Dec 30, 2025: Persist state for crash recovery
        self._persist_backpressure_state()

        # Emit event for coordination layer
        try:
            from app.coordination.event_router import get_event_bus
            from app.core.async_context import fire_and_forget
            bus = get_event_bus()
            coro = bus.publish("BACKPRESSURE_RELEASED", {
                "pending_count": pending_count,
                "recovery_threshold": BACKPRESSURE_RECOVERY_THRESHOLD,
                "timestamp": time.time(),
            })
            try:
                fire_and_forget(coro)
            except (RuntimeError, TypeError, AttributeError):
                coro.close()  # Event loop not available
        except ImportError:
            pass  # Event system not available

    def _persist_backpressure_state(self) -> None:
        """Persist backpressure state to database for crash recovery.

        Dec 30, 2025: Ensures backpressure state survives restarts.
        """
        if not self._db_initialized or self._readonly_mode:
            return

        try:
            with sqlite3.connect(self.db_path, timeout=SQLITE_CONNECT_TIMEOUT) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE backpressure_state SET
                        active = ?,
                        activations = ?,
                        rejections = ?,
                        last_activation_at = ?,
                        last_rejection_at = ?,
                        updated_at = ?
                    WHERE id = 1
                """, (
                    1 if self._backpressure_active else 0,
                    self._backpressure_stats.get("activations", 0),
                    self._backpressure_stats.get("rejections", 0),
                    self._backpressure_stats.get("last_activation_at", 0.0),
                    self._backpressure_stats.get("last_rejection_at", 0.0),
                    time.time(),
                ))
                conn.commit()
        except sqlite3.Error as e:
            logger.warning(f"[WorkQueue] Failed to persist backpressure state: {e}")

    def _load_backpressure_state(self) -> None:
        """Load backpressure state from database on startup.

        Dec 30, 2025: Restores backpressure state after restart.
        Validates state against current queue depth to avoid stale state.
        """
        if not self._db_initialized or self._readonly_mode:
            return

        try:
            with sqlite3.connect(self.db_path, timeout=SQLITE_CONNECT_TIMEOUT) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT active, activations, rejections, last_activation_at, last_rejection_at
                    FROM backpressure_state WHERE id = 1
                """)
                row = cursor.fetchone()
                if row:
                    was_active = bool(row[0])
                    self._backpressure_stats["activations"] = row[1]
                    self._backpressure_stats["rejections"] = row[2]
                    self._backpressure_stats["last_activation_at"] = row[3]
                    self._backpressure_stats["last_rejection_at"] = row[4]

                    # Validate state against current queue depth
                    pending = sum(1 for i in self.items.values() if i.status == WorkStatus.PENDING)

                    if was_active and pending <= BACKPRESSURE_RECOVERY_THRESHOLD:
                        # Stale state - queue has drained, deactivate
                        logger.info(
                            f"[WorkQueue] Backpressure state was active but queue is below threshold "
                            f"({pending}/{BACKPRESSURE_RECOVERY_THRESHOLD}). Deactivating."
                        )
                        self._backpressure_active = False
                        self._persist_backpressure_state()
                    elif was_active:
                        logger.info(
                            f"[WorkQueue] Restored backpressure active state from DB "
                            f"(pending={pending})"
                        )
                        self._backpressure_active = True
        except sqlite3.Error as e:
            logger.warning(f"[WorkQueue] Failed to load backpressure state: {e}")

    def validate_startup_state(self) -> dict[str, Any]:
        """Validate queue state after startup and fix inconsistencies.

        Dec 30, 2025: Detects and fixes common startup issues:
        - Empty queue after restart (logs warning)
        - Stale claimed items from crashed workers
        - Backpressure active but queue empty

        Returns:
            Dict with validation results and any fixes applied
        """
        results: dict[str, Any] = {
            "issues_found": [],
            "fixes_applied": [],
            "pending_count": 0,
            "claimed_count": 0,
            "stale_claimed_count": 0,
        }

        with self.lock:
            pending = [i for i in self.items.values() if i.status == WorkStatus.PENDING]
            claimed = [i for i in self.items.values() if i.status == WorkStatus.CLAIMED]

            results["pending_count"] = len(pending)
            results["claimed_count"] = len(claimed)

            # Check for empty queue
            if len(pending) == 0 and len(claimed) == 0:
                results["issues_found"].append("queue_empty")
                logger.warning(
                    "[WorkQueue] Queue is empty after startup. "
                    "Selfplay jobs may need to be dispatched."
                )

            # Check for stale claimed items (claimed > 30 min ago)
            stale_threshold = time.time() - 1800  # 30 minutes
            stale_claimed = [
                i for i in claimed
                if i.claimed_at and i.claimed_at < stale_threshold
            ]
            results["stale_claimed_count"] = len(stale_claimed)

            if stale_claimed:
                results["issues_found"].append("stale_claimed_items")
                for item in stale_claimed:
                    # Reset stale items to pending for retry
                    item.status = WorkStatus.PENDING
                    item.claimed_by = None
                    item.claimed_at = None
                    item.retry_count += 1
                    results["fixes_applied"].append(f"reset_stale_{item.work_id}")

                logger.warning(
                    f"[WorkQueue] Reset {len(stale_claimed)} stale claimed items to pending. "
                    f"These were claimed >30 min ago and likely from crashed workers."
                )

            # Check for backpressure inconsistency
            if self._backpressure_active and len(pending) <= BACKPRESSURE_RECOVERY_THRESHOLD:
                results["issues_found"].append("stale_backpressure")
                self._backpressure_active = False
                self._persist_backpressure_state()
                results["fixes_applied"].append("deactivated_stale_backpressure")
                logger.info(
                    f"[WorkQueue] Deactivated stale backpressure state "
                    f"(pending={len(pending)} <= threshold={BACKPRESSURE_RECOVERY_THRESHOLD})"
                )

        return results

    # =========================================================================
    # Job Reaper Support Methods
    # =========================================================================

    def get_running_items(self) -> list[dict[str, Any]]:
        """Get all running work items with full details.

        Used by JobReaperDaemon to check for timed-out jobs.

        Returns:
            List of running work items as dicts with timing info
        """
        with self.lock:
            running = []
            for item in self.items.values():
                if item.status in (WorkStatus.CLAIMED, WorkStatus.RUNNING):
                    d = item.to_dict()
                    # Add extra fields for reaper
                    d["started_at"] = item.started_at or item.claimed_at
                    d["pid"] = item.config.get("pid")  # If tracked
                    running.append(d)
            return running

    def timeout_work(self, work_id: str) -> bool:
        """Mark a specific work item as timed out.

        Used by JobReaperDaemon when it detects a stuck job.
        Does NOT automatically retry - that's handled by reset_for_retry().

        Args:
            work_id: ID of the work item to timeout

        Returns:
            True if item was marked as timeout, False otherwise
        """
        with self.lock:
            item = self.items.get(work_id)
            if not item:
                return False

            if item.status not in (WorkStatus.CLAIMED, WorkStatus.RUNNING):
                return False

            item.status = WorkStatus.TIMEOUT
            item.completed_at = time.time()
            item.error = "Job timed out - killed by reaper"
            self.stats["total_timeout"] += 1
            self._save_item(item)
            self._save_stats()

            logger.warning(f"Work {work_id} marked as TIMEOUT by reaper")

        # Notify (outside lock)
        self.notifier.on_work_timeout(item, permanent=True)
        return True

    def get_retriable_items(self, max_attempts: int = 3) -> list[dict[str, Any]]:
        """Get failed/timed-out items that can be retried.

        Used by JobReaperDaemon for automatic job reassignment.

        Args:
            max_attempts: Maximum attempts before giving up

        Returns:
            List of retriable work items as dicts
        """
        with self.lock:
            retriable = []
            for item in self.items.values():
                if item.status in (WorkStatus.FAILED, WorkStatus.TIMEOUT):
                    if item.attempts < max_attempts:
                        d = item.to_dict()
                        d["failed_node"] = item.claimed_by
                        retriable.append(d)
            return retriable

    def reset_for_retry(
        self,
        work_id: str,
        excluded_nodes: list[str] | None = None,
    ) -> bool:
        """Reset a failed/timed-out work item for retry.

        Used by JobReaperDaemon for automatic job reassignment.
        The excluded_nodes list prevents reassignment to nodes that failed.

        Args:
            work_id: ID of the work item to reset
            excluded_nodes: Nodes that should not claim this work

        Returns:
            True if item was reset, False otherwise
        """
        with self.lock:
            item = self.items.get(work_id)
            if not item:
                return False

            if item.status not in (WorkStatus.FAILED, WorkStatus.TIMEOUT):
                return False

            # Store excluded nodes in config for claim_work to check
            if excluded_nodes:
                item.config["_excluded_nodes"] = list(excluded_nodes)

            # Reset for retry
            item.status = WorkStatus.PENDING
            item.claimed_by = ""
            item.claimed_at = 0.0
            item.started_at = 0.0
            # Don't reset attempts - that tracks total tries
            self._save_item(item)

            logger.info(
                f"Work {work_id} reset for retry (attempt {item.attempts + 1}), "
                f"excluding nodes: {excluded_nodes or []}"
            )
            return True

    def ensure_work_available(self, num_idle_nodes: int, max_batch: int = 10) -> int:
        """Ensure queue has enough work for idle nodes.

        Auto-generates selfplay work based on curriculum weights when the queue
        is empty. This ensures idle nodes always have work to do.

        Args:
            num_idle_nodes: Number of nodes that are currently idle
            max_batch: Maximum work items to generate at once

        Returns:
            Number of work items generated
        """
        pending = self.get_pending_count()
        if pending >= num_idle_nodes:
            return 0  # Already have enough work

        # Calculate how many items to generate
        needed = min(max_batch, num_idle_nodes - pending)
        if needed <= 0:
            return 0

        # Try to load curriculum weights for prioritized selfplay
        curriculum_weights = self._load_curriculum_weights()

        generated = 0
        for board_type, weight in curriculum_weights.items():
            if generated >= needed:
                break

            if weight <= 0:
                continue

            # Parse board type to extract num_players using canonical utility
            # Formats: "square8_2p", "hexagonal_3p", etc.
            parsed = parse_config_key(board_type)
            if parsed:
                board = parsed.board_type
                num_players = parsed.num_players
            else:
                board = board_type
                num_players = 2

            # Create selfplay work item
            item = WorkItem(
                work_type=WorkType.SELFPLAY,
                priority=int(weight * 100),  # Higher weight = higher priority
                config={
                    "board_type": board,
                    "num_players": num_players,
                    "num_games": 500,
                    "auto_generated": True,
                },
                timeout_seconds=3600.0,  # 1 hour
            )

            self.add_work(item)
            generated += 1

        if generated:
            logger.info(f"Auto-generated {generated} selfplay work items for {num_idle_nodes} idle nodes")

        return generated

    def _load_curriculum_weights(self) -> dict[str, float]:
        """Load curriculum weights for selfplay prioritization.

        Returns:
            Dict mapping board_type_players to weight (0.0-1.0)
        """
        try:
            # Try to load from curriculum module
            from app.coordination.curriculum_weights import load_curriculum_weights
            return load_curriculum_weights()
        except ImportError:
            pass

        # Fallback to default curriculum
        return {
            "square8_2p": 1.0,
            "square8_3p": 0.7,
            "square8_4p": 0.5,
            "square19_2p": 0.8,
            "hexagonal_2p": 0.6,
        }

    # =========================================================================
    # Health Check Support (December 2025)
    # =========================================================================

    def health_check(self) -> HealthCheckResult:
        """Return health status of the work queue.

        Returns:
            HealthCheckResult with queue status and metrics
        """
        with self.lock:
            pending = sum(1 for item in self.items.values() if item.status == WorkStatus.PENDING)
            running = sum(
                1 for item in self.items.values()
                if item.status in (WorkStatus.CLAIMED, WorkStatus.RUNNING)
            )
            failed = self.stats.get("total_failed", 0)
            completed = self.stats.get("total_completed", 0)

            # Check for potential issues
            issues = []
            if pending > 100:
                issues.append(f"High pending count: {pending}")
            if running > 50:
                issues.append(f"High running count: {running}")

            # Calculate error rate (avoid division by zero)
            total = completed + failed
            error_rate = (failed / total * 100) if total > 0 else 0.0

            if error_rate > 20:
                status = CoordinatorStatus.DEGRADED
                message = f"High error rate: {error_rate:.1f}%"
            elif issues:
                status = CoordinatorStatus.DEGRADED
                message = "; ".join(issues)
            else:
                status = CoordinatorStatus.RUNNING
                message = f"Healthy: {pending} pending, {running} running"

            return HealthCheckResult(
                healthy=status != CoordinatorStatus.ERROR,
                status=status,
                message=message,
                details={
                    "pending": pending,
                    "running": running,
                    "completed": completed,
                    "failed": failed,
                    "error_rate": round(error_rate, 2),
                    "total_items": len(self.items),
                    "db_path": str(self.db_path) if self.db_path else None,
                },
            )

    # =========================================================================
    # Async Wrappers for Event Loop Safety (Sprint 17.3 - January 2026)
    # =========================================================================
    #
    # These async methods wrap synchronous WorkQueue operations using
    # asyncio.to_thread() to prevent blocking the event loop. Use these
    # from HandlerBase subclasses and other async code paths.
    #
    # The underlying sync methods are unchanged for backward compatibility.

    async def add_work_async(self, item: WorkItem, force: bool = False) -> str:
        """Async wrapper for add_work().

        Adds work to the queue without blocking the event loop.

        Args:
            item: The work item to add
            force: If True, bypass backpressure limits

        Returns:
            work_id on success

        Raises:
            RuntimeError: If queue is at hard limit and force=False

        Sprint 17.3 (Jan 4, 2026): Added for async-safe work queue access.
        """
        import asyncio
        return await asyncio.to_thread(self.add_work, item, force)

    async def add_work_batch_async(
        self, items: list[WorkItem], force: bool = False
    ) -> list[str]:
        """Async wrapper for add_work_batch().

        Adds multiple work items efficiently without blocking the event loop.

        Args:
            items: List of work items to add
            force: If True, bypass backpressure limits

        Returns:
            List of work_ids for successfully added items

        Sprint 17.3 (Jan 4, 2026): Added for async-safe batch operations.
        """
        import asyncio
        return await asyncio.to_thread(self.add_work_batch, items, force)

    async def claim_work_async(
        self, node_id: str, capabilities: list[str] | None = None
    ) -> WorkItem | None:
        """Async wrapper for claim_work().

        Claims work for a node without blocking the event loop.

        Args:
            node_id: The node claiming work
            capabilities: Work types this node can handle

        Returns:
            WorkItem if work was claimed, None otherwise

        Sprint 17.3 (Jan 4, 2026): Added for async-safe work claiming.
        """
        import asyncio
        return await asyncio.to_thread(self.claim_work, node_id, capabilities)

    async def start_work_async(self, work_id: str) -> bool:
        """Async wrapper for start_work().

        Marks work as started without blocking the event loop.

        Args:
            work_id: The work item ID to start

        Returns:
            True if work was started, False otherwise

        Sprint 17.3 (Jan 4, 2026): Added for async-safe work status updates.
        """
        import asyncio
        return await asyncio.to_thread(self.start_work, work_id)

    async def complete_work_async(
        self, work_id: str, result: dict[str, Any] | None = None
    ) -> bool:
        """Async wrapper for complete_work().

        Marks work as completed without blocking the event loop.

        Args:
            work_id: The work item ID to complete
            result: Optional result data

        Returns:
            True if work was completed, False otherwise

        Sprint 17.3 (Jan 4, 2026): Added for async-safe work completion.
        """
        import asyncio
        return await asyncio.to_thread(self.complete_work, work_id, result)

    async def fail_work_async(self, work_id: str, error: str = "") -> bool:
        """Async wrapper for fail_work().

        Marks work as failed without blocking the event loop.

        Args:
            work_id: The work item ID to mark as failed
            error: Error message describing the failure

        Returns:
            True if work was marked as failed, False otherwise

        Sprint 17.3 (Jan 4, 2026): Added for async-safe work failure reporting.
        """
        import asyncio
        return await asyncio.to_thread(self.fail_work, work_id, error)

    async def get_work_async(self, work_id: str) -> WorkItem | None:
        """Get a work item by ID without blocking the event loop.

        Args:
            work_id: The work item ID to retrieve

        Returns:
            WorkItem if found, None otherwise

        Sprint 17.3 (Jan 4, 2026): Added for async-safe work retrieval.
        """
        import asyncio

        def _get_work() -> WorkItem | None:
            with self.lock:
                return self.items.get(work_id)

        return await asyncio.to_thread(_get_work)

    async def cancel_work_async(self, work_id: str) -> bool:
        """Async wrapper for cancel_work().

        Cancels a work item without blocking the event loop.

        Args:
            work_id: The work item ID to cancel

        Returns:
            True if work was cancelled, False otherwise

        Sprint 17.3 (Jan 4, 2026): Added for async-safe work cancellation.
        """
        import asyncio
        return await asyncio.to_thread(self.cancel_work, work_id)

    async def get_pending_work_async(
        self, work_type: WorkType | None = None
    ) -> list[WorkItem]:
        """Get all pending work items without blocking the event loop.

        Args:
            work_type: Optional filter by work type

        Returns:
            List of pending work items sorted by priority (highest first)

        Sprint 17.3 (Jan 4, 2026): Added for async-safe work queue queries.
        """
        import asyncio

        def _get_pending() -> list[WorkItem]:
            with self.lock:
                pending = [
                    item for item in self.items.values()
                    if item.status == WorkStatus.PENDING
                    and (work_type is None or item.work_type == work_type)
                ]
                # Sort by priority (highest first)
                pending.sort(key=lambda x: -x.priority)
                return pending

        return await asyncio.to_thread(_get_pending)

    async def health_check_async(self) -> HealthCheckResult:
        """Async wrapper for health_check().

        Returns health status without blocking the event loop.

        Returns:
            HealthCheckResult with queue status and metrics

        Sprint 17.3 (Jan 4, 2026): Added for async-safe health checks.
        """
        import asyncio
        return await asyncio.to_thread(self.health_check)

    # =========================================================================
    # Lifecycle Management (December 2025)
    # =========================================================================

    def close(self) -> None:
        """Close the work queue and release resources.

        Performs graceful shutdown:
        1. Saves current stats to database
        2. Logs queue status at shutdown
        3. Clears in-memory items cache

        Note: SQLite connections are per-operation, so no connections
        need explicit closing. WAL mode ensures durability.
        """
        with self.lock:
            # Persist final stats
            try:
                self._save_stats()
            except Exception as e:
                logger.warning(f"Failed to save final stats on close: {e}")

            # Log shutdown status
            pending = sum(1 for item in self.items.values() if item.status == WorkStatus.PENDING)
            running = sum(
                1 for item in self.items.values()
                if item.status in (WorkStatus.CLAIMED, WorkStatus.RUNNING)
            )
            logger.info(
                f"WorkQueue closing: {pending} pending, {running} running, "
                f"{self.stats.get('total_completed', 0)} completed lifetime"
            )

            # Clear in-memory cache (DB retains data for restart)
            self.items.clear()

        logger.info("WorkQueue closed")

    # =========================================================================
    # Event System Integration (December 2025)
    # =========================================================================

    def _emit_work_event(self, event_type: str, item: WorkItem, **extra) -> None:
        """Emit work queue event to unified event system.

        Integrates work queue with the coordination layer so all work
        flows through unified event routing.

        Args:
            event_type: Event type name (e.g., "WORK_QUEUED", "WORK_COMPLETED")
            item: Work item that triggered the event
            **extra: Additional payload fields
        """
        try:
            from app.coordination.event_router import get_event_bus, DataEventType

            bus = get_event_bus()
            if bus is None:
                return

            # Map string event types to DataEventType enum
            event_type_map = {
                "WORK_QUEUED": DataEventType.WORK_QUEUED,
                "WORK_CLAIMED": DataEventType.WORK_CLAIMED,
                "WORK_STARTED": DataEventType.WORK_STARTED,
                "WORK_COMPLETED": DataEventType.WORK_COMPLETED,
                "WORK_FAILED": DataEventType.WORK_FAILED,
                "WORK_RETRY": DataEventType.WORK_RETRY,
                "WORK_TIMEOUT": DataEventType.WORK_TIMEOUT,
                "WORK_CANCELLED": DataEventType.WORK_CANCELLED,
            }
            typed_event = event_type_map.get(event_type, event_type)

            payload = {
                "work_id": item.work_id,
                "work_type": item.work_type.value,
                "priority": item.priority,
                "board_type": item.config.get("board_type", ""),
                "num_players": item.config.get("num_players", 2),
                "claimed_by": item.claimed_by,
                "attempts": item.attempts,
                "timestamp": time.time(),
                **extra,
            }

            # Use fire-and-forget for non-blocking event emission
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(bus.publish(typed_event, payload))
            except RuntimeError:
                # No running loop - use sync publish
                bus.publish_sync(typed_event, payload)

        except ImportError:
            pass  # Event system not available
        except Exception as e:
            logger.debug(f"Failed to emit work event {event_type}: {e}")


# Singleton instance (created on demand by leader)
_work_queue: WorkQueue | None = None
_work_queue_lock = threading.Lock()


def get_work_queue() -> WorkQueue:
    """Get the singleton WorkQueue instance."""
    global _work_queue
    with _work_queue_lock:
        if _work_queue is None:
            _work_queue = WorkQueue()
        return _work_queue


def reset_work_queue() -> None:
    """Reset the singleton WorkQueue instance.

    Call this during graceful shutdown to clean up resources.
    After reset, the next call to get_work_queue() creates a fresh instance.
    """
    global _work_queue
    with _work_queue_lock:
        if _work_queue is not None:
            _work_queue.close()
            _work_queue = None
            logger.info("Work queue singleton reset")


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Classes
    "SlackWorkQueueNotifier",
    # Data classes
    "WorkItem",
    "WorkQueue",
    "WorkStatus",
    # Enums
    "WorkType",
    # Backend types (Dec 30, 2025 - P5.1)
    "WorkQueueBackendType",
    # Functions
    "get_work_queue",
    "reset_work_queue",
    # Raft support (Dec 30, 2025 - P5.1)
    "get_raft_work_queue",
    "reset_raft_work_queue_cache",
]
