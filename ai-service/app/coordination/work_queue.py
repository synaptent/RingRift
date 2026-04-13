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
import sys
import threading
import time
from pathlib import Path
from typing import Any

# December 2025: Import WorkStatus from canonical source
from app.coordination.types import WorkStatus  # noqa: E402
from app.coordination.contracts import HealthCheckResult  # noqa: E402
# Jan 6, 2026: Import node circuit breaker for claim filtering
from app.coordination.node_circuit_breaker import get_node_circuit_breaker  # noqa: E402

# Jan 2, 2026: Strategy pattern for Raft/SQLite backends
from app.coordination.work_queue_backends import (
    BackendType,
    WorkQueueBackend,
)
from app.coordination.work_queue_models import (
    BACKPRESSURE_HARD_LIMIT,
    BACKPRESSURE_RECOVERY_THRESHOLD,
    BACKPRESSURE_SOFT_LIMIT,
    ClaimRejectionStats,
    WorkItem,
    WorkQueueBackendType,
    WorkType,
)

logger = logging.getLogger(__name__)


# ============================================
# Work Queue Backend Selection (Dec 30, 2025 - P5.1)
# ============================================


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

        # Try to get work queue from P2P orchestrator without importing it.
        # The script entrypoint configures root logging at import time; if the
        # module is not already loaded, no in-process singleton can exist.
        try:
            orchestrator_module = sys.modules.get("scripts.p2p_orchestrator")
            if orchestrator_module is None:
                logger.debug("Raft work queue: P2P orchestrator not loaded")
                _raft_wq_available = False
                return False

            P2POrchestrator = getattr(orchestrator_module, "P2POrchestrator", None)
            if P2POrchestrator is None:
                logger.debug("Raft work queue: P2POrchestrator class not available")
                _raft_wq_available = False
                return False

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

        except AttributeError:
            logger.debug("Raft work queue: P2P orchestrator state unavailable")
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


from app.coordination.work_queue_storage import WorkQueueStorageMixin

class WorkQueue(WorkQueueStorageMixin):
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
            # Feb 2026: Inline cleanup when queue is large to prevent unbounded growth.
            # This runs inside the existing lock so no extra synchronization needed.
            # Trigger at 200 (not 500) — by 500 we're already hitting soft backpressure.
            if len(self.items) > 200:
                self._inline_cleanup()

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

            # Feb 26, 2026: Combine 3 separate iterations into a single pass.
            # Previously iterated self.items 3 times (completed_ids, claimable,
            # node_active_configs). With 1000+ items this was taking seconds.
            completed_ids: set[str] = set()
            claimable: list = []
            node_active_configs: set[tuple[str, int, str]] = set()

            for existing in self.items.values():
                if existing.status == WorkStatus.COMPLETED:
                    completed_ids.add(existing.work_id)
                elif existing.status == WorkStatus.CLAIMED and existing.claimed_by == node_id:
                    bt = existing.config.get("board_type", "")
                    np_ = existing.config.get("num_players", 0)
                    wt = existing.work_type.value
                    if bt and np_:
                        node_active_configs.add((bt, np_, wt))

            # Second pass: find claimable items (needs completed_ids from first pass)
            for item in self.items.values():
                if item.is_claimable() and not item.has_pending_dependencies(completed_ids):
                    claimable.append(item)

            if not claimable:
                return None

            # Sort by priority (descending)
            claimable.sort(key=lambda x: -x.priority)

            # Find work matching capabilities and policies
            for item in claimable:
                work_type = item.work_type.value
                config_key = str(
                    item.config.get("config_key")
                    or (
                        f"{item.config.get('board_type', '')}_{item.config.get('num_players', 0)}p"
                        if item.config.get("board_type") and item.config.get("num_players")
                        else ""
                    )
                ).strip()

                # Check capabilities
                if capabilities and work_type not in capabilities:
                    self._claim_rejection_stats.rejected_by_capability += 1
                    continue

                # Apr 2026: Manifest-backed workload-role gate for selfplay.
                # Trainers and other manifest-disabled nodes must never claim
                # P2P selfplay even if callers send broad capabilities.
                if work_type == WorkType.SELFPLAY.value:
                    try:
                        from app.config.node_roles import node_allows_work_type

                        if not node_allows_work_type(
                            node_id,
                            work_type,
                            config_key=config_key or None,
                        ):
                            self._claim_rejection_stats.rejected_by_policy += 1
                            logger.debug(
                                "Work %s denied for %s by node role policy "
                                "(work_type=%s, config_key=%s)",
                                item.work_id,
                                node_id,
                                work_type,
                                config_key or "-",
                            )
                            continue
                    except Exception as exc:
                        logger.debug(
                            "Node role gate lookup failed for %s/%s: %s",
                            node_id,
                            item.work_id,
                            exc,
                        )

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
                config_key = str(
                    item.config.get("config_key")
                    or (
                        f"{item.config.get('board_type', '')}_{item.config.get('num_players', 0)}p"
                        if item.config.get("board_type") and item.config.get("num_players")
                        else ""
                    )
                ).strip()

                # Check capabilities
                if capabilities and work_type not in capabilities:
                    continue

                if work_type == WorkType.SELFPLAY.value:
                    try:
                        from app.config.node_roles import node_allows_work_type

                        if not node_allows_work_type(
                            node_id,
                            work_type,
                            config_key=config_key or None,
                        ):
                            continue
                    except Exception as exc:
                        logger.debug(
                            "Batch node role gate lookup failed for %s/%s: %s",
                            node_id,
                            item.work_id,
                            exc,
                        )

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

    def release_work(self, work_id: str) -> bool:
        """Release a claimed work item back to PENDING without counting as a failure.

        Mar 2026: Used when a push-dispatch fails (node unreachable) to avoid
        burning through max_attempts on network errors. Unlike fail_work(), this
        decrements the attempts counter so the item gets a fresh retry via pull.

        Args:
            work_id: ID of the work item to release

        Returns:
            True if released successfully, False if item not found or not claimable
        """
        with self.lock:
            item = self._items.get(work_id)
            if not item:
                return False
            if item.status not in (WorkStatus.CLAIMED, WorkStatus.RUNNING):
                return False

            # Decrement attempts since the claim incremented it but no work was done
            item.attempts = max(0, item.attempts - 1)
            item.status = WorkStatus.PENDING
            item.claimed_by = ""
            item.claimed_at = 0.0
            item.error = ""
            self._save_item(item)
            logger.info(f"Work {work_id} released back to pending (attempts={item.attempts})")
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







    # =========================================================================
    # Backpressure Management (Dec 28, 2025)
    # =========================================================================









    # =========================================================================
    # Job Reaper Support Methods
    # =========================================================================







    # =========================================================================
    # Health Check Support (December 2025)
    # =========================================================================


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
        Mar 4, 2026: Uses dedicated executor (see claim_work_async).
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._get_wq_executor(), self.add_work, item, force
        )

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
        Mar 4, 2026: Uses dedicated executor (see claim_work_async).
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._get_wq_executor(), self.add_work_batch, items, force
        )

    def _get_wq_executor(self):
        """Get or create a dedicated thread pool for work queue operations.

        Mar 4, 2026: Work queue async methods were using asyncio.to_thread()
        which submits to the default executor (shared 8-thread P2P pool).
        Heavy operations (sync, export, gauntlet) saturate the shared pool,
        causing work claim/complete handlers to queue for 60s+ and timeout
        with 503. A dedicated 2-thread pool ensures work queue operations
        always get a thread immediately.
        """
        if not hasattr(self, "_wq_executor"):
            from concurrent.futures import ThreadPoolExecutor
            self._wq_executor = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="wq_"
            )
        return self._wq_executor

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

        Mar 4, 2026: Uses dedicated executor instead of shared P2P thread pool.
        The shared pool (8 threads) gets saturated by heavy sync/export/gauntlet
        ops, causing work claim handlers to timeout (60s) and return 503 to all
        cluster nodes. With a dedicated 2-thread pool, claims always succeed.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._get_wq_executor(), self.claim_work, node_id, capabilities
        )

    async def start_work_async(self, work_id: str) -> bool:
        """Async wrapper for start_work().

        Marks work as started without blocking the event loop.

        Args:
            work_id: The work item ID to start

        Returns:
            True if work was started, False otherwise

        Mar 4, 2026: Uses dedicated executor (see claim_work_async).
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._get_wq_executor(), self.start_work, work_id
        )

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

        Mar 4, 2026: Uses dedicated executor (see claim_work_async).
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._get_wq_executor(), self.complete_work, work_id, result
        )

    async def fail_work_async(self, work_id: str, error: str = "") -> bool:
        """Async wrapper for fail_work().

        Marks work as failed without blocking the event loop.

        Args:
            work_id: The work item ID to mark as failed
            error: Error message describing the failure

        Returns:
            True if work was marked as failed, False otherwise

        Sprint 17.3 (Jan 4, 2026): Added for async-safe work failure reporting.
        Mar 4, 2026: Uses dedicated executor (see claim_work_async).
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._get_wq_executor(), self.fail_work, work_id, error
        )

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

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._get_wq_executor(), _get_work)

    async def cancel_work_async(self, work_id: str) -> bool:
        """Async wrapper for cancel_work().

        Cancels a work item without blocking the event loop.

        Args:
            work_id: The work item ID to cancel

        Returns:
            True if work was cancelled, False otherwise

        Sprint 17.3 (Jan 4, 2026): Added for async-safe work cancellation.
        Mar 4, 2026: Uses dedicated executor (see claim_work_async).
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._get_wq_executor(), self.cancel_work, work_id
        )

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

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._get_wq_executor(), _get_pending)

    async def health_check_async(self) -> HealthCheckResult:
        """Async wrapper for health_check().

        Returns health status without blocking the event loop.

        Returns:
            HealthCheckResult with queue status and metrics

        Sprint 17.3 (Jan 4, 2026): Added for async-safe health checks.
        Mar 4, 2026: Uses dedicated executor (see claim_work_async).
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._get_wq_executor(), self.health_check
        )

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
            from app.coordination.event_router import DataEventType, publish_sync

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

            publish_sync(typed_event, payload, source="WorkQueue")

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
