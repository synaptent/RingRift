"""Centralized Work Queue for Cluster Work Distribution.

The leader maintains a prioritized work queue. Workers pull appropriate work
based on their capabilities and policies.

Architecture:
- Leader: Maintains work queue, assigns work to workers
- Workers: Poll for work, report completion/failure
- Work items: Typed (training, cmaes, tournament, etc.) with priorities

Usage:
    # On leader
    queue = WorkQueue()
    queue.add_work(WorkItem(work_type="training", config={"board": "square8"}))

    # On worker (via API)
    work = queue.claim_work(node_id="gpu-node-1", capabilities=["training", "gpu_cmaes"])
    # ... do work ...
    queue.complete_work(work.work_id)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# December 2025: Import WorkStatus from canonical source
from app.coordination.types import WorkStatus  # noqa: E402
from app.coordination.contracts import CoordinatorStatus, HealthCheckResult  # noqa: E402

logger = logging.getLogger(__name__)

# Default path for work queue database
# Respect RINGRIFT_WORK_QUEUE_DB environment variable for consistency across all components
_DEFAULT_DB_DIR = Path(__file__).parent.parent.parent / "data"
DEFAULT_DB_PATH = Path(os.environ.get("RINGRIFT_WORK_QUEUE_DB", str(_DEFAULT_DB_DIR / "work_queue.db")))


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
        except Exception as e:
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
        return cls(**d)

    def is_claimable(self) -> bool:
        """Check if this work can be claimed (doesn't check dependencies)."""
        if self.status != WorkStatus.PENDING:
            return False
        return not self.attempts >= self.max_attempts

    def has_pending_dependencies(self, completed_ids: set) -> bool:
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


class WorkQueue:
    """Centralized work queue managed by the leader.

    Features:
    - Priority-based scheduling
    - Capability-based work matching
    - Policy enforcement
    - Timeout handling
    - Retry logic
    - SQLite persistence for durability across leader changes
    """

    def __init__(self, policy_manager=None, db_path: Path | None = None, slack_webhook: str | None = None):
        self.items: dict[str, WorkItem] = {}  # work_id -> WorkItem
        self.lock = threading.RLock()
        self.db_path = db_path or DEFAULT_DB_PATH

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

        # Statistics
        self.stats = {
            "total_added": 0,
            "total_completed": 0,
            "total_failed": 0,
            "total_timeout": 0,
        }

        # Track initialization state (December 2025: Lazy initialization)
        self._db_initialized = False
        self._readonly_mode = False

        # Database initialization is now lazy - deferred to first use
        # This allows importing the module on read-only filesystems

    def _init_db(self) -> None:
        """Initialize SQLite database for work queue persistence."""
        max_retries = 3
        retry_delay = 0.5

        for attempt in range(max_retries):
            try:
                os.makedirs(self.db_path.parent, exist_ok=True)
                conn = sqlite3.connect(str(self.db_path), timeout=10.0)
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

                # Create indexes for common queries
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON work_items(status)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_priority ON work_items(priority DESC)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_claimed_by ON work_items(claimed_by)")

                conn.commit()
                conn.close()
                self._db_initialized = True
                logger.info(f"Work queue database initialized at {self.db_path}")
                return

            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(f"Database locked, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"Failed to initialize work queue database after {attempt + 1} attempts: {e}")
            except Exception as e:
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

                item = WorkItem(
                    work_id=row["work_id"],
                    work_type=WorkType(row["work_type"]),
                    priority=row["priority"],
                    config=json.loads(row["config"]),
                    created_at=row["created_at"],
                    claimed_at=row["claimed_at"],
                    started_at=row["started_at"],
                    completed_at=row["completed_at"],
                    status=WorkStatus(row["status"]),
                    claimed_by=row["claimed_by"],
                    attempts=row["attempts"],
                    max_attempts=row["max_attempts"],
                    timeout_seconds=row["timeout_seconds"],
                    result=json.loads(row["result"]),
                    error=row["error"],
                    depends_on=depends_on,
                )
                self.items[item.work_id] = item

            # Load stats
            cursor.execute("SELECT key, value FROM work_stats")
            for row in cursor.fetchall():
                if row["key"] in self.stats:
                    self.stats[row["key"]] = row["value"]

            logger.info(f"Loaded {len(self.items)} work items from database")
        except (sqlite3.OperationalError, sqlite3.IntegrityError) as e:
            logger.error(f"Database error loading work items: {e}")
        except Exception as e:
            logger.error(f"Failed to load work items from database: {e}")
        finally:
            # Dec 2025: Ensure connection is closed even on error
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

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
        except (sqlite3.OperationalError, sqlite3.IntegrityError) as e:
            logger.error(f"Database error saving work item {item.work_id}: {e}")
        except Exception as e:
            logger.error(f"Failed to save work item {item.work_id}: {e}")
        finally:
            # Dec 2025: Ensure connection is closed even on error
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def _save_stats(self) -> None:
        """Save stats to the database."""
        # Skip write if in readonly mode (December 2025: Lazy init)
        if self._readonly_mode:
            return
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            for key, value in self.stats.items():
                cursor.execute(
                    "INSERT OR REPLACE INTO work_stats (key, value) VALUES (?, ?)",
                    (key, value)
                )

            conn.commit()
        except (sqlite3.OperationalError, sqlite3.IntegrityError) as e:
            logger.error(f"Database error saving work stats: {e}")
        except Exception as e:
            logger.error(f"Failed to save work stats: {e}")
        finally:
            # Dec 2025: Ensure connection is closed even on error
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def _delete_item(self, work_id: str) -> None:
        """Delete a work item from the database."""
        # Skip write if in readonly mode (December 2025: Lazy init)
        if self._readonly_mode:
            return
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM work_items WHERE work_id = ?", (work_id,))
            conn.commit()
        except (sqlite3.OperationalError, sqlite3.IntegrityError) as e:
            logger.error(f"Database error deleting work item {work_id}: {e}")
        except Exception as e:
            logger.error(f"Failed to delete work item {work_id}: {e}")
        finally:
            # Dec 2025: Ensure connection is closed even on error
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def add_work(self, item: WorkItem) -> str:
        """Add work to the queue. Returns work_id."""
        with self.lock:
            self.items[item.work_id] = item
            self.stats["total_added"] += 1
            self._save_item(item)
            self._save_stats()
            logger.info(f"Added work {item.work_id}: {item.work_type.value} (priority: {item.priority})")
        # Notify (outside lock to avoid blocking)
        self.notifier.on_work_added(item)
        # Emit event to unified coordination (December 2025)
        self._emit_work_event("WORK_QUEUED", item)
        return item.work_id

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

        Uses atomic database operations to prevent TOCTOU race conditions where
        multiple workers could claim the same work item. The claim is performed
        via a conditional UPDATE that only succeeds if the item is still PENDING.

        Args:
            node_id: The node claiming work
            capabilities: Work types this node can handle (if None, check all)

        Returns:
            WorkItem if work was claimed, None otherwise
        """
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
                return None

            # Sort by priority (descending)
            claimable.sort(key=lambda x: -x.priority)

            # Find work matching capabilities and policies
            for item in claimable:
                work_type = item.work_type.value

                # Check capabilities
                if capabilities and work_type not in capabilities:
                    continue

                # Check if this node is excluded (set by JobReaperDaemon for failed nodes)
                excluded_nodes = item.config.get("_excluded_nodes", [])
                if node_id in excluded_nodes:
                    logger.debug(f"Node {node_id} excluded from {item.work_id}")
                    continue

                # Check policy
                if self.policy_manager and not self.policy_manager.is_work_allowed(node_id, work_type):
                    logger.debug(f"Policy denies {work_type} on {node_id}")
                    continue

                # Attempt atomic claim via database (December 2025 - TOCTOU fix)
                claimed_at = time.time()
                if self._atomic_claim(item.work_id, node_id, claimed_at):
                    # Update in-memory state
                    item.status = WorkStatus.CLAIMED
                    item.claimed_by = node_id
                    item.claimed_at = claimed_at
                    item.attempts += 1
                    logger.info(f"Work {item.work_id} claimed by {node_id}: {work_type}")
                    return item
                else:
                    # Another worker claimed it first, skip to next
                    logger.debug(f"Work {item.work_id} already claimed, skipping")
                    continue

            return None

    def _atomic_claim(self, work_id: str, node_id: str, claimed_at: float) -> bool:
        """Atomically claim a work item in the database.

        Uses a conditional UPDATE with WHERE status='pending' to ensure
        only one worker can claim the item. This prevents TOCTOU race conditions.

        Args:
            work_id: ID of work item to claim
            node_id: Node claiming the work
            claimed_at: Timestamp of claim

        Returns:
            True if claim succeeded, False if item was already claimed
        """
        # Check if database is initialized and writable (December 2025: Lazy init)
        if not getattr(self, '_db_initialized', False):
            self._ensure_db()
        if not self._db_initialized or self._readonly_mode:
            logger.warning(f"Cannot claim {work_id}: database not initialized or readonly")
            return False

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Use BEGIN IMMEDIATE for write intent, preventing concurrent claims
            cursor.execute("BEGIN IMMEDIATE")

            # Get current attempts count first
            cursor.execute(
                "SELECT attempts FROM work_items WHERE work_id = ?",
                (work_id,)
            )
            row = cursor.fetchone()
            if not row:
                conn.rollback()
                conn.close()
                return False

            current_attempts = row[0]

            # Atomic conditional update - only succeeds if still PENDING
            cursor.execute("""
                UPDATE work_items
                SET status = 'claimed',
                    claimed_by = ?,
                    claimed_at = ?,
                    attempts = ?
                WHERE work_id = ? AND status = 'pending'
            """, (node_id, claimed_at, current_attempts + 1, work_id))

            # Check if update affected exactly one row
            if cursor.rowcount == 1:
                conn.commit()
                conn.close()
                return True
            else:
                # Item was already claimed by another worker
                conn.rollback()
                conn.close()
                return False

        except Exception as e:
            logger.error(f"Atomic claim failed for {work_id}: {e}")
            try:
                conn.rollback()
                conn.close()
            except (sqlite3.Error, OSError):
                pass
            return False

    def start_work(self, work_id: str) -> bool:
        """Mark work as started (running)."""
        with self.lock:
            item = self.items.get(work_id)
            if not item or item.status != WorkStatus.CLAIMED:
                return False

            item.status = WorkStatus.RUNNING
            item.started_at = time.time()
            self._save_item(item)
            return True

    def complete_work(self, work_id: str, result: dict[str, Any] | None = None) -> bool:
        """Mark work as completed successfully.

        P0.3 Dec 2025: Event emission moved inside lock for atomicity.
        This prevents work being marked COMPLETED but event never emitted
        if crash occurs between DB write and event emission.
        """
        with self.lock:
            item = self.items.get(work_id)
            if not item or item.status not in (WorkStatus.CLAIMED, WorkStatus.RUNNING):
                return False

            item.status = WorkStatus.COMPLETED
            item.completed_at = time.time()
            item.result = result or {}
            self.stats["total_completed"] += 1
            self._save_item(item)
            self._save_stats()

            logger.info(f"Work {work_id} completed by {item.claimed_by}")

            # P0.3 Dec 2025: Event emission now atomic with state change
            # Notify and emit inside lock to prevent crash window
            try:
                self.notifier.on_work_completed(item)
                self._emit_work_event("WORK_COMPLETED", item, result=result or {})
            except Exception as e:
                # Event emission failure should not break work completion
                logger.warning(f"Failed to emit WORK_COMPLETED event: {e}")

        return True

    def fail_work(self, work_id: str, error: str = "") -> bool:
        """Mark work as failed. May be retried if attempts < max_attempts.

        P0.3 Dec 2025: Event emission moved inside lock for atomicity.
        """
        permanent = False
        with self.lock:
            item = self.items.get(work_id)
            if not item:
                return False

            if item.attempts < item.max_attempts:
                # Reset for retry
                item.status = WorkStatus.PENDING
                item.claimed_by = ""
                item.claimed_at = 0.0
                item.error = error
                self._save_item(item)
                logger.warning(f"Work {work_id} failed (attempt {item.attempts}), will retry: {error}")
            else:
                # Permanently failed
                permanent = True
                item.status = WorkStatus.FAILED
                item.completed_at = time.time()
                item.error = error
                self.stats["total_failed"] += 1
                self._save_item(item)
                self._save_stats()
                logger.error(f"Work {work_id} permanently failed: {error}")

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
        max_pending_age_hours: float = 24.0,
        max_claimed_age_hours: float = 1.0,
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

            items = []
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

            conn.close()
            return items
        except Exception as e:
            logger.error(f"Failed to get work history: {e}")
            return []

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

            # Parse board type to extract num_players
            # Formats: "square8_2p", "hexagonal_3p", etc.
            parts = board_type.rsplit("_", 1)
            if len(parts) == 2 and parts[1].endswith("p"):
                board = parts[0]
                try:
                    num_players = int(parts[1].rstrip("p"))
                except ValueError:
                    num_players = 2
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
    # Functions
    "get_work_queue",
    "reset_work_queue",
]
