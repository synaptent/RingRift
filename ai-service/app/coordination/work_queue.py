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
    work = queue.claim_work(node_id="lambda-gh200-a", capabilities=["training", "gpu_cmaes"])
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
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Default path for work queue database
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "work_queue.db"


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


class WorkStatus(str, Enum):
    """Status of a work item."""
    PENDING = "pending"
    CLAIMED = "claimed"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class WorkItem:
    """A unit of work to be executed."""
    work_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    work_type: WorkType = WorkType.SELFPLAY
    priority: int = 50  # Higher = more urgent (0-100)
    config: Dict[str, Any] = field(default_factory=dict)

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
    result: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["work_type"] = self.work_type.value
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorkItem":
        d = d.copy()
        d["work_type"] = WorkType(d.get("work_type", "selfplay"))
        d["status"] = WorkStatus(d.get("status", "pending"))
        return cls(**d)

    def is_claimable(self) -> bool:
        """Check if this work can be claimed."""
        if self.status != WorkStatus.PENDING:
            return False
        if self.attempts >= self.max_attempts:
            return False
        return True

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

    def __init__(self, policy_manager=None, db_path: Optional[Path] = None):
        self.items: Dict[str, WorkItem] = {}  # work_id -> WorkItem
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

        # Statistics
        self.stats = {
            "total_added": 0,
            "total_completed": 0,
            "total_failed": 0,
            "total_timeout": 0,
        }

        # Initialize SQLite database and load existing items
        self._init_db()
        self._load_items()

    def _init_db(self) -> None:
        """Initialize SQLite database for work queue persistence."""
        try:
            os.makedirs(self.db_path.parent, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

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
                    error TEXT NOT NULL DEFAULT ''
                )
            """)

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
            logger.info(f"Work queue database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize work queue database: {e}")

    def _load_items(self) -> None:
        """Load work items from database on startup."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Load all non-terminal work items (pending, claimed, running)
            cursor.execute("""
                SELECT * FROM work_items
                WHERE status IN ('pending', 'claimed', 'running')
            """)

            for row in cursor.fetchall():
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
                )
                self.items[item.work_id] = item

            # Load stats
            cursor.execute("SELECT key, value FROM work_stats")
            for row in cursor.fetchall():
                if row["key"] in self.stats:
                    self.stats[row["key"]] = row["value"]

            conn.close()
            logger.info(f"Loaded {len(self.items)} work items from database")
        except Exception as e:
            logger.error(f"Failed to load work items from database: {e}")

    def _save_item(self, item: WorkItem) -> None:
        """Save a work item to the database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO work_items
                (work_id, work_type, priority, config, created_at, claimed_at,
                 started_at, completed_at, status, claimed_by, attempts,
                 max_attempts, timeout_seconds, result, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            ))

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save work item {item.work_id}: {e}")

    def _save_stats(self) -> None:
        """Save stats to the database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            for key, value in self.stats.items():
                cursor.execute(
                    "INSERT OR REPLACE INTO work_stats (key, value) VALUES (?, ?)",
                    (key, value)
                )

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save work stats: {e}")

    def _delete_item(self, work_id: str) -> None:
        """Delete a work item from the database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("DELETE FROM work_items WHERE work_id = ?", (work_id,))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to delete work item {work_id}: {e}")

    def add_work(self, item: WorkItem) -> str:
        """Add work to the queue. Returns work_id."""
        with self.lock:
            self.items[item.work_id] = item
            self.stats["total_added"] += 1
            self._save_item(item)
            self._save_stats()
            logger.info(f"Added work {item.work_id}: {item.work_type.value} (priority: {item.priority})")
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

    def claim_work(self, node_id: str, capabilities: Optional[List[str]] = None) -> Optional[WorkItem]:
        """Claim work for a node based on capabilities and policies.

        Args:
            node_id: The node claiming work
            capabilities: Work types this node can handle (if None, check all)

        Returns:
            WorkItem if work was claimed, None otherwise
        """
        with self.lock:
            # Get claimable items sorted by priority (highest first)
            claimable = [
                item for item in self.items.values()
                if item.is_claimable()
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

                # Check policy
                if self.policy_manager:
                    if not self.policy_manager.is_work_allowed(node_id, work_type):
                        logger.debug(f"Policy denies {work_type} on {node_id}")
                        continue

                # Claim it
                item.status = WorkStatus.CLAIMED
                item.claimed_by = node_id
                item.claimed_at = time.time()
                item.attempts += 1
                self._save_item(item)

                logger.info(f"Work {item.work_id} claimed by {node_id}: {work_type}")
                return item

            return None

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

    def complete_work(self, work_id: str, result: Optional[Dict[str, Any]] = None) -> bool:
        """Mark work as completed successfully."""
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
            return True

    def fail_work(self, work_id: str, error: str = "") -> bool:
        """Mark work as failed. May be retried if attempts < max_attempts."""
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
                item.status = WorkStatus.FAILED
                item.completed_at = time.time()
                item.error = error
                self.stats["total_failed"] += 1
                self._save_item(item)
                self._save_stats()
                logger.error(f"Work {work_id} permanently failed: {error}")

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
            return True

    def check_timeouts(self) -> List[str]:
        """Check for timed out work and reset for retry. Returns list of timed out work_ids."""
        timed_out = []
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
                        logger.warning(f"Work {item.work_id} timed out, will retry")
                    else:
                        item.status = WorkStatus.TIMEOUT
                        item.completed_at = time.time()
                        self.stats["total_timeout"] += 1
                        self._save_item(item)
                        self._save_stats()
                        logger.error(f"Work {item.work_id} timed out permanently")

        return timed_out

    def get_queue_status(self) -> Dict[str, Any]:
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

    def get_work_for_node(self, node_id: str) -> List[Dict[str, Any]]:
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

    def get_history(self, limit: int = 50, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get work history from the database.

        Args:
            limit: Maximum number of items to return
            status_filter: Optional status to filter by (e.g., "completed", "failed")

        Returns:
            List of work items as dicts, most recent first
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
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


# Singleton instance (created on demand by leader)
_work_queue: Optional[WorkQueue] = None


def get_work_queue() -> WorkQueue:
    """Get the singleton WorkQueue instance."""
    global _work_queue
    if _work_queue is None:
        _work_queue = WorkQueue()
    return _work_queue
