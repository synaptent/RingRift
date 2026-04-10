"""Storage, cleanup, and health helpers for ``WorkQueue``."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Any

from app.config.coordination_defaults import WorkQueueCleanupDefaults
from app.config.thresholds import SQLITE_CONNECT_TIMEOUT, SQLITE_SHORT_TIMEOUT
from app.coordination.contracts import CoordinatorStatus, HealthCheckResult
from app.coordination.event_utils import parse_config_key
from app.coordination.types import WorkStatus
from app.coordination.work_queue import (
    BACKPRESSURE_HARD_LIMIT,
    BACKPRESSURE_RECOVERY_THRESHOLD,
    BACKPRESSURE_SOFT_LIMIT,
    WorkItem,
    WorkQueueBackendType,
    WorkType,
)
from app.coordination.work_queue_backends import (
    BackendType,
    RaftBackend,
    WorkQueueBackend,
    create_backend,
)
from app.utils.disk_utils import handle_enospc_error, is_enospc_error
from app.utils.retry import RetryConfig

logger = logging.getLogger(__name__)


class WorkQueueStorageMixin:
    """SQLite persistence, cleanup, queue status, and health helpers."""

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
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=timeout)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=10000")
            return conn
        except sqlite3.ProgrammingError as e:
            if "closed database" in str(e).lower():
                # Feb 2026: Connection wrapper may have been prematurely closed.
                # Re-create a fresh connection.
                logger.warning(f"[WorkQueue] Retrying after closed database: {e}")
                conn = sqlite3.connect(str(self.db_path), timeout=timeout)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA busy_timeout=10000")
                return conn
            raise
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
    def _inline_cleanup(self) -> int:
        """Fast inline cleanup of stale items. Called from add_work() under lock.

        Feb 2026: Prevents unbounded queue growth between hourly daemon cleanups.
        Uses aggressive TTLs since stale items degrade claim_work() O(n) performance.
        """
        now = time.time()
        # Read TTLs from config defaults (not hardcoded) to stay in sync
        try:
            from app.config.coordination_defaults import WorkQueueCleanupDefaults
            max_pending_age = WorkQueueCleanupDefaults.MAX_PENDING_AGE_HOURS * 3600
            max_claimed_age = WorkQueueCleanupDefaults.MAX_CLAIMED_AGE_HOURS * 3600
        except ImportError:
            max_pending_age = 1.5 * 3600
            max_claimed_age = 1.0 * 3600
        max_terminal_age = 2 * 3600  # completed/failed can be cleaned fast
        to_remove = []
        for wid, item in self.items.items():
            age = now - item.created_at
            if item.status == WorkStatus.PENDING and age > max_pending_age:
                to_remove.append(wid)
            elif item.status == WorkStatus.CLAIMED and age > max_claimed_age:
                to_remove.append(wid)
            elif item.status in (WorkStatus.COMPLETED, WorkStatus.FAILED, WorkStatus.CANCELLED, WorkStatus.TIMEOUT) and age > max_terminal_age:
                to_remove.append(wid)
        for wid in to_remove:
            del self.items[wid]
            self._delete_item(wid)
        if to_remove:
            logger.info(f"Inline cleanup removed {len(to_remove)} stale work items")
        return len(to_remove)
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
                item_dict: dict[str, Any] = {
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
                }
                # Feb 2026: Include result for completed items (needed for
                # cross-process evaluation completion polling)
                try:
                    item_dict["result"] = json.loads(row["result"]) if row["result"] else {}
                except (json.JSONDecodeError, KeyError):
                    item_dict["result"] = {}
                items.append(item_dict)

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
            from app.coordination.event_router import publish_sync

            publish_sync(
                "BACKPRESSURE_ACTIVATED",
                {
                    "pending_count": pending_count,
                    "trigger": trigger,
                    "soft_limit": BACKPRESSURE_SOFT_LIMIT,
                    "hard_limit": BACKPRESSURE_HARD_LIMIT,
                    "timestamp": time.time(),
                },
                source="WorkQueue",
            )
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
            from app.coordination.event_router import publish_sync

            publish_sync(
                "BACKPRESSURE_RELEASED",
                {
                    "pending_count": pending_count,
                    "recovery_threshold": BACKPRESSURE_RECOVERY_THRESHOLD,
                    "timestamp": time.time(),
                },
                source="WorkQueue",
            )
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
