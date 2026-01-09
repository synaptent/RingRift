"""PersistentEvaluationQueue - SQLite-backed priority queue for model evaluation.

Sprint 13 Session 4 (January 3, 2026): Part of model evaluation automation.

This module provides a persistent, priority-based queue for model evaluation requests.
Unlike the in-memory queue in EvaluationDaemon, this queue:
- Survives daemon restart
- Supports priority weighting (curriculum-aware)
- Enables stuck evaluation recovery
- Provides deduplication

Key Features:
- SQLite-backed persistence for queue state
- Priority-based retrieval (underserved configs first)
- Stuck evaluation detection and recovery
- Integration with existing EvaluationDaemon

Usage:
    from app.coordination.evaluation_queue import (
        PersistentEvaluationQueue,
        EvaluationRequest,
        get_evaluation_queue,
    )

    # Get singleton instance
    queue = get_evaluation_queue()

    # Add evaluation request
    request_id = queue.add_request(
        model_path="/path/to/model.pth",
        board_type="hex8",
        num_players=4,
        priority=75,  # Higher = sooner
        source="owc_import",
    )

    # Claim next request for processing
    request = queue.claim_next()
    if request:
        # Process evaluation...
        queue.complete(request.request_id, elo=1450.0)
        # Or on failure:
        queue.fail(request.request_id, "GPU OOM")
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.coordination.contracts import HealthCheckResult
from app.coordination.event_utils import make_config_key

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_QUEUE_PATH = Path("data/coordination/evaluation_queue.db")

# Board-specific stuck evaluation timeouts (seconds)
# Larger boards need more evaluation time, so stuck detection uses longer timeouts
STUCK_TIMEOUT_SECONDS = {
    "hex8": 3600,       # 1 hour
    "square8": 7200,    # 2 hours
    "square19": 10800,  # 3 hours
    "hexagonal": 14400, # 4 hours
}
DEFAULT_STUCK_TIMEOUT = 7200  # 2 hours fallback

# Request status values
class RequestStatus:
    """Request status constants."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class EvaluationRequest:
    """An evaluation request in the queue."""

    request_id: str
    model_path: str
    board_type: str
    num_players: int
    config_key: str
    status: str
    priority: int
    created_at: float
    started_at: float
    completed_at: float
    attempts: int
    max_attempts: int
    error: str
    result_elo: float | None
    source: str
    harness_type: str = ""  # Jan 2026: AI harness type for composite Elo tracking

    @property
    def is_stuck(self) -> bool:
        """Check if this request is stuck (running too long)."""
        if self.status != RequestStatus.RUNNING:
            return False
        timeout = STUCK_TIMEOUT_SECONDS.get(self.board_type, DEFAULT_STUCK_TIMEOUT)
        return time.time() - self.started_at > timeout

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "model_path": self.model_path,
            "board_type": self.board_type,
            "num_players": self.num_players,
            "config_key": self.config_key,
            "status": self.status,
            "priority": self.priority,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "error": self.error,
            "result_elo": self.result_elo,
            "source": self.source,
            "harness_type": self.harness_type,
        }


@dataclass
class QueueStats:
    """Statistics for the evaluation queue."""

    pending_count: int = 0
    running_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    stuck_recoveries: int = 0
    duplicate_requests_skipped: int = 0


class PersistentEvaluationQueue:
    """SQLite-backed priority queue for model evaluation.

    Thread-safe implementation with SQLite for persistence.
    """

    def __init__(
        self,
        db_path: Path | str = DEFAULT_QUEUE_PATH,
        max_attempts: int = 3,
    ):
        """Initialize the evaluation queue.

        Args:
            db_path: Path to SQLite database for persistence
            max_attempts: Maximum evaluation attempts before marking as failed
        """
        self.db_path = Path(db_path)
        self.max_attempts = max_attempts

        # Thread lock for concurrent access
        self._lock = threading.RLock()

        # Statistics
        self._stats = QueueStats()

        # Initialize database
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_requests (
                    request_id TEXT PRIMARY KEY,
                    model_path TEXT NOT NULL,
                    board_type TEXT NOT NULL,
                    num_players INTEGER NOT NULL,
                    config_key TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    priority INTEGER NOT NULL DEFAULT 50,
                    created_at REAL NOT NULL,
                    started_at REAL DEFAULT 0.0,
                    completed_at REAL DEFAULT 0.0,
                    attempts INTEGER DEFAULT 0,
                    max_attempts INTEGER DEFAULT 3,
                    error TEXT DEFAULT '',
                    result_elo REAL,
                    source TEXT DEFAULT 'training',
                    harness_type TEXT DEFAULT '',
                    UNIQUE(model_path, config_key, harness_type)
                )
            """)

            # Jan 2026: Add harness_type column to existing databases
            try:
                conn.execute("ALTER TABLE evaluation_requests ADD COLUMN harness_type TEXT DEFAULT ''")
                conn.commit()
            except sqlite3.OperationalError:
                pass  # Column already exists

            # Index for efficient status + priority queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status_priority
                ON evaluation_requests(status, priority DESC)
            """)

            # Index for stuck detection (running requests ordered by start time)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_started_at
                ON evaluation_requests(started_at)
            """)

            # Index for model path lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_path
                ON evaluation_requests(model_path)
            """)

            conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def add_request(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
        priority: int = 50,
        source: str = "training",
        harness_type: str = "",
    ) -> str | None:
        """Add an evaluation request to the queue.

        Uses UPSERT semantics: if a request for the same model+config+harness
        exists and is still pending, update priority if higher. If already
        completed/failed/running, skip.

        Args:
            model_path: Path to the model file
            board_type: Board type (hex8, square8, etc.)
            num_players: Number of players (2, 3, or 4)
            priority: Priority (higher = evaluated sooner), default 50
            source: Source of the request (training, scanner, owc_import)
            harness_type: AI harness type for composite Elo (Jan 2026)

        Returns:
            Request ID if added/updated, None if skipped (duplicate)
        """
        config_key = make_config_key(board_type, num_players)
        request_id = str(uuid.uuid4())

        with self._lock:
            with self._get_connection() as conn:
                # Check for existing request (now includes harness_type)
                existing = conn.execute(
                    """
                    SELECT request_id, status, priority
                    FROM evaluation_requests
                    WHERE model_path = ? AND config_key = ? AND harness_type = ?
                    """,
                    (model_path, config_key, harness_type),
                ).fetchone()

                if existing:
                    # If completed or failed, skip (already evaluated)
                    if existing["status"] in (RequestStatus.COMPLETED, RequestStatus.FAILED):
                        self._stats.duplicate_requests_skipped += 1
                        logger.debug(
                            f"[EvaluationQueue] Skipping duplicate: {model_path} "
                            f"({config_key}, {harness_type or 'default'}) - already {existing['status']}"
                        )
                        return None

                    # If running or pending, update priority if higher
                    if priority > existing["priority"]:
                        conn.execute(
                            """
                            UPDATE evaluation_requests
                            SET priority = ?, source = ?
                            WHERE request_id = ?
                            """,
                            (priority, source, existing["request_id"]),
                        )
                        conn.commit()
                        logger.info(
                            f"[EvaluationQueue] Updated priority: {model_path} "
                            f"({config_key}) {existing['priority']} -> {priority}"
                        )
                        return existing["request_id"]
                    else:
                        self._stats.duplicate_requests_skipped += 1
                        return existing["request_id"]

                # Insert new request
                conn.execute(
                    """
                    INSERT INTO evaluation_requests (
                        request_id, model_path, board_type, num_players,
                        config_key, status, priority, created_at,
                        max_attempts, source, harness_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request_id,
                        model_path,
                        board_type,
                        num_players,
                        config_key,
                        RequestStatus.PENDING,
                        priority,
                        time.time(),
                        self.max_attempts,
                        source,
                        harness_type,
                    ),
                )
                conn.commit()

                harness_info = f" harness={harness_type}" if harness_type else ""
                logger.info(
                    f"[EvaluationQueue] Added request: {model_path} "
                    f"({config_key}) priority={priority} source={source}{harness_info}"
                )
                return request_id

    def claim_next(self) -> EvaluationRequest | None:
        """Claim the highest-priority pending request.

        Atomically marks the request as RUNNING and returns it.

        Returns:
            EvaluationRequest if one is available, None otherwise
        """
        with self._lock:
            with self._get_connection() as conn:
                # Get highest priority pending request
                row = conn.execute(
                    """
                    SELECT * FROM evaluation_requests
                    WHERE status = ?
                    ORDER BY priority DESC, created_at ASC
                    LIMIT 1
                    """,
                    (RequestStatus.PENDING,),
                ).fetchone()

                if not row:
                    return None

                request_id = row["request_id"]

                # Mark as running
                conn.execute(
                    """
                    UPDATE evaluation_requests
                    SET status = ?, started_at = ?, attempts = attempts + 1
                    WHERE request_id = ?
                    """,
                    (RequestStatus.RUNNING, time.time(), request_id),
                )
                conn.commit()

                # Fetch updated row
                row = conn.execute(
                    "SELECT * FROM evaluation_requests WHERE request_id = ?",
                    (request_id,),
                ).fetchone()

                return self._row_to_request(row)

    def complete(self, request_id: str, elo: float) -> None:
        """Mark a request as completed with Elo result.

        Args:
            request_id: ID of the request
            elo: Resulting Elo rating
        """
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    UPDATE evaluation_requests
                    SET status = ?, completed_at = ?, result_elo = ?, error = ''
                    WHERE request_id = ?
                    """,
                    (RequestStatus.COMPLETED, time.time(), elo, request_id),
                )
                conn.commit()

                logger.info(
                    f"[EvaluationQueue] Completed: {request_id} -> Elo {elo:.0f}"
                )

    def fail(self, request_id: str, error: str) -> None:
        """Mark a request as failed.

        If attempts < max_attempts, the request is reset to PENDING for retry.
        Otherwise, it's marked as FAILED permanently.

        Args:
            request_id: ID of the request
            error: Error message
        """
        with self._lock:
            with self._get_connection() as conn:
                # Get current attempt count
                row = conn.execute(
                    "SELECT attempts, max_attempts, model_path FROM evaluation_requests WHERE request_id = ?",
                    (request_id,),
                ).fetchone()

                if not row:
                    logger.warning(f"[EvaluationQueue] Unknown request: {request_id}")
                    return

                if row["attempts"] < row["max_attempts"]:
                    # Reset to pending for retry
                    conn.execute(
                        """
                        UPDATE evaluation_requests
                        SET status = ?, started_at = 0.0, error = ?
                        WHERE request_id = ?
                        """,
                        (RequestStatus.PENDING, error, request_id),
                    )
                    logger.info(
                        f"[EvaluationQueue] Retry queued ({row['attempts']}/{row['max_attempts']}): "
                        f"{row['model_path']} - {error}"
                    )
                else:
                    # Mark as permanently failed
                    conn.execute(
                        """
                        UPDATE evaluation_requests
                        SET status = ?, completed_at = ?, error = ?
                        WHERE request_id = ?
                        """,
                        (RequestStatus.FAILED, time.time(), error, request_id),
                    )
                    logger.warning(
                        f"[EvaluationQueue] Permanently failed: {row['model_path']} - {error}"
                    )

                conn.commit()

    def get_stuck_evaluations(self, timeout_seconds: float | None = None) -> list[EvaluationRequest]:
        """Find RUNNING requests that have exceeded their timeout.

        Args:
            timeout_seconds: Optional override for stuck timeout.
                If not provided, uses board-specific timeouts.

        Returns:
            List of stuck evaluation requests
        """
        stuck = []
        cutoff_time = time.time()

        with self._lock:
            with self._get_connection() as conn:
                # Get all running requests
                rows = conn.execute(
                    """
                    SELECT * FROM evaluation_requests
                    WHERE status = ?
                    ORDER BY started_at ASC
                    """,
                    (RequestStatus.RUNNING,),
                ).fetchall()

                for row in rows:
                    # Determine timeout for this board type
                    if timeout_seconds is not None:
                        timeout = timeout_seconds
                    else:
                        timeout = STUCK_TIMEOUT_SECONDS.get(
                            row["board_type"], DEFAULT_STUCK_TIMEOUT
                        )

                    # Check if stuck
                    if cutoff_time - row["started_at"] > timeout:
                        stuck.append(self._row_to_request(row))

        return stuck

    def reset_stuck(self, request_id: str) -> None:
        """Reset a stuck request to PENDING for retry.

        Args:
            request_id: ID of the stuck request
        """
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    UPDATE evaluation_requests
                    SET status = ?, started_at = 0.0, error = 'Recovered from stuck'
                    WHERE request_id = ?
                    """,
                    (RequestStatus.PENDING, request_id),
                )
                conn.commit()

                self._stats.stuck_recoveries += 1
                logger.info(f"[EvaluationQueue] Reset stuck request: {request_id}")

    def get_request(self, request_id: str) -> EvaluationRequest | None:
        """Get a specific request by ID.

        Args:
            request_id: ID of the request

        Returns:
            EvaluationRequest if found, None otherwise
        """
        with self._lock:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM evaluation_requests WHERE request_id = ?",
                    (request_id,),
                ).fetchone()
                return self._row_to_request(row) if row else None

    def get_pending_count(self) -> int:
        """Get count of pending requests."""
        with self._lock:
            with self._get_connection() as conn:
                result = conn.execute(
                    "SELECT COUNT(*) FROM evaluation_requests WHERE status = ?",
                    (RequestStatus.PENDING,),
                ).fetchone()
                return result[0] if result else 0

    def get_running_count(self) -> int:
        """Get count of running requests."""
        with self._lock:
            with self._get_connection() as conn:
                result = conn.execute(
                    "SELECT COUNT(*) FROM evaluation_requests WHERE status = ?",
                    (RequestStatus.RUNNING,),
                ).fetchone()
                return result[0] if result else 0

    def get_queue_status(self) -> dict[str, Any]:
        """Get overall queue status.

        Returns:
            Dictionary with queue statistics
        """
        with self._lock:
            with self._get_connection() as conn:
                pending = conn.execute(
                    "SELECT COUNT(*) FROM evaluation_requests WHERE status = ?",
                    (RequestStatus.PENDING,),
                ).fetchone()[0]

                running = conn.execute(
                    "SELECT COUNT(*) FROM evaluation_requests WHERE status = ?",
                    (RequestStatus.RUNNING,),
                ).fetchone()[0]

                completed = conn.execute(
                    "SELECT COUNT(*) FROM evaluation_requests WHERE status = ?",
                    (RequestStatus.COMPLETED,),
                ).fetchone()[0]

                failed = conn.execute(
                    "SELECT COUNT(*) FROM evaluation_requests WHERE status = ?",
                    (RequestStatus.FAILED,),
                ).fetchone()[0]

                return {
                    "pending": pending,
                    "running": running,
                    "completed": completed,
                    "failed": failed,
                    "stuck_recoveries": self._stats.stuck_recoveries,
                    "duplicates_skipped": self._stats.duplicate_requests_skipped,
                }

    def get_models_for_config(
        self, board_type: str, num_players: int, status: str | None = None
    ) -> list[EvaluationRequest]:
        """Get all requests for a specific config.

        Args:
            board_type: Board type
            num_players: Number of players
            status: Optional filter by status

        Returns:
            List of matching requests
        """
        config_key = make_config_key(board_type, num_players)

        with self._lock:
            with self._get_connection() as conn:
                if status:
                    rows = conn.execute(
                        """
                        SELECT * FROM evaluation_requests
                        WHERE config_key = ? AND status = ?
                        ORDER BY priority DESC, created_at ASC
                        """,
                        (config_key, status),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT * FROM evaluation_requests
                        WHERE config_key = ?
                        ORDER BY priority DESC, created_at ASC
                        """,
                        (config_key,),
                    ).fetchall()

                return [self._row_to_request(row) for row in rows]

    def health_check(self) -> HealthCheckResult:
        """Check queue health.

        Returns:
            HealthCheckResult with queue status
        """
        try:
            status = self.get_queue_status()
            stuck = self.get_stuck_evaluations()

            # Queue is degraded if there are stuck evaluations
            if stuck:
                return HealthCheckResult(
                    healthy=True,  # Still functional
                    status="degraded",
                    message=f"{len(stuck)} stuck evaluations detected",
                    details={
                        **status,
                        "stuck_count": len(stuck),
                        "stuck_request_ids": [r.request_id for r in stuck[:5]],
                    },
                )

            return HealthCheckResult(
                healthy=True,
                status="healthy",
                message=f"{status['pending']} pending, {status['running']} running",
                details=status,
            )

        except Exception as e:
            logger.error(f"[EvaluationQueue] Health check failed: {e}")
            return HealthCheckResult(
                healthy=False,
                status="unhealthy",
                message=str(e),
                details={},
            )

    def _row_to_request(self, row: sqlite3.Row) -> EvaluationRequest:
        """Convert a database row to EvaluationRequest."""
        # Handle old rows that may not have harness_type column
        try:
            harness_type = row["harness_type"] if "harness_type" in row.keys() else ""
        except (KeyError, IndexError):
            harness_type = ""

        return EvaluationRequest(
            request_id=row["request_id"],
            model_path=row["model_path"],
            board_type=row["board_type"],
            num_players=row["num_players"],
            config_key=row["config_key"],
            status=row["status"],
            priority=row["priority"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            attempts=row["attempts"],
            max_attempts=row["max_attempts"],
            error=row["error"],
            result_elo=row["result_elo"],
            source=row["source"],
            harness_type=harness_type,
        )


# Singleton instance
_queue_instance: PersistentEvaluationQueue | None = None
_queue_lock = threading.Lock()


def get_evaluation_queue(
    db_path: Path | str | None = None,
) -> PersistentEvaluationQueue:
    """Get the singleton PersistentEvaluationQueue instance.

    Args:
        db_path: Optional database path (only used on first call)

    Returns:
        The singleton queue instance
    """
    global _queue_instance

    with _queue_lock:
        if _queue_instance is None:
            _queue_instance = PersistentEvaluationQueue(
                db_path=db_path or DEFAULT_QUEUE_PATH
            )
        return _queue_instance


def reset_evaluation_queue() -> None:
    """Reset the singleton (for testing)."""
    global _queue_instance
    with _queue_lock:
        _queue_instance = None


__all__ = [
    "DEFAULT_QUEUE_PATH",
    "EvaluationRequest",
    "PersistentEvaluationQueue",
    "QueueStats",
    "RequestStatus",
    "STUCK_TIMEOUT_SECONDS",
    "get_evaluation_queue",
    "reset_evaluation_queue",
]
