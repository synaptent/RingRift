"""Metrics Manager for P2P Orchestrator.

Extracted from p2p_orchestrator.py on December 26, 2025.

This module provides:
- Buffered metric recording to SQLite
- Metrics history retrieval
- Metrics summary aggregation

Usage as standalone:
    manager = MetricsManager(db_path)
    manager.record_metric("gpu_utilization", 85.5, board_type="hex8")
    history = manager.get_metrics_history("gpu_utilization", hours=24)
    summary = manager.get_metrics_summary(hours=24)

Usage as mixin (in P2POrchestrator):
    class P2POrchestrator(MetricsManagerMixin, ...):
        pass
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MetricsManager:
    """Standalone metrics manager for recording and querying metrics.

    Features:
    - Buffered writes for performance (batches every 30s or 100 entries)
    - Thread-safe buffer management
    - Historical queries by metric type, board type, player count
    - Summary aggregation with min/max/avg/latest values
    """

    def __init__(
        self,
        db_path: Path | str,
        flush_interval: float = 30.0,
        max_buffer: int = 100,
    ):
        """Initialize metrics manager.

        Args:
            db_path: Path to SQLite database
            flush_interval: Seconds between automatic flushes
            max_buffer: Max entries before automatic flush
        """
        self.db_path = Path(db_path)
        self._metrics_buffer: list[tuple] = []
        self._metrics_buffer_lock = threading.Lock()
        self._metrics_last_flush: float = time.time()  # Initialize to now to avoid immediate flush
        self._metrics_flush_interval = flush_interval
        self._metrics_max_buffer = max_buffer

        # Ensure metrics table exists
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create metrics_history table if it doesn't exist."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    metric_type TEXT NOT NULL,
                    board_type TEXT,
                    num_players INTEGER,
                    value REAL NOT NULL,
                    metadata TEXT,
                    UNIQUE(timestamp, metric_type, board_type, num_players)
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_type_time
                ON metrics_history(metric_type, timestamp DESC)
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Could not ensure metrics table: {e}")

    def record_metric(
        self,
        metric_type: str,
        value: float,
        board_type: str | None = None,
        num_players: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a metric to the history table for observability.

        Metric types:
        - training_loss: NNUE training loss
        - elo_rating: Model Elo rating
        - gpu_utilization: GPU utilization percentage
        - selfplay_games_per_hour: Game generation rate
        - validation_rate: GPU selfplay validation rate
        - tournament_win_rate: Tournament win rate for new model

        Uses buffered writes for better performance (batches every 30s or 100 entries).
        """
        entry = (
            time.time(),
            metric_type,
            board_type,
            num_players,
            value,
            json.dumps(metadata) if metadata else None,
        )

        with self._metrics_buffer_lock:
            self._metrics_buffer.append(entry)
            should_flush = (
                len(self._metrics_buffer) >= self._metrics_max_buffer
                or time.time() - self._metrics_last_flush > self._metrics_flush_interval
            )

        if should_flush:
            self.flush()

    def flush(self) -> int:
        """Flush buffered metrics to database using batch insert.

        Returns:
            Number of entries flushed
        """
        with self._metrics_buffer_lock:
            if not self._metrics_buffer:
                return 0
            entries = self._metrics_buffer.copy()
            self._metrics_buffer.clear()
            self._metrics_last_flush = time.time()

        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT OR IGNORE INTO metrics_history
                (timestamp, metric_type, board_type, num_players, value, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                entries,
            )
            conn.commit()
            return len(entries)
        except Exception as e:
            logger.error(f"Failed to flush metrics buffer ({len(entries)} entries): {e}")
            return 0
        finally:
            if conn:
                conn.close()

    def get_history(
        self,
        metric_type: str,
        board_type: str | None = None,
        num_players: int | None = None,
        hours: float = 24,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get metrics history for a specific metric type."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            since = time.time() - (hours * 3600)
            query = """
                SELECT timestamp, value, board_type, num_players, metadata
                FROM metrics_history
                WHERE metric_type = ? AND timestamp > ?
            """
            params: list[Any] = [metric_type, since]

            if board_type:
                query += " AND board_type = ?"
                params.append(board_type)
            if num_players:
                query += " AND num_players = ?"
                params.append(num_players)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            results = []
            for row in cursor.fetchall():
                results.append({
                    "timestamp": row[0],
                    "value": row[1],
                    "board_type": row[2],
                    "num_players": row[3],
                    "metadata": json.loads(row[4]) if row[4] else None,
                })
            return results
        except Exception as e:
            logger.error(f"Failed to get metrics history: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def get_summary(self, hours: float = 24) -> dict[str, Any]:
        """Get summary of all metrics over the specified time period."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            since = time.time() - (hours * 3600)

            cursor.execute(
                """
                SELECT metric_type, COUNT(*), AVG(value), MIN(value), MAX(value)
                FROM metrics_history
                WHERE timestamp > ?
                GROUP BY metric_type
            """,
                (since,),
            )

            summary: dict[str, Any] = {}
            for row in cursor.fetchall():
                summary[row[0]] = {
                    "count": row[1],
                    "avg": row[2],
                    "min": row[3],
                    "max": row[4],
                }

            cursor.execute("""
                SELECT metric_type, value, timestamp
                FROM metrics_history m1
                WHERE timestamp = (
                    SELECT MAX(timestamp) FROM metrics_history m2
                    WHERE m2.metric_type = m1.metric_type
                )
            """)
            for row in cursor.fetchall():
                if row[0] in summary:
                    summary[row[0]]["latest"] = row[1]
                    summary[row[0]]["latest_time"] = row[2]

            return {"period_hours": hours, "since": since, "metrics": summary}
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {}
        finally:
            if conn:
                conn.close()

    def get_pending_count(self) -> int:
        """Get count of pending (not yet flushed) metrics."""
        with self._metrics_buffer_lock:
            return len(self._metrics_buffer)

    def health_check(self) -> dict[str, Any]:
        """Return health status for the metrics manager.

        Returns:
            dict with is_healthy, pending_count, last_flush details
        """
        pending = self.get_pending_count()
        time_since_flush = time.time() - self._metrics_last_flush
        # Unhealthy if buffer hasn't flushed in 5x normal interval
        is_healthy = time_since_flush < (self._metrics_flush_interval * 5)
        return {
            "is_healthy": is_healthy,
            "pending_count": pending,
            "last_flush": self._metrics_last_flush,
            "time_since_flush": time_since_flush,
            "max_buffer": self._metrics_max_buffer,
            "db_path": str(self.db_path),
        }


class MetricsManagerMixin:
    """Mixin class for adding metrics functionality to P2POrchestrator.

    This mixin provides backward-compatible method names for P2POrchestrator.
    The orchestrator should initialize self._metrics_manager in __init__.

    Example:
        class P2POrchestrator(MetricsManagerMixin, ...):
            def __init__(self, ...):
                ...
                self._metrics_manager = MetricsManager(self.db_path)
    """

    _metrics_manager: MetricsManager

    def record_metric(
        self,
        metric_type: str,
        value: float,
        board_type: str | None = None,
        num_players: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a metric (delegates to MetricsManager)."""
        self._metrics_manager.record_metric(
            metric_type, value, board_type, num_players, metadata
        )

    def _flush_metrics_buffer(self) -> None:
        """Flush buffered metrics (delegates to MetricsManager)."""
        self._metrics_manager.flush()

    def get_metrics_history(
        self,
        metric_type: str,
        board_type: str | None = None,
        num_players: int | None = None,
        hours: float = 24,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get metrics history (delegates to MetricsManager)."""
        return self._metrics_manager.get_history(
            metric_type, board_type, num_players, hours, limit
        )

    def get_metrics_summary(self, hours: float = 24) -> dict[str, Any]:
        """Get metrics summary (delegates to MetricsManager)."""
        return self._metrics_manager.get_summary(hours)

    def metrics_health_check(self) -> dict[str, Any]:
        """Return health status for metrics subsystem (delegates to MetricsManager)."""
        return self._metrics_manager.health_check()
