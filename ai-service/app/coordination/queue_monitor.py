#!/usr/bin/env python3
"""Queue Depth Monitoring with Backpressure.

This module monitors various queue depths (training data, pending games,
evaluation queue, etc.) and applies backpressure when queues get too deep.

Features:
- Multi-queue monitoring (training data, games, evaluations, etc.)
- Configurable thresholds per queue
- Backpressure signals to upstream producers
- Historical tracking for capacity planning
- Integration with cross-process events

Usage:
    from app.coordination.queue_monitor import (
        get_queue_monitor,
        check_backpressure,
        report_queue_depth,
        should_throttle_production,
    )

    # Check if we should apply backpressure
    if check_backpressure("training_data"):
        slow_down_selfplay()

    # Report queue depth from a producer
    report_queue_depth("training_data", depth=50000, host="gh200-a")

    # Consumer: check if should throttle
    if should_throttle_production("training_data", current_depth=50000):
        reduce_production_rate()
"""

from __future__ import annotations

import json
import socket
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

# Default database location
DEFAULT_MONITOR_DB = Path("/tmp/ringrift_coordination/queue_monitor.db")

# Import centralized timeout thresholds
try:
    from app.config.thresholds import SQLITE_BUSY_TIMEOUT_MS, SQLITE_TIMEOUT
except ImportError:
    SQLITE_BUSY_TIMEOUT_MS = 10000
    SQLITE_TIMEOUT = 30

# Import centralized defaults (December 2025)
try:
    from app.config.coordination_defaults import QueueDefaults
    _TRAINING_DATA_SOFT = QueueDefaults.TRAINING_DATA_SOFT_LIMIT
    _TRAINING_DATA_HARD = QueueDefaults.TRAINING_DATA_HARD_LIMIT
    _TRAINING_DATA_TARGET = QueueDefaults.TRAINING_DATA_TARGET
    _PENDING_GAMES_SOFT = QueueDefaults.PENDING_GAMES_SOFT_LIMIT
    _PENDING_GAMES_HARD = QueueDefaults.PENDING_GAMES_HARD_LIMIT
    _PENDING_GAMES_TARGET = QueueDefaults.PENDING_GAMES_TARGET
    _EVALUATION_SOFT = QueueDefaults.EVALUATION_SOFT_LIMIT
    _EVALUATION_HARD = QueueDefaults.EVALUATION_HARD_LIMIT
    _EVALUATION_TARGET = QueueDefaults.EVALUATION_TARGET
    _SYNC_SOFT = QueueDefaults.SYNC_SOFT_LIMIT
    _SYNC_HARD = QueueDefaults.SYNC_HARD_LIMIT
    _SYNC_TARGET = QueueDefaults.SYNC_TARGET
except ImportError:
    # Fallback defaults
    _TRAINING_DATA_SOFT = 100000
    _TRAINING_DATA_HARD = 500000
    _TRAINING_DATA_TARGET = 50000
    _PENDING_GAMES_SOFT = 1000
    _PENDING_GAMES_HARD = 5000
    _PENDING_GAMES_TARGET = 500
    _EVALUATION_SOFT = 50
    _EVALUATION_HARD = 200
    _EVALUATION_TARGET = 20
    _SYNC_SOFT = 100
    _SYNC_HARD = 500
    _SYNC_TARGET = 50


# Queue types
class QueueType(Enum):
    """Types of queues to monitor."""
    TRAINING_DATA = "training_data"      # Raw game data waiting for training
    PENDING_GAMES = "pending_games"      # Games being played
    EVALUATION_QUEUE = "evaluation"      # Models waiting for evaluation
    PROMOTION_QUEUE = "promotion"        # Models waiting for promotion
    SYNC_QUEUE = "sync"                  # Data waiting to sync
    EXPORT_QUEUE = "export"              # Models waiting to export


# Default thresholds and settings per queue (uses centralized defaults)
DEFAULT_QUEUE_CONFIG = {
    QueueType.TRAINING_DATA: {
        "soft_limit": _TRAINING_DATA_SOFT,
        "hard_limit": _TRAINING_DATA_HARD,
        "target_depth": _TRAINING_DATA_TARGET,
        "drain_rate": 10000,       # Expected consumption per hour
    },
    QueueType.PENDING_GAMES: {
        "soft_limit": _PENDING_GAMES_SOFT,
        "hard_limit": _PENDING_GAMES_HARD,
        "target_depth": _PENDING_GAMES_TARGET,
        "drain_rate": 500,
    },
    QueueType.EVALUATION_QUEUE: {
        "soft_limit": _EVALUATION_SOFT,
        "hard_limit": _EVALUATION_HARD,
        "target_depth": _EVALUATION_TARGET,
        "drain_rate": 20,
    },
    QueueType.PROMOTION_QUEUE: {
        "soft_limit": 10,
        "hard_limit": 50,
        "target_depth": 5,
        "drain_rate": 5,
    },
    QueueType.SYNC_QUEUE: {
        "soft_limit": _SYNC_SOFT,
        "hard_limit": _SYNC_HARD,
        "target_depth": _SYNC_TARGET,
        "drain_rate": 100,
    },
    QueueType.EXPORT_QUEUE: {
        "soft_limit": 20,
        "hard_limit": 100,
        "target_depth": 10,
        "drain_rate": 10,
    },
}


class BackpressureLevel(Enum):
    """Level of backpressure to apply."""
    NONE = "none"           # No backpressure, operate normally
    SOFT = "soft"           # Soft throttle - reduce production by 50%
    HARD = "hard"           # Hard throttle - reduce production by 90%
    STOP = "stop"           # Stop production entirely


@dataclass
class QueueStatus:
    """Current status of a queue."""

    queue_type: QueueType
    current_depth: int
    soft_limit: int
    hard_limit: int
    target_depth: int
    backpressure: BackpressureLevel
    last_updated: float
    trend: str  # "rising", "stable", "falling"

    def to_dict(self) -> dict[str, Any]:
        return {
            "queue_type": self.queue_type.value,
            "current_depth": self.current_depth,
            "soft_limit": self.soft_limit,
            "hard_limit": self.hard_limit,
            "target_depth": self.target_depth,
            "backpressure": self.backpressure.value,
            "last_updated": datetime.fromtimestamp(self.last_updated).isoformat(),
            "trend": self.trend,
        }


@dataclass
class QueueMetric:
    """A single queue depth measurement."""

    queue_type: str
    depth: int
    host: str
    timestamp: float


class QueueMonitor:
    """Monitor queue depths and apply backpressure."""

    def __init__(
        self,
        db_path: Path | None = None,
        config: dict[QueueType, dict[str, int]] | None = None,
    ):
        self.db_path = db_path or DEFAULT_MONITOR_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.config = config or DEFAULT_QUEUE_CONFIG
        self._local = threading.local()
        self._backpressure_callbacks: dict[QueueType, list[Callable[[BackpressureLevel], None]]] = {}
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path), timeout=float(SQLITE_TIMEOUT))
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute('PRAGMA journal_mode=WAL')
            self._local.conn.execute(f'PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}')
            self._local.conn.execute('PRAGMA synchronous=NORMAL')
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        conn.executescript('''
            -- Queue depth measurements
            CREATE TABLE IF NOT EXISTS queue_metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                queue_type TEXT NOT NULL,
                depth INTEGER NOT NULL,
                host TEXT NOT NULL DEFAULT '',
                timestamp REAL NOT NULL
            );

            -- Current queue status (latest per queue)
            CREATE TABLE IF NOT EXISTS queue_status (
                queue_type TEXT PRIMARY KEY,
                current_depth INTEGER NOT NULL,
                backpressure TEXT NOT NULL DEFAULT 'none',
                last_updated REAL NOT NULL,
                trend TEXT DEFAULT 'stable'
            );

            -- Backpressure events
            CREATE TABLE IF NOT EXISTS backpressure_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                queue_type TEXT NOT NULL,
                old_level TEXT NOT NULL,
                new_level TEXT NOT NULL,
                depth INTEGER NOT NULL,
                timestamp REAL NOT NULL
            );

            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_metrics_queue ON queue_metrics(queue_type);
            CREATE INDEX IF NOT EXISTS idx_metrics_time ON queue_metrics(timestamp);
        ''')
        conn.commit()

    def report_depth(
        self,
        queue_type: QueueType,
        depth: int,
        host: str = "",
    ) -> BackpressureLevel:
        """Report current queue depth and get backpressure level.

        Args:
            queue_type: Type of queue
            depth: Current queue depth
            host: Host reporting the depth

        Returns:
            Current backpressure level for this queue
        """
        conn = self._get_connection()
        now = time.time()
        host = host or socket.gethostname()

        # Record metric
        conn.execute(
            'INSERT INTO queue_metrics (queue_type, depth, host, timestamp) VALUES (?, ?, ?, ?)',
            (queue_type.value, depth, host, now)
        )

        # Calculate trend (compare to 5 minutes ago)
        cursor = conn.execute(
            '''SELECT AVG(depth) as avg_depth FROM queue_metrics
               WHERE queue_type = ? AND timestamp > ? AND timestamp < ?''',
            (queue_type.value, now - 600, now - 300)
        )
        old_row = cursor.fetchone()
        old_avg = old_row["avg_depth"] if old_row and old_row["avg_depth"] else depth

        if depth > old_avg * 1.1:
            trend = "rising"
        elif depth < old_avg * 0.9:
            trend = "falling"
        else:
            trend = "stable"

        # Calculate backpressure level
        config = self.config.get(queue_type, {})
        soft_limit = config.get("soft_limit", 10000)
        hard_limit = config.get("hard_limit", 50000)

        if depth >= hard_limit:
            backpressure = BackpressureLevel.STOP
        elif depth >= hard_limit * 0.9:
            backpressure = BackpressureLevel.HARD
        elif depth >= soft_limit:
            backpressure = BackpressureLevel.SOFT
        else:
            backpressure = BackpressureLevel.NONE

        # Update status
        cursor = conn.execute(
            'SELECT backpressure FROM queue_status WHERE queue_type = ?',
            (queue_type.value,)
        )
        old_status = cursor.fetchone()
        old_level = BackpressureLevel(old_status["backpressure"]) if old_status else BackpressureLevel.NONE

        conn.execute(
            '''INSERT OR REPLACE INTO queue_status
               (queue_type, current_depth, backpressure, last_updated, trend)
               VALUES (?, ?, ?, ?, ?)''',
            (queue_type.value, depth, backpressure.value, now, trend)
        )

        # Record backpressure change event
        if backpressure != old_level:
            conn.execute(
                '''INSERT INTO backpressure_events
                   (queue_type, old_level, new_level, depth, timestamp)
                   VALUES (?, ?, ?, ?, ?)''',
                (queue_type.value, old_level.value, backpressure.value, depth, now)
            )

            # Notify callbacks
            if queue_type in self._backpressure_callbacks:
                for callback in self._backpressure_callbacks[queue_type]:
                    try:
                        callback(backpressure)
                    except Exception as e:
                        print(f"[QueueMonitor] Callback error: {e}")

        conn.commit()
        return backpressure

    def get_status(self, queue_type: QueueType) -> QueueStatus | None:
        """Get current status of a queue."""
        conn = self._get_connection()
        cursor = conn.execute(
            'SELECT * FROM queue_status WHERE queue_type = ?',
            (queue_type.value,)
        )
        row = cursor.fetchone()
        if not row:
            return None

        config = self.config.get(queue_type, {})
        return QueueStatus(
            queue_type=queue_type,
            current_depth=row["current_depth"],
            soft_limit=config.get("soft_limit", 10000),
            hard_limit=config.get("hard_limit", 50000),
            target_depth=config.get("target_depth", 5000),
            backpressure=BackpressureLevel(row["backpressure"]),
            last_updated=row["last_updated"],
            trend=row["trend"],
        )

    def get_all_status(self) -> dict[str, QueueStatus]:
        """Get status of all monitored queues."""
        return {
            qt.value: status
            for qt in QueueType
            if (status := self.get_status(qt)) is not None
        }

    def check_backpressure(self, queue_type: QueueType) -> BackpressureLevel:
        """Check current backpressure level for a queue."""
        status = self.get_status(queue_type)
        return status.backpressure if status else BackpressureLevel.NONE

    def should_throttle(self, queue_type: QueueType) -> bool:
        """Check if production should be throttled for this queue."""
        level = self.check_backpressure(queue_type)
        return level in (BackpressureLevel.SOFT, BackpressureLevel.HARD, BackpressureLevel.STOP)

    def should_stop(self, queue_type: QueueType) -> bool:
        """Check if production should stop for this queue."""
        level = self.check_backpressure(queue_type)
        return level == BackpressureLevel.STOP

    def get_throttle_factor(self, queue_type: QueueType) -> float:
        """Get production throttle factor (1.0 = full speed, 0 = stopped)."""
        level = self.check_backpressure(queue_type)
        return {
            BackpressureLevel.NONE: 1.0,
            BackpressureLevel.SOFT: 0.5,
            BackpressureLevel.HARD: 0.1,
            BackpressureLevel.STOP: 0.0,
        }[level]

    def register_callback(
        self,
        queue_type: QueueType,
        callback: Callable[[BackpressureLevel], None],
    ) -> None:
        """Register a callback for backpressure changes."""
        if queue_type not in self._backpressure_callbacks:
            self._backpressure_callbacks[queue_type] = []
        self._backpressure_callbacks[queue_type].append(callback)

    def get_history(
        self,
        queue_type: QueueType,
        hours: int = 24,
        resolution_minutes: int = 5,
    ) -> list[dict[str, Any]]:
        """Get queue depth history for graphing."""
        conn = self._get_connection()
        cutoff = time.time() - (hours * 3600)
        resolution_seconds = resolution_minutes * 60

        cursor = conn.execute(
            '''SELECT
                 CAST(timestamp / ? AS INTEGER) * ? as bucket,
                 AVG(depth) as avg_depth,
                 MAX(depth) as max_depth,
                 MIN(depth) as min_depth
               FROM queue_metrics
               WHERE queue_type = ? AND timestamp > ?
               GROUP BY bucket
               ORDER BY bucket''',
            (resolution_seconds, resolution_seconds, queue_type.value, cutoff)
        )

        return [
            {
                "timestamp": datetime.fromtimestamp(row["bucket"]).isoformat(),
                "avg_depth": round(row["avg_depth"]),
                "max_depth": row["max_depth"],
                "min_depth": row["min_depth"],
            }
            for row in cursor.fetchall()
        ]

    def get_backpressure_events(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get recent backpressure events."""
        conn = self._get_connection()
        cutoff = time.time() - (hours * 3600)

        cursor = conn.execute(
            '''SELECT queue_type, old_level, new_level, depth, timestamp
               FROM backpressure_events
               WHERE timestamp > ?
               ORDER BY timestamp DESC''',
            (cutoff,)
        )

        return [
            {
                "queue_type": row["queue_type"],
                "old_level": row["old_level"],
                "new_level": row["new_level"],
                "depth": row["depth"],
                "timestamp": datetime.fromtimestamp(row["timestamp"]).isoformat(),
            }
            for row in cursor.fetchall()
        ]

    def cleanup(self, max_age_hours: int = 168) -> int:
        """Clean up old metrics (default: 7 days)."""
        conn = self._get_connection()
        cutoff = time.time() - (max_age_hours * 3600)
        cursor = conn.execute('DELETE FROM queue_metrics WHERE timestamp < ?', (cutoff,))
        conn.commit()
        return cursor.rowcount

    def get_stats(self) -> dict[str, Any]:
        """Get monitoring statistics."""
        all_status = self.get_all_status()
        events = self.get_backpressure_events(hours=24)

        return {
            "queues": {k: v.to_dict() for k, v in all_status.items()},
            "recent_events": len(events),
            "throttled_queues": [
                k for k, v in all_status.items()
                if v.backpressure != BackpressureLevel.NONE
            ],
        }

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# Global singleton
_monitor: QueueMonitor | None = None
_monitor_lock = threading.RLock()


def get_queue_monitor(db_path: Path | None = None) -> QueueMonitor:
    """Get the global queue monitor singleton."""
    global _monitor
    with _monitor_lock:
        if _monitor is None:
            _monitor = QueueMonitor(db_path)
        return _monitor


def reset_queue_monitor() -> None:
    """Reset the global monitor (for testing)."""
    global _monitor
    with _monitor_lock:
        if _monitor is not None:
            _monitor.close()
        _monitor = None


# Convenience functions


def report_queue_depth(
    queue_type: QueueType,
    depth: int,
    host: str = "",
) -> BackpressureLevel:
    """Report queue depth and get backpressure level."""
    return get_queue_monitor().report_depth(queue_type, depth, host)


def check_backpressure(queue_type: QueueType) -> BackpressureLevel:
    """Check backpressure level for a queue."""
    return get_queue_monitor().check_backpressure(queue_type)


def should_throttle_production(queue_type: QueueType) -> bool:
    """Check if production should be throttled."""
    return get_queue_monitor().should_throttle(queue_type)


def should_stop_production(queue_type: QueueType) -> bool:
    """Check if production should stop."""
    return get_queue_monitor().should_stop(queue_type)


def get_throttle_factor(queue_type: QueueType) -> float:
    """Get production throttle factor."""
    return get_queue_monitor().get_throttle_factor(queue_type)


def get_queue_stats() -> dict[str, Any]:
    """Get queue monitoring statistics."""
    return get_queue_monitor().get_stats()


# Command-line interface

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Queue depth monitoring")
    parser.add_argument("--status", action="store_true", help="Show all queue status")
    parser.add_argument("--history", type=str, help="Show history for queue type")
    parser.add_argument("--events", action="store_true", help="Show backpressure events")
    parser.add_argument("--report", nargs=2, help="Report depth: <queue_type> <depth>")
    parser.add_argument("--cleanup", type=int, help="Cleanup metrics older than N hours")
    args = parser.parse_args()

    monitor = get_queue_monitor()

    if args.status:
        print(json.dumps(monitor.get_stats(), indent=2))

    elif args.history:
        try:
            qt = QueueType(args.history)
            history = monitor.get_history(qt)
            print(json.dumps(history, indent=2))
        except ValueError:
            print(f"Unknown queue type: {args.history}")
            print(f"Valid types: {[t.value for t in QueueType]}")

    elif args.events:
        events = monitor.get_backpressure_events()
        print(json.dumps(events, indent=2))

    elif args.report:
        try:
            qt = QueueType(args.report[0])
            depth = int(args.report[1])
            level = monitor.report_depth(qt, depth)
            print(f"Reported {qt.value}: {depth}, backpressure level: {level.value}")
        except ValueError as e:
            print(f"Error: {e}")

    elif args.cleanup:
        count = monitor.cleanup(args.cleanup)
        print(f"Cleaned up {count} old metrics")

    else:
        parser.print_help()


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Constants
    "DEFAULT_QUEUE_CONFIG",
    "BackpressureLevel",
    "QueueMetric",
    # Main class
    "QueueMonitor",
    # Data classes
    "QueueStatus",
    # Enums
    "QueueType",
    "check_backpressure",
    # Functions
    "get_queue_monitor",
    "get_queue_stats",
    "get_throttle_factor",
    "report_queue_depth",
    "reset_queue_monitor",
    "should_stop_production",
    "should_throttle_production",
]
