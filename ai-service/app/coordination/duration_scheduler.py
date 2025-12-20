#!/usr/bin/env python3
"""Duration-Aware Task Scheduler.

This module provides intelligent scheduling based on expected task durations.
It helps coordinate long-running tasks and predict resource availability.

Features:
- Expected duration tracking for different task types
- Historical duration learning from completed tasks
- Resource availability prediction
- Time-window based scheduling for long tasks
- Soft and hard deadlines

Usage:
    from app.coordination.duration_scheduler import (
        get_scheduler,
        estimate_task_duration,
        schedule_task,
        get_resource_availability,
    )

    # Estimate when a task will complete
    est_duration = estimate_task_duration("training", config="square8_2p")

    # Check when resources will be available
    available_at = get_resource_availability("gpu-server-1", task_type="training")

    # Schedule a task with duration awareness
    if schedule_task("training", "gpu-server-1", min_duration_hours=2):
        launch_training()
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Default database location
DEFAULT_SCHEDULER_DB = Path("/tmp/ringrift_coordination/duration_scheduler.db")

# Import centralized defaults (December 2025)
try:
    from app.config.coordination_defaults import DurationDefaults
    _SELFPLAY = DurationDefaults.SELFPLAY_DURATION
    _GPU_SELFPLAY = DurationDefaults.GPU_SELFPLAY_DURATION
    _TRAINING = DurationDefaults.TRAINING_DURATION
    _CMAES = DurationDefaults.CMAES_DURATION
    _TOURNAMENT = DurationDefaults.TOURNAMENT_DURATION
    _EVALUATION = DurationDefaults.EVALUATION_DURATION
    _SYNC = DurationDefaults.SYNC_DURATION
    _EXPORT = DurationDefaults.EXPORT_DURATION
    _PIPELINE = DurationDefaults.PIPELINE_DURATION
    _IMPROVEMENT = DurationDefaults.IMPROVEMENT_LOOP_DURATION
    PEAK_HOURS_START = DurationDefaults.PEAK_HOURS_START
    PEAK_HOURS_END = DurationDefaults.PEAK_HOURS_END
except ImportError:
    # Fallback defaults
    _SELFPLAY = 3600
    _GPU_SELFPLAY = 7200
    _TRAINING = 14400
    _CMAES = 28800
    _TOURNAMENT = 1800
    _EVALUATION = 3600
    _SYNC = 600
    _EXPORT = 300
    _PIPELINE = 21600
    _IMPROVEMENT = 43200
    PEAK_HOURS_START = 14
    PEAK_HOURS_END = 22

# Default expected durations in seconds (uses centralized defaults)
DEFAULT_DURATIONS = {
    "selfplay": _SELFPLAY,
    "gpu_selfplay": _GPU_SELFPLAY,
    "training": _TRAINING,
    "cmaes": _CMAES,
    "tournament": _TOURNAMENT,
    "evaluation": _EVALUATION,
    "sync": _SYNC,
    "export": _EXPORT,
    "pipeline": _PIPELINE,
    "improvement_loop": _IMPROVEMENT,
}

# Task types that should avoid peak hours (intensive tasks)
INTENSIVE_TASK_TYPES = {"training", "cmaes", "pipeline", "improvement_loop"}


@dataclass
class TaskDurationRecord:
    """Record of a completed task's duration."""

    task_type: str
    config: str
    host: str
    started_at: float
    completed_at: float
    success: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        return self.completed_at - self.started_at

    @property
    def duration_hours(self) -> float:
        return self.duration_seconds / 3600


@dataclass
class ScheduledTask:
    """A task scheduled for execution."""

    task_id: str
    task_type: str
    host: str
    scheduled_start: float
    expected_end: float
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def expected_duration_seconds(self) -> float:
        return self.expected_end - self.scheduled_start


class DurationScheduler:
    """Duration-aware task scheduler with historical learning."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DEFAULT_SCHEDULER_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute('PRAGMA journal_mode=WAL')
            self._local.conn.execute('PRAGMA busy_timeout=10000')
            self._local.conn.execute('PRAGMA synchronous=NORMAL')
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        conn.executescript('''
            -- Duration history for learning
            CREATE TABLE IF NOT EXISTS duration_history (
                record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL,
                config TEXT NOT NULL DEFAULT '',
                host TEXT NOT NULL,
                started_at REAL NOT NULL,
                completed_at REAL NOT NULL,
                duration_seconds REAL NOT NULL,
                success INTEGER NOT NULL DEFAULT 1,
                metadata TEXT DEFAULT '{}'
            );

            -- Currently running tasks (for availability prediction)
            CREATE TABLE IF NOT EXISTS running_tasks (
                task_id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                host TEXT NOT NULL,
                started_at REAL NOT NULL,
                expected_end REAL NOT NULL,
                pid INTEGER,
                metadata TEXT DEFAULT '{}'
            );

            -- Scheduled future tasks
            CREATE TABLE IF NOT EXISTS scheduled_tasks (
                schedule_id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT UNIQUE NOT NULL,
                task_type TEXT NOT NULL,
                host TEXT NOT NULL,
                scheduled_start REAL NOT NULL,
                expected_end REAL NOT NULL,
                priority INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                metadata TEXT DEFAULT '{}'
            );

            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_duration_type ON duration_history(task_type);
            CREATE INDEX IF NOT EXISTS idx_duration_host ON duration_history(host);
            CREATE INDEX IF NOT EXISTS idx_running_host ON running_tasks(host);
            CREATE INDEX IF NOT EXISTS idx_scheduled_start ON scheduled_tasks(scheduled_start);
        ''')
        conn.commit()

    def estimate_duration(
        self,
        task_type: str,
        config: str = "",
        host: str = "",
        use_history: bool = True,
    ) -> float:
        """Estimate expected duration for a task in seconds.

        Args:
            task_type: Type of task
            config: Configuration string (e.g., "square8_2p")
            host: Target host (for host-specific estimates)
            use_history: Whether to use historical data

        Returns:
            Expected duration in seconds
        """
        if not use_history:
            return DEFAULT_DURATIONS.get(task_type, 3600)

        conn = self._get_connection()

        # Try to find historical data (most specific to least)
        queries = [
            # Exact match
            ('''SELECT AVG(duration_seconds) as avg_dur, COUNT(*) as cnt
                FROM duration_history
                WHERE task_type = ? AND config = ? AND host = ? AND success = 1
                AND started_at > ?''',
             (task_type, config, host, time.time() - 7 * 86400)),

            # Same type and config
            ('''SELECT AVG(duration_seconds) as avg_dur, COUNT(*) as cnt
                FROM duration_history
                WHERE task_type = ? AND config = ? AND success = 1
                AND started_at > ?''',
             (task_type, config, time.time() - 7 * 86400)),

            # Same type only
            ('''SELECT AVG(duration_seconds) as avg_dur, COUNT(*) as cnt
                FROM duration_history
                WHERE task_type = ? AND success = 1
                AND started_at > ?''',
             (task_type, time.time() - 30 * 86400)),
        ]

        for query, params in queries:
            cursor = conn.execute(query, params)
            row = cursor.fetchone()
            if row and row["cnt"] >= 3 and row["avg_dur"]:
                # Add 20% buffer for safety
                return row["avg_dur"] * 1.2

        # Fall back to default
        return DEFAULT_DURATIONS.get(task_type, 3600)

    def record_completion(
        self,
        task_type: str,
        host: str,
        started_at: float,
        completed_at: float,
        success: bool = True,
        config: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a task completion for duration learning.

        Args:
            task_type: Type of task
            host: Host where task ran
            started_at: Start timestamp
            completed_at: Completion timestamp
            success: Whether task succeeded
            config: Configuration string
            metadata: Additional metadata
        """
        conn = self._get_connection()
        duration = completed_at - started_at

        conn.execute(
            '''INSERT INTO duration_history
               (task_type, config, host, started_at, completed_at, duration_seconds, success, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (task_type, config, host, started_at, completed_at, duration,
             1 if success else 0, json.dumps(metadata or {}))
        )
        conn.commit()

        # Also remove from running tasks
        conn.execute('DELETE FROM running_tasks WHERE task_type = ? AND host = ? AND started_at = ?',
                     (task_type, host, started_at))
        conn.commit()

    def register_running(
        self,
        task_id: str,
        task_type: str,
        host: str,
        expected_duration: float | None = None,
        pid: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a task as currently running.

        Args:
            task_id: Unique task identifier
            task_type: Type of task
            host: Host where task is running
            expected_duration: Expected duration in seconds (auto-estimated if None)
            pid: Process ID
            metadata: Additional metadata
        """
        conn = self._get_connection()
        now = time.time()

        if expected_duration is None:
            expected_duration = self.estimate_duration(task_type, host=host)

        # Sanity check: cap expected_duration at 24 hours max
        # This guards against callers accidentally passing PIDs or other large values
        MAX_DURATION_SECONDS = 24 * 3600  # 24 hours
        if expected_duration > MAX_DURATION_SECONDS:
            expected_duration = self.estimate_duration(task_type, host=host)

        expected_end = now + expected_duration

        conn.execute(
            '''INSERT OR REPLACE INTO running_tasks
               (task_id, task_type, host, started_at, expected_end, pid, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (task_id, task_type, host, now, expected_end, pid, json.dumps(metadata or {}))
        )
        conn.commit()

    def unregister_running(self, task_id: str) -> bool:
        """Unregister a running task.

        Returns:
            True if task was found and removed
        """
        conn = self._get_connection()
        cursor = conn.execute('DELETE FROM running_tasks WHERE task_id = ?', (task_id,))
        conn.commit()
        return cursor.rowcount > 0

    def get_host_availability(self, host: str, task_type: str = "") -> tuple[bool, float]:
        """Check when a host will be available for a new task.

        Args:
            host: Host to check
            task_type: Optional task type (some tasks can run concurrently)

        Returns:
            Tuple of (is_available_now, available_at_timestamp)
        """
        conn = self._get_connection()
        now = time.time()

        # Check running tasks on this host
        cursor = conn.execute(
            '''SELECT task_type, expected_end FROM running_tasks
               WHERE host = ? AND expected_end > ?
               ORDER BY expected_end ASC''',
            (host, now)
        )

        running = cursor.fetchall()

        if not running:
            return True, now

        # Check for intensive tasks running
        intensive_running = [r for r in running if r["task_type"] in INTENSIVE_TASK_TYPES]

        # Intensive tasks block other intensive tasks
        if intensive_running and task_type in INTENSIVE_TASK_TYPES:
            latest_end = max(r["expected_end"] for r in intensive_running)
            return False, latest_end

        # Non-intensive tasks don't block anything
        # Intensive tasks can start even when non-intensive tasks are running
        return True, now

    def can_schedule_now(
        self,
        task_type: str,
        host: str,
        min_duration_hours: float = 0,
        avoid_peak_hours: bool = True,
    ) -> tuple[bool, str]:
        """Check if a task can be scheduled to start now.

        Args:
            task_type: Type of task
            host: Target host
            min_duration_hours: Minimum expected duration
            avoid_peak_hours: Whether to avoid scheduling intensive tasks during peak hours

        Returns:
            Tuple of (can_schedule, reason)
        """
        now = time.time()
        now_dt = datetime.utcnow()

        # Check host availability
        is_available, available_at = self.get_host_availability(host, task_type)
        if not is_available:
            wait_minutes = (available_at - now) / 60
            return False, f"Host busy until {datetime.fromtimestamp(available_at).strftime('%H:%M')}, ~{wait_minutes:.0f}m"

        # Check peak hours for intensive tasks
        if avoid_peak_hours and task_type in INTENSIVE_TASK_TYPES:
            hour = now_dt.hour
            if PEAK_HOURS_START <= hour < PEAK_HOURS_END:
                expected_duration = self.estimate_duration(task_type)
                end_hour = (hour + int(expected_duration / 3600)) % 24

                if end_hour >= PEAK_HOURS_START and end_hour < PEAK_HOURS_END:
                    return False, f"Peak hours ({PEAK_HOURS_START}:00-{PEAK_HOURS_END}:00 UTC) - intensive task would complete during peak"

        # Check if task fits in remaining day (for long tasks)
        if min_duration_hours > 0:
            expected_duration = max(min_duration_hours * 3600, self.estimate_duration(task_type))
            expected_end = now + expected_duration
            end_hour = datetime.fromtimestamp(expected_end).hour

            # Don't start long tasks that would complete in the middle of the night
            if (expected_duration > 6 * 3600  # > 6 hours
                    and end_hour >= 2 and end_hour <= 6):  # Would end 2-6 AM
                return False, f"Task would complete at {end_hour}:00 (monitoring gap)"

        return True, "OK"

    def schedule_task(
        self,
        task_id: str,
        task_type: str,
        host: str,
        scheduled_start: float | None = None,
        priority: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Schedule a task for future execution.

        Args:
            task_id: Unique task identifier
            task_type: Type of task
            host: Target host
            scheduled_start: Start time (default: now)
            priority: Priority (higher = more important)
            metadata: Additional metadata

        Returns:
            True if scheduled successfully
        """
        conn = self._get_connection()
        now = time.time()
        start = scheduled_start or now
        expected_duration = self.estimate_duration(task_type, host=host)
        expected_end = start + expected_duration

        try:
            conn.execute(
                '''INSERT OR REPLACE INTO scheduled_tasks
                   (task_id, task_type, host, scheduled_start, expected_end, priority, created_at, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                (task_id, task_type, host, start, expected_end, priority, now, json.dumps(metadata or {}))
            )
            conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"[DurationScheduler] Error scheduling task: {e}")
            return False

    def get_scheduled_tasks(
        self,
        host: str | None = None,
        task_type: str | None = None,
        limit: int = 50,
    ) -> list[ScheduledTask]:
        """Get scheduled tasks.

        Args:
            host: Filter by host
            task_type: Filter by task type
            limit: Maximum tasks to return

        Returns:
            List of scheduled tasks
        """
        conn = self._get_connection()

        query = '''SELECT task_id, task_type, host, scheduled_start, expected_end, priority, metadata
                   FROM scheduled_tasks WHERE 1=1'''
        params: list[Any] = []

        if host:
            query += ' AND host = ?'
            params.append(host)
        if task_type:
            query += ' AND task_type = ?'
            params.append(task_type)

        query += ' ORDER BY priority DESC, scheduled_start ASC LIMIT ?'
        params.append(limit)

        cursor = conn.execute(query, params)
        return [
            ScheduledTask(
                task_id=row["task_id"],
                task_type=row["task_type"],
                host=row["host"],
                scheduled_start=row["scheduled_start"],
                expected_end=row["expected_end"],
                priority=row["priority"],
                metadata=json.loads(row["metadata"]),
            )
            for row in cursor.fetchall()
        ]

    def cancel_scheduled(self, task_id: str) -> bool:
        """Cancel a scheduled task."""
        conn = self._get_connection()
        cursor = conn.execute('DELETE FROM scheduled_tasks WHERE task_id = ?', (task_id,))
        conn.commit()
        return cursor.rowcount > 0

    def get_duration_stats(self, task_type: str | None = None) -> dict[str, Any]:
        """Get duration statistics."""
        conn = self._get_connection()

        if task_type:
            cursor = conn.execute(
                '''SELECT
                     task_type,
                     COUNT(*) as count,
                     AVG(duration_seconds) as avg_seconds,
                     MIN(duration_seconds) as min_seconds,
                     MAX(duration_seconds) as max_seconds,
                     SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count
                   FROM duration_history
                   WHERE task_type = ? AND started_at > ?
                   GROUP BY task_type''',
                (task_type, time.time() - 30 * 86400)
            )
        else:
            cursor = conn.execute(
                '''SELECT
                     task_type,
                     COUNT(*) as count,
                     AVG(duration_seconds) as avg_seconds,
                     MIN(duration_seconds) as min_seconds,
                     MAX(duration_seconds) as max_seconds,
                     SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count
                   FROM duration_history
                   WHERE started_at > ?
                   GROUP BY task_type''',
                (time.time() - 30 * 86400,)
            )

        stats = {}
        for row in cursor.fetchall():
            stats[row["task_type"]] = {
                "count": row["count"],
                "avg_hours": round(row["avg_seconds"] / 3600, 2) if row["avg_seconds"] else None,
                "min_hours": round(row["min_seconds"] / 3600, 2) if row["min_seconds"] else None,
                "max_hours": round(row["max_seconds"] / 3600, 2) if row["max_seconds"] else None,
                "success_rate": round(row["success_count"] / row["count"], 2) if row["count"] else 0,
            }

        return stats

    def cleanup_old_records(self, max_age_days: int = 90) -> int:
        """Clean up old duration records."""
        conn = self._get_connection()
        cutoff = time.time() - (max_age_days * 86400)
        cursor = conn.execute('DELETE FROM duration_history WHERE started_at < ?', (cutoff,))
        conn.commit()
        return cursor.rowcount

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# Global singleton
_scheduler: DurationScheduler | None = None
_scheduler_lock = threading.RLock()


def get_scheduler(db_path: Path | None = None) -> DurationScheduler:
    """Get the global duration scheduler singleton."""
    global _scheduler
    with _scheduler_lock:
        if _scheduler is None:
            _scheduler = DurationScheduler(db_path)
        return _scheduler


def reset_scheduler() -> None:
    """Reset the global scheduler (for testing)."""
    global _scheduler
    with _scheduler_lock:
        if _scheduler is not None:
            _scheduler.close()
        _scheduler = None


# Convenience functions


def estimate_task_duration(task_type: str, config: str = "", host: str = "") -> float:
    """Estimate duration for a task in seconds."""
    return get_scheduler().estimate_duration(task_type, config, host)


def record_task_completion(
    task_type: str,
    host: str,
    started_at: float,
    completed_at: float,
    success: bool = True,
    config: str = "",
) -> None:
    """Record a task completion."""
    get_scheduler().record_completion(task_type, host, started_at, completed_at, success, config)


def register_running_task(
    task_id: str,
    task_type: str,
    host: str,
    expected_duration: float | None = None,
) -> None:
    """Register a task as running."""
    get_scheduler().register_running(task_id, task_type, host, expected_duration)


def get_resource_availability(host: str, task_type: str = "") -> tuple[bool, float]:
    """Check resource availability on a host."""
    return get_scheduler().get_host_availability(host, task_type)


def can_schedule_task(
    task_type: str,
    host: str,
    min_duration_hours: float = 0,
) -> tuple[bool, str]:
    """Check if a task can be scheduled now."""
    return get_scheduler().can_schedule_now(task_type, host, min_duration_hours)


# Command-line interface

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Duration scheduler management")
    parser.add_argument("--stats", action="store_true", help="Show duration statistics")
    parser.add_argument("--estimate", type=str, help="Estimate duration for task type")
    parser.add_argument("--availability", type=str, help="Check availability for host")
    parser.add_argument("--cleanup", type=int, help="Cleanup records older than N days")
    args = parser.parse_args()

    scheduler = get_scheduler()

    if args.stats:
        print(json.dumps(scheduler.get_duration_stats(), indent=2))

    elif args.estimate:
        duration = scheduler.estimate_duration(args.estimate)
        hours = duration / 3600
        print(f"Estimated duration for {args.estimate}: {hours:.1f} hours ({duration:.0f} seconds)")

    elif args.availability:
        available, at = scheduler.get_host_availability(args.availability)
        if available:
            print(f"{args.availability}: Available now")
        else:
            print(f"{args.availability}: Available at {datetime.fromtimestamp(at).strftime('%Y-%m-%d %H:%M')}")

    elif args.cleanup:
        count = scheduler.cleanup_old_records(args.cleanup)
        print(f"Cleaned up {count} records")

    else:
        parser.print_help()


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Constants
    "DEFAULT_DURATIONS",
    "INTENSIVE_TASK_TYPES",
    "PEAK_HOURS_END",
    "PEAK_HOURS_START",
    # Main class
    "DurationScheduler",
    "ScheduledTask",
    # Data classes
    "TaskDurationRecord",
    "can_schedule_task",
    "estimate_task_duration",
    "get_resource_availability",
    # Functions
    "get_scheduler",
    "record_task_completion",
    "register_running_task",
    "reset_scheduler",
]
