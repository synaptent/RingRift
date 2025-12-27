"""Unit tests for duration_scheduler.py (December 2025).

Tests the duration-aware task scheduler with historical learning.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest


class TestTaskDurationRecord:
    """Test TaskDurationRecord dataclass."""

    def test_duration_seconds(self):
        """Test duration_seconds property."""
        from app.coordination.duration_scheduler import TaskDurationRecord

        record = TaskDurationRecord(
            task_type="training",
            config="hex8_2p",
            host="gpu-1",
            started_at=1000.0,
            completed_at=2000.0,
            success=True,
        )

        assert record.duration_seconds == 1000.0

    def test_duration_hours(self):
        """Test duration_hours property."""
        from app.coordination.duration_scheduler import TaskDurationRecord

        record = TaskDurationRecord(
            task_type="training",
            config="hex8_2p",
            host="gpu-1",
            started_at=0.0,
            completed_at=7200.0,  # 2 hours
            success=True,
        )

        assert record.duration_hours == 2.0


class TestScheduledTask:
    """Test ScheduledTask dataclass."""

    def test_expected_duration_seconds(self):
        """Test expected_duration_seconds property."""
        from app.coordination.duration_scheduler import ScheduledTask

        task = ScheduledTask(
            task_id="task-1",
            task_type="training",
            host="gpu-1",
            scheduled_start=1000.0,
            expected_end=4600.0,  # 1 hour later
        )

        assert task.expected_duration_seconds == 3600.0


class TestDurationSchedulerInit:
    """Test DurationScheduler initialization."""

    def test_init_creates_db(self):
        """Test scheduler creates database file."""
        from app.coordination.duration_scheduler import DurationScheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_scheduler.db"
            scheduler = DurationScheduler(db_path=db_path)

            assert db_path.exists()
            scheduler.close()

    def test_init_creates_tables(self):
        """Test scheduler creates required tables."""
        from app.coordination.duration_scheduler import DurationScheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_scheduler.db"
            scheduler = DurationScheduler(db_path=db_path)

            conn = scheduler._get_connection()
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row[0] for row in cursor.fetchall()}

            assert "duration_history" in tables
            assert "running_tasks" in tables
            assert "scheduled_tasks" in tables

            scheduler.close()


class TestDurationEstimation:
    """Test duration estimation methods."""

    @pytest.fixture
    def scheduler(self):
        """Create a fresh scheduler for each test."""
        from app.coordination.duration_scheduler import DurationScheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_scheduler.db"
            sched = DurationScheduler(db_path=db_path)
            yield sched
            sched.close()

    def test_estimate_without_history(self, scheduler):
        """Test estimation without historical data uses defaults."""
        from app.coordination.duration_scheduler import DEFAULT_DURATIONS

        duration = scheduler.estimate_duration("training", use_history=False)
        assert duration == DEFAULT_DURATIONS["training"]

    def test_estimate_unknown_type_fallback(self, scheduler):
        """Test estimation for unknown type uses default."""
        duration = scheduler.estimate_duration("unknown_task_type", use_history=False)
        assert duration == 3600  # Default fallback

    def test_estimate_with_history(self, scheduler):
        """Test estimation uses historical data when available."""
        # Record several completions
        now = time.time()
        for i in range(5):
            scheduler.record_completion(
                task_type="training",
                host="gpu-1",
                started_at=now - 7200 - i,
                completed_at=now - i,  # 2 hours each
                success=True,
                config="hex8_2p",
            )

        # Estimate should use historical data (with 20% buffer)
        duration = scheduler.estimate_duration("training", config="hex8_2p")
        # Historical average is 7200 seconds, with 20% buffer = 8640
        assert 8000 < duration < 9000


class TestRunningTasks:
    """Test running task registration and completion."""

    @pytest.fixture
    def scheduler(self):
        """Create a fresh scheduler for each test."""
        from app.coordination.duration_scheduler import DurationScheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_scheduler.db"
            sched = DurationScheduler(db_path=db_path)
            yield sched
            sched.close()

    def test_register_running(self, scheduler):
        """Test registering a running task."""
        scheduler.register_running(
            task_id="task-1",
            task_type="training",
            host="gpu-1",
            expected_duration=3600.0,
        )

        # Verify it's in the database
        conn = scheduler._get_connection()
        cursor = conn.execute(
            "SELECT * FROM running_tasks WHERE task_id = ?", ("task-1",)
        )
        row = cursor.fetchone()

        assert row is not None
        assert row["task_type"] == "training"
        assert row["host"] == "gpu-1"

    def test_unregister_running(self, scheduler):
        """Test unregistering a running task."""
        scheduler.register_running(
            task_id="task-2",
            task_type="selfplay",
            host="gpu-2",
            expected_duration=1800.0,
        )

        # Unregister the task
        result = scheduler.unregister_running(task_id="task-2")

        assert result is True

        # Verify it's removed from running_tasks
        conn = scheduler._get_connection()
        cursor = conn.execute(
            "SELECT * FROM running_tasks WHERE task_id = ?", ("task-2",)
        )
        row = cursor.fetchone()
        assert row is None

    def test_unregister_nonexistent_returns_false(self, scheduler):
        """Test unregistering a non-existent task returns False."""
        result = scheduler.unregister_running(task_id="nonexistent")
        assert result is False


class TestHostAvailability:
    """Test host availability checking."""

    @pytest.fixture
    def scheduler(self):
        """Create a fresh scheduler for each test."""
        from app.coordination.duration_scheduler import DurationScheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_scheduler.db"
            sched = DurationScheduler(db_path=db_path)
            yield sched
            sched.close()

    def test_available_when_no_tasks(self, scheduler):
        """Test host is available when no tasks running."""
        available, at = scheduler.get_host_availability("gpu-1")

        assert available is True
        assert at <= time.time()

    def test_unavailable_when_intensive_task_running(self, scheduler):
        """Test host is unavailable for intensive tasks when intensive task is running."""
        now = time.time()
        # Register an intensive task (training)
        scheduler.register_running(
            task_id="task-1",
            task_type="training",  # intensive task
            host="gpu-1",
            expected_duration=7200.0,  # 2 hours
        )

        # Check availability for another intensive task
        available, at = scheduler.get_host_availability("gpu-1", task_type="training")

        assert available is False
        assert at > now

    def test_available_for_non_intensive_when_intensive_running(self, scheduler):
        """Test host is available for non-intensive tasks even when intensive task is running."""
        scheduler.register_running(
            task_id="task-1",
            task_type="training",  # intensive task
            host="gpu-1",
            expected_duration=7200.0,
        )

        # Check availability for a non-intensive task type
        available, at = scheduler.get_host_availability("gpu-1", task_type="sync")

        assert available is True


class TestPeakHoursConstants:
    """Test peak hours constants."""

    def test_peak_hours_defined(self):
        """Test peak hours constants are defined."""
        from app.coordination.duration_scheduler import PEAK_HOURS_START, PEAK_HOURS_END

        assert isinstance(PEAK_HOURS_START, int)
        assert isinstance(PEAK_HOURS_END, int)
        assert 0 <= PEAK_HOURS_START <= 23
        assert 0 <= PEAK_HOURS_END <= 23


class TestScheduleNow:
    """Test can_schedule_now method."""

    @pytest.fixture
    def scheduler(self):
        """Create a fresh scheduler for each test."""
        from app.coordination.duration_scheduler import DurationScheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_scheduler.db"
            sched = DurationScheduler(db_path=db_path)
            yield sched
            sched.close()

    def test_can_schedule_when_host_free(self, scheduler):
        """Test scheduling possible when host is free."""
        can_schedule, reason = scheduler.can_schedule_now(
            task_type="selfplay",
            host="gpu-1",
        )

        assert can_schedule is True
        assert reason == "OK"

    def test_cannot_schedule_when_host_busy(self, scheduler):
        """Test scheduling blocked when host is busy."""
        scheduler.register_running(
            task_id="task-1",
            task_type="training",
            host="gpu-1",
            expected_duration=7200.0,
        )

        can_schedule, reason = scheduler.can_schedule_now(
            task_type="selfplay",
            host="gpu-1",
        )

        assert can_schedule is False
        assert "busy" in reason.lower() or "unavailable" in reason.lower()


class TestScheduleTask:
    """Test task scheduling."""

    @pytest.fixture
    def scheduler(self):
        """Create a fresh scheduler for each test."""
        from app.coordination.duration_scheduler import DurationScheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_scheduler.db"
            sched = DurationScheduler(db_path=db_path)
            yield sched
            sched.close()

    def test_schedule_task(self, scheduler):
        """Test scheduling a future task."""
        now = time.time()
        scheduled = scheduler.schedule_task(
            task_id="task-future",
            task_type="training",
            host="gpu-1",
            start_time=now + 3600,  # 1 hour from now
        )

        assert scheduled is True

        # Verify it's in scheduled_tasks
        conn = scheduler._get_connection()
        cursor = conn.execute(
            "SELECT * FROM scheduled_tasks WHERE task_id = ?", ("task-future",)
        )
        row = cursor.fetchone()
        assert row is not None


class TestDurationStats:
    """Test duration statistics."""

    @pytest.fixture
    def scheduler(self):
        """Create a fresh scheduler for each test."""
        from app.coordination.duration_scheduler import DurationScheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_scheduler.db"
            sched = DurationScheduler(db_path=db_path)
            yield sched
            sched.close()

    def test_get_duration_stats_empty(self, scheduler):
        """Test stats with no history."""
        stats = scheduler.get_duration_stats()
        assert isinstance(stats, dict)
        assert len(stats) == 0

    def test_get_duration_stats_with_data(self, scheduler):
        """Test stats with historical data."""
        now = time.time()

        # Add some history
        for i in range(3):
            scheduler.record_completion(
                task_type="training",
                host="gpu-1",
                started_at=now - 7200 - i,
                completed_at=now - i,
                success=True,
            )

        stats = scheduler.get_duration_stats()

        assert "training" in stats
        assert stats["training"]["count"] == 3
        assert stats["training"]["avg_hours"] is not None


class TestCleanup:
    """Test cleanup functionality."""

    @pytest.fixture
    def scheduler(self):
        """Create a fresh scheduler for each test."""
        from app.coordination.duration_scheduler import DurationScheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_scheduler.db"
            sched = DurationScheduler(db_path=db_path)
            yield sched
            sched.close()

    def test_cleanup_old_records(self, scheduler):
        """Test cleaning up old records."""
        now = time.time()

        # Add old and new records
        scheduler.record_completion(
            task_type="training",
            host="gpu-1",
            started_at=now - (100 * 86400),  # 100 days ago
            completed_at=now - (100 * 86400) + 3600,
            success=True,
        )
        scheduler.record_completion(
            task_type="training",
            host="gpu-1",
            started_at=now - 3600,
            completed_at=now,
            success=True,
        )

        # Cleanup records older than 90 days
        cleaned = scheduler.cleanup_old_records(max_age_days=90)

        assert cleaned == 1


class TestSingletonPattern:
    """Test global singleton pattern."""

    def test_get_scheduler_returns_singleton(self):
        """Test get_scheduler returns same instance."""
        from app.coordination.duration_scheduler import get_scheduler, reset_scheduler

        reset_scheduler()

        s1 = get_scheduler()
        s2 = get_scheduler()

        assert s1 is s2

        reset_scheduler()

    def test_reset_scheduler(self):
        """Test reset_scheduler creates new instance."""
        from app.coordination.duration_scheduler import get_scheduler, reset_scheduler

        reset_scheduler()

        s1 = get_scheduler()
        reset_scheduler()
        s2 = get_scheduler()

        assert s1 is not s2

        reset_scheduler()


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_estimate_task_duration(self):
        """Test estimate_task_duration function."""
        from app.coordination.duration_scheduler import (
            estimate_task_duration,
            reset_scheduler,
        )

        reset_scheduler()
        duration = estimate_task_duration("training")
        assert isinstance(duration, (int, float))
        assert duration > 0
        reset_scheduler()

    def test_can_schedule_task(self):
        """Test can_schedule_task function."""
        from app.coordination.duration_scheduler import can_schedule_task, reset_scheduler

        reset_scheduler()
        can_sched, reason = can_schedule_task("selfplay", "test-host")
        assert isinstance(can_sched, bool)
        assert isinstance(reason, str)
        reset_scheduler()


class TestDefaultDurations:
    """Test default duration constants."""

    def test_default_durations_exist(self):
        """Test DEFAULT_DURATIONS has expected entries."""
        from app.coordination.duration_scheduler import DEFAULT_DURATIONS

        expected_types = ["selfplay", "training", "evaluation", "sync"]
        for task_type in expected_types:
            assert task_type in DEFAULT_DURATIONS
            assert isinstance(DEFAULT_DURATIONS[task_type], (int, float))
            assert DEFAULT_DURATIONS[task_type] > 0

    def test_intensive_task_types_defined(self):
        """Test INTENSIVE_TASK_TYPES is defined."""
        from app.coordination.duration_scheduler import INTENSIVE_TASK_TYPES

        assert isinstance(INTENSIVE_TASK_TYPES, set)
        assert "training" in INTENSIVE_TASK_TYPES
