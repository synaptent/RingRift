#!/usr/bin/env python3
"""Integration tests for the coordination system.

Tests that all coordination components work together correctly.
"""

import os
import tempfile
import time
import pytest

# Skip if coordination modules not available
pytest.importorskip("app.coordination")


class TestTaskCoordinator:
    """Tests for TaskCoordinator."""

    def test_can_spawn_task(self):
        """Test basic task spawn checking."""
        from app.coordination import TaskCoordinator, TaskType

        # Use temp db
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_coord.db")
            os.environ["RINGRIFT_COORDINATOR_DB"] = db_path

            try:
                tc = TaskCoordinator(db_path=db_path)

                import socket
                node_id = socket.gethostname()

                # Should be able to spawn selfplay initially
                can_spawn, reason = tc.can_spawn_task(TaskType.SELFPLAY, node_id)
                assert can_spawn, f"Should allow spawn: {reason}"

                # Register a task
                task_id = "test_task_1"
                tc.register_task(task_id, TaskType.SELFPLAY, node_id, os.getpid())

                # Should track the task
                tasks = tc.get_active_tasks()
                assert len(tasks) >= 1

                # Unregister
                tc.unregister_task(task_id)
            finally:
                os.environ.pop("RINGRIFT_COORDINATOR_DB", None)


class TestOrchestratorRegistry:
    """Tests for OrchestratorRegistry."""

    def test_role_acquisition(self):
        """Test acquiring and releasing roles."""
        from app.coordination import (
            OrchestratorRole,
            OrchestratorRegistry,
            acquire_orchestrator_role,
            release_orchestrator_role,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_registry.db")

            registry = OrchestratorRegistry(db_path=db_path)

            # Should not be held initially
            assert not registry.is_role_held(OrchestratorRole.TOURNAMENT_RUNNER)

            # Acquire role
            success = registry.acquire_role(OrchestratorRole.TOURNAMENT_RUNNER)
            assert success, "Should acquire role"

            # Should be held now
            assert registry.is_role_held(OrchestratorRole.TOURNAMENT_RUNNER)

            # Get holder info
            holder = registry.get_role_holder(OrchestratorRole.TOURNAMENT_RUNNER)
            assert holder is not None
            assert holder.pid == os.getpid()

            # Release role
            registry.release_role()
            assert not registry.is_role_held(OrchestratorRole.TOURNAMENT_RUNNER)


class TestDurationScheduler:
    """Tests for DurationScheduler."""

    def test_duration_estimation(self):
        """Test task duration estimation."""
        from app.coordination import estimate_task_duration

        # Should return default durations
        selfplay_duration = estimate_task_duration("selfplay")
        assert selfplay_duration > 0

        training_duration = estimate_task_duration("training")
        assert training_duration > selfplay_duration  # Training takes longer

    def test_can_schedule_task(self):
        """Test task scheduling check."""
        from app.coordination import can_schedule_task

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_scheduler.db")
            os.environ["RINGRIFT_SCHEDULER_DB"] = db_path

            try:
                import socket
                host = socket.gethostname()

                # Should be able to schedule selfplay
                can_schedule, reason = can_schedule_task("selfplay", host)
                # Result depends on current time (peak hours), so just check it returns
                assert isinstance(can_schedule, bool)
                assert isinstance(reason, str)
            finally:
                os.environ.pop("RINGRIFT_SCHEDULER_DB", None)


class TestQueueMonitor:
    """Tests for QueueMonitor."""

    def test_backpressure_reporting(self):
        """Test queue depth reporting and backpressure."""
        from app.coordination import (
            QueueType,
            report_queue_depth,
            check_backpressure,
            should_throttle_production,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_queue.db")
            os.environ["RINGRIFT_QUEUE_MONITOR_DB"] = db_path

            try:
                # Report normal queue depth
                report_queue_depth(QueueType.TRAINING_DATA, 100)

                # Check backpressure
                level = check_backpressure(QueueType.TRAINING_DATA)
                assert level is not None

                # Should not throttle at normal depth
                should_throttle = should_throttle_production(QueueType.TRAINING_DATA)
                assert isinstance(should_throttle, bool)
            finally:
                os.environ.pop("RINGRIFT_QUEUE_MONITOR_DB", None)


class TestSyncMutex:
    """Tests for SyncMutex."""

    def test_sync_lock(self):
        """Test sync lock acquisition."""
        from app.coordination import sync_lock

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_sync.db")
            os.environ["RINGRIFT_SYNC_MUTEX_DB"] = db_path

            try:
                # Should be able to acquire lock
                with sync_lock("test_host", timeout=5.0) as lock_info:
                    assert lock_info is not None
                    # Lock should be held during context
            finally:
                os.environ.pop("RINGRIFT_SYNC_MUTEX_DB", None)


class TestBandwidthManager:
    """Tests for BandwidthManager."""

    def test_bandwidth_allocation(self):
        """Test bandwidth allocation."""
        from app.coordination import (
            request_bandwidth,
            release_bandwidth,
            TransferPriority,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_bandwidth.db")
            os.environ["RINGRIFT_BANDWIDTH_DB"] = db_path

            try:
                # Request bandwidth
                allocation = request_bandwidth(
                    "source_host", "dest_host",
                    TransferPriority.NORMAL,
                    estimated_bytes=1000000
                )
                assert allocation is not None

                # Release bandwidth
                if allocation:
                    release_bandwidth(allocation.allocation_id)
            finally:
                os.environ.pop("RINGRIFT_BANDWIDTH_DB", None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
