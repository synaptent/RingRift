"""Tests for TaskLifecycleCoordinator.

Tests cover:
- Task registration and tracking
- Task state transitions (spawned, completed, failed, cancelled)
- Orphan detection and handling
- Node online/offline handling
- Statistics and history
- Callbacks
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from unittest.mock import patch

import pytest

from app.coordination.task_lifecycle_coordinator import (
    TaskLifecycleCoordinator,
    TaskLifecycleStats,
    TaskStatus,
    TrackedTask,
    get_task_lifecycle_coordinator,
    get_task_stats,
    get_active_task_count,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def coordinator():
    """Create a fresh TaskLifecycleCoordinator for each test."""
    return TaskLifecycleCoordinator(
        heartbeat_threshold_seconds=60.0,
        orphan_check_interval_seconds=1.0,  # Fast for testing
    )


@pytest.fixture
def mock_event():
    """Create a mock event with payload."""
    @dataclass
    class MockEvent:
        payload: dict[str, Any]
    return MockEvent


# =============================================================================
# Initialization Tests
# =============================================================================


class TestTaskLifecycleCoordinatorInit:
    """Test TaskLifecycleCoordinator initialization."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        coord = TaskLifecycleCoordinator()
        assert coord.heartbeat_threshold == 60.0
        assert coord.orphan_check_interval == 30.0
        assert coord.max_history == 1000

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        coord = TaskLifecycleCoordinator(
            heartbeat_threshold_seconds=120.0,
            orphan_check_interval_seconds=60.0,
            max_history=500,
        )
        assert coord.heartbeat_threshold == 120.0
        assert coord.orphan_check_interval == 60.0
        assert coord.max_history == 500

    def test_init_empty_state(self, coordinator):
        """Test initial state is empty."""
        stats = coordinator.get_stats()
        assert stats.active_tasks == 0
        assert stats.orphaned_tasks == 0
        assert stats.total_spawned == 0


class TestTaskLifecycleCoordinatorSubscriptions:
    """Tests for event subscription wiring."""

    def test_subscribe_to_events_uses_router_helper(self, coordinator):
        """Should subscribe task and host events through the unified helper."""
        from app.coordination.event_router import DataEventType

        with patch("app.coordination.event_router.subscribe") as mock_subscribe:
            result = coordinator.subscribe_to_events()

        assert result is True
        assert coordinator._subscribed is True
        subscribed = [(call.args[0], call.args[1]) for call in mock_subscribe.call_args_list]
        assert (DataEventType.TASK_SPAWNED, coordinator._on_task_spawned) in subscribed
        assert (DataEventType.TASK_COMPLETED, coordinator._on_task_completed) in subscribed
        assert (DataEventType.TASK_FAILED, coordinator._on_task_failed) in subscribed
        assert (DataEventType.TASK_HEARTBEAT, coordinator._on_task_heartbeat) in subscribed
        assert (DataEventType.TASK_ORPHANED, coordinator._on_task_orphaned) in subscribed
        assert (DataEventType.TASK_CANCELLED, coordinator._on_task_cancelled) in subscribed
        assert (DataEventType.HOST_ONLINE, coordinator._on_host_online) in subscribed
        assert (DataEventType.HOST_OFFLINE, coordinator._on_host_offline) in subscribed
        assert (DataEventType.NODE_RECOVERED, coordinator._on_node_recovered) in subscribed

    def test_emit_orphan_event_serializes_float_heartbeat(self, coordinator):
        """Should emit orphan events even though heartbeat is stored as a float."""
        task = coordinator.register_task("task-001", "selfplay", "gpu-1")
        task.last_heartbeat = 1_700_000_000.0

        with patch("app.coordination.event_emission_helpers.safe_emit_event", return_value=True) as mock_emit:
            coordinator._emit_orphan_event(task)

        payload = mock_emit.call_args.args[1]
        assert payload["task_id"] == "task-001"
        assert payload["last_heartbeat"] == datetime.fromtimestamp(task.last_heartbeat).isoformat()


# =============================================================================
# Task Registration Tests
# =============================================================================


class TestTaskRegistration:
    """Test task registration functionality."""

    def test_register_task(self, coordinator):
        """Test basic task registration."""
        task = coordinator.register_task(
            task_id="task-001",
            task_type="selfplay",
            node_id="gpu-1",
        )

        assert task.task_id == "task-001"
        assert task.task_type == "selfplay"
        assert task.node_id == "gpu-1"
        assert task.status == TaskStatus.RUNNING

    def test_register_task_increments_count(self, coordinator):
        """Test registration increments spawn count."""
        coordinator.register_task("task-001", "selfplay", "gpu-1")
        coordinator.register_task("task-002", "training", "gpu-2")

        stats = coordinator.get_stats()
        assert stats.total_spawned == 2
        assert stats.active_tasks == 2

    def test_get_task(self, coordinator):
        """Test getting a task by ID."""
        coordinator.register_task("task-001", "selfplay", "gpu-1")

        task = coordinator.get_task("task-001")
        assert task is not None
        assert task.task_id == "task-001"

    def test_get_task_not_found(self, coordinator):
        """Test getting a non-existent task."""
        task = coordinator.get_task("nonexistent")
        assert task is None


# =============================================================================
# Task Spawned Event Tests
# =============================================================================


class TestTaskSpawnedEvent:
    """Test TASK_SPAWNED event handling."""

    @pytest.mark.asyncio
    async def test_task_spawned_event(self, coordinator, mock_event):
        """Test handling TASK_SPAWNED event."""
        event = mock_event(payload={
            "task_id": "task-001",
            "task_type": "selfplay",
            "node_id": "gpu-1",
        })

        await coordinator._on_task_spawned(event)

        task = coordinator.get_task("task-001")
        assert task is not None
        assert task.status == TaskStatus.RUNNING

    @pytest.mark.asyncio
    async def test_task_spawned_increments_count(self, coordinator, mock_event):
        """Test spawned event increments count."""
        event = mock_event(payload={
            "task_id": "task-001",
            "task_type": "selfplay",
            "node_id": "gpu-1",
        })

        await coordinator._on_task_spawned(event)

        stats = coordinator.get_stats()
        assert stats.total_spawned == 1


# =============================================================================
# Task Completed Event Tests
# =============================================================================


class TestTaskCompletedEvent:
    """Test TASK_COMPLETED event handling."""

    @pytest.mark.asyncio
    async def test_task_completed_event(self, coordinator, mock_event):
        """Test handling TASK_COMPLETED event."""
        coordinator.register_task("task-001", "selfplay", "gpu-1")

        event = mock_event(payload={
            "task_id": "task-001",
            "duration": 100.0,
            "result": {"games_played": 10},
        })

        await coordinator._on_task_completed(event)

        task = coordinator.get_task("task-001")
        # Task should be in history, not active
        active = coordinator.get_active_tasks()
        assert len(active) == 0

        stats = coordinator.get_stats()
        assert stats.completed_tasks == 1

    @pytest.mark.asyncio
    async def test_task_completed_unknown_task(self, coordinator, mock_event):
        """Test completing unknown task creates record."""
        event = mock_event(payload={
            "task_id": "unknown-task",
            "task_type": "selfplay",
            "node_id": "gpu-1",
            "duration": 50.0,
        })

        await coordinator._on_task_completed(event)

        stats = coordinator.get_stats()
        assert stats.completed_tasks == 1


# =============================================================================
# Task Failed Event Tests
# =============================================================================


class TestTaskFailedEvent:
    """Test TASK_FAILED event handling."""

    @pytest.mark.asyncio
    async def test_task_failed_event(self, coordinator, mock_event):
        """Test handling TASK_FAILED event."""
        coordinator.register_task("task-001", "selfplay", "gpu-1")

        event = mock_event(payload={
            "task_id": "task-001",
            "error": "GPU out of memory",
        })

        await coordinator._on_task_failed(event)

        stats = coordinator.get_stats()
        assert stats.failed_tasks == 1

    @pytest.mark.asyncio
    async def test_task_failed_callback(self, coordinator, mock_event):
        """Test failure callback is called."""
        callbacks_called = []
        coordinator.on_failure(lambda task: callbacks_called.append(task))

        coordinator.register_task("task-001", "selfplay", "gpu-1")

        event = mock_event(payload={
            "task_id": "task-001",
            "error": "Error",
        })

        await coordinator._on_task_failed(event)

        assert len(callbacks_called) == 1
        assert callbacks_called[0].task_id == "task-001"


# =============================================================================
# Task Cancelled Event Tests
# =============================================================================


class TestTaskCancelledEvent:
    """Test TASK_CANCELLED event handling."""

    @pytest.mark.asyncio
    async def test_task_cancelled_event(self, coordinator, mock_event):
        """Test handling TASK_CANCELLED event."""
        coordinator.register_task("task-001", "selfplay", "gpu-1")

        event = mock_event(payload={
            "task_id": "task-001",
        })

        await coordinator._on_task_cancelled(event)

        stats = coordinator.get_stats()
        assert stats.cancelled_tasks == 1


# =============================================================================
# Heartbeat Tests
# =============================================================================


class TestHeartbeat:
    """Test heartbeat handling."""

    def test_update_heartbeat(self, coordinator):
        """Test updating heartbeat."""
        coordinator.register_task("task-001", "selfplay", "gpu-1")

        result = coordinator.update_heartbeat("task-001")

        assert result is True
        task = coordinator.get_task("task-001")
        assert task.heartbeat_count == 1

    def test_update_heartbeat_not_found(self, coordinator):
        """Test updating heartbeat for non-existent task."""
        result = coordinator.update_heartbeat("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_heartbeat_event(self, coordinator, mock_event):
        """Test handling TASK_HEARTBEAT event."""
        coordinator.register_task("task-001", "selfplay", "gpu-1")

        event = mock_event(payload={"task_id": "task-001"})
        await coordinator._on_task_heartbeat(event)

        task = coordinator.get_task("task-001")
        assert task.heartbeat_count == 1


# =============================================================================
# Orphan Detection Tests
# =============================================================================


class TestOrphanDetection:
    """Test orphan detection functionality."""

    def test_check_for_orphans(self, coordinator):
        """Test checking for orphaned tasks."""
        # Create task with old heartbeat
        task = coordinator.register_task("task-001", "selfplay", "gpu-1")
        task.last_heartbeat = time.time() - 120  # 2 minutes ago

        # Force orphan check
        coordinator._last_orphan_check = 0

        orphaned = coordinator.check_for_orphans()

        assert len(orphaned) == 1
        assert orphaned[0].task_id == "task-001"
        assert orphaned[0].status == TaskStatus.ORPHANED

    def test_orphan_callback(self, coordinator):
        """Test orphan callback is called."""
        callbacks_called = []
        coordinator.on_orphan(lambda task: callbacks_called.append(task))

        task = coordinator.register_task("task-001", "selfplay", "gpu-1")
        task.last_heartbeat = time.time() - 120

        coordinator._last_orphan_check = 0
        coordinator.check_for_orphans()

        assert len(callbacks_called) == 1

    def test_get_orphaned_tasks(self, coordinator):
        """Test getting orphaned tasks."""
        task = coordinator.register_task("task-001", "selfplay", "gpu-1")
        task.last_heartbeat = time.time() - 120

        coordinator._last_orphan_check = 0
        coordinator.check_for_orphans()

        orphaned = coordinator.get_orphaned_tasks()
        assert len(orphaned) == 1

    @pytest.mark.asyncio
    async def test_orphaned_task_recovers_on_heartbeat(self, coordinator, mock_event):
        """Test orphaned task recovers on heartbeat."""
        task = coordinator.register_task("task-001", "selfplay", "gpu-1")
        task.last_heartbeat = time.time() - 120

        coordinator._last_orphan_check = 0
        coordinator.check_for_orphans()

        # Task is now orphaned
        assert len(coordinator.get_orphaned_tasks()) == 1

        # Heartbeat received
        event = mock_event(payload={"task_id": "task-001"})
        await coordinator._on_task_heartbeat(event)

        # Task should be recovered
        assert len(coordinator.get_orphaned_tasks()) == 0
        assert len(coordinator.get_active_tasks()) == 1


# =============================================================================
# Node Online/Offline Tests
# =============================================================================


class TestNodeEvents:
    """Test node online/offline event handling."""

    @pytest.mark.asyncio
    async def test_host_online_event(self, coordinator, mock_event):
        """Test handling HOST_ONLINE event."""
        event = mock_event(payload={
            "node_id": "gpu-1",
        })

        await coordinator._on_host_online(event)

        assert coordinator.is_node_online("gpu-1")

    @pytest.mark.asyncio
    async def test_host_offline_orphans_tasks(self, coordinator, mock_event):
        """Test HOST_OFFLINE marks tasks as orphaned."""
        coordinator.register_task("task-001", "selfplay", "gpu-1")
        coordinator.register_task("task-002", "training", "gpu-1")
        coordinator.register_task("task-003", "selfplay", "gpu-2")

        event = mock_event(payload={"node_id": "gpu-1"})
        await coordinator._on_host_offline(event)

        orphaned = coordinator.get_orphaned_tasks()
        active = coordinator.get_active_tasks()

        assert len(orphaned) == 2
        assert len(active) == 1  # task-003 on gpu-2

    @pytest.mark.asyncio
    async def test_node_recovered_restores_tasks(self, coordinator, mock_event):
        """Test NODE_RECOVERED restores orphaned tasks."""
        coordinator.register_task("task-001", "selfplay", "gpu-1")

        # Node goes offline
        await coordinator._on_host_offline(mock_event(payload={"node_id": "gpu-1"}))

        assert len(coordinator.get_orphaned_tasks()) == 1

        # Node recovers
        await coordinator._on_node_recovered(mock_event(payload={"node_id": "gpu-1"}))

        assert len(coordinator.get_orphaned_tasks()) == 0
        assert len(coordinator.get_active_tasks()) == 1


# =============================================================================
# Query Tests
# =============================================================================


class TestQueries:
    """Test query methods."""

    def test_get_active_tasks(self, coordinator):
        """Test getting active tasks."""
        coordinator.register_task("task-001", "selfplay", "gpu-1")
        coordinator.register_task("task-002", "training", "gpu-2")

        active = coordinator.get_active_tasks()
        assert len(active) == 2

    def test_get_tasks_by_node(self, coordinator):
        """Test getting tasks by node."""
        coordinator.register_task("task-001", "selfplay", "gpu-1")
        coordinator.register_task("task-002", "training", "gpu-1")
        coordinator.register_task("task-003", "selfplay", "gpu-2")

        gpu1_tasks = coordinator.get_tasks_by_node("gpu-1")
        assert len(gpu1_tasks) == 2

    def test_get_tasks_by_type(self, coordinator):
        """Test getting tasks by type."""
        coordinator.register_task("task-001", "selfplay", "gpu-1")
        coordinator.register_task("task-002", "selfplay", "gpu-2")
        coordinator.register_task("task-003", "training", "gpu-1")

        selfplay_tasks = coordinator.get_tasks_by_type("selfplay")
        assert len(selfplay_tasks) == 2


# =============================================================================
# History Tests
# =============================================================================


class TestHistory:
    """Test history tracking."""

    @pytest.mark.asyncio
    async def test_history_records_completed(self, coordinator, mock_event):
        """Test history records completed tasks."""
        coordinator.register_task("task-001", "selfplay", "gpu-1")

        event = mock_event(payload={
            "task_id": "task-001",
            "duration": 100.0,
        })
        await coordinator._on_task_completed(event)

        history = coordinator.get_history()
        assert len(history) == 1
        assert history[0].task_id == "task-001"

    @pytest.mark.asyncio
    async def test_history_limit(self, coordinator, mock_event):
        """Test history is limited."""
        coord = TaskLifecycleCoordinator(max_history=5)

        for i in range(10):
            coord.register_task(f"task-{i}", "selfplay", "gpu-1")
            await coord._on_task_completed(mock_event(payload={
                "task_id": f"task-{i}",
                "duration": 10.0,
            }))

        history = coord.get_history()
        assert len(history) == 5


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Test statistics calculation."""

    @pytest.mark.asyncio
    async def test_stats_by_type(self, coordinator, mock_event):
        """Test statistics count by type."""
        coordinator.register_task("task-001", "selfplay", "gpu-1")
        coordinator.register_task("task-002", "selfplay", "gpu-2")
        coordinator.register_task("task-003", "training", "gpu-1")

        stats = coordinator.get_stats()

        assert stats.by_type.get("selfplay", 0) == 2
        assert stats.by_type.get("training", 0) == 1

    @pytest.mark.asyncio
    async def test_stats_by_node(self, coordinator, mock_event):
        """Test statistics count by node."""
        coordinator.register_task("task-001", "selfplay", "gpu-1")
        coordinator.register_task("task-002", "training", "gpu-1")
        coordinator.register_task("task-003", "selfplay", "gpu-2")

        stats = coordinator.get_stats()

        assert stats.by_node.get("gpu-1", 0) == 2
        assert stats.by_node.get("gpu-2", 0) == 1

    @pytest.mark.asyncio
    async def test_stats_failure_rate(self, coordinator, mock_event):
        """Test failure rate calculation."""
        # Complete 3 tasks
        for i in range(3):
            coordinator.register_task(f"success-{i}", "selfplay", "gpu-1")
            await coordinator._on_task_completed(mock_event(payload={
                "task_id": f"success-{i}",
            }))

        # Fail 1 task
        coordinator.register_task("failed-1", "selfplay", "gpu-1")
        await coordinator._on_task_failed(mock_event(payload={
            "task_id": "failed-1",
            "error": "Error",
        }))

        stats = coordinator.get_stats()
        assert stats.failure_rate == 0.25  # 1/4

    @pytest.mark.asyncio
    async def test_stats_average_duration(self, coordinator, mock_event):
        """Test average duration calculation."""
        coordinator.register_task("task-001", "selfplay", "gpu-1")
        await coordinator._on_task_completed(mock_event(payload={
            "task_id": "task-001",
            "duration": 100.0,
        }))

        coordinator.register_task("task-002", "selfplay", "gpu-1")
        await coordinator._on_task_completed(mock_event(payload={
            "task_id": "task-002",
            "duration": 200.0,
        }))

        stats = coordinator.get_stats()
        assert stats.average_duration == 150.0


# =============================================================================
# Status Tests
# =============================================================================


class TestStatus:
    """Test status reporting."""

    def test_get_status(self, coordinator):
        """Test get_status returns proper structure."""
        coordinator.register_task("task-001", "selfplay", "gpu-1")

        status = coordinator.get_status()

        assert "active_tasks" in status
        assert "orphaned_tasks" in status
        assert "completed_tasks" in status
        assert "failure_rate" in status
        assert "by_type" in status
        assert "by_node" in status
        assert "online_nodes" in status


# =============================================================================
# TrackedTask Tests
# =============================================================================


class TestTrackedTask:
    """Test TrackedTask dataclass."""

    def test_age_property(self):
        """Test age property."""
        task = TrackedTask(
            task_id="test",
            task_type="selfplay",
            node_id="gpu-1",
            spawned_at=time.time() - 60,
        )
        assert 59 < task.age < 61

    def test_time_since_heartbeat(self):
        """Test time_since_heartbeat property."""
        task = TrackedTask(
            task_id="test",
            task_type="selfplay",
            node_id="gpu-1",
            last_heartbeat=time.time() - 30,
        )
        assert 29 < task.time_since_heartbeat < 31

    def test_is_alive_recent(self):
        """Test is_alive with recent heartbeat."""
        task = TrackedTask(
            task_id="test",
            task_type="selfplay",
            node_id="gpu-1",
        )
        assert task.is_alive(heartbeat_threshold=60.0)

    def test_is_alive_stale(self):
        """Test is_alive with stale heartbeat."""
        task = TrackedTask(
            task_id="test",
            task_type="selfplay",
            node_id="gpu-1",
            last_heartbeat=time.time() - 120,
        )
        assert not task.is_alive(heartbeat_threshold=60.0)

    def test_is_alive_completed(self):
        """Test is_alive returns False for completed tasks."""
        task = TrackedTask(
            task_id="test",
            task_type="selfplay",
            node_id="gpu-1",
            status=TaskStatus.COMPLETED,
        )
        assert not task.is_alive()


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingletonBehavior:
    """Test singleton behavior."""

    def test_get_task_lifecycle_coordinator_returns_singleton(self):
        """Test get_task_lifecycle_coordinator returns same instance."""
        import app.coordination.task_lifecycle_coordinator as tlc
        tlc._task_lifecycle_coordinator = None

        coord1 = get_task_lifecycle_coordinator()
        coord2 = get_task_lifecycle_coordinator()

        assert coord1 is coord2

    def test_convenience_functions(self):
        """Test convenience functions work."""
        import app.coordination.task_lifecycle_coordinator as tlc
        tlc._task_lifecycle_coordinator = None

        coord = get_task_lifecycle_coordinator()
        coord.register_task("test-task", "selfplay", "gpu-1")

        assert get_active_task_count() == 1
        stats = get_task_stats()
        assert stats.active_tasks == 1


# =============================================================================
# TaskStatus Enum Tests
# =============================================================================


class TestTaskStatus:
    """Test TaskStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"
        assert TaskStatus.ORPHANED.value == "orphaned"
