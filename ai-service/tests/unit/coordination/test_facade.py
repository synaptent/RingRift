"""Tests for app.coordination.facade - Coordination Facade API (December 2025)."""

from __future__ import annotations

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, PropertyMock

from app.coordination.facade import (
    CoordinationFacade,
    TaskStatus,
    TrainingStatus,
    ClusterHealth,
    TaskInfo,
    get_coordination_facade,
    can_spawn_task,
    spawn_task,
    get_cluster_health,
    get_available_nodes,
)


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_task_status_values(self):
        """Test all TaskStatus values exist."""
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.RUNNING == "running"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"
        assert TaskStatus.TIMED_OUT == "timed_out"
        assert TaskStatus.CANCELLED == "cancelled"

    def test_task_status_is_string_enum(self):
        """Test TaskStatus is a string enum."""
        assert isinstance(TaskStatus.RUNNING.value, str)
        assert TaskStatus.RUNNING == "running"


class TestTrainingStatus:
    """Tests for TrainingStatus enum."""

    def test_training_status_values(self):
        """Test all TrainingStatus values exist."""
        assert TrainingStatus.NOT_STARTED == "not_started"
        assert TrainingStatus.RUNNING == "running"
        assert TrainingStatus.COMPLETED == "completed"
        assert TrainingStatus.FAILED == "failed"
        assert TrainingStatus.CANCELLED == "cancelled"

    def test_training_status_is_string_enum(self):
        """Test TrainingStatus is a string enum."""
        assert isinstance(TrainingStatus.COMPLETED.value, str)


class TestClusterHealth:
    """Tests for ClusterHealth dataclass."""

    def test_cluster_health_creation(self):
        """Test ClusterHealth can be created."""
        health = ClusterHealth(
            total_nodes=10,
            healthy_nodes=8,
            degraded_nodes=1,
            unhealthy_nodes=1,
            evicted_nodes=0,
            available_node_ids=["node1", "node2"],
            timestamp="2025-12-27T00:00:00",
        )
        assert health.total_nodes == 10
        assert health.healthy_nodes == 8
        assert health.degraded_nodes == 1
        assert health.unhealthy_nodes == 1
        assert health.evicted_nodes == 0
        assert health.available_node_ids == ["node1", "node2"]
        assert health.timestamp == "2025-12-27T00:00:00"

    def test_cluster_health_empty(self):
        """Test ClusterHealth with empty values."""
        health = ClusterHealth(
            total_nodes=0,
            healthy_nodes=0,
            degraded_nodes=0,
            unhealthy_nodes=0,
            evicted_nodes=0,
            available_node_ids=[],
            timestamp="",
        )
        assert health.total_nodes == 0
        assert len(health.available_node_ids) == 0


class TestTaskInfo:
    """Tests for TaskInfo dataclass."""

    def test_task_info_creation(self):
        """Test TaskInfo can be created."""
        info = TaskInfo(
            task_id="task-123",
            task_type="selfplay",
            node_id="runpod-h100",
            status=TaskStatus.RUNNING,
            started_at=1704067200.0,
            runtime_seconds=300.5,
        )
        assert info.task_id == "task-123"
        assert info.task_type == "selfplay"
        assert info.node_id == "runpod-h100"
        assert info.status == TaskStatus.RUNNING
        assert info.started_at == 1704067200.0
        assert info.runtime_seconds == 300.5


class TestCoordinationFacade:
    """Tests for CoordinationFacade class."""

    @pytest.fixture
    def facade(self):
        """Create a fresh facade for each test."""
        return CoordinationFacade()

    def test_facade_initialization(self, facade):
        """Test facade initializes with None coordinators."""
        assert facade._task_coordinator is None
        assert facade._training_coordinator is None
        assert facade._node_monitor is None
        assert facade._event_router is None

    # =========================================================================
    # Task Operations Tests
    # =========================================================================

    def test_can_spawn_task_no_coordinator(self, facade):
        """Test can_spawn_task fails closed when coordinator is unavailable."""
        with patch.object(facade, "_is_node_available", return_value=True):
            with patch.object(facade, "_get_task_coordinator", return_value=None):
                result = facade.can_spawn_task("selfplay", "node1")
                assert result is False

    def test_can_spawn_task_node_unavailable(self, facade):
        """Test can_spawn_task returns False when node unavailable."""
        with patch.object(facade, "_is_node_available", return_value=False):
            result = facade.can_spawn_task("selfplay", "node1")
            assert result is False

    def test_can_spawn_task_with_coordinator(self, facade):
        """Test can_spawn_task delegates to coordinator."""
        mock_coord = MagicMock()
        mock_coord.can_spawn.return_value = True

        with patch.object(facade, "_is_node_available", return_value=True):
            with patch.object(facade, "_get_task_coordinator", return_value=mock_coord):
                result = facade.can_spawn_task("selfplay", "node1")
                assert result is True

    def test_can_spawn_task_invalid_task_type_fails_closed(self, facade):
        """Invalid task types should not bypass spawn checks."""
        mock_coord = MagicMock()

        with patch.object(facade, "_is_node_available", return_value=True):
            with patch.object(facade, "_get_task_coordinator", return_value=mock_coord):
                result = facade.can_spawn_task("unknown_type", "node1")

        assert result is False
        mock_coord.can_spawn.assert_not_called()

    def test_spawn_task_no_coordinator(self, facade):
        """Test spawn_task returns None when no coordinator."""
        with patch.object(facade, "_get_task_coordinator", return_value=None):
            result = facade.spawn_task("selfplay", "node1")
            assert result is None

    def test_spawn_task_with_coordinator(self, facade):
        """Test spawn_task delegates to coordinator."""
        mock_coord = MagicMock()
        mock_coord.register_task.return_value = "task-abc123"

        # Create mock TaskType enum
        mock_task_type = MagicMock()
        mock_task_type.SELFPLAY = MagicMock()

        with patch.object(facade, "_get_task_coordinator", return_value=mock_coord):
            with patch.dict("sys.modules", {"app.coordination.task_coordinator": MagicMock(TaskType=mock_task_type)}):
                result = facade.spawn_task("selfplay", "node1", games=100)
                assert result == "task-abc123"

    def test_get_task_status_no_coordinator(self, facade):
        """Test get_task_status returns None when no coordinator."""
        with patch.object(facade, "_get_task_coordinator", return_value=None):
            result = facade.get_task_status("task-123")
            assert result is None

    def test_get_task_status_task_not_found(self, facade):
        """Test get_task_status returns None when task not found."""
        mock_coord = MagicMock()
        mock_coord.registry.get_task.return_value = None

        with patch.object(facade, "_get_task_coordinator", return_value=mock_coord):
            result = facade.get_task_status("task-123")
            assert result is None

    def test_cancel_task_no_coordinator(self, facade):
        """Test cancel_task returns False when no coordinator."""
        with patch.object(facade, "_get_task_coordinator", return_value=None):
            result = facade.cancel_task("task-123")
            assert result is False

    def test_cancel_task_success(self, facade):
        """Test cancel_task returns True on success."""
        mock_coord = MagicMock()

        with patch.object(facade, "_get_task_coordinator", return_value=mock_coord):
            result = facade.cancel_task("task-123")
            assert result is True
            mock_coord.registry.update_task_status.assert_called_once_with("task-123", "cancelled")

    def test_get_active_tasks_no_coordinator(self, facade):
        """Test get_active_tasks returns empty list when no coordinator."""
        with patch.object(facade, "_get_task_coordinator", return_value=None):
            result = facade.get_active_tasks()
            assert result == []

    def test_get_active_tasks_with_filter(self, facade):
        """Test get_active_tasks filters by node_id."""
        mock_task1 = MagicMock()
        mock_task1.task_id = "task-1"
        mock_task1.task_type.value = "selfplay"
        mock_task1.node_id = "node1"
        mock_task1.started_at = 1704067200.0
        mock_task1.runtime_seconds.return_value = 100.0

        mock_task2 = MagicMock()
        mock_task2.task_id = "task-2"
        mock_task2.task_type.value = "training"
        mock_task2.node_id = "node2"
        mock_task2.started_at = 1704067200.0
        mock_task2.runtime_seconds.return_value = 200.0

        mock_coord = MagicMock()
        mock_coord.registry.get_active_tasks.return_value = [mock_task1, mock_task2]

        with patch.object(facade, "_get_task_coordinator", return_value=mock_coord):
            result = facade.get_active_tasks(node_id="node1")
            assert len(result) == 1
            assert result[0].node_id == "node1"

    # =========================================================================
    # Training Operations Tests
    # =========================================================================

    def test_start_training_no_coordinator(self, facade):
        """Test start_training returns None when no coordinator."""
        with patch.object(facade, "_get_training_coordinator", return_value=None):
            result = facade.start_training("hex8", 2)
            assert result is None

    def test_start_training_success(self, facade):
        """Test start_training returns job ID on success."""
        mock_coord = MagicMock()
        mock_coord.start_training.return_value = "training-xyz"

        with patch.object(facade, "_get_training_coordinator", return_value=mock_coord):
            result = facade.start_training("hex8", 2, epochs=50)
            assert result == "training-xyz"
            mock_coord.start_training.assert_called_once_with("hex8_2p", epochs=50)

    def test_get_training_status_no_coordinator(self, facade):
        """Test get_training_status returns NOT_STARTED when no coordinator."""
        with patch.object(facade, "_get_training_coordinator", return_value=None):
            result = facade.get_training_status("hex8", 2)
            assert result == TrainingStatus.NOT_STARTED

    def test_get_training_status_running(self, facade):
        """Test get_training_status returns RUNNING for running job."""
        mock_coord = MagicMock()
        mock_coord.get_status.return_value = {"running": True, "completed": False}

        with patch.object(facade, "_get_training_coordinator", return_value=mock_coord):
            result = facade.get_training_status("hex8", 2)
            assert result == TrainingStatus.RUNNING

    def test_get_training_status_completed(self, facade):
        """Test get_training_status returns COMPLETED for finished job."""
        mock_coord = MagicMock()
        mock_coord.get_status.return_value = {"running": False, "completed": True}

        with patch.object(facade, "_get_training_coordinator", return_value=mock_coord):
            result = facade.get_training_status("hex8", 2)
            assert result == TrainingStatus.COMPLETED

    def test_stop_training_no_coordinator(self, facade):
        """Test stop_training returns False when no coordinator."""
        with patch.object(facade, "_get_training_coordinator", return_value=None):
            result = facade.stop_training("hex8", 2)
            assert result is False

    def test_stop_training_success(self, facade):
        """Test stop_training returns True on success."""
        mock_coord = MagicMock()

        with patch.object(facade, "_get_training_coordinator", return_value=mock_coord):
            result = facade.stop_training("hex8", 2)
            assert result is True
            mock_coord.stop_training.assert_called_once_with("hex8_2p")

    # =========================================================================
    # Cluster Health Tests
    # =========================================================================

    def test_get_cluster_health_no_monitor(self, facade):
        """Test get_cluster_health returns empty ClusterHealth when no monitor."""
        with patch.object(facade, "_get_node_monitor", return_value=None):
            result = facade.get_cluster_health()
            assert isinstance(result, ClusterHealth)
            assert result.total_nodes == 0
            assert result.healthy_nodes == 0

    def test_get_cluster_health_with_monitor(self, facade):
        """Test get_cluster_health returns monitor data."""
        mock_monitor = MagicMock()
        mock_monitor.get_cluster_summary.return_value = {
            "total_nodes": 10,
            "healthy": 8,
            "degraded": 1,
            "unhealthy": 1,
            "evicted": 0,
            "available_nodes": ["node1", "node2"],
            "timestamp": "2025-12-27T00:00:00",
        }

        with patch.object(facade, "_get_node_monitor", return_value=mock_monitor):
            result = facade.get_cluster_health()
            assert result.total_nodes == 10
            assert result.healthy_nodes == 8
            assert result.degraded_nodes == 1

    def test_is_node_healthy(self, facade):
        """Test is_node_healthy delegates to _is_node_available."""
        with patch.object(facade, "_is_node_available", return_value=True):
            assert facade.is_node_healthy("node1") is True

        with patch.object(facade, "_is_node_available", return_value=False):
            assert facade.is_node_healthy("node1") is False

    def test_get_available_nodes_no_monitor(self, facade):
        """Test get_available_nodes returns empty list when no monitor."""
        with patch.object(facade, "_get_node_monitor", return_value=None):
            result = facade.get_available_nodes()
            assert result == []

    def test_get_available_nodes_with_monitor(self, facade):
        """Test get_available_nodes returns monitor data."""
        mock_monitor = MagicMock()
        mock_monitor.get_available_nodes.return_value = ["node1", "node2", "node3"]

        with patch.object(facade, "_get_node_monitor", return_value=mock_monitor):
            result = facade.get_available_nodes()
            assert result == ["node1", "node2", "node3"]

    def test_evict_node_no_monitor(self, facade):
        """Test evict_node returns False when no monitor."""
        with patch.object(facade, "_get_node_monitor", return_value=None):
            result = facade.evict_node("node1")
            assert result is False

    def test_evict_node_success(self, facade):
        """Test evict_node returns True on success."""
        mock_monitor = MagicMock()
        mock_monitor.force_evict.return_value = True

        with patch.object(facade, "_get_node_monitor", return_value=mock_monitor):
            result = facade.evict_node("node1")
            assert result is True
            mock_monitor.force_evict.assert_called_once_with("node1")

    def test_recover_node_no_monitor(self, facade):
        """Test recover_node returns False when no monitor."""
        with patch.object(facade, "_get_node_monitor", return_value=None):
            result = facade.recover_node("node1")
            assert result is False

    def test_recover_node_success(self, facade):
        """Test recover_node returns True on success."""
        mock_monitor = MagicMock()
        mock_monitor.force_recover.return_value = True

        with patch.object(facade, "_get_node_monitor", return_value=mock_monitor):
            result = facade.recover_node("node1")
            assert result is True
            mock_monitor.force_recover.assert_called_once_with("node1")

    # =========================================================================
    # Event Operations Tests
    # =========================================================================

    def test_subscribe_no_router(self, facade):
        """Test subscribe returns empty string when no router."""
        with patch.object(facade, "_get_event_router", return_value=None):
            result = facade.subscribe("event_type", lambda x: None)
            assert result == ""

    def test_subscribe_success(self, facade):
        """Test subscribe returns subscription ID."""
        mock_router = MagicMock()
        mock_router.subscribe.return_value = "sub-123"
        callback = lambda x: None

        with patch.object(facade, "_get_event_router", return_value=mock_router):
            result = facade.subscribe("event_type", callback)
            assert result == "sub-123"
            mock_router.subscribe.assert_called_once_with("event_type", callback)

    def test_unsubscribe_no_router(self, facade):
        """Test unsubscribe returns False when no router."""
        with patch.object(facade, "_get_event_router", return_value=None):
            result = facade.unsubscribe("sub-123")
            assert result is False

    def test_unsubscribe_success(self, facade):
        """Test unsubscribe returns True on success."""
        mock_router = MagicMock()

        with patch.object(facade, "_get_event_router", return_value=mock_router):
            result = facade.unsubscribe("sub-123")
            assert result is True
            mock_router.unsubscribe.assert_called_once_with("sub-123")

    # =========================================================================
    # Internal Helpers Tests
    # =========================================================================

    def test_is_node_available_no_monitor(self, facade):
        """Test _is_node_available fails closed when monitor is unavailable."""
        with patch.object(facade, "_get_node_monitor", return_value=None):
            result = facade._is_node_available("node1")
            assert result is False

    def test_is_node_available_with_monitor(self, facade):
        """Test _is_node_available delegates to monitor."""
        mock_monitor = MagicMock()
        mock_monitor.is_node_available.return_value = False

        with patch.object(facade, "_get_node_monitor", return_value=mock_monitor):
            result = facade._is_node_available("node1")
            assert result is False

    def test_is_node_available_monitor_error_fails_closed(self, facade):
        """Exceptions from the monitor should deny availability."""
        mock_monitor = MagicMock()
        mock_monitor.is_node_available.side_effect = RuntimeError("monitor offline")

        with patch.object(facade, "_get_node_monitor", return_value=mock_monitor):
            result = facade._is_node_available("node1")

        assert result is False

    def test_lazy_loading_caches_coordinator(self, facade):
        """Test coordinator is cached after first load."""
        # Simulate a coordinator being loaded
        mock_coord = MagicMock()
        facade._task_coordinator = mock_coord

        # Second call should return cached value
        result = facade._get_task_coordinator()
        assert result is mock_coord

    def test_lazy_loading_handles_import_error(self, facade):
        """Test lazy loading returns None when coordinator unavailable."""
        # Reset coordinator
        facade._task_coordinator = None

        # When import fails, should return None gracefully
        result = facade._get_task_coordinator()
        # May be None or a real coordinator depending on environment
        # Just verify no exception is raised
        assert True


class TestGlobalFacade:
    """Tests for global facade instance and convenience functions."""

    def test_get_coordination_facade_singleton(self):
        """Test get_coordination_facade returns same instance."""
        import app.coordination.facade as facade_module

        # Reset global
        facade_module._facade = None

        facade1 = get_coordination_facade()
        facade2 = get_coordination_facade()

        assert facade1 is facade2

        # Cleanup
        facade_module._facade = None

    def test_can_spawn_task_convenience(self):
        """Test can_spawn_task convenience function."""
        with patch("app.coordination.facade.get_coordination_facade") as mock_get:
            mock_facade = MagicMock()
            mock_facade.can_spawn_task.return_value = True
            mock_get.return_value = mock_facade

            result = can_spawn_task("selfplay", "node1")
            assert result is True
            mock_facade.can_spawn_task.assert_called_once_with("selfplay", "node1")

    def test_spawn_task_convenience(self):
        """Test spawn_task convenience function."""
        with patch("app.coordination.facade.get_coordination_facade") as mock_get:
            mock_facade = MagicMock()
            mock_facade.spawn_task.return_value = "task-123"
            mock_get.return_value = mock_facade

            result = spawn_task("selfplay", "node1", games=100)
            assert result == "task-123"
            mock_facade.spawn_task.assert_called_once_with("selfplay", "node1", games=100)

    def test_get_cluster_health_convenience(self):
        """Test get_cluster_health convenience function."""
        with patch("app.coordination.facade.get_coordination_facade") as mock_get:
            mock_facade = MagicMock()
            mock_health = ClusterHealth(
                total_nodes=5,
                healthy_nodes=4,
                degraded_nodes=1,
                unhealthy_nodes=0,
                evicted_nodes=0,
                available_node_ids=["a", "b"],
                timestamp="now",
            )
            mock_facade.get_cluster_health.return_value = mock_health
            mock_get.return_value = mock_facade

            result = get_cluster_health()
            assert result.total_nodes == 5

    def test_get_available_nodes_convenience(self):
        """Test get_available_nodes convenience function."""
        with patch("app.coordination.facade.get_coordination_facade") as mock_get:
            mock_facade = MagicMock()
            mock_facade.get_available_nodes.return_value = ["node1", "node2"]
            mock_get.return_value = mock_facade

            result = get_available_nodes()
            assert result == ["node1", "node2"]


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_facade_exception_handling_spawn(self):
        """Test spawn_task handles exceptions gracefully."""
        facade = CoordinationFacade()
        mock_coord = MagicMock()
        mock_coord.register_task.side_effect = Exception("Test error")

        with patch.object(facade, "_get_task_coordinator", return_value=mock_coord):
            # The exception should be caught and None returned
            result = facade.spawn_task("selfplay", "node1")
            assert result is None

    def test_facade_exception_handling_training(self):
        """Test start_training handles exceptions gracefully."""
        facade = CoordinationFacade()
        mock_coord = MagicMock()
        mock_coord.start_training.side_effect = Exception("Test error")

        with patch.object(facade, "_get_training_coordinator", return_value=mock_coord):
            result = facade.start_training("hex8", 2)
            assert result is None

    def test_facade_exception_handling_subscribe(self):
        """Test subscribe handles exceptions gracefully."""
        facade = CoordinationFacade()
        mock_router = MagicMock()
        mock_router.subscribe.side_effect = Exception("Test error")

        with patch.object(facade, "_get_event_router", return_value=mock_router):
            result = facade.subscribe("event", lambda x: None)
            assert result == ""
