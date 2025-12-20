"""Integration tests for orchestrator event propagation (December 2025).

Tests that events properly flow between orchestrators:
- OptimizationCoordinator -> DataPipelineOrchestrator (CMAES/NAS triggered)
- ResourceMonitoringCoordinator -> SelfplayOrchestrator (backpressure)
- CacheCoordinationOrchestrator -> DataPipelineOrchestrator (cache invalidation)
- TaskLifecycleCoordinator -> host online/offline handling
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

# Test fixtures for event bus mocking


class MockDataEvent:
    """Mock DataEvent for testing."""

    def __init__(self, event_type, payload, source="test"):
        self.event_type = event_type
        self.payload = payload
        self.source = source


class MockDataEventType:
    """Mock DataEventType enum values."""

    CMAES_TRIGGERED = MagicMock(value="cmaes_triggered")
    NAS_TRIGGERED = MagicMock(value="nas_triggered")
    BACKPRESSURE_ACTIVATED = MagicMock(value="backpressure_activated")
    BACKPRESSURE_RELEASED = MagicMock(value="backpressure_released")
    CACHE_INVALIDATED = MagicMock(value="cache_invalidated")
    QUALITY_DISTRIBUTION_CHANGED = MagicMock(value="quality_distribution_changed")
    HOST_ONLINE = MagicMock(value="host_online")
    HOST_OFFLINE = MagicMock(value="host_offline")
    NODE_RECOVERED = MagicMock(value="node_recovered")
    REGRESSION_DETECTED = MagicMock(value="regression_detected")
    RESOURCE_CONSTRAINT_DETECTED = MagicMock(value="resource_constraint_detected")


class TestSelfplayOrchestratorSubscriptions:
    """Tests for SelfplayOrchestrator event subscriptions."""

    @pytest.mark.asyncio
    async def test_backpressure_activated_updates_state(self):
        """Test that BACKPRESSURE_ACTIVATED updates node backpressure state."""
        from app.coordination.selfplay_orchestrator import SelfplayOrchestrator

        orch = SelfplayOrchestrator()

        # Create mock event
        event = MockDataEvent(
            MockDataEventType.BACKPRESSURE_ACTIVATED,
            {
                "node_id": "gh200-a",
                "level": "high",
                "reason": "GPU memory pressure",
            },
        )

        # Call handler directly
        await orch._on_backpressure_activated(event)

        # Verify state updated
        assert orch.is_node_under_backpressure("gh200-a")
        assert orch.get_node_backpressure_level("gh200-a") == "high"

    @pytest.mark.asyncio
    async def test_backpressure_released_clears_state(self):
        """Test that BACKPRESSURE_RELEASED clears node backpressure state."""
        from app.coordination.selfplay_orchestrator import SelfplayOrchestrator

        orch = SelfplayOrchestrator()

        # Set up initial backpressure
        orch._backpressure_nodes["gh200-a"] = "high"

        # Create mock event
        event = MockDataEvent(
            MockDataEventType.BACKPRESSURE_RELEASED,
            {"node_id": "gh200-a"},
        )

        # Call handler directly
        await orch._on_backpressure_released(event)

        # Verify state cleared
        assert not orch.is_node_under_backpressure("gh200-a")
        assert orch.get_node_backpressure_level("gh200-a") is None

    @pytest.mark.asyncio
    async def test_regression_detected_pauses_for_severe(self):
        """Test that severe regression pauses selfplay."""
        from app.coordination.selfplay_orchestrator import SelfplayOrchestrator

        orch = SelfplayOrchestrator()

        # Create mock event
        event = MockDataEvent(
            MockDataEventType.REGRESSION_DETECTED,
            {
                "severity": "severe",
                "metric_name": "elo",
            },
        )

        # Call handler directly
        await orch._on_regression_detected(event)

        # Verify paused
        assert orch.is_paused_for_regression()

    @pytest.mark.asyncio
    async def test_regression_detected_ignores_minor(self):
        """Test that minor regression does not pause selfplay."""
        from app.coordination.selfplay_orchestrator import SelfplayOrchestrator

        orch = SelfplayOrchestrator()

        # Create mock event
        event = MockDataEvent(
            MockDataEventType.REGRESSION_DETECTED,
            {
                "severity": "minor",
                "metric_name": "elo",
            },
        )

        # Call handler directly
        await orch._on_regression_detected(event)

        # Verify not paused
        assert not orch.is_paused_for_regression()


class TestDataPipelineOrchestratorSubscriptions:
    """Tests for DataPipelineOrchestrator event subscriptions."""

    @pytest.mark.asyncio
    async def test_cache_invalidated_sets_refresh_flag(self):
        """Test that model cache invalidation sets pending refresh flag."""
        from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

        orch = DataPipelineOrchestrator()

        # Create mock event
        event = MockDataEvent(
            MockDataEventType.CACHE_INVALIDATED,
            {
                "invalidation_type": "model",
                "target_id": "model_v42",
                "count": 10,
            },
        )

        # Call handler directly
        await orch._on_cache_invalidated(event)

        # Verify flag set
        assert orch.needs_cache_refresh()
        assert orch._cache_invalidation_count == 10

    @pytest.mark.asyncio
    async def test_optimization_triggered_tracks_state(self):
        """Test that CMAES_TRIGGERED updates optimization tracking."""
        from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

        orch = DataPipelineOrchestrator()

        # Create mock event
        event = MockDataEvent(
            MockDataEventType.CMAES_TRIGGERED,
            {
                "run_id": "cmaes_123",
                "reason": "plateau_detected",
            },
        )

        # Call handler directly
        await orch._on_optimization_triggered(event)

        # Verify state tracked
        assert orch.is_optimization_active()
        assert orch.get_active_optimization() == "cmaes"

    @pytest.mark.asyncio
    async def test_quality_distribution_changed_updates_state(self):
        """Test that quality distribution updates are tracked."""
        from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

        orch = DataPipelineOrchestrator()

        # Create mock event
        event = MockDataEvent(
            MockDataEventType.QUALITY_DISTRIBUTION_CHANGED,
            {
                "distribution": {
                    "high": 0.3,
                    "medium": 0.5,
                    "low": 0.2,
                }
            },
        )

        # Call handler directly
        await orch._on_quality_distribution_changed(event)

        # Verify distribution updated
        dist = orch.get_quality_distribution()
        assert dist["high"] == 0.3
        assert dist["low"] == 0.2


class TestTaskLifecycleCoordinatorSubscriptions:
    """Tests for TaskLifecycleCoordinator event subscriptions."""

    @pytest.mark.asyncio
    async def test_host_offline_orphans_tasks(self):
        """Test that HOST_OFFLINE marks node tasks as orphaned."""
        from app.coordination.task_lifecycle_coordinator import (
            TaskLifecycleCoordinator,
            TrackedTask,
            TaskStatus,
        )

        coord = TaskLifecycleCoordinator()

        # Register some active tasks on the node
        coord._active_tasks["task1"] = TrackedTask(
            task_id="task1",
            task_type="selfplay",
            node_id="gh200-a",
        )
        coord._active_tasks["task2"] = TrackedTask(
            task_id="task2",
            task_type="training",
            node_id="gh200-a",
        )
        coord._active_tasks["task3"] = TrackedTask(
            task_id="task3",
            task_type="selfplay",
            node_id="gh200-b",  # Different node
        )

        # Create mock event
        event = MockDataEvent(
            MockDataEventType.HOST_OFFLINE,
            {"node_id": "gh200-a"},
        )

        # Call handler directly
        await coord._on_host_offline(event)

        # Verify tasks orphaned
        assert "task1" not in coord._active_tasks
        assert "task2" not in coord._active_tasks
        assert "task3" in coord._active_tasks  # Different node - not affected

        assert "task1" in coord._orphaned_tasks
        assert "task2" in coord._orphaned_tasks
        assert coord._orphaned_tasks["task1"].status == TaskStatus.ORPHANED

    @pytest.mark.asyncio
    async def test_node_recovered_restores_tasks(self):
        """Test that NODE_RECOVERED restores orphaned tasks."""
        from app.coordination.task_lifecycle_coordinator import (
            TaskLifecycleCoordinator,
            TrackedTask,
            TaskStatus,
        )

        coord = TaskLifecycleCoordinator()

        # Set up orphaned tasks
        task = TrackedTask(
            task_id="task1",
            task_type="selfplay",
            node_id="gh200-a",
            status=TaskStatus.ORPHANED,
        )
        coord._orphaned_tasks["task1"] = task
        coord._offline_nodes["gh200-a"] = 0.0

        # Create mock event
        event = MockDataEvent(
            MockDataEventType.NODE_RECOVERED,
            {"node_id": "gh200-a"},
        )

        # Call handler directly
        await coord._on_node_recovered(event)

        # Verify task restored
        assert "task1" not in coord._orphaned_tasks
        assert "task1" in coord._active_tasks
        assert coord._active_tasks["task1"].status == TaskStatus.RUNNING

    @pytest.mark.asyncio
    async def test_host_online_tracks_node(self):
        """Test that HOST_ONLINE adds node to online set."""
        from app.coordination.task_lifecycle_coordinator import TaskLifecycleCoordinator

        coord = TaskLifecycleCoordinator()

        # Create mock event
        event = MockDataEvent(
            MockDataEventType.HOST_ONLINE,
            {"node_id": "gh200-new"},
        )

        # Call handler directly
        await coord._on_host_online(event)

        # Verify tracked
        assert coord.is_node_online("gh200-new")


class TestCacheCoordinationOrchestratorEmissions:
    """Tests for CacheCoordinationOrchestrator event emissions."""

    def test_invalidate_model_emits_event(self):
        """Test that invalidate_model emits CACHE_INVALIDATED event."""
        from app.coordination.cache_coordination_orchestrator import (
            CacheCoordinationOrchestrator,
            CacheType,
        )

        orch = CacheCoordinationOrchestrator()

        # Register some caches
        orch.register_cache("gh200-a", "nnue_weights", "model_v41", size_bytes=1000)
        orch.register_cache("gh200-b", "nnue_weights", "model_v41", size_bytes=1000)

        # Mock the emit method
        emitted_events = []
        original_emit = orch._emit_cache_invalidated

        def capture_emit(*args, **kwargs):
            emitted_events.append({"args": args, "kwargs": kwargs})

        orch._emit_cache_invalidated = capture_emit

        # Invalidate model
        count = orch.invalidate_model("model_v41")

        # Verify emission was called
        assert count == 2
        assert len(emitted_events) == 1
        assert emitted_events[0]["kwargs"]["invalidation_type"] == "model"
        assert emitted_events[0]["kwargs"]["target_id"] == "model_v41"
        assert emitted_events[0]["kwargs"]["count"] == 2


class TestUnifiedInitialization:
    """Tests for initialize_all_coordinators()."""

    def test_initialize_returns_status_dict(self):
        """Test that initialize_all_coordinators returns status dict."""
        # This test may fail if event bus is not available
        try:
            from app.coordination import initialize_all_coordinators

            status = initialize_all_coordinators()

            # Verify structure
            assert isinstance(status, dict)
            assert "selfplay" in status
            assert "pipeline" in status
            assert "task_lifecycle" in status
            assert "optimization" in status
            assert "metrics" in status
            assert "resources" in status
            assert "cache" in status
            assert "event_coordinator" in status
        except ImportError:
            pytest.skip("Event bus not available")

    def test_get_all_coordinator_status(self):
        """Test that get_all_coordinator_status returns comprehensive status."""
        try:
            from app.coordination import get_all_coordinator_status

            status = get_all_coordinator_status()

            # Verify structure
            assert isinstance(status, dict)
            assert "selfplay" in status
            assert "pipeline" in status

            # Each should have subscribed key
            if "subscribed" in status.get("selfplay", {}):
                assert isinstance(status["selfplay"]["subscribed"], bool)
        except ImportError:
            pytest.skip("Coordination package not fully available")


class TestStatusReporting:
    """Tests for status reporting includes new state."""

    def test_selfplay_status_includes_backpressure(self):
        """Test SelfplayOrchestrator status includes backpressure tracking."""
        from app.coordination.selfplay_orchestrator import SelfplayOrchestrator

        orch = SelfplayOrchestrator()
        orch._backpressure_nodes["gh200-a"] = "high"
        orch._paused_for_regression = True

        status = orch.get_status()

        assert "nodes_under_backpressure" in status
        assert "gh200-a" in status["nodes_under_backpressure"]
        assert status["paused_for_regression"] is True

    def test_pipeline_status_includes_optimization(self):
        """Test DataPipelineOrchestrator status includes optimization tracking."""
        from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

        orch = DataPipelineOrchestrator()
        orch._active_optimization = "cmaes"
        orch._optimization_run_id = "run_123"

        status = orch.get_status()

        assert "active_optimization" in status
        assert status["active_optimization"] == "cmaes"
        assert status["optimization_run_id"] == "run_123"

    def test_task_lifecycle_status_includes_nodes(self):
        """Test TaskLifecycleCoordinator status includes node tracking."""
        from app.coordination.task_lifecycle_coordinator import TaskLifecycleCoordinator

        coord = TaskLifecycleCoordinator()
        coord._online_nodes.add("gh200-a")
        coord._offline_nodes["gh200-b"] = 0.0

        status = coord.get_status()

        assert "online_nodes" in status
        assert "gh200-a" in status["online_nodes"]
        assert "offline_nodes" in status
        assert "gh200-b" in status["offline_nodes"]
