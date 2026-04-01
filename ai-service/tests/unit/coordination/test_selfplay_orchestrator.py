"""Tests for SelfplayOrchestrator - Selfplay event coordination.

Tests cover:
- SelfplayType enum
- SelfplayTaskInfo dataclass
- SelfplayStats dataclass
- SelfplayOrchestrator class (task tracking, stats, callbacks)
- Module helper functions
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.selfplay_orchestrator import (
    SelfplayOrchestrator,
    SelfplayStats,
    SelfplayTaskInfo,
    SelfplayType,
    get_selfplay_orchestrator,
    get_selfplay_stats,
    wire_selfplay_events,
)

# ============================================
# Tests for SelfplayType enum
# ============================================

class TestSelfplayType:
    """Tests for SelfplayType enum."""

    def test_canonical_value(self):
        """Canonical selfplay should have correct value."""
        assert SelfplayType.CANONICAL.value == "canonical"

    def test_gpu_accelerated_value(self):
        """GPU selfplay should have correct value."""
        assert SelfplayType.GPU_ACCELERATED.value == "gpu_selfplay"

    def test_hybrid_value(self):
        """Hybrid selfplay should have correct value."""
        assert SelfplayType.HYBRID.value == "hybrid_selfplay"

    def test_background_value(self):
        """Background selfplay should have correct value."""
        assert SelfplayType.BACKGROUND.value == "background"

    def test_all_types_defined(self):
        """All expected types should be defined."""
        assert len(SelfplayType) == 4


# ============================================
# Tests for SelfplayTaskInfo dataclass
# ============================================

class TestSelfplayTaskInfo:
    """Tests for SelfplayTaskInfo dataclass."""

    def test_minimal_creation(self):
        """Should create with minimal required fields."""
        task = SelfplayTaskInfo(
            task_id="test-task-1",
            selfplay_type=SelfplayType.CANONICAL,
            node_id="node-1",
        )

        assert task.task_id == "test-task-1"
        assert task.selfplay_type == SelfplayType.CANONICAL
        assert task.node_id == "node-1"
        assert task.board_type == "square8"
        assert task.num_players == 2
        assert task.games_requested == 0
        assert task.games_generated == 0
        assert task.success is False
        assert task.error == ""
        assert task.iteration == 0

    def test_full_creation(self):
        """Should create with all fields."""
        start = time.time()
        task = SelfplayTaskInfo(
            task_id="test-task-2",
            selfplay_type=SelfplayType.GPU_ACCELERATED,
            node_id="gpu-node-1",
            board_type="hex8",
            num_players=3,
            games_requested=1000,
            games_generated=500,
            start_time=start,
            end_time=start + 100,
            success=True,
            error="",
            iteration=5,
        )

        assert task.board_type == "hex8"
        assert task.num_players == 3
        assert task.games_requested == 1000
        assert task.games_generated == 500
        assert task.success is True
        assert task.iteration == 5

    def test_duration_completed(self):
        """Should calculate duration for completed task."""
        start = time.time()
        task = SelfplayTaskInfo(
            task_id="task-1",
            selfplay_type=SelfplayType.CANONICAL,
            node_id="node-1",
            start_time=start,
            end_time=start + 60.0,
        )

        assert task.duration == 60.0

    def test_duration_ongoing(self):
        """Should calculate duration for ongoing task."""
        start = time.time() - 30.0  # Started 30 seconds ago
        task = SelfplayTaskInfo(
            task_id="task-1",
            selfplay_type=SelfplayType.CANONICAL,
            node_id="node-1",
            start_time=start,
            end_time=0.0,  # Not completed
        )

        assert 29 < task.duration < 32  # Allow some tolerance

    def test_games_per_second_with_duration(self):
        """Should calculate games per second."""
        start = time.time()
        task = SelfplayTaskInfo(
            task_id="task-1",
            selfplay_type=SelfplayType.CANONICAL,
            node_id="node-1",
            games_generated=100,
            start_time=start,
            end_time=start + 10.0,  # 10 second duration
        )

        assert task.games_per_second == 10.0  # 100 games / 10 seconds

    def test_games_per_second_zero_duration(self):
        """Should handle zero duration gracefully."""
        start = time.time()
        task = SelfplayTaskInfo(
            task_id="task-1",
            selfplay_type=SelfplayType.CANONICAL,
            node_id="node-1",
            games_generated=100,
            start_time=start,
            end_time=start,  # Same time = 0 duration
        )

        assert task.games_per_second == 0.0


# ============================================
# Tests for SelfplayStats dataclass
# ============================================

class TestSelfplayStats:
    """Tests for SelfplayStats dataclass."""

    def test_default_values(self):
        """Should have sensible default values."""
        stats = SelfplayStats()

        assert stats.active_tasks == 0
        assert stats.completed_tasks == 0
        assert stats.failed_tasks == 0
        assert stats.total_games_generated == 0
        assert stats.total_duration_seconds == 0.0
        assert stats.average_games_per_task == 0.0
        assert stats.average_games_per_second == 0.0
        assert stats.by_node == {}
        assert stats.by_type == {}
        assert stats.last_activity_time == 0.0

    def test_custom_values(self):
        """Should accept custom values."""
        stats = SelfplayStats(
            active_tasks=5,
            completed_tasks=100,
            failed_tasks=3,
            total_games_generated=50000,
            total_duration_seconds=3600.0,
            average_games_per_task=500.0,
            average_games_per_second=14.0,
            by_node={"node-1": 50, "node-2": 50},
            by_type={"canonical": 80, "gpu_selfplay": 20},
            last_activity_time=time.time(),
        )

        assert stats.active_tasks == 5
        assert stats.completed_tasks == 100
        assert stats.failed_tasks == 3
        assert stats.total_games_generated == 50000
        assert stats.by_node["node-1"] == 50


# ============================================
# Tests for SelfplayOrchestrator class
# ============================================

class TestSelfplayOrchestrator:
    """Tests for SelfplayOrchestrator class."""

    @pytest.fixture
    def orchestrator(self):
        """Create a fresh orchestrator for each test."""
        return SelfplayOrchestrator(max_history=100, stats_window_seconds=3600.0)

    def test_initialization(self, orchestrator):
        """Should initialize with correct defaults."""
        assert orchestrator.max_history == 100
        assert orchestrator.stats_window_seconds == 3600.0
        assert orchestrator._subscribed is False
        assert len(orchestrator._active_tasks) == 0
        assert len(orchestrator._completed_history) == 0

    def test_register_task(self, orchestrator):
        """Should register a new selfplay task."""
        task = orchestrator.register_task(
            task_id="task-1",
            selfplay_type=SelfplayType.CANONICAL,
            node_id="node-1",
            games_requested=1000,
        )

        assert task.task_id == "task-1"
        assert "task-1" in orchestrator._active_tasks
        assert orchestrator._active_tasks["task-1"].games_requested == 1000

    @pytest.mark.asyncio
    async def test_complete_task(self, orchestrator):
        """Should complete a task and move to history."""
        orchestrator.register_task(
            task_id="task-1",
            selfplay_type=SelfplayType.CANONICAL,
            node_id="node-1",
            games_requested=1000,
        )

        result = await orchestrator.complete_task("task-1", games_generated=800, success=True)

        assert result is not None
        assert result.games_generated == 800
        assert result.success is True
        assert "task-1" not in orchestrator._active_tasks
        assert len(orchestrator._completed_history) == 1

    @pytest.mark.asyncio
    async def test_complete_nonexistent_task(self, orchestrator):
        """Should handle completing nonexistent task gracefully."""
        result = await orchestrator.complete_task("nonexistent", games_generated=100, success=True)
        assert result is None

    @pytest.mark.asyncio
    async def test_fail_task(self, orchestrator):
        """Should fail a task with error message."""
        orchestrator.register_task(
            task_id="task-1",
            selfplay_type=SelfplayType.CANONICAL,
            node_id="node-1",
        )

        result = await orchestrator.complete_task("task-1", success=False, error="Connection lost")

        assert result is not None
        assert result.success is False
        assert result.error == "Connection lost"
        assert "task-1" not in orchestrator._active_tasks
        assert len(orchestrator._completed_history) == 1

    def test_get_active_tasks(self, orchestrator):
        """Should return all active tasks."""
        for i in range(3):
            orchestrator.register_task(
                task_id=f"task-{i}",
                selfplay_type=SelfplayType.CANONICAL,
                node_id=f"node-{i}",
            )

        active = orchestrator.get_active_tasks()
        assert len(active) == 3

    def test_get_task(self, orchestrator):
        """Should return specific task by ID."""
        orchestrator.register_task(
            task_id="task-1",
            selfplay_type=SelfplayType.CANONICAL,
            node_id="node-1",
        )

        task = orchestrator.get_task("task-1")
        assert task is not None
        assert task.task_id == "task-1"

        assert orchestrator.get_task("nonexistent") is None

    @pytest.mark.asyncio
    async def test_get_stats(self, orchestrator):
        """Should calculate accurate statistics."""
        # Register active task
        orchestrator.register_task(
            task_id="active-1",
            selfplay_type=SelfplayType.CANONICAL,
            node_id="node-1",
        )

        # Complete some tasks
        for i in range(3):
            orchestrator.register_task(
                task_id=f"completed-{i}",
                selfplay_type=SelfplayType.CANONICAL,
                node_id="node-1",
                games_requested=100,
            )
            await orchestrator.complete_task(f"completed-{i}", games_generated=100, success=True)

        stats = orchestrator.get_stats()

        assert stats.active_tasks == 1
        assert stats.completed_tasks == 3
        assert stats.total_games_generated == 300

    @pytest.mark.asyncio
    async def test_history_max_limit(self, orchestrator):
        """Should enforce max history limit."""
        orchestrator.max_history = 5

        # Complete more tasks than max_history
        for i in range(10):
            orchestrator.register_task(
                task_id=f"task-{i}",
                selfplay_type=SelfplayType.CANONICAL,
                node_id="node-1",
            )
            await orchestrator.complete_task(f"task-{i}", games_generated=10, success=True)

        assert len(orchestrator._completed_history) == 5

    @pytest.mark.asyncio
    async def test_on_completion_callback(self, orchestrator):
        """Should call completion callbacks."""
        callback_data = []

        def on_complete(task: SelfplayTaskInfo):
            callback_data.append(task)

        orchestrator.on_completion(on_complete)

        orchestrator.register_task(
            task_id="task-1",
            selfplay_type=SelfplayType.CANONICAL,
            node_id="node-1",
        )
        await orchestrator.complete_task("task-1", games_generated=100, success=True)

        assert len(callback_data) == 1
        assert callback_data[0].task_id == "task-1"

    def test_is_selfplay_task(self, orchestrator):
        """Should correctly identify selfplay task types."""
        assert orchestrator._is_selfplay_task("selfplay") is True
        assert orchestrator._is_selfplay_task("gpu_selfplay") is True
        assert orchestrator._is_selfplay_task("hybrid_selfplay") is True
        assert orchestrator._is_selfplay_task("canonical_selfplay") is True
        assert orchestrator._is_selfplay_task("background_selfplay") is True

        assert orchestrator._is_selfplay_task("training") is False
        assert orchestrator._is_selfplay_task("sync") is False
        assert orchestrator._is_selfplay_task("pipeline") is False

    def test_get_selfplay_type_mapping(self, orchestrator):
        """Should map task type strings to enum."""
        assert orchestrator._get_selfplay_type("selfplay") == SelfplayType.CANONICAL
        assert orchestrator._get_selfplay_type("gpu_selfplay") == SelfplayType.GPU_ACCELERATED
        assert orchestrator._get_selfplay_type("hybrid_selfplay") == SelfplayType.HYBRID
        assert orchestrator._get_selfplay_type("background_selfplay") == SelfplayType.BACKGROUND

    def test_backpressure_tracking(self, orchestrator):
        """Should track backpressure states."""
        assert len(orchestrator._backpressure_nodes) == 0
        assert orchestrator._paused_for_regression is False

        # Simulate backpressure
        orchestrator._backpressure_nodes["node-1"] = "high"
        assert "node-1" in orchestrator._backpressure_nodes

    def test_get_history(self, orchestrator):
        """Should return task history with limit."""
        # Add tasks directly to history for testing
        for i in range(10):
            task = SelfplayTaskInfo(
                task_id=f"task-{i}",
                selfplay_type=SelfplayType.CANONICAL,
                node_id="node-1",
                success=True,
            )
            orchestrator._completed_history.append(task)

        history = orchestrator.get_history(limit=5)
        assert len(history) == 5


class TestSelfplayOrchestratorEventHandling:
    """Tests for event handling in SelfplayOrchestrator."""

    @pytest.fixture
    def orchestrator(self):
        """Create a fresh orchestrator for each test."""
        return SelfplayOrchestrator()

    @pytest.mark.asyncio
    async def test_on_task_spawned_selfplay(self, orchestrator):
        """Should handle selfplay task spawned event."""
        # Create mock event with payload attribute
        event = MagicMock()
        event.payload = {
            "task_id": "selfplay-123",
            "task_type": "selfplay",
            "node_id": "node-1",
            "board_type": "square8",
            "num_players": 2,
            "games_requested": 500,
        }

        await orchestrator._on_task_spawned(event)

        assert "selfplay-123" in orchestrator._active_tasks
        task = orchestrator._active_tasks["selfplay-123"]
        assert task.selfplay_type == SelfplayType.CANONICAL
        assert task.games_requested == 500

    @pytest.mark.asyncio
    async def test_on_task_spawned_non_selfplay(self, orchestrator):
        """Should ignore non-selfplay task events."""
        event = MagicMock()
        event.payload = {
            "task_id": "training-123",
            "task_type": "training",
            "node_id": "node-1",
        }

        await orchestrator._on_task_spawned(event)

        assert "training-123" not in orchestrator._active_tasks

    @pytest.mark.asyncio
    async def test_on_task_completed(self, orchestrator):
        """Should handle task completed event."""
        # First register the task
        orchestrator.register_task(
            task_id="selfplay-123",
            selfplay_type=SelfplayType.CANONICAL,
            node_id="node-1",
            games_requested=500,
        )

        event = MagicMock()
        event.payload = {
            "task_id": "selfplay-123",
            "games_generated": 500,
            "success": True,
        }

        await orchestrator._on_task_completed(event)

        assert "selfplay-123" not in orchestrator._active_tasks
        assert len(orchestrator._completed_history) == 1
        assert orchestrator._completed_history[0].success is True

    @pytest.mark.asyncio
    async def test_on_task_failed(self, orchestrator):
        """Should handle task failed event."""
        orchestrator.register_task(
            task_id="selfplay-123",
            selfplay_type=SelfplayType.CANONICAL,
            node_id="node-1",
        )

        event = MagicMock()
        event.payload = {
            "task_id": "selfplay-123",
            "error": "Out of memory",
        }

        await orchestrator._on_task_failed(event)

        assert "selfplay-123" not in orchestrator._active_tasks
        assert len(orchestrator._completed_history) == 1
        assert orchestrator._completed_history[0].success is False
        assert "memory" in orchestrator._completed_history[0].error.lower()

    @pytest.mark.asyncio
    async def test_on_quality_updated_uses_publish_sync_for_adjustment_and_rebalance(self, orchestrator):
        """Quality updates from this async handler should still use sync-safe publish for side effects."""
        orchestrator._quality_scores["hex8_2p"] = 0.8
        orchestrator._quality_budget_multipliers["hex8_2p"] = 1.0
        orchestrator._emit_selfplay_budget_adjusted = MagicMock()

        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "quality_score": 0.4,
        }
        mock_router = MagicMock()

        with patch("app.coordination.event_router.get_router", return_value=mock_router):
            await orchestrator._on_quality_updated(event)

        assert mock_router.publish_sync.call_count == 2
        first_call = mock_router.publish_sync.call_args_list[0]
        second_call = mock_router.publish_sync.call_args_list[1]

        assert first_call.args[0] == "quality_feedback_adjusted"
        assert first_call.args[1]["config_key"] == "hex8_2p"
        assert first_call.args[2] == "SelfplayOrchestrator"

        assert second_call.args[0] == "CURRICULUM_REBALANCED"
        assert second_call.args[1]["changed_configs"] == ["hex8_2p"]
        assert second_call.args[2] == "SelfplayOrchestrator"

    def test_sync_emit_helpers_use_publish_sync(self, orchestrator):
        """Synchronous helper emitters should not drop async router coroutines."""
        mock_router = MagicMock()

        with patch("app.coordination.event_router.get_router", return_value=mock_router):
            orchestrator._emit_selfplay_request_queued(
                task_id="selfplay-123",
                config_key="hex8_2p",
                num_games=50,
                priority="high",
                source="quality_feedback",
                routed_to_p2p=True,
            )
            orchestrator._emit_selfplay_budget_adjusted(
                config_key="hex8_2p",
                old_multiplier=1.0,
                new_multiplier=1.5,
                quality_score=0.42,
                reason="quality_feedback",
            )
            orchestrator._emit_curriculum_allocation_changed(
                changed_weights={"hex8_2p": 1.5},
                trigger="quality_feedback",
                num_changes=1,
            )

        assert mock_router.publish_sync.call_count == 3
        emitted_types = [call.args[0] for call in mock_router.publish_sync.call_args_list]
        assert emitted_types == [
            "REQUEST_SELFPLAY_QUEUED",
            "SELFPLAY_BUDGET_ADJUSTED",
            "CURRICULUM_ALLOCATION_CHANGED",
        ]

    @pytest.mark.asyncio
    async def test_on_task_orphaned(self, orchestrator):
        """Should handle task orphaned event."""
        orchestrator.register_task(
            task_id="selfplay-123",
            selfplay_type=SelfplayType.CANONICAL,
            node_id="node-1",
        )

        event = MagicMock()
        event.payload = {
            "task_id": "selfplay-123",
            "task_type": "selfplay",
            "reason": "no heartbeat",
        }

        await orchestrator._on_task_orphaned(event)

        assert "selfplay-123" not in orchestrator._active_tasks
        assert len(orchestrator._completed_history) == 1
        assert orchestrator._completed_history[0].success is False


# ============================================
# Tests for module-level functions
# ============================================

class TestModuleFunctions:
    """Tests for module-level helper functions."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton for each test."""
        # Clear the module-level singleton
        import app.coordination.selfplay_orchestrator as module
        module._selfplay_orchestrator = None
        yield
        module._selfplay_orchestrator = None

    def test_get_selfplay_orchestrator(self):
        """Should return singleton instance."""
        orch1 = get_selfplay_orchestrator()
        orch2 = get_selfplay_orchestrator()

        assert orch1 is orch2
        assert isinstance(orch1, SelfplayOrchestrator)

    def test_wire_selfplay_events(self):
        """Should wire events and return orchestrator."""
        orchestrator = wire_selfplay_events()

        assert isinstance(orchestrator, SelfplayOrchestrator)
        # Subscribed state depends on event bus availability
        # Just verify it returns the orchestrator

    def test_get_selfplay_stats(self):
        """Should return stats from singleton."""
        stats = get_selfplay_stats()

        assert isinstance(stats, SelfplayStats)
        assert stats.active_tasks == 0  # Fresh orchestrator


# ============================================
# Integration tests
# ============================================

class TestSelfplayOrchestratorIntegration:
    """Integration tests for selfplay orchestration workflow."""

    @pytest.fixture
    def orchestrator(self):
        """Create a fresh orchestrator for each test."""
        return SelfplayOrchestrator(max_history=100)

    @pytest.mark.asyncio
    async def test_full_selfplay_lifecycle(self, orchestrator):
        """Should handle complete selfplay task lifecycle."""
        # Register multiple tasks
        for i in range(5):
            orchestrator.register_task(
                task_id=f"selfplay-{i}",
                selfplay_type=SelfplayType.CANONICAL if i % 2 == 0 else SelfplayType.GPU_ACCELERATED,
                node_id=f"node-{i % 3}",
                games_requested=1000,
            )

        assert len(orchestrator.get_active_tasks()) == 5

        # Complete some tasks
        for i in range(3):
            await orchestrator.complete_task(
                f"selfplay-{i}",
                games_generated=1000,
                success=True,
            )

        # Fail one task
        await orchestrator.complete_task("selfplay-3", success=False, error="Node disconnected")

        # Check stats
        stats = orchestrator.get_stats()
        assert stats.active_tasks == 1  # Only selfplay-4 remains
        assert stats.completed_tasks == 3
        assert stats.failed_tasks == 1
        assert stats.total_games_generated == 3000

    @pytest.mark.asyncio
    async def test_throughput_calculation(self, orchestrator):
        """Should calculate accurate throughput."""
        start = time.time()

        # Complete several tasks with known durations
        for i in range(5):
            orchestrator.register_task(
                task_id=f"task-{i}",
                selfplay_type=SelfplayType.CANONICAL,
                node_id="node-1",
                games_requested=100,
            )
            # Manually set start_time for predictable duration
            orchestrator._active_tasks[f"task-{i}"].start_time = start - 10
            await orchestrator.complete_task(f"task-{i}", games_generated=100, success=True)

        stats = orchestrator.get_stats()
        assert stats.total_games_generated == 500
        assert stats.average_games_per_task == 100.0

    def test_multi_node_distribution(self, orchestrator):
        """Should track distribution across nodes."""
        nodes = ["node-1", "node-2", "node-3"]

        # Register tasks across nodes
        for i, node in enumerate(nodes * 3):  # 9 tasks total
            orchestrator.register_task(
                task_id=f"task-{i}",
                selfplay_type=SelfplayType.CANONICAL,
                node_id=node,
            )

        # Check distribution via stats
        stats = orchestrator.get_stats()
        assert stats.active_tasks == 9
        assert len(stats.by_node) == 3
        for node in nodes:
            assert stats.by_node[node] == 3

    def test_type_distribution(self, orchestrator):
        """Should track distribution across selfplay types."""
        # Register different types
        types = [
            SelfplayType.CANONICAL,
            SelfplayType.CANONICAL,
            SelfplayType.GPU_ACCELERATED,
            SelfplayType.HYBRID,
        ]

        for i, sp_type in enumerate(types):
            orchestrator.register_task(
                task_id=f"task-{i}",
                selfplay_type=sp_type,
                node_id="node-1",
            )

        # Check distribution via stats
        stats = orchestrator.get_stats()
        assert stats.by_type["canonical"] == 2
        assert stats.by_type["gpu_selfplay"] == 1
        assert stats.by_type["hybrid_selfplay"] == 1
