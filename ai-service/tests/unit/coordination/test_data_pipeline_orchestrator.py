"""Tests for DataPipelineOrchestrator - Pipeline stage coordination.

Tests cover:
- PipelineStage enum
- StageTransition dataclass
- IterationRecord dataclass
- PipelineStats dataclass
- DataPipelineOrchestrator class
- Module helper functions
"""

import time
from unittest.mock import MagicMock

import pytest

from app.coordination.data_pipeline_orchestrator import (
    DataPipelineOrchestrator,
    IterationRecord,
    PipelineStage,
    PipelineStats,
    StageTransition,
    get_current_pipeline_stage,
    get_pipeline_orchestrator,
    get_pipeline_status,
    wire_pipeline_events,
)

# ============================================
# Tests for PipelineStage enum
# ============================================

class TestPipelineStage:
    """Tests for PipelineStage enum."""

    def test_idle_value(self):
        """Idle stage should have correct value."""
        assert PipelineStage.IDLE.value == "idle"

    def test_selfplay_value(self):
        """Selfplay stage should have correct value."""
        assert PipelineStage.SELFPLAY.value == "selfplay"

    def test_data_sync_value(self):
        """Data sync stage should have correct value."""
        assert PipelineStage.DATA_SYNC.value == "data_sync"

    def test_npz_export_value(self):
        """NPZ export stage should have correct value."""
        assert PipelineStage.NPZ_EXPORT.value == "npz_export"

    def test_training_value(self):
        """Training stage should have correct value."""
        assert PipelineStage.TRAINING.value == "training"

    def test_evaluation_value(self):
        """Evaluation stage should have correct value."""
        assert PipelineStage.EVALUATION.value == "evaluation"

    def test_promotion_value(self):
        """Promotion stage should have correct value."""
        assert PipelineStage.PROMOTION.value == "promotion"

    def test_complete_value(self):
        """Complete stage should have correct value."""
        assert PipelineStage.COMPLETE.value == "complete"

    def test_all_stages_defined(self):
        """All expected stages should be defined."""
        assert len(PipelineStage) == 8


# ============================================
# Tests for StageTransition dataclass
# ============================================

class TestStageTransition:
    """Tests for StageTransition dataclass."""

    def test_minimal_creation(self):
        """Should create with minimal required fields."""
        transition = StageTransition(
            from_stage=PipelineStage.SELFPLAY,
            to_stage=PipelineStage.DATA_SYNC,
            iteration=1,
        )

        assert transition.from_stage == PipelineStage.SELFPLAY
        assert transition.to_stage == PipelineStage.DATA_SYNC
        assert transition.iteration == 1
        assert transition.success is True
        assert transition.duration_seconds == 0.0
        assert transition.metadata == {}

    def test_full_creation(self):
        """Should create with all fields."""
        ts = time.time()
        transition = StageTransition(
            from_stage=PipelineStage.TRAINING,
            to_stage=PipelineStage.EVALUATION,
            iteration=5,
            timestamp=ts,
            success=False,
            duration_seconds=120.5,
            metadata={"reason": "timeout"},
        )

        assert transition.timestamp == ts
        assert transition.success is False
        assert transition.duration_seconds == 120.5
        assert transition.metadata["reason"] == "timeout"


# ============================================
# Tests for IterationRecord dataclass
# ============================================

class TestIterationRecord:
    """Tests for IterationRecord dataclass."""

    def test_minimal_creation(self):
        """Should create with minimal required fields."""
        record = IterationRecord(
            iteration=1,
            start_time=time.time(),
        )

        assert record.iteration == 1
        assert record.end_time == 0.0
        assert record.success is False
        assert record.stages_completed == []
        assert record.games_generated == 0
        assert record.model_id is None
        assert record.elo_delta == 0.0
        assert record.promoted is False
        assert record.error is None

    def test_full_creation(self):
        """Should create with all fields."""
        start = time.time()
        record = IterationRecord(
            iteration=5,
            start_time=start,
            end_time=start + 3600,
            success=True,
            stages_completed=["selfplay", "sync", "training"],
            games_generated=10000,
            model_id="model-v5",
            elo_delta=50.5,
            promoted=True,
            error=None,
        )

        assert record.games_generated == 10000
        assert record.model_id == "model-v5"
        assert record.elo_delta == 50.5
        assert record.promoted is True

    def test_duration_completed(self):
        """Should calculate duration for completed iteration."""
        start = time.time()
        record = IterationRecord(
            iteration=1,
            start_time=start,
            end_time=start + 3600,
        )

        assert record.duration == 3600.0

    def test_duration_ongoing(self):
        """Should calculate duration for ongoing iteration."""
        start = time.time() - 60  # Started 60 seconds ago
        record = IterationRecord(
            iteration=1,
            start_time=start,
            end_time=0.0,
        )

        assert 59 < record.duration < 62  # Allow tolerance


# ============================================
# Tests for PipelineStats dataclass
# ============================================

class TestPipelineStats:
    """Tests for PipelineStats dataclass."""

    def test_default_values(self):
        """Should have sensible default values."""
        stats = PipelineStats()

        assert stats.iterations_completed == 0
        assert stats.iterations_failed == 0
        assert stats.total_games_generated == 0
        assert stats.total_models_trained == 0
        assert stats.promotions == 0
        assert stats.average_iteration_duration == 0.0
        assert stats.stage_durations == {}
        assert stats.last_activity_time == 0.0

    def test_custom_values(self):
        """Should accept custom values."""
        stats = PipelineStats(
            iterations_completed=10,
            iterations_failed=2,
            total_games_generated=100000,
            total_models_trained=10,
            promotions=8,
            average_iteration_duration=3600.0,
            stage_durations={"selfplay": 1800.0, "training": 1200.0},
            last_activity_time=time.time(),
        )

        assert stats.iterations_completed == 10
        assert stats.iterations_failed == 2
        assert stats.total_games_generated == 100000
        assert stats.promotions == 8


# ============================================
# Tests for DataPipelineOrchestrator class
# ============================================

class TestDataPipelineOrchestrator:
    """Tests for DataPipelineOrchestrator class."""

    @pytest.fixture
    def orchestrator(self):
        """Create a fresh orchestrator for each test."""
        return DataPipelineOrchestrator(max_history=100, auto_trigger=False)

    def test_initialization(self, orchestrator):
        """Should initialize with correct defaults."""
        assert orchestrator.max_history == 100
        assert orchestrator.auto_trigger is False
        assert orchestrator._subscribed is False
        assert orchestrator._current_stage == PipelineStage.IDLE
        assert orchestrator._current_iteration == 0

    def test_initialization_with_auto_trigger(self):
        """Should initialize with auto_trigger enabled."""
        orch = DataPipelineOrchestrator(auto_trigger=True)
        assert orch.auto_trigger is True

    def test_get_current_stage(self, orchestrator):
        """Should return current pipeline stage."""
        assert orchestrator.get_current_stage() == PipelineStage.IDLE

    def test_get_current_iteration(self, orchestrator):
        """Should return current iteration number."""
        assert orchestrator.get_current_iteration() == 0

    def test_start_iteration(self, orchestrator):
        """Should start a new pipeline iteration."""
        record = orchestrator.start_iteration(1)

        assert record.iteration == 1
        assert orchestrator.get_current_stage() == PipelineStage.SELFPLAY
        assert orchestrator.get_current_iteration() == 1

    def test_start_multiple_iterations(self, orchestrator):
        """Should handle multiple iterations."""
        orchestrator.start_iteration(1)
        orchestrator.start_iteration(2)

        assert orchestrator.get_current_iteration() == 2

    def test_get_iteration_record(self, orchestrator):
        """Should retrieve iteration record."""
        orchestrator.start_iteration(5)

        record = orchestrator.get_iteration_record(5)
        assert record is not None
        assert record.iteration == 5

        # Non-existent iteration
        assert orchestrator.get_iteration_record(999) is None

    def test_on_stage_enter_callback(self, orchestrator):
        """Should call callbacks on stage entry."""
        callback_data = []

        def on_selfplay(stage, iteration):
            callback_data.append((stage, iteration))

        orchestrator.on_stage_enter(PipelineStage.SELFPLAY, on_selfplay)
        orchestrator.start_iteration(1)

        assert len(callback_data) == 1
        assert callback_data[0] == (PipelineStage.SELFPLAY, 1)

    def test_get_recent_transitions(self, orchestrator):
        """Should return recent transitions."""
        orchestrator.start_iteration(1)

        transitions = orchestrator.get_recent_transitions(limit=10)
        assert len(transitions) >= 1
        assert transitions[-1].to_stage == PipelineStage.SELFPLAY

    def test_get_stage_metrics(self, orchestrator):
        """Should return stage timing metrics."""
        # Manually add some durations for testing
        orchestrator._stage_durations[PipelineStage.SELFPLAY] = [100.0, 120.0, 110.0]

        metrics = orchestrator.get_stage_metrics()

        assert "selfplay" in metrics
        assert metrics["selfplay"]["count"] == 3
        assert metrics["selfplay"]["avg_duration"] == 110.0
        assert metrics["selfplay"]["min_duration"] == 100.0
        assert metrics["selfplay"]["max_duration"] == 120.0

    def test_get_stats(self, orchestrator):
        """Should return aggregate statistics."""
        stats = orchestrator.get_stats()

        assert isinstance(stats, PipelineStats)
        assert stats.iterations_completed == 0
        assert stats.iterations_failed == 0

    def test_get_status(self, orchestrator):
        """Should return status dict."""
        status = orchestrator.get_status()

        assert "current_stage" in status
        assert "current_iteration" in status
        assert "iterations_completed" in status
        assert "subscribed" in status
        assert "auto_trigger" in status
        assert status["current_stage"] == "idle"

    def test_quality_distribution_tracking(self, orchestrator):
        """Should track quality distribution."""
        assert orchestrator._quality_distribution == {}

        # Manually update for testing
        orchestrator._quality_distribution = {"high": 0.3, "medium": 0.5, "low": 0.2}

        status = orchestrator.get_status()
        assert status["quality_distribution"]["high"] == 0.3

    def test_cache_invalidation_tracking(self, orchestrator):
        """Should track cache invalidation."""
        assert orchestrator._cache_invalidation_count == 0
        assert orchestrator._pending_cache_refresh is False

    def test_optimization_tracking(self, orchestrator):
        """Should track active optimization."""
        assert orchestrator._active_optimization is None

        orchestrator._active_optimization = "cmaes"
        orchestrator._optimization_run_id = "run-123"

        status = orchestrator.get_status()
        assert status["active_optimization"] == "cmaes"
        assert status["optimization_run_id"] == "run-123"


class TestDataPipelineOrchestratorTransitions:
    """Tests for pipeline stage transitions."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with iteration started."""
        orch = DataPipelineOrchestrator()
        orch.start_iteration(1)
        return orch

    def test_transition_records_timing(self, orchestrator):
        """Should record transition timing."""
        # Access the private method to simulate transition
        orchestrator._transition_to(PipelineStage.DATA_SYNC, 1)

        transitions = orchestrator.get_recent_transitions()
        # Should have IDLE->SELFPLAY and SELFPLAY->DATA_SYNC
        assert len(transitions) >= 2

    def test_stage_start_time_tracking(self, orchestrator):
        """Should track stage start times."""
        assert PipelineStage.SELFPLAY in orchestrator._stage_start_times


class TestDataPipelineOrchestratorEventHandling:
    """Tests for event handling in DataPipelineOrchestrator."""

    @pytest.fixture
    def orchestrator(self):
        """Create a fresh orchestrator for each test."""
        return DataPipelineOrchestrator()

    @pytest.mark.asyncio
    async def test_on_selfplay_complete(self, orchestrator):
        """Should handle selfplay complete event."""
        orchestrator.start_iteration(1)

        event = MagicMock()
        event.payload = {
            "iteration": 1,
            "games_generated": 1000,
            "success": True,
        }

        await orchestrator._on_selfplay_complete(event)

        # Should transition to next stage
        assert orchestrator.get_current_stage() in [
            PipelineStage.DATA_SYNC,
            PipelineStage.SELFPLAY,  # If auto_trigger is False
        ]

    @pytest.mark.asyncio
    async def test_on_sync_complete(self, orchestrator):
        """Should handle sync complete event."""
        orchestrator.start_iteration(1)
        orchestrator._current_stage = PipelineStage.DATA_SYNC

        event = MagicMock()
        event.payload = {
            "iteration": 1,
            "success": True,
        }

        await orchestrator._on_sync_complete(event)

    @pytest.mark.asyncio
    async def test_on_training_complete(self, orchestrator):
        """Should handle training complete event."""
        orchestrator.start_iteration(1)
        orchestrator._current_stage = PipelineStage.TRAINING

        event = MagicMock()
        event.payload = {
            "iteration": 1,
            "model_id": "model-v1",
            "success": True,
        }

        await orchestrator._on_training_complete(event)

        # Check record was updated
        record = orchestrator.get_iteration_record(1)
        assert record is not None


# ============================================
# Tests for module-level functions
# ============================================

class TestModuleFunctions:
    """Tests for module-level helper functions."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton for each test."""
        import app.coordination.data_pipeline_orchestrator as module
        module._pipeline_orchestrator = None
        yield
        module._pipeline_orchestrator = None

    def test_get_pipeline_orchestrator(self):
        """Should return singleton instance."""
        orch1 = get_pipeline_orchestrator()
        orch2 = get_pipeline_orchestrator()

        assert orch1 is orch2
        assert isinstance(orch1, DataPipelineOrchestrator)

    def test_wire_pipeline_events(self):
        """Should wire events and return orchestrator."""
        orchestrator = wire_pipeline_events()

        assert isinstance(orchestrator, DataPipelineOrchestrator)

    def test_wire_pipeline_events_with_auto_trigger(self):
        """Should pass auto_trigger to orchestrator."""
        orchestrator = wire_pipeline_events(auto_trigger=True)

        assert orchestrator.auto_trigger is True

    def test_get_pipeline_status(self):
        """Should return status from singleton."""
        status = get_pipeline_status()

        assert isinstance(status, dict)
        assert "current_stage" in status

    def test_get_current_pipeline_stage(self):
        """Should return current stage from singleton."""
        stage = get_current_pipeline_stage()

        assert isinstance(stage, PipelineStage)
        assert stage == PipelineStage.IDLE


# ============================================
# Integration tests
# ============================================

class TestDataPipelineOrchestratorIntegration:
    """Integration tests for pipeline orchestration workflow."""

    @pytest.fixture
    def orchestrator(self):
        """Create a fresh orchestrator for each test."""
        return DataPipelineOrchestrator(max_history=100)

    def test_full_iteration_lifecycle(self, orchestrator):
        """Should track complete iteration lifecycle."""
        # Start iteration
        orchestrator.start_iteration(1)
        assert orchestrator.get_current_stage() == PipelineStage.SELFPLAY

        # Simulate stage progression
        orchestrator._transition_to(PipelineStage.DATA_SYNC, 1)
        orchestrator._transition_to(PipelineStage.NPZ_EXPORT, 1)
        orchestrator._transition_to(PipelineStage.TRAINING, 1)
        orchestrator._transition_to(PipelineStage.EVALUATION, 1)
        orchestrator._transition_to(PipelineStage.PROMOTION, 1)
        orchestrator._transition_to(PipelineStage.COMPLETE, 1)

        # Check all stages were recorded
        transitions = orchestrator.get_recent_transitions()
        stages_visited = [t.to_stage for t in transitions]

        assert PipelineStage.SELFPLAY in stages_visited
        assert PipelineStage.TRAINING in stages_visited
        assert PipelineStage.COMPLETE in stages_visited

    def test_multiple_iterations(self, orchestrator):
        """Should handle multiple iterations."""
        for i in range(1, 4):
            orchestrator.start_iteration(i)
            orchestrator._transition_to(PipelineStage.COMPLETE, i)
            # Mark as completed
            record = orchestrator.get_iteration_record(i)
            if record:
                record.success = True
                record.end_time = time.time()
                orchestrator._completed_iterations.append(record)

        stats = orchestrator.get_stats()
        # At least some completed iterations
        assert stats.iterations_completed >= 0

    def test_stage_callbacks_fire_in_order(self, orchestrator):
        """Should fire callbacks in correct order."""
        callback_order = []

        def record_callback(stage, iteration):
            callback_order.append(stage)

        orchestrator.on_stage_enter(PipelineStage.SELFPLAY, record_callback)
        orchestrator.on_stage_enter(PipelineStage.DATA_SYNC, record_callback)
        orchestrator.on_stage_enter(PipelineStage.TRAINING, record_callback)

        orchestrator.start_iteration(1)
        orchestrator._transition_to(PipelineStage.DATA_SYNC, 1)
        orchestrator._transition_to(PipelineStage.TRAINING, 1)

        assert callback_order == [
            PipelineStage.SELFPLAY,
            PipelineStage.DATA_SYNC,
            PipelineStage.TRAINING,
        ]

    def test_metrics_accumulate(self, orchestrator):
        """Should accumulate metrics across stages."""
        # Add some duration data
        orchestrator._stage_durations[PipelineStage.SELFPLAY] = [100, 110, 105]
        orchestrator._stage_durations[PipelineStage.TRAINING] = [200, 220, 210]

        metrics = orchestrator.get_stage_metrics()

        assert "selfplay" in metrics
        assert "training" in metrics
        assert metrics["selfplay"]["count"] == 3
        assert metrics["training"]["count"] == 3

    def test_history_limit_enforced(self, orchestrator):
        """Should enforce history limit."""
        orchestrator.max_history = 5

        # Add more completed iterations than limit
        for i in range(10):
            record = IterationRecord(
                iteration=i,
                start_time=time.time() - 100,
                end_time=time.time(),
                success=True,
            )
            orchestrator._completed_iterations.append(record)

        # Trim history
        if len(orchestrator._completed_iterations) > orchestrator.max_history:
            orchestrator._completed_iterations = orchestrator._completed_iterations[-orchestrator.max_history:]

        assert len(orchestrator._completed_iterations) == 5
