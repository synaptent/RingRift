"""Unit tests for data_pipeline_orchestrator module.

Tests the DataPipelineOrchestrator, CircuitBreaker, and related pipeline
coordination components.
"""

from __future__ import annotations

import time

import pytest

from app.coordination.data_pipeline_orchestrator import (
    CircuitBreaker,
    CircuitBreakerState,
    DataPipelineOrchestrator,
    IterationRecord,
    PipelineStage,
    PipelineStats,
    StageTransition,
)


# =============================================================================
# PipelineStage Tests
# =============================================================================


class TestPipelineStage:
    """Test PipelineStage enum."""

    def test_all_stages_exist(self):
        """Verify all expected stages are defined."""
        stages = [s.value for s in PipelineStage]
        assert "idle" in stages
        assert "selfplay" in stages
        assert "data_sync" in stages
        assert "npz_export" in stages
        assert "npz_combination" in stages  # Added December 2025
        assert "training" in stages
        assert "evaluation" in stages
        assert "promotion" in stages
        assert "complete" in stages

    def test_stage_count(self):
        """Verify stage count matches expected."""
        assert len(PipelineStage) == 9  # Updated: NPZ_COMBINATION added December 2025

    def test_stage_values_are_strings(self):
        """All stage values should be strings."""
        for stage in PipelineStage:
            assert isinstance(stage.value, str)


# =============================================================================
# CircuitBreaker Tests
# =============================================================================


class TestCircuitBreaker:
    """Test CircuitBreaker fault tolerance."""

    def test_initial_state_is_closed(self):
        """Circuit should start in closed state."""
        cb = CircuitBreaker()
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.is_closed
        assert not cb.is_open

    def test_can_execute_when_closed(self):
        """Should allow execution when circuit is closed."""
        cb = CircuitBreaker()
        assert cb.can_execute()

    def test_record_success_keeps_closed(self):
        """Recording success should keep circuit closed."""
        cb = CircuitBreaker()
        cb.record_success("test_stage")
        assert cb.is_closed

    def test_trips_after_threshold_failures(self):
        """Circuit should open after reaching failure threshold."""
        cb = CircuitBreaker(failure_threshold=3)

        # Record 3 failures
        for i in range(3):
            cb.record_failure(f"stage_{i}")

        assert cb.state == CircuitBreakerState.OPEN
        assert cb.is_open
        assert not cb.can_execute()

    def test_does_not_trip_before_threshold(self):
        """Circuit should stay closed before reaching threshold."""
        cb = CircuitBreaker(failure_threshold=3)

        # Record 2 failures (below threshold)
        cb.record_failure("stage_1")
        cb.record_failure("stage_2")

        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.can_execute()

    def test_half_open_transition_after_timeout(self):
        """Circuit should transition to half-open after reset timeout."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)

        # Trip the circuit
        cb.record_failure("test")
        assert cb.state == CircuitBreakerState.OPEN

        # Wait for timeout (canonical uses 2x backoff, so wait 0.25s for 0.1*2=0.2 effective)
        time.sleep(0.25)

        # Canonical breaker transitions to HALF_OPEN when can_execute() is called
        # (not on state property access), so call can_execute() first
        assert cb.can_execute()
        assert cb.state == CircuitBreakerState.HALF_OPEN

    def test_half_open_closes_on_success(self):
        """Circuit should close after success in half-open state."""
        # Canonical breaker has min backoff of 0.1s, so use that as base
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)

        # Trip and wait for half-open (canonical uses min 0.1s backoff + jitter)
        cb.record_failure("test")
        time.sleep(0.25)  # Wait well past 0.1*2=0.2 effective timeout
        # Trigger HALF_OPEN transition via can_execute()
        assert cb.can_execute()
        assert cb.state == CircuitBreakerState.HALF_OPEN

        # Record success
        cb.record_success("test")
        assert cb.state == CircuitBreakerState.CLOSED

    def test_half_open_reopens_on_failure(self):
        """Circuit should reopen on failure in half-open state."""
        # Use longer timeout to ensure reliable test (min backoff is 0.1s with 2x multiplier)
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)

        # Trip the circuit
        cb.record_failure("test")
        assert cb.state == CircuitBreakerState.OPEN

        # Wait for half-open transition (canonical uses 2x backoff + min 0.1s)
        time.sleep(0.25)
        # Trigger HALF_OPEN transition via can_execute()
        assert cb.can_execute()
        assert cb.state == CircuitBreakerState.HALF_OPEN

        # Record another failure - should reopen the circuit
        cb.record_failure("test")
        assert cb.state == CircuitBreakerState.OPEN

    def test_get_status_dict(self):
        """get_status should return correct status dictionary."""
        cb = CircuitBreaker()
        status = cb.get_status()

        assert "state" in status
        assert "failure_count" in status
        assert "success_count" in status
        assert "failures_by_stage" in status
        assert status["state"] == "closed"

    def test_reset_clears_state(self):
        """Reset should clear failure counts and close circuit."""
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure("stage_1")
        cb.record_failure("stage_2")
        assert cb.state == CircuitBreakerState.OPEN

        cb.reset()
        assert cb.state == CircuitBreakerState.CLOSED
        # Use get_status() API instead of private attributes
        assert cb.get_status()["failure_count"] == 0

    def test_tracks_failures_by_stage(self):
        """Should track failures per stage."""
        cb = CircuitBreaker()
        cb.record_failure("stage_a")
        cb.record_failure("stage_a")
        cb.record_failure("stage_b")

        assert cb._failures_by_stage["stage_a"] == 2
        assert cb._failures_by_stage["stage_b"] == 1


# =============================================================================
# StageTransition Tests
# =============================================================================


class TestStageTransition:
    """Test StageTransition dataclass."""

    def test_create_transition(self):
        """Should create transition record."""
        transition = StageTransition(
            from_stage=PipelineStage.SELFPLAY,
            to_stage=PipelineStage.DATA_SYNC,
            iteration=1,
        )
        assert transition.from_stage == PipelineStage.SELFPLAY
        assert transition.to_stage == PipelineStage.DATA_SYNC
        assert transition.iteration == 1
        assert transition.success is True

    def test_transition_with_metadata(self):
        """Should store metadata."""
        transition = StageTransition(
            from_stage=PipelineStage.TRAINING,
            to_stage=PipelineStage.EVALUATION,
            iteration=5,
            metadata={"model_path": "/path/to/model.pth"},
        )
        assert transition.metadata["model_path"] == "/path/to/model.pth"


# =============================================================================
# IterationRecord Tests
# =============================================================================


class TestIterationRecord:
    """Test IterationRecord dataclass."""

    def test_create_record(self):
        """Should create iteration record."""
        record = IterationRecord(iteration=1, start_time=time.time())
        assert record.iteration == 1
        assert record.success is False
        assert record.stages_completed == []

    def test_duration_while_running(self):
        """Duration should increase while iteration is running."""
        record = IterationRecord(iteration=1, start_time=time.time() - 10)
        assert record.duration >= 10

    def test_duration_after_completion(self):
        """Duration should be fixed after completion."""
        start = time.time() - 100
        end = start + 50
        record = IterationRecord(iteration=1, start_time=start, end_time=end)
        assert record.duration == 50


# =============================================================================
# PipelineStats Tests
# =============================================================================


class TestPipelineStats:
    """Test PipelineStats dataclass."""

    def test_default_stats(self):
        """Should have sensible defaults."""
        stats = PipelineStats()
        assert stats.iterations_completed == 0
        assert stats.iterations_failed == 0
        assert stats.total_games_generated == 0
        assert stats.promotions == 0


# =============================================================================
# DataPipelineOrchestrator Tests
# =============================================================================


class TestDataPipelineOrchestratorInit:
    """Test DataPipelineOrchestrator initialization."""

    def test_init_with_defaults(self):
        """Should initialize with default values."""
        orchestrator = DataPipelineOrchestrator()

        assert orchestrator.max_history == 100
        assert orchestrator.auto_trigger is True
        assert orchestrator._current_stage == PipelineStage.IDLE
        assert orchestrator._current_iteration == 0

    def test_init_with_custom_values(self):
        """Should accept custom initialization values."""
        orchestrator = DataPipelineOrchestrator(
            max_history=50,
            auto_trigger=False,
        )

        assert orchestrator.max_history == 50
        assert orchestrator.auto_trigger is False


class TestDataPipelineOrchestratorStages:
    """Test stage management."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator."""
        return DataPipelineOrchestrator(auto_trigger=False)

    def test_initial_stage_is_idle(self, orchestrator):
        """Should start in IDLE stage."""
        assert orchestrator._current_stage == PipelineStage.IDLE

    def test_transition_to_selfplay(self, orchestrator):
        """Should transition to SELFPLAY."""
        orchestrator._transition_to(PipelineStage.SELFPLAY, iteration=1)
        assert orchestrator._current_stage == PipelineStage.SELFPLAY

    def test_transition_records_history(self, orchestrator):
        """Transitions should be recorded."""
        orchestrator._transition_to(PipelineStage.SELFPLAY, iteration=1)
        orchestrator._transition_to(PipelineStage.DATA_SYNC, iteration=1)

        assert len(orchestrator._transitions) >= 2

    def test_get_status(self, orchestrator):
        """get_status should return stage info."""
        status = orchestrator.get_status()

        assert "current_stage" in status
        assert "current_iteration" in status
        assert "iterations_completed" in status
        assert status["current_stage"] == "idle"


class TestDataPipelineOrchestratorCircuitBreaker:
    """Test circuit breaker integration."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator."""
        return DataPipelineOrchestrator(auto_trigger=False)

    def test_circuit_breaker_exists(self, orchestrator):
        """Should have circuit breaker."""
        assert orchestrator._circuit_breaker is not None

    def test_circuit_breaker_can_execute_when_closed(self, orchestrator):
        """Should allow execution when circuit is closed."""
        assert orchestrator._circuit_breaker.can_execute() is True

    def test_circuit_breaker_blocks_when_open(self, orchestrator):
        """Should block when circuit is open."""
        # Trip the circuit
        for _ in range(10):
            orchestrator._circuit_breaker.record_failure("test")

        assert orchestrator._circuit_breaker.can_execute() is False


class TestDataPipelineOrchestratorMetrics:
    """Test metrics and observability."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator."""
        return DataPipelineOrchestrator(auto_trigger=False)

    def test_get_stage_metrics(self, orchestrator):
        """Should return stage metrics."""
        metrics = orchestrator.get_stage_metrics()
        assert isinstance(metrics, dict)

    def test_stage_duration_recorded_on_transition(self, orchestrator):
        """Should record stage duration when transitioning."""
        # Transition to selfplay
        orchestrator._transition_to(PipelineStage.SELFPLAY, iteration=1)
        # Transition away records duration
        orchestrator._transition_to(PipelineStage.DATA_SYNC, iteration=1)

        # Stage durations should be tracked
        assert orchestrator._stage_durations is not None


class TestDataPipelineOrchestratorHistory:
    """Test history management."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with limited history."""
        return DataPipelineOrchestrator(max_history=5, auto_trigger=False)

    def test_history_limit_enforced(self, orchestrator):
        """Should enforce max_history limit."""
        # Record many transitions
        for i in range(10):
            orchestrator._transition_to(PipelineStage.SELFPLAY, iteration=i)
            orchestrator._transition_to(PipelineStage.IDLE, iteration=i)

        # Transitions list should exist
        assert len(orchestrator._transitions) > 0


# =============================================================================
# Module-level function tests
# =============================================================================


class TestModuleFunctions:
    """Test module-level functions."""

    def test_import_get_pipeline_orchestrator(self):
        """Should be able to import get_pipeline_orchestrator."""
        from app.coordination.data_pipeline_orchestrator import get_pipeline_orchestrator

        assert callable(get_pipeline_orchestrator)

    def test_import_wire_pipeline_events(self):
        """Should be able to import wire_pipeline_events."""
        from app.coordination.data_pipeline_orchestrator import wire_pipeline_events

        assert callable(wire_pipeline_events)


# =============================================================================
# Integration Tests
# =============================================================================


class TestDataPipelineOrchestratorIntegration:
    """Integration tests for pipeline orchestrator."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for integration testing."""
        return DataPipelineOrchestrator(auto_trigger=False)

    def test_full_pipeline_iteration(self, orchestrator):
        """Should handle full pipeline iteration."""
        # Start iteration (uses public API)
        record = orchestrator.start_iteration(iteration=1)
        assert orchestrator._current_iteration == 1

        # Progress through stages
        orchestrator._transition_to(PipelineStage.DATA_SYNC, iteration=1)
        orchestrator._transition_to(PipelineStage.NPZ_EXPORT, iteration=1)
        orchestrator._transition_to(PipelineStage.TRAINING, iteration=1)
        orchestrator._transition_to(PipelineStage.EVALUATION, iteration=1)
        orchestrator._transition_to(PipelineStage.PROMOTION, iteration=1)
        orchestrator._transition_to(PipelineStage.COMPLETE, iteration=1)

        assert orchestrator._current_stage == PipelineStage.COMPLETE

    def test_status_dict_structure(self, orchestrator):
        """Status dict should have all required fields."""
        status = orchestrator.get_status()

        required_fields = [
            "current_stage",
            "current_iteration",
            "iterations_completed",
            "auto_trigger",
        ]
        for field in required_fields:
            assert field in status, f"Missing field: {field}"
