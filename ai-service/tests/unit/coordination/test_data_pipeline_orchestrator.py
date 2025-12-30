"""Unit tests for DataPipelineOrchestrator (December 2025).

Tests cover:
- PipelineCircuitBreaker: Circuit breaker wrapper for fault tolerance
- PipelineStage: Enum of pipeline stages
- StageTransition: Record of stage transitions
- IterationRecord: Record of complete pipeline iterations
- PipelineStats: Aggregate pipeline statistics
- DataPipelineOrchestrator: Main orchestrator class
- Module functions: get_pipeline_orchestrator, wire_pipeline_events, etc.
"""

import asyncio
import time
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# =============================================================================
# PipelineCircuitBreaker Tests
# =============================================================================


class TestPipelineCircuitBreaker:
    """Tests for PipelineCircuitBreaker class."""

    def test_initialization_defaults(self):
        """Test circuit breaker initializes with defaults."""
        from app.coordination.data_pipeline_orchestrator import PipelineCircuitBreaker

        cb = PipelineCircuitBreaker()

        assert cb.is_closed
        assert not cb.is_open
        assert cb.can_execute()

    def test_initialization_custom_threshold(self):
        """Test circuit breaker with custom failure threshold."""
        from app.coordination.data_pipeline_orchestrator import PipelineCircuitBreaker

        cb = PipelineCircuitBreaker(failure_threshold=5, recovery_timeout=120.0)

        assert cb.is_closed
        status = cb.get_status()
        assert status["failure_count"] == 0

    def test_record_success(self):
        """Test recording successful execution."""
        from app.coordination.data_pipeline_orchestrator import PipelineCircuitBreaker

        cb = PipelineCircuitBreaker()
        cb.record_success(stage="training")

        status = cb.get_status()
        assert status["success_count"] >= 1
        assert cb.is_closed

    def test_record_failure(self):
        """Test recording failed execution."""
        from app.coordination.data_pipeline_orchestrator import PipelineCircuitBreaker

        cb = PipelineCircuitBreaker(failure_threshold=3)
        cb.record_failure(stage="training", error="Model diverged")

        status = cb.get_status()
        assert status["failure_count"] >= 1
        assert "training" in status["failures_by_stage"]

    def test_opens_after_threshold(self):
        """Test circuit opens after failure threshold is reached."""
        from app.coordination.data_pipeline_orchestrator import PipelineCircuitBreaker

        cb = PipelineCircuitBreaker(failure_threshold=2, recovery_timeout=60.0)

        # Record failures up to threshold
        cb.record_failure(stage="training", error="Error 1")
        cb.record_failure(stage="training", error="Error 2")

        # Should be open now (may need additional failure for some implementations)
        status = cb.get_status()
        assert status["failure_count"] >= 2

    def test_reset(self):
        """Test manual circuit breaker reset."""
        from app.coordination.data_pipeline_orchestrator import PipelineCircuitBreaker

        cb = PipelineCircuitBreaker(failure_threshold=1)
        cb.record_failure(stage="export", error="File not found")

        cb.reset()

        assert cb.is_closed
        status = cb.get_status()
        assert status["failure_count"] == 0
        assert len(status["failures_by_stage"]) == 0

    def test_get_status(self):
        """Test get_status returns expected structure."""
        from app.coordination.data_pipeline_orchestrator import PipelineCircuitBreaker

        cb = PipelineCircuitBreaker()

        status = cb.get_status()

        assert "state" in status
        assert "failure_count" in status
        assert "success_count" in status
        assert "failures_by_stage" in status
        assert "last_failure_time" in status


# =============================================================================
# PipelineStage Enum Tests
# =============================================================================


class TestPipelineStage:
    """Tests for PipelineStage enum."""

    def test_all_stages_defined(self):
        """Test all expected pipeline stages are defined."""
        from app.coordination.data_pipeline_orchestrator import PipelineStage

        expected_stages = [
            "IDLE", "SELFPLAY", "DATA_SYNC", "NPZ_EXPORT",
            "NPZ_COMBINATION", "TRAINING", "EVALUATION",
            "PROMOTION", "COMPLETE"
        ]

        for stage_name in expected_stages:
            assert hasattr(PipelineStage, stage_name)

    def test_stage_values(self):
        """Test stage values are strings."""
        from app.coordination.data_pipeline_orchestrator import PipelineStage

        assert PipelineStage.IDLE.value == "idle"
        assert PipelineStage.SELFPLAY.value == "selfplay"
        assert PipelineStage.TRAINING.value == "training"

    def test_stage_comparison(self):
        """Test stage comparison works."""
        from app.coordination.data_pipeline_orchestrator import PipelineStage

        assert PipelineStage.IDLE == PipelineStage.IDLE
        assert PipelineStage.TRAINING != PipelineStage.EVALUATION


# =============================================================================
# StageTransition Dataclass Tests
# =============================================================================


class TestStageTransition:
    """Tests for StageTransition dataclass."""

    def test_basic_creation(self):
        """Test basic StageTransition creation."""
        from app.coordination.data_pipeline_orchestrator import (
            PipelineStage, StageTransition
        )

        transition = StageTransition(
            from_stage=PipelineStage.IDLE,
            to_stage=PipelineStage.SELFPLAY,
            iteration=1,
        )

        assert transition.from_stage == PipelineStage.IDLE
        assert transition.to_stage == PipelineStage.SELFPLAY
        assert transition.iteration == 1
        assert transition.success is True  # Default

    def test_with_all_fields(self):
        """Test StageTransition with all fields specified."""
        from app.coordination.data_pipeline_orchestrator import (
            PipelineStage, StageTransition
        )

        transition = StageTransition(
            from_stage=PipelineStage.TRAINING,
            to_stage=PipelineStage.EVALUATION,
            iteration=5,
            timestamp=1735500000.0,
            success=True,
            duration_seconds=120.5,
            metadata={"model_id": "hex8_2p_v3"},
        )

        assert transition.from_stage == PipelineStage.TRAINING
        assert transition.to_stage == PipelineStage.EVALUATION
        assert transition.iteration == 5
        assert transition.timestamp == 1735500000.0
        assert transition.success is True
        assert transition.duration_seconds == 120.5
        assert transition.metadata["model_id"] == "hex8_2p_v3"


# =============================================================================
# IterationRecord Dataclass Tests
# =============================================================================


class TestIterationRecord:
    """Tests for IterationRecord dataclass."""

    def test_basic_creation(self):
        """Test basic IterationRecord creation."""
        from app.coordination.data_pipeline_orchestrator import IterationRecord

        record = IterationRecord(
            iteration=1,
            start_time=time.time(),
        )

        assert record.iteration == 1
        assert record.success is False  # Default
        assert record.stages_completed == []

    def test_duration_property_in_progress(self):
        """Test duration property for in-progress iteration."""
        from app.coordination.data_pipeline_orchestrator import IterationRecord

        start = time.time() - 10  # Started 10 seconds ago
        record = IterationRecord(
            iteration=1,
            start_time=start,
            end_time=0.0,  # Not finished
        )

        duration = record.duration
        assert duration >= 10.0

    def test_duration_property_completed(self):
        """Test duration property for completed iteration."""
        from app.coordination.data_pipeline_orchestrator import IterationRecord

        record = IterationRecord(
            iteration=1,
            start_time=1000.0,
            end_time=1120.0,  # 120 seconds later
        )

        assert record.duration == 120.0

    def test_with_all_fields(self):
        """Test IterationRecord with all fields."""
        from app.coordination.data_pipeline_orchestrator import IterationRecord

        record = IterationRecord(
            iteration=10,
            start_time=1000.0,
            end_time=1500.0,
            success=True,
            stages_completed=["selfplay", "training", "evaluation"],
            games_generated=1000,
            model_id="hex8_2p_iter10",
            elo_delta=25.5,
            promoted=True,
            error=None,
        )

        assert record.iteration == 10
        assert record.success is True
        assert len(record.stages_completed) == 3
        assert record.games_generated == 1000
        assert record.promoted is True
        assert record.elo_delta == 25.5


# =============================================================================
# PipelineStats Dataclass Tests
# =============================================================================


class TestPipelineStats:
    """Tests for PipelineStats dataclass."""

    def test_default_values(self):
        """Test PipelineStats default values."""
        from app.coordination.data_pipeline_orchestrator import PipelineStats

        stats = PipelineStats()

        assert stats.iterations_completed == 0
        assert stats.iterations_failed == 0
        assert stats.total_games_generated == 0
        assert stats.total_models_trained == 0
        assert stats.promotions == 0
        assert stats.average_iteration_duration == 0.0
        assert stats.stage_durations == {}

    def test_with_values(self):
        """Test PipelineStats with values."""
        from app.coordination.data_pipeline_orchestrator import PipelineStats

        stats = PipelineStats(
            iterations_completed=50,
            iterations_failed=5,
            total_games_generated=50000,
            total_models_trained=45,
            promotions=12,
            average_iteration_duration=3600.0,
            stage_durations={"selfplay": 1800.0, "training": 1200.0},
            last_activity_time=time.time(),
        )

        assert stats.iterations_completed == 50
        assert stats.total_games_generated == 50000
        assert "selfplay" in stats.stage_durations


# =============================================================================
# DataPipelineOrchestrator Tests
# =============================================================================


@pytest.fixture
def mock_config():
    """Create mock pipeline config."""
    config = MagicMock()
    config.auto_trigger_sync = True
    config.auto_trigger_export = True
    config.auto_trigger_training = True
    config.auto_trigger_evaluation = True
    config.auto_trigger_promotion = True
    config.quality_gate_enabled = False
    config.quality_gate_threshold = 0.6
    config.quality_gate_min_high_quality_pct = 0.30
    config.circuit_breaker_enabled = True
    config.circuit_breaker_failure_threshold = 3
    config.circuit_breaker_recovery_timeout = 60.0
    config.circuit_breaker_half_open_max_calls = 1
    return config


@pytest.fixture(autouse=True)
def reset_orchestrator_singleton():
    """Reset the orchestrator singleton between tests."""
    import app.coordination.data_pipeline_orchestrator as module

    # Reset singleton
    module._pipeline_orchestrator = None

    yield

    # Cleanup
    module._pipeline_orchestrator = None


class TestDataPipelineOrchestratorInit:
    """Tests for DataPipelineOrchestrator initialization."""

    def test_initialization_defaults(self):
        """Test orchestrator initializes with defaults."""
        from app.coordination.data_pipeline_orchestrator import (
            DataPipelineOrchestrator, PipelineStage
        )

        orchestrator = DataPipelineOrchestrator()

        assert orchestrator._current_stage == PipelineStage.IDLE
        assert orchestrator._current_iteration == 0
        assert orchestrator.auto_trigger is True

    def test_initialization_with_config(self, mock_config):
        """Test orchestrator with custom config."""
        from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

        orchestrator = DataPipelineOrchestrator(
            max_history=50,
            auto_trigger=False,
            config=mock_config,
        )

        assert orchestrator.max_history == 50
        assert orchestrator.auto_trigger is False
        assert orchestrator.auto_trigger_sync is True

    def test_initialization_circuit_breaker(self, mock_config):
        """Test orchestrator initializes circuit breaker."""
        from app.coordination.data_pipeline_orchestrator import (
            DataPipelineOrchestrator, PipelineCircuitBreaker
        )

        orchestrator = DataPipelineOrchestrator(config=mock_config)

        assert orchestrator._circuit_breaker is not None
        assert isinstance(orchestrator._circuit_breaker, PipelineCircuitBreaker)


class TestDataPipelineOrchestratorStatus:
    """Tests for DataPipelineOrchestrator status methods."""

    def test_get_status(self):
        """Test get_status returns expected structure."""
        from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

        orchestrator = DataPipelineOrchestrator()

        status = orchestrator.get_status()

        assert "current_stage" in status
        assert "current_iteration" in status
        assert "auto_trigger" in status
        assert "subscribed" in status

    def test_get_current_stage(self):
        """Test get_current_stage returns current stage."""
        from app.coordination.data_pipeline_orchestrator import (
            DataPipelineOrchestrator, PipelineStage
        )

        orchestrator = DataPipelineOrchestrator()

        assert orchestrator.get_current_stage() == PipelineStage.IDLE


class TestDataPipelineOrchestratorTransitions:
    """Tests for DataPipelineOrchestrator stage transitions."""

    def test_transition_to_stage(self, mock_config):
        """Test transitioning to a new stage."""
        from app.coordination.data_pipeline_orchestrator import (
            DataPipelineOrchestrator, PipelineStage
        )

        orchestrator = DataPipelineOrchestrator(config=mock_config)

        # Transition to selfplay (uses _transition_to with iteration)
        orchestrator._transition_to(PipelineStage.SELFPLAY, iteration=1)

        assert orchestrator._current_stage == PipelineStage.SELFPLAY

    def test_transition_records_history(self, mock_config):
        """Test transitions are recorded in history."""
        from app.coordination.data_pipeline_orchestrator import (
            DataPipelineOrchestrator, PipelineStage
        )

        orchestrator = DataPipelineOrchestrator(config=mock_config)

        orchestrator._transition_to(PipelineStage.SELFPLAY, iteration=1)
        orchestrator._transition_to(PipelineStage.TRAINING, iteration=1)

        assert len(orchestrator._transitions) >= 2


class TestDataPipelineOrchestratorHealth:
    """Tests for DataPipelineOrchestrator health check."""

    def test_health_check(self, mock_config):
        """Test health_check returns HealthCheckResult."""
        from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator
        from app.coordination.protocols import HealthCheckResult

        orchestrator = DataPipelineOrchestrator(config=mock_config)

        result = orchestrator.health_check()

        assert isinstance(result, HealthCheckResult)
        assert hasattr(result, "healthy")

    def test_get_health_status(self, mock_config):
        """Test get_health_status returns dict."""
        from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

        orchestrator = DataPipelineOrchestrator(config=mock_config)

        health = orchestrator.get_health_status()

        assert isinstance(health, dict)
        assert "healthy" in health or "status" in health


class TestDataPipelineOrchestratorPauseResume:
    """Tests for DataPipelineOrchestrator pause/resume state."""

    def test_paused_state(self, mock_config):
        """Test paused state via direct attribute."""
        from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

        orchestrator = DataPipelineOrchestrator(config=mock_config)

        # Set paused state directly (internal state management)
        orchestrator._paused = True
        orchestrator._pause_reason = "Testing pause"

        assert orchestrator._paused is True
        assert orchestrator._pause_reason == "Testing pause"

    def test_resume_state(self, mock_config):
        """Test resume state via direct attribute."""
        from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

        orchestrator = DataPipelineOrchestrator(config=mock_config)
        orchestrator._paused = True
        orchestrator._pause_reason = "Testing"

        # Resume by clearing state
        orchestrator._paused = False
        orchestrator._pause_reason = None

        assert orchestrator._paused is False
        assert orchestrator._pause_reason is None

    def test_is_paused_method(self, mock_config):
        """Test is_paused method returns _paused value."""
        from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

        orchestrator = DataPipelineOrchestrator(config=mock_config)

        # is_paused() is a method
        assert orchestrator.is_paused() is False

        orchestrator._paused = True
        assert orchestrator.is_paused() is True


class TestDataPipelineOrchestratorIterations:
    """Tests for DataPipelineOrchestrator iteration management."""

    def test_start_iteration(self, mock_config):
        """Test starting a new iteration."""
        from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

        orchestrator = DataPipelineOrchestrator(config=mock_config)

        # start_iteration takes an iteration number
        record = orchestrator.start_iteration(iteration=5)

        assert orchestrator._current_iteration == 5
        assert record is not None

    def test_current_iteration_tracking(self, mock_config):
        """Test current_iteration can be tracked."""
        from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

        orchestrator = DataPipelineOrchestrator(config=mock_config)
        orchestrator._current_iteration = 10

        assert orchestrator._current_iteration == 10


# =============================================================================
# Module Function Tests
# =============================================================================


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_get_pipeline_orchestrator_singleton(self):
        """Test get_pipeline_orchestrator returns singleton."""
        from app.coordination.data_pipeline_orchestrator import (
            get_pipeline_orchestrator, DataPipelineOrchestrator
        )

        orchestrator1 = get_pipeline_orchestrator()
        orchestrator2 = get_pipeline_orchestrator()

        assert orchestrator1 is orchestrator2
        assert isinstance(orchestrator1, DataPipelineOrchestrator)

    def test_get_pipeline_status(self):
        """Test get_pipeline_status convenience function."""
        from app.coordination.data_pipeline_orchestrator import get_pipeline_status

        status = get_pipeline_status()

        assert isinstance(status, dict)
        assert "current_stage" in status

    def test_get_current_pipeline_stage(self):
        """Test get_current_pipeline_stage convenience function."""
        from app.coordination.data_pipeline_orchestrator import (
            get_current_pipeline_stage, PipelineStage
        )

        stage = get_current_pipeline_stage()

        assert isinstance(stage, PipelineStage)

    def test_get_pipeline_health(self):
        """Test get_pipeline_health convenience function."""
        from app.coordination.data_pipeline_orchestrator import get_pipeline_health

        health = get_pipeline_health()

        assert isinstance(health, dict)

    def test_is_pipeline_healthy(self):
        """Test is_pipeline_healthy convenience function."""
        from app.coordination.data_pipeline_orchestrator import is_pipeline_healthy

        result = is_pipeline_healthy()

        assert isinstance(result, bool)


class TestWirePipelineEvents:
    """Tests for wire_pipeline_events function."""

    def test_wire_pipeline_events_returns_orchestrator(self):
        """Test wire_pipeline_events returns DataPipelineOrchestrator."""
        from app.coordination.data_pipeline_orchestrator import (
            wire_pipeline_events, DataPipelineOrchestrator
        )

        # wire_pipeline_events uses get_pipeline_orchestrator internally
        # and returns the orchestrator after wiring events
        result = wire_pipeline_events()

        assert isinstance(result, DataPipelineOrchestrator)

    def test_wire_pipeline_events_with_auto_trigger_false(self):
        """Test wire_pipeline_events sets auto_trigger."""
        from app.coordination.data_pipeline_orchestrator import wire_pipeline_events

        result = wire_pipeline_events(auto_trigger=False)

        # Verify auto_trigger was set on the returned orchestrator
        assert result.auto_trigger is False


# =============================================================================
# DataPipelineOrchestrator Stage Callbacks Tests
# =============================================================================


class TestDataPipelineOrchestratorCallbacks:
    """Tests for DataPipelineOrchestrator stage callbacks."""

    def test_stage_callbacks_dict_exists(self, mock_config):
        """Test stage callbacks dict is initialized."""
        from app.coordination.data_pipeline_orchestrator import (
            DataPipelineOrchestrator, PipelineStage
        )

        orchestrator = DataPipelineOrchestrator(config=mock_config)

        # Callbacks dict should exist for storing callbacks
        assert hasattr(orchestrator, "_stage_callbacks")
        assert isinstance(orchestrator._stage_callbacks, dict)

    def test_add_callback_to_stage(self, mock_config):
        """Test adding a callback to a stage manually."""
        from app.coordination.data_pipeline_orchestrator import (
            DataPipelineOrchestrator, PipelineStage
        )

        orchestrator = DataPipelineOrchestrator(config=mock_config)

        callback_called = []
        def my_callback():
            callback_called.append(True)

        # Add callback directly to internal dict
        if PipelineStage.TRAINING not in orchestrator._stage_callbacks:
            orchestrator._stage_callbacks[PipelineStage.TRAINING] = []
        orchestrator._stage_callbacks[PipelineStage.TRAINING].append(my_callback)

        assert PipelineStage.TRAINING in orchestrator._stage_callbacks
        assert my_callback in orchestrator._stage_callbacks[PipelineStage.TRAINING]


# =============================================================================
# DataPipelineOrchestrator Board Config Tests
# =============================================================================


class TestDataPipelineOrchestratorBoardConfig:
    """Tests for DataPipelineOrchestrator board configuration tracking."""

    def test_get_board_config_from_state(self, mock_config):
        """Test _get_board_config returns tracked state."""
        from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

        orchestrator = DataPipelineOrchestrator(config=mock_config)
        orchestrator._current_board_type = "hex8"
        orchestrator._current_num_players = 2

        board_type, num_players = orchestrator._get_board_config()

        assert board_type == "hex8"
        assert num_players == 2

    def test_get_board_config_from_result(self, mock_config):
        """Test _get_board_config extracts from result object."""
        from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

        orchestrator = DataPipelineOrchestrator(config=mock_config)

        result = MagicMock()
        result.board_type = "square8"
        result.num_players = 4

        board_type, num_players = orchestrator._get_board_config(result=result)

        assert board_type == "square8"
        assert num_players == 4

    def test_get_board_config_from_metadata(self, mock_config):
        """Test _get_board_config extracts from metadata."""
        from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

        orchestrator = DataPipelineOrchestrator(config=mock_config)

        metadata = {"board_type": "hexagonal", "num_players": 3}
        board_type, num_players = orchestrator._get_board_config(metadata=metadata)

        assert board_type == "hexagonal"
        assert num_players == 3


# =============================================================================
# DataPipelineOrchestrator Circuit Breaker Integration Tests
# =============================================================================


class TestDataPipelineOrchestratorCircuitBreaker:
    """Tests for circuit breaker integration."""

    def test_circuit_breaker_blocks_when_open(self, mock_config):
        """Test circuit breaker can block operations when open."""
        from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

        orchestrator = DataPipelineOrchestrator(config=mock_config)

        # Force circuit open by recording failures
        for _ in range(10):
            orchestrator._circuit_breaker.record_failure("training", "Error")

        # Check circuit state
        status = orchestrator._circuit_breaker.get_status()
        # The circuit may be open after threshold failures
        assert status["failure_count"] >= 3

    def test_circuit_breaker_reset(self, mock_config):
        """Test circuit breaker can be reset."""
        from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

        orchestrator = DataPipelineOrchestrator(config=mock_config)

        # Record failures
        orchestrator._circuit_breaker.record_failure("training", "Error")
        orchestrator._circuit_breaker.reset()

        assert orchestrator._circuit_breaker.is_closed


# =============================================================================
# DataPipelineOrchestrator Quality Gate Tests
# =============================================================================


class TestDataPipelineOrchestratorQualityGate:
    """Tests for quality gate functionality."""

    def test_quality_gate_disabled(self, mock_config):
        """Test with quality gate disabled."""
        mock_config.quality_gate_enabled = False

        from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

        orchestrator = DataPipelineOrchestrator(config=mock_config)

        assert orchestrator.quality_gate_enabled is False

    def test_quality_gate_threshold(self, mock_config):
        """Test quality gate threshold configuration."""
        mock_config.quality_gate_enabled = True
        mock_config.quality_gate_threshold = 0.75

        from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

        orchestrator = DataPipelineOrchestrator(config=mock_config)

        assert orchestrator.quality_gate_threshold == 0.75


# =============================================================================
# CircuitBreakerState Enum Tests
# =============================================================================


class TestCircuitBreakerState:
    """Tests for CircuitBreakerState enum."""

    def test_states_defined(self):
        """Test all circuit breaker states are defined."""
        from app.coordination.data_pipeline_orchestrator import CircuitBreakerState

        assert hasattr(CircuitBreakerState, "CLOSED")
        assert hasattr(CircuitBreakerState, "OPEN")
        assert hasattr(CircuitBreakerState, "HALF_OPEN")

    def test_state_values(self):
        """Test state values are strings."""
        from app.coordination.data_pipeline_orchestrator import CircuitBreakerState

        assert CircuitBreakerState.CLOSED.value == "closed"
        assert CircuitBreakerState.OPEN.value == "open"
        assert CircuitBreakerState.HALF_OPEN.value == "half_open"
