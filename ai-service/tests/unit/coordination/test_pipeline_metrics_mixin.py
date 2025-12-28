"""Unit tests for PipelineMetricsMixin - metrics, status, and health reporting.

December 2025: Tests for the metrics mixin extracted from DataPipelineOrchestrator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Mock PipelineStage Enum
# =============================================================================


class MockPipelineStage(Enum):
    """Mock PipelineStage for testing."""

    IDLE = "idle"
    DATA_SYNC = "data_sync"
    NPZ_EXPORT = "npz_export"
    TRAINING = "training"
    EVALUATION = "evaluation"
    PROMOTION = "promotion"
    SELFPLAY = "selfplay"
    COMPLETE = "complete"


# =============================================================================
# Mock Coordinator Status
# =============================================================================


class MockCoordinatorStatus(Enum):
    """Mock CoordinatorStatus for testing."""

    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    DEGRADED = "degraded"


# =============================================================================
# Mock PipelineStats
# =============================================================================


@dataclass
class MockPipelineStats:
    """Mock PipelineStats for testing."""

    iterations_completed: int = 0
    iterations_failed: int = 0
    total_games_generated: int = 0
    total_models_trained: int = 0
    promotions: int = 0
    average_iteration_duration: float = 0.0
    stage_durations: dict[str, float] = field(default_factory=dict)
    last_activity_time: float = 0.0


# =============================================================================
# Mock IterationRecord
# =============================================================================


@dataclass
class MockIterationRecord:
    """Mock IterationRecord for testing."""

    iteration: int
    success: bool = True
    duration: float = 0.0
    games_generated: int = 0
    model_id: str | None = None
    elo_delta: float = 0.0


# =============================================================================
# Mock StageTransition
# =============================================================================


@dataclass
class MockStageTransition:
    """Mock StageTransition for testing."""

    from_stage: MockPipelineStage
    to_stage: MockPipelineStage
    timestamp: float = 0.0
    iteration: int = 0


# =============================================================================
# Mock CircuitBreaker
# =============================================================================


class MockCircuitBreaker:
    """Mock CircuitBreaker for testing."""

    def __init__(self, state: str = "closed"):
        self.state = SimpleNamespace(value=state)
        self._status = {
            "state": state,
            "failure_count": 0,
            "time_until_retry": 0,
        }

    def get_status(self) -> dict[str, Any]:
        return self._status

    def reset(self) -> None:
        self.state.value = "closed"
        self._status["state"] = "closed"
        self._status["failure_count"] = 0


# =============================================================================
# Test Class with Mixin
# =============================================================================


class MockOrchestratorWithMetricsMixin:
    """Mock class to test PipelineMetricsMixin.

    Provides all expected attributes and methods.
    """

    def __init__(self):
        # Required attributes from docstring
        self._current_stage = MockPipelineStage.IDLE
        self._current_iteration = 0
        self._iteration_records: dict[int, MockIterationRecord] = {}
        self._completed_iterations: list[MockIterationRecord] = []
        self._stage_start_times: dict[MockPipelineStage, float] = {}
        self._stage_durations: dict[MockPipelineStage, list[float]] = {}
        self._transitions: list[MockStageTransition] = []
        self._total_games = 0
        self._total_models = 0
        self._total_promotions = 0
        self._subscribed = False
        self.auto_trigger = True
        self.auto_trigger_sync = True
        self.auto_trigger_export = True
        self.auto_trigger_training = True
        self.auto_trigger_evaluation = True
        self.auto_trigger_promotion = True
        self._circuit_breaker: MockCircuitBreaker | None = None
        self._quality_distribution: dict[str, int] = {}
        self._pending_cache_refresh = False
        self._cache_invalidation_count = 0
        self._active_optimization: str | None = None
        self._optimization_run_id: str | None = None
        self._paused = False
        self._pause_reason: str | None = None
        self._backpressure_active = False
        self._resource_constraints: dict[str, Any] = {}
        self._coordinator_status = MockCoordinatorStatus.RUNNING
        self._start_time = time.time()
        self._events_processed = 0
        self._errors_count = 0
        self._last_error = ""
        self.max_history = 100
        self.name = "TestOrchestrator"

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time


# Dynamically add mixin methods to our test class
def _setup_test_class():
    """Import and apply mixin to test class."""
    from app.coordination.pipeline_metrics_mixin import PipelineMetricsMixin

    # Add mixin methods to test class
    for attr in dir(PipelineMetricsMixin):
        if not attr.startswith("_") or attr in (
            "_extract_stage_result",
        ):
            method = getattr(PipelineMetricsMixin, attr)
            if callable(method) and not isinstance(method, type):
                setattr(MockOrchestratorWithMetricsMixin, attr, method)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def orchestrator():
    """Create a test orchestrator instance."""
    _setup_test_class()
    return MockOrchestratorWithMetricsMixin()


@pytest.fixture
def orchestrator_with_data(orchestrator):
    """Create orchestrator with some test data."""
    # Add completed iterations
    for i in range(5):
        record = MockIterationRecord(
            iteration=i,
            success=i != 2,  # One failure
            duration=100.0 + i * 10,
            games_generated=1000 + i * 100,
        )
        orchestrator._completed_iterations.append(record)

    # Add stage durations
    orchestrator._stage_durations[MockPipelineStage.TRAINING] = [60.0, 70.0, 80.0]
    orchestrator._stage_durations[MockPipelineStage.EVALUATION] = [30.0, 35.0]

    # Add transitions
    for i in range(10):
        transition = MockStageTransition(
            from_stage=MockPipelineStage.IDLE,
            to_stage=MockPipelineStage.DATA_SYNC,
            timestamp=time.time() - (10 - i) * 60,
            iteration=i,
        )
        orchestrator._transitions.append(transition)

    # Set totals
    orchestrator._total_games = 6500
    orchestrator._total_models = 4
    orchestrator._total_promotions = 2

    return orchestrator


# =============================================================================
# Test get_current_stage
# =============================================================================


class TestGetCurrentStage:
    """Tests for get_current_stage method."""

    def test_returns_current_stage(self, orchestrator):
        """Returns the current pipeline stage."""
        orchestrator._current_stage = MockPipelineStage.TRAINING
        result = orchestrator.get_current_stage()
        assert result == MockPipelineStage.TRAINING

    def test_returns_idle_by_default(self, orchestrator):
        """Returns IDLE by default."""
        result = orchestrator.get_current_stage()
        assert result == MockPipelineStage.IDLE


# =============================================================================
# Test get_current_iteration
# =============================================================================


class TestGetCurrentIteration:
    """Tests for get_current_iteration method."""

    def test_returns_current_iteration(self, orchestrator):
        """Returns the current iteration number."""
        orchestrator._current_iteration = 5
        result = orchestrator.get_current_iteration()
        assert result == 5

    def test_returns_zero_by_default(self, orchestrator):
        """Returns 0 by default."""
        result = orchestrator.get_current_iteration()
        assert result == 0


# =============================================================================
# Test get_iteration_record
# =============================================================================


class TestGetIterationRecord:
    """Tests for get_iteration_record method."""

    def test_returns_active_record(self, orchestrator):
        """Returns record from active iterations."""
        record = MockIterationRecord(iteration=3, games_generated=100)
        orchestrator._iteration_records[3] = record

        result = orchestrator.get_iteration_record(3)
        assert result == record

    def test_returns_completed_record(self, orchestrator_with_data):
        """Returns record from completed iterations."""
        result = orchestrator_with_data.get_iteration_record(1)
        assert result is not None
        assert result.iteration == 1

    def test_returns_none_for_missing(self, orchestrator):
        """Returns None for non-existent iteration."""
        result = orchestrator.get_iteration_record(999)
        assert result is None


# =============================================================================
# Test get_recent_transitions
# =============================================================================


class TestGetRecentTransitions:
    """Tests for get_recent_transitions method."""

    def test_returns_transitions(self, orchestrator_with_data):
        """Returns recent transitions."""
        result = orchestrator_with_data.get_recent_transitions()
        assert len(result) == 10

    def test_respects_limit(self, orchestrator_with_data):
        """Respects limit parameter."""
        result = orchestrator_with_data.get_recent_transitions(limit=5)
        assert len(result) == 5

    def test_returns_empty_when_none(self, orchestrator):
        """Returns empty list when no transitions."""
        result = orchestrator.get_recent_transitions()
        assert result == []


# =============================================================================
# Test get_stage_metrics
# =============================================================================


class TestGetStageMetrics:
    """Tests for get_stage_metrics method."""

    def test_returns_stage_metrics(self, orchestrator_with_data):
        """Returns metrics for each stage with durations."""
        result = orchestrator_with_data.get_stage_metrics()

        assert "training" in result
        assert result["training"]["count"] == 3
        assert result["training"]["avg_duration"] == 70.0
        assert result["training"]["min_duration"] == 60.0
        assert result["training"]["max_duration"] == 80.0
        assert result["training"]["total_duration"] == 210.0

    def test_returns_empty_when_no_durations(self, orchestrator):
        """Returns empty dict when no stage durations recorded."""
        result = orchestrator.get_stage_metrics()
        assert result == {}


# =============================================================================
# Test get_stats
# =============================================================================


class TestGetStats:
    """Tests for get_stats method."""

    def test_returns_stats_object(self, orchestrator_with_data):
        """Returns PipelineStats-like object with aggregated data."""
        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStats",
            MockPipelineStats,
        ):
            result = orchestrator_with_data.get_stats()

        assert result.iterations_completed == 4  # 5 total, 1 failed
        assert result.iterations_failed == 1
        assert result.total_games_generated == 6500
        assert result.total_models_trained == 4
        assert result.promotions == 2

    def test_calculates_average_duration(self, orchestrator_with_data):
        """Calculates average iteration duration correctly."""
        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStats",
            MockPipelineStats,
        ):
            result = orchestrator_with_data.get_stats()

        # Average of successful iterations: (100, 110, 130, 140) / 4 = 120
        assert result.average_iteration_duration == 120.0


# =============================================================================
# Test get_status
# =============================================================================


class TestGetStatus:
    """Tests for get_status method."""

    def test_returns_status_dict(self, orchestrator_with_data):
        """Returns comprehensive status dictionary."""
        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStats",
            MockPipelineStats,
        ):
            result = orchestrator_with_data.get_status()

        assert result["current_stage"] == "idle"
        assert result["current_iteration"] == 0
        assert result["subscribed"] is False
        assert result["auto_trigger"] is True
        assert result["paused"] is False

    def test_includes_auto_trigger_flags(self, orchestrator):
        """Includes per-stage auto-trigger flags."""
        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStats",
            MockPipelineStats,
        ):
            result = orchestrator.get_status()

        assert result["auto_trigger_sync"] is True
        assert result["auto_trigger_export"] is True
        assert result["auto_trigger_training"] is True
        assert result["auto_trigger_evaluation"] is True
        assert result["auto_trigger_promotion"] is True


# =============================================================================
# Test get_health_status
# =============================================================================


class TestGetHealthStatus:
    """Tests for get_health_status method."""

    def test_healthy_when_no_issues(self, orchestrator):
        """Reports healthy when no issues detected."""
        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStage",
            MockPipelineStage,
        ), patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStats",
            MockPipelineStats,
        ):
            result = orchestrator.get_health_status()

        assert result["healthy"] is True
        assert result["status"] == "healthy"
        assert result["issues"] == []

    def test_detects_stuck_stage(self, orchestrator):
        """Detects when a stage is stuck (exceeded timeout)."""
        orchestrator._current_stage = MockPipelineStage.DATA_SYNC
        orchestrator._stage_start_times[MockPipelineStage.DATA_SYNC] = (
            time.time() - 2000  # 33+ minutes
        )

        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStage",
            MockPipelineStage,
        ), patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStats",
            MockPipelineStats,
        ):
            result = orchestrator.get_health_status()

        assert result["healthy"] is False
        assert any("stuck" in issue for issue in result["issues"])

    def test_detects_open_circuit_breaker(self, orchestrator):
        """Detects when circuit breaker is open."""
        orchestrator._circuit_breaker = MockCircuitBreaker(state="open")

        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStage",
            MockPipelineStage,
        ), patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStats",
            MockPipelineStats,
        ):
            result = orchestrator.get_health_status()

        assert result["healthy"] is False
        assert any("OPEN" in issue for issue in result["issues"])

    def test_detects_high_error_rate(self, orchestrator_with_data):
        """Detects high error rate across iterations."""
        # Add more failed iterations
        for i in range(10, 20):
            record = MockIterationRecord(iteration=i, success=False, duration=50.0)
            orchestrator_with_data._completed_iterations.append(record)

        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStage",
            MockPipelineStage,
        ), patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStats",
            MockPipelineStats,
        ):
            result = orchestrator_with_data.get_health_status()

        assert any("error rate" in issue.lower() for issue in result["issues"])

    def test_detects_paused_state(self, orchestrator):
        """Detects when pipeline is paused."""
        orchestrator._paused = True
        orchestrator._pause_reason = "Manual pause"

        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStage",
            MockPipelineStage,
        ), patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStats",
            MockPipelineStats,
        ):
            result = orchestrator.get_health_status()

        assert result["healthy"] is False
        assert any("paused" in issue.lower() for issue in result["issues"])

    def test_detects_backpressure(self, orchestrator):
        """Detects when backpressure is active."""
        orchestrator._backpressure_active = True

        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStage",
            MockPipelineStage,
        ), patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStats",
            MockPipelineStats,
        ):
            result = orchestrator.get_health_status()

        assert any("backpressure" in issue.lower() for issue in result["issues"])


# =============================================================================
# Test check_stage_timeout
# =============================================================================


class TestCheckStageTimeout:
    """Tests for check_stage_timeout method."""

    def test_returns_false_when_not_timed_out(self, orchestrator):
        """Returns (False, None) when stage hasn't timed out."""
        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStage",
            MockPipelineStage,
        ), patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStats",
            MockPipelineStats,
        ):
            timed_out, message = orchestrator.check_stage_timeout()

        assert timed_out is False
        assert message is None

    def test_returns_true_when_timed_out(self, orchestrator):
        """Returns (True, message) when stage has timed out."""
        orchestrator._current_stage = MockPipelineStage.PROMOTION
        orchestrator._stage_start_times[MockPipelineStage.PROMOTION] = (
            time.time() - 700  # 11+ minutes (timeout is 10 min)
        )

        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStage",
            MockPipelineStage,
        ), patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStats",
            MockPipelineStats,
        ):
            timed_out, message = orchestrator.check_stage_timeout()

        assert timed_out is True
        assert message is not None


# =============================================================================
# Test Circuit Breaker Methods
# =============================================================================


class TestCircuitBreakerMethods:
    """Tests for circuit breaker related methods."""

    def test_get_circuit_breaker_status_with_breaker(self, orchestrator):
        """Returns status when circuit breaker exists."""
        orchestrator._circuit_breaker = MockCircuitBreaker()

        result = orchestrator.get_circuit_breaker_status()
        assert result["state"] == "closed"

    def test_get_circuit_breaker_status_without_breaker(self, orchestrator):
        """Returns None when no circuit breaker."""
        result = orchestrator.get_circuit_breaker_status()
        assert result is None

    def test_reset_circuit_breaker(self, orchestrator):
        """Resets circuit breaker state."""
        orchestrator._circuit_breaker = MockCircuitBreaker(state="open")

        orchestrator.reset_circuit_breaker()

        assert orchestrator._circuit_breaker.state.value == "closed"


# =============================================================================
# Test get_metrics
# =============================================================================


class TestGetMetrics:
    """Tests for get_metrics method."""

    def test_returns_protocol_compliant_metrics(self, orchestrator):
        """Returns metrics in protocol-compliant format."""
        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStats",
            MockPipelineStats,
        ):
            result = orchestrator.get_metrics()

        assert result["name"] == "TestOrchestrator"
        assert result["status"] == "running"
        assert "uptime_seconds" in result
        assert result["current_stage"] == "idle"
        assert result["current_iteration"] == 0
        assert result["subscribed"] is False


# =============================================================================
# Test format_pipeline_report
# =============================================================================


class TestFormatPipelineReport:
    """Tests for format_pipeline_report method."""

    def test_returns_formatted_report(self, orchestrator_with_data):
        """Returns human-readable report string."""
        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStats",
            MockPipelineStats,
        ):
            result = orchestrator_with_data.format_pipeline_report()

        assert "DATA PIPELINE STATUS REPORT" in result
        assert "Current Stage:" in result
        assert "Iterations Completed:" in result
        assert "Total Games Generated:" in result

    def test_includes_stage_durations(self, orchestrator_with_data):
        """Includes stage duration information."""
        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStats",
            MockPipelineStats,
        ):
            result = orchestrator_with_data.format_pipeline_report()

        assert "Stage Durations" in result
