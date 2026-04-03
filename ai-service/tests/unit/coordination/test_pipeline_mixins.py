"""Unit tests for pipeline mixins (metrics and stage handlers).

Tests for:
- PipelineMetricsMixin - metrics, status, and health reporting
- PipelineStageMixin - stage callback handlers

December 2025: Created as part of mixin-based refactoring test coverage.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.contracts import CoordinatorStatus, HealthCheckResult


# ==============================================================================
# Mock Classes for Testing
# ==============================================================================


class MockPipelineStage(Enum):
    """Mock pipeline stage enum."""
    IDLE = "idle"
    SELFPLAY = "selfplay"
    DATA_SYNC = "data_sync"
    NPZ_EXPORT = "npz_export"
    TRAINING = "training"
    EVALUATION = "evaluation"
    PROMOTION = "promotion"
    COMPLETE = "complete"


class MockCoordinatorStatus(Enum):
    """Mock coordinator status enum."""
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class MockPipelineStats:
    """Mock pipeline stats dataclass."""
    iterations_completed: int = 0
    iterations_failed: int = 0
    total_games_generated: int = 0
    total_models_trained: int = 0
    promotions: int = 0
    average_iteration_duration: float = 0.0
    stage_durations: dict = None
    last_activity_time: float = 0.0

    def __post_init__(self):
        if self.stage_durations is None:
            self.stage_durations = {}


@dataclass
class MockIterationRecord:
    """Mock iteration record dataclass."""
    iteration: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    success: bool = True
    duration: float = 0.0
    games_generated: int = 0
    model_id: str | None = None
    elo_delta: float = 0.0
    promoted: bool = False
    error: str | None = None


@dataclass
class MockStageTransition:
    """Mock stage transition dataclass."""
    from_stage: str = "idle"
    to_stage: str = "selfplay"
    iteration: int = 0
    timestamp: float = 0.0
    success: bool = True
    metadata: dict = None


class MockCircuitBreaker:
    """Mock circuit breaker."""

    class State(Enum):
        CLOSED = "closed"
        OPEN = "open"
        HALF_OPEN = "half_open"

    def __init__(self):
        self.state = self.State.CLOSED
        self._failure_count = 0
        self._reset_called = False

    def get_status(self) -> dict:
        return {
            "state": self.state.value,
            "failure_count": self._failure_count,
            "time_until_retry": 30 if self.state == self.State.OPEN else 0,
        }

    def reset(self):
        self._reset_called = True
        self.state = self.State.CLOSED
        self._failure_count = 0


class MockRouterEvent:
    """Mock router event for testing stage result extraction."""

    def __init__(
        self,
        event_type: str = "test_event",
        payload: dict | None = None,
        stage_result=None,
    ):
        self.event_type = event_type
        self.payload = payload or {}
        self.stage_result = stage_result


# ==============================================================================
# Metrics Mixin Tests
# ==============================================================================


class TestPipelineMetricsMixin:
    """Tests for PipelineMetricsMixin."""

    def _create_mock_orchestrator(self, **overrides):
        """Create a mock orchestrator with metrics mixin attributes."""
        from app.coordination.pipeline_metrics_mixin import PipelineMetricsMixin

        class MockOrchestrator(PipelineMetricsMixin):
            def __init__(self):
                # Required attributes
                self.name = "DataPipelineOrchestrator"
                self._current_stage = MockPipelineStage.IDLE
                self._current_iteration = 0
                self._iteration_records = {}
                self._completed_iterations = []
                self._stage_start_times = {}
                self._stage_durations = defaultdict(list)
                self._transitions = []
                self._total_games = 0
                self._total_models = 0
                self._total_promotions = 0
                self._subscribed = True
                self.auto_trigger = True
                self.auto_trigger_sync = True
                self.auto_trigger_export = True
                self.auto_trigger_training = True
                self.auto_trigger_evaluation = True
                self.auto_trigger_promotion = True
                self._circuit_breaker = None
                self._quality_distribution = {}
                self._pending_cache_refresh = False
                self._cache_invalidation_count = 0
                self._active_optimization = None
                self._optimization_run_id = None
                self._paused = False
                self._pause_reason = None
                self._backpressure_active = False
                self._resource_constraints = {}
                self._coordinator_status = MockCoordinatorStatus.RUNNING
                self._start_time = time.time()
                self._events_processed = 0
                self._errors_count = 0
                self._last_error = ""
                self.max_history = 100

                @property
                def uptime_seconds(self):
                    return time.time() - self._start_time

                self.uptime_seconds = property(lambda self: time.time() - self._start_time)

        orchestrator = MockOrchestrator()

        # Apply overrides
        for key, value in overrides.items():
            setattr(orchestrator, key, value)

        return orchestrator

    def test_get_metrics_basic(self):
        """Test get_metrics returns expected keys."""
        with patch(
            "app.coordination.pipeline_metrics_mixin.PipelineMetricsMixin.get_stats"
        ) as mock_stats:
            mock_stats.return_value = MockPipelineStats(
                iterations_completed=5,
                iterations_failed=1,
                total_games_generated=1000,
                total_models_trained=5,
                promotions=3,
            )

            orchestrator = self._create_mock_orchestrator()
            metrics = orchestrator.get_metrics()

            assert "name" in metrics
            assert "status" in metrics
            assert "current_stage" in metrics
            assert "current_iteration" in metrics
            assert "iterations_completed" in metrics
            assert "iterations_failed" in metrics
            assert "total_games_generated" in metrics
            assert "total_models_trained" in metrics
            assert "promotions" in metrics
            assert "subscribed" in metrics
            assert "paused" in metrics

    def test_get_metrics_with_circuit_breaker(self):
        """Test get_metrics includes circuit breaker state."""
        with patch(
            "app.coordination.pipeline_metrics_mixin.PipelineMetricsMixin.get_stats"
        ) as mock_stats:
            mock_stats.return_value = MockPipelineStats()

            cb = MockCircuitBreaker()
            cb.state = MockCircuitBreaker.State.OPEN
            orchestrator = self._create_mock_orchestrator(_circuit_breaker=cb)

            metrics = orchestrator.get_metrics()
            assert metrics["circuit_breaker_state"] == "open"

    def test_get_current_stage(self):
        """Test get_current_stage returns the current stage."""
        orchestrator = self._create_mock_orchestrator(
            _current_stage=MockPipelineStage.TRAINING
        )
        assert orchestrator.get_current_stage() == MockPipelineStage.TRAINING

    def test_get_current_iteration(self):
        """Test get_current_iteration returns the current iteration."""
        orchestrator = self._create_mock_orchestrator(_current_iteration=42)
        assert orchestrator.get_current_iteration() == 42

    def test_get_iteration_record_from_active(self):
        """Test get_iteration_record finds active iterations."""
        record = MockIterationRecord(iteration=5, games_generated=100)
        orchestrator = self._create_mock_orchestrator(
            _iteration_records={5: record}
        )

        result = orchestrator.get_iteration_record(5)
        assert result is not None
        assert result.iteration == 5
        assert result.games_generated == 100

    def test_get_iteration_record_from_completed(self):
        """Test get_iteration_record finds completed iterations."""
        record = MockIterationRecord(iteration=3, success=True, promoted=True)
        orchestrator = self._create_mock_orchestrator(
            _completed_iterations=[record]
        )

        result = orchestrator.get_iteration_record(3)
        assert result is not None
        assert result.iteration == 3
        assert result.promoted is True

    def test_get_iteration_record_not_found(self):
        """Test get_iteration_record returns None for unknown iterations."""
        orchestrator = self._create_mock_orchestrator()
        assert orchestrator.get_iteration_record(999) is None

    def test_get_recent_transitions(self):
        """Test get_recent_transitions returns last N transitions."""
        transitions = [
            MockStageTransition(from_stage="idle", to_stage="selfplay", iteration=i)
            for i in range(30)
        ]
        orchestrator = self._create_mock_orchestrator(_transitions=transitions)

        result = orchestrator.get_recent_transitions(10)
        assert len(result) == 10
        # Should be the last 10
        assert result[0].iteration == 20
        assert result[-1].iteration == 29

    def test_get_recent_transitions_all(self):
        """Test get_recent_transitions with limit larger than list."""
        transitions = [
            MockStageTransition(iteration=i) for i in range(5)
        ]
        orchestrator = self._create_mock_orchestrator(_transitions=transitions)

        result = orchestrator.get_recent_transitions(100)
        assert len(result) == 5

    def test_get_stage_metrics(self):
        """Test get_stage_metrics computes statistics correctly."""
        durations = defaultdict(list)
        durations[MockPipelineStage.TRAINING] = [100.0, 200.0, 300.0]
        durations[MockPipelineStage.EVALUATION] = [50.0, 60.0]

        orchestrator = self._create_mock_orchestrator(_stage_durations=durations)
        metrics = orchestrator.get_stage_metrics()

        assert "training" in metrics
        assert metrics["training"]["count"] == 3
        assert metrics["training"]["avg_duration"] == 200.0
        assert metrics["training"]["min_duration"] == 100.0
        assert metrics["training"]["max_duration"] == 300.0
        assert metrics["training"]["total_duration"] == 600.0

        assert "evaluation" in metrics
        assert metrics["evaluation"]["count"] == 2

    def test_get_stage_metrics_empty(self):
        """Test get_stage_metrics with no durations."""
        orchestrator = self._create_mock_orchestrator()
        metrics = orchestrator.get_stage_metrics()
        assert metrics == {}

    def test_get_stats(self):
        """Test get_stats aggregates statistics correctly."""
        completed = [
            MockIterationRecord(iteration=1, success=True, duration=100.0),
            MockIterationRecord(iteration=2, success=True, duration=200.0),
            MockIterationRecord(iteration=3, success=False, duration=50.0),
        ]

        orchestrator = self._create_mock_orchestrator(
            _completed_iterations=completed,
            _total_games=500,
            _total_models=2,
            _total_promotions=1,
        )

        # Patch at import location inside get_stats()
        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStats",
            MockPipelineStats,
        ):
            stats = orchestrator.get_stats()

        assert stats.iterations_completed == 2
        assert stats.iterations_failed == 1
        assert stats.total_games_generated == 500
        assert stats.total_models_trained == 2
        assert stats.promotions == 1
        # Average of successful: (100 + 200) / 2 = 150
        assert stats.average_iteration_duration == 150.0

    def test_get_status(self):
        """Test get_status returns comprehensive status."""
        with patch.object(
            self._create_mock_orchestrator().__class__,
            "get_stats",
            return_value=MockPipelineStats(),
        ):
            with patch.object(
                self._create_mock_orchestrator().__class__,
                "get_circuit_breaker_status",
                return_value=None,
            ):
                orchestrator = self._create_mock_orchestrator(
                    _current_stage=MockPipelineStage.TRAINING,
                    _current_iteration=10,
                    _quality_distribution={"high": 50, "medium": 30, "low": 20},
                    _paused=True,
                    _pause_reason="Manual pause",
                )
                # Mock the methods
                orchestrator.get_stats = MagicMock(return_value=MockPipelineStats())
                orchestrator.get_circuit_breaker_status = MagicMock(return_value=None)

                status = orchestrator.get_status()

        assert status["current_stage"] == "training"
        assert status["current_iteration"] == 10
        assert status["subscribed"] is True
        assert status["auto_trigger"] is True
        assert status["paused"] is True
        assert status["pause_reason"] == "Manual pause"
        assert status["quality_distribution"] == {"high": 50, "medium": 30, "low": 20}

    def test_get_health_status_healthy(self):
        """Test get_health_status when pipeline is healthy."""
        orchestrator = self._create_mock_orchestrator()
        orchestrator.get_stats = MagicMock(
            return_value=MockPipelineStats(iterations_completed=10, iterations_failed=0)
        )
        orchestrator.get_circuit_breaker_status = MagicMock(return_value=None)

        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStage",
            MockPipelineStage,
        ):
            health = orchestrator.get_health_status()

        assert health["healthy"] is True
        assert health["status"] == "healthy"
        assert len(health["issues"]) == 0
        assert "stage_health" in health
        assert "stats" in health

    def test_get_health_status_circuit_breaker_open(self):
        """Test get_health_status when circuit breaker is open."""
        cb = MockCircuitBreaker()
        cb.state = MockCircuitBreaker.State.OPEN

        orchestrator = self._create_mock_orchestrator(_circuit_breaker=cb)
        orchestrator.get_stats = MagicMock(return_value=MockPipelineStats())
        orchestrator.get_circuit_breaker_status = MagicMock(
            return_value={"state": "open", "time_until_retry": 30}
        )

        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStage",
            MockPipelineStage,
        ):
            health = orchestrator.get_health_status()

        assert health["healthy"] is False
        assert "Circuit breaker is OPEN" in health["issues"][0]
        assert len(health["recommendations"]) > 0

    def test_get_health_status_high_error_rate(self):
        """Test get_health_status with high error rate."""
        orchestrator = self._create_mock_orchestrator()
        orchestrator.get_stats = MagicMock(
            return_value=MockPipelineStats(iterations_completed=5, iterations_failed=5)
        )
        orchestrator.get_circuit_breaker_status = MagicMock(return_value=None)

        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStage",
            MockPipelineStage,
        ):
            health = orchestrator.get_health_status()

        assert health["healthy"] is False
        assert any("error rate" in issue.lower() for issue in health["issues"])

    def test_get_health_status_paused(self):
        """Test get_health_status when pipeline is paused."""
        orchestrator = self._create_mock_orchestrator(
            _paused=True,
            _pause_reason="Resource constraints",
        )
        orchestrator.get_stats = MagicMock(return_value=MockPipelineStats())
        orchestrator.get_circuit_breaker_status = MagicMock(return_value=None)

        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStage",
            MockPipelineStage,
        ):
            health = orchestrator.get_health_status()

        assert health["healthy"] is False
        assert health["paused"] is True
        assert any("paused" in issue.lower() for issue in health["issues"])

    def test_get_health_status_backpressure(self):
        """Test get_health_status with backpressure active."""
        orchestrator = self._create_mock_orchestrator(_backpressure_active=True)
        orchestrator.get_stats = MagicMock(return_value=MockPipelineStats())
        orchestrator.get_circuit_breaker_status = MagicMock(return_value=None)

        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStage",
            MockPipelineStage,
        ):
            health = orchestrator.get_health_status()

        assert health["healthy"] is False
        assert health["backpressure"] is True

    def test_check_stage_timeout_not_timed_out(self):
        """Test check_stage_timeout when stage is within timeout."""
        orchestrator = self._create_mock_orchestrator()
        orchestrator.get_health_result = MagicMock(
            return_value=HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                details={
                    "stage_health": {"pct_timeout_used": 50},
                    "issues": [],
                },
            )
        )

        timed_out, msg = orchestrator.check_stage_timeout()
        assert timed_out is False
        assert msg is None

    def test_check_stage_timeout_timed_out(self):
        """Test check_stage_timeout when stage has timed out."""
        orchestrator = self._create_mock_orchestrator()
        orchestrator.get_health_result = MagicMock(
            return_value=HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.DEGRADED,
                details={
                    "stage_health": {"pct_timeout_used": 150},
                    "issues": ["Stage training stuck for 5.0 min"],
                },
            )
        )

        timed_out, msg = orchestrator.check_stage_timeout()
        assert timed_out is True
        assert "stuck" in msg.lower()

    def test_get_circuit_breaker_status_none(self):
        """Test get_circuit_breaker_status with no circuit breaker."""
        orchestrator = self._create_mock_orchestrator(_circuit_breaker=None)
        assert orchestrator.get_circuit_breaker_status() is None

    def test_get_circuit_breaker_status(self):
        """Test get_circuit_breaker_status returns status dict."""
        cb = MockCircuitBreaker()
        cb.state = MockCircuitBreaker.State.HALF_OPEN
        orchestrator = self._create_mock_orchestrator(_circuit_breaker=cb)

        status = orchestrator.get_circuit_breaker_status()
        assert status["state"] == "half_open"

    def test_reset_circuit_breaker(self):
        """Test reset_circuit_breaker resets the breaker."""
        cb = MockCircuitBreaker()
        cb.state = MockCircuitBreaker.State.OPEN
        orchestrator = self._create_mock_orchestrator(_circuit_breaker=cb)

        orchestrator.reset_circuit_breaker()

        assert cb._reset_called is True
        assert cb.state == MockCircuitBreaker.State.CLOSED

    def test_reset_circuit_breaker_none(self):
        """Test reset_circuit_breaker with no circuit breaker."""
        orchestrator = self._create_mock_orchestrator(_circuit_breaker=None)
        # Should not raise
        orchestrator.reset_circuit_breaker()

    def test_format_pipeline_report(self):
        """Test format_pipeline_report generates readable output."""
        orchestrator = self._create_mock_orchestrator(
            _current_stage=MockPipelineStage.TRAINING,
            _current_iteration=5,
        )
        orchestrator.get_stats = MagicMock(
            return_value=MockPipelineStats(
                iterations_completed=10,
                iterations_failed=2,
                total_games_generated=5000,
                total_models_trained=10,
                promotions=3,
                stage_durations={"training": 120.5, "evaluation": 45.2},
            )
        )

        report = orchestrator.format_pipeline_report()

        assert "DATA PIPELINE STATUS REPORT" in report
        assert "TRAINING" in report
        assert "5" in report  # iteration
        assert "10" in report  # completed
        assert "5,000" in report  # games
        assert "training:" in report.lower()


# ==============================================================================
# Stage Mixin Tests
# ==============================================================================


class TestPipelineStageMixin:
    """Tests for PipelineStageMixin."""

    def _create_mock_orchestrator(self, **overrides):
        """Create a mock orchestrator with stage mixin attributes."""
        from app.coordination.pipeline_stage_mixin import PipelineStageMixin

        class MockOrchestrator(PipelineStageMixin):
            def __init__(self):
                self._current_stage = MockPipelineStage.IDLE
                self._current_iteration = 0
                self._current_board_type = None
                self._current_num_players = None
                self._iteration_records = {}
                self._completed_iterations = []
                self._stage_start_times = {}
                self._total_games = 0
                self._total_models = 0
                self._total_promotions = 0
                self.auto_trigger = True
                self.auto_trigger_sync = True
                self.auto_trigger_export = True
                self.auto_trigger_training = True
                self.auto_trigger_evaluation = True
                self.auto_trigger_promotion = True
                self.quality_gate_enabled = False
                self._last_quality_score = 0.0
                self.max_history = 100

                # Mock methods
                self._transition_to = MagicMock()
                self._auto_trigger_sync = AsyncMock()
                self._auto_trigger_export = AsyncMock()
                self._auto_trigger_training = AsyncMock()
                self._auto_trigger_evaluation = AsyncMock()
                self._auto_trigger_promotion = AsyncMock()
                self._check_training_data_quality = AsyncMock(return_value=True)
                self._emit_training_blocked_by_quality = AsyncMock()
                self._trigger_model_sync_after_evaluation = AsyncMock()
                self._trigger_model_sync_after_promotion = AsyncMock()
                self._update_curriculum_on_promotion = AsyncMock()

            def _ensure_iteration_record(self, iteration):
                """Create or get iteration record and store in _iteration_records."""
                if iteration not in self._iteration_records:
                    self._iteration_records[iteration] = MockIterationRecord(
                        iteration=iteration
                    )
                return self._iteration_records[iteration]

        orchestrator = MockOrchestrator()

        for key, value in overrides.items():
            setattr(orchestrator, key, value)

        return orchestrator

    # ==========================================================================
    # Stage Result Extraction Tests
    # ==========================================================================

    def test_extract_stage_result_direct_result(self):
        """Test _extract_stage_result with direct StageCompletionResult."""
        orchestrator = self._create_mock_orchestrator()

        # Direct result object
        result = SimpleNamespace(iteration=5, success=True, games_generated=100)
        extracted = orchestrator._extract_stage_result(result)

        assert extracted is result

    def test_extract_stage_result_router_event_with_stage_result(self):
        """Test _extract_stage_result with RouterEvent containing stage_result."""
        orchestrator = self._create_mock_orchestrator()

        inner_result = SimpleNamespace(iteration=10, success=True)
        event = MockRouterEvent(stage_result=inner_result)

        extracted = orchestrator._extract_stage_result(event)
        assert extracted is inner_result
        assert extracted.iteration == 10

    def test_extract_stage_result_router_event_with_payload(self):
        """Test _extract_stage_result with RouterEvent containing payload only."""
        orchestrator = self._create_mock_orchestrator()

        event = MockRouterEvent(
            event_type="test_event",
            payload={
                "iteration": 7,
                "success": False,
                "games_generated": 50,
                "board_type": "hex8",
                "num_players": 2,
                "error": "Test error",
            },
        )

        extracted = orchestrator._extract_stage_result(event)

        assert extracted.iteration == 7
        assert extracted.success is False
        assert extracted.games_generated == 50
        assert extracted.board_type == "hex8"
        assert extracted.num_players == 2
        assert extracted.error == "Test error"

    def test_extract_stage_result_router_event_empty_payload(self):
        """Test _extract_stage_result with RouterEvent with empty payload."""
        orchestrator = self._create_mock_orchestrator()

        event = MockRouterEvent(event_type="test_event", payload={})

        extracted = orchestrator._extract_stage_result(event)

        assert extracted.iteration == 0
        assert extracted.success is True
        assert extracted.games_generated == 0

    # ==========================================================================
    # Selfplay Handler Tests
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_on_selfplay_complete_success(self):
        """Test _on_selfplay_complete with successful result."""
        orchestrator = self._create_mock_orchestrator()

        result = SimpleNamespace(
            iteration=1,
            success=True,
            games_generated=500,
            board_type="hex8",
            num_players=2,
        )

        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStage",
            MockPipelineStage,
        ):
            await orchestrator._on_selfplay_complete(result)

        assert orchestrator._total_games == 500
        assert orchestrator._current_board_type == "hex8"
        assert orchestrator._current_num_players == 2
        orchestrator._transition_to.assert_called()
        orchestrator._auto_trigger_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_selfplay_complete_failure(self):
        """Test _on_selfplay_complete with failed result."""
        orchestrator = self._create_mock_orchestrator()

        result = SimpleNamespace(
            iteration=1,
            success=False,
            games_generated=0,
            error="Selfplay failed",
        )

        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStage",
            MockPipelineStage,
        ):
            await orchestrator._on_selfplay_complete(result)

        # Should transition to IDLE on failure
        call_args = orchestrator._transition_to.call_args
        assert call_args[0][0] == MockPipelineStage.IDLE
        orchestrator._auto_trigger_sync.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_selfplay_complete_auto_trigger_disabled(self):
        """Test _on_selfplay_complete with auto_trigger disabled."""
        orchestrator = self._create_mock_orchestrator(auto_trigger=False)

        result = SimpleNamespace(iteration=1, success=True, games_generated=100)

        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStage",
            MockPipelineStage,
        ):
            await orchestrator._on_selfplay_complete(result)

        orchestrator._auto_trigger_sync.assert_not_called()

    # ==========================================================================
    # Sync Handler Tests
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_on_sync_complete_success(self):
        """Test _on_sync_complete with successful result."""
        orchestrator = self._create_mock_orchestrator()

        result = SimpleNamespace(iteration=1, success=True, metadata={"files": 10})

        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStage",
            MockPipelineStage,
        ):
            await orchestrator._on_sync_complete(result)

        call_args = orchestrator._transition_to.call_args
        assert call_args[0][0] == MockPipelineStage.NPZ_EXPORT
        orchestrator._auto_trigger_export.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_sync_complete_failure(self):
        """Test _on_sync_complete with failed result."""
        orchestrator = self._create_mock_orchestrator()

        result = SimpleNamespace(iteration=1, success=False, error="Sync failed")

        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStage",
            MockPipelineStage,
        ):
            await orchestrator._on_sync_complete(result)

        call_args = orchestrator._transition_to.call_args
        assert call_args[0][0] == MockPipelineStage.IDLE
        orchestrator._auto_trigger_export.assert_not_called()

    # ==========================================================================
    # NPZ Export Handler Tests
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_on_npz_export_complete_success(self):
        """Test _on_npz_export_complete with successful result."""
        orchestrator = self._create_mock_orchestrator()

        result = SimpleNamespace(
            iteration=1,
            success=True,
            output_path="/path/to/data.npz",
            metadata={},
        )

        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStage",
            MockPipelineStage,
        ):
            await orchestrator._on_npz_export_complete(result)

        call_args = orchestrator._transition_to.call_args
        assert call_args[0][0] == MockPipelineStage.TRAINING
        orchestrator._auto_trigger_training.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_npz_export_complete_quality_gate_blocks(self):
        """Test _on_npz_export_complete when quality gate blocks training."""
        orchestrator = self._create_mock_orchestrator(quality_gate_enabled=True)
        orchestrator._check_training_data_quality = AsyncMock(return_value=False)
        orchestrator._last_quality_score = 0.3

        result = SimpleNamespace(
            iteration=1,
            success=True,
            output_path="/path/to/data.npz",
            metadata={},
        )

        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStage",
            MockPipelineStage,
        ):
            await orchestrator._on_npz_export_complete(result)

        # Should NOT transition to training
        orchestrator._transition_to.assert_not_called()
        orchestrator._emit_training_blocked_by_quality.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_npz_export_complete_failure(self):
        """Test _on_npz_export_complete with failed result."""
        orchestrator = self._create_mock_orchestrator()

        result = SimpleNamespace(iteration=1, success=False, error="Export failed")

        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStage",
            MockPipelineStage,
        ):
            await orchestrator._on_npz_export_complete(result)

        call_args = orchestrator._transition_to.call_args
        assert call_args[0][0] == MockPipelineStage.IDLE

    # ==========================================================================
    # Training Handler Tests
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_on_training_started(self):
        """Test _on_training_started updates stage start time."""
        orchestrator = self._create_mock_orchestrator()

        result = SimpleNamespace(iteration=1)

        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStage",
            MockPipelineStage,
        ):
            await orchestrator._on_training_started(result)

        assert MockPipelineStage.TRAINING in orchestrator._stage_start_times
        # Verify iteration record was created
        assert 1 in orchestrator._iteration_records

    @pytest.mark.asyncio
    async def test_on_training_complete_success(self):
        """Test _on_training_complete with successful result."""
        orchestrator = self._create_mock_orchestrator()

        result = SimpleNamespace(
            iteration=1,
            success=True,
            model_id="model_v1",
            model_path="/path/to/model.pth",
            train_loss=0.5,
            val_loss=0.6,
            metadata={"model_path": "/path/to/model.pth"},
        )

        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStage",
            MockPipelineStage,
        ):
            await orchestrator._on_training_complete(result)

        assert orchestrator._total_models == 1
        call_args = orchestrator._transition_to.call_args
        assert call_args[0][0] == MockPipelineStage.EVALUATION
        orchestrator._auto_trigger_evaluation.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_training_complete_failure(self):
        """Test _on_training_complete with failed result."""
        orchestrator = self._create_mock_orchestrator()

        result = SimpleNamespace(
            iteration=1,
            success=False,
            error="Training diverged",
        )

        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStage",
            MockPipelineStage,
        ):
            await orchestrator._on_training_complete(result)

        call_args = orchestrator._transition_to.call_args
        assert call_args[0][0] == MockPipelineStage.IDLE

    @pytest.mark.asyncio
    async def test_on_training_failed(self):
        """Test _on_training_failed transitions to IDLE."""
        orchestrator = self._create_mock_orchestrator()

        result = SimpleNamespace(iteration=1, error="OOM error")

        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStage",
            MockPipelineStage,
        ):
            await orchestrator._on_training_failed(result)

        call_args = orchestrator._transition_to.call_args
        assert call_args[0][0] == MockPipelineStage.IDLE
        assert call_args[1]["success"] is False

    # ==========================================================================
    # Evaluation Handler Tests
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_on_evaluation_complete_success(self):
        """Test _on_evaluation_complete with successful result."""
        orchestrator = self._create_mock_orchestrator()

        result = SimpleNamespace(
            iteration=1,
            success=True,
            win_rate=0.65,
            elo_delta=50.0,
            model_path="/path/to/model.pth",
            metadata={"model_path": "/path/to/model.pth"},
        )

        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStage",
            MockPipelineStage,
        ):
            await orchestrator._on_evaluation_complete(result)

        call_args = orchestrator._transition_to.call_args
        assert call_args[0][0] == MockPipelineStage.PROMOTION
        orchestrator._auto_trigger_promotion.assert_called_once()
        orchestrator._trigger_model_sync_after_evaluation.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_evaluation_complete_failure(self):
        """Test _on_evaluation_complete with failed result."""
        orchestrator = self._create_mock_orchestrator()

        result = SimpleNamespace(iteration=1, success=False, error="Gauntlet failed")

        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStage",
            MockPipelineStage,
        ):
            await orchestrator._on_evaluation_complete(result)

        call_args = orchestrator._transition_to.call_args
        assert call_args[0][0] == MockPipelineStage.IDLE

    # ==========================================================================
    # Promotion Handler Tests
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_on_promotion_complete_promoted(self):
        """Test _on_promotion_complete when model is promoted."""
        orchestrator = self._create_mock_orchestrator()

        result = SimpleNamespace(
            iteration=1,
            promoted=True,
            promotion_reason="Beat baseline by 10 Elo",
        )

        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStage",
            MockPipelineStage,
        ):
            await orchestrator._on_promotion_complete(result)

        assert orchestrator._total_promotions == 1
        call_args = orchestrator._transition_to.call_args
        assert call_args[0][0] == MockPipelineStage.COMPLETE
        orchestrator._update_curriculum_on_promotion.assert_called_once()
        orchestrator._trigger_model_sync_after_promotion.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_promotion_complete_not_promoted(self):
        """Test _on_promotion_complete when model is not promoted."""
        orchestrator = self._create_mock_orchestrator()

        result = SimpleNamespace(
            iteration=1,
            promoted=False,
            promotion_reason="Below threshold",
        )

        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStage",
            MockPipelineStage,
        ):
            await orchestrator._on_promotion_complete(result)

        assert orchestrator._total_promotions == 0
        orchestrator._trigger_model_sync_after_promotion.assert_not_called()

    # ==========================================================================
    # Iteration Complete Handler Tests
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_on_iteration_complete(self):
        """Test _on_iteration_complete finalizes iteration."""
        record = MockIterationRecord(iteration=5, start_time=time.time() - 100)
        orchestrator = self._create_mock_orchestrator(
            _iteration_records={5: record},
            max_history=10,
        )

        result = SimpleNamespace(iteration=5, success=True)

        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStage",
            MockPipelineStage,
        ):
            await orchestrator._on_iteration_complete(result)

        # Record should be moved to completed
        assert 5 not in orchestrator._iteration_records
        assert record in orchestrator._completed_iterations
        assert record.success is True
        assert record.end_time > 0

    @pytest.mark.asyncio
    async def test_on_iteration_complete_trims_history(self):
        """Test _on_iteration_complete trims history to max_history."""
        # Pre-populate with max_history records
        completed = [MockIterationRecord(iteration=i) for i in range(100)]
        new_record = MockIterationRecord(iteration=100, start_time=time.time())

        orchestrator = self._create_mock_orchestrator(
            _iteration_records={100: new_record},
            _completed_iterations=completed,
            max_history=50,
        )

        result = SimpleNamespace(iteration=100, success=True)

        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStage",
            MockPipelineStage,
        ):
            await orchestrator._on_iteration_complete(result)

        # Should be trimmed to max_history
        assert len(orchestrator._completed_iterations) == 50

    @pytest.mark.asyncio
    async def test_on_iteration_complete_unknown_iteration(self):
        """Test _on_iteration_complete with unknown iteration is a no-op."""
        orchestrator = self._create_mock_orchestrator()

        result = SimpleNamespace(iteration=999, success=True)

        with patch(
            "app.coordination.data_pipeline_orchestrator.PipelineStage",
            MockPipelineStage,
        ):
            await orchestrator._on_iteration_complete(result)

        # Should not raise, just transition
        orchestrator._transition_to.assert_called()


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestMixinIntegration:
    """Integration tests for combined mixin behavior."""

    def test_both_mixins_compatible(self):
        """Test that both mixins can be used together."""
        from app.coordination.pipeline_metrics_mixin import PipelineMetricsMixin
        from app.coordination.pipeline_stage_mixin import PipelineStageMixin

        class CombinedOrchestrator(PipelineMetricsMixin, PipelineStageMixin):
            def __init__(self):
                # Minimal required attributes
                self._current_stage = MockPipelineStage.IDLE
                self._current_iteration = 0
                self._iteration_records = {}
                self._completed_iterations = []
                self._stage_start_times = {}
                self._stage_durations = defaultdict(list)
                self._transitions = []
                self._total_games = 0
                self._total_models = 0
                self._total_promotions = 0

        orchestrator = CombinedOrchestrator()

        # Both mixin methods should be available
        assert hasattr(orchestrator, "get_metrics")
        assert hasattr(orchestrator, "get_health_status")
        assert hasattr(orchestrator, "_on_selfplay_complete")
        assert hasattr(orchestrator, "_extract_stage_result")
