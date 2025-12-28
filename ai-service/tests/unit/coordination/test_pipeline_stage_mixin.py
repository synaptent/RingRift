"""Unit tests for pipeline_stage_mixin.py.

December 2025: Tests for stage callback handlers extracted from DataPipelineOrchestrator.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest


# ============================================================================
# Mock Types (avoid import from actual modules to test mixin in isolation)
# ============================================================================


class MockPipelineStage:
    """Mock PipelineStage enum."""

    IDLE = "idle"
    SELFPLAY = "selfplay"
    DATA_SYNC = "data_sync"
    NPZ_EXPORT = "npz_export"
    TRAINING = "training"
    EVALUATION = "evaluation"
    PROMOTION = "promotion"
    COMPLETE = "complete"


@dataclass
class MockIterationRecord:
    """Mock iteration record."""

    iteration: int
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    games_generated: int = 0
    model_id: str | None = None
    elo_delta: float = 0.0
    promoted: bool = False
    error: str | None = None
    success: bool = True


@dataclass
class MockStageCompletionResult:
    """Mock stage completion result."""

    iteration: int = 0
    success: bool = True
    games_generated: int = 0
    board_type: str | None = None
    num_players: int | None = None
    error: str | None = None
    metadata: dict = field(default_factory=dict)
    model_id: str | None = None
    model_path: str | None = None
    train_loss: float | None = None
    val_loss: float | None = None
    win_rate: float | None = None
    elo_delta: float | None = None
    promoted: bool = False
    promotion_reason: str = ""
    output_path: str | None = None


@dataclass
class MockRouterEvent:
    """Mock RouterEvent that wraps stage results."""

    event_type: str
    payload: dict = field(default_factory=dict)
    stage_result: Any = None
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# Test Mixin Class
# ============================================================================


class MockOrchestratorWithStageMixin:
    """Mock orchestrator using PipelineStageMixin."""

    def __init__(self):
        # Required state
        self._current_stage = MockPipelineStage.IDLE
        self._current_iteration = 0
        self._current_board_type: str | None = None
        self._current_num_players: int | None = None
        self._iteration_records: dict[int, MockIterationRecord] = {}
        self._completed_iterations: list[MockIterationRecord] = []
        self._stage_start_times: dict[str, float] = {}
        self._total_games = 0
        self._total_models = 0
        self._total_promotions = 0

        # Auto-trigger flags
        self.auto_trigger = True
        self.auto_trigger_sync = True
        self.auto_trigger_export = True
        self.auto_trigger_training = True
        self.auto_trigger_evaluation = True
        self.auto_trigger_promotion = True

        # Quality gate
        self.quality_gate_enabled = False
        self._last_quality_score = 0.0

        # History limit
        self.max_history = 50

        # Track transitions for assertions
        self._transitions: list[dict] = []

        # Mock async methods
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

    def _ensure_iteration_record(self, iteration: int) -> MockIterationRecord:
        """Ensure iteration record exists."""
        if iteration not in self._iteration_records:
            self._iteration_records[iteration] = MockIterationRecord(iteration=iteration)
        return self._iteration_records[iteration]

    def _transition_to(
        self,
        stage,
        iteration: int,
        success: bool = True,
        metadata: dict | None = None,
    ) -> None:
        """Track stage transitions."""
        self._current_stage = stage
        self._current_iteration = iteration
        self._transitions.append(
            {
                "stage": stage,
                "iteration": iteration,
                "success": success,
                "metadata": metadata or {},
            }
        )

    # Import mixin methods
    from app.coordination.pipeline_stage_mixin import PipelineStageMixin

    _extract_stage_result = PipelineStageMixin._extract_stage_result
    _on_selfplay_complete = PipelineStageMixin._on_selfplay_complete
    _on_sync_complete = PipelineStageMixin._on_sync_complete
    _on_npz_export_complete = PipelineStageMixin._on_npz_export_complete
    _on_training_started = PipelineStageMixin._on_training_started
    _on_training_complete = PipelineStageMixin._on_training_complete
    _on_training_failed = PipelineStageMixin._on_training_failed
    _on_evaluation_complete = PipelineStageMixin._on_evaluation_complete
    _on_promotion_complete = PipelineStageMixin._on_promotion_complete
    _on_iteration_complete = PipelineStageMixin._on_iteration_complete


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def orchestrator():
    """Create mock orchestrator with stage mixin."""
    return MockOrchestratorWithStageMixin()


# ============================================================================
# Test: _extract_stage_result
# ============================================================================


class TestExtractStageResult:
    """Tests for _extract_stage_result helper."""

    def test_returns_stage_result_from_router_event(self, orchestrator):
        """Test extracting stage_result from RouterEvent."""
        inner_result = MockStageCompletionResult(iteration=5, success=True)
        event = MockRouterEvent(event_type="test", stage_result=inner_result)

        extracted = orchestrator._extract_stage_result(event)

        assert extracted is inner_result
        assert extracted.iteration == 5

    def test_extracts_from_payload_when_no_stage_result(self, orchestrator):
        """Test creating SimpleNamespace from payload when no stage_result."""
        event = MockRouterEvent(
            event_type="test",
            payload={
                "iteration": 10,
                "success": True,
                "games_generated": 100,
                "board_type": "hex8",
                "num_players": 2,
            },
            stage_result=None,
        )

        extracted = orchestrator._extract_stage_result(event)

        assert isinstance(extracted, SimpleNamespace)
        assert extracted.iteration == 10
        assert extracted.success is True
        assert extracted.games_generated == 100
        assert extracted.board_type == "hex8"
        assert extracted.num_players == 2

    def test_returns_result_directly_when_not_router_event(self, orchestrator):
        """Test returning result as-is when not a RouterEvent."""
        result = MockStageCompletionResult(iteration=3, success=False)

        extracted = orchestrator._extract_stage_result(result)

        assert extracted is result

    def test_handles_empty_payload(self, orchestrator):
        """Test handling RouterEvent with empty payload."""
        event = MockRouterEvent(event_type="test", payload={}, stage_result=None)

        extracted = orchestrator._extract_stage_result(event)

        assert extracted.iteration == 0
        assert extracted.success is True


# ============================================================================
# Test: _on_selfplay_complete
# ============================================================================


class TestOnSelfplayComplete:
    """Tests for _on_selfplay_complete handler."""

    @pytest.mark.asyncio
    async def test_successful_selfplay_transitions_to_data_sync(self, orchestrator):
        """Test successful selfplay triggers transition to DATA_SYNC."""
        result = MockStageCompletionResult(
            iteration=1,
            success=True,
            games_generated=500,
            board_type="hex8",
            num_players=2,
        )

        await orchestrator._on_selfplay_complete(result)

        assert len(orchestrator._transitions) == 1
        # Compare enum value (real PipelineStage imported by mixin)
        stage = orchestrator._transitions[0]["stage"]
        assert stage.value == "data_sync"
        assert orchestrator._transitions[0]["iteration"] == 1
        assert orchestrator._total_games == 500

    @pytest.mark.asyncio
    async def test_failed_selfplay_transitions_to_idle(self, orchestrator):
        """Test failed selfplay transitions to IDLE."""
        result = MockStageCompletionResult(
            iteration=1,
            success=False,
            error="Out of memory",
        )

        await orchestrator._on_selfplay_complete(result)

        assert len(orchestrator._transitions) == 1
        stage = orchestrator._transitions[0]["stage"]
        assert stage.value == "idle"
        assert orchestrator._transitions[0]["success"] is False

    @pytest.mark.asyncio
    async def test_auto_triggers_sync_when_enabled(self, orchestrator):
        """Test auto-trigger sync is called when enabled."""
        result = MockStageCompletionResult(
            iteration=1,
            success=True,
            games_generated=100,
        )
        orchestrator.auto_trigger = True
        orchestrator.auto_trigger_sync = True

        await orchestrator._on_selfplay_complete(result)

        orchestrator._auto_trigger_sync.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_skips_auto_sync_when_disabled(self, orchestrator):
        """Test auto-trigger sync is skipped when disabled."""
        result = MockStageCompletionResult(
            iteration=1,
            success=True,
            games_generated=100,
        )
        orchestrator.auto_trigger_sync = False

        await orchestrator._on_selfplay_complete(result)

        orchestrator._auto_trigger_sync.assert_not_called()

    @pytest.mark.asyncio
    async def test_tracks_board_configuration(self, orchestrator):
        """Test board type and player count are tracked."""
        result = MockStageCompletionResult(
            iteration=1,
            success=True,
            games_generated=100,
            board_type="square8",
            num_players=4,
        )

        await orchestrator._on_selfplay_complete(result)

        assert orchestrator._current_board_type == "square8"
        assert orchestrator._current_num_players == 4


# ============================================================================
# Test: _on_sync_complete
# ============================================================================


class TestOnSyncComplete:
    """Tests for _on_sync_complete handler."""

    @pytest.mark.asyncio
    async def test_successful_sync_transitions_to_npz_export(self, orchestrator):
        """Test successful sync triggers transition to NPZ_EXPORT."""
        result = MockStageCompletionResult(iteration=2, success=True)

        await orchestrator._on_sync_complete(result)

        assert len(orchestrator._transitions) == 1
        assert orchestrator._transitions[0]["stage"].value == "npz_export"
        assert orchestrator._transitions[0]["iteration"] == 2

    @pytest.mark.asyncio
    async def test_failed_sync_transitions_to_idle(self, orchestrator):
        """Test failed sync transitions to IDLE."""
        result = MockStageCompletionResult(
            iteration=2,
            success=False,
            error="Connection refused",
        )

        await orchestrator._on_sync_complete(result)

        assert orchestrator._transitions[0]["stage"].value == "idle"
        assert orchestrator._transitions[0]["success"] is False

    @pytest.mark.asyncio
    async def test_auto_triggers_export_when_enabled(self, orchestrator):
        """Test auto-trigger export is called when enabled."""
        result = MockStageCompletionResult(iteration=2, success=True)

        await orchestrator._on_sync_complete(result)

        orchestrator._auto_trigger_export.assert_called_once_with(2)


# ============================================================================
# Test: _on_npz_export_complete
# ============================================================================


class TestOnNpzExportComplete:
    """Tests for _on_npz_export_complete handler."""

    @pytest.mark.asyncio
    async def test_successful_export_transitions_to_training(self, orchestrator):
        """Test successful export triggers transition to TRAINING."""
        result = MockStageCompletionResult(
            iteration=3,
            success=True,
            output_path="/data/training/hex8_2p.npz",
        )

        await orchestrator._on_npz_export_complete(result)

        assert orchestrator._transitions[0]["stage"].value == "training"

    @pytest.mark.asyncio
    async def test_quality_gate_blocks_low_quality(self, orchestrator):
        """Test quality gate blocks training on low quality data."""
        result = MockStageCompletionResult(
            iteration=3,
            success=True,
            output_path="/data/training/hex8_2p.npz",
        )
        orchestrator.quality_gate_enabled = True
        orchestrator._check_training_data_quality.return_value = False
        orchestrator._last_quality_score = 0.3

        await orchestrator._on_npz_export_complete(result)

        # Should not transition to training
        assert len(orchestrator._transitions) == 0
        orchestrator._emit_training_blocked_by_quality.assert_called_once()

    @pytest.mark.asyncio
    async def test_quality_gate_allows_high_quality(self, orchestrator):
        """Test quality gate allows training on high quality data."""
        result = MockStageCompletionResult(
            iteration=3,
            success=True,
            output_path="/data/training/hex8_2p.npz",
        )
        orchestrator.quality_gate_enabled = True
        orchestrator._check_training_data_quality.return_value = True

        await orchestrator._on_npz_export_complete(result)

        assert orchestrator._transitions[0]["stage"].value == "training"

    @pytest.mark.asyncio
    async def test_auto_triggers_training_when_enabled(self, orchestrator):
        """Test auto-trigger training is called when enabled."""
        result = MockStageCompletionResult(
            iteration=3,
            success=True,
            output_path="/data/training/hex8_2p.npz",
        )

        await orchestrator._on_npz_export_complete(result)

        orchestrator._auto_trigger_training.assert_called_once_with(
            3, "/data/training/hex8_2p.npz"
        )


# ============================================================================
# Test: _on_training_started
# ============================================================================


class TestOnTrainingStarted:
    """Tests for _on_training_started handler."""

    @pytest.mark.asyncio
    async def test_records_training_start_time(self, orchestrator):
        """Test training start time is recorded."""
        from app.coordination.data_pipeline_orchestrator import PipelineStage

        result = MockStageCompletionResult(iteration=4)

        await orchestrator._on_training_started(result)

        assert PipelineStage.TRAINING in orchestrator._stage_start_times

    @pytest.mark.asyncio
    async def test_ensures_iteration_record_exists(self, orchestrator):
        """Test iteration record is created if missing."""
        result = MockStageCompletionResult(iteration=4)

        await orchestrator._on_training_started(result)

        assert 4 in orchestrator._iteration_records


# ============================================================================
# Test: _on_training_complete
# ============================================================================


class TestOnTrainingComplete:
    """Tests for _on_training_complete handler."""

    @pytest.mark.asyncio
    async def test_successful_training_transitions_to_evaluation(self, orchestrator):
        """Test successful training triggers transition to EVALUATION."""
        result = MockStageCompletionResult(
            iteration=4,
            success=True,
            model_id="model_v1",
            train_loss=0.05,
            val_loss=0.08,
        )

        await orchestrator._on_training_complete(result)

        assert orchestrator._transitions[0]["stage"].value == "evaluation"
        assert orchestrator._total_models == 1

    @pytest.mark.asyncio
    async def test_failed_training_transitions_to_idle(self, orchestrator):
        """Test failed training transitions to IDLE."""
        result = MockStageCompletionResult(
            iteration=4,
            success=False,
            error="CUDA out of memory",
        )

        await orchestrator._on_training_complete(result)

        assert orchestrator._transitions[0]["stage"].value == "idle"
        assert orchestrator._transitions[0]["success"] is False

    @pytest.mark.asyncio
    async def test_auto_triggers_evaluation_when_enabled(self, orchestrator):
        """Test auto-trigger evaluation is called when enabled."""
        result = MockStageCompletionResult(
            iteration=4,
            success=True,
            model_path="/models/hex8_2p.pth",
        )

        await orchestrator._on_training_complete(result)

        orchestrator._auto_trigger_evaluation.assert_called_once_with(
            4, "/models/hex8_2p.pth"
        )

    @pytest.mark.asyncio
    async def test_records_model_id_in_iteration_record(self, orchestrator):
        """Test model_id is recorded in iteration record."""
        result = MockStageCompletionResult(
            iteration=4,
            success=True,
            model_id="best_model_v2",
        )

        await orchestrator._on_training_complete(result)

        assert orchestrator._iteration_records[4].model_id == "best_model_v2"


# ============================================================================
# Test: _on_training_failed
# ============================================================================


class TestOnTrainingFailed:
    """Tests for _on_training_failed handler."""

    @pytest.mark.asyncio
    async def test_transitions_to_idle_on_failure(self, orchestrator):
        """Test training failure transitions to IDLE."""
        result = MockStageCompletionResult(
            iteration=4,
            error="Training diverged",
        )

        await orchestrator._on_training_failed(result)

        assert orchestrator._transitions[0]["stage"].value == "idle"
        assert orchestrator._transitions[0]["success"] is False

    @pytest.mark.asyncio
    async def test_records_error_in_iteration_record(self, orchestrator):
        """Test error is recorded in iteration record."""
        result = MockStageCompletionResult(
            iteration=4,
            error="Gradient explosion",
        )

        await orchestrator._on_training_failed(result)

        assert orchestrator._iteration_records[4].error == "Gradient explosion"


# ============================================================================
# Test: _on_evaluation_complete
# ============================================================================


class TestOnEvaluationComplete:
    """Tests for _on_evaluation_complete handler."""

    @pytest.mark.asyncio
    async def test_successful_evaluation_transitions_to_promotion(self, orchestrator):
        """Test successful evaluation triggers transition to PROMOTION."""
        result = MockStageCompletionResult(
            iteration=5,
            success=True,
            win_rate=0.85,
            elo_delta=50.0,
        )

        await orchestrator._on_evaluation_complete(result)

        assert orchestrator._transitions[0]["stage"].value == "promotion"

    @pytest.mark.asyncio
    async def test_records_elo_delta_in_iteration_record(self, orchestrator):
        """Test elo_delta is recorded in iteration record."""
        result = MockStageCompletionResult(
            iteration=5,
            success=True,
            elo_delta=75.0,
        )

        await orchestrator._on_evaluation_complete(result)

        assert orchestrator._iteration_records[5].elo_delta == 75.0

    @pytest.mark.asyncio
    async def test_auto_triggers_promotion_when_enabled(self, orchestrator):
        """Test auto-trigger promotion is called when enabled."""
        result = MockStageCompletionResult(
            iteration=5,
            success=True,
            model_path="/models/hex8_2p.pth",
            metadata={"win_rate": 0.85},
        )

        await orchestrator._on_evaluation_complete(result)

        orchestrator._auto_trigger_promotion.assert_called_once()

    @pytest.mark.asyncio
    async def test_triggers_model_sync_after_evaluation(self, orchestrator):
        """Test model sync is triggered after successful evaluation."""
        result = MockStageCompletionResult(
            iteration=5,
            success=True,
        )

        await orchestrator._on_evaluation_complete(result)

        orchestrator._trigger_model_sync_after_evaluation.assert_called_once()


# ============================================================================
# Test: _on_promotion_complete
# ============================================================================


class TestOnPromotionComplete:
    """Tests for _on_promotion_complete handler."""

    @pytest.mark.asyncio
    async def test_transitions_to_complete(self, orchestrator):
        """Test promotion completion transitions to COMPLETE."""
        result = MockStageCompletionResult(
            iteration=6,
            promoted=True,
            promotion_reason="Win rate > 85%",
        )

        await orchestrator._on_promotion_complete(result)

        assert orchestrator._transitions[0]["stage"].value == "complete"

    @pytest.mark.asyncio
    async def test_increments_total_promotions_when_promoted(self, orchestrator):
        """Test total_promotions is incremented when model is promoted."""
        result = MockStageCompletionResult(
            iteration=6,
            promoted=True,
        )

        await orchestrator._on_promotion_complete(result)

        assert orchestrator._total_promotions == 1

    @pytest.mark.asyncio
    async def test_does_not_increment_when_not_promoted(self, orchestrator):
        """Test total_promotions is not incremented when model is not promoted."""
        result = MockStageCompletionResult(
            iteration=6,
            promoted=False,
        )

        await orchestrator._on_promotion_complete(result)

        assert orchestrator._total_promotions == 0

    @pytest.mark.asyncio
    async def test_updates_curriculum_on_promotion(self, orchestrator):
        """Test curriculum is updated after promotion."""
        result = MockStageCompletionResult(
            iteration=6,
            promoted=True,
        )

        await orchestrator._on_promotion_complete(result)

        orchestrator._update_curriculum_on_promotion.assert_called_once()

    @pytest.mark.asyncio
    async def test_triggers_model_sync_when_promoted(self, orchestrator):
        """Test model sync is triggered when promoted."""
        result = MockStageCompletionResult(
            iteration=6,
            promoted=True,
        )

        await orchestrator._on_promotion_complete(result)

        orchestrator._trigger_model_sync_after_promotion.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_model_sync_when_not_promoted(self, orchestrator):
        """Test model sync is skipped when not promoted."""
        result = MockStageCompletionResult(
            iteration=6,
            promoted=False,
        )

        await orchestrator._on_promotion_complete(result)

        orchestrator._trigger_model_sync_after_promotion.assert_not_called()


# ============================================================================
# Test: _on_iteration_complete
# ============================================================================


class TestOnIterationComplete:
    """Tests for _on_iteration_complete handler."""

    @pytest.mark.asyncio
    async def test_moves_record_to_completed_iterations(self, orchestrator):
        """Test iteration record is moved to completed list."""
        # Create an iteration record first
        orchestrator._iteration_records[7] = MockIterationRecord(iteration=7)

        result = MockStageCompletionResult(
            iteration=7,
            success=True,
        )

        await orchestrator._on_iteration_complete(result)

        assert 7 not in orchestrator._iteration_records
        assert len(orchestrator._completed_iterations) == 1
        assert orchestrator._completed_iterations[0].iteration == 7

    @pytest.mark.asyncio
    async def test_records_end_time(self, orchestrator):
        """Test end time is recorded."""
        orchestrator._iteration_records[7] = MockIterationRecord(iteration=7)

        result = MockStageCompletionResult(iteration=7, success=True)

        await orchestrator._on_iteration_complete(result)

        assert orchestrator._completed_iterations[0].end_time is not None

    @pytest.mark.asyncio
    async def test_respects_max_history(self, orchestrator):
        """Test completed iterations list respects max_history limit."""
        orchestrator.max_history = 3

        # Add 5 iterations
        for i in range(5):
            orchestrator._iteration_records[i] = MockIterationRecord(iteration=i)
            result = MockStageCompletionResult(iteration=i, success=True)
            await orchestrator._on_iteration_complete(result)

        assert len(orchestrator._completed_iterations) == 3
        # Should keep the last 3
        assert orchestrator._completed_iterations[-1].iteration == 4
        assert orchestrator._completed_iterations[0].iteration == 2

    @pytest.mark.asyncio
    async def test_transitions_to_idle_for_next_iteration(self, orchestrator):
        """Test transitions to IDLE with incremented iteration."""
        orchestrator._iteration_records[7] = MockIterationRecord(iteration=7)

        result = MockStageCompletionResult(iteration=7, success=True)

        await orchestrator._on_iteration_complete(result)

        assert orchestrator._transitions[0]["stage"].value == "idle"
        assert orchestrator._transitions[0]["iteration"] == 8

    @pytest.mark.asyncio
    async def test_handles_missing_iteration_record(self, orchestrator):
        """Test gracefully handles missing iteration record."""
        result = MockStageCompletionResult(iteration=99, success=True)

        # Should not raise
        await orchestrator._on_iteration_complete(result)

        assert orchestrator._transitions[0]["stage"].value == "idle"


# ============================================================================
# Test: RouterEvent Handling
# ============================================================================


class TestRouterEventHandling:
    """Tests for handling RouterEvent wrappers in stage handlers."""

    @pytest.mark.asyncio
    async def test_handles_router_event_in_selfplay_complete(self, orchestrator):
        """Test _on_selfplay_complete handles RouterEvent correctly."""
        inner = MockStageCompletionResult(
            iteration=1,
            success=True,
            games_generated=200,
        )
        event = MockRouterEvent(event_type="selfplay_complete", stage_result=inner)

        await orchestrator._on_selfplay_complete(event)

        assert orchestrator._total_games == 200
        assert orchestrator._transitions[0]["iteration"] == 1

    @pytest.mark.asyncio
    async def test_handles_router_event_with_payload_only(self, orchestrator):
        """Test handles RouterEvent with payload but no stage_result."""
        event = MockRouterEvent(
            event_type="training_complete",
            payload={
                "iteration": 5,
                "success": True,
                "model_id": "test_model",
                "model_path": "/models/test.pth",
            },
            stage_result=None,
        )

        await orchestrator._on_training_complete(event)

        assert orchestrator._transitions[0]["stage"].value == "evaluation"
        assert orchestrator._transitions[0]["iteration"] == 5
