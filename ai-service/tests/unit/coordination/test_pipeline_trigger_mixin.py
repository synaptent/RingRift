"""Unit tests for PipelineTriggerMixin.

Tests for app/coordination/pipeline_trigger_mixin.py (489 LOC).

This mixin provides stage triggering methods for DataPipelineOrchestrator:
- 5 auto-trigger methods (sync, export, training, evaluation, promotion)
- 2 priority trigger methods (orphan recovery, data regeneration)
- 2 model sync triggers (after evaluation, after promotion)
- 1 curriculum update method (on promotion)

December 28, 2025: Created as part of critical infrastructure testing.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.pipeline_trigger_mixin import PipelineTriggerMixin


# =============================================================================
# Mock Host Class
# =============================================================================


@dataclass
class MockIterationRecord:
    """Mock iteration record for testing."""

    metadata: dict = field(default_factory=dict)
    model_id: str | None = None
    elo_delta: float = 0.0


class MockHostClass(PipelineTriggerMixin):
    """Mock host class that provides required attributes and methods for mixin."""

    def __init__(
        self,
        auto_trigger: bool = True,
        auto_trigger_sync: bool = True,
        auto_trigger_export: bool = True,
        auto_trigger_training: bool = True,
        auto_trigger_evaluation: bool = True,
        auto_trigger_promotion: bool = True,
        board_type: str | None = "hex8",
        num_players: int | None = 2,
        can_auto_trigger_result: bool = True,
    ):
        self.auto_trigger = auto_trigger
        self.auto_trigger_sync = auto_trigger_sync
        self.auto_trigger_export = auto_trigger_export
        self.auto_trigger_training = auto_trigger_training
        self.auto_trigger_evaluation = auto_trigger_evaluation
        self.auto_trigger_promotion = auto_trigger_promotion
        self._current_board_type = board_type
        self._current_num_players = num_players
        self._iteration_records: dict[int, MockIterationRecord] = {}
        self._circuit_breaker = MagicMock()
        self._last_quality_score = 0.75
        self._can_auto_trigger_result = can_auto_trigger_result

        # Track method calls for verification
        self._circuit_successes: list[str] = []
        self._circuit_failures: list[tuple[str, str]] = []

    def _can_auto_trigger(self) -> bool:
        """Mock auto-trigger check."""
        return self._can_auto_trigger_result

    def _get_board_config(
        self, result: Any = None, metadata: dict | None = None
    ) -> tuple[str | None, int | None]:
        """Mock board config retrieval."""
        return (self._current_board_type, self._current_num_players)

    def _record_circuit_success(self, stage: str) -> None:
        """Mock circuit success recording."""
        self._circuit_successes.append(stage)

    def _record_circuit_failure(self, stage: str, error: str) -> None:
        """Mock circuit failure recording."""
        self._circuit_failures.append((stage, error))


@dataclass
class MockTriggerResult:
    """Mock result from PipelineTrigger methods."""

    success: bool
    message: str = "Test message"
    error: str | None = None
    output_path: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class MockPromotionResult:
    """Mock result for promotion operations."""

    promoted: bool = True
    board_type: str | None = "hex8"
    num_players: int | None = 2
    new_elo: float | None = 1600.0
    promotion_reason: str = "test"
    metadata: dict = field(default_factory=dict)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_host():
    """Create a mock host class for testing the mixin."""
    return MockHostClass()


@pytest.fixture
def mock_host_disabled():
    """Create a mock host with auto-trigger disabled."""
    return MockHostClass(can_auto_trigger_result=False)


@pytest.fixture
def mock_host_no_config():
    """Create a mock host with no board config."""
    return MockHostClass(board_type=None, num_players=None)


# =============================================================================
# Auto-Trigger Sync Tests
# =============================================================================


class TestAutoTriggerSync:
    """Tests for _auto_trigger_sync method."""

    @pytest.mark.asyncio
    async def test_sync_skipped_when_auto_trigger_disabled(
        self, mock_host_disabled, caplog
    ):
        """Verify sync is skipped when auto-trigger is disabled."""
        await mock_host_disabled._auto_trigger_sync(iteration=1)

        # No circuit breaker calls should happen
        assert len(mock_host_disabled._circuit_successes) == 0
        assert len(mock_host_disabled._circuit_failures) == 0

    @pytest.mark.asyncio
    async def test_sync_skipped_when_no_board_config(self, mock_host_no_config, caplog):
        """Verify sync is skipped when board config is missing."""
        with caplog.at_level(logging.WARNING):
            await mock_host_no_config._auto_trigger_sync(iteration=1)

        assert "missing board config" in caplog.text

    @pytest.mark.asyncio
    async def test_sync_success_records_circuit_success(self, mock_host):
        """Verify successful sync records circuit success."""
        mock_result = MockTriggerResult(success=True, message="Sync completed")

        with patch(
            "app.coordination.pipeline_triggers.PipelineTrigger"
        ) as MockTrigger:
            mock_trigger = MockTrigger.return_value
            mock_trigger.trigger_sync_after_selfplay = AsyncMock(
                return_value=mock_result
            )

            await mock_host._auto_trigger_sync(iteration=1)

        assert "data_sync" in mock_host._circuit_successes

    @pytest.mark.asyncio
    async def test_sync_failure_records_circuit_failure(self, mock_host):
        """Verify failed sync records circuit failure."""
        mock_result = MockTriggerResult(
            success=False,
            error="Connection failed",
            message="Sync failed"
        )

        with patch(
            "app.coordination.pipeline_triggers.PipelineTrigger"
        ) as MockTrigger:
            mock_trigger = MockTrigger.return_value
            mock_trigger.trigger_sync_after_selfplay = AsyncMock(
                return_value=mock_result
            )

            await mock_host._auto_trigger_sync(iteration=1)

        assert ("data_sync", "Connection failed") in mock_host._circuit_failures

    @pytest.mark.asyncio
    async def test_sync_exception_records_circuit_failure(self, mock_host):
        """Verify exception during sync records circuit failure."""
        with patch(
            "app.coordination.pipeline_triggers.PipelineTrigger"
        ) as MockTrigger:
            mock_trigger = MockTrigger.return_value
            mock_trigger.trigger_sync_after_selfplay = AsyncMock(
                side_effect=RuntimeError("Network error")
            )

            await mock_host._auto_trigger_sync(iteration=1)

        assert len(mock_host._circuit_failures) == 1
        assert mock_host._circuit_failures[0][0] == "data_sync"
        assert "Network error" in mock_host._circuit_failures[0][1]


# =============================================================================
# Auto-Trigger Export Tests
# =============================================================================


class TestAutoTriggerExport:
    """Tests for _auto_trigger_export method."""

    @pytest.mark.asyncio
    async def test_export_skipped_when_auto_trigger_disabled(self, mock_host_disabled):
        """Verify export is skipped when auto-trigger is disabled."""
        await mock_host_disabled._auto_trigger_export(iteration=1)

        assert len(mock_host_disabled._circuit_successes) == 0

    @pytest.mark.asyncio
    async def test_export_success_stores_npz_path(self, mock_host):
        """Verify successful export stores NPZ path in iteration records."""
        mock_result = MockTriggerResult(
            success=True,
            output_path="/path/to/training.npz"
        )
        mock_host._iteration_records[1] = MockIterationRecord()

        with patch(
            "app.coordination.pipeline_triggers.PipelineTrigger"
        ) as MockTrigger:
            mock_trigger = MockTrigger.return_value
            mock_trigger.trigger_export_after_sync = AsyncMock(
                return_value=mock_result
            )

            await mock_host._auto_trigger_export(iteration=1)

        assert "npz_export" in mock_host._circuit_successes
        assert mock_host._iteration_records[1].metadata.get("npz_path") == "/path/to/training.npz"

    @pytest.mark.asyncio
    async def test_export_failure_records_circuit_failure(self, mock_host):
        """Verify failed export records circuit failure."""
        mock_result = MockTriggerResult(
            success=False,
            error="Export error"
        )

        with patch(
            "app.coordination.pipeline_triggers.PipelineTrigger"
        ) as MockTrigger:
            mock_trigger = MockTrigger.return_value
            mock_trigger.trigger_export_after_sync = AsyncMock(
                return_value=mock_result
            )

            await mock_host._auto_trigger_export(iteration=1)

        assert ("npz_export", "Export error") in mock_host._circuit_failures


# =============================================================================
# Auto-Trigger Training Tests
# =============================================================================


class TestAutoTriggerTraining:
    """Tests for _auto_trigger_training method."""

    @pytest.mark.asyncio
    async def test_training_skipped_when_auto_trigger_disabled(self, mock_host_disabled):
        """Verify training is skipped when auto-trigger is disabled."""
        await mock_host_disabled._auto_trigger_training(
            iteration=1, npz_path="/path/to/data.npz"
        )

        assert len(mock_host_disabled._circuit_successes) == 0

    @pytest.mark.asyncio
    async def test_training_success_stores_model_id(self, mock_host):
        """Verify successful training stores model ID in iteration records."""
        mock_result = MockTriggerResult(
            success=True,
            metadata={"model_id": "model_123"}
        )
        mock_host._iteration_records[1] = MockIterationRecord()

        with patch(
            "app.coordination.pipeline_triggers.PipelineTrigger"
        ) as MockTrigger:
            mock_trigger = MockTrigger.return_value
            mock_trigger.trigger_training_after_export = AsyncMock(
                return_value=mock_result
            )

            await mock_host._auto_trigger_training(
                iteration=1, npz_path="/path/to/data.npz"
            )

        assert "training" in mock_host._circuit_successes
        assert mock_host._iteration_records[1].model_id == "model_123"

    @pytest.mark.asyncio
    async def test_training_exception_records_failure(self, mock_host):
        """Verify exception during training records circuit failure."""
        with patch(
            "app.coordination.pipeline_triggers.PipelineTrigger"
        ) as MockTrigger:
            mock_trigger = MockTrigger.return_value
            mock_trigger.trigger_training_after_export = AsyncMock(
                side_effect=ValueError("Invalid parameters")
            )

            await mock_host._auto_trigger_training(
                iteration=1, npz_path="/path/to/data.npz"
            )

        assert len(mock_host._circuit_failures) == 1
        assert mock_host._circuit_failures[0][0] == "training"


# =============================================================================
# Auto-Trigger Evaluation Tests
# =============================================================================


class TestAutoTriggerEvaluation:
    """Tests for _auto_trigger_evaluation method."""

    @pytest.mark.asyncio
    async def test_evaluation_skipped_when_disabled(self, mock_host_disabled):
        """Verify evaluation is skipped when auto-trigger is disabled."""
        await mock_host_disabled._auto_trigger_evaluation(
            iteration=1, model_path="/path/to/model.pth"
        )

        assert len(mock_host_disabled._circuit_successes) == 0

    @pytest.mark.asyncio
    async def test_evaluation_success_stores_elo_delta(self, mock_host):
        """Verify successful evaluation stores Elo delta."""
        mock_result = MockTriggerResult(
            success=True,
            metadata={"elo_delta": 50.5}
        )
        mock_host._iteration_records[1] = MockIterationRecord()

        with patch(
            "app.coordination.pipeline_triggers.get_pipeline_trigger"
        ) as mock_get_trigger:
            mock_trigger = MagicMock()
            mock_trigger.trigger_evaluation_after_training = AsyncMock(
                return_value=mock_result
            )
            mock_get_trigger.return_value = mock_trigger

            await mock_host._auto_trigger_evaluation(
                iteration=1, model_path="/path/to/model.pth"
            )

        assert "evaluation" in mock_host._circuit_successes
        assert mock_host._iteration_records[1].elo_delta == 50.5

    @pytest.mark.asyncio
    async def test_evaluation_failure_records_circuit_failure(self, mock_host):
        """Verify failed evaluation records circuit failure."""
        mock_result = MockTriggerResult(
            success=False,
            error="Gauntlet failed"
        )

        with patch(
            "app.coordination.pipeline_triggers.get_pipeline_trigger"
        ) as mock_get_trigger:
            mock_trigger = MagicMock()
            mock_trigger.trigger_evaluation_after_training = AsyncMock(
                return_value=mock_result
            )
            mock_get_trigger.return_value = mock_trigger

            await mock_host._auto_trigger_evaluation(
                iteration=1, model_path="/path/to/model.pth"
            )

        assert ("evaluation", "Gauntlet failed") in mock_host._circuit_failures


# =============================================================================
# Auto-Trigger Promotion Tests
# =============================================================================


class TestAutoTriggerPromotion:
    """Tests for _auto_trigger_promotion method."""

    @pytest.mark.asyncio
    async def test_promotion_skipped_when_disabled(self, mock_host_disabled):
        """Verify promotion is skipped when auto-trigger is disabled."""
        await mock_host_disabled._auto_trigger_promotion(
            iteration=1,
            model_path="/path/to/model.pth",
            gauntlet_results={"win_rates": {"random": 0.9, "heuristic": 0.7}}
        )

        assert len(mock_host_disabled._circuit_successes) == 0

    @pytest.mark.asyncio
    async def test_promotion_success_records_circuit_success(self, mock_host):
        """Verify successful promotion records circuit success."""
        mock_result = MockTriggerResult(success=True)

        with patch(
            "app.coordination.pipeline_triggers.get_pipeline_trigger"
        ) as mock_get_trigger:
            mock_trigger = MagicMock()
            mock_trigger.trigger_promotion_after_evaluation = AsyncMock(
                return_value=mock_result
            )
            mock_get_trigger.return_value = mock_trigger

            await mock_host._auto_trigger_promotion(
                iteration=1,
                model_path="/path/to/model.pth",
                gauntlet_results={"win_rates": {"random": 0.9, "heuristic": 0.7}}
            )

        assert "promotion" in mock_host._circuit_successes

    @pytest.mark.asyncio
    async def test_promotion_skipped_does_not_record_failure(self, mock_host, caplog):
        """Verify skipped promotion does not record circuit failure."""
        mock_result = MockTriggerResult(
            success=False,
            metadata={"reason": "Below threshold"}
        )

        with patch(
            "app.coordination.pipeline_triggers.get_pipeline_trigger"
        ) as mock_get_trigger:
            mock_trigger = MagicMock()
            mock_trigger.trigger_promotion_after_evaluation = AsyncMock(
                return_value=mock_result
            )
            mock_get_trigger.return_value = mock_trigger

            with caplog.at_level(logging.INFO):
                await mock_host._auto_trigger_promotion(
                    iteration=1,
                    model_path="/path/to/model.pth",
                    gauntlet_results={}
                )

        # Promotion failure is NOT a circuit-breaking event
        assert len(mock_host._circuit_failures) == 0
        assert "Promotion skipped" in caplog.text

    @pytest.mark.asyncio
    async def test_promotion_extracts_win_rates_correctly(self, mock_host):
        """Verify promotion correctly extracts win rates from gauntlet results."""
        mock_result = MockTriggerResult(success=True)

        with patch(
            "app.coordination.pipeline_triggers.get_pipeline_trigger"
        ) as mock_get_trigger:
            mock_trigger = MagicMock()
            mock_trigger.trigger_promotion_after_evaluation = AsyncMock(
                return_value=mock_result
            )
            mock_get_trigger.return_value = mock_trigger

            # Test with nested win_rates structure
            await mock_host._auto_trigger_promotion(
                iteration=1,
                model_path="/path/to/model.pth",
                gauntlet_results={"win_rates": {"random": 0.95, "heuristic": 0.80}}
            )

            # Verify the trigger was called with extracted win rates
            call_kwargs = mock_trigger.trigger_promotion_after_evaluation.call_args.kwargs
            assert call_kwargs["win_rate_vs_random"] == 0.95
            assert call_kwargs["win_rate_vs_heuristic"] == 0.80

    @pytest.mark.asyncio
    async def test_promotion_uses_fallback_win_rate_keys(self, mock_host):
        """Verify promotion falls back to legacy win rate keys."""
        mock_result = MockTriggerResult(success=True)

        with patch(
            "app.coordination.pipeline_triggers.get_pipeline_trigger"
        ) as mock_get_trigger:
            mock_trigger = MagicMock()
            mock_trigger.trigger_promotion_after_evaluation = AsyncMock(
                return_value=mock_result
            )
            mock_get_trigger.return_value = mock_trigger

            # Test with legacy flat structure
            await mock_host._auto_trigger_promotion(
                iteration=1,
                model_path="/path/to/model.pth",
                gauntlet_results={
                    "win_rate_vs_random": 0.88,
                    "win_rate_vs_heuristic": 0.65
                }
            )

            call_kwargs = mock_trigger.trigger_promotion_after_evaluation.call_args.kwargs
            assert call_kwargs["win_rate_vs_random"] == 0.88
            assert call_kwargs["win_rate_vs_heuristic"] == 0.65


# =============================================================================
# Orphan Recovery Sync Tests
# =============================================================================


class TestTriggerOrphanRecoverySync:
    """Tests for _trigger_orphan_recovery_sync method."""

    @pytest.mark.asyncio
    async def test_orphan_sync_success(self, mock_host):
        """Verify successful orphan sync returns True."""
        with patch(
            "app.coordination.sync_facade.get_sync_facade"
        ) as mock_get_facade:
            mock_facade = MagicMock()
            mock_facade.trigger_priority_sync = AsyncMock()
            mock_get_facade.return_value = mock_facade

            result = await mock_host._trigger_orphan_recovery_sync(
                source_node="worker-1",
                config_key="hex8_2p",
                orphan_count=10
            )

        assert result is True
        mock_facade.trigger_priority_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_orphan_sync_returns_false_when_facade_unavailable(
        self, mock_host, caplog
    ):
        """Verify orphan sync returns False when facade is unavailable."""
        with patch(
            "app.coordination.sync_facade.get_sync_facade"
        ) as mock_get_facade:
            mock_get_facade.return_value = None

            with caplog.at_level(logging.WARNING):
                result = await mock_host._trigger_orphan_recovery_sync(
                    source_node="worker-1",
                    config_key="hex8_2p",
                    orphan_count=5
                )

        assert result is False
        assert "SyncFacade not available" in caplog.text

    @pytest.mark.asyncio
    async def test_orphan_sync_retries_on_connection_error(self, mock_host):
        """Verify orphan sync retries on connection errors."""
        call_count = 0

        async def mock_sync(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network unreachable")
            return None

        with patch(
            "app.coordination.sync_facade.get_sync_facade"
        ) as mock_get_facade:
            mock_facade = MagicMock()
            mock_facade.trigger_priority_sync = mock_sync
            mock_get_facade.return_value = mock_facade

            result = await mock_host._trigger_orphan_recovery_sync(
                source_node="worker-1",
                config_key="hex8_2p",
                orphan_count=10
            )

        assert result is True
        assert call_count == 3  # Failed twice, succeeded on third

    @pytest.mark.asyncio
    async def test_orphan_sync_fails_after_max_retries(self, mock_host, caplog):
        """Verify orphan sync fails after exhausting retries."""
        with patch(
            "app.coordination.sync_facade.get_sync_facade"
        ) as mock_get_facade:
            mock_facade = MagicMock()
            mock_facade.trigger_priority_sync = AsyncMock(
                side_effect=TimeoutError("Request timed out")
            )
            mock_get_facade.return_value = mock_facade

            with caplog.at_level(logging.ERROR):
                result = await mock_host._trigger_orphan_recovery_sync(
                    source_node="worker-1",
                    config_key="hex8_2p",
                    orphan_count=10
                )

        assert result is False
        assert "after" in caplog.text and "attempts" in caplog.text

    @pytest.mark.asyncio
    async def test_orphan_sync_no_retry_on_import_error(self, mock_host):
        """Verify orphan sync does not retry on ImportError."""
        with patch(
            "app.coordination.sync_facade.get_sync_facade",
            side_effect=ImportError("Module not found")
        ):
            result = await mock_host._trigger_orphan_recovery_sync(
                source_node="worker-1",
                config_key="hex8_2p",
                orphan_count=10
            )

        assert result is False


# =============================================================================
# Data Regeneration Tests
# =============================================================================


class TestTriggerDataRegeneration:
    """Tests for _trigger_data_regeneration method."""

    @pytest.mark.asyncio
    async def test_data_regeneration_publishes_event(self, mock_host, caplog):
        """Verify data regeneration publishes SELFPLAY_TARGET_UPDATED event."""
        with patch(
            "app.coordination.event_router.publish",
            new_callable=AsyncMock,
        ) as mock_publish:
            with caplog.at_level(logging.INFO):
                await mock_host._trigger_data_regeneration(
                    board_type="hex8",
                    num_players=2,
                    iteration=5
                )

        mock_publish.assert_awaited_once()
        call_kwargs = mock_publish.await_args.kwargs
        assert call_kwargs["payload"]["extra_games"] == 2000
        assert call_kwargs["payload"]["reason"] == "quality_gate_failed"
        assert call_kwargs["payload"]["quality_score"] == 0.75
        assert "Triggered data regeneration" in caplog.text

    @pytest.mark.asyncio
    async def test_data_regeneration_handles_publish_exception(self, mock_host, caplog):
        """Verify data regeneration logs warning on publish failure."""
        with patch(
            "app.coordination.event_router.publish",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Publish failed"),
        ):
            with caplog.at_level(logging.WARNING):
                await mock_host._trigger_data_regeneration(
                    board_type="square8",
                    num_players=4,
                    iteration=10
                )

        assert "Failed to trigger data regeneration" in caplog.text


# =============================================================================
# Model Sync After Evaluation Tests
# =============================================================================


class TestTriggerModelSyncAfterEvaluation:
    """Tests for _trigger_model_sync_after_evaluation method."""

    @pytest.mark.asyncio
    async def test_model_sync_after_eval_triggers_priority_sync(
        self, mock_host, caplog
    ):
        """Verify model sync after evaluation triggers priority sync."""
        mock_result = MockPromotionResult()

        with patch(
            "app.coordination.sync_facade.get_sync_facade"
        ) as mock_get_facade:
            mock_facade = MagicMock()
            mock_facade.trigger_priority_sync = AsyncMock()
            mock_get_facade.return_value = mock_facade

            with caplog.at_level(logging.INFO):
                await mock_host._trigger_model_sync_after_evaluation(mock_result)

        mock_facade.trigger_priority_sync.assert_called_once()
        call_kwargs = mock_facade.trigger_priority_sync.call_args.kwargs
        assert call_kwargs["reason"] == "post_evaluation_sync"
        assert call_kwargs["data_type"] == "models"
        assert "Triggering model sync after evaluation" in caplog.text

    @pytest.mark.asyncio
    async def test_model_sync_after_eval_uses_result_metadata(self, mock_host):
        """Verify model sync uses config_key from result metadata."""
        mock_result = MockPromotionResult(
            metadata={"config_key": "square8_4p"}
        )

        with patch(
            "app.coordination.sync_facade.get_sync_facade"
        ) as mock_get_facade:
            mock_facade = MagicMock()
            mock_facade.trigger_priority_sync = AsyncMock()
            mock_get_facade.return_value = mock_facade

            await mock_host._trigger_model_sync_after_evaluation(mock_result)

        call_kwargs = mock_facade.trigger_priority_sync.call_args.kwargs
        assert call_kwargs["config_key"] == "square8_4p"

    @pytest.mark.asyncio
    async def test_model_sync_after_eval_handles_import_error(self, mock_host, caplog):
        """Verify model sync handles ImportError gracefully."""
        mock_result = MockPromotionResult()

        with patch(
            "app.coordination.sync_facade.get_sync_facade",
            side_effect=ImportError("sync_facade not available")
        ):
            with caplog.at_level(logging.DEBUG):
                await mock_host._trigger_model_sync_after_evaluation(mock_result)

        assert "sync_facade not available" in caplog.text


# =============================================================================
# Model Sync After Promotion Tests
# =============================================================================


class TestTriggerModelSyncAfterPromotion:
    """Tests for _trigger_model_sync_after_promotion method."""

    @pytest.mark.asyncio
    async def test_model_sync_after_promotion_triggers_sync(self, mock_host, caplog):
        """Verify model sync after promotion triggers priority sync."""
        mock_result = MockPromotionResult(
            board_type="hex8",
            num_players=2
        )

        with patch(
            "app.coordination.sync_facade.get_sync_facade"
        ) as mock_get_facade:
            mock_facade = MagicMock()
            mock_facade.trigger_priority_sync = AsyncMock()
            mock_get_facade.return_value = mock_facade

            with caplog.at_level(logging.INFO):
                await mock_host._trigger_model_sync_after_promotion(mock_result)

        call_kwargs = mock_facade.trigger_priority_sync.call_args.kwargs
        assert call_kwargs["reason"] == "post_promotion_sync"
        assert call_kwargs["config_key"] == "hex8_2p"

    @pytest.mark.asyncio
    async def test_model_sync_after_promotion_falls_back_to_tracked_config(
        self, mock_host
    ):
        """Verify model sync falls back to tracked board config."""
        mock_result = MockPromotionResult(
            board_type=None,
            num_players=None,
            metadata={}
        )
        mock_host._current_board_type = "square19"
        mock_host._current_num_players = 3

        with patch(
            "app.coordination.sync_facade.get_sync_facade"
        ) as mock_get_facade:
            mock_facade = MagicMock()
            mock_facade.trigger_priority_sync = AsyncMock()
            mock_get_facade.return_value = mock_facade

            await mock_host._trigger_model_sync_after_promotion(mock_result)

        call_kwargs = mock_facade.trigger_priority_sync.call_args.kwargs
        assert call_kwargs["config_key"] == "square19_3p"

    @pytest.mark.asyncio
    async def test_model_sync_after_promotion_handles_exception(
        self, mock_host, caplog
    ):
        """Verify model sync handles exception gracefully."""
        mock_result = MockPromotionResult()

        with patch(
            "app.coordination.sync_facade.get_sync_facade"
        ) as mock_get_facade:
            mock_facade = MagicMock()
            mock_facade.trigger_priority_sync = AsyncMock(
                side_effect=RuntimeError("Sync failed")
            )
            mock_get_facade.return_value = mock_facade

            with caplog.at_level(logging.WARNING):
                await mock_host._trigger_model_sync_after_promotion(mock_result)

        assert "Model sync after promotion failed" in caplog.text


# =============================================================================
# Curriculum Update on Promotion Tests
# =============================================================================


class TestUpdateCurriculumOnPromotion:
    """Tests for _update_curriculum_on_promotion method."""

    @pytest.mark.asyncio
    async def test_curriculum_update_records_promotion(self, mock_host, caplog):
        """Verify curriculum update records promotion event."""
        mock_result = MockPromotionResult(
            promoted=True,
            board_type="hex8",
            num_players=2,
            new_elo=1700.0,
            promotion_reason="win_rate_exceeded"
        )

        with patch(
            "app.training.curriculum_feedback.get_curriculum_feedback"
        ) as mock_get_feedback:
            mock_feedback = MagicMock()
            mock_get_feedback.return_value = mock_feedback

            with caplog.at_level(logging.INFO):
                await mock_host._update_curriculum_on_promotion(mock_result)

        mock_feedback.record_promotion.assert_called_once_with(
            config_key="hex8_2p",
            promoted=True,
            new_elo=1700.0,
            promotion_reason="win_rate_exceeded"
        )
        assert "Curriculum updated for hex8_2p" in caplog.text

    @pytest.mark.asyncio
    async def test_curriculum_update_skips_without_config_key(
        self, mock_host, caplog
    ):
        """Verify curriculum update skips when no config key available."""
        mock_result = MockPromotionResult(
            board_type=None,
            num_players=None,
            metadata={}
        )
        mock_host._current_board_type = None
        mock_host._current_num_players = None

        with patch(
            "app.training.curriculum_feedback.get_curriculum_feedback"
        ) as mock_get_feedback:
            mock_feedback = MagicMock()
            mock_get_feedback.return_value = mock_feedback

            with caplog.at_level(logging.DEBUG):
                await mock_host._update_curriculum_on_promotion(mock_result)

        mock_feedback.record_promotion.assert_not_called()
        assert "No config_key for curriculum update" in caplog.text

    @pytest.mark.asyncio
    async def test_curriculum_update_uses_metadata_fallback(self, mock_host):
        """Verify curriculum update uses metadata for Elo and reason."""
        mock_result = MockPromotionResult(
            promoted=True,
            board_type="square8",
            num_players=4,
            new_elo=None,
            promotion_reason="",
            metadata={"new_elo": 1650.0, "reason": "threshold_met"}
        )

        with patch(
            "app.training.curriculum_feedback.get_curriculum_feedback"
        ) as mock_get_feedback:
            mock_feedback = MagicMock()
            mock_get_feedback.return_value = mock_feedback

            await mock_host._update_curriculum_on_promotion(mock_result)

        mock_feedback.record_promotion.assert_called_once()
        call_kwargs = mock_feedback.record_promotion.call_args.kwargs
        assert call_kwargs["new_elo"] == 1650.0
        assert call_kwargs["promotion_reason"] == "threshold_met"

    @pytest.mark.asyncio
    async def test_curriculum_update_handles_import_error(self, mock_host, caplog):
        """Verify curriculum update handles ImportError gracefully."""
        mock_result = MockPromotionResult()

        with patch(
            "app.training.curriculum_feedback.get_curriculum_feedback",
            side_effect=ImportError("curriculum_feedback not available")
        ):
            with caplog.at_level(logging.DEBUG):
                await mock_host._update_curriculum_on_promotion(mock_result)

        assert "curriculum_feedback not available" in caplog.text

    @pytest.mark.asyncio
    async def test_curriculum_update_handles_exception(self, mock_host, caplog):
        """Verify curriculum update logs warning on exception."""
        mock_result = MockPromotionResult()

        with patch(
            "app.training.curriculum_feedback.get_curriculum_feedback"
        ) as mock_get_feedback:
            mock_feedback = MagicMock()
            mock_feedback.record_promotion.side_effect = RuntimeError("DB error")
            mock_get_feedback.return_value = mock_feedback

            with caplog.at_level(logging.WARNING):
                await mock_host._update_curriculum_on_promotion(mock_result)

        assert "Curriculum update failed" in caplog.text


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================


class TestEdgeCases:
    """Edge case tests for PipelineTriggerMixin."""

    @pytest.mark.asyncio
    async def test_sync_with_no_iteration_records(self, mock_host):
        """Verify sync works when iteration record doesn't exist."""
        mock_result = MockTriggerResult(success=True, output_path="/path/to/npz")

        with patch(
            "app.coordination.pipeline_triggers.PipelineTrigger"
        ) as MockTrigger:
            mock_trigger = MockTrigger.return_value
            mock_trigger.trigger_export_after_sync = AsyncMock(
                return_value=mock_result
            )

            # No iteration record for iteration 99
            await mock_host._auto_trigger_export(iteration=99)

        # Should still record success
        assert "npz_export" in mock_host._circuit_successes
        # No metadata stored since no iteration record
        assert 99 not in mock_host._iteration_records

    @pytest.mark.asyncio
    async def test_result_with_none_metadata(self, mock_host):
        """Verify methods handle results with None metadata."""
        mock_result = MagicMock()
        mock_result.metadata = None
        mock_result.board_type = "hex8"
        mock_result.num_players = 2
        mock_result.promoted = True
        mock_result.new_elo = 1500.0
        mock_result.promotion_reason = "test"

        with patch(
            "app.training.curriculum_feedback.get_curriculum_feedback"
        ) as mock_get_feedback:
            mock_feedback = MagicMock()
            mock_get_feedback.return_value = mock_feedback

            # Should not raise despite None metadata
            await mock_host._update_curriculum_on_promotion(mock_result)

        mock_feedback.record_promotion.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_triggers(self, mock_host):
        """Verify concurrent trigger calls don't interfere."""
        mock_result = MockTriggerResult(success=True)

        async def mock_slow_sync(*args, **kwargs):
            await asyncio.sleep(0.01)
            return mock_result

        with patch(
            "app.coordination.pipeline_triggers.PipelineTrigger"
        ) as MockTrigger:
            mock_trigger = MockTrigger.return_value
            mock_trigger.trigger_sync_after_selfplay = mock_slow_sync

            # Run multiple syncs concurrently
            await asyncio.gather(
                mock_host._auto_trigger_sync(iteration=1),
                mock_host._auto_trigger_sync(iteration=2),
                mock_host._auto_trigger_sync(iteration=3),
            )

        # All should record success
        assert mock_host._circuit_successes.count("data_sync") == 3
