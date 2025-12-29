"""Unit tests for training event integration module.

Tests for event publishing, subscription, and the various event dataclasses.
December 2025: Created to improve training module test coverage.
"""

from __future__ import annotations

import asyncio
from dataclasses import fields
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Test TrainingTopics Constants
# =============================================================================


class TestTrainingTopics:
    """Tests for TrainingTopics constants."""

    def test_training_lifecycle_topics(self) -> None:
        """Test training lifecycle topic constants."""
        from app.training.event_integration import TrainingTopics

        assert TrainingTopics.TRAINING_STARTED == "training.started"
        assert TrainingTopics.TRAINING_COMPLETED == "training.completed"
        assert TrainingTopics.TRAINING_FAILED == "training.failed"
        assert TrainingTopics.EPOCH_COMPLETED == "training.epoch.completed"
        assert TrainingTopics.STEP_COMPLETED == "training.step.completed"

    def test_evaluation_topics(self) -> None:
        """Test evaluation topic constants."""
        from app.training.event_integration import TrainingTopics

        assert TrainingTopics.EVAL_COMPLETED == "training.eval.completed"
        assert TrainingTopics.ELO_CHANGED == "training.eval.elo_changed"
        assert TrainingTopics.BASELINE_GATING_FAILED == "training.eval.gating_failed"

    def test_checkpoint_topics(self) -> None:
        """Test checkpoint topic constants."""
        from app.training.event_integration import TrainingTopics

        assert TrainingTopics.CHECKPOINT_SAVED == "training.checkpoint.saved"
        assert TrainingTopics.CHECKPOINT_LOADED == "training.checkpoint.loaded"
        assert TrainingTopics.CHECKPOINT_ERROR == "training.checkpoint.error"

    def test_model_topics(self) -> None:
        """Test model promotion topic constants."""
        from app.training.event_integration import TrainingTopics

        assert TrainingTopics.MODEL_PROMOTED == "training.model.promoted"
        assert TrainingTopics.MODEL_ROLLED_BACK == "training.model.rolled_back"

    def test_selfplay_topics(self) -> None:
        """Test selfplay topic constants."""
        from app.training.event_integration import TrainingTopics

        assert TrainingTopics.SELFPLAY_STARTED == "training.selfplay.started"
        assert TrainingTopics.SELFPLAY_COMPLETED == "training.selfplay.completed"
        assert TrainingTopics.SELFPLAY_FAILED == "training.selfplay.failed"

    def test_composite_elo_topics(self) -> None:
        """Test composite ELO system topics."""
        from app.training.event_integration import TrainingTopics

        assert TrainingTopics.COMPOSITE_ELO_UPDATED == "training.composite.elo_updated"
        assert TrainingTopics.COMPOSITE_GAUNTLET_COMPLETED == "training.composite.gauntlet_completed"
        assert TrainingTopics.COMPOSITE_NN_CULLED == "training.composite.nn_culled"
        assert TrainingTopics.COMPOSITE_ALGORITHM_RANKING == "training.composite.algorithm_ranking"
        assert TrainingTopics.COMPOSITE_CONSISTENCY_CHECK == "training.composite.consistency_check"

    def test_pattern_topics(self) -> None:
        """Test wildcard pattern topics."""
        from app.training.event_integration import TrainingTopics

        assert TrainingTopics.ALL_TRAINING == "training.*"
        assert TrainingTopics.ALL_EVAL == "training.eval.*"
        assert TrainingTopics.ALL_CHECKPOINT == "training.checkpoint.*"
        assert TrainingTopics.ALL_MODEL == "training.model.*"
        assert TrainingTopics.ALL_SELFPLAY == "training.selfplay.*"
        assert TrainingTopics.ALL_COMPOSITE == "training.composite.*"


# =============================================================================
# Test Base Event Types
# =============================================================================


class TestTrainingEvent:
    """Tests for TrainingEvent dataclass."""

    def test_default_values(self) -> None:
        """Test default field values."""
        from app.training.event_integration import TrainingEvent

        event = TrainingEvent()
        assert event.config_key == ""
        assert event.job_id == ""

    def test_custom_values(self) -> None:
        """Test custom field values."""
        from app.training.event_integration import TrainingEvent

        event = TrainingEvent(
            config_key="hex8_2p",
            job_id="job-123",
        )
        assert event.config_key == "hex8_2p"
        assert event.job_id == "job-123"


class TestEvaluationEvent:
    """Tests for EvaluationEvent dataclass."""

    def test_inherits_from_training_event(self) -> None:
        """Test EvaluationEvent inherits from TrainingEvent."""
        from app.training.event_integration import EvaluationEvent, TrainingEvent

        assert issubclass(EvaluationEvent, TrainingEvent)

    def test_default_values(self) -> None:
        """Test default field values."""
        from app.training.event_integration import EvaluationEvent

        event = EvaluationEvent()
        assert event.eval_step == 0
        assert event.elo == 0.0
        assert event.config_key == ""

    def test_custom_values(self) -> None:
        """Test custom field values."""
        from app.training.event_integration import EvaluationEvent

        event = EvaluationEvent(
            config_key="square8_2p",
            eval_step=100,
            elo=1500.0,
        )
        assert event.config_key == "square8_2p"
        assert event.eval_step == 100
        assert event.elo == 1500.0


class TestCheckpointEvent:
    """Tests for CheckpointEvent dataclass."""

    def test_inherits_from_training_event(self) -> None:
        """Test CheckpointEvent inherits from TrainingEvent."""
        from app.training.event_integration import CheckpointEvent, TrainingEvent

        assert issubclass(CheckpointEvent, TrainingEvent)

    def test_default_values(self) -> None:
        """Test default field values."""
        from app.training.event_integration import CheckpointEvent

        event = CheckpointEvent()
        assert event.checkpoint_path == ""
        assert event.step == 0


class TestSelfplayEvent:
    """Tests for SelfplayEvent dataclass."""

    def test_inherits_from_training_event(self) -> None:
        """Test SelfplayEvent inherits from TrainingEvent."""
        from app.training.event_integration import SelfplayEvent, TrainingEvent

        assert issubclass(SelfplayEvent, TrainingEvent)

    def test_default_values(self) -> None:
        """Test default field values."""
        from app.training.event_integration import SelfplayEvent

        event = SelfplayEvent()
        assert event.iteration == 0
        assert event.games_count == 0


# =============================================================================
# Test Specific Event Types
# =============================================================================


class TestTrainingStartedEvent:
    """Tests for TrainingStartedEvent dataclass."""

    def test_default_values(self) -> None:
        """Test default field values."""
        from app.training.event_integration import TrainingStartedEvent

        event = TrainingStartedEvent()
        assert event.total_epochs == 0
        assert event.batch_size == 0
        assert event.learning_rate == 0.0

    def test_custom_values(self) -> None:
        """Test custom field values."""
        from app.training.event_integration import TrainingStartedEvent

        event = TrainingStartedEvent(
            config_key="hex8_2p",
            job_id="job-456",
            total_epochs=50,
            batch_size=512,
            learning_rate=0.001,
        )
        assert event.config_key == "hex8_2p"
        assert event.total_epochs == 50
        assert event.batch_size == 512
        assert event.learning_rate == 0.001


class TestTrainingCompletedEvent:
    """Tests for TrainingCompletedEvent dataclass."""

    def test_default_values(self) -> None:
        """Test default field values."""
        from app.training.event_integration import TrainingCompletedEvent

        event = TrainingCompletedEvent()
        assert event.epochs_completed == 0
        assert event.final_loss == 0.0
        assert event.final_elo == 0.0
        assert event.duration_seconds == 0.0

    def test_custom_values(self) -> None:
        """Test custom field values."""
        from app.training.event_integration import TrainingCompletedEvent

        event = TrainingCompletedEvent(
            config_key="hex8_2p",
            epochs_completed=50,
            final_loss=0.123,
            final_elo=1650.0,
            duration_seconds=3600.0,
        )
        assert event.epochs_completed == 50
        assert event.final_loss == 0.123
        assert event.final_elo == 1650.0
        assert event.duration_seconds == 3600.0


class TestTrainingFailedEvent:
    """Tests for TrainingFailedEvent dataclass."""

    def test_default_values(self) -> None:
        """Test default field values."""
        from app.training.event_integration import TrainingFailedEvent

        event = TrainingFailedEvent()
        assert event.error_type == ""
        assert event.error_message == ""
        assert event.epoch == 0
        assert event.step == 0

    def test_custom_values(self) -> None:
        """Test custom field values."""
        from app.training.event_integration import TrainingFailedEvent

        event = TrainingFailedEvent(
            config_key="hex8_2p",
            error_type="RuntimeError",
            error_message="CUDA out of memory",
            epoch=10,
            step=5000,
        )
        assert event.error_type == "RuntimeError"
        assert event.error_message == "CUDA out of memory"
        assert event.epoch == 10
        assert event.step == 5000


class TestEpochCompletedEvent:
    """Tests for EpochCompletedEvent dataclass."""

    def test_default_values(self) -> None:
        """Test default field values."""
        from app.training.event_integration import EpochCompletedEvent

        event = EpochCompletedEvent()
        assert event.epoch == 0
        assert event.total_epochs == 0
        assert event.train_loss == 0.0
        assert event.val_loss is None
        assert event.learning_rate == 0.0

    def test_custom_values(self) -> None:
        """Test custom field values."""
        from app.training.event_integration import EpochCompletedEvent

        event = EpochCompletedEvent(
            epoch=5,
            total_epochs=50,
            train_loss=0.5,
            val_loss=0.6,
            learning_rate=0.001,
        )
        assert event.epoch == 5
        assert event.total_epochs == 50
        assert event.train_loss == 0.5
        assert event.val_loss == 0.6


class TestStepCompletedEvent:
    """Tests for StepCompletedEvent dataclass."""

    def test_default_values(self) -> None:
        """Test default field values."""
        from app.training.event_integration import StepCompletedEvent

        event = StepCompletedEvent()
        assert event.step == 0
        assert event.loss == 0.0
        assert event.learning_rate == 0.0
        assert event.samples_per_second == 0.0


class TestEvaluationCompletedEvent:
    """Tests for EvaluationCompletedEvent dataclass."""

    def test_default_values(self) -> None:
        """Test default field values."""
        from app.training.event_integration import EvaluationCompletedEvent

        event = EvaluationCompletedEvent()
        assert event.games_played == 0
        assert event.win_rate == 0.0
        assert event.baseline_results == {}
        assert event.passes_gating is True
        assert event.failed_baselines == []

    def test_custom_values(self) -> None:
        """Test custom field values."""
        from app.training.event_integration import EvaluationCompletedEvent

        event = EvaluationCompletedEvent(
            games_played=100,
            win_rate=0.85,
            baseline_results={"random": 0.95, "heuristic": 0.65},
            passes_gating=True,
            failed_baselines=[],
        )
        assert event.games_played == 100
        assert event.win_rate == 0.85
        assert event.baseline_results == {"random": 0.95, "heuristic": 0.65}


class TestEloChangedEvent:
    """Tests for EloChangedEvent dataclass."""

    def test_default_values(self) -> None:
        """Test default field values."""
        from app.training.event_integration import EloChangedEvent

        event = EloChangedEvent()
        assert event.old_elo == 0.0
        assert event.new_elo == 0.0
        assert event.elo_delta == 0.0
        assert event.is_improvement is False
        assert event.is_drop is False

    def test_custom_values(self) -> None:
        """Test custom field values."""
        from app.training.event_integration import EloChangedEvent

        event = EloChangedEvent(
            old_elo=1500.0,
            new_elo=1550.0,
            elo_delta=50.0,
            is_improvement=True,
            is_drop=False,
        )
        assert event.old_elo == 1500.0
        assert event.new_elo == 1550.0
        assert event.elo_delta == 50.0
        assert event.is_improvement is True


class TestCheckpointSavedEvent:
    """Tests for CheckpointSavedEvent dataclass."""

    def test_default_values(self) -> None:
        """Test default field values."""
        from app.training.event_integration import CheckpointSavedEvent

        event = CheckpointSavedEvent()
        assert event.is_best is False
        assert event.elo_at_save == 0.0
        assert event.metrics == {}


class TestModelPromotedEvent:
    """Tests for ModelPromotedEvent dataclass."""

    def test_default_values(self) -> None:
        """Test default field values."""
        from app.training.event_integration import ModelPromotedEvent

        event = ModelPromotedEvent()
        assert event.model_id == ""
        assert event.from_state == ""
        assert event.to_state == ""
        assert event.promotion_type == ""
        assert event.elo == 0.0


class TestSelfplayStartedEvent:
    """Tests for SelfplayStartedEvent dataclass."""

    def test_default_values(self) -> None:
        """Test default field values."""
        from app.training.event_integration import SelfplayStartedEvent

        event = SelfplayStartedEvent()
        assert event.engine == ""


class TestSelfplayCompletedEvent:
    """Tests for SelfplayCompletedEvent dataclass."""

    def test_default_values(self) -> None:
        """Test default field values."""
        from app.training.event_integration import SelfplayCompletedEvent

        event = SelfplayCompletedEvent()
        assert event.success is True
        assert event.output_path == ""
        assert event.duration_seconds == 0.0


# =============================================================================
# Test Composite ELO Event Types
# =============================================================================


class TestCompositeEloEvent:
    """Tests for CompositeEloEvent dataclass."""

    def test_default_values(self) -> None:
        """Test default field values."""
        from app.training.event_integration import CompositeEloEvent

        event = CompositeEloEvent()
        assert event.board_type == ""
        assert event.num_players == 2


class TestCompositeEloUpdatedEvent:
    """Tests for CompositeEloUpdatedEvent dataclass."""

    def test_default_values(self) -> None:
        """Test default field values."""
        from app.training.event_integration import CompositeEloUpdatedEvent

        event = CompositeEloUpdatedEvent()
        assert event.nn_id == ""
        assert event.ai_type == ""
        assert event.config_hash == ""
        assert event.participant_id == ""
        assert event.old_elo == 0.0
        assert event.new_elo == 0.0
        assert event.elo_delta == 0.0
        assert event.games_played == 0
        assert event.is_improvement is False

    def test_custom_values(self) -> None:
        """Test custom field values."""
        from app.training.event_integration import CompositeEloUpdatedEvent

        event = CompositeEloUpdatedEvent(
            board_type="hex8",
            num_players=2,
            nn_id="nn-123",
            ai_type="gumbel",
            old_elo=1500.0,
            new_elo=1550.0,
            elo_delta=50.0,
            games_played=100,
            is_improvement=True,
        )
        assert event.nn_id == "nn-123"
        assert event.is_improvement is True


class TestCompositeGauntletCompletedEvent:
    """Tests for CompositeGauntletCompletedEvent dataclass."""

    def test_default_values(self) -> None:
        """Test default field values."""
        from app.training.event_integration import CompositeGauntletCompletedEvent

        event = CompositeGauntletCompletedEvent()
        assert event.phase1_nn_count == 0
        assert event.phase1_passed_count == 0
        assert event.phase2_participants == 0
        assert event.total_games_played == 0
        assert event.duration_seconds == 0.0
        assert event.top_nn_ids == []
        assert event.top_algorithm == ""


class TestCompositeNNCulledEvent:
    """Tests for CompositeNNCulledEvent dataclass."""

    def test_default_values(self) -> None:
        """Test default field values."""
        from app.training.event_integration import CompositeNNCulledEvent

        event = CompositeNNCulledEvent()
        assert event.nn_id == ""
        assert event.reason == ""
        assert event.final_elo == 0.0
        assert event.games_played == 0
        assert event.algorithms_tested == []
        assert event.cull_level == 1


class TestCompositeAlgorithmRankingEvent:
    """Tests for CompositeAlgorithmRankingEvent dataclass."""

    def test_default_values(self) -> None:
        """Test default field values."""
        from app.training.event_integration import CompositeAlgorithmRankingEvent

        event = CompositeAlgorithmRankingEvent()
        assert event.rankings == []
        assert event.expected_order_violations == 0
        assert event.top_algorithm == ""
        assert event.nn_count_evaluated == 0


class TestCompositeConsistencyCheckEvent:
    """Tests for CompositeConsistencyCheckEvent dataclass."""

    def test_default_values(self) -> None:
        """Test default field values."""
        from app.training.event_integration import CompositeConsistencyCheckEvent

        event = CompositeConsistencyCheckEvent()
        assert event.overall_healthy is True
        assert event.checks_passed == 0
        assert event.checks_failed == 0
        assert event.warnings_count == 0
        assert event.errors_count == 0
        assert event.check_results == {}


# =============================================================================
# Test Publisher Functions
# =============================================================================


class TestPublishTrainingStarted:
    """Tests for publish_training_started function."""

    @pytest.mark.asyncio
    async def test_publish_training_started(self) -> None:
        """Test publishing training started event."""
        from app.training.event_integration import publish_training_started

        with patch("app.training.event_integration.publish") as mock_publish:
            mock_publish.return_value = 1

            result = await publish_training_started(
                config_key="hex8_2p",
                job_id="job-123",
                total_epochs=50,
                batch_size=512,
                learning_rate=0.001,
            )

            assert result == 1
            mock_publish.assert_called_once()
            event = mock_publish.call_args[0][0]
            assert event.config_key == "hex8_2p"
            assert event.job_id == "job-123"
            assert event.total_epochs == 50


class TestPublishTrainingCompleted:
    """Tests for publish_training_completed function."""

    @pytest.mark.asyncio
    async def test_publish_training_completed(self) -> None:
        """Test publishing training completed event."""
        from app.training.event_integration import publish_training_completed

        with patch("app.training.event_integration.publish") as mock_publish:
            mock_publish.return_value = 1

            result = await publish_training_completed(
                config_key="hex8_2p",
                job_id="job-123",
                epochs_completed=50,
                final_loss=0.123,
                final_elo=1650.0,
                duration_seconds=3600.0,
            )

            assert result == 1
            mock_publish.assert_called_once()
            event = mock_publish.call_args[0][0]
            assert event.epochs_completed == 50
            assert event.final_loss == 0.123


class TestPublishTrainingFailed:
    """Tests for publish_training_failed function."""

    @pytest.mark.asyncio
    async def test_publish_training_failed(self) -> None:
        """Test publishing training failed event."""
        from app.training.event_integration import publish_training_failed

        with patch("app.training.event_integration.publish") as mock_publish:
            mock_publish.return_value = 1

            error = RuntimeError("CUDA out of memory")
            result = await publish_training_failed(
                config_key="hex8_2p",
                error=error,
                job_id="job-123",
                epoch=10,
                step=5000,
            )

            assert result == 1
            mock_publish.assert_called_once()
            event = mock_publish.call_args[0][0]
            assert event.error_type == "RuntimeError"
            assert "CUDA out of memory" in event.error_message


class TestPublishEpochCompleted:
    """Tests for publish_epoch_completed function."""

    @pytest.mark.asyncio
    async def test_publish_epoch_completed(self) -> None:
        """Test publishing epoch completed event."""
        from app.training.event_integration import publish_epoch_completed

        with patch("app.training.event_integration.publish") as mock_publish:
            mock_publish.return_value = 1

            result = await publish_epoch_completed(
                config_key="hex8_2p",
                epoch=5,
                total_epochs=50,
                train_loss=0.5,
                val_loss=0.6,
                learning_rate=0.001,
            )

            assert result == 1
            mock_publish.assert_called_once()


class TestPublishStepCompleted:
    """Tests for publish_step_completed function."""

    @pytest.mark.asyncio
    async def test_publish_step_completed(self) -> None:
        """Test publishing step completed event."""
        from app.training.event_integration import publish_step_completed

        with patch("app.training.event_integration.publish") as mock_publish:
            mock_publish.return_value = 1

            result = await publish_step_completed(
                config_key="hex8_2p",
                step=1000,
                loss=0.5,
                learning_rate=0.001,
                samples_per_second=5000.0,
            )

            assert result == 1
            mock_publish.assert_called_once()


class TestPublishEvaluationCompleted:
    """Tests for publish_evaluation_completed function."""

    @pytest.mark.asyncio
    async def test_publish_evaluation_completed(self) -> None:
        """Test publishing evaluation completed event."""
        from app.training.event_integration import publish_evaluation_completed

        with patch("app.training.event_integration.publish") as mock_publish:
            mock_publish.return_value = 1

            result = await publish_evaluation_completed(
                config_key="hex8_2p",
                eval_step=100,
                elo=1550.0,
                games_played=100,
                win_rate=0.85,
                baseline_results={"random": 0.95},
                passes_gating=True,
            )

            assert result == 1
            mock_publish.assert_called_once()


class TestPublishEloChanged:
    """Tests for publish_elo_changed function."""

    @pytest.mark.asyncio
    async def test_publish_elo_changed_improvement(self) -> None:
        """Test publishing elo changed event for improvement."""
        from app.training.event_integration import publish_elo_changed

        with patch("app.training.event_integration.publish") as mock_publish:
            mock_publish.return_value = 1

            result = await publish_elo_changed(
                config_key="hex8_2p",
                old_elo=1500.0,
                new_elo=1550.0,
            )

            assert result == 1
            mock_publish.assert_called_once()
            event = mock_publish.call_args[0][0]
            assert event.elo_delta == 50.0
            assert event.is_improvement is True
            assert event.is_drop is False

    @pytest.mark.asyncio
    async def test_publish_elo_changed_drop(self) -> None:
        """Test publishing elo changed event for significant drop."""
        from app.training.event_integration import publish_elo_changed

        with patch("app.training.event_integration.publish") as mock_publish:
            mock_publish.return_value = 1

            result = await publish_elo_changed(
                config_key="hex8_2p",
                old_elo=1500.0,
                new_elo=1480.0,  # -20 is significant drop
            )

            assert result == 1
            event = mock_publish.call_args[0][0]
            assert event.elo_delta == -20.0
            assert event.is_improvement is False
            assert event.is_drop is True


class TestPublishCheckpointSaved:
    """Tests for publish_checkpoint_saved function."""

    @pytest.mark.asyncio
    async def test_publish_checkpoint_saved(self) -> None:
        """Test publishing checkpoint saved event."""
        from app.training.event_integration import publish_checkpoint_saved

        with patch("app.training.event_integration.publish") as mock_publish:
            mock_publish.return_value = 1

            result = await publish_checkpoint_saved(
                config_key="hex8_2p",
                checkpoint_path="/models/checkpoint.pth",
                step=1000,
                is_best=True,
                elo_at_save=1550.0,
            )

            assert result == 1
            mock_publish.assert_called_once()


class TestPublishModelPromoted:
    """Tests for publish_model_promoted function."""

    @pytest.mark.asyncio
    async def test_publish_model_promoted(self) -> None:
        """Test publishing model promoted event."""
        from app.training.event_integration import publish_model_promoted

        with patch("app.training.event_integration.publish") as mock_publish:
            mock_publish.return_value = 1

            result = await publish_model_promoted(
                config_key="hex8_2p",
                model_id="model-123",
                from_state="training",
                to_state="production",
                promotion_type="gauntlet_passed",
                elo=1650.0,
            )

            assert result == 1
            mock_publish.assert_called_once()


class TestPublishSelfplayStarted:
    """Tests for publish_selfplay_started function."""

    @pytest.mark.asyncio
    async def test_publish_selfplay_started(self) -> None:
        """Test publishing selfplay started event."""
        from app.training.event_integration import publish_selfplay_started

        with patch("app.training.event_integration.publish") as mock_publish:
            mock_publish.return_value = 1

            result = await publish_selfplay_started(
                config_key="hex8_2p",
                iteration=5,
                games_count=1000,
                engine="gumbel",
            )

            assert result == 1
            mock_publish.assert_called_once()


class TestPublishSelfplayCompleted:
    """Tests for publish_selfplay_completed function."""

    @pytest.mark.asyncio
    async def test_publish_selfplay_completed(self) -> None:
        """Test publishing selfplay completed event."""
        from app.training.event_integration import publish_selfplay_completed

        with patch("app.training.event_integration.publish") as mock_publish:
            mock_publish.return_value = 1

            result = await publish_selfplay_completed(
                config_key="hex8_2p",
                iteration=5,
                games_count=1000,
                success=True,
                output_path="/data/games/selfplay.db",
                duration_seconds=600.0,
            )

            assert result == 1
            mock_publish.assert_called_once()


# =============================================================================
# Test Composite ELO Publishers
# =============================================================================


class TestPublishCompositeEloUpdated:
    """Tests for publish_composite_elo_updated function."""

    @pytest.mark.asyncio
    async def test_publish_composite_elo_updated(self) -> None:
        """Test publishing composite elo updated event."""
        from app.training.event_integration import publish_composite_elo_updated

        with patch("app.training.event_integration.publish") as mock_publish:
            mock_publish.return_value = 1

            result = await publish_composite_elo_updated(
                nn_id="nn-123",
                ai_type="gumbel",
                config_hash="abc123",
                participant_id="nn-123_gumbel",
                old_elo=1500.0,
                new_elo=1550.0,
                games_played=100,
            )

            assert result == 1
            mock_publish.assert_called_once()
            event = mock_publish.call_args[0][0]
            assert event.elo_delta == 50.0
            assert event.is_improvement is True


class TestPublishCompositeGauntletCompleted:
    """Tests for publish_composite_gauntlet_completed function."""

    @pytest.mark.asyncio
    async def test_publish_composite_gauntlet_completed(self) -> None:
        """Test publishing composite gauntlet completed event."""
        from app.training.event_integration import publish_composite_gauntlet_completed

        with patch("app.training.event_integration.publish") as mock_publish:
            mock_publish.return_value = 1

            result = await publish_composite_gauntlet_completed(
                board_type="hex8",
                num_players=2,
                phase1_nn_count=10,
                phase1_passed_count=8,
                phase2_participants=24,
                total_games_played=1000,
                duration_seconds=3600.0,
                top_nn_ids=["nn-1", "nn-2"],
                top_algorithm="gumbel",
            )

            assert result == 1
            mock_publish.assert_called_once()


class TestPublishCompositeNnCulled:
    """Tests for publish_composite_nn_culled function."""

    @pytest.mark.asyncio
    async def test_publish_composite_nn_culled(self) -> None:
        """Test publishing composite NN culled event."""
        from app.training.event_integration import publish_composite_nn_culled

        with patch("app.training.event_integration.publish") as mock_publish:
            mock_publish.return_value = 1

            result = await publish_composite_nn_culled(
                nn_id="nn-old",
                reason="underperforming",
                final_elo=1200.0,
                games_played=500,
                cull_level=1,
                algorithms_tested=["gumbel", "mcts"],
            )

            assert result == 1
            mock_publish.assert_called_once()


class TestPublishCompositeAlgorithmRanking:
    """Tests for publish_composite_algorithm_ranking function."""

    @pytest.mark.asyncio
    async def test_publish_composite_algorithm_ranking(self) -> None:
        """Test publishing algorithm ranking event."""
        from app.training.event_integration import publish_composite_algorithm_ranking

        with patch("app.training.event_integration.publish") as mock_publish:
            mock_publish.return_value = 1

            rankings = [
                {"algorithm": "gumbel", "avg_elo": 1650.0, "games": 500, "rank": 1},
                {"algorithm": "mcts", "avg_elo": 1600.0, "games": 500, "rank": 2},
            ]

            result = await publish_composite_algorithm_ranking(
                rankings=rankings,
                expected_order_violations=2,
                top_algorithm="gumbel",
                nn_count_evaluated=10,
            )

            assert result == 1
            mock_publish.assert_called_once()


class TestPublishCompositeConsistencyCheck:
    """Tests for publish_composite_consistency_check function."""

    @pytest.mark.asyncio
    async def test_publish_composite_consistency_check(self) -> None:
        """Test publishing consistency check event."""
        from app.training.event_integration import publish_composite_consistency_check

        with patch("app.training.event_integration.publish") as mock_publish:
            mock_publish.return_value = 1

            result = await publish_composite_consistency_check(
                overall_healthy=True,
                checks_passed=4,
                checks_failed=0,
                warnings_count=1,
                errors_count=0,
                check_results={"nn_consistency": True, "transitivity": True},
            )

            assert result == 1
            mock_publish.assert_called_once()


# =============================================================================
# Test Synchronous Publishers
# =============================================================================


class TestSyncPublishers:
    """Tests for synchronous publisher functions."""

    def test_publish_composite_elo_updated_sync(self) -> None:
        """Test synchronous composite elo updated publishing."""
        from app.training.event_integration import publish_composite_elo_updated_sync

        with patch("app.training.event_integration.get_event_bus") as mock_bus:
            mock_instance = MagicMock()
            mock_instance.publish_sync.return_value = 1
            mock_bus.return_value = mock_instance

            result = publish_composite_elo_updated_sync(
                nn_id="nn-123",
                ai_type="gumbel",
                config_hash="abc",
                participant_id="nn-123_gumbel",
                old_elo=1500.0,
                new_elo=1550.0,
                games_played=100,
            )

            assert result == 1
            mock_instance.publish_sync.assert_called_once()

    def test_publish_training_started_sync(self) -> None:
        """Test synchronous training started publishing."""
        from app.training.event_integration import publish_training_started_sync

        with patch("app.training.event_integration.get_event_bus") as mock_bus:
            mock_instance = MagicMock()
            mock_instance.publish_sync.return_value = 1
            mock_bus.return_value = mock_instance

            result = publish_training_started_sync(
                config_key="hex8_2p",
                job_id="job-123",
            )

            assert result == 1
            mock_instance.publish_sync.assert_called_once()

    def test_publish_step_completed_sync(self) -> None:
        """Test synchronous step completed publishing."""
        from app.training.event_integration import publish_step_completed_sync

        with patch("app.training.event_integration.get_event_bus") as mock_bus:
            mock_instance = MagicMock()
            mock_instance.publish_sync.return_value = 1
            mock_bus.return_value = mock_instance

            result = publish_step_completed_sync(
                config_key="hex8_2p",
                step=1000,
                loss=0.5,
            )

            assert result == 1

    def test_publish_checkpoint_saved_sync(self) -> None:
        """Test synchronous checkpoint saved publishing."""
        from app.training.event_integration import publish_checkpoint_saved_sync

        with patch("app.training.event_integration.get_event_bus") as mock_bus:
            mock_instance = MagicMock()
            mock_instance.publish_sync.return_value = 1
            mock_bus.return_value = mock_instance

            result = publish_checkpoint_saved_sync(
                config_key="hex8_2p",
                checkpoint_path="/models/checkpoint.pth",
                step=1000,
            )

            assert result == 1


# =============================================================================
# Test Subscription Helpers
# =============================================================================


class TestSubscribeToTrainingEvents:
    """Tests for subscribe_to_training_events decorator."""

    def test_exact_topic_subscription(self) -> None:
        """Test subscribing to exact topic."""
        from app.training.event_integration import subscribe_to_training_events

        with patch("app.training.event_integration.subscribe") as mock_subscribe:
            decorator = subscribe_to_training_events("training.completed")
            mock_subscribe.assert_called_once_with("training.completed", priority=0)

    def test_pattern_subscription(self) -> None:
        """Test subscribing to topic pattern."""
        from app.training.event_integration import subscribe_to_training_events

        with patch("app.training.event_integration.subscribe") as mock_subscribe:
            decorator = subscribe_to_training_events("training.*")
            # Should use EventFilter for pattern
            mock_subscribe.assert_called_once()
            call_args = mock_subscribe.call_args
            assert call_args.kwargs["priority"] == 0

    def test_custom_priority(self) -> None:
        """Test subscribing with custom priority."""
        from app.training.event_integration import subscribe_to_training_events

        with patch("app.training.event_integration.subscribe") as mock_subscribe:
            decorator = subscribe_to_training_events("training.completed", priority=10)
            mock_subscribe.assert_called_once()
            call_args = mock_subscribe.call_args
            assert call_args.kwargs["priority"] == 10


class TestSubscribeToEvaluationEvents:
    """Tests for subscribe_to_evaluation_events decorator."""

    def test_evaluation_subscription(self) -> None:
        """Test subscribing to evaluation events."""
        from app.training.event_integration import subscribe_to_evaluation_events

        with patch("app.training.event_integration.subscribe") as mock_subscribe:
            decorator = subscribe_to_evaluation_events()
            mock_subscribe.assert_called_once()


class TestSubscribeToCompositeEvents:
    """Tests for subscribe_to_composite_events decorator."""

    def test_composite_subscription(self) -> None:
        """Test subscribing to composite events."""
        from app.training.event_integration import subscribe_to_composite_events

        with patch("app.training.event_integration.subscribe") as mock_subscribe:
            decorator = subscribe_to_composite_events()
            mock_subscribe.assert_called_once()

    def test_composite_subscription_with_priority(self) -> None:
        """Test subscribing to composite events with priority."""
        from app.training.event_integration import subscribe_to_composite_events

        with patch("app.training.event_integration.subscribe") as mock_subscribe:
            decorator = subscribe_to_composite_events(priority=10)
            call_args = mock_subscribe.call_args
            assert call_args.kwargs["priority"] == 10


# =============================================================================
# Test Wiring Functions
# =============================================================================


class TestWireBackgroundEvaluatorEvents:
    """Tests for wire_background_evaluator_events function."""

    def test_wire_background_evaluator(self) -> None:
        """Test wiring background evaluator to events."""
        from app.training.event_integration import wire_background_evaluator_events

        # Create mock evaluator
        evaluator = MagicMock()
        original_process = MagicMock()
        evaluator._process_result = original_process
        evaluator._config_key = "hex8_2p"

        wire_background_evaluator_events(evaluator)

        # Verify _process_result was replaced (function object changed)
        assert evaluator._process_result is not original_process


class TestWireCheckpointManagerEvents:
    """Tests for wire_checkpoint_manager_events function."""

    def test_wire_checkpoint_manager(self) -> None:
        """Test wiring checkpoint manager to events."""
        from app.training.event_integration import wire_checkpoint_manager_events

        # Create mock manager
        manager = MagicMock()
        original_save = manager.save_checkpoint

        wire_checkpoint_manager_events(manager, config_key="hex8_2p")

        # Verify save_checkpoint was replaced
        assert manager.save_checkpoint != original_save


# =============================================================================
# Test Unified Routing
# =============================================================================


class TestUnifiedRouting:
    """Tests for unified event routing."""

    @pytest.mark.asyncio
    async def test_route_to_unified_when_available(self) -> None:
        """Test routing to unified router when available."""
        from app.training.event_integration import _route_to_unified

        with patch("app.training.event_integration.HAS_UNIFIED_ROUTER", True):
            with patch("app.training.event_integration.get_unified_router") as mock_get:
                mock_router = AsyncMock()
                mock_get.return_value = mock_router

                await _route_to_unified("test.event", {"key": "value"})

                mock_router.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_to_unified_when_unavailable(self) -> None:
        """Test graceful handling when unified router unavailable."""
        from app.training.event_integration import _route_to_unified

        with patch("app.training.event_integration.HAS_UNIFIED_ROUTER", False):
            # Should not raise
            await _route_to_unified("test.event", {"key": "value"})

    def test_route_to_unified_sync_when_available(self) -> None:
        """Test synchronous routing to unified router."""
        from app.training.event_integration import _route_to_unified_sync

        with patch("app.training.event_integration.HAS_UNIFIED_ROUTER", True):
            with patch("app.training.event_integration.get_unified_router") as mock_get:
                mock_router = MagicMock()
                mock_router.publish_sync = MagicMock()
                mock_get.return_value = mock_router

                _route_to_unified_sync("test.event", {"key": "value"})

                mock_router.publish_sync.assert_called_once()


# =============================================================================
# Test Module Exports
# =============================================================================


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports_importable(self) -> None:
        """Test that all __all__ exports are importable."""
        from app.training import event_integration

        for name in event_integration.__all__:
            assert hasattr(event_integration, name), f"Missing export: {name}"

    def test_key_exports_present(self) -> None:
        """Test key exports are present."""
        from app.training.event_integration import (
            TrainingTopics,
            TrainingEvent,
            EvaluationEvent,
            CheckpointEvent,
            SelfplayEvent,
            TrainingStartedEvent,
            TrainingCompletedEvent,
            publish_training_started,
            publish_training_completed,
            subscribe_to_training_events,
        )

        # All imports should succeed
        assert TrainingTopics is not None
        assert TrainingEvent is not None
        assert publish_training_started is not None
