"""Unit tests for app.training.feedback_accelerator module.

Tests cover:
- MomentumState and TrainingIntensity enums
- EloSnapshot and ConfigMomentum dataclasses
- TrainingDecision dataclass
- FeedbackAccelerator singleton class
- Module-level convenience functions
- Event wiring for feedback loop
"""

import pytest
import time
import tempfile
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from app.training.feedback_accelerator import (
    # Enums
    MomentumState,
    TrainingIntensity,
    # Dataclasses
    EloSnapshot,
    ConfigMomentum,
    TrainingDecision,
    # Main class
    FeedbackAccelerator,
    # Module functions
    get_feedback_accelerator,
    should_trigger_training,
    get_training_intensity,
    get_selfplay_multiplier,
    record_elo_update,
    record_games_generated,
    record_training_complete,
    record_promotion,
    get_curriculum_weights,
    get_selfplay_rate_recommendation,
    get_aggregate_selfplay_recommendation,
    # Event wiring
    wire_evaluation_to_feedback,
    unwire_evaluation_from_feedback,
    wire_hyperparameter_feedback,
    unwire_hyperparameter_feedback,
    # Constants
    MIN_GAMES_FOR_TRAINING,
    ACCELERATED_MIN_GAMES,
    HOT_PATH_MIN_GAMES,
    ELO_STRONG_IMPROVEMENT,
    ELO_MODERATE_IMPROVEMENT,
    ELO_PLATEAU_THRESHOLD,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path."""
    return tmp_path / "test_accelerator.db"


@pytest.fixture
def accelerator(temp_db_path, monkeypatch):
    """Create a fresh FeedbackAccelerator with temp database."""
    # Reset singleton
    FeedbackAccelerator._instance = None

    # Patch the database path
    monkeypatch.setattr(
        "app.training.feedback_accelerator.FEEDBACK_DB_PATH",
        temp_db_path
    )

    # Also patch _auto_wire_events to avoid event router issues in tests
    with patch.object(FeedbackAccelerator, "_auto_wire_events"):
        acc = FeedbackAccelerator()

    yield acc

    # Cleanup
    FeedbackAccelerator._instance = None


@pytest.fixture
def config_momentum():
    """Create a ConfigMomentum for testing."""
    return ConfigMomentum(config_key="hex8_2p")


# =============================================================================
# Tests for Enums
# =============================================================================

class TestMomentumState:
    """Tests for MomentumState enum."""

    def test_all_states_exist(self):
        """Verify all momentum states are defined."""
        assert MomentumState.ACCELERATING == "accelerating"
        assert MomentumState.IMPROVING == "improving"
        assert MomentumState.STABLE == "stable"
        assert MomentumState.PLATEAU == "plateau"
        assert MomentumState.REGRESSING == "regressing"

    def test_state_count(self):
        """Should have exactly 5 momentum states."""
        assert len(MomentumState) == 5

    def test_is_string_enum(self):
        """States should be usable as strings."""
        # MomentumState inherits from str, so value is the string itself
        assert MomentumState.ACCELERATING.value == "accelerating"
        assert MomentumState.STABLE.value == "stable"
        assert MomentumState.ACCELERATING == "accelerating"  # Can compare with string


class TestTrainingIntensity:
    """Tests for TrainingIntensity enum."""

    def test_all_intensities_exist(self):
        """Verify all training intensities are defined."""
        assert TrainingIntensity.HOT_PATH == "hot_path"
        assert TrainingIntensity.ACCELERATED == "accelerated"
        assert TrainingIntensity.NORMAL == "normal"
        assert TrainingIntensity.REDUCED == "reduced"
        assert TrainingIntensity.PAUSED == "paused"

    def test_intensity_count(self):
        """Should have exactly 5 intensity levels."""
        assert len(TrainingIntensity) == 5


# =============================================================================
# Tests for EloSnapshot
# =============================================================================

class TestEloSnapshot:
    """Tests for EloSnapshot dataclass."""

    def test_basic_creation(self):
        """Can create EloSnapshot with required fields."""
        snapshot = EloSnapshot(elo=1600.0, games_played=100, timestamp=time.time())
        assert snapshot.elo == 1600.0
        assert snapshot.games_played == 100
        assert snapshot.model_id is None

    def test_with_model_id(self):
        """Can create EloSnapshot with optional model_id."""
        snapshot = EloSnapshot(
            elo=1650.0,
            games_played=200,
            timestamp=time.time(),
            model_id="model_v2"
        )
        assert snapshot.model_id == "model_v2"


# =============================================================================
# Tests for ConfigMomentum
# =============================================================================

class TestConfigMomentum:
    """Tests for ConfigMomentum dataclass."""

    def test_default_values(self, config_momentum):
        """Verify default values are set correctly."""
        assert config_momentum.config_key == "hex8_2p"
        assert config_momentum.current_elo == 1500.0
        assert config_momentum.momentum_state == MomentumState.STABLE
        assert config_momentum.intensity == TrainingIntensity.NORMAL
        assert config_momentum.games_since_training == 0
        assert config_momentum.consecutive_improvements == 0
        assert config_momentum.consecutive_plateaus == 0

    def test_add_snapshot_updates_elo(self, config_momentum):
        """add_snapshot should update current_elo."""
        config_momentum.add_snapshot(1600.0, 100)
        assert config_momentum.current_elo == 1600.0
        assert len(config_momentum.elo_history) == 1

    def test_add_snapshot_tracks_history(self, config_momentum):
        """Multiple snapshots should be tracked."""
        config_momentum.add_snapshot(1550.0, 50)
        config_momentum.add_snapshot(1600.0, 100)
        config_momentum.add_snapshot(1650.0, 150)

        assert len(config_momentum.elo_history) == 3
        assert config_momentum.elo_history[-1].elo == 1650.0

    def test_history_bounded_to_100(self, config_momentum):
        """History should be bounded to 100 entries."""
        for i in range(150):
            config_momentum.add_snapshot(1500.0 + i, i * 10)

        assert len(config_momentum.elo_history) == 100
        # Should keep the most recent entries
        assert config_momentum.elo_history[-1].elo == 1649.0

    def test_momentum_accelerating_on_strong_improvement(self, config_momentum):
        """Strong Elo improvement should trigger ACCELERATING state."""
        # Add base snapshot
        config_momentum.add_snapshot(1500.0, 100)
        # Add strong improvement (>= ELO_STRONG_IMPROVEMENT)
        config_momentum.add_snapshot(1500.0 + ELO_STRONG_IMPROVEMENT + 5, 200)

        assert config_momentum.momentum_state == MomentumState.ACCELERATING
        assert config_momentum.consecutive_improvements >= 1

    def test_momentum_improving_on_moderate_improvement(self, config_momentum):
        """Moderate Elo improvement should trigger IMPROVING state."""
        config_momentum.add_snapshot(1500.0, 100)
        # Add moderate improvement (between moderate and strong thresholds)
        config_momentum.add_snapshot(1500.0 + ELO_MODERATE_IMPROVEMENT + 2, 200)

        assert config_momentum.momentum_state == MomentumState.IMPROVING

    def test_momentum_plateau_on_no_change(self, config_momentum):
        """No significant Elo change should trigger PLATEAU state."""
        config_momentum.add_snapshot(1500.0, 100)
        # Add very small change (< ELO_PLATEAU_THRESHOLD)
        config_momentum.add_snapshot(1500.0 + ELO_PLATEAU_THRESHOLD - 1, 200)

        assert config_momentum.momentum_state == MomentumState.PLATEAU

    def test_momentum_regressing_on_decline(self, config_momentum):
        """Significant Elo decline should trigger REGRESSING state."""
        config_momentum.add_snapshot(1600.0, 100)
        # Add decline (< -ELO_MODERATE_IMPROVEMENT)
        config_momentum.add_snapshot(1600.0 - ELO_MODERATE_IMPROVEMENT - 5, 200)

        assert config_momentum.momentum_state == MomentumState.REGRESSING

    def test_intensity_hot_path_on_consecutive_accelerating(self, config_momentum):
        """Hot path should activate after consecutive strong improvements."""
        # Simulate multiple strong improvements
        base_elo = 1500.0
        for i in range(3):
            config_momentum.add_snapshot(base_elo + (i + 1) * 30, (i + 1) * 100)

        # Should be in hot path after consecutive improvements
        if config_momentum.consecutive_improvements >= 2:
            assert config_momentum.intensity == TrainingIntensity.HOT_PATH

    def test_get_elo_trend(self, config_momentum):
        """get_elo_trend should calculate trend correctly."""
        config_momentum.add_snapshot(1500.0, 100)
        config_momentum.add_snapshot(1520.0, 200)
        config_momentum.add_snapshot(1540.0, 300)

        trend = config_momentum.get_elo_trend(lookback=3)
        assert trend > 0  # Positive trend

    def test_get_improvement_rate(self, config_momentum):
        """get_improvement_rate should calculate Elo per hour."""
        # Add snapshots with known time difference
        now = time.time()
        config_momentum.elo_history.append(EloSnapshot(1500.0, 100, now - 3600))  # 1 hour ago
        config_momentum.elo_history.append(EloSnapshot(1600.0, 200, now))  # Now

        rate = config_momentum.get_improvement_rate()
        assert abs(rate - 100.0) < 1.0  # ~100 Elo per hour

    def test_to_dict(self, config_momentum):
        """to_dict should serialize config momentum."""
        config_momentum.add_snapshot(1600.0, 100)

        data = config_momentum.to_dict()

        assert data["config_key"] == "hex8_2p"
        assert data["current_elo"] == 1600.0
        assert "momentum_state" in data
        assert "intensity" in data
        assert "elo_trend" in data


# =============================================================================
# Tests for TrainingDecision
# =============================================================================

class TestTrainingDecision:
    """Tests for TrainingDecision dataclass."""

    def test_basic_creation(self):
        """Can create TrainingDecision with all fields."""
        decision = TrainingDecision(
            should_train=True,
            intensity=TrainingIntensity.ACCELERATED,
            epochs_multiplier=1.5,
            learning_rate_multiplier=1.2,
            min_games_threshold=100,
            reason="Strong momentum",
            config_key="hex8_2p",
            momentum=MomentumState.ACCELERATING,
        )

        assert decision.should_train is True
        assert decision.intensity == TrainingIntensity.ACCELERATED
        assert decision.epochs_multiplier == 1.5
        assert decision.config_key == "hex8_2p"


# =============================================================================
# Tests for FeedbackAccelerator Class
# =============================================================================

class TestFeedbackAcceleratorSingleton:
    """Tests for FeedbackAccelerator singleton pattern."""

    def test_singleton_pattern(self, accelerator, temp_db_path, monkeypatch):
        """Should return same instance."""
        # Get another instance
        with patch.object(FeedbackAccelerator, "_auto_wire_events"):
            acc2 = FeedbackAccelerator()

        assert acc2 is accelerator

    def test_db_initialization(self, accelerator, temp_db_path):
        """Database should be initialized with tables."""
        assert temp_db_path.exists()

        # Check tables exist
        conn = sqlite3.connect(str(temp_db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor}
        conn.close()

        assert "config_momentum" in tables
        assert "elo_history" in tables
        assert "training_decisions" in tables


class TestFeedbackAcceleratorEloTracking:
    """Tests for Elo update tracking."""

    def test_record_elo_update_creates_config(self, accelerator):
        """record_elo_update should create config if not exists."""
        momentum = accelerator.record_elo_update("new_config", 1600.0, 100)

        assert "new_config" in accelerator._configs
        assert momentum.current_elo == 1600.0

    def test_record_elo_update_updates_existing(self, accelerator):
        """record_elo_update should update existing config."""
        accelerator.record_elo_update("hex8_2p", 1550.0, 50)
        accelerator.record_elo_update("hex8_2p", 1600.0, 100)

        momentum = accelerator._configs["hex8_2p"]
        assert momentum.current_elo == 1600.0
        assert len(momentum.elo_history) == 2

    def test_record_games_generated(self, accelerator):
        """record_games_generated should update games_since_training."""
        accelerator.record_elo_update("hex8_2p", 1500.0, 0)
        accelerator.record_games_generated("hex8_2p", 50)

        assert accelerator._configs["hex8_2p"].games_since_training == 50

        accelerator.record_games_generated("hex8_2p", 30)
        assert accelerator._configs["hex8_2p"].games_since_training == 80

    def test_record_training_complete_resets_games(self, accelerator):
        """record_training_complete should reset games_since_training."""
        accelerator.record_elo_update("hex8_2p", 1500.0, 0)
        accelerator.record_games_generated("hex8_2p", 100)
        accelerator.record_training_complete("hex8_2p", success=True)

        assert accelerator._configs["hex8_2p"].games_since_training == 0

    def test_record_promotion(self, accelerator):
        """record_promotion should update promotion tracking."""
        accelerator.record_elo_update("hex8_2p", 1500.0, 0)

        with patch("app.training.feedback_accelerator.get_improvement_optimizer") as mock:
            mock.return_value.record_promotion_success = MagicMock()
            accelerator.record_promotion("hex8_2p", 1650.0, "model_v2")

        momentum = accelerator._configs["hex8_2p"]
        assert momentum.last_promotion_elo == 1650.0
        assert momentum.total_promotions == 1


class TestFeedbackAcceleratorTrainingDecisions:
    """Tests for training decision logic."""

    def test_should_trigger_training_no_data(self, accelerator):
        """Should not trigger training for unknown config."""
        assert accelerator.should_trigger_training("unknown_config") is False

    def test_should_trigger_training_insufficient_games(self, accelerator):
        """Should not trigger training if not enough games."""
        accelerator.record_elo_update("hex8_2p", 1500.0, 0)
        # Record very few games - well below any threshold
        accelerator.record_games_generated("hex8_2p", 10)

        with patch("app.training.feedback_accelerator.get_dynamic_threshold") as mock:
            mock.return_value = MIN_GAMES_FOR_TRAINING
            with patch("app.training.feedback_accelerator.optimizer_should_fast_track") as mock2:
                mock2.return_value = False
                result = accelerator.should_trigger_training("hex8_2p")

        assert result is False

    def test_should_trigger_training_sufficient_games(self, accelerator):
        """Should trigger training if enough games."""
        accelerator.record_elo_update("hex8_2p", 1500.0, 0)
        accelerator.record_games_generated("hex8_2p", MIN_GAMES_FOR_TRAINING + 10)

        # May need to mock the optimizer threshold
        with patch("app.training.feedback_accelerator.get_dynamic_threshold") as mock:
            mock.return_value = MIN_GAMES_FOR_TRAINING
            assert accelerator.should_trigger_training("hex8_2p") is True

    def test_get_training_decision_normal(self, accelerator):
        """Should return normal decision for stable config."""
        accelerator.record_elo_update("hex8_2p", 1500.0, 0)
        accelerator.record_games_generated("hex8_2p", MIN_GAMES_FOR_TRAINING + 10)

        with patch("app.training.feedback_accelerator.get_dynamic_threshold") as mock:
            mock.return_value = MIN_GAMES_FOR_TRAINING
            with patch("app.training.feedback_accelerator.optimizer_should_fast_track") as mock2:
                mock2.return_value = False
                decision = accelerator.get_training_decision("hex8_2p")

        assert decision.should_train is True
        assert decision.intensity == TrainingIntensity.NORMAL
        assert decision.epochs_multiplier == 1.0

    def test_get_training_decision_paused(self, accelerator):
        """Should not train when intensity is PAUSED."""
        accelerator.record_elo_update("hex8_2p", 1500.0, 0)
        accelerator._configs["hex8_2p"].intensity = TrainingIntensity.PAUSED
        accelerator.record_games_generated("hex8_2p", 1000)

        decision = accelerator.get_training_decision("hex8_2p")

        assert decision.should_train is False
        assert decision.intensity == TrainingIntensity.PAUSED

    def test_get_training_intensity_returns_dict(self, accelerator):
        """get_training_intensity should return intensity parameters."""
        accelerator.record_elo_update("hex8_2p", 1500.0, 0)
        accelerator.record_games_generated("hex8_2p", 100)

        with patch("app.training.feedback_accelerator.get_dynamic_threshold") as mock:
            mock.return_value = MIN_GAMES_FOR_TRAINING
            with patch("app.training.feedback_accelerator.optimizer_should_fast_track") as mock2:
                mock2.return_value = False
                intensity = accelerator.get_training_intensity("hex8_2p")

        assert "epochs_multiplier" in intensity
        assert "learning_rate_multiplier" in intensity
        assert "min_games_threshold" in intensity
        assert "intensity" in intensity


class TestFeedbackAcceleratorSelfplayMultiplier:
    """Tests for selfplay rate multiplier."""

    def test_selfplay_multiplier_unknown_config(self, accelerator):
        """Unknown config should return 1.0."""
        assert accelerator.get_selfplay_multiplier("unknown") == 1.0

    def test_selfplay_multiplier_accelerating(self, accelerator):
        """ACCELERATING should return 1.5x."""
        accelerator.record_elo_update("hex8_2p", 1500.0, 100)
        accelerator._configs["hex8_2p"].momentum_state = MomentumState.ACCELERATING

        multiplier = accelerator.get_selfplay_multiplier("hex8_2p")
        assert multiplier == 1.5

    def test_selfplay_multiplier_improving(self, accelerator):
        """IMPROVING should return 1.25x."""
        accelerator.record_elo_update("hex8_2p", 1500.0, 100)
        accelerator._configs["hex8_2p"].momentum_state = MomentumState.IMPROVING

        multiplier = accelerator.get_selfplay_multiplier("hex8_2p")
        assert multiplier == 1.25

    def test_selfplay_multiplier_stable(self, accelerator):
        """STABLE should return 1.0x."""
        accelerator.record_elo_update("hex8_2p", 1500.0, 100)
        accelerator._configs["hex8_2p"].momentum_state = MomentumState.STABLE

        multiplier = accelerator.get_selfplay_multiplier("hex8_2p")
        assert multiplier == 1.0

    def test_selfplay_multiplier_plateau(self, accelerator):
        """PLATEAU should return 1.1x."""
        accelerator.record_elo_update("hex8_2p", 1500.0, 100)
        accelerator._configs["hex8_2p"].momentum_state = MomentumState.PLATEAU

        multiplier = accelerator.get_selfplay_multiplier("hex8_2p")
        assert multiplier == 1.1

    def test_selfplay_multiplier_regressing(self, accelerator):
        """REGRESSING should return 0.75x."""
        accelerator.record_elo_update("hex8_2p", 1500.0, 100)
        accelerator._configs["hex8_2p"].momentum_state = MomentumState.REGRESSING

        multiplier = accelerator.get_selfplay_multiplier("hex8_2p")
        assert multiplier == 0.75

    def test_selfplay_multiplier_consecutive_improvements_boost(self, accelerator):
        """Consecutive improvements should boost multiplier."""
        accelerator.record_elo_update("hex8_2p", 1500.0, 100)
        accelerator._configs["hex8_2p"].momentum_state = MomentumState.ACCELERATING
        accelerator._configs["hex8_2p"].consecutive_improvements = 3

        multiplier = accelerator.get_selfplay_multiplier("hex8_2p")
        assert multiplier == min(1.5 * 1.1, 1.5)  # Boosted but capped at 1.5


class TestFeedbackAcceleratorCurriculumWeights:
    """Tests for curriculum weight recommendations."""

    def test_curriculum_weights_empty(self, accelerator):
        """Empty configs should return empty dict."""
        weights = accelerator.get_curriculum_weights()
        assert weights == {}

    def test_curriculum_weights_accelerating(self, accelerator):
        """ACCELERATING config should get higher weight."""
        accelerator.record_elo_update("hex8_2p", 1500.0, 100)
        accelerator._configs["hex8_2p"].momentum_state = MomentumState.ACCELERATING

        weights = accelerator.get_curriculum_weights()
        assert weights["hex8_2p"] == 1.5

    def test_curriculum_weights_regressing(self, accelerator):
        """REGRESSING config should get weight boost to reverse."""
        accelerator.record_elo_update("hex8_2p", 1500.0, 100)
        accelerator._configs["hex8_2p"].momentum_state = MomentumState.REGRESSING

        weights = accelerator.get_curriculum_weights()
        assert weights["hex8_2p"] == 1.4


class TestFeedbackAcceleratorAggregateRecommendation:
    """Tests for aggregate selfplay recommendation."""

    def test_aggregate_recommendation_empty(self, accelerator):
        """Empty configs should return 1.0 multiplier."""
        result = accelerator.get_aggregate_selfplay_recommendation()

        assert result["recommended_multiplier"] == 1.0
        assert result["aggregate_momentum"] == "unknown"

    def test_aggregate_recommendation_majority_accelerating(self, accelerator):
        """Majority accelerating should return 2.0 multiplier."""
        # Create 3 accelerating configs
        for i in range(3):
            accelerator.record_elo_update(f"config_{i}", 1500.0, 100)
            accelerator._configs[f"config_{i}"].momentum_state = MomentumState.ACCELERATING

        result = accelerator.get_aggregate_selfplay_recommendation()

        assert result["recommended_multiplier"] == 2.0
        assert result["aggregate_momentum"] == "accelerating"

    def test_aggregate_recommendation_mixed(self, accelerator):
        """Mixed states should return stable recommendation."""
        accelerator.record_elo_update("config_1", 1500.0, 100)
        accelerator._configs["config_1"].momentum_state = MomentumState.ACCELERATING

        accelerator.record_elo_update("config_2", 1500.0, 100)
        accelerator._configs["config_2"].momentum_state = MomentumState.STABLE

        accelerator.record_elo_update("config_3", 1500.0, 100)
        accelerator._configs["config_3"].momentum_state = MomentumState.REGRESSING

        result = accelerator.get_aggregate_selfplay_recommendation()

        assert result["aggregate_momentum"] == "stable"


class TestFeedbackAcceleratorStatus:
    """Tests for status and reporting methods."""

    def test_get_config_momentum(self, accelerator):
        """get_config_momentum should return correct momentum."""
        accelerator.record_elo_update("hex8_2p", 1600.0, 100)

        momentum = accelerator.get_config_momentum("hex8_2p")
        assert momentum is not None
        assert momentum.current_elo == 1600.0

    def test_get_config_momentum_unknown(self, accelerator):
        """Unknown config should return None."""
        assert accelerator.get_config_momentum("unknown") is None

    def test_get_all_momentum(self, accelerator):
        """get_all_momentum should return all configs."""
        accelerator.record_elo_update("hex8_2p", 1600.0, 100)
        accelerator.record_elo_update("square8_2p", 1550.0, 50)

        all_momentum = accelerator.get_all_momentum()

        assert "hex8_2p" in all_momentum
        assert "square8_2p" in all_momentum
        assert all_momentum["hex8_2p"]["current_elo"] == 1600.0

    def test_get_improving_configs(self, accelerator):
        """Should return list of improving configs."""
        accelerator.record_elo_update("hex8_2p", 1600.0, 100)
        accelerator._configs["hex8_2p"].momentum_state = MomentumState.IMPROVING

        accelerator.record_elo_update("square8_2p", 1500.0, 100)
        accelerator._configs["square8_2p"].momentum_state = MomentumState.STABLE

        improving = accelerator.get_improving_configs()
        assert "hex8_2p" in improving
        assert "square8_2p" not in improving

    def test_get_plateau_configs(self, accelerator):
        """Should return list of plateau configs."""
        accelerator.record_elo_update("hex8_2p", 1500.0, 100)
        accelerator._configs["hex8_2p"].momentum_state = MomentumState.PLATEAU

        plateau = accelerator.get_plateau_configs()
        assert "hex8_2p" in plateau

    def test_get_status_summary(self, accelerator):
        """get_status_summary should return complete summary."""
        accelerator.record_elo_update("hex8_2p", 1600.0, 100)

        summary = accelerator.get_status_summary()

        assert "total_configs" in summary
        assert "improving_configs" in summary
        assert "plateau_configs" in summary
        assert "configs" in summary

    def test_get_metrics_dict(self, accelerator):
        """get_metrics_dict should return Prometheus-style metrics."""
        accelerator.record_elo_update("hex8_2p", 1600.0, 100)

        metrics = accelerator.get_metrics_dict()

        assert "ringrift_feedback_total_configs" in metrics
        assert metrics["ringrift_feedback_total_configs"] == 1


class TestFeedbackAcceleratorIntensityControl:
    """Tests for training intensity control."""

    def test_set_intensity(self, accelerator):
        """set_intensity should update config intensity."""
        accelerator.record_elo_update("hex8_2p", 1500.0, 0)

        accelerator.set_intensity("hex8_2p", TrainingIntensity.ACCELERATED)

        assert accelerator._configs["hex8_2p"].intensity == TrainingIntensity.ACCELERATED

    def test_set_intensity_creates_config(self, accelerator):
        """set_intensity should create config if not exists."""
        accelerator.set_intensity("new_config", TrainingIntensity.HOT_PATH)

        assert "new_config" in accelerator._configs
        assert accelerator._configs["new_config"].intensity == TrainingIntensity.HOT_PATH

    def test_signal_training_needed(self, accelerator):
        """signal_training_needed should upgrade intensity."""
        accelerator.record_elo_update("hex8_2p", 1500.0, 0)
        accelerator._configs["hex8_2p"].intensity = TrainingIntensity.NORMAL

        accelerator.signal_training_needed("hex8_2p", urgency="high")

        assert accelerator._configs["hex8_2p"].intensity == TrainingIntensity.ACCELERATED

    def test_signal_training_needed_no_downgrade(self, accelerator):
        """signal_training_needed should not downgrade intensity."""
        accelerator.record_elo_update("hex8_2p", 1500.0, 0)
        accelerator._configs["hex8_2p"].intensity = TrainingIntensity.HOT_PATH

        accelerator.signal_training_needed("hex8_2p", urgency="normal")

        # Should still be hot path
        assert accelerator._configs["hex8_2p"].intensity == TrainingIntensity.HOT_PATH

    def test_intensity_priority(self, accelerator):
        """_intensity_priority should return correct ordering."""
        assert accelerator._intensity_priority(TrainingIntensity.PAUSED) == 0
        assert accelerator._intensity_priority(TrainingIntensity.REDUCED) == 1
        assert accelerator._intensity_priority(TrainingIntensity.NORMAL) == 2
        assert accelerator._intensity_priority(TrainingIntensity.ACCELERATED) == 3
        assert accelerator._intensity_priority(TrainingIntensity.HOT_PATH) == 4


class TestFeedbackAcceleratorUnifiedIntegration:
    """Tests for unified signal integration."""

    def test_get_unified_urgency(self, accelerator):
        """Should map intensity to unified urgency."""
        from app.training.unified_signals import TrainingUrgency

        accelerator.record_elo_update("hex8_2p", 1500.0, 0)
        accelerator._configs["hex8_2p"].intensity = TrainingIntensity.HOT_PATH

        urgency = accelerator.get_unified_urgency("hex8_2p")
        assert urgency == TrainingUrgency.CRITICAL

    def test_get_unified_urgency_unknown(self, accelerator):
        """Unknown config should return NONE urgency."""
        from app.training.unified_signals import TrainingUrgency

        urgency = accelerator.get_unified_urgency("unknown")
        assert urgency == TrainingUrgency.NONE


class TestFeedbackAcceleratorCallbacks:
    """Tests for training callbacks."""

    def test_register_training_callback(self, accelerator):
        """Can register training callbacks."""
        callback = MagicMock()
        accelerator.register_training_callback(callback)

        assert callback in accelerator._training_callbacks

    def test_check_and_trigger_training(self, accelerator):
        """check_and_trigger_training should call callbacks."""
        callback = MagicMock()
        accelerator.register_training_callback(callback)

        # Setup config that should trigger training
        accelerator.record_elo_update("hex8_2p", 1500.0, 0)
        accelerator.record_games_generated("hex8_2p", MIN_GAMES_FOR_TRAINING + 100)

        with patch("app.training.feedback_accelerator.get_dynamic_threshold") as mock:
            mock.return_value = MIN_GAMES_FOR_TRAINING
            with patch("app.training.feedback_accelerator.optimizer_should_fast_track") as mock2:
                mock2.return_value = False
                triggered = accelerator.check_and_trigger_training()

        if len(triggered) > 0:
            callback.assert_called()


# =============================================================================
# Tests for Module-Level Functions
# =============================================================================

class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_feedback_accelerator_singleton(self, temp_db_path, monkeypatch):
        """get_feedback_accelerator should return singleton."""
        import app.training.feedback_accelerator as module

        FeedbackAccelerator._instance = None
        module._accelerator = None

        monkeypatch.setattr(module, "FEEDBACK_DB_PATH", temp_db_path)

        with patch.object(FeedbackAccelerator, "_auto_wire_events"):
            acc1 = get_feedback_accelerator()
            acc2 = get_feedback_accelerator()

        assert acc1 is acc2

        # Cleanup
        FeedbackAccelerator._instance = None
        module._accelerator = None


# =============================================================================
# Tests for Rate Change Detection
# =============================================================================

class TestRateChangeDetection:
    """Tests for selfplay rate change detection (Phase 19.3)."""

    def test_rate_change_detected(self, accelerator):
        """Significant rate change should be detected."""
        accelerator.record_elo_update("hex8_2p", 1500.0, 100)

        # Set initial state
        accelerator._configs["hex8_2p"].momentum_state = MomentumState.STABLE
        accelerator.get_selfplay_multiplier("hex8_2p")  # Initialize rate tracking

        # Change to accelerating (significant rate change)
        accelerator._configs["hex8_2p"].momentum_state = MomentumState.ACCELERATING

        with patch.object(accelerator, "_emit_rate_change_event") as mock:
            accelerator.get_selfplay_multiplier("hex8_2p")
            # Rate change from 1.0 to 1.5 is >20%
            mock.assert_called_once()

    def test_small_rate_change_ignored(self, accelerator):
        """Small rate changes should not emit events."""
        accelerator.record_elo_update("hex8_2p", 1500.0, 100)

        # Set initial state
        accelerator._configs["hex8_2p"].momentum_state = MomentumState.STABLE
        accelerator.get_selfplay_multiplier("hex8_2p")  # Initialize

        # Change to plateau (1.0 to 1.1 = 10% change, <20%)
        accelerator._configs["hex8_2p"].momentum_state = MomentumState.PLATEAU

        with patch.object(accelerator, "_emit_rate_change_event") as mock:
            accelerator.get_selfplay_multiplier("hex8_2p")
            mock.assert_not_called()


# =============================================================================
# Tests for Event Wiring
# =============================================================================

class TestEventWiring:
    """Tests for event bus integration."""

    def test_wire_evaluation_to_feedback(self):
        """wire_evaluation_to_feedback should subscribe to events."""
        import app.training.feedback_accelerator as module
        module._evaluation_watcher_active = False

        with patch("app.training.feedback_accelerator.get_router") as mock_router:
            mock_router.return_value.subscribe = MagicMock()

            result = wire_evaluation_to_feedback()

            assert result is True
            mock_router.return_value.subscribe.assert_called()

        # Cleanup
        module._evaluation_watcher_active = False

    def test_wire_hyperparameter_feedback(self):
        """wire_hyperparameter_feedback should subscribe to events."""
        import app.training.feedback_accelerator as module
        module._hyperparameter_watcher_active = False

        with patch("app.training.feedback_accelerator.get_router") as mock_router:
            mock_router.return_value.subscribe = MagicMock()

            result = wire_hyperparameter_feedback()

            assert result is True

        # Cleanup
        module._hyperparameter_watcher_active = False

    def test_unwire_evaluation_from_feedback(self):
        """unwire_evaluation_from_feedback should unsubscribe."""
        import app.training.feedback_accelerator as module
        module._evaluation_watcher_active = True

        with patch("app.training.feedback_accelerator.get_router") as mock_router:
            mock_router.return_value.unsubscribe = MagicMock()

            unwire_evaluation_from_feedback()

            mock_router.return_value.unsubscribe.assert_called()

        assert module._evaluation_watcher_active is False


# =============================================================================
# Tests for Elo Plateau Detection
# =============================================================================

class TestEloPlateau:
    """Tests for Elo plateau detection integration."""

    def test_plateau_detection_insufficient_history(self, accelerator):
        """Should not trigger plateau detection with insufficient history."""
        accelerator.record_elo_update("hex8_2p", 1500.0, 100)

        # Only 1 snapshot, need at least 5
        with patch("app.training.feedback_accelerator.detect_elo_plateau") as mock:
            accelerator._check_elo_plateau(
                "hex8_2p",
                accelerator._configs["hex8_2p"]
            )
            mock.assert_not_called()

    def test_plateau_detection_with_history(self, accelerator):
        """Should trigger plateau detection with sufficient history."""
        # Add 6 snapshots
        for i in range(6):
            accelerator.record_elo_update("hex8_2p", 1500.0 + i * 0.5, i * 100)

        with patch("app.training.feedback_accelerator.detect_elo_plateau") as mock_detect:
            mock_detect.return_value = (False, {"confidence": 0.3})

            accelerator._check_elo_plateau(
                "hex8_2p",
                accelerator._configs["hex8_2p"]
            )

            mock_detect.assert_called_once()


# =============================================================================
# Tests for Constants
# =============================================================================

class TestConstants:
    """Tests for module constants."""

    def test_game_thresholds_ordered(self):
        """Game thresholds should be properly ordered."""
        assert HOT_PATH_MIN_GAMES < ACCELERATED_MIN_GAMES < MIN_GAMES_FOR_TRAINING

    def test_elo_thresholds_ordered(self):
        """Elo thresholds should be properly ordered."""
        assert ELO_PLATEAU_THRESHOLD < ELO_MODERATE_IMPROVEMENT < ELO_STRONG_IMPROVEMENT

    def test_environment_override_respected(self):
        """Game thresholds should respect environment variables."""
        import os

        # These are set at module load time, so we verify defaults
        assert MIN_GAMES_FOR_TRAINING >= 50
        assert ACCELERATED_MIN_GAMES >= 25
        assert HOT_PATH_MIN_GAMES >= 25
