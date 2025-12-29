"""Tests for ImprovementOptimizer.

Tests cover:
- ImprovementSignal enum
- ImprovementState dataclass
- OptimizationRecommendation dataclass
- ImprovementOptimizer class methods
- Module-level convenience functions
- State persistence
- Dynamic threshold calculation
- Positive feedback acceleration
- Regression handling
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.training.improvement_optimizer import (
    ImprovementOptimizer,
    ImprovementSignal,
    ImprovementState,
    OptimizationRecommendation,
    get_dynamic_threshold,
    get_evaluation_interval,
    get_improvement_metrics,
    get_improvement_optimizer,
    get_selfplay_priority_boost,
    get_training_adjustment,
    record_promotion_failure,
    record_promotion_success,
    record_regression,
    record_training_complete,
    reset_improvement_optimizer,
    should_fast_track_training,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_state_path():
    """Create a temporary file path for state storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "optimizer_state.json"


@pytest.fixture
def optimizer(temp_state_path):
    """Create a fresh optimizer instance for testing."""
    # Reset singleton first
    ImprovementOptimizer.reset_instance()
    opt = ImprovementOptimizer(state_path=temp_state_path)
    yield opt
    # Clean up singleton after test
    ImprovementOptimizer.reset_instance()


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before each test."""
    ImprovementOptimizer.reset_instance()
    yield
    ImprovementOptimizer.reset_instance()


# =============================================================================
# ImprovementSignal Tests
# =============================================================================


class TestImprovementSignal:
    """Tests for ImprovementSignal enum."""

    def test_success_signals_exist(self):
        """Test that success signals are defined."""
        assert ImprovementSignal.PROMOTION_STREAK.value == "promotion_streak"
        assert ImprovementSignal.ELO_BREAKTHROUGH.value == "elo_breakthrough"
        assert ImprovementSignal.QUALITY_DATA_SURGE.value == "quality_data_surge"
        assert ImprovementSignal.CALIBRATION_EXCELLENT.value == "calibration_excellent"
        assert ImprovementSignal.EFFICIENCY_OPTIMAL.value == "efficiency_optimal"

    def test_momentum_signals_exist(self):
        """Test that momentum signals are defined."""
        assert ImprovementSignal.STEADY_IMPROVEMENT.value == "steady_improvement"
        assert ImprovementSignal.HEALTHY_PIPELINE.value == "healthy_pipeline"

    def test_opportunity_signals_exist(self):
        """Test that opportunity signals are defined."""
        assert ImprovementSignal.UNDERUTILIZED_CAPACITY.value == "underutilized_capacity"
        assert ImprovementSignal.DATA_QUALITY_HIGH.value == "data_quality_high"
        assert ImprovementSignal.LOW_QUEUE_DEPTH.value == "low_queue_depth"

    def test_warning_signals_exist(self):
        """Test that warning signals are defined."""
        assert ImprovementSignal.REGRESSION_DETECTED.value == "regression_detected"

    def test_signal_is_string(self):
        """Test that signal values can be used as strings."""
        assert str(ImprovementSignal.PROMOTION_STREAK) == "ImprovementSignal.PROMOTION_STREAK"
        assert ImprovementSignal.PROMOTION_STREAK.value == "promotion_streak"


# =============================================================================
# ImprovementState Tests
# =============================================================================


class TestImprovementState:
    """Tests for ImprovementState dataclass."""

    def test_default_values(self):
        """Test default state values."""
        state = ImprovementState()

        assert state.consecutive_promotions == 0
        assert state.total_promotions_24h == 0
        assert state.last_promotion_time == 0.0
        assert state.promotion_times == []
        assert state.avg_elo_gain_per_promotion == 25.0
        assert state.best_elo_gain == 0.0
        assert state.elo_gains == []
        assert state.training_runs_24h == 0
        assert state.data_quality_score == 1.0
        assert state.parity_success_rate == 1.0
        assert state.threshold_multiplier == 1.0
        assert state.evaluation_frequency_multiplier == 1.0
        assert state.config_boosts == {}

    def test_custom_values(self):
        """Test state with custom values."""
        state = ImprovementState(
            consecutive_promotions=5,
            avg_elo_gain_per_promotion=30.0,
            threshold_multiplier=0.8,
            config_boosts={"hex8_2p": 0.7},
        )

        assert state.consecutive_promotions == 5
        assert state.avg_elo_gain_per_promotion == 30.0
        assert state.threshold_multiplier == 0.8
        assert state.config_boosts == {"hex8_2p": 0.7}


# =============================================================================
# OptimizationRecommendation Tests
# =============================================================================


class TestOptimizationRecommendation:
    """Tests for OptimizationRecommendation dataclass."""

    def test_recommendation_creation(self):
        """Test creating a recommendation."""
        rec = OptimizationRecommendation(
            signal=ImprovementSignal.PROMOTION_STREAK,
            config_key="hex8_2p",
            threshold_adjustment=0.8,
            evaluation_adjustment=0.9,
            reason="Testing",
            confidence=0.9,
        )

        assert rec.signal == ImprovementSignal.PROMOTION_STREAK
        assert rec.config_key == "hex8_2p"
        assert rec.threshold_adjustment == 0.8
        assert rec.evaluation_adjustment == 0.9
        assert rec.reason == "Testing"
        assert rec.confidence == 0.9
        assert rec.metadata == {}

    def test_recommendation_with_metadata(self):
        """Test recommendation with metadata."""
        rec = OptimizationRecommendation(
            signal=ImprovementSignal.ELO_BREAKTHROUGH,
            config_key="square8_2p",
            threshold_adjustment=0.7,
            evaluation_adjustment=0.8,
            reason="Large Elo gain",
            confidence=0.95,
            metadata={"elo_gain": 75.0, "model_id": "model_123"},
        )

        assert rec.metadata["elo_gain"] == 75.0
        assert rec.metadata["model_id"] == "model_123"


# =============================================================================
# ImprovementOptimizer Tests
# =============================================================================


class TestImprovementOptimizerInit:
    """Tests for ImprovementOptimizer initialization."""

    def test_creates_instance(self, temp_state_path):
        """Test optimizer can be created."""
        opt = ImprovementOptimizer(state_path=temp_state_path)
        assert opt is not None
        assert opt._state is not None

    def test_uses_provided_state_path(self, temp_state_path):
        """Test optimizer uses provided state path."""
        opt = ImprovementOptimizer(state_path=temp_state_path)
        assert opt._state_path == temp_state_path

    def test_loads_existing_state(self, temp_state_path):
        """Test optimizer loads existing state from disk."""
        # Save state manually
        state = {
            "consecutive_promotions": 3,
            "threshold_multiplier": 0.7,
            "config_boosts": {"hex8_2p": 0.6},
        }
        temp_state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_state_path, "w") as f:
            json.dump(state, f)

        # Create optimizer - should load state
        opt = ImprovementOptimizer(state_path=temp_state_path)

        assert opt._state.consecutive_promotions == 3
        assert opt._state.threshold_multiplier == 0.7
        assert opt._state.config_boosts == {"hex8_2p": 0.6}


class TestImprovementOptimizerStatePersistence:
    """Tests for state persistence."""

    def test_saves_state_on_promotion(self, optimizer, temp_state_path):
        """Test state is saved after promotion."""
        optimizer.record_promotion_success("hex8_2p", 50.0)

        # Verify file was created
        assert temp_state_path.exists()

        with open(temp_state_path) as f:
            data = json.load(f)

        assert data["consecutive_promotions"] == 1

    def test_saves_state_on_training_complete(self, optimizer, temp_state_path):
        """Test state is saved after training complete."""
        optimizer.record_training_complete("hex8_2p", 3600.0, 0.05)

        assert temp_state_path.exists()

    def test_saves_state_on_regression(self, optimizer, temp_state_path):
        """Test state is saved after regression."""
        optimizer.record_regression("hex8_2p", "elo_drop", "moderate", 30.0)

        assert temp_state_path.exists()


class TestImprovementOptimizerPromotionSuccess:
    """Tests for record_promotion_success."""

    def test_increments_consecutive_promotions(self, optimizer):
        """Test promotion success increments counter."""
        assert optimizer._state.consecutive_promotions == 0

        optimizer.record_promotion_success("hex8_2p", 25.0)
        assert optimizer._state.consecutive_promotions == 1

        optimizer.record_promotion_success("hex8_2p", 30.0)
        assert optimizer._state.consecutive_promotions == 2

    def test_records_elo_gain(self, optimizer):
        """Test Elo gain is recorded."""
        optimizer.record_promotion_success("hex8_2p", 50.0)

        assert 50.0 in optimizer._state.elo_gains
        assert optimizer._state.best_elo_gain == 50.0

    def test_updates_best_elo_gain(self, optimizer):
        """Test best Elo gain is updated."""
        optimizer.record_promotion_success("hex8_2p", 30.0)
        assert optimizer._state.best_elo_gain == 30.0

        optimizer.record_promotion_success("hex8_2p", 50.0)
        assert optimizer._state.best_elo_gain == 50.0

        # Smaller gain shouldn't update best
        optimizer.record_promotion_success("hex8_2p", 20.0)
        assert optimizer._state.best_elo_gain == 50.0

    def test_acceleration_on_streak(self, optimizer):
        """Test threshold multiplier decreases on streak."""
        initial = optimizer._state.threshold_multiplier

        # Build a 3-promotion streak
        for _ in range(3):
            optimizer.record_promotion_success("hex8_2p", 25.0)

        # Multiplier should have decreased (faster training)
        assert optimizer._state.threshold_multiplier < initial

    def test_elo_breakthrough_bonus(self, optimizer):
        """Test large Elo gain gives bonus acceleration."""
        initial = optimizer._state.threshold_multiplier

        # Record a breakthrough (+50 Elo)
        rec = optimizer.record_promotion_success("hex8_2p", 75.0)

        assert optimizer._state.threshold_multiplier < initial
        assert rec.signal == ImprovementSignal.ELO_BREAKTHROUGH

    def test_returns_recommendation(self, optimizer):
        """Test returns OptimizationRecommendation."""
        rec = optimizer.record_promotion_success("hex8_2p", 25.0)

        assert isinstance(rec, OptimizationRecommendation)
        assert rec.config_key == "hex8_2p"
        assert rec.metadata["elo_gain"] == 25.0

    def test_config_boost_applied(self, optimizer):
        """Test config-specific boost is applied."""
        optimizer.record_promotion_success("hex8_2p", 60.0)

        # Config should have earned a boost
        assert "hex8_2p" in optimizer._state.config_boosts
        assert optimizer._state.config_boosts["hex8_2p"] < 1.0


class TestImprovementOptimizerPromotionFailure:
    """Tests for record_promotion_failure."""

    def test_resets_consecutive_promotions(self, optimizer):
        """Test failure resets promotion streak."""
        # Build a streak
        for _ in range(3):
            optimizer.record_promotion_success("hex8_2p", 25.0)

        assert optimizer._state.consecutive_promotions == 3

        optimizer.record_promotion_failure("hex8_2p")

        assert optimizer._state.consecutive_promotions == 0

    def test_slows_down_training(self, optimizer):
        """Test failure increases threshold multiplier."""
        # First accelerate with success
        optimizer.record_promotion_success("hex8_2p", 50.0)
        accelerated = optimizer._state.threshold_multiplier

        optimizer.record_promotion_failure("hex8_2p")

        assert optimizer._state.threshold_multiplier > accelerated


class TestImprovementOptimizerRegression:
    """Tests for record_regression."""

    def test_resets_promotion_streak(self, optimizer):
        """Test regression resets promotion streak."""
        for _ in range(3):
            optimizer.record_promotion_success("hex8_2p", 25.0)

        optimizer.record_regression("hex8_2p", "elo_drop", "moderate", 30.0)

        assert optimizer._state.consecutive_promotions == 0

    def test_slows_down_by_severity(self, optimizer):
        """Test different severities apply different slowdowns."""
        initial = optimizer._state.threshold_multiplier

        optimizer.record_regression("hex8_2p", "elo_drop", "mild", 10.0)
        mild_slowdown = optimizer._state.threshold_multiplier

        # Reset and try moderate
        optimizer._state.threshold_multiplier = initial
        optimizer.record_regression("hex8_2p", "elo_drop", "moderate", 30.0)
        moderate_slowdown = optimizer._state.threshold_multiplier

        # Reset and try severe
        optimizer._state.threshold_multiplier = initial
        optimizer.record_regression("hex8_2p", "elo_drop", "severe", 60.0)
        severe_slowdown = optimizer._state.threshold_multiplier

        # More severe = higher multiplier (slower training)
        assert mild_slowdown < moderate_slowdown < severe_slowdown

    def test_returns_recommendation_with_metadata(self, optimizer):
        """Test returns recommendation with regression metadata."""
        rec = optimizer.record_regression("hex8_2p", "elo_drop", "severe", 60.0, 0.40)

        assert rec.signal == ImprovementSignal.REGRESSION_DETECTED
        assert rec.metadata["regression_type"] == "elo_drop"
        assert rec.metadata["severity"] == "severe"
        assert rec.metadata["elo_drop"] == 60.0
        assert rec.metadata["win_rate"] == 0.40


class TestImprovementOptimizerTrainingComplete:
    """Tests for record_training_complete."""

    def test_increments_training_count(self, optimizer):
        """Test training runs are counted."""
        assert optimizer._state.training_runs_24h == 0

        optimizer.record_training_complete("hex8_2p", 3600.0, 0.05)

        assert optimizer._state.training_runs_24h == 1

    def test_updates_avg_duration(self, optimizer):
        """Test average duration is updated."""
        initial = optimizer._state.avg_training_duration_seconds

        optimizer.record_training_complete("hex8_2p", 1800.0, 0.05)

        # Should move towards 1800
        assert optimizer._state.avg_training_duration_seconds < initial

    def test_excellent_calibration_bonus(self, optimizer):
        """Test excellent calibration gives bonus."""
        initial = optimizer._state.threshold_multiplier

        # ECE below threshold (0.05) triggers bonus
        rec = optimizer.record_training_complete("hex8_2p", 3600.0, 0.05, calibration_ece=0.03)

        assert optimizer._state.threshold_multiplier < initial
        assert rec.signal == ImprovementSignal.CALIBRATION_EXCELLENT


class TestImprovementOptimizerDataQuality:
    """Tests for record_data_quality."""

    def test_updates_quality_scores(self, optimizer):
        """Test quality scores are updated."""
        optimizer.record_data_quality(0.99, 0.95)

        assert optimizer._state.parity_success_rate == 0.99
        assert optimizer._state.data_quality_score == 0.95

    def test_high_quality_accelerates(self, optimizer):
        """Test high quality data accelerates training."""
        initial = optimizer._state.threshold_multiplier

        optimizer.record_data_quality(0.99, 0.98)

        assert optimizer._state.threshold_multiplier < initial


class TestImprovementOptimizerClusterUtilization:
    """Tests for record_cluster_utilization."""

    def test_optimal_utilization_signal(self, optimizer):
        """Test optimal utilization returns correct signal."""
        rec = optimizer.record_cluster_utilization(70.0, 70.0)

        assert rec.signal == ImprovementSignal.EFFICIENCY_OPTIMAL

    def test_underutilization_accelerates(self, optimizer):
        """Test underutilization accelerates training."""
        initial = optimizer._state.threshold_multiplier

        optimizer.record_cluster_utilization(30.0, 25.0)

        assert optimizer._state.threshold_multiplier < initial


class TestImprovementOptimizerDynamicThreshold:
    """Tests for get_dynamic_threshold."""

    def test_baseline_threshold(self, optimizer):
        """Test baseline threshold without modifications."""
        threshold = optimizer.get_dynamic_threshold("hex8_2p")

        # Should be close to baseline (500) with hex8 factor (0.7) = 350
        assert 200 <= threshold <= 750

    def test_threshold_decreases_on_success(self, optimizer):
        """Test threshold decreases with promotion success."""
        initial = optimizer.get_dynamic_threshold("hex8_2p")

        # Build success streak
        for _ in range(5):
            optimizer.record_promotion_success("hex8_2p", 40.0)

        accelerated = optimizer.get_dynamic_threshold("hex8_2p")

        assert accelerated < initial

    def test_threshold_within_bounds(self, optimizer):
        """Test threshold stays within bounds."""
        # Extreme acceleration
        for _ in range(20):
            optimizer.record_promotion_success("hex8_2p", 100.0)

        threshold = optimizer.get_dynamic_threshold("hex8_2p")
        min_threshold = int(optimizer.BASELINE_TRAINING_THRESHOLD * optimizer.MIN_THRESHOLD_MULTIPLIER)

        assert threshold >= min_threshold

    def test_board_complexity_factor_applied(self, optimizer):
        """Test different board types have different thresholds."""
        # hex8 has complexity factor 0.7
        hex_threshold = optimizer.get_dynamic_threshold("hex8_2p")

        # square19 has complexity factor 1.4
        square19_threshold = optimizer.get_dynamic_threshold("square19_2p")

        # square19 should need more data
        assert square19_threshold > hex_threshold


class TestImprovementOptimizerEvaluationInterval:
    """Tests for get_evaluation_interval."""

    def test_baseline_interval(self, optimizer):
        """Test baseline interval without modifications."""
        interval = optimizer.get_evaluation_interval(900)

        # Should be close to base with multiplier 1.0
        assert 450 <= interval <= 1800

    def test_interval_decreases_on_regression(self, optimizer):
        """Test evaluation becomes more frequent after regression."""
        initial = optimizer.get_evaluation_interval(900)

        optimizer.record_regression("hex8_2p", "elo_drop", "severe", 50.0)

        accelerated = optimizer.get_evaluation_interval(900)

        assert accelerated < initial


class TestImprovementOptimizerFastTrack:
    """Tests for should_fast_track_training."""

    def test_fast_track_on_promotion_streak(self, optimizer):
        """Test fast-track on 2+ promotion streak."""
        # Set low parity rate to disable fast-track (default 1.0 triggers it)
        optimizer.record_data_quality(0.90, 0.90)
        assert not optimizer.should_fast_track_training("hex8_2p")

        optimizer.record_promotion_success("hex8_2p", 25.0)
        optimizer.record_promotion_success("hex8_2p", 25.0)

        assert optimizer.should_fast_track_training("hex8_2p")

    def test_fast_track_on_high_quality(self, optimizer):
        """Test fast-track on high data quality."""
        # Set low parity rate first to disable fast-track
        optimizer.record_data_quality(0.90, 0.90)
        assert not optimizer.should_fast_track_training("hex8_2p")

        # Now set high parity rate to enable fast-track
        optimizer.record_data_quality(0.99, 0.99)

        assert optimizer.should_fast_track_training("hex8_2p")


class TestImprovementOptimizerSelfplayPriorityBoost:
    """Tests for get_selfplay_priority_boost."""

    def test_boost_on_promotion_streak(self, optimizer):
        """Test boost increases with promotion streak."""
        base_boost = optimizer.get_selfplay_priority_boost("hex8_2p")

        for _ in range(3):
            optimizer.record_promotion_success("hex8_2p", 25.0)

        streak_boost = optimizer.get_selfplay_priority_boost("hex8_2p")

        assert streak_boost > base_boost

    def test_penalty_on_low_quality(self, optimizer):
        """Test penalty for low data quality."""
        base_boost = optimizer.get_selfplay_priority_boost("hex8_2p")

        optimizer.record_data_quality(0.70, 0.70)

        low_quality_boost = optimizer.get_selfplay_priority_boost("hex8_2p")

        assert low_quality_boost < base_boost

    def test_boost_clamped_to_range(self, optimizer):
        """Test boost is clamped to -0.10 to +0.15."""
        # Max positive
        for _ in range(10):
            optimizer.record_promotion_success("hex8_2p", 100.0)

        boost = optimizer.get_selfplay_priority_boost("hex8_2p")
        assert boost <= 0.15

        # Max negative
        optimizer._state.parity_success_rate = 0.5
        boost = optimizer.get_selfplay_priority_boost("square8_2p")
        assert boost >= -0.10


class TestImprovementOptimizerTrainingAdjustment:
    """Tests for get_training_adjustment."""

    def test_baseline_adjustment(self, optimizer):
        """Test baseline adjustment with no modifications."""
        adj = optimizer.get_training_adjustment("hex8_2p")

        assert adj["lr_multiplier"] == pytest.approx(1.0)
        assert adj["regularization_boost"] == 0.0
        assert adj["batch_size_multiplier"] == 1.0
        assert adj["should_fast_track"] is False
        assert adj["reason"] == "baseline"

    def test_promotion_streak_adjustment(self, optimizer):
        """Test adjustment on promotion streak."""
        for _ in range(4):
            optimizer.record_promotion_success("hex8_2p", 25.0)

        adj = optimizer.get_training_adjustment("hex8_2p")

        # LR should be reduced on streak
        assert adj["lr_multiplier"] < 1.0
        assert adj["should_fast_track"] is True
        assert "promotion_streak" in adj["reason"]

    def test_low_quality_adjustment(self, optimizer):
        """Test adjustment for low data quality."""
        optimizer._state.data_quality_score = 0.5

        adj = optimizer.get_training_adjustment("hex8_2p")

        assert adj["regularization_boost"] > 0
        assert adj["lr_multiplier"] > 1.0
        assert "low_data_quality" in adj["reason"]


class TestImprovementOptimizerMetrics:
    """Tests for get_improvement_metrics."""

    def test_returns_metrics_dict(self, optimizer):
        """Test returns dict with expected keys."""
        metrics = optimizer.get_improvement_metrics()

        assert "consecutive_promotions" in metrics
        assert "promotions_24h" in metrics
        assert "avg_elo_gain" in metrics
        assert "training_runs_24h" in metrics
        assert "threshold_multiplier" in metrics
        assert "config_boosts" in metrics

    def test_metrics_update_with_actions(self, optimizer):
        """Test metrics update with optimizer actions."""
        optimizer.record_promotion_success("hex8_2p", 40.0)
        optimizer.record_training_complete("hex8_2p", 3600.0, 0.05)

        metrics = optimizer.get_improvement_metrics()

        assert metrics["consecutive_promotions"] == 1
        assert metrics["training_runs_24h"] == 1


class TestImprovementOptimizerCallbacks:
    """Tests for callback functionality."""

    def test_register_callback(self, optimizer):
        """Test callback registration."""
        callback = MagicMock()
        optimizer.register_callback(callback)

        optimizer.record_promotion_success("hex8_2p", 25.0)

        callback.assert_called_once()

    def test_callback_receives_recommendation(self, optimizer):
        """Test callback receives OptimizationRecommendation."""
        received_rec = []

        def callback(rec):
            received_rec.append(rec)

        optimizer.register_callback(callback)
        optimizer.record_promotion_success("hex8_2p", 50.0)

        assert len(received_rec) == 1
        assert isinstance(received_rec[0], OptimizationRecommendation)

    def test_callback_errors_dont_crash(self, optimizer):
        """Test callback errors are caught."""

        def bad_callback(rec):
            raise ValueError("Test error")

        optimizer.register_callback(bad_callback)

        # Should not raise
        optimizer.record_promotion_success("hex8_2p", 25.0)


# =============================================================================
# Module-Level Function Tests
# =============================================================================


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    def test_get_improvement_optimizer_returns_singleton(self, temp_state_path):
        """Test get_improvement_optimizer returns singleton."""
        ImprovementOptimizer.reset_instance()

        with patch(
            "app.training.improvement_optimizer.OPTIMIZER_STATE_PATH",
            temp_state_path,
        ):
            opt1 = get_improvement_optimizer()
            opt2 = get_improvement_optimizer()

            assert opt1 is opt2

    def test_should_fast_track_training_function(self, temp_state_path):
        """Test should_fast_track_training function."""
        ImprovementOptimizer.reset_instance()

        with patch(
            "app.training.improvement_optimizer.OPTIMIZER_STATE_PATH",
            temp_state_path,
        ):
            result = should_fast_track_training("hex8_2p")
            assert isinstance(result, bool)

    def test_get_dynamic_threshold_function(self, temp_state_path):
        """Test get_dynamic_threshold function."""
        ImprovementOptimizer.reset_instance()

        with patch(
            "app.training.improvement_optimizer.OPTIMIZER_STATE_PATH",
            temp_state_path,
        ):
            threshold = get_dynamic_threshold("hex8_2p")
            assert isinstance(threshold, int)
            assert threshold > 0

    def test_record_promotion_success_function(self, temp_state_path):
        """Test record_promotion_success function."""
        ImprovementOptimizer.reset_instance()

        with patch(
            "app.training.improvement_optimizer.OPTIMIZER_STATE_PATH",
            temp_state_path,
        ):
            # Should not raise
            record_promotion_success("hex8_2p", 25.0, "model_123")

            metrics = get_improvement_metrics()
            assert metrics["consecutive_promotions"] == 1

    def test_record_regression_function(self, temp_state_path):
        """Test record_regression function."""
        ImprovementOptimizer.reset_instance()

        with patch(
            "app.training.improvement_optimizer.OPTIMIZER_STATE_PATH",
            temp_state_path,
        ):
            # Should not raise
            record_regression("hex8_2p", "elo_drop", "moderate", 30.0, 0.45)

    def test_reset_improvement_optimizer(self, temp_state_path):
        """Test reset_improvement_optimizer clears singleton."""
        ImprovementOptimizer.reset_instance()

        with patch(
            "app.training.improvement_optimizer.OPTIMIZER_STATE_PATH",
            temp_state_path,
        ):
            opt1 = get_improvement_optimizer()
            reset_improvement_optimizer()
            opt2 = get_improvement_optimizer()

            # After reset, should be different instance
            assert opt1 is not opt2


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_config_key(self, optimizer):
        """Test handling of empty config key."""
        threshold = optimizer.get_dynamic_threshold("")
        assert threshold > 0

    def test_cleanup_old_times(self, optimizer):
        """Test old timestamps are cleaned up."""
        # Add old timestamps
        old_time = time.time() - 100000  # > 24 hours ago
        optimizer._state.promotion_times = [old_time, old_time]
        optimizer._state.training_times = [old_time]

        optimizer._cleanup_old_times()

        assert len(optimizer._state.promotion_times) == 0
        assert len(optimizer._state.training_times) == 0

    def test_elo_gains_list_capped(self, optimizer):
        """Test Elo gains list is capped at 50."""
        for i in range(60):
            optimizer.record_promotion_success("hex8_2p", float(i))

        assert len(optimizer._state.elo_gains) <= 50

    def test_threshold_multiplier_bounds(self, optimizer):
        """Test threshold multiplier stays within bounds."""
        # Try to go below minimum
        for _ in range(100):
            optimizer.record_promotion_success("hex8_2p", 100.0)

        assert optimizer._state.threshold_multiplier >= optimizer.MIN_THRESHOLD_MULTIPLIER

    def test_negative_elo_gain(self, optimizer):
        """Test handling of negative Elo gain (shouldn't happen but be safe)."""
        # Should not crash
        rec = optimizer.record_promotion_success("hex8_2p", -10.0)
        assert rec is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
