"""Unit tests for signal_emitter.py module.

Tests signal emission utilities for training feedback including adaptive
training parameters, exploration adjustments, and selfplay updates.
"""

from unittest.mock import MagicMock, patch
import asyncio

import pytest

from app.coordination.signal_emitter import (
    # Constants
    ELO_PLATEAU_THRESHOLD,
    ELO_STRONG_IMPROVEMENT,
    ELO_REGRESSION_THRESHOLD,
    POLICY_LOW_THRESHOLD,
    POLICY_HIGH_THRESHOLD,
    CURRICULUM_WEIGHT_ADJUSTMENT_UP,
    CURRICULUM_WEIGHT_ADJUSTMENT_DOWN,
    EXPLORATION_BOOST_MAX,
    FAILURE_EXPLORATION_BOOST,
    # Enums
    PositionDifficulty,
    SignalPriority,
    # Data classes
    AdaptiveTrainingSignal,
    ExplorationAdjustment,
    SelfplayAdjustment,
    CurriculumFeedback,
    # Pure functions
    compute_adaptive_signal,
    compute_exploration_adjustment,
    compute_curriculum_adjustment,
    compute_selfplay_adjustment,
    # Emission functions
    emit_selfplay_adjustment,
    emit_exploration_adjustment,
    emit_adaptive_training_signal,
    emit_curriculum_training_feedback,
    emit_quality_degraded,
    emit_training_ready,
    # Convenience functions
    emit_velocity_based_selfplay_update,
    emit_quality_based_exploration_update,
)


# ============================================================================
# Constant Tests
# ============================================================================


class TestConstants:
    """Test threshold constants."""

    def test_elo_thresholds_ordering(self):
        """Elo thresholds should be in logical order."""
        assert ELO_REGRESSION_THRESHOLD < 0  # Regression is negative
        assert ELO_PLATEAU_THRESHOLD > 0  # Plateau is small positive
        assert ELO_STRONG_IMPROVEMENT > ELO_PLATEAU_THRESHOLD  # Strong > plateau

    def test_policy_thresholds_ordering(self):
        """Policy accuracy thresholds should be ordered."""
        assert POLICY_LOW_THRESHOLD < POLICY_HIGH_THRESHOLD
        assert 0 < POLICY_LOW_THRESHOLD < 1
        assert 0 < POLICY_HIGH_THRESHOLD < 1

    def test_curriculum_adjustments_opposite(self):
        """Up and down adjustments should have opposite signs."""
        assert CURRICULUM_WEIGHT_ADJUSTMENT_UP > 0
        assert CURRICULUM_WEIGHT_ADJUSTMENT_DOWN < 0


# ============================================================================
# PositionDifficulty Tests
# ============================================================================


class TestPositionDifficulty:
    """Test PositionDifficulty enum."""

    def test_all_levels_defined(self):
        """All 4 difficulty levels should be defined."""
        levels = [
            PositionDifficulty.EASY,
            PositionDifficulty.NORMAL,
            PositionDifficulty.MEDIUM_HARD,
            PositionDifficulty.HARD,
        ]
        assert len(levels) == 4

    def test_string_values(self):
        """Difficulty levels should have correct string values."""
        assert PositionDifficulty.EASY.value == "easy"
        assert PositionDifficulty.NORMAL.value == "normal"
        assert PositionDifficulty.MEDIUM_HARD.value == "medium-hard"
        assert PositionDifficulty.HARD.value == "hard"

    def test_is_string_enum(self):
        """PositionDifficulty should be a string enum."""
        assert isinstance(PositionDifficulty.NORMAL, str)


# ============================================================================
# SignalPriority Tests
# ============================================================================


class TestSignalPriority:
    """Test SignalPriority enum."""

    def test_all_levels_defined(self):
        """All 5 priority levels should be defined."""
        levels = [
            SignalPriority.LOW,
            SignalPriority.NORMAL,
            SignalPriority.HIGH,
            SignalPriority.URGENT,
            SignalPriority.CRITICAL,
        ]
        assert len(levels) == 5

    def test_string_values(self):
        """Priority levels should have correct string values."""
        assert SignalPriority.LOW.value == "low"
        assert SignalPriority.NORMAL.value == "normal"
        assert SignalPriority.HIGH.value == "high"
        assert SignalPriority.URGENT.value == "urgent"
        assert SignalPriority.CRITICAL.value == "critical"


# ============================================================================
# AdaptiveTrainingSignal Tests
# ============================================================================


class TestAdaptiveTrainingSignal:
    """Test AdaptiveTrainingSignal dataclass."""

    def test_default_values(self):
        """Should have sensible defaults (no adjustment)."""
        signal = AdaptiveTrainingSignal()
        assert signal.learning_rate_multiplier == 1.0
        assert signal.batch_size_multiplier == 1.0
        assert signal.epochs_extension == 0
        assert signal.gradient_clip_enabled is False
        assert signal.reason == ""

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        signal = AdaptiveTrainingSignal(
            learning_rate_multiplier=0.5,
            epochs_extension=10,
            reason="test",
        )
        d = signal.to_dict()
        assert d["learning_rate_multiplier"] == 0.5
        assert d["epochs_extension"] == 10
        assert d["reason"] == "test"

    def test_has_adjustment_true(self):
        """Should detect when adjustments exist."""
        assert AdaptiveTrainingSignal(learning_rate_multiplier=0.5).has_adjustment
        assert AdaptiveTrainingSignal(batch_size_multiplier=1.5).has_adjustment
        assert AdaptiveTrainingSignal(epochs_extension=5).has_adjustment
        assert AdaptiveTrainingSignal(gradient_clip_enabled=True).has_adjustment

    def test_has_adjustment_false(self):
        """Should detect no adjustment with defaults."""
        assert not AdaptiveTrainingSignal().has_adjustment


# ============================================================================
# ExplorationAdjustment Tests
# ============================================================================


class TestExplorationAdjustment:
    """Test ExplorationAdjustment dataclass."""

    def test_default_values(self):
        """Should have sensible defaults (normal, no adjustment)."""
        adj = ExplorationAdjustment()
        assert adj.position_difficulty == PositionDifficulty.NORMAL
        assert adj.mcts_budget_multiplier == 1.0
        assert adj.exploration_temp_boost == 1.0
        assert adj.noise_boost == 0.0

    def test_to_dict(self):
        """Should convert to dictionary with string enum value."""
        adj = ExplorationAdjustment(
            position_difficulty=PositionDifficulty.HARD,
            mcts_budget_multiplier=1.5,
        )
        d = adj.to_dict()
        assert d["position_difficulty"] == "hard"  # String value
        assert d["mcts_budget_multiplier"] == 1.5

    def test_has_adjustment_true(self):
        """Should detect adjustments from baseline."""
        assert ExplorationAdjustment(
            position_difficulty=PositionDifficulty.HARD
        ).has_adjustment
        assert ExplorationAdjustment(mcts_budget_multiplier=1.2).has_adjustment
        assert ExplorationAdjustment(exploration_temp_boost=1.1).has_adjustment
        assert ExplorationAdjustment(noise_boost=0.05).has_adjustment

    def test_has_adjustment_false(self):
        """Should detect no adjustment with defaults."""
        assert not ExplorationAdjustment().has_adjustment


# ============================================================================
# SelfplayAdjustment Tests
# ============================================================================


class TestSelfplayAdjustment:
    """Test SelfplayAdjustment dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        adj = SelfplayAdjustment()
        assert adj.search_budget == 150
        assert adj.exploration_boost == 1.0
        assert adj.priority == SignalPriority.NORMAL

    def test_to_dict(self):
        """Should convert to dictionary with string priority."""
        adj = SelfplayAdjustment(
            elo_gap=500.0,
            velocity=15.0,
            priority=SignalPriority.HIGH,
        )
        d = adj.to_dict()
        assert d["elo_gap"] == 500.0
        assert d["velocity"] == 15.0
        assert d["priority"] == "high"  # String value


# ============================================================================
# CurriculumFeedback Tests
# ============================================================================


class TestCurriculumFeedback:
    """Test CurriculumFeedback dataclass."""

    def test_creation(self):
        """Should create with required field."""
        fb = CurriculumFeedback(config_key="hex8_2p")
        assert fb.config_key == "hex8_2p"
        assert fb.policy_accuracy == 0.0
        assert fb.adjustment == 0.0

    def test_to_dict_has_timestamp(self):
        """Should include timestamp in dict."""
        fb = CurriculumFeedback(config_key="hex8_2p", policy_accuracy=0.5)
        d = fb.to_dict()
        assert "timestamp" in d
        assert d["config"] == "hex8_2p"
        assert d["policy_accuracy"] == 0.5


# ============================================================================
# Compute Adaptive Signal Tests
# ============================================================================


class TestComputeAdaptiveSignal:
    """Test compute_adaptive_signal function."""

    def test_strong_improvement(self):
        """Strong Elo improvement should extend training."""
        signal = compute_adaptive_signal(elo_improvement=60.0)
        assert signal.epochs_extension == 10
        assert "Strong improvement" in signal.reason

    def test_plateau(self):
        """Plateau should reduce LR and enable gradient clipping."""
        signal = compute_adaptive_signal(elo_improvement=5.0)
        assert signal.learning_rate_multiplier == 0.5
        assert signal.batch_size_multiplier == 1.5
        assert signal.gradient_clip_enabled is True
        assert "Plateau" in signal.reason

    def test_regression(self):
        """Regression should aggressively reduce LR."""
        signal = compute_adaptive_signal(elo_improvement=-40.0)
        assert signal.learning_rate_multiplier == 0.2
        assert signal.gradient_clip_enabled is True
        assert signal.epochs_extension == 0  # Don't extend on regression
        assert "Regression" in signal.reason

    def test_moderate_improvement(self):
        """Moderate improvement should make no changes."""
        signal = compute_adaptive_signal(elo_improvement=25.0)
        assert signal.learning_rate_multiplier == 1.0
        assert signal.epochs_extension == 0
        assert not signal.has_adjustment

    def test_boundary_strong(self):
        """Exactly at strong improvement threshold."""
        # Just above threshold
        signal = compute_adaptive_signal(elo_improvement=50.1)
        assert signal.epochs_extension == 10

    def test_boundary_plateau(self):
        """Exactly at plateau threshold."""
        # Just below plateau
        signal = compute_adaptive_signal(elo_improvement=9.9)
        assert signal.learning_rate_multiplier == 0.5


# ============================================================================
# Compute Exploration Adjustment Tests
# ============================================================================


class TestComputeExplorationAdjustment:
    """Test compute_exploration_adjustment function."""

    def test_very_low_quality(self):
        """Very low quality should trigger aggressive exploration."""
        adj = compute_exploration_adjustment(quality_score=0.3)
        assert adj.position_difficulty == PositionDifficulty.HARD
        assert adj.mcts_budget_multiplier == 1.5
        assert adj.exploration_temp_boost == 1.3
        assert adj.noise_boost == 0.10

    def test_medium_quality(self):
        """Medium quality should trigger moderate exploration boost."""
        adj = compute_exploration_adjustment(quality_score=0.6)
        assert adj.position_difficulty == PositionDifficulty.MEDIUM_HARD
        assert adj.mcts_budget_multiplier == 1.2
        assert adj.exploration_temp_boost == 1.15

    def test_high_quality(self):
        """High quality can reduce exploration budget."""
        adj = compute_exploration_adjustment(quality_score=0.95)
        assert adj.position_difficulty == PositionDifficulty.NORMAL
        assert adj.mcts_budget_multiplier == 0.8
        assert adj.exploration_temp_boost == 1.0

    def test_normal_quality(self):
        """Normal quality (0.7-0.9) should have no adjustment."""
        adj = compute_exploration_adjustment(quality_score=0.8)
        assert adj.position_difficulty == PositionDifficulty.NORMAL
        assert adj.mcts_budget_multiplier == 1.0
        assert not adj.has_adjustment

    def test_declining_trend_boosts(self):
        """Declining trend should boost exploration."""
        # Normal quality with declining trend
        adj = compute_exploration_adjustment(quality_score=0.75, trend="declining")
        assert adj.exploration_temp_boost > 1.0
        assert adj.mcts_budget_multiplier >= 1.3

    def test_improving_trend_no_effect(self):
        """Improving trend should not affect exploration."""
        adj = compute_exploration_adjustment(quality_score=0.8, trend="improving")
        assert not adj.has_adjustment


# ============================================================================
# Compute Curriculum Adjustment Tests
# ============================================================================


class TestComputeCurriculumAdjustment:
    """Test compute_curriculum_adjustment function."""

    def test_low_policy_accuracy_boosts(self):
        """Low policy accuracy should boost curriculum weight."""
        fb = compute_curriculum_adjustment(policy_accuracy=0.30, value_accuracy=0.50)
        assert fb.adjustment == CURRICULUM_WEIGHT_ADJUSTMENT_UP
        assert fb.new_weight > 1.0

    def test_high_policy_accuracy_reduces(self):
        """High policy accuracy should reduce curriculum weight."""
        fb = compute_curriculum_adjustment(policy_accuracy=0.80, value_accuracy=0.70)
        assert fb.adjustment == CURRICULUM_WEIGHT_ADJUSTMENT_DOWN
        assert fb.new_weight < 1.0

    def test_medium_policy_accuracy_no_change(self):
        """Medium policy accuracy should not change weight."""
        fb = compute_curriculum_adjustment(policy_accuracy=0.55, value_accuracy=0.60)
        assert fb.adjustment == 0.0
        assert fb.new_weight == 1.0

    def test_weight_capped_at_max(self):
        """Weight should not exceed max."""
        fb = compute_curriculum_adjustment(
            policy_accuracy=0.30,
            value_accuracy=0.50,
            current_weight=1.95,
            weight_max=2.0,
        )
        assert fb.new_weight == 2.0

    def test_weight_capped_at_min(self):
        """Weight should not go below min."""
        fb = compute_curriculum_adjustment(
            policy_accuracy=0.80,
            value_accuracy=0.70,
            current_weight=0.55,
            weight_min=0.5,
        )
        assert fb.new_weight == 0.5


# ============================================================================
# Compute Selfplay Adjustment Tests
# ============================================================================


class TestComputeSelfplayAdjustment:
    """Test compute_selfplay_adjustment function."""

    def test_normal_conditions(self):
        """Normal Elo gap and velocity should be normal priority."""
        adj = compute_selfplay_adjustment(elo_gap=200.0, velocity=20.0)
        assert adj.priority == SignalPriority.NORMAL

    def test_high_elo_gap(self):
        """High Elo gap should increase priority."""
        adj = compute_selfplay_adjustment(elo_gap=600.0, velocity=20.0)
        assert adj.priority == SignalPriority.HIGH

    def test_very_high_elo_gap(self):
        """Very high Elo gap should be urgent."""
        adj = compute_selfplay_adjustment(elo_gap=1200.0, velocity=20.0)
        assert adj.priority == SignalPriority.URGENT

    def test_low_velocity_plateau(self):
        """Low velocity (plateau) should increase priority."""
        adj = compute_selfplay_adjustment(elo_gap=100.0, velocity=5.0)
        assert adj.priority == SignalPriority.HIGH

    def test_includes_parameters(self):
        """Should include all parameters in adjustment."""
        adj = compute_selfplay_adjustment(
            elo_gap=300.0,
            velocity=15.0,
            current_search_budget=200,
            current_exploration_boost=1.2,
        )
        assert adj.search_budget == 200
        assert adj.exploration_boost == 1.2
        assert adj.elo_gap == 300.0
        assert adj.velocity == 15.0


# ============================================================================
# Emission Function Tests
# ============================================================================


class TestEmitSelfplayAdjustment:
    """Test emit_selfplay_adjustment function."""

    @patch("app.coordination.signal_emitter._emit_event_sync")
    def test_emits_correct_event(self, mock_emit):
        """Should emit SELFPLAY_TARGET_UPDATED with correct payload."""
        mock_emit.return_value = True

        adj = SelfplayAdjustment(
            elo_gap=300.0,
            velocity=15.0,
            priority=SignalPriority.HIGH,
        )
        result = emit_selfplay_adjustment("hex8_2p", adj)

        assert result is True
        mock_emit.assert_called_once()
        args = mock_emit.call_args
        assert args[0][0] == "SELFPLAY_TARGET_UPDATED"
        payload = args[0][1]
        assert payload["config_key"] == "hex8_2p"
        assert payload["elo_gap"] == 300.0


class TestEmitExplorationAdjustment:
    """Test emit_exploration_adjustment function."""

    @patch("app.coordination.signal_emitter._emit_event_sync")
    def test_emits_when_has_adjustment(self, mock_emit):
        """Should emit when adjustment differs from baseline."""
        mock_emit.return_value = True

        adj = ExplorationAdjustment(mcts_budget_multiplier=1.5)
        result = emit_exploration_adjustment("hex8_2p", adj)

        assert result is True
        mock_emit.assert_called_once()

    @patch("app.coordination.signal_emitter._emit_event_sync")
    def test_skips_when_no_adjustment(self, mock_emit):
        """Should not emit when no adjustment needed."""
        adj = ExplorationAdjustment()  # All defaults
        result = emit_exploration_adjustment("hex8_2p", adj)

        assert result is True  # Success (no emission needed)
        mock_emit.assert_not_called()


class TestEmitAdaptiveTrainingSignal:
    """Test emit_adaptive_training_signal function."""

    @patch("app.coordination.signal_emitter._emit_event_sync")
    def test_emits_when_has_adjustment(self, mock_emit):
        """Should emit when signal has adjustments."""
        mock_emit.return_value = True

        signal = AdaptiveTrainingSignal(learning_rate_multiplier=0.5)
        result = emit_adaptive_training_signal("hex8_2p", signal)

        assert result is True
        mock_emit.assert_called_once()
        args = mock_emit.call_args
        assert args[0][0] == "ADAPTIVE_PARAMS_CHANGED"

    @patch("app.coordination.signal_emitter._emit_event_sync")
    def test_skips_when_no_adjustment(self, mock_emit):
        """Should not emit when no adjustment."""
        signal = AdaptiveTrainingSignal()  # All defaults
        result = emit_adaptive_training_signal("hex8_2p", signal)

        assert result is True
        mock_emit.assert_not_called()


class TestEmitCurriculumTrainingFeedback:
    """Test emit_curriculum_training_feedback function."""

    @patch("app.coordination.signal_emitter._emit_event_sync")
    def test_emits_curriculum_rebalanced(self, mock_emit):
        """Should emit CURRICULUM_REBALANCED event."""
        mock_emit.return_value = True

        feedback = CurriculumFeedback(
            config_key="hex8_2p",
            policy_accuracy=0.45,
            adjustment=0.15,
        )
        result = emit_curriculum_training_feedback(feedback)

        assert result is True
        mock_emit.assert_called_once()
        args = mock_emit.call_args
        assert args[0][0] == "CURRICULUM_REBALANCED"

    @patch("app.coordination.signal_emitter._emit_event_sync")
    def test_includes_weights_dict(self, mock_emit):
        """Should include weights dict when provided."""
        mock_emit.return_value = True

        feedback = CurriculumFeedback(config_key="hex8_2p")
        weights = {"hex8_2p": 1.15, "square8_2p": 0.95}
        emit_curriculum_training_feedback(feedback, weights_dict=weights)

        payload = mock_emit.call_args[0][1]
        assert payload["new_weights"] == weights


class TestEmitQualityDegraded:
    """Test emit_quality_degraded function."""

    @patch("app.coordination.signal_emitter._emit_event_sync")
    def test_emits_quality_degraded(self, mock_emit):
        """Should emit QUALITY_DEGRADED event."""
        mock_emit.return_value = True

        result = emit_quality_degraded(
            config_key="hex8_2p",
            quality_score=0.55,
            threshold=0.70,
            previous_score=0.75,
        )

        assert result is True
        mock_emit.assert_called_once()
        args = mock_emit.call_args
        assert args[0][0] == "QUALITY_DEGRADED"
        payload = args[0][1]
        assert payload["config_key"] == "hex8_2p"
        assert payload["quality_score"] == 0.55
        assert payload["threshold"] == 0.70


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestEmitVelocityBasedSelfplayUpdate:
    """Test emit_velocity_based_selfplay_update convenience function."""

    @patch("app.coordination.signal_emitter._emit_event_sync")
    def test_computes_and_emits(self, mock_emit):
        """Should compute adjustment and emit."""
        mock_emit.return_value = True

        result = emit_velocity_based_selfplay_update(
            config_key="hex8_2p",
            elo_gap=600.0,
            velocity=5.0,
        )

        assert result is True
        mock_emit.assert_called_once()
        # Should detect high priority from elo_gap > 500
        payload = mock_emit.call_args[0][1]
        assert payload["priority"] == "high"


class TestEmitQualityBasedExplorationUpdate:
    """Test emit_quality_based_exploration_update convenience function."""

    @patch("app.coordination.signal_emitter._emit_event_sync")
    def test_computes_and_emits(self, mock_emit):
        """Should compute adjustment and emit when needed."""
        mock_emit.return_value = True

        result = emit_quality_based_exploration_update(
            config_key="hex8_2p",
            quality_score=0.4,
            trend="declining",
        )

        assert result is True
        mock_emit.assert_called_once()

    @patch("app.coordination.signal_emitter._emit_event_sync")
    def test_skips_when_no_adjustment(self, mock_emit):
        """Should skip when quality is normal and stable."""
        result = emit_quality_based_exploration_update(
            config_key="hex8_2p",
            quality_score=0.8,
            trend="stable",
        )

        assert result is True  # Success (no emission needed)
        mock_emit.assert_not_called()


# ============================================================================
# Integration Tests
# ============================================================================


class TestSignalEmitterIntegration:
    """Integration tests for signal emission workflow."""

    @patch("app.coordination.signal_emitter._emit_event_sync")
    def test_full_training_feedback_workflow(self, mock_emit):
        """Test complete training feedback signal workflow."""
        mock_emit.return_value = True

        # Step 1: Compute adaptive signal from Elo improvement
        signal = compute_adaptive_signal(elo_improvement=55.0)
        assert signal.epochs_extension == 10

        # Step 2: Emit the signal
        result = emit_adaptive_training_signal("hex8_2p", signal)
        assert result is True

        # Step 3: Compute curriculum adjustment
        curriculum = compute_curriculum_adjustment(
            policy_accuracy=0.35,
            value_accuracy=0.50,
        )
        assert curriculum.adjustment > 0  # Should boost

        # Step 4: Emit curriculum feedback
        curriculum.config_key = "hex8_2p"
        result = emit_curriculum_training_feedback(curriculum)
        assert result is True

        # Verify both events were emitted
        assert mock_emit.call_count == 2

    @patch("app.coordination.signal_emitter._emit_event_sync")
    def test_quality_degradation_workflow(self, mock_emit):
        """Test quality degradation signal workflow."""
        mock_emit.return_value = True

        # Step 1: Compute exploration adjustment for low quality
        adj = compute_exploration_adjustment(quality_score=0.4, trend="declining")
        assert adj.position_difficulty == PositionDifficulty.HARD

        # Step 2: Emit exploration adjustment
        emit_exploration_adjustment("hex8_2p", adj)

        # Step 3: Emit quality degraded signal
        emit_quality_degraded(
            config_key="hex8_2p",
            quality_score=0.4,
            threshold=0.7,
            previous_score=0.75,
        )

        # Verify both events were emitted
        assert mock_emit.call_count == 2
        event_types = [call[0][0] for call in mock_emit.call_args_list]
        assert "EXPLORATION_ADJUSTED" in event_types
        assert "QUALITY_DEGRADED" in event_types
