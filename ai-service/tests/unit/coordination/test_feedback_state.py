"""Tests for canonical FeedbackState classes (December 2025).

This module tests the unified FeedbackState hierarchy:
- CanonicalFeedbackState (base class)
- SignalFeedbackState (for orchestrator)
- MonitoringFeedbackState (for monitoring/decisions)

Coverage:
- Field initialization and defaults
- Serialization (to_dict, from_dict)
- Inheritance chain
- Method implementations (update_*, compute_*)
- Edge cases and boundary conditions
"""

from __future__ import annotations

import time
from collections import deque

import pytest


class TestCanonicalFeedbackState:
    """Tests for CanonicalFeedbackState base class."""

    def test_default_initialization(self):
        """Should create state with default values."""
        from app.coordination.feedback_state import CanonicalFeedbackState

        state = CanonicalFeedbackState(config_key="hex8_2p")

        assert state.config_key == "hex8_2p"
        assert state.quality_score == 0.0
        assert state.training_accuracy == 0.0
        assert state.win_rate == 0.5
        assert state.elo_current == 1500.0
        assert state.elo_velocity == 0.0
        assert state.consecutive_successes == 0
        assert state.consecutive_failures == 0
        assert state.parity_failure_rate == 0.0
        assert state.data_quality_score == 1.0
        assert state.curriculum_weight == 1.0

    def test_custom_initialization(self):
        """Should accept custom values."""
        from app.coordination.feedback_state import CanonicalFeedbackState

        state = CanonicalFeedbackState(
            config_key="square8_4p",
            quality_score=0.85,
            elo_current=1650.0,
            consecutive_successes=3,
        )

        assert state.config_key == "square8_4p"
        assert state.quality_score == 0.85
        assert state.elo_current == 1650.0
        assert state.consecutive_successes == 3

    def test_elo_history_is_deque(self):
        """elo_history should be a deque with maxlen."""
        from app.coordination.feedback_state import CanonicalFeedbackState

        state = CanonicalFeedbackState(config_key="hex8_2p")

        assert isinstance(state.elo_history, deque)
        assert state.elo_history.maxlen == 20

    def test_elo_history_independent_instances(self):
        """Each instance should have its own elo_history."""
        from app.coordination.feedback_state import CanonicalFeedbackState

        state1 = CanonicalFeedbackState(config_key="hex8_2p")
        state2 = CanonicalFeedbackState(config_key="hex8_3p")

        state1.elo_history.append((time.time(), 1600.0))

        assert len(state1.elo_history) == 1
        assert len(state2.elo_history) == 0

    def test_to_dict_includes_all_fields(self):
        """to_dict should serialize all important fields."""
        from app.coordination.feedback_state import CanonicalFeedbackState

        state = CanonicalFeedbackState(
            config_key="hex8_2p",
            quality_score=0.9,
            elo_current=1700.0,
            curriculum_weight=1.5,
        )

        result = state.to_dict()

        assert result["config_key"] == "hex8_2p"
        assert result["quality_score"] == 0.9
        assert result["elo_current"] == 1700.0
        assert result["curriculum_weight"] == 1.5
        assert "last_selfplay_time" in result
        assert "last_training_time" in result

    def test_from_dict_creates_state(self):
        """from_dict should deserialize properly."""
        from app.coordination.feedback_state import CanonicalFeedbackState

        data = {
            "config_key": "square8_2p",
            "quality_score": 0.75,
            "elo_current": 1550.0,
            "consecutive_failures": 2,
        }

        state = CanonicalFeedbackState.from_dict(data)

        assert state.config_key == "square8_2p"
        assert state.quality_score == 0.75
        assert state.elo_current == 1550.0
        assert state.consecutive_failures == 2

    def test_from_dict_uses_defaults_for_missing(self):
        """from_dict should use defaults for missing fields."""
        from app.coordination.feedback_state import CanonicalFeedbackState

        data = {"config_key": "hex8_2p"}

        state = CanonicalFeedbackState.from_dict(data)

        assert state.quality_score == 0.0
        assert state.elo_current == 1500.0
        assert state.curriculum_weight == 1.0

    def test_timing_fields_default_to_zero(self):
        """Timing fields should default to 0.0."""
        from app.coordination.feedback_state import CanonicalFeedbackState

        state = CanonicalFeedbackState(config_key="hex8_2p")

        assert state.last_selfplay_time == 0.0
        assert state.last_training_time == 0.0
        assert state.last_evaluation_time == 0.0
        assert state.last_promotion_time == 0.0


class TestSignalFeedbackState:
    """Tests for SignalFeedbackState subclass."""

    def test_inherits_from_canonical(self):
        """Should inherit from CanonicalFeedbackState."""
        from app.coordination.feedback_state import (
            CanonicalFeedbackState,
            SignalFeedbackState,
        )

        state = SignalFeedbackState(config_key="hex8_2p")
        assert isinstance(state, CanonicalFeedbackState)

    def test_default_signal_values(self):
        """Should have correct signal defaults."""
        from app.coordination.feedback_state import SignalFeedbackState

        state = SignalFeedbackState(config_key="hex8_2p")

        assert state.training_intensity == "normal"
        assert state.exploration_boost == 1.0
        assert state.data_freshness_hours == float("inf")
        assert state.consecutive_anomalies == 0
        assert state.quality_penalties_applied == 0
        assert state.last_adjustment_time == 0.0

    def test_signal_assignment(self):
        """Should allow signal value assignment."""
        from app.coordination.feedback_state import SignalFeedbackState

        state = SignalFeedbackState(config_key="hex8_2p")
        state.training_intensity = "hot_path"
        state.exploration_boost = 1.5

        assert state.training_intensity == "hot_path"
        assert state.exploration_boost == 1.5

    def test_to_dict_includes_signal_fields(self):
        """to_dict should include signal-specific fields."""
        from app.coordination.feedback_state import SignalFeedbackState

        state = SignalFeedbackState(config_key="hex8_2p")
        state.training_intensity = "accelerated"
        state.exploration_boost = 1.3

        result = state.to_dict()

        assert result["training_intensity"] == "accelerated"
        assert result["exploration_boost"] == 1.3
        assert result["data_freshness_hours"] == float("inf")
        assert result["consecutive_anomalies"] == 0
        # Also includes base fields
        assert result["config_key"] == "hex8_2p"
        assert result["elo_current"] == 1500.0

    def test_intensity_values(self):
        """Should accept valid intensity values."""
        from app.coordination.feedback_state import SignalFeedbackState

        valid_intensities = ["paused", "reduced", "normal", "accelerated", "hot_path"]

        for intensity in valid_intensities:
            state = SignalFeedbackState(config_key="hex8_2p")
            state.training_intensity = intensity
            assert state.training_intensity == intensity


class TestMonitoringFeedbackState:
    """Tests for MonitoringFeedbackState subclass."""

    def test_inherits_from_canonical(self):
        """Should inherit from CanonicalFeedbackState."""
        from app.coordination.feedback_state import (
            CanonicalFeedbackState,
            MonitoringFeedbackState,
        )

        state = MonitoringFeedbackState(config_key="hex8_2p")
        assert isinstance(state, CanonicalFeedbackState)

    def test_default_monitoring_values(self):
        """Should have correct monitoring defaults."""
        from app.coordination.feedback_state import MonitoringFeedbackState

        state = MonitoringFeedbackState(config_key="hex8_2p")

        assert state.elo_trend == 0.0
        assert state.elo_peak == 1500.0
        assert state.elo_plateau_count == 0
        assert state.win_rate_trend == 0.0
        assert state.consecutive_high_win_rate == 0
        assert state.consecutive_low_win_rate == 0
        assert state.parity_checks_total == 0
        assert state.urgency_score == 0.0

    def test_update_parity_passed(self):
        """update_parity should decrease failure rate on pass."""
        from app.coordination.feedback_state import MonitoringFeedbackState

        state = MonitoringFeedbackState(config_key="hex8_2p")
        state.parity_failure_rate = 0.5

        state.update_parity(passed=True, alpha=0.1)

        # 0.1 * 0.0 + 0.9 * 0.5 = 0.45
        assert state.parity_failure_rate == pytest.approx(0.45)
        assert state.parity_checks_total == 1

    def test_update_parity_failed(self):
        """update_parity should increase failure rate on fail."""
        from app.coordination.feedback_state import MonitoringFeedbackState

        state = MonitoringFeedbackState(config_key="hex8_2p")
        state.parity_failure_rate = 0.0

        state.update_parity(passed=False, alpha=0.1)

        # 0.1 * 1.0 + 0.9 * 0.0 = 0.1
        assert state.parity_failure_rate == pytest.approx(0.1)
        assert state.parity_checks_total == 1

    def test_update_elo_tracks_trend(self):
        """update_elo should track Elo trend."""
        from app.coordination.feedback_state import MonitoringFeedbackState

        state = MonitoringFeedbackState(config_key="hex8_2p")
        state.elo_current = 1500.0

        state.update_elo(1550.0)

        assert state.elo_current == 1550.0
        assert state.elo_trend == 50.0
        assert state.elo_peak == 1550.0

    def test_update_elo_tracks_peak(self):
        """update_elo should track peak Elo."""
        from app.coordination.feedback_state import MonitoringFeedbackState

        state = MonitoringFeedbackState(config_key="hex8_2p")
        state.elo_current = 1600.0
        state.elo_peak = 1600.0

        # Drop below peak
        state.update_elo(1550.0)

        assert state.elo_current == 1550.0
        assert state.elo_peak == 1600.0  # Peak unchanged

    def test_update_elo_plateau_detection(self):
        """update_elo should detect plateaus."""
        from app.coordination.feedback_state import MonitoringFeedbackState

        state = MonitoringFeedbackState(config_key="hex8_2p")
        state.elo_current = 1500.0

        # Small change within threshold
        state.update_elo(1505.0, plateau_threshold=15.0)
        assert state.elo_plateau_count == 1

        state.update_elo(1510.0, plateau_threshold=15.0)
        assert state.elo_plateau_count == 2

        # Large change resets plateau
        state.update_elo(1530.0, plateau_threshold=15.0)
        assert state.elo_plateau_count == 0

    def test_update_elo_adds_to_history(self):
        """update_elo should add entries to elo_history."""
        from app.coordination.feedback_state import MonitoringFeedbackState

        state = MonitoringFeedbackState(config_key="hex8_2p")

        state.update_elo(1550.0)
        state.update_elo(1600.0)

        assert len(state.elo_history) == 2
        assert state.elo_history[-1][1] == 1600.0

    def test_update_win_rate_tracks_trend(self):
        """update_win_rate should track trend."""
        from app.coordination.feedback_state import MonitoringFeedbackState

        state = MonitoringFeedbackState(config_key="hex8_2p")
        state.win_rate = 0.5

        state.update_win_rate(0.65)

        assert state.win_rate == 0.65
        assert state.win_rate_trend == pytest.approx(0.15)

    def test_update_win_rate_high_streak(self):
        """update_win_rate should track high win rate streaks."""
        from app.coordination.feedback_state import MonitoringFeedbackState

        state = MonitoringFeedbackState(config_key="hex8_2p")

        state.update_win_rate(0.75)
        assert state.consecutive_high_win_rate == 1
        assert state.consecutive_low_win_rate == 0

        state.update_win_rate(0.80)
        assert state.consecutive_high_win_rate == 2

    def test_update_win_rate_low_streak(self):
        """update_win_rate should track low win rate streaks."""
        from app.coordination.feedback_state import MonitoringFeedbackState

        state = MonitoringFeedbackState(config_key="hex8_2p")

        state.update_win_rate(0.40)
        assert state.consecutive_low_win_rate == 1
        assert state.consecutive_high_win_rate == 0

        state.update_win_rate(0.35)
        assert state.consecutive_low_win_rate == 2

    def test_update_win_rate_resets_streaks(self):
        """update_win_rate should reset streaks on mid-range rate."""
        from app.coordination.feedback_state import MonitoringFeedbackState

        state = MonitoringFeedbackState(config_key="hex8_2p")
        state.consecutive_high_win_rate = 3

        state.update_win_rate(0.55)  # Mid-range

        assert state.consecutive_high_win_rate == 0
        assert state.consecutive_low_win_rate == 0

    def test_compute_urgency_low_win_rate(self):
        """compute_urgency should increase for low win rate."""
        from app.coordination.feedback_state import MonitoringFeedbackState

        state = MonitoringFeedbackState(config_key="hex8_2p")
        state.win_rate = 0.3

        urgency = state.compute_urgency()

        assert urgency > 0.0
        assert state.urgency_score == urgency

    def test_compute_urgency_plateau(self):
        """compute_urgency should increase for Elo plateau."""
        from app.coordination.feedback_state import MonitoringFeedbackState

        state = MonitoringFeedbackState(config_key="hex8_2p")
        state.win_rate = 0.6
        state.elo_plateau_count = 5

        urgency = state.compute_urgency()

        assert urgency > 0.0
        assert urgency <= 1.0

    def test_compute_urgency_poor_data_reduces(self):
        """compute_urgency should be reduced by poor data quality."""
        from app.coordination.feedback_state import MonitoringFeedbackState

        state = MonitoringFeedbackState(config_key="hex8_2p")
        state.win_rate = 0.3
        state.parity_failure_rate = 0.0

        urgency_good = state.compute_urgency()

        state.parity_failure_rate = 0.2

        urgency_poor = state.compute_urgency()

        assert urgency_poor < urgency_good

    def test_compute_urgency_capped_at_one(self):
        """compute_urgency should be capped at 1.0."""
        from app.coordination.feedback_state import MonitoringFeedbackState

        state = MonitoringFeedbackState(config_key="hex8_2p")
        state.win_rate = 0.1
        state.win_rate_trend = -0.5
        state.elo_plateau_count = 10
        state.curriculum_weight = 3.0

        urgency = state.compute_urgency()

        assert urgency <= 1.0

    def test_compute_data_quality(self):
        """compute_data_quality should compute composite score."""
        from app.coordination.feedback_state import MonitoringFeedbackState

        state = MonitoringFeedbackState(config_key="hex8_2p")
        state.parity_failure_rate = 0.0

        quality = state.compute_data_quality(
            sample_diversity=1.0,
            avg_game_length=50.0,
        )

        assert quality > 0.9  # Near perfect with no failures
        assert state.data_quality_score == quality

    def test_compute_data_quality_with_failures(self):
        """compute_data_quality should penalize parity failures."""
        from app.coordination.feedback_state import MonitoringFeedbackState

        state = MonitoringFeedbackState(config_key="hex8_2p")
        state.parity_failure_rate = 0.5

        quality = state.compute_data_quality()

        # With 50% parity failures: 0.5*0.4 + 1.0*0.3 + 1.0*0.3 = 0.8
        assert quality <= 0.8  # Penalized for failures

    def test_compute_data_quality_short_games(self):
        """compute_data_quality should penalize short games."""
        from app.coordination.feedback_state import MonitoringFeedbackState

        state = MonitoringFeedbackState(config_key="hex8_2p")
        state.parity_failure_rate = 0.0

        quality = state.compute_data_quality(
            avg_game_length=5.0,  # Very short
            min_game_length=10.0,
        )

        assert quality < 1.0

    def test_is_data_quality_acceptable_true(self):
        """is_data_quality_acceptable should return True above threshold."""
        from app.coordination.feedback_state import MonitoringFeedbackState

        state = MonitoringFeedbackState(config_key="hex8_2p")
        state.data_quality_score = 0.8

        assert state.is_data_quality_acceptable(threshold=0.7) is True

    def test_is_data_quality_acceptable_false(self):
        """is_data_quality_acceptable should return False below threshold."""
        from app.coordination.feedback_state import MonitoringFeedbackState

        state = MonitoringFeedbackState(config_key="hex8_2p")
        state.data_quality_score = 0.5

        assert state.is_data_quality_acceptable(threshold=0.7) is False

    def test_to_dict_includes_monitoring_fields(self):
        """to_dict should include monitoring-specific fields."""
        from app.coordination.feedback_state import MonitoringFeedbackState

        state = MonitoringFeedbackState(config_key="hex8_2p")
        state.elo_trend = 50.0
        state.elo_peak = 1700.0
        state.urgency_score = 0.5

        result = state.to_dict()

        assert result["elo_trend"] == 50.0
        assert result["elo_peak"] == 1700.0
        assert result["urgency_score"] == 0.5
        # Also includes base fields
        assert result["config_key"] == "hex8_2p"


class TestModuleExports:
    """Tests for module-level exports."""

    def test_all_exports_exist(self):
        """All items in __all__ should be importable."""
        from app.coordination.feedback_state import __all__

        import app.coordination.feedback_state as module

        for name in __all__:
            assert hasattr(module, name), f"Missing export: {name}"

    def test_feedbackstate_alias(self):
        """FeedbackState should be alias for CanonicalFeedbackState."""
        from app.coordination.feedback_state import (
            CanonicalFeedbackState,
            FeedbackState,
        )

        assert FeedbackState is CanonicalFeedbackState

    def test_direct_imports(self):
        """All classes should be directly importable."""
        from app.coordination.feedback_state import (
            CanonicalFeedbackState,
            FeedbackState,
            MonitoringFeedbackState,
            SignalFeedbackState,
        )

        assert CanonicalFeedbackState is not None
        assert SignalFeedbackState is not None
        assert MonitoringFeedbackState is not None
        assert FeedbackState is not None


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_config_key(self):
        """Should handle empty config_key."""
        from app.coordination.feedback_state import CanonicalFeedbackState

        state = CanonicalFeedbackState(config_key="")
        assert state.config_key == ""

    def test_negative_elo(self):
        """Should handle negative Elo values."""
        from app.coordination.feedback_state import MonitoringFeedbackState

        state = MonitoringFeedbackState(config_key="hex8_2p")
        state.elo_current = 1000.0

        state.update_elo(-500.0)

        assert state.elo_current == -500.0
        assert state.elo_trend == -1500.0

    def test_extreme_win_rates(self):
        """Should handle boundary win rates (0.0 and 1.0)."""
        from app.coordination.feedback_state import MonitoringFeedbackState

        state = MonitoringFeedbackState(config_key="hex8_2p")

        state.update_win_rate(0.0)
        assert state.consecutive_low_win_rate == 1

        state.update_win_rate(1.0)
        assert state.consecutive_high_win_rate == 1

    def test_parity_alpha_extremes(self):
        """update_parity should handle extreme alpha values."""
        from app.coordination.feedback_state import MonitoringFeedbackState

        state = MonitoringFeedbackState(config_key="hex8_2p")
        state.parity_failure_rate = 0.5

        # Alpha = 0 (no update)
        state.update_parity(passed=False, alpha=0.0)
        assert state.parity_failure_rate == 0.5

        # Alpha = 1 (full update)
        state.update_parity(passed=True, alpha=1.0)
        assert state.parity_failure_rate == 0.0

    def test_urgency_with_zero_values(self):
        """compute_urgency should handle all-zero state."""
        from app.coordination.feedback_state import MonitoringFeedbackState

        state = MonitoringFeedbackState(config_key="hex8_2p")
        # All defaults

        urgency = state.compute_urgency()

        assert 0.0 <= urgency <= 1.0
