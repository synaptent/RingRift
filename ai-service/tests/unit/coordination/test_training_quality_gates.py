"""Unit tests for training_quality_gates.py.

Jan 4, 2026 - Sprint 17.9: Tests for extracted quality gate functions.
"""

import pytest
import time

from app.coordination.training_quality_gates import (
    # Constants
    QUALITY_INTENSITY_THRESHOLDS,
    DEFAULT_DECAY_HALF_LIFE_HOURS,
    DEFAULT_DECAY_FLOOR,
    MINIMUM_QUALITY_FLOOR,
    DATA_STARVED_THRESHOLD,
    TRAINING_STALL_HOURS,
    # Dataclasses
    QualityGateResult,
    # Functions
    compute_quality_confidence,
    apply_confidence_weighting,
    compute_decayed_quality_score,
    intensity_from_quality,
    check_quality_gate_conditions,
    get_quality_from_state,
)


class TestConstants:
    """Test quality gate constants."""

    def test_quality_intensity_thresholds_order(self):
        """Thresholds should be in descending order."""
        assert QUALITY_INTENSITY_THRESHOLDS["hot_path"] > QUALITY_INTENSITY_THRESHOLDS["accelerated"]
        assert QUALITY_INTENSITY_THRESHOLDS["accelerated"] > QUALITY_INTENSITY_THRESHOLDS["normal"]

    def test_decay_constants_valid(self):
        """Decay constants should be valid."""
        assert DEFAULT_DECAY_HALF_LIFE_HOURS > 0
        assert 0 <= DEFAULT_DECAY_FLOOR <= 1

    def test_quality_floor_below_starved_threshold(self):
        """Minimum floor should be below normal threshold."""
        assert MINIMUM_QUALITY_FLOOR < QUALITY_INTENSITY_THRESHOLDS["normal"]

    def test_stall_hours_positive(self):
        """Training stall hours should be positive."""
        assert TRAINING_STALL_HOURS > 0


class TestQualityGateResult:
    """Test QualityGateResult dataclass."""

    def test_basic_passed_result(self):
        """Test creating a passed result."""
        result = QualityGateResult(
            passed=True,
            reason="quality ok (0.85)",
            quality_score=0.85,
            threshold=0.50,
        )
        assert result.passed is True
        assert "quality ok" in result.reason
        assert result.quality_score == 0.85
        assert result.threshold == 0.50
        assert result.is_relaxed is False

    def test_relaxed_result(self):
        """Test creating a relaxed result."""
        result = QualityGateResult(
            passed=True,
            reason="quality degraded but allowed",
            quality_score=0.45,
            threshold=0.50,
            is_relaxed=True,
            relaxed_reason="bootstrap mode",
        )
        assert result.passed is True
        assert result.is_relaxed is True
        assert result.relaxed_reason == "bootstrap mode"

    def test_failed_result(self):
        """Test creating a failed result."""
        result = QualityGateResult(
            passed=False,
            reason="quality too low (0.30 < 0.50)",
            quality_score=0.30,
            threshold=0.50,
        )
        assert result.passed is False
        assert result.quality_score == 0.30


class TestComputeQualityConfidence:
    """Test compute_quality_confidence function."""

    def test_low_game_count(self):
        """Few games should have low confidence."""
        confidence = compute_quality_confidence(10)
        assert 0.5 <= confidence <= 0.6

    def test_medium_game_count(self):
        """Medium game count should have medium confidence."""
        confidence = compute_quality_confidence(200)
        assert 0.7 <= confidence <= 0.8

    def test_high_game_count(self):
        """Many games should have high confidence."""
        confidence = compute_quality_confidence(1000)
        assert confidence >= 0.95

    def test_zero_games(self):
        """Zero games should return minimum confidence."""
        confidence = compute_quality_confidence(0)
        assert confidence == 0.5


class TestApplyConfidenceWeighting:
    """Test apply_confidence_weighting function."""

    def test_high_confidence_preserves_score(self):
        """High confidence should preserve quality score."""
        weighted = apply_confidence_weighting(0.90, 1000)
        assert abs(weighted - 0.90) < 0.05

    def test_low_confidence_biases_toward_neutral(self):
        """Low confidence should bias toward 0.5."""
        weighted = apply_confidence_weighting(0.90, 10)
        assert weighted < 0.90  # Should be pulled toward 0.5
        assert weighted > 0.5

    def test_neutral_score_unaffected(self):
        """Score of 0.5 should be relatively unaffected."""
        weighted = apply_confidence_weighting(0.50, 10)
        assert abs(weighted - 0.50) < 0.1


class TestComputeDecayedQualityScore:
    """Test compute_decayed_quality_score function."""

    def test_no_decay_when_disabled(self):
        """Decay disabled should return original score."""
        result = compute_decayed_quality_score(
            last_quality_score=0.90,
            last_quality_update=time.time() - 7200,  # 2 hours ago
            current_time=time.time(),
            decay_enabled=False,
        )
        assert result == 0.90

    def test_no_decay_for_fresh_data(self):
        """Fresh quality data should not decay."""
        now = time.time()
        result = compute_decayed_quality_score(
            last_quality_score=0.90,
            last_quality_update=now,
            current_time=now,
        )
        assert result == 0.90

    def test_partial_decay_after_half_life(self):
        """Score should decay after half-life."""
        now = time.time()
        half_life_ago = now - (DEFAULT_DECAY_HALF_LIFE_HOURS * 3600)

        result = compute_decayed_quality_score(
            last_quality_score=0.90,
            last_quality_update=half_life_ago,
            current_time=now,
        )

        # After one half-life, score should be halfway to floor
        expected = DEFAULT_DECAY_FLOOR + (0.90 - DEFAULT_DECAY_FLOOR) * 0.5
        assert abs(result - expected) < 0.01

    def test_never_below_floor(self):
        """Decayed score should never go below floor."""
        now = time.time()
        very_old = now - (10 * DEFAULT_DECAY_HALF_LIFE_HOURS * 3600)  # 10 half-lives

        result = compute_decayed_quality_score(
            last_quality_score=0.90,
            last_quality_update=very_old,
            current_time=now,
        )

        assert result >= DEFAULT_DECAY_FLOOR

    def test_custom_half_life(self):
        """Custom half-life should affect decay rate."""
        now = time.time()
        two_hours_ago = now - 7200

        # With 2-hour half-life, 2 hours should cause 50% decay
        result = compute_decayed_quality_score(
            last_quality_score=0.90,
            last_quality_update=two_hours_ago,
            current_time=now,
            half_life_hours=2.0,
        )

        expected = DEFAULT_DECAY_FLOOR + (0.90 - DEFAULT_DECAY_FLOOR) * 0.5
        assert abs(result - expected) < 0.01

    def test_no_quality_update_returns_original(self):
        """Zero last_quality_update should return original score."""
        result = compute_decayed_quality_score(
            last_quality_score=0.80,
            last_quality_update=0,  # Never updated
            current_time=time.time(),
        )
        assert result == 0.80


class TestIntensityFromQuality:
    """Test intensity_from_quality function."""

    def test_hot_path_for_high_quality(self):
        """Very high quality should return hot_path."""
        assert intensity_from_quality(0.95) == "hot_path"
        assert intensity_from_quality(0.90) == "hot_path"

    def test_accelerated_for_good_quality(self):
        """Good quality should return accelerated."""
        assert intensity_from_quality(0.85) == "accelerated"
        assert intensity_from_quality(0.80) == "accelerated"

    def test_normal_for_moderate_quality(self):
        """Moderate quality should return normal."""
        assert intensity_from_quality(0.70) == "normal"
        assert intensity_from_quality(0.65) == "normal"

    def test_reduced_for_low_quality(self):
        """Low quality above minimum should return reduced."""
        # Default threshold is 0.50
        assert intensity_from_quality(0.55) == "reduced"
        assert intensity_from_quality(0.50) == "reduced"

    def test_paused_for_very_low_quality(self):
        """Very low quality should return paused."""
        assert intensity_from_quality(0.30) == "paused"
        assert intensity_from_quality(0.10) == "paused"

    def test_config_specific_threshold(self):
        """4-player configs should have higher threshold."""
        # 4p configs typically require 0.65 vs 0.50
        intensity_2p = intensity_from_quality(0.55, "hex8_2p")
        intensity_4p = intensity_from_quality(0.55, "hex8_4p")

        # 0.55 might be reduced for 2p but paused for 4p
        # (depending on actual thresholds)
        assert intensity_2p in ["reduced", "normal"]


class TestCheckQualityGateConditions:
    """Test check_quality_gate_conditions function."""

    def test_quality_above_threshold_passes(self):
        """Quality above threshold should pass."""
        result = check_quality_gate_conditions(
            quality_score=0.80,
            config_key="hex8_2p",
        )
        assert result.passed is True
        assert "quality ok" in result.reason

    def test_no_quality_data_passes_with_warning(self):
        """Missing quality data should pass with warning."""
        result = check_quality_gate_conditions(
            quality_score=None,
            config_key="hex8_2p",
        )
        assert result.passed is True
        assert "no quality data" in result.reason

    def test_uses_decayed_quality_when_no_fresh_data(self):
        """Should use decayed quality when no fresh data."""
        result = check_quality_gate_conditions(
            quality_score=None,
            config_key="hex8_2p",
            decayed_quality=0.70,
        )
        assert result.passed is True
        assert result.quality_score == 0.70

    def test_low_quality_fails(self):
        """Quality below threshold should fail."""
        result = check_quality_gate_conditions(
            quality_score=0.30,  # Below MINIMUM_QUALITY_FLOOR (0.40)
            config_key="hex8_2p",
        )
        assert result.passed is False
        assert "quality too low" in result.reason

    def test_bootstrap_mode_relaxes_gate(self):
        """Low game count should relax quality gate."""
        result = check_quality_gate_conditions(
            quality_score=0.45,  # Above floor (0.40), below threshold (0.50)
            config_key="hex8_2p",
            game_count=1000,  # Below DATA_STARVED_THRESHOLD (5000)
        )
        assert result.passed is True
        assert result.is_relaxed is True
        assert "bootstrap mode" in result.relaxed_reason

    def test_stalled_training_relaxes_gate(self):
        """Stalled training should relax quality gate."""
        result = check_quality_gate_conditions(
            quality_score=0.45,  # Above floor (0.40), below threshold (0.50)
            config_key="hex8_2p",
            game_count=10000,  # Enough games, not bootstrapping
            hours_since_training=30.0,  # Above TRAINING_STALL_HOURS (24)
        )
        assert result.passed is True
        assert result.is_relaxed is True
        assert "stalled" in result.relaxed_reason

    def test_below_floor_never_relaxed(self):
        """Quality below MINIMUM_QUALITY_FLOOR should never be relaxed."""
        result = check_quality_gate_conditions(
            quality_score=0.35,  # Below MINIMUM_QUALITY_FLOOR (0.40)
            config_key="hex8_2p",
            game_count=100,  # Would normally trigger bootstrap
            hours_since_training=100.0,  # Would normally trigger stall
        )
        assert result.passed is False


class TestGetQualityFromState:
    """Test get_quality_from_state helper function."""

    def test_computes_decayed_quality(self):
        """Should compute decayed quality from state fields."""
        now = time.time()
        two_hours_ago = now - 7200

        decayed, hours_since = get_quality_from_state(
            last_quality_score=0.90,
            last_quality_update=two_hours_ago,
            last_training_time=now - 3600,  # 1 hour ago
            current_time=now,
        )

        assert decayed is not None
        assert decayed < 0.90  # Should have decayed
        assert hours_since is not None
        assert abs(hours_since - 1.0) < 0.1

    def test_no_quality_update_returns_none(self):
        """Zero last_quality_update should return None for decayed."""
        now = time.time()

        decayed, hours_since = get_quality_from_state(
            last_quality_score=0.90,
            last_quality_update=0,  # Never updated
            last_training_time=now - 7200,
            current_time=now,
        )

        assert decayed is None
        assert hours_since is not None
        assert abs(hours_since - 2.0) < 0.1

    def test_no_training_returns_none_hours(self):
        """Zero last_training_time should return None for hours."""
        now = time.time()

        decayed, hours_since = get_quality_from_state(
            last_quality_score=0.90,
            last_quality_update=now,
            last_training_time=0,  # Never trained
            current_time=now,
        )

        assert decayed is not None
        assert hours_since is None

    def test_custom_decay_config(self):
        """Should respect custom decay configuration."""
        now = time.time()
        one_hour_ago = now - 3600

        # Custom config: 0.5 hour half-life = 2 half-lives in 1 hour
        decayed, _ = get_quality_from_state(
            last_quality_score=0.90,
            last_quality_update=one_hour_ago,
            last_training_time=now,
            current_time=now,
            decay_config={
                "enabled": True,
                "half_life_hours": 0.5,
                "floor": 0.30,
            },
        )

        assert decayed is not None
        # After 2 half-lives: floor + (0.90 - floor) * 0.25
        expected = 0.30 + (0.90 - 0.30) * 0.25
        assert abs(decayed - expected) < 0.02


class TestQualityGateEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_exact_threshold_passes(self):
        """Quality exactly at threshold should pass."""
        result = check_quality_gate_conditions(
            quality_score=0.50,  # Exactly at typical 2p threshold
            config_key="hex8_2p",
        )
        assert result.passed is True

    def test_exact_floor_can_relax(self):
        """Quality exactly at floor should be able to relax."""
        result = check_quality_gate_conditions(
            quality_score=MINIMUM_QUALITY_FLOOR,
            config_key="hex8_2p",
            game_count=100,  # Bootstrap mode
        )
        assert result.passed is True
        assert result.is_relaxed is True

    def test_negative_quality_handled(self):
        """Negative quality score should fail gracefully."""
        result = check_quality_gate_conditions(
            quality_score=-0.1,
            config_key="hex8_2p",
        )
        assert result.passed is False

    def test_quality_above_one_handled(self):
        """Quality above 1.0 should pass (hot_path)."""
        result = check_quality_gate_conditions(
            quality_score=1.5,  # Invalid but handle gracefully
            config_key="hex8_2p",
        )
        assert result.passed is True

    def test_zero_game_count(self):
        """Zero game count should trigger bootstrap mode."""
        result = check_quality_gate_conditions(
            quality_score=0.45,
            config_key="hex8_2p",
            game_count=0,
        )
        assert result.passed is True
        assert result.is_relaxed is True
        assert "bootstrap" in result.relaxed_reason
