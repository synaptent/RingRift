"""Unit tests for stability_heuristic module (December 2025).

Tests cover:
- StabilityLevel: Enum values for model stability classification
- StabilityAssessment: Dataclass for stability metrics and recommendations
- assess_model_stability: Main assessment function
- _calculate_volatility_score: Volatility calculation helper
- _classify_stability: Stability classification helper
- _add_recommendations: Recommendation generation helper
- is_promotion_safe: Convenience function for promotion gates
- get_stability_summary: Convenience function for config-based queries
"""

import math
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# StabilityLevel Tests
# =============================================================================


class TestStabilityLevel:
    """Tests for StabilityLevel enum."""

    def test_all_levels_defined(self):
        """Test all expected stability levels are defined."""
        from app.coordination.stability_heuristic import StabilityLevel

        expected = ["STABLE", "DEVELOPING", "VOLATILE", "DECLINING", "UNKNOWN"]
        for name in expected:
            assert hasattr(StabilityLevel, name)

    def test_level_values(self):
        """Test stability level values are strings."""
        from app.coordination.stability_heuristic import StabilityLevel

        assert StabilityLevel.STABLE.value == "stable"
        assert StabilityLevel.DEVELOPING.value == "developing"
        assert StabilityLevel.VOLATILE.value == "volatile"
        assert StabilityLevel.DECLINING.value == "declining"
        assert StabilityLevel.UNKNOWN.value == "unknown"

    def test_level_comparison(self):
        """Test stability levels are distinct."""
        from app.coordination.stability_heuristic import StabilityLevel

        assert StabilityLevel.STABLE != StabilityLevel.VOLATILE
        assert StabilityLevel.DEVELOPING != StabilityLevel.DECLINING


# =============================================================================
# StabilityAssessment Tests
# =============================================================================


class TestStabilityAssessment:
    """Tests for StabilityAssessment dataclass."""

    def test_basic_creation(self):
        """Test basic StabilityAssessment creation."""
        from app.coordination.stability_heuristic import (
            StabilityAssessment, StabilityLevel
        )

        assessment = StabilityAssessment(
            model_id="canonical",
            board_type="hex8",
            num_players=2,
        )

        assert assessment.model_id == "canonical"
        assert assessment.board_type == "hex8"
        assert assessment.num_players == 2
        assert assessment.level == StabilityLevel.UNKNOWN  # Default
        assert assessment.volatility_score == 0.0
        assert assessment.promotion_safe is True  # Default
        assert assessment.investigation_needed is False  # Default
        assert assessment.recommended_actions == []  # Default empty list

    def test_full_creation(self):
        """Test StabilityAssessment with all fields."""
        from app.coordination.stability_heuristic import (
            StabilityAssessment, StabilityLevel
        )

        assessment = StabilityAssessment(
            model_id="canonical",
            board_type="square8",
            num_players=4,
            level=StabilityLevel.STABLE,
            volatility_score=0.15,
            rating_variance=400.0,
            rating_std_dev=20.0,
            max_swing=35.0,
            oscillation_count=3,
            is_plateau=True,
            is_declining=False,
            slope=-0.5,
            trend_confidence=0.85,
            sample_count=25,
            duration_hours=48.0,
            promotion_safe=True,
            investigation_needed=False,
            recommended_actions=["Continue monitoring"],
        )

        assert assessment.level == StabilityLevel.STABLE
        assert assessment.volatility_score == 0.15
        assert assessment.rating_variance == 400.0
        assert assessment.rating_std_dev == 20.0
        assert assessment.max_swing == 35.0
        assert assessment.oscillation_count == 3
        assert assessment.is_plateau is True
        assert assessment.is_declining is False
        assert assessment.slope == -0.5
        assert assessment.trend_confidence == 0.85
        assert assessment.sample_count == 25
        assert assessment.duration_hours == 48.0
        assert "Continue monitoring" in assessment.recommended_actions

    def test_to_dict(self):
        """Test to_dict returns expected structure."""
        from app.coordination.stability_heuristic import (
            StabilityAssessment, StabilityLevel
        )

        assessment = StabilityAssessment(
            model_id="canonical",
            board_type="hex8",
            num_players=2,
            level=StabilityLevel.DEVELOPING,
            volatility_score=0.456789,
            rating_std_dev=35.678,
            slope=1.234567,
        )

        result = assessment.to_dict()

        assert isinstance(result, dict)
        assert result["model_id"] == "canonical"
        assert result["board_type"] == "hex8"
        assert result["num_players"] == 2
        assert result["level"] == "developing"  # String value
        assert result["volatility_score"] == 0.457  # Rounded to 3 decimals
        assert result["rating_std_dev"] == 35.68  # Rounded to 2 decimals
        assert result["slope"] == 1.235  # Rounded to 3 decimals

    def test_to_dict_all_fields(self):
        """Test to_dict includes all expected fields."""
        from app.coordination.stability_heuristic import StabilityAssessment

        assessment = StabilityAssessment(
            model_id="test",
            board_type="hex8",
            num_players=2,
        )

        result = assessment.to_dict()

        expected_keys = [
            "model_id", "board_type", "num_players", "level",
            "volatility_score", "rating_variance", "rating_std_dev",
            "max_swing", "oscillation_count", "is_plateau", "is_declining",
            "slope", "trend_confidence", "sample_count", "duration_hours",
            "promotion_safe", "investigation_needed", "recommended_actions",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"


# =============================================================================
# Volatility Score Calculation Tests
# =============================================================================


class TestCalculateVolatilityScore:
    """Tests for _calculate_volatility_score helper."""

    def test_zero_volatility_for_stable_model(self):
        """Test stable model gets low volatility score."""
        from app.coordination.stability_heuristic import (
            StabilityAssessment, _calculate_volatility_score
        )

        # Very stable model metrics
        assessment = StabilityAssessment(
            model_id="stable",
            board_type="hex8",
            num_players=2,
            rating_std_dev=10.0,  # Low std dev
            max_swing=15.0,      # Small swings
            oscillation_count=1,
            sample_count=20,
            is_declining=False,
        )

        score = _calculate_volatility_score(assessment)

        assert score < 0.3  # Should be stable

    def test_high_volatility_for_unstable_model(self):
        """Test unstable model gets high volatility score."""
        from app.coordination.stability_heuristic import (
            StabilityAssessment, _calculate_volatility_score
        )

        # Unstable model metrics
        assessment = StabilityAssessment(
            model_id="unstable",
            board_type="hex8",
            num_players=2,
            rating_std_dev=100.0,  # High std dev
            max_swing=200.0,       # Large swings
            oscillation_count=15,
            sample_count=20,
            is_declining=True,
            slope=-5.0,
        )

        score = _calculate_volatility_score(assessment)

        assert score >= 0.6  # Should be volatile

    def test_std_dev_component_weight(self):
        """Test standard deviation contributes to volatility."""
        from app.coordination.stability_heuristic import (
            StabilityAssessment, _calculate_volatility_score
        )

        # Only std dev, no other factors
        assessment = StabilityAssessment(
            model_id="test",
            board_type="hex8",
            num_players=2,
            rating_std_dev=50.0,  # Normalized to 1.0
            sample_count=5,
        )

        score = _calculate_volatility_score(assessment)

        # Should have contribution from std dev (~0.4 * 50/50 = 0.4)
        assert 0.35 <= score <= 0.5

    def test_oscillation_component(self):
        """Test oscillation count contributes to volatility."""
        from app.coordination.stability_heuristic import (
            StabilityAssessment, _calculate_volatility_score
        )

        # High oscillation count
        assessment = StabilityAssessment(
            model_id="oscillating",
            board_type="hex8",
            num_players=2,
            rating_std_dev=20.0,
            oscillation_count=18,  # Near max (sample_count - 2)
            sample_count=20,
        )

        score = _calculate_volatility_score(assessment)

        # Oscillation adds ~0.2 weight
        assert score > 0.3


# =============================================================================
# Stability Classification Tests
# =============================================================================


class TestClassifyStability:
    """Tests for _classify_stability helper."""

    def test_classify_stable(self):
        """Test classification as STABLE."""
        from app.coordination.stability_heuristic import (
            StabilityAssessment, StabilityLevel, _classify_stability
        )

        assessment = StabilityAssessment(
            model_id="test",
            board_type="hex8",
            num_players=2,
            volatility_score=0.2,  # Below 0.3 threshold
            is_declining=False,
        )

        level = _classify_stability(assessment)

        assert level == StabilityLevel.STABLE

    def test_classify_developing(self):
        """Test classification as DEVELOPING."""
        from app.coordination.stability_heuristic import (
            StabilityAssessment, StabilityLevel, _classify_stability
        )

        assessment = StabilityAssessment(
            model_id="test",
            board_type="hex8",
            num_players=2,
            volatility_score=0.45,  # Between 0.3 and 0.6
            is_declining=False,
        )

        level = _classify_stability(assessment)

        assert level == StabilityLevel.DEVELOPING

    def test_classify_volatile(self):
        """Test classification as VOLATILE."""
        from app.coordination.stability_heuristic import (
            StabilityAssessment, StabilityLevel, _classify_stability
        )

        assessment = StabilityAssessment(
            model_id="test",
            board_type="hex8",
            num_players=2,
            volatility_score=0.8,  # Above 0.6 threshold
            is_declining=False,
        )

        level = _classify_stability(assessment)

        assert level == StabilityLevel.VOLATILE

    def test_classify_declining(self):
        """Test classification as DECLINING overrides volatility score."""
        from app.coordination.stability_heuristic import (
            StabilityAssessment, StabilityLevel, _classify_stability
        )

        assessment = StabilityAssessment(
            model_id="test",
            board_type="hex8",
            num_players=2,
            volatility_score=0.1,  # Would be stable
            is_declining=True,
            slope=-3.0,  # Steep decline (< -2.0)
        )

        level = _classify_stability(assessment)

        assert level == StabilityLevel.DECLINING


# =============================================================================
# Recommendation Tests
# =============================================================================


class TestAddRecommendations:
    """Tests for _add_recommendations helper."""

    def test_stable_no_recommendations(self):
        """Test stable models get no recommendations."""
        from app.coordination.stability_heuristic import (
            StabilityAssessment, StabilityLevel, _add_recommendations
        )

        assessment = StabilityAssessment(
            model_id="test",
            board_type="hex8",
            num_players=2,
            level=StabilityLevel.STABLE,
        )

        _add_recommendations(assessment)

        assert assessment.promotion_safe is True
        assert assessment.investigation_needed is False
        assert assessment.recommended_actions == []

    def test_developing_gets_guidance(self):
        """Test developing models get monitoring guidance."""
        from app.coordination.stability_heuristic import (
            StabilityAssessment, StabilityLevel, _add_recommendations
        )

        assessment = StabilityAssessment(
            model_id="test",
            board_type="hex8",
            num_players=2,
            level=StabilityLevel.DEVELOPING,
            sample_count=10,  # Below 20
        )

        _add_recommendations(assessment)

        assert assessment.promotion_safe is True
        assert assessment.investigation_needed is False
        assert len(assessment.recommended_actions) >= 1
        assert "Continue evaluation" in assessment.recommended_actions[0]

    def test_volatile_blocks_promotion(self):
        """Test volatile models are blocked from promotion."""
        from app.coordination.stability_heuristic import (
            StabilityAssessment, StabilityLevel, _add_recommendations
        )

        assessment = StabilityAssessment(
            model_id="test",
            board_type="hex8",
            num_players=2,
            level=StabilityLevel.VOLATILE,
            volatility_score=0.8,
            max_swing=100.0,  # Large swing
            oscillation_count=15,
            sample_count=20,
        )

        _add_recommendations(assessment)

        assert assessment.promotion_safe is False
        assert assessment.investigation_needed is True
        assert len(assessment.recommended_actions) >= 1

    def test_declining_blocks_promotion(self):
        """Test declining models are blocked from promotion."""
        from app.coordination.stability_heuristic import (
            StabilityAssessment, StabilityLevel, _add_recommendations
        )

        assessment = StabilityAssessment(
            model_id="test",
            board_type="hex8",
            num_players=2,
            level=StabilityLevel.DECLINING,
            slope=-3.0,
            is_plateau=True,
        )

        _add_recommendations(assessment)

        assert assessment.promotion_safe is False
        assert assessment.investigation_needed is True
        assert any("Declining trend" in a for a in assessment.recommended_actions)


# =============================================================================
# Main Assessment Function Tests
# =============================================================================


class TestAssessModelStability:
    """Tests for assess_model_stability function."""

    def test_returns_assessment(self):
        """Test function returns StabilityAssessment."""
        from app.coordination.stability_heuristic import (
            assess_model_stability, StabilityAssessment
        )

        # Mock at source since import is local to function
        with patch.dict("sys.modules", {"app.training.elo_service": None}):
            result = assess_model_stability("canonical", "hex8", 2)

            assert isinstance(result, StabilityAssessment)
            assert result.model_id == "canonical"
            assert result.board_type == "hex8"
            assert result.num_players == 2

    def test_handles_missing_elo_service(self):
        """Test graceful handling when elo_service not available."""
        from app.coordination.stability_heuristic import (
            assess_model_stability, StabilityLevel
        )

        # Mock at source since import is local to function
        with patch.dict("sys.modules", {"app.training.elo_service": None}):
            result = assess_model_stability("canonical", "hex8", 2)

            # Returns UNKNOWN when can't import elo_service
            assert result.level == StabilityLevel.UNKNOWN

    def test_with_provided_elo_service(self):
        """Test using provided elo_service."""
        from app.coordination.stability_heuristic import (
            assess_model_stability, StabilityLevel
        )

        mock_elo_service = MagicMock()
        mock_elo_module = MagicMock()

        # Mock the functions that get imported inside the try block
        mock_elo_module.get_elo_trend = MagicMock(return_value={
            "is_plateau": False,
            "is_declining": False,
            "slope": 0.5,
            "confidence": 0.9,
            "sample_count": 30,
            "duration_hours": 48.0,
        })
        mock_elo_module.get_rating_history = MagicMock(return_value=[
            {"rating": 1200 + i * 2} for i in range(20)
        ])

        with patch.dict("sys.modules", {"app.training.elo_service": mock_elo_module}):
            result = assess_model_stability(
                "canonical", "hex8", 2, elo_service=mock_elo_service
            )

            assert result.sample_count == 30
            assert result.level != StabilityLevel.UNKNOWN

    def test_insufficient_samples(self):
        """Test handling of insufficient data."""
        from app.coordination.stability_heuristic import (
            assess_model_stability, StabilityLevel
        )

        mock_elo_service = MagicMock()
        mock_elo_module = MagicMock()

        # Mock with insufficient samples
        mock_elo_module.get_elo_trend = MagicMock(return_value={
            "sample_count": 3,
            "duration_hours": 2.0,
        })
        mock_elo_module.get_rating_history = MagicMock(return_value=[
            {"rating": 1200},
            {"rating": 1210},
            {"rating": 1205},
        ])

        with patch.dict("sys.modules", {"app.training.elo_service": mock_elo_module}):
            result = assess_model_stability(
                "canonical", "hex8", 2,
                min_samples=5,
                elo_service=mock_elo_service
            )

            assert result.level == StabilityLevel.UNKNOWN
            assert any("Need" in a for a in result.recommended_actions)


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestIsPromotionSafe:
    """Tests for is_promotion_safe convenience function."""

    def test_returns_tuple(self):
        """Test function returns tuple of (bool, StabilityAssessment)."""
        from app.coordination.stability_heuristic import is_promotion_safe

        with patch("app.coordination.stability_heuristic.assess_model_stability") as mock_assess:
            from app.coordination.stability_heuristic import (
                StabilityAssessment, StabilityLevel
            )
            mock_assess.return_value = StabilityAssessment(
                model_id="canonical",
                board_type="hex8",
                num_players=2,
                level=StabilityLevel.STABLE,
                promotion_safe=True,
            )

            is_safe, assessment = is_promotion_safe("canonical", "hex8", 2)

            assert isinstance(is_safe, bool)
            assert is_safe is True
            assert assessment.level == StabilityLevel.STABLE

    def test_returns_false_for_volatile(self):
        """Test returns False for volatile models."""
        from app.coordination.stability_heuristic import is_promotion_safe

        with patch("app.coordination.stability_heuristic.assess_model_stability") as mock_assess:
            from app.coordination.stability_heuristic import (
                StabilityAssessment, StabilityLevel
            )
            mock_assess.return_value = StabilityAssessment(
                model_id="canonical",
                board_type="hex8",
                num_players=2,
                level=StabilityLevel.VOLATILE,
                promotion_safe=False,
            )

            is_safe, assessment = is_promotion_safe("canonical", "hex8", 2)

            assert is_safe is False


class TestGetStabilitySummary:
    """Tests for get_stability_summary convenience function."""

    def test_parses_config_key(self):
        """Test config key parsing."""
        from app.coordination.stability_heuristic import get_stability_summary

        with patch("app.coordination.stability_heuristic.assess_model_stability") as mock_assess:
            from app.coordination.stability_heuristic import (
                StabilityAssessment, StabilityLevel
            )
            mock_assess.return_value = StabilityAssessment(
                model_id="canonical",
                board_type="square8",
                num_players=2,
                level=StabilityLevel.STABLE,
            )

            result = get_stability_summary("square8_2p")

            mock_assess.assert_called_once_with("canonical", "square8", 2)
            assert "level" in result

    def test_invalid_config_key_format(self):
        """Test handling of invalid config key."""
        from app.coordination.stability_heuristic import get_stability_summary

        result = get_stability_summary("invalid")

        assert "error" in result
        assert "Invalid config_key format" in result["error"]

    def test_invalid_player_count(self):
        """Test handling of non-numeric player count."""
        from app.coordination.stability_heuristic import get_stability_summary

        result = get_stability_summary("square8_xp")

        assert "error" in result
        assert "Invalid config_key format" in result["error"]

    def test_custom_participant_id(self):
        """Test custom participant_id."""
        from app.coordination.stability_heuristic import get_stability_summary

        with patch("app.coordination.stability_heuristic.assess_model_stability") as mock_assess:
            from app.coordination.stability_heuristic import (
                StabilityAssessment, StabilityLevel
            )
            mock_assess.return_value = StabilityAssessment(
                model_id="challenger",
                board_type="hex8",
                num_players=4,
                level=StabilityLevel.DEVELOPING,
            )

            result = get_stability_summary("hex8_4p", participant_id="challenger")

            mock_assess.assert_called_once_with("challenger", "hex8", 4)
            assert result["model_id"] == "challenger"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_sample_count(self):
        """Test handling of zero samples."""
        from app.coordination.stability_heuristic import (
            StabilityAssessment, _calculate_volatility_score
        )

        assessment = StabilityAssessment(
            model_id="empty",
            board_type="hex8",
            num_players=2,
            sample_count=0,
        )

        # Should not raise
        score = _calculate_volatility_score(assessment)
        assert score >= 0.0

    def test_negative_slope(self):
        """Test handling of steep negative slope."""
        from app.coordination.stability_heuristic import (
            StabilityAssessment, _calculate_volatility_score
        )

        assessment = StabilityAssessment(
            model_id="declining",
            board_type="hex8",
            num_players=2,
            is_declining=True,
            slope=-10.0,  # Very steep decline
            sample_count=20,
        )

        score = _calculate_volatility_score(assessment)

        # Should have decline penalty contribution (capped at 0.5 * 0.1 = 0.05)
        assert score > 0.0

    def test_zero_std_dev(self):
        """Test handling of zero standard deviation."""
        from app.coordination.stability_heuristic import (
            StabilityAssessment, _calculate_volatility_score
        )

        assessment = StabilityAssessment(
            model_id="constant",
            board_type="hex8",
            num_players=2,
            rating_std_dev=0.0,  # Perfectly constant ratings
            max_swing=0.0,
            sample_count=10,
        )

        score = _calculate_volatility_score(assessment)

        assert score == 0.0  # Perfectly stable

    def test_very_large_values(self):
        """Test handling of very large metric values."""
        from app.coordination.stability_heuristic import (
            StabilityAssessment, _calculate_volatility_score, _classify_stability,
            StabilityLevel
        )

        assessment = StabilityAssessment(
            model_id="extreme",
            board_type="hex8",
            num_players=2,
            rating_std_dev=1000.0,  # Very high
            max_swing=2000.0,       # Extreme swing
            oscillation_count=50,
            sample_count=52,
            is_declining=False,
        )

        score = _calculate_volatility_score(assessment)

        # Very high std_dev (1000/50 = 20) * 0.4 = 8.0 base contribution
        # Plus swing and oscillation contributions
        assert score > 0.6  # Should exceed volatile threshold

        # Now set volatility_score on assessment for classification
        assessment.volatility_score = score
        level = _classify_stability(assessment)

        assert level == StabilityLevel.VOLATILE
