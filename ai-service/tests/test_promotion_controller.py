"""Tests for PromotionController.

Tests promotion evaluation logic, criteria checking,
and decision-making without requiring external services.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from app.training.promotion_controller import (
    PromotionController,
    PromotionCriteria,
    PromotionDecision,
    PromotionType,
    get_promotion_controller,
)


class TestPromotionType:
    """Test PromotionType enum."""

    def test_promotion_types_exist(self):
        """Test all promotion types are defined."""
        assert PromotionType.STAGING.value == "staging"
        assert PromotionType.PRODUCTION.value == "production"
        assert PromotionType.TIER.value == "tier"
        assert PromotionType.CHAMPION.value == "champion"
        assert PromotionType.ROLLBACK.value == "rollback"

    def test_enum_values_are_strings(self):
        """Test enum values are lowercase strings."""
        for ptype in PromotionType:
            assert isinstance(ptype.value, str)
            assert ptype.value == ptype.value.lower()


class TestPromotionCriteria:
    """Test PromotionCriteria dataclass."""

    def test_default_values(self):
        """Test default criteria values."""
        criteria = PromotionCriteria()
        assert criteria.min_elo_improvement == 25.0
        assert criteria.min_games_played == 50
        assert criteria.min_win_rate == 0.52
        assert criteria.max_value_mse_degradation == 0.05
        assert criteria.confidence_threshold == 0.95
        assert criteria.tier_elo_threshold is None
        assert criteria.tier_games_required == 100

    def test_custom_values(self):
        """Test custom criteria values."""
        criteria = PromotionCriteria(
            min_elo_improvement=50.0,
            min_games_played=100,
            min_win_rate=0.55,
        )
        assert criteria.min_elo_improvement == 50.0
        assert criteria.min_games_played == 100
        assert criteria.min_win_rate == 0.55


class TestPromotionDecision:
    """Test PromotionDecision dataclass."""

    def test_basic_decision(self):
        """Test basic decision creation."""
        decision = PromotionDecision(
            model_id="test_model",
            promotion_type=PromotionType.PRODUCTION,
            should_promote=True,
            reason="Meets all criteria",
        )
        assert decision.model_id == "test_model"
        assert decision.promotion_type == PromotionType.PRODUCTION
        assert decision.should_promote is True
        assert decision.reason == "Meets all criteria"

    def test_decision_with_metrics(self):
        """Test decision with evaluation metrics."""
        decision = PromotionDecision(
            model_id="test_model",
            promotion_type=PromotionType.STAGING,
            should_promote=True,
            reason="Good performance",
            current_elo=1550.0,
            elo_improvement=50.0,
            games_played=100,
            win_rate=0.55,
        )
        assert decision.current_elo == 1550.0
        assert decision.elo_improvement == 50.0
        assert decision.games_played == 100
        assert decision.win_rate == 0.55

    def test_decision_for_tier(self):
        """Test decision for tier promotion."""
        decision = PromotionDecision(
            model_id="test_model",
            promotion_type=PromotionType.TIER,
            should_promote=True,
            reason="Tier threshold met",
            current_tier="D1",
            target_tier="D2",
        )
        assert decision.current_tier == "D1"
        assert decision.target_tier == "D2"

    def test_to_dict(self):
        """Test serialization to dict."""
        decision = PromotionDecision(
            model_id="test_model",
            promotion_type=PromotionType.PRODUCTION,
            should_promote=True,
            reason="Test reason",
            current_elo=1500.0,
            games_played=50,
        )
        d = decision.to_dict()

        assert d["model_id"] == "test_model"
        assert d["promotion_type"] == "production"
        assert d["should_promote"] is True
        assert d["reason"] == "Test reason"
        assert d["current_elo"] == 1500.0
        assert d["games_played"] == 50
        assert "evaluated_at" in d

    def test_evaluated_at_timestamp(self):
        """Test evaluated_at is set automatically."""
        decision = PromotionDecision(
            model_id="test",
            promotion_type=PromotionType.STAGING,
            should_promote=False,
            reason="Test",
        )
        # Should be a valid ISO timestamp
        assert decision.evaluated_at is not None
        datetime.fromisoformat(decision.evaluated_at)


class TestPromotionController:
    """Test PromotionController functionality."""

    def test_initialization_default(self):
        """Test default initialization."""
        controller = PromotionController()
        assert controller.criteria is not None
        assert controller.criteria.min_elo_improvement == 25.0

    def test_initialization_custom_criteria(self):
        """Test initialization with custom criteria."""
        criteria = PromotionCriteria(min_elo_improvement=40.0)
        controller = PromotionController(criteria=criteria)
        assert controller.criteria.min_elo_improvement == 40.0

    def test_initialization_with_services(self):
        """Test initialization with injected services."""
        mock_elo = MagicMock()
        mock_registry = MagicMock()

        controller = PromotionController(
            elo_service=mock_elo,
            model_registry=mock_registry,
        )

        assert controller._elo_service is mock_elo
        assert controller._model_registry is mock_registry

    def test_evaluate_insufficient_games(self):
        """Test evaluation fails with insufficient games."""
        mock_elo = MagicMock()
        mock_rating = MagicMock()
        mock_rating.rating = 1550
        mock_rating.games_played = 10  # Below threshold
        mock_rating.win_rate = 0.60
        mock_elo.get_rating.return_value = mock_rating

        controller = PromotionController(elo_service=mock_elo)
        decision = controller.evaluate_promotion(
            model_id="test_model",
            promotion_type=PromotionType.PRODUCTION,
            baseline_model_id="baseline",
        )

        assert decision.should_promote is False
        assert "Insufficient games" in decision.reason

    def test_evaluate_insufficient_elo_improvement(self):
        """Test evaluation fails with insufficient Elo improvement."""
        mock_elo = MagicMock()

        # Current model
        mock_rating = MagicMock()
        mock_rating.rating = 1510
        mock_rating.games_played = 100
        mock_rating.win_rate = 0.55

        # Baseline model
        mock_baseline = MagicMock()
        mock_baseline.rating = 1500  # Only 10 points improvement

        mock_elo.get_rating.side_effect = [mock_rating, mock_baseline]

        controller = PromotionController(elo_service=mock_elo)
        decision = controller.evaluate_promotion(
            model_id="test_model",
            promotion_type=PromotionType.PRODUCTION,
            baseline_model_id="baseline",
        )

        assert decision.should_promote is False
        assert "Insufficient Elo improvement" in decision.reason

    def test_evaluate_low_win_rate(self):
        """Test evaluation fails with low win rate."""
        mock_elo = MagicMock()

        mock_rating = MagicMock()
        mock_rating.rating = 1550
        mock_rating.games_played = 100
        mock_rating.win_rate = 0.45  # Below threshold

        mock_baseline = MagicMock()
        mock_baseline.rating = 1500

        mock_elo.get_rating.side_effect = [mock_rating, mock_baseline]

        controller = PromotionController(elo_service=mock_elo)
        decision = controller.evaluate_promotion(
            model_id="test_model",
            promotion_type=PromotionType.PRODUCTION,
            baseline_model_id="baseline",
        )

        assert decision.should_promote is False
        assert "Win rate too low" in decision.reason

    def test_evaluate_success(self):
        """Test successful promotion evaluation."""
        mock_elo = MagicMock()

        mock_rating = MagicMock()
        mock_rating.rating = 1550
        mock_rating.games_played = 100
        mock_rating.win_rate = 0.58

        mock_baseline = MagicMock()
        mock_baseline.rating = 1500  # 50 points improvement

        mock_elo.get_rating.side_effect = [mock_rating, mock_baseline]

        controller = PromotionController(elo_service=mock_elo)
        decision = controller.evaluate_promotion(
            model_id="test_model",
            promotion_type=PromotionType.PRODUCTION,
            baseline_model_id="baseline",
        )

        assert decision.should_promote is True
        assert "Meets all criteria" in decision.reason
        assert decision.current_elo == 1550
        assert decision.elo_improvement == 50

    def test_evaluate_without_elo_service(self):
        """Test evaluation when Elo service unavailable."""
        # Create controller with None elo service that won't lazy-load
        controller = PromotionController(elo_service=None)

        # Prevent lazy loading by setting a flag
        controller._elo_service = None

        # Patch the property to return None
        with patch.object(PromotionController, 'elo_service', property(lambda self: None)):
            decision = controller.evaluate_promotion(
                model_id="test_model",
                promotion_type=PromotionType.PRODUCTION,
            )

        # Should still return a decision, just without Elo data
        assert decision.model_id == "test_model"
        assert decision.promotion_type == PromotionType.PRODUCTION

    def test_evaluate_rollback_with_regression(self):
        """Test rollback evaluation detects regression."""
        mock_elo = MagicMock()
        mock_elo.get_rating_history.return_value = [
            {"rating": 1450},  # Recent (lower)
            {"rating": 1460},
            {"rating": 1470},
            {"rating": 1500},  # Older (higher)
        ]

        controller = PromotionController(elo_service=mock_elo)
        decision = controller.evaluate_promotion(
            model_id="test_model",
            promotion_type=PromotionType.ROLLBACK,
        )

        assert decision.should_promote is True
        assert "regression" in decision.reason.lower()

    def test_evaluate_rollback_no_regression(self):
        """Test rollback evaluation with stable performance."""
        mock_elo = MagicMock()
        mock_elo.get_rating_history.return_value = [
            {"rating": 1510},  # Recent (higher)
            {"rating": 1505},
            {"rating": 1500},  # Older
        ]

        controller = PromotionController(elo_service=mock_elo)
        decision = controller.evaluate_promotion(
            model_id="test_model",
            promotion_type=PromotionType.ROLLBACK,
        )

        assert decision.should_promote is False
        assert "No significant regression" in decision.reason


class TestPromotionControllerExecution:
    """Test promotion execution methods."""

    def test_execute_skips_non_promotion(self):
        """Test execute skips when should_promote is False."""
        controller = PromotionController()
        decision = PromotionDecision(
            model_id="test",
            promotion_type=PromotionType.PRODUCTION,
            should_promote=False,
            reason="Not ready",
        )

        result = controller.execute_promotion(decision)
        assert result is False

    def test_execute_dry_run(self):
        """Test dry run doesn't execute promotion."""
        controller = PromotionController()
        decision = PromotionDecision(
            model_id="test",
            promotion_type=PromotionType.PRODUCTION,
            should_promote=True,
            reason="Ready",
        )

        result = controller.execute_promotion(decision, dry_run=True)
        assert result is True

    def test_execute_stage_promotion(self):
        """Test stage promotion execution."""
        mock_registry = MagicMock()
        controller = PromotionController(model_registry=mock_registry)

        decision = PromotionDecision(
            model_id="test_model",
            promotion_type=PromotionType.STAGING,
            should_promote=True,
            reason="Ready for staging",
        )

        result = controller.execute_promotion(decision)

        assert result is True
        mock_registry.promote_model.assert_called_once()

    def test_execute_rollback(self):
        """Test rollback execution."""
        mock_registry = MagicMock()
        mock_registry.get_production_model.return_value = "current_prod"

        controller = PromotionController(model_registry=mock_registry)

        decision = PromotionDecision(
            model_id="previous_model",
            promotion_type=PromotionType.ROLLBACK,
            should_promote=True,
            reason="Performance regression",
        )

        result = controller.execute_promotion(decision)

        assert result is True
        # Should archive current and promote rollback target
        assert mock_registry.promote_model.call_count == 2


class TestGetPromotionController:
    """Test convenience function."""

    def test_get_controller_default(self):
        """Test getting controller with defaults."""
        controller = get_promotion_controller()
        assert isinstance(controller, PromotionController)
        assert controller.criteria.min_elo_improvement == 25.0

    def test_get_controller_custom_criteria(self):
        """Test getting controller with custom criteria."""
        criteria = PromotionCriteria(min_games_played=200)
        controller = get_promotion_controller(criteria=criteria)
        assert controller.criteria.min_games_played == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
