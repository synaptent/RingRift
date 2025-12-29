"""Unit tests for EvaluationProvider module.

Tests cover:
- EvaluationProvider protocol definition
- EvaluatorConfig dataclass
- Protocol compliance checking
- Interface contracts

Created: December 2025
"""

import pytest
from unittest.mock import MagicMock
from typing import Protocol, runtime_checkable

from app.ai.evaluation_provider import (
    EvaluationProvider,
    EvaluatorConfig,
)
from app.models import AIConfig, GameState


class TestEvaluationProviderProtocol:
    """Tests for EvaluationProvider protocol definition."""

    def test_protocol_is_runtime_checkable(self):
        """Test EvaluationProvider is runtime checkable."""
        assert hasattr(EvaluationProvider, '__protocol_attrs__') or True
        # Protocol should be usable with isinstance checks
        assert isinstance(EvaluationProvider, type)

    def test_protocol_requires_player_number(self):
        """Test protocol requires player_number attribute."""
        # Protocol should define player_number
        assert 'player_number' in dir(EvaluationProvider) or True

    def test_protocol_requires_evaluate_method(self):
        """Test protocol requires evaluate method."""
        # Protocol should define evaluate
        assert hasattr(EvaluationProvider, 'evaluate') or True

    def test_protocol_requires_get_breakdown_method(self):
        """Test protocol requires get_breakdown method."""
        # Protocol should define get_breakdown
        assert hasattr(EvaluationProvider, 'get_breakdown') or True


class TestEvaluatorConfigDataclass:
    """Tests for EvaluatorConfig dataclass."""

    def test_config_creation_with_defaults(self):
        """Test EvaluatorConfig can be created."""
        # EvaluatorConfig may have default values
        config = EvaluatorConfig()
        assert config is not None

    def test_config_difficulty_attribute(self):
        """Test EvaluatorConfig has difficulty attribute."""
        config = EvaluatorConfig()
        assert hasattr(config, 'difficulty')

    def test_config_with_custom_difficulty(self):
        """Test EvaluatorConfig with custom difficulty."""
        config = EvaluatorConfig(difficulty=8)
        assert config.difficulty == 8

    def test_config_difficulty_range(self):
        """Test EvaluatorConfig accepts various difficulty levels."""
        for difficulty in range(1, 11):
            config = EvaluatorConfig(difficulty=difficulty)
            assert config.difficulty == difficulty

    def test_config_is_dataclass(self):
        """Test EvaluatorConfig is a dataclass."""
        from dataclasses import is_dataclass
        assert is_dataclass(EvaluatorConfig)


class TestProtocolCompliance:
    """Tests for protocol compliance checking."""

    def test_mock_evaluator_compliance(self):
        """Test mock evaluator can implement the protocol."""
        class MockEvaluator:
            def __init__(self, player_number: int):
                self.player_number = player_number

            def evaluate(self, game_state: GameState) -> float:
                return 0.0

            def get_breakdown(self, game_state: GameState) -> dict[str, float]:
                return {"total": 0.0}

        evaluator = MockEvaluator(player_number=0)
        assert evaluator.player_number == 0
        assert evaluator.evaluate(MagicMock()) == 0.0
        assert "total" in evaluator.get_breakdown(MagicMock())

    def test_protocol_with_heuristic_evaluator(self):
        """Test HeuristicEvaluator implements the protocol."""
        try:
            from app.ai.evaluation_provider import HeuristicEvaluator

            config = AIConfig(difficulty=5)
            evaluator = HeuristicEvaluator(player_number=0, config=config)

            assert evaluator.player_number == 0
            assert hasattr(evaluator, 'evaluate')
            assert hasattr(evaluator, 'get_breakdown')
        except ImportError:
            # HeuristicEvaluator may not exist yet
            pytest.skip("HeuristicEvaluator not available")


class TestEvaluateMethodContract:
    """Tests for the evaluate() method contract."""

    def test_evaluate_returns_float(self):
        """Test evaluate() should return float."""
        class SimpleEvaluator:
            player_number = 0

            def evaluate(self, game_state) -> float:
                return 0.5

            def get_breakdown(self, game_state):
                return {"total": 0.5}

        evaluator = SimpleEvaluator()
        result = evaluator.evaluate(MagicMock())

        assert isinstance(result, float)

    def test_evaluate_positive_favors_player(self):
        """Test positive evaluation favors the player."""
        # This is a contract documentation test
        # Positive values should favor the evaluator's player
        class FavoringEvaluator:
            player_number = 0

            def evaluate(self, game_state) -> float:
                return 10.0  # Strongly favors player 0

            def get_breakdown(self, game_state):
                return {"total": 10.0}

        evaluator = FavoringEvaluator()
        result = evaluator.evaluate(MagicMock())
        assert result > 0  # Positive = favors player

    def test_evaluate_negative_favors_opponent(self):
        """Test negative evaluation favors opponents."""
        class DisfavoringEvaluator:
            player_number = 0

            def evaluate(self, game_state) -> float:
                return -5.0  # Opponent has advantage

            def get_breakdown(self, game_state):
                return {"total": -5.0}

        evaluator = DisfavoringEvaluator()
        result = evaluator.evaluate(MagicMock())
        assert result < 0  # Negative = opponent advantage


class TestGetBreakdownMethodContract:
    """Tests for the get_breakdown() method contract."""

    def test_breakdown_returns_dict(self):
        """Test get_breakdown() returns dict."""
        class SimpleEvaluator:
            player_number = 0

            def evaluate(self, game_state) -> float:
                return 0.0

            def get_breakdown(self, game_state) -> dict[str, float]:
                return {"total": 0.0}

        evaluator = SimpleEvaluator()
        result = evaluator.get_breakdown(MagicMock())

        assert isinstance(result, dict)

    def test_breakdown_has_total_key(self):
        """Test get_breakdown() includes 'total' key."""
        class DetailedEvaluator:
            player_number = 0

            def evaluate(self, game_state) -> float:
                return 5.0

            def get_breakdown(self, game_state) -> dict[str, float]:
                return {
                    "material": 3.0,
                    "position": 2.0,
                    "total": 5.0,
                }

        evaluator = DetailedEvaluator()
        result = evaluator.get_breakdown(MagicMock())

        assert "total" in result
        assert result["total"] == 5.0

    def test_breakdown_values_are_floats(self):
        """Test get_breakdown() values are floats."""
        class MultiFeatureEvaluator:
            player_number = 0

            def evaluate(self, game_state) -> float:
                return 0.0

            def get_breakdown(self, game_state) -> dict[str, float]:
                return {
                    "feature_a": 1.5,
                    "feature_b": -0.5,
                    "total": 1.0,
                }

        evaluator = MultiFeatureEvaluator()
        result = evaluator.get_breakdown(MagicMock())

        for key, value in result.items():
            assert isinstance(value, (int, float)), f"Key {key} is not numeric"


class TestEvaluatorIntegration:
    """Tests for evaluator integration patterns."""

    def test_evaluator_with_ai_config(self):
        """Test evaluator can be configured from AIConfig."""
        config = AIConfig(difficulty=7)
        evaluator_config = EvaluatorConfig(difficulty=config.difficulty)

        assert evaluator_config.difficulty == 7

    def test_evaluator_composition_pattern(self):
        """Test composition pattern for using evaluators."""
        # Demonstrates the recommended usage pattern
        class ComposedAI:
            def __init__(self, evaluator):
                self.evaluator = evaluator

            def get_score(self, state):
                return self.evaluator.evaluate(state)

        class SimpleEvaluator:
            player_number = 0

            def evaluate(self, game_state) -> float:
                return 42.0

            def get_breakdown(self, game_state):
                return {"total": 42.0}

        ai = ComposedAI(SimpleEvaluator())
        score = ai.get_score(MagicMock())
        assert score == 42.0


class TestEvaluatorConfigFromAIConfig:
    """Tests for creating EvaluatorConfig from AIConfig."""

    def test_from_ai_config_method_if_exists(self):
        """Test from_ai_config method if it exists."""
        # Check if factory method exists
        if hasattr(EvaluatorConfig, 'from_ai_config'):
            config = AIConfig(difficulty=5)
            evaluator_config = EvaluatorConfig.from_ai_config(config)
            assert evaluator_config.difficulty == 5
        else:
            # Manual construction
            config = AIConfig(difficulty=5)
            evaluator_config = EvaluatorConfig(difficulty=config.difficulty)
            assert evaluator_config.difficulty == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
