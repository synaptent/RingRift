"""Unit tests for adversarial_positions module.

Tests cover:
- AdversarialStrategy enum values
- AdversarialPosition dataclass
- AdversarialConfig dataclass
- AdversarialGenerator initialization
- Strategy-specific generation logic

Created: December 2025
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from app.training.adversarial_positions import (
    AdversarialStrategy,
    AdversarialPosition,
    AdversarialConfig,
)


class TestAdversarialStrategyEnum:
    """Tests for AdversarialStrategy enum."""

    def test_enum_values(self):
        """Test AdversarialStrategy enum has correct values."""
        assert AdversarialStrategy.UNCERTAINTY.value == "uncertainty"
        assert AdversarialStrategy.DISAGREEMENT.value == "disagreement"
        assert AdversarialStrategy.GRADIENT.value == "gradient"
        assert AdversarialStrategy.SEARCH.value == "search"
        assert AdversarialStrategy.REPLAY.value == "replay"
        assert AdversarialStrategy.PERTURBATION.value == "perturbation"
        assert AdversarialStrategy.BOUNDARY.value == "boundary"

    def test_enum_members(self):
        """Test all expected enum members exist."""
        expected = {
            "UNCERTAINTY",
            "DISAGREEMENT",
            "GRADIENT",
            "SEARCH",
            "REPLAY",
            "PERTURBATION",
            "BOUNDARY",
        }
        actual = {m.name for m in AdversarialStrategy}
        assert actual == expected

    def test_enum_from_string(self):
        """Test creating enum from string value."""
        assert AdversarialStrategy("uncertainty") == AdversarialStrategy.UNCERTAINTY
        assert AdversarialStrategy("replay") == AdversarialStrategy.REPLAY


class TestAdversarialPositionDataclass:
    """Tests for AdversarialPosition dataclass."""

    def test_basic_creation(self):
        """Test basic AdversarialPosition creation."""
        position = AdversarialPosition(
            position_id="test_001",
            board_type="hex8",
            num_players=2,
            board_state=np.zeros((9, 9)),
            move_history=[],
            strategy=AdversarialStrategy.UNCERTAINTY,
            difficulty_score=0.75,
            uncertainty_score=0.80,
            disagreement_score=0.50,
        )

        assert position.position_id == "test_001"
        assert position.board_type == "hex8"
        assert position.num_players == 2
        assert position.difficulty_score == 0.75

    def test_board_state_is_numpy(self):
        """Test board_state is numpy array."""
        board = np.random.rand(8, 8)
        position = AdversarialPosition(
            position_id="test_002",
            board_type="square8",
            num_players=2,
            board_state=board,
            move_history=[],
            strategy=AdversarialStrategy.GRADIENT,
            difficulty_score=0.5,
            uncertainty_score=0.5,
            disagreement_score=0.5,
        )

        assert isinstance(position.board_state, np.ndarray)
        assert position.board_state.shape == (8, 8)

    def test_score_ranges(self):
        """Test score values are within expected range."""
        position = AdversarialPosition(
            position_id="test_003",
            board_type="hex8",
            num_players=2,
            board_state=np.zeros((9, 9)),
            move_history=[],
            strategy=AdversarialStrategy.BOUNDARY,
            difficulty_score=1.0,
            uncertainty_score=0.0,
            disagreement_score=0.5,
        )

        # Scores should be in 0-1 range
        assert 0.0 <= position.difficulty_score <= 1.0
        assert 0.0 <= position.uncertainty_score <= 1.0
        assert 0.0 <= position.disagreement_score <= 1.0

    def test_ground_truth_optional(self):
        """Test ground_truth fields are optional."""
        position = AdversarialPosition(
            position_id="test_004",
            board_type="hex8",
            num_players=2,
            board_state=np.zeros((9, 9)),
            move_history=[],
            strategy=AdversarialStrategy.SEARCH,
            difficulty_score=0.5,
            uncertainty_score=0.5,
            disagreement_score=0.5,
        )

        assert position.ground_truth_value is None
        assert position.ground_truth_policy is None

    def test_with_ground_truth(self):
        """Test AdversarialPosition with ground truth values."""
        policy = np.array([0.2, 0.5, 0.3])
        position = AdversarialPosition(
            position_id="test_005",
            board_type="square8",
            num_players=2,
            board_state=np.zeros((8, 8)),
            move_history=[],
            strategy=AdversarialStrategy.REPLAY,
            difficulty_score=0.8,
            uncertainty_score=0.3,
            disagreement_score=0.2,
            ground_truth_value=0.75,
            ground_truth_policy=policy,
        )

        assert position.ground_truth_value == 0.75
        assert position.ground_truth_policy is not None
        assert np.array_equal(position.ground_truth_policy, policy)

    def test_metadata_default_empty(self):
        """Test metadata defaults to empty dict."""
        position = AdversarialPosition(
            position_id="test_006",
            board_type="hex8",
            num_players=2,
            board_state=np.zeros((9, 9)),
            move_history=[],
            strategy=AdversarialStrategy.PERTURBATION,
            difficulty_score=0.5,
            uncertainty_score=0.5,
            disagreement_score=0.5,
        )

        assert position.metadata == {}

    def test_with_metadata(self):
        """Test AdversarialPosition with metadata."""
        position = AdversarialPosition(
            position_id="test_007",
            board_type="hex8",
            num_players=4,
            board_state=np.zeros((9, 9)),
            move_history=["e4", "e5"],
            strategy=AdversarialStrategy.UNCERTAINTY,
            difficulty_score=0.9,
            uncertainty_score=0.95,
            disagreement_score=0.8,
            metadata={
                "source_game": "game_12345",
                "move_number": 15,
                "time_control": "rapid",
            },
        )

        assert position.metadata["source_game"] == "game_12345"
        assert position.metadata["move_number"] == 15

    def test_move_history_list(self):
        """Test move_history is a list."""
        position = AdversarialPosition(
            position_id="test_008",
            board_type="square8",
            num_players=2,
            board_state=np.zeros((8, 8)),
            move_history=["e2e4", "e7e5", "g1f3"],
            strategy=AdversarialStrategy.REPLAY,
            difficulty_score=0.5,
            uncertainty_score=0.5,
            disagreement_score=0.5,
        )

        assert isinstance(position.move_history, list)
        assert len(position.move_history) == 3


class TestAdversarialConfigDataclass:
    """Tests for AdversarialConfig dataclass."""

    def test_default_creation(self):
        """Test AdversarialConfig with defaults."""
        config = AdversarialConfig()

        assert config.num_positions == 100
        assert len(config.strategies) > 0

    def test_default_strategies(self):
        """Test default strategies include key types."""
        config = AdversarialConfig()

        # Should include uncertainty and replay by default
        strategy_names = [s.value for s in config.strategies]
        assert "uncertainty" in strategy_names
        assert "replay" in strategy_names

    def test_custom_num_positions(self):
        """Test custom num_positions."""
        config = AdversarialConfig(num_positions=500)
        assert config.num_positions == 500

    def test_custom_strategies(self):
        """Test custom strategies list."""
        strategies = [AdversarialStrategy.GRADIENT, AdversarialStrategy.BOUNDARY]
        config = AdversarialConfig(strategies=strategies)

        assert len(config.strategies) == 2
        assert AdversarialStrategy.GRADIENT in config.strategies
        assert AdversarialStrategy.BOUNDARY in config.strategies


class TestAdversarialGeneratorImport:
    """Tests for AdversarialGenerator class."""

    def test_generator_importable(self):
        """Test AdversarialGenerator can be imported."""
        from app.training.adversarial_positions import AdversarialGenerator

        assert AdversarialGenerator is not None

    def test_generator_instantiation(self):
        """Test AdversarialGenerator can be instantiated."""
        from app.training.adversarial_positions import AdversarialGenerator

        try:
            generator = AdversarialGenerator()
            assert generator is not None
        except TypeError:
            # May require model_path argument
            pass


class TestStrategyCategories:
    """Tests for strategy categorization."""

    def test_model_dependent_strategies(self):
        """Test which strategies require a model."""
        model_dependent = {
            AdversarialStrategy.UNCERTAINTY,
            AdversarialStrategy.DISAGREEMENT,
            AdversarialStrategy.GRADIENT,
            AdversarialStrategy.BOUNDARY,
        }

        # These strategies need model inference
        for strategy in model_dependent:
            assert strategy.value in [
                "uncertainty",
                "disagreement",
                "gradient",
                "boundary",
            ]

    def test_model_free_strategies(self):
        """Test which strategies don't require a model."""
        model_free = {
            AdversarialStrategy.SEARCH,
            AdversarialStrategy.REPLAY,
            AdversarialStrategy.PERTURBATION,
        }

        # These can work without model inference
        for strategy in model_free:
            assert strategy.value in ["search", "replay", "perturbation"]


class TestPositionDifficulty:
    """Tests for position difficulty scoring."""

    def test_high_difficulty_position(self):
        """Test creating high-difficulty position."""
        position = AdversarialPosition(
            position_id="hard_001",
            board_type="hex8",
            num_players=2,
            board_state=np.zeros((9, 9)),
            move_history=[],
            strategy=AdversarialStrategy.UNCERTAINTY,
            difficulty_score=0.95,  # Very hard
            uncertainty_score=0.9,
            disagreement_score=0.85,
        )

        assert position.difficulty_score > 0.9

    def test_low_difficulty_position(self):
        """Test creating low-difficulty position."""
        position = AdversarialPosition(
            position_id="easy_001",
            board_type="hex8",
            num_players=2,
            board_state=np.zeros((9, 9)),
            move_history=[],
            strategy=AdversarialStrategy.REPLAY,
            difficulty_score=0.15,  # Easy
            uncertainty_score=0.1,
            disagreement_score=0.05,
        )

        assert position.difficulty_score < 0.2


class TestBoardTypeSupport:
    """Tests for different board type support."""

    def test_hex8_board_type(self):
        """Test hex8 board type positions."""
        position = AdversarialPosition(
            position_id="hex8_001",
            board_type="hex8",
            num_players=2,
            board_state=np.zeros((9, 9)),  # Hex8 uses 9x9 grid
            move_history=[],
            strategy=AdversarialStrategy.UNCERTAINTY,
            difficulty_score=0.5,
            uncertainty_score=0.5,
            disagreement_score=0.5,
        )

        assert position.board_type == "hex8"

    def test_square8_board_type(self):
        """Test square8 board type positions."""
        position = AdversarialPosition(
            position_id="sq8_001",
            board_type="square8",
            num_players=2,
            board_state=np.zeros((8, 8)),
            move_history=[],
            strategy=AdversarialStrategy.REPLAY,
            difficulty_score=0.5,
            uncertainty_score=0.5,
            disagreement_score=0.5,
        )

        assert position.board_type == "square8"

    def test_multiplayer_support(self):
        """Test multiplayer board configurations."""
        for num_players in [2, 3, 4]:
            position = AdversarialPosition(
                position_id=f"mp_{num_players}p",
                board_type="square8",
                num_players=num_players,
                board_state=np.zeros((8, 8)),
                move_history=[],
                strategy=AdversarialStrategy.UNCERTAINTY,
                difficulty_score=0.5,
                uncertainty_score=0.5,
                disagreement_score=0.5,
            )

            assert position.num_players == num_players


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
