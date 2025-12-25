"""Tests for GNNAI and HybridAI implementations.

These tests verify:
1. GNN and Hybrid AI classes inherit from BaseAI (factory contract)
2. Abstract methods are properly implemented
3. Factory can create these AI types
4. Difficulty tier integration works correctly
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from app.models import AIConfig
from app.ai.base import BaseAI
from app.ai.canonical_move_encoding import encode_move_for_board
from app.ai.neural_net.constants import get_policy_size_for_board
from app.game_engine import GameEngine
from app.models import BoardType
from app.training.initial_state import create_initial_state


class TestGNNAIInheritance:
    """Tests verifying GNNAI inherits from BaseAI correctly."""

    def test_gnnai_inherits_from_base_ai(self):
        """GNNAI should inherit from BaseAI."""
        from app.ai.gnn_ai import GNNAI

        assert issubclass(GNNAI, BaseAI), "GNNAI must inherit from BaseAI"

    def test_gnnai_has_required_abstract_methods(self):
        """GNNAI should implement required abstract methods."""
        from app.ai.gnn_ai import GNNAI

        # Check that abstract methods are implemented
        assert hasattr(GNNAI, "select_move")
        assert hasattr(GNNAI, "evaluate_position")
        assert callable(getattr(GNNAI, "select_move", None))
        assert callable(getattr(GNNAI, "evaluate_position", None))

    def test_gnnai_instance_has_base_ai_attributes(self):
        """GNNAI instance should have all BaseAI attributes."""
        from app.ai.gnn_ai import GNNAI

        config = AIConfig(difficulty=5)
        ai = GNNAI(player_number=1, config=config)

        # Check BaseAI attributes
        assert hasattr(ai, "player_number")
        assert hasattr(ai, "config")
        assert hasattr(ai, "move_count")
        assert hasattr(ai, "rng")
        assert hasattr(ai, "rng_seed")
        assert hasattr(ai, "rules_engine")

        # Verify values
        assert ai.player_number == 1
        assert ai.config is config
        assert ai.move_count == 0

    def test_gnnai_can_reset_for_new_game(self):
        """GNNAI should support reset_for_new_game from BaseAI."""
        from app.ai.gnn_ai import GNNAI

        config = AIConfig(difficulty=5)
        ai = GNNAI(player_number=1, config=config)

        # Simulate some game activity
        ai.move_count = 10
        initial_seed = ai.rng_seed

        # Reset for new game with new seed
        ai.reset_for_new_game(rng_seed=12345)

        assert ai.move_count == 0
        assert ai.rng_seed == 12345
        assert ai.rng_seed != initial_seed


class TestHybridAIInheritance:
    """Tests verifying HybridAI inherits from BaseAI correctly."""

    def test_hybrid_ai_inherits_from_base_ai(self):
        """HybridAI should inherit from BaseAI."""
        from app.ai.hybrid_ai import HybridAI

        assert issubclass(HybridAI, BaseAI), "HybridAI must inherit from BaseAI"

    def test_hybrid_ai_has_required_abstract_methods(self):
        """HybridAI should implement required abstract methods."""
        from app.ai.hybrid_ai import HybridAI

        assert hasattr(HybridAI, "select_move")
        assert hasattr(HybridAI, "evaluate_position")
        assert callable(getattr(HybridAI, "select_move", None))
        assert callable(getattr(HybridAI, "evaluate_position", None))

    def test_hybrid_ai_instance_has_base_ai_attributes(self):
        """HybridAI instance should have all BaseAI attributes."""
        from app.ai.hybrid_ai import HybridAI

        config = AIConfig(difficulty=5)
        ai = HybridAI(player_number=2, config=config)

        # Check BaseAI attributes
        assert hasattr(ai, "player_number")
        assert hasattr(ai, "config")
        assert hasattr(ai, "move_count")
        assert hasattr(ai, "rng")
        assert hasattr(ai, "rng_seed")
        assert hasattr(ai, "rules_engine")

        # Verify values
        assert ai.player_number == 2
        assert ai.config is config


class TestGNNAIFunctionality:
    """Tests for GNNAI functionality without a loaded model."""

    def test_gnnai_evaluate_position_without_model(self):
        """GNNAI.evaluate_position should return 0.0 without model."""
        from app.ai.gnn_ai import GNNAI

        config = AIConfig(difficulty=5)
        ai = GNNAI(player_number=1, config=config)

        # Create a mock game state
        mock_state = MagicMock()
        mock_state.board = MagicMock()
        mock_state.board.type = "hex8"

        # Without a model, should return 0.0
        value = ai.evaluate_position(mock_state)
        assert value == 0.0

    def test_gnnai_get_value_returns_float(self):
        """GNNAI.get_value should return a float."""
        from app.ai.gnn_ai import GNNAI

        config = AIConfig(difficulty=5)
        ai = GNNAI(player_number=1, config=config)

        mock_state = MagicMock()
        value = ai.get_value(mock_state)

        assert isinstance(value, float)

    def test_gnnai_custom_device(self):
        """GNNAI should accept custom device parameter."""
        from app.ai.gnn_ai import GNNAI

        config = AIConfig(difficulty=5)
        ai = GNNAI(player_number=1, config=config, device="cpu")

        assert ai.device == "cpu"

    def test_gnnai_custom_temperature(self):
        """GNNAI should accept custom temperature parameter."""
        from app.ai.gnn_ai import GNNAI

        config = AIConfig(difficulty=5)
        ai = GNNAI(player_number=1, config=config, temperature=0.5)

        assert ai.temperature == 0.5


class _FakeGNNModel:
    """Minimal GNN policy stub that prefers a specific action index."""

    def __init__(self, action_space_size: int, preferred_idx: int):
        self._action_space_size = action_space_size
        self._preferred_idx = preferred_idx

    def __call__(self, **_: object):
        logits = torch.full((1, self._action_space_size), -1e9, dtype=torch.float32)
        logits[0, self._preferred_idx] = 0.0
        value = torch.zeros((1, 4), dtype=torch.float32)
        return logits, value


class TestGNNAIPolicySelection:
    """Tests for GNNAI policy selection with canonical encoding."""

    def test_gnnai_selects_highest_logit_move_square8(self, monkeypatch):
        from app.ai.gnn_ai import GNNAI

        state = create_initial_state(board_type=BoardType.SQUARE8, num_players=2)
        legal_moves = GameEngine.get_valid_moves(state, 1)
        assert legal_moves, "Expected legal moves in initial state."

        preferred_move = legal_moves[0]
        preferred_idx = encode_move_for_board(preferred_move, state)
        assert preferred_idx >= 0

        action_space_size = get_policy_size_for_board(BoardType.SQUARE8)

        ai = GNNAI(player_number=1, config=AIConfig(difficulty=8), model_path=None)
        ai.action_space_size = action_space_size
        ai.model = _FakeGNNModel(action_space_size, preferred_idx)

        dummy_graph = SimpleNamespace(
            x=torch.zeros((1, 32), dtype=torch.float32),
            edge_index=torch.zeros((2, 1), dtype=torch.long),
        )
        monkeypatch.setattr(ai, "_state_to_graph", lambda _: dummy_graph)

        np.random.seed(0)

        selected = ai.select_move(state)
        assert selected == preferred_move

    def test_gnnai_selects_highest_logit_move_hex8(self, monkeypatch):
        from app.ai.gnn_ai import GNNAI

        state = create_initial_state(board_type=BoardType.HEX8, num_players=2)
        legal_moves = GameEngine.get_valid_moves(state, 1)
        assert legal_moves, "Expected legal moves in initial state."

        preferred_move = None
        preferred_idx = -1
        for move in legal_moves:
            idx = encode_move_for_board(move, state)
            if idx >= 0:
                preferred_move = move
                preferred_idx = idx
                break

        assert preferred_move is not None

        action_space_size = get_policy_size_for_board(BoardType.HEX8)

        ai = GNNAI(player_number=1, config=AIConfig(difficulty=8), model_path=None)
        ai.action_space_size = action_space_size
        ai.model = _FakeGNNModel(action_space_size, preferred_idx)

        dummy_graph = SimpleNamespace(
            x=torch.zeros((1, 32), dtype=torch.float32),
            edge_index=torch.zeros((2, 1), dtype=torch.long),
        )
        monkeypatch.setattr(ai, "_state_to_graph", lambda _: dummy_graph)

        np.random.seed(0)

        selected = ai.select_move(state)
        assert selected == preferred_move


class TestHybridAIFunctionality:
    """Tests for HybridAI functionality without a loaded model."""

    def test_hybrid_ai_evaluate_position_without_model(self):
        """HybridAI.evaluate_position should return 0.0 without model."""
        from app.ai.hybrid_ai import HybridAI

        config = AIConfig(difficulty=5)
        ai = HybridAI(player_number=1, config=config)

        mock_state = MagicMock()
        value = ai.evaluate_position(mock_state)

        assert value == 0.0

    def test_hybrid_ai_get_value_returns_float(self):
        """HybridAI.get_value should return a float."""
        from app.ai.hybrid_ai import HybridAI

        config = AIConfig(difficulty=5)
        ai = HybridAI(player_number=1, config=config)

        mock_state = MagicMock()
        value = ai.get_value(mock_state)

        assert isinstance(value, float)


class TestFactoryIntegration:
    """Tests for AIFactory integration with GNN/Hybrid AI."""

    def test_factory_can_get_gnn_class(self):
        """AIFactory should be able to get GNNAI class."""
        from app.ai.factory import AIFactory
        from app.models.core import AIType

        ai_class = AIFactory._get_ai_class(AIType.GNN)

        from app.ai.gnn_ai import GNNAI
        assert ai_class is GNNAI

    def test_factory_can_get_hybrid_class(self):
        """AIFactory should be able to get HybridAI class."""
        from app.ai.factory import AIFactory
        from app.models.core import AIType

        ai_class = AIFactory._get_ai_class(AIType.HYBRID)

        from app.ai.hybrid_ai import HybridAI
        assert ai_class is HybridAI

    def test_factory_create_gnn_ai(self):
        """AIFactory should be able to create GNNAI instances."""
        from app.ai.factory import AIFactory
        from app.ai.base import BaseAI
        from app.models.core import AIType

        config = AIConfig(difficulty=5)
        ai = AIFactory.create(AIType.GNN, player_number=1, config=config)

        assert isinstance(ai, BaseAI)
        assert ai.player_number == 1

    def test_factory_create_hybrid_ai(self):
        """AIFactory should be able to create HybridAI instances."""
        from app.ai.factory import AIFactory
        from app.ai.base import BaseAI
        from app.models.core import AIType

        config = AIConfig(difficulty=5)
        ai = AIFactory.create(AIType.HYBRID, player_number=1, config=config)

        assert isinstance(ai, BaseAI)
        assert ai.player_number == 1

    def test_factory_tournament_gnn(self):
        """AIFactory.create_for_tournament should work with 'gnn' agent_id."""
        from app.ai.factory import AIFactory
        from app.ai.base import BaseAI

        ai = AIFactory.create_for_tournament(
            agent_id="gnn",
            player_number=1,
            board_type="hex8",
            num_players=2,
        )

        assert isinstance(ai, BaseAI)

    def test_factory_tournament_hybrid(self):
        """AIFactory.create_for_tournament should work with 'hybrid' agent_id."""
        from app.ai.factory import AIFactory
        from app.ai.base import BaseAI

        ai = AIFactory.create_for_tournament(
            agent_id="hybrid",
            player_number=1,
            board_type="hex8",
            num_players=2,
        )

        assert isinstance(ai, BaseAI)


class TestDifficultyProfiles:
    """Tests for difficulty profile integration."""

    def test_gnn_difficulty_profiles_exist(self):
        """Difficulty profiles for GNN should exist (D22-D23)."""
        from app.ai.factory import CANONICAL_DIFFICULTY_PROFILES
        from app.models.core import AIType

        assert 22 in CANONICAL_DIFFICULTY_PROFILES
        assert 23 in CANONICAL_DIFFICULTY_PROFILES

        assert CANONICAL_DIFFICULTY_PROFILES[22]["ai_type"] == AIType.GNN
        assert CANONICAL_DIFFICULTY_PROFILES[23]["ai_type"] == AIType.GNN

    def test_hybrid_difficulty_profile_exists(self):
        """Difficulty profile for Hybrid should exist (D24)."""
        from app.ai.factory import CANONICAL_DIFFICULTY_PROFILES
        from app.models.core import AIType

        assert 24 in CANONICAL_DIFFICULTY_PROFILES
        assert CANONICAL_DIFFICULTY_PROFILES[24]["ai_type"] == AIType.HYBRID

    def test_get_difficulty_profile_gnn(self):
        """get_difficulty_profile should return GNN profile for D22."""
        from app.ai.factory import get_difficulty_profile
        from app.models.core import AIType

        profile = get_difficulty_profile(22)

        assert profile["ai_type"] == AIType.GNN
        assert profile["use_neural_net"] is True

    def test_get_difficulty_profile_hybrid(self):
        """get_difficulty_profile should return Hybrid profile for D24."""
        from app.ai.factory import get_difficulty_profile
        from app.models.core import AIType

        profile = get_difficulty_profile(24)

        assert profile["ai_type"] == AIType.HYBRID
        assert profile["use_neural_net"] is True

    def test_difficulty_descriptions_exist(self):
        """Difficulty descriptions should exist for GNN/Hybrid levels."""
        from app.ai.factory import DIFFICULTY_DESCRIPTIONS

        assert 22 in DIFFICULTY_DESCRIPTIONS
        assert 23 in DIFFICULTY_DESCRIPTIONS
        assert 24 in DIFFICULTY_DESCRIPTIONS

        assert "GNN" in DIFFICULTY_DESCRIPTIONS[22]
        assert "GNN" in DIFFICULTY_DESCRIPTIONS[23]
        assert "Hybrid" in DIFFICULTY_DESCRIPTIONS[24]


class TestCreateFunctions:
    """Tests for factory functions (create_gnn_ai, create_hybrid_ai)."""

    def test_create_gnn_ai_default_config(self):
        """create_gnn_ai should create AI with default config."""
        from app.ai.gnn_ai import create_gnn_ai

        ai = create_gnn_ai(player_number=1)

        assert ai.player_number == 1
        assert ai.config.difficulty == 6  # Default difficulty

    def test_create_gnn_ai_custom_config(self):
        """create_gnn_ai should accept custom config."""
        from app.ai.gnn_ai import create_gnn_ai

        config = AIConfig(difficulty=8)
        ai = create_gnn_ai(player_number=1, config=config)

        assert ai.config.difficulty == 8

    def test_create_hybrid_ai_default_config(self):
        """create_hybrid_ai should create AI with default config."""
        from app.ai.hybrid_ai import create_hybrid_ai

        ai = create_hybrid_ai(player_number=1)

        assert ai.player_number == 1
        assert ai.config.difficulty == 6

    def test_create_hybrid_ai_custom_config(self):
        """create_hybrid_ai should accept custom config."""
        from app.ai.hybrid_ai import create_hybrid_ai

        config = AIConfig(difficulty=9)
        ai = create_hybrid_ai(player_number=2, config=config)

        assert ai.player_number == 2
        assert ai.config.difficulty == 9


class TestBaseAIMethodAvailability:
    """Tests verifying BaseAI helper methods are available."""

    def test_gnnai_has_get_valid_moves(self):
        """GNNAI should have get_valid_moves from BaseAI."""
        from app.ai.gnn_ai import GNNAI

        config = AIConfig(difficulty=5)
        ai = GNNAI(player_number=1, config=config)

        assert hasattr(ai, "get_valid_moves")
        assert callable(ai.get_valid_moves)

    def test_gnnai_has_should_pick_random_move(self):
        """GNNAI should have should_pick_random_move from BaseAI."""
        from app.ai.gnn_ai import GNNAI

        config = AIConfig(difficulty=5, randomness=0.5)
        ai = GNNAI(player_number=1, config=config)

        assert hasattr(ai, "should_pick_random_move")
        assert callable(ai.should_pick_random_move)

    def test_gnnai_has_get_random_element(self):
        """GNNAI should have get_random_element from BaseAI."""
        from app.ai.gnn_ai import GNNAI

        config = AIConfig(difficulty=5)
        ai = GNNAI(player_number=1, config=config)

        assert hasattr(ai, "get_random_element")
        result = ai.get_random_element([1, 2, 3])
        assert result in [1, 2, 3]

    def test_hybrid_ai_has_shuffle_array(self):
        """HybridAI should have shuffle_array from BaseAI."""
        from app.ai.hybrid_ai import HybridAI

        config = AIConfig(difficulty=5)
        ai = HybridAI(player_number=1, config=config)

        assert hasattr(ai, "shuffle_array")
        items = [1, 2, 3, 4, 5]
        result = ai.shuffle_array(items)
        assert set(result) == {1, 2, 3, 4, 5}

    def test_gnnai_repr(self):
        """GNNAI should have proper __repr__ from BaseAI."""
        from app.ai.gnn_ai import GNNAI

        config = AIConfig(difficulty=7)
        ai = GNNAI(player_number=2, config=config)

        repr_str = repr(ai)
        assert "GNNAI" in repr_str
        assert "player=2" in repr_str
        assert "difficulty=7" in repr_str
