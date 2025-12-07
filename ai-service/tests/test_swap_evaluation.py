"""Tests for swap rule (pie rule) AI evaluation and training diversity.

This test suite verifies that:
1. HeuristicAI can evaluate swap opportunities strategically
2. Training randomness creates diverse swap decisions
3. Swap evaluation works correctly across board types
"""

import pytest
import numpy as np
from app.ai.heuristic_ai import HeuristicAI
from app.models import AIConfig, GameState, BoardType, MoveType
from app.game_engine import GameEngine


class TestSwapEvaluation:
    """Test swap rule evaluation in HeuristicAI."""

    def test_deterministic_swap_evaluation(self):
        """Test that swap evaluation is deterministic by default."""
        # Create a game state where P1 has played center
        config = AIConfig(
            ai_type="heuristic",
            difficulty=5,
            randomness=0.0,
            heuristic_profile_id="heuristic_v1_balanced"
        )

        ai = HeuristicAI(player_number=2, config=config)

        # Verify temperature is <= 0 (deterministic - noise only added when > 0)
        assert ai.WEIGHT_SWAP_EXPLORATION_TEMPERATURE <= 0
        
    @pytest.mark.skip(reason="Uses non-existent GameEngine.create_game - needs rewrite")
    def test_swap_strong_center_opening(self):
        """Test that AI swaps when P1 plays a strong center opening."""
        # Create initial 2-player game
        state = GameEngine.create_game(
            board_type=BoardType.SQUARE8,
            board_size=8,
            num_players=2,
            rules_options={"swapRuleEnabled": True}
        )
        
        # P1 places ring in center (strong opening)
        center_move = None
        p1_moves = GameEngine.get_valid_moves(state, 1)
        for move in p1_moves:
            if move.type == MoveType.PLACE_RING:
                # Find center position (3,3) or (4,4) on 8x8
                if (move.to.x in [3, 4]) and (move.to.y in [3, 4]):
                    center_move = move
                    break
        
        assert center_move is not None, "Should find center placement"
        state = GameEngine.apply_move(state, center_move)
        
        # Now P2 should see swap_sides as an option
        p2_moves = GameEngine.get_valid_moves(state, 2)
        swap_moves = [m for m in p2_moves if m.type == MoveType.SWAP_SIDES]
        assert len(swap_moves) == 1, "P2 should see exactly one swap move"
        
        # Evaluate swap opportunity (should be positive for center opening)
        config = AIConfig(ai_type="heuristic", difficulty=5, randomness=0.0)
        ai = HeuristicAI(player_number=2, config=config)
        
        swap_value = ai.evaluate_swap_opening_bonus(state)
        assert swap_value > 0, "Center opening should have positive swap value"
        
    @pytest.mark.skip(reason="Uses non-existent GameEngine.create_game - needs rewrite")
    def test_swap_weak_corner_opening(self):
        """Test that AI does NOT swap when P1 plays a weak corner opening."""
        # Create initial 2-player game
        state = GameEngine.create_game(
            board_type=BoardType.SQUARE8,
            board_size=8,
            num_players=2,
            rules_options={"swapRuleEnabled": True}
        )
        
        # P1 places ring in corner (weak opening)
        corner_move = None
        p1_moves = GameEngine.get_valid_moves(state, 1)
        for move in p1_moves:
            if move.type == MoveType.PLACE_RING:
                if move.to.x == 0 and move.to.y == 0:
                    corner_move = move
                    break
        
        assert corner_move is not None, "Should find corner placement"
        state = GameEngine.apply_move(state, corner_move)
        
        # Evaluate swap opportunity (should be low/negative for corner)
        config = AIConfig(ai_type="heuristic", difficulty=5, randomness=0.0)
        ai = HeuristicAI(player_number=2, config=config)
        
        swap_value = ai.evaluate_swap_opening_bonus(state)
        # Corner should have lower value than center
        assert swap_value < 15.0, "Corner opening should have lower swap value"
        
    @pytest.mark.skip(reason="Uses non-existent GameEngine.create_game - needs rewrite")
    def test_stochastic_swap_creates_diversity(self):
        """Test that training randomness creates diverse swap decisions."""
        # Create game with center opening
        state = GameEngine.create_game(
            board_type=BoardType.SQUARE8,
            board_size=8,
            num_players=2,
            rules_options={"swapRuleEnabled": True}
        )
        
        # P1 plays center
        p1_moves = GameEngine.get_valid_moves(state, 1)
        center_move = next(
            (m for m in p1_moves 
             if m.type == MoveType.PLACE_RING 
             and m.to.x in [3, 4] and m.to.y in [3, 4]),
            None
        )
        state = GameEngine.apply_move(state, center_move)
        
        # Create AI with training randomness enabled
        config = AIConfig(ai_type="heuristic", difficulty=5, randomness=0.0)
        ai = HeuristicAI(player_number=2, config=config)
        
        # Override temperature for training
        ai.WEIGHT_SWAP_EXPLORATION_TEMPERATURE = 10.0
        
        # Evaluate swap multiple times and collect decisions
        swap_values = []
        for _ in range(100):
            # Get valid moves including swap
            moves = ai.get_valid_moves(state)
            swap_move = next(
                (m for m in moves if m.type == MoveType.SWAP_SIDES),
                None
            )
            
            if swap_move:
                # Simulate the evaluation with randomness
                value = ai.evaluate_swap_opening_bonus(state)
                # Add noise as done in select_move
                if ai.WEIGHT_SWAP_EXPLORATION_TEMPERATURE > 0:
                    noise = np.random.normal(
                        0,
                        ai.WEIGHT_SWAP_EXPLORATION_TEMPERATURE
                    )
                    value += noise
                swap_values.append(value)
        
        # Verify diversity: values should have variance
        assert len(swap_values) > 0
        variance = np.var(swap_values)
        assert variance > 0, "Stochastic mode should create variance in swap values"
        
        # Some evaluations should be positive, some negative (with enough noise)
        # This creates training diversity
        
    def test_swap_training_mode_flag(self):
        """Test that training mode can be control via weight."""
        config = AIConfig(ai_type="heuristic", difficulty=5, randomness=0.0)
        ai = HeuristicAI(player_number=2, config=config)

        # Default: deterministic (temperature <= 0)
        assert ai.WEIGHT_SWAP_EXPLORATION_TEMPERATURE <= 0
        
        # Can be overridden for training
        ai.WEIGHT_SWAP_EXPLORATION_TEMPERATURE = 0.15
        assert ai.WEIGHT_SWAP_EXPLORATION_TEMPERATURE == 0.15


class TestSwapMultiplayer:
    """Test that swap does NOT apply in multiplayer games."""

    @pytest.mark.skip(reason="Uses non-existent GameEngine.create_game - needs rewrite")
    def test_no_swap_in_3player(self):
        """Verify swap is not offered in 3-player games."""
        state = GameEngine.create_game(
            board_type=BoardType.SQUARE8,
            board_size=8,
            num_players=3,
            rules_options={"swapRuleEnabled": True}  # Even if enabled
        )
        
        # P1 makes a move
        p1_moves = GameEngine.get_valid_moves(state, 1)
        state = GameEngine.apply_move(state, p1_moves[0])
        
        # P2 should NOT see swap
        p2_moves = GameEngine.get_valid_moves(state, 2)
        swap_moves = [m for m in p2_moves if m.type == MoveType.SWAP_SIDES]
        assert len(swap_moves) == 0, "Swap should not be offered in multiplayer"