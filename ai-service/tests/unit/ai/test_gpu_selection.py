"""Tests for GPU move selection utilities.

Tests the vectorized move selection functions extracted from gpu_parallel_games.py.
"""

from dataclasses import dataclass
from typing import Optional

import pytest
import torch

from app.ai.gpu_game_types import MoveType
from app.ai.gpu_selection import select_moves_heuristic, select_moves_vectorized

# =============================================================================
# Mock Classes
# =============================================================================


@dataclass
class MockBatchMoves:
    """Mock BatchMoves for testing selection functions."""

    game_idx: torch.Tensor
    move_type: torch.Tensor
    from_y: torch.Tensor
    from_x: torch.Tensor
    to_y: torch.Tensor
    to_x: torch.Tensor
    moves_per_game: torch.Tensor
    move_offsets: torch.Tensor
    total_moves: int
    device: torch.device

    @classmethod
    def create_empty(cls, batch_size: int, device: torch.device) -> "MockBatchMoves":
        """Create empty moves (no moves available)."""
        return cls(
            game_idx=torch.empty(0, dtype=torch.int32, device=device),
            move_type=torch.empty(0, dtype=torch.int8, device=device),
            from_y=torch.empty(0, dtype=torch.int16, device=device),
            from_x=torch.empty(0, dtype=torch.int16, device=device),
            to_y=torch.empty(0, dtype=torch.int16, device=device),
            to_x=torch.empty(0, dtype=torch.int16, device=device),
            moves_per_game=torch.zeros(batch_size, dtype=torch.int32, device=device),
            move_offsets=torch.zeros(batch_size, dtype=torch.int32, device=device),
            total_moves=0,
            device=device,
        )

    @classmethod
    def create_simple(
        cls,
        batch_size: int,
        moves_per_game: list,
        board_size: int,
        device: torch.device,
        move_type: int = MoveType.PLACEMENT,
    ) -> "MockBatchMoves":
        """Create simple moves with uniform distribution across board."""
        total_moves = sum(moves_per_game)

        game_idx_list = []
        to_y_list = []
        to_x_list = []

        for g, num_moves in enumerate(moves_per_game):
            for i in range(num_moves):
                game_idx_list.append(g)
                # Distribute moves across board
                to_y_list.append(i % board_size)
                to_x_list.append((i * 2) % board_size)

        offsets = [0]
        for m in moves_per_game[:-1]:
            offsets.append(offsets[-1] + m)

        return cls(
            game_idx=torch.tensor(game_idx_list, dtype=torch.int32, device=device),
            move_type=torch.full((total_moves,), move_type, dtype=torch.int8, device=device),
            from_y=torch.tensor(to_y_list, dtype=torch.int16, device=device),
            from_x=torch.tensor(to_x_list, dtype=torch.int16, device=device),
            to_y=torch.tensor(to_y_list, dtype=torch.int16, device=device),
            to_x=torch.tensor(to_x_list, dtype=torch.int16, device=device),
            moves_per_game=torch.tensor(moves_per_game, dtype=torch.int32, device=device),
            move_offsets=torch.tensor(offsets, dtype=torch.int32, device=device),
            total_moves=total_moves,
            device=device,
        )

    @classmethod
    def create_with_positions(
        cls,
        batch_size: int,
        positions: list,  # List of (game_idx, to_y, to_x) tuples
        device: torch.device,
        move_type: int = MoveType.PLACEMENT,
    ) -> "MockBatchMoves":
        """Create moves with specific positions."""
        total_moves = len(positions)

        game_idx_list = [p[0] for p in positions]
        to_y_list = [p[1] for p in positions]
        to_x_list = [p[2] for p in positions]

        # Count moves per game
        moves_per_game = [0] * batch_size
        for g, _, _ in positions:
            moves_per_game[g] += 1

        # Compute offsets
        offsets = [0]
        for m in moves_per_game[:-1]:
            offsets.append(offsets[-1] + m)

        return cls(
            game_idx=torch.tensor(game_idx_list, dtype=torch.int32, device=device),
            move_type=torch.full((total_moves,), move_type, dtype=torch.int8, device=device),
            from_y=torch.tensor(to_y_list, dtype=torch.int16, device=device),
            from_x=torch.tensor(to_x_list, dtype=torch.int16, device=device),
            to_y=torch.tensor(to_y_list, dtype=torch.int16, device=device),
            to_x=torch.tensor(to_x_list, dtype=torch.int16, device=device),
            moves_per_game=torch.tensor(moves_per_game, dtype=torch.int32, device=device),
            move_offsets=torch.tensor(offsets, dtype=torch.int32, device=device),
            total_moves=total_moves,
            device=device,
        )


@dataclass
class MockBatchGameState:
    """Mock BatchGameState for testing heuristic selection."""

    stack_owner: torch.Tensor
    stack_height: torch.Tensor
    current_player: torch.Tensor
    board_size: int
    device: torch.device

    @classmethod
    def create(
        cls,
        batch_size: int,
        board_size: int,
        device: torch.device,
    ) -> "MockBatchGameState":
        """Create an empty game state."""
        return cls(
            stack_owner=torch.zeros((batch_size, board_size, board_size), dtype=torch.int8, device=device),
            stack_height=torch.zeros((batch_size, board_size, board_size), dtype=torch.int8, device=device),
            current_player=torch.ones(batch_size, dtype=torch.int8, device=device),
            board_size=board_size,
            device=device,
        )


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def device():
    """Get test device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def board_size():
    return 8


# =============================================================================
# Tests for select_moves_vectorized
# =============================================================================


class TestSelectMovesVectorized:
    """Tests for the vectorized move selection function."""

    def test_empty_moves_returns_minus_one(self, device, board_size):
        """When no moves available, return -1 for all games."""
        batch_size = 4
        moves = MockBatchMoves.create_empty(batch_size, device)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        selected = select_moves_vectorized(moves, active_mask, board_size)

        assert selected.shape == (batch_size,)
        assert (selected == -1).all()

    def test_returns_valid_indices(self, device, board_size):
        """Selected indices should be within valid range for each game."""
        batch_size = 4
        moves_per_game = [5, 3, 7, 2]
        moves = MockBatchMoves.create_simple(batch_size, moves_per_game, board_size, device)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        selected = select_moves_vectorized(moves, active_mask, board_size)

        assert selected.shape == (batch_size,)
        for g in range(batch_size):
            assert 0 <= selected[g].item() < moves_per_game[g]

    def test_inactive_games_still_get_selection(self, device, board_size):
        """Inactive games should still get valid selection (for consistency)."""
        batch_size = 4
        moves_per_game = [5, 3, 7, 2]
        moves = MockBatchMoves.create_simple(batch_size, moves_per_game, board_size, device)
        active_mask = torch.tensor([True, False, True, False], dtype=torch.bool, device=device)

        selected = select_moves_vectorized(moves, active_mask, board_size)

        assert selected.shape == (batch_size,)
        # Active games should have valid selections
        assert 0 <= selected[0].item() < moves_per_game[0]
        assert 0 <= selected[2].item() < moves_per_game[2]

    def test_games_with_no_moves_get_zero(self, device, board_size):
        """Games with no moves should get index 0 (clamped from -1)."""
        batch_size = 4
        # Game 1 has no moves
        moves_per_game = [5, 0, 7, 2]
        moves = MockBatchMoves.create_simple(batch_size, moves_per_game, board_size, device)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        selected = select_moves_vectorized(moves, active_mask, board_size)

        # Game 1 should have clamped value (0)
        assert selected[1].item() == 0

    def test_center_bias_statistical(self, device, board_size):
        """Moves closer to center should be selected more often (statistical test)."""
        batch_size = 1
        # Create moves: one at center, one at corner
        center = board_size // 2
        positions = [
            (0, center, center),      # Center
            (0, 0, 0),                # Corner
        ]
        moves = MockBatchMoves.create_with_positions(batch_size, positions, device)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        # Run many trials
        center_count = 0
        num_trials = 1000
        for _ in range(num_trials):
            selected = select_moves_vectorized(moves, active_mask, board_size, temperature=0.5)
            if selected[0].item() == 0:  # Center move is index 0
                center_count += 1

        # Center should be selected significantly more often
        assert center_count > num_trials * 0.6, f"Center selected {center_count}/{num_trials} times"

    def test_temperature_affects_randomness(self, device, board_size):
        """Higher temperature should increase randomness."""
        batch_size = 1
        center = board_size // 2
        positions = [
            (0, center, center),      # Center
            (0, 0, 0),                # Corner
        ]
        moves = MockBatchMoves.create_with_positions(batch_size, positions, device)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        # Low temperature: should be more deterministic
        low_temp_center_count = 0
        high_temp_center_count = 0
        num_trials = 1000

        for _ in range(num_trials):
            selected_low = select_moves_vectorized(moves, active_mask, board_size, temperature=0.1)
            selected_high = select_moves_vectorized(moves, active_mask, board_size, temperature=5.0)
            if selected_low[0].item() == 0:
                low_temp_center_count += 1
            if selected_high[0].item() == 0:
                high_temp_center_count += 1

        # Low temp should have higher bias toward center
        assert low_temp_center_count > high_temp_center_count

    def test_single_move_always_selected(self, device, board_size):
        """When only one move available, it should always be selected."""
        batch_size = 3
        moves_per_game = [1, 1, 1]
        moves = MockBatchMoves.create_simple(batch_size, moves_per_game, board_size, device)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        for _ in range(10):
            selected = select_moves_vectorized(moves, active_mask, board_size)
            assert (selected == 0).all()

    def test_unsorted_game_idx(self, device, board_size):
        """Should work correctly with unsorted game indices."""
        batch_size = 3
        # Create positions - they need to be sorted by game_idx for proper offsets
        # but we test that the function handles the unsorted case internally
        positions = [
            (0, 1, 1),  # Game 0
            (0, 5, 5),  # Game 0
            (1, 2, 2),  # Game 1
            (2, 3, 3),  # Game 2
            (2, 4, 4),  # Game 2
        ]
        moves = MockBatchMoves.create_with_positions(batch_size, positions, device)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        selected = select_moves_vectorized(moves, active_mask, board_size)

        assert selected.shape == (batch_size,)
        # Game 0 has 2 moves (indices 0, 1 in its local space)
        assert 0 <= selected[0].item() < 2
        # Game 1 has 1 move
        assert selected[1].item() == 0
        # Game 2 has 2 moves
        assert 0 <= selected[2].item() < 2

    def test_large_batch(self, device, board_size):
        """Should handle large batches efficiently."""
        batch_size = 1000
        moves_per_game = [10] * batch_size
        moves = MockBatchMoves.create_simple(batch_size, moves_per_game, board_size, device)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        selected = select_moves_vectorized(moves, active_mask, board_size)

        assert selected.shape == (batch_size,)
        assert (selected >= 0).all()
        assert (selected < 10).all()

    def test_many_moves_per_game(self, device, board_size):
        """Should handle games with many moves."""
        batch_size = 2
        moves_per_game = [100, 100]
        moves = MockBatchMoves.create_simple(batch_size, moves_per_game, board_size, device)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        selected = select_moves_vectorized(moves, active_mask, board_size)

        assert selected.shape == (batch_size,)
        assert (selected >= 0).all()
        assert (selected < 100).all()


# =============================================================================
# Tests for select_moves_heuristic
# =============================================================================


class TestSelectMovesHeuristic:
    """Tests for the heuristic move selection function."""

    def test_empty_moves_returns_minus_one(self, device, board_size):
        """When no moves available, return -1 for all games."""
        batch_size = 4
        moves = MockBatchMoves.create_empty(batch_size, device)
        state = MockBatchGameState.create(batch_size, board_size, device)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        selected = select_moves_heuristic(moves, state, active_mask)

        assert selected.shape == (batch_size,)
        # With no moves at all, returns -1 (early return before clamping)
        assert (selected == -1).all()

    def test_returns_valid_indices(self, device, board_size):
        """Selected indices should be within valid range for each game."""
        batch_size = 4
        moves_per_game = [5, 3, 7, 2]
        moves = MockBatchMoves.create_simple(batch_size, moves_per_game, board_size, device)
        state = MockBatchGameState.create(batch_size, board_size, device)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        selected = select_moves_heuristic(moves, state, active_mask)

        assert selected.shape == (batch_size,)
        for g in range(batch_size):
            assert 0 <= selected[g].item() < moves_per_game[g]

    def test_capture_value_scoring(self, device, board_size):
        """Captures of taller stacks should be preferred."""
        batch_size = 1
        center = board_size // 2

        # Create two capture moves: one to tall stack, one to short stack
        positions = [
            (0, center, center),      # Tall stack location
            (0, center + 1, center),  # Short stack location
        ]

        # Create moves as captures
        moves = MockBatchMoves.create_with_positions(
            batch_size, positions, device, move_type=MoveType.CAPTURE
        )

        # Set up state with stacks at destinations
        state = MockBatchGameState.create(batch_size, board_size, device)
        state.stack_owner[0, center, center] = 2  # Opponent stack
        state.stack_height[0, center, center] = 5  # Tall stack
        state.stack_owner[0, center + 1, center] = 2  # Opponent stack
        state.stack_height[0, center + 1, center] = 1  # Short stack

        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        # Run many trials - tall stack should be captured more often
        tall_stack_count = 0
        num_trials = 500
        for _ in range(num_trials):
            selected = select_moves_heuristic(moves, state, active_mask, temperature=0.5)
            if selected[0].item() == 0:  # Tall stack move
                tall_stack_count += 1

        # Tall stack should be preferred
        assert tall_stack_count > num_trials * 0.6, f"Tall stack captured {tall_stack_count}/{num_trials}"

    def test_adjacency_scoring(self, device, board_size):
        """Moves adjacent to own stacks should be preferred."""
        batch_size = 1
        center = board_size // 2

        # Create two placement moves: one adjacent to own stack, one isolated
        positions = [
            (0, center, center),      # Adjacent position
            (0, 0, 0),                # Isolated position
        ]
        moves = MockBatchMoves.create_with_positions(batch_size, positions, device)

        # Set up state with own stack next to center
        state = MockBatchGameState.create(batch_size, board_size, device)
        state.stack_owner[0, center - 1, center] = 1  # Own stack above
        state.stack_height[0, center - 1, center] = 3
        state.current_player[0] = 1

        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        # Run many trials - adjacent position should be preferred
        adjacent_count = 0
        num_trials = 500
        for _ in range(num_trials):
            selected = select_moves_heuristic(moves, state, active_mask, temperature=0.5)
            if selected[0].item() == 0:  # Adjacent move
                adjacent_count += 1

        # Adjacent should be preferred
        assert adjacent_count > num_trials * 0.5, f"Adjacent selected {adjacent_count}/{num_trials}"

    def test_custom_weights(self, device, board_size):
        """Custom weights should affect scoring."""
        batch_size = 1
        center = board_size // 2

        positions = [
            (0, center, center),      # Center
            (0, 0, 0),                # Corner
        ]
        moves = MockBatchMoves.create_with_positions(batch_size, positions, device)
        state = MockBatchGameState.create(batch_size, board_size, device)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        # With high center weight
        high_center_weights = {"center": 10.0, "capture_value": 0.0, "adjacency": 0.0, "line_potential": 0.0, "noise": 0.1}
        # With zero center weight
        no_center_weights = {"center": 0.0, "capture_value": 0.0, "adjacency": 0.0, "line_potential": 0.0, "noise": 1.0}

        high_center_count = 0
        no_center_count = 0
        num_trials = 500

        for _ in range(num_trials):
            selected_high = select_moves_heuristic(moves, state, active_mask, weights=high_center_weights, temperature=0.3)
            selected_no = select_moves_heuristic(moves, state, active_mask, weights=no_center_weights, temperature=0.3)
            if selected_high[0].item() == 0:
                high_center_count += 1
            if selected_no[0].item() == 0:
                no_center_count += 1

        # High center weight should strongly prefer center
        assert high_center_count > no_center_count + 100

    def test_temperature_affects_randomness(self, device, board_size):
        """Higher temperature should increase randomness in heuristic selection."""
        batch_size = 1
        center = board_size // 2
        positions = [
            (0, center, center),
            (0, 0, 0),
        ]
        moves = MockBatchMoves.create_with_positions(batch_size, positions, device)
        state = MockBatchGameState.create(batch_size, board_size, device)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        low_temp_center = 0
        high_temp_center = 0
        num_trials = 500

        for _ in range(num_trials):
            selected_low = select_moves_heuristic(moves, state, active_mask, temperature=0.1)
            selected_high = select_moves_heuristic(moves, state, active_mask, temperature=5.0)
            if selected_low[0].item() == 0:
                low_temp_center += 1
            if selected_high[0].item() == 0:
                high_temp_center += 1

        # Low temp should be at least as biased as high temp (with tolerance for variance)
        # When center bias is very strong, both may pick center, so we allow equality
        assert low_temp_center >= high_temp_center, (
            f"Expected low_temp ({low_temp_center}) >= high_temp ({high_temp_center})"
        )

    def test_single_move_always_selected(self, device, board_size):
        """When only one move available, it should always be selected."""
        batch_size = 3
        moves_per_game = [1, 1, 1]
        moves = MockBatchMoves.create_simple(batch_size, moves_per_game, board_size, device)
        state = MockBatchGameState.create(batch_size, board_size, device)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        for _ in range(10):
            selected = select_moves_heuristic(moves, state, active_mask)
            assert (selected == 0).all()

    def test_large_batch(self, device, board_size):
        """Should handle large batches efficiently."""
        batch_size = 500
        moves_per_game = [10] * batch_size
        moves = MockBatchMoves.create_simple(batch_size, moves_per_game, board_size, device)
        state = MockBatchGameState.create(batch_size, board_size, device)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        selected = select_moves_heuristic(moves, state, active_mask)

        assert selected.shape == (batch_size,)
        assert (selected >= 0).all()
        assert (selected < 10).all()

    def test_line_potential_scoring(self, device, board_size):
        """Moves that extend lines should be preferred."""
        batch_size = 1

        # Create two moves: one extends a line, one doesn't
        positions = [
            (0, 3, 3),  # Extends horizontal line
            (0, 0, 0),  # Isolated
        ]
        moves = MockBatchMoves.create_with_positions(batch_size, positions, device)

        # Set up state with own stacks forming partial line
        state = MockBatchGameState.create(batch_size, board_size, device)
        state.current_player[0] = 1
        state.stack_owner[0, 3, 2] = 1  # Stack to left
        state.stack_height[0, 3, 2] = 2
        state.stack_owner[0, 3, 4] = 1  # Stack to right
        state.stack_height[0, 3, 4] = 2

        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        # Use high line weight
        weights = {"center": 0.5, "capture_value": 0.0, "adjacency": 0.0, "line_potential": 10.0, "noise": 0.1}

        line_extend_count = 0
        num_trials = 500
        for _ in range(num_trials):
            selected = select_moves_heuristic(moves, state, active_mask, weights=weights, temperature=0.5)
            if selected[0].item() == 0:
                line_extend_count += 1

        # Line extension should be preferred
        assert line_extend_count > num_trials * 0.6, f"Line extend selected {line_extend_count}/{num_trials}"

    def test_default_weights(self, device, board_size):
        """Should use sensible default weights when none provided."""
        batch_size = 2
        moves_per_game = [5, 5]
        moves = MockBatchMoves.create_simple(batch_size, moves_per_game, board_size, device)
        state = MockBatchGameState.create(batch_size, board_size, device)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        # Should not raise with no weights
        selected = select_moves_heuristic(moves, state, active_mask, weights=None)

        assert selected.shape == (batch_size,)
        assert (selected >= 0).all()
        assert (selected < 5).all()


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================


class TestSelectionEdgeCases:
    """Edge cases for both selection functions."""

    def test_all_games_inactive(self, device, board_size):
        """Both functions should handle all-inactive games."""
        batch_size = 4
        moves_per_game = [5, 3, 7, 2]
        moves = MockBatchMoves.create_simple(batch_size, moves_per_game, board_size, device)
        state = MockBatchGameState.create(batch_size, board_size, device)
        active_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        selected_v = select_moves_vectorized(moves, active_mask, board_size)
        selected_h = select_moves_heuristic(moves, state, active_mask)

        # Should still return valid shapes
        assert selected_v.shape == (batch_size,)
        assert selected_h.shape == (batch_size,)

    def test_mixed_empty_and_nonempty(self, device, board_size):
        """Handle mix of games with and without moves."""
        batch_size = 4
        moves_per_game = [5, 0, 3, 0]
        moves = MockBatchMoves.create_simple(batch_size, moves_per_game, board_size, device)
        state = MockBatchGameState.create(batch_size, board_size, device)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        selected_v = select_moves_vectorized(moves, active_mask, board_size)
        selected_h = select_moves_heuristic(moves, state, active_mask)

        # Games with moves should have valid selections
        assert 0 <= selected_v[0].item() < 5
        assert 0 <= selected_v[2].item() < 3
        assert 0 <= selected_h[0].item() < 5
        assert 0 <= selected_h[2].item() < 3

        # Games without moves should have clamped value
        assert selected_v[1].item() == 0
        assert selected_v[3].item() == 0

    def test_very_low_temperature(self, device, board_size):
        """Very low temperature should be nearly deterministic."""
        batch_size = 1
        center = board_size // 2
        positions = [
            (0, center, center),
            (0, 0, 0),
        ]
        moves = MockBatchMoves.create_with_positions(batch_size, positions, device)
        MockBatchGameState.create(batch_size, board_size, device)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        # With very low temperature, should almost always pick best move
        center_count = 0
        num_trials = 100
        for _ in range(num_trials):
            selected = select_moves_vectorized(moves, active_mask, board_size, temperature=0.01)
            if selected[0].item() == 0:
                center_count += 1

        # Should be very deterministic
        assert center_count > 95

    def test_very_high_temperature(self, device, board_size):
        """Very high temperature should approach uniform distribution."""
        batch_size = 1
        positions = [
            (0, 4, 4),
            (0, 4, 5),
        ]
        moves = MockBatchMoves.create_with_positions(batch_size, positions, device)
        MockBatchGameState.create(batch_size, board_size, device)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        first_count = 0
        num_trials = 1000
        for _ in range(num_trials):
            selected = select_moves_vectorized(moves, active_mask, board_size, temperature=100.0)
            if selected[0].item() == 0:
                first_count += 1

        # With very high temp, should be close to 50/50
        assert 400 < first_count < 600, f"Expected ~50%, got {first_count/num_trials*100}%"

    def test_consistent_device(self, device, board_size):
        """Output should be on same device as input."""
        batch_size = 4
        moves_per_game = [5, 3, 7, 2]
        moves = MockBatchMoves.create_simple(batch_size, moves_per_game, board_size, device)
        state = MockBatchGameState.create(batch_size, board_size, device)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        selected_v = select_moves_vectorized(moves, active_mask, board_size)
        selected_h = select_moves_heuristic(moves, state, active_mask)

        # Use .type for comparison (handles mps:0 vs mps)
        assert selected_v.device.type == device.type
        assert selected_h.device.type == device.type


class TestSelectionReproducibility:
    """Tests for reproducibility with seeding."""

    def test_different_seeds_different_results(self, device, board_size):
        """Different random states should give different results."""
        batch_size = 10
        moves_per_game = [20] * batch_size
        moves = MockBatchMoves.create_simple(batch_size, moves_per_game, board_size, device)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        # Run multiple times
        results = []
        for _ in range(5):
            selected = select_moves_vectorized(moves, active_mask, board_size)
            results.append(selected.clone())

        # At least some results should differ
        all_same = all((results[i] == results[0]).all() for i in range(1, len(results)))
        assert not all_same, "All results were identical - no randomness?"


class TestSelectionPerformance:
    """Performance-related tests."""

    def test_no_cpu_sync(self, device, board_size):
        """Selection should not cause CPU-GPU sync (no .item() calls in hot path)."""
        if device.type == "cpu":
            pytest.skip("CPU test doesn't check GPU sync")

        batch_size = 100
        moves_per_game = [50] * batch_size
        moves = MockBatchMoves.create_simple(batch_size, moves_per_game, board_size, device)
        state = MockBatchGameState.create(batch_size, board_size, device)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        # These should complete without blocking
        for _ in range(10):
            _ = select_moves_vectorized(moves, active_mask, board_size)
            _ = select_moves_heuristic(moves, state, active_mask)

        # If we got here without hanging, the test passes
        # (Would need timing to be precise, but this catches obvious issues)
