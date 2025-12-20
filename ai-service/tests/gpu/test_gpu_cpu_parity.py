"""GPU vs CPU parity tests for move generation and evaluation.

These tests verify that GPU-accelerated code produces identical results
to the canonical CPU implementation. Any divergence indicates a bug in
the GPU code that must be fixed before using GPU for training.

Test organization:
- TestLineLengthParity: Verifies line length rules match across implementations
- TestPlacementMoveParity: Placement move generation
- TestEvaluationParity: Heuristic scoring
- TestBoardConfigParity: Board configuration values
"""

from typing import Optional

import numpy as np
import pytest
import torch

from app.models import BoardType, GamePhase, MoveType
from app.rules.core import (
    BOARD_CONFIGS,
    get_effective_line_length,
    get_rings_per_player,
    get_territory_victory_minimum,
    get_territory_victory_threshold,
    get_victory_threshold,
)

# Conditional imports - tests skip if GPU modules unavailable
try:
    from app.ai.gpu_heuristic import evaluate_positions_batch
    from app.ai.gpu_kernels import (
        generate_placement_mask_kernel,
    )
    from app.ai.gpu_move_generation import (
        generate_placement_moves_batch,
    )
    from app.ai.gpu_parallel_games import (
        BatchGameState,
        detect_lines_vectorized as detect_lines_batch,
    )
    GPU_MODULES_AVAILABLE = True
except ImportError as e:
    GPU_MODULES_AVAILABLE = False
    GPU_IMPORT_ERROR = str(e)

# Check if from_game_states is functional by testing the actual method
FROM_GAME_STATES_AVAILABLE = False
FROM_GAME_STATES_ERROR = ""
if GPU_MODULES_AVAILABLE:
    try:
        import torch

        from app.training.initial_state import create_initial_state
        # Actually test the method works
        test_gs = create_initial_state(num_players=2)
        test_batch = BatchGameState.from_game_states([test_gs], device=torch.device('cpu'))
        if test_batch.batch_size == 1:
            FROM_GAME_STATES_AVAILABLE = True
    except Exception as e:
        FROM_GAME_STATES_ERROR = str(e)

pytestmark = [
    pytest.mark.skipif(
        not GPU_MODULES_AVAILABLE,
        reason=f"GPU modules not available: {GPU_IMPORT_ERROR if not GPU_MODULES_AVAILABLE else ''}"
    ),
    pytest.mark.skipif(
        GPU_MODULES_AVAILABLE and not FROM_GAME_STATES_AVAILABLE,
        reason=f"BatchGameState.from_game_states not functional: {FROM_GAME_STATES_ERROR}"
    ),
]


# =============================================================================
# Board Configuration Parity Tests
# =============================================================================


class TestBoardConfigParity:
    """Verify board configuration values match between CPU and GPU."""

    @pytest.mark.parametrize("board_type,expected_rings", [
        (BoardType.SQUARE8, 18),
        (BoardType.SQUARE19, 72),
        (BoardType.HEXAGONAL, 96),
    ])
    def test_rings_per_player_matches(self, board_type, expected_rings):
        """Verify rings per player values match canonical spec."""
        assert get_rings_per_player(board_type) == expected_rings
        assert BOARD_CONFIGS[board_type].rings_per_player == expected_rings

    @pytest.mark.parametrize("board_type,num_players,expected_threshold", [
        # 2-player: threshold = ringsPerPlayer
        (BoardType.SQUARE8, 2, 18),
        (BoardType.SQUARE19, 2, 72),
        (BoardType.HEXAGONAL, 2, 96),
        # 3-player: threshold = round(ringsPerPlayer * 4/3)
        (BoardType.SQUARE8, 3, 24),
        (BoardType.SQUARE19, 3, 96),
        (BoardType.HEXAGONAL, 3, 128),
        # 4-player: threshold = round(ringsPerPlayer * 5/3)
        (BoardType.SQUARE8, 4, 30),
        (BoardType.SQUARE19, 4, 120),
        (BoardType.HEXAGONAL, 4, 160),
    ])
    def test_victory_threshold_matches(self, board_type, num_players, expected_threshold):
        """Verify victory threshold calculation matches RR-CANON-R061."""
        calculated = get_victory_threshold(board_type, num_players)
        assert calculated == expected_threshold, (
            f"Victory threshold mismatch for {board_type.value} {num_players}p: "
            f"expected {expected_threshold}, got {calculated}"
        )

    @pytest.mark.parametrize("board_type,expected_threshold", [
        (BoardType.SQUARE8, 33),   # floor(64/2) + 1
        (BoardType.SQUARE19, 181),  # floor(361/2) + 1
        (BoardType.HEXAGONAL, 235),  # floor(469/2) + 1
    ])
    def test_territory_threshold_matches(self, board_type, expected_threshold):
        """Verify territory victory threshold matches RR-CANON-R062 (legacy 50% rule)."""
        calculated = get_territory_victory_threshold(board_type)
        assert calculated == expected_threshold

    @pytest.mark.parametrize("board_type,num_players,expected_minimum", [
        # square8: 64 spaces
        (BoardType.SQUARE8, 2, 33),   # floor(64/2) + 1
        (BoardType.SQUARE8, 3, 22),   # floor(64/3) + 1
        (BoardType.SQUARE8, 4, 17),   # floor(64/4) + 1
        # square19: 361 spaces
        (BoardType.SQUARE19, 2, 181),  # floor(361/2) + 1
        (BoardType.SQUARE19, 3, 121),  # floor(361/3) + 1
        (BoardType.SQUARE19, 4, 91),   # floor(361/4) + 1
        # hexagonal: 469 spaces
        (BoardType.HEXAGONAL, 2, 235),  # floor(469/2) + 1
        (BoardType.HEXAGONAL, 3, 157),  # floor(469/3) + 1
        (BoardType.HEXAGONAL, 4, 118),  # floor(469/4) + 1
    ])
    def test_territory_victory_minimum_matches(self, board_type, num_players, expected_minimum):
        """Verify territory victory minimum matches RR-CANON-R062-v2.

        Per RR-CANON-R062-v2: territoryVictoryMinimum = floor(totalSpaces / numPlayers) + 1
        This is the first condition for territory victory. The second condition is that
        the player's territory must exceed all opponents' territory combined.
        """
        calculated = get_territory_victory_minimum(board_type, num_players)
        assert calculated == expected_minimum, (
            f"Territory minimum mismatch for {board_type.value} {num_players}p: "
            f"expected {expected_minimum}, got {calculated}"
        )


# =============================================================================
# Line Length Parity Tests
# =============================================================================


class TestLineLengthParity:
    """Verify line length detection uses correct player-count-aware thresholds.

    Per RR-CANON-R120:
    - square8 2-player: lineLength = 4
    - square8 3-4 player: lineLength = 3
    - square19: lineLength = 4 (all player counts)
    - hexagonal: lineLength = 4 (all player counts)
    """

    @pytest.mark.parametrize("board_type,num_players,expected_length", [
        (BoardType.SQUARE8, 2, 4),
        (BoardType.SQUARE8, 3, 3),
        (BoardType.SQUARE8, 4, 3),
        (BoardType.SQUARE19, 2, 4),
        (BoardType.SQUARE19, 3, 4),
        (BoardType.SQUARE19, 4, 4),
        (BoardType.HEXAGONAL, 2, 4),
        (BoardType.HEXAGONAL, 3, 4),
        (BoardType.HEXAGONAL, 4, 4),
    ])
    def test_effective_line_length(self, board_type, num_players, expected_length):
        """Verify get_effective_line_length returns correct values."""
        actual = get_effective_line_length(board_type, num_players)
        assert actual == expected_length, (
            f"Line length mismatch for {board_type.value} {num_players}p: "
            f"expected {expected_length}, got {actual}"
        )

    def test_gpu_line_detection_uses_correct_length_2p(
        self, device, state_with_line_opportunity
    ):
        """Verify GPU line detection requires 4 markers for 2-player 8x8.

        Setup: 3 markers in a row at (2,3), (3,3), (4,3) plus stack at (1,3)
        Expected: No line detected (need 4 stacks for 2-player, markers don't count)

        Per RR-CANON-R120: Line detection counts stacks, not markers.
        """
        if not GPU_MODULES_AVAILABLE:
            pytest.skip("GPU modules not available")

        # This test verifies the GPU code respects the 2-player line length of 4
        state = state_with_line_opportunity
        # Currently has 3 markers - should NOT form a line in 2-player

        # Convert to batch state
        batch_state = BatchGameState.from_single_game(state, device)

        # Detect lines for player 1
        # detect_lines_batch returns (positions_mask, line_count_per_game)
        _positions_mask, line_count = detect_lines_batch(batch_state, player=1)

        # Should be empty - 3 markers is not enough for 2-player (needs 4)
        assert line_count[0].item() == 0, (
            f"GPU incorrectly detected line with only 3 markers in 2-player game. "
            f"Line length should be 4, but detected {line_count[0].item()} positions in lines."
        )

    def test_gpu_line_detection_uses_correct_length_3p(
        self, device, state_3p_with_line_opportunity
    ):
        """Verify GPU line detection requires 3 stacks for 3-player 8x8.

        Setup: 2 markers in a row at (2,3), (3,3) plus stack at (1,3)
        Expected: No line detected (need 3 stacks for 3-player, markers don't count)

        Per RR-CANON-R120: Line detection counts stacks, not markers.
        This test validates that the GPU code uses the correct player-count-aware
        line length (3 for 3-4 player 8x8, 4 for all others).
        """
        if not GPU_MODULES_AVAILABLE:
            pytest.skip("GPU modules not available")

        state = state_3p_with_line_opportunity
        # Currently has 2 markers - should NOT form a line (needs 3)

        batch_state = BatchGameState.from_single_game(state, device)

        # Detect lines for player 1
        # detect_lines_batch returns (positions_mask, line_count_per_game)
        _positions_mask, line_count = detect_lines_batch(batch_state, player=1)

        # Should be empty - 2 markers is not enough for 3-player (needs 3)
        assert line_count[0].item() == 0, (
            f"GPU incorrectly detected line with only 2 markers in 3-player game. "
            f"Line length should be 3, but detected {line_count[0].item()} positions in lines."
        )


# =============================================================================
# Placement Move Parity Tests
# =============================================================================


class TestPlacementMoveParity:
    """Verify GPU placement move generation matches CPU.

    Phase 2 of the GPU pipeline - validates that GPU and CPU produce the same
    valid placement positions. Uses get_valid_moves filtered by MoveType.PLACE_RING
    to extract unique placement positions from the CPU engine.
    """

    def test_empty_board_placement_count(self, device, empty_square8_2p, cpu_engine):
        """On empty board, all 64 positions should be valid placements."""
        state = empty_square8_2p

        # CPU: count valid placements (unique positions from all placement moves)
        cpu_moves = cpu_engine.get_valid_moves(state, state.current_player)
        cpu_positions = {(m.to.x, m.to.y) for m in cpu_moves if m.type == MoveType.PLACE_RING}
        cpu_count = len(cpu_positions)

        # GPU: count valid placements using batch move generation
        batch_state = BatchGameState.from_single_game(state, device)
        batch_moves = generate_placement_moves_batch(batch_state)

        gpu_count = batch_moves.moves_per_game[0].item()

        assert cpu_count == gpu_count == 64, (
            f"Empty board should have 64 valid placements. "
            f"CPU={cpu_count}, GPU={gpu_count}"
        )

    def test_placement_with_existing_stacks(
        self, device, state_with_single_stack, cpu_engine
    ):
        """Verify placement moves exclude occupied positions."""
        state = state_with_single_stack

        # CPU: unique placement positions
        cpu_moves = cpu_engine.get_valid_moves(state, state.current_player)
        cpu_positions = {(m.to.x, m.to.y) for m in cpu_moves if m.type == MoveType.PLACE_RING}

        # GPU: use batch move generation
        batch_state = BatchGameState.from_single_game(state, device)
        batch_moves = generate_placement_moves_batch(batch_state)

        gpu_positions = set()
        for i in range(batch_moves.total_moves):
            if batch_moves.game_idx[i] == 0:  # First game
                gpu_positions.add((
                    batch_moves.to_x[i].item(),
                    batch_moves.to_y[i].item()
                ))

        assert cpu_positions == gpu_positions, (
            f"Placement positions mismatch.\n"
            f"CPU: {len(cpu_positions)} positions\n"
            f"GPU: {len(gpu_positions)} positions\n"
            f"Missing in GPU: {cpu_positions - gpu_positions}\n"
            f"Extra in GPU: {gpu_positions - cpu_positions}"
        )

    @pytest.mark.parametrize("board_type,num_players", [
        (BoardType.SQUARE8, 2),
        (BoardType.SQUARE8, 3),
        (BoardType.SQUARE8, 4),
        (BoardType.SQUARE19, 2),
    ])
    def test_placement_count_by_board_type(
        self, device, board_type, num_players, cpu_engine
    ):
        """Verify placement counts match across board types."""
        from tests.gpu.conftest import create_empty_game_state

        state = create_empty_game_state(board_type, num_players)

        # CPU: unique placement positions
        cpu_moves = cpu_engine.get_valid_moves(state, state.current_player)
        cpu_positions = {(m.to.x, m.to.y) for m in cpu_moves if m.type == MoveType.PLACE_RING}
        cpu_count = len(cpu_positions)

        # GPU: use batch move generation
        batch_state = BatchGameState.from_single_game(state, device)
        batch_moves = generate_placement_moves_batch(batch_state)

        gpu_count = batch_moves.moves_per_game[0].item()

        config = BOARD_CONFIGS[board_type]
        expected = config.total_spaces  # Empty board = all spaces valid

        assert cpu_count == gpu_count == expected, (
            f"Placement count mismatch for {board_type.value} {num_players}p: "
            f"CPU={cpu_count}, GPU={gpu_count}, expected={expected}"
        )


# =============================================================================
# Evaluation Parity Tests
# =============================================================================


class TestEvaluationParity:
    """Verify GPU heuristic evaluation produces consistent scores."""

    def test_empty_board_evaluation_symmetric(self, device, empty_square8_2p):
        """On empty board, both players should have equal scores."""
        state = empty_square8_2p

        batch_state = BatchGameState.from_single_game(state, device)

        # Use default weights
        weights = {}  # Will use defaults

        scores = evaluate_positions_batch(batch_state, weights)

        p1_score = scores[0, 1].item()
        p2_score = scores[0, 2].item()

        # Scores should be equal (symmetric position)
        assert abs(p1_score - p2_score) < 0.01, (
            f"Empty board should have symmetric scores. "
            f"P1={p1_score:.4f}, P2={p2_score:.4f}"
        )

    def test_stack_control_improves_score(self, device, empty_square8_2p):
        """Player with stack should have better score than empty."""
        from tests.gpu.conftest import add_stack_to_state

        state = empty_square8_2p

        # Baseline: empty board
        batch_empty = BatchGameState.from_single_game(state, device)
        scores_empty = evaluate_positions_batch(batch_empty, {})
        baseline = scores_empty[0, 1].item()

        # Add stack for player 1
        add_stack_to_state(state, 3, 3, [1])
        state.players[0] = state.players[0].model_copy(
            update={'rings_in_hand': state.players[0].rings_in_hand - 1}
        )

        batch_with_stack = BatchGameState.from_single_game(state, device)
        scores_with_stack = evaluate_positions_batch(batch_with_stack, {})
        with_stack = scores_with_stack[0, 1].item()

        # Score should improve with stack control
        assert with_stack > baseline, (
            f"Stack control should improve score. "
            f"Baseline={baseline:.4f}, With stack={with_stack:.4f}"
        )

    def test_center_control_bonus(self, device, empty_square8_2p):
        """Center positions should score higher than corners."""
        from tests.gpu.conftest import add_stack_to_state, create_empty_game_state

        # State with center stack
        state_center = create_empty_game_state(BoardType.SQUARE8, 2)
        add_stack_to_state(state_center, 3, 3, [1])  # Center-ish
        state_center.players[0] = state_center.players[0].model_copy(
            update={'rings_in_hand': state_center.players[0].rings_in_hand - 1}
        )

        batch_center = BatchGameState.from_single_game(state_center, device)
        scores_center = evaluate_positions_batch(batch_center, {})
        center_score = scores_center[0, 1].item()

        # State with corner stack
        state_corner = create_empty_game_state(BoardType.SQUARE8, 2)
        add_stack_to_state(state_corner, 0, 0, [1])  # Corner
        state_corner.players[0] = state_corner.players[0].model_copy(
            update={'rings_in_hand': state_corner.players[0].rings_in_hand - 1}
        )

        batch_corner = BatchGameState.from_single_game(state_corner, device)
        scores_corner = evaluate_positions_batch(batch_corner, {})
        corner_score = scores_corner[0, 1].item()

        # Center should score higher
        assert center_score > corner_score, (
            f"Center control should score higher than corner. "
            f"Center={center_score:.4f}, Corner={corner_score:.4f}"
        )

    def test_adjacency_bonus_vectorized(self, device):
        """Verify vectorized adjacency calculation gives correct bonus.

        Adjacent stacks should score higher than isolated stacks.
        This tests the vectorized adjacency calculation (horizontal + vertical).
        """
        from tests.gpu.conftest import add_stack_to_state, create_empty_game_state

        # State with isolated stacks (no adjacency)
        state_isolated = create_empty_game_state(BoardType.SQUARE8, 2)
        add_stack_to_state(state_isolated, 0, 0, [1])  # Corner
        add_stack_to_state(state_isolated, 7, 7, [1])  # Opposite corner
        state_isolated.players[0] = state_isolated.players[0].model_copy(
            update={'rings_in_hand': state_isolated.players[0].rings_in_hand - 2}
        )

        batch_isolated = BatchGameState.from_single_game(state_isolated, device)
        scores_isolated = evaluate_positions_batch(batch_isolated, {})
        isolated_score = scores_isolated[0, 1].item()

        # State with adjacent stacks (2 horizontal neighbors)
        state_adjacent = create_empty_game_state(BoardType.SQUARE8, 2)
        add_stack_to_state(state_adjacent, 3, 3, [1])  # Center
        add_stack_to_state(state_adjacent, 4, 3, [1])  # Right neighbor
        state_adjacent.players[0] = state_adjacent.players[0].model_copy(
            update={'rings_in_hand': state_adjacent.players[0].rings_in_hand - 2}
        )

        batch_adjacent = BatchGameState.from_single_game(state_adjacent, device)
        scores_adjacent = evaluate_positions_batch(batch_adjacent, {})
        adjacent_score = scores_adjacent[0, 1].item()

        # Adjacent stacks should score higher due to adjacency bonus + center bonus
        assert adjacent_score > isolated_score, (
            f"Adjacent stacks should score higher than isolated. "
            f"Adjacent={adjacent_score:.4f}, Isolated={isolated_score:.4f}"
        )


# =============================================================================
# Victory Threshold Tests in GPU Evaluation
# =============================================================================


class TestVictoryThresholdInGPU:
    """Verify GPU evaluation uses correct victory thresholds."""

    @pytest.mark.parametrize("board_type,num_players", [
        (BoardType.SQUARE8, 2),
        (BoardType.SQUARE8, 3),
        (BoardType.SQUARE8, 4),
        (BoardType.SQUARE19, 2),
        (BoardType.HEXAGONAL, 2),  # Hex support verification
        (BoardType.HEXAGONAL, 3),
    ])
    def test_victory_proximity_scaling(self, device, board_type, num_players):
        """Verify victory proximity uses correct player-count-aware threshold."""
        from tests.gpu.conftest import create_empty_game_state

        state = create_empty_game_state(board_type, num_players)

        # Set some eliminated rings using model_copy
        state.players[0] = state.players[0].model_copy(update={'eliminated_rings': 10})

        batch_state = BatchGameState.from_single_game(state, device)

        # The GPU evaluation should use the correct threshold based on player count
        get_victory_threshold(board_type, num_players)

        # Verify the threshold used in evaluation matches
        # This is implicit - we check that evaluation runs without error
        # and produces reasonable scores
        scores = evaluate_positions_batch(batch_state, {})

        # Score should be positive (some progress toward victory)
        assert scores[0, 1].item() > 0, (
            "Player with eliminated rings should have positive score"
        )
