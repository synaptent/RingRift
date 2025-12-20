"""Tests for GPU batch game state.

Tests the BatchGameState class extracted from gpu_parallel_games.py.
"""

import pytest
import torch

from app.ai.gpu_batch_state import BatchGameState
from app.ai.gpu_game_types import GamePhase, GameStatus, MoveType

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


# =============================================================================
# Tests for BatchGameState.create_batch
# =============================================================================


class TestCreateBatch:
    """Tests for the create_batch factory method."""

    def test_creates_correct_tensor_shapes(self, device):
        """Tensor shapes should match batch_size, board_size, num_players."""
        batch_size = 8
        board_size = 8
        num_players = 2

        state = BatchGameState.create_batch(
            batch_size=batch_size,
            board_size=board_size,
            num_players=num_players,
            device=device,
        )

        # Board tensors
        assert state.stack_owner.shape == (batch_size, board_size, board_size)
        assert state.stack_height.shape == (batch_size, board_size, board_size)
        assert state.cap_height.shape == (batch_size, board_size, board_size)
        assert state.marker_owner.shape == (batch_size, board_size, board_size)
        assert state.territory_owner.shape == (batch_size, board_size, board_size)
        assert state.is_collapsed.shape == (batch_size, board_size, board_size)

        # Player tensors (num_players + 1 for 1-indexing)
        assert state.rings_in_hand.shape == (batch_size, num_players + 1)
        assert state.territory_count.shape == (batch_size, num_players + 1)
        assert state.is_eliminated.shape == (batch_size, num_players + 1)
        assert state.eliminated_rings.shape == (batch_size, num_players + 1)
        assert state.buried_rings.shape == (batch_size, num_players + 1)
        assert state.rings_caused_eliminated.shape == (batch_size, num_players + 1)

        # Game metadata
        assert state.current_player.shape == (batch_size,)
        assert state.current_phase.shape == (batch_size,)
        assert state.move_count.shape == (batch_size,)
        assert state.game_status.shape == (batch_size,)
        assert state.winner.shape == (batch_size,)

    def test_initializes_rings_for_small_board(self, device):
        """Small boards (<=8) should get 18 rings per player."""
        state = BatchGameState.create_batch(
            batch_size=4,
            board_size=8,
            num_players=2,
            device=device,
        )

        # Players 1 and 2 should have 18 rings
        assert (state.rings_in_hand[:, 1] == 18).all()
        assert (state.rings_in_hand[:, 2] == 18).all()
        # Index 0 should be 0 (unused)
        assert (state.rings_in_hand[:, 0] == 0).all()

    def test_initializes_rings_for_medium_board(self, device):
        """Medium boards (9-13) should get 24 rings per player."""
        state = BatchGameState.create_batch(
            batch_size=4,
            board_size=13,
            num_players=2,
            device=device,
        )

        assert (state.rings_in_hand[:, 1] == 24).all()
        assert (state.rings_in_hand[:, 2] == 24).all()

    def test_initializes_rings_for_large_board(self, device):
        """Large boards (>13) should get 36 rings per player."""
        state = BatchGameState.create_batch(
            batch_size=4,
            board_size=19,
            num_players=2,
            device=device,
        )

        assert (state.rings_in_hand[:, 1] == 36).all()
        assert (state.rings_in_hand[:, 2] == 36).all()

    def test_custom_rings_per_player(self, device):
        """Should respect custom rings_per_player parameter."""
        state = BatchGameState.create_batch(
            batch_size=4,
            board_size=8,
            num_players=2,
            device=device,
            rings_per_player=25,
        )

        assert (state.rings_in_hand[:, 1] == 25).all()
        assert (state.rings_in_hand[:, 2] == 25).all()

    def test_initializes_game_metadata(self, device):
        """Game metadata should be correctly initialized."""
        state = BatchGameState.create_batch(
            batch_size=4,
            board_size=8,
            num_players=2,
            device=device,
        )

        # Current player starts at 1
        assert (state.current_player == 1).all()

        # Phase starts at RING_PLACEMENT
        assert (state.current_phase == GamePhase.RING_PLACEMENT).all()

        # Move count starts at 0
        assert (state.move_count == 0).all()

        # Status starts as ACTIVE
        assert (state.game_status == GameStatus.ACTIVE).all()

        # No winner yet
        assert (state.winner == 0).all()

    def test_movement_constraints_initialized(self, device):
        """Movement constraints should be -1 (no constraint)."""
        state = BatchGameState.create_batch(
            batch_size=4,
            board_size=8,
            num_players=2,
            device=device,
        )

        assert (state.must_move_from_y == -1).all()
        assert (state.must_move_from_x == -1).all()

    def test_lps_tracking_initialized(self, device):
        """LPS tracking tensors should be zeroed."""
        state = BatchGameState.create_batch(
            batch_size=4,
            board_size=8,
            num_players=2,
            device=device,
        )

        assert (state.lps_round_index == 0).all()
        assert (state.lps_current_round_first_player == 0).all()
        assert (not state.lps_current_round_seen_mask).all()
        assert (state.lps_consecutive_exclusive_rounds == 0).all()

    def test_move_history_initialized(self, device):
        """Move history should be -1 (unused slots)."""
        state = BatchGameState.create_batch(
            batch_size=4,
            board_size=8,
            num_players=2,
            device=device,
            max_history_moves=100,
        )

        assert state.move_history.shape == (4, 100, 6)
        assert (state.move_history == -1).all()

    def test_hexagonal_board_marks_oob(self, device):
        """Hexagonal board should mark out-of-bounds cells as collapsed."""
        state = BatchGameState.create_batch(
            batch_size=2,
            board_size=25,
            num_players=2,
            device=device,
            board_type="hexagonal",
        )

        # Corner cells should be collapsed
        assert state.is_collapsed[0, 0, 0].item()
        assert state.is_collapsed[0, 24, 24].item()
        assert state.is_collapsed[0, 0, 24].item()
        assert state.is_collapsed[0, 24, 0].item()

        # Center should not be collapsed
        assert not state.is_collapsed[0, 12, 12].item()

    def test_four_player_game(self, device):
        """Should correctly initialize 4-player games."""
        state = BatchGameState.create_batch(
            batch_size=4,
            board_size=8,
            num_players=4,
            device=device,
        )

        # All 4 players should have rings
        for p in range(1, 5):
            assert (state.rings_in_hand[:, p] == 18).all()

        # Player state tensors should have 5 slots (0 unused, 1-4 for players)
        assert state.rings_in_hand.shape == (4, 5)
        assert state.is_eliminated.shape == (4, 5)

    def test_tensors_on_correct_device(self, device):
        """All tensors should be on the specified device."""
        state = BatchGameState.create_batch(
            batch_size=4,
            board_size=8,
            num_players=2,
            device=device,
        )

        # Use .type for comparison (handles mps:0 vs mps)
        assert state.stack_owner.device.type == device.type
        assert state.stack_height.device.type == device.type
        assert state.rings_in_hand.device.type == device.type
        assert state.current_player.device.type == device.type
        assert state.move_history.device.type == device.type


# =============================================================================
# Tests for get_active_mask and count_active
# =============================================================================


class TestActiveMask:
    """Tests for active game tracking."""

    def test_all_active_initially(self, device):
        """All games should be active initially."""
        state = BatchGameState.create_batch(
            batch_size=8,
            board_size=8,
            num_players=2,
            device=device,
        )

        mask = state.get_active_mask()
        assert mask.shape == (8,)
        assert mask.all()
        assert state.count_active() == 8

    def test_completed_games_not_active(self, device):
        """Completed games should not be in active mask."""
        state = BatchGameState.create_batch(
            batch_size=8,
            board_size=8,
            num_players=2,
            device=device,
        )

        # Mark some games as completed
        state.game_status[0] = GameStatus.COMPLETED
        state.game_status[3] = GameStatus.COMPLETED
        state.game_status[7] = GameStatus.COMPLETED

        mask = state.get_active_mask()
        assert not mask[0].item()
        assert mask[1].item()
        assert mask[2].item()
        assert not mask[3].item()
        assert not mask[7].item()

        assert state.count_active() == 5

    def test_all_completed(self, device):
        """When all games completed, count should be 0."""
        state = BatchGameState.create_batch(
            batch_size=4,
            board_size=8,
            num_players=2,
            device=device,
        )

        state.game_status[:] = GameStatus.COMPLETED

        assert state.count_active() == 0
        assert not state.get_active_mask().any()


# =============================================================================
# Tests for to_feature_tensor
# =============================================================================


class TestToFeatureTensor:
    """Tests for neural network feature tensor generation."""

    def test_output_shape(self, device):
        """Output should have correct shape."""
        batch_size = 4
        board_size = 8

        state = BatchGameState.create_batch(
            batch_size=batch_size,
            board_size=board_size,
            num_players=2,
            device=device,
        )

        features = state.to_feature_tensor(history_length=4)

        # Should be (batch, channels, height, width)
        assert features.shape[0] == batch_size
        assert features.shape[2] == board_size
        assert features.shape[3] == board_size
        # Channels: 5 + 5 + 5 + 5 + 1 + 4*5 = 41
        expected_channels = 21 + 4 * 5  # base + history
        assert features.shape[1] == expected_channels

    def test_stack_ownership_encoding(self, device):
        """Stack ownership should be one-hot encoded."""
        state = BatchGameState.create_batch(
            batch_size=2,
            board_size=8,
            num_players=2,
            device=device,
        )

        # Place a stack
        state.stack_owner[0, 3, 3] = 1
        state.stack_height[0, 3, 3] = 2

        features = state.to_feature_tensor(history_length=0)

        # Channel 0 should be 0 for empty cells (stack_owner == 0)
        # Channel 1 should be 1 where stack_owner == 1
        assert features[0, 1, 3, 3].item() == 1.0
        assert features[0, 0, 3, 3].item() == 0.0  # Not empty
        assert features[0, 2, 3, 3].item() == 0.0  # Not player 2

    def test_collapsed_cells_encoded(self, device):
        """Collapsed cells should be encoded in channel 20."""
        state = BatchGameState.create_batch(
            batch_size=2,
            board_size=8,
            num_players=2,
            device=device,
        )

        state.is_collapsed[0, 2, 4] = True

        features = state.to_feature_tensor(history_length=0)

        assert features[0, 20, 2, 4].item() == 1.0
        assert features[0, 20, 0, 0].item() == 0.0

    def test_normalized_heights(self, device):
        """Stack heights should be normalized to 0-1."""
        state = BatchGameState.create_batch(
            batch_size=2,
            board_size=8,
            num_players=2,
            device=device,
        )

        state.stack_height[0, 3, 3] = 5  # Max height
        state.stack_height[0, 4, 4] = 2

        features = state.to_feature_tensor(history_length=0)

        # Channel 5 is stack height normalized
        assert abs(features[0, 5, 3, 3].item() - 1.0) < 0.01  # 5/5 = 1.0
        assert abs(features[0, 5, 4, 4].item() - 0.4) < 0.01  # 2/5 = 0.4

    def test_output_on_correct_device(self, device):
        """Features should be on same device as state."""
        state = BatchGameState.create_batch(
            batch_size=2,
            board_size=8,
            num_players=2,
            device=device,
        )

        features = state.to_feature_tensor()

        # Use .type for comparison (handles mps:0 vs mps)
        assert features.device.type == device.type
        assert features.dtype == torch.float32


# =============================================================================
# Tests for extract_move_history
# =============================================================================


class TestExtractMoveHistory:
    """Tests for move history extraction."""

    def test_empty_history(self, device):
        """Empty history should return empty list."""
        state = BatchGameState.create_batch(
            batch_size=2,
            board_size=8,
            num_players=2,
            device=device,
        )

        moves = state.extract_move_history(0)
        assert moves == []

    def test_extracts_recorded_moves(self, device):
        """Should extract moves that were recorded."""
        state = BatchGameState.create_batch(
            batch_size=2,
            board_size=8,
            num_players=2,
            device=device,
            max_history_moves=100,
        )

        # Record some moves
        state.move_history[0, 0, 0] = MoveType.PLACEMENT
        state.move_history[0, 0, 1] = 1  # player
        state.move_history[0, 0, 2] = 3  # from_y
        state.move_history[0, 0, 3] = 3  # from_x
        state.move_history[0, 0, 4] = 3  # to_y
        state.move_history[0, 0, 5] = 3  # to_x

        state.move_history[0, 1, 0] = MoveType.MOVEMENT
        state.move_history[0, 1, 1] = 2  # player
        state.move_history[0, 1, 2] = 3  # from_y
        state.move_history[0, 1, 3] = 3  # from_x
        state.move_history[0, 1, 4] = 4  # to_y
        state.move_history[0, 1, 5] = 4  # to_x

        moves = state.extract_move_history(0)

        assert len(moves) == 2
        assert moves[0]["move_type"] == "PLACEMENT"
        assert moves[0]["player"] == 1
        assert moves[0]["from_pos"] == (3, 3)
        assert moves[0]["to_pos"] == (3, 3)

        assert moves[1]["move_type"] == "MOVEMENT"
        assert moves[1]["player"] == 2
        assert moves[1]["from_pos"] == (3, 3)
        assert moves[1]["to_pos"] == (4, 4)

    def test_stops_at_unused_slots(self, device):
        """Should stop extracting at first unused slot."""
        state = BatchGameState.create_batch(
            batch_size=2,
            board_size=8,
            num_players=2,
            device=device,
            max_history_moves=100,
        )

        # Record one move
        state.move_history[0, 0, 0] = MoveType.PLACEMENT
        state.move_history[0, 0, 1] = 1
        state.move_history[0, 0, 2] = 3
        state.move_history[0, 0, 3] = 3
        state.move_history[0, 0, 4] = 3
        state.move_history[0, 0, 5] = 3
        # Slot 1 is still -1 (unused)

        moves = state.extract_move_history(0)
        assert len(moves) == 1

    def test_different_games_independent(self, device):
        """Each game should have independent history."""
        state = BatchGameState.create_batch(
            batch_size=2,
            board_size=8,
            num_players=2,
            device=device,
        )

        # Record moves in game 0
        state.move_history[0, 0, 0] = MoveType.PLACEMENT
        state.move_history[0, 0, 1] = 1
        state.move_history[0, 0, 2] = 0
        state.move_history[0, 0, 3] = 0
        state.move_history[0, 0, 4] = 0
        state.move_history[0, 0, 5] = 0

        # Game 1 has no moves

        moves_0 = state.extract_move_history(0)
        moves_1 = state.extract_move_history(1)

        assert len(moves_0) == 1
        assert len(moves_1) == 0


# =============================================================================
# Tests for derive_victory_type
# =============================================================================


class TestDeriveVictoryType:
    """Tests for victory type derivation."""

    def test_in_progress_game(self, device):
        """Games without a winner should return in_progress."""
        state = BatchGameState.create_batch(
            batch_size=2,
            board_size=8,
            num_players=2,
            device=device,
        )

        victory_type, tiebreaker = state.derive_victory_type(0, max_moves=500)
        assert victory_type == "in_progress"
        assert tiebreaker is None

    def test_territory_victory(self, device):
        """Should detect territory victory."""
        state = BatchGameState.create_batch(
            batch_size=2,
            board_size=8,
            num_players=2,
            device=device,
        )

        # Set winner and give them territory
        state.winner[0] = 1
        state.game_status[0] = GameStatus.COMPLETED
        # Territory threshold for 8x8 2-player is 33
        state.territory_count[0, 1] = 35

        victory_type, tiebreaker = state.derive_victory_type(0, max_moves=500)
        assert victory_type == "territory"
        assert tiebreaker is None

    def test_elimination_victory(self, device):
        """Should detect elimination victory."""
        state = BatchGameState.create_batch(
            batch_size=2,
            board_size=8,
            num_players=2,
            device=device,
        )

        state.winner[0] = 1
        state.game_status[0] = GameStatus.COMPLETED
        # Elimination threshold for 8x8 2-player is typically 18
        state.rings_caused_eliminated[0, 1] = 20

        victory_type, tiebreaker = state.derive_victory_type(0, max_moves=500)
        assert victory_type == "elimination"
        assert tiebreaker is None

    def test_lps_victory(self, device):
        """Should detect LPS victory."""
        state = BatchGameState.create_batch(
            batch_size=2,
            board_size=8,
            num_players=2,
            device=device,
            lps_rounds_required=3,
        )

        state.winner[0] = 1
        state.game_status[0] = GameStatus.COMPLETED
        state.lps_consecutive_exclusive_rounds[0] = 3
        state.lps_consecutive_exclusive_player[0] = 1

        victory_type, tiebreaker = state.derive_victory_type(0, max_moves=500)
        assert victory_type == "lps"
        assert tiebreaker is None

    def test_stalemate_detection(self, device):
        """Should detect stalemate when max moves reached."""
        state = BatchGameState.create_batch(
            batch_size=2,
            board_size=8,
            num_players=2,
            device=device,
        )

        state.winner[0] = 1
        state.game_status[0] = GameStatus.COMPLETED
        state.move_count[0] = 500  # Max moves

        victory_type, tiebreaker = state.derive_victory_type(0, max_moves=500)
        assert victory_type == "stalemate"
        assert tiebreaker is not None


# =============================================================================
# Tests for _determine_tiebreaker
# =============================================================================


class TestDetermineTiebreaker:
    """Tests for tiebreaker determination."""

    def test_territory_tiebreaker(self, device):
        """Should detect territory as tiebreaker."""
        state = BatchGameState.create_batch(
            batch_size=2,
            board_size=8,
            num_players=2,
            device=device,
        )

        state.winner[0] = 1
        state.territory_count[0, 1] = 5
        state.territory_count[0, 2] = 3

        tiebreaker = state._determine_tiebreaker(0)
        assert tiebreaker == "territory"

    def test_eliminations_tiebreaker(self, device):
        """Should detect eliminations as tiebreaker when territory tied."""
        state = BatchGameState.create_batch(
            batch_size=2,
            board_size=8,
            num_players=2,
            device=device,
        )

        state.winner[0] = 1
        # Territory tied
        state.territory_count[0, 1] = 5
        state.territory_count[0, 2] = 5
        # Eliminations differ
        state.rings_caused_eliminated[0, 1] = 10
        state.rings_caused_eliminated[0, 2] = 5

        tiebreaker = state._determine_tiebreaker(0)
        assert tiebreaker == "eliminations"

    def test_no_winner_returns_none(self, device):
        """Should return 'none' when no winner."""
        state = BatchGameState.create_batch(
            batch_size=2,
            board_size=8,
            num_players=2,
            device=device,
        )

        state.winner[0] = 0

        tiebreaker = state._determine_tiebreaker(0)
        assert tiebreaker == "none"


# =============================================================================
# Integration Tests
# =============================================================================


class TestBatchStateIntegration:
    """Integration tests for BatchGameState."""

    def test_create_and_modify_state(self, device):
        """Should be able to create and modify state."""
        state = BatchGameState.create_batch(
            batch_size=4,
            board_size=8,
            num_players=2,
            device=device,
        )

        # Place some stacks
        state.stack_owner[0, 3, 3] = 1
        state.stack_height[0, 3, 3] = 3
        state.cap_height[0, 3, 3] = 2

        state.stack_owner[1, 4, 4] = 2
        state.stack_height[1, 4, 4] = 2
        state.cap_height[1, 4, 4] = 2

        # Verify modifications
        assert state.stack_owner[0, 3, 3].item() == 1
        assert state.stack_height[0, 3, 3].item() == 3
        assert state.stack_owner[1, 4, 4].item() == 2

    def test_batch_operations(self, device):
        """Should support batch tensor operations."""
        state = BatchGameState.create_batch(
            batch_size=100,
            board_size=8,
            num_players=2,
            device=device,
        )

        # Should be able to do batch operations
        total_rings = state.rings_in_hand.sum(dim=1)
        assert total_rings.shape == (100,)

        # Active mask should work
        state.game_status[50:] = GameStatus.COMPLETED
        assert state.count_active() == 50

    def test_large_batch(self, device):
        """Should handle large batches."""
        state = BatchGameState.create_batch(
            batch_size=1000,
            board_size=8,
            num_players=2,
            device=device,
        )

        assert state.batch_size == 1000
        assert state.stack_owner.shape[0] == 1000
        assert state.count_active() == 1000

    def test_different_board_sizes(self, device):
        """Should handle different board sizes."""
        for board_size in [8, 13, 19]:
            state = BatchGameState.create_batch(
                batch_size=4,
                board_size=board_size,
                num_players=2,
                device=device,
            )

            assert state.board_size == board_size
            assert state.stack_owner.shape == (4, board_size, board_size)


class TestConfigAttributes:
    """Tests for configuration attributes."""

    def test_stores_config(self, device):
        """Should store configuration attributes."""
        state = BatchGameState.create_batch(
            batch_size=8,
            board_size=19,
            num_players=4,
            device=device,
            max_history_moves=200,
            lps_rounds_required=5,
        )

        assert state.batch_size == 8
        assert state.board_size == 19
        assert state.num_players == 4
        assert state.device == device
        assert state.max_history_moves == 200
        assert state.lps_rounds_required == 5

    def test_device_attribute(self, device):
        """Device should be correctly stored."""
        state = BatchGameState.create_batch(
            batch_size=4,
            board_size=8,
            num_players=2,
            device=device,
        )

        # Use .type for comparison (handles mps:0 vs mps)
        assert state.device.type == device.type
        assert state.stack_owner.device.type == device.type
