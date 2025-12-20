import torch

from app.ai.gpu_parallel_games import (
    BatchGameState,
    BatchMoves,
    apply_capture_moves_batch as apply_capture_moves_vectorized,
    apply_movement_moves_batch as apply_movement_moves_vectorized,
    apply_single_chain_capture,
    generate_chain_capture_moves_from_position,
)
from app.ai.gpu_game_types import MoveType


def _single_move_batch(
    device: torch.device,
    move_type: MoveType,
    from_y: int,
    from_x: int,
    to_y: int,
    to_x: int,
) -> BatchMoves:
    return BatchMoves(
        game_idx=torch.tensor([0], dtype=torch.int32, device=device),
        move_type=torch.tensor([int(move_type)], dtype=torch.int8, device=device),
        from_y=torch.tensor([from_y], dtype=torch.int32, device=device),
        from_x=torch.tensor([from_x], dtype=torch.int32, device=device),
        to_y=torch.tensor([to_y], dtype=torch.int32, device=device),
        to_x=torch.tensor([to_x], dtype=torch.int32, device=device),
        moves_per_game=torch.tensor([1], dtype=torch.int32, device=device),
        move_offsets=torch.tensor([0], dtype=torch.int32, device=device),
        total_moves=1,
        device=device,
    )


def test_movement_leaves_departure_marker_and_pays_landing_cost_on_any_marker(device):
    state = BatchGameState.create_batch(batch_size=1, board_size=8, num_players=2, device=device)
    state.current_player[0] = 1

    state.stack_owner[0, 0, 0] = 1
    state.stack_height[0, 0, 0] = 2
    state.cap_height[0, 0, 0] = 2

    # Landing on an opponent marker is legal and must remove the marker and
    # eliminate the top ring of the moving stack's cap.
    state.marker_owner[0, 0, 2] = 2

    moves = _single_move_batch(device, MoveType.MOVEMENT, from_y=0, from_x=0, to_y=0, to_x=2)
    selected = torch.tensor([0], dtype=torch.int64, device=device)

    apply_movement_moves_vectorized(state, selected, moves)

    assert int(state.marker_owner[0, 0, 0].item()) == 1
    assert int(state.stack_owner[0, 0, 0].item()) == 0
    assert int(state.stack_height[0, 0, 0].item()) == 0
    assert int(state.cap_height[0, 0, 0].item()) == 0

    assert int(state.marker_owner[0, 0, 2].item()) == 0
    assert int(state.stack_owner[0, 0, 2].item()) == 1
    assert int(state.stack_height[0, 0, 2].item()) == 1  # 2 - 1 landing cost
    assert int(state.cap_height[0, 0, 2].item()) == 1

    assert int(state.eliminated_rings[0, 1].item()) == 1
    assert int(state.rings_caused_eliminated[0, 1].item()) == 1


def test_movement_collapses_own_marker_on_path_increments_territory(device):
    state = BatchGameState.create_batch(batch_size=1, board_size=8, num_players=2, device=device)
    state.current_player[0] = 1

    state.stack_owner[0, 0, 0] = 1
    state.stack_height[0, 0, 0] = 3
    state.cap_height[0, 0, 0] = 3

    # Own marker on an intermediate cell should collapse to territory when crossed.
    state.marker_owner[0, 0, 1] = 1

    moves = _single_move_batch(device, MoveType.MOVEMENT, from_y=0, from_x=0, to_y=0, to_x=3)
    selected = torch.tensor([0], dtype=torch.int64, device=device)

    apply_movement_moves_vectorized(state, selected, moves)

    assert int(state.marker_owner[0, 0, 1].item()) == 0
    assert bool(state.is_collapsed[0, 0, 1].item()) is True
    assert int(state.territory_owner[0, 0, 1].item()) == 1
    assert int(state.territory_count[0, 1].item()) == 1


def test_capture_transfers_ring_to_attacker_and_updates_target(device):
    state = BatchGameState.create_batch(batch_size=1, board_size=8, num_players=2, device=device)
    state.current_player[0] = 1

    # Attacker stack at (0,0), target stack at (0,1), landing at (0,2)
    state.stack_owner[0, 0, 0] = 1
    state.stack_height[0, 0, 0] = 2
    state.cap_height[0, 0, 0] = 2

    state.stack_owner[0, 0, 1] = 2
    state.stack_height[0, 0, 1] = 2
    state.cap_height[0, 0, 1] = 2

    moves = _single_move_batch(device, MoveType.CAPTURE, from_y=0, from_x=0, to_y=0, to_x=2)
    selected = torch.tensor([0], dtype=torch.int64, device=device)

    apply_capture_moves_vectorized(state, selected, moves)

    # Departure marker placed
    assert int(state.marker_owner[0, 0, 0].item()) == 1
    assert int(state.stack_owner[0, 0, 0].item()) == 0

    # Target stack loses top ring (height 2 -> 1)
    assert int(state.stack_owner[0, 0, 1].item()) == 2
    assert int(state.stack_height[0, 0, 1].item()) == 1
    assert int(state.cap_height[0, 0, 1].item()) == 1

    # Attacker lands with +1 captured ring on bottom (height 2 -> 3)
    assert int(state.stack_owner[0, 0, 2].item()) == 1
    assert int(state.stack_height[0, 0, 2].item()) == 3
    assert int(state.cap_height[0, 0, 2].item()) == 2

    # Captured ring becomes "buried" for the captured ring's owner (player 2)
    assert int(state.buried_rings[0, 2].item()) == 1
    assert int(state.eliminated_rings[0, 1].item()) == 0
    assert int(state.rings_caused_eliminated[0, 1].item()) == 0


def test_capture_landing_marker_cost_eliminates_cap_ring(device):
    state = BatchGameState.create_batch(batch_size=1, board_size=8, num_players=2, device=device)
    state.current_player[0] = 1

    state.stack_owner[0, 0, 0] = 1
    state.stack_height[0, 0, 0] = 2
    state.cap_height[0, 0, 0] = 2

    state.stack_owner[0, 0, 1] = 2
    state.stack_height[0, 0, 1] = 2
    state.cap_height[0, 0, 1] = 2

    # Landing marker triggers cap-elimination cost.
    state.marker_owner[0, 0, 2] = 2

    moves = _single_move_batch(device, MoveType.CAPTURE, from_y=0, from_x=0, to_y=0, to_x=2)
    selected = torch.tensor([0], dtype=torch.int64, device=device)

    apply_capture_moves_vectorized(state, selected, moves)

    assert int(state.marker_owner[0, 0, 2].item()) == 0

    # Height: +1 captured ring, -1 landing cost => unchanged (2)
    assert int(state.stack_height[0, 0, 2].item()) == 2
    assert int(state.cap_height[0, 0, 2].item()) == 1

    assert int(state.eliminated_rings[0, 1].item()) == 1
    assert int(state.rings_caused_eliminated[0, 1].item()) == 1


def test_chain_capture_generation_returns_landing_positions_and_apply(device):
    state = BatchGameState.create_batch(batch_size=1, board_size=8, num_players=2, device=device)
    state.current_player[0] = 1

    state.stack_owner[0, 0, 0] = 1
    state.stack_height[0, 0, 0] = 2
    state.cap_height[0, 0, 0] = 2

    state.stack_owner[0, 0, 1] = 2
    state.stack_height[0, 0, 1] = 1
    state.cap_height[0, 0, 1] = 1

    landings = generate_chain_capture_moves_from_position(state, 0, 0, 0)
    assert (0, 2) in landings

    apply_single_chain_capture(state, 0, 0, 0, 0, 2)

    assert int(state.marker_owner[0, 0, 0].item()) == 1
    assert int(state.stack_owner[0, 0, 2].item()) == 1
    assert int(state.stack_height[0, 0, 2].item()) == 3  # 2 + captured 1
    assert int(state.stack_height[0, 0, 1].item()) == 0  # target removed
    assert int(state.buried_rings[0, 2].item()) == 1
