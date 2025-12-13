import pytest
import torch


try:
    from app.ai.gpu_parallel_games import (
        BatchGameState,
        GamePhase,
        GameStatus,
        ParallelGameRunner,
    )

    GPU_MODULES_AVAILABLE = True
except Exception as exc:  # pragma: no cover - import guard
    GPU_MODULES_AVAILABLE = False
    GPU_IMPORT_ERROR = str(exc)


pytestmark = pytest.mark.skipif(
    not GPU_MODULES_AVAILABLE,
    reason=f"GPU modules not available: {GPU_IMPORT_ERROR if not GPU_MODULES_AVAILABLE else ''}",
)


def test_gpu_placement_phase_advances_to_movement_when_no_placements() -> None:
    """RR-CANON-R073: ring_placement is part of the turn, not a global stage."""
    device = torch.device("cpu")
    runner = ParallelGameRunner(
        batch_size=1,
        board_size=8,
        num_players=4,
        device=device,
        shadow_validation=False,
        state_validation=False,
        swap_enabled=False,
    )
    state: BatchGameState = runner.state

    g = 0
    p1 = 1

    # Collapse every cell so no placement targets exist.
    state.is_collapsed[g, :, :] = True

    # Give the current player rings; they still must advance to movement.
    state.rings_in_hand[g, p1] = 3

    state.current_player[g] = p1
    state.current_phase[g] = GamePhase.RING_PLACEMENT
    state.game_status[g] = GameStatus.ACTIVE

    weights = [runner._default_weights()]
    runner._step_games(weights)

    assert state.current_player[g].item() == p1
    assert state.current_phase[g].item() == GamePhase.MOVEMENT
    assert state.rings_in_hand[g, p1].item() == 3


def test_gpu_placement_move_does_not_rotate_player() -> None:
    """Placement application should not rotate current_player (rotation happens in END_TURN)."""
    device = torch.device("cpu")
    runner = ParallelGameRunner(
        batch_size=1,
        board_size=8,
        num_players=2,
        device=device,
        shadow_validation=False,
        state_validation=False,
        swap_enabled=False,
    )
    state: BatchGameState = runner.state

    g = 0
    p1 = 1
    p2 = 2

    # Avoid any recovery-eligibility edge cases.
    state.marker_owner[g, :, :] = 0
    state.buried_rings[g, p1] = 0
    state.buried_rings[g, p2] = 0

    state.current_player[g] = p1
    state.current_phase[g] = GamePhase.RING_PLACEMENT
    state.game_status[g] = GameStatus.ACTIVE

    before = state.rings_in_hand[g, p1].item()
    weights = [runner._default_weights()]
    runner._step_games(weights)

    assert state.current_player[g].item() == p1
    assert state.current_phase[g].item() == GamePhase.MOVEMENT
    assert state.rings_in_hand[g, p1].item() == before - 1

    placed = (state.stack_owner[g] == p1) & (state.stack_height[g] == 1) & (state.cap_height[g] == 1)
    assert placed.sum().item() == 1
