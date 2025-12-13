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


def test_gpu_lps_victory_requires_exclusive_rounds() -> None:
    """RR-CANON-R172: LPS is round-based, not material-only.

    Note: This test explicitly sets lps_victory_rounds=2 to test the configurable
    threshold feature with a shorter game. The canonical default is 3 rounds.
    """
    device = torch.device("cpu")
    torch.manual_seed(0)
    runner = ParallelGameRunner(
        batch_size=1,
        board_size=8,
        num_players=2,
        device=device,
        shadow_validation=False,
        state_validation=False,
        swap_enabled=False,
        lps_victory_rounds=2,  # Testing custom threshold (default is 3)
    )
    state: BatchGameState = runner.state

    g = 0
    p1 = 1
    p2 = 2

    # Player 1 has real actions via ring placement.
    state.rings_in_hand[g, p1] = 5
    state.rings_in_hand[g, p2] = 0

    # Player 2 has no real actions (no rings, no stacks), but is not permanently
    # eliminated because recovery moves exist (buried rings + marker).
    state.buried_rings[g, p2] = 5
    state.marker_owner[g, 4, 4] = p2

    # Start at the beginning of P1's turn.
    state.current_player[g] = p1
    state.current_phase[g] = GamePhase.RING_PLACEMENT
    state.game_status[g] = GameStatus.ACTIVE

    weights = [runner._default_weights()]

    completed_at = None
    for step in range(80):
        runner._step_games(weights)
        runner._check_victory_conditions()
        if state.game_status[g].item() == GameStatus.COMPLETED:
            completed_at = step
            break

    assert completed_at is not None
    assert state.winner[g].item() == p1
    victory_type, _ = state.derive_victory_type(g, max_moves=500)
    assert victory_type == "lps"
    assert state.lps_consecutive_exclusive_player[g].item() == p1
    assert state.lps_consecutive_exclusive_rounds[g].item() >= 2
