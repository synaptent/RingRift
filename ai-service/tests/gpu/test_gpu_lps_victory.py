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


def test_check_real_actions_batch_rings_in_hand() -> None:
    """Test _check_real_actions_batch: rings in hand counts as real action."""
    device = torch.device("cpu")
    runner = ParallelGameRunner(
        batch_size=4,
        board_size=8,
        num_players=2,
        device=device,
        shadow_validation=False,
        state_validation=False,
        swap_enabled=False,
    )
    state = runner.state

    # Game 0: P1 has rings -> has action
    state.rings_in_hand[0, 1] = 5
    state.rings_in_hand[0, 2] = 0

    # Game 1: P1 has no rings but has stacks
    state.rings_in_hand[1, 1] = 0
    state.stack_owner[1, 2, 2] = 1
    state.stack_height[1, 2, 2] = 3

    # Game 2: P1 has neither -> no action
    state.rings_in_hand[2, 1] = 0

    # Game 3: P1 has rings -> has action
    state.rings_in_hand[3, 1] = 2

    mask = torch.ones(4, dtype=torch.bool, device=device)
    result = runner._check_real_actions_batch(mask, player=1)

    # Games 0 and 3 have rings
    assert result[0].item() is True
    assert result[3].item() is True
    # Game 2 has nothing
    assert result[2].item() is False


def test_check_real_actions_batch_empty_mask() -> None:
    """Test _check_real_actions_batch with empty mask returns zeros."""
    device = torch.device("cpu")
    runner = ParallelGameRunner(
        batch_size=4,
        board_size=8,
        num_players=2,
        device=device,
        shadow_validation=False,
        state_validation=False,
        swap_enabled=False,
    )

    # Even with rings, empty mask should return all False
    runner.state.rings_in_hand[:, 1] = 5
    mask = torch.zeros(4, dtype=torch.bool, device=device)
    result = runner._check_real_actions_batch(mask, player=1)

    assert not result.any()


def test_check_real_actions_batch_partial_mask() -> None:
    """Test _check_real_actions_batch only checks masked games."""
    device = torch.device("cpu")
    runner = ParallelGameRunner(
        batch_size=4,
        board_size=8,
        num_players=2,
        device=device,
        shadow_validation=False,
        state_validation=False,
        swap_enabled=False,
    )

    # All games have rings
    runner.state.rings_in_hand[:, 1] = 5

    # But only mask games 1 and 2
    mask = torch.tensor([False, True, True, False], device=device)
    result = runner._check_real_actions_batch(mask, player=1)

    # Only masked games should report real actions
    assert result[0].item() is False
    assert result[1].item() is True
    assert result[2].item() is True
    assert result[3].item() is False


def test_check_real_actions_batch_no_stacks() -> None:
    """Test _check_real_actions_batch: no rings + no stacks = no action."""
    device = torch.device("cpu")
    runner = ParallelGameRunner(
        batch_size=2,
        board_size=8,
        num_players=2,
        device=device,
        shadow_validation=False,
        state_validation=False,
        swap_enabled=False,
    )
    state = runner.state

    # No rings, no stacks for P1
    state.rings_in_hand[0, 1] = 0
    state.rings_in_hand[1, 1] = 0

    mask = torch.ones(2, dtype=torch.bool, device=device)
    result = runner._check_real_actions_batch(mask, player=1)

    assert not result[0].item()
    assert not result[1].item()
