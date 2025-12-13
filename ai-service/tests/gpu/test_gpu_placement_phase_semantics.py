import torch


def _fill_square_board_with_stacks(state, *, owner: int = 2) -> None:
    # Fill every cell with a height-1 stack so there are no legal placement targets.
    state.stack_owner[:, :, :] = owner
    state.stack_height[:, :, :] = 1
    state.cap_height[:, :, :] = 1


def test_gpu_placement_phase_advances_to_movement_when_board_full(device):
    from app.ai.gpu_parallel_games import BatchGameState, GamePhase, ParallelGameRunner

    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=4, device=device)
    runner.reset_games()

    # Ensure current player still has rings, but there are no legal placement moves.
    runner.state.current_player[:] = 1
    runner.state.current_phase[:] = GamePhase.RING_PLACEMENT
    runner.state.rings_in_hand[:, 1] = 18

    _fill_square_board_with_stacks(runner.state, owner=2)

    mask = torch.tensor([True], dtype=torch.bool, device=device)
    runner._step_placement_phase(mask, [runner._default_weights()])

    assert int(runner.state.current_phase[0].item()) == int(GamePhase.MOVEMENT)
    assert int(runner.state.current_player[0].item()) == 1


def test_gpu_placement_does_not_rotate_current_player(device):
    from app.ai.gpu_parallel_games import GamePhase, ParallelGameRunner

    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=4, device=device)
    runner.reset_games()

    runner.state.current_player[:] = 3
    runner.state.current_phase[:] = GamePhase.RING_PLACEMENT
    runner.state.rings_in_hand[:, 3] = 18

    mask = torch.tensor([True], dtype=torch.bool, device=device)
    before_rings = int(runner.state.rings_in_hand[0, 3].item())
    runner._step_placement_phase(mask, [runner._default_weights()])

    after_rings = int(runner.state.rings_in_hand[0, 3].item())
    assert after_rings == before_rings - 1
    assert int(runner.state.current_phase[0].item()) == int(GamePhase.MOVEMENT)
    assert int(runner.state.current_player[0].item()) == 3

    # Exactly one stack should now be controlled by the placing player.
    assert int((runner.state.stack_owner[0] == 3).sum().item()) == 1

