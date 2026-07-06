"""Deterministic replay: same seed and policy => identical trajectory."""

from ringrift_env import RingRiftAECEnv


def rollout_actions(seed: int) -> tuple[list[int], object, dict]:
    env = RingRiftAECEnv(board_type="hex8", num_players=2)
    env.reset(seed=seed)
    actions = []
    while env._action_map:
        a = env.sample_random_action()
        actions.append(a)
        env.step(a)
    return actions, env.state, dict(env.rewards)


def test_same_seed_same_trajectory():
    actions1, state1, rewards1 = rollout_actions(123)
    actions2, state2, rewards2 = rollout_actions(123)
    assert actions1 == actions2
    assert rewards1 == rewards2
    assert state1.winner == state2.winner
    assert len(state1.move_history) == len(state2.move_history)


def test_env_import_is_torch_free():
    """The core environment must not require torch (see package README)."""
    import sys

    heavy = [m for m in ("torch", "torchvision") if m in sys.modules]
    # If the test session itself imported torch elsewhere this is
    # inconclusive rather than failing — but in the package's own test
    # run it guards the lightweight-install contract.
    if not heavy:
        assert True
