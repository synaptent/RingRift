"""Random full-game playouts across board types and player counts."""

import pytest

from ringrift_env import RingRiftAECEnv


def run_episode(board_type: str, num_players: int, seed: int, max_moves: int | None = None):
    env = RingRiftAECEnv(board_type=board_type, num_players=num_players, max_moves=max_moves)
    env.reset(seed=seed)
    steps = 0
    limit = (max_moves or 10_000) + 10
    while env._action_map and steps <= limit:
        env.step(env.sample_random_action())
        steps += 1
    assert all(
        env.terminations[a] or env.truncations[a] for a in env.possible_agents
    ), f"episode did not finish after {steps} steps"
    return env, steps


@pytest.mark.parametrize("board_type", ["hex8", "square8"])
@pytest.mark.parametrize("num_players", [2, 3, 4])
def test_small_board_full_game(board_type, num_players):
    env, steps = run_episode(board_type, num_players, seed=11 if board_type == "hex8" else 13)
    assert steps > 0
    info = env.infos[env.possible_agents[0]]
    assert "victory_reason" in info
    winner = info["winner"]
    if winner is not None and not env.truncations[env.possible_agents[0]]:
        assert env.rewards[f"player_{winner}"] == 1.0
        losers = [a for a in env.possible_agents if a != f"player_{winner}"]
        assert all(env.rewards[a] == -1.0 for a in losers)


@pytest.mark.parametrize("board_type,num_players", [("square19", 2), ("hexagonal", 2)])
def test_large_board_truncated_episode(board_type, num_players):
    # Large boards take thousands of random moves to finish; a capped
    # episode exercises the full step/mask/truncation path quickly.
    env, steps = run_episode(board_type, num_players, seed=5, max_moves=120)
    assert steps <= 130
    assert all(
        env.truncations[a] or env.terminations[a] for a in env.possible_agents
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "board_type,num_players",
    [("square19", 2), ("square19", 3), ("square19", 4),
     ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
     ("hex8", 2), ("square8", 2)],
)
def test_full_length_games(board_type, num_players):
    env, steps = run_episode(board_type, num_players, seed=99)
    assert steps > 0
