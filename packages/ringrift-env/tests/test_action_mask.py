"""Action-mask correctness: the mask must exactly mirror engine legality."""

import numpy as np
import pytest

from ringrift_env import RingRiftAECEnv


@pytest.mark.parametrize("board_type", ["hex8", "square8"])
def test_mask_matches_engine_enumeration(board_type):
    """Across 100+ sampled states, mask indices == encoded legal moves."""
    env = RingRiftAECEnv(board_type=board_type, num_players=2)
    states_checked = 0
    for episode_seed in (1, 2, 3):
        env.reset(seed=episode_seed)
        while env._action_map and states_checked < 120:
            obs = env.observe()
            mask = obs["action_mask"]
            legal = env.legal_action_indices()

            # Mask is exactly the set of legal indices — no more, no fewer.
            assert mask.dtype == bool
            assert mask.shape == (env.action_space_size,)
            assert sorted(np.flatnonzero(mask).tolist()) == legal
            # Every legal engine move maps to a distinct valid index.
            assert len(legal) == len(env.legal_moves())
            assert all(0 <= i < env.action_space_size for i in legal)
            # decode(index) returns the move that produced the index.
            idx = legal[0]
            assert env.decode_action(idx) is env._action_map[idx]

            env.step(env.sample_random_action())
            states_checked += 1
    assert states_checked >= 100


def test_illegal_action_raises():
    env = RingRiftAECEnv(board_type="hex8", num_players=2)
    env.reset(seed=0)
    legal = set(env.legal_action_indices())
    illegal = next(i for i in range(env.action_space_size) if i not in legal)
    with pytest.raises(ValueError, match="not legal"):
        env.step(illegal)
