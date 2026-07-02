"""ringrift-env: RingRift as a multi-agent RL environment.

Quickstart::

    from ringrift_env import RingRiftAECEnv

    env = RingRiftAECEnv(board_type="hex8", num_players=2)
    env.reset(seed=42)
    while env.agents and not (any(env.terminations.values()) or any(env.truncations.values())):
        env.step(env.sample_random_action())
    print(env.rewards, env.infos)
"""

from .aec import ActionEncodingError, RingRiftAECEnv

__version__ = "0.1.0"
__all__ = ["RingRiftAECEnv", "ActionEncodingError", "__version__"]
