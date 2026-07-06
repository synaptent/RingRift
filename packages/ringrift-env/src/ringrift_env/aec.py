"""PettingZoo AEC-style multi-agent environment for RingRift.

This is a thin facade over the canonical training environment
(``app.training.env.RingRiftEnv``) and the canonical action encoding
(``app.ai.canonical_move_encoding``). It adds:

- an integer action space with per-state action masks, and
- per-agent reward/termination bookkeeping in the AEC style.

It deliberately does NOT reimplement any rules logic: legality, phase
bookkeeping, termination, and move application are all delegated to the
rules engine, which is parity-tested against the TypeScript source of
truth in ``src/shared/engine``.

The API mirrors PettingZoo's ``AECEnv`` (``reset`` / ``observe`` / ``last``
/ ``step`` / ``agent_selection``) without depending on the pettingzoo
package, so the core install stays lightweight.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._bootstrap import ensure_engine_importable

ensure_engine_importable()

from app.ai.canonical_move_encoding import (  # noqa: E402
    encode_move_for_board,
    get_encoder_for_board,
)
from app.models import BoardType, GameState, Move  # noqa: E402
from app.training.env import TrainingEnvConfig, make_env  # noqa: E402

_BOARD_TYPES = {bt.value: bt for bt in BoardType}


class ActionEncodingError(RuntimeError):
    """A legal move failed to map to a unique, valid policy index."""


class RingRiftAECEnv:
    """Turn-based multi-agent RingRift environment (AEC style).

    Parameters
    ----------
    board_type:
        One of ``"square8"``, ``"square19"``, ``"hex8"``, ``"hexagonal"``
        (or a ``BoardType`` enum value).
    num_players:
        2, 3, or 4.
    max_moves:
        Optional episode cap. Defaults to the engine's theoretical
        maximum for the board/player combination. Hitting the cap ends
        the episode as a *truncation*.

    Notes
    -----
    * **Observation**: ``observe()`` returns ``{"state": GameState,
      "action_mask": np.ndarray}``. The ``GameState`` is the full
      canonical (perfect-information) state; callers who want tensor
      planes should use the encoders in ``app.training.encoding``.
    * **Actions**: integer indices. Indices below
      ``canonical_action_space_size`` are the board's canonical policy
      indices (the same indexing used by RingRift's trained networks).
      RingRift's canonical encoding intentionally maps some *choice*
      moves (line/territory option selection, ring-elimination targets,
      and every hex special move) onto shared indices — the policy head
      treats them as one categorical and the search layer disambiguates
      with Move objects. Because an integer action must identify exactly
      one move, legal moves that cannot be distinguished canonically are
      exposed in an *overflow block* starting at
      ``canonical_action_space_size``, assigned deterministically in
      engine enumeration order and valid only for the current state.
      Only indices where ``action_mask`` is ``True`` are legal.
    * **Rewards**: emitted only at episode end. The winner receives
      ``+1.0`` and every other agent ``-1.0``; draws/stalemates/
      truncations give everyone ``0.0``. (Rank-aware multiplayer
      rewards are a planned extension.)
    * **Determinism**: the rules engine is fully deterministic; the only
      randomness is whatever policy selects actions. ``reset(seed=...)``
      seeds ``self.rng`` (used by ``sample_random_action``), so identical
      seeds and policies yield identical trajectories.
    """

    metadata = {"name": "ringrift", "is_parallelizable": False}

    #: Number of per-state overflow slots appended after the canonical
    #: policy indices for choice moves the canonical encoding collapses.
    #: Sized generously: collisions come from simultaneous line/territory
    #: choice moves, which are bounded by board line/region counts.
    DEFAULT_OVERFLOW_SLOTS = 256

    def __init__(
        self,
        board_type: str | BoardType = "hex8",
        num_players: int = 2,
        max_moves: int | None = None,
        overflow_slots: int = DEFAULT_OVERFLOW_SLOTS,
    ) -> None:
        if isinstance(board_type, str):
            try:
                board_type = _BOARD_TYPES[board_type]
            except KeyError:
                raise ValueError(
                    f"Unknown board_type {board_type!r}; expected one of "
                    f"{sorted(_BOARD_TYPES)}"
                ) from None
        if num_players not in (2, 3, 4):
            raise ValueError(f"num_players must be 2, 3, or 4 (got {num_players})")

        self.board_type = board_type
        self.num_players = num_players
        self._encoder = get_encoder_for_board(board_type)
        self.canonical_action_space_size: int = self._encoder.policy_size
        self.overflow_slots = overflow_slots
        self.action_space_size: int = self.canonical_action_space_size + overflow_slots
        self.possible_agents: list[str] = [
            f"player_{i}" for i in range(1, num_players + 1)
        ]

        self._env = make_env(
            TrainingEnvConfig(
                board_type=board_type,
                num_players=num_players,
                max_moves=max_moves,
            )
        )
        self.rng: np.random.Generator = np.random.default_rng()

        # Populated by reset().
        self.agents: list[str] = []
        self.rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict[str, Any]] = {}
        self._state: GameState | None = None
        self._action_map: dict[int, Move] = {}
        self._done = False

    # ------------------------------------------------------------------
    # Core AEC API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> None:
        """Start a new episode. Seeds only this env's ``rng``."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # NOTE: the inner reset is called unseeded on purpose — the engine
        # is deterministic, and seeded inner resets would require torch.
        self._state = self._env.reset()
        self.agents = list(self.possible_agents)
        self.rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._done = False
        self._rebuild_action_map()

    @property
    def state(self) -> GameState:
        """The full canonical game state (perfect information)."""
        assert self._state is not None, "Call reset() first"
        return self._state

    @property
    def agent_selection(self) -> str:
        """Name of the agent whose turn it is."""
        return f"player_{self.state.current_player}"

    def observe(self, agent: str | None = None) -> dict[str, Any]:
        """Observation for ``agent`` (defaults to the acting agent).

        RingRift is perfect-information, so every agent sees the same
        state; the ``action_mask`` is meaningful for the acting agent.
        """
        mask = np.zeros(self.action_space_size, dtype=bool)
        if not self._done:
            mask[list(self._action_map)] = True
        return {"state": self.state, "action_mask": mask}

    def last(self) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """(observation, reward, terminated, truncated, info) for the acting agent."""
        agent = self.agent_selection
        return (
            self.observe(agent),
            self.rewards.get(agent, 0.0),
            self.terminations.get(agent, False),
            self.truncations.get(agent, False),
            self.infos.get(agent, {}),
        )

    def step(self, action: int) -> None:
        """Apply an integer action for the acting agent."""
        self.step_move(self.decode_action(action))

    def step_move(self, move: Move) -> None:
        """Apply a fully-specified :class:`Move` (bypasses the index layer)."""
        if self._done:
            raise RuntimeError("Episode is over; call reset()")
        state, _reward, done, info = self._env.step(move)
        self._state = state
        self._done = done
        if done:
            self._finalize(info)
        else:
            self._rebuild_action_map()
            acting = self.agent_selection
            for agent in self.agents:
                self.infos[agent] = {"move_count": info.get("move_count")}
            self.infos[acting]["legal_moves"] = len(self._action_map)

    # ------------------------------------------------------------------
    # Action-space helpers
    # ------------------------------------------------------------------

    def legal_action_indices(self) -> list[int]:
        """Sorted legal action indices for the acting agent."""
        return sorted(self._action_map)

    def legal_moves(self) -> list[Move]:
        """Legal :class:`Move` objects for the acting agent."""
        return list(self._action_map.values())

    def decode_action(self, action: int) -> Move:
        """Map a legal action index back to its :class:`Move`."""
        try:
            return self._action_map[int(action)]
        except KeyError:
            raise ValueError(
                f"Action {action} is not legal in the current state "
                f"({len(self._action_map)} legal actions)"
            ) from None

    def sample_random_action(self) -> int:
        """Uniformly sample a legal action using this env's ``rng``."""
        indices = self.legal_action_indices()
        if not indices:
            raise RuntimeError("No legal actions (episode is over?)")
        return int(indices[self.rng.integers(len(indices))])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _rebuild_action_map(self) -> None:
        assert self._state is not None
        moves = self._env.legal_moves()
        action_map: dict[int, Move] = {}
        overflow: list[Move] = []
        for move in moves:
            idx = encode_move_for_board(move, self._state.board)
            if idx < 0 or idx in action_map:
                # The canonical encoding collapses choice moves (and every
                # hex special move) onto shared indices by design; moves it
                # cannot uniquely represent go to the overflow block, in
                # deterministic engine enumeration order.
                overflow.append(move)
            else:
                action_map[idx] = move
        if len(overflow) > self.overflow_slots:
            raise ActionEncodingError(
                f"{len(overflow)} indistinguishable legal moves exceed the "
                f"overflow block ({self.overflow_slots} slots); construct "
                f"the env with a larger overflow_slots"
            )
        for offset, move in enumerate(overflow):
            action_map[self.canonical_action_space_size + offset] = move
        self._action_map = action_map

    def _finalize(self, info: dict[str, Any]) -> None:
        winner = info.get("winner")
        reason = info.get("victory_reason", "unknown")
        truncated = reason == "max_moves"
        for agent in self.agents:
            self.truncations[agent] = truncated
            self.terminations[agent] = not truncated
            if winner is None or truncated:
                self.rewards[agent] = 0.0
            else:
                self.rewards[agent] = (
                    1.0 if agent == f"player_{winner}" else -1.0
                )
            self.infos[agent] = {
                "winner": winner,
                "victory_reason": reason,
                "move_count": info.get("move_count"),
            }
        self._action_map = {}
