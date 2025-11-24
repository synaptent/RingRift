from typing import Optional, Tuple, List, Dict, Any
from app.models import GameState, Move, BoardType, GameStatus
from app.game_engine import GameEngine


class RingRiftEnv:
    """RL-style environment wrapper for RingRift.

    Provides a minimal `reset()`, `step()`, and `legal_moves()` interface
    for self-play style loops. The underlying Python GameEngine and
    GameState models are already N-player aware; the `num_players`
    parameter exposes that capability to training code (2–4 players).
    """

    def __init__(
        self,
        board_type: BoardType = BoardType.SQUARE8,
        max_moves: int = 200,
        reward_on: str = "terminal",  # "terminal" or "shaped"
        num_players: int = 2,
    ):
        self.board_type = board_type
        self.max_moves = max_moves
        self.reward_on = reward_on
        self.num_players = num_players
        self._state: Optional[GameState] = None
        self._move_count: int = 0

    def reset(self, seed: Optional[int] = None) -> GameState:
        """Create a fresh GameState for self-play.

        When `num_players` > 2, the initial state will contain that many
        AI-controlled players with per-player ring caps and victory
        thresholds aligned with the shared TypeScript initial-state
        helpers.

        If `seed` is provided, it is threaded into Python RNGs used by
        any stochastic components (future variants); the core game rules
        remain deterministic.
        """
        if seed is not None:
            import random
            import numpy as np
            import torch

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Reuse a shared helper from generate_data
        # Avoid circular import by importing inside method
        from app.training.generate_data import create_initial_state

        self._state = create_initial_state(
            board_type=self.board_type,
            num_players=self.num_players,
        )
        self._move_count = 0
        return self._state

    @property
    def state(self) -> GameState:
        assert self._state is not None, "Call reset() before using env"
        return self._state

    def legal_moves(self) -> List[Move]:
        """
        Return legal moves for the current player, using the same logic
        as AIs: GameEngine.get_valid_moves.
        """
        return GameEngine.get_valid_moves(
            self.state, self.state.current_player
        )

    def step(
        self, move: Move
    ) -> Tuple[GameState, float, bool, Dict[str, Any]]:
        """
        Apply a move, returning (new_state, reward, done, info).

        - new_state: updated GameState from GameEngine.apply_move.
        - reward: from the perspective of the player who just moved,
          according to reward_on:
            * "terminal": +1/-1/0 only at terminal states, 0 otherwise.
            * "shaped": use calculate_outcome-style shaping at terminal.
        - done: True when game_status != ACTIVE or max_moves reached.
        - info: may include raw winner, reason, and move_count.
        """
        self._state = GameEngine.apply_move(self.state, move)
        self._move_count += 1

        done = (
            self._state.game_status != GameStatus.ACTIVE
            or self._move_count >= self.max_moves
        )

        reward = 0.0
        info: Dict[str, Any] = {"winner": self._state.winner}

        if done:
            if self.reward_on == "terminal":
                # Perspective: player who just moved is move.player
                perspective = move.player
                if self._state.winner is None:
                    reward = 0.0
                elif self._state.winner == perspective:
                    reward = 1.0
                else:
                    reward = -1.0
            else:
                # Reuse calculate_outcome-like shaping
                from app.training.generate_data import calculate_outcome
                reward = calculate_outcome(
                    self._state,
                    player_number=move.player,
                    depth=self._move_count
                )

        info["move_count"] = self._move_count
        return self._state, reward, done, info
