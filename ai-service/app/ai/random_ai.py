"""Random AI implementation for RingRift.

This agent selects uniformly random legal moves using the per‑instance RNG on
the :class:`BaseAI`. It is primarily intended for testing, baselines, and
very low difficulties, rather than competitive play.
"""

from __future__ import annotations

from ..models import GameState, Move
from .base import BaseAI


class RandomAI(BaseAI):
    """AI that selects random valid moves."""

    def select_move(self, game_state: GameState) -> Move | None:
        """Select a random valid move for ``game_state``.

        Args:
            game_state: Current game state.

        Returns:
            A random valid :class:`Move` or ``None`` if no legal moves exist.
        """
        # Get all legal moves using the canonical RulesEngine host, which
        # includes both interactive moves and required bookkeeping moves
        # (no_*_action / forced_elimination) when applicable.
        valid_moves = self.get_valid_moves(game_state)

        if not valid_moves:
            return None

        # Select random move using the per-instance RNG.
        selected = self.get_random_element(valid_moves)

        self.move_count += 1
        return selected

    def evaluate_position(self, game_state: GameState) -> float:
        """Return a small random evaluation for ``game_state``.

        RandomAI does not attempt to evaluate positions meaningfully. It
        returns a small random value to introduce variance in diagnostic
        tooling that inspects scalar evaluations.

        Args:
            game_state: Current game state (unused).

        Returns:
            A small random float in ``[-0.1, 0.1]``.
        """
        _ = game_state  # unused in this implementation
        return self.rng.uniform(-0.1, 0.1)

    def get_evaluation_breakdown(
        self,
        game_state: GameState,
    ) -> dict[str, float]:
        """Return a simple breakdown for :meth:`evaluate_position`.

        Args:
            game_state: Current game state (unused).

        Returns:
            A mapping containing a neutral ``\"total\"`` score and a single
            ``\"random_variance\"`` component drawn from the per‑instance RNG.
        """
        _ = game_state  # unused in this implementation
        return {
            "total": 0.0,
            "random_variance": self.rng.uniform(-0.1, 0.1),
        }
