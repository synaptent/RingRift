"""Base AI player abstractions for the RingRift AI service.

This module defines the :class:`BaseAI` interface that all concrete AI
implementations (Random, Heuristic, Minimax, MCTS, Descent, etc.) inherit
from, plus small utilities for RNG seeding and per‑instance randomness.

The base class:

* Owns an instance of the canonical Python ``RulesEngine`` host, used to
  enumerate legal moves and apply state transitions.
* Provides per‑instance RNGs wired to ``AIConfig.rng_seed`` (or a derived
  training seed) so that random move selection and any rollout policies
  remain reproducible under a fixed seed.
* Leaves search/evaluation behaviour to subclasses via the abstract
  :meth:`select_move` and :meth:`evaluate_position` methods.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, TypeVar

from ..models import AIConfig, GameState, Move, MoveType
from ..rules.interfaces import RulesEngine
from .swap_evaluation import SwapEvaluator

# Centralized seeding utilities (Lane 3 Consolidation 2025-12)
try:
    from app.training.seed_utils import derive_ai_seed as _derive_ai_seed
    HAS_SEED_UTILS = True
except ImportError:
    HAS_SEED_UTILS = False
    _derive_ai_seed = None

T = TypeVar("T")


def derive_training_seed(config: AIConfig, player_number: int, game_id: str = "") -> int:
    """Derive an RNG seed for AI instances.

    When no explicit seed is provided, generates a random seed using system
    entropy (time, process ID, object ID) to ensure each AI instance gets
    different random behavior. This prevents identical games when running
    multiple simulations.

    When an explicit seed IS provided (via config.rng_seed), derives a
    deterministic seed for reproducibility.

    Args:
        config: AI configuration used to derive the seed.
        player_number: The player index this AI controls (1‑based).
        game_id: Optional game identifier for per-game variation.

    Returns:
        A 32‑bit integer seed suitable for initialising :class:`random.Random`.

    Note:
        Prior to Jan 2026, this function always returned a deterministic seed
        derived from difficulty * 1_000_003, causing identical games when
        running multiple simulations with the same difficulty setting.
    """
    import os
    import time

    # No explicit seed - use random entropy for non-deterministic behavior
    # This ensures each AI instance gets different random behavior
    entropy = int(time.time() * 1_000_000) ^ os.getpid() ^ id(config) ^ (player_number * 31)
    return int(entropy & 0xFFFFFFFF)


class BaseAI(ABC):
    """Abstract base class for all AI implementations.

    Subclasses must implement:

    * :meth:`select_move` – choose a legal move (or ``None``) for the current
      position.
    * :meth:`evaluate_position` – return a scalar score from the AI's
      perspective (larger is better).

    The shared helpers (``get_valid_moves``, ``should_pick_random_move``,
    etc.) provide consistent RNG behaviour and rules‑engine integration
    across all AI families.
    """

    def __init__(self, player_number: int, config: AIConfig) -> None:
        """Initialize a new AI instance.

        Args:
            player_number: The player number this AI controls (1‑based).
            config: AI configuration settings for this instance.
        """
        self.player_number: int = player_number
        self.config: AIConfig = config
        # Incremented each time select_move returns a (non‑None) move.
        self.move_count: int = 0
        # Late import to avoid circular dependency (rules.factory -> default_engine -> game_engine -> ai)
        from ..rules.factory import get_rules_engine
        self.rules_engine: RulesEngine = get_rules_engine()

        # Per-instance RNG used for all stochastic behaviour (random move
        # selection, rollout policies, etc.). Prefer an explicit
        # rng_seed from AIConfig when provided; otherwise fall back to a
        # deterministic but non-SSOT training seed derived from the config.
        # In production, the FastAPI /ai/move endpoint is responsible for
        # supplying a concrete seed based on the TypeScript engine's
        # GameState.rngSeed so that cross-host parity tools can treat that
        # value as the single source of truth.
        if self.config.rng_seed is not None:
            self.rng_seed: int = int(self.config.rng_seed)
        else:
            self.rng_seed = derive_training_seed(self.config, self.player_number)
        self.rng: random.Random = random.Random(self.rng_seed)
        # Lazy swap evaluator for pie-rule decisions (used by search AIs).
        self._swap_evaluator_cache: SwapEvaluator | None = None

    @abstractmethod
    def select_move(self, game_state: GameState) -> Move | None:
        """
        Select the best move for the current game state

        Args:
            game_state: Current game state

        Returns:
            Selected move or None if no valid moves
        """

    @abstractmethod
    def evaluate_position(self, game_state: GameState) -> float:
        """
        Evaluate the current position from this AI's perspective

        Args:
            game_state: Current game state

        Returns:
            Evaluation score (positive = good for this AI, negative = bad)
        """

    def get_evaluation_breakdown(self, game_state: GameState) -> dict[str, float]:
        """Return a structured breakdown of the evaluation for ``game_state``.

        Subclasses may override this to expose richer diagnostics (for example
        per‑feature scores). The default implementation exposes only a
        ``\"total\"`` entry mirroring :meth:`evaluate_position`.

        Args:
            game_state: The position to evaluate.

        Returns:
            A mapping from component name to scalar score.
        """
        return {"total": self.evaluate_position(game_state)}

    def get_valid_moves(self, game_state: GameState) -> list[Move]:
        """Return all legal moves for the current position.

        This is a thin convenience wrapper around the canonical Python
        :class:`RulesEngine`, configured at construction time.

        Args:
            game_state: The position to generate moves for.

        Returns:
            A list of legal :class:`Move` objects for ``self.player_number``.
        """
        return self.rules_engine.get_valid_moves(
            game_state,
            self.player_number,
        )

    def should_pick_random_move(self) -> bool:
        """Return ``True`` if this move should be chosen at random.

        The decision is a Bernoulli draw using the per‑instance RNG and the
        configured ``randomness`` field on :class:`AIConfig`. A ``None`` or
        zero ``randomness`` disables random play entirely.

        Returns:
            ``True`` when a random move should be selected instead of the
            deterministic best move.
        """
        if self.config.randomness is None or self.config.randomness == 0:
            return False
        return self.rng.random() < self.config.randomness

    def get_random_element(self, items: Sequence[T]) -> T | None:
        """Return a random element from ``items`` using the per‑instance RNG.

        Args:
            items: Sequence of candidate values.

        Returns:
            A randomly chosen element, or ``None`` if ``items`` is empty.
        """
        if not items:
            return None
        return self.rng.choice(list(items))

    def shuffle_array(self, items: list[T]) -> list[T]:
        """Shuffle ``items`` in-place using the per‑instance RNG.

        Args:
            items: List to shuffle.

        Returns:
            The same list instance, after shuffling.
        """
        self.rng.shuffle(items)
        return items

    def get_opponent_numbers(self, game_state: GameState) -> list[int]:
        """Return the list of opponent player numbers for this game state.

        Args:
            game_state: The current game state.

        Returns:
            A list of player numbers for all players other than
            ``self.player_number``.
        """
        return [
            p.player_number
            for p in game_state.players
            if p.player_number != self.player_number
        ]

    def maybe_select_swap_move(
        self,
        game_state: GameState,
        valid_moves: list[Move],
    ) -> Move | None:
        """Return a SWAP_SIDES move when it is clearly advantageous.

        Generic tree search does not model the identity swap semantics of the
        pie rule. When Player 2 is offered a `swap_sides` meta‑move, we handle
        the decision explicitly using the Opening Position Classifier.
        """
        if self.player_number != 2:
            return None
        if len(game_state.players) != 2:
            return None

        swap_moves = [m for m in valid_moves if m.type == MoveType.SWAP_SIDES]
        if not swap_moves:
            return None

        evaluator: SwapEvaluator | None = None
        try:
            evaluator = self.swap_evaluator
        except AttributeError:
            evaluator = None

        if not isinstance(evaluator, SwapEvaluator):
            if self._swap_evaluator_cache is None:
                self._swap_evaluator_cache = SwapEvaluator()
            evaluator = self._swap_evaluator_cache

        score = evaluator.evaluate_swap_with_classifier(game_state)
        threshold = float(getattr(self.config, "swap_threshold", 0.0))
        if score > threshold:
            return swap_moves[0]

        return None

    def get_player_info(
        self,
        game_state: GameState,
        player_number: int | None = None,
    ) -> Any | None:
        """Return the player record for ``player_number`` (or this AI).

        Args:
            game_state: The current game state.
            player_number: Player to look up; when ``None``, this method
                returns information for ``self.player_number``.

        Returns:
            The matching player object from ``game_state.players``, or
            ``None`` if no such player exists.
        """
        target_player = (
            player_number if player_number is not None else self.player_number
        )
        for player in game_state.players:
            if player.player_number == target_player:
                return player
        return None

    def reset_for_new_game(self, *, rng_seed: int | None = None) -> None:
        """Reset per-game mutable state for self-play and evaluation loops.

        Some training/evaluation drivers reuse a single AI instance across
        multiple games (to amortize neural model loading). This helper makes
        that reuse safer and more reproducible by:

        - Resetting ``move_count``.
        - Optionally reseeding the per-instance RNG.
        - Clearing any retained search tree when supported.
        """
        self.move_count = 0
        if rng_seed is not None:
            self.rng_seed = int(rng_seed) & 0xFFFFFFFF
            self.rng = random.Random(self.rng_seed)

        clear_tree = getattr(self, "clear_search_tree", None)
        if callable(clear_tree):
            try:
                clear_tree()
            except (AttributeError, RuntimeError, TypeError):
                # Best-effort only: failures should not break self-play.
                pass

    def __repr__(self) -> str:
        """String representation of AI"""
        return (
            f"{self.__class__.__name__}"
            f"(player={self.player_number}, "
            f"difficulty={self.config.difficulty})"
        )
