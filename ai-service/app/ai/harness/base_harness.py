"""Abstract base class for AI harnesses.

A harness wraps an AI evaluation algorithm (Gumbel MCTS, Minimax, MaxN, etc.)
and provides a unified interface for move selection with metadata capture.

This abstraction enables:
- Consistent Elo tracking across different algorithms
- Visit distribution capture for soft policy targets
- Model type agnostic evaluation (NN, NNUE, heuristic)

Player Validation (Dec 2025):
- Each harness type has specific player count restrictions
- MINIMAX: 2-player only (uses alpha-beta pruning)
- MAXN/BRS: 3-4 player only (multiplayer-specific algorithms)
- Others: 2-4 players supported
- Validation is performed in AIHarness.__init__()
"""

from __future__ import annotations

import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from .evaluation_metadata import EvaluationMetadata

if TYPE_CHECKING:
    from ...models import AIConfig, BoardType, GameState, Move

logger = logging.getLogger(__name__)


class HarnessType(Enum):
    """Available AI harness types."""

    GUMBEL_MCTS = "gumbel_mcts"
    GPU_GUMBEL = "gpu_gumbel"
    MINIMAX = "minimax"
    MAXN = "maxn"
    BRS = "brs"
    POLICY_ONLY = "policy_only"
    DESCENT = "descent"
    HEURISTIC = "heuristic"
    RANDOM = "random"  # Jan 1, 2026: Added for baseline comparison and diversity


class ModelType(Enum):
    """Types of models that can be used with harnesses."""

    NEURAL_NET = "nn"          # Full neural network (NN v2-v6)
    NNUE = "nnue"              # NNUE (value-only or with policy)
    HEURISTIC = "heuristic"    # Hand-crafted heuristic evaluation


@dataclass
class HarnessConfig:
    """Configuration for a harness instance.

    Attributes:
        harness_type: Type of search algorithm to use.
        model_type: Type of evaluation model (NN, NNUE, heuristic).
        model_path: Path to model checkpoint (optional for heuristic).
        model_id: Identifier for Elo tracking.
        board_type: Board type for model compatibility.
        num_players: Number of players (2, 3, or 4).
        difficulty: AI difficulty level (1-10).
        think_time_ms: Maximum time per move in milliseconds.
        simulations: Number of MCTS simulations (for tree search).
        depth: Search depth (for minimax-style search).
        extra: Additional harness-specific options.
    """

    harness_type: HarnessType = HarnessType.GUMBEL_MCTS
    model_type: ModelType = ModelType.NEURAL_NET
    model_path: str | None = None
    model_id: str = ""
    board_type: BoardType | None = None
    num_players: int = 2
    difficulty: int = 5
    think_time_ms: int | None = None
    simulations: int = 200
    depth: int = 3
    extra: dict[str, Any] = field(default_factory=dict)

    def get_config_hash(self) -> str:
        """Generate a hash of configuration for Elo tracking.

        Only includes parameters that affect play strength.
        """
        key_params = {
            "harness": self.harness_type.value,
            "model_type": self.model_type.value,
            "difficulty": self.difficulty,
            "simulations": self.simulations if self.harness_type in (
                HarnessType.GUMBEL_MCTS, HarnessType.GPU_GUMBEL
            ) else None,
            "depth": self.depth if self.harness_type in (
                HarnessType.MINIMAX, HarnessType.MAXN, HarnessType.BRS
            ) else None,
        }
        # Remove None values
        key_params = {k: v for k, v in key_params.items() if v is not None}
        param_str = str(sorted(key_params.items()))
        return hashlib.md5(param_str.encode()).hexdigest()[:8]


class AIHarness(ABC):
    """Abstract base class for AI harnesses.

    A harness provides a unified interface for evaluating game positions
    and selecting moves, capturing metadata for Elo tracking and training.

    Subclasses implement specific search algorithms (MCTS, Minimax, etc.)
    while this base class handles common functionality like timing and
    metadata assembly.

    Class Attributes:
        supports_nn: Whether this harness can use full neural networks.
        supports_nnue: Whether this harness can use NNUE evaluation.
        requires_policy_head: Whether the harness requires policy output.
        min_players: Minimum supported player count (default 2).
        max_players: Maximum supported player count (default 4).
    """

    # Subclasses must override these
    supports_nn: bool = True
    supports_nnue: bool = True
    requires_policy_head: bool = False

    # Player count restrictions (subclasses can override)
    min_players: int = 2
    max_players: int = 4

    def __init__(self, config: HarnessConfig) -> None:
        """Initialize the harness with configuration.

        Args:
            config: Harness configuration including model and search params.

        Raises:
            ValueError: If num_players is outside the valid range for this harness.
        """
        # Validate player count before storing config
        self._validate_player_count(config.num_players)

        self.config = config
        self._underlying_ai: Any = None
        self._last_visit_distribution: dict[str, float] | None = None
        self._last_policy_distribution: dict[str, float] | None = None

    def _validate_player_count(self, num_players: int) -> None:
        """Validate that the player count is supported by this harness.

        Args:
            num_players: Number of players to validate.

        Raises:
            ValueError: If num_players is outside the valid range.
        """
        if num_players < self.min_players or num_players > self.max_players:
            raise ValueError(
                f"{self.__class__.__name__} requires {self.min_players}-{self.max_players} players, "
                f"got {num_players}"
            )

    @classmethod
    def get_player_range(cls) -> tuple[int, int]:
        """Get the valid player count range for this harness.

        Returns:
            Tuple of (min_players, max_players).
        """
        return (cls.min_players, cls.max_players)

    @classmethod
    def supports_player_count(cls, num_players: int) -> bool:
        """Check if this harness supports a given player count.

        Args:
            num_players: Number of players to check.

        Returns:
            True if the player count is supported.
        """
        return cls.min_players <= num_players <= cls.max_players

    @abstractmethod
    def _create_underlying_ai(self, player_number: int) -> Any:
        """Create the underlying AI instance for a player.

        Args:
            player_number: The player number (1-based).

        Returns:
            The underlying AI instance (MinimaxAI, GumbelMCTSAI, etc.)
        """

    @abstractmethod
    def _select_move_impl(
        self,
        game_state: GameState,
        player_number: int,
    ) -> tuple[Move | None, dict[str, Any]]:
        """Implementation of move selection.

        Args:
            game_state: Current game state.
            player_number: Player to select move for.

        Returns:
            Tuple of (selected_move, additional_metadata)
            The metadata dict should include:
            - value_estimate: Position evaluation
            - nodes_visited: Search nodes explored
            - search_depth: Max depth reached
            - simulations: Number of MCTS simulations (if applicable)
        """

    def evaluate(
        self,
        game_state: GameState,
        player_number: int,
    ) -> tuple[Move | None, EvaluationMetadata]:
        """Evaluate a position and select a move with full metadata.

        This is the main entry point for harness evaluation. It wraps the
        implementation-specific _select_move_impl with timing and metadata
        assembly.

        Args:
            game_state: Current game state.
            player_number: Player to select move for.

        Returns:
            Tuple of (selected_move, evaluation_metadata)
        """
        start_time = time.perf_counter()

        # Ensure underlying AI is created for this player
        if self._underlying_ai is None:
            self._underlying_ai = self._create_underlying_ai(player_number)

        # Call implementation
        move, impl_metadata = self._select_move_impl(game_state, player_number)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Assemble full metadata
        metadata = EvaluationMetadata(
            value_estimate=impl_metadata.get("value_estimate", 0.0),
            visit_distribution=self._last_visit_distribution,
            policy_distribution=self._last_policy_distribution,
            search_depth=impl_metadata.get("search_depth"),
            nodes_visited=impl_metadata.get("nodes_visited", 0),
            time_ms=elapsed_ms,
            harness_type=self.config.harness_type.value,
            model_type=self.config.model_type.value,
            model_id=self.config.model_id,
            config_hash=self.config.get_config_hash(),
            simulations=impl_metadata.get("simulations"),
            extra=impl_metadata.get("extra", {}),
        )

        return move, metadata

    def get_visit_distribution(self) -> dict[str, float] | None:
        """Get the visit distribution from the last search.

        Returns:
            Dict mapping action keys to visit counts, or None if not available.
        """
        return self._last_visit_distribution

    def get_policy_distribution(self) -> dict[str, float] | None:
        """Get the policy distribution from the last evaluation.

        Returns:
            Dict mapping action keys to policy probabilities, or None.
        """
        return self._last_policy_distribution

    def get_composite_participant_id(self) -> str:
        """Generate composite ID for Elo tracking.

        Format: {model_id}:{harness_type}:{config_hash}
        """
        return ":".join([
            self.config.model_id or "unknown",
            self.config.harness_type.value,
            self.config.get_config_hash(),
        ])

    def reset(self) -> None:
        """Reset internal state between games.

        Call this when starting a new game to clear caches.
        """
        self._last_visit_distribution = None
        self._last_policy_distribution = None
        if hasattr(self._underlying_ai, 'reset'):
            self._underlying_ai.reset()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"harness={self.config.harness_type.value}, "
            f"model={self.config.model_type.value}, "
            f"id={self.config.model_id})"
        )
