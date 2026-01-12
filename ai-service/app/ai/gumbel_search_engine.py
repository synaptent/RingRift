"""Unified Gumbel MCTS Search Engine (December 2025).

Provides a single entry point for all Gumbel MCTS search variants, with automatic
mode selection based on use case.

This consolidates the functionality from:
- gumbel_mcts_ai.py (single-game CPU)
- tensor_gumbel_tree.py (single-game GPU)
- batched_gumbel_mcts.py (multi-game batch)
- multi_game_gumbel.py (multi-game parallel GPU)

Usage:
    from app.ai.gumbel_search_engine import GumbelSearchEngine, SearchMode

    # For single game play
    engine = GumbelSearchEngine(
        neural_net=my_nn,
        mode=SearchMode.SINGLE_GAME,
    )
    move = engine.search(game_state)

    # For selfplay (high throughput)
    engine = GumbelSearchEngine(
        neural_net=my_nn,
        mode=SearchMode.MULTI_GAME_PARALLEL,
        num_games=64,
    )
    results = engine.search_batch(initial_states)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from app.ai.gumbel_common import (
    GUMBEL_BUDGET_QUALITY,
    GUMBEL_BUDGET_STANDARD,
    GUMBEL_BUDGET_THROUGHPUT,
    GumbelAction,
)

if TYPE_CHECKING:
    from app.ai.neural_net import NeuralNetAI
    from app.models import BoardType, GameState, Move

logger = logging.getLogger(__name__)


class SearchMode(str, Enum):
    """Search mode selection for GumbelSearchEngine."""

    # Single game modes
    SINGLE_GAME = "single_game"  # Standard single-game search (CPU or GPU)
    SINGLE_GAME_FAST = "single_game_fast"  # Optimized for speed

    # Multi-game modes
    MULTI_GAME_BATCH = "multi_batch"  # Batched evaluation across games
    MULTI_GAME_PARALLEL = "multi_parallel"  # Full parallel GPU (selfplay)

    # Auto-detection
    AUTO = "auto"  # Automatically select based on context


@dataclass
class SearchConfig:
    """Configuration for Gumbel MCTS search."""

    simulation_budget: int = GUMBEL_BUDGET_STANDARD
    num_sampled_actions: int = 16
    temperature: float = 1.0
    temperature_threshold: int = 30  # Switch to greedy after this move
    c_puct: float = 1.5
    use_root_noise: bool = True  # Enable Dirichlet noise at root (AlphaZero-style)
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    @classmethod
    def for_throughput(cls) -> "SearchConfig":
        """Config optimized for throughput (fast selfplay with exploration).

        Uses lower budget for speed but maintains exploration via temperature=1.0.
        """
        return cls(
            simulation_budget=GUMBEL_BUDGET_THROUGHPUT,
            num_sampled_actions=8,
            temperature=1.0,  # Maintain exploration for training data diversity
        )

    @classmethod
    def for_selfplay(cls) -> "SearchConfig":
        """Config for training data generation - prioritizes move diversity.

        Uses high budget with temperature=1.0 to generate diverse training data.
        This is the recommended config for selfplay data generation.
        """
        return cls(
            simulation_budget=GUMBEL_BUDGET_QUALITY,
            num_sampled_actions=16,
            temperature=1.0,  # Exploration for diverse training data
            use_root_noise=True,
        )

    @classmethod
    def for_evaluation(cls) -> "SearchConfig":
        """Config for strength evaluation - prioritizes best moves.

        Uses high budget with temperature=0.0 for deterministic best-move selection.
        Use this for gauntlet evaluation and tournament play.
        """
        return cls(
            simulation_budget=GUMBEL_BUDGET_QUALITY,
            num_sampled_actions=16,
            temperature=0.0,  # Deterministic for consistent strength measurement
            use_root_noise=False,  # No noise for evaluation
        )

    @classmethod
    def for_quality(cls) -> "SearchConfig":
        """DEPRECATED: Use for_selfplay() or for_evaluation() instead.

        This method is ambiguous - 'quality' could mean:
        - Quality training data (needs exploration) -> use for_selfplay()
        - Quality moves for evaluation (needs determinism) -> use for_evaluation()

        Kept for backward compatibility, behaves like for_evaluation().
        """
        logger.warning(
            "SearchConfig.for_quality() is deprecated. "
            "Use for_selfplay() for training data or for_evaluation() for strength testing."
        )
        return cls.for_evaluation()

    @classmethod
    def for_balanced(cls) -> "SearchConfig":
        """Balanced config for general use."""
        return cls(
            simulation_budget=GUMBEL_BUDGET_STANDARD,
            num_sampled_actions=16,
            temperature=1.0,
        )


@dataclass
class SearchResult:
    """Result from a single-game search."""

    move: "Move"
    visit_counts: dict[str, int]  # move -> visit count
    value: float  # Estimated position value
    policy: dict[str, float]  # move -> policy probability


@dataclass
class BatchSearchResult:
    """Result from batch/parallel search."""

    moves: list["Move"]  # One move per game
    game_results: list[dict[str, Any]]  # Detailed results per game


class GumbelSearchEngine:
    """Unified interface for Gumbel MCTS search.

    Automatically selects the appropriate backend based on:
    - mode: Explicit mode selection
    - device: CPU vs GPU
    - batch_size: Single vs multiple games

    Example:
        # Single game
        engine = GumbelSearchEngine(neural_net=nn)
        move = engine.search(state)

        # Batch search
        engine = GumbelSearchEngine(neural_net=nn, mode=SearchMode.MULTI_GAME_PARALLEL)
        results = engine.search_batch(states)
    """

    def __init__(
        self,
        neural_net: "NeuralNetAI | None" = None,
        mode: SearchMode = SearchMode.AUTO,
        config: SearchConfig | None = None,
        device: str = "cuda",
        num_games: int = 64,
        board_type: "BoardType | None" = None,
        num_players: int = 2,
    ):
        """Initialize the search engine.

        Args:
            neural_net: Neural network for position evaluation.
            mode: Search mode (AUTO will select based on context).
            config: Search configuration (uses default if None).
            device: Device for computation ("cpu" or "cuda").
            num_games: Number of games for parallel modes.
            board_type: Board type (for initialization).
            num_players: Number of players.
        """
        self.neural_net = neural_net
        self.mode = mode
        self.config = config or SearchConfig.for_balanced()
        self.device = device
        self.num_games = num_games
        self.board_type = board_type
        self.num_players = num_players

        # Lazy-loaded backends
        self._single_game_ai = None
        self._multi_game_runner = None

    def search(
        self,
        game_state: "GameState",
        move_number: int = 0,
    ) -> "Move":
        """Search for the best move in a single game.

        Args:
            game_state: Current game state.
            move_number: Current move number (for temperature scheduling).

        Returns:
            Best move found.
        """
        ai = self._get_single_game_ai()

        # Apply temperature based on move number
        temp = self.config.temperature
        if move_number >= self.config.temperature_threshold:
            temp = 0.0  # Greedy after threshold

        return ai.select_move(game_state, temperature=temp)

    def search_with_details(
        self,
        game_state: "GameState",
        move_number: int = 0,
    ) -> SearchResult:
        """Search with detailed results including visit counts and policy.

        Args:
            game_state: Current game state.
            move_number: Current move number.

        Returns:
            SearchResult with move, visits, value, and policy.
        """
        ai = self._get_single_game_ai()

        temp = self.config.temperature
        if move_number >= self.config.temperature_threshold:
            temp = 0.0

        move, details = ai.select_move_with_details(game_state, temperature=temp)

        return SearchResult(
            move=move,
            visit_counts=details.get("visit_counts", {}),
            value=details.get("value", 0.0),
            policy=details.get("policy", {}),
        )

    def search_batch(
        self,
        game_states: list["GameState"] | None = None,
        num_games: int | None = None,
    ) -> BatchSearchResult:
        """Search multiple games in parallel.

        Args:
            game_states: Optional initial states (generates new if None).
            num_games: Number of games to run (uses self.num_games if None).

        Returns:
            BatchSearchResult with moves and per-game results.
        """
        runner = self._get_multi_game_runner()
        n = num_games or self.num_games

        results = runner.run_batch(num_games=n, initial_states=game_states)

        return BatchSearchResult(
            moves=[r.moves[-1] if r.moves else None for r in results],
            game_results=[
                {
                    "game_idx": r.game_idx,
                    "winner": r.winner,
                    "move_count": r.move_count,
                    "moves": r.moves,
                }
                for r in results
            ],
        )

    def run_selfplay(
        self,
        num_games: int | None = None,
    ) -> list[dict]:
        """Run selfplay games for training data generation.

        Args:
            num_games: Number of games to play.

        Returns:
            List of game results with moves, winner, etc.
        """
        runner = self._get_multi_game_runner()
        n = num_games or self.num_games

        results = runner.run_batch(num_games=n)

        return [
            {
                "game_idx": r.game_idx,
                "winner": r.winner,
                "status": r.status,
                "move_count": r.move_count,
                "moves": r.moves,
                "duration_ms": r.duration_ms,
            }
            for r in results
        ]

    def _get_single_game_ai(self):
        """Get or create single-game AI backend."""
        if self._single_game_ai is None:
            try:
                from app.ai.gumbel_mcts_ai import GumbelMCTSAI

                self._single_game_ai = GumbelMCTSAI(
                    neural_net=self.neural_net,
                    simulation_budget=self.config.simulation_budget,
                    num_sampled_actions=self.config.num_sampled_actions,
                    c_puct=self.config.c_puct,
                    dirichlet_alpha=self.config.dirichlet_alpha,
                )
            except ImportError as e:
                logger.error(f"Failed to load GumbelMCTSAI: {e}")
                raise

        return self._single_game_ai

    def _get_multi_game_runner(self):
        """Get or create multi-game runner backend."""
        if self._multi_game_runner is None:
            try:
                from app.ai.multi_game_gumbel import MultiGameGumbelRunner

                self._multi_game_runner = MultiGameGumbelRunner(
                    num_games=self.num_games,
                    simulation_budget=self.config.simulation_budget,
                    num_sampled_actions=self.config.num_sampled_actions,
                    neural_net=self.neural_net,
                    device=self.device,
                    board_type=self.board_type,
                    num_players=self.num_players,
                    temperature=self.config.temperature,
                    temperature_threshold=self.config.temperature_threshold,
                )
            except ImportError as e:
                logger.error(f"Failed to load MultiGameGumbelRunner: {e}")
                raise

        return self._multi_game_runner

    @classmethod
    def for_selfplay(
        cls,
        neural_net: "NeuralNetAI",
        board_type: "BoardType",
        num_players: int = 2,
        num_games: int = 64,
        device: str = "cuda",
    ) -> "GumbelSearchEngine":
        """Create an engine optimized for selfplay.

        Args:
            neural_net: Neural network.
            board_type: Board configuration.
            num_players: Number of players.
            num_games: Games to run in parallel.
            device: Computation device.

        Returns:
            Configured GumbelSearchEngine.
        """
        # CRITICAL FIX (Jan 2026): Use for_selfplay() (800 sims) instead of for_throughput() (64 sims)
        # The 64-sim budget was producing garbage training data, causing NNs to be weaker than heuristic
        return cls(
            neural_net=neural_net,
            mode=SearchMode.MULTI_GAME_PARALLEL,
            config=SearchConfig.for_selfplay(),
            device=device,
            num_games=num_games,
            board_type=board_type,
            num_players=num_players,
        )

    @classmethod
    def for_evaluation(
        cls,
        neural_net: "NeuralNetAI",
    ) -> "GumbelSearchEngine":
        """Create an engine optimized for move quality evaluation.

        Args:
            neural_net: Neural network.

        Returns:
            Configured GumbelSearchEngine.
        """
        return cls(
            neural_net=neural_net,
            mode=SearchMode.SINGLE_GAME,
            config=SearchConfig.for_quality(),
        )

    @classmethod
    def for_play(
        cls,
        neural_net: "NeuralNetAI",
    ) -> "GumbelSearchEngine":
        """Create an engine for interactive play.

        Args:
            neural_net: Neural network.

        Returns:
            Configured GumbelSearchEngine.
        """
        return cls(
            neural_net=neural_net,
            mode=SearchMode.SINGLE_GAME,
            config=SearchConfig.for_balanced(),
        )


# Convenience factory functions
def create_selfplay_engine(
    neural_net: "NeuralNetAI",
    board_type: "BoardType",
    num_players: int = 2,
    num_games: int = 64,
    device: str = "cuda",
) -> GumbelSearchEngine:
    """Create a search engine configured for selfplay.

    Args:
        neural_net: Neural network for evaluation.
        board_type: Board configuration.
        num_players: Number of players.
        num_games: Games to run in parallel.
        device: Computation device.

    Returns:
        Configured GumbelSearchEngine.
    """
    return GumbelSearchEngine.for_selfplay(
        neural_net=neural_net,
        board_type=board_type,
        num_players=num_players,
        num_games=num_games,
        device=device,
    )


def create_evaluation_engine(neural_net: "NeuralNetAI") -> GumbelSearchEngine:
    """Create a search engine configured for high-quality evaluation.

    Args:
        neural_net: Neural network for evaluation.

    Returns:
        Configured GumbelSearchEngine.
    """
    return GumbelSearchEngine.for_evaluation(neural_net)


def create_play_engine(neural_net: "NeuralNetAI") -> GumbelSearchEngine:
    """Create a search engine configured for interactive play.

    Args:
        neural_net: Neural network for evaluation.

    Returns:
        Configured GumbelSearchEngine.
    """
    return GumbelSearchEngine.for_play(neural_net)
