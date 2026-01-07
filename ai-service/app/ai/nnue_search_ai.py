"""NNUE-based Search AI implementations.

This module provides AI classes that combine NNUE models with various search
algorithms (Gumbel MCTS, MCTS, BRS, MaxN). These allow NNUE to be used in
the same way as standard neural network models for Elo evaluation.

The key insight is that NNUE models (especially those with policy heads)
can replace CNN models in search algorithms, providing different
speed/accuracy tradeoffs.

Usage:
    from app.ai.nnue_search_ai import NNUEGumbelAI

    ai = NNUEGumbelAI(player_number=1, config=AIConfig(difficulty=8))
    move = ai.select_move(game_state)

December 2025: Created for Unified AI Evaluation Architecture.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.ai.base import BaseAI
from app.models import AIConfig, BoardType, GameState, Move

if TYPE_CHECKING:
    from app.ai.nnue_adapter import NNUEMCTSAdapter, NNUEWithPolicyAdapter

logger = logging.getLogger(__name__)


# =============================================================================
# NNUE Model Discovery
# =============================================================================


def find_nnue_model(board_type: BoardType, num_players: int) -> Path | None:
    """Find the canonical NNUE model for a board/player configuration.

    Searches common model paths for NNUE checkpoints.

    Args:
        board_type: Board type (enum or string)
        num_players: Number of players

    Returns:
        Path to NNUE model if found, None otherwise
    """
    from app.models.discovery import find_canonical_model_path

    # Handle both BoardType enum and string arguments
    if hasattr(board_type, "value"):
        board_type_value = board_type.value
        board_type_name = board_type.name.lower()
    else:
        # board_type is a string
        board_type_value = str(board_type).lower()
        board_type_name = str(board_type).lower()

    # Try various NNUE naming conventions
    prefixes = [
        f"nnue_{board_type_value}_{num_players}p",
        f"nnue_{board_type_name}_{num_players}p",
        f"canonical_nnue_{board_type_value}_{num_players}p",
    ]

    for prefix in prefixes:
        path = find_canonical_model_path(prefix)
        if path and path.exists():
            return path

    return None


def get_nnue_adapter(
    board_type: BoardType,
    num_players: int,
    model_path: str | Path | None = None,
) -> "NNUEMCTSAdapter | NNUEWithPolicyAdapter":
    """Get or create an NNUE adapter for search.

    Args:
        board_type: Board type
        num_players: Number of players
        model_path: Optional explicit model path

    Returns:
        NNUE adapter ready for search

    Raises:
        FileNotFoundError: If no NNUE model is found
    """
    from app.ai.nnue_adapter import create_nnue_for_search

    if model_path is None:
        model_path = find_nnue_model(board_type, num_players)
        if model_path is None:
            raise FileNotFoundError(
                f"No NNUE model found for {board_type.value} {num_players}p. "
                "Train an NNUE model or specify model_path explicitly."
            )

    return create_nnue_for_search(
        model_path=model_path,
        board_type=board_type,
        num_players=num_players,
    )


# =============================================================================
# NNUE + Gumbel MCTS
# =============================================================================


class NNUEGumbelAI(BaseAI):
    """NNUE-powered Gumbel MCTS AI.

    Combines NNUE evaluation (value + optional policy) with Gumbel MCTS
    search. This allows NNUE models to be used for high-quality move
    selection with the same interface as CNN-based Gumbel MCTS.

    The NNUE model can be:
    - Value-only: Policy is derived from position evaluations (slower)
    - With policy head: Uses native policy (faster, recommended)

    Attributes:
        _nnue_adapter: The NNUE adapter providing evaluate_batch interface
        _gumbel_ai: The underlying GumbelMCTSAI using the adapter
    """

    AI_NAME = "NNUE_GUMBEL"
    AI_VERSION = "1.0.0"

    def __init__(
        self,
        player_number: int,
        config: AIConfig,
        board_type: BoardType | None = None,
        num_players: int = 2,
        model_path: str | Path | None = None,
    ):
        """Initialize NNUE Gumbel AI.

        Args:
            player_number: Player number (1-indexed)
            config: AI configuration
            board_type: Board type (detected from config if None)
            num_players: Number of players
            model_path: Optional explicit NNUE model path
        """
        super().__init__(player_number, config)

        self._board_type = board_type or BoardType.SQUARE8
        self._num_players = num_players
        self._model_path = model_path

        # Lazy initialization
        self._nnue_adapter = None
        self._gumbel_ai = None

    def _ensure_initialized(self, game_state: GameState) -> None:
        """Ensure NNUE adapter and Gumbel AI are initialized."""
        if self._gumbel_ai is not None:
            return

        # Detect board type from game state if not set
        if hasattr(game_state, "board") and hasattr(game_state.board, "type"):
            self._board_type = game_state.board.type

        # Create NNUE adapter
        try:
            self._nnue_adapter = get_nnue_adapter(
                board_type=self._board_type,
                num_players=self._num_players,
                model_path=self._model_path,
            )
        except FileNotFoundError:
            logger.warning(
                f"NNUE model not found for {self._board_type.value}, "
                "falling back to heuristic"
            )
            self._nnue_adapter = None

        # Create GumbelMCTSAI with NNUE adapter as neural_net
        from app.ai.gumbel_mcts_ai import GumbelMCTSAI

        # Configure Gumbel based on difficulty
        budget = self._get_simulation_budget()

        self._gumbel_ai = GumbelMCTSAI(
            neural_net=self._nnue_adapter,
            simulation_budget=budget,
            num_sampled_actions=min(16, budget // 10),
            c_puct=1.5,
            dirichlet_alpha=0.3,
        )

        logger.info(
            f"NNUEGumbelAI initialized: {self._board_type.value}, "
            f"budget={budget}, has_nnue={self._nnue_adapter is not None}"
        )

    def _get_simulation_budget(self) -> int:
        """Get simulation budget based on config."""
        from app.ai.gumbel_common import (
            GUMBEL_BUDGET_QUALITY,
            GUMBEL_BUDGET_STANDARD,
            GUMBEL_BUDGET_THROUGHPUT,
        )

        difficulty = getattr(self.config, "difficulty", 5)

        if difficulty >= 9:
            return GUMBEL_BUDGET_QUALITY
        elif difficulty >= 7:
            return GUMBEL_BUDGET_STANDARD
        else:
            return GUMBEL_BUDGET_THROUGHPUT

    def select_move(self, game_state: GameState) -> Move:
        """Select the best move using NNUE + Gumbel MCTS.

        Args:
            game_state: Current game state

        Returns:
            Selected move
        """
        self._ensure_initialized(game_state)

        if self._gumbel_ai is None:
            # Fallback to heuristic
            from app.ai.heuristic_ai import HeuristicAI

            fallback = HeuristicAI(self.player_number, self.config)
            return fallback.select_move(game_state)

        return self._gumbel_ai.select_move(game_state)

    def select_move_with_details(
        self, game_state: GameState
    ) -> tuple[Move, dict[str, Any]]:
        """Select move with detailed search information.

        Args:
            game_state: Current game state

        Returns:
            Tuple of (move, details dict)
        """
        self._ensure_initialized(game_state)

        if self._gumbel_ai is None:
            move = self.select_move(game_state)
            return move, {"fallback": True}

        return self._gumbel_ai.select_move_with_details(game_state)

    def evaluate_position(self, game_state: GameState) -> float:
        """Evaluate position using NNUE.

        Args:
            game_state: Current game state

        Returns:
            Position value in [-1, 1]
        """
        self._ensure_initialized(game_state)

        if self._nnue_adapter is None:
            from app.ai.heuristic_ai import HeuristicAI

            fallback = HeuristicAI(self.player_number, self.config)
            return fallback.evaluate_position(game_state)

        values, _ = self._nnue_adapter.evaluate_batch(
            [game_state], value_head=self.player_number
        )
        return values[0] if values else 0.0


# =============================================================================
# NNUE + Standard MCTS
# =============================================================================


class NNUEMCTSAI(BaseAI):
    """NNUE-powered standard MCTS AI.

    Uses NNUE for position evaluation within MCTS tree search.
    """

    AI_NAME = "NNUE_MCTS"
    AI_VERSION = "1.0.0"

    def __init__(
        self,
        player_number: int,
        config: AIConfig,
        board_type: BoardType | None = None,
        num_players: int = 2,
        model_path: str | Path | None = None,
    ):
        super().__init__(player_number, config)
        self._board_type = board_type or BoardType.SQUARE8
        self._num_players = num_players
        self._model_path = model_path
        self._nnue_adapter = None
        self._mcts_ai = None

    def _ensure_initialized(self, game_state: GameState) -> None:
        if self._mcts_ai is not None:
            return

        if hasattr(game_state, "board") and hasattr(game_state.board, "type"):
            self._board_type = game_state.board.type

        try:
            self._nnue_adapter = get_nnue_adapter(
                board_type=self._board_type,
                num_players=self._num_players,
                model_path=self._model_path,
            )
        except FileNotFoundError:
            logger.warning("NNUE model not found, falling back to heuristic")
            self._nnue_adapter = None

        from app.ai.mcts_ai import MCTSAI

        self._mcts_ai = MCTSAI(
            self.player_number,
            self.config,
            neural_net=self._nnue_adapter,
        )

    def select_move(self, game_state: GameState) -> Move:
        self._ensure_initialized(game_state)

        if self._mcts_ai is None:
            from app.ai.heuristic_ai import HeuristicAI

            return HeuristicAI(self.player_number, self.config).select_move(game_state)

        return self._mcts_ai.select_move(game_state)

    def evaluate_position(self, game_state: GameState) -> float:
        """Evaluate position using NNUE."""
        self._ensure_initialized(game_state)

        if self._nnue_adapter is None:
            from app.ai.heuristic_ai import HeuristicAI

            return HeuristicAI(self.player_number, self.config).evaluate_position(game_state)

        values, _ = self._nnue_adapter.evaluate_batch(
            [game_state], value_head=self.player_number
        )
        return values[0] if values else 0.0


# =============================================================================
# NNUE + Best Reply Search (Multiplayer)
# =============================================================================


class NNUEBRSAI(BaseAI):
    """NNUE-powered Best Reply Search AI for multiplayer games.

    BRS is faster than MaxN but less accurate. Good for 3-4 player games
    where computational budget is limited.
    """

    AI_NAME = "NNUE_BRS"
    AI_VERSION = "1.0.0"

    def __init__(
        self,
        player_number: int,
        config: AIConfig,
        board_type: BoardType | None = None,
        num_players: int = 3,
        model_path: str | Path | None = None,
    ):
        super().__init__(player_number, config)
        self._board_type = board_type or BoardType.SQUARE8
        self._num_players = max(num_players, 3)  # BRS is for 3+ players
        self._model_path = model_path
        self._nnue_adapter = None
        self._brs_ai = None

    def _ensure_initialized(self, game_state: GameState) -> None:
        if self._brs_ai is not None:
            return

        if hasattr(game_state, "board") and hasattr(game_state.board, "type"):
            self._board_type = game_state.board.type

        try:
            self._nnue_adapter = get_nnue_adapter(
                board_type=self._board_type,
                num_players=self._num_players,
                model_path=self._model_path,
            )
        except FileNotFoundError:
            logger.warning("NNUE model not found, falling back to heuristic BRS")
            self._nnue_adapter = None

        from app.ai.maxn_ai import BRSAI

        # BRS with NNUE evaluation function
        self._brs_ai = BRSAI(
            self.player_number,
            self.config,
            evaluator=self._create_evaluator(),
        )

    def _create_evaluator(self):
        """Create an evaluation function using NNUE."""
        if self._nnue_adapter is None:
            return None

        def nnue_evaluator(game_state: GameState, player: int) -> float:
            """Evaluate position from player's perspective using NNUE."""
            values, _ = self._nnue_adapter.evaluate_batch(
                [game_state], value_head=player
            )
            return values[0] if values else 0.0

        return nnue_evaluator

    def select_move(self, game_state: GameState) -> Move:
        self._ensure_initialized(game_state)

        if self._brs_ai is None:
            from app.ai.heuristic_ai import HeuristicAI

            return HeuristicAI(self.player_number, self.config).select_move(game_state)

        return self._brs_ai.select_move(game_state)

    def evaluate_position(self, game_state: GameState) -> float:
        """Evaluate position using NNUE."""
        self._ensure_initialized(game_state)

        if self._nnue_adapter is None:
            from app.ai.heuristic_ai import HeuristicAI

            return HeuristicAI(self.player_number, self.config).evaluate_position(game_state)

        values, _ = self._nnue_adapter.evaluate_batch(
            [game_state], value_head=self.player_number
        )
        return values[0] if values else 0.0


# =============================================================================
# NNUE + MaxN Search (Multiplayer)
# =============================================================================


class NNUEMaxNAI(BaseAI):
    """NNUE-powered MaxN Search AI for multiplayer games.

    MaxN is the most accurate multiplayer search but also the slowest.
    Each player maximizes their own score independently.
    """

    AI_NAME = "NNUE_MAXN"
    AI_VERSION = "1.0.0"

    def __init__(
        self,
        player_number: int,
        config: AIConfig,
        board_type: BoardType | None = None,
        num_players: int = 3,
        model_path: str | Path | None = None,
    ):
        super().__init__(player_number, config)
        self._board_type = board_type or BoardType.SQUARE8
        self._num_players = max(num_players, 3)
        self._model_path = model_path
        self._nnue_adapter = None
        self._maxn_ai = None

    def _ensure_initialized(self, game_state: GameState) -> None:
        if self._maxn_ai is not None:
            return

        if hasattr(game_state, "board") and hasattr(game_state.board, "type"):
            self._board_type = game_state.board.type

        try:
            self._nnue_adapter = get_nnue_adapter(
                board_type=self._board_type,
                num_players=self._num_players,
                model_path=self._model_path,
            )
        except FileNotFoundError:
            logger.warning("NNUE model not found, falling back to heuristic MaxN")
            self._nnue_adapter = None

        from app.ai.maxn_ai import MaxNAI

        self._maxn_ai = MaxNAI(
            self.player_number,
            self.config,
            evaluator=self._create_evaluator(),
        )

    def _create_evaluator(self):
        """Create an evaluation function using NNUE."""
        if self._nnue_adapter is None:
            return None

        def nnue_evaluator(game_state: GameState, player: int) -> float:
            values, _ = self._nnue_adapter.evaluate_batch(
                [game_state], value_head=player
            )
            return values[0] if values else 0.0

        return nnue_evaluator

    def select_move(self, game_state: GameState) -> Move:
        self._ensure_initialized(game_state)

        if self._maxn_ai is None:
            from app.ai.heuristic_ai import HeuristicAI

            return HeuristicAI(self.player_number, self.config).select_move(game_state)

        return self._maxn_ai.select_move(game_state)

    def evaluate_position(self, game_state: GameState) -> float:
        """Evaluate position using NNUE."""
        self._ensure_initialized(game_state)

        if self._nnue_adapter is None:
            from app.ai.heuristic_ai import HeuristicAI

            return HeuristicAI(self.player_number, self.config).evaluate_position(game_state)

        values, _ = self._nnue_adapter.evaluate_batch(
            [game_state], value_head=self.player_number
        )
        return values[0] if values else 0.0
