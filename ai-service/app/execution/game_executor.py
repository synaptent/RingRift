"""Unified Game Executor for RingRift.

This module provides a centralized game execution framework that consolidates
the game-playing logic previously scattered across tournament scripts, selfplay
runners, and other orchestrators. All game execution should use this module
to ensure consistent behavior, proper result collection, and training data
generation.

Usage:
    from app.execution.game_executor import GameExecutor, GameResult

    # Create executor
    executor = GameExecutor(board_type="square8", num_players=2)

    # Run a single game
    result = executor.run_game(
        player_configs=[
            {"ai_type": "mcts", "difficulty": 5},
            {"ai_type": "heuristic", "difficulty": 3},
        ],
        max_moves=10000,
    )

    # Run multiple games
    results = executor.run_games(
        num_games=100,
        player_configs=[...],
        progress_callback=lambda i, n: print(f"{i}/{n}"),
    )

    # Run games with specific AI instances
    from app.ai.factory import AIFactory
    ai1 = AIFactory.create_from_difficulty(5, player_number=1)
    ai2 = AIFactory.create_from_difficulty(3, player_number=2)
    result = executor.run_game_with_ais([ai1, ai2])
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.ai.base import BaseAI
    from app.models import GameState, Move

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Result types
# -----------------------------------------------------------------------------


class GameOutcome(str, Enum):
    """Possible game outcomes."""
    WIN = "win"  # A player won normally
    DRAW = "draw"  # Game ended in a draw
    TIMEOUT = "timeout"  # Max moves reached, winner by tiebreak
    ERROR = "error"  # Game ended due to an error
    FORFEIT = "forfeit"  # A player forfeited (e.g., no valid moves)


@dataclass
class GameResult:
    """Result of a single game execution.

    This dataclass provides a standardized format for game results,
    suitable for tournament tracking, Elo calculation, and training
    data generation.
    """
    # Core identifiers
    game_id: str
    board_type: str
    num_players: int

    # Outcome
    winner: int | None  # 1-indexed player number, None for draw
    outcome: GameOutcome
    move_count: int

    # Player information
    player_types: list[str]  # AI type names
    player_configs: list[dict[str, Any]]  # Full configs

    # Game data (for training)
    moves: list[dict[str, Any]] = field(default_factory=list)
    initial_state: dict[str, Any] | None = None
    final_state: dict[str, Any] | None = None

    # Timing
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_seconds: float = 0.0

    # Metadata
    seed: int | None = None
    source: str = "GameExecutor"
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "game_id": self.game_id,
            "board_type": self.board_type,
            "num_players": self.num_players,
            "winner": self.winner,
            "outcome": self.outcome.value,
            "move_count": self.move_count,
            "player_types": self.player_types,
            "player_configs": self.player_configs,
            "moves": self.moves,
            "initial_state": self.initial_state,
            "final_state": self.final_state,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "seed": self.seed,
            "source": self.source,
            "error_message": self.error_message,
        }

    def to_training_record(self) -> dict[str, Any]:
        """Convert to training data record format.

        This format is compatible with the NPZ export pipeline
        and the existing selfplay database schema.
        """
        return {
            "game_id": self.game_id,
            "board_type": self.board_type,
            "num_players": self.num_players,
            "winner": self.winner,
            "move_count": self.move_count,
            "status": "completed" if self.outcome != GameOutcome.ERROR else "error",
            "game_status": "completed" if self.winner else "draw",
            "moves": self.moves,
            "initial_state": self.initial_state,
            "timestamp": self.end_time.isoformat() if self.end_time else datetime.now().isoformat(),
            "created_at": self.start_time.isoformat() if self.start_time else datetime.now().isoformat(),
            "source": self.source,
            "opponent_type": "selfplay",
            "player_types": self.player_types,
            "seed": self.seed,
        }


# -----------------------------------------------------------------------------
# Game Executor
# -----------------------------------------------------------------------------


class GameExecutor:
    """Unified game execution engine.

    This class provides a consistent interface for running games,
    regardless of the underlying AI types or execution backend.
    It handles:
    - Game state initialization
    - Move selection and application
    - Timeout/max-move handling with tiebreakers
    - Result collection and training data generation
    """

    def __init__(
        self,
        board_type: str = "square8",
        num_players: int = 2,
        *,
        max_moves: int = 10000,
        verbose: bool = False,
        source_name: str = "GameExecutor",
    ):
        """Initialize the game executor.

        Args:
            board_type: Board type (square8, square19, hexagonal)
            num_players: Number of players (2, 3, or 4)
            max_moves: Default maximum moves before timeout
            verbose: Enable verbose logging
            source_name: Source identifier for tracking
        """
        self.board_type = board_type
        self.num_players = num_players
        self.max_moves = max_moves
        self.verbose = verbose
        self.source_name = source_name

        # Lazy-loaded dependencies
        self._rules_engine = None
        self._create_initial_state = None

    @property
    def rules_engine(self):
        """Lazy-load rules engine."""
        if self._rules_engine is None:
            from app.rules.default_engine import DefaultRulesEngine
            self._rules_engine = DefaultRulesEngine()
        return self._rules_engine

    def _get_create_initial_state(self):
        """Lazy-load state creation function."""
        if self._create_initial_state is None:
            from app.training.generate_data import create_initial_state
            self._create_initial_state = create_initial_state
        return self._create_initial_state

    def _create_game_state(self, seed: int | None = None) -> GameState:
        """Create a new game state.

        Args:
            seed: Optional random seed for reproducibility

        Returns:
            Fresh game state
        """
        from app.models import BoardType

        board_type_enum = getattr(BoardType, self.board_type.upper(), BoardType.SQUARE8)
        return self._get_create_initial_state()(
            board_type=board_type_enum,
            num_players=self.num_players,
        )

    def _compute_tiebreak_winner(self, game_state: GameState) -> int:
        """Compute winner for timeout/draw scenarios.

        Uses a deterministic tiebreak based on:
        1. Territory spaces
        2. Eliminated rings (opponent's)
        3. Marker count
        4. Last actor
        5. Player number (lower wins ties)

        Args:
            game_state: Final game state

        Returns:
            Winner player number (1-indexed)
        """
        players = getattr(game_state, "players", None) or []
        if not players:
            return 1

        # Count markers per player
        marker_counts: dict[int, int] = {i + 1: 0 for i in range(len(players))}
        try:
            for marker in game_state.board.markers.values():
                owner = int(marker.player)
                marker_counts[owner] = marker_counts.get(owner, 0) + 1
        except (AttributeError, ValueError, TypeError, KeyError):
            pass

        # Get last actor
        last_actor = None
        try:
            if game_state.move_history:
                last_actor = game_state.move_history[-1].player
        except (AttributeError, IndexError, TypeError):
            pass

        # Score each player
        best_player: int | None = None
        best_key: tuple | None = None

        for idx, player in enumerate(players):
            player_num = getattr(player, "player_number", idx + 1)
            pid = int(player_num)

            try:
                eliminated = int(getattr(player, "eliminated_rings", 0) or 0)
            except (ValueError, TypeError):
                eliminated = 0

            try:
                territory = int(getattr(player, "territory_spaces", 0) or 0)
            except (ValueError, TypeError):
                territory = 0

            markers = marker_counts.get(pid, 0)
            last = 1 if last_actor == pid else 0

            # Tiebreak key: higher is better, except player number
            key = (territory, eliminated, markers, last, -pid)

            if best_key is None or key > best_key:
                best_key = key
                best_player = pid

        return best_player or 1

    def _serialize_move(self, move: Move) -> dict[str, Any]:
        """Serialize a move for storage.

        Args:
            move: Move object

        Returns:
            Serialized move dictionary
        """
        if hasattr(move, "model_dump"):
            return move.model_dump(mode="json")
        elif hasattr(move, "to_dict"):
            return move.to_dict()
        else:
            # Manual serialization
            result: dict[str, Any] = {
                "type": move.type.value if hasattr(move.type, "value") else str(move.type),
                "player": move.player,
            }
            if hasattr(move, "to") and move.to is not None:
                result["to"] = {"x": move.to.x, "y": move.to.y}
            if hasattr(move, "from_pos") and move.from_pos is not None:
                result["from"] = {"x": move.from_pos.x, "y": move.from_pos.y}
            return result

    def run_game_with_ais(
        self,
        ais: list[BaseAI],
        *,
        max_moves: int | None = None,
        seed: int | None = None,
        game_id: str | None = None,
    ) -> GameResult:
        """Run a game with pre-created AI instances.

        Args:
            ais: List of AI instances (one per player)
            max_moves: Maximum moves (uses default if None)
            seed: Random seed for reproducibility
            game_id: Optional game ID (auto-generated if None)

        Returns:
            GameResult with outcome and game data
        """
        if len(ais) != self.num_players:
            raise ValueError(
                f"Expected {self.num_players} AIs, got {len(ais)}"
            )

        max_moves = max_moves or self.max_moves
        game_id = game_id or f"{self.source_name}_{uuid.uuid4().hex[:12]}"

        # Create game state
        game_state = self._create_game_state(seed=seed)
        initial_state = game_state.model_dump(mode="json")

        # Extract player info
        player_types = [ai.__class__.__name__ for ai in ais]
        player_configs = [
            {
                "type": ai.__class__.__name__,
                "difficulty": getattr(ai.config, "difficulty", 5) if hasattr(ai, "config") else 5,
            }
            for ai in ais
        ]

        # Run game loop
        start_time = datetime.now()
        moves_played: list[dict[str, Any]] = []
        move_count = 0
        outcome = GameOutcome.WIN
        error_message = None

        while game_state.game_status == "active" and move_count < max_moves:
            current_player = game_state.current_player
            current_ai = ais[current_player - 1]

            # Ensure AI has correct player number
            current_ai.player_number = current_player

            try:
                # Get move from AI
                move = current_ai.select_move(game_state)

                if move is None:
                    # Check for bookkeeping moves
                    from app.game_engine import GameEngine
                    requirement = GameEngine.get_phase_requirement(
                        game_state, current_player
                    )
                    if requirement is not None:
                        move = GameEngine.synthesize_bookkeeping_move(
                            requirement, game_state
                        )
                    else:
                        # No valid moves - forfeit
                        winner = self._get_opponent(current_player)
                        outcome = GameOutcome.FORFEIT
                        break

                # Apply move
                game_state = self.rules_engine.apply_move(game_state, move)
                moves_played.append(self._serialize_move(move))
                move_count += 1

                if self.verbose and move_count % 100 == 0:
                    logger.debug(f"Game {game_id}: {move_count} moves")

            except Exception as e:
                logger.error(f"Error in game {game_id}: {e}")
                error_message = str(e)
                outcome = GameOutcome.ERROR
                break

        end_time = datetime.now()

        # Determine winner
        if outcome == GameOutcome.ERROR:
            winner = None
        elif game_state.game_status != "active":
            winner = game_state.winner
            outcome = GameOutcome.WIN if winner else GameOutcome.DRAW
        elif move_count >= max_moves:
            winner = self._compute_tiebreak_winner(game_state)
            outcome = GameOutcome.TIMEOUT
        else:
            winner = game_state.winner

        return GameResult(
            game_id=game_id,
            board_type=self.board_type,
            num_players=self.num_players,
            winner=winner,
            outcome=outcome,
            move_count=move_count,
            player_types=player_types,
            player_configs=player_configs,
            moves=moves_played,
            initial_state=initial_state,
            final_state=game_state.model_dump(mode="json") if game_state else None,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=(end_time - start_time).total_seconds(),
            seed=seed,
            source=self.source_name,
            error_message=error_message,
        )

    def _get_opponent(self, player: int) -> int:
        """Get the opponent player number for 2-player games."""
        return 2 if player == 1 else 1

    def run_game(
        self,
        player_configs: list[dict[str, Any]],
        *,
        max_moves: int | None = None,
        seed: int | None = None,
        game_id: str | None = None,
    ) -> GameResult:
        """Run a game with player configurations.

        Args:
            player_configs: List of player configs, each with:
                - ai_type: AI type string (e.g., "mcts", "heuristic")
                - difficulty: Difficulty level (1-10)
                - Optional: rng_seed, think_time, etc.
            max_moves: Maximum moves (uses default if None)
            seed: Random seed for game state
            game_id: Optional game ID

        Returns:
            GameResult with outcome and game data
        """
        from app.ai.factory import AIFactory

        # Create AI instances
        ais = []
        for i, config in enumerate(player_configs):
            player_number = i + 1
            ai_type = config.get("ai_type", "heuristic")
            config.get("difficulty", 5)

            ai = AIFactory.create_for_tournament(
                agent_id=ai_type,
                player_number=player_number,
                board_type=self.board_type,
                num_players=self.num_players,
                rng_seed=config.get("rng_seed"),
                nn_model_id=config.get("nn_model_id"),
            )
            ais.append(ai)

        return self.run_game_with_ais(
            ais,
            max_moves=max_moves,
            seed=seed,
            game_id=game_id,
        )

    def run_games(
        self,
        num_games: int,
        player_configs: list[dict[str, Any]],
        *,
        max_moves: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        error_callback: Callable[[int, Exception], None] | None = None,
    ) -> list[GameResult]:
        """Run multiple games.

        Args:
            num_games: Number of games to run
            player_configs: Player configurations
            max_moves: Maximum moves per game
            progress_callback: Optional callback(current, total)
            error_callback: Optional callback(game_index, exception)

        Returns:
            List of GameResult objects
        """
        results = []

        for i in range(num_games):
            try:
                result = self.run_game(
                    player_configs=player_configs,
                    max_moves=max_moves,
                    seed=i,
                    game_id=f"{self.source_name}_{i}_{uuid.uuid4().hex[:8]}",
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Game {i} failed: {e}")
                if error_callback:
                    error_callback(i, e)
                # Create error result
                results.append(GameResult(
                    game_id=f"{self.source_name}_{i}_error",
                    board_type=self.board_type,
                    num_players=self.num_players,
                    winner=None,
                    outcome=GameOutcome.ERROR,
                    move_count=0,
                    player_types=[],
                    player_configs=player_configs,
                    error_message=str(e),
                    source=self.source_name,
                ))

            if progress_callback:
                progress_callback(i + 1, num_games)

        return results

    def run_selfplay_games(
        self,
        num_games: int,
        difficulty: int = 5,
        ai_type: str = "mcts",
        *,
        max_moves: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[GameResult]:
        """Run self-play games (same AI vs itself).

        This is a convenience method for generating training data.

        Args:
            num_games: Number of games
            difficulty: AI difficulty level
            ai_type: AI type to use
            max_moves: Maximum moves per game
            progress_callback: Progress callback

        Returns:
            List of GameResult objects
        """
        player_configs = [
            {"ai_type": ai_type, "difficulty": difficulty}
            for _ in range(self.num_players)
        ]
        return self.run_games(
            num_games=num_games,
            player_configs=player_configs,
            max_moves=max_moves,
            progress_callback=progress_callback,
        )


# -----------------------------------------------------------------------------
# Parallel Executor
# -----------------------------------------------------------------------------


class ParallelGameExecutor:
    """Parallel game executor using worker threads or processes.

    This executor runs multiple games concurrently using a thread pool.
    For GPU-accelerated games, use the ParallelGameRunner from
    gpu_parallel_games.py instead.
    """

    def __init__(
        self,
        board_type: str = "square8",
        num_players: int = 2,
        max_workers: int = 4,
        **kwargs,
    ):
        """Initialize parallel executor.

        Args:
            board_type: Board type
            num_players: Number of players
            max_workers: Maximum concurrent workers
            **kwargs: Additional arguments passed to GameExecutor
        """
        self.board_type = board_type
        self.num_players = num_players
        self.max_workers = max_workers
        self.executor_kwargs = kwargs

    def run_games(
        self,
        num_games: int,
        player_configs: list[dict[str, Any]],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[GameResult]:
        """Run games in parallel.

        Args:
            num_games: Number of games
            player_configs: Player configurations
            progress_callback: Progress callback

        Returns:
            List of GameResult objects
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: list[GameResult] = []
        completed = 0

        def run_single_game(game_index: int) -> GameResult:
            executor = GameExecutor(
                board_type=self.board_type,
                num_players=self.num_players,
                **self.executor_kwargs,
            )
            return executor.run_game(
                player_configs=player_configs,
                seed=game_index,
            )

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(run_single_game, i): i
                for i in range(num_games)
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Game failed: {e}")
                    results.append(GameResult(
                        game_id=f"error_{futures[future]}",
                        board_type=self.board_type,
                        num_players=self.num_players,
                        winner=None,
                        outcome=GameOutcome.ERROR,
                        move_count=0,
                        player_types=[],
                        player_configs=player_configs,
                        error_message=str(e),
                    ))

                completed += 1
                if progress_callback:
                    progress_callback(completed, num_games)

        return results


# -----------------------------------------------------------------------------
# Convenience functions
# -----------------------------------------------------------------------------


def run_quick_game(
    p1_type: str = "heuristic",
    p2_type: str = "heuristic",
    p1_difficulty: int = 5,
    p2_difficulty: int = 5,
    board_type: str = "square8",
    max_moves: int = 10000,
) -> GameResult:
    """Run a quick game between two AIs.

    Args:
        p1_type: Player 1 AI type
        p2_type: Player 2 AI type
        p1_difficulty: Player 1 difficulty
        p2_difficulty: Player 2 difficulty
        board_type: Board type
        max_moves: Maximum moves

    Returns:
        GameResult
    """
    executor = GameExecutor(board_type=board_type, num_players=2)
    return executor.run_game(
        player_configs=[
            {"ai_type": p1_type, "difficulty": p1_difficulty},
            {"ai_type": p2_type, "difficulty": p2_difficulty},
        ],
        max_moves=max_moves,
    )


def run_selfplay_batch(
    num_games: int,
    ai_type: str = "mcts",
    difficulty: int = 5,
    board_type: str = "square8",
    num_players: int = 2,
    max_workers: int = 4,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[GameResult]:
    """Run a batch of self-play games.

    Args:
        num_games: Number of games
        ai_type: AI type for self-play
        difficulty: AI difficulty
        board_type: Board type
        num_players: Number of players
        max_workers: Parallel workers
        progress_callback: Progress callback

    Returns:
        List of GameResult objects
    """
    executor = ParallelGameExecutor(
        board_type=board_type,
        num_players=num_players,
        max_workers=max_workers,
    )
    player_configs = [
        {"ai_type": ai_type, "difficulty": difficulty}
        for _ in range(num_players)
    ]
    return executor.run_games(
        num_games=num_games,
        player_configs=player_configs,
        progress_callback=progress_callback,
    )
