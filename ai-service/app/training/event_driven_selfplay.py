"""Event-driven selfplay that automatically updates models on promotion.

This module provides a wrapper around selfplay that:
1. Subscribes to MODEL_PROMOTED events
2. Hot-reloads new models when promoted
3. Gracefully transitions between model versions

Usage:
    # Create event-driven selfplay manager
    manager = EventDrivenSelfplay(
        board_type="square8",
        num_players=2,
        batch_size=16,
    )

    # Start selfplay (runs until stopped)
    await manager.run()

    # Or run a specific number of games
    games = await manager.run_games(num_games=1000)

Integration with unified_ai_loop:
    The manager can be wired into the unified AI loop to automatically
    use the latest trained models for data generation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from app.ai.batched_gumbel_mcts import BatchedGumbelMCTS

logger = logging.getLogger(__name__)


@dataclass
class SelfplayStats:
    """Statistics for selfplay session."""

    games_completed: int = 0
    total_moves: int = 0
    model_reloads: int = 0
    current_model: str = ""
    start_time: float = field(default_factory=time.time)
    wins_by_player: dict[int, int] = field(default_factory=dict)

    @property
    def games_per_second(self) -> float:
        elapsed = time.time() - self.start_time
        return self.games_completed / max(elapsed, 1e-6)

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time


class EventDrivenSelfplay:
    """Selfplay manager that responds to model promotion events.

    This class manages selfplay workers and automatically updates them
    when new models are promoted to production.
    """

    def __init__(
        self,
        board_type: str,
        num_players: int = 2,
        batch_size: int = 16,
        mcts_sims: int = 800,
        max_moves: int = 500,
        output_dir: str | Path = "data/selfplay/event_driven",
        prefer_nnue: bool = True,
        use_gpu_mcts: bool = False,
        gpu_device: str = "cuda",
        gpu_eval_mode: str = "heuristic",
    ):
        """Initialize event-driven selfplay manager.

        Args:
            board_type: Board type (square8, hex8, etc.)
            num_players: Number of players (2, 3, or 4)
            batch_size: Number of games to run in parallel
            mcts_sims: MCTS simulation budget per move
            max_moves: Maximum moves per game before termination
            output_dir: Directory for game output
            prefer_nnue: Prefer NNUE models over policy/value nets
            use_gpu_mcts: Use GPU-accelerated MultiTreeMCTS (Phase 3)
            gpu_device: Device for GPU MCTS (cuda, cpu, mps)
            gpu_eval_mode: Evaluation mode for GPU MCTS (heuristic, nn, hybrid)
        """
        self.board_type = board_type
        self.num_players = num_players
        self.batch_size = batch_size
        self.mcts_sims = mcts_sims
        self.max_moves = max_moves
        self.output_dir = Path(output_dir)
        self.prefer_nnue = prefer_nnue
        self.use_gpu_mcts = use_gpu_mcts
        self.gpu_device = gpu_device
        self.gpu_eval_mode = gpu_eval_mode

        self._config_key = f"{board_type.lower()}_{num_players}p"
        self._batched_mcts: BatchedGumbelMCTS | None = None
        self._multi_tree_mcts = None  # MultiTreeMCTS instance
        self._current_model_path: Path | None = None
        self._model_update_pending = False
        self._pending_model_path: Path | None = None
        self._running = False
        self._stats = SelfplayStats()
        self._event_subscription = None
        self._callbacks: list[Callable[[Path], None]] = []

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"EventDrivenSelfplay initialized: config={self._config_key}, "
            f"batch_size={batch_size}, mcts_sims={mcts_sims}"
        )

    async def start(self) -> None:
        """Start the event-driven selfplay manager.

        This initializes the MCTS engine and subscribes to model events.
        """
        if self._running:
            logger.warning("EventDrivenSelfplay already running")
            return

        # Initialize model selector and get current model
        from app.training.selfplay_model_selector import SelfplayModelSelector

        selector = SelfplayModelSelector(
            board_type=self.board_type,
            num_players=self.num_players,
            prefer_nnue=self.prefer_nnue,
        )
        self._current_model_path = selector.get_current_model()

        # Initialize batched MCTS
        await self._initialize_mcts()

        # Subscribe to model promotion events
        self._subscribe_to_events()

        self._running = True
        self._stats = SelfplayStats(current_model=str(self._current_model_path or ""))

        logger.info(
            f"EventDrivenSelfplay started with model: {self._current_model_path}"
        )

    async def stop(self) -> None:
        """Stop the event-driven selfplay manager."""
        self._running = False
        self._unsubscribe_from_events()

        logger.info(
            f"EventDrivenSelfplay stopped. Stats: "
            f"{self._stats.games_completed} games, "
            f"{self._stats.games_per_second:.2f} g/s, "
            f"{self._stats.model_reloads} model reloads"
        )

    async def run_games(self, num_games: int) -> list[dict]:
        """Run a specific number of games.

        Args:
            num_games: Number of games to run.

        Returns:
            List of game records.
        """
        if not self._running:
            await self.start()

        games = []
        games_remaining = num_games

        while games_remaining > 0 and self._running:
            # Check for pending model update
            if self._model_update_pending:
                await self._apply_model_update()

            # Run a batch of games
            batch_size = min(self.batch_size, games_remaining)
            batch_games = await self._run_batch(batch_size)
            games.extend(batch_games)
            games_remaining -= len(batch_games)

            # Log progress
            if self._stats.games_completed % 10 == 0:
                logger.info(
                    f"Progress: {self._stats.games_completed}/{num_games} games, "
                    f"{self._stats.games_per_second:.2f} g/s"
                )

        return games

    async def run_continuous(self) -> None:
        """Run selfplay continuously until stopped.

        Call stop() to terminate the loop.
        """
        if not self._running:
            await self.start()

        while self._running:
            # Check for pending model update
            if self._model_update_pending:
                await self._apply_model_update()

            # Run a batch
            await self._run_batch(self.batch_size)

            # Log progress periodically
            if self._stats.games_completed % 50 == 0:
                logger.info(
                    f"Continuous selfplay: {self._stats.games_completed} games, "
                    f"{self._stats.games_per_second:.2f} g/s, "
                    f"model: {self._current_model_path}"
                )

            # Small yield to allow other tasks
            await asyncio.sleep(0.01)

    async def _initialize_mcts(self) -> None:
        """Initialize the MCTS engine (batched or GPU-accelerated)."""
        from app.models import BoardType

        # Map board type string to enum
        board_type_map = {
            "square8": BoardType.SQUARE8,
            "square19": BoardType.SQUARE19,
            "hex8": BoardType.HEX8,
            "hex": BoardType.HEXAGONAL,
            "hexagonal": BoardType.HEXAGONAL,
        }
        board_type_enum = board_type_map.get(self.board_type.lower(), BoardType.SQUARE8)

        # Create neural network if we have a model
        neural_net = None
        if self._current_model_path and self._current_model_path.exists():
            neural_net = await self._load_neural_net(self._current_model_path)

        if self.use_gpu_mcts:
            # Use GPU-accelerated MultiTreeMCTS (Phase 3)
            from app.ai.tensor_gumbel_tree import MultiTreeMCTS, MultiTreeMCTSConfig

            config = MultiTreeMCTSConfig(
                simulation_budget=self.mcts_sims,
                num_sampled_actions=16,
                device=self.gpu_device,
                eval_mode=self.gpu_eval_mode,
            )
            self._multi_tree_mcts = MultiTreeMCTS(config)
            self._neural_net = neural_net
            logger.info(
                f"Initialized GPU MultiTreeMCTS: device={self.gpu_device}, "
                f"budget={self.mcts_sims}"
            )
        else:
            # Use standard BatchedGumbelMCTS
            from app.ai.batched_gumbel_mcts import create_batched_gumbel_mcts

            self._batched_mcts = create_batched_gumbel_mcts(
                board_type=board_type_enum,
                num_players=self.num_players,
                batch_size=self.batch_size,
                num_sampled_actions=16,
                simulation_budget=self.mcts_sims,
                neural_net=neural_net,
            )

    async def _load_neural_net(self, model_path: Path):
        """Load neural network from path.

        Args:
            model_path: Path to model file.

        Returns:
            NeuralNetAI instance or None.
        """
        try:
            from app.ai.neural_net import NeuralNetAI
            from app.models import BoardType
            from app.models.core import AIConfig, AIType

            board_type_map = {
                "square8": BoardType.SQUARE8,
                "square19": BoardType.SQUARE19,
                "hex8": BoardType.HEX8,
                "hex": BoardType.HEXAGONAL,
                "hexagonal": BoardType.HEXAGONAL,
            }
            board_type_enum = board_type_map.get(self.board_type.lower(), BoardType.SQUARE8)

            config = AIConfig(
                ai_type=AIType.NEURAL_NET,
                difficulty=7,
                use_neural_net=True,
            )

            nn = NeuralNetAI(
                player_number=1,
                config=config,
                board_type=board_type_enum,
            )

            nn.load_model(str(model_path))
            logger.info(f"Loaded neural network from {model_path}")
            return nn

        except Exception as e:
            logger.warning(f"Failed to load neural network: {e}")
            return None

    async def _run_batch(self, batch_size: int) -> list[dict]:
        """Run a batch of games.

        Args:
            batch_size: Number of games in batch.

        Returns:
            List of game records.
        """
        if self.use_gpu_mcts:
            return await self._run_batch_gpu(batch_size)
        else:
            return await self._run_batch_standard(batch_size)

    async def _run_batch_gpu(self, batch_size: int) -> list[dict]:
        """Run a batch of games using GPU-accelerated MultiTreeMCTS.

        This method runs all games in true parallel, using MultiTreeMCTS
        to search all active game positions simultaneously.

        Args:
            batch_size: Number of games in batch.

        Returns:
            List of game records.
        """
        from app.game_engine import GameEngine
        from app.training.initial_state import create_initial_state
        from app.models import BoardType

        board_type_map = {
            "square8": BoardType.SQUARE8,
            "square19": BoardType.SQUARE19,
            "hex8": BoardType.HEX8,
            "hex": BoardType.HEXAGONAL,
            "hexagonal": BoardType.HEXAGONAL,
        }
        board_type_enum = board_type_map.get(self.board_type.lower(), BoardType.SQUARE8)

        # Create initial states for all games
        game_states = [
            create_initial_state(
                board_type=board_type_enum,
                num_players=self.num_players,
            )
            for _ in range(batch_size)
        ]

        # Track game progress
        game_moves: list[list] = [[] for _ in range(batch_size)]
        active_mask = [True] * batch_size
        max_moves = self.max_moves

        # Run all games in parallel
        while any(active_mask):
            # Get active games
            active_indices = [i for i, active in enumerate(active_mask) if active]
            active_states = [game_states[i] for i in active_indices]

            if not active_states:
                break

            # Get moves for all active games using MultiTreeMCTS
            moves, _ = self._multi_tree_mcts.search_batch(
                active_states, neural_net=getattr(self, '_neural_net', None)
            )

            # Apply moves to each game
            for idx, (game_idx, move) in enumerate(zip(active_indices, moves)):
                if move is None:
                    active_mask[game_idx] = False
                    continue

                # Apply move
                game_states[game_idx] = GameEngine.apply_move(game_states[game_idx], move)
                game_moves[game_idx].append(move)

                # Check if game is over
                gs = game_states[game_idx]
                if gs.game_status != "active" or len(game_moves[game_idx]) >= max_moves:
                    active_mask[game_idx] = False

            # Yield to allow other tasks
            await asyncio.sleep(0)

        # Collect results
        games = []
        for i, (game_state, moves) in enumerate(zip(game_states, game_moves)):
            winner = game_state.winner or 0
            self._stats.games_completed += 1
            self._stats.total_moves += len(moves)
            self._stats.wins_by_player[winner] = (
                self._stats.wins_by_player.get(winner, 0) + 1
            )

            games.append({
                "game_idx": self._stats.games_completed,
                "winner": winner,
                "move_count": len(moves),
                "model": str(self._current_model_path or ""),
            })

        return games

    async def _run_batch_standard(self, batch_size: int) -> list[dict]:
        """Run a batch of games using standard BatchedGumbelMCTS.

        Args:
            batch_size: Number of games in batch.

        Returns:
            List of game records.
        """
        from app.game_engine import GameEngine
        from app.training.initial_state import create_initial_state
        from app.models import BoardType

        board_type_map = {
            "square8": BoardType.SQUARE8,
            "square19": BoardType.SQUARE19,
            "hex8": BoardType.HEX8,
            "hex": BoardType.HEXAGONAL,
            "hexagonal": BoardType.HEXAGONAL,
        }
        board_type_enum = board_type_map.get(self.board_type.lower(), BoardType.SQUARE8)

        # Create initial states
        game_states = [
            create_initial_state(
                board_type=board_type_enum,
                num_players=self.num_players,
            )
            for _ in range(batch_size)
        ]

        # Run games (simplified - full implementation would use run_parallel_batch)
        games = []
        for i, initial_state in enumerate(game_states):
            game_state = initial_state
            moves = []
            max_moves = self.max_moves

            while game_state.game_status == "active" and len(moves) < max_moves:
                # Get move
                self._batched_mcts.player_number = game_state.current_player
                batch_moves = self._batched_mcts.select_moves_batch([game_state])
                move = batch_moves[0] if batch_moves else None

                if move is None:
                    break

                # Apply move
                game_state = GameEngine.apply_move(game_state, move)
                moves.append(move)

            # Record game
            winner = game_state.winner or 0
            self._stats.games_completed += 1
            self._stats.total_moves += len(moves)
            self._stats.wins_by_player[winner] = (
                self._stats.wins_by_player.get(winner, 0) + 1
            )

            games.append({
                "game_idx": self._stats.games_completed,
                "winner": winner,
                "move_count": len(moves),
                "model": str(self._current_model_path or ""),
            })

        return games

    def _subscribe_to_events(self) -> None:
        """Subscribe to model promotion events."""
        try:
            from app.coordination.event_router import get_event_router

            router = get_event_router()
            if router:
                self._event_subscription = router.subscribe(
                    "model_promoted",
                    self._on_model_promoted_sync,
                )
                logger.debug("Subscribed to model_promoted events")

        except ImportError:
            logger.debug("Event router not available")
        except Exception as e:
            logger.warning(f"Failed to subscribe to events: {e}")

    def _unsubscribe_from_events(self) -> None:
        """Unsubscribe from model promotion events."""
        try:
            if self._event_subscription:
                from app.coordination.event_router import get_event_router

                router = get_event_router()
                if router:
                    router.unsubscribe(self._event_subscription)
                self._event_subscription = None

        except Exception as e:
            logger.debug(f"Error unsubscribing: {e}")

    def _on_model_promoted_sync(self, event) -> None:
        """Handle model promotion event (sync wrapper).

        This is called from the event router and sets a flag for the
        async run loop to pick up.
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else event

            # Check if this event is for our config
            event_config = payload.get("config_key", "")
            model_path = payload.get("model_path", "")

            if event_config and event_config != self._config_key:
                logger.debug(f"Ignoring model promotion for {event_config}")
                return

            if model_path:
                logger.info(
                    f"Model promotion event received: {model_path}"
                )
                self._pending_model_path = Path(model_path)
                self._model_update_pending = True

        except Exception as e:
            logger.warning(f"Error handling model promotion event: {e}")

    async def _apply_model_update(self) -> None:
        """Apply a pending model update.

        Called during the run loop to hot-reload the model.
        """
        if not self._pending_model_path:
            self._model_update_pending = False
            return

        logger.info(f"Applying model update: {self._pending_model_path}")

        try:
            # Load new neural network
            neural_net = await self._load_neural_net(self._pending_model_path)

            if neural_net:
                # Update MCTS with new network
                self._batched_mcts.neural_net = neural_net
                self._current_model_path = self._pending_model_path
                self._stats.current_model = str(self._pending_model_path)
                self._stats.model_reloads += 1

                logger.info(
                    f"Model hot-reloaded successfully. "
                    f"Total reloads: {self._stats.model_reloads}"
                )

                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        callback(self._pending_model_path)
                    except Exception as e:
                        logger.warning(f"Model update callback error: {e}")

        except Exception as e:
            logger.error(f"Failed to apply model update: {e}")

        self._pending_model_path = None
        self._model_update_pending = False

    def on_model_update(self, callback: Callable[[Path], None]) -> None:
        """Register callback for model updates.

        Args:
            callback: Function to call when model is updated.
        """
        self._callbacks.append(callback)

    def get_stats(self) -> dict:
        """Get current selfplay statistics.

        Returns:
            Dictionary of statistics.
        """
        return {
            "games_completed": self._stats.games_completed,
            "total_moves": self._stats.total_moves,
            "games_per_second": self._stats.games_per_second,
            "elapsed_time": self._stats.elapsed_time,
            "model_reloads": self._stats.model_reloads,
            "current_model": self._stats.current_model,
            "wins_by_player": dict(self._stats.wins_by_player),
            "config_key": self._config_key,
        }


# Convenience function for quick usage
async def run_event_driven_selfplay(
    board_type: str,
    num_players: int = 2,
    num_games: int = 100,
    batch_size: int = 16,
    mcts_sims: int = 800,
) -> list[dict]:
    """Run event-driven selfplay for a specific number of games.

    Args:
        board_type: Board type (square8, hex8, etc.)
        num_players: Number of players
        num_games: Number of games to run
        batch_size: Parallel batch size
        mcts_sims: MCTS simulations per move

    Returns:
        List of game records.
    """
    manager = EventDrivenSelfplay(
        board_type=board_type,
        num_players=num_players,
        batch_size=batch_size,
        mcts_sims=mcts_sims,
    )

    try:
        games = await manager.run_games(num_games)
        return games
    finally:
        await manager.stop()
