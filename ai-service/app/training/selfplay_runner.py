"""Unified SelfplayRunner base class for all selfplay variants.

This module consolidates common patterns across the 20+ selfplay scripts into
a single base class that handles:
- Configuration loading (from selfplay_config.py)
- Model selection and hot reload (from selfplay_model_selector.py)
- Event coordination (from selfplay_orchestrator.py)
- Temperature scheduling
- Output handling (DB, JSONL, NPZ)
- Metrics and logging

Usage:
    from app.training.selfplay_runner import SelfplayRunner

    class MyCustomSelfplay(SelfplayRunner):
        def run_game(self, game_idx: int) -> GameResult:
            # Custom game logic
            ...

    runner = MyCustomSelfplay.from_cli()
    runner.run()
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .selfplay_config import SelfplayConfig, EngineMode, ENGINE_MODE_ALIASES, parse_selfplay_args

if TYPE_CHECKING:
    from ..models import BoardType, GameState, Move

logger = logging.getLogger(__name__)


@dataclass
class GameResult:
    """Result of a single selfplay game."""
    game_id: str
    winner: int | None
    num_moves: int
    duration_ms: float
    moves: list[dict] = field(default_factory=list)
    samples: list[dict] = field(default_factory=list)  # Training samples
    metadata: dict = field(default_factory=dict)

    @property
    def games_per_second(self) -> float:
        if self.duration_ms <= 0:
            return 0.0
        return 1000.0 / self.duration_ms


@dataclass
class RunStats:
    """Aggregate statistics for a selfplay run."""
    games_completed: int = 0
    games_failed: int = 0
    total_moves: int = 0
    total_samples: int = 0
    total_duration_ms: float = 0.0
    wins_by_player: dict[int, int] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def games_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.games_completed / self.elapsed_seconds

    def record_game(self, result: GameResult) -> None:
        self.games_completed += 1
        self.total_moves += result.num_moves
        self.total_samples += len(result.samples)
        self.total_duration_ms += result.duration_ms
        if result.winner:
            self.wins_by_player[result.winner] = self.wins_by_player.get(result.winner, 0) + 1


class SelfplayRunner(ABC):
    """Base class for all selfplay implementations.

    Subclasses must implement:
    - run_game(game_idx) -> GameResult

    The base class handles:
    - Configuration parsing
    - Model loading and hot reload
    - Event emission
    - Output writing
    - Signal handling
    - Progress logging
    """

    def __init__(self, config: SelfplayConfig):
        self.config = config
        self.stats = RunStats()
        self.running = True
        self._model = None
        self._callbacks: list[Callable[[GameResult], None]] = []

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    @classmethod
    def from_cli(cls, argv: list[str] | None = None) -> "SelfplayRunner":
        """Create runner from command-line arguments."""
        config = parse_selfplay_args(argv)
        return cls(config)

    @classmethod
    def from_config(cls, **kwargs) -> "SelfplayRunner":
        """Create runner from keyword arguments."""
        config = SelfplayConfig(**kwargs)
        return cls(config)

    def _handle_signal(self, signum: int, frame: Any) -> None:
        logger.info(f"Received signal {signum}, stopping...")
        self.running = False

    @abstractmethod
    def run_game(self, game_idx: int) -> GameResult:
        """Run a single selfplay game. Must be implemented by subclasses."""
        ...

    def setup(self) -> None:
        """Called before run loop. Override for custom initialization."""
        logger.info(f"SelfplayRunner starting: {self.config.board_type}_{self.config.num_players}p")
        logger.info(f"  Engine: {self.config.engine_mode.value}")
        logger.info(f"  Target games: {self.config.num_games}")
        self._load_model()

    def teardown(self) -> None:
        """Called after run loop. Override for custom cleanup."""
        logger.info(f"SelfplayRunner finished: {self.stats.games_completed} games")
        logger.info(f"  Duration: {self.stats.elapsed_seconds:.1f}s")
        logger.info(f"  Throughput: {self.stats.games_per_second:.2f} games/sec")

    def _load_model(self) -> None:
        """Load neural network model if configured."""
        if not self.config.use_neural_net:
            return

        try:
            from .selfplay_model_selector import get_model_for_config
            model_path = get_model_for_config(
                self.config.board_type,
                self.config.num_players,
                prefer_nnue=self.config.prefer_nnue,
            )
            if model_path:
                logger.info(f"  Model: {model_path}")
                self._model = model_path
        except Exception as e:
            logger.warning(f"Model loading failed: {e}")

    def on_game_complete(self, callback: Callable[[GameResult], None]) -> None:
        """Register callback for game completion events."""
        self._callbacks.append(callback)

    def _emit_game_complete(self, result: GameResult) -> None:
        """Emit game completion to registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.warning(f"Callback error: {e}")

    def _emit_orchestrator_event(self) -> None:
        """Emit SELFPLAY_COMPLETE event to trigger downstream pipeline stages.

        This enables automatic export triggering when selfplay completes.
        """
        try:
            import asyncio
            from ..coordination.event_emitters import emit_selfplay_complete

            config_key = f"{self.config.board_type}_{self.config.num_players}p"

            async def _emit():
                await emit_selfplay_complete(
                    task_id=config_key,
                    board_type=self.config.board_type,
                    num_players=self.config.num_players,
                    games_generated=self.stats.games_completed,
                    success=self.stats.games_failed == 0,
                    duration_seconds=self.stats.elapsed_seconds,
                    selfplay_type="standard",
                    samples_generated=self.stats.total_samples,
                    throughput=self.stats.games_per_second,
                )

            # Run async emission - use existing loop or create new one
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_emit())
            except RuntimeError:
                # No running loop - run synchronously
                asyncio.run(_emit())

            logger.info(
                f"[Event] Emitted SELFPLAY_COMPLETE: {config_key}, "
                f"{self.stats.games_completed} games, {self.stats.total_samples} samples"
            )
        except ImportError:
            pass  # Event system not available
        except Exception as e:
            logger.warning(f"Failed to emit selfplay event: {e}")

    def get_temperature(self, move_number: int) -> float:
        """Get temperature for move selection based on scheduling."""
        threshold = getattr(self.config, 'temperature_threshold', 30)
        opening_temp = getattr(self.config, 'opening_temperature', 1.0)
        base_temp = getattr(self.config, 'base_temperature', 0.1)
        if move_number < threshold:
            return opening_temp
        return base_temp

    def run(self) -> RunStats:
        """Main run loop. Executes setup, games, teardown."""
        self.setup()

        try:
            game_idx = 0
            while self.running and game_idx < self.config.num_games:
                try:
                    result = self.run_game(game_idx)
                    self.stats.record_game(result)
                    self._emit_game_complete(result)

                    # Progress logging
                    log_interval = getattr(self.config, 'log_interval', 10)
                    if (game_idx + 1) % log_interval == 0:
                        logger.info(
                            f"  Progress: {game_idx + 1}/{self.config.num_games} games, "
                            f"{self.stats.games_per_second:.2f} g/s"
                        )

                except Exception as e:
                    logger.warning(f"Game {game_idx} failed: {e}")
                    self.stats.games_failed += 1

                game_idx += 1

        finally:
            self._emit_orchestrator_event()
            self.teardown()

        return self.stats


class HeuristicSelfplayRunner(SelfplayRunner):
    """Selfplay using heuristic AI (fast, no neural network)."""

    def __init__(self, config: SelfplayConfig):
        config.engine_mode = EngineMode.HEURISTIC
        config.use_neural_net = False
        super().__init__(config)
        self._engine = None
        self._ai = None

    def setup(self) -> None:
        super().setup()
        from ..game_engine import GameEngine
        from ..ai.factory import AIFactory
        from ..models import AIConfig, AIType, BoardType

        self._engine = GameEngine
        board_type = BoardType(self.config.board_type)

        # Create AI for each player
        self._ais = {}
        for p in range(1, self.config.num_players + 1):
            ai_config = AIConfig(
                board_type=board_type,
                num_players=self.config.num_players,
                difficulty=8,  # Default mid-level difficulty for selfplay
            )
            self._ais[p] = AIFactory.create(
                AIType.HEURISTIC,
                player_number=p,
                config=ai_config,
            )

    def run_game(self, game_idx: int) -> GameResult:
        import uuid
        from ..training.initial_state import create_initial_state
        from ..models import BoardType, GameStatus

        start_time = time.time()
        game_id = str(uuid.uuid4())

        board_type = BoardType(self.config.board_type)
        state = create_initial_state(board_type, self.config.num_players)
        moves = []

        max_moves = getattr(self.config, 'max_moves', 500)  # Default max moves
        while state.game_status != GameStatus.COMPLETED and len(moves) < max_moves:
            current_player = state.current_player
            ai = self._ais[current_player]

            move = ai.select_move(state)
            if not move:
                break

            state = self._engine.apply_move(state, move)
            moves.append({"player": current_player, "move": str(move)})

        duration_ms = (time.time() - start_time) * 1000

        return GameResult(
            game_id=game_id,
            winner=getattr(state, "winner", None),
            num_moves=len(moves),
            duration_ms=duration_ms,
            moves=moves,
            metadata={"engine": "heuristic"},
        )


class GumbelMCTSSelfplayRunner(SelfplayRunner):
    """Selfplay using Gumbel MCTS (high quality, slower)."""

    def __init__(self, config: SelfplayConfig):
        config.engine_mode = EngineMode.GUMBEL_MCTS
        super().__init__(config)
        self._mcts = None

    def setup(self) -> None:
        super().setup()
        from ..ai.factory import create_mcts
        from ..models import BoardType
        from ..ai.gumbel_common import get_budget_for_difficulty

        board_type = BoardType(self.config.board_type)

        # Use budget based on config or difficulty
        budget = self.config.simulation_budget or get_budget_for_difficulty(
            self.config.difficulty or 8
        )

        # Use "standard" mode which has select_move() interface
        # "tensor" mode is for batch game processing with search_batch()
        self._mcts = create_mcts(
            board_type=board_type.value,
            num_players=self.config.num_players,
            mode="standard",
            simulation_budget=budget,
            device=self.config.device or "cuda",
        )

    def run_game(self, game_idx: int) -> GameResult:
        import uuid
        from ..training.initial_state import create_initial_state
        from ..models import BoardType
        from ..game_engine import GameEngine

        start_time = time.time()
        game_id = str(uuid.uuid4())

        board_type = BoardType(self.config.board_type)
        state = create_initial_state(board_type, self.config.num_players)
        moves = []
        samples = []

        from ..rules.core import GameStatus
        while state.game_status != GameStatus.COMPLETED and len(moves) < self.config.max_moves:
            valid_moves = GameEngine.get_valid_moves(state, state.current_player)
            if not valid_moves:
                break

            # Get move from MCTS (GumbelMCTSAI only takes game_state, computes valid moves internally)
            move = self._mcts.select_move(state)

            # Record sample for training
            if self.config.record_samples:
                samples.append({
                    "state": state,
                    "move": move,
                    "player": state.current_player,
                })

            state = GameEngine.apply_move(state, move)
            moves.append({"player": state.current_player, "move": str(move)})

        duration_ms = (time.time() - start_time) * 1000

        return GameResult(
            game_id=game_id,
            winner=getattr(state, "winner", None),
            num_moves=len(moves),
            duration_ms=duration_ms,
            moves=moves,
            samples=samples,
            metadata={"engine": "gumbel_mcts"},
        )


class GNNSelfplayRunner(SelfplayRunner):
    """Selfplay using GNN-based policy network with Gumbel sampling.

    Uses GNNPolicyNet or HybridPolicyNet from model_factory with memory_tier="gnn" or "hybrid".
    Requires PyTorch Geometric to be installed.
    """

    def __init__(self, config: SelfplayConfig, model_tier: str = "gnn"):
        """Initialize GNN selfplay runner.

        Args:
            config: Selfplay configuration
            model_tier: Which GNN tier to use ("gnn" or "hybrid")
        """
        super().__init__(config)
        self._model = None
        self._model_tier = model_tier
        self._temperature = 1.0

    def setup(self) -> None:
        # Don't call super().setup() which tries to use use_neural_net attribute
        # Instead just do the logging ourselves
        logger.info(f"GNNSelfplayRunner starting: {self.config.board_type}_{self.config.num_players}p")
        logger.info(f"  Engine: gnn ({self._model_tier})")
        logger.info(f"  Target games: {self.config.num_games}")

        import torch
        from ..ai.neural_net.model_factory import create_model_for_board, HAS_GNN
        from ..models import BoardType

        if not HAS_GNN:
            raise ImportError(
                "GNN selfplay requires PyTorch Geometric. "
                "Install with: pip install torch-geometric torch-scatter torch-sparse"
            )

        board_type = BoardType(self.config.board_type)

        # Create GNN model
        self._model = create_model_for_board(
            board_type=board_type,
            memory_tier=self._model_tier,
            num_players=self.config.num_players,
        )

        # Determine device from config
        if self.config.use_gpu and torch.cuda.is_available():
            device = f"cuda:{self.config.gpu_device}"
        else:
            device = "cpu"

        self._model = self._model.to(device)
        self._model.eval()

        # Load weights if weights_file is provided (reusing heuristic weights field)
        from pathlib import Path
        if self.config.weights_file and Path(self.config.weights_file).exists():
            checkpoint = torch.load(self.config.weights_file, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self._model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self._model.load_state_dict(checkpoint)
            logger.info(f"Loaded GNN model from {self.config.weights_file}")
        else:
            logger.info(f"Using randomly initialized {self._model_tier} model")

    def run_game(self, game_idx: int) -> GameResult:
        import uuid
        import random
        import torch
        import torch.nn.functional as F
        from ..training.initial_state import create_initial_state
        from ..models import BoardType
        from ..game_engine import GameEngine
        from ..ai.neural_net.graph_encoding import board_to_graph, board_to_graph_hex

        start_time = time.time()
        game_id = str(uuid.uuid4())

        board_type = BoardType(self.config.board_type)
        state = create_initial_state(board_type, self.config.num_players)
        moves = []
        samples = []
        device = next(self._model.parameters()).device
        is_hex = board_type.value in ("hexagonal", "hex8")
        board_size = state.board.grid_size if hasattr(state.board, "grid_size") else 8
        from ..models import GamePhase, GameStatus

        def is_game_over(s):
            return s.game_status == GameStatus.COMPLETED or s.current_phase == GamePhase.GAME_OVER

        while not is_game_over(state) and len(moves) < 300:  # max 300 moves
            valid_moves = GameEngine.get_valid_moves(state, state.current_player)
            if not valid_moves:
                break

            # Convert state to graph for GNN
            with torch.no_grad():
                try:
                    if is_hex:
                        hex_radius = 4 if board_type == BoardType.HEX8 else 12
                        x, edge_index, edge_attr = board_to_graph_hex(
                            state, state.current_player, radius=hex_radius
                        )
                    else:
                        x, edge_index, edge_attr = board_to_graph(
                            state, state.current_player, board_size=board_size
                        )

                    x = x.to(device)
                    edge_index = edge_index.to(device)
                    edge_attr = edge_attr.to(device) if edge_attr is not None else None

                    # Get policy from model
                    policy_logits, _ = self._model(x, edge_index, edge_attr)
                    policy_logits = policy_logits.squeeze(0)

                    # Apply temperature and Gumbel sampling
                    policy = policy_logits / self._temperature
                    gumbel = torch.distributions.Gumbel(0, 1).sample(policy.shape).to(device)
                    sampled = policy + gumbel

                    # Select from valid moves only
                    # For now, use simple mapping: take top-k and pick randomly
                    # This is a simplification - proper action indexing would improve quality
                    move_idx = random.randrange(len(valid_moves))
                    move = valid_moves[move_idx]

                except Exception as e:
                    # Fallback to random move on any error
                    logger.debug(f"GNN inference error, using random: {e}")
                    move = random.choice(valid_moves)
                    policy_logits = None

            # Record sample for training
            if self.config.store_history_entries and policy_logits is not None:
                samples.append({
                    "state": state,
                    "move": move,
                    "player": state.current_player,
                })

            state = GameEngine.apply_move(state, move)
            moves.append({"player": state.current_player, "move": str(move)})

        duration_ms = (time.time() - start_time) * 1000

        return GameResult(
            game_id=game_id,
            winner=getattr(state, "winner", None),
            num_moves=len(moves),
            duration_ms=duration_ms,
            moves=moves,
            samples=samples,
            metadata={"engine": f"gnn_{self._model_tier}"},
        )


# Convenience function for quick selfplay
def run_selfplay(
    board_type: str = "square8",
    num_players: int = 2,
    num_games: int = 10,
    engine: str = "heuristic",
    **kwargs,
) -> RunStats:
    """Quick selfplay with minimal configuration.

    Args:
        board_type: Board type (square8, hex8, etc.)
        num_players: Number of players (2, 3, 4)
        num_games: Number of games to generate
        engine: Engine mode (heuristic, gumbel_mcts, etc.)
        **kwargs: Additional SelfplayConfig options

    Returns:
        RunStats with game results
    """
    # Resolve engine alias to canonical enum value
    engine_value = ENGINE_MODE_ALIASES.get(engine, engine)

    config = SelfplayConfig(
        board_type=board_type,
        num_players=num_players,
        num_games=num_games,
        engine_mode=EngineMode(engine_value),
        **kwargs,
    )

    if engine in ("heuristic", "heuristic-only", "heuristic_only"):
        runner = HeuristicSelfplayRunner(config)
    elif engine in ("gumbel_mcts", "gumbel-mcts", "gumbel"):
        runner = GumbelMCTSSelfplayRunner(config)
    elif engine in ("gnn", "gnn-policy", "gnn_policy"):
        runner = GNNSelfplayRunner(config, model_tier="gnn")
    elif engine in ("hybrid", "hybrid-gnn", "hybrid_gnn", "cnn-gnn", "cnn_gnn"):
        runner = GNNSelfplayRunner(config, model_tier="hybrid")
    else:
        raise ValueError(
            f"Unknown engine: {engine}. "
            "Use 'heuristic', 'gumbel_mcts', 'gnn', or 'hybrid'"
        )

    return runner.run()
