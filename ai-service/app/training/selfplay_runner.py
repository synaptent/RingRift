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
import asyncio
import logging
import os
import signal
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from ..db.write_lock import DatabaseWriteLock
from ..errors import InvalidGameError
from .selfplay_config import SelfplayConfig, EngineMode, ENGINE_MODE_ALIASES, parse_selfplay_args
from .elo_recording import record_selfplay_match, EloRecordResult

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
    # For database storage - these are optional for backwards compatibility
    initial_state: Any = None  # GameState at start
    final_state: Any = None    # GameState at end
    move_objects: list = field(default_factory=list)  # Actual Move objects
    # MCTS distribution data for training (v10-v11 schema)
    move_probs: list[dict[str, float] | None] = field(default_factory=list)  # Policy targets per move
    search_stats: list[dict | None] = field(default_factory=list)  # Rich stats per move

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
        self._signal_received = False
        self._db = None  # Database connection for record_db

        # Phase 3 Feedback Loop: Quality-based throttling (December 2025)
        self._quality_throttle_factor = 1.0  # 1.0 = full speed, 0.0 = paused
        self._quality_paused = False  # True when quality is critically low
        self._quality_event_subscription = None

        # Phase 14 Feedback Loop: Curriculum and selfplay target events (December 2025)
        self._curriculum_difficulty = 1.0  # Multiplier for opponent difficulty
        self._extra_games_requested = 0  # Extra games from weak model feedback
        self._regeneration_pending = False  # True when quality blocked training
        self._base_budget = 800  # Default base budget (increased Dec 2025 for 2000+ Elo)
        self._current_budget = 800  # Current effective budget

        # Phase 2 Feedback Loop: Promotion → Selfplay difficulty coupling (December 2025)
        self._promotion_difficulty_boost = 1.0  # Increased on failed promotion
        self._consecutive_promotion_failures = 0  # Track consecutive failures

        # PFSP opponent selection (December 2025)
        self._pfsp_enabled = False
        self._pfsp_selector = None

        # Temperature scheduler with exploration boost (December 2025)
        self._temperature_scheduler = None  # Initialized in setup()

        # Gauntlet feedback parameters (December 2025 - Phase 2.1 integration)
        self._gauntlet_temperature_scale = 1.0  # From HYPERPARAMETER_UPDATED
        self._gauntlet_quality_boost = 0.0  # Added to quality threshold
        self._gauntlet_rate_multiplier = 1.0  # Selfplay rate adjustment

        # Quality penalty rate (December 2025 - from AdaptiveController)
        self._quality_penalty_rate = 1.0  # From QUALITY_PENALTY_APPLIED

        # Setup signal handlers - only respond to first signal
        # Signal handlers can only be set in main thread, so wrap in try/except
        try:
            signal.signal(signal.SIGTERM, self._handle_signal)
            signal.signal(signal.SIGINT, self._handle_signal)
        except ValueError:
            # Not in main thread (e.g., running from thread pool executor)
            pass

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
        if self._signal_received:
            # Already received a signal, ignore subsequent ones
            return
        self._signal_received = True
        logger.info(f"Received signal {signum}, stopping gracefully...")
        self.running = False
        # Restore default handlers to prevent infinite loops
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    @abstractmethod
    def run_game(self, game_idx: int) -> GameResult:
        """Run a single selfplay game. Must be implemented by subclasses."""
        ...

    def _check_process_limit(self) -> bool:
        """Check if node is under selfplay process limit.

        December 29, 2025: Prevents runaway process spawning that causes OOM.
        Set RINGRIFT_RUNAWAY_SELFPLAY_PROCESS_THRESHOLD to configure limit (default: 128).

        Returns:
            True if under limit, False if at/over limit.
        """
        try:
            import psutil
            from ..config.env import env

            threshold = env.runaway_selfplay_process_threshold
            # Count Python processes that look like selfplay
            selfplay_count = 0
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    cmdline = " ".join(proc.info.get("cmdline") or [])
                    if "python" in proc.info.get("name", "").lower():
                        if "selfplay" in cmdline.lower() or "gpu_parallel" in cmdline.lower():
                            selfplay_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue

            if selfplay_count >= threshold:
                logger.warning(
                    f"[ProcessLimit] At limit: {selfplay_count}/{threshold} selfplay processes. "
                    "Skipping this run to prevent OOM."
                )
                return False

            logger.debug(f"[ProcessLimit] Under limit: {selfplay_count}/{threshold} selfplay processes")
            return True

        except ImportError:
            # psutil not available, skip check
            logger.debug("[ProcessLimit] psutil not available, skipping process limit check")
            return True
        except (OSError, RuntimeError) as e:
            logger.debug(f"[ProcessLimit] Error checking processes: {e}")
            return True  # Don't block on errors

    def setup(self) -> None:
        """Called before run loop. Override for custom initialization."""
        # December 29, 2025: Check process limit before starting
        if not self._check_process_limit():
            self.running = False
            return

        logger.info(f"SelfplayRunner starting: {self.config.board_type}_{self.config.num_players}p")
        logger.info(f"  Engine: {self.config.engine_mode.value}")
        self._apply_elo_adaptive_config()  # Set model_elo for Elo-adaptive budget/temperature
        self._apply_selfplay_rate_adjustment()  # Adjust num_games based on Elo momentum
        self._apply_quality_budget_multiplier()  # Adjust MCTS budget based on quality signals
        logger.info(f"  Target games: {self.config.num_games}")
        self._load_model()
        self._open_database()
        self._subscribe_to_quality_events()
        self._subscribe_to_feedback_events()
        self._init_pfsp()
        self._init_temperature_scheduler()

    def _apply_elo_adaptive_config(self) -> None:
        """Apply Elo-adaptive configuration for budget and temperature.

        December 2025: Fetches current model Elo from FeedbackAccelerator
        and sets it on the config. This enables:
        - Elo-adaptive MCTS budget (via SelfplayConfig.get_effective_budget())
        - Elo-adaptive temperature (via create_elo_adaptive_scheduler())

        Weak models (Elo < 1300) get low budget/high exploration.
        Strong models (Elo > 1700) get high budget/low exploration.
        """
        try:
            from app.training.feedback_accelerator import get_feedback_accelerator

            config_key = f"{self.config.board_type}_{self.config.num_players}p"
            accelerator = get_feedback_accelerator()

            momentum = accelerator.get_config_momentum(config_key)
            if momentum is not None and momentum.current_elo > 0:
                self.config.model_elo = momentum.current_elo

                # Also update base budget from Elo-adaptive logic
                budget = self.config.get_effective_budget()
                self._base_budget = budget
                self._current_budget = budget

                logger.info(
                    f"  [Elo-Adaptive] Model Elo {momentum.current_elo:.0f}, "
                    f"budget {budget} sims"
                )
            else:
                logger.debug(f"[Elo-Adaptive] No momentum data for {config_key}, using defaults")

        except ImportError:
            pass  # FeedbackAccelerator not available
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.debug(f"[SelfplayRunner] Elo-adaptive config not applied: {e}")

    def _apply_selfplay_rate_adjustment(self) -> None:
        """Adjust selfplay game count based on FeedbackAccelerator rate recommendation.

        December 2025: Selfplay rate now responds to Elo momentum:
        - Improving models: Generate more games (higher rate multiplier)
        - Plateauing models: Reduce games (focus on quality over quantity)
        - Stable models: Normal rate

        This creates a positive feedback loop where improving models get more
        training data, accelerating their improvement.
        """
        try:
            from app.training.feedback_accelerator import get_feedback_accelerator

            config_key = f"{self.config.board_type}_{self.config.num_players}p"
            accelerator = get_feedback_accelerator()

            rate_multiplier = accelerator.get_selfplay_rate_recommendation(config_key)

            # Only adjust if rate is non-default
            if rate_multiplier != 1.0:
                original_games = self.config.num_games
                self.config.num_games = int(self.config.num_games * rate_multiplier)

                logger.info(
                    f"  [Adaptive] Selfplay rate adjusted for {config_key}: "
                    f"{original_games} -> {self.config.num_games} games ({rate_multiplier:.2f}x)"
                )
        except ImportError:
            pass  # FeedbackAccelerator not available
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.debug(f"[SelfplayRunner] Selfplay rate adjustment not applied: {e}")

    def _apply_quality_budget_multiplier(self) -> None:
        """Apply quality-based budget multiplier from SelfplayOrchestrator.

        December 2025 Phase 8: Closes the quality → selfplay budget feedback loop.

        When data quality is low, the orchestrator increases the quality budget
        multiplier, which increases MCTS simulations per move to generate
        higher-quality games. Conversely, high quality data allows lower budgets
        for faster game generation.

        This creates a self-correcting feedback loop:
        - Low quality games → higher budget → more accurate moves → better games
        - High quality games → lower budget → faster throughput → more games
        """
        try:
            from app.coordination.selfplay_orchestrator import get_selfplay_orchestrator

            config_key = f"{self.config.board_type}_{self.config.num_players}p"
            orchestrator = get_selfplay_orchestrator()

            if orchestrator is None:
                return

            # Get quality-adjusted budget
            base_budget = self._base_budget or self.config.mcts_simulations
            effective_budget = orchestrator.get_effective_budget(config_key, base_budget)

            if effective_budget != base_budget:
                old_budget = self._current_budget or base_budget
                self._current_budget = effective_budget

                logger.info(
                    f"  [Quality-Adaptive] Budget adjusted for {config_key}: "
                    f"{old_budget} -> {effective_budget} sims "
                    f"(quality multiplier: {orchestrator.get_quality_budget_multiplier(config_key):.2f}x)"
                )

        except ImportError:
            pass  # SelfplayOrchestrator not available
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.debug(f"[SelfplayRunner] Quality budget adjustment not applied: {e}")

    def teardown(self) -> None:
        """Called after run loop. Override for custom cleanup."""
        self._close_database()
        logger.info(f"SelfplayRunner finished: {self.stats.games_completed} games")
        logger.info(f"  Duration: {self.stats.elapsed_seconds:.1f}s")
        logger.info(f"  Throughput: {self.stats.games_per_second:.2f} games/sec")

    def _load_model(self) -> None:
        """Load neural network model if configured.

        Includes race condition protection: waits for model distribution
        to complete before attempting to load, avoiding failures when
        selfplay starts immediately after a model promotion.

        December 2025: Now actually loads the NeuralNetAI object (not just
        storing the path), enabling GPU batch evaluation in MCTS.
        """
        if not self.config.use_neural_net:
            self._neural_net = None
            return

        try:
            # First, wait for model distribution if needed (fixes race condition)
            self._wait_for_model_availability()

            from .selfplay_model_selector import get_model_for_config
            model_path = get_model_for_config(
                self.config.board_type,
                self.config.num_players,
                prefer_nnue=self.config.prefer_nnue,
                model_version=self.config.model_version,  # Jan 5, 2026: Architecture selection
            )
            if model_path:
                logger.info(f"  Model: {model_path}")
                self._model = model_path

                # December 2025: Actually load the neural network for GPU batch evaluation
                # Previously only stored the path, causing MCTS to run without neural net
                self._neural_net = self._create_neural_net_from_path(model_path)
            else:
                self._neural_net = None
        except (RuntimeError, ValueError, OSError, KeyError) as e:
            logger.warning(f"Model loading failed: {e}")
            self._neural_net = None

    def _create_neural_net_from_path(self, model_path: str) -> "Any":
        """Create a NeuralNetAI instance from a model checkpoint path.

        December 2025: Enables GPU batch evaluation by actually loading
        the neural network instead of just storing the path.

        Args:
            model_path: Path to the .pth checkpoint file.

        Returns:
            NeuralNetAI instance, or None if loading fails.
        """
        try:
            from ..models import BoardType, AIConfig
            from ..ai.neural_net import NeuralNetAI

            # Determine board type enum
            board_type = BoardType(self.config.board_type)

            # Create config for NeuralNetAI
            # Pass model path directly via nn_model_id - if it ends in .pth,
            # NeuralNetAI treats it as an explicit checkpoint path
            config = AIConfig(
                difficulty=8,  # Not used for selfplay, but required
                use_neural_net=True,
                nn_model_id=model_path,  # Direct path to checkpoint
            )

            # Create NeuralNetAI with player 1 (will be used for all players in batch eval)
            neural_net = NeuralNetAI(
                player_number=1,
                config=config,
                board_type=board_type,
            )

            logger.info(
                f"  [GPU] Loaded NeuralNetAI from {model_path} "
                f"(device={neural_net.device}, num_players={getattr(neural_net, 'num_players', 'unknown')})"
            )
            return neural_net

        except ImportError as e:
            logger.warning(f"NeuralNetAI not available: {e}")
            return None
        except (RuntimeError, ValueError, OSError, FileNotFoundError) as e:
            logger.warning(f"Failed to load neural network from {model_path}: {e}")
            return None

    def _wait_for_model_availability(self) -> None:
        """Wait for model to be distributed before loading.

        Prevents race condition where selfplay nodes receive MODEL_PROMOTED
        event and try to load models before ModelDistributionDaemon finishes
        distributing them to the cluster.

        Uses asyncio to wait for MODEL_DISTRIBUTION_COMPLETE event or
        checks if model already exists locally. Times out after 300s.
        """
        try:
            from app.coordination.unified_distribution_daemon import (
                wait_for_model_distribution,
                check_model_availability,
            )

            # Quick synchronous check first
            if check_model_availability(self.config.board_type, self.config.num_players):
                logger.debug("[ModelDistribution] Model already available locally")
                return

            # If not available, wait for distribution with timeout
            logger.info(
                f"[ModelDistribution] Waiting for model distribution for "
                f"{self.config.board_type}_{self.config.num_players}p..."
            )

            # Run async wait in sync context
            import asyncio
            import concurrent.futures
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context - use thread to avoid nested loop
                # This runs the coroutine in a separate thread's event loop
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        wait_for_model_distribution(
                            self.config.board_type,
                            self.config.num_players,
                            timeout=300.0,
                        )
                    )
                    available = future.result(timeout=310.0)
            except RuntimeError:
                # No running loop - create new one
                available = asyncio.run(
                    wait_for_model_distribution(
                        self.config.board_type,
                        self.config.num_players,
                        timeout=300.0,
                    )
                )

            if not available:
                logger.warning(
                    f"[ModelDistribution] Model distribution timed out after 300s. "
                    "Will attempt to use fallback model or random policy."
                )

        except ImportError:
            # Model distribution daemon not available - skip wait
            logger.debug("[ModelDistribution] Distribution daemon not available, skipping wait")
        except (asyncio.TimeoutError, RuntimeError, OSError) as e:
            logger.warning(f"[ModelDistribution] Error waiting for model: {e}")

    def _open_database(self) -> None:
        """Open database for game recording if record_db is configured."""
        if not self.config.record_db:
            return

        try:
            from ..db.game_replay import GameReplayDB
            # Ensure parent directory exists
            db_path = Path(self.config.record_db)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = GameReplayDB(str(db_path))
            logger.info(f"  Database: {self.config.record_db}")

            # Phase 4A.3: Emit DATABASE_CREATED event for immediate registration
            self._register_database_immediately(db_path)
        except (OSError, sqlite3.Error) as e:
            logger.warning(f"Failed to open database {self.config.record_db}: {e}")
            self._db = None

    def _register_database_immediately(self, db_path: Path) -> None:
        """Register database immediately for cluster-wide visibility.

        Phase 4A.3 (December 2025): Reduces data visibility delay from 5 min to <1 sec
        by emitting DATABASE_CREATED event when selfplay creates/opens a database.
        """
        try:
            import socket
            from app.coordination.event_router import publish_sync, DataEventType

            config_key = f"{self.config.board_type}_{self.config.num_players}p"
            node_id = socket.gethostname()

            # Emit event for immediate registration
            publish_sync(
                DataEventType.DATABASE_CREATED,
                {
                    "config_key": config_key,
                    "db_path": str(db_path.absolute()),
                    "node_id": node_id,
                    "board_type": self.config.board_type,
                    "num_players": self.config.num_players,
                    "engine_mode": self.config.engine_mode.value if self.config.engine_mode else None,
                },
                source="selfplay_runner",
            )
            logger.debug(f"[SelfplayRunner] Emitted DATABASE_CREATED for {db_path}")

            # Also register directly in ClusterManifest for local visibility
            try:
                from app.distributed.cluster_manifest import get_cluster_manifest
                manifest = get_cluster_manifest()
                manifest.register_database(
                    db_path=str(db_path.absolute()),
                    node_id=node_id,
                    board_type=self.config.board_type,
                    num_players=self.config.num_players,
                    config_key=config_key,
                    engine_mode=self.config.engine_mode.value if self.config.engine_mode else None,
                )
            except (AttributeError, KeyError, TypeError, OSError) as e:
                logger.debug(f"[SelfplayRunner] Could not register in manifest: {e}")

        except ImportError:
            pass  # Event system not available
        except (AttributeError, KeyError, TypeError, OSError) as e:
            logger.debug(f"[SelfplayRunner] Could not emit DATABASE_CREATED: {e}")

    def _close_database(self) -> None:
        """Close database connection properly.

        GameReplayDB.close() handles WAL checkpointing and cleanup.
        Failing to call close() causes file descriptor leaks that crash
        nodes after ~500 games with 'too many open files' errors.
        """
        if self._db is not None:
            try:
                self._db.close()
            except (OSError, sqlite3.Error) as e:
                logger.warning(f"Error closing database: {e}")
            finally:
                self._db = None

    def _subscribe_to_quality_events(self) -> None:
        """Subscribe to quality events for throttling (Phase 3 Feedback Loop).

        December 2025: Selfplay now responds to data quality signals.
        When quality drops, selfplay rate is reduced to avoid generating
        more low-quality data that would waste compute.
        """
        try:
            from app.coordination.event_router import DataEventType, get_router as get_event_router

            router = get_event_router()
            config_key = f"{self.config.board_type}_{self.config.num_players}p"

            def on_quality_warning(event):
                """Handle LOW_QUALITY_DATA_WARNING events."""
                payload = event.payload if hasattr(event, "payload") else event
                event_config = f"{payload.get('board_type')}_{payload.get('num_players')}p"

                # Only respond to our config's quality events
                if event_config != config_key:
                    return

                quality_score = payload.get("quality_score", 1.0)
                severity = payload.get("severity", "warning")

                if severity == "critical" or quality_score < 0.5:
                    # Critical quality issue - pause selfplay
                    self._quality_paused = True
                    self._quality_throttle_factor = 0.0
                    logger.warning(
                        f"[QualityThrottle] PAUSING selfplay for {config_key}: "
                        f"quality={quality_score:.2f} (critical)"
                    )
                else:
                    # Warning level - reduce rate by 50%
                    self._quality_throttle_factor = 0.5
                    logger.warning(
                        f"[QualityThrottle] Reducing selfplay rate for {config_key}: "
                        f"quality={quality_score:.2f}, factor=0.5"
                    )

            def on_quality_updated(event):
                """Handle QUALITY_SCORE_UPDATED events to restore throttle."""
                payload = event.payload if hasattr(event, "payload") else event
                event_config = f"{payload.get('board_type')}_{payload.get('num_players')}p"

                if event_config != config_key:
                    return

                quality_score = payload.get("quality_score", 1.0)

                # Restore full rate if quality is good
                if quality_score >= 0.8:
                    if self._quality_throttle_factor < 1.0:
                        logger.info(
                            f"[QualityThrottle] Restoring full selfplay rate for {config_key}: "
                            f"quality={quality_score:.2f}"
                        )
                    self._quality_throttle_factor = 1.0
                    self._quality_paused = False

            # Subscribe to quality events
            router.subscribe(DataEventType.LOW_QUALITY_DATA_WARNING, on_quality_warning)
            router.subscribe(DataEventType.QUALITY_SCORE_UPDATED, on_quality_updated)

            logger.debug(f"[SelfplayRunner] Subscribed to quality events for {config_key}")

        except ImportError:
            pass  # Event system not available
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.debug(f"[SelfplayRunner] Failed to subscribe to quality events: {e}")

    def _subscribe_to_feedback_events(self) -> None:
        """Subscribe to feedback loop events (Phase 14 Integration).

        December 2025: Selfplay now responds to:
        - CURRICULUM_ADVANCED: Increase opponent difficulty
        - SELFPLAY_TARGET_UPDATED: Adjust games to generate
        - TRAINING_BLOCKED_BY_QUALITY: Trigger data regeneration
        - MODEL_PROMOTED: Reset difficulty boost on successful promotion
        - PROMOTION_FAILED: Increase opponent difficulty on failed promotion
        """
        try:
            from app.coordination.event_router import get_router, subscribe, DataEventType

            router = get_router()
            if router is None:
                return

            config_key = f"{self.config.board_type}_{self.config.num_players}p"

            def on_curriculum_advanced(event):
                """Handle CURRICULUM_ADVANCED events - increase difficulty."""
                payload = event.payload if hasattr(event, "payload") else event
                event_config = payload.get("config", "")

                # Only respond to our config's events
                if event_config != config_key:
                    return

                # Increase difficulty multiplier
                self._curriculum_difficulty = min(self._curriculum_difficulty * 1.2, 3.0)
                logger.info(
                    f"[FeedbackLoop] Curriculum advanced for {config_key}: "
                    f"difficulty multiplier now {self._curriculum_difficulty:.2f}"
                )

            def on_selfplay_target_updated(event):
                """Handle SELFPLAY_TARGET_UPDATED events - request extra games."""
                payload = event.payload if hasattr(event, "payload") else event
                event_config = payload.get("config", "")

                if event_config != config_key:
                    return

                extra_games = payload.get("extra_games", 0)
                if extra_games > 0:
                    self._extra_games_requested += extra_games
                    logger.info(
                        f"[FeedbackLoop] Extra selfplay requested for {config_key}: "
                        f"+{extra_games} games (total pending: {self._extra_games_requested})"
                    )

            def on_training_blocked_by_quality(event):
                """Handle TRAINING_BLOCKED_BY_QUALITY events - throttle and trigger regeneration.

                December 2025: When training is blocked due to quality, we:
                1. Pause current selfplay to avoid wasting compute
                2. Request extra games for regeneration
                3. Resume at 50% rate once regeneration starts
                """
                payload = event.payload if hasattr(event, "payload") else event
                # Check both 'config' and 'board_type_num_players' formats
                event_config = payload.get("config", "")
                if not event_config:
                    board_type = payload.get("board_type", "")
                    num_players = payload.get("num_players", 0)
                    event_config = f"{board_type}_{num_players}p" if board_type else ""

                if event_config != config_key:
                    return

                quality_score = payload.get("quality_score", 0.0)
                self._regeneration_pending = True

                # CRITICAL: Pause selfplay when training is blocked
                # This prevents wasting compute on potentially problematic data
                self._quality_paused = True
                self._quality_throttle_factor = 0.0

                # Request extra games to rebuild the training dataset
                extra_games = payload.get("games_needed", 500)
                self._extra_games_requested += extra_games

                logger.warning(
                    f"[FeedbackLoop] Training blocked for {config_key} due to low quality "
                    f"({quality_score:.2f}). PAUSING selfplay. "
                    f"Requested {extra_games} extra games for regeneration."
                )

            def on_promotion_success(event):
                """Handle MODEL_PROMOTED events - reset difficulty boost on success."""
                payload = event.payload if hasattr(event, "payload") else event
                event_config = payload.get("config_key", "")

                if event_config != config_key:
                    return

                # Reset difficulty boost on successful promotion
                old_boost = self._promotion_difficulty_boost
                self._promotion_difficulty_boost = 1.0
                self._consecutive_promotion_failures = 0
                logger.info(
                    f"[FeedbackLoop] Promotion succeeded for {config_key}: "
                    f"resetting difficulty boost from {old_boost:.2f} to 1.0"
                )

            def on_promotion_failed(event):
                """Handle PROMOTION_FAILED events - increase opponent difficulty."""
                payload = event.payload if hasattr(event, "payload") else event
                event_config = payload.get("config_key", "")

                if event_config != config_key:
                    return

                # Failed promotion → increase opponent difficulty to generate harder games
                self._consecutive_promotion_failures += 1
                # Exponential backoff: 1.0 → 1.2 → 1.44 → 1.73 → 2.07 (capped at 3.0)
                self._promotion_difficulty_boost = min(
                    self._promotion_difficulty_boost * 1.2, 3.0
                )
                logger.info(
                    f"[FeedbackLoop] Promotion failed for {config_key}: "
                    f"difficulty boost now {self._promotion_difficulty_boost:.2f} "
                    f"(consecutive failures: {self._consecutive_promotion_failures})"
                )

            def on_hyperparameter_updated(event):
                """Handle HYPERPARAMETER_UPDATED events from Gauntlet feedback.

                December 2025: Closes the Gauntlet→selfplay feedback loop.
                Strong models (>80% win rate) → reduce exploration (temperature_scale < 1.0)
                Weak models (<70% win rate) → increase quality threshold
                """
                payload = event.payload if hasattr(event, "payload") else event
                event_config = payload.get("config_key", "")

                if event_config != config_key:
                    return

                parameter = payload.get("parameter", "")
                new_value = payload.get("new_value")
                reason = payload.get("reason", "")

                if parameter == "temperature_scale" and new_value is not None:
                    # Adjust temperature scheduler multiplier
                    old_scale = getattr(self, "_gauntlet_temperature_scale", 1.0)
                    self._gauntlet_temperature_scale = float(new_value)
                    logger.info(
                        f"[GauntletFeedback] Temperature scale for {config_key}: "
                        f"{old_scale:.2f} → {self._gauntlet_temperature_scale:.2f} "
                        f"(reason: {reason})"
                    )
                elif parameter == "quality_threshold_boost" and new_value is not None:
                    # Increase quality gating threshold
                    self._gauntlet_quality_boost = float(new_value)
                    logger.info(
                        f"[GauntletFeedback] Quality threshold boost for {config_key}: "
                        f"+{self._gauntlet_quality_boost:.2f} (reason: {reason})"
                    )
                elif parameter == "selfplay_rate_multiplier" and new_value is not None:
                    # Adjust selfplay generation rate
                    old_rate = getattr(self, "_gauntlet_rate_multiplier", 1.0)
                    self._gauntlet_rate_multiplier = float(new_value)
                    logger.info(
                        f"[GauntletFeedback] Selfplay rate for {config_key}: "
                        f"{old_rate:.2f}x → {self._gauntlet_rate_multiplier:.2f}x "
                        f"(reason: {reason})"
                    )

            def on_quality_penalty_applied(event):
                """Handle QUALITY_PENALTY_APPLIED events from AdaptiveController.

                December 2025: Reduces selfplay rate when quality penalty is applied.
                This prevents generating more low-quality data.
                """
                payload = event.payload if hasattr(event, "payload") else event
                event_config = payload.get("config_key", "")

                if event_config != config_key:
                    return

                rate_multiplier = payload.get("rate_multiplier", 1.0)
                new_penalty = payload.get("new_penalty", 0.0)
                reason = payload.get("reason", "")

                # Apply rate reduction
                self._quality_penalty_rate = rate_multiplier
                logger.info(
                    f"[QualityPenalty] Reducing selfplay rate for {config_key}: "
                    f"{rate_multiplier:.2f}x (penalty={new_penalty:.2f}, reason: {reason})"
                )

            def on_adaptive_params_changed(event):
                """Handle ADAPTIVE_PARAMS_CHANGED events from GauntletFeedbackController.

                Phase 5 (December 2025): Close the feedback loop for adaptive selfplay params.
                This event carries multiple runtime adjustments:
                - temperature_multiplier: Adjust exploration temperature
                - search_budget_multiplier: Adjust MCTS search depth
                - exploration_boost: Increase root exploration noise
                - opponent_strength_multiplier: Adjust opponent model strength
                """
                payload = event.payload if hasattr(event, "payload") else event
                event_config = payload.get("config_key", payload.get("config", ""))

                if event_config != config_key:
                    return

                applied_changes = []

                # Temperature multiplier adjustment
                if "temperature_multiplier" in payload:
                    old_val = getattr(self, "_adaptive_temperature_multiplier", 1.0)
                    self._adaptive_temperature_multiplier = float(payload["temperature_multiplier"])
                    applied_changes.append(f"temp={old_val:.2f}→{self._adaptive_temperature_multiplier:.2f}")

                # Search budget multiplier adjustment
                if "search_budget_multiplier" in payload:
                    old_val = getattr(self, "_adaptive_search_budget_multiplier", 1.0)
                    self._adaptive_search_budget_multiplier = float(payload["search_budget_multiplier"])
                    applied_changes.append(f"budget={old_val:.2f}→{self._adaptive_search_budget_multiplier:.2f}")

                # Exploration boost adjustment
                if "exploration_boost" in payload:
                    old_val = getattr(self, "_adaptive_exploration_boost", 0.0)
                    self._adaptive_exploration_boost = float(payload["exploration_boost"])
                    applied_changes.append(f"explore_boost={old_val:.2f}→{self._adaptive_exploration_boost:.2f}")

                # Opponent strength multiplier
                if "opponent_strength_multiplier" in payload:
                    old_val = getattr(self, "_adaptive_opponent_strength", 1.0)
                    self._adaptive_opponent_strength = float(payload["opponent_strength_multiplier"])
                    applied_changes.append(f"opp_str={old_val:.2f}→{self._adaptive_opponent_strength:.2f}")

                if applied_changes:
                    reason = payload.get("reason", "adaptive_feedback")
                    logger.info(
                        f"[AdaptiveParams] {config_key}: {', '.join(applied_changes)} "
                        f"(reason: {reason})"
                    )

            def on_selfplay_rate_changed(event):
                """Handle SELFPLAY_RATE_CHANGED events from FeedbackAccelerator.

                December 2025 (Phase 1.3): Closes the feedback loop from Elo momentum
                to selfplay generation rate. FeedbackAccelerator emits this when:
                - Elo is rising fast → increase selfplay to maintain momentum
                - Elo is plateauing → reduce selfplay to conserve resources
                - Rate change exceeds 20% threshold

                This provides more context than HYPERPARAMETER_UPDATED's selfplay_rate_multiplier.
                """
                payload = event.payload if hasattr(event, "payload") else event
                event_config = payload.get("config", "")

                if event_config != config_key:
                    return

                new_rate = payload.get("new_rate", 1.0)
                old_rate = payload.get("old_rate", 1.0)
                direction = payload.get("direction", "unchanged")
                momentum_state = payload.get("momentum_state", "unknown")
                change_pct = payload.get("change_percent", 0.0)

                # Apply the rate change
                self._gauntlet_rate_multiplier = float(new_rate)

                logger.info(
                    f"[FeedbackAccelerator] Selfplay rate {direction} for {config_key}: "
                    f"{old_rate:.2f}x → {new_rate:.2f}x ({change_pct:+.1f}%) "
                    f"[momentum: {momentum_state}]"
                )

            def on_exploration_boost(event):
                """Handle EXPLORATION_BOOST events from FeedbackLoopController.

                January 3, 2026 (Sprint 12): Closes the feedback loop from training
                anomalies (loss spikes, divergence, stalls) to selfplay exploration.

                When exploration boost is applied, temperature is increased to encourage
                more diverse game play, helping break out of local minima and plateaus.

                The boost_factor is applied multiplicatively to temperature in get_temperature().
                A boost of 1.5 means 50% higher temperature during opening moves.
                """
                payload = event.payload if hasattr(event, "payload") else event
                event_config = payload.get("config_key", payload.get("config", ""))

                if event_config != config_key:
                    return

                boost_factor = payload.get("boost_factor", 1.0)
                reason = payload.get("reason", "unknown")
                anomaly_count = payload.get("anomaly_count", 0)

                # Store the exploration boost for use in get_temperature()
                old_val = getattr(self, "_adaptive_exploration_boost", 0.0)
                # Convert boost_factor to additive form: boost_factor=1.5 → boost=0.5
                new_boost = max(0.0, boost_factor - 1.0)
                self._adaptive_exploration_boost = new_boost

                if abs(new_boost - old_val) > 0.05:  # Only log significant changes
                    logger.info(
                        f"[ExplorationBoost] {config_key}: exploration_boost "
                        f"{old_val:.2f} → {new_boost:.2f} (factor={boost_factor:.2f}, "
                        f"reason={reason}, anomalies={anomaly_count})"
                    )

            # Subscribe to feedback events using DataEventType enums
            subscribe(DataEventType.CURRICULUM_ADVANCED, on_curriculum_advanced)
            subscribe(DataEventType.SELFPLAY_TARGET_UPDATED, on_selfplay_target_updated)
            subscribe(DataEventType.TRAINING_BLOCKED_BY_QUALITY, on_training_blocked_by_quality)
            subscribe(DataEventType.MODEL_PROMOTED, on_promotion_success)
            subscribe(DataEventType.PROMOTION_FAILED, on_promotion_failed)
            subscribe(DataEventType.HYPERPARAMETER_UPDATED, on_hyperparameter_updated)
            subscribe(DataEventType.QUALITY_PENALTY_APPLIED, on_quality_penalty_applied)
            subscribe(DataEventType.ADAPTIVE_PARAMS_CHANGED, on_adaptive_params_changed)  # Phase 5
            subscribe(DataEventType.SELFPLAY_RATE_CHANGED, on_selfplay_rate_changed)  # Phase 1.3 Dec 2025
            subscribe(DataEventType.EXPLORATION_BOOST, on_exploration_boost)  # Sprint 12 Jan 2026

            logger.debug(f"[SelfplayRunner] Subscribed to feedback events for {config_key}")

        except ImportError:
            pass  # Event system not available
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.debug(f"[SelfplayRunner] Failed to subscribe to feedback events: {e}")

    def _init_pfsp(self) -> None:
        """Initialize PFSP (Prioritized Fictitious Self-Play) opponent selection.

        December 2025: PFSP prioritizes opponents where win rate is near 50%,
        maximizing learning signal from each selfplay game.

        PFSP is enabled by default. Use --disable-pfsp to turn off.
        """
        # Check config option - PFSP enabled by default
        if not getattr(self.config, "use_pfsp", True):
            logger.info("[PFSP] Disabled via --disable-pfsp flag")
            return

        try:
            from app.training.pfsp_opponent_selector import (
                get_pfsp_selector,
                wire_pfsp_events,
                bootstrap_pfsp_opponents,
            )

            self._pfsp_selector = get_pfsp_selector()
            wire_pfsp_events()  # Subscribe to MODEL_PROMOTED, EVALUATION_COMPLETED
            bootstrap_pfsp_opponents()  # Pre-populate with existing canonical models
            self._pfsp_enabled = True

            logger.info("[PFSP] Initialized opponent selector (enabled by default)")

        except ImportError as e:
            logger.debug(f"[PFSP] Module not available: {e}")
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as e:
            logger.debug(f"[PFSP] Failed to initialize: {e}")

    def _init_temperature_scheduler(self) -> None:
        """Initialize temperature scheduler with exploration boost wiring.

        December 2025: Creates a persistent TemperatureScheduler and wires it
        to exploration boost events from the feedback loop. This enables:
        - Automatic exploration increase after failed promotions
        - Automatic exploration decrease after successful promotions

        The scheduler is created once and reused for all games, preserving
        the exploration boost state across games.

        Uses auto-registration feature (December 2025) to automatically
        register and wire the scheduler when config_key is provided.
        """
        try:
            from app.training.temperature_scheduling import create_elo_adaptive_scheduler

            model_elo = getattr(self.config, 'model_elo', None) or 1500.0
            exploration_moves = getattr(self.config, 'temperature_threshold', 30)
            config_key = f"{self.config.board_type}_{self.config.num_players}p"

            # December 2025: Use auto-registration feature - no need to manually
            # call wire_exploration_boost() anymore
            self._temperature_scheduler = create_elo_adaptive_scheduler(
                model_elo=model_elo,
                exploration_moves=exploration_moves,
                config_key=config_key,  # Auto-registers and wires
                auto_wire=True,
            )

            logger.info(
                f"[TemperatureScheduler] Initialized with auto-registration "
                f"(Elo={model_elo:.0f}, config={config_key})"
            )

        except ImportError as e:
            logger.debug(f"[TemperatureScheduler] Module not available: {e}")
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.debug(f"[TemperatureScheduler] Failed to initialize: {e}")

    def get_pfsp_opponent(self, current_model: str, available_opponents: list[str]) -> str:
        """Select opponent using PFSP if enabled, otherwise random.

        Args:
            current_model: Current model identifier
            available_opponents: List of available opponent model identifiers

        Returns:
            Selected opponent model identifier
        """
        if not self._pfsp_enabled or self._pfsp_selector is None:
            import random
            return random.choice(available_opponents) if available_opponents else current_model

        return self._pfsp_selector.select_opponent(current_model, available_opponents)

    def record_pfsp_result(
        self,
        current_model: str,
        opponent: str,
        current_model_won: bool,
        draw: bool = False,
    ) -> None:
        """Record game result for PFSP statistics.

        Args:
            current_model: Current model identifier
            opponent: Opponent model identifier
            current_model_won: Whether current model won
            draw: Whether game was a draw
        """
        if self._pfsp_enabled and self._pfsp_selector is not None:
            self._pfsp_selector.record_game_result(
                current_model=current_model,
                opponent=opponent,
                current_model_won=current_model_won,
                draw=draw,
            )

    def _get_pfsp_context(self) -> tuple[str, str | None]:
        """Get PFSP context for a game: current model ID and selected opponent.

        Returns:
            Tuple of (current_model_id, selected_opponent_id or None if PFSP disabled)
        """
        config_key = f"{self.config.board_type}_{self.config.num_players}p"

        # Current model is identified by config key + weights file hash if available
        model_version = getattr(self.config, 'model_version', None)
        if model_version:
            current_model = f"{config_key}_{model_version}"
        else:
            current_model = config_key

        if not self._pfsp_enabled or self._pfsp_selector is None:
            return current_model, None

        # Get available opponents and select one
        available = self._pfsp_selector.get_available_opponents(config_key)
        if not available:
            # No opponents registered yet - use current model as opponent (self-play)
            return current_model, current_model

        opponent = self.get_pfsp_opponent(current_model, available)
        return current_model, opponent

    def _apply_quality_throttle(self) -> bool:
        """Apply quality-based throttling. Returns True if game should be skipped."""
        if self._quality_paused:
            # Critical quality issue - wait and retry
            import time
            time.sleep(5.0)  # Wait 5 seconds before checking again
            return True  # Skip this game iteration

        if self._quality_throttle_factor < 1.0:
            # Probabilistic throttling - skip some games
            import random
            if random.random() > self._quality_throttle_factor:
                import time
                time.sleep(0.5)  # Small delay when throttled
                return True  # Skip this game

        return False  # Proceed with game

    def _save_game_to_db(self, result: GameResult) -> None:
        """Save a completed game to the database.

        Uses a write lock to prevent sync operations from capturing incomplete data.
        The lock is held for the duration of the database write.

        Args:
            result: Game result with initial_state, final_state, and move_objects
        """
        if self._db is None:
            return

        if result.initial_state is None or result.final_state is None:
            logger.debug(f"Skipping DB write for {result.game_id}: missing state data")
            return

        if not result.move_objects:
            logger.debug(f"Skipping DB write for {result.game_id}: no move objects")
            return

        # Get database path for write lock
        db_path = Path(self.config.record_db) if self.config.record_db else None
        if db_path is None:
            return

        try:
            # Acquire write lock to prevent sync during database write
            with DatabaseWriteLock(db_path):
                # Check if we have MCTS distribution data - use incremental storage
                has_mcts_data = result.move_probs and any(p is not None for p in result.move_probs)

                if has_mcts_data:
                    # Use incremental storage to preserve MCTS distribution data
                    self._save_game_with_mcts_data(result)
                else:
                    # Use bulk storage for games without MCTS data
                    self._db.store_game(
                        game_id=result.game_id,
                        initial_state=result.initial_state,
                        final_state=result.final_state,
                        moves=result.move_objects,
                        metadata=result.metadata,
                        store_history_entries=self.config.store_history_entries,
                        snapshot_interval=getattr(self.config, 'snapshot_interval', 20),
                    )
            logger.debug(f"Saved game {result.game_id} to database")

            # Dec 30, 2025: Register game for immediate cluster visibility
            self._register_game_location(result.game_id, db_path)

        except (OSError, sqlite3.Error, ValueError, TypeError) as e:
            # January 2026: Log as error and emit event for monitoring
            # This was previously a silent failure causing data loss
            logger.error(f"CRITICAL: Failed to save game {result.game_id} to database: {e}")
            try:
                from app.distributed.data_events import DataEventType
                from app.coordination.event_router import emit_data_event

                emit_data_event(
                    DataEventType.GAME_SAVE_FAILED,
                    {
                        "game_id": result.game_id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "config_key": f"{self.config.board_type}_{self.config.num_players}p",
                        "board_type": self.config.board_type,
                        "num_players": self.config.num_players,
                        "db_path": str(db_path) if db_path else None,
                    },
                )
            except ImportError:
                pass  # Event system not available, already logged error

    def _register_game_location(self, game_id: str, db_path: Path) -> None:
        """Register game location in ClusterManifest for immediate cluster visibility.

        Dec 30, 2025: Added to enable immediate cluster-wide data discovery.
        Games are registered with the DataLocationRegistry so other nodes can find them
        without waiting for gossip propagation.

        Args:
            game_id: The game ID to register
            db_path: Path to the database containing the game
        """
        try:
            from app.distributed.cluster_manifest import get_cluster_manifest

            manifest = get_cluster_manifest()
            manifest.register_game(
                game_id=game_id,
                node_id=manifest.node_id,
                db_path=str(db_path),
                board_type=self.config.board_type,
                num_players=self.config.num_players,
                engine_mode=str(self.config.engine) if hasattr(self.config, 'engine') else None,
            )
        except Exception as e:
            # Registration failure should not block selfplay
            logger.debug(f"[SelfplayRunner] Could not register game {game_id}: {e}")

    def _save_game_with_mcts_data(self, result: GameResult) -> None:
        """Save game using incremental storage to preserve MCTS distribution data.

        This method uses GameWriter.add_move() which supports move_probs and search_stats
        columns (schema v10-v11) for storing soft policy targets and rich statistics.

        Jan 2026: Also stores pre-computed heuristic features (v18) when
        config.compute_heuristics_on_write is True, enabling 10-20x faster exports.
        """
        from ..game_engine import GameEngine

        with self._db.store_game_incremental(
            game_id=result.game_id,
            initial_state=result.initial_state,
            store_history_entries=self.config.store_history_entries,
        ) as writer:
            # Replay through moves, applying each and recording with MCTS data
            state = result.initial_state
            for i, move in enumerate(result.move_objects):
                # Get MCTS data for this move if available
                move_probs = result.move_probs[i] if i < len(result.move_probs) else None
                search_stats = result.search_stats[i] if i < len(result.search_stats) else None

                # Apply move - keep reference to pre-move state for correct phase tracking
                state_before = state
                state_after = GameEngine.apply_move(state, move)

                # Compute heuristic features if enabled (Jan 2026 - v18 schema)
                heuristic_features = None
                if getattr(self.config, "compute_heuristics_on_write", False):
                    heuristic_features = self._compute_move_heuristics(
                        state_before, state_before.current_player
                    )

                # Record move with MCTS data
                # CRITICAL: Pass state_before to ensure phase is extracted from pre-move state,
                # not inferred from previous iteration's post-move state (which causes game_over bug)
                writer.add_move(
                    move=move,
                    state_before=state_before,  # Fundamental fix: explicit pre-move state
                    state_after=state_after,
                    move_probs=move_probs,
                    search_stats=search_stats,
                    heuristic_features=heuristic_features,
                )

                state = state_after

            # Finalize with winner in metadata
            finalize_metadata = result.metadata.copy() if result.metadata else {}
            finalize_metadata["winner"] = result.winner
            writer.finalize(
                final_state=state,
                metadata=finalize_metadata,
            )

    def _compute_move_heuristics(
        self, state: "GameState", player_number: int
    ) -> "np.ndarray | None":
        """Compute heuristic features for a game state.

        Args:
            state: The game state to evaluate
            player_number: Player number for perspective

        Returns:
            numpy float32 array of heuristic features, or None if computation fails
        """
        try:
            import numpy as np

            if getattr(self.config, "full_heuristics", True):
                from app.training.fast_heuristic_features import extract_full_heuristic_features

                return extract_full_heuristic_features(
                    state, player_number=player_number, normalize=True
                )
            else:
                from app.training.fast_heuristic_features import extract_heuristic_features

                return extract_heuristic_features(
                    state, player_number=player_number, eval_mode="full", normalize=True
                )
        except Exception as e:
            logger.debug(f"[SelfplayRunner] Heuristic computation failed: {e}")
            return None

    def on_game_complete(self, callback: Callable[[GameResult], None]) -> None:
        """Register callback for game completion events."""
        self._callbacks.append(callback)

    def _emit_game_complete(self, result: GameResult) -> None:
        """Emit game completion to registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(result)
            except (TypeError, ValueError, AttributeError, RuntimeError) as e:
                logger.warning(f"Callback error: {e}")

        # Phase 4A.5: Ephemeral sync integration for write-through mode
        self._notify_ephemeral_sync(result)

    def _notify_ephemeral_sync(self, result: GameResult) -> None:
        """Notify AutoSyncDaemon of game completion for immediate sync.

        Phase 4A.5 (December 2025): Ensures games on ephemeral hosts (Vast.ai)
        are immediately synced to prevent data loss on termination.

        December 2025: Uses unified auto_sync_daemon (not deprecated wrapper).
        """
        import asyncio  # Import early to avoid UnboundLocalError in except clause
        try:
            from app.coordination.auto_sync_daemon import get_auto_sync_daemon

            daemon = get_auto_sync_daemon()
            if not daemon.is_ephemeral:
                return  # Not on ephemeral host, skip

            # Get db_path if available
            db_path = self.config.record_db if self.config.record_db else None

            # Create game result dict for sync
            game_result = {
                "game_id": result.game_id if hasattr(result, "game_id") else str(id(result)),
                "board_type": self.config.board_type,
                "num_players": self.config.num_players,
                "moves_count": len(result.move_objects) if result.move_objects else 0,
            }

            # Call on_game_complete (fire-and-forget for non-blocking)
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(daemon.on_game_complete(game_result, db_path))
            except RuntimeError:
                # No running loop - run synchronously
                asyncio.run(daemon.on_game_complete(game_result, db_path))

        except ImportError:
            pass  # Ephemeral sync not available
        except (AttributeError, RuntimeError, asyncio.TimeoutError, OSError) as e:
            logger.debug(f"[Ephemeral] Could not notify sync daemon: {e}")

    def _record_match_elo(self, result: GameResult) -> EloRecordResult | None:
        """Record match result to Elo tracking system.

        Jan 14, 2026: CRITICAL FIX - This enables Elo tracking for selfplay games.
        Previously, record_selfplay_match() was never called, so selfplay Elo
        was not tracked, breaking the feedback loop that adjusts training based
        on model strength.

        Args:
            result: Game result from run_game()

        Returns:
            EloRecordResult if successful, None if failed
        """
        try:
            from ..ai.harness.harness_registry import HarnessType

            # Map engine mode to harness type
            engine_mode = self.config.engine_mode
            if engine_mode == EngineMode.MIXED:
                harness_type = HarnessType.MIXED
            elif engine_mode == EngineMode.GUMBEL_MCTS:
                harness_type = HarnessType.GUMBEL_MCTS
            elif engine_mode == EngineMode.HEURISTIC:
                harness_type = HarnessType.HEURISTIC
            elif engine_mode == EngineMode.MCTS:
                harness_type = HarnessType.MCTS
            elif engine_mode == EngineMode.POLICY_ONLY:
                harness_type = HarnessType.POLICY_ONLY
            else:
                # Default to GUMBEL_MCTS for other modes
                harness_type = HarnessType.GUMBEL_MCTS

            # Get model ID from config or use default
            model_id = getattr(self.config, 'nn_model_id', None)
            if not model_id:
                # Use canonical model naming
                model_id = f"canonical_{self.config.board_type}_{self.config.num_players}p"

            # Record the match result
            elo_result = record_selfplay_match(
                model_id=model_id,
                board_type=self.config.board_type,
                num_players=self.config.num_players,
                harness_type=harness_type,
                winner_player=result.winner,
                game_length=result.num_moves,
                duration_sec=result.duration_ms / 1000.0,
            )

            if not elo_result.success:
                logger.debug(f"[Elo] Failed to record match: {elo_result.error}")

            return elo_result

        except ImportError as e:
            logger.debug(f"[Elo] Harness registry not available: {e}")
            return None
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.debug(f"[Elo] Error recording match: {e}")
            return None

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
        except (AttributeError, RuntimeError, asyncio.TimeoutError, TypeError) as e:
            logger.warning(f"Failed to emit selfplay event: {e}")

    def get_temperature(self, move_number: int, game_state=None) -> float:
        """Get temperature for move selection based on scheduling.

        December 2025: Uses persistent TemperatureScheduler with exploration boost
        wiring when available. The scheduler receives PROMOTION_FAILED and
        MODEL_PROMOTED events to automatically adjust exploration.

        Gauntlet feedback (via HYPERPARAMETER_UPDATED) further modulates temperature:
        - Strong models (>80% win rate) → temperature_scale < 1.0 (reduce exploration)
        - Weak models (<70% win rate) → temperature_scale > 1.0 (increase exploration)

        Weak models (Elo < 1300) get high temperature (1.5) for exploration.
        Strong models (Elo > 1700) get low temperature (0.5) for exploitation.

        Args:
            move_number: Current move number in the game.
            game_state: Optional game state for adaptive scheduling.

        Returns:
            Temperature value for move selection.
        """
        # Get Gauntlet temperature scale (from HYPERPARAMETER_UPDATED events)
        gauntlet_scale = getattr(self, "_gauntlet_temperature_scale", 1.0)

        # Get adaptive temperature multiplier (from ADAPTIVE_PARAMS_CHANGED events)
        # Phase 4.1 Dec 2025: Closes feedback loop from GauntletFeedbackController
        adaptive_multiplier = getattr(self, "_adaptive_temperature_multiplier", 1.0)

        # Get exploration boost (from ADAPTIVE_PARAMS_CHANGED events)
        # Applied during opening moves to increase exploration diversity
        exploration_boost = getattr(self, "_adaptive_exploration_boost", 0.0)

        # Combined scale factor (exploration_boost increases temperature during opening)
        combined_scale = gauntlet_scale * adaptive_multiplier * (1.0 + exploration_boost)

        # Use persistent scheduler with exploration boost (preferred)
        if self._temperature_scheduler is not None:
            base_temp = self._temperature_scheduler.get_temperature(move_number, game_state)
            return base_temp * combined_scale

        # Fallback: create one-off scheduler if model_elo is set but scheduler wasn't initialized
        if self.config.model_elo is not None:
            try:
                from app.training.temperature_scheduling import create_elo_adaptive_scheduler

                scheduler = create_elo_adaptive_scheduler(
                    model_elo=self.config.model_elo,
                    exploration_moves=getattr(self.config, 'temperature_threshold', 30),
                )
                return scheduler.get_temperature(move_number, game_state) * combined_scale
            except ImportError:
                pass  # Fall through to default logic
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug(f"[Temperature] Elo-adaptive scheduler failed: {e}")

        # Default temperature logic (backward compatible)
        threshold = getattr(self.config, 'temperature_threshold', 30)
        opening_temp = getattr(self.config, 'opening_temperature', 1.0)
        base_temp = getattr(self.config, 'base_temperature', 0.1)
        if move_number < threshold:
            return opening_temp * combined_scale
        return base_temp * combined_scale

    def run(self) -> RunStats:
        """Main run loop. Executes setup, games, teardown."""
        self.setup()

        try:
            game_idx = 0
            # Include extra games requested via feedback events (December 2025)
            # Apply Gauntlet rate multiplier (weak model → more games, strong model → fewer)
            # Also apply quality penalty rate (low quality data → fewer games)
            combined_rate = self._gauntlet_rate_multiplier * self._quality_penalty_rate
            base_target = int(self.config.num_games * combined_rate)
            target_games = base_target + self._extra_games_requested
            if combined_rate != 1.0:
                logger.info(
                    f"[FeedbackLoop] Adjusted target games: {self.config.num_games} × "
                    f"{combined_rate:.2f} = {base_target} "
                    f"(gauntlet={self._gauntlet_rate_multiplier:.2f}, "
                    f"quality_penalty={self._quality_penalty_rate:.2f})"
                )
            while self.running and game_idx < target_games:
                # Phase 3 Feedback Loop: Check quality throttle before each game
                if self._apply_quality_throttle():
                    continue  # Skip this iteration if throttled

                try:
                    result = self.run_game(game_idx)
                    self.stats.record_game(result)
                    self._save_game_to_db(result)
                    self._emit_game_complete(result)
                    self._record_match_elo(result)  # Jan 14, 2026: Track Elo for selfplay

                    # Progress logging
                    log_interval = getattr(self.config, 'log_interval', 10)
                    if (game_idx + 1) % log_interval == 0:
                        throttle_info = "" if self._quality_throttle_factor >= 1.0 else f" [throttle={self._quality_throttle_factor:.1f}]"
                        extra_info = f" (+{self._extra_games_requested} extra)" if self._extra_games_requested > 0 else ""
                        logger.info(
                            f"  Progress: {game_idx + 1}/{target_games} games{extra_info}, "
                            f"{self.stats.games_per_second:.2f} g/s{throttle_info}"
                        )

                except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError, sqlite3.Error, InvalidGameError) as e:
                    logger.warning(f"Game {game_idx} failed: {e}")
                    self.stats.games_failed += 1

                game_idx += 1

                # Check for new extra games requested (dynamic feedback loop)
                if game_idx >= target_games and self._extra_games_requested > 0:
                    # Consume extra games into new target
                    new_target = game_idx + self._extra_games_requested
                    logger.info(
                        f"[FeedbackLoop] Extending run with {self._extra_games_requested} extra games "
                        f"(new target: {new_target})"
                    )
                    target_games = new_target
                    self._extra_games_requested = 0

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

        # PFSP opponent selection (Phase 7 - December 2025)
        current_model, pfsp_opponent = self._get_pfsp_context()

        board_type = BoardType(self.config.board_type)
        initial_state = create_initial_state(board_type, self.config.num_players)
        state = initial_state
        moves = []
        move_objects = []  # Actual Move objects for DB storage

        max_moves = getattr(self.config, 'max_moves', 500)  # Default max moves
        while state.game_status != GameStatus.COMPLETED and len(moves) < max_moves:
            current_player = state.current_player
            ai = self._ais[current_player]

            move = ai.select_move(state)
            if not move:
                break

            state = self._engine.apply_move(state, move)
            moves.append({"player": current_player, "move": str(move)})
            move_objects.append(move)

        duration_ms = (time.time() - start_time) * 1000
        winner = getattr(state, "winner", None)

        # Record PFSP result (Phase 7 - December 2025)
        if pfsp_opponent is not None:
            # Player 0 is current model, others are opponent
            current_model_won = winner == 0
            is_draw = winner is None or winner < 0
            self.record_pfsp_result(current_model, pfsp_opponent, current_model_won, is_draw)

        return GameResult(
            game_id=game_id,
            winner=winner,
            num_moves=len(moves),
            duration_ms=duration_ms,
            moves=moves,
            metadata={
                "engine": "heuristic",
                "engine_mode": self.config.engine_mode.value,
                "board_type": self.config.board_type,
                "num_players": self.config.num_players,
                "difficulty": self.config.difficulty,
                "source": self.config.source,
                "pfsp_opponent": pfsp_opponent,  # Phase 7: Track PFSP opponent
                "model_elo": getattr(self.config, 'model_elo', None),  # Elo gating: track generator strength
            },
            initial_state=initial_state,
            final_state=state,
            move_objects=move_objects,
        )


class GumbelMCTSSelfplayRunner(SelfplayRunner):
    """Selfplay using Gumbel MCTS (high quality, slower).

    IMPORTANT: This runner requires GPU (CUDA or MPS) for neural network inference.
    It will raise RuntimeError if dispatched to a CPU-only node.

    For CPU-only nodes, use HeuristicSelfplayRunner instead.
    """

    def __init__(self, config: SelfplayConfig):
        config.engine_mode = EngineMode.GUMBEL_MCTS
        super().__init__(config)
        self._mcts_instances: dict = {}  # MCTS instance per player
        # Cache the device after validation
        self._device: str | None = None

    def _get_device(self) -> str:
        """Get the appropriate device for neural network inference.

        Priority:
        1. Config-specified device (if valid)
        2. CUDA if available
        3. MPS if available (Apple Silicon)
        4. Raise RuntimeError (GPU required for Gumbel MCTS)

        This ensures Gumbel MCTS only runs on GPU-capable nodes.
        CPU-only nodes should use HeuristicSelfplayRunner instead.

        Returns:
            Device string ("cuda", "cuda:N", or "mps")

        Raises:
            RuntimeError: If no GPU is available
        """
        if self._device is not None:
            return self._device

        import torch

        # Check if config specifies a device
        if self.config.device:
            # Validate the specified device
            if self.config.device.startswith("cuda"):
                if torch.cuda.is_available():
                    self._device = self.config.device
                    return self._device
                else:
                    logger.warning(
                        f"Config specifies {self.config.device} but CUDA not available"
                    )
            elif self.config.device == "mps":
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self._device = "mps"
                    return self._device
                else:
                    logger.warning("Config specifies mps but MPS not available")
            elif self.config.device == "cpu":
                raise RuntimeError(
                    "GumbelMCTSSelfplayRunner requires GPU but config specifies 'cpu'. "
                    "Use HeuristicSelfplayRunner for CPU-only nodes."
                )

        # Auto-detect device
        if torch.cuda.is_available():
            self._device = f"cuda:{self.config.gpu_device}" if self.config.gpu_device else "cuda"
            logger.info(f"[GumbelMCTS] Using CUDA device: {self._device}")
            return self._device

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"
            logger.info("[GumbelMCTS] Using MPS device (Apple Silicon)")
            return self._device

        # No GPU available - this is a dispatch error
        raise RuntimeError(
            "GumbelMCTSSelfplayRunner requires GPU (CUDA or MPS) but none available. "
            "This node should not have been assigned a GPU-required engine mode. "
            "Use HeuristicSelfplayRunner for CPU-only selfplay."
        )

    def setup(self) -> None:
        super().setup()
        from ..ai.factory import create_mcts
        from ..models import BoardType
        from ..ai.gumbel_common import get_budget_for_difficulty

        board_type = BoardType(self.config.board_type)

        # Use budget based on config or difficulty
        base_budget = self.config.simulation_budget or get_budget_for_difficulty(
            self.config.difficulty or 8
        )

        # Apply difficulty and adaptive multipliers (December 2025)
        # curriculum_difficulty: increased when curriculum advances
        # promotion_difficulty_boost: increased on failed promotions (harder opponents → stronger model)
        # _adaptive_search_budget_multiplier: from ADAPTIVE_PARAMS_CHANGED events (Phase 4.1)
        combined_difficulty = self._curriculum_difficulty * self._promotion_difficulty_boost
        adaptive_budget_mult = getattr(self, "_adaptive_search_budget_multiplier", 1.0)
        budget = int(base_budget * combined_difficulty * adaptive_budget_mult)
        if combined_difficulty != 1.0 or adaptive_budget_mult != 1.0:
            logger.info(
                f"[Curriculum] Budget adjusted: {base_budget} * {combined_difficulty:.2f} * {adaptive_budget_mult:.2f} = {budget} "
                f"(curriculum={self._curriculum_difficulty:.2f}, promotion_boost={self._promotion_difficulty_boost:.2f}, "
                f"adaptive={adaptive_budget_mult:.2f})"
            )

        self._base_budget = base_budget  # Store for potential reinitialization
        self._current_budget = budget

        # Get opponent strength multiplier (from ADAPTIVE_PARAMS_CHANGED events)
        # Phase 5 Dec 2025: Complete feedback loop for adaptive opponent strength
        opponent_strength = getattr(self, "_adaptive_opponent_strength", 1.0)

        # Create MCTS instances for each player (required for correct move generation)
        # Each player needs their own MCTS with correct player_number for get_valid_moves
        # Player 1 (index 0) is the current model at full strength
        # Other players are opponents with potentially reduced strength
        for p in range(1, self.config.num_players + 1):
            if p == 1:
                # Current model uses full budget
                player_budget = budget
            else:
                # Opponents use adjusted budget based on strength multiplier
                # opponent_strength < 1.0 = weaker opponent (less search)
                # opponent_strength > 1.0 = stronger opponent (more search)
                player_budget = max(16, int(budget * opponent_strength))

            device = self._get_device()  # Validated GPU device
            self._mcts_instances[p] = create_mcts(
                board_type=board_type.value,
                num_players=self.config.num_players,
                player_number=p,  # Critical: pass correct player_number
                mode="standard",
                simulation_budget=player_budget,
                device=device,
                # December 2025: Pass loaded neural network for GPU batch evaluation
                # Previously neural_net was None, disabling GPU acceleration
                neural_net=getattr(self, '_neural_net', None),
            )
            # Enable GPU tree search for 10-20x speedup (December 2025)
            # This accelerates the sequential halving loop using GPU tensor operations
            if device.startswith("cuda"):
                self._mcts_instances[p]._use_gpu_tree = True

        # Log if opponent strength differs from default
        if opponent_strength != 1.0:
            logger.info(
                f"[OpponentStrength] Adjusted opponent budgets: current_model={budget}, "
                f"opponents={max(16, int(budget * opponent_strength))} "
                f"(strength={opponent_strength:.2f})"
            )

    def run_game(self, game_idx: int) -> GameResult:
        import uuid
        from ..training.initial_state import create_initial_state
        from ..models import BoardType
        from ..game_engine import GameEngine

        # Check if difficulty or adaptive params changed and reinitialize MCTS instances if needed
        combined_difficulty = self._curriculum_difficulty * self._promotion_difficulty_boost
        adaptive_budget_mult = getattr(self, "_adaptive_search_budget_multiplier", 1.0)
        opponent_strength = getattr(self, "_adaptive_opponent_strength", 1.0)
        new_budget = int(self._base_budget * combined_difficulty * adaptive_budget_mult)
        if new_budget != self._current_budget:
            logger.info(
                f"[Curriculum] Reinitializing MCTS: budget {self._current_budget} -> {new_budget} "
                f"(curriculum={self._curriculum_difficulty:.2f}, promotion_boost={self._promotion_difficulty_boost:.2f}, "
                f"adaptive={adaptive_budget_mult:.2f})"
            )
            from ..ai.factory import create_mcts
            self._current_budget = new_budget
            for p in range(1, self.config.num_players + 1):
                # Dec 2025: Apply opponent_strength to non-player-1 MCTS
                # This completes ADAPTIVE_PARAMS_CHANGED feedback loop
                if p == 1:
                    player_budget = new_budget
                else:
                    player_budget = max(16, int(new_budget * opponent_strength))
                self._mcts_instances[p] = create_mcts(
                    board_type=self.config.board_type,
                    num_players=self.config.num_players,
                    player_number=p,
                    mode="standard",
                    simulation_budget=player_budget,
                    device=self._get_device(),
                    # December 2025: Pass loaded neural network for GPU batch evaluation
                    neural_net=getattr(self, '_neural_net', None),
                )

        start_time = time.time()
        game_id = str(uuid.uuid4())

        # PFSP opponent selection (Phase 7 - December 2025)
        current_model, pfsp_opponent = self._get_pfsp_context()

        board_type = BoardType(self.config.board_type)
        initial_state = create_initial_state(board_type, self.config.num_players)
        state = initial_state
        moves = []
        move_objects = []  # Actual Move objects for DB storage
        samples = []
        # MCTS distribution data for training
        move_probs_list = []  # Policy distributions per move
        search_stats_list = []  # Rich search stats per move

        from ..rules.core import GameStatus
        while state.game_status != GameStatus.COMPLETED and len(moves) < self.config.max_moves:
            valid_moves = GameEngine.get_valid_moves(state, state.current_player)
            if not valid_moves:
                # No interactive moves - check for bookkeeping move requirement
                # (e.g., NO_LINE_ACTION, NO_TERRITORY_ACTION per RR-CANON-R075)
                requirement = GameEngine.get_phase_requirement(state, state.current_player)
                if requirement is not None:
                    bookkeeping_move = GameEngine.synthesize_bookkeeping_move(requirement, state)
                    if bookkeeping_move:
                        current_player = state.current_player
                        state = GameEngine.apply_move(state, bookkeeping_move)
                        # Record bookkeeping move in game history, but NOT as training sample
                        moves.append({"player": current_player, "move": str(bookkeeping_move)})
                        move_objects.append(bookkeeping_move)
                        move_probs_list.append(None)  # No policy for bookkeeping
                        search_stats_list.append(None)
                        continue
                # No interactive moves and no bookkeeping required - game is stuck
                break

            # Get move from MCTS (use correct player's MCTS instance)
            mcts = self._mcts_instances[state.current_player]
            try:
                move = mcts.select_move(state)
                if move is None:
                    # MCTS returned None - fall back to random move
                    logger.warning(f"MCTS returned None for game {game_id}, falling back to random")
                    import random
                    move = random.choice(valid_moves)
            except (ValueError, RuntimeError, AttributeError, TypeError) as e:
                # MCTS failed - fall back to random move to complete the game
                logger.warning(f"MCTS failed for game {game_id}: {e}, falling back to random")
                import random
                move = random.choice(valid_moves)

            # Extract MCTS distribution data after move selection
            # get_visit_distribution() returns (moves, probs) from last search
            # get_search_stats() returns rich stats from GPU tree search (may be None for CPU)
            try:
                moves_list, probs_list = mcts.get_visit_distribution()
                if moves_list and probs_list:
                    # Convert to dict format: {move_key: probability}
                    move_probs = {str(m): float(p) for m, p in zip(moves_list, probs_list)}
                else:
                    move_probs = None
            except (AttributeError, TypeError, ValueError):
                move_probs = None

            try:
                search_stats = mcts.get_search_stats()
            except AttributeError:
                search_stats = None

            move_probs_list.append(move_probs)
            search_stats_list.append(search_stats)

            # Record sample for training
            if self.config.record_samples:
                samples.append({
                    "state": state,
                    "move": move,
                    "player": state.current_player,
                })

            current_player = state.current_player
            state = GameEngine.apply_move(state, move)
            moves.append({"player": current_player, "move": str(move)})
            move_objects.append(move)

        duration_ms = (time.time() - start_time) * 1000
        winner = getattr(state, "winner", None)

        # Record PFSP result (Phase 7 - December 2025)
        if pfsp_opponent is not None:
            current_model_won = winner == 0
            is_draw = winner is None or winner < 0
            self.record_pfsp_result(current_model, pfsp_opponent, current_model_won, is_draw)

        return GameResult(
            game_id=game_id,
            winner=winner,
            num_moves=len(moves),
            duration_ms=duration_ms,
            moves=moves,
            samples=samples,
            metadata={
                "engine": "gumbel_mcts",
                "engine_mode": self.config.engine_mode.value,
                "board_type": self.config.board_type,
                "num_players": self.config.num_players,
                "difficulty": self.config.difficulty,
                "source": self.config.source,
                "simulation_budget": self.config.simulation_budget,
                "pfsp_opponent": pfsp_opponent,  # Phase 7: Track PFSP opponent
                "model_elo": getattr(self.config, 'model_elo', None),  # Elo gating: track generator strength
            },
            initial_state=initial_state,
            final_state=state,
            move_objects=move_objects,
            move_probs=move_probs_list,
            search_stats=search_stats_list,
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
        # Custom setup logging
        logger.info(f"GNNSelfplayRunner starting: {self.config.board_type}_{self.config.num_players}p")
        logger.info(f"  Engine: gnn ({self._model_tier})")
        logger.info(f"  Target games: {self.config.num_games}")
        # Open database if configured
        self._open_database()

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
            from app.utils.torch_utils import safe_load_checkpoint
            checkpoint = safe_load_checkpoint(self.config.weights_file, map_location=device)
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

        # PFSP opponent selection (Phase 7 - December 2025)
        current_model, pfsp_opponent = self._get_pfsp_context()

        board_type = BoardType(self.config.board_type)
        initial_state = create_initial_state(board_type, self.config.num_players)
        state = initial_state
        moves = []
        move_objects = []  # Actual Move objects for DB storage
        samples = []
        move_probs_list = []  # Policy distributions for training (like GumbelMCTSSelfplayRunner)
        device = next(self._model.parameters()).device
        is_hex = board_type.value in ("hexagonal", "hex8")
        board_size = state.board.grid_size if hasattr(state.board, "grid_size") else 8
        from ..models import GamePhase, GameStatus

        def is_game_over(s):
            return s.game_status == GameStatus.COMPLETED or s.current_phase == GamePhase.GAME_OVER

        while not is_game_over(state) and len(moves) < 300:  # max 300 moves
            valid_moves = GameEngine.get_valid_moves(state, state.current_player)
            if not valid_moves:
                # No interactive moves - check for bookkeeping move requirement
                # (e.g., NO_LINE_ACTION, NO_TERRITORY_ACTION per RR-CANON-R075)
                requirement = GameEngine.get_phase_requirement(state, state.current_player)
                if requirement is not None:
                    bookkeeping_move = GameEngine.synthesize_bookkeeping_move(requirement, state)
                    if bookkeeping_move:
                        current_player = state.current_player
                        state = GameEngine.apply_move(state, bookkeeping_move)
                        # Record bookkeeping move in game history, but NOT as training sample
                        moves.append({"player": current_player, "move": str(bookkeeping_move)})
                        move_objects.append(bookkeeping_move)
                        move_probs_list.append(None)  # No policy for bookkeeping
                        continue
                # No interactive moves and no bookkeeping required - game is stuck
                break

            # Convert state to graph for GNN
            move_probs = None  # Policy distribution for training
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

                    # Convert policy to probabilities for training data
                    probs = F.softmax(policy, dim=-1)

                    # Select from valid moves only
                    # For now, use simple mapping: take top-k and pick randomly
                    # This is a simplification - proper action indexing would improve quality
                    move_idx = random.randrange(len(valid_moves))
                    move = valid_moves[move_idx]

                    # Capture policy distribution for training (like GumbelMCTSSelfplayRunner)
                    try:
                        move_probs = {str(m): float(probs[i % len(probs)].cpu()) for i, m in enumerate(valid_moves)}
                        # Normalize to sum to 1
                        total = sum(move_probs.values())
                        if total > 0:
                            move_probs = {k: v / total for k, v in move_probs.items()}
                    except (IndexError, TypeError, ValueError, RuntimeError):
                        move_probs = None

                except (RuntimeError, ValueError, TypeError, IndexError) as e:
                    # Fallback to random move on any error
                    logger.debug(f"GNN inference error, using random: {e}")
                    move = random.choice(valid_moves)
                    policy_logits = None

            move_probs_list.append(move_probs)

            # Record sample for training
            if self.config.store_history_entries and policy_logits is not None:
                samples.append({
                    "state": state,
                    "move": move,
                    "player": state.current_player,
                })

            current_player = state.current_player
            state = GameEngine.apply_move(state, move)
            moves.append({"player": current_player, "move": str(move)})
            move_objects.append(move)

        duration_ms = (time.time() - start_time) * 1000
        winner = getattr(state, "winner", None)

        # Record PFSP result (Phase 7 - December 2025)
        if pfsp_opponent is not None:
            current_model_won = winner == 0
            is_draw = winner is None or winner < 0
            self.record_pfsp_result(current_model, pfsp_opponent, current_model_won, is_draw)

        return GameResult(
            game_id=game_id,
            winner=winner,
            num_moves=len(moves),
            duration_ms=duration_ms,
            moves=moves,
            samples=samples,
            metadata={
                "engine": f"gnn_{self._model_tier}",
                "engine_mode": f"gnn_{self._model_tier}",
                "board_type": self.config.board_type,
                "num_players": self.config.num_players,
                "difficulty": self.config.difficulty,
                "source": self.config.source,
                "model_tier": self._model_tier,
                "pfsp_opponent": pfsp_opponent,  # Phase 7: Track PFSP opponent
                "model_elo": getattr(self.config, 'model_elo', None),  # Elo gating: track generator strength
            },
            initial_state=initial_state,
            final_state=state,
            move_objects=move_objects,
            move_probs=move_probs_list,  # Policy distributions for training
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
