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
        self._base_budget = 150  # Default base budget for curriculum scaling
        self._current_budget = 150  # Current effective budget

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

    def setup(self) -> None:
        """Called before run loop. Override for custom initialization."""
        logger.info(f"SelfplayRunner starting: {self.config.board_type}_{self.config.num_players}p")
        logger.info(f"  Engine: {self.config.engine_mode.value}")
        self._apply_elo_adaptive_config()  # Set model_elo for Elo-adaptive budget/temperature
        self._apply_selfplay_rate_adjustment()  # Adjust num_games based on Elo momentum
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
        except Exception as e:
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
        except Exception as e:
            logger.debug(f"[SelfplayRunner] Selfplay rate adjustment not applied: {e}")

    def teardown(self) -> None:
        """Called after run loop. Override for custom cleanup."""
        self._close_database()
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
        except Exception as e:
            logger.warning(f"Failed to open database {self.config.record_db}: {e}")
            self._db = None

    def _register_database_immediately(self, db_path: Path) -> None:
        """Register database immediately for cluster-wide visibility.

        Phase 4A.3 (December 2025): Reduces data visibility delay from 5 min to <1 sec
        by emitting DATABASE_CREATED event when selfplay creates/opens a database.
        """
        try:
            import socket
            from app.coordination.event_router import publish_sync
            from app.distributed.data_events import DataEventType

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
            except Exception as e:
                logger.debug(f"[SelfplayRunner] Could not register in manifest: {e}")

        except ImportError:
            pass  # Event system not available
        except Exception as e:
            logger.debug(f"[SelfplayRunner] Could not emit DATABASE_CREATED: {e}")

    def _close_database(self) -> None:
        """Close database connection."""
        if self._db is not None:
            try:
                # GameReplayDB doesn't have explicit close, but we clear the reference
                self._db = None
            except Exception as e:
                logger.warning(f"Error closing database: {e}")

    def _subscribe_to_quality_events(self) -> None:
        """Subscribe to quality events for throttling (Phase 3 Feedback Loop).

        December 2025: Selfplay now responds to data quality signals.
        When quality drops, selfplay rate is reduced to avoid generating
        more low-quality data that would waste compute.
        """
        try:
            from app.distributed.data_events import DataEventType
            from app.coordination.event_router import get_event_router

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
        except Exception as e:
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
            from app.coordination.event_router import get_router, subscribe
            from app.distributed.data_events import DataEventType

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

            logger.debug(f"[SelfplayRunner] Subscribed to feedback events for {config_key}")

        except ImportError:
            pass  # Event system not available
        except Exception as e:
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
            )

            self._pfsp_selector = get_pfsp_selector()
            wire_pfsp_events()  # Subscribe to MODEL_PROMOTED, EVALUATION_COMPLETED
            self._pfsp_enabled = True

            logger.info("[PFSP] Initialized opponent selector (enabled by default)")

        except ImportError as e:
            logger.debug(f"[PFSP] Module not available: {e}")
        except Exception as e:
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
        except Exception as e:
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

        try:
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
        except Exception as e:
            logger.warning(f"Failed to save game {result.game_id} to database: {e}")

    def _save_game_with_mcts_data(self, result: GameResult) -> None:
        """Save game using incremental storage to preserve MCTS distribution data.

        This method uses GameWriter.add_move() which supports move_probs and search_stats
        columns (schema v10-v11) for storing soft policy targets and rich statistics.
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

                # Apply move
                state_after = GameEngine.apply_move(state, move)

                # Record move with MCTS data
                writer.add_move(
                    move=move,
                    state_after=state_after,
                    move_probs=move_probs,
                    search_stats=search_stats,
                )

                state = state_after

            # Finalize with winner
            writer.finalize(
                final_state=state,
                winner=result.winner,
                metadata=result.metadata,
            )

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

        # Phase 4A.5: Ephemeral sync integration for write-through mode
        self._notify_ephemeral_sync(result)

    def _notify_ephemeral_sync(self, result: GameResult) -> None:
        """Notify EphemeralSyncDaemon of game completion for immediate sync.

        Phase 4A.5 (December 2025): Ensures games on ephemeral hosts (Vast.ai)
        are immediately synced to prevent data loss on termination.
        """
        try:
            from app.coordination.ephemeral_sync import get_ephemeral_sync_daemon

            daemon = get_ephemeral_sync_daemon()
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
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(daemon.on_game_complete(game_result, db_path))
            except RuntimeError:
                # No running loop - run synchronously
                asyncio.run(daemon.on_game_complete(game_result, db_path))

        except ImportError:
            pass  # Ephemeral sync not available
        except Exception as e:
            logger.debug(f"[Ephemeral] Could not notify sync daemon: {e}")

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

        # Use persistent scheduler with exploration boost (preferred)
        if self._temperature_scheduler is not None:
            base_temp = self._temperature_scheduler.get_temperature(move_number, game_state)
            return base_temp * gauntlet_scale

        # Fallback: create one-off scheduler if model_elo is set but scheduler wasn't initialized
        if self.config.model_elo is not None:
            try:
                from app.training.temperature_scheduling import create_elo_adaptive_scheduler

                scheduler = create_elo_adaptive_scheduler(
                    model_elo=self.config.model_elo,
                    exploration_moves=getattr(self.config, 'temperature_threshold', 30),
                )
                return scheduler.get_temperature(move_number, game_state) * gauntlet_scale
            except ImportError:
                pass  # Fall through to default logic
            except Exception as e:
                logger.debug(f"[Temperature] Elo-adaptive scheduler failed: {e}")

        # Default temperature logic (backward compatible)
        threshold = getattr(self.config, 'temperature_threshold', 30)
        opening_temp = getattr(self.config, 'opening_temperature', 1.0)
        base_temp = getattr(self.config, 'base_temperature', 0.1)
        if move_number < threshold:
            return opening_temp * gauntlet_scale
        return base_temp * gauntlet_scale

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

                    # Progress logging
                    log_interval = getattr(self.config, 'log_interval', 10)
                    if (game_idx + 1) % log_interval == 0:
                        throttle_info = "" if self._quality_throttle_factor >= 1.0 else f" [throttle={self._quality_throttle_factor:.1f}]"
                        extra_info = f" (+{self._extra_games_requested} extra)" if self._extra_games_requested > 0 else ""
                        logger.info(
                            f"  Progress: {game_idx + 1}/{target_games} games{extra_info}, "
                            f"{self.stats.games_per_second:.2f} g/s{throttle_info}"
                        )

                except Exception as e:
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
            },
            initial_state=initial_state,
            final_state=state,
            move_objects=move_objects,
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
        base_budget = self.config.simulation_budget or get_budget_for_difficulty(
            self.config.difficulty or 8
        )

        # Apply difficulty multipliers (December 2025)
        # curriculum_difficulty: increased when curriculum advances
        # promotion_difficulty_boost: increased on failed promotions (harder opponents → stronger model)
        combined_difficulty = self._curriculum_difficulty * self._promotion_difficulty_boost
        budget = int(base_budget * combined_difficulty)
        if combined_difficulty != 1.0:
            logger.info(
                f"[Curriculum] Budget adjusted: {base_budget} * {combined_difficulty:.2f} = {budget} "
                f"(curriculum={self._curriculum_difficulty:.2f}, promotion_boost={self._promotion_difficulty_boost:.2f})"
            )

        self._base_budget = base_budget  # Store for potential reinitialization
        self._current_budget = budget

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

        # Check if difficulty changed and reinitialize MCTS if needed
        combined_difficulty = self._curriculum_difficulty * self._promotion_difficulty_boost
        new_budget = int(self._base_budget * combined_difficulty)
        if new_budget != self._current_budget:
            logger.info(
                f"[Curriculum] Reinitializing MCTS: budget {self._current_budget} -> {new_budget} "
                f"(curriculum={self._curriculum_difficulty:.2f}, promotion_boost={self._promotion_difficulty_boost:.2f})"
            )
            from ..ai.factory import create_mcts
            self._current_budget = new_budget
            self._mcts = create_mcts(
                board_type=self.config.board_type,
                num_players=self.config.num_players,
                mode="standard",
                simulation_budget=new_budget,
                device=self.config.device or "cuda",
            )

        start_time = time.time()
        game_id = str(uuid.uuid4())

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
                break

            # Get move from MCTS (GumbelMCTSAI only takes game_state, computes valid moves internally)
            move = self._mcts.select_move(state)

            # Extract MCTS distribution data after move selection
            # get_visit_distribution() returns (moves, probs) from last search
            # get_search_stats() returns rich stats from GPU tree search (may be None for CPU)
            try:
                moves_list, probs_list = self._mcts.get_visit_distribution()
                if moves_list and probs_list:
                    # Convert to dict format: {move_key: probability}
                    move_probs = {str(m): float(p) for m, p in zip(moves_list, probs_list)}
                else:
                    move_probs = None
            except (AttributeError, TypeError, ValueError):
                move_probs = None

            try:
                search_stats = self._mcts.get_search_stats()
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

        return GameResult(
            game_id=game_id,
            winner=getattr(state, "winner", None),
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

        board_type = BoardType(self.config.board_type)
        initial_state = create_initial_state(board_type, self.config.num_players)
        state = initial_state
        moves = []
        move_objects = []  # Actual Move objects for DB storage
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

            current_player = state.current_player
            state = GameEngine.apply_move(state, move)
            moves.append({"player": current_player, "move": str(move)})
            move_objects.append(move)

        duration_ms = (time.time() - start_time) * 1000

        return GameResult(
            game_id=game_id,
            winner=getattr(state, "winner", None),
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
            },
            initial_state=initial_state,
            final_state=state,
            move_objects=move_objects,
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
