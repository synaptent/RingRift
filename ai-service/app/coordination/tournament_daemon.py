"""Tournament Scheduling Daemon - Automatic tournament execution on model events.

This daemon automatically schedules and runs tournaments when:
1. A new model is trained (TRAINING_COMPLETED event)
2. A model is promoted (MODEL_PROMOTED event)
3. Periodic ladder tournaments (configurable interval)

Features:
- Subscribes to training/promotion events
- Auto-schedules evaluation tournaments using RoundRobinScheduler
- Integrates with EloService for rating updates
- Supports gauntlet-style evaluation against baselines
- Emits EVALUATION_COMPLETED events for downstream processing

Usage:
    from app.coordination.tournament_daemon import (
        TournamentDaemon,
        TournamentDaemonConfig,
        get_tournament_daemon,
    )

    # Get singleton daemon
    daemon = get_tournament_daemon()

    # Start daemon (subscribes to events)
    await daemon.start()

    # Manually trigger evaluation
    await daemon.evaluate_model("path/to/model.pth", "hex8", 2)

    # Stop daemon
    await daemon.stop()
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# December 2025: Use consolidated daemon stats base class
from app.coordination.daemon_stats import EvaluationDaemonStats
from app.coordination.event_utils import make_config_key, parse_config_key
from app.coordination.handler_base import HandlerBase
from app.coordination.contracts import CoordinatorStatus, HealthCheckResult

# December 30, 2025: Game count for graduated thresholds
from app.utils.game_discovery import get_game_counts_summary

__all__ = [
    "TournamentDaemon",
    "TournamentDaemonConfig",
    "TournamentStats",
    "get_tournament_daemon",
    "reset_tournament_daemon",
]


@dataclass
class TournamentDaemonConfig:
    """Configuration for the tournament daemon."""
    # Event subscriptions
    trigger_on_training_completed: bool = True
    trigger_on_model_promoted: bool = False  # Usually not needed since promotion follows evaluation

    # Periodic tournaments
    enable_periodic_ladder: bool = True
    ladder_interval_seconds: float = 3600.0  # 1 hour

    # Evaluation settings
    games_per_evaluation: int = 20
    games_per_baseline: int = 10
    # Extended baselines for diverse Elo population (Dec 2025)
    # Includes random, heuristic variants, MCTS, and NNUE at different strengths
    baselines: list[str] = field(default_factory=lambda: [
        "random",           # ~400 Elo - baseline anchor
        "heuristic",        # ~1200 Elo - standard gate
        "heuristic_strong", # ~1400 Elo - difficulty 8
        "mcts_light",       # ~1500 Elo - 32 simulations
        "mcts_medium",      # ~1700 Elo - 128 simulations
    ])

    # Dec 31, 2025: NNUE baselines for architecture diversity
    # Added separately since they require NNUE model for the config
    nnue_baselines_2p: list[str] = field(default_factory=lambda: [
        "nnue_minimax_d4",  # ~1600 Elo - NNUE + minimax depth 4 (2-player only)
    ])
    nnue_baselines_mp: list[str] = field(default_factory=lambda: [
        "nnue_brs_d3",      # ~1550 Elo - NNUE + BRS depth 3 (3-4 player)
        "nnue_maxn_d3",     # ~1650 Elo - NNUE + MaxN depth 3 (3-4 player)
    ])
    enable_nnue_baselines: bool = True  # Test against NNUE baselines when available

    # Game recording for training data (Dec 2025)
    # When enabled, tournament games are saved to canonical databases for training
    enable_game_recording: bool = True
    recording_db_prefix: str = "tournament"
    recording_db_dir: str = "data/games"

    # Calibration tournaments - validate Elo ladder (Dec 2025)
    enable_calibration_tournaments: bool = True
    calibration_interval_seconds: float = 3600.0 * 24  # Daily
    calibration_games: int = 10  # Games per calibration matchup

    # Cross-NN version tournaments - compare model versions (Dec 2025)
    # NOTE: Disabled Jan 2026 - UnifiedNeuralNetFactory not implemented
    enable_cross_nn_tournaments: bool = False
    cross_nn_interval_seconds: float = 3600.0 * 4  # Every 4 hours
    cross_nn_games_per_pairing: int = 20

    # Multi-harness evaluation - evaluate model under all compatible harnesses (Dec 31, 2025)
    # Tests model with MINIMAX, MAXN, BRS, GUMBEL_MCTS, etc. to find best (model, harness) combo
    enable_multi_harness_evaluation: bool = True
    multi_harness_games_per_baseline: int = 10
    # Only run multi-harness after model passes basic RANDOM/HEURISTIC gate
    multi_harness_min_baseline_winrate: float = 0.60  # 60% vs heuristic required

    # Concurrency
    max_concurrent_games: int = 4

    # Timeouts
    game_timeout_seconds: float = 300.0  # 5 minutes per game
    evaluation_timeout_seconds: float = 1800.0  # 30 minutes per evaluation


@dataclass
class TournamentStats(EvaluationDaemonStats):
    """Statistics about tournament daemon activity.

    December 2025: Now extends EvaluationDaemonStats for consistent tracking.
    Inherits: evaluations_completed, evaluations_failed, games_played,
              models_evaluated, promotions_triggered, is_healthy(), etc.
    """

    # Tournament-specific fields
    event_triggers: int = 0

    # Backward compatibility aliases
    @property
    def tournaments_completed(self) -> int:
        """Alias for evaluations_completed (backward compatibility)."""
        return self.evaluations_completed

    @property
    def last_tournament_time(self) -> float:
        """Alias for last_evaluation_time (backward compatibility)."""
        return self.last_evaluation_time

    @property
    def errors(self) -> list[str]:
        """Return last error as list for backward compatibility."""
        if self.last_error:
            return [self.last_error]
        return []

    def record_tournament_success(self, games: int = 0) -> None:
        """Record a successful tournament."""
        self.record_evaluation_success(duration=0.0, games=games)

    def record_tournament_failure(self, error: str) -> None:
        """Record a failed tournament."""
        self.record_evaluation_failure(error)

    def record_event_trigger(self) -> None:
        """Record an event trigger."""
        self.event_triggers += 1
        self.evaluations_triggered += 1


class TournamentDaemon(HandlerBase):
    """Daemon that automatically schedules tournaments based on events.

    Subscribes to training/promotion events and triggers appropriate
    evaluation tournaments using the existing tournament infrastructure.

    January 2026: Migrated to HandlerBase for unified lifecycle and singleton.
    Uses _on_start() to spawn multiple background tasks with different intervals.
    """

    def __init__(self, config: TournamentDaemonConfig | None = None):
        """Initialize the tournament daemon.

        Args:
            config: Daemon configuration
        """
        self.config = config or TournamentDaemonConfig()
        self.node_id = socket.gethostname()

        # Initialize HandlerBase - use evaluation queue check interval as cycle
        super().__init__(
            name=f"tournament_daemon_{self.node_id}",
            cycle_interval=10.0,  # Check evaluation queue every 10s
        )

        # Background tasks for different intervals
        self._periodic_task: asyncio.Task | None = None
        self._calibration_task: asyncio.Task | None = None  # Dec 2025
        self._cross_nn_task: asyncio.Task | None = None  # Dec 2025
        self._stats = TournamentStats()
        self._evaluation_queue: asyncio.Queue = asyncio.Queue()

    def _get_event_subscriptions(self) -> dict:
        """Get declarative event subscriptions (HandlerBase pattern).

        Returns:
            Dict mapping event names to handler methods
        """
        subscriptions = {}

        if self.config.trigger_on_training_completed:
            subscriptions["TRAINING_COMPLETED"] = self._on_training_completed

        if self.config.trigger_on_model_promoted:
            subscriptions["MODEL_PROMOTED"] = self._on_model_promoted

        return subscriptions

    async def _on_start(self) -> None:
        """Hook called after HandlerBase start (spawns background tasks)."""
        # Start periodic ladder tournaments
        if self.config.enable_periodic_ladder:
            self._periodic_task = self._safe_create_task(
                self._periodic_ladder_loop(),
                context="tournament_periodic_ladder"
            )

        # Start calibration tournaments (Dec 2025)
        if self.config.enable_calibration_tournaments:
            self._calibration_task = self._safe_create_task(
                self._calibration_loop(),
                context="tournament_calibration"
            )

        # Start cross-NN version tournaments (Dec 2025)
        if self.config.enable_cross_nn_tournaments:
            self._cross_nn_task = self._safe_create_task(
                self._cross_nn_loop(),
                context="tournament_cross_nn"
            )

        logger.info(
            f"TournamentDaemon started (periodic_ladder={self.config.enable_periodic_ladder}, "
            f"calibration={self.config.enable_calibration_tournaments}, "
            f"cross_nn={self.config.enable_cross_nn_tournaments})"
        )

    async def _on_stop(self) -> None:
        """Hook called before HandlerBase stop (cancels background tasks)."""
        # Cancel periodic task
        if self._periodic_task:
            self._periodic_task.cancel()
            try:
                await asyncio.wait_for(self._periodic_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._periodic_task = None

        # Cancel calibration task (Dec 2025)
        if self._calibration_task:
            self._calibration_task.cancel()
            try:
                await asyncio.wait_for(self._calibration_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._calibration_task = None

        # Cancel cross-NN task (Dec 2025)
        if self._cross_nn_task:
            self._cross_nn_task.cancel()
            try:
                await asyncio.wait_for(self._cross_nn_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._cross_nn_task = None

        logger.info("TournamentDaemon stopped")

    async def _run_cycle(self) -> None:
        """Process evaluation queue (HandlerBase main work loop).

        Called every cycle_interval (10s) to check for pending evaluations.
        """
        try:
            # Non-blocking check for evaluation request
            try:
                request = self._evaluation_queue.get_nowait()
            except asyncio.QueueEmpty:
                return  # Nothing to process

            # Run evaluation
            await self._run_evaluation(
                model_path=request["model_path"],
                board_type=request["board_type"],
                num_players=request["num_players"],
                trigger=request.get("trigger", "unknown"),
            )
            self._record_success()

        except Exception as e:
            logger.error(f"Error processing evaluation: {e}")
            self._record_error(str(e))

    async def start(self) -> None:
        """Start the tournament daemon (delegates to HandlerBase)."""
        await super().start()

    async def stop(self) -> None:
        """Stop the tournament daemon (delegates to HandlerBase)."""
        await super().stop()

    def is_running(self) -> bool:
        """Check if daemon is running."""
        return self._running


    def _on_training_completed(self, event) -> None:
        """Handle TRAINING_COMPLETED event."""
        self._stats.event_triggers += 1

        payload = getattr(event, "payload", {}) or {}
        model_path = payload.get("model_path")
        config = payload.get("config", "")
        success = payload.get("success", True)

        if not success:
            logger.debug(f"Skipping evaluation for failed training: {config}")
            return

        if not model_path:
            logger.warning(f"TRAINING_COMPLETED event missing model_path: {config}")
            return

        # Parse config
        board_type, num_players = self._parse_config(config)
        if not board_type:
            logger.warning(f"Could not parse config: {config}")
            return

        logger.info(f"Training completed for {config}, queueing evaluation")

        # Queue evaluation
        self._evaluation_queue.put_nowait({
            "model_path": model_path,
            "board_type": board_type,
            "num_players": num_players,
            "trigger": "training_completed",
        })

    def _on_model_promoted(self, event) -> None:
        """Handle MODEL_PROMOTED event."""
        self._stats.event_triggers += 1

        payload = getattr(event, "payload", {}) or {}
        model_id = payload.get("model_id")
        config = payload.get("config", "")

        logger.info(f"Model promoted: {model_id} ({config})")

        # Promotion events don't need immediate evaluation since
        # evaluation already passed to trigger promotion

    def _parse_config(self, config: str) -> tuple[str | None, int | None]:
        """Parse board_type and num_players from config string.

        December 30, 2025: Migrated to use consolidated parse_config_key utility.
        """
        parsed = parse_config_key(config)
        if parsed:
            return parsed.board_type, parsed.num_players
        return None, None


    async def _periodic_ladder_loop(self) -> None:
        """Periodic ladder tournament loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.ladder_interval_seconds)

                if not self._running:
                    break

                logger.info("Running periodic ladder tournament")
                await self._run_ladder_tournament()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic ladder: {e}")
                self._stats.record_failure(str(e))

    async def _calibration_loop(self) -> None:
        """Periodic calibration tournament loop (Dec 2025).

        Runs calibration tournaments at configured interval to validate
        that Elo ladder gaps match expected win rates.
        """
        while self._running:
            try:
                await asyncio.sleep(self.config.calibration_interval_seconds)

                if not self._running:
                    break

                logger.info("Running scheduled calibration tournament")
                results = await self._run_calibration_tournament()

                if results.get("all_valid"):
                    logger.info("Calibration tournament: all matchups valid")
                else:
                    logger.warning(f"Calibration tournament: some matchups invalid")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in calibration loop: {e}")
                self._stats.record_failure(str(e))

    async def _cross_nn_loop(self) -> None:
        """Periodic cross-NN version tournament loop (Dec 2025).

        Runs tournaments between different NN versions (v2, v3, v4, etc.)
        to track model evolution and maintain accurate Elo ratings.
        """
        while self._running:
            try:
                await asyncio.sleep(self.config.cross_nn_interval_seconds)

                if not self._running:
                    break

                logger.info("Running scheduled cross-NN tournament")
                results = await self._run_cross_nn_tournament()

                if results.get("success"):
                    games_played = results.get("games_played", 0)
                    logger.info(f"Cross-NN tournament completed: {games_played} games")
                else:
                    logger.warning(f"Cross-NN tournament failed: {results.get('error')}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cross-NN loop: {e}")
                self._stats.record_failure(str(e))

    async def _run_evaluation(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
        trigger: str = "manual",
    ) -> dict[str, Any]:
        """Run evaluation for a trained model.

        Args:
            model_path: Path to model checkpoint
            board_type: Board type
            num_players: Number of players
            trigger: What triggered this evaluation

        Returns:
            Evaluation results dict
        """
        self._stats.evaluations_triggered += 1
        start_time = time.time()

        logger.info(f"Starting evaluation: {model_path} ({board_type}_{num_players}p)")

        results = {
            "model_path": model_path,
            "board_type": board_type,
            "num_players": num_players,
            "trigger": trigger,
            "success": False,
            "win_rates": {},
            "elo": None,
            "games_played": 0,
            "duration_seconds": 0.0,
        }

        try:
            # Run gauntlet evaluation
            from app.training.game_gauntlet import BaselineOpponent, run_baseline_gauntlet

            # Create recording config if enabled (Dec 2025 - tournament games for training)
            recording_config = None
            if self.config.enable_game_recording:
                try:
                    from app.db.unified_recording import RecordingConfig, RecordSource
                    recording_config = RecordingConfig(
                        board_type=board_type,
                        num_players=num_players,
                        source=RecordSource.TOURNAMENT,
                        engine_mode="gauntlet",
                        db_prefix=self.config.recording_db_prefix,
                        db_dir=self.config.recording_db_dir,
                        store_history_entries=True,
                    )
                    logger.debug(f"Tournament recording enabled for {board_type}_{num_players}p")
                except ImportError:
                    logger.debug("Recording module not available, games will not be saved")

            # Dec 30, 2025: Get game count for graduated thresholds
            config_key = make_config_key(board_type, num_players)
            try:
                game_counts = get_game_counts_summary()
                game_count = game_counts.get(config_key, 0)
            except (OSError, RuntimeError):
                game_count = None  # Will use fallback thresholds

            gauntlet_results = await asyncio.wait_for(
                asyncio.to_thread(
                    run_baseline_gauntlet,
                    model_path=model_path,
                    board_type=board_type,
                    num_players=num_players,
                    games_per_opponent=self.config.games_per_baseline,
                    opponents=[BaselineOpponent.RANDOM, BaselineOpponent.HEURISTIC],
                    recording_config=recording_config,
                    game_count=game_count,  # Dec 30: Graduated thresholds
                ),
                timeout=self.config.evaluation_timeout_seconds,
            )

            results["success"] = True
            results["games_played"] = gauntlet_results.total_games
            results["win_rates"] = {
                opponent: stats.get("win_rate", 0.0)
                for opponent, stats in gauntlet_results.opponent_results.items()
            }

            # Update ELO
            if gauntlet_results.estimated_elo:
                results["elo"] = gauntlet_results.estimated_elo
                await self._update_elo(model_path, board_type, num_players, gauntlet_results)

            self._stats.games_played += results["games_played"]
            self._stats.last_evaluation_time = time.time()

            logger.info(
                f"Evaluation complete: {model_path} - "
                f"win_rates={results['win_rates']}, elo={results['elo']}"
            )

            # Dec 31, 2025: Multi-harness evaluation for harness diversity
            # Only run if model passed the baseline gate (60%+ vs heuristic)
            heuristic_winrate = results["win_rates"].get("heuristic", 0.0)
            if (
                self.config.enable_multi_harness_evaluation
                and heuristic_winrate >= self.config.multi_harness_min_baseline_winrate
            ):
                try:
                    multi_harness_results = await self._run_multi_harness_evaluation(
                        model_path, board_type, num_players
                    )
                    results["multi_harness"] = multi_harness_results
                    logger.info(
                        f"Multi-harness evaluation complete: best_harness={multi_harness_results.get('best_harness')}, "
                        f"best_elo={multi_harness_results.get('best_elo')}"
                    )
                except Exception as e:
                    logger.warning(f"Multi-harness evaluation failed (non-fatal): {e}")
                    results["multi_harness_error"] = str(e)
            elif self.config.enable_multi_harness_evaluation:
                logger.debug(
                    f"Skipping multi-harness: heuristic winrate {heuristic_winrate:.1%} < "
                    f"{self.config.multi_harness_min_baseline_winrate:.1%} threshold"
                )

            # Dec 31, 2025: NNUE baseline evaluation for architecture diversity
            # Tests model against NNUE-based baselines when NNUE model exists for config
            if self.config.enable_nnue_baselines and heuristic_winrate >= 0.50:
                try:
                    nnue_results = await self._run_nnue_baseline_evaluation(
                        model_path, board_type, num_players, recording_config
                    )
                    if nnue_results:
                        results["nnue_baselines"] = nnue_results
                        logger.info(
                            f"NNUE baseline evaluation complete: {nnue_results}"
                        )
                except Exception as e:
                    logger.warning(f"NNUE baseline evaluation failed (non-fatal): {e}")
                    results["nnue_baseline_error"] = str(e)

        except asyncio.TimeoutError:
            logger.error(f"Evaluation timeout: {model_path}")
            results["error"] = "timeout"
            self._stats.record_failure(f"Evaluation timeout: {model_path}")

        except ImportError as e:
            logger.warning(f"GameGauntlet not available: {e}")
            # Fall back to basic match execution
            results = await self._run_basic_evaluation(
                model_path, board_type, num_players
            )

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            results["error"] = str(e)
            self._stats.record_failure(str(e))

        results["duration_seconds"] = time.time() - start_time

        # Emit EVALUATION_COMPLETED event
        await self._emit_evaluation_completed(results)

        return results

    async def _run_basic_evaluation(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
    ) -> dict[str, Any]:
        """Run basic evaluation when GameGauntlet is not available.

        Uses the scheduler directly for match generation.
        """
        results = {
            "model_path": model_path,
            "board_type": board_type,
            "num_players": num_players,
            "success": False,
            "win_rates": {},
            "games_played": 0,
        }

        try:
            from app.models import BoardType
            from app.tournament.scheduler import RoundRobinScheduler

            # Create scheduler
            scheduler = RoundRobinScheduler(
                games_per_pairing=2,
                shuffle_order=True,
            )

            # Generate matches against baselines
            model_id = Path(model_path).stem
            agents = [model_id] + self.config.baselines

            board_type_enum = BoardType(board_type)
            matches = scheduler.generate_matches(
                agent_ids=agents,
                board_type=board_type_enum,
                num_players=num_players,
            )

            logger.info(f"Generated {len(matches)} matches for basic evaluation")

            # Note: Match execution would require game engine integration
            # For now, just report the scheduled matches
            results["scheduled_matches"] = len(matches)
            results["success"] = True

        except Exception as e:
            logger.error(f"Basic evaluation failed: {e}")
            results["error"] = str(e)

        return results

    async def _run_multi_harness_evaluation(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
    ) -> dict[str, Any]:
        """Run multi-harness evaluation to find best (model, harness) combo.

        December 31, 2025: Integrates MultiHarnessGauntlet for harness diversity.
        Tests model under MINIMAX, MAXN, BRS, GUMBEL_MCTS, etc.

        Args:
            model_path: Path to model checkpoint
            board_type: Board type
            num_players: Number of players

        Returns:
            Dict with per-harness Elo ratings and best harness
        """
        from app.training.multi_harness_gauntlet import MultiHarnessGauntlet

        gauntlet = MultiHarnessGauntlet(
            default_games_per_baseline=self.config.multi_harness_games_per_baseline,
            default_baselines=["random", "heuristic"],
        )

        # Run evaluation under all compatible harnesses
        result = await gauntlet.evaluate_model(
            model_path=model_path,
            board_type=board_type,
            num_players=num_players,
        )

        # Extract per-harness results for tracking
        harness_elos = {}
        for harness_type, elo_rating in result.harness_results.items():
            harness_name = harness_type.value if hasattr(harness_type, "value") else str(harness_type)
            harness_elos[harness_name] = {
                "elo": elo_rating.elo,
                "win_rate": elo_rating.win_rate,
                "games": elo_rating.games_played,
            }

            # Emit per-harness EVALUATION_COMPLETED event for Elo tracking
            await self._emit_harness_evaluation_completed(
                model_path=model_path,
                board_type=board_type,
                num_players=num_players,
                harness_type=harness_name,
                elo=elo_rating.elo,
                win_rate=elo_rating.win_rate,
            )

        return {
            "best_harness": result.best_harness.value if result.best_harness else None,
            "best_elo": result.best_elo,
            "total_games": result.total_games,
            "evaluation_time_seconds": result.evaluation_time_seconds,
            "harness_results": harness_elos,
        }

    async def _run_nnue_baseline_evaluation(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
        recording_config: Any = None,
    ) -> dict[str, Any] | None:
        """Run evaluation against NNUE-based baselines (Dec 31, 2025).

        Tests the model against NNUE baselines to compare NN vs NNUE performance:
        - 2-player: Uses NNUE_MINIMAX_D4
        - 3-4 player: Uses NNUE_BRS_D3 and NNUE_MAXN_D3

        Returns:
            Results dict with win rates per NNUE baseline, or None if NNUE unavailable
        """
        from pathlib import Path

        # Check if NNUE model exists for this config
        nnue_model_path = Path(f"models/nnue/nnue_{board_type}_{num_players}p.pt")
        if not nnue_model_path.exists():
            # Also check alternative naming convention
            nnue_model_path = Path(f"models/nnue_{board_type}_{num_players}p.pt")
            if not nnue_model_path.exists():
                logger.debug(f"No NNUE model found for {board_type}_{num_players}p")
                return None

        try:
            from app.training.game_gauntlet import BaselineOpponent, run_baseline_gauntlet

            # Select appropriate NNUE baselines based on player count
            if num_players == 2:
                nnue_opponents = [BaselineOpponent.NNUE_MINIMAX_D4]
            else:
                nnue_opponents = [
                    BaselineOpponent.NNUE_BRS_D3,
                    BaselineOpponent.NNUE_MAXN_D3,
                ]

            # Run gauntlet against NNUE baselines
            gauntlet_results = await asyncio.wait_for(
                asyncio.to_thread(
                    run_baseline_gauntlet,
                    model_path=model_path,
                    board_type=board_type,
                    num_players=num_players,
                    games_per_opponent=self.config.games_per_baseline,
                    opponents=nnue_opponents,
                    recording_config=recording_config,
                ),
                timeout=self.config.evaluation_timeout_seconds / 2,  # Use half timeout
            )

            # Extract win rates per NNUE opponent
            results = {
                "nnue_model": str(nnue_model_path),
                "win_rates": {},
            }
            for opponent, stats in gauntlet_results.opponent_results.items():
                results["win_rates"][opponent] = stats.get("win_rate", 0.0)

            return results

        except asyncio.TimeoutError:
            logger.warning(f"NNUE baseline evaluation timed out for {board_type}_{num_players}p")
            return None
        except ImportError as e:
            logger.debug(f"NNUE baseline dependencies not available: {e}")
            return None
        except Exception as e:
            logger.warning(f"NNUE baseline evaluation error: {e}")
            return None

    async def _emit_harness_evaluation_completed(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
        harness_type: str,
        elo: float,
        win_rate: float,
    ) -> None:
        """Emit HARNESS_EVALUATION_COMPLETED event for per-harness Elo tracking.

        December 31, 2025: Enables tracking Elo per (model, harness) combination.
        """
        try:
            from app.distributed.data_events import DataEventType, emit_data_event

            config_key = make_config_key(board_type, num_players)
            await emit_data_event(
                DataEventType.HARNESS_EVALUATION_COMPLETED,
                payload={
                    "config_key": config_key,
                    "model_path": model_path,
                    "harness_type": harness_type,
                    "elo": elo,
                    "win_rate": win_rate,
                },
                source="tournament_daemon",
            )
        except ImportError:
            pass  # Event system not available
        except Exception as e:
            logger.debug(f"Failed to emit HARNESS_EVALUATION_COMPLETED: {e}")

    async def _run_ladder_tournament(self) -> dict[str, Any]:
        """Run a ladder tournament across all configurations.

        Returns:
            Tournament results dict
        """
        self._stats.evaluations_completed += 1  # Fixed: use base class field, not property
        start_time = time.time()

        results = {
            "tournament_id": str(uuid.uuid4()),
            "success": False,
            "configs_evaluated": 0,
        }

        try:
            # Find all canonical models
            from app.models.discovery import find_canonical_models

            models = find_canonical_models()

            # find_canonical_models() returns {(board_type, num_players): Path}
            for (board_type, num_players), model_path in models.items():
                # Queue evaluation
                self._evaluation_queue.put_nowait({
                    "model_path": str(model_path),
                    "board_type": board_type,
                    "num_players": num_players,
                    "trigger": "periodic_ladder",
                })

                results["configs_evaluated"] += 1

            results["success"] = True
            self._stats.last_tournament_time = time.time()

        except ImportError:
            logger.warning("Model discovery not available, skipping ladder tournament")
        except Exception as e:
            logger.error(f"Ladder tournament failed: {e}")
            results["error"] = str(e)
            self._stats.errors.append(str(e))

        results["duration_seconds"] = time.time() - start_time
        return results

    async def _run_calibration_tournament(self) -> dict[str, Any]:
        """Run calibration tournament to validate Elo ladder.

        Tests expected win rates between baseline opponents:
        - Heuristic vs Random: ~95%+ win rate (validates 800 Elo gap)
        - Heuristic_Strong vs Heuristic: ~60% win rate (validates 200 Elo gap)
        - MCTS_Light vs Heuristic_Strong: ~55% win rate (validates 100 Elo gap)

        Returns:
            Tournament results with calibration validation status
        """
        logger.info("Running calibration tournament to validate Elo ladder")
        start_time = time.time()

        results = {
            "tournament_id": str(uuid.uuid4()),
            "tournament_type": "calibration",
            "success": False,
            "matchups": {},
        }

        try:
            from app.training.game_gauntlet import (
                create_baseline_ai,
                play_single_game,
                BaselineOpponent,
            )
            from app.models import BoardType

            # Define calibration matchups: (stronger, weaker, expected_win_rate)
            # Dec 31, 2025: Added NNUE calibration pairs for architecture diversity
            calibration_pairs = [
                (BaselineOpponent.HEURISTIC, BaselineOpponent.RANDOM, 0.90),
                (BaselineOpponent.HEURISTIC_STRONG, BaselineOpponent.HEURISTIC, 0.55),
                (BaselineOpponent.MCTS_LIGHT, BaselineOpponent.HEURISTIC_STRONG, 0.55),
                # NNUE calibration (NNUE_MINIMAX should beat HEURISTIC_STRONG)
                (BaselineOpponent.NNUE_MINIMAX_D4, BaselineOpponent.HEURISTIC_STRONG, 0.55),
            ]

            # Use square8_2p for calibration (fast games)
            board_type = BoardType.SQUARE8
            num_players = 2
            games_per_matchup = self.config.calibration_games

            # Create recording config if enabled (Dec 2025 - tournament games for training)
            recording_config = None
            if self.config.enable_game_recording:
                try:
                    from app.db.unified_recording import RecordingConfig, RecordSource
                    recording_config = RecordingConfig(
                        board_type=board_type.value,
                        num_players=num_players,
                        source=RecordSource.TOURNAMENT,
                        engine_mode="calibration",
                        db_prefix=self.config.recording_db_prefix,
                        db_dir=self.config.recording_db_dir,
                        store_history_entries=True,
                    )
                except ImportError:
                    logger.debug("Recording module not available for calibration tournament")

            for stronger, weaker, expected_rate in calibration_pairs:
                matchup_key = f"{stronger.value}_vs_{weaker.value}"
                wins = 0

                for game_num in range(games_per_matchup):
                    try:
                        # Alternate which player is "stronger" for position fairness
                        # Dec 29: Fixed argument order (baseline, player, board_type, num_players)
                        if game_num % 2 == 0:
                            player_0_ai = create_baseline_ai(
                                stronger, 1, board_type, num_players=num_players
                            )
                            player_1_ai = create_baseline_ai(
                                weaker, 2, board_type, num_players=num_players
                            )
                            stronger_player = 0
                        else:
                            player_0_ai = create_baseline_ai(
                                weaker, 1, board_type, num_players=num_players
                            )
                            player_1_ai = create_baseline_ai(
                                stronger, 2, board_type, num_players=num_players
                            )

                        game_result = play_single_game(
                            board_type=board_type,
                            num_players=num_players,
                            player_ais=[player_0_ai, player_1_ai],
                            timeout=self.config.game_timeout_seconds,
                            recording_config=recording_config,
                        )

                        if game_result.get("winner") == stronger_player:
                            wins += 1

                        self._stats.games_played += 1

                    except Exception as e:
                        logger.warning(f"Calibration game failed: {e}")

                actual_rate = wins / games_per_matchup if games_per_matchup > 0 else 0
                # Allow 10% margin below expected rate
                calibration_valid = actual_rate >= expected_rate * 0.9

                results["matchups"][matchup_key] = {
                    "wins": wins,
                    "games": games_per_matchup,
                    "win_rate": actual_rate,
                    "expected_rate": expected_rate,
                    "calibration_valid": calibration_valid,
                }

                if not calibration_valid:
                    logger.warning(
                        f"Calibration FAILED: {matchup_key} win rate {actual_rate:.1%} "
                        f"below expected {expected_rate:.1%}"
                    )

            results["success"] = True
            results["all_valid"] = all(
                m["calibration_valid"] for m in results["matchups"].values()
            )

        except ImportError as e:
            logger.warning(f"Calibration tournament dependencies not available: {e}")
            results["error"] = str(e)
        except Exception as e:
            logger.error(f"Calibration tournament failed: {e}")
            results["error"] = str(e)
            self._stats.errors.append(str(e))

        results["duration_seconds"] = time.time() - start_time
        logger.info(f"Calibration tournament completed: {results.get('all_valid', False)}")
        return results

    async def _run_cross_nn_tournament(self) -> dict[str, Any]:
        """Run cross-NN version tournament to compare model generations (Dec 2025).

        Discovers all model versions for each configuration and runs tournaments
        between adjacent versions (e.g., v2 vs v3, v3 vs v4) to:
        - Validate newer models are stronger than older ones
        - Maintain accurate Elo ratings across model generations
        - Identify potential regressions in model quality

        Returns:
            Tournament results with per-pairing win rates and Elo updates
        """
        logger.info("Running cross-NN version tournament")
        start_time = time.time()

        results = {
            "tournament_id": str(uuid.uuid4()),
            "tournament_type": "cross_nn",
            "success": False,
            "pairings": {},
            "games_played": 0,
        }

        try:
            from app.training.game_gauntlet import play_single_game
            from app.models import BoardType
            from app.ai.neural_net import UnifiedNeuralNetFactory
            from app.training.elo_service import get_elo_service
            from pathlib import Path
            import re

            elo_service = get_elo_service()
            models_dir = Path("models")

            # Find all canonical models per config
            # Pattern: canonical_{board}_{n}p.pth or canonical_{board}_{n}p_v{version}.pth
            model_pattern = re.compile(
                r"canonical_(?P<board>\w+)_(?P<players>\d)p(?:_v(?P<version>\d+))?\.pth"
            )

            # Group models by config (board_type, num_players)
            config_models: dict[tuple[str, int], list[tuple[str, Path]]] = {}

            for model_path in models_dir.glob("canonical_*.pth"):
                match = model_pattern.match(model_path.name)
                if match:
                    board = match.group("board")
                    players = int(match.group("players"))
                    version = match.group("version") or "base"
                    config_key = (board, players)

                    if config_key not in config_models:
                        config_models[config_key] = []
                    config_models[config_key].append((version, model_path))

            # Also check for versioned models like hex8_2p_v2.pth, hex8_2p_v3.pth
            version_pattern = re.compile(
                r"(?:canonical_)?(?P<board>\w+)_(?P<players>\d)p_v(?P<version>\d+)\.pth"
            )

            for model_path in models_dir.glob("*_v*.pth"):
                if "canonical" in model_path.name:
                    continue  # Already captured above
                match = version_pattern.match(model_path.name)
                if match:
                    board = match.group("board")
                    players = int(match.group("players"))
                    version = f"v{match.group('version')}"
                    config_key = (board, players)

                    if config_key not in config_models:
                        config_models[config_key] = []
                    config_models[config_key].append((version, model_path))

            # Dec 31, 2025: Also discover named architecture variants (v5heavy, v5-heavy-large)
            # Pattern: canonical_{board}_{n}p_{variant}.pth
            variant_pattern = re.compile(
                r"canonical_(?P<board>\w+)_(?P<players>\d)p_(?P<variant>v5heavy|v5-heavy|v5-heavy-large|v4|nnue)\.pth"
            )

            for model_path in models_dir.glob("canonical_*_*.pth"):
                match = variant_pattern.match(model_path.name)
                if match:
                    board = match.group("board")
                    players = int(match.group("players"))
                    variant = match.group("variant")
                    config_key = (board, players)

                    if config_key not in config_models:
                        config_models[config_key] = []
                    # Use variant name as version identifier
                    config_models[config_key].append((variant, model_path))

            games_per_pairing = self.config.cross_nn_games_per_pairing
            total_games = 0

            for (board, num_players), models in config_models.items():
                if len(models) < 2:
                    continue  # Need at least 2 versions to compare

                # Sort by version (base < v2 < v3 < ... < v4 < v5heavy < nnue)
                # Dec 31, 2025: Extended to handle named architecture variants
                def version_key(item: tuple[str, Path]) -> tuple[int, str]:
                    v = item[0]
                    # Known architectures in order of complexity/recency
                    version_order = {
                        "base": (0, ""),
                        "v2": (2, ""),
                        "v3": (3, ""),
                        "v4": (4, ""),
                        "v5heavy": (5, "heavy"),
                        "v5-heavy": (5, "heavy"),
                        "v5-heavy-large": (5, "heavy-large"),
                        "nnue": (6, ""),  # NNUE is evaluated separately
                    }
                    if v in version_order:
                        return version_order[v]
                    # Handle numeric versions like "v5", "v10"
                    if v.startswith("v") and v[1:].isdigit():
                        return (int(v[1:]), "")
                    return (100, v)  # Unknown versions sort last

                models.sort(key=version_key)

                # Get board type enum
                try:
                    board_type = BoardType(board)
                except ValueError:
                    logger.warning(f"Unknown board type: {board}")
                    continue

                # Create recording config if enabled (Dec 2025 - tournament games for training)
                recording_config = None
                if self.config.enable_game_recording:
                    try:
                        from app.db.unified_recording import RecordingConfig, RecordSource
                        recording_config = RecordingConfig(
                            board_type=board_type.value,
                            num_players=num_players,
                            source=RecordSource.TOURNAMENT,
                            engine_mode="cross_nn",
                            db_prefix=self.config.recording_db_prefix,
                            db_dir=self.config.recording_db_dir,
                            store_history_entries=True,
                        )
                    except ImportError:
                        logger.debug("Recording module not available for cross-NN tournament")

                # Run tournaments between adjacent versions
                for i in range(len(models) - 1):
                    older_version, older_path = models[i]
                    newer_version, newer_path = models[i + 1]

                    pairing_key = f"{board}_{num_players}p:{older_version}_vs_{newer_version}"
                    logger.info(f"Cross-NN pairing: {pairing_key}")

                    # Load models
                    try:
                        older_ai = UnifiedNeuralNetFactory.create(
                            str(older_path),
                            board_type=board_type,
                            num_players=num_players,
                        )
                        newer_ai = UnifiedNeuralNetFactory.create(
                            str(newer_path),
                            board_type=board_type,
                            num_players=num_players,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to load models for {pairing_key}: {e}")
                        results["pairings"][pairing_key] = {"error": str(e)}
                        continue

                    wins_newer = 0
                    wins_older = 0

                    for game_num in range(games_per_pairing):
                        try:
                            # Alternate positions for fairness
                            if game_num % 2 == 0:
                                player_ais = [newer_ai, older_ai]
                                newer_player = 0
                            else:
                                player_ais = [older_ai, newer_ai]
                                newer_player = 1

                            game_result = play_single_game(
                                board_type=board_type,
                                num_players=num_players,
                                player_ais=player_ais,
                                timeout=self.config.game_timeout_seconds,
                                recording_config=recording_config,
                            )

                            winner = game_result.get("winner")
                            if winner == newer_player:
                                wins_newer += 1
                            elif winner is not None:
                                wins_older += 1

                            total_games += 1
                            self._stats.games_played += 1

                            # Record match for Elo update
                            if winner is not None:
                                winner_id = newer_path.stem if winner == newer_player else older_path.stem
                                loser_id = older_path.stem if winner == newer_player else newer_path.stem
                                elo_service.record_match(
                                    winner_id=winner_id,
                                    loser_id=loser_id,
                                    board_type=board,
                                    num_players=num_players,
                                )

                        except Exception as e:
                            logger.warning(f"Cross-NN game failed: {e}")

                    win_rate_newer = wins_newer / games_per_pairing if games_per_pairing > 0 else 0
                    # Newer model should win >50% if it's actually better
                    improvement_validated = win_rate_newer >= 0.5

                    results["pairings"][pairing_key] = {
                        "newer_wins": wins_newer,
                        "older_wins": wins_older,
                        "draws": games_per_pairing - wins_newer - wins_older,
                        "games": games_per_pairing,
                        "newer_win_rate": win_rate_newer,
                        "improvement_validated": improvement_validated,
                    }

                    if not improvement_validated:
                        logger.warning(
                            f"Potential regression: {newer_version} only {win_rate_newer:.1%} "
                            f"vs {older_version} in {board}_{num_players}p"
                        )

            results["success"] = True
            results["games_played"] = total_games
            results["configs_tested"] = len([k for k, v in config_models.items() if len(v) >= 2])

        except ImportError as e:
            logger.warning(f"Cross-NN tournament dependencies not available: {e}")
            results["error"] = str(e)
        except Exception as e:
            logger.error(f"Cross-NN tournament failed: {e}")
            results["error"] = str(e)
            self._stats.errors.append(str(e))

        results["duration_seconds"] = time.time() - start_time
        logger.info(f"Cross-NN tournament completed: {results.get('games_played', 0)} games")
        return results

    async def _update_elo(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
        gauntlet_results: Any,
    ) -> None:
        """Update ELO ratings based on gauntlet results.

        NOTE (December 2025): Match recording is now done inline in
        game_gauntlet._evaluate_single_opponent(). This method now only
        ensures model registration - individual matches are NOT re-recorded
        to avoid double-counting Elo changes.
        """
        try:
            from app.training.elo_service import get_elo_service

            elo_service = get_elo_service()
            model_id = Path(model_path).stem

            # Register model if not already registered
            elo_service.register_model(
                model_id=model_id,
                board_type=board_type,
                num_players=num_players,
                model_path=model_path,
            )

            # December 2025: Match recording moved to game_gauntlet.py
            # Games are now recorded inline during gauntlet evaluation
            # to ensure ALL configs (including 3p/4p) are tracked.
            # Previously, only games through tournament_daemon were recorded.
            if hasattr(gauntlet_results, "opponent_results"):
                total_games = sum(
                    int(stats.get("games", 0))
                    for stats in gauntlet_results.opponent_results.values()
                )
                logger.info(
                    f"ELO for {model_id}: {total_games} games already recorded inline "
                    f"(estimated_elo={gauntlet_results.estimated_elo:.0f})"
                )
                return

            # Legacy path for older gauntlet result formats (now deprecated)
            match_results = gauntlet_results.get("matches", [])
            for match in match_results:
                opponent_id = match.get("opponent")
                winner = match.get("winner")

                if not opponent_id:
                    continue

                winner_id = model_id if winner == 0 else (opponent_id if winner == 1 else None)

                elo_service.record_match(
                    participant_a=model_id,
                    participant_b=opponent_id,
                    winner=winner_id,
                    board_type=board_type,
                    num_players=num_players,
                    game_length=match.get("game_length", 0),
                    duration_sec=match.get("duration", 0.0),
                )

            logger.info(f"Updated ELO for {model_id} with {len(match_results)} matches")

        except ImportError:
            logger.warning("EloService not available for ELO updates")
        except Exception as e:
            logger.error(f"Failed to update ELO: {e}")

    async def _emit_evaluation_completed(self, results: dict[str, Any]) -> None:
        """Emit EVALUATION_COMPLETED event."""
        try:
            from app.coordination.event_router import publish, DataEventType

            await publish(
                event_type=DataEventType.EVALUATION_COMPLETED,
                payload={
                    "model_path": results.get("model_path"),
                    "board_type": results.get("board_type"),
                    "num_players": results.get("num_players"),
                    "success": results.get("success", False),
                    "win_rates": results.get("win_rates", {}),
                    "elo": results.get("elo"),
                    "games_played": results.get("games_played", 0),
                },
                source="tournament_daemon",
            )

        except ImportError:
            pass  # Event router not available
        except Exception as e:
            logger.debug(f"Failed to emit EVALUATION_COMPLETED: {e}")

    async def evaluate_model(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
    ) -> dict[str, Any]:
        """Manually trigger evaluation for a model.

        Args:
            model_path: Path to model checkpoint
            board_type: Board type
            num_players: Number of players

        Returns:
            Evaluation results
        """
        return await self._run_evaluation(
            model_path=model_path,
            board_type=board_type,
            num_players=num_players,
            trigger="manual",
        )

    def get_status(self) -> dict[str, Any]:
        """Get current daemon status.

        Returns:
            Status dict with stats and configuration
        """
        # Combine HandlerBase stats with daemon-specific stats
        base_status = {
            "node_id": self.node_id,
            "running": self._running,
            "queue_size": self._evaluation_queue.qsize(),
            "cycle_count": self._cycle_count,
            "error_count": len(self._recent_errors),
            "stats": {
                "tournaments_completed": self._stats.tournaments_completed,
                "games_played": self._stats.games_played,
                "evaluations_triggered": self._stats.evaluations_triggered,
                "event_triggers": self._stats.event_triggers,
                "last_tournament_time": self._stats.last_tournament_time,
                "last_evaluation_time": self._stats.last_evaluation_time,
                "recent_errors": self._stats.errors[-5:],
            },
            "config": {
                "trigger_on_training_completed": self.config.trigger_on_training_completed,
                "enable_periodic_ladder": self.config.enable_periodic_ladder,
                "ladder_interval_seconds": self.config.ladder_interval_seconds,
                "games_per_baseline": self.config.games_per_baseline,
                "baselines": self.config.baselines,
            },
        }
        return base_status

    def health_check(self) -> HealthCheckResult:
        """Check daemon health (January 2026: HandlerBase pattern).

        Returns:
            HealthCheckResult with status and details
        """
        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="Tournament daemon not running",
            )

        # Check for high error rate (use HandlerBase error tracking)
        error_count = len(self._recent_errors)
        if error_count > 10:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"Tournament daemon has {error_count} recent errors",
                details=self.get_status(),
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"Tournament daemon running ({self._stats.games_played} games played)",
            details=self.get_status(),
        )


# Module-level singleton functions (delegate to HandlerBase pattern)


def get_tournament_daemon(
    config: TournamentDaemonConfig | None = None,
) -> TournamentDaemon:
    """Get the singleton TournamentDaemon instance.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        TournamentDaemon instance

    January 2026: Now delegates to HandlerBase.get_instance() singleton pattern.
    """
    return TournamentDaemon.get_instance(config)


def reset_tournament_daemon() -> None:
    """Reset the singleton (for testing).

    January 2026: Now delegates to HandlerBase.reset_instance().
    """
    TournamentDaemon.reset_instance()
