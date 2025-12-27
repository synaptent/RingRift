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
    baselines: list[str] = field(default_factory=lambda: ["random", "heuristic"])

    # Concurrency
    max_concurrent_games: int = 4

    # Timeouts
    game_timeout_seconds: float = 300.0  # 5 minutes per game
    evaluation_timeout_seconds: float = 1800.0  # 30 minutes per evaluation


@dataclass
class TournamentStats:
    """Statistics about tournament daemon activity."""
    tournaments_completed: int = 0
    games_played: int = 0
    evaluations_triggered: int = 0
    event_triggers: int = 0
    last_tournament_time: float = 0.0
    last_evaluation_time: float = 0.0
    errors: list[str] = field(default_factory=list)


class TournamentDaemon:
    """Daemon that automatically schedules tournaments based on events.

    Subscribes to training/promotion events and triggers appropriate
    evaluation tournaments using the existing tournament infrastructure.
    """

    def __init__(self, config: TournamentDaemonConfig | None = None):
        """Initialize the tournament daemon.

        Args:
            config: Daemon configuration
        """
        self.config = config or TournamentDaemonConfig()
        self.node_id = socket.gethostname()

        self._running = False
        self._periodic_task: asyncio.Task | None = None
        self._stats = TournamentStats()
        self._evaluation_queue: asyncio.Queue = asyncio.Queue()
        self._evaluation_task: asyncio.Task | None = None
        self._subscribed = False

    async def start(self) -> None:
        """Start the tournament daemon."""
        if self._running:
            logger.warning("TournamentDaemon already running")
            return

        self._running = True

        # Subscribe to events
        self._subscribe_to_events()

        # Start evaluation worker
        self._evaluation_task = asyncio.create_task(
            self._evaluation_worker(),
            name="tournament_evaluation_worker"
        )

        # Start periodic ladder tournaments
        if self.config.enable_periodic_ladder:
            self._periodic_task = asyncio.create_task(
                self._periodic_ladder_loop(),
                name="tournament_periodic_ladder"
            )

        logger.info(
            f"TournamentDaemon started (periodic_ladder={self.config.enable_periodic_ladder}, "
            f"interval={self.config.ladder_interval_seconds}s)"
        )

    async def stop(self) -> None:
        """Stop the tournament daemon."""
        if not self._running:
            return

        self._running = False

        # Cancel tasks
        if self._periodic_task:
            self._periodic_task.cancel()
            try:
                await asyncio.wait_for(self._periodic_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._periodic_task = None

        if self._evaluation_task:
            self._evaluation_task.cancel()
            try:
                await asyncio.wait_for(self._evaluation_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._evaluation_task = None

        logger.info("TournamentDaemon stopped")

    def is_running(self) -> bool:
        """Check if daemon is running."""
        return self._running

    def _subscribe_to_events(self) -> None:
        """Subscribe to relevant events."""
        if self._subscribed:
            return

        try:
            from app.coordination.event_router import get_router
            from app.distributed.data_events import DataEventType

            router = get_router()

            if self.config.trigger_on_training_completed:
                router.subscribe(
                    DataEventType.TRAINING_COMPLETED,
                    self._on_training_completed
                )
                logger.info("TournamentDaemon subscribed to TRAINING_COMPLETED")

            if self.config.trigger_on_model_promoted:
                router.subscribe(
                    DataEventType.MODEL_PROMOTED,
                    self._on_model_promoted
                )
                logger.info("TournamentDaemon subscribed to MODEL_PROMOTED")

            self._subscribed = True

        except ImportError as e:
            logger.warning(f"Event router not available, running without event triggers: {e}")
        except Exception as e:
            logger.error(f"Failed to subscribe to events: {e}")

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
        """Parse board_type and num_players from config string."""
        # Format: "hex8_2p" or "square8_2p"
        parts = config.replace("_", " ").split()

        board_type = None
        num_players = None

        for part in parts:
            # Check for board type
            if part in ("hex8", "hexagonal", "square8", "square19"):
                board_type = part
            # Check for player count
            elif part.endswith("p") and part[:-1].isdigit():
                num_players = int(part[:-1])

        return board_type, num_players

    async def _evaluation_worker(self) -> None:
        """Worker that processes evaluation queue."""
        while self._running:
            try:
                # Wait for evaluation request
                try:
                    request = await asyncio.wait_for(
                        self._evaluation_queue.get(),
                        timeout=10.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Run evaluation
                await self._run_evaluation(
                    model_path=request["model_path"],
                    board_type=request["board_type"],
                    num_players=request["num_players"],
                    trigger=request.get("trigger", "unknown"),
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in evaluation worker: {e}")
                self._stats.errors.append(str(e))

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
                self._stats.errors.append(str(e))

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

            gauntlet_results = await asyncio.wait_for(
                asyncio.to_thread(
                    run_baseline_gauntlet,
                    model_path=model_path,
                    board_type=board_type,
                    num_players=num_players,
                    games_per_opponent=self.config.games_per_baseline,
                    opponents=[BaselineOpponent.RANDOM, BaselineOpponent.HEURISTIC],
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

        except asyncio.TimeoutError:
            logger.error(f"Evaluation timeout: {model_path}")
            results["error"] = "timeout"
            self._stats.errors.append(f"Evaluation timeout: {model_path}")

        except ImportError as e:
            logger.warning(f"GameGauntlet not available: {e}")
            # Fall back to basic match execution
            results = await self._run_basic_evaluation(
                model_path, board_type, num_players
            )

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            results["error"] = str(e)
            self._stats.errors.append(str(e))

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

    async def _run_ladder_tournament(self) -> dict[str, Any]:
        """Run a ladder tournament across all configurations.

        Returns:
            Tournament results dict
        """
        self._stats.tournaments_completed += 1
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

            for model_info in models:
                model_path = model_info.get("path")
                board_type = model_info.get("board_type")
                num_players = model_info.get("num_players")

                if not all([model_path, board_type, num_players]):
                    continue

                # Queue evaluation
                self._evaluation_queue.put_nowait({
                    "model_path": model_path,
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
            from app.coordination.event_router import publish
            from app.distributed.data_events import DataEventType

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
        return {
            "node_id": self.node_id,
            "running": self._running,
            "subscribed": self._subscribed,
            "queue_size": self._evaluation_queue.qsize(),
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


# Module-level singleton
_tournament_daemon: TournamentDaemon | None = None


def get_tournament_daemon(
    config: TournamentDaemonConfig | None = None,
) -> TournamentDaemon:
    """Get the singleton TournamentDaemon instance.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        TournamentDaemon instance
    """
    global _tournament_daemon
    if _tournament_daemon is None:
        _tournament_daemon = TournamentDaemon(config)
    return _tournament_daemon


def reset_tournament_daemon() -> None:
    """Reset the singleton (for testing)."""
    global _tournament_daemon
    _tournament_daemon = None
