"""EvaluationDaemon - Auto-evaluate models after training completes.

December 2025: Part of Phase 11 (Auto-Evaluation Pipeline).

This daemon subscribes to TRAINING_COMPLETE events and automatically triggers
gauntlet evaluation for newly trained models. This closes the training loop
by ensuring every trained model gets evaluated without manual intervention.

Key Features:
- Subscribes to TRAINING_COMPLETE events
- Runs baseline gauntlet evaluation against RANDOM and HEURISTIC
- Emits EVALUATION_COMPLETED events for promotion consideration
- Supports early stopping based on statistical confidence

Usage:
    from app.coordination.evaluation_daemon import (
        EvaluationDaemon,
        get_evaluation_daemon,
    )

    # Start the daemon
    daemon = get_evaluation_daemon()
    await daemon.start()

    # Or use via DaemonManager
    from app.coordination.daemon_manager import get_daemon_manager, DaemonType
    manager = get_daemon_manager()
    await manager.start(DaemonType.EVALUATION)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Singleton instance
_daemon: "EvaluationDaemon | None" = None


@dataclass
class EvaluationStats:
    """Statistics for the evaluation daemon."""

    evaluations_triggered: int = 0
    evaluations_completed: int = 0
    evaluations_failed: int = 0
    total_games_played: int = 0
    average_evaluation_time: float = 0.0
    last_evaluation_time: float = 0.0


@dataclass
class EvaluationConfig:
    """Configuration for the evaluation daemon."""

    # Games per baseline opponent
    games_per_baseline: int = 20

    # Baselines to evaluate against
    baselines: list[str] = field(default_factory=lambda: ["random", "heuristic"])

    # Early stopping configuration
    early_stopping_enabled: bool = True
    early_stopping_confidence: float = 0.95
    early_stopping_min_games: int = 10

    # Concurrency
    max_concurrent_evaluations: int = 1

    # Timeouts
    evaluation_timeout_seconds: float = 600.0  # 10 minutes


class EvaluationDaemon:
    """Daemon that auto-evaluates models after training completes."""

    def __init__(self, config: EvaluationConfig | None = None):
        self.config = config or EvaluationConfig()
        self.stats = EvaluationStats()
        self._running = False
        self._subscribed = False
        self._evaluation_queue: asyncio.Queue = asyncio.Queue()
        self._active_evaluations: set[str] = set()

    async def start(self) -> None:
        """Start the evaluation daemon."""
        if self._running:
            logger.warning("[EvaluationDaemon] Already running")
            return

        self._running = True
        self._subscribe_to_events()

        # Start the evaluation worker
        asyncio.create_task(self._evaluation_worker())

        logger.info(
            f"[EvaluationDaemon] Started. "
            f"Games per baseline: {self.config.games_per_baseline}, "
            f"Early stopping: {self.config.early_stopping_enabled}"
        )

    async def stop(self) -> None:
        """Stop the evaluation daemon."""
        if not self._running:
            return

        self._running = False
        self._unsubscribe_from_events()

        logger.info(
            f"[EvaluationDaemon] Stopped. "
            f"Evaluations: {self.stats.evaluations_completed}/{self.stats.evaluations_triggered}"
        )

    def is_running(self) -> bool:
        """Check if daemon is running."""
        return self._running

    def _subscribe_to_events(self) -> None:
        """Subscribe to training completion events."""
        if self._subscribed:
            return

        try:
            from app.coordination.event_router import get_event_bus
            from app.distributed.data_events import DataEventType

            bus = get_event_bus()
            if bus is None:
                logger.warning("[EvaluationDaemon] Event bus not available")
                return

            bus.subscribe(DataEventType.TRAINING_COMPLETED, self._on_training_complete)
            self._subscribed = True

            logger.debug("[EvaluationDaemon] Subscribed to TRAINING_COMPLETED events")

        except ImportError as e:
            logger.warning(f"[EvaluationDaemon] Event system not available: {e}")
        except Exception as e:
            logger.error(f"[EvaluationDaemon] Failed to subscribe: {e}")

    def _unsubscribe_from_events(self) -> None:
        """Unsubscribe from events."""
        if not self._subscribed:
            return

        try:
            from app.coordination.event_router import get_event_bus
            from app.distributed.data_events import DataEventType

            bus = get_event_bus()
            if bus:
                bus.unsubscribe(DataEventType.TRAINING_COMPLETED, self._on_training_complete)
            self._subscribed = False

        except Exception as e:
            logger.debug(f"[EvaluationDaemon] Error unsubscribing: {e}")

    async def _on_training_complete(self, event: Any) -> None:
        """Handle TRAINING_COMPLETE event."""
        try:
            # Extract metadata from event
            if hasattr(event, "payload"):
                metadata = event.payload
            elif hasattr(event, "metadata"):
                metadata = event.metadata
            else:
                metadata = event if isinstance(event, dict) else {}

            # train.py emits "checkpoint_path", but also support "model_path" for backwards compatibility
            model_path = (
                metadata.get("checkpoint_path")
                or metadata.get("model_path")
                or metadata.get("model_id")
            )
            board_type = metadata.get("board_type")
            num_players = metadata.get("num_players", 2)

            if not model_path:
                logger.warning("[EvaluationDaemon] No checkpoint_path/model_path in TRAINING_COMPLETE event")
                return

            # Queue the evaluation
            await self._evaluation_queue.put({
                "model_path": model_path,
                "board_type": board_type,
                "num_players": num_players,
                "timestamp": time.time(),
            })

            self.stats.evaluations_triggered += 1
            logger.info(
                f"[EvaluationDaemon] Queued evaluation for {model_path} "
                f"({board_type}_{num_players}p)"
            )

        except Exception as e:
            logger.error(f"[EvaluationDaemon] Error handling training complete: {e}")

    async def _evaluation_worker(self) -> None:
        """Worker that processes evaluation requests from the queue."""
        while self._running:
            try:
                # Wait for an evaluation request
                request = await asyncio.wait_for(
                    self._evaluation_queue.get(),
                    timeout=5.0,  # Check running status periodically
                )

                # Skip if already evaluating this model
                model_path = request["model_path"]
                if model_path in self._active_evaluations:
                    logger.debug(f"[EvaluationDaemon] Skipping duplicate: {model_path}")
                    continue

                # Check concurrency limit
                if len(self._active_evaluations) >= self.config.max_concurrent_evaluations:
                    # Re-queue and wait
                    await self._evaluation_queue.put(request)
                    await asyncio.sleep(1.0)
                    continue

                # Run evaluation
                self._active_evaluations.add(model_path)
                try:
                    await self._run_evaluation(request)
                finally:
                    self._active_evaluations.discard(model_path)

            except asyncio.TimeoutError:
                continue  # Normal - check running status
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[EvaluationDaemon] Worker error: {e}")
                await asyncio.sleep(1.0)

    async def _run_evaluation(self, request: dict) -> None:
        """Run gauntlet evaluation for a model."""
        model_path = request["model_path"]
        board_type = request["board_type"]
        num_players = request["num_players"]

        start_time = time.time()
        logger.info(f"[EvaluationDaemon] Starting evaluation: {model_path}")

        try:
            # Run the gauntlet
            result = await self._run_gauntlet(
                model_path=model_path,
                board_type=board_type,
                num_players=num_players,
            )

            elapsed = time.time() - start_time
            self.stats.evaluations_completed += 1
            self.stats.last_evaluation_time = elapsed
            self._update_average_time(elapsed)

            # Count games played
            total_games = sum(
                opp.get("games_played", 0)
                for opp in result.get("opponent_results", {}).values()
            )
            self.stats.total_games_played += total_games

            # Emit evaluation completed event
            await self._emit_evaluation_completed(
                model_path=model_path,
                board_type=board_type,
                num_players=num_players,
                result=result,
            )

            logger.info(
                f"[EvaluationDaemon] Evaluation completed: {model_path} "
                f"(win_rate={result.get('overall_win_rate', 0):.1%}, "
                f"{total_games} games, {elapsed:.1f}s)"
            )

        except asyncio.TimeoutError:
            self.stats.evaluations_failed += 1
            logger.error(f"[EvaluationDaemon] Evaluation timed out: {model_path}")
        except Exception as e:
            self.stats.evaluations_failed += 1
            logger.error(f"[EvaluationDaemon] Evaluation failed: {model_path}: {e}")

    async def _run_gauntlet(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
    ) -> dict:
        """Run baseline gauntlet with optional early stopping."""
        from app.training.game_gauntlet import run_baseline_gauntlet, BaselineOpponent

        # Map baseline names to enum values
        baseline_map = {
            "random": BaselineOpponent.RANDOM,
            "heuristic": BaselineOpponent.HEURISTIC,
        }
        opponents = [
            baseline_map[b]
            for b in self.config.baselines
            if b in baseline_map
        ]

        # Run with timeout and early stopping
        result = await asyncio.wait_for(
            asyncio.to_thread(
                run_baseline_gauntlet,
                model_path=model_path,
                board_type=board_type,
                opponents=opponents,
                games_per_opponent=self.config.games_per_baseline,
                num_players=num_players,
                verbose=False,
                early_stopping=self.config.early_stopping_enabled,
                early_stopping_confidence=self.config.early_stopping_confidence,
                early_stopping_min_games=self.config.early_stopping_min_games,
            ),
            timeout=self.config.evaluation_timeout_seconds,
        )

        # Convert to dict if needed
        if hasattr(result, "opponent_results"):
            return {
                "overall_win_rate": result.win_rate,
                "opponent_results": result.opponent_results,
                "early_stopped_baselines": getattr(result, "early_stopped_baselines", []),
                "games_saved_by_early_stopping": getattr(result, "games_saved_by_early_stopping", 0),
            }
        elif isinstance(result, dict):
            return result
        else:
            return {"overall_win_rate": 0.0, "opponent_results": {}}

    async def _emit_evaluation_completed(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
        result: dict,
    ) -> None:
        """Emit EVALUATION_COMPLETED event."""
        try:
            from app.coordination.event_emitters import emit_evaluation_completed

            await emit_evaluation_completed(
                model_path=model_path,
                board_type=board_type,
                num_players=num_players,
                win_rate=result.get("overall_win_rate", 0.0),
                opponent_results=result.get("opponent_results", {}),
                games_played=sum(
                    opp.get("games_played", 0)
                    for opp in result.get("opponent_results", {}).values()
                ),
            )
        except ImportError:
            logger.debug("[EvaluationDaemon] Event emitters not available")
        except Exception as e:
            logger.debug(f"[EvaluationDaemon] Failed to emit event: {e}")

    def _update_average_time(self, elapsed: float) -> None:
        """Update running average of evaluation time."""
        n = self.stats.evaluations_completed
        if n == 1:
            self.stats.average_evaluation_time = elapsed
        else:
            # Exponential moving average
            alpha = 0.2
            self.stats.average_evaluation_time = (
                alpha * elapsed +
                (1 - alpha) * self.stats.average_evaluation_time
            )

    def get_stats(self) -> dict:
        """Get daemon statistics."""
        return {
            "running": self._running,
            "evaluations_triggered": self.stats.evaluations_triggered,
            "evaluations_completed": self.stats.evaluations_completed,
            "evaluations_failed": self.stats.evaluations_failed,
            "evaluations_pending": self._evaluation_queue.qsize(),
            "active_evaluations": len(self._active_evaluations),
            "total_games_played": self.stats.total_games_played,
            "average_evaluation_time": round(self.stats.average_evaluation_time, 1),
        }


def get_evaluation_daemon(config: EvaluationConfig | None = None) -> EvaluationDaemon:
    """Get or create the singleton evaluation daemon."""
    global _daemon
    if _daemon is None:
        _daemon = EvaluationDaemon(config)
    return _daemon


async def start_evaluation_daemon(config: EvaluationConfig | None = None) -> EvaluationDaemon:
    """Start the evaluation daemon (convenience function)."""
    daemon = get_evaluation_daemon(config)
    await daemon.start()
    return daemon
