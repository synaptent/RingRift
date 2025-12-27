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

__all__ = [
    "EvaluationStats",
    "EvaluationConfig",
    "EvaluationDaemon",
    "get_evaluation_daemon",
    "start_evaluation_daemon",
]

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

    # Deduplication settings (December 2025)
    dedup_cooldown_seconds: float = 300.0  # 5 minute cooldown per model
    dedup_max_tracked_models: int = 1000  # Max models to track for dedup


class EvaluationDaemon:
    """Daemon that auto-evaluates models after training completes."""

    def __init__(self, config: EvaluationConfig | None = None):
        self.config = config or EvaluationConfig()
        self.stats = EvaluationStats()
        self._running = False
        self._subscribed = False
        self._evaluation_queue: asyncio.Queue = asyncio.Queue()
        self._active_evaluations: set[str] = set()

        # Deduplication tracking (December 2025)
        # Track recently evaluated models: model_path -> last_evaluation_timestamp
        self._recently_evaluated: dict[str, float] = {}
        # Track seen event content hashes to prevent duplicate triggers
        self._seen_event_hashes: set[str] = set()
        # Stats for deduplication
        self._dedup_stats = {
            "cooldown_skips": 0,
            "content_hash_skips": 0,
            "concurrent_skips": 0,
        }

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
            from app.coordination.event_router import get_event_bus, DataEventType

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
            from app.coordination.event_router import get_event_bus, DataEventType

            bus = get_event_bus()
            if bus:
                bus.unsubscribe(DataEventType.TRAINING_COMPLETED, self._on_training_complete)
            self._subscribed = False

        except Exception as e:
            logger.debug(f"[EvaluationDaemon] Error unsubscribing: {e}")

    def _compute_event_hash(self, model_path: str, board_type: str, num_players: int) -> str:
        """Compute a content hash for deduplication.

        December 2025: Prevents duplicate evaluations from multiple event sources.
        """
        import hashlib
        content = f"{model_path}:{board_type}:{num_players}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _is_duplicate_event(self, event_hash: str) -> bool:
        """Check if this event has been seen recently.

        December 2025: Content-based deduplication.
        """
        if event_hash in self._seen_event_hashes:
            return True

        # Add to seen set with LRU eviction
        self._seen_event_hashes.add(event_hash)
        if len(self._seen_event_hashes) > self.config.dedup_max_tracked_models:
            # Remove oldest (arbitrary in set, but prevents unbounded growth)
            self._seen_event_hashes.pop()

        return False

    def _is_in_cooldown(self, model_path: str) -> bool:
        """Check if model was recently evaluated (within cooldown period).

        December 2025: Time-based deduplication.
        """
        now = time.time()

        # Clean up old entries
        expired = [
            path for path, ts in self._recently_evaluated.items()
            if now - ts > self.config.dedup_cooldown_seconds
        ]
        for path in expired:
            del self._recently_evaluated[path]

        # Check if model is in cooldown
        last_eval = self._recently_evaluated.get(model_path)
        if last_eval and now - last_eval < self.config.dedup_cooldown_seconds:
            return True

        return False

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

            # December 2025: Deduplication checks
            # Check 1: Content hash deduplication (same event from multiple sources)
            event_hash = self._compute_event_hash(model_path, board_type, num_players)
            if self._is_duplicate_event(event_hash):
                self._dedup_stats["content_hash_skips"] += 1
                logger.debug(
                    f"[EvaluationDaemon] Skipping duplicate event (content hash): {model_path}"
                )
                return

            # Check 2: Cooldown period (recently evaluated model)
            if self._is_in_cooldown(model_path):
                self._dedup_stats["cooldown_skips"] += 1
                logger.debug(
                    f"[EvaluationDaemon] Skipping model in cooldown: {model_path}"
                )
                return

            # Check 3: Already being evaluated
            if model_path in self._active_evaluations:
                self._dedup_stats["concurrent_skips"] += 1
                logger.debug(
                    f"[EvaluationDaemon] Skipping already-evaluating model: {model_path}"
                )
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

        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"[EvaluationDaemon] Invalid event data: {e}")
        except (OSError, IOError) as e:
            logger.error(f"[EvaluationDaemon] I/O error handling training complete: {e}")

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

            # December 2025: Mark as recently evaluated for deduplication
            self._recently_evaluated[model_path] = time.time()

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
            # December 2025: Deduplication stats
            "dedup_cooldown_skips": self._dedup_stats["cooldown_skips"],
            "dedup_content_hash_skips": self._dedup_stats["content_hash_skips"],
            "dedup_concurrent_skips": self._dedup_stats["concurrent_skips"],
            "tracked_recently_evaluated": len(self._recently_evaluated),
        }

    def health_check(self):
        """Check daemon health (December 2025: CoordinatorProtocol compliance).

        Returns:
            HealthCheckResult with status and details
        """
        from app.coordination.protocols import HealthCheckResult, CoordinatorStatus

        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="Evaluation daemon not running",
            )

        # Check for high failure rate
        total = self.stats.evaluations_triggered
        failed = self.stats.evaluations_failed
        if total > 5 and failed / total > 0.5:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"High evaluation failure rate: {failed}/{total}",
                details=self.get_stats(),
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"Evaluation daemon running ({self.stats.evaluations_completed} completed)",
            details=self.get_stats(),
        )


def get_evaluation_daemon(config: EvaluationConfig | None = None) -> EvaluationDaemon:
    """Get or create the singleton evaluation daemon.

    Args:
        config: Optional configuration. Only used on first call.

    Returns:
        EvaluationDaemon: The singleton daemon instance.
    """
    global _daemon
    if _daemon is None:
        _daemon = EvaluationDaemon(config)
    return _daemon


async def start_evaluation_daemon(config: EvaluationConfig | None = None) -> EvaluationDaemon:
    """Start the evaluation daemon (convenience function).

    Combines get_evaluation_daemon() and start() in one call.

    Args:
        config: Optional configuration. Only used on first call.

    Returns:
        EvaluationDaemon: The started daemon instance.
    """
    daemon = get_evaluation_daemon(config)
    await daemon.start()
    return daemon
