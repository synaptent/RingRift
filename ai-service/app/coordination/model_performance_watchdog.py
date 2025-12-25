"""Model Performance Watchdog - Monitors model win rates and detects degradation.

Subscribes to EVALUATION_COMPLETED events and tracks rolling win rates per model.
Emits alerts when a model's performance degrades below acceptable thresholds.

Usage:
    from app.coordination.model_performance_watchdog import ModelPerformanceWatchdog

    watchdog = ModelPerformanceWatchdog()
    await watchdog.start()

Events Subscribed:
    - EVALUATION_COMPLETED: Triggered when gauntlet evaluation finishes

Events Emitted:
    - REGRESSION_DETECTED: Model performance dropped significantly
    - MODEL_PROMOTED: Candidate when model exceeds thresholds (info only)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformance:
    """Tracks performance metrics for a single model."""
    model_id: str
    board_type: str
    num_players: int

    # Win rates vs different opponents
    win_rate_vs_random: float = 0.0
    win_rate_vs_heuristic: float = 0.0
    win_rate_vs_previous: float = 0.0

    # Rolling history (last N evaluations)
    recent_vs_random: list[float] = field(default_factory=list)
    recent_vs_heuristic: list[float] = field(default_factory=list)

    # Timestamps
    first_seen: float = field(default_factory=time.time)
    last_evaluation: float = field(default_factory=time.time)
    evaluation_count: int = 0

    # Performance state
    is_degraded: bool = False
    degraded_since: float | None = None


@dataclass
class WatchdogConfig:
    """Configuration for ModelPerformanceWatchdog."""
    # Thresholds
    min_vs_random: float = 0.85  # Must beat RANDOM 85%+ of the time
    min_vs_heuristic: float = 0.60  # Must beat HEURISTIC 60%+ of the time
    degradation_threshold: float = 0.55  # Below this vs heuristic is degraded

    # Rolling window
    rolling_window_size: int = 5  # Number of recent evaluations to track

    # Alert cooldown
    alert_cooldown: float = 300.0  # Seconds between repeated alerts for same model


class ModelPerformanceWatchdog:
    """Watchdog that monitors model performance from evaluation events.

    Tracks win rates for each model and emits alerts when performance
    degrades below acceptable thresholds. This enables:
    - Early detection of regression
    - Automatic rollback triggers
    - Curriculum feedback based on model strength

    Attributes:
        config: Watchdog configuration
        models: Dict of model performance records
    """

    def __init__(self, config: WatchdogConfig | None = None):
        self.config = config or WatchdogConfig()
        self._running = False
        self._task: asyncio.Task | None = None

        # Model performance tracking
        self.models: dict[str, ModelPerformance] = {}
        self._last_alert_time: dict[str, float] = defaultdict(float)

        # Subscription handle
        self._unsubscribe: callable | None = None

    async def start(self) -> None:
        """Start the watchdog and subscribe to evaluation events."""
        if self._running:
            logger.warning("ModelPerformanceWatchdog already running")
            return

        self._running = True
        await self._subscribe_to_events()
        logger.info("ModelPerformanceWatchdog started")

    async def stop(self) -> None:
        """Stop the watchdog."""
        self._running = False
        if self._unsubscribe:
            try:
                self._unsubscribe()
            except Exception as e:
                logger.debug(f"Error unsubscribing: {e}")
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("ModelPerformanceWatchdog stopped")

    def is_running(self) -> bool:
        """Check if watchdog is running."""
        return self._running

    async def _subscribe_to_events(self) -> None:
        """Subscribe to EVALUATION_COMPLETED events."""
        try:
            from app.distributed.data_events import DataEventType
            from app.coordination.event_router import get_router

            router = get_router()
            self._unsubscribe = router.subscribe(
                DataEventType.EVALUATION_COMPLETED,
                self._on_evaluation_completed
            )
            logger.debug("Subscribed to EVALUATION_COMPLETED events")

        except ImportError as e:
            logger.warning(f"Event system not available: {e}")

    async def _on_evaluation_completed(self, event) -> None:
        """Handle EVALUATION_COMPLETED events."""
        try:
            payload = event.payload if hasattr(event, 'payload') else event

            model_id = payload.get("model_id", payload.get("model_path", "unknown"))
            board_type = payload.get("board_type", "unknown")
            num_players = payload.get("num_players", 2)

            # Extract win rates
            win_rate_vs_random = payload.get("win_rate_vs_random", 0.0)
            win_rate_vs_heuristic = payload.get("win_rate_vs_heuristic", 0.0)
            win_rate_vs_previous = payload.get("win_rate_vs_previous", 0.0)

            # Also check for result dict structure
            results = payload.get("results", {})
            if results:
                for opponent, data in results.items():
                    if "random" in opponent.lower():
                        win_rate_vs_random = data.get("win_rate", win_rate_vs_random)
                    elif "heuristic" in opponent.lower():
                        win_rate_vs_heuristic = data.get("win_rate", win_rate_vs_heuristic)

            await self._update_model_performance(
                model_id=model_id,
                board_type=board_type,
                num_players=num_players,
                win_rate_vs_random=win_rate_vs_random,
                win_rate_vs_heuristic=win_rate_vs_heuristic,
                win_rate_vs_previous=win_rate_vs_previous,
            )

        except Exception as e:
            logger.error(f"Error handling evaluation event: {e}")

    async def _update_model_performance(
        self,
        model_id: str,
        board_type: str,
        num_players: int,
        win_rate_vs_random: float,
        win_rate_vs_heuristic: float,
        win_rate_vs_previous: float = 0.0,
    ) -> None:
        """Update performance record for a model."""
        # Get or create performance record
        if model_id not in self.models:
            self.models[model_id] = ModelPerformance(
                model_id=model_id,
                board_type=board_type,
                num_players=num_players,
            )

        perf = self.models[model_id]
        perf.evaluation_count += 1
        perf.last_evaluation = time.time()

        # Update current win rates
        perf.win_rate_vs_random = win_rate_vs_random
        perf.win_rate_vs_heuristic = win_rate_vs_heuristic
        perf.win_rate_vs_previous = win_rate_vs_previous

        # Update rolling history
        perf.recent_vs_random.append(win_rate_vs_random)
        perf.recent_vs_heuristic.append(win_rate_vs_heuristic)

        # Trim to rolling window size
        if len(perf.recent_vs_random) > self.config.rolling_window_size:
            perf.recent_vs_random = perf.recent_vs_random[-self.config.rolling_window_size:]
        if len(perf.recent_vs_heuristic) > self.config.rolling_window_size:
            perf.recent_vs_heuristic = perf.recent_vs_heuristic[-self.config.rolling_window_size:]

        # Check for degradation
        was_degraded = perf.is_degraded
        is_degraded = win_rate_vs_heuristic < self.config.degradation_threshold

        if is_degraded and not was_degraded:
            perf.is_degraded = True
            perf.degraded_since = time.time()
            await self._emit_degradation_alert(perf)
        elif not is_degraded and was_degraded:
            perf.is_degraded = False
            perf.degraded_since = None
            logger.info(f"Model {model_id} recovered from degradation")

        # Log performance
        logger.debug(
            f"Model {model_id}: vs_random={win_rate_vs_random:.2%}, "
            f"vs_heuristic={win_rate_vs_heuristic:.2%}"
        )

    async def _emit_degradation_alert(self, perf: ModelPerformance) -> None:
        """Emit alert when model performance degrades."""
        now = time.time()
        last_alert = self._last_alert_time[perf.model_id]

        if now - last_alert < self.config.alert_cooldown:
            logger.debug(f"Skipping alert for {perf.model_id} due to cooldown")
            return

        try:
            from app.distributed.data_events import DataEventType
            from app.coordination.event_router import get_router

            router = get_router()

            payload = {
                "model_id": perf.model_id,
                "board_type": perf.board_type,
                "num_players": perf.num_players,
                "win_rate_vs_heuristic": perf.win_rate_vs_heuristic,
                "win_rate_vs_random": perf.win_rate_vs_random,
                "threshold": self.config.degradation_threshold,
                "severity": "moderate" if perf.win_rate_vs_heuristic >= 0.45 else "severe",
                "timestamp": now,
            }

            await router.publish(DataEventType.REGRESSION_DETECTED, payload)
            self._last_alert_time[perf.model_id] = now

            logger.warning(
                f"Model performance degraded: {perf.model_id} "
                f"(vs_heuristic={perf.win_rate_vs_heuristic:.2%} < {self.config.degradation_threshold:.2%})"
            )

        except ImportError as e:
            logger.warning(f"Event system not available: {e}")
        except Exception as e:
            logger.error(f"Error emitting degradation alert: {e}")

    def get_model_summary(self) -> dict[str, dict]:
        """Get summary of all tracked models."""
        return {
            model_id: {
                "board_type": perf.board_type,
                "num_players": perf.num_players,
                "win_rate_vs_random": perf.win_rate_vs_random,
                "win_rate_vs_heuristic": perf.win_rate_vs_heuristic,
                "is_degraded": perf.is_degraded,
                "evaluation_count": perf.evaluation_count,
                "rolling_avg_vs_heuristic": (
                    sum(perf.recent_vs_heuristic) / len(perf.recent_vs_heuristic)
                    if perf.recent_vs_heuristic else 0.0
                ),
            }
            for model_id, perf in self.models.items()
        }


# Singleton instance
_watchdog: ModelPerformanceWatchdog | None = None


def get_watchdog() -> ModelPerformanceWatchdog:
    """Get or create the singleton watchdog instance."""
    global _watchdog
    if _watchdog is None:
        _watchdog = ModelPerformanceWatchdog()
    return _watchdog


# Factory function for DaemonManager integration
async def create_model_performance_watchdog() -> None:
    """Factory function for DaemonManager.

    Creates and runs the model performance watchdog until cancelled.
    """
    watchdog = get_watchdog()
    await watchdog.start()

    try:
        # Keep running until cancelled
        while watchdog.is_running():
            await asyncio.sleep(60)
    except asyncio.CancelledError:
        await watchdog.stop()
        raise
