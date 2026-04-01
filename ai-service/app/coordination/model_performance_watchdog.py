"""Model Performance Watchdog - Monitors model win rates and detects degradation.

Subscribes to EVALUATION_COMPLETED events and tracks rolling win rates per model.
Emits alerts when a model's performance degrades below acceptable thresholds.

Usage:
    from app.coordination.model_performance_watchdog import ModelPerformanceWatchdog

    watchdog = ModelPerformanceWatchdog.get_instance()
    await watchdog.start()
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from app.coordination.contracts import CoordinatorStatus, HealthCheckResult
from app.coordination.handler_base import HandlerBase

logger = logging.getLogger(__name__)

__all__ = [
    "ModelPerformance",
    "ModelPerformanceWatchdog",
    "get_watchdog",
    "create_model_performance_watchdog",
]


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


class ModelPerformanceWatchdog(HandlerBase):
    """Watchdog that monitors model performance from evaluation events.

    Tracks win rates for each model and emits alerts when performance
    degrades below acceptable thresholds.
    """

    _event_source = "ModelPerformanceWatchdog"

    def __init__(self) -> None:
        super().__init__(
            name="model_performance_watchdog",
            cycle_interval=300.0,  # Event-driven, long cycle for health checks
        )

        # Thresholds
        self._degradation_threshold = 0.55
        self._rolling_window_size = 5
        self._alert_cooldown = 300.0

        # Model performance tracking
        self.models: dict[str, ModelPerformance] = {}
        self._last_alert_time: dict[str, float] = defaultdict(float)

    def _get_event_subscriptions(self) -> dict[str, Callable]:
        """Define event subscriptions.

        Returns:
            Dict mapping event names to handlers.
        """
        return {
            "EVALUATION_COMPLETED": self._on_evaluation_completed,
        }

    async def _run_cycle(self) -> None:
        """Periodic cycle - minimal for event-driven monitor."""
        pass

    async def _on_evaluation_completed(self, event) -> None:
        """Handle EVALUATION_COMPLETED events."""
        try:
            # Check for duplicate events
            if self._is_duplicate_event(event, key_fields=["model_id", "board_type", "num_players"]):
                logger.debug("Skipping duplicate EVALUATION_COMPLETED event")
                return

            self._stats.events_processed += 1

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
            self._record_error(str(e))
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
        if len(perf.recent_vs_random) > self._rolling_window_size:
            perf.recent_vs_random = perf.recent_vs_random[-self._rolling_window_size:]
        if len(perf.recent_vs_heuristic) > self._rolling_window_size:
            perf.recent_vs_heuristic = perf.recent_vs_heuristic[-self._rolling_window_size:]

        # Check for degradation
        was_degraded = perf.is_degraded
        is_degraded = win_rate_vs_heuristic < self._degradation_threshold

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

        if now - last_alert < self._alert_cooldown:
            logger.debug(f"Skipping alert for {perf.model_id} due to cooldown")
            return

        try:
            from app.coordination.event_router import DataEventType, publish

            if DataEventType is None:
                logger.debug("DataEventType not available, skipping degradation alert")
                return

            payload = {
                "model_id": perf.model_id,
                "board_type": perf.board_type,
                "num_players": perf.num_players,
                "win_rate_vs_heuristic": perf.win_rate_vs_heuristic,
                "win_rate_vs_random": perf.win_rate_vs_random,
                "threshold": self._degradation_threshold,
                "severity": "moderate" if perf.win_rate_vs_heuristic >= 0.45 else "severe",
                "timestamp": now,
            }

            await publish(
                event_type=DataEventType.REGRESSION_DETECTED,
                payload=payload,
                source="ModelPerformanceWatchdog",
            )
            self._last_alert_time[perf.model_id] = now

            logger.warning(
                f"Model performance degraded: {perf.model_id} "
                f"(vs_heuristic={perf.win_rate_vs_heuristic:.2%} < {self._degradation_threshold:.2%})"
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

    def health_check(self) -> "HealthCheckResult":
        """Check daemon health with model-specific details.

        Overrides base health_check to add model performance tracking info.

        Returns:
            HealthCheckResult with health status and model tracking details.
        """
        # Get base health check
        base_result = super().health_check()

        # Add model-specific details
        degraded_count = sum(1 for p in self.models.values() if p.is_degraded)
        model_count = len(self.models)

        # Override healthy status if too many models degraded
        if model_count > 0 and degraded_count > model_count // 2:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"Too many degraded models: {degraded_count}/{model_count}",
                details={
                    **base_result.details,
                    "model_count": model_count,
                    "degraded_count": degraded_count,
                    "models": list(self.models.keys())[:10],  # First 10 for brevity
                },
            )

        # Add model tracking to base details
        base_result.details["model_count"] = model_count
        base_result.details["degraded_count"] = degraded_count
        base_result.details["models"] = list(self.models.keys())[:10]

        return base_result


def get_watchdog() -> ModelPerformanceWatchdog:
    """Get or create the singleton watchdog instance.

    Returns:
        ModelPerformanceWatchdog: The singleton watchdog instance.

    Example:
        watchdog = get_watchdog()
        await watchdog.start()
    """
    return ModelPerformanceWatchdog.get_instance()


async def create_model_performance_watchdog() -> None:
    """Factory function for DaemonManager integration.

    Creates and runs the model performance watchdog until cancelled.
    Used by DaemonManager.start(DaemonType.MODEL_PERFORMANCE_WATCHDOG).

    Raises:
        asyncio.CancelledError: When the daemon is stopped.
    """
    import asyncio

    watchdog = get_watchdog()
    await watchdog.start()

    try:
        # Keep running until cancelled
        while watchdog.is_running():
            await asyncio.sleep(60)
    except asyncio.CancelledError:
        await watchdog.stop()
        raise
