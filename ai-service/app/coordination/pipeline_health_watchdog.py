"""Pipeline Health Watchdog — detects stalled training pipeline stages.

Subscribes to key pipeline events (TRAINING_COMPLETED, EVALUATION_COMPLETED,
MODEL_PROMOTED) and tracks per-config timestamps. Alerts when any config
hasn't completed a full cycle in 24 hours, or when evaluations pile up
without promotions.

Mar 24, 2026: Created to detect "pipeline cycling garbage" — the state where
selfplay/training runs continuously but produces no quality improvement. The
Feb-Mar 2026 regression ran 171 iterations without detection.
"""
import logging
import time
from typing import Optional

from app.coordination.handler_base import HandlerBase

logger = logging.getLogger(__name__)

STALL_THRESHOLD_HOURS = 24
PROMOTION_DROUGHT_EVALS = 10  # 10+ evals without promotion = alert
EVALUATION_BOTTLENECK_TRAINS = 3  # 3+ trains without eval = alert


class PipelineHealthWatchdog(HandlerBase):
    """Monitors pipeline health by tracking stage completion timestamps."""

    _event_source = "PipelineHealthWatchdog"

    def __init__(self, cycle_interval: float = 1800.0):  # 30 min
        super().__init__(name="pipeline_health_watchdog", cycle_interval=cycle_interval)
        # Per-config timestamps: {config_key: {stage: timestamp}}
        self._stage_times: dict[str, dict[str, float]] = {}
        # Counters for drought detection
        self._eval_count_since_promotion: dict[str, int] = {}
        self._train_count_since_eval: dict[str, int] = {}

    def _get_event_subscriptions(self) -> dict:
        return {
            "TRAINING_COMPLETED": self._on_training_completed,
            "EVALUATION_COMPLETED": self._on_evaluation_completed,
            "MODEL_PROMOTED": self._on_model_promoted,
            "NEW_GAMES_AVAILABLE": self._on_new_games,
        }

    async def _on_training_completed(self, event: dict) -> None:
        config_key = event.get("config_key", "unknown")
        self._record_stage(config_key, "training")
        self._train_count_since_eval.setdefault(config_key, 0)
        self._train_count_since_eval[config_key] += 1

    async def _on_evaluation_completed(self, event: dict) -> None:
        config_key = event.get("config_key", "unknown")
        self._record_stage(config_key, "evaluation")
        self._eval_count_since_promotion.setdefault(config_key, 0)
        self._eval_count_since_promotion[config_key] += 1
        self._train_count_since_eval[config_key] = 0

    async def _on_model_promoted(self, event: dict) -> None:
        config_key = event.get("config_key", "unknown")
        self._record_stage(config_key, "promotion")
        self._eval_count_since_promotion[config_key] = 0

    async def _on_new_games(self, event: dict) -> None:
        config_key = event.get("config_key", "unknown")
        self._record_stage(config_key, "selfplay")

    def _record_stage(self, config_key: str, stage: str) -> None:
        if config_key not in self._stage_times:
            self._stage_times[config_key] = {}
        self._stage_times[config_key][stage] = time.time()

    async def _run_cycle(self) -> None:
        """Check for pipeline stalls and bottlenecks."""
        now = time.time()
        stall_threshold = STALL_THRESHOLD_HOURS * 3600
        alerts = []

        for config_key, stages in self._stage_times.items():
            # Check for full pipeline stall
            last_activity = max(stages.values()) if stages else 0
            if last_activity > 0 and (now - last_activity) > stall_threshold:
                hours = (now - last_activity) / 3600
                alerts.append(f"STALL: {config_key} no activity for {hours:.0f}h")
                await self._safe_emit_event_async(
                    "PIPELINE_STALLED",
                    {"config_key": config_key, "hours_since_activity": hours},
                )

            # Check for evaluation bottleneck
            trains = self._train_count_since_eval.get(config_key, 0)
            if trains >= EVALUATION_BOTTLENECK_TRAINS:
                alerts.append(
                    f"EVAL_BOTTLENECK: {config_key} {trains} trains without evaluation"
                )
                await self._safe_emit_event_async(
                    "EVALUATION_BOTTLENECK",
                    {"config_key": config_key, "trains_without_eval": trains},
                )

            # Check for promotion drought
            evals = self._eval_count_since_promotion.get(config_key, 0)
            if evals >= PROMOTION_DROUGHT_EVALS:
                alerts.append(
                    f"PROMOTION_DROUGHT: {config_key} {evals} evals without promotion"
                )
                await self._safe_emit_event_async(
                    "PROMOTION_DROUGHT",
                    {"config_key": config_key, "evals_without_promotion": evals},
                )

        if alerts:
            logger.warning(
                f"[PipelineHealth] {len(alerts)} alerts:\n  " + "\n  ".join(alerts)
            )
        else:
            active = len(self._stage_times)
            if active > 0:
                logger.info(f"[PipelineHealth] {active} configs active, no stalls")
            else:
                logger.info("[PipelineHealth] No pipeline activity recorded yet")

    def get_status(self) -> dict:
        """Get pipeline health summary for HTTP endpoint."""
        now = time.time()
        status = {}
        for config_key, stages in self._stage_times.items():
            status[config_key] = {
                stage: {
                    "last": ts,
                    "age_hours": (now - ts) / 3600,
                }
                for stage, ts in stages.items()
            }
            status[config_key]["trains_since_eval"] = (
                self._train_count_since_eval.get(config_key, 0)
            )
            status[config_key]["evals_since_promotion"] = (
                self._eval_count_since_promotion.get(config_key, 0)
            )
        return status
