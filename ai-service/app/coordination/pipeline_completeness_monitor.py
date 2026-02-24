"""Pipeline Completeness Monitor Daemon.

Tracks the last completion timestamp for each pipeline stage per config
and emits PIPELINE_STAGE_OVERDUE events when stages exceed thresholds.

Pipeline stages tracked:
- selfplay_complete
- data_sync_completed
- npz_export_complete
- training_completed
- evaluation_completed
- model_promoted

Thresholds (hours):
- selfplay: 6h (small boards) / 12h (large boards: square19, hexagonal)
- npz_export: 8h
- training: 24h
- evaluation: 48h

February 2026: Created as part of pipeline observability improvements.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from app.coordination.contracts import CoordinatorStatus, HealthCheckResult
from app.coordination.handler_base import HandlerBase

logger = logging.getLogger(__name__)

# Large boards have longer selfplay thresholds
LARGE_BOARDS = frozenset({"square19", "hexagonal"})

# Pipeline stages in order
PIPELINE_STAGES = [
    "selfplay",
    "data_sync",
    "npz_export",
    "training",
    "evaluation",
    "promotion",
]

# Default thresholds in hours per stage
DEFAULT_THRESHOLDS_HOURS: dict[str, float] = {
    "selfplay": 6.0,
    "data_sync": 8.0,
    "npz_export": 8.0,
    "training": 24.0,
    "evaluation": 48.0,
    "promotion": 96.0,  # Promotion is rare, allow longer window
}

# Large board overrides
LARGE_BOARD_THRESHOLDS_HOURS: dict[str, float] = {
    "selfplay": 12.0,  # Large boards take longer for selfplay
}

# Event-to-stage mapping
EVENT_TO_STAGE: dict[str, str] = {
    "selfplay_complete": "selfplay",
    "data_sync_completed": "data_sync",
    "npz_export_complete": "npz_export",
    "training_completed": "training",
    "evaluation_completed": "evaluation",
    "model_promoted": "promotion",
}


class PipelineCompletenessMonitor(HandlerBase):
    """Monitors pipeline stage completion across all configs.

    Tracks the last completion timestamp for each pipeline stage per config
    and emits PIPELINE_STAGE_OVERDUE events when stages exceed thresholds.

    Health check returns RED if any config has 2+ overdue stages.
    """

    _event_source = "PipelineCompletenessMonitor"

    # Feb 23, 2026: Consecutive rejection threshold before emitting overdue alert.
    # When the same config is rejected by the same gate 5+ times in a row,
    # it indicates a systematic blocker that needs attention.
    CONSECUTIVE_REJECTION_ALERT_THRESHOLD = 5

    def __init__(self) -> None:
        super().__init__(
            name="pipeline_completeness_monitor",
            cycle_interval=1800.0,  # 30 minutes
        )
        # {config_key: {stage: last_timestamp}}
        self._stage_timestamps: dict[str, dict[str, float]] = {}
        # Track overdue counts for health check
        self._overdue_counts: dict[str, int] = 0  # type: ignore[assignment]
        self._overdue_counts = {}
        # Import ALL_CONFIGS lazily to avoid circular imports at module level
        self._all_configs: list[str] = []

        # Feb 23, 2026: Track consecutive promotion rejections per (config, gate)
        # Key: (config_key, gate) -> consecutive rejection count
        self._consecutive_rejections: dict[tuple[str, str], int] = {}
        # Track which alerts have already been emitted to avoid spam
        self._rejection_alerts_emitted: set[tuple[str, str]] = set()

    def _get_event_subscriptions(self) -> dict[str, Callable[[Any], Any]]:
        """Subscribe to pipeline stage completion events and promotion rejections."""
        return {
            "selfplay_complete": self._on_stage_complete,
            "data_sync_completed": self._on_stage_complete,
            "npz_export_complete": self._on_stage_complete,
            "training_completed": self._on_stage_complete,
            "evaluation_completed": self._on_stage_complete,
            "model_promoted": self._on_stage_complete,
            # Feb 23, 2026: Track promotion rejections for pipeline blocking detection
            "promotion_rejected": self._on_promotion_rejected,
        }

    async def _on_start(self) -> None:
        """Load ALL_CONFIGS on startup."""
        try:
            from app.coordination.priority_calculator import ALL_CONFIGS

            self._all_configs = list(ALL_CONFIGS)
        except ImportError:
            logger.warning(
                "[pipeline_completeness_monitor] Could not import ALL_CONFIGS, "
                "using empty config list"
            )
            self._all_configs = []

        # Initialize timestamps for all configs
        now = time.time()
        for config_key in self._all_configs:
            if config_key not in self._stage_timestamps:
                self._stage_timestamps[config_key] = {}
                # Seed all stages with current time to avoid false alarms on startup
                for stage in PIPELINE_STAGES:
                    self._stage_timestamps[config_key][stage] = now

        logger.info(
            f"[pipeline_completeness_monitor] Initialized tracking for "
            f"{len(self._all_configs)} configs"
        )

    async def _on_stage_complete(self, event: Any) -> None:
        """Handle a pipeline stage completion event."""
        payload = self._get_payload(event)

        # Determine event type
        event_type = ""
        if hasattr(event, "event_type"):
            event_type = event.event_type
        elif isinstance(event, dict):
            event_type = event.get("type", event.get("event_type", ""))

        stage = EVENT_TO_STAGE.get(event_type, "")
        if not stage:
            return

        config_key = self._extract_config_key(payload)
        if config_key == "unknown":
            return

        # Update timestamp
        now = time.time()
        if config_key not in self._stage_timestamps:
            self._stage_timestamps[config_key] = {}
        self._stage_timestamps[config_key][stage] = now

        # Feb 23, 2026: On successful promotion, reset rejection tracking
        # since the pipeline is no longer blocked for this config
        if stage == "promotion":
            self._reset_rejection_tracking(config_key)

        self._record_success()
        logger.debug(
            f"[pipeline_completeness_monitor] Recorded {stage} for {config_key}"
        )

    async def _on_promotion_rejected(self, event: Any) -> None:
        """Handle PROMOTION_REJECTED event to track consecutive rejections.

        Feb 23, 2026: When a model fails a promotion gate, track the rejection
        per (config_key, gate). If the same gate blocks the same config 5+
        consecutive times, emit a PIPELINE_STAGE_OVERDUE event identifying
        the specific blocking gate.

        A successful promotion (model_promoted event via _on_stage_complete)
        resets the rejection counter for that config.
        """
        payload = self._get_payload(event)
        config_key = payload.get("config_key", "")
        gate = payload.get("gate", "unknown")

        if not config_key:
            return

        key = (config_key, gate)
        self._consecutive_rejections[key] = self._consecutive_rejections.get(key, 0) + 1
        count = self._consecutive_rejections[key]

        logger.debug(
            f"[pipeline_completeness_monitor] Promotion rejected: "
            f"{config_key}/{gate} (consecutive={count})"
        )

        # Check if threshold reached and alert not yet emitted
        if count >= self.CONSECUTIVE_REJECTION_ALERT_THRESHOLD and key not in self._rejection_alerts_emitted:
            self._rejection_alerts_emitted.add(key)
            reason = payload.get("reason", "")
            await self._safe_emit_event_async(
                "PIPELINE_STAGE_OVERDUE",
                {
                    "config_key": config_key,
                    "stage": "promotion",
                    "blocking_gate": gate,
                    "consecutive_rejections": count,
                    "reason": f"Promotion blocked {count}x by {gate}: {reason}",
                    "hours_since": 0.0,  # Not time-based, rejection-count-based
                    "threshold": self.CONSECUTIVE_REJECTION_ALERT_THRESHOLD,
                },
            )
            logger.warning(
                f"[pipeline_completeness_monitor] {config_key} promotion blocked "
                f"{count}x by gate '{gate}': {reason}"
            )

        self._record_success()

    def _reset_rejection_tracking(self, config_key: str) -> None:
        """Reset consecutive rejection counters for a config after promotion.

        Feb 23, 2026: When a model is promoted, clear all rejection tracking
        for that config since the pipeline is no longer blocked.
        """
        keys_to_remove = [
            key for key in self._consecutive_rejections if key[0] == config_key
        ]
        for key in keys_to_remove:
            del self._consecutive_rejections[key]
            self._rejection_alerts_emitted.discard(key)

    async def _run_cycle(self) -> None:
        """Check each config x stage against thresholds."""
        if not self._all_configs:
            return

        now = time.time()
        overdue_counts: dict[str, int] = {}
        total_overdue = 0

        for config_key in self._all_configs:
            timestamps = self._stage_timestamps.get(config_key, {})
            config_overdue = 0

            # Determine if this is a large board config
            board_type = config_key.rsplit("_", 1)[0] if "_" in config_key else config_key
            is_large = board_type in LARGE_BOARDS

            for stage in PIPELINE_STAGES:
                last_ts = timestamps.get(stage)
                if last_ts is None:
                    # No record of this stage ever completing - skip
                    continue

                threshold_hours = self._get_threshold_hours(stage, is_large)
                hours_since = (now - last_ts) / 3600.0

                if hours_since > threshold_hours:
                    config_overdue += 1
                    total_overdue += 1

                    # Emit overdue event
                    await self._safe_emit_event_async(
                        "PIPELINE_STAGE_OVERDUE",
                        {
                            "config_key": config_key,
                            "stage": stage,
                            "hours_since": round(hours_since, 1),
                            "threshold": threshold_hours,
                        },
                    )
                    logger.warning(
                        f"[pipeline_completeness_monitor] {config_key}/{stage} overdue: "
                        f"{hours_since:.1f}h (threshold: {threshold_hours}h)"
                    )

            overdue_counts[config_key] = config_overdue

        self._overdue_counts = overdue_counts

        if total_overdue > 0:
            logger.info(
                f"[pipeline_completeness_monitor] {total_overdue} overdue stage(s) "
                f"across {sum(1 for c in overdue_counts.values() if c > 0)} config(s)"
            )

    def _get_threshold_hours(self, stage: str, is_large_board: bool) -> float:
        """Get the threshold in hours for a given stage and board size."""
        if is_large_board and stage in LARGE_BOARD_THRESHOLDS_HOURS:
            return LARGE_BOARD_THRESHOLDS_HOURS[stage]
        return DEFAULT_THRESHOLDS_HOURS.get(stage, 48.0)

    def health_check(self) -> HealthCheckResult:
        """Return RED if any config has 2+ overdue stages."""
        base = super().health_check()
        if not base.healthy:
            return base

        # Count configs with 2+ overdue stages
        configs_red = [
            config_key
            for config_key, count in self._overdue_counts.items()
            if count >= 2
        ]

        if configs_red:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=(
                    f"{len(configs_red)} config(s) have 2+ overdue pipeline stages: "
                    f"{', '.join(configs_red[:5])}"
                ),
                details={
                    **self._get_health_details(),
                    "configs_red": configs_red,
                    "overdue_counts": self._overdue_counts,
                },
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message="All pipeline stages within thresholds",
            details={
                **self._get_health_details(),
                "tracked_configs": len(self._all_configs),
                "overdue_counts": self._overdue_counts,
            },
        )
