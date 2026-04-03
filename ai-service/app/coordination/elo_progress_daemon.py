"""Elo Progress Daemon - Snapshots, velocity monitoring, and alerts.

Periodically snapshots best model Elo for each config, computes velocity
trends, and emits alerts on stalls/regressions.

Usage:
    # As standalone daemon
    python -m app.coordination.elo_progress_daemon

    # Via DaemonManager
    from app.coordination.daemon_manager import get_daemon_manager
    from app.coordination.daemon_types import DaemonType
    dm = get_daemon_manager()
    await dm.start(DaemonType.ELO_PROGRESS)

December 31, 2025: Created for training loop effectiveness tracking.
February 2026 (Sprint 18): Added velocity monitoring and alert system.

Alert levels:
- WARNING: Velocity below threshold for >6 hours
- ALERT: Elo regression detected (negative delta over 24h window)
- CRITICAL: Multiple configs regressing simultaneously

Subscribes to: EVALUATION_COMPLETED, MODEL_PROMOTED
Emits: ELO_VELOCITY_ALERT, ELO_REGRESSION_ALERT, ELO_PROGRESS_SUMMARY
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from app.coordination.contracts import CoordinatorStatus, HealthCheckResult
from app.coordination.event_emission_helpers import safe_emit_event
from app.coordination.event_utils import extract_config_key, normalize_event_payload
from app.coordination.handler_base import HandlerBase

logger = logging.getLogger(__name__)

# Snapshot settings
DEFAULT_SNAPSHOT_INTERVAL = 900.0   # 15 minutes
MIN_SNAPSHOT_INTERVAL = 300.0       # 5 minutes

# Alert thresholds
MIN_VELOCITY_ELO_PER_DAY = 0.5     # Below this = stall warning
REGRESSION_THRESHOLD = -2.0         # Elo drop over 24h = regression alert
MULTI_REGRESSION_COUNT = 3          # N configs regressing = critical
STALL_WINDOW_HOURS = 6.0            # Hours of low velocity before warning
LOOKBACK_DAYS_SHORT = 1.0           # Short-term velocity window
LOOKBACK_DAYS_MEDIUM = 7.0          # Medium-term trend window
ALERT_COOLDOWN_SECONDS = 3600.0     # 1 hour between repeated alerts per config
SUMMARY_INTERVAL_SECONDS = 21600.0  # 6 hours between progress summaries

ALL_CONFIGS = [
    "hex8_2p", "hex8_3p", "hex8_4p",
    "square8_2p", "square8_3p", "square8_4p",
    "square19_2p", "square19_3p", "square19_4p",
    "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
]


@dataclass
class EloProgressDaemonConfig:
    """Configuration for EloProgressDaemon."""

    snapshot_interval: float = DEFAULT_SNAPSHOT_INTERVAL
    enabled: bool = True


@dataclass
class ConfigVelocity:
    """Tracked velocity state for a single config."""

    config_key: str
    last_elo: float = 0.0
    last_snapshot_time: float = 0.0
    velocity_per_day: float = 0.0
    velocity_7d_per_day: float = 0.0
    stall_since: float | None = None
    last_alert_time: float = 0.0
    alert_level: str = "none"  # none, warning, alert, critical
    consecutive_regressions: int = 0


class EloProgressDaemon(HandlerBase):
    """Snapshots Elo progress and emits velocity alerts.

    Combines periodic Elo snapshotting (original) with velocity monitoring
    and alert emission (Sprint 18 addition).
    """

    _instance: EloProgressDaemon | None = None

    def __init__(self, config: EloProgressDaemonConfig | None = None):
        resolved_config = config or EloProgressDaemonConfig()
        super().__init__(
            name="elo_progress_daemon",
            config=resolved_config,
            cycle_interval=resolved_config.snapshot_interval,
        )
        self._last_snapshot_time: float = 0.0
        self._snapshot_count: int = 0
        self._last_error: str | None = None
        self._velocities: dict[str, ConfigVelocity] = {
            cfg: ConfigVelocity(config_key=cfg) for cfg in ALL_CONFIGS
        }
        self._last_summary_time: float = 0.0

    @classmethod
    def get_instance(cls) -> EloProgressDaemon:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """Subscribe to evaluation events for immediate snapshots."""
        return {
            "EVALUATION_COMPLETED": self._on_evaluation_completed,
            "MODEL_PROMOTED": self._on_model_promoted,
        }

    # =========================================================================
    # Main cycle: snapshot + velocity check + alerts
    # =========================================================================

    async def _run_cycle(self) -> None:
        """Take Elo snapshots and check velocity alerts."""
        if not self.config.enabled:
            return

        now = time.time()
        if now - self._last_snapshot_time < MIN_SNAPSHOT_INTERVAL:
            return

        # Phase 1: Take snapshots
        await self._take_snapshots()

        # Phase 2: Update velocity calculations from tracker
        self._update_velocities_from_tracker()

        # Phase 3: Check alert thresholds
        self._check_alerts()

        # Phase 4: Emit periodic summary
        self._maybe_emit_summary()

    async def _take_snapshots(self) -> None:
        """Take Elo snapshots for all configs."""
        try:
            from app.coordination.elo_progress_tracker import snapshot_all_configs

            results = await snapshot_all_configs()
            success_count = sum(1 for v in results.values() if v is not None)
            self._last_snapshot_time = time.time()
            self._snapshot_count += 1
            self._last_error = None

            logger.info(
                f"[EloProgress] Snapshot: {success_count}/{len(results)} configs "
                f"(total: {self._snapshot_count})"
            )
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"[EloProgress] Snapshot failed: {e}")

    # =========================================================================
    # Event handlers
    # =========================================================================

    async def _on_evaluation_completed(self, event: dict[str, Any]) -> None:
        """Snapshot on evaluation completion + update velocity."""
        payload = normalize_event_payload(event)
        config_key = extract_config_key(payload)
        if not config_key:
            return

        # Update real-time velocity from event data
        elo = payload.get("elo")
        if elo is not None and config_key in self._velocities:
            state = self._velocities[config_key]
            now = time.time()
            if state.last_elo > 0 and state.last_snapshot_time > 0:
                elapsed_days = (now - state.last_snapshot_time) / 86400
                if elapsed_days > 0.001:
                    state.velocity_per_day = (elo - state.last_elo) / elapsed_days
            state.last_elo = elo
            state.last_snapshot_time = now

        # Trigger snapshot if enough time passed
        if time.time() - self._last_snapshot_time >= MIN_SNAPSHOT_INTERVAL:
            logger.info(f"[EloProgress] Evaluation for {config_key}, triggering snapshot")
            await self._take_snapshots()

    async def _on_model_promoted(self, event: dict[str, Any]) -> None:
        """Snapshot on promotion + reset regression tracking."""
        payload = normalize_event_payload(event)
        config_key = extract_config_key(payload)
        if not config_key:
            return

        # Reset regression tracking — promotion means progress
        if config_key in self._velocities:
            state = self._velocities[config_key]
            state.consecutive_regressions = 0
            if state.alert_level in ("alert", "critical"):
                state.alert_level = "none"
                state.stall_since = None

        if time.time() - self._last_snapshot_time >= MIN_SNAPSHOT_INTERVAL:
            logger.info(f"[EloProgress] Model promoted for {config_key}, triggering snapshot")
            await self._take_snapshots()

    # =========================================================================
    # Velocity monitoring
    # =========================================================================

    def _update_velocities_from_tracker(self) -> None:
        """Query EloProgressTracker for velocity calculations."""
        try:
            from app.coordination.elo_progress_tracker import get_elo_progress_tracker

            tracker = get_elo_progress_tracker()

            for config_key in ALL_CONFIGS:
                state = self._velocities[config_key]

                # Short-term velocity (24h)
                report_short = tracker.get_progress_report(config_key, days=LOOKBACK_DAYS_SHORT)
                if report_short.num_snapshots >= 2 and report_short.improvement_rate_per_day is not None:
                    state.velocity_per_day = report_short.improvement_rate_per_day

                # Medium-term velocity (7d)
                report_medium = tracker.get_progress_report(config_key, days=LOOKBACK_DAYS_MEDIUM)
                if report_medium.num_snapshots >= 2 and report_medium.improvement_rate_per_day is not None:
                    state.velocity_7d_per_day = report_medium.improvement_rate_per_day

                # Update latest Elo
                latest = tracker.get_latest_snapshot(config_key)
                if latest:
                    state.last_elo = latest.best_elo
                    state.last_snapshot_time = latest.timestamp

        except ImportError:
            logger.debug("[EloProgress] EloProgressTracker not available")
        except Exception as e:
            logger.warning(f"[EloProgress] Error updating velocities: {e}")

    # =========================================================================
    # Alert system
    # =========================================================================

    def _check_alerts(self) -> None:
        """Check all configs for alert conditions."""
        now = time.time()
        regressing_configs = []

        for config_key, state in self._velocities.items():
            if state.last_snapshot_time == 0:
                continue

            # Regression: negative velocity exceeding threshold
            if state.velocity_per_day < REGRESSION_THRESHOLD:
                state.consecutive_regressions += 1
                regressing_configs.append(config_key)

                if self._can_alert(state, now):
                    state.alert_level = "alert"
                    state.last_alert_time = now
                    self._emit_regression_alert(config_key, state)

            # Stall: low positive velocity for extended period
            elif state.velocity_per_day < MIN_VELOCITY_ELO_PER_DAY:
                if state.stall_since is None:
                    state.stall_since = now
                stall_hours = (now - state.stall_since) / 3600

                if stall_hours >= STALL_WINDOW_HOURS and self._can_alert(state, now):
                    state.alert_level = "warning"
                    state.last_alert_time = now
                    self._emit_stall_warning(config_key, state, stall_hours)

            else:
                # Making progress — clear alerts
                if state.stall_since is not None:
                    logger.info(
                        f"[EloProgress] {config_key} recovered: "
                        f"velocity={state.velocity_per_day:.1f} Elo/day"
                    )
                state.stall_since = None
                state.consecutive_regressions = 0
                state.alert_level = "none"

        # Multi-config regression = critical
        if len(regressing_configs) >= MULTI_REGRESSION_COUNT:
            self._emit_critical_alert(regressing_configs)

    def _can_alert(self, state: ConfigVelocity, now: float) -> bool:
        """Check cooldown since last alert for this config."""
        return (now - state.last_alert_time) >= ALERT_COOLDOWN_SECONDS

    def _emit_regression_alert(self, config_key: str, state: ConfigVelocity) -> None:
        """Emit alert for Elo regression."""
        logger.warning(
            f"[EloProgress] REGRESSION: {config_key} "
            f"velocity={state.velocity_per_day:.1f} Elo/day, "
            f"elo={state.last_elo:.0f}, regressions={state.consecutive_regressions}"
        )
        safe_emit_event(
            "elo_regression_alert",
            {
                "config_key": config_key,
                "velocity_per_day": state.velocity_per_day,
                "velocity_7d_per_day": state.velocity_7d_per_day,
                "current_elo": state.last_elo,
                "consecutive_regressions": state.consecutive_regressions,
                "severity": "alert",
                "source": "EloProgressDaemon",
            },
            context="EloProgressDaemon",
        )

    def _emit_stall_warning(
        self, config_key: str, state: ConfigVelocity, stall_hours: float
    ) -> None:
        """Emit warning for Elo velocity stall."""
        logger.warning(
            f"[EloProgress] STALL: {config_key} "
            f"velocity={state.velocity_per_day:.1f} Elo/day for {stall_hours:.1f}h, "
            f"elo={state.last_elo:.0f}"
        )
        safe_emit_event(
            "elo_velocity_alert",
            {
                "config_key": config_key,
                "velocity_per_day": state.velocity_per_day,
                "stall_hours": stall_hours,
                "current_elo": state.last_elo,
                "severity": "warning",
                "source": "EloProgressDaemon",
            },
            context="EloProgressDaemon",
        )

    def _emit_critical_alert(self, regressing_configs: list[str]) -> None:
        """Emit critical alert for multi-config regression."""
        logger.error(
            f"[EloProgress] CRITICAL: {len(regressing_configs)} configs regressing: "
            f"{', '.join(regressing_configs)}"
        )
        safe_emit_event(
            "elo_regression_alert",
            {
                "regressing_configs": regressing_configs,
                "count": len(regressing_configs),
                "severity": "critical",
                "source": "EloProgressDaemon",
            },
            context="EloProgressDaemon",
        )

    def _maybe_emit_summary(self) -> None:
        """Emit periodic progress summary event."""
        now = time.time()
        if (now - self._last_summary_time) < SUMMARY_INTERVAL_SECONDS:
            return
        self._last_summary_time = now

        improving, stalled, regressing = [], [], []
        for config_key, state in self._velocities.items():
            if state.last_snapshot_time == 0:
                continue
            if state.velocity_per_day > MIN_VELOCITY_ELO_PER_DAY:
                improving.append(config_key)
            elif state.velocity_per_day < REGRESSION_THRESHOLD:
                regressing.append(config_key)
            else:
                stalled.append(config_key)

        logger.info(
            f"[EloProgress] Summary: {len(improving)} improving, "
            f"{len(stalled)} stalled, {len(regressing)} regressing"
        )
        safe_emit_event(
            "elo_progress_summary",
            {
                "improving": improving,
                "stalled": stalled,
                "regressing": regressing,
                "velocities": {
                    cfg: {
                        "velocity_1d": s.velocity_per_day,
                        "velocity_7d": s.velocity_7d_per_day,
                        "current_elo": s.last_elo,
                    }
                    for cfg, s in self._velocities.items()
                    if s.last_snapshot_time > 0
                },
                "source": "EloProgressDaemon",
            },
            context="EloProgressDaemon",
        )

    # =========================================================================
    # Health check & status
    # =========================================================================

    def health_check(self) -> HealthCheckResult:
        """Return health status with velocity info."""
        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="EloProgressDaemon not running",
            )

        regressing = [
            cfg for cfg, s in self._velocities.items()
            if s.alert_level in ("alert", "critical")
        ]
        stalled = [
            cfg for cfg, s in self._velocities.items()
            if s.alert_level == "warning"
        ]
        tracked = sum(1 for s in self._velocities.values() if s.last_snapshot_time > 0)

        if self._last_error:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"Last error: {self._last_error}",
                details={"snapshot_count": self._snapshot_count},
            )

        msg = f"Tracking {tracked}/12 configs"
        if regressing:
            msg += f", {len(regressing)} regressing"
        if stalled:
            msg += f", {len(stalled)} stalled"

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=msg,
            details={
                "tracked_configs": tracked,
                "snapshot_count": self._snapshot_count,
                "regressing": regressing,
                "stalled": stalled,
                "cycles": self._stats.cycles_completed,
            },
        )

    def get_velocity_summary(self) -> dict[str, dict[str, Any]]:
        """Return velocity data for all tracked configs (for API/dashboard)."""
        return {
            cfg: {
                "velocity_1d": s.velocity_per_day,
                "velocity_7d": s.velocity_7d_per_day,
                "current_elo": s.last_elo,
                "alert_level": s.alert_level,
                "stall_hours": (
                    (time.time() - s.stall_since) / 3600
                    if s.stall_since else 0.0
                ),
            }
            for cfg, s in self._velocities.items()
            if s.last_snapshot_time > 0
        }


def get_elo_progress_daemon() -> EloProgressDaemon:
    """Get singleton EloProgressDaemon."""
    return EloProgressDaemon.get_instance()


async def create_elo_progress_daemon() -> EloProgressDaemon:
    """Factory function for DaemonManager."""
    return get_elo_progress_daemon()


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Elo Progress Daemon")
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_SNAPSHOT_INTERVAL,
        help=f"Snapshot interval in seconds (default: {DEFAULT_SNAPSHOT_INTERVAL})",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Take one snapshot and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    async def main():
        config = EloProgressDaemonConfig(snapshot_interval=args.interval)
        daemon = EloProgressDaemon(config)

        if args.once:
            await daemon._run_cycle()
        else:
            await daemon.start()
            try:
                while daemon._running:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                await daemon.stop()

    asyncio.run(main())
