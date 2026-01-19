"""Progress Watchdog Daemon for 48-Hour Autonomous Operation.

This daemon monitors Elo velocity across all 12 configurations and detects
training stalls. When progress stalls (Elo velocity goes negative or near-zero),
it triggers recovery actions like boosting selfplay for the stalled config.

Key features:
- Tracks Elo velocity for all 12 canonical configs
- Detects stalls (6+ hours without positive velocity)
- Triggers recovery via PROGRESS_STALL_DETECTED event
- Configurable thresholds and recovery actions

December 2025: Created for 48-hour autonomous operation enablement.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.contracts import CoordinatorStatus
from app.coordination.event_emission_helpers import safe_emit_event

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


CANONICAL_CONFIGS = [
    "hex8_2p", "hex8_3p", "hex8_4p",
    "square8_2p", "square8_3p", "square8_4p",
    "square19_2p", "square19_3p", "square19_4p",
    "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
]


@dataclass
class ProgressWatchdogConfig:
    """Configuration for ProgressWatchdog daemon.

    December 2025: Simplified - no longer inherits from DaemonConfig.
    HandlerBase uses cycle_interval directly.

    January 2026: Added tiered escalation for enhanced stall recovery.
    Longer stalls get progressively more aggressive interventions.

    Attributes:
        check_interval_seconds: How often to check Elo velocity (default: 1 hour)
        min_elo_velocity: Minimum Elo gain per hour to consider progress (default: 0.5)
        stall_threshold_hours: Hours of stall before triggering recovery (default: 6)
        recovery_action: What to do on stall (default: boost_selfplay)
        boost_multiplier: Base selfplay boost multiplier (default: 2.0)
        max_recovery_attempts: Max recoveries per config per 24h (default: 4)

        # January 2026: Tiered escalation thresholds
        extended_stall_hours: Stall duration for 4x boost (default: 48)
        prolonged_stall_hours: Stall duration for 4x + curriculum reset (default: 96)
        severe_stall_hours: Stall duration for architecture exploration (default: 168)
        extended_boost_multiplier: Boost multiplier for extended stalls (default: 4.0)
    """

    check_interval_seconds: int = 3600  # 1 hour
    min_elo_velocity: float = 0.5  # Elo/hour minimum
    stall_threshold_hours: float = 6.0  # Hours without progress before alert
    recovery_action: str = "boost_selfplay"  # Recovery action type
    boost_multiplier: float = 2.0  # Selfplay boost multiplier
    max_recovery_attempts: int = 4  # Max recoveries per 24h per config

    # January 2026: Tiered escalation for enhanced stall recovery
    extended_stall_hours: float = 48.0  # 2 days: 4x boost
    prolonged_stall_hours: float = 96.0  # 4 days: 4x + curriculum reset
    severe_stall_hours: float = 168.0  # 7 days: architecture exploration
    extended_boost_multiplier: float = 4.0  # Boost for extended/prolonged stalls

    @classmethod
    def from_env(cls) -> "ProgressWatchdogConfig":
        """Load config from environment variables."""
        config = cls()

        if os.environ.get("RINGRIFT_PROGRESS_INTERVAL"):
            try:
                config.check_interval_seconds = int(
                    os.environ.get("RINGRIFT_PROGRESS_INTERVAL", "3600")
                )
            except ValueError:
                pass

        if os.environ.get("RINGRIFT_PROGRESS_MIN_VELOCITY"):
            try:
                config.min_elo_velocity = float(
                    os.environ.get("RINGRIFT_PROGRESS_MIN_VELOCITY", "0.5")
                )
            except ValueError:
                pass

        if os.environ.get("RINGRIFT_PROGRESS_STALL_HOURS"):
            try:
                config.stall_threshold_hours = float(
                    os.environ.get("RINGRIFT_PROGRESS_STALL_HOURS", "6")
                )
            except ValueError:
                pass

        return config


# =============================================================================
# Stall Tracking
# =============================================================================


@dataclass
class ConfigProgress:
    """Progress tracking for a single config."""

    config_key: str
    last_elo: float = 1500.0
    current_elo: float = 1500.0
    velocity: float = 0.0  # Elo per hour
    last_update_time: float = 0.0
    stall_start_time: float = 0.0  # When stall started (0 = not stalled)
    recovery_attempts: int = 0  # Recoveries in last 24h
    last_recovery_time: float = 0.0

    @property
    def is_stalled(self) -> bool:
        """Check if this config is currently stalled."""
        return self.stall_start_time > 0

    @property
    def stall_duration_hours(self) -> float:
        """How long has this config been stalled."""
        if not self.is_stalled:
            return 0.0
        return (time.time() - self.stall_start_time) / 3600

    def reset_recovery_counter_if_needed(self) -> None:
        """Reset recovery counter if 24h has passed."""
        if time.time() - self.last_recovery_time >= 86400:  # 24 hours
            self.recovery_attempts = 0


# =============================================================================
# Progress Watchdog Daemon
# =============================================================================


class ProgressWatchdogDaemon(HandlerBase):
    """Daemon that monitors Elo velocity and detects training stalls.

    December 2025: Migrated to HandlerBase pattern.
    - Uses HandlerBase singleton (get_instance/reset_instance)
    - Uses _stats for metrics tracking

    Workflow:
    1. Periodically fetch Elo velocity for all configs
    2. Track velocity trends and detect stalls
    3. Emit PROGRESS_STALL_DETECTED event for recovery

    Events Emitted:
    - PROGRESS_STALL_DETECTED: When a config has stalled
    - PROGRESS_RECOVERED: When a stalled config resumes progress
    """

    def __init__(self, config: ProgressWatchdogConfig | None = None):
        self._daemon_config = config or ProgressWatchdogConfig.from_env()

        super().__init__(
            name="ProgressWatchdog",
            config=self._daemon_config,
            cycle_interval=float(self._daemon_config.check_interval_seconds),
        )

        self._progress: dict[str, ConfigProgress] = {}
        self._init_progress_tracking()
        self._total_stalls_detected = 0
        self._total_recoveries_triggered = 0

    def _init_progress_tracking(self) -> None:
        """Initialize progress tracking for all configs."""
        for config_key in CANONICAL_CONFIGS:
            self._progress[config_key] = ConfigProgress(config_key=config_key)

    @property
    def config(self) -> ProgressWatchdogConfig:
        """Get daemon configuration."""
        return self._daemon_config

    # =========================================================================
    # Main Cycle
    # =========================================================================

    async def _run_cycle(self) -> None:
        """Main daemon cycle - check all configs for stalls."""
        for config_key in CANONICAL_CONFIGS:
            try:
                await self._check_config_progress(config_key)
            except asyncio.CancelledError:
                # Dec 29, 2025: Task cancellation must always propagate
                raise
            except asyncio.TimeoutError as e:
                # Timeout on database/P2P queries - expected in distributed context
                logger.warning(f"Timeout checking progress for {config_key}: {e}")
                self._record_error(f"timeout: {e}")
            except (KeyError, ValueError) as e:
                # Invalid config_key or data validation errors
                logger.error(f"Invalid config or data for {config_key}: {e}")
                self._record_error(str(e))
            except Exception as e:
                # Other errors - daemon loop must continue
                logger.error(f"Error checking progress for {config_key}: {e}")
                self._record_error(str(e))

    async def _check_config_progress(self, config_key: str) -> None:
        """Check progress for a single config."""
        progress = self._progress[config_key]
        progress.reset_recovery_counter_if_needed()

        # Get current Elo velocity
        velocity = await self._get_elo_velocity(config_key)

        # Update tracking
        now = time.time()
        progress.velocity = velocity
        progress.last_update_time = now

        # Check for stall
        if velocity < self.config.min_elo_velocity:
            if not progress.is_stalled:
                # Just started stalling
                progress.stall_start_time = now
                logger.warning(
                    f"Config {config_key} started stalling: velocity={velocity:.2f}"
                )
            elif progress.stall_duration_hours >= self.config.stall_threshold_hours:
                # Stalled long enough - trigger recovery
                await self._trigger_recovery(config_key, progress)
        else:
            # Not stalled - check if recovering
            if progress.is_stalled:
                logger.info(
                    f"Config {config_key} recovered: velocity={velocity:.2f}"
                )
                await self._emit_recovery_event(config_key, progress)
                progress.stall_start_time = 0.0

            # Update Elo tracking
            progress.last_elo = progress.current_elo
            current_elo = await self._get_current_elo(config_key)
            if current_elo is not None:
                progress.current_elo = current_elo

    async def _get_elo_velocity(self, config_key: str) -> float:
        """Get Elo velocity for a config.

        Returns Elo gain per hour based on recent history.
        """
        try:
            # Try to get from SelfplayScheduler if available
            try:
                from app.coordination.selfplay_scheduler import get_selfplay_scheduler

                scheduler = get_selfplay_scheduler()
                if scheduler and hasattr(scheduler, "get_elo_velocity"):
                    velocity = scheduler.get_elo_velocity(config_key)
                    if velocity is not None:
                        return velocity
            except ImportError:
                pass

            # Fallback: calculate from Elo database
            return await self._calculate_elo_velocity_from_db(config_key)
        except (sqlite3.Error, OSError, ValueError, AttributeError) as e:
            logger.debug(f"Could not get Elo velocity for {config_key}: {e}")
            return 0.0

    async def _calculate_elo_velocity_from_db(self, config_key: str) -> float:
        """Calculate Elo velocity from rating history in database.

        Dec 29, 2025: Wrapped SQLite operations in asyncio.to_thread() to avoid
        blocking the event loop.
        """
        try:
            from pathlib import Path

            db_path = Path("data/elo_ratings.db")
            if not db_path.exists():
                return 0.0

            def _fetch_ratings() -> list[tuple[float, str]]:
                """Sync helper to fetch ratings from database."""
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT rating, timestamp
                    FROM rating_history
                    WHERE config_key = ?
                    AND timestamp > datetime('now', '-6 hours')
                    ORDER BY timestamp
                    """,
                    (config_key,),
                )
                rows = cursor.fetchall()
                conn.close()
                return rows

            rows = await asyncio.to_thread(_fetch_ratings)

            if len(rows) < 2:
                return 0.0

            # Calculate velocity from first to last
            first_rating, first_time = rows[0]
            last_rating, last_time = rows[-1]

            from datetime import datetime

            t1 = datetime.fromisoformat(first_time)
            t2 = datetime.fromisoformat(last_time)
            hours = (t2 - t1).total_seconds() / 3600

            if hours < 0.1:
                return 0.0

            return (last_rating - first_rating) / hours
        except (sqlite3.Error, OSError, ValueError, TypeError) as e:
            logger.debug(f"Error calculating Elo velocity: {e}")
            return 0.0

    async def _get_current_elo(self, config_key: str) -> float | None:
        """Get current Elo for a config.

        Dec 29, 2025: Wrapped SQLite operations in asyncio.to_thread() to avoid
        blocking the event loop.
        """
        try:
            from pathlib import Path

            db_path = Path("data/elo_ratings.db")
            if not db_path.exists():
                return None

            def _fetch_elo() -> float | None:
                """Sync helper to fetch Elo from database."""
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT rating FROM ratings WHERE config_key = ?",
                    (config_key,),
                )
                row = cursor.fetchone()
                conn.close()
                return row[0] if row else None

            return await asyncio.to_thread(_fetch_elo)
        except (sqlite3.Error, OSError):
            # Database access or file system errors
            return None

    # =========================================================================
    # Recovery
    # =========================================================================

    async def _trigger_recovery(
        self, config_key: str, progress: ConfigProgress
    ) -> None:
        """Trigger recovery action for a stalled config.

        January 2026: Implements tiered escalation based on stall duration:
        - 6-48h: 2x boost (standard recovery)
        - 48-96h: 4x boost (extended stall)
        - 96-168h: 4x boost + curriculum reset (prolonged stall)
        - 168h+: 4x boost + architecture exploration (severe stall)
        """
        # Check if we can trigger recovery
        if progress.recovery_attempts >= self.config.max_recovery_attempts:
            logger.warning(
                f"Config {config_key} stalled but max recoveries ({self.config.max_recovery_attempts}) reached"
            )
            return

        stall_hours = progress.stall_duration_hours

        # January 2026: Determine escalation tier and actions
        boost_multiplier = self.config.boost_multiplier  # Default 2x
        trigger_curriculum_reset = False
        trigger_architecture_exploration = False
        escalation_tier = "standard"

        if stall_hours >= self.config.severe_stall_hours:
            # Severe stall (168h+): Maximum intervention
            boost_multiplier = self.config.extended_boost_multiplier
            trigger_curriculum_reset = True
            trigger_architecture_exploration = True
            escalation_tier = "severe"
            logger.warning(
                f"SEVERE STALL: {config_key} stalled {stall_hours:.0f}h - "
                f"triggering {boost_multiplier}x boost + curriculum reset + architecture exploration"
            )
        elif stall_hours >= self.config.prolonged_stall_hours:
            # Prolonged stall (96h+): Major intervention
            boost_multiplier = self.config.extended_boost_multiplier
            trigger_curriculum_reset = True
            escalation_tier = "prolonged"
            logger.warning(
                f"PROLONGED STALL: {config_key} stalled {stall_hours:.0f}h - "
                f"triggering {boost_multiplier}x boost + curriculum reset"
            )
        elif stall_hours >= self.config.extended_stall_hours:
            # Extended stall (48h+): Increased boost
            boost_multiplier = self.config.extended_boost_multiplier
            escalation_tier = "extended"
            logger.warning(
                f"EXTENDED STALL: {config_key} stalled {stall_hours:.0f}h - "
                f"triggering {boost_multiplier}x boost"
            )
        else:
            # Standard stall (6h+): Normal recovery
            logger.warning(
                f"Triggering recovery for {config_key}: stalled {stall_hours:.1f}h"
            )

        # Update tracking
        progress.recovery_attempts += 1
        progress.last_recovery_time = time.time()
        self._total_stalls_detected += 1
        self._total_recoveries_triggered += 1

        # Emit primary stall event
        safe_emit_event(
            "progress_stall_detected",
            {
                "config_key": config_key,
                "action": self.config.recovery_action,
                "stall_duration_hours": stall_hours,
                "current_velocity": progress.velocity,
                "boost_multiplier": boost_multiplier,
                "recovery_attempt": progress.recovery_attempts,
                "escalation_tier": escalation_tier,
                "source": "ProgressWatchdogDaemon",
            },
            context="ProgressWatchdog",
        )

        # January 2026: Emit additional events for escalated interventions
        if trigger_curriculum_reset:
            safe_emit_event(
                "curriculum_reset_requested",
                {
                    "config_key": config_key,
                    "reason": f"stall_recovery_{escalation_tier}",
                    "stall_duration_hours": stall_hours,
                    "source": "ProgressWatchdogDaemon",
                },
                context="ProgressWatchdog",
            )

        if trigger_architecture_exploration:
            safe_emit_event(
                "architecture_exploration_requested",
                {
                    "config_key": config_key,
                    "reason": f"stall_recovery_{escalation_tier}",
                    "stall_duration_hours": stall_hours,
                    "source": "ProgressWatchdogDaemon",
                },
                context="ProgressWatchdog",
            )

    async def _emit_recovery_event(
        self, config_key: str, progress: ConfigProgress
    ) -> None:
        """Emit event when a config recovers from stall."""
        safe_emit_event(
            "progress_recovered",
            {
                "config_key": config_key,
                "recovery_duration_hours": progress.stall_duration_hours,
                "current_velocity": progress.velocity,
                "source": "ProgressWatchdogDaemon",
            },
            context="ProgressWatchdog",
        )

    # =========================================================================
    # Health & Status
    # =========================================================================

    def health_check(self) -> HealthCheckResult:
        """Return health status."""
        stalled_configs = [
            k for k, v in self._progress.items()
            if v.is_stalled and v.stall_duration_hours >= self.config.stall_threshold_hours
        ]

        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="ProgressWatchdog not running",
                details={"stalled_configs": stalled_configs},
            )

        if stalled_configs:
            return HealthCheckResult(
                healthy=True,  # Still healthy, just detecting stalls
                status=CoordinatorStatus.RUNNING,
                message=f"{len(stalled_configs)} configs stalled: {stalled_configs}",
                details={
                    "stalled_configs": stalled_configs,
                    "cycles_completed": self._stats.cycles_completed,
                    "total_stalls": self._total_stalls_detected,
                },
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message="All configs making progress",
            details={
                "cycles_completed": self._stats.cycles_completed,
                "configs_tracked": len(self._progress),
                "total_stalls": self._total_stalls_detected,
                "total_recoveries": self._total_recoveries_triggered,
            },
        )

    def get_status(self) -> dict[str, Any]:
        """Return detailed status."""
        base_status = super().get_status()

        # Add progress details
        progress_summary = {}
        for config_key, progress in self._progress.items():
            progress_summary[config_key] = {
                "velocity": progress.velocity,
                "current_elo": progress.current_elo,
                "stalled": progress.is_stalled,
                "stall_hours": progress.stall_duration_hours if progress.is_stalled else 0,
                "recovery_attempts_24h": progress.recovery_attempts,
            }

        base_status["progress"] = progress_summary
        base_status["total_stalls_detected"] = self._total_stalls_detected
        base_status["total_recoveries_triggered"] = self._total_recoveries_triggered

        return base_status

    def get_stalled_configs(self) -> list[str]:
        """Get list of currently stalled configs."""
        return [
            k for k, v in self._progress.items()
            if v.is_stalled and v.stall_duration_hours >= self.config.stall_threshold_hours
        ]

    def get_progress_summary(self) -> dict[str, dict[str, Any]]:
        """Get progress summary for all configs."""
        return {
            config_key: {
                "velocity": progress.velocity,
                "elo": progress.current_elo,
                "stalled": progress.is_stalled,
            }
            for config_key, progress in self._progress.items()
        }


# =============================================================================
# Singleton Access (using HandlerBase class methods)
# =============================================================================


def get_progress_watchdog() -> ProgressWatchdogDaemon:
    """Get or create the singleton ProgressWatchdog instance.

    Uses HandlerBase.get_instance() for thread-safe singleton access.
    """
    return ProgressWatchdogDaemon.get_instance()


def reset_progress_watchdog() -> None:
    """Reset the singleton instance (for testing).

    Uses HandlerBase.reset_instance() for thread-safe cleanup.
    """
    ProgressWatchdogDaemon.reset_instance()
