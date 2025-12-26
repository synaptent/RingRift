"""Quality Monitor Daemon - Continuous selfplay quality monitoring.

Monitors selfplay data quality and emits events when quality changes.
This enables the feedback loop to react to quality degradation by:
- Throttling selfplay when quality drops
- Triggering regeneration when quality is too low
- Resuming normal operation when quality recovers

Usage:
    from app.coordination.quality_monitor_daemon import QualityMonitorDaemon

    daemon = QualityMonitorDaemon()
    await daemon.start()

Events Emitted:
    - LOW_QUALITY_DATA_WARNING: Quality dropped below warning threshold
    - HIGH_QUALITY_DATA_AVAILABLE: Quality above good threshold
    - QUALITY_SCORE_UPDATED: Quality score changed significantly
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class QualityState(Enum):
    """State of data quality."""
    UNKNOWN = "unknown"
    EXCELLENT = "excellent"  # >= 0.9
    GOOD = "good"           # >= 0.7
    DEGRADED = "degraded"   # >= 0.5
    POOR = "poor"           # < 0.5


@dataclass
class QualityMonitorConfig:
    """Configuration for QualityMonitorDaemon."""
    check_interval: float = 15.0  # Dec 2025: Reduced from 60s for faster feedback
    warning_threshold: float = 0.6  # Quality below this triggers warning
    good_threshold: float = 0.8  # Quality above this triggers high quality event
    significant_change: float = 0.1  # Change threshold for update events
    data_dir: str = "data/games"  # Directory to monitor
    database_pattern: str = "selfplay*.db"  # Database file pattern


class QualityMonitorDaemon:
    """Daemon that monitors selfplay data quality continuously.

    Periodically checks quality of recent selfplay data and emits events
    when quality changes, enabling the feedback loop to react appropriately.

    Attributes:
        config: Monitor configuration
        last_quality: Most recent quality score
        current_state: Current quality state
    """

    def __init__(self, config: QualityMonitorConfig | None = None):
        self.config = config or QualityMonitorConfig()
        self._running = False
        self._task: asyncio.Task | None = None
        self.last_quality: float = 1.0
        self.current_state: QualityState = QualityState.UNKNOWN
        self._last_event_time: float = 0.0
        self._event_cooldown: float = 30.0  # Min seconds between same-type events
        self._subscribed: bool = False
        # Phase 9: Track per-config quality for targeted checks
        self._config_quality: dict[str, float] = {}

    async def start(self) -> None:
        """Start the quality monitor daemon."""
        if self._running:
            logger.warning("QualityMonitorDaemon already running")
            return

        self._running = True
        self._subscribe_to_events()
        self._task = asyncio.create_task(self._monitor_loop())
        self._task.add_done_callback(self._handle_task_error)
        logger.info("QualityMonitorDaemon started")

    def _subscribe_to_events(self) -> None:
        """Subscribe to quality check request events (Phase 9)."""
        if self._subscribed:
            return
        try:
            from app.coordination.event_router import DataEventType, get_event_bus

            bus = get_event_bus()
            if hasattr(DataEventType, 'QUALITY_CHECK_REQUESTED'):
                bus.subscribe(DataEventType.QUALITY_CHECK_REQUESTED, self._on_quality_check_requested)
                logger.info("[QualityMonitorDaemon] Subscribed to QUALITY_CHECK_REQUESTED")
            self._subscribed = True
        except Exception as e:
            logger.warning(f"[QualityMonitorDaemon] Failed to subscribe to events: {e}")

    async def _on_quality_check_requested(self, event) -> None:
        """Handle on-demand quality check requests (Phase 9).

        Triggered by FeedbackLoopController when training loss anomalies
        or degradation is detected. Performs an immediate quality check
        for the specific config and emits results.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            config_key = payload.get("config_key", "")
            reason = payload.get("reason", "unknown")
            priority = payload.get("priority", "normal")

            logger.info(
                f"[QualityMonitorDaemon] On-demand quality check requested for {config_key}: "
                f"reason={reason}, priority={priority}"
            )

            # Perform immediate quality check
            quality = await self._get_current_quality()

            # Store per-config quality
            if config_key:
                self._config_quality[config_key] = quality

            # Update state and emit events
            old_state = self.current_state
            new_state = self._quality_to_state(quality)

            # Always emit for on-demand checks (bypass cooldown)
            await self._emit_quality_event(quality, new_state, old_state)

            self.last_quality = quality
            self.current_state = new_state

            logger.info(
                f"[QualityMonitorDaemon] On-demand check complete for {config_key}: "
                f"quality={quality:.3f}, state={new_state.value}"
            )

        except Exception as e:
            logger.error(f"[QualityMonitorDaemon] Error handling quality check request: {e}")

    def _handle_task_error(self, task: asyncio.Task) -> None:
        """Handle errors from the monitor task."""
        try:
            if task.cancelled():
                return
            exc = task.exception()
            if exc:
                logger.error(f"QualityMonitorDaemon task failed: {exc}")
        except asyncio.InvalidStateError:
            pass

    async def stop(self) -> None:
        """Stop the quality monitor daemon."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("QualityMonitorDaemon stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_quality()
                await asyncio.sleep(self.config.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Quality check error: {e}")
                await asyncio.sleep(self.config.check_interval)

    async def _check_quality(self) -> None:
        """Check current quality and emit events if needed."""
        try:
            quality = await self._get_current_quality()
            old_state = self.current_state
            new_state = self._quality_to_state(quality)

            # Check for significant change
            quality_changed = abs(quality - self.last_quality) >= self.config.significant_change
            state_changed = old_state != new_state

            if quality_changed or state_changed:
                await self._emit_quality_event(quality, new_state, old_state)

            self.last_quality = quality
            self.current_state = new_state

        except Exception as e:
            logger.error(f"Error checking quality: {e}")

    async def _get_current_quality(self) -> float:
        """Get current quality score from recent selfplay data using UnifiedQualityScorer.

        Returns:
            Quality score between 0.0 and 1.0
        """
        try:
            import sqlite3
            from app.quality.unified_quality import compute_game_quality_from_params

            data_dir = Path(self.config.data_dir)

            if not data_dir.exists():
                logger.debug(f"Data directory not found: {data_dir}")
                return 1.0

            db_files = sorted(
                data_dir.glob(self.config.database_pattern),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )

            if not db_files:
                logger.debug("No selfplay databases found")
                return 1.0

            recent_db = db_files[0]
            try:
                with sqlite3.connect(str(recent_db)) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute("""
                        SELECT game_id, game_status, winner, termination_reason, total_moves
                        FROM games
                        WHERE game_status IN ('complete', 'finished', 'COMPLETE', 'FINISHED')
                        ORDER BY created_at DESC
                        LIMIT 30
                    """)
                    games = cursor.fetchall()

                if not games:
                    logger.debug(f"No completed games in {recent_db.name}")
                    return 0.5

                quality_scores = []
                for game in games:
                    try:
                        quality = compute_game_quality_from_params(
                            game_id=game["game_id"],
                            game_status=game["game_status"],
                            winner=game["winner"],
                            termination_reason=game["termination_reason"],
                            total_moves=game["total_moves"] or 0,
                        )
                        quality_scores.append(quality.quality_score)
                    except Exception:
                        continue

                if not quality_scores:
                    return 0.5

                avg_quality = sum(quality_scores) / len(quality_scores)
                logger.debug(f"Quality score for {recent_db.name}: {avg_quality:.3f}")
                return avg_quality

            except sqlite3.Error as e:
                logger.debug(f"Database error for {recent_db.name}: {e}")
                return self.last_quality

        except ImportError:
            logger.warning("UnifiedQualityScorer not available")
            return 1.0
        except Exception as e:
            logger.error(f"Error getting quality: {e}")
            return self.last_quality

    def _quality_to_state(self, quality: float) -> QualityState:
        """Convert quality score to state."""
        if quality >= 0.9:
            return QualityState.EXCELLENT
        elif quality >= 0.7:
            return QualityState.GOOD
        elif quality >= 0.5:
            return QualityState.DEGRADED
        else:
            return QualityState.POOR

    async def _emit_quality_event(
        self,
        quality: float,
        new_state: QualityState,
        old_state: QualityState,
    ) -> None:
        """Emit appropriate quality event based on state change."""
        try:
            from app.distributed.data_events import DataEventType
            from app.coordination.event_router import get_router

            router = get_router()

            payload = {
                "quality_score": quality,
                "old_state": old_state.value,
                "new_state": new_state.value,
                "timestamp": time.time(),
            }

            # Determine which event to emit based on transition
            now = time.time()
            if now - self._last_event_time < self._event_cooldown:
                logger.debug("Skipping event due to cooldown")
                return

            if quality < self.config.warning_threshold:
                # Quality dropped below warning threshold
                await router.publish(DataEventType.LOW_QUALITY_DATA_WARNING, payload)
                logger.warning(f"Low quality warning: {quality:.3f} (state: {new_state.value})")
            elif quality >= self.config.good_threshold and old_state in (
                QualityState.DEGRADED, QualityState.POOR, QualityState.UNKNOWN
            ):
                # Quality recovered to good threshold
                await router.publish(DataEventType.HIGH_QUALITY_DATA_AVAILABLE, payload)
                logger.info(f"High quality data available: {quality:.3f} (state: {new_state.value})")
            else:
                # Just report the quality update
                await router.publish(DataEventType.QUALITY_SCORE_UPDATED, payload)
                logger.debug(f"Quality updated: {quality:.3f} (state: {new_state.value})")

            self._last_event_time = now

        except ImportError as e:
            logger.warning(f"Event system not available: {e}")
        except Exception as e:
            logger.error(f"Error emitting quality event: {e}")


# Factory function for DaemonManager integration
async def create_quality_monitor() -> None:
    """Factory function for DaemonManager.

    Creates and runs the quality monitor daemon until cancelled.
    """
    daemon = QualityMonitorDaemon()
    await daemon.start()

    try:
        # Keep running until cancelled
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        await daemon.stop()
        raise
