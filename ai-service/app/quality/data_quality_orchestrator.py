"""DataQualityOrchestrator - Unified monitoring for data quality events (December 2025).

This module provides centralized monitoring of data quality events across the system.
It subscribes to quality-related events and provides:

1. Quality state tracking across configs
2. Quality trend analysis
3. Threshold violation detection
4. Integration with training triggers

Event Integration:
- Subscribes to QUALITY_SCORE_UPDATED: Tracks individual quality updates
- Subscribes to QUALITY_DISTRIBUTION_CHANGED: Tracks distribution shifts
- Subscribes to HIGH_QUALITY_DATA_AVAILABLE: Tracks readiness for training
- Subscribes to LOW_QUALITY_DATA_WARNING: Tracks quality degradation
- Subscribes to DATA_QUALITY_ALERT: Tracks critical quality issues

Usage:
    from app.quality.data_quality_orchestrator import (
        DataQualityOrchestrator,
        wire_quality_events,
        get_quality_orchestrator,
    )

    # Wire quality events to orchestrator
    orchestrator = wire_quality_events()

    # Get quality status
    status = orchestrator.get_status()
    print(f"Configs tracked: {status['configs_tracked']}")

    # Get config quality
    quality = orchestrator.get_config_quality("square8_2p")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    """Quality level classification."""
    EXCELLENT = "excellent"  # >= 0.9
    GOOD = "good"  # >= 0.7
    ADEQUATE = "adequate"  # >= 0.5
    POOR = "poor"  # >= 0.3
    CRITICAL = "critical"  # < 0.3

    @classmethod
    def from_score(cls, score: float) -> "QualityLevel":
        """Get quality level from numeric score."""
        if score >= 0.9:
            return cls.EXCELLENT
        elif score >= 0.7:
            return cls.GOOD
        elif score >= 0.5:
            return cls.ADEQUATE
        elif score >= 0.3:
            return cls.POOR
        else:
            return cls.CRITICAL


@dataclass
class ConfigQualityState:
    """Quality state for a single configuration."""

    config_key: str
    avg_quality_score: float = 0.0
    high_quality_count: int = 0
    low_quality_count: int = 0
    total_games: int = 0
    last_update_time: float = 0.0
    last_alert_time: float = 0.0
    quality_level: QualityLevel = QualityLevel.ADEQUATE
    is_ready_for_training: bool = False
    has_active_warning: bool = False

    # Trend tracking
    quality_trend: float = 0.0  # Positive = improving, negative = degrading
    samples_since_last_training: int = 0


@dataclass
class QualityStats:
    """Aggregate quality statistics."""

    configs_tracked: int = 0
    total_quality_updates: int = 0
    total_alerts: int = 0
    total_warnings: int = 0
    configs_ready_for_training: int = 0
    configs_with_warnings: int = 0
    avg_quality_across_configs: float = 0.0
    last_activity_time: float = 0.0


class DataQualityOrchestrator:
    """Orchestrates data quality monitoring across all configurations.

    Subscribes to quality events and maintains a unified view of data quality
    state across all active configurations.
    """

    def __init__(
        self,
        stale_threshold_seconds: float = 3600.0,
        max_history_per_config: int = 100,
        high_quality_threshold: float = 0.7,
        low_quality_threshold: float = 0.3,
    ):
        """Initialize DataQualityOrchestrator.

        Args:
            stale_threshold_seconds: Time after which config data is considered stale
            max_history_per_config: Maximum history entries per config
            high_quality_threshold: Score threshold for high quality
            low_quality_threshold: Score threshold for low quality warning
        """
        self.stale_threshold_seconds = stale_threshold_seconds
        self.max_history_per_config = max_history_per_config
        self.high_quality_threshold = high_quality_threshold
        self.low_quality_threshold = low_quality_threshold

        # Config tracking
        self._configs: Dict[str, ConfigQualityState] = {}
        self._config_history: Dict[str, List[Dict[str, Any]]] = {}

        # Statistics
        self._total_updates: int = 0
        self._total_alerts: int = 0
        self._total_warnings: int = 0
        self._subscribed: bool = False

    def subscribe_to_events(self) -> bool:
        """Subscribe to quality-related events from the event bus.

        Returns:
            True if successfully subscribed
        """
        if self._subscribed:
            return True

        try:
            from app.distributed.data_events import (
                DataEventType,
                get_event_bus,
            )

            bus = get_event_bus()

            # Subscribe to quality events
            bus.subscribe(DataEventType.QUALITY_SCORE_UPDATED, self._on_quality_score_updated)
            bus.subscribe(DataEventType.QUALITY_DISTRIBUTION_CHANGED, self._on_distribution_changed)
            bus.subscribe(DataEventType.HIGH_QUALITY_DATA_AVAILABLE, self._on_high_quality_available)
            bus.subscribe(DataEventType.LOW_QUALITY_DATA_WARNING, self._on_low_quality_warning)
            bus.subscribe(DataEventType.DATA_QUALITY_ALERT, self._on_quality_alert)

            self._subscribed = True
            logger.info("[DataQualityOrchestrator] Subscribed to quality events")
            return True

        except Exception as e:
            logger.warning(f"[DataQualityOrchestrator] Failed to subscribe to events: {e}")
            return False

    def unsubscribe(self) -> None:
        """Unsubscribe from quality events."""
        if not self._subscribed:
            return

        try:
            from app.distributed.data_events import (
                DataEventType,
                get_event_bus,
            )

            bus = get_event_bus()
            bus.unsubscribe(DataEventType.QUALITY_SCORE_UPDATED, self._on_quality_score_updated)
            bus.unsubscribe(DataEventType.QUALITY_DISTRIBUTION_CHANGED, self._on_distribution_changed)
            bus.unsubscribe(DataEventType.HIGH_QUALITY_DATA_AVAILABLE, self._on_high_quality_available)
            bus.unsubscribe(DataEventType.LOW_QUALITY_DATA_WARNING, self._on_low_quality_warning)
            bus.unsubscribe(DataEventType.DATA_QUALITY_ALERT, self._on_quality_alert)
            self._subscribed = False

        except Exception:
            pass

    def _get_or_create_config(self, config_key: str) -> ConfigQualityState:
        """Get or create a config quality state."""
        if config_key not in self._configs:
            self._configs[config_key] = ConfigQualityState(config_key=config_key)
        return self._configs[config_key]

    def _on_quality_score_updated(self, event: Any) -> None:
        """Handle QUALITY_SCORE_UPDATED event."""
        payload = event.payload if hasattr(event, 'payload') else {}

        config_key = payload.get("config", "")
        quality_score = payload.get("quality_score", payload.get("new_score", 0.5))

        if not config_key:
            return

        config = self._get_or_create_config(config_key)

        # Update running average
        old_avg = config.avg_quality_score
        config.total_games += 1
        config.avg_quality_score = (
            (old_avg * (config.total_games - 1) + quality_score) / config.total_games
        )

        # Track trend
        if old_avg > 0:
            config.quality_trend = quality_score - old_avg

        # Update counts
        if quality_score >= self.high_quality_threshold:
            config.high_quality_count += 1
        elif quality_score < self.low_quality_threshold:
            config.low_quality_count += 1

        config.last_update_time = time.time()
        config.quality_level = QualityLevel.from_score(config.avg_quality_score)
        config.samples_since_last_training += 1

        self._total_updates += 1

    def _on_distribution_changed(self, event: Any) -> None:
        """Handle QUALITY_DISTRIBUTION_CHANGED event."""
        payload = event.payload if hasattr(event, 'payload') else {}

        config_key = payload.get("config", "")
        avg_quality = payload.get("avg_quality", 0.5)
        high_quality_ratio = payload.get("high_quality_ratio", 0.5)
        low_quality_ratio = payload.get("low_quality_ratio", 0.1)
        total_games = payload.get("total_games", 0)

        if not config_key:
            return

        config = self._get_or_create_config(config_key)

        # Calculate trend from old to new average
        old_avg = config.avg_quality_score
        config.quality_trend = avg_quality - old_avg if old_avg > 0 else 0

        config.avg_quality_score = avg_quality
        config.total_games = total_games
        config.high_quality_count = int(total_games * high_quality_ratio)
        config.low_quality_count = int(total_games * low_quality_ratio)
        config.last_update_time = time.time()
        config.quality_level = QualityLevel.from_score(avg_quality)

        logger.info(
            f"[DataQualityOrchestrator] Distribution changed for {config_key}: "
            f"avg={avg_quality:.2f}, level={config.quality_level.value}"
        )

        self._add_to_history(config_key, "distribution_changed", {
            "avg_quality": avg_quality,
            "high_quality_ratio": high_quality_ratio,
            "low_quality_ratio": low_quality_ratio,
            "trend": config.quality_trend,
            "timestamp": time.time(),
        })

    def _on_high_quality_available(self, event: Any) -> None:
        """Handle HIGH_QUALITY_DATA_AVAILABLE event."""
        payload = event.payload if hasattr(event, 'payload') else {}

        config_key = payload.get("config", "")
        high_quality_count = payload.get("high_quality_count", 0)
        avg_quality = payload.get("avg_quality", 0.7)

        if not config_key:
            return

        config = self._get_or_create_config(config_key)
        config.is_ready_for_training = True
        config.high_quality_count = high_quality_count
        config.avg_quality_score = avg_quality
        config.last_update_time = time.time()

        logger.info(
            f"[DataQualityOrchestrator] High-quality data ready for {config_key}: "
            f"{high_quality_count} games (avg: {avg_quality:.2f})"
        )

        self._add_to_history(config_key, "ready_for_training", {
            "high_quality_count": high_quality_count,
            "avg_quality": avg_quality,
            "timestamp": time.time(),
        })

    def _on_low_quality_warning(self, event: Any) -> None:
        """Handle LOW_QUALITY_DATA_WARNING event."""
        payload = event.payload if hasattr(event, 'payload') else {}

        config_key = payload.get("config", "")
        low_quality_count = payload.get("low_quality_count", 0)
        low_quality_ratio = payload.get("low_quality_ratio", 0.0)
        avg_quality = payload.get("avg_quality", 0.5)

        if not config_key:
            return

        config = self._get_or_create_config(config_key)
        config.has_active_warning = True
        config.low_quality_count = low_quality_count
        config.avg_quality_score = avg_quality
        config.last_alert_time = time.time()
        config.quality_level = QualityLevel.from_score(avg_quality)

        logger.warning(
            f"[DataQualityOrchestrator] Low quality warning for {config_key}: "
            f"{low_quality_ratio:.1%} low-quality games"
        )

        self._total_warnings += 1

        self._add_to_history(config_key, "low_quality_warning", {
            "low_quality_count": low_quality_count,
            "low_quality_ratio": low_quality_ratio,
            "avg_quality": avg_quality,
            "timestamp": time.time(),
        })

    def _on_quality_alert(self, event: Any) -> None:
        """Handle DATA_QUALITY_ALERT event."""
        payload = event.payload if hasattr(event, 'payload') else {}

        config_key = payload.get("config", "")
        alert_type = payload.get("alert_type", "unknown")
        message = payload.get("message", "")

        if not config_key:
            return

        config = self._get_or_create_config(config_key)
        config.has_active_warning = True
        config.last_alert_time = time.time()

        logger.error(
            f"[DataQualityOrchestrator] Quality alert for {config_key}: "
            f"{alert_type} - {message}"
        )

        self._total_alerts += 1

        self._add_to_history(config_key, "quality_alert", {
            "alert_type": alert_type,
            "message": message,
            "timestamp": time.time(),
        })

    def _add_to_history(self, config_key: str, event_type: str, data: Dict[str, Any]) -> None:
        """Add entry to config history."""
        if config_key not in self._config_history:
            self._config_history[config_key] = []

        history = self._config_history[config_key]
        history.append({
            "event_type": event_type,
            **data,
        })

        # Trim history if needed
        if len(history) > self.max_history_per_config:
            self._config_history[config_key] = history[-self.max_history_per_config:]

    def get_config_quality(self, config_key: str) -> Optional[ConfigQualityState]:
        """Get quality state for a specific config."""
        return self._configs.get(config_key)

    def get_configs_ready_for_training(self) -> List[ConfigQualityState]:
        """Get all configs that are ready for training."""
        return [c for c in self._configs.values() if c.is_ready_for_training]

    def get_configs_with_warnings(self) -> List[ConfigQualityState]:
        """Get all configs with active quality warnings."""
        return [c for c in self._configs.values() if c.has_active_warning]

    def get_config_history(self, config_key: Optional[str] = None) -> Dict[str, List[Dict]]:
        """Get quality history."""
        if config_key:
            return {config_key: self._config_history.get(config_key, [])}
        return dict(self._config_history)

    def get_stats(self) -> QualityStats:
        """Get aggregate quality statistics."""
        configs_ready = len(self.get_configs_ready_for_training())
        configs_warned = len(self.get_configs_with_warnings())

        avg_quality = 0.0
        if self._configs:
            avg_quality = sum(c.avg_quality_score for c in self._configs.values()) / len(self._configs)

        last_activity = 0.0
        for config in self._configs.values():
            last_activity = max(last_activity, config.last_update_time, config.last_alert_time)

        return QualityStats(
            configs_tracked=len(self._configs),
            total_quality_updates=self._total_updates,
            total_alerts=self._total_alerts,
            total_warnings=self._total_warnings,
            configs_ready_for_training=configs_ready,
            configs_with_warnings=configs_warned,
            avg_quality_across_configs=avg_quality,
            last_activity_time=last_activity,
        )

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status for monitoring."""
        stats = self.get_stats()

        return {
            "subscribed": self._subscribed,
            "configs_tracked": stats.configs_tracked,
            "total_quality_updates": stats.total_quality_updates,
            "total_alerts": stats.total_alerts,
            "total_warnings": stats.total_warnings,
            "configs_ready_for_training": stats.configs_ready_for_training,
            "configs_with_warnings": stats.configs_with_warnings,
            "avg_quality_across_configs": stats.avg_quality_across_configs,
            "last_activity_time": stats.last_activity_time,
            "config_keys": list(self._configs.keys()),
        }

    def mark_training_started(self, config_key: str) -> None:
        """Mark that training has started for a config.

        Resets training readiness and sample counts.
        """
        config = self._configs.get(config_key)
        if config:
            config.is_ready_for_training = False
            config.samples_since_last_training = 0

    def clear_warning(self, config_key: str) -> None:
        """Clear active warning for a config."""
        config = self._configs.get(config_key)
        if config:
            config.has_active_warning = False


# Singleton instance
_quality_orchestrator: Optional[DataQualityOrchestrator] = None


def wire_quality_events(
    stale_threshold_seconds: float = 3600.0,
    high_quality_threshold: float = 0.7,
) -> DataQualityOrchestrator:
    """Wire quality events to the orchestrator.

    This enables centralized monitoring of data quality state across
    all configurations.

    Args:
        stale_threshold_seconds: Time after which data is stale
        high_quality_threshold: Quality score for high-quality classification

    Returns:
        DataQualityOrchestrator instance
    """
    global _quality_orchestrator

    if _quality_orchestrator is None:
        _quality_orchestrator = DataQualityOrchestrator(
            stale_threshold_seconds=stale_threshold_seconds,
            high_quality_threshold=high_quality_threshold,
        )
        _quality_orchestrator.subscribe_to_events()

        logger.info(
            f"[wire_quality_events] Quality events wired to orchestrator "
            f"(high_threshold={high_quality_threshold})"
        )

    return _quality_orchestrator


def get_quality_orchestrator() -> Optional[DataQualityOrchestrator]:
    """Get the global quality orchestrator if configured."""
    return _quality_orchestrator


def reset_quality_orchestrator() -> None:
    """Reset the quality orchestrator singleton (for testing)."""
    global _quality_orchestrator
    if _quality_orchestrator:
        _quality_orchestrator.unsubscribe()
    _quality_orchestrator = None


__all__ = [
    "DataQualityOrchestrator",
    "ConfigQualityState",
    "QualityStats",
    "QualityLevel",
    "wire_quality_events",
    "get_quality_orchestrator",
    "reset_quality_orchestrator",
]
