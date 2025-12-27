"""Unified Feedback Orchestrator - Consolidates all feedback signals into one controller.

This module replaces 4 separate feedback systems with a single orchestrator:
1. FeedbackLoopController → training_intensity, exploration_boost
2. GauntletFeedbackController → exploration, training_intensity (DUPLICATE)
3. CurriculumIntegration → curriculum_weights
4. TrainingFreshness → data_freshness (ISOLATED)

The UnifiedFeedbackOrchestrator provides:
- Single source of truth for all feedback signals
- Consolidated event emission to event bus
- Configurable strategy per signal type
- Observable metrics for dashboard integration
- Per-config feedback state tracking

Key Improvements:
- Eliminates duplicate exploration/intensity logic between FeedbackLoop and GauntletFeedback
- Integrates isolated data freshness signals into main feedback loop
- Provides unified curriculum weight calculation from all sources
- Emits consolidated events for downstream consumption

Usage:
    from app.coordination.unified_feedback import (
        UnifiedFeedbackOrchestrator,
        get_unified_feedback,
        FeedbackSignal,
        FeedbackStrategy,
    )

    # Get singleton instance
    orchestrator = get_unified_feedback()

    # Start the orchestrator (subscribes to all relevant events)
    await orchestrator.start()

    # Query current feedback state
    state = orchestrator.get_config_state("hex8_2p")
    print(f"Training intensity: {state.training_intensity}")
    print(f"Exploration boost: {state.exploration_boost}")
    print(f"Curriculum weight: {state.curriculum_weight}")
    print(f"Data freshness: {state.data_freshness}")

    # Get dashboard metrics
    metrics = orchestrator.get_metrics()

Created: December 2025
Purpose: Consolidate feedback system fragmentation (4 → 1)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.config.thresholds import (
    MEDIUM_QUALITY_THRESHOLD,
    QUALITY_EXCELLENT_THRESHOLD,
    QUALITY_GOOD_THRESHOLD,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration & Enums
# =============================================================================


class FeedbackSignal(str, Enum):
    """Types of feedback signals managed by orchestrator."""

    TRAINING_INTENSITY = "training_intensity"  # paused, reduced, normal, accelerated, hot_path
    EXPLORATION_BOOST = "exploration_boost"  # 1.0 = normal, >1.0 = more exploration
    CURRICULUM_WEIGHT = "curriculum_weight"  # Relative priority for selfplay allocation
    DATA_FRESHNESS = "data_freshness"  # Hours since data was last updated


class FeedbackStrategy(str, Enum):
    """Strategy for computing feedback signals."""

    # Training Intensity strategies
    QUALITY_GRADIENT = "quality_gradient"  # Based on quality score gradient
    MOMENTUM_BASED = "momentum_based"  # Based on Elo velocity
    HYBRID = "hybrid"  # Combines quality and momentum

    # Exploration strategies
    QUALITY_REACTIVE = "quality_reactive"  # React to low quality
    LOSS_REACTIVE = "loss_reactive"  # React to training loss anomalies
    PLATEAU_REACTIVE = "plateau_reactive"  # React to Elo plateaus
    ADAPTIVE = "adaptive"  # Combines all signals

    # Curriculum weight strategies
    MOMENTUM_WEIGHTED = "momentum_weighted"  # Weight by Elo velocity
    QUALITY_WEIGHTED = "quality_weighted"  # Weight by data quality
    BALANCED = "balanced"  # Balance momentum and quality


@dataclass
class FeedbackConfig:
    """Configuration for unified feedback orchestrator."""

    # Strategy selection
    intensity_strategy: FeedbackStrategy = FeedbackStrategy.HYBRID
    exploration_strategy: FeedbackStrategy = FeedbackStrategy.ADAPTIVE
    curriculum_strategy: FeedbackStrategy = FeedbackStrategy.BALANCED

    # Training intensity thresholds (from app.config.thresholds)
    quality_excellent_threshold: float = QUALITY_EXCELLENT_THRESHOLD  # → hot_path
    quality_good_threshold: float = QUALITY_GOOD_THRESHOLD  # → accelerated
    quality_adequate_threshold: float = 0.65  # → normal
    quality_poor_threshold: float = 0.50  # → reduced
    # Below 0.50 → paused

    # Exploration boost parameters
    low_quality_threshold: float = MEDIUM_QUALITY_THRESHOLD
    base_exploration_boost: float = 1.3  # For low quality
    anomaly_exploration_boost: float = 1.15  # Per anomaly, cumulative
    max_exploration_boost: float = 2.0

    # Curriculum weight parameters
    weight_min: float = 0.5
    weight_max: float = 2.0
    momentum_weight_factor: float = 0.3  # Boost for accelerating momentum
    quality_penalty_factor: float = 0.15  # Reduction per quality penalty

    # Data freshness parameters
    max_data_age_hours: float = 1.0
    freshness_check_interval: float = 300.0  # 5 minutes

    # Promotion thresholds
    policy_accuracy_threshold: float = 0.75
    promotion_win_rate_threshold: float = 0.60

    # Cooldowns
    adjustment_cooldown: float = 300.0  # 5 minutes between adjustments


@dataclass
class FeedbackState:
    """Unified feedback state for a single configuration."""

    config_key: str

    # Current signal values
    training_intensity: str = "normal"  # paused, reduced, normal, accelerated, hot_path
    exploration_boost: float = 1.0
    curriculum_weight: float = 1.0
    data_freshness_hours: float = float("inf")

    # Source metrics (inputs to feedback computation)
    last_selfplay_quality: float = 0.0
    last_training_accuracy: float = 0.0
    last_evaluation_elo: float = 1000.0
    last_evaluation_win_rate: float = 0.0
    elo_velocity: float = 0.0  # Elo/hour

    # Status tracking
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    consecutive_anomalies: int = 0
    quality_penalties_applied: int = 0

    # Timing
    last_selfplay_time: float = 0.0
    last_training_time: float = 0.0
    last_evaluation_time: float = 0.0
    last_promotion_time: float = 0.0
    last_adjustment_time: float = 0.0

    # Elo history for velocity tracking
    elo_history: deque = field(default_factory=lambda: deque(maxlen=10))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config_key": self.config_key,
            "signals": {
                "training_intensity": self.training_intensity,
                "exploration_boost": self.exploration_boost,
                "curriculum_weight": self.curriculum_weight,
                "data_freshness_hours": self.data_freshness_hours,
            },
            "metrics": {
                "selfplay_quality": self.last_selfplay_quality,
                "training_accuracy": self.last_training_accuracy,
                "evaluation_elo": self.last_evaluation_elo,
                "evaluation_win_rate": self.last_evaluation_win_rate,
                "elo_velocity": self.elo_velocity,
            },
            "status": {
                "consecutive_successes": self.consecutive_successes,
                "consecutive_failures": self.consecutive_failures,
                "consecutive_anomalies": self.consecutive_anomalies,
                "quality_penalties": self.quality_penalties_applied,
            },
        }


# =============================================================================
# Unified Feedback Orchestrator
# =============================================================================


class UnifiedFeedbackOrchestrator:
    """Consolidates all feedback signals into a single controller.

    This orchestrator replaces 4 separate feedback systems:
    1. FeedbackLoopController - training_intensity, exploration_boost
    2. GauntletFeedbackController - exploration, training_intensity (duplicate)
    3. CurriculumIntegration - curriculum_weights
    4. TrainingFreshness - data_freshness

    Event subscriptions:
    - SELFPLAY_COMPLETE → update quality, compute intensity/curriculum
    - TRAINING_COMPLETED → update accuracy, trigger evaluation
    - EVALUATION_COMPLETED → update elo, compute exploration/curriculum
    - MODEL_PROMOTED → update success/failure counters
    - TRAINING_LOSS_ANOMALY → boost exploration
    - QUALITY_DEGRADED → adjust intensity/curriculum
    - DATA_FRESH/DATA_STALE → update freshness signal
    - PLATEAU_DETECTED → adjust exploration
    - REGRESSION_DETECTED → boost intensity

    Emits consolidated events:
    - FEEDBACK_SIGNALS_UPDATED → {config, intensity, exploration, weight, freshness}
    - TRAINING_INTENSITY_CHANGED → FeedbackAccelerator
    - EXPLORATION_BOOST_CHANGED → TemperatureScheduler
    - CURRICULUM_WEIGHT_CHANGED → CurriculumFeedback
    """

    def __init__(self, config: FeedbackConfig | None = None):
        """Initialize the unified feedback orchestrator.

        Args:
            config: Feedback configuration (uses defaults if None)
        """
        self.config = config or FeedbackConfig()
        self._states: dict[str, FeedbackState] = {}
        self._running = False
        self._subscribed = False
        self._lock = threading.Lock()

        # Metrics for observability
        self._total_adjustments = 0
        self._adjustments_by_type: dict[FeedbackSignal, int] = {
            signal: 0 for signal in FeedbackSignal
        }
        self._last_adjustment_time = 0.0

        logger.debug("[UnifiedFeedbackOrchestrator] Initialized")

    def _get_or_create_state(self, config_key: str) -> FeedbackState:
        """Get or create feedback state for a config.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")

        Returns:
            FeedbackState for the config
        """
        with self._lock:
            if config_key not in self._states:
                self._states[config_key] = FeedbackState(config_key=config_key)
            return self._states[config_key]

    async def start(self) -> None:
        """Start the orchestrator and subscribe to events."""
        if self._running:
            logger.warning("[UnifiedFeedbackOrchestrator] Already running")
            return

        self._running = True
        self._subscribe_to_events()

        logger.info("[UnifiedFeedbackOrchestrator] Started")

    async def stop(self) -> None:
        """Stop the orchestrator and unsubscribe from events."""
        self._running = False
        self._unsubscribe_from_events()
        logger.info("[UnifiedFeedbackOrchestrator] Stopped")

    async def health_check(self) -> bool:
        """Check if the orchestrator is healthy.

        Returns True if running and functioning properly.
        Used by DaemonManager for crash detection and auto-restart.
        """
        if not self._running:
            return False

        # Check we have at least processed some configs
        # (indicates event subscription is working)
        if not self._subscribed:
            logger.warning(
                "[UnifiedFeedbackOrchestrator] health_check failed: not subscribed to events"
            )
            return False

        # Check for stale adjustments (if we should be adjusting but haven't)
        if self._total_adjustments == 0 and len(self._states) > 0:
            # We have states but no adjustments - might be stuck
            # Only warn if we've been running for a while
            uptime = time.time() - self._last_adjustment_time if self._last_adjustment_time > 0 else 0
            if uptime > 3600:  # 1 hour without adjustments is concerning
                logger.warning(
                    f"[UnifiedFeedbackOrchestrator] health_check warning: "
                    f"no adjustments in {uptime:.0f}s with {len(self._states)} active states"
                )

        return True

    def _subscribe_to_events(self) -> None:
        """Subscribe to all relevant events."""
        if self._subscribed:
            return

        try:
            from app.coordination.event_router import DataEventType, get_event_bus

            bus = get_event_bus()
            if bus is None:
                logger.warning("[UnifiedFeedbackOrchestrator] Event bus not available")
                return

            # Core pipeline events
            bus.subscribe(DataEventType.SELFPLAY_COMPLETE, self._on_selfplay_complete)
            bus.subscribe(DataEventType.TRAINING_COMPLETED, self._on_training_complete)
            bus.subscribe(DataEventType.EVALUATION_COMPLETED, self._on_evaluation_complete)
            bus.subscribe(DataEventType.MODEL_PROMOTED, self._on_promotion_complete)

            # Quality & anomaly events
            bus.subscribe(DataEventType.TRAINING_LOSS_ANOMALY, self._on_loss_anomaly)
            bus.subscribe(DataEventType.QUALITY_DEGRADED, self._on_quality_degraded)
            bus.subscribe(DataEventType.PLATEAU_DETECTED, self._on_plateau_detected)
            bus.subscribe(DataEventType.REGRESSION_DETECTED, self._on_regression_detected)

            # Data freshness events
            bus.subscribe(DataEventType.DATA_FRESH, self._on_data_fresh)
            bus.subscribe(DataEventType.DATA_STALE, self._on_data_stale)

            # Elo momentum events
            if hasattr(DataEventType, "ELO_VELOCITY_CHANGED"):
                bus.subscribe(DataEventType.ELO_VELOCITY_CHANGED, self._on_elo_velocity_changed)

            # Quality penalty events
            if hasattr(DataEventType, "QUALITY_PENALTY_APPLIED"):
                bus.subscribe(DataEventType.QUALITY_PENALTY_APPLIED, self._on_quality_penalty)

            logger.info("[UnifiedFeedbackOrchestrator] Subscribed to events")

        except Exception as e:
            logger.warning(f"[UnifiedFeedbackOrchestrator] Failed to subscribe: {e}")
        finally:
            # December 27, 2025: Always set _subscribed = True in finally block
            # This ensures cleanup runs even if subscription partially fails
            self._subscribed = True

    def _unsubscribe_from_events(self) -> None:
        """Unsubscribe from all events."""
        if not self._subscribed:
            return

        try:
            from app.coordination.event_router import DataEventType, get_event_bus

            bus = get_event_bus()
            if bus:
                bus.unsubscribe(DataEventType.SELFPLAY_COMPLETE, self._on_selfplay_complete)
                bus.unsubscribe(DataEventType.TRAINING_COMPLETED, self._on_training_complete)
                bus.unsubscribe(DataEventType.EVALUATION_COMPLETED, self._on_evaluation_complete)
                bus.unsubscribe(DataEventType.MODEL_PROMOTED, self._on_promotion_complete)
                bus.unsubscribe(DataEventType.TRAINING_LOSS_ANOMALY, self._on_loss_anomaly)
                bus.unsubscribe(DataEventType.QUALITY_DEGRADED, self._on_quality_degraded)
                bus.unsubscribe(DataEventType.PLATEAU_DETECTED, self._on_plateau_detected)
                bus.unsubscribe(DataEventType.REGRESSION_DETECTED, self._on_regression_detected)
                bus.unsubscribe(DataEventType.DATA_FRESH, self._on_data_fresh)
                bus.unsubscribe(DataEventType.DATA_STALE, self._on_data_stale)

                if hasattr(DataEventType, "ELO_VELOCITY_CHANGED"):
                    bus.unsubscribe(DataEventType.ELO_VELOCITY_CHANGED, self._on_elo_velocity_changed)

                if hasattr(DataEventType, "QUALITY_PENALTY_APPLIED"):
                    bus.unsubscribe(DataEventType.QUALITY_PENALTY_APPLIED, self._on_quality_penalty)

            self._subscribed = False

        except Exception as e:
            logger.debug(f"[UnifiedFeedbackOrchestrator] Error unsubscribing: {e}")

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def _on_selfplay_complete(self, event: Any) -> None:
        """Handle SELFPLAY_COMPLETE event."""
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            config_key = payload.get("config", "")
            games_count = payload.get("games_count", 0)

            if not config_key:
                return

            state = self._get_or_create_state(config_key)
            state.last_selfplay_time = time.time()

            # Assess quality (simplified - actual implementation would use UnifiedQualityScorer)
            quality_score = self._estimate_quality_from_payload(payload)
            old_quality = state.last_selfplay_quality
            state.last_selfplay_quality = quality_score

            logger.info(
                f"[UnifiedFeedbackOrchestrator] Selfplay complete: {config_key}, "
                f"quality={quality_score:.2f} (prev={old_quality:.2f})"
            )

            # Recompute feedback signals
            self._recompute_signals(state)

        except Exception as e:
            logger.error(f"[UnifiedFeedbackOrchestrator] Error handling selfplay complete: {e}")

    def _on_training_complete(self, event: Any) -> None:
        """Handle TRAINING_COMPLETED event."""
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            config_key = payload.get("config", "")
            policy_accuracy = payload.get("policy_accuracy", 0.0)

            if not config_key:
                return

            state = self._get_or_create_state(config_key)
            state.last_training_time = time.time()
            state.last_training_accuracy = policy_accuracy

            logger.info(
                f"[UnifiedFeedbackOrchestrator] Training complete: {config_key}, "
                f"accuracy={policy_accuracy:.2%}"
            )

            # Recompute signals
            self._recompute_signals(state)

        except Exception as e:
            logger.error(f"[UnifiedFeedbackOrchestrator] Error handling training complete: {e}")

    def _on_evaluation_complete(self, event: Any) -> None:
        """Handle EVALUATION_COMPLETED event."""
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            config_key = payload.get("config", "")
            elo = payload.get("elo", 1000.0)
            win_rate = payload.get("win_rate", 0.5)

            if not config_key:
                return

            state = self._get_or_create_state(config_key)
            state.last_evaluation_time = time.time()

            # Update Elo and compute velocity
            old_elo = state.last_evaluation_elo
            state.last_evaluation_elo = elo
            state.last_evaluation_win_rate = win_rate
            state.elo_history.append((time.time(), elo))

            # Compute Elo velocity (change per hour)
            state.elo_velocity = self._compute_elo_velocity(state)

            logger.info(
                f"[UnifiedFeedbackOrchestrator] Evaluation complete: {config_key}, "
                f"elo={elo:.0f} (Δ{elo-old_elo:+.0f}), velocity={state.elo_velocity:.1f}/hr"
            )

            # Recompute signals
            self._recompute_signals(state)

        except Exception as e:
            logger.error(f"[UnifiedFeedbackOrchestrator] Error handling evaluation complete: {e}")

    def _on_promotion_complete(self, event: Any) -> None:
        """Handle MODEL_PROMOTED event."""
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            metadata = payload.get("metadata", payload)
            config_key = metadata.get("config") or metadata.get("config_key", "")
            promoted = metadata.get("promoted", False)

            if not config_key:
                return

            state = self._get_or_create_state(config_key)
            state.last_promotion_time = time.time()

            if promoted:
                state.consecutive_successes += 1
                state.consecutive_failures = 0
                logger.info(
                    f"[UnifiedFeedbackOrchestrator] Promotion SUCCESS: {config_key}, "
                    f"streak={state.consecutive_successes}"
                )
            else:
                state.consecutive_failures += 1
                state.consecutive_successes = 0
                logger.info(
                    f"[UnifiedFeedbackOrchestrator] Promotion FAILED: {config_key}, "
                    f"streak={state.consecutive_failures}"
                )

            # Recompute signals
            self._recompute_signals(state)

        except Exception as e:
            logger.error(f"[UnifiedFeedbackOrchestrator] Error handling promotion: {e}")

    def _on_loss_anomaly(self, event: Any) -> None:
        """Handle TRAINING_LOSS_ANOMALY event."""
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            config_key = payload.get("config", "")

            if not config_key:
                return

            state = self._get_or_create_state(config_key)
            state.consecutive_anomalies += 1

            logger.warning(
                f"[UnifiedFeedbackOrchestrator] Loss anomaly: {config_key}, "
                f"count={state.consecutive_anomalies}"
            )

            # Recompute signals (will boost exploration)
            self._recompute_signals(state)

        except Exception as e:
            logger.error(f"[UnifiedFeedbackOrchestrator] Error handling loss anomaly: {e}")

    def _on_quality_degraded(self, event: Any) -> None:
        """Handle QUALITY_DEGRADED event."""
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            config_key = payload.get("config_key", "")
            quality_score = payload.get("quality_score", 0.5)

            if not config_key:
                return

            state = self._get_or_create_state(config_key)
            state.last_selfplay_quality = quality_score

            logger.warning(
                f"[UnifiedFeedbackOrchestrator] Quality degraded: {config_key}, "
                f"score={quality_score:.2f}"
            )

            # Recompute signals
            self._recompute_signals(state)

        except Exception as e:
            logger.error(f"[UnifiedFeedbackOrchestrator] Error handling quality degraded: {e}")

    def _on_plateau_detected(self, event: Any) -> None:
        """Handle PLATEAU_DETECTED event."""
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            config_key = payload.get("config_key", payload.get("config", ""))

            if not config_key:
                return

            state = self._get_or_create_state(config_key)

            logger.info(f"[UnifiedFeedbackOrchestrator] Plateau detected: {config_key}")

            # Boost exploration to escape plateau
            self._recompute_signals(state)

        except Exception as e:
            logger.error(f"[UnifiedFeedbackOrchestrator] Error handling plateau: {e}")

    def _on_regression_detected(self, event: Any) -> None:
        """Handle REGRESSION_DETECTED event."""
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            config_key = payload.get("config_key", payload.get("config", ""))

            if not config_key:
                return

            state = self._get_or_create_state(config_key)
            state.consecutive_failures += 1

            logger.warning(
                f"[UnifiedFeedbackOrchestrator] Regression detected: {config_key}, "
                f"failures={state.consecutive_failures}"
            )

            # Recompute signals (will increase intensity)
            self._recompute_signals(state)

        except Exception as e:
            logger.error(f"[UnifiedFeedbackOrchestrator] Error handling regression: {e}")

    def _on_data_fresh(self, event: Any) -> None:
        """Handle DATA_FRESH event."""
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            config_key = f"{payload.get('board_type', '')}_{payload.get('num_players', 0)}p"
            data_age_hours = payload.get("data_age_hours", 0.0)

            state = self._get_or_create_state(config_key)
            state.data_freshness_hours = data_age_hours

            logger.debug(f"[UnifiedFeedbackOrchestrator] Data fresh: {config_key}, age={data_age_hours:.1f}h")

        except Exception as e:
            logger.debug(f"[UnifiedFeedbackOrchestrator] Error handling data fresh: {e}")

    def _on_data_stale(self, event: Any) -> None:
        """Handle DATA_STALE event."""
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            config_key = f"{payload.get('board_type', '')}_{payload.get('num_players', 0)}p"
            data_age_hours = payload.get("data_age_hours", float("inf"))

            state = self._get_or_create_state(config_key)
            state.data_freshness_hours = data_age_hours

            logger.warning(f"[UnifiedFeedbackOrchestrator] Data stale: {config_key}, age={data_age_hours:.1f}h")

        except Exception as e:
            logger.debug(f"[UnifiedFeedbackOrchestrator] Error handling data stale: {e}")

    def _on_elo_velocity_changed(self, event: Any) -> None:
        """Handle ELO_VELOCITY_CHANGED event."""
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            config_key = payload.get("config_key", "")
            velocity = payload.get("velocity", 0.0)

            if not config_key:
                return

            state = self._get_or_create_state(config_key)
            state.elo_velocity = velocity

            logger.info(f"[UnifiedFeedbackOrchestrator] Elo velocity changed: {config_key}, v={velocity:.1f}/hr")

            # Recompute curriculum weight based on momentum
            self._recompute_signals(state)

        except Exception as e:
            logger.debug(f"[UnifiedFeedbackOrchestrator] Error handling velocity change: {e}")

    def _on_quality_penalty(self, event: Any) -> None:
        """Handle QUALITY_PENALTY_APPLIED event."""
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            config_key = payload.get("config_key", "")
            penalty = payload.get("new_penalty", 0)

            if not config_key:
                return

            state = self._get_or_create_state(config_key)
            state.quality_penalties_applied = int(penalty)

            logger.info(f"[UnifiedFeedbackOrchestrator] Quality penalty: {config_key}, penalty={penalty}")

            # Recompute signals
            self._recompute_signals(state)

        except Exception as e:
            logger.debug(f"[UnifiedFeedbackOrchestrator] Error handling quality penalty: {e}")

    # =========================================================================
    # Signal Computation
    # =========================================================================

    def _recompute_signals(self, state: FeedbackState) -> None:
        """Recompute all feedback signals for a config.

        Args:
            state: Feedback state to update
        """
        # Check cooldown
        now = time.time()
        if now - state.last_adjustment_time < self.config.adjustment_cooldown:
            return

        # Compute each signal
        old_intensity = state.training_intensity
        old_exploration = state.exploration_boost
        old_weight = state.curriculum_weight

        state.training_intensity = self._compute_training_intensity(state)
        state.exploration_boost = self._compute_exploration_boost(state)
        state.curriculum_weight = self._compute_curriculum_weight(state)

        # Check if anything changed
        intensity_changed = old_intensity != state.training_intensity
        exploration_changed = abs(old_exploration - state.exploration_boost) > 0.05
        weight_changed = abs(old_weight - state.curriculum_weight) > 0.05

        if not (intensity_changed or exploration_changed or weight_changed):
            return

        # Update adjustment time and counters
        state.last_adjustment_time = now
        self._total_adjustments += 1
        self._last_adjustment_time = now

        if intensity_changed:
            self._adjustments_by_type[FeedbackSignal.TRAINING_INTENSITY] += 1
        if exploration_changed:
            self._adjustments_by_type[FeedbackSignal.EXPLORATION_BOOST] += 1
        if weight_changed:
            self._adjustments_by_type[FeedbackSignal.CURRICULUM_WEIGHT] += 1

        # Log changes
        logger.info(
            f"[UnifiedFeedbackOrchestrator] Signals updated for {state.config_key}: "
            f"intensity={old_intensity}→{state.training_intensity}, "
            f"exploration={old_exploration:.2f}→{state.exploration_boost:.2f}, "
            f"weight={old_weight:.2f}→{state.curriculum_weight:.2f}"
        )

        # Emit consolidated event
        self._emit_signals_updated(state)

    def _compute_training_intensity(self, state: FeedbackState) -> str:
        """Compute training intensity level.

        Args:
            state: Feedback state

        Returns:
            Intensity level: "paused", "reduced", "normal", "accelerated", "hot_path"
        """
        if self.config.intensity_strategy == FeedbackStrategy.QUALITY_GRADIENT:
            return self._intensity_from_quality(state.last_selfplay_quality)

        elif self.config.intensity_strategy == FeedbackStrategy.MOMENTUM_BASED:
            return self._intensity_from_momentum(state)

        else:  # HYBRID
            quality_intensity = self._intensity_from_quality(state.last_selfplay_quality)
            momentum_intensity = self._intensity_from_momentum(state)

            # Take the more aggressive setting
            intensity_levels = ["paused", "reduced", "normal", "accelerated", "hot_path"]
            q_level = intensity_levels.index(quality_intensity)
            m_level = intensity_levels.index(momentum_intensity)
            return intensity_levels[max(q_level, m_level)]

    def _intensity_from_quality(self, quality: float) -> str:
        """Map quality score to intensity level."""
        if quality >= self.config.quality_excellent_threshold:
            return "hot_path"
        elif quality >= self.config.quality_good_threshold:
            return "accelerated"
        elif quality >= self.config.quality_adequate_threshold:
            return "normal"
        elif quality >= self.config.quality_poor_threshold:
            return "reduced"
        else:
            return "paused"

    def _intensity_from_momentum(self, state: FeedbackState) -> str:
        """Map momentum (success/failure streaks) to intensity level."""
        # Hot path after 3+ consecutive successes
        if state.consecutive_successes >= 3:
            return "hot_path"

        # Accelerated after success or 3+ failures
        if state.consecutive_successes >= 1 or state.consecutive_failures >= 3:
            return "accelerated"

        # Normal by default
        return "normal"

    def _compute_exploration_boost(self, state: FeedbackState) -> float:
        """Compute exploration boost factor.

        Args:
            state: Feedback state

        Returns:
            Exploration boost (1.0 = normal, >1.0 = more exploration)
        """
        boost = 1.0

        if self.config.exploration_strategy == FeedbackStrategy.QUALITY_REACTIVE:
            if state.last_selfplay_quality < self.config.low_quality_threshold:
                boost = self.config.base_exploration_boost

        elif self.config.exploration_strategy == FeedbackStrategy.LOSS_REACTIVE:
            # Boost per anomaly
            boost = 1.0 + (state.consecutive_anomalies * (self.config.anomaly_exploration_boost - 1.0))

        elif self.config.exploration_strategy == FeedbackStrategy.PLATEAU_REACTIVE:
            # Boost if Elo velocity is near zero
            if abs(state.elo_velocity) < 5.0:
                boost = self.config.base_exploration_boost

        else:  # ADAPTIVE - combines all signals
            # Base boost from quality
            if state.last_selfplay_quality < self.config.low_quality_threshold:
                boost = max(boost, self.config.base_exploration_boost)

            # Additional boost from anomalies
            if state.consecutive_anomalies > 0:
                anomaly_boost = 1.0 + (state.consecutive_anomalies * (self.config.anomaly_exploration_boost - 1.0))
                boost = max(boost, anomaly_boost)

            # Additional boost from plateau
            if abs(state.elo_velocity) < 5.0:
                boost = max(boost, 1.2)

            # Reduce after successful promotion
            if state.consecutive_successes > 0:
                boost = max(1.0, boost * 0.9)

        # Cap at max
        return min(self.config.max_exploration_boost, boost)

    def _compute_curriculum_weight(self, state: FeedbackState) -> float:
        """Compute curriculum weight.

        Args:
            state: Feedback state

        Returns:
            Curriculum weight (relative priority for selfplay allocation)
        """
        base_weight = 1.0

        if self.config.curriculum_strategy == FeedbackStrategy.MOMENTUM_WEIGHTED:
            # Weight by Elo velocity
            if state.elo_velocity > 10.0:
                base_weight *= (1.0 + self.config.momentum_weight_factor)
            elif state.elo_velocity < -10.0:
                base_weight *= (1.0 - self.config.momentum_weight_factor)

        elif self.config.curriculum_strategy == FeedbackStrategy.QUALITY_WEIGHTED:
            # Weight by quality
            if state.last_selfplay_quality < 0.5:
                base_weight *= (1.0 + self.config.momentum_weight_factor)
            elif state.last_selfplay_quality > 0.7:
                base_weight *= (1.0 - self.config.momentum_weight_factor * 0.5)

        else:  # BALANCED
            # Combine momentum and quality
            if state.elo_velocity > 10.0 or state.last_selfplay_quality < 0.5:
                base_weight *= (1.0 + self.config.momentum_weight_factor)
            elif state.elo_velocity < -10.0 and state.last_selfplay_quality > 0.7:
                base_weight *= (1.0 - self.config.momentum_weight_factor * 0.5)

        # Apply quality penalty
        if state.quality_penalties_applied > 0:
            penalty_factor = 1.0 - (state.quality_penalties_applied * self.config.quality_penalty_factor)
            base_weight *= max(0.3, penalty_factor)

        # Clamp to range
        return max(self.config.weight_min, min(self.config.weight_max, base_weight))

    def _compute_elo_velocity(self, state: FeedbackState) -> float:
        """Compute Elo velocity (change per hour).

        Args:
            state: Feedback state

        Returns:
            Elo velocity in points/hour
        """
        if len(state.elo_history) < 2:
            return 0.0

        # Linear regression over Elo history
        points = list(state.elo_history)
        n = len(points)

        sum_t = sum(t for t, _ in points)
        sum_elo = sum(elo for _, elo in points)
        sum_t_elo = sum(t * elo for t, elo in points)
        sum_t2 = sum(t * t for t, _ in points)

        # Slope = (n * sum(t*elo) - sum(t)*sum(elo)) / (n * sum(t^2) - sum(t)^2)
        denominator = n * sum_t2 - sum_t * sum_t
        if abs(denominator) < 1e-6:
            return 0.0

        slope = (n * sum_t_elo - sum_t * sum_elo) / denominator

        # Convert from points/second to points/hour
        return slope * 3600

    def _estimate_quality_from_payload(self, payload: dict[str, Any]) -> float:
        """Estimate quality score from event payload.

        Args:
            payload: Event payload

        Returns:
            Quality score 0.0-1.0
        """
        # Check if quality score is directly provided
        if "quality_score" in payload:
            return payload["quality_score"]

        # Estimate from game count (fallback)
        games_count = payload.get("games_count", 0)
        if games_count < 100:
            return 0.3
        elif games_count < 500:
            return 0.6
        elif games_count < 1000:
            return 0.8
        else:
            return 0.9

    # =========================================================================
    # Event Emission
    # =========================================================================

    def _emit_signals_updated(self, state: FeedbackState) -> None:
        """Emit consolidated FEEDBACK_SIGNALS_UPDATED event.

        Args:
            state: Updated feedback state
        """
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            if router:
                # Emit consolidated event
                router.publish_sync(
                    "FEEDBACK_SIGNALS_UPDATED",
                    {
                        "config_key": state.config_key,
                        "signals": {
                            "training_intensity": state.training_intensity,
                            "exploration_boost": state.exploration_boost,
                            "curriculum_weight": state.curriculum_weight,
                            "data_freshness_hours": state.data_freshness_hours,
                        },
                        "metrics": {
                            "selfplay_quality": state.last_selfplay_quality,
                            "training_accuracy": state.last_training_accuracy,
                            "elo": state.last_evaluation_elo,
                            "elo_velocity": state.elo_velocity,
                        },
                        "timestamp": time.time(),
                    },
                    source="unified_feedback_orchestrator",
                )

                # Also emit individual signal change events for backward compatibility
                self._emit_individual_signals(state)

        except Exception as e:
            logger.debug(f"[UnifiedFeedbackOrchestrator] Error emitting event: {e}")

    def _emit_individual_signals(self, state: FeedbackState) -> None:
        """Emit individual signal change events for backward compatibility.

        Args:
            state: Feedback state
        """
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            if not router:
                return

            # Training intensity changed
            router.publish_sync(
                "TRAINING_INTENSITY_CHANGED",
                {
                    "config_key": state.config_key,
                    "intensity": state.training_intensity,
                    "quality": state.last_selfplay_quality,
                },
                source="unified_feedback_orchestrator",
            )

            # Exploration boost changed
            router.publish_sync(
                "EXPLORATION_BOOST",
                {
                    "config_key": state.config_key,
                    "boost_factor": state.exploration_boost,
                    "reason": "unified_feedback",
                },
                source="unified_feedback_orchestrator",
            )

            # Curriculum weight changed
            router.publish_sync(
                "CURRICULUM_REBALANCED",
                {
                    "trigger": "unified_feedback",
                    "changed_configs": [state.config_key],
                    "new_weights": {state.config_key: state.curriculum_weight},
                    "timestamp": time.time(),
                },
                source="unified_feedback_orchestrator",
            )

        except Exception as e:
            logger.debug(f"[UnifiedFeedbackOrchestrator] Error emitting individual signals: {e}")

    # =========================================================================
    # Public API
    # =========================================================================

    def get_config_state(self, config_key: str) -> FeedbackState | None:
        """Get feedback state for a configuration.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")

        Returns:
            FeedbackState or None if not found
        """
        return self._states.get(config_key)

    def get_all_states(self) -> dict[str, FeedbackState]:
        """Get all feedback states.

        Returns:
            Dictionary of config_key -> FeedbackState
        """
        return dict(self._states)

    def health_check(self) -> "HealthCheckResult":
        """Check orchestrator health for CoordinatorProtocol compliance.

        December 2025: Added for unified daemon health monitoring.
        """
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

        if not self._running:
            return HealthCheckResult(
                healthy=True,  # Stopped is not unhealthy
                status=CoordinatorStatus.STOPPED,
                message="UnifiedFeedbackOrchestrator not running",
            )

        if not self._subscribed:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message="Not subscribed to events",
            )

        # Check if we have any states (have we processed any events?)
        configs_tracked = len(self._states)
        if configs_tracked == 0:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message="Running (no configs tracked yet)",
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"Healthy ({configs_tracked} configs, {self._total_adjustments} adjustments)",
        )

    def get_metrics(self) -> dict[str, Any]:
        """Get orchestrator metrics for dashboard integration.

        Returns:
            Dictionary of metrics
        """
        return {
            "running": self._running,
            "subscribed": self._subscribed,
            "configs_tracked": len(self._states),
            "total_adjustments": self._total_adjustments,
            "adjustments_by_type": {
                signal.value: count
                for signal, count in self._adjustments_by_type.items()
            },
            "last_adjustment_time": self._last_adjustment_time,
            "states": {
                config_key: state.to_dict()
                for config_key, state in self._states.items()
            },
        }

    def force_recompute(self, config_key: str) -> None:
        """Force recomputation of signals for a config (ignores cooldown).

        Args:
            config_key: Configuration key
        """
        state = self._states.get(config_key)
        if state:
            state.last_adjustment_time = 0.0  # Reset cooldown
            self._recompute_signals(state)


# =============================================================================
# Singleton
# =============================================================================

_orchestrator: UnifiedFeedbackOrchestrator | None = None
_orchestrator_lock = threading.Lock()


def get_unified_feedback(config: FeedbackConfig | None = None) -> UnifiedFeedbackOrchestrator:
    """Get the singleton UnifiedFeedbackOrchestrator instance.

    Args:
        config: Configuration (only used on first call)

    Returns:
        The singleton orchestrator instance
    """
    global _orchestrator
    if _orchestrator is None:
        with _orchestrator_lock:
            if _orchestrator is None:
                _orchestrator = UnifiedFeedbackOrchestrator(config)
    return _orchestrator


def reset_unified_feedback() -> None:
    """Reset the singleton (for testing)."""
    global _orchestrator
    _orchestrator = None


__all__ = [
    # Main class
    "UnifiedFeedbackOrchestrator",
    # Configuration
    "FeedbackConfig",
    "FeedbackSignal",
    "FeedbackStrategy",
    # State
    "FeedbackState",
    # Singleton
    "get_unified_feedback",
    "reset_unified_feedback",
]
