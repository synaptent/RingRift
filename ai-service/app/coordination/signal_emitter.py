"""Signal emission utilities for training feedback signals.

Extracted from feedback_loop_controller.py (December 2025).
Provides pure functions for computing and emitting training signals
including adaptive training parameters, exploration adjustments, and
selfplay target updates.

Usage:
    from app.coordination.signal_emitter import (
        emit_selfplay_adjustment,
        emit_exploration_adjustment,
        emit_curriculum_training_feedback,
        emit_adaptive_training_signal,
        AdaptiveTrainingSignal,
        ExplorationAdjustment,
        SelfplayAdjustment,
        compute_adaptive_signal,
        compute_exploration_adjustment,
    )

    # Compute and emit adaptive signal
    signal = compute_adaptive_signal(elo_improvement=60)
    emit_adaptive_training_signal("hex8_2p", signal)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Elo-based thresholds
ELO_PLATEAU_THRESHOLD = 10.0  # Elo improvement below this is plateau
ELO_STRONG_IMPROVEMENT = 50.0  # Strong improvement threshold
ELO_REGRESSION_THRESHOLD = -30.0  # Regression threshold

# Training accuracy thresholds
POLICY_LOW_THRESHOLD = 0.40  # Policy accuracy below this needs attention
POLICY_HIGH_THRESHOLD = 0.70  # Policy accuracy above this is excellent

# Curriculum weight adjustments
CURRICULUM_WEIGHT_ADJUSTMENT_UP = 0.15  # Boost for struggling configs
CURRICULUM_WEIGHT_ADJUSTMENT_DOWN = -0.05  # Reduction for successful configs

# Exploration settings
EXPLORATION_BOOST_MAX = 2.0  # Maximum exploration boost
FAILURE_EXPLORATION_BOOST = 1.3  # Boost multiplier after failures


class PositionDifficulty(str, Enum):
    """Position difficulty levels for selfplay."""

    EASY = "easy"
    NORMAL = "normal"
    MEDIUM_HARD = "medium-hard"
    HARD = "hard"


class SignalPriority(str, Enum):
    """Priority levels for training signals."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class AdaptiveTrainingSignal:
    """Adaptive training parameters based on evaluation feedback.

    December 2025: Phase 6 - Training parameters adapt to eval outcomes.

    Attributes:
        learning_rate_multiplier: Multiplier for learning rate (1.0 = no change)
        batch_size_multiplier: Multiplier for batch size (1.0 = no change)
        epochs_extension: Additional epochs to add (0 = no change)
        gradient_clip_enabled: Whether to enable gradient clipping
        reason: Human-readable reason for the adjustment
    """

    learning_rate_multiplier: float = 1.0
    batch_size_multiplier: float = 1.0
    epochs_extension: int = 0
    gradient_clip_enabled: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for event payload."""
        return {
            "learning_rate_multiplier": self.learning_rate_multiplier,
            "batch_size_multiplier": self.batch_size_multiplier,
            "epochs_extension": self.epochs_extension,
            "gradient_clip_enabled": self.gradient_clip_enabled,
            "reason": self.reason,
        }

    @property
    def has_adjustment(self) -> bool:
        """Check if any adjustment is needed."""
        return (
            self.learning_rate_multiplier != 1.0
            or self.batch_size_multiplier != 1.0
            or self.epochs_extension != 0
            or self.gradient_clip_enabled
        )


@dataclass
class ExplorationAdjustment:
    """Exploration adjustment parameters for selfplay.

    December 2025: Quality-driven selfplay exploration signals.

    Attributes:
        position_difficulty: Difficulty level for position selection
        mcts_budget_multiplier: Multiplier for MCTS search budget
        exploration_temp_boost: Multiplier for exploration temperature
        noise_boost: Additional Dirichlet noise (can be negative)
        quality_score: Quality score that triggered adjustment
        trend: Quality trend (improving, declining, stable)
    """

    position_difficulty: PositionDifficulty = PositionDifficulty.NORMAL
    mcts_budget_multiplier: float = 1.0
    exploration_temp_boost: float = 1.0
    noise_boost: float = 0.0
    quality_score: float = 0.0
    trend: str = "stable"

    def to_dict(self) -> dict:
        """Convert to dictionary for event payload."""
        return {
            "position_difficulty": self.position_difficulty.value,
            "mcts_budget_multiplier": self.mcts_budget_multiplier,
            "exploration_temp_boost": self.exploration_temp_boost,
            "noise_boost": self.noise_boost,
            "quality_score": self.quality_score,
            "trend": self.trend,
        }

    @property
    def has_adjustment(self) -> bool:
        """Check if any adjustment differs from baseline."""
        return (
            self.position_difficulty != PositionDifficulty.NORMAL
            or self.mcts_budget_multiplier != 1.0
            or self.exploration_temp_boost != 1.0
            or self.noise_boost != 0.0
        )


@dataclass
class SelfplayAdjustment:
    """Selfplay target adjustment based on Elo feedback.

    December 2025: Closes evaluation → selfplay feedback loop.

    Attributes:
        search_budget: Updated MCTS search budget
        exploration_boost: Exploration temperature boost
        elo_gap: Gap to target Elo
        velocity: Current Elo velocity (Elo/hour)
        priority: Priority level for this adjustment
        reason: Human-readable reason
    """

    search_budget: int = 150
    exploration_boost: float = 1.0
    elo_gap: float = 0.0
    velocity: float = 0.0
    priority: SignalPriority = SignalPriority.NORMAL
    reason: str = "velocity_feedback"

    def to_dict(self) -> dict:
        """Convert to dictionary for event payload."""
        return {
            "search_budget": self.search_budget,
            "exploration_boost": self.exploration_boost,
            "elo_gap": self.elo_gap,
            "velocity": self.velocity,
            "priority": self.priority.value,
            "reason": self.reason,
        }


@dataclass
class CurriculumFeedback:
    """Curriculum feedback based on training metrics.

    December 2025: Training → curriculum feedback loop.

    Attributes:
        config_key: Configuration key
        policy_accuracy: Policy head accuracy
        value_accuracy: Value head accuracy
        adjustment: Weight adjustment amount
        new_weight: New curriculum weight after adjustment
        trigger: What triggered this feedback
    """

    config_key: str
    policy_accuracy: float = 0.0
    value_accuracy: float = 0.0
    adjustment: float = 0.0
    new_weight: float = 1.0
    trigger: str = "training_complete"

    def to_dict(self) -> dict:
        """Convert to dictionary for event payload."""
        return {
            "config": self.config_key,
            "policy_accuracy": self.policy_accuracy,
            "value_accuracy": self.value_accuracy,
            "adjustment": self.adjustment,
            "new_weight": self.new_weight,
            "trigger": self.trigger,
            "timestamp": time.time(),
        }


# ============================================================================
# Pure Computation Functions
# ============================================================================


def compute_adaptive_signal(
    elo_improvement: float,
    current_elo: float = 1500.0,
) -> AdaptiveTrainingSignal:
    """Compute adaptive training parameters based on Elo improvement.

    Strategy:
    - Strong improvement (>50 Elo): Extend training epochs to capitalize
    - Plateau (<10 Elo improvement): Reduce LR, increase batch size
    - Regression (<-30 Elo): Aggressive LR reduction, enable gradient clipping

    Args:
        elo_improvement: Elo gain from previous evaluation
        current_elo: Current Elo rating (for logging context)

    Returns:
        AdaptiveTrainingSignal with adjusted parameters
    """
    signal = AdaptiveTrainingSignal()

    # Strong improvement: extend training to capitalize on momentum
    if elo_improvement > ELO_STRONG_IMPROVEMENT:
        signal.epochs_extension = 10
        signal.reason = f"Strong improvement ({elo_improvement:.0f} Elo) - extending training"

    # Plateau: reduce LR, increase batch size for smoother convergence
    elif elo_improvement < ELO_PLATEAU_THRESHOLD:
        signal.learning_rate_multiplier = 0.5
        signal.batch_size_multiplier = 1.5
        signal.gradient_clip_enabled = True
        signal.reason = (
            f"Plateau ({elo_improvement:.0f} Elo) - reducing LR, enabling grad clip"
        )

    # Regression: aggressive LR reduction
    if elo_improvement < ELO_REGRESSION_THRESHOLD:
        signal.learning_rate_multiplier = 0.2
        signal.gradient_clip_enabled = True
        signal.epochs_extension = 0  # Don't extend on regression
        signal.reason = (
            f"Regression ({elo_improvement:.0f} Elo) - aggressive LR reduction"
        )

    return signal


def compute_exploration_adjustment(
    quality_score: float,
    trend: str = "stable",
) -> ExplorationAdjustment:
    """Compute exploration adjustment based on quality score.

    Quality-driven selfplay exploration signals:
    - Very low quality: Harder positions, more MCTS budget, higher temperature
    - Medium quality: Slightly harder positions
    - High quality: Can reduce budget for efficiency
    - Declining trend: Extra exploration boost

    Args:
        quality_score: Current quality score (0.0-1.0)
        trend: Quality trend ("improving", "declining", "stable")

    Returns:
        ExplorationAdjustment with computed parameters
    """
    adjustment = ExplorationAdjustment(quality_score=quality_score, trend=trend)

    # Determine base adjustments based on quality level
    if quality_score < 0.5:
        # Very low quality → aggressive exploration
        adjustment.position_difficulty = PositionDifficulty.HARD
        adjustment.mcts_budget_multiplier = 1.5  # 50% more MCTS budget
        adjustment.exploration_temp_boost = 1.3
        adjustment.noise_boost = 0.10
    elif quality_score < 0.7:
        # Medium quality → slightly harder positions
        adjustment.position_difficulty = PositionDifficulty.MEDIUM_HARD
        adjustment.mcts_budget_multiplier = 1.2  # 20% more MCTS budget
        adjustment.exploration_temp_boost = 1.15
        adjustment.noise_boost = 0.05
    elif quality_score > 0.9:
        # High quality → can reduce budget for efficiency
        adjustment.position_difficulty = PositionDifficulty.NORMAL
        adjustment.mcts_budget_multiplier = 0.8  # 20% less budget
        adjustment.exploration_temp_boost = 1.0
        adjustment.noise_boost = 0.0
    else:
        # Normal quality
        adjustment.position_difficulty = PositionDifficulty.NORMAL
        adjustment.mcts_budget_multiplier = 1.0
        adjustment.exploration_temp_boost = 1.0
        adjustment.noise_boost = 0.0

    # Boost exploration if trend is declining
    if trend == "declining":
        adjustment.exploration_temp_boost *= 1.2
        adjustment.mcts_budget_multiplier = max(adjustment.mcts_budget_multiplier, 1.3)
        adjustment.noise_boost = max(adjustment.noise_boost, 0.05)

    return adjustment


def compute_curriculum_adjustment(
    policy_accuracy: float,
    value_accuracy: float,
    current_weight: float = 1.0,
    weight_min: float = 0.5,
    weight_max: float = 2.0,
) -> CurriculumFeedback:
    """Compute curriculum weight adjustment based on training accuracy.

    Logic:
    - Low policy accuracy (< 40%): Boost weight (needs more training)
    - High policy accuracy (> 70%): Reduce weight (learning well)
    - Otherwise: No change

    Args:
        policy_accuracy: Policy head accuracy (0.0-1.0)
        value_accuracy: Value head accuracy (0.0-1.0)
        current_weight: Current curriculum weight
        weight_min: Minimum allowed weight
        weight_max: Maximum allowed weight

    Returns:
        CurriculumFeedback with adjustment info
    """
    adjustment = 0.0

    if policy_accuracy < POLICY_LOW_THRESHOLD:
        adjustment = CURRICULUM_WEIGHT_ADJUSTMENT_UP  # Boost
    elif policy_accuracy > POLICY_HIGH_THRESHOLD:
        adjustment = CURRICULUM_WEIGHT_ADJUSTMENT_DOWN  # Reduce

    new_weight = current_weight
    if adjustment != 0.0:
        new_weight = max(weight_min, min(weight_max, current_weight + adjustment))

    return CurriculumFeedback(
        config_key="",  # Filled by caller
        policy_accuracy=policy_accuracy,
        value_accuracy=value_accuracy,
        adjustment=adjustment,
        new_weight=new_weight,
    )


def compute_selfplay_adjustment(
    elo_gap: float,
    velocity: float,
    current_search_budget: int = 150,
    current_exploration_boost: float = 1.0,
    velocity_threshold: float = 10.0,
) -> SelfplayAdjustment:
    """Compute selfplay target adjustment based on Elo feedback.

    Args:
        elo_gap: Gap to target Elo
        velocity: Current Elo velocity (Elo/hour)
        current_search_budget: Current MCTS search budget
        current_exploration_boost: Current exploration boost
        velocity_threshold: Velocity below this is considered plateau

    Returns:
        SelfplayAdjustment with priority and parameters
    """
    priority = SignalPriority.NORMAL

    # High priority if far from target or plateau detected
    if elo_gap > 500 or velocity < velocity_threshold:
        priority = SignalPriority.HIGH

    # Urgent if very far from target
    if elo_gap > 1000:
        priority = SignalPriority.URGENT

    return SelfplayAdjustment(
        search_budget=current_search_budget,
        exploration_boost=current_exploration_boost,
        elo_gap=elo_gap,
        velocity=velocity,
        priority=priority,
        reason="velocity_feedback",
    )


# ============================================================================
# Safe Async/Sync Helpers
# ============================================================================


def _safe_create_task(
    coro: Any,
    name: str,
    error_callback: Callable[[Exception], None] | None = None,
) -> asyncio.Task | None:
    """Safely create an asyncio task with error handling.

    Args:
        coro: Coroutine to run as a task
        name: Name for the task (for logging)
        error_callback: Optional callback for errors

    Returns:
        Task if created successfully, None otherwise
    """
    try:
        loop = asyncio.get_running_loop()

        def handle_done(task: asyncio.Task) -> None:
            try:
                exc = task.exception()
                if exc is not None:
                    logger.debug(f"Task {name} failed: {exc}")
                    if error_callback:
                        error_callback(exc)
            except (asyncio.CancelledError, asyncio.InvalidStateError):
                pass

        task = loop.create_task(coro, name=name)
        task.add_done_callback(handle_done)
        return task

    except RuntimeError:
        # No running event loop
        return None


async def _emit_event_async(event_type: str, payload: dict, source: str) -> bool:
    """Emit an event asynchronously.

    Args:
        event_type: Event type name (e.g., "SELFPLAY_TARGET_UPDATED")
        payload: Event payload dictionary
        source: Event source identifier

    Returns:
        True if event was emitted successfully
    """
    try:
        from app.coordination.event_router import DataEventType, get_event_bus
        from app.distributed.data_events import DataEvent

        bus = get_event_bus()
        if bus is None:
            return False

        event_enum = getattr(DataEventType, event_type, None)
        if event_enum is None:
            logger.debug(f"Event type {event_type} not found in DataEventType")
            return False

        event = DataEvent(
            event_type=event_enum,
            payload=payload,
            source=source,
        )
        await bus.publish(event)
        return True

    except (ImportError, AttributeError, TypeError, RuntimeError) as e:
        logger.debug(f"Failed to emit {event_type}: {e}")
        return False


def _emit_event_sync(event_type: str, payload: dict, source: str) -> bool:
    """Emit an event synchronously (fire-and-forget).

    Args:
        event_type: Event type name (e.g., "SELFPLAY_TARGET_UPDATED")
        payload: Event payload dictionary
        source: Event source identifier

    Returns:
        True if event was queued successfully
    """
    try:
        from app.coordination.event_router import DataEventType, get_event_bus

        bus = get_event_bus()
        if bus is None:
            return False

        event_enum = getattr(DataEventType, event_type, None)
        if event_enum is None:
            return False

        # Use emit() which is synchronous
        bus.emit(event_enum, payload)
        return True

    except (ImportError, AttributeError, TypeError, RuntimeError) as e:
        logger.debug(f"Failed to emit {event_type}: {e}")
        return False


# ============================================================================
# Signal Emission Functions
# ============================================================================


def emit_selfplay_adjustment(
    config_key: str,
    adjustment: SelfplayAdjustment,
) -> bool:
    """Emit SELFPLAY_TARGET_UPDATED event.

    Args:
        config_key: Configuration key
        adjustment: Selfplay adjustment parameters

    Returns:
        True if event was emitted
    """
    payload = {
        "config_key": config_key,
        **adjustment.to_dict(),
    }
    return _emit_event_sync(
        "SELFPLAY_TARGET_UPDATED",
        payload,
        "signal_emitter",
    )


def emit_exploration_adjustment(
    config_key: str,
    adjustment: ExplorationAdjustment,
) -> bool:
    """Emit EXPLORATION_ADJUSTED event.

    Only emits if adjustment differs from baseline.

    Args:
        config_key: Configuration key
        adjustment: Exploration adjustment parameters

    Returns:
        True if event was emitted (or no adjustment needed)
    """
    if not adjustment.has_adjustment:
        return True  # No adjustment needed, success

    payload = {
        "config_key": config_key,
        **adjustment.to_dict(),
        "timestamp": time.time(),
    }
    result = _emit_event_sync(
        "EXPLORATION_ADJUSTED",
        payload,
        "signal_emitter",
    )

    if result:
        logger.info(
            f"[SignalEmitter] Exploration adjusted for {config_key}: "
            f"difficulty={adjustment.position_difficulty.value}, "
            f"mcts_mult={adjustment.mcts_budget_multiplier:.1f}, "
            f"temp_boost={adjustment.exploration_temp_boost:.2f}"
        )
    return result


def emit_adaptive_training_signal(
    config_key: str,
    signal: AdaptiveTrainingSignal,
) -> bool:
    """Emit ADAPTIVE_PARAMS_CHANGED event.

    Only emits if signal has adjustments.

    Args:
        config_key: Configuration key
        signal: Adaptive training parameters

    Returns:
        True if event was emitted (or no adjustment needed)
    """
    if not signal.has_adjustment:
        return True  # No adjustment needed

    payload = {
        "config_key": config_key,
        **signal.to_dict(),
    }
    result = _emit_event_sync(
        "ADAPTIVE_PARAMS_CHANGED",
        payload,
        "signal_emitter",
    )

    if result:
        logger.debug(f"[SignalEmitter] Emitted ADAPTIVE_PARAMS_CHANGED for {config_key}")
    return result


def emit_curriculum_training_feedback(
    feedback: CurriculumFeedback,
    weights_dict: dict[str, float] | None = None,
) -> bool:
    """Emit CURRICULUM_REBALANCED event.

    Args:
        feedback: Curriculum feedback with adjustment info
        weights_dict: Optional current weights dictionary

    Returns:
        True if event was emitted
    """
    payload = feedback.to_dict()
    if weights_dict:
        payload["new_weights"] = weights_dict

    result = _emit_event_sync(
        "CURRICULUM_REBALANCED",
        payload,
        "signal_emitter",
    )

    if result and feedback.adjustment != 0.0:
        logger.info(
            f"[SignalEmitter] Curriculum adjusted for {feedback.config_key}: "
            f"policy_acc={feedback.policy_accuracy:.2%} → weight={feedback.new_weight:.2f}"
        )
    return result


def emit_quality_degraded(
    config_key: str,
    quality_score: float,
    threshold: float,
    previous_score: float,
) -> bool:
    """Emit QUALITY_DEGRADED event.

    Args:
        config_key: Configuration key
        quality_score: Current quality score
        threshold: Quality threshold that was crossed
        previous_score: Previous quality score

    Returns:
        True if event was emitted
    """
    payload = {
        "config_key": config_key,
        "quality_score": quality_score,
        "threshold": threshold,
        "previous_score": previous_score,
        "source": "signal_emitter",
    }
    result = _emit_event_sync(
        "QUALITY_DEGRADED",
        payload,
        "signal_emitter",
    )

    if result:
        logger.warning(
            f"[SignalEmitter] Quality degraded for {config_key}: "
            f"{quality_score:.2f} < {threshold:.2f} (prev: {previous_score:.2f})"
        )
    return result


def emit_training_ready(
    config_key: str,
    quality_score: float,
    samples_available: int = 0,
) -> bool:
    """Emit event signaling training data is ready.

    Args:
        config_key: Configuration key
        quality_score: Quality score of the data
        samples_available: Number of samples available

    Returns:
        True if event was emitted

    January 2026: Migrated to safe_emit_event for consistent event handling.
    """
    from app.coordination.event_emission_helpers import safe_emit_event

    return safe_emit_event(
        "DATA_QUALITY_ASSESSED",
        {
            "config": config_key,
            "quality_score": quality_score,
            "samples_available": samples_available,
            "ready_for_training": True,
        },
        context="signal_emitter.emit_training_ready",
    )


# ============================================================================
# Convenience Functions
# ============================================================================


def emit_velocity_based_selfplay_update(
    config_key: str,
    elo_gap: float,
    velocity: float,
    search_budget: int = 150,
    exploration_boost: float = 1.0,
) -> bool:
    """Convenience function to emit selfplay update based on Elo velocity.

    Args:
        config_key: Configuration key
        elo_gap: Gap to target Elo
        velocity: Current Elo velocity (Elo/hour)
        search_budget: Current MCTS search budget
        exploration_boost: Current exploration boost

    Returns:
        True if event was emitted
    """
    adjustment = compute_selfplay_adjustment(
        elo_gap=elo_gap,
        velocity=velocity,
        current_search_budget=search_budget,
        current_exploration_boost=exploration_boost,
    )
    return emit_selfplay_adjustment(config_key, adjustment)


def emit_quality_based_exploration_update(
    config_key: str,
    quality_score: float,
    trend: str = "stable",
) -> bool:
    """Convenience function to emit exploration update based on quality.

    Args:
        config_key: Configuration key
        quality_score: Current quality score (0.0-1.0)
        trend: Quality trend ("improving", "declining", "stable")

    Returns:
        True if event was emitted
    """
    adjustment = compute_exploration_adjustment(quality_score, trend)
    return emit_exploration_adjustment(config_key, adjustment)
