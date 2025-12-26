"""Feedback Signal Definitions - Single source of truth for all training feedback signals.

This module defines all feedback signals that flow through the training loop.
Each signal represents a measurable outcome that should trigger adjustments.

Signal Categories:
1. INTENSITY - Training speed/resources (epochs, batch size, learning rate)
2. EXPLORATION - Selfplay diversity (temperature, Dirichlet noise)
3. CURRICULUM - Config prioritization (weights, stages)
4. FRESHNESS - Data staleness (sync triggers)
5. QUALITY - Data quality gates (export triggers)

Usage:
    from app.coordination.feedback_signals import (
        FeedbackSignal,
        SignalType,
        emit_signal,
        subscribe_to_signal,
    )

    # Emit a signal
    emit_signal(FeedbackSignal(
        signal_type=SignalType.INTENSITY,
        config_key="hex8_2p",
        value="hot_path",
        reason="low_accuracy",
    ))

    # Subscribe to signals
    def on_intensity_change(signal: FeedbackSignal):
        print(f"{signal.config_key}: intensity={signal.value}")

    subscribe_to_signal(SignalType.INTENSITY, on_intensity_change)

December 2025: Created for Phase 2 feedback consolidation.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Import event bus for cross-process propagation (P0.1 Dec 2025)
try:
    from app.distributed.data_events import (
        DataEventType,
        emit_event,
    )
    HAS_EVENT_BUS = True
except ImportError:
    HAS_EVENT_BUS = False
    DataEventType = None
    emit_event = None


# =============================================================================
# Signal Types
# =============================================================================


class SignalType(Enum):
    """Categories of feedback signals."""

    # Training intensity control
    INTENSITY = auto()  # Values: "normal", "hot_path", "cool_down"

    # Exploration control
    EXPLORATION = auto()  # Values: float multiplier (0.5-2.0)

    # Curriculum prioritization
    CURRICULUM = auto()  # Values: dict of config_key -> weight

    # Data freshness
    FRESHNESS = auto()  # Values: "fresh", "stale", "critical"

    # Quality gates
    QUALITY = auto()  # Values: float score (0.0-1.0)

    # Regression alerts
    REGRESSION = auto()  # Values: "detected", "recovered"

    # Promotion outcomes
    PROMOTION = auto()  # Values: "promoted", "rejected", "pending"


class SignalSource(Enum):
    """Origin of a feedback signal."""

    SELFPLAY = auto()
    TRAINING = auto()
    EVALUATION = auto()
    GAUNTLET = auto()
    PROMOTION = auto()
    SYNC = auto()
    QUALITY_MONITOR = auto()
    CURRICULUM = auto()
    MANUAL = auto()


# =============================================================================
# Signal to Event Mapping (P0.1 Dec 2025)
# =============================================================================

def _get_event_type_for_signal(signal: "FeedbackSignal") -> "DataEventType | None":
    """Map a FeedbackSignal to the appropriate DataEventType.

    This bridges the in-process FeedbackSignal system to the cross-process
    DataEvent system, enabling cluster-wide feedback propagation.
    """
    if not HAS_EVENT_BUS or DataEventType is None:
        return None

    signal_type = signal.signal_type
    value = signal.value

    # Map signal types to data events
    if signal_type == SignalType.INTENSITY:
        # Intensity changes affect selfplay rate
        if value in ("hot_path", "cool_down"):
            return DataEventType.SELFPLAY_RATE_CHANGED
        return None

    elif signal_type == SignalType.QUALITY:
        # Quality degradation
        if isinstance(value, (int, float)) and value < 0.5:
            return DataEventType.QUALITY_DEGRADED
        return None

    elif signal_type == SignalType.REGRESSION:
        if value == "detected":
            return DataEventType.REGRESSION_DETECTED
        elif value == "recovered":
            return DataEventType.REGRESSION_CLEARED
        return None

    elif signal_type == SignalType.PROMOTION:
        if value == "promoted":
            return DataEventType.MODEL_PROMOTED
        elif value == "rejected":
            return DataEventType.PROMOTION_FAILED
        return None

    elif signal_type == SignalType.CURRICULUM:
        return DataEventType.CURRICULUM_REBALANCED

    elif signal_type == SignalType.FRESHNESS:
        if value == "stale" or value == "critical":
            return DataEventType.QUALITY_DEGRADED  # Freshness issues = quality concern
        return None

    elif signal_type == SignalType.EXPLORATION:
        # Dec 2025: Route exploration adjustments to DataEventBus
        # This enables cross-process coordination of exploration strategies
        return DataEventType.EXPLORATION_ADJUSTED

    return None


# =============================================================================
# Signal Definition
# =============================================================================


@dataclass
class FeedbackSignal:
    """A feedback signal carrying information about training state changes."""

    signal_type: SignalType
    config_key: str
    value: Any
    reason: str = ""
    source: SignalSource = SignalSource.MANUAL
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"Signal({self.signal_type.name}, {self.config_key}, "
            f"value={self.value}, reason={self.reason})"
        )


# =============================================================================
# Signal Bus (In-Process)
# =============================================================================

_subscribers: dict[SignalType, list[Callable[[FeedbackSignal], None]]] = {
    st: [] for st in SignalType
}
_all_subscribers: list[Callable[[FeedbackSignal], None]] = []
_signal_lock = threading.Lock()
_signal_history: list[FeedbackSignal] = []
_history_max_size = 1000


def subscribe_to_signal(
    signal_type: SignalType,
    callback: Callable[[FeedbackSignal], None],
) -> Callable[[], None]:
    """Subscribe to a specific signal type.

    Args:
        signal_type: Type of signal to subscribe to
        callback: Function to call when signal is emitted

    Returns:
        Unsubscribe function
    """
    with _signal_lock:
        _subscribers[signal_type].append(callback)

    def unsubscribe():
        with _signal_lock:
            if callback in _subscribers[signal_type]:
                _subscribers[signal_type].remove(callback)

    return unsubscribe


def subscribe_to_all_signals(
    callback: Callable[[FeedbackSignal], None],
) -> Callable[[], None]:
    """Subscribe to all signal types.

    Args:
        callback: Function to call when any signal is emitted

    Returns:
        Unsubscribe function
    """
    with _signal_lock:
        _all_subscribers.append(callback)

    def unsubscribe():
        with _signal_lock:
            if callback in _all_subscribers:
                _all_subscribers.remove(callback)

    return unsubscribe


def emit_signal(signal: FeedbackSignal) -> None:
    """Emit a feedback signal to all subscribers.

    This function emits to both:
    1. In-process subscribers (immediate callbacks)
    2. Cross-process DataEventBus (cluster-wide propagation)

    P0.1 Dec 2025: Added event bus bridging for feedback loop closure.

    Args:
        signal: The signal to emit
    """
    with _signal_lock:
        # Store in history
        _signal_history.append(signal)
        if len(_signal_history) > _history_max_size:
            _signal_history.pop(0)

        # Get subscribers
        type_subs = list(_subscribers[signal.signal_type])
        all_subs = list(_all_subscribers)

    # Call subscribers outside lock
    for callback in type_subs + all_subs:
        try:
            callback(signal)
        except Exception as e:
            logger.error(f"[FeedbackSignals] Subscriber error: {e}")

    # Bridge to event bus for cross-process propagation (P0.1 Dec 2025)
    if HAS_EVENT_BUS and emit_event is not None:
        event_type = _get_event_type_for_signal(signal)
        if event_type is not None:
            try:
                # Build payload from signal
                payload = {
                    "config_key": signal.config_key,
                    "value": signal.value,
                    "reason": signal.reason,
                    "source": signal.source.name if signal.source else "UNKNOWN",
                    "signal_type": signal.signal_type.name,
                    **signal.metadata,
                }
                emit_event(event_type, payload)
                logger.debug(
                    f"[FeedbackSignals] Bridged to event bus: "
                    f"{signal.signal_type.name} -> {event_type.name}"
                )
            except Exception as e:
                logger.warning(f"[FeedbackSignals] Event bus bridge error: {e}")

    logger.debug(f"[FeedbackSignals] Emitted: {signal}")


def get_signal_history(
    signal_type: SignalType | None = None,
    config_key: str | None = None,
    limit: int = 100,
) -> list[FeedbackSignal]:
    """Get recent signal history.

    Args:
        signal_type: Filter by signal type (optional)
        config_key: Filter by config key (optional)
        limit: Maximum number of signals to return

    Returns:
        List of recent signals (newest first)
    """
    with _signal_lock:
        signals = list(reversed(_signal_history))

    if signal_type:
        signals = [s for s in signals if s.signal_type == signal_type]
    if config_key:
        signals = [s for s in signals if s.config_key == config_key]

    return signals[:limit]


def get_latest_signal(
    signal_type: SignalType,
    config_key: str,
) -> FeedbackSignal | None:
    """Get the most recent signal for a type/config combination.

    Args:
        signal_type: Type of signal
        config_key: Config key to filter by

    Returns:
        Most recent signal or None
    """
    history = get_signal_history(signal_type=signal_type, config_key=config_key, limit=1)
    return history[0] if history else None


# =============================================================================
# Convenience Emitters
# =============================================================================


def emit_intensity_signal(
    config_key: str,
    intensity: str,  # "normal", "hot_path", "cool_down"
    reason: str = "",
    source: SignalSource = SignalSource.TRAINING,
) -> None:
    """Emit a training intensity signal."""
    emit_signal(FeedbackSignal(
        signal_type=SignalType.INTENSITY,
        config_key=config_key,
        value=intensity,
        reason=reason,
        source=source,
    ))


def emit_exploration_signal(
    config_key: str,
    multiplier: float,  # 0.5-2.0, 1.0 = normal
    reason: str = "",
    source: SignalSource = SignalSource.EVALUATION,
) -> None:
    """Emit an exploration adjustment signal."""
    emit_signal(FeedbackSignal(
        signal_type=SignalType.EXPLORATION,
        config_key=config_key,
        value=max(0.5, min(2.0, multiplier)),  # Clamp to valid range
        reason=reason,
        source=source,
    ))


def emit_curriculum_signal(
    weights: dict[str, float],
    reason: str = "",
    source: SignalSource = SignalSource.CURRICULUM,
) -> None:
    """Emit a curriculum weight update signal."""
    emit_signal(FeedbackSignal(
        signal_type=SignalType.CURRICULUM,
        config_key="all",
        value=weights,
        reason=reason,
        source=source,
    ))


def emit_quality_signal(
    config_key: str,
    score: float,  # 0.0-1.0
    reason: str = "",
    source: SignalSource = SignalSource.QUALITY_MONITOR,
) -> None:
    """Emit a quality score signal."""
    emit_signal(FeedbackSignal(
        signal_type=SignalType.QUALITY,
        config_key=config_key,
        value=max(0.0, min(1.0, score)),
        reason=reason,
        source=source,
    ))


def emit_regression_signal(
    config_key: str,
    detected: bool,
    elo_drop: float = 0.0,
    source: SignalSource = SignalSource.EVALUATION,
) -> None:
    """Emit a regression detection signal."""
    emit_signal(FeedbackSignal(
        signal_type=SignalType.REGRESSION,
        config_key=config_key,
        value="detected" if detected else "recovered",
        reason=f"elo_drop={elo_drop:.1f}" if detected else "recovered",
        source=source,
        metadata={"elo_drop": elo_drop},
    ))


def emit_promotion_signal(
    config_key: str,
    outcome: str,  # "promoted", "rejected", "pending"
    model_path: str = "",
    source: SignalSource = SignalSource.PROMOTION,
) -> None:
    """Emit a promotion outcome signal."""
    emit_signal(FeedbackSignal(
        signal_type=SignalType.PROMOTION,
        config_key=config_key,
        value=outcome,
        reason=f"model={model_path}" if model_path else "",
        source=source,
        metadata={"model_path": model_path},
    ))


# =============================================================================
# Signal State Snapshot
# =============================================================================


@dataclass
class FeedbackState:
    """Current state of all feedback signals for a config."""

    config_key: str
    intensity: str = "normal"
    exploration_multiplier: float = 1.0
    curriculum_weight: float = 1.0
    quality_score: float = 0.0
    freshness: str = "unknown"
    regression_detected: bool = False
    last_promotion: str = "unknown"
    last_update: float = field(default_factory=time.time)


_feedback_states: dict[str, FeedbackState] = {}
_state_lock = threading.Lock()


def get_feedback_state(config_key: str) -> FeedbackState:
    """Get current feedback state for a config.

    Args:
        config_key: Config key (e.g., "hex8_2p")

    Returns:
        Current feedback state
    """
    with _state_lock:
        if config_key not in _feedback_states:
            _feedback_states[config_key] = FeedbackState(config_key=config_key)
        return _feedback_states[config_key]


def update_feedback_state(signal: FeedbackSignal) -> None:
    """Update feedback state from a signal.

    This is called automatically when signals are emitted if
    auto_update_state is enabled.
    """
    with _state_lock:
        if signal.config_key not in _feedback_states:
            _feedback_states[signal.config_key] = FeedbackState(
                config_key=signal.config_key
            )
        state = _feedback_states[signal.config_key]
        state.last_update = signal.timestamp

        if signal.signal_type == SignalType.INTENSITY:
            state.intensity = signal.value
        elif signal.signal_type == SignalType.EXPLORATION:
            state.exploration_multiplier = signal.value
        elif signal.signal_type == SignalType.CURRICULUM:
            if isinstance(signal.value, dict):
                state.curriculum_weight = signal.value.get(signal.config_key, 1.0)
        elif signal.signal_type == SignalType.QUALITY:
            state.quality_score = signal.value
        elif signal.signal_type == SignalType.FRESHNESS:
            state.freshness = signal.value
        elif signal.signal_type == SignalType.REGRESSION:
            state.regression_detected = signal.value == "detected"
        elif signal.signal_type == SignalType.PROMOTION:
            state.last_promotion = signal.value


def get_all_feedback_states() -> dict[str, FeedbackState]:
    """Get current feedback state for all configs."""
    with _state_lock:
        return dict(_feedback_states)


# Auto-update state when signals are emitted
subscribe_to_all_signals(update_feedback_state)
