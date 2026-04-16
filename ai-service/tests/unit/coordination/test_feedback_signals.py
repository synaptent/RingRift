"""Tests for app.coordination.feedback_signals module.

Tests the feedback signal system that coordinates training loop adjustments:
- SignalType and SignalSource enums
- FeedbackSignal dataclass
- Signal subscription and emission
- Signal history tracking
- Convenience emitters
- FeedbackState management

Created Dec 2025 as part of Phase 3 test coverage improvement.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.feedback_signals import (
    FeedbackSignal,
    FeedbackState,
    SignalSource,
    SignalType,
    emit_exploration_signal,
    emit_intensity_signal,
    emit_promotion_signal,
    emit_quality_signal,
    emit_regression_signal,
    emit_signal,
    get_feedback_state,
    get_latest_signal,
    get_signal_history,
    subscribe_to_all_signals,
    subscribe_to_signal,
    update_feedback_state,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_signal_state():
    """Reset signal state before each test."""
    # Import the module-level state
    from app.coordination import feedback_signals

    # Clear subscribers
    with feedback_signals._signal_lock:
        for st in SignalType:
            feedback_signals._subscribers[st] = []
        feedback_signals._all_subscribers = []
        feedback_signals._signal_history = []

    # Clear feedback states
    with feedback_signals._state_lock:
        feedback_signals._feedback_states = {}

    yield


# =============================================================================
# SignalType and SignalSource Tests
# =============================================================================


class TestSignalType:
    """Tests for SignalType enum."""

    def test_all_signal_types_exist(self):
        """Verify all expected signal types are defined."""
        expected = [
            "INTENSITY",
            "EXPLORATION",
            "CURRICULUM",
            "FRESHNESS",
            "QUALITY",
            "REGRESSION",
            "PROMOTION",
        ]
        for name in expected:
            assert hasattr(SignalType, name), f"Missing SignalType.{name}"

    def test_signal_types_are_unique(self):
        """Verify all signal types have unique values."""
        values = [st.value for st in SignalType]
        assert len(values) == len(set(values)), "Duplicate signal type values"


class TestSignalSource:
    """Tests for SignalSource enum."""

    def test_all_signal_sources_exist(self):
        """Verify all expected signal sources are defined."""
        expected = [
            "SELFPLAY",
            "TRAINING",
            "EVALUATION",
            "GAUNTLET",
            "PROMOTION",
            "SYNC",
            "QUALITY_MONITOR",
            "CURRICULUM",
            "MANUAL",
        ]
        for name in expected:
            assert hasattr(SignalSource, name), f"Missing SignalSource.{name}"


# =============================================================================
# FeedbackSignal Tests
# =============================================================================


class TestFeedbackSignal:
    """Tests for FeedbackSignal dataclass."""

    def test_create_signal_with_required_fields(self):
        """Test creating a signal with only required fields."""
        signal = FeedbackSignal(
            signal_type=SignalType.INTENSITY,
            config_key="hex8_2p",
            value="hot_path",
        )
        assert signal.signal_type == SignalType.INTENSITY
        assert signal.config_key == "hex8_2p"
        assert signal.value == "hot_path"
        assert signal.reason == ""
        assert signal.source == SignalSource.MANUAL
        assert signal.timestamp > 0
        assert signal.metadata == {}

    def test_create_signal_with_all_fields(self):
        """Test creating a signal with all fields."""
        now = time.time()
        signal = FeedbackSignal(
            signal_type=SignalType.QUALITY,
            config_key="square8_4p",
            value=0.85,
            reason="high accuracy",
            source=SignalSource.EVALUATION,
            timestamp=now,
            metadata={"extra": "data"},
        )
        assert signal.value == 0.85
        assert signal.reason == "high accuracy"
        assert signal.source == SignalSource.EVALUATION
        assert signal.timestamp == now
        assert signal.metadata == {"extra": "data"}

    def test_signal_str_representation(self):
        """Test string representation of a signal."""
        signal = FeedbackSignal(
            signal_type=SignalType.REGRESSION,
            config_key="hex8_2p",
            value="detected",
            reason="elo_drop=50.0",
        )
        s = str(signal)
        assert "REGRESSION" in s
        assert "hex8_2p" in s
        assert "detected" in s


# =============================================================================
# Subscription Tests
# =============================================================================


class TestSubscription:
    """Tests for signal subscription."""

    def test_subscribe_to_signal_type(self):
        """Test subscribing to a specific signal type."""
        received = []

        def callback(signal: FeedbackSignal):
            received.append(signal)

        unsubscribe = subscribe_to_signal(SignalType.INTENSITY, callback)

        # Emit matching signal
        emit_signal(FeedbackSignal(
            signal_type=SignalType.INTENSITY,
            config_key="hex8_2p",
            value="hot_path",
        ))

        # Emit non-matching signal
        emit_signal(FeedbackSignal(
            signal_type=SignalType.QUALITY,
            config_key="hex8_2p",
            value=0.9,
        ))

        assert len(received) == 1
        assert received[0].signal_type == SignalType.INTENSITY

        # Test unsubscribe
        unsubscribe()
        emit_signal(FeedbackSignal(
            signal_type=SignalType.INTENSITY,
            config_key="hex8_2p",
            value="cool_down",
        ))
        assert len(received) == 1  # No new signals received

    def test_subscribe_to_all_signals(self):
        """Test subscribing to all signal types."""
        received = []

        def callback(signal: FeedbackSignal):
            received.append(signal)

        unsubscribe = subscribe_to_all_signals(callback)

        # Emit different signal types
        emit_signal(FeedbackSignal(
            signal_type=SignalType.INTENSITY,
            config_key="hex8_2p",
            value="hot_path",
        ))
        emit_signal(FeedbackSignal(
            signal_type=SignalType.QUALITY,
            config_key="square8_2p",
            value=0.8,
        ))

        assert len(received) == 2

        unsubscribe()

    def test_multiple_subscribers(self):
        """Test multiple subscribers to the same signal type."""
        received1 = []
        received2 = []

        def callback1(signal: FeedbackSignal):
            received1.append(signal)

        def callback2(signal: FeedbackSignal):
            received2.append(signal)

        unsubscribe1 = subscribe_to_signal(SignalType.EXPLORATION, callback1)
        unsubscribe2 = subscribe_to_signal(SignalType.EXPLORATION, callback2)

        emit_signal(FeedbackSignal(
            signal_type=SignalType.EXPLORATION,
            config_key="hex8_2p",
            value=1.5,
        ))

        assert len(received1) == 1
        assert len(received2) == 1

        unsubscribe1()
        unsubscribe2()

    def test_subscriber_error_does_not_stop_others(self):
        """Test that an error in one subscriber doesn't affect others."""
        received = []

        def bad_callback(signal: FeedbackSignal):
            raise RuntimeError("Intentional error")

        def good_callback(signal: FeedbackSignal):
            received.append(signal)

        subscribe_to_signal(SignalType.INTENSITY, bad_callback)
        subscribe_to_signal(SignalType.INTENSITY, good_callback)

        # Should not raise, good_callback should still receive
        emit_signal(FeedbackSignal(
            signal_type=SignalType.INTENSITY,
            config_key="hex8_2p",
            value="hot_path",
        ))

        assert len(received) == 1


# =============================================================================
# Signal History Tests
# =============================================================================


class TestSignalHistory:
    """Tests for signal history tracking."""

    def test_signals_stored_in_history(self):
        """Test that emitted signals are stored in history."""
        emit_signal(FeedbackSignal(
            signal_type=SignalType.INTENSITY,
            config_key="hex8_2p",
            value="hot_path",
        ))
        emit_signal(FeedbackSignal(
            signal_type=SignalType.QUALITY,
            config_key="square8_2p",
            value=0.9,
        ))

        history = get_signal_history()
        assert len(history) == 2

    def test_history_ordered_newest_first(self):
        """Test that history returns newest signals first."""
        emit_signal(FeedbackSignal(
            signal_type=SignalType.INTENSITY,
            config_key="first",
            value="1",
        ))
        emit_signal(FeedbackSignal(
            signal_type=SignalType.INTENSITY,
            config_key="second",
            value="2",
        ))

        history = get_signal_history()
        assert history[0].config_key == "second"
        assert history[1].config_key == "first"

    def test_filter_by_signal_type(self):
        """Test filtering history by signal type."""
        emit_signal(FeedbackSignal(
            signal_type=SignalType.INTENSITY,
            config_key="hex8_2p",
            value="hot_path",
        ))
        emit_signal(FeedbackSignal(
            signal_type=SignalType.QUALITY,
            config_key="hex8_2p",
            value=0.9,
        ))

        history = get_signal_history(signal_type=SignalType.INTENSITY)
        assert len(history) == 1
        assert history[0].signal_type == SignalType.INTENSITY

    def test_filter_by_config_key(self):
        """Test filtering history by config key."""
        emit_signal(FeedbackSignal(
            signal_type=SignalType.INTENSITY,
            config_key="hex8_2p",
            value="hot_path",
        ))
        emit_signal(FeedbackSignal(
            signal_type=SignalType.INTENSITY,
            config_key="square8_4p",
            value="cool_down",
        ))

        history = get_signal_history(config_key="hex8_2p")
        assert len(history) == 1
        assert history[0].config_key == "hex8_2p"

    def test_history_limit(self):
        """Test limiting history results."""
        for i in range(10):
            emit_signal(FeedbackSignal(
                signal_type=SignalType.INTENSITY,
                config_key=f"config_{i}",
                value="normal",
            ))

        history = get_signal_history(limit=3)
        assert len(history) == 3

    def test_get_latest_signal(self):
        """Test getting the most recent signal."""
        emit_signal(FeedbackSignal(
            signal_type=SignalType.QUALITY,
            config_key="hex8_2p",
            value=0.5,
        ))
        emit_signal(FeedbackSignal(
            signal_type=SignalType.QUALITY,
            config_key="hex8_2p",
            value=0.9,
        ))

        latest = get_latest_signal(SignalType.QUALITY, "hex8_2p")
        assert latest is not None
        assert latest.value == 0.9

    def test_get_latest_signal_none(self):
        """Test getting latest signal when none exists."""
        latest = get_latest_signal(SignalType.REGRESSION, "nonexistent")
        assert latest is None


# =============================================================================
# Convenience Emitter Tests
# =============================================================================


class TestConvenienceEmitters:
    """Tests for convenience emitter functions."""

    def test_emit_intensity_signal(self):
        """Test emit_intensity_signal."""
        emit_intensity_signal("hex8_2p", "hot_path", reason="low accuracy")

        history = get_signal_history(signal_type=SignalType.INTENSITY)
        assert len(history) == 1
        assert history[0].config_key == "hex8_2p"
        assert history[0].value == "hot_path"
        assert history[0].reason == "low accuracy"
        assert history[0].source == SignalSource.TRAINING

    def test_emit_exploration_signal_clamped(self):
        """Test emit_exploration_signal clamps values."""
        emit_exploration_signal("hex8_2p", 3.0)  # Over max
        emit_exploration_signal("square8_2p", 0.1)  # Under min

        history = get_signal_history(signal_type=SignalType.EXPLORATION)
        assert len(history) == 2
        assert history[0].value == 0.5  # Clamped to min
        assert history[1].value == 2.0  # Clamped to max

    def test_emit_quality_signal_clamped(self):
        """Test emit_quality_signal clamps values."""
        emit_quality_signal("hex8_2p", 1.5)  # Over max
        emit_quality_signal("square8_2p", -0.5)  # Under min

        history = get_signal_history(signal_type=SignalType.QUALITY)
        assert len(history) == 2
        assert history[0].value == 0.0  # Clamped to min
        assert history[1].value == 1.0  # Clamped to max

    def test_emit_regression_signal_detected(self):
        """Test emit_regression_signal for detection."""
        emit_regression_signal("hex8_2p", detected=True, elo_drop=42.5)

        history = get_signal_history(signal_type=SignalType.REGRESSION)
        assert len(history) == 1
        assert history[0].value == "detected"
        assert "42.5" in history[0].reason
        assert history[0].metadata.get("elo_drop") == 42.5

    def test_emit_regression_signal_recovered(self):
        """Test emit_regression_signal for recovery."""
        emit_regression_signal("hex8_2p", detected=False)

        history = get_signal_history(signal_type=SignalType.REGRESSION)
        assert len(history) == 1
        assert history[0].value == "recovered"

    def test_emit_promotion_signal(self):
        """Test emit_promotion_signal."""
        emit_promotion_signal(
            "hex8_2p",
            outcome="promoted",
            model_path="/models/best.pth",
        )

        history = get_signal_history(signal_type=SignalType.PROMOTION)
        assert len(history) == 1
        assert history[0].value == "promoted"
        assert history[0].metadata.get("model_path") == "/models/best.pth"


# =============================================================================
# FeedbackState Tests
# =============================================================================


class TestFeedbackState:
    """Tests for FeedbackState management."""

    def test_get_feedback_state_creates_default(self):
        """Test get_feedback_state creates default state if not exists."""
        state = get_feedback_state("new_config")
        assert state.config_key == "new_config"
        assert state.intensity == "normal"
        assert state.exploration_multiplier == 1.0
        assert state.quality_score == 0.0

    def test_get_feedback_state_returns_same_instance(self):
        """Test get_feedback_state returns the same instance."""
        state1 = get_feedback_state("hex8_2p")
        state2 = get_feedback_state("hex8_2p")
        assert state1 is state2

    def test_update_feedback_state_from_signal(self):
        """Test updating feedback state from a signal."""
        signal = FeedbackSignal(
            signal_type=SignalType.INTENSITY,
            config_key="hex8_2p",
            value="hot_path",
        )
        update_feedback_state(signal)

        state = get_feedback_state("hex8_2p")
        assert state.last_update == signal.timestamp


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety of signal operations."""

    def test_concurrent_emissions(self):
        """Test concurrent signal emissions don't cause issues."""
        received = []
        lock = threading.Lock()

        def callback(signal: FeedbackSignal):
            with lock:
                received.append(signal)

        subscribe_to_all_signals(callback)

        threads = []
        for i in range(10):
            def emit_signals(config_id=i):
                for j in range(10):
                    emit_signal(FeedbackSignal(
                        signal_type=SignalType.INTENSITY,
                        config_key=f"config_{config_id}_{j}",
                        value="normal",
                    ))

            t = threading.Thread(target=emit_signals)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All 100 signals should be received
        assert len(received) == 100


# =============================================================================
# Event Bus Bridge Tests
# =============================================================================


class TestEventBusBridge:
    """Tests for event bus bridging (P0.1 Dec 2025)."""

    @patch("app.coordination.feedback_signals.safe_emit_event", return_value=True)
    @patch("app.coordination.feedback_signals.HAS_EVENT_BUS", True)
    def test_intensity_signal_bridges_to_event_bus(self, mock_emit):
        """Test intensity signals bridge to event bus."""
        from app.coordination.feedback_signals import DataEventType

        emit_signal(FeedbackSignal(
            signal_type=SignalType.INTENSITY,
            config_key="hex8_2p",
            value="hot_path",
        ))

        mock_emit.assert_called_once()
        call_args = mock_emit.call_args
        assert call_args is not None
        assert call_args.args[0] == DataEventType.SELFPLAY_RATE_CHANGED

    @patch("app.coordination.feedback_signals.safe_emit_event", return_value=True)
    @patch("app.coordination.feedback_signals.HAS_EVENT_BUS", True)
    def test_regression_signal_bridges_to_event_bus(self, mock_emit):
        """Test regression signals bridge to event bus."""
        from app.coordination.feedback_signals import DataEventType

        emit_signal(FeedbackSignal(
            signal_type=SignalType.REGRESSION,
            config_key="hex8_2p",
            value="detected",
        ))

        mock_emit.assert_called_once()
        call_args = mock_emit.call_args
        assert call_args is not None
        assert call_args.args[0] == DataEventType.REGRESSION_DETECTED
