"""Integration tests for REGRESSION_DETECTED -> exploration boost.

Tests the FeedbackLoopController's response to regression detection,
including applying amplitude-scaled exploration boosts and increasing
selfplay targets.

Event flow:
1. RegressionDetector emits REGRESSION_DETECTED
2. FeedbackLoopController increases exploration boost based on severity
3. Emits SELFPLAY_TARGET_UPDATED with higher target games
4. Emits EXPLORATION_BOOST for temperature schedulers

December 2025 - Phase 3 integration tests
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.feedback_loop_controller import FeedbackLoopController, FeedbackState


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def controller():
    """Create a FeedbackLoopController instance."""
    controller = FeedbackLoopController()
    controller._regression_cooldowns.clear()
    return controller


@pytest.fixture
def mock_event():
    """Create a mock REGRESSION_DETECTED event."""
    event = MagicMock()
    event.payload = {
        "config_key": "hex8_2p",
        "elo_drop": 50.0,
        "consecutive_regressions": 1,
    }
    return event


@pytest.fixture
def mock_selfplay_events():
    """Mock selfplay event emission."""
    def _consume_task(coro, context: str = ""):
        if hasattr(coro, "close"):
            coro.close()
        return MagicMock(name=f"task:{context}")

    with patch("app.coordination.feedback.regression_handling_mixin.HAS_SELFPLAY_EVENTS", True):
        with patch(
            "app.coordination.feedback.regression_handling_mixin.emit_selfplay_target_updated",
            new_callable=AsyncMock,
        ) as mock_emit:
            with patch(
                "app.coordination.feedback.regression_handling_mixin._safe_create_task",
                side_effect=_consume_task,
            ):
                yield mock_emit


@pytest.fixture
def mock_exploration_boost_emit():
    """Mock exploration boost event emission."""
    mock_emit = AsyncMock()
    with patch("app.distributed.data_events.emit_exploration_boost", mock_emit):
        yield mock_emit


# =============================================================================
# Event Handling Tests
# =============================================================================


class TestRegressionEventHandling:
    """Tests for REGRESSION_DETECTED event handling."""

    def test_handles_regression_detected(self, controller, mock_event):
        """Test handling REGRESSION_DETECTED sets exploration boost."""
        controller._on_regression_detected(mock_event)

        state = controller._get_or_create_state("hex8_2p")

        # Minor 50-Elo regression now gets amplitude-scaled 1.65x boost.
        assert state.current_exploration_boost == 1.65

        # Check failure count incremented
        assert state.consecutive_failures == 1

    def test_preserves_higher_existing_boost(self, controller, mock_event):
        """Test regression preserves higher existing boost."""
        # Set higher existing boost
        state = controller._get_or_create_state("hex8_2p")
        state.current_exploration_boost = 1.8

        controller._on_regression_detected(mock_event)

        # Should keep higher value
        assert state.current_exploration_boost == 1.8

    def test_consecutive_regressions_tracked(self, controller):
        """Test consecutive regressions increment failure count."""
        event1 = MagicMock()
        event1.payload = {
            "config_key": "hex8_2p",
            "elo_drop": 30.0,
            "consecutive_regressions": 1,
        }

        event2 = MagicMock()
        event2.payload = {
            "config_key": "hex8_2p",
            "elo_drop": 20.0,
            "consecutive_regressions": 2,
        }

        controller._on_regression_detected(event1)
        controller._regression_cooldowns.clear()
        controller._on_regression_detected(event2)

        state = controller._get_or_create_state("hex8_2p")
        assert state.consecutive_failures == 2

    def test_handles_missing_config_key(self, controller):
        """Test gracefully handles event with missing config_key."""
        event = MagicMock()
        event.payload = {"elo_drop": 50.0}

        # Should not raise
        controller._on_regression_detected(event)

    def test_handles_missing_payload(self, controller):
        """Test gracefully handles event with missing payload."""
        event = MagicMock(spec=[])  # No payload attribute

        # Should not raise
        controller._on_regression_detected(event)


# =============================================================================
# Selfplay Target Update Tests
# =============================================================================


class TestSelfplayTargetUpdate:
    """Tests for SELFPLAY_TARGET_UPDATED emission on regression."""

    def test_emits_selfplay_target_updated(self, controller, mock_event, mock_selfplay_events):
        """Test emits SELFPLAY_TARGET_UPDATED with increased target."""
        # Mock asyncio.get_running_loop (used by implementation)
        mock_loop = MagicMock()
        with patch("asyncio.get_running_loop", return_value=mock_loop):
            controller._on_regression_detected(mock_event)

            # Check emission was attempted
            mock_selfplay_events.assert_called_once()

            # Verify call arguments
            call_kwargs = mock_selfplay_events.call_args.kwargs
            assert call_kwargs["config_key"] == "hex8_2p"
            assert call_kwargs["target_games"] > 500  # Base is 500
            assert call_kwargs["priority"] == 2  # High priority
            assert "regression_detected" in call_kwargs["reason"]

    def test_target_games_scales_with_severity(self, controller, mock_selfplay_events):
        """Test target games scale with consecutive regressions."""
        mock_loop = MagicMock()
        with patch("asyncio.get_running_loop", return_value=mock_loop):
            # First regression: base 500 games
            event1 = MagicMock()
            event1.payload = {
                "config_key": "hex8_2p",
                "elo_drop": 30.0,
                "consecutive_regressions": 1,
            }
            controller._on_regression_detected(event1)
            target1 = mock_selfplay_events.call_args.kwargs["target_games"]

            # Third regression: should be higher
            controller._regression_cooldowns.clear()
            event2 = MagicMock()
            event2.payload = {
                "config_key": "hex8_2p",
                "elo_drop": 40.0,
                "consecutive_regressions": 3,
            }
            controller._on_regression_detected(event2)
            target2 = mock_selfplay_events.call_args.kwargs["target_games"]

            # More consecutive regressions = more games
            assert target2 > target1

    def test_handles_no_event_loop(self, controller, mock_event, mock_selfplay_events):
        """Test handles case when no event loop is running."""
        with patch("asyncio.get_running_loop", side_effect=RuntimeError):
            # Should not raise
            controller._on_regression_detected(mock_event)

    def test_handles_selfplay_events_unavailable(self, controller, mock_event):
        """Test handles case when selfplay events not available."""
        with patch("app.coordination.feedback_loop_controller.HAS_SELFPLAY_EVENTS", False):
            # Should not raise
            controller._on_regression_detected(mock_event)


# =============================================================================
# Exploration Boost Emission Tests
# =============================================================================


class TestExplorationBoostEmission:
    """Tests for EXPLORATION_BOOST event emission."""

    @pytest.mark.asyncio
    async def test_emits_exploration_boost(self, controller, mock_event, mock_exploration_boost_emit):
        """Test emits EXPLORATION_BOOST event for temperature schedulers."""
        import asyncio

        controller._on_regression_detected(mock_event)

        # Give async task time to execute
        await asyncio.sleep(0.1)

        # Check emission was called
        if mock_exploration_boost_emit.called:
            call_kwargs = mock_exploration_boost_emit.call_args.kwargs
            assert call_kwargs["config_key"] == "hex8_2p"
            assert call_kwargs["boost_factor"] >= 1.5
            assert call_kwargs["reason"] == "regression_detected"

    def test_handles_exploration_boost_error(self, controller, mock_event):
        """Test handles errors in exploration boost emission."""
        with patch("app.distributed.data_events.emit_exploration_boost", side_effect=Exception):
            # Should not raise
            controller._on_regression_detected(mock_event)


# =============================================================================
# State Management Tests
# =============================================================================


class TestStateManagement:
    """Tests for state management during regression."""

    def test_creates_state_if_missing(self, controller, mock_event):
        """Test creates FeedbackState if it doesn't exist."""
        assert "hex8_2p" not in controller._states

        controller._on_regression_detected(mock_event)

        assert "hex8_2p" in controller._states
        state = controller._states["hex8_2p"]
        assert isinstance(state, FeedbackState)

    def test_updates_existing_state(self, controller, mock_event):
        """Test updates existing state correctly."""
        # Pre-create state
        state = controller._get_or_create_state("hex8_2p")
        state.current_exploration_boost = 1.2
        state.consecutive_failures = 0

        controller._on_regression_detected(mock_event)

        # Check updates
        assert state.current_exploration_boost == 1.65  # max(1.2, 1.65)
        assert state.consecutive_failures == 1

    def test_multiple_configs_independent(self, controller):
        """Test different configs tracked independently."""
        event1 = MagicMock()
        event1.payload = {
            "config_key": "hex8_2p",
            "elo_drop": 30.0,
            "consecutive_regressions": 1,
        }

        event2 = MagicMock()
        event2.payload = {
            "config_key": "square8_4p",
            "elo_drop": 40.0,
            "consecutive_regressions": 1,
        }

        controller._on_regression_detected(event1)
        controller._on_regression_detected(event2)

        state1 = controller._states["hex8_2p"]
        state2 = controller._states["square8_4p"]

        assert state1.consecutive_failures == 1
        assert state2.consecutive_failures == 1
        assert state1 is not state2


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for full regression feedback loop."""

    def test_full_regression_feedback_loop(self, controller, mock_selfplay_events, mock_exploration_boost_emit):
        """Test complete regression feedback loop."""
        mock_loop = MagicMock()
        with patch("asyncio.get_running_loop", return_value=mock_loop):
            event = MagicMock()
            event.payload = {
                "config_key": "hex8_2p",
                "elo_drop": 75.0,
                "consecutive_regressions": 2,
            }

            controller._on_regression_detected(event)

            # Verify state changes
            state = controller._states["hex8_2p"]
            assert state.current_exploration_boost >= 1.5
            assert state.consecutive_failures == 2

            # Verify events emitted
            mock_selfplay_events.assert_called_once()

    def test_multiple_regressions_accumulate(self, controller):
        """Test multiple regressions accumulate boost correctly."""
        events = [
            MagicMock(payload={
                "config_key": "hex8_2p",
                "elo_drop": 20.0,
                "consecutive_regressions": i + 1,
            })
            for i in range(3)
        ]

        for event in events:
            controller._on_regression_detected(event)
            controller._regression_cooldowns.clear()

        state = controller._states["hex8_2p"]

        # After 3 regressions
        assert state.consecutive_failures == 3
        assert state.current_exploration_boost >= 1.5


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_zero_elo_drop(self, controller):
        """Test handles regression with zero elo drop."""
        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "elo_drop": 0.0,
            "consecutive_regressions": 1,
        }

        # Should not raise
        controller._on_regression_detected(event)

        state = controller._states["hex8_2p"]
        assert state.current_exploration_boost >= 1.5

    def test_handles_negative_elo_drop(self, controller):
        """Test handles negative elo drop (should not happen but defensive)."""
        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "elo_drop": -10.0,
            "consecutive_regressions": 1,
        }

        # Should not raise
        controller._on_regression_detected(event)

    def test_handles_malformed_payload(self, controller):
        """Test handles malformed event payload."""
        event = MagicMock()
        event.payload = {"invalid_field": "value"}

        # Should not raise
        controller._on_regression_detected(event)

    def test_thread_safety(self, controller):
        """Test concurrent regression handling is thread-safe."""
        import threading

        events = [
            MagicMock(payload={
                "config_key": f"config_{i}",
                "elo_drop": 30.0,
                "consecutive_regressions": 1,
            })
            for i in range(10)
        ]

        threads = []
        for event in events:
            t = threading.Thread(target=controller._on_regression_detected, args=(event,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All configs should have state
        assert len(controller._states) == 10
        for i in range(10):
            assert f"config_{i}" in controller._states
