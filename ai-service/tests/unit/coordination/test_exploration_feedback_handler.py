"""Tests for ExplorationFeedbackHandler.

December 2025: Tests for handler extracted from FeedbackLoopController.
"""

import time
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest


@dataclass
class MockFeedbackState:
    """Mock FeedbackState for testing."""

    config_key: str
    current_exploration_boost: float = 1.0
    last_selfplay_quality: float = 0.0
    last_selfplay_time: float = 0.0


@pytest.fixture
def mock_states():
    """Create mock states dictionary."""
    return {}


@pytest.fixture
def handler(mock_states):
    """Create ExplorationFeedbackHandler instance for testing."""
    from app.coordination.exploration_feedback_handler import (
        ExplorationFeedbackHandler,
        reset_exploration_feedback_handler,
    )

    reset_exploration_feedback_handler()

    def get_or_create_state(config_key: str):
        if config_key not in mock_states:
            mock_states[config_key] = MockFeedbackState(config_key=config_key)
        return mock_states[config_key]

    handler = ExplorationFeedbackHandler.get_instance(
        states=mock_states,
        get_or_create_state_fn=get_or_create_state,
    )
    yield handler
    reset_exploration_feedback_handler()


class TestExplorationFeedbackHandlerInit:
    """Tests for handler initialization."""

    def test_singleton_pattern(self, handler):
        """Test that get_instance returns the same instance."""
        from app.coordination.exploration_feedback_handler import (
            ExplorationFeedbackHandler,
        )

        handler2 = ExplorationFeedbackHandler.get_instance()
        assert handler is handler2

    def test_reset_instance(self):
        """Test singleton reset."""
        from app.coordination.exploration_feedback_handler import (
            ExplorationFeedbackHandler,
            reset_exploration_feedback_handler,
        )

        handler1 = ExplorationFeedbackHandler.get_instance()
        reset_exploration_feedback_handler()
        handler2 = ExplorationFeedbackHandler.get_instance()
        assert handler1 is not handler2
        reset_exploration_feedback_handler()

    def test_event_subscriptions(self, handler):
        """Test event subscriptions are correctly defined."""
        subs = handler._get_event_subscriptions()

        assert "LOSS_ANOMALY_DETECTED" in subs
        assert "TRAINING_STALL_DETECTED" in subs
        assert "TRAINING_IMPROVED" in subs
        assert "ELO_PLATEAU_DETECTED" in subs


class TestBoostExplorationForAnomaly:
    """Tests for boost_exploration_for_anomaly."""

    def test_boost_increases_with_anomaly_count(self, handler, mock_states):
        """Test that boost increases with more anomalies."""
        handler.boost_exploration_for_anomaly("hex8_2p", 1)
        boost1 = mock_states["hex8_2p"].current_exploration_boost

        handler.boost_exploration_for_anomaly("square8_2p", 3)
        boost3 = mock_states["square8_2p"].current_exploration_boost

        assert boost3 > boost1

    def test_boost_capped_at_max(self, handler, mock_states):
        """Test that boost is capped at maximum."""
        from app.config.thresholds import EXPLORATION_BOOST_MAX

        handler.boost_exploration_for_anomaly("hex8_2p", 100)  # Large anomaly count
        assert mock_states["hex8_2p"].current_exploration_boost <= EXPLORATION_BOOST_MAX

    def test_tracks_boost_count(self, handler):
        """Test that boost count is tracked."""
        assert handler._exploration_boosts_applied == 0
        handler.boost_exploration_for_anomaly("hex8_2p", 1)
        assert handler._exploration_boosts_applied == 1

    def test_updates_last_boost_time(self, handler):
        """Test that last boost time is updated."""
        before = handler._last_boost_time
        handler.boost_exploration_for_anomaly("hex8_2p", 1)
        assert handler._last_boost_time > before


class TestBoostExplorationForStall:
    """Tests for boost_exploration_for_stall."""

    def test_boost_increases_with_stall_duration(self, handler, mock_states):
        """Test that boost increases with longer stalls."""
        handler.boost_exploration_for_stall("hex8_2p", 5)
        boost5 = mock_states["hex8_2p"].current_exploration_boost

        # Reset state for fair comparison
        mock_states.clear()
        handler.boost_exploration_for_stall("hex8_2p", 20)
        boost20 = mock_states["hex8_2p"].current_exploration_boost

        assert boost20 >= boost5

    def test_boost_capped_at_stall_max(self, handler, mock_states):
        """Test that stall boost is capped at maximum."""
        from app.config.thresholds import EXPLORATION_BOOST_STALL_MAX

        handler.boost_exploration_for_stall("hex8_2p", 1000)  # Very long stall
        assert mock_states["hex8_2p"].current_exploration_boost <= EXPLORATION_BOOST_STALL_MAX

    def test_uses_max_of_current_and_new(self, handler, mock_states):
        """Test that existing boost is preserved if higher."""
        # Set high initial boost
        mock_states["hex8_2p"] = MockFeedbackState(
            config_key="hex8_2p",
            current_exploration_boost=3.0
        )

        handler.boost_exploration_for_stall("hex8_2p", 5)  # Would give ~1.2x

        # Should keep the higher boost
        assert mock_states["hex8_2p"].current_exploration_boost >= 3.0


class TestReduceExplorationAfterImprovement:
    """Tests for reduce_exploration_after_improvement."""

    def test_reduces_boost_towards_one(self, handler, mock_states):
        """Test that boost is reduced towards 1.0."""
        mock_states["hex8_2p"] = MockFeedbackState(
            config_key="hex8_2p",
            current_exploration_boost=2.0
        )

        handler.reduce_exploration_after_improvement("hex8_2p")
        assert mock_states["hex8_2p"].current_exploration_boost < 2.0

    def test_no_reduction_below_one(self, handler, mock_states):
        """Test that boost is not reduced below 1.0."""
        mock_states["hex8_2p"] = MockFeedbackState(
            config_key="hex8_2p",
            current_exploration_boost=1.1
        )

        handler.reduce_exploration_after_improvement("hex8_2p")
        handler.reduce_exploration_after_improvement("hex8_2p")
        handler.reduce_exploration_after_improvement("hex8_2p")

        assert mock_states["hex8_2p"].current_exploration_boost >= 1.0

    def test_no_reduction_when_at_baseline(self, handler, mock_states):
        """Test no reduction when already at baseline."""
        mock_states["hex8_2p"] = MockFeedbackState(
            config_key="hex8_2p",
            current_exploration_boost=1.0
        )

        before = handler._exploration_reductions_applied
        handler.reduce_exploration_after_improvement("hex8_2p")

        # Should not count as reduction when already at 1.0
        assert handler._exploration_reductions_applied == before


class TestEventHandlers:
    """Tests for event handlers."""

    def test_on_loss_anomaly_detected(self, handler, mock_states):
        """Test LOSS_ANOMALY_DETECTED handler."""
        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "anomaly_count": 3,
        }

        handler._on_loss_anomaly_detected(event)

        assert "hex8_2p" in mock_states
        assert mock_states["hex8_2p"].current_exploration_boost > 1.0

    def test_on_training_stall_detected(self, handler, mock_states):
        """Test TRAINING_STALL_DETECTED handler."""
        event = MagicMock()
        event.payload = {
            "config_key": "square8_2p",
            "stall_epochs": 10,
        }

        handler._on_training_stall_detected(event)

        assert "square8_2p" in mock_states
        assert mock_states["square8_2p"].current_exploration_boost >= 1.0

    def test_on_training_improved(self, handler, mock_states):
        """Test TRAINING_IMPROVED handler."""
        # Set up initial boost
        mock_states["hex8_2p"] = MockFeedbackState(
            config_key="hex8_2p",
            current_exploration_boost=2.0
        )

        event = MagicMock()
        event.payload = {"config_key": "hex8_2p"}

        handler._on_training_improved(event)

        assert mock_states["hex8_2p"].current_exploration_boost < 2.0

    def test_on_elo_plateau_detected(self, handler, mock_states):
        """Test ELO_PLATEAU_DETECTED handler."""
        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "duration_hours": 24,
        }

        handler._on_elo_plateau_detected(event)

        assert "hex8_2p" in mock_states
        assert mock_states["hex8_2p"].current_exploration_boost > 1.0

    def test_event_handler_ignores_empty_config(self, handler, mock_states):
        """Test that event handlers ignore empty config_key."""
        event = MagicMock()
        event.payload = {"config_key": "", "anomaly_count": 5}

        before = handler._exploration_boosts_applied
        handler._on_loss_anomaly_detected(event)

        assert handler._exploration_boosts_applied == before


class TestPublicAPI:
    """Tests for public API methods."""

    def test_get_exploration_boost(self, handler, mock_states):
        """Test get_exploration_boost."""
        mock_states["hex8_2p"] = MockFeedbackState(
            config_key="hex8_2p",
            current_exploration_boost=1.5
        )

        boost = handler.get_exploration_boost("hex8_2p")
        assert boost == 1.5

    def test_get_exploration_boost_creates_state(self, handler, mock_states):
        """Test get_exploration_boost creates state if missing."""
        boost = handler.get_exploration_boost("new_config")
        assert boost == 1.0  # Default value
        assert "new_config" in mock_states

    def test_set_exploration_boost(self, handler, mock_states):
        """Test set_exploration_boost."""
        handler.set_exploration_boost("hex8_2p", 1.75)
        assert mock_states["hex8_2p"].current_exploration_boost == 1.75

    def test_set_exploration_boost_capped(self, handler, mock_states):
        """Test set_exploration_boost is capped at max."""
        from app.config.thresholds import EXPLORATION_BOOST_MAX

        handler.set_exploration_boost("hex8_2p", 100.0)
        assert mock_states["hex8_2p"].current_exploration_boost == EXPLORATION_BOOST_MAX

    def test_set_exploration_boost_minimum(self, handler, mock_states):
        """Test set_exploration_boost has minimum of 1.0."""
        handler.set_exploration_boost("hex8_2p", 0.5)
        assert mock_states["hex8_2p"].current_exploration_boost == 1.0


class TestHealthCheck:
    """Tests for health_check method."""

    def test_health_check_returns_result(self, handler):
        """Test health check returns valid result."""
        result = handler.health_check()

        assert result is not None
        assert "exploration_boosts_applied" in result.details
        assert "exploration_reductions_applied" in result.details

    def test_health_check_includes_metrics(self, handler, mock_states):
        """Test health check includes event metrics."""
        # Apply some boosts
        handler.boost_exploration_for_anomaly("hex8_2p", 2)
        handler.boost_exploration_for_stall("square8_2p", 5)

        result = handler.health_check()

        assert result.details["exploration_boosts_applied"] == 2
        assert result.details["states_tracked"] >= 2


class TestRunCycle:
    """Tests for _run_cycle method."""

    @pytest.mark.asyncio
    async def test_run_cycle_updates_time(self, handler):
        """Test run cycle updates last cycle time."""
        before = handler._last_cycle_time
        await handler._run_cycle()
        assert handler._last_cycle_time >= before


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_get_exploration_feedback_handler(self):
        """Test get_exploration_feedback_handler returns handler."""
        from app.coordination.exploration_feedback_handler import (
            get_exploration_feedback_handler,
            reset_exploration_feedback_handler,
        )

        reset_exploration_feedback_handler()
        handler = get_exploration_feedback_handler()
        assert handler is not None
        reset_exploration_feedback_handler()

    def test_exports(self):
        """Test module exports."""
        from app.coordination.exploration_feedback_handler import __all__

        assert "ExplorationFeedbackHandler" in __all__
        assert "get_exploration_feedback_handler" in __all__
        assert "reset_exploration_feedback_handler" in __all__
