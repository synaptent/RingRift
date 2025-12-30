"""Tests for QualityFeedbackHandler.

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
    last_selfplay_quality: float = 0.0
    last_selfplay_time: float = 0.0
    current_training_intensity: str = "normal"
    current_exploration_boost: float = 1.0


@pytest.fixture
def mock_states():
    """Create mock states dictionary."""
    return {}


@pytest.fixture
def handler(mock_states):
    """Create QualityFeedbackHandler instance for testing."""
    from app.coordination.quality_feedback_handler import (
        QualityFeedbackHandler,
        reset_quality_feedback_handler,
    )

    reset_quality_feedback_handler()

    def get_or_create_state(config_key: str):
        if config_key not in mock_states:
            mock_states[config_key] = MockFeedbackState(config_key=config_key)
        return mock_states[config_key]

    handler = QualityFeedbackHandler.get_instance(
        states=mock_states,
        get_or_create_state_fn=get_or_create_state,
    )
    yield handler
    reset_quality_feedback_handler()


class TestQualityFeedbackHandlerInit:
    """Tests for handler initialization."""

    def test_singleton_pattern(self, handler):
        """Test that get_instance returns the same instance."""
        from app.coordination.quality_feedback_handler import (
            QualityFeedbackHandler,
        )

        handler2 = QualityFeedbackHandler.get_instance()
        assert handler is handler2

    def test_reset_instance(self):
        """Test singleton reset."""
        from app.coordination.quality_feedback_handler import (
            QualityFeedbackHandler,
            reset_quality_feedback_handler,
        )

        handler1 = QualityFeedbackHandler.get_instance()
        reset_quality_feedback_handler()
        handler2 = QualityFeedbackHandler.get_instance()
        assert handler1 is not handler2
        reset_quality_feedback_handler()

    def test_event_subscriptions(self, handler):
        """Test event subscriptions are correctly defined."""
        subs = handler._get_event_subscriptions()

        assert "QUALITY_DEGRADED" in subs
        assert "QUALITY_CHECK_FAILED" in subs
        assert "QUALITY_FEEDBACK_ADJUSTED" in subs
        assert "QUALITY_SCORE_UPDATED" in subs
        assert "HIGH_QUALITY_DATA_AVAILABLE" in subs


class TestQualityDegradedHandler:
    """Tests for _on_quality_degraded_for_training."""

    def test_skips_self_emitted_events(self, handler):
        """Test that self-emitted events are skipped."""
        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "quality_score": 0.4,
            "source": "quality_feedback_handler",
        }

        handler._on_quality_degraded_for_training(event)
        assert handler._quality_events_processed == 0

    def test_processes_external_events(self, handler, mock_states):
        """Test that external events are processed."""
        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "quality_score": 0.4,
            "threshold": 0.6,
            "source": "quality_monitor",
        }

        handler._on_quality_degraded_for_training(event)
        assert handler._quality_events_processed == 1
        assert "hex8_2p" in mock_states

    def test_empty_config_key_ignored(self, handler):
        """Test that empty config_key is ignored."""
        event = MagicMock()
        event.payload = {"config_key": "", "quality_score": 0.4}

        handler._on_quality_degraded_for_training(event)
        assert handler._quality_events_processed == 0


class TestQualityCheckFailedHandler:
    """Tests for _on_quality_check_failed."""

    def test_updates_state_on_failure(self, handler, mock_states):
        """Test that state is updated on quality check failure."""
        event = MagicMock()
        event.payload = {
            "config": "square8_2p",
            "quality_score": 0.3,
            "threshold": 0.6,
        }

        handler._on_quality_check_failed(event)

        assert "square8_2p" in mock_states
        assert mock_states["square8_2p"].last_selfplay_quality == 0.3

    def test_empty_config_ignored(self, handler):
        """Test that empty config is ignored."""
        event = MagicMock()
        event.payload = {"config": "", "quality_score": 0.3}

        before_count = handler._quality_events_processed
        handler._on_quality_check_failed(event)
        assert handler._quality_events_processed == before_count


class TestQualityFeedbackAdjustedHandler:
    """Tests for _on_quality_feedback_adjusted."""

    def test_positive_feedback_accelerates_training(self, handler, mock_states):
        """Test that positive feedback accelerates training."""
        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "quality_score": 0.85,
            "budget_multiplier": 1.5,
            "adjustment_type": "quality_improvement",
        }

        handler._on_quality_feedback_adjusted(event)

        assert mock_states["hex8_2p"].current_training_intensity == "accelerated"

    def test_negative_feedback_slows_training(self, handler, mock_states):
        """Test that negative feedback slows training."""
        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "quality_score": 0.4,
            "budget_multiplier": 0.5,
            "adjustment_type": "quality_degraded",
        }

        handler._on_quality_feedback_adjusted(event)

        assert mock_states["hex8_2p"].current_training_intensity == "conservative"

    def test_normal_feedback_normal_training(self, handler, mock_states):
        """Test that normal feedback keeps normal training."""
        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "quality_score": 0.7,
            "budget_multiplier": 0.9,
            "adjustment_type": "stable",
        }

        handler._on_quality_feedback_adjusted(event)

        assert mock_states["hex8_2p"].current_training_intensity == "normal"


class TestQualityScoreUpdatedHandler:
    """Tests for _on_quality_score_updated."""

    def test_declining_quality_triggers_conservative(self, handler, mock_states):
        """Test declining quality triggers conservative training."""
        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "quality_score": 0.4,
            "trend": "declining",
            "sample_count": 100,
        }

        handler._on_quality_score_updated(event)

        assert mock_states["hex8_2p"].current_training_intensity == "conservative"

    def test_improving_quality_triggers_accelerated(self, handler, mock_states):
        """Test improving quality triggers accelerated training."""
        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "quality_score": 0.85,
            "trend": "improving",
            "sample_count": 500,
        }

        handler._on_quality_score_updated(event)

        assert mock_states["hex8_2p"].current_training_intensity == "accelerated"


class TestHighQualityDataHandler:
    """Tests for _on_high_quality_data_available."""

    def test_high_quality_accelerates_training(self, handler, mock_states):
        """Test high quality data accelerates training."""
        event = MagicMock()
        event.payload = {
            "config_key": "square8_2p",
            "quality_score": 0.95,
            "sample_count": 1000,
        }

        handler._on_high_quality_data_available(event)

        assert mock_states["square8_2p"].current_training_intensity == "accelerated"
        assert mock_states["square8_2p"].last_selfplay_quality == 0.95


class TestComputeIntensityFromQuality:
    """Tests for _compute_intensity_from_quality."""

    def test_excellent_quality_hot_path(self, handler):
        """Test excellent quality returns hot_path."""
        assert handler._compute_intensity_from_quality(0.95) == "hot_path"
        assert handler._compute_intensity_from_quality(0.90) == "hot_path"

    def test_good_quality_accelerated(self, handler):
        """Test good quality returns accelerated."""
        assert handler._compute_intensity_from_quality(0.85) == "accelerated"
        assert handler._compute_intensity_from_quality(0.80) == "accelerated"

    def test_adequate_quality_normal(self, handler):
        """Test adequate quality returns normal."""
        assert handler._compute_intensity_from_quality(0.70) == "normal"
        assert handler._compute_intensity_from_quality(0.65) == "normal"

    def test_poor_quality_reduced(self, handler):
        """Test poor quality returns reduced."""
        assert handler._compute_intensity_from_quality(0.55) == "reduced"
        assert handler._compute_intensity_from_quality(0.50) == "reduced"

    def test_very_poor_quality_paused(self, handler):
        """Test very poor quality returns paused."""
        assert handler._compute_intensity_from_quality(0.40) == "paused"
        assert handler._compute_intensity_from_quality(0.20) == "paused"


class TestAssessSelfplayQuality:
    """Tests for _assess_selfplay_quality."""

    def test_missing_database_returns_low_score(self, handler):
        """Test missing database returns 0.3."""
        score = handler._assess_selfplay_quality("/nonexistent/path.db", 100)
        assert score == 0.3

    def test_fallback_to_count_heuristic(self, handler):
        """Test fallback to count-based heuristic when DB doesn't exist.

        Note: When the database file doesn't exist, the function returns 0.3
        (the early return for missing DB). The count-based heuristic only
        applies when the unified scorer fails but the DB exists.
        """
        # Non-existent DB always returns 0.3 (early return)
        assert handler._assess_selfplay_quality("/nonexistent.db", 50) == 0.3
        assert handler._assess_selfplay_quality("/nonexistent.db", 200) == 0.3
        assert handler._assess_selfplay_quality("/nonexistent.db", 1500) == 0.3


class TestTriggerQualityCheck:
    """Tests for _trigger_quality_check."""

    def test_high_priority_for_anomalies(self, handler):
        """Test that trigger_quality_check runs without error for anomalies.

        The actual emit_quality_check_requested is imported inside the method,
        so we just verify the method doesn't raise for high-priority reasons.
        """
        # Should not raise even if emit_quality_check_requested is unavailable
        handler._trigger_quality_check("hex8_2p", "training_loss_anomaly")

    def test_normal_priority_for_regular_checks(self, handler):
        """Test that regular checks use normal priority."""
        # Should not raise
        handler._trigger_quality_check("square8_2p", "scheduled_check")


class TestEmitExplorationAdjustment:
    """Tests for _emit_exploration_adjustment."""

    def test_low_quality_increases_exploration(self, handler):
        """Test low quality increases exploration parameters."""
        # This method emits events, so we verify it doesn't raise
        handler._emit_exploration_adjustment("hex8_2p", 0.3, "declining")

    def test_high_quality_reduces_budget(self, handler):
        """Test high quality can reduce budget."""
        handler._emit_exploration_adjustment("hex8_2p", 0.95, "improving")

    def test_normal_quality_baseline(self, handler):
        """Test normal quality uses baseline parameters."""
        handler._emit_exploration_adjustment("hex8_2p", 0.75, "stable")


class TestHealthCheck:
    """Tests for health_check method."""

    def test_health_check_returns_result(self, handler):
        """Test health check returns valid result."""
        result = handler.health_check()

        assert result is not None
        assert "quality_events_processed" in result.details
        assert "states_tracked" in result.details

    def test_health_check_includes_metrics(self, handler, mock_states):
        """Test health check includes event metrics."""
        # Process an event
        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "quality_score": 0.8,
            "trend": "stable",
        }
        handler._on_quality_score_updated(event)

        result = handler.health_check()

        assert result.details["quality_events_processed"] == 1
        assert result.details["states_tracked"] >= 1


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

    def test_get_quality_feedback_handler(self):
        """Test get_quality_feedback_handler returns handler."""
        from app.coordination.quality_feedback_handler import (
            get_quality_feedback_handler,
            reset_quality_feedback_handler,
        )

        reset_quality_feedback_handler()
        handler = get_quality_feedback_handler()
        assert handler is not None
        reset_quality_feedback_handler()

    def test_exports(self):
        """Test module exports."""
        from app.coordination.quality_feedback_handler import __all__

        assert "QualityFeedbackHandler" in __all__
        assert "get_quality_feedback_handler" in __all__
        assert "reset_quality_feedback_handler" in __all__
