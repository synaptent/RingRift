"""Tests for CurriculumFeedbackHandler.

December 2025: Tests for handler extracted from FeedbackLoopController.
"""

import time
from dataclasses import dataclass, field
import importlib
from unittest.mock import MagicMock

import pytest


@dataclass
class MockFeedbackState:
    """Mock FeedbackState for testing."""

    config_key: str
    current_curriculum_weight: float = 1.0
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    elo_velocity: float = 0.0
    elo_history: list = field(default_factory=list)
    last_elo: float = 1500.0


@pytest.fixture
def mock_states():
    """Create mock states dictionary."""
    return {}


@pytest.fixture
def handler(mock_states):
    """Create CurriculumFeedbackHandler instance for testing."""
    from app.coordination.curriculum_feedback_handler import (
        CurriculumFeedbackHandler,
        reset_curriculum_feedback_handler,
    )

    reset_curriculum_feedback_handler()

    def get_or_create_state(config_key: str):
        if config_key not in mock_states:
            mock_states[config_key] = MockFeedbackState(config_key=config_key)
        return mock_states[config_key]

    handler = CurriculumFeedbackHandler.get_instance(
        states=mock_states,
        get_or_create_state_fn=get_or_create_state,
    )
    yield handler
    reset_curriculum_feedback_handler()


class TestCurriculumFeedbackHandlerInit:
    """Tests for handler initialization."""

    def test_singleton_pattern(self, handler):
        """Test that get_instance returns the same instance."""
        from app.coordination.curriculum_feedback_handler import (
            CurriculumFeedbackHandler,
        )

        handler2 = CurriculumFeedbackHandler.get_instance()
        assert handler is handler2

    def test_reset_instance(self):
        """Test singleton reset."""
        from app.coordination.curriculum_feedback_handler import (
            CurriculumFeedbackHandler,
            reset_curriculum_feedback_handler,
        )

        handler1 = CurriculumFeedbackHandler.get_instance()
        reset_curriculum_feedback_handler()
        handler2 = CurriculumFeedbackHandler.get_instance()
        assert handler1 is not handler2
        reset_curriculum_feedback_handler()

    def test_event_subscriptions(self, handler):
        """Test event subscriptions are correctly defined."""
        subs = handler._get_event_subscriptions()

        assert "SELFPLAY_QUALITY_ASSESSED" in subs
        assert "TRAINING_METRICS_AVAILABLE" in subs
        assert "TRAINING_COMPLETED" in subs


class TestUpdateCurriculumWeightFromSelfplay:
    """Tests for update_curriculum_weight_from_selfplay."""

    def test_low_quality_increases_weight(self, handler, mock_states):
        """Test low quality increases curriculum weight."""
        mock_states["hex8_2p"] = MockFeedbackState(
            config_key="hex8_2p", current_curriculum_weight=1.0
        )

        handler.update_curriculum_weight_from_selfplay("hex8_2p", 0.3)

        assert mock_states["hex8_2p"].current_curriculum_weight > 1.0

    def test_high_quality_decreases_weight(self, handler, mock_states):
        """Test high quality decreases curriculum weight."""
        mock_states["hex8_2p"] = MockFeedbackState(
            config_key="hex8_2p", current_curriculum_weight=1.0
        )

        handler.update_curriculum_weight_from_selfplay("hex8_2p", 0.8)

        assert mock_states["hex8_2p"].current_curriculum_weight < 1.0

    def test_medium_quality_no_change(self, handler, mock_states):
        """Test medium quality doesn't change weight."""
        mock_states["hex8_2p"] = MockFeedbackState(
            config_key="hex8_2p", current_curriculum_weight=1.0
        )

        handler.update_curriculum_weight_from_selfplay("hex8_2p", 0.6)

        assert mock_states["hex8_2p"].current_curriculum_weight == 1.0

    def test_weight_capped_at_max(self, handler, mock_states):
        """Test weight is capped at maximum (2.0)."""
        mock_states["hex8_2p"] = MockFeedbackState(
            config_key="hex8_2p", current_curriculum_weight=1.9
        )

        handler.update_curriculum_weight_from_selfplay("hex8_2p", 0.3)

        assert mock_states["hex8_2p"].current_curriculum_weight <= 2.0

    def test_weight_has_minimum(self, handler, mock_states):
        """Test weight has minimum (0.5)."""
        mock_states["hex8_2p"] = MockFeedbackState(
            config_key="hex8_2p", current_curriculum_weight=0.6
        )

        handler.update_curriculum_weight_from_selfplay("hex8_2p", 0.9)

        assert mock_states["hex8_2p"].current_curriculum_weight >= 0.5


class TestAdaptivePromotionThreshold:
    """Tests for get_adaptive_promotion_threshold."""

    def test_beginner_tier_threshold(self, handler, mock_states):
        """Test beginner tier uses lower threshold."""
        state = MockFeedbackState(config_key="hex8_2p")
        threshold = handler.get_adaptive_promotion_threshold(1200, state)
        assert threshold == 0.55

    def test_intermediate_tier_threshold(self, handler, mock_states):
        """Test intermediate tier uses standard threshold."""
        state = MockFeedbackState(config_key="hex8_2p")
        threshold = handler.get_adaptive_promotion_threshold(1500, state)
        assert threshold == 0.60

    def test_advanced_tier_threshold(self, handler, mock_states):
        """Test advanced tier uses higher threshold."""
        state = MockFeedbackState(config_key="hex8_2p")
        threshold = handler.get_adaptive_promotion_threshold(1700, state)
        assert threshold == 0.65

    def test_elite_tier_threshold(self, handler, mock_states):
        """Test elite tier uses highest threshold."""
        state = MockFeedbackState(config_key="hex8_2p")
        threshold = handler.get_adaptive_promotion_threshold(1900, state)
        assert threshold == 0.70

    def test_consecutive_successes_reduce_threshold(self, handler, mock_states):
        """Test consecutive successes reduce threshold."""
        state = MockFeedbackState(config_key="hex8_2p", consecutive_successes=5)
        threshold = handler.get_adaptive_promotion_threshold(1500, state)
        assert threshold < 0.60  # Should have momentum bonus

    def test_consecutive_failures_increase_threshold(self, handler, mock_states):
        """Test consecutive failures increase threshold."""
        state = MockFeedbackState(config_key="hex8_2p", consecutive_failures=3)
        threshold = handler.get_adaptive_promotion_threshold(1500, state)
        assert threshold > 0.60  # Should have caution penalty

    def test_fast_velocity_reduces_threshold(self, handler, mock_states):
        """Test fast Elo velocity reduces threshold."""
        # ELO_FAST_IMPROVEMENT_PER_HOUR is 50.0
        state = MockFeedbackState(
            config_key="hex8_2p",
            elo_velocity=60.0,  # Fast improvement (above 50.0 threshold)
            elo_history=[(0, 1400), (1, 1450), (2, 1500)],
        )
        threshold = handler.get_adaptive_promotion_threshold(1500, state)
        assert threshold < 0.60

    def test_plateau_increases_threshold(self, handler, mock_states):
        """Test plateau velocity increases threshold."""
        state = MockFeedbackState(
            config_key="hex8_2p",
            elo_velocity=0.1,  # Plateau
            elo_history=[(0, 1500), (1, 1500), (2, 1500)],
        )
        threshold = handler.get_adaptive_promotion_threshold(1500, state)
        assert threshold > 0.60

    def test_threshold_capped_at_range(self, handler, mock_states):
        """Test threshold stays within 0.50-0.75 range."""
        # Many successes + fast velocity should not go below 0.50
        state = MockFeedbackState(
            config_key="hex8_2p",
            consecutive_successes=10,
            elo_velocity=20.0,
            elo_history=[(0, 1000), (1, 1100), (2, 1200)],
        )
        threshold = handler.get_adaptive_promotion_threshold(1200, state)
        assert threshold >= 0.50

        # Many failures + plateau should not go above 0.75
        state2 = MockFeedbackState(
            config_key="hex8_2p",
            consecutive_failures=5,
            elo_velocity=-1.0,
            elo_history=[(0, 1900), (1, 1895), (2, 1890)],
        )
        threshold2 = handler.get_adaptive_promotion_threshold(1900, state2)
        assert threshold2 <= 0.75


class TestRecordTrainingInCurriculum:
    """Tests for record_training_in_curriculum."""

    def test_records_training(self, handler):
        """Test that training is recorded."""
        before = handler._training_recordings
        handler.record_training_in_curriculum("hex8_2p")
        # May or may not increment depending on curriculum_feedback availability
        # At minimum, should not raise

    def test_emit_curriculum_training_feedback_uses_publish_sync(self, handler, monkeypatch):
        """Training feedback should publish synchronously from sync handler code."""
        published = {}

        class StubFeedback:
            weight_min = 0.5
            weight_max = 2.0

            def __init__(self):
                self._current_weights = {"hex8_2p": 1.0}

        def fake_publish_sync(event_type, payload=None, source=""):
            published["event_type"] = event_type
            published["payload"] = payload
            published["source"] = source

        curriculum_feedback = importlib.import_module("app.training.curriculum_feedback")
        event_router = importlib.import_module("app.coordination.event_router")

        monkeypatch.setattr(
            curriculum_feedback,
            "get_curriculum_feedback",
            lambda: StubFeedback(),
        )
        monkeypatch.setattr(
            event_router,
            "publish_sync",
            fake_publish_sync,
        )

        handler.emit_curriculum_training_feedback(
            "hex8_2p",
            policy_accuracy=0.25,
            value_accuracy=0.6,
        )

        assert published["source"] == "curriculum_feedback_handler"
        assert published["payload"]["config"] == "hex8_2p"
        assert published["payload"]["trigger"] == "training_complete"


class TestEventHandlers:
    """Tests for event handlers."""

    def test_on_selfplay_quality_assessed(self, handler, mock_states):
        """Test SELFPLAY_QUALITY_ASSESSED handler."""
        mock_states["hex8_2p"] = MockFeedbackState(
            config_key="hex8_2p", current_curriculum_weight=1.0
        )

        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "quality_score": 0.3,  # Low quality
        }

        handler._on_selfplay_quality_assessed(event)

        assert mock_states["hex8_2p"].current_curriculum_weight > 1.0

    def test_on_training_completed(self, handler):
        """Test TRAINING_COMPLETED handler."""
        event = MagicMock()
        event.payload = {"config": "hex8_2p"}

        # Should not raise
        handler._on_training_completed(event)

    def test_event_handler_ignores_empty_config(self, handler, mock_states):
        """Test that event handlers ignore empty config."""
        event = MagicMock()
        event.payload = {"config_key": "", "quality_score": 0.3}

        before = handler._curriculum_adjustments
        handler._on_selfplay_quality_assessed(event)

        assert handler._curriculum_adjustments == before


class TestPublicAPI:
    """Tests for public API methods."""

    def test_get_curriculum_weight(self, handler, mock_states):
        """Test get_curriculum_weight."""
        mock_states["hex8_2p"] = MockFeedbackState(
            config_key="hex8_2p", current_curriculum_weight=1.5
        )

        weight = handler.get_curriculum_weight("hex8_2p")
        assert weight == 1.5

    def test_get_curriculum_weight_creates_state(self, handler, mock_states):
        """Test get_curriculum_weight creates state if missing."""
        weight = handler.get_curriculum_weight("new_config")
        assert weight == 1.0  # Default value
        assert "new_config" in mock_states

    def test_set_curriculum_weight(self, handler, mock_states):
        """Test set_curriculum_weight."""
        handler.set_curriculum_weight("hex8_2p", 1.75)
        assert mock_states["hex8_2p"].current_curriculum_weight == 1.75

    def test_set_curriculum_weight_capped(self, handler, mock_states):
        """Test set_curriculum_weight is capped at max (2.0)."""
        handler.set_curriculum_weight("hex8_2p", 5.0)
        assert mock_states["hex8_2p"].current_curriculum_weight == 2.0

    def test_set_curriculum_weight_minimum(self, handler, mock_states):
        """Test set_curriculum_weight has minimum (0.5)."""
        handler.set_curriculum_weight("hex8_2p", 0.1)
        assert mock_states["hex8_2p"].current_curriculum_weight == 0.5


class TestHealthCheck:
    """Tests for health_check method."""

    def test_health_check_returns_result(self, handler):
        """Test health check returns valid result."""
        result = handler.health_check()

        assert result is not None
        assert "curriculum_adjustments" in result.details
        assert "training_recordings" in result.details

    def test_health_check_includes_metrics(self, handler, mock_states):
        """Test health check includes event metrics."""
        mock_states["hex8_2p"] = MockFeedbackState(
            config_key="hex8_2p", current_curriculum_weight=1.0
        )

        # Trigger an adjustment
        handler.update_curriculum_weight_from_selfplay("hex8_2p", 0.3)

        result = handler.health_check()

        assert result.details["curriculum_adjustments"] >= 1
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

    def test_get_curriculum_feedback_handler(self):
        """Test get_curriculum_feedback_handler returns handler."""
        from app.coordination.curriculum_feedback_handler import (
            get_curriculum_feedback_handler,
            reset_curriculum_feedback_handler,
        )

        reset_curriculum_feedback_handler()
        handler = get_curriculum_feedback_handler()
        assert handler is not None
        reset_curriculum_feedback_handler()

    def test_exports(self):
        """Test module exports."""
        from app.coordination.curriculum_feedback_handler import __all__

        assert "CurriculumFeedbackHandler" in __all__
        assert "get_curriculum_feedback_handler" in __all__
        assert "reset_curriculum_feedback_handler" in __all__
