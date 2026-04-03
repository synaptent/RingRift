"""Integration tests for PROMOTION_FAILED → curriculum weight increase.

Tests the PromotionFailedToCurriculumWatcher class that increases curriculum
weights when model promotion fails, ensuring more diverse training data is
generated for the next training cycle.

Event flow:
1. Promotion fails (emits PROMOTION_FAILED)
2. PromotionFailedToCurriculumWatcher increases curriculum weight by 20%
3. Weight caps at 2.5x after multiple consecutive failures
4. Emits CURRICULUM_REBALANCED to notify downstream systems

December 2025 - Phase 3 integration tests
"""

from unittest.mock import MagicMock, patch

import pytest

from app.coordination.curriculum_integration import PromotionFailedToCurriculumWatcher


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def watcher():
    """Create a PromotionFailedToCurriculumWatcher instance."""
    return PromotionFailedToCurriculumWatcher()


@pytest.fixture
def mock_event_router():
    """Mock event router for subscription testing."""
    router = MagicMock()
    router.subscribe = MagicMock()
    router.unsubscribe = MagicMock()
    router.publish_sync = MagicMock()
    return router


@pytest.fixture
def mock_curriculum_feedback():
    """Mock curriculum feedback for weight updates."""
    feedback = MagicMock()
    feedback._current_weights = {}
    feedback.weight_min = 0.5
    feedback.weight_max = 2.5
    return feedback


@pytest.fixture
def mock_event():
    """Create a mock PROMOTION_FAILED event."""
    event = MagicMock()
    event.payload = {
        "config_key": "hex8_2p",
        "error": "gauntlet_failed",
        "model_id": "hex8_2p_v123",
    }
    return event


# =============================================================================
# Subscription Tests
# =============================================================================


class TestSubscription:
    """Tests for event subscription."""

    def test_subscribe_success(self, watcher):
        """Test successful subscription to PROMOTION_FAILED."""
        with patch("app.coordination.event_router.subscribe") as mock_subscribe:
            result = watcher.subscribe()

            assert result is True
            assert watcher.is_subscribed is True
            mock_subscribe.assert_called_once()

            event_type, handler = mock_subscribe.call_args.args
            event_name = getattr(event_type, "name", str(event_type))
            assert event_name == "PROMOTION_FAILED"
            assert handler == watcher._handle_event

    def test_subscribe_already_subscribed(self, watcher):
        """Test subscribing when already subscribed returns True."""
        watcher._subscribed = True
        result = watcher.subscribe()

        assert result is True

    def test_subscribe_router_not_available(self, watcher):
        """Test subscription fails gracefully when router not available."""
        with patch("app.coordination.event_router.subscribe", side_effect=RuntimeError("router unavailable")):
            result = watcher.subscribe()

            assert result is False
            assert watcher.is_subscribed is False

    def test_subscribe_import_error(self, watcher):
        """Test subscription handles import errors."""
        with patch("app.coordination.event_router.subscribe", side_effect=ImportError):
            result = watcher.subscribe()

            assert result is False
            assert watcher.is_subscribed is False

    def test_unsubscribe_success(self, watcher):
        """Test successful unsubscription."""
        watcher._subscribed = True

        with patch("app.coordination.event_router.unsubscribe") as mock_unsubscribe:
            watcher.unsubscribe()

            assert watcher.is_subscribed is False
            mock_unsubscribe.assert_called_once()


# =============================================================================
# Event Handling Tests
# =============================================================================


class TestEventHandling:
    """Tests for PROMOTION_FAILED event handling."""

    def test_handles_promotion_failed_first_failure(self, watcher, mock_event, mock_curriculum_feedback):
        """Test handling first promotion failure increases weight by 20%."""
        with patch("app.training.curriculum_feedback.get_curriculum_feedback", return_value=mock_curriculum_feedback):
            mock_curriculum_feedback._current_weights["hex8_2p"] = 1.0

            watcher._handle_event(mock_event)

            assert "hex8_2p" in mock_curriculum_feedback._current_weights
            assert mock_curriculum_feedback._current_weights["hex8_2p"] == 1.2
            assert watcher.get_failure_counts()["hex8_2p"] == 1

    def test_handles_consecutive_failures(self, watcher, mock_event, mock_curriculum_feedback):
        """Test consecutive failures increase weight based on total failure count."""
        with patch("app.training.curriculum_feedback.get_curriculum_feedback", return_value=mock_curriculum_feedback):
            mock_curriculum_feedback._current_weights = {"hex8_2p": 1.0}
            mock_curriculum_feedback.weight_min = 0.5
            mock_curriculum_feedback.weight_max = 2.5

            watcher._handle_event(mock_event)
            weight1 = mock_curriculum_feedback._current_weights["hex8_2p"]
            assert abs(weight1 - 1.2) < 0.01

            watcher._handle_event(mock_event)
            weight2 = mock_curriculum_feedback._current_weights["hex8_2p"]
            assert abs(weight2 - 1.68) < 0.01

            watcher._handle_event(mock_event)
            weight3 = mock_curriculum_feedback._current_weights["hex8_2p"]
            assert abs(weight3 - 2.5) < 0.01
            assert watcher.get_failure_counts()["hex8_2p"] == 3

    def test_weight_caps_at_max(self, watcher, mock_event, mock_curriculum_feedback):
        """Test weight increase caps at 2.5x."""
        with patch("app.training.curriculum_feedback.get_curriculum_feedback", return_value=mock_curriculum_feedback):
            mock_curriculum_feedback._current_weights["hex8_2p"] = 2.3

            watcher._handle_event(mock_event)
            assert mock_curriculum_feedback._current_weights["hex8_2p"] == 2.5

            watcher._handle_event(mock_event)
            assert mock_curriculum_feedback._current_weights["hex8_2p"] == 2.5

    def test_handles_missing_config_key(self, watcher):
        """Test gracefully handles event with missing config_key."""
        event = MagicMock()
        event.payload = {"error": "test"}

        watcher._handle_event(event)

        assert watcher.get_failure_counts() == {}

    def test_tracks_failure_count_per_config(self, watcher, mock_curriculum_feedback):
        """Test failure counts tracked separately per config."""
        with patch("app.training.curriculum_feedback.get_curriculum_feedback", return_value=mock_curriculum_feedback):
            mock_curriculum_feedback._current_weights["hex8_2p"] = 1.0
            mock_curriculum_feedback._current_weights["square8_4p"] = 1.0

            event1 = MagicMock()
            event1.payload = {"config_key": "hex8_2p", "error": "test"}
            watcher._handle_event(event1)
            watcher._handle_event(event1)

            event2 = MagicMock()
            event2.payload = {"config_key": "square8_4p", "error": "test"}
            watcher._handle_event(event2)

            assert watcher.get_failure_counts() == {"hex8_2p": 2, "square8_4p": 1}


# =============================================================================
# Curriculum Weight Update Tests
# =============================================================================


class TestCurriculumWeightUpdate:
    """Tests for curriculum weight update logic."""

    def test_increase_from_default_weight(self, watcher, mock_curriculum_feedback):
        """Test weight increase from default 1.0."""
        with patch("app.training.curriculum_feedback.get_curriculum_feedback", return_value=mock_curriculum_feedback):
            mock_curriculum_feedback._current_weights["hex8_2p"] = 1.0

            adjustment = watcher._adjust_curriculum_weight(
                "hex8_2p",
                1.2,
                {"error": "test_error"},
            )

            assert mock_curriculum_feedback._current_weights["hex8_2p"] == 1.2
            assert adjustment is not None
            assert adjustment.old_weight == 1.0
            assert adjustment.new_weight == 1.2

    def test_respects_weight_min(self, watcher, mock_curriculum_feedback):
        """Test respects weight_min boundary."""
        with patch("app.training.curriculum_feedback.get_curriculum_feedback", return_value=mock_curriculum_feedback):
            mock_curriculum_feedback._current_weights["hex8_2p"] = 0.5

            watcher._adjust_curriculum_weight(
                "hex8_2p",
                1.2,
                {"error": "test_error"},
            )

            assert mock_curriculum_feedback._current_weights["hex8_2p"] == 0.6

    def test_respects_weight_max(self, watcher, mock_curriculum_feedback):
        """Test respects weight_max boundary."""
        with patch("app.training.curriculum_feedback.get_curriculum_feedback", return_value=mock_curriculum_feedback):
            mock_curriculum_feedback._current_weights["hex8_2p"] = 2.4

            watcher._adjust_curriculum_weight(
                "hex8_2p",
                2.0,
                {"error": "test_error"},
            )

            assert mock_curriculum_feedback._current_weights["hex8_2p"] == 2.5

    def test_handles_missing_weight(self, watcher, mock_curriculum_feedback):
        """Test handles config with no existing weight."""
        with patch("app.training.curriculum_feedback.get_curriculum_feedback", return_value=mock_curriculum_feedback):
            watcher._adjust_curriculum_weight(
                "new_config",
                1.2,
                {"error": "test_error"},
            )

            assert mock_curriculum_feedback._current_weights["new_config"] == 1.2


# =============================================================================
# Event Emission Tests
# =============================================================================


class TestEventEmission:
    """Tests for CURRICULUM_REBALANCED event emission."""

    def test_emits_rebalance_event(self, watcher, mock_event, mock_curriculum_feedback):
        """Test emits CURRICULUM_REBALANCED after weight increase."""
        with patch("app.coordination.event_router.publish_sync") as mock_publish_sync:
            with patch("app.training.curriculum_feedback.get_curriculum_feedback", return_value=mock_curriculum_feedback):
                mock_curriculum_feedback._current_weights["hex8_2p"] = 1.0

                watcher._handle_event(mock_event)

                mock_publish_sync.assert_called_once()
                event_type, payload = mock_publish_sync.call_args.args[:2]
                assert event_type == "CURRICULUM_REBALANCED"
                assert payload["trigger"] == watcher.WATCHER_NAME
                assert payload["changed_configs"] == ["hex8_2p"]
                assert payload["old_weights"] == {"hex8_2p": 1.0}
                assert payload["new_weights"] == {"hex8_2p": 1.2}
                assert payload["failure_count"] == 1

    def test_event_emission_failure_handled(self, watcher, mock_event, mock_curriculum_feedback):
        """Test gracefully handles event emission failures."""
        with patch("app.coordination.event_router.publish_sync", side_effect=RuntimeError("publish failed")):
            with patch("app.training.curriculum_feedback.get_curriculum_feedback", return_value=mock_curriculum_feedback):
                mock_curriculum_feedback._current_weights["hex8_2p"] = 1.0

                watcher._handle_event(mock_event)

                assert mock_curriculum_feedback._current_weights["hex8_2p"] == 1.2


# =============================================================================
# Failure Count Management Tests
# =============================================================================


class TestFailureCountManagement:
    """Tests for failure count tracking and reset."""

    def test_get_failure_counts(self, watcher):
        """Test getting current failure counts."""
        watcher.set_state("hex8_2p:failure_count", 3)
        watcher.set_state("square8_4p:failure_count", 1)

        counts = watcher.get_failure_counts()

        assert counts == {"hex8_2p": 3, "square8_4p": 1}
        assert counts is not watcher._state

    def test_reset_failure_count(self, watcher):
        """Test resetting failure count for a config."""
        watcher.set_state("hex8_2p:failure_count", 5)
        watcher.set_state("square8_4p:failure_count", 2)

        watcher.reset_failure_count("hex8_2p")

        assert watcher.get_failure_counts() == {"square8_4p": 2}

    def test_reset_nonexistent_count(self, watcher):
        """Test resetting count for config without failures."""
        watcher.reset_failure_count("nonexistent_config")


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_malformed_event(self, watcher):
        """Test handles event with malformed payload."""
        event = MagicMock()
        event.payload = None

        watcher._handle_event(event)
        assert watcher.get_failure_counts() == {}

    def test_handles_curriculum_import_error(self, watcher, mock_event):
        """Test handles curriculum_feedback import failure."""
        with patch("app.training.curriculum_feedback.get_curriculum_feedback", side_effect=ImportError):
            watcher._handle_event(mock_event)
            assert watcher.get_failure_counts()["hex8_2p"] == 1

    def test_handles_attribute_error(self, watcher):
        """Test handles event missing payload attribute."""
        event = MagicMock(spec=[])  # No payload attribute

        watcher._handle_event(event)
        assert watcher.get_failure_counts() == {}

    def test_handles_type_error(self, watcher, mock_curriculum_feedback):
        """Test handles type errors in weight calculation."""
        with patch("app.training.curriculum_feedback.get_curriculum_feedback", return_value=mock_curriculum_feedback):
            mock_curriculum_feedback._current_weights["hex8_2p"] = "invalid"

            adjustment = watcher._adjust_curriculum_weight(
                "hex8_2p",
                1.2,
                {"error": "test"},
            )
            assert adjustment is None
