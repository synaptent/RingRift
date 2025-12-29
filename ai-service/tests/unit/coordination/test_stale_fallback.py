"""Tests for stale_fallback.py module.

December 2025: Tests for Training Stale Data Fallback Controller.
Part of 48-hour autonomous operation testing.
"""

import time
from unittest.mock import patch, MagicMock

import pytest

from app.coordination.stale_fallback import (
    FallbackDecision,
    FallbackState,
    TrainingFallbackController,
    get_training_fallback_controller,
    reset_training_fallback_controller,
    should_allow_stale_training,
)


class TestFallbackDecision:
    """Tests for FallbackDecision dataclass."""

    def test_default_values(self):
        """Test default values for FallbackDecision."""
        decision = FallbackDecision(allowed=True, reason="test")
        assert decision.allowed is True
        assert decision.reason == "test"
        assert decision.is_fallback is False
        assert decision.data_age_hours == 0.0
        assert decision.sync_failures == 0
        assert decision.elapsed_time == 0.0

    def test_all_fields(self):
        """Test FallbackDecision with all fields set."""
        decision = FallbackDecision(
            allowed=True,
            reason="Too many failures",
            is_fallback=True,
            data_age_hours=2.5,
            sync_failures=5,
            elapsed_time=1800.0,
        )
        assert decision.allowed is True
        assert decision.is_fallback is True
        assert decision.data_age_hours == 2.5
        assert decision.sync_failures == 5
        assert decision.elapsed_time == 1800.0


class TestFallbackState:
    """Tests for FallbackState dataclass."""

    def test_default_values(self):
        """Test default values for FallbackState."""
        state = FallbackState(config_key="hex8_2p")
        assert state.config_key == "hex8_2p"
        assert state.sync_failures == 0
        assert state.first_failure_time == 0.0
        assert state.last_fallback_time == 0.0
        assert state.fallback_count == 0

    def test_reset(self):
        """Test reset() clears failure tracking."""
        state = FallbackState(
            config_key="hex8_2p",
            sync_failures=5,
            first_failure_time=12345.0,
            last_fallback_time=12000.0,
            fallback_count=2,
        )
        state.reset()
        assert state.sync_failures == 0
        assert state.first_failure_time == 0.0
        # These should NOT be reset
        assert state.last_fallback_time == 12000.0
        assert state.fallback_count == 2

    def test_record_failure_first(self):
        """Test record_failure() sets first_failure_time on first failure."""
        state = FallbackState(config_key="hex8_2p")
        before = time.time()
        state.record_failure()
        after = time.time()

        assert state.sync_failures == 1
        assert before <= state.first_failure_time <= after

    def test_record_failure_subsequent(self):
        """Test record_failure() increments count without changing first_failure_time."""
        state = FallbackState(config_key="hex8_2p", first_failure_time=1000.0)
        state.record_failure()

        assert state.sync_failures == 1
        assert state.first_failure_time == 1000.0  # Unchanged

    def test_record_fallback(self):
        """Test record_fallback() updates last_fallback_time and count."""
        state = FallbackState(config_key="hex8_2p")
        before = time.time()
        state.record_fallback()
        after = time.time()

        assert state.fallback_count == 1
        assert before <= state.last_fallback_time <= after

        # Second fallback
        state.record_fallback()
        assert state.fallback_count == 2


class TestTrainingFallbackController:
    """Tests for TrainingFallbackController class."""

    @pytest.fixture
    def controller(self):
        """Create fresh controller for each test."""
        return TrainingFallbackController()

    def test_get_state_creates_new(self, controller):
        """Test _get_state creates new state if not exists."""
        state = controller._get_state("hex8_2p")
        assert state.config_key == "hex8_2p"
        assert state.sync_failures == 0

    def test_get_state_returns_existing(self, controller):
        """Test _get_state returns existing state."""
        state1 = controller._get_state("hex8_2p")
        state1.sync_failures = 5
        state2 = controller._get_state("hex8_2p")
        assert state2.sync_failures == 5
        assert state1 is state2

    def test_record_sync_failure(self, controller):
        """Test record_sync_failure increments counter."""
        controller.record_sync_failure("hex8_2p")
        assert controller._get_state("hex8_2p").sync_failures == 1

        controller.record_sync_failure("hex8_2p")
        assert controller._get_state("hex8_2p").sync_failures == 2

    def test_record_sync_success(self, controller):
        """Test record_sync_success resets failure tracking."""
        controller.record_sync_failure("hex8_2p")
        controller.record_sync_failure("hex8_2p")
        assert controller._get_state("hex8_2p").sync_failures == 2

        controller.record_sync_success("hex8_2p")
        assert controller._get_state("hex8_2p").sync_failures == 0

    @patch("app.coordination.stale_fallback.StaleFallbackDefaults")
    def test_disabled_fallback(self, mock_defaults, controller):
        """Test fallback is blocked when disabled."""
        mock_defaults.ENABLE_STALE_FALLBACK = False

        decision = controller.should_allow_training(
            config_key="hex8_2p",
            data_age_hours=1.0,
            sync_failures=10,  # Would normally trigger
        )

        assert decision.allowed is False
        assert "disabled" in decision.reason.lower()

    @patch("app.coordination.stale_fallback.StaleFallbackDefaults")
    def test_data_too_old(self, mock_defaults, controller):
        """Test fallback is blocked when data exceeds absolute max age."""
        mock_defaults.ENABLE_STALE_FALLBACK = True
        mock_defaults.ABSOLUTE_MAX_DATA_AGE = 24.0

        decision = controller.should_allow_training(
            config_key="hex8_2p",
            data_age_hours=25.0,
        )

        assert decision.allowed is False
        assert "too old" in decision.reason.lower()

    @patch("app.coordination.stale_fallback.StaleFallbackDefaults")
    def test_insufficient_games(self, mock_defaults, controller):
        """Test fallback is blocked with too few games."""
        mock_defaults.ENABLE_STALE_FALLBACK = True
        mock_defaults.ABSOLUTE_MAX_DATA_AGE = 24.0
        mock_defaults.MIN_GAMES_FOR_FALLBACK = 100

        decision = controller.should_allow_training(
            config_key="hex8_2p",
            data_age_hours=2.0,
            games_available=50,
        )

        assert decision.allowed is False
        assert "insufficient" in decision.reason.lower()

    @patch("app.coordination.stale_fallback.StaleFallbackDefaults")
    def test_cooldown_active(self, mock_defaults, controller):
        """Test fallback is blocked during cooldown period."""
        mock_defaults.ENABLE_STALE_FALLBACK = True
        mock_defaults.ABSOLUTE_MAX_DATA_AGE = 24.0
        mock_defaults.MIN_GAMES_FOR_FALLBACK = 0
        mock_defaults.FALLBACK_COOLDOWN = 3600.0  # 1 hour

        # Record recent fallback
        state = controller._get_state("hex8_2p")
        state.last_fallback_time = time.time() - 1800  # 30 minutes ago

        decision = controller.should_allow_training(
            config_key="hex8_2p",
            data_age_hours=2.0,
            sync_failures=10,  # Would normally trigger
        )

        assert decision.allowed is False
        assert "cooldown" in decision.reason.lower()

    @patch("app.coordination.stale_fallback.StaleFallbackDefaults")
    def test_too_many_failures_triggers_fallback(self, mock_defaults, controller):
        """Test fallback is allowed after too many sync failures."""
        mock_defaults.ENABLE_STALE_FALLBACK = True
        mock_defaults.ABSOLUTE_MAX_DATA_AGE = 24.0
        mock_defaults.MIN_GAMES_FOR_FALLBACK = 0
        mock_defaults.MAX_SYNC_FAILURES = 5
        mock_defaults.MAX_SYNC_DURATION = 3600.0
        mock_defaults.FALLBACK_COOLDOWN = 0.0
        mock_defaults.EMIT_FALLBACK_EVENTS = False

        decision = controller.should_allow_training(
            config_key="hex8_2p",
            data_age_hours=2.0,
            sync_failures=5,
            games_available=1000,
        )

        assert decision.allowed is True
        assert decision.is_fallback is True
        assert "5 times" in decision.reason

    @patch("app.coordination.stale_fallback.StaleFallbackDefaults")
    def test_sync_duration_triggers_fallback(self, mock_defaults, controller):
        """Test fallback is allowed after sync takes too long."""
        mock_defaults.ENABLE_STALE_FALLBACK = True
        mock_defaults.ABSOLUTE_MAX_DATA_AGE = 24.0
        mock_defaults.MIN_GAMES_FOR_FALLBACK = 0
        mock_defaults.MAX_SYNC_FAILURES = 100  # Won't trigger
        mock_defaults.MAX_SYNC_DURATION = 1800.0  # 30 minutes
        mock_defaults.FALLBACK_COOLDOWN = 0.0
        mock_defaults.EMIT_FALLBACK_EVENTS = False

        decision = controller.should_allow_training(
            config_key="hex8_2p",
            data_age_hours=2.0,
            sync_failures=2,
            elapsed_sync_time=2000.0,  # Exceeds 1800s
            games_available=1000,
        )

        assert decision.allowed is True
        assert decision.is_fallback is True
        assert "exceeded" in decision.reason.lower()

    @patch("app.coordination.stale_fallback.StaleFallbackDefaults")
    def test_not_yet_time_to_fallback(self, mock_defaults, controller):
        """Test fallback is denied when conditions aren't met."""
        mock_defaults.ENABLE_STALE_FALLBACK = True
        mock_defaults.ABSOLUTE_MAX_DATA_AGE = 24.0
        mock_defaults.MIN_GAMES_FOR_FALLBACK = 0
        mock_defaults.MAX_SYNC_FAILURES = 10
        mock_defaults.MAX_SYNC_DURATION = 3600.0
        mock_defaults.FALLBACK_COOLDOWN = 0.0

        decision = controller.should_allow_training(
            config_key="hex8_2p",
            data_age_hours=2.0,
            sync_failures=3,  # Below threshold
            elapsed_sync_time=1000.0,  # Below threshold
            games_available=1000,
        )

        assert decision.allowed is False
        assert "in progress" in decision.reason.lower()

    def test_get_status(self, controller):
        """Test get_status returns correct data."""
        controller.record_sync_failure("hex8_2p")
        controller.record_sync_failure("hex8_2p")

        status = controller.get_status()

        assert "enabled" in status
        assert "config" in status
        assert "states" in status
        assert "uptime_seconds" in status
        assert "hex8_2p" in status["states"]
        assert status["states"]["hex8_2p"]["sync_failures"] == 2


class TestSingletonFunctions:
    """Tests for module-level singleton functions."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_training_fallback_controller()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_training_fallback_controller()

    def test_get_returns_singleton(self):
        """Test get_training_fallback_controller returns same instance."""
        c1 = get_training_fallback_controller()
        c2 = get_training_fallback_controller()
        assert c1 is c2

    def test_reset_clears_singleton(self):
        """Test reset_training_fallback_controller clears instance."""
        c1 = get_training_fallback_controller()
        c1.record_sync_failure("hex8_2p")

        reset_training_fallback_controller()

        c2 = get_training_fallback_controller()
        assert c2 is not c1
        assert c2._get_state("hex8_2p").sync_failures == 0


class TestShouldAllowStaleTraining:
    """Tests for should_allow_stale_training convenience function."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_training_fallback_controller()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_training_fallback_controller()

    @patch("app.coordination.stale_fallback.StaleFallbackDefaults")
    def test_returns_tuple(self, mock_defaults):
        """Test function returns (allowed, reason) tuple."""
        mock_defaults.ENABLE_STALE_FALLBACK = False

        result = should_allow_stale_training(
            config_key="hex8_2p",
            data_age_hours=1.0,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        allowed, reason = result
        assert isinstance(allowed, bool)
        assert isinstance(reason, str)

    @patch("app.coordination.stale_fallback.StaleFallbackDefaults")
    def test_allowed_case(self, mock_defaults):
        """Test function returns True when fallback allowed."""
        mock_defaults.ENABLE_STALE_FALLBACK = True
        mock_defaults.ABSOLUTE_MAX_DATA_AGE = 24.0
        mock_defaults.MIN_GAMES_FOR_FALLBACK = 0
        mock_defaults.MAX_SYNC_FAILURES = 5
        mock_defaults.MAX_SYNC_DURATION = 3600.0
        mock_defaults.FALLBACK_COOLDOWN = 0.0
        mock_defaults.EMIT_FALLBACK_EVENTS = False

        allowed, reason = should_allow_stale_training(
            config_key="hex8_2p",
            data_age_hours=2.0,
            sync_failures=10,
            games_available=1000,
        )

        assert allowed is True
        assert "failed" in reason.lower() or "times" in reason.lower()

    @patch("app.coordination.stale_fallback.StaleFallbackDefaults")
    def test_denied_case(self, mock_defaults):
        """Test function returns False when fallback denied."""
        mock_defaults.ENABLE_STALE_FALLBACK = False

        allowed, reason = should_allow_stale_training(
            config_key="hex8_2p",
            data_age_hours=2.0,
            sync_failures=10,
        )

        assert allowed is False
        assert "disabled" in reason.lower()


class TestEventEmission:
    """Tests for fallback event emission."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_training_fallback_controller()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_training_fallback_controller()

    @patch("app.coordination.stale_fallback.StaleFallbackDefaults")
    def test_event_emitted_on_fallback(self, mock_defaults):
        """Test STALE_TRAINING_FALLBACK event is emitted."""
        mock_defaults.ENABLE_STALE_FALLBACK = True
        mock_defaults.ABSOLUTE_MAX_DATA_AGE = 24.0
        mock_defaults.MIN_GAMES_FOR_FALLBACK = 0
        mock_defaults.MAX_SYNC_FAILURES = 5
        mock_defaults.MAX_SYNC_DURATION = 3600.0
        mock_defaults.FALLBACK_COOLDOWN = 0.0
        mock_defaults.EMIT_FALLBACK_EVENTS = True

        controller = TrainingFallbackController()

        # Patch at the import location inside _emit_fallback_event
        with patch("app.coordination.event_router.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_get_router.return_value = mock_router

            controller.should_allow_training(
                config_key="hex8_2p",
                data_age_hours=2.0,
                sync_failures=10,
                games_available=1000,
            )

            mock_router.publish_sync.assert_called_once()
            call_kwargs = mock_router.publish_sync.call_args[1]
            assert call_kwargs["event_type"] == "STALE_TRAINING_FALLBACK"
            assert call_kwargs["payload"]["config_key"] == "hex8_2p"

    @patch("app.coordination.stale_fallback.StaleFallbackDefaults")
    def test_event_not_emitted_when_disabled(self, mock_defaults):
        """Test event is not emitted when EMIT_FALLBACK_EVENTS is False."""
        mock_defaults.ENABLE_STALE_FALLBACK = True
        mock_defaults.ABSOLUTE_MAX_DATA_AGE = 24.0
        mock_defaults.MIN_GAMES_FOR_FALLBACK = 0
        mock_defaults.MAX_SYNC_FAILURES = 5
        mock_defaults.MAX_SYNC_DURATION = 3600.0
        mock_defaults.FALLBACK_COOLDOWN = 0.0
        mock_defaults.EMIT_FALLBACK_EVENTS = False

        controller = TrainingFallbackController()

        # Patch at the import location inside _emit_fallback_event
        with patch("app.coordination.event_router.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_get_router.return_value = mock_router

            controller.should_allow_training(
                config_key="hex8_2p",
                data_age_hours=2.0,
                sync_failures=10,
                games_available=1000,
            )

            mock_router.publish_sync.assert_not_called()
