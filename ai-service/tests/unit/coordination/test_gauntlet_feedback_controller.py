"""Tests for GauntletFeedbackController.

Tests the evaluation→training feedback loop that adjusts training parameters
based on model performance in gauntlet evaluations.

December 2025: Created as part of test coverage audit.
"""

import asyncio
import pytest
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from app.coordination.gauntlet_feedback_controller import (
    GauntletFeedbackController,
    GauntletFeedbackConfig,
    FeedbackAction,
    EvaluationRecord,
    ConfigTracker,
    get_gauntlet_feedback_controller,
)
from app.coordination.protocols import CoordinatorStatus


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    """Create test configuration with shorter cooldowns."""
    return GauntletFeedbackConfig(
        strong_vs_heuristic_threshold=0.80,
        weak_vs_random_threshold=0.70,
        plateau_variance_threshold=0.01,
        plateau_window_size=5,
        exploration_reduction_factor=0.8,
        quality_threshold_boost=0.05,
        extra_selfplay_games=1000,  # Smaller for testing
        epoch_extension_factor=1.5,
        adjustment_cooldown=1.0,  # Short for testing
        curriculum_cooldown=2.0,
        regression_elo_drop=50.0,
        severe_regression_elo_drop=100.0,
        consecutive_regressions_for_rollback=3,
    )


@pytest.fixture
def controller(config):
    """Create a fresh controller instance."""
    return GauntletFeedbackController(config=config)


@pytest.fixture
def strong_evaluation():
    """Evaluation record for a strong model."""
    return EvaluationRecord(
        timestamp=1000.0,
        config_key="hex8_2p",
        board_type="hex8",
        num_players=2,
        elo=1250.0,
        win_rate_vs_random=0.95,
        win_rate_vs_heuristic=0.85,  # Above threshold
        games_played=100,
    )


@pytest.fixture
def weak_evaluation():
    """Evaluation record for a weak model."""
    return EvaluationRecord(
        timestamp=1000.0,
        config_key="hex8_2p",
        board_type="hex8",
        num_players=2,
        elo=950.0,
        win_rate_vs_random=0.65,  # Below threshold
        win_rate_vs_heuristic=0.45,
        games_played=100,
    )


@pytest.fixture
def average_evaluation():
    """Evaluation record for an average model (no action)."""
    return EvaluationRecord(
        timestamp=1000.0,
        config_key="hex8_2p",
        board_type="hex8",
        num_players=2,
        elo=1100.0,
        win_rate_vs_random=0.80,  # Between thresholds
        win_rate_vs_heuristic=0.70,
        games_played=100,
    )


# =============================================================================
# Initialization Tests
# =============================================================================


class TestControllerInitialization:
    """Test controller initialization and configuration."""

    def test_default_config(self):
        """Controller initializes with default config."""
        controller = GauntletFeedbackController()
        assert controller.config is not None
        assert controller.config.strong_vs_heuristic_threshold > 0
        assert controller.config.weak_vs_random_threshold > 0

    def test_custom_config(self, config):
        """Controller accepts custom configuration."""
        controller = GauntletFeedbackController(config=config)
        assert controller.config.adjustment_cooldown == 1.0
        assert controller.config.extra_selfplay_games == 1000

    def test_initial_state(self, controller):
        """Controller starts in correct initial state.

        December 30, 2025: Fixed to use _status (private attr in BaseEventHandler).
        Initial status is STOPPED, not INITIALIZING.
        """
        assert controller._status == CoordinatorStatus.STOPPED
        assert not controller.is_running
        assert len(controller._config_trackers) == 0

    def test_name_property(self, controller):
        """Controller has correct name."""
        assert controller.name == "GauntletFeedbackController"


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestControllerLifecycle:
    """Test controller start/stop lifecycle.

    December 30, 2025: Refactored to use BaseEventHandler API
    (_subscribe_all_events, _unsubscribe_all_events, _on_start, _on_stop).
    """

    @pytest.mark.asyncio
    async def test_start_success(self, controller):
        """Controller starts successfully with mocked event subscription."""
        # Mock the sync event subscription method
        with patch.object(controller, "_subscribe_all_events"):
            await controller.start()

            assert controller._running is True
            assert controller._status == CoordinatorStatus.RUNNING
            assert controller.uptime_seconds >= 0

            # Cleanup
            await controller.stop()

    @pytest.mark.asyncio
    async def test_double_start_warns(self, controller):
        """Starting an already-started controller logs warning."""
        with patch.object(controller, "_subscribe_all_events"):
            await controller.start()

            # Second start should just warn, not error
            await controller.start()

            assert controller._running is True

            # Cleanup
            await controller.stop()

    @pytest.mark.asyncio
    async def test_stop(self, controller):
        """Controller stops cleanly."""
        with patch.object(controller, "_subscribe_all_events"):
            with patch.object(controller, "_unsubscribe_all_events") as mock_unsub:
                await controller.start()
                await controller.stop()

                assert controller._running is False
                assert controller._status == CoordinatorStatus.STOPPED
                mock_unsub.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self, controller):
        """Stopping a non-running controller is a no-op."""
        # Stop without starting first
        await controller.stop()

        assert controller._running is False
        assert controller._status == CoordinatorStatus.STOPPED


# =============================================================================
# Analysis Logic Tests
# =============================================================================


class TestAnalyzeAndAct:
    """Test the core _analyze_and_act logic."""

    @pytest.mark.asyncio
    async def test_strong_model_reduces_exploration(self, controller, strong_evaluation):
        """Strong model triggers REDUCE_EXPLORATION action."""
        # Create tracker for the config
        tracker = ConfigTracker()
        controller._config_trackers["hex8_2p"] = tracker

        with patch.object(controller, "_reduce_exploration", new_callable=AsyncMock) as mock_reduce:
            with patch.object(controller, "_raise_quality_threshold", new_callable=AsyncMock):
                with patch.object(controller, "_emit_adaptive_params_changed", new_callable=AsyncMock):
                    actions = await controller._analyze_and_act(strong_evaluation, tracker)

                    # Should reduce exploration for strong models
                    mock_reduce.assert_called_once()

    @pytest.mark.asyncio
    async def test_weak_model_triggers_extra_selfplay(self, controller, weak_evaluation):
        """Weak model triggers TRIGGER_EXTRA_SELFPLAY action."""
        # Create tracker for the config
        tracker = ConfigTracker()
        controller._config_trackers["hex8_2p"] = tracker

        with patch.object(controller, "_trigger_extra_selfplay", new_callable=AsyncMock) as mock_selfplay:
            with patch.object(controller, "_extend_training_epochs", new_callable=AsyncMock):
                with patch.object(controller, "_emit_adaptive_params_changed", new_callable=AsyncMock):
                    actions = await controller._analyze_and_act(weak_evaluation, tracker)

                    # Should trigger extra selfplay for weak models
                    mock_selfplay.assert_called_once()

    @pytest.mark.asyncio
    async def test_average_model_no_action(self, controller, average_evaluation):
        """Average model takes no action."""
        # Create tracker for the config
        tracker = ConfigTracker()
        controller._config_trackers["hex8_2p"] = tracker

        actions = await controller._analyze_and_act(average_evaluation, tracker)

        # No action for models in between thresholds
        # (unless other conditions like plateau detected)
        assert FeedbackAction.NO_ACTION in actions or len(actions) == 1

    @pytest.mark.asyncio
    async def test_cooldown_prevents_rapid_actions(self, controller, strong_evaluation):
        """Cooldown period prevents rapid consecutive actions."""
        import time

        # Set up tracker with recent adjustment (within cooldown period)
        tracker = ConfigTracker()
        tracker.last_adjustment_time = time.time()  # Use time.time() to match implementation
        controller._config_trackers["hex8_2p"] = tracker

        with patch.object(controller, "_reduce_exploration", new_callable=AsyncMock) as mock_reduce:
            with patch.object(controller, "_raise_quality_threshold", new_callable=AsyncMock):
                with patch.object(controller, "_emit_adaptive_params_changed", new_callable=AsyncMock):
                    actions = await controller._analyze_and_act(strong_evaluation, tracker)

                    # Should skip due to cooldown - returns NO_ACTION
                    mock_reduce.assert_not_called()
                    assert FeedbackAction.NO_ACTION in actions


# =============================================================================
# Regression Detection Tests
# =============================================================================


class TestRegressionDetection:
    """Test ELO regression detection logic.

    January 2026: Updated to use consolidated RegressionDetector singleton
    instead of patching individual emission methods.
    """

    @pytest.mark.asyncio
    async def test_detects_elo_drop(self, controller):
        """Detects regression when ELO drops significantly."""
        from app.training.regression_detector import RegressionEvent, RegressionSeverity

        # Set up tracker with previous high ELO
        tracker = ConfigTracker()
        tracker.last_elo = 1200.0
        tracker.consecutive_regressions = 0
        controller._config_trackers["hex8_2p"] = tracker

        # Evaluation with significant ELO drop
        regression_eval = EvaluationRecord(
            timestamp=1000.0,
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
            elo=1100.0,  # 100 point drop (>= regression_elo_drop)
            win_rate_vs_random=0.75,
            win_rate_vs_heuristic=0.55,
            games_played=100,
        )

        # Mock the RegressionDetector to return a moderate regression event
        mock_event = RegressionEvent(
            model_id="hex8_2p",
            severity=RegressionSeverity.MODERATE,
            timestamp=1000.0,
            current_elo=1100.0,
            baseline_elo=1200.0,
            elo_drop=100.0,
            consecutive_count=1,
        )

        with patch("app.coordination.gauntlet_feedback_controller.get_regression_detector") as mock_get_detector:
            mock_detector = MagicMock()
            mock_detector.check_regression.return_value = mock_event
            mock_get_detector.return_value = mock_detector

            await controller._analyze_and_act(regression_eval, tracker)

            # Should sync consecutive_regressions from detector
            assert tracker.consecutive_regressions == 1

    @pytest.mark.asyncio
    async def test_consecutive_regressions_trigger_rollback(self, controller):
        """Multiple consecutive regressions trigger rollback consideration."""
        from app.training.regression_detector import RegressionEvent, RegressionSeverity

        tracker = ConfigTracker()
        tracker.last_elo = 1200.0
        tracker.consecutive_regressions = 2  # Will be overwritten by detector's count
        controller._config_trackers["hex8_2p"] = tracker

        regression_eval = EvaluationRecord(
            timestamp=1000.0,
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
            elo=1100.0,
            win_rate_vs_random=0.75,
            win_rate_vs_heuristic=0.55,
            games_played=100,
        )

        # Mock detector to return a CRITICAL regression (3+ consecutive)
        mock_event = RegressionEvent(
            model_id="hex8_2p",
            severity=RegressionSeverity.CRITICAL,  # Critical triggers rollback
            timestamp=1000.0,
            current_elo=1100.0,
            baseline_elo=1200.0,
            elo_drop=100.0,
            consecutive_count=3,  # 3 consecutive = CRITICAL
        )

        with patch("app.coordination.gauntlet_feedback_controller.get_regression_detector") as mock_get_detector:
            mock_detector = MagicMock()
            mock_detector.check_regression.return_value = mock_event
            mock_get_detector.return_value = mock_detector

            actions = await controller._analyze_and_act(regression_eval, tracker)

            # CRITICAL severity triggers rollback consideration
            assert FeedbackAction.CONSIDER_ROLLBACK in actions
            # Consecutive count synced from detector
            assert tracker.consecutive_regressions == 3


# =============================================================================
# Plateau Detection Tests
# =============================================================================


class TestPlateauDetection:
    """Test ELO plateau detection logic."""

    @pytest.mark.asyncio
    async def test_detects_stable_plateau(self, controller):
        """Detects plateau when ELO is stable."""
        tracker = ConfigTracker()
        # Add stable ELO history
        for _ in range(controller.config.plateau_window_size):
            tracker.elo_history.append(1100.0)

        result = await controller._detect_elo_plateau(tracker)

        # Variance is 0, should detect plateau
        assert result is True

    @pytest.mark.asyncio
    async def test_no_plateau_with_variance(self, controller):
        """No plateau detected with ELO variance."""
        tracker = ConfigTracker()
        # Add varying ELO history
        elos = [1000.0, 1050.0, 1100.0, 1150.0, 1200.0]
        for elo in elos:
            tracker.elo_history.append(elo)

        result = await controller._detect_elo_plateau(tracker)

        # High variance, should not detect plateau
        assert result is False

    @pytest.mark.asyncio
    async def test_insufficient_history_no_plateau(self, controller):
        """No plateau with insufficient history."""
        tracker = ConfigTracker()
        # Only 2 evaluations (less than window_size)
        tracker.elo_history.append(1100.0)
        tracker.elo_history.append(1100.0)

        result = await controller._detect_elo_plateau(tracker)

        # Not enough data, should not detect plateau
        assert result is False


# =============================================================================
# Event Handling Tests
# =============================================================================


class TestEventHandling:
    """Test evaluation event handling."""

    @pytest.mark.asyncio
    async def test_handles_valid_event(self, controller):
        """Handles valid evaluation event."""
        event = MagicMock()
        event.payload = {
            "board_type": "hex8",
            "num_players": 2,
            "elo": 1150.0,
            "win_rate_vs_random": 0.90,
            "win_rate_vs_heuristic": 0.75,
            "games_played": 100,
        }

        with patch.object(controller, "_analyze_and_act", new_callable=AsyncMock) as mock_analyze:
            # December 27, 2025: Migration to BaseEventHandler
            # Handler is now _on_evaluation_completed (async)
            await controller._on_evaluation_completed(event)

            mock_analyze.assert_called_once()


class TestEventEmissionHelpers:
    """Test direct event emission helpers use the unified publish helper."""

    @pytest.mark.asyncio
    async def test_trigger_extra_selfplay_uses_publish_helper(self, controller, weak_evaluation):
        from app.coordination.event_router import DataEventType

        with patch("app.coordination.gauntlet_feedback_controller.publish", new_callable=AsyncMock) as mock_publish:
            await controller._trigger_extra_selfplay(weak_evaluation)

        mock_publish.assert_awaited_once()
        _, kwargs = mock_publish.call_args
        assert kwargs["event_type"] == DataEventType.SELFPLAY_TARGET_UPDATED
        assert kwargs["payload"]["config"] == "hex8_2p"
        assert kwargs["payload"]["extra_games"] == controller.config.extra_selfplay_games
        assert kwargs["source"] == controller.name

    @pytest.mark.asyncio
    async def test_advance_curriculum_stage_uses_publish_helper(self, controller, average_evaluation):
        from app.coordination.event_router import DataEventType

        with patch("app.coordination.gauntlet_feedback_controller.publish", new_callable=AsyncMock) as mock_publish:
            await controller._advance_curriculum_stage(average_evaluation)

        mock_publish.assert_awaited_once()
        _, kwargs = mock_publish.call_args
        assert kwargs["event_type"] == DataEventType.CURRICULUM_ADVANCED
        assert kwargs["payload"]["config"] == "hex8_2p"
        assert kwargs["payload"]["reason"] == "elo_plateau"

    @pytest.mark.asyncio
    async def test_emit_regression_detected_uses_publish_helper(self, controller, average_evaluation):
        from app.coordination.event_router import DataEventType

        tracker = ConfigTracker()
        tracker.consecutive_regressions = 2
        controller._config_trackers["hex8_2p"] = tracker

        with patch("app.coordination.gauntlet_feedback_controller.publish", new_callable=AsyncMock) as mock_publish:
            await controller._emit_regression_detected(average_evaluation, 75.0)

        mock_publish.assert_awaited_once()
        _, kwargs = mock_publish.call_args
        assert kwargs["event_type"] == DataEventType.REGRESSION_DETECTED
        assert kwargs["payload"]["config_key"] == "hex8_2p"
        assert kwargs["payload"]["elo_drop"] == pytest.approx(75.0)
        assert kwargs["payload"]["consecutive_regressions"] == 2

    @pytest.mark.asyncio
    async def test_emit_rollback_consideration_uses_publish_helper(self, controller, average_evaluation):
        from app.coordination.event_router import DataEventType

        tracker = ConfigTracker()
        tracker.consecutive_regressions = 3
        controller._config_trackers["hex8_2p"] = tracker

        with patch("app.coordination.gauntlet_feedback_controller.publish", new_callable=AsyncMock) as mock_publish:
            await controller._emit_rollback_consideration(
                average_evaluation,
                severity="consecutive",
                elo_drop=90.0,
            )

        mock_publish.assert_awaited_once()
        _, kwargs = mock_publish.call_args
        assert kwargs["event_type"] == DataEventType.REGRESSION_CRITICAL
        assert kwargs["payload"]["config_key"] == "hex8_2p"
        assert kwargs["payload"]["recommendation"] == "rollback"
        assert "consecutive_regressions_3" in kwargs["payload"]["reason"]

    @pytest.mark.asyncio
    async def test_emit_hyperparameter_update_uses_publish_helper(self, controller):
        from app.coordination.event_router import DataEventType

        with patch("app.coordination.gauntlet_feedback_controller.publish", new_callable=AsyncMock) as mock_publish:
            with patch("app.coordination.gauntlet_feedback_controller.record_hyperparameter_update") as mock_record:
                await controller._emit_hyperparameter_update(
                    config_key="hex8_2p",
                    board_type="hex8",
                    num_players=2,
                    parameter="epoch_multiplier",
                    old_value=1.0,
                    new_value=1.5,
                    reason="weak_performance",
                )

        mock_publish.assert_awaited_once()
        _, kwargs = mock_publish.call_args
        assert kwargs["event_type"] == DataEventType.HYPERPARAMETER_UPDATED
        assert kwargs["payload"]["parameter"] == "epoch_multiplier"
        assert kwargs["payload"]["new_value"] == pytest.approx(1.5)
        mock_record.assert_called_once_with("hex8_2p", "epoch_multiplier", 1.5, "weak_performance")

    @pytest.mark.asyncio
    async def test_emit_plateau_detected_uses_publish_helper(self, controller):
        from app.coordination.event_router import DataEventType

        with patch("app.coordination.gauntlet_feedback_controller.publish", new_callable=AsyncMock) as mock_publish:
            await controller._emit_plateau_detected("hex8_2p", 1180.0, 5)

        mock_publish.assert_awaited_once()
        _, kwargs = mock_publish.call_args
        assert kwargs["event_type"] == DataEventType.PLATEAU_DETECTED
        assert kwargs["payload"]["config_key"] == "hex8_2p"
        assert kwargs["payload"]["window_size"] == 5

    @pytest.mark.asyncio
    async def test_emit_adaptive_params_changed_uses_publish_helper(self, controller):
        from app.coordination.event_router import DataEventType

        with patch("app.coordination.gauntlet_feedback_controller.publish", new_callable=AsyncMock) as mock_publish:
            await controller._emit_adaptive_params_changed(
                config_key="hex8_2p",
                board_type="hex8",
                num_players=2,
                actions_taken=[FeedbackAction.TRIGGER_EXTRA_SELFPLAY],
            )

        mock_publish.assert_awaited_once()
        _, kwargs = mock_publish.call_args
        assert kwargs["event_type"] == DataEventType.ADAPTIVE_PARAMS_CHANGED
        assert kwargs["payload"]["config_key"] == "hex8_2p"
        assert kwargs["payload"]["temperature_multiplier"] == pytest.approx(1.2)
        assert kwargs["payload"]["search_budget_multiplier"] == pytest.approx(1.3)

    @pytest.mark.asyncio
    async def test_handles_missing_fields_gracefully(self, controller):
        """Gracefully handles events with missing fields."""
        event = MagicMock()
        event.payload = {
            "board_type": "hex8",
            # Missing num_players, elo, etc.
        }

        # Should not raise, just log and skip
        # December 27, 2025: Migration to BaseEventHandler
        await controller._on_evaluation_completed(event)

        # No crash means success

    @pytest.mark.asyncio
    async def test_handles_dict_event(self, controller):
        """Handles event with payload dict attribute."""
        # Implementation expects event.payload, not a raw dict
        event = MagicMock()
        event.payload = {
            "board_type": "hex8",
            "num_players": 2,
            "elo": 1100.0,
            "win_rate_vs_random": 0.85,
            "win_rate_vs_heuristic": 0.70,
            "games_played": 50,
        }

        with patch.object(controller, "_analyze_and_act", new_callable=AsyncMock) as mock_analyze:
            # December 27, 2025: Migration to BaseEventHandler
            await controller._on_evaluation_completed(event)

            mock_analyze.assert_called_once()


# =============================================================================
# Metrics and Health Check Tests
# =============================================================================


class TestMetricsAndHealth:
    """Test metrics and health check functionality.

    December 30, 2025: Updated to match actual get_metrics() implementation.
    """

    def test_get_metrics(self, controller):
        """Metrics returns expected fields."""
        metrics = controller.get_metrics()

        # December 30, 2025: Updated to match actual implementation
        assert "is_running" in metrics
        assert "uptime_seconds" in metrics
        assert "configs_tracked" in metrics
        assert "actions_taken" in metrics
        assert "subscribed" in metrics

    def test_health_check_healthy(self, controller):
        """Health check returns healthy for running controller.

        December 30, 2025: Fixed to set _running which is what health_check checks.
        """
        controller._running = True
        controller._status = CoordinatorStatus.RUNNING

        result = controller.health_check()

        assert result.healthy is True
        assert result.status == CoordinatorStatus.RUNNING

    def test_health_check_unhealthy_when_not_running(self, controller):
        """Health check returns unhealthy when not running."""
        controller._running = False
        controller._status = CoordinatorStatus.STOPPED

        result = controller.health_check()

        assert result.healthy is False
        assert "not running" in result.message.lower()


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Test singleton pattern."""

    @pytest.mark.asyncio
    async def test_get_gauntlet_feedback_controller_returns_same_instance(self):
        """Singleton returns same instance."""
        # Reset singleton for test
        import app.coordination.gauntlet_feedback_controller as module
        module._controller_instance = None

        controller1 = await get_gauntlet_feedback_controller()
        controller2 = await get_gauntlet_feedback_controller()

        assert controller1 is controller2

        # Reset for other tests
        module._controller_instance = None


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the feedback loop.

    December 30, 2025: Refactored to use BaseEventHandler API.
    """

    @pytest.mark.asyncio
    async def test_full_feedback_cycle(self, controller, strong_evaluation):
        """Test complete feedback cycle from event to action."""
        # Start controller with mocked subscriptions
        with patch.object(controller, "_subscribe_all_events"):
            await controller.start()

        # Create tracker for the config
        tracker = ConfigTracker()
        controller._config_trackers["hex8_2p"] = tracker

        try:
            # Mock event emission
            with patch.object(controller, "_emit_hyperparameter_update", new_callable=AsyncMock):
                with patch.object(controller, "_emit_adaptive_params_changed", new_callable=AsyncMock):
                    # Process a strong evaluation
                    actions = await controller._analyze_and_act(strong_evaluation, tracker)

                    # Should have taken some action
                    assert len(actions) > 0
                    assert FeedbackAction.REDUCE_EXPLORATION in actions or FeedbackAction.NO_ACTION not in actions
        finally:
            # Cleanup
            await controller.stop()

    @pytest.mark.asyncio
    async def test_tracker_persistence_across_evaluations(self, controller):
        """Config trackers persist across multiple evaluations."""
        eval1 = EvaluationRecord(
            timestamp=1000.0,
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
            elo=1100.0,
            win_rate_vs_random=0.85,
            win_rate_vs_heuristic=0.70,
            games_played=100,
        )

        eval2 = EvaluationRecord(
            timestamp=2000.0,
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
            elo=1150.0,
            win_rate_vs_random=0.88,
            win_rate_vs_heuristic=0.72,
            games_played=100,
        )

        # Create initial tracker
        tracker = controller._config_trackers.setdefault("hex8_2p", ConfigTracker())

        await controller._analyze_and_act(eval1, tracker)
        await controller._analyze_and_act(eval2, tracker)

        # Should have same tracker with history
        assert "hex8_2p" in controller._config_trackers
        tracker = controller._config_trackers["hex8_2p"]
        assert len(tracker.elo_history) == 2
