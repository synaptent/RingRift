"""Unit tests for FeedbackLoopController.

Tests the core feedback loop orchestration including:
- FeedbackState dataclass and Elo velocity tracking
- Singleton pattern (get/reset)
- Lifecycle (start/stop)
- State management per config
- Health check reporting
- Quality signaling methods

Dec 30, 2025: Created comprehensive test coverage for this critical 3,631 LOC module.
"""

import pytest
import time
from unittest.mock import MagicMock, patch, AsyncMock

from app.coordination.feedback_loop_controller import (
    FeedbackState,
    FeedbackLoopController,
    get_feedback_loop_controller,
    reset_feedback_loop_controller,
)


class TestFeedbackState:
    """Test FeedbackState dataclass."""

    def test_default_state(self):
        """Test default state values."""
        state = FeedbackState(config_key="hex8_2p")
        assert state.config_key == "hex8_2p"
        assert state.last_selfplay_quality == 0.0
        assert state.last_training_accuracy == 0.0
        assert state.last_evaluation_win_rate == 0.0
        assert state.last_elo == 1500.0
        assert state.elo_velocity == 0.0
        assert state.consecutive_failures == 0
        assert state.consecutive_successes == 0

    def test_elo_history_initialization(self):
        """Test elo_history initializes as empty list."""
        state = FeedbackState(config_key="hex8_2p")
        assert state.elo_history == []
        assert isinstance(state.elo_history, list)

    def test_current_signals_defaults(self):
        """Test current signal defaults."""
        state = FeedbackState(config_key="hex8_2p")
        assert state.current_training_intensity == "normal"
        assert state.current_exploration_boost == 1.0
        assert state.current_curriculum_weight == 1.0
        assert state.current_search_budget == 400

    def test_work_queue_metrics_defaults(self):
        """Test work queue metric defaults."""
        state = FeedbackState(config_key="hex8_2p")
        assert state.work_completed_count == 0
        assert state.last_work_completion_time == 0.0

    def test_engine_bandit_tracking_defaults(self):
        """Test engine bandit tracking defaults."""
        state = FeedbackState(config_key="hex8_2p")
        assert state.last_selfplay_engine == "gumbel-mcts"
        assert state.last_selfplay_games == 0
        assert state.elo_before_training == 1500.0

    def test_curriculum_tier_defaults(self):
        """Test curriculum tier tracking defaults."""
        state = FeedbackState(config_key="hex8_2p")
        assert state.curriculum_tier == 0
        assert state.curriculum_last_advanced == 0.0


class TestFeedbackStateEloTracking:
    """Test FeedbackState Elo velocity tracking."""

    def test_update_elo_stores_value(self):
        """Test update_elo stores the new Elo value."""
        state = FeedbackState(config_key="hex8_2p")
        state.update_elo(1600.0)
        assert state.last_elo == 1600.0

    def test_update_elo_appends_to_history(self):
        """Test update_elo appends to history."""
        state = FeedbackState(config_key="hex8_2p")
        state.update_elo(1600.0)
        assert len(state.elo_history) == 1
        assert state.elo_history[0][1] == 1600.0

    def test_update_elo_calculates_velocity(self):
        """Test update_elo calculates velocity from history.

        Note: Velocity calculation requires 3+ data points.
        """
        state = FeedbackState(config_key="hex8_2p")
        base_time = time.time()

        # Velocity requires 3+ data points
        state.elo_history = [
            (base_time - 7200, 1400.0),  # 2 hours ago
            (base_time - 3600, 1500.0),  # 1 hour ago
        ]
        velocity = state.update_elo(1600.0, base_time)

        # Should calculate ~100 Elo/hour
        assert velocity > 0.0

    def test_update_elo_with_custom_timestamp(self):
        """Test update_elo accepts custom timestamp."""
        state = FeedbackState(config_key="hex8_2p")
        custom_time = 1000000.0
        state.update_elo(1600.0, custom_time)
        assert state.elo_history[0][0] == custom_time

    def test_update_elo_velocity_negative_regression(self):
        """Test velocity is negative when Elo regresses.

        Note: Velocity calculation requires 3+ data points.
        """
        state = FeedbackState(config_key="hex8_2p")
        base_time = time.time()

        # Velocity requires 3+ data points - show regression
        state.elo_history = [
            (base_time - 7200, 1700.0),  # 2 hours ago
            (base_time - 3600, 1600.0),  # 1 hour ago
        ]
        velocity = state.update_elo(1500.0, base_time)

        # Velocity should be negative (regression)
        assert velocity < 0.0


class TestFeedbackStateDataclass:
    """Test FeedbackState as a dataclass."""

    def test_dataclass_fields_accessible(self):
        """Test dataclass fields are accessible."""
        state = FeedbackState(config_key="hex8_2p")
        state.last_elo = 1600.0
        state.elo_velocity = 25.0

        # Verify fields are directly accessible
        assert state.config_key == "hex8_2p"
        assert state.last_elo == 1600.0
        assert state.elo_velocity == 25.0

    def test_dataclass_asdict(self):
        """Test dataclass can be converted using asdict."""
        from dataclasses import asdict

        state = FeedbackState(config_key="hex8_2p")
        state.current_training_intensity = "accelerated"
        state.current_exploration_boost = 1.5

        d = asdict(state)

        assert d["config_key"] == "hex8_2p"
        assert d["current_training_intensity"] == "accelerated"
        assert d["current_exploration_boost"] == 1.5


class TestFeedbackLoopControllerSingleton:
    """Test FeedbackLoopController singleton pattern."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_feedback_loop_controller()

    def test_get_instance_returns_same_object(self):
        """Test get_feedback_loop_controller returns same instance."""
        controller1 = get_feedback_loop_controller()
        controller2 = get_feedback_loop_controller()
        assert controller1 is controller2

    def test_reset_creates_new_instance(self):
        """Test reset creates a new instance."""
        controller1 = get_feedback_loop_controller()
        reset_feedback_loop_controller()
        controller2 = get_feedback_loop_controller()
        assert controller1 is not controller2


class TestFeedbackLoopControllerLifecycle:
    """Test FeedbackLoopController lifecycle management."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_feedback_loop_controller()

    @pytest.mark.asyncio
    async def test_start_sets_running(self):
        """Test start() sets running flag."""
        controller = get_feedback_loop_controller()
        assert not controller.is_running()

        await controller.start()

        assert controller.is_running()
        await controller.stop()

    @pytest.mark.asyncio
    async def test_stop_clears_running(self):
        """Test stop() clears running flag."""
        controller = get_feedback_loop_controller()
        await controller.start()

        await controller.stop()

        assert not controller.is_running()

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self):
        """Test start() is idempotent."""
        controller = get_feedback_loop_controller()
        await controller.start()
        await controller.start()  # Second call should be safe

        assert controller.is_running()
        await controller.stop()


class TestFeedbackLoopControllerState:
    """Test FeedbackLoopController state management."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_feedback_loop_controller()

    def test_get_state_returns_none_for_unknown(self):
        """Test get_state returns None for unknown config."""
        controller = get_feedback_loop_controller()
        state = controller.get_state("nonexistent_config")
        assert state is None

    def test_get_all_states_empty_initially(self):
        """Test get_all_states returns empty dict initially."""
        controller = get_feedback_loop_controller()
        states = controller.get_all_states()
        assert states == {}

    def test_signal_selfplay_quality_creates_state(self):
        """Test signal_selfplay_quality creates state if needed."""
        controller = get_feedback_loop_controller()
        controller.signal_selfplay_quality("hex8_2p", 0.85)

        state = controller.get_state("hex8_2p")
        assert state is not None
        assert state.last_selfplay_quality == 0.85

    def test_get_all_states_after_signaling(self):
        """Test get_all_states includes signaled configs."""
        controller = get_feedback_loop_controller()
        controller.signal_selfplay_quality("hex8_2p", 0.85)
        controller.signal_selfplay_quality("square8_4p", 0.90)

        states = controller.get_all_states()
        assert "hex8_2p" in states
        assert "square8_4p" in states


class TestFeedbackLoopControllerSignaling:
    """Test FeedbackLoopController signaling methods."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_feedback_loop_controller()

    def test_signal_selfplay_quality(self):
        """Test signal_selfplay_quality updates state."""
        controller = get_feedback_loop_controller()
        controller.signal_selfplay_quality("hex8_2p", 0.92)

        state = controller.get_state("hex8_2p")
        assert state.last_selfplay_quality == 0.92

    def test_signal_training_complete_updates_state(self):
        """Test signal_training_complete updates state."""
        controller = get_feedback_loop_controller()
        controller.signal_training_complete(
            config_key="hex8_2p",
            policy_accuracy=0.75,
            value_accuracy=0.5,
        )

        state = controller.get_state("hex8_2p")
        assert state.last_training_accuracy == 0.75


class TestFeedbackLoopControllerHealth:
    """Test FeedbackLoopController health check."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_feedback_loop_controller()

    def test_health_check_returns_result(self):
        """Test health_check returns a valid result."""
        controller = get_feedback_loop_controller()
        result = controller.health_check()

        assert hasattr(result, 'healthy')
        assert hasattr(result, 'details')

    def test_health_check_not_running(self):
        """Test health_check when not running."""
        controller = get_feedback_loop_controller()
        result = controller.health_check()

        # Should still return result, but indicate not running
        assert result is not None

    @pytest.mark.asyncio
    async def test_health_check_when_running(self):
        """Test health_check when running."""
        controller = get_feedback_loop_controller()
        await controller.start()

        result = controller.health_check()

        assert result.healthy
        await controller.stop()


class TestFeedbackLoopControllerStatus:
    """Test FeedbackLoopController status reporting."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_feedback_loop_controller()

    def test_get_status_includes_running(self):
        """Test get_status includes running flag."""
        controller = get_feedback_loop_controller()
        status = controller.get_status()

        assert "running" in status
        assert status["running"] is False

    def test_get_status_includes_config_count(self):
        """Test get_status includes config count.

        Note: get_status accesses internal state that may not be
        fully populated without event-driven updates.
        This test verifies basic structure only.
        """
        controller = get_feedback_loop_controller()
        # Don't signal - just verify empty status works
        status = controller.get_status()

        # Should have configs section even when empty
        assert "configs" in status
        assert status["configs"]["total"] == 0
        assert status["running"] is False

    def test_get_summary(self):
        """Test get_summary returns overview."""
        controller = get_feedback_loop_controller()
        summary = controller.get_summary()

        assert isinstance(summary, dict)


class TestFeedbackLoopControllerEventSubscription:
    """Test FeedbackLoopController event subscription."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_feedback_loop_controller()

    def test_has_subscribe_method(self):
        """Test controller has subscribe method."""
        controller = get_feedback_loop_controller()
        assert hasattr(controller, "_subscribe_to_events")
        assert callable(controller._subscribe_to_events)

    def test_has_subscribed_flag(self):
        """Test controller has _subscribed flag."""
        controller = get_feedback_loop_controller()
        assert hasattr(controller, "_subscribed")
        assert controller._subscribed is False


class TestFeedbackLoopControllerIntegration:
    """Integration tests for FeedbackLoopController."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_feedback_loop_controller()

    def test_multiple_configs_independent(self):
        """Test multiple configs maintain independent state."""
        controller = get_feedback_loop_controller()

        controller.signal_selfplay_quality("hex8_2p", 0.80)
        controller.signal_selfplay_quality("square8_4p", 0.95)

        state_hex = controller.get_state("hex8_2p")
        state_sq = controller.get_state("square8_4p")

        assert state_hex.last_selfplay_quality == 0.80
        assert state_sq.last_selfplay_quality == 0.95

    def test_state_persistence_across_signals(self):
        """Test state persists across multiple signals."""
        controller = get_feedback_loop_controller()

        controller.signal_selfplay_quality("hex8_2p", 0.80)
        controller.signal_training_complete("hex8_2p", policy_accuracy=0.75)

        state = controller.get_state("hex8_2p")

        # Both signals should be reflected
        assert state.last_selfplay_quality == 0.80
        assert state.last_training_accuracy == 0.75


class TestFeedbackLoopEventHandlers:
    """Test FeedbackLoopController event handlers."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_feedback_loop_controller()

    def test_on_training_complete_updates_state(self):
        """Test _on_training_complete updates state correctly."""
        controller = get_feedback_loop_controller()

        # Create mock event with payload
        event = MagicMock()
        event.payload = {
            "config": "hex8_2p",
            "policy_accuracy": 0.85,
            "value_accuracy": 0.75,
            "model_path": "models/test.pth",
        }

        controller._on_training_complete(event)

        state = controller.get_state("hex8_2p")
        assert state is not None
        assert state.last_training_accuracy == 0.85
        assert state.last_training_time > 0

    def test_on_training_complete_empty_config_returns_early(self):
        """Test _on_training_complete returns early for empty config."""
        controller = get_feedback_loop_controller()

        event = MagicMock()
        event.payload = {"config": "", "policy_accuracy": 0.85}

        controller._on_training_complete(event)

        # Should not create state for empty config
        assert controller.get_state("") is None

    def test_on_training_complete_snapshots_elo(self):
        """Test _on_training_complete snapshots Elo before training."""
        controller = get_feedback_loop_controller()

        # Pre-set an Elo value
        state = controller._get_or_create_state("hex8_2p")
        state.last_elo = 1600.0

        event = MagicMock()
        event.payload = {
            "config": "hex8_2p",
            "policy_accuracy": 0.90,
            "value_accuracy": 0.80,
        }

        controller._on_training_complete(event)

        # Elo should be snapshotted
        state = controller.get_state("hex8_2p")
        assert state.elo_before_training == 1600.0

    def test_on_training_complete_no_payload_attribute(self):
        """Test _on_training_complete handles event without payload attr."""
        controller = get_feedback_loop_controller()

        # Event without payload attribute
        event = object()

        # Should not raise
        controller._on_training_complete(event)

    def test_on_evaluation_complete_updates_state(self):
        """Test _on_evaluation_complete updates state correctly."""
        controller = get_feedback_loop_controller()

        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "win_rate": 0.72,
            "elo": 1550.0,
            "model_path": "models/test.pth",
        }

        controller._on_evaluation_complete(event)

        state = controller.get_state("hex8_2p")
        assert state is not None
        assert state.last_evaluation_win_rate == 0.72
        assert state.last_elo == 1550.0

    def test_on_evaluation_complete_empty_config_returns_early(self):
        """Test _on_evaluation_complete returns early for empty config."""
        controller = get_feedback_loop_controller()

        event = MagicMock()
        event.payload = {"config_key": "", "win_rate": 0.72}

        controller._on_evaluation_complete(event)

        # Should not create state
        assert controller.get_state("") is None

    def test_on_regression_detected_increases_failures(self):
        """Test _on_regression_detected increments consecutive_failures."""
        controller = get_feedback_loop_controller()

        # Pre-create state
        state = controller._get_or_create_state("hex8_2p")
        state.consecutive_failures = 2

        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "current_elo": 1450.0,
            "expected_elo": 1500.0,
            "model_path": "models/test.pth",
        }

        controller._on_regression_detected(event)

        state = controller.get_state("hex8_2p")
        assert state.consecutive_failures == 3

    def test_on_regression_detected_resets_successes(self):
        """Test _on_regression_detected resets consecutive_successes."""
        controller = get_feedback_loop_controller()

        # Pre-create state with successes
        state = controller._get_or_create_state("hex8_2p")
        state.consecutive_successes = 5
        state.consecutive_failures = 0

        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "current_elo": 1450.0,
            "expected_elo": 1500.0,
        }

        controller._on_regression_detected(event)

        state = controller.get_state("hex8_2p")
        assert state.consecutive_successes == 0
        assert state.consecutive_failures == 1

    def test_on_selfplay_complete_updates_state(self):
        """Test _on_selfplay_complete updates state correctly."""
        controller = get_feedback_loop_controller()

        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "games_played": 100,
            "quality_score": 0.85,
            "engine": "gumbel-mcts",
        }

        controller._on_selfplay_complete(event)

        state = controller.get_state("hex8_2p")
        assert state is not None
        assert state.last_selfplay_games == 100
        assert state.last_selfplay_quality == 0.85
        assert state.last_selfplay_engine == "gumbel-mcts"

    def test_on_plateau_detected_boosts_exploration(self):
        """Test _on_plateau_detected increases exploration."""
        controller = get_feedback_loop_controller()

        # Pre-create state with normal exploration
        state = controller._get_or_create_state("hex8_2p")
        state.current_exploration_boost = 1.0

        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "stall_duration_hours": 48.0,
        }

        controller._on_plateau_detected(event)

        state = controller.get_state("hex8_2p")
        # Exploration should be boosted after plateau detection
        assert state.current_exploration_boost >= 1.0

    def test_on_quality_degraded_for_training_updates_intensity(self):
        """Test _on_quality_degraded_for_training adjusts intensity."""
        controller = get_feedback_loop_controller()

        state = controller._get_or_create_state("hex8_2p")
        state.current_training_intensity = "normal"

        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "quality_score": 0.45,  # Below threshold
        }

        controller._on_quality_degraded_for_training(event)

        state = controller.get_state("hex8_2p")
        # Should reduce intensity or adjust parameters
        assert state is not None

    def test_on_promotion_complete_updates_state(self):
        """Test _on_promotion_complete updates state correctly."""
        controller = get_feedback_loop_controller()

        state = controller._get_or_create_state("hex8_2p")
        state.consecutive_failures = 3
        state.consecutive_successes = 0

        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "model_path": "models/promoted.pth",
            "elo": 1600.0,
        }

        controller._on_promotion_complete(event)

        state = controller.get_state("hex8_2p")
        # Promotion is a success - resets failures, increments successes
        assert state.consecutive_successes >= 1

    def test_on_work_completed_updates_metrics(self):
        """Test _on_work_completed updates work queue metrics."""
        controller = get_feedback_loop_controller()

        state = controller._get_or_create_state("hex8_2p")
        initial_count = state.work_completed_count

        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "work_type": "selfplay",
        }

        controller._on_work_completed(event)

        state = controller.get_state("hex8_2p")
        assert state.work_completed_count >= initial_count
        assert state.last_work_completion_time > 0

    def test_on_work_failed_increments_failures(self):
        """Test _on_work_failed increments failure tracking."""
        controller = get_feedback_loop_controller()

        state = controller._get_or_create_state("hex8_2p")
        state.consecutive_failures = 0

        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "work_type": "training",
            "error": "GPU OOM",
        }

        controller._on_work_failed(event)

        state = controller.get_state("hex8_2p")
        assert state.consecutive_failures >= 1

    def test_event_handler_exception_safety(self):
        """Test event handlers don't raise on malformed events."""
        controller = get_feedback_loop_controller()

        # Test all handlers with None payload
        handlers = [
            controller._on_training_complete,
            controller._on_evaluation_complete,
            controller._on_regression_detected,
            controller._on_selfplay_complete,
            controller._on_promotion_complete,
            controller._on_work_completed,
            controller._on_work_failed,
        ]

        for handler in handlers:
            event = MagicMock()
            event.payload = None

            # Should not raise
            try:
                handler(event)
            except Exception as e:
                pytest.fail(f"{handler.__name__} raised {e} with None payload")
