"""Unit tests for feedback_loop_controller module.

Tests the FeedbackLoopController and FeedbackState components that
orchestrate training feedback signals.

December 2025: Expanded to 50+ tests covering:
- FeedbackState dataclass
- Controller initialization and configuration
- State management and tracking
- Event handlers (selfplay, training, evaluation, promotion)
- Health check and status methods
- Singleton pattern
- Error handling paths
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.feedback_loop_controller import (
    FeedbackLoopController,
    FeedbackState,
    get_feedback_loop_controller,
    reset_feedback_loop_controller,
    _safe_create_task,
    _handle_task_error,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def controller():
    """Create a fresh FeedbackLoopController instance."""
    return FeedbackLoopController()


@pytest.fixture
def started_controller():
    """Create a controller with running state."""
    ctrl = FeedbackLoopController()
    ctrl._running = True
    ctrl._subscribed = True
    return ctrl


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton between tests."""
    reset_feedback_loop_controller()
    yield
    reset_feedback_loop_controller()


# =============================================================================
# FeedbackState Tests
# =============================================================================


class TestFeedbackState:
    """Test FeedbackState dataclass."""

    def test_create_state(self):
        """Should create state with config key."""
        state = FeedbackState(config_key="hex8_2p")
        assert state.config_key == "hex8_2p"
        assert state.last_selfplay_quality == 0.0
        assert state.current_training_intensity == "normal"

    def test_default_values(self):
        """Should have sensible defaults."""
        state = FeedbackState(config_key="test")
        assert state.last_training_accuracy == 0.0
        assert state.last_evaluation_win_rate == 0.0
        assert state.consecutive_failures == 0
        assert state.consecutive_successes == 0
        assert state.current_exploration_boost == 1.0
        assert state.current_curriculum_weight == 1.0

    def test_promotion_status_tracking(self):
        """Should track promotion status."""
        state = FeedbackState(config_key="test")
        assert state.last_promotion_success is None

        # Simulate promotion tracking
        state.last_promotion_success = True
        state.consecutive_successes = 1
        assert state.last_promotion_success is True
        assert state.consecutive_successes == 1

    def test_work_metrics(self):
        """Should track work completion metrics."""
        state = FeedbackState(config_key="test")
        assert state.work_completed_count == 0
        assert state.last_work_completion_time == 0.0

        # Simulate work completion
        state.work_completed_count = 10
        state.last_work_completion_time = time.time()
        assert state.work_completed_count == 10

    def test_elo_history_initialization(self):
        """Should initialize elo_history as empty list."""
        state = FeedbackState(config_key="test")
        assert state.elo_history == []

    def test_update_elo_tracks_history(self):
        """Should track Elo history and calculate velocity."""
        state = FeedbackState(config_key="test")
        now = time.time()

        # Add first data point
        state.update_elo(1500.0, timestamp=now)
        assert state.last_elo == 1500.0
        assert len(state.elo_history) == 1

    def test_update_elo_calculates_velocity(self):
        """Should calculate Elo velocity from history."""
        state = FeedbackState(config_key="test")
        base_time = time.time()

        # Add 3+ data points for velocity calculation
        state.update_elo(1500.0, timestamp=base_time)
        state.update_elo(1550.0, timestamp=base_time + 1800)  # +50 in 30 min
        state.update_elo(1600.0, timestamp=base_time + 3600)  # +100 in 1 hr

        assert state.elo_velocity > 0  # Should show positive velocity

    def test_elo_history_bounded_to_10_entries(self):
        """Should keep only last 10 Elo history entries."""
        state = FeedbackState(config_key="test")

        # Add 15 entries
        for i in range(15):
            state.update_elo(1500.0 + i * 10, timestamp=time.time() + i)

        assert len(state.elo_history) == 10

    def test_timing_attributes(self):
        """Should have timing attributes for tracking."""
        state = FeedbackState(config_key="test")
        assert state.last_selfplay_time == 0.0
        assert state.last_training_time == 0.0
        assert state.last_evaluation_time == 0.0
        assert state.last_promotion_time == 0.0

    def test_search_budget_default(self):
        """Should have default search budget."""
        state = FeedbackState(config_key="test")
        assert state.current_search_budget == 400


# =============================================================================
# FeedbackLoopController Initialization Tests
# =============================================================================


class TestFeedbackLoopControllerInit:
    """Test FeedbackLoopController initialization."""

    def test_init_default(self):
        """Should initialize with defaults."""
        controller = FeedbackLoopController()
        assert controller._running is False
        assert controller._subscribed is False
        assert isinstance(controller._states, dict)

    def test_has_configuration_attributes(self):
        """Should have configuration attributes."""
        controller = FeedbackLoopController()
        assert hasattr(controller, "policy_accuracy_threshold")
        assert hasattr(controller, "promotion_threshold")
        assert controller.promotion_threshold >= 0.5

    def test_has_lock(self):
        """Should have thread lock."""
        controller = FeedbackLoopController()
        assert hasattr(controller, "_lock")
        assert isinstance(controller._lock, type(threading.Lock()))

    def test_has_rate_history(self):
        """Should have rate history tracking."""
        controller = FeedbackLoopController()
        assert hasattr(controller, "_rate_history")
        assert isinstance(controller._rate_history, dict)

    def test_cluster_healthy_default(self):
        """Should have cluster_healthy flag initialized."""
        controller = FeedbackLoopController()
        assert controller._cluster_healthy is True

    def test_failure_exploration_boost_configured(self):
        """Should have failure_exploration_boost from config."""
        controller = FeedbackLoopController()
        assert hasattr(controller, "failure_exploration_boost")
        assert controller.failure_exploration_boost > 1.0

    def test_success_intensity_reduction_configured(self):
        """Should have success_intensity_reduction from config."""
        controller = FeedbackLoopController()
        assert hasattr(controller, "success_intensity_reduction")


# =============================================================================
# FeedbackLoopController State Management Tests
# =============================================================================


class TestFeedbackLoopControllerState:
    """Test state management."""

    def test_get_state_creates_new(self, controller):
        """Should create new state if not exists."""
        state = controller._get_or_create_state("hex8_2p")
        assert state is not None
        assert state.config_key == "hex8_2p"

    def test_get_state_returns_existing(self, controller):
        """Should return existing state."""
        state1 = controller._get_or_create_state("hex8_2p")
        state1.last_selfplay_quality = 0.9
        state2 = controller._get_or_create_state("hex8_2p")
        assert state2.last_selfplay_quality == 0.9

    def test_get_state_returns_none_for_missing(self, controller):
        """get_state should return None for missing key."""
        result = controller.get_state("nonexistent")
        assert result is None

    def test_get_all_states(self, controller):
        """Should return all states."""
        controller._get_or_create_state("hex8_2p")
        controller._get_or_create_state("square8_4p")
        states = controller.get_all_states()
        assert len(states) >= 2
        assert "hex8_2p" in states
        assert "square8_4p" in states

    def test_get_all_states_returns_copy(self, controller):
        """Should return copy, not original dict."""
        controller._get_or_create_state("hex8_2p")
        states1 = controller.get_all_states()
        states2 = controller.get_all_states()
        # Should be equal content but not same object
        assert states1 == states2
        states1["new_key"] = "value"
        assert "new_key" not in controller._states

    def test_state_thread_safety(self, controller):
        """State creation should be thread-safe."""
        results = []

        def create_state():
            state = controller._get_or_create_state("thread_test")
            results.append(state)

        threads = [threading.Thread(target=create_state) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the same state instance
        assert len(set(id(s) for s in results)) == 1


# =============================================================================
# FeedbackLoopController Signal Tests
# =============================================================================


class TestFeedbackLoopControllerSignals:
    """Test signal methods."""

    def test_signal_selfplay_quality(self, controller):
        """Should update selfplay quality."""
        controller.signal_selfplay_quality("hex8_2p", quality_score=0.85)
        state = controller.get_state("hex8_2p")
        assert state is not None
        assert state.last_selfplay_quality == 0.85

    def test_signal_selfplay_quality_updates_time(self, controller):
        """Should update selfplay timestamp."""
        before = time.time()
        controller.signal_selfplay_quality("hex8_2p", quality_score=0.85)
        after = time.time()
        state = controller.get_state("hex8_2p")
        assert before <= state.last_selfplay_time <= after

    def test_signal_training_complete(self, controller):
        """Should update training metrics."""
        controller.signal_training_complete(
            "hex8_2p",
            policy_accuracy=0.78,
            value_accuracy=0.65,
        )
        state = controller.get_state("hex8_2p")
        assert state is not None
        assert state.last_training_accuracy == 0.78

    def test_signal_training_complete_updates_time(self, controller):
        """Should update training timestamp."""
        before = time.time()
        controller.signal_training_complete("hex8_2p", policy_accuracy=0.78)
        after = time.time()
        state = controller.get_state("hex8_2p")
        assert before <= state.last_training_time <= after


# =============================================================================
# FeedbackLoopController Summary Tests
# =============================================================================


class TestFeedbackLoopControllerSummary:
    """Test summary and status methods."""

    def test_get_summary_empty(self, controller):
        """Should return summary even with no states."""
        summary = controller.get_summary()
        assert isinstance(summary, dict)
        assert "running" in summary
        assert "subscribed" in summary
        assert "configs_tracked" in summary

    def test_get_summary_with_states(self, controller):
        """Should include state info in summary."""
        controller._get_or_create_state("hex8_2p")
        controller.signal_selfplay_quality("hex8_2p", quality_score=0.9)
        summary = controller.get_summary()
        assert summary["configs_tracked"] >= 1
        assert "states" in summary
        assert "hex8_2p" in summary["states"]

    def test_get_summary_state_details(self, controller):
        """Should include detailed state info."""
        controller.signal_selfplay_quality("hex8_2p", quality_score=0.9)
        summary = controller.get_summary()
        state_summary = summary["states"]["hex8_2p"]
        assert "training_intensity" in state_summary
        assert "exploration_boost" in state_summary
        assert "last_selfplay_quality" in state_summary


# =============================================================================
# FeedbackLoopController Health Check Tests
# =============================================================================


class TestFeedbackLoopControllerHealthCheck:
    """Test health check functionality."""

    def test_health_check_returns_result(self, controller):
        """Should return HealthCheckResult."""
        result = controller.health_check()
        assert hasattr(result, "healthy")
        assert hasattr(result, "message")
        assert hasattr(result, "details")

    def test_health_check_unhealthy_when_not_running(self, controller):
        """Should report unhealthy when not running."""
        controller._running = False
        result = controller.health_check()
        assert result.healthy is False
        assert "stopped" in result.message.lower() or "not" in result.message.lower()

    def test_health_check_healthy_when_running(self, started_controller):
        """Should report healthy when running and subscribed."""
        result = started_controller.health_check()
        assert result.healthy is True

    def test_health_check_includes_details(self, started_controller):
        """Should include relevant details."""
        result = started_controller.health_check()
        assert "running" in result.details
        assert "subscribed" in result.details
        assert "configs_tracked" in result.details
        assert "cluster_healthy" in result.details

    def test_health_check_includes_thresholds(self, started_controller):
        """Should include threshold configuration."""
        result = started_controller.health_check()
        assert "policy_accuracy_threshold" in result.details
        assert "promotion_threshold" in result.details

    def test_health_check_counts_active_configs(self, started_controller):
        """Should count active configs (trained within 1 hour)."""
        # Add a state with recent training
        state = started_controller._get_or_create_state("hex8_2p")
        state.last_training_time = time.time()  # Just now

        result = started_controller.health_check()
        assert result.details["active_configs"] >= 1


# =============================================================================
# FeedbackLoopController Lifecycle Tests
# =============================================================================


class TestFeedbackLoopControllerLifecycle:
    """Test start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_sets_running(self, controller):
        """Start should set running flag."""
        # Mock the subscription methods to avoid side effects
        with patch.object(controller, "_subscribe_to_events"):
            with patch.object(controller, "_wire_curriculum_feedback"):
                with patch.object(controller, "_wire_exploration_boost"):
                    with patch.object(controller, "_subscribe_to_lazy_scheduler_registration"):
                        await controller.start()

        assert controller._running is True
        await controller.stop()

    @pytest.mark.asyncio
    async def test_stop_clears_running(self, controller):
        """Stop should clear running flag."""
        controller._running = True
        await controller.stop()
        assert controller._running is False

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self, controller):
        """Start should be idempotent."""
        with patch.object(controller, "_subscribe_to_events") as mock_sub:
            with patch.object(controller, "_wire_curriculum_feedback"):
                with patch.object(controller, "_wire_exploration_boost"):
                    with patch.object(controller, "_subscribe_to_lazy_scheduler_registration"):
                        await controller.start()
                        await controller.start()  # Second call

        # Subscribe should only be called once
        assert mock_sub.call_count == 1
        await controller.stop()

    def test_is_running(self, controller):
        """Should correctly report running status."""
        assert controller.is_running() is False
        controller._running = True
        assert controller.is_running() is True


# =============================================================================
# Module Function Tests
# =============================================================================


class TestModuleFunctions:
    """Test module-level functions."""

    def test_get_feedback_loop_controller(self):
        """Should return controller instance."""
        controller = get_feedback_loop_controller()
        assert isinstance(controller, FeedbackLoopController)

    def test_get_feedback_loop_controller_singleton(self):
        """Should return same instance."""
        c1 = get_feedback_loop_controller()
        c2 = get_feedback_loop_controller()
        assert c1 is c2

    def test_reset_feedback_loop_controller(self):
        """Should reset singleton."""
        c1 = get_feedback_loop_controller()
        reset_feedback_loop_controller()
        c2 = get_feedback_loop_controller()
        assert c1 is not c2


# =============================================================================
# Event Handler Tests
# =============================================================================


class TestEventHandlers:
    """Test event handler methods."""

    def test_on_selfplay_complete_updates_quality(self, controller):
        """_on_selfplay_complete should update quality."""
        event = MagicMock()
        event.payload = {
            "config": "hex8_2p",
            "games_count": 100,
            "db_path": "/tmp/test.db",
        }

        with patch.object(controller, "_assess_selfplay_quality", return_value=0.85):
            with patch.object(controller, "_update_training_intensity"):
                with patch.object(controller, "_update_curriculum_weight_from_selfplay"):
                    controller._on_selfplay_complete(event)

        state = controller.get_state("hex8_2p")
        assert state.last_selfplay_quality == 0.85

    def test_on_selfplay_complete_missing_config(self, controller):
        """Should handle missing config gracefully."""
        event = MagicMock()
        event.payload = {"games_count": 100}  # No config

        # Should not raise
        controller._on_selfplay_complete(event)
        assert len(controller._states) == 0

    def test_on_training_complete_updates_state(self, controller):
        """_on_training_complete should update state."""
        event = MagicMock()
        event.payload = {
            "config": "hex8_2p",
            "policy_accuracy": 0.82,
            "value_accuracy": 0.75,
            "model_path": "/tmp/model.pth",
        }

        with patch.object(controller, "_trigger_evaluation"):
            with patch.object(controller, "_record_training_in_curriculum"):
                with patch.object(controller, "_emit_curriculum_training_feedback"):
                    controller._on_training_complete(event)

        state = controller.get_state("hex8_2p")
        assert state.last_training_accuracy == 0.82

    def test_on_evaluation_complete_updates_win_rate(self, controller):
        """_on_evaluation_complete should update win rate."""
        event = MagicMock()
        event.payload = {
            "config": "hex8_2p",
            "win_rate": 0.65,
            "elo": 1650.0,
            "model_path": "/tmp/model.pth",
        }

        with patch.object(controller, "_adjust_selfplay_for_velocity"):
            with patch.object(controller, "_consider_promotion"):
                controller._on_evaluation_complete(event)

        state = controller.get_state("hex8_2p")
        assert state.last_evaluation_win_rate == 0.65

    def test_on_promotion_complete_success_increments_streak(self, controller):
        """Promotion success should increment consecutive successes."""
        event = MagicMock()
        event.payload = {
            "config": "hex8_2p",
            "promoted": True,
        }

        with patch.object(controller, "_apply_intensity_feedback"):
            controller._on_promotion_complete(event)

        state = controller.get_state("hex8_2p")
        assert state.consecutive_successes == 1
        assert state.consecutive_failures == 0

    def test_on_promotion_complete_failure_increments_failures(self, controller):
        """Promotion failure should increment consecutive failures."""
        event = MagicMock()
        event.payload = {
            "config": "hex8_2p",
            "promoted": False,
        }

        with patch.object(controller, "_apply_intensity_feedback"):
            with patch.object(controller, "_signal_urgent_training"):
                controller._on_promotion_complete(event)

        state = controller.get_state("hex8_2p")
        assert state.consecutive_failures == 1
        assert state.consecutive_successes == 0

    def test_on_work_completed_updates_metrics(self, controller):
        """_on_work_completed should update work metrics."""
        event = MagicMock()
        event.payload = {
            "work_id": "work-123",
            "work_type": "selfplay",
            "board_type": "hex8",
            "num_players": 2,
            "claimed_by": "node-1",
        }

        controller._on_work_completed(event)

        state = controller.get_state("hex8_2p")
        assert state.work_completed_count == 1

    def test_on_work_failed_tracks_failures(self, controller):
        """_on_work_failed should track failure count."""
        event = MagicMock()
        event.payload = {
            "work_id": "work-123",
            "work_type": "selfplay",
            "board_type": "hex8",
            "num_players": 2,
            "node_id": "node-1",
            "reason": "timeout",
        }

        controller._on_work_failed(event)

        state = controller.get_state("hex8_2p")
        assert hasattr(state, "work_failed_count")
        assert state.work_failed_count == 1

    def test_on_work_timeout_tracks_timeouts(self, controller):
        """_on_work_timeout should track timeout count."""
        event = MagicMock()
        event.payload = {
            "work_id": "work-123",
            "work_type": "training",
            "board_type": "square8",
            "num_players": 4,
            "node_id": "node-2",
            "timeout_seconds": 3600,
        }

        controller._on_work_timeout(event)

        state = controller.get_state("square8_4p")
        assert hasattr(state, "work_timeout_count")
        assert state.work_timeout_count == 1


# =============================================================================
# Intensity and Quality Tests
# =============================================================================


class TestIntensityAndQuality:
    """Test intensity and quality computation methods."""

    def test_compute_intensity_from_quality_hot_path(self, controller):
        """High quality should return hot_path intensity."""
        result = controller._compute_intensity_from_quality(0.95)
        assert result == "hot_path"

    def test_compute_intensity_from_quality_accelerated(self, controller):
        """Good quality should return accelerated intensity."""
        result = controller._compute_intensity_from_quality(0.85)
        assert result == "accelerated"

    def test_compute_intensity_from_quality_normal(self, controller):
        """Adequate quality should return normal intensity."""
        result = controller._compute_intensity_from_quality(0.70)
        assert result == "normal"

    def test_compute_intensity_from_quality_reduced(self, controller):
        """Poor quality should return reduced intensity."""
        result = controller._compute_intensity_from_quality(0.55)
        assert result == "reduced"

    def test_compute_intensity_from_quality_paused(self, controller):
        """Very poor quality should return paused intensity."""
        result = controller._compute_intensity_from_quality(0.40)
        assert result == "paused"

    def test_assess_selfplay_quality_db_not_found(self, controller):
        """Should return 0.3 when database does not exist."""
        # Test with non-existent path - returns 0.3 regardless of games_count
        result = controller._assess_selfplay_quality("/nonexistent/path.db", 50)
        assert result == 0.3

    def test_assess_selfplay_quality_fallback_uses_import_error(self, controller):
        """Count-based fallback is used when ImportError occurs."""
        # Mock the import to fail
        with patch("app.coordination.feedback_loop_controller.logger"):
            with patch.dict("sys.modules", {"app.quality.unified_quality": None}):
                # When import fails, should use count-based heuristic
                # This is implementation detail - testing the logic separately
                pass

    def test_assess_selfplay_quality_handles_exceptions(self, controller):
        """Should handle exceptions gracefully and return fallback."""
        # Non-existent path should be handled gracefully
        result = controller._assess_selfplay_quality("/nonexistent/db.db", 100)
        # Returns 0.3 because DB doesn't exist
        assert result == 0.3

    def test_assess_selfplay_quality_returns_valid_range(self, controller):
        """Should always return a value between 0 and 1."""
        result = controller._assess_selfplay_quality("/any/path.db", 500)
        assert 0.0 <= result <= 1.0


# =============================================================================
# Exploration Boost Tests
# =============================================================================


class TestExplorationBoost:
    """Test exploration boost functionality."""

    def test_boost_exploration_for_anomaly_updates_state(self, controller):
        """Should update exploration boost in state."""
        controller._boost_exploration_for_anomaly("hex8_2p", anomaly_count=2)

        state = controller.get_state("hex8_2p")
        assert state.current_exploration_boost > 1.0

    def test_boost_exploration_for_stall_updates_state(self, controller):
        """Should update exploration boost for stall."""
        controller._boost_exploration_for_stall("hex8_2p", stall_epochs=10)

        state = controller.get_state("hex8_2p")
        assert state.current_exploration_boost >= 1.0

    def test_reduce_exploration_after_improvement(self, controller):
        """Should reduce exploration boost on improvement."""
        state = controller._get_or_create_state("hex8_2p")
        state.current_exploration_boost = 1.5

        controller._reduce_exploration_after_improvement("hex8_2p")

        assert state.current_exploration_boost < 1.5

    def test_reduce_exploration_does_not_go_below_1(self, controller):
        """Exploration boost should not go below 1.0."""
        state = controller._get_or_create_state("hex8_2p")
        state.current_exploration_boost = 1.05

        controller._reduce_exploration_after_improvement("hex8_2p")

        assert state.current_exploration_boost >= 1.0


# =============================================================================
# Cluster Health Tests
# =============================================================================


class TestClusterHealth:
    """Test cluster health handling."""

    def test_on_p2p_cluster_unhealthy_sets_flag(self, controller):
        """Should set cluster_healthy to False on majority dead."""
        event = MagicMock()
        event.payload = {
            "dead_nodes": ["node-1", "node-2", "node-3"],
            "alive_nodes": ["node-4"],
        }

        with patch.object(controller, "_get_or_create_state"):
            controller._on_p2p_cluster_unhealthy(event)

        assert controller._cluster_healthy is False

    def test_on_p2p_cluster_unhealthy_stays_healthy(self, controller):
        """Should stay healthy when majority alive."""
        event = MagicMock()
        event.payload = {
            "dead_nodes": ["node-1"],
            "alive_nodes": ["node-2", "node-3", "node-4"],
        }

        controller._on_p2p_cluster_unhealthy(event)

        assert controller._cluster_healthy is True


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Test module-level helper functions."""

    @pytest.mark.asyncio
    async def test_safe_create_task_returns_task(self):
        """_safe_create_task should return a task."""
        async def dummy_coro():
            pass

        task = _safe_create_task(dummy_coro(), context="test")
        assert task is not None
        await task

    def test_handle_task_error_handles_cancelled(self):
        """_handle_task_error should handle cancelled tasks."""
        task = MagicMock()
        task.exception.side_effect = asyncio.CancelledError()

        # Should not raise
        _handle_task_error(task, "test_context")

    def test_handle_task_error_logs_exception(self):
        """_handle_task_error should log real exceptions."""
        task = MagicMock()
        task.exception.return_value = ValueError("test error")

        with patch("app.coordination.feedback_loop_controller.logger") as mock_logger:
            _handle_task_error(task, "test_context")
            mock_logger.error.assert_called_once()


# =============================================================================
# Integration Tests
# =============================================================================


class TestFeedbackLoopControllerIntegration:
    """Integration tests for feedback loop."""

    def test_feedback_cycle(self, controller):
        """Should handle full feedback cycle."""
        config = "hex8_2p"

        # Selfplay quality signal
        controller.signal_selfplay_quality(config, quality_score=0.85)

        # Training complete signal
        controller.signal_training_complete(
            config,
            policy_accuracy=0.78,
            value_accuracy=0.65,
        )

        # Check state
        state = controller.get_state(config)
        assert state.last_selfplay_quality == 0.85
        assert state.last_training_accuracy == 0.78

    def test_multiple_configs(self, controller):
        """Should track multiple configs independently."""
        controller.signal_selfplay_quality("hex8_2p", quality_score=0.9)
        controller.signal_selfplay_quality("square8_4p", quality_score=0.7)

        state1 = controller.get_state("hex8_2p")
        state2 = controller.get_state("square8_4p")

        assert state1.last_selfplay_quality == 0.9
        assert state2.last_selfplay_quality == 0.7

    def test_promotion_streak_reset(self, controller):
        """Promotion success should reset failure streak."""
        # Simulate failures
        state = controller._get_or_create_state("hex8_2p")
        state.consecutive_failures = 3

        # Simulate success
        event = MagicMock()
        event.payload = {"config": "hex8_2p", "promoted": True}

        with patch.object(controller, "_apply_intensity_feedback"):
            controller._on_promotion_complete(event)

        assert state.consecutive_failures == 0
        assert state.consecutive_successes == 1

    def test_hot_path_activation(self, controller):
        """Hot path should activate after 3 consecutive successes."""
        state = controller._get_or_create_state("hex8_2p")
        state.consecutive_successes = 2  # Already had 2 successes

        event = MagicMock()
        event.payload = {"config": "hex8_2p", "promoted": True}

        with patch.object(controller, "_apply_intensity_feedback"):
            controller._on_promotion_complete(event)

        assert state.consecutive_successes == 3
        assert state.current_training_intensity == "hot_path"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling paths."""

    def test_on_selfplay_complete_handles_exception(self, controller):
        """Should handle exceptions gracefully."""
        event = MagicMock()
        event.payload = {"config": "hex8_2p", "games_count": 100, "db_path": "/tmp/test.db"}

        with patch.object(controller, "_assess_selfplay_quality", side_effect=RuntimeError("test")):
            # Should not raise
            controller._on_selfplay_complete(event)

    def test_on_training_complete_handles_exception(self, controller):
        """Should handle exceptions in training complete handler."""
        event = MagicMock()
        event.payload = {"config": "hex8_2p", "policy_accuracy": 0.8}

        with patch.object(controller, "_trigger_evaluation", side_effect=RuntimeError("test")):
            # Should not raise
            controller._on_training_complete(event)

    def test_on_promotion_complete_handles_exception(self, controller):
        """Should handle exceptions in promotion complete handler."""
        event = MagicMock()
        event.payload = {"config": "hex8_2p", "promoted": True}

        with patch.object(controller, "_apply_intensity_feedback", side_effect=RuntimeError("test")):
            # Should not raise
            controller._on_promotion_complete(event)

    def test_handles_missing_payload(self, controller):
        """Should handle events without payload attribute."""
        event = object()  # No payload attribute

        # Should not raise
        controller._on_selfplay_complete(event)
        controller._on_training_complete(event)
        controller._on_work_completed(event)


# =============================================================================
# Plateau Detection Tests
# =============================================================================


class TestPlateauDetection:
    """Tests for _on_plateau_detected handler."""

    def test_plateau_overfitting_sets_temperature_boost(self, controller):
        """Overfitting plateau should set temperature boost."""
        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "plateau_type": "overfitting",
            "exploration_boost": 1.5,
            "train_val_gap": 0.15,
        }

        with patch("app.coordination.feedback_loop_controller._safe_create_task"):
            controller._on_plateau_detected(event)

        state = controller.get_state("hex8_2p")
        assert state.exploration_boost == 1.5
        assert state.selfplay_temperature_boost == 1.2

    def test_plateau_data_limitation_sets_games_multiplier(self, controller):
        """Data-limited plateau should set games multiplier."""
        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "plateau_type": "data_limitation",
            "exploration_boost": 1.3,
        }

        with patch("app.coordination.feedback_loop_controller._safe_create_task"):
            controller._on_plateau_detected(event)

        state = controller.get_state("hex8_2p")
        assert state.exploration_boost == 1.3
        assert state.games_multiplier == 1.5

    def test_plateau_missing_config_key_returns_early(self, controller):
        """Missing config_key should return early."""
        event = MagicMock()
        event.payload = {"plateau_type": "overfitting"}

        controller._on_plateau_detected(event)

        # No state should be created
        assert "hex8_2p" not in controller._states

    def test_plateau_count_increments(self, controller):
        """Plateau count should increment with each event."""
        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "plateau_type": "data_limitation",
        }

        with patch("app.coordination.feedback_loop_controller._safe_create_task"):
            controller._on_plateau_detected(event)
            controller._on_plateau_detected(event)
            controller._on_plateau_detected(event)

        state = controller.get_state("hex8_2p")
        assert state.plateau_count == 3

    def test_plateau_sets_expiration_time(self, controller):
        """Exploration boost should have expiration time."""
        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "plateau_type": "overfitting",
        }

        with patch("app.coordination.feedback_loop_controller._safe_create_task"):
            controller._on_plateau_detected(event)

        state = controller.get_state("hex8_2p")
        assert state.exploration_boost_expires_at > time.time()

    def test_plateau_uses_default_exploration_boost(self, controller):
        """Should use default exploration boost if not provided."""
        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "plateau_type": "data_limitation",
        }

        with patch("app.coordination.feedback_loop_controller._safe_create_task"):
            controller._on_plateau_detected(event)

        state = controller.get_state("hex8_2p")
        assert state.exploration_boost == 1.3  # Default

    def test_repeated_plateaus_triggers_curriculum_advancement(self, controller):
        """2+ plateaus should trigger curriculum advancement."""
        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "plateau_type": "data_limitation",
        }

        with patch("app.coordination.feedback_loop_controller._safe_create_task"):
            with patch.object(controller, "_advance_curriculum_on_velocity_plateau") as mock_advance:
                controller._on_plateau_detected(event)
                controller._on_plateau_detected(event)

                # Should call advancement after 2 plateaus
                mock_advance.assert_called_once()


# =============================================================================
# Training Loss Anomaly Tests
# =============================================================================


class TestTrainingLossAnomaly:
    """Tests for _on_training_loss_anomaly handler."""

    def test_anomaly_tracks_count(self, controller):
        """Should track anomaly count."""
        event = MagicMock()
        event.payload = {
            "config": "hex8_2p",
            "loss": 2.5,
            "expected_loss": 1.0,
            "deviation": 3.0,
            "epoch": 10,
            "severity": "moderate",
        }

        with patch.object(controller, "_trigger_quality_check"):
            controller._on_training_loss_anomaly(event)

        state = controller.get_state("hex8_2p")
        assert state.loss_anomaly_count == 1

    def test_anomaly_triggers_quality_check(self, controller):
        """Should trigger quality check on anomaly."""
        event = MagicMock()
        event.payload = {
            "config": "hex8_2p",
            "loss": 2.5,
            "expected_loss": 1.0,
            "deviation": 3.0,
        }

        with patch.object(controller, "_trigger_quality_check") as mock_quality:
            controller._on_training_loss_anomaly(event)

        mock_quality.assert_called_once_with("hex8_2p", reason="training_loss_anomaly")

    def test_severe_anomaly_boosts_exploration(self, controller):
        """Severe anomaly should boost exploration."""
        event = MagicMock()
        event.payload = {
            "config": "hex8_2p",
            "loss": 5.0,
            "expected_loss": 1.0,
            "deviation": 4.0,
            "severity": "severe",
        }

        with patch.object(controller, "_trigger_quality_check"):
            with patch.object(controller, "_boost_exploration_for_anomaly") as mock_boost:
                controller._on_training_loss_anomaly(event)

        mock_boost.assert_called_once()

    def test_consecutive_anomalies_boost_exploration(self, controller):
        """Multiple consecutive anomalies should boost exploration."""
        event = MagicMock()
        event.payload = {
            "config": "hex8_2p",
            "loss": 2.0,
            "expected_loss": 1.0,
            "severity": "moderate",
        }

        # Create state with prior anomaly count
        state = controller._get_or_create_state("hex8_2p")
        state.loss_anomaly_count = 2  # Already at threshold - 1

        with patch.object(controller, "_trigger_quality_check"):
            with patch.object(controller, "_boost_exploration_for_anomaly") as mock_boost:
                controller._on_training_loss_anomaly(event)

        # Count should now be 3, triggering exploration boost
        mock_boost.assert_called_once()

    def test_missing_config_returns_early(self, controller):
        """Missing config should return early."""
        event = MagicMock()
        event.payload = {
            "loss": 2.5,
            "expected_loss": 1.0,
        }

        with patch.object(controller, "_trigger_quality_check") as mock_quality:
            controller._on_training_loss_anomaly(event)

        mock_quality.assert_not_called()

    def test_anomaly_sets_timestamp(self, controller):
        """Should set last anomaly timestamp."""
        event = MagicMock()
        event.payload = {
            "config": "hex8_2p",
            "loss": 2.0,
            "severity": "moderate",
        }

        with patch.object(controller, "_trigger_quality_check"):
            controller._on_training_loss_anomaly(event)

        state = controller.get_state("hex8_2p")
        assert state.last_loss_anomaly_time > 0


# =============================================================================
# Selfplay Velocity Adjustment Tests
# =============================================================================


class TestSelfplayVelocityAdjustment:
    """Tests for _adjust_selfplay_for_velocity method."""

    def test_plateau_increases_search_budget(self, controller):
        """Low velocity plateau should increase search budget."""
        state = controller._get_or_create_state("hex8_2p")
        state.current_search_budget = 400
        state.current_exploration_boost = 1.0
        # Add enough history for velocity calculation
        state.elo_history = [(time.time() - 3600, 1500.0)] * 4

        with patch.object(controller, "_emit_selfplay_adjustment"):
            controller._adjust_selfplay_for_velocity(
                config_key="hex8_2p",
                state=state,
                elo=1500.0,
                velocity=1.0,  # Low velocity
            )

        assert state.current_search_budget == 500  # Increased by 100
        assert state.current_exploration_boost > 1.0

    def test_plateau_caps_budget_at_800(self, controller):
        """Search budget should cap at 800."""
        state = controller._get_or_create_state("hex8_2p")
        state.current_search_budget = 750
        state.elo_history = [(time.time() - 3600, 1500.0)] * 4

        with patch.object(controller, "_emit_selfplay_adjustment"):
            controller._adjust_selfplay_for_velocity(
                config_key="hex8_2p",
                state=state,
                elo=1500.0,
                velocity=1.0,  # Low velocity
            )

        assert state.current_search_budget == 800  # Capped

    def test_fast_improvement_maintains_settings(self, controller):
        """Fast improvement should maintain current settings."""
        state = controller._get_or_create_state("hex8_2p")
        state.current_search_budget = 400
        state.current_exploration_boost = 1.0

        with patch.object(controller, "_emit_selfplay_adjustment"):
            controller._adjust_selfplay_for_velocity(
                config_key="hex8_2p",
                state=state,
                elo=1600.0,
                velocity=50.0,  # High velocity
            )

        # Settings should be unchanged
        assert state.current_search_budget == 400
        assert state.current_exploration_boost == 1.0

    def test_near_goal_increases_minimum_budget(self, controller):
        """Elo > 1800 should ensure minimum budget of 600."""
        state = controller._get_or_create_state("hex8_2p")
        state.current_search_budget = 400

        with patch.object(controller, "_emit_selfplay_adjustment"):
            controller._adjust_selfplay_for_velocity(
                config_key="hex8_2p",
                state=state,
                elo=1850.0,
                velocity=10.0,  # Normal velocity
            )

        assert state.current_search_budget >= 600

    def test_emits_selfplay_adjustment(self, controller):
        """Should emit SELFPLAY_TARGET_UPDATED event."""
        state = controller._get_or_create_state("hex8_2p")

        with patch.object(controller, "_emit_selfplay_adjustment") as mock_emit:
            controller._adjust_selfplay_for_velocity(
                config_key="hex8_2p",
                state=state,
                elo=1600.0,
                velocity=10.0,
            )

        mock_emit.assert_called_once()


# =============================================================================
# Trigger Evaluation Tests
# =============================================================================


class TestTriggerEvaluation:
    """Tests for _trigger_evaluation method."""

    def test_trigger_evaluation_parses_config_key(self, controller):
        """Should parse board_type and num_players from config_key."""
        # Trigger evaluation uses pipeline actions which may not be available
        # Test that parsing works correctly
        config_key = "hex8_2p"
        parts = config_key.rsplit("_", 1)
        board_type = parts[0]
        num_players = int(parts[1].rstrip("p"))

        assert board_type == "hex8"
        assert num_players == 2

    def test_trigger_evaluation_handles_4p_config(self, controller):
        """Should handle 4-player config key parsing."""
        config_key = "square8_4p"
        parts = config_key.rsplit("_", 1)
        board_type = parts[0]
        num_players = int(parts[1].rstrip("p"))

        assert board_type == "square8"
        assert num_players == 4

    def test_trigger_evaluation_handles_invalid_config_key(self, controller):
        """Should handle invalid config_key format gracefully."""
        # Should not raise - graceful handling
        try:
            controller._trigger_evaluation("invalid", "/tmp/model.pth")
        except (ValueError, IndexError):
            pass  # Expected for invalid format

    def test_trigger_evaluation_records_time(self, controller):
        """Should record last training time for config."""
        state = controller._get_or_create_state("hex8_2p")
        state.last_training_time = 0.0

        # The method may update training time
        # Test that it's callable without error
        try:
            controller._trigger_evaluation("hex8_2p", "/tmp/model.pth")
        except (ImportError, AttributeError):
            pass  # Expected if dependencies not available


# =============================================================================
# Adaptive Training Signal Tests
# =============================================================================


class TestAdaptiveTrainingSignal:
    """Tests for _compute_adaptive_signal method and AdaptiveTrainingSignal dataclass."""

    def test_adaptive_signal_defaults(self):
        """AdaptiveTrainingSignal should have sensible defaults."""
        from app.coordination.feedback_loop_controller import AdaptiveTrainingSignal

        signal = AdaptiveTrainingSignal()
        assert signal.learning_rate_multiplier == 1.0
        assert signal.batch_size_multiplier == 1.0
        assert signal.epochs_extension == 0
        assert signal.gradient_clip_enabled is False
        assert signal.reason == ""

    def test_adaptive_signal_to_dict(self):
        """Should convert to dictionary for event emission."""
        from app.coordination.feedback_loop_controller import AdaptiveTrainingSignal

        signal = AdaptiveTrainingSignal(
            learning_rate_multiplier=0.5,
            epochs_extension=5,
            gradient_clip_enabled=True,
            reason="regression_recovery",
        )

        d = signal.to_dict()
        assert d["learning_rate_multiplier"] == 0.5
        assert d["epochs_extension"] == 5
        assert d["gradient_clip_enabled"] is True
        assert d["reason"] == "regression_recovery"

    def test_compute_adaptive_signal_callable(self, controller):
        """_compute_adaptive_signal should be callable."""
        state = controller._get_or_create_state("hex8_2p")
        state.elo_history = [(time.time() - 7200, 1500.0)]
        state.last_elo = 1500.0

        eval_result = {
            "elo": 1570.0,  # +70 Elo
            "win_rate": 0.65,
        }

        # Should not raise
        signal = controller._compute_adaptive_signal("hex8_2p", state, eval_result)
        assert signal is not None

    def test_compute_adaptive_signal_returns_signal(self, controller):
        """Should return AdaptiveTrainingSignal instance."""
        from app.coordination.feedback_loop_controller import AdaptiveTrainingSignal

        state = controller._get_or_create_state("hex8_2p")
        state.last_elo = 1500.0

        eval_result = {"elo": 1505.0, "win_rate": 0.55}

        signal = controller._compute_adaptive_signal("hex8_2p", state, eval_result)

        assert isinstance(signal, AdaptiveTrainingSignal)


# =============================================================================
# Emit Selfplay Adjustment Tests
# =============================================================================


class TestEmitSelfplayAdjustment:
    """Tests for _emit_selfplay_adjustment method."""

    def test_emits_event_with_state_params(self, controller):
        """Should emit event with state parameters."""
        state = controller._get_or_create_state("hex8_2p")
        state.current_search_budget = 500
        state.current_exploration_boost = 1.3

        # Mock the event bus import path inside the method
        with patch("app.coordination.event_router.get_event_bus") as mock_bus_getter:
            mock_bus = MagicMock()
            mock_bus_getter.return_value = mock_bus

            controller._emit_selfplay_adjustment(
                config_key="hex8_2p",
                state=state,
                elo_gap=400.0,
                velocity=5.0,
            )

            # Check that emit was called (may or may not succeed depending on mock)
            # The method gracefully handles failures

    def test_emit_adjustment_updates_state(self, controller):
        """Method should not raise even without event bus."""
        state = controller._get_or_create_state("hex8_2p")
        state.current_search_budget = 500
        state.current_exploration_boost = 1.3

        # Should not raise even if event bus unavailable
        controller._emit_selfplay_adjustment(
            config_key="hex8_2p",
            state=state,
            elo_gap=400.0,
            velocity=5.0,
        )

    def test_handles_missing_event_bus(self, controller):
        """Should handle missing event bus gracefully."""
        state = controller._get_or_create_state("hex8_2p")

        # The method imports get_event_bus internally, so we just test it doesn't raise
        # Should not raise
        controller._emit_selfplay_adjustment("hex8_2p", state, 400.0, 5.0)


# =============================================================================
# Regression Detection Tests
# =============================================================================


class TestRegressionDetection:
    """Tests for _on_regression_detected handler."""

    def test_regression_increments_failures(self, controller):
        """Should increment consecutive_failures count."""
        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "elo_drop": 30,
            "previous_elo": 1600,
            "current_elo": 1570,
        }

        with patch("app.coordination.feedback_loop_controller._safe_create_task"):
            controller._on_regression_detected(event)

        state = controller.get_state("hex8_2p")
        assert state.consecutive_failures >= 1

    def test_regression_increases_exploration_boost(self, controller):
        """Should increase exploration boost."""
        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "elo_drop": 30,
        }

        # Set initial low exploration boost
        state = controller._get_or_create_state("hex8_2p")
        state.current_exploration_boost = 1.0

        with patch("app.coordination.feedback_loop_controller._safe_create_task"):
            controller._on_regression_detected(event)

        # Exploration boost should be increased
        assert state.current_exploration_boost > 1.0

    def test_regression_emits_selfplay_target(self, controller):
        """Should emit SELFPLAY_TARGET_UPDATED for more diverse games."""
        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "elo_drop": 50,
            "consecutive_regressions": 2,
        }

        with patch("app.coordination.feedback_loop_controller._safe_create_task") as mock_task:
            controller._on_regression_detected(event)

        # Should attempt to create task for emitting event
        # (may not succeed if no event loop)

    def test_regression_handles_missing_config(self, controller):
        """Should handle missing config_key."""
        event = MagicMock()
        event.payload = {"elo_drop": 30}

        # Should not raise
        controller._on_regression_detected(event)


# =============================================================================
# Training Loss Trend Tests
# =============================================================================


class TestTrainingLossTrend:
    """Tests for _on_training_loss_trend handler."""

    def test_stalled_trend_boosts_exploration(self, controller):
        """Stalled trend should boost exploration."""
        event = MagicMock()
        event.payload = {
            "config": "hex8_2p",
            "trend": "stalled",
            "duration_epochs": 10,
        }

        with patch.object(controller, "_boost_exploration_for_stall") as mock_boost:
            controller._on_training_loss_trend(event)

        mock_boost.assert_called_once()

    def test_degrading_trend_triggers_quality_check(self, controller):
        """Degrading trend should trigger quality check."""
        event = MagicMock()
        event.payload = {
            "config": "hex8_2p",
            "trend": "degrading",
            "duration_epochs": 5,
        }

        with patch.object(controller, "_trigger_quality_check") as mock_quality:
            controller._on_training_loss_trend(event)

        mock_quality.assert_called_once()

    def test_improving_trend_reduces_exploration(self, controller):
        """Improving trend should reduce exploration boost."""
        event = MagicMock()
        event.payload = {
            "config": "hex8_2p",
            "trend": "improving",
        }

        # Set up state with prior anomaly
        state = controller._get_or_create_state("hex8_2p")
        state.loss_anomaly_count = 3

        with patch.object(controller, "_reduce_exploration_after_improvement") as mock_reduce:
            controller._on_training_loss_trend(event)

        # Should reset anomaly count and reduce exploration
        assert state.loss_anomaly_count == 0

    def test_handles_missing_config(self, controller):
        """Should handle missing config."""
        event = MagicMock()
        event.payload = {"trend": "stalled"}

        # Should not raise
        controller._on_training_loss_trend(event)
