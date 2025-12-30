"""Tests for ArchitectureFeedbackController.

Comprehensive test suite for the architecture feedback controller that
bridges evaluation results to selfplay allocation weights.

December 29, 2025 - Created as part of improvement plan.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.architecture_feedback_controller import (
    ArchitectureFeedbackConfig,
    ArchitectureFeedbackController,
    ArchitectureFeedbackState,
    get_architecture_feedback_controller,
    start_architecture_feedback_controller,
)


# =============================================================================
# ArchitectureFeedbackConfig Tests
# =============================================================================


class TestArchitectureFeedbackConfig:
    """Tests for configuration dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ArchitectureFeedbackConfig()
        assert config.min_allocation_per_arch == 0.10
        assert config.weight_update_interval == 1800.0
        assert config.weight_temperature == 0.5
        assert len(config.supported_architectures) == 7

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ArchitectureFeedbackConfig(
            min_allocation_per_arch=0.05,
            weight_update_interval=900.0,
            weight_temperature=1.0,
        )
        assert config.min_allocation_per_arch == 0.05
        assert config.weight_update_interval == 900.0
        assert config.weight_temperature == 1.0

    def test_supported_architectures_include_nnue(self):
        """Test that supported architectures include NNUE variants."""
        config = ArchitectureFeedbackConfig()
        assert "nnue_v1" in config.supported_architectures
        assert "nnue_v1_policy" in config.supported_architectures

    def test_supported_architectures_include_standard(self):
        """Test that supported architectures include standard NN versions."""
        config = ArchitectureFeedbackConfig()
        assert "v2" in config.supported_architectures
        assert "v4" in config.supported_architectures
        assert "v5" in config.supported_architectures
        assert "v6" in config.supported_architectures


# =============================================================================
# ArchitectureFeedbackState Tests
# =============================================================================


class TestArchitectureFeedbackState:
    """Tests for state dataclass."""

    def test_default_state(self):
        """Test default state values."""
        state = ArchitectureFeedbackState()
        assert state.last_weight_update_time == 0.0
        assert state.cached_weights == {}
        assert state.evaluations_processed == 0
        assert state.trainings_processed == 0

    def test_state_mutation(self):
        """Test that state can be modified."""
        state = ArchitectureFeedbackState()
        state.evaluations_processed = 10
        state.trainings_processed = 5
        state.cached_weights["hex8_2p"] = {"v5": 0.5, "v6": 0.5}

        assert state.evaluations_processed == 10
        assert state.trainings_processed == 5
        assert "hex8_2p" in state.cached_weights


# =============================================================================
# ArchitectureFeedbackController Tests
# =============================================================================


class TestArchitectureFeedbackControllerInit:
    """Tests for controller initialization."""

    def setup_method(self):
        """Reset singleton before each test."""
        ArchitectureFeedbackController.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        ArchitectureFeedbackController.reset_instance()

    def test_init_default_config(self):
        """Test initialization with default config."""
        controller = ArchitectureFeedbackController()
        assert controller._config.min_allocation_per_arch == 0.10
        assert controller._running is False

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = ArchitectureFeedbackConfig(min_allocation_per_arch=0.15)
        controller = ArchitectureFeedbackController(config=config)
        assert controller._config.min_allocation_per_arch == 0.15

    def test_inherits_handler_base(self):
        """Test that controller inherits from HandlerBase."""
        controller = ArchitectureFeedbackController()
        assert hasattr(controller, "_run_cycle")
        assert hasattr(controller, "start")
        assert hasattr(controller, "stop")


class TestArchitectureFeedbackControllerSingleton:
    """Tests for singleton pattern."""

    def setup_method(self):
        """Reset singleton before each test."""
        ArchitectureFeedbackController.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        ArchitectureFeedbackController.reset_instance()

    def test_get_instance_creates_singleton(self):
        """Test that get_instance creates singleton."""
        controller1 = ArchitectureFeedbackController.get_instance()
        controller2 = ArchitectureFeedbackController.get_instance()
        assert controller1 is controller2

    def test_reset_instance_clears_singleton(self):
        """Test that reset_instance clears singleton."""
        controller1 = ArchitectureFeedbackController.get_instance()
        ArchitectureFeedbackController.reset_instance()
        controller2 = ArchitectureFeedbackController.get_instance()
        assert controller1 is not controller2

    def test_module_accessor_returns_singleton(self):
        """Test that get_architecture_feedback_controller returns singleton."""
        controller1 = get_architecture_feedback_controller()
        controller2 = get_architecture_feedback_controller()
        assert controller1 is controller2


class TestEventSubscriptions:
    """Tests for event subscription handling."""

    def setup_method(self):
        """Reset singleton before each test."""
        ArchitectureFeedbackController.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        ArchitectureFeedbackController.reset_instance()

    def test_event_subscriptions(self):
        """Test that correct events are subscribed."""
        controller = ArchitectureFeedbackController()
        subs = controller._get_event_subscriptions()

        assert "EVALUATION_COMPLETED" in subs
        assert "TRAINING_COMPLETED" in subs
        assert callable(subs["EVALUATION_COMPLETED"])
        assert callable(subs["TRAINING_COMPLETED"])


class TestOnEvaluationCompleted:
    """Tests for evaluation completed handler."""

    def setup_method(self):
        """Reset singleton before each test."""
        ArchitectureFeedbackController.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        ArchitectureFeedbackController.reset_instance()

    @pytest.mark.asyncio
    async def test_on_evaluation_completed_records_elo(self):
        """Test that evaluation results are recorded."""
        controller = ArchitectureFeedbackController()

        with patch(
            "app.coordination.architecture_feedback_controller.get_architecture_tracker"
        ) as mock_get_tracker, patch(
            "app.coordination.architecture_feedback_controller.extract_architecture_from_model_path",
            return_value="v5",
        ):
            mock_tracker = MagicMock()
            mock_get_tracker.return_value = mock_tracker

            event = {
                "config_key": "hex8_2p",
                "model_path": "models/hex8_2p_v5.pth",
                "elo": 1150.0,
                "games": 50,
            }

            await controller._on_evaluation_completed(event)

            mock_tracker.record_evaluation.assert_called_once()
            call_args = mock_tracker.record_evaluation.call_args
            assert call_args.kwargs["architecture"] == "v5"
            assert call_args.kwargs["board_type"] == "hex8"
            assert call_args.kwargs["num_players"] == 2
            assert call_args.kwargs["elo"] == 1150.0

    @pytest.mark.asyncio
    async def test_on_evaluation_completed_increments_counter(self):
        """Test that evaluation counter is incremented."""
        controller = ArchitectureFeedbackController()

        with patch(
            "app.coordination.architecture_feedback_controller.get_architecture_tracker"
        ) as mock_get_tracker, patch(
            "app.coordination.architecture_feedback_controller.extract_architecture_from_model_path",
            return_value="v5",
        ):
            mock_tracker = MagicMock()
            mock_get_tracker.return_value = mock_tracker

            event = {
                "config_key": "hex8_2p",
                "model_path": "models/hex8_2p_v5.pth",
                "elo": 1150.0,
                "games": 50,
            }

            assert controller._state.evaluations_processed == 0
            await controller._on_evaluation_completed(event)
            assert controller._state.evaluations_processed == 1

    @pytest.mark.asyncio
    async def test_on_evaluation_completed_missing_config_key(self):
        """Test that missing config_key is handled gracefully."""
        controller = ArchitectureFeedbackController()

        event = {
            "model_path": "models/hex8_2p_v5.pth",
            "elo": 1150.0,
        }

        # Should not raise
        await controller._on_evaluation_completed(event)
        assert controller._state.evaluations_processed == 0

    @pytest.mark.asyncio
    async def test_on_evaluation_completed_invalid_config_key(self):
        """Test that invalid config_key format is handled."""
        controller = ArchitectureFeedbackController()

        with patch(
            "app.coordination.architecture_feedback_controller.get_architecture_tracker"
        ), patch(
            "app.coordination.architecture_feedback_controller.extract_architecture_from_model_path",
            return_value="v5",
        ):
            event = {
                "config_key": "invalid_format",  # No player count suffix
                "model_path": "models/hex8_2p_v5.pth",
                "elo": 1150.0,
            }

            # Should not raise or increment counter
            await controller._on_evaluation_completed(event)


class TestOnTrainingCompleted:
    """Tests for training completed handler."""

    def setup_method(self):
        """Reset singleton before each test."""
        ArchitectureFeedbackController.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        ArchitectureFeedbackController.reset_instance()

    @pytest.mark.asyncio
    async def test_on_training_completed_records_hours(self):
        """Test that training duration is recorded."""
        controller = ArchitectureFeedbackController()

        with patch(
            "app.coordination.architecture_feedback_controller.get_architecture_tracker"
        ) as mock_get_tracker, patch(
            "app.coordination.architecture_feedback_controller.extract_architecture_from_model_path",
            return_value="v5",
        ):
            mock_tracker = MagicMock()
            mock_get_tracker.return_value = mock_tracker

            event = {
                "config_key": "square8_4p",
                "model_path": "models/square8_4p_v5.pth",
                "duration_seconds": 7200.0,  # 2 hours
            }

            await controller._on_training_completed(event)

            mock_tracker.record_evaluation.assert_called_once()
            call_args = mock_tracker.record_evaluation.call_args
            assert call_args.kwargs["architecture"] == "v5"
            assert call_args.kwargs["training_hours"] == 2.0
            assert call_args.kwargs["games_evaluated"] == 0  # No games for training

    @pytest.mark.asyncio
    async def test_on_training_completed_increments_counter(self):
        """Test that training counter is incremented."""
        controller = ArchitectureFeedbackController()

        with patch(
            "app.coordination.architecture_feedback_controller.get_architecture_tracker"
        ) as mock_get_tracker, patch(
            "app.coordination.architecture_feedback_controller.extract_architecture_from_model_path",
            return_value="v5",
        ):
            mock_tracker = MagicMock()
            mock_get_tracker.return_value = mock_tracker

            event = {
                "config_key": "hex8_2p",
                "model_path": "models/hex8_2p_v5.pth",
                "duration_seconds": 3600.0,
            }

            assert controller._state.trainings_processed == 0
            await controller._on_training_completed(event)
            assert controller._state.trainings_processed == 1


class TestEnforceMinimumAllocation:
    """Tests for minimum allocation enforcement."""

    def setup_method(self):
        """Reset singleton before each test."""
        ArchitectureFeedbackController.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        ArchitectureFeedbackController.reset_instance()

    def test_enforce_minimum_empty_weights(self):
        """Test that empty weights are returned unchanged."""
        controller = ArchitectureFeedbackController()
        result = controller._enforce_minimum_allocation({})
        assert result == {}

    def test_enforce_minimum_applies_floor(self):
        """Test that minimum floor is applied."""
        controller = ArchitectureFeedbackController()

        # v5 has very low weight, should be boosted to 10%
        weights = {"v5": 0.02, "v6": 0.98}
        result = controller._enforce_minimum_allocation(weights)

        # v5 should be at least 10%
        assert result["v5"] >= 0.10
        # Weights should sum to 1.0
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_enforce_minimum_equal_distribution_when_exceeds(self):
        """Test equal distribution when minimum would exceed 100%."""
        config = ArchitectureFeedbackConfig(min_allocation_per_arch=0.20)
        controller = ArchitectureFeedbackController(config=config)

        # 10 architectures * 20% = 200%, which exceeds 100%
        weights = {f"v{i}": 0.1 for i in range(10)}
        result = controller._enforce_minimum_allocation(weights)

        # Should be equal distribution
        expected = 1.0 / 10
        for arch, weight in result.items():
            assert abs(weight - expected) < 0.01

    def test_enforce_minimum_normalizes(self):
        """Test that weights are normalized to sum to 1.0."""
        controller = ArchitectureFeedbackController()

        weights = {"v5": 0.3, "v6": 0.3, "nnue_v1": 0.4}
        result = controller._enforce_minimum_allocation(weights)

        # Sum should be 1.0
        assert abs(sum(result.values()) - 1.0) < 0.01


class TestMaybeEmitWeightUpdate:
    """Tests for weight update timing."""

    def setup_method(self):
        """Reset singleton before each test."""
        ArchitectureFeedbackController.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        ArchitectureFeedbackController.reset_instance()

    @pytest.mark.asyncio
    async def test_does_not_emit_if_too_recent(self):
        """Test that weight update is skipped if too recent."""
        controller = ArchitectureFeedbackController()
        controller._state.last_weight_update_time = time.time()

        with patch.object(
            controller, "_emit_architecture_weights_updated"
        ) as mock_emit:
            await controller._maybe_emit_weight_update("hex8_2p")
            mock_emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_emits_if_interval_passed(self):
        """Test that weight update is emitted after interval."""
        config = ArchitectureFeedbackConfig(weight_update_interval=0.0)  # No wait
        controller = ArchitectureFeedbackController(config=config)

        with patch.object(
            controller, "_emit_architecture_weights_updated"
        ) as mock_emit:
            mock_emit.return_value = None
            await controller._maybe_emit_weight_update("hex8_2p")
            mock_emit.assert_called_once_with("hex8_2p")


class TestHealthCheck:
    """Tests for health check functionality."""

    def setup_method(self):
        """Reset singleton before each test."""
        ArchitectureFeedbackController.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        ArchitectureFeedbackController.reset_instance()

    def test_health_check_when_running(self):
        """Test health check when controller is running."""
        controller = ArchitectureFeedbackController()
        controller._running = True

        result = controller.health_check()

        assert result.message == "Running"
        assert "evaluations_processed" in result.details
        assert "version" in result.details

    def test_health_check_when_not_running(self):
        """Test health check when controller is not running."""
        controller = ArchitectureFeedbackController()
        controller._running = False

        result = controller.health_check()

        assert result.message == "Not running"

    def test_health_check_includes_stats(self):
        """Test that health check includes processing stats."""
        controller = ArchitectureFeedbackController()
        controller._state.evaluations_processed = 10
        controller._state.trainings_processed = 5
        controller._state.cached_weights = {"hex8_2p": {}, "square8_4p": {}}

        result = controller.health_check()

        assert result.details["evaluations_processed"] == 10
        assert result.details["trainings_processed"] == 5
        assert result.details["cached_configs"] == 2


class TestModuleAccessors:
    """Tests for module-level accessor functions."""

    def setup_method(self):
        """Reset singleton before each test."""
        ArchitectureFeedbackController.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        ArchitectureFeedbackController.reset_instance()

    def test_get_architecture_feedback_controller(self):
        """Test that module accessor returns singleton."""
        controller = get_architecture_feedback_controller()
        assert isinstance(controller, ArchitectureFeedbackController)

    @pytest.mark.asyncio
    async def test_start_architecture_feedback_controller(self):
        """Test that start function starts controller."""
        with patch.object(
            ArchitectureFeedbackController, "start", new_callable=AsyncMock
        ):
            controller = await start_architecture_feedback_controller()
            assert isinstance(controller, ArchitectureFeedbackController)
            controller.start.assert_called_once()
