"""Tests for OptimizationCoordinator.

Tests cover:
- Plateau detection
- CMA-ES and NAS triggering
- Optimization run tracking
- Cooldown management
- Event handling
- Statistics
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

from app.coordination.optimization_coordinator import (
    OptimizationCoordinator,
    OptimizationRun,
    OptimizationStats,
    OptimizationStatus,
    OptimizationType,
    PlateauDetection,
    PlateauType,
    get_optimization_coordinator,
    get_optimization_stats,
    is_optimization_running,
    trigger_cmaes,
    wire_optimization_events,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def coordinator():
    """Create a fresh OptimizationCoordinator."""
    return OptimizationCoordinator(
        plateau_window=5,
        plateau_threshold=0.001,
        cooldown_seconds=60.0,
    )


@pytest.fixture
def mock_event():
    """Create a mock event with payload."""
    @dataclass
    class MockEvent:
        payload: dict[str, Any]
    return MockEvent


# =============================================================================
# PlateauDetection Tests
# =============================================================================


class TestPlateauDetection:
    """Test PlateauDetection dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        plateau = PlateauDetection(
            plateau_type=PlateauType.LOSS,
            metric_name="train_loss",
            current_value=0.5,
            best_value=0.45,
            epochs_since_improvement=10,
        )

        assert plateau.plateau_type == PlateauType.LOSS
        assert plateau.epochs_since_improvement == 10
        assert plateau.triggered_optimization is False


# =============================================================================
# OptimizationRun Tests
# =============================================================================


class TestOptimizationRun:
    """Test OptimizationRun dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        run = OptimizationRun(
            run_id="test-001",
            optimization_type=OptimizationType.CMAES,
        )

        assert run.run_id == "test-001"
        assert run.status == OptimizationStatus.PENDING
        assert run.evaluations == 0

    def test_duration_not_completed(self):
        """Test duration when not completed."""
        run = OptimizationRun(
            run_id="test",
            optimization_type=OptimizationType.CMAES,
            started_at=time.time() - 60,
        )

        assert 59 < run.duration < 61

    def test_duration_completed(self):
        """Test duration when completed."""
        start = time.time() - 120
        end = time.time() - 60

        run = OptimizationRun(
            run_id="test",
            optimization_type=OptimizationType.CMAES,
            started_at=start,
            completed_at=end,
        )

        # Use approximate comparison for floating point
        assert abs(run.duration - 60.0) < 0.1


# =============================================================================
# OptimizationCoordinator Initialization Tests
# =============================================================================


class TestOptimizationCoordinatorInit:
    """Test OptimizationCoordinator initialization."""

    def test_init_default_values(self, coordinator):
        """Test default initialization."""
        assert coordinator.plateau_window == 5
        assert coordinator.plateau_threshold == 0.001
        assert coordinator.cooldown_seconds == 60.0

    def test_init_empty_state(self, coordinator):
        """Test initial state is empty."""
        assert not coordinator.is_optimization_running()
        assert len(coordinator.get_optimization_history()) == 0
        assert len(coordinator.get_plateau_history()) == 0


class TestOptimizationCoordinatorSubscriptions:
    """Tests for event subscription wiring."""

    def test_subscribe_to_events_uses_router_helper(self, coordinator):
        """Should subscribe optimization events through the unified helper."""
        from app.coordination.event_router import DataEventType

        with patch("app.coordination.event_router.subscribe") as mock_subscribe:
            result = coordinator.subscribe_to_events()

        assert result is True
        assert coordinator._subscribed is True
        subscribed = [(call.args[0], call.args[1]) for call in mock_subscribe.call_args_list]
        assert (DataEventType.PLATEAU_DETECTED, coordinator._on_plateau_detected) in subscribed
        assert (DataEventType.CMAES_TRIGGERED, coordinator._on_cmaes_triggered) in subscribed
        assert (DataEventType.CMAES_COMPLETED, coordinator._on_cmaes_completed) in subscribed
        assert (DataEventType.NAS_TRIGGERED, coordinator._on_nas_triggered) in subscribed
        assert (DataEventType.NAS_COMPLETED, coordinator._on_nas_completed) in subscribed
        assert (DataEventType.TRAINING_PROGRESS, coordinator._on_training_progress) in subscribed
        assert (DataEventType.HYPERPARAMETER_UPDATED, coordinator._on_hyperparameter_updated) in subscribed


# =============================================================================
# Metric Tracking Tests
# =============================================================================


class TestMetricTracking:
    """Test metric tracking and plateau detection."""

    def test_update_metric_tracks_values(self, coordinator):
        """Test _update_metric tracks values."""
        coordinator._update_metric("train_loss", 0.5)
        coordinator._update_metric("train_loss", 0.45)
        coordinator._update_metric("train_loss", 0.4)

        assert len(coordinator._metric_history["train_loss"]) == 3

    def test_update_metric_tracks_best(self, coordinator):
        """Test _update_metric tracks best value."""
        coordinator._update_metric("train_loss", 0.5)
        coordinator._update_metric("train_loss", 0.4)  # New best
        coordinator._update_metric("train_loss", 0.45)  # Not best

        assert coordinator._metric_best["train_loss"] == 0.4

    def test_update_metric_increments_epochs_without_improvement(self, coordinator):
        """Test epochs counter increments when no improvement."""
        coordinator._update_metric("train_loss", 0.5)  # First value, sets baseline
        coordinator._update_metric("train_loss", 0.5)  # No improvement
        coordinator._update_metric("train_loss", 0.5)  # No improvement

        # Implementation counts all updates that don't beat the best
        # (including when equal to best)
        assert coordinator._epochs_since_improvement["train_loss"] >= 2

    def test_update_metric_resets_epochs_on_improvement(self, coordinator):
        """Test epochs counter resets on improvement."""
        coordinator._update_metric("train_loss", 0.5)
        coordinator._update_metric("train_loss", 0.5)
        coordinator._update_metric("train_loss", 0.3)  # Improvement

        assert coordinator._epochs_since_improvement["train_loss"] == 0


# =============================================================================
# Plateau Detection Tests
# =============================================================================


class TestPlateauDetectionLogic:
    """Test plateau detection logic."""

    def test_detect_plateau_no_data(self, coordinator):
        """Test detect_plateau with no data."""
        result = coordinator.detect_plateau("unknown_metric")
        assert result is None

    def test_detect_plateau_not_stalled(self, coordinator):
        """Test detect_plateau when not stalled enough."""
        for i in range(3):
            coordinator._update_metric("train_loss", 0.5)

        result = coordinator.detect_plateau("train_loss")
        assert result is None  # Not stalled long enough

    def test_detect_plateau_stalled(self, coordinator):
        """Test detect_plateau when stalled."""
        # Initialize metric
        coordinator._update_metric("train_loss", 0.5)

        # Manually set epochs since improvement
        coordinator._epochs_since_improvement["train_loss"] = 10

        result = coordinator.detect_plateau("train_loss")

        assert result is not None
        assert result.plateau_type == PlateauType.LOSS
        assert result.epochs_since_improvement == 10


# =============================================================================
# CMAES Trigger Tests
# =============================================================================


class TestCMAESTrigger:
    """Test CMA-ES triggering."""

    def test_trigger_cmaes_success(self, coordinator):
        """Test successful CMA-ES trigger."""
        with patch.object(coordinator, "_emit_optimization_triggered"):
            run = coordinator.trigger_cmaes(
                reason="test",
                parameters=["lr", "batch_size"],
                generations=10,
            )

            assert run is not None
            assert run.optimization_type == OptimizationType.CMAES
            assert run.trigger_reason == "test"
            assert coordinator.is_optimization_running()

    def test_trigger_cmaes_blocked_when_running(self, coordinator):
        """Test CMA-ES blocked when optimization running."""
        with patch.object(coordinator, "_emit_optimization_triggered"):
            coordinator.trigger_cmaes("first")
            run = coordinator.trigger_cmaes("second")

            assert run is None

    def test_trigger_cmaes_blocked_during_cooldown(self, coordinator):
        """Test CMA-ES blocked during cooldown."""
        coordinator._last_optimization_end = time.time()  # Just ended

        run = coordinator.trigger_cmaes("test")
        assert run is None


# =============================================================================
# NAS Trigger Tests
# =============================================================================


class TestNASTrigger:
    """Test NAS triggering."""

    def test_trigger_nas_success(self, coordinator):
        """Test successful NAS trigger."""
        with patch.object(coordinator, "_emit_optimization_triggered"):
            run = coordinator.trigger_nas(
                reason="architecture_search",
                generations=5,
            )

            assert run is not None
            assert run.optimization_type == OptimizationType.NAS

    def test_trigger_nas_blocked_when_running(self, coordinator):
        """Test NAS blocked when optimization running."""
        with patch.object(coordinator, "_emit_optimization_triggered"):
            coordinator.trigger_cmaes("first")
            run = coordinator.trigger_nas("second")

            assert run is None


# =============================================================================
# Optimization Cancellation Tests
# =============================================================================


class TestCancelOptimization:
    """Test optimization cancellation."""

    def test_cancel_optimization_success(self, coordinator):
        """Test successful cancellation."""
        with patch.object(coordinator, "_emit_optimization_triggered"):
            coordinator.trigger_cmaes("test")
            result = coordinator.cancel_optimization()

            assert result is True
            assert not coordinator.is_optimization_running()

    def test_cancel_optimization_nothing_running(self, coordinator):
        """Test cancellation when nothing running."""
        result = coordinator.cancel_optimization()
        assert result is False


# =============================================================================
# Event Handler Tests
# =============================================================================


class TestEventHandlers:
    """Test event handler methods."""

    @pytest.mark.asyncio
    async def test_on_plateau_detected(self, coordinator, mock_event):
        """Test PLATEAU_DETECTED event handling."""
        event = mock_event(payload={
            "plateau_type": "loss",
            "metric_name": "train_loss",
            "current_value": 0.5,
            "best_value": 0.45,
            "epochs_since_improvement": 20,
        })

        await coordinator._on_plateau_detected(event)

        history = coordinator.get_plateau_history()
        assert len(history) == 1
        assert history[0].plateau_type == PlateauType.LOSS

    @pytest.mark.asyncio
    async def test_on_cmaes_triggered(self, coordinator, mock_event):
        """Test CMAES_TRIGGERED event handling."""
        event = mock_event(payload={
            "run_id": "test-001",
            "reason": "external_trigger",
            "parameters": ["lr"],
            "generations": 10,
        })

        await coordinator._on_cmaes_triggered(event)

        assert coordinator.is_optimization_running()
        assert coordinator._current_optimization.run_id == "test-001"

    @pytest.mark.asyncio
    async def test_on_cmaes_completed(self, coordinator, mock_event):
        """Test CMAES_COMPLETED event handling."""
        # First trigger
        trigger_event = mock_event(payload={
            "run_id": "test-001",
            "reason": "test",
        })
        await coordinator._on_cmaes_triggered(trigger_event)

        # Then complete
        complete_event = mock_event(payload={
            "success": True,
            "best_params": {"lr": 0.001},
            "best_score": 0.95,
            "evaluations": 100,
        })
        await coordinator._on_cmaes_completed(complete_event)

        assert not coordinator.is_optimization_running()
        history = coordinator.get_optimization_history()
        assert len(history) == 1
        assert history[0].status == OptimizationStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_on_training_progress(self, coordinator, mock_event):
        """Test TRAINING_PROGRESS event handling."""
        event = mock_event(payload={
            "train_loss": 0.5,
            "val_loss": 0.55,
            "elo": 1500.0,
        })

        await coordinator._on_training_progress(event)

        assert "train_loss" in coordinator._metric_history
        assert "val_loss" in coordinator._metric_history
        assert "elo" in coordinator._metric_history


# =============================================================================
# Callback Tests
# =============================================================================


class TestCallbacks:
    """Test callback functionality."""

    def test_on_plateau_callback(self, coordinator):
        """Test plateau callback is called."""
        callbacks_called = []
        coordinator.on_plateau(lambda p: callbacks_called.append(p))

        plateau = PlateauDetection(
            plateau_type=PlateauType.LOSS,
            metric_name="test",
            current_value=0.5,
            best_value=0.45,
            epochs_since_improvement=10,
        )
        coordinator._plateaus.append(plateau)

        # Manually trigger callback
        for callback in coordinator._plateau_callbacks:
            callback(plateau)

        assert len(callbacks_called) == 1

    def test_on_optimization_complete_callback(self, coordinator):
        """Test optimization complete callback is called."""
        callbacks_called = []
        coordinator.on_optimization_complete(lambda r: callbacks_called.append(r))

        run = OptimizationRun(
            run_id="test",
            optimization_type=OptimizationType.CMAES,
            status=OptimizationStatus.COMPLETED,
        )
        coordinator._record_optimization(run)

        assert len(callbacks_called) == 1


# =============================================================================
# State Query Tests
# =============================================================================


class TestStateQueries:
    """Test state query methods."""

    def test_is_optimization_running(self, coordinator):
        """Test is_optimization_running."""
        assert not coordinator.is_optimization_running()

        with patch.object(coordinator, "_emit_optimization_triggered"):
            coordinator.trigger_cmaes("test")
            assert coordinator.is_optimization_running()

    def test_is_in_cooldown(self, coordinator):
        """Test is_in_cooldown."""
        assert not coordinator.is_in_cooldown()

        coordinator._last_optimization_end = time.time()
        assert coordinator.is_in_cooldown()

    def test_can_trigger_optimization(self, coordinator):
        """Test can_trigger_optimization."""
        assert coordinator.can_trigger_optimization()

        with patch.object(coordinator, "_emit_optimization_triggered"):
            coordinator.trigger_cmaes("test")
            assert not coordinator.can_trigger_optimization()

    def test_get_current_optimization(self, coordinator):
        """Test get_current_optimization."""
        assert coordinator.get_current_optimization() is None

        with patch.object(coordinator, "_emit_optimization_triggered"):
            coordinator.trigger_cmaes("test")
            current = coordinator.get_current_optimization()
            assert current is not None


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Test statistics calculation."""

    def test_get_stats_empty(self, coordinator):
        """Test get_stats with no history."""
        stats = coordinator.get_stats()

        assert stats.total_runs == 0
        assert stats.successful_runs == 0
        assert stats.current_running is None

    def test_get_stats_with_history(self, coordinator):
        """Test get_stats with history."""
        # Add completed runs
        run1 = OptimizationRun(
            run_id="run-1",
            optimization_type=OptimizationType.CMAES,
            status=OptimizationStatus.COMPLETED,
            evaluations=50,
        )
        run2 = OptimizationRun(
            run_id="run-2",
            optimization_type=OptimizationType.NAS,
            status=OptimizationStatus.FAILED,
            evaluations=30,
        )
        coordinator._record_optimization(run1)
        coordinator._record_optimization(run2)

        stats = coordinator.get_stats()

        assert stats.total_runs == 2
        assert stats.successful_runs == 1
        assert stats.failed_runs == 1

    def test_get_status(self, coordinator):
        """Test get_status returns proper structure."""
        status = coordinator.get_status()

        assert "total_runs" in status
        assert "is_running" in status
        assert "in_cooldown" in status
        assert "subscribed" in status


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingletonBehavior:
    """Test singleton behavior."""

    def test_get_optimization_coordinator_returns_singleton(self):
        """Test get_optimization_coordinator returns same instance."""
        import app.coordination.optimization_coordinator as oc
        oc._optimization_coordinator = None

        coord1 = get_optimization_coordinator()
        coord2 = get_optimization_coordinator()

        assert coord1 is coord2

    def test_convenience_functions(self):
        """Test convenience functions work."""
        import app.coordination.optimization_coordinator as oc
        oc._optimization_coordinator = None

        assert not is_optimization_running()
        stats = get_optimization_stats()
        assert stats.total_runs == 0


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Test enum values."""

    def test_optimization_type_values(self):
        """Test OptimizationType enum values."""
        assert OptimizationType.CMAES.value == "cmaes"
        assert OptimizationType.NAS.value == "nas"
        assert OptimizationType.PBT.value == "pbt"

    def test_optimization_status_values(self):
        """Test OptimizationStatus enum values."""
        assert OptimizationStatus.PENDING.value == "pending"
        assert OptimizationStatus.RUNNING.value == "running"
        assert OptimizationStatus.COMPLETED.value == "completed"
        assert OptimizationStatus.FAILED.value == "failed"

    def test_plateau_type_values(self):
        """Test PlateauType enum values."""
        assert PlateauType.LOSS.value == "loss"
        assert PlateauType.ELO.value == "elo"
        assert PlateauType.WIN_RATE.value == "win_rate"
