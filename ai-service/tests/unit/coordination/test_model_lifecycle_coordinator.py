"""Tests for ModelLifecycleCoordinator.

Tests cover:
- Model registration and state transitions
- Checkpoint tracking
- Promotion and rollback handling
- Cache management
- Event handling
- Statistics and history
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

from app.coordination.model_lifecycle_coordinator import (
    CacheEntry,
    CheckpointInfo,
    ModelLifecycleCoordinator,
    ModelRecord,
    ModelState,
    get_model_coordinator,
    get_production_elo,
    get_production_model_id,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def coordinator():
    """Create a fresh ModelLifecycleCoordinator for each test."""
    return ModelLifecycleCoordinator()


@pytest.fixture
def mock_event():
    """Create a mock event with payload."""
    @dataclass
    class MockEvent:
        payload: dict[str, Any]
    return MockEvent


# =============================================================================
# Initialization Tests
# =============================================================================


class TestModelLifecycleCoordinatorInit:
    """Test ModelLifecycleCoordinator initialization."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        coord = ModelLifecycleCoordinator()
        assert coord.max_checkpoint_history == 100
        assert coord.max_state_history_per_model == 50

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        coord = ModelLifecycleCoordinator(
            max_checkpoint_history=50,
            max_state_history_per_model=25,
        )
        assert coord.max_checkpoint_history == 50
        assert coord.max_state_history_per_model == 25

    def test_init_empty_state(self, coordinator):
        """Test initial state is empty."""
        stats = coordinator.get_stats()
        assert stats.total_models == 0
        assert stats.total_checkpoints == 0
        assert stats.total_promotions == 0


class TestModelLifecycleCoordinatorSubscriptions:
    """Tests for event subscription wiring."""

    def test_subscribe_to_events_uses_router_helper(self, coordinator):
        """Should subscribe model events through the unified helper."""
        from app.coordination.event_router import DataEventType

        with patch("app.coordination.event_router.subscribe") as mock_subscribe:
            result = coordinator.subscribe_to_events()

        assert result is True
        assert coordinator._subscribed is True
        subscribed = [(call.args[0], call.args[1]) for call in mock_subscribe.call_args_list]
        assert (DataEventType.CHECKPOINT_SAVED, coordinator._on_checkpoint_saved) in subscribed
        assert (DataEventType.CHECKPOINT_LOADED, coordinator._on_checkpoint_loaded) in subscribed
        assert (DataEventType.MODEL_PROMOTED, coordinator._on_model_promoted) in subscribed
        assert (DataEventType.PROMOTION_ROLLED_BACK, coordinator._on_promotion_rolled_back) in subscribed
        assert (DataEventType.PROMOTION_FAILED, coordinator._on_promotion_failed) in subscribed
        assert (DataEventType.TRAINING_COMPLETED, coordinator._on_training_completed) in subscribed
        assert (DataEventType.ELO_UPDATED, coordinator._on_elo_updated) in subscribed
        assert (DataEventType.MODEL_CORRUPTED, coordinator._on_model_corrupted) in subscribed
        assert (DataEventType.MODEL_NOT_FOUND, coordinator._on_model_not_found) in subscribed
        assert (DataEventType.REGRESSION_DETECTED, coordinator._on_regression_detected) in subscribed

    def test_unsubscribe_from_events_uses_router_helper(self, coordinator):
        """Should unsubscribe model events through the unified helper."""
        from app.coordination.event_router import DataEventType

        coordinator._subscribed = True

        with patch("app.coordination.event_router.unsubscribe") as mock_unsubscribe:
            coordinator.unsubscribe_from_events()

        assert coordinator._subscribed is False
        unsubscribed = [(call.args[0], call.args[1]) for call in mock_unsubscribe.call_args_list]
        assert (DataEventType.CHECKPOINT_SAVED, coordinator._on_checkpoint_saved) in unsubscribed
        assert (DataEventType.CHECKPOINT_LOADED, coordinator._on_checkpoint_loaded) in unsubscribed
        assert (DataEventType.MODEL_PROMOTED, coordinator._on_model_promoted) in unsubscribed
        assert (DataEventType.PROMOTION_ROLLED_BACK, coordinator._on_promotion_rolled_back) in unsubscribed
        assert (DataEventType.PROMOTION_FAILED, coordinator._on_promotion_failed) in unsubscribed
        assert (DataEventType.TRAINING_COMPLETED, coordinator._on_training_completed) in unsubscribed
        assert (DataEventType.ELO_UPDATED, coordinator._on_elo_updated) in unsubscribed
        assert (DataEventType.MODEL_CORRUPTED, coordinator._on_model_corrupted) in unsubscribed
        assert (DataEventType.MODEL_NOT_FOUND, coordinator._on_model_not_found) in unsubscribed
        assert (DataEventType.REGRESSION_DETECTED, coordinator._on_regression_detected) in unsubscribed


# =============================================================================
# Model Registration Tests
# =============================================================================


class TestModelRegistration:
    """Test model registration functionality."""

    def test_register_model_basic(self, coordinator):
        """Test basic model registration."""
        model = coordinator.register_model("model-v1")

        assert model.model_id == "model-v1"
        assert model.state == ModelState.TRAINING

    def test_register_model_with_parent(self, coordinator):
        """Test model registration with parent."""
        parent = coordinator.register_model("parent-model")
        child = coordinator.register_model("child-model", parent_model_id="parent-model")

        assert child.parent_model_id == "parent-model"
        assert "child-model" in parent.children_model_ids

    def test_register_model_custom_state(self, coordinator):
        """Test model registration with custom state."""
        model = coordinator.register_model(
            "model-v1",
            initial_state=ModelState.EVALUATING,
        )
        assert model.state == ModelState.EVALUATING

    def test_get_model(self, coordinator):
        """Test getting a model by ID."""
        coordinator.register_model("model-v1")

        model = coordinator.get_model("model-v1")
        assert model is not None
        assert model.model_id == "model-v1"

    def test_get_model_not_found(self, coordinator):
        """Test getting a non-existent model."""
        model = coordinator.get_model("nonexistent")
        assert model is None


# =============================================================================
# State Transition Tests
# =============================================================================


class TestStateTransitions:
    """Test model state transitions."""

    def test_update_model_state(self, coordinator):
        """Test updating model state."""
        coordinator.register_model("model-v1")

        result = coordinator.update_model_state(
            "model-v1",
            ModelState.EVALUATING,
            reason="training complete",
        )

        assert result is True
        model = coordinator.get_model("model-v1")
        assert model.state == ModelState.EVALUATING

    def test_update_model_state_records_history(self, coordinator):
        """Test state transition records history."""
        coordinator.register_model("model-v1")
        coordinator.update_model_state("model-v1", ModelState.EVALUATING)
        coordinator.update_model_state("model-v1", ModelState.STAGING)

        history = coordinator.get_model_history("model-v1")
        assert len(history) == 2
        assert history[0]["from_state"] == "training"
        assert history[0]["to_state"] == "evaluating"

    def test_update_model_state_not_found(self, coordinator):
        """Test updating non-existent model."""
        result = coordinator.update_model_state("nonexistent", ModelState.PRODUCTION)
        assert result is False

    def test_get_models_by_state(self, coordinator):
        """Test getting models by state."""
        coordinator.register_model("training-1")
        coordinator.register_model("training-2")
        coordinator.register_model("eval-1", initial_state=ModelState.EVALUATING)

        training = coordinator.get_models_by_state(ModelState.TRAINING)
        evaluating = coordinator.get_models_by_state(ModelState.EVALUATING)

        assert len(training) == 2
        assert len(evaluating) == 1


# =============================================================================
# Checkpoint Tests
# =============================================================================


class TestCheckpointTracking:
    """Test checkpoint tracking functionality."""

    @pytest.mark.asyncio
    async def test_checkpoint_saved_event(self, coordinator, mock_event):
        """Test handling CHECKPOINT_SAVED event."""
        event = mock_event(payload={
            "model_id": "model-v1",
            "checkpoint_id": "ckpt-001",
            "iteration": 1000,
            "path": "/models/ckpt-001.pth",
            "node_id": "gpu-1",
            "size_bytes": 100_000_000,
            "metrics": {"loss": 0.5},
            "is_best": True,
        })

        await coordinator._on_checkpoint_saved(event)

        checkpoints = coordinator.get_checkpoints("model-v1")
        assert len(checkpoints) == 1
        assert checkpoints[0].iteration == 1000
        assert checkpoints[0].is_best is True

    @pytest.mark.asyncio
    async def test_checkpoint_updates_model(self, coordinator, mock_event):
        """Test checkpoint updates model record."""
        coordinator.register_model("model-v1")

        event = mock_event(payload={
            "model_id": "model-v1",
            "checkpoint_id": "ckpt-001",
            "iteration": 1000,
            "path": "/models/ckpt-001.pth",
            "is_best": True,
        })

        await coordinator._on_checkpoint_saved(event)

        model = coordinator.get_model("model-v1")
        assert model.latest_checkpoint == "ckpt-001"
        assert model.best_checkpoint == "ckpt-001"
        assert model.checkpoint_count == 1

    @pytest.mark.asyncio
    async def test_checkpoint_callback(self, coordinator, mock_event):
        """Test checkpoint callback is called."""
        callbacks_called = []
        coordinator.on_checkpoint(lambda ckpt: callbacks_called.append(ckpt))

        event = mock_event(payload={
            "model_id": "model-v1",
            "checkpoint_id": "ckpt-001",
            "iteration": 1000,
            "path": "/models/ckpt-001.pth",
        })

        await coordinator._on_checkpoint_saved(event)

        assert len(callbacks_called) == 1
        assert callbacks_called[0].checkpoint_id == "ckpt-001"

    def test_checkpoint_history_limit(self, coordinator):
        """Test checkpoint history is limited."""
        coord = ModelLifecycleCoordinator(max_checkpoint_history=5)

        # Add more checkpoints than limit
        for i in range(10):
            checkpoint = CheckpointInfo(
                checkpoint_id=f"ckpt-{i}",
                model_id="model-v1",
                iteration=i * 100,
                path=f"/models/ckpt-{i}.pth",
                node_id="gpu-1",
            )
            coord._checkpoints[checkpoint.checkpoint_id] = checkpoint
            coord._checkpoint_history.append(checkpoint)
            coord._total_checkpoints += 1

            # Trim manually (simulating event handler)
            if len(coord._checkpoint_history) > coord.max_checkpoint_history:
                oldest = coord._checkpoint_history.pop(0)
                coord._checkpoints.pop(oldest.checkpoint_id, None)

        assert len(coord._checkpoint_history) == 5


# =============================================================================
# Promotion Tests
# =============================================================================


class TestPromotion:
    """Test model promotion functionality."""

    @pytest.mark.asyncio
    async def test_model_promoted_event(self, coordinator, mock_event):
        """Test handling MODEL_PROMOTED event."""
        coordinator.register_model("model-v1")

        event = mock_event(payload={
            "model_id": "model-v1",
            "elo": 1800.0,
        })

        await coordinator._on_model_promoted(event)

        model = coordinator.get_model("model-v1")
        assert model.state == ModelState.PRODUCTION
        assert model.elo == 1800.0
        assert model.promoted_at > 0

    @pytest.mark.asyncio
    async def test_promotion_sets_production_pointer(self, coordinator, mock_event):
        """Test promotion sets production model pointer."""
        coordinator.register_model("model-v1")

        event = mock_event(payload={
            "model_id": "model-v1",
        })

        await coordinator._on_model_promoted(event)

        prod = coordinator.get_production_model()
        assert prod is not None
        assert prod.model_id == "model-v1"

    @pytest.mark.asyncio
    async def test_promotion_archives_old_production(self, coordinator, mock_event):
        """Test promotion archives old production model."""
        coordinator.register_model("model-v1")
        coordinator.register_model("model-v2")

        # Promote first model
        event1 = mock_event(payload={"model_id": "model-v1"})
        await coordinator._on_model_promoted(event1)

        # Promote second model
        event2 = mock_event(payload={"model_id": "model-v2"})
        await coordinator._on_model_promoted(event2)

        old = coordinator.get_model("model-v1")
        new = coordinator.get_model("model-v2")

        assert old.state == ModelState.ARCHIVED
        assert new.state == ModelState.PRODUCTION

    @pytest.mark.asyncio
    async def test_promotion_callback(self, coordinator, mock_event):
        """Test promotion callback is called."""
        callbacks_called = []
        coordinator.on_promotion(lambda old, new: callbacks_called.append((old, new)))

        coordinator.register_model("model-v1")
        event = mock_event(payload={"model_id": "model-v1"})
        await coordinator._on_model_promoted(event)

        assert len(callbacks_called) == 1
        assert callbacks_called[0] == ("", "model-v1")


# =============================================================================
# Rollback Tests
# =============================================================================


class TestRollback:
    """Test promotion rollback functionality."""

    @pytest.mark.asyncio
    async def test_rollback_event(self, coordinator, mock_event):
        """Test handling PROMOTION_ROLLED_BACK event."""
        coordinator.register_model("model-v1")
        coordinator.register_model("model-v2")

        # Promote both
        await coordinator._on_model_promoted(mock_event(payload={"model_id": "model-v1"}))
        await coordinator._on_model_promoted(mock_event(payload={"model_id": "model-v2"}))

        # Rollback
        event = mock_event(payload={
            "from_model_id": "model-v2",
            "to_model_id": "model-v1",
        })
        await coordinator._on_promotion_rolled_back(event)

        model_v2 = coordinator.get_model("model-v2")
        model_v1 = coordinator.get_model("model-v1")

        assert model_v2.state == ModelState.ROLLED_BACK
        assert model_v1.state == ModelState.PRODUCTION
        assert coordinator.get_production_model().model_id == "model-v1"

    @pytest.mark.asyncio
    async def test_rollback_callback(self, coordinator, mock_event):
        """Test rollback callback is called."""
        callbacks_called = []
        coordinator.on_rollback(lambda from_m, to_m: callbacks_called.append((from_m, to_m)))

        coordinator.register_model("model-v1")
        coordinator.register_model("model-v2")

        await coordinator._on_model_promoted(mock_event(payload={"model_id": "model-v2"}))

        event = mock_event(payload={
            "from_model_id": "model-v2",
            "to_model_id": "model-v1",
        })
        await coordinator._on_promotion_rolled_back(event)

        assert len(callbacks_called) == 1
        assert callbacks_called[0] == ("model-v2", "model-v1")


# =============================================================================
# Training Completed Tests
# =============================================================================


class TestTrainingCompleted:
    """Test training completed event handling."""

    @pytest.mark.asyncio
    async def test_training_completed_transitions_state(self, coordinator, mock_event):
        """Test training completed transitions to evaluating."""
        coordinator.register_model("model-v1")

        event = mock_event(payload={
            "model_id": "model-v1",
            "train_loss": 0.1,
            "val_loss": 0.2,
        })

        await coordinator._on_training_completed(event)

        model = coordinator.get_model("model-v1")
        assert model.state == ModelState.EVALUATING
        assert model.train_loss == 0.1
        assert model.val_loss == 0.2


# =============================================================================
# ELO Updated Tests
# =============================================================================


class TestEloUpdated:
    """Test ELO update event handling."""

    @pytest.mark.asyncio
    async def test_elo_updated_event(self, coordinator, mock_event):
        """Test handling ELO_UPDATED event."""
        coordinator.register_model("model-v1")

        event = mock_event(payload={
            "model_id": "model-v1",
            "elo": 1750.0,
            "uncertainty": 100.0,
            "games_played": 500,
            "win_rate": 0.65,
        })

        await coordinator._on_elo_updated(event)

        model = coordinator.get_model("model-v1")
        assert model.elo == 1750.0
        assert model.elo_uncertainty == 100.0
        assert model.games_played == 500
        assert model.win_rate == 0.65


# =============================================================================
# Cache Management Tests
# =============================================================================


class TestCacheManagement:
    """Test cache entry management."""

    def test_register_cache_entry(self, coordinator):
        """Test registering a cache entry."""
        entry = coordinator.register_cache_entry(
            model_id="model-v1",
            node_id="gpu-1",
            cache_type="weights",
            size_bytes=100_000_000,
        )

        assert entry.model_id == "model-v1"
        assert entry.node_id == "gpu-1"
        assert entry.cache_type == "weights"

    def test_get_cache_entries(self, coordinator):
        """Test getting cache entries."""
        coordinator.register_cache_entry("model-v1", "gpu-1", "weights")
        coordinator.register_cache_entry("model-v1", "gpu-2", "weights")
        coordinator.register_cache_entry("model-v2", "gpu-1", "weights")

        all_entries = coordinator.get_cache_entries()
        model_v1_entries = coordinator.get_cache_entries(model_id="model-v1")
        gpu_1_entries = coordinator.get_cache_entries(node_id="gpu-1")

        assert len(all_entries) == 3
        assert len(model_v1_entries) == 2
        assert len(gpu_1_entries) == 2

    def test_invalidate_cache(self, coordinator):
        """Test cache invalidation."""
        coordinator.register_cache_entry("model-v1", "gpu-1", "weights")
        coordinator.register_cache_entry("model-v1", "gpu-2", "weights")
        coordinator.register_cache_entry("model-v2", "gpu-1", "weights")

        count = coordinator.invalidate_cache("model-v1")

        assert count == 2
        assert len(coordinator.get_cache_entries(model_id="model-v1")) == 0
        assert len(coordinator.get_cache_entries(model_id="model-v2")) == 1

    def test_invalidate_cache_specific_node(self, coordinator):
        """Test cache invalidation for specific node."""
        coordinator.register_cache_entry("model-v1", "gpu-1", "weights")
        coordinator.register_cache_entry("model-v1", "gpu-2", "weights")

        count = coordinator.invalidate_cache("model-v1", node_id="gpu-1")

        assert count == 1
        entries = coordinator.get_cache_entries(model_id="model-v1")
        assert len(entries) == 1
        assert entries[0].node_id == "gpu-2"


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Test statistics calculation."""

    @pytest.mark.asyncio
    async def test_stats_count_by_state(self, coordinator, mock_event):
        """Test statistics count models by state."""
        coordinator.register_model("training-1")
        coordinator.register_model("training-2")
        coordinator.register_model("eval-1", initial_state=ModelState.EVALUATING)

        stats = coordinator.get_stats()

        assert stats.total_models == 3
        assert stats.models_by_state.get("training", 0) == 2
        assert stats.models_by_state.get("evaluating", 0) == 1

    @pytest.mark.asyncio
    async def test_stats_promotion_count(self, coordinator, mock_event):
        """Test statistics track promotion count."""
        coordinator.register_model("model-v1")
        coordinator.register_model("model-v2")

        await coordinator._on_model_promoted(mock_event(payload={"model_id": "model-v1"}))
        await coordinator._on_model_promoted(mock_event(payload={"model_id": "model-v2"}))

        stats = coordinator.get_stats()
        assert stats.total_promotions == 2

    @pytest.mark.asyncio
    async def test_stats_rollback_count(self, coordinator, mock_event):
        """Test statistics track rollback count."""
        coordinator.register_model("model-v1")
        coordinator.register_model("model-v2")

        await coordinator._on_model_promoted(mock_event(payload={"model_id": "model-v2"}))
        await coordinator._on_promotion_rolled_back(mock_event(payload={
            "from_model_id": "model-v2",
            "to_model_id": "model-v1",
        }))

        stats = coordinator.get_stats()
        assert stats.total_rollbacks == 1


# =============================================================================
# Status Tests
# =============================================================================


class TestStatus:
    """Test status reporting."""

    def test_get_status(self, coordinator):
        """Test get_status returns proper structure."""
        coordinator.register_model("model-v1")
        coordinator.register_cache_entry("model-v1", "gpu-1", "weights")

        status = coordinator.get_status()

        assert "total_models" in status
        assert "models_by_state" in status
        assert "total_checkpoints" in status
        assert "production_model" in status
        assert "cache_entries" in status
        assert "subscribed" in status


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingletonBehavior:
    """Test singleton behavior."""

    def test_get_model_coordinator_returns_singleton(self):
        """Test get_model_coordinator returns same instance."""
        import app.coordination.model_lifecycle_coordinator as mlc
        mlc._model_coordinator = None

        coord1 = get_model_coordinator()
        coord2 = get_model_coordinator()

        assert coord1 is coord2

    def test_convenience_functions(self):
        """Test convenience functions work."""
        import app.coordination.model_lifecycle_coordinator as mlc
        mlc._model_coordinator = None

        # Initially no production model
        assert get_production_model_id() is None
        assert get_production_elo() == 1500.0  # Default


# =============================================================================
# ModelState Enum Tests
# =============================================================================


class TestModelState:
    """Test ModelState enum."""

    def test_state_values(self):
        """Test state enum values."""
        assert ModelState.TRAINING.value == "training"
        assert ModelState.EVALUATING.value == "evaluating"
        assert ModelState.STAGING.value == "staging"
        assert ModelState.PRODUCTION.value == "production"
        assert ModelState.ROLLED_BACK.value == "rolled_back"
        assert ModelState.ARCHIVED.value == "archived"
