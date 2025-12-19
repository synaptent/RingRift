"""ModelLifecycleCoordinator - Unified model state coordination (December 2025).

This module provides centralized monitoring of model lifecycle events including
checkpoints, promotions, and cache management. It tracks model state transitions
across the cluster.

Event Integration:
- Subscribes to CHECKPOINT_SAVED: Track checkpoint creation
- Subscribes to CHECKPOINT_LOADED: Track checkpoint restoration
- Subscribes to MODEL_PROMOTED: Track model promotions
- Subscribes to PROMOTION_ROLLED_BACK: Track rollbacks
- Subscribes to TRAINING_COMPLETED: Track training completions
- Subscribes to ELO_UPDATED: Track Elo changes

Key Responsibilities:
1. Track model state transitions (training -> eval -> staging -> production)
2. Track checkpoint creation and restoration
3. Coordinate model cache invalidation
4. Provide model lineage and history

Usage:
    from app.coordination.model_lifecycle_coordinator import (
        ModelLifecycleCoordinator,
        wire_model_events,
        get_model_coordinator,
    )

    # Wire model events
    coordinator = wire_model_events()

    # Get current production model
    prod = coordinator.get_production_model()
    print(f"Production model: {prod.model_id}, Elo: {prod.elo}")

    # Get model history
    history = coordinator.get_model_history("model_v42")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ModelState(Enum):
    """Model lifecycle states."""

    TRAINING = "training"
    EVALUATING = "evaluating"
    STAGING = "staging"
    PRODUCTION = "production"
    ROLLED_BACK = "rolled_back"
    ARCHIVED = "archived"


@dataclass
class CheckpointInfo:
    """Information about a checkpoint."""

    checkpoint_id: str
    model_id: str
    iteration: int
    path: str
    node_id: str
    created_at: float = field(default_factory=time.time)
    size_bytes: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    is_best: bool = False


@dataclass
class ModelRecord:
    """Record of a model's state and history."""

    model_id: str
    state: ModelState = ModelState.TRAINING
    created_at: float = field(default_factory=time.time)
    promoted_at: float = 0.0
    rolled_back_at: float = 0.0

    # Model metrics
    elo: float = 1500.0
    elo_uncertainty: float = 350.0
    games_played: int = 0
    win_rate: float = 0.0
    train_loss: float = 0.0
    val_loss: float = 0.0

    # Checkpoints
    latest_checkpoint: Optional[str] = None
    best_checkpoint: Optional[str] = None
    checkpoint_count: int = 0

    # Lineage
    parent_model_id: Optional[str] = None
    children_model_ids: List[str] = field(default_factory=list)

    # State history
    state_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CacheEntry:
    """Entry in the model cache."""

    model_id: str
    node_id: str
    cache_type: str  # "inference", "weights", "embedding"
    cached_at: float = field(default_factory=time.time)
    size_bytes: int = 0
    hits: int = 0
    last_access: float = field(default_factory=time.time)


@dataclass
class ModelLifecycleStats:
    """Aggregate model lifecycle statistics."""

    total_models: int = 0
    models_by_state: Dict[str, int] = field(default_factory=dict)
    total_checkpoints: int = 0
    total_promotions: int = 0
    total_rollbacks: int = 0
    current_production_model: Optional[str] = None
    current_production_elo: float = 0.0
    avg_promotion_time: float = 0.0  # Time from training to production
    cache_entries: int = 0


class ModelLifecycleCoordinator:
    """Coordinates model lifecycle events across the cluster.

    Tracks model state transitions, checkpoints, promotions, and cache entries
    to provide unified visibility into model management.
    """

    def __init__(
        self,
        max_checkpoint_history: int = 100,
        max_state_history_per_model: int = 50,
    ):
        """Initialize ModelLifecycleCoordinator.

        Args:
            max_checkpoint_history: Maximum checkpoints to track
            max_state_history_per_model: Maximum state transitions per model
        """
        self.max_checkpoint_history = max_checkpoint_history
        self.max_state_history_per_model = max_state_history_per_model

        # Model tracking
        self._models: Dict[str, ModelRecord] = {}
        self._production_model_id: Optional[str] = None

        # Checkpoint tracking
        self._checkpoints: Dict[str, CheckpointInfo] = {}
        self._checkpoint_history: List[CheckpointInfo] = []

        # Cache tracking
        self._cache_entries: Dict[str, CacheEntry] = {}  # key = f"{model_id}:{node_id}:{type}"

        # Statistics
        self._total_promotions = 0
        self._total_rollbacks = 0
        self._total_checkpoints = 0
        self._promotion_times: List[float] = []

        # Callbacks
        self._promotion_callbacks: List[Callable[[str, str], None]] = []  # old, new
        self._rollback_callbacks: List[Callable[[str, str], None]] = []  # from, to
        self._checkpoint_callbacks: List[Callable[[CheckpointInfo], None]] = []

        # Subscription state
        self._subscribed = False

    def subscribe_to_events(self) -> bool:
        """Subscribe to model lifecycle events.

        Returns:
            True if successfully subscribed
        """
        if self._subscribed:
            return True

        try:
            from app.distributed.data_events import DataEventType, get_event_bus

            bus = get_event_bus()

            bus.subscribe(DataEventType.CHECKPOINT_SAVED, self._on_checkpoint_saved)
            bus.subscribe(DataEventType.CHECKPOINT_LOADED, self._on_checkpoint_loaded)
            bus.subscribe(DataEventType.MODEL_PROMOTED, self._on_model_promoted)
            bus.subscribe(DataEventType.PROMOTION_ROLLED_BACK, self._on_promotion_rolled_back)
            bus.subscribe(DataEventType.TRAINING_COMPLETED, self._on_training_completed)
            bus.subscribe(DataEventType.ELO_UPDATED, self._on_elo_updated)

            self._subscribed = True
            logger.info("[ModelLifecycleCoordinator] Subscribed to model events")
            return True

        except ImportError:
            logger.warning("[ModelLifecycleCoordinator] data_events not available")
            return False
        except Exception as e:
            logger.error(f"[ModelLifecycleCoordinator] Failed to subscribe: {e}")
            return False

    def _ensure_model_record(self, model_id: str) -> ModelRecord:
        """Ensure a model record exists."""
        if model_id not in self._models:
            self._models[model_id] = ModelRecord(model_id=model_id)
        return self._models[model_id]

    def _record_state_transition(
        self, model: ModelRecord, new_state: ModelState, reason: str = ""
    ) -> None:
        """Record a state transition for a model."""
        old_state = model.state
        model.state = new_state

        transition = {
            "from_state": old_state.value,
            "to_state": new_state.value,
            "timestamp": time.time(),
            "reason": reason,
        }

        model.state_history.append(transition)

        # Trim history
        if len(model.state_history) > self.max_state_history_per_model:
            model.state_history = model.state_history[-self.max_state_history_per_model:]

        logger.debug(
            f"[ModelLifecycleCoordinator] Model {model.model_id} state: "
            f"{old_state.value} -> {new_state.value}"
        )

    async def _on_checkpoint_saved(self, event) -> None:
        """Handle CHECKPOINT_SAVED event."""
        payload = event.payload
        model_id = payload.get("model_id", "")
        checkpoint_id = payload.get("checkpoint_id", f"ckpt_{int(time.time())}")

        checkpoint = CheckpointInfo(
            checkpoint_id=checkpoint_id,
            model_id=model_id,
            iteration=payload.get("iteration", 0),
            path=payload.get("path", ""),
            node_id=payload.get("node_id", ""),
            size_bytes=payload.get("size_bytes", 0),
            metrics=payload.get("metrics", {}),
            is_best=payload.get("is_best", False),
        )

        # Track checkpoint
        self._checkpoints[checkpoint_id] = checkpoint
        self._checkpoint_history.append(checkpoint)
        self._total_checkpoints += 1

        # Trim history
        if len(self._checkpoint_history) > self.max_checkpoint_history:
            oldest = self._checkpoint_history.pop(0)
            self._checkpoints.pop(oldest.checkpoint_id, None)

        # Update model record
        model = self._ensure_model_record(model_id)
        model.latest_checkpoint = checkpoint_id
        model.checkpoint_count += 1
        if checkpoint.is_best:
            model.best_checkpoint = checkpoint_id

        # Notify callbacks
        for callback in self._checkpoint_callbacks:
            try:
                callback(checkpoint)
            except Exception as e:
                logger.error(f"[ModelLifecycleCoordinator] Checkpoint callback error: {e}")

        logger.debug(
            f"[ModelLifecycleCoordinator] Checkpoint saved: {checkpoint_id} "
            f"for model {model_id}, iteration {checkpoint.iteration}"
        )

    async def _on_checkpoint_loaded(self, event) -> None:
        """Handle CHECKPOINT_LOADED event."""
        payload = event.payload
        checkpoint_id = payload.get("checkpoint_id", "")
        model_id = payload.get("model_id", "")
        node_id = payload.get("node_id", "")

        logger.debug(
            f"[ModelLifecycleCoordinator] Checkpoint loaded: {checkpoint_id} "
            f"on {node_id}"
        )

        # Update cache entry
        cache_key = f"{model_id}:{node_id}:weights"
        if cache_key in self._cache_entries:
            self._cache_entries[cache_key].hits += 1
            self._cache_entries[cache_key].last_access = time.time()
        else:
            self._cache_entries[cache_key] = CacheEntry(
                model_id=model_id,
                node_id=node_id,
                cache_type="weights",
            )

    async def _on_model_promoted(self, event) -> None:
        """Handle MODEL_PROMOTED event."""
        payload = event.payload
        model_id = payload.get("model_id", "")
        old_production = self._production_model_id

        # Update model state
        model = self._ensure_model_record(model_id)
        model.promoted_at = time.time()
        self._record_state_transition(model, ModelState.PRODUCTION, "promoted")

        # Calculate promotion time
        if model.created_at > 0:
            promotion_time = model.promoted_at - model.created_at
            self._promotion_times.append(promotion_time)
            if len(self._promotion_times) > 100:
                self._promotion_times = self._promotion_times[-100:]

        # Archive old production model
        if old_production and old_production in self._models:
            old_model = self._models[old_production]
            self._record_state_transition(old_model, ModelState.ARCHIVED, "replaced")

        # Update production pointer
        self._production_model_id = model_id
        self._total_promotions += 1

        # Update Elo if provided
        if "elo" in payload:
            model.elo = payload["elo"]

        # Notify callbacks
        for callback in self._promotion_callbacks:
            try:
                callback(old_production or "", model_id)
            except Exception as e:
                logger.error(f"[ModelLifecycleCoordinator] Promotion callback error: {e}")

        logger.info(
            f"[ModelLifecycleCoordinator] Model promoted: {model_id} "
            f"(Elo: {model.elo:.0f})"
        )

    async def _on_promotion_rolled_back(self, event) -> None:
        """Handle PROMOTION_ROLLED_BACK event."""
        payload = event.payload
        from_model = payload.get("from_model_id", "")
        to_model = payload.get("to_model_id", "")

        # Update from model state
        if from_model in self._models:
            model = self._models[from_model]
            model.rolled_back_at = time.time()
            self._record_state_transition(model, ModelState.ROLLED_BACK, "rollback")

        # Restore to model as production
        if to_model:
            to_model_record = self._ensure_model_record(to_model)
            self._record_state_transition(
                to_model_record, ModelState.PRODUCTION, "restored after rollback"
            )
            self._production_model_id = to_model

        self._total_rollbacks += 1

        # Notify callbacks
        for callback in self._rollback_callbacks:
            try:
                callback(from_model, to_model)
            except Exception as e:
                logger.error(f"[ModelLifecycleCoordinator] Rollback callback error: {e}")

        logger.warning(
            f"[ModelLifecycleCoordinator] Promotion rolled back: "
            f"{from_model} -> {to_model}"
        )

    async def _on_training_completed(self, event) -> None:
        """Handle TRAINING_COMPLETED event."""
        payload = event.payload
        model_id = payload.get("model_id", "")

        model = self._ensure_model_record(model_id)

        # Update metrics
        model.train_loss = payload.get("train_loss", model.train_loss)
        model.val_loss = payload.get("val_loss", model.val_loss)

        # Transition to evaluating
        if model.state == ModelState.TRAINING:
            self._record_state_transition(model, ModelState.EVALUATING, "training complete")

        logger.debug(
            f"[ModelLifecycleCoordinator] Training completed for {model_id}, "
            f"val_loss={model.val_loss:.4f}"
        )

    async def _on_elo_updated(self, event) -> None:
        """Handle ELO_UPDATED event."""
        payload = event.payload
        model_id = payload.get("model_id", "")

        if model_id in self._models:
            model = self._models[model_id]
            model.elo = payload.get("elo", model.elo)
            model.elo_uncertainty = payload.get("uncertainty", model.elo_uncertainty)
            model.games_played = payload.get("games_played", model.games_played)
            model.win_rate = payload.get("win_rate", model.win_rate)

    def register_model(
        self,
        model_id: str,
        parent_model_id: Optional[str] = None,
        initial_state: ModelState = ModelState.TRAINING,
    ) -> ModelRecord:
        """Manually register a new model.

        Returns:
            The created ModelRecord
        """
        model = self._ensure_model_record(model_id)
        model.state = initial_state
        model.parent_model_id = parent_model_id

        if parent_model_id and parent_model_id in self._models:
            self._models[parent_model_id].children_model_ids.append(model_id)

        return model

    def update_model_state(self, model_id: str, new_state: ModelState, reason: str = "") -> bool:
        """Manually update a model's state.

        Returns:
            True if model was found and updated
        """
        if model_id not in self._models:
            return False

        self._record_state_transition(self._models[model_id], new_state, reason)
        return True

    def register_cache_entry(
        self,
        model_id: str,
        node_id: str,
        cache_type: str,
        size_bytes: int = 0,
    ) -> CacheEntry:
        """Register a cache entry.

        Returns:
            The created CacheEntry
        """
        cache_key = f"{model_id}:{node_id}:{cache_type}"
        entry = CacheEntry(
            model_id=model_id,
            node_id=node_id,
            cache_type=cache_type,
            size_bytes=size_bytes,
        )
        self._cache_entries[cache_key] = entry
        return entry

    def invalidate_cache(self, model_id: str, node_id: Optional[str] = None) -> int:
        """Invalidate cache entries for a model.

        Args:
            model_id: Model to invalidate
            node_id: Specific node, or None for all nodes

        Returns:
            Number of entries invalidated
        """
        keys_to_remove = []
        for key, entry in self._cache_entries.items():
            if entry.model_id == model_id:
                if node_id is None or entry.node_id == node_id:
                    keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._cache_entries[key]

        return len(keys_to_remove)

    def on_promotion(self, callback: Callable[[str, str], None]) -> None:
        """Register callback for model promotions.

        Args:
            callback: Function(old_model_id, new_model_id)
        """
        self._promotion_callbacks.append(callback)

    def on_rollback(self, callback: Callable[[str, str], None]) -> None:
        """Register callback for rollbacks.

        Args:
            callback: Function(from_model_id, to_model_id)
        """
        self._rollback_callbacks.append(callback)

    def on_checkpoint(self, callback: Callable[[CheckpointInfo], None]) -> None:
        """Register callback for checkpoint saves."""
        self._checkpoint_callbacks.append(callback)

    def get_model(self, model_id: str) -> Optional[ModelRecord]:
        """Get a model record by ID."""
        return self._models.get(model_id)

    def get_production_model(self) -> Optional[ModelRecord]:
        """Get the current production model."""
        if self._production_model_id:
            return self._models.get(self._production_model_id)
        return None

    def get_models_by_state(self, state: ModelState) -> List[ModelRecord]:
        """Get all models in a specific state."""
        return [m for m in self._models.values() if m.state == state]

    def get_model_history(self, model_id: str) -> List[Dict[str, Any]]:
        """Get state transition history for a model."""
        if model_id in self._models:
            return list(self._models[model_id].state_history)
        return []

    def get_checkpoints(self, model_id: Optional[str] = None) -> List[CheckpointInfo]:
        """Get checkpoints, optionally filtered by model."""
        if model_id:
            return [c for c in self._checkpoint_history if c.model_id == model_id]
        return list(self._checkpoint_history)

    def get_cache_entries(
        self, model_id: Optional[str] = None, node_id: Optional[str] = None
    ) -> List[CacheEntry]:
        """Get cache entries with optional filtering."""
        entries = list(self._cache_entries.values())

        if model_id:
            entries = [e for e in entries if e.model_id == model_id]
        if node_id:
            entries = [e for e in entries if e.node_id == node_id]

        return entries

    def get_stats(self) -> ModelLifecycleStats:
        """Get aggregate model lifecycle statistics."""
        # Count by state
        by_state: Dict[str, int] = {}
        for model in self._models.values():
            by_state[model.state.value] = by_state.get(model.state.value, 0) + 1

        # Average promotion time
        avg_promotion = (
            sum(self._promotion_times) / len(self._promotion_times)
            if self._promotion_times
            else 0.0
        )

        # Current production Elo
        prod_elo = 0.0
        if self._production_model_id and self._production_model_id in self._models:
            prod_elo = self._models[self._production_model_id].elo

        return ModelLifecycleStats(
            total_models=len(self._models),
            models_by_state=by_state,
            total_checkpoints=self._total_checkpoints,
            total_promotions=self._total_promotions,
            total_rollbacks=self._total_rollbacks,
            current_production_model=self._production_model_id,
            current_production_elo=prod_elo,
            avg_promotion_time=avg_promotion,
            cache_entries=len(self._cache_entries),
        )

    def get_status(self) -> Dict[str, Any]:
        """Get coordinator status for monitoring."""
        stats = self.get_stats()

        return {
            "total_models": stats.total_models,
            "models_by_state": stats.models_by_state,
            "total_checkpoints": stats.total_checkpoints,
            "total_promotions": stats.total_promotions,
            "total_rollbacks": stats.total_rollbacks,
            "production_model": stats.current_production_model,
            "production_elo": round(stats.current_production_elo, 0),
            "avg_promotion_time": round(stats.avg_promotion_time, 0),
            "cache_entries": stats.cache_entries,
            "subscribed": self._subscribed,
        }


# =============================================================================
# Singleton and convenience functions
# =============================================================================

_model_coordinator: Optional[ModelLifecycleCoordinator] = None


def get_model_coordinator() -> ModelLifecycleCoordinator:
    """Get the global ModelLifecycleCoordinator singleton."""
    global _model_coordinator
    if _model_coordinator is None:
        _model_coordinator = ModelLifecycleCoordinator()
    return _model_coordinator


def wire_model_events() -> ModelLifecycleCoordinator:
    """Wire model lifecycle events to the coordinator.

    Returns:
        The wired ModelLifecycleCoordinator instance
    """
    coordinator = get_model_coordinator()
    coordinator.subscribe_to_events()
    return coordinator


def get_production_model_id() -> Optional[str]:
    """Convenience function to get current production model ID."""
    prod = get_model_coordinator().get_production_model()
    return prod.model_id if prod else None


def get_production_elo() -> float:
    """Convenience function to get current production model Elo."""
    prod = get_model_coordinator().get_production_model()
    return prod.elo if prod else 1500.0


__all__ = [
    "ModelLifecycleCoordinator",
    "ModelState",
    "ModelRecord",
    "CheckpointInfo",
    "CacheEntry",
    "ModelLifecycleStats",
    "get_model_coordinator",
    "wire_model_events",
    "get_production_model_id",
    "get_production_elo",
]
