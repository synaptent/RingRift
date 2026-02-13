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
- Subscribes to MODEL_CORRUPTED: Handle corrupted model recovery (December 2025)
- Subscribes to MODEL_NOT_FOUND: Trigger model sync when model missing (January 2026)
- Subscribes to REGRESSION_DETECTED: Handle model performance regression (December 2025)

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
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.coordination.contracts import CoordinatorStatus, HealthCheckResult
from app.coordination.event_handler_utils import extract_config_from_path, extract_config_key

logger = logging.getLogger(__name__)

# Generation tracking for research demonstration
try:
    from app.coordination.generation_tracker import get_generation_tracker
    HAS_GENERATION_TRACKER = True
except ImportError:
    HAS_GENERATION_TRACKER = False
    get_generation_tracker = None


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
    metrics: dict[str, float] = field(default_factory=dict)
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
    latest_checkpoint: str | None = None
    best_checkpoint: str | None = None
    checkpoint_count: int = 0

    # Lineage
    parent_model_id: str | None = None
    children_model_ids: list[str] = field(default_factory=list)

    # State history
    state_history: list[dict[str, Any]] = field(default_factory=list)


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
    models_by_state: dict[str, int] = field(default_factory=dict)
    total_checkpoints: int = 0
    total_promotions: int = 0
    total_rollbacks: int = 0
    current_production_model: str | None = None
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
        self._models: dict[str, ModelRecord] = {}
        self._production_model_id: str | None = None

        # Checkpoint tracking
        self._checkpoints: dict[str, CheckpointInfo] = {}
        self._checkpoint_history: list[CheckpointInfo] = []

        # Cache tracking
        self._cache_entries: dict[str, CacheEntry] = {}  # key = f"{model_id}:{node_id}:{type}"

        # Statistics
        self._total_promotions = 0
        self._total_rollbacks = 0
        self._total_checkpoints = 0
        self._promotion_times: list[float] = []

        # Callbacks
        self._promotion_callbacks: list[Callable[[str, str], None]] = []  # old, new
        self._rollback_callbacks: list[Callable[[str, str], None]] = []  # from, to
        self._checkpoint_callbacks: list[Callable[[CheckpointInfo], None]] = []

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
            from app.coordination.event_router import get_router
            from app.coordination.event_router import DataEventType

            router = get_router()

            router.subscribe(DataEventType.CHECKPOINT_SAVED.value, self._on_checkpoint_saved)
            router.subscribe(DataEventType.CHECKPOINT_LOADED.value, self._on_checkpoint_loaded)
            router.subscribe(DataEventType.MODEL_PROMOTED.value, self._on_model_promoted)
            router.subscribe(DataEventType.PROMOTION_ROLLED_BACK.value, self._on_promotion_rolled_back)
            router.subscribe(DataEventType.PROMOTION_FAILED.value, self._on_promotion_failed)
            router.subscribe(DataEventType.TRAINING_COMPLETED.value, self._on_training_completed)
            router.subscribe(DataEventType.ELO_UPDATED.value, self._on_elo_updated)
            router.subscribe(DataEventType.MODEL_CORRUPTED.value, self._on_model_corrupted)
            router.subscribe(DataEventType.MODEL_NOT_FOUND.value, self._on_model_not_found)
            router.subscribe(DataEventType.REGRESSION_DETECTED.value, self._on_regression_detected)

            self._subscribed = True
            logger.info("[ModelLifecycleCoordinator] Subscribed to model events")
            return True

        except ImportError:
            logger.warning("[ModelLifecycleCoordinator] data_events not available")
            return False
        except Exception as e:
            logger.error(f"[ModelLifecycleCoordinator] Failed to subscribe: {e}")
            return False

    def unsubscribe_from_events(self) -> None:
        """Unsubscribe from model lifecycle events.

        December 27, 2025: Added to fix missing cleanup method (Wave 4 Phase 1).
        This enables graceful shutdown without memory leaks from orphaned callbacks.
        """
        if not self._subscribed:
            return

        try:
            from app.coordination.event_router import get_router
            from app.coordination.event_router import DataEventType

            router = get_router()

            # Unsubscribe from all events
            router.unsubscribe(DataEventType.CHECKPOINT_SAVED.value, self._on_checkpoint_saved)
            router.unsubscribe(DataEventType.CHECKPOINT_LOADED.value, self._on_checkpoint_loaded)
            router.unsubscribe(DataEventType.MODEL_PROMOTED.value, self._on_model_promoted)
            router.unsubscribe(DataEventType.PROMOTION_ROLLED_BACK.value, self._on_promotion_rolled_back)
            router.unsubscribe(DataEventType.PROMOTION_FAILED.value, self._on_promotion_failed)
            router.unsubscribe(DataEventType.TRAINING_COMPLETED.value, self._on_training_completed)
            router.unsubscribe(DataEventType.ELO_UPDATED.value, self._on_elo_updated)
            router.unsubscribe(DataEventType.MODEL_CORRUPTED.value, self._on_model_corrupted)
            router.unsubscribe(DataEventType.MODEL_NOT_FOUND.value, self._on_model_not_found)
            router.unsubscribe(DataEventType.REGRESSION_DETECTED.value, self._on_regression_detected)

            logger.info("[ModelLifecycleCoordinator] Unsubscribed from model events")

        except (ImportError, ValueError, KeyError, AttributeError) as e:
            logger.debug(f"[ModelLifecycleCoordinator] Error unsubscribing: {e}")
        finally:
            # Always mark as unsubscribed even if cleanup fails
            self._subscribed = False

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

        # Notify ImprovementOptimizer for positive feedback acceleration
        try:
            from app.training.improvement_optimizer import record_promotion_success

            config_key = extract_config_key(payload)
            elo_gain = payload.get("elo_gain", payload.get("elo_delta", 0.0))

            if config_key:
                record_promotion_success(
                    config_key=config_key,
                    elo_gain=elo_gain if elo_gain else 0.0,
                    model_id=model_id,
                )
                logger.debug(
                    f"[ModelLifecycleCoordinator] Recorded promotion success for {config_key} "
                    f"in ImprovementOptimizer (elo_gain={elo_gain})"
                )
        except ImportError:
            pass  # ImprovementOptimizer not available
        except Exception as e:
            logger.debug(f"[ModelLifecycleCoordinator] Could not update ImprovementOptimizer: {e}")

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

    async def _on_promotion_failed(self, event) -> None:
        """Handle PROMOTION_FAILED event - track failed promotions.

        December 2025: Wire PROMOTION_FAILED to track failed promotions and
        update model state history. This enables visibility into promotion
        failures and triggers potential remediation actions.
        """
        payload = event.payload if hasattr(event, "payload") else event
        model_id = payload.get("model_id", "")
        config_key = extract_config_key(payload)
        error = payload.get("error", "unknown")
        reason = payload.get("reason", error)

        # Track the failure in model state
        if model_id:
            model = self._ensure_model_record(model_id)
            self._record_state_transition(
                model, ModelState.EVALUATING, f"promotion_failed: {reason}"
            )

        logger.warning(
            f"[ModelLifecycleCoordinator] Promotion failed for {model_id or config_key}: "
            f"{reason}"
        )

        # Notify ImprovementOptimizer of promotion failure for negative feedback
        try:
            from app.training.improvement_optimizer import record_promotion_failure

            if config_key:
                record_promotion_failure(
                    config_key=config_key,
                    model_id=model_id,
                    reason=reason,
                )
                logger.debug(
                    f"[ModelLifecycleCoordinator] Recorded promotion failure for {config_key} "
                    f"in ImprovementOptimizer"
                )
        except ImportError:
            pass  # ImprovementOptimizer not available
        except Exception as e:
            logger.debug(f"[ModelLifecycleCoordinator] Could not update ImprovementOptimizer: {e}")

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

        # Notify ImprovementOptimizer for positive feedback acceleration
        try:
            from app.training.improvement_optimizer import record_training_complete

            config_key = extract_config_key(payload)
            duration_seconds = payload.get("duration_seconds", 0.0)
            val_loss = payload.get("val_loss", model.val_loss)
            calibration_ece = payload.get("calibration_ece")

            if config_key and duration_seconds > 0:
                record_training_complete(
                    config_key=config_key,
                    duration_seconds=duration_seconds,
                    val_loss=val_loss if val_loss else 0.0,
                    calibration_ece=calibration_ece,
                )
                logger.debug(
                    f"[ModelLifecycleCoordinator] Recorded training completion for {config_key} "
                    f"in ImprovementOptimizer"
                )
        except ImportError:
            pass  # ImprovementOptimizer not available
        except Exception as e:
            logger.debug(f"[ModelLifecycleCoordinator] Could not update ImprovementOptimizer: {e}")

        logger.debug(
            f"[ModelLifecycleCoordinator] Training completed for {model_id}, "
            f"val_loss={model.val_loss:.4f}"
        )

        # Record generation for research demonstration (January 2026)
        if HAS_GENERATION_TRACKER and get_generation_tracker is not None:
            try:
                tracker = get_generation_tracker()
                config_key = extract_config_key(payload)
                if config_key:
                    # Parse board_type and num_players from config_key (e.g., "hex8_2p")
                    parts = config_key.rsplit("_", 1)
                    if len(parts) == 2 and parts[1].endswith("p"):
                        board_type = parts[0]
                        num_players = int(parts[1][:-1])

                        # Get training metadata
                        training_samples = payload.get("training_samples", 0)
                        training_games = payload.get("training_games", 0)
                        # Feb 2026: Construct canonical path when payload lacks model_path
                        # (98.7% of generations had empty model_path, blocking evaluation)
                        model_path = (
                            payload.get("model_path")
                            or payload.get("checkpoint_path")
                            or model_id
                            or f"models/canonical_{board_type}_{num_players}p.pth"
                        )

                        # Find parent generation (latest for this config)
                        latest = tracker.get_latest_generation(board_type, num_players)
                        parent_id = latest.generation_id if latest else None

                        # Record the new generation
                        gen_id = tracker.record_generation(
                            model_path=model_path,
                            board_type=board_type,
                            num_players=num_players,
                            parent_generation_id=parent_id,
                            training_games=training_games,
                            training_samples=training_samples,
                        )
                        logger.info(
                            f"[ModelLifecycleCoordinator] Recorded generation {gen_id} "
                            f"for {config_key} (parent={parent_id})"
                        )
            except Exception as e:
                logger.warning(f"[ModelLifecycleCoordinator] Failed to record generation: {e}")

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

        # Record Elo snapshot in generation tracker (January 2026)
        if HAS_GENERATION_TRACKER and get_generation_tracker is not None:
            try:
                config_key = extract_config_key(payload)
                elo = payload.get("elo")
                games_played = payload.get("games_played", 0)

                if config_key and elo is not None:
                    parts = config_key.rsplit("_", 1)
                    if len(parts) == 2 and parts[1].endswith("p"):
                        board_type = parts[0]
                        num_players = int(parts[1][:-1])

                        tracker = get_generation_tracker()
                        latest = tracker.get_latest_generation(board_type, num_players)
                        if latest:
                            tracker.record_elo_snapshot(
                                generation_id=latest.generation_id,
                                elo=elo,
                                games_played=games_played,
                            )
                            logger.debug(
                                f"[ModelLifecycleCoordinator] Recorded Elo snapshot: "
                                f"gen {latest.generation_id} = {elo:.0f}"
                            )
            except Exception as e:
                logger.debug(f"[ModelLifecycleCoordinator] Failed to record Elo snapshot: {e}")

    async def _on_model_corrupted(self, event) -> None:
        """Handle MODEL_CORRUPTED event - trigger model recovery/re-download.

        December 2025: Wired handler for MODEL_CORRUPTED events.
        When a model file is detected as corrupted, this handler:
        1. Records the corruption in model history
        2. Invalidates any cached versions
        3. Attempts to recover from cluster peers or rollback to previous version
        """
        payload = event.payload if hasattr(event, "payload") else event
        model_path = payload.get("model_path", "")
        model_id = payload.get("model_id", "")
        corruption_type = payload.get("corruption_type", "unknown")
        error_msg = payload.get("error", "")
        node_id = payload.get("node_id", "")

        # Extract config key from model path or use provided
        config_key = extract_config_key(payload)
        if not config_key and model_path:
            # December 30, 2025: Use consolidated utility for path extraction
            config_key = extract_config_from_path(model_path) or ""

        logger.warning(
            f"[ModelLifecycleCoordinator] MODEL_CORRUPTED: {model_path} "
            f"(type={corruption_type}, node={node_id}, error={error_msg})"
        )

        # Update model state if we have a record
        if model_id and model_id in self._models:
            model = self._models[model_id]
            self._record_state_transition(
                model, ModelState.ARCHIVED, f"corrupted: {corruption_type}"
            )

        # Invalidate caches for corrupted model
        if model_id:
            invalidated = self.invalidate_cache(model_id, node_id if node_id else None)
            if invalidated > 0:
                logger.info(
                    f"[ModelLifecycleCoordinator] Invalidated {invalidated} cache entries "
                    f"for corrupted model {model_id}"
                )

        # Attempt recovery via model distribution daemon
        try:
            from app.coordination.event_router import get_router, DataEventType

            router = get_router()

            # Emit MODEL_SYNC_REQUESTED to trigger re-download from healthy nodes
            await router.publish(
                DataEventType.MODEL_SYNC_REQUESTED.value,
                {
                    "model_id": model_id,
                    "config_key": config_key,
                    "model_path": model_path,
                    "target_node": node_id,
                    "reason": f"corruption_recovery:{corruption_type}",
                    "priority": "high",
                },
            )
            logger.info(
                f"[ModelLifecycleCoordinator] Requested model re-sync for {config_key} "
                f"to {node_id or 'all nodes'}"
            )

        except Exception as e:
            logger.error(f"[ModelLifecycleCoordinator] Failed to request model recovery: {e}")

        # If this was the production model, attempt rollback
        if model_id and self._production_model_id == model_id:
            logger.warning(
                f"[ModelLifecycleCoordinator] Production model corrupted! "
                f"Attempting rollback..."
            )
            try:
                from app.training.rollback_manager import RollbackManager
                from app.training.model_registry import get_model_registry

                registry = get_model_registry()
                rollback_mgr = RollbackManager(registry)

                result = rollback_mgr.rollback_model(
                    model_id=config_key or model_id,
                    reason=f"Model corruption detected: {corruption_type}",
                    triggered_by="model_corrupted_handler",
                )

                if result.get("success"):
                    logger.info(
                        f"[ModelLifecycleCoordinator] Rollback successful: "
                        f"v{result.get('from_version')} -> v{result.get('to_version')}"
                    )
                else:
                    logger.error(
                        f"[ModelLifecycleCoordinator] Rollback failed: {result.get('error')}"
                    )

            except ImportError:
                logger.warning(
                    "[ModelLifecycleCoordinator] RollbackManager not available for recovery"
                )
            except Exception as e:
                logger.error(f"[ModelLifecycleCoordinator] Rollback failed: {e}")

    async def _on_model_not_found(self, event) -> None:
        """Handle MODEL_NOT_FOUND event - trigger model sync from cluster peers.

        Sprint 17.9 (Jan 5, 2026): Added for stale model detection.
        When a selfplay job attempts to dispatch but the model file is missing,
        this handler triggers MODEL_SYNC_REQUESTED to fetch the model from
        another node in the cluster.
        """
        payload = event.payload if hasattr(event, "payload") else event
        config_key = payload.get("config_key", "")
        board_type = payload.get("board_type", "")
        num_players = payload.get("num_players", 0)
        model_version = payload.get("model_version", "v5")
        expected_path = payload.get("expected_path", "")
        node_id = payload.get("node_id", "")

        logger.warning(
            f"[ModelLifecycleCoordinator] MODEL_NOT_FOUND: {config_key} "
            f"(version={model_version}, path={expected_path}, node={node_id})"
        )

        # Construct model_id from config if available
        model_id = f"canonical_{config_key}" if config_key else ""

        # Emit MODEL_SYNC_REQUESTED to trigger download from healthy nodes
        try:
            from app.coordination.event_router import get_router, DataEventType

            router = get_router()

            await router.publish(
                DataEventType.MODEL_SYNC_REQUESTED.value,
                {
                    "model_id": model_id,
                    "config_key": config_key,
                    "board_type": board_type,
                    "num_players": num_players,
                    "model_version": model_version,
                    "model_path": expected_path,
                    "target_node": node_id,
                    "reason": "model_not_found",
                    "priority": "high",
                },
            )
            logger.info(
                f"[ModelLifecycleCoordinator] Requested model sync for {config_key} "
                f"to {node_id or 'local node'}"
            )

        except Exception as e:
            logger.error(f"[ModelLifecycleCoordinator] Failed to request model sync: {e}")

    async def _on_regression_detected(self, event) -> None:
        """Handle REGRESSION_DETECTED event - track and respond to model regression.

        December 2025: Wired handler for REGRESSION_DETECTED events.
        When a model shows performance regression (e.g., declining win rate,
        Elo drop), this handler:
        1. Records the regression in model history
        2. Notifies ImprovementOptimizer for negative feedback
        3. Triggers rollback if regression is severe and model is in production
        """
        payload = event.payload if hasattr(event, "payload") else event
        model_id = payload.get("model_id", "")
        config_key = extract_config_key(payload)
        regression_type = payload.get("regression_type", "unknown")
        severity = payload.get("severity", "moderate")  # mild, moderate, severe
        current_elo = payload.get("current_elo", 0.0)
        baseline_elo = payload.get("baseline_elo", 0.0)
        elo_drop = payload.get("elo_drop", baseline_elo - current_elo if baseline_elo else 0.0)
        win_rate = payload.get("win_rate", 0.0)
        games_analyzed = payload.get("games_analyzed", 0)
        source = payload.get("source", "unknown")

        logger.warning(
            f"[ModelLifecycleCoordinator] REGRESSION_DETECTED: {model_id or config_key} "
            f"(type={regression_type}, severity={severity}, elo_drop={elo_drop:.1f}, "
            f"win_rate={win_rate:.1%}, games={games_analyzed}, source={source})"
        )

        # Update model state if we have a record
        if model_id and model_id in self._models:
            model = self._models[model_id]
            self._record_state_transition(
                model,
                ModelState.EVALUATING,
                f"regression_detected: {regression_type} ({severity})",
            )

        # Notify ImprovementOptimizer of regression for negative feedback
        try:
            from app.training.improvement_optimizer import record_regression

            if config_key or model_id:
                record_regression(
                    config_key=config_key or model_id,
                    regression_type=regression_type,
                    severity=severity,
                    elo_drop=elo_drop,
                    win_rate=win_rate,
                )
                logger.debug(
                    f"[ModelLifecycleCoordinator] Recorded regression for "
                    f"{config_key or model_id} in ImprovementOptimizer"
                )
        except ImportError:
            pass  # ImprovementOptimizer not available
        except Exception as e:
            logger.debug(f"[ModelLifecycleCoordinator] Could not update ImprovementOptimizer: {e}")

        # Trigger rollback for severe regression on production model
        if severity == "severe" and model_id and self._production_model_id == model_id:
            logger.warning(
                f"[ModelLifecycleCoordinator] Severe regression on production model! "
                f"Triggering automatic rollback..."
            )
            try:
                from app.training.rollback_manager import RollbackManager
                from app.training.model_registry import get_model_registry

                registry = get_model_registry()
                rollback_mgr = RollbackManager(registry)

                result = rollback_mgr.rollback_model(
                    model_id=config_key or model_id,
                    reason=f"Severe regression detected: {regression_type}, "
                    f"Elo drop={elo_drop:.1f}, win_rate={win_rate:.1%}",
                    triggered_by="regression_detected_handler",
                )

                if result.get("success"):
                    logger.info(
                        f"[ModelLifecycleCoordinator] Auto-rollback successful: "
                        f"v{result.get('from_version')} -> v{result.get('to_version')}"
                    )
                    # Emit rollback event
                    try:
                        from app.coordination.event_router import get_router, DataEventType

                        router = get_router()
                        await router.publish(
                            DataEventType.PROMOTION_ROLLED_BACK.value,
                            {
                                "from_model_id": model_id,
                                "to_model_id": result.get("to_model_id", ""),
                                "reason": f"auto_rollback:regression:{regression_type}",
                                "triggered_by": "ModelLifecycleCoordinator",
                            },
                        )
                    except (ImportError, RuntimeError, OSError) as emit_err:
                        logger.debug(
                            f"[ModelLifecycleCoordinator] Failed to emit rollback event: {emit_err}"
                        )
                else:
                    logger.error(
                        f"[ModelLifecycleCoordinator] Auto-rollback failed: {result.get('error')}"
                    )

            except ImportError:
                logger.warning(
                    "[ModelLifecycleCoordinator] RollbackManager not available for auto-rollback"
                )
            except Exception as e:
                logger.error(f"[ModelLifecycleCoordinator] Auto-rollback failed: {e}")

    def register_model(
        self,
        model_id: str,
        parent_model_id: str | None = None,
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

    def invalidate_cache(self, model_id: str, node_id: str | None = None) -> int:
        """Invalidate cache entries for a model.

        Args:
            model_id: Model to invalidate
            node_id: Specific node, or None for all nodes

        Returns:
            Number of entries invalidated
        """
        keys_to_remove = []
        for key, entry in self._cache_entries.items():
            if entry.model_id == model_id and (node_id is None or entry.node_id == node_id):
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

    def get_model(self, model_id: str) -> ModelRecord | None:
        """Get a model record by ID."""
        return self._models.get(model_id)

    def get_production_model(self) -> ModelRecord | None:
        """Get the current production model."""
        if self._production_model_id:
            return self._models.get(self._production_model_id)
        return None

    def get_models_by_state(self, state: ModelState) -> list[ModelRecord]:
        """Get all models in a specific state."""
        return [m for m in self._models.values() if m.state == state]

    def get_model_history(self, model_id: str) -> list[dict[str, Any]]:
        """Get state transition history for a model."""
        if model_id in self._models:
            return list(self._models[model_id].state_history)
        return []

    def get_checkpoints(self, model_id: str | None = None) -> list[CheckpointInfo]:
        """Get checkpoints, optionally filtered by model."""
        if model_id:
            return [c for c in self._checkpoint_history if c.model_id == model_id]
        return list(self._checkpoint_history)

    def get_cache_entries(
        self, model_id: str | None = None, node_id: str | None = None
    ) -> list[CacheEntry]:
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
        by_state: dict[str, int] = {}
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

    def get_status(self) -> dict[str, Any]:
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

    def health_check(self) -> HealthCheckResult:
        """Perform health check on model lifecycle coordinator (December 2025).

        Returns:
            HealthCheckResult with status and metrics.
        """
        stats = self.get_stats()

        # Calculate rollback rate
        promotions = stats.total_promotions
        rollbacks = stats.total_rollbacks
        rollback_rate = rollbacks / max(promotions, 1)

        # Overall health criteria
        healthy = (
            self._subscribed  # Must be subscribed to events
            and rollback_rate < 0.5  # Less than 50% rollback rate
        )

        status = CoordinatorStatus.RUNNING if healthy else CoordinatorStatus.DEGRADED
        message = f"High rollback rate {rollback_rate:.1%}" if rollback_rate >= 0.5 else ""

        return HealthCheckResult(
            healthy=healthy,
            status=status,
            message=message,
            details={
                "total_models": stats.total_models,
                "has_production_model": stats.current_production_model is not None,
                "production_model": stats.current_production_model,
                "total_promotions": promotions,
                "total_rollbacks": rollbacks,
                "rollback_rate": round(rollback_rate, 3),
                "subscribed": self._subscribed,
            },
        )


# =============================================================================
# Singleton and convenience functions
# =============================================================================

_model_coordinator: ModelLifecycleCoordinator | None = None


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


def get_production_model_id() -> str | None:
    """Convenience function to get current production model ID."""
    prod = get_model_coordinator().get_production_model()
    return prod.model_id if prod else None


def get_production_elo() -> float:
    """Convenience function to get current production model Elo."""
    prod = get_model_coordinator().get_production_model()
    return prod.elo if prod else 1500.0


__all__ = [
    "CacheEntry",
    "CheckpointInfo",
    "ModelLifecycleCoordinator",
    "ModelLifecycleStats",
    "ModelRecord",
    "ModelState",
    "get_model_coordinator",
    "get_production_elo",
    "get_production_model_id",
    "wire_model_events",
]
