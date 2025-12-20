"""Unified Cache Invalidation for Model Promotion (December 2025).

This module provides centralized cache invalidation when models are promoted.
It subscribes to MODEL_PROMOTED events and clears all relevant caches to ensure
the new model is loaded fresh.

Caches managed:
- model_cache: Neural network instances (app/ai/model_cache.py)
- move_cache: Move prediction cache (app/ai/move_cache.py)
- territory_cache: Territory scoring cache (app/ai/territory_cache.py)
- eval_cache: Evaluation results cache (app/training/eval_cache.py)
- export_cache: Model export cache (app/training/export_cache.py)

Usage:
    from app.ai.cache_invalidation import (
        wire_promotion_to_cache_invalidation,
        invalidate_all_caches,
    )

    # Wire MODEL_PROMOTED events to cache invalidation
    watcher = wire_promotion_to_cache_invalidation()

    # Or manually invalidate all caches
    invalidate_all_caches()
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CacheInvalidationResult:
    """Result of a cache invalidation operation."""
    cache_name: str
    success: bool
    items_cleared: int = 0
    error: str | None = None
    duration_ms: float = 0.0


@dataclass
class FullInvalidationResult:
    """Result of invalidating all caches."""
    total_success: bool = True
    caches_cleared: int = 0
    total_items_cleared: int = 0
    results: list[CacheInvalidationResult] = field(default_factory=list)
    trigger_reason: str = ""
    model_id: str = ""


class ModelPromotionCacheInvalidator:
    """Watches for model promotions and invalidates all caches.

    Subscribes to MODEL_PROMOTED events from the event bus and triggers
    cache invalidation for all managed caches.
    """

    def __init__(
        self,
        invalidation_cooldown_seconds: float = 5.0,
        clear_gpu_memory: bool = True,
    ):
        """Initialize the cache invalidator.

        Args:
            invalidation_cooldown_seconds: Minimum time between invalidations
            clear_gpu_memory: Also clear GPU/MPS caches
        """
        self.invalidation_cooldown_seconds = invalidation_cooldown_seconds
        self.clear_gpu_memory = clear_gpu_memory

        self._last_invalidation_time: float = 0.0
        self._invalidation_count: int = 0
        self._subscribed = False

        # Registry of cache invalidation functions
        self._cache_invalidators: dict[str, Callable[[], int]] = {}
        self._register_default_invalidators()

    def _register_default_invalidators(self) -> None:
        """Register default cache invalidation functions."""
        # Model cache
        def invalidate_model_cache() -> int:
            try:
                from app.ai.model_cache import clear_model_cache, get_cached_model_count
                count = get_cached_model_count()
                clear_model_cache()
                return count
            except ImportError:
                return 0

        # Move cache
        def invalidate_move_cache() -> int:
            try:
                from app.ai.move_cache import clear_move_cache, get_move_cache
                cache = get_move_cache()
                count = len(cache._cache) if hasattr(cache, '_cache') else 0
                clear_move_cache()
                return count
            except (ImportError, AttributeError):
                return 0

        # Territory cache
        def invalidate_territory_cache() -> int:
            try:
                from app.ai.territory_cache import get_territory_cache
                cache = get_territory_cache()
                count = len(cache._cache) if hasattr(cache, '_cache') else 0
                if hasattr(cache, 'clear'):
                    cache.clear()
                return count
            except (ImportError, AttributeError):
                return 0

        # Eval cache
        def invalidate_eval_cache() -> int:
            try:
                from app.training.eval_cache import get_eval_cache
                cache = get_eval_cache()
                count = cache.size() if hasattr(cache, 'size') else 0
                if hasattr(cache, 'clear'):
                    cache.clear()
                return count
            except (ImportError, AttributeError):
                return 0

        # Export cache
        def invalidate_export_cache() -> int:
            try:
                from app.training.export_cache import get_export_cache
                cache = get_export_cache()
                count = cache.size() if hasattr(cache, 'size') else 0
                if hasattr(cache, 'invalidate'):
                    cache.invalidate()
                elif hasattr(cache, 'clear'):
                    cache.clear()
                return count
            except (ImportError, AttributeError):
                return 0

        self._cache_invalidators = {
            "model_cache": invalidate_model_cache,
            "move_cache": invalidate_move_cache,
            "territory_cache": invalidate_territory_cache,
            "eval_cache": invalidate_eval_cache,
            "export_cache": invalidate_export_cache,
        }

    def register_cache_invalidator(
        self,
        name: str,
        invalidator: Callable[[], int],
    ) -> None:
        """Register a custom cache invalidation function.

        Args:
            name: Name of the cache
            invalidator: Function that clears the cache and returns items cleared
        """
        self._cache_invalidators[name] = invalidator
        logger.debug(f"[CacheInvalidator] Registered invalidator for {name}")

    def subscribe_to_promotion_events(self) -> bool:
        """Subscribe to MODEL_PROMOTED events from the event bus.

        Returns:
            True if successfully subscribed
        """
        try:
            from app.distributed.data_events import (
                DataEventType,
                get_event_bus,
            )

            bus = get_event_bus()
            bus.subscribe(DataEventType.MODEL_PROMOTED, self._on_model_promoted)
            self._subscribed = True
            logger.info("[CacheInvalidator] Subscribed to MODEL_PROMOTED events")
            return True
        except Exception as e:
            logger.warning(f"[CacheInvalidator] Failed to subscribe: {e}")
            return False

    def subscribe_to_all_invalidation_triggers(self) -> int:
        """Subscribe to all events that should trigger cache invalidation.

        This expands beyond MODEL_PROMOTED to include:
        - TRAINING_COMPLETED: New model trained, caches may reference old model
        - HYPERPARAMETER_UPDATED: Hyperparams changed, evaluation results stale
        - REGRESSION_DETECTED: Model regression, fresh evaluation needed
        - NAS_COMPLETED: New architecture found, model cache stale
        - CMAES_COMPLETED: Hyperparams optimized, evaluation cache stale

        Returns:
            Number of event types subscribed to
        """
        try:
            from app.distributed.data_events import (
                DataEventType,
                get_event_bus,
            )

            bus = get_event_bus()
            subscribed = 0

            # Core promotion event (always subscribe)
            if not self._subscribed:
                bus.subscribe(DataEventType.MODEL_PROMOTED, self._on_model_promoted)
                subscribed += 1

            # Training events - clear eval cache when new models are trained
            bus.subscribe(DataEventType.TRAINING_COMPLETED, self._on_training_completed)
            subscribed += 1

            # Hyperparameter changes - invalidate evaluation results
            bus.subscribe(DataEventType.HYPERPARAMETER_UPDATED, self._on_hyperparameter_updated)
            subscribed += 1

            # Regression detection - may need fresh evaluation
            bus.subscribe(DataEventType.REGRESSION_DETECTED, self._on_regression_detected)
            subscribed += 1

            # NAS completion - architecture changed
            bus.subscribe(DataEventType.NAS_COMPLETED, self._on_optimization_completed)
            subscribed += 1

            # CMA-ES completion - hyperparams changed
            bus.subscribe(DataEventType.CMAES_COMPLETED, self._on_optimization_completed)
            subscribed += 1

            self._subscribed = True
            logger.info(
                f"[CacheInvalidator] Subscribed to {subscribed} cache invalidation triggers"
            )
            return subscribed

        except Exception as e:
            logger.warning(f"[CacheInvalidator] Failed to subscribe to all triggers: {e}")
            return 0

    def _on_training_completed(self, event) -> None:
        """Handle TRAINING_COMPLETED - invalidate eval cache for fresh evaluation."""
        payload = event.payload if hasattr(event, 'payload') else {}
        config = payload.get("config", "")
        model_path = payload.get("model_path", "")

        logger.debug(f"[CacheInvalidator] Training completed: {config}")

        # Only invalidate eval cache - model cache refreshes on load
        self._invalidate_selective(
            caches=["eval_cache", "export_cache"],
            trigger_reason="training_completed",
            model_id=model_path,
        )

    def _on_hyperparameter_updated(self, event) -> None:
        """Handle HYPERPARAMETER_UPDATED - invalidate evaluation caches."""
        payload = event.payload if hasattr(event, 'payload') else {}
        param_name = payload.get("param_name", "")
        optimizer = payload.get("optimizer", "")

        logger.debug(f"[CacheInvalidator] Hyperparameter updated: {param_name} by {optimizer}")

        # Invalidate eval cache since evaluations may use old hyperparams
        self._invalidate_selective(
            caches=["eval_cache"],
            trigger_reason=f"hyperparameter_{param_name}",
        )

    def _on_regression_detected(self, event) -> None:
        """Handle REGRESSION_DETECTED - clear caches for fresh evaluation."""
        payload = event.payload if hasattr(event, 'payload') else {}
        config = payload.get("config", "")
        severity = payload.get("severity", "unknown")

        logger.info(f"[CacheInvalidator] Regression detected: {config} (severity={severity})")

        # Clear all caches on regression to ensure fresh evaluation
        self.invalidate_all(
            trigger_reason=f"regression_{severity}",
            model_id=config,
        )

    def _on_optimization_completed(self, event) -> None:
        """Handle NAS_COMPLETED or CMAES_COMPLETED - clear relevant caches."""
        payload = event.payload if hasattr(event, 'payload') else {}
        config = payload.get("config", "")

        logger.debug(f"[CacheInvalidator] Optimization completed: {config}")

        # Clear model and eval caches - architecture or hyperparams changed
        self._invalidate_selective(
            caches=["model_cache", "eval_cache", "export_cache"],
            trigger_reason="optimization_completed",
            model_id=config,
        )

    def _invalidate_selective(
        self,
        caches: list[str],
        trigger_reason: str = "selective",
        model_id: str = "",
    ) -> FullInvalidationResult:
        """Invalidate only specific caches.

        Args:
            caches: List of cache names to invalidate
            trigger_reason: Why invalidation was triggered
            model_id: Model ID if applicable

        Returns:
            FullInvalidationResult with details
        """
        result = FullInvalidationResult(
            trigger_reason=trigger_reason,
            model_id=model_id,
        )

        for cache_name in caches:
            if cache_name not in self._cache_invalidators:
                continue

            invalidator = self._cache_invalidators[cache_name]
            cache_result = self._invalidate_cache(cache_name, invalidator)
            result.results.append(cache_result)

            if cache_result.success:
                result.caches_cleared += 1
                result.total_items_cleared += cache_result.items_cleared
            else:
                result.total_success = False

        if result.caches_cleared > 0:
            logger.debug(
                f"[CacheInvalidator] Selective invalidation: {result.caches_cleared} caches, "
                f"{result.total_items_cleared} items (reason={trigger_reason})"
            )

        return result

    def unsubscribe(self) -> None:
        """Unsubscribe from model promotion events."""
        if not self._subscribed:
            return

        try:
            from app.distributed.data_events import (
                DataEventType,
                get_event_bus,
            )

            bus = get_event_bus()
            bus.unsubscribe(DataEventType.MODEL_PROMOTED, self._on_model_promoted)
            self._subscribed = False
        except Exception:
            pass

    def _on_model_promoted(self, event) -> None:
        """Handle MODEL_PROMOTED event.

        Triggers cache invalidation for all registered caches.
        """
        payload = event.payload if hasattr(event, 'payload') else {}

        model_id = payload.get("model_id", "")
        config = payload.get("config", "")
        elo_gain = payload.get("elo_gain", 0)

        logger.info(
            f"[CacheInvalidator] Model promoted: {model_id} "
            f"(config={config}, elo_gain={elo_gain})"
        )

        # Check cooldown
        now = time.time()
        if now - self._last_invalidation_time < self.invalidation_cooldown_seconds:
            logger.debug("[CacheInvalidator] Invalidation cooldown active, skipping")
            return

        # Invalidate all caches
        result = self.invalidate_all(
            trigger_reason="model_promoted",
            model_id=model_id,
        )

        logger.info(
            f"[CacheInvalidator] Invalidated {result.caches_cleared} caches, "
            f"{result.total_items_cleared} items cleared"
        )

    def invalidate_all(
        self,
        trigger_reason: str = "manual",
        model_id: str = "",
    ) -> FullInvalidationResult:
        """Invalidate all registered caches.

        Args:
            trigger_reason: Why invalidation was triggered
            model_id: Model ID that triggered invalidation (if applicable)

        Returns:
            FullInvalidationResult with details of each cache
        """
        result = FullInvalidationResult(
            trigger_reason=trigger_reason,
            model_id=model_id,
        )

        for cache_name, invalidator in self._cache_invalidators.items():
            cache_result = self._invalidate_cache(cache_name, invalidator)
            result.results.append(cache_result)

            if cache_result.success:
                result.caches_cleared += 1
                result.total_items_cleared += cache_result.items_cleared
            else:
                result.total_success = False

        # Clear GPU memory if enabled
        if self.clear_gpu_memory:
            self._clear_gpu_memory()

        self._last_invalidation_time = time.time()
        self._invalidation_count += 1

        # Emit invalidation event
        self._emit_invalidation_event(result)

        return result

    def _invalidate_cache(
        self,
        cache_name: str,
        invalidator: Callable[[], int],
    ) -> CacheInvalidationResult:
        """Invalidate a single cache.

        Args:
            cache_name: Name of the cache
            invalidator: Function to call

        Returns:
            CacheInvalidationResult
        """
        start_time = time.time()

        try:
            items_cleared = invalidator()
            duration_ms = (time.time() - start_time) * 1000

            return CacheInvalidationResult(
                cache_name=cache_name,
                success=True,
                items_cleared=items_cleared,
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.warning(f"[CacheInvalidator] Failed to invalidate {cache_name}: {e}")

            return CacheInvalidationResult(
                cache_name=cache_name,
                success=False,
                error=str(e),
                duration_ms=duration_ms,
            )

    def _clear_gpu_memory(self) -> None:
        """Clear GPU/MPS memory caches."""
        try:
            import gc

            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                with contextlib.suppress(Exception):
                    torch.mps.empty_cache()

            gc.collect()
            logger.debug("[CacheInvalidator] GPU/MPS memory cleared")

        except ImportError:
            pass

    def _emit_invalidation_event(self, result: FullInvalidationResult) -> None:
        """Emit cache invalidation completed event."""
        try:
            from app.distributed.data_events import (
                DataEvent,
                DataEventType,
                get_event_bus,
            )

            # Use a generic event type or quality score updated
            event = DataEvent(
                event_type=DataEventType.QUALITY_SCORE_UPDATED,
                payload={
                    "event_subtype": "cache_invalidation",
                    "trigger_reason": result.trigger_reason,
                    "model_id": result.model_id,
                    "caches_cleared": result.caches_cleared,
                    "total_items_cleared": result.total_items_cleared,
                    "success": result.total_success,
                    "timestamp": time.time(),
                },
                source="cache_invalidator",
            )

            bus = get_event_bus()
            bus.publish_sync(event)

        except Exception as e:
            logger.debug(f"Failed to emit invalidation event: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get invalidation statistics.

        Returns:
            Dict with invalidation stats
        """
        return {
            "subscribed": self._subscribed,
            "invalidation_count": self._invalidation_count,
            "last_invalidation_time": self._last_invalidation_time,
            "registered_caches": list(self._cache_invalidators.keys()),
            "cooldown_seconds": self.invalidation_cooldown_seconds,
        }


# Singleton cache invalidator
_cache_invalidator: ModelPromotionCacheInvalidator | None = None


def wire_promotion_to_cache_invalidation(
    invalidation_cooldown_seconds: float = 5.0,
    clear_gpu_memory: bool = True,
) -> ModelPromotionCacheInvalidator:
    """Wire MODEL_PROMOTED events to cache invalidation.

    This connects model promotion to automatic cache clearing across all
    cache implementations to ensure fresh model loading.

    Args:
        invalidation_cooldown_seconds: Minimum time between invalidations
        clear_gpu_memory: Also clear GPU/MPS caches

    Returns:
        ModelPromotionCacheInvalidator instance
    """
    global _cache_invalidator

    _cache_invalidator = ModelPromotionCacheInvalidator(
        invalidation_cooldown_seconds=invalidation_cooldown_seconds,
        clear_gpu_memory=clear_gpu_memory,
    )
    _cache_invalidator.subscribe_to_promotion_events()

    logger.info(
        f"[wire_promotion_to_cache_invalidation] MODEL_PROMOTED events wired to cache invalidation "
        f"(cooldown={invalidation_cooldown_seconds}s, clear_gpu={clear_gpu_memory})"
    )

    return _cache_invalidator


def wire_all_cache_invalidation_triggers(
    invalidation_cooldown_seconds: float = 5.0,
    clear_gpu_memory: bool = True,
) -> ModelPromotionCacheInvalidator:
    """Wire all events that should trigger cache invalidation.

    Expands beyond MODEL_PROMOTED to include:
    - TRAINING_COMPLETED: New model trained
    - HYPERPARAMETER_UPDATED: Hyperparams changed
    - REGRESSION_DETECTED: Model regression
    - NAS_COMPLETED: New architecture found
    - CMAES_COMPLETED: Hyperparams optimized

    Args:
        invalidation_cooldown_seconds: Minimum time between full invalidations
        clear_gpu_memory: Also clear GPU/MPS caches

    Returns:
        ModelPromotionCacheInvalidator instance with all triggers wired
    """
    global _cache_invalidator

    _cache_invalidator = ModelPromotionCacheInvalidator(
        invalidation_cooldown_seconds=invalidation_cooldown_seconds,
        clear_gpu_memory=clear_gpu_memory,
    )

    num_triggers = _cache_invalidator.subscribe_to_all_invalidation_triggers()

    logger.info(
        f"[wire_all_cache_invalidation_triggers] {num_triggers} event types wired to cache invalidation "
        f"(cooldown={invalidation_cooldown_seconds}s, clear_gpu={clear_gpu_memory})"
    )

    return _cache_invalidator


def get_cache_invalidator() -> ModelPromotionCacheInvalidator | None:
    """Get the global cache invalidator if configured."""
    return _cache_invalidator


def invalidate_all_caches(
    trigger_reason: str = "manual",
    model_id: str = "",
) -> FullInvalidationResult:
    """Convenience function to invalidate all caches.

    Creates a temporary invalidator if none exists.

    Args:
        trigger_reason: Why invalidation was triggered
        model_id: Model ID that triggered invalidation

    Returns:
        FullInvalidationResult with details
    """
    invalidator = _cache_invalidator or ModelPromotionCacheInvalidator()
    return invalidator.invalidate_all(
        trigger_reason=trigger_reason,
        model_id=model_id,
    )
