"""CacheCoordinationOrchestrator - Unified cache management (December 2025).

This module provides centralized coordination of cache management across the
cluster. It tracks cache entries, coordinates invalidation, and provides
visibility into cache efficiency.

Key Responsibilities:
1. Track cache entries across all nodes (NNUE weights, feature caches, etc.)
2. Coordinate cache invalidation on model promotion
3. Track cache hit rates and memory usage
4. Provide cache warming recommendations
5. Coordinate cross-node cache sharing

Cache Types:
- nnue_weights: Neural network weights cache
- feature_cache: NNUE feature caching
- inference_cache: Model inference results
- game_replay: Game replay buffers
- npz_data: NPZ training data cache

Usage:
    from app.coordination.cache_coordination_orchestrator import (
        CacheCoordinationOrchestrator,
        wire_cache_events,
        get_cache_orchestrator,
    )

    # Wire cache events
    orchestrator = wire_cache_events()

    # Register a cache entry
    orchestrator.register_cache("gh200-a", "nnue_weights", "model_v42", size_mb=150)

    # Invalidate caches for old model
    orchestrator.invalidate_model("model_v41")

    # Get cache status
    status = orchestrator.get_status()
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CacheType(Enum):
    """Types of caches in the system."""

    NNUE_WEIGHTS = "nnue_weights"
    FEATURE_CACHE = "feature_cache"
    INFERENCE_CACHE = "inference_cache"
    GAME_REPLAY = "game_replay"
    NPZ_DATA = "npz_data"
    CHECKPOINT = "checkpoint"
    EMBEDDING = "embedding"


class CacheStatus(Enum):
    """Status of a cache entry."""

    VALID = "valid"
    STALE = "stale"
    INVALIDATED = "invalidated"
    WARMING = "warming"


@dataclass
class CacheEntry:
    """A single cache entry."""

    cache_id: str
    node_id: str
    cache_type: CacheType
    model_id: str
    size_bytes: int = 0
    status: CacheStatus = CacheStatus.VALID
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    hits: int = 0
    misses: int = 0
    ttl_seconds: float = 3600.0  # 1 hour default

    @property
    def size_mb(self) -> float:
        """Size in megabytes."""
        return self.size_bytes / (1024 * 1024)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return time.time() - self.created_at > self.ttl_seconds

    @property
    def is_stale(self) -> bool:
        """Check if cache entry is stale (no access for 30 minutes)."""
        return time.time() - self.last_access > 1800.0


@dataclass
class NodeCacheState:
    """Cache state for a single node."""

    node_id: str
    total_cache_bytes: int = 0
    cache_limit_bytes: int = 0
    cache_entries: int = 0
    total_hits: int = 0
    total_misses: int = 0
    last_update: float = field(default_factory=time.time)
    caches_by_type: dict[str, int] = field(default_factory=dict)

    @property
    def utilization(self) -> float:
        """Cache utilization percentage."""
        if self.cache_limit_bytes == 0:
            return 0.0
        return (self.total_cache_bytes / self.cache_limit_bytes) * 100.0

    @property
    def hit_rate(self) -> float:
        """Overall cache hit rate."""
        total = self.total_hits + self.total_misses
        return self.total_hits / total if total > 0 else 0.0


@dataclass
class CacheStats:
    """Aggregate cache statistics."""

    total_entries: int = 0
    total_size_bytes: int = 0
    valid_entries: int = 0
    stale_entries: int = 0
    invalidated_entries: int = 0
    total_hits: int = 0
    total_misses: int = 0
    overall_hit_rate: float = 0.0
    by_type: dict[str, int] = field(default_factory=dict)
    by_node: dict[str, int] = field(default_factory=dict)
    by_model: dict[str, int] = field(default_factory=dict)

    @property
    def total_caches(self) -> int:
        """Backwards-compatible alias for total entries."""
        return self.total_entries


class CacheCoordinationOrchestrator:
    """Orchestrates cache management across the cluster.

    Tracks cache entries, coordinates invalidation, and provides
    unified visibility into cache efficiency.
    """

    def __init__(
        self,
        default_ttl_seconds: float = 3600.0,
        max_entries_per_node: int = 1000,
        stale_threshold_seconds: float = 1800.0,
    ):
        """Initialize CacheCoordinationOrchestrator.

        Args:
            default_ttl_seconds: Default TTL for cache entries
            max_entries_per_node: Maximum entries to track per node
            stale_threshold_seconds: Time without access to mark stale
        """
        self.default_ttl_seconds = default_ttl_seconds
        self.max_entries_per_node = max_entries_per_node
        self.stale_threshold_seconds = stale_threshold_seconds

        # Cache tracking
        self._entries: dict[str, CacheEntry] = {}  # cache_id -> entry
        self._by_node: dict[str, set[str]] = {}  # node_id -> cache_ids
        self._by_model: dict[str, set[str]] = {}  # model_id -> cache_ids
        self._by_type: dict[CacheType, set[str]] = {}  # cache_type -> cache_ids

        # Node state
        self._node_states: dict[str, NodeCacheState] = {}

        # Statistics
        self._total_invalidations = 0
        self._cache_id_counter = 0

        # Callbacks
        self._invalidation_callbacks: list[Callable[[str, str], None]] = []  # model_id, node_id

        # Subscription state
        self._subscribed = False

    def subscribe_to_events(self) -> bool:
        """Subscribe to cache-related events."""
        if self._subscribed:
            return True

        try:
            from app.coordination.event_router import get_router
            from app.distributed.data_events import DataEventType

            router = get_router()

            # Listen for model promotion to invalidate old caches
            router.subscribe(DataEventType.MODEL_PROMOTED.value, self._on_model_promoted)
            router.subscribe(DataEventType.PROMOTION_ROLLED_BACK.value, self._on_promotion_rolled_back)

            self._subscribed = True
            logger.info("[CacheCoordinationOrchestrator] Subscribed to events")
            return True

        except ImportError:
            logger.warning("[CacheCoordinationOrchestrator] data_events not available")
            return False
        except Exception as e:
            logger.error(f"[CacheCoordinationOrchestrator] Failed to subscribe: {e}")
            return False

    def _generate_cache_id(self, node_id: str, cache_type: CacheType, model_id: str) -> str:
        """Generate a unique cache ID."""
        return f"{node_id}:{cache_type.value}:{model_id}"

    async def _on_model_promoted(self, event) -> None:
        """Handle MODEL_PROMOTED - invalidate old model caches."""
        payload = event.payload
        new_model = payload.get("model_id", "")

        # Invalidate caches for all models except the new one
        models_to_invalidate = set(self._by_model.keys()) - {new_model}
        for model_id in models_to_invalidate:
            count = self.invalidate_model(model_id)
            if count > 0:
                logger.info(
                    f"[CacheCoordinationOrchestrator] Invalidated {count} caches "
                    f"for old model {model_id}"
                )

    async def _on_promotion_rolled_back(self, event) -> None:
        """Handle PROMOTION_ROLLED_BACK - invalidate rolled back model caches."""
        payload = event.payload
        from_model = payload.get("from_model_id", "")

        if from_model:
            count = self.invalidate_model(from_model)
            logger.info(
                f"[CacheCoordinationOrchestrator] Invalidated {count} caches "
                f"after rollback of {from_model}"
            )

    def register_cache(
        self,
        node_id: str,
        cache_type: str,
        model_id: str,
        size_bytes: int = 0,
        ttl_seconds: float | None = None,
    ) -> CacheEntry:
        """Register a new cache entry.

        Args:
            node_id: Node hosting the cache
            cache_type: Type of cache
            model_id: Model associated with cache
            size_bytes: Size of cached data
            ttl_seconds: Time-to-live (None for default)

        Returns:
            The created CacheEntry
        """
        try:
            cache_type_enum = CacheType(cache_type)
        except ValueError:
            cache_type_enum = CacheType.NNUE_WEIGHTS

        cache_id = self._generate_cache_id(node_id, cache_type_enum, model_id)

        entry = CacheEntry(
            cache_id=cache_id,
            node_id=node_id,
            cache_type=cache_type_enum,
            model_id=model_id,
            size_bytes=size_bytes,
            ttl_seconds=ttl_seconds or self.default_ttl_seconds,
        )

        # Store in indices
        self._entries[cache_id] = entry

        if node_id not in self._by_node:
            self._by_node[node_id] = set()
        self._by_node[node_id].add(cache_id)

        if model_id not in self._by_model:
            self._by_model[model_id] = set()
        self._by_model[model_id].add(cache_id)

        if cache_type_enum not in self._by_type:
            self._by_type[cache_type_enum] = set()
        self._by_type[cache_type_enum].add(cache_id)

        # Update node state
        self._update_node_state(node_id)

        logger.debug(
            f"[CacheCoordinationOrchestrator] Registered cache: {cache_id}, "
            f"size={entry.size_mb:.1f}MB"
        )

        return entry

    def record_hit(self, node_id: str, cache_type: str, model_id: str) -> bool:
        """Record a cache hit.

        Returns:
            True if cache entry was found
        """
        try:
            cache_type_enum = CacheType(cache_type)
        except ValueError:
            return False

        cache_id = self._generate_cache_id(node_id, cache_type_enum, model_id)

        if cache_id in self._entries:
            self._entries[cache_id].hits += 1
            self._entries[cache_id].last_access = time.time()
            return True
        return False

    def record_miss(self, node_id: str, cache_type: str, model_id: str) -> bool:
        """Record a cache miss.

        Returns:
            True if cache entry was found
        """
        try:
            cache_type_enum = CacheType(cache_type)
        except ValueError:
            return False

        cache_id = self._generate_cache_id(node_id, cache_type_enum, model_id)

        if cache_id in self._entries:
            self._entries[cache_id].misses += 1
            return True
        return False

    def invalidate(self, cache_id: str) -> bool:
        """Invalidate a specific cache entry.

        Returns:
            True if cache was found and invalidated
        """
        if cache_id not in self._entries:
            return False

        entry = self._entries[cache_id]
        entry.status = CacheStatus.INVALIDATED

        self._total_invalidations += 1

        logger.debug(f"[CacheCoordinationOrchestrator] Invalidated cache: {cache_id}")
        return True

    def invalidate_model(self, model_id: str) -> int:
        """Invalidate all caches for a model.

        Returns:
            Number of caches invalidated
        """
        if model_id not in self._by_model:
            return 0

        cache_ids = list(self._by_model[model_id])
        count = 0
        affected_nodes: set[str] = set()

        for cache_id in cache_ids:
            if cache_id in self._entries:
                affected_nodes.add(self._entries[cache_id].node_id)
            if self.invalidate(cache_id):
                count += 1

        # Emit CACHE_INVALIDATED event (December 2025)
        if count > 0:
            self._emit_cache_invalidated(
                invalidation_type="model",
                target_id=model_id,
                count=count,
                affected_nodes=list(affected_nodes),
            )

        # Notify callbacks
        for callback in self._invalidation_callbacks:
            try:
                callback(model_id, "all")
            except Exception as e:
                logger.error(f"[CacheCoordinationOrchestrator] Callback error: {e}")

        return count

    def invalidate_node(self, node_id: str) -> int:
        """Invalidate all caches on a node.

        Returns:
            Number of caches invalidated
        """
        if node_id not in self._by_node:
            return 0

        cache_ids = list(self._by_node[node_id])
        count = 0
        affected_models: set[str] = set()

        for cache_id in cache_ids:
            if cache_id in self._entries:
                affected_models.add(self._entries[cache_id].model_id)
            if self.invalidate(cache_id):
                count += 1

        # Emit CACHE_INVALIDATED event (December 2025)
        if count > 0:
            self._emit_cache_invalidated(
                invalidation_type="node",
                target_id=node_id,
                count=count,
                affected_nodes=[node_id],
                affected_models=list(affected_models),
            )

        return count

    def remove_cache(self, cache_id: str) -> bool:
        """Remove a cache entry completely.

        Returns:
            True if cache was found and removed
        """
        if cache_id not in self._entries:
            return False

        entry = self._entries.pop(cache_id)

        # Remove from indices
        if entry.node_id in self._by_node:
            self._by_node[entry.node_id].discard(cache_id)
        if entry.model_id in self._by_model:
            self._by_model[entry.model_id].discard(cache_id)
        if entry.cache_type in self._by_type:
            self._by_type[entry.cache_type].discard(cache_id)

        self._update_node_state(entry.node_id)

        return True

    def cleanup_stale(self) -> int:
        """Remove stale and expired cache entries.

        Returns:
            Number of entries cleaned up
        """
        stale_ids = []
        for cache_id, entry in self._entries.items():
            if entry.is_expired or entry.status == CacheStatus.INVALIDATED:
                stale_ids.append(cache_id)

        for cache_id in stale_ids:
            self.remove_cache(cache_id)

        if stale_ids:
            logger.info(
                f"[CacheCoordinationOrchestrator] Cleaned up {len(stale_ids)} stale entries"
            )

        return len(stale_ids)

    def _update_node_state(self, node_id: str) -> None:
        """Update node cache state."""
        if node_id not in self._by_node:
            return

        cache_ids = self._by_node[node_id]
        entries = [self._entries[cid] for cid in cache_ids if cid in self._entries]

        total_size = sum(e.size_bytes for e in entries)
        total_hits = sum(e.hits for e in entries)
        total_misses = sum(e.misses for e in entries)

        by_type: dict[str, int] = {}
        for e in entries:
            by_type[e.cache_type.value] = by_type.get(e.cache_type.value, 0) + 1

        if node_id not in self._node_states:
            self._node_states[node_id] = NodeCacheState(node_id=node_id)

        state = self._node_states[node_id]
        state.total_cache_bytes = total_size
        state.cache_entries = len(entries)
        state.total_hits = total_hits
        state.total_misses = total_misses
        state.caches_by_type = by_type
        state.last_update = time.time()

    def _emit_cache_invalidated(
        self,
        invalidation_type: str,
        target_id: str,
        count: int,
        affected_nodes: list[str],
        affected_models: list[str] | None = None,
    ) -> None:
        """Emit CACHE_INVALIDATED event (December 2025).

        Uses centralized event_emitters for consistent event emission.

        Args:
            invalidation_type: "model" or "node"
            target_id: Model ID or node ID that was invalidated
            count: Number of caches invalidated
            affected_nodes: List of affected node IDs
            affected_models: List of affected model IDs (for node invalidation)
        """
        try:
            import asyncio

            from app.coordination.event_emitters import emit_cache_invalidated

            try:
                asyncio.get_running_loop()
                asyncio.create_task(emit_cache_invalidated(
                    invalidation_type=invalidation_type,
                    target_id=target_id,
                    count=count,
                    affected_nodes=affected_nodes,
                    affected_models=affected_models,
                ))
            except RuntimeError:
                asyncio.run(emit_cache_invalidated(
                    invalidation_type=invalidation_type,
                    target_id=target_id,
                    count=count,
                    affected_nodes=affected_nodes,
                    affected_models=affected_models,
                ))

            logger.debug(
                f"[CacheCoordinationOrchestrator] Emitted CACHE_INVALIDATED: "
                f"{invalidation_type}={target_id}, count={count}"
            )

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[CacheCoordinationOrchestrator] Failed to emit event: {e}")

    def on_invalidation(self, callback: Callable[[str, str], None]) -> None:
        """Register callback for cache invalidation.

        Args:
            callback: Function(model_id, node_id)
        """
        self._invalidation_callbacks.append(callback)

    def get_cache(self, cache_id: str) -> CacheEntry | None:
        """Get a specific cache entry."""
        return self._entries.get(cache_id)

    def get_caches_by_node(self, node_id: str) -> list[CacheEntry]:
        """Get all cache entries for a node."""
        if node_id not in self._by_node:
            return []
        return [
            self._entries[cid]
            for cid in self._by_node[node_id]
            if cid in self._entries
        ]

    def get_caches_by_model(self, model_id: str) -> list[CacheEntry]:
        """Get all cache entries for a model."""
        if model_id not in self._by_model:
            return []
        return [
            self._entries[cid]
            for cid in self._by_model[model_id]
            if cid in self._entries
        ]

    def get_caches_by_type(self, cache_type: CacheType) -> list[CacheEntry]:
        """Get all cache entries of a type."""
        if cache_type not in self._by_type:
            return []
        return [
            self._entries[cid]
            for cid in self._by_type[cache_type]
            if cid in self._entries
        ]

    def get_node_state(self, node_id: str) -> NodeCacheState | None:
        """Get cache state for a node."""
        return self._node_states.get(node_id)

    def get_stats(self) -> CacheStats:
        """Get aggregate cache statistics."""
        entries = list(self._entries.values())

        valid = [e for e in entries if e.status == CacheStatus.VALID]
        stale = [e for e in entries if e.is_stale]
        invalidated = [e for e in entries if e.status == CacheStatus.INVALIDATED]

        total_hits = sum(e.hits for e in entries)
        total_misses = sum(e.misses for e in entries)
        total_requests = total_hits + total_misses

        by_type = {
            ct.value: len([e for e in entries if e.cache_type == ct])
            for ct in CacheType
            if ct in self._by_type
        }

        by_node = {
            node_id: len(cache_ids)
            for node_id, cache_ids in self._by_node.items()
        }

        by_model = {
            model_id: len(cache_ids)
            for model_id, cache_ids in self._by_model.items()
        }

        return CacheStats(
            total_entries=len(entries),
            total_size_bytes=sum(e.size_bytes for e in entries),
            valid_entries=len(valid),
            stale_entries=len(stale),
            invalidated_entries=len(invalidated),
            total_hits=total_hits,
            total_misses=total_misses,
            overall_hit_rate=total_hits / total_requests if total_requests > 0 else 0.0,
            by_type=by_type,
            by_node=by_node,
            by_model=by_model,
        )

    def get_status(self) -> dict[str, Any]:
        """Get orchestrator status for monitoring."""
        stats = self.get_stats()
        total_size_mb = stats.total_size_bytes / (1024 * 1024)

        return {
            "total_caches": stats.total_caches,
            "total_entries": stats.total_entries,
            "total_size_mb": round(total_size_mb, 1),
            "valid_entries": stats.valid_entries,
            "stale_entries": stats.stale_entries,
            "invalidated_entries": stats.invalidated_entries,
            "overall_hit_rate": round(stats.overall_hit_rate * 100, 1),
            "total_invalidations": self._total_invalidations,
            "by_type": stats.by_type,
            "by_node": stats.by_node,
            "models_cached": list(self._by_model.keys()),
            "subscribed": self._subscribed,
        }


# =============================================================================
# Singleton and convenience functions
# =============================================================================

_cache_orchestrator: CacheCoordinationOrchestrator | None = None


def get_cache_orchestrator() -> CacheCoordinationOrchestrator:
    """Get the global CacheCoordinationOrchestrator singleton."""
    global _cache_orchestrator
    if _cache_orchestrator is None:
        _cache_orchestrator = CacheCoordinationOrchestrator()
    return _cache_orchestrator


def wire_cache_events() -> CacheCoordinationOrchestrator:
    """Wire cache events to the orchestrator."""
    orchestrator = get_cache_orchestrator()
    orchestrator.subscribe_to_events()
    return orchestrator


def register_cache(
    node_id: str, cache_type: str, model_id: str, size_bytes: int = 0
) -> CacheEntry:
    """Convenience function to register a cache entry."""
    return get_cache_orchestrator().register_cache(
        node_id, cache_type, model_id, size_bytes
    )


def invalidate_model_caches(model_id: str) -> int:
    """Convenience function to invalidate model caches."""
    return get_cache_orchestrator().invalidate_model(model_id)


__all__ = [
    "CacheCoordinationOrchestrator",
    "CacheEntry",
    "CacheStats",
    "CacheStatus",
    "CacheType",
    "NodeCacheState",
    "get_cache_orchestrator",
    "invalidate_model_caches",
    "register_cache",
    "wire_cache_events",
]
