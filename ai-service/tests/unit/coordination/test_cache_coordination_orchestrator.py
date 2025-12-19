"""Tests for CacheCoordinationOrchestrator (unified cache management).

Tests cover:
- CacheType and CacheStatus enums
- CacheEntry, NodeCacheState, CacheStats dataclasses
- CacheCoordinationOrchestrator methods
- Module functions
"""

import time
import pytest
from unittest.mock import MagicMock, patch


# =============================================================================
# Test CacheType Enum
# =============================================================================

class TestCacheType:
    """Tests for CacheType enum."""

    def test_nnue_weights_value(self):
        """Test NNUE_WEIGHTS type value."""
        from app.coordination.cache_coordination_orchestrator import CacheType
        assert CacheType.NNUE_WEIGHTS.value == "nnue_weights"

    def test_feature_cache_value(self):
        """Test FEATURE_CACHE type value."""
        from app.coordination.cache_coordination_orchestrator import CacheType
        assert CacheType.FEATURE_CACHE.value == "feature_cache"

    def test_inference_cache_value(self):
        """Test INFERENCE_CACHE type value."""
        from app.coordination.cache_coordination_orchestrator import CacheType
        assert CacheType.INFERENCE_CACHE.value == "inference_cache"

    def test_game_replay_value(self):
        """Test GAME_REPLAY type value."""
        from app.coordination.cache_coordination_orchestrator import CacheType
        assert CacheType.GAME_REPLAY.value == "game_replay"

    def test_npz_data_value(self):
        """Test NPZ_DATA type value."""
        from app.coordination.cache_coordination_orchestrator import CacheType
        assert CacheType.NPZ_DATA.value == "npz_data"

    def test_checkpoint_value(self):
        """Test CHECKPOINT type value."""
        from app.coordination.cache_coordination_orchestrator import CacheType
        assert CacheType.CHECKPOINT.value == "checkpoint"

    def test_embedding_value(self):
        """Test EMBEDDING type value."""
        from app.coordination.cache_coordination_orchestrator import CacheType
        assert CacheType.EMBEDDING.value == "embedding"


# =============================================================================
# Test CacheStatus Enum
# =============================================================================

class TestCacheStatus:
    """Tests for CacheStatus enum."""

    def test_valid_value(self):
        """Test VALID status value."""
        from app.coordination.cache_coordination_orchestrator import CacheStatus
        assert CacheStatus.VALID.value == "valid"

    def test_stale_value(self):
        """Test STALE status value."""
        from app.coordination.cache_coordination_orchestrator import CacheStatus
        assert CacheStatus.STALE.value == "stale"

    def test_invalidated_value(self):
        """Test INVALIDATED status value."""
        from app.coordination.cache_coordination_orchestrator import CacheStatus
        assert CacheStatus.INVALIDATED.value == "invalidated"

    def test_warming_value(self):
        """Test WARMING status value."""
        from app.coordination.cache_coordination_orchestrator import CacheStatus
        assert CacheStatus.WARMING.value == "warming"


# =============================================================================
# Test CacheEntry Dataclass
# =============================================================================

class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_create_entry(self):
        """Test creating a cache entry."""
        from app.coordination.cache_coordination_orchestrator import (
            CacheEntry,
            CacheType,
            CacheStatus,
        )

        entry = CacheEntry(
            cache_id="gh200-a:nnue_weights:model_v42",
            node_id="gh200-a",
            cache_type=CacheType.NNUE_WEIGHTS,
            model_id="model_v42",
        )

        assert entry.cache_id == "gh200-a:nnue_weights:model_v42"
        assert entry.node_id == "gh200-a"
        assert entry.cache_type == CacheType.NNUE_WEIGHTS
        assert entry.model_id == "model_v42"

    def test_default_values(self):
        """Test default values."""
        from app.coordination.cache_coordination_orchestrator import (
            CacheEntry,
            CacheType,
            CacheStatus,
        )

        entry = CacheEntry(
            cache_id="test",
            node_id="node1",
            cache_type=CacheType.FEATURE_CACHE,
            model_id="model_v1",
        )

        assert entry.size_bytes == 0
        assert entry.status == CacheStatus.VALID
        assert entry.hits == 0
        assert entry.misses == 0
        assert entry.ttl_seconds == 3600.0

    def test_size_mb_property(self):
        """Test size_mb property."""
        from app.coordination.cache_coordination_orchestrator import (
            CacheEntry,
            CacheType,
        )

        entry = CacheEntry(
            cache_id="test",
            node_id="node1",
            cache_type=CacheType.NNUE_WEIGHTS,
            model_id="model",
            size_bytes=150 * 1024 * 1024,  # 150 MB
        )

        assert entry.size_mb == 150.0

    def test_hit_rate_property(self):
        """Test hit_rate property."""
        from app.coordination.cache_coordination_orchestrator import (
            CacheEntry,
            CacheType,
        )

        entry = CacheEntry(
            cache_id="test",
            node_id="node1",
            cache_type=CacheType.INFERENCE_CACHE,
            model_id="model",
            hits=80,
            misses=20,
        )

        assert entry.hit_rate == 0.8

    def test_hit_rate_zero_requests(self):
        """Test hit_rate with zero requests."""
        from app.coordination.cache_coordination_orchestrator import (
            CacheEntry,
            CacheType,
        )

        entry = CacheEntry(
            cache_id="test",
            node_id="node1",
            cache_type=CacheType.FEATURE_CACHE,
            model_id="model",
        )

        assert entry.hit_rate == 0.0

    def test_is_expired_property(self):
        """Test is_expired property."""
        from app.coordination.cache_coordination_orchestrator import (
            CacheEntry,
            CacheType,
        )

        entry = CacheEntry(
            cache_id="test",
            node_id="node1",
            cache_type=CacheType.NPZ_DATA,
            model_id="model",
            ttl_seconds=1.0,  # 1 second TTL
            created_at=time.time() - 2.0,  # Created 2 seconds ago
        )

        assert entry.is_expired is True

    def test_is_stale_property(self):
        """Test is_stale property."""
        from app.coordination.cache_coordination_orchestrator import (
            CacheEntry,
            CacheType,
        )

        entry = CacheEntry(
            cache_id="test",
            node_id="node1",
            cache_type=CacheType.GAME_REPLAY,
            model_id="model",
            last_access=time.time() - 2000.0,  # Not accessed for 33 minutes
        )

        assert entry.is_stale is True


# =============================================================================
# Test NodeCacheState Dataclass
# =============================================================================

class TestNodeCacheState:
    """Tests for NodeCacheState dataclass."""

    def test_create_state(self):
        """Test creating node cache state."""
        from app.coordination.cache_coordination_orchestrator import NodeCacheState

        state = NodeCacheState(node_id="gh200-a")

        assert state.node_id == "gh200-a"
        assert state.total_cache_bytes == 0
        assert state.cache_entries == 0

    def test_utilization_property(self):
        """Test utilization property."""
        from app.coordination.cache_coordination_orchestrator import NodeCacheState

        state = NodeCacheState(
            node_id="node1",
            total_cache_bytes=500_000_000,  # 500 MB
            cache_limit_bytes=1_000_000_000,  # 1 GB
        )

        assert state.utilization == pytest.approx(50.0, abs=0.1)

    def test_utilization_zero_limit(self):
        """Test utilization with zero limit."""
        from app.coordination.cache_coordination_orchestrator import NodeCacheState

        state = NodeCacheState(
            node_id="node1",
            cache_limit_bytes=0,
        )

        assert state.utilization == 0.0

    def test_hit_rate_property(self):
        """Test hit_rate property."""
        from app.coordination.cache_coordination_orchestrator import NodeCacheState

        state = NodeCacheState(
            node_id="node1",
            total_hits=90,
            total_misses=10,
        )

        assert state.hit_rate == 0.9


# =============================================================================
# Test CacheStats Dataclass
# =============================================================================

class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_create_stats(self):
        """Test creating cache stats."""
        from app.coordination.cache_coordination_orchestrator import CacheStats

        stats = CacheStats(
            total_entries=100,
            total_size_bytes=1024 * 1024 * 500,  # 500 MB
            valid_entries=80,
            stale_entries=15,
            invalidated_entries=5,
        )

        assert stats.total_entries == 100
        assert stats.valid_entries == 80

    def test_total_caches_alias(self):
        """Test total_caches property alias."""
        from app.coordination.cache_coordination_orchestrator import CacheStats

        stats = CacheStats(total_entries=42)
        assert stats.total_caches == 42

    def test_default_values(self):
        """Test default values."""
        from app.coordination.cache_coordination_orchestrator import CacheStats

        stats = CacheStats()

        assert stats.total_entries == 0
        assert stats.total_hits == 0
        assert stats.overall_hit_rate == 0.0
        assert stats.by_type == {}


# =============================================================================
# Test CacheCoordinationOrchestrator
# =============================================================================

class TestCacheCoordinationOrchestrator:
    """Tests for CacheCoordinationOrchestrator class."""

    @pytest.fixture
    def orchestrator(self):
        """Create a fresh orchestrator instance."""
        from app.coordination.cache_coordination_orchestrator import (
            CacheCoordinationOrchestrator,
        )
        return CacheCoordinationOrchestrator()

    def test_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.default_ttl_seconds == 3600.0
        assert orchestrator.max_entries_per_node == 1000
        assert orchestrator.stale_threshold_seconds == 1800.0
        assert len(orchestrator._entries) == 0
        assert orchestrator._subscribed is False

    def test_initialization_custom(self):
        """Test initialization with custom values."""
        from app.coordination.cache_coordination_orchestrator import (
            CacheCoordinationOrchestrator,
        )

        orch = CacheCoordinationOrchestrator(
            default_ttl_seconds=7200.0,
            max_entries_per_node=500,
            stale_threshold_seconds=900.0,
        )

        assert orch.default_ttl_seconds == 7200.0
        assert orch.max_entries_per_node == 500
        assert orch.stale_threshold_seconds == 900.0

    def test_register_cache(self, orchestrator):
        """Test registering a cache entry."""
        entry = orchestrator.register_cache(
            node_id="gh200-a",
            cache_type="nnue_weights",
            model_id="model_v42",
            size_bytes=150 * 1024 * 1024,
        )

        assert entry.node_id == "gh200-a"
        assert entry.model_id == "model_v42"
        assert entry.size_bytes == 150 * 1024 * 1024
        assert entry.cache_id in orchestrator._entries

    def test_register_cache_unknown_type(self, orchestrator):
        """Test registering cache with unknown type defaults to NNUE_WEIGHTS."""
        from app.coordination.cache_coordination_orchestrator import CacheType

        entry = orchestrator.register_cache(
            node_id="node1",
            cache_type="unknown_type",
            model_id="model",
        )

        assert entry.cache_type == CacheType.NNUE_WEIGHTS

    def test_register_cache_custom_ttl(self, orchestrator):
        """Test registering cache with custom TTL."""
        entry = orchestrator.register_cache(
            node_id="node1",
            cache_type="nnue_weights",
            model_id="model",
            ttl_seconds=7200.0,
        )

        assert entry.ttl_seconds == 7200.0

    def test_record_hit(self, orchestrator):
        """Test recording cache hit."""
        orchestrator.register_cache(
            node_id="node1",
            cache_type="nnue_weights",
            model_id="model",
        )

        result = orchestrator.record_hit("node1", "nnue_weights", "model")
        assert result is True

        entry = list(orchestrator._entries.values())[0]
        assert entry.hits == 1

    def test_record_hit_not_found(self, orchestrator):
        """Test recording hit for non-existent cache."""
        result = orchestrator.record_hit("node1", "nnue_weights", "nonexistent")
        assert result is False

    def test_record_miss(self, orchestrator):
        """Test recording cache miss."""
        orchestrator.register_cache(
            node_id="node1",
            cache_type="nnue_weights",
            model_id="model",
        )

        result = orchestrator.record_miss("node1", "nnue_weights", "model")
        assert result is True

        entry = list(orchestrator._entries.values())[0]
        assert entry.misses == 1

    def test_record_miss_not_found(self, orchestrator):
        """Test recording miss for non-existent cache."""
        result = orchestrator.record_miss("node1", "nnue_weights", "nonexistent")
        assert result is False

    def test_invalidate(self, orchestrator):
        """Test invalidating a cache entry."""
        from app.coordination.cache_coordination_orchestrator import CacheStatus

        entry = orchestrator.register_cache(
            node_id="node1",
            cache_type="nnue_weights",
            model_id="model",
        )

        result = orchestrator.invalidate(entry.cache_id)
        assert result is True
        assert orchestrator._entries[entry.cache_id].status == CacheStatus.INVALIDATED

    def test_invalidate_not_found(self, orchestrator):
        """Test invalidating non-existent cache."""
        result = orchestrator.invalidate("nonexistent")
        assert result is False

    def test_invalidate_model(self, orchestrator):
        """Test invalidating all caches for a model."""
        # Register caches for model_v1
        orchestrator.register_cache("node1", "nnue_weights", "model_v1")
        orchestrator.register_cache("node2", "nnue_weights", "model_v1")
        orchestrator.register_cache("node1", "feature_cache", "model_v1")

        # Register cache for different model
        orchestrator.register_cache("node1", "nnue_weights", "model_v2")

        count = orchestrator.invalidate_model("model_v1")

        assert count == 3
        assert orchestrator._total_invalidations == 3

    def test_invalidate_model_not_found(self, orchestrator):
        """Test invalidating non-existent model."""
        count = orchestrator.invalidate_model("nonexistent")
        assert count == 0

    def test_invalidate_node(self, orchestrator):
        """Test invalidating all caches on a node."""
        orchestrator.register_cache("node1", "nnue_weights", "model_v1")
        orchestrator.register_cache("node1", "feature_cache", "model_v2")
        orchestrator.register_cache("node2", "nnue_weights", "model_v1")

        count = orchestrator.invalidate_node("node1")

        assert count == 2

    def test_remove_cache(self, orchestrator):
        """Test removing a cache entry."""
        entry = orchestrator.register_cache(
            node_id="node1",
            cache_type="nnue_weights",
            model_id="model",
        )

        result = orchestrator.remove_cache(entry.cache_id)
        assert result is True
        assert entry.cache_id not in orchestrator._entries

    def test_remove_cache_not_found(self, orchestrator):
        """Test removing non-existent cache."""
        result = orchestrator.remove_cache("nonexistent")
        assert result is False

    def test_cleanup_stale(self, orchestrator):
        """Test cleaning up stale entries."""
        from app.coordination.cache_coordination_orchestrator import CacheStatus

        # Register and immediately invalidate
        entry = orchestrator.register_cache(
            node_id="node1",
            cache_type="nnue_weights",
            model_id="model",
        )
        orchestrator._entries[entry.cache_id].status = CacheStatus.INVALIDATED

        count = orchestrator.cleanup_stale()
        assert count == 1
        assert entry.cache_id not in orchestrator._entries

    def test_get_cache(self, orchestrator):
        """Test getting a cache entry."""
        entry = orchestrator.register_cache(
            node_id="node1",
            cache_type="nnue_weights",
            model_id="model",
        )

        result = orchestrator.get_cache(entry.cache_id)
        assert result is not None
        assert result.model_id == "model"

    def test_get_cache_not_found(self, orchestrator):
        """Test getting non-existent cache."""
        result = orchestrator.get_cache("nonexistent")
        assert result is None

    def test_get_caches_by_node(self, orchestrator):
        """Test getting caches by node."""
        orchestrator.register_cache("node1", "nnue_weights", "model_v1")
        orchestrator.register_cache("node1", "feature_cache", "model_v2")
        orchestrator.register_cache("node2", "nnue_weights", "model_v1")

        caches = orchestrator.get_caches_by_node("node1")
        assert len(caches) == 2

    def test_get_caches_by_model(self, orchestrator):
        """Test getting caches by model."""
        orchestrator.register_cache("node1", "nnue_weights", "model_v1")
        orchestrator.register_cache("node2", "nnue_weights", "model_v1")
        orchestrator.register_cache("node1", "nnue_weights", "model_v2")

        caches = orchestrator.get_caches_by_model("model_v1")
        assert len(caches) == 2

    def test_get_caches_by_type(self, orchestrator):
        """Test getting caches by type."""
        from app.coordination.cache_coordination_orchestrator import CacheType

        orchestrator.register_cache("node1", "nnue_weights", "model_v1")
        orchestrator.register_cache("node2", "nnue_weights", "model_v2")
        orchestrator.register_cache("node1", "feature_cache", "model_v1")

        caches = orchestrator.get_caches_by_type(CacheType.NNUE_WEIGHTS)
        assert len(caches) == 2

    def test_get_node_state(self, orchestrator):
        """Test getting node state."""
        orchestrator.register_cache(
            node_id="node1",
            cache_type="nnue_weights",
            model_id="model",
            size_bytes=100 * 1024 * 1024,
        )

        state = orchestrator.get_node_state("node1")
        assert state is not None
        assert state.node_id == "node1"
        assert state.total_cache_bytes == 100 * 1024 * 1024

    def test_get_stats(self, orchestrator):
        """Test getting aggregate stats."""
        orchestrator.register_cache("node1", "nnue_weights", "model_v1", size_bytes=100)
        orchestrator.register_cache("node2", "feature_cache", "model_v1", size_bytes=200)

        stats = orchestrator.get_stats()

        assert stats.total_entries == 2
        assert stats.total_size_bytes == 300

    def test_get_status(self, orchestrator):
        """Test getting status."""
        orchestrator.register_cache("node1", "nnue_weights", "model_v1")
        orchestrator.register_cache("node2", "nnue_weights", "model_v2")

        status = orchestrator.get_status()

        assert "total_caches" in status
        assert "total_entries" in status
        assert "models_cached" in status
        assert "subscribed" in status
        assert status["total_entries"] == 2

    def test_on_invalidation_callback(self, orchestrator):
        """Test invalidation callback registration."""
        callback = MagicMock()
        orchestrator.on_invalidation(callback)

        orchestrator.register_cache("node1", "nnue_weights", "model_v1")
        orchestrator.invalidate_model("model_v1")

        callback.assert_called()


# =============================================================================
# Test Module Functions
# =============================================================================

class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_get_cache_orchestrator_singleton(self):
        """Test singleton pattern."""
        from app.coordination.cache_coordination_orchestrator import (
            get_cache_orchestrator,
        )

        orch1 = get_cache_orchestrator()
        orch2 = get_cache_orchestrator()

        assert orch1 is orch2

    def test_register_cache_function(self):
        """Test register_cache convenience function."""
        from app.coordination.cache_coordination_orchestrator import (
            register_cache,
            get_cache_orchestrator,
        )

        entry = register_cache(
            node_id="convenience_node",
            cache_type="nnue_weights",
            model_id="convenience_model",
            size_bytes=1024,
        )

        orch = get_cache_orchestrator()
        assert orch.get_cache(entry.cache_id) is not None

    def test_invalidate_model_caches_function(self):
        """Test invalidate_model_caches convenience function."""
        from app.coordination.cache_coordination_orchestrator import (
            register_cache,
            invalidate_model_caches,
        )

        register_cache("node1", "nnue_weights", "func_test_model")
        count = invalidate_model_caches("func_test_model")

        assert count >= 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestCacheIntegration:
    """Integration tests for cache coordination orchestrator."""

    def test_full_cache_lifecycle(self):
        """Test complete cache lifecycle."""
        from app.coordination.cache_coordination_orchestrator import (
            CacheCoordinationOrchestrator,
            CacheStatus,
        )

        orch = CacheCoordinationOrchestrator()

        # Register cache
        entry = orch.register_cache(
            node_id="gh200-a",
            cache_type="nnue_weights",
            model_id="model_v42",
            size_bytes=150 * 1024 * 1024,
        )

        assert entry.status == CacheStatus.VALID

        # Record hits
        for _ in range(10):
            orch.record_hit("gh200-a", "nnue_weights", "model_v42")

        assert orch._entries[entry.cache_id].hits == 10

        # Invalidate model
        count = orch.invalidate_model("model_v42")
        assert count == 1
        assert orch._entries[entry.cache_id].status == CacheStatus.INVALIDATED

        # Cleanup
        cleaned = orch.cleanup_stale()
        assert cleaned == 1
        assert entry.cache_id not in orch._entries

    def test_multi_node_cache_management(self):
        """Test managing caches across multiple nodes."""
        from app.coordination.cache_coordination_orchestrator import (
            CacheCoordinationOrchestrator,
        )

        orch = CacheCoordinationOrchestrator()

        # Register caches on multiple nodes
        for i, node in enumerate(["gh200-a", "gh200-b", "h100-1"]):
            orch.register_cache(
                node_id=node,
                cache_type="nnue_weights",
                model_id="model_v42",
                size_bytes=(100 + i * 50) * 1024 * 1024,
            )

        # Verify stats
        stats = orch.get_stats()
        assert stats.total_entries == 3
        assert len(stats.by_node) == 3

        # Invalidate on single node
        count = orch.invalidate_node("gh200-a")
        assert count == 1

        # Other nodes unaffected
        remaining = orch.get_caches_by_model("model_v42")
        assert len([c for c in remaining if c.status.value == "valid"]) == 2

    def test_model_promotion_workflow(self):
        """Test workflow when promoting a new model."""
        from app.coordination.cache_coordination_orchestrator import (
            CacheCoordinationOrchestrator,
        )

        orch = CacheCoordinationOrchestrator()

        # Old model caches
        orch.register_cache("node1", "nnue_weights", "model_v41")
        orch.register_cache("node2", "nnue_weights", "model_v41")

        # Simulate promotion: invalidate old model
        count = orch.invalidate_model("model_v41")
        assert count == 2

        # Register new model caches
        orch.register_cache("node1", "nnue_weights", "model_v42")
        orch.register_cache("node2", "nnue_weights", "model_v42")

        # Verify status
        status = orch.get_status()
        assert "model_v41" in status["models_cached"]
        assert "model_v42" in status["models_cached"]
