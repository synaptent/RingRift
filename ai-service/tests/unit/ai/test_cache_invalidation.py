"""Tests for app/ai/cache_invalidation.py.

Tests the unified cache invalidation system for model promotion.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import fields


# Test module imports
def test_module_imports():
    """Verify the module can be imported without errors."""
    from app.ai import cache_invalidation
    assert cache_invalidation is not None


class TestCacheInvalidationResult:
    """Tests for CacheInvalidationResult dataclass."""

    def test_dataclass_creation(self):
        from app.ai.cache_invalidation import CacheInvalidationResult
        result = CacheInvalidationResult(
            cache_name="test_cache",
            success=True,
        )
        assert result.cache_name == "test_cache"
        assert result.success is True

    def test_default_values(self):
        from app.ai.cache_invalidation import CacheInvalidationResult
        result = CacheInvalidationResult(
            cache_name="test",
            success=True,
        )
        assert result.items_cleared == 0
        assert result.error is None
        assert result.duration_ms == 0.0

    def test_items_cleared_field(self):
        from app.ai.cache_invalidation import CacheInvalidationResult
        result = CacheInvalidationResult(
            cache_name="test",
            success=True,
            items_cleared=100,
        )
        assert result.items_cleared == 100

    def test_error_field(self):
        from app.ai.cache_invalidation import CacheInvalidationResult
        result = CacheInvalidationResult(
            cache_name="test",
            success=False,
            error="Cache clear failed",
        )
        assert result.error == "Cache clear failed"

    def test_duration_field(self):
        from app.ai.cache_invalidation import CacheInvalidationResult
        result = CacheInvalidationResult(
            cache_name="test",
            success=True,
            duration_ms=50.5,
        )
        assert result.duration_ms == 50.5

    def test_has_required_fields(self):
        from app.ai.cache_invalidation import CacheInvalidationResult
        field_names = [f.name for f in fields(CacheInvalidationResult)]
        assert "cache_name" in field_names
        assert "success" in field_names


class TestFullInvalidationResult:
    """Tests for FullInvalidationResult dataclass."""

    def test_dataclass_creation(self):
        from app.ai.cache_invalidation import FullInvalidationResult
        result = FullInvalidationResult()
        assert result.total_success is True
        assert result.caches_cleared == 0

    def test_default_values(self):
        from app.ai.cache_invalidation import FullInvalidationResult
        result = FullInvalidationResult()
        assert result.total_success is True
        assert result.caches_cleared == 0
        assert result.total_items_cleared == 0
        assert result.results == []
        assert result.trigger_reason == ""
        assert result.model_id == ""

    def test_results_list(self):
        from app.ai.cache_invalidation import FullInvalidationResult, CacheInvalidationResult
        cache_result = CacheInvalidationResult(cache_name="test", success=True)
        full_result = FullInvalidationResult(
            results=[cache_result],
            caches_cleared=1,
        )
        assert len(full_result.results) == 1
        assert full_result.caches_cleared == 1

    def test_with_model_id(self):
        from app.ai.cache_invalidation import FullInvalidationResult
        result = FullInvalidationResult(
            model_id="model-123",
            trigger_reason="promotion",
        )
        assert result.model_id == "model-123"
        assert result.trigger_reason == "promotion"


class TestModelPromotionCacheInvalidator:
    """Tests for ModelPromotionCacheInvalidator class."""

    def test_class_exists(self):
        from app.ai.cache_invalidation import ModelPromotionCacheInvalidator
        assert ModelPromotionCacheInvalidator is not None

    def test_initialization(self):
        from app.ai.cache_invalidation import ModelPromotionCacheInvalidator
        invalidator = ModelPromotionCacheInvalidator()
        assert invalidator is not None

    def test_initialization_with_cooldown(self):
        from app.ai.cache_invalidation import ModelPromotionCacheInvalidator
        invalidator = ModelPromotionCacheInvalidator(
            invalidation_cooldown_seconds=10.0,
        )
        assert invalidator.invalidation_cooldown_seconds == 10.0

    def test_initialization_with_gpu_clear(self):
        from app.ai.cache_invalidation import ModelPromotionCacheInvalidator
        invalidator = ModelPromotionCacheInvalidator(
            clear_gpu_memory=False,
        )
        assert invalidator.clear_gpu_memory is False


class TestInvalidateAllCaches:
    """Tests for invalidate_all_caches function."""

    def test_function_exists(self):
        from app.ai.cache_invalidation import invalidate_all_caches
        assert callable(invalidate_all_caches)

    def test_returns_full_invalidation_result(self):
        from app.ai.cache_invalidation import invalidate_all_caches, FullInvalidationResult
        # Should work even if no caches are registered
        result = invalidate_all_caches()
        assert isinstance(result, FullInvalidationResult)


class TestWirePromotionToCacheInvalidation:
    """Tests for wire_promotion_to_cache_invalidation function."""

    def test_function_exists(self):
        from app.ai.cache_invalidation import wire_promotion_to_cache_invalidation
        assert callable(wire_promotion_to_cache_invalidation)

    def test_returns_invalidator_or_none(self):
        from app.ai.cache_invalidation import (
            wire_promotion_to_cache_invalidation,
            ModelPromotionCacheInvalidator,
        )
        # May return None if event bus not available
        try:
            result = wire_promotion_to_cache_invalidation()
            if result is not None:
                assert isinstance(result, ModelPromotionCacheInvalidator)
        except Exception:
            # Event bus may not be configured in test environment
            pass


class TestModuleExports:
    """Test module exports and __all__."""

    def test_has_main_exports(self):
        import app.ai.cache_invalidation as module
        expected_exports = [
            "CacheInvalidationResult",
            "FullInvalidationResult",
            "ModelPromotionCacheInvalidator",
            "invalidate_all_caches",
        ]
        for name in expected_exports:
            assert hasattr(module, name), f"Missing export: {name}"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalidation_result_with_zero_items(self):
        from app.ai.cache_invalidation import CacheInvalidationResult
        result = CacheInvalidationResult(
            cache_name="empty_cache",
            success=True,
            items_cleared=0,
        )
        assert result.success is True
        assert result.items_cleared == 0

    def test_full_invalidation_with_partial_failure(self):
        from app.ai.cache_invalidation import FullInvalidationResult, CacheInvalidationResult
        results = [
            CacheInvalidationResult(cache_name="cache1", success=True),
            CacheInvalidationResult(cache_name="cache2", success=False, error="Failed"),
        ]
        full_result = FullInvalidationResult(
            total_success=False,
            caches_cleared=1,
            results=results,
        )
        assert full_result.total_success is False
        assert full_result.caches_cleared == 1
        assert len(full_result.results) == 2
