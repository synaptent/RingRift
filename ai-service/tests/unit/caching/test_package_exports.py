"""Focused tests for app.caching package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_caching_surface() -> None:
    module = importlib.import_module("app.caching")

    expected = [
        "Cache",
        "CacheConfig",
        "CacheEntry",
        "CacheStats",
        "LRUCache",
        "MemoryCache",
        "TTLCache",
        "FileCache",
        "ValidatedFileCache",
        "async_cached",
        "cached",
        "invalidate_cache",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
