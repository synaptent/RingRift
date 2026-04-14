"""Focused tests for app.storage package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_storage_surface() -> None:
    module = importlib.import_module("app.storage")

    expected = [
        "GCSStorage",
        "LocalStorage",
        "S3Storage",
        "StorageBackend",
        "get_storage_backend",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
