"""Focused tests for app.errors package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_key_public_exports() -> None:
    module = importlib.import_module("app.errors")

    expected = {
        "ErrorCode",
        "RingRiftError",
        "DiskSpaceError",
        "SSHError",
        "SyncIntegrityError",
        "ModelVersioningError",
        "EmergencyHaltError",
        "FatalError",
        "RecoverableError",
    }

    assert expected.issubset(set(module.__all__))
    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)


def test_package_dir_covers_declared_public_surface() -> None:
    module = importlib.import_module("app.errors")

    assert len(module.__all__) == len(set(module.__all__))
    assert set(module.__all__).issubset(set(dir(module)))
