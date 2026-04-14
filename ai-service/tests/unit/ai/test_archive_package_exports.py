"""Focused tests for app.ai.archive package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_archive_surface() -> None:
    module = importlib.import_module("app.ai.archive")

    expected = [
        "GMOMCTSHybrid",
        "GMOMCTSConfig",
        "CAGE_AI",
        "CAGEConfig",
        "EBMOOnlineAI",
        "EBMOOnlineConfig",
        "EBMOOnlineLearner",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
