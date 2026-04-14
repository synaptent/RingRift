"""Focused tests for app.metrics package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_key_public_exports() -> None:
    module = importlib.import_module("app.metrics")

    expected = {
        "AI_MOVE_REQUESTS",
        "PYTHON_INVARIANT_VIOLATIONS",
        "record_evaluation",
        "record_promotion_execution",
        "record_rollback_check",
        "record_auto_rollback",
        "MetricCatalog",
        "safe_metric",
        "start_metrics_server",
        "is_metrics_server_running",
        "create_training_logger",
    }

    assert expected.issubset(set(module.__all__))
    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)


def test_package_dir_covers_declared_public_surface() -> None:
    module = importlib.import_module("app.metrics")

    assert len(module.__all__) == len(set(module.__all__))
    assert set(module.__all__).issubset(set(dir(module)))
