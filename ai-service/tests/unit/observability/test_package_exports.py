"""Focused tests for app.observability package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_observability_surface() -> None:
    module = importlib.import_module("app.observability")

    expected = [
        "configure_tracing",
        "get_tracer",
        "trace_async",
        "trace_sync",
        "TraceConfig",
        "TracingState",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
