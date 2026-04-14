"""Focused tests for app.events package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_events_surface() -> None:
    module = importlib.import_module("app.events")

    expected = [
        "RingRiftEventType",
        "EventCategory",
        "get_events_by_category",
        "CROSS_PROCESS_EVENT_TYPES",
        "is_cross_process_event",
        "DataEventType",
        "StageEvent",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
