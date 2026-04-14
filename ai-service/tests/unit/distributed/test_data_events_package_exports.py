"""Focused tests for app.distributed.data_events package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_key_data_event_exports() -> None:
    module = importlib.import_module("app.distributed.data_events")

    expected = {
        "CROSS_PROCESS_EVENT_TYPES",
        "DataEvent",
        "DataEventType",
        "EventBus",
        "EventCallback",
        "emit_checkpoint_saved",
        "emit_cluster_capacity_changed",
        "emit_data_event",
        "emit_evaluation_completed",
        "emit_model_promoted",
        "emit_new_games",
        "emit_node_overloaded",
        "emit_p2p_cluster_healthy",
        "emit_promotion_rejected",
        "emit_quality_score_updated",
        "emit_selfplay_complete",
        "emit_sync_triggered",
        "emit_training_completed",
        "emit_training_failed",
        "emit_training_started",
        "emit_weight_updated",
        "get_event_bus",
        "reset_event_bus",
    }

    assert expected.issubset(set(module.__all__))
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
