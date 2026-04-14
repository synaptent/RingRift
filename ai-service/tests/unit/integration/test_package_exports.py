"""Focused tests for app.integration package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_integration_surface() -> None:
    module = importlib.import_module("app.integration")

    expected = [
        "EvaluationCurriculumBridge",
        "ExtensionConfig",
        "FeedbackAction",
        "FeedbackSignal",
        "FeedbackSignalRouter",
        "LifecycleConfig",
        "ModelLifecycleManager",
        "OpponentWinRateTracker",
        "P2PIntegrationConfig",
        "P2PIntegrationManager",
        "PipelineFeedbackController",
        "UnifiedLoopExtensions",
        "connect_to_cluster",
        "create_evaluation_bridge",
        "create_feedback_controller",
        "create_feedback_router",
        "create_full_selfplay_training_loop",
        "create_lifecycle_manager",
        "create_opponent_tracker",
        "get_evaluation_curriculum_bridge",
        "get_feedback_signal_router",
        "get_model_lifecycle_manager",
        "get_opponent_win_rate_tracker",
        "get_p2p_integration_manager",
        "get_pipeline_feedback_controller",
        "get_unified_loop_extensions",
        "integrate_evaluation_with_curriculum",
        "integrate_extensions",
        "integrate_feedback_with_selfplay",
        "integrate_lifecycle_with_p2p",
        "integrate_selfplay_with_training",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
