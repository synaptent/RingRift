"""Tests for coordination package entrypoint exports."""

from __future__ import annotations

import app.coordination.cluster as cluster_pkg
import app.coordination.selfplay as selfplay_pkg
import app.coordination.training as training_pkg


def test_training_package_declares_public_exports() -> None:
    expected = {
        "TrainingCoordinator",
        "SelfplayOrchestrator",
        "PriorityJobScheduler",
        "UnifiedScheduler",
        "get_training_coordinator",
        "get_selfplay_orchestrator",
        "get_unified_scheduler",
        "wire_training_events",
        "wire_selfplay_events",
    }
    assert expected.issubset(set(training_pkg.__all__))
    assert training_pkg.TrainingCoordinator.__name__ == "TrainingCoordinator"
    assert training_pkg.SelfplayOrchestrator.__name__ == "SelfplayOrchestrator"
    assert training_pkg.PriorityJobScheduler.__name__ == "PriorityJobScheduler"
    assert training_pkg.UnifiedScheduler.__name__ == "UnifiedScheduler"
    for name in expected:
        assert name in dir(training_pkg)


def test_cluster_package_declares_public_exports() -> None:
    assert cluster_pkg.__all__ == ["health", "transport", "p2p"]
    assert cluster_pkg.health.__name__ == "app.coordination.cluster.health"
    assert cluster_pkg.transport.__name__ == "app.coordination.cluster_transport"
    assert cluster_pkg.p2p.__name__ == "app.coordination.p2p_backend"
    assert "health" in dir(cluster_pkg)
    assert "transport" in dir(cluster_pkg)
    assert "p2p" in dir(cluster_pkg)


def test_selfplay_package_dir_lists_lazy_scheduler_exports() -> None:
    for name in (
        "SelfplayScheduler",
        "get_selfplay_scheduler",
        "reset_selfplay_scheduler",
    ):
        assert name in selfplay_pkg.__all__
        assert name in dir(selfplay_pkg)
