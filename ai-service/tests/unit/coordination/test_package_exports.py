"""Tests for coordination package entrypoint exports."""

from __future__ import annotations

import warnings

import app.coordination.cluster as cluster_pkg
import app.coordination.selfplay as selfplay_pkg
import app.coordination.training as training_pkg


def test_training_package_declares_public_submodules() -> None:
    assert training_pkg.__all__ == ["orchestrator", "scheduler"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert (
            training_pkg.orchestrator.__name__
            == "app.coordination.training.orchestrator"
        )
        assert (
            training_pkg.scheduler.__name__
            == "app.coordination.training.scheduler"
        )
    assert "orchestrator" in dir(training_pkg)
    assert "scheduler" in dir(training_pkg)


def test_cluster_package_declares_public_submodules() -> None:
    assert cluster_pkg.__all__ == ["health", "sync", "transport", "p2p"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert cluster_pkg.health.__name__ == "app.coordination.cluster.health"
        assert cluster_pkg.sync.__name__ == "app.coordination.cluster.sync"
        assert cluster_pkg.transport.__name__ == "app.coordination.cluster_transport"
        assert cluster_pkg.p2p.__name__ == "app.coordination.p2p_backend"
    assert "health" in dir(cluster_pkg)
    assert "sync" in dir(cluster_pkg)


def test_selfplay_package_dir_lists_lazy_scheduler_exports() -> None:
    for name in (
        "SelfplayScheduler",
        "get_selfplay_scheduler",
        "reset_selfplay_scheduler",
    ):
        assert name in selfplay_pkg.__all__
        assert name in dir(selfplay_pkg)
