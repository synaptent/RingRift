"""Master loop daemon profile contract tests."""

from __future__ import annotations

from typing import Iterable

import pytest

from app.config.env import env
from app.coordination.daemon_registry import DAEMON_REGISTRY, get_deprecated_types
from app.coordination.daemon_types import DaemonType
from scripts.master_loop import MasterLoopController


def _profile_controller(profile: str, monkeypatch: pytest.MonkeyPatch) -> MasterLoopController:
    """Create a MasterLoopController shell without touching data/ state DBs."""
    controller = object.__new__(MasterLoopController)
    controller.daemon_profile = profile
    controller._has_aws_credentials = lambda: False
    controller._has_npx = lambda: False

    monkeypatch.setattr(env, "is_coordinator", False, raising=False)
    monkeypatch.setattr(env, "is_standby_coordinator", False, raising=False)
    monkeypatch.setattr(env, "node_id", "unit-test-node", raising=False)

    return controller


def _positions(daemons: Iterable[DaemonType]) -> dict[DaemonType, int]:
    return {daemon: index for index, daemon in enumerate(daemons)}


class TestLeanProfileContracts:
    """Lean profile should stay reusable without legacy/deprecated daemons."""

    def test_lean_profile_uses_registered_active_daemons(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        controller = _profile_controller("lean", monkeypatch)
        daemons = controller._get_daemons_for_profile()
        deprecated = get_deprecated_types()

        assert daemons
        assert all(daemon in DAEMON_REGISTRY for daemon in daemons)
        assert not (set(daemons) & deprecated)

    def test_lean_profile_contains_required_health_contracts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        controller = _profile_controller("lean", monkeypatch)
        daemons = set(controller._get_daemons_for_profile())

        required_health_daemons = {
            DaemonType.HEALTH_SERVER,
            DaemonType.COORDINATOR_HEALTH_MONITOR,
            DaemonType.CLUSTER_MONITOR,
            DaemonType.DISK_SPACE_MANAGER,
            DaemonType.MEMORY_MONITOR,
            DaemonType.P2P_RECOVERY,
            DaemonType.PROGRESS_WATCHDOG,
        }

        assert required_health_daemons <= daemons

        for daemon in required_health_daemons:
            spec = DAEMON_REGISTRY[daemon]
            effective_interval = spec.health_check_interval or 60.0
            assert spec.auto_restart is True
            assert 0 < effective_interval <= 1800.0

    def test_lean_profile_orders_hard_dependencies_before_dependents(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        controller = _profile_controller("lean", monkeypatch)
        daemons = controller._get_daemons_for_profile()
        positions = _positions(daemons)

        for daemon in daemons:
            spec = DAEMON_REGISTRY[daemon]
            for dependency in spec.depends_on:
                if dependency in positions:
                    assert positions[dependency] < positions[daemon], (
                        f"{dependency.name} must start before {daemon.name}"
                    )


class TestStartupOrdering:
    """Dependency-aware ordering should fix profile list mistakes."""

    def test_model_distribution_starts_after_evaluation_chain(self) -> None:
        daemons = [
            DaemonType.MODEL_DISTRIBUTION,
            DaemonType.AUTO_PROMOTION,
            DaemonType.EVALUATION,
            DaemonType.EVENT_ROUTER,
        ]

        ordered = MasterLoopController._order_daemons_for_startup(daemons)
        positions = _positions(ordered)

        assert ordered[0] == DaemonType.EVENT_ROUTER
        assert positions[DaemonType.EVALUATION] < positions[DaemonType.AUTO_PROMOTION]
        assert positions[DaemonType.AUTO_PROMOTION] < positions[DaemonType.MODEL_DISTRIBUTION]

    def test_ordering_preserves_all_selected_daemons_once(self) -> None:
        daemons = [
            DaemonType.MAINTENANCE,
            DaemonType.EVENT_ROUTER,
            DaemonType.TRAINING_TRIGGER,
            DaemonType.AUTO_SYNC,
            DaemonType.DATA_PIPELINE,
            DaemonType.FEEDBACK_LOOP,
        ]

        ordered = MasterLoopController._order_daemons_for_startup(daemons)

        assert set(ordered) == set(daemons)
        assert len(ordered) == len(daemons)
