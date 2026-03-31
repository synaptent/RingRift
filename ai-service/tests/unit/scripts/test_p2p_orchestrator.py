"""Unit tests for P2P Orchestrator components.

Tests cover standalone components that can be tested in isolation:
- Constants and configuration
- Basic imports verification

December 2025: Initial test coverage for P2P orchestrator components.
Manager classes require full orchestrator context and are tested via integration tests
in tests/unit/p2p/.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

# =============================================================================
# Test Constants and Configuration
# =============================================================================


class TestP2PConstants:
    """Tests for P2P constants and configuration."""

    def test_default_port(self):
        """Default port is set correctly."""
        from scripts.p2p.constants import DEFAULT_PORT

        assert DEFAULT_PORT == 8770

    def test_bootstrap_seeds_exist(self):
        """Bootstrap seeds are defined."""
        from scripts.p2p.constants import BOOTSTRAP_SEEDS

        assert isinstance(BOOTSTRAP_SEEDS, (list, tuple))

    def test_node_role_enum(self):
        """NodeRole enum has expected values."""
        from scripts.p2p.models import NodeRole

        assert NodeRole.LEADER.value == "leader"
        assert NodeRole.FOLLOWER.value == "follower"
        assert NodeRole.CANDIDATE.value == "candidate"


# =============================================================================
# Test Module Imports
# =============================================================================


class TestModuleImports:
    """Tests for module import availability."""

    def test_models_module_importable(self):
        """Models module is importable."""
        import scripts.p2p.models as models
        assert models is not None

    def test_leader_election_module_importable(self):
        """Leader election module is importable."""
        import scripts.p2p.leader_election as leader_election
        assert leader_election is not None

    def test_gossip_protocol_module_importable(self):
        """Gossip protocol module is importable."""
        import scripts.p2p.gossip_protocol as gossip_protocol
        assert gossip_protocol is not None

    def test_network_utils_module_importable(self):
        """Network utils module is importable."""
        import scripts.p2p.network_utils as network_utils
        assert network_utils is not None

    def test_orchestrator_module_importable(self):
        """Orchestrator module is importable."""
        import scripts.p2p_orchestrator as p2p_orchestrator
        assert hasattr(p2p_orchestrator, "P2POrchestrator")

    def test_managers_importable(self):
        """Manager modules are importable."""
        import scripts.p2p.managers.job_manager as job_manager
        import scripts.p2p.managers.selfplay_scheduler as selfplay_scheduler
        import scripts.p2p.managers.state_manager as state_manager
        assert state_manager is not None
        assert job_manager is not None
        assert selfplay_scheduler is not None


# =============================================================================
# Test Manager Health Validation
# =============================================================================


class MockManager:
    """Mock manager for testing health validation."""

    def __init__(self, health_status: str = "healthy", has_health_check: bool = True):
        self._health_status = health_status
        self._has_health_check = has_health_check

    def health_check(self):
        if not self._has_health_check:
            raise AttributeError("No health_check method")
        return {
            "status": self._health_status,
            "operations_count": 10,
            "errors_count": 0,
        }


class MockManagerWithHealthCheckResult:
    """Mock manager that returns HealthCheckResult-like object."""

    class MockHealthResult:
        def __init__(self, status: str):
            self.status = status

    def __init__(self, status: str = "ready"):
        self._status = status

    def health_check(self):
        return self.MockHealthResult(self._status)


class TestValidateManagerHealth:
    """Tests for HealthMetricsManager.validate_manager_health().

    December 27, 2025: Tests for the consolidated manager health validation
    that runs at P2P startup to catch initialization issues early.
    """

    def _create_mock_orchestrator(
        self,
        state_manager=None,
        node_selector=None,
        sync_planner=None,
        selfplay_scheduler=None,
        job_manager=None,
        training_coordinator=None,
        loop_manager=None,
        job_orchestration=None,
    ):
        """Create a mock orchestrator with the given managers."""
        class MockOrchestrator:
            def _get_loop_manager(self_inner):
                return self_inner._loop_manager

        orch = MockOrchestrator()
        orch.state_manager = state_manager
        orch.node_selector = node_selector
        orch.sync_planner = sync_planner
        orch.selfplay_scheduler = selfplay_scheduler
        orch.job_manager = job_manager
        orch.training_coordinator = training_coordinator
        orch._loop_manager = loop_manager
        orch.job_orchestration = job_orchestration
        orch.start_time = 0.0
        return orch

    def _validate_manager_health(self, orch):
        """Run the extracted health metrics manager directly."""
        from scripts.p2p.managers.health_metrics_manager import (
            HealthMetricsConfig,
            HealthMetricsManager,
        )

        manager = HealthMetricsManager(
            config=HealthMetricsConfig(startup_grace_period=0.0),
            orchestrator=orch,
        )
        try:
            return manager.validate_manager_health()
        finally:
            if manager._health_check_executor is not None:
                manager._health_check_executor.shutdown(wait=False, cancel_futures=True)

    def test_all_managers_healthy(self):
        """All managers healthy returns all_healthy=True."""
        orch = self._create_mock_orchestrator(
            state_manager=MockManager("healthy"),
            node_selector=MockManager("healthy"),
            sync_planner=MockManager("healthy"),
            selfplay_scheduler=MockManager("healthy"),
            job_manager=MockManager("healthy"),
            training_coordinator=MockManager("healthy"),
            loop_manager=MockManager("healthy"),
            job_orchestration=MockManager("healthy"),
        )

        status = self._validate_manager_health(orch)

        assert status["all_healthy"] is True
        assert status["unhealthy_count"] == 0
        assert len(status["managers"]) == 8

    def test_one_manager_unhealthy(self):
        """One unhealthy manager sets all_healthy=False."""
        orch = self._create_mock_orchestrator(
            state_manager=MockManager("healthy"),
            node_selector=MockManager("healthy"),
            sync_planner=MockManager("degraded"),  # Unhealthy
            selfplay_scheduler=MockManager("healthy"),
            job_manager=MockManager("healthy"),
            training_coordinator=MockManager("healthy"),
            loop_manager=MockManager("healthy"),
            job_orchestration=MockManager("healthy"),
        )

        status = self._validate_manager_health(orch)

        assert status["all_healthy"] is False
        assert status["unhealthy_count"] == 1
        assert status["managers"]["sync_planner"]["status"] == "degraded"

    def test_none_manager_detected(self):
        """None manager is detected and marked unhealthy."""
        orch = self._create_mock_orchestrator(
            state_manager=MockManager("healthy"),
            node_selector=None,  # Not initialized
            sync_planner=MockManager("healthy"),
            selfplay_scheduler=MockManager("healthy"),
            job_manager=MockManager("healthy"),
            training_coordinator=MockManager("healthy"),
            loop_manager=MockManager("healthy"),
            job_orchestration=MockManager("healthy"),
        )

        status = self._validate_manager_health(orch)

        assert status["all_healthy"] is False
        assert status["unhealthy_count"] == 1
        assert status["managers"]["node_selector"]["status"] == "not_initialized"

    def test_manager_without_health_check(self):
        """Manager without health_check is marked as initialized."""
        class NoHealthCheckManager:
            pass

        orch = self._create_mock_orchestrator(
            state_manager=NoHealthCheckManager(),
            node_selector=MockManager("healthy"),
            sync_planner=MockManager("healthy"),
            selfplay_scheduler=MockManager("healthy"),
            job_manager=MockManager("healthy"),
            training_coordinator=MockManager("healthy"),
            loop_manager=MockManager("healthy"),
            job_orchestration=MockManager("healthy"),
        )

        status = self._validate_manager_health(orch)

        # Manager without health_check is still considered OK (initialized)
        assert status["managers"]["state_manager"]["status"] == "initialized"
        assert status["managers"]["state_manager"]["health_check"] == "not_available"

    def test_health_check_exception_handled(self):
        """Exception in health_check is caught and logged."""
        class BrokenManager:
            def health_check(self):
                raise RuntimeError("Health check failed!")

        orch = self._create_mock_orchestrator(
            state_manager=BrokenManager(),
            node_selector=MockManager("healthy"),
            sync_planner=MockManager("healthy"),
            selfplay_scheduler=MockManager("healthy"),
            job_manager=MockManager("healthy"),
            training_coordinator=MockManager("healthy"),
            loop_manager=MockManager("healthy"),
            job_orchestration=MockManager("healthy"),
        )

        status = self._validate_manager_health(orch)

        assert status["all_healthy"] is False
        assert status["unhealthy_count"] == 1
        assert status["managers"]["state_manager"]["status"] == "error"
        assert "Health check failed!" in status["managers"]["state_manager"]["error"]

    def test_health_check_result_object_supported(self):
        """HealthCheckResult-like objects are properly parsed."""
        orch = self._create_mock_orchestrator(
            state_manager=MockManagerWithHealthCheckResult("ready"),
            node_selector=MockManager("healthy"),
            sync_planner=MockManager("healthy"),
            selfplay_scheduler=MockManager("healthy"),
            job_manager=MockManager("healthy"),
            training_coordinator=MockManager("healthy"),
            loop_manager=MockManager("healthy"),
            job_orchestration=MockManager("healthy"),
        )

        status = self._validate_manager_health(orch)

        # "ready" status should be recognized as healthy
        assert status["all_healthy"] is True
        assert status["managers"]["state_manager"]["status"] == "ready"

    def test_timestamp_included(self):
        """Status includes timestamp."""
        import time

        orch = self._create_mock_orchestrator(
            state_manager=MockManager("healthy"),
            node_selector=MockManager("healthy"),
            sync_planner=MockManager("healthy"),
            selfplay_scheduler=MockManager("healthy"),
            job_manager=MockManager("healthy"),
            training_coordinator=MockManager("healthy"),
            loop_manager=MockManager("healthy"),
            job_orchestration=MockManager("healthy"),
        )

        before = time.time()
        status = self._validate_manager_health(orch)
        after = time.time()

        assert "timestamp" in status
        assert before <= status["timestamp"] <= after

    def test_all_expected_managers_validated(self):
        """All extracted managers are validated."""
        orch = self._create_mock_orchestrator(
            state_manager=MockManager("healthy"),
            node_selector=MockManager("healthy"),
            sync_planner=MockManager("healthy"),
            selfplay_scheduler=MockManager("healthy"),
            job_manager=MockManager("healthy"),
            training_coordinator=MockManager("healthy"),
            loop_manager=MockManager("healthy"),
            job_orchestration=MockManager("healthy"),
        )

        status = self._validate_manager_health(orch)

        expected_managers = [
            "state_manager",
            "node_selector",
            "sync_planner",
            "selfplay_scheduler",
            "job_manager",
            "training_coordinator",
            "loop_manager",
            "job_orchestration",
        ]
        assert set(status["managers"].keys()) == set(expected_managers)


class TestGitUpdateSafety:
    """Tests for git update safety checks."""

    @pytest.mark.asyncio
    async def test_check_local_changes_fails_closed_when_git_status_errors(self):
        """Auto-update must stop if git status cannot confirm a clean tree."""
        from scripts.p2p_orchestrator import P2POrchestrator

        orch = P2POrchestrator.__new__(P2POrchestrator)
        orch.ringrift_path = "/tmp/ringrift"
        orch._git_safe_directory = "/tmp/ringrift"

        failed_status = SimpleNamespace(
            returncode=128,
            stdout="",
            stderr="fatal: not a git repository",
        )

        with patch(
            "scripts.p2p_orchestrator.async_subprocess_run",
            AsyncMock(return_value=failed_status),
        ):
            assert await P2POrchestrator._check_local_changes(orch) is True
