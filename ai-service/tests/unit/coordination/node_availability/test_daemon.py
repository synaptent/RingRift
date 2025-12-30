"""Tests for node_availability/daemon.py - Cloud provider state sync.

Tests cover:
- NodeAvailabilityConfig from environment variables
- DaemonStats tracking
- NodeAvailabilityDaemon initialization and checker setup
- Grace period logic for terminated nodes
- Health check reporting
- Status API
- Singleton pattern
- Cycle execution with mocked providers

December 2025 - Test coverage for critical daemon module.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.node_availability.daemon import (
    DaemonStats,
    NodeAvailabilityConfig,
    NodeAvailabilityDaemon,
    get_node_availability_daemon,
    reset_daemon_instance,
)
from app.coordination.node_availability.state_checker import (
    InstanceInfo,
    ProviderInstanceState,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before and after each test."""
    reset_daemon_instance()
    yield
    reset_daemon_instance()


@pytest.fixture
def mock_env():
    """Fixture to mock environment variables."""
    original = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original)


@pytest.fixture
def mock_checkers():
    """Mock all provider checkers as disabled."""
    with patch(
        "app.coordination.node_availability.daemon.VastChecker"
    ) as vast, patch(
        "app.coordination.node_availability.daemon.LambdaChecker"
    ) as lambda_c, patch(
        "app.coordination.node_availability.daemon.RunPodChecker"
    ) as runpod:
        vast.return_value.is_enabled = False
        lambda_c.return_value.is_enabled = False
        runpod.return_value.is_enabled = False
        yield {"vast": vast, "lambda": lambda_c, "runpod": runpod}


@pytest.fixture
def mock_config_updater():
    """Mock ConfigUpdater."""
    with patch(
        "app.coordination.node_availability.daemon.ConfigUpdater"
    ) as updater_class:
        updater = MagicMock()
        updater.load_config = MagicMock(return_value={"hosts": {}})
        updater.update_node_statuses = AsyncMock(
            return_value=MagicMock(
                success=True,
                update_count=0,
                dry_run=True,
                changes={},
                error=None,
            )
        )
        updater_class.return_value = updater
        yield updater


# =============================================================================
# NodeAvailabilityConfig Tests
# =============================================================================


class TestNodeAvailabilityConfig:
    """Tests for NodeAvailabilityConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = NodeAvailabilityConfig()

        assert config.check_interval_seconds == 300.0
        assert config.dry_run is True
        assert config.grace_period_seconds == 60.0
        assert config.vast_enabled is True
        assert config.lambda_enabled is True
        assert config.runpod_enabled is True
        assert config.vultr_enabled is True
        assert config.hetzner_enabled is True
        assert config.auto_update_voters is False

    def test_from_env_defaults(self, mock_env: None) -> None:
        """Test from_env with no environment variables."""
        config = NodeAvailabilityConfig.from_env()

        assert config.enabled is True
        assert config.dry_run is True
        assert config.check_interval_seconds == 300.0

    def test_from_env_disabled(self, mock_env: None) -> None:
        """Test from_env with daemon disabled."""
        os.environ["RINGRIFT_NODE_AVAILABILITY_ENABLED"] = "false"

        config = NodeAvailabilityConfig.from_env()
        assert config.enabled is False

    def test_from_env_dry_run_false(self, mock_env: None) -> None:
        """Test from_env with dry_run disabled."""
        os.environ["RINGRIFT_NODE_AVAILABILITY_DRY_RUN"] = "0"

        config = NodeAvailabilityConfig.from_env()
        assert config.dry_run is False

    def test_from_env_custom_interval(self, mock_env: None) -> None:
        """Test from_env with custom interval."""
        os.environ["RINGRIFT_NODE_AVAILABILITY_INTERVAL"] = "60"

        config = NodeAvailabilityConfig.from_env()
        assert config.check_interval_seconds == 60.0

    def test_from_env_custom_grace_period(self, mock_env: None) -> None:
        """Test from_env with custom grace period."""
        os.environ["RINGRIFT_NODE_AVAILABILITY_GRACE_PERIOD"] = "120"

        config = NodeAvailabilityConfig.from_env()
        assert config.grace_period_seconds == 120.0

    def test_from_env_provider_toggles(self, mock_env: None) -> None:
        """Test from_env with provider toggles."""
        os.environ["RINGRIFT_NODE_AVAILABILITY_VAST"] = "0"
        os.environ["RINGRIFT_NODE_AVAILABILITY_LAMBDA"] = "false"
        os.environ["RINGRIFT_NODE_AVAILABILITY_RUNPOD"] = "1"

        config = NodeAvailabilityConfig.from_env()
        assert config.vast_enabled is False
        assert config.lambda_enabled is False
        assert config.runpod_enabled is True

    def test_from_env_auto_voters(self, mock_env: None) -> None:
        """Test from_env with auto_update_voters enabled."""
        os.environ["RINGRIFT_NODE_AVAILABILITY_AUTO_VOTERS"] = "true"

        config = NodeAvailabilityConfig.from_env()
        assert config.auto_update_voters is True


# =============================================================================
# DaemonStats Tests
# =============================================================================


class TestDaemonStats:
    """Tests for DaemonStats dataclass."""

    def test_initial_values(self) -> None:
        """Test initial statistics values."""
        stats = DaemonStats()

        assert stats.cycles_completed == 0
        assert stats.last_cycle_time is None
        assert stats.last_cycle_duration_seconds == 0.0
        assert stats.provider_checks == {}
        assert stats.provider_errors == {}
        assert stats.total_updates == 0
        assert stats.nodes_updated == 0
        assert stats.dry_run_updates == 0

    def test_record_cycle(self) -> None:
        """Test recording a completed cycle."""
        stats = DaemonStats()

        stats.record_cycle(1.5)

        assert stats.cycles_completed == 1
        assert stats.last_cycle_time is not None
        assert stats.last_cycle_duration_seconds == 1.5

    def test_record_multiple_cycles(self) -> None:
        """Test recording multiple cycles."""
        stats = DaemonStats()

        stats.record_cycle(1.0)
        stats.record_cycle(2.0)
        stats.record_cycle(3.0)

        assert stats.cycles_completed == 3
        assert stats.last_cycle_duration_seconds == 3.0

    def test_record_provider_check_success(self) -> None:
        """Test recording successful provider check."""
        stats = DaemonStats()

        stats.record_provider_check("vast", success=True)

        assert stats.provider_checks["vast"] == 1
        assert stats.provider_errors.get("vast", 0) == 0

    def test_record_provider_check_failure(self) -> None:
        """Test recording failed provider check."""
        stats = DaemonStats()

        stats.record_provider_check("lambda", success=False)

        assert stats.provider_checks["lambda"] == 1
        assert stats.provider_errors["lambda"] == 1

    def test_record_provider_check_multiple(self) -> None:
        """Test recording multiple provider checks."""
        stats = DaemonStats()

        stats.record_provider_check("vast", success=True)
        stats.record_provider_check("vast", success=True)
        stats.record_provider_check("vast", success=False)

        assert stats.provider_checks["vast"] == 3
        assert stats.provider_errors["vast"] == 1

    def test_record_update(self) -> None:
        """Test recording an update result."""
        stats = DaemonStats()

        result = MagicMock()
        result.update_count = 5
        result.dry_run = False

        stats.record_update(result)

        assert stats.total_updates == 1
        assert stats.nodes_updated == 5
        assert stats.dry_run_updates == 0

    def test_record_update_dry_run(self) -> None:
        """Test recording a dry-run update."""
        stats = DaemonStats()

        result = MagicMock()
        result.update_count = 3
        result.dry_run = True

        stats.record_update(result)

        assert stats.total_updates == 1
        assert stats.nodes_updated == 3
        assert stats.dry_run_updates == 1


# =============================================================================
# NodeAvailabilityDaemon Tests
# =============================================================================


class TestNodeAvailabilityDaemonInit:
    """Tests for NodeAvailabilityDaemon initialization."""

    def test_default_initialization(
        self, mock_checkers: dict, mock_config_updater: Any
    ) -> None:
        """Test default daemon initialization."""
        daemon = NodeAvailabilityDaemon()

        assert daemon.config.dry_run is True
        assert isinstance(daemon._daemon_stats, DaemonStats)
        assert daemon._pending_terminations == {}

    def test_custom_config_initialization(
        self, mock_checkers: dict, mock_config_updater: Any
    ) -> None:
        """Test initialization with custom config."""
        config = NodeAvailabilityConfig(
            dry_run=False,
            check_interval_seconds=60.0,
            grace_period_seconds=30.0,
        )

        daemon = NodeAvailabilityDaemon(config)

        assert daemon.config.dry_run is False
        assert daemon.config.check_interval_seconds == 60.0
        assert daemon.config.grace_period_seconds == 30.0

    def test_checkers_disabled_no_api_keys(
        self, mock_checkers: dict, mock_config_updater: Any
    ) -> None:
        """Test that checkers are not added when API keys are missing."""
        daemon = NodeAvailabilityDaemon()

        # All checkers disabled (no API keys)
        assert len(daemon._checkers) == 0


class TestNodeAvailabilityDaemonGracePeriod:
    """Tests for grace period logic."""

    def test_first_termination_starts_grace_period(
        self, mock_checkers: dict, mock_config_updater: Any
    ) -> None:
        """Test that first termination starts grace period."""
        config = NodeAvailabilityConfig(grace_period_seconds=60.0)
        daemon = NodeAvailabilityDaemon(config)

        result = daemon._check_grace_period("node-001")

        assert result is False
        assert "node-001" in daemon._pending_terminations

    def test_grace_period_not_passed(
        self, mock_checkers: dict, mock_config_updater: Any
    ) -> None:
        """Test that node in grace period returns False."""
        config = NodeAvailabilityConfig(grace_period_seconds=60.0)
        daemon = NodeAvailabilityDaemon(config)

        # First call starts grace period
        daemon._check_grace_period("node-001")

        # Second call within grace period
        result = daemon._check_grace_period("node-001")

        assert result is False
        assert "node-001" in daemon._pending_terminations

    def test_grace_period_passed(
        self, mock_checkers: dict, mock_config_updater: Any
    ) -> None:
        """Test that node past grace period returns True."""
        config = NodeAvailabilityConfig(grace_period_seconds=0.1)  # 100ms
        daemon = NodeAvailabilityDaemon(config)

        # Start grace period
        daemon._check_grace_period("node-001")

        # Wait for grace period to pass
        time.sleep(0.15)

        result = daemon._check_grace_period("node-001")

        assert result is True
        # Node removed from pending after grace period
        assert "node-001" not in daemon._pending_terminations

    def test_multiple_nodes_independent_grace_periods(
        self, mock_checkers: dict, mock_config_updater: Any
    ) -> None:
        """Test that each node has independent grace period."""
        config = NodeAvailabilityConfig(grace_period_seconds=60.0)
        daemon = NodeAvailabilityDaemon(config)

        daemon._check_grace_period("node-001")
        daemon._check_grace_period("node-002")

        assert "node-001" in daemon._pending_terminations
        assert "node-002" in daemon._pending_terminations


class TestNodeAvailabilityDaemonHealthCheck:
    """Tests for health_check method."""

    def test_health_check_no_checkers(
        self, mock_checkers: dict, mock_config_updater: Any
    ) -> None:
        """Test health check with no enabled checkers."""
        daemon = NodeAvailabilityDaemon()

        result = daemon.health_check()

        assert result.healthy is False
        assert "No provider checkers enabled" in result.message

    def test_health_check_with_checkers(
        self, mock_config_updater: Any
    ) -> None:
        """Test health check with enabled checkers."""
        with patch(
            "app.coordination.node_availability.daemon.VastChecker"
        ) as vast:
            vast.return_value.is_enabled = True
            with patch(
                "app.coordination.node_availability.daemon.LambdaChecker"
            ) as lambda_c:
                lambda_c.return_value.is_enabled = False
                with patch(
                    "app.coordination.node_availability.daemon.RunPodChecker"
                ) as runpod:
                    runpod.return_value.is_enabled = False

                    daemon = NodeAvailabilityDaemon()

                    result = daemon.health_check()

                    assert result.healthy is True
                    assert "vast" in result.details["enabled_providers"]

    def test_health_check_high_error_rate(
        self, mock_config_updater: Any
    ) -> None:
        """Test health check with high provider error rate."""
        with patch(
            "app.coordination.node_availability.daemon.VastChecker"
        ) as vast:
            vast.return_value.is_enabled = True
            with patch(
                "app.coordination.node_availability.daemon.LambdaChecker"
            ), patch(
                "app.coordination.node_availability.daemon.RunPodChecker"
            ):
                daemon = NodeAvailabilityDaemon()

                # Simulate high error rate
                daemon._daemon_stats.provider_checks["vast"] = 10
                daemon._daemon_stats.provider_errors["vast"] = 6

                result = daemon.health_check()

                assert result.healthy is False
                assert "high error rate" in result.message.lower()

    def test_health_check_details(
        self, mock_checkers: dict, mock_config_updater: Any
    ) -> None:
        """Test health check includes correct details."""
        daemon = NodeAvailabilityDaemon()
        daemon._daemon_stats.cycles_completed = 5
        daemon._daemon_stats.total_updates = 10
        daemon._daemon_stats.nodes_updated = 25

        result = daemon.health_check()

        assert result.details["cycles_completed"] == 5
        assert result.details["total_updates"] == 10
        assert result.details["nodes_updated"] == 25
        assert result.details["dry_run"] is True


class TestNodeAvailabilityDaemonStatus:
    """Tests for get_status method."""

    def test_get_status_initial(
        self, mock_checkers: dict, mock_config_updater: Any
    ) -> None:
        """Test status with initial daemon state."""
        daemon = NodeAvailabilityDaemon()

        status = daemon.get_status()

        assert status["running"] is False
        assert status["config"]["enabled"] is True
        assert status["config"]["dry_run"] is True
        assert status["config"]["interval_seconds"] == 300.0
        assert status["stats"]["cycles_completed"] == 0
        assert status["stats"]["total_updates"] == 0
        assert status["pending_terminations"] == []

    def test_get_status_with_stats(
        self, mock_checkers: dict, mock_config_updater: Any
    ) -> None:
        """Test status after recording activity."""
        daemon = NodeAvailabilityDaemon()

        # Simulate activity
        daemon._daemon_stats.record_cycle(2.5)
        daemon._daemon_stats.total_updates = 3
        daemon._daemon_stats.nodes_updated = 7
        daemon._pending_terminations["node-001"] = time.time()

        status = daemon.get_status()

        assert status["stats"]["cycles_completed"] == 1
        assert status["stats"]["last_cycle_duration"] == 2.5
        assert status["stats"]["total_updates"] == 3
        assert status["stats"]["nodes_updated"] == 7
        assert "node-001" in status["pending_terminations"]


# =============================================================================
# Singleton Pattern Tests
# =============================================================================


class TestSingletonPattern:
    """Tests for singleton pattern functions."""

    def test_get_node_availability_daemon_creates_instance(
        self, mock_checkers: dict, mock_config_updater: Any
    ) -> None:
        """Test that get_node_availability_daemon creates singleton."""
        daemon1 = get_node_availability_daemon()
        daemon2 = get_node_availability_daemon()

        assert daemon1 is daemon2

    def test_get_node_availability_daemon_with_config(
        self, mock_checkers: dict, mock_config_updater: Any
    ) -> None:
        """Test that config is only used on first call."""
        config1 = NodeAvailabilityConfig(check_interval_seconds=60.0)
        config2 = NodeAvailabilityConfig(check_interval_seconds=120.0)

        daemon1 = get_node_availability_daemon(config1)
        daemon2 = get_node_availability_daemon(config2)

        assert daemon1.config.check_interval_seconds == 60.0
        assert daemon2.config.check_interval_seconds == 60.0  # Same instance

    def test_reset_daemon_instance(
        self, mock_checkers: dict, mock_config_updater: Any
    ) -> None:
        """Test that reset_daemon_instance clears singleton."""
        daemon1 = get_node_availability_daemon()

        reset_daemon_instance()

        daemon2 = get_node_availability_daemon()

        assert daemon1 is not daemon2


# =============================================================================
# Run Cycle Tests
# =============================================================================


class TestNodeAvailabilityDaemonCycle:
    """Tests for _run_cycle method."""

    @pytest.mark.asyncio
    async def test_run_cycle_no_updates(
        self, mock_config_updater: Any
    ) -> None:
        """Test run cycle with no updates needed."""
        with patch(
            "app.coordination.node_availability.daemon.VastChecker"
        ) as vast_class:
            checker = MagicMock()
            checker.is_enabled = True
            checker.get_instance_states = AsyncMock(return_value=[])
            checker.get_terminated_instances = AsyncMock(return_value=[])
            vast_class.return_value = checker

            with patch(
                "app.coordination.node_availability.daemon.LambdaChecker"
            ) as lambda_class, patch(
                "app.coordination.node_availability.daemon.RunPodChecker"
            ) as runpod_class:
                lambda_class.return_value.is_enabled = False
                runpod_class.return_value.is_enabled = False
                daemon = NodeAvailabilityDaemon()

                await daemon._run_cycle()

                assert daemon._daemon_stats.cycles_completed == 1
                assert daemon._daemon_stats.provider_checks.get("vast", 0) == 1

    @pytest.mark.asyncio
    async def test_run_cycle_with_updates(
        self, mock_config_updater: Any
    ) -> None:
        """Test run cycle with state updates."""
        mock_config_updater.load_config.return_value = {
            "hosts": {
                "vast-12345": {"status": "ready"},
            }
        }

        # Mock update result with changes
        update_result = MagicMock()
        update_result.success = True
        update_result.update_count = 1
        update_result.dry_run = True
        update_result.changes = {"vast-12345": ("ready", "offline")}
        mock_config_updater.update_node_statuses = AsyncMock(
            return_value=update_result
        )

        with patch(
            "app.coordination.node_availability.daemon.VastChecker"
        ) as vast_class:
            checker = MagicMock()
            checker.is_enabled = True

            instance = InstanceInfo(
                instance_id="12345",
                state=ProviderInstanceState.STOPPED,
                provider="vast",
                node_name="vast-12345",
            )
            checker.get_instance_states = AsyncMock(return_value=[instance])
            checker.get_terminated_instances = AsyncMock(return_value=[])
            checker.correlate_with_config = MagicMock(return_value=[instance])
            vast_class.return_value = checker

            with patch(
                "app.coordination.node_availability.daemon.LambdaChecker"
            ) as lambda_class, patch(
                "app.coordination.node_availability.daemon.RunPodChecker"
            ) as runpod_class:
                lambda_class.return_value.is_enabled = False
                runpod_class.return_value.is_enabled = False
                daemon = NodeAvailabilityDaemon()

                with patch.object(
                    daemon, "_emit_state_change_event", new_callable=AsyncMock
                ) as emit_mock:
                    await daemon._run_cycle()

                    assert daemon._daemon_stats.total_updates == 1
                    emit_mock.assert_called_once_with(
                        "vast-12345", "ready", "offline"
                    )

    @pytest.mark.asyncio
    async def test_run_cycle_provider_error(
        self, mock_config_updater: Any
    ) -> None:
        """Test run cycle handles provider errors gracefully."""
        with patch(
            "app.coordination.node_availability.daemon.VastChecker"
        ) as vast_class:
            checker = MagicMock()
            checker.is_enabled = True
            checker.get_instance_states = AsyncMock(
                side_effect=asyncio.TimeoutError("API timeout")
            )
            vast_class.return_value = checker

            with patch(
                "app.coordination.node_availability.daemon.LambdaChecker"
            ) as lambda_class, patch(
                "app.coordination.node_availability.daemon.RunPodChecker"
            ) as runpod_class:
                lambda_class.return_value.is_enabled = False
                runpod_class.return_value.is_enabled = False
                daemon = NodeAvailabilityDaemon()

                await daemon._run_cycle()

                assert daemon._daemon_stats.cycles_completed == 1
                assert daemon._daemon_stats.provider_errors.get("vast", 0) == 1


# =============================================================================
# Stop Tests
# =============================================================================


class TestNodeAvailabilityDaemonStop:
    """Tests for stop method."""

    @pytest.mark.asyncio
    async def test_stop_closes_checker_sessions(
        self, mock_config_updater: Any
    ) -> None:
        """Test that stop closes HTTP sessions."""
        with patch(
            "app.coordination.node_availability.daemon.VastChecker"
        ) as vast_class:
            checker = MagicMock()
            checker.is_enabled = True
            checker.close = AsyncMock()
            vast_class.return_value = checker

            with patch(
                "app.coordination.node_availability.daemon.LambdaChecker"
            ) as lambda_class, patch(
                "app.coordination.node_availability.daemon.RunPodChecker"
            ) as runpod_class:
                lambda_class.return_value.is_enabled = False
                runpod_class.return_value.is_enabled = False
                daemon = NodeAvailabilityDaemon()

                await daemon.stop()

                checker.close.assert_called_once()
