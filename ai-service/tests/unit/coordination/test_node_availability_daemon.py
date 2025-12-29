"""Unit tests for NodeAvailabilityDaemon (December 2025).

Tests for the daemon that synchronizes cloud provider instance state
with distributed_hosts.yaml.
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.contracts import HealthCheckResult
from app.coordination.node_availability.daemon import (
    DaemonStats,
    NodeAvailabilityConfig,
    NodeAvailabilityDaemon,
    get_node_availability_daemon,
    reset_daemon_instance,
)


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

    def test_from_env_enabled(self) -> None:
        """Test loading enabled state from environment."""
        with patch.dict(os.environ, {"RINGRIFT_NODE_AVAILABILITY_ENABLED": "0"}):
            config = NodeAvailabilityConfig.from_env()
            assert config.enabled is False

        with patch.dict(os.environ, {"RINGRIFT_NODE_AVAILABILITY_ENABLED": "1"}):
            config = NodeAvailabilityConfig.from_env()
            assert config.enabled is True

    def test_from_env_dry_run(self) -> None:
        """Test loading dry_run from environment."""
        with patch.dict(os.environ, {"RINGRIFT_NODE_AVAILABILITY_DRY_RUN": "false"}):
            config = NodeAvailabilityConfig.from_env()
            assert config.dry_run is False

        with patch.dict(os.environ, {"RINGRIFT_NODE_AVAILABILITY_DRY_RUN": "true"}):
            config = NodeAvailabilityConfig.from_env()
            assert config.dry_run is True

    def test_from_env_interval(self) -> None:
        """Test loading check interval from environment."""
        with patch.dict(os.environ, {"RINGRIFT_NODE_AVAILABILITY_INTERVAL": "60"}):
            config = NodeAvailabilityConfig.from_env()
            assert config.check_interval_seconds == 60.0

    def test_from_env_grace_period(self) -> None:
        """Test loading grace period from environment."""
        with patch.dict(os.environ, {"RINGRIFT_NODE_AVAILABILITY_GRACE_PERIOD": "120"}):
            config = NodeAvailabilityConfig.from_env()
            assert config.grace_period_seconds == 120.0

    def test_from_env_provider_toggles(self) -> None:
        """Test loading provider toggles from environment."""
        with patch.dict(os.environ, {
            "RINGRIFT_NODE_AVAILABILITY_VAST": "0",
            "RINGRIFT_NODE_AVAILABILITY_LAMBDA": "0",
            "RINGRIFT_NODE_AVAILABILITY_RUNPOD": "0",
        }):
            config = NodeAvailabilityConfig.from_env()
            assert config.vast_enabled is False
            assert config.lambda_enabled is False
            assert config.runpod_enabled is False

    def test_from_env_auto_voters(self) -> None:
        """Test loading auto_update_voters from environment."""
        with patch.dict(os.environ, {"RINGRIFT_NODE_AVAILABILITY_AUTO_VOTERS": "1"}):
            config = NodeAvailabilityConfig.from_env()
            assert config.auto_update_voters is True


class TestDaemonStats:
    """Tests for DaemonStats dataclass."""

    def test_default_values(self) -> None:
        """Test default stats values."""
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
        """Test recording a cycle."""
        stats = DaemonStats()

        before = datetime.now()
        stats.record_cycle(1.5)
        after = datetime.now()

        assert stats.cycles_completed == 1
        assert stats.last_cycle_duration_seconds == 1.5
        assert stats.last_cycle_time is not None
        assert before <= stats.last_cycle_time <= after

    def test_record_cycle_increments(self) -> None:
        """Test that record_cycle increments counter."""
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
        stats.record_provider_check("vast", success=True)

        assert stats.provider_checks["vast"] == 2
        assert stats.provider_errors.get("vast", 0) == 0

    def test_record_provider_check_failure(self) -> None:
        """Test recording failed provider check."""
        stats = DaemonStats()

        stats.record_provider_check("lambda", success=False)

        assert stats.provider_checks["lambda"] == 1
        assert stats.provider_errors["lambda"] == 1

    def test_record_provider_check_mixed(self) -> None:
        """Test recording mixed provider check results."""
        stats = DaemonStats()

        stats.record_provider_check("runpod", success=True)
        stats.record_provider_check("runpod", success=False)
        stats.record_provider_check("runpod", success=True)

        assert stats.provider_checks["runpod"] == 3
        assert stats.provider_errors["runpod"] == 1

    def test_record_update(self) -> None:
        """Test recording an update result."""
        stats = DaemonStats()

        mock_result = MagicMock()
        mock_result.update_count = 5
        mock_result.dry_run = False

        stats.record_update(mock_result)

        assert stats.total_updates == 1
        assert stats.nodes_updated == 5
        assert stats.dry_run_updates == 0

    def test_record_update_dry_run(self) -> None:
        """Test recording a dry-run update result."""
        stats = DaemonStats()

        mock_result = MagicMock()
        mock_result.update_count = 3
        mock_result.dry_run = True

        stats.record_update(mock_result)

        assert stats.total_updates == 1
        assert stats.nodes_updated == 3
        assert stats.dry_run_updates == 1


class TestNodeAvailabilityDaemonInit:
    """Tests for NodeAvailabilityDaemon initialization."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_daemon_instance()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_daemon_instance()

    @patch("app.coordination.node_availability.daemon.VastChecker")
    @patch("app.coordination.node_availability.daemon.LambdaChecker")
    @patch("app.coordination.node_availability.daemon.RunPodChecker")
    def test_init_with_default_config(
        self,
        mock_runpod: MagicMock,
        mock_lambda: MagicMock,
        mock_vast: MagicMock,
    ) -> None:
        """Test daemon initialization with default config."""
        # Disable all checkers
        mock_vast.return_value.is_enabled = False
        mock_lambda.return_value.is_enabled = False
        mock_runpod.return_value.is_enabled = False

        daemon = NodeAvailabilityDaemon()

        assert daemon.config is not None
        assert daemon._stats is not None
        assert daemon._checkers == {}

    @patch("app.coordination.node_availability.daemon.VastChecker")
    @patch("app.coordination.node_availability.daemon.LambdaChecker")
    @patch("app.coordination.node_availability.daemon.RunPodChecker")
    def test_init_with_custom_config(
        self,
        mock_runpod: MagicMock,
        mock_lambda: MagicMock,
        mock_vast: MagicMock,
    ) -> None:
        """Test daemon initialization with custom config."""
        mock_vast.return_value.is_enabled = False
        mock_lambda.return_value.is_enabled = False
        mock_runpod.return_value.is_enabled = False

        config = NodeAvailabilityConfig(
            check_interval_seconds=60.0,
            dry_run=False,
        )
        daemon = NodeAvailabilityDaemon(config)

        assert daemon.config.check_interval_seconds == 60.0
        assert daemon.config.dry_run is False

    @patch("app.coordination.node_availability.daemon.VastChecker")
    @patch("app.coordination.node_availability.daemon.LambdaChecker")
    @patch("app.coordination.node_availability.daemon.RunPodChecker")
    def test_init_checkers_enabled(
        self,
        mock_runpod: MagicMock,
        mock_lambda: MagicMock,
        mock_vast: MagicMock,
    ) -> None:
        """Test that enabled checkers are registered."""
        mock_vast.return_value.is_enabled = True
        mock_lambda.return_value.is_enabled = True
        mock_runpod.return_value.is_enabled = False

        daemon = NodeAvailabilityDaemon()

        assert "vast" in daemon._checkers
        assert "lambda" in daemon._checkers
        assert "runpod" not in daemon._checkers

    @patch("app.coordination.node_availability.daemon.VastChecker")
    @patch("app.coordination.node_availability.daemon.LambdaChecker")
    @patch("app.coordination.node_availability.daemon.RunPodChecker")
    def test_init_checkers_provider_disabled(
        self,
        mock_runpod: MagicMock,
        mock_lambda: MagicMock,
        mock_vast: MagicMock,
    ) -> None:
        """Test that disabled providers are not checked."""
        config = NodeAvailabilityConfig(
            vast_enabled=False,
            lambda_enabled=True,
            runpod_enabled=False,
        )

        mock_lambda.return_value.is_enabled = True

        daemon = NodeAvailabilityDaemon(config)

        # Vast and RunPod constructors should not be called
        mock_vast.assert_not_called()
        mock_runpod.assert_not_called()
        assert "lambda" in daemon._checkers


class TestNodeAvailabilityDaemonGracePeriod:
    """Tests for grace period logic."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_daemon_instance()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_daemon_instance()

    @patch("app.coordination.node_availability.daemon.VastChecker")
    @patch("app.coordination.node_availability.daemon.LambdaChecker")
    @patch("app.coordination.node_availability.daemon.RunPodChecker")
    def test_grace_period_first_detection(
        self,
        mock_runpod: MagicMock,
        mock_lambda: MagicMock,
        mock_vast: MagicMock,
    ) -> None:
        """Test first detection starts grace period."""
        mock_vast.return_value.is_enabled = False
        mock_lambda.return_value.is_enabled = False
        mock_runpod.return_value.is_enabled = False

        daemon = NodeAvailabilityDaemon()

        # First check should return False
        result = daemon._check_grace_period("test-node")
        assert result is False
        assert "test-node" in daemon._pending_terminations

    @patch("app.coordination.node_availability.daemon.VastChecker")
    @patch("app.coordination.node_availability.daemon.LambdaChecker")
    @patch("app.coordination.node_availability.daemon.RunPodChecker")
    def test_grace_period_not_elapsed(
        self,
        mock_runpod: MagicMock,
        mock_lambda: MagicMock,
        mock_vast: MagicMock,
    ) -> None:
        """Test grace period not elapsed returns False."""
        mock_vast.return_value.is_enabled = False
        mock_lambda.return_value.is_enabled = False
        mock_runpod.return_value.is_enabled = False

        config = NodeAvailabilityConfig(grace_period_seconds=60.0)
        daemon = NodeAvailabilityDaemon(config)

        # First check
        daemon._check_grace_period("test-node")

        # Immediate second check should return False
        result = daemon._check_grace_period("test-node")
        assert result is False

    @patch("app.coordination.node_availability.daemon.VastChecker")
    @patch("app.coordination.node_availability.daemon.LambdaChecker")
    @patch("app.coordination.node_availability.daemon.RunPodChecker")
    @patch("app.coordination.node_availability.daemon.time")
    def test_grace_period_elapsed(
        self,
        mock_time: MagicMock,
        mock_runpod: MagicMock,
        mock_lambda: MagicMock,
        mock_vast: MagicMock,
    ) -> None:
        """Test grace period elapsed returns True."""
        mock_vast.return_value.is_enabled = False
        mock_lambda.return_value.is_enabled = False
        mock_runpod.return_value.is_enabled = False

        config = NodeAvailabilityConfig(grace_period_seconds=60.0)
        daemon = NodeAvailabilityDaemon(config)

        # First check at t=0
        mock_time.time.return_value = 0.0
        daemon._check_grace_period("test-node")

        # Second check at t=70 (after grace period)
        mock_time.time.return_value = 70.0
        result = daemon._check_grace_period("test-node")

        assert result is True
        # Should be removed from pending
        assert "test-node" not in daemon._pending_terminations


class TestNodeAvailabilityDaemonHealthCheck:
    """Tests for health_check() method."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_daemon_instance()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_daemon_instance()

    @patch("app.coordination.node_availability.daemon.VastChecker")
    @patch("app.coordination.node_availability.daemon.LambdaChecker")
    @patch("app.coordination.node_availability.daemon.RunPodChecker")
    def test_health_check_no_checkers(
        self,
        mock_runpod: MagicMock,
        mock_lambda: MagicMock,
        mock_vast: MagicMock,
    ) -> None:
        """Test health check with no enabled checkers."""
        mock_vast.return_value.is_enabled = False
        mock_lambda.return_value.is_enabled = False
        mock_runpod.return_value.is_enabled = False

        daemon = NodeAvailabilityDaemon()
        result = daemon.health_check()

        assert isinstance(result, HealthCheckResult)
        assert result.healthy is False
        assert "No provider checkers enabled" in result.message

    @patch("app.coordination.node_availability.daemon.VastChecker")
    @patch("app.coordination.node_availability.daemon.LambdaChecker")
    @patch("app.coordination.node_availability.daemon.RunPodChecker")
    def test_health_check_with_checkers(
        self,
        mock_runpod: MagicMock,
        mock_lambda: MagicMock,
        mock_vast: MagicMock,
    ) -> None:
        """Test health check with enabled checkers."""
        mock_vast.return_value.is_enabled = True
        mock_lambda.return_value.is_enabled = False
        mock_runpod.return_value.is_enabled = False

        daemon = NodeAvailabilityDaemon()
        result = daemon.health_check()

        assert isinstance(result, HealthCheckResult)
        assert result.healthy is True
        assert result.message == "OK"
        assert "vast" in result.details["enabled_providers"]

    @patch("app.coordination.node_availability.daemon.VastChecker")
    @patch("app.coordination.node_availability.daemon.LambdaChecker")
    @patch("app.coordination.node_availability.daemon.RunPodChecker")
    def test_health_check_high_error_rate(
        self,
        mock_runpod: MagicMock,
        mock_lambda: MagicMock,
        mock_vast: MagicMock,
    ) -> None:
        """Test health check detects high error rate."""
        mock_vast.return_value.is_enabled = True
        mock_lambda.return_value.is_enabled = False
        mock_runpod.return_value.is_enabled = False

        daemon = NodeAvailabilityDaemon()

        # Simulate high error rate
        daemon._stats.provider_checks["vast"] = 10
        daemon._stats.provider_errors["vast"] = 6  # 60% error rate

        result = daemon.health_check()

        assert result.healthy is False
        assert "high error rate" in result.message


class TestNodeAvailabilityDaemonGetStatus:
    """Tests for get_status() method."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_daemon_instance()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_daemon_instance()

    @patch("app.coordination.node_availability.daemon.VastChecker")
    @patch("app.coordination.node_availability.daemon.LambdaChecker")
    @patch("app.coordination.node_availability.daemon.RunPodChecker")
    def test_get_status_structure(
        self,
        mock_runpod: MagicMock,
        mock_lambda: MagicMock,
        mock_vast: MagicMock,
    ) -> None:
        """Test get_status returns expected structure."""
        mock_vast.return_value.is_enabled = False
        mock_lambda.return_value.is_enabled = False
        mock_runpod.return_value.is_enabled = False

        daemon = NodeAvailabilityDaemon()
        status = daemon.get_status()

        assert "running" in status
        assert "config" in status
        assert "stats" in status
        assert "providers" in status
        assert "pending_terminations" in status

    @patch("app.coordination.node_availability.daemon.VastChecker")
    @patch("app.coordination.node_availability.daemon.LambdaChecker")
    @patch("app.coordination.node_availability.daemon.RunPodChecker")
    def test_get_status_config_values(
        self,
        mock_runpod: MagicMock,
        mock_lambda: MagicMock,
        mock_vast: MagicMock,
    ) -> None:
        """Test get_status returns correct config values."""
        mock_vast.return_value.is_enabled = False
        mock_lambda.return_value.is_enabled = False
        mock_runpod.return_value.is_enabled = False

        config = NodeAvailabilityConfig(
            dry_run=False,
            check_interval_seconds=120.0,
        )
        daemon = NodeAvailabilityDaemon(config)
        status = daemon.get_status()

        assert status["config"]["dry_run"] is False
        assert status["config"]["interval_seconds"] == 120.0


class TestSingletonPattern:
    """Tests for singleton pattern."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_daemon_instance()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_daemon_instance()

    @patch("app.coordination.node_availability.daemon.VastChecker")
    @patch("app.coordination.node_availability.daemon.LambdaChecker")
    @patch("app.coordination.node_availability.daemon.RunPodChecker")
    def test_get_node_availability_daemon_creates_singleton(
        self,
        mock_runpod: MagicMock,
        mock_lambda: MagicMock,
        mock_vast: MagicMock,
    ) -> None:
        """Test get_node_availability_daemon creates singleton."""
        mock_vast.return_value.is_enabled = False
        mock_lambda.return_value.is_enabled = False
        mock_runpod.return_value.is_enabled = False

        daemon1 = get_node_availability_daemon()
        daemon2 = get_node_availability_daemon()

        assert daemon1 is daemon2

    @patch("app.coordination.node_availability.daemon.VastChecker")
    @patch("app.coordination.node_availability.daemon.LambdaChecker")
    @patch("app.coordination.node_availability.daemon.RunPodChecker")
    def test_reset_daemon_instance_clears_singleton(
        self,
        mock_runpod: MagicMock,
        mock_lambda: MagicMock,
        mock_vast: MagicMock,
    ) -> None:
        """Test reset_daemon_instance clears singleton."""
        mock_vast.return_value.is_enabled = False
        mock_lambda.return_value.is_enabled = False
        mock_runpod.return_value.is_enabled = False

        daemon1 = get_node_availability_daemon()
        reset_daemon_instance()
        daemon2 = get_node_availability_daemon()

        assert daemon1 is not daemon2


class TestNodeAvailabilityDaemonCheckProvider:
    """Tests for _check_provider method."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_daemon_instance()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_daemon_instance()

    @patch("app.coordination.node_availability.daemon.VastChecker")
    @patch("app.coordination.node_availability.daemon.LambdaChecker")
    @patch("app.coordination.node_availability.daemon.RunPodChecker")
    @pytest.mark.asyncio
    async def test_check_provider_no_instances(
        self,
        mock_runpod: MagicMock,
        mock_lambda: MagicMock,
        mock_vast: MagicMock,
    ) -> None:
        """Test _check_provider with no instances returns empty updates."""
        mock_vast.return_value.is_enabled = False
        mock_lambda.return_value.is_enabled = False
        mock_runpod.return_value.is_enabled = False

        daemon = NodeAvailabilityDaemon()

        mock_checker = AsyncMock()
        mock_checker.get_instance_states = AsyncMock(return_value=[])
        mock_checker.get_terminated_instances = AsyncMock(return_value=[])

        updates = await daemon._check_provider(mock_checker, {})

        assert updates == {}

    @patch("app.coordination.node_availability.daemon.VastChecker")
    @patch("app.coordination.node_availability.daemon.LambdaChecker")
    @patch("app.coordination.node_availability.daemon.RunPodChecker")
    @pytest.mark.asyncio
    async def test_check_provider_detects_status_change(
        self,
        mock_runpod: MagicMock,
        mock_lambda: MagicMock,
        mock_vast: MagicMock,
    ) -> None:
        """Test _check_provider detects status changes."""
        mock_vast.return_value.is_enabled = False
        mock_lambda.return_value.is_enabled = False
        mock_runpod.return_value.is_enabled = False

        daemon = NodeAvailabilityDaemon()

        # Mock instance with state change
        from app.coordination.node_availability.state_checker import (
            InstanceInfo,
            ProviderInstanceState,
        )

        mock_instance = MagicMock(spec=InstanceInfo)
        mock_instance.node_name = "test-node"
        mock_instance.yaml_status = "offline"
        mock_instance.state = ProviderInstanceState.STOPPED

        mock_checker = AsyncMock()
        mock_checker.get_instance_states = AsyncMock(return_value=[mock_instance])
        mock_checker.correlate_with_config = MagicMock(return_value=[mock_instance])
        mock_checker.get_terminated_instances = AsyncMock(return_value=[])

        config_hosts = {
            "test-node": {"status": "ready"},
        }

        updates = await daemon._check_provider(mock_checker, config_hosts)

        assert updates.get("test-node") == "offline"


class TestNodeAvailabilityDaemonRunCycle:
    """Tests for _run_cycle method."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_daemon_instance()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_daemon_instance()

    @patch("app.coordination.node_availability.daemon.VastChecker")
    @patch("app.coordination.node_availability.daemon.LambdaChecker")
    @patch("app.coordination.node_availability.daemon.RunPodChecker")
    @pytest.mark.asyncio
    async def test_run_cycle_records_stats(
        self,
        mock_runpod: MagicMock,
        mock_lambda: MagicMock,
        mock_vast: MagicMock,
    ) -> None:
        """Test _run_cycle records cycle stats."""
        mock_vast.return_value.is_enabled = False
        mock_lambda.return_value.is_enabled = False
        mock_runpod.return_value.is_enabled = False

        daemon = NodeAvailabilityDaemon()

        # Mock config updater
        daemon._config_updater.load_config = MagicMock(return_value={"hosts": {}})

        await daemon._run_cycle()

        assert daemon._stats.cycles_completed == 1
        assert daemon._stats.last_cycle_time is not None

    @patch("app.coordination.node_availability.daemon.VastChecker")
    @patch("app.coordination.node_availability.daemon.LambdaChecker")
    @patch("app.coordination.node_availability.daemon.RunPodChecker")
    @pytest.mark.asyncio
    async def test_run_cycle_handles_provider_error(
        self,
        mock_runpod: MagicMock,
        mock_lambda: MagicMock,
        mock_vast: MagicMock,
    ) -> None:
        """Test _run_cycle handles provider errors gracefully."""
        mock_checker = AsyncMock()
        mock_checker.get_instance_states = AsyncMock(side_effect=Exception("API error"))

        mock_vast.return_value.is_enabled = True
        mock_vast.return_value = mock_checker
        mock_lambda.return_value.is_enabled = False
        mock_runpod.return_value.is_enabled = False

        daemon = NodeAvailabilityDaemon()
        daemon._checkers = {"vast": mock_checker}
        daemon._config_updater.load_config = MagicMock(return_value={"hosts": {}})

        # Should not raise
        await daemon._run_cycle()

        # Should record error
        assert daemon._stats.provider_errors.get("vast", 0) >= 1
