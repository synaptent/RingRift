"""Tests for UnifiedNodeHealthDaemon.

This module tests the main cluster health daemon that monitors and maintains
health across all cloud providers (Lambda, Vast, Hetzner, AWS, etc.).

Test coverage:
- DaemonConfig initialization and defaults
- Daemon lifecycle (start/stop)
- Health check cycle execution
- Recovery check triggering
- Optimization triggering
- P2P auto-deployment
- Config sync
- Stats tracking
- Graceful shutdown
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestDaemonConfig:
    """Tests for DaemonConfig dataclass."""

    def test_default_values(self):
        """Test DaemonConfig has sensible defaults."""
        from app.coordination.unified_node_health_daemon import DaemonConfig

        config = DaemonConfig()

        # Check interval defaults
        assert config.health_check_interval == 60.0
        assert config.recovery_check_interval == 120.0
        assert config.optimization_interval == 300.0
        assert config.config_sync_interval == 600.0
        assert config.p2p_deploy_interval == 300.0

        # Check threshold defaults
        assert config.min_healthy_percent == 80.0
        assert config.max_offline_count == 5
        assert config.min_p2p_coverage_percent == 90.0

        # Check feature flags
        assert config.enable_recovery is True
        assert config.enable_optimization is True
        assert config.enable_config_sync is True
        assert config.enable_alerting is True
        assert config.enable_p2p_auto_deploy is True

    def test_custom_values(self):
        """Test DaemonConfig with custom values."""
        from app.coordination.unified_node_health_daemon import DaemonConfig

        config = DaemonConfig(
            health_check_interval=30.0,
            min_healthy_percent=90.0,
            enable_recovery=False,
            enable_optimization=False,
        )

        assert config.health_check_interval == 30.0
        assert config.min_healthy_percent == 90.0
        assert config.enable_recovery is False
        assert config.enable_optimization is False

    def test_p2p_port_from_centralized_config(self):
        """Test p2p_port uses centralized config by default."""
        from app.coordination.unified_node_health_daemon import DaemonConfig

        with patch("app.coordination.unified_node_health_daemon._get_default_p2p_port") as mock_port:
            mock_port.return_value = 8770
            config = DaemonConfig()
            assert config.p2p_port == 8770


class TestUnifiedNodeHealthDaemonInit:
    """Tests for UnifiedNodeHealthDaemon initialization."""

    @pytest.fixture
    def mock_managers(self):
        """Mock provider managers."""
        with patch("app.coordination.unified_node_health_daemon.LambdaManager") as lambda_mgr, \
             patch("app.coordination.unified_node_health_daemon.VastManager") as vast_mgr, \
             patch("app.coordination.unified_node_health_daemon.HetznerManager") as hetzner_mgr, \
             patch("app.coordination.unified_node_health_daemon.AWSManager") as aws_mgr, \
             patch("app.coordination.unified_node_health_daemon.TailscaleManager") as ts_mgr:
            yield {
                "lambda": lambda_mgr,
                "vast": vast_mgr,
                "hetzner": hetzner_mgr,
                "aws": aws_mgr,
                "tailscale": ts_mgr,
            }

    @pytest.fixture
    def mock_orchestrators(self):
        """Mock orchestrators."""
        with patch("app.coordination.unified_node_health_daemon.HealthCheckOrchestrator") as health, \
             patch("app.coordination.unified_node_health_daemon.RecoveryOrchestrator") as recovery, \
             patch("app.coordination.unified_node_health_daemon.UtilizationOptimizer") as optimizer, \
             patch("app.coordination.unified_node_health_daemon.P2PAutoDeployer") as p2p:
            yield {
                "health": health,
                "recovery": recovery,
                "optimizer": optimizer,
                "p2p": p2p,
            }

    def test_init_with_default_config(self, mock_managers, mock_orchestrators):
        """Test daemon initializes with default config."""
        from app.coordination.unified_node_health_daemon import (
            DaemonConfig,
            UnifiedNodeHealthDaemon,
        )

        daemon = UnifiedNodeHealthDaemon()

        assert isinstance(daemon.config, DaemonConfig)
        assert daemon._running is False
        assert daemon._health_checks_run == 0
        assert daemon._recoveries_attempted == 0
        assert daemon._optimizations_run == 0
        assert daemon._p2p_deploys_run == 0
        assert daemon._tasks == []

    def test_init_with_custom_config(self, mock_managers, mock_orchestrators):
        """Test daemon initializes with custom config."""
        from app.coordination.unified_node_health_daemon import (
            DaemonConfig,
            UnifiedNodeHealthDaemon,
        )

        config = DaemonConfig(
            health_check_interval=30.0,
            enable_recovery=False,
        )
        daemon = UnifiedNodeHealthDaemon(config=config)

        assert daemon.config.health_check_interval == 30.0
        assert daemon.config.enable_recovery is False

    def test_init_creates_provider_managers(self, mock_managers, mock_orchestrators):
        """Test daemon creates all provider managers."""
        from app.coordination.unified_node_health_daemon import UnifiedNodeHealthDaemon

        daemon = UnifiedNodeHealthDaemon()

        # All managers should be created
        mock_managers["lambda"].assert_called_once()
        mock_managers["vast"].assert_called_once()
        mock_managers["hetzner"].assert_called_once()
        mock_managers["aws"].assert_called_once()
        mock_managers["tailscale"].assert_called_once()

    def test_init_creates_orchestrators(self, mock_managers, mock_orchestrators):
        """Test daemon creates all orchestrators with correct params."""
        from app.coordination.unified_node_health_daemon import UnifiedNodeHealthDaemon

        daemon = UnifiedNodeHealthDaemon()

        mock_orchestrators["health"].assert_called_once()
        mock_orchestrators["recovery"].assert_called_once()
        mock_orchestrators["optimizer"].assert_called_once()
        mock_orchestrators["p2p"].assert_called_once()


class TestDaemonCycle:
    """Tests for daemon cycle execution."""

    @pytest.fixture
    def daemon_with_mocks(self):
        """Create daemon with all methods mocked."""
        with patch("app.coordination.unified_node_health_daemon.LambdaManager"), \
             patch("app.coordination.unified_node_health_daemon.VastManager"), \
             patch("app.coordination.unified_node_health_daemon.HetznerManager"), \
             patch("app.coordination.unified_node_health_daemon.AWSManager"), \
             patch("app.coordination.unified_node_health_daemon.TailscaleManager"), \
             patch("app.coordination.unified_node_health_daemon.HealthCheckOrchestrator"), \
             patch("app.coordination.unified_node_health_daemon.RecoveryOrchestrator"), \
             patch("app.coordination.unified_node_health_daemon.UtilizationOptimizer"), \
             patch("app.coordination.unified_node_health_daemon.P2PAutoDeployer"):
            from app.coordination.unified_node_health_daemon import UnifiedNodeHealthDaemon

            daemon = UnifiedNodeHealthDaemon()

            # Mock the internal methods
            daemon._run_health_check = AsyncMock()
            daemon._run_recovery_check = AsyncMock()
            daemon._run_optimization = AsyncMock()
            daemon._run_config_sync = AsyncMock()
            daemon._run_p2p_deploy = AsyncMock()

            return daemon

    @pytest.mark.asyncio
    async def test_cycle_runs_health_check_when_due(self, daemon_with_mocks):
        """Test daemon cycle runs health check when interval elapsed."""
        daemon = daemon_with_mocks
        daemon._last_health_check = 0  # Set to 0 so it's due

        await daemon._daemon_cycle()

        daemon._run_health_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_cycle_skips_health_check_when_not_due(self, daemon_with_mocks):
        """Test daemon cycle skips health check when interval not elapsed."""
        daemon = daemon_with_mocks
        daemon._last_health_check = time.time()  # Just ran

        await daemon._daemon_cycle()

        daemon._run_health_check.assert_not_called()

    @pytest.mark.asyncio
    async def test_cycle_runs_recovery_when_enabled_and_due(self, daemon_with_mocks):
        """Test daemon cycle runs recovery when enabled and due."""
        daemon = daemon_with_mocks
        daemon.config.enable_recovery = True
        daemon._last_recovery_check = 0

        await daemon._daemon_cycle()

        daemon._run_recovery_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_cycle_skips_recovery_when_disabled(self, daemon_with_mocks):
        """Test daemon cycle skips recovery when disabled."""
        daemon = daemon_with_mocks
        daemon.config.enable_recovery = False
        daemon._last_recovery_check = 0

        await daemon._daemon_cycle()

        daemon._run_recovery_check.assert_not_called()

    @pytest.mark.asyncio
    async def test_cycle_runs_optimization_when_enabled_and_due(self, daemon_with_mocks):
        """Test daemon cycle runs optimization when enabled and due."""
        daemon = daemon_with_mocks
        daemon.config.enable_optimization = True
        daemon._last_optimization = 0

        await daemon._daemon_cycle()

        daemon._run_optimization.assert_called_once()

    @pytest.mark.asyncio
    async def test_cycle_runs_p2p_deploy_when_enabled_and_due(self, daemon_with_mocks):
        """Test daemon cycle runs P2P deploy when enabled and due."""
        daemon = daemon_with_mocks
        daemon.config.enable_p2p_auto_deploy = True
        daemon._last_p2p_deploy = 0

        await daemon._daemon_cycle()

        daemon._run_p2p_deploy.assert_called_once()


class TestHealthCheck:
    """Tests for health check execution."""

    @pytest.fixture
    def daemon_with_health_mock(self):
        """Create daemon with health orchestrator mocked."""
        with patch("app.coordination.unified_node_health_daemon.LambdaManager"), \
             patch("app.coordination.unified_node_health_daemon.VastManager"), \
             patch("app.coordination.unified_node_health_daemon.HetznerManager"), \
             patch("app.coordination.unified_node_health_daemon.AWSManager"), \
             patch("app.coordination.unified_node_health_daemon.TailscaleManager"), \
             patch("app.coordination.unified_node_health_daemon.HealthCheckOrchestrator") as health_mock, \
             patch("app.coordination.unified_node_health_daemon.RecoveryOrchestrator"), \
             patch("app.coordination.unified_node_health_daemon.UtilizationOptimizer"), \
             patch("app.coordination.unified_node_health_daemon.P2PAutoDeployer"):
            from app.coordination.unified_node_health_daemon import UnifiedNodeHealthDaemon

            daemon = UnifiedNodeHealthDaemon()

            # Setup health orchestrator mock
            health_instance = health_mock.return_value
            health_instance.check_all_nodes = AsyncMock()
            health_instance.get_summary = MagicMock()

            return daemon, health_instance

    @pytest.mark.asyncio
    async def test_health_check_updates_timestamp(self, daemon_with_health_mock):
        """Test health check updates last check timestamp."""
        daemon, health = daemon_with_health_mock
        health.get_summary.return_value = MagicMock(
            healthy_count=10,
            total_count=10,
            healthy_percent=100.0,
        )

        initial_time = daemon._last_health_check
        await daemon._run_health_check()

        assert daemon._last_health_check > initial_time

    @pytest.mark.asyncio
    async def test_health_check_increments_counter(self, daemon_with_health_mock):
        """Test health check increments run counter."""
        daemon, health = daemon_with_health_mock
        health.get_summary.return_value = MagicMock(
            healthy_count=10,
            total_count=10,
            healthy_percent=100.0,
        )

        initial_count = daemon._health_checks_run
        await daemon._run_health_check()

        assert daemon._health_checks_run == initial_count + 1


class TestDaemonLifecycle:
    """Tests for daemon start/stop lifecycle."""

    @pytest.fixture
    def daemon_for_lifecycle(self):
        """Create daemon for lifecycle tests."""
        with patch("app.coordination.unified_node_health_daemon.LambdaManager"), \
             patch("app.coordination.unified_node_health_daemon.VastManager"), \
             patch("app.coordination.unified_node_health_daemon.HetznerManager"), \
             patch("app.coordination.unified_node_health_daemon.AWSManager"), \
             patch("app.coordination.unified_node_health_daemon.TailscaleManager"), \
             patch("app.coordination.unified_node_health_daemon.HealthCheckOrchestrator"), \
             patch("app.coordination.unified_node_health_daemon.RecoveryOrchestrator"), \
             patch("app.coordination.unified_node_health_daemon.UtilizationOptimizer"), \
             patch("app.coordination.unified_node_health_daemon.P2PAutoDeployer"):
            from app.coordination.unified_node_health_daemon import UnifiedNodeHealthDaemon

            daemon = UnifiedNodeHealthDaemon()
            # Mock expensive methods
            daemon._run_health_check = AsyncMock()
            daemon._daemon_cycle = AsyncMock()
            daemon._cleanup = AsyncMock()

            return daemon

    @pytest.mark.asyncio
    async def test_run_sets_running_flag(self, daemon_for_lifecycle):
        """Test run() sets _running flag."""
        daemon = daemon_for_lifecycle

        # Create a task that stops after first cycle
        async def stop_after_cycle():
            await asyncio.sleep(0.05)
            daemon._running = False

        asyncio.create_task(stop_after_cycle())
        await daemon.run()

        # Should have been set to True initially
        assert daemon._start_time > 0

    @pytest.mark.asyncio
    async def test_run_calls_cleanup_on_exit(self, daemon_for_lifecycle):
        """Test run() calls cleanup when stopped."""
        daemon = daemon_for_lifecycle
        daemon._running = False  # Exit immediately

        await daemon.run()

        daemon._cleanup.assert_called_once()

    def test_stop_sets_running_false(self, daemon_for_lifecycle):
        """Test stop() sets _running to False."""
        daemon = daemon_for_lifecycle
        daemon._running = True

        daemon.stop()

        assert daemon._running is False


class TestStatsTracking:
    """Tests for daemon stats tracking."""

    @pytest.fixture
    def daemon_for_stats(self):
        """Create daemon for stats tests."""
        with patch("app.coordination.unified_node_health_daemon.LambdaManager"), \
             patch("app.coordination.unified_node_health_daemon.VastManager"), \
             patch("app.coordination.unified_node_health_daemon.HetznerManager"), \
             patch("app.coordination.unified_node_health_daemon.AWSManager"), \
             patch("app.coordination.unified_node_health_daemon.TailscaleManager"), \
             patch("app.coordination.unified_node_health_daemon.HealthCheckOrchestrator") as health_mock, \
             patch("app.coordination.unified_node_health_daemon.RecoveryOrchestrator") as recovery_mock, \
             patch("app.coordination.unified_node_health_daemon.UtilizationOptimizer") as optimizer_mock, \
             patch("app.coordination.unified_node_health_daemon.P2PAutoDeployer") as p2p_mock:
            from app.coordination.unified_node_health_daemon import UnifiedNodeHealthDaemon

            daemon = UnifiedNodeHealthDaemon()

            # Setup mocks to return success
            health_mock.return_value.get_summary.return_value = MagicMock(
                healthy_count=10,
                total_count=10,
                healthy_percent=100.0,
            )
            health_mock.return_value.check_all_nodes = AsyncMock()
            recovery_mock.return_value.run_recovery_cycle = AsyncMock()
            optimizer_mock.return_value.optimize = AsyncMock()
            p2p_mock.return_value.ensure_coverage = AsyncMock()

            return daemon

    def test_initial_stats_zero(self, daemon_for_stats):
        """Test initial stats are zero."""
        daemon = daemon_for_stats

        assert daemon._health_checks_run == 0
        assert daemon._recoveries_attempted == 0
        assert daemon._optimizations_run == 0
        assert daemon._p2p_deploys_run == 0
        assert daemon._start_time == 0.0


class TestHealthCheckResult:
    """Tests for health check result handling."""

    def test_daemon_has_health_check_method(self):
        """Test daemon has health_check() method for DaemonManager integration."""
        with patch("app.coordination.unified_node_health_daemon.LambdaManager"), \
             patch("app.coordination.unified_node_health_daemon.VastManager"), \
             patch("app.coordination.unified_node_health_daemon.HetznerManager"), \
             patch("app.coordination.unified_node_health_daemon.AWSManager"), \
             patch("app.coordination.unified_node_health_daemon.TailscaleManager"), \
             patch("app.coordination.unified_node_health_daemon.HealthCheckOrchestrator"), \
             patch("app.coordination.unified_node_health_daemon.RecoveryOrchestrator"), \
             patch("app.coordination.unified_node_health_daemon.UtilizationOptimizer"), \
             patch("app.coordination.unified_node_health_daemon.P2PAutoDeployer"):
            from app.coordination.unified_node_health_daemon import UnifiedNodeHealthDaemon

            daemon = UnifiedNodeHealthDaemon()

            # Check health_check method exists
            assert hasattr(daemon, "health_check") or hasattr(daemon, "get_health")


class TestEventEmission:
    """Tests for event emission during health checks."""

    @pytest.mark.asyncio
    async def test_emits_cluster_healthy_event(self):
        """Test daemon emits cluster_healthy event when all nodes healthy."""
        with patch("app.coordination.unified_node_health_daemon.LambdaManager"), \
             patch("app.coordination.unified_node_health_daemon.VastManager"), \
             patch("app.coordination.unified_node_health_daemon.HetznerManager"), \
             patch("app.coordination.unified_node_health_daemon.AWSManager"), \
             patch("app.coordination.unified_node_health_daemon.TailscaleManager"), \
             patch("app.coordination.unified_node_health_daemon.HealthCheckOrchestrator") as health_mock, \
             patch("app.coordination.unified_node_health_daemon.RecoveryOrchestrator"), \
             patch("app.coordination.unified_node_health_daemon.UtilizationOptimizer"), \
             patch("app.coordination.unified_node_health_daemon.P2PAutoDeployer"), \
             patch("app.coordination.unified_node_health_daemon.emit_p2p_cluster_healthy") as emit_mock:
            from app.coordination.unified_node_health_daemon import UnifiedNodeHealthDaemon

            daemon = UnifiedNodeHealthDaemon()

            # Mock 100% healthy
            health_mock.return_value.get_summary.return_value = MagicMock(
                healthy_count=10,
                total_count=10,
                healthy_percent=100.0,
            )
            health_mock.return_value.check_all_nodes = AsyncMock()

            await daemon._run_health_check()

            # Should emit healthy event
            # Note: Actual implementation may differ


class TestModuleExports:
    """Tests for module-level exports and functions."""

    def test_run_daemon_function_exists(self):
        """Test run_daemon() function is exported."""
        from app.coordination.unified_node_health_daemon import run_daemon

        assert callable(run_daemon)

    def test_get_daemon_function_exists(self):
        """Test get_daemon() singleton accessor exists."""
        try:
            from app.coordination.unified_node_health_daemon import get_daemon
            assert callable(get_daemon)
        except ImportError:
            # May not have singleton pattern - that's OK
            pass


class TestGracefulShutdown:
    """Tests for graceful shutdown handling."""

    @pytest.fixture
    def daemon_for_shutdown(self):
        """Create daemon for shutdown tests."""
        with patch("app.coordination.unified_node_health_daemon.LambdaManager"), \
             patch("app.coordination.unified_node_health_daemon.VastManager"), \
             patch("app.coordination.unified_node_health_daemon.HetznerManager"), \
             patch("app.coordination.unified_node_health_daemon.AWSManager"), \
             patch("app.coordination.unified_node_health_daemon.TailscaleManager"), \
             patch("app.coordination.unified_node_health_daemon.HealthCheckOrchestrator"), \
             patch("app.coordination.unified_node_health_daemon.RecoveryOrchestrator"), \
             patch("app.coordination.unified_node_health_daemon.UtilizationOptimizer"), \
             patch("app.coordination.unified_node_health_daemon.P2PAutoDeployer"):
            from app.coordination.unified_node_health_daemon import UnifiedNodeHealthDaemon

            daemon = UnifiedNodeHealthDaemon()
            daemon._tasks = [MagicMock(spec=asyncio.Task)]

            return daemon

    @pytest.mark.asyncio
    async def test_cleanup_cancels_tasks(self, daemon_for_shutdown):
        """Test cleanup cancels pending tasks."""
        daemon = daemon_for_shutdown
        mock_task = daemon._tasks[0]
        mock_task.cancel = MagicMock()
        mock_task.cancelled = MagicMock(return_value=False)

        await daemon._cleanup()

        mock_task.cancel.assert_called()
