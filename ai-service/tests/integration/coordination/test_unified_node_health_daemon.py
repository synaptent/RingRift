"""Unit and integration tests for UnifiedNodeHealthDaemon.

Tests verify:
1. Initialization and configuration
2. Main loop cycling with all daemon paths
3. Health check execution
4. Recovery check execution
5. Optimization execution
6. Config sync execution
7. P2P auto-deployment
8. Event emission for cluster health
9. CoordinatorProtocol health_check compliance
10. Signal handling and graceful shutdown

Created: December 27, 2025
Purpose: Ensure critical cluster health daemon operates correctly
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_hosts_config():
    """Create temporary hosts config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config = {
            'hosts': {
                'test-node-1': {
                    'tailscale_ip': '100.1.2.3',
                    'status': 'ready',
                    'role': 'gpu',
                },
                'test-node-2': {
                    'tailscale_ip': '100.1.2.4',
                    'status': 'ready',
                    'role': 'gpu',
                },
            }
        }
        yaml.dump(config, f)
        f.flush()
        yield Path(f.name)
    os.unlink(f.name)


def create_mock_daemon(custom_config=None):
    """Create a daemon with all dependencies mocked via attribute injection."""
    from app.coordination.unified_node_health_daemon import DaemonConfig

    # Create mock orchestrators
    mock_health_orchestrator = MagicMock()
    mock_health_orchestrator.run_full_health_check = AsyncMock()
    mock_health_orchestrator.get_cluster_health = AsyncMock()
    mock_health_orchestrator.get_node_health = MagicMock(return_value=None)
    mock_health_orchestrator.stop = AsyncMock()

    mock_recovery_orchestrator = MagicMock()
    mock_recovery_orchestrator.recover_all_unhealthy = AsyncMock(return_value=[])
    mock_recovery_orchestrator.slack_webhook_url = None

    mock_utilization_optimizer = MagicMock()
    mock_utilization_optimizer.optimize_cluster = AsyncMock(return_value=[])

    mock_p2p_deployer = MagicMock()
    mock_p2p_deployer.check_and_deploy = AsyncMock()
    mock_p2p_deployer.get_latest_coverage = MagicMock(return_value=None)

    # Create daemon with patched __init__ to skip real initialization
    with patch('app.coordination.unified_node_health_daemon.HealthCheckOrchestrator'), \
         patch('app.coordination.unified_node_health_daemon.RecoveryOrchestrator'), \
         patch('app.coordination.unified_node_health_daemon.UtilizationOptimizer'), \
         patch('app.coordination.unified_node_health_daemon.P2PAutoDeployer'), \
         patch('app.coordination.unified_node_health_daemon.LambdaManager'), \
         patch('app.coordination.unified_node_health_daemon.VastManager'), \
         patch('app.coordination.unified_node_health_daemon.HetznerManager'), \
         patch('app.coordination.unified_node_health_daemon.AWSManager'), \
         patch('app.coordination.unified_node_health_daemon.TailscaleManager'):

        from app.coordination.unified_node_health_daemon import UnifiedNodeHealthDaemon

        config = custom_config if custom_config else DaemonConfig()
        daemon = UnifiedNodeHealthDaemon(config=config)

        # Inject mocks
        daemon.health_orchestrator = mock_health_orchestrator
        daemon.recovery_orchestrator = mock_recovery_orchestrator
        daemon.utilization_optimizer = mock_utilization_optimizer
        daemon.p2p_auto_deployer = mock_p2p_deployer

        # Also mock provider managers
        daemon.lambda_mgr = MagicMock()
        daemon.lambda_mgr.close = AsyncMock()
        daemon.vast_mgr = MagicMock()
        daemon.hetzner_mgr = MagicMock()
        daemon.aws_mgr = MagicMock()
        daemon.tailscale_mgr = MagicMock()

        return daemon, {
            'health_orchestrator': mock_health_orchestrator,
            'recovery_orchestrator': mock_recovery_orchestrator,
            'utilization_optimizer': mock_utilization_optimizer,
            'p2p_deployer': mock_p2p_deployer,
        }


# =============================================================================
# Initialization Tests
# =============================================================================


class TestDaemonConfig:
    """Tests for DaemonConfig dataclass."""

    def test_default_config_values(self):
        """Default config should have sensible values."""
        from app.coordination.unified_node_health_daemon import DaemonConfig

        config = DaemonConfig()

        assert config.health_check_interval == 60.0
        assert config.recovery_check_interval == 120.0
        assert config.optimization_interval == 300.0
        assert config.config_sync_interval == 600.0
        assert config.p2p_deploy_interval == 300.0
        assert config.min_healthy_percent == 80.0
        assert config.enable_recovery is True
        assert config.enable_optimization is True
        assert config.enable_config_sync is True
        assert config.enable_alerting is True
        assert config.enable_p2p_auto_deploy is True

    def test_custom_config_values(self):
        """Custom config should override defaults."""
        from app.coordination.unified_node_health_daemon import DaemonConfig

        config = DaemonConfig(
            health_check_interval=30.0,
            enable_recovery=False,
            enable_optimization=False,
            min_healthy_percent=90.0,
        )

        assert config.health_check_interval == 30.0
        assert config.enable_recovery is False
        assert config.enable_optimization is False
        assert config.min_healthy_percent == 90.0

    def test_p2p_port_uses_centralized_config(self):
        """P2P port should use centralized config default."""
        from app.coordination.unified_node_health_daemon import DaemonConfig

        with patch('app.coordination.unified_node_health_daemon._get_default_p2p_port', return_value=8770):
            config = DaemonConfig()
            assert config.p2p_port == 8770


class TestDaemonInitialization:
    """Tests for UnifiedNodeHealthDaemon initialization."""

    def test_initialization_with_default_config(self):
        """Daemon should initialize with default config."""
        daemon, _ = create_mock_daemon()

        from app.coordination.unified_node_health_daemon import DaemonConfig

        assert daemon.config is not None
        assert isinstance(daemon.config, DaemonConfig)
        assert daemon._running is False
        assert daemon._health_checks_run == 0
        assert daemon._recoveries_attempted == 0
        assert daemon._optimizations_run == 0

    def test_initialization_with_custom_config(self):
        """Daemon should accept custom config."""
        from app.coordination.unified_node_health_daemon import DaemonConfig

        custom_config = DaemonConfig(
            health_check_interval=30.0,
            enable_recovery=False,
        )

        daemon, _ = create_mock_daemon(custom_config)

        assert daemon.config.health_check_interval == 30.0
        assert daemon.config.enable_recovery is False


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheckCycle:
    """Tests for health check cycle execution."""

    @pytest.mark.asyncio
    async def test_run_health_check_calls_orchestrator(self):
        """Health check should call orchestrator methods."""
        daemon, mocks = create_mock_daemon()
        mock_health = mocks['health_orchestrator']

        # Create mock cluster health summary
        mock_summary = MagicMock()
        mock_summary.total_nodes = 10
        mock_summary.healthy = 8
        mock_summary.degraded = 1
        mock_summary.unhealthy = 1
        mock_summary.offline = 0
        mock_summary.retired = 0
        mock_health.get_cluster_health.return_value = mock_summary

        daemon.config.enable_alerting = False

        await daemon._run_health_check()

        mock_health.run_full_health_check.assert_called_once()
        mock_health.get_cluster_health.assert_called_once()
        assert daemon._health_checks_run == 1
        assert daemon._last_health_check > 0

    @pytest.mark.asyncio
    async def test_health_check_handles_orchestrator_error(self):
        """Health check should handle orchestrator errors gracefully."""
        daemon, mocks = create_mock_daemon()
        mock_health = mocks['health_orchestrator']

        mock_health.run_full_health_check.side_effect = RuntimeError("Network error")
        daemon.config.enable_alerting = False

        # Should not raise, should log error
        await daemon._run_health_check()

        # Still updates timestamp and count
        assert daemon._health_checks_run == 1


# =============================================================================
# Recovery Check Tests
# =============================================================================


class TestRecoveryCheck:
    """Tests for recovery check execution."""

    @pytest.mark.asyncio
    async def test_run_recovery_check_calls_orchestrator(self):
        """Recovery check should call recovery orchestrator."""
        daemon, mocks = create_mock_daemon()
        mock_recovery = mocks['recovery_orchestrator']

        # Mock successful recovery
        mock_result = MagicMock()
        mock_result.success = True
        mock_recovery.recover_all_unhealthy.return_value = [mock_result]

        await daemon._run_recovery_check()

        mock_recovery.recover_all_unhealthy.assert_called_once()
        assert daemon._recoveries_attempted == 1
        assert daemon._last_recovery_check > 0

    @pytest.mark.asyncio
    async def test_recovery_check_tracks_multiple_recoveries(self):
        """Recovery check should count all recovery attempts."""
        daemon, mocks = create_mock_daemon()
        mock_recovery = mocks['recovery_orchestrator']

        # Mock multiple recovery results
        results = [MagicMock(success=True), MagicMock(success=False), MagicMock(success=True)]
        mock_recovery.recover_all_unhealthy.return_value = results

        await daemon._run_recovery_check()

        assert daemon._recoveries_attempted == 3

    @pytest.mark.asyncio
    async def test_recovery_check_handles_error(self):
        """Recovery check should handle errors gracefully."""
        daemon, mocks = create_mock_daemon()
        mock_recovery = mocks['recovery_orchestrator']

        mock_recovery.recover_all_unhealthy.side_effect = RuntimeError("API error")

        # Should not raise
        await daemon._run_recovery_check()

        assert daemon._last_recovery_check > 0


# =============================================================================
# Optimization Tests
# =============================================================================


class TestOptimization:
    """Tests for utilization optimization."""

    @pytest.mark.asyncio
    async def test_run_optimization_calls_optimizer(self):
        """Optimization should call utilization optimizer."""
        daemon, mocks = create_mock_daemon()
        mock_optimizer = mocks['utilization_optimizer']

        await daemon._run_optimization()

        mock_optimizer.optimize_cluster.assert_called_once()
        assert daemon._optimizations_run == 1
        assert daemon._last_optimization > 0

    @pytest.mark.asyncio
    async def test_optimization_handles_error(self):
        """Optimization should handle errors gracefully."""
        daemon, mocks = create_mock_daemon()
        mock_optimizer = mocks['utilization_optimizer']

        mock_optimizer.optimize_cluster.side_effect = RuntimeError("Resource error")

        # Should not raise
        await daemon._run_optimization()

        assert daemon._last_optimization > 0


# =============================================================================
# Config Sync Tests
# =============================================================================


class TestConfigSync:
    """Tests for config file synchronization."""

    @pytest.mark.asyncio
    async def test_run_config_sync_updates_timestamp(self, temp_hosts_config):
        """Config sync should update timestamp."""
        from app.coordination.unified_node_health_daemon import DaemonConfig

        config = DaemonConfig(hosts_config_path=str(temp_hosts_config))
        daemon, _ = create_mock_daemon(config)

        await daemon._run_config_sync()

        assert daemon._last_config_sync > 0

    @pytest.mark.asyncio
    async def test_config_sync_handles_missing_file(self):
        """Config sync should handle missing config file."""
        from app.coordination.unified_node_health_daemon import DaemonConfig

        config = DaemonConfig(hosts_config_path="/nonexistent/path.yaml")
        daemon, _ = create_mock_daemon(config)

        # Should not raise
        await daemon._run_config_sync()


# =============================================================================
# P2P Auto-Deploy Tests
# =============================================================================


class TestP2PAutoDeploy:
    """Tests for P2P auto-deployment."""

    @pytest.mark.asyncio
    async def test_run_p2p_deploy_calls_deployer(self):
        """P2P deploy should call auto-deployer."""
        daemon, mocks = create_mock_daemon()
        mock_deployer = mocks['p2p_deployer']

        # Mock coverage report
        mock_report = MagicMock()
        mock_report.coverage_percent = 95.0
        mock_report.nodes_with_p2p = 19
        mock_report.total_nodes = 20
        mock_report.nodes_needing_deployment = ['test-node']
        mock_deployer.check_and_deploy.return_value = mock_report

        daemon.config.enable_alerting = False

        await daemon._run_p2p_deploy()

        mock_deployer.check_and_deploy.assert_called_once()
        assert daemon._p2p_deploys_run == 1
        assert daemon._last_p2p_deploy > 0

    @pytest.mark.asyncio
    async def test_p2p_deploy_handles_error(self):
        """P2P deploy should handle errors gracefully."""
        daemon, mocks = create_mock_daemon()
        mock_deployer = mocks['p2p_deployer']

        mock_deployer.check_and_deploy.side_effect = RuntimeError("SSH error")

        # Should not raise
        await daemon._run_p2p_deploy()

        assert daemon._last_p2p_deploy > 0


# =============================================================================
# Event Emission Tests
# =============================================================================


class TestEventEmission:
    """Tests for cluster health event emission."""

    @pytest.mark.asyncio
    async def test_emits_cluster_healthy_event_above_threshold(self):
        """Should emit cluster healthy event when above threshold."""
        with patch('app.coordination.unified_node_health_daemon.emit_p2p_cluster_healthy') as mock_emit:
            mock_emit.return_value = None

            daemon, _ = create_mock_daemon()

            # 90% healthy is above 80% threshold
            mock_summary = MagicMock()
            mock_summary.total_nodes = 10
            mock_summary.healthy = 8
            mock_summary.degraded = 1
            mock_summary.unhealthy = 1
            mock_summary.offline = 0
            mock_summary.retired = 0

            await daemon._emit_cluster_health_events(
                healthy_percent=90.0,
                healthy_nodes=9,
                active_nodes=10,
                summary=mock_summary,
            )

            mock_emit.assert_called_once()

    @pytest.mark.asyncio
    async def test_emits_cluster_unhealthy_event_below_threshold(self):
        """Should emit cluster unhealthy event when below threshold."""
        with patch('app.coordination.unified_node_health_daemon.emit_p2p_cluster_unhealthy') as mock_emit:
            mock_emit.return_value = None

            daemon, _ = create_mock_daemon()

            # 50% healthy is below 80% threshold
            mock_summary = MagicMock()
            mock_summary.total_nodes = 10
            mock_summary.healthy = 3
            mock_summary.degraded = 2
            mock_summary.unhealthy = 3
            mock_summary.offline = 2
            mock_summary.retired = 0

            await daemon._emit_cluster_health_events(
                healthy_percent=50.0,
                healthy_nodes=5,
                active_nodes=10,
                summary=mock_summary,
            )

            mock_emit.assert_called_once()


# =============================================================================
# Health Check Protocol Tests
# =============================================================================


class TestHealthCheckProtocol:
    """Tests for CoordinatorProtocol health_check compliance."""

    def test_health_check_when_not_running(self):
        """health_check should return stopped status when not running."""
        daemon, _ = create_mock_daemon()

        result = daemon.health_check()

        assert result.healthy is False
        assert result.status.value == "stopped"
        assert "not running" in result.message

    def test_health_check_when_running_healthy(self):
        """health_check should return healthy when daemon is running normally."""
        daemon, _ = create_mock_daemon()
        daemon._running = True
        daemon._last_health_check = time.time()  # Recent health check
        daemon._health_checks_run = 5

        result = daemon.health_check()

        assert result.healthy is True
        assert result.status.value == "running"
        assert "5 checks" in result.message

    def test_health_check_degraded_when_stale(self):
        """health_check should return degraded when checks are stale."""
        from app.coordination.unified_node_health_daemon import DaemonConfig

        config = DaemonConfig(health_check_interval=60.0)
        daemon, _ = create_mock_daemon(config)
        daemon._running = True
        # Set last health check to 4 minutes ago (> 3x interval of 60s)
        daemon._last_health_check = time.time() - 240

        result = daemon.health_check()

        assert result.healthy is False
        assert result.status.value == "degraded"
        assert "stale" in result.message


# =============================================================================
# Daemon Cycle Tests
# =============================================================================


class TestDaemonCycle:
    """Tests for daemon main loop cycle."""

    @pytest.mark.asyncio
    async def test_daemon_cycle_respects_intervals(self):
        """Daemon cycle should only run tasks when intervals elapsed."""
        from app.coordination.unified_node_health_daemon import DaemonConfig

        config = DaemonConfig(
            health_check_interval=1000.0,  # Very long interval
            recovery_check_interval=1000.0,
            optimization_interval=1000.0,
            config_sync_interval=1000.0,
            p2p_deploy_interval=1000.0,
            enable_alerting=False,
        )
        daemon, mocks = create_mock_daemon(config)

        # Set recent timestamps so intervals haven't elapsed
        now = time.time()
        daemon._last_health_check = now
        daemon._last_recovery_check = now
        daemon._last_optimization = now
        daemon._last_config_sync = now
        daemon._last_p2p_deploy = now

        await daemon._daemon_cycle()

        # Nothing should have run since intervals haven't elapsed
        mocks['health_orchestrator'].run_full_health_check.assert_not_called()
        mocks['recovery_orchestrator'].recover_all_unhealthy.assert_not_called()
        mocks['utilization_optimizer'].optimize_cluster.assert_not_called()

    @pytest.mark.asyncio
    async def test_daemon_cycle_runs_health_check_when_due(self):
        """Daemon cycle should run health check when interval elapsed."""
        from app.coordination.unified_node_health_daemon import DaemonConfig

        # Mock cluster health summary for alerting
        mock_summary = MagicMock()
        mock_summary.total_nodes = 10
        mock_summary.healthy = 10
        mock_summary.degraded = 0
        mock_summary.unhealthy = 0
        mock_summary.offline = 0
        mock_summary.retired = 0

        config = DaemonConfig(
            health_check_interval=0.1,  # Very short interval
            recovery_check_interval=1000.0,
            optimization_interval=1000.0,
            config_sync_interval=1000.0,
            p2p_deploy_interval=1000.0,
            enable_recovery=False,
            enable_optimization=False,
            enable_config_sync=False,
            enable_p2p_auto_deploy=False,
            enable_alerting=False,
        )
        daemon, mocks = create_mock_daemon(config)
        mocks['health_orchestrator'].get_cluster_health.return_value = mock_summary

        # Set old timestamp so interval has elapsed
        daemon._last_health_check = time.time() - 1.0

        await daemon._daemon_cycle()

        mocks['health_orchestrator'].run_full_health_check.assert_called_once()


# =============================================================================
# Stats Tests
# =============================================================================


class TestDaemonStats:
    """Tests for daemon statistics."""

    def test_get_stats_returns_all_fields(self):
        """get_stats should return all expected fields."""
        daemon, _ = create_mock_daemon()
        daemon._start_time = time.time() - 100
        daemon._health_checks_run = 5
        daemon._recoveries_attempted = 3
        daemon._optimizations_run = 2
        daemon._p2p_deploys_run = 1
        daemon._running = True

        stats = daemon.get_stats()

        assert 'uptime_seconds' in stats
        assert stats['uptime_seconds'] >= 100
        assert stats['health_checks_run'] == 5
        assert stats['recoveries_attempted'] == 3
        assert stats['optimizations_run'] == 2
        assert stats['p2p_deploys_run'] == 1
        assert stats['running'] is True

    def test_get_stats_includes_p2p_coverage(self):
        """get_stats should include P2P coverage when available."""
        daemon, mocks = create_mock_daemon()

        # Mock P2P coverage report
        mock_report = MagicMock()
        mock_report.coverage_percent = 95.0
        mock_report.nodes_with_p2p = 19
        mock_report.total_nodes = 20
        mock_report.nodes_needing_deployment = ['test-node']
        mocks['p2p_deployer'].get_latest_coverage.return_value = mock_report

        stats = daemon.get_stats()

        assert stats['p2p_coverage'] is not None
        assert stats['p2p_coverage']['coverage_percent'] == 95.0
        assert stats['p2p_coverage']['nodes_with_p2p'] == 19


# =============================================================================
# Shutdown Tests
# =============================================================================


class TestShutdown:
    """Tests for daemon shutdown."""

    def test_stop_sets_running_false(self):
        """stop() should set _running to False."""
        daemon, _ = create_mock_daemon()
        daemon._running = True

        daemon.stop()

        assert daemon._running is False

    @pytest.mark.asyncio
    async def test_cleanup_stops_orchestrator(self):
        """_cleanup() should stop health orchestrator."""
        daemon, mocks = create_mock_daemon()

        await daemon._cleanup()

        mocks['health_orchestrator'].stop.assert_called_once()
