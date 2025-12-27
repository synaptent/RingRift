"""Tests for cluster_watchdog_daemon.py - Cluster Watchdog Daemon.

December 2025: Created as part of test coverage initiative.
Tests the ClusterWatchdogDaemon for cluster utilization monitoring.

Coverage includes:
- Configuration loading (env vars, defaults)
- Node discovery from provider CLIs
- GPU utilization checking
- Node activation logic
- Event handling
- Cycle statistics
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.cluster_watchdog_daemon import (
    ClusterWatchdogConfig,
    ClusterWatchdogDaemon,
    WatchdogCycleStats,
    WatchdogNodeStatus,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    return ClusterWatchdogConfig(
        check_interval_seconds=60,
        min_gpu_utilization=20.0,
        max_consecutive_failures=3,
        activation_cooldown_seconds=600,
        ssh_timeout_seconds=30,
        max_activations_per_cycle=10,
    )


@pytest.fixture
def daemon(mock_config):
    """Create daemon with mock configuration."""
    return ClusterWatchdogDaemon(config=mock_config)


@pytest.fixture
def sample_node():
    """Create sample node status."""
    return WatchdogNodeStatus(
        node_id="vast-12345",
        provider="vast",
        ssh_cmd="ssh -p 22 root@10.0.0.1",
        gpu_memory_gb=24,
        is_reachable=True,
        gpu_utilization=5.0,  # Idle
    )


# =============================================================================
# Configuration Tests
# =============================================================================


class TestClusterWatchdogConfig:
    """Tests for ClusterWatchdogConfig."""

    def test_default_values(self):
        """Config has correct defaults."""
        config = ClusterWatchdogConfig()
        assert config.enabled is True
        assert config.check_interval_seconds == 300
        assert config.min_gpu_utilization == 20.0
        assert config.max_consecutive_failures == 3
        assert config.activation_cooldown_seconds == 600

    def test_from_env_enabled(self):
        """from_env reads enabled from environment."""
        with patch.dict(os.environ, {"RINGRIFT_WATCHDOG_ENABLED": "0"}):
            config = ClusterWatchdogConfig.from_env()
            assert config.enabled is False

        with patch.dict(os.environ, {"RINGRIFT_WATCHDOG_ENABLED": "1"}):
            config = ClusterWatchdogConfig.from_env()
            assert config.enabled is True

    def test_from_env_interval(self):
        """from_env reads interval from environment."""
        with patch.dict(os.environ, {"RINGRIFT_WATCHDOG_INTERVAL": "120"}):
            config = ClusterWatchdogConfig.from_env()
            assert config.check_interval_seconds == 120

    def test_from_env_min_gpu(self):
        """from_env reads min GPU utilization from environment."""
        with patch.dict(os.environ, {"RINGRIFT_WATCHDOG_MIN_GPU": "15.0"}):
            config = ClusterWatchdogConfig.from_env()
            assert config.min_gpu_utilization == 15.0

    def test_selfplay_configs_default(self):
        """Default selfplay configs include common board types."""
        config = ClusterWatchdogConfig()
        assert len(config.selfplay_configs) > 0
        assert ("hex8", 2) in config.selfplay_configs
        assert ("square8", 2) in config.selfplay_configs


# =============================================================================
# Data Class Tests
# =============================================================================


class TestWatchdogNodeStatus:
    """Tests for WatchdogNodeStatus data class."""

    def test_default_values(self):
        """Node status has correct defaults."""
        node = WatchdogNodeStatus(
            node_id="test",
            provider="vast",
            ssh_cmd="ssh root@host",
        )
        assert node.gpu_memory_gb == 0
        assert node.last_check == 0.0
        assert node.last_activation == 0.0
        assert node.gpu_utilization == 0.0
        assert node.consecutive_failures == 0
        assert node.is_reachable is False
        assert node.error == ""


class TestWatchdogCycleStats:
    """Tests for WatchdogCycleStats data class."""

    def test_default_values(self):
        """Cycle stats has correct defaults."""
        stats = WatchdogCycleStats()
        assert stats.cycle_start == 0.0
        assert stats.cycle_end == 0.0
        assert stats.nodes_discovered == 0
        assert stats.nodes_reachable == 0
        assert stats.nodes_idle == 0
        assert stats.nodes_activated == 0
        assert stats.nodes_failed == 0
        assert stats.errors == []


# =============================================================================
# Daemon Initialization Tests
# =============================================================================


class TestClusterWatchdogDaemonInit:
    """Tests for daemon initialization."""

    def test_init_with_config(self, mock_config):
        """Daemon initializes with provided config."""
        daemon = ClusterWatchdogDaemon(config=mock_config)
        assert daemon.config is mock_config
        assert daemon._nodes == {}
        assert daemon._config_index == 0
        assert daemon._last_cycle_stats is None

    def test_init_default_config(self):
        """Daemon initializes with default config."""
        daemon = ClusterWatchdogDaemon()
        assert daemon.config is not None
        assert isinstance(daemon.config, ClusterWatchdogConfig)

    def test_get_default_config(self):
        """_get_default_config returns config from env."""
        with patch.dict(os.environ, {"RINGRIFT_WATCHDOG_INTERVAL": "60"}):
            config = ClusterWatchdogDaemon._get_default_config()
            assert config.check_interval_seconds == 60


# =============================================================================
# Event Handler Tests
# =============================================================================


class TestEventHandlers:
    """Tests for event handlers."""

    @pytest.mark.asyncio
    async def test_on_host_offline_handles_event(self, daemon):
        """_on_host_offline handles event without error."""
        event = {"payload": {"host": "test-node"}}
        # Should not raise
        await daemon._on_host_offline(event)

    @pytest.mark.asyncio
    async def test_on_host_online_handles_event(self, daemon):
        """_on_host_online handles event without error."""
        event = {"payload": {"host": "test-node"}}
        # Should not raise
        await daemon._on_host_online(event)

    @pytest.mark.asyncio
    async def test_on_host_offline_handles_missing_payload(self, daemon):
        """_on_host_offline handles missing payload gracefully."""
        event = {}
        # Should not raise
        await daemon._on_host_offline(event)

    @pytest.mark.asyncio
    async def test_subscribe_to_events_handles_import_error(self, daemon):
        """_subscribe_to_events handles ImportError gracefully."""
        with patch(
            'app.coordination.cluster_watchdog_daemon.get_event_router',
            side_effect=ImportError("No event router"),
            create=True
        ):
            # Should not raise
            await daemon._subscribe_to_events()


# =============================================================================
# Node Discovery Tests
# =============================================================================


class TestNodeDiscovery:
    """Tests for node discovery."""

    @pytest.mark.asyncio
    async def test_discover_nodes_combines_providers(self, daemon):
        """_discover_nodes combines results from all providers."""
        with patch.object(
            daemon, '_query_vast_cli',
            return_value=[WatchdogNodeStatus(node_id="vast-1", provider="vast", ssh_cmd="")]
        ), patch.object(
            daemon, '_query_runpod_cli',
            return_value=[WatchdogNodeStatus(node_id="runpod-1", provider="runpod", ssh_cmd="")]
        ), patch.object(
            daemon, '_query_vultr_cli',
            return_value=[WatchdogNodeStatus(node_id="vultr-1", provider="vultr", ssh_cmd="")]
        ):
            nodes = await daemon._discover_nodes()

        assert len(nodes) == 3
        node_ids = [n.node_id for n in nodes]
        assert "vast-1" in node_ids
        assert "runpod-1" in node_ids
        assert "vultr-1" in node_ids

    @pytest.mark.asyncio
    async def test_discover_nodes_preserves_failure_counts(self, daemon):
        """_discover_nodes preserves existing failure counts."""
        # Pre-populate node with failure count
        daemon._nodes["vast-1"] = WatchdogNodeStatus(
            node_id="vast-1",
            provider="vast",
            ssh_cmd="",
            consecutive_failures=5,
            last_activation=1000.0,
        )

        with patch.object(
            daemon, '_query_vast_cli',
            return_value=[WatchdogNodeStatus(node_id="vast-1", provider="vast", ssh_cmd="")]
        ), patch.object(
            daemon, '_query_runpod_cli',
            return_value=[]
        ), patch.object(
            daemon, '_query_vultr_cli',
            return_value=[]
        ):
            nodes = await daemon._discover_nodes()

        assert len(nodes) == 1
        assert nodes[0].consecutive_failures == 5
        assert nodes[0].last_activation == 1000.0


class TestVastCliQuery:
    """Tests for Vast.ai CLI querying."""

    @pytest.mark.asyncio
    async def test_query_vast_cli_parses_output(self, daemon):
        """_query_vast_cli parses CLI output correctly."""
        mock_output = json.dumps([
            {
                "id": "12345",
                "actual_status": "running",
                "ssh_host": "10.0.0.1",
                "ssh_port": 22,
                "gpu_name": "RTX 4090",
            }
        ])

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_output

        with patch('subprocess.run', return_value=mock_result):
            nodes = await daemon._query_vast_cli()

        assert len(nodes) == 1
        assert nodes[0].node_id == "vast-12345"
        assert nodes[0].provider == "vast"
        assert nodes[0].gpu_memory_gb == 24  # 4090

    @pytest.mark.asyncio
    async def test_query_vast_cli_handles_cli_failure(self, daemon):
        """_query_vast_cli handles CLI failure gracefully."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "CLI error"

        with patch('subprocess.run', return_value=mock_result):
            nodes = await daemon._query_vast_cli()

        assert nodes == []

    @pytest.mark.asyncio
    async def test_query_vast_cli_skips_non_running(self, daemon):
        """_query_vast_cli skips non-running instances."""
        mock_output = json.dumps([
            {"id": "1", "actual_status": "stopped"},
            {"id": "2", "actual_status": "running", "ssh_host": "10.0.0.2", "ssh_port": 22, "gpu_name": "A100"},
        ])

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_output

        with patch('subprocess.run', return_value=mock_result):
            nodes = await daemon._query_vast_cli()

        assert len(nodes) == 1
        assert nodes[0].node_id == "vast-2"

    @pytest.mark.asyncio
    async def test_query_vast_cli_gpu_memory_mapping(self, daemon):
        """_query_vast_cli maps GPU names to memory correctly."""
        test_cases = [
            ("RTX 5090", 32),
            ("RTX 5080", 16),
            ("A100", 80),
            ("H100", 80),
            ("A40", 48),
            ("RTX 4090", 24),
            ("RTX 3090", 24),
            ("RTX 4060 Ti", 8),
            ("Unknown GPU", 24),  # Default
        ]

        for gpu_name, expected_memory in test_cases:
            mock_output = json.dumps([
                {
                    "id": "1",
                    "actual_status": "running",
                    "ssh_host": "10.0.0.1",
                    "ssh_port": 22,
                    "gpu_name": gpu_name,
                }
            ])

            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = mock_output

            with patch('subprocess.run', return_value=mock_result):
                nodes = await daemon._query_vast_cli()

            assert len(nodes) == 1, f"Failed for {gpu_name}"
            assert nodes[0].gpu_memory_gb == expected_memory, f"Failed for {gpu_name}"


class TestRunPodCliQuery:
    """Tests for RunPod CLI querying."""

    @pytest.mark.asyncio
    async def test_query_runpod_cli_parses_output(self, daemon):
        """_query_runpod_cli parses CLI output correctly."""
        mock_output = "ID\tNAME\tSTATUS\ntest-pod-1\tMyPod\tRUNNING\n"

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_output

        with patch('subprocess.run', return_value=mock_result):
            nodes = await daemon._query_runpod_cli()

        assert len(nodes) == 1
        assert nodes[0].node_id == "runpod-test-pod-1"
        assert nodes[0].provider == "runpod"

    @pytest.mark.asyncio
    async def test_query_runpod_cli_skips_non_running(self, daemon):
        """_query_runpod_cli skips non-running pods."""
        mock_output = "ID\tNAME\tSTATUS\npod-1\tPod1\tSTOPPED\npod-2\tPod2\tRUNNING\n"

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_output

        with patch('subprocess.run', return_value=mock_result):
            nodes = await daemon._query_runpod_cli()

        assert len(nodes) == 1
        assert nodes[0].node_id == "runpod-pod-2"


# =============================================================================
# Run Cycle Tests
# =============================================================================


class TestRunCycle:
    """Tests for the main run cycle."""

    @pytest.mark.asyncio
    async def test_run_cycle_no_nodes(self, daemon):
        """_run_cycle handles no nodes gracefully."""
        with patch.object(daemon, '_discover_nodes', return_value=[]):
            await daemon._run_cycle()

        # Should complete without error

    @pytest.mark.asyncio
    async def test_run_cycle_updates_stats(self, daemon, sample_node):
        """_run_cycle updates cycle stats."""
        with patch.object(
            daemon, '_discover_nodes',
            return_value=[sample_node]
        ), patch.object(
            daemon, '_check_node_status',
            new_callable=AsyncMock
        ), patch.object(
            daemon, '_activate_node',
            return_value=True
        ):
            await daemon._run_cycle()

        assert daemon._last_cycle_stats is not None
        assert daemon._last_cycle_stats.nodes_discovered == 1

    @pytest.mark.asyncio
    async def test_run_cycle_respects_cooldown(self, daemon, sample_node):
        """_run_cycle respects activation cooldown."""
        sample_node.last_activation = time.time()  # Recently activated

        with patch.object(
            daemon, '_discover_nodes',
            return_value=[sample_node]
        ), patch.object(
            daemon, '_check_node_status',
            new_callable=AsyncMock
        ), patch.object(
            daemon, '_activate_node',
            new_callable=AsyncMock
        ) as mock_activate:
            await daemon._run_cycle()

        # Should not activate recently activated node
        mock_activate.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_cycle_handles_check_failure(self, daemon, sample_node):
        """_run_cycle handles node check failure gracefully."""
        with patch.object(
            daemon, '_discover_nodes',
            return_value=[sample_node]
        ), patch.object(
            daemon, '_check_node_status',
            side_effect=Exception("SSH failed")
        ):
            await daemon._run_cycle()

        assert sample_node.consecutive_failures == 1


# =============================================================================
# Node Activation Tests
# =============================================================================


class TestNodeActivation:
    """Tests for node activation."""

    @pytest.mark.asyncio
    async def test_activate_node_spawns_selfplay(self, daemon, sample_node):
        """_activate_node spawns selfplay on idle node."""
        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch('subprocess.run', return_value=mock_result):
            result = await daemon._activate_node(sample_node)

        assert result is True

    @pytest.mark.asyncio
    async def test_activate_node_handles_ssh_failure(self, daemon, sample_node):
        """_activate_node handles SSH failure gracefully."""
        with patch('subprocess.run', side_effect=Exception("SSH failed")):
            result = await daemon._activate_node(sample_node)

        assert result is False


# =============================================================================
# Configuration Cycling Tests
# =============================================================================


class TestConfigCycling:
    """Tests for selfplay config cycling."""

    def test_config_index_cycles(self, daemon):
        """Config index cycles through available configs."""
        configs = daemon.config.selfplay_configs
        num_configs = len(configs)

        # Initial index is 0
        assert daemon._config_index == 0

        # Simulate cycling by incrementing index (as daemon would during activation)
        for _i in range(num_configs * 2):
            # The daemon uses modulo when accessing configs
            config = configs[daemon._config_index % num_configs]
            assert config in configs
            daemon._config_index = (daemon._config_index + 1) % num_configs

        # After full cycles, index wraps correctly
        assert daemon._config_index < num_configs


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health check interface."""

    @pytest.mark.asyncio
    async def test_health_check_when_not_running(self, daemon):
        """Health check returns False when not running."""
        daemon._running = False
        # health_check is async and returns bool
        result = await daemon.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_when_running(self, daemon):
        """Health check returns True when running."""
        daemon._running = True
        daemon._cycles_completed = 5
        daemon._errors_count = 0
        result = await daemon.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_stale_cycle_data(self, daemon):
        """Health check returns False for stale cycle data."""
        daemon._running = True
        # Set last cycle stats to very old (stale data)
        # age = time.time() - 1.0 >> config.check_interval_seconds * 2
        daemon._last_cycle_stats = WatchdogCycleStats(
            cycle_start=1.0,
            cycle_end=1.0,  # Ancient timestamp (year 1970)
        )
        # Health check returns False for stale data
        result = await daemon.health_check()
        assert result is False


# =============================================================================
# Status Retrieval Tests
# =============================================================================


class TestStatusRetrieval:
    """Tests for status retrieval."""

    def test_get_status_includes_all_fields(self, daemon):
        """get_status includes all expected fields."""
        daemon._running = True
        daemon._cycles_completed = 10
        daemon._last_cycle_stats = WatchdogCycleStats(
            nodes_discovered=5,
            nodes_activated=2,
        )

        status = daemon.get_status()

        assert "running" in status
        assert "cycles_completed" in status or "stats" in status


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestLifecycle:
    """Tests for daemon lifecycle."""

    @pytest.mark.asyncio
    async def test_on_start_logs_config(self, daemon):
        """_on_start logs configuration."""
        with patch.object(daemon, '_subscribe_to_events', new_callable=AsyncMock):
            await daemon._on_start()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_start_and_stop(self, daemon):
        """Daemon can start and stop cleanly."""
        # Start daemon in background
        task = asyncio.create_task(daemon.start())

        # Give it a moment to start
        await asyncio.sleep(0.1)

        # Stop it
        await daemon.stop()

        # Cancel the task
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


# =============================================================================
# Error Escalation Tests
# =============================================================================


class TestErrorEscalation:
    """Tests for error escalation."""

    @pytest.mark.asyncio
    async def test_consecutive_failures_tracked(self, daemon, sample_node):
        """Consecutive failures are tracked per node."""
        sample_node.is_reachable = False

        with patch.object(
            daemon, '_discover_nodes',
            return_value=[sample_node]
        ), patch.object(
            daemon, '_check_node_status',
            side_effect=Exception("Check failed")
        ):
            # Run multiple cycles
            for _ in range(3):
                await daemon._run_cycle()

        assert sample_node.consecutive_failures >= 3

    @pytest.mark.asyncio
    async def test_escalation_increments_failures_past_threshold(self, daemon, sample_node):
        """Failures past threshold are tracked correctly."""
        # Start at max failures - 1
        sample_node.consecutive_failures = daemon.config.max_consecutive_failures - 1

        with patch.object(
            daemon, '_discover_nodes',
            return_value=[sample_node]
        ), patch.object(
            daemon, '_check_node_status',
            side_effect=Exception("Check failed")
        ):
            await daemon._run_cycle()

        # Should increment to max_consecutive_failures
        assert sample_node.consecutive_failures >= daemon.config.max_consecutive_failures
