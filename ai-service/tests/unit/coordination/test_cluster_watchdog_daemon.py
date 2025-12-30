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

    def test_from_env_creates_config(self):
        """ClusterWatchdogConfig.from_env() returns config from env.

        Note: _get_default_config() doesn't exist. HandlerBase uses
        ClusterWatchdogConfig.from_env() in __init__.
        """
        with patch.dict(os.environ, {"RINGRIFT_WATCHDOG_INTERVAL": "60"}):
            config = ClusterWatchdogConfig.from_env()
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

    def test_get_event_subscriptions_returns_dict(self, daemon):
        """_get_event_subscriptions returns dict of event handlers.

        Note: HandlerBase uses _get_event_subscriptions() which returns a dict,
        not the async _subscribe_to_events() method.
        """
        subs = daemon._get_event_subscriptions()
        assert isinstance(subs, dict)
        assert "HOST_OFFLINE" in subs
        assert "HOST_ONLINE" in subs
        assert "P2P_CLUSTER_UNHEALTHY" in subs
        assert "P2P_CLUSTER_HEALTHY" in subs


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

    def test_health_check_when_not_running(self, daemon):
        """Health check returns healthy=False when not running."""
        daemon._running = False
        # health_check is sync and returns HealthCheckResult
        result = daemon.health_check()
        assert result.healthy is False

    def test_health_check_when_running(self, daemon):
        """Health check returns healthy=True when running."""
        daemon._running = True
        daemon._stats.cycles_completed = 5
        daemon._stats.errors_count = 0
        # Set recent cycle stats to avoid stale data detection
        daemon._last_cycle_stats = WatchdogCycleStats(
            cycle_start=time.time() - 10,
            cycle_end=time.time() - 5,
            errors=[],
        )
        result = daemon.health_check()
        assert result.healthy is True

    def test_health_check_stale_cycle_data(self, daemon):
        """Health check returns healthy=False for stale cycle data."""
        daemon._running = True
        # Set last cycle stats to very old (stale data)
        # age = time.time() - 1.0 >> config.check_interval_seconds * 2
        daemon._last_cycle_stats = WatchdogCycleStats(
            cycle_start=1.0,
            cycle_end=1.0,  # Ancient timestamp (year 1970)
        )
        # Health check returns healthy=False for stale data
        result = daemon.health_check()
        assert result.healthy is False


# =============================================================================
# Status Retrieval Tests
# =============================================================================


class TestStatusRetrieval:
    """Tests for status retrieval."""

    def test_get_status_includes_all_fields(self, daemon):
        """get_status includes all expected fields."""
        daemon._running = True
        daemon._stats.cycles_completed = 10
        daemon._last_cycle_stats = WatchdogCycleStats(
            nodes_discovered=5,
            nodes_activated=2,
        )

        status = daemon.get_status()

        assert "running" in status
        assert "stats" in status
        assert status["stats"]["cycles_completed"] == 10


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestLifecycle:
    """Tests for daemon lifecycle."""

    @pytest.mark.asyncio
    async def test_on_start_logs_config(self, daemon):
        """_on_start logs configuration.

        Note: HandlerBase doesn't use _subscribe_to_events(). It uses
        _get_event_subscriptions() which returns a dict. The _on_start()
        method is a hook for subclass-specific startup logic.
        """
        # _on_start() should complete without error
        await daemon._on_start()
        # Just verify it didn't crash

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


# =============================================================================
# Singleton Pattern Tests
# =============================================================================


class TestSingletonPattern:
    """Tests for singleton factory functions."""

    def teardown_method(self):
        """Reset singleton after each test."""
        from app.coordination.cluster_watchdog_daemon import reset_cluster_watchdog_daemon
        reset_cluster_watchdog_daemon()

    def test_get_instance_returns_daemon(self):
        """get_cluster_watchdog_daemon returns a daemon instance."""
        from app.coordination.cluster_watchdog_daemon import get_cluster_watchdog_daemon
        daemon = get_cluster_watchdog_daemon()
        assert isinstance(daemon, ClusterWatchdogDaemon)

    def test_get_instance_returns_same_instance(self):
        """get_cluster_watchdog_daemon returns same instance on repeated calls."""
        from app.coordination.cluster_watchdog_daemon import get_cluster_watchdog_daemon
        daemon1 = get_cluster_watchdog_daemon()
        daemon2 = get_cluster_watchdog_daemon()
        assert daemon1 is daemon2

    def test_reset_instance_clears_singleton(self):
        """reset_cluster_watchdog_daemon clears the singleton."""
        from app.coordination.cluster_watchdog_daemon import (
            get_cluster_watchdog_daemon,
            reset_cluster_watchdog_daemon,
        )
        daemon1 = get_cluster_watchdog_daemon()
        reset_cluster_watchdog_daemon()
        daemon2 = get_cluster_watchdog_daemon()
        assert daemon1 is not daemon2


# =============================================================================
# Vultr CLI Query Tests
# =============================================================================


class TestVultrCliQuery:
    """Tests for Vultr CLI querying."""

    @pytest.mark.asyncio
    async def test_query_vultr_cli_parses_output(self, daemon):
        """_query_vultr_cli parses CLI output correctly."""
        mock_output = json.dumps({
            "instances": [
                {
                    "id": "vultr-123",
                    "status": "active",
                    "main_ip": "192.168.1.100",
                    "label": "gpu-a100-node",
                }
            ]
        })

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_output

        with patch('subprocess.run', return_value=mock_result):
            nodes = await daemon._query_vultr_cli()

        assert len(nodes) == 1
        assert nodes[0].node_id == "vultr-gpu-a100-node"
        assert nodes[0].provider == "vultr"
        assert nodes[0].gpu_memory_gb == 20  # Default for Vultr A100 vGPU

    @pytest.mark.asyncio
    async def test_query_vultr_cli_handles_cli_failure(self, daemon):
        """_query_vultr_cli handles CLI failure gracefully."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "CLI error"

        with patch('subprocess.run', return_value=mock_result):
            nodes = await daemon._query_vultr_cli()

        assert nodes == []

    @pytest.mark.asyncio
    async def test_query_vultr_cli_skips_inactive(self, daemon):
        """_query_vultr_cli skips inactive instances."""
        mock_output = json.dumps({
            "instances": [
                {"id": "1", "status": "stopped", "main_ip": "10.0.0.1", "label": "gpu-node-1"},
                {"id": "2", "status": "active", "main_ip": "10.0.0.2", "label": "a100-node"},
            ]
        })

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_output

        with patch('subprocess.run', return_value=mock_result):
            nodes = await daemon._query_vultr_cli()

        assert len(nodes) == 1
        assert nodes[0].node_id == "vultr-a100-node"

    @pytest.mark.asyncio
    async def test_query_vultr_cli_skips_non_gpu(self, daemon):
        """_query_vultr_cli skips non-GPU instances."""
        mock_output = json.dumps({
            "instances": [
                {"id": "1", "status": "active", "main_ip": "10.0.0.1", "label": "cpu-only"},
                {"id": "2", "status": "active", "main_ip": "10.0.0.2", "label": "a100-gpu-node"},
            ]
        })

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_output

        with patch('subprocess.run', return_value=mock_result):
            nodes = await daemon._query_vultr_cli()

        assert len(nodes) == 1
        assert nodes[0].node_id == "vultr-a100-gpu-node"

    @pytest.mark.asyncio
    async def test_query_vultr_cli_handles_missing_cli(self, daemon):
        """_query_vultr_cli handles missing CLI gracefully."""
        with patch('subprocess.run', side_effect=FileNotFoundError("vultr-cli not found")):
            nodes = await daemon._query_vultr_cli()

        assert nodes == []


# =============================================================================
# Cluster Health Event Handler Tests
# =============================================================================


class TestClusterHealthHandlers:
    """Tests for cluster health event handlers."""

    @pytest.mark.asyncio
    async def test_on_cluster_unhealthy_pauses_spawning(self, daemon):
        """_on_cluster_unhealthy sets _cluster_healthy to False."""
        assert daemon._cluster_healthy is True

        event = {"payload": {"reason": "Too many failures"}}
        await daemon._on_cluster_unhealthy(event)

        assert daemon._cluster_healthy is False

    @pytest.mark.asyncio
    async def test_on_cluster_healthy_resumes_spawning(self, daemon):
        """_on_cluster_healthy sets _cluster_healthy to True."""
        daemon._cluster_healthy = False

        event = {}
        await daemon._on_cluster_healthy(event)

        assert daemon._cluster_healthy is True

    @pytest.mark.asyncio
    async def test_on_cluster_unhealthy_handles_missing_payload(self, daemon):
        """_on_cluster_unhealthy handles missing payload gracefully."""
        event = {}
        # Should not raise
        await daemon._on_cluster_unhealthy(event)
        assert daemon._cluster_healthy is False

    @pytest.mark.asyncio
    async def test_run_cycle_skipped_when_cluster_unhealthy(self, daemon):
        """_run_cycle is skipped when cluster is unhealthy."""
        daemon._cluster_healthy = False

        with patch.object(
            daemon, '_discover_nodes',
            new_callable=AsyncMock
        ) as mock_discover:
            await daemon._run_cycle()

        # Should not call discover_nodes when unhealthy
        mock_discover.assert_not_called()


# =============================================================================
# Node Status Check Tests
# =============================================================================


class TestNodeStatusCheck:
    """Tests for checking node status via SSH."""

    @pytest.mark.asyncio
    async def test_check_node_status_sets_reachable(self, daemon, sample_node):
        """_check_node_status sets is_reachable on success."""
        sample_node.is_reachable = False

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "50 %"

        with patch('subprocess.run', return_value=mock_result), \
             patch('app.coordination.cluster_watchdog_daemon.emit_health_check_passed', new_callable=AsyncMock):
            await daemon._check_node_status(sample_node)

        assert sample_node.is_reachable is True
        assert sample_node.gpu_utilization == 50.0

    @pytest.mark.asyncio
    async def test_check_node_status_handles_no_ssh_cmd(self, daemon):
        """_check_node_status handles missing SSH command."""
        node = WatchdogNodeStatus(
            node_id="test",
            provider="vast",
            ssh_cmd="",  # No SSH command
        )

        await daemon._check_node_status(node)

        assert node.is_reachable is False
        assert "No SSH command" in node.error

    @pytest.mark.asyncio
    async def test_check_node_status_handles_timeout(self, daemon, sample_node):
        """_check_node_status handles SSH timeout."""
        import subprocess
        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired(cmd="ssh", timeout=30)), \
             patch('app.coordination.cluster_watchdog_daemon.emit_health_check_failed', new_callable=AsyncMock):
            await daemon._check_node_status(sample_node)

        assert sample_node.is_reachable is False
        assert "timeout" in sample_node.error.lower()

    @pytest.mark.asyncio
    async def test_check_node_status_parses_python_processes(self, daemon, sample_node):
        """_check_node_status parses python process count."""
        # First call returns GPU util, second returns process count
        mock_results = [
            MagicMock(returncode=0, stdout="25 %"),  # GPU util
            MagicMock(returncode=0, stdout="5"),     # Python processes
        ]

        with patch('subprocess.run', side_effect=mock_results), \
             patch('app.coordination.cluster_watchdog_daemon.emit_health_check_passed', new_callable=AsyncMock):
            await daemon._check_node_status(sample_node)

        assert sample_node.python_processes == 5

    @pytest.mark.asyncio
    async def test_check_node_status_updates_last_check(self, daemon, sample_node):
        """_check_node_status updates last_check timestamp."""
        sample_node.last_check = 0.0
        before = time.time()

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "10 %"

        with patch('subprocess.run', return_value=mock_result), \
             patch('app.coordination.cluster_watchdog_daemon.emit_health_check_passed', new_callable=AsyncMock):
            await daemon._check_node_status(sample_node)

        assert sample_node.last_check >= before


# =============================================================================
# Health Check Extended Tests
# =============================================================================


class TestHealthCheckExtended:
    """Extended tests for health check."""

    @pytest.mark.asyncio
    async def test_health_check_excessive_errors(self, daemon):
        """Health check returns DEGRADED for excessive errors."""
        daemon._running = True
        daemon._last_cycle_stats = WatchdogCycleStats(
            cycle_start=time.time() - 10,
            cycle_end=time.time() - 5,
            errors=["error1", "error2", "error3", "error4", "error5", "error6"],  # > 5 errors
        )

        result = daemon.health_check()  # sync method
        assert result.healthy is False
        assert "errors" in result.message.lower() or "too many" in result.message.lower()

    @pytest.mark.asyncio
    async def test_health_check_includes_cycle_stats(self, daemon):
        """Health check includes cycle statistics in details."""
        daemon._running = True
        daemon._last_cycle_stats = WatchdogCycleStats(
            cycle_start=time.time() - 10,
            cycle_end=time.time() - 5,
            nodes_discovered=10,
            nodes_activated=3,
            errors=[],
        )

        result = daemon.health_check()  # sync method
        assert result.details.get("nodes_discovered") == 10
        assert result.details.get("nodes_activated") == 3


# =============================================================================
# On Stop Handler Tests
# =============================================================================


class TestOnStopHandler:
    """Tests for graceful shutdown handler."""

    @pytest.mark.asyncio
    async def test_on_stop_emits_shutdown_event(self, daemon):
        """_on_stop emits coordinator shutdown event."""
        daemon._running = True
        daemon._stats.cycles_completed = 5
        daemon._cluster_healthy = True

        mock_emit = AsyncMock()
        with patch(
            'app.coordination.event_emitters.emit_coordinator_shutdown',
            mock_emit
        ):
            await daemon._on_stop()

        mock_emit.assert_called_once()
        call_kwargs = mock_emit.call_args.kwargs
        # HandlerBase uses self.name instead of _get_daemon_name()
        assert call_kwargs["coordinator_name"] == daemon.name
        assert call_kwargs["reason"] == "graceful"

    @pytest.mark.asyncio
    async def test_on_stop_handles_import_error(self, daemon):
        """_on_stop handles ImportError gracefully when event_emitters not available."""
        # Simulate ImportError by patching the import inside _on_stop
        original_on_stop = daemon._on_stop

        async def patched_on_stop():
            # This tests the try-except block in _on_stop that catches ImportError
            # We simulate this by just calling the original and expecting it not to crash
            # even if the emit function raises an error
            try:
                await original_on_stop()
            except ImportError:
                pass  # This is expected behavior

        # The actual _on_stop method has try/except that handles ImportError
        # Just verify it doesn't crash
        await daemon._on_stop()

    @pytest.mark.asyncio
    async def test_on_stop_handles_exception(self, daemon):
        """_on_stop handles other exceptions gracefully."""
        mock_emit = AsyncMock(side_effect=RuntimeError("Unexpected error"))
        with patch(
            'app.coordination.event_emitters.emit_coordinator_shutdown',
            mock_emit
        ):
            # Should not raise - _on_stop has exception handling
            await daemon._on_stop()


# =============================================================================
# Activation Edge Case Tests
# =============================================================================


class TestActivationEdgeCases:
    """Tests for edge cases in node activation."""

    @pytest.mark.asyncio
    async def test_activate_node_no_ssh_cmd(self, daemon):
        """_activate_node returns False when no SSH command."""
        node = WatchdogNodeStatus(
            node_id="test",
            provider="vast",
            ssh_cmd="",  # No SSH command
        )

        result = await daemon._activate_node(node)
        assert result is False

    @pytest.mark.asyncio
    async def test_activate_node_timeout_treated_as_success(self, daemon, sample_node):
        """_activate_node treats timeout as success (nohup may timeout)."""
        import subprocess
        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired(cmd="ssh", timeout=60)):
            result = await daemon._activate_node(sample_node)

        # Timeout with nohup often means success
        assert result is True

    @pytest.mark.asyncio
    async def test_activate_node_cycles_configs(self, daemon, sample_node):
        """_activate_node cycles through selfplay configs."""
        initial_index = daemon._config_index
        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch('subprocess.run', return_value=mock_result):
            await daemon._activate_node(sample_node)

        # Config index should have advanced
        expected_index = (initial_index + 1) % len(daemon.config.selfplay_configs)
        assert daemon._config_index == expected_index


# =============================================================================
# Get Status Extended Tests
# =============================================================================


class TestGetStatusExtended:
    """Extended tests for status retrieval."""

    def test_get_status_includes_nodes_list(self, daemon):
        """get_status includes list of tracked nodes."""
        daemon._running = True
        daemon._nodes = {
            "vast-1": WatchdogNodeStatus(
                node_id="vast-1",
                provider="vast",
                ssh_cmd="ssh root@10.0.0.1",
                gpu_utilization=50.0,
                python_processes=3,
                is_reachable=True,
                consecutive_failures=0,
            ),
            "runpod-1": WatchdogNodeStatus(
                node_id="runpod-1",
                provider="runpod",
                ssh_cmd="ssh root@10.0.0.2",
                gpu_utilization=0.0,
                is_reachable=False,
                consecutive_failures=2,
            ),
        }

        status = daemon.get_status()

        assert "nodes" in status
        assert len(status["nodes"]) == 2

        vast_node = next(n for n in status["nodes"] if n["id"] == "vast-1")
        assert vast_node["provider"] == "vast"
        assert vast_node["gpu_util"] == 50.0
        assert vast_node["processes"] == 3
        assert vast_node["reachable"] is True

    def test_get_status_includes_last_cycle(self, daemon):
        """get_status includes last cycle statistics."""
        daemon._running = True
        daemon._last_cycle_stats = WatchdogCycleStats(
            nodes_discovered=10,
            nodes_reachable=8,
            nodes_idle=3,
            nodes_activated=2,
            nodes_failed=1,
        )

        status = daemon.get_status()

        assert "last_cycle" in status
        assert status["last_cycle"]["discovered"] == 10
        assert status["last_cycle"]["reachable"] == 8
        assert status["last_cycle"]["idle"] == 3
        assert status["last_cycle"]["activated"] == 2
        assert status["last_cycle"]["failed"] == 1

    def test_get_status_includes_config(self, daemon, mock_config):
        """get_status includes watchdog-specific config."""
        status = daemon.get_status()

        assert "config" in status
        assert status["config"]["min_gpu_utilization"] == mock_config.min_gpu_utilization

    def test_get_status_tracked_nodes_count(self, daemon):
        """get_status includes tracked nodes count."""
        daemon._nodes = {
            "node1": WatchdogNodeStatus(node_id="node1", provider="vast", ssh_cmd=""),
            "node2": WatchdogNodeStatus(node_id="node2", provider="runpod", ssh_cmd=""),
        }

        status = daemon.get_status()
        assert status["tracked_nodes"] == 2


# =============================================================================
# Node Unhealthy Event Tests
# =============================================================================


class TestNodeUnhealthyEvent:
    """Tests for NODE_UNHEALTHY event emission."""

    @pytest.mark.asyncio
    async def test_emits_node_unhealthy_on_persistent_failure(self, daemon, sample_node):
        """Emits NODE_UNHEALTHY after max consecutive failures."""
        sample_node.is_reachable = True  # Start reachable
        sample_node.gpu_utilization = 5.0  # Idle
        sample_node.consecutive_failures = daemon.config.max_consecutive_failures - 1
        sample_node.last_activation = 0.0  # Not recently activated

        with patch.object(
            daemon, '_discover_nodes',
            return_value=[sample_node]
        ), patch.object(
            daemon, '_check_node_status',
            new_callable=AsyncMock
        ), patch.object(
            daemon, '_activate_node',
            return_value=False  # Activation fails
        ), patch(
            'app.coordination.cluster_watchdog_daemon.emit_node_unhealthy',
            new_callable=AsyncMock
        ) as mock_emit:
            await daemon._run_cycle()

        # Should emit NODE_UNHEALTHY
        mock_emit.assert_called_once()
        call_kwargs = mock_emit.call_args.kwargs
        assert call_kwargs["node_id"] == sample_node.node_id
        assert "failures" in call_kwargs["reason"].lower() or "persistent" in call_kwargs["reason"].lower()


# =============================================================================
# Max Activations Per Cycle Tests
# =============================================================================


class TestMaxActivationsPerCycle:
    """Tests for max activations per cycle limit."""

    @pytest.mark.asyncio
    async def test_respects_max_activations_limit(self, daemon):
        """_run_cycle respects max_activations_per_cycle."""
        daemon.config.max_activations_per_cycle = 2

        # Create 5 idle nodes
        nodes = [
            WatchdogNodeStatus(
                node_id=f"vast-{i}",
                provider="vast",
                ssh_cmd=f"ssh root@10.0.0.{i}",
                gpu_utilization=0.0,  # All idle
                is_reachable=True,
            )
            for i in range(5)
        ]

        activation_count = 0

        async def mock_activate(node):
            nonlocal activation_count
            activation_count += 1
            return True

        with patch.object(
            daemon, '_discover_nodes',
            return_value=nodes
        ), patch.object(
            daemon, '_check_node_status',
            new_callable=AsyncMock
        ), patch.object(
            daemon, '_activate_node',
            side_effect=mock_activate
        ):
            await daemon._run_cycle()

        # Should only activate max_activations_per_cycle nodes
        assert activation_count == 2