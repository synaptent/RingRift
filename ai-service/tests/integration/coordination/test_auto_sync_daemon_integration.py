"""Integration tests for AutoSyncDaemon multi-component coordination.

These tests verify:
1. AutoSyncConfig loading and validation
2. SyncStrategy resolution
3. Daemon lifecycle (start/stop)
4. Status and health reporting
5. Event integration

December 2025 - RingRift AI Service
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.auto_sync_daemon import (
    AutoSyncConfig,
    AutoSyncDaemon,
    SyncStats,
    SyncStrategy,
)
from app.coordination.protocols import CoordinatorStatus, HealthCheckResult


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_config():
    """Create a minimal AutoSyncConfig for testing."""
    return AutoSyncConfig(
        enabled=True,
        strategy=SyncStrategy.HYBRID,
        interval_seconds=60,
        exclude_hosts=["coordinator"],
        max_concurrent_syncs=2,
        min_games_to_sync=5,
        quality_filter_enabled=False,
        enable_quality_extraction=False,
    )


@pytest.fixture
def mock_manifest():
    """Create a mock ClusterManifest."""
    manifest = MagicMock()

    # Mock capacity
    capacity = MagicMock()
    capacity.usage_percent = 30.0
    manifest.update_local_capacity.return_value = capacity
    manifest.get_node_capacity.return_value = capacity

    # Mock inventory
    inventory = MagicMock()
    inventory.game_count = 100
    inventory.model_count = 5
    inventory.npz_count = 10
    manifest.get_node_inventory.return_value = inventory

    # Mock sync policy
    policy = MagicMock()
    policy.receive_games = True
    policy.receive_models = True
    policy.exclusion_reason = ""
    manifest.get_sync_policy.return_value = policy

    return manifest


@pytest.fixture
def mock_cluster_config():
    """Create a mock ClusterConfig for load_cluster_config."""
    config = MagicMock()
    config.hosts_raw = {"test-node": {"status": "ready", "role": "selfplay"}}
    config.get_raw_section.return_value = {
        "max_disk_usage_percent": 70.0,
        "target_disk_usage_percent": 60.0,
        "priority_hosts": [],
    }

    # Mock sync_routing
    sync_routing = MagicMock()
    sync_routing.max_disk_usage_percent = 70.0
    sync_routing.target_disk_usage_percent = 60.0
    sync_routing.allowed_external_storage = []
    config.sync_routing = sync_routing

    # Mock auto_sync
    auto_sync = MagicMock()
    auto_sync.enabled = True
    auto_sync.interval_seconds = 60
    auto_sync.gossip_interval_seconds = 30
    auto_sync.exclude_hosts = []
    auto_sync.skip_nfs_sync = True
    auto_sync.max_concurrent_syncs = 4
    auto_sync.min_games_to_sync = 10
    auto_sync.bandwidth_limit_mbps = 20
    config.auto_sync = auto_sync

    return config


@pytest.fixture
def daemon(mock_config, mock_manifest, mock_cluster_config):
    """Create AutoSyncDaemon with mocked dependencies."""
    # Patch get_cluster_manifest at its source (imported lazily in _initialize_cluster_manifest)
    with patch("app.distributed.cluster_manifest.get_cluster_manifest", return_value=mock_manifest), \
         patch("app.config.cluster_config.load_cluster_config", return_value=mock_cluster_config), \
         patch("app.distributed.cluster_manifest.load_cluster_config", return_value=mock_cluster_config), \
         patch("socket.gethostname", return_value="test-node"):
        daemon = AutoSyncDaemon(config=mock_config)
        yield daemon


# =============================================================================
# Test SyncStrategy
# =============================================================================


class TestSyncStrategy:
    """Tests for SyncStrategy enum."""

    def test_strategy_values(self):
        """SyncStrategy should have correct string values."""
        assert SyncStrategy.HYBRID == "hybrid"
        assert SyncStrategy.EPHEMERAL == "ephemeral"
        assert SyncStrategy.BROADCAST == "broadcast"
        assert SyncStrategy.AUTO == "auto"

    def test_all_strategies_exist(self):
        """All expected strategies should exist."""
        strategies = [SyncStrategy.HYBRID, SyncStrategy.EPHEMERAL,
                     SyncStrategy.BROADCAST, SyncStrategy.AUTO]
        assert len(strategies) == 4


# =============================================================================
# Test AutoSyncConfig
# =============================================================================


class TestAutoSyncConfig:
    """Tests for AutoSyncConfig dataclass."""

    def test_default_values(self):
        """AutoSyncConfig should have sensible defaults."""
        config = AutoSyncConfig()

        assert config.enabled is True
        assert config.strategy == SyncStrategy.AUTO
        assert config.interval_seconds == 60
        assert config.gossip_interval_seconds == 30
        assert config.max_concurrent_syncs == 4
        assert config.min_games_to_sync == 10
        assert config.max_disk_usage_percent == 70.0
        assert config.skip_nfs_sync is True

    def test_custom_values(self):
        """AutoSyncConfig should accept custom values."""
        config = AutoSyncConfig(
            enabled=False,
            strategy=SyncStrategy.EPHEMERAL,
            interval_seconds=30,
            exclude_hosts=["node-a", "node-b"],
            max_concurrent_syncs=8,
        )

        assert config.enabled is False
        assert config.strategy == SyncStrategy.EPHEMERAL
        assert config.interval_seconds == 30
        assert config.exclude_hosts == ["node-a", "node-b"]
        assert config.max_concurrent_syncs == 8

    def test_ephemeral_specific_fields(self):
        """AutoSyncConfig should have ephemeral-specific fields."""
        config = AutoSyncConfig()

        assert config.ephemeral_poll_seconds == 5
        assert config.ephemeral_write_through is True
        assert config.ephemeral_write_through_timeout == 60
        assert config.ephemeral_wal_enabled is True

    def test_quality_filter_fields(self):
        """AutoSyncConfig should have quality filter fields."""
        config = AutoSyncConfig()

        assert config.quality_filter_enabled is True
        assert config.min_quality_for_sync > 0
        assert config.quality_sample_size > 0


# =============================================================================
# Test SyncStats
# =============================================================================


class TestSyncStats:
    """Tests for SyncStats dataclass."""

    def test_default_values(self):
        """SyncStats should have zero defaults."""
        stats = SyncStats()

        assert stats.games_synced == 0
        assert stats.databases_skipped_quality == 0
        assert stats.databases_quality_checked == 0

    def test_backward_compat_aliases(self):
        """SyncStats should have backward-compatible aliases."""
        stats = SyncStats()

        # Access via aliases
        assert stats.total_syncs == 0
        assert stats.successful_syncs == 0
        assert stats.failed_syncs == 0
        assert stats.bytes_transferred == 0
        assert stats.last_sync_time >= 0

    def test_to_dict(self):
        """SyncStats.to_dict should include all fields."""
        stats = SyncStats(games_synced=50)

        d = stats.to_dict()

        assert isinstance(d, dict)
        assert "games_synced" in d
        assert d["games_synced"] == 50
        assert "total_syncs" in d
        assert "successful_syncs" in d


# =============================================================================
# Test AutoSyncDaemon Initialization
# =============================================================================


class TestAutoSyncDaemonInit:
    """Tests for AutoSyncDaemon initialization."""

    def test_initialization(self, daemon):
        """Daemon should initialize with correct node_id."""
        assert daemon.node_id == "test-node"
        assert daemon._running is False

    def test_initialization_with_config(self, mock_config, mock_cluster_config, mock_manifest):
        """Daemon should use provided config."""
        with patch("app.distributed.cluster_manifest.get_cluster_manifest", return_value=mock_manifest), \
             patch("app.distributed.cluster_manifest.load_cluster_config", return_value=mock_cluster_config), \
             patch("socket.gethostname", return_value="test-node"):
            daemon = AutoSyncDaemon(config=mock_config)

            assert daemon.config.strategy == SyncStrategy.HYBRID
            assert daemon.config.interval_seconds == 60

    def test_coordinator_status_initial(self, daemon):
        """Daemon should start in INITIALIZING status."""
        # After __init__, status is INITIALIZING (before start())
        assert daemon.status == CoordinatorStatus.INITIALIZING

    def test_name_property(self, daemon):
        """Daemon should have correct name property."""
        assert daemon.name == "AutoSyncDaemon"


# =============================================================================
# Test Daemon Lifecycle
# =============================================================================


class TestAutoSyncDaemonLifecycle:
    """Tests for daemon start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_sets_running(self, daemon):
        """start() should set _running to True."""
        with patch.object(daemon, "_start_gossip_sync", new_callable=AsyncMock), \
             patch.object(daemon, "_subscribe_to_events"), \
             patch("app.coordination.auto_sync_daemon.safe_create_task") as mock_task, \
             patch("app.coordination.auto_sync_daemon.register_coordinator"):

            mock_task.return_value = MagicMock()

            await daemon.start()

            assert daemon._running is True
            assert daemon.status == CoordinatorStatus.RUNNING

    @pytest.mark.asyncio
    async def test_start_when_disabled(self, mock_manifest, mock_cluster_config):
        """start() should not run when config.enabled is False."""
        config = AutoSyncConfig(enabled=False)

        with patch("app.distributed.cluster_manifest.get_cluster_manifest", return_value=mock_manifest), \
             patch("app.distributed.cluster_manifest.load_cluster_config", return_value=mock_cluster_config), \
             patch("socket.gethostname", return_value="test-node"):
            daemon = AutoSyncDaemon(config=config)

            await daemon.start()

            assert daemon._running is False
            assert daemon.status == CoordinatorStatus.STOPPED

    @pytest.mark.asyncio
    async def test_stop_sets_stopped(self, daemon):
        """stop() should set status to STOPPED."""
        # First start the daemon
        with patch.object(daemon, "_start_gossip_sync", new_callable=AsyncMock), \
             patch.object(daemon, "_subscribe_to_events"), \
             patch("app.coordination.auto_sync_daemon.safe_create_task") as mock_task, \
             patch("app.coordination.auto_sync_daemon.register_coordinator"):

            mock_task.return_value = MagicMock()
            await daemon.start()

        # Then stop it
        with patch("app.coordination.auto_sync_daemon.unregister_coordinator"):
            daemon._sync_task = None  # Clear task to avoid cancel
            await daemon.stop()

        assert daemon._running is False
        assert daemon.status == CoordinatorStatus.STOPPED

    @pytest.mark.asyncio
    async def test_stop_idempotent(self, daemon):
        """stop() should be safe to call when already stopped."""
        daemon._coordinator_status = CoordinatorStatus.STOPPED

        # Should not raise
        await daemon.stop()

        assert daemon.status == CoordinatorStatus.STOPPED


# =============================================================================
# Test Status Reporting
# =============================================================================


class TestAutoSyncDaemonStatus:
    """Tests for daemon status reporting."""

    def test_get_status_structure(self, daemon):
        """get_status should return complete status dict."""
        status = daemon.get_status()

        assert isinstance(status, dict)
        assert "node_id" in status
        assert "running" in status
        assert "provider" in status
        assert "config" in status
        assert "stats" in status

    def test_get_status_config_section(self, daemon):
        """get_status should include config details."""
        status = daemon.get_status()

        config = status["config"]
        assert "enabled" in config
        assert "interval_seconds" in config
        assert "exclude_hosts" in config

    def test_get_status_stats_section(self, daemon):
        """get_status should include stats details."""
        status = daemon.get_status()

        stats = status["stats"]
        assert "total_syncs" in stats
        assert "successful_syncs" in stats
        assert "failed_syncs" in stats
        assert "games_synced" in stats

    def test_is_running_reflects_state(self, daemon):
        """is_running should reflect _running state."""
        assert daemon.is_running() is False

        daemon._running = True
        assert daemon.is_running() is True


# =============================================================================
# Test Health Check
# =============================================================================


class TestAutoSyncDaemonHealth:
    """Tests for daemon health checking."""

    def test_health_check_when_stopped(self, daemon):
        """Health check should pass when daemon is stopped."""
        daemon._coordinator_status = CoordinatorStatus.STOPPED

        result = daemon.health_check()

        assert isinstance(result, HealthCheckResult)
        assert result.healthy is True

    def test_health_check_when_disabled(self, mock_manifest, mock_cluster_config):
        """Health check should pass when daemon is disabled."""
        config = AutoSyncConfig(enabled=False)

        with patch("app.distributed.cluster_manifest.get_cluster_manifest", return_value=mock_manifest), \
             patch("app.distributed.cluster_manifest.load_cluster_config", return_value=mock_cluster_config), \
             patch("socket.gethostname", return_value="test-node"):
            daemon = AutoSyncDaemon(config=config)

            result = daemon.health_check()

            assert result.healthy is True

    def test_health_check_when_error_state(self, daemon):
        """Health check should fail when in error state."""
        daemon._coordinator_status = CoordinatorStatus.ERROR
        daemon._last_error = "Test error"

        result = daemon.health_check()

        assert result.healthy is False

    def test_health_check_high_failure_rate(self, daemon):
        """Health check should degrade with high failure rate."""
        daemon._coordinator_status = CoordinatorStatus.RUNNING
        daemon._stats.syncs_completed = 5
        daemon._stats.syncs_failed = 10  # More failures than successes
        daemon._stats.operations_attempted = 15

        result = daemon.health_check()

        # Should be degraded due to high failure rate
        assert result.status == CoordinatorStatus.DEGRADED


# =============================================================================
# Test Strategy Resolution
# =============================================================================


class TestStrategyResolution:
    """Tests for sync strategy resolution."""

    def test_explicit_hybrid_strategy(self, mock_manifest, mock_cluster_config):
        """Explicit HYBRID strategy should be used."""
        config = AutoSyncConfig(strategy=SyncStrategy.HYBRID)

        with patch("app.distributed.cluster_manifest.get_cluster_manifest", return_value=mock_manifest), \
             patch("app.distributed.cluster_manifest.load_cluster_config", return_value=mock_cluster_config), \
             patch("socket.gethostname", return_value="test-node"):
            daemon = AutoSyncDaemon(config=config)

            assert daemon._resolved_strategy == SyncStrategy.HYBRID
            assert daemon._is_ephemeral is False

    def test_explicit_ephemeral_strategy(self, mock_manifest, mock_cluster_config):
        """Explicit EPHEMERAL strategy should be used."""
        config = AutoSyncConfig(strategy=SyncStrategy.EPHEMERAL)

        with patch("app.distributed.cluster_manifest.get_cluster_manifest", return_value=mock_manifest), \
             patch("app.distributed.cluster_manifest.load_cluster_config", return_value=mock_cluster_config), \
             patch("socket.gethostname", return_value="test-node"):
            daemon = AutoSyncDaemon(config=config)

            assert daemon._resolved_strategy == SyncStrategy.EPHEMERAL
            assert daemon._is_ephemeral is True

    def test_explicit_broadcast_strategy(self, mock_manifest, mock_cluster_config):
        """Explicit BROADCAST strategy should be used."""
        config = AutoSyncConfig(strategy=SyncStrategy.BROADCAST)

        with patch("app.distributed.cluster_manifest.get_cluster_manifest", return_value=mock_manifest), \
             patch("app.distributed.cluster_manifest.load_cluster_config", return_value=mock_cluster_config), \
             patch("socket.gethostname", return_value="test-node"):
            daemon = AutoSyncDaemon(config=config)

            assert daemon._resolved_strategy == SyncStrategy.BROADCAST
            assert daemon._is_broadcast is True


# =============================================================================
# Test sync_now
# =============================================================================


class TestSyncNow:
    """Tests for sync_now method."""

    @pytest.mark.asyncio
    async def test_sync_now_when_not_running(self, daemon):
        """sync_now should return 0 when daemon not running."""
        daemon._running = False

        result = await daemon.sync_now()

        assert result == 0

    @pytest.mark.asyncio
    async def test_sync_now_when_running(self, daemon):
        """sync_now should call _sync_cycle when running."""
        daemon._running = True

        with patch.object(daemon, "_sync_cycle", new_callable=AsyncMock) as mock_cycle:
            mock_cycle.return_value = 10  # 10 games synced

            result = await daemon.sync_now()

            assert result == 10
            mock_cycle.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_now_handles_error(self, daemon):
        """sync_now should handle errors gracefully."""
        daemon._running = True

        with patch.object(daemon, "_sync_cycle", new_callable=AsyncMock) as mock_cycle:
            mock_cycle.side_effect = RuntimeError("Sync failed")

            result = await daemon.sync_now()

            assert result == 0


# =============================================================================
# Test Event Integration
# =============================================================================


class TestEventIntegration:
    """Tests for event system integration."""

    def test_subscribe_to_events(self, daemon):
        """_subscribe_to_events should set up subscriptions."""
        with patch("app.coordination.event_router.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_get_router.return_value = mock_router

            daemon._subscribe_to_events()

            # Should have attempted to subscribe
            assert daemon._subscribed or mock_router.subscribe.called


# =============================================================================
# Test Metrics
# =============================================================================


class TestMetrics:
    """Tests for daemon metrics."""

    def test_get_metrics_structure(self, daemon):
        """get_metrics should return protocol-compliant metrics."""
        daemon._start_time = 1000.0

        metrics = daemon.get_metrics()

        assert isinstance(metrics, dict)
        assert "name" in metrics
        assert "status" in metrics
        assert "uptime_seconds" in metrics
        assert "events_processed" in metrics
        assert "errors_count" in metrics

    def test_uptime_seconds(self, daemon):
        """uptime_seconds should calculate correctly."""
        import time
        daemon._start_time = time.time() - 100  # Started 100 seconds ago

        uptime = daemon.uptime_seconds

        assert uptime >= 100
        assert uptime < 110  # Allow some tolerance
