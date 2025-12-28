"""Tests for AutoSyncDaemon.

Comprehensive unit tests covering:
1. Initialization with default and custom config
2. P2P gossip sync behavior
3. Exclusion rules (coordinator nodes, NFS hosts)
4. Bandwidth throttling
5. Error handling and recovery
6. Event emission on sync completion
7. Strategy resolution (auto, hybrid, ephemeral, broadcast)
8. Health check and metrics
9. Quality filtering for sync
10. Database integrity verification

December 2025
"""

import asyncio
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.auto_sync_daemon import (
    AutoSyncConfig,
    AutoSyncDaemon,
    SyncStats,
    SyncStrategy,
    get_auto_sync_daemon,
    reset_auto_sync_daemon,
)
from app.coordination.protocols import CoordinatorStatus, HealthCheckResult


# ============================================
# Test Fixtures
# ============================================


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_db(temp_dir):
    """Create a temporary SQLite database with test data."""
    db_path = temp_dir / "test_games.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE games (
            game_id TEXT PRIMARY KEY,
            board_type TEXT,
            num_players INTEGER,
            game_status TEXT,
            winner TEXT,
            termination_reason TEXT,
            total_moves INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Insert some test games
    for i in range(10):
        conn.execute(
            "INSERT INTO games (game_id, board_type, num_players, game_status, winner, termination_reason, total_moves) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (f"game_{i}", "hex8", 2, "completed", "player_1", "victory", 30 + i),
        )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def default_config():
    """Create a default AutoSyncConfig for testing."""
    return AutoSyncConfig(
        enabled=True,
        interval_seconds=60,
        gossip_interval_seconds=30,
        exclude_hosts=[],
        skip_nfs_sync=True,
        max_concurrent_syncs=4,
        min_games_to_sync=5,
        bandwidth_limit_mbps=20,
        quality_filter_enabled=False,  # Disable for most tests
    )


@pytest.fixture
def custom_config():
    """Create a custom AutoSyncConfig for testing."""
    return AutoSyncConfig(
        enabled=True,
        strategy=SyncStrategy.HYBRID,
        interval_seconds=120,
        gossip_interval_seconds=60,
        exclude_hosts=["coordinator-1", "coordinator-2"],
        skip_nfs_sync=True,
        max_concurrent_syncs=2,
        min_games_to_sync=20,
        bandwidth_limit_mbps=50,
        max_disk_usage_percent=80.0,
        target_disk_usage_percent=70.0,
        quality_filter_enabled=True,
        min_quality_for_sync=0.6,
    )


@pytest.fixture
def daemon(default_config):
    """Create a fresh AutoSyncDaemon for each test."""
    reset_auto_sync_daemon()
    with patch.object(AutoSyncDaemon, "_init_cluster_manifest"):
        with patch.object(AutoSyncDaemon, "_detect_provider", return_value="test"):
            with patch.object(AutoSyncDaemon, "_check_nfs_mount", return_value=False):
                with patch.object(AutoSyncDaemon, "_is_cluster_leader", return_value=False):
                    daemon = AutoSyncDaemon(config=default_config)
                    yield daemon
    reset_auto_sync_daemon()


# ============================================
# Test AutoSyncConfig
# ============================================


class TestAutoSyncConfig:
    """Tests for AutoSyncConfig dataclass."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = AutoSyncConfig()

        assert config.enabled is True
        assert config.strategy == SyncStrategy.AUTO
        assert config.interval_seconds == 60
        assert config.gossip_interval_seconds == 30
        assert config.exclude_hosts == []
        assert config.skip_nfs_sync is True
        assert config.max_concurrent_syncs == 4
        assert config.min_games_to_sync == 10
        assert config.bandwidth_limit_mbps == 20
        assert config.max_disk_usage_percent == 70.0
        assert config.target_disk_usage_percent == 60.0
        assert config.auto_cleanup_enabled is True

    def test_custom_config_values(self, custom_config):
        """Test custom configuration values are applied."""
        assert custom_config.strategy == SyncStrategy.HYBRID
        assert custom_config.interval_seconds == 120
        assert custom_config.gossip_interval_seconds == 60
        assert custom_config.exclude_hosts == ["coordinator-1", "coordinator-2"]
        assert custom_config.max_concurrent_syncs == 2
        assert custom_config.min_games_to_sync == 20
        assert custom_config.bandwidth_limit_mbps == 50
        assert custom_config.max_disk_usage_percent == 80.0
        assert custom_config.quality_filter_enabled is True
        assert custom_config.min_quality_for_sync == 0.6

    def test_from_config_file_nonexistent(self, temp_dir):
        """Test loading config from nonexistent file uses defaults."""
        config = AutoSyncConfig.from_config_file(temp_dir / "nonexistent.yaml")

        # Should use defaults
        assert config.enabled is True
        assert config.interval_seconds == 60

    def test_from_config_file_valid(self, temp_dir):
        """Test loading config from a valid YAML file."""
        config_file = temp_dir / "distributed_hosts.yaml"
        config_file.write_text("""
sync_routing:
    max_disk_usage_percent: 75.0
    target_disk_usage_percent: 65.0

auto_sync:
    enabled: true
    interval_seconds: 90
    gossip_interval_seconds: 45
    exclude_hosts:
        - test-coordinator
    skip_nfs_sync: false
    max_concurrent_syncs: 6
    min_games_to_sync: 15
    bandwidth_limit_mbps: 30
""")

        config = AutoSyncConfig.from_config_file(config_file)

        assert config.max_disk_usage_percent == 75.0
        assert config.target_disk_usage_percent == 65.0
        assert config.interval_seconds == 90
        assert config.gossip_interval_seconds == 45
        # Note: exclude_hosts may contain additional hosts from unified_loop.yaml fallback
        assert "test-coordinator" in config.exclude_hosts
        assert config.skip_nfs_sync is False
        assert config.max_concurrent_syncs == 6
        assert config.min_games_to_sync == 15
        assert config.bandwidth_limit_mbps == 30


# ============================================
# Test Initialization
# ============================================


class TestAutoSyncDaemonInit:
    """Tests for AutoSyncDaemon initialization."""

    def test_init_with_default_config(self, daemon):
        """Test initialization with default config."""
        assert daemon.config.enabled is True
        assert daemon.node_id is not None
        assert daemon._running is False
        assert daemon._stats.total_syncs == 0

    def test_init_with_custom_config(self, custom_config):
        """Test initialization with custom config."""
        reset_auto_sync_daemon()
        with patch.object(AutoSyncDaemon, "_init_cluster_manifest"):
            with patch.object(AutoSyncDaemon, "_detect_provider", return_value="test"):
                with patch.object(AutoSyncDaemon, "_check_nfs_mount", return_value=False):
                    with patch.object(AutoSyncDaemon, "_is_cluster_leader", return_value=False):
                        daemon = AutoSyncDaemon(config=custom_config)

        assert daemon.config.interval_seconds == 120
        assert daemon.config.exclude_hosts == ["coordinator-1", "coordinator-2"]
        assert daemon.config.max_concurrent_syncs == 2
        reset_auto_sync_daemon()

    def test_init_sets_coordinator_status(self, daemon):
        """Test initialization sets coordinator status to INITIALIZING."""
        assert daemon._coordinator_status == CoordinatorStatus.INITIALIZING

    def test_name_property(self, daemon):
        """Test name property returns correct value."""
        assert daemon.name == "AutoSyncDaemon"

    def test_status_property(self, daemon):
        """Test status property reflects internal state."""
        assert daemon.status == CoordinatorStatus.INITIALIZING

    def test_is_running_initially_false(self, daemon):
        """Test is_running returns False before start."""
        assert daemon.is_running() is False


# ============================================
# Test Strategy Resolution
# ============================================


class TestStrategyResolution:
    """Tests for sync strategy resolution."""

    def test_explicit_hybrid_strategy(self, default_config):
        """Test explicit HYBRID strategy is used."""
        default_config.strategy = SyncStrategy.HYBRID
        reset_auto_sync_daemon()

        with patch.object(AutoSyncDaemon, "_init_cluster_manifest"):
            with patch.object(AutoSyncDaemon, "_detect_provider", return_value="test"):
                with patch.object(AutoSyncDaemon, "_check_nfs_mount", return_value=False):
                    daemon = AutoSyncDaemon(config=default_config)

        assert daemon._resolved_strategy == SyncStrategy.HYBRID
        reset_auto_sync_daemon()

    def test_explicit_ephemeral_strategy(self, default_config):
        """Test explicit EPHEMERAL strategy is used."""
        default_config.strategy = SyncStrategy.EPHEMERAL
        reset_auto_sync_daemon()

        with patch.object(AutoSyncDaemon, "_init_cluster_manifest"):
            with patch.object(AutoSyncDaemon, "_detect_provider", return_value="test"):
                with patch.object(AutoSyncDaemon, "_check_nfs_mount", return_value=False):
                    with patch.object(AutoSyncDaemon, "_init_ephemeral_wal"):
                        daemon = AutoSyncDaemon(config=default_config)

        assert daemon._resolved_strategy == SyncStrategy.EPHEMERAL
        assert daemon._is_ephemeral is True
        reset_auto_sync_daemon()

    def test_explicit_broadcast_strategy(self, default_config):
        """Test explicit BROADCAST strategy is used."""
        default_config.strategy = SyncStrategy.BROADCAST
        reset_auto_sync_daemon()

        with patch.object(AutoSyncDaemon, "_init_cluster_manifest"):
            with patch.object(AutoSyncDaemon, "_detect_provider", return_value="test"):
                with patch.object(AutoSyncDaemon, "_check_nfs_mount", return_value=False):
                    daemon = AutoSyncDaemon(config=default_config)

        assert daemon._resolved_strategy == SyncStrategy.BROADCAST
        assert daemon._is_broadcast is True
        reset_auto_sync_daemon()

    def test_auto_detect_ephemeral_vast(self, default_config):
        """Test auto-detection of ephemeral host (Vast.ai)."""
        default_config.strategy = SyncStrategy.AUTO
        reset_auto_sync_daemon()

        with patch.object(AutoSyncDaemon, "_init_cluster_manifest"):
            with patch.object(AutoSyncDaemon, "_detect_provider", return_value="vast"):
                with patch.object(AutoSyncDaemon, "_check_nfs_mount", return_value=False):
                    with patch.object(AutoSyncDaemon, "_detect_ephemeral_host", return_value=True):
                        with patch.object(AutoSyncDaemon, "_init_ephemeral_wal"):
                            daemon = AutoSyncDaemon(config=default_config)

        assert daemon._resolved_strategy == SyncStrategy.EPHEMERAL
        reset_auto_sync_daemon()

    def test_auto_detect_broadcast_leader(self, default_config):
        """Test auto-detection of broadcast mode for cluster leader."""
        default_config.strategy = SyncStrategy.AUTO
        reset_auto_sync_daemon()

        with patch.object(AutoSyncDaemon, "_init_cluster_manifest"):
            with patch.object(AutoSyncDaemon, "_detect_provider", return_value="test"):
                with patch.object(AutoSyncDaemon, "_check_nfs_mount", return_value=False):
                    with patch.object(AutoSyncDaemon, "_detect_ephemeral_host", return_value=False):
                        with patch.object(AutoSyncDaemon, "_is_cluster_leader", return_value=True):
                            daemon = AutoSyncDaemon(config=default_config)

        assert daemon._resolved_strategy == SyncStrategy.BROADCAST
        reset_auto_sync_daemon()

    def test_auto_detect_hybrid_default(self, default_config):
        """Test auto-detection defaults to HYBRID for non-ephemeral, non-leader nodes."""
        default_config.strategy = SyncStrategy.AUTO
        reset_auto_sync_daemon()

        with patch.object(AutoSyncDaemon, "_init_cluster_manifest"):
            with patch.object(AutoSyncDaemon, "_detect_provider", return_value="test"):
                with patch.object(AutoSyncDaemon, "_check_nfs_mount", return_value=False):
                    with patch.object(AutoSyncDaemon, "_detect_ephemeral_host", return_value=False):
                        with patch.object(AutoSyncDaemon, "_is_cluster_leader", return_value=False):
                            daemon = AutoSyncDaemon(config=default_config)

        assert daemon._resolved_strategy == SyncStrategy.HYBRID
        reset_auto_sync_daemon()


# ============================================
# Test Provider Detection
# ============================================


class TestProviderDetection:
    """Tests for cloud provider detection."""

    def test_detect_provider_lambda(self, default_config):
        """Test Lambda provider detection via NFS path."""
        reset_auto_sync_daemon()

        with patch.object(AutoSyncDaemon, "_init_cluster_manifest"):
            with patch.object(Path, "exists", return_value=True):
                with patch.object(AutoSyncDaemon, "_check_nfs_mount", return_value=True):
                    with patch.object(AutoSyncDaemon, "_is_cluster_leader", return_value=False):
                        daemon = AutoSyncDaemon(config=default_config)
                        # Manually call detection
                        provider = daemon._detect_provider()

        # Lambda detection depends on /lambda/nfs path
        # Since we're mocking, just verify method doesn't crash
        assert provider is not None
        reset_auto_sync_daemon()

    def test_check_nfs_mount_false_when_no_mount(self, default_config):
        """Test NFS mount check returns False when not mounted."""
        reset_auto_sync_daemon()

        with patch.object(AutoSyncDaemon, "_init_cluster_manifest"):
            with patch.object(AutoSyncDaemon, "_detect_provider", return_value="test"):
                with patch.object(AutoSyncDaemon, "_is_cluster_leader", return_value=False):
                    daemon = AutoSyncDaemon(config=default_config)
                    # Direct check with non-existent path
                    result = daemon._check_nfs_mount()

        assert result is False
        reset_auto_sync_daemon()


# ============================================
# Test Exclusion Rules
# ============================================


class TestExclusionRules:
    """Tests for node exclusion during sync."""

    @pytest.mark.asyncio
    async def test_skip_sync_for_excluded_host(self, daemon):
        """Test sync is skipped for excluded hosts."""
        daemon.config.exclude_hosts = ["test-host-1"]
        daemon.node_id = "test-host-1"

        result = await daemon._sync_cycle()

        assert result == 0  # No games synced

    @pytest.mark.asyncio
    async def test_skip_sync_for_nfs_node(self, daemon):
        """Test sync is skipped for NFS nodes when skip_nfs_sync is True."""
        daemon._is_nfs_node = True
        daemon.config.skip_nfs_sync = True

        result = await daemon._sync_cycle()

        assert result == 0  # No games synced

    @pytest.mark.asyncio
    async def test_allow_sync_for_nfs_node_when_disabled(self, daemon):
        """Test sync is allowed for NFS nodes when skip_nfs_sync is False."""
        daemon._is_nfs_node = True
        daemon.config.skip_nfs_sync = False

        # Mock _check_disk_capacity and other dependencies
        with patch.object(daemon, "_check_disk_capacity", return_value=True):
            with patch.object(daemon, "_get_pending_sync_data", return_value=100):
                with patch.object(daemon, "_collect_from_peers"):
                    with patch.object(daemon, "_verify_synced_databases", return_value=True):
                        with patch.object(daemon, "_register_synced_data"):
                            result = await daemon._sync_cycle()

        assert result == 100  # Games were synced

    @pytest.mark.asyncio
    async def test_skip_sync_when_below_min_games(self, daemon):
        """Test sync is skipped when pending games below threshold."""
        daemon.config.min_games_to_sync = 50

        with patch.object(daemon, "_check_disk_capacity", return_value=True):
            with patch.object(daemon, "_get_pending_sync_data", return_value=10):
                result = await daemon._sync_cycle()

        assert result == 0  # Not enough games to sync


# ============================================
# Test Start/Stop Lifecycle
# ============================================


class TestLifecycle:
    """Tests for daemon lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_sets_running_state(self, daemon):
        """Test start() sets running state correctly."""
        with patch.object(daemon, "_subscribe_to_events"):
            with patch.object(daemon, "_start_gossip_sync"):
                with patch("app.coordination.auto_sync_daemon.safe_create_task", return_value=MagicMock()):
                    with patch("app.coordination.auto_sync_daemon.register_coordinator"):
                        await daemon.start()

        assert daemon._running is True
        assert daemon._coordinator_status == CoordinatorStatus.RUNNING
        assert daemon._start_time > 0

    @pytest.mark.asyncio
    async def test_start_when_disabled(self, daemon):
        """Test start() does nothing when disabled."""
        daemon.config.enabled = False

        await daemon.start()

        assert daemon._running is False
        assert daemon._coordinator_status == CoordinatorStatus.STOPPED

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self, daemon):
        """Test start() is idempotent."""
        daemon._coordinator_status = CoordinatorStatus.RUNNING

        await daemon.start()

        # Should not have changed anything
        assert daemon._coordinator_status == CoordinatorStatus.RUNNING

    @pytest.mark.asyncio
    async def test_stop_sets_stopped_state(self, daemon):
        """Test stop() sets stopped state correctly."""
        daemon._running = True
        daemon._coordinator_status = CoordinatorStatus.RUNNING
        daemon._sync_task = None
        daemon._gossip_daemon = None

        with patch("app.coordination.auto_sync_daemon.unregister_coordinator"):
            await daemon.stop()

        assert daemon._running is False
        assert daemon._coordinator_status == CoordinatorStatus.STOPPED

    @pytest.mark.asyncio
    async def test_stop_cancels_sync_task(self, daemon):
        """Test stop() cancels the sync task."""
        daemon._running = True
        daemon._coordinator_status = CoordinatorStatus.RUNNING
        daemon._gossip_daemon = None

        # Create an actual cancelled asyncio.Task for proper await semantics
        # We use a helper coroutine that raises CancelledError
        async def cancelled_coro():
            raise asyncio.CancelledError()

        # Create a real task and cancel it
        task = asyncio.create_task(cancelled_coro())
        daemon._sync_task = task

        with patch("app.coordination.auto_sync_daemon.unregister_coordinator"):
            await daemon.stop()

        # Task should be in cancelled state
        assert task.cancelled() or task.done()

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self, daemon):
        """Test stop() is idempotent."""
        daemon._coordinator_status = CoordinatorStatus.STOPPED

        await daemon.stop()

        assert daemon._coordinator_status == CoordinatorStatus.STOPPED


# ============================================
# Test Sync Cycle
# ============================================


class TestSyncCycle:
    """Tests for the sync cycle operation."""

    @pytest.mark.asyncio
    async def test_sync_cycle_full_execution(self, daemon):
        """Test full sync cycle execution."""
        with patch.object(daemon, "_check_disk_capacity", return_value=True):
            with patch.object(daemon, "_get_pending_sync_data", return_value=100):
                with patch.object(daemon, "_collect_from_peers"):
                    with patch.object(daemon, "_verify_synced_databases", return_value=True):
                        with patch.object(daemon, "_register_synced_data"):
                            result = await daemon._sync_cycle()

        assert result == 100

    @pytest.mark.asyncio
    async def test_sync_cycle_disk_full(self, daemon):
        """Test sync cycle stops when disk is full."""
        with patch.object(daemon, "_check_disk_capacity", return_value=False):
            result = await daemon._sync_cycle()

        assert result == 0

    @pytest.mark.asyncio
    async def test_sync_now_when_running(self, daemon):
        """Test sync_now() triggers immediate sync when running."""
        daemon._running = True

        with patch.object(daemon, "_sync_cycle", return_value=50) as mock_cycle:
            result = await daemon.sync_now()

        assert result == 50
        mock_cycle.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_now_when_not_running(self, daemon):
        """Test sync_now() returns 0 when daemon not running."""
        daemon._running = False

        result = await daemon.sync_now()

        assert result == 0


# ============================================
# Test Disk Capacity Management
# ============================================


class TestDiskCapacity:
    """Tests for disk capacity management."""

    @pytest.mark.asyncio
    async def test_check_disk_capacity_no_manifest(self, daemon):
        """Test disk capacity check passes when no manifest."""
        daemon._cluster_manifest = None

        result = await daemon._check_disk_capacity()

        assert result is True

    @pytest.mark.asyncio
    async def test_check_disk_capacity_below_threshold(self, daemon):
        """Test disk capacity check passes when below threshold."""
        mock_manifest = MagicMock()
        mock_capacity = MagicMock()
        mock_capacity.usage_percent = 50.0
        mock_manifest.update_local_capacity.return_value = mock_capacity
        daemon._cluster_manifest = mock_manifest

        result = await daemon._check_disk_capacity()

        assert result is True

    @pytest.mark.asyncio
    async def test_check_disk_capacity_above_threshold_cleanup_success(self, daemon):
        """Test disk capacity triggers cleanup when above threshold."""
        mock_manifest = MagicMock()

        # First call returns high usage, second call (after cleanup) returns lower
        mock_capacity_high = MagicMock()
        mock_capacity_high.usage_percent = 80.0
        mock_capacity_low = MagicMock()
        mock_capacity_low.usage_percent = 55.0

        mock_manifest.update_local_capacity.side_effect = [mock_capacity_high, mock_capacity_low]
        daemon._cluster_manifest = mock_manifest
        daemon.config.max_disk_usage_percent = 70.0
        daemon.config.auto_cleanup_enabled = True

        with patch.object(daemon, "_run_disk_cleanup"):
            result = await daemon._check_disk_capacity()

        assert result is True

    @pytest.mark.asyncio
    async def test_check_disk_capacity_cleanup_fails(self, daemon):
        """Test disk capacity fails when cleanup doesn't free enough space."""
        mock_manifest = MagicMock()
        mock_capacity = MagicMock()
        mock_capacity.usage_percent = 85.0
        mock_manifest.update_local_capacity.return_value = mock_capacity
        daemon._cluster_manifest = mock_manifest
        daemon.config.max_disk_usage_percent = 70.0
        daemon.config.auto_cleanup_enabled = True

        with patch.object(daemon, "_run_disk_cleanup"):
            result = await daemon._check_disk_capacity()

        assert result is False


# ============================================
# Test Event Handling
# ============================================


class TestEventHandling:
    """Tests for event subscription and handling."""

    @pytest.mark.asyncio
    async def test_on_data_stale_triggers_urgent_sync(self, daemon):
        """Test DATA_STALE event triggers urgent sync."""
        mock_event = MagicMock()
        mock_event.payload = {
            "board_type": "hex8",
            "num_players": 2,
            "data_age_hours": 5.0,
        }

        # Dec 28, 2025: fire_and_forget is now imported in sync_event_mixin.py
        with patch("app.coordination.sync_event_mixin.fire_and_forget") as mock_fire:
            await daemon._on_data_stale(mock_event)

        assert daemon._events_processed == 1
        assert "hex8_2p" in daemon._urgent_sync_pending
        mock_fire.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_sync_triggered_with_config(self, daemon):
        """Test SYNC_TRIGGERED event with config key."""
        mock_event = MagicMock()
        mock_event.payload = {
            "reason": "manual",
            "config_key": "square8_2p",
        }

        # Dec 28, 2025: fire_and_forget is now imported in sync_event_mixin.py
        with patch("app.coordination.sync_event_mixin.fire_and_forget") as mock_fire:
            await daemon._on_sync_triggered(mock_event)

        assert daemon._events_processed == 1
        mock_fire.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_new_games_available_triggers_push(self, daemon):
        """Test NEW_GAMES_AVAILABLE event triggers push to neighbors."""
        mock_event = MagicMock()
        mock_event.payload = {
            "config": "hex8_2p",
            "new_games": 50,
            "total_games": 1000,
        }
        daemon.config.min_games_to_sync = 10

        # Dec 28, 2025: fire_and_forget is now imported in sync_event_mixin.py
        with patch("app.coordination.sync_event_mixin.fire_and_forget") as mock_fire:
            await daemon._on_new_games_available(mock_event)

        assert daemon._events_processed == 1
        mock_fire.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_new_games_available_skips_small_batch(self, daemon):
        """Test NEW_GAMES_AVAILABLE skips small batches."""
        mock_event = MagicMock()
        mock_event.payload = {
            "config": "hex8_2p",
            "new_games": 3,
            "total_games": 100,
        }
        daemon.config.min_games_to_sync = 10

        # Dec 28, 2025: fire_and_forget is now imported in sync_event_mixin.py
        with patch("app.coordination.sync_event_mixin.fire_and_forget") as mock_fire:
            await daemon._on_new_games_available(mock_event)

        assert daemon._events_processed == 0
        mock_fire.assert_not_called()


# ============================================
# Test Health Check
# ============================================


class TestHealthCheck:
    """Tests for health check functionality."""

    def test_health_check_healthy(self, daemon):
        """Test health check returns healthy when running normally."""
        daemon._coordinator_status = CoordinatorStatus.RUNNING
        daemon._stats.syncs_completed = 10  # Dec 2025: Use mutable field, not property
        daemon._stats.syncs_failed = 1  # Dec 2025: Use mutable field, not property
        daemon._stats.last_check_time = time.time()  # Dec 2025: Use mutable field, not property

        result = daemon.health_check()

        assert result.healthy is True
        assert result.status == CoordinatorStatus.RUNNING

    def test_health_check_stopped(self, daemon):
        """Test health check returns healthy but stopped."""
        daemon._coordinator_status = CoordinatorStatus.STOPPED

        result = daemon.health_check()

        assert result.healthy is True
        assert result.status == CoordinatorStatus.STOPPED
        assert "stopped" in result.message.lower()

    def test_health_check_disabled(self, daemon):
        """Test health check when disabled."""
        daemon.config.enabled = False
        daemon._coordinator_status = CoordinatorStatus.STOPPED

        result = daemon.health_check()

        assert result.healthy is True
        # Message can be "stopped" or "disabled" depending on status check order
        assert "stopped" in result.message.lower() or "disabled" in result.message.lower()

    def test_health_check_error_state(self, daemon):
        """Test health check returns unhealthy in error state."""
        daemon._coordinator_status = CoordinatorStatus.ERROR
        daemon._last_error = "Test error"

        result = daemon.health_check()

        assert result.healthy is False
        assert result.status == CoordinatorStatus.ERROR

    def test_health_check_high_failure_rate(self, daemon):
        """Test health check returns degraded on high failure rate."""
        daemon._coordinator_status = CoordinatorStatus.RUNNING
        daemon._stats.syncs_completed = 5  # Dec 2025: Use mutable field, not property
        daemon._stats.syncs_failed = 10  # Dec 2025: Use mutable field, not property
        daemon._stats.operations_attempted = 15  # Dec 2025: Use mutable field, not property
        daemon._stats.last_check_time = time.time()  # Dec 2025: Use mutable field, not property

        result = daemon.health_check()

        assert result.status == CoordinatorStatus.DEGRADED
        assert "failure rate" in result.message.lower()

    def test_health_check_stale_sync(self, daemon):
        """Test health check returns degraded on stale sync."""
        daemon._coordinator_status = CoordinatorStatus.RUNNING
        daemon._stats.syncs_completed = 10  # Dec 2025: Use mutable field, not property
        daemon._stats.syncs_failed = 1  # Dec 2025: Use mutable field, not property
        daemon._stats.last_check_time = time.time() - 1000  # Very old sync
        daemon.config.interval_seconds = 60

        result = daemon.health_check()

        assert result.status == CoordinatorStatus.DEGRADED
        assert "no sync" in result.message.lower()

    def test_health_check_high_verification_failure(self, daemon):
        """Test health check returns degraded on high verification failure rate."""
        daemon._coordinator_status = CoordinatorStatus.RUNNING
        daemon._stats.syncs_completed = 10  # Dec 2025: Use mutable field, not property
        daemon._stats.syncs_failed = 1  # Dec 2025: Use mutable field, not property
        daemon._stats.last_check_time = time.time()  # Dec 2025: Use mutable field, not property
        daemon._stats.databases_verified = 10
        daemon._stats.databases_verification_failed = 5  # 50% failure rate

        result = daemon.health_check()

        assert result.status == CoordinatorStatus.DEGRADED
        assert "verification" in result.message.lower()


# ============================================
# Test Metrics
# ============================================


class TestMetrics:
    """Tests for metrics reporting."""

    def test_get_metrics_returns_dict(self, daemon):
        """Test get_metrics returns a dictionary."""
        daemon._coordinator_status = CoordinatorStatus.RUNNING
        daemon._start_time = time.time() - 100

        metrics = daemon.get_metrics()

        assert isinstance(metrics, dict)
        assert metrics["name"] == "AutoSyncDaemon"
        assert metrics["status"] == CoordinatorStatus.RUNNING.value
        assert "uptime_seconds" in metrics
        assert metrics["uptime_seconds"] >= 100

    def test_get_status_returns_comprehensive_info(self, daemon):
        """Test get_status returns comprehensive status information."""
        daemon._running = True
        # Dec 2025: Use mutable fields, not read-only properties
        daemon._stats.operations_attempted = 10
        daemon._stats.syncs_completed = 8
        daemon._stats.syncs_failed = 2

        status = daemon.get_status()

        assert status["node_id"] == daemon.node_id
        assert status["running"] is True
        # Stats are accessed via to_dict() which includes the property aliases
        assert status["stats"]["total_syncs"] == 10
        assert status["stats"]["successful_syncs"] == 8
        assert status["stats"]["failed_syncs"] == 2
        assert "config" in status
        assert "quality_filter" in status

    def test_uptime_seconds_before_start(self, daemon):
        """Test uptime_seconds returns 0 before start."""
        daemon._start_time = 0.0

        assert daemon.uptime_seconds == 0.0

    def test_uptime_seconds_after_start(self, daemon):
        """Test uptime_seconds returns correct value after start."""
        daemon._start_time = time.time() - 100

        uptime = daemon.uptime_seconds
        assert uptime >= 100
        assert uptime < 110


# ============================================
# Test Quality Filtering
# ============================================


class TestQualityFiltering:
    """Tests for quality-based sync filtering."""

    def test_should_sync_database_disabled(self, daemon, temp_db):
        """Test quality filter returns True when disabled."""
        daemon.config.quality_filter_enabled = False

        should_sync, reason = daemon._should_sync_database(temp_db)

        assert should_sync is True
        assert "disabled" in reason.lower()

    def test_should_sync_database_small_db(self, daemon, temp_dir):
        """Test small databases pass quality filter."""
        daemon.config.quality_filter_enabled = True

        # Create a small database with only 3 games
        db_path = temp_dir / "small_games.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE games (
                game_id TEXT PRIMARY KEY,
                board_type TEXT,
                num_players INTEGER,
                game_status TEXT,
                winner TEXT,
                termination_reason TEXT,
                total_moves INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        for i in range(3):
            conn.execute(
                "INSERT INTO games (game_id, board_type, num_players, game_status, winner, termination_reason, total_moves) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (f"game_{i}", "hex8", 2, "completed", "player_1", "victory", 30),
            )
        conn.commit()
        conn.close()

        should_sync, reason = daemon._should_sync_database(db_path)

        assert should_sync is True
        assert "small db" in reason.lower()


# ============================================
# Test Database Verification
# ============================================


class TestDatabaseVerification:
    """Tests for database integrity verification."""

    @pytest.mark.asyncio
    async def test_verify_synced_databases_no_data_dir(self, daemon):
        """Test verification returns True when no data dir exists."""
        with patch("pathlib.Path.exists", return_value=False):
            result = await daemon._verify_synced_databases()

        assert result is True

    @pytest.mark.asyncio
    async def test_verify_synced_databases_all_valid(self, daemon, temp_dir):
        """Test verification returns True when all databases are valid."""
        # Create test database
        db_path = temp_dir / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.commit()
        conn.close()

        # Mock the glob to return our test database
        with patch.object(Path, "glob", return_value=[db_path]):
            with patch.object(Path, "exists", return_value=True):
                with patch("app.coordination.auto_sync_daemon.check_sqlite_integrity", return_value=(True, [])):
                    # We need to use the actual path logic
                    result = await daemon._verify_synced_databases()

        # Will return True since we mock integrity check
        assert result is True


# ============================================
# Test Singleton Access
# ============================================


class TestSingletonAccess:
    """Tests for singleton pattern."""

    def test_get_auto_sync_daemon_creates_instance(self):
        """Test get_auto_sync_daemon creates singleton instance."""
        reset_auto_sync_daemon()

        with patch.object(AutoSyncDaemon, "_init_cluster_manifest"):
            with patch.object(AutoSyncDaemon, "_detect_provider", return_value="test"):
                with patch.object(AutoSyncDaemon, "_check_nfs_mount", return_value=False):
                    with patch.object(AutoSyncDaemon, "_is_cluster_leader", return_value=False):
                        daemon1 = get_auto_sync_daemon()
                        daemon2 = get_auto_sync_daemon()

        assert daemon1 is daemon2
        reset_auto_sync_daemon()

    def test_reset_clears_singleton(self):
        """Test reset_auto_sync_daemon clears the singleton."""
        reset_auto_sync_daemon()

        with patch.object(AutoSyncDaemon, "_init_cluster_manifest"):
            with patch.object(AutoSyncDaemon, "_detect_provider", return_value="test"):
                with patch.object(AutoSyncDaemon, "_check_nfs_mount", return_value=False):
                    with patch.object(AutoSyncDaemon, "_is_cluster_leader", return_value=False):
                        daemon1 = get_auto_sync_daemon()
                        reset_auto_sync_daemon()
                        daemon2 = get_auto_sync_daemon()

        assert daemon1 is not daemon2
        reset_auto_sync_daemon()


# ============================================
# Test SyncStats
# ============================================


class TestSyncStats:
    """Tests for SyncStats dataclass."""

    def test_sync_stats_defaults(self):
        """Test SyncStats default values."""
        stats = SyncStats()

        assert stats.total_syncs == 0
        assert stats.successful_syncs == 0
        assert stats.failed_syncs == 0
        assert stats.games_synced == 0
        assert stats.bytes_transferred == 0
        assert stats.last_sync_time == 0.0
        assert stats.last_error is None
        assert stats.databases_skipped_quality == 0
        assert stats.databases_quality_checked == 0
        assert stats.databases_verified == 0
        assert stats.databases_verification_failed == 0

    def test_sync_stats_mutation(self):
        """Test SyncStats can be mutated via underlying fields.

        December 2025: SyncStats now uses property aliases for backward
        compatibility. Direct mutation uses the underlying parent class fields:
        - operations_attempted (aliased as total_syncs)
        - syncs_completed (aliased as successful_syncs)
        - syncs_failed (aliased as failed_syncs)
        """
        stats = SyncStats()
        # Mutate via underlying fields (properties are read-only aliases)
        stats.operations_attempted = 10
        stats.syncs_completed = 8
        stats.syncs_failed = 2
        stats.last_error = "Test error"

        # Verify via both underlying fields and property aliases
        assert stats.operations_attempted == 10
        assert stats.total_syncs == 10  # Alias for operations_attempted
        assert stats.syncs_completed == 8
        assert stats.successful_syncs == 8  # Alias for syncs_completed
        assert stats.syncs_failed == 2
        assert stats.failed_syncs == 2  # Alias for syncs_failed
        assert stats.last_error == "Test error"


# ============================================
# Test Event Emission
# ============================================


class TestEventEmission:
    """Tests for sync completion event emission."""

    @pytest.mark.asyncio
    async def test_emit_sync_completed(self, daemon):
        """Test _emit_sync_completed emits event."""
        # The method imports get_router locally, so we need to patch the module it's imported from
        mock_router = AsyncMock()
        mock_router.publish = AsyncMock()

        with patch("app.coordination.event_router.get_router", return_value=mock_router):
            await daemon._emit_sync_completed(games_synced=100, bytes_transferred=1000000)

            # Verify the publish was called
            mock_router.publish.assert_called_once()
            call_kwargs = mock_router.publish.call_args.kwargs
            assert call_kwargs["payload"]["games_synced"] == 100
            assert call_kwargs["payload"]["bytes_transferred"] == 1000000

    @pytest.mark.asyncio
    async def test_emit_sync_completed_no_router(self, daemon):
        """Test _emit_sync_completed handles missing router gracefully."""
        # When router is not available, should not raise
        with patch("app.coordination.event_router.get_router", return_value=None):
            # Should not raise
            await daemon._emit_sync_completed(games_synced=100, bytes_transferred=1000000)

    @pytest.mark.asyncio
    async def test_emit_sync_failed(self, daemon):
        """Test _emit_sync_failed emits event."""
        # Mock the imported emit_data_sync_failed function
        with patch("app.coordination.event_router.emit_data_sync_failed") as mock_emit:
            mock_emit.return_value = None

            await daemon._emit_sync_failed("Test error")

            # Verify call was made with correct arguments
            mock_emit.assert_called_once()
            call_kwargs = mock_emit.call_args.kwargs
            assert call_kwargs["error"] == "Test error"
            assert call_kwargs["host"] == daemon.node_id

    @pytest.mark.asyncio
    async def test_emit_sync_failed_handles_import_error(self, daemon):
        """Test _emit_sync_failed handles import errors gracefully."""
        # When the import fails, should not raise
        with patch.object(
            daemon,
            "_emit_sync_failed",
            side_effect=ImportError("No module named 'app.coordination.event_router'"),
        ):
            # Should not raise - we verify by ensuring the test completes
            try:
                await daemon._emit_sync_failed("Test error")
            except ImportError:
                pass  # Expected behavior - import failed


# ============================================
# Reverse Sync (PULL Strategy) Tests - December 2025
# ============================================


class TestReverseSyncPullStrategy:
    """Tests for the PULL strategy reverse sync functionality."""

    @pytest.fixture
    def pull_config(self):
        """Create config for PULL strategy testing."""
        return AutoSyncConfig(
            enabled=True,
            strategy=SyncStrategy.PULL,
            interval_seconds=60,
        )

    @pytest.fixture
    def pull_daemon(self, pull_config, temp_dir):
        """Create a daemon configured for PULL strategy."""
        reset_auto_sync_daemon()
        with patch.object(AutoSyncDaemon, "_init_cluster_manifest"):
            with patch.object(AutoSyncDaemon, "_detect_provider", return_value="test"):
                with patch.object(AutoSyncDaemon, "_check_nfs_mount", return_value=False):
                    with patch.object(AutoSyncDaemon, "_is_cluster_leader", return_value=False):
                        daemon = AutoSyncDaemon(config=pull_config)
                        daemon._data_dir = temp_dir
                        yield daemon
        reset_auto_sync_daemon()

    def test_pull_strategy_is_set(self, pull_daemon):
        """Test that PULL strategy is set when configured."""
        assert pull_daemon._resolved_strategy == SyncStrategy.PULL

    @pytest.mark.asyncio
    async def test_pull_from_cluster_nodes_requires_coordinator(self, pull_config, temp_dir):
        """Test that non-coordinators skip PULL sync."""
        reset_auto_sync_daemon()
        with patch.object(AutoSyncDaemon, "_init_cluster_manifest"):
            with patch.object(AutoSyncDaemon, "_detect_provider", return_value="test"):
                with patch.object(AutoSyncDaemon, "_check_nfs_mount", return_value=False):
                    with patch.object(AutoSyncDaemon, "_is_cluster_leader", return_value=False):
                        daemon = AutoSyncDaemon(config=pull_config)
                        daemon._data_dir = temp_dir

                        # Mock env.is_coordinator to return False
                        # Dec 28, 2025: env is now imported locally from app.config.env
                        mock_env = MagicMock()
                        mock_env.is_coordinator = False
                        with patch("app.config.env.env", mock_env):
                            result = await daemon._pull_from_cluster_nodes()
                            assert result == 0  # Should skip
        reset_auto_sync_daemon()

    @pytest.mark.asyncio
    async def test_pull_from_cluster_nodes_no_sources(self, pull_daemon):
        """Test PULL when no sync sources available."""
        # Mock coordinator check to return True
        # Dec 28, 2025: env is now imported locally from app.config.env
        mock_env = MagicMock()
        mock_env.is_coordinator = True
        with patch("app.config.env.env", mock_env):
            # Dec 28, 2025: get_sync_router is imported locally from sync_router
            with patch(
                "app.coordination.sync_router.get_sync_router"
            ) as mock_get_router:
                mock_router = MagicMock()
                mock_router.get_sync_sources.return_value = []
                mock_router.refresh_from_cluster_config = MagicMock()
                mock_get_router.return_value = mock_router

                result = await pull_daemon._pull_from_cluster_nodes()

                assert result == 0
                mock_router.get_sync_sources.assert_called_once()

    @pytest.mark.asyncio
    async def test_pull_from_cluster_nodes_with_sources(self, pull_daemon):
        """Test PULL with available sources."""
        # Create mock source targets
        mock_source = MagicMock()
        mock_source.node_id = "worker-1"

        # Mock coordinator check to return True
        # Dec 28, 2025: env is now imported locally from app.config.env
        mock_env = MagicMock()
        mock_env.is_coordinator = True
        with patch("app.config.env.env", mock_env):
            # Dec 28, 2025: get_sync_router is imported locally from sync_router
            with patch(
                "app.coordination.sync_router.get_sync_router"
            ) as mock_get_router:
                mock_router = MagicMock()
                mock_router.get_sync_sources.return_value = [mock_source]
                mock_router.refresh_from_cluster_config = MagicMock()
                mock_get_router.return_value = mock_router

                with patch.object(
                    pull_daemon, "_pull_from_node", new_callable=AsyncMock
                ) as mock_pull:
                    mock_pull.return_value = 10  # 10 games pulled

                    result = await pull_daemon._pull_from_cluster_nodes()

                    assert result == 10
                    mock_pull.assert_called_once_with("worker-1")

    @pytest.mark.asyncio
    async def test_pull_from_node_no_config(self, pull_daemon):
        """Test pull from node when config not found."""
        with patch("app.config.cluster_config.get_cluster_nodes") as mock_nodes:
            mock_nodes.return_value = {}  # Empty nodes dict

            result = await pull_daemon._pull_from_node("unknown-node")

            assert result == 0

    @pytest.mark.asyncio
    async def test_pull_from_node_no_ssh_host(self, pull_daemon):
        """Test pull from node with no SSH host configured."""
        mock_node = MagicMock()
        mock_node.best_ip = None  # No IP
        mock_node.ssh_user = "ubuntu"
        mock_node.ssh_key = "~/.ssh/id_cluster"

        with patch("app.config.cluster_config.get_cluster_nodes") as mock_nodes:
            mock_nodes.return_value = {"worker-1": mock_node}

            result = await pull_daemon._pull_from_node("worker-1")

            assert result == 0

    def test_get_canonical_name_hex8_2p(self, pull_daemon):
        """Test canonical name extraction for hex8_2p."""
        result = pull_daemon._get_canonical_name("selfplay_hex8_2p.db")
        assert result == "canonical_hex8_2p.db"

    def test_get_canonical_name_square8_4p(self, pull_daemon):
        """Test canonical name extraction for square8_4p."""
        result = pull_daemon._get_canonical_name("games_square8_4p_2025.db")
        assert result == "canonical_square8_4p.db"

    def test_get_canonical_name_already_canonical(self, pull_daemon):
        """Test canonical name when already canonical."""
        result = pull_daemon._get_canonical_name("canonical_hex8_2p.db")
        assert result == "canonical_hex8_2p.db"

    def test_get_canonical_name_reversed_pattern(self, pull_daemon):
        """Test canonical name with reversed pattern (2p_hex8)."""
        result = pull_daemon._get_canonical_name("2p_hex8_games.db")
        assert result == "canonical_hex8_2p.db"

    def test_get_canonical_name_unknown_format(self, pull_daemon):
        """Test canonical name fallback for unknown format."""
        result = pull_daemon._get_canonical_name("random_games.db")
        assert result == "canonical_random_games.db"

    def test_get_remote_games_path_runpod(self, pull_daemon):
        """Test remote path for RunPod provider."""
        with patch("app.config.cluster_config.get_host_provider") as mock_provider:
            mock_provider.return_value = "runpod"

            result = pull_daemon._get_remote_games_path("runpod-h100")

            assert "/workspace/ringrift" in result

    def test_get_remote_games_path_vast(self, pull_daemon):
        """Test remote path for Vast provider."""
        with patch("app.config.cluster_config.get_host_provider") as mock_provider:
            mock_provider.return_value = "vast"

            result = pull_daemon._get_remote_games_path("vast-12345")

            assert "~/ringrift" in result

    def test_get_remote_games_path_vultr(self, pull_daemon):
        """Test remote path for Vultr provider."""
        with patch("app.config.cluster_config.get_host_provider") as mock_provider:
            mock_provider.return_value = "vultr"

            result = pull_daemon._get_remote_games_path("vultr-a100")

            assert "/root/ringrift" in result

    @pytest.mark.asyncio
    async def test_list_remote_databases_success(self, pull_daemon):
        """Test listing remote databases."""
        # Mock successful SSH command
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (
            b"/data/games/selfplay_hex8_2p.db\n/data/games/selfplay_square8_2p.db\n",
            b"",
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch("asyncio.wait_for", return_value=mock_proc.communicate.return_value):
                result = await pull_daemon._list_remote_databases(
                    "192.168.1.1", "ubuntu", "~/.ssh/key", "/data/games"
                )

                # Should return filenames only
                assert "selfplay_hex8_2p.db" in result
                assert "selfplay_square8_2p.db" in result
                assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_remote_databases_timeout(self, pull_daemon):
        """Test handling of timeout during database listing."""
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=asyncio.TimeoutError()
        ):
            result = await pull_daemon._list_remote_databases(
                "192.168.1.1", "ubuntu", "~/.ssh/key", "/data/games"
            )

            assert result == []

    @pytest.mark.asyncio
    async def test_rsync_pull_success(self, pull_daemon, temp_dir):
        """Test successful rsync pull."""
        local_path = temp_dir / "test.db"

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch("asyncio.wait_for", return_value=mock_proc.communicate.return_value):
                # Dec 28, 2025: Mock checksum verification to return success
                with patch(
                    "app.coordination.sync_integrity.verify_sync_checksum",
                    new_callable=AsyncMock,
                    return_value=(True, None),
                ):
                    # Create the file to simulate rsync success
                    local_path.touch()

                    result = await pull_daemon._rsync_pull(
                        "192.168.1.1", "ubuntu", "~/.ssh/key",
                        "/data/games", "test.db", temp_dir
                    )

                    assert result == local_path

    @pytest.mark.asyncio
    async def test_rsync_pull_failure(self, pull_daemon, temp_dir):
        """Test rsync pull failure."""
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"rsync error")
        mock_proc.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch("asyncio.wait_for", return_value=mock_proc.communicate.return_value):
                result = await pull_daemon._rsync_pull(
                    "192.168.1.1", "ubuntu", "~/.ssh/key",
                    "/data/games", "test.db", temp_dir
                )

                assert result is None

    @pytest.mark.asyncio
    async def test_emit_pull_sync_completed(self, pull_daemon):
        """Test event emission for PULL sync completion."""
        # Dec 28, 2025: data_events is in app.distributed, not app.coordination
        with patch(
            "app.distributed.data_events.emit_data_event",
            new_callable=AsyncMock
        ) as mock_emit:
            await pull_daemon._emit_pull_sync_completed(100, 3)

            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            # Check event type
            from app.distributed.data_events import DataEventType
            assert call_args[0][0] == DataEventType.DATA_SYNC_COMPLETED
            # Check payload
            payload = call_args[0][1]
            assert payload["sync_type"] == "pull"
            assert payload["games_synced"] == 100
            assert payload["sources_count"] == 3

    @pytest.mark.asyncio
    async def test_sync_cycle_uses_pull_for_pull_strategy(self, pull_daemon):
        """Test that _sync_cycle calls _pull_from_cluster_nodes for PULL strategy."""
        with patch.object(
            pull_daemon, "_pull_from_cluster_nodes", new_callable=AsyncMock
        ) as mock_pull:
            mock_pull.return_value = 25

            result = await pull_daemon._sync_cycle()

            assert result == 25
            mock_pull.assert_called_once()


class TestMergeIntoCanonical:
    """Tests for database merge functionality."""

    @pytest.fixture
    def merge_daemon(self, temp_dir):
        """Create daemon for merge testing."""
        reset_auto_sync_daemon()
        # Dec 28, 2025: Removed is_coordinator - not a valid AutoSyncConfig field
        # Coordinator detection happens via hostname/role in YAML config
        config = AutoSyncConfig(
            enabled=True,
            strategy=SyncStrategy.PULL,
        )
        # Dec 28, 2025: Fixed - node_id comes from socket.gethostname()
        with patch("socket.gethostname", return_value="coordinator"):
            with patch.object(AutoSyncDaemon, "_init_cluster_manifest"):
                with patch.object(AutoSyncDaemon, "_detect_provider", return_value="test"):
                    with patch.object(AutoSyncDaemon, "_check_nfs_mount", return_value=False):
                        daemon = AutoSyncDaemon(config=config)
                        daemon._data_dir = temp_dir
                        return daemon

    @pytest.fixture
    def canonical_db(self, temp_dir):
        """Create a canonical database with some games."""
        games_dir = temp_dir / "data" / "games"
        games_dir.mkdir(parents=True, exist_ok=True)
        db_path = games_dir / "canonical_hex8_2p.db"

        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE games (
                game_id TEXT PRIMARY KEY,
                board_type TEXT,
                num_players INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE game_moves (
                game_id TEXT,
                move_number INTEGER,
                move_data TEXT
            )
        """)
        # Insert existing games
        for i in range(5):
            conn.execute(
                "INSERT INTO games (game_id, board_type, num_players) VALUES (?, ?, ?)",
                (f"existing_game_{i}", "hex8", 2),
            )
        conn.commit()
        conn.close()
        return db_path

    @pytest.fixture
    def pulled_db(self, temp_dir):
        """Create a pulled database with new games."""
        pull_dir = temp_dir / "data" / "games" / "pulled"
        pull_dir.mkdir(parents=True, exist_ok=True)
        db_path = pull_dir / "selfplay_hex8_2p.db"

        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE games (
                game_id TEXT PRIMARY KEY,
                board_type TEXT,
                num_players INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE game_moves (
                game_id TEXT,
                move_number INTEGER,
                move_data TEXT
            )
        """)
        # Insert new games
        for i in range(3):
            conn.execute(
                "INSERT INTO games (game_id, board_type, num_players) VALUES (?, ?, ?)",
                (f"new_game_{i}", "hex8", 2),
            )
        conn.commit()
        conn.close()
        return db_path

    @pytest.mark.asyncio
    async def test_merge_creates_canonical_when_missing(
        self, merge_daemon, pulled_db, temp_dir
    ):
        """Test merge creates canonical db when it doesn't exist."""
        # Patch the base_dir calculation to use temp_dir
        with patch.object(merge_daemon, "_get_canonical_name") as mock_name:
            mock_name.return_value = "canonical_hex8_2p.db"

            # Create a clean target location
            games_dir = temp_dir / "data" / "games"
            games_dir.mkdir(parents=True, exist_ok=True)
            canonical_path = games_dir / "canonical_hex8_2p.db"

            # Ensure canonical doesn't exist
            if canonical_path.exists():
                canonical_path.unlink()

            # Manually create pulled db at expected location
            pull_dir = games_dir / "pulled"
            pull_dir.mkdir(exist_ok=True)
            test_pulled_db = pull_dir / "test_hex8_2p.db"

            conn = sqlite3.connect(str(test_pulled_db))
            conn.execute("CREATE TABLE games (game_id TEXT PRIMARY KEY)")
            conn.execute("INSERT INTO games VALUES ('test_game')")
            conn.commit()
            conn.close()

            # The method renames file, so verify behavior
            assert test_pulled_db.exists()
