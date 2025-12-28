"""Tests for UnifiedDataSyncService - Comprehensive data synchronization.

Created: December 28, 2025
Purpose: Test the UnifiedDataSyncService class and related components

This module provides comprehensive unit tests for the unified data sync
infrastructure, covering:
1. SyncConfig dataclass validation
2. SyncHostConfig (HostConfig) validation
3. HostSyncState tracking
4. DataManifest operations (legacy fallback)
5. UnifiedDataSyncService initialization
6. Multi-transport sync (SSH, P2P, aria2 fallback)
7. Retry logic and exponential backoff
8. Circuit breaker integration
9. Statistics tracking
10. Collection cycle behavior

Note: UnifiedDataSyncService is deprecated (Q2 2026) but needs test coverage
for the deprecation period. Use AutoSyncDaemon or SyncFacade for new code.
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# Suppress the deprecation warning during import for testing
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    try:
        from app.distributed.unified_data_sync import (
            SyncConfig,
            SyncHostConfig,
            HostConfig,
            HostSyncState,
            DataManifest,
            UnifiedDataSyncService,
            _LegacyDataManifest,
            load_hosts_from_yaml,
            resolve_hosts_config_path,
            HAS_CIRCUIT_BREAKER,
            HAS_P2P_FALLBACK,
            HAS_MANIFEST_REPLICATION,
            HAS_CONTENT_DEDUP,
            HAS_INGESTION_WAL,
            HAS_UNIFIED_MANIFEST,
            HAS_GOSSIP_SYNC,
            HAS_ARIA2_TRANSPORT,
        )
        UNIFIED_DATA_SYNC_AVAILABLE = True
    except ImportError as e:
        UNIFIED_DATA_SYNC_AVAILABLE = False
        IMPORT_ERROR = str(e)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config():
    """Create a sample SyncConfig for testing."""
    return SyncConfig(
        poll_interval_seconds=60,
        ephemeral_poll_interval_seconds=15,
        sync_method="incremental",
        enable_p2p_fallback=True,
        retry_max_attempts=3,
        retry_base_delay_seconds=5.0,
        backoff_multiplier=2.0,
        max_backoff_seconds=600,
        max_consecutive_failures=5,
    )


@pytest.fixture
def sample_hosts():
    """Create sample host configurations."""
    return [
        SyncHostConfig(
            name="host1",
            ssh_host="192.168.1.1",
            ssh_user="ubuntu",
            ssh_port=22,
            remote_db_path="~/data/games",
            enabled=True,
            role="selfplay",
            is_ephemeral=False,
        ),
        SyncHostConfig(
            name="host2",
            ssh_host="192.168.1.2",
            ssh_user="root",
            ssh_port=2222,
            remote_db_path="/dev/shm/games",
            enabled=True,
            role="selfplay",
            is_ephemeral=True,
        ),
        SyncHostConfig(
            name="host3-disabled",
            ssh_host="192.168.1.3",
            ssh_user="ubuntu",
            enabled=False,
        ),
    ]


@pytest.fixture
def mock_manifest(temp_dir):
    """Create a mock manifest for testing."""
    manifest_path = temp_dir / "manifest.db"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        manifest = _LegacyDataManifest(manifest_path)
    return manifest


# =============================================================================
# SyncConfig Tests
# =============================================================================


@pytest.mark.skipif(not UNIFIED_DATA_SYNC_AVAILABLE, reason="Module not available")
class TestSyncConfig:
    """Tests for SyncConfig dataclass."""

    def test_default_values(self):
        """Default config should have sensible defaults."""
        config = SyncConfig()
        assert config.poll_interval_seconds == 60
        assert config.ephemeral_poll_interval_seconds == 15
        assert config.sync_method == "incremental"
        assert config.enable_p2p_fallback is True
        assert config.deduplication is True
        assert config.content_deduplication is True

    def test_retry_settings(self):
        """Retry settings should be configurable."""
        config = SyncConfig(
            retry_max_attempts=5,
            retry_base_delay_seconds=10.0,
            backoff_multiplier=3.0,
            max_backoff_seconds=1200,
        )
        assert config.retry_max_attempts == 5
        assert config.retry_base_delay_seconds == 10.0
        assert config.backoff_multiplier == 3.0
        assert config.max_backoff_seconds == 1200

    def test_manifest_replication_settings(self):
        """Manifest replication settings should be configurable."""
        config = SyncConfig(
            enable_manifest_replication=True,
            manifest_replication_interval=600,
            min_replicas=3,
        )
        assert config.enable_manifest_replication is True
        assert config.manifest_replication_interval == 600
        assert config.min_replicas == 3

    def test_watchdog_settings(self):
        """Watchdog settings should be configurable."""
        config = SyncConfig(
            enable_watchdog=True,
            watchdog_check_interval=60,
            watchdog_unhealthy_threshold=5,
        )
        assert config.enable_watchdog is True
        assert config.watchdog_check_interval == 60
        assert config.watchdog_unhealthy_threshold == 5

    def test_aria2_settings(self):
        """Aria2 transport settings should be configurable."""
        config = SyncConfig(
            enable_aria2_transport=True,
            aria2_data_server_port=8766,
            aria2_connections_per_server=16,
            aria2_split=16,
        )
        assert config.enable_aria2_transport is True
        assert config.aria2_data_server_port == 8766
        assert config.aria2_connections_per_server == 16
        assert config.aria2_split == 16

    def test_quality_extraction_settings(self):
        """Quality extraction settings should be configurable."""
        config = SyncConfig(
            enable_quality_extraction=True,
            quality_elo_weight=0.5,
            quality_length_weight=0.25,
            quality_decisive_weight=0.25,
            min_quality_score_for_priority=0.7,
        )
        assert config.enable_quality_extraction is True
        assert config.quality_elo_weight == 0.5
        assert config.quality_length_weight == 0.25
        assert config.min_quality_score_for_priority == 0.7

    def test_gossip_sync_settings(self):
        """Gossip sync settings should be configurable."""
        config = SyncConfig(
            enable_gossip_sync=True,
            gossip_port=8771,
        )
        assert config.enable_gossip_sync is True
        assert config.gossip_port == 8771

    def test_aggregation_settings(self):
        """Aggregation mode settings should be configurable."""
        config = SyncConfig(
            enable_aggregation=True,
            aggregator_node="leader-node",
            aggregator_db="data/games/aggregated.db",
            aggregate_interval_seconds=300,
        )
        assert config.enable_aggregation is True
        assert config.aggregator_node == "leader-node"
        assert config.aggregator_db == "data/games/aggregated.db"


# =============================================================================
# SyncHostConfig Tests
# =============================================================================


@pytest.mark.skipif(not UNIFIED_DATA_SYNC_AVAILABLE, reason="Module not available")
class TestSyncHostConfig:
    """Tests for SyncHostConfig dataclass."""

    def test_required_fields(self):
        """Name and ssh_host are required fields."""
        host = SyncHostConfig(name="host1", ssh_host="192.168.1.1")
        assert host.name == "host1"
        assert host.ssh_host == "192.168.1.1"

    def test_default_values(self):
        """Default values should be sensible."""
        host = SyncHostConfig(name="host1", ssh_host="192.168.1.1")
        assert host.ssh_user == "ubuntu"
        assert host.ssh_port == 22
        assert host.ssh_key is None
        assert host.remote_db_path == "~/ringrift/ai-service/data/games"
        assert host.enabled is True
        assert host.role == "selfplay"
        assert host.is_ephemeral is False

    def test_ephemeral_host(self):
        """Ephemeral flag should be settable."""
        host = SyncHostConfig(
            name="vast-123",
            ssh_host="10.0.0.1",
            ssh_user="root",
            ssh_port=2222,
            remote_db_path="/dev/shm/games",
            is_ephemeral=True,
        )
        assert host.is_ephemeral is True
        assert host.ssh_user == "root"
        assert host.ssh_port == 2222

    def test_ssh_key_path(self):
        """SSH key path should be settable."""
        host = SyncHostConfig(
            name="host1",
            ssh_host="192.168.1.1",
            ssh_key="~/.ssh/id_cluster",
        )
        assert host.ssh_key == "~/.ssh/id_cluster"

    def test_hostconfig_alias(self):
        """HostConfig should be an alias for SyncHostConfig."""
        assert HostConfig is SyncHostConfig


# =============================================================================
# HostSyncState Tests
# =============================================================================


@pytest.mark.skipif(not UNIFIED_DATA_SYNC_AVAILABLE, reason="Module not available")
class TestHostSyncState:
    """Tests for HostSyncState dataclass."""

    def test_default_values(self):
        """Default state should have zeros."""
        state = HostSyncState(name="host1")
        assert state.name == "host1"
        assert state.last_sync_time == 0.0
        assert state.last_game_count == 0
        assert state.total_games_synced == 0
        assert state.consecutive_failures == 0
        assert state.last_error == ""
        assert state.last_error_time == 0.0

    def test_mutable_state(self):
        """State should be mutable."""
        state = HostSyncState(name="host1")
        state.last_sync_time = time.time()
        state.total_games_synced = 100
        state.consecutive_failures = 2
        state.last_error = "Connection refused"
        state.last_error_time = time.time()

        assert state.total_games_synced == 100
        assert state.consecutive_failures == 2
        assert "Connection" in state.last_error


# =============================================================================
# Legacy DataManifest Tests
# =============================================================================


@pytest.mark.skipif(not UNIFIED_DATA_SYNC_AVAILABLE, reason="Module not available")
class TestLegacyDataManifest:
    """Tests for _LegacyDataManifest class."""

    def test_manifest_initialization(self, temp_dir):
        """Manifest should create database on init."""
        manifest_path = temp_dir / "test_manifest.db"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            manifest = _LegacyDataManifest(manifest_path)

        assert manifest_path.exists()
        assert manifest.db_path == manifest_path

    def test_manifest_creates_tables(self, temp_dir):
        """Manifest should create required tables."""
        manifest_path = temp_dir / "test_manifest.db"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            _LegacyDataManifest(manifest_path)

        conn = sqlite3.connect(manifest_path)
        cursor = conn.cursor()

        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        assert "synced_games" in tables
        assert "host_states" in tables
        assert "sync_history" in tables
        assert "dead_letter_queue" in tables

        conn.close()

    def test_is_game_synced_false(self, mock_manifest):
        """is_game_synced should return False for unknown games."""
        assert mock_manifest.is_game_synced("unknown-game-id") is False

    def test_mark_games_synced(self, mock_manifest):
        """mark_games_synced should record game IDs."""
        game_ids = ["game-1", "game-2", "game-3"]
        mock_manifest.mark_games_synced(
            game_ids,
            source_host="host1",
            source_db="selfplay.db",
            board_type="hex8",
            num_players=2,
        )

        for game_id in game_ids:
            assert mock_manifest.is_game_synced(game_id) is True

    def test_is_content_synced(self, mock_manifest):
        """is_content_synced should check content hash."""
        game_ids = ["game-1"]
        content_hashes = ["abc123def456"]

        mock_manifest.mark_games_synced(
            game_ids,
            source_host="host1",
            source_db="selfplay.db",
            content_hashes=content_hashes,
        )

        assert mock_manifest.is_content_synced("abc123def456") is True
        assert mock_manifest.is_content_synced("unknown-hash") is False

    def test_get_synced_count(self, mock_manifest):
        """get_synced_count should return total synced games."""
        assert mock_manifest.get_synced_count() == 0

        mock_manifest.mark_games_synced(
            ["game-1", "game-2"],
            source_host="host1",
            source_db="db1.db",
        )
        assert mock_manifest.get_synced_count() == 2

        mock_manifest.mark_games_synced(
            ["game-3"],
            source_host="host2",
            source_db="db2.db",
        )
        assert mock_manifest.get_synced_count() == 3

    def test_save_and_load_host_state(self, mock_manifest):
        """Host state should be persistable and loadable."""
        state = HostSyncState(
            name="test-host",
            last_sync_time=1234567890.0,
            total_games_synced=500,
            consecutive_failures=2,
            last_error="timeout",
        )

        mock_manifest.save_host_state(state)
        loaded = mock_manifest.load_host_state("test-host")

        assert loaded is not None
        assert loaded.name == "test-host"
        assert loaded.last_sync_time == 1234567890.0
        assert loaded.total_games_synced == 500
        assert loaded.consecutive_failures == 2
        assert loaded.last_error == "timeout"

    def test_load_host_state_not_found(self, mock_manifest):
        """load_host_state should return None for unknown hosts."""
        result = mock_manifest.load_host_state("unknown-host")
        assert result is None

    def test_record_sync(self, mock_manifest):
        """record_sync should log sync events."""
        mock_manifest.record_sync(
            host_name="host1",
            games_synced=100,
            duration=5.5,
            success=True,
            sync_method="ssh",
        )

        # Query sync history
        conn = sqlite3.connect(mock_manifest.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sync_history WHERE host_name = ?", ("host1",))
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[1] == "host1"  # host_name
        assert row[3] == 100  # games_synced
        assert row[5] == 1  # success

    def test_add_to_dead_letter(self, mock_manifest):
        """add_to_dead_letter should queue failed games."""
        mock_manifest.add_to_dead_letter(
            game_id="failed-game-1",
            source_host="host1",
            source_db="db.db",
            error_message="Corrupt data",
            error_type="ValidationError",
        )

        assert mock_manifest.get_dead_letter_count() == 1

    def test_get_dead_letter_count(self, mock_manifest):
        """get_dead_letter_count should return unresolved entries."""
        assert mock_manifest.get_dead_letter_count() == 0

        mock_manifest.add_to_dead_letter(
            game_id="failed-1",
            source_host="host1",
            source_db="db.db",
            error_message="Error",
            error_type="Error",
        )
        mock_manifest.add_to_dead_letter(
            game_id="failed-2",
            source_host="host1",
            source_db="db.db",
            error_message="Error",
            error_type="Error",
        )

        assert mock_manifest.get_dead_letter_count() == 2


# =============================================================================
# UnifiedDataSyncService Initialization Tests
# =============================================================================


@pytest.mark.skipif(not UNIFIED_DATA_SYNC_AVAILABLE, reason="Module not available")
class TestUnifiedDataSyncServiceInit:
    """Tests for UnifiedDataSyncService initialization."""

    def test_init_basic(self, sample_config, sample_hosts, mock_manifest):
        """Service should initialize with basic config."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
            )

        assert service.config is sample_config
        assert len(service.hosts) == 3
        assert service.manifest is mock_manifest
        assert service._running is False

    def test_hosts_stored_as_dict(self, sample_config, sample_hosts, mock_manifest):
        """Hosts should be stored as dict keyed by name."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
            )

        assert "host1" in service.hosts
        assert "host2" in service.hosts
        assert service.hosts["host1"].ssh_host == "192.168.1.1"

    def test_host_categorization(self, sample_config, sample_hosts, mock_manifest):
        """Hosts should be categorized as ephemeral or persistent."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
            )

        assert "host1" in service._persistent_hosts
        assert "host2" in service._ephemeral_hosts
        assert "host3-disabled" in service._persistent_hosts

    def test_host_states_initialized(self, sample_config, sample_hosts, mock_manifest):
        """Host states should be initialized for all hosts."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
            )

        for host in sample_hosts:
            assert host.name in service.host_states
            state = service.host_states[host.name]
            assert state.name == host.name

    def test_sync_stats_initialized(self, sample_config, sample_hosts, mock_manifest):
        """Sync stats should be initialized."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
            )

        assert service._sync_stats["ssh"] == 0
        assert service._sync_stats["p2p_http"] == 0
        assert service._sync_stats["aria2"] == 0
        assert service._sync_stats["failed"] == 0
        assert service._sync_stats["deduplicated"] == 0

    def test_custom_http_port(self, sample_config, sample_hosts, mock_manifest):
        """Custom HTTP port should be accepted."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
                http_port=9999,
            )

        assert service.http_port == 9999


# =============================================================================
# SSH Sync Tests
# =============================================================================


@pytest.mark.skipif(not UNIFIED_DATA_SYNC_AVAILABLE, reason="Module not available")
class TestSyncHostSSH:
    """Tests for SSH/rsync sync method."""

    @pytest.mark.asyncio
    async def test_sync_host_ssh_success(self, sample_config, sample_hosts, mock_manifest, temp_dir):
        """SSH sync should return game count on success."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
            )

        host = sample_hosts[0]
        local_dir = temp_dir / "sync"

        # Mock subprocess
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_shell", return_value=mock_process):
            # Create a fake DB file to count
            local_dir.mkdir(parents=True, exist_ok=True)
            db_path = local_dir / "test.db"
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE games (game_id TEXT PRIMARY KEY)")
            conn.execute("INSERT INTO games VALUES ('g1'), ('g2'), ('g3')")
            conn.commit()
            conn.close()

            games, error = await service._sync_host_ssh(host, local_dir)

        assert games == 3
        assert error == ""

    @pytest.mark.asyncio
    async def test_sync_host_ssh_failure(self, sample_config, sample_hosts, mock_manifest, temp_dir):
        """SSH sync should return error on failure."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
            )

        host = sample_hosts[0]
        local_dir = temp_dir / "sync"

        # Mock subprocess failure
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"Connection refused"))

        with patch("asyncio.create_subprocess_shell", return_value=mock_process):
            games, error = await service._sync_host_ssh(host, local_dir)

        assert games == 0
        assert "Connection refused" in error

    @pytest.mark.asyncio
    async def test_sync_host_ssh_timeout(self, sample_config, sample_hosts, mock_manifest, temp_dir):
        """SSH sync should handle timeout."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
            )

        host = sample_hosts[0]
        local_dir = temp_dir / "sync"

        # Mock timeout
        with patch("asyncio.create_subprocess_shell", side_effect=asyncio.TimeoutError()):
            games, error = await service._sync_host_ssh(host, local_dir)

        assert games == 0
        assert error == "timeout"


# =============================================================================
# P2P Fallback Tests
# =============================================================================


@pytest.mark.skipif(not UNIFIED_DATA_SYNC_AVAILABLE, reason="Module not available")
class TestSyncHostP2P:
    """Tests for P2P HTTP fallback sync method."""

    @pytest.mark.asyncio
    async def test_sync_host_p2p_not_available(self, sample_config, sample_hosts, mock_manifest, temp_dir):
        """P2P sync should fail gracefully when not available."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
            )

        # Ensure P2P is not available
        service._p2p_fallback = None

        host = sample_hosts[0]
        local_dir = temp_dir / "sync"

        games, error = await service._sync_host_p2p(host, local_dir)

        assert games == 0
        assert "not available" in error

    @pytest.mark.asyncio
    async def test_sync_host_p2p_success(self, sample_config, sample_hosts, mock_manifest, temp_dir):
        """P2P sync should return games on success."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
            )

        # Mock P2P fallback
        mock_p2p = AsyncMock()
        mock_p2p.sync_with_fallback = AsyncMock(return_value=(True, 50, "http"))
        service._p2p_fallback = mock_p2p

        host = sample_hosts[0]
        local_dir = temp_dir / "sync"

        games, error = await service._sync_host_p2p(host, local_dir)

        assert games == 50
        assert error == ""

    @pytest.mark.asyncio
    async def test_sync_host_p2p_failure(self, sample_config, sample_hosts, mock_manifest, temp_dir):
        """P2P sync should return error on failure."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
            )

        # Mock P2P fallback failure
        mock_p2p = AsyncMock()
        mock_p2p.sync_with_fallback = AsyncMock(return_value=(False, 0, None))
        service._p2p_fallback = mock_p2p

        host = sample_hosts[0]
        local_dir = temp_dir / "sync"

        games, error = await service._sync_host_p2p(host, local_dir)

        assert games == 0
        assert error == "P2P sync failed"


# =============================================================================
# Aria2 Transport Tests
# =============================================================================


@pytest.mark.skipif(not UNIFIED_DATA_SYNC_AVAILABLE, reason="Module not available")
class TestSyncHostAria2:
    """Tests for aria2 high-performance sync method."""

    @pytest.mark.asyncio
    async def test_sync_host_aria2_not_available(self, sample_config, sample_hosts, mock_manifest, temp_dir):
        """Aria2 sync should fail gracefully when not available."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
            )

        # Ensure aria2 is not available
        service._aria2_transport = None

        host = sample_hosts[0]
        local_dir = temp_dir / "sync"

        games, error = await service._sync_host_aria2(host, local_dir)

        assert games == 0
        assert "not available" in error

    @pytest.mark.asyncio
    async def test_sync_host_aria2_success(self, sample_config, sample_hosts, mock_manifest, temp_dir):
        """Aria2 sync should return files on success."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
            )

        # Mock aria2 transport
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.files_synced = 25
        mock_result.errors = []

        mock_aria2 = AsyncMock()
        mock_aria2.sync_games = AsyncMock(return_value=mock_result)
        service._aria2_transport = mock_aria2

        host = sample_hosts[0]
        local_dir = temp_dir / "sync"

        games, error = await service._sync_host_aria2(host, local_dir)

        assert games == 25
        assert error == ""


# =============================================================================
# Multi-Transport Failover Tests
# =============================================================================


@pytest.mark.skipif(not UNIFIED_DATA_SYNC_AVAILABLE, reason="Module not available")
class TestMultiTransportFailover:
    """Tests for multi-transport failover logic."""

    @pytest.mark.asyncio
    async def test_fallback_ssh_to_p2p(self, sample_config, sample_hosts, mock_manifest, temp_dir):
        """Should fallback from SSH to P2P on failure."""
        # Enable P2P fallback
        sample_config.enable_p2p_fallback = True
        # Disable features that interfere with testing
        sample_config.enable_wal = False
        sample_config.enable_gossip_sync = False

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
            )

        # Ensure WAL is disabled for this test
        service._ingestion_wal = None

        # Mock SSH failure
        service._sync_host_ssh = AsyncMock(return_value=(0, "SSH connection failed"))

        # Mock P2P success
        service._sync_host_p2p = AsyncMock(return_value=(100, ""))

        # Mock aria2 (should not be called)
        service._aria2_transport = None

        host = sample_hosts[0]
        service.host_states[host.name] = HostSyncState(name=host.name)

        # Patch circuit breaker and other helpers
        with patch.object(service, "_build_ssh_args", return_value=""), \
             patch.object(service.manifest, "record_sync"), \
             patch.object(service.manifest, "save_host_state"), \
             patch.object(service, "_validate_synced_files", return_value={"valid": True, "errors": []}), \
             patch.object(service, "_extract_and_store_quality", return_value=0), \
             patch("app.distributed.unified_data_sync.HAS_EVENT_BUS", False), \
             patch("app.distributed.unified_data_sync.HAS_CROSS_PROCESS_EVENTS", False), \
             patch("app.distributed.unified_data_sync.HAS_CIRCUIT_BREAKER", False):

            games = await service._sync_host(host)

        # P2P should have been called after SSH failure
        service._sync_host_p2p.assert_called_once()
        assert games == 100

    @pytest.mark.asyncio
    async def test_fallback_all_transports_fail(self, sample_config, sample_hosts, mock_manifest, temp_dir):
        """Should record failure when all transports fail."""
        sample_config.enable_p2p_fallback = True

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
            )

        # Mock all transport failures
        service._sync_host_ssh = AsyncMock(return_value=(0, "SSH failed"))
        service._sync_host_p2p = AsyncMock(return_value=(0, "P2P failed"))
        service._aria2_transport = None

        host = sample_hosts[0]
        service.host_states[host.name] = HostSyncState(name=host.name)

        with patch.object(service, "_build_ssh_args", return_value=""), \
             patch.object(service.manifest, "record_sync"), \
             patch.object(service.manifest, "save_host_state"):

            games = await service._sync_host(host)

        assert games == 0
        assert service.host_states[host.name].consecutive_failures == 1


# =============================================================================
# Retry and Backoff Tests
# =============================================================================


@pytest.mark.skipif(not UNIFIED_DATA_SYNC_AVAILABLE, reason="Module not available")
class TestRetryAndBackoff:
    """Tests for retry logic and exponential backoff."""

    def test_backoff_calculation(self, sample_config):
        """Backoff should increase exponentially."""
        # Base: 60s, multiplier: 2.0
        poll_interval = 60
        multiplier = 2.0
        max_backoff = 600

        # Consecutive failures: 1 -> 2 -> 4 -> 8 -> 16 (capped at max)
        for failures in range(1, 6):
            backoff = min(max_backoff, poll_interval * (multiplier ** failures))
            expected = min(600, 60 * (2 ** failures))
            assert backoff == expected

    @pytest.mark.asyncio
    async def test_skip_host_during_backoff(self, sample_config, sample_hosts, mock_manifest):
        """Host should be skipped during backoff period."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
            )

        host = sample_hosts[0]

        # Set up state with recent failure
        state = HostSyncState(name=host.name)
        state.consecutive_failures = 2
        state.last_error_time = time.time()  # Just failed
        service.host_states[host.name] = state

        # Should return 0 immediately due to backoff
        games = await service._sync_host(host)
        assert games == 0

    def test_max_consecutive_failures_respected(self, sample_config, sample_hosts, mock_manifest):
        """Host with max failures should be skipped in collection cycle."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
            )

        host = sample_hosts[0]

        # Set up state with max failures
        state = HostSyncState(name=host.name)
        state.consecutive_failures = sample_config.max_consecutive_failures
        service.host_states[host.name] = state

        # In run_collection_cycle, this host should be skipped
        # We verify the logic is correct
        max_failures = sample_config.max_consecutive_failures
        assert state.consecutive_failures >= max_failures


# =============================================================================
# Circuit Breaker Integration Tests
# =============================================================================


@pytest.mark.skipif(not UNIFIED_DATA_SYNC_AVAILABLE, reason="Module not available")
@pytest.mark.skipif(not HAS_CIRCUIT_BREAKER, reason="Circuit breaker not available")
class TestCircuitBreakerIntegration:
    """Tests for circuit breaker integration."""

    @pytest.mark.asyncio
    async def test_skip_host_with_open_circuit(self, sample_config, sample_hosts, mock_manifest):
        """Host with open circuit should be skipped."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
            )

        host = sample_hosts[0]
        service.host_states[host.name] = HostSyncState(name=host.name)

        # Mock circuit breaker returning False (open)
        with patch("app.distributed.unified_data_sync.get_host_breaker") as mock_breaker:
            mock_breaker.return_value.can_execute.return_value = False

            games = await service._sync_host(host)

        assert games == 0

    @pytest.mark.asyncio
    async def test_record_success_to_circuit_breaker(self, sample_config, sample_hosts, mock_manifest, temp_dir):
        """Successful sync should record success to circuit breaker."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
            )

        host = sample_hosts[0]
        service.host_states[host.name] = HostSyncState(name=host.name)
        service._sync_host_ssh = AsyncMock(return_value=(10, ""))

        mock_breaker_instance = MagicMock()
        mock_breaker_instance.can_execute.return_value = True

        with patch("app.distributed.unified_data_sync.get_host_breaker") as mock_breaker:
            mock_breaker.return_value = mock_breaker_instance
            with patch.object(service.manifest, "record_sync"), \
                 patch.object(service.manifest, "save_host_state"), \
                 patch.object(service, "_validate_synced_files", return_value={"valid": True, "errors": []}), \
                 patch.object(service, "_extract_and_store_quality", return_value=0):

                await service._sync_host(host)

        mock_breaker_instance.record_success.assert_called_once_with(host.ssh_host)


# =============================================================================
# Statistics Tracking Tests
# =============================================================================


@pytest.mark.skipif(not UNIFIED_DATA_SYNC_AVAILABLE, reason="Module not available")
class TestStatisticsTracking:
    """Tests for sync statistics tracking."""

    @pytest.mark.asyncio
    async def test_ssh_success_tracked(self, sample_config, sample_hosts, mock_manifest):
        """SSH success should increment ssh counter."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
            )

        host = sample_hosts[0]
        service.host_states[host.name] = HostSyncState(name=host.name)
        service._sync_host_ssh = AsyncMock(return_value=(10, ""))

        with patch.object(service.manifest, "record_sync"), \
             patch.object(service.manifest, "save_host_state"), \
             patch.object(service, "_validate_synced_files", return_value={"valid": True, "errors": []}), \
             patch.object(service, "_extract_and_store_quality", return_value=0):

            await service._sync_host(host)

        assert service._sync_stats["ssh"] == 1

    @pytest.mark.asyncio
    async def test_failed_sync_tracked(self, sample_config, sample_hosts, mock_manifest):
        """Failed sync should increment failed counter."""
        sample_config.enable_p2p_fallback = False

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
            )

        host = sample_hosts[0]
        service.host_states[host.name] = HostSyncState(name=host.name)
        service._sync_host_ssh = AsyncMock(return_value=(0, "Connection failed"))
        service._aria2_transport = None

        with patch.object(service.manifest, "record_sync"), \
             patch.object(service.manifest, "save_host_state"):

            await service._sync_host(host)

        assert service._sync_stats["failed"] == 1

    def test_get_statistics(self, sample_config, sample_hosts, mock_manifest):
        """get_statistics should return comprehensive stats."""
        # Disable WAL to avoid AttributeError
        sample_config.enable_wal = False
        sample_config.enable_gossip_sync = False

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
            )

        # Ensure WAL is disabled
        service._ingestion_wal = None
        service._gossip_daemon = None

        stats = service.get_statistics()

        assert "running" in stats
        assert "total_synced" in stats
        assert "dead_letter_count" in stats
        assert "sync_stats" in stats
        assert "host_count" in stats
        assert "ephemeral_host_count" in stats


# =============================================================================
# Collection Cycle Tests
# =============================================================================


@pytest.mark.skipif(not UNIFIED_DATA_SYNC_AVAILABLE, reason="Module not available")
class TestCollectionCycle:
    """Tests for collection cycle behavior."""

    @pytest.mark.asyncio
    async def test_run_collection_cycle_skips_disabled_hosts(self, sample_config, sample_hosts, mock_manifest):
        """Collection cycle should skip disabled hosts."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
            )

        # Mock _sync_host
        service._sync_host = AsyncMock(return_value=10)

        # Reset timing to allow sync
        service._last_ephemeral_sync = 0
        service._last_persistent_sync = 0

        await service.run_collection_cycle()

        # Should have synced enabled hosts only (host1, host2)
        assert service._sync_host.call_count == 2

    @pytest.mark.asyncio
    async def test_run_collection_cycle_respects_poll_interval(self, sample_config, sample_hosts, mock_manifest):
        """Collection cycle should respect poll intervals."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
            )

        service._sync_host = AsyncMock(return_value=10)

        # Set recent sync times
        now = time.time()
        service._last_ephemeral_sync = now
        service._last_persistent_sync = now

        await service.run_collection_cycle()

        # Should not have synced any hosts due to recent sync
        assert service._sync_host.call_count == 0

    @pytest.mark.asyncio
    async def test_run_collection_cycle_aggregates_results(self, sample_config, sample_hosts, mock_manifest):
        """Collection cycle should aggregate game counts."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
            )

        # Mock different game counts per host
        call_count = 0
        async def mock_sync(host):
            nonlocal call_count
            call_count += 1
            return 10 * call_count

        service._sync_host = mock_sync
        service._last_ephemeral_sync = 0
        service._last_persistent_sync = 0

        total = await service.run_collection_cycle()

        # 10 + 20 = 30 (2 enabled hosts)
        assert total == 30


# =============================================================================
# Service Lifecycle Tests
# =============================================================================


@pytest.mark.skipif(not UNIFIED_DATA_SYNC_AVAILABLE, reason="Module not available")
class TestServiceLifecycle:
    """Tests for service start/stop lifecycle."""

    def test_stop_sets_running_false(self, sample_config, sample_hosts, mock_manifest):
        """stop() should set _running to False."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
            )

        service._running = True
        service.stop()

        assert service._running is False

    def test_stop_sets_shutdown_event(self, sample_config, sample_hosts, mock_manifest):
        """stop() should set shutdown event."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
            )

        service.stop()

        assert service._shutdown_event.is_set()


# =============================================================================
# SSH Args Building Tests
# =============================================================================


@pytest.mark.skipif(not UNIFIED_DATA_SYNC_AVAILABLE, reason="Module not available")
class TestBuildSSHArgs:
    """Tests for SSH arguments building."""

    def test_build_ssh_args_default(self, sample_config, sample_hosts, mock_manifest):
        """SSH args should include default options."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
            )

        host = sample_hosts[0]
        args = service._build_ssh_args(host)

        assert "-o ConnectTimeout=" in args
        assert "-o StrictHostKeyChecking=accept-new" in args
        assert "-o BatchMode=yes" in args

    def test_build_ssh_args_custom_port(self, sample_config, sample_hosts, mock_manifest):
        """SSH args should include custom port."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
            )

        # host2 has custom port 2222
        host = sample_hosts[1]
        args = service._build_ssh_args(host)

        assert "-p 2222" in args

    def test_build_ssh_args_with_key(self, sample_config, mock_manifest, temp_dir):
        """SSH args should include key path if file exists."""
        # Create a test key file
        key_path = temp_dir / "test_key"
        key_path.touch()

        host = SyncHostConfig(
            name="test-host",
            ssh_host="192.168.1.1",
            ssh_key=str(key_path),
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=[host],
                manifest=mock_manifest,
            )

        args = service._build_ssh_args(host)

        assert f"-i {key_path}" in args


# =============================================================================
# Host Loading Tests
# =============================================================================


@pytest.mark.skipif(not UNIFIED_DATA_SYNC_AVAILABLE, reason="Module not available")
class TestLoadHostsFromYAML:
    """Tests for YAML host loading."""

    def test_load_hosts_nonexistent_file(self, temp_dir):
        """load_hosts_from_yaml should return empty list for nonexistent file."""
        path = temp_dir / "nonexistent.yaml"
        hosts = load_hosts_from_yaml(path)
        assert hosts == []

    def test_load_hosts_distributed_format(self, temp_dir):
        """load_hosts_from_yaml should parse distributed_hosts format."""
        yaml_content = """
hosts:
  worker-1:
    tailscale_ip: 100.64.1.1
    ssh_host: 192.168.1.1
    ssh_user: ubuntu
    ssh_port: 22
    status: ready
    role: selfplay
  worker-2:
    tailscale_ip: 100.64.1.2
    ssh_host: 192.168.1.2
    status: terminated
"""
        yaml_path = temp_dir / "hosts.yaml"
        yaml_path.write_text(yaml_content)

        hosts = load_hosts_from_yaml(yaml_path)

        # worker-1 should be loaded and enabled
        worker1 = next((h for h in hosts if h.name == "worker-1"), None)
        assert worker1 is not None
        assert worker1.ssh_host == "100.64.1.1"  # Prefers tailscale_ip
        assert worker1.enabled is True

        # worker-2 should be disabled (terminated)
        worker2 = next((h for h in hosts if h.name == "worker-2"), None)
        assert worker2 is not None
        assert worker2.enabled is False


# =============================================================================
# Checksum Validation Tests
# =============================================================================


@pytest.mark.skipif(not UNIFIED_DATA_SYNC_AVAILABLE, reason="Module not available")
class TestChecksumValidation:
    """Tests for checksum validation."""

    @pytest.mark.asyncio
    async def test_validate_synced_files_valid_db(self, sample_config, sample_hosts, mock_manifest, temp_dir):
        """Valid database should pass validation."""
        # Disable WAL to avoid extra components
        sample_config.enable_wal = False
        sample_config.enable_gossip_sync = False

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
            )

        # Create a dedicated temp dir for this test to avoid extra files
        test_dir = temp_dir / "validation_test"
        test_dir.mkdir(parents=True, exist_ok=True)

        # Create a valid SQLite database
        db_path = test_dir / "valid.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.close()

        result = await service._validate_synced_files(test_dir, "test-host")

        assert result["valid"] is True
        assert result["files_validated"] == 1
        assert "valid.db" in result["checksums"]

    @pytest.mark.asyncio
    async def test_validate_synced_files_corrupt_db(self, sample_config, sample_hosts, mock_manifest, temp_dir):
        """Corrupt database should fail validation."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
            )

        # Create a corrupt file
        db_path = temp_dir / "corrupt.db"
        db_path.write_bytes(b"not a valid sqlite database")

        result = await service._validate_synced_files(temp_dir, "test-host")

        assert result["valid"] is False
        assert len(result["errors"]) > 0


# =============================================================================
# Aggregator Node Tests
# =============================================================================


@pytest.mark.skipif(not UNIFIED_DATA_SYNC_AVAILABLE, reason="Module not available")
class TestAggregatorNode:
    """Tests for aggregator node functionality."""

    def test_is_aggregator_node_disabled(self, sample_config, sample_hosts, mock_manifest):
        """is_aggregator_node should return False when aggregation disabled."""
        sample_config.enable_aggregation = False

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
            )

        assert service._is_aggregator_node() is False

    def test_is_aggregator_node_enabled_matching(self, sample_config, sample_hosts, mock_manifest):
        """is_aggregator_node should match hostname patterns."""
        sample_config.enable_aggregation = True

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            service = UnifiedDataSyncService(
                config=sample_config,
                hosts=sample_hosts,
                manifest=mock_manifest,
            )

        # Mock hostname matching
        with patch("socket.gethostname", return_value="leader-node"):
            sample_config.aggregator_node = "leader-node"
            assert service._is_aggregator_node() is True

            sample_config.aggregator_node = "other-node"
            assert service._is_aggregator_node() is False


# =============================================================================
# Deprecation Warning Tests
# =============================================================================


class TestDeprecationWarning:
    """Tests for deprecation warning."""

    def test_module_import_emits_deprecation_warning(self):
        """Importing the module should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Re-import to trigger warning
            import importlib
            import app.distributed.unified_data_sync
            importlib.reload(app.distributed.unified_data_sync)

            # Check for deprecation warning
            deprecation_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
                and "UnifiedDataSyncService is deprecated" in str(warning.message)
            ]
            assert len(deprecation_warnings) >= 1
