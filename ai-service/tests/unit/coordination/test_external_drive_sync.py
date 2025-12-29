"""Tests for ExternalDriveSyncDaemon.

December 2025: Tests for the external drive sync daemon that pulls
cluster data to external storage (OWC drive on mac-studio).
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.external_drive_sync import (
    ExternalDriveSyncConfig,
    ExternalDriveSyncDaemon,
    get_external_drive_sync_daemon,
    reset_external_drive_sync_daemon,
    DEFAULT_SYNC_INTERVAL,
    DEFAULT_S3_BUCKET,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_storage_config():
    """Create a mock ExternalStorageConfig."""
    config = MagicMock()
    config.host = "mac-studio"
    config.path = "/tmp/test_external_storage"
    config.receive_games = True
    config.receive_npz = True
    config.receive_models = True
    config.subdirs = {
        "games": "selfplay_repository",
        "npz": "canonical_data",
        "models": "canonical_models",
    }
    return config


@pytest.fixture
def mock_cluster_node():
    """Create a mock ClusterNode."""
    node = MagicMock()
    node.name = "runpod-h100"
    node.best_ip = "10.0.0.1"
    node.ssh_user = "root"
    node.ssh_port = 22
    node.ssh_key = "~/.ssh/id_ed25519"
    node.ringrift_path = "/workspace/ringrift/ai-service"
    node.is_active = True
    node.is_coordinator = False
    return node


@pytest.fixture
def temp_storage_path():
    """Create a temporary storage path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before and after each test."""
    reset_external_drive_sync_daemon()
    yield
    reset_external_drive_sync_daemon()


# =============================================================================
# Configuration Tests
# =============================================================================


class TestExternalDriveSyncConfig:
    """Tests for ExternalDriveSyncConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ExternalDriveSyncConfig()

        assert config.games_sync_interval == 1800.0
        assert config.npz_sync_interval == 3600.0
        assert config.models_sync_interval == 1800.0
        assert config.s3_bucket == DEFAULT_S3_BUCKET
        assert config.s3_consolidated_prefix == "consolidated"
        assert config.bandwidth_limit_kbps == 50000
        assert config.rsync_timeout == 600
        assert config.max_concurrent_syncs == 3
        assert config.max_retries == 3
        assert config.retry_delay == 30.0
        assert config.sync_games is True
        assert config.sync_npz is True
        assert config.sync_models is True
        assert config.dry_run is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ExternalDriveSyncConfig(
            games_sync_interval=900.0,
            sync_npz=False,
            dry_run=True,
            bandwidth_limit_kbps=10000,
        )

        assert config.games_sync_interval == 900.0
        assert config.sync_npz is False
        assert config.dry_run is True
        assert config.bandwidth_limit_kbps == 10000

    def test_default_constants(self):
        """Test module-level default constants."""
        assert DEFAULT_SYNC_INTERVAL == 1800
        assert DEFAULT_S3_BUCKET == "ringrift-models-20251214"


# =============================================================================
# Daemon Initialization Tests
# =============================================================================


class TestDaemonInitialization:
    """Tests for daemon initialization."""

    def test_init_with_defaults(self, mock_storage_config):
        """Test initialization with default config."""
        daemon = ExternalDriveSyncDaemon(storage_config=mock_storage_config)

        assert daemon.config is not None
        assert daemon._storage_config == mock_storage_config
        assert daemon._running is False
        assert daemon._start_time == 0.0
        assert daemon._cycles_completed == 0
        assert daemon._errors_count == 0

    def test_init_with_custom_config(self, mock_storage_config):
        """Test initialization with custom config."""
        config = ExternalDriveSyncConfig(
            games_sync_interval=600.0,
            dry_run=True,
        )
        daemon = ExternalDriveSyncDaemon(
            config=config,
            storage_config=mock_storage_config,
        )

        assert daemon.config.games_sync_interval == 600.0
        assert daemon.config.dry_run is True

    def test_init_stats(self, mock_storage_config):
        """Test initial stats dictionary."""
        daemon = ExternalDriveSyncDaemon(storage_config=mock_storage_config)

        assert daemon._stats["games_synced"] == 0
        assert daemon._stats["npz_synced"] == 0
        assert daemon._stats["models_synced"] == 0
        assert daemon._stats["bytes_transferred"] == 0
        assert daemon._stats["nodes_synced"] == 0
        assert daemon._stats["sync_errors"] == 0

    @patch("app.coordination.external_drive_sync.get_sync_routing")
    @patch("socket.gethostname")
    def test_auto_detect_storage(self, mock_hostname, mock_get_sync_routing):
        """Test automatic storage detection."""
        mock_hostname.return_value = "mac-studio"

        storage = MagicMock()
        storage.host = "mac-studio"
        storage.path = "/Volumes/RingRift-Data"

        routing = MagicMock()
        routing.allowed_external_storage = [storage]
        mock_get_sync_routing.return_value = routing

        daemon = ExternalDriveSyncDaemon()

        assert daemon._storage_config == storage


# =============================================================================
# Property Tests
# =============================================================================


class TestDaemonProperties:
    """Tests for daemon properties."""

    def test_uptime_not_started(self, mock_storage_config):
        """Test uptime when daemon hasn't started."""
        daemon = ExternalDriveSyncDaemon(storage_config=mock_storage_config)

        assert daemon.uptime_seconds == 0.0

    def test_uptime_started(self, mock_storage_config):
        """Test uptime when daemon has started."""
        daemon = ExternalDriveSyncDaemon(storage_config=mock_storage_config)
        daemon._start_time = 1000.0

        with patch("time.time", return_value=1060.0):
            assert daemon.uptime_seconds == 60.0

    def test_is_enabled_no_storage(self):
        """Test is_enabled without storage config."""
        daemon = ExternalDriveSyncDaemon(storage_config=None)

        assert daemon.is_enabled is False

    def test_is_enabled_path_not_exists(self, mock_storage_config):
        """Test is_enabled when path doesn't exist."""
        mock_storage_config.path = "/nonexistent/path"
        daemon = ExternalDriveSyncDaemon(storage_config=mock_storage_config)

        assert daemon.is_enabled is False

    def test_is_enabled_path_exists(self, mock_storage_config, temp_storage_path):
        """Test is_enabled when path exists."""
        mock_storage_config.path = temp_storage_path
        daemon = ExternalDriveSyncDaemon(storage_config=mock_storage_config)

        assert daemon.is_enabled is True


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health_check method."""

    def test_health_disabled(self):
        """Test health check when daemon is disabled."""
        daemon = ExternalDriveSyncDaemon(storage_config=None)
        result = daemon.health_check()

        assert result.healthy is True  # Not unhealthy, just disabled
        assert "disabled" in result.message.lower()
        assert result.details["enabled"] is False

    def test_health_not_running(self, mock_storage_config, temp_storage_path):
        """Test health check when daemon is not running."""
        mock_storage_config.path = temp_storage_path
        daemon = ExternalDriveSyncDaemon(storage_config=mock_storage_config)

        result = daemon.health_check()

        assert result.healthy is False
        assert "not running" in result.message.lower()
        assert result.details["running"] is False

    def test_health_running_healthy(self, mock_storage_config, temp_storage_path):
        """Test health check when daemon is running and healthy."""
        mock_storage_config.path = temp_storage_path
        daemon = ExternalDriveSyncDaemon(storage_config=mock_storage_config)
        daemon._running = True
        daemon._stats["games_synced"] = 10
        daemon._stats["nodes_synced"] = 5

        result = daemon.health_check()

        assert result.healthy is True
        assert "running" in result.message.lower()

    def test_health_high_error_rate(self, mock_storage_config, temp_storage_path):
        """Test health check with high error rate."""
        mock_storage_config.path = temp_storage_path
        daemon = ExternalDriveSyncDaemon(storage_config=mock_storage_config)
        daemon._running = True
        daemon._stats["games_synced"] = 5
        daemon._stats["sync_errors"] = 10

        result = daemon.health_check()

        assert result.healthy is False
        assert "error rate" in result.message.lower()

    def test_health_details_include_stats(self, mock_storage_config, temp_storage_path):
        """Test that health check includes stats in details."""
        mock_storage_config.path = temp_storage_path
        daemon = ExternalDriveSyncDaemon(storage_config=mock_storage_config)
        daemon._running = True
        daemon._cycles_completed = 5
        daemon._errors_count = 2

        result = daemon.health_check()

        assert "cycles_completed" in result.details
        assert result.details["cycles_completed"] == 5
        assert result.details["errors_count"] == 2
        assert "stats" in result.details


# =============================================================================
# Status Tests
# =============================================================================


class TestGetStatus:
    """Tests for get_status method."""

    def test_status_includes_name(self, mock_storage_config, temp_storage_path):
        """Test that status includes daemon name."""
        mock_storage_config.path = temp_storage_path
        daemon = ExternalDriveSyncDaemon(storage_config=mock_storage_config)

        status = daemon.get_status()

        assert status["name"] == "ExternalDriveSyncDaemon"

    def test_status_includes_config(self, mock_storage_config, temp_storage_path):
        """Test that status includes configuration."""
        mock_storage_config.path = temp_storage_path
        daemon = ExternalDriveSyncDaemon(storage_config=mock_storage_config)

        status = daemon.get_status()

        assert "config" in status
        assert status["config"]["sync_games"] is True
        assert status["config"]["sync_npz"] is True
        assert status["config"]["sync_models"] is True

    def test_status_includes_health(self, mock_storage_config, temp_storage_path):
        """Test that status includes health info."""
        mock_storage_config.path = temp_storage_path
        daemon = ExternalDriveSyncDaemon(storage_config=mock_storage_config)

        status = daemon.get_status()

        assert "health" in status
        assert "healthy" in status["health"]
        assert "status" in status["health"]
        assert "message" in status["health"]


# =============================================================================
# SSH Command Building Tests
# =============================================================================


class TestBuildSshCommand:
    """Tests for SSH command building."""

    def test_basic_ssh_command(self, mock_storage_config, mock_cluster_node):
        """Test basic SSH command without port or key."""
        mock_cluster_node.ssh_port = 22
        mock_cluster_node.ssh_key = None

        daemon = ExternalDriveSyncDaemon(storage_config=mock_storage_config)
        cmd = daemon._build_ssh_command(mock_cluster_node)

        assert "ssh" in cmd
        assert "-o StrictHostKeyChecking=no" in cmd
        assert "-o ConnectTimeout=30" in cmd

    def test_ssh_command_with_port(self, mock_storage_config, mock_cluster_node):
        """Test SSH command with custom port."""
        mock_cluster_node.ssh_port = 30178

        daemon = ExternalDriveSyncDaemon(storage_config=mock_storage_config)
        cmd = daemon._build_ssh_command(mock_cluster_node)

        assert "-p 30178" in cmd

    def test_ssh_command_with_key(self, mock_storage_config, mock_cluster_node):
        """Test SSH command with SSH key."""
        mock_cluster_node.ssh_key = "~/.ssh/id_cluster"

        daemon = ExternalDriveSyncDaemon(storage_config=mock_storage_config)
        cmd = daemon._build_ssh_command(mock_cluster_node)

        # expanduser should expand the path
        assert "-i" in cmd
        assert "id_cluster" in cmd


# =============================================================================
# Sync Source Node Tests
# =============================================================================


class TestGetSyncSourceNodes:
    """Tests for sync source node filtering."""

    @patch("app.coordination.external_drive_sync.get_cluster_nodes")
    def test_filters_inactive_nodes(self, mock_get_nodes, mock_storage_config):
        """Test that inactive nodes are filtered."""
        node = MagicMock()
        node.is_active = False
        node.best_ip = "10.0.0.1"
        node.ssh_user = "root"
        node.is_coordinator = False

        mock_get_nodes.return_value = {"node1": node}

        daemon = ExternalDriveSyncDaemon(storage_config=mock_storage_config)
        nodes = daemon._get_sync_source_nodes()

        assert len(nodes) == 0

    @patch("app.coordination.external_drive_sync.get_cluster_nodes")
    def test_filters_nodes_without_ip(self, mock_get_nodes, mock_storage_config):
        """Test that nodes without IP are filtered."""
        node = MagicMock()
        node.is_active = True
        node.best_ip = None
        node.ssh_user = "root"
        node.is_coordinator = False

        mock_get_nodes.return_value = {"node1": node}

        daemon = ExternalDriveSyncDaemon(storage_config=mock_storage_config)
        nodes = daemon._get_sync_source_nodes()

        assert len(nodes) == 0

    @patch("app.coordination.external_drive_sync.get_cluster_nodes")
    def test_filters_coordinators(self, mock_get_nodes, mock_storage_config):
        """Test that coordinator nodes are filtered."""
        node = MagicMock()
        node.is_active = True
        node.best_ip = "10.0.0.1"
        node.ssh_user = "root"
        node.is_coordinator = True

        mock_get_nodes.return_value = {"node1": node}

        daemon = ExternalDriveSyncDaemon(storage_config=mock_storage_config)
        nodes = daemon._get_sync_source_nodes()

        assert len(nodes) == 0

    @patch("app.coordination.external_drive_sync.get_cluster_nodes")
    def test_includes_valid_nodes(self, mock_get_nodes, mock_storage_config):
        """Test that valid nodes are included."""
        node = MagicMock()
        node.is_active = True
        node.best_ip = "10.0.0.1"
        node.ssh_user = "root"
        node.is_coordinator = False

        mock_get_nodes.return_value = {"node1": node}

        daemon = ExternalDriveSyncDaemon(storage_config=mock_storage_config)
        nodes = daemon._get_sync_source_nodes()

        assert len(nodes) == 1
        assert nodes[0] == node

    @patch("app.coordination.external_drive_sync.get_cluster_nodes")
    def test_handles_exception(self, mock_get_nodes, mock_storage_config):
        """Test that exceptions return empty list."""
        mock_get_nodes.side_effect = Exception("Network error")

        daemon = ExternalDriveSyncDaemon(storage_config=mock_storage_config)
        nodes = daemon._get_sync_source_nodes()

        assert len(nodes) == 0


# =============================================================================
# AWS Credentials Tests
# =============================================================================


class TestHasAwsCredentials:
    """Tests for AWS credentials detection."""

    def test_has_env_var(self, mock_storage_config):
        """Test detection via environment variable."""
        daemon = ExternalDriveSyncDaemon(storage_config=mock_storage_config)

        with patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key"}):
            assert daemon._has_aws_credentials() is True

    def test_no_credentials(self, mock_storage_config):
        """Test no credentials available."""
        daemon = ExternalDriveSyncDaemon(storage_config=mock_storage_config)

        with patch.dict(os.environ, {}, clear=True):
            # Also mock the credentials file not existing
            with patch.object(Path, "exists", return_value=False):
                assert daemon._has_aws_credentials() is False


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_returns_same_instance(self):
        """Test that get returns the same instance."""
        # Note: This test uses auto-detected storage which may be None
        daemon1 = get_external_drive_sync_daemon()
        daemon2 = get_external_drive_sync_daemon()

        assert daemon1 is daemon2

    def test_reset_clears_instance(self):
        """Test that reset clears the singleton."""
        daemon1 = get_external_drive_sync_daemon()
        reset_external_drive_sync_daemon()
        daemon2 = get_external_drive_sync_daemon()

        assert daemon1 is not daemon2


# =============================================================================
# Start/Stop Tests
# =============================================================================


class TestStartStop:
    """Tests for daemon start/stop."""

    @pytest.mark.asyncio
    async def test_start_disabled(self, mock_storage_config):
        """Test start when daemon is disabled."""
        mock_storage_config.path = "/nonexistent/path"
        daemon = ExternalDriveSyncDaemon(storage_config=mock_storage_config)

        await daemon.start()

        assert daemon._running is False

    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self, mock_storage_config, temp_storage_path):
        """Test stop sets running to false."""
        mock_storage_config.path = temp_storage_path
        daemon = ExternalDriveSyncDaemon(storage_config=mock_storage_config)
        daemon._running = True

        await daemon.stop()

        assert daemon._running is False


# =============================================================================
# Sync Games Tests
# =============================================================================


class TestSyncGamesFromNode:
    """Tests for games sync from individual nodes."""

    @pytest.mark.asyncio
    async def test_sync_success(
        self, mock_storage_config, mock_cluster_node, temp_storage_path
    ):
        """Test successful games sync from node."""
        mock_storage_config.path = temp_storage_path
        daemon = ExternalDriveSyncDaemon(storage_config=mock_storage_config)

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_exec.return_value = mock_proc

            dest_base = Path(temp_storage_path) / "games"
            dest_base.mkdir(parents=True, exist_ok=True)

            result = await daemon._sync_games_from_node(mock_cluster_node, dest_base)

            assert result is True
            assert daemon._stats["nodes_synced"] == 1
            assert daemon._stats["games_synced"] == 1

    @pytest.mark.asyncio
    async def test_sync_failure(
        self, mock_storage_config, mock_cluster_node, temp_storage_path
    ):
        """Test failed games sync from node."""
        mock_storage_config.path = temp_storage_path
        daemon = ExternalDriveSyncDaemon(storage_config=mock_storage_config)

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = MagicMock()
            mock_proc.returncode = 1
            mock_proc.communicate = AsyncMock(return_value=(b"", b"rsync error"))
            mock_exec.return_value = mock_proc

            dest_base = Path(temp_storage_path) / "games"
            dest_base.mkdir(parents=True, exist_ok=True)

            result = await daemon._sync_games_from_node(mock_cluster_node, dest_base)

            assert result is False
            assert daemon._stats["sync_errors"] == 1

    @pytest.mark.asyncio
    async def test_sync_exception(
        self, mock_storage_config, mock_cluster_node, temp_storage_path
    ):
        """Test exception during games sync."""
        mock_storage_config.path = temp_storage_path
        daemon = ExternalDriveSyncDaemon(storage_config=mock_storage_config)

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.side_effect = Exception("Connection refused")

            dest_base = Path(temp_storage_path) / "games"
            dest_base.mkdir(parents=True, exist_ok=True)

            result = await daemon._sync_games_from_node(mock_cluster_node, dest_base)

            assert result is False
            assert daemon._stats["sync_errors"] == 1
