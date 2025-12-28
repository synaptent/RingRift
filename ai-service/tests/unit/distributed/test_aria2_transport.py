"""Unit tests for aria2_transport.py.

Tests the Aria2Transport class for high-performance multi-connection downloads.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.distributed.aria2_transport import (
    Aria2Config,
    Aria2SyncResult,
    Aria2Transport,
    FileInfo,
    NodeInventory,
    SyncResult,
    check_aria2_available,
    create_aria2_transport,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def aria2_config():
    """Create a test Aria2Config."""
    return Aria2Config(
        connections_per_server=4,
        split=4,
        max_concurrent_downloads=2,
        timeout=30,
        data_server_port=8766,
    )


@pytest.fixture
def transport(aria2_config):
    """Create an Aria2Transport instance for testing."""
    return Aria2Transport(aria2_config)


# =============================================================================
# Test Aria2Config Dataclass
# =============================================================================


class TestAria2Config:
    """Tests for Aria2Config dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = Aria2Config()
        assert config.connections_per_server == 16
        assert config.split == 16
        assert config.min_split_size == "1M"
        assert config.max_concurrent_downloads == 5
        assert config.connect_timeout == 10
        assert config.timeout == 300
        assert config.retry_wait == 3
        assert config.max_tries == 5
        assert config.continue_download is True
        assert config.check_integrity is True
        assert config.allow_overwrite is True
        assert config.data_server_port == 8766
        assert config.verify_checksum is True

    def test_bittorrent_defaults(self):
        """Test BitTorrent configuration defaults."""
        config = Aria2Config()
        assert config.enable_bittorrent is True
        assert config.bt_enable_dht is True
        assert config.bt_enable_lpd is True
        assert config.bt_enable_pex is True
        assert config.bt_max_peers == 55
        assert config.bt_tracker_timeout == 60
        assert config.bt_listen_port == 51413
        assert config.bt_dht_listen_port == 6881
        assert config.seed_ratio == 2.0
        assert config.seed_time == 3600

    def test_large_file_threshold(self):
        """Test large file threshold configuration."""
        config = Aria2Config()
        assert config.prefer_torrent_for_large_files is True
        assert config.large_file_threshold_bytes == 50_000_000

    def test_custom_config(self):
        """Test custom configuration values."""
        config = Aria2Config(
            connections_per_server=8,
            timeout=60,
            max_tries=10,
            seed_ratio=1.5,
        )
        assert config.connections_per_server == 8
        assert config.timeout == 60
        assert config.max_tries == 10
        assert config.seed_ratio == 1.5


# =============================================================================
# Test FileInfo Dataclass
# =============================================================================


class TestFileInfo:
    """Tests for FileInfo dataclass."""

    def test_basic_file_info(self):
        """Test basic FileInfo creation."""
        info = FileInfo(
            name="test.db",
            path="games/test.db",
            size_bytes=1024,
            mtime=1704067200.0,
            category="games",
        )
        assert info.name == "test.db"
        assert info.path == "games/test.db"
        assert info.size_bytes == 1024
        assert info.mtime == 1704067200.0
        assert info.category == "games"
        assert info.checksum is None
        assert info.sources == []

    def test_file_info_with_checksum(self):
        """Test FileInfo with checksum and sources."""
        info = FileInfo(
            name="model.pth",
            path="models/model.pth",
            size_bytes=104857600,
            mtime=1704067200.0,
            category="models",
            checksum="abc123def456",
            sources=["http://node1:8766/models/model.pth"],
        )
        assert info.checksum == "abc123def456"
        assert len(info.sources) == 1
        assert "node1" in info.sources[0]


# =============================================================================
# Test NodeInventory Dataclass
# =============================================================================


class TestNodeInventory:
    """Tests for NodeInventory dataclass."""

    def test_empty_inventory(self):
        """Test empty NodeInventory."""
        inv = NodeInventory(url="http://node1:8766")
        assert inv.url == "http://node1:8766"
        assert inv.hostname == ""
        assert inv.files == {}
        assert inv.reachable is False
        assert inv.total_size_mb == 0

    def test_inventory_with_files(self):
        """Test NodeInventory with files."""
        files = {
            "games/test.db": FileInfo(
                name="test.db",
                path="games/test.db",
                size_bytes=1024,
                mtime=time.time(),
                category="games",
            )
        }
        inv = NodeInventory(
            url="http://node1:8766",
            hostname="node1",
            files=files,
            reachable=True,
            total_size_mb=0.001,
        )
        assert inv.reachable is True
        assert len(inv.files) == 1
        assert "test.db" in inv.files["games/test.db"].name


# =============================================================================
# Test Aria2SyncResult Dataclass
# =============================================================================


class TestAria2SyncResult:
    """Tests for Aria2SyncResult dataclass."""

    def test_default_result(self):
        """Test default sync result."""
        result = Aria2SyncResult(success=True)
        assert result.success is True
        assert result.files_synced == 0
        assert result.files_failed == 0
        assert result.bytes_transferred == 0
        assert result.duration_seconds == 0.0
        assert result.errors == []
        assert result.method == "aria2"

    def test_failed_result(self):
        """Test failed sync result."""
        result = Aria2SyncResult(
            success=False,
            files_failed=5,
            errors=["Network error", "Timeout"],
            duration_seconds=30.5,
        )
        assert result.success is False
        assert result.files_failed == 5
        assert len(result.errors) == 2

    def test_sync_result_alias(self):
        """Test SyncResult is alias for Aria2SyncResult."""
        assert SyncResult is Aria2SyncResult


# =============================================================================
# Test check_aria2_available
# =============================================================================


class TestCheckAria2Available:
    """Tests for check_aria2_available function."""

    def test_aria2_available(self):
        """Test when aria2c is available."""
        with patch.object(shutil, "which", return_value="/usr/bin/aria2c"):
            assert check_aria2_available() is True

    def test_aria2_not_available(self):
        """Test when aria2c is not available."""
        with patch.object(shutil, "which", return_value=None):
            assert check_aria2_available() is False


# =============================================================================
# Test Aria2Transport Class
# =============================================================================


class TestAria2TransportInit:
    """Tests for Aria2Transport initialization."""

    def test_default_init(self):
        """Test default initialization."""
        transport = Aria2Transport()
        assert transport.config is not None
        assert transport.config.connections_per_server == 16

    def test_custom_config_init(self):
        """Test initialization with custom config."""
        config = Aria2Config(connections_per_server=8)
        transport = Aria2Transport(config)
        assert transport.config.connections_per_server == 8

    def test_session_initially_none(self):
        """Test that session is initially None."""
        transport = Aria2Transport()
        assert transport._session is None


class TestAria2TransportIsAvailable:
    """Tests for is_available method."""

    def test_caches_result(self):
        """Test that availability check is cached."""
        transport = Aria2Transport()
        with patch.object(shutil, "which", return_value="/usr/bin/aria2c") as mock:
            assert transport.is_available() is True
            assert transport.is_available() is True
            # Should only call which() once
            mock.assert_called_once()

    def test_returns_false_when_unavailable(self):
        """Test returns False when aria2c not found."""
        transport = Aria2Transport()
        with patch.object(shutil, "which", return_value=None):
            assert transport.is_available() is False


class TestAria2TransportParseInventory:
    """Tests for _parse_inventory method."""

    def test_parse_empty_inventory(self):
        """Test parsing empty inventory response."""
        transport = Aria2Transport()
        data = {"files": {}, "hostname": "node1"}
        result = transport._parse_inventory("http://node1:8766", data)

        assert isinstance(result, NodeInventory)
        assert result.hostname == "node1"
        assert result.files == {}
        assert result.reachable is True

    def test_parse_files_map(self):
        """Test parsing files from files map."""
        transport = Aria2Transport()
        data = {
            "hostname": "node1",
            "files": {
                "games/test.db": {
                    "name": "test.db",
                    "size_bytes": 1024,
                    "mtime": 1704067200.0,
                    "category": "games",
                }
            },
        }
        result = transport._parse_inventory("http://node1:8766", data)

        assert len(result.files) == 1
        assert "games/test.db" in result.files
        file_info = result.files["games/test.db"]
        assert file_info.name == "test.db"
        assert file_info.size_bytes == 1024
        assert file_info.category == "games"

    def test_parse_category_arrays(self):
        """Test parsing category-based arrays."""
        transport = Aria2Transport()
        data = {
            "hostname": "node1",
            "files": {},
            "games": [
                {
                    "name": "game1.db",
                    "path": "games/game1.db",
                    "size_bytes": 2048,
                    "mtime": 1704067200.0,
                }
            ],
            "models": [
                {
                    "name": "model.pth",
                    "path": "models/model.pth",
                    "size_bytes": 104857600,
                    "mtime": 1704067200.0,
                }
            ],
        }
        result = transport._parse_inventory("http://node1:8766", data)

        assert len(result.files) == 2
        assert result.files["games/game1.db"].category == "games"
        assert result.files["models/model.pth"].category == "models"

    def test_parse_sets_sources(self):
        """Test that sources are set correctly."""
        transport = Aria2Transport()
        data = {
            "hostname": "node1",
            "files": {
                "games/test.db": {
                    "name": "test.db",
                    "size_bytes": 1024,
                    "mtime": 1704067200.0,
                    "category": "games",
                }
            },
        }
        result = transport._parse_inventory("http://node1:8766", data)

        file_info = result.files["games/test.db"]
        assert len(file_info.sources) == 1
        assert file_info.sources[0] == "http://node1:8766/games/test.db"

    def test_parse_calculates_total_size(self):
        """Test that total size is calculated."""
        transport = Aria2Transport()
        data = {
            "hostname": "node1",
            "files": {
                "file1.db": {
                    "name": "file1.db",
                    "size_bytes": 1024 * 1024,  # 1MB
                    "mtime": 1704067200.0,
                    "category": "games",
                },
                "file2.db": {
                    "name": "file2.db",
                    "size_bytes": 2 * 1024 * 1024,  # 2MB
                    "mtime": 1704067200.0,
                    "category": "games",
                },
            },
        }
        result = transport._parse_inventory("http://node1:8766", data)

        assert result.total_size_mb == pytest.approx(3.0, rel=0.01)


class TestAria2TransportBuildCommand:
    """Tests for _build_aria2_command method."""

    def test_basic_command(self, temp_dir):
        """Test basic command construction."""
        transport = Aria2Transport(Aria2Config(
            connections_per_server=4,
            split=4,
            timeout=60,
        ))
        cmd = transport._build_aria2_command(temp_dir, urls=["http://example.com/file"])

        assert cmd[0] == "aria2c"
        assert f"--dir={temp_dir}" in cmd
        assert "--max-connection-per-server=4" in cmd
        assert "--split=4" in cmd
        assert "--timeout=60" in cmd
        assert "http://example.com/file" in cmd

    def test_bittorrent_options(self, temp_dir):
        """Test BitTorrent options are included."""
        config = Aria2Config(enable_bittorrent=True, bt_enable_dht=True)
        transport = Aria2Transport(config)
        cmd = transport._build_aria2_command(temp_dir, urls=["http://example.com/file"])

        assert "--enable-dht=true" in cmd
        assert any("--dht-listen-port=" in c for c in cmd)

    def test_bittorrent_disabled(self, temp_dir):
        """Test command when BitTorrent is disabled."""
        config = Aria2Config(enable_bittorrent=False)
        transport = Aria2Transport(config)
        cmd = transport._build_aria2_command(temp_dir, urls=["http://example.com/file"])

        assert "--enable-dht=true" not in cmd

    def test_continue_download_option(self, temp_dir):
        """Test continue download option."""
        config = Aria2Config(continue_download=True)
        transport = Aria2Transport(config)
        cmd = transport._build_aria2_command(temp_dir, urls=["http://example.com/file"])

        assert "--continue=true" in cmd

    def test_input_file(self, temp_dir):
        """Test command with input file."""
        transport = Aria2Transport()
        url_file = temp_dir / "urls.txt"
        url_file.touch()
        cmd = transport._build_aria2_command(temp_dir, url_file=url_file)

        assert f"--input-file={url_file}" in cmd


class TestAria2TransportResolveCategoryDir:
    """Tests for _resolve_category_dir method."""

    def test_games_category(self, temp_dir):
        """Test resolving games category directory."""
        transport = Aria2Transport()
        result = transport._resolve_category_dir(temp_dir, "games")
        assert result == temp_dir / "games"

    def test_games_dir_already_named(self, temp_dir):
        """Test when local_dir is already named 'games'."""
        games_dir = temp_dir / "games"
        games_dir.mkdir()
        transport = Aria2Transport()
        result = transport._resolve_category_dir(games_dir, "games")
        assert result == games_dir

    def test_synced_subdir(self, temp_dir):
        """Test synced subdirectory pattern."""
        synced_dir = temp_dir / "games" / "synced"
        synced_dir.mkdir(parents=True)
        transport = Aria2Transport()
        result = transport._resolve_category_dir(synced_dir, "games")
        assert result == synced_dir

    def test_models_category(self, temp_dir):
        """Test resolving models category directory."""
        models_dir = temp_dir / "models"
        models_dir.mkdir()
        transport = Aria2Transport()
        result = transport._resolve_category_dir(models_dir, "models")
        assert result == models_dir


# =============================================================================
# Test Async Methods
# =============================================================================


class TestAria2TransportFetchInventory:
    """Tests for fetch_inventory method."""

    @pytest.mark.asyncio
    async def test_fetch_inventory_success(self, transport):
        """Test successful inventory fetch."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "hostname": "node1",
            "files": {
                "games/test.db": {
                    "name": "test.db",
                    "size_bytes": 1024,
                    "mtime": 1704067200.0,
                    "category": "games",
                }
            },
        })

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)))

        with patch.object(transport, "_get_session", return_value=mock_session):
            result = await transport.fetch_inventory("http://node1:8766")

        assert result is not None
        assert result.reachable is True
        assert len(result.files) == 1

    @pytest.mark.asyncio
    async def test_fetch_inventory_timeout(self, transport):
        """Test inventory fetch with timeout."""
        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=asyncio.TimeoutError())

        with patch.object(transport, "_get_session", return_value=mock_session):
            result = await transport.fetch_inventory("http://node1:8766")

        assert result is None


class TestAria2TransportDiscoverSources:
    """Tests for discover_sources method."""

    @pytest.mark.asyncio
    async def test_discover_sources_aggregates_files(self, transport):
        """Test that discover_sources aggregates files from multiple nodes."""
        inv1 = NodeInventory(
            url="http://node1:8766",
            files={
                "games/test.db": FileInfo(
                    name="test.db",
                    path="games/test.db",
                    size_bytes=1024,
                    mtime=time.time(),
                    category="games",
                    sources=["http://node1:8766/games/test.db"],
                )
            },
            reachable=True,
        )
        inv2 = NodeInventory(
            url="http://node2:8766",
            files={
                "games/test.db": FileInfo(
                    name="test.db",
                    path="games/test.db",
                    size_bytes=1024,
                    mtime=time.time(),
                    category="games",
                    sources=["http://node2:8766/games/test.db"],
                )
            },
            reachable=True,
        )

        with patch.object(transport, "fetch_inventory", side_effect=[inv1, inv2]):
            inventories, file_sources = await transport.discover_sources(
                ["http://node1:8766", "http://node2:8766"]
            )

        assert len(inventories) == 2
        assert "test.db" in file_sources
        assert len(file_sources["test.db"]) == 2


class TestAria2TransportDownloadFile:
    """Tests for download_file method."""

    @pytest.mark.asyncio
    async def test_download_file_unavailable(self, transport, temp_dir):
        """Test download when aria2 is unavailable."""
        with patch.object(transport, "is_available", return_value=False):
            success, bytes_dl, error = await transport.download_file(
                sources=["http://example.com/file"],
                output_dir=temp_dir,
            )

        assert success is False
        assert "not available" in error.lower()

    @pytest.mark.asyncio
    async def test_download_file_no_sources(self, transport, temp_dir):
        """Test download with no sources."""
        with patch.object(transport, "is_available", return_value=True):
            success, bytes_dl, error = await transport.download_file(
                sources=[],
                output_dir=temp_dir,
            )

        assert success is False
        assert "no sources" in error.lower()


class TestAria2TransportDownloadBatch:
    """Tests for download_batch method."""

    @pytest.mark.asyncio
    async def test_download_batch_unavailable(self, transport, temp_dir):
        """Test batch download when aria2 is unavailable."""
        with patch.object(transport, "is_available", return_value=False):
            result = await transport.download_batch(
                file_sources={"file.txt": ["http://example.com/file.txt"]},
                output_dir=temp_dir,
            )

        assert result.success is False
        assert "not available" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_download_batch_empty(self, transport, temp_dir):
        """Test batch download with empty file list."""
        with patch.object(transport, "is_available", return_value=True):
            result = await transport.download_batch(
                file_sources={},
                output_dir=temp_dir,
            )

        assert result.success is True
        assert result.files_synced == 0


class TestAria2TransportClose:
    """Tests for close method."""

    @pytest.mark.asyncio
    async def test_close_session(self, transport):
        """Test closing the HTTP session."""
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()

        transport._session = mock_session
        await transport.close()

        mock_session.close.assert_called_once()
        assert transport._session is None

    @pytest.mark.asyncio
    async def test_close_already_closed(self, transport):
        """Test closing when session already closed."""
        mock_session = MagicMock()
        mock_session.closed = True

        transport._session = mock_session
        await transport.close()

        # Should not try to close again
        assert not mock_session.close.called


# =============================================================================
# Test Factory Function
# =============================================================================


class TestCreateAria2Transport:
    """Tests for create_aria2_transport factory function."""

    def test_create_with_defaults(self):
        """Test creating transport with default config."""
        transport = create_aria2_transport()
        assert isinstance(transport, Aria2Transport)
        assert transport.config.connections_per_server == 16

    def test_create_with_config_dict(self):
        """Test creating transport with config dict."""
        config = {
            "connections_per_server": 8,
            "timeout": 120,
            "max_concurrent_downloads": 10,
        }
        transport = create_aria2_transport(config)
        assert transport.config.connections_per_server == 8
        assert transport.config.timeout == 120
        assert transport.config.max_concurrent_downloads == 10

    def test_create_with_partial_config(self):
        """Test creating transport with partial config dict."""
        config = {"timeout": 60}
        transport = create_aria2_transport(config)
        assert transport.config.timeout == 60
        # Other values should be defaults
        assert transport.config.connections_per_server == 16


# =============================================================================
# Test Sync Methods (sync_games, sync_models, etc.)
# =============================================================================


class TestAria2TransportSyncGames:
    """Tests for sync_games method."""

    @pytest.mark.asyncio
    async def test_sync_games_no_sources(self, transport, temp_dir):
        """Test sync_games with no reachable sources."""
        with patch.object(transport, "discover_sources", return_value=([], {})):
            result = await transport.sync_games(
                source_urls=["http://node1:8766"],
                local_dir=temp_dir,
            )

        assert result.success is False
        assert "no reachable sources" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_sync_games_dry_run(self, transport, temp_dir):
        """Test sync_games with dry run."""
        inv = NodeInventory(
            url="http://node1:8766",
            files={
                "games/new.db": FileInfo(
                    name="new.db",
                    path="games/new.db",
                    size_bytes=1024,
                    mtime=time.time(),
                    category="games",
                    sources=["http://node1:8766/games/new.db"],
                )
            },
            reachable=True,
        )

        with patch.object(transport, "discover_sources", return_value=([inv], {"new.db": ["http://node1:8766/games/new.db"]})):
            result = await transport.sync_games(
                source_urls=["http://node1:8766"],
                local_dir=temp_dir,
                dry_run=True,
            )

        assert result.success is True
        assert result.files_synced == 0


class TestAria2TransportFullClusterSync:
    """Tests for full_cluster_sync method."""

    @pytest.mark.asyncio
    async def test_full_cluster_sync_default_categories(self, transport, temp_dir):
        """Test full_cluster_sync with default categories."""
        with patch.object(transport, "sync_games", return_value=Aria2SyncResult(success=True)):
            with patch.object(transport, "sync_models", return_value=Aria2SyncResult(success=True)):
                with patch.object(transport, "sync_training_data", return_value=Aria2SyncResult(success=True)):
                    results = await transport.full_cluster_sync(
                        source_urls=["http://node1:8766"],
                        local_dir=temp_dir,
                    )

        assert "games" in results
        assert "models" in results
        assert "training" in results

    @pytest.mark.asyncio
    async def test_full_cluster_sync_specific_categories(self, transport, temp_dir):
        """Test full_cluster_sync with specific categories."""
        with patch.object(transport, "sync_games", return_value=Aria2SyncResult(success=True)):
            results = await transport.full_cluster_sync(
                source_urls=["http://node1:8766"],
                local_dir=temp_dir,
                categories=["games"],
            )

        assert "games" in results
        assert "models" not in results
        assert "training" not in results
