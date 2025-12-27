"""Tests for DatabaseSyncManager base class.

December 2025: Comprehensive test coverage for the consolidated database
sync manager that is the base for EloSyncManager and RegistrySyncManager.

Tests cover:
- DatabaseSyncState dataclass (serialization, deserialization)
- SyncNodeInfo dataclass
- DatabaseSyncManager initialization and state management
- Transport methods (Tailscale, SSH, Vast SSH, HTTP)
- Rsync operations (pull, push)
- Node discovery (P2P, YAML)
- Callbacks (on_sync_complete, on_sync_failed)
- Utility methods (ensure_latest, get_status, health_check)
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.database_sync_manager import (
    DatabaseSyncManager,
    DatabaseSyncState,
    SyncNodeInfo,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class MockDatabaseSyncManager(DatabaseSyncManager):
    """Concrete implementation of DatabaseSyncManager for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.merge_called = False
        self.merge_result = True

    async def _merge_databases(self, remote_db_path: Path) -> bool:
        """Mock merge implementation."""
        self.merge_called = True
        return self.merge_result

    def _update_local_stats(self) -> None:
        """Mock stats update."""
        self._db_state.local_record_count = 100
        self._db_state.local_hash = "test_hash_123"

    def _get_remote_db_path(self) -> str:
        """Return test remote path."""
        return "ai-service/data/test.db"

    def _get_remote_count_query(self) -> str:
        """Return test count query."""
        return "SELECT COUNT(*) FROM test_table"


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def temp_db(temp_dir):
    """Create a temporary test database file."""
    db_path = temp_dir / "test.db"
    db_path.write_bytes(b"test database content")
    return db_path


@pytest.fixture
def state_path(temp_dir):
    """Create a path for state file."""
    return temp_dir / "sync_state.json"


@pytest.fixture
def sync_manager(temp_db, state_path):
    """Create a test sync manager instance."""
    return MockDatabaseSyncManager(
        db_path=temp_db,
        state_path=state_path,
        db_type="test",
        coordinator_host="test-coordinator",
        sync_interval=60.0,
        p2p_url="http://localhost:8770",
        enable_merge=True,
    )


# =============================================================================
# DatabaseSyncState Tests
# =============================================================================


class TestDatabaseSyncState:
    """Tests for DatabaseSyncState dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        state = DatabaseSyncState()
        assert state.last_sync_timestamp == 0.0
        assert state.synced_nodes == set()
        assert state.pending_syncs == set()
        assert state.failed_nodes == set()
        assert state.sync_count == 0
        assert state.last_error is None
        assert state.local_record_count == 0
        assert state.local_hash == ""
        assert state.synced_from == ""
        assert state.merge_conflicts == 0
        assert state.total_syncs == 0
        assert state.successful_syncs == 0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        state = DatabaseSyncState(
            last_sync_timestamp=1234567890.0,
            synced_nodes={"node1", "node2"},
            local_record_count=500,
            local_hash="abc123",
            synced_from="node1:tailscale",
            merge_conflicts=3,
            total_syncs=10,
            successful_syncs=8,
        )

        data = state.to_dict()

        assert data["last_sync_timestamp"] == 1234567890.0
        assert set(data["synced_nodes"]) == {"node1", "node2"}
        assert data["local_record_count"] == 500
        assert data["local_hash"] == "abc123"
        assert data["synced_from"] == "node1:tailscale"
        assert data["merge_conflicts"] == 3
        assert data["total_syncs"] == 10
        assert data["successful_syncs"] == 8

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "last_sync_timestamp": 1234567890.0,
            "synced_nodes": ["node1", "node2"],
            "pending_syncs": ["node3"],
            "failed_nodes": ["node4"],
            "sync_count": 5,
            "last_error": "test error",
            "local_record_count": 250,
            "local_hash": "def456",
            "synced_from": "node2:ssh",
            "merge_conflicts": 1,
            "total_syncs": 20,
            "successful_syncs": 15,
        }

        state = DatabaseSyncState.from_dict(data)

        assert state.last_sync_timestamp == 1234567890.0
        assert state.synced_nodes == {"node1", "node2"}
        assert state.pending_syncs == {"node3"}
        assert state.failed_nodes == {"node4"}
        assert state.sync_count == 5
        assert state.last_error == "test error"
        assert state.local_record_count == 250
        assert state.local_hash == "def456"
        assert state.synced_from == "node2:ssh"
        assert state.merge_conflicts == 1
        assert state.total_syncs == 20
        assert state.successful_syncs == 15

    def test_from_dict_with_missing_fields(self):
        """Test deserialization with missing optional fields."""
        data = {"last_sync_timestamp": 100.0}
        state = DatabaseSyncState.from_dict(data)

        assert state.last_sync_timestamp == 100.0
        assert state.synced_nodes == set()
        assert state.local_record_count == 0
        assert state.local_hash == ""

    def test_roundtrip(self):
        """Test serialization/deserialization roundtrip."""
        original = DatabaseSyncState(
            last_sync_timestamp=time.time(),
            synced_nodes={"a", "b", "c"},
            pending_syncs={"d"},
            failed_nodes={"e"},
            sync_count=42,
            last_error="some error",
            local_record_count=1000,
            local_hash="xyz789",
            synced_from="a:http",
            merge_conflicts=7,
            total_syncs=100,
            successful_syncs=93,
        )

        restored = DatabaseSyncState.from_dict(original.to_dict())

        assert restored.last_sync_timestamp == original.last_sync_timestamp
        assert restored.synced_nodes == original.synced_nodes
        assert restored.pending_syncs == original.pending_syncs
        assert restored.failed_nodes == original.failed_nodes
        assert restored.sync_count == original.sync_count
        assert restored.last_error == original.last_error
        assert restored.local_record_count == original.local_record_count
        assert restored.local_hash == original.local_hash
        assert restored.synced_from == original.synced_from
        assert restored.merge_conflicts == original.merge_conflicts
        assert restored.total_syncs == original.total_syncs
        assert restored.successful_syncs == original.successful_syncs


# =============================================================================
# SyncNodeInfo Tests
# =============================================================================


class TestSyncNodeInfo:
    """Tests for SyncNodeInfo dataclass."""

    def test_default_values(self):
        """Test default initialization with required name."""
        node = SyncNodeInfo(name="test-node")

        assert node.name == "test-node"
        assert node.tailscale_ip is None
        assert node.ssh_host is None
        assert node.ssh_port == 22
        assert node.http_url is None
        assert node.vast_ssh_host is None
        assert node.vast_ssh_port is None
        assert node.is_coordinator is False
        assert node.last_seen == 0.0
        assert node.reachable is True
        assert node.remote_db_path == ""
        assert node.record_count == 0

    def test_full_initialization(self):
        """Test initialization with all fields."""
        node = SyncNodeInfo(
            name="gpu-node-1",
            tailscale_ip="100.123.45.67",
            ssh_host="192.168.1.100",
            ssh_port=2222,
            http_url="http://gpu-node-1:8770",
            vast_ssh_host="ssh.vast.ai",
            vast_ssh_port=12345,
            is_coordinator=True,
            last_seen=1234567890.0,
            reachable=False,
            remote_db_path="/workspace/data/test.db",
            record_count=500,
        )

        assert node.name == "gpu-node-1"
        assert node.tailscale_ip == "100.123.45.67"
        assert node.ssh_host == "192.168.1.100"
        assert node.ssh_port == 2222
        assert node.http_url == "http://gpu-node-1:8770"
        assert node.vast_ssh_host == "ssh.vast.ai"
        assert node.vast_ssh_port == 12345
        assert node.is_coordinator is True
        assert node.last_seen == 1234567890.0
        assert node.reachable is False
        assert node.remote_db_path == "/workspace/data/test.db"
        assert node.record_count == 500


# =============================================================================
# DatabaseSyncManager Initialization Tests
# =============================================================================


class TestDatabaseSyncManagerInit:
    """Tests for DatabaseSyncManager initialization."""

    def test_basic_initialization(self, temp_db, state_path):
        """Test basic manager initialization."""
        manager = MockDatabaseSyncManager(
            db_path=temp_db,
            state_path=state_path,
            db_type="test",
        )

        assert manager.db_path == temp_db
        assert manager.state_path == state_path
        assert manager.db_type == "test"
        assert manager.coordinator_host == "nebius-backbone-1"  # Default
        assert manager.sync_interval == 300.0  # Default
        assert manager.enable_merge is True

    def test_custom_configuration(self, temp_db, state_path):
        """Test manager with custom configuration."""
        manager = MockDatabaseSyncManager(
            db_path=temp_db,
            state_path=state_path,
            db_type="custom",
            coordinator_host="custom-coordinator",
            sync_interval=120.0,
            p2p_url="http://custom:9000",
            enable_merge=False,
        )

        assert manager.coordinator_host == "custom-coordinator"
        assert manager.sync_interval == 120.0
        assert manager.p2p_url == "http://custom:9000"
        assert manager.enable_merge is False

    def test_nodes_initialized_empty(self, sync_manager):
        """Test that nodes dict starts empty."""
        assert sync_manager.nodes == {}

    def test_callbacks_initialized_empty(self, sync_manager):
        """Test that callback lists start empty."""
        assert sync_manager._on_sync_complete_callbacks == []
        assert sync_manager._on_sync_failed_callbacks == []


# =============================================================================
# State Persistence Tests
# =============================================================================


class TestStatePersistence:
    """Tests for state loading and saving."""

    def test_save_state(self, sync_manager, state_path):
        """Test state is saved correctly."""
        sync_manager._db_state.local_record_count = 42
        sync_manager._db_state.local_hash = "saved_hash"
        sync_manager._db_state.synced_from = "test:tailscale"

        sync_manager._save_db_state()

        assert state_path.exists()
        with open(state_path) as f:
            data = json.load(f)
        assert data["local_record_count"] == 42
        assert data["local_hash"] == "saved_hash"
        assert data["synced_from"] == "test:tailscale"

    def test_load_existing_state(self, temp_db, state_path):
        """Test state is loaded from existing file."""
        # Pre-create state file
        state_data = {
            "last_sync_timestamp": 1234567890.0,
            "synced_nodes": ["node1"],
            "local_record_count": 999,
            "local_hash": "loaded_hash",
            "synced_from": "node1:ssh",
            "merge_conflicts": 5,
            "total_syncs": 50,
            "successful_syncs": 45,
        }
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, "w") as f:
            json.dump(state_data, f)

        # Create manager - should load state
        manager = MockDatabaseSyncManager(
            db_path=temp_db,
            state_path=state_path,
            db_type="test",
        )

        assert manager._db_state.last_sync_timestamp == 1234567890.0
        assert manager._db_state.local_record_count == 999
        assert manager._db_state.local_hash == "loaded_hash"

    def test_load_missing_state(self, temp_db, temp_dir):
        """Test graceful handling of missing state file."""
        state_path = temp_dir / "nonexistent_state.json"

        manager = MockDatabaseSyncManager(
            db_path=temp_db,
            state_path=state_path,
            db_type="test",
        )

        # Should use default state
        assert manager._db_state.local_record_count == 0
        assert manager._db_state.local_hash == ""

    def test_load_corrupted_state(self, temp_db, state_path):
        """Test graceful handling of corrupted state file."""
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("not valid json {{{")

        manager = MockDatabaseSyncManager(
            db_path=temp_db,
            state_path=state_path,
            db_type="test",
        )

        # Should use default state
        assert manager._db_state.local_record_count == 0


# =============================================================================
# Transport Method Tests
# =============================================================================


class TestTransportMethods:
    """Tests for transport methods."""

    def test_sync_via_tailscale_no_ip(self, sync_manager):
        """Test tailscale sync fails without IP."""
        node = SyncNodeInfo(name="test", tailscale_ip=None)

        result = asyncio.get_event_loop().run_until_complete(
            sync_manager._sync_via_tailscale(node)
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_sync_via_tailscale_success(self, sync_manager):
        """Test successful tailscale sync."""
        node = SyncNodeInfo(
            name="test",
            tailscale_ip="100.123.45.67",
            remote_db_path="data/test.db",
        )

        with patch.object(sync_manager, "_rsync_pull", return_value=True) as mock_rsync:
            result = await sync_manager._sync_via_tailscale(node)

            assert result is True
            mock_rsync.assert_called_once_with(
                host="100.123.45.67",
                remote_path="data/test.db",
                ssh_port=22,
            )

    @pytest.mark.asyncio
    async def test_sync_via_ssh_no_host(self, sync_manager):
        """Test SSH sync fails without host."""
        node = SyncNodeInfo(name="test", ssh_host=None)

        result = await sync_manager._sync_via_ssh(node)

        assert result is False

    @pytest.mark.asyncio
    async def test_sync_via_ssh_success(self, sync_manager):
        """Test successful SSH sync."""
        node = SyncNodeInfo(
            name="test",
            ssh_host="192.168.1.100",
            ssh_port=2222,
        )

        with patch.object(sync_manager, "_rsync_pull", return_value=True) as mock_rsync:
            result = await sync_manager._sync_via_ssh(node)

            assert result is True
            mock_rsync.assert_called_once()
            call_kwargs = mock_rsync.call_args[1]
            assert call_kwargs["host"] == "192.168.1.100"
            assert call_kwargs["ssh_port"] == 2222

    @pytest.mark.asyncio
    async def test_sync_via_vast_ssh_no_host(self, sync_manager):
        """Test Vast SSH sync fails without host/port."""
        node = SyncNodeInfo(name="test", vast_ssh_host=None, vast_ssh_port=None)

        result = await sync_manager._sync_via_vast_ssh(node)

        assert result is False

    @pytest.mark.asyncio
    async def test_sync_via_vast_ssh_success(self, sync_manager):
        """Test successful Vast SSH sync with workspace path."""
        node = SyncNodeInfo(
            name="test",
            vast_ssh_host="ssh.vast.ai",
            vast_ssh_port=12345,
        )

        with patch.object(sync_manager, "_rsync_pull", return_value=True) as mock_rsync:
            result = await sync_manager._sync_via_vast_ssh(node)

            assert result is True
            call_kwargs = mock_rsync.call_args[1]
            assert call_kwargs["host"] == "ssh.vast.ai"
            assert call_kwargs["ssh_port"] == 12345
            # Vast.ai uses /workspace prefix
            assert "/workspace/ringrift/" in call_kwargs["remote_path"]

    @pytest.mark.asyncio
    async def test_sync_via_http_no_url(self, sync_manager):
        """Test HTTP sync fails without URL."""
        node = SyncNodeInfo(name="test", http_url=None)

        result = await sync_manager._sync_via_http(node)

        assert result is False


# =============================================================================
# Rsync Operation Tests
# =============================================================================


class TestRsyncOperations:
    """Tests for rsync pull and push operations."""

    @pytest.mark.asyncio
    async def test_rsync_pull_success(self, sync_manager, temp_dir):
        """Test successful rsync pull."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"success", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            with patch.object(sync_manager, "_merge_databases", return_value=True):
                result = await sync_manager._rsync_pull(
                    host="test-host",
                    remote_path="/data/test.db",
                    ssh_port=22,
                )

                assert result is True
                mock_exec.assert_called_once()
                # Check rsync command was called
                call_args = mock_exec.call_args[0]
                assert call_args[0] == "rsync"

    @pytest.mark.asyncio
    async def test_rsync_pull_failure(self, sync_manager):
        """Test rsync pull failure handling."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error message"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await sync_manager._rsync_pull(
                host="test-host",
                remote_path="/data/test.db",
            )

            assert result is False

    @pytest.mark.asyncio
    async def test_rsync_pull_timeout(self, sync_manager):
        """Test rsync pull timeout handling."""
        mock_proc = AsyncMock()
        mock_proc.kill = MagicMock()
        mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await sync_manager._rsync_pull(
                host="test-host",
                remote_path="/data/test.db",
                timeout=0.1,
            )

            assert result is False
            mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_rsync_push_no_local_db(self, sync_manager, temp_dir):
        """Test rsync push fails when local DB doesn't exist."""
        sync_manager.db_path = temp_dir / "nonexistent.db"

        result = await sync_manager._rsync_push(
            host="test-host",
            remote_path="/data/test.db",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_rsync_push_success(self, sync_manager):
        """Test successful rsync push."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"success", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await sync_manager._rsync_push(
                host="test-host",
                remote_path="/data/test.db",
            )

            assert result is True


# =============================================================================
# Node Discovery Tests
# =============================================================================


class TestNodeDiscovery:
    """Tests for node discovery methods."""

    @pytest.mark.asyncio
    async def test_discover_nodes_from_p2p(self, sync_manager):
        """Test node discovery from P2P status endpoint."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "peers": [
                {
                    "node_id": "gpu-node-1",
                    "tailscale_ip": "100.123.1.1",
                    "ssh_host": "192.168.1.1",
                    "ssh_port": 22,
                    "http_url": "http://gpu-node-1:8770",
                },
                {
                    "node_id": "gpu-node-2",
                    "tailscale_ip": "100.123.1.2",
                },
            ],
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock()

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await sync_manager.discover_nodes()

            assert "gpu-node-1" in sync_manager.nodes
            assert "gpu-node-2" in sync_manager.nodes
            node1 = sync_manager.nodes["gpu-node-1"]
            assert node1.tailscale_ip == "100.123.1.1"
            assert node1.ssh_host == "192.168.1.1"
            assert node1.http_url == "http://gpu-node-1:8770"

    @pytest.mark.asyncio
    async def test_discover_nodes_fallback_to_yaml(self, sync_manager):
        """Test fallback to YAML when P2P unavailable."""
        # Mock P2P failure
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get = MagicMock(side_effect=Exception("Connection failed"))
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()
            mock_session_class.return_value = mock_session

            # Mock YAML discovery
            with patch.object(
                sync_manager, "_discover_nodes_from_yaml", new_callable=AsyncMock
            ) as mock_yaml:
                await sync_manager.discover_nodes()

                mock_yaml.assert_called_once()


# =============================================================================
# Callback Tests
# =============================================================================


class TestCallbacks:
    """Tests for sync callbacks."""

    def test_register_sync_complete_callback(self, sync_manager):
        """Test registering sync complete callback."""
        callback = MagicMock()

        sync_manager.on_sync_complete(callback)

        assert callback in sync_manager._on_sync_complete_callbacks

    def test_register_sync_failed_callback(self, sync_manager):
        """Test registering sync failed callback."""
        callback = MagicMock()

        sync_manager.on_sync_failed(callback)

        assert callback in sync_manager._on_sync_failed_callbacks

    @pytest.mark.asyncio
    async def test_notify_sync_complete(self, sync_manager):
        """Test sync complete notification."""
        sync_callback = MagicMock()
        async_callback = AsyncMock()

        sync_manager.on_sync_complete(sync_callback)
        sync_manager.on_sync_complete(async_callback)

        await sync_manager._notify_sync_complete("test-node", "tailscale")

        sync_callback.assert_called_once()
        async_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_notify_sync_failed(self, sync_manager):
        """Test sync failed notification."""
        sync_callback = MagicMock()
        async_callback = AsyncMock()

        sync_manager.on_sync_failed(sync_callback)
        sync_manager.on_sync_failed(async_callback)

        await sync_manager._notify_sync_failed("test-node", "timeout")

        sync_callback.assert_called_once()
        async_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_callback_error_handled(self, sync_manager):
        """Test callback errors are caught and logged."""
        bad_callback = MagicMock(side_effect=Exception("Callback error"))
        good_callback = MagicMock()

        sync_manager.on_sync_complete(bad_callback)
        sync_manager.on_sync_complete(good_callback)

        # Should not raise, and should call subsequent callbacks
        await sync_manager._notify_sync_complete("test-node", "ssh")

        bad_callback.assert_called_once()
        good_callback.assert_called_once()


# =============================================================================
# Utility Method Tests
# =============================================================================


class TestUtilityMethods:
    """Tests for utility methods."""

    @pytest.mark.asyncio
    async def test_ensure_latest_recent_sync(self, sync_manager):
        """Test ensure_latest returns True if recently synced."""
        sync_manager._db_state.last_sync_timestamp = time.time()

        result = await sync_manager.ensure_latest()

        assert result is True

    @pytest.mark.asyncio
    async def test_ensure_latest_triggers_sync(self, sync_manager):
        """Test ensure_latest triggers sync if stale."""
        sync_manager._db_state.last_sync_timestamp = 0  # Very old

        with patch.object(
            sync_manager, "sync_with_cluster", new_callable=AsyncMock
        ) as mock_sync:
            mock_sync.return_value = {"node1": True}

            result = await sync_manager.ensure_latest()

            assert result is True
            mock_sync.assert_called_once()

    def test_get_status(self, sync_manager):
        """Test get_status returns comprehensive info."""
        sync_manager.nodes["node1"] = SyncNodeInfo(name="node1")
        sync_manager._db_state.local_record_count = 100

        status = sync_manager.get_status()

        assert status["db_type"] == "test"
        assert status["db_path"] == str(sync_manager.db_path)
        assert status["db_exists"] is True
        assert status["node_count"] == 1
        assert "node1" in status["nodes"]
        assert status["db_state"]["local_record_count"] == 100

    def test_get_nodes(self, sync_manager):
        """Test _get_nodes returns node list."""
        sync_manager.nodes = {
            "node1": SyncNodeInfo(name="node1"),
            "node2": SyncNodeInfo(name="node2"),
            "node3": SyncNodeInfo(name="node3"),
        }

        nodes = sync_manager._get_nodes()

        assert set(nodes) == {"node1", "node2", "node3"}


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health_check method."""

    def test_health_check_not_running(self, sync_manager):
        """Test health check when not running."""
        sync_manager._running = False

        result = sync_manager.health_check()

        assert result.healthy is False
        assert "not running" in result.message

    def test_health_check_missing_db(self, sync_manager, temp_dir):
        """Test health check when database missing."""
        sync_manager._running = True
        sync_manager.db_path = temp_dir / "nonexistent.db"

        result = sync_manager.health_check()

        assert result.healthy is False
        assert "not found" in result.message

    def test_health_check_healthy(self, sync_manager):
        """Test health check when healthy."""
        sync_manager._running = True
        sync_manager._db_state.last_sync_timestamp = time.time()
        sync_manager._db_state.total_syncs = 10
        sync_manager._db_state.successful_syncs = 9
        sync_manager._db_state.local_record_count = 500

        result = sync_manager.health_check()

        assert result.healthy is True
        assert result.details["local_record_count"] == 500
        assert result.details["sync_rate"] == 0.9

    def test_health_check_degraded_low_sync_rate(self, sync_manager):
        """Test health check degraded when sync rate low."""
        sync_manager._running = True
        sync_manager._db_state.last_sync_timestamp = time.time()
        sync_manager._db_state.total_syncs = 10
        sync_manager._db_state.successful_syncs = 3  # 30% success

        result = sync_manager.health_check()

        assert result.healthy is False

    def test_health_check_degraded_old_sync(self, sync_manager):
        """Test health check degraded when sync is old."""
        sync_manager._running = True
        sync_manager._db_state.last_sync_timestamp = time.time() - 1000  # Old
        sync_manager._db_state.total_syncs = 10
        sync_manager._db_state.successful_syncs = 10
        sync_manager.sync_interval = 60  # Much smaller than age

        result = sync_manager.health_check()

        assert result.healthy is False


# =============================================================================
# Do Sync Tests
# =============================================================================


class TestDoSync:
    """Tests for _do_sync method."""

    @pytest.mark.asyncio
    async def test_do_sync_no_node_info(self, sync_manager):
        """Test _do_sync fails without node info."""
        result = await sync_manager._do_sync("unknown-node")

        assert result is False

    @pytest.mark.asyncio
    async def test_do_sync_no_transports(self, sync_manager):
        """Test _do_sync fails without available transports."""
        sync_manager.nodes["test"] = SyncNodeInfo(name="test")

        result = await sync_manager._do_sync("test")

        assert result is False

    @pytest.mark.asyncio
    async def test_do_sync_success(self, sync_manager):
        """Test successful _do_sync updates state."""
        sync_manager.nodes["test"] = SyncNodeInfo(
            name="test",
            tailscale_ip="100.123.1.1",
        )

        with patch.object(sync_manager, "_sync_via_tailscale", return_value=True):
            with patch.object(
                sync_manager, "_notify_sync_complete", new_callable=AsyncMock
            ):
                result = await sync_manager._do_sync("test")

                assert result is True
                assert "test:tailscale" in sync_manager._db_state.synced_from
                assert sync_manager._db_state.successful_syncs == 1

    @pytest.mark.asyncio
    async def test_do_sync_transport_failover(self, sync_manager):
        """Test _do_sync tries multiple transports."""
        sync_manager.nodes["test"] = SyncNodeInfo(
            name="test",
            tailscale_ip="100.123.1.1",
            ssh_host="192.168.1.1",
        )

        with patch.object(sync_manager, "_sync_via_tailscale", return_value=False):
            with patch.object(sync_manager, "_sync_via_ssh", return_value=True):
                with patch.object(
                    sync_manager, "_notify_sync_complete", new_callable=AsyncMock
                ):
                    result = await sync_manager._do_sync("test")

                    assert result is True
                    # Should have used SSH after tailscale failed
                    assert "ssh" in sync_manager._db_state.synced_from
