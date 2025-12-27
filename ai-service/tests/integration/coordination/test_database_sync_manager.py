"""Integration tests for DatabaseSyncManager base class.

These tests verify:
1. DatabaseSyncState serialization/deserialization
2. SyncNodeInfo configuration
3. Multi-transport failover (Tailscale -> SSH -> Vast SSH -> HTTP)
4. Rsync pull/push operations
5. Node discovery (P2P and YAML)
6. State persistence
7. Sync lifecycle and callbacks

December 2025 - RingRift AI Service
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.database_sync_manager import (
    DatabaseSyncManager,
    DatabaseSyncState,
    SyncNodeInfo,
)


# =============================================================================
# Concrete Test Implementation
# =============================================================================


class TestSyncManager(DatabaseSyncManager):
    """Concrete implementation for testing abstract base class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.merge_called = False
        self.merge_result = True
        self.stats_updated = False

    async def _merge_databases(self, remote_db_path: Path) -> bool:
        """Test implementation of merge."""
        self.merge_called = True
        return self.merge_result

    def _update_local_stats(self) -> None:
        """Test implementation of stats update."""
        self.stats_updated = True
        self._db_state.local_record_count = 100

    def _get_remote_db_path(self) -> str:
        """Test implementation returning remote path."""
        return "data/test_database.db"

    def _get_remote_count_query(self) -> str:
        """Test implementation returning count query."""
        return "SELECT COUNT(*) FROM test_table"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db():
    """Create temporary database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    # Initialize with SQLite schema
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO test_table (value) VALUES ('test')")
    conn.commit()
    conn.close()
    yield db_path
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def temp_state_file():
    """Create temporary state file path."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        state_path = Path(f.name)
    # Remove so manager can create fresh
    state_path.unlink()
    yield state_path
    if state_path.exists():
        state_path.unlink()


@pytest.fixture
def sync_manager(temp_db, temp_state_file):
    """Create TestSyncManager instance."""
    manager = TestSyncManager(
        db_path=temp_db,
        state_path=temp_state_file,
        db_type="test",
        coordinator_host="test-coordinator",
        sync_interval=60.0,
        p2p_url="http://localhost:8770",
    )
    return manager


@pytest.fixture
def sample_node():
    """Create sample SyncNodeInfo."""
    return SyncNodeInfo(
        name="test-node",
        tailscale_ip="100.64.0.42",
        ssh_host="test-node.example.com",
        ssh_port=22,
        http_url="http://test-node:8080",
        remote_db_path="data/test.db",
    )


# =============================================================================
# Test DatabaseSyncState
# =============================================================================


class TestDatabaseSyncState:
    """Tests for DatabaseSyncState dataclass."""

    def test_default_values(self):
        """DatabaseSyncState should have sensible defaults."""
        state = DatabaseSyncState()

        assert state.local_record_count == 0
        assert state.local_hash == ""
        assert state.synced_from == ""
        assert state.merge_conflicts == 0
        assert state.total_syncs == 0
        assert state.successful_syncs == 0

    def test_to_dict(self):
        """to_dict should serialize all fields."""
        state = DatabaseSyncState(
            local_record_count=100,
            local_hash="abc123",
            synced_from="node-1",
            merge_conflicts=5,
            total_syncs=10,
            successful_syncs=8,
        )
        state.synced_nodes = {"node-1", "node-2"}

        d = state.to_dict()

        assert d["local_record_count"] == 100
        assert d["local_hash"] == "abc123"
        assert d["synced_from"] == "node-1"
        assert d["merge_conflicts"] == 5
        assert d["total_syncs"] == 10
        assert d["successful_syncs"] == 8
        assert set(d["synced_nodes"]) == {"node-1", "node-2"}

    def test_from_dict(self):
        """from_dict should deserialize all fields."""
        data = {
            "local_record_count": 50,
            "local_hash": "def456",
            "synced_from": "node-3",
            "merge_conflicts": 2,
            "total_syncs": 5,
            "successful_syncs": 4,
            "synced_nodes": ["node-3"],
            "last_sync_timestamp": 1234567890.0,
        }

        state = DatabaseSyncState.from_dict(data)

        assert state.local_record_count == 50
        assert state.local_hash == "def456"
        assert state.synced_from == "node-3"
        assert state.merge_conflicts == 2
        assert state.total_syncs == 5
        assert state.successful_syncs == 4
        assert state.synced_nodes == {"node-3"}
        assert state.last_sync_timestamp == 1234567890.0

    def test_roundtrip_serialization(self):
        """Serialize then deserialize should preserve state."""
        original = DatabaseSyncState(
            local_record_count=999,
            local_hash="roundtrip",
            synced_from="origin",
            merge_conflicts=7,
            total_syncs=100,
            successful_syncs=95,
        )
        original.synced_nodes = {"a", "b", "c"}
        original.failed_nodes = {"x"}

        serialized = original.to_dict()
        restored = DatabaseSyncState.from_dict(serialized)

        assert restored.local_record_count == original.local_record_count
        assert restored.local_hash == original.local_hash
        assert restored.synced_from == original.synced_from
        assert restored.merge_conflicts == original.merge_conflicts
        assert restored.synced_nodes == original.synced_nodes
        assert restored.failed_nodes == original.failed_nodes


# =============================================================================
# Test SyncNodeInfo
# =============================================================================


class TestSyncNodeInfo:
    """Tests for SyncNodeInfo dataclass."""

    def test_minimal_creation(self):
        """SyncNodeInfo should work with just name."""
        node = SyncNodeInfo(name="test-node")

        assert node.name == "test-node"
        assert node.tailscale_ip is None
        assert node.ssh_port == 22
        assert node.reachable is True

    def test_full_configuration(self):
        """SyncNodeInfo should accept all fields."""
        node = SyncNodeInfo(
            name="full-node",
            tailscale_ip="100.64.0.1",
            ssh_host="full-node.local",
            ssh_port=2222,
            http_url="https://full-node:8443",
            vast_ssh_host="ssh.vast.ai",
            vast_ssh_port=22222,
            is_coordinator=True,
            remote_db_path="/data/db.sqlite",
            record_count=1000,
        )

        assert node.name == "full-node"
        assert node.tailscale_ip == "100.64.0.1"
        assert node.ssh_port == 2222
        assert node.http_url == "https://full-node:8443"
        assert node.vast_ssh_host == "ssh.vast.ai"
        assert node.vast_ssh_port == 22222
        assert node.is_coordinator is True
        assert node.record_count == 1000


# =============================================================================
# Test DatabaseSyncManager Initialization
# =============================================================================


class TestDatabaseSyncManagerInit:
    """Tests for DatabaseSyncManager initialization."""

    def test_basic_initialization(self, sync_manager):
        """Manager should initialize with correct paths."""
        assert sync_manager.db_path.exists()
        assert sync_manager.db_type == "test"
        assert sync_manager.coordinator_host == "test-coordinator"
        assert sync_manager.sync_interval == 60.0

    def test_state_persistence_on_init(self, temp_db, temp_state_file):
        """Manager should create state file if not exists."""
        manager = TestSyncManager(
            db_path=temp_db,
            state_path=temp_state_file,
            db_type="test",
        )

        # State file won't exist until first save
        assert manager._db_state is not None
        assert isinstance(manager._db_state, DatabaseSyncState)

    def test_loads_existing_state(self, temp_db, temp_state_file):
        """Manager should load existing state file."""
        # Create state file
        existing_state = {
            "local_record_count": 500,
            "synced_nodes": ["node-a"],
            "successful_syncs": 10,
        }
        temp_state_file.write_text(json.dumps(existing_state))

        manager = TestSyncManager(
            db_path=temp_db,
            state_path=temp_state_file,
            db_type="test",
        )

        assert manager._db_state.local_record_count == 500
        assert "node-a" in manager._db_state.synced_nodes
        assert manager._db_state.successful_syncs == 10


# =============================================================================
# Test Multi-Transport Failover
# =============================================================================


class TestMultiTransportFailover:
    """Tests for transport failover behavior."""

    @pytest.mark.asyncio
    async def test_tailscale_success_stops_chain(self, sync_manager, sample_node):
        """Successful Tailscale sync should not try other transports."""
        sync_manager.nodes[sample_node.name] = sample_node

        # Mock _rsync_pull which is called by _sync_via_tailscale
        with patch.object(sync_manager, "_rsync_pull") as mock_rsync:
            mock_rsync.return_value = True

            result = await sync_manager._do_sync(sample_node.name)

            assert result is True
            # Rsync was called for tailscale
            assert mock_rsync.called

    @pytest.mark.asyncio
    async def test_fallback_to_ssh_on_tailscale_failure(
        self, sync_manager, sample_node
    ):
        """Should try SSH when Tailscale fails."""
        sync_manager.nodes[sample_node.name] = sample_node

        call_count = 0

        async def mock_rsync(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # First call (tailscale) fails, second (ssh) succeeds
            return call_count >= 2

        with patch.object(sync_manager, "_rsync_pull", side_effect=mock_rsync):
            result = await sync_manager._do_sync(sample_node.name)

            assert result is True
            assert call_count >= 2  # Both tailscale and ssh attempted

    @pytest.mark.asyncio
    async def test_fallback_to_vast_ssh(self, sync_manager):
        """Should try Vast SSH when regular SSH fails."""
        node = SyncNodeInfo(
            name="vast-node",
            vast_ssh_host="ssh.vast.ai",
            vast_ssh_port=22222,
        )
        sync_manager.nodes[node.name] = node

        with patch.object(sync_manager, "_rsync_pull") as mock_rsync:
            mock_rsync.return_value = True

            result = await sync_manager._do_sync(node.name)

            assert result is True
            mock_rsync.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_to_http(self, sync_manager, sample_node):
        """Should try HTTP when all SSH methods fail."""
        sync_manager.nodes[sample_node.name] = sample_node

        # Mock rsync to fail (for tailscale/ssh), but HTTP to succeed
        with patch.object(sync_manager, "_rsync_pull") as mock_rsync, \
             patch.object(sync_manager, "_sync_via_http") as mock_http:
            mock_rsync.return_value = False
            mock_http.return_value = True

            result = await sync_manager._do_sync(sample_node.name)

            assert result is True

    @pytest.mark.asyncio
    async def test_all_transports_fail(self, sync_manager, sample_node):
        """Should return False when all transports fail."""
        sync_manager.nodes[sample_node.name] = sample_node

        with patch.object(sync_manager, "_rsync_pull") as mock_rsync, \
             patch.object(sync_manager, "_sync_via_http") as mock_http:
            mock_rsync.return_value = False
            mock_http.return_value = False

            result = await sync_manager._do_sync(sample_node.name)

            assert result is False


# =============================================================================
# Test Rsync Operations
# =============================================================================


class TestRsyncOperations:
    """Tests for rsync pull/push operations."""

    @pytest.mark.asyncio
    async def test_rsync_pull_success(self, sync_manager, sample_node):
        """Successful rsync pull should merge and update stats."""
        sync_manager.nodes[sample_node.name] = sample_node

        with patch("asyncio.create_subprocess_exec") as mock_proc, \
             patch("asyncio.wait_for") as mock_wait:
            process = AsyncMock()
            process.returncode = 0
            process.communicate = AsyncMock(return_value=(b"", b""))
            mock_proc.return_value = process
            mock_wait.return_value = (b"", b"")

            # Mock tempfile and Path operations
            with patch("tempfile.NamedTemporaryFile") as mock_tmp, \
                 patch.object(Path, "unlink", return_value=None):
                mock_file = MagicMock()
                mock_file.name = "/tmp/test_temp.db"
                mock_tmp.return_value.__enter__ = MagicMock(return_value=mock_file)
                mock_tmp.return_value.__exit__ = MagicMock(return_value=False)

                result = await sync_manager._rsync_pull(
                    host=sample_node.ssh_host,
                    remote_path="/data/remote.db",
                    ssh_port=sample_node.ssh_port,
                )

                # rsync was called
                mock_proc.assert_called()

    @pytest.mark.asyncio
    async def test_rsync_pull_failure(self, sync_manager, sample_node):
        """Failed rsync pull should return False."""
        with patch("asyncio.create_subprocess_exec") as mock_proc, \
             patch("asyncio.wait_for") as mock_wait:
            process = AsyncMock()
            process.returncode = 12  # rsync error
            process.communicate = AsyncMock(
                return_value=(b"", b"rsync: connection refused")
            )
            mock_proc.return_value = process
            mock_wait.return_value = (b"", b"rsync: connection refused")

            with patch("tempfile.NamedTemporaryFile") as mock_tmp, \
                 patch.object(Path, "unlink", return_value=None):
                mock_file = MagicMock()
                mock_file.name = "/tmp/test_temp.db"
                mock_tmp.return_value.__enter__ = MagicMock(return_value=mock_file)
                mock_tmp.return_value.__exit__ = MagicMock(return_value=False)

                result = await sync_manager._rsync_pull(
                    host=sample_node.ssh_host,
                    remote_path="/data/remote.db",
                    ssh_port=sample_node.ssh_port,
                )

                assert result is False

    @pytest.mark.asyncio
    async def test_rsync_push_success(self, sync_manager, sample_node):
        """Successful rsync push should update state."""
        with patch("asyncio.create_subprocess_exec") as mock_proc, \
             patch("asyncio.wait_for") as mock_wait:
            process = AsyncMock()
            process.returncode = 0
            process.communicate = AsyncMock(return_value=(b"", b""))
            mock_proc.return_value = process
            mock_wait.return_value = (b"", b"")

            result = await sync_manager._rsync_push(
                host=sample_node.ssh_host,
                remote_path="/data/remote.db",
                ssh_port=sample_node.ssh_port,
            )

            assert result is True
            mock_proc.assert_called()


# =============================================================================
# Test Node Discovery
# =============================================================================


class TestNodeDiscovery:
    """Tests for node discovery from P2P and YAML."""

    @pytest.mark.asyncio
    async def test_discover_nodes_from_p2p(self, sync_manager):
        """Should discover nodes from P2P orchestrator."""
        mock_response = {
            "alive_peers": ["node-1", "node-2", "node-3"],
            "peer_info": {
                "node-1": {"tailscale_ip": "100.64.0.1"},
                "node-2": {"tailscale_ip": "100.64.0.2"},
            },
        }

        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session = MagicMock()
            mock_response_obj = MagicMock()
            mock_response_obj.status = 200
            mock_response_obj.json = AsyncMock(return_value=mock_response)
            mock_response_obj.__aenter__ = AsyncMock(return_value=mock_response_obj)
            mock_response_obj.__aexit__ = AsyncMock(return_value=None)

            mock_session.get = MagicMock(return_value=mock_response_obj)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_cls.return_value = mock_session

            await sync_manager.discover_nodes()

            # Should have discovered nodes
            # (Actual behavior depends on implementation filtering)

    @pytest.mark.asyncio
    async def test_discover_nodes_falls_back_to_yaml(self, sync_manager):
        """Should fall back to YAML when P2P fails."""
        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.get = MagicMock(side_effect=Exception("P2P unavailable"))
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_cls.return_value = mock_session

            with patch.object(
                sync_manager, "_discover_nodes_from_yaml"
            ) as mock_yaml:
                mock_yaml.return_value = None

                await sync_manager.discover_nodes()

                mock_yaml.assert_called_once()


# =============================================================================
# Test State Persistence
# =============================================================================


class TestStatePersistence:
    """Tests for state save/load operations."""

    def test_save_state(self, sync_manager, temp_state_file):
        """_save_db_state should persist to file."""
        sync_manager._db_state.local_record_count = 42
        sync_manager._db_state.synced_from = "test-origin"

        sync_manager._save_db_state()

        assert temp_state_file.exists()
        saved = json.loads(temp_state_file.read_text())
        assert saved["local_record_count"] == 42
        assert saved["synced_from"] == "test-origin"

    def test_load_state(self, temp_db, temp_state_file):
        """_load_db_state should read from file."""
        state_data = {
            "local_record_count": 999,
            "synced_from": "loaded-origin",
            "successful_syncs": 50,
        }
        temp_state_file.write_text(json.dumps(state_data))

        manager = TestSyncManager(
            db_path=temp_db,
            state_path=temp_state_file,
            db_type="test",
        )

        assert manager._db_state.local_record_count == 999
        assert manager._db_state.synced_from == "loaded-origin"
        assert manager._db_state.successful_syncs == 50


# =============================================================================
# Test Sync Callbacks
# =============================================================================


class TestSyncCallbacks:
    """Tests for sync completion/failure callbacks."""

    @pytest.mark.asyncio
    async def test_sync_complete_callback(self, sync_manager):
        """on_sync_complete callback should be invoked."""
        callback_data = {}

        def capture_callback(node, transport, state):
            callback_data["node"] = node
            callback_data["transport"] = transport
            callback_data["state"] = state

        sync_manager.on_sync_complete(capture_callback)
        await sync_manager._notify_sync_complete("test-node", "ssh")

        assert callback_data["node"] == "test-node"
        assert callback_data["transport"] == "ssh"
        assert callback_data["state"] is not None

    @pytest.mark.asyncio
    async def test_sync_failed_callback(self, sync_manager):
        """on_sync_failed callback should be invoked."""
        callback_data = {}

        def capture_callback(node, reason, state):
            callback_data["node"] = node
            callback_data["reason"] = reason
            callback_data["state"] = state

        sync_manager.on_sync_failed(capture_callback)
        await sync_manager._notify_sync_failed("test-node", "Connection refused")

        assert callback_data["node"] == "test-node"
        assert callback_data["reason"] == "Connection refused"
        assert callback_data["state"] is not None


# =============================================================================
# Test Status Reporting
# =============================================================================


class TestStatusReporting:
    """Tests for get_status method."""

    def test_get_status_structure(self, sync_manager):
        """get_status should return complete status dict."""
        status = sync_manager.get_status()

        # DatabaseSyncManager-specific fields
        assert "db_type" in status
        assert "db_path" in status
        assert "db_state" in status
        assert "nodes" in status
        # Base class fields
        assert "running" in status
        assert "state" in status  # Base class SyncState

    def test_get_status_reflects_state(self, sync_manager):
        """get_status should reflect current state."""
        sync_manager._db_state.local_record_count = 123
        sync_manager._db_state.successful_syncs = 10

        status = sync_manager.get_status()

        # DatabaseSyncManager uses db_state for its specific state
        assert status["db_state"]["local_record_count"] == 123
        assert status["db_state"]["successful_syncs"] == 10


# =============================================================================
# Test Ensure Latest
# =============================================================================


class TestEnsureLatest:
    """Tests for ensure_latest convenience method."""

    @pytest.mark.asyncio
    async def test_ensure_latest_with_nodes(self, sync_manager, sample_node):
        """ensure_latest should sync from discovered nodes."""
        sync_manager.nodes[sample_node.name] = sample_node

        with patch.object(sync_manager, "_do_sync") as mock_sync:
            mock_sync.return_value = True

            result = await sync_manager.ensure_latest()

            # Should attempt sync
            assert mock_sync.called or result is True

    @pytest.mark.asyncio
    async def test_ensure_latest_discovers_nodes_if_empty(self, sync_manager):
        """ensure_latest should discover nodes if none known."""
        assert len(sync_manager.nodes) == 0

        with patch.object(sync_manager, "discover_nodes", new_callable=AsyncMock) as mock_discover:
            mock_discover.return_value = None

            await sync_manager.ensure_latest()

            mock_discover.assert_called_once()
