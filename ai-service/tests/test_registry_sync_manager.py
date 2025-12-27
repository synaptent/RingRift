"""Tests for RegistrySyncManager.

Tests sync state management, circuit breaker logic,
and local registry stats without requiring network connections.
"""

import json
import sqlite3
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.distributed.circuit_breaker import CircuitBreaker
from app.training.registry_sync_manager import (
    NodeInfo,
    RegistrySyncManager,
    SyncState,
)


class TestSyncState:
    """Test SyncState dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        state = SyncState()
        assert state.last_sync_timestamp == 0.0
        assert state.local_model_count == 0
        assert state.local_version_count == 0
        assert state.synced_nodes == {}
        assert state.pending_syncs == []

    def test_custom_values(self):
        """Test custom initialization."""
        state = SyncState(
            last_sync_timestamp=1000.0,
            local_model_count=5,
            local_version_count=10,
            synced_nodes={"node1": 900.0},
            pending_syncs=["node2"],
        )
        assert state.last_sync_timestamp == 1000.0
        assert state.local_model_count == 5
        assert state.local_version_count == 10
        assert state.synced_nodes == {"node1": 900.0}
        assert state.pending_syncs == ["node2"]


class TestNodeInfo:
    """Test NodeInfo dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        node = NodeInfo(hostname="test-node")
        assert node.hostname == "test-node"
        assert node.registry_path == "ai-service/data/model_registry.db"
        assert node.last_seen == 0.0
        assert node.model_count == 0
        assert node.version_count == 0
        assert node.reachable is True
        assert node.tailscale_ip is None
        assert node.ssh_port == 22

    def test_custom_values(self):
        """Test custom initialization."""
        node = NodeInfo(
            hostname="gpu-node",
            tailscale_ip="100.64.0.1",
            ssh_port=2222,
            model_count=3,
        )
        assert node.hostname == "gpu-node"
        assert node.tailscale_ip == "100.64.0.1"
        assert node.ssh_port == 2222
        assert node.model_count == 3


class TestCircuitBreaker:
    """Test CircuitBreaker fault tolerance."""

    def test_initial_state(self):
        """Test initial state is closed."""
        cb = CircuitBreaker()
        assert cb.state == "closed"
        assert cb.failures == 0
        assert cb.can_attempt()

    def test_record_success_resets_failures(self):
        """Test that success resets failure count."""
        cb = CircuitBreaker(failure_threshold=3)
        cb.failures = 2
        cb.record_success()
        assert cb.failures == 0
        assert cb.state == "closed"

    def test_record_failure_increments(self):
        """Test that failures increment counter."""
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        assert cb.failures == 1
        assert cb.state == "closed"

    def test_circuit_opens_after_threshold(self):
        """Test circuit opens after failure threshold."""
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.failures == 3
        assert cb.state == "open"
        assert not cb.can_attempt()

    def test_circuit_half_open_after_timeout(self):
        """Test circuit goes half-open after recovery timeout."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"
        assert not cb.can_attempt()

        # Wait for recovery timeout
        time.sleep(1.1)
        assert cb.can_attempt()
        assert cb.state == "half-open"

    def test_custom_thresholds(self):
        """Test custom threshold values."""
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=600)
        assert cb.failure_threshold == 5
        assert cb.recovery_timeout == 600


class TestRegistrySyncManager:
    """Test RegistrySyncManager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry_path = Path(self.temp_dir) / "model_registry.db"
        self.state_path = Path(self.temp_dir) / "sync_state.json"

        # Create a test registry database
        self._create_test_registry()

    def teardown_method(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_registry(self):
        """Create a test registry database."""
        conn = sqlite3.connect(str(self.registry_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE models (
                model_id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                model_type TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT,
                version INTEGER,
                stage TEXT,
                file_path TEXT,
                file_hash TEXT,
                file_size_bytes INTEGER,
                metrics_json TEXT,
                training_config_json TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # Insert test data
        cursor.execute("""
            INSERT INTO models VALUES (?, ?, ?, ?, ?, ?)
        """, ("test_model", "Test Model", "Description", "policy_value",
              datetime.now().isoformat(), datetime.now().isoformat()))

        cursor.execute("""
            INSERT INTO versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (1, "test_model", 1, "production", "/path/to/model.pt",
              "abc123", 1024, "{}", "{}", datetime.now().isoformat(),
              datetime.now().isoformat()))

        cursor.execute("""
            INSERT INTO versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (2, "test_model", 2, "staging", "/path/to/model2.pt",
              "def456", 2048, "{}", "{}", datetime.now().isoformat(),
              datetime.now().isoformat()))

        conn.commit()
        conn.close()

    def test_initialization(self):
        """Test manager initializes correctly."""
        manager = RegistrySyncManager(registry_path=self.registry_path)
        assert manager.registry_path == self.registry_path
        assert manager.sync_interval == 600
        assert manager.state is not None

    def test_custom_sync_interval(self):
        """Test custom sync interval."""
        manager = RegistrySyncManager(
            registry_path=self.registry_path,
            sync_interval=300,
        )
        assert manager.sync_interval == 300

    def test_update_local_stats(self):
        """Test local stats are updated from registry."""
        manager = RegistrySyncManager(registry_path=self.registry_path)
        manager._update_local_stats()

        assert manager.state.local_model_count == 1
        assert manager.state.local_version_count == 2

    def test_update_local_stats_missing_registry(self):
        """Test local stats with missing registry."""
        missing_path = Path(self.temp_dir) / "nonexistent.db"
        manager = RegistrySyncManager(registry_path=missing_path)
        manager._update_local_stats()

        # Should not raise, just leave counts at 0
        assert manager.state.local_model_count == 0
        assert manager.state.local_version_count == 0

    def test_save_and_load_state(self):
        """Test state persistence."""
        manager = RegistrySyncManager(registry_path=self.registry_path)

        # Modify state
        manager.state.last_sync_timestamp = 12345.0
        manager.state.synced_nodes = {"node1": 10000.0}

        # Patch the state path to use temp directory
        with patch('app.training.registry_sync_manager.SYNC_STATE_PATH', self.state_path):
            manager._save_state()

            # Verify file exists
            assert self.state_path.exists()

            # Create new manager and load state
            manager2 = RegistrySyncManager(registry_path=self.registry_path)
            manager2._load_state()

            # State should be loaded (if file exists at default location)
            # For test, manually load from temp path
            with open(self.state_path) as f:
                data = json.load(f)

            assert data['last_sync_timestamp'] == 12345.0
            assert data['synced_nodes'] == {"node1": 10000.0}

    def test_get_sync_status(self):
        """Test get_sync_status returns expected structure."""
        manager = RegistrySyncManager(registry_path=self.registry_path)
        manager._update_local_stats()

        status = manager.get_sync_status()

        assert 'last_sync' in status
        assert 'local_models' in status
        assert 'local_versions' in status
        assert 'synced_nodes' in status
        assert 'nodes_available' in status
        assert 'circuit_breakers' in status

        assert status['local_models'] == 1
        assert status['local_versions'] == 2

    def test_circuit_breakers_per_node(self):
        """Test circuit breakers are created per node."""
        manager = RegistrySyncManager(registry_path=self.registry_path)

        # Access circuit breakers for different nodes
        cb1 = manager.circuit_breakers["node1"]
        cb2 = manager.circuit_breakers["node2"]

        assert cb1 is not cb2
        assert cb1.state == "closed"
        assert cb2.state == "closed"

        # Fail node1
        for _ in range(3):
            cb1.record_failure()

        assert cb1.state == "open"
        assert cb2.state == "closed"

    def test_on_sync_callbacks(self):
        """Test callback registration."""
        manager = RegistrySyncManager(registry_path=self.registry_path)

        callback_called = []

        def on_complete():
            callback_called.append("complete")

        def on_failed():
            callback_called.append("failed")

        manager.on_sync_complete(on_complete)
        manager.on_sync_failed(on_failed)

        assert len(manager._on_sync_complete) == 1
        assert len(manager._on_sync_failed) == 1

    def test_transport_methods_defined(self):
        """Test transport methods are defined in priority order."""
        manager = RegistrySyncManager(registry_path=self.registry_path)

        assert len(manager.transport_methods) == 3
        transport_names = [t[0] for t in manager.transport_methods]
        assert transport_names == ["tailscale", "ssh", "http"]


class TestRegistrySyncManagerAsync:
    """Test async methods of RegistrySyncManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry_path = Path(self.temp_dir) / "model_registry.db"

        # Create test registry
        conn = sqlite3.connect(str(self.registry_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE models (
                model_id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                model_type TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT,
                version INTEGER,
                stage TEXT,
                file_path TEXT,
                file_hash TEXT,
                file_size_bytes INTEGER,
                metrics_json TEXT,
                training_config_json TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        conn.commit()
        conn.close()

    def teardown_method(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test async initialization."""
        manager = RegistrySyncManager(registry_path=self.registry_path)

        # Mock node discovery to avoid file reads
        with patch.object(manager, '_discover_nodes', new_callable=AsyncMock):
            await manager.initialize()

        assert manager.state.local_model_count == 0  # Empty test DB
        assert manager.state.local_version_count == 0

    @pytest.mark.asyncio
    async def test_sync_with_cluster_no_nodes(self):
        """Test sync with no nodes configured."""
        manager = RegistrySyncManager(registry_path=self.registry_path)
        manager.nodes = {}

        result = await manager.sync_with_cluster()

        # With no nodes, sync should succeed but sync 0 nodes
        assert result['nodes_synced'] == 0
        assert result['nodes_failed'] == 0
        assert result['success'] is False  # No nodes synced

    @pytest.mark.asyncio
    async def test_sync_with_cluster_circuit_open(self):
        """Test sync skips nodes with open circuit breaker."""
        manager = RegistrySyncManager(registry_path=self.registry_path)
        manager.nodes = {"node1": NodeInfo(hostname="node1")}

        # Open the circuit breaker
        for _ in range(3):
            manager.circuit_breakers["node1"].record_failure()

        result = await manager.sync_with_cluster()

        # Node should be skipped
        assert result['nodes_synced'] == 0
        assert result['nodes_failed'] == 0

    @pytest.mark.asyncio
    async def test_sync_via_tailscale_no_ip(self):
        """Test tailscale sync fails without IP."""
        manager = RegistrySyncManager(registry_path=self.registry_path)
        node = NodeInfo(hostname="test-node", tailscale_ip=None)

        result = await manager._sync_via_tailscale("test-node", node)

        assert result['success'] is False
        assert 'No Tailscale IP' in result['error']


class TestMergeDatabases:
    """Test database merging functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.local_path = Path(self.temp_dir) / "local.db"
        self.remote_path = Path(self.temp_dir) / "remote.db"

        # Create local database
        self._create_db(self.local_path, [
            ("model1", "Model 1", "local"),
        ], [
            ("model1", 1, "production"),
        ])

        # Create remote database with additional data
        self._create_db(self.remote_path, [
            ("model1", "Model 1", "remote"),  # Same model
            ("model2", "Model 2", "remote"),  # New model
        ], [
            ("model1", 1, "production"),  # Same version
            ("model1", 2, "staging"),     # New version
            ("model2", 1, "development"), # New model version
        ])

    def teardown_method(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_db(self, path: Path, models: list, versions: list):
        """Create a test database."""
        conn = sqlite3.connect(str(path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE models (
                model_id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                model_type TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT,
                version INTEGER,
                stage TEXT,
                file_path TEXT,
                file_hash TEXT,
                file_size_bytes INTEGER,
                metrics_json TEXT,
                training_config_json TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        now = datetime.now().isoformat()
        for model_id, name, desc in models:
            cursor.execute(
                "INSERT INTO models VALUES (?, ?, ?, ?, ?, ?)",
                (model_id, name, desc, "policy_value", now, now)
            )

        for model_id, version, stage in versions:
            cursor.execute(
                "INSERT INTO versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (None, model_id, version, stage, f"/path/{model_id}_v{version}.pt",
                 "hash", 1024, "{}", "{}", now, now)
            )

        conn.commit()
        conn.close()

    @pytest.mark.asyncio
    async def test_merge_databases(self):
        """Test merging remote database into local."""
        manager = RegistrySyncManager(registry_path=self.local_path)

        result = await manager._merge_databases(self.remote_path, "remote-host")

        assert result['success']
        assert result['models_merged'] == 1   # model2
        assert result['versions_merged'] == 2  # model1 v2, model2 v1

        # Verify local database now has all data
        conn = sqlite3.connect(str(self.local_path))
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM models")
        assert cursor.fetchone()[0] == 2

        cursor.execute("SELECT COUNT(*) FROM versions")
        assert cursor.fetchone()[0] == 3

        conn.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
