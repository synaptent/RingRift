"""Tests for sync_base.py - Base class for sync managers.

December 2025: Created for comprehensive sync infrastructure testing.
"""

import asyncio
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.coordination.sync_base import (
    BaseSyncProgress,
    CircuitBreakerConfig,
    SyncManagerBase,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class MockSyncManager(SyncManagerBase):
    """Concrete implementation for testing abstract base class."""

    def __init__(self, nodes: list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        self._nodes = nodes or ["node1", "node2", "node3"]
        self._sync_results: dict[str, bool] = {}

    async def _do_sync(self, node: str) -> bool:
        """Mock sync implementation."""
        return self._sync_results.get(node, True)

    def _get_nodes(self) -> list[str]:
        """Return configured nodes."""
        return self._nodes

    def set_sync_result(self, node: str, success: bool) -> None:
        """Configure sync result for a node."""
        self._sync_results[node] = success


@pytest.fixture
def temp_state_path():
    """Create a temporary state file path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "sync_state.json"


@pytest.fixture
def sync_manager():
    """Create a mock sync manager."""
    return MockSyncManager()


@pytest.fixture
def sync_manager_with_state(temp_state_path):
    """Create a mock sync manager with state persistence."""
    return MockSyncManager(state_path=temp_state_path)


# =============================================================================
# BaseSyncProgress Tests
# =============================================================================


class TestBaseSyncProgress:
    """Tests for BaseSyncProgress dataclass."""

    def test_default_initialization(self):
        """Test default values for BaseSyncProgress."""
        progress = BaseSyncProgress()

        assert progress.last_sync_timestamp == 0.0
        assert progress.synced_nodes == set()
        assert progress.pending_syncs == set()
        assert progress.failed_nodes == set()
        assert progress.sync_count == 0
        assert progress.last_error is None

    def test_custom_initialization(self):
        """Test custom initialization."""
        progress = BaseSyncProgress(
            last_sync_timestamp=12345.0,
            synced_nodes={"node1", "node2"},
            pending_syncs={"node3"},
            failed_nodes={"node4"},
            sync_count=10,
            last_error="Test error",
        )

        assert progress.last_sync_timestamp == 12345.0
        assert progress.synced_nodes == {"node1", "node2"}
        assert progress.pending_syncs == {"node3"}
        assert progress.failed_nodes == {"node4"}
        assert progress.sync_count == 10
        assert progress.last_error == "Test error"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        progress = BaseSyncProgress(
            last_sync_timestamp=12345.0,
            synced_nodes={"node1", "node2"},
            sync_count=5,
        )

        data = progress.to_dict()

        assert data["last_sync_timestamp"] == 12345.0
        assert set(data["synced_nodes"]) == {"node1", "node2"}
        assert data["sync_count"] == 5
        assert data["last_error"] is None

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "last_sync_timestamp": 99999.0,
            "synced_nodes": ["a", "b"],
            "pending_syncs": ["c"],
            "failed_nodes": ["d"],
            "sync_count": 100,
            "last_error": "Something went wrong",
        }

        progress = BaseSyncProgress.from_dict(data)

        assert progress.last_sync_timestamp == 99999.0
        assert progress.synced_nodes == {"a", "b"}
        assert progress.pending_syncs == {"c"}
        assert progress.failed_nodes == {"d"}
        assert progress.sync_count == 100
        assert progress.last_error == "Something went wrong"

    def test_from_dict_with_missing_keys(self):
        """Test deserialization with missing keys uses defaults."""
        data = {}

        progress = BaseSyncProgress.from_dict(data)

        assert progress.last_sync_timestamp == 0.0
        assert progress.synced_nodes == set()
        assert progress.sync_count == 0

    def test_round_trip_serialization(self):
        """Test that to_dict and from_dict are inverses."""
        original = BaseSyncProgress(
            last_sync_timestamp=12345.0,
            synced_nodes={"x", "y", "z"},
            pending_syncs={"a"},
            failed_nodes={"b"},
            sync_count=42,
            last_error="Error message",
        )

        data = original.to_dict()
        restored = BaseSyncProgress.from_dict(data)

        assert restored.last_sync_timestamp == original.last_sync_timestamp
        assert restored.synced_nodes == original.synced_nodes
        assert restored.pending_syncs == original.pending_syncs
        assert restored.failed_nodes == original.failed_nodes
        assert restored.sync_count == original.sync_count
        assert restored.last_error == original.last_error


# =============================================================================
# CircuitBreakerConfig Tests
# =============================================================================


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()

        assert config.failure_threshold == 3
        assert config.recovery_timeout == 60.0
        assert config.half_open_max_calls == 1

    def test_custom_values(self):
        """Test custom configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=120.0,
            half_open_max_calls=3,
        )

        assert config.failure_threshold == 5
        assert config.recovery_timeout == 120.0
        assert config.half_open_max_calls == 3


# =============================================================================
# SyncManagerBase Tests
# =============================================================================


class TestSyncManagerBaseInitialization:
    """Tests for SyncManagerBase initialization."""

    def test_default_initialization(self, sync_manager):
        """Test default initialization."""
        assert sync_manager.state_path is None
        assert sync_manager.sync_interval == 300.0
        assert not sync_manager._running

    def test_custom_sync_interval(self):
        """Test custom sync interval."""
        manager = MockSyncManager(sync_interval=60.0)
        assert manager.sync_interval == 60.0

    def test_custom_circuit_breaker_config(self):
        """Test custom circuit breaker configuration."""
        config = CircuitBreakerConfig(failure_threshold=10)
        manager = MockSyncManager(circuit_breaker_config=config)
        assert manager.circuit_breaker_config.failure_threshold == 10


class TestSyncManagerStatePersistence:
    """Tests for state persistence."""

    def test_save_and_load_state(self, sync_manager_with_state, temp_state_path):
        """Test saving and loading state."""
        manager = sync_manager_with_state

        # Modify state
        manager._state.synced_nodes.add("test_node")
        manager._state.sync_count = 5
        manager._state.last_sync_timestamp = 12345.0

        # Save
        manager._save_state()

        # Verify file exists
        assert temp_state_path.exists()

        # Load into new manager
        new_manager = MockSyncManager(state_path=temp_state_path)

        assert "test_node" in new_manager._state.synced_nodes
        assert new_manager._state.sync_count == 5
        assert new_manager._state.last_sync_timestamp == 12345.0

    def test_load_state_missing_file(self, temp_state_path):
        """Test loading state when file doesn't exist."""
        manager = MockSyncManager(state_path=temp_state_path)

        # Should not raise, uses defaults
        assert manager._state.sync_count == 0
        assert manager._state.synced_nodes == set()

    def test_load_state_corrupted_file(self, temp_state_path):
        """Test loading corrupted state file."""
        # Write invalid JSON
        temp_state_path.parent.mkdir(parents=True, exist_ok=True)
        temp_state_path.write_text("not valid json {{{")

        # Should not raise, uses defaults
        manager = MockSyncManager(state_path=temp_state_path)
        assert manager._state.sync_count == 0


class TestSyncWithNode:
    """Tests for sync_with_node method."""

    @pytest.mark.asyncio
    async def test_successful_sync(self, sync_manager):
        """Test successful node sync."""
        sync_manager.set_sync_result("node1", True)

        result = await sync_manager.sync_with_node("node1")

        assert result is True
        assert "node1" in sync_manager._state.synced_nodes

    @pytest.mark.asyncio
    async def test_failed_sync(self, sync_manager):
        """Test failed node sync."""
        sync_manager.set_sync_result("node1", False)

        result = await sync_manager.sync_with_node("node1")

        assert result is False
        assert "node1" in sync_manager._state.failed_nodes

    @pytest.mark.asyncio
    async def test_sync_with_exception(self, sync_manager):
        """Test sync that raises exception."""
        # Create manager with mock that raises
        manager = MockSyncManager()

        async def raise_error(node):
            raise RuntimeError("Test error")

        manager._do_sync = raise_error

        result = await manager.sync_with_node("node1")

        assert result is False
        assert "node1" in manager._state.failed_nodes
        assert "Test error" in manager._state.last_error

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_sync(self, sync_manager):
        """Test that open circuit breaker blocks sync."""
        # Simulate failures to open circuit
        for _ in range(5):
            sync_manager._record_sync_failure("node1")

        # Should be blocked
        result = await sync_manager.sync_with_node("node1")

        assert result is False


class TestSyncWithCluster:
    """Tests for sync_with_cluster method."""

    @pytest.mark.asyncio
    async def test_sync_with_cluster(self, sync_manager):
        """Test syncing with entire cluster."""
        results = await sync_manager.sync_with_cluster()

        assert len(results) == 3  # 3 default nodes
        assert all(results.values())
        assert sync_manager._state.sync_count == 1
        assert sync_manager._state.last_sync_timestamp > 0

    @pytest.mark.asyncio
    async def test_sync_with_cluster_partial_failure(self, sync_manager):
        """Test cluster sync with some failures."""
        sync_manager.set_sync_result("node2", False)

        results = await sync_manager.sync_with_cluster()

        assert results["node1"] is True
        assert results["node2"] is False
        assert results["node3"] is True


class TestStartStop:
    """Tests for start and stop methods."""

    @pytest.mark.asyncio
    async def test_start_and_stop(self, sync_manager):
        """Test starting and stopping the manager."""
        # Start in background
        task = asyncio.create_task(sync_manager.start())

        # Wait briefly
        await asyncio.sleep(0.1)

        assert sync_manager._running is True

        # Stop
        await sync_manager.stop()

        assert sync_manager._running is False

        # Cancel the task to clean up
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_start_when_already_running(self, sync_manager):
        """Test that starting when already running logs warning."""
        sync_manager._running = True

        # Should return immediately without error
        # Use short timeout to verify it returns quickly
        task = asyncio.create_task(sync_manager.start())
        done, pending = await asyncio.wait([task], timeout=0.1)

        # Should be done (returned early)
        assert len(done) == 1
        assert len(pending) == 0


class TestGetStatus:
    """Tests for get_status method."""

    def test_get_status_initial(self, sync_manager):
        """Test initial status."""
        status = sync_manager.get_status()

        assert status["running"] is False
        assert "state" in status
        assert "circuit_breaker" in status

    def test_get_status_after_sync(self, sync_manager):
        """Test status after syncs."""
        sync_manager._state.synced_nodes = {"a", "b"}
        sync_manager._state.failed_nodes = {"c"}
        sync_manager._state.sync_count = 10

        status = sync_manager.get_status()

        state = status["state"]
        assert set(state["synced_nodes"]) == {"a", "b"}
        assert set(state["failed_nodes"]) == {"c"}
        assert state["sync_count"] == 10


class TestHealthCheck:
    """Tests for health_check method."""

    def test_health_check_not_running(self, sync_manager):
        """Test health check when not running."""
        result = sync_manager.health_check()

        assert result.healthy is True
        assert result.status.value == "stopped"

    def test_health_check_healthy(self, sync_manager):
        """Test health check when healthy."""
        sync_manager._running = True
        sync_manager._state.synced_nodes = {"a", "b", "c"}
        sync_manager._state.failed_nodes = set()

        result = sync_manager.health_check()

        assert result.healthy is True
        assert result.status.value == "running"

    def test_health_check_high_failure_rate(self, sync_manager):
        """Test health check with high failure rate."""
        sync_manager._running = True
        sync_manager._state.synced_nodes = {"a"}
        sync_manager._state.failed_nodes = {"b", "c", "d", "e"}

        result = sync_manager.health_check()

        assert result.healthy is False
        assert result.status.value == "degraded"
        assert "failure rate" in result.message.lower()

    def test_health_check_stale_sync(self, sync_manager):
        """Test health check when sync is stale."""
        import time

        sync_manager._running = True
        sync_manager._state.last_sync_timestamp = time.time() - 1000
        sync_manager.sync_interval = 60.0

        result = sync_manager.health_check()

        assert result.status.value == "degraded"
        assert "ago" in result.message


# =============================================================================
# Integration Tests
# =============================================================================


class TestSyncManagerIntegration:
    """Integration tests for SyncManagerBase."""

    @pytest.mark.asyncio
    async def test_full_sync_cycle(self, sync_manager_with_state, temp_state_path):
        """Test a full sync cycle with state persistence."""
        manager = sync_manager_with_state

        # First sync
        results = await manager.sync_with_cluster()

        assert all(results.values())
        assert manager._state.sync_count == 1
        assert temp_state_path.exists()

        # Second sync
        results = await manager.sync_with_cluster()

        assert manager._state.sync_count == 2

    @pytest.mark.asyncio
    async def test_recovery_after_failures(self, sync_manager):
        """Test that nodes can recover after circuit breaker opens."""
        # Cause circuit breaker to open (3 failures is default threshold)
        for _ in range(3):
            sync_manager._record_sync_failure("node1")

        # Should be blocked initially
        assert not sync_manager._can_sync_with_node("node1")

        # Wait for recovery (mock immediate recovery)
        sync_manager._circuit_breaker.reset("node1")

        # Should be allowed again
        assert sync_manager._can_sync_with_node("node1")
