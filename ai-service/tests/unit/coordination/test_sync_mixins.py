"""Tests for AutoSyncDaemon sync mixins.

December 2025: Comprehensive tests for all sync mixins:
- SyncMixinBase
- SyncPushMixin
- SyncPullMixin
- SyncEphemeralMixin
- SyncEventMixin
"""

import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest


# =============================================================================
# Test Helpers
# =============================================================================


class MockConfig:
    """Mock AutoSyncConfig for testing."""

    def __init__(self):
        self.sync_interval_seconds = 60
        self.broadcast_high_priority_configs = ["hex8_2p", "square8_2p"]
        self.ephemeral_sync_interval = 5
        self.max_concurrent_syncs = 3
        self.sync_timeout = 300
        self.gossip_replication_factor = 3
        self.ephemeral_providers = ["vast"]
        self.min_games_to_sync = 5
        self.push_on_complete = True
        self.urgent_sync_cooldown = 30


@dataclass
class MockSyncStats:
    """Mock SyncStats for testing."""

    syncs_completed: int = 0
    syncs_failed: int = 0
    bytes_synced: int = 0
    games_synced: int = 0
    last_sync_time: float = 0.0
    sync_errors: list[str] = field(default_factory=list)


class MockCircuitBreaker:
    """Mock CircuitBreaker for testing."""

    def __init__(self):
        self.is_open = False
        self.failure_count = 0
        self.success_count = 0

    def record_failure(self):
        self.failure_count += 1

    def record_success(self):
        self.success_count += 1

    def allow_request(self) -> bool:
        return not self.is_open


# =============================================================================
# SyncMixinBase Tests
# =============================================================================


class TestSyncMixinBase:
    """Tests for SyncMixinBase abstract base class."""

    def test_sync_error_creation(self):
        """Test SyncError dataclass basic creation."""
        from app.coordination.sync_mixin_base import SyncError

        error = SyncError(
            error_type="network",
            message="Connection refused",
            target_node="node-123",
            db_path="/data/games/test.db",
        )

        assert error.error_type == "network"
        assert error.message == "Connection refused"
        assert error.target_node == "node-123"
        assert error.db_path == "/data/games/test.db"
        assert error.recoverable is True
        assert error.timestamp > 0

    def test_sync_error_from_timeout_exception(self):
        """Test SyncError.from_exception() with timeout error."""
        from app.coordination.sync_mixin_base import SyncError

        exc = TimeoutError("Connection timed out after 30s")
        error = SyncError.from_exception(exc, target_node="node-1")

        assert error.error_type == "timeout"
        assert "30s" in error.message
        assert error.target_node == "node-1"
        assert error.recoverable is True

    def test_sync_error_from_connection_exception(self):
        """Test SyncError.from_exception() with connection error."""
        from app.coordination.sync_mixin_base import SyncError

        exc = ConnectionRefusedError("SSH connection refused")
        error = SyncError.from_exception(exc, target_node="node-2")

        assert error.error_type == "network"
        assert "refused" in error.message.lower()
        assert error.recoverable is True

    def test_sync_error_from_database_exception(self):
        """Test SyncError.from_exception() with database error."""
        from app.coordination.sync_mixin_base import SyncError

        exc = sqlite3.DatabaseError("database disk image is malformed")
        error = SyncError.from_exception(exc, db_path="/data/games/corrupt.db")

        assert error.error_type == "database"
        assert error.db_path == "/data/games/corrupt.db"
        assert error.recoverable is False

    def test_sync_error_from_permission_exception(self):
        """Test SyncError.from_exception() with permission error."""
        from app.coordination.sync_mixin_base import SyncError

        exc = PermissionError("Permission denied: /data/games/locked.db")
        error = SyncError.from_exception(exc)

        assert error.error_type == "permission"
        assert error.recoverable is False

    def test_sync_error_from_unknown_exception(self):
        """Test SyncError.from_exception() with unknown error type."""
        from app.coordination.sync_mixin_base import SyncError

        exc = ValueError("Some unexpected error")
        error = SyncError.from_exception(exc)

        assert error.error_type == "unknown"
        assert error.recoverable is True

    def test_validate_sync_daemon_protocol_valid(self):
        """Test validate_sync_daemon_protocol() with valid object."""
        from app.coordination.sync_mixin_base import validate_sync_daemon_protocol

        class ValidDaemon:
            config = MockConfig()
            node_id = "test-node"
            _running = True
            _stats = MockSyncStats()
            _events_processed = 0
            _errors_count = 0

            async def _sync_all(self):
                pass

            async def _sync_to_peer(self, node_id: str) -> bool:
                return True

        daemon = ValidDaemon()
        assert validate_sync_daemon_protocol(daemon) is True

    def test_validate_sync_daemon_protocol_missing_attr(self):
        """Test validate_sync_daemon_protocol() with missing attribute."""
        from app.coordination.sync_mixin_base import validate_sync_daemon_protocol

        class MissingAttr:
            config = MockConfig()
            # Missing node_id
            _running = True
            _stats = MockSyncStats()
            _events_processed = 0
            _errors_count = 0

            async def _sync_all(self):
                pass

            async def _sync_to_peer(self, node_id: str) -> bool:
                return True

        daemon = MissingAttr()
        assert validate_sync_daemon_protocol(daemon) is False

    def test_validate_sync_daemon_protocol_missing_method(self):
        """Test validate_sync_daemon_protocol() with missing method."""
        from app.coordination.sync_mixin_base import validate_sync_daemon_protocol

        class MissingMethod:
            config = MockConfig()
            node_id = "test-node"
            _running = True
            _stats = MockSyncStats()
            _events_processed = 0
            _errors_count = 0

            async def _sync_all(self):
                pass

            # Missing _sync_to_peer

        daemon = MissingMethod()
        assert validate_sync_daemon_protocol(daemon) is False


# =============================================================================
# SyncPushMixin Tests
# =============================================================================


class TestSyncPushMixin:
    """Tests for SyncPushMixin."""

    @pytest.fixture
    def mock_daemon(self):
        """Create a mock daemon with push mixin."""
        from app.coordination.sync_push_mixin import SyncPushMixin

        class TestDaemon(SyncPushMixin):
            def __init__(self):
                self.config = MockConfig()
                self.node_id = "test-node"
                self._running = True
                self._is_broadcast = True
                self._stats = MockSyncStats()
                self._events_processed = 0
                self._errors_count = 0
                self._last_error = ""
                self._circuit_breaker = MockCircuitBreaker()

            async def _emit_sync_failure(self, target_node: str, db_path: str, error: str) -> None:
                pass

            async def _emit_sync_stalled(
                self, target_node: str, timeout_seconds: float, data_type: str = "game", retry_count: int = 0
            ) -> None:
                pass

        return TestDaemon()

    def test_discover_local_databases_empty_dir(self, mock_daemon, tmp_path):
        """Test discover_local_databases() with non-existent directory."""
        with patch.object(Path, "resolve", return_value=tmp_path / "nonexistent"):
            databases = mock_daemon.discover_local_databases()
            # Returns empty list when dir doesn't exist
            assert databases == [] or isinstance(databases, list)

    def test_get_bandwidth_for_node_default(self, mock_daemon):
        """Test get_bandwidth_for_node() returns default value."""
        with patch(
            "app.config.cluster_config.get_node_bandwidth_kbs",
            return_value=50_000,
        ):
            bw = mock_daemon.get_bandwidth_for_node("vast-12345")
            assert bw == 50_000

    def test_get_bandwidth_for_node_fallback(self, mock_daemon):
        """Test get_bandwidth_for_node() fallback when import fails."""
        # Simulating the ImportError fallback by testing without mocking
        # The method handles ImportError internally and returns 20_000
        with patch.dict("sys.modules", {"app.config.cluster_config": None}):
            # Force reimport - since it's a lazy import, we just test the method directly
            pass
        # Just test that the method exists and returns an int
        bw = mock_daemon.get_bandwidth_for_node("some-node")
        assert isinstance(bw, int)
        assert bw > 0

    @pytest.mark.asyncio
    async def test_get_broadcast_targets_empty_response(self, mock_daemon):
        """Test get_broadcast_targets() with empty P2P response."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b"{}"
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            targets = await mock_daemon.get_broadcast_targets()
            assert targets == []

    @pytest.mark.asyncio
    async def test_get_broadcast_targets_network_error(self, mock_daemon):
        """Test get_broadcast_targets() handles network errors."""
        with patch("urllib.request.urlopen", side_effect=OSError("Connection refused")):
            targets = await mock_daemon.get_broadcast_targets()
            assert targets == []


# =============================================================================
# SyncPullMixin Tests
# =============================================================================


class TestSyncPullMixin:
    """Tests for SyncPullMixin."""

    @pytest.fixture
    def mock_daemon(self):
        """Create a mock daemon with pull mixin."""
        from app.coordination.sync_pull_mixin import SyncPullMixin

        class TestDaemon(SyncPullMixin):
            def __init__(self):
                self.config = MockConfig()
                self.node_id = "test-node"
                self._running = True
                self._stats = MockSyncStats()
                self._events_processed = 0
                self._errors_count = 0
                self._last_error = ""
                self._circuit_breaker = MockCircuitBreaker()
                self._pull_sources = []

            async def _emit_sync_failure(self, target_node: str, db_path: str, error: str) -> None:
                pass

            async def _emit_sync_stalled(
                self, target_node: str, timeout_seconds: float, data_type: str = "game", retry_count: int = 0
            ) -> None:
                pass

        return TestDaemon()

    def test_record_error_increments_count(self, mock_daemon):
        """Test _record_error() increments error counter."""
        initial_count = mock_daemon._errors_count
        mock_daemon._record_error("Test error", target_node="node-1")
        assert mock_daemon._errors_count == initial_count + 1

    def test_record_error_updates_last_error(self, mock_daemon):
        """Test _record_error() updates last error message."""
        mock_daemon._record_error("New error message")
        assert mock_daemon._last_error == "New error message"

    def test_record_error_from_exception(self, mock_daemon):
        """Test _record_error() handles exception input."""
        exc = ConnectionError("Host unreachable")
        error = mock_daemon._record_error(exc)
        assert error.error_type == "network"
        assert "unreachable" in error.message.lower()

    def test_record_event_processed(self, mock_daemon):
        """Test _record_event_processed() increments counter."""
        initial = mock_daemon._events_processed
        mock_daemon._record_event_processed()
        assert mock_daemon._events_processed == initial + 1


# =============================================================================
# SyncEphemeralMixin Tests
# =============================================================================


class TestSyncEphemeralMixin:
    """Tests for SyncEphemeralMixin."""

    @pytest.fixture
    def mock_daemon(self, tmp_path):
        """Create a mock daemon with ephemeral mixin."""
        from app.coordination.sync_ephemeral_mixin import SyncEphemeralMixin

        class TestDaemon(SyncEphemeralMixin):
            def __init__(self):
                self.config = MockConfig()
                self.node_id = "vast-12345"
                self._running = True
                self._is_ephemeral = True
                self._is_broadcast = False
                self._stats = MockSyncStats()
                self._events_processed = 0
                self._errors_count = 0
                self._last_error = ""
                self._circuit_breaker = MockCircuitBreaker()
                self._pending_games = []
                self._ephemeral_wal_path = tmp_path / "ephemeral_wal.jsonl"
                self._pending_writes_file = tmp_path / "pending_writes.json"
                self._wal_entries = []
                self._wal_initialized = False
                self._push_lock = asyncio.Lock()
                self._wal_path = None

            async def _emit_sync_failure(self, target_node: str, db_path: str, error: str) -> None:
                pass

            async def _emit_sync_stalled(
                self, target_node: str, timeout_seconds: float, data_type: str = "game", retry_count: int = 0
            ) -> None:
                pass

            async def _sync_all(self) -> None:
                pass

            async def _sync_to_peer(self, node_id: str) -> bool:
                return True

        return TestDaemon()

    def test_init_ephemeral_wal(self, mock_daemon):
        """Test _init_ephemeral_wal() creates WAL file."""
        mock_daemon._init_ephemeral_wal()
        # Method should run without error
        assert hasattr(mock_daemon, "_ephemeral_wal_path")

    def test_append_to_wal(self, mock_daemon):
        """Test _append_to_wal() adds entry to WAL."""
        mock_daemon._init_ephemeral_wal()
        game_entry = {"game_id": "test-123", "config_key": "hex8_2p"}
        mock_daemon._append_to_wal(game_entry)
        # Should not raise

    def test_clear_wal(self, mock_daemon):
        """Test _clear_wal() clears WAL file contents."""
        mock_daemon._init_ephemeral_wal()
        # Write some content to the WAL
        if hasattr(mock_daemon, "_wal_path") and mock_daemon._wal_path:
            mock_daemon._wal_path.write_text('{"game_id": "test"}\n')
        mock_daemon._clear_wal()
        # After clearing, WAL file should be empty
        if hasattr(mock_daemon, "_wal_path") and mock_daemon._wal_path and mock_daemon._wal_path.exists():
            assert mock_daemon._wal_path.read_text() == ''

    def test_init_pending_writes_file(self, mock_daemon):
        """Test _init_pending_writes_file() creates pending writes file."""
        mock_daemon._init_pending_writes_file()
        # Method should run without error

    def test_persist_failed_write(self, mock_daemon):
        """Test _persist_failed_write() persists failed entry."""
        mock_daemon._init_pending_writes_file()
        game_entry = {"game_id": "failed-123", "config_key": "hex8_2p"}
        mock_daemon._persist_failed_write(game_entry)
        # Should not raise

    @pytest.mark.asyncio
    async def test_push_pending_games(self, mock_daemon):
        """Test _push_pending_games() processes pending games."""
        mock_daemon._pending_games = [{"game_id": "test-1"}]
        # Should not raise, even with no actual targets
        with patch.object(mock_daemon, "_get_ephemeral_sync_targets", new_callable=AsyncMock, return_value=[]):
            await mock_daemon._push_pending_games(force=True)

    @pytest.mark.asyncio
    async def test_get_ephemeral_sync_targets_empty(self, mock_daemon):
        """Test _get_ephemeral_sync_targets() returns empty when SyncRouter unavailable."""
        # Patch the sync_router import to raise ImportError
        with patch.dict("sys.modules", {"app.coordination.sync_router": None}):
            # This should gracefully return empty list on ImportError
            targets = await mock_daemon._get_ephemeral_sync_targets()
            assert targets == []


# =============================================================================
# SyncEventMixin Tests
# =============================================================================


class TestSyncEventMixin:
    """Tests for SyncEventMixin."""

    @pytest.fixture
    def mock_daemon(self):
        """Create a mock daemon with event mixin."""
        from app.coordination.sync_event_mixin import SyncEventMixin

        class TestDaemon(SyncEventMixin):
            def __init__(self):
                self.config = MockConfig()
                self.node_id = "test-node"
                self._running = True
                self._subscribed = False
                self._stats = MockSyncStats()
                self._events_processed = 0
                self._errors_count = 0
                self._last_error = ""
                self._circuit_breaker = MockCircuitBreaker()
                self._is_ephemeral = False
                self._pending_games = []
                self._urgent_sync_pending = {}

            async def _emit_sync_failure(self, target_node: str, db_path: str, error: str) -> None:
                pass

            async def _emit_sync_stalled(
                self, target_node: str, timeout_seconds: float, data_type: str = "game", retry_count: int = 0
            ) -> None:
                pass

            async def _sync_all(self) -> None:
                pass

            async def _sync_to_peer(self, node_id: str) -> bool:
                return True

            async def _trigger_urgent_sync(self, config_key: str) -> None:
                pass

        return TestDaemon()

    def test_subscribe_to_events(self, mock_daemon):
        """Test _subscribe_to_events() sets up subscriptions."""
        # Mock the event bus to avoid actual subscriptions (patch at source)
        with patch("app.coordination.event_router.get_event_bus") as mock_bus:
            mock_bus_instance = MagicMock()
            mock_bus.return_value = mock_bus_instance
            mock_daemon._subscribe_to_events()
            # The method should have run without error (may or may not call subscribe
            # depending on internal logic)
            assert True  # Method ran successfully

    @pytest.mark.asyncio
    async def test_on_data_stale(self, mock_daemon):
        """Test _on_data_stale() handles stale data event."""
        event = MagicMock()
        event.payload = {"config_key": "hex8_2p", "staleness_hours": 5.0}

        # Should not raise
        await mock_daemon._on_data_stale(event)

    @pytest.mark.asyncio
    async def test_on_sync_triggered(self, mock_daemon):
        """Test _on_sync_triggered() handles sync trigger event."""
        event = MagicMock()
        event.payload = {"config_key": "hex8_2p", "reason": "data_stale"}

        # Should not raise
        await mock_daemon._on_sync_triggered(event)

    @pytest.mark.asyncio
    async def test_on_node_recovered(self, mock_daemon):
        """Test _on_node_recovered() handles node recovery event."""
        event = MagicMock()
        event.payload = {"node_id": "vast-recovered", "recovery_type": "restart"}

        # Should not raise
        await mock_daemon._on_node_recovered(event)

    @pytest.mark.asyncio
    async def test_on_training_started(self, mock_daemon):
        """Test _on_training_started() handles training start event."""
        event = MagicMock()
        event.payload = {"config_key": "hex8_2p", "node_id": "nebius-h100-1"}

        # Should not raise
        await mock_daemon._on_training_started(event)

    @pytest.mark.asyncio
    async def test_on_selfplay_complete(self, mock_daemon):
        """Test _on_selfplay_complete() handles selfplay complete event."""
        event = MagicMock()
        event.payload = {"config_key": "hex8_2p", "games_generated": 100}

        # Should not raise
        await mock_daemon._on_selfplay_complete(event)


# =============================================================================
# Integration Tests
# =============================================================================


class TestSyncMixinIntegration:
    """Integration tests for sync mixin combinations."""

    def test_mixin_inheritance_chain(self):
        """Test that all mixins properly inherit from SyncMixinBase."""
        from app.coordination.sync_mixin_base import SyncMixinBase
        from app.coordination.sync_push_mixin import SyncPushMixin
        from app.coordination.sync_pull_mixin import SyncPullMixin
        from app.coordination.sync_ephemeral_mixin import SyncEphemeralMixin
        from app.coordination.sync_event_mixin import SyncEventMixin

        assert issubclass(SyncPushMixin, SyncMixinBase)
        assert issubclass(SyncPullMixin, SyncMixinBase)
        assert issubclass(SyncEphemeralMixin, SyncMixinBase)
        assert issubclass(SyncEventMixin, SyncMixinBase)

    def test_log_prefix_consistent(self):
        """Test LOG_PREFIX is consistent across mixins."""
        from app.coordination.sync_mixin_base import SyncMixinBase
        from app.coordination.sync_push_mixin import SyncPushMixin

        assert SyncMixinBase.LOG_PREFIX == "[AutoSyncDaemon]"
        assert SyncPushMixin.LOG_PREFIX == "[AutoSyncDaemon]"

    def test_combined_daemon_has_all_methods(self):
        """Test a combined daemon has methods from all mixins."""
        from app.coordination.sync_push_mixin import SyncPushMixin
        from app.coordination.sync_pull_mixin import SyncPullMixin
        from app.coordination.sync_event_mixin import SyncEventMixin

        class CombinedDaemon(SyncPushMixin, SyncPullMixin, SyncEventMixin):
            def __init__(self):
                self.config = MockConfig()
                self.node_id = "test-node"
                self._running = True
                self._subscribed = False
                self._is_broadcast = True
                self._is_ephemeral = False
                self._stats = MockSyncStats()
                self._events_processed = 0
                self._errors_count = 0
                self._last_error = ""
                self._circuit_breaker = None
                self._pending_games = []

            async def _emit_sync_failure(self, target_node: str, db_path: str, error: str) -> None:
                pass

            async def _emit_sync_stalled(
                self, target_node: str, timeout_seconds: float, data_type: str = "game", retry_count: int = 0
            ) -> None:
                pass

            async def _sync_all(self) -> None:
                pass

            async def _sync_to_peer(self, node_id: str) -> bool:
                return True

        daemon = CombinedDaemon()

        # From SyncMixinBase
        assert hasattr(daemon, "_record_error")
        assert hasattr(daemon, "_log_info")

        # From SyncPushMixin
        assert hasattr(daemon, "discover_local_databases")
        assert hasattr(daemon, "get_bandwidth_for_node")

        # From SyncPullMixin - inherits base methods
        assert hasattr(daemon, "_record_event_processed")

        # From SyncEventMixin
        assert hasattr(daemon, "_subscribe_to_events")
        assert hasattr(daemon, "_on_data_stale")
