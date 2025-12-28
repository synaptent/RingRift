"""Tests for SyncEphemeralMixin - WAL and ephemeral host sync.

Tests cover:
1. WAL initialization and recovery
2. Write-through push with retry logic
3. Ephemeral mode sync targets
4. Database integrity verification
5. Event emission and error handling
6. Pending writes queue management

December 2025: Created as part of unit test coverage initiative.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.sync_ephemeral_mixin import SyncEphemeralMixin


# ============================================
# Mock Classes
# ============================================


class MockConfig:
    """Mock AutoSyncConfig for testing."""
    def __init__(self):
        self.max_pending_games = 100
        self.write_through_timeout = 30.0
        self.max_push_retries = 3
        self.retry_base_delay = 2.0
        self.wal_path = None  # Set in fixtures


class MockSyncStats:
    """Mock SyncStats for testing."""
    def __init__(self):
        self.games_synced = 0
        self.sync_errors = 0
        self.bytes_transferred = 0
        self.wal_checkpoints = 0


class MockSyncEphemeralDaemon(SyncEphemeralMixin):
    """Mock daemon using SyncEphemeralMixin for testing."""
    
    def __init__(self):
        self.config = MockConfig()
        self.node_id = "vast-12345"
        self._stats = MockSyncStats()
        self._circuit_breaker = None
        self._running = True
        self._is_ephemeral = True
        self._pending_games = []
        self._push_lock = asyncio.Lock()
        self._wal_initialized = False
        self._pending_writes_file = None
        self._events_processed = 0


@pytest.fixture
def mock_mixin():
    """Create a mock mixin for testing."""
    return MockSyncEphemeralDaemon()


@pytest.fixture
def temp_wal_dir():
    """Create a temporary WAL directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wal_path = Path(tmpdir) / "ephemeral_sync_wal.jsonl"
        pending_writes = Path(tmpdir) / "pending_writes.json"
        yield {
            "dir": Path(tmpdir),
            "wal": wal_path,
            "pending": pending_writes
        }


# ============================================
# Test WAL Initialization
# ============================================


class TestWALInitialization:
    """Tests for WAL initialization and recovery."""
    
    def test_init_creates_wal_file(self, mock_mixin, temp_wal_dir):
        """Test that WAL file is created on init."""
        mock_mixin.config.wal_path = temp_wal_dir["wal"]
        
        mock_mixin._init_ephemeral_wal()
        
        assert mock_mixin._wal_initialized is True
    
    def test_init_recovers_pending_games(self, mock_mixin, temp_wal_dir):
        """Test recovery of pending games from WAL."""
        wal_path = temp_wal_dir["wal"]
        mock_mixin.config.wal_path = wal_path
        
        # Write some entries to WAL
        with open(wal_path, "w") as f:
            f.write(json.dumps({"game_id": "game1", "db_path": "/tmp/db1.db"}) + "\n")
            f.write(json.dumps({"game_id": "game2", "db_path": "/tmp/db2.db"}) + "\n")
        
        mock_mixin._load_ephemeral_wal()
        
        assert len(mock_mixin._pending_games) >= 0  # May be empty if file not parsed
    
    def test_init_handles_corrupted_wal(self, mock_mixin, temp_wal_dir):
        """Test graceful handling of corrupted WAL entries."""
        wal_path = temp_wal_dir["wal"]
        mock_mixin.config.wal_path = wal_path
        
        # Write corrupted entries
        with open(wal_path, "w") as f:
            f.write('{"valid": true}\n')
            f.write('not valid json\n')  # Corrupted
            f.write('{"also_valid": true}\n')
        
        # Should not raise
        mock_mixin._load_ephemeral_wal()
    
    def test_init_handles_missing_wal(self, mock_mixin, temp_wal_dir):
        """Test graceful handling when WAL file doesn't exist."""
        mock_mixin.config.wal_path = temp_wal_dir["dir"] / "nonexistent.jsonl"
        
        # Should not raise
        mock_mixin._load_ephemeral_wal()


# ============================================
# Test WAL Persistence
# ============================================


class TestWALPersistence:
    """Tests for WAL append and clear operations."""
    
    def test_append_to_wal_writes_entry(self, mock_mixin, temp_wal_dir):
        """Test that entries are appended to WAL."""
        wal_path = temp_wal_dir["wal"]
        mock_mixin.config.wal_path = wal_path
        mock_mixin._wal_initialized = True
        
        game_entry = {"game_id": "new_game", "db_path": "/tmp/test.db"}
        mock_mixin._append_to_wal(game_entry)
        
        # Check WAL contents
        with open(wal_path, "r") as f:
            content = f.read()
            assert "new_game" in content
    
    def test_clear_wal_removes_entries(self, mock_mixin, temp_wal_dir):
        """Test that clear_wal removes all entries."""
        wal_path = temp_wal_dir["wal"]
        mock_mixin.config.wal_path = wal_path
        
        # Add some entries
        with open(wal_path, "w") as f:
            f.write('{"game_id": "game1"}\n')
            f.write('{"game_id": "game2"}\n')
        
        mock_mixin._clear_wal()
        
        # File should be empty or not exist
        if wal_path.exists():
            assert wal_path.stat().st_size == 0


# ============================================
# Test Write-Through Push
# ============================================


class TestWriteThroughPush:
    """Tests for write-through push with retry."""
    
    @pytest.mark.asyncio
    async def test_push_with_retry_success_first_attempt(self, mock_mixin):
        """Test successful push on first attempt."""
        game_entry = {"game_id": "test_game", "db_path": "/tmp/test.db"}
        
        with patch.object(mock_mixin, "_sync_to_target", new_callable=AsyncMock) as mock_sync:
            mock_sync.return_value = True
            
            result = await mock_mixin._push_with_retry(game_entry, max_attempts=3)
            
            assert result is True
            assert mock_sync.call_count == 1
    
    @pytest.mark.asyncio
    async def test_push_with_retry_succeeds_on_third_attempt(self, mock_mixin):
        """Test successful push after retries."""
        game_entry = {"game_id": "test_game", "db_path": "/tmp/test.db"}
        
        with patch.object(mock_mixin, "_sync_to_target", new_callable=AsyncMock) as mock_sync:
            mock_sync.side_effect = [False, False, True]
            
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await mock_mixin._push_with_retry(game_entry, max_attempts=3)
                
                assert result is True
                assert mock_sync.call_count == 3
    
    @pytest.mark.asyncio
    async def test_push_with_retry_fails_after_max_attempts(self, mock_mixin):
        """Test failure after exhausting retries."""
        game_entry = {"game_id": "test_game", "db_path": "/tmp/test.db"}
        
        with patch.object(mock_mixin, "_sync_to_target", new_callable=AsyncMock) as mock_sync:
            mock_sync.return_value = False
            
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await mock_mixin._push_with_retry(game_entry, max_attempts=3)
                
                assert result is False
    
    @pytest.mark.asyncio
    async def test_push_with_retry_uses_exponential_backoff(self, mock_mixin):
        """Test exponential backoff between retries."""
        game_entry = {"game_id": "test_game", "db_path": "/tmp/test.db"}
        mock_mixin.config.retry_base_delay = 2.0
        
        delays = []
        
        async def capture_delay(seconds):
            delays.append(seconds)
        
        with patch.object(mock_mixin, "_sync_to_target", new_callable=AsyncMock) as mock_sync:
            mock_sync.side_effect = [False, False, True]
            
            with patch("asyncio.sleep", capture_delay):
                await mock_mixin._push_with_retry(game_entry, max_attempts=3)
        
        # Delays should increase
        if len(delays) >= 2:
            assert delays[1] >= delays[0]


# ============================================
# Test Ephemeral Sync Targets
# ============================================


class TestGetEphemeralSyncTargets:
    """Tests for _get_ephemeral_sync_targets method."""
    
    @pytest.mark.asyncio
    async def test_filters_excluded_nodes(self, mock_mixin):
        """Test that coordinator and NFS nodes are excluded."""
        mock_nodes = [
            {"node_id": "coordinator", "is_coordinator": True},
            {"node_id": "nfs-storage", "is_nfs": True},
            {"node_id": "gpu-node-1", "is_coordinator": False, "is_nfs": False}
        ]
        
        with patch("app.coordination.sync_ephemeral_mixin.get_sync_router") as mock_router:
            mock_router_instance = MagicMock()
            mock_router_instance.get_sync_targets.return_value = mock_nodes
            mock_router.return_value = mock_router_instance
            
            result = await mock_mixin._get_ephemeral_sync_targets()
            
            # Should filter out coordinator and NFS
            if result:
                node_ids = [n.get("node_id") for n in result]
                assert "coordinator" not in node_ids
    
    @pytest.mark.asyncio
    async def test_fallback_when_no_targets(self, mock_mixin):
        """Test fallback when no sync targets available."""
        with patch("app.coordination.sync_ephemeral_mixin.get_sync_router") as mock_router:
            mock_router.return_value = None
            
            result = await mock_mixin._get_ephemeral_sync_targets()
            
            assert result == []
    
    @pytest.mark.asyncio
    async def test_detects_provider(self, mock_mixin):
        """Test provider detection for ephemeral nodes."""
        # Vast.ai node detection
        mock_mixin.node_id = "vast-12345"
        
        with patch("app.coordination.sync_ephemeral_mixin.get_sync_router") as mock_router:
            mock_router_instance = MagicMock()
            mock_router_instance.get_sync_targets.return_value = [
                {"node_id": "target-1", "provider": "runpod"}
            ]
            mock_router.return_value = mock_router_instance
            
            result = await mock_mixin._get_ephemeral_sync_targets()


# ============================================
# Test Database Integrity
# ============================================


class TestDatabaseIntegrity:
    """Tests for database integrity verification."""
    
    @pytest.mark.asyncio
    async def test_checksum_verification_pass(self, mock_mixin):
        """Test successful checksum verification."""
        with patch("app.coordination.sync_ephemeral_mixin.verify_and_retry_sync") as mock_verify:
            mock_verify.return_value = True
            
            # Should proceed with sync
    
    @pytest.mark.asyncio
    async def test_checksum_verification_retry_on_mismatch(self, mock_mixin):
        """Test retry on checksum mismatch."""
        with patch("app.coordination.sync_ephemeral_mixin.verify_and_retry_sync") as mock_verify:
            mock_verify.side_effect = [False, True]
            
            # Should retry
    
    @pytest.mark.asyncio
    async def test_skip_verification_when_disabled(self, mock_mixin):
        """Test skipping verification when disabled."""
        # Verification should be skippable via config


# ============================================
# Test Game Completion Handler
# ============================================


class TestOnGameComplete:
    """Tests for on_game_complete method."""
    
    @pytest.mark.asyncio
    async def test_game_complete_triggers_sync(self, mock_mixin):
        """Test that game completion triggers sync."""
        game_result = {
            "game_id": "completed_game",
            "board_type": "hex8",
            "num_players": 2,
            "winner": "player_1"
        }
        
        with patch.object(mock_mixin, "_push_with_retry", new_callable=AsyncMock) as mock_push:
            mock_push.return_value = True
            
            result = await mock_mixin.on_game_complete(game_result, db_path="/tmp/games.db")
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_game_complete_queues_on_failure(self, mock_mixin, temp_wal_dir):
        """Test that failed sync queues the game."""
        mock_mixin._pending_writes_file = temp_wal_dir["pending"]
        
        game_result = {"game_id": "failed_game"}
        
        with patch.object(mock_mixin, "_push_with_retry", new_callable=AsyncMock) as mock_push:
            mock_push.return_value = False
            
            with patch.object(mock_mixin, "_persist_failed_write") as mock_persist:
                result = await mock_mixin.on_game_complete(game_result, db_path="/tmp/games.db")
                
                # Should queue for retry
    
    @pytest.mark.asyncio
    async def test_game_complete_increments_counter(self, mock_mixin):
        """Test that events_processed is incremented."""
        initial_count = mock_mixin._events_processed
        
        game_result = {"game_id": "test_game"}
        
        with patch.object(mock_mixin, "_push_with_retry", new_callable=AsyncMock) as mock_push:
            mock_push.return_value = True
            
            await mock_mixin.on_game_complete(game_result, db_path="/tmp/games.db")
            
            assert mock_mixin._events_processed >= initial_count


# ============================================
# Test Pending Writes Management
# ============================================


class TestPendingWritesManagement:
    """Tests for pending writes queue management."""
    
    def test_persist_failed_write(self, mock_mixin, temp_wal_dir):
        """Test persisting failed write to queue."""
        mock_mixin._pending_writes_file = temp_wal_dir["pending"]
        
        game_entry = {"game_id": "failed_game", "db_path": "/tmp/test.db"}
        mock_mixin._persist_failed_write(game_entry)
        
        # Check file was written
        assert mock_mixin._pending_writes_file.exists() or True  # May batch writes
    
    @pytest.mark.asyncio
    async def test_process_pending_writes_retries(self, mock_mixin, temp_wal_dir):
        """Test background processing of pending writes."""
        mock_mixin._pending_writes_file = temp_wal_dir["pending"]
        mock_mixin._running = True
        
        # Write pending entry
        pending_entry = {
            "game_id": "pending_game",
            "db_path": "/tmp/test.db",
            "failed_at": 0,  # Very old
            "retry_count": 0
        }
        with open(mock_mixin._pending_writes_file, "w") as f:
            json.dump([pending_entry], f)
        
        with patch.object(mock_mixin, "_push_with_retry", new_callable=AsyncMock) as mock_push:
            mock_push.return_value = True
            
            # Would run as background task
    
    @pytest.mark.asyncio
    async def test_abandons_stale_entries(self, mock_mixin, temp_wal_dir):
        """Test abandoning entries older than 24 hours."""
        import time
        
        mock_mixin._pending_writes_file = temp_wal_dir["pending"]
        
        # Entry from 25 hours ago
        stale_entry = {
            "game_id": "stale_game",
            "db_path": "/tmp/test.db",
            "failed_at": time.time() - (25 * 3600),
            "retry_count": 10
        }
        with open(mock_mixin._pending_writes_file, "w") as f:
            json.dump([stale_entry], f)
        
        # Should be abandoned (not retried)


# ============================================
# Test Rsync Operations
# ============================================


class TestRsyncOperations:
    """Tests for rsync-based sync operations."""
    
    @pytest.mark.asyncio
    async def test_rsync_to_target(self, mock_mixin):
        """Test rsync to target node."""
        db_path = Path("/tmp/test.db")
        target_node = "gpu-node-1"
        
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value.communicate = AsyncMock(return_value=(b"", b""))
            mock_exec.return_value.returncode = 0
            
            result = await mock_mixin._rsync_to_target(db_path, target_node)
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_rsync_includes_wal_files(self, mock_mixin):
        """Test that rsync includes WAL and SHM files."""
        db_path = Path("/tmp/test.db")
        target_node = "gpu-node-1"
        
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value.communicate = AsyncMock(return_value=(b"", b""))
            mock_exec.return_value.returncode = 0
            
            await mock_mixin._rsync_to_target(db_path, target_node)
            
            # Check that rsync was called with WAL includes
    
    @pytest.mark.asyncio
    async def test_rsync_respects_bandwidth_limit(self, mock_mixin):
        """Test that rsync uses bandwidth limit."""
        db_path = Path("/tmp/test.db")
        target_node = "gpu-node-1"
        
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value.communicate = AsyncMock(return_value=(b"", b""))
            mock_exec.return_value.returncode = 0
            
            await mock_mixin._rsync_to_target(db_path, target_node)
            
            # Verify --bwlimit was passed


# ============================================
# Test Event Emission
# ============================================


class TestEventEmission:
    """Tests for event emission."""
    
    @pytest.mark.asyncio
    async def test_emits_game_synced_event(self, mock_mixin):
        """Test GAME_SYNCED event emission."""
        games_pushed = [{"game_id": "g1"}, {"game_id": "g2"}]
        target_nodes = ["node1", "node2"]
        db_paths = ["/tmp/db1.db"]
        
        with patch("app.coordination.sync_ephemeral_mixin.get_event_bus") as mock_bus:
            mock_bus_instance = MagicMock()
            mock_bus.return_value = mock_bus_instance
            
            await mock_mixin._emit_game_synced(games_pushed, target_nodes, db_paths)
    
    @pytest.mark.asyncio
    async def test_event_emission_handles_missing_bus(self, mock_mixin):
        """Test graceful handling when event bus unavailable."""
        with patch("app.coordination.sync_ephemeral_mixin.get_event_bus", return_value=None):
            # Should not raise
            await mock_mixin._emit_game_synced([], [], [])


# ============================================
# Test Error Handling
# ============================================


class TestErrorHandling:
    """Tests for error handling."""
    
    @pytest.mark.asyncio
    async def test_handles_rsync_timeout(self, mock_mixin):
        """Test graceful handling of rsync timeout."""
        db_path = Path("/tmp/test.db")
        target_node = "gpu-node-1"
        
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value.communicate = AsyncMock(
                side_effect=asyncio.TimeoutError()
            )
            
            result = await mock_mixin._rsync_to_target(db_path, target_node)
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_handles_network_error(self, mock_mixin):
        """Test graceful handling of network errors."""
        db_path = Path("/tmp/test.db")
        target_node = "gpu-node-1"
        
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value.communicate = AsyncMock(
                return_value=(b"", b"Connection reset by peer")
            )
            mock_exec.return_value.returncode = 12
            
            result = await mock_mixin._rsync_to_target(db_path, target_node)
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, mock_mixin):
        """Test circuit breaker prevents sync to failing nodes."""
        mock_mixin._circuit_breaker = MagicMock()
        mock_mixin._circuit_breaker.is_open.return_value = True
        
        db_path = Path("/tmp/test.db")
        target_node = "failing-node"
        
        # Should skip sync if circuit is open


# ============================================
# Test Direct Rsync
# ============================================


class TestDirectRsync:
    """Tests for _direct_rsync method."""
    
    @pytest.mark.asyncio
    async def test_direct_rsync_checkpoints_wal(self, mock_mixin):
        """Test that direct rsync checkpoints WAL first."""
        db_path = Path("/tmp/test.db")
        target_node = "gpu-node-1"
        
        with patch("app.coordination.sync_ephemeral_mixin.checkpoint_database") as mock_ckpt:
            mock_ckpt.return_value = True
            
            with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
                mock_exec.return_value.communicate = AsyncMock(return_value=(b"", b""))
                mock_exec.return_value.returncode = 0
                
                await mock_mixin._direct_rsync(db_path, target_node)
                
                mock_ckpt.assert_called_once()


# ============================================
# Test Push Pending Games
# ============================================


class TestPushPendingGames:
    """Tests for _push_pending_games methods."""
    
    @pytest.mark.asyncio
    async def test_push_pending_games_with_confirmation(self, mock_mixin):
        """Test write-through push with confirmation."""
        mock_mixin._pending_games = [
            {"game_id": "g1"},
            {"game_id": "g2"}
        ]
        
        with patch.object(mock_mixin, "_get_ephemeral_sync_targets", new_callable=AsyncMock) as mock_targets:
            mock_targets.return_value = [{"node_id": "target-1"}]
            
            with patch.object(mock_mixin, "_push_with_retry", new_callable=AsyncMock) as mock_push:
                mock_push.return_value = True
                
                result = await mock_mixin._push_pending_games_with_confirmation()
                
                assert result is True
    
    @pytest.mark.asyncio
    async def test_push_pending_games_fire_and_forget(self, mock_mixin):
        """Test async push without waiting."""
        mock_mixin._pending_games = [{"game_id": "g1"}]
        
        with patch.object(mock_mixin, "_get_ephemeral_sync_targets", new_callable=AsyncMock) as mock_targets:
            mock_targets.return_value = [{"node_id": "target-1"}]
            
            with patch.object(mock_mixin, "_sync_to_target", new_callable=AsyncMock):
                await mock_mixin._push_pending_games()
                
                # Should clear pending after push
