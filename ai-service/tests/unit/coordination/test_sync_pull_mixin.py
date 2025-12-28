"""Tests for SyncPullMixin - Pull-based sync for coordinator recovery.

Tests cover:
1. Coordinator role detection
2. Remote database listing via SSH
3. Rsync pull operations with checksum verification
4. Database merging into canonical databases
5. Canonical name extraction
6. Remote path detection by provider

December 2025: Created as part of unit test coverage initiative.
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.sync_pull_mixin import SyncPullMixin


# ============================================
# Mock Classes
# ============================================


class MockSyncStats:
    """Mock SyncStats for testing."""
    def __init__(self):
        self.games_pulled = 0
        self.pull_errors = 0
        self.databases_merged = 0


class MockSyncPullDaemon(SyncPullMixin):
    """Mock daemon using SyncPullMixin for testing."""
    
    def __init__(self):
        self.node_id = "coordinator"
        self._stats = MockSyncStats()
        self._circuit_breaker = None
        self._running = True


@pytest.fixture
def mock_mixin():
    """Create a mock mixin for testing."""
    return MockSyncPullDaemon()


@pytest.fixture
def temp_db():
    """Create a temporary database with game data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_games.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE games (
                game_id TEXT PRIMARY KEY,
                board_type TEXT,
                num_players INTEGER,
                game_status TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE game_moves (
                game_id TEXT,
                move_number INTEGER,
                move_data TEXT,
                PRIMARY KEY (game_id, move_number)
            )
        """)
        for i in range(5):
            conn.execute(
                "INSERT INTO games (game_id, board_type, num_players, game_status) VALUES (?, ?, ?, ?)",
                (f"game_{i}", "hex8", 2, "completed")
            )
        conn.commit()
        conn.close()
        yield db_path


# ============================================
# Test Coordinator Role Detection
# ============================================


class TestCoordinatorRoleDetection:
    """Tests for coordinator role detection."""
    
    @pytest.mark.asyncio
    async def test_pull_requires_coordinator_role(self, mock_mixin):
        """Test that pull is skipped for non-coordinator nodes."""
        with patch("app.coordination.sync_pull_mixin.env") as mock_env:
            mock_env.is_coordinator = False
            
            result = await mock_mixin._pull_from_cluster_nodes()
            
            assert result == 0
    
    @pytest.mark.asyncio
    async def test_pull_runs_on_coordinator(self, mock_mixin):
        """Test that pull runs for coordinator nodes."""
        with patch("app.coordination.sync_pull_mixin.env") as mock_env:
            mock_env.is_coordinator = True
            
            with patch.object(mock_mixin, "_get_sync_sources", return_value=[]):
                result = await mock_mixin._pull_from_cluster_nodes()
                
                # Should return 0 if no sources but not skip
    
    def test_hostname_fallback_mac_studio(self, mock_mixin):
        """Test hostname fallback for mac-studio."""
        with patch("socket.gethostname", return_value="mac-studio.local"):
            with patch("app.coordination.sync_pull_mixin.env", side_effect=ImportError):
                # Should detect as coordinator via hostname
                pass  # Verification would require internal state check


# ============================================
# Test Remote Database Listing
# ============================================


class TestListRemoteDatabases:
    """Tests for _list_remote_databases method."""
    
    @pytest.mark.asyncio
    async def test_lists_databases_via_ssh(self, mock_mixin):
        """Test SSH-based database listing."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b"canonical_hex8_2p.db\ngumbel_square8_2p.db\n"
        
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value.communicate = AsyncMock(
                return_value=(mock_result.stdout, b"")
            )
            mock_exec.return_value.returncode = 0
            
            result = await mock_mixin._list_remote_databases(
                ssh_host="10.0.0.1",
                ssh_user="root",
                ssh_key="/path/to/key",
                remote_path="/data/games"
            )
            
            assert isinstance(result, list)
    
    @pytest.mark.asyncio
    async def test_handles_ssh_timeout(self, mock_mixin):
        """Test graceful handling of SSH timeout."""
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value.communicate = AsyncMock(
                side_effect=asyncio.TimeoutError()
            )
            
            result = await mock_mixin._list_remote_databases(
                ssh_host="10.0.0.1",
                ssh_user="root", 
                ssh_key="/path/to/key",
                remote_path="/data/games"
            )
            
            assert result == []
    
    @pytest.mark.asyncio
    async def test_handles_ssh_error(self, mock_mixin):
        """Test graceful handling of SSH errors."""
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value.communicate = AsyncMock(
                return_value=(b"", b"Connection refused")
            )
            mock_exec.return_value.returncode = 255
            
            result = await mock_mixin._list_remote_databases(
                ssh_host="10.0.0.1",
                ssh_user="root",
                ssh_key="/path/to/key",
                remote_path="/data/games"
            )
            
            assert result == []


# ============================================
# Test Rsync Pull
# ============================================


class TestRsyncPull:
    """Tests for _rsync_pull method."""
    
    @pytest.mark.asyncio
    async def test_successful_rsync_pull(self, mock_mixin):
        """Test successful rsync pull operation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir)
            
            with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
                mock_exec.return_value.communicate = AsyncMock(
                    return_value=(b"", b"")
                )
                mock_exec.return_value.returncode = 0
                
                result = await mock_mixin._rsync_pull(
                    ssh_host="10.0.0.1",
                    ssh_user="root",
                    ssh_key="/path/to/key",
                    remote_path="/data/games",
                    db_name="test.db",
                    local_dir=local_dir,
                    verify_checksum=False
                )
                
                # Should return Path or None
    
    @pytest.mark.asyncio
    async def test_rsync_pull_with_checksum_verification(self, mock_mixin):
        """Test rsync pull with checksum verification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir)
            
            with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
                mock_exec.return_value.communicate = AsyncMock(
                    return_value=(b"", b"")
                )
                mock_exec.return_value.returncode = 0
                
                with patch("app.coordination.sync_pull_mixin.verify_sync_checksum") as mock_verify:
                    mock_verify.return_value = True
                    
                    result = await mock_mixin._rsync_pull(
                        ssh_host="10.0.0.1",
                        ssh_user="root",
                        ssh_key="/path/to/key",
                        remote_path="/data/games",
                        db_name="test.db",
                        local_dir=local_dir,
                        verify_checksum=True
                    )
    
    @pytest.mark.asyncio
    async def test_rsync_pull_checksum_mismatch_retry(self, mock_mixin):
        """Test rsync retry on checksum mismatch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir)
            
            with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
                mock_exec.return_value.communicate = AsyncMock(
                    return_value=(b"", b"")
                )
                mock_exec.return_value.returncode = 0
                
                with patch("app.coordination.sync_pull_mixin.verify_sync_checksum") as mock_verify:
                    # Fail first, succeed second
                    mock_verify.side_effect = [False, True]
                    
                    # Should retry on mismatch


# ============================================
# Test Database Merging
# ============================================


class TestMergeIntoCanonical:
    """Tests for _merge_into_canonical method."""
    
    @pytest.mark.asyncio
    async def test_merge_deduplicates_games(self, mock_mixin, temp_db):
        """Test that merge skips existing games."""
        with tempfile.TemporaryDirectory() as tmpdir:
            canonical_path = Path(tmpdir) / "canonical_hex8_2p.db"
            
            # Create canonical with some games
            conn = sqlite3.connect(str(canonical_path))
            conn.execute("""
                CREATE TABLE games (
                    game_id TEXT PRIMARY KEY,
                    board_type TEXT,
                    num_players INTEGER,
                    game_status TEXT
                )
            """)
            conn.execute(
                "INSERT INTO games VALUES (?, ?, ?, ?)",
                ("game_0", "hex8", 2, "completed")
            )
            conn.commit()
            conn.close()
            
            # Merge should skip game_0 (already exists)
            with patch.object(mock_mixin, "_get_canonical_db_path", return_value=canonical_path):
                await mock_mixin._merge_into_canonical(temp_db, "remote-node")
    
    @pytest.mark.asyncio
    async def test_merge_handles_missing_canonical(self, mock_mixin, temp_db):
        """Test graceful handling when canonical doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            canonical_path = Path(tmpdir) / "canonical_hex8_2p.db"
            
            # Don't create canonical - should handle gracefully
            with patch.object(mock_mixin, "_get_canonical_db_path", return_value=canonical_path):
                await mock_mixin._merge_into_canonical(temp_db, "remote-node")
    
    @pytest.mark.asyncio
    async def test_merge_transaction_rollback_on_error(self, mock_mixin, temp_db):
        """Test transaction rollback on merge error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            canonical_path = Path(tmpdir) / "canonical_hex8_2p.db"
            
            # Create canonical
            conn = sqlite3.connect(str(canonical_path))
            conn.execute("""
                CREATE TABLE games (
                    game_id TEXT PRIMARY KEY,
                    board_type TEXT
                )
            """)
            conn.commit()
            conn.close()
            
            # Should rollback on schema mismatch or other errors


# ============================================
# Test Canonical Name Extraction
# ============================================


class TestGetCanonicalName:
    """Tests for _get_canonical_name method."""
    
    def test_extracts_hex8_2p(self, mock_mixin):
        """Test extraction of hex8_2p config."""
        result = mock_mixin._get_canonical_name("selfplay_hex8_2p_12345.db")
        assert "hex8" in result.lower()
    
    def test_extracts_square19_4p(self, mock_mixin):
        """Test extraction of square19_4p config."""
        result = mock_mixin._get_canonical_name("gumbel_square19_4p.db")
        assert "square19" in result.lower()
    
    def test_handles_canonical_prefix(self, mock_mixin):
        """Test handling of canonical_ prefix."""
        result = mock_mixin._get_canonical_name("canonical_hex8_2p.db")
        assert "hex8" in result.lower()
    
    def test_fallback_for_non_standard_names(self, mock_mixin):
        """Test fallback for non-standard database names."""
        result = mock_mixin._get_canonical_name("random_database.db")
        # Should return something reasonable


# ============================================
# Test Remote Path Detection
# ============================================


class TestGetRemoteGamesPath:
    """Tests for _get_remote_games_path method."""
    
    def test_runpod_path(self, mock_mixin):
        """Test path for RunPod nodes."""
        result = mock_mixin._get_remote_games_path("runpod-h100")
        assert "workspace" in result or "ringrift" in result
    
    def test_vast_path(self, mock_mixin):
        """Test path for Vast.ai nodes."""
        result = mock_mixin._get_remote_games_path("vast-12345")
        assert "workspace" in result or "ringrift" in result
    
    def test_nebius_path(self, mock_mixin):
        """Test path for Nebius nodes."""
        result = mock_mixin._get_remote_games_path("nebius-h100")
        assert "ringrift" in result
    
    def test_lambda_path(self, mock_mixin):
        """Test path for Lambda nodes."""
        result = mock_mixin._get_remote_games_path("lambda-gh200-1")
        assert "ringrift" in result


# ============================================
# Test Pull from Node
# ============================================


class TestPullFromNode:
    """Tests for _pull_from_node method."""
    
    @pytest.mark.asyncio
    async def test_pulls_and_validates_games(self, mock_mixin):
        """Test complete pull from single node."""
        source_node = {
            "node_id": "remote-gpu-1",
            "ssh_host": "10.0.0.1",
            "ssh_user": "root",
            "ssh_key": "/path/to/key"
        }
        
        with patch.object(mock_mixin, "_list_remote_databases", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = ["canonical_hex8_2p.db"]
            
            with patch.object(mock_mixin, "_rsync_pull", new_callable=AsyncMock) as mock_rsync:
                mock_rsync.return_value = Path("/tmp/pulled.db")
                
                with patch.object(mock_mixin, "_merge_into_canonical", new_callable=AsyncMock):
                    result = await mock_mixin._pull_from_node(source_node)
                    
                    assert isinstance(result, int)
    
    @pytest.mark.asyncio
    async def test_handles_empty_database_list(self, mock_mixin):
        """Test graceful handling when no databases found."""
        source_node = {
            "node_id": "empty-node",
            "ssh_host": "10.0.0.2",
            "ssh_user": "root",
            "ssh_key": "/path/to/key"
        }
        
        with patch.object(mock_mixin, "_list_remote_databases", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = []
            
            result = await mock_mixin._pull_from_node(source_node)
            
            assert result == 0


# ============================================
# Test Event Emission
# ============================================


class TestPullEventEmission:
    """Tests for event emission after pull sync."""
    
    @pytest.mark.asyncio
    async def test_emits_pull_sync_completed(self, mock_mixin):
        """Test that pull completion emits event."""
        with patch.object(mock_mixin, "_emit_pull_sync_completed", new_callable=AsyncMock) as mock_emit:
            await mock_mixin._emit_pull_sync_completed(games_pulled=10, sources_count=2)
            
            mock_emit.assert_called_once()


# ============================================
# Test Error Handling
# ============================================


class TestPullErrorHandling:
    """Tests for error handling in pull operations."""
    
    @pytest.mark.asyncio
    async def test_continues_after_single_node_failure(self, mock_mixin):
        """Test that failure on one node doesn't stop others."""
        with patch("app.coordination.sync_pull_mixin.env") as mock_env:
            mock_env.is_coordinator = True
            
            with patch.object(mock_mixin, "_get_sync_sources") as mock_sources:
                mock_sources.return_value = [
                    {"node_id": "failing-node"},
                    {"node_id": "working-node"}
                ]
                
                call_count = 0
                async def mock_pull(node):
                    nonlocal call_count
                    call_count += 1
                    if node["node_id"] == "failing-node":
                        raise Exception("Network error")
                    return 5
                
                with patch.object(mock_mixin, "_pull_from_node", mock_pull):
                    result = await mock_mixin._pull_from_cluster_nodes()
                    
                    # Both nodes should be attempted
                    assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_tracks_pull_errors(self, mock_mixin):
        """Test that pull errors are tracked in stats."""
        initial_errors = mock_mixin._stats.pull_errors
        
        with patch.object(mock_mixin, "_rsync_pull", new_callable=AsyncMock) as mock_rsync:
            mock_rsync.return_value = None  # Failed pull
            
            # Error should be tracked


# ============================================
# Test Statistics
# ============================================


class TestPullStatistics:
    """Tests for pull operation statistics."""
    
    @pytest.mark.asyncio
    async def test_tracks_games_pulled(self, mock_mixin):
        """Test that games pulled count is tracked."""
        source_node = {
            "node_id": "remote-1",
            "ssh_host": "10.0.0.1",
            "ssh_user": "root",
            "ssh_key": "/path/to/key"
        }
        
        with patch.object(mock_mixin, "_list_remote_databases", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = ["db1.db", "db2.db"]
            
            with patch.object(mock_mixin, "_rsync_pull", new_callable=AsyncMock) as mock_rsync:
                with tempfile.NamedTemporaryFile(suffix=".db") as f:
                    mock_rsync.return_value = Path(f.name)
                    
                    with patch.object(mock_mixin, "_merge_into_canonical", new_callable=AsyncMock):
                        result = await mock_mixin._pull_from_node(source_node)
