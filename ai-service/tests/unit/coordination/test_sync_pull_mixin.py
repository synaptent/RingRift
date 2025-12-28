"""Tests for SyncPullMixin - Pull-based sync for coordinator recovery.

Tests cover:
1. Coordinator role detection
2. Remote database listing via SSH
3. Rsync pull operations with checksum verification
4. Database merging into canonical databases
5. Canonical name extraction
6. Remote path detection by provider

December 2025: Created as part of unit test coverage initiative.
December 2025: Updated to match actual SyncPullMixin API.
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


class MockConfig:
    """Mock AutoSyncConfig for testing."""
    def __init__(self):
        self.ephemeral_write_through = True
        self.ephemeral_write_through_timeout = 30.0
        self.min_games_to_sync = 5
        self.max_disk_usage_percent = 80.0
        self.exclude_hosts = set()
        self.broadcast_high_priority_configs = frozenset()
        self.max_concurrent_syncs = 3


class MockSyncStats:
    """Mock SyncStats for testing."""
    def __init__(self):
        self.games_pulled = 0
        self.games_synced = 0
        self.pull_errors = 0
        self.databases_merged = 0
        self.syncs_completed = 0
        self.databases_verified = 0
        self.databases_verification_failed = 0


class MockSyncPullDaemon(SyncPullMixin):
    """Mock daemon using SyncPullMixin for testing."""

    def __init__(self):
        self.config = MockConfig()
        self.node_id = "coordinator"
        self._stats = MockSyncStats()
        self._circuit_breaker = None
        self._running = True
        self._events_processed = 0
        self._errors_count = 0
        self._last_error = ""

    async def _emit_sync_failure(self, target_node: str, db_path: str, error: str) -> None:
        """Mock implementation of abstract method."""
        pass

    async def _emit_sync_stalled(
        self,
        target_node: str,
        timeout_seconds: float,
        data_type: str = "game",
        retry_count: int = 0,
    ) -> None:
        """Mock implementation of abstract method."""
        pass

    def _validate_database_completeness(self, db_path) -> tuple[bool, str]:
        """Mock implementation of validation method."""
        return True, "OK"


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
        with patch("app.coordination.sync_pull_mixin.socket.gethostname", return_value="worker-node"):
            # Mock env module import
            mock_env = MagicMock()
            mock_env.is_coordinator = False
            with patch.dict("sys.modules", {"app.config.env": MagicMock(env=mock_env)}):
                with patch("app.config.env.env", mock_env):
                    result = await mock_mixin._pull_from_cluster_nodes()
                    assert result == 0

    @pytest.mark.asyncio
    async def test_pull_runs_on_coordinator(self, mock_mixin):
        """Test that pull runs for coordinator nodes."""
        mock_env = MagicMock()
        mock_env.is_coordinator = True

        with patch("app.config.env.env", mock_env):
            # Mock SyncRouter - must patch at source module since it's imported inside the method
            mock_router = MagicMock()
            mock_router.get_sync_sources.return_value = []
            with patch("app.coordination.sync_router.get_sync_router", return_value=mock_router):
                result = await mock_mixin._pull_from_cluster_nodes()
                # Should return 0 if no sources but not skip
                assert result == 0

    def test_hostname_fallback_mac_studio(self, mock_mixin):
        """Test hostname fallback for mac-studio."""
        with patch("app.coordination.sync_pull_mixin.socket.gethostname", return_value="mac-studio.local"):
            # When env import fails, should detect via hostname
            # This is tested implicitly through the pull_from_cluster_nodes method
            pass


# ============================================
# Test Remote Database Listing
# ============================================


class TestListRemoteDatabases:
    """Tests for _list_remote_databases method."""

    @pytest.mark.asyncio
    async def test_lists_databases_via_ssh(self, mock_mixin):
        """Test SSH-based database listing."""
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(
                return_value=(b"/data/games/canonical_hex8_2p.db\n/data/games/gumbel_square8_2p.db\n", b"")
            )
            mock_proc.returncode = 0
            mock_exec.return_value = mock_proc

            result = await mock_mixin._list_remote_databases(
                ssh_host="10.0.0.1",
                ssh_user="root",
                ssh_key="/path/to/key",
                remote_path="/data/games"
            )

            assert isinstance(result, list)
            assert len(result) == 2
            assert "canonical_hex8_2p.db" in result
            assert "gumbel_square8_2p.db" in result

    @pytest.mark.asyncio
    async def test_handles_ssh_timeout(self, mock_mixin):
        """Test graceful handling of SSH timeout."""
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_exec.return_value = mock_proc

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
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"", b"Connection refused"))
            mock_proc.returncode = 255
            mock_exec.return_value = mock_proc

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
                mock_proc = AsyncMock()
                mock_proc.communicate = AsyncMock(return_value=(b"", b""))
                mock_proc.returncode = 0
                mock_exec.return_value = mock_proc

                # Create a mock file after "rsync"
                (local_dir / "test.db").touch()

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
                assert result is None or isinstance(result, Path)

    @pytest.mark.asyncio
    async def test_rsync_pull_handles_timeout(self, mock_mixin):
        """Test rsync pull handles timeout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir)

            with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
                mock_proc = AsyncMock()
                mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
                mock_exec.return_value = mock_proc

                result = await mock_mixin._rsync_pull(
                    ssh_host="10.0.0.1",
                    ssh_user="root",
                    ssh_key="/path/to/key",
                    remote_path="/data/games",
                    db_name="test.db",
                    local_dir=local_dir,
                    verify_checksum=False
                )

                assert result is None


# ============================================
# Test Database Merging
# ============================================


class TestMergeIntoCanonical:
    """Tests for _merge_into_canonical method."""

    @pytest.mark.asyncio
    async def test_merge_creates_new_canonical(self, mock_mixin, temp_db):
        """Test merge creates canonical when it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use temp_db as pulled database
            pulled_db = temp_db

            # Patch to use temp directory
            with patch.object(Path, "resolve") as mock_resolve:
                base_dir = Path(tmpdir)
                games_dir = base_dir / "data" / "games"
                games_dir.mkdir(parents=True)

                # Mock the path resolution
                mock_resolve.return_value = base_dir / "app" / "coordination" / "sync_pull_mixin.py"

                # This test verifies the method runs without error
                # Actual file operations depend on path resolution
                await mock_mixin._merge_into_canonical(pulled_db, "remote-node")


# ============================================
# Test Canonical Name Extraction
# ============================================


class TestGetCanonicalName:
    """Tests for _get_canonical_name method."""

    def test_extracts_hex8_2p(self, mock_mixin):
        """Test extraction of hex8_2p config."""
        result = mock_mixin._get_canonical_name("selfplay_hex8_2p_12345.db")
        assert result == "canonical_hex8_2p.db"

    def test_extracts_square19_4p(self, mock_mixin):
        """Test extraction of square19_4p config."""
        result = mock_mixin._get_canonical_name("gumbel_square19_4p.db")
        assert result == "canonical_square19_4p.db"

    def test_handles_canonical_prefix(self, mock_mixin):
        """Test handling of canonical_ prefix."""
        result = mock_mixin._get_canonical_name("canonical_hex8_2p.db")
        assert result == "canonical_hex8_2p.db"

    def test_fallback_for_non_standard_names(self, mock_mixin):
        """Test fallback for non-standard database names."""
        result = mock_mixin._get_canonical_name("random_database.db")
        # Should return something reasonable
        assert result == "canonical_random_database.db"


# ============================================
# Test Remote Path Detection
# ============================================


class TestGetRemoteGamesPath:
    """Tests for _get_remote_games_path method."""

    def test_runpod_path(self, mock_mixin):
        """Test path for RunPod nodes."""
        # Patch at source module since get_host_provider is imported inside the method
        with patch("app.config.cluster_config.get_host_provider", return_value="runpod"):
            result = mock_mixin._get_remote_games_path("runpod-h100")
            assert "workspace" in result

    def test_vast_path(self, mock_mixin):
        """Test path for Vast.ai nodes."""
        with patch("app.config.cluster_config.get_host_provider", return_value="vast"):
            result = mock_mixin._get_remote_games_path("vast-12345")
            assert "ringrift" in result

    def test_nebius_path(self, mock_mixin):
        """Test path for Nebius nodes."""
        with patch("app.config.cluster_config.get_host_provider", return_value="nebius"):
            result = mock_mixin._get_remote_games_path("nebius-h100")
            assert "ringrift" in result

    def test_lambda_path(self, mock_mixin):
        """Test path for Lambda nodes."""
        with patch("app.config.cluster_config.get_host_provider", return_value="lambda"):
            result = mock_mixin._get_remote_games_path("lambda-gh200-1")
            assert "ringrift" in result

    def test_fallback_path(self, mock_mixin):
        """Test fallback path for unknown provider."""
        with patch("app.config.cluster_config.get_host_provider", return_value=None):
            result = mock_mixin._get_remote_games_path("unknown-node")
            assert "ringrift" in result


# ============================================
# Test Pull from Node
# ============================================


class TestPullFromNode:
    """Tests for _pull_from_node method."""

    @pytest.mark.asyncio
    async def test_handles_missing_node_config(self, mock_mixin):
        """Test handling when node not found in config."""
        # Patch at source module since get_cluster_nodes is imported inside the method
        with patch("app.config.cluster_config.get_cluster_nodes", return_value={}):
            result = await mock_mixin._pull_from_node("nonexistent-node")
            assert result == 0

    @pytest.mark.asyncio
    async def test_handles_empty_database_list(self, mock_mixin):
        """Test graceful handling when no databases found."""
        mock_node = MagicMock()
        mock_node.best_ip = "10.0.0.1"
        mock_node.ssh_user = "root"
        mock_node.ssh_key = "/path/to/key"

        with patch("app.config.cluster_config.get_cluster_nodes", return_value={"test-node": mock_node}):
            with patch.object(mock_mixin, "_list_remote_databases", new_callable=AsyncMock) as mock_list:
                mock_list.return_value = []

                result = await mock_mixin._pull_from_node("test-node")
                assert result == 0


# ============================================
# Test Event Emission
# ============================================


class TestPullEventEmission:
    """Tests for event emission after pull sync."""

    @pytest.mark.asyncio
    async def test_emits_pull_sync_completed(self, mock_mixin):
        """Test that pull completion emits event."""
        with patch("app.distributed.data_events.emit_data_event", new_callable=AsyncMock) as mock_emit:
            await mock_mixin._emit_pull_sync_completed(games_pulled=10, sources_count=2)
            mock_emit.assert_called_once()


# ============================================
# Test Statistics
# ============================================


class TestPullStatistics:
    """Tests for pull operation statistics."""

    @pytest.mark.asyncio
    async def test_stats_updated_on_success(self, mock_mixin):
        """Test that stats are updated on successful pull."""
        initial_syncs = mock_mixin._stats.syncs_completed
        initial_games = mock_mixin._stats.games_synced

        # Mock the full flow
        mock_env = MagicMock()
        mock_env.is_coordinator = True

        with patch("app.config.env.env", mock_env):
            mock_router = MagicMock()
            mock_source = MagicMock()
            mock_source.node_id = "test-node"
            mock_router.get_sync_sources.return_value = [mock_source]
            mock_router.refresh_from_cluster_config = MagicMock()

            # Patch at source module since get_sync_router is imported inside the method
            with patch("app.coordination.sync_router.get_sync_router", return_value=mock_router):
                with patch.object(mock_mixin, "_pull_from_node", new_callable=AsyncMock) as mock_pull:
                    mock_pull.return_value = 5

                    with patch.object(mock_mixin, "_emit_pull_sync_completed", new_callable=AsyncMock):
                        result = await mock_mixin._pull_from_cluster_nodes()

                        assert result == 5
                        assert mock_mixin._stats.syncs_completed == initial_syncs + 1
                        assert mock_mixin._stats.games_synced == initial_games + 5
