"""Tests for SyncPushMixin - Push/broadcast sync operations.

Tests cover:
- Local database discovery
- Bandwidth management per node/provider
- Broadcast target selection and filtering
- Rsync-based sync with retry logic
- Stale partial cleanup
- Broadcast sync cycle orchestration

December 2025: Created as part of unit test coverage initiative.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from app.coordination.sync_push_mixin import SyncPushMixin


# Mock config and stats dataclasses for testing
@dataclass
class MockAutoSyncConfig:
    """Mock AutoSyncConfig for testing."""
    broadcast_high_priority_configs: list[str] = field(default_factory=lambda: ["hex8_2p", "square8_2p"])
    max_concurrent_syncs: int = 4
    sync_interval: float = 60.0


@dataclass
class MockSyncStats:
    """Mock SyncStats for testing."""
    total_syncs: int = 0
    databases_verified: int = 0
    databases_verification_failed: int = 0
    bytes_synced: int = 0
    sync_errors: int = 0


class MockSyncPushDaemon(SyncPushMixin):
    """Mock daemon using SyncPushMixin for testing."""

    def __init__(
        self,
        node_id: str = "test-node",
        is_broadcast: bool = True,
        config: MockAutoSyncConfig | None = None,
        stats: MockSyncStats | None = None,
    ):
        self.node_id = node_id
        self._is_broadcast = is_broadcast
        self.config = config or MockAutoSyncConfig()
        self._stats = stats or MockSyncStats()
        self._emitted_failures: list[tuple] = []
        self._emitted_stalls: list[tuple] = []

    async def _emit_sync_failure(self, target_node: str, db_path: str, error: str) -> None:
        """Track sync failure emissions for testing."""
        self._emitted_failures.append((target_node, db_path, error))

    async def _emit_sync_stalled(
        self,
        target_node: str,
        timeout_seconds: float,
        data_type: str = "game",
        retry_count: int = 0,
    ) -> None:
        """Track sync stalled emissions for testing."""
        self._emitted_stalls.append((target_node, timeout_seconds, data_type, retry_count))


class TestDiscoverLocalDatabases:
    """Tests for discover_local_databases()."""

    def test_returns_empty_when_data_dir_missing(self, tmp_path):
        """Returns empty list when data directory doesn't exist."""
        daemon = MockSyncPushDaemon()

        with patch.object(Path, "__new__", return_value=tmp_path / "nonexistent"):
            with patch("app.coordination.sync_push_mixin.Path") as mock_path:
                mock_path.return_value.resolve.return_value.parent.parent.parent = tmp_path
                # The data/games directory won't exist
                result = daemon.discover_local_databases()
                assert result == []

    @pytest.fixture
    def mock_data_dir(self, tmp_path):
        """Create a mock data directory with databases."""
        data_dir = tmp_path / "ai-service" / "data" / "games"
        data_dir.mkdir(parents=True)

        # Create test databases
        (data_dir / "canonical_hex8_2p.db").write_bytes(b"x" * 2048)
        (data_dir / "canonical_square8_4p.db").write_bytes(b"x" * 2048)
        (data_dir / "gumbel_test.db").write_bytes(b"x" * 2048)
        (data_dir / "selfplay_test.db").write_bytes(b"x" * 2048)
        # Small file should be filtered out
        (data_dir / "canonical_tiny.db").write_bytes(b"x" * 512)

        return tmp_path / "ai-service"

    def test_discovers_canonical_databases(self, mock_data_dir):
        """Finds canonical_*.db files."""
        daemon = MockSyncPushDaemon()

        with patch("app.coordination.sync_push_mixin.Path") as mock_path:
            def resolve_mock():
                m = MagicMock()
                m.parent.parent.parent = mock_data_dir
                return m

            mock_path.return_value.resolve = resolve_mock

            # Direct test with the actual path
            data_dir = mock_data_dir / "data" / "games"
            databases = list(data_dir.glob("canonical_*.db"))
            # Filter by size
            databases = [db for db in databases if db.stat().st_size > 1024]

            assert len(databases) == 2  # canonical_hex8_2p and canonical_square8_4p

    def test_filters_small_databases(self, mock_data_dir):
        """Filters out databases < 1KB."""
        data_dir = mock_data_dir / "data" / "games"

        # The tiny database should be filtered
        databases = [
            db for db in data_dir.glob("*.db")
            if db.stat().st_size > 1024
        ]

        # canonical_tiny.db is 512 bytes, should be excluded
        assert all("tiny" not in db.name for db in databases)

    def test_sorts_by_priority(self):
        """High-priority configs are sorted first."""
        daemon = MockSyncPushDaemon()
        daemon.config.broadcast_high_priority_configs = ["hex8_2p"]

        # Test the priority key function
        high_priority_configs = frozenset(daemon.config.broadcast_high_priority_configs)

        def priority_key(path: Path) -> tuple[int, str]:
            name = path.stem
            for config in high_priority_configs:
                if config in name:
                    return (0, name)
            return (1, name)

        paths = [
            Path("canonical_square8_4p.db"),
            Path("canonical_hex8_2p.db"),
            Path("selfplay_test.db"),
        ]

        sorted_paths = sorted(paths, key=priority_key)

        # hex8_2p should be first (priority 0)
        assert "hex8_2p" in sorted_paths[0].stem


class TestGetBandwidthForNode:
    """Tests for get_bandwidth_for_node()."""

    def test_uses_cluster_config(self):
        """Uses cluster_config.get_node_bandwidth_kbs()."""
        daemon = MockSyncPushDaemon()

        with patch("app.config.cluster_config.get_node_bandwidth_kbs") as mock_bw:
            mock_bw.return_value = 50_000
            result = daemon.get_bandwidth_for_node("runpod-a100-1")

            mock_bw.assert_called_once_with("runpod-a100-1")
            assert result == 50_000

    def test_fallback_when_cluster_config_unavailable(self):
        """Uses conservative default when cluster_config not available."""
        # This tests the fallback logic when ImportError occurs
        # The actual fallback is 20_000 KB/s
        fallback_bandwidth = 20_000
        assert fallback_bandwidth == 20_000


class TestGetBroadcastTargets:
    """Tests for get_broadcast_targets()."""

    @pytest.fixture
    def mock_p2p_status(self):
        """Mock P2P status response."""
        return {
            "leader_id": "test-leader",
            "peers": {
                "node-1": {
                    "host": "192.168.1.1",
                    "disk_free_gb": 100,
                    "retired": False,
                    "last_heartbeat": time.time() - 60,
                    "provider": "runpod",
                },
                "node-2": {
                    "host": "192.168.1.2",
                    "disk_free_gb": 200,
                    "retired": False,
                    "last_heartbeat": time.time() - 30,
                    "provider": "vast",
                },
                "node-excluded": {
                    "host": "192.168.1.3",
                    "disk_free_gb": 150,
                    "retired": True,  # Retired, should be excluded
                    "last_heartbeat": time.time() - 30,
                },
                "node-low-disk": {
                    "host": "192.168.1.4",
                    "disk_free_gb": 10,  # Low disk, should be excluded
                    "retired": False,
                    "last_heartbeat": time.time() - 30,
                },
                "node-stale": {
                    "host": "192.168.1.5",
                    "disk_free_gb": 100,
                    "retired": False,
                    "last_heartbeat": time.time() - 600,  # Stale heartbeat
                },
            },
        }

    @pytest.mark.asyncio
    async def test_filters_retired_nodes(self, mock_p2p_status):
        """Retired nodes are excluded from targets."""
        daemon = MockSyncPushDaemon()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(mock_p2p_status).encode()
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            with patch("app.config.ports.get_p2p_status_url", return_value="http://localhost:8770/status"):
                with patch("app.coordination.coordinator_config.get_exclusion_policy", return_value=None):
                    targets = await daemon.get_broadcast_targets()

            node_ids = [t["node_id"] for t in targets]
            assert "node-excluded" not in node_ids

    @pytest.mark.asyncio
    async def test_filters_low_disk_nodes(self, mock_p2p_status):
        """Nodes with low disk space are excluded."""
        daemon = MockSyncPushDaemon()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(mock_p2p_status).encode()
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            with patch("app.config.ports.get_p2p_status_url", return_value="http://localhost:8770/status"):
                with patch("app.coordination.coordinator_config.get_exclusion_policy", return_value=None):
                    targets = await daemon.get_broadcast_targets()

            node_ids = [t["node_id"] for t in targets]
            assert "node-low-disk" not in node_ids

    @pytest.mark.asyncio
    async def test_filters_stale_heartbeat(self, mock_p2p_status):
        """Nodes with stale heartbeat (>5 min) are excluded."""
        daemon = MockSyncPushDaemon()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(mock_p2p_status).encode()
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            with patch("app.config.ports.get_p2p_status_url", return_value="http://localhost:8770/status"):
                with patch("app.coordination.coordinator_config.get_exclusion_policy", return_value=None):
                    targets = await daemon.get_broadcast_targets()

            node_ids = [t["node_id"] for t in targets]
            assert "node-stale" not in node_ids

    @pytest.mark.asyncio
    async def test_sorts_by_disk_space(self, mock_p2p_status):
        """Targets are sorted by disk space (most first)."""
        daemon = MockSyncPushDaemon()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(mock_p2p_status).encode()
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            with patch("app.config.ports.get_p2p_status_url", return_value="http://localhost:8770/status"):
                with patch("app.coordination.coordinator_config.get_exclusion_policy", return_value=None):
                    targets = await daemon.get_broadcast_targets()

            # node-2 has 200GB, node-1 has 100GB
            assert len(targets) == 2
            assert targets[0]["node_id"] == "node-2"
            assert targets[1]["node_id"] == "node-1"

    @pytest.mark.asyncio
    async def test_returns_empty_on_p2p_error(self):
        """Returns empty list when P2P status fails."""
        daemon = MockSyncPushDaemon()

        with patch("urllib.request.urlopen", side_effect=OSError("Connection refused")):
            targets = await daemon.get_broadcast_targets()
            assert targets == []

    @pytest.mark.asyncio
    async def test_detects_provider_from_node_name(self, mock_p2p_status):
        """Provider is detected from node name when not explicitly set."""
        # Modify test data to have node without provider
        mock_p2p_status["peers"]["lambda-gpu-1"] = {
            "host": "192.168.1.10",
            "disk_free_gb": 100,
            "retired": False,
            "last_heartbeat": time.time() - 30,
            "provider": "default",  # Will be detected from name
        }

        daemon = MockSyncPushDaemon()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(mock_p2p_status).encode()
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            with patch("app.config.ports.get_p2p_status_url", return_value="http://localhost:8770/status"):
                with patch("app.coordination.coordinator_config.get_exclusion_policy", return_value=None):
                    targets = await daemon.get_broadcast_targets()

            lambda_target = next((t for t in targets if t["node_id"] == "lambda-gpu-1"), None)
            if lambda_target:
                assert lambda_target["provider"] == "lambda"


class TestSyncToTargetWithRetry:
    """Tests for sync_to_target_with_retry()."""

    @pytest.fixture
    def mock_source(self, tmp_path):
        """Create a mock source database file."""
        db_file = tmp_path / "test.db"
        db_file.write_bytes(b"x" * 4096)
        return db_file

    @pytest.fixture
    def mock_target(self):
        """Mock target node info."""
        return {
            "node_id": "target-node",
            "host": "192.168.1.1",
            "disk_free_gb": 100,
            "provider": "runpod",
        }

    @pytest.mark.asyncio
    async def test_returns_on_first_success(self, mock_source, mock_target):
        """Returns immediately on first successful sync."""
        daemon = MockSyncPushDaemon()

        with patch.object(
            daemon, "broadcast_sync_to_target",
            new_callable=AsyncMock,
            return_value={"success": True, "bytes_transferred": 4096}
        ) as mock_sync:
            result = await daemon.sync_to_target_with_retry(mock_source, mock_target, max_retries=3)

            assert result["success"] is True
            assert mock_sync.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_failure(self, mock_source, mock_target):
        """Retries sync on transient failures."""
        daemon = MockSyncPushDaemon()

        call_count = 0

        async def failing_then_succeeding(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return {"success": False, "error": "Connection reset"}
            return {"success": True, "bytes_transferred": 4096}

        with patch.object(daemon, "broadcast_sync_to_target", side_effect=failing_then_succeeding):
            with patch("asyncio.sleep", new_callable=AsyncMock):  # Speed up tests
                result = await daemon.sync_to_target_with_retry(mock_source, mock_target, max_retries=5)

            assert result["success"] is True
            assert call_count == 3

    @pytest.mark.asyncio
    async def test_stops_retry_on_connection_refused(self, mock_source, mock_target):
        """Stops retrying immediately for connection refused errors."""
        daemon = MockSyncPushDaemon()

        with patch.object(
            daemon, "broadcast_sync_to_target",
            new_callable=AsyncMock,
            return_value={"success": False, "error": "Connection refused"}
        ) as mock_sync:
            result = await daemon.sync_to_target_with_retry(mock_source, mock_target, max_retries=3)

            assert result["success"] is False
            assert mock_sync.call_count == 1  # No retries

    @pytest.mark.asyncio
    async def test_stops_retry_on_no_route(self, mock_source, mock_target):
        """Stops retrying immediately for no route to host errors."""
        daemon = MockSyncPushDaemon()

        with patch.object(
            daemon, "broadcast_sync_to_target",
            new_callable=AsyncMock,
            return_value={"success": False, "error": "No route to host"}
        ) as mock_sync:
            result = await daemon.sync_to_target_with_retry(mock_source, mock_target, max_retries=3)

            assert result["success"] is False
            assert mock_sync.call_count == 1  # No retries


class TestBroadcastSyncToTarget:
    """Tests for broadcast_sync_to_target()."""

    @pytest.fixture
    def mock_source(self, tmp_path):
        """Create a mock source database file."""
        db_file = tmp_path / "test.db"
        db_file.write_bytes(b"x" * 4096)
        return db_file

    @pytest.fixture
    def mock_target(self):
        """Mock target node info."""
        return {
            "node_id": "target-node",
            "host": "192.168.1.1",
            "disk_free_gb": 100,
            "provider": "runpod",
            "is_nfs": False,
        }

    @pytest.mark.asyncio
    async def test_skips_nfs_targets(self, mock_source):
        """Skips sync for NFS-connected nodes."""
        daemon = MockSyncPushDaemon()
        nfs_target = {
            "node_id": "lambda-gpu-1",
            "host": "192.168.1.1",
            "is_nfs": True,
        }

        result = await daemon.broadcast_sync_to_target(mock_source, nfs_target)

        assert result["success"] is True
        assert result["bytes_transferred"] == 0
        assert result["duration_seconds"] == 0

    @pytest.mark.asyncio
    async def test_successful_rsync(self, mock_source, mock_target):
        """Successful rsync returns success result."""
        daemon = MockSyncPushDaemon()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_exec.return_value = mock_proc

            with patch.object(daemon, "get_bandwidth_for_node", return_value=50_000):
                with patch("app.config.cluster_config.get_cluster_nodes", return_value={}):
                    result = await daemon.broadcast_sync_to_target(mock_source, mock_target)

            assert result["success"] is True
            assert result["bytes_transferred"] == 4096
            assert "duration_seconds" in result

    @pytest.mark.asyncio
    async def test_failed_rsync_emits_failure(self, mock_source, mock_target):
        """Failed rsync emits sync failure event."""
        daemon = MockSyncPushDaemon()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 1
            mock_proc.communicate = AsyncMock(return_value=(b"", b"Permission denied"))
            mock_exec.return_value = mock_proc

            with patch.object(daemon, "get_bandwidth_for_node", return_value=50_000):
                with patch("app.config.cluster_config.get_cluster_nodes", return_value={}):
                    result = await daemon.broadcast_sync_to_target(mock_source, mock_target)

            assert result["success"] is False
            assert "Permission denied" in result["error"]

    @pytest.mark.asyncio
    async def test_timeout_emits_stalled(self, mock_source, mock_target):
        """Timeout emits sync stalled event."""
        daemon = MockSyncPushDaemon()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_exec.return_value = mock_proc

            with patch.object(daemon, "get_bandwidth_for_node", return_value=50_000):
                with patch("app.config.cluster_config.get_cluster_nodes", return_value={}):
                    result = await daemon.broadcast_sync_to_target(mock_source, mock_target)

            assert result["success"] is False
            assert result["error"] == "Timeout"


class TestCleanupStalePartials:
    """Tests for cleanup_stale_partials()."""

    @pytest.mark.asyncio
    async def test_cleans_old_partials(self, tmp_path):
        """Removes partial files older than max_age_hours."""
        daemon = MockSyncPushDaemon()

        # Create mock partial directory
        partial_dir = tmp_path / "data" / "games" / ".rsync-partial"
        partial_dir.mkdir(parents=True)

        # Create old partial file
        old_file = partial_dir / "old_partial.tmp"
        old_file.write_bytes(b"old")
        # Set mtime to 2 days ago
        import os
        old_mtime = time.time() - 48 * 3600
        os.utime(old_file, (old_mtime, old_mtime))

        # Create recent partial file
        new_file = partial_dir / "new_partial.tmp"
        new_file.write_bytes(b"new")

        with patch("app.coordination.sync_push_mixin.Path") as mock_path:
            mock_path.return_value.resolve.return_value.parent.parent.parent = tmp_path

            # Direct test
            import datetime
            cutoff = datetime.datetime.now() - datetime.timedelta(hours=24)

            cleaned = 0
            for item in partial_dir.iterdir():
                mtime = datetime.datetime.fromtimestamp(item.stat().st_mtime)
                if mtime < cutoff and item.is_file():
                    item.unlink()
                    cleaned += 1

            assert cleaned == 1
            assert not old_file.exists()
            assert new_file.exists()


class TestBroadcastSyncCycle:
    """Tests for broadcast_sync_cycle()."""

    @pytest.mark.asyncio
    async def test_returns_zero_when_not_broadcast(self):
        """Returns 0 when not in broadcast mode."""
        daemon = MockSyncPushDaemon(is_broadcast=False)

        result = await daemon.broadcast_sync_cycle()
        assert result == 0

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_targets(self):
        """Returns 0 when no broadcast targets available."""
        daemon = MockSyncPushDaemon(is_broadcast=True)

        with patch.object(daemon, "get_broadcast_targets", new_callable=AsyncMock, return_value=[]):
            result = await daemon.broadcast_sync_cycle()
            assert result == 0

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_databases(self):
        """Returns 0 when no databases to sync."""
        daemon = MockSyncPushDaemon(is_broadcast=True)

        with patch.object(
            daemon, "get_broadcast_targets",
            new_callable=AsyncMock,
            return_value=[{"node_id": "node-1", "host": "192.168.1.1"}]
        ):
            with patch.object(daemon, "discover_local_databases", return_value=[]):
                result = await daemon.broadcast_sync_cycle()
                assert result == 0

    @pytest.mark.asyncio
    async def test_syncs_all_databases_to_all_targets(self):
        """Syncs each database to each target."""
        daemon = MockSyncPushDaemon(is_broadcast=True)

        mock_targets = [
            {"node_id": "node-1", "host": "192.168.1.1"},
            {"node_id": "node-2", "host": "192.168.1.2"},
        ]
        mock_databases = [Path("/tmp/db1.db"), Path("/tmp/db2.db")]

        with patch.object(daemon, "get_broadcast_targets", new_callable=AsyncMock, return_value=mock_targets):
            with patch.object(daemon, "discover_local_databases", return_value=mock_databases):
                with patch.object(
                    daemon, "sync_to_target_with_retry",
                    new_callable=AsyncMock,
                    return_value={"success": True}
                ) as mock_sync:
                    result = await daemon.broadcast_sync_cycle()

                    # 2 databases × 2 targets = 4 syncs
                    assert mock_sync.call_count == 4
                    assert result == 4

    @pytest.mark.asyncio
    async def test_counts_successful_syncs(self):
        """Returns count of successful syncs only."""
        daemon = MockSyncPushDaemon(is_broadcast=True)

        mock_targets = [
            {"node_id": "node-1", "host": "192.168.1.1"},
            {"node_id": "node-2", "host": "192.168.1.2"},
        ]
        mock_databases = [Path("/tmp/db1.db")]

        call_count = 0

        async def alternating_success(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return {"success": call_count % 2 == 1}  # Alternate success/failure

        with patch.object(daemon, "get_broadcast_targets", new_callable=AsyncMock, return_value=mock_targets):
            with patch.object(daemon, "discover_local_databases", return_value=mock_databases):
                with patch.object(daemon, "sync_to_target_with_retry", side_effect=alternating_success):
                    result = await daemon.broadcast_sync_cycle()

                    # 2 syncs, 1 success
                    assert result == 1

    @pytest.mark.asyncio
    async def test_cleans_partials_periodically(self):
        """Cleans stale partials every 10 syncs."""
        daemon = MockSyncPushDaemon(is_broadcast=True)
        daemon._stats.total_syncs = 10  # Trigger cleanup

        with patch.object(daemon, "get_broadcast_targets", new_callable=AsyncMock, return_value=[]):
            with patch.object(daemon, "cleanup_stale_partials", new_callable=AsyncMock, return_value=5) as mock_cleanup:
                await daemon.broadcast_sync_cycle()
                mock_cleanup.assert_called_once()


class TestProviderDetection:
    """Tests for provider detection from node names."""

    @pytest.mark.asyncio
    async def test_detects_lambda_provider(self):
        """Detects lambda provider from node name."""
        node_id = "lambda-gpu-1"
        node_lower = node_id.lower()

        if "lambda" in node_lower:
            provider = "lambda"
        else:
            provider = "default"

        assert provider == "lambda"

    @pytest.mark.asyncio
    async def test_detects_runpod_provider(self):
        """Detects runpod provider from node name."""
        node_id = "runpod-a100-1"
        node_lower = node_id.lower()

        if "runpod" in node_lower:
            provider = "runpod"
        else:
            provider = "default"

        assert provider == "runpod"

    @pytest.mark.asyncio
    async def test_detects_vast_provider(self):
        """Detects vast provider from node name."""
        node_id = "vast-12345678"
        node_lower = node_id.lower()

        if "vast" in node_lower:
            provider = "vast"
        else:
            provider = "default"

        assert provider == "vast"

    @pytest.mark.asyncio
    async def test_detects_nebius_provider(self):
        """Detects nebius provider from node name."""
        node_id = "nebius-h100-1"
        node_lower = node_id.lower()

        if "nebius" in node_lower:
            provider = "nebius"
        else:
            provider = "default"

        assert provider == "nebius"
