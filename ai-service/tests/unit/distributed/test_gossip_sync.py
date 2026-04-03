"""Tests for gossip_sync module.

Tests the P2P gossip-based data synchronization components:
- BloomFilter: Probabilistic set membership testing
- GossipPeer: Peer configuration dataclass
- SyncState: Sync daemon state
- GossipSyncDaemon: Main sync daemon (partial testing without network)
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.distributed.gossip_sync import (
    BLOOM_FILTER_SIZE,
    BLOOM_HASH_COUNT,
    GOSSIP_PORT,
    MAX_GAMES_PER_PUSH,
    SYNC_INTERVAL,
    BloomFilter,
    GossipPeer,
    GossipSyncDaemon,
    SyncState,
    load_peer_config,
)


# ============================================================================
# BloomFilter Tests
# ============================================================================


class TestBloomFilter:
    """Tests for the BloomFilter class."""

    def test_default_size(self):
        """Test bloom filter uses default size."""
        bf = BloomFilter()
        assert bf.size == BLOOM_FILTER_SIZE
        assert bf.hash_count == BLOOM_HASH_COUNT

    def test_custom_size(self):
        """Test bloom filter with custom size."""
        bf = BloomFilter(size=1000, hash_count=3)
        assert bf.size == 1000
        assert bf.hash_count == 3

    def test_add_and_contains(self):
        """Test adding items and checking membership."""
        bf = BloomFilter(size=10000)

        bf.add("game1")
        bf.add("game2")
        bf.add("game3")

        assert "game1" in bf
        assert "game2" in bf
        assert "game3" in bf

    def test_not_contains(self):
        """Test items not added are (usually) not found."""
        bf = BloomFilter(size=100000)

        bf.add("game1")
        bf.add("game2")

        # With proper size, false positives should be rare
        # Test a few items that weren't added
        false_positive_count = 0
        for i in range(100):
            if f"not_added_{i}" in bf:
                false_positive_count += 1

        # Should have very few false positives
        assert false_positive_count < 10

    def test_many_items(self):
        """Test adding many items."""
        bf = BloomFilter()

        # Add 1000 items
        items = [f"game_{i}" for i in range(1000)]
        for item in items:
            bf.add(item)

        # All should be found
        for item in items:
            assert item in bf

    def test_to_bytes(self):
        """Test serialization to bytes."""
        bf = BloomFilter(size=1000)
        bf.add("game1")
        bf.add("game2")

        data = bf.to_bytes()
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_from_bytes(self):
        """Test deserialization from bytes."""
        bf = BloomFilter(size=1000)
        bf.add("game1")
        bf.add("game2")

        data = bf.to_bytes()
        restored = BloomFilter.from_bytes(data, size=1000)

        assert "game1" in restored
        assert "game2" in restored

    def test_serialization_roundtrip(self):
        """Test serialization roundtrip preserves data."""
        bf = BloomFilter(size=10000)

        items = [f"item_{i}" for i in range(100)]
        for item in items:
            bf.add(item)

        # Serialize and restore
        data = bf.to_bytes()
        restored = BloomFilter.from_bytes(data, size=10000)

        # All items should still be found
        for item in items:
            assert item in restored

    def test_empty_filter(self):
        """Test empty filter contains nothing."""
        bf = BloomFilter(size=1000)

        assert "anything" not in bf
        assert "game1" not in bf

    def test_hash_determinism(self):
        """Test hashing is deterministic."""
        bf1 = BloomFilter(size=10000)
        bf2 = BloomFilter(size=10000)

        bf1.add("test_item")
        bf2.add("test_item")

        # Both should have same bits set
        assert bf1.to_bytes() == bf2.to_bytes()


# ============================================================================
# GossipPeer Tests
# ============================================================================


class TestGossipPeer:
    """Tests for the GossipPeer dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        peer = GossipPeer(name="test", host="localhost")

        assert peer.name == "test"
        assert peer.host == "localhost"
        assert peer.port == GOSSIP_PORT
        assert peer.ssh_host == ""
        assert peer.ssh_user == "ubuntu"
        assert peer.ssh_port == 22
        assert peer.last_seen == 0.0
        assert peer.last_sync == 0.0
        assert peer.games_synced == 0
        assert peer.is_healthy is True

    def test_custom_values(self):
        """Test custom values are set correctly."""
        peer = GossipPeer(
            name="node1",
            host="192.168.1.1",
            port=9000,
            ssh_host="10.0.0.1",
            ssh_user="admin",
            ssh_port=2222,
            is_healthy=False,
        )

        assert peer.name == "node1"
        assert peer.host == "192.168.1.1"
        assert peer.port == 9000
        assert peer.ssh_host == "10.0.0.1"
        assert peer.ssh_user == "admin"
        assert peer.ssh_port == 2222
        assert peer.is_healthy is False

    def test_mutable_fields(self):
        """Test mutable fields can be updated."""
        peer = GossipPeer(name="test", host="localhost")

        peer.last_seen = 1000.0
        peer.last_sync = 900.0
        peer.games_synced = 50
        peer.is_healthy = False

        assert peer.last_seen == 1000.0
        assert peer.last_sync == 900.0
        assert peer.games_synced == 50
        assert peer.is_healthy is False


# ============================================================================
# SyncState Tests
# ============================================================================


class TestSyncState:
    """Tests for the SyncState dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        state = SyncState(node_id="node1")

        assert state.node_id == "node1"
        assert state.peers == {}
        assert state.known_game_ids == set()
        assert state.last_sync_time == 0.0
        assert state.total_games_pushed == 0
        assert state.total_games_pulled == 0
        assert state.sync_cycles == 0

    def test_add_peers(self):
        """Test adding peers to state."""
        state = SyncState(node_id="node1")

        peer = GossipPeer(name="peer1", host="localhost")
        state.peers["peer1"] = peer

        assert "peer1" in state.peers
        assert state.peers["peer1"].host == "localhost"

    def test_add_game_ids(self):
        """Test adding game IDs."""
        state = SyncState(node_id="node1")

        state.known_game_ids.add("game1")
        state.known_game_ids.add("game2")

        assert "game1" in state.known_game_ids
        assert "game2" in state.known_game_ids
        assert len(state.known_game_ids) == 2

    def test_update_counters(self):
        """Test updating counters."""
        state = SyncState(node_id="node1")

        state.total_games_pushed += 10
        state.total_games_pulled += 5
        state.sync_cycles += 1

        assert state.total_games_pushed == 10
        assert state.total_games_pulled == 5
        assert state.sync_cycles == 1


# ============================================================================
# GossipSyncDaemon Tests
# ============================================================================


class TestGossipSyncDaemon:
    """Tests for the GossipSyncDaemon class."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_peers_config(self):
        """Sample peers configuration."""
        return {
            "node1": {"ssh_host": "10.0.0.1", "gossip_host": "10.0.0.1"},
            "node2": {"ssh_host": "10.0.0.2", "gossip_host": "10.0.0.2"},
            "node3": {"ssh_host": "10.0.0.3", "gossip_host": "10.0.0.3"},
        }

    def test_init_basic(self, temp_data_dir, sample_peers_config):
        """Test basic initialization."""
        daemon = GossipSyncDaemon(
            node_id="node1",
            data_dir=temp_data_dir,
            peers_config=sample_peers_config,
        )

        assert daemon.node_id == "node1"
        assert daemon.data_dir == temp_data_dir
        assert daemon.listen_port == GOSSIP_PORT

    def test_init_excludes_self(self, temp_data_dir, sample_peers_config):
        """Test self is excluded from peers."""
        daemon = GossipSyncDaemon(
            node_id="node1",
            data_dir=temp_data_dir,
            peers_config=sample_peers_config,
        )

        # node1 (self) should not be in peers
        assert "node1" not in daemon.state.peers
        assert "node2" in daemon.state.peers
        assert "node3" in daemon.state.peers

    def test_init_custom_port(self, temp_data_dir, sample_peers_config):
        """Test custom listen port."""
        daemon = GossipSyncDaemon(
            node_id="node1",
            data_dir=temp_data_dir,
            peers_config=sample_peers_config,
            listen_port=9000,
        )

        assert daemon.listen_port == 9000

    def test_load_known_games_empty_dir(self, temp_data_dir, sample_peers_config):
        """Test loading games from empty directory."""
        daemon = GossipSyncDaemon(
            node_id="node1",
            data_dir=temp_data_dir,
            peers_config=sample_peers_config,
        )

        assert len(daemon.state.known_game_ids) == 0

    def test_load_known_games_with_db(self, temp_data_dir, sample_peers_config):
        """Test loading games from database."""
        # Create a canonical database with some games
        db_path = temp_data_dir / "canonical_square8.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE games (game_id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO games VALUES ('game1')")
        conn.execute("INSERT INTO games VALUES ('game2')")
        conn.execute("INSERT INTO games VALUES ('game3')")
        conn.commit()
        conn.close()

        daemon = GossipSyncDaemon(
            node_id="node1",
            data_dir=temp_data_dir,
            peers_config=sample_peers_config,
        )

        assert len(daemon.state.known_game_ids) == 3
        assert "game1" in daemon.state.known_game_ids
        assert "game2" in daemon.state.known_game_ids
        assert "game3" in daemon.state.known_game_ids

    def test_load_skips_schema_db(self, temp_data_dir, sample_peers_config):
        """Test schema databases are skipped."""
        # Create a schema database (should be skipped)
        schema_db = temp_data_dir / "schema_migrations.db"
        conn = sqlite3.connect(schema_db)
        conn.execute("CREATE TABLE games (game_id TEXT)")
        conn.execute("INSERT INTO games VALUES ('should_skip')")
        conn.commit()
        conn.close()

        daemon = GossipSyncDaemon(
            node_id="node1",
            data_dir=temp_data_dir,
            peers_config=sample_peers_config,
        )

        assert "should_skip" not in daemon.state.known_game_ids

    def test_build_bloom_filter(self, temp_data_dir, sample_peers_config):
        """Test building bloom filter from known games."""
        # Create canonical database with games
        db_path = temp_data_dir / "canonical_square8.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE games (game_id TEXT PRIMARY KEY)")
        for i in range(100):
            conn.execute(f"INSERT INTO games VALUES ('game_{i}')")
        conn.commit()
        conn.close()

        daemon = GossipSyncDaemon(
            node_id="node1",
            data_dir=temp_data_dir,
            peers_config=sample_peers_config,
        )

        bf = daemon._build_bloom_filter()

        # All known games should be in bloom filter
        for i in range(100):
            assert f"game_{i}" in bf

    def test_get_status(self, temp_data_dir, sample_peers_config):
        """Test getting daemon status."""
        daemon = GossipSyncDaemon(
            node_id="node1",
            data_dir=temp_data_dir,
            peers_config=sample_peers_config,
        )

        status = daemon.get_status()

        assert status["node_id"] == "node1"
        assert status["known_games"] == 0
        assert status["total_pushed"] == 0
        assert status["total_pulled"] == 0
        assert status["sync_cycles"] == 0
        assert "peers" in status

    def test_get_status_with_peers(self, temp_data_dir, sample_peers_config):
        """Test status includes peer information."""
        daemon = GossipSyncDaemon(
            node_id="node1",
            data_dir=temp_data_dir,
            peers_config=sample_peers_config,
        )

        status = daemon.get_status()

        assert "node2" in status["peers"]
        assert "node3" in status["peers"]
        assert status["peers"]["node2"]["is_healthy"] is True

    def test_store_games(self, temp_data_dir, sample_peers_config):
        """Test storing received games."""
        daemon = GossipSyncDaemon(
            node_id="node1",
            data_dir=temp_data_dir,
            peers_config=sample_peers_config,
        )

        games = [
            {"game_id": "new_game_1", "moves_json": json.dumps([{}, {}, {}, {}, {}])},
            {"game_id": "new_game_2", "moves_json": json.dumps([{}, {}, {}, {}, {}])},
        ]

        stored = daemon._store_games(games)

        assert stored == 2
        assert "new_game_1" in daemon.state.known_game_ids
        assert "new_game_2" in daemon.state.known_game_ids

    def test_store_games_empty_list(self, temp_data_dir, sample_peers_config):
        """Test storing empty game list."""
        daemon = GossipSyncDaemon(
            node_id="node1",
            data_dir=temp_data_dir,
            peers_config=sample_peers_config,
        )

        stored = daemon._store_games([])
        assert stored == 0

    def test_get_games_data_empty(self, temp_data_dir, sample_peers_config):
        """Test getting games when none exist."""
        daemon = GossipSyncDaemon(
            node_id="node1",
            data_dir=temp_data_dir,
            peers_config=sample_peers_config,
        )

        games = daemon._get_games_data(["game1", "game2"])
        assert games == []

    def test_get_games_data_with_db(self, temp_data_dir, sample_peers_config):
        """Test getting games from database."""
        # Create database with games
        db_path = temp_data_dir / "test.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE games (game_id TEXT PRIMARY KEY, data TEXT)")
        conn.execute("INSERT INTO games VALUES ('game1', 'data1')")
        conn.execute("INSERT INTO games VALUES ('game2', 'data2')")
        conn.commit()
        conn.close()

        daemon = GossipSyncDaemon(
            node_id="node1",
            data_dir=temp_data_dir,
            peers_config=sample_peers_config,
        )

        games = daemon._get_games_data(["game1", "game2"])

        assert len(games) == 2
        game_ids = {g["game_id"] for g in games}
        assert "game1" in game_ids
        assert "game2" in game_ids


# ============================================================================
# Async Daemon Tests
# ============================================================================


class TestGossipSyncDaemonAsync:
    """Async tests for GossipSyncDaemon."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_peers_config(self):
        """Sample peers configuration."""
        return {
            "node1": {"ssh_host": "10.0.0.1"},
            "node2": {"ssh_host": "10.0.0.2"},
        }

    @pytest.mark.asyncio
    async def test_start_and_stop(self, temp_data_dir, sample_peers_config):
        """Test starting and stopping daemon."""
        daemon = GossipSyncDaemon(
            node_id="node1",
            data_dir=temp_data_dir,
            peers_config=sample_peers_config,
            listen_port=0,  # Use random port
        )

        # Mock the sync loop to avoid actual syncing
        daemon._sync_loop = AsyncMock()

        await daemon.start()
        assert daemon._running is True
        assert daemon._server is not None

        await daemon.stop()
        assert daemon._running is False

    @pytest.mark.asyncio
    async def test_send_message(self, temp_data_dir, sample_peers_config):
        """Test message sending format."""
        daemon = GossipSyncDaemon(
            node_id="node1",
            data_dir=temp_data_dir,
            peers_config=sample_peers_config,
        )

        # Create mock writer
        writer = MagicMock()
        writer.drain = AsyncMock()

        message = {"type": "test", "data": "hello"}
        await daemon._send_message(writer, message)

        # Should have written length prefix + data
        assert writer.write.call_count == 2


# ============================================================================
# load_peer_config Tests
# ============================================================================


class TestLoadPeerConfig:
    """Tests for load_peer_config function."""

    def test_load_standard_hosts(self):
        """Test loading standard hosts from config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
standard_hosts:
  node1:
    ssh_host: "10.0.0.1"
    ssh_user: "admin"
    ssh_port: 2222
  node2:
    ssh_host: "10.0.0.2"
""")
            f.flush()

            peers = load_peer_config(Path(f.name))

            assert "node1" in peers
            assert "node2" in peers
            assert peers["node1"]["ssh_host"] == "10.0.0.1"
            assert peers["node1"]["ssh_user"] == "admin"
            assert peers["node1"]["ssh_port"] == 2222
            assert peers["node2"]["ssh_host"] == "10.0.0.2"

    def test_skip_tailscale_duplicates(self):
        """Test Tailscale duplicates are skipped."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
standard_hosts:
  node1:
    ssh_host: "10.0.0.1"
  lambda_gh200_a:
    ssh_host: "100.0.0.1"
""")
            f.flush()

            peers = load_peer_config(Path(f.name))

            assert "node1" in peers
            # Tailscale duplicate should be skipped
            assert "lambda_gh200_a" not in peers

    def test_default_gossip_port(self):
        """Test default gossip port is set."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
standard_hosts:
  node1:
    ssh_host: "10.0.0.1"
""")
            f.flush()

            peers = load_peer_config(Path(f.name))

            assert peers["node1"]["gossip_port"] == GOSSIP_PORT


# ============================================================================
# Constants Tests
# ============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_gossip_port(self):
        """Test gossip port is reasonable."""
        assert GOSSIP_PORT > 1024
        assert GOSSIP_PORT < 65536

    def test_sync_interval(self):
        """Test sync interval is reasonable."""
        assert SYNC_INTERVAL > 0
        assert SYNC_INTERVAL < 3600  # Less than 1 hour

    def test_max_games_per_push(self):
        """Test max games per push is reasonable."""
        assert MAX_GAMES_PER_PUSH > 0
        assert MAX_GAMES_PER_PUSH <= 1000

    def test_bloom_filter_size(self):
        """Test bloom filter size is reasonable."""
        assert BLOOM_FILTER_SIZE > 10000
        assert BLOOM_FILTER_SIZE <= 10000000

    def test_bloom_hash_count(self):
        """Test bloom hash count is reasonable."""
        assert BLOOM_HASH_COUNT >= 3
        assert BLOOM_HASH_COUNT <= 15
