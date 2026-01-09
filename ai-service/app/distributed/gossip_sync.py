"""P2P Gossip-based Data Synchronization for RingRift.

Implements an eventually-consistent data replication system where each host
replicates data to its neighbors. This provides:
- No single point of failure
- Automatic recovery when hosts rejoin
- Load distributed across the cluster
- Resilience to network partitions

Architecture:
- Each host maintains a list of 2-3 "neighbors"
- Every sync interval, hosts exchange data summaries (bloom filters)
- Missing data is pulled from neighbors that have it
- New data is pushed to neighbors that don't have it

Usage:
    # Start gossip daemon on each host
    python -m app.distributed.gossip_sync --start

    # Check sync status
    python -m app.distributed.gossip_sync --status
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import socket
import sqlite3
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Import centralized port constants (December 2025)
from app.config.ports import GOSSIP_PORT

logger = logging.getLogger(__name__)

# Constants (GOSSIP_PORT imported from app.config.ports)
SYNC_INTERVAL = 60  # Seconds between sync cycles
MAX_GAMES_PER_PUSH = 100  # Max games to push per cycle
BLOOM_FILTER_SIZE = 100000  # Bloom filter bits
BLOOM_HASH_COUNT = 7  # Number of hash functions


@dataclass
class GossipPeer:
    """Configuration for a gossip peer."""
    name: str
    host: str
    port: int = GOSSIP_PORT
    ssh_host: str = ""
    ssh_user: str = "ubuntu"
    ssh_port: int = 22
    last_seen: float = 0.0
    last_sync: float = 0.0
    games_synced: int = 0
    is_healthy: bool = True


@dataclass
class GossipSyncState:
    """State of the gossip sync daemon.

    Note: This is gossip-specific state tracking.
    For sync operation states (PENDING, IN_PROGRESS, etc.), use sync_constants.SyncState.
    """
    node_id: str
    peers: dict[str, GossipPeer] = field(default_factory=dict)
    known_game_ids: set[str] = field(default_factory=set)
    last_sync_time: float = 0.0
    total_games_pushed: int = 0
    total_games_pulled: int = 0
    sync_cycles: int = 0


# Backward-compatible alias (tests and older callers import SyncState)
SyncState = GossipSyncState


# Import enhanced BloomFilter from centralized module (December 2025)
# This provides additional features: compression, merge, stats, optimal sizing
from app.coordination.sync_bloom_filter import SyncBloomFilter as BloomFilter  # noqa: E402


class _LegacyBloomFilter:
    """Legacy bloom filter - DEPRECATED. Use app.coordination.sync_bloom_filter instead.

    Kept for reference only. This class is not used; BloomFilter is imported from
    the centralized sync_bloom_filter module which provides additional features.
    """

    def __init__(self, size: int = BLOOM_FILTER_SIZE, hash_count: int = BLOOM_HASH_COUNT):
        self.size = size
        self.hash_count = hash_count
        self.bits = bytearray(size // 8 + 1)

    def _hashes(self, item: str) -> list[int]:
        """Generate hash positions for an item (not for security, bloom filter only)."""
        h1 = int(hashlib.md5(item.encode(), usedforsecurity=False).hexdigest(), 16)
        h2 = int(hashlib.sha1(item.encode(), usedforsecurity=False).hexdigest(), 16)
        return [(h1 + i * h2) % self.size for i in range(self.hash_count)]

    def add(self, item: str):
        """Add an item to the filter."""
        for pos in self._hashes(item):
            self.bits[pos // 8] |= (1 << (pos % 8))

    def __contains__(self, item: str) -> bool:
        """Check if an item might be in the filter."""
        return all(
            self.bits[pos // 8] & (1 << (pos % 8))
            for pos in self._hashes(item)
        )

    def to_bytes(self) -> bytes:
        """Serialize the filter."""
        return bytes(self.bits)

    @classmethod
    def from_bytes(cls, data: bytes, size: int = BLOOM_FILTER_SIZE) -> BloomFilter:
        """Deserialize a filter."""
        bf = cls(size=size)
        bf.bits = bytearray(data)
        return bf


class GossipSyncDaemon:
    """Daemon that handles P2P gossip-based data synchronization."""

    def __init__(
        self,
        node_id: str,
        data_dir: Path,
        peers_config: dict[str, dict],
        listen_port: int = GOSSIP_PORT,
        exclude_hosts: list[str] | None = None,
    ):
        self.node_id = node_id
        self.data_dir = Path(data_dir)
        self.listen_port = listen_port
        self.state = GossipSyncState(node_id=node_id)
        self._running = False
        self._server: asyncio.Server | None = None

        # Load hosts to exclude from receiving synced data (e.g., coordinator)
        self.exclude_hosts = set(exclude_hosts or [])
        self.exclude_hosts.update(self._load_exclude_hosts_from_config())

        # Initialize peers from config
        for name, config in peers_config.items():
            if name == node_id:
                continue  # Don't add self
            if name in self.exclude_hosts:
                continue  # Don't sync TO excluded hosts
            self.state.peers[name] = GossipPeer(
                name=name,
                host=config.get("gossip_host", config.get("ssh_host", "")),
                port=config.get("gossip_port", GOSSIP_PORT),
                ssh_host=config.get("ssh_host", ""),
                ssh_user=config.get("ssh_user", "ubuntu"),
                ssh_port=config.get("ssh_port", 22),
            )

        if self.exclude_hosts:
            logger.info(f"Excluding hosts from sync: {self.exclude_hosts}")

        # Load known game IDs from local databases
        self._load_known_games()

    def _load_exclude_hosts_from_config(self) -> set[str]:
        """Load hosts that should never receive synced data from config."""
        try:
            # Try unified_loop.yaml first
            script_dir = Path(__file__).resolve().parent.parent.parent
            config_path = script_dir / "config" / "unified_loop.yaml"
            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                # Check auto_sync.exclude_hosts
                auto_sync = config.get("auto_sync", {})
                exclude = set(auto_sync.get("exclude_hosts", []))
                # Also check data_aggregation.excluded_nodes for compatibility
                data_agg = config.get("data_aggregation", {})
                exclude.update(data_agg.get("excluded_nodes", []))
                return exclude
        except Exception as e:
            logger.warning(f"Could not load exclude hosts from config: {e}")
        return set()

    def _load_known_games(self):
        """Load game IDs from local databases into memory."""
        game_ids = set()
        for db_path in self.data_dir.glob("*.db"):
            if "schema" in db_path.name:
                continue
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.execute("SELECT game_id FROM games")
                for row in cursor:
                    game_ids.add(row[0])
                conn.close()
            except Exception as e:
                logger.error(f"Error loading {db_path}: {e}")

        self.state.known_game_ids = game_ids
        logger.info(f"Loaded {len(game_ids):,} known game IDs")

    def _build_bloom_filter(self) -> BloomFilter:
        """Build a bloom filter of known game IDs."""
        bf = BloomFilter()
        for game_id in self.state.known_game_ids:
            bf.add(game_id)
        return bf

    async def start(self):
        """Start the gossip daemon."""
        self._running = True

        # Start TCP server for incoming gossip
        self._server = await asyncio.start_server(
            self._handle_connection,
            "0.0.0.0",
            self.listen_port,
        )
        logger.info(f"Server listening on port {self.listen_port}")

        # Start background sync loop
        asyncio.create_task(self._sync_loop())

        logger.info(f"Daemon started for {self.node_id}")

    async def stop(self):
        """Stop the gossip daemon."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        logger.info("Daemon stopped")

    async def _sync_loop(self):
        """Main sync loop - periodically sync with peers."""
        while self._running:
            try:
                await self._sync_cycle()
                self.state.sync_cycles += 1
            except Exception as e:
                logger.error(f"Sync cycle error: {e}")

            await asyncio.sleep(SYNC_INTERVAL)

    async def _sync_cycle(self):
        """Execute one sync cycle with peers."""
        # Select random subset of peers to sync with
        active_peers = [p for p in self.state.peers.values() if p.is_healthy]
        if not active_peers:
            return

        # Sync with 2-3 random peers
        sync_peers = random.sample(active_peers, min(3, len(active_peers)))

        for peer in sync_peers:
            try:
                await self._sync_with_peer(peer)
                peer.last_sync = time.time()
            except Exception as e:
                logger.warning(f"Failed to sync with {peer.name}: {e}")
                peer.is_healthy = False

        self.state.last_sync_time = time.time()

    async def _sync_with_peer(self, peer: GossipPeer):
        """Sync data with a specific peer."""
        # Connect to peer
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(peer.host, peer.port),
                timeout=10,
            )
        except Exception as e:
            raise ConnectionError(f"Cannot connect to {peer.name}: {e}")

        try:
            # Phase 1: Exchange bloom filters
            my_bloom = self._build_bloom_filter()
            await self._send_message(writer, {
                "type": "bloom_filter",
                "node_id": self.node_id,
                "bloom": my_bloom.to_bytes().hex(),
                "game_count": len(self.state.known_game_ids),
            })

            # Receive peer's bloom filter
            response = await self._recv_message(reader)
            if response["type"] != "bloom_filter":
                return

            peer_bloom = BloomFilter.from_bytes(bytes.fromhex(response["bloom"]))
            peer.last_seen = time.time()
            peer.is_healthy = True

            # Phase 2: Find games peer doesn't have
            games_to_push = []
            for game_id in self.state.known_game_ids:
                if game_id not in peer_bloom and len(games_to_push) < MAX_GAMES_PER_PUSH:
                    games_to_push.append(game_id)

            if games_to_push:
                # Push games peer doesn't have
                games_data = self._get_games_data(games_to_push)
                await self._send_message(writer, {
                    "type": "push_games",
                    "games": games_data,
                })
                self.state.total_games_pushed += len(games_data)
                logger.debug(f"Pushed {len(games_data)} games to {peer.name}")

            # Phase 3: Find games we don't have
            games_to_pull = []
            for game_id in self._get_peer_game_ids(response.get("sample_ids", [])):
                if game_id not in self.state.known_game_ids:
                    games_to_pull.append(game_id)

            if games_to_pull:
                # Request games from peer
                await self._send_message(writer, {
                    "type": "pull_games",
                    "game_ids": games_to_pull[:MAX_GAMES_PER_PUSH],
                })

                # Receive games
                pull_response = await self._recv_message(reader)
                if pull_response["type"] == "games_data":
                    received = self._store_games(pull_response["games"])
                    self.state.total_games_pulled += received
                    logger.debug(f"Pulled {received} games from {peer.name}")

            peer.games_synced += len(games_to_push)

        finally:
            writer.close()
            await writer.wait_closed()

    def _get_games_data(self, game_ids: list[str]) -> list[dict]:
        """Get game data for specified IDs from local databases."""
        games = []
        for db_path in self.data_dir.glob("*.db"):
            if "schema" in db_path.name:
                continue
            try:
                conn = sqlite3.connect(db_path)
                placeholders = ",".join(["?" for _ in game_ids])
                cursor = conn.execute(
                    f"SELECT * FROM games WHERE game_id IN ({placeholders})",
                    game_ids
                )
                columns = [desc[0] for desc in cursor.description]
                for row in cursor:
                    games.append(dict(zip(columns, row, strict=False)))
                conn.close()
            except (sqlite3.Error, OSError) as e:
                logger.debug(f"[GossipSync] Failed to fetch games from {db_path}: {e}")
        return games

    def _store_games(self, games: list[dict]) -> int:
        """Store received games in local database with transaction isolation.

        CRITICAL: Only stores games that have move data to prevent orphan games.
        """
        if not games:
            return 0

        # Filter games to only include those with move data
        valid_games, skipped = self._filter_games_with_moves(games)
        if skipped > 0:
            logger.debug(f"[GossipSync] Skipped {skipped} games without move data")
        if not valid_games:
            return 0

        # Store in a gossip-specific database
        gossip_db = self.data_dir / "gossip_received.db"
        conn = sqlite3.connect(gossip_db, isolation_level="IMMEDIATE")

        try:
            # Create table if needed (use first game's columns)
            columns = list(valid_games[0].keys())
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS games (
                    {', '.join(f'{col} TEXT' for col in columns)},
                    UNIQUE(game_id)
                )
            """)
            conn.commit()  # DDL in separate transaction

            # Begin transaction for data writes
            conn.execute("BEGIN IMMEDIATE")
            stored = 0
            stored_game_ids: list[str] = []

            # Use executemany for better performance and atomicity
            placeholders = ",".join(["?" for _ in columns])
            for game in valid_games:
                try:
                    conn.execute(
                        f"INSERT OR IGNORE INTO games ({','.join(columns)}) VALUES ({placeholders})",
                        [game.get(col) for col in columns]
                    )
                    if game_id := game.get("game_id"):
                        stored_game_ids.append(game_id)
                    stored += 1
                except (sqlite3.Error, KeyError, TypeError) as e:
                    logger.debug(f"[GossipSync] Failed to store game: {e}")

            conn.commit()

            # Only update known_game_ids after successful commit
            for game_id in stored_game_ids:
                self.state.known_game_ids.add(game_id)

            return stored

        except sqlite3.Error as e:
            logger.warning(f"[GossipSync] Transaction failed, rolling back: {e}")
            try:
                conn.rollback()
            except sqlite3.Error as rollback_err:
                logger.debug(f"[GossipSync] Rollback also failed: {rollback_err}")
            return 0
        finally:
            conn.close()

    def _filter_games_with_moves(
        self,
        games: list[dict],
    ) -> tuple[list[dict], int]:
        """Filter games to only include those with move data.

        Args:
            games: List of game dicts from gossip sync.

        Returns:
            Tuple of (valid_games, skipped_count).
            valid_games: Games that have move data.
            skipped_count: Number of games without move data that were skipped.
        """
        from app.db.move_data_validator import MIN_MOVES_REQUIRED

        valid_games = []
        skipped = 0

        for game in games:
            # Check for moves in various fields used by different sync protocols
            moves = game.get("moves") or game.get("moves_json") or game.get("canonical_history")

            if moves is None:
                skipped += 1
                continue

            # Parse moves if it's a JSON string
            if isinstance(moves, str):
                if not moves or moves == "[]" or moves == "null":
                    skipped += 1
                    continue
                try:
                    moves = json.loads(moves)
                except json.JSONDecodeError:
                    # Invalid JSON - skip this game
                    skipped += 1
                    continue

            # Check if moves is a list with sufficient entries
            if not isinstance(moves, list):
                skipped += 1
                continue

            if len(moves) < MIN_MOVES_REQUIRED:
                skipped += 1
                continue

            valid_games.append(game)

        return valid_games, skipped

    def _get_peer_game_ids(self, sample: list[str]) -> list[str]:
        """Get game IDs that peer might have but we don't."""
        # In a full implementation, this would be more sophisticated
        return sample

    async def _send_message(self, writer: asyncio.StreamWriter, message: dict):
        """Send a JSON message with length prefix."""
        data = json.dumps(message).encode()
        writer.write(struct.pack(">I", len(data)))
        writer.write(data)
        await writer.drain()

    async def _recv_message(self, reader: asyncio.StreamReader) -> dict:
        """Receive a JSON message with length prefix."""
        length_data = await asyncio.wait_for(reader.readexactly(4), timeout=30)
        length = struct.unpack(">I", length_data)[0]
        data = await asyncio.wait_for(reader.readexactly(length), timeout=60)
        return json.loads(data.decode())

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter
    ):
        """Handle incoming gossip connection."""
        try:
            while True:
                try:
                    message = await self._recv_message(reader)
                except asyncio.TimeoutError:
                    break

                if message["type"] == "bloom_filter":
                    # Respond with our bloom filter
                    my_bloom = self._build_bloom_filter()
                    # Include sample of game IDs for pull requests
                    sample = random.sample(
                        list(self.state.known_game_ids),
                        min(100, len(self.state.known_game_ids))
                    ) if self.state.known_game_ids else []

                    await self._send_message(writer, {
                        "type": "bloom_filter",
                        "node_id": self.node_id,
                        "bloom": my_bloom.to_bytes().hex(),
                        "game_count": len(self.state.known_game_ids),
                        "sample_ids": sample,
                    })

                elif message["type"] == "push_games":
                    # Store received games
                    stored = self._store_games(message["games"])
                    self.state.total_games_pulled += stored

                elif message["type"] == "pull_games":
                    # Send requested games
                    games_data = self._get_games_data(message["game_ids"])
                    await self._send_message(writer, {
                        "type": "games_data",
                        "games": games_data,
                    })

        except Exception as e:
            logger.error(f"Connection handler error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    def health_check(self) -> "HealthCheckResult":
        """Check daemon health status.

        December 2025: Added to satisfy CoordinatorProtocol for unified health monitoring.

        Returns:
            HealthCheckResult with status and details
        """
        try:
            from app.coordination.protocols import HealthCheckResult, CoordinatorStatus
        except ImportError:
            # Fallback if protocols not available
            return {"healthy": self._running, "message": "GossipSyncDaemon running" if self._running else "not running"}

        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="GossipSyncDaemon not running",
            )

        # Check if sync cycles are happening
        now = time.time()
        if self.state.last_sync_time > 0:
            time_since_sync = now - self.state.last_sync_time
            # Warning if no sync in 3x the interval
            if time_since_sync > SYNC_INTERVAL * 3:
                return HealthCheckResult(
                    healthy=False,
                    status=CoordinatorStatus.DEGRADED,
                    message=f"GossipSyncDaemon sync stale ({time_since_sync:.0f}s since last sync)",
                    details=self.get_status(),
                )

        # Check peer health
        healthy_peers = sum(1 for p in self.state.peers.values() if p.is_healthy)
        total_peers = len(self.state.peers)
        if total_peers > 0 and healthy_peers == 0:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"No healthy peers ({total_peers} peers all unhealthy)",
                details=self.get_status(),
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"GossipSyncDaemon running ({self.state.sync_cycles} cycles, {healthy_peers}/{total_peers} healthy peers)",
            details=self.get_status(),
        )

    def get_status(self) -> dict[str, Any]:
        """Get current sync status."""
        return {
            "node_id": self.node_id,
            "known_games": len(self.state.known_game_ids),
            "total_pushed": self.state.total_games_pushed,
            "total_pulled": self.state.total_games_pulled,
            "sync_cycles": self.state.sync_cycles,
            "last_sync": self.state.last_sync_time,
            "peers": {
                name: {
                    "host": p.host,
                    "is_healthy": p.is_healthy,
                    "last_seen": p.last_seen,
                    "games_synced": p.games_synced,
                }
                for name, p in self.state.peers.items()
            }
        }


def _load_peers_from_distributed_hosts(data: dict) -> dict[str, dict]:
    peers = {}
    hosts = data.get("hosts", {})
    for name, config in hosts.items():
        if not config.get("p2p_enabled", True):
            continue
        status = str(config.get("status", "")).lower()
        if status in {"terminated", "offline", "suspended"}:
            continue
        ssh_host = config.get("tailscale_ip") or config.get("ssh_host", "")
        if not ssh_host:
            continue
        peers[name] = {
            "ssh_host": ssh_host,
            "ssh_user": config.get("ssh_user", "ubuntu"),
            "ssh_port": config.get("ssh_port", 22),
            "gossip_host": ssh_host,
            "gossip_port": GOSSIP_PORT,
        }
    return peers


def _load_peers_from_remote_hosts(data: dict) -> dict[str, dict]:
    peers = {}
    # Use standard_hosts as peers (legacy schema)
    for name, config in data.get("standard_hosts", {}).items():
        # Skip Tailscale duplicates
        if name.startswith("lambda_gh200"):
            continue
        peers[name] = {
            "ssh_host": config.get("ssh_host", ""),
            "ssh_user": config.get("ssh_user", "ubuntu"),
            "ssh_port": config.get("ssh_port", 22),
            "gossip_host": config.get("ssh_host", ""),
            "gossip_port": GOSSIP_PORT,
        }
    return peers


def load_peer_config(config_path: Path) -> dict[str, dict]:
    """Load peer configuration from distributed_hosts.yaml (legacy: remote_hosts.yaml)."""
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    if "hosts" in data:
        return _load_peers_from_distributed_hosts(data)
    return _load_peers_from_remote_hosts(data)


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="P2P Gossip Data Sync")
    parser.add_argument("--start", action="store_true", help="Start gossip daemon")
    parser.add_argument("--status", action="store_true", help="Show sync status")
    parser.add_argument("--node-id", type=str, help="Node ID (defaults to hostname)")
    parser.add_argument("--data-dir", type=str, default="data/games",
                       help="Data directory")
    parser.add_argument("--port", type=int, default=GOSSIP_PORT,
                       help="Listen port")
    args = parser.parse_args()

    # Determine node ID
    node_id = args.node_id or socket.gethostname()

    # Find config
    script_dir = Path(__file__).resolve().parent.parent.parent
    config_path = script_dir / "config" / "distributed_hosts.yaml"
    if not config_path.exists():
        config_path = script_dir / "config" / "remote_hosts.yaml"
    data_dir = script_dir / args.data_dir

    if args.status:
        # Just show what would be synced
        peers = load_peer_config(config_path)
        print(f"Node: {node_id}")
        print(f"Data dir: {data_dir}")
        print(f"Configured peers: {len(peers)}")
        for name, config in list(peers.items())[:5]:
            print(f"  - {name}: {config['ssh_host']}")
        if len(peers) > 5:
            print(f"  ... and {len(peers) - 5} more")
        return

    if args.start:
        peers = load_peer_config(config_path)
        daemon = GossipSyncDaemon(
            node_id=node_id,
            data_dir=data_dir,
            peers_config=peers,
            listen_port=args.port,
        )
        await daemon.start()

        # Keep running
        try:
            while True:
                await asyncio.sleep(60)
                status = daemon.get_status()
                print(f"[Gossip] Status: {status['known_games']:,} games, "
                      f"{status['sync_cycles']} cycles, "
                      f"{status['total_pushed']} pushed, {status['total_pulled']} pulled")
        except KeyboardInterrupt:
            await daemon.stop()


if __name__ == "__main__":
    asyncio.run(main())
