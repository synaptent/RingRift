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
import random
import socket
import sqlite3
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Constants
GOSSIP_PORT = 8771  # Port for gossip protocol
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


class BloomFilter:
    """Simple bloom filter for efficient set membership testing."""

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
            print(f"[Gossip] Excluding hosts from sync: {self.exclude_hosts}")

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
            print(f"[Gossip] Warning: Could not load exclude hosts from config: {e}")
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
                print(f"[Gossip] Error loading {db_path}: {e}")

        self.state.known_game_ids = game_ids
        print(f"[Gossip] Loaded {len(game_ids):,} known game IDs")

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
        print(f"[Gossip] Server listening on port {self.listen_port}")

        # Start background sync loop
        asyncio.create_task(self._sync_loop())

        print(f"[Gossip] Daemon started for {self.node_id}")

    async def stop(self):
        """Stop the gossip daemon."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        print("[Gossip] Daemon stopped")

    async def _sync_loop(self):
        """Main sync loop - periodically sync with peers."""
        while self._running:
            try:
                await self._sync_cycle()
                self.state.sync_cycles += 1
            except Exception as e:
                print(f"[Gossip] Sync cycle error: {e}")

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
                print(f"[Gossip] Failed to sync with {peer.name}: {e}")
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
                print(f"[Gossip] Pushed {len(games_data)} games to {peer.name}")

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
                    print(f"[Gossip] Pulled {received} games from {peer.name}")

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
            except Exception:
                pass
        return games

    def _store_games(self, games: list[dict]) -> int:
        """Store received games in local database."""
        if not games:
            return 0

        # Store in a gossip-specific database
        gossip_db = self.data_dir / "gossip_received.db"
        conn = sqlite3.connect(gossip_db)

        # Create table if needed (use first game's columns)
        columns = list(games[0].keys())
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS games (
                {', '.join(f'{col} TEXT' for col in columns)},
                UNIQUE(game_id)
            )
        """)

        stored = 0
        for game in games:
            try:
                placeholders = ",".join(["?" for _ in columns])
                conn.execute(
                    f"INSERT OR IGNORE INTO games ({','.join(columns)}) VALUES ({placeholders})",
                    [game.get(col) for col in columns]
                )
                if game.get("game_id"):
                    self.state.known_game_ids.add(game["game_id"])
                stored += 1
            except Exception:
                pass

        conn.commit()
        conn.close()
        return stored

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
            print(f"[Gossip] Connection handler error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

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


def load_peer_config(config_path: Path) -> dict[str, dict]:
    """Load peer configuration from remote_hosts.yaml."""
    with open(config_path) as f:
        data = yaml.safe_load(f)

    peers = {}

    # Use standard_hosts as peers
    if "standard_hosts" in data:
        for name, config in data["standard_hosts"].items():
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
