"""
Elo Database Synchronization Manager

Keeps unified_elo.db consistent across all cluster nodes using multiple transport methods:
- Tailscale direct connections (primary - mesh network, NAT traversal)
- SSH via public endpoints (fallback for Vast.ai instances)
- HTTP via Cloudflare Zero Trust (NAT traversal, works through firewalls)
- aria2 multi-source downloads (bulk transfers, resumable)

Resilience features:
- Circuit breakers per node (exponential backoff on failures)
- Merge-based conflict resolution (preserves all matches by game_id)
- Local WAL queue for offline sync
- Gossip-based peer discovery

Integrates with:
- P2P Orchestrator (sync after game batches)
- Training Loop (sync before/after training)
- Tournament Scripts (sync after each round)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sqlite3
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from app.utils.checksum_utils import compute_string_checksum

# Optional async libraries (may not be available on all nodes)
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    aiohttp = None

try:
    import aiofiles
    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False
    aiofiles = None

logger = logging.getLogger(__name__)

# Use canonical circuit breaker from distributed module
import contextlib

from app.distributed.circuit_breaker import (
    CircuitBreaker as CanonicalCircuitBreaker,
)

# Default paths
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "unified_elo.db"
SYNC_STATE_PATH = Path(__file__).parent.parent.parent / "data" / "elo_sync_state.json"


@dataclass
class EloManagerSyncState:
    """Tracks Elo manager sync state for consistency checking.

    Note: This is Elo manager-specific state tracking.
    For sync operation states (PENDING, IN_PROGRESS, etc.), use sync_constants.SyncState.
    """
    last_sync_timestamp: float = 0
    last_sync_hash: str = ""
    local_match_count: int = 0
    synced_from: str = ""
    sync_errors: list[str] = field(default_factory=list)
    # Enhanced tracking
    pending_matches: list[dict] = field(default_factory=list)  # WAL for offline sync
    merge_conflicts: int = 0
    total_syncs: int = 0
    successful_syncs: int = 0


@dataclass
class NodeInfo:
    """Information about a cluster node for syncing."""
    name: str
    tailscale_ip: str | None = None
    ssh_host: str | None = None
    ssh_port: int = 22
    http_url: str | None = None
    cloudflare_tunnel: str | None = None  # Cloudflare Zero Trust tunnel URL
    aria2_url: str | None = None  # aria2 RPC endpoint
    is_coordinator: bool = False
    last_seen: float = 0
    match_count: int = 0
    db_hash: str = ""
    # Vast.ai specific
    vast_instance_id: str | None = None
    vast_ssh_host: str | None = None  # e.g., ssh7.vast.ai
    vast_ssh_port: int | None = None  # e.g., 14398


class EloSyncManager:
    """
    Manages Elo database synchronization across cluster nodes.

    Features:
    - Multi-transport failover (Tailscale -> SSH -> Cloudflare -> aria2 -> HTTP)
    - Circuit breakers per node for fault tolerance
    - Merge-based conflict resolution (preserves all unique matches)
    - Local WAL queue for offline sync
    - Integration with P2P orchestrator and training loop

    Usage:
        sync_manager = EloSyncManager(db_path=Path("data/unified_elo.db"))
        await sync_manager.initialize()

        # After playing games:
        await sync_manager.push_new_matches(new_matches)

        # Periodic sync:
        await sync_manager.sync_with_cluster()

        # Before training:
        await sync_manager.ensure_latest()
    """

    # Known Vast.ai instances for auto-discovery
    VAST_INSTANCES = {
        "4xRTX5090": {"host": "ssh7.vast.ai", "port": 14398},
        "2xRTX3060Ti": {"host": "ssh8.vast.ai", "port": 17016},
        "RTX4060Ti": {"host": "ssh1.vast.ai", "port": 14400},
        "RTX4060Ti-b": {"host": "ssh2.vast.ai", "port": 19768},
        "RTX3060Ti": {"host": "ssh3.vast.ai", "port": 19766},
        "4xRTX3060": {"host": "ssh3.vast.ai", "port": 38740},
        "A40": {"host": "ssh8.vast.ai", "port": 38742},
        "2xRTX4080S": {"host": "ssh3.vast.ai", "port": 19940},
        "RTX5080": {"host": "ssh1.vast.ai", "port": 19942},
    }

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        coordinator_host: str = "lambda-h100",
        sync_interval: int = 300,
        p2p_url: str | None = None,
        enable_merge: bool = True,  # Use merge instead of replace
    ):
        self.db_path = Path(db_path)
        self.coordinator_host = coordinator_host
        self.sync_interval = sync_interval
        self.p2p_url = p2p_url or os.environ.get("P2P_URL", "https://p2p.ringrift.ai")
        self.enable_merge = enable_merge

        self.state = EloManagerSyncState()
        self.nodes: dict[str, NodeInfo] = {}
        # Use single canonical circuit breaker for all nodes (tracks targets internally)
        self._circuit_breaker = CanonicalCircuitBreaker(
            failure_threshold=3,
            recovery_timeout=60.0,
            backoff_multiplier=2.0,
            max_backoff=3600.0,  # Max 1 hour
            operation_type="elo_sync",
        )
        self._sync_lock = asyncio.Lock()
        self._running = False
        self._sync_task: asyncio.Task | None = None

        # Transport methods in priority order (most reliable first)
        self.transport_methods = [
            ("tailscale", self._sync_via_tailscale),
            ("vast_ssh", self._sync_via_vast_ssh),
            ("ssh", self._sync_via_ssh),
            ("cloudflare", self._sync_via_cloudflare),
            ("aria2", self._sync_via_aria2),
            ("http", self._sync_via_http),
        ]

        # Event callbacks for integration
        self._on_sync_complete: list[Callable] = []
        self._on_sync_failed: list[Callable] = []

    async def initialize(self):
        """Initialize the sync manager."""
        self._load_state()
        await self._discover_nodes()
        self._update_local_stats()
        logger.info(f"EloSyncManager initialized: {self.state.local_match_count} local matches")

    def _load_state(self):
        """Load sync state from disk."""
        if SYNC_STATE_PATH.exists():
            try:
                with open(SYNC_STATE_PATH) as f:
                    data = json.load(f)
                    self.state = EloManagerSyncState(**data)
            except Exception as e:
                logger.warning(f"Failed to load sync state: {e}")

    def _save_state(self):
        """Save sync state to disk."""
        try:
            SYNC_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(SYNC_STATE_PATH, 'w') as f:
                json.dump({
                    'last_sync_timestamp': self.state.last_sync_timestamp,
                    'last_sync_hash': self.state.last_sync_hash,
                    'local_match_count': self.state.local_match_count,
                    'synced_from': self.state.synced_from,
                    'sync_errors': self.state.sync_errors[-10:],  # Keep last 10 errors
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save sync state: {e}")

    def _update_local_stats(self):
        """Update local database statistics."""
        if not self.db_path.exists():
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM match_history")
            self.state.local_match_count = cursor.fetchone()[0]

            # Calculate hash for change detection
            cursor.execute("SELECT COUNT(*), MAX(timestamp) FROM match_history")
            count, max_ts = cursor.fetchone()
            self.state.last_sync_hash = compute_string_checksum(
                f"{count}:{max_ts}", algorithm="md5"
            )

            conn.close()
        except Exception as e:
            logger.warning(f"Failed to update local stats: {e}")

    async def _discover_nodes(self):
        """Discover cluster nodes from P2P coordinator."""
        if not HAS_AIOHTTP:
            return
        try:
            async with aiohttp.ClientSession() as session, session.get(
                f"{self.p2p_url}/status",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    peers = data.get('peers', {})
                    # peers is a dict {node_id: info}, iterate over values
                    peer_list = peers.values() if isinstance(peers, dict) else peers
                    for peer in peer_list:
                        if isinstance(peer, str):
                            # Handle case where peer is just a node ID string
                            name = peer
                            peer = {}
                        else:
                            name = peer.get('name', peer.get('node_id', 'unknown'))
                        self.nodes[name] = NodeInfo(
                            name=name,
                            tailscale_ip=peer.get('tailscale_ip'),
                            ssh_host=peer.get('ssh_host'),
                            http_url=peer.get('http_url'),
                            is_coordinator=(name == self.coordinator_host),
                            last_seen=time.time()
                        )
                    logger.info(f"Discovered {len(self.nodes)} nodes from P2P coordinator")
        except Exception as e:
            logger.warning(f"Failed to discover nodes from P2P: {e}")
            # Add default coordinator
            self.nodes[self.coordinator_host] = NodeInfo(
                name=self.coordinator_host,
                ssh_host=self.coordinator_host,
                is_coordinator=True
            )

    async def start_background_sync(self):
        """Start background sync loop."""
        if self._running:
            return

        self._running = True
        self._sync_task = asyncio.create_task(self._sync_loop())
        logger.info(f"Started background Elo sync (interval: {self.sync_interval}s)")

    async def stop_background_sync(self):
        """Stop background sync loop."""
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._sync_task
        logger.info("Stopped background Elo sync")

    async def _sync_loop(self):
        """Background sync loop."""
        while self._running:
            try:
                await self.sync_with_cluster()
            except Exception as e:
                logger.error(f"Sync loop error: {e}")
                self.state.sync_errors.append(f"{datetime.now().isoformat()}: {e}")

            await asyncio.sleep(self.sync_interval)

    def on_sync_complete(self, callback: Callable):
        """Register callback for successful sync events."""
        self._on_sync_complete.append(callback)

    def on_sync_failed(self, callback: Callable):
        """Register callback for failed sync events."""
        self._on_sync_failed.append(callback)

    async def _notify_sync_complete(self, node_name: str, method: str, matches: int):
        """Notify listeners of successful sync."""
        for callback in self._on_sync_complete:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(node_name, method, matches)
                else:
                    callback(node_name, method, matches)
            except Exception as e:
                logger.warning(f"Sync complete callback failed: {e}")

    async def _notify_sync_failed(self, errors: list[str]):
        """Notify listeners of failed sync."""
        for callback in self._on_sync_failed:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(errors)
                else:
                    callback(errors)
            except Exception as e:
                logger.warning(f"Sync failed callback failed: {e}")

    async def sync_with_cluster(self) -> bool:
        """
        Synchronize with cluster nodes using multi-transport failover.
        Returns True if sync was successful with any node.
        """
        async with self._sync_lock:
            logger.info("Starting cluster sync...")
            self.state.total_syncs += 1
            errors = []

            # Build ordered list of nodes to try (coordinator first)
            nodes_to_try = []
            coordinator = self.nodes.get(self.coordinator_host)
            if coordinator:
                nodes_to_try.append(coordinator)
            for node in self.nodes.values():
                if node.name != self.coordinator_host:
                    nodes_to_try.append(node)

            # Also add known Vast instances if not already discovered
            for name, info in self.VAST_INSTANCES.items():
                if name not in self.nodes:
                    nodes_to_try.append(NodeInfo(
                        name=name,
                        vast_ssh_host=info["host"],
                        vast_ssh_port=info["port"]
                    ))

            # Try each node with each transport method
            for node in nodes_to_try:
                # Check circuit breaker (canonical version uses target-based tracking)
                if not self._circuit_breaker.can_execute(node.name):
                    logger.debug(f"Skipping {node.name} - circuit open")
                    continue

                for method_name, method in self.transport_methods:
                    try:
                        success = await method(node)
                        if success:
                            self._circuit_breaker.record_success(node.name)
                            self.state.last_sync_timestamp = time.time()
                            self.state.synced_from = node.name
                            self.state.successful_syncs += 1
                            self._update_local_stats()
                            self._save_state()
                            logger.info(f"Synced with {node.name} via {method_name}")
                            await self._notify_sync_complete(
                                node.name, method_name, self.state.local_match_count
                            )
                            return True
                    except Exception as e:
                        error_msg = f"{node.name}/{method_name}: {e}"
                        errors.append(error_msg)
                        logger.debug(f"Sync method failed: {error_msg}")
                        continue

                # All methods failed for this node
                self._circuit_breaker.record_failure(node.name)

            # All nodes failed
            logger.warning(f"All sync methods failed. Errors: {errors[:5]}")
            self.state.sync_errors.extend(errors[:5])
            self._save_state()
            await self._notify_sync_failed(errors)
            return False

    async def _sync_via_tailscale(self, node: NodeInfo) -> bool:
        """Sync via Tailscale direct connection using rsync."""
        if not node.tailscale_ip and not node.ssh_host:
            return False

        host = node.tailscale_ip or node.ssh_host

        # First, get remote stats
        result = await asyncio.create_subprocess_exec(
            'ssh', '-o', 'ConnectTimeout=5', '-o', 'StrictHostKeyChecking=no',
            host,
            "sqlite3 ~/ringrift/ai-service/data/unified_elo.db 'SELECT COUNT(*) FROM match_history' 2>/dev/null",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await asyncio.wait_for(result.communicate(), timeout=15)

        if result.returncode != 0:
            return False

        try:
            remote_count = int(stdout.decode().strip())
        except (ValueError, AttributeError):
            return False

        # Compare counts - sync from node with more data
        if remote_count > self.state.local_match_count:
            # Pull from remote
            return await self._rsync_pull(host)
        elif remote_count < self.state.local_match_count:
            # Push to remote
            return await self._rsync_push(host)
        else:
            # Same count, check hash for consistency
            logger.debug(f"Same match count with {node.name}, skipping sync")
            return True

    async def _rsync_pull(self, host: str) -> bool:
        """Pull database from remote host."""
        temp_path = self.db_path.with_suffix('.db.tmp')

        result = await asyncio.create_subprocess_exec(
            'rsync', '-az', '--timeout=60',
            f'{host}:~/ringrift/ai-service/data/unified_elo.db',
            str(temp_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        _, _stderr = await asyncio.wait_for(result.communicate(), timeout=180)

        if result.returncode == 0 and temp_path.exists():
            # Verify the downloaded database
            try:
                conn = sqlite3.connect(temp_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM match_history")
                count = cursor.fetchone()[0]
                conn.close()

                # Use merge instead of replace to preserve local data
                if self.enable_merge:
                    success = await self._merge_databases(temp_path)
                    if success:
                        logger.info(f"Merged {count} matches from {host}")
                    return success
                else:
                    shutil.move(temp_path, self.db_path)
                    logger.info(f"Pulled {count} matches from {host}")
                    return True
            except Exception as e:
                logger.error(f"Database verification failed: {e}")
                temp_path.unlink(missing_ok=True)
                return False

        return False

    async def _rsync_push(self, host: str) -> bool:
        """Push database to remote host."""
        result = await asyncio.create_subprocess_exec(
            'rsync', '-az', '--timeout=60',
            str(self.db_path),
            f'{host}:~/ringrift/ai-service/data/unified_elo.db',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await asyncio.wait_for(result.communicate(), timeout=180)

        if result.returncode == 0:
            logger.info(f"Pushed {self.state.local_match_count} matches to {host}")
            return True
        return False

    async def _sync_via_ssh(self, node: NodeInfo) -> bool:
        """Sync via SSH (fallback for Tailscale)."""
        host = node.ssh_host or node.tailscale_ip
        if not host:
            return False
        return await self._sync_via_tailscale(node)  # Same logic

    async def _sync_via_vast_ssh(self, node: NodeInfo) -> bool:
        """Sync via Vast.ai SSH endpoint (handles NAT via their jump host)."""
        host = node.vast_ssh_host
        port = node.vast_ssh_port
        if not host or not port:
            return False

        # Use specific port for Vast SSH
        remote_db = "~/ringrift/ai-service/data/unified_elo.db"

        # Get remote match count
        result = await asyncio.create_subprocess_exec(
            'ssh', '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=no',
            '-o', 'BatchMode=yes', '-p', str(port), f'root@{host}',
            f"sqlite3 {remote_db} 'SELECT COUNT(*) FROM match_history' 2>/dev/null || echo 0",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, _ = await asyncio.wait_for(result.communicate(), timeout=20)
        except asyncio.TimeoutError:
            return False

        if result.returncode != 0:
            return False

        try:
            remote_count = int(stdout.decode().strip().split('\n')[-1])
        except (ValueError, AttributeError, IndexError):
            return False

        # Sync logic
        if remote_count > self.state.local_match_count:
            return await self._rsync_pull_vast(host, port)
        elif remote_count < self.state.local_match_count:
            return await self._rsync_push_vast(host, port)
        return True

    async def _rsync_pull_vast(self, host: str, port: int) -> bool:
        """Pull from Vast.ai instance via SSH."""
        temp_path = self.db_path.with_suffix('.db.tmp')

        result = await asyncio.create_subprocess_exec(
            'rsync', '-az', '--timeout=60',
            '-e', f'ssh -o StrictHostKeyChecking=no -o BatchMode=yes -p {port}',
            f'root@{host}:~/ringrift/ai-service/data/unified_elo.db',
            str(temp_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        try:
            await asyncio.wait_for(result.communicate(), timeout=120)
        except asyncio.TimeoutError:
            temp_path.unlink(missing_ok=True)
            return False

        if result.returncode == 0 and temp_path.exists():
            if self.enable_merge:
                return await self._merge_databases(temp_path)
            else:
                shutil.move(temp_path, self.db_path)
                return True
        return False

    async def _rsync_push_vast(self, host: str, port: int) -> bool:
        """Push to Vast.ai instance via SSH."""
        result = await asyncio.create_subprocess_exec(
            'rsync', '-az', '--timeout=60',
            '-e', f'ssh -o StrictHostKeyChecking=no -o BatchMode=yes -p {port}',
            str(self.db_path),
            f'root@{host}:~/ringrift/ai-service/data/unified_elo.db',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        try:
            await asyncio.wait_for(result.communicate(), timeout=120)
        except asyncio.TimeoutError:
            return False

        return result.returncode == 0

    async def _sync_via_cloudflare(self, node: NodeInfo) -> bool:
        """Sync via Cloudflare Zero Trust tunnel (works through NAT/firewalls)."""
        if not HAS_AIOHTTP or not HAS_AIOFILES:
            return False
        tunnel_url = node.cloudflare_tunnel
        if not tunnel_url:
            return False

        try:
            async with aiohttp.ClientSession() as session:
                # Cloudflare tunnel endpoint for ELO sync
                async with session.get(
                    f"{tunnel_url}/api/elo/status",
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    if resp.status != 200:
                        return False
                    status = await resp.json()
                    remote_count = status.get('match_count', 0)

                if remote_count > self.state.local_match_count:
                    async with session.get(
                        f"{tunnel_url}/api/elo/download",
                        timeout=aiohttp.ClientTimeout(total=180)
                    ) as resp:
                        if resp.status == 200:
                            temp_path = self.db_path.with_suffix('.db.tmp')
                            async with aiofiles.open(temp_path, 'wb') as f:
                                await f.write(await resp.read())
                            if self.enable_merge:
                                return await self._merge_databases(temp_path)
                            else:
                                shutil.move(temp_path, self.db_path)
                                return True

                elif remote_count < self.state.local_match_count:
                    async with aiofiles.open(self.db_path, 'rb') as f:
                        data = await f.read()
                    async with session.post(
                        f"{tunnel_url}/api/elo/upload",
                        data=data,
                        timeout=aiohttp.ClientTimeout(total=180)
                    ) as resp:
                        return resp.status == 200

                return True
        except Exception as e:
            logger.debug(f"Cloudflare sync failed: {e}")
            return False

    async def _sync_via_aria2(self, node: NodeInfo) -> bool:
        """Sync via aria2 for resumable bulk downloads."""
        aria2_url = node.aria2_url or node.http_url
        if not aria2_url:
            return False

        # Check if aria2c is available
        if not shutil.which('aria2c'):
            return False

        try:
            # Try to download via aria2 (supports resume, multi-connection)
            temp_path = self.db_path.with_suffix('.db.aria2.tmp')
            download_url = f"{aria2_url}/elo/unified_elo.db"

            result = await asyncio.create_subprocess_exec(
                'aria2c',
                '--max-connection-per-server=4',
                '--split=4',
                '--continue=true',
                '--timeout=60',
                '--max-tries=3',
                '--retry-wait=5',
                '-d', str(temp_path.parent),
                '-o', temp_path.name,
                download_url,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                await asyncio.wait_for(result.communicate(), timeout=180)
            except asyncio.TimeoutError:
                temp_path.unlink(missing_ok=True)
                return False

            if result.returncode == 0 and temp_path.exists():
                if self.enable_merge:
                    return await self._merge_databases(temp_path)
                else:
                    shutil.move(temp_path, self.db_path)
                    return True
            return False

        except Exception as e:
            logger.debug(f"aria2 sync failed: {e}")
            return False

    async def _merge_databases(self, remote_db_path: Path) -> bool:
        """
        Merge remote database into local, preserving all unique matches.
        Uses game_id for deduplication if available, otherwise match signature.

        After merging match_history, recalculates all elo_ratings from scratch
        to ensure win/loss conservation invariant is maintained.
        """
        if not remote_db_path.exists():
            return False

        try:
            # Open both databases
            local_conn = sqlite3.connect(self.db_path)
            remote_conn = sqlite3.connect(remote_db_path)

            local_cur = local_conn.cursor()
            remote_cur = remote_conn.cursor()

            # Get existing game_ids from local
            local_cur.execute("""
                SELECT COALESCE(game_id,
                    participant_a || '|' || participant_b || '|' || timestamp)
                FROM match_history
            """)
            existing_ids = {row[0] for row in local_cur.fetchall() if row[0]}

            # Get all columns from remote
            remote_cur.execute("PRAGMA table_info(match_history)")
            columns = [col[1] for col in remote_cur.fetchall()]

            # Fetch remote matches
            remote_cur.execute("SELECT * FROM match_history")
            remote_matches = remote_cur.fetchall()

            # Find new matches
            inserted = 0
            for match in remote_matches:
                match_dict = dict(zip(columns, match, strict=False))
                match_id = match_dict.get('game_id') or \
                    f"{match_dict.get('participant_a')}|{match_dict.get('participant_b')}|{match_dict.get('timestamp')}"

                if match_id not in existing_ids:
                    # Insert new match
                    cols = ', '.join(columns)
                    placeholders = ', '.join(['?' for _ in columns])
                    local_cur.execute(
                        f"INSERT OR IGNORE INTO match_history ({cols}) VALUES ({placeholders})",
                        match
                    )
                    if local_cur.rowcount > 0:
                        inserted += 1
                        existing_ids.add(match_id)

            local_conn.commit()
            remote_conn.close()

            # Cleanup temp file
            remote_db_path.unlink(missing_ok=True)

            if inserted > 0:
                logger.info(f"Merged {inserted} new matches from remote")
                # Recalculate ratings from merged match history
                await self._recalculate_ratings_from_history(local_conn)
            else:
                logger.debug("No new matches to merge")

            local_conn.close()
            return True

        except Exception as e:
            logger.error(f"Database merge failed: {e}")
            remote_db_path.unlink(missing_ok=True)
            return False

    async def _recalculate_ratings_from_history(self, conn: sqlite3.Connection) -> None:
        """
        Recalculate all ELO ratings from match history.

        This ensures win/loss conservation after merging databases.
        Replays all matches chronologically to rebuild accurate ratings.
        """
        from collections import defaultdict

        # ELO calculation constants
        INITIAL_RATING = 1500.0
        K_FACTOR = 32.0

        # Pinned baselines (anchors to prevent ELO inflation)
        PINNED_BASELINES = {
            "baseline_random": 400.0,
        }

        def get_pinned_rating(participant_id: str):
            for prefix, rating in PINNED_BASELINES.items():
                if participant_id.startswith(prefix):
                    return rating
            return None

        def expected_score(rating_a: float, rating_b: float) -> float:
            return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

        cur = conn.cursor()

        logger.info("Recalculating ratings from match history...")

        # Get all matches ordered by timestamp
        cur.execute("""
            SELECT participant_a, participant_b, winner, board_type, num_players, timestamp
            FROM match_history
            WHERE winner IS NOT NULL
            ORDER BY timestamp
        """)
        matches = cur.fetchall()

        # Initialize ratings storage: (board_type, num_players, participant_id) -> rating_data
        ratings = defaultdict(lambda: {
            "rating": INITIAL_RATING,
            "games_played": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
        })

        # Replay all matches
        for p_a, p_b, winner, board_type, num_players, _ts in matches:
            key_a = (board_type, num_players, p_a)
            key_b = (board_type, num_players, p_b)

            r_a = ratings[key_a]
            r_b = ratings[key_b]

            # Calculate expected scores
            exp_a = expected_score(r_a["rating"], r_b["rating"])
            exp_b = 1.0 - exp_a

            # Determine actual scores
            if winner == p_a:
                score_a, score_b = 1.0, 0.0
                r_a["wins"] += 1
                r_b["losses"] += 1
            elif winner == p_b:
                score_a, score_b = 0.0, 1.0
                r_a["losses"] += 1
                r_b["wins"] += 1
            elif winner == "draw":
                score_a, score_b = 0.5, 0.5
                r_a["draws"] += 1
                r_b["draws"] += 1
            else:
                continue

            # Update ratings (unless pinned)
            pinned_a = get_pinned_rating(p_a)
            pinned_b = get_pinned_rating(p_b)

            if pinned_a is None:
                r_a["rating"] += K_FACTOR * (score_a - exp_a)
            else:
                r_a["rating"] = pinned_a

            if pinned_b is None:
                r_b["rating"] += K_FACTOR * (score_b - exp_b)
            else:
                r_b["rating"] = pinned_b

            r_a["games_played"] += 1
            r_b["games_played"] += 1

        # Clear existing ratings
        cur.execute("DELETE FROM elo_ratings")

        # Insert recalculated ratings
        now = time.time()
        for (board_type, num_players, participant_id), data in ratings.items():
            cur.execute("""
                INSERT INTO elo_ratings
                (participant_id, board_type, num_players, rating, games_played,
                 wins, losses, draws, rating_deviation, last_update)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                participant_id, board_type, num_players,
                data["rating"], data["games_played"],
                data["wins"], data["losses"], data["draws"],
                350.0,  # Initial rating deviation
                now,
            ))

        conn.commit()
        logger.info(f"Recalculated {len(ratings)} ratings from {len(matches)} matches")

    async def _sync_via_http(self, node: NodeInfo) -> bool:
        """Sync via HTTP (Cloudflare Zero Trust compatible)."""
        if not HAS_AIOHTTP or not HAS_AIOFILES:
            return False
        if not node.http_url:
            return False

        try:
            async with aiohttp.ClientSession() as session:
                # Get remote status
                async with session.get(
                    f"{node.http_url}/elo/status",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status != 200:
                        return False
                    status = await resp.json()
                    remote_count = status.get('match_count', 0)

                if remote_count > self.state.local_match_count:
                    # Download database
                    async with session.get(
                        f"{node.http_url}/elo/db",
                        timeout=aiohttp.ClientTimeout(total=120)
                    ) as resp:
                        if resp.status == 200:
                            temp_path = self.db_path.with_suffix('.db.tmp')
                            async with aiofiles.open(temp_path, 'wb') as f:
                                await f.write(await resp.read())
                            # Use merge instead of replace to preserve local data
                            if self.enable_merge:
                                success = await self._merge_databases(temp_path)
                                if success:
                                    logger.info(f"Merged {remote_count} matches via HTTP")
                                return success
                            else:
                                shutil.move(temp_path, self.db_path)
                                logger.info(f"Downloaded {remote_count} matches via HTTP")
                                return True

                elif remote_count < self.state.local_match_count:
                    # Upload database
                    async with aiofiles.open(self.db_path, 'rb') as f:
                        data = await f.read()

                    async with session.post(
                        f"{node.http_url}/elo/upload",
                        data=data,
                        timeout=aiohttp.ClientTimeout(total=120)
                    ) as resp:
                        if resp.status == 200:
                            logger.info(f"Uploaded {self.state.local_match_count} matches via HTTP")
                            return True

                return True  # Same count

        except Exception as e:
            logger.warning(f"HTTP sync failed: {e}")
            return False

    async def ensure_latest(self) -> bool:
        """
        Ensure we have the latest database before critical operations.
        Use before training or making decisions based on Elo.
        """
        return await self.sync_with_cluster()

    async def push_new_matches(self, matches: list[dict]) -> int:
        """
        Push new matches to the cluster.
        Call after playing games locally.
        Returns number of matches pushed.
        """
        if not matches:
            return 0

        # First, insert locally
        inserted = self._insert_matches_locally(matches)
        self._update_local_stats()

        # Then push to coordinator
        coordinator = self.nodes.get(self.coordinator_host)
        if coordinator:
            try:
                await self._push_matches_to_node(coordinator, matches)
            except Exception as e:
                logger.warning(f"Failed to push matches to coordinator: {e}")

        self._save_state()
        return inserted

    def _insert_matches_locally(self, matches: list[dict]) -> int:
        """Insert matches into local database."""
        if not self.db_path.exists():
            return 0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get existing game_ids
        cursor.execute("SELECT game_id FROM match_history WHERE game_id IS NOT NULL")
        existing = {row[0] for row in cursor.fetchall()}

        inserted = 0
        for match in matches:
            game_id = match.get('game_id')
            if game_id and game_id in existing:
                continue

            cursor.execute("""
                INSERT INTO match_history
                (participant_a, participant_b, board_type, num_players, winner,
                 game_length, duration_sec, timestamp, tournament_id, game_id, metadata, worker)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                match['participant_a'], match['participant_b'], match['board_type'],
                match['num_players'], match.get('winner'), match.get('game_length'),
                match.get('duration_sec'), match.get('timestamp', time.time()),
                match.get('tournament_id'), game_id, match.get('metadata'),
                match.get('worker')
            ))
            inserted += 1
            if game_id:
                existing.add(game_id)

        conn.commit()
        conn.close()
        return inserted

    async def _push_matches_to_node(self, node: NodeInfo, matches: list[dict]):
        """Push matches to a specific node."""
        host = node.tailscale_ip or node.ssh_host
        if not host:
            return

        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(matches, f)
            temp_file = f.name

        try:
            # Copy and merge on remote
            result = await asyncio.create_subprocess_exec(
                'ssh', '-o', 'ConnectTimeout=10', host,
                """python3 -c "
import json
import sqlite3
import time

matches = json.load(open('/dev/stdin'))
db = sqlite3.connect('/root/ringrift/ai-service/data/unified_elo.db')
cur = db.cursor()
cur.execute('SELECT game_id FROM match_history WHERE game_id IS NOT NULL')
existing = {r[0] for r in cur.fetchall()}

inserted = 0
for m in matches:
    gid = m.get('game_id')
    if gid and gid in existing:
        continue
    cur.execute('''INSERT INTO match_history
        (participant_a, participant_b, board_type, num_players, winner, timestamp, game_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)''',
        (m['participant_a'], m['participant_b'], m['board_type'], m['num_players'],
         m.get('winner'), m.get('timestamp', time.time()), gid))
    inserted += 1
    if gid:
        existing.add(gid)

db.commit()
print(f'Inserted {inserted} matches')
"
""",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            with open(temp_file, 'rb') as f:
                stdout, _ = await asyncio.wait_for(
                    result.communicate(input=f.read()),
                    timeout=30
                )

            logger.info(f"Push result: {stdout.decode().strip()}")
        finally:
            os.unlink(temp_file)

    def get_status(self) -> dict[str, Any]:
        """Get current sync status."""
        return {
            'local_matches': self.state.local_match_count,
            'last_sync': self.state.last_sync_timestamp,
            'synced_from': self.state.synced_from,
            'db_hash': self.state.last_sync_hash,
            'nodes_known': len(self.nodes),
            'coordinator': self.coordinator_host,
            'recent_errors': self.state.sync_errors[-5:]
        }


# Singleton instance for easy access
_sync_manager: EloSyncManager | None = None


def get_elo_sync_manager(
    db_path: Path | None = None,
    coordinator_host: str = "lambda-h100"
) -> EloSyncManager:
    """Get or create the singleton EloSyncManager instance."""
    global _sync_manager
    if _sync_manager is None:
        _sync_manager = EloSyncManager(
            db_path=db_path or DEFAULT_DB_PATH,
            coordinator_host=coordinator_host
        )
    return _sync_manager


async def sync_elo_after_games(matches: list[dict]) -> int:
    """
    Convenience function to sync after playing games.
    Call this after each batch of games.
    """
    manager = get_elo_sync_manager()
    if not manager.state.local_match_count:
        await manager.initialize()
    return await manager.push_new_matches(matches)


async def ensure_elo_synced() -> bool:
    """
    Convenience function to ensure database is synced.
    Call before training or making Elo-based decisions.
    """
    manager = get_elo_sync_manager()
    if not manager.state.local_match_count:
        await manager.initialize()
    return await manager.sync_with_cluster()
