"""ClusterConsolidationDaemon - Pulls games from cluster nodes into canonical databases.

This daemon solves the critical gap in the training pipeline where selfplay games
are generated on 30+ cluster nodes but never reach the coordinator's canonical
databases for training.

The Problem:
    AutoSyncDaemon syncs game DBs BETWEEN cluster nodes (P2P gossip), but NOT
    to canonical databases. consolidate_selfplay.py only looks at local
    data/selfplay/p2p_hybrid/ directories, which are empty on the coordinator.

    Result: Games accumulate on cluster nodes but training starves.

The Solution:
    1. Get list of alive peers from P2P status
    2. For each peer with selfplay data, rsync their games.db to local
    3. Merge synced DBs into canonical databases using INSERT OR IGNORE
    4. Emit CONSOLIDATION_COMPLETE to trigger training pipeline

Event-Driven Flow:
    NEW_GAMES_AVAILABLE → ClusterConsolidationDaemon._on_new_games()
        ↓ (triggers priority sync from that node)
    rsync games from cluster nodes
        ↓
    Merge into canonical_{board}_{n}p.db
        ↓ (deduplicate by game_id)
    CONSOLIDATION_COMPLETE → DataPipelineOrchestrator

January 2026: Created as part of P2P & Training Loop stability improvements.
Fixes the critical handoff failure between distributed selfplay and training.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import socket
import sqlite3
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.contracts import CoordinatorStatus
from app.coordination.event_utils import make_config_key
from app.coordination.event_emission_helpers import safe_emit_event
from app.config.thresholds import SQLITE_TIMEOUT, SQLITE_MERGE_TIMEOUT

logger = logging.getLogger(__name__)

__all__ = [
    "ClusterConsolidationDaemon",
    "ClusterConsolidationConfig",
    "get_cluster_consolidation_daemon",
    "reset_cluster_consolidation_daemon",
]

# All canonical configurations
CANONICAL_CONFIGS = [
    ("hex8", 2), ("hex8", 3), ("hex8", 4),
    ("square8", 2), ("square8", 3), ("square8", 4),
    ("square19", 2), ("square19", 3), ("square19", 4),
    ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
]


@dataclass
class ClusterConsolidationConfig:
    """Configuration for the cluster consolidation daemon."""

    # Daemon control
    enabled: bool = True
    cycle_interval_seconds: int = 300  # 5 minutes default

    # Base paths
    synced_dir: Path = field(default_factory=lambda: Path("data/games/cluster_synced"))
    canonical_dir: Path = field(default_factory=lambda: Path("data/games"))

    # Sync settings
    sync_timeout_seconds: int = 120  # Per-node sync timeout
    max_concurrent_syncs: int = 5  # Max parallel node syncs
    remote_db_path: str = "ringrift/ai-service/data/games/selfplay.db"

    # Merge settings
    min_moves_for_valid: int = 5  # Minimum moves for a valid game
    batch_commit_size: int = 100  # Commit every N games

    # Transport settings (Tailscale -> SSH -> skip)
    prefer_tailscale: bool = True
    ssh_timeout_seconds: int = 30
    ssh_connect_timeout: int = 10

    # Coordinator detection
    coordinator_only: bool = True  # Only run on coordinator nodes

    @classmethod
    def from_env(cls) -> "ClusterConsolidationConfig":
        """Load configuration from environment variables."""
        return cls(
            enabled=os.getenv("RINGRIFT_CLUSTER_CONSOLIDATION_ENABLED", "true").lower() == "true",
            cycle_interval_seconds=int(os.getenv("RINGRIFT_CLUSTER_CONSOLIDATION_INTERVAL", "300")),
            synced_dir=Path(os.getenv("RINGRIFT_CLUSTER_SYNCED_DIR", "data/games/cluster_synced")),
            canonical_dir=Path(os.getenv("RINGRIFT_CANONICAL_DIR", "data/games")),
            max_concurrent_syncs=int(os.getenv("RINGRIFT_CLUSTER_MAX_CONCURRENT_SYNCS", "5")),
            coordinator_only=os.getenv("RINGRIFT_CLUSTER_CONSOLIDATION_COORDINATOR_ONLY", "true").lower() == "true",
        )


@dataclass
class SyncStats:
    """Statistics for a sync cycle."""
    nodes_attempted: int = 0
    nodes_synced: int = 0
    nodes_failed: int = 0
    games_merged: int = 0
    games_duplicate: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


class ClusterConsolidationDaemon(HandlerBase):
    """Daemon that pulls games from cluster nodes and merges into canonical DBs.

    This is the critical missing piece that bridges distributed selfplay
    (on 30+ cluster nodes) with the training pipeline (needs canonical DBs).

    Subscribes to:
    - NEW_GAMES_AVAILABLE: Priority sync from specific node
    - DATA_SYNC_COMPLETED: Sync completed, trigger merge

    Emits:
    - CONSOLIDATION_STARTED: Beginning sync cycle
    - CONSOLIDATION_COMPLETE: Games merged (triggers training pipeline)
    """

    def __init__(self, config: ClusterConsolidationConfig | None = None):
        """Initialize the cluster consolidation daemon.

        Args:
            config: Configuration for consolidation behavior
        """
        self._daemon_config = config or ClusterConsolidationConfig.from_env()
        super().__init__(
            name="ClusterConsolidation",
            config=self._daemon_config,
            cycle_interval=float(self._daemon_config.cycle_interval_seconds),
        )

        # State tracking
        self._priority_nodes: set[str] = set()  # Nodes to sync ASAP
        self._last_sync_time: dict[str, float] = {}  # node_id -> timestamp
        self._sync_stats_history: list[SyncStats] = []
        self._subscribed = False
        self._is_coordinator: bool | None = None

        # Concurrency control
        self._sync_semaphore = asyncio.Semaphore(self._daemon_config.max_concurrent_syncs)
        self._lock = asyncio.Lock()

    @property
    def config(self) -> ClusterConsolidationConfig:
        """Return daemon configuration."""
        return self._daemon_config

    async def _on_start(self) -> None:
        """Called after daemon starts."""
        # Check if we should run (coordinator only by default)
        if self._daemon_config.coordinator_only and not self._is_coordinator_node():
            logger.info("[ClusterConsolidation] Not a coordinator node, skipping")
            self._status = CoordinatorStatus.STOPPED
            return

        # Ensure directories exist
        self._daemon_config.synced_dir.mkdir(parents=True, exist_ok=True)
        self._daemon_config.canonical_dir.mkdir(parents=True, exist_ok=True)

        await self._subscribe_to_events()

        # Initial sync on startup
        logger.info("[ClusterConsolidation] Starting initial sync cycle...")
        try:
            await self._run_sync_cycle()
        except Exception as e:
            logger.warning(f"[ClusterConsolidation] Initial sync failed: {e}")

    async def _on_stop(self) -> None:
        """Called before daemon stops."""
        await self._unsubscribe_from_events()

    async def _run_cycle(self) -> None:
        """Run one sync and merge cycle.

        Called by HandlerBase's main loop.
        """
        if self._daemon_config.coordinator_only and not self._is_coordinator_node():
            return

        await self._run_sync_cycle()

    def _is_coordinator_node(self) -> bool:
        """Check if this node is a coordinator.

        Coordinators typically have IS_COORDINATOR=true or are mac-studio/local-mac.
        """
        if self._is_coordinator is not None:
            return self._is_coordinator

        # Check environment
        is_coordinator = os.environ.get("RINGRIFT_IS_COORDINATOR", "").lower() == "true"

        # Also check common coordinator hostnames
        hostname = socket.gethostname().lower()
        coordinator_names = ("mac-studio", "local-mac", "coordinator", "macbook")
        is_coordinator = is_coordinator or any(n in hostname for n in coordinator_names)

        self._is_coordinator = is_coordinator
        return is_coordinator

    async def _subscribe_to_events(self) -> None:
        """Subscribe to relevant events."""
        if self._subscribed:
            return

        try:
            from app.coordination.event_router import get_event_bus
            from app.distributed.data_events import DataEventType

            bus = get_event_bus()
            bus.subscribe(DataEventType.NEW_GAMES_AVAILABLE, self._on_new_games_available)
            bus.subscribe(DataEventType.SELFPLAY_COMPLETE, self._on_selfplay_complete)

            self._subscribed = True
            logger.info("[ClusterConsolidation] Subscribed to events")

        except ImportError as e:
            logger.warning(f"[ClusterConsolidation] Could not subscribe to events: {e}")

    async def _unsubscribe_from_events(self) -> None:
        """Unsubscribe from events on shutdown."""
        if not self._subscribed:
            return

        try:
            from app.coordination.event_router import get_event_bus
            from app.distributed.data_events import DataEventType

            bus = get_event_bus()
            bus.unsubscribe(DataEventType.NEW_GAMES_AVAILABLE, self._on_new_games_available)
            bus.unsubscribe(DataEventType.SELFPLAY_COMPLETE, self._on_selfplay_complete)

            self._subscribed = False
        except Exception as e:
            logger.warning(f"[ClusterConsolidation] Error unsubscribing: {e}")

    def _on_new_games_available(self, event: Any) -> None:
        """Handle NEW_GAMES_AVAILABLE event - queue priority sync."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            source_node = payload.get("source", payload.get("host", payload.get("node_id", "")))
            if source_node:
                self._priority_nodes.add(source_node)
                logger.debug(f"[ClusterConsolidation] Priority sync queued for {source_node}")
        except Exception as e:
            logger.debug(f"[ClusterConsolidation] Error handling NEW_GAMES_AVAILABLE: {e}")

    def _on_selfplay_complete(self, event: Any) -> None:
        """Handle SELFPLAY_COMPLETE event - queue priority sync."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            source_node = payload.get("node_id", payload.get("host", ""))
            if source_node:
                self._priority_nodes.add(source_node)
        except Exception as e:
            logger.debug(f"[ClusterConsolidation] Error handling SELFPLAY_COMPLETE: {e}")

    async def _run_sync_cycle(self) -> None:
        """Run a full sync cycle: get peers, sync, merge."""
        stats = SyncStats()
        start_time = time.time()

        try:
            # Get alive peers from P2P
            peers = await self._get_alive_peers()
            if not peers:
                logger.debug("[ClusterConsolidation] No alive peers found")
                return

            # Filter to nodes that have selfplay data
            sync_nodes = await self._filter_sync_candidates(peers)
            if not sync_nodes:
                logger.debug("[ClusterConsolidation] No nodes with new data to sync")
                return

            stats.nodes_attempted = len(sync_nodes)
            logger.info(f"[ClusterConsolidation] Syncing from {len(sync_nodes)} nodes")

            # Sync from each node in parallel
            sync_tasks = [
                self._sync_from_node(node_id, node_info)
                for node_id, node_info in sync_nodes.items()
            ]
            results = await asyncio.gather(*sync_tasks, return_exceptions=True)

            # Count successes/failures
            for result in results:
                if isinstance(result, Exception):
                    stats.nodes_failed += 1
                    stats.errors.append(str(result))
                elif result:
                    stats.nodes_synced += 1
                else:
                    stats.nodes_failed += 1

            # Clear priority nodes that were synced
            self._priority_nodes.clear()

            # Merge synced databases into canonical
            merge_stats = await self._merge_all_configs()
            stats.games_merged = merge_stats.get("merged", 0)
            stats.games_duplicate = merge_stats.get("duplicate", 0)

        except Exception as e:
            logger.error(f"[ClusterConsolidation] Sync cycle error: {e}")
            stats.errors.append(str(e))

        finally:
            stats.duration_seconds = time.time() - start_time
            self._sync_stats_history.append(stats)

            # Keep only recent history
            if len(self._sync_stats_history) > 50:
                self._sync_stats_history = self._sync_stats_history[-50:]

            if stats.nodes_synced > 0 or stats.games_merged > 0:
                logger.info(
                    f"[ClusterConsolidation] Cycle complete: "
                    f"synced {stats.nodes_synced}/{stats.nodes_attempted} nodes, "
                    f"merged {stats.games_merged} games, "
                    f"duration={stats.duration_seconds:.1f}s"
                )

    async def _get_alive_peers(self) -> list[dict[str, Any]]:
        """Get list of alive peers from P2P status.

        Returns:
            List of peer info dicts with node_id, host, etc.
        """
        try:
            from app.coordination.p2p_integration import get_p2p_alive_nodes

            nodes = await get_p2p_alive_nodes()
            # Convert to dicts for easier handling
            return [
                {
                    "node_id": n.node_id,
                    "host": n.host or n.node_id,
                    "provider": n.provider,
                    "has_gpu": n.has_gpu,
                }
                for n in nodes
                if n.node_id != socket.gethostname()  # Exclude self
            ]
        except ImportError as e:
            logger.debug(f"[ClusterConsolidation] P2P integration unavailable: {e}")
            return []
        except Exception as e:
            logger.warning(f"[ClusterConsolidation] Error getting peers: {e}")
            return []

    async def _filter_sync_candidates(
        self,
        peers: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Filter peers to those that should be synced.

        Priority nodes are always included. Others are included if enough time
        has passed since last sync.

        Args:
            peers: List of peer info dicts

        Returns:
            Dict of node_id -> node_info for candidates
        """
        candidates = {}
        now = time.time()
        min_interval = 60.0  # Minimum 1 minute between syncs to same node

        for peer in peers:
            node_id = peer.get("node_id", "")
            if not node_id:
                continue

            # Priority nodes always sync
            if node_id in self._priority_nodes:
                candidates[node_id] = peer
                continue

            # Others: check time since last sync
            last_sync = self._last_sync_time.get(node_id, 0)
            if now - last_sync >= min_interval:
                candidates[node_id] = peer

        return candidates

    async def _sync_from_node(
        self,
        node_id: str,
        node_info: dict[str, Any],
    ) -> bool:
        """Sync selfplay database from a single node.

        Uses rsync over SSH with Tailscale fallback.

        Args:
            node_id: Node identifier
            node_info: Node information dict

        Returns:
            True if sync successful
        """
        async with self._sync_semaphore:
            try:
                # Get host to connect to (prefer Tailscale)
                host = await self._get_best_host(node_id, node_info)
                if not host:
                    logger.debug(f"[ClusterConsolidation] No reachable host for {node_id}")
                    return False

                # Determine SSH user
                ssh_user = self._get_ssh_user(node_id, node_info)

                # Target local path
                local_path = self._daemon_config.synced_dir / f"{node_id}_selfplay.db"

                # Sync via rsync
                success = await self._rsync_database(
                    host=host,
                    user=ssh_user,
                    remote_path=self._daemon_config.remote_db_path,
                    local_path=local_path,
                )

                if success:
                    self._last_sync_time[node_id] = time.time()
                    logger.debug(f"[ClusterConsolidation] Synced from {node_id}")

                return success

            except Exception as e:
                logger.debug(f"[ClusterConsolidation] Sync failed for {node_id}: {e}")
                return False

    async def _get_best_host(
        self,
        node_id: str,
        node_info: dict[str, Any],
    ) -> str | None:
        """Get the best reachable host for a node.

        Tries Tailscale IP first, then SSH host, then node_id.

        Args:
            node_id: Node identifier
            node_info: Node information dict

        Returns:
            Best host string or None if unreachable
        """
        # Try to get Tailscale IP from cluster config
        try:
            from app.config.cluster_config import get_cluster_nodes

            nodes = get_cluster_nodes()
            if node_id in nodes:
                cluster_node = nodes[node_id]
                if self._daemon_config.prefer_tailscale and cluster_node.tailscale_ip:
                    return cluster_node.tailscale_ip
                if cluster_node.ssh_host:
                    return cluster_node.ssh_host
        except ImportError:
            pass

        # Fallback to node_info
        host = node_info.get("host", node_id)
        if host:
            return host

        return None

    def _get_ssh_user(self, node_id: str, node_info: dict[str, Any]) -> str:
        """Get SSH user for a node.

        Args:
            node_id: Node identifier
            node_info: Node information dict

        Returns:
            SSH username
        """
        try:
            from app.config.cluster_config import get_cluster_nodes

            nodes = get_cluster_nodes()
            if node_id in nodes:
                return nodes[node_id].ssh_user
        except ImportError:
            pass

        # Provider-based defaults
        provider = node_info.get("provider", "").lower()
        if provider in ("vast", "runpod", "vultr"):
            return "root"

        return "ubuntu"

    async def _rsync_database(
        self,
        host: str,
        user: str,
        remote_path: str,
        local_path: Path,
    ) -> bool:
        """Rsync a database file from a remote host.

        Args:
            host: Remote host
            user: SSH username
            remote_path: Path on remote host
            local_path: Local destination path

        Returns:
            True if successful
        """
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Build rsync command
        ssh_opts = f"-o ConnectTimeout={self._daemon_config.ssh_connect_timeout} -o StrictHostKeyChecking=no"
        remote_spec = f"{user}@{host}:{remote_path}"

        cmd = [
            "rsync",
            "-az",  # Archive mode, compress
            "--timeout", str(self._daemon_config.ssh_timeout_seconds),
            "-e", f"ssh {ssh_opts}",
            remote_spec,
            str(local_path),
        ]

        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=float(self._daemon_config.sync_timeout_seconds),
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                return True

            # Check for common non-fatal errors
            stderr_str = stderr.decode() if stderr else ""
            if "No such file" in stderr_str or "does not exist" in stderr_str:
                logger.debug(f"[ClusterConsolidation] No database on {host}")
                return False

            logger.debug(f"[ClusterConsolidation] rsync to {host} failed: {stderr_str}")
            return False

        except asyncio.TimeoutError:
            logger.debug(f"[ClusterConsolidation] rsync to {host} timed out")
            return False
        except Exception as e:
            logger.debug(f"[ClusterConsolidation] rsync error: {e}")
            return False

    async def _merge_all_configs(self) -> dict[str, int]:
        """Merge all synced databases into canonical databases.

        Uses INSERT OR IGNORE for deduplication (preserves existing data).

        Returns:
            Dict with merge statistics
        """
        stats = {"merged": 0, "duplicate": 0, "errors": 0}

        # Find all synced databases
        synced_dbs = list(self._daemon_config.synced_dir.glob("*.db"))
        if not synced_dbs:
            return stats

        # Merge into each canonical config
        for board_type, num_players in CANONICAL_CONFIGS:
            config_key = make_config_key(board_type, num_players)
            canonical_db = self._daemon_config.canonical_dir / f"canonical_{config_key}.db"

            try:
                config_stats = await self._merge_config(
                    config_key=config_key,
                    board_type=board_type,
                    num_players=num_players,
                    synced_dbs=synced_dbs,
                    canonical_db=canonical_db,
                )
                stats["merged"] += config_stats.get("merged", 0)
                stats["duplicate"] += config_stats.get("duplicate", 0)

                if config_stats.get("merged", 0) > 0:
                    await self._emit_consolidation_complete(
                        config_key=config_key,
                        games_merged=config_stats["merged"],
                        canonical_db=str(canonical_db),
                    )

            except Exception as e:
                logger.warning(f"[ClusterConsolidation] Merge error for {config_key}: {e}")
                stats["errors"] += 1

        return stats

    async def _merge_config(
        self,
        config_key: str,
        board_type: str,
        num_players: int,
        synced_dbs: list[Path],
        canonical_db: Path,
    ) -> dict[str, int]:
        """Merge synced databases into a single canonical database.

        Uses INSERT OR IGNORE to prevent duplicates and preserve existing data.
        CRITICAL: Never deletes existing canonical data.

        Args:
            config_key: Config identifier (e.g., "hex8_2p")
            board_type: Board type to filter
            num_players: Player count to filter
            synced_dbs: List of synced database paths
            canonical_db: Target canonical database

        Returns:
            Dict with merge statistics
        """
        # Run merge in thread pool to avoid blocking
        return await asyncio.to_thread(
            self._merge_config_sync,
            config_key,
            board_type,
            num_players,
            synced_dbs,
            canonical_db,
        )

    def _merge_config_sync(
        self,
        config_key: str,
        board_type: str,
        num_players: int,
        synced_dbs: list[Path],
        canonical_db: Path,
    ) -> dict[str, int]:
        """Synchronous merge implementation.

        IMPORTANT: Uses INSERT OR IGNORE to deduplicate by game_id.
        This pattern ensures:
        1. Existing data is NEVER deleted (see incident Jan 2, 2026)
        2. Duplicate game_ids are silently ignored
        3. Only new games are added
        """
        stats = {"merged": 0, "duplicate": 0, "scanned": 0}

        # Ensure canonical DB has correct schema
        self._ensure_canonical_schema(canonical_db)

        # Get existing game IDs for fast duplicate checking
        existing_ids = self._get_existing_game_ids(canonical_db)

        target_conn = None
        try:
            target_conn = sqlite3.connect(str(canonical_db), timeout=SQLITE_MERGE_TIMEOUT)

            for source_db in synced_dbs:
                try:
                    source_stats = self._merge_single_db(
                        source_db=source_db,
                        target_conn=target_conn,
                        board_type=board_type,
                        num_players=num_players,
                        existing_ids=existing_ids,
                    )
                    stats["merged"] += source_stats.get("merged", 0)
                    stats["duplicate"] += source_stats.get("duplicate", 0)
                    stats["scanned"] += source_stats.get("scanned", 0)

                    # Update existing_ids with new games
                    existing_ids.update(source_stats.get("new_ids", set()))

                except Exception as e:
                    logger.debug(f"[ClusterConsolidation] Error merging {source_db}: {e}")

            target_conn.commit()

        except Exception as e:
            logger.error(f"[ClusterConsolidation] Merge failed for {config_key}: {e}")
        finally:
            if target_conn:
                target_conn.close()

        return stats

    def _merge_single_db(
        self,
        source_db: Path,
        target_conn: sqlite3.Connection,
        board_type: str,
        num_players: int,
        existing_ids: set[str],
    ) -> dict[str, Any]:
        """Merge games from a single source database.

        Args:
            source_db: Source database path
            target_conn: Target database connection
            board_type: Filter by board type
            num_players: Filter by player count
            existing_ids: Set of existing game IDs

        Returns:
            Dict with merge stats
        """
        stats: dict[str, Any] = {"merged": 0, "duplicate": 0, "scanned": 0, "new_ids": set()}

        source_conn = None
        try:
            source_conn = sqlite3.connect(str(source_db), timeout=SQLITE_TIMEOUT)
            source_conn.row_factory = sqlite3.Row

            # Query games for this config
            cursor = source_conn.execute("""
                SELECT * FROM games
                WHERE board_type = ? AND num_players = ?
                AND game_status IN ('completed', 'finished', 'complete', 'victory')
            """, (board_type, num_players))

            # Get target columns
            target_cursor = target_conn.execute("PRAGMA table_info(games)")
            target_columns = {row[1] for row in target_cursor.fetchall()}

            for row in cursor:
                stats["scanned"] += 1
                game_id = row["game_id"]

                # Skip duplicates
                if game_id in existing_ids:
                    stats["duplicate"] += 1
                    continue

                # Validate game has moves
                total_moves = row["total_moves"] or 0
                if total_moves < self._daemon_config.min_moves_for_valid:
                    continue

                # Insert game (only columns that exist in target)
                try:
                    source_cols = [desc[0] for desc in cursor.description]
                    common_cols = [c for c in source_cols if c in target_columns]
                    col_indices = [source_cols.index(c) for c in common_cols]

                    raw_row = list(row)
                    values = [raw_row[i] for i in col_indices]

                    placeholders = ",".join("?" * len(common_cols))
                    target_conn.execute(
                        f"INSERT OR IGNORE INTO games ({','.join(common_cols)}) VALUES ({placeholders})",
                        values,
                    )

                    # Copy related tables
                    self._copy_game_data(source_conn, target_conn, game_id)

                    # January 2026: Post-insert validation to prevent orphan games
                    MIN_MOVES_REQUIRED = 5
                    cursor = target_conn.execute(
                        "SELECT COUNT(*) FROM game_moves WHERE game_id = ?",
                        (game_id,)
                    )
                    move_count = cursor.fetchone()[0]

                    if move_count < MIN_MOVES_REQUIRED:
                        # Delete orphan game - insufficient move data
                        target_conn.execute("DELETE FROM games WHERE game_id = ?", (game_id,))
                        stats["orphans_prevented"] = stats.get("orphans_prevented", 0) + 1
                        logger.debug(
                            f"[ClusterConsolidation] Prevented orphan game {game_id}: "
                            f"only {move_count} moves (need {MIN_MOVES_REQUIRED})"
                        )
                        safe_emit_event(
                            "orphan_game_prevented",
                            {
                                "game_id": game_id,
                                "move_count": move_count,
                                "min_required": MIN_MOVES_REQUIRED,
                                "source": "cluster_consolidation_daemon",
                            },
                            context="ClusterConsolidation",
                        )
                        continue  # Skip counting as merged

                    stats["merged"] += 1
                    stats["new_ids"].add(game_id)

                except sqlite3.Error as e:
                    logger.debug(f"[ClusterConsolidation] Insert error for {game_id}: {e}")

        except sqlite3.Error as e:
            logger.debug(f"[ClusterConsolidation] Source DB error: {e}")
        finally:
            if source_conn:
                source_conn.close()

        return stats

    def _copy_game_data(
        self,
        source_conn: sqlite3.Connection,
        target_conn: sqlite3.Connection,
        game_id: str,
    ) -> None:
        """Copy related game data (moves, states, players).

        Uses INSERT OR IGNORE to prevent duplicate inserts.
        """
        tables = ["game_moves", "game_initial_state", "game_state_snapshots", "game_players"]

        for table in tables:
            try:
                # Get target columns
                target_cursor = target_conn.execute(f"PRAGMA table_info({table})")
                target_cols = {row[1] for row in target_cursor.fetchall()}
                if not target_cols:
                    continue

                # Query source
                cursor = source_conn.execute(
                    f"SELECT * FROM {table} WHERE game_id = ?",
                    (game_id,),
                )
                rows = cursor.fetchall()
                if not rows:
                    continue

                # Filter columns
                source_cols = [desc[0] for desc in cursor.description]
                common_cols = [c for c in source_cols if c in target_cols]
                col_indices = [source_cols.index(c) for c in common_cols]

                # Insert rows
                filtered_rows = [
                    tuple(row[i] for i in col_indices)
                    for row in rows
                ]
                placeholders = ",".join("?" * len(common_cols))
                target_conn.executemany(
                    f"INSERT OR IGNORE INTO {table} ({','.join(common_cols)}) VALUES ({placeholders})",
                    filtered_rows,
                )

            except sqlite3.Error as e:
                logger.debug(f"[ClusterConsolidation] Error copying {table}: {e}")

    def _ensure_canonical_schema(self, db_path: Path) -> None:
        """Ensure canonical database has correct schema."""
        db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(str(db_path), timeout=SQLITE_TIMEOUT) as conn:
            # Main games table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS games (
                    game_id TEXT PRIMARY KEY,
                    board_type TEXT NOT NULL,
                    num_players INTEGER NOT NULL,
                    rng_seed INTEGER,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    game_status TEXT NOT NULL,
                    winner INTEGER,
                    termination_reason TEXT,
                    total_moves INTEGER NOT NULL,
                    total_turns INTEGER NOT NULL,
                    duration_ms INTEGER,
                    source TEXT,
                    schema_version INTEGER NOT NULL DEFAULT 5,
                    time_control_type TEXT DEFAULT 'none',
                    initial_time_ms INTEGER,
                    time_increment_ms INTEGER,
                    metadata_json TEXT,
                    quality_score REAL,
                    quality_category TEXT
                )
            """)

            # Moves table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS game_moves (
                    game_id TEXT NOT NULL,
                    move_number INTEGER NOT NULL,
                    player INTEGER NOT NULL,
                    position_q INTEGER,
                    position_r INTEGER,
                    move_type TEXT,
                    move_probs TEXT,
                    PRIMARY KEY (game_id, move_number),
                    FOREIGN KEY (game_id) REFERENCES games(game_id)
                )
            """)

            # Initial state table (must use initial_state_json to match TypeScript)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS game_initial_state (
                    game_id TEXT PRIMARY KEY,
                    initial_state_json TEXT NOT NULL,
                    compressed INTEGER DEFAULT 0,
                    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
                )
            """)

            # State snapshots table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS game_state_snapshots (
                    game_id TEXT NOT NULL,
                    move_number INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    PRIMARY KEY (game_id, move_number),
                    FOREIGN KEY (game_id) REFERENCES games(game_id)
                )
            """)

            # Players table (must match TypeScript schema in SelfPlayGameService)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS game_players (
                    game_id TEXT NOT NULL,
                    player_number INTEGER NOT NULL,
                    player_type TEXT,
                    ai_type TEXT,
                    ai_difficulty INTEGER,
                    ai_profile_id TEXT,
                    final_eliminated_rings INTEGER,
                    final_territory_spaces INTEGER,
                    final_rings_in_hand INTEGER,
                    model_version TEXT,
                    PRIMARY KEY (game_id, player_number),
                    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
                )
            """)

            # Indexes for performance
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_games_board_players
                ON games(board_type, num_players)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_moves_game_id
                ON game_moves(game_id)
            """)

            conn.commit()

    def _get_existing_game_ids(self, db_path: Path) -> set[str]:
        """Get set of existing game IDs in a database."""
        if not db_path.exists():
            return set()

        try:
            with sqlite3.connect(str(db_path), timeout=SQLITE_TIMEOUT) as conn:
                cursor = conn.execute("SELECT game_id FROM games")
                return {row[0] for row in cursor.fetchall()}
        except sqlite3.Error:
            return set()

    async def _emit_consolidation_complete(
        self,
        config_key: str,
        games_merged: int,
        canonical_db: str,
    ) -> None:
        """Emit CONSOLIDATION_COMPLETE event."""
        safe_emit_event(
            "consolidation_complete",
            {
                "config_key": config_key,
                "games_merged": games_merged,
                "canonical_db": canonical_db,
                "timestamp": time.time(),
                "source": "cluster_consolidation",
            },
            context="ClusterConsolidation",
        )
        logger.debug(f"[ClusterConsolidation] Emitted CONSOLIDATION_COMPLETE for {config_key}")

    def get_status(self) -> dict[str, Any]:
        """Get daemon status."""
        recent_stats = self._sync_stats_history[-10:] if self._sync_stats_history else []
        return {
            "running": self._running,
            "subscribed": self._subscribed,
            "is_coordinator": self._is_coordinator,
            "priority_nodes": list(self._priority_nodes),
            "last_sync_times": dict(self._last_sync_time),
            "recent_stats": [
                {
                    "nodes_synced": s.nodes_synced,
                    "games_merged": s.games_merged,
                    "duration": s.duration_seconds,
                }
                for s in recent_stats
            ],
        }

    def health_check(self) -> HealthCheckResult:
        """Return health check result for DaemonManager integration."""
        details = {
            "running": self._running,
            "subscribed": self._subscribed,
            "is_coordinator": self._is_coordinator,
            "priority_nodes_count": len(self._priority_nodes),
            "recent_syncs": len(self._sync_stats_history),
            "total_games_merged": sum(s.games_merged for s in self._sync_stats_history),
            "cycles_completed": self._stats.cycles_completed,
            "errors_count": self._stats.errors_count,
        }

        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="ClusterConsolidationDaemon is not running",
                details=details,
            )

        # Non-coordinator nodes are expected to not run
        if self._daemon_config.coordinator_only and not self._is_coordinator:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message="ClusterConsolidationDaemon: not coordinator (expected)",
                details=details,
            )

        if not self._subscribed:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message="ClusterConsolidationDaemon not subscribed to events",
                details=details,
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"ClusterConsolidationDaemon healthy ({len(self._sync_stats_history)} syncs)",
            details=details,
        )


# Singleton pattern
_instance: ClusterConsolidationDaemon | None = None
_instance_lock = asyncio.Lock()


def get_cluster_consolidation_daemon() -> ClusterConsolidationDaemon:
    """Get the singleton ClusterConsolidationDaemon instance."""
    global _instance
    if _instance is None:
        _instance = ClusterConsolidationDaemon()
    return _instance


def reset_cluster_consolidation_daemon() -> None:
    """Reset the singleton instance (for testing)."""
    global _instance
    _instance = None
