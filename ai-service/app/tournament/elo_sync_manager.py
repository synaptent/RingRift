"""
Elo Database Synchronization Manager

Keeps unified_elo.db consistent across all cluster nodes using multiple transport methods.
Inherits from DatabaseSyncManager for common sync functionality.

December 2025: Migrated to DatabaseSyncManager base class (~670 LOC savings).

Features:
- Multi-transport failover (Tailscale → SSH → Vast.ai SSH → HTTP)
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
import sqlite3
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from app.coordination.database_sync_manager import (
    DatabaseSyncManager,
    DatabaseSyncState,
    SyncNodeInfo,
)
from app.coordination.sync_base import CircuitBreakerConfig
from app.utils.checksum_utils import compute_string_checksum
# Jan 13, 2026: Import harness extraction for preserving harness_type in sync
from app.training.composite_participant import extract_harness_type

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "unified_elo.db"
SYNC_STATE_PATH = Path(__file__).parent.parent.parent / "data" / "elo_sync_state.json"
# December 2025: Default to True - all GPU nodes should receive Elo updates
# Previously defaulted to False, causing Vast.ai nodes to miss Elo sync
ENABLE_VAST_ELO_SYNC = os.getenv("RINGRIFT_ENABLE_VAST_ELO_SYNC", "true").lower() in ("1", "true", "yes")


@dataclass
class EloManagerSyncState(DatabaseSyncState):
    """Elo-specific sync state extending DatabaseSyncState.

    Adds Elo-specific fields:
    - pending_matches: WAL for offline sync
    - sync_errors: Recent error messages
    """

    pending_matches: list[dict] = field(default_factory=list)
    sync_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dictionary."""
        base = super().to_dict()
        base.update({
            "pending_matches": self.pending_matches,
            "sync_errors": self.sync_errors[-10:],  # Keep last 10
        })
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EloManagerSyncState:
        """Deserialize state from dictionary."""
        return cls(
            last_sync_timestamp=data.get("last_sync_timestamp", 0.0),
            synced_nodes=set(data.get("synced_nodes", [])),
            pending_syncs=set(data.get("pending_syncs", [])),
            failed_nodes=set(data.get("failed_nodes", [])),
            sync_count=data.get("sync_count", 0),
            last_error=data.get("last_error"),
            local_record_count=data.get("local_record_count", 0),
            local_hash=data.get("local_hash", ""),
            synced_from=data.get("synced_from", ""),
            merge_conflicts=data.get("merge_conflicts", 0),
            total_syncs=data.get("total_syncs", 0),
            successful_syncs=data.get("successful_syncs", 0),
            pending_matches=data.get("pending_matches", []),
            sync_errors=data.get("sync_errors", []),
        )


# Backward-compatible alias
SyncState = EloManagerSyncState


# NodeInfo alias for backward compatibility
NodeInfo = SyncNodeInfo


class EloSyncManager(DatabaseSyncManager):
    """
    Manages Elo database synchronization across cluster nodes.

    Inherits from DatabaseSyncManager for common functionality:
    - Multi-transport failover
    - Circuit breakers per node
    - State persistence
    - Node discovery

    Elo-specific features:
    - Merge-based conflict resolution preserving all matches
    - Rating recalculation after merge
    - Push new matches to cluster
    - Vast.ai instance auto-discovery

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

    # December 2025: Hardcoded Vast instances deprecated
    # Now using dynamic discovery from distributed_hosts.yaml via discover_nodes()
    # The base class get_ready_nodes() includes all active Vast.ai nodes
    VAST_INSTANCES: dict[str, dict[str, str | int]] = {}  # Empty - use dynamic discovery

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        coordinator_host: str = "nebius-backbone-1",
        sync_interval: int = 300,
        p2p_url: str | None = None,
        enable_merge: bool = True,
    ):
        super().__init__(
            db_path=Path(db_path),
            state_path=SYNC_STATE_PATH,
            db_type="elo",
            coordinator_host=coordinator_host,
            sync_interval=float(sync_interval),
            p2p_url=p2p_url,
            enable_merge=enable_merge,
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=60.0,
            ),
        )

        # Elo-specific state
        self._elo_state = EloManagerSyncState()
        self._initialized = False
        if SYNC_STATE_PATH.exists():
            self._load_elo_state()

        # January 7, 2026: TTL tracking for failed nodes auto-recovery
        # Nodes in failed_nodes list will be removed after TTL expires
        self._failed_node_times: dict[str, float] = {}
        self._failed_node_ttl_seconds: float = 14400.0  # 4 hours default

    def _load_elo_state(self) -> None:
        """Load Elo-specific state from disk."""
        try:
            with open(SYNC_STATE_PATH) as f:
                data = json.load(f)
                self._elo_state = EloManagerSyncState.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load Elo sync state: {e}")

    def _save_elo_state(self) -> None:
        """Save Elo-specific state to disk."""
        try:
            SYNC_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(SYNC_STATE_PATH, "w") as f:
                json.dump(self._elo_state.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save Elo sync state: {e}")

    def _decay_failed_nodes(self) -> int:
        """Remove nodes from failed_nodes list after TTL expires.

        January 7, 2026: Added to prevent permanent exclusion of nodes that
        had transient failures. Without TTL decay, 96/97 nodes were stuck
        in failed_nodes list, blocking Elo sync propagation.

        Returns:
            Number of nodes recovered from failed_nodes.
        """
        now = time.time()
        recovered = []

        # Update timestamps for newly failed nodes
        for node in self._elo_state.failed_nodes:
            if node not in self._failed_node_times:
                self._failed_node_times[node] = now

        # Remove nodes that have exceeded TTL
        for node in list(self._elo_state.failed_nodes):
            added_time = self._failed_node_times.get(node, now)
            age_seconds = now - added_time
            if age_seconds > self._failed_node_ttl_seconds:
                recovered.append(node)
                self._elo_state.failed_nodes.discard(node)
                self._failed_node_times.pop(node, None)
                # Also sync with base class state if it exists
                if hasattr(self, "_db_state") and hasattr(self._db_state, "failed_nodes"):
                    self._db_state.failed_nodes.discard(node)

        if recovered:
            self._save_elo_state()
            logger.info(
                f"[EloSync] TTL decay recovered {len(recovered)} nodes: "
                + ", ".join(recovered[:5]) + ("..." if len(recovered) > 5 else "")
            )

        # Cleanup timestamps for nodes no longer in failed_nodes
        for node in list(self._failed_node_times.keys()):
            if node not in self._elo_state.failed_nodes:
                self._failed_node_times.pop(node, None)

        return len(recovered)

    # Backward compatibility property
    @property
    def state(self) -> EloManagerSyncState:
        """Get Elo sync state (backward compatibility)."""
        return self._elo_state

    # =========================================================================
    # DatabaseSyncManager abstract method implementations
    # =========================================================================

    def _get_remote_db_path(self) -> str:
        """Get remote database path for rsync."""
        return "~/ringrift/ai-service/data/unified_elo.db"

    def _get_remote_count_query(self) -> str:
        """Get SQL query for counting remote records."""
        return "SELECT COUNT(*) FROM match_history"

    def _update_local_stats(self) -> None:
        """Update local database statistics."""
        if not self.db_path.exists():
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM match_history")
            self._elo_state.local_record_count = cursor.fetchone()[0]
            self._db_state.local_record_count = self._elo_state.local_record_count

            # Calculate hash for change detection
            cursor.execute("SELECT COUNT(*), MAX(timestamp) FROM match_history")
            count, max_ts = cursor.fetchone()
            hash_value = compute_string_checksum(f"{count}:{max_ts}", algorithm="md5")
            self._elo_state.local_hash = hash_value
            self._db_state.local_hash = hash_value

            conn.close()
        except Exception as e:
            logger.warning(f"Failed to update local stats: {e}")

    async def _merge_databases(self, remote_db_path: Path) -> bool:
        """
        Merge remote database into local, preserving all unique matches.
        Uses game_id for deduplication if available, otherwise match signature.

        After merging match_history, recalculates all elo_ratings from scratch
        to ensure win/loss conservation invariant is maintained.

        Jan 2, 2026: Wrapped blocking SQLite operations with asyncio.to_thread()
        to avoid blocking the event loop.
        """
        if not remote_db_path.exists():
            return False

        try:
            # Run blocking SQLite operations in thread pool
            inserted = await asyncio.to_thread(
                self._merge_databases_sync, remote_db_path
            )

            if inserted > 0:
                logger.info(f"Merged {inserted} new matches from remote")
                self._elo_state.merge_conflicts += inserted
            else:
                logger.debug("No new matches to merge")

            return True

        except Exception as e:
            logger.error(f"Database merge failed: {e}")
            remote_db_path.unlink(missing_ok=True)
            return False

    def _detect_schema_variant(self, conn: sqlite3.Connection) -> dict[str, str]:
        """Detect schema variant and return column mapping to canonical names.

        Jan 5, 2026: Added to handle 3 different schema variants across cluster:
        - Local coordinator: participant_a, participant_b
        - Some remotes: model_a, model_b
        - Others: model_a_id, model_b_id

        Returns:
            Dict mapping canonical names to actual column names in this database.
            Keys: 'model_a', 'model_b' (canonical)
            Values: actual column name in this database
        """
        cursor = conn.execute("PRAGMA table_info(match_history)")
        columns = {row[1] for row in cursor.fetchall()}

        # Map to canonical internal keys (model_a, model_b)
        if "participant_a" in columns:
            return {"model_a": "participant_a", "model_b": "participant_b"}
        elif "model_a_id" in columns:
            return {"model_a": "model_a_id", "model_b": "model_b_id"}
        elif "model_a" in columns:
            return {"model_a": "model_a", "model_b": "model_b"}
        else:
            # Fallback to participant_a/participant_b (most common)
            logger.warning(f"Unknown schema variant, columns: {columns}. Using participant_a/b fallback.")
            return {"model_a": "participant_a", "model_b": "participant_b"}

    def _merge_databases_sync(self, remote_db_path: Path) -> int:
        """Synchronous helper for database merge operations.

        Jan 2, 2026: Extracted from async _merge_databases to avoid blocking
        the event loop with SQLite operations.

        Jan 5, 2026: Added schema detection for cross-cluster compatibility.

        Jan 12, 2026: C6 fix - Added transaction with savepoint for rollback
        on partial merge failure. Prevents database corruption from partial merges.

        Returns:
            Number of new matches inserted.
        """
        # Open both databases
        local_conn = sqlite3.connect(self.db_path)
        remote_conn = sqlite3.connect(remote_db_path)

        try:
            # C6 fix: Start transaction with savepoint for rollback on failure
            local_conn.execute("BEGIN IMMEDIATE")
            local_conn.execute("SAVEPOINT merge_start")

            local_cur = local_conn.cursor()
            remote_cur = remote_conn.cursor()

            # Detect schema variants for both databases
            local_schema = self._detect_schema_variant(local_conn)
            remote_schema = self._detect_schema_variant(remote_conn)

            # Get existing game_ids from local using detected column names
            local_model_a = local_schema["model_a"]
            local_model_b = local_schema["model_b"]
            local_cur.execute(f"""
                SELECT COALESCE(game_id,
                    {local_model_a} || '|' || {local_model_b} || '|' || timestamp)
                FROM match_history
            """)
            existing_ids = {row[0] for row in local_cur.fetchall() if row[0]}

            # Get all columns from remote
            remote_cur.execute("PRAGMA table_info(match_history)")
            columns = [col[1] for col in remote_cur.fetchall()]

            # Fetch remote matches
            remote_cur.execute("SELECT * FROM match_history")
            remote_matches = remote_cur.fetchall()

            # December 29, 2025: Optimized to use bulk insert with executemany()
            # Jan 5, 2026: Use schema-aware column names for match_id construction
            # Filter to only new matches first, then bulk insert
            remote_model_a = remote_schema["model_a"]
            remote_model_b = remote_schema["model_b"]
            new_matches = []
            for match in remote_matches:
                match_dict = dict(zip(columns, match, strict=False))
                # Use schema-aware column names for match_id construction
                match_id = match_dict.get("game_id") or \
                    f"{match_dict.get(remote_model_a)}|{match_dict.get(remote_model_b)}|{match_dict.get('timestamp')}"

                if match_id not in existing_ids:
                    new_matches.append(match)
                    existing_ids.add(match_id)

            # Bulk insert using executemany (much faster than N individual INSERTs)
            inserted = 0
            if new_matches:
                cols = ", ".join(columns)
                placeholders = ", ".join(["?" for _ in columns])
                local_cur.executemany(
                    f"INSERT OR IGNORE INTO match_history ({cols}) VALUES ({placeholders})",
                    new_matches
                )
                inserted = len(new_matches)

            # C6 fix: Release savepoint and commit on success
            local_conn.execute("RELEASE merge_start")
            local_conn.commit()
            remote_conn.close()

            # Cleanup temp file
            remote_db_path.unlink(missing_ok=True)

            # Recalculate ratings if we inserted new matches
            if inserted > 0:
                self._recalculate_ratings_from_history_sync(local_conn)

            local_conn.close()
            return inserted

        except Exception as e:
            # C6 fix: Rollback to savepoint on failure to prevent partial merges
            try:
                local_conn.execute("ROLLBACK TO merge_start")
                local_conn.execute("ROLLBACK")
                logger.warning(f"[EloSync] Merge rolled back due to error: {e}")
            except sqlite3.Error as rollback_error:
                logger.error(f"[EloSync] Rollback failed: {rollback_error}")
            finally:
                local_conn.close()
                remote_conn.close()
            raise

    # =========================================================================
    # Elo-specific methods
    # =========================================================================

    async def initialize(self) -> None:
        """Initialize the sync manager."""
        self._load_elo_state()

        # Checkpoint WAL before discovery to ensure database consistency
        # WAL files may contain uncommitted changes that affect node stats
        from app.coordination.wal_sync_utils import checkpoint_database

        if self.db_path.exists():
            checkpoint_database(str(self.db_path))

        await self.discover_nodes()
        if not ENABLE_VAST_ELO_SYNC:
            self.nodes = {
                name: node
                for name, node in self.nodes.items()
                if not self._is_vast_node(name, node)
            }
        # Add known Vast instances
        if ENABLE_VAST_ELO_SYNC:
            for name, info in self.VAST_INSTANCES.items():
                if name not in self.nodes:
                    self.nodes[name] = SyncNodeInfo(
                        name=name,
                        vast_ssh_host=info["host"],
                        vast_ssh_port=info["port"],
                        remote_db_path=self._get_remote_db_path(),
                    )
        else:
            logger.debug("VAST ELO sync disabled; skipping VAST instance discovery")
        self._update_local_stats()
        self._initialized = True
        logger.info(f"EloSyncManager initialized: {self._elo_state.local_record_count} local matches")

    @staticmethod
    def _is_vast_node(name: str, node: SyncNodeInfo) -> bool:
        host = (node.ssh_host or node.vast_ssh_host or "").lower()
        return name.startswith("vast-") or host.endswith("vast.ai")

    async def start_background_sync(self) -> None:
        """Start background sync loop."""
        await self.start()

    async def stop_background_sync(self) -> None:
        """Stop background sync loop."""
        await self.stop()

    async def start(self) -> None:
        """Start the Elo sync manager with bi-directional sync.

        December 30, 2025: Override to add push step after pull.
        The coordinator pushes its authoritative data to cluster nodes
        after pulling any new data from them.

        This fixes the issue where cluster nodes had stale Elo data
        because the base class only pulled (never pushed).
        """
        if self._running:
            logger.warning("EloSyncManager already running")
            return

        self._running = True
        is_coordinator = os.environ.get("RINGRIFT_IS_COORDINATOR", "").lower() in ("true", "1", "yes")

        logger.info(
            f"Starting EloSyncManager (interval={self.sync_interval}s, "
            f"is_coordinator={is_coordinator})"
        )

        while self._running:
            try:
                # Step 0: TTL decay for failed nodes (January 7, 2026)
                # Prevents permanent exclusion of nodes after transient failures
                self._decay_failed_nodes()

                # Step 1: Pull from cluster (may get new matches from training nodes)
                await self.sync_with_cluster()

                # Step 2: Push to cluster (coordinator distributes authoritative data)
                # Only coordinator pushes to avoid conflicts
                if is_coordinator:
                    await self.push_to_cluster()

            except Exception as e:
                logger.error(f"Elo sync cycle error: {e}")

            await asyncio.sleep(self.sync_interval)

    async def sync_with_cluster(self) -> bool:
        """Synchronize with cluster nodes. Returns True if any sync succeeded."""
        self._elo_state.total_syncs += 1
        results = await super().sync_with_cluster()
        success = any(results.values())
        if success:
            self._elo_state.successful_syncs += 1
            self._save_elo_state()
        return success

    async def push_to_cluster(self) -> dict[str, bool]:
        """Push local Elo database to all cluster nodes.

        December 30, 2025: Added to fix sync issue where coordinator's
        authoritative data never reached cluster nodes. The existing
        sync_with_cluster() only pulls and merges, which doesn't help
        when cluster nodes have stale/empty data.

        This method pushes the coordinator's database to all nodes,
        enabling them to make promotion decisions with up-to-date Elo ratings.

        Returns:
            Dict mapping node name to push success status
        """
        if not self._initialized:
            await self.initialize()

        results: dict[str, bool] = {}
        self._update_local_stats()

        logger.info(
            f"[EloSync] Pushing {self._elo_state.local_record_count} matches "
            f"to {len(self.nodes)} cluster nodes"
        )

        for node_name, node_info in self.nodes.items():
            # Skip nodes in circuit breaker (using base class method)
            if not self._can_sync_with_node(node_name):
                logger.debug(f"[EloSync] Skipping {node_name} (circuit open)")
                results[node_name] = False
                continue

            # Try push with failover (Tailscale → SSH → Vast SSH)
            success = await self._push_to_node(node_name, node_info)
            results[node_name] = success

            if success:
                logger.info(f"[EloSync] Successfully pushed to {node_name}")
            else:
                logger.warning(f"[EloSync] Failed to push to {node_name}")

        successful = sum(results.values())
        logger.info(f"[EloSync] Push complete: {successful}/{len(results)} nodes updated")

        return results

    async def _push_to_node(self, node_name: str, node_info: SyncNodeInfo) -> bool:
        """Push database to a single node with transport failover."""
        remote_path = node_info.remote_db_path or self._get_remote_db_path()

        # Try Tailscale first (most reliable within mesh)
        if node_info.tailscale_ip:
            try:
                success = await self._rsync_push_with_retry(
                    host=node_info.tailscale_ip,
                    remote_path=remote_path,
                    ssh_port=22,
                    verify=True,
                )
                if success:
                    return True
            except Exception as e:
                logger.debug(f"[EloSync] Tailscale push to {node_name} failed: {e}")

        # Try direct SSH
        if node_info.ssh_host:
            try:
                success = await self._rsync_push_with_retry(
                    host=node_info.ssh_host,
                    remote_path=remote_path,
                    ssh_port=node_info.ssh_port,
                    verify=True,
                )
                if success:
                    return True
            except Exception as e:
                logger.debug(f"[EloSync] SSH push to {node_name} failed: {e}")

        # Try Vast.ai SSH
        if node_info.vast_ssh_host and node_info.vast_ssh_port:
            vast_path = f"/workspace/ringrift/{remote_path}"
            try:
                success = await self._rsync_push_with_retry(
                    host=node_info.vast_ssh_host,
                    remote_path=vast_path,
                    ssh_port=node_info.vast_ssh_port,
                    verify=True,
                )
                if success:
                    return True
            except Exception as e:
                logger.debug(f"[EloSync] Vast SSH push to {node_name} failed: {e}")

        return False

    def _recalculate_ratings_from_history_sync(self, conn: sqlite3.Connection) -> None:
        """
        Recalculate all ELO ratings from match history (sync version).

        Jan 2, 2026: Renamed from async _recalculate_ratings_from_history to
        sync _recalculate_ratings_from_history_sync for use within asyncio.to_thread().

        Jan 27, 2026: Added support for participant_ids JSON column. Newer matches
        store participants in JSON array instead of participant_a/participant_b columns.

        This ensures win/loss conservation after merging databases.
        Replays all matches chronologically to rebuild accurate ratings.
        """
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

        # Jan 5, 2026: Detect schema variant for this database
        schema = self._detect_schema_variant(conn)
        model_a_col = schema["model_a"]
        model_b_col = schema["model_b"]

        # Jan 27, 2026: Also select participant_ids JSON and winner_id columns for newer matches
        # Newer matches store participants in JSON array and winner in winner_id, old columns may be NULL
        cur.execute(f"""
            SELECT {model_a_col}, {model_b_col}, winner, board_type, num_players, timestamp, participant_ids, winner_id
            FROM match_history
            WHERE winner IS NOT NULL OR winner_id IS NOT NULL
            ORDER BY timestamp
        """)
        matches = cur.fetchall()

        # Initialize ratings storage
        ratings = defaultdict(lambda: {
            "rating": INITIAL_RATING,
            "games_played": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
        })

        # Replay all matches
        # Jan 27, 2026: Handle both old format (participant_a/b + winner columns) and
        # new format (participant_ids JSON + winner_id)
        skipped_matches = 0
        for p_a, p_b, winner, board_type, num_players, _ts, participant_ids_json, winner_id in matches:
            # If old columns are NULL, try to parse from participant_ids JSON
            if (p_a is None or p_b is None) and participant_ids_json:
                try:
                    participant_ids = json.loads(participant_ids_json)
                    if isinstance(participant_ids, list) and len(participant_ids) >= 2:
                        p_a = participant_ids[0]
                        p_b = participant_ids[1]
                except (json.JSONDecodeError, TypeError, IndexError):
                    pass

            # If winner is NULL, try winner_id (newer format)
            if winner is None and winner_id:
                winner = winner_id

            # Skip if we still don't have valid participants or winner
            if not p_a or not p_b or not winner:
                skipped_matches += 1
                continue

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
        logger.info(
            f"Recalculated {len(ratings)} ratings from {len(matches)} matches"
            + (f" ({skipped_matches} skipped - missing participants)" if skipped_matches else "")
        )

    async def push_new_matches(self, matches: list[dict]) -> int:
        """
        Push new matches to the cluster.
        Call after playing games locally.
        Returns number of matches inserted locally.
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
                self._elo_state.sync_errors.append(f"{datetime.now().isoformat()}: {e}")

        self._save_elo_state()
        return inserted

    def _insert_matches_locally(self, matches: list[dict]) -> int:
        """Insert matches into local database.

        December 29, 2025: Optimized to use bulk insert with executemany()
        instead of N+1 individual INSERT statements.
        """
        if not self.db_path.exists():
            return 0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get existing game_ids in one query
        cursor.execute("SELECT game_id FROM match_history WHERE game_id IS NOT NULL")
        existing = {row[0] for row in cursor.fetchall()}

        # Filter to only new matches (avoid N lookups in loop)
        new_matches = []
        for match in matches:
            game_id = match.get("game_id")
            if game_id and game_id in existing:
                continue
            # Jan 13, 2026: Extract harness_type from participant_id or use provided value
            harness_type = match.get("harness_type")
            if not harness_type:
                # Try to extract from composite participant ID
                harness_type = extract_harness_type(match.get("participant_a", ""))
            new_matches.append((
                match["participant_a"], match["participant_b"], match["board_type"],
                match["num_players"], match.get("winner"), match.get("game_length"),
                match.get("duration_sec"), match.get("timestamp", time.time()),
                match.get("tournament_id"), game_id, match.get("metadata"),
                match.get("worker"), harness_type
            ))

        # Bulk insert using executemany (much faster than N individual INSERTs)
        if new_matches:
            cursor.executemany("""
                INSERT INTO match_history
                (participant_a, participant_b, board_type, num_players, winner,
                 game_length, duration_sec, timestamp, tournament_id, game_id, metadata, worker, harness_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, new_matches)

        conn.commit()
        conn.close()
        return len(new_matches)

    async def _push_matches_to_node(self, node: SyncNodeInfo, matches: list[dict]) -> None:
        """Push matches to a specific node."""
        host = node.tailscale_ip or node.ssh_host
        if not host:
            return

        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(matches, f)
            temp_file = f.name

        try:
            # Copy and merge on remote (December 29, 2025: optimized to use executemany)
            # Jan 13, 2026: Added harness_type extraction and preservation
            result = await asyncio.create_subprocess_exec(
                "ssh", "-o", "ConnectTimeout=10", host,
                """python3 -c "
import json
import sqlite3
import time
import re

def extract_harness_type(participant_id):
    '''Extract harness type from composite participant ID (model:harness:config).'''
    if not participant_id or ':' not in participant_id:
        return None
    parts = participant_id.split(':')
    if len(parts) >= 2:
        return parts[1]
    return None

matches = json.load(open('/dev/stdin'))
db = sqlite3.connect('/root/ringrift/ai-service/data/unified_elo.db')
cur = db.cursor()
cur.execute('SELECT game_id FROM match_history WHERE game_id IS NOT NULL')
existing = {r[0] for r in cur.fetchall()}

# Filter to new matches and prepare for bulk insert
new_matches = []
for m in matches:
    gid = m.get('game_id')
    if gid and gid in existing:
        continue
    # Jan 13, 2026: Extract harness_type from participant_id or use provided value
    harness_type = m.get('harness_type')
    if not harness_type:
        harness_type = extract_harness_type(m.get('participant_a', ''))
    new_matches.append((
        m['participant_a'], m['participant_b'], m['board_type'], m['num_players'],
        m.get('winner'), m.get('timestamp', time.time()), gid, harness_type
    ))
    if gid:
        existing.add(gid)

# Bulk insert using executemany
if new_matches:
    cur.executemany('''INSERT INTO match_history
        (participant_a, participant_b, board_type, num_players, winner, timestamp, game_id, harness_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', new_matches)

db.commit()
print(f'Inserted {len(new_matches)} matches')
"
""",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            with open(temp_file, "rb") as f:
                stdout, _ = await asyncio.wait_for(
                    result.communicate(input=f.read()),
                    timeout=30
                )

            logger.info(f"Push result: {stdout.decode().strip()}")
        finally:
            os.unlink(temp_file)

    def get_status(self) -> dict[str, Any]:
        """Get current sync status."""
        base_status = super().get_status()
        base_status.update({
            "initialized": getattr(self, "_initialized", False),
            "local_matches": self._elo_state.local_record_count,
            "last_sync": self._elo_state.last_sync_timestamp,
            "synced_from": self._elo_state.synced_from,
            "db_hash": self._elo_state.local_hash,
            "nodes_known": len(self.nodes),
            "nodes_list": list(self.nodes.keys()),
            "nodes_details": {
                name: {
                    "tailscale_ip": node.tailscale_ip,
                    "ssh_host": node.ssh_host,
                    "http_url": node.http_url,
                }
                for name, node in list(self.nodes.items())[:10]  # Limit output
            },
            "coordinator": self.coordinator_host,
            "recent_errors": self._elo_state.sync_errors[-5:],
            "total_syncs": self._elo_state.total_syncs,
            "successful_syncs": self._elo_state.successful_syncs,
        })
        return base_status


# Singleton instance for easy access
_sync_manager: EloSyncManager | None = None


def get_elo_sync_manager(
    db_path: Path | None = None,
    coordinator_host: str = "nebius-backbone-1"
) -> EloSyncManager:
    """Get or create the singleton EloSyncManager instance."""
    global _sync_manager
    if _sync_manager is None:
        _sync_manager = EloSyncManager(
            db_path=db_path or DEFAULT_DB_PATH,
            coordinator_host=coordinator_host
        )
    return _sync_manager


def reset_elo_sync_manager() -> None:
    """Reset the singleton (for testing)."""
    global _sync_manager
    _sync_manager = None


async def sync_elo_after_games(matches: list[dict]) -> int:
    """
    Convenience function to sync after playing games.
    Call this after each batch of games.
    """
    manager = get_elo_sync_manager()
    if not manager._elo_state.local_record_count:
        await manager.initialize()
    return await manager.push_new_matches(matches)


async def ensure_elo_synced() -> bool:
    """
    Convenience function to ensure database is synced.
    Call before training or making Elo-based decisions.
    """
    manager = get_elo_sync_manager()
    if not manager._elo_state.local_record_count:
        await manager.initialize()
    return await manager.sync_with_cluster()
