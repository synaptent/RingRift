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

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "unified_elo.db"
SYNC_STATE_PATH = Path(__file__).parent.parent.parent / "data" / "elo_sync_state.json"


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
        if SYNC_STATE_PATH.exists():
            self._load_elo_state()

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
                match_id = match_dict.get("game_id") or \
                    f"{match_dict.get('participant_a')}|{match_dict.get('participant_b')}|{match_dict.get('timestamp')}"

                if match_id not in existing_ids:
                    # Insert new match
                    cols = ", ".join(columns)
                    placeholders = ", ".join(["?" for _ in columns])
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
                self._elo_state.merge_conflicts += inserted
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

    # =========================================================================
    # Elo-specific methods
    # =========================================================================

    async def initialize(self) -> None:
        """Initialize the sync manager."""
        self._load_elo_state()
        await self.discover_nodes()
        # Add known Vast instances
        for name, info in self.VAST_INSTANCES.items():
            if name not in self.nodes:
                self.nodes[name] = SyncNodeInfo(
                    name=name,
                    vast_ssh_host=info["host"],
                    vast_ssh_port=info["port"],
                    remote_db_path=self._get_remote_db_path(),
                )
        self._update_local_stats()
        logger.info(f"EloSyncManager initialized: {self._elo_state.local_record_count} local matches")

    async def start_background_sync(self) -> None:
        """Start background sync loop."""
        await self.start()

    async def stop_background_sync(self) -> None:
        """Stop background sync loop."""
        await self.stop()

    async def sync_with_cluster(self) -> bool:
        """Synchronize with cluster nodes. Returns True if any sync succeeded."""
        self._elo_state.total_syncs += 1
        results = await super().sync_with_cluster()
        success = any(results.values())
        if success:
            self._elo_state.successful_syncs += 1
            self._save_elo_state()
        return success

    async def _recalculate_ratings_from_history(self, conn: sqlite3.Connection) -> None:
        """
        Recalculate all ELO ratings from match history.

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

        # Get all matches ordered by timestamp
        cur.execute("""
            SELECT participant_a, participant_b, winner, board_type, num_players, timestamp
            FROM match_history
            WHERE winner IS NOT NULL
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
            game_id = match.get("game_id")
            if game_id and game_id in existing:
                continue

            cursor.execute("""
                INSERT INTO match_history
                (participant_a, participant_b, board_type, num_players, winner,
                 game_length, duration_sec, timestamp, tournament_id, game_id, metadata, worker)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                match["participant_a"], match["participant_b"], match["board_type"],
                match["num_players"], match.get("winner"), match.get("game_length"),
                match.get("duration_sec"), match.get("timestamp", time.time()),
                match.get("tournament_id"), game_id, match.get("metadata"),
                match.get("worker")
            ))
            inserted += 1
            if game_id:
                existing.add(game_id)

        conn.commit()
        conn.close()
        return inserted

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
            # Copy and merge on remote
            result = await asyncio.create_subprocess_exec(
                "ssh", "-o", "ConnectTimeout=10", host,
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
            "local_matches": self._elo_state.local_record_count,
            "last_sync": self._elo_state.last_sync_timestamp,
            "synced_from": self._elo_state.synced_from,
            "db_hash": self._elo_state.local_hash,
            "nodes_known": len(self.nodes),
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
