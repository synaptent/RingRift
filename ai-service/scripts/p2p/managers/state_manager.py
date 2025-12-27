"""StateManager: SQLite persistence for P2P orchestrator state.

Extracted from p2p_orchestrator.py for better modularity.
Handles database initialization, state loading/saving, and cluster epoch tracking.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.config.coordination_defaults import SQLiteDefaults

if TYPE_CHECKING:
    from ..models import ClusterJob, NodeInfo
    from ..types import NodeRole

logger = logging.getLogger(__name__)


@dataclass
class PersistedLeaderState:
    """Leader election state that is persisted across restarts."""

    leader_id: str = ""
    leader_lease_id: str = ""
    leader_lease_expires: float = 0.0
    last_lease_renewal: float = 0.0
    role: str = "follower"  # NodeRole.value

    # Voter grant state (for voter quorum)
    voter_grant_leader_id: str = ""
    voter_grant_lease_id: str = ""
    voter_grant_expires: float = 0.0

    # Voter configuration (learned or persisted)
    voter_node_ids: list[str] = field(default_factory=list)
    voter_config_source: str = ""


@dataclass
class PersistedState:
    """All persisted orchestrator state."""

    peers: dict[str, dict[str, Any]] = field(default_factory=dict)  # node_id -> info dict
    jobs: list[dict[str, Any]] = field(default_factory=list)  # List of job dicts
    leader_state: PersistedLeaderState = field(default_factory=PersistedLeaderState)


class StateManager:
    """Manages SQLite persistence for P2P orchestrator state.

    Responsibilities:
    - Database initialization and schema management
    - Loading and saving peers, jobs, and leader state
    - Cluster epoch tracking for split-brain resolution
    - Thread-safe database operations

    Usage:
        state_mgr = StateManager(db_path)
        state_mgr.init_database()

        # Load state
        state = state_mgr.load_state()
        peers = state.peers
        jobs = state.jobs

        # Save state
        state_mgr.save_state(peers_dict, jobs_list, leader_state)

        # Cluster epoch
        epoch = state_mgr.get_cluster_epoch()
        state_mgr.increment_cluster_epoch()
    """

    def __init__(self, db_path: Path, verbose: bool = False):
        """Initialize the StateManager.

        Args:
            db_path: Path to the SQLite database file
            verbose: Enable verbose logging
        """
        self.db_path = db_path
        self.verbose = verbose
        self._cluster_epoch: int = 0
        self._lock = threading.Lock()

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _db_connect(self) -> sqlite3.Connection:
        """Create a database connection with proper settings.

        Uses WAL mode for concurrent access and extended timeout for busy handling.
        Timeouts are configurable via RINGRIFT_SQLITE_* environment variables.
        """
        conn = sqlite3.connect(str(self.db_path), timeout=SQLiteDefaults.WRITE_TIMEOUT)
        # Use WRITE_TIMEOUT * 1000 for busy_timeout (P2P state is critical infrastructure)
        busy_timeout_ms = int(SQLiteDefaults.WRITE_TIMEOUT * 1000)
        conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
        return conn

    def init_database(self) -> None:
        """Initialize SQLite database schema for state persistence.

        Creates tables for:
        - peers: Node information
        - jobs: Running jobs
        - state: Key-value state (leader, role, etc.)
        - metrics_history: Observability metrics
        - ab_tests: A/B testing experiments
        - ab_test_games: A/B test game results
        - peer_cache: Persistent peer storage with reputation
        - config: Cluster epoch and other persistent settings
        """
        conn = sqlite3.connect(str(self.db_path), timeout=SQLiteDefaults.WRITE_TIMEOUT)
        cursor = conn.cursor()

        try:
            # Enable WAL mode for concurrent readers/writers
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            busy_timeout_ms = int(SQLiteDefaults.WRITE_TIMEOUT * 1000)
            cursor.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")

            # Peers table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS peers (
                    node_id TEXT PRIMARY KEY,
                    host TEXT,
                    port INTEGER,
                    last_heartbeat REAL,
                    info_json TEXT
                )
            """)

            # Jobs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT,
                    node_id TEXT,
                    board_type TEXT,
                    num_players INTEGER,
                    engine_mode TEXT,
                    pid INTEGER,
                    started_at REAL,
                    status TEXT
                )
            """)

            # State table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS state (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

            # Metrics history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    metric_type TEXT NOT NULL,
                    board_type TEXT,
                    num_players INTEGER,
                    value REAL NOT NULL,
                    metadata TEXT
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_type_time
                ON metrics_history(metric_type, timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_config
                ON metrics_history(board_type, num_players, timestamp)
            """)

            # A/B Testing tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ab_tests (
                    test_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    board_type TEXT NOT NULL,
                    num_players INTEGER NOT NULL,
                    model_a TEXT NOT NULL,
                    model_b TEXT NOT NULL,
                    target_games INTEGER DEFAULT 100,
                    confidence_threshold REAL DEFAULT 0.95,
                    status TEXT DEFAULT 'running',
                    winner TEXT,
                    created_at REAL NOT NULL,
                    completed_at REAL,
                    metadata TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ab_test_games (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL,
                    game_id TEXT NOT NULL,
                    model_a_result TEXT NOT NULL,
                    model_a_score REAL NOT NULL,
                    model_b_score REAL NOT NULL,
                    game_length INTEGER,
                    played_at REAL NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (test_id) REFERENCES ab_tests(test_id)
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ab_games_test
                ON ab_test_games(test_id, played_at)
            """)

            # Peer cache table for persistent peer storage
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS peer_cache (
                    node_id TEXT PRIMARY KEY,
                    host TEXT NOT NULL,
                    port INTEGER NOT NULL,
                    tailscale_ip TEXT,
                    last_seen REAL,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    reputation_score REAL DEFAULT 0.5,
                    is_bootstrap_seed BOOLEAN DEFAULT FALSE
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_peer_cache_reputation
                ON peer_cache(reputation_score DESC, last_seen DESC)
            """)

            # Config table for cluster epoch and other persistent settings
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)

            conn.commit()
        finally:
            conn.close()

    def load_state(self, node_id: str) -> PersistedState:
        """Load persisted state from database.

        Args:
            node_id: Current node's ID (to exclude self from peers)

        Returns:
            PersistedState containing peers, jobs, and leader state
        """
        state = PersistedState()
        conn = None

        try:
            conn = self._db_connect()
            cursor = conn.cursor()

            # Load peers
            cursor.execute("SELECT node_id, info_json FROM peers")
            for row in cursor.fetchall():
                try:
                    if row[0] == node_id:
                        continue
                    state.peers[row[0]] = json.loads(row[1])
                except Exception as e:
                    logger.error(f"Failed to load peer {row[0]}: {e}")

            # Load running jobs
            cursor.execute("SELECT * FROM jobs WHERE status = 'running'")
            for row in cursor.fetchall():
                state.jobs.append({
                    "job_id": row[0],
                    "job_type": row[1],
                    "node_id": row[2],
                    "board_type": row[3],
                    "num_players": row[4],
                    "engine_mode": row[5],
                    "pid": row[6],
                    "started_at": row[7],
                    "status": row[8],
                })

            # Load leader state
            cursor.execute("SELECT key, value FROM state")
            state_rows = {row[0]: row[1] for row in cursor.fetchall() if row and row[0]}

            state.leader_state.leader_id = state_rows.get("leader_id", "")

            if raw_lease_id := state_rows.get("leader_lease_id"):
                state.leader_state.leader_lease_id = raw_lease_id

            if raw_lease_expires := state_rows.get("leader_lease_expires"):
                with contextlib.suppress(Exception):
                    state.leader_state.leader_lease_expires = float(raw_lease_expires)

            if raw_last_renewal := state_rows.get("last_lease_renewal"):
                with contextlib.suppress(Exception):
                    state.leader_state.last_lease_renewal = float(raw_last_renewal)

            if raw_role := state_rows.get("role"):
                state.leader_state.role = str(raw_role)

            if raw_grant_leader := state_rows.get("voter_grant_leader_id"):
                state.leader_state.voter_grant_leader_id = str(raw_grant_leader)

            if raw_grant_lease := state_rows.get("voter_grant_lease_id"):
                state.leader_state.voter_grant_lease_id = str(raw_grant_lease)

            if raw_grant_expires := state_rows.get("voter_grant_expires"):
                with contextlib.suppress(Exception):
                    state.leader_state.voter_grant_expires = float(raw_grant_expires)

            # Load voter configuration
            if raw_voters := state_rows.get("voter_node_ids"):
                try:
                    parsed = json.loads(raw_voters)
                    if isinstance(parsed, list):
                        state.leader_state.voter_node_ids = [
                            str(v).strip() for v in parsed if str(v).strip()
                        ]
                except (json.JSONDecodeError, KeyError, IndexError, AttributeError):
                    state.leader_state.voter_node_ids = [
                        t.strip() for t in str(raw_voters).split(",") if t.strip()
                    ]

            if raw_config_source := state_rows.get("voter_config_source"):
                state.leader_state.voter_config_source = str(raw_config_source)

            logger.info(f"Loaded state: {len(state.peers)} peers, {len(state.jobs)} jobs")

        except Exception as e:
            logger.error(f"Failed to load state: {e}")
        finally:
            if conn:
                conn.close()

        return state

    def save_state(
        self,
        node_id: str,
        peers: dict[str, Any],
        jobs: dict[str, Any],
        leader_state: PersistedLeaderState,
        peers_lock: threading.Lock | None = None,
        jobs_lock: threading.Lock | None = None,
    ) -> None:
        """Save current state to database.

        Args:
            node_id: Current node's ID (to exclude self from peers)
            peers: Dict of node_id -> NodeInfo (or dict with to_dict method)
            jobs: Dict of job_id -> ClusterJob (or dict with to_dict method)
            leader_state: Current leader election state
            peers_lock: Optional lock for thread-safe peer access
            jobs_lock: Optional lock for thread-safe job access
        """
        conn = None
        try:
            conn = self._db_connect()
            cursor = conn.cursor()

            # Save peers (with optional locking)
            cursor.execute("DELETE FROM peers WHERE node_id = ?", (node_id,))

            if peers_lock:
                with peers_lock:
                    self._save_peers_unlocked(cursor, node_id, peers)
            else:
                self._save_peers_unlocked(cursor, node_id, peers)

            # Save jobs (with optional locking)
            if jobs_lock:
                with jobs_lock:
                    self._save_jobs_unlocked(cursor, jobs)
            else:
                self._save_jobs_unlocked(cursor, jobs)

            # Save leader state
            role_value = leader_state.role
            voter_node_ids_json = json.dumps(sorted(set(leader_state.voter_node_ids or [])))

            state_payload = [
                ("leader_id", leader_state.leader_id or ""),
                ("leader_lease_id", leader_state.leader_lease_id or ""),
                ("leader_lease_expires", str(float(leader_state.leader_lease_expires or 0.0))),
                ("last_lease_renewal", str(float(leader_state.last_lease_renewal or 0.0))),
                ("role", role_value),
                ("voter_node_ids", voter_node_ids_json),
                ("voter_config_source", leader_state.voter_config_source or ""),
                ("voter_grant_leader_id", leader_state.voter_grant_leader_id or ""),
                ("voter_grant_lease_id", leader_state.voter_grant_lease_id or ""),
                ("voter_grant_expires", str(float(leader_state.voter_grant_expires or 0.0))),
            ]
            cursor.executemany(
                "INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
                state_payload,
            )

            conn.commit()
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
        finally:
            if conn:
                conn.close()

    def _save_peers_unlocked(
        self, cursor: sqlite3.Cursor, node_id: str, peers: dict[str, Any]
    ) -> None:
        """Save peers without locking (caller must hold lock if needed)."""
        for peer_id, info in peers.items():
            if peer_id == node_id:
                continue
            # Handle both NodeInfo objects and dicts
            if hasattr(info, "to_dict"):
                info_dict = info.to_dict()
                host = info.host
                port = info.port
                last_heartbeat = info.last_heartbeat
            else:
                info_dict = info
                host = info.get("host", "")
                port = info.get("port", 0)
                last_heartbeat = info.get("last_heartbeat", 0.0)

            cursor.execute(
                """
                INSERT OR REPLACE INTO peers (node_id, host, port, last_heartbeat, info_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (peer_id, host, port, last_heartbeat, json.dumps(info_dict)),
            )

    def _save_jobs_unlocked(self, cursor: sqlite3.Cursor, jobs: dict[str, Any]) -> None:
        """Save jobs without locking (caller must hold lock if needed)."""
        for _job_id, job in jobs.items():
            # Handle both ClusterJob objects and dicts
            if hasattr(job, "job_type"):
                job_type = job.job_type.value if hasattr(job.job_type, "value") else str(job.job_type)
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO jobs
                    (job_id, job_type, node_id, board_type, num_players, engine_mode, pid, started_at, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job.job_id,
                        job_type,
                        job.node_id,
                        job.board_type,
                        job.num_players,
                        job.engine_mode,
                        job.pid,
                        job.started_at,
                        job.status,
                    ),
                )
            else:
                job_type = job.get("job_type", "selfplay")
                if hasattr(job_type, "value"):
                    job_type = job_type.value
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO jobs
                    (job_id, job_type, node_id, board_type, num_players, engine_mode, pid, started_at, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job.get("job_id"),
                        job_type,
                        job.get("node_id"),
                        job.get("board_type", "square8"),
                        job.get("num_players", 2),
                        job.get("engine_mode", "descent-only"),
                        job.get("pid", 0),
                        job.get("started_at", 0.0),
                        job.get("status", "running"),
                    ),
                )

    # =========================================================================
    # Cluster Epoch Management
    # =========================================================================

    def get_cluster_epoch(self) -> int:
        """Get the current cluster epoch."""
        return self._cluster_epoch

    def load_cluster_epoch(self) -> int:
        """Load cluster epoch from database.

        Returns:
            The loaded cluster epoch value
        """
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=SQLiteDefaults.READ_TIMEOUT)
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM config WHERE key = 'cluster_epoch'")
            row = cursor.fetchone()
            if row:
                self._cluster_epoch = int(row[0])
                logger.info(f"Loaded cluster epoch: {self._cluster_epoch}")
        except Exception as e:
            if self.verbose:
                logger.debug(f"Error loading cluster epoch: {e}")
        finally:
            if conn:
                conn.close()
        return self._cluster_epoch

    def save_cluster_epoch(self) -> None:
        """Save cluster epoch to database."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=SQLiteDefaults.WRITE_TIMEOUT)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO config (key, value) VALUES ('cluster_epoch', ?)",
                (str(self._cluster_epoch),),
            )
            conn.commit()
        except Exception as e:
            if self.verbose:
                logger.debug(f"Error saving cluster epoch: {e}")
        finally:
            if conn:
                conn.close()

    def increment_cluster_epoch(self) -> int:
        """Increment cluster epoch (called on leader change).

        Returns:
            The new cluster epoch value
        """
        with self._lock:
            self._cluster_epoch += 1
            self.save_cluster_epoch()
            logger.info(f"Incremented cluster epoch to {self._cluster_epoch}")
            return self._cluster_epoch

    def set_cluster_epoch(self, epoch: int) -> None:
        """Set the cluster epoch to a specific value.

        Args:
            epoch: The new epoch value
        """
        with self._lock:
            self._cluster_epoch = epoch
            self.save_cluster_epoch()

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def delete_job(self, job_id: str) -> None:
        """Delete a job from the database.

        Args:
            job_id: The job ID to delete
        """
        conn = None
        try:
            conn = self._db_connect()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to delete job {job_id}: {e}")
        finally:
            if conn:
                conn.close()

    def update_job_status(self, job_id: str, status: str) -> None:
        """Update a job's status in the database.

        Args:
            job_id: The job ID to update
            status: The new status value
        """
        conn = None
        try:
            conn = self._db_connect()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE jobs SET status = ? WHERE job_id = ?",
                (status, job_id),
            )
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to update job {job_id} status: {e}")
        finally:
            if conn:
                conn.close()

    def clear_stale_jobs(self) -> int:
        """Clear jobs that are no longer running.

        Returns:
            Number of jobs cleared
        """
        conn = None
        cleared = 0
        try:
            conn = self._db_connect()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM jobs WHERE status != 'running'")
            cleared = cursor.rowcount
            conn.commit()
            if cleared > 0:
                logger.info(f"Cleared {cleared} stale jobs from database")
        except Exception as e:
            logger.error(f"Failed to clear stale jobs: {e}")
        finally:
            if conn:
                conn.close()
        return cleared

    # =========================================================================
    # Health Check (December 2025)
    # =========================================================================

    def health_check(self) -> dict[str, Any]:
        """Check health status of StateManager.

        Returns:
            Dict with status, operations metrics, and error info
        """
        status = "healthy"
        errors_count = 0
        last_error: str | None = None
        peer_count = 0
        job_count = 0

        # Check database connectivity
        # December 27, 2025: Fixed connection leak - moved close to finally block
        conn = None
        try:
            conn = self._db_connect()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM peers")
            peer_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM jobs WHERE status = 'running'")
            job_count = cursor.fetchone()[0]
        except Exception as e:
            status = "unhealthy"
            errors_count = 1
            last_error = f"Database connection failed: {e}"
        finally:
            if conn:
                conn.close()

        # Check if database file exists
        if not self.db_path.exists():
            status = "unhealthy"
            errors_count += 1
            last_error = f"Database file not found: {self.db_path}"

        return {
            "status": status,
            "operations_count": peer_count + job_count,
            "errors_count": errors_count,
            "last_error": last_error,
            "peer_count": peer_count,
            "job_count": job_count,
            "cluster_epoch": self._cluster_epoch,
            "db_path": str(self.db_path),
        }
