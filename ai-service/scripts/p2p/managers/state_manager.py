"""StateManager: SQLite persistence for P2P orchestrator state.

Extracted from p2p_orchestrator.py for better modularity.
Handles database initialization, state loading/saving, and cluster epoch tracking.

Events emitted:
- STATE_PERSISTED: When state is saved to database
- EPOCH_ADVANCED: When cluster epoch is incremented (leader change)
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

# Dec 28, 2025: Phase 7 - Peer health state persistence
@dataclass
class PeerHealthState:
    """Persisted peer health state for recovery across restarts.

    December 2025: Part of Phase 7 cluster availability improvements.
    Prevents 'amnesia' after P2P restart by preserving peer health history.
    """

    node_id: str
    state: str  # "alive", "suspect", "dead", "retired"
    failure_count: int = 0
    gossip_failure_count: int = 0
    last_seen: float = 0.0
    last_failure: float = 0.0
    circuit_state: str = "closed"  # "closed", "open", "half_open"
    circuit_opened_at: float = 0.0
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "state": self.state,
            "failure_count": self.failure_count,
            "gossip_failure_count": self.gossip_failure_count,
            "last_seen": self.last_seen,
            "last_failure": self.last_failure,
            "circuit_state": self.circuit_state,
            "circuit_opened_at": self.circuit_opened_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PeerHealthState":
        """Create from dictionary."""
        return cls(
            node_id=data.get("node_id", ""),
            state=data.get("state", "alive"),
            failure_count=data.get("failure_count", 0),
            gossip_failure_count=data.get("gossip_failure_count", 0),
            last_seen=data.get("last_seen", 0.0),
            last_failure=data.get("last_failure", 0.0),
            circuit_state=data.get("circuit_state", "closed"),
            circuit_opened_at=data.get("circuit_opened_at", 0.0),
            updated_at=data.get("updated_at", time.time()),
        )


# Event emission helper (optional dependency)
_event_bus = None


def _get_event_bus():
    """Lazy-load event bus to avoid circular imports."""
    global _event_bus
    if _event_bus is None:
        try:
            from app.coordination.event_router import get_event_bus
            _event_bus = get_event_bus()
        except ImportError:
            _event_bus = False  # Mark as unavailable
    return _event_bus if _event_bus else None


def _safe_emit_event(event_type: str, payload: dict) -> bool:
    """Safely emit an event, returning False if unavailable."""
    bus = _get_event_bus()
    if bus is None:
        return False
    try:
        from app.distributed.data_events import DataEventType, DataEvent
        from app.core.async_context import fire_and_forget

        event = DataEvent(
            event_type=DataEventType(event_type),
            payload=payload,
            source="StateManager",
        )
        # Use fire_and_forget to properly handle the async publish
        coro = bus.publish(event)
        try:
            fire_and_forget(coro)
        except (AttributeError, RuntimeError, TypeError) as e:
            # AttributeError: coro.close() not available
            # RuntimeError: event bus not initialized
            # TypeError: coro has wrong type
            logger.debug(f"Event emission error for {event_type}: {e}")
            coro.close()
            return False
        return True
    except Exception as e:
        logger.debug(f"Failed to emit {event_type} event: {e}")
        return False

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

    # Phase 15.1.1 (Dec 2025): Fenced lease tokens for split-brain protection
    # These must be persisted to prevent stale leaders after restart
    lease_epoch: int = 0  # Monotonic epoch, never decreases
    fence_token: str = ""  # Current fence token: node_id:epoch:timestamp
    last_seen_epoch: int = 0  # Highest epoch seen from any leader


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
        # Dec 30, 2025: Track persistence failures for observability
        self._persistence_failures: int = 0
        self._last_persistence_error: str | None = None

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

    @contextlib.contextmanager
    def _db_connection(self, *, read_only: bool = False):
        """Context manager for database connections with guaranteed cleanup.

        Args:
            read_only: If True, uses READ_TIMEOUT instead of WRITE_TIMEOUT.

        Yields:
            sqlite3.Connection with proper settings configured.

        Example:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM peers")
        """
        timeout = SQLiteDefaults.READ_TIMEOUT if read_only else SQLiteDefaults.WRITE_TIMEOUT
        conn = sqlite3.connect(str(self.db_path), timeout=timeout)
        try:
            busy_timeout_ms = int(timeout * 1000)
            conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
            yield conn
        finally:
            conn.close()

    def _invalidate_stale_lease(
        self, leader_state: PersistedLeaderState, node_id: str
    ) -> None:
        """Invalidate stale leader lease on startup.

        P0.5 Dec 2025: After restart, node may read stale lease from state file
        and think it's still leader. This forces re-election on startup.

        Args:
            leader_state: The loaded leader state to potentially invalidate
            node_id: This node's ID

        Behavior:
            - If this node was the leader (leader_id == node_id), invalidate
            - If the lease has expired, invalidate
            - Otherwise, preserve state (may be follower with valid leader info)
        """
        now = time.time()
        was_leader = leader_state.leader_id == node_id
        lease_expired = leader_state.leader_lease_expires < now

        if was_leader or lease_expired:
            # Force re-election - don't trust old lease after restart
            old_leader = leader_state.leader_id
            old_expiry = leader_state.leader_lease_expires

            leader_state.leader_id = ""
            leader_state.leader_lease_id = ""
            leader_state.leader_lease_expires = 0.0
            leader_state.last_lease_renewal = 0.0
            leader_state.role = "follower"

            if was_leader:
                logger.warning(
                    f"P0.5: Invalidated stale leader claim on restart. "
                    f"Was leader, forcing re-election."
                )
            elif lease_expired:
                logger.info(
                    f"P0.5: Invalidated expired leader lease on startup. "
                    f"Old leader: {old_leader}, expired at: {old_expiry:.0f}, "
                    f"now: {now:.0f}"
                )

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
        with self._db_connection() as conn:
            cursor = conn.cursor()

            # Enable WAL mode for concurrent readers/writers
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")

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

            # Dec 28, 2025 (Phase 7): Peer health history table for state persistence
            # Preserves peer health across P2P restarts to prevent amnesia
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS peer_health_history (
                    node_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL DEFAULT 'alive',
                    failure_count INTEGER DEFAULT 0,
                    gossip_failure_count INTEGER DEFAULT 0,
                    last_seen REAL DEFAULT 0,
                    last_failure REAL DEFAULT 0,
                    circuit_state TEXT DEFAULT 'closed',
                    circuit_opened_at REAL DEFAULT 0,
                    updated_at REAL NOT NULL
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_peer_health_state
                ON peer_health_history(state, updated_at)
            """)

            conn.commit()

    def load_state(self, node_id: str) -> PersistedState:
        """Load persisted state from database.

        Args:
            node_id: Current node's ID (to exclude self from peers)

        Returns:
            PersistedState containing peers, jobs, and leader state
        """
        state = PersistedState()

        try:
            with self._db_connection(read_only=True) as conn:
                cursor = conn.cursor()

                # Load peers
                cursor.execute("SELECT node_id, info_json FROM peers")
                for row in cursor.fetchall():
                    try:
                        if row[0] == node_id:
                            continue
                        state.peers[row[0]] = json.loads(row[1])
                    except (json.JSONDecodeError, TypeError) as e:
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
                    with contextlib.suppress(ValueError, TypeError):
                        state.leader_state.leader_lease_expires = float(raw_lease_expires)

                if raw_last_renewal := state_rows.get("last_lease_renewal"):
                    with contextlib.suppress(ValueError, TypeError):
                        state.leader_state.last_lease_renewal = float(raw_last_renewal)

                if raw_role := state_rows.get("role"):
                    state.leader_state.role = str(raw_role)

                if raw_grant_leader := state_rows.get("voter_grant_leader_id"):
                    state.leader_state.voter_grant_leader_id = str(raw_grant_leader)

                if raw_grant_lease := state_rows.get("voter_grant_lease_id"):
                    state.leader_state.voter_grant_lease_id = str(raw_grant_lease)

                if raw_grant_expires := state_rows.get("voter_grant_expires"):
                    with contextlib.suppress(ValueError, TypeError):
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

                # P0.5 Dec 2025: Invalidate stale leader lease on restart
                # Without this, node may think it's still leader after restart
                # even if a new leader was elected during downtime
                self._invalidate_stale_lease(state.leader_state, node_id)

                logger.info(f"Loaded state: {len(state.peers)} peers, {len(state.jobs)} jobs")

        except sqlite3.Error as e:
            logger.error(f"Failed to load state: {e}")

        return state

    def clean_stale_state(
        self,
        state: PersistedState,
        *,
        stale_job_threshold_seconds: float = 86400.0,
        stale_peer_threshold_seconds: float = 300.0,
    ) -> tuple[int, int]:
        """Remove stale jobs and peers from the loaded state.

        Call this after validate_loaded_state() to clean up before using state.

        Args:
            state: The PersistedState to clean (modified in place)
            stale_job_threshold_seconds: Max age for jobs
            stale_peer_threshold_seconds: Max time since last heartbeat

        Returns:
            Tuple of (jobs_removed, peers_removed)
        """
        now = time.time()
        jobs_removed = 0
        peers_removed = 0

        # Remove stale jobs
        original_jobs = state.jobs
        state.jobs = [
            job for job in original_jobs
            if not job.get("started_at")
            or (now - job["started_at"]) <= stale_job_threshold_seconds
        ]
        jobs_removed = len(original_jobs) - len(state.jobs)

        # Remove stale peers
        peers_to_remove = []
        for node_id, peer_info in state.peers.items():
            last_heartbeat = peer_info.get("last_heartbeat", 0)
            last_seen = peer_info.get("last_seen", last_heartbeat)
            last_contact = max(last_heartbeat, last_seen)

            if last_contact and (now - last_contact) > stale_peer_threshold_seconds:
                peers_to_remove.append(node_id)

        for node_id in peers_to_remove:
            del state.peers[node_id]
        peers_removed = len(peers_to_remove)

        if jobs_removed or peers_removed:
            logger.info(
                f"[StateManager] Cleaned stale state: "
                f"removed {jobs_removed} jobs, {peers_removed} peers"
            )

        return jobs_removed, peers_removed

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
        try:
            with self._db_connection() as conn:
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

                # Emit STATE_PERSISTED event (Dec 2025)
                _safe_emit_event("state_persisted", {
                    "peer_count": len(peers) - (1 if node_id in peers else 0),
                    "job_count": len(jobs),
                    "leader_id": leader_state.leader_id,
                    "cluster_epoch": self._cluster_epoch,
                })
        except sqlite3.Error as e:
            # Dec 30, 2025: Enhanced error handling with tracking and event emission
            self._persistence_failures += 1
            self._last_persistence_error = str(e)
            logger.error(
                f"Failed to save state (attempt {self._persistence_failures}): {e}"
            )
            # Emit event for observability and coordination
            _safe_emit_event("state_persistence_failed", {
                "error": str(e),
                "failure_count": self._persistence_failures,
                "db_path": str(self.db_path),
            })

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
                        job.get("engine_mode", "gumbel-mcts"),  # Jan 2026: Default to high-quality GPU mode
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
        try:
            with self._db_connection(read_only=True) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM config WHERE key = 'cluster_epoch'")
                row = cursor.fetchone()
                if row:
                    self._cluster_epoch = int(row[0])
                    logger.info(f"Loaded cluster epoch: {self._cluster_epoch}")
        except (sqlite3.Error, ValueError, TypeError) as e:
            if self.verbose:
                logger.debug(f"Error loading cluster epoch: {e}")
        return self._cluster_epoch

    def save_cluster_epoch(self) -> None:
        """Save cluster epoch to database."""
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO config (key, value) VALUES ('cluster_epoch', ?)",
                    (str(self._cluster_epoch),),
                )
                conn.commit()
        except sqlite3.Error as e:
            if self.verbose:
                logger.debug(f"Error saving cluster epoch: {e}")

    def increment_cluster_epoch(self) -> int:
        """Increment cluster epoch (called on leader change).

        Returns:
            The new cluster epoch value
        """
        with self._lock:
            old_epoch = self._cluster_epoch
            self._cluster_epoch += 1
            self.save_cluster_epoch()
            logger.info(f"Incremented cluster epoch to {self._cluster_epoch}")

            # Emit EPOCH_ADVANCED event (Dec 2025)
            _safe_emit_event("epoch_advanced", {
                "old_epoch": old_epoch,
                "new_epoch": self._cluster_epoch,
            })

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
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to delete job {job_id}: {e}")

    def update_job_status(self, job_id: str, status: str) -> None:
        """Update a job's status in the database.

        Args:
            job_id: The job ID to update
            status: The new status value
        """
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE jobs SET status = ? WHERE job_id = ?",
                    (status, job_id),
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to update job {job_id} status: {e}")

    def clear_stale_jobs(self) -> int:
        """Clear jobs that are no longer running.

        Returns:
            Number of jobs cleared
        """
        cleared = 0
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM jobs WHERE status != 'running'")
                cleared = cursor.rowcount
                conn.commit()
                if cleared > 0:
                    logger.info(f"Cleared {cleared} stale jobs from database")
        except sqlite3.Error as e:
            logger.error(f"Failed to clear stale jobs: {e}")
        return cleared

    # =========================================================================
    # Startup State Validation (December 2025 P2P Hardening)
    # =========================================================================

    def validate_loaded_state(
        self, state: PersistedState, max_job_age_hours: float = 24.0, max_peer_stale_seconds: float = 300.0
    ) -> tuple[bool, list[str]]:
        """Validate loaded state for consistency and staleness.

        Called on startup to detect and report issues with persisted state.
        Does NOT modify state - use clear_stale_jobs/peers for cleanup.

        Args:
            state: The loaded PersistedState to validate
            max_job_age_hours: Jobs older than this are considered stale (default 24h)
            max_peer_stale_seconds: Peers without heartbeat for this long are stale (default 5min)

        Returns:
            Tuple of (is_valid, list_of_issues)
            is_valid is True if no critical issues found
        """
        issues: list[str] = []
        now = time.time()
        max_job_age_seconds = max_job_age_hours * 3600

        # 1. Check for stale jobs (started too long ago)
        stale_job_count = 0
        for job in state.jobs:
            started_at = job.get("started_at", 0)
            if started_at and (now - started_at) > max_job_age_seconds:
                stale_job_count += 1
                if stale_job_count <= 3:  # Only log first 3
                    job_id = job.get("job_id", "unknown")
                    age_hours = (now - started_at) / 3600
                    issues.append(f"Stale job {job_id}: started {age_hours:.1f}h ago")

        if stale_job_count > 3:
            issues.append(f"  ... and {stale_job_count - 3} more stale jobs")

        # 2. Check for stale peers (no heartbeat in a while)
        stale_peer_count = 0
        for peer_id, peer_info in state.peers.items():
            last_heartbeat = peer_info.get("last_heartbeat", 0)
            if last_heartbeat and (now - last_heartbeat) > max_peer_stale_seconds:
                stale_peer_count += 1
                if stale_peer_count <= 3:  # Only log first 3
                    stale_minutes = (now - last_heartbeat) / 60
                    issues.append(f"Stale peer {peer_id}: no heartbeat in {stale_minutes:.1f}min")

        if stale_peer_count > 3:
            issues.append(f"  ... and {stale_peer_count - 3} more stale peers")

        # 3. Check leader lease validity
        leader_state = state.leader_state
        if leader_state.leader_id and leader_state.leader_lease_expires < now:
            expired_ago = (now - leader_state.leader_lease_expires) / 60
            issues.append(f"Leader lease expired {expired_ago:.1f}min ago (leader: {leader_state.leader_id})")

        # 4. Check voter configuration
        if not leader_state.voter_node_ids:
            issues.append("No voter nodes configured - leader election may fail")

        # Emit validation event
        _safe_emit_event("startup_state_validated", {
            "is_valid": len(issues) == 0,
            "issue_count": len(issues),
            "stale_jobs": stale_job_count,
            "stale_peers": stale_peer_count,
            "total_jobs": len(state.jobs),
            "total_peers": len(state.peers),
        })

        return len(issues) == 0, issues

    def clear_stale_peers(self, max_stale_seconds: float = 300.0) -> int:
        """Clear peers that haven't sent heartbeat recently.

        Args:
            max_stale_seconds: Peers without heartbeat for this long are cleared (default 5min)

        Returns:
            Number of peers cleared
        """
        cleared = 0
        now = time.time()
        cutoff = now - max_stale_seconds

        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM peers WHERE last_heartbeat < ? OR last_heartbeat IS NULL",
                    (cutoff,),
                )
                cleared = cursor.rowcount
                conn.commit()
                if cleared > 0:
                    logger.info(f"Cleared {cleared} stale peers from database (no heartbeat in {max_stale_seconds}s)")
        except sqlite3.Error as e:
            logger.error(f"Failed to clear stale peers: {e}")
        return cleared

    def clear_stale_jobs_by_age(self, max_age_hours: float = 24.0) -> int:
        """Clear jobs that are older than the specified age.

        Args:
            max_age_hours: Jobs started more than this many hours ago are cleared

        Returns:
            Number of jobs cleared
        """
        cleared = 0
        now = time.time()
        cutoff = now - (max_age_hours * 3600)

        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM jobs WHERE started_at < ? AND started_at > 0",
                    (cutoff,),
                )
                cleared = cursor.rowcount
                conn.commit()
                if cleared > 0:
                    logger.info(f"Cleared {cleared} stale jobs from database (older than {max_age_hours}h)")
        except sqlite3.Error as e:
            logger.error(f"Failed to clear stale jobs by age: {e}")
        return cleared

    # =========================================================================
    # Health Check (December 2025)
    # =========================================================================

    def health_check(self):
        """Check health status of StateManager.

        Returns:
            HealthCheckResult with status, operations metrics, and error info
        """
        # Import from contracts (zero dependencies)
        from app.coordination.contracts import HealthCheckResult, CoordinatorStatus

        status = CoordinatorStatus.RUNNING
        is_healthy = True
        errors_count = 0
        last_error: str | None = None
        peer_count = 0
        job_count = 0

        # Check database connectivity using context manager for guaranteed cleanup
        try:
            with self._db_connection(read_only=True) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM peers")
                peer_count = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM jobs WHERE status = 'running'")
                job_count = cursor.fetchone()[0]
        except sqlite3.Error as e:
            status = CoordinatorStatus.ERROR
            is_healthy = False
            errors_count = 1
            last_error = f"Database connection failed: {e}"

        # Check if database file exists
        if not self.db_path.exists():
            status = CoordinatorStatus.ERROR
            is_healthy = False
            errors_count += 1
            last_error = f"Database file not found: {self.db_path}"

        return HealthCheckResult(
            healthy=is_healthy,
            status=status if isinstance(status, str) else status,
            message=last_error or "StateManager healthy",
            details={
                "operations_count": peer_count + job_count,
                "errors_count": errors_count,
                "peer_count": peer_count,
                "job_count": job_count,
                "cluster_epoch": self._cluster_epoch,
                "db_path": str(self.db_path),
            },
        )

    # =========================================================================
    # Peer Health Persistence (December 2025, Phase 7)
    # =========================================================================

    def save_peer_health(self, peer_health: PeerHealthState) -> bool:
        """Save peer health state to database.

        Part of Phase 7 cluster availability improvements.
        Preserves peer health history across P2P restarts.

        Args:
            peer_health: The peer health state to persist

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO peer_health_history
                    (node_id, state, failure_count, gossip_failure_count,
                     last_seen, last_failure, circuit_state, circuit_opened_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        peer_health.node_id,
                        peer_health.state,
                        peer_health.failure_count,
                        peer_health.gossip_failure_count,
                        peer_health.last_seen,
                        peer_health.last_failure,
                        peer_health.circuit_state,
                        peer_health.circuit_opened_at,
                        time.time(),
                    ),
                )
                conn.commit()
                return True
        except sqlite3.Error as e:
            logger.error(f"Failed to save peer health for {peer_health.node_id}: {e}")
            return False

    def save_peer_health_batch(self, peer_healths: list[PeerHealthState]) -> int:
        """Save multiple peer health states in a single transaction.

        Args:
            peer_healths: List of peer health states to persist

        Returns:
            Number of records saved successfully
        """
        if not peer_healths:
            return 0

        saved = 0
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                now = time.time()
                for ph in peer_healths:
                    try:
                        cursor.execute(
                            """
                            INSERT OR REPLACE INTO peer_health_history
                            (node_id, state, failure_count, gossip_failure_count,
                             last_seen, last_failure, circuit_state, circuit_opened_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                ph.node_id,
                                ph.state,
                                ph.failure_count,
                                ph.gossip_failure_count,
                                ph.last_seen,
                                ph.last_failure,
                                ph.circuit_state,
                                ph.circuit_opened_at,
                                now,
                            ),
                        )
                        saved += 1
                    except sqlite3.Error as e:
                        logger.warning(f"Failed to save peer health for {ph.node_id}: {e}")

                conn.commit()
                if saved > 0:
                    logger.debug(f"[StateManager] Saved {saved} peer health records")
        except sqlite3.Error as e:
            logger.error(f"Failed to save peer health batch: {e}")

        return saved

    def load_peer_health(self, node_id: str) -> PeerHealthState | None:
        """Load peer health state from database.

        Args:
            node_id: The node ID to load health state for

        Returns:
            PeerHealthState if found, None otherwise
        """
        try:
            with self._db_connection(read_only=True) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT node_id, state, failure_count, gossip_failure_count,
                           last_seen, last_failure, circuit_state, circuit_opened_at, updated_at
                    FROM peer_health_history
                    WHERE node_id = ?
                    """,
                    (node_id,),
                )
                row = cursor.fetchone()
                if row:
                    return PeerHealthState(
                        node_id=row[0],
                        state=row[1],
                        failure_count=row[2] or 0,
                        gossip_failure_count=row[3] or 0,
                        last_seen=row[4] or 0.0,
                        last_failure=row[5] or 0.0,
                        circuit_state=row[6] or "closed",
                        circuit_opened_at=row[7] or 0.0,
                        updated_at=row[8] or time.time(),
                    )
                return None
        except sqlite3.Error as e:
            logger.error(f"Failed to load peer health for {node_id}: {e}")
            return None

    def load_all_peer_health(
        self, max_age_seconds: float = 3600.0
    ) -> dict[str, PeerHealthState]:
        """Load all peer health states from database.

        Args:
            max_age_seconds: Maximum age of records to load (default: 1 hour).
                            Records older than this are considered stale.

        Returns:
            Dict mapping node_id to PeerHealthState
        """
        result: dict[str, PeerHealthState] = {}
        try:
            cutoff = time.time() - max_age_seconds
            with self._db_connection(read_only=True) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT node_id, state, failure_count, gossip_failure_count,
                           last_seen, last_failure, circuit_state, circuit_opened_at, updated_at
                    FROM peer_health_history
                    WHERE updated_at > ?
                    ORDER BY updated_at DESC
                    """,
                    (cutoff,),
                )
                for row in cursor.fetchall():
                    result[row[0]] = PeerHealthState(
                        node_id=row[0],
                        state=row[1],
                        failure_count=row[2] or 0,
                        gossip_failure_count=row[3] or 0,
                        last_seen=row[4] or 0.0,
                        last_failure=row[5] or 0.0,
                        circuit_state=row[6] or "closed",
                        circuit_opened_at=row[7] or 0.0,
                        updated_at=row[8] or time.time(),
                    )
                logger.info(f"[StateManager] Loaded {len(result)} peer health records")
        except sqlite3.Error as e:
            logger.error(f"Failed to load peer health: {e}")

        return result

    def clear_stale_peer_health(self, max_age_seconds: float = 86400.0) -> int:
        """Clear stale peer health records older than max_age_seconds.

        Args:
            max_age_seconds: Maximum age before clearing (default: 24 hours)

        Returns:
            Number of records cleared
        """
        cleared = 0
        try:
            cutoff = time.time() - max_age_seconds
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM peer_health_history WHERE updated_at < ?",
                    (cutoff,),
                )
                cleared = cursor.rowcount
                conn.commit()
                if cleared > 0:
                    logger.info(
                        f"[StateManager] Cleared {cleared} stale peer health records "
                        f"(older than {max_age_seconds/3600:.1f}h)"
                    )
        except sqlite3.Error as e:
            logger.error(f"Failed to clear stale peer health: {e}")

        return cleared

    def get_peer_health_summary(self) -> dict[str, Any]:
        """Get summary of peer health states.

        Returns:
            Dict with counts per state, circuit state, and staleness info
        """
        try:
            with self._db_connection(read_only=True) as conn:
                cursor = conn.cursor()

                # Count by state
                cursor.execute(
                    "SELECT state, COUNT(*) FROM peer_health_history GROUP BY state"
                )
                state_counts = dict(cursor.fetchall())

                # Count by circuit state
                cursor.execute(
                    "SELECT circuit_state, COUNT(*) FROM peer_health_history GROUP BY circuit_state"
                )
                circuit_counts = dict(cursor.fetchall())

                # Get total and stale counts
                cursor.execute("SELECT COUNT(*) FROM peer_health_history")
                total = cursor.fetchone()[0]

                cutoff_1h = time.time() - 3600
                cursor.execute(
                    "SELECT COUNT(*) FROM peer_health_history WHERE updated_at < ?",
                    (cutoff_1h,),
                )
                stale_1h = cursor.fetchone()[0]

                return {
                    "total_records": total,
                    "state_counts": state_counts,
                    "circuit_state_counts": circuit_counts,
                    "stale_count_1h": stale_1h,
                    "fresh_count": total - stale_1h,
                }
        except sqlite3.Error as e:
            logger.error(f"Failed to get peer health summary: {e}")
            return {
                "total_records": 0,
                "state_counts": {},
                "circuit_state_counts": {},
                "stale_count_1h": 0,
                "fresh_count": 0,
                "error": str(e),
            }
