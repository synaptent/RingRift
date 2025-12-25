"""
Unified Orchestrator Registry for RingRift AI.

Provides SQLite-based coordination between all orchestrators to prevent:
1. Multiple orchestrators running simultaneously without awareness
2. Race conditions in shared resource access
3. Stale orchestrator state affecting decisions

Features:
- Heartbeat-based liveness detection (stale orchestrators auto-expire)
- Role-based coordination (only one orchestrator per role)
- Cross-process visibility into all active orchestrators
- Graceful handoff when orchestrators restart

Architecture Relationship (December 2025):
-----------------------------------------
This module is part of a layered coordination architecture:

1. **TaskCoordinator** (:mod:`app.coordination.task_coordinator`)
   - Canonical for TASK ADMISSION CONTROL
   - Decides how many tasks can run based on limits/resources

2. **OrchestratorRegistry** (this module)
   - Canonical for ROLE-BASED COORDINATION
   - Ensures only one orchestrator per role (cluster_orchestrator, etc.)
   - Uses heartbeat-based liveness detection

3. **TrainingCoordinator** (:mod:`app.coordination.training_coordinator`)
   - Specialized facade for TRAINING COORDINATION
   - Adds NFS-based locking for GH200 cluster

These modules answer different questions:
- TaskCoordinator: "Can I spawn another task?"
- OrchestratorRegistry: "Am I the designated orchestrator?"
- TrainingCoordinator: "Can I start training this config?"

Usage:
    from app.coordination.orchestrator_registry import OrchestratorRegistry, OrchestratorRole

    # Register as an orchestrator
    registry = OrchestratorRegistry.get_instance()
    if registry.acquire_role(OrchestratorRole.CLUSTER_ORCHESTRATOR):
        try:
            # Run orchestration logic
            while running:
                registry.heartbeat()  # Call every 30s
                ...
        finally:
            registry.release_role()
    else:
        print("Another orchestrator already holds this role")

    # Check what's running
    active = registry.get_active_orchestrators()
    print(f"Active orchestrators: {active}")
"""

import atexit
import logging
import os
import socket
import sqlite3
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from app.core.async_context import fire_and_forget

logger = logging.getLogger(__name__)

# Import DataCatalog for data availability tracking (December 2025)
try:
    from app.distributed.data_catalog import CatalogStats, DataCatalog, get_data_catalog
    HAS_DATA_CATALOG = True
except ImportError:
    HAS_DATA_CATALOG = False
    DataCatalog = None
    CatalogStats = None

    def get_data_catalog():
        return None

# Import centralized quality thresholds
try:
    from app.quality.thresholds import MIN_QUALITY_FOR_TRAINING
except ImportError:
    MIN_QUALITY_FOR_TRAINING = 0.3

# Import centralized timeout thresholds
try:
    from app.config.thresholds import (
        SQLITE_BUSY_TIMEOUT_SHORT_MS,
        SQLITE_SHORT_TIMEOUT,
    )
except ImportError:
    SQLITE_BUSY_TIMEOUT_SHORT_MS = 5000
    SQLITE_SHORT_TIMEOUT = 10


# ============================================
# Configuration
# ============================================

REGISTRY_DIR = Path("/tmp/ringrift_coordination")
REGISTRY_DB = REGISTRY_DIR / "orchestrator_registry.db"

# Heartbeat settings - import from centralized thresholds (December 2025)
try:
    from app.config.thresholds import (
        HEARTBEAT_INTERVAL,
        PEER_TIMEOUT,
        STALE_CLEANUP_INTERVAL as STALE_CLEANUP_INTERVAL_CONFIG,
    )
    HEARTBEAT_INTERVAL_SECONDS = HEARTBEAT_INTERVAL
    HEARTBEAT_TIMEOUT_SECONDS = PEER_TIMEOUT
    STALE_CLEANUP_INTERVAL = STALE_CLEANUP_INTERVAL_CONFIG
except ImportError:
    # Fallback for testing/standalone use
    HEARTBEAT_INTERVAL_SECONDS = 30  # How often to update heartbeat
    HEARTBEAT_TIMEOUT_SECONDS = 60  # Consider dead if no heartbeat (reduced from 90s)
    STALE_CLEANUP_INTERVAL = 30  # How often to clean up stale entries (reduced from 60s)

# Warning threshold - emit warning at 50% of timeout
HEARTBEAT_WARNING_THRESHOLD = HEARTBEAT_TIMEOUT_SECONDS * 0.5

# Number of missed heartbeats before marking stale (requires 2 consecutive)
MISSED_HEARTBEATS_BEFORE_STALE = 2


class OrchestratorRole(Enum):
    """Roles that orchestrators can hold. Only one orchestrator per role."""
    CLUSTER_ORCHESTRATOR = "cluster_orchestrator"
    IMPROVEMENT_DAEMON = "improvement_daemon"
    PIPELINE_ORCHESTRATOR = "pipeline_orchestrator"
    P2P_LEADER = "p2p_leader"
    TOURNAMENT_RUNNER = "tournament_runner"
    MODEL_SYNC = "model_sync"
    DATA_SYNC = "data_sync"
    UNIFIED_LOOP = "unified_loop"

    # Training enhancement roles (December 2025)
    DISTILLATION_LEADER = "distillation_leader"
    PROMOTION_LEADER = "promotion_leader"
    EXTERNAL_SYNC_LEADER = "external_sync_leader"
    VAST_PIPELINE_LEADER = "vast_pipeline_leader"

    # Cluster-wide data sync (December 2025)
    CLUSTER_DATA_SYNC_LEADER = "cluster_data_sync_leader"


class OrchestratorState(Enum):
    """State of an orchestrator."""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    DEAD = "dead"  # No heartbeat received


@dataclass
class OrchestratorInfo:
    """Information about a registered orchestrator."""
    id: str
    role: str
    hostname: str
    pid: int
    state: str
    started_at: str
    last_heartbeat: str
    metadata: dict[str, Any]

    def is_alive(self) -> bool:
        """Check if orchestrator is considered alive based on heartbeat."""
        if self.state in (OrchestratorState.STOPPED.value, OrchestratorState.DEAD.value):
            return False
        try:
            last_hb = datetime.fromisoformat(self.last_heartbeat)
            timeout = timedelta(seconds=HEARTBEAT_TIMEOUT_SECONDS)
            return datetime.now() - last_hb < timeout
        except (ValueError, TypeError):
            return False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OrchestratorRegistry:
    """
    SQLite-based registry for orchestrator coordination.

    Ensures mutual exclusion between orchestrators and provides
    visibility into what's running across the cluster.
    """

    _instance: Optional['OrchestratorRegistry'] = None
    _lock = threading.RLock()

    def __init__(self):
        self._db_path = REGISTRY_DB
        self._my_id: str | None = None
        self._my_role: OrchestratorRole | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._running = False
        self._init_db()

        # Register cleanup on exit
        atexit.register(self._cleanup_on_exit)

    @classmethod
    def get_instance(cls) -> 'OrchestratorRegistry':
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._cleanup_on_exit()
                cls._instance = None

    def _init_db(self):
        """Initialize the SQLite database with schema."""
        REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self._db_path), timeout=float(SQLITE_SHORT_TIMEOUT))
        conn.execute('PRAGMA journal_mode=WAL')  # Enable WAL for better concurrency
        conn.execute(f'PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_SHORT_MS}')

        conn.executescript('''
            CREATE TABLE IF NOT EXISTS orchestrators (
                id TEXT PRIMARY KEY,
                role TEXT NOT NULL,
                hostname TEXT NOT NULL,
                pid INTEGER NOT NULL,
                state TEXT NOT NULL DEFAULT 'starting',
                started_at TEXT NOT NULL,
                last_heartbeat TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                UNIQUE(role)  -- Only one orchestrator per role
            );

            CREATE INDEX IF NOT EXISTS idx_orchestrators_role ON orchestrators(role);
            CREATE INDEX IF NOT EXISTS idx_orchestrators_state ON orchestrators(state);
            CREATE INDEX IF NOT EXISTS idx_orchestrators_heartbeat ON orchestrators(last_heartbeat);

            -- Event log for debugging coordination issues
            CREATE TABLE IF NOT EXISTS orchestrator_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                orchestrator_id TEXT,
                role TEXT,
                event_type TEXT NOT NULL,
                message TEXT,
                metadata TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_events_timestamp ON orchestrator_events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_events_role ON orchestrator_events(role);
        ''')
        conn.commit()
        conn.close()

        # Clean up stale entries on startup
        self._cleanup_stale_orchestrators()

    @contextmanager
    def _get_conn(self):
        """Get a database connection with proper settings."""
        conn = sqlite3.connect(str(self._db_path), timeout=float(SQLITE_SHORT_TIMEOUT))
        conn.row_factory = sqlite3.Row
        conn.execute(f'PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_SHORT_MS}')
        try:
            yield conn
        finally:
            conn.close()

    def _generate_id(self, role: OrchestratorRole) -> str:
        """Generate unique orchestrator ID."""
        hostname = socket.gethostname()
        pid = os.getpid()
        timestamp = int(time.time() * 1000)
        return f"{role.value}_{hostname}_{pid}_{timestamp}"

    def _log_event(self, conn: sqlite3.Connection, event_type: str, message: str,
                   orchestrator_id: str | None = None, role: str | None = None, metadata: dict | None = None):
        """Log an event for debugging."""
        import json
        conn.execute('''
            INSERT INTO orchestrator_events (timestamp, orchestrator_id, role, event_type, message, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            orchestrator_id or self._my_id,
            role or (self._my_role.value if self._my_role else None),
            event_type,
            message,
            json.dumps(metadata or {})
        ))

    def acquire_role(
        self,
        role: OrchestratorRole,
        force: bool = False,
        metadata: dict[str, Any] | None = None
    ) -> bool:
        """
        Acquire a role as an orchestrator.

        Args:
            role: The role to acquire
            force: If True, forcefully take over from dead orchestrator
            metadata: Optional metadata about this orchestrator

        Returns:
            True if role acquired, False if another orchestrator holds it
        """
        import json

        hostname = socket.gethostname()
        pid = os.getpid()
        now = datetime.now().isoformat()

        with self._get_conn() as conn:
            # Check if role is already held
            cursor = conn.execute(
                'SELECT * FROM orchestrators WHERE role = ?',
                (role.value,)
            )
            existing = cursor.fetchone()

            if existing:
                # Check if holder is still alive
                last_hb = datetime.fromisoformat(existing['last_heartbeat'])
                timeout = timedelta(seconds=HEARTBEAT_TIMEOUT_SECONDS)
                is_alive = datetime.now() - last_hb < timeout

                # Also check if process is actually running
                if is_alive:
                    try:
                        # Check if it's us from a previous run
                        if existing['hostname'] == hostname and existing['pid'] == pid:
                            # Same process, just update
                            pass
                        elif existing['hostname'] == hostname:
                            # Same host, different PID - check if old process alive
                            os.kill(existing['pid'], 0)
                            # Process is alive, can't take over
                            logger.warning(
                                f"Role {role.value} held by PID {existing['pid']} "
                                f"(last heartbeat: {existing['last_heartbeat']})"
                            )
                            self._log_event(conn, 'ACQUIRE_DENIED',
                                f"Role held by active process {existing['pid']}", role=role.value)
                            conn.commit()
                            return False
                    except OSError:
                        # Process not running, we can take over
                        is_alive = False

                if is_alive and not force:
                    logger.warning(
                        f"Role {role.value} held by {existing['hostname']}:{existing['pid']} "
                        f"(last heartbeat: {existing['last_heartbeat']})"
                    )
                    self._log_event(conn, 'ACQUIRE_DENIED',
                        f"Role held by {existing['hostname']}:{existing['pid']}", role=role.value)
                    conn.commit()
                    return False

                # Take over - delete old entry
                self._log_event(conn, 'TAKEOVER',
                    f"Taking over from {existing['hostname']}:{existing['pid']} (alive={is_alive})",
                    role=role.value)
                conn.execute('DELETE FROM orchestrators WHERE role = ?', (role.value,))

            # Register ourselves
            self._my_id = self._generate_id(role)
            self._my_role = role

            conn.execute('''
                INSERT INTO orchestrators (id, role, hostname, pid, state, started_at, last_heartbeat, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self._my_id,
                role.value,
                hostname,
                pid,
                OrchestratorState.RUNNING.value,
                now,
                now,
                json.dumps(metadata or {})
            ))

            self._log_event(conn, 'ACQUIRED', f"Acquired role {role.value}")
            conn.commit()

        # Start heartbeat thread
        self._start_heartbeat()

        logger.info(f"Acquired orchestrator role: {role.value} (id={self._my_id})")
        return True

    def release_role(self):
        """Release the currently held role."""
        if not self._my_role:
            return

        # Stop heartbeat
        self._stop_heartbeat()

        with self._get_conn() as conn:
            conn.execute(
                'UPDATE orchestrators SET state = ? WHERE id = ?',
                (OrchestratorState.STOPPED.value, self._my_id)
            )
            self._log_event(conn, 'RELEASED', f"Released role {self._my_role.value}")
            conn.commit()

            # Actually delete after logging
            conn.execute('DELETE FROM orchestrators WHERE id = ?', (self._my_id,))
            conn.commit()

        logger.info(f"Released orchestrator role: {self._my_role.value}")
        self._my_id = None
        self._my_role = None

    def heartbeat(self, metadata_update: dict[str, Any] | None = None):
        """Update heartbeat timestamp. Call periodically to stay registered."""
        if not self._my_id:
            return

        import json
        now = datetime.now().isoformat()

        with self._get_conn() as conn:
            if metadata_update:
                # Merge metadata
                cursor = conn.execute(
                    'SELECT metadata FROM orchestrators WHERE id = ?',
                    (self._my_id,)
                )
                row = cursor.fetchone()
                if row:
                    existing = json.loads(row['metadata'] or '{}')
                    existing.update(metadata_update)
                    conn.execute(
                        'UPDATE orchestrators SET last_heartbeat = ?, metadata = ? WHERE id = ?',
                        (now, json.dumps(existing), self._my_id)
                    )
            else:
                conn.execute(
                    'UPDATE orchestrators SET last_heartbeat = ? WHERE id = ?',
                    (now, self._my_id)
                )
            conn.commit()

    def _start_heartbeat(self):
        """Start background heartbeat thread."""
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return

        self._running = True

        def heartbeat_loop():
            while self._running and self._my_id:
                try:
                    self.heartbeat()
                    # Also clean up stale entries periodically
                    self._cleanup_stale_orchestrators()
                except Exception as e:
                    logger.error(f"Heartbeat error: {e}")
                time.sleep(HEARTBEAT_INTERVAL_SECONDS)

        self._heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def _stop_heartbeat(self):
        """Stop background heartbeat thread."""
        self._running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)
            self._heartbeat_thread = None
        if self._my_id:
            stale_time = datetime.now() - timedelta(seconds=HEARTBEAT_TIMEOUT_SECONDS + 1)
            with self._get_conn() as conn:
                conn.execute(
                    'UPDATE orchestrators SET last_heartbeat = ? WHERE id = ?',
                    (stale_time.isoformat(), self._my_id)
                )
                conn.commit()

    def _cleanup_stale_orchestrators(self):
        """Remove orchestrators that haven't sent heartbeat.

        Two-phase cleanup:
        1. Warn at 50% of timeout (HEARTBEAT_WARNING_THRESHOLD)
        2. Remove at 100% of timeout (HEARTBEAT_TIMEOUT_SECONDS)
        """
        warning_cutoff = (
            datetime.now() - timedelta(seconds=HEARTBEAT_WARNING_THRESHOLD)
        ).isoformat()
        stale_cutoff = (
            datetime.now() - timedelta(seconds=HEARTBEAT_TIMEOUT_SECONDS)
        ).isoformat()

        with self._get_conn() as conn:
            # First, warn about orchestrators approaching timeout (50%)
            cursor = conn.execute('''
                SELECT id, role, hostname, pid, last_heartbeat FROM orchestrators
                WHERE last_heartbeat < ? AND last_heartbeat >= ? AND state NOT IN (?, ?)
            ''', (warning_cutoff, stale_cutoff,
                  OrchestratorState.STOPPED.value, OrchestratorState.DEAD.value))

            warning_entries = cursor.fetchall()
            for row in warning_entries:
                try:
                    last_hb = datetime.fromisoformat(row['last_heartbeat'])
                    elapsed = (datetime.now() - last_hb).total_seconds()
                    logger.warning(
                        f"Orchestrator heartbeat delayed: {row['role']} "
                        f"({row['hostname']}:{row['pid']}) - "
                        f"{elapsed:.0f}s since last heartbeat "
                        f"(warning at {HEARTBEAT_WARNING_THRESHOLD:.0f}s)"
                    )
                except Exception:
                    pass

            # Find fully stale entries (100% of timeout)
            cursor = conn.execute('''
                SELECT id, role, hostname, pid FROM orchestrators
                WHERE last_heartbeat < ? AND state NOT IN (?, ?)
            ''', (stale_cutoff,
                  OrchestratorState.STOPPED.value, OrchestratorState.DEAD.value))

            stale = cursor.fetchall()
            for row in stale:
                logger.warning(f"Cleaning up stale orchestrator: {row['role']} "
                              f"({row['hostname']}:{row['pid']})")
                self._log_event(conn, 'STALE_CLEANUP',
                    f"Removed stale {row['role']} ({row['hostname']}:{row['pid']})",
                    orchestrator_id=row['id'], role=row['role'])

            # Delete stale entries
            conn.execute('''
                DELETE FROM orchestrators
                WHERE last_heartbeat < ? AND state NOT IN (?, ?)
            ''', (stale_cutoff,
                  OrchestratorState.STOPPED.value, OrchestratorState.DEAD.value))
            conn.commit()

    def _cleanup_on_exit(self):
        """Cleanup handler called on process exit."""
        with suppress(Exception):
            self.release_role()

    def probe_orchestrator_liveness(self, orchestrator_id: str) -> bool:
        """Actively probe if an orchestrator process is still alive.

        Uses OS-level process checking (same host) or heartbeat recency (remote).
        More aggressive than passive heartbeat timeout detection.

        Returns:
            True if orchestrator appears alive, False otherwise.
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                'SELECT hostname, pid, last_heartbeat FROM orchestrators WHERE id = ?',
                (orchestrator_id,)
            )
            row = cursor.fetchone()
            if not row:
                return False

            hostname = row['hostname']
            pid = row['pid']

            # If on same host, check if process is actually running
            if hostname == socket.gethostname():
                try:
                    # Check if process exists (doesn't send signal)
                    os.kill(pid, 0)
                    return True
                except OSError:
                    # Process doesn't exist - mark as stale immediately
                    logger.info(f"Probe found dead process: {orchestrator_id} (pid {pid})")
                    self._mark_stale(orchestrator_id, "Process not found on same host")
                    return False

            # For remote hosts, check heartbeat recency with warning threshold
            try:
                last_hb = datetime.fromisoformat(row['last_heartbeat'])
                elapsed = (datetime.now() - last_hb).total_seconds()

                if elapsed > HEARTBEAT_WARNING_THRESHOLD:
                    logger.warning(
                        f"Orchestrator {orchestrator_id} heartbeat delayed: "
                        f"{elapsed:.0f}s (warning at {HEARTBEAT_WARNING_THRESHOLD:.0f}s)"
                    )

                return elapsed < HEARTBEAT_TIMEOUT_SECONDS
            except (ValueError, TypeError):
                return False

    def _mark_stale(
        self,
        orchestrator_id: str,
        reason: str = "Marked stale by active probe",
    ) -> None:
        """Immediately mark an orchestrator as stale and clean it up.

        Unlike passive cleanup which waits for timeout, this is for cases
        where we know definitively the orchestrator is dead.
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                'SELECT role, hostname, pid FROM orchestrators WHERE id = ?',
                (orchestrator_id,)
            )
            row = cursor.fetchone()
            if not row:
                return

            logger.info(f"Marking orchestrator as stale: {row['role']} "
                       f"({row['hostname']}:{row['pid']}) - {reason}")

            self._log_event(
                conn, 'ACTIVE_STALE_CLEANUP',
                f"Immediately removed {row['role']} - {reason}",
                orchestrator_id=orchestrator_id,
                role=row['role']
            )

            conn.execute(
                'DELETE FROM orchestrators WHERE id = ?',
                (orchestrator_id,)
            )
            conn.commit()

    def probe_all_orchestrators(self) -> dict[str, bool]:
        """Probe liveness of all registered orchestrators.

        Returns:
            Dict mapping orchestrator_id to liveness status.
        """
        results = {}
        with self._get_conn() as conn:
            cursor = conn.execute(
                'SELECT id FROM orchestrators WHERE state NOT IN (?, ?)',
                (OrchestratorState.STOPPED.value, OrchestratorState.DEAD.value)
            )
            for row in cursor:
                orch_id = row['id']
                results[orch_id] = self.probe_orchestrator_liveness(orch_id)
        return results

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_active_orchestrators(self) -> list[OrchestratorInfo]:
        """Get all currently active orchestrators."""
        import json
        result = []

        with self._get_conn() as conn:
            cursor = conn.execute('''
                SELECT * FROM orchestrators
                WHERE state NOT IN (?, ?)
                ORDER BY role
            ''', (OrchestratorState.STOPPED.value, OrchestratorState.DEAD.value))

            for row in cursor:
                info = OrchestratorInfo(
                    id=row['id'],
                    role=row['role'],
                    hostname=row['hostname'],
                    pid=row['pid'],
                    state=row['state'],
                    started_at=row['started_at'],
                    last_heartbeat=row['last_heartbeat'],
                    metadata=json.loads(row['metadata'] or '{}')
                )
                # Only include if actually alive
                if info.is_alive():
                    result.append(info)

        return result

    def is_role_held(self, role: OrchestratorRole) -> bool:
        """Check if a role is currently held by an active orchestrator."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                'SELECT last_heartbeat FROM orchestrators WHERE role = ?',
                (role.value,)
            )
            row = cursor.fetchone()
            if not row:
                return False

            # Check if heartbeat is recent
            last_hb = datetime.fromisoformat(row['last_heartbeat'])
            timeout = timedelta(seconds=HEARTBEAT_TIMEOUT_SECONDS)
            return datetime.now() - last_hb < timeout

    def get_role_holder(self, role: OrchestratorRole) -> OrchestratorInfo | None:
        """Get info about the orchestrator holding a role."""
        import json

        with self._get_conn() as conn:
            cursor = conn.execute(
                'SELECT * FROM orchestrators WHERE role = ?',
                (role.value,)
            )
            row = cursor.fetchone()
            if not row:
                return None

            return OrchestratorInfo(
                id=row['id'],
                role=row['role'],
                hostname=row['hostname'],
                pid=row['pid'],
                state=row['state'],
                started_at=row['started_at'],
                last_heartbeat=row['last_heartbeat'],
                metadata=json.loads(row['metadata'] or '{}')
            )

    def wait_for_role(
        self,
        role: OrchestratorRole,
        timeout: float = 300.0,
        poll_interval: float = 5.0
    ) -> bool:
        """
        Wait for a role to become available and acquire it.

        Args:
            role: Role to wait for
            timeout: Maximum time to wait in seconds
            poll_interval: How often to check

        Returns:
            True if role acquired, False if timeout
        """
        start = time.time()
        while time.time() - start < timeout:
            if self.acquire_role(role):
                return True
            time.sleep(poll_interval)
        return False

    def get_recent_events(self, limit: int = 100, role: str | None = None) -> list[dict[str, Any]]:
        """Get recent orchestrator events for debugging."""
        import json

        with self._get_conn() as conn:
            if role:
                cursor = conn.execute('''
                    SELECT * FROM orchestrator_events
                    WHERE role = ?
                    ORDER BY timestamp DESC LIMIT ?
                ''', (role, limit))
            else:
                cursor = conn.execute('''
                    SELECT * FROM orchestrator_events
                    ORDER BY timestamp DESC LIMIT ?
                ''', (limit,))

            return [
                {
                    'id': row['id'],
                    'timestamp': row['timestamp'],
                    'orchestrator_id': row['orchestrator_id'],
                    'role': row['role'],
                    'event_type': row['event_type'],
                    'message': row['message'],
                    'metadata': json.loads(row['metadata'] or '{}')
                }
                for row in cursor
            ]

    def get_status_summary(self) -> dict[str, Any]:
        """Get summary of orchestrator registry status."""
        active = self.get_active_orchestrators()
        roles_held = {o.role: o.hostname for o in active}
        roles_available = [r.value for r in OrchestratorRole if r.value not in roles_held]

        return {
            'active_count': len(active),
            'roles_held': roles_held,
            'roles_available': roles_available,
            'my_role': self._my_role.value if self._my_role else None,
            'my_id': self._my_id,
            'active_orchestrators': [o.to_dict() for o in active],
        }

    # =========================================================================
    # DataCatalog Integration (December 2025)
    # =========================================================================

    def get_data_availability(
        self,
        board_type: str | None = None,
        num_players: int | None = None,
        min_quality: float = 0.0,
    ) -> dict[str, Any]:
        """Query DataCatalog for current data availability.

        Args:
            board_type: Optional filter by board type
            num_players: Optional filter by player count
            min_quality: Minimum quality threshold

        Returns:
            Dict with data availability stats
        """
        if not HAS_DATA_CATALOG:
            return {
                'available': False,
                'error': 'DataCatalog not available',
                'total_games': 0,
                'high_quality_games': 0,
            }

        try:
            catalog = get_data_catalog()
            if catalog is None:
                return {
                    'available': False,
                    'error': 'DataCatalog instance not available',
                    'total_games': 0,
                    'high_quality_games': 0,
                }

            # Get catalog stats
            stats = catalog.get_stats()

            # Get high-quality game count
            high_quality_games = stats.high_quality_games if stats else 0

            # Get filtered game count if filters provided
            filtered_games = 0
            if board_type or num_players:
                training_data = catalog.get_training_data(
                    min_quality=min_quality,
                    max_games=100000,
                    board_type=board_type,
                    num_players=num_players,
                )
                filtered_games = len(training_data)

            return {
                'available': True,
                'total_sources': stats.total_sources if stats else 0,
                'total_games': stats.total_games if stats else 0,
                'high_quality_games': high_quality_games,
                'avg_quality_score': stats.avg_quality_score if stats else 0.0,
                'board_type_distribution': dict(stats.board_type_distribution) if stats else {},
                'filtered_games': filtered_games if (board_type or num_players) else None,
                'filter_board_type': board_type,
                'filter_num_players': num_players,
                'filter_min_quality': min_quality,
                'timestamp': datetime.now().isoformat(),
            }

        except Exception as e:
            logger.warning(f"Failed to get data availability: {e}")
            return {
                'available': False,
                'error': str(e),
                'total_games': 0,
                'high_quality_games': 0,
            }

    def heartbeat_with_data_status(
        self,
        board_type: str | None = None,
        num_players: int | None = None,
    ) -> None:
        """Update heartbeat with data availability status.

        This enriches the orchestrator's metadata with current data
        availability from the DataCatalog, enabling data-aware coordination.

        Args:
            board_type: Optional filter by board type
            num_players: Optional filter by player count
        """
        data_availability = self.get_data_availability(
            board_type=board_type,
            num_players=num_players,
        )

        metadata_update = {
            'data_availability': data_availability,
            'last_data_check': datetime.now().isoformat(),
        }

        self.heartbeat(metadata_update=metadata_update)

    def has_sufficient_data(
        self,
        min_games: int = 500,
        board_type: str | None = None,
        num_players: int | None = None,
        min_quality: float = 0.0,
    ) -> bool:
        """Check if there's sufficient data for training.

        Args:
            min_games: Minimum number of games required
            board_type: Optional filter by board type
            num_players: Optional filter by player count
            min_quality: Minimum quality threshold

        Returns:
            True if sufficient data is available
        """
        availability = self.get_data_availability(
            board_type=board_type,
            num_players=num_players,
            min_quality=min_quality,
        )

        if not availability.get('available', False):
            return False

        # Use filtered count if filters were applied
        if board_type or num_players:
            return (availability.get('filtered_games', 0) or 0) >= min_games

        return availability.get('total_games', 0) >= min_games

    def get_training_readiness(
        self,
        config_key: str = "",
        min_games: int = 500,
        min_quality: float = MIN_QUALITY_FOR_TRAINING,
    ) -> dict[str, Any]:
        """Check training readiness including data availability.

        Combines data catalog stats with training signal checks to
        provide a comprehensive readiness assessment.

        Args:
            config_key: Configuration key (e.g., "square8_2p")
            min_games: Minimum games required for training
            min_quality: Minimum average quality threshold (default from quality.thresholds)

        Returns:
            Dict with training readiness assessment
        """
        # Parse config_key
        board_type = None
        num_players = None
        if config_key:
            parts = config_key.replace("_", " ").replace("p", "").split()
            if len(parts) >= 1:
                board_type = parts[0]
            if len(parts) >= 2 and parts[1].isdigit():
                num_players = int(parts[1])

        # Get data availability
        availability = self.get_data_availability(
            board_type=board_type,
            num_players=num_players,
            min_quality=min_quality,
        )

        # Compute readiness
        total_games = availability.get('total_games', 0)
        avg_quality = availability.get('avg_quality_score', 0.0)
        has_data = availability.get('available', False)

        is_ready = (
            has_data and
            total_games >= min_games and
            avg_quality >= min_quality
        )

        reasons = []
        if not has_data:
            reasons.append("DataCatalog not available")
        if total_games < min_games:
            reasons.append(f"Insufficient games: {total_games} < {min_games}")
        if avg_quality < min_quality:
            reasons.append(f"Low quality: {avg_quality:.2f} < {min_quality}")

        return {
            'ready': is_ready,
            'config_key': config_key,
            'reasons': reasons if reasons else ['Ready for training'],
            'data_availability': availability,
            'thresholds': {
                'min_games': min_games,
                'min_quality': min_quality,
            },
            'timestamp': datetime.now().isoformat(),
        }


# ============================================
# Convenience Functions
# ============================================

def get_registry() -> OrchestratorRegistry:
    """Get the orchestrator registry singleton."""
    return OrchestratorRegistry.get_instance()


def acquire_orchestrator_role(role: OrchestratorRole, **kwargs) -> bool:
    """Convenience function to acquire a role."""
    return get_registry().acquire_role(role, **kwargs)


def release_orchestrator_role():
    """Convenience function to release current role."""
    get_registry().release_role()


def is_orchestrator_role_available(role: OrchestratorRole) -> bool:
    """Check if a role is available."""
    return not get_registry().is_role_held(role)


@contextmanager
def orchestrator_role(role: OrchestratorRole, **kwargs):
    """Context manager for holding an orchestrator role."""
    registry = get_registry()
    if not registry.acquire_role(role, **kwargs):
        raise RuntimeError(f"Failed to acquire orchestrator role: {role.value}")
    try:
        yield registry
    finally:
        registry.release_role()


# ============================================
# Cross-Coordinator Health Protocol (December 2025)
# ============================================

@dataclass
class CoordinatorHealth:
    """Health status of a coordinator."""
    coordinator_id: str
    role: str
    is_healthy: bool
    last_seen: str
    response_time_ms: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinator_id": self.coordinator_id,
            "role": self.role,
            "is_healthy": self.is_healthy,
            "last_seen": self.last_seen,
            "response_time_ms": self.response_time_ms,
            "error": self.error,
            "metadata": self.metadata,
        }


class CrossCoordinatorHealthProtocol:
    """Protocol for cross-coordinator health checks.

    Allows coordinators to monitor each other's health and detect failures.
    Uses the OrchestratorRegistry as the source of truth for liveness.

    Usage:
        from app.coordination.orchestrator_registry import (
            get_cross_coordinator_health,
            CrossCoordinatorHealthProtocol,
        )

        health_protocol = get_cross_coordinator_health()

        # Check health of all coordinators
        health_report = health_protocol.check_all_coordinators()

        # Check specific coordinator
        is_healthy = health_protocol.is_coordinator_healthy(
            OrchestratorRole.CLUSTER_ORCHESTRATOR
        )

        # Get unhealthy coordinators
        unhealthy = health_protocol.get_unhealthy_coordinators()
    """

    def __init__(
        self,
        registry: OrchestratorRegistry | None = None,
        health_check_timeout_ms: float = 5000.0,
    ):
        """Initialize cross-coordinator health protocol.

        Args:
            registry: OrchestratorRegistry instance (uses singleton if None)
            health_check_timeout_ms: Timeout for health checks in milliseconds
        """
        self._registry = registry
        self._health_check_timeout_ms = health_check_timeout_ms
        self._health_cache: dict[str, CoordinatorHealth] = {}
        self._last_check_time: float = 0.0
        self._check_cooldown_seconds: float = 10.0

    @property
    def registry(self) -> OrchestratorRegistry:
        """Get registry, initializing if needed."""
        if self._registry is None:
            self._registry = get_registry()
        return self._registry

    def check_all_coordinators(self) -> dict[str, CoordinatorHealth]:
        """Check health of all registered coordinators.

        Returns:
            Dict mapping coordinator_id to CoordinatorHealth
        """
        now = time.time()

        # Use cache if recent
        if now - self._last_check_time < self._check_cooldown_seconds:
            return self._health_cache

        orchestrators = self.registry.get_active_orchestrators()
        health_report: dict[str, CoordinatorHealth] = {}

        for orch in orchestrators:
            start_time = time.time()
            is_healthy = orch.is_alive()
            response_time_ms = (time.time() - start_time) * 1000

            health = CoordinatorHealth(
                coordinator_id=orch.id,
                role=orch.role,
                is_healthy=is_healthy,
                last_seen=orch.last_heartbeat,
                response_time_ms=response_time_ms,
                error=None if is_healthy else "Heartbeat timeout",
                metadata=orch.metadata,
            )
            health_report[orch.id] = health

        self._health_cache = health_report
        self._last_check_time = now

        return health_report

    def is_coordinator_healthy(self, role: OrchestratorRole) -> bool:
        """Check if a specific coordinator role is healthy.

        Args:
            role: The orchestrator role to check

        Returns:
            True if healthy coordinator exists for this role
        """
        orchestrators = self.registry.get_active_orchestrators()
        return any(orch.role == role.value and orch.is_alive() for orch in orchestrators)

    def get_healthy_coordinators(self) -> list[CoordinatorHealth]:
        """Get list of all healthy coordinators."""
        health_report = self.check_all_coordinators()
        return [h for h in health_report.values() if h.is_healthy]

    def get_unhealthy_coordinators(self) -> list[CoordinatorHealth]:
        """Get list of all unhealthy coordinators."""
        health_report = self.check_all_coordinators()
        return [h for h in health_report.values() if not h.is_healthy]

    def get_role_health(self, role: OrchestratorRole) -> CoordinatorHealth | None:
        """Get health status for a specific role.

        Args:
            role: The orchestrator role to check

        Returns:
            CoordinatorHealth if role is registered, None otherwise
        """
        orchestrators = self.registry.get_active_orchestrators()
        for orch in orchestrators:
            if orch.role == role.value:
                start_time = time.time()
                is_healthy = orch.is_alive()
                response_time_ms = (time.time() - start_time) * 1000

                return CoordinatorHealth(
                    coordinator_id=orch.id,
                    role=orch.role,
                    is_healthy=is_healthy,
                    last_seen=orch.last_heartbeat,
                    response_time_ms=response_time_ms,
                    error=None if is_healthy else "Heartbeat timeout",
                    metadata=orch.metadata,
                )
        return None

    def get_cluster_health_summary(self) -> dict[str, Any]:
        """Get overall cluster health summary.

        Returns:
            Dict with cluster-wide health metrics
        """
        health_report = self.check_all_coordinators()

        total = len(health_report)
        healthy = sum(1 for h in health_report.values() if h.is_healthy)
        unhealthy = total - healthy

        # Group by role
        by_role: dict[str, dict[str, Any]] = {}
        for health in health_report.values():
            role = health.role
            if role not in by_role:
                by_role[role] = {"healthy": 0, "unhealthy": 0}
            if health.is_healthy:
                by_role[role]["healthy"] += 1
            else:
                by_role[role]["unhealthy"] += 1

        # Identify critical roles that are down
        critical_roles = {
            OrchestratorRole.CLUSTER_ORCHESTRATOR.value,
            OrchestratorRole.UNIFIED_LOOP.value,
        }
        critical_down = [
            role for role in critical_roles
            if role in by_role and by_role[role]["healthy"] == 0
        ]

        return {
            "timestamp": datetime.now().isoformat(),
            "total_coordinators": total,
            "healthy_count": healthy,
            "unhealthy_count": unhealthy,
            "health_percentage": (healthy / total * 100) if total > 0 else 0,
            "by_role": by_role,
            "critical_roles_down": critical_down,
            "cluster_healthy": len(critical_down) == 0 and healthy > 0,
        }

    def on_coordinator_failure(
        self,
        callback: Any,  # Callable[[CoordinatorHealth], None]
    ) -> None:
        """Register a callback for coordinator failures.

        The callback will be called whenever an unhealthy coordinator is detected.
        """
        # Store callback for future use
        if not hasattr(self, '_failure_callbacks'):
            self._failure_callbacks = []
        self._failure_callbacks.append(callback)

    def emit_health_event(self, health: CoordinatorHealth) -> None:
        """Emit a health event to the event bus."""
        try:
            from app.coordination.event_router import (
                DataEvent,
                DataEventType,
                get_event_bus,
            )

            event_type = (
                DataEventType.COORDINATOR_HEALTHY
                if health.is_healthy
                else DataEventType.COORDINATOR_UNHEALTHY
            )

            event = DataEvent(
                event_type=event_type,
                payload=health.to_dict(),
                source="cross_coordinator_health",
            )

            bus = get_event_bus()
            import asyncio
            try:
                asyncio.get_running_loop()
                fire_and_forget(
                    bus.publish(event),
                    name="orchestrator_health_event",
                )
            except RuntimeError:
                if hasattr(bus, 'publish_sync'):
                    bus.publish_sync(event)

        except Exception as e:
            logger.debug(f"Failed to emit health event: {e}")


# Singleton health protocol
_health_protocol: CrossCoordinatorHealthProtocol | None = None


def get_cross_coordinator_health() -> CrossCoordinatorHealthProtocol:
    """Get the global cross-coordinator health protocol instance."""
    global _health_protocol
    if _health_protocol is None:
        _health_protocol = CrossCoordinatorHealthProtocol()
    return _health_protocol


def check_cluster_health() -> dict[str, Any]:
    """Convenience function to check cluster health."""
    return get_cross_coordinator_health().get_cluster_health_summary()


# ============================================
# Coordinator Registration (December 2025)
# ============================================

# Registry of active coordinators (non-orchestrator singletons)
_coordinator_registry: dict[str, dict[str, Any]] = {}
_coordinator_lock = threading.Lock()


def register_coordinator(
    name: str,
    coordinator: Any,
    health_callback: Callable[[], bool] | None = None,
    shutdown_callback: Callable[[], None] | None = None,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Register a coordinator (non-orchestrator singleton) for visibility.

    This allows tracking coordinators that aren't role-based orchestrators but
    still need lifecycle management and health monitoring.

    Args:
        name: Unique coordinator name
        coordinator: Coordinator instance
        health_callback: Optional callback to check health (returns bool)
        shutdown_callback: Optional callback for graceful shutdown
        metadata: Additional metadata

    Returns:
        True if registered successfully

    Example:
        from app.coordination.orchestrator_registry import register_coordinator

        register_coordinator(
            "data_coordinator",
            data_coordinator_instance,
            health_callback=lambda: data_coordinator_instance.is_healthy(),
            shutdown_callback=data_coordinator_instance.shutdown,
            metadata={"type": "training", "priority": "high"},
        )
    """
    with _coordinator_lock:
        _coordinator_registry[name] = {
            "coordinator": coordinator,
            "health_callback": health_callback,
            "shutdown_callback": shutdown_callback,
            "metadata": metadata or {},
            "registered_at": time.time(),
            "type": type(coordinator).__name__,
        }
    logger.debug(f"Registered coordinator: {name}")
    return True


def unregister_coordinator(name: str) -> bool:
    """Unregister a coordinator.

    Args:
        name: Coordinator name

    Returns:
        True if unregistered
    """
    with _coordinator_lock:
        if name in _coordinator_registry:
            del _coordinator_registry[name]
            return True
    return False


def get_coordinator(name: str) -> Any | None:
    """Get a registered coordinator by name."""
    with _coordinator_lock:
        entry = _coordinator_registry.get(name)
        return entry["coordinator"] if entry else None


def get_registered_coordinators() -> dict[str, dict[str, Any]]:
    """Get all registered coordinators with their status.

    Returns:
        Dict mapping coordinator name to status info
    """
    result = {}
    with _coordinator_lock:
        for name, entry in _coordinator_registry.items():
            is_healthy = True
            if entry.get("health_callback"):
                try:
                    is_healthy = entry["health_callback"]()
                except Exception:
                    is_healthy = False

            result[name] = {
                "type": entry["type"],
                "healthy": is_healthy,
                "registered_at": entry["registered_at"],
                "metadata": entry["metadata"],
            }
    return result


def shutdown_all_coordinators() -> dict[str, bool]:
    """Shutdown all registered coordinators.

    Returns:
        Dict mapping coordinator name to shutdown success
    """
    results = {}
    with _coordinator_lock:
        for name, entry in list(_coordinator_registry.items()):
            try:
                if entry.get("shutdown_callback"):
                    entry["shutdown_callback"]()
                results[name] = True
            except Exception as e:
                logger.warning(f"Error shutting down coordinator {name}: {e}")
                results[name] = False
    return results


def auto_register_known_coordinators() -> dict[str, bool]:
    """Auto-register known training/quality coordinators.

    Discovers and registers common coordinator singletons that should be
    tracked for health monitoring.

    Returns:
        Dict mapping coordinator name to registration success
    """
    results = {}

    # Only include coordinators with singleton patterns
    # Note: training_coordinator self-registers on first access
    known_coordinators = [
        # Core training coordinators
        {
            "name": "training_coordinator",
            "module": "app.coordination.training_coordinator",
            "getter": "get_training_coordinator",
        },
        {
            "name": "training_data_coordinator",
            "module": "app.training.data_coordinator",
            "getter": "get_training_data_coordinator",
        },
        {
            "name": "elo_service",
            "module": "app.training.elo_service",
            "getter": "get_elo_service",
        },
        {
            "name": "background_evaluator",
            "module": "app.training.background_eval",
            "getter": "get_background_evaluator",
        },
        {
            "name": "quality_orchestrator",
            "module": "app.quality.data_quality_orchestrator",
            "getter": "get_quality_orchestrator",
        },
        # December 2025 orchestrators
        {
            "name": "selfplay_orchestrator",
            "module": "app.coordination.selfplay_orchestrator",
            "getter": "get_selfplay_orchestrator",
        },
        {
            "name": "pipeline_orchestrator",
            "module": "app.coordination.data_pipeline_orchestrator",
            "getter": "get_pipeline_orchestrator",
        },
        {
            "name": "task_lifecycle_coordinator",
            "module": "app.coordination.task_lifecycle_coordinator",
            "getter": "get_task_lifecycle_coordinator",
        },
        {
            "name": "optimization_coordinator",
            "module": "app.coordination.optimization_coordinator",
            "getter": "get_optimization_coordinator",
        },
        {
            "name": "metrics_orchestrator",
            "module": "app.coordination.metrics_analysis_orchestrator",
            "getter": "get_metrics_orchestrator",
        },
        {
            "name": "resource_coordinator",
            "module": "app.coordination.resource_monitoring_coordinator",
            "getter": "get_resource_coordinator",
        },
        {
            "name": "cache_orchestrator",
            "module": "app.coordination.cache_coordination_orchestrator",
            "getter": "get_cache_orchestrator",
        },
        {
            "name": "model_coordinator",
            "module": "app.coordination.model_lifecycle_coordinator",
            "getter": "get_model_coordinator",
        },
        {
            "name": "error_coordinator",
            "module": "app.coordination.error_recovery_coordinator",
            "getter": "get_error_coordinator",
        },
        {
            "name": "leadership_coordinator",
            "module": "app.coordination.leadership_coordinator",
            "getter": "get_leadership_coordinator",
        },
        {
            "name": "sync_scheduler",
            "module": "app.coordination.sync_coordinator",
            "getter": "get_sync_coordinator",
        },
        {
            "name": "event_coordinator",
            "module": "app.coordination.unified_event_coordinator",
            "getter": "get_event_coordinator",
        },
    ]

    # Allowed module prefixes for security (defense-in-depth)
    ALLOWED_MODULE_PREFIXES = ("app.coordination.", "app.training.", "app.distributed.")

    for coord_def in known_coordinators:
        name = coord_def["name"]
        try:
            # Validate module path before dynamic import
            module_path = coord_def["module"]
            if not any(module_path.startswith(prefix) for prefix in ALLOWED_MODULE_PREFIXES):
                logger.warning(f"Skipping disallowed module: {module_path}")
                results[name] = False
                continue

            # Dynamic import
            module = __import__(module_path, fromlist=[coord_def["getter"]])
            getter = getattr(module, coord_def["getter"])
            coordinator = getter()

            if coordinator is not None:
                # Get health method if available
                health_cb = None
                if hasattr(coordinator, "is_healthy"):
                    health_cb = coordinator.is_healthy
                elif hasattr(coordinator, "get_health"):
                    def health_cb(c=coordinator):
                        return c.get_health().get("healthy", True)

                # Get shutdown method if available
                shutdown_cb = None
                if hasattr(coordinator, "shutdown"):
                    shutdown_cb = coordinator.shutdown
                elif hasattr(coordinator, "stop"):
                    shutdown_cb = coordinator.stop

                register_coordinator(
                    name,
                    coordinator,
                    health_callback=health_cb,
                    shutdown_callback=shutdown_cb,
                    metadata={"auto_registered": True},
                )
                results[name] = True
            else:
                results[name] = False  # Coordinator not initialized yet

        except ImportError:
            results[name] = False
        except Exception as e:
            logger.debug(f"Could not register {name}: {e}")
            results[name] = False

    registered = sum(1 for v in results.values() if v)
    logger.info(f"Auto-registered {registered}/{len(known_coordinators)} coordinators")
    return results


# ============================================
# Auto-discovery of Known Orchestrators (December 2025)
# ============================================

def discover_and_register_orchestrators() -> dict[str, Any]:
    """Discover and register known orchestrators.

    Attempts to import and reference all known orchestrators, adding them
    to the registry's health monitoring. This enables centralized visibility
    into all orchestration components.

    Returns:
        Dict mapping orchestrator names to discovery status
    """
    results = {}

    # Known orchestrator modules and their metadata
    known_orchestrators = [
        {
            "name": "unified_training",
            "module": "app.training.unified_orchestrator",
            "class_or_getter": "UnifiedTrainingOrchestrator",
            "role": OrchestratorRole.UNIFIED_LOOP,
        },
        {
            "name": "sync_orchestrator",
            "module": "app.distributed.sync_orchestrator",
            "class_or_getter": "get_sync_orchestrator",
            "role": OrchestratorRole.DATA_SYNC,
        },
        {
            "name": "event_coordinator",
            "module": "app.coordination.unified_event_coordinator",
            "class_or_getter": "get_event_coordinator",
            "role": None,  # Not a role-based orchestrator
        },
        {
            "name": "tournament_orchestrator",
            "module": "app.tournament.orchestrator",
            "class_or_getter": "TournamentOrchestrator",
            "role": OrchestratorRole.TOURNAMENT_RUNNER,
        },
    ]

    # Allowed module prefixes for security (defense-in-depth)
    ALLOWED_ORCH_PREFIXES = (
        "app.coordination.", "app.training.", "app.distributed.", "app.tournament."
    )

    for orch_def in known_orchestrators:
        name = orch_def["name"]
        try:
            # Validate module path before dynamic import
            module_path = orch_def["module"]
            if not any(module_path.startswith(prefix) for prefix in ALLOWED_ORCH_PREFIXES):
                logger.warning(f"Skipping disallowed orchestrator module: {module_path}")
                results[name] = {"success": False, "error": "Disallowed module path"}
                continue

            # Dynamic import
            module_parts = module_path.rsplit(".", 1)
            package = __import__(module_parts[0], fromlist=[module_parts[1]])
            module = getattr(package, module_parts[1])

            # Verify class or getter function exists
            getattr(module, orch_def["class_or_getter"])

            # Record discovery (don't instantiate unless needed)
            results[name] = {
                "success": True,
                "module": orch_def["module"],
                "class": orch_def["class_or_getter"],
                "role": orch_def["role"].value if orch_def["role"] else None,
            }
            logger.debug(f"Discovered orchestrator: {name}")

        except ImportError as e:
            results[name] = {"success": False, "error": f"Import error: {e}"}
        except AttributeError as e:
            results[name] = {"success": False, "error": f"Attribute error: {e}"}
        except Exception as e:
            results[name] = {"success": False, "error": str(e)}

    discovered = sum(1 for r in results.values() if r.get("success"))
    logger.info(f"Discovered {discovered}/{len(known_orchestrators)} orchestrators")
    return results


def get_orchestrator_inventory() -> dict[str, Any]:
    """Get inventory of all known orchestrators and their status.

    Returns:
        Dict with discovery results and active orchestrators
    """
    registry = get_registry()

    return {
        "discovered": discover_and_register_orchestrators(),
        "active": [o.to_dict() for o in registry.get_active_orchestrators()],
        "roles_held": {o.role: o.hostname for o in registry.get_active_orchestrators()},
        "health": check_cluster_health(),
        "timestamp": datetime.now().isoformat(),
    }


# ============================================
# Module Exports
# ============================================

__all__ = [
    "OrchestratorInfo",
    # Core classes
    "OrchestratorRegistry",
    "OrchestratorRole",
    "auto_register_known_coordinators",
    # Health and status
    "check_cluster_health",
    # Discovery
    "discover_and_register_orchestrators",
    "get_coordinator",
    "get_orchestrator_inventory",
    "get_registered_coordinators",
    # Singleton access
    "get_registry",
    # Coordinator registration (December 2025)
    "register_coordinator",
    "shutdown_all_coordinators",
    "unregister_coordinator",
]


# ============================================
# CLI for Debugging
# ============================================

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Orchestrator Registry CLI")
    parser.add_argument("--status", action="store_true", help="Show registry status")
    parser.add_argument("--events", type=int, metavar="N", help="Show last N events")
    parser.add_argument("--role-events", type=str, metavar="ROLE", help="Show events for role")
    parser.add_argument("--cleanup", action="store_true", help="Force cleanup of stale entries")
    args = parser.parse_args()

    registry = get_registry()

    if args.status:
        status = registry.get_status_summary()
        print(json.dumps(status, indent=2, default=str))

    elif args.events:
        events = registry.get_recent_events(limit=args.events)
        print(json.dumps(events, indent=2))

    elif args.role_events:
        events = registry.get_recent_events(limit=50, role=args.role_events)
        print(json.dumps(events, indent=2))

    elif args.cleanup:
        registry._cleanup_stale_orchestrators()
        print("Cleanup complete")
        status = registry.get_status_summary()
        print(json.dumps(status, indent=2, default=str))

    else:
        parser.print_help()
