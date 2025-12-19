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
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# Import DataCatalog for data availability tracking (December 2025)
try:
    from app.distributed.data_catalog import DataCatalog, get_data_catalog, CatalogStats
    HAS_DATA_CATALOG = True
except ImportError:
    HAS_DATA_CATALOG = False
    DataCatalog = None
    CatalogStats = None

    def get_data_catalog():
        return None


# ============================================
# Configuration
# ============================================

REGISTRY_DIR = Path("/tmp/ringrift_coordination")
REGISTRY_DB = REGISTRY_DIR / "orchestrator_registry.db"

# Heartbeat settings
HEARTBEAT_INTERVAL_SECONDS = 30  # How often to update heartbeat
HEARTBEAT_TIMEOUT_SECONDS = 90  # Consider dead if no heartbeat in this time
STALE_CLEANUP_INTERVAL = 60  # How often to clean up stale entries


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
    metadata: Dict[str, Any]

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

    def to_dict(self) -> Dict[str, Any]:
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
        self._my_id: Optional[str] = None
        self._my_role: Optional[OrchestratorRole] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
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

    def _init_db(self):
        """Initialize the SQLite database with schema."""
        REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.execute('PRAGMA journal_mode=WAL')  # Enable WAL for better concurrency
        conn.execute('PRAGMA busy_timeout=5000')  # 5s busy timeout

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
        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA busy_timeout=5000')
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
                   orchestrator_id: str = None, role: str = None, metadata: Dict = None):
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
        metadata: Dict[str, Any] = None
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

    def heartbeat(self, metadata_update: Dict[str, Any] = None):
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

    def _cleanup_stale_orchestrators(self):
        """Remove orchestrators that haven't sent heartbeat."""
        cutoff = (datetime.now() - timedelta(seconds=HEARTBEAT_TIMEOUT_SECONDS)).isoformat()

        with self._get_conn() as conn:
            # Find stale entries
            cursor = conn.execute('''
                SELECT id, role, hostname, pid FROM orchestrators
                WHERE last_heartbeat < ? AND state != ?
            ''', (cutoff, OrchestratorState.STOPPED.value))

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
                WHERE last_heartbeat < ? AND state != ?
            ''', (cutoff, OrchestratorState.STOPPED.value))
            conn.commit()

    def _cleanup_on_exit(self):
        """Cleanup handler called on process exit."""
        try:
            self.release_role()
        except Exception:
            pass

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_active_orchestrators(self) -> List[OrchestratorInfo]:
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

    def get_role_holder(self, role: OrchestratorRole) -> Optional[OrchestratorInfo]:
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

    def get_recent_events(self, limit: int = 100, role: str = None) -> List[Dict[str, Any]]:
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

    def get_status_summary(self) -> Dict[str, Any]:
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
        board_type: Optional[str] = None,
        num_players: Optional[int] = None,
        min_quality: float = 0.0,
    ) -> Dict[str, Any]:
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
        board_type: Optional[str] = None,
        num_players: Optional[int] = None,
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
        board_type: Optional[str] = None,
        num_players: Optional[int] = None,
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
        min_quality: float = 0.3,
    ) -> Dict[str, Any]:
        """Check training readiness including data availability.

        Combines data catalog stats with training signal checks to
        provide a comprehensive readiness assessment.

        Args:
            config_key: Configuration key (e.g., "square8_2p")
            min_games: Minimum games required for training
            min_quality: Minimum average quality threshold

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
