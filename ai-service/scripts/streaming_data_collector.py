#!/usr/bin/env python3
"""Streaming Data Collector - Continuous incremental sync from cluster hosts.

DEPRECATED: This script is deprecated in favor of unified_data_sync.py
Please use: python scripts/unified_data_sync.py
All functionality has been preserved in the unified service.

This service replaces batch data sync (30-min intervals) with continuous
60-second polling for new games. Key features:

1. Incremental sync via rsync --append
2. Game ID deduplication across all sources
3. Event emission for downstream triggers
4. Per-host manifest tracking
5. Automatic retry with exponential backoff

Usage:
    # Run as standalone service (DEPRECATED - use unified_data_sync.py instead)
    python scripts/streaming_data_collector.py

    # With custom config
    python scripts/streaming_data_collector.py --config config/unified_loop.yaml

    # One-shot sync
    python scripts/streaming_data_collector.py --once

    # Dry run (check what would sync)
    python scripts/streaming_data_collector.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import warnings
import yaml

# Emit deprecation warning at runtime
warnings.warn(
    "streaming_data_collector.py is deprecated. "
    "Please use: python scripts/unified_data_sync.py\n"
    "All functionality has been preserved in the unified service.",
    DeprecationWarning,
    stacklevel=2,
)

# Allow imports from app/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]

# Import event bus helpers (consolidated imports)
from app.distributed.event_helpers import (
    has_event_bus,
    get_event_bus_safe,
    emit_new_games_safe,
    DataEventType,
    DataEvent,
)
HAS_EVENT_BUS = has_event_bus()

# For backwards compatibility
if HAS_EVENT_BUS:
    from app.distributed.data_events import get_event_bus, emit_new_games
else:
    get_event_bus = get_event_bus_safe
    # emit_new_games wrapper
    async def emit_new_games(host, new_games, config="", source=""):
        return await emit_new_games_safe(host, new_games, config, source)

# Import coordination helpers (consolidated imports)
from app.coordination.helpers import (
    has_coordination,
    get_registry_safe,
    OrchestratorRole,
)
HAS_COORDINATION = has_coordination()

# For backwards compatibility
if HAS_COORDINATION:
    from app.coordination import get_registry
    from app.coordination.orchestrator_registry import orchestrator_role
else:
    get_registry = get_registry_safe
    orchestrator_role = None  # No-op if coordination unavailable

HAS_ORCHESTRATOR_REGISTRY = HAS_COORDINATION

# Try to import cross-process event queue for distributed coordination
try:
    from app.coordination.cross_process_events import publish_event as publish_cross_process_event
    HAS_CROSS_PROCESS_EVENTS = True
except ImportError:
    HAS_CROSS_PROCESS_EVENTS = False


def bridge_event_to_cross_process(event_type: str, payload: dict, source: str = "streaming_data_collector") -> None:
    """Bridge an event to the cross-process event queue.

    This allows other daemons (unified_ai_loop, cluster_orchestrator, etc.)
    to react to events from this collector.
    """
    if not HAS_CROSS_PROCESS_EVENTS:
        return
    try:
        publish_cross_process_event(
            event_type=event_type,
            payload=payload,
            source=source,
        )
    except Exception as e:
        print(f"[Collector] Warning: Failed to bridge event to cross-process: {e}")


# Try to import sync_lock for coordinating rsync operations
try:
    from app.coordination.sync_mutex import acquire_sync_lock, release_sync_lock
    HAS_SYNC_LOCK = True
except ImportError:
    HAS_SYNC_LOCK = False

# Try to import BandwidthManager for large transfers
try:
    from app.coordination.bandwidth_manager import (
        request_bandwidth,
        release_bandwidth,
        TransferPriority,
    )
    HAS_BANDWIDTH_MANAGER = True
except ImportError:
    HAS_BANDWIDTH_MANAGER = False
    TransferPriority = None

# Try to import CircuitBreaker for fault tolerance
try:
    from app.distributed.circuit_breaker import (
        get_host_breaker,
        CircuitOpenError,
        CircuitState,
    )
    HAS_CIRCUIT_BREAKER = True
except ImportError:
    HAS_CIRCUIT_BREAKER = False
    CircuitOpenError = Exception  # Fallback

# Try to import ManifestReplicator for distributed manifest
try:
    from app.distributed.manifest_replication import (
        ManifestReplicator,
        ReplicaHost,
        create_replicator_from_config,
    )
    HAS_MANIFEST_REPLICATION = True
except ImportError:
    HAS_MANIFEST_REPLICATION = False

# Try to import P2P fallback sync
try:
    from app.distributed.p2p_sync_client import (
        P2PFallbackSync,
        P2PSyncClient,
    )
    HAS_P2P_FALLBACK = True
except ImportError:
    HAS_P2P_FALLBACK = False

# Try to import host classification (consolidated module)
try:
    from app.distributed.host_classification import (
        StorageType,
        HostTier,
        HostSyncProfile,
        classify_host_storage,
        classify_host_tier,
        get_ephemeral_hosts,
        create_sync_profile,
    )
    HAS_HOST_CLASSIFICATION = True
except ImportError:
    HAS_HOST_CLASSIFICATION = False
    StorageType = None
    HostSyncProfile = None

# Try to import robust data sync for WAL and Elo replication
try:
    from app.distributed.data_sync_robust import (
        WriteAheadLog,
        EloReplicator,
        create_elo_replicator,
    )
    HAS_ROBUST_SYNC = True
except ImportError:
    HAS_ROBUST_SYNC = False
    WriteAheadLog = None
    EloReplicator = None

# Try to import unified manifest (consolidated implementation)
try:
    from app.distributed.unified_manifest import (
        DataManifest as UnifiedDataManifest,
        HostSyncState as UnifiedHostSyncState,
        create_manifest,
    )
    HAS_UNIFIED_MANIFEST = True
except ImportError:
    HAS_UNIFIED_MANIFEST = False
    UnifiedDataManifest = None
    UnifiedHostSyncState = None


@dataclass
class HostConfig:
    """Configuration for a remote host."""
    name: str
    ssh_host: str
    ssh_user: str = "ubuntu"
    ssh_port: int = 22
    ssh_key: Optional[str] = None
    remote_db_path: str = "~/ringrift/ai-service/data/games"
    enabled: bool = True
    role: str = "selfplay"
    # Ephemeral storage support (RAM disk hosts like Vast.ai)
    storage_type: str = "persistent"  # "persistent", "ephemeral", "ram"
    is_ephemeral: bool = False  # True for RAM disk hosts (data lost on termination)


# HostSyncState - use unified implementation if available
if HAS_UNIFIED_MANIFEST:
    HostSyncState = UnifiedHostSyncState
else:
    @dataclass
    class HostSyncState:
        """Sync state for a host (legacy fallback)."""
        name: str
        last_sync_time: float = 0.0
        last_game_count: int = 0
        total_games_synced: int = 0
        consecutive_failures: int = 0
        last_error: str = ""
        last_error_time: float = 0.0


@dataclass
class CollectorConfig:
    """Configuration for the data collector."""
    poll_interval_seconds: int = 60
    ephemeral_poll_interval_seconds: int = 15  # Aggressive sync for RAM disk hosts
    sync_method: str = "incremental"  # "incremental" or "full"
    deduplication: bool = True
    min_games_per_sync: int = 10
    ssh_timeout: int = 30
    rsync_timeout: int = 300
    max_consecutive_failures: int = 5
    backoff_multiplier: float = 2.0
    max_backoff_seconds: int = 600
    local_sync_dir: str = "data/games/synced"
    manifest_db_path: str = "data/data_manifest.db"
    # Hardening options
    checksum_validation: bool = True
    retry_max_attempts: int = 3
    retry_base_delay_seconds: float = 5.0
    dead_letter_enabled: bool = True
    dead_letter_dir: str = "data/dead_letter"
    # Write-ahead log for crash recovery
    wal_enabled: bool = True
    wal_db_path: str = "data/sync_wal.db"
    # Elo database replication
    elo_replication_enabled: bool = True
    elo_replication_interval_seconds: int = 60


# =============================================================================
# DataManifest - Legacy implementation (kept for fallback)
# For new code, use: from app.distributed.unified_manifest import DataManifest
# =============================================================================


class _LegacyDataManifest:
    """Tracks synced game IDs for deduplication (legacy fallback)."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the manifest database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS synced_games (
                game_id TEXT PRIMARY KEY,
                source_host TEXT NOT NULL,
                source_db TEXT NOT NULL,
                synced_at REAL NOT NULL,
                board_type TEXT,
                num_players INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_synced_games_host
            ON synced_games(source_host);

            CREATE INDEX IF NOT EXISTS idx_synced_games_time
            ON synced_games(synced_at);

            CREATE TABLE IF NOT EXISTS host_states (
                host_name TEXT PRIMARY KEY,
                last_sync_time REAL,
                last_game_count INTEGER,
                total_games_synced INTEGER,
                consecutive_failures INTEGER,
                last_error TEXT,
                last_error_time REAL
            );

            CREATE TABLE IF NOT EXISTS sync_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                host_name TEXT NOT NULL,
                sync_time REAL NOT NULL,
                games_synced INTEGER NOT NULL,
                duration_seconds REAL,
                success INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS dead_letter_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                source_host TEXT NOT NULL,
                source_db TEXT NOT NULL,
                error_message TEXT NOT NULL,
                error_type TEXT NOT NULL,
                added_at REAL NOT NULL,
                retry_count INTEGER DEFAULT 0,
                last_retry_at REAL,
                resolved INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_dead_letter_unresolved
            ON dead_letter_queue(resolved, added_at);
        """)
        conn.commit()
        conn.close()

    def is_game_synced(self, game_id: str) -> bool:
        """Check if a game has already been synced."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM synced_games WHERE game_id = ?", (game_id,))
        result = cursor.fetchone() is not None
        conn.close()
        return result

    def mark_games_synced(
        self,
        game_ids: List[str],
        source_host: str,
        source_db: str,
        board_type: Optional[str] = None,
        num_players: Optional[int] = None,
    ):
        """Mark games as synced."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = time.time()

        for game_id in game_ids:
            cursor.execute("""
                INSERT OR IGNORE INTO synced_games
                (game_id, source_host, source_db, synced_at, board_type, num_players)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (game_id, source_host, source_db, now, board_type, num_players))

        conn.commit()
        conn.close()

    def get_synced_count(self) -> int:
        """Get total number of synced games."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM synced_games")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def save_host_state(self, state: HostSyncState):
        """Save host sync state."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO host_states
            (host_name, last_sync_time, last_game_count, total_games_synced,
             consecutive_failures, last_error, last_error_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            state.name, state.last_sync_time, state.last_game_count,
            state.total_games_synced, state.consecutive_failures,
            state.last_error, state.last_error_time
        ))
        conn.commit()
        conn.close()

    def load_host_state(self, host_name: str) -> Optional[HostSyncState]:
        """Load host sync state."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT host_name, last_sync_time, last_game_count, total_games_synced,
                   consecutive_failures, last_error, last_error_time
            FROM host_states WHERE host_name = ?
        """, (host_name,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return HostSyncState(
                name=row[0],
                last_sync_time=row[1] or 0.0,
                last_game_count=row[2] or 0,
                total_games_synced=row[3] or 0,
                consecutive_failures=row[4] or 0,
                last_error=row[5] or "",
                last_error_time=row[6] or 0.0,
            )
        return None

    def record_sync(self, host_name: str, games_synced: int, duration: float, success: bool):
        """Record a sync event to history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO sync_history (host_name, sync_time, games_synced, duration_seconds, success)
            VALUES (?, ?, ?, ?, ?)
        """, (host_name, time.time(), games_synced, duration, int(success)))
        conn.commit()
        conn.close()

    def add_to_dead_letter(
        self,
        game_id: str,
        source_host: str,
        source_db: str,
        error_message: str,
        error_type: str,
    ):
        """Add a failed game to the dead-letter queue."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO dead_letter_queue
            (game_id, source_host, source_db, error_message, error_type, added_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (game_id, source_host, source_db, error_message, error_type, time.time()))
        conn.commit()
        conn.close()

    def get_dead_letter_count(self) -> int:
        """Get count of unresolved dead-letter entries."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM dead_letter_queue WHERE resolved = 0")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_dead_letter_entries(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get unresolved dead-letter entries for retry."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, game_id, source_host, source_db, error_message, error_type,
                   added_at, retry_count, last_retry_at
            FROM dead_letter_queue
            WHERE resolved = 0
            ORDER BY added_at ASC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "id": row[0],
                "game_id": row[1],
                "source_host": row[2],
                "source_db": row[3],
                "error_message": row[4],
                "error_type": row[5],
                "added_at": row[6],
                "retry_count": row[7],
                "last_retry_at": row[8],
            }
            for row in rows
        ]

    def mark_dead_letter_resolved(self, entry_id: int):
        """Mark a dead-letter entry as resolved."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE dead_letter_queue SET resolved = 1 WHERE id = ?",
            (entry_id,)
        )
        conn.commit()
        conn.close()

    def increment_dead_letter_retry(self, entry_id: int):
        """Increment retry count for a dead-letter entry."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE dead_letter_queue
            SET retry_count = retry_count + 1, last_retry_at = ?
            WHERE id = ?
        """, (time.time(), entry_id))
        conn.commit()
        conn.close()


# DataManifest - use unified implementation if available
if HAS_UNIFIED_MANIFEST:
    DataManifest = UnifiedDataManifest
else:
    DataManifest = _LegacyDataManifest


class StreamingDataCollector:
    """Continuous data collection from cluster hosts."""

    def __init__(
        self,
        config: CollectorConfig,
        hosts: List[HostConfig],
        manifest: DataManifest,
        http_port: int = 8772,
        manifest_replicator: Optional[Any] = None,
        p2p_fallback: Optional[Any] = None,
        wal: Optional[Any] = None,
        elo_replicator: Optional[Any] = None,
    ):
        self.config = config
        self.hosts = {h.name: h for h in hosts}
        self.manifest = manifest
        self.host_states: Dict[str, HostSyncState] = {}
        self._running = False
        self._shutdown_event = asyncio.Event()
        self.http_port = http_port
        self._app: Optional[Any] = None
        self._http_runner: Optional[Any] = None
        self._last_cycle_time: float = 0.0
        self._last_cycle_games: int = 0
        self._manifest_replicator = manifest_replicator
        self._p2p_fallback = p2p_fallback
        self._wal = wal
        self._elo_replicator = elo_replicator

        # Track sync method statistics
        self._sync_stats = {"ssh": 0, "p2p_http": 0, "failed": 0}

        # Classify hosts by storage type for differentiated sync intervals
        self._ephemeral_hosts: Set[str] = set()
        self._persistent_hosts: Set[str] = set()
        for host in hosts:
            if getattr(host, 'is_ephemeral', False):
                self._ephemeral_hosts.add(host.name)
            else:
                self._persistent_hosts.add(host.name)

        # Track last sync time per category for differentiated intervals
        self._last_ephemeral_sync: float = 0.0
        self._last_persistent_sync: float = 0.0
        self._last_elo_replication: float = 0.0

        # Load previous host states
        for host in hosts:
            state = manifest.load_host_state(host.name)
            if state:
                self.host_states[host.name] = state
            else:
                self.host_states[host.name] = HostSyncState(name=host.name)

        # Log ephemeral host detection
        if self._ephemeral_hosts:
            print(f"[Collector] Detected {len(self._ephemeral_hosts)} ephemeral hosts "
                  f"(aggressive {config.ephemeral_poll_interval_seconds}s sync): "
                  f"{list(self._ephemeral_hosts)}")

    async def _get_remote_game_count(self, host: HostConfig) -> int:
        """Get the current game count on a remote host."""
        ssh_args = self._build_ssh_args(host)

        # Query all DBs for game counts
        cmd = f"""ssh {ssh_args} {host.ssh_user}@{host.ssh_host} "
            cd {host.remote_db_path} 2>/dev/null && \\
            for db in *.db; do \\
                [ -f \\"\\$db\\" ] && sqlite3 \\"\\$db\\" 'SELECT COUNT(*) FROM games' 2>/dev/null || true; \\
            done | awk '{{s+=\\$1}} END {{print s+0}}'
        " """

        try:
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.ssh_timeout
            )

            count = int(stdout.decode().strip() or "0")
            return count

        except (asyncio.TimeoutError, ValueError) as e:
            raise RuntimeError(f"Failed to get game count: {e}")

    def _build_ssh_args(self, host: HostConfig) -> str:
        """Build SSH arguments string."""
        args = [f"-o ConnectTimeout={self.config.ssh_timeout}"]

        if host.ssh_port != 22:
            args.append(f"-p {host.ssh_port}")

        if host.ssh_key:
            args.append(f"-i {host.ssh_key}")

        return " ".join(args)

    def _release_resources(self, bandwidth_allocated: bool, sync_lock_acquired: bool, host: HostConfig) -> None:
        """Release bandwidth and sync lock resources."""
        if bandwidth_allocated and HAS_BANDWIDTH_MANAGER:
            try:
                release_bandwidth(host.ssh_host)
            except Exception as e:
                print(f"[Collector] {host.name}: bandwidth release error: {e}")

        if sync_lock_acquired and HAS_SYNC_LOCK:
            try:
                release_sync_lock(host.name)
            except Exception as e:
                print(f"[Collector] {host.name}: sync lock release error: {e}")

    async def _sync_host(self, host: HostConfig) -> int:
        """Sync games from a single host. Returns count of new games."""
        state = self.host_states[host.name]

        # Circuit breaker check - skip hosts with open circuits
        if HAS_CIRCUIT_BREAKER:
            breaker = get_host_breaker()
            if not breaker.can_execute(host.ssh_host):
                circuit_state = breaker.get_state(host.ssh_host)
                if circuit_state == CircuitState.OPEN:
                    print(f"[Collector] {host.name}: Circuit OPEN, skipping sync")
                return 0

        # Check backoff for failed hosts
        if state.consecutive_failures > 0:
            backoff = min(
                self.config.max_backoff_seconds,
                self.config.poll_interval_seconds * (self.config.backoff_multiplier ** state.consecutive_failures)
            )
            if time.time() - state.last_error_time < backoff:
                return 0

        start_time = time.time()

        try:
            # Get current game count
            current_count = await self._get_remote_game_count(host)
            new_games = max(0, current_count - state.last_game_count)

            if new_games < self.config.min_games_per_sync:
                # Record success even if no games - connection worked
                if HAS_CIRCUIT_BREAKER:
                    get_host_breaker().record_success(host.ssh_host)
                return 0

            print(f"[Collector] {host.name}: {new_games} new games detected")

            # Perform sync
            if self.config.sync_method == "incremental":
                synced = await self._incremental_sync(host)
            else:
                synced = await self._full_sync(host)

            # Update state
            state.last_sync_time = time.time()
            state.last_game_count = current_count
            state.total_games_synced += synced
            state.consecutive_failures = 0
            state.last_error = ""

            # Record sync
            duration = time.time() - start_time
            self.manifest.record_sync(host.name, synced, duration, True)
            self.manifest.save_host_state(state)

            # Record circuit breaker success
            if HAS_CIRCUIT_BREAKER:
                get_host_breaker().record_success(host.ssh_host)

            # Emit event to in-memory bus
            if HAS_EVENT_BUS and synced > 0:
                await emit_new_games(host.name, synced, current_count, "streaming_data_collector")

            # Bridge to cross-process queue for other daemons
            if synced > 0:
                bridge_event_to_cross_process(
                    event_type="new_games",
                    payload={
                        "host": host.name,
                        "new_games": synced,
                        "total_games": current_count,
                    },
                    source="streaming_data_collector",
                )

            return synced

        except CircuitOpenError:
            # Circuit just opened - skip without incrementing failures
            print(f"[Collector] {host.name}: Circuit opened during sync")
            return 0

        except Exception as e:
            state.consecutive_failures += 1
            state.last_error = str(e)
            state.last_error_time = time.time()

            duration = time.time() - start_time
            self.manifest.record_sync(host.name, 0, duration, False)
            self.manifest.save_host_state(state)

            # Record circuit breaker failure
            if HAS_CIRCUIT_BREAKER:
                get_host_breaker().record_failure(host.ssh_host, e)

            print(f"[Collector] {host.name}: Sync failed ({state.consecutive_failures}): {e}")

            # Bridge failure to cross-process for distributed awareness
            bridge_event_to_cross_process(
                event_type="sync_failed",
                payload={
                    "host": host.name,
                    "error": str(e)[:200],
                    "retry_count": state.consecutive_failures,
                },
                source="streaming_data_collector",
            )

            if state.consecutive_failures >= self.config.max_consecutive_failures:
                print(f"[Collector] {host.name}: Disabling after {state.consecutive_failures} failures")
                # Bridge host offline event
                bridge_event_to_cross_process(
                    event_type="host_offline",
                    payload={
                        "host": host.name,
                        "reason": f"max_failures_exceeded ({state.consecutive_failures})",
                    },
                    source="streaming_data_collector",
                )

            return 0

    async def _sync_with_retry(self, host: HostConfig) -> int:
        """Sync with exponential backoff retry."""
        last_error = None

        for attempt in range(self.config.retry_max_attempts):
            try:
                return await self._sync_host(host)
            except Exception as e:
                last_error = e
                if attempt < self.config.retry_max_attempts - 1:
                    delay = self.config.retry_base_delay_seconds * (2 ** attempt)
                    print(f"[Collector] {host.name}: Retry {attempt + 1}/{self.config.retry_max_attempts} after {delay}s")
                    await asyncio.sleep(delay)

        # All retries failed - add to dead-letter if enabled
        if self.config.dead_letter_enabled and last_error:
            self.manifest.add_to_dead_letter(
                game_id=f"sync_failure_{host.name}_{time.time()}",
                source_host=host.name,
                source_db="*",
                error_message=str(last_error),
                error_type=type(last_error).__name__,
            )

        raise last_error if last_error else RuntimeError("Unknown sync error")

    def _compute_db_checksum(self, db_path: Path) -> str:
        """Compute SHA256 checksum of a database file."""
        import hashlib
        sha256 = hashlib.sha256()
        with open(db_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    async def _validate_game_integrity(self, db_path: Path, host_name: str) -> List[str]:
        """Validate game integrity in a synced database.

        Returns list of valid game IDs. Invalid games are added to dead-letter queue.
        """
        valid_games = []
        invalid_games = []

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Get all games with their essential fields
            cursor.execute("""
                SELECT game_id, board_type, num_players, moves, game_length
                FROM games
            """)

            for row in cursor.fetchall():
                game_id, board_type, num_players, moves, game_length = row

                # Basic integrity checks
                errors = []
                if not game_id:
                    errors.append("missing game_id")
                if not board_type:
                    errors.append("missing board_type")
                if num_players is None or num_players < 2:
                    errors.append(f"invalid num_players: {num_players}")
                if game_length is not None and game_length < 0:
                    errors.append(f"invalid game_length: {game_length}")

                if errors:
                    invalid_games.append((game_id or "unknown", ", ".join(errors)))
                else:
                    valid_games.append(game_id)

            conn.close()

            # Add invalid games to dead-letter queue
            if self.config.dead_letter_enabled:
                for game_id, error_msg in invalid_games:
                    self.manifest.add_to_dead_letter(
                        game_id=game_id,
                        source_host=host_name,
                        source_db=db_path.name,
                        error_message=error_msg,
                        error_type="integrity_check_failed",
                    )

            if invalid_games:
                print(f"[Collector] {host_name}: {len(invalid_games)} invalid games added to dead-letter queue")

        except Exception as e:
            print(f"[Collector] {host_name}: Validation error: {e}")

        return valid_games

    async def _incremental_sync(self, host: HostConfig) -> int:
        """Perform incremental rsync with P2P HTTP fallback. Returns count of synced games."""
        local_dir = AI_SERVICE_ROOT / self.config.local_sync_dir / host.name
        local_dir.mkdir(parents=True, exist_ok=True)

        ssh_args = self._build_ssh_args(host)
        # Add checksum for rsync integrity
        rsync_cmd = f'rsync -avz --checksum --progress -e "ssh {ssh_args}" {host.ssh_user}@{host.ssh_host}:{host.remote_db_path}/*.db {local_dir}/'

        # Acquire sync_lock to prevent concurrent rsync to same host
        sync_lock_acquired = False
        if HAS_SYNC_LOCK:
            try:
                sync_lock_acquired = acquire_sync_lock(host.name, "rsync-inbound", wait=True, wait_timeout=60.0)
                if not sync_lock_acquired:
                    print(f"[Collector] {host.name}: could not acquire sync lock, skipping")
                    return 0
            except Exception as e:
                print(f"[Collector] {host.name}: sync lock error: {e}")
                # Continue without lock

        # Request bandwidth allocation for the transfer
        bandwidth_allocated = False
        if HAS_BANDWIDTH_MANAGER:
            try:
                bandwidth_allocated = request_bandwidth(
                    host.ssh_host, estimated_mb=100, priority=TransferPriority.NORMAL, timeout=30.0
                )
                if not bandwidth_allocated:
                    print(f"[Collector] {host.name}: bandwidth unavailable, proceeding anyway")
            except Exception as e:
                print(f"[Collector] {host.name}: bandwidth request error: {e}")

        rsync_failed = False
        rsync_error = ""

        try:
            process = await asyncio.create_subprocess_shell(
                rsync_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.rsync_timeout
            )

            if process.returncode != 0:
                rsync_failed = True
                rsync_error = stderr.decode()[:200]

        except asyncio.TimeoutError:
            rsync_failed = True
            rsync_error = "timeout"
        except Exception as e:
            rsync_failed = True
            rsync_error = str(e)[:200]

        # P2P HTTP fallback if rsync failed
        if rsync_failed:
            if self._p2p_fallback and HAS_P2P_FALLBACK:
                print(f"[Collector] {host.name}: rsync failed ({rsync_error}), trying P2P fallback")
                try:
                    success, games, method = await self._p2p_fallback.sync_with_fallback(
                        host=host.name,
                        ssh_host=host.ssh_host,
                        ssh_user=host.ssh_user,
                        ssh_port=host.ssh_port,
                        remote_db_path=host.remote_db_path,
                        local_dir=local_dir,
                    )
                    if success:
                        self._sync_stats["p2p_http"] += 1
                        print(f"[Collector] {host.name}: P2P fallback succeeded")
                    else:
                        self._sync_stats["failed"] += 1
                        self._release_resources(bandwidth_allocated, sync_lock_acquired, host)
                        raise RuntimeError("Both rsync and P2P fallback failed")
                except RuntimeError:
                    raise
                except Exception as e:
                    self._sync_stats["failed"] += 1
                    self._release_resources(bandwidth_allocated, sync_lock_acquired, host)
                    raise RuntimeError(f"P2P fallback failed: {e}")
            else:
                self._sync_stats["failed"] += 1
                self._release_resources(bandwidth_allocated, sync_lock_acquired, host)
                raise RuntimeError(f"rsync failed and no P2P fallback: {rsync_error}")
        else:
            self._sync_stats["ssh"] += 1

        # Release resources on success
        self._release_resources(bandwidth_allocated, sync_lock_acquired, host)

        # Count and validate games in synced DBs
        total = 0
        for db_file in local_dir.glob("*.db"):
            try:
                if self.config.checksum_validation:
                    # Validate game integrity
                    valid_games = await self._validate_game_integrity(db_file, host.name)
                    total += len(valid_games)
                else:
                    # Just count games
                    conn = sqlite3.connect(db_file)
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM games")
                    total += cursor.fetchone()[0]
                    conn.close()
            except Exception as e:
                print(f"[Collector] Error processing {db_file}: {e}")

        return total

    async def _full_sync(self, host: HostConfig) -> int:
        """Full sync (same as incremental for now)."""
        return await self._incremental_sync(host)

    async def run_collection_cycle(
        self,
        ephemeral_only: bool = False,
        persistent_only: bool = False,
    ) -> int:
        """Run one data collection cycle. Returns total new games.

        Args:
            ephemeral_only: Only sync ephemeral (RAM disk) hosts
            persistent_only: Only sync persistent storage hosts
        """
        total_new = 0
        tasks = []

        for host in self.hosts.values():
            if not host.enabled:
                continue

            # Filter by host type if requested
            host_is_ephemeral = host.name in self._ephemeral_hosts
            if ephemeral_only and not host_is_ephemeral:
                continue
            if persistent_only and host_is_ephemeral:
                continue

            state = self.host_states.get(host.name)
            if state and state.consecutive_failures >= self.config.max_consecutive_failures:
                continue
            tasks.append(self._sync_host(host))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, int):
                    total_new += result

        return total_new

    async def run_ephemeral_sync(self) -> int:
        """Run sync cycle for ephemeral hosts only (aggressive interval)."""
        if not self._ephemeral_hosts:
            return 0
        return await self.run_collection_cycle(ephemeral_only=True)

    async def run_persistent_sync(self) -> int:
        """Run sync cycle for persistent hosts only (normal interval)."""
        if not self._persistent_hosts:
            return 0
        return await self.run_collection_cycle(persistent_only=True)

    async def run(self):
        """Main collection loop."""
        self._running = True
        print(f"[Collector] Starting with {len(self.hosts)} hosts, {self.config.poll_interval_seconds}s interval")

        # Recover manifest from replicas if needed (on startup)
        if self._manifest_replicator:
            try:
                recovered = await self._manifest_replicator.recover_if_needed()
                if recovered:
                    print("[Collector] Recovered manifest from replica")
                    # Reload host states after recovery
                    for host in self.hosts.values():
                        state = self.manifest.load_host_state(host.name)
                        if state:
                            self.host_states[host.name] = state
            except Exception as e:
                print(f"[Collector] Manifest recovery error (continuing with local): {e}")

        # Acquire DATA_SYNC role via OrchestratorRegistry
        has_role = False
        if HAS_ORCHESTRATOR_REGISTRY:
            try:
                registry = get_registry()
                import socket
                node_id = socket.gethostname()
                has_role = registry.try_acquire(OrchestratorRole.DATA_SYNC, node_id)
                if has_role:
                    print("[Collector] Acquired DATA_SYNC orchestrator role")
                else:
                    print("[Collector] Warning: Could not acquire DATA_SYNC role (another collector may be running)")
            except Exception as e:
                print(f"[Collector] OrchestratorRegistry error: {e}")

        # Start HTTP API
        await self._setup_http()

        heartbeat_interval = 30
        last_heartbeat = time.time()
        last_replication = time.time()
        replication_interval = 60  # Replicate manifest every minute (was 5 min)

        # Differentiated sync intervals
        ephemeral_interval = self.config.ephemeral_poll_interval_seconds  # 15s default
        persistent_interval = self.config.poll_interval_seconds  # 60s default
        elo_replication_interval = getattr(self.config, 'elo_replication_interval_seconds', 60)

        if self._ephemeral_hosts:
            print(f"[Collector] Ephemeral hosts ({len(self._ephemeral_hosts)}): "
                  f"sync every {ephemeral_interval}s")
        if self._persistent_hosts:
            print(f"[Collector] Persistent hosts ({len(self._persistent_hosts)}): "
                  f"sync every {persistent_interval}s")

        try:
            while self._running:
                try:
                    cycle_start = time.time()
                    new_games = 0

                    # Sync ephemeral hosts more frequently (aggressive interval)
                    if self._ephemeral_hosts and                        (cycle_start - self._last_ephemeral_sync) >= ephemeral_interval:
                        ephemeral_games = await self.run_ephemeral_sync()
                        new_games += ephemeral_games
                        self._last_ephemeral_sync = cycle_start
                        if ephemeral_games > 0:
                            print(f"[Collector] Ephemeral sync: {ephemeral_games} games "
                                  f"(priority hosts: {list(self._ephemeral_hosts)})")

                    # Sync persistent hosts at normal interval
                    if self._persistent_hosts and                        (cycle_start - self._last_persistent_sync) >= persistent_interval:
                        persistent_games = await self.run_persistent_sync()
                        new_games += persistent_games
                        self._last_persistent_sync = cycle_start

                    # Track cycle metrics
                    self._last_cycle_time = time.time()
                    self._last_cycle_games = new_games

                    if new_games > 0:
                        total = self.manifest.get_synced_count()
                        print(f"[Collector] Cycle complete: {new_games} new games (total: {total})")

                        # Replicate manifest after successful sync (rate-limited)
                        if self._manifest_replicator and (time.time() - last_replication) >= replication_interval:
                            try:
                                replicas = await self._manifest_replicator.replicate_async()
                                if replicas > 0:
                                    print(f"[Collector] Manifest replicated to {replicas} hosts")
                                last_replication = time.time()
                            except Exception as e:
                                print(f"[Collector] Manifest replication error: {e}")

                    # Replicate Elo database (rate-limited)
                    if self._elo_replicator and (time.time() - self._last_elo_replication) >= elo_replication_interval:
                        try:
                            elo_replicas = await self._elo_replicator.replicate()
                            if elo_replicas > 0:
                                print(f"[Collector] Elo DB replicated to {elo_replicas} hosts")
                            self._last_elo_replication = time.time()
                        except Exception as e:
                            print(f"[Collector] Elo replication error: {e}")

                    # Heartbeat for OrchestratorRegistry
                    if HAS_ORCHESTRATOR_REGISTRY and has_role and (time.time() - last_heartbeat) >= heartbeat_interval:
                        try:
                            registry.heartbeat(OrchestratorRole.DATA_SYNC)
                            last_heartbeat = time.time()
                        except Exception as e:
                            print(f"[Collector] Heartbeat error: {e}")

                except Exception as e:
                    print(f"[Collector] Cycle error: {e}")

                # Wait for next cycle - use minimum of ephemeral and persistent intervals
                elapsed = time.time() - cycle_start
                min_interval = min(
                    ephemeral_interval if self._ephemeral_hosts else persistent_interval,
                    persistent_interval
                )
                sleep_time = max(0, min_interval - elapsed)

                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=sleep_time)
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    pass
        finally:
            # Release OrchestratorRegistry role on shutdown
            if HAS_ORCHESTRATOR_REGISTRY and has_role:
                try:
                    registry.release(OrchestratorRole.DATA_SYNC)
                    print("[Collector] Released DATA_SYNC orchestrator role")
                except Exception as e:
                    print(f"[Collector] Error releasing role: {e}")

        # Cleanup HTTP
        await self._cleanup_http()
        print("[Collector] Stopped")

    def stop(self):
        """Request graceful shutdown."""
        self._running = False
        self._shutdown_event.set()

    # HTTP API methods
    async def _setup_http(self):
        """Set up HTTP API server."""
        try:
            from aiohttp import web
        except ImportError:
            print("[Collector] aiohttp not installed, HTTP API disabled")
            return

        self._app = web.Application()
        self._app.router.add_get('/health', self._handle_health)
        self._app.router.add_get('/status', self._handle_status)
        self._app.router.add_get('/hosts', self._handle_hosts)
        self._app.router.add_get('/dead-letter', self._handle_dead_letter)
        self._app.router.add_post('/trigger', self._handle_trigger)

        self._http_runner = web.AppRunner(self._app)
        await self._http_runner.setup()
        site = web.TCPSite(self._http_runner, '0.0.0.0', self.http_port)
        await site.start()
        print(f"[Collector] HTTP API listening on port {self.http_port}")

    async def _cleanup_http(self):
        """Clean up HTTP server."""
        if self._http_runner:
            await self._http_runner.cleanup()

    async def _handle_health(self, request) -> Any:
        """GET /health - Health check."""
        from aiohttp import web
        return web.json_response({"status": "healthy", "running": self._running})

    async def _handle_status(self, request) -> Any:
        """GET /status - Collector status."""
        from aiohttp import web

        status = {
            "running": self._running,
            "poll_interval": self.config.poll_interval_seconds,
            "total_synced": self.manifest.get_synced_count(),
            "dead_letter_count": self.manifest.get_dead_letter_count(),
            "hosts_count": len(self.hosts),
            "last_cycle_time": self._last_cycle_time,
            "last_cycle_games": self._last_cycle_games,
            "sync_method": self.config.sync_method,
        }
        return web.json_response(status)

    async def _handle_hosts(self, request) -> Any:
        """GET /hosts - Host status summary."""
        from aiohttp import web

        hosts = []
        for name, state in self.host_states.items():
            host = self.hosts.get(name)
            hosts.append({
                "name": name,
                "enabled": host.enabled if host else False,
                "role": host.role if host else "unknown",
                "last_sync_time": state.last_sync_time,
                "last_game_count": state.last_game_count,
                "total_games_synced": state.total_games_synced,
                "consecutive_failures": state.consecutive_failures,
                "last_error": state.last_error[:100] if state.last_error else "",
                "healthy": state.consecutive_failures < self.config.max_consecutive_failures,
            })
        return web.json_response(hosts)

    async def _handle_dead_letter(self, request) -> Any:
        """GET /dead-letter - Dead letter queue entries."""
        from aiohttp import web

        limit = int(request.query.get('limit', '50'))
        entries = self.manifest.get_dead_letter_entries(limit)
        return web.json_response({
            "count": self.manifest.get_dead_letter_count(),
            "entries": entries,
        })

    async def _handle_trigger(self, request) -> Any:
        """POST /trigger - Trigger sync cycle manually."""
        from aiohttp import web

        try:
            data = await request.json()
        except Exception:
            data = {}

        host_filter = data.get('host')

        if host_filter:
            host = self.hosts.get(host_filter)
            if host:
                asyncio.create_task(self._sync_host(host))
                return web.json_response({"triggered": host_filter, "status": "started"})
            return web.json_response({"error": f"Host {host_filter} not found"}, status=404)
        else:
            asyncio.create_task(self.run_collection_cycle())
            return web.json_response({"triggered": "all", "status": "started"})


def load_hosts_from_yaml(path: Path) -> List[HostConfig]:
    """Load host configurations from YAML file."""
    if not path.exists():
        return []

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    hosts = []

    # Load standard hosts
    for name, host_data in data.get("standard_hosts", {}).items():
        hosts.append(HostConfig(
            name=name,
            ssh_host=host_data.get("ssh_host", ""),
            ssh_user=host_data.get("ssh_user", "ubuntu"),
            ssh_port=host_data.get("ssh_port", 22),
            remote_db_path=host_data.get("remote_path", "~/ringrift/ai-service/data/games"),
            role=host_data.get("role", "selfplay"),
        ))

    # Load vast hosts (ephemeral - RAM disk storage)
    for name, host_data in data.get("vast_hosts", {}).items():
        # Determine if ephemeral based on path or explicit storage_type
        remote_path = host_data.get("remote_path", "/dev/shm/games")
        storage_type = host_data.get("storage_type", "")
        is_ephemeral = (
            storage_type == "ram" or
            "/dev/shm" in remote_path or
            "/run/shm" in remote_path
        )
        hosts.append(HostConfig(
            name=name,
            ssh_host=host_data.get("host", ""),
            ssh_user=host_data.get("user", "root"),
            ssh_port=host_data.get("port", 22),
            remote_db_path=remote_path,
            role=host_data.get("role", "selfplay"),
            storage_type="ephemeral" if is_ephemeral else "persistent",
            is_ephemeral=is_ephemeral,
        ))

    return hosts


def main():
    parser = argparse.ArgumentParser(description="Streaming Data Collector")
    parser.add_argument("--config", type=str, default="config/unified_loop.yaml", help="Config file")
    parser.add_argument("--hosts", type=str, default="config/remote_hosts.yaml", help="Hosts file")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--dry-run", action="store_true", help="Check what would sync without syncing")
    parser.add_argument("--interval", type=int, default=60, help="Poll interval in seconds")
    parser.add_argument("--http-port", type=int, default=8772, help="HTTP API port")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Load configuration
    config = CollectorConfig(poll_interval_seconds=args.interval)

    config_path = AI_SERVICE_ROOT / args.config
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        if "data_ingestion" in data:
            for key, value in data["data_ingestion"].items():
                if hasattr(config, key):
                    setattr(config, key, value)

    # Load hosts
    hosts = load_hosts_from_yaml(AI_SERVICE_ROOT / args.hosts)
    if not hosts:
        print("No hosts configured")
        return

    print(f"[Collector] Loaded {len(hosts)} hosts")

    # Initialize manifest
    manifest = DataManifest(AI_SERVICE_ROOT / config.manifest_db_path)

    # Initialize manifest replicator for fault tolerance
    manifest_replicator = None
    if HAS_MANIFEST_REPLICATION:
        try:
            manifest_replicator = create_replicator_from_config(
                manifest_path=AI_SERVICE_ROOT / config.manifest_db_path,
                hosts_config_path=AI_SERVICE_ROOT / args.hosts,
                min_replicas=2,
            )
            print(f"[Collector] Manifest replication enabled with {len(manifest_replicator.replica_hosts)} replica hosts")
        except Exception as e:
            print(f"[Collector] Warning: Could not initialize manifest replication: {e}")

    # Initialize P2P fallback sync for redundant data transfer
    p2p_fallback = None
    if HAS_P2P_FALLBACK:
        try:
            p2p_fallback = P2PFallbackSync(p2p_port=8770)
            print("[Collector] P2P HTTP fallback enabled")
        except Exception as e:
            print(f"[Collector] Warning: Could not initialize P2P fallback: {e}")

    # Initialize Write-Ahead Log for crash recovery
    wal = None
    if HAS_ROBUST_SYNC and getattr(config, 'wal_enabled', True):
        try:
            wal_path = AI_SERVICE_ROOT / getattr(config, 'wal_db_path', 'data/sync_wal.db')
            wal = WriteAheadLog(wal_path)
            print(f"[Collector] Write-ahead log enabled: {wal_path}")
        except Exception as e:
            print(f"[Collector] Warning: Could not initialize WAL: {e}")

    # Initialize Elo database replicator
    elo_replicator = None
    if HAS_ROBUST_SYNC and getattr(config, 'elo_replication_enabled', True):
        try:
            # Load hosts config for replica selection
            with open(AI_SERVICE_ROOT / args.hosts) as f:
                hosts_config = yaml.safe_load(f) or {}
            elo_replicator = create_elo_replicator(
                data_dir=AI_SERVICE_ROOT / "data",
                hosts_config=hosts_config,
                min_replicas=2,
            )
            if elo_replicator:
                print(f"[Collector] Elo replication enabled with "
                      f"{len(elo_replicator.replica_hosts)} replica hosts")
        except Exception as e:
            print(f"[Collector] Warning: Could not initialize Elo replicator: {e}")

    # Create collector
    collector = StreamingDataCollector(
        config, hosts, manifest,
        http_port=args.http_port,
        manifest_replicator=manifest_replicator,
        p2p_fallback=p2p_fallback,
        wal=wal,
        elo_replicator=elo_replicator,
    )

    if args.dry_run:
        print("[Collector] Dry run - checking hosts...")
        for host in hosts:
            print(f"  {host.name}: {host.ssh_user}@{host.ssh_host}:{host.ssh_port}")
        return

    # Handle signals
    import signal

    def signal_handler(sig, frame):
        print("\n[Collector] Shutdown requested")
        collector.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run
    if args.once:
        asyncio.run(collector.run_collection_cycle())
    else:
        asyncio.run(collector.run())


if __name__ == "__main__":
    main()
