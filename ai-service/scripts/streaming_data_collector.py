#!/usr/bin/env python3
"""Streaming Data Collector - Continuous incremental sync from cluster hosts.

This service replaces batch data sync (30-min intervals) with continuous
60-second polling for new games. Key features:

1. Incremental sync via rsync --append
2. Game ID deduplication across all sources
3. Event emission for downstream triggers
4. Per-host manifest tracking
5. Automatic retry with exponential backoff

Usage:
    # Run as standalone service
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

import yaml

# Allow imports from app/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]

# Try to import event bus
try:
    from app.distributed.data_events import (
        DataEventType,
        DataEvent,
        get_event_bus,
        emit_new_games,
    )
    HAS_EVENT_BUS = True
except ImportError:
    HAS_EVENT_BUS = False

# Try to import sync_lock for coordinating rsync operations
try:
    from app.coordination.sync_mutex import acquire_sync_lock, release_sync_lock
    HAS_SYNC_LOCK = True
except ImportError:
    HAS_SYNC_LOCK = False

# Try to import OrchestratorRegistry for daemon role management
try:
    from app.coordination.orchestrator_registry import (
        OrchestratorRole,
        orchestrator_role,
        get_registry,
    )
    HAS_ORCHESTRATOR_REGISTRY = True
except ImportError:
    HAS_ORCHESTRATOR_REGISTRY = False
    OrchestratorRole = None

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


@dataclass
class HostSyncState:
    """Sync state for a host."""
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


class DataManifest:
    """Tracks synced game IDs for deduplication."""

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


class StreamingDataCollector:
    """Continuous data collection from cluster hosts."""

    def __init__(
        self,
        config: CollectorConfig,
        hosts: List[HostConfig],
        manifest: DataManifest,
        http_port: int = 8772,
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

        # Load previous host states
        for host in hosts:
            state = manifest.load_host_state(host.name)
            if state:
                self.host_states[host.name] = state
            else:
                self.host_states[host.name] = HostSyncState(name=host.name)

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

            # Emit event
            if HAS_EVENT_BUS and synced > 0:
                await emit_new_games(host.name, synced, current_count, "streaming_data_collector")

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

            if state.consecutive_failures >= self.config.max_consecutive_failures:
                print(f"[Collector] {host.name}: Disabling after {state.consecutive_failures} failures")

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
        """Perform incremental rsync. Returns count of synced games."""
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
                raise RuntimeError(f"rsync failed: {stderr.decode()}")

        finally:
            # Release bandwidth allocation
            if bandwidth_allocated and HAS_BANDWIDTH_MANAGER:
                try:
                    release_bandwidth(host.ssh_host)
                except Exception as e:
                    print(f"[Collector] {host.name}: bandwidth release error: {e}")

            # Release sync_lock
            if sync_lock_acquired and HAS_SYNC_LOCK:
                try:
                    release_sync_lock(host.name)
                except Exception as e:
                    print(f"[Collector] {host.name}: sync lock release error: {e}")

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

    async def run_collection_cycle(self) -> int:
        """Run one data collection cycle. Returns total new games."""
        total_new = 0
        tasks = []

        for host in self.hosts.values():
            if not host.enabled:
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

    async def run(self):
        """Main collection loop."""
        self._running = True
        print(f"[Collector] Starting with {len(self.hosts)} hosts, {self.config.poll_interval_seconds}s interval")

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

        try:
            while self._running:
                try:
                    cycle_start = time.time()
                    new_games = await self.run_collection_cycle()

                    # Track cycle metrics
                    self._last_cycle_time = time.time()
                    self._last_cycle_games = new_games

                    if new_games > 0:
                        total = self.manifest.get_synced_count()
                        print(f"[Collector] Cycle complete: {new_games} new games (total: {total})")

                    # Heartbeat for OrchestratorRegistry
                    if HAS_ORCHESTRATOR_REGISTRY and has_role and (time.time() - last_heartbeat) >= heartbeat_interval:
                        try:
                            registry.heartbeat(OrchestratorRole.DATA_SYNC)
                            last_heartbeat = time.time()
                        except Exception as e:
                            print(f"[Collector] Heartbeat error: {e}")

                except Exception as e:
                    print(f"[Collector] Cycle error: {e}")

                # Wait for next cycle
                elapsed = time.time() - cycle_start
                sleep_time = max(0, self.config.poll_interval_seconds - elapsed)

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

    # Load vast hosts
    for name, host_data in data.get("vast_hosts", {}).items():
        hosts.append(HostConfig(
            name=name,
            ssh_host=host_data.get("host", ""),
            ssh_user=host_data.get("user", "root"),
            ssh_port=host_data.get("port", 22),
            remote_db_path=host_data.get("remote_path", "/dev/shm/games"),
            role=host_data.get("role", "selfplay"),
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

    # Create collector
    collector = StreamingDataCollector(config, hosts, manifest, http_port=args.http_port)

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
