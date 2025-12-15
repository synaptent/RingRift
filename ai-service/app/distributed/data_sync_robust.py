"""Robust Data Sync - Enhanced data synchronization with fault tolerance.

This module provides robust, efficient data synchronization across distributed hosts:

1. Aggressive sync for ephemeral hosts (Vast.ai with RAM storage) - 15s interval
2. Write-ahead log (WAL) for game data - prevents data loss on crash
3. Elo database replication - distributes unified_elo.db across cluster
4. Parallel batch sync - concurrent multi-host rsync operations

Usage:
    from app.distributed.data_sync_robust import (
        RobustDataSync,
        WriteAheadLog,
        EloReplicator,
        get_ephemeral_hosts,
    )

    # Create robust sync manager
    sync = RobustDataSync(hosts, config)
    await sync.run()
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Storage Types and Host Classification
# =============================================================================


class StorageType(str, Enum):
    """Storage type classification for hosts."""
    PERSISTENT = "persistent"  # Normal disk storage
    EPHEMERAL = "ephemeral"    # RAM disk (/dev/shm) - data lost on termination
    SSD = "ssd"                # Fast SSD storage
    NETWORK = "network"        # Network-attached storage


@dataclass
class HostSyncProfile:
    """Sync profile for a host based on its characteristics."""
    host_name: str
    storage_type: StorageType = StorageType.PERSISTENT
    poll_interval_seconds: int = 60
    priority: int = 1  # Higher = sync first
    max_parallel_transfers: int = 2
    compress_in_transit: bool = False
    # Ephemeral-specific settings
    is_ephemeral: bool = False
    aggressive_sync: bool = False  # True for ephemeral hosts
    last_sync_time: float = 0.0
    games_at_risk: int = 0  # Estimated games that could be lost

    @classmethod
    def for_ephemeral_host(cls, host_name: str) -> "HostSyncProfile":
        """Create profile for ephemeral (RAM disk) host."""
        return cls(
            host_name=host_name,
            storage_type=StorageType.EPHEMERAL,
            poll_interval_seconds=15,  # Aggressive: 15s instead of 60s
            priority=10,  # High priority
            max_parallel_transfers=4,  # More parallelism
            compress_in_transit=True,  # Compress to speed up
            is_ephemeral=True,
            aggressive_sync=True,
        )

    @classmethod
    def for_persistent_host(cls, host_name: str) -> "HostSyncProfile":
        """Create profile for persistent storage host."""
        return cls(
            host_name=host_name,
            storage_type=StorageType.PERSISTENT,
            poll_interval_seconds=60,
            priority=1,
            max_parallel_transfers=2,
            compress_in_transit=False,
            is_ephemeral=False,
            aggressive_sync=False,
        )


def classify_host_storage(host_config: Dict[str, Any]) -> StorageType:
    """Classify host storage type from configuration."""
    # Explicit storage_type in config
    storage_type = host_config.get("storage_type", "").lower()
    if storage_type == "ram":
        return StorageType.EPHEMERAL
    if storage_type == "ssd":
        return StorageType.SSD
    if storage_type == "network":
        return StorageType.NETWORK

    # Infer from remote_path
    remote_path = host_config.get("remote_path", "")
    if "/dev/shm" in remote_path or "/run/shm" in remote_path:
        return StorageType.EPHEMERAL

    return StorageType.PERSISTENT


def get_ephemeral_hosts(hosts_config: Dict[str, Any]) -> List[str]:
    """Get list of ephemeral host names from config."""
    ephemeral = []

    # Check vast_hosts (typically ephemeral)
    for name, config in hosts_config.get("vast_hosts", {}).items():
        if classify_host_storage(config) == StorageType.EPHEMERAL:
            ephemeral.append(name)

    # Check standard_hosts for any with RAM storage
    for name, config in hosts_config.get("standard_hosts", {}).items():
        if classify_host_storage(config) == StorageType.EPHEMERAL:
            ephemeral.append(name)

    return ephemeral


# =============================================================================
# Write-Ahead Log for Game Data
# =============================================================================


@dataclass
class WALEntry:
    """Write-ahead log entry for a game."""
    entry_id: int
    game_id: str
    source_host: str
    source_db: str
    game_data_hash: str
    created_at: float
    synced_at: Optional[float] = None
    sync_confirmed: bool = False


class WriteAheadLog:
    """Write-ahead log for game data to prevent data loss.

    Before syncing a game to the central database, we first write it to the WAL.
    This ensures that if the sync process crashes, we can recover the pending games.

    WAL lifecycle:
    1. Game discovered on remote host -> WAL entry created (pending)
    2. Game synced to local DB -> WAL entry marked synced
    3. Sync confirmed successful -> WAL entry removed (or marked confirmed)
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize WAL database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS wal_entries (
                entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL UNIQUE,
                source_host TEXT NOT NULL,
                source_db TEXT NOT NULL,
                game_data_hash TEXT NOT NULL,
                created_at REAL NOT NULL,
                synced_at REAL,
                sync_confirmed INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_wal_pending
            ON wal_entries(sync_confirmed, created_at);

            CREATE INDEX IF NOT EXISTS idx_wal_game_id
            ON wal_entries(game_id);

            CREATE TABLE IF NOT EXISTS wal_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
        """)
        conn.commit()
        conn.close()

    def append(
        self,
        game_id: str,
        source_host: str,
        source_db: str,
        game_data_hash: str,
    ) -> int:
        """Append entry to WAL. Returns entry_id."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO wal_entries
                (game_id, source_host, source_db, game_data_hash, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (game_id, source_host, source_db, game_data_hash, time.time()))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def append_batch(
        self,
        entries: List[Tuple[str, str, str, str]],  # (game_id, host, db, hash)
    ) -> int:
        """Append multiple entries to WAL. Returns count added."""
        if not entries:
            return 0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = time.time()
        try:
            cursor.executemany("""
                INSERT OR IGNORE INTO wal_entries
                (game_id, source_host, source_db, game_data_hash, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, [(g, h, d, hsh, now) for g, h, d, hsh in entries])
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def mark_synced(self, game_ids: List[str]) -> int:
        """Mark games as synced. Returns count updated."""
        if not game_ids:
            return 0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            placeholders = ",".join("?" * len(game_ids))
            cursor.execute(f"""
                UPDATE wal_entries
                SET synced_at = ?
                WHERE game_id IN ({placeholders}) AND synced_at IS NULL
            """, [time.time()] + list(game_ids))
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def confirm_synced(self, game_ids: List[str]) -> int:
        """Confirm sync complete (can be cleaned up). Returns count updated."""
        if not game_ids:
            return 0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            placeholders = ",".join("?" * len(game_ids))
            cursor.execute(f"""
                UPDATE wal_entries
                SET sync_confirmed = 1
                WHERE game_id IN ({placeholders})
            """, game_ids)
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def get_pending_entries(self, limit: int = 1000) -> List[WALEntry]:
        """Get entries that haven't been synced yet."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT entry_id, game_id, source_host, source_db,
                       game_data_hash, created_at, synced_at, sync_confirmed
                FROM wal_entries
                WHERE synced_at IS NULL
                ORDER BY created_at ASC
                LIMIT ?
            """, (limit,))
            return [
                WALEntry(
                    entry_id=row[0],
                    game_id=row[1],
                    source_host=row[2],
                    source_db=row[3],
                    game_data_hash=row[4],
                    created_at=row[5],
                    synced_at=row[6],
                    sync_confirmed=bool(row[7]),
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def get_unconfirmed_entries(self, limit: int = 1000) -> List[WALEntry]:
        """Get entries synced but not confirmed."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT entry_id, game_id, source_host, source_db,
                       game_data_hash, created_at, synced_at, sync_confirmed
                FROM wal_entries
                WHERE synced_at IS NOT NULL AND sync_confirmed = 0
                ORDER BY synced_at ASC
                LIMIT ?
            """, (limit,))
            return [
                WALEntry(
                    entry_id=row[0],
                    game_id=row[1],
                    source_host=row[2],
                    source_db=row[3],
                    game_data_hash=row[4],
                    created_at=row[5],
                    synced_at=row[6],
                    sync_confirmed=bool(row[7]),
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def cleanup_confirmed(self, older_than_seconds: int = 3600) -> int:
        """Remove confirmed entries older than threshold."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cutoff = time.time() - older_than_seconds
            cursor.execute("""
                DELETE FROM wal_entries
                WHERE sync_confirmed = 1 AND synced_at < ?
            """, (cutoff,))
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get WAL statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN synced_at IS NULL THEN 1 ELSE 0 END) as pending,
                    SUM(CASE WHEN synced_at IS NOT NULL AND sync_confirmed = 0 THEN 1 ELSE 0 END) as unconfirmed,
                    SUM(CASE WHEN sync_confirmed = 1 THEN 1 ELSE 0 END) as confirmed
                FROM wal_entries
            """)
            row = cursor.fetchone()
            return {
                "total": row[0] or 0,
                "pending": row[1] or 0,
                "unconfirmed": row[2] or 0,
                "confirmed": row[3] or 0,
            }
        finally:
            conn.close()


# =============================================================================
# Elo Database Replication
# =============================================================================


class EloReplicator:
    """Replicates the unified Elo database to cluster hosts.

    The Elo database is critical for training decisions. This replicator
    ensures it's available on multiple hosts for fault tolerance.
    """

    def __init__(
        self,
        local_elo_path: Path,
        replica_hosts: List[Dict[str, Any]],
        min_replicas: int = 2,
        replication_interval_seconds: int = 60,  # More aggressive than manifest (was 300)
        ssh_timeout: int = 30,
    ):
        self.local_path = local_elo_path
        self.replica_hosts = replica_hosts
        self.min_replicas = min_replicas
        self.replication_interval = replication_interval_seconds
        self.ssh_timeout = ssh_timeout

        self._last_replication_time = 0.0
        self._last_checksum = ""
        self._replication_lock = asyncio.Lock()

    def _compute_checksum(self, path: Path) -> str:
        """Compute SHA256 checksum of Elo database."""
        if not path.exists():
            return ""
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    async def should_replicate(self) -> bool:
        """Check if replication is needed."""
        if not self.local_path.exists():
            return False

        # Check time interval
        if time.time() - self._last_replication_time < self.replication_interval:
            return False

        # Check if content changed
        current_checksum = self._compute_checksum(self.local_path)
        if current_checksum == self._last_checksum:
            return False

        return True

    async def _replicate_to_host(self, host: Dict[str, Any]) -> bool:
        """Replicate Elo DB to a single host. Returns True on success."""
        ssh_host = host.get("ssh_host", "")
        ssh_user = host.get("ssh_user", "ubuntu")
        ssh_port = host.get("ssh_port", 22)
        remote_path = host.get("remote_elo_path", "~/ringrift/ai-service/data/unified_elo.db")

        if not ssh_host:
            return False

        ssh_args = f"-o ConnectTimeout={self.ssh_timeout} -o StrictHostKeyChecking=no -o BatchMode=yes"
        if ssh_port != 22:
            ssh_args += f" -p {ssh_port}"

        # Create remote directory
        mkdir_cmd = f'ssh {ssh_args} {ssh_user}@{ssh_host} "mkdir -p $(dirname {remote_path})"'
        try:
            process = await asyncio.create_subprocess_shell(
                mkdir_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(process.wait(), timeout=self.ssh_timeout)
        except Exception as e:
            logger.warning(f"Failed to create remote dir on {ssh_host}: {e}")
            return False

        # Use rsync with compression for efficient transfer
        rsync_args = [
            "rsync", "-az", "--compress",
            "-e", f"ssh {ssh_args}",
            str(self.local_path),
            f"{ssh_user}@{ssh_host}:{remote_path}",
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *rsync_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(process.wait(), timeout=120)
            return process.returncode == 0
        except Exception as e:
            logger.warning(f"Elo replication to {ssh_host} failed: {e}")
            return False

    async def replicate(self) -> int:
        """Replicate Elo DB to all replica hosts. Returns successful count."""
        async with self._replication_lock:
            if not await self.should_replicate():
                return 0

            tasks = [
                self._replicate_to_host(host)
                for host in self.replica_hosts
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)
            successful = sum(1 for r in results if r is True)

            if successful >= self.min_replicas:
                self._last_replication_time = time.time()
                self._last_checksum = self._compute_checksum(self.local_path)
                logger.info(f"Elo DB replicated to {successful}/{len(self.replica_hosts)} hosts")

            return successful

    async def recover_from_replica(self) -> bool:
        """Attempt to recover Elo DB from a replica if local is missing/stale."""
        if self.local_path.exists():
            # Check if local is recent (within last hour)
            mtime = self.local_path.stat().st_mtime
            if time.time() - mtime < 3600:
                return False  # Local is recent enough

        # Try each replica
        for host in self.replica_hosts:
            ssh_host = host.get("ssh_host", "")
            ssh_user = host.get("ssh_user", "ubuntu")
            ssh_port = host.get("ssh_port", 22)
            remote_path = host.get("remote_elo_path", "~/ringrift/ai-service/data/unified_elo.db")

            if not ssh_host:
                continue

            ssh_args = f"-o ConnectTimeout={self.ssh_timeout} -o StrictHostKeyChecking=no -o BatchMode=yes"
            if ssh_port != 22:
                ssh_args += f" -p {ssh_port}"

            # Check if remote exists
            check_cmd = f'ssh {ssh_args} {ssh_user}@{ssh_host} "test -f {remote_path} && echo EXISTS"'
            try:
                process = await asyncio.create_subprocess_shell(
                    check_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(process.communicate(), timeout=self.ssh_timeout)
                if b"EXISTS" not in stdout:
                    continue
            except Exception:
                continue

            # Download from replica
            self.local_path.parent.mkdir(parents=True, exist_ok=True)
            rsync_args = [
                "rsync", "-az",
                "-e", f"ssh {ssh_args}",
                f"{ssh_user}@{ssh_host}:{remote_path}",
                str(self.local_path),
            ]

            try:
                process = await asyncio.create_subprocess_exec(
                    *rsync_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(process.wait(), timeout=120)
                if process.returncode == 0:
                    logger.info(f"Recovered Elo DB from {ssh_host}")
                    return True
            except Exception as e:
                logger.warning(f"Failed to recover Elo DB from {ssh_host}: {e}")

        return False


# =============================================================================
# Robust Data Sync Coordinator
# =============================================================================


@dataclass
class SyncConfig:
    """Configuration for robust data sync."""
    # Polling intervals
    default_poll_interval: int = 60
    ephemeral_poll_interval: int = 15  # Aggressive for RAM disk hosts
    # Parallelism
    max_concurrent_syncs: int = 8
    max_concurrent_per_host: int = 2
    # Timeouts
    ssh_timeout: int = 30
    rsync_timeout: int = 300
    # WAL settings
    wal_enabled: bool = True
    wal_cleanup_interval: int = 3600
    # Elo replication
    elo_replication_enabled: bool = True
    elo_replication_interval: int = 60
    # Compression
    compress_ephemeral_transfers: bool = True


class RobustDataSync:
    """Coordinator for robust, efficient data synchronization.

    Features:
    1. Ephemeral host priority - Aggressive sync for RAM disk hosts
    2. Write-ahead log - Prevents data loss on crash
    3. Elo replication - Distributes Elo DB across cluster
    4. Parallel batch sync - Concurrent multi-host operations
    """

    def __init__(
        self,
        hosts: List[Dict[str, Any]],
        config: SyncConfig,
        data_dir: Path,
    ):
        self.hosts = hosts
        self.config = config
        self.data_dir = data_dir

        # Classify hosts by storage type
        self.ephemeral_hosts: List[str] = []
        self.persistent_hosts: List[str] = []
        self.host_profiles: Dict[str, HostSyncProfile] = {}

        for host in hosts:
            name = host.get("name", "")
            storage_type = classify_host_storage(host)

            if storage_type == StorageType.EPHEMERAL:
                self.ephemeral_hosts.append(name)
                self.host_profiles[name] = HostSyncProfile.for_ephemeral_host(name)
            else:
                self.persistent_hosts.append(name)
                self.host_profiles[name] = HostSyncProfile.for_persistent_host(name)

        # Initialize WAL
        self.wal: Optional[WriteAheadLog] = None
        if config.wal_enabled:
            self.wal = WriteAheadLog(data_dir / "sync_wal.db")

        # Initialize Elo replicator
        self.elo_replicator: Optional[EloReplicator] = None
        if config.elo_replication_enabled:
            # Use persistent hosts as Elo replicas
            replica_hosts = [h for h in hosts if h.get("name") in self.persistent_hosts]
            if replica_hosts:
                self.elo_replicator = EloReplicator(
                    local_elo_path=data_dir / "unified_elo.db",
                    replica_hosts=replica_hosts[:5],  # Max 5 replicas
                    replication_interval_seconds=config.elo_replication_interval,
                )

        # Sync state
        self._running = False
        self._sync_semaphore = asyncio.Semaphore(config.max_concurrent_syncs)
        self._last_ephemeral_sync = 0.0
        self._last_persistent_sync = 0.0

    def get_hosts_due_for_sync(self) -> Tuple[List[str], List[str]]:
        """Get lists of (ephemeral_hosts, persistent_hosts) due for sync."""
        now = time.time()
        ephemeral_due = []
        persistent_due = []

        for name in self.ephemeral_hosts:
            profile = self.host_profiles[name]
            if now - profile.last_sync_time >= self.config.ephemeral_poll_interval:
                ephemeral_due.append(name)

        for name in self.persistent_hosts:
            profile = self.host_profiles[name]
            if now - profile.last_sync_time >= self.config.default_poll_interval:
                persistent_due.append(name)

        return ephemeral_due, persistent_due

    async def sync_host(self, host_name: str, host_config: Dict[str, Any]) -> int:
        """Sync data from a single host. Returns games synced."""
        async with self._sync_semaphore:
            profile = self.host_profiles.get(host_name)
            if not profile:
                return 0

            # Placeholder for actual sync logic
            # This would call the streaming_data_collector's sync methods
            profile.last_sync_time = time.time()
            return 0

    async def run_sync_cycle(self) -> Dict[str, int]:
        """Run one sync cycle. Returns {host: games_synced}."""
        results = {}
        ephemeral_due, persistent_due = self.get_hosts_due_for_sync()

        # Sync ephemeral hosts first (priority)
        if ephemeral_due:
            logger.info(f"Syncing {len(ephemeral_due)} ephemeral hosts (priority)")
            tasks = []
            for name in ephemeral_due:
                host_config = next((h for h in self.hosts if h.get("name") == name), {})
                tasks.append(self.sync_host(name, host_config))

            sync_results = await asyncio.gather(*tasks, return_exceptions=True)
            for name, result in zip(ephemeral_due, sync_results):
                if isinstance(result, int):
                    results[name] = result

        # Then sync persistent hosts
        if persistent_due:
            tasks = []
            for name in persistent_due:
                host_config = next((h for h in self.hosts if h.get("name") == name), {})
                tasks.append(self.sync_host(name, host_config))

            sync_results = await asyncio.gather(*tasks, return_exceptions=True)
            for name, result in zip(persistent_due, sync_results):
                if isinstance(result, int):
                    results[name] = result

        # Replicate Elo DB after sync
        if self.elo_replicator:
            try:
                await self.elo_replicator.replicate()
            except Exception as e:
                logger.warning(f"Elo replication failed: {e}")

        # Cleanup old WAL entries
        if self.wal:
            self.wal.cleanup_confirmed()

        return results

    def get_status(self) -> Dict[str, Any]:
        """Get sync status summary."""
        wal_stats = self.wal.get_stats() if self.wal else {}

        return {
            "ephemeral_hosts": len(self.ephemeral_hosts),
            "persistent_hosts": len(self.persistent_hosts),
            "ephemeral_poll_interval": self.config.ephemeral_poll_interval,
            "persistent_poll_interval": self.config.default_poll_interval,
            "wal_stats": wal_stats,
            "elo_replication_enabled": self.elo_replicator is not None,
            "host_profiles": {
                name: {
                    "storage_type": profile.storage_type.value,
                    "is_ephemeral": profile.is_ephemeral,
                    "last_sync": profile.last_sync_time,
                }
                for name, profile in self.host_profiles.items()
            },
        }


# =============================================================================
# Module-level utilities
# =============================================================================


def create_wal(data_dir: Path) -> WriteAheadLog:
    """Create a write-ahead log instance."""
    return WriteAheadLog(data_dir / "sync_wal.db")


def create_elo_replicator(
    data_dir: Path,
    hosts_config: Dict[str, Any],
    min_replicas: int = 2,
) -> Optional[EloReplicator]:
    """Create Elo replicator from hosts config."""
    # Select persistent hosts for replication
    replica_hosts = []

    for name, config in hosts_config.get("standard_hosts", {}).items():
        if classify_host_storage(config) == StorageType.PERSISTENT:
            replica_hosts.append({
                "name": name,
                "ssh_host": config.get("ssh_host", ""),
                "ssh_user": config.get("ssh_user", "ubuntu"),
                "ssh_port": config.get("ssh_port", 22),
                "remote_elo_path": "~/ringrift/ai-service/data/unified_elo.db",
            })

    if len(replica_hosts) < min_replicas:
        logger.warning(f"Only {len(replica_hosts)} replica hosts available, need {min_replicas}")
        return None

    return EloReplicator(
        local_elo_path=data_dir / "unified_elo.db",
        replica_hosts=replica_hosts[:5],  # Max 5
        min_replicas=min_replicas,
    )
