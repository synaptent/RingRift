"""Database Sync Manager - Unified base for SQLite database synchronization.

This module consolidates common patterns from EloSyncManager and RegistrySyncManager:
- Multi-transport failover (Tailscale → SSH → Vast.ai SSH → HTTP)
- Rsync-based database transfers
- Merge-based conflict resolution
- Node discovery from P2P/YAML config

December 2025: Consolidation of sync managers for code reuse.
Expected savings: ~470 LOC (from 1,580 → ~1,110 LOC)

Usage:
    from app.coordination.database_sync_manager import DatabaseSyncManager

    class MySyncManager(DatabaseSyncManager):
        async def _merge_databases(self, remote_db_path: Path) -> bool:
            # Implement type-specific merge logic
            pass

        def _update_local_stats(self) -> None:
            # Update local record count, hash, etc.
            pass

See also:
- app/tournament/elo_sync_manager.py - EloSyncManager subclass
- app/training/registry_sync_manager.py - RegistrySyncManager subclass
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import socket
import sqlite3
import subprocess
import tempfile
import time
import urllib.request
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.coordination.sync_base import (
    BaseSyncProgress,
    CircuitBreakerConfig,
    SyncManagerBase,
    try_transports,
)
from app.coordination.wal_sync_utils import (
    checkpoint_database,
    validate_synced_database,
)
from app.utils.retry import RetryConfig
from app.config.coordination_defaults import build_ssh_options, build_ssh_options_list

logger = logging.getLogger(__name__)


def _get_ssh_user_for_host(host: str) -> str:
    """Look up SSH user for a target host from cluster config.

    Args:
        host: Target hostname or IP address

    Returns:
        SSH user from cluster config, or 'ubuntu' as default (most common).
        Only returns 'root' if explicitly configured.
    """
    try:
        from app.config.cluster_config import get_cluster_nodes

        # get_cluster_nodes() returns dict[str, ClusterNode], iterate values
        for node in get_cluster_nodes().values():
            # Match by Tailscale IP, SSH host, or name
            if host in (node.tailscale_ip, node.ssh_host, node.name):
                return node.ssh_user

        # Not found in cluster config - default to 'ubuntu' (most common)
        # Lambda, Nebius use ubuntu; Hetzner/Vast/RunPod/Vultr use root
        return "ubuntu"
    except Exception as e:
        # Fallback if cluster config unavailable
        logger.debug(f"Failed to lookup SSH user for {host}: {e}")
        return "ubuntu"


# Retry configuration for network operations
# December 30, 2025: Migrated to centralized RetryConfig for consistency
# HTTP sync retry configuration
HTTP_RETRY_CONFIG = RetryConfig(
    max_attempts=4,      # Was HTTP_MAX_RETRIES + 1 = 4
    base_delay=1.0,      # Was HTTP_INITIAL_BACKOFF
    max_delay=30.0,      # Was HTTP_MAX_BACKOFF
    exponential=True,    # Was HTTP_BACKOFF_MULTIPLIER = 2.0
    jitter=0.1,          # Add 10% jitter for distributed systems
)

# Push sync retry configuration (longer delays for rsync operations)
PUSH_RETRY_CONFIG = RetryConfig(
    max_attempts=3,      # Default max_retries in _push_with_retry
    base_delay=10.0,     # Was 10 * (2 ** attempt)
    max_delay=60.0,      # Was min(..., 60)
    exponential=True,    # Exponential backoff
    jitter=0.1,          # Was random.uniform(0, 5) / 10 ≈ 8%
)


# =============================================================================
# Atomic File Operations
# =============================================================================


def atomic_copy(src: Path, dst: Path) -> None:
    """Atomically copy a file using temp file + rename pattern.

    On POSIX systems, rename within the same filesystem is atomic.
    This prevents corrupted destination files if the copy is interrupted.

    Args:
        src: Source file path
        dst: Destination file path

    Raises:
        OSError: If the copy or rename fails
    """
    # Create temp file in the same directory as destination for atomic rename
    dst_dir = dst.parent
    dst_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        dir=dst_dir,
        prefix=".tmp_atomic_",
        suffix=dst.suffix,
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Copy to temp file
        shutil.copy2(src, tmp_path)
        # Atomic rename (POSIX guarantees atomicity for same-filesystem renames)
        tmp_path.rename(dst)
    except (OSError, shutil.Error):
        # Clean up temp file on failure (file operations can raise OSError or shutil.Error)
        tmp_path.unlink(missing_ok=True)
        raise


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class DatabaseSyncState(BaseSyncProgress):
    """Extended sync state for database synchronization.

    Adds database-specific fields to BaseSyncProgress:
    - local_record_count: Number of records in local database
    - local_hash: Hash of database content for change detection
    - merge_conflicts: Count of resolved merge conflicts
    - successful_syncs: Count of successful syncs
    """

    local_record_count: int = 0
    local_hash: str = ""
    synced_from: str = ""
    merge_conflicts: int = 0
    total_syncs: int = 0
    successful_syncs: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dictionary."""
        base = super().to_dict()
        base.update({
            "local_record_count": self.local_record_count,
            "local_hash": self.local_hash,
            "synced_from": self.synced_from,
            "merge_conflicts": self.merge_conflicts,
            "total_syncs": self.total_syncs,
            "successful_syncs": self.successful_syncs,
        })
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatabaseSyncState:
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
        )


@dataclass
class SyncNodeInfo:
    """Unified node info for database sync managers.

    Combines fields from EloSyncManager.NodeInfo and RegistrySyncManager.NodeInfo.
    """

    name: str
    tailscale_ip: str | None = None
    ssh_host: str | None = None
    ssh_port: int = 22
    http_url: str | None = None
    vast_ssh_host: str | None = None
    vast_ssh_port: int | None = None
    is_coordinator: bool = False
    last_seen: float = 0.0
    reachable: bool = True

    # Database-specific
    remote_db_path: str = ""
    record_count: int = 0


# =============================================================================
# Database Sync Manager Base Class
# =============================================================================


class DatabaseSyncManager(SyncManagerBase):
    """Base class for SQLite database synchronization across cluster nodes.

    Provides common functionality for syncing SQLite databases:
    - Multi-transport failover (Tailscale → SSH → HTTP)
    - Rsync-based database transfers
    - Merge-based conflict resolution (abstract, subclass-specific)
    - Node discovery from P2P or YAML config

    Subclasses must implement:
    - _merge_databases(remote_db_path): Type-specific merge logic
    - _update_local_stats(): Update local record count, hash, etc.
    - _get_remote_db_path(): Return remote database path
    - _get_remote_count_query(): Return SQL query for counting records
    """

    def __init__(
        self,
        db_path: Path,
        state_path: Path,
        db_type: str,
        coordinator_host: str = "nebius-backbone-1",
        sync_interval: float = 300.0,  # See TIMEOUTS.SYNC_INTERVAL
        p2p_url: str | None = None,
        enable_merge: bool = True,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
    ):
        """Initialize database sync manager.

        Args:
            db_path: Path to local database file
            state_path: Path to persist sync state JSON
            db_type: Type identifier (e.g., "elo", "registry")
            coordinator_host: Primary coordinator hostname
            sync_interval: Seconds between sync cycles
            p2p_url: P2P orchestrator URL for node discovery
            enable_merge: Whether to merge on pull (vs replace)
            circuit_breaker_config: Config for per-node circuit breakers
        """
        super().__init__(
            state_path=state_path,
            sync_interval=sync_interval,
            circuit_breaker_config=circuit_breaker_config,
        )

        self.db_path = Path(db_path)
        self.db_type = db_type
        self.coordinator_host = coordinator_host
        # Dec 2025: Use centralized P2P URL helper from app.config.ports
        if p2p_url:
            self.p2p_url = p2p_url
        else:
            from app.config.ports import get_local_p2p_url
            self.p2p_url = get_local_p2p_url()
        self.enable_merge = enable_merge

        # Override state with database-specific version
        self._db_state = DatabaseSyncState()
        if self.state_path and self.state_path.exists():
            self._load_db_state()

        # Node tracking
        self.nodes: dict[str, SyncNodeInfo] = {}

        # Callbacks
        self._on_sync_complete_callbacks: list[Callable] = []
        self._on_sync_failed_callbacks: list[Callable] = []

        logger.info(
            f"[{self.__class__.__name__}] Initialized for {db_type} "
            f"db={db_path}, coordinator={coordinator_host}"
        )

    def _load_db_state(self) -> None:
        """Load database-specific state from persistent storage."""
        try:
            if self.state_path and self.state_path.exists():
                with open(self.state_path) as f:
                    data = json.load(f)
                    self._db_state = DatabaseSyncState.from_dict(data)
                    logger.debug(f"Loaded {self.db_type} sync state from {self.state_path}")
        except (OSError, json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning(f"Failed to load {self.db_type} sync state: {e}")

    def _save_db_state(self) -> None:
        """Save database-specific state to persistent storage."""
        try:
            if self.state_path:
                self.state_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.state_path, "w") as f:
                    json.dump(self._db_state.to_dict(), f, indent=2)
        except (OSError, TypeError) as e:
            # OSError for file operations, TypeError if to_dict returns non-serializable data
            logger.warning(f"Failed to save {self.db_type} sync state: {e}")

    # =========================================================================
    # Abstract methods (subclass-specific)
    # =========================================================================

    @abstractmethod
    async def _merge_databases(self, remote_db_path: Path) -> bool:
        """Merge remote database into local database.

        Args:
            remote_db_path: Path to downloaded remote database

        Returns:
            True if merge succeeded, False otherwise
        """

    @abstractmethod
    def _update_local_stats(self) -> None:
        """Update local database statistics (record count, hash, etc.)."""

    @abstractmethod
    def _get_remote_db_path(self) -> str:
        """Get the remote database path for rsync.

        Returns:
            Remote path string (e.g., "ai-service/data/unified_elo.db")
        """

    @abstractmethod
    def _get_remote_count_query(self) -> str:
        """Get SQL query for counting remote records.

        Returns:
            SQL query string (e.g., "SELECT COUNT(*) FROM match_history")
        """

    # =========================================================================
    # SyncManagerBase implementation
    # =========================================================================

    async def _do_sync(self, node: str) -> bool:
        """Perform sync with a specific node using transport failover."""
        node_info = self.nodes.get(node)
        if not node_info:
            logger.warning(f"[{self.db_type}] No node info for {node}")
            return False

        # Build transport list based on node capabilities
        transports: list[tuple[str, Callable]] = []

        if node_info.tailscale_ip:
            transports.append(("tailscale", lambda _n, ni=node_info: self._sync_via_tailscale(ni)))

        if node_info.vast_ssh_host:
            transports.append(("vast_ssh", lambda _n, ni=node_info: self._sync_via_vast_ssh(ni)))

        if node_info.ssh_host:
            transports.append(("ssh", lambda _n, ni=node_info: self._sync_via_ssh(ni)))

        if node_info.http_url:
            transports.append(("http", lambda _n, ni=node_info: self._sync_via_http(ni)))

        if not transports:
            logger.warning(f"[{self.db_type}] No transports available for {node}")
            return False

        success, transport_used = await try_transports(node, transports)

        if success:
            self._db_state.synced_from = f"{node}:{transport_used}"
            self._db_state.successful_syncs += 1
            await self._notify_sync_complete(node, transport_used)
        else:
            await self._notify_sync_failed(node, "all_transports_failed")

        return success

    def _get_nodes(self) -> list[str]:
        """Get list of nodes to sync with."""
        return list(self.nodes.keys())

    # =========================================================================
    # Transport methods (reusable across subclasses)
    # =========================================================================

    async def _sync_via_tailscale(self, node: SyncNodeInfo) -> bool:
        """Sync via Tailscale direct connection."""
        if not node.tailscale_ip:
            return False

        remote_path = node.remote_db_path or self._get_remote_db_path()
        return await self._rsync_pull(
            host=node.tailscale_ip,
            remote_path=remote_path,
            ssh_port=22,
        )

    async def _sync_via_ssh(self, node: SyncNodeInfo) -> bool:
        """Sync via SSH public endpoint."""
        if not node.ssh_host:
            return False

        remote_path = node.remote_db_path or self._get_remote_db_path()
        return await self._rsync_pull(
            host=node.ssh_host,
            remote_path=remote_path,
            ssh_port=node.ssh_port,
        )

    async def _sync_via_vast_ssh(self, node: SyncNodeInfo) -> bool:
        """Sync via Vast.ai SSH endpoint (different path structure)."""
        if not node.vast_ssh_host or not node.vast_ssh_port:
            return False

        # Vast.ai uses different workspace path
        remote_path = node.remote_db_path or f"/workspace/ringrift/{self._get_remote_db_path()}"
        return await self._rsync_pull(
            host=node.vast_ssh_host,
            remote_path=remote_path,
            ssh_port=node.vast_ssh_port,
        )

    async def _sync_via_http(self, node: SyncNodeInfo) -> bool:
        """Sync via HTTP download with retry logic.

        December 30, 2025: Migrated to use centralized RetryConfig.
        Uses exponential backoff for transient network failures.
        """
        if not node.http_url:
            return False

        try:
            import aiohttp
        except ImportError:
            logger.debug(f"[{self.db_type}] aiohttp not available for HTTP sync")
            return False

        url = f"{node.http_url.rstrip('/')}/data/{self.db_type}.db"
        last_error: Exception | None = None

        for attempt in HTTP_RETRY_CONFIG.attempts():
            tmp_path: Path | None = None
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                        if resp.status != 200:
                            # Non-retryable status codes
                            if resp.status in (404, 403, 401):
                                logger.debug(f"[{self.db_type}] HTTP {resp.status} for {url}")
                                return False
                            # Retryable server errors (5xx)
                            if resp.status >= 500 and attempt.should_retry:
                                delay = attempt.delay
                                logger.warning(
                                    f"[{self.db_type}] HTTP {resp.status} from {node.name}, "
                                    f"retry {attempt.number}/{HTTP_RETRY_CONFIG.max_attempts} in {delay:.1f}s"
                                )
                                await attempt.wait_async()
                                continue
                            return False

                        # Download to temp file (use asyncio.to_thread to avoid blocking)
                        # Dec 29, 2025: Large DB files could stall event loop with sync I/O
                        response_data = await resp.read()

                        def _write_temp_file():
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
                                tmp.write(response_data)
                                return Path(tmp.name)

                        tmp_path = await asyncio.to_thread(_write_temp_file)

                        # Merge or replace
                        if self.enable_merge:
                            result = await self._merge_databases(tmp_path)
                        else:
                            atomic_copy(tmp_path, self.db_path)
                            result = True

                        tmp_path.unlink()
                        return result

            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                last_error = e
                if tmp_path and tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
                if attempt.should_retry:
                    delay = attempt.delay
                    logger.warning(
                        f"[{self.db_type}] HTTP sync to {node.name} failed: {e}. "
                        f"Retry {attempt.number}/{HTTP_RETRY_CONFIG.max_attempts} in {delay:.1f}s"
                    )
                    await attempt.wait_async()

        logger.warning(
            f"[{self.db_type}] HTTP sync failed after {HTTP_RETRY_CONFIG.max_attempts} attempts: {last_error}"
        )
        return False

    async def _rsync_pull(
        self,
        host: str,
        remote_path: str,
        ssh_port: int = 22,
        timeout: float = 60.0,
    ) -> bool:
        """Pull database from remote host via rsync.

        Args:
            host: Remote hostname or IP
            remote_path: Path to database on remote host
            ssh_port: SSH port number
            timeout: Operation timeout in seconds

        Returns:
            True if pull and merge succeeded
        """
        try:
            # Create temp directory for download (to hold DB and WAL files)
            tmp_dir = Path(tempfile.mkdtemp(prefix="db_sync_"))
            tmp_path = tmp_dir / Path(remote_path).name

            # Build rsync command (Dec 2025: use centralized timeout)
            from app.config.thresholds import RSYNC_TIMEOUT
            from app.utils.env_config import get_str
            # Dec 30, 2025: Look up SSH user from cluster config instead of global default
            ssh_user = _get_ssh_user_for_host(host)
            ssh_key = get_str("RINGRIFT_SSH_KEY", "") or "~/.ssh/id_cluster"
            # Dec 30, 2025: Use centralized SSH config for consistent timeouts
            ssh_cmd = build_ssh_options(
                key_path=ssh_key,
                port=ssh_port,
                include_keepalive=True,  # Rsync transfers may be long
            )

            # Dec 2025: Include WAL files (.db-wal, .db-shm) to prevent data loss
            # Remote path might be /path/to/file.db, we need to sync from directory
            remote_dir = str(Path(remote_path).parent) + "/"
            db_name = Path(remote_path).name

            rsync_cmd = [
                "rsync",
                "-avz",
                f"--timeout={RSYNC_TIMEOUT}",
                f"--include={db_name}",
                f"--include={db_name}-wal",
                f"--include={db_name}-shm",
                "--exclude=*",
                "-e", ssh_cmd,
                f"{ssh_user}@{host}:{remote_dir}",
                str(tmp_dir) + "/",
            ]

            # Run rsync
            proc = await asyncio.create_subprocess_exec(
                *rsync_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                logger.warning(f"[{self.db_type}] Rsync to {host} timed out")
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return False

            if proc.returncode != 0:
                logger.warning(
                    f"[{self.db_type}] Rsync from {host} failed: {stderr.decode()[:200]}"
                )
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return False

            # Dec 2025: Validate synced database integrity
            is_valid, errors = validate_synced_database(tmp_path, check_integrity=True)
            if not is_valid:
                logger.warning(
                    f"[{self.db_type}] Synced database failed validation: {errors}"
                )
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return False

            # Merge or replace
            if self.enable_merge:
                result = await self._merge_databases(tmp_path)
            else:
                atomic_copy(tmp_path, self.db_path)
                result = True

            # Clean up temp directory (includes WAL files)
            shutil.rmtree(tmp_dir, ignore_errors=True)

            if result:
                self._update_local_stats()
                self._db_state.last_sync_timestamp = time.time()
                self._save_db_state()
                logger.info(f"[{self.db_type}] Synced from {host}")

            return result

        except (OSError, asyncio.TimeoutError, sqlite3.Error, shutil.Error) as e:
            # OSError: subprocess/file errors, TimeoutError: async timeout
            # sqlite3.Error: database validation, shutil.Error: cleanup operations
            logger.error(f"[{self.db_type}] Rsync pull error: {e}")
            # Clean up on exception
            if 'tmp_dir' in locals():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            return False

    async def _rsync_push(
        self,
        host: str,
        remote_path: str,
        ssh_port: int = 22,
        timeout: float = 60.0,
    ) -> bool:
        """Push local database to remote host via rsync.

        Args:
            host: Remote hostname or IP
            remote_path: Path to database on remote host
            ssh_port: SSH port number
            timeout: Operation timeout in seconds

        Returns:
            True if push succeeded
        """
        try:
            if not self.db_path.exists():
                logger.warning(f"[{self.db_type}] Local database not found: {self.db_path}")
                return False

            # Dec 2025: Checkpoint WAL before push to ensure all data is in main .db file
            checkpoint_database(str(self.db_path))

            # Build rsync command (Dec 2025: use centralized timeout)
            from app.config.thresholds import RSYNC_TIMEOUT
            from app.utils.env_config import get_str
            # Dec 30, 2025: Look up SSH user from cluster config instead of global default
            ssh_user = _get_ssh_user_for_host(host)
            ssh_key = get_str("RINGRIFT_SSH_KEY", "") or "~/.ssh/id_cluster"
            # Dec 30, 2025: Use centralized SSH config for consistent timeouts
            ssh_cmd = build_ssh_options(
                key_path=ssh_key,
                port=ssh_port,
                include_keepalive=True,  # Rsync pushes may be long
            )

            # Dec 2025: Include WAL files (.db-wal, .db-shm) to prevent data loss
            db_name = self.db_path.name
            parent_dir = str(self.db_path.parent) + "/"
            remote_dir = str(Path(remote_path).parent) + "/"

            # Dec 29, 2025: Added --checksum for data integrity verification during transfer
            # Added --partial for resume on network glitches
            rsync_cmd = [
                "rsync",
                "-avz",
                "--checksum",  # Verify file integrity using checksums
                "--partial",   # Keep partial files for resume
                f"--timeout={RSYNC_TIMEOUT}",
                f"--include={db_name}",
                f"--include={db_name}-wal",
                f"--include={db_name}-shm",
                "--exclude=*",
                "-e", ssh_cmd,
                parent_dir,
                f"{ssh_user}@{host}:{remote_dir}",
            ]

            # Run rsync
            proc = await asyncio.create_subprocess_exec(
                *rsync_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                logger.warning(f"[{self.db_type}] Rsync push to {host} timed out")
                return False

            if proc.returncode != 0:
                logger.warning(
                    f"[{self.db_type}] Rsync push to {host} failed: {stderr.decode()[:200]}"
                )
                return False

            logger.info(f"[{self.db_type}] Pushed to {host}")
            return True

        except (OSError, asyncio.TimeoutError, sqlite3.Error) as e:
            # OSError: subprocess/file errors, TimeoutError: async timeout
            # sqlite3.Error: checkpoint operation errors
            logger.error(f"[{self.db_type}] Rsync push error: {e}")
            return False

    async def _compute_remote_checksum(
        self,
        host: str,
        remote_path: str,
        ssh_port: int = 22,
        timeout: float = 30.0,
    ) -> str | None:
        """Compute SHA256 checksum of remote file via SSH.

        Dec 29, 2025: Added for post-transfer verification (Phase 1.1).

        Args:
            host: Remote hostname or IP
            remote_path: Path to file on remote host
            ssh_port: SSH port number
            timeout: SSH command timeout

        Returns:
            SHA256 checksum string, or None on failure
        """
        try:
            from app.utils.env_config import get_str
            ssh_user = _get_ssh_user_for_host(host)
            ssh_key = get_str("RINGRIFT_SSH_KEY", "") or "~/.ssh/id_cluster"

            # Dec 30, 2025: Use centralized SSH config for consistent timeouts
            ssh_args = build_ssh_options_list(
                key_path=ssh_key,
                port=ssh_port,
                include_keepalive=False,  # Quick checksum command, no keepalive needed
            )
            ssh_args.append(f"{ssh_user}@{host}")

            # Use sha256sum on Linux, shasum on macOS (detected remotely)
            checksum_cmd = (
                f"if command -v sha256sum >/dev/null 2>&1; then "
                f"sha256sum '{remote_path}' | cut -d' ' -f1; "
                f"elif command -v shasum >/dev/null 2>&1; then "
                f"shasum -a 256 '{remote_path}' | cut -d' ' -f1; "
                f"else echo 'NO_CHECKSUM_TOOL'; fi"
            )
            ssh_args.append(checksum_cmd)

            proc = await asyncio.create_subprocess_exec(
                *ssh_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

            if proc.returncode != 0:
                logger.warning(
                    f"[{self.db_type}] Remote checksum failed for {host}:{remote_path}: "
                    f"{stderr.decode()[:100]}"
                )
                return None

            checksum = stdout.decode().strip()
            if checksum == "NO_CHECKSUM_TOOL":
                logger.warning(f"[{self.db_type}] No checksum tool on {host}")
                return None

            return checksum

        except asyncio.TimeoutError:
            logger.warning(f"[{self.db_type}] Remote checksum timed out for {host}")
            return None
        except (OSError, subprocess.SubprocessError) as e:
            logger.warning(f"[{self.db_type}] Remote checksum error: {e}")
            return None

    async def _verify_push(
        self,
        host: str,
        remote_path: str,
        ssh_port: int = 22,
    ) -> bool:
        """Verify pushed file matches local file via checksum comparison.

        Dec 29, 2025: Added for post-transfer verification (Phase 1.1).

        Args:
            host: Remote hostname or IP
            remote_path: Path to database on remote host
            ssh_port: SSH port number

        Returns:
            True if checksums match
        """
        try:
            from app.coordination.sync_integrity import compute_file_checksum

            local_checksum = compute_file_checksum(self.db_path)
            remote_checksum = await self._compute_remote_checksum(host, remote_path, ssh_port)

            if remote_checksum is None:
                logger.warning(
                    f"[{self.db_type}] Cannot verify push to {host}: remote checksum unavailable"
                )
                # Return True to not block on hosts without checksum tools
                return True

            if local_checksum != remote_checksum:
                logger.error(
                    f"[{self.db_type}] Checksum mismatch for {host}:{remote_path}: "
                    f"local={local_checksum[:16]}... remote={remote_checksum[:16]}..."
                )
                return False

            logger.debug(
                f"[{self.db_type}] Verified push to {host}: checksum={local_checksum[:16]}..."
            )
            return True

        except FileNotFoundError:
            logger.error(f"[{self.db_type}] Local file not found for verification: {self.db_path}")
            return False
        except (OSError, ValueError) as e:
            logger.warning(f"[{self.db_type}] Verification error: {e}")
            # Return True to not block on verification errors
            return True

    async def _rsync_push_with_retry(
        self,
        host: str,
        remote_path: str,
        ssh_port: int = 22,
        timeout: float = 60.0,
        max_retries: int = 3,
        verify: bool = True,
    ) -> bool:
        """Push local database with exponential backoff retry.

        Dec 29, 2025: Added for reliable transfers (Phase 1.4).
        Dec 30, 2025: Migrated to centralized RetryConfig.

        Args:
            host: Remote hostname or IP
            remote_path: Path to database on remote host
            ssh_port: SSH port number
            timeout: Per-attempt timeout in seconds
            max_retries: Maximum retry attempts (uses PUSH_RETRY_CONFIG if default)
            verify: Whether to verify checksum after push

        Returns:
            True if push succeeded (and verified if verify=True)
        """
        # Use custom max_retries if provided, otherwise use default config
        retry_config = (
            PUSH_RETRY_CONFIG
            if max_retries == 3
            else RetryConfig(
                max_attempts=max_retries,
                base_delay=PUSH_RETRY_CONFIG.base_delay,
                max_delay=PUSH_RETRY_CONFIG.max_delay,
                exponential=True,
                jitter=PUSH_RETRY_CONFIG.jitter,
            )
        )

        for attempt in retry_config.attempts():
            try:
                success = await self._rsync_push(host, remote_path, ssh_port, timeout)

                if success:
                    # Verify the transfer if requested
                    if verify:
                        verified = await self._verify_push(host, remote_path, ssh_port)
                        if not verified:
                            logger.warning(
                                f"[{self.db_type}] Push to {host} succeeded but verification failed, "
                                f"attempt {attempt.number}/{retry_config.max_attempts}"
                            )
                            # Don't count verification failure as success
                            success = False

                if success:
                    if attempt.number > 1:
                        logger.info(
                            f"[{self.db_type}] Push to {host} succeeded on attempt {attempt.number}"
                        )
                    return True

            except (OSError, asyncio.TimeoutError) as e:
                logger.warning(
                    f"[{self.db_type}] Push to {host} failed "
                    f"(attempt {attempt.number}/{retry_config.max_attempts}): {e}"
                )

            # Wait before retry using centralized backoff calculation
            if attempt.should_retry:
                delay = attempt.delay
                logger.info(
                    f"[{self.db_type}] Retrying push to {host} in {delay:.1f}s"
                )
                await attempt.wait_async()

        logger.error(
            f"[{self.db_type}] Push to {host} failed after {retry_config.max_attempts} attempts"
        )
        return False

    # =========================================================================
    # Node Discovery
    # =========================================================================

    async def discover_nodes(self) -> None:
        """Discover cluster nodes from P2P status or YAML config."""
        try:
            # Try P2P discovery first
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.p2p_url}/status",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        peers = data.get("peers", {})
                        # Dec 30, 2025: peers is a dict {node_id: peer_info}
                        # Iterate over values (peer_info dicts), not keys (strings)
                        peer_list = peers.values() if isinstance(peers, dict) else peers
                        skipped_count = 0
                        for peer in peer_list:
                            if isinstance(peer, str):
                                # Skip if peer is just a string (old format)
                                logger.debug(
                                    f"[{self.db_type}] Skipping string-format peer: {peer}"
                                )
                                skipped_count += 1
                                continue
                            name = peer.get("node_id", peer.get("host", "unknown"))
                            self.nodes[name] = SyncNodeInfo(
                                name=name,
                                tailscale_ip=peer.get("tailscale_ip"),
                                ssh_host=peer.get("ssh_host"),
                                ssh_port=peer.get("ssh_port", 22),
                                http_url=peer.get("http_url"),
                                remote_db_path=self._get_remote_db_path(),
                                last_seen=time.time(),
                            )
                        logger.info(
                            f"[{self.db_type}] Discovered {len(self.nodes)} nodes from P2P"
                            + (f" (skipped {skipped_count} string-format)" if skipped_count else "")
                        )
                        return

        except ImportError:
            pass
        except (OSError, asyncio.TimeoutError, ValueError, KeyError) as e:
            # OSError: network errors, TimeoutError: request timeout
            # ValueError/KeyError: malformed P2P response
            logger.debug(f"[{self.db_type}] P2P discovery failed: {e}")

        # Fallback to YAML config
        await self._discover_nodes_from_yaml()

    async def _discover_nodes_from_yaml(self) -> None:
        """Discover nodes from distributed_hosts.yaml config.

        December 2025: Consolidated to use cluster_config.py helpers.
        """
        try:
            from app.config.cluster_config import get_ready_nodes

            ready_nodes = get_ready_nodes()
            for node in ready_nodes:
                # Build HTTP URL from data_server_base_url if available
                http_url = node.data_server_base_url

                self.nodes[node.name] = SyncNodeInfo(
                    name=node.name,
                    tailscale_ip=node.tailscale_ip,
                    ssh_host=node.ssh_host,
                    ssh_port=node.ssh_port,
                    http_url=http_url,
                    remote_db_path=self._get_remote_db_path(),
                    is_coordinator=node.is_coordinator,
                )

            logger.info(f"[{self.db_type}] Loaded {len(self.nodes)} nodes from cluster_config")

        except ImportError:
            logger.warning(f"[{self.db_type}] cluster_config not available for discovery")
        except (OSError, ValueError, KeyError, AttributeError, TypeError) as e:
            # OSError: file not found
            # ValueError/KeyError/AttributeError/TypeError: malformed config data
            logger.warning(f"[{self.db_type}] YAML discovery failed: {e}")

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on_sync_complete(self, callback: Callable) -> None:
        """Register callback for successful sync."""
        self._on_sync_complete_callbacks.append(callback)

    def on_sync_failed(self, callback: Callable) -> None:
        """Register callback for failed sync."""
        self._on_sync_failed_callbacks.append(callback)

    async def _notify_sync_complete(self, node: str, transport: str) -> None:
        """Notify callbacks of successful sync."""
        for callback in self._on_sync_complete_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(node, transport, self._db_state)
                else:
                    callback(node, transport, self._db_state)
            except (TypeError, AttributeError, RuntimeError) as e:
                # Narrow to callback-specific errors (December 2025 exception narrowing)
                logger.warning(f"[{self.db_type}] Sync complete callback error: {e}")

    async def _notify_sync_failed(self, node: str, reason: str) -> None:
        """Notify callbacks of failed sync."""
        for callback in self._on_sync_failed_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(node, reason, self._db_state)
                else:
                    callback(node, reason, self._db_state)
            except (TypeError, AttributeError, RuntimeError) as e:
                # Narrow to callback-specific errors (December 2025 exception narrowing)
                logger.warning(f"[{self.db_type}] Sync failed callback error: {e}")

    # =========================================================================
    # Utility methods
    # =========================================================================

    async def ensure_latest(self) -> bool:
        """Ensure local database is up-to-date before operations.

        Returns:
            True if database is current (either already synced or sync succeeded)
        """
        # Check if recent sync
        if time.time() - self._db_state.last_sync_timestamp < self.sync_interval:
            return True

        # Otherwise sync now
        results = await self.sync_with_cluster()
        return any(results.values())

    def get_status(self) -> dict[str, Any]:
        """Get current sync manager status."""
        base_status = super().get_status()
        base_status.update({
            "db_type": self.db_type,
            "db_path": str(self.db_path),
            "db_exists": self.db_path.exists(),
            "db_state": self._db_state.to_dict(),
            "node_count": len(self.nodes),
            "nodes": list(self.nodes.keys()),
        })
        return base_status

    def health_check(self) -> "HealthCheckResult":
        """Check sync manager health (CoordinatorProtocol compliance).

        Returns:
            HealthCheckResult with sync status and database details.

        December 2025: Added for unified health monitoring. Inherited by
        EloSyncManager and RegistrySyncManager. Added exception handling to
        prevent health_check crashes from causing daemon restart loops.

        December 2025 (Session 2): Added P2P connectivity check to health details.
        """
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

        try:
            if not self._running:
                return HealthCheckResult(
                    healthy=False,
                    status=CoordinatorStatus.STOPPED,
                    message=f"{self.db_type}SyncManager not running",
                )

            # Check database exists
            if not self.db_path.exists():
                return HealthCheckResult(
                    healthy=False,
                    status=CoordinatorStatus.DEGRADED,
                    message=f"Database not found: {self.db_path}",
                )

            # Calculate sync health metrics
            sync_rate = (
                self._db_state.successful_syncs / self._db_state.total_syncs
                if self._db_state.total_syncs > 0
                else 1.0
            )
            is_recent = time.time() - self._db_state.last_sync_timestamp < self.sync_interval * 2

            # Check P2P connectivity (December 2025 - Critical Gap Fix)
            p2p_healthy = self._check_p2p_health_sync()

            # Healthy if: sync rate > 50% AND last sync is recent
            # Degraded if P2P is down but sync is working (can still sync via SSH)
            is_healthy = sync_rate >= 0.5 and is_recent
            status = CoordinatorStatus.RUNNING if is_healthy else CoordinatorStatus.DEGRADED
            if not p2p_healthy and is_healthy:
                status = CoordinatorStatus.DEGRADED  # Degrade if P2P is down

            return HealthCheckResult(
                healthy=is_healthy,
                status=status,
                message=f"Syncing {self.db_type}: {self._db_state.local_record_count} records, "
                        f"{self._db_state.successful_syncs}/{self._db_state.total_syncs} syncs ok"
                        f"{'' if p2p_healthy else ' (P2P unavailable)'}",
                details={
                    "db_type": self.db_type,
                    "db_path": str(self.db_path),
                    "local_record_count": self._db_state.local_record_count,
                    "sync_rate": sync_rate,
                    "last_sync_age_seconds": time.time() - self._db_state.last_sync_timestamp,
                    "synced_from": self._db_state.synced_from,
                    "node_count": len(self.nodes),
                    "p2p_healthy": p2p_healthy,
                    "p2p_url": self.p2p_url,
                },
            )
        except Exception as e:
            # Prevent health_check crashes from causing daemon restart loops
            logger.warning(f"[{self.db_type}SyncManager] health_check error: {e}")
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"Health check error: {e}",
                details={"error": str(e)},
            )

    def _check_p2p_health_sync(self) -> bool:
        """Synchronous P2P health check.

        Returns:
            True if P2P is reachable, False otherwise.

        December 2025: Added for health_check() P2P status reporting.
        Uses a quick HTTP GET with short timeout.
        """
        import urllib.request
        import urllib.error

        try:
            url = f"{self.p2p_url}/health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except urllib.error.URLError:
            pass
        except (socket.error, socket.timeout, TimeoutError, ConnectionError, OSError):
            pass
        return False


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    "DatabaseSyncManager",
    "DatabaseSyncState",
    "SyncNodeInfo",
]
