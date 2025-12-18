"""P2P Sync Client - HTTP-based data sync fallback.

This module provides HTTP-based data synchronization as a fallback when
SSH/rsync fails. Uses the P2P orchestrator's sync endpoints.

Key features:
1. Fallback to HTTP when SSH fails
2. Streaming file transfer with backpressure
3. Checksum verification
4. Circuit breaker integration

Usage:
    client = P2PSyncClient()

    # Sync files from a peer
    result = await client.sync_from_peer(
        peer_host="192.168.1.100",
        peer_port=8770,
        files=["data/games/selfplay.db"],
        local_dir=Path("data/games/synced/peer1"),
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Import circuit breaker for fault tolerance
try:
    from app.distributed.circuit_breaker import (
        get_operation_breaker,
        get_adaptive_timeout,
        CircuitOpenError,
    )
    HAS_CIRCUIT_BREAKER = True
except ImportError:
    HAS_CIRCUIT_BREAKER = False

# Health check should be fast - don't wait 300s for a health check!
HEALTH_CHECK_TIMEOUT = 5  # seconds


@dataclass
class P2PSyncConfig:
    """Configuration for P2P sync client."""
    connect_timeout: int = 10
    read_timeout: int = 300
    chunk_size: int = 65536  # 64KB chunks
    max_retries: int = 3
    retry_delay: float = 2.0
    verify_checksum: bool = True


@dataclass
class SyncResult:
    """Result of a sync operation."""
    success: bool
    files_synced: int = 0
    bytes_transferred: int = 0
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)
    checksums: Dict[str, str] = field(default_factory=dict)


class P2PSyncClient:
    """HTTP-based P2P sync client."""

    def __init__(self, config: Optional[P2PSyncConfig] = None):
        self.config = config or P2PSyncConfig()
        self._session: Optional[Any] = None

    async def _get_session(self):
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            import aiohttp
            timeout = aiohttp.ClientTimeout(
                total=self.config.read_timeout,
                connect=self.config.connect_timeout,
            )
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def check_peer_health(self, peer_host: str, peer_port: int) -> bool:
        """Check if a peer is healthy and reachable.

        Returns True if peer is healthy.
        """
        # Circuit breaker check
        if HAS_CIRCUIT_BREAKER:
            breaker = get_operation_breaker("p2p")
            if not breaker.can_execute(peer_host):
                logger.debug(f"Circuit breaker open for {peer_host}, skipping health check")
                return False

        try:
            # Use dedicated health check timeout, not the full 300s read timeout
            import aiohttp
            health_timeout = aiohttp.ClientTimeout(
                total=HEALTH_CHECK_TIMEOUT,
                connect=HEALTH_CHECK_TIMEOUT,
            )
            async with aiohttp.ClientSession(timeout=health_timeout) as session:
                url = f"http://{peer_host}:{peer_port}/health"

                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        is_healthy = data.get("status") == "ok" or data.get("healthy", False)
                        if is_healthy and HAS_CIRCUIT_BREAKER:
                            get_operation_breaker("p2p").record_success(peer_host)
                        return is_healthy
                    return False

        except Exception as e:
            logger.debug(f"Peer health check failed for {peer_host}:{peer_port}: {e}")
            if HAS_CIRCUIT_BREAKER:
                get_operation_breaker("p2p").record_failure(peer_host, e)
            return False

    async def list_peer_files(
        self,
        peer_host: str,
        peer_port: int,
        pattern: str = "data/games/*.db",
    ) -> List[Dict[str, Any]]:
        """List files available on a peer.

        Returns list of file info dicts with path, size, mtime.
        """
        try:
            session = await self._get_session()
            url = f"http://{peer_host}:{peer_port}/sync/files"
            params = {"pattern": pattern}

            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("files", [])
                return []

        except Exception as e:
            logger.warning(f"Failed to list files on {peer_host}:{peer_port}: {e}")
            return []

    async def _download_file(
        self,
        peer_host: str,
        peer_port: int,
        remote_path: str,
        local_path: Path,
    ) -> Tuple[bool, int, str]:
        """Download a single file from peer.

        Returns (success, bytes_transferred, checksum).
        """
        # Circuit breaker check
        if HAS_CIRCUIT_BREAKER:
            breaker = get_operation_breaker("p2p")
            if not breaker.can_execute(peer_host):
                logger.debug(f"Circuit breaker open for {peer_host}, skipping download")
                return False, 0, ""

        try:
            session = await self._get_session()
            url = f"http://{peer_host}:{peer_port}/sync/file"
            params = {"path": remote_path}

            # Ensure local directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Stream download
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.warning(f"Failed to download {remote_path}: {resp.status} - {error_text}")
                    if HAS_CIRCUIT_BREAKER:
                        get_operation_breaker("p2p").record_failure(peer_host)
                    return False, 0, ""

                # Write to temp file first
                temp_path = local_path.with_suffix(".tmp")
                sha256 = hashlib.sha256()
                bytes_written = 0

                with open(temp_path, 'wb') as f:
                    async for chunk in resp.content.iter_chunked(self.config.chunk_size):
                        f.write(chunk)
                        sha256.update(chunk)
                        bytes_written += len(chunk)

                checksum = sha256.hexdigest()

                # Verify checksum if provided in headers
                expected_checksum = resp.headers.get("X-Checksum")
                if expected_checksum and self.config.verify_checksum:
                    if checksum != expected_checksum:
                        logger.error(f"Checksum mismatch for {remote_path}: expected {expected_checksum}, got {checksum}")
                        temp_path.unlink(missing_ok=True)
                        if HAS_CIRCUIT_BREAKER:
                            get_operation_breaker("p2p").record_failure(peer_host)
                        return False, 0, ""

                # Move to final location
                temp_path.rename(local_path)
                # Record success
                if HAS_CIRCUIT_BREAKER:
                    get_operation_breaker("p2p").record_success(peer_host)
                return True, bytes_written, checksum

        except Exception as e:
            logger.error(f"Error downloading {remote_path}: {e}")
            if HAS_CIRCUIT_BREAKER:
                get_operation_breaker("p2p").record_failure(peer_host, e)
            return False, 0, ""

    async def sync_file(
        self,
        peer_host: str,
        peer_port: int,
        remote_path: str,
        local_path: Path,
    ) -> SyncResult:
        """Sync a single file from peer with retries.

        Returns SyncResult.
        """
        start_time = time.time()
        errors = []

        for attempt in range(self.config.max_retries):
            success, bytes_transferred, checksum = await self._download_file(
                peer_host, peer_port, remote_path, local_path
            )

            if success:
                return SyncResult(
                    success=True,
                    files_synced=1,
                    bytes_transferred=bytes_transferred,
                    duration_seconds=time.time() - start_time,
                    checksums={remote_path: checksum},
                )

            errors.append(f"Attempt {attempt + 1} failed")

            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))

        return SyncResult(
            success=False,
            files_synced=0,
            duration_seconds=time.time() - start_time,
            errors=errors,
        )

    async def sync_from_peer(
        self,
        peer_host: str,
        peer_port: int,
        files: Optional[List[str]] = None,
        pattern: str = "data/games/*.db",
        local_dir: Optional[Path] = None,
    ) -> SyncResult:
        """Sync files from a peer.

        Args:
            peer_host: Peer hostname or IP
            peer_port: Peer P2P port
            files: Specific files to sync (if None, uses pattern)
            pattern: Glob pattern for files to sync
            local_dir: Local directory to sync to

        Returns:
            SyncResult with sync statistics
        """
        start_time = time.time()

        # Check peer health first
        if not await self.check_peer_health(peer_host, peer_port):
            return SyncResult(
                success=False,
                errors=[f"Peer {peer_host}:{peer_port} is not healthy"],
            )

        # Get list of files to sync
        if files is None:
            file_list = await self.list_peer_files(peer_host, peer_port, pattern)
            files = [f["path"] for f in file_list]

        if not files:
            return SyncResult(
                success=True,
                files_synced=0,
                duration_seconds=time.time() - start_time,
            )

        # Determine local directory
        if local_dir is None:
            local_dir = Path("data/games/synced") / f"{peer_host}_{peer_port}"
        local_dir.mkdir(parents=True, exist_ok=True)

        # Sync each file
        total_bytes = 0
        synced_count = 0
        errors = []
        checksums = {}

        for remote_path in files:
            # Determine local path
            filename = Path(remote_path).name
            local_path = local_dir / filename

            result = await self.sync_file(peer_host, peer_port, remote_path, local_path)

            if result.success:
                synced_count += result.files_synced
                total_bytes += result.bytes_transferred
                checksums.update(result.checksums)
            else:
                errors.extend(result.errors)

        return SyncResult(
            success=len(errors) == 0,
            files_synced=synced_count,
            bytes_transferred=total_bytes,
            duration_seconds=time.time() - start_time,
            errors=errors,
            checksums=checksums,
        )

    async def request_pull_from_peer(
        self,
        orchestrator_host: str,
        orchestrator_port: int,
        source_node_id: str,
        target_node_id: str,
        files: List[str],
    ) -> bool:
        """Request the P2P orchestrator to initiate a pull.

        This is for coordinated syncs where the orchestrator manages the transfer.

        Returns True if request was accepted.
        """
        try:
            session = await self._get_session()
            url = f"http://{orchestrator_host}:{orchestrator_port}/sync/pull"

            payload = {
                "source_node_id": source_node_id,
                "target_node_id": target_node_id,
                "files": files,
            }

            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("success", False)
                return False

        except Exception as e:
            logger.warning(f"Failed to request pull: {e}")
            return False


class P2PFallbackSync:
    """Fallback sync mechanism using P2P HTTP when SSH fails.

    Integrates with streaming_data_collector to provide redundant sync.
    """

    def __init__(
        self,
        p2p_port: int = 8770,
        connect_timeout: int = 10,
    ):
        self.p2p_port = p2p_port
        self.client = P2PSyncClient(P2PSyncConfig(connect_timeout=connect_timeout))

        # Track which hosts support P2P
        self._p2p_capable_hosts: Dict[str, bool] = {}

    async def close(self):
        """Close the client."""
        await self.client.close()

    async def is_p2p_available(self, host: str) -> bool:
        """Check if P2P sync is available for a host.

        Caches results to avoid repeated checks.
        """
        if host in self._p2p_capable_hosts:
            return self._p2p_capable_hosts[host]

        is_available = await self.client.check_peer_health(host, self.p2p_port)
        self._p2p_capable_hosts[host] = is_available
        return is_available

    async def sync_with_fallback(
        self,
        host: str,
        ssh_host: str,
        ssh_user: str,
        ssh_port: int,
        remote_db_path: str,
        local_dir: Path,
        ssh_timeout: int = 30,
        rsync_timeout: int = 300,
    ) -> Tuple[bool, int, str]:
        """Sync from host with SSH as primary and P2P HTTP as fallback.

        Returns (success, games_synced, method_used).
        """
        # Try SSH/rsync first
        try:
            ssh_args = f"-o ConnectTimeout={ssh_timeout}"
            if ssh_port != 22:
                ssh_args += f" -p {ssh_port}"

            rsync_cmd = f'rsync -avz --checksum -e "ssh {ssh_args}" {ssh_user}@{ssh_host}:{remote_db_path}/*.db {local_dir}/'

            process = await asyncio.create_subprocess_shell(
                rsync_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=rsync_timeout,
            )

            if process.returncode == 0:
                # Count synced games
                games = 0
                for db_file in local_dir.glob("*.db"):
                    try:
                        import sqlite3
                        conn = sqlite3.connect(db_file)
                        cursor = conn.cursor()
                        cursor.execute("SELECT COUNT(*) FROM games")
                        games += cursor.fetchone()[0]
                        conn.close()
                    except Exception:
                        pass
                return True, games, "ssh"

        except asyncio.TimeoutError:
            logger.warning(f"SSH sync to {host} timed out, trying P2P fallback")
        except Exception as e:
            logger.warning(f"SSH sync to {host} failed: {e}, trying P2P fallback")

        # Fallback to P2P HTTP
        if await self.is_p2p_available(ssh_host):
            logger.info(f"Using P2P HTTP fallback for {host}")

            result = await self.client.sync_from_peer(
                peer_host=ssh_host,
                peer_port=self.p2p_port,
                pattern=f"{remote_db_path}/*.db",
                local_dir=local_dir,
            )

            if result.success:
                # Count synced games
                games = 0
                for db_file in local_dir.glob("*.db"):
                    try:
                        import sqlite3
                        conn = sqlite3.connect(db_file)
                        cursor = conn.cursor()
                        cursor.execute("SELECT COUNT(*) FROM games")
                        games += cursor.fetchone()[0]
                        conn.close()
                    except Exception:
                        pass
                return True, games, "p2p_http"
            else:
                logger.error(f"P2P fallback failed for {host}: {result.errors}")
                return False, 0, "failed"

        return False, 0, "no_fallback"


# Convenience function for integration
async def sync_with_p2p_fallback(
    host_name: str,
    ssh_host: str,
    ssh_user: str,
    ssh_port: int,
    remote_db_path: str,
    local_dir: Path,
    p2p_port: int = 8770,
) -> Tuple[bool, int, str]:
    """Convenience function for one-shot sync with P2P fallback.

    Returns (success, games_synced, method_used).
    """
    fallback = P2PFallbackSync(p2p_port=p2p_port)
    try:
        return await fallback.sync_with_fallback(
            host=host_name,
            ssh_host=ssh_host,
            ssh_user=ssh_user,
            ssh_port=ssh_port,
            remote_db_path=remote_db_path,
            local_dir=local_dir,
        )
    finally:
        await fallback.close()
