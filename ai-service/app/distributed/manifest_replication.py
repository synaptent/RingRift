"""Manifest Replication - Distributed data manifest for fault tolerance.

This module provides replicated data manifest storage to prevent data loss
when the streaming collector crashes or is restarted. Key features:

1. Replicate manifest DB to multiple hosts after each sync cycle
2. Recover manifest from replicas on startup if local is stale/missing
3. SHA256 checksum verification for integrity
4. Async replication to avoid blocking the sync loop

Usage:
    replicator = ManifestReplicator(
        local_manifest_path=Path("data/data_manifest.db"),
        replica_hosts=["gh200_a", "gh200_b", "lambda_h100"],
    )

    # After each sync cycle
    await replicator.replicate_async()

    # On startup
    recovered = await replicator.recover_if_needed()
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ReplicaHost:
    """Configuration for a manifest replica host."""
    name: str
    ssh_host: str
    ssh_user: str = "ubuntu"
    ssh_port: int = 22
    remote_path: str = "~/ringrift/ai-service/data/data_manifest.db"
    enabled: bool = True
    last_replicated: float = 0.0
    last_checksum: str = ""


@dataclass
class ReplicationStatus:
    """Status of manifest replication."""
    local_checksum: str
    local_mtime: float
    local_size: int
    replicas: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    last_replication_time: float = 0.0
    replication_count: int = 0


class ManifestReplicator:
    """Handles manifest replication across cluster hosts."""

    def __init__(
        self,
        local_manifest_path: Path,
        replica_hosts: List[ReplicaHost],
        min_replicas: int = 3,  # Increased from 2 for better resilience
        replication_interval_seconds: int = 300,
        ssh_timeout: int = 30,
        scp_timeout: int = 120,
        external_backup_path: Optional[Path] = None,  # External drive backup
    ):
        """Initialize the manifest replicator.

        Args:
            local_manifest_path: Path to local manifest DB
            replica_hosts: List of hosts to replicate to
            min_replicas: Minimum successful replicas before considering safe (default: 3)
            replication_interval_seconds: Minimum time between replications
            ssh_timeout: SSH connection timeout
            scp_timeout: SCP transfer timeout
            external_backup_path: Optional path to external drive for local backup
        """
        self.local_path = local_manifest_path
        self.replica_hosts = {h.name: h for h in replica_hosts}
        self.min_replicas = min_replicas
        self.replication_interval = replication_interval_seconds
        self.ssh_timeout = ssh_timeout
        self.scp_timeout = scp_timeout
        self.external_backup_path = external_backup_path

        self._status = ReplicationStatus(
            local_checksum="",
            local_mtime=0.0,
            local_size=0,
        )
        self._replication_lock = asyncio.Lock()
        self._last_external_backup = 0.0

    def _compute_checksum(self, path: Path) -> str:
        """Compute SHA256 checksum of a file."""
        if not path.exists():
            return ""
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _get_local_manifest_info(self) -> Tuple[str, float, int]:
        """Get local manifest checksum, mtime, and size."""
        if not self.local_path.exists():
            return "", 0.0, 0
        stat = self.local_path.stat()
        checksum = self._compute_checksum(self.local_path)
        return checksum, stat.st_mtime, stat.st_size

    def _build_ssh_args(self, host: ReplicaHost) -> str:
        """Build SSH arguments string."""
        args = [
            f"-o ConnectTimeout={self.ssh_timeout}",
            "-o StrictHostKeyChecking=no",
            "-o BatchMode=yes",
        ]
        if host.ssh_port != 22:
            args.append(f"-p {host.ssh_port}")
        return " ".join(args)

    async def _replicate_to_host(self, host: ReplicaHost) -> bool:
        """Replicate manifest to a single host.

        Returns True on success.
        """
        if not host.enabled:
            return False

        if not self.local_path.exists():
            logger.warning(f"Local manifest does not exist: {self.local_path}")
            return False

        ssh_args = self._build_ssh_args(host)

        # Create remote directory if needed
        mkdir_cmd = f'ssh {ssh_args} {host.ssh_user}@{host.ssh_host} "mkdir -p $(dirname {host.remote_path})"'

        try:
            process = await asyncio.create_subprocess_shell(
                mkdir_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(process.communicate(), timeout=self.ssh_timeout)
        except Exception as e:
            logger.warning(f"Failed to create remote dir on {host.name}: {e}")
            return False

        # SCP the manifest
        # Use -P for scp port (different from ssh -p)
        scp_port_arg = f"-P {host.ssh_port}" if host.ssh_port != 22 else ""
        scp_cmd = f'scp -o ConnectTimeout={self.ssh_timeout} -o StrictHostKeyChecking=no {scp_port_arg} {self.local_path} {host.ssh_user}@{host.ssh_host}:{host.remote_path}'

        try:
            process = await asyncio.create_subprocess_shell(
                scp_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.scp_timeout,
            )

            if process.returncode != 0:
                logger.warning(f"SCP to {host.name} failed: {stderr.decode()}")
                return False

            # Verify checksum on remote
            local_checksum = self._compute_checksum(self.local_path)
            verify_cmd = f'ssh {ssh_args} {host.ssh_user}@{host.ssh_host} "sha256sum {host.remote_path} | cut -d\\  -f1"'

            process = await asyncio.create_subprocess_shell(
                verify_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.ssh_timeout,
            )

            remote_checksum = stdout.decode().strip()
            if remote_checksum != local_checksum:
                logger.warning(f"Checksum mismatch on {host.name}: local={local_checksum[:16]}... remote={remote_checksum[:16]}...")
                return False

            # Update host state
            host.last_replicated = time.time()
            host.last_checksum = local_checksum

            logger.info(f"Replicated manifest to {host.name} (checksum: {local_checksum[:16]}...)")
            return True

        except asyncio.TimeoutError:
            logger.warning(f"SCP to {host.name} timed out")
            return False
        except Exception as e:
            logger.warning(f"Replication to {host.name} failed: {e}")
            return False

    async def replicate_async(self, force: bool = False) -> int:
        """Replicate manifest to all replica hosts.

        Args:
            force: Force replication even if interval hasn't elapsed

        Returns:
            Number of successful replications
        """
        async with self._replication_lock:
            # Check if replication is needed
            if not force:
                elapsed = time.time() - self._status.last_replication_time
                if elapsed < self.replication_interval:
                    return 0

            # Check if manifest has changed
            local_checksum, local_mtime, local_size = self._get_local_manifest_info()
            if local_checksum == self._status.local_checksum and not force:
                return 0

            # Update local status
            self._status.local_checksum = local_checksum
            self._status.local_mtime = local_mtime
            self._status.local_size = local_size

            # Replicate to all hosts in parallel
            tasks = [
                self._replicate_to_host(host)
                for host in self.replica_hosts.values()
                if host.enabled
            ]

            if not tasks:
                logger.warning("No replica hosts configured")
                return 0

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Count successes
            success_count = sum(
                1 for r in results
                if isinstance(r, bool) and r
            )

            # Update status
            self._status.last_replication_time = time.time()
            self._status.replication_count += 1

            for host in self.replica_hosts.values():
                self._status.replicas[host.name] = {
                    "last_replicated": host.last_replicated,
                    "last_checksum": host.last_checksum,
                    "enabled": host.enabled,
                }

            if success_count >= self.min_replicas:
                logger.info(f"Manifest replicated to {success_count}/{len(tasks)} hosts")
            else:
                logger.warning(f"Only {success_count}/{len(tasks)} replicas succeeded (min: {self.min_replicas})")

            # Also backup to external drive if configured
            if self.external_backup_path:
                await self._backup_to_external_drive()

            return success_count

    async def _backup_to_external_drive(self) -> bool:
        """Backup manifest to external drive for additional resilience.

        Returns True on success.
        """
        if not self.external_backup_path:
            return False

        if not self.local_path.exists():
            return False

        # Rate limit external backups to every 15 minutes
        now = time.time()
        if now - self._last_external_backup < 900:  # 15 minutes
            return True  # Consider it success, just skipped

        try:
            # Check if external drive is mounted
            external_dir = self.external_backup_path.parent
            if not external_dir.exists():
                logger.warning(f"External drive not mounted: {external_dir}")
                return False

            # Create backup directory if needed
            self.external_backup_path.parent.mkdir(parents=True, exist_ok=True)

            # Compute checksums before and after copy
            local_checksum = self._compute_checksum(self.local_path)

            # Copy with timestamp for versioning
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            versioned_path = self.external_backup_path.with_suffix(f".{timestamp}.db")

            # Use shutil for local copy
            shutil.copy2(self.local_path, versioned_path)

            # Also update the "latest" symlink/copy
            if self.external_backup_path.exists():
                self.external_backup_path.unlink()
            shutil.copy2(self.local_path, self.external_backup_path)

            # Verify copy
            copied_checksum = self._compute_checksum(self.external_backup_path)
            if copied_checksum != local_checksum:
                logger.warning(f"External backup checksum mismatch")
                return False

            self._last_external_backup = now

            # Clean up old versioned backups (keep last 10)
            backup_dir = self.external_backup_path.parent
            pattern = self.external_backup_path.stem + ".*.db"
            old_backups = sorted(backup_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
            for old_backup in old_backups[:-10]:  # Keep last 10
                try:
                    old_backup.unlink()
                except OSError as e:
                    logger.debug(f"Could not remove old backup {old_backup.name}: {e}")

            logger.info(f"Backed up manifest to external drive: {self.external_backup_path}")
            return True

        except Exception as e:
            logger.warning(f"External backup failed: {e}")
            return False

    async def recover_from_external_drive(self) -> bool:
        """Recover manifest from external drive backup.

        Returns True on success.
        """
        if not self.external_backup_path or not self.external_backup_path.exists():
            return False

        try:
            # Backup existing local manifest
            if self.local_path.exists():
                backup_path = self.local_path.with_suffix(".db.bak")
                shutil.copy2(self.local_path, backup_path)

            # Ensure local directory exists
            self.local_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy from external drive
            shutil.copy2(self.external_backup_path, self.local_path)

            # Verify the recovered DB is valid
            conn = sqlite3.connect(self.local_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM synced_games")
            count = cursor.fetchone()[0]
            conn.close()

            logger.info(f"Recovered manifest from external drive with {count} synced games")
            return True

        except Exception as e:
            logger.warning(f"Recovery from external drive failed: {e}")
            # Restore backup if recovery failed
            backup_path = self.local_path.with_suffix(".db.bak")
            if backup_path.exists():
                shutil.copy2(backup_path, self.local_path)
            return False

    async def _get_remote_manifest_info(self, host: ReplicaHost) -> Optional[Tuple[str, float, int]]:
        """Get remote manifest checksum, mtime, and size.

        Returns None if remote manifest doesn't exist or is inaccessible.
        """
        ssh_args = self._build_ssh_args(host)

        # Get checksum, mtime, and size in one command
        cmd = f'''ssh {ssh_args} {host.ssh_user}@{host.ssh_host} "
            if [ -f {host.remote_path} ]; then
                echo -n 'CHECKSUM:' && sha256sum {host.remote_path} | cut -d\\  -f1
                echo -n 'MTIME:' && stat -c %Y {host.remote_path} 2>/dev/null || stat -f %m {host.remote_path}
                echo -n 'SIZE:' && stat -c %s {host.remote_path} 2>/dev/null || stat -f %z {host.remote_path}
            else
                echo 'NOTFOUND'
            fi
        "'''

        try:
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.ssh_timeout,
            )

            output = stdout.decode().strip()
            if "NOTFOUND" in output:
                return None

            # Parse output
            checksum = ""
            mtime = 0.0
            size = 0

            for line in output.split('\n'):
                line = line.strip()
                if line.startswith("CHECKSUM:"):
                    checksum = line.replace("CHECKSUM:", "").strip()
                elif line.startswith("MTIME:"):
                    mtime = float(line.replace("MTIME:", "").strip())
                elif line.startswith("SIZE:"):
                    size = int(line.replace("SIZE:", "").strip())

            if checksum:
                return checksum, mtime, size

            return None

        except Exception as e:
            logger.debug(f"Failed to get manifest info from {host.name}: {e}")
            return None

    async def _recover_from_host(self, host: ReplicaHost) -> bool:
        """Recover manifest from a remote host.

        Returns True on success.
        """
        ssh_args = self._build_ssh_args(host)

        # Backup existing local manifest if present
        if self.local_path.exists():
            backup_path = self.local_path.with_suffix(".db.bak")
            shutil.copy2(self.local_path, backup_path)
            logger.info(f"Backed up existing manifest to {backup_path}")

        # SCP from remote
        scp_port_arg = f"-P {host.ssh_port}" if host.ssh_port != 22 else ""
        scp_cmd = f'scp -o ConnectTimeout={self.ssh_timeout} -o StrictHostKeyChecking=no {scp_port_arg} {host.ssh_user}@{host.ssh_host}:{host.remote_path} {self.local_path}'

        try:
            # Ensure local directory exists
            self.local_path.parent.mkdir(parents=True, exist_ok=True)

            process = await asyncio.create_subprocess_shell(
                scp_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.scp_timeout,
            )

            if process.returncode != 0:
                logger.warning(f"Recovery from {host.name} failed: {stderr.decode()}")
                return False

            # Verify the recovered DB is valid
            try:
                conn = sqlite3.connect(self.local_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM synced_games")
                count = cursor.fetchone()[0]
                conn.close()
                logger.info(f"Recovered manifest from {host.name} with {count} synced games")
                return True
            except Exception as e:
                logger.warning(f"Recovered manifest from {host.name} is invalid: {e}")
                # Restore backup if recovery failed
                backup_path = self.local_path.with_suffix(".db.bak")
                if backup_path.exists():
                    shutil.copy2(backup_path, self.local_path)
                return False

        except Exception as e:
            logger.warning(f"Recovery from {host.name} failed: {e}")
            return False

    async def recover_if_needed(self) -> bool:
        """Recover manifest from replicas if local is missing or stale.

        Checks all replicas and recovers from the one with the most recent data.

        Returns True if recovery was performed successfully.
        """
        local_checksum, local_mtime, local_size = self._get_local_manifest_info()

        # If local exists and has data, check if replicas have newer versions
        local_game_count = 0
        if self.local_path.exists():
            try:
                conn = sqlite3.connect(self.local_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM synced_games")
                local_game_count = cursor.fetchone()[0]
                conn.close()
            except Exception:
                local_game_count = 0

        # Get info from all replicas
        replica_infos: List[Tuple[ReplicaHost, str, float, int]] = []

        tasks = [
            self._get_remote_manifest_info(host)
            for host in self.replica_hosts.values()
            if host.enabled
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for host, result in zip(
            [h for h in self.replica_hosts.values() if h.enabled],
            results,
        ):
            if isinstance(result, tuple) and len(result) == 3:
                checksum, mtime, size = result
                replica_infos.append((host, checksum, mtime, size))
                logger.debug(f"Replica {host.name}: mtime={mtime}, size={size}")

        if not replica_infos:
            logger.info("No replica manifests available")
            return False

        # Find the most recent replica (by mtime)
        best_replica = max(replica_infos, key=lambda x: x[2])
        best_host, best_checksum, best_mtime, best_size = best_replica

        # Decide if we need to recover
        should_recover = False
        reason = ""

        if not self.local_path.exists():
            should_recover = True
            reason = "local manifest missing"
        elif local_game_count == 0:
            should_recover = True
            reason = "local manifest empty"
        elif best_mtime > local_mtime + 60:  # 60s tolerance
            should_recover = True
            reason = f"replica {best_host.name} is newer (replica: {best_mtime}, local: {local_mtime})"
        elif best_checksum != local_checksum:
            # Check if replica has more data by size (crude heuristic)
            if best_size > local_size * 1.1:  # 10% larger
                should_recover = True
                reason = f"replica {best_host.name} has more data (replica: {best_size}, local: {local_size})"

        if not should_recover:
            logger.info("Local manifest is up-to-date, no recovery needed")
            return False

        logger.info(f"Recovering manifest: {reason}")
        success = await self._recover_from_host(best_host)

        if success:
            # Update local status
            self._status.local_checksum = best_checksum
            self._status.local_mtime = best_mtime
            self._status.local_size = best_size

        return success

    def get_status(self) -> Dict[str, Any]:
        """Get replication status."""
        local_checksum, local_mtime, local_size = self._get_local_manifest_info()

        return {
            "local_checksum": local_checksum[:16] + "..." if local_checksum else "",
            "local_mtime": local_mtime,
            "local_size": local_size,
            "last_replication_time": self._status.last_replication_time,
            "replication_count": self._status.replication_count,
            "replicas": {
                name: {
                    "last_replicated": host.last_replicated,
                    "last_checksum": host.last_checksum[:16] + "..." if host.last_checksum else "",
                    "enabled": host.enabled,
                }
                for name, host in self.replica_hosts.items()
            },
            "healthy": len([
                h for h in self.replica_hosts.values()
                if h.last_replicated > time.time() - self.replication_interval * 2
            ]) >= self.min_replicas,
        }


def create_replicator_from_config(
    manifest_path: Path,
    hosts_config_path: Path,
    min_replicas: int = 3,  # Increased from 2 for better resilience
    external_backup_path: Optional[Path] = None,
) -> ManifestReplicator:
    """Create a ManifestReplicator from configuration files.

    Args:
        manifest_path: Path to local manifest DB
        hosts_config_path: Path to remote_hosts.yaml
        min_replicas: Minimum replicas required for safety (default: 3)
        external_backup_path: Optional path for external drive backup

    Returns:
        Configured ManifestReplicator instance
    """
    import yaml

    replica_hosts = []

    if hosts_config_path.exists():
        with open(hosts_config_path) as f:
            hosts_data = yaml.safe_load(f) or {}

        # Use standard hosts as replicas (prefer GH200 nodes for redundancy)
        for name, host_data in hosts_data.get("standard_hosts", {}).items():
            # Skip training-only hosts, prefer selfplay hosts
            role = host_data.get("role", "")
            if "training" in role.lower() and "selfplay" not in role.lower():
                continue

            replica_hosts.append(ReplicaHost(
                name=name,
                ssh_host=host_data.get("ssh_host", ""),
                ssh_user=host_data.get("ssh_user", "ubuntu"),
                ssh_port=host_data.get("ssh_port", 22),
                remote_path=host_data.get(
                    "data_manifest_path",
                    "~/ringrift/ai-service/data/data_manifest.db"
                ),
            ))

        # Limit to first N hosts to avoid excessive replication
        replica_hosts = replica_hosts[:5]

    # Default external backup path if on macOS with external drive
    if external_backup_path is None:
        default_external = Path("/Volumes/RingRift-Data/selfplay_repository/manifest_backup/data_manifest.db")
        if default_external.parent.parent.exists():  # Check if drive is mounted
            external_backup_path = default_external

    return ManifestReplicator(
        local_manifest_path=manifest_path,
        replica_hosts=replica_hosts,
        min_replicas=min_replicas,
        external_backup_path=external_backup_path,
    )
