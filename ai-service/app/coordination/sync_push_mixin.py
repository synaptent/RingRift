"""Push/broadcast sync operations mixin for AutoSyncDaemon.

December 2025: Extracted from auto_sync_daemon.py as part of mixin-based refactoring.
December 2025: Updated to inherit from SyncMixinBase for common functionality.

This mixin provides:
- Local database discovery
- Bandwidth management per node
- Broadcast target selection
- Rsync-based sync with retry logic
- Broadcast sync cycle management
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.core.async_context import fire_and_forget
from app.coordination.sync_mixin_base import SyncMixinBase
from app.config.coordination_defaults import build_ssh_options

if TYPE_CHECKING:
    from app.coordination.sync_strategies import AutoSyncConfig, SyncStats

logger = logging.getLogger(__name__)


class SyncPushMixin(SyncMixinBase):
    """Mixin providing push/broadcast sync operations for AutoSyncDaemon.

    Inherits from SyncMixinBase for common error handling and logging utilities.

    Additional expected attributes from main class:
    - _is_broadcast: bool

    Event emission methods inherited from SyncMixinBase:
    - _emit_sync_failure(target_node, db_path, error)
    - _emit_sync_stalled(target_node, timeout_seconds, ...)
    """

    # Additional type hints specific to this mixin
    _is_broadcast: bool

    def discover_local_databases(self) -> list[Path]:
        """Find all game databases on this node that should be synced.

        December 2025: Consolidated from cluster_data_sync.py
        """
        base_dir = Path(__file__).resolve().parent.parent.parent
        data_dir = base_dir / "data" / "games"

        if not data_dir.exists():
            return []

        # Database patterns to sync
        sync_patterns = [
            "canonical_*.db",
            "gumbel_*.db",
            "selfplay_*.db",
            "synced/*.db",
        ]

        databases = []
        for pattern in sync_patterns:
            databases.extend(data_dir.glob(pattern))

        # Filter out empty databases
        databases = [db for db in databases if db.stat().st_size > 1024]

        # Sort by priority (high-priority configs first)
        high_priority_configs = frozenset(self.config.broadcast_high_priority_configs)

        def priority_key(path: Path) -> tuple[int, str]:
            name = path.stem
            for config in high_priority_configs:
                if config in name:
                    return (0, name)
            return (1, name)

        databases.sort(key=priority_key)

        logger.info(f"[AutoSyncDaemon] Found {len(databases)} databases to sync")
        return databases

    def get_bandwidth_for_node(self, node_id: str, provider: str = "default") -> int:
        """Get bandwidth limit in KB/s for a specific node.

        December 2025: Consolidated to use cluster_config.get_node_bandwidth_kbs()
        which provides a unified source of truth for bandwidth limits.

        Args:
            node_id: Target node ID
            provider: Provider name (unused, kept for backward compatibility)

        Returns:
            Bandwidth limit in KB/s
        """
        try:
            from app.config.cluster_config import get_node_bandwidth_kbs

            bw = get_node_bandwidth_kbs(node_id)
            logger.debug(f"[AutoSyncDaemon] Using bandwidth {bw}KB/s for {node_id}")
            return bw
        except ImportError:
            # Fallback if cluster_config not available
            logger.warning("[AutoSyncDaemon] cluster_config not available, using defaults")
            return 20_000  # Conservative default

    async def get_broadcast_targets(self) -> list[dict[str, Any]]:
        """Get nodes eligible to receive broadcast sync data.

        December 2025: Consolidated from cluster_data_sync.py

        Filters:
        - Not excluded by policy
        - Has sufficient free disk space
        - Not retired
        - Is reachable (recent heartbeat)
        """
        from urllib.request import Request, urlopen

        try:
            from app.config.ports import get_p2p_status_url
            from app.coordination.coordinator_config import get_exclusion_policy

            url = get_p2p_status_url()
            req = Request(url, headers={"Accept": "application/json"})

            with urlopen(req, timeout=10) as resp:
                status = json.loads(resp.read().decode())

        except (OSError, ValueError, json.JSONDecodeError, TimeoutError) as e:
            logger.warning(f"[AutoSyncDaemon] Failed to get P2P status: {e}")
            return []

        if not status:
            return []

        targets = []

        try:
            exclusion_policy = get_exclusion_policy()
        except ImportError:
            exclusion_policy = None

        peers = status.get("peers", {})
        for node_id, info in peers.items():
            # Skip excluded nodes
            if exclusion_policy and exclusion_policy.should_exclude(node_id):
                continue

            # Skip retired nodes
            if info.get("retired", False):
                continue

            # Check disk space
            disk_free = info.get("disk_free_gb", 0)
            min_disk = 50  # Default
            if exclusion_policy:
                min_disk = getattr(exclusion_policy, 'min_disk_free_gb', 50)
            if disk_free < min_disk:
                continue

            # Check for stale heartbeat (>5 min old)
            last_heartbeat = info.get("last_heartbeat", 0)
            if time.time() - last_heartbeat > 300:
                continue

            # Get host address
            host = info.get("host", "")
            if not host:
                continue

            # Detect provider for bandwidth hints
            provider = info.get("provider", "default")
            if not provider or provider == "default":
                node_lower = node_id.lower()
                if "lambda" in node_lower:
                    provider = "lambda"
                elif "runpod" in node_lower:
                    provider = "runpod"
                elif "nebius" in node_lower:
                    provider = "nebius"
                elif "vast" in node_lower:
                    provider = "vast"
                elif "vultr" in node_lower:
                    provider = "vultr"
                elif "hetzner" in node_lower:
                    provider = "hetzner"

            targets.append({
                "node_id": node_id,
                "host": host,
                "disk_free_gb": disk_free,
                "is_nfs": info.get("nfs_accessible", False),
                "provider": provider,
            })

        # Sort by disk space (push to nodes with most space first)
        targets.sort(key=lambda t: t["disk_free_gb"], reverse=True)

        logger.info(f"[AutoSyncDaemon] Found {len(targets)} broadcast targets")
        return targets

    async def sync_to_target_with_retry(
        self,
        source: Path,
        target: dict[str, Any],
        max_retries: int | None = None,
    ) -> dict[str, Any]:
        """Sync database to target with exponential backoff retry.

        December 2025: Uses _retry_with_backoff from SyncMixinBase for
        consolidated retry logic.

        Args:
            source: Source database path
            target: Target node info dict
            max_retries: Max retry attempts (default: SYNC_MAX_RETRIES)

        Returns:
            Sync result dict with success, bytes_transferred, duration, error
        """
        target_id = target.get("node_id", "unknown")

        # Use base class retry helper
        return await self._retry_with_backoff(
            self.broadcast_sync_to_target,
            source,
            target,
            max_retries=max_retries,
            operation_name=f"Sync to {target_id}",
        )

    async def broadcast_sync_to_target(
        self,
        source: Path,
        target: dict[str, Any],
    ) -> dict[str, Any]:
        """Push a database to a target node using rsync (broadcast mode).

        December 2025: Consolidated from cluster_data_sync.py

        Args:
            source: Source database path
            target: Target node info dict

        Returns:
            Sync result dict with success, bytes_transferred, duration, error
        """
        start_time = time.time()

        # NFS optimization: Lambda nodes share storage, no sync needed
        if target.get("is_nfs", False):
            logger.debug(f"[AutoSyncDaemon] Skipping sync to {target['node_id']}: NFS-connected")
            return {
                "source": str(source),
                "target": target["node_id"],
                "success": True,
                "bytes_transferred": 0,
                "duration_seconds": 0,
            }

        # Dec 2025: CRITICAL - Checkpoint WAL before transfer to prevent corruption
        try:
            from app.coordination.sync_integrity import prepare_database_for_transfer
            prep_success, prep_msg = prepare_database_for_transfer(source)
            if not prep_success:
                logger.warning(
                    f"[AutoSyncDaemon] Failed to prepare {source} for broadcast: {prep_msg}"
                )
        except ImportError:
            pass  # sync_integrity not available
        except (OSError, sqlite3.Error) as e:
            logger.warning(f"[AutoSyncDaemon] Error preparing database for broadcast: {e}")

        # Get provider-specific bandwidth limit
        bandwidth_kbps = self.get_bandwidth_for_node(
            target["node_id"],
            target.get("provider", "default"),
        )

        # Dec 2025: Get storage path from cluster config (supports OWC routing)
        from app.config.cluster_config import get_cluster_nodes
        cluster_nodes = get_cluster_nodes()
        node_config = cluster_nodes.get(target["node_id"])
        if node_config:
            games_path = node_config.get_storage_path("games")
        else:
            games_path = "~/ringrift/ai-service/data/games"

        # Feb 2026: Memory-aware transfer - try remote pull if memory is high
        try:
            from app.coordination.rsync_command_builder import should_use_rsync, trigger_remote_pull
            if not should_use_rsync():
                logger.info(
                    f"[AutoSyncDaemon] Memory-aware: triggering remote pull on "
                    f"{target['node_id']} instead of rsync push"
                )
                pull_ok = await trigger_remote_pull(
                    target_host=target["host"],
                    target_port=target.get("port", 8770),
                    source_node_id=getattr(self, "node_id", "coordinator"),
                    files=[str(source)],
                )
                if pull_ok:
                    return {
                        "source": str(source),
                        "target": target["node_id"],
                        "success": True,
                        "bytes_transferred": source.stat().st_size if source.exists() else 0,
                        "duration_seconds": time.time() - start_time,
                        "method": "remote_pull",
                    }
                logger.info("[AutoSyncDaemon] Remote pull failed, falling back to rsync")
        except ImportError:
            pass

        # Build rsync command
        ssh_user = node_config.ssh_user if node_config else "ubuntu"
        target_path = f"{ssh_user}@{target['host']}:{games_path}/synced/"
        # December 30, 2025: Use centralized SSH config for consistent timeouts
        # and per-provider adjustments (Vast/Hetzner get 15s, others 10s)
        ssh_opts = build_ssh_options(
            key_path="~/.ssh/id_cluster",
            node_id=target.get("node_id"),  # Auto-detects provider from node ID
            include_keepalive=True,  # Long-running transfers need keepalive
        )
        # December 2025: Removed --partial to prevent corruption from stitched segments
        # on connection resets. Fresh transfers are safer than resumed partial ones.
        # December 27, 2025: Removed --inplace as it conflicts with --delay-updates
        # --delay-updates provides atomic file updates (safer for databases)
        # --inplace writes directly to file (faster but not atomic)
        # These are mutually exclusive - choose safety over speed
        cmd = [
            "rsync",
            "-avz",
            "--progress",
            f"--bwlimit={bandwidth_kbps}",
            "--timeout=60",
            "--delay-updates",
            "--checksum",
            "-e", ssh_opts,
            str(source),
            target_path,
        ]

        # Dynamic timeout: 2 seconds per MB, minimum 120s, maximum 1800s
        file_size_mb = source.stat().st_size / (1024 * 1024) if source.exists() else 100
        dynamic_timeout = max(120, min(1800, int(60 + file_size_mb * 2)))

        try:
            logger.info(f"[AutoSyncDaemon] Syncing {source.name} to {target['node_id']}")

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            _stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=dynamic_timeout,
            )

            duration = time.time() - start_time

            if proc.returncode == 0:
                bytes_transferred = source.stat().st_size

                # December 2025: Checksum verification after broadcast sync
                verify_checksum = target.get("verify_checksum", True)
                if verify_checksum and node_config:
                    try:
                        from app.coordination.sync_integrity import verify_and_retry_sync

                        remote_file_path = f"{games_path}/synced/{source.name}"

                        async def retry_broadcast():
                            # Retry with same rsync command
                            retry_proc = await asyncio.create_subprocess_exec(
                                *cmd,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE,
                            )
                            _, _ = await asyncio.wait_for(
                                retry_proc.communicate(),
                                timeout=dynamic_timeout,
                            )
                            return retry_proc.returncode == 0

                        checksum_ok, error = await verify_and_retry_sync(
                            source_path=str(source),
                            dest_path=remote_file_path,
                            ssh_host=target["host"],
                            ssh_user=ssh_user,
                            ssh_key="~/.ssh/id_cluster",
                            sync_func=retry_broadcast,
                            max_retries=1,
                        )

                        if not checksum_ok:
                            logger.error(
                                f"[AutoSyncDaemon] Checksum verification failed for broadcast "
                                f"{source.name} -> {target['node_id']}: {error}"
                            )
                            self._stats.databases_verification_failed += 1
                            return {
                                "source": str(source),
                                "target": target["node_id"],
                                "success": False,
                                "duration_seconds": duration,
                                "error": f"Checksum verification failed: {error}",
                            }
                        else:
                            self._stats.databases_verified += 1
                            logger.debug(
                                f"[AutoSyncDaemon] Broadcast checksum verified: "
                                f"{source.name} -> {target['node_id']}"
                            )

                    except ImportError:
                        logger.debug("[AutoSyncDaemon] sync_integrity not available, skipping verification")
                    except Exception as e:
                        logger.warning(f"[AutoSyncDaemon] Broadcast checksum verification error: {e}")
                        # Don't fail the sync if verification fails - just log it
                        self._stats.databases_verification_failed += 1

                logger.info(
                    f"[AutoSyncDaemon] Synced {source.name} to {target['node_id']} in {duration:.1f}s"
                )
                return {
                    "source": str(source),
                    "target": target["node_id"],
                    "success": True,
                    "bytes_transferred": bytes_transferred,
                    "duration_seconds": duration,
                }
            else:
                error = stderr.decode().strip() if stderr else "Unknown error"
                logger.warning(f"[AutoSyncDaemon] Sync failed to {target['node_id']}: {error}")
                # Dec 2025: Emit DATA_SYNC_FAILED for individual sync failures
                fire_and_forget(
                    self._emit_sync_failure(target["node_id"], str(source), error),
                    on_error=lambda e: logger.debug(f"Failed to emit sync failure: {e}"),
                )
                return {
                    "source": str(source),
                    "target": target["node_id"],
                    "success": False,
                    "duration_seconds": duration,
                    "error": error,
                }

        except asyncio.TimeoutError:
            logger.warning(f"[AutoSyncDaemon] Rsync to {target['node_id']} timed out, trying fallback")
            # December 2025: Use robust_push fallback on timeout
            fallback_result = await self._robust_push_fallback(
                source, target, ssh_user, games_path, start_time
            )
            if fallback_result.get("success"):
                return fallback_result

            # Emit SYNC_STALLED for failover routing (Dec 2025)
            fire_and_forget(
                self._emit_sync_stalled(
                    target_node=target["node_id"],
                    timeout_seconds=dynamic_timeout,
                    data_type="game",
                ),
                on_error=lambda e: logger.debug(f"Failed to emit SYNC_STALLED: {e}"),
            )
            return {
                "source": str(source),
                "target": target["node_id"],
                "success": False,
                "duration_seconds": time.time() - start_time,
                "error": "Timeout (all transports failed)",
            }
        except (OSError, asyncio.CancelledError, subprocess.SubprocessError) as e:
            logger.warning(f"[AutoSyncDaemon] Rsync to {target['node_id']} error: {e}, trying fallback")
            # December 2025: Use robust_push fallback on rsync failure
            fallback_result = await self._robust_push_fallback(
                source, target, ssh_user, games_path, start_time
            )
            if fallback_result.get("success"):
                return fallback_result

            # Dec 2025: Emit DATA_SYNC_FAILED for sync exceptions
            fire_and_forget(
                self._emit_sync_failure(target["node_id"], str(source), str(e)),
                on_error=lambda exc: logger.debug(f"Failed to emit sync failure: {exc}"),
            )
            return {
                "source": str(source),
                "target": target["node_id"],
                "success": False,
                "duration_seconds": time.time() - start_time,
                "error": str(e),
            }

    async def _robust_push_fallback(
        self,
        source: Path,
        target: dict[str, Any],
        ssh_user: str,
        games_path: str,
        start_time: float,
    ) -> dict[str, Any]:
        """Fallback to robust_push when rsync fails.

        December 2025: Added for improved sync reliability.
        Uses multi-transport fallback: rsync → scp → chunked/base64
        February 2026: Added file size guard to prevent OOM on large files.

        Args:
            source: Source database path
            target: Target node info dict
            ssh_user: SSH username
            games_path: Remote games directory path
            start_time: Sync start time for duration tracking

        Returns:
            Sync result dict with success, bytes_transferred, duration, error
        """
        # Feb 2026: For large files, use HTTP pull instead of push to avoid OOM.
        # The coordinator's P2P server streams files in 1MB chunks via
        # GET /files/data/*, so the remote node can pull without either side
        # loading the entire file into memory.
        http_pull_threshold = 500 * 1024 * 1024  # 500MB
        try:
            file_size = source.stat().st_size
        except OSError:
            file_size = 0

        if file_size > http_pull_threshold:
            http_result = await self._http_pull_fallback(
                source, target, ssh_user, games_path, start_time
            )
            if http_result.get("success"):
                return http_result
            # If HTTP pull also failed, don't try robust_push for huge files
            if file_size > 2 * 1024 * 1024 * 1024:  # >2GB
                logger.warning(
                    f"[AutoSyncDaemon] HTTP pull failed and file too large for push fallback "
                    f"({file_size / 1024 / 1024 / 1024:.1f}GB) to {target['node_id']}"
                )
                return http_result
            # For 500MB-2GB, fall through to robust_push (which uses chunked, not base64)

        try:
            from scripts.lib.transfer import robust_push, TransferConfig

            # Get SSH key from cluster config or use default
            ssh_key = "~/.ssh/id_cluster"
            try:
                from app.config.cluster_config import get_cluster_nodes
                cluster_nodes = get_cluster_nodes()
                node_config = cluster_nodes.get(target["node_id"])
                if node_config and node_config.ssh_key:
                    ssh_key = node_config.ssh_key
            except ImportError:
                pass

            config = TransferConfig(
                ssh_key=ssh_key,
                ssh_user=ssh_user,
                ssh_port=22,
                connect_timeout=10,
                transfer_timeout=300,
                max_retries=2,
                compress=True,
                verify_checksum=True,
            )

            remote_path = f"{games_path}/synced/{source.name}"

            # Run in thread pool since robust_push is synchronous
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: robust_push(source, target["host"], 22, remote_path, config),
            )

            if result.success:
                logger.info(
                    f"[AutoSyncDaemon] Fallback sync succeeded via {result.method}: "
                    f"{source.name} -> {target['node_id']}"
                )
                return {
                    "source": str(source),
                    "target": target["node_id"],
                    "success": True,
                    "bytes_transferred": result.bytes_transferred,
                    "duration_seconds": time.time() - start_time,
                    "method": result.method,
                }
            else:
                logger.warning(
                    f"[AutoSyncDaemon] All fallback transports failed for {target['node_id']}: "
                    f"{result.error}"
                )
                return {
                    "source": str(source),
                    "target": target["node_id"],
                    "success": False,
                    "duration_seconds": time.time() - start_time,
                    "error": result.error,
                }

        except ImportError:
            logger.debug("[AutoSyncDaemon] robust_push not available")
            return {
                "source": str(source),
                "target": target["node_id"],
                "success": False,
                "duration_seconds": time.time() - start_time,
                "error": "Fallback transport not available",
            }
        except Exception as e:
            logger.warning(f"[AutoSyncDaemon] Fallback sync error: {e}")
            return {
                "source": str(source),
                "target": target["node_id"],
                "success": False,
                "duration_seconds": time.time() - start_time,
                "error": str(e),
            }

    async def _http_pull_fallback(
        self,
        source: Path,
        target: dict[str, Any],
        ssh_user: str,
        games_path: str,
        start_time: float,
    ) -> dict[str, Any]:
        """Fallback: tell remote node to pull file via HTTP from coordinator.

        February 2026: Added to avoid OOM from push-based transfers of large
        database files. The coordinator's P2P server streams files in 1MB chunks
        via GET /files/data/*, so neither side loads the entire file into memory.

        Flow:
        1. Determine coordinator's accessible IP (Tailscale or SSH host)
        2. SSH to remote node
        3. Remote runs: curl -o <dest> http://<coordinator>:8770/files/data/games/<file>
        4. Verify file size matches
        """
        try:
            # Get coordinator's IP accessible from the remote node
            coordinator_ip = None
            try:
                from app.config.node_identity import get_tailscale_ip
                coordinator_ip = get_tailscale_ip()
            except ImportError:
                pass

            if not coordinator_ip:
                return {
                    "source": str(source),
                    "target": target["node_id"],
                    "success": False,
                    "duration_seconds": time.time() - start_time,
                    "error": "Cannot determine coordinator IP for HTTP pull",
                }

            from app.config.ports import P2P_DEFAULT_PORT

            # Build the relative path for the /files/data/ endpoint
            # source is like /path/to/ai-service/data/games/canonical_hex8_2p.db
            # The endpoint serves from data/ dir, so we need games/<filename>
            data_rel_path = f"games/{source.name}"
            pull_url = f"http://{coordinator_ip}:{P2P_DEFAULT_PORT}/files/data/{data_rel_path}"

            file_size = source.stat().st_size
            remote_dest = f"{games_path}/synced/{source.name}"

            logger.info(
                f"[AutoSyncDaemon] HTTP pull fallback: {target['node_id']} pulling "
                f"{source.name} ({file_size / 1024 / 1024:.0f}MB) from {pull_url}"
            )

            # Get SSH key from cluster config
            ssh_key = "~/.ssh/id_cluster"
            try:
                from app.config.cluster_config import get_cluster_nodes
                cluster_nodes = get_cluster_nodes()
                node_config = cluster_nodes.get(target["node_id"])
                if node_config and node_config.ssh_key:
                    ssh_key = node_config.ssh_key
            except ImportError:
                pass

            # Build SSH command to run curl on the remote node
            ssh_opts = [
                "-o", "StrictHostKeyChecking=no",
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=10",
            ]
            import os
            key_path = os.path.expanduser(ssh_key)
            if os.path.exists(key_path):
                ssh_opts.extend(["-i", key_path])

            # curl with: create dirs, timeout 30min, retry 2x, output to dest
            curl_cmd = (
                f"mkdir -p $(dirname '{remote_dest}') && "
                f"curl -sS --fail --retry 2 --max-time 1800 "
                f"-o '{remote_dest}' '{pull_url}'"
            )

            ssh_cmd = [
                "ssh", *ssh_opts,
                f"{ssh_user}@{target['host']}",
                curl_cmd,
            ]

            # Run in thread pool (blocking subprocess)
            def _run_pull():
                return subprocess.run(
                    ssh_cmd,
                    capture_output=True,
                    text=True,
                    timeout=1860,  # 31 minutes (slightly more than curl's 30min)
                )

            proc_result = await asyncio.get_running_loop().run_in_executor(
                None, _run_pull
            )

            if proc_result.returncode == 0:
                # Verify size on remote
                size_cmd = [
                    "ssh", *ssh_opts,
                    f"{ssh_user}@{target['host']}",
                    f"stat -c%s '{remote_dest}' 2>/dev/null || stat -f%z '{remote_dest}'",
                ]
                size_result = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: subprocess.run(
                        size_cmd, capture_output=True, text=True, timeout=30
                    ),
                )
                remote_size = 0
                try:
                    remote_size = int(size_result.stdout.strip())
                except (ValueError, AttributeError):
                    pass

                if remote_size == file_size:
                    duration = time.time() - start_time
                    logger.info(
                        f"[AutoSyncDaemon] HTTP pull succeeded: {source.name} -> "
                        f"{target['node_id']} ({file_size / 1024 / 1024:.0f}MB in {duration:.1f}s)"
                    )
                    return {
                        "source": str(source),
                        "target": target["node_id"],
                        "success": True,
                        "bytes_transferred": file_size,
                        "duration_seconds": duration,
                        "method": "http_pull",
                    }
                else:
                    error = f"Size mismatch after HTTP pull: local={file_size}, remote={remote_size}"
                    logger.warning(f"[AutoSyncDaemon] {error}")
                    return {
                        "source": str(source),
                        "target": target["node_id"],
                        "success": False,
                        "duration_seconds": time.time() - start_time,
                        "error": error,
                    }
            else:
                stderr = proc_result.stderr.strip()[:200] if proc_result.stderr else "Unknown"
                error = f"HTTP pull failed (rc={proc_result.returncode}): {stderr}"
                logger.warning(f"[AutoSyncDaemon] {error}")
                return {
                    "source": str(source),
                    "target": target["node_id"],
                    "success": False,
                    "duration_seconds": time.time() - start_time,
                    "error": error,
                }

        except subprocess.TimeoutExpired:
            return {
                "source": str(source),
                "target": target["node_id"],
                "success": False,
                "duration_seconds": time.time() - start_time,
                "error": "HTTP pull timed out (31 minutes)",
            }
        except Exception as e:
            logger.warning(f"[AutoSyncDaemon] HTTP pull error: {e}")
            return {
                "source": str(source),
                "target": target["node_id"],
                "success": False,
                "duration_seconds": time.time() - start_time,
                "error": f"HTTP pull error: {e}",
            }

    async def cleanup_stale_partials(self, max_age_hours: int = 24) -> int:
        """Remove stale .rsync-partial directories to prevent disk bloat.

        December 2025: Consolidated from cluster_data_sync.py

        Args:
            max_age_hours: Delete partial dirs older than this

        Returns:
            Number of files cleaned up
        """
        import datetime

        base_dir = Path(__file__).resolve().parent.parent.parent
        data_dir = base_dir / "data" / "games"
        partial_dir = data_dir / ".rsync-partial"

        cleaned = 0

        if partial_dir.exists():
            cutoff = datetime.datetime.now() - datetime.timedelta(hours=max_age_hours)

            for item in partial_dir.iterdir():
                try:
                    mtime = datetime.datetime.fromtimestamp(item.stat().st_mtime)
                    if mtime < cutoff and item.is_file():
                        item.unlink()
                        cleaned += 1
                        logger.debug(f"[AutoSyncDaemon] Cleaned stale partial: {item}")
                except OSError as e:
                    logger.debug(f"[AutoSyncDaemon] Error cleaning {item}: {e}")

        return cleaned

    async def broadcast_sync_cycle(self) -> int:
        """Execute one broadcast sync cycle (leader-only).

        December 2025: Consolidated from cluster_data_sync.py

        Returns:
            Number of successful syncs
        """
        if not self._is_broadcast:
            return 0

        logger.info("[AutoSyncDaemon] Starting broadcast sync cycle")

        # Clean up stale partial transfers periodically
        # Jan 2026: Use _sync_stats (SyncStats) not _stats (HandlerStats)
        if self._sync_stats.total_syncs % 10 == 0:
            try:
                cleaned = await self.cleanup_stale_partials()
                if cleaned > 0:
                    logger.info(f"[AutoSyncDaemon] Cleaned {cleaned} stale partial files")
            except OSError as e:
                logger.debug(f"[AutoSyncDaemon] Partial cleanup error: {e}")

        # Get eligible targets
        targets = await self.get_broadcast_targets()
        if not targets:
            logger.info("[AutoSyncDaemon] No broadcast targets available")
            return 0

        # Get databases to sync
        databases = self.discover_local_databases()
        if not databases:
            logger.info("[AutoSyncDaemon] No databases to sync")
            return 0

        # Sync each database to each target (with concurrency limit and retry)
        results: list[dict[str, Any]] = []
        semaphore = asyncio.Semaphore(self.config.max_concurrent_syncs)

        async def sync_with_limit(db: Path, target: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                # December 2025: Use retry wrapper for improved reliability
                return await self.sync_to_target_with_retry(db, target)

        # Create all sync tasks
        tasks = []
        for db in databases:
            for target in targets:
                tasks.append(sync_with_limit(db, target))

        # Execute concurrently
        if tasks:
            task_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions
            results = [
                r for r in task_results
                if isinstance(r, dict)
            ]

            # Log summary
            successful = sum(1 for r in results if r.get("success", False))
            failed = len(results) - successful
            logger.info(
                f"[AutoSyncDaemon] Broadcast sync complete: {successful} successful, {failed} failed"
            )

            return successful

        return 0

    # Note: _emit_sync_failure() and _emit_sync_stalled() are inherited from SyncMixinBase
