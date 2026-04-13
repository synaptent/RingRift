"""File delivery helpers for UnifiedDistributionDaemon."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

from app.coordination.distribution_shared import (
    CIRCUIT_BREAKER_AVAILABLE,
    REMOTE_PATH_PATTERNS,
    SSH_CONFIG_AVAILABLE,
    CircuitBreakerRegistry,
    DataType,
    DeliveryResult,
    _remote_path_cache,
    _remote_path_cache_lock,
    build_ssh_options,
    get_adaptive_timeout,
)

logger = logging.getLogger(__name__)


class DistributionDeliveryMixin:
    """Extracted helpers for UnifiedDistributionDaemon."""

    async def _distribute_file_to_target(
        self,
        file_path: Path,
        target: dict[str, Any] | str,
        data_type: DataType,
        use_bittorrent: bool,
        source_checksum: str | None,
    ) -> bool:
        """Distribute a single file to a single target node.

        December 29, 2025: Extracted from _run_smart_distribution for parallel execution.
        Tries transport methods in order: BitTorrent > HTTP > rsync

        Returns:
            True if distribution succeeded, False otherwise.
        """
        node_host = target if isinstance(target, str) else target.get("host", "")
        node_id = target if isinstance(target, str) else target.get("node_id", node_host)
        start_time = time.time()

        # December 29, 2025: Circuit breaker check - skip nodes that are failing
        if CIRCUIT_BREAKER_AVAILABLE and CircuitBreakerRegistry:
            try:
                registry = CircuitBreakerRegistry.get_instance()
                breaker = registry.get_breaker(f"distribution:{node_id}")
                if not breaker.can_execute(f"distribution:{node_id}"):
                    logger.debug(
                        f"Skipping distribution to {node_id}: circuit breaker open"
                    )
                    self._record_delivery(
                        node_id, node_host, str(file_path), data_type,
                        False, False, 0.0, "skipped", "Circuit breaker open"
                    )
                    return False
            except Exception as cb_err:
                logger.debug(f"Circuit breaker check failed: {cb_err}")

        # Try BitTorrent for large files
        if use_bittorrent and await self._distribute_via_bittorrent(file_path, target):
            self._record_delivery(
                node_id, node_host, str(file_path), data_type,
                True, True, time.time() - start_time, "bittorrent"
            )
            self._record_circuit_breaker_success(node_id)
            # January 2026: Register model in manifest after successful distribution
            self._register_model_in_manifest(file_path, node_id, data_type)
            return True

        # Try HTTP
        if self.config.use_http_distribution and await self._distribute_via_http(
            file_path, target, data_type
        ):
            # Verify checksum
            checksum_ok = True
            if source_checksum:
                checksum_ok = await self._verify_remote_checksum(
                    target, file_path, source_checksum, data_type
                )
            if checksum_ok:
                self._record_delivery(
                    node_id, node_host, str(file_path), data_type,
                    True, checksum_ok, time.time() - start_time, "http"
                )
                self._record_circuit_breaker_success(node_id)
                # January 2026: Register model in manifest after successful distribution
                self._register_model_in_manifest(file_path, node_id, data_type)
                return True

        # Fallback to rsync
        if self.config.fallback_to_rsync and await self._distribute_via_rsync(
            file_path, target, data_type
        ):
            checksum_ok = True
            if source_checksum:
                checksum_ok = await self._verify_remote_checksum(
                    target, file_path, source_checksum, data_type
                )
            if checksum_ok:
                self._record_delivery(
                    node_id, node_host, str(file_path), data_type,
                    True, checksum_ok, time.time() - start_time, "rsync"
                )
                self._record_circuit_breaker_success(node_id)
                # January 2026: Register model in manifest after successful distribution
                self._register_model_in_manifest(file_path, node_id, data_type)
                return True

        # All methods failed
        self._record_delivery(
            node_id, node_host, str(file_path), data_type,
            False, False, time.time() - start_time, "none", "All transport methods failed"
        )

        # December 29, 2025: Record failure in circuit breaker
        if CIRCUIT_BREAKER_AVAILABLE and CircuitBreakerRegistry:
            try:
                registry = CircuitBreakerRegistry.get_instance()
                breaker = registry.get_breaker(f"distribution:{node_id}")
                breaker.record_failure(f"distribution:{node_id}")
            except Exception as cb_err:
                logger.debug(f"Circuit breaker failure recording failed: {cb_err}")

        return False

    async def _distribute_via_rsync(
        self, file_path: Path, target: dict[str, Any] | str, data_type: DataType
    ) -> bool:
        """Distribute file via rsync.

        December 27, 2025: Added external storage routing.
        Nodes with use_external_storage: true get files routed to their
        configured storage_paths (e.g., mac-studio -> /Volumes/RingRift-Data).

        December 28, 2025: Added remote path discovery fallback.
        When no explicit remote_path is provided, probes the node to discover
        the correct path from REMOTE_PATH_PATTERNS.
        """
        # Get node_name before path resolution for external storage lookup
        node_name = target.get("node_id") if isinstance(target, dict) else None

        # December 28, 2025: Use path discovery with fallback
        host, user, remote_path, ssh_key = await self._get_remote_path(target)

        # December 27, 2025: Check for external storage routing
        # Nodes with use_external_storage get files routed to OWC/external drives
        external_dest = self._get_external_storage_dest(
            host, node_name, data_type, user, remote_path
        )

        if external_dest:
            remote_dest = external_dest
        elif data_type == DataType.MODEL:
            remote_dest = f"{user}@{host}:{remote_path}/models/"
        else:
            remote_dest = f"{user}@{host}:{remote_path}/data/training/"

        # Dec 30, 2025: Use centralized SSH config for consistent timeouts
        if SSH_CONFIG_AVAILABLE and build_ssh_options:
            ssh_opts_str = build_ssh_options(
                key_path=ssh_key,
                include_keepalive=False,  # rsync has its own timeout
            )
        else:
            ssh_opts = f"-i {ssh_key}" if ssh_key else ""
            ssh_opts_str = f"ssh {ssh_opts} -o StrictHostKeyChecking=no -o ConnectTimeout=10"
        cmd = [
            "rsync", "-avz",
            "-e", ssh_opts_str,
            str(file_path),
            remote_dest,
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.sync_timeout_seconds,
            )

            if process.returncode == 0:
                logger.debug(f"rsync to {host} succeeded: {file_path.name}")
                return True
            logger.debug(f"rsync to {host} failed: {stderr.decode()[:100]}")
            return False

        except (OSError, asyncio.TimeoutError, subprocess.SubprocessError) as e:
            logger.debug(f"rsync to {host} failed: {e}")
            return False

    async def _distribute_via_bittorrent(
        self, file_path: Path, target: dict[str, Any] | str
    ) -> bool:
        """Distribute file via BitTorrent.

        Phase 5 Step 9: Adds S3 web seeds to torrent files so peers can
        fall back to HTTP download from S3 when BitTorrent swarm is thin.
        """
        try:
            from app.distributed.aria2_transport import Aria2Config, Aria2Transport

            transport = Aria2Transport(Aria2Config(enable_bittorrent=True))

            # Phase 5 Step 9: Add S3 web seed for hybrid HTTP+BT downloads.
            # Determine the S3 key based on file type (model vs training data).
            web_seeds = []
            bucket = self.config.s3_bucket
            if bucket:
                file_name = file_path.name
                if file_name.endswith(".npz"):
                    # NPZ files stored under consolidated/training/
                    # Derive config_key from filename (e.g., hex8_2p.npz -> hex8_2p)
                    config_key = file_name.replace(".npz", "")
                    s3_url = (
                        f"https://{bucket}.s3.amazonaws.com/"
                        f"consolidated/training/{config_key}.npz"
                    )
                    web_seeds.append(s3_url)
                elif file_name.endswith(".pth"):
                    s3_url = (
                        f"https://{bucket}.s3.amazonaws.com/"
                        f"consolidated/models/{file_name}"
                    )
                    web_seeds.append(s3_url)

            torrent_path, info_hash, _error = await transport.create_and_register_torrent(
                file_path, web_seeds=web_seeds if web_seeds else None
            )

            if not info_hash:
                return False

            # Start seeding
            await transport.seed_file(file_path, torrent_path, duration_seconds=600)
            await transport.close()
            return True

        except ImportError:
            return False
        except (OSError, asyncio.TimeoutError, ConnectionError, RuntimeError) as e:
            logger.debug(f"BitTorrent distribution failed: {e}")
            return False

    def _get_external_storage_dest(
        self,
        host: str,
        node_name: str | None,
        data_type: DataType,
        user: str,
        fallback_path: str,
    ) -> str | None:
        """Get external storage destination for nodes with use_external_storage.

        December 27, 2025: Supports routing to OWC/external drives on coordinator
        nodes like mac-studio.

        Args:
            host: Target host IP or hostname
            node_name: Node name from cluster config (optional)
            data_type: Type of data being distributed
            user: SSH user
            fallback_path: Default ringrift path if no external storage

        Returns:
            Full rsync destination path (user@host:path/) or None if no external storage
        """
        try:
            from app.config.cluster_config import get_cluster_nodes

            nodes = get_cluster_nodes()

            # Find node by name or IP
            target_node = None
            if node_name and node_name in nodes:
                target_node = nodes[node_name]
            else:
                # Try to match by IP
                for n in nodes.values():
                    if n.best_ip == host or n.ssh_host == host:
                        target_node = n
                        break

            if not target_node:
                return None

            # Check if external storage is configured
            if not target_node.use_external_storage:
                return None

            # Get storage path for this data type
            storage_type = "models" if data_type == DataType.MODEL else "training_data"
            storage_path = target_node.get_storage_path(storage_type)

            if not storage_path:
                return None

            # Use node's configured user if available
            actual_user = target_node.ssh_user or user
            actual_host = target_node.best_ip or host

            logger.debug(
                f"[UnifiedDistributionDaemon] Routing {data_type.name} to "
                f"external storage: {actual_host}:{storage_path}"
            )

            return f"{actual_user}@{actual_host}:{storage_path}/"

        except (ImportError, KeyError, AttributeError, TypeError) as e:
            logger.debug(f"External storage lookup failed: {e}")
            return None

    async def _discover_remote_path(
        self,
        host: str,
        user: str = "root",
        ssh_key: str | None = None,
    ) -> str:
        """Discover the correct remote path for ai-service on a node.

        Probes the remote node to find which path pattern exists. Results are
        cached per host for efficiency.

        Args:
            host: Remote host IP or hostname
            user: SSH user (default: root)
            ssh_key: Optional SSH key path

        Returns:
            The discovered remote path, or default "~/ringrift/ai-service" if
            no path is found or all probes fail.
        """
        # Check cache first
        with _remote_path_cache_lock:
            if host in _remote_path_cache:
                return _remote_path_cache[host]

        # Build SSH options
        ssh_opts = ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10"]
        if ssh_key:
            ssh_opts.extend(["-i", ssh_key])

        # Probe each path pattern
        for path_pattern in REMOTE_PATH_PATTERNS:
            # Expand ~ for the test command (shell will expand it)
            test_cmd = f"test -d {path_pattern} && echo exists"

            cmd = ["ssh"] + ssh_opts + [f"{user}@{host}", test_cmd]

            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(
                    process.communicate(),
                    timeout=15.0,  # Quick timeout for path probing
                )

                if process.returncode == 0 and b"exists" in stdout:
                    logger.debug(
                        f"[UnifiedDistributionDaemon] Discovered remote path "
                        f"for {host}: {path_pattern}"
                    )
                    # Cache the result
                    with _remote_path_cache_lock:
                        _remote_path_cache[host] = path_pattern
                    return path_pattern

            except (OSError, asyncio.TimeoutError, subprocess.SubprocessError) as e:
                logger.debug(
                    f"[UnifiedDistributionDaemon] Path probe failed for "
                    f"{host}:{path_pattern}: {e}"
                )
                continue

        # Default fallback if no path found
        default_path = "~/ringrift/ai-service"
        logger.warning(
            f"[UnifiedDistributionDaemon] No valid remote path found for {host}, "
            f"using default: {default_path}"
        )
        with _remote_path_cache_lock:
            _remote_path_cache[host] = default_path
        return default_path

    async def _get_training_nodes(self) -> list[dict[str, Any]]:
        """Get list of training-capable nodes.

        December 2025: Fallback migrated to use cluster_config helpers instead of inline YAML.
        December 29, 2025: Fixed bug - convert SyncCandidateNode objects to dicts.
        get_sync_targets() returns SyncCandidateNode but callers expect dicts with
        'node_id', 'host', 'user' keys.
        """
        try:
            from app.config.cluster_config import get_cluster_nodes
            from app.coordination.sync_router import (
                DataType as SRDataType,
                get_sync_router,
            )
            router = get_sync_router()
            sync_targets = router.get_sync_targets(SRDataType.NPZ)

            # December 29, 2025: Convert SyncCandidateNode to dicts with SSH info
            # SyncCandidateNode only has node_id, priority, reason, capacity
            # We need to look up host/user from cluster_config
            cluster_nodes = get_cluster_nodes()
            result = []
            for target in sync_targets:
                node_config = cluster_nodes.get(target.node_id)
                if node_config:
                    result.append({
                        "node_id": target.node_id,
                        "host": node_config.best_ip,
                        "user": node_config.ssh_user,
                    })
            return result
        except ImportError:
            # Fallback to cluster_config helpers (Dec 2025)
            try:
                from app.config.cluster_config import get_cluster_nodes

                nodes = []
                for node_id, node in get_cluster_nodes().items():
                    # Dec 28, 2025: Fixed critical bug - check training_enabled flag
                    # instead of just role. GH200 nodes have role="gpu_training_primary"
                    # but training_enabled=true, and were being silently excluded.
                    is_training_node = getattr(node, "training_enabled", False)
                    has_training_role = node.role in (
                        "training",
                        "selfplay",
                        "gpu_training_primary",
                        "nn_training_primary",
                    )
                    if is_training_node or has_training_role:
                        nodes.append({
                            "node_id": node_id,
                            "host": node.best_ip,
                            "user": node.ssh_user,
                        })
                return nodes
            except (ImportError, OSError, KeyError, TypeError):
                return []

    async def _create_remote_symlinks(
        self,
        model_paths: list[Path],
        target_nodes: list[str],
    ) -> None:
        """Create ringrift_best_*.pth symlinks on remote nodes.

        December 2025: Harvested from deprecated model_distribution_daemon.py.
        After distributing canonical models, create corresponding symlinks on
        each target node so selfplay nodes can find models by config key.

        December 28, 2025: Uses remote path discovery for each node.
        """
        if not model_paths or not target_nodes:
            return

        # Collect symlink info (path-independent)
        symlink_info: list[tuple[str, str]] = []  # (canonical_name, symlink_name)
        for path in model_paths:
            if not path.stem.startswith("canonical_"):
                continue
            config_key = path.stem[len("canonical_"):]
            canonical_name = path.name
            symlink_name = f"ringrift_best_{config_key}.pth"
            symlink_info.append((canonical_name, symlink_name))

        if not symlink_info:
            return

        created_count = 0
        failed_nodes: list[str] = []

        try:
            from app.core.ssh import get_ssh_client

            async def create_on_node(host: str) -> bool:
                """Create symlinks for canonical models on a remote cluster node.

                Creates ringrift_best_* symlinks pointing to canonical model files,
                enabling standard model loading paths across all cluster nodes.

                Args:
                    host: Hostname or IP of target node (e.g., 'nebius-h100-1')

                Returns:
                    True if symlinks created successfully, False on any error

                Raises:
                    None - all exceptions are caught and return False

                SSH Commands:
                    cd {remote_path}/models && rm -f {symlink} && ln -sf {canonical} {symlink}
                """
                try:
                    # December 28, 2025: Discover remote path for this node
                    remote_path = await self._discover_remote_path(host)

                    # Build symlink commands with discovered path
                    symlink_cmds = []
                    for canonical_name, symlink_name in symlink_info:
                        # Create relative symlink in models directory
                        # rm -f to avoid "file exists" errors, -sf for force symlink
                        symlink_cmds.append(
                            f"cd {remote_path}/models && "
                            f"rm -f {symlink_name} && "
                            f"ln -sf {canonical_name} {symlink_name}"
                        )

                    combined_cmd = " && ".join(symlink_cmds)
                    client = get_ssh_client(host)
                    # Sprint 10 (Jan 3, 2026): Use adaptive timeout based on host history
                    adaptive_timeout = get_adaptive_timeout("ssh", host, default=30.0)
                    result = await client.run_async(combined_cmd, timeout=adaptive_timeout)
                    return result.success
                except (OSError, asyncio.TimeoutError, RuntimeError) as e:
                    logger.debug(f"Failed to create symlinks on {host}: {e}")
                    return False

            tasks = [create_on_node(node) for node in target_nodes]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, (node, result) in enumerate(zip(target_nodes, results)):
                if isinstance(result, Exception):
                    failed_nodes.append(node)
                elif result:
                    created_count += 1
                else:
                    failed_nodes.append(node)

        except ImportError:
            logger.debug("SSH client not available, skipping remote symlinks")
            return

        if created_count > 0:
            logger.info(
                f"Created remote symlinks on {created_count}/{len(target_nodes)} nodes"
            )
        if failed_nodes:
            logger.debug(
                f"Failed to create symlinks on {len(failed_nodes)} nodes: "
                f"{failed_nodes[:3]}{'...' if len(failed_nodes) > 3 else ''}"
            )

    def _record_delivery(
        self,
        node_id: str,
        host: str,
        path: str,
        data_type: DataType,
        success: bool,
        checksum_ok: bool,
        time_seconds: float,
        method: str,
        error: str = "",
    ) -> None:
        """Record delivery result in history and persistent ledger."""
        result = DeliveryResult(
            node_id=node_id,
            host=host,
            data_path=path,
            data_type=data_type,
            success=success,
            checksum_verified=checksum_ok,
            transfer_time_seconds=time_seconds,
            method=method,
            error_message=error,
        )
        self._delivery_history.append(result)
        if len(self._delivery_history) > 200:
            self._delivery_history = self._delivery_history[-200:]

        # Persist to delivery ledger (Dec 2025 Phase 3)
        if self._delivery_ledger is not None:
            try:
                # Map DataType to ledger data_type string
                data_type_str = data_type.value if hasattr(data_type, 'value') else str(data_type)

                # Record to ledger
                record = self._delivery_ledger.record_delivery_started(
                    data_type=data_type_str,
                    data_path=path,
                    target_node=node_id,
                )

                if success and checksum_ok:
                    # Calculate checksum for verified delivery
                    checksum = self._checksum_cache.get(path, "")
                    self._delivery_ledger.record_delivery_transferred(
                        delivery_id=record.delivery_id,
                        checksum=checksum,
                    )
                    self._delivery_ledger.record_delivery_verified(record.delivery_id)
                elif success and not checksum_ok:
                    # Transferred but checksum failed
                    self._delivery_ledger.record_delivery_transferred(
                        delivery_id=record.delivery_id,
                        checksum="",
                    )
                    self._delivery_ledger.record_delivery_failed(
                        record.delivery_id,
                        "Checksum verification failed",
                    )
                else:
                    # Failed to transfer
                    self._delivery_ledger.record_delivery_failed(
                        record.delivery_id,
                        error or "Transfer failed",
                    )

                    # Enqueue for retry if eligible
                    if self._retry_queue is not None:
                        updated = self._delivery_ledger.get_delivery(record.delivery_id)
                        if updated and updated.can_retry:
                            self._retry_queue.enqueue_retry(updated)
                            logger.debug(
                                f"[UnifiedDistributionDaemon] Enqueued {record.delivery_id[:8]} for retry"
                            )

            except Exception as e:  # noqa: BLE001
                logger.debug(f"[UnifiedDistributionDaemon] Failed to record to ledger: {e}")
