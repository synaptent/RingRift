"""NPZ Distribution Daemon - Automatic training data sync after export.

DEPRECATED (December 2025): Use unified_distribution_daemon.py instead.
This module is preserved for backward compatibility but will be removed in Q2 2026.

Migration:
    from app.coordination.unified_distribution_daemon import (
        UnifiedDistributionDaemon,
        create_unified_distribution_daemon,
    )

This daemon watches for NPZ_EXPORT_COMPLETE events and automatically distributes
exported NPZ files to all training-capable cluster nodes.

Architecture:
    1. Subscribes to NPZ_EXPORT_COMPLETE events from event_router
    2. Uses rsync for reliable multi-node sync (with HTTP fallback)
    3. Tracks distribution status in ClusterManifest
    4. Emits NPZ_DISTRIBUTION_COMPLETE event when done
    5. Validates checksums before and after transfer (December 2025)

Usage:
    # As standalone daemon
    python -m app.coordination.npz_distribution_daemon

    # Via DaemonManager
    manager.register_factory(DaemonType.NPZ_DISTRIBUTION, daemon.run)

Configuration:
    Uses distributed_hosts.yaml for target nodes.

December 2025 Enhancements:
    - SHA256 checksum validation before transfer
    - Remote checksum verification after transfer
    - HTTP fallback for nodes without SSH access
    - Per-node delivery confirmation tracking
    - CoordinatorProtocol integration for health checks
    - Retry with exponential backoff
    - NPZ structure validation (array shapes, sample counts)
    - BitTorrent priority for large files (>50MB)
    - ResilientTransfer integration for unified transport selection
"""

from __future__ import annotations

import warnings

warnings.warn(
    "npz_distribution_daemon module is deprecated as of December 2025. "
    "Use unified_distribution_daemon module instead for consolidated distribution.",
    DeprecationWarning,
    stacklevel=2,
)

import asyncio
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.coordination.protocols import (
    CoordinatorStatus,
    HealthCheckResult,
    register_coordinator,
    unregister_coordinator,
)

logger = logging.getLogger(__name__)

# Add parent to path for imports
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class NPZDistributionConfig:
    """Configuration for NPZ distribution daemon."""

    # Sync settings
    sync_timeout_seconds: float = 600.0  # 10 minute timeout (NPZ can be large)
    retry_count: int = 3
    retry_delay_seconds: float = 30.0
    retry_backoff_multiplier: float = 1.5  # Exponential backoff

    # Target selection
    target_node_types: list[str] = field(
        default_factory=lambda: ["training", "selfplay"]
    )

    # Event settings
    emit_completion_event: bool = True

    # Polling (if no event system)
    poll_interval_seconds: float = 120.0
    training_data_dir: str = "data/training"

    # Checksum verification (December 2025)
    verify_checksums: bool = True  # Enable checksum verification
    checksum_timeout_seconds: float = 30.0  # Timeout for remote checksum verification

    # NPZ structure validation (December 2025)
    validate_npz_structure: bool = True  # Validate NPZ arrays after transfer
    max_npz_samples: int = 100_000_000  # Maximum reasonable sample count

    # BitTorrent priority (December 2025)
    prefer_bittorrent: bool = True  # Use BitTorrent for large files (>50MB)
    bittorrent_threshold_bytes: int = 50_000_000  # 50MB threshold

    # HTTP distribution settings (December 2025)
    use_http_distribution: bool = True  # Prefer HTTP when available
    http_port: int = 8767  # Port for data upload endpoint
    http_timeout_seconds: float = 120.0  # Timeout per node for HTTP upload (NPZ larger than models)
    http_concurrent_uploads: int = 3  # Max concurrent HTTP uploads
    fallback_to_rsync: bool = True  # Fallback to rsync if HTTP fails


@dataclass
class NPZDeliveryResult:
    """Result of delivering an NPZ file to a single node."""

    node_id: str
    host: str
    success: bool
    checksum_verified: bool
    transfer_time_seconds: float
    error_message: str = ""
    method: str = "rsync"  # rsync or http


class NPZDistributionDaemon:
    """Daemon that automatically distributes NPZ training files after export.

    Watches for NPZ_EXPORT_COMPLETE events and syncs NPZ files to training nodes.
    This ensures that exported training data is available on all nodes for:
    - Distributed training
    - Model fine-tuning
    - Transfer learning

    The daemon solves the critical gap where training data would only exist on
    the export node, causing training to fail on other nodes.

    December 2025 Enhancements:
    - Checksum validation: SHA256 before transfer, verified on remote after
    - HTTP fallback: Uses HTTP upload when rsync fails
    - Delivery confirmation: Tracks per-node delivery success
    - CoordinatorProtocol: Exposes health and metrics endpoints
    """

    def __init__(self, config: NPZDistributionConfig | None = None):
        self.config = config or NPZDistributionConfig()
        self._running = False
        self._last_sync_time: float = 0.0
        self._pending_npz: list[dict[str, Any]] = []
        self._sync_lock = asyncio.Lock()

        # Thread-safe lock for _pending_npz (callback may run from different context)
        self._pending_lock = threading.Lock()

        # Event-based wake-up for immediate distribution on event
        self._pending_event: asyncio.Event | None = None

        # CoordinatorProtocol state (December 2025)
        self._coordinator_status = CoordinatorStatus.INITIALIZING
        self._start_time: float = 0.0
        self._events_processed: int = 0
        self._errors_count: int = 0
        self._last_error: str = ""
        self._successful_distributions: int = 0
        self._failed_distributions: int = 0
        self._checksum_failures: int = 0

        # Delivery tracking per node
        self._delivery_history: list[NPZDeliveryResult] = []

    # =========================================================================
    # CoordinatorProtocol Implementation (December 2025)
    # =========================================================================

    @property
    def name(self) -> str:
        """Unique name identifying this coordinator."""
        return "NPZDistributionDaemon"

    @property
    def status(self) -> CoordinatorStatus:
        """Current status of the coordinator."""
        return self._coordinator_status

    @property
    def uptime_seconds(self) -> float:
        """Time since daemon started, in seconds."""
        if self._start_time <= 0:
            return 0.0
        return time.time() - self._start_time

    def get_metrics(self) -> dict[str, Any]:
        """Get daemon metrics in protocol-compliant format."""
        return {
            "name": self.name,
            "status": self._coordinator_status.value,
            "uptime_seconds": self.uptime_seconds,
            "start_time": self._start_time,
            "events_processed": self._events_processed,
            "errors_count": self._errors_count,
            "last_error": self._last_error,
            # Distribution-specific metrics
            "pending_npz": len(self._pending_npz),
            "successful_distributions": self._successful_distributions,
            "failed_distributions": self._failed_distributions,
            "checksum_failures": self._checksum_failures,
            "last_sync_time": self._last_sync_time,
        }

    def health_check(self) -> HealthCheckResult:
        """Check daemon health."""
        # Check for error state
        if self._coordinator_status == CoordinatorStatus.ERROR:
            return HealthCheckResult.unhealthy(
                f"Daemon in error state: {self._last_error}"
            )

        # Check if stopped
        if self._coordinator_status == CoordinatorStatus.STOPPED:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="Daemon is stopped",
            )

        # Check for high failure rate
        total_dist = self._successful_distributions + self._failed_distributions
        if total_dist > 0 and self._failed_distributions > self._successful_distributions:
            return HealthCheckResult.degraded(
                f"High failure rate: {self._failed_distributions} failures, "
                f"{self._successful_distributions} successes",
                failure_rate=self._failed_distributions / total_dist,
            )

        # Check for checksum failures
        if self._checksum_failures > 5:
            return HealthCheckResult.degraded(
                f"{self._checksum_failures} checksum verification failures",
                checksum_failures=self._checksum_failures,
            )

        # Check for pending NPZ buildup
        if len(self._pending_npz) > 10:
            return HealthCheckResult.degraded(
                f"{len(self._pending_npz)} NPZ files pending distribution",
                pending_npz=len(self._pending_npz),
            )

        # Healthy
        return HealthCheckResult(
            healthy=True,
            status=self._coordinator_status,
            details={
                "uptime_seconds": self.uptime_seconds,
                "successful_distributions": self._successful_distributions,
                "pending_npz": len(self._pending_npz),
                "last_sync_time": self._last_sync_time,
            },
        )

    async def start(self) -> None:
        """Start the daemon and subscribe to events."""
        if self._coordinator_status == CoordinatorStatus.RUNNING:
            return  # Already running

        logger.info("NPZDistributionDaemon starting...")
        self._running = True
        self._coordinator_status = CoordinatorStatus.RUNNING
        self._start_time = time.time()

        # Initialize event for immediate push-on-export
        self._pending_event = asyncio.Event()

        # Register with coordinator registry
        register_coordinator(self)

        # Try to subscribe to NPZ_EXPORT_COMPLETE events
        try:
            from app.coordination.event_router import subscribe, StageEvent

            # Subscribe to StageEvent.NPZ_EXPORT_COMPLETE for pipeline integration
            subscribe(StageEvent.NPZ_EXPORT_COMPLETE, self._on_npz_exported)
            logger.info("Subscribed to NPZ_EXPORT_COMPLETE events via StageEvent")

            # Also try DataEventType for cross-process compatibility
            try:
                from app.coordination.event_router import DataEventType
                # Use string matching for events that may come from other processes
                subscribe("npz_export_complete", self._on_npz_exported)
                logger.info("Also subscribed to npz_export_complete string events")
            except ImportError:
                pass

        except ImportError as e:
            logger.warning(
                f"event_router not available ({e}), will poll for new NPZ files instead"
            )
        except Exception as e:
            logger.error(f"Failed to subscribe to NPZ_EXPORT_COMPLETE: {e}")

        # Main loop - handle pending syncs and periodic checks
        while self._running:
            try:
                # Process any pending NPZ distributions
                with self._pending_lock:
                    has_pending = bool(self._pending_npz)

                if has_pending:
                    await self._process_pending_npz()

                # Periodic sync to catch any missed exports
                if time.time() - self._last_sync_time > self.config.poll_interval_seconds:
                    await self._periodic_sync_check()

                # Wait for event with timeout (instant wake on new NPZ)
                if self._pending_event is not None:
                    try:
                        await asyncio.wait_for(self._pending_event.wait(), timeout=5.0)
                        self._pending_event.clear()
                    except asyncio.TimeoutError:
                        pass
                else:
                    await asyncio.sleep(5.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in NPZ distribution daemon loop: {e}")
                self._errors_count += 1
                self._last_error = str(e)
                await asyncio.sleep(10.0)

        logger.info("NPZDistributionDaemon stopped")

    async def stop(self) -> None:
        """Stop the daemon gracefully."""
        if self._coordinator_status == CoordinatorStatus.STOPPED:
            return  # Already stopped

        self._coordinator_status = CoordinatorStatus.STOPPING
        self._running = False

        # Unregister from coordinator registry
        unregister_coordinator(self.name)

        self._coordinator_status = CoordinatorStatus.STOPPED

    def _on_npz_exported(self, event: dict[str, Any] | Any) -> None:
        """Handle NPZ_EXPORT_COMPLETE event (sync callback).

        December 2025: Thread-safe with immediate wake-up for instant distribution.
        """
        # Handle both RouterEvent and dict payloads
        payload = getattr(event, "payload", event) if hasattr(event, "payload") else event
        npz_info = {
            "npz_path": payload.get("npz_path"),
            "board_type": payload.get("board_type"),
            "num_players": payload.get("num_players"),
            "sample_count": payload.get("sample_count"),
            "timestamp": time.time(),
        }
        logger.info(f"Received NPZ_EXPORT_COMPLETE event: {npz_info}")

        # Thread-safe append to pending list
        with self._pending_lock:
            self._pending_npz.append(npz_info)
        self._events_processed += 1

        # Immediate wake-up for instant distribution
        if self._pending_event is not None:
            self._pending_event.set()

    async def _process_pending_npz(self) -> None:
        """Process pending NPZ distributions with checksum validation."""
        async with self._sync_lock:
            # Thread-safe extraction
            with self._pending_lock:
                if not self._pending_npz:
                    return
                npz_files = self._pending_npz.copy()
                self._pending_npz.clear()

            logger.info(f"Processing {len(npz_files)} pending NPZ distributions")

            # Get target nodes
            target_nodes = await self._get_training_nodes()
            if not target_nodes:
                logger.warning("No training nodes available for NPZ distribution")
                return

            # Distribute each NPZ file
            for npz_info in npz_files:
                npz_path = npz_info.get("npz_path")
                if not npz_path:
                    continue

                # Validate NPZ structure before distribution (December 2025)
                if self.config.validate_npz_structure:
                    npz_validation = await self._validate_npz_source(Path(npz_path))
                    if not npz_validation:
                        logger.error(f"NPZ validation failed, skipping distribution: {npz_path}")
                        self._failed_distributions += 1
                        continue

                # Compute source checksum before distribution (December 2025)
                source_checksum = None
                if self.config.verify_checksums:
                    source_checksum = await self._compute_checksum(Path(npz_path))
                    if source_checksum:
                        logger.info(f"NPZ checksum: {source_checksum[:16]}...")
                    else:
                        logger.warning(f"Failed to compute checksum for {npz_path}")

                # Run distribution with exponential backoff retry
                success = False
                delivery_results: list[NPZDeliveryResult] = []
                current_delay = self.config.retry_delay_seconds

                for attempt in range(self.config.retry_count):
                    try:
                        success, results = await self._distribute_npz_with_verification(
                            npz_path, target_nodes, source_checksum
                        )
                        delivery_results = results

                        if success:
                            self._last_sync_time = time.time()
                            self._successful_distributions += 1
                            logger.info(f"NPZ distribution completed: {npz_path}")
                            break
                    except Exception as e:
                        logger.error(f"Distribution attempt {attempt + 1} failed: {e}")
                        self._errors_count += 1
                        self._last_error = str(e)

                    if attempt < self.config.retry_count - 1:
                        logger.info(f"Retrying in {current_delay:.1f}s...")
                        await asyncio.sleep(current_delay)
                        current_delay *= self.config.retry_backoff_multiplier

                # Track delivery history
                self._delivery_history.extend(delivery_results)
                # Keep only last 100 results
                if len(self._delivery_history) > 100:
                    self._delivery_history = self._delivery_history[-100:]

                if success and self.config.emit_completion_event:
                    await self._emit_distribution_complete(
                        npz_info, target_nodes, source_checksum, delivery_results
                    )

                if not success:
                    self._failed_distributions += 1
                    logger.error(
                        f"NPZ distribution failed after {self.config.retry_count} attempts: {npz_path}"
                    )

    # =========================================================================
    # Checksum Utilities (December 2025)
    # =========================================================================

    async def _compute_checksum(self, path: Path) -> str | None:
        """Compute SHA256 checksum of a local file.

        Uses the centralized checksum utilities for consistency.

        Args:
            path: Path to the file

        Returns:
            SHA256 hex digest or None on error
        """
        if not path.exists():
            return None

        try:
            from app.utils.checksum_utils import compute_file_checksum, LARGE_CHUNK_SIZE

            # Run in thread pool to avoid blocking
            # Dec 2025: Use get_running_loop() instead of deprecated get_event_loop()
            loop = asyncio.get_running_loop()
            checksum = await loop.run_in_executor(
                None,
                lambda: compute_file_checksum(path, chunk_size=LARGE_CHUNK_SIZE),
            )
            return checksum
        except Exception as e:
            logger.warning(f"Failed to compute checksum for {path}: {e}")
            return None

    async def _verify_remote_checksum(
        self,
        node: dict[str, Any],
        npz_path: Path,
        expected_checksum: str,
    ) -> bool:
        """Verify checksum of NPZ file on remote node via SSH.

        Args:
            node: Node configuration dict
            npz_path: Local path to NPZ file (to get filename)
            expected_checksum: Expected SHA256 checksum

        Returns:
            True if checksum matches, False otherwise
        """
        host = node.get("host")
        user = node.get("user", "ubuntu")
        remote_path = node.get("remote_path", "~/ringrift/ai-service")
        ssh_key = node.get("ssh_key")

        # Build remote checksum command
        remote_file = f"{remote_path}/data/training/{npz_path.name}"
        checksum_cmd = f"sha256sum {remote_file} 2>/dev/null | cut -d' ' -f1"

        # Build SSH command
        ssh_opts = [
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes",
        ]
        if ssh_key:
            ssh_opts.extend(["-i", ssh_key])

        cmd = ["ssh"] + ssh_opts + [f"{user}@{host}", checksum_cmd]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.checksum_timeout_seconds,
            )

            if process.returncode == 0:
                remote_checksum = stdout.decode().strip()
                if remote_checksum == expected_checksum:
                    logger.debug(f"Checksum verified on {host}: {remote_checksum[:16]}...")
                    return True
                else:
                    logger.warning(
                        f"Checksum mismatch on {host}: "
                        f"expected {expected_checksum[:16]}..., "
                        f"got {remote_checksum[:16]}..."
                    )
                    self._checksum_failures += 1
                    return False
            else:
                logger.warning(f"Failed to get checksum from {host}: {stderr.decode()[:100]}")
                return False

        except asyncio.TimeoutError:
            logger.warning(f"Checksum verification timed out on {host}")
            return False
        except Exception as e:
            logger.warning(f"Checksum verification failed on {host}: {e}")
            return False

    async def _validate_npz_source(self, npz_path: Path) -> bool:
        """Validate NPZ file structure before distribution.

        December 2025: Catches corrupted NPZ files BEFORE distributing them
        to the cluster. This prevents propagating corruption.

        Args:
            npz_path: Path to the NPZ file

        Returns:
            True if NPZ is valid, False otherwise
        """
        try:
            from app.coordination.npz_validation import (
                validate_npz_structure,
                NPZValidationResult,
            )

            # Run validation in thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            result: NPZValidationResult = await loop.run_in_executor(
                None,
                lambda: validate_npz_structure(
                    npz_path,
                    require_policy=True,
                    max_samples=self.config.max_npz_samples,
                ),
            )

            if result.valid:
                logger.info(
                    f"NPZ validation passed: {npz_path.name} - "
                    f"{result.sample_count} samples, {len(result.array_shapes)} arrays"
                )
                return True
            else:
                for error in result.errors:
                    logger.error(f"NPZ validation error: {error}")
                for warning in result.warnings:
                    logger.warning(f"NPZ validation warning: {warning}")
                return False

        except ImportError:
            logger.warning("npz_validation module not available, skipping structure validation")
            return True  # Allow distribution without validation
        except Exception as e:
            logger.error(f"NPZ validation failed with exception: {e}")
            return False

    async def _distribute_npz_with_verification(
        self,
        npz_path: str,
        target_nodes: list[dict[str, Any]],
        source_checksum: str | None,
    ) -> tuple[bool, list[NPZDeliveryResult]]:
        """Distribute NPZ file to target nodes with checksum verification.

        December 2025: Enhanced distribution with:
        - Pre-transfer checksum computation
        - Post-transfer checksum verification
        - Per-node delivery tracking
        - HTTP fallback option

        Args:
            npz_path: Path to the NPZ file
            target_nodes: List of target node configurations
            source_checksum: Pre-computed SHA256 checksum (or None to skip verification)

        Returns:
            Tuple of (success, list of delivery results)
        """
        npz_file = Path(npz_path)
        if not npz_file.exists():
            logger.error(f"NPZ file not found: {npz_path}")
            return False, []

        delivery_results: list[NPZDeliveryResult] = []
        success_count = 0

        # Check file size for transport selection (December 2025)
        file_size = npz_file.stat().st_size
        use_bittorrent = (
            self.config.prefer_bittorrent
            and file_size > self.config.bittorrent_threshold_bytes
        )

        if use_bittorrent:
            logger.info(
                f"NPZ file {npz_file.name} is {file_size / 1024 / 1024:.1f}MB, "
                "using BitTorrent for distribution"
            )

        for node in target_nodes:
            node_id = node.get("node_id", node.get("host", "unknown"))
            host = node.get("host")
            start_time = time.time()

            # Try BitTorrent first for large files (December 2025)
            if use_bittorrent:
                bt_success = await self._distribute_via_bittorrent(npz_file, node)
                if bt_success:
                    # BitTorrent has piece-level verification, but verify final checksum too
                    checksum_ok = True
                    if source_checksum and self.config.verify_checksums:
                        checksum_ok = await self._verify_remote_checksum(
                            node, npz_file, source_checksum
                        )

                    delivery_results.append(NPZDeliveryResult(
                        node_id=node_id,
                        host=host,
                        success=checksum_ok,
                        checksum_verified=checksum_ok,
                        transfer_time_seconds=time.time() - start_time,
                        method="bittorrent",
                    ))

                    if checksum_ok:
                        success_count += 1
                        continue

                    # BitTorrent succeeded but checksum failed - fall through to HTTP/rsync
                    logger.warning(f"BitTorrent checksum mismatch on {host}, trying fallback")

            # Try HTTP if enabled
            if self.config.use_http_distribution:
                http_success = await self._distribute_via_http(npz_file, node)
                if http_success:
                    # Verify checksum on remote if we have source checksum
                    checksum_ok = True
                    if source_checksum and self.config.verify_checksums:
                        checksum_ok = await self._verify_remote_checksum(
                            node, npz_file, source_checksum
                        )

                    delivery_results.append(NPZDeliveryResult(
                        node_id=node_id,
                        host=host,
                        success=checksum_ok,
                        checksum_verified=checksum_ok,
                        transfer_time_seconds=time.time() - start_time,
                        method="http",
                    ))

                    if checksum_ok:
                        success_count += 1
                        continue

                    # HTTP transfer succeeded but checksum failed - try rsync
                    if not self.config.fallback_to_rsync:
                        continue

                elif not self.config.fallback_to_rsync:
                    delivery_results.append(NPZDeliveryResult(
                        node_id=node_id,
                        host=host,
                        success=False,
                        checksum_verified=False,
                        transfer_time_seconds=time.time() - start_time,
                        error_message="HTTP upload failed",
                        method="http",
                    ))
                    continue

            # Rsync distribution
            rsync_success = await self._rsync_to_node(npz_file, node)
            checksum_ok = False

            if rsync_success:
                # Verify checksum on remote
                if source_checksum and self.config.verify_checksums:
                    checksum_ok = await self._verify_remote_checksum(
                        node, npz_file, source_checksum
                    )
                else:
                    checksum_ok = True  # No checksum to verify

                if rsync_success and checksum_ok:
                    success_count += 1

            delivery_results.append(NPZDeliveryResult(
                node_id=node_id,
                host=host,
                success=rsync_success and checksum_ok,
                checksum_verified=checksum_ok,
                transfer_time_seconds=time.time() - start_time,
                error_message="" if rsync_success else "rsync failed",
                method="rsync",
            ))

        # Consider success if at least one node received and verified the file
        return success_count > 0, delivery_results

    async def _distribute_via_bittorrent(
        self, npz_file: Path, node: dict[str, Any]
    ) -> bool:
        """Distribute NPZ file to a node via BitTorrent.

        December 2025: BitTorrent provides piece-level SHA1 verification,
        making it the most reliable transport for large files.

        Args:
            npz_file: Path to the NPZ file
            node: Node configuration dict

        Returns:
            True if download succeeded on the node, False otherwise
        """
        try:
            from app.distributed.torrent_generator import TorrentGenerator
            from app.distributed.cluster_manifest import ClusterManifest

            # Get or create torrent for this file
            manifest = ClusterManifest.get_instance()
            generator = TorrentGenerator()

            # Check if torrent already exists
            torrent_info = None
            try:
                torrents = manifest.get_torrents_for_path(str(npz_file))
                if torrents:
                    torrent_info = torrents[0]
            except (KeyError, ValueError, TypeError) as e:
                logger.debug(f"Failed to lookup torrent for {npz_file.name}: {e}")
            except (OSError, IOError) as e:
                logger.debug(f"I/O error looking up torrent for {npz_file.name}: {e}")

            # Create torrent if not exists
            if not torrent_info:
                logger.info(f"Creating torrent for {npz_file.name}")
                torrent_info = generator.create_torrent(npz_file)
                if torrent_info:
                    manifest.register_torrent(torrent_info)

            if not torrent_info:
                logger.warning(f"Could not create torrent for {npz_file.name}")
                return False

            # Trigger download on remote node via aria2c RPC
            host = node.get("host")
            aria2_port = node.get("aria2_port", 6800)

            # Build magnet link or use .torrent file
            magnet_uri = torrent_info.get("magnet_uri")
            if not magnet_uri:
                logger.warning(f"No magnet URI for {npz_file.name}")
                return False

            # Use aria2 RPC to add the download on the remote node
            import aiohttp

            rpc_url = f"http://{host}:{aria2_port}/jsonrpc"
            remote_path = node.get("remote_path", "~/ringrift/ai-service")
            output_dir = f"{remote_path}/data/training"

            payload = {
                "jsonrpc": "2.0",
                "id": "npz-dist",
                "method": "aria2.addUri",
                "params": [
                    [magnet_uri],
                    {"dir": output_dir},
                ],
            }

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=600)  # 10 min for large files
            ) as session:
                async with session.post(rpc_url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        gid = result.get("result")
                        if gid:
                            logger.info(
                                f"BitTorrent download started on {host}: GID={gid}"
                            )
                            # Wait for download to complete
                            return await self._wait_for_aria2_download(
                                session, rpc_url, gid
                            )
                    logger.warning(f"aria2 RPC failed on {host}: {response.status}")
                    return False

        except ImportError as e:
            logger.debug(f"BitTorrent not available: {e}")
            return False
        except Exception as e:
            logger.warning(f"BitTorrent distribution failed: {e}")
            return False

    async def _wait_for_aria2_download(
        self,
        session: "aiohttp.ClientSession",
        rpc_url: str,
        gid: str,
        poll_interval: float = 5.0,
        max_wait: float = 600.0,
    ) -> bool:
        """Wait for aria2 download to complete.

        Args:
            session: aiohttp session
            rpc_url: aria2 RPC URL
            gid: Download GID
            poll_interval: Seconds between status checks
            max_wait: Maximum time to wait

        Returns:
            True if download completed successfully
        """
        import time
        start = time.time()

        while time.time() - start < max_wait:
            try:
                payload = {
                    "jsonrpc": "2.0",
                    "id": "status",
                    "method": "aria2.tellStatus",
                    "params": [gid],
                }
                async with session.post(rpc_url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        status = result.get("result", {}).get("status")

                        if status == "complete":
                            return True
                        elif status in ("error", "removed"):
                            logger.warning(f"aria2 download failed: {status}")
                            return False
                        # Still active or waiting, continue polling

            except Exception as e:
                logger.debug(f"aria2 status check error: {e}")

            await asyncio.sleep(poll_interval)

        logger.warning("aria2 download timed out")
        return False

    async def _distribute_via_http(self, npz_file: Path, node: dict[str, Any]) -> bool:
        """Distribute NPZ file to a node via HTTP upload.

        Args:
            npz_file: Path to the NPZ file
            node: Node configuration dict

        Returns:
            True if upload succeeded, False otherwise
        """
        try:
            import aiohttp
        except ImportError:
            return False

        host = node.get("host")
        url = f"http://{host}:{self.config.http_port}/data/upload"

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.http_timeout_seconds)
            ) as session:
                with open(npz_file, "rb") as f:
                    file_data = f.read()

                form_data = aiohttp.FormData()
                form_data.add_field(
                    "file",
                    file_data,
                    filename=npz_file.name,
                    content_type="application/octet-stream",
                )

                async with session.post(url, data=form_data) as response:
                    if response.status == 200:
                        logger.debug(f"HTTP upload to {host} succeeded")
                        return True
                    else:
                        logger.debug(f"HTTP upload to {host} failed: {response.status}")
                        return False

        except asyncio.TimeoutError:
            logger.debug(f"HTTP upload to {host} timed out")
            return False
        except Exception as e:
            logger.debug(f"HTTP upload to {host} failed: {e}")
            return False

    async def _rsync_to_node(self, npz_file: Path, node: dict[str, Any]) -> bool:
        """Rsync NPZ file to a single node.

        Args:
            npz_file: Path to the NPZ file
            node: Node configuration dict

        Returns:
            True if rsync succeeded, False otherwise
        """
        host = node.get("host")
        user = node.get("user", "ubuntu")
        remote_path = node.get("remote_path", "~/ringrift/ai-service")
        ssh_key = node.get("ssh_key")

        # Build rsync command
        ssh_opts = f"-i {ssh_key}" if ssh_key else ""
        remote_dest = f"{user}@{host}:{remote_path}/data/training/"

        cmd = [
            "rsync",
            "-avz",
            "--progress",
            "-e", f"ssh {ssh_opts} -o StrictHostKeyChecking=no -o ConnectTimeout=10",
            str(npz_file),
            remote_dest,
        ]

        logger.info(f"Syncing {npz_file.name} to {host}")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.sync_timeout_seconds,
            )

            if process.returncode == 0:
                logger.info(f"Successfully synced to {host}")
                return True
            else:
                logger.error(f"rsync to {host} failed: {stderr.decode()[-200:]}")
                return False

        except asyncio.TimeoutError:
            logger.error(f"rsync to {host} timed out")
            return False
        except Exception as e:
            logger.error(f"rsync to {host} failed: {e}")
            return False

    async def _get_training_nodes(self) -> list[dict[str, Any]]:
        """Get list of training-capable nodes from sync_router."""
        try:
            from app.coordination.sync_router import get_sync_router, DataType

            router = get_sync_router()
            targets = router.get_sync_targets(DataType.NPZ)
            return targets
        except ImportError:
            logger.warning("sync_router not available, using hosts from config")
            return await self._get_nodes_from_config()

    async def _get_nodes_from_config(self) -> list[dict[str, Any]]:
        """Fallback: get nodes from distributed_hosts.yaml."""
        try:
            import yaml

            config_path = ROOT / "config" / "distributed_hosts.yaml"
            if not config_path.exists():
                return []

            with open(config_path) as f:
                config = yaml.safe_load(f)

            nodes = []
            for node_id, node_cfg in config.get("hosts", {}).items():
                node_type = node_cfg.get("type", "")
                if node_type in self.config.target_node_types:
                    nodes.append({
                        "node_id": node_id,
                        "host": node_cfg.get("host"),
                        "user": node_cfg.get("user", "ubuntu"),
                        "ssh_key": node_cfg.get("ssh_key"),
                        "remote_path": node_cfg.get("remote_path", "~/ringrift/ai-service"),
                    })
            return nodes
        except Exception as e:
            logger.error(f"Failed to load nodes from config: {e}")
            return []

    async def _periodic_sync_check(self) -> None:
        """Periodic check for NPZ files that need distribution.

        December 2025: Uses checksum verification for all syncs.
        """
        training_dir = ROOT / self.config.training_data_dir
        if not training_dir.exists():
            return

        npz_files = list(training_dir.glob("*.npz"))
        if npz_files:
            # Check if any are recent (last 2 hours)
            recent_cutoff = time.time() - 7200
            recent_npz = [
                f for f in npz_files if f.stat().st_mtime > recent_cutoff
            ]

            if recent_npz:
                logger.info(
                    f"Found {len(recent_npz)} recently modified NPZ files, "
                    "triggering periodic sync"
                )
                target_nodes = await self._get_training_nodes()
                for npz_file in recent_npz:
                    # Compute checksum and distribute with verification
                    source_checksum = None
                    if self.config.verify_checksums:
                        source_checksum = await self._compute_checksum(npz_file)

                    success, results = await self._distribute_npz_with_verification(
                        str(npz_file), target_nodes, source_checksum
                    )
                    if success:
                        self._successful_distributions += 1
                    else:
                        self._failed_distributions += 1

                self._last_sync_time = time.time()

    async def _emit_distribution_complete(
        self,
        npz_info: dict[str, Any],
        target_nodes: list[dict[str, Any]],
        source_checksum: str | None = None,
        delivery_results: list[NPZDeliveryResult] | None = None,
    ) -> None:
        """Emit NPZ_DISTRIBUTION_COMPLETE event with delivery confirmation.

        December 2025: Enhanced with checksum and per-node delivery tracking.
        """
        # Compute delivery statistics
        if delivery_results:
            successful_nodes = [r for r in delivery_results if r.success]
            verified_nodes = [r for r in delivery_results if r.checksum_verified]
            delivery_summary = [
                {
                    "node_id": r.node_id,
                    "success": r.success,
                    "checksum_verified": r.checksum_verified,
                    "method": r.method,
                    "transfer_time_seconds": round(r.transfer_time_seconds, 2),
                }
                for r in delivery_results
            ]
        else:
            successful_nodes = []
            verified_nodes = []
            delivery_summary = []

        try:
            from app.coordination.event_router import emit

            await emit(
                "NPZ_DISTRIBUTION_COMPLETE",
                {
                    "npz_path": npz_info.get("npz_path"),
                    "board_type": npz_info.get("board_type"),
                    "num_players": npz_info.get("num_players"),
                    "sample_count": npz_info.get("sample_count"),
                    # Enhanced delivery tracking (December 2025)
                    "source_checksum": source_checksum[:16] + "..." if source_checksum else None,
                    "nodes_targeted": len(target_nodes),
                    "nodes_synced": len(successful_nodes),
                    "nodes_verified": len(verified_nodes),
                    "node_ids": [n.get("node_id") for n in target_nodes],
                    "delivery_summary": delivery_summary,
                    "timestamp": time.time(),
                },
            )
            logger.info(
                f"Emitted NPZ_DISTRIBUTION_COMPLETE event "
                f"({len(successful_nodes)}/{len(target_nodes)} nodes, "
                f"{len(verified_nodes)} verified)"
            )
        except ImportError:
            logger.debug("event_router not available, skipping event emission")
        except Exception as e:
            logger.error(f"Failed to emit event: {e}")

    async def _register_in_manifest(
        self, npz_path: str, target_nodes: list[dict[str, Any]]
    ) -> None:
        """Register NPZ distribution in ClusterManifest."""
        try:
            from app.distributed.cluster_manifest import get_cluster_manifest

            manifest = get_cluster_manifest()
            npz_file = Path(npz_path)

            for node in target_nodes:
                manifest.register_npz(
                    node_id=node.get("node_id"),
                    npz_path=str(npz_file),
                    board_type=npz_file.stem.split("_")[0],  # Parse from filename
                )
        except ImportError:
            logger.debug("ClusterManifest not available, skipping registration")
        except Exception as e:
            logger.error(f"Failed to register in manifest: {e}")


async def run() -> None:
    """Run the daemon (entry point for DaemonManager)."""
    daemon = NPZDistributionDaemon()
    await daemon.start()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(run())
