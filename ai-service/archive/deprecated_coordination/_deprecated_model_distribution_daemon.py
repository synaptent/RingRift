"""Model Distribution Daemon - Automatic model sync after promotion.

DEPRECATED (December 2025): Use unified_distribution_daemon.py instead.
This module is preserved for backward compatibility but will be removed in Q2 2026.

Migration:
    from app.coordination.unified_distribution_daemon import (
        UnifiedDistributionDaemon,
        create_unified_distribution_daemon,
    )

This daemon watches for MODEL_PROMOTED events and automatically distributes
promoted models to all cluster nodes, solving the gap where models would
only exist on the node where training completed.

Architecture:
    1. Subscribes to MODEL_PROMOTED events from event_router
    2. Uses sync_models.py --distribute for reliable multi-node sync
    3. Tracks distribution status in ClusterManifest
    4. Emits MODEL_DISTRIBUTION_COMPLETE event when done
    5. Validates checksums before and after transfer (December 2025)

Usage:
    # As standalone daemon
    python -m app.coordination.model_distribution_daemon

    # Via DaemonManager
    manager.register_factory(DaemonType.MODEL_DISTRIBUTION, daemon.run)

Configuration:
    Uses distributed_hosts.yaml for target nodes.
    See config/promotion_daemon.yaml for promotion thresholds.

December 2025 Enhancements:
    - SHA256 checksum validation before transfer
    - Remote checksum verification after transfer
    - Per-node delivery confirmation tracking
    - Metrics for checksum verification failures
"""

from __future__ import annotations

import warnings

warnings.warn(
    "model_distribution_daemon module is deprecated as of December 2025. "
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
class ModelDistributionConfig:
    """Configuration for model distribution daemon."""

    # Sync settings
    sync_timeout_seconds: float = 300.0  # 5 minute timeout for sync
    retry_count: int = 3
    retry_delay_seconds: float = 30.0

    # Priority settings
    priority_node_types: list[str] = field(
        default_factory=lambda: ["training", "selfplay"]
    )

    # Event settings
    emit_completion_event: bool = True

    # Polling (if no event system)
    poll_interval_seconds: float = 10.0  # Dec 2025: Reduced from 60s for faster model distribution
    models_dir: str = "models"

    # HTTP distribution settings (December 2025 - Phase 14)
    # HTTP is faster than rsync for model streaming
    use_http_distribution: bool = True  # Prefer HTTP when available
    http_port: int = 8767  # Port for model upload endpoint
    http_timeout_seconds: float = 60.0  # Timeout per node for HTTP upload
    http_concurrent_uploads: int = 5  # Max concurrent HTTP uploads
    fallback_to_rsync: bool = True  # Fallback to rsync if HTTP fails

    # Checksum verification settings (December 2025)
    verify_checksums: bool = True  # Enable SHA256 checksum verification
    checksum_timeout_seconds: float = 30.0  # Timeout for remote checksum verification

    # BitTorrent distribution settings (December 2025)
    # BitTorrent provides piece-level verification, preventing corruption on flaky connections
    use_bittorrent_for_large_files: bool = True  # Use BitTorrent for files > threshold
    bittorrent_size_threshold_bytes: int = 50_000_000  # 50MB - use BT above this
    bittorrent_timeout_seconds: float = 600.0  # 10 minute timeout for BT distribution
    bittorrent_min_seeders: int = 1  # Minimum seeders required to use BitTorrent
    create_torrents_for_models: bool = True  # Auto-create .torrent files for models


@dataclass
class ModelDeliveryResult:
    """Result of delivering a model to a single node."""

    node_id: str
    host: str
    model_name: str
    success: bool
    checksum_verified: bool
    transfer_time_seconds: float
    error_message: str = ""
    method: str = "http"  # http or rsync


class ModelDistributionDaemon:
    """Daemon that automatically distributes models after promotion.

    Watches for MODEL_PROMOTED events and syncs models to all cluster nodes.
    This ensures that newly trained models are available everywhere for:
    - Selfplay on GPU nodes
    - Tournament evaluation
    - Production serving

    The daemon solves the critical gap where models would only exist on the
    training node after promotion, causing selfplay failures on other nodes.
    """

    def __init__(self, config: ModelDistributionConfig | None = None):
        self.config = config or ModelDistributionConfig()
        self._running = False
        self._last_sync_time: float = 0.0
        self._pending_models: list[dict[str, Any]] = []
        self._sync_lock = asyncio.Lock()

        # December 2025: Thread-safe lock for _pending_models
        # The event callback (_on_model_promoted) runs from event router context
        # which may be different from the main loop context, causing race conditions.
        # This lock protects append and copy+clear operations on the list.
        self._pending_lock = threading.Lock()

        # Phase 5 (Dec 2025): Event-based wake-up for immediate push-on-promotion
        self._pending_event: asyncio.Event | None = None

        # CoordinatorProtocol state (December 2025 - Phase 14)
        self._coordinator_status = CoordinatorStatus.INITIALIZING
        self._start_time: float = 0.0
        self._events_processed: int = 0
        self._errors_count: int = 0
        self._last_error: str = ""
        self._successful_distributions: int = 0
        self._failed_distributions: int = 0
        self._checksum_failures: int = 0  # December 2025: Track checksum verification failures

        # Delivery tracking per node (December 2025)
        self._delivery_history: list[ModelDeliveryResult] = []

        # Cache of computed checksums to avoid recomputation
        self._model_checksums: dict[str, str] = {}

    # =========================================================================
    # CoordinatorProtocol Implementation (December 2025 - Phase 14)
    # =========================================================================

    @property
    def name(self) -> str:
        """Unique name identifying this coordinator."""
        return "ModelDistributionDaemon"

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

    def is_running(self) -> bool:
        """Check if the daemon is currently running."""
        return self._running

    def get_metrics(self) -> dict[str, Any]:
        """Get daemon metrics in protocol-compliant format.

        Returns:
            Dictionary of metrics including distribution-specific stats.
        """
        return {
            "name": self.name,
            "status": self._coordinator_status.value,
            "uptime_seconds": self.uptime_seconds,
            "start_time": self._start_time,
            "events_processed": self._events_processed,
            "errors_count": self._errors_count,
            "last_error": self._last_error,
            # Distribution-specific metrics
            "pending_models": len(self._pending_models),
            "successful_distributions": self._successful_distributions,
            "failed_distributions": self._failed_distributions,
            "checksum_failures": self._checksum_failures,  # December 2025
            "last_sync_time": self._last_sync_time,
        }

    def health_check(self) -> HealthCheckResult:
        """Check daemon health.

        Returns:
            Health check result with status and distribution details.
        """
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

        # Check for pending models buildup
        if len(self._pending_models) > 10:
            return HealthCheckResult.degraded(
                f"{len(self._pending_models)} models pending distribution",
                pending_models=len(self._pending_models),
            )

        # Check for checksum verification failures (December 2025)
        if self._checksum_failures > 5:
            return HealthCheckResult.degraded(
                f"{self._checksum_failures} checksum verification failures",
                checksum_failures=self._checksum_failures,
            )

        # Healthy
        return HealthCheckResult(
            healthy=True,
            status=self._coordinator_status,
            details={
                "uptime_seconds": self.uptime_seconds,
                "successful_distributions": self._successful_distributions,
                "pending_models": len(self._pending_models),
                "checksum_failures": self._checksum_failures,
                "last_sync_time": self._last_sync_time,
            },
        )

    async def start(self) -> None:
        """Start the daemon and subscribe to events."""
        if self._coordinator_status == CoordinatorStatus.RUNNING:
            return  # Already running

        logger.info("ModelDistributionDaemon starting...")
        self._running = True
        self._coordinator_status = CoordinatorStatus.RUNNING
        self._start_time = time.time()

        # Phase 5 (Dec 2025): Initialize event for immediate push-on-promotion
        self._pending_event = asyncio.Event()

        # Register with coordinator registry
        register_coordinator(self)

        # Subscribe to MODEL_PROMOTED events for automatic distribution
        try:
            from app.coordination.event_router import subscribe, DataEventType

            subscribe(DataEventType.MODEL_PROMOTED, self._on_model_promoted)
            logger.info("Subscribed to MODEL_PROMOTED events via event_router")

            # Also subscribe to PROMOTION_COMPLETE for stage events compatibility
            try:
                from app.coordination.event_router import StageEvent
                subscribe(StageEvent.PROMOTION_COMPLETE, self._on_model_promoted)
                logger.info("Also subscribed to PROMOTION_COMPLETE stage events")
            except ImportError:
                pass  # Stage events not available

        except ImportError as e:
            logger.warning(
                f"event_router not available ({e}), will poll for new models instead"
            )
        except Exception as e:
            logger.error(f"Failed to subscribe to MODEL_PROMOTED: {e}")

        # Main loop - handle pending syncs and periodic checks
        # Phase 5 (Dec 2025): Use event-based wake-up for immediate push-on-promotion
        while self._running:
            try:
                # Process any pending model distributions
                if self._pending_models:
                    await self._process_pending_models()

                # Periodic sync to catch any missed promotions
                if time.time() - self._last_sync_time > self.config.poll_interval_seconds:
                    await self._periodic_sync_check()

                # Phase 5: Wait for event with timeout (instant wake on promotion)
                # Instead of fixed 5s sleep, we wake immediately when event is set
                if self._pending_event is not None:
                    try:
                        await asyncio.wait_for(self._pending_event.wait(), timeout=5.0)
                        self._pending_event.clear()  # Reset for next event
                    except asyncio.TimeoutError:
                        pass  # Normal timeout, continue loop
                else:
                    await asyncio.sleep(5.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in distribution daemon loop: {e}")
                await asyncio.sleep(10.0)

        logger.info("ModelDistributionDaemon stopped")

    async def stop(self) -> None:
        """Stop the daemon gracefully."""
        if self._coordinator_status == CoordinatorStatus.STOPPED:
            return  # Already stopped

        self._coordinator_status = CoordinatorStatus.STOPPING
        self._running = False

        # Unregister from coordinator registry
        unregister_coordinator(self.name)

        self._coordinator_status = CoordinatorStatus.STOPPED

    def _on_model_promoted(self, event: dict[str, Any]) -> None:
        """Handle MODEL_PROMOTED event (sync callback).

        Phase 5 (Dec 2025): Immediately signals the main loop to process
        the new model, reducing distribution delay from 5s to near-instant.

        December 2025: Uses thread lock to prevent race condition with
        _process_pending_models which runs in the main async loop.
        """
        # Handle both RouterEvent and dict payloads
        payload = getattr(event, "payload", event) if hasattr(event, "payload") else event
        model_info = {
            "model_path": payload.get("model_path"),
            "model_id": payload.get("model_id"),
            "board_type": payload.get("board_type"),
            "num_players": payload.get("num_players"),
            "elo": payload.get("elo"),
            "timestamp": time.time(),
        }
        logger.info(f"Received MODEL_PROMOTED event: {model_info}")

        # Thread-safe append to pending list
        with self._pending_lock:
            self._pending_models.append(model_info)
        self._events_processed += 1

        # Phase 5: Immediate wake-up for push-on-promotion (no 5s wait)
        if self._pending_event is not None:
            self._pending_event.set()

    async def _process_pending_models(self) -> None:
        """Process pending model distributions.

        December 2025: Uses thread lock to safely copy+clear pending models,
        preventing race condition with _on_model_promoted callback.
        """
        async with self._sync_lock:
            # Thread-safe check and extract pending models
            with self._pending_lock:
                if not self._pending_models:
                    return
                # Atomically copy and clear while holding lock
                models = self._pending_models.copy()
                self._pending_models.clear()

            logger.info(f"Processing {len(models)} pending model distributions")

            # Run sync with retry (uses HTTP first, falls back to rsync)
            for attempt in range(self.config.retry_count):
                try:
                    success = await self._run_smart_sync()
                    if success:
                        self._last_sync_time = time.time()
                        logger.info("Model distribution completed successfully")

                        # Create symlinks for distributed models (Dec 2025)
                        # Selfplay engines look for ringrift_best_*.pth, but we sync canonical_*.pth
                        await self._create_model_symlinks()

                        # Emit completion event
                        if self.config.emit_completion_event:
                            await self._emit_distribution_complete(models)
                        return

                except Exception as e:
                    logger.error(f"Sync attempt {attempt + 1} failed: {e}")

                if attempt < self.config.retry_count - 1:
                    await asyncio.sleep(self.config.retry_delay_seconds)

            logger.error(
                f"Model distribution failed after {self.config.retry_count} attempts"
            )

    async def _run_model_sync(self) -> bool:
        """Execute model sync using sync_models.py --distribute."""
        sync_script = ROOT / "scripts" / "sync_models.py"
        if not sync_script.exists():
            logger.error(f"Sync script not found: {sync_script}")
            return False

        cmd = [
            sys.executable,
            str(sync_script),
            "--distribute",
        ]

        logger.info(f"Running: {' '.join(cmd)}")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(ROOT),
                env={**os.environ, "PYTHONPATH": str(ROOT)},
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.sync_timeout_seconds,
            )

            if process.returncode == 0:
                logger.info("sync_models.py --distribute completed successfully")
                if stdout:
                    logger.debug(f"stdout: {stdout.decode()[-500:]}")
                return True
            else:
                logger.error(f"sync_models.py failed with code {process.returncode}")
                if stderr:
                    logger.error(f"stderr: {stderr.decode()[-500:]}")
                return False

        except asyncio.TimeoutError:
            logger.error(
                f"sync_models.py timed out after {self.config.sync_timeout_seconds}s"
            )
            return False

    async def _distribute_via_http(
        self,
        model_paths: list[Path] | None = None,
    ) -> bool:
        """Distribute models to cluster nodes via HTTP streaming.

        This is faster than rsync for model distribution (December 2025).
        Uploads models directly to nodes' HTTP endpoints in parallel.

        Args:
            model_paths: Specific models to distribute. If None, distributes
                        all canonical models.

        Returns:
            True if all uploads succeeded, False otherwise.
        """
        try:
            import aiohttp
        except ImportError:
            logger.warning("aiohttp not available for HTTP distribution")
            return False

        # Get models to distribute
        models_dir = ROOT / self.config.models_dir
        if model_paths is None:
            model_paths = list(models_dir.glob("canonical_*.pth"))
            model_paths.extend(models_dir.glob("ringrift_best_*.pth"))

        if not model_paths:
            logger.info("No models to distribute")
            return True

        # Get target nodes from distributed hosts config
        target_nodes = self._get_distribution_targets()
        if not target_nodes:
            logger.warning("No target nodes for HTTP distribution")
            return False

        logger.info(
            f"Distributing {len(model_paths)} models to {len(target_nodes)} nodes via HTTP"
        )

        # Track results
        success_count = 0
        failure_count = 0

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.http_timeout_seconds)
        ) as session:
            # Create upload tasks with concurrency limit
            semaphore = asyncio.Semaphore(self.config.http_concurrent_uploads)

            async def upload_model(model_path: Path, node: str) -> bool:
                async with semaphore:
                    return await self._upload_model_to_node(
                        session, model_path, node
                    )

            # Upload all models to all nodes
            tasks = []
            for model_path in model_paths:
                for node in target_nodes:
                    tasks.append(upload_model(model_path, node))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.debug(f"Upload failed with exception: {result}")
                    failure_count += 1
                elif result:
                    success_count += 1
                else:
                    failure_count += 1

        total = len(model_paths) * len(target_nodes)
        logger.info(
            f"HTTP distribution complete: {success_count}/{total} successful, "
            f"{failure_count} failures"
        )

        # Update metrics
        self._successful_distributions += success_count
        self._failed_distributions += failure_count

        # Consider successful if most uploads worked
        return success_count > total * 0.5

    async def _distribute_via_bittorrent(
        self,
        model_paths: list[Path] | None = None,
    ) -> bool:
        """Distribute large models via BitTorrent with piece-level verification.

        December 2025: BitTorrent is ideal for large models (>50MB) because:
        - Piece-level verification catches corruption that rsync --partial misses
        - Multi-peer downloads provide redundancy on flaky connections
        - DHT enables trackerless peer discovery within the cluster
        - Seeding after download helps other nodes in the cluster

        This method:
        1. Creates .torrent files for models if they don't exist
        2. Registers torrents in ClusterManifest for peer discovery
        3. Seeds the torrents for other nodes to download

        Args:
            model_paths: Specific models to distribute. If None, distributes
                        all canonical models above size threshold.

        Returns:
            True if BitTorrent distribution initiated successfully, False otherwise.
        """
        try:
            from app.distributed.aria2_transport import Aria2Transport, Aria2Config
            from app.distributed.torrent_generator import get_torrent_generator
            from app.distributed.cluster_manifest import get_cluster_manifest
        except ImportError as e:
            logger.debug(f"BitTorrent support not available: {e}")
            return False

        # Get models to distribute
        models_dir = ROOT / self.config.models_dir
        if model_paths is None:
            model_paths = list(models_dir.glob("canonical_*.pth"))
            model_paths.extend(models_dir.glob("ringrift_best_*.pth"))

        if not model_paths:
            logger.info("No models found for BitTorrent distribution")
            return True

        # Filter to large files only
        large_models = [
            p for p in model_paths
            if p.exists() and p.stat().st_size > self.config.bittorrent_size_threshold_bytes
        ]

        if not large_models:
            logger.debug("No models above BitTorrent size threshold")
            return False  # Signal to use fallback transport

        logger.info(
            f"Distributing {len(large_models)} large models via BitTorrent "
            f"(>{self.config.bittorrent_size_threshold_bytes / 1024 / 1024:.0f}MB)"
        )

        # Create torrents and start seeding
        transport = Aria2Transport(Aria2Config(
            enable_bittorrent=True,
            bt_enable_dht=True,
            bt_enable_lpd=True,
            bt_enable_pex=True,
        ))

        success_count = 0
        for model_path in large_models:
            try:
                # Create and register torrent
                torrent_path, info_hash, error = await transport.create_and_register_torrent(
                    model_path,
                    web_seeds=self._get_web_seed_urls(model_path),
                )

                if torrent_path and info_hash:
                    logger.info(
                        f"Created torrent for {model_path.name}: {info_hash[:16]}..."
                    )
                    success_count += 1

                    # Start seeding this model
                    seed_success, seed_error = await transport.seed_file(
                        model_path,
                        torrent_path,
                        duration_seconds=int(self.config.bittorrent_timeout_seconds),
                    )

                    if seed_success:
                        logger.info(f"Started seeding {model_path.name}")
                    else:
                        logger.warning(f"Failed to seed {model_path.name}: {seed_error}")
                else:
                    logger.warning(f"Failed to create torrent for {model_path.name}: {error}")

            except Exception as e:
                logger.warning(f"BitTorrent distribution failed for {model_path.name}: {e}")

        await transport.close()

        logger.info(f"BitTorrent: Created {success_count}/{len(large_models)} torrents")
        return success_count > 0

    def _get_web_seed_urls(self, model_path: Path) -> list[str]:
        """Get web seed URLs for hybrid HTTP+BitTorrent downloads.

        Web seeds allow nodes without peers to still download via HTTP
        while getting piece-level verification from the torrent.

        Args:
            model_path: Path to the model file

        Returns:
            List of HTTP URLs that can serve this file
        """
        urls = []
        target_nodes = self._get_distribution_targets()

        for node in target_nodes[:5]:  # Limit to 5 web seeds
            # Assume HTTP data server on port 8766
            url = f"http://{node}:8766/models/{model_path.name}"
            urls.append(url)

        return urls

    async def _upload_model_to_node(
        self,
        session: "aiohttp.ClientSession",
        model_path: Path,
        node: str,
    ) -> bool:
        """Upload a single model to a node via HTTP.

        Args:
            session: aiohttp client session
            model_path: Path to model file
            node: Node hostname/IP

        Returns:
            True if upload succeeded, False otherwise
        """
        try:
            import aiohttp

            url = f"http://{node}:{self.config.http_port}/models/upload"

            # Read model data
            with open(model_path, "rb") as f:
                model_data = f.read()

            model_name = model_path.name

            # Create multipart form data
            form_data = aiohttp.FormData()
            form_data.add_field(
                "model",
                model_data,
                filename=model_name,
                content_type="application/octet-stream",
            )

            async with session.post(url, data=form_data) as response:
                if response.status == 200:
                    logger.debug(f"Successfully uploaded {model_name} to {node}")
                    return True
                else:
                    text = await response.text()
                    logger.debug(
                        f"Upload to {node} failed: {response.status} - {text[:100]}"
                    )
                    return False

        except asyncio.TimeoutError:
            logger.debug(f"Upload to {node} timed out")
            return False
        except Exception as e:
            logger.debug(f"Upload to {node} failed: {e}")
            return False

    def _get_distribution_targets(self) -> list[str]:
        """Get list of target nodes for distribution.

        Reads from distributed_hosts.yaml to find nodes that should
        receive model updates.

        December 2025: Fixed to use 'status' field (ready/offline/terminated)
        instead of 'active' field which doesn't exist in the config.

        Returns:
            List of node hostnames/IPs
        """
        try:
            import yaml

            config_path = ROOT / "config" / "distributed_hosts.yaml"
            if not config_path.exists():
                return []

            with open(config_path) as f:
                config = yaml.safe_load(f)

            hosts = config.get("hosts", {})
            targets = []

            for name, host_config in hosts.items():
                # Skip hosts that are not ready (check status field)
                status = host_config.get("status", "ready")
                if status not in ("ready", "active"):
                    continue

                # Get IP or hostname (check multiple field names)
                host = (
                    host_config.get("ssh_host")
                    or host_config.get("tailscale_ip")
                    or host_config.get("host")
                    or host_config.get("ip")
                )
                if host:
                    targets.append(host)

            return targets

        except Exception as e:
            logger.warning(f"Failed to read distribution targets: {e}")
            return []

    # =========================================================================
    # Checksum Verification (December 2025)
    # =========================================================================

    async def _compute_model_checksum(self, model_path: Path) -> str | None:
        """Compute SHA256 checksum of a model file.

        Uses cached checksums when available to avoid recomputation.

        Args:
            model_path: Path to the model file

        Returns:
            SHA256 hex digest or None on error
        """
        path_key = str(model_path)

        # Check cache first
        if path_key in self._model_checksums:
            return self._model_checksums[path_key]

        if not model_path.exists():
            return None

        try:
            from app.utils.checksum_utils import compute_file_checksum, LARGE_CHUNK_SIZE

            # Run in thread pool to avoid blocking
            # Dec 2025: Use get_running_loop() instead of deprecated get_event_loop()
            loop = asyncio.get_running_loop()
            checksum = await loop.run_in_executor(
                None,
                lambda: compute_file_checksum(model_path, chunk_size=LARGE_CHUNK_SIZE),
            )

            # Cache the result
            self._model_checksums[path_key] = checksum
            return checksum
        except Exception as e:
            logger.warning(f"Failed to compute checksum for {model_path}: {e}")
            return None

    async def _verify_model_on_remote(
        self,
        node: str,
        model_name: str,
        expected_checksum: str,
    ) -> bool:
        """Verify checksum of a model on a remote node via SSH.

        Args:
            node: Remote node hostname/IP
            model_name: Name of the model file (e.g., canonical_hex8_2p.pth)
            expected_checksum: Expected SHA256 checksum

        Returns:
            True if checksum matches, False otherwise
        """
        # Try to get SSH user from config
        ssh_user = "root"  # Default for most cloud nodes
        remote_path = "~/ringrift/ai-service"

        try:
            import yaml

            config_path = ROOT / "config" / "distributed_hosts.yaml"
            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f)

                for host_config in config.get("hosts", {}).values():
                    host = (
                        host_config.get("ssh_host")
                        or host_config.get("tailscale_ip")
                        or host_config.get("host")
                    )
                    if host == node:
                        ssh_user = host_config.get("ssh_user", "root")
                        remote_path = host_config.get("remote_path", remote_path)
                        break
        except (OSError, KeyError) as e:
            logger.debug(f"Could not read SSH config for node {node}: {e}")
            pass

        # Build remote checksum command
        remote_file = f"{remote_path}/models/{model_name}"
        checksum_cmd = f"sha256sum {remote_file} 2>/dev/null | cut -d' ' -f1"

        # Build SSH command
        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes",
            f"{ssh_user}@{node}",
            checksum_cmd,
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *ssh_cmd,
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
                    logger.debug(f"Checksum verified on {node}: {remote_checksum[:16]}...")
                    return True
                else:
                    logger.warning(
                        f"Checksum mismatch on {node}: "
                        f"expected {expected_checksum[:16]}..., "
                        f"got {remote_checksum[:16]}..."
                    )
                    self._checksum_failures += 1
                    return False
            else:
                logger.warning(f"Failed to get checksum from {node}: {stderr.decode()[:100]}")
                return False

        except asyncio.TimeoutError:
            logger.warning(f"Checksum verification timed out on {node}")
            return False
        except Exception as e:
            logger.warning(f"Checksum verification failed on {node}: {e}")
            return False

    async def _verify_all_models_on_node(
        self,
        node: str,
        models: list[Path],
    ) -> list[ModelDeliveryResult]:
        """Verify all distributed models on a remote node.

        Args:
            node: Remote node hostname/IP
            models: List of model paths that were distributed

        Returns:
            List of delivery results for each model
        """
        results = []
        start_time = time.time()

        for model_path in models:
            expected_checksum = await self._compute_model_checksum(model_path)
            if not expected_checksum:
                results.append(ModelDeliveryResult(
                    node_id=node,
                    host=node,
                    model_name=model_path.name,
                    success=True,  # Transfer may have succeeded
                    checksum_verified=False,
                    transfer_time_seconds=time.time() - start_time,
                    error_message="Could not compute source checksum",
                    method="unknown",
                ))
                continue

            verified = await self._verify_model_on_remote(
                node, model_path.name, expected_checksum
            )

            results.append(ModelDeliveryResult(
                node_id=node,
                host=node,
                model_name=model_path.name,
                success=verified,
                checksum_verified=verified,
                transfer_time_seconds=time.time() - start_time,
                method="verified",
            ))

        return results

    async def _run_smart_sync(self) -> bool:
        """Run model sync using best available method.

        December 2025 transport priority:
        1. BitTorrent for large files (>50MB) - piece-level verification
        2. HTTP streaming - fast for smaller files
        3. rsync fallback - reliable but slower

        BitTorrent is preferred for large models because it provides:
        - Piece-level SHA1 verification (catches corruption rsync --partial misses)
        - Multi-peer downloads for redundancy
        - DHT for trackerless peer discovery
        - Resume capability across connection drops

        Returns:
            True if sync succeeded, False otherwise
        """
        distribution_succeeded = False
        method = "unknown"

        # Try BitTorrent first for large models (December 2025)
        if self.config.use_bittorrent_for_large_files:
            bt_success = await self._distribute_via_bittorrent()
            if bt_success:
                distribution_succeeded = True
                method = "bittorrent"
                logger.info("Large model distribution via BitTorrent succeeded")
                # Note: BitTorrent only handles large files, continue with HTTP for small files

        # Try HTTP for remaining/smaller files
        if self.config.use_http_distribution:
            http_success = await self._distribute_via_http()
            if http_success:
                distribution_succeeded = True
                if method == "unknown":
                    method = "http"
                else:
                    method = f"{method}+http"
            elif self.config.fallback_to_rsync and not distribution_succeeded:
                logger.info("HTTP distribution failed, falling back to rsync")
                rsync_success = await self._run_model_sync()
                if rsync_success:
                    distribution_succeeded = True
                    method = "rsync"
        elif not distribution_succeeded:
            # Rsync only (no HTTP)
            rsync_success = await self._run_model_sync()
            if rsync_success:
                distribution_succeeded = True
                method = "rsync"

        if not distribution_succeeded:
            return False

        # December 2025: Verify checksums on remote nodes after distribution
        if self.config.verify_checksums:
            models_dir = ROOT / self.config.models_dir
            canonical_models = list(models_dir.glob("canonical_*.pth"))
            targets = self._get_distribution_targets()

            all_results: list[ModelDeliveryResult] = []
            verified_count = 0

            for node in targets:
                results = await self._verify_all_models_on_node(node, canonical_models)
                all_results.extend(results)
                verified_count += sum(1 for r in results if r.checksum_verified)

            # Track delivery history
            self._delivery_history.extend(all_results)
            if len(self._delivery_history) > 200:
                self._delivery_history = self._delivery_history[-200:]

            # Log verification summary
            total_checks = len(all_results)
            if total_checks > 0:
                logger.info(
                    f"Model distribution verification: {verified_count}/{total_checks} "
                    f"checksums verified via {method}"
                )

        return True

    async def _periodic_sync_check(self) -> None:
        """Periodic check for models that need distribution."""
        # Check if there are local canonical models that may need sync
        models_dir = ROOT / self.config.models_dir
        if not models_dir.exists():
            return

        canonical_models = list(models_dir.glob("canonical_*.pth"))
        if canonical_models:
            # Check if any are recent (last hour)
            recent_cutoff = time.time() - 3600
            recent_models = [
                m for m in canonical_models if m.stat().st_mtime > recent_cutoff
            ]

            if recent_models:
                logger.info(
                    f"Found {len(recent_models)} recently modified canonical models, "
                    "triggering periodic sync"
                )
                # Use smart sync (HTTP first, fallback to rsync)
                await self._run_smart_sync()
                self._last_sync_time = time.time()

    async def _create_model_symlinks(self) -> None:
        """Create ringrift_best_*.pth symlinks pointing to canonical_*.pth models.

        December 2025: This ensures selfplay engines can find promoted models.
        Selfplay looks for 'ringrift_best_{board}_{n}p.pth' but we distribute
        'canonical_{board}_{n}p.pth'. Symlinks bridge this gap.

        Creates symlinks locally and on all cluster nodes via SSH.
        """
        models_dir = ROOT / self.config.models_dir

        # Find all canonical models
        canonical_models = list(models_dir.glob("canonical_*.pth"))
        if not canonical_models:
            logger.debug("No canonical models found for symlink creation")
            return

        created_count = 0
        for canonical_path in canonical_models:
            # Extract config from canonical_hex8_2p.pth -> hex8_2p
            name = canonical_path.stem  # canonical_hex8_2p
            if not name.startswith("canonical_"):
                continue

            config_key = name[len("canonical_"):]  # hex8_2p
            symlink_name = f"ringrift_best_{config_key}.pth"
            symlink_path = models_dir / symlink_name

            try:
                # Remove existing file/symlink if present
                if symlink_path.exists() or symlink_path.is_symlink():
                    symlink_path.unlink()

                # Create relative symlink (canonical_hex8_2p.pth, not absolute path)
                symlink_path.symlink_to(canonical_path.name)
                created_count += 1
                logger.debug(f"Created symlink: {symlink_name} -> {canonical_path.name}")
            except OSError as e:
                logger.warning(f"Failed to create symlink {symlink_name}: {e}")

        if created_count > 0:
            logger.info(f"Created {created_count} model symlinks locally")

        # Also create symlinks on cluster nodes via SSH
        await self._create_remote_symlinks(canonical_models)

    async def _create_remote_symlinks(self, canonical_models: list[Path]) -> None:
        """Create symlinks on remote cluster nodes.

        Args:
            canonical_models: List of canonical model paths to create symlinks for
        """
        # Get target nodes
        target_nodes = self._get_distribution_targets()
        if not target_nodes:
            return

        # Build symlink commands
        symlink_commands = []
        for canonical_path in canonical_models:
            name = canonical_path.stem
            if not name.startswith("canonical_"):
                continue

            config_key = name[len("canonical_"):]
            # Remote command: cd models && ln -sf canonical_X.pth ringrift_best_X.pth
            cmd = (
                f"cd ~/ringrift/ai-service/models 2>/dev/null || "
                f"cd ~/RingRift/ai-service/models 2>/dev/null && "
                f"ln -sf {canonical_path.name} ringrift_best_{config_key}.pth"
            )
            symlink_commands.append(cmd)

        if not symlink_commands:
            return

        # Combine all symlink commands
        combined_cmd = " && ".join(symlink_commands)

        # Run on all nodes in parallel
        success_count = 0
        tasks = []

        for node in target_nodes:
            task = self._run_remote_command(node, combined_cmd)
            tasks.append((node, task))

        for node, task in tasks:
            try:
                result = await task
                if result:
                    success_count += 1
            except Exception as e:
                logger.debug(f"Failed to create symlinks on {node}: {e}")

        if success_count > 0:
            logger.info(
                f"Created model symlinks on {success_count}/{len(target_nodes)} cluster nodes"
            )

    async def _run_remote_command(self, node: str, command: str) -> bool:
        """Run a command on a remote node via SSH.

        Args:
            node: Node hostname/IP
            command: Command to run

        Returns:
            True if command succeeded, False otherwise
        """
        try:
            # Try to get SSH user from config
            ssh_user = "root"  # Default for most cloud nodes
            try:
                import yaml

                config_path = ROOT / "config" / "distributed_hosts.yaml"
                if config_path.exists():
                    with open(config_path) as f:
                        config = yaml.safe_load(f)

                    for host_config in config.get("hosts", {}).values():
                        host = (
                            host_config.get("ssh_host")
                            or host_config.get("tailscale_ip")
                            or host_config.get("host")
                        )
                        if host == node:
                            ssh_user = host_config.get("ssh_user", "root")
                            break
            except (OSError, KeyError) as e:
                logger.debug(f"Could not read SSH config for remote command on {node}: {e}")
                pass

            ssh_cmd = [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=10",
                "-o", "BatchMode=yes",
                f"{ssh_user}@{node}",
                command,
            ]

            process = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )

            await asyncio.wait_for(process.wait(), timeout=30.0)
            return process.returncode == 0

        except asyncio.TimeoutError:
            return False
        except (OSError, subprocess.SubprocessError):
            return False

    async def _emit_distribution_complete(
        self, models: list[dict[str, Any]]
    ) -> None:
        """Emit MODEL_DISTRIBUTION_COMPLETE event with distribution confirmation."""
        # Query ClusterManifest for model locations to confirm distribution
        confirmed_distributions = []
        try:
            from app.distributed.cluster_manifest import get_cluster_manifest

            manifest = get_cluster_manifest()
            for model_info in models:
                model_path = model_info.get("path", "")
                if model_path:
                    locations = manifest.find_model(model_path)
                    node_ids = [loc.node_id for loc in locations]
                    confirmed_distributions.append({
                        "model": model_path,
                        "nodes": node_ids,
                        "confirmed_count": len(node_ids),
                    })
        except ImportError:
            logger.debug("ClusterManifest not available for distribution confirmation")
        except Exception as e:
            logger.warning(f"Failed to query model locations: {e}")

        try:
            from app.coordination.event_router import emit

            await emit(
                event_type="MODEL_DISTRIBUTION_COMPLETE",
                data={
                    "models": models,
                    "confirmed_distributions": confirmed_distributions,
                    "timestamp": time.time(),
                    "node_id": os.environ.get("RINGRIFT_NODE_ID", "unknown"),
                },
            )
            # Log confirmation summary
            if confirmed_distributions:
                total_confirmations = sum(d["confirmed_count"] for d in confirmed_distributions)
                logger.info(
                    f"Emitted MODEL_DISTRIBUTION_COMPLETE event "
                    f"({len(models)} models, {total_confirmations} node confirmations)"
                )
            else:
                logger.info("Emitted MODEL_DISTRIBUTION_COMPLETE event")
        except Exception as e:
            logger.warning(f"Failed to emit distribution event: {e}")


# =========================================================================
# Public API for waiting on model distribution
# =========================================================================

async def wait_for_model_distribution(
    board_type: str,
    num_players: int,
    timeout: float = 300.0,
) -> bool:
    """Wait for a model to be distributed to this node.

    This function waits for MODEL_DISTRIBUTION_COMPLETE event or checks
    if the model already exists locally. Use this before selfplay starts
    to avoid race conditions where nodes try to load models before
    distribution completes.

    Args:
        board_type: Board type (hex8, square8, etc.)
        num_players: Number of players (2, 3, 4)
        timeout: Maximum time to wait in seconds (default: 300)

    Returns:
        True if model is available, False if timed out

    Example:
        # In selfplay runner setup
        available = await wait_for_model_distribution("hex8", 2, timeout=300)
        if not available:
            logger.warning("Model distribution timed out, using fallback")
    """
    config_key = f"{board_type}_{num_players}p"
    model_name = f"canonical_{config_key}.pth"

    # Check if model already exists locally
    models_dir = ROOT / "models"
    model_path = models_dir / model_name

    if model_path.exists():
        logger.debug(f"[ModelDistribution] Model already available: {model_path}")
        return True

    # Wait for MODEL_DISTRIBUTION_COMPLETE event
    logger.info(
        f"[ModelDistribution] Waiting for {model_name} distribution "
        f"(timeout: {timeout}s)..."
    )

    distribution_event = asyncio.Event()

    def on_distribution_complete(event):
        """Handle MODEL_DISTRIBUTION_COMPLETE event."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            models = payload.get("models", [])

            # Check if our model was distributed
            for model_info in models:
                model_path_str = model_info.get("model_path", "")
                if model_name in model_path_str:
                    logger.info(f"[ModelDistribution] Received {model_name} distribution complete")
                    distribution_event.set()
                    return
        except Exception as e:
            logger.warning(f"[ModelDistribution] Error handling event: {e}")

    # Subscribe to distribution events
    try:
        from app.coordination.event_router import subscribe, DataEventType

        subscribe(DataEventType.MODEL_DISTRIBUTION_COMPLETE, on_distribution_complete)

        # Wait with timeout
        try:
            await asyncio.wait_for(distribution_event.wait(), timeout=timeout)
            logger.info(f"[ModelDistribution] Model {model_name} is now available")
            return True
        except asyncio.TimeoutError:
            logger.warning(
                f"[ModelDistribution] Timed out waiting for {model_name} "
                f"after {timeout}s"
            )
            return False

    except ImportError:
        logger.debug("[ModelDistribution] Event system not available")
        # Without event system, just check if file appeared
        start_time = time.time()
        while time.time() - start_time < timeout:
            if model_path.exists():
                logger.info(f"[ModelDistribution] Model {model_name} found on disk")
                return True
            await asyncio.sleep(5.0)

        logger.warning(f"[ModelDistribution] Model {model_name} not found after {timeout}s")
        return False


def check_model_availability(
    board_type: str,
    num_players: int,
) -> bool:
    """Synchronously check if a model is available locally.

    Args:
        board_type: Board type (hex8, square8, etc.)
        num_players: Number of players (2, 3, 4)

    Returns:
        True if model exists locally, False otherwise

    Example:
        if not check_model_availability("hex8", 2):
            logger.warning("Model not available yet")
    """
    config_key = f"{board_type}_{num_players}p"
    model_name = f"canonical_{config_key}.pth"
    models_dir = ROOT / "models"
    model_path = models_dir / model_name

    # Also check for symlinks (ringrift_best_*.pth)
    symlink_name = f"ringrift_best_{config_key}.pth"
    symlink_path = models_dir / symlink_name

    return model_path.exists() or symlink_path.exists()


# Daemon adapter for DaemonManager integration
class ModelDistributionDaemonAdapter:
    """Adapter for integrating with DaemonManager."""

    def __init__(self, config: ModelDistributionConfig | None = None):
        self.config = config
        self._daemon: ModelDistributionDaemon | None = None

    @property
    def daemon_type(self) -> str:
        return "MODEL_DISTRIBUTION"

    @property
    def depends_on(self) -> list[str]:
        return []  # No dependencies

    async def run(self) -> None:
        """Run the daemon (DaemonManager entry point)."""
        self._daemon = ModelDistributionDaemon(self.config)
        await self._daemon.start()

    async def stop(self) -> None:
        """Stop the daemon."""
        if self._daemon:
            await self._daemon.stop()


async def main() -> None:
    """Run daemon standalone."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    daemon = ModelDistributionDaemon()
    try:
        await daemon.start()
    except KeyboardInterrupt:
        await daemon.stop()


if __name__ == "__main__":
    asyncio.run(main())
