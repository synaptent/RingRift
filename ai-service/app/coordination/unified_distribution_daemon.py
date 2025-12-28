"""Unified Distribution Daemon - Automatic model and NPZ sync after promotion/export.

This daemon consolidates ModelDistributionDaemon and NPZDistributionDaemon into a
single daemon that handles distribution of all data types (models, NPZ, torrents).

Architecture:
    1. Subscribes to MODEL_PROMOTED and NPZ_EXPORT_COMPLETE events
    2. Uses smart transport selection (BitTorrent > HTTP > rsync)
    3. Tracks distribution status with checksum verification
    4. Emits completion events when done
    5. CoordinatorProtocol integration for health monitoring

December 2025 Consolidation:
    - Combines model_distribution_daemon.py (1444 lines) and npz_distribution_daemon.py (1173 lines)
    - Eliminates ~1100 lines of duplicate code
    - Single daemon handles all data distribution
    - Factory functions for backward compatibility

Usage:
    # As standalone daemon
    python -m app.coordination.unified_distribution_daemon

    # Via DaemonManager
    from app.coordination.unified_distribution_daemon import (
        create_unified_distribution_daemon,
        create_model_distribution_daemon,  # Backward compat
        create_npz_distribution_daemon,    # Backward compat
    )

Configuration:
    Uses distributed_hosts.yaml for target nodes.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

from app.coordination.protocols import (
    CoordinatorStatus,
    HealthCheckResult,
    register_coordinator,
    unregister_coordinator,
)

# Delivery ledger for persistent tracking (Dec 2025 Phase 3)
try:
    from app.coordination.delivery_ledger import (
        DeliveryLedger,
        DeliveryStatus,
        get_delivery_ledger,
    )
    from app.coordination.delivery_retry_queue import (
        DeliveryRetryQueue,
        get_delivery_retry_queue,
    )
    DELIVERY_LEDGER_AVAILABLE = True
except ImportError:
    DELIVERY_LEDGER_AVAILABLE = False
    DeliveryLedger = None  # type: ignore
    DeliveryStatus = None  # type: ignore
    get_delivery_ledger = None  # type: ignore
    DeliveryRetryQueue = None  # type: ignore
    get_delivery_retry_queue = None  # type: ignore

logger = logging.getLogger(__name__)

# Add parent to path for imports
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# =============================================================================
# Remote Path Detection
# =============================================================================

# Known remote path patterns for different providers (in order of preference)
REMOTE_PATH_PATTERNS: list[str] = [
    "/workspace/ringrift/ai-service",      # RunPod, some Vast.ai
    "~/ringrift/ai-service",                # Lambda, Nebius, most providers
    "/root/ringrift/ai-service",            # Vultr, Hetzner (non-tilde expanded)
    "~/Development/RingRift/ai-service",    # Mac Studio coordinator
]

# Cache for discovered remote paths per host
_remote_path_cache: dict[str, str] = {}
_remote_path_cache_lock = threading.Lock()


# =============================================================================
# Enums and Configuration
# =============================================================================

class DataType(Enum):
    """Types of data that can be distributed."""
    MODEL = auto()
    NPZ = auto()
    TORRENT = auto()


@dataclass
class DistributionConfig:
    """Unified configuration for distribution daemon."""

    # Sync settings
    sync_timeout_seconds: float = 300.0
    retry_count: int = 3
    retry_delay_seconds: float = 30.0
    retry_backoff_multiplier: float = 1.5

    # Event settings
    emit_completion_event: bool = True
    poll_interval_seconds: float = 60.0

    # HTTP distribution settings
    use_http_distribution: bool = True
    http_port: int = 8767
    http_timeout_seconds: float = 120.0
    http_concurrent_uploads: int = 5
    fallback_to_rsync: bool = True

    # Checksum verification
    verify_checksums: bool = True
    checksum_timeout_seconds: float = 30.0

    # BitTorrent settings
    use_bittorrent_for_large_files: bool = True
    bittorrent_threshold_bytes: int = 50_000_000  # 50MB

    # Data type specific paths
    models_dir: str = "models"
    training_data_dir: str = "data/training"

    # NPZ-specific settings
    validate_npz_structure: bool = True
    max_npz_samples: int = 100_000_000

    # Model-specific settings
    create_symlinks: bool = True


@dataclass
class DeliveryResult:
    """Result of delivering data to a single node."""

    node_id: str
    host: str
    data_path: str
    data_type: DataType
    success: bool
    checksum_verified: bool
    transfer_time_seconds: float
    error_message: str = ""
    method: str = "http"  # http, rsync, bittorrent


# =============================================================================
# Unified Distribution Daemon
# =============================================================================

class UnifiedDistributionDaemon:
    """Daemon that distributes models and NPZ files to cluster nodes.

    Consolidates ModelDistributionDaemon and NPZDistributionDaemon into a single
    daemon that handles all data distribution with smart transport selection.

    Features:
    - Event-driven distribution (MODEL_PROMOTED, NPZ_EXPORT_COMPLETE)
    - Smart transport: BitTorrent for large files > HTTP > rsync fallback
    - SHA256 checksum verification before and after transfer
    - Per-node delivery tracking with metrics
    - CoordinatorProtocol integration for health monitoring
    """

    def __init__(self, config: DistributionConfig | None = None):
        self.config = config or DistributionConfig()
        self._running = False
        self._last_sync_time: float = 0.0

        # Pending items for distribution
        self._pending_items: list[dict[str, Any]] = []
        self._pending_lock = threading.Lock()
        self._pending_event: asyncio.Event | None = None
        self._sync_lock = asyncio.Lock()

        # CoordinatorProtocol state
        self._coordinator_status = CoordinatorStatus.INITIALIZING
        self._start_time: float = 0.0
        self._events_processed: int = 0
        self._errors_count: int = 0
        self._last_error: str = ""

        # Distribution metrics
        self._successful_distributions: int = 0
        self._failed_distributions: int = 0
        self._checksum_failures: int = 0
        self._model_distributions: int = 0
        self._npz_distributions: int = 0

        # Delivery history
        self._delivery_history: list[DeliveryResult] = []
        self._checksum_cache: dict[str, str] = {}

        # Persistent delivery ledger (Dec 2025 Phase 3)
        self._delivery_ledger: DeliveryLedger | None = None
        self._retry_queue: DeliveryRetryQueue | None = None
        if DELIVERY_LEDGER_AVAILABLE:
            try:
                self._delivery_ledger = get_delivery_ledger()
                self._retry_queue = get_delivery_retry_queue()
                logger.debug("[UnifiedDistributionDaemon] Delivery ledger initialized")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[UnifiedDistributionDaemon] Failed to init ledger: {e}")

    # =========================================================================
    # CoordinatorProtocol Implementation
    # =========================================================================

    @property
    def name(self) -> str:
        return "UnifiedDistributionDaemon"

    @property
    def status(self) -> CoordinatorStatus:
        return self._coordinator_status

    @property
    def uptime_seconds(self) -> float:
        if self._start_time <= 0:
            return 0.0
        return time.time() - self._start_time

    def is_running(self) -> bool:
        return self._running

    def get_metrics(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self._coordinator_status.value,
            "uptime_seconds": self.uptime_seconds,
            "start_time": self._start_time,
            "events_processed": self._events_processed,
            "errors_count": self._errors_count,
            "last_error": self._last_error,
            "pending_items": len(self._pending_items),
            "successful_distributions": self._successful_distributions,
            "failed_distributions": self._failed_distributions,
            "checksum_failures": self._checksum_failures,
            "model_distributions": self._model_distributions,
            "npz_distributions": self._npz_distributions,
            "last_sync_time": self._last_sync_time,
        }

    def health_check(self) -> HealthCheckResult:
        if self._coordinator_status == CoordinatorStatus.ERROR:
            return HealthCheckResult.unhealthy(f"Daemon in error state: {self._last_error}")

        if self._coordinator_status == CoordinatorStatus.STOPPED:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="Daemon is stopped",
            )

        # Check for high failure rate
        total = self._successful_distributions + self._failed_distributions
        if total > 0 and self._failed_distributions > self._successful_distributions:
            return HealthCheckResult.degraded(
                f"High failure rate: {self._failed_distributions}/{total}",
                failure_rate=self._failed_distributions / total,
            )

        # Check for pending buildup
        if len(self._pending_items) > 15:
            return HealthCheckResult.degraded(
                f"{len(self._pending_items)} items pending distribution",
            )

        # Check for checksum failures
        if self._checksum_failures > 5:
            return HealthCheckResult.degraded(
                f"{self._checksum_failures} checksum failures",
            )

        return HealthCheckResult(
            healthy=True,
            status=self._coordinator_status,
            details={
                "uptime_seconds": self.uptime_seconds,
                "successful_distributions": self._successful_distributions,
                "model_distributions": self._model_distributions,
                "npz_distributions": self._npz_distributions,
            },
        )

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> None:
        """Start the daemon and subscribe to events."""
        if self._coordinator_status == CoordinatorStatus.RUNNING:
            return

        logger.info("UnifiedDistributionDaemon starting...")
        self._running = True
        self._coordinator_status = CoordinatorStatus.RUNNING
        self._start_time = time.time()
        self._pending_event = asyncio.Event()

        register_coordinator(self)

        # Subscribe to distribution events
        self._subscribe_to_events()

        # Main loop
        while self._running:
            try:
                if self._pending_items:
                    await self._process_pending_items()

                # Periodic sync check
                if time.time() - self._last_sync_time > self.config.poll_interval_seconds:
                    await self._periodic_sync_check()

                # Wait for event or timeout
                if self._pending_event:
                    try:
                        await asyncio.wait_for(self._pending_event.wait(), timeout=5.0)
                        self._pending_event.clear()
                    except asyncio.TimeoutError:
                        pass
                else:
                    await asyncio.sleep(5.0)

            except asyncio.CancelledError:
                break
            except (OSError, RuntimeError, ValueError) as e:
                logger.error(f"Error in distribution daemon loop: {e}")
                self._errors_count += 1
                self._last_error = str(e)
                await asyncio.sleep(10.0)

        logger.info("UnifiedDistributionDaemon stopped")

    async def stop(self) -> None:
        """Stop the daemon gracefully."""
        if self._coordinator_status == CoordinatorStatus.STOPPED:
            return

        self._coordinator_status = CoordinatorStatus.STOPPING
        self._running = False
        unregister_coordinator(self.name)
        self._coordinator_status = CoordinatorStatus.STOPPED

    def _subscribe_to_events(self) -> None:
        """Subscribe to MODEL_PROMOTED and NPZ_EXPORT_COMPLETE events."""
        try:
            from app.coordination.event_router import DataEventType, subscribe

            # Model events
            subscribe(DataEventType.MODEL_PROMOTED, self._on_model_promoted)
            subscribe(DataEventType.MODEL_UPDATED, self._on_model_updated)
            subscribe(DataEventType.MODEL_DISTRIBUTION_STARTED, self._on_model_distribution_started)
            subscribe(DataEventType.MODEL_DISTRIBUTION_FAILED, self._on_model_distribution_failed)
            logger.info("Subscribed to MODEL_PROMOTED, MODEL_UPDATED events")

            # NPZ events
            try:
                from app.coordination.event_router import StageEvent
                subscribe(StageEvent.NPZ_EXPORT_COMPLETE, self._on_npz_exported)
                logger.info("Subscribed to NPZ_EXPORT_COMPLETE events")
            except (ImportError, AttributeError):
                subscribe("npz_export_complete", self._on_npz_exported)
                logger.info("Subscribed to npz_export_complete string events")

        except ImportError as e:
            logger.warning(f"Event system not available ({e}), will poll for new files")

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def _on_model_promoted(self, event: dict[str, Any] | Any) -> None:
        """Handle MODEL_PROMOTED event."""
        payload = getattr(event, "payload", event) if hasattr(event, "payload") else event
        item = {
            "data_type": DataType.MODEL,
            "path": payload.get("model_path"),
            "model_id": payload.get("model_id"),
            "board_type": payload.get("board_type"),
            "num_players": payload.get("num_players"),
            "elo": payload.get("elo"),
            "timestamp": time.time(),
        }
        logger.info(f"Received MODEL_PROMOTED: {item.get('path')}")
        self._enqueue_item(item)

    def _on_model_updated(self, event: dict[str, Any] | Any) -> None:
        """Handle MODEL_UPDATED event - model metadata or path changed (pre-promotion).

        December 2025: Wire MODEL_UPDATED to trigger distribution when a model
        file path or metadata changes, even before formal promotion. This enables
        faster propagation of model updates across the cluster.
        """
        payload = getattr(event, "payload", event) if hasattr(event, "payload") else event

        model_path = payload.get("model_path", payload.get("path"))
        model_id = payload.get("model_id", "")
        update_type = payload.get("update_type", "metadata")
        config_key = payload.get("config_key", "")

        # Only trigger distribution for path changes or explicit sync requests
        if update_type not in ("path_changed", "sync_requested", "symlink_updated"):
            logger.debug(
                f"MODEL_UPDATED: Ignoring metadata update for {model_id or config_key} "
                f"(type={update_type})"
            )
            return

        if not model_path:
            logger.warning(f"MODEL_UPDATED: No path in event for {model_id or config_key}")
            return

        item = {
            "data_type": DataType.MODEL,
            "path": model_path,
            "model_id": model_id,
            "config_key": config_key,
            "update_type": update_type,
            "timestamp": time.time(),
        }
        logger.info(f"Received MODEL_UPDATED: {model_path} (type={update_type})")
        self._enqueue_item(item)

    def _on_model_distribution_started(self, event: dict[str, Any] | Any) -> None:
        """Handle MODEL_DISTRIBUTION_STARTED event (external distribution)."""
        payload = getattr(event, "payload", event) if hasattr(event, "payload") else event
        total_models = payload.get("total_models")
        target_hosts = payload.get("target_hosts")
        logger.info(
            "MODEL_DISTRIBUTION_STARTED (external): "
            f"{total_models} models -> {target_hosts} hosts"
        )

    def _on_model_distribution_failed(self, event: dict[str, Any] | Any) -> None:
        """Handle MODEL_DISTRIBUTION_FAILED event (external distribution)."""
        payload = getattr(event, "payload", event) if hasattr(event, "payload") else event
        error = payload.get("error", "unknown_error")
        self._errors_count += 1
        self._last_error = str(error)
        logger.warning(f"MODEL_DISTRIBUTION_FAILED (external): {error}")

    def _on_npz_exported(self, event: dict[str, Any] | Any) -> None:
        """Handle NPZ_EXPORT_COMPLETE event."""
        payload = getattr(event, "payload", event) if hasattr(event, "payload") else event
        item = {
            "data_type": DataType.NPZ,
            "path": payload.get("npz_path"),
            "board_type": payload.get("board_type"),
            "num_players": payload.get("num_players"),
            "sample_count": payload.get("sample_count"),
            "timestamp": time.time(),
        }
        logger.info(f"Received NPZ_EXPORT_COMPLETE: {item.get('path')}")
        self._enqueue_item(item)

    def _enqueue_item(self, item: dict[str, Any]) -> None:
        """Thread-safe enqueue of distribution item."""
        with self._pending_lock:
            self._pending_items.append(item)
        self._events_processed += 1
        if self._pending_event:
            self._pending_event.set()

    # =========================================================================
    # Distribution Logic
    # =========================================================================

    async def _process_pending_items(self) -> None:
        """Process all pending distribution items."""
        async with self._sync_lock:
            with self._pending_lock:
                if not self._pending_items:
                    return
                items = self._pending_items.copy()
                self._pending_items.clear()

            logger.info(f"Processing {len(items)} pending distributions")

            # Group by data type for batch processing
            model_items = [i for i in items if i.get("data_type") == DataType.MODEL]
            npz_items = [i for i in items if i.get("data_type") == DataType.NPZ]

            # Process models
            if model_items:
                await self._distribute_models(model_items)

            # Process NPZ files
            if npz_items:
                await self._distribute_npz_files(npz_items)

    async def _distribute_models(self, items: list[dict[str, Any]]) -> None:
        """Distribute model files to cluster nodes."""
        target_nodes = self._get_distribution_targets()
        if not target_nodes:
            logger.warning("No target nodes for model distribution")
            return

        # Collect model paths
        model_paths = []
        for item in items:
            path = item.get("path")
            if path:
                model_paths.append(Path(path))

        # Also include canonical models
        models_dir = ROOT / self.config.models_dir
        if models_dir.exists():
            model_paths.extend(models_dir.glob("canonical_*.pth"))

        if not model_paths:
            logger.debug("No models to distribute")
            return

        # Deduplicate
        model_paths = list(set(model_paths))
        logger.info(f"Distributing {len(model_paths)} models to {len(target_nodes)} nodes")

        success = await self._run_smart_distribution(
            model_paths, target_nodes, DataType.MODEL
        )

        if success:
            self._model_distributions += len(model_paths)
            self._successful_distributions += 1
            self._last_sync_time = time.time()

            # Create local symlinks for models
            if self.config.create_symlinks:
                await self._create_model_symlinks(model_paths)

            # December 2025: Create remote symlinks on target nodes
            # (Harvested from deprecated model_distribution_daemon.py)
            if self.config.create_symlinks:
                await self._create_remote_symlinks(model_paths, target_nodes)

            # Emit completion event
            if self.config.emit_completion_event:
                await self._emit_completion_event(DataType.MODEL, items)
        else:
            self._failed_distributions += 1
            # Emit failure event (Dec 2025)
            await self._emit_failure_event(
                DataType.MODEL, items, "Smart distribution failed for models"
            )

    async def _distribute_npz_files(self, items: list[dict[str, Any]]) -> None:
        """Distribute NPZ files to training nodes."""
        target_nodes = await self._get_training_nodes()
        if not target_nodes:
            logger.warning("No training nodes for NPZ distribution")
            return

        for item in items:
            npz_path = item.get("path")
            if not npz_path:
                continue

            npz_file = Path(npz_path)
            if not npz_file.exists():
                logger.error(f"NPZ file not found: {npz_path}")
                continue

            # Validate NPZ structure before distribution
            if self.config.validate_npz_structure and not await self._validate_npz(npz_file):
                logger.error(f"NPZ validation failed: {npz_path}")
                self._failed_distributions += 1
                # Emit failure event for validation failure (Dec 2025)
                await self._emit_failure_event(
                    DataType.NPZ, [item], f"NPZ validation failed: {npz_path}"
                )
                continue

            success = await self._run_smart_distribution(
                [npz_file], target_nodes, DataType.NPZ
            )

            if success:
                self._npz_distributions += 1
                self._successful_distributions += 1
                self._last_sync_time = time.time()

                if self.config.emit_completion_event:
                    await self._emit_completion_event(DataType.NPZ, [item])
            else:
                self._failed_distributions += 1
                # Emit failure event (Dec 2025)
                await self._emit_failure_event(
                    DataType.NPZ, [item], f"Smart distribution failed for NPZ: {npz_path}"
                )

    async def _run_smart_distribution(
        self,
        files: list[Path],
        targets: list[dict[str, Any] | str],
        data_type: DataType,
    ) -> bool:
        """Run distribution with smart transport selection.

        Priority: BitTorrent (for large files) > HTTP > rsync
        """
        success_count = 0
        total_count = len(files) * len(targets)

        for file_path in files:
            if not file_path.exists():
                continue

            file_size = file_path.stat().st_size
            use_bittorrent = (
                self.config.use_bittorrent_for_large_files
                and file_size > self.config.bittorrent_threshold_bytes
            )

            # Compute source checksum
            source_checksum = None
            if self.config.verify_checksums:
                source_checksum = await self._compute_checksum(file_path)

            for target in targets:
                node_host = target if isinstance(target, str) else target.get("host", "")
                node_id = target if isinstance(target, str) else target.get("node_id", node_host)
                start_time = time.time()

                # Try BitTorrent for large files
                if use_bittorrent and await self._distribute_via_bittorrent(file_path, target):
                    success_count += 1
                    self._record_delivery(
                        node_id, node_host, str(file_path), data_type,
                        True, True, time.time() - start_time, "bittorrent"
                    )
                    continue

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
                        success_count += 1
                        self._record_delivery(
                            node_id, node_host, str(file_path), data_type,
                            True, checksum_ok, time.time() - start_time, "http"
                        )
                        continue

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
                        success_count += 1
                        self._record_delivery(
                            node_id, node_host, str(file_path), data_type,
                            True, checksum_ok, time.time() - start_time, "rsync"
                        )
                        continue

                # All methods failed
                self._record_delivery(
                    node_id, node_host, str(file_path), data_type,
                    False, False, time.time() - start_time, "none", "All transport methods failed"
                )

        return success_count > total_count * 0.5

    # =========================================================================
    # Transport Methods
    # =========================================================================

    async def _distribute_via_http(
        self, file_path: Path, target: dict[str, Any] | str, data_type: DataType
    ) -> bool:
        """Distribute file via HTTP upload."""
        try:
            import aiohttp
        except ImportError:
            return False

        host = target if isinstance(target, str) else target.get("host", "")
        endpoint = "models/upload" if data_type == DataType.MODEL else "data/upload"
        url = f"http://{host}:{self.config.http_port}/{endpoint}"

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.http_timeout_seconds)
            ) as session:
                with open(file_path, "rb") as f:
                    file_data = f.read()

                form_data = aiohttp.FormData()
                form_data.add_field(
                    "file", file_data,
                    filename=file_path.name,
                    content_type="application/octet-stream",
                )

                async with session.post(url, data=form_data) as response:
                    if response.status == 200:
                        logger.debug(f"HTTP upload to {host} succeeded: {file_path.name}")
                        return True
                    logger.debug(f"HTTP upload to {host} failed: {response.status}")
                    return False

        except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError, OSError) as e:
            logger.debug(f"HTTP upload to {host} failed: {e}")
            return False

    async def _distribute_via_rsync(
        self, file_path: Path, target: dict[str, Any] | str, data_type: DataType
    ) -> bool:
        """Distribute file via rsync.

        December 27, 2025: Added external storage routing.
        Nodes with use_external_storage: true get files routed to their
        configured storage_paths (e.g., mac-studio -> /Volumes/RingRift-Data).
        """
        if isinstance(target, str):
            host = target
            user = "root"
            remote_path = "~/ringrift/ai-service"
            ssh_key = None
            node_name = None
        else:
            host = target.get("host", "")
            user = target.get("user", target.get("ssh_user", "root"))
            remote_path = target.get("remote_path", "~/ringrift/ai-service")
            ssh_key = target.get("ssh_key")
            node_name = target.get("node_id")

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

        ssh_opts = f"-i {ssh_key}" if ssh_key else ""
        cmd = [
            "rsync", "-avz",
            "-e", f"ssh {ssh_opts} -o StrictHostKeyChecking=no -o ConnectTimeout=10",
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

        except (OSError, asyncio.TimeoutError, asyncio.SubprocessError) as e:
            logger.debug(f"rsync to {host} failed: {e}")
            return False

    async def _distribute_via_bittorrent(
        self, file_path: Path, target: dict[str, Any] | str
    ) -> bool:
        """Distribute file via BitTorrent."""
        try:
            from app.distributed.aria2_transport import Aria2Config, Aria2Transport

            transport = Aria2Transport(Aria2Config(enable_bittorrent=True))
            torrent_path, info_hash, _error = await transport.create_and_register_torrent(file_path)

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

    # =========================================================================
    # External Storage Routing (December 27, 2025)
    # =========================================================================

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

    # =========================================================================
    # Checksum Verification
    # =========================================================================

    async def _compute_checksum(self, path: Path) -> str | None:
        """Compute SHA256 checksum of a file."""
        cache_key = str(path)
        if cache_key in self._checksum_cache:
            return self._checksum_cache[cache_key]

        if not path.exists():
            return None

        try:
            from app.utils.checksum_utils import LARGE_CHUNK_SIZE, compute_file_checksum
            loop = asyncio.get_running_loop()
            checksum = await loop.run_in_executor(
                None,
                lambda: compute_file_checksum(path, chunk_size=LARGE_CHUNK_SIZE),
            )
            self._checksum_cache[cache_key] = checksum
            return checksum
        except (OSError, ValueError, ImportError) as e:
            logger.warning(f"Failed to compute checksum for {path}: {e}")
            return None

    async def _verify_remote_checksum(
        self,
        target: dict[str, Any] | str,
        file_path: Path,
        expected_checksum: str,
        data_type: DataType,
    ) -> bool:
        """Verify checksum on remote node."""
        if isinstance(target, str):
            host = target
            user = "root"
            remote_path = "~/ringrift/ai-service"
        else:
            host = target.get("host", "")
            user = target.get("user", target.get("ssh_user", "root"))
            remote_path = target.get("remote_path", "~/ringrift/ai-service")

        if data_type == DataType.MODEL:
            remote_file = f"{remote_path}/models/{file_path.name}"
        else:
            remote_file = f"{remote_path}/data/training/{file_path.name}"

        cmd = [
            "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
            f"{user}@{host}",
            f"sha256sum {remote_file} 2>/dev/null | cut -d' ' -f1",
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.checksum_timeout_seconds,
            )

            if process.returncode == 0:
                remote_checksum = stdout.decode().strip()
                if remote_checksum == expected_checksum:
                    return True
                logger.warning(f"Checksum mismatch on {host}")
                self._checksum_failures += 1
            return False

        except (OSError, asyncio.TimeoutError, asyncio.SubprocessError) as e:
            logger.debug(f"Checksum verification failed on {host}: {e}")
            return False

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def _validate_npz(self, npz_path: Path) -> bool:
        """Validate NPZ file structure."""
        try:
            from app.coordination.npz_validation import validate_npz_structure
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: validate_npz_structure(npz_path, require_policy=True),
            )
            return result.valid
        except ImportError:
            return True  # Skip validation if module not available
        except (OSError, ValueError, KeyError) as e:
            logger.error(f"NPZ validation failed: {e}")
            return False

    def _get_distribution_targets(self) -> list[str]:
        """Get list of target nodes for model distribution.

        December 2025: Migrated to use cluster_config helpers instead of inline YAML.
        """
        try:
            from app.config.cluster_config import get_active_nodes

            targets = []
            for node in get_active_nodes():
                # Only include nodes with "ready" or "active" status
                if node.status not in ("ready", "active"):
                    continue
                host = node.best_ip
                if host:
                    targets.append(host)
            return targets

        except (ImportError, OSError, KeyError, TypeError) as e:
            logger.warning(f"Failed to get distribution targets: {e}")
            return []

    async def _get_training_nodes(self) -> list[dict[str, Any]]:
        """Get list of training-capable nodes.

        December 2025: Fallback migrated to use cluster_config helpers instead of inline YAML.
        """
        try:
            from app.coordination.sync_router import (
                DataType as SRDataType,
                get_sync_router,
            )
            router = get_sync_router()
            return router.get_sync_targets(SRDataType.NPZ)
        except ImportError:
            # Fallback to cluster_config helpers (Dec 2025)
            try:
                from app.config.cluster_config import get_cluster_nodes

                nodes = []
                for node_id, node in get_cluster_nodes().items():
                    # Training and selfplay roles are eligible for NPZ distribution
                    if node.role in ("training", "selfplay"):
                        nodes.append({
                            "node_id": node_id,
                            "host": node.best_ip,
                            "user": node.ssh_user,
                        })
                return nodes
            except (ImportError, OSError, KeyError, TypeError):
                return []

    async def _create_model_symlinks(self, model_paths: list[Path]) -> None:
        """Create ringrift_best_*.pth symlinks for canonical models."""
        models_dir = ROOT / self.config.models_dir
        created = 0

        for path in model_paths:
            if not path.stem.startswith("canonical_"):
                continue

            config_key = path.stem[len("canonical_"):]
            symlink_name = f"ringrift_best_{config_key}.pth"
            symlink_path = models_dir / symlink_name

            try:
                if symlink_path.exists() or symlink_path.is_symlink():
                    symlink_path.unlink()
                symlink_path.symlink_to(path.name)
                created += 1
            except OSError as e:
                logger.debug(f"Failed to create symlink {symlink_name}: {e}")

        if created > 0:
            logger.info(f"Created {created} model symlinks")

    async def _create_remote_symlinks(
        self,
        model_paths: list[Path],
        target_nodes: list[str],
    ) -> None:
        """Create ringrift_best_*.pth symlinks on remote nodes.

        December 2025: Harvested from deprecated model_distribution_daemon.py.
        After distributing canonical models, create corresponding symlinks on
        each target node so selfplay nodes can find models by config key.
        """
        if not model_paths or not target_nodes:
            return

        # Build symlink commands for canonical models
        symlink_cmds: list[str] = []
        for path in model_paths:
            if not path.stem.startswith("canonical_"):
                continue
            config_key = path.stem[len("canonical_"):]
            canonical_name = path.name
            symlink_name = f"ringrift_best_{config_key}.pth"
            # Create relative symlink in models directory
            # rm -f to avoid "file exists" errors, -sf for force symlink
            symlink_cmds.append(
                f"cd ~/ringrift/ai-service/models && "
                f"rm -f {symlink_name} && "
                f"ln -sf {canonical_name} {symlink_name}"
            )

        if not symlink_cmds:
            return

        # Execute on all target nodes concurrently
        combined_cmd = " && ".join(symlink_cmds)
        created_count = 0
        failed_nodes: list[str] = []

        try:
            from app.core.ssh import get_ssh_client

            async def create_on_node(host: str) -> bool:
                try:
                    client = get_ssh_client(host)
                    result = await client.run_async(combined_cmd, timeout=30)
                    return result.success
                except Exception as e:
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

    async def _emit_completion_event(
        self, data_type: DataType, items: list[dict[str, Any]]
    ) -> None:
        """Emit distribution completion event."""
        try:
            from app.coordination.event_router import emit

            event_type = (
                "MODEL_DISTRIBUTION_COMPLETE"
                if data_type == DataType.MODEL
                else "NPZ_DISTRIBUTION_COMPLETE"
            )

            await emit(
                event_type=event_type,
                data={
                    "items": items,
                    "timestamp": time.time(),
                    "node_id": os.environ.get("RINGRIFT_NODE_ID", "unknown"),
                },
            )
            logger.info(f"Emitted {event_type} event")
        except (ImportError, RuntimeError, TypeError) as e:
            logger.debug(f"Failed to emit completion event: {e}")

    async def _emit_failure_event(
        self, data_type: DataType, items: list[dict[str, Any]], reason: str = ""
    ) -> None:
        """Emit distribution failure event (Dec 2025).

        Args:
            data_type: Type of data that failed to distribute
            items: Items that failed to distribute
            reason: Reason for the failure
        """
        try:
            from app.coordination.event_router import emit

            event_type = (
                "MODEL_DISTRIBUTION_FAILED"
                if data_type == DataType.MODEL
                else "NPZ_DISTRIBUTION_FAILED"
            )

            await emit(
                event_type=event_type,
                data={
                    "items": items,
                    "reason": reason,
                    "timestamp": time.time(),
                    "node_id": os.environ.get("RINGRIFT_NODE_ID", "unknown"),
                },
            )
            logger.warning(f"Emitted {event_type} event: {reason}")
        except (ImportError, RuntimeError, TypeError) as e:
            logger.debug(f"Failed to emit failure event: {e}")

    async def _periodic_sync_check(self) -> None:
        """Periodic check for files that need distribution."""
        # Check for recent models
        models_dir = ROOT / self.config.models_dir
        if models_dir.exists():
            recent_cutoff = time.time() - 3600
            recent_models = [
                m for m in models_dir.glob("canonical_*.pth")
                if m.stat().st_mtime > recent_cutoff
            ]
            if recent_models:
                logger.info(f"Found {len(recent_models)} recent models for periodic sync")
                targets = self._get_distribution_targets()
                if targets:
                    await self._run_smart_distribution(recent_models, targets, DataType.MODEL)

        # Check for recent NPZ files
        training_dir = ROOT / self.config.training_data_dir
        if training_dir.exists():
            recent_cutoff = time.time() - 7200
            recent_npz = [
                f for f in training_dir.glob("*.npz")
                if f.stat().st_mtime > recent_cutoff
            ]
            if recent_npz:
                logger.info(f"Found {len(recent_npz)} recent NPZ files for periodic sync")
                targets = await self._get_training_nodes()
                if targets:
                    await self._run_smart_distribution(recent_npz, targets, DataType.NPZ)

        self._last_sync_time = time.time()


# =============================================================================
# Factory Functions for Backward Compatibility
# =============================================================================

def create_unified_distribution_daemon(
    config: DistributionConfig | None = None,
) -> UnifiedDistributionDaemon:
    """Create a unified distribution daemon.

    Args:
        config: Optional distribution configuration

    Returns:
        UnifiedDistributionDaemon instance
    """
    return UnifiedDistributionDaemon(config)


def create_model_distribution_daemon() -> UnifiedDistributionDaemon:
    """Create a distribution daemon configured for models only.

    DEPRECATED: Use create_unified_distribution_daemon() instead.
    This factory exists for backward compatibility with code that
    expects separate model and NPZ daemons.

    Returns:
        UnifiedDistributionDaemon instance
    """
    import warnings
    warnings.warn(
        "create_model_distribution_daemon() is deprecated. "
        "Use create_unified_distribution_daemon() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return UnifiedDistributionDaemon()


def create_npz_distribution_daemon() -> UnifiedDistributionDaemon:
    """Create a distribution daemon configured for NPZ files only.

    DEPRECATED: Use create_unified_distribution_daemon() instead.
    This factory exists for backward compatibility with code that
    expects separate model and NPZ daemons.

    Returns:
        UnifiedDistributionDaemon instance
    """
    import warnings
    warnings.warn(
        "create_npz_distribution_daemon() is deprecated. "
        "Use create_unified_distribution_daemon() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return UnifiedDistributionDaemon()


# =============================================================================
# Helper Functions for Model Availability Checking
# =============================================================================


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
        available = await wait_for_model_distribution("hex8", 2, timeout=300)
        if not available:
            logger.warning("Model distribution timed out, using fallback")
    """
    import asyncio
    import time

    config_key = f"{board_type}_{num_players}p"
    model_name = f"canonical_{config_key}.pth"
    models_dir = ROOT / "models"
    model_path = models_dir / model_name

    # Check if model already exists locally
    if model_path.exists():
        logger.debug(f"[ModelDistribution] Model already available: {model_path}")
        return True

    # Wait for MODEL_DISTRIBUTION_COMPLETE event
    logger.info(
        f"[ModelDistribution] Waiting for {model_name} distribution "
        f"(timeout: {timeout}s)..."
    )

    distribution_event = asyncio.Event()

    def on_distribution_complete(event: Any) -> None:
        """Handle MODEL_DISTRIBUTION_COMPLETE event."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            models = payload.get("models", [])
            for model_info in models:
                model_path_str = model_info.get("model_path", "")
                if model_name in model_path_str:
                    logger.info(
                        f"[ModelDistribution] Received {model_name} distribution complete"
                    )
                    distribution_event.set()
                    return
        except (AttributeError, KeyError, TypeError) as e:
            logger.warning(f"[ModelDistribution] Error handling event: {e}")

    # Subscribe to distribution events
    try:
        from app.coordination.event_router import DataEventType, subscribe

        subscribe(DataEventType.MODEL_DISTRIBUTION_COMPLETE, on_distribution_complete)

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

        logger.warning(
            f"[ModelDistribution] Model {model_name} not found after {timeout}s"
        )
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


# =============================================================================
# Standalone Entry Point
# =============================================================================


async def main() -> None:
    """Run daemon standalone."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    daemon = UnifiedDistributionDaemon()
    try:
        await daemon.start()
    except KeyboardInterrupt:
        await daemon.stop()


if __name__ == "__main__":
    asyncio.run(main())
