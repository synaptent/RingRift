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
import subprocess
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

# December 29, 2025: Circuit breaker integration for distribution reliability
try:
    from app.distributed.circuit_breaker import CircuitBreakerRegistry, CircuitState
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False
    CircuitBreakerRegistry = None  # type: ignore
    CircuitState = None  # type: ignore

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

# December 2025: Renamed from DataType to DistributionDataType to avoid collision
# Import from canonical source for new code:
#   from app.coordination.enums import DistributionDataType
from app.coordination.enums import DistributionDataType

# Backward-compatible alias (deprecated, remove Q2 2026)
DataType = DistributionDataType


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

        # December 29, 2025: Background model prefetch during training
        # Tracks checkpoints already prefetched to avoid redundant work
        self._prefetched_checkpoints: set[str] = set()
        self._prefetch_threshold: float = 0.80  # Start prefetch at 80% progress
        self._prefetch_enabled: bool = True

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
        """Get current daemon metrics for monitoring and debugging.

        Returns:
            Dictionary containing:
            - name, status, uptime metrics
            - events_processed, errors_count, last_error
            - Distribution stats (successful, failed, checksum failures)
            - Model and NPZ distribution counts
            - Prefetch configuration and status
        """
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
            # December 29, 2025: Background prefetch metrics
            "prefetch_enabled": self._prefetch_enabled,
            "prefetch_threshold": self._prefetch_threshold,
            "prefetched_checkpoints_count": len(self._prefetched_checkpoints),
        }

    def health_check(self) -> HealthCheckResult:
        """Check daemon health status for DaemonManager integration.

        Returns:
            HealthCheckResult indicating:
            - UNHEALTHY if daemon is in error state
            - DEGRADED if high failure rate, pending buildup, or checksum failures
            - HEALTHY with distribution metrics otherwise
        """
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
    # Distribution Verification (December 2025 - Phase 3)
    # =========================================================================

    async def verify_distribution(
        self,
        model_path: str,
        min_nodes: int | None = None,
    ) -> tuple[bool, int]:
        """Check if model is distributed to at least min_nodes.

        Uses ClusterManifest to query model locations across the cluster.
        This is the core verification method used before promotion/evaluation.

        Args:
            model_path: Path to the model file (can be relative or absolute)
            min_nodes: Minimum number of nodes required (default from DistributionDefaults)

        Returns:
            Tuple of (success, actual_node_count)

        Example:
            success, count = await daemon.verify_distribution("models/canonical_hex8_2p.pth")
            if not success:
                logger.warning(f"Model only on {count} nodes, distribution incomplete")
        """
        try:
            from app.config.coordination_defaults import DistributionDefaults
            from app.distributed.cluster_manifest import get_cluster_manifest

            if min_nodes is None:
                min_nodes = DistributionDefaults.MIN_NODES_FOR_PROMOTION

            manifest = get_cluster_manifest()

            # Normalize model path to just filename for matching
            model_name = Path(model_path).name
            locations = manifest.find_model(model_name)

            # Get unique nodes
            unique_nodes = {loc.node_id for loc in locations}
            actual_count = len(unique_nodes)

            success = actual_count >= min_nodes

            logger.debug(
                f"[DistributionVerification] Model {model_name}: "
                f"{actual_count}/{min_nodes} nodes (success={success})"
            )

            return (success, actual_count)

        except ImportError as e:
            logger.warning(f"[DistributionVerification] ClusterManifest unavailable: {e}")
            return (False, 0)
        except (OSError, RuntimeError, ValueError) as e:
            logger.error(f"[DistributionVerification] Error checking distribution: {e}")
            return (False, 0)

    def get_model_availability_score(self, model_path: str) -> float:
        """Return 0-1 score for how well model is distributed.

        Score = (nodes with model) / (total GPU nodes in cluster)

        Args:
            model_path: Path to the model file

        Returns:
            Float between 0.0 and 1.0 indicating distribution coverage
        """
        try:
            from app.config.cluster_config import get_gpu_nodes
            from app.distributed.cluster_manifest import get_cluster_manifest

            manifest = get_cluster_manifest()
            model_name = Path(model_path).name
            locations = manifest.find_model(model_name)

            # Get unique nodes with model
            nodes_with_model = len({loc.node_id for loc in locations})

            # Get total GPU nodes
            try:
                gpu_nodes = get_gpu_nodes()
                total_gpu_nodes = len(gpu_nodes)
            except (ImportError, RuntimeError) as e:
                logger.debug(f"[DistributionVerification] Could not get GPU nodes: {e}")
                # Fallback to reasonable estimate
                total_gpu_nodes = 30

            if total_gpu_nodes == 0:
                return 0.0

            score = nodes_with_model / total_gpu_nodes
            return min(1.0, score)

        except ImportError:
            return 0.0
        except (OSError, RuntimeError, ValueError) as e:
            logger.error(f"[DistributionVerification] Error calculating score: {e}")
            return 0.0

    async def wait_for_adequate_distribution(
        self,
        model_path: str,
        min_nodes: int | None = None,
        timeout: float | None = None,
    ) -> tuple[bool, int]:
        """Wait for model to be distributed to adequate number of nodes.

        Polls verify_distribution() periodically until success or timeout.

        Args:
            model_path: Path to the model file
            min_nodes: Minimum nodes required (default from DistributionDefaults)
            timeout: Maximum wait time in seconds (default from DistributionDefaults)

        Returns:
            Tuple of (success, final_node_count)
        """
        try:
            from app.config.coordination_defaults import DistributionDefaults

            if min_nodes is None:
                min_nodes = DistributionDefaults.MIN_NODES_FOR_PROMOTION
            if timeout is None:
                timeout = DistributionDefaults.DISTRIBUTION_TIMEOUT_SECONDS

            retry_interval = DistributionDefaults.DISTRIBUTION_RETRY_INTERVAL
            start_time = time.time()

            model_name = Path(model_path).name
            logger.info(
                f"[DistributionVerification] Waiting for {model_name} "
                f"to reach {min_nodes} nodes (timeout: {timeout}s)"
            )

            while (time.time() - start_time) < timeout:
                success, count = await self.verify_distribution(model_path, min_nodes)
                if success:
                    logger.info(
                        f"[DistributionVerification] {model_name} reached {count} nodes"
                    )
                    return (True, count)

                await asyncio.sleep(retry_interval)

            # Final check
            success, count = await self.verify_distribution(model_path, min_nodes)
            if not success:
                logger.warning(
                    f"[DistributionVerification] Timeout: {model_name} only on {count}/{min_nodes} nodes"
                )
                # Emit DISTRIBUTION_INCOMPLETE event
                try:
                    from app.distributed.data_events import emit_event

                    emit_event("DISTRIBUTION_INCOMPLETE", {
                        "model_path": model_path,
                        "required_nodes": min_nodes,
                        "actual_nodes": count,
                        "timeout_seconds": timeout,
                    })
                except ImportError:
                    pass

            return (success, count)

        except ImportError as e:
            logger.warning(f"[DistributionVerification] Config unavailable: {e}")
            return (False, 0)

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

        # Main loop - Dec 2025 fix: Use while True to prevent natural exit
        # which triggers daemon restart loop. Only exit via CancelledError.
        try:
            while True:
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

                except (OSError, RuntimeError, ValueError) as e:
                    logger.error(f"Error in distribution daemon loop: {e}")
                    self._errors_count += 1
                    self._last_error = str(e)
                    await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            logger.info("UnifiedDistributionDaemon cancelled")
        finally:
            self._running = False
            self._coordinator_status = CoordinatorStatus.STOPPED
            unregister_coordinator(self.name)
            logger.info("UnifiedDistributionDaemon stopped")

    async def stop(self) -> None:
        """Stop the daemon gracefully."""
        if self._coordinator_status == CoordinatorStatus.STOPPED:
            return

        self._coordinator_status = CoordinatorStatus.STOPPING
        self._running = False
        unregister_coordinator(self.name)
        self._coordinator_status = CoordinatorStatus.STOPPED

    def _subscribe_to_events(self, max_retries: int = 3) -> None:
        """Subscribe to MODEL_PROMOTED and NPZ_EXPORT_COMPLETE events.

        December 28, 2025: Added retry logic with exponential backoff for
        robustness during startup when event bus may not be ready.

        Args:
            max_retries: Maximum retry attempts before falling back to polling
        """
        import time as time_module

        for attempt in range(max_retries):
            try:
                from app.coordination.event_router import DataEventType, subscribe

                # Model events
                subscribe(DataEventType.MODEL_PROMOTED, self._on_model_promoted)
                subscribe(DataEventType.MODEL_UPDATED, self._on_model_updated)
                subscribe(DataEventType.MODEL_DISTRIBUTION_STARTED, self._on_model_distribution_started)
                subscribe(DataEventType.MODEL_DISTRIBUTION_FAILED, self._on_model_distribution_failed)
                logger.info("Subscribed to MODEL_PROMOTED, MODEL_UPDATED events")

                # December 29, 2025: Background model prefetch during training
                # Subscribe to TRAINING_PROGRESS to start distributing checkpoints
                # before training completes, reducing post-training latency
                subscribe(DataEventType.TRAINING_PROGRESS, self._on_training_progress_for_prefetch)
                logger.info("Subscribed to TRAINING_PROGRESS for background prefetch")

                # December 29, 2025: Re-distribute models blocked from evaluation
                # When evaluation is blocked due to insufficient model distribution,
                # prioritize re-distribution of that model
                subscribe(DataEventType.MODEL_EVALUATION_BLOCKED, self._on_model_evaluation_blocked)
                logger.info("Subscribed to MODEL_EVALUATION_BLOCKED for re-distribution")

                # NPZ events
                try:
                    from app.coordination.event_router import StageEvent
                    subscribe(StageEvent.NPZ_EXPORT_COMPLETE, self._on_npz_exported)
                    logger.info("Subscribed to NPZ_EXPORT_COMPLETE events")
                except (ImportError, AttributeError):
                    subscribe("npz_export_complete", self._on_npz_exported)
                    logger.info("Subscribed to npz_export_complete string events")

                # Success - exit retry loop
                return

            except ImportError as e:
                if attempt < max_retries - 1:
                    backoff = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning(
                        f"Event system not ready, retry {attempt + 1}/{max_retries} "
                        f"in {backoff}s: {e}"
                    )
                    time_module.sleep(backoff)
                else:
                    logger.warning(
                        f"Event system not available after {max_retries} attempts ({e}), "
                        "will poll for new files"
                    )
            except Exception as e:
                logger.warning(f"Subscription failed: {e}, will poll for new files")
                return

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
        """Handle MODEL_DISTRIBUTION_FAILED event (external distribution) with retry.

        December 2025: Added retry logic for distribution failures.
        Re-enqueues failed distributions up to max_retries times.
        """
        payload = getattr(event, "payload", event) if hasattr(event, "payload") else event
        error = payload.get("error", "unknown_error")
        model_name = payload.get("model_name", "")
        expected_path = payload.get("expected_path", "")
        retry_count = payload.get("retry_count", 0)

        self._errors_count += 1
        self._last_error = str(error)

        # Maximum 3 retries per distribution
        max_retries = 3
        if retry_count >= max_retries:
            logger.error(
                f"MODEL_DISTRIBUTION_FAILED (final, {retry_count} retries): "
                f"{model_name} - {error}"
            )
            return

        # Re-enqueue for retry with exponential backoff
        if model_name and expected_path:
            retry_delay = 2 ** retry_count * 10  # 10s, 20s, 40s
            item = {
                "data_type": DataType.MODEL,
                "path": expected_path,
                "model_name": model_name,
                "retry_count": retry_count + 1,
                "retry_after": time.time() + retry_delay,
                "timestamp": time.time(),
            }
            self._enqueue_item(item)
            logger.warning(
                f"MODEL_DISTRIBUTION_FAILED (retry {retry_count + 1}/{max_retries}): "
                f"{model_name} - {error} - retrying in {retry_delay}s"
            )
        else:
            logger.warning(f"MODEL_DISTRIBUTION_FAILED (no retry - missing info): {error}")

    def _on_model_evaluation_blocked(self, event: dict[str, Any] | Any) -> None:
        """Handle MODEL_EVALUATION_BLOCKED event - prioritize model re-distribution.

        December 29, 2025: When evaluation is blocked because a model isn't
        distributed to enough nodes, prioritize re-distributing that model.

        Args:
            event: Event with model_path, required_nodes, actual_nodes
        """
        payload = getattr(event, "payload", event) if hasattr(event, "payload") else event
        model_path = payload.get("model_path", "")
        required_nodes = payload.get("required_nodes", 0)
        actual_nodes = payload.get("actual_nodes", 0)
        reason = payload.get("reason", "")

        if not model_path:
            logger.warning("MODEL_EVALUATION_BLOCKED received without model_path")
            return

        logger.info(
            f"MODEL_EVALUATION_BLOCKED: {model_path} has {actual_nodes}/{required_nodes} nodes "
            f"(reason: {reason}), prioritizing re-distribution"
        )

        # Enqueue for priority distribution (timestamp in past = high priority)
        item = {
            "data_type": DataType.MODEL,
            "path": model_path,
            "priority": True,  # Mark as priority
            "reason": "evaluation_blocked",
            "required_nodes": required_nodes,
            "actual_nodes": actual_nodes,
            "timestamp": time.time() - 3600,  # Priority: 1 hour ago
        }
        self._enqueue_item(item)

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

    def _on_training_progress_for_prefetch(self, event: dict[str, Any] | Any) -> None:
        """Handle TRAINING_PROGRESS to start background model prefetch.

        December 29, 2025: Reduces post-training latency by distributing
        checkpoints to other nodes while training is still running.
        When training progress exceeds the threshold (default 80%), we start
        prefetching the latest checkpoint so it's already distributed when
        training completes.

        Args:
            event: Event containing training progress and checkpoint path

        Expected payload:
            - epochs_completed: int - Current epoch number
            - total_epochs: int - Total epochs planned
            - checkpoint_path: str - Path to latest checkpoint (optional)
            - config_key: str - Training config key (e.g., "hex8_2p")
            - node_id: str - Node running training
        """
        if not self._prefetch_enabled:
            return

        payload = getattr(event, "payload", event) if hasattr(event, "payload") else event

        epochs_completed = payload.get("epochs_completed", 0)
        total_epochs = payload.get("total_epochs", 0)
        checkpoint_path = payload.get("checkpoint_path", payload.get("best_checkpoint_path"))
        config_key = payload.get("config_key", "")

        # Skip if no checkpoint path or no progress info
        if not checkpoint_path or total_epochs <= 0:
            return

        # Calculate progress
        progress = epochs_completed / total_epochs

        # Only prefetch if progress exceeds threshold
        if progress < self._prefetch_threshold:
            return

        # Skip if already prefetched this checkpoint
        if checkpoint_path in self._prefetched_checkpoints:
            return

        # Mark as prefetched to avoid redundant work
        self._prefetched_checkpoints.add(checkpoint_path)

        # Enqueue for distribution with lower priority (prefetch flag)
        item = {
            "data_type": DataType.MODEL,
            "path": checkpoint_path,
            "config_key": config_key,
            "is_prefetch": True,  # Flag to indicate prefetch vs final distribution
            "training_progress": progress,
            "timestamp": time.time(),
        }
        logger.info(
            f"[Prefetch] Training {config_key} at {progress:.0%}, "
            f"prefetching checkpoint: {checkpoint_path}"
        )
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
        """Process all pending distribution items.

        December 2025: Added retry_after support for exponential backoff retries.
        Items with retry_after > now are kept in queue until their delay expires.
        """
        async with self._sync_lock:
            with self._pending_lock:
                if not self._pending_items:
                    return

                # Separate items ready to process from items still waiting for retry
                now = time.time()
                ready_items = []
                waiting_items = []
                for item in self._pending_items:
                    retry_after = item.get("retry_after", 0)
                    if retry_after <= now:
                        ready_items.append(item)
                    else:
                        waiting_items.append(item)

                # Keep waiting items in queue, process ready items
                self._pending_items = waiting_items

                if not ready_items:
                    return
                items = ready_items

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

        December 29, 2025: Refactored for parallel distribution to all nodes.
        Uses asyncio.gather with semaphore to limit concurrent uploads.
        Expected 5-10x speedup for multi-node clusters.
        """
        success_count = 0
        total_count = len(files) * len(targets)

        # Semaphore to limit concurrent uploads (default 5, configurable)
        max_concurrent = self.config.http_concurrent_uploads
        semaphore = asyncio.Semaphore(max_concurrent)

        for file_path in files:
            if not file_path.exists():
                continue

            file_size = file_path.stat().st_size
            use_bittorrent = (
                self.config.use_bittorrent_for_large_files
                and file_size > self.config.bittorrent_threshold_bytes
            )

            # Compute source checksum once per file (not per target)
            source_checksum = None
            if self.config.verify_checksums:
                source_checksum = await self._compute_checksum(file_path)

            # December 29, 2025: Parallel distribution to all targets
            async def distribute_to_target(target: dict[str, Any] | str) -> bool:
                async with semaphore:
                    return await self._distribute_file_to_target(
                        file_path, target, data_type, use_bittorrent, source_checksum
                    )

            # Run distributions in parallel
            results = await asyncio.gather(
                *[distribute_to_target(target) for target in targets],
                return_exceptions=True
            )

            # Count successes
            for result in results:
                if result is True:
                    success_count += 1
                elif isinstance(result, Exception):
                    logger.warning(f"Distribution failed with exception: {result}")

        return success_count > total_count * 0.5

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

        except (OSError, asyncio.TimeoutError, subprocess.SubprocessError) as e:
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
    # Remote Path Detection (December 28, 2025)
    # =========================================================================

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

    async def _get_remote_path(
        self,
        target: dict[str, Any] | str,
    ) -> tuple[str, str, str, str | None]:
        """Get remote path for a target, discovering if needed.

        Args:
            target: Target host as string or dict with host/user/remote_path

        Returns:
            Tuple of (host, user, remote_path, ssh_key)
        """
        if isinstance(target, str):
            host = target
            user = "root"
            explicit_path = None
            ssh_key = None
        else:
            host = target.get("host", "")
            user = target.get("user", target.get("ssh_user", "root"))
            explicit_path = target.get("remote_path")
            ssh_key = target.get("ssh_key")

        # Use explicit path if provided, otherwise discover
        if explicit_path:
            remote_path = explicit_path
        else:
            remote_path = await self._discover_remote_path(host, user, ssh_key)

        return host, user, remote_path, ssh_key

    def clear_remote_path_cache(self, host: str | None = None) -> None:
        """Clear cached remote path(s).

        Args:
            host: Specific host to clear, or None to clear all
        """
        with _remote_path_cache_lock:
            if host:
                _remote_path_cache.pop(host, None)
            else:
                _remote_path_cache.clear()

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
        """Verify checksum on remote node.

        December 28, 2025: Uses remote path discovery for consistent paths.
        """
        # December 28, 2025: Use path discovery with fallback
        host, user, remote_path, ssh_key = await self._get_remote_path(target)

        if data_type == DataType.MODEL:
            remote_file = f"{remote_path}/models/{file_path.name}"
        else:
            remote_file = f"{remote_path}/data/training/{file_path.name}"

        ssh_opts = ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10"]
        if ssh_key:
            ssh_opts.extend(["-i", ssh_key])

        cmd = ["ssh"] + ssh_opts + [
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

        except (OSError, asyncio.TimeoutError, subprocess.SubprocessError) as e:
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
                    result = await client.run_async(combined_cmd, timeout=30)
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

    def _record_circuit_breaker_success(self, node_id: str) -> None:
        """Record successful distribution in circuit breaker (December 29, 2025).

        This resets the failure count for the node, allowing it to be
        used for future distributions even after previous failures.
        """
        if not CIRCUIT_BREAKER_AVAILABLE or not CircuitBreakerRegistry:
            return
        try:
            registry = CircuitBreakerRegistry.get_instance()
            breaker = registry.get_breaker(f"distribution:{node_id}")
            breaker.record_success(f"distribution:{node_id}")
        except Exception as e:
            logger.debug(f"Circuit breaker success recording failed for {node_id}: {e}")

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


def _is_valid_model_file(path: str | Path) -> bool:
    """Check if a model file exists and appears to be valid.

    Validates that:
    1. File exists
    2. File is at least 1MB (smaller files are likely corrupted)
    3. File starts with PK header (zip/pth format)

    Args:
        path: Path to the model file

    Returns:
        True if file exists and appears valid, False otherwise
    """
    path = Path(path) if isinstance(path, str) else path

    if not path.exists():
        return False

    try:
        size = path.stat().st_size
        if size < 1_000_000:  # < 1MB likely corrupted or incomplete
            logger.debug(f"[ModelValidation] File too small: {path} ({size} bytes)")
            return False

        # Check for zip/pth magic bytes (PyTorch saves as zip archives)
        with open(path, "rb") as f:
            magic = f.read(8)
            if magic[:4] != b"PK\x03\x04":  # ZIP file magic number
                logger.debug(f"[ModelValidation] Invalid magic bytes: {path}")
                return False

        return True
    except (OSError, IOError) as e:
        logger.debug(f"[ModelValidation] Error checking file {path}: {e}")
        return False


async def _emit_distribution_failed_event(
    model_name: str,
    expected_path: str | Path,
    timeout_seconds: float,
    reason: str = "timeout",
) -> None:
    """Emit MODEL_DISTRIBUTION_FAILED event.

    Args:
        model_name: Name of the model that failed to distribute
        expected_path: Expected path where model should be
        timeout_seconds: How long we waited before giving up
        reason: Reason for the failure
    """
    try:
        from app.coordination.event_router import emit

        await emit(
            event_type="MODEL_DISTRIBUTION_FAILED",
            data={
                "model_name": model_name,
                "expected_path": str(expected_path),
                "timeout_seconds": timeout_seconds,
                "reason": reason,
                "timestamp": time.time(),
                "node_id": os.environ.get("RINGRIFT_NODE_ID", "unknown"),
            },
        )
        logger.warning(
            f"[ModelDistribution] Emitted MODEL_DISTRIBUTION_FAILED: {model_name} "
            f"(reason: {reason})"
        )
    except (ImportError, RuntimeError, TypeError) as e:
        logger.debug(f"[ModelDistribution] Failed to emit failure event: {e}")


async def wait_for_model_distribution(
    board_type: str,
    num_players: int,
    timeout: float = 300.0,
    disk_check_interval: float = 30.0,
) -> bool:
    """Wait for a model to be distributed to this node.

    This function waits for MODEL_DISTRIBUTION_COMPLETE event or checks
    if the model already exists locally. Use this before selfplay starts
    to avoid race conditions where nodes try to load models before
    distribution completes.

    Features (Dec 2025 improvements):
        - Initial check for existing valid model on disk
        - Periodic disk checks every 30s during wait (catches rsync from other sources)
        - Fallback to local disk on timeout (if model appeared but event was missed)
        - Emits MODEL_DISTRIBUTION_FAILED event on final timeout
        - Validates model file integrity (size > 1MB, valid zip header)

    Args:
        board_type: Board type (hex8, square8, etc.)
        num_players: Number of players (2, 3, 4)
        timeout: Maximum time to wait in seconds (default: 300)
        disk_check_interval: How often to check disk during wait (default: 30)

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

    # Check if model already exists locally and is valid
    if _is_valid_model_file(model_path):
        logger.debug(f"[ModelDistribution] Valid model already available: {model_path}")
        return True

    # Also check if file exists but may not be valid yet (still downloading)
    if model_path.exists():
        logger.debug(
            f"[ModelDistribution] Model file exists but may be incomplete: {model_path}"
        )

    # Wait for MODEL_DISTRIBUTION_COMPLETE event
    logger.info(
        f"[ModelDistribution] Waiting for {model_name} distribution "
        f"(timeout: {timeout}s, disk_check_interval: {disk_check_interval}s)..."
    )

    distribution_event = asyncio.Event()
    disk_found_event = asyncio.Event()

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

    async def periodic_disk_check() -> None:
        """Periodically check if model appeared on disk (e.g., via rsync)."""
        while not distribution_event.is_set() and not disk_found_event.is_set():
            await asyncio.sleep(disk_check_interval)
            if _is_valid_model_file(model_path):
                logger.info(
                    f"[ModelDistribution] Model {model_name} found on disk during wait "
                    "(may have been synced via rsync from another source)"
                )
                disk_found_event.set()
                return

    # Subscribe to distribution events
    try:
        from app.coordination.event_router import DataEventType, subscribe

        subscribe(DataEventType.MODEL_DISTRIBUTION_COMPLETE, on_distribution_complete)

        # Start periodic disk check task
        disk_check_task = asyncio.create_task(periodic_disk_check())

        try:
            # Wait for either event or disk discovery
            start_time = time.time()
            while time.time() - start_time < timeout:
                # Check if either event fired
                if distribution_event.is_set():
                    disk_check_task.cancel()
                    logger.info(
                        f"[ModelDistribution] Model {model_name} is now available (via event)"
                    )
                    return True

                if disk_found_event.is_set():
                    disk_check_task.cancel()
                    logger.info(
                        f"[ModelDistribution] Model {model_name} is now available (via disk check)"
                    )
                    return True

                # Wait a short time before checking again
                try:
                    await asyncio.wait_for(
                        asyncio.gather(
                            distribution_event.wait(),
                            disk_found_event.wait(),
                            return_exceptions=True,
                        ),
                        timeout=min(5.0, timeout - (time.time() - start_time)),
                    )
                except asyncio.TimeoutError:
                    continue  # Keep waiting

            # Timeout reached - cancel disk check task
            disk_check_task.cancel()

            # FALLBACK: Check if model appeared on disk after timeout
            if _is_valid_model_file(model_path):
                logger.info(
                    f"[ModelDistribution] Using existing local model after distribution "
                    f"timeout: {model_path}"
                )
                return True

            # Final failure - emit event
            logger.warning(
                f"[ModelDistribution] Timed out waiting for {model_name} "
                f"after {timeout}s and no local fallback found"
            )
            await _emit_distribution_failed_event(
                model_name=model_name,
                expected_path=model_path,
                timeout_seconds=timeout,
                reason=f"Distribution timeout after {timeout}s, no local fallback",
            )
            return False

        except asyncio.CancelledError:
            disk_check_task.cancel()
            raise

    except ImportError:
        logger.debug("[ModelDistribution] Event system not available")
        # Without event system, just check if file appeared
        start_time = time.time()
        while time.time() - start_time < timeout:
            if _is_valid_model_file(model_path):
                logger.info(
                    f"[ModelDistribution] Valid model {model_name} found on disk"
                )
                return True
            await asyncio.sleep(disk_check_interval)

        # Final fallback check
        if _is_valid_model_file(model_path):
            logger.info(
                f"[ModelDistribution] Using existing local model after wait: {model_path}"
            )
            return True

        logger.warning(
            f"[ModelDistribution] Model {model_name} not found after {timeout}s"
        )
        # Emit failure event (best effort without event system)
        try:
            await _emit_distribution_failed_event(
                model_name=model_name,
                expected_path=model_path,
                timeout_seconds=timeout,
                reason=f"Distribution timeout after {timeout}s (event system unavailable)",
            )
        except (ImportError, RuntimeError):
            pass  # Event system truly unavailable
        return False


def check_model_availability(
    board_type: str,
    num_players: int,
    validate: bool = True,
) -> bool:
    """Synchronously check if a model is available locally.

    Args:
        board_type: Board type (hex8, square8, etc.)
        num_players: Number of players (2, 3, 4)
        validate: If True, also validates file integrity (size > 1MB, valid zip header)
                  If False, only checks existence (backward compatible behavior)

    Returns:
        True if model exists locally (and is valid if validate=True), False otherwise

    Example:
        if not check_model_availability("hex8", 2):
            logger.warning("Model not available yet")

        # Quick existence check only
        if check_model_availability("hex8", 2, validate=False):
            logger.info("Model file exists (may still be downloading)")
    """
    config_key = f"{board_type}_{num_players}p"
    model_name = f"canonical_{config_key}.pth"
    models_dir = ROOT / "models"
    model_path = models_dir / model_name

    # Also check for symlinks (ringrift_best_*.pth)
    symlink_name = f"ringrift_best_{config_key}.pth"
    symlink_path = models_dir / symlink_name

    if validate:
        # Check for valid model file (preferred)
        if _is_valid_model_file(model_path):
            return True
        # Fall back to symlink
        if _is_valid_model_file(symlink_path):
            return True
        return False
    else:
        # Backward-compatible existence check only
        return model_path.exists() or symlink_path.exists()


def is_valid_model_file(path: str | Path) -> bool:
    """Check if a model file exists and appears to be valid.

    Public wrapper for model validation. Validates that:
    1. File exists
    2. File is at least 1MB (smaller files are likely corrupted)
    3. File starts with PK header (zip/pth format)

    Args:
        path: Path to the model file

    Returns:
        True if file exists and appears valid, False otherwise

    Example:
        if is_valid_model_file("models/canonical_hex8_2p.pth"):
            logger.info("Model is valid and ready to use")
    """
    return _is_valid_model_file(path)


# =============================================================================
# Remote Path Discovery Helpers (December 28, 2025)
# =============================================================================


def get_remote_path_patterns() -> list[str]:
    """Get the list of known remote path patterns.

    Returns:
        List of path patterns that are probed when discovering remote paths.
        Patterns are tried in order, with provider-specific paths first.

    Example:
        patterns = get_remote_path_patterns()
        # ['/workspace/ringrift/ai-service', '~/ringrift/ai-service', ...]
    """
    return REMOTE_PATH_PATTERNS.copy()


def get_cached_remote_path(host: str) -> str | None:
    """Get the cached remote path for a host, if any.

    Args:
        host: Remote host IP or hostname

    Returns:
        Cached path string, or None if not cached.

    Example:
        path = get_cached_remote_path("runpod-h100")
        if path:
            print(f"Using cached path: {path}")
    """
    with _remote_path_cache_lock:
        return _remote_path_cache.get(host)


def clear_remote_path_cache(host: str | None = None) -> None:
    """Clear cached remote path(s).

    Args:
        host: Specific host to clear, or None to clear all.

    Example:
        # Clear cache for specific host after reconfiguration
        clear_remote_path_cache("runpod-h100")

        # Clear all cache entries
        clear_remote_path_cache()
    """
    with _remote_path_cache_lock:
        if host:
            _remote_path_cache.pop(host, None)
        else:
            _remote_path_cache.clear()


def get_all_cached_remote_paths() -> dict[str, str]:
    """Get all cached remote paths.

    Returns:
        Dict mapping host to cached remote path.

    Example:
        cache = get_all_cached_remote_paths()
        for host, path in cache.items():
            print(f"{host}: {path}")
    """
    with _remote_path_cache_lock:
        return _remote_path_cache.copy()


# =============================================================================
# Distribution Verification Convenience Functions (December 2025 - Phase 3)
# =============================================================================


async def verify_model_distribution(
    model_path: str,
    min_nodes: int | None = None,
) -> tuple[bool, int]:
    """Module-level convenience function for verifying model distribution.

    Creates a temporary daemon instance to perform verification.
    For repeated checks, prefer creating a daemon instance directly.

    Args:
        model_path: Path to the model file
        min_nodes: Minimum nodes required (default from DistributionDefaults)

    Returns:
        Tuple of (success, actual_node_count)

    Example:
        success, count = await verify_model_distribution("models/canonical_hex8_2p.pth")
        if not success:
            print(f"Warning: Model only on {count} nodes")
    """
    daemon = UnifiedDistributionDaemon()
    return await daemon.verify_distribution(model_path, min_nodes)


def get_model_availability_score(model_path: str) -> float:
    """Module-level convenience function for checking model availability.

    Args:
        model_path: Path to the model file

    Returns:
        Float between 0.0 and 1.0 indicating distribution coverage

    Example:
        score = get_model_availability_score("models/canonical_hex8_2p.pth")
        if score < 0.3:
            print("Warning: Model poorly distributed")
    """
    daemon = UnifiedDistributionDaemon()
    return daemon.get_model_availability_score(model_path)


async def wait_for_model_availability(
    model_path: str,
    min_nodes: int | None = None,
    timeout: float | None = None,
) -> tuple[bool, int]:
    """Wait for a model to be distributed to adequate nodes.

    Module-level convenience function that polls until success or timeout.

    Args:
        model_path: Path to the model file
        min_nodes: Minimum nodes required (default from DistributionDefaults)
        timeout: Maximum wait time in seconds (default from DistributionDefaults)

    Returns:
        Tuple of (success, final_node_count)

    Example:
        success, count = await wait_for_model_availability(
            "models/canonical_hex8_2p.pth",
            min_nodes=5,
            timeout=300,
        )
        if success:
            print(f"Model distributed to {count} nodes, proceeding with evaluation")
    """
    daemon = UnifiedDistributionDaemon()
    return await daemon.wait_for_adequate_distribution(model_path, min_nodes, timeout)


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
