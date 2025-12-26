"""Model Distribution Daemon - Automatic model sync after promotion.

This daemon watches for MODEL_PROMOTED events and automatically distributes
promoted models to all cluster nodes, solving the gap where models would
only exist on the node where training completed.

Architecture:
    1. Subscribes to MODEL_PROMOTED events from event_router
    2. Uses sync_models.py --distribute for reliable multi-node sync
    3. Tracks distribution status in ClusterManifest
    4. Emits MODEL_DISTRIBUTION_COMPLETE event when done

Usage:
    # As standalone daemon
    python -m app.coordination.model_distribution_daemon

    # Via DaemonManager
    manager.register_factory(DaemonType.MODEL_DISTRIBUTION, daemon.run)

Configuration:
    Uses distributed_hosts.yaml for target nodes.
    See config/promotion_daemon.yaml for promotion thresholds.
"""

from __future__ import annotations

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
    poll_interval_seconds: float = 60.0
    models_dir: str = "models"

    # HTTP distribution settings (December 2025 - Phase 14)
    # HTTP is faster than rsync for model streaming
    use_http_distribution: bool = True  # Prefer HTTP when available
    http_port: int = 8767  # Port for model upload endpoint
    http_timeout_seconds: float = 60.0  # Timeout per node for HTTP upload
    http_concurrent_uploads: int = 5  # Max concurrent HTTP uploads
    fallback_to_rsync: bool = True  # Fallback to rsync if HTTP fails


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

        # Healthy
        return HealthCheckResult(
            healthy=True,
            status=self._coordinator_status,
            details={
                "uptime_seconds": self.uptime_seconds,
                "successful_distributions": self._successful_distributions,
                "pending_models": len(self._pending_models),
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
            from app.coordination.event_router import subscribe
            from app.distributed.data_events import DataEventType

            subscribe(DataEventType.MODEL_PROMOTED, self._on_model_promoted)
            logger.info("Subscribed to MODEL_PROMOTED events via event_router")

            # Also subscribe to PROMOTION_COMPLETE for stage events compatibility
            try:
                from app.coordination.stage_events import StageEvent
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
        model_info = {
            "model_path": event.get("model_path"),
            "model_id": event.get("model_id"),
            "board_type": event.get("board_type"),
            "num_players": event.get("num_players"),
            "elo": event.get("elo"),
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

    async def _run_smart_sync(self) -> bool:
        """Run model sync using best available method.

        Tries HTTP distribution first (faster), falls back to rsync if:
        - HTTP is disabled
        - HTTP distribution fails
        - aiohttp is not available

        Returns:
            True if sync succeeded, False otherwise
        """
        # Try HTTP first if enabled
        if self.config.use_http_distribution:
            http_success = await self._distribute_via_http()
            if http_success:
                return True

            if self.config.fallback_to_rsync:
                logger.info("HTTP distribution failed, falling back to rsync")
            else:
                return False

        # Fallback to rsync
        return await self._run_model_sync()

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
