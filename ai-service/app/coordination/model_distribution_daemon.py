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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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

    async def start(self) -> None:
        """Start the daemon and subscribe to events."""
        logger.info("ModelDistributionDaemon starting...")
        self._running = True

        # Try to subscribe to MODEL_PROMOTED events
        try:
            from app.coordination.event_router import subscribe

            subscribe("MODEL_PROMOTED", self._on_model_promoted)
            logger.info("Subscribed to MODEL_PROMOTED events")
        except ImportError:
            logger.warning(
                "event_router not available, will poll for new models instead"
            )

        # Main loop - handle pending syncs and periodic checks
        while self._running:
            try:
                # Process any pending model distributions
                if self._pending_models:
                    await self._process_pending_models()

                # Periodic sync to catch any missed promotions
                if time.time() - self._last_sync_time > self.config.poll_interval_seconds:
                    await self._periodic_sync_check()

                await asyncio.sleep(5.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in distribution daemon loop: {e}")
                await asyncio.sleep(10.0)

        logger.info("ModelDistributionDaemon stopped")

    async def stop(self) -> None:
        """Stop the daemon gracefully."""
        self._running = False

    def _on_model_promoted(self, event: dict[str, Any]) -> None:
        """Handle MODEL_PROMOTED event (sync callback)."""
        model_info = {
            "model_path": event.get("model_path"),
            "model_id": event.get("model_id"),
            "board_type": event.get("board_type"),
            "num_players": event.get("num_players"),
            "elo": event.get("elo"),
            "timestamp": time.time(),
        }
        logger.info(f"Received MODEL_PROMOTED event: {model_info}")
        self._pending_models.append(model_info)

    async def _process_pending_models(self) -> None:
        """Process pending model distributions."""
        async with self._sync_lock:
            if not self._pending_models:
                return

            # Take all pending models
            models = self._pending_models.copy()
            self._pending_models.clear()

            logger.info(f"Processing {len(models)} pending model distributions")

            # Run sync with retry
            for attempt in range(self.config.retry_count):
                try:
                    success = await self._run_model_sync()
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
                await self._run_model_sync()
                self._last_sync_time = time.time()

    async def _emit_distribution_complete(
        self, models: list[dict[str, Any]]
    ) -> None:
        """Emit MODEL_DISTRIBUTION_COMPLETE event."""
        try:
            from app.coordination.event_router import emit

            await emit(
                event_type="MODEL_DISTRIBUTION_COMPLETE",
                data={
                    "models": models,
                    "timestamp": time.time(),
                    "node_id": os.environ.get("RINGRIFT_NODE_ID", "unknown"),
                },
            )
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
