"""NPZ Distribution Daemon - Automatic training data sync after export.

This daemon watches for NPZ_EXPORT_COMPLETE events and automatically distributes
exported NPZ files to all training-capable cluster nodes.

Architecture:
    1. Subscribes to NPZ_EXPORT_COMPLETE events from event_router
    2. Uses rsync for reliable multi-node sync
    3. Tracks distribution status in ClusterManifest
    4. Emits NPZ_DISTRIBUTION_COMPLETE event when done

Usage:
    # As standalone daemon
    python -m app.coordination.npz_distribution_daemon

    # Via DaemonManager
    manager.register_factory(DaemonType.NPZ_DISTRIBUTION, daemon.run)

Configuration:
    Uses distributed_hosts.yaml for target nodes.
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
class NPZDistributionConfig:
    """Configuration for NPZ distribution daemon."""

    # Sync settings
    sync_timeout_seconds: float = 600.0  # 10 minute timeout (NPZ can be large)
    retry_count: int = 3
    retry_delay_seconds: float = 30.0

    # Target selection
    target_node_types: list[str] = field(
        default_factory=lambda: ["training", "selfplay"]
    )

    # Event settings
    emit_completion_event: bool = True

    # Polling (if no event system)
    poll_interval_seconds: float = 120.0
    training_data_dir: str = "data/training"


class NPZDistributionDaemon:
    """Daemon that automatically distributes NPZ training files after export.

    Watches for NPZ_EXPORT_COMPLETE events and syncs NPZ files to training nodes.
    This ensures that exported training data is available on all nodes for:
    - Distributed training
    - Model fine-tuning
    - Transfer learning

    The daemon solves the critical gap where training data would only exist on
    the export node, causing training to fail on other nodes.
    """

    def __init__(self, config: NPZDistributionConfig | None = None):
        self.config = config or NPZDistributionConfig()
        self._running = False
        self._last_sync_time: float = 0.0
        self._pending_npz: list[dict[str, Any]] = []
        self._sync_lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the daemon and subscribe to events."""
        logger.info("NPZDistributionDaemon starting...")
        self._running = True

        # Try to subscribe to NPZ_EXPORT_COMPLETE events
        try:
            from app.coordination.event_router import subscribe

            subscribe("NPZ_EXPORT_COMPLETE", self._on_npz_exported)
            logger.info("Subscribed to NPZ_EXPORT_COMPLETE events")
        except ImportError:
            logger.warning(
                "event_router not available, will poll for new NPZ files instead"
            )

        # Main loop - handle pending syncs and periodic checks
        while self._running:
            try:
                # Process any pending NPZ distributions
                if self._pending_npz:
                    await self._process_pending_npz()

                # Periodic sync to catch any missed exports
                if time.time() - self._last_sync_time > self.config.poll_interval_seconds:
                    await self._periodic_sync_check()

                await asyncio.sleep(5.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in NPZ distribution daemon loop: {e}")
                await asyncio.sleep(10.0)

        logger.info("NPZDistributionDaemon stopped")

    async def stop(self) -> None:
        """Stop the daemon gracefully."""
        self._running = False

    def _on_npz_exported(self, event: dict[str, Any]) -> None:
        """Handle NPZ_EXPORT_COMPLETE event (sync callback)."""
        npz_info = {
            "npz_path": event.get("npz_path"),
            "board_type": event.get("board_type"),
            "num_players": event.get("num_players"),
            "sample_count": event.get("sample_count"),
            "timestamp": time.time(),
        }
        logger.info(f"Received NPZ_EXPORT_COMPLETE event: {npz_info}")
        self._pending_npz.append(npz_info)

    async def _process_pending_npz(self) -> None:
        """Process pending NPZ distributions."""
        async with self._sync_lock:
            if not self._pending_npz:
                return

            # Take all pending NPZ files
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

                # Run sync with retry
                success = False
                for attempt in range(self.config.retry_count):
                    try:
                        success = await self._distribute_npz(npz_path, target_nodes)
                        if success:
                            self._last_sync_time = time.time()
                            logger.info(f"NPZ distribution completed: {npz_path}")
                            break
                    except Exception as e:
                        logger.error(f"Distribution attempt {attempt + 1} failed: {e}")

                    if attempt < self.config.retry_count - 1:
                        await asyncio.sleep(self.config.retry_delay_seconds)

                if success and self.config.emit_completion_event:
                    await self._emit_distribution_complete(npz_info, target_nodes)

                if not success:
                    logger.error(
                        f"NPZ distribution failed after {self.config.retry_count} attempts: {npz_path}"
                    )

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

    async def _distribute_npz(
        self, npz_path: str, target_nodes: list[dict[str, Any]]
    ) -> bool:
        """Distribute NPZ file to target nodes using rsync."""
        npz_file = Path(npz_path)
        if not npz_file.exists():
            logger.error(f"NPZ file not found: {npz_path}")
            return False

        success_count = 0
        for node in target_nodes:
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
                    success_count += 1
                else:
                    logger.error(f"rsync to {host} failed: {stderr.decode()[-200:]}")

            except asyncio.TimeoutError:
                logger.error(f"rsync to {host} timed out")
            except Exception as e:
                logger.error(f"rsync to {host} failed: {e}")

        # Consider success if at least one node received the file
        return success_count > 0

    async def _periodic_sync_check(self) -> None:
        """Periodic check for NPZ files that need distribution."""
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
                    await self._distribute_npz(str(npz_file), target_nodes)
                self._last_sync_time = time.time()

    async def _emit_distribution_complete(
        self, npz_info: dict[str, Any], target_nodes: list[dict[str, Any]]
    ) -> None:
        """Emit NPZ_DISTRIBUTION_COMPLETE event."""
        try:
            from app.coordination.event_router import emit

            await emit(
                "NPZ_DISTRIBUTION_COMPLETE",
                {
                    "npz_path": npz_info.get("npz_path"),
                    "board_type": npz_info.get("board_type"),
                    "num_players": npz_info.get("num_players"),
                    "sample_count": npz_info.get("sample_count"),
                    "nodes_synced": len(target_nodes),
                    "node_ids": [n.get("node_id") for n in target_nodes],
                    "timestamp": time.time(),
                },
            )
            logger.info("Emitted NPZ_DISTRIBUTION_COMPLETE event")
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
