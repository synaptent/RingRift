"""Unified Data Plane Daemon - Consolidated Data Synchronization.

December 28, 2025 - Phase 4 of Unified Data Plane implementation.

This daemon consolidates the fragmented data sync infrastructure into a single
unified daemon that coordinates all data movement across the RingRift cluster.

Consolidated Modules (~4,514 LOC):
- AutoSyncDaemon (P2P gossip sync)
- SyncFacade (backend routing)
- S3NodeSyncDaemon (S3 backup)
- dynamic_data_distribution.py (OWC distribution)
- SyncRouter (intelligent routing)

Key Components:
- DataCatalog: Central registry of what data exists where
- SyncPlanner: Intelligent routing (what to sync where and when)
- TransportManager: Unified transfer layer with fallback chains
- EventBridge: Event routing with chain completion

Event Chain Completion:
    SELFPLAY_COMPLETE → DATA_SYNC_STARTED → DATA_SYNC_COMPLETED → NEW_GAMES_AVAILABLE
    TRAINING_COMPLETED → MODEL_PROMOTED → MODEL_DISTRIBUTION_STARTED → MODEL_DISTRIBUTION_COMPLETE

Usage:
    from app.coordination.unified_data_plane_daemon import (
        UnifiedDataPlaneDaemon,
        get_data_plane_daemon,
    )

    daemon = get_data_plane_daemon()
    await daemon.start()

    # Query status
    status = daemon.get_status()

    # Trigger priority sync
    await daemon.trigger_priority_sync("hex8_2p", source_node="vast-12345")

    # Stop daemon
    await daemon.stop()
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.coordination.data_catalog import (
    DataCatalog,
    DataEntry,
    DataType,
    get_data_catalog,
)
from app.coordination.sync_planner_v2 import (
    SyncPlanner,
    SyncPlan,
    SyncPriority,
    get_sync_planner,
)
from app.coordination.transport_manager import (
    TransportManager,
    Transport,
    get_transport_manager,
)
from app.coordination.protocols import (
    CoordinatorProtocol,
    CoordinatorStatus,
    HealthCheckResult,
    register_coordinator,
    unregister_coordinator,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Data classes
    "DataPlaneConfig",
    "DataPlaneStats",
    # Main class
    "UnifiedDataPlaneDaemon",
    # Singleton accessors
    "get_data_plane_daemon",
    "reset_data_plane_daemon",
]


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DataPlaneConfig:
    """Configuration for the Unified Data Plane Daemon."""

    # Background task intervals
    catalog_refresh_interval: float = 60.0  # 1 min
    replication_check_interval: float = 300.0  # 5 min
    s3_backup_interval: float = 3600.0  # 1 hour
    manifest_broadcast_interval: float = 120.0  # 2 min

    # Sync settings
    min_replication_factor: int = 3
    target_replication_factor: int = 5
    max_concurrent_syncs: int = 5

    # S3 backup settings
    s3_enabled: bool = False
    s3_bucket: str = ""
    s3_prefix: str = "consolidated/"

    # OWC distribution settings
    owc_enabled: bool = False
    owc_host: str = "mac-studio"
    owc_port: int = 8780
    owc_path: str = "/Volumes/RingRift-Data"

    # Event settings
    event_timeout: float = 30.0
    event_retry_count: int = 3

    @classmethod
    def from_env(cls) -> "DataPlaneConfig":
        """Create config from environment variables."""
        import os

        return cls(
            catalog_refresh_interval=float(
                os.getenv("RINGRIFT_CATALOG_REFRESH_INTERVAL", "60")
            ),
            replication_check_interval=float(
                os.getenv("RINGRIFT_REPLICATION_CHECK_INTERVAL", "300")
            ),
            s3_enabled=os.getenv("RINGRIFT_S3_BACKUP_ENABLED", "false").lower() == "true",
            s3_bucket=os.getenv("RINGRIFT_S3_BUCKET", "ringrift-models-20251214"),
            min_replication_factor=int(
                os.getenv("RINGRIFT_MIN_REPLICATION", "3")
            ),
            owc_enabled=os.getenv("RINGRIFT_OWC_ENABLED", "false").lower() == "true",
            owc_host=os.getenv("RINGRIFT_OWC_HOST", "mac-studio"),
        )


@dataclass
class DataPlaneStats:
    """Statistics for the Unified Data Plane."""

    # Event counters
    events_received: int = 0
    events_processed: int = 0
    events_failed: int = 0

    # Sync counters
    syncs_initiated: int = 0
    syncs_completed: int = 0
    syncs_failed: int = 0
    bytes_synced: int = 0

    # Catalog counters
    catalog_entries: int = 0
    catalog_updates: int = 0

    # S3 backup counters
    s3_backups: int = 0
    s3_bytes: int = 0

    # Timing
    start_time: float = 0.0
    last_sync_time: float = 0.0
    last_catalog_update: float = 0.0
    last_s3_backup: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        uptime = time.time() - self.start_time if self.start_time > 0 else 0

        return {
            "events": {
                "received": self.events_received,
                "processed": self.events_processed,
                "failed": self.events_failed,
            },
            "syncs": {
                "initiated": self.syncs_initiated,
                "completed": self.syncs_completed,
                "failed": self.syncs_failed,
                "bytes": self.bytes_synced,
            },
            "catalog": {
                "entries": self.catalog_entries,
                "updates": self.catalog_updates,
            },
            "s3": {
                "backups": self.s3_backups,
                "bytes": self.s3_bytes,
            },
            "uptime_seconds": uptime,
            "last_sync": self.last_sync_time,
            "last_catalog_update": self.last_catalog_update,
        }


# =============================================================================
# Event Bridge
# =============================================================================


class EventBridge:
    """Bridges events between the data plane and the event system.

    Ensures complete event chains by:
    1. Subscribing to trigger events
    2. Dispatching to appropriate handlers
    3. Emitting completion events
    """

    def __init__(
        self,
        on_event: Callable[[str, dict], None],
    ):
        """Initialize the event bridge.

        Args:
            on_event: Callback for handling events
        """
        self._on_event = on_event
        self._subscriptions: list[str] = []
        self._running = False

        # Events we subscribe to
        self._subscribed_events = [
            "SELFPLAY_COMPLETE",
            "TRAINING_STARTED",
            "TRAINING_COMPLETED",
            "MODEL_PROMOTED",
            "ORPHAN_GAMES_DETECTED",
            "NODE_TERMINATING",
            "DATA_SYNC_REQUESTED",
            "SYNC_REQUEST",
            "NEW_GAMES_AVAILABLE",
            "HOST_OFFLINE",
            "HOST_ONLINE",
        ]

    async def start(self) -> None:
        """Start listening for events."""
        if self._running:
            return

        self._running = True

        try:
            from app.coordination.event_router import get_event_bus

            bus = get_event_bus()
            if bus:
                for event_type in self._subscribed_events:
                    bus.subscribe(event_type, self._handle_event)
                    self._subscriptions.append(event_type)

                logger.info(
                    f"[EventBridge] Subscribed to {len(self._subscriptions)} events"
                )
        except ImportError:
            logger.warning("[EventBridge] Event router not available")
        except Exception as e:
            logger.error(f"[EventBridge] Failed to subscribe: {e}")

    async def stop(self) -> None:
        """Stop listening for events."""
        self._running = False

        try:
            from app.coordination.event_router import get_event_bus

            bus = get_event_bus()
            if bus:
                for event_type in self._subscriptions:
                    try:
                        bus.unsubscribe(event_type, self._handle_event)
                    except (ValueError, KeyError, AttributeError):
                        pass  # Subscription already removed
        except (ImportError, AttributeError):
            pass  # Event bus not available

        self._subscriptions.clear()

    async def _handle_event(self, event: Any) -> None:
        """Handle an incoming event."""
        try:
            # Extract event type and payload
            if hasattr(event, "event_type"):
                event_type = event.event_type
                payload = event.payload if hasattr(event, "payload") else {}
            elif isinstance(event, dict):
                event_type = event.get("event_type", event.get("type", "UNKNOWN"))
                payload = event.get("payload", event)
            else:
                return

            # Dispatch to handler
            self._on_event(event_type, payload)

        except Exception as e:
            logger.error(f"[EventBridge] Error handling event: {e}")

    def emit(self, event_type: str, payload: dict) -> None:
        """Emit an event.

        Args:
            event_type: Type of event
            payload: Event payload
        """
        try:
            from app.coordination.event_router import safe_emit_event

            safe_emit_event(event_type, payload, source="UnifiedDataPlane")
        except ImportError:
            logger.debug(f"[EventBridge] Could not emit {event_type}: no event router")
        except Exception as e:
            logger.warning(f"[EventBridge] Failed to emit {event_type}: {e}")


# =============================================================================
# Main Daemon Class
# =============================================================================


class UnifiedDataPlaneDaemon(CoordinatorProtocol):
    """Unified daemon for all data synchronization.

    Consolidates AutoSyncDaemon, S3NodeSyncDaemon, dynamic_data_distribution,
    and SyncRouter into a single coherent daemon.

    Components:
    - DataCatalog: Tracks what data exists where
    - SyncPlanner: Decides what to sync where and when
    - TransportManager: Executes transfers with fallback chains
    - EventBridge: Handles events and ensures chain completion

    Background Tasks:
    - Catalog refresh: Periodic update of data locations
    - Replication check: Ensure min replication factor
    - S3 backup: Periodic backup to S3 (optional)
    - Manifest broadcast: Share manifest with peers
    """

    def __init__(
        self,
        config: DataPlaneConfig | None = None,
        catalog: DataCatalog | None = None,
        planner: SyncPlanner | None = None,
        transport: TransportManager | None = None,
    ):
        """Initialize the Unified Data Plane Daemon.

        Args:
            config: Daemon configuration
            catalog: DataCatalog instance
            planner: SyncPlanner instance
            transport: TransportManager instance
        """
        self._config = config or DataPlaneConfig.from_env()
        self._catalog = catalog or get_data_catalog()
        self._planner = planner or get_sync_planner()
        self._transport = transport or get_transport_manager()

        self._node_id = socket.gethostname()
        self._stats = DataPlaneStats()
        self._running = False

        # Status
        self._status = CoordinatorStatus.INITIALIZING
        self._start_time: float = 0.0

        # Event bridge
        self._event_bridge = EventBridge(self._on_event)

        # Background tasks
        self._tasks: list[asyncio.Task] = []

        # Sync semaphore
        self._sync_semaphore = asyncio.Semaphore(self._config.max_concurrent_syncs)

        logger.info(f"[UnifiedDataPlane] Initialized on {self._node_id}")

    # =========================================================================
    # Lifecycle - CoordinatorProtocol
    # =========================================================================

    async def start(self) -> None:
        """Start the daemon."""
        if self._running:
            return

        self._running = True
        self._start_time = time.time()
        self._stats.start_time = self._start_time
        self._status = CoordinatorStatus.STARTING

        try:
            # Register with coordinator registry
            register_coordinator("unified_data_plane", self)

            # Start event bridge
            await self._event_bridge.start()

            # Start sync planner
            await self._planner.start()

            # Start background tasks
            self._tasks = [
                asyncio.create_task(
                    self._catalog_refresh_loop(),
                    name="data_plane_catalog_refresh",
                ),
                asyncio.create_task(
                    self._replication_loop(),
                    name="data_plane_replication",
                ),
                asyncio.create_task(
                    self._manifest_broadcast_loop(),
                    name="data_plane_manifest",
                ),
            ]

            # Start S3 backup if enabled
            if self._config.s3_enabled:
                self._tasks.append(
                    asyncio.create_task(
                        self._s3_backup_loop(),
                        name="data_plane_s3_backup",
                    )
                )

            self._status = CoordinatorStatus.RUNNING
            logger.info("[UnifiedDataPlane] Started successfully")

        except Exception as e:
            self._status = CoordinatorStatus.STOPPED
            logger.error(f"[UnifiedDataPlane] Failed to start: {e}")
            raise

    async def stop(self) -> None:
        """Stop the daemon."""
        if not self._running:
            return

        self._running = False
        self._status = CoordinatorStatus.STOPPING

        # Stop event bridge
        await self._event_bridge.stop()

        # Stop sync planner
        await self._planner.stop()

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()

        # Unregister
        unregister_coordinator("unified_data_plane")

        self._status = CoordinatorStatus.STOPPED
        logger.info("[UnifiedDataPlane] Stopped")

    def health_check(self) -> HealthCheckResult:
        """Check daemon health.

        Returns:
            HealthCheckResult with daemon status
        """
        status = self._status
        message = ""
        errors = 0

        # Check component health
        components = {
            "catalog": self._catalog.health_check(),
            "planner": self._planner.health_check(),
            "transport": self._transport.health_check(),
        }

        unhealthy_components = [
            name for name, health in components.items()
            if not health.healthy
        ]

        if unhealthy_components:
            status = CoordinatorStatus.DEGRADED
            message = f"Unhealthy components: {', '.join(unhealthy_components)}"
            errors = len(unhealthy_components)

        # Check task health
        failed_tasks = [
            t.get_name() for t in self._tasks
            if t.done() and t.exception()
        ]

        if failed_tasks:
            status = CoordinatorStatus.DEGRADED
            message = f"Failed tasks: {', '.join(failed_tasks)}"
            errors += len(failed_tasks)

        return HealthCheckResult(
            healthy=status == CoordinatorStatus.RUNNING,
            status=status,
            message=message,
            details={
                "running": self._running,
                "node_id": self._node_id,
                "uptime": time.time() - self._start_time if self._start_time else 0,
                "stats": self._stats.to_dict(),
                "components": {
                    name: health.to_dict() if hasattr(health, "to_dict") else {"healthy": health.healthy}
                    for name, health in components.items()
                },
                "errors_count": errors,
            },
        )

    # =========================================================================
    # Event Handling
    # =========================================================================

    def _on_event(self, event_type: str, payload: dict) -> None:
        """Handle an event from the event bridge."""
        self._stats.events_received += 1

        try:
            # Route to sync planner
            plans = self._planner.plan_for_event(event_type, payload)

            for plan in plans:
                # Submit for async execution
                asyncio.create_task(self._execute_plan(plan))

            self._stats.events_processed += 1

            if plans:
                logger.info(
                    f"[UnifiedDataPlane] Event {event_type} → {len(plans)} sync plans"
                )

        except Exception as e:
            self._stats.events_failed += 1
            logger.error(f"[UnifiedDataPlane] Error handling {event_type}: {e}")

    async def _execute_plan(self, plan: SyncPlan) -> None:
        """Execute a sync plan with semaphore limiting."""
        async with self._sync_semaphore:
            self._stats.syncs_initiated += 1

            success = await self._planner.execute_plan(plan)

            if success:
                self._stats.syncs_completed += 1
                self._stats.bytes_synced += plan.total_bytes
                self._stats.last_sync_time = time.time()

                # Emit completion event
                self._emit_sync_completed(plan)
            else:
                self._stats.syncs_failed += 1
                self._emit_sync_failed(plan)

    def _emit_sync_completed(self, plan: SyncPlan) -> None:
        """Emit DATA_SYNC_COMPLETED event."""
        self._event_bridge.emit(
            "DATA_SYNC_COMPLETED",
            {
                "source_node": plan.source_node,
                "target_nodes": plan.target_nodes,
                "config_key": plan.config_key,
                "reason": plan.reason,
                "entry_count": len(plan.entries),
                "bytes_synced": plan.total_bytes,
            },
        )

        # If this was selfplay data, also emit NEW_GAMES_AVAILABLE
        if "selfplay" in plan.reason.lower():
            self._event_bridge.emit(
                "NEW_GAMES_AVAILABLE",
                {
                    "config_key": plan.config_key,
                    "source_node": plan.source_node,
                    "count": len(plan.entries),
                },
            )

    def _emit_sync_failed(self, plan: SyncPlan) -> None:
        """Emit DATA_SYNC_FAILED event."""
        self._event_bridge.emit(
            "DATA_SYNC_FAILED",
            {
                "source_node": plan.source_node,
                "target_nodes": plan.target_nodes,
                "config_key": plan.config_key,
                "reason": plan.reason,
                "error": plan.error,
            },
        )

    # =========================================================================
    # Background Tasks
    # =========================================================================

    async def _catalog_refresh_loop(self) -> None:
        """Periodically refresh the data catalog."""
        while self._running:
            try:
                await asyncio.sleep(self._config.catalog_refresh_interval)

                if not self._running:
                    break

                # Scan local data directory
                count = self._catalog.scan_local_directory()
                self._stats.catalog_updates += 1
                self._stats.catalog_entries = self._catalog.get_total_entries()
                self._stats.last_catalog_update = time.time()

                if count > 0:
                    logger.debug(f"[UnifiedDataPlane] Catalog refresh: {count} entries")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[UnifiedDataPlane] Catalog refresh error: {e}")
                await asyncio.sleep(60.0)

    async def _replication_loop(self) -> None:
        """Periodically check and enforce replication requirements."""
        while self._running:
            try:
                await asyncio.sleep(self._config.replication_check_interval)

                if not self._running:
                    break

                # Plan replication syncs
                plans = self._planner.plan_replication(
                    min_factor=self._config.min_replication_factor
                )

                for plan in plans:
                    await self._planner.submit_plan(plan)

                if plans:
                    logger.info(
                        f"[UnifiedDataPlane] Replication check: {len(plans)} plans"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[UnifiedDataPlane] Replication loop error: {e}")
                await asyncio.sleep(60.0)

    async def _manifest_broadcast_loop(self) -> None:
        """Periodically broadcast manifest to peers."""
        while self._running:
            try:
                await asyncio.sleep(self._config.manifest_broadcast_interval)

                if not self._running:
                    break

                # Get local manifest
                manifest = self._catalog.get_manifest()

                # Broadcast via P2P if available
                await self._broadcast_manifest(manifest)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[UnifiedDataPlane] Manifest broadcast error: {e}")
                await asyncio.sleep(60.0)

    async def _broadcast_manifest(self, manifest: dict) -> None:
        """Broadcast manifest to P2P peers."""
        try:
            import aiohttp
            from app.config.ports import get_p2p_status_url, P2P_DEFAULT_PORT

            # Get peer list from P2P status
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    get_p2p_status_url(),
                    timeout=aiohttp.ClientTimeout(total=5.0),
                ) as resp:
                    if resp.status != 200:
                        return
                    status = await resp.json()

            peers = status.get("alive_peers_list", [])
            if not peers:
                return

            # POST manifest to each peer's catalog endpoint
            for peer in peers[:10]:  # Limit to 10 peers
                peer_id = peer.get("node_id", peer.get("id", ""))
                if peer_id == self._node_id:
                    continue

                ip = peer.get("ip", "")
                if not ip:
                    continue

                try:
                    async with session.post(
                        f"http://{ip}:{P2P_DEFAULT_PORT}/data-plane/manifest",
                        json=manifest,
                        timeout=aiohttp.ClientTimeout(total=10.0),
                    ) as resp:
                        if resp.status == 200:
                            logger.debug(
                                f"[UnifiedDataPlane] Manifest sent to {peer_id}"
                            )
                except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                    # Network errors are expected for some peers
                    pass

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[UnifiedDataPlane] Manifest broadcast failed: {e}")

    async def _s3_backup_loop(self) -> None:
        """Periodically backup to S3."""
        while self._running:
            try:
                await asyncio.sleep(self._config.s3_backup_interval)

                if not self._running:
                    break

                if not self._config.s3_enabled:
                    continue

                # Run S3 backup
                await self._run_s3_backup()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[UnifiedDataPlane] S3 backup error: {e}")
                await asyncio.sleep(300.0)

    async def _run_s3_backup(self) -> None:
        """Execute S3 backup of important data."""
        import subprocess

        # Get under-replicated entries that need S3 backup
        entries = self._catalog.get_under_replicated(
            min_factor=self._config.target_replication_factor
        )

        if not entries:
            return

        # Filter to models and canonical databases
        backup_entries = [
            e for e in entries
            if e.data_type in (DataType.MODELS, DataType.GAMES)
            and ("canonical" in e.path or "models/" in e.path)
        ]

        if not backup_entries:
            return

        for entry in backup_entries[:10]:  # Limit per cycle
            try:
                # Find local path
                local_path = Path(entry.path)
                if not local_path.exists():
                    # Try data directory
                    local_path = Path("data") / entry.path
                    if not local_path.exists():
                        continue

                # Build S3 path
                s3_path = f"s3://{self._config.s3_bucket}/{self._config.s3_prefix}{entry.path}"

                # Run aws s3 cp
                result = subprocess.run(
                    ["aws", "s3", "cp", str(local_path), s3_path],
                    capture_output=True,
                    timeout=300,
                )

                if result.returncode == 0:
                    self._stats.s3_backups += 1
                    self._stats.s3_bytes += entry.size_bytes
                    self._stats.last_s3_backup = time.time()

                    # Mark as having S3 location
                    self._catalog.mark_synced(entry.path, "s3")

                    logger.info(f"[UnifiedDataPlane] S3 backup: {entry.path}")

            except subprocess.TimeoutExpired:
                logger.warning(f"[UnifiedDataPlane] S3 backup timeout: {entry.path}")
            except Exception as e:
                logger.error(f"[UnifiedDataPlane] S3 backup error: {e}")

    # =========================================================================
    # Public API
    # =========================================================================

    async def trigger_priority_sync(
        self,
        config_key: str,
        source_node: str | None = None,
        target_nodes: list[str] | None = None,
        reason: str = "priority_sync",
    ) -> bool:
        """Trigger a high-priority sync operation.

        Args:
            config_key: Configuration to sync (e.g., "hex8_2p")
            source_node: Source node (defaults to best available)
            target_nodes: Target nodes (defaults to training nodes)
            reason: Reason for sync

        Returns:
            True if sync initiated successfully
        """
        # Get entries for this config
        entries = self._catalog.get_by_config(config_key)
        if not entries:
            logger.warning(f"[UnifiedDataPlane] No entries for {config_key}")
            return False

        # Determine source
        if not source_node:
            # Find node with most entries
            location_counts: dict[str, int] = {}
            for entry in entries:
                for loc in entry.locations:
                    location_counts[loc] = location_counts.get(loc, 0) + 1

            if location_counts:
                source_node = max(location_counts.keys(), key=lambda k: location_counts[k])
            else:
                source_node = self._node_id

        # Determine targets
        if not target_nodes:
            # Get training/GPU nodes
            try:
                from app.config.cluster_config import get_cluster_nodes

                nodes = get_cluster_nodes()
                target_nodes = [
                    name for name, node in nodes.items()
                    if name != source_node
                    and (node.is_gpu_node if hasattr(node, "is_gpu_node") else False)
                ][:5]
            except (KeyError, AttributeError, ValueError, OSError) as e:
                # Config loading or node access failures
                logger.debug(f"[UnifiedDataPlane] Node lookup failed: {e}")
                target_nodes = []

        if not target_nodes:
            logger.warning(f"[UnifiedDataPlane] No targets for {config_key}")
            return False

        # Create priority plan
        plan = SyncPlan(
            source_node=source_node,
            target_nodes=target_nodes,
            entries=entries[:50],
            priority=SyncPriority.HIGH,
            reason=reason,
            config_key=config_key,
            transport_preference=[Transport.RSYNC, Transport.HTTP_FETCH],
            deadline=time.time() + 300,  # 5 min deadline
        )

        # Submit for execution
        await self._planner.submit_plan(plan)

        logger.info(
            f"[UnifiedDataPlane] Priority sync triggered: {config_key} "
            f"({len(entries)} entries, {len(target_nodes)} targets)"
        )

        return True

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive daemon status."""
        health = self.health_check()

        return {
            "node_id": self._node_id,
            "running": self._running,
            "status": self._status.name,
            "health": {
                "healthy": health.healthy,
                "status": health.status.name,
                "message": health.message,
            },
            "config": {
                "min_replication": self._config.min_replication_factor,
                "target_replication": self._config.target_replication_factor,
                "s3_enabled": self._config.s3_enabled,
                "owc_enabled": self._config.owc_enabled,
            },
            "stats": self._stats.to_dict(),
            "catalog": {
                "total_entries": self._catalog.get_total_entries(),
            },
            "planner": self._planner.get_status(),
        }

    def receive_manifest(self, node_id: str, manifest: dict) -> int:
        """Receive a manifest from a peer.

        Args:
            node_id: Peer node ID
            manifest: Manifest data

        Returns:
            Number of entries registered
        """
        return self._catalog.register_from_manifest(node_id, manifest)


# =============================================================================
# Module-Level Singleton
# =============================================================================

_data_plane_daemon: UnifiedDataPlaneDaemon | None = None


def get_data_plane_daemon() -> UnifiedDataPlaneDaemon:
    """Get the singleton UnifiedDataPlaneDaemon instance."""
    global _data_plane_daemon
    if _data_plane_daemon is None:
        _data_plane_daemon = UnifiedDataPlaneDaemon()
    return _data_plane_daemon


def reset_data_plane_daemon() -> None:
    """Reset the singleton (for testing)."""
    global _data_plane_daemon
    if _data_plane_daemon is not None:
        asyncio.create_task(_data_plane_daemon.stop())
    _data_plane_daemon = None
