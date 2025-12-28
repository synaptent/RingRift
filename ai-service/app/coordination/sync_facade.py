"""Unified Sync Facade - Single entry point for all cluster sync operations.

This facade consolidates 8+ competing sync implementations into a single,
clean API. It routes sync requests to the appropriate underlying implementation
based on the sync type and configuration.

Background (December 2025):
The codebase evolved 8 different sync implementations over time:
1. SyncCoordinator (app/distributed/) - Low-level transport execution
2. SyncScheduler (app/coordination/) - Scheduling layer (DEPRECATED)
3. UnifiedDataSync - Legacy unified sync (DEPRECATED)
4. SyncOrchestrator - Orchestration wrapper (DEPRECATED)
5. AutoSyncDaemon - Automated P2P sync (ACTIVE - now supports all strategies)
6. ClusterDataSyncDaemon - Push-based cluster sync (DEPRECATED - absorbed into AutoSyncDaemon)
7. SyncRouter - Intelligent routing (ACTIVE)
8. EphemeralSyncDaemon - Aggressive sync for ephemeral hosts (DEPRECATED - absorbed into AutoSyncDaemon)

December 26, 2025 Consolidation:
- AutoSyncDaemon now supports HYBRID, EPHEMERAL, BROADCAST, and AUTO strategies
- EphemeralSyncDaemon and ClusterDataSyncDaemon are absorbed into AutoSyncDaemon
- Use create_ephemeral_sync_daemon() or create_cluster_data_sync_daemon() for strategy-specific daemons

This facade provides a clean interface while preserving backward compatibility
and allowing gradual migration to unified patterns.

Usage:
    from app.coordination.sync_facade import SyncFacade, sync

    # Get facade instance
    facade = SyncFacade.get_instance()

    # Sync specific data type to targets
    await facade.sync(
        data_type="games",
        targets=["node-1", "node-2"],
        board_type="hex8",
        priority="high"
    )

    # Or use convenience function
    await sync("models", targets=["all"])

Architecture:
    - Routes requests to appropriate backend based on data_type and targets
    - Logs which implementation is being used for transparency
    - Provides metrics on sync operations
    - Handles backend failures gracefully with fallbacks
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.coordination.contracts import CoordinatorStatus, HealthCheckResult

logger = logging.getLogger(__name__)

__all__ = [
    "SyncFacade",
    "SyncRequest",
    "SyncResponse",
    "SyncBackend",
    "sync",
    "get_sync_facade",
    "reset_sync_facade",
]


class SyncBackend(Enum):
    """Sync backend implementations."""
    AUTO_SYNC = "auto_sync"              # AutoSyncDaemon (P2P gossip)
    CLUSTER_SYNC = "cluster_data_sync"   # ClusterDataSyncDaemon (push-based)
    DISTRIBUTED = "distributed"          # SyncCoordinator (low-level transport)
    EPHEMERAL = "ephemeral_sync"         # EphemeralSyncDaemon (aggressive)
    ROUTER = "router"                    # SyncRouter (intelligent routing)
    # Deprecated backends (emit warnings)
    SCHEDULER = "scheduler"              # SyncScheduler (DEPRECATED)
    UNIFIED = "unified_data_sync"        # UnifiedDataSync (DEPRECATED)
    ORCHESTRATOR = "orchestrator"        # SyncOrchestrator (wrapper, may retire)


@dataclass
class SyncRequest:
    """Request to sync data across cluster."""
    data_type: str  # "games", "models", "npz", "all"
    targets: list[str] | None = None  # None = all eligible nodes
    board_type: str | None = None
    num_players: int | None = None
    priority: str = "normal"  # "low", "normal", "high", "critical"
    timeout_seconds: float = 300.0
    bandwidth_limit_mbps: int | None = None
    # Routing hints
    exclude_nodes: list[str] | None = None
    prefer_ephemeral: bool = False  # Prefer ephemeral sync daemon
    require_confirmation: bool = False  # Wait for sync confirmation


@dataclass
class SyncResponse:
    """Response from sync operation."""
    success: bool
    backend_used: SyncBackend
    nodes_synced: int = 0
    bytes_transferred: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


class SyncFacade:
    """Unified facade for all sync operations.

    Routes sync requests to appropriate backend based on:
    - Data type
    - Target nodes
    - Priority
    - Node capabilities (ephemeral, NFS, etc.)
    """

    def __init__(self):
        """Initialize sync facade."""
        self._backends_loaded: dict[SyncBackend, bool] = {}
        self._backend_cache: dict[SyncBackend, Any] = {}
        self._stats = {
            "total_syncs": 0,
            "by_backend": {},
            "total_bytes": 0,
            "total_errors": 0,
        }

    async def sync(self, request: SyncRequest | dict | str, **kwargs) -> SyncResponse:
        """Execute a sync request.

        Args:
            request: SyncRequest, dict, or data_type string
            **kwargs: Additional parameters if request is a string

        Returns:
            SyncResponse with sync results

        Examples:
            # Sync games to all nodes
            await facade.sync("games")

            # Sync models to specific nodes
            await facade.sync("models", targets=["node-1", "node-2"])

            # Full request object
            await facade.sync(SyncRequest(
                data_type="games",
                board_type="hex8",
                priority="high"
            ))
        """
        start_time = time.time()

        # Normalize request
        if isinstance(request, str):
            request = SyncRequest(data_type=request, **kwargs)
        elif isinstance(request, dict):
            request = SyncRequest(**request)

        # Select backend
        backend = self._select_backend(request)
        logger.info(
            f"[SyncFacade] Routing {request.data_type} sync to {backend.value} "
            f"(targets={request.targets}, priority={request.priority})"
        )

        # Execute sync with backend fallback (Dec 27, 2025)
        # If primary backend fails, try fallback chain to ensure sync completes
        fallback_chain = self._get_backend_fallback_chain(backend)
        all_errors: list[str] = []

        for attempt_backend in fallback_chain:
            try:
                response = await self._execute_sync(attempt_backend, request)
                response.duration_seconds = time.time() - start_time

                # Update stats
                self._stats["total_syncs"] += 1
                self._stats["by_backend"][attempt_backend.value] = (
                    self._stats["by_backend"].get(attempt_backend.value, 0) + 1
                )
                self._stats["total_bytes"] += response.bytes_transferred

                if response.success:
                    if attempt_backend != backend:
                        logger.info(
                            f"[SyncFacade] Sync succeeded with fallback backend "
                            f"{attempt_backend.value} (primary was {backend.value})"
                        )
                    return response

                # Backend returned failure but didn't throw
                all_errors.append(
                    f"{attempt_backend.value}: {', '.join(response.errors or ['unknown error'])}"
                )
                logger.warning(
                    f"[SyncFacade] {attempt_backend.value} returned failure, "
                    f"trying next backend"
                )

            except Exception as e:
                all_errors.append(f"{attempt_backend.value}: {e}")
                logger.warning(
                    f"[SyncFacade] {attempt_backend.value} raised exception: {e}, "
                    f"trying next backend"
                )

        # All backends failed
        self._stats["total_errors"] += 1
        logger.error(
            f"[SyncFacade] All backends failed for {request.data_type} sync. "
            f"Errors: {all_errors}"
        )
        return SyncResponse(
            success=False,
            backend_used=backend,
            duration_seconds=time.time() - start_time,
            errors=all_errors,
        )

    def _select_backend(self, request: SyncRequest) -> SyncBackend:
        """Select appropriate backend for sync request.

        Selection logic:
        1. Ephemeral nodes → EphemeralSyncDaemon
        2. High priority → ClusterDataSyncDaemon (push-based)
        3. P2P gossip → AutoSyncDaemon
        4. Intelligent routing → SyncRouter
        5. Fallback → DistributedSyncCoordinator (transport layer)
        """
        # Check if ephemeral sync is preferred
        if request.prefer_ephemeral:
            return SyncBackend.EPHEMERAL

        # High-priority syncs use push-based daemon
        if request.priority in ["high", "critical"]:
            return SyncBackend.CLUSTER_SYNC

        # For specific targets, use router for intelligent routing
        if request.targets and request.targets != ["all"]:
            return SyncBackend.ROUTER

        # Default to auto-sync daemon (P2P gossip)
        return SyncBackend.AUTO_SYNC

    def _get_backend_fallback_chain(self, primary: SyncBackend) -> list[SyncBackend]:
        """Get fallback chain for a backend.

        December 27, 2025: Provides fallback backends to try if primary fails.
        This ensures sync completes even if one backend is unavailable.

        Fallback logic:
        - AUTO_SYNC → ROUTER → DISTRIBUTED (P2P → intelligent routing → raw transport)
        - CLUSTER_SYNC → AUTO_SYNC → DISTRIBUTED (push → gossip → raw)
        - ROUTER → DISTRIBUTED → AUTO_SYNC (routing → raw → gossip)
        - DISTRIBUTED → AUTO_SYNC (raw → gossip)
        - EPHEMERAL → AUTO_SYNC → DISTRIBUTED (ephemeral → gossip → raw)

        Args:
            primary: Primary backend that was selected

        Returns:
            List of backends to try in order (includes primary first)
        """
        # Define fallback chains per primary backend
        fallback_chains: dict[SyncBackend, list[SyncBackend]] = {
            SyncBackend.AUTO_SYNC: [
                SyncBackend.AUTO_SYNC,
                SyncBackend.ROUTER,
                SyncBackend.DISTRIBUTED,
            ],
            SyncBackend.CLUSTER_SYNC: [
                SyncBackend.CLUSTER_SYNC,
                SyncBackend.AUTO_SYNC,
                SyncBackend.DISTRIBUTED,
            ],
            SyncBackend.ROUTER: [
                SyncBackend.ROUTER,
                SyncBackend.DISTRIBUTED,
                SyncBackend.AUTO_SYNC,
            ],
            SyncBackend.DISTRIBUTED: [
                SyncBackend.DISTRIBUTED,
                SyncBackend.AUTO_SYNC,
            ],
            SyncBackend.EPHEMERAL: [
                SyncBackend.EPHEMERAL,
                SyncBackend.AUTO_SYNC,
                SyncBackend.DISTRIBUTED,
            ],
            # Deprecated backends get mapped to their replacements
            SyncBackend.SCHEDULER: [
                SyncBackend.AUTO_SYNC,
                SyncBackend.DISTRIBUTED,
            ],
            SyncBackend.UNIFIED: [
                SyncBackend.DISTRIBUTED,
                SyncBackend.AUTO_SYNC,
            ],
            SyncBackend.ORCHESTRATOR: [
                SyncBackend.ORCHESTRATOR,
                SyncBackend.DISTRIBUTED,
                SyncBackend.AUTO_SYNC,
            ],
        }

        return fallback_chains.get(primary, [primary, SyncBackend.DISTRIBUTED])

    async def _execute_sync(
        self,
        backend: SyncBackend,
        request: SyncRequest,
    ) -> SyncResponse:
        """Execute sync using the specified backend.

        Routes the sync request to the appropriate backend implementation.
        Handles deprecated backends by logging warnings and redirecting
        to their replacements.

        Args:
            backend: The SyncBackend to use for this operation.
            request: The sync request with data type, targets, and options.

        Returns:
            SyncResponse with success status, backend used, and details.

        Raises:
            ValueError: If an unknown backend is specified.
        """
        if backend == SyncBackend.AUTO_SYNC:
            return await self._sync_via_auto_sync(request)
        elif backend == SyncBackend.CLUSTER_SYNC:
            return await self._sync_via_cluster_sync(request)
        elif backend == SyncBackend.DISTRIBUTED:
            return await self._sync_via_distributed(request)
        elif backend == SyncBackend.EPHEMERAL:
            return await self._sync_via_ephemeral(request)
        elif backend == SyncBackend.ROUTER:
            return await self._sync_via_router(request)
        elif backend == SyncBackend.SCHEDULER:
            logger.warning(
                "[SyncFacade] SyncScheduler is DEPRECATED. "
                "Falling back to AutoSyncDaemon."
            )
            return await self._sync_via_auto_sync(request)
        elif backend == SyncBackend.UNIFIED:
            logger.warning(
                "[SyncFacade] UnifiedDataSync is DEPRECATED. "
                "Falling back to DistributedSyncCoordinator."
            )
            return await self._sync_via_distributed(request)
        elif backend == SyncBackend.ORCHESTRATOR:
            logger.warning(
                "[SyncFacade] SyncOrchestrator may be deprecated. "
                "Using for now, but consider migration."
            )
            return await self._sync_via_orchestrator(request)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    async def _sync_via_auto_sync(self, request: SyncRequest) -> SyncResponse:
        """Sync using AutoSyncDaemon (P2P gossip-based sync)."""
        try:
            from app.coordination.auto_sync_daemon import get_auto_sync_daemon

            daemon = get_auto_sync_daemon()

            # Trigger sync cycle
            if not daemon.is_running():
                logger.warning("[SyncFacade] AutoSyncDaemon not running, starting...")
                await daemon.start()

            # Dec 2025: Use sync_now() for immediate sync
            games_synced = await daemon.sync_now()

            return SyncResponse(
                success=True,
                backend_used=SyncBackend.AUTO_SYNC,
                details={
                    "daemon_running": daemon.is_running(),
                    "games_synced": games_synced,
                },
            )

        except ImportError as e:
            logger.error(f"[SyncFacade] AutoSyncDaemon not available: {e}")
            return SyncResponse(
                success=False,
                backend_used=SyncBackend.AUTO_SYNC,
                errors=[f"AutoSyncDaemon not available: {e}"],
            )

    async def _sync_via_cluster_sync(self, request: SyncRequest) -> SyncResponse:
        """Sync using AutoSyncDaemon with BROADCAST strategy (push-based sync).

        December 2025: Now uses consolidated AutoSyncDaemon with BROADCAST strategy
        instead of deprecated ClusterDataSyncDaemon.
        """
        try:
            from app.coordination.auto_sync_daemon import (
                create_cluster_data_sync_daemon,
            )

            daemon = create_cluster_data_sync_daemon()

            # Trigger immediate sync
            if not daemon.is_running():
                logger.warning(
                    "[SyncFacade] AutoSyncDaemon (broadcast) not running, starting..."
                )
                await daemon.start()

            # Dec 2025: Use broadcast_sync_cycle() for push-based sync
            files_synced = await daemon.broadcast_sync_cycle()

            return SyncResponse(
                success=True,
                backend_used=SyncBackend.CLUSTER_SYNC,
                details={
                    "daemon_running": daemon.is_running(),
                    "files_synced": files_synced,
                    "strategy": "broadcast",
                },
            )

        except ImportError as e:
            logger.error(f"[SyncFacade] AutoSyncDaemon not available: {e}")
            return SyncResponse(
                success=False,
                backend_used=SyncBackend.CLUSTER_SYNC,
                errors=[f"AutoSyncDaemon not available: {e}"],
            )

    async def _sync_via_distributed(self, request: SyncRequest) -> SyncResponse:
        """Sync using DistributedSyncCoordinator (transport layer)."""
        try:
            from app.distributed.sync_coordinator import SyncCoordinator

            coordinator = SyncCoordinator.get_instance()

            # Map request to coordinator methods
            if request.data_type == "games":
                result = await coordinator.sync_games(
                    board_type=request.board_type,
                    num_players=request.num_players,
                )
            elif request.data_type == "models":
                result = await coordinator.sync_models()
            elif request.data_type == "npz":
                result = await coordinator.sync_training_data()
            elif request.data_type == "all":
                result = await coordinator.full_cluster_sync()
            else:
                raise ValueError(f"Unknown data_type: {request.data_type}")

            return SyncResponse(
                success=True,
                backend_used=SyncBackend.DISTRIBUTED,
                nodes_synced=result.nodes_synced if hasattr(result, "nodes_synced") else 0,
                bytes_transferred=result.bytes_transferred
                if hasattr(result, "bytes_transferred")
                else 0,
                details={"result": str(result)},
            )

        except ImportError as e:
            logger.error(f"[SyncFacade] DistributedSyncCoordinator not available: {e}")
            return SyncResponse(
                success=False,
                backend_used=SyncBackend.DISTRIBUTED,
                errors=[f"DistributedSyncCoordinator not available: {e}"],
            )

    async def _sync_via_ephemeral(self, request: SyncRequest) -> SyncResponse:
        """Sync using AutoSyncDaemon with EPHEMERAL strategy (aggressive sync).

        December 2025: Now uses consolidated AutoSyncDaemon with EPHEMERAL strategy
        instead of deprecated EphemeralSyncDaemon.
        """
        try:
            from app.coordination.auto_sync_daemon import (
                create_ephemeral_sync_daemon,
            )

            daemon = create_ephemeral_sync_daemon()

            if not daemon.is_running():
                logger.warning(
                    "[SyncFacade] AutoSyncDaemon (ephemeral) not running, starting..."
                )
                await daemon.start()

            # Dec 2025: Use sync_now() for immediate sync
            games_pushed = await daemon.sync_now()

            return SyncResponse(
                success=True,
                backend_used=SyncBackend.EPHEMERAL,
                details={
                    "daemon_running": daemon.is_running(),
                    "is_ephemeral": daemon._is_ephemeral,
                    "games_pushed": games_pushed,
                    "strategy": "ephemeral",
                },
            )

        except ImportError as e:
            logger.error(f"[SyncFacade] AutoSyncDaemon not available: {e}")
            return SyncResponse(
                success=False,
                backend_used=SyncBackend.EPHEMERAL,
                errors=[f"AutoSyncDaemon not available: {e}"],
            )

    async def _sync_via_router(self, request: SyncRequest) -> SyncResponse:
        """Sync using SyncRouter (intelligent routing)."""
        try:
            from app.coordination.sync_router import get_sync_router

            router = get_sync_router()

            # Get sync targets
            targets = router.get_sync_targets(
                data_type=request.data_type,
                board_type=request.board_type,
                num_players=request.num_players,
                exclude_nodes=request.exclude_nodes or [],
            )

            logger.info(
                f"[SyncFacade] Router found {len(targets)} eligible sync targets"
            )

            # Use distributed coordinator for actual sync
            return await self._sync_via_distributed(request)

        except ImportError as e:
            logger.error(f"[SyncFacade] SyncRouter not available: {e}")
            return SyncResponse(
                success=False,
                backend_used=SyncBackend.ROUTER,
                errors=[f"SyncRouter not available: {e}"],
            )

    async def _sync_via_orchestrator(self, request: SyncRequest) -> SyncResponse:
        """Sync using SyncOrchestrator (wrapper around multiple sync components)."""
        try:
            from app.distributed.sync_orchestrator import get_sync_orchestrator

            orchestrator = get_sync_orchestrator()

            if not orchestrator.state.initialized:
                await orchestrator.initialize()

            # Trigger full sync
            result = await orchestrator.sync_all()

            return SyncResponse(
                success=result.success,
                backend_used=SyncBackend.ORCHESTRATOR,
                nodes_synced=len(result.component_results),
                details={"component_results": len(result.component_results)},
            )

        except ImportError as e:
            logger.error(f"[SyncFacade] SyncOrchestrator not available: {e}")
            return SyncResponse(
                success=False,
                backend_used=SyncBackend.ORCHESTRATOR,
                errors=[f"SyncOrchestrator not available: {e}"],
            )

    async def trigger_priority_sync(
        self,
        reason: str = "priority_sync",
        source_node: str | None = None,
        config_key: str | None = None,
        data_type: str = "games",
    ) -> SyncResponse:
        """Trigger a priority sync for urgent data recovery.

        This is used for critical sync scenarios like orphan game recovery
        where data needs to be synced immediately before potential loss.

        Args:
            reason: Reason for priority sync (for logging)
            source_node: Node to pull data from (if known)
            config_key: Config key to filter sync (e.g., "hex8_2p")
            data_type: Type of data to sync

        Returns:
            SyncResponse from the sync operation
        """
        logger.info(
            f"[SyncFacade] Priority sync triggered: reason={reason}, "
            f"source={source_node}, config={config_key}"
        )

        # Parse board_type and num_players from config_key if provided
        board_type = None
        num_players = None
        if config_key and "_" in config_key:
            parts = config_key.rsplit("_", 1)
            if len(parts) == 2 and parts[1].endswith("p"):
                board_type = parts[0]
                try:
                    num_players = int(parts[1][:-1])
                except ValueError:
                    pass

        # Build request with high priority
        request = SyncRequest(
            data_type=data_type,
            targets=[source_node] if source_node else None,
            board_type=board_type,
            num_players=num_players,
            priority="critical",  # Use critical priority for immediate handling
            require_confirmation=True,
        )

        response = await self.sync(request)

        # December 2025: Emit DATA_SYNC_COMPLETED/FAILED for pipeline coordination
        # This is critical for orphan game recovery flow to continue
        await self._emit_sync_event(response, source_node, config_key, reason)

        return response

    async def _emit_sync_event(
        self,
        response: SyncResponse,
        source_node: str | None,
        config_key: str | None,
        reason: str,
    ) -> None:
        """Emit DATA_SYNC_COMPLETED or DATA_SYNC_FAILED event.

        December 2025: Added to fix orphan game recovery pipeline.
        Without this event, DataPipelineOrchestrator._on_orphan_games_detected()
        would wait indefinitely for sync completion.
        """
        try:
            if response.success:
                from app.distributed.data_events import emit_data_sync_completed

                await emit_data_sync_completed(
                    host=source_node or "cluster",
                    games_synced=response.nodes_synced,
                    duration=response.duration_seconds,
                    bytes_transferred=response.bytes_transferred,
                    source="sync_facade.trigger_priority_sync",
                    config=config_key or "",
                )
                logger.debug(
                    f"[SyncFacade] Emitted DATA_SYNC_COMPLETED: {response.nodes_synced} games"
                )
            else:
                from app.distributed.data_events import emit_data_sync_failed

                await emit_data_sync_failed(
                    host=source_node or "cluster",
                    error="; ".join(response.errors) if response.errors else reason,
                    source="sync_facade.trigger_priority_sync",
                )
                logger.debug(f"[SyncFacade] Emitted DATA_SYNC_FAILED: {response.errors}")
        except ImportError:
            logger.debug("[SyncFacade] data_events module not available for event emission")
        except Exception as e:
            logger.debug(f"[SyncFacade] Failed to emit sync event: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get sync statistics for monitoring and debugging.

        Returns:
            Dictionary containing:
            - total_syncs: Number of sync operations attempted
            - by_backend: Dict mapping backend name to operation count
            - total_bytes: Total bytes transferred across all syncs
            - total_errors: Number of failed sync operations
            - backends_loaded: Dict mapping backend name to load status
        """
        return {
            **self._stats,
            "backends_loaded": {
                k.value: v for k, v in self._backends_loaded.items()
            },
        }

    def health_check(self) -> HealthCheckResult:
        """Check health status of SyncFacade.

        December 27, 2025: Added to meet P2P manager health_check() standard.
        Updated to return HealthCheckResult (Dec 27, 2025).

        Returns:
            HealthCheckResult with status, sync statistics, and backend health.
        """
        coordinator_status = CoordinatorStatus.RUNNING
        errors_count = self._stats.get("total_errors", 0)
        message = ""

        # Check error rate
        total_syncs = self._stats.get("total_syncs", 0)
        if total_syncs > 0:
            error_rate = errors_count / total_syncs
            if error_rate > 0.5:
                coordinator_status = CoordinatorStatus.STOPPED
                message = f"High error rate: {error_rate:.0%}"
            elif error_rate > 0.2:
                coordinator_status = CoordinatorStatus.DEGRADED
                message = f"Elevated error rate: {error_rate:.0%}"

        # Check if any backends are loaded
        backends_loaded_count = sum(1 for v in self._backends_loaded.values() if v)
        if backends_loaded_count == 0 and total_syncs == 0:
            # No backends loaded yet (normal at startup)
            coordinator_status = CoordinatorStatus.RUNNING
        elif backends_loaded_count == 0 and total_syncs > 0:
            # Tried syncs but no backends available
            coordinator_status = CoordinatorStatus.DEGRADED
            message = "No sync backends available"

        healthy = coordinator_status == CoordinatorStatus.RUNNING
        return HealthCheckResult(
            healthy=healthy,
            status=coordinator_status,
            message=message,
            details={
                "operations_count": total_syncs,
                "errors_count": errors_count,
                "backends_loaded": backends_loaded_count,
                "total_bytes_transferred": self._stats.get("total_bytes", 0),
                "by_backend": self._stats.get("by_backend", {}),
            },
        )

    @classmethod
    def get_instance(cls) -> SyncFacade:
        """Get singleton instance of SyncFacade.

        Use this method instead of direct instantiation to ensure
        a single SyncFacade is shared across the application.

        Returns:
            SyncFacade: The singleton instance.
        """
        if not hasattr(cls, "_instance"):
            cls._instance = cls()
        return cls._instance


# =============================================================================
# Module-level convenience functions
# =============================================================================

_facade_singleton: SyncFacade | None = None


def get_sync_facade() -> SyncFacade:
    """Get the global SyncFacade singleton.

    Preferred module-level accessor for sync operations.
    Thread-safe initialization on first call.

    Returns:
        SyncFacade: The global singleton instance.

    Example:
        facade = get_sync_facade()
        response = await facade.sync("games")
    """
    global _facade_singleton
    if _facade_singleton is None:
        _facade_singleton = SyncFacade()
    return _facade_singleton


def reset_sync_facade() -> None:
    """Reset the sync facade singleton.

    Primarily for testing - clears the cached singleton instance,
    allowing a fresh facade to be created on next access.

    Note:
        This does NOT stop any running sync operations.
        Call this only when you need a clean slate (e.g., between tests).
    """
    global _facade_singleton
    _facade_singleton = None


async def sync(
    data_type: str,
    targets: list[str] | None = None,
    **kwargs,
) -> SyncResponse:
    """Convenience function to sync data.

    Args:
        data_type: Type of data to sync ("games", "models", "npz", "all")
        targets: Target nodes (None = all eligible)
        **kwargs: Additional SyncRequest parameters

    Returns:
        SyncResponse

    Examples:
        # Sync all games
        await sync("games")

        # Sync models to specific nodes
        await sync("models", targets=["node-1", "node-2"])

        # High-priority sync
        await sync("games", board_type="hex8", priority="high")
    """
    facade = get_sync_facade()
    request = SyncRequest(data_type=data_type, targets=targets, **kwargs)
    return await facade.sync(request)
