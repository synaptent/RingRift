"""P2P Event Emission Mixin - Consolidated Event Emission for P2P Orchestrator.

This module provides a mixin class that consolidates all event emission methods
for the P2P orchestrator, reducing ~935 lines in p2p_orchestrator.py to a single
mixin import.

Created: December 28, 2025
Phase 8 of P2P orchestrator decomposition.

Consolidates 27 `_emit_*` functions:
- Host lifecycle: HOST_OFFLINE, HOST_ONLINE, NODE_DEAD
- Leader events: LEADER_ELECTED, LEADER_LOST
- Cluster health: CLUSTER_HEALTHY, CLUSTER_UNHEALTHY, SPLIT_BRAIN_DETECTED
- Data sync: DATA_SYNC_STARTED, DATA_SYNC_COMPLETED, DATA_SYNC_FAILED
- Model events: MODEL_DISTRIBUTION_STARTED/COMPLETE/FAILED, MODEL_PROMOTED
- Batch events: BATCH_SCHEDULED, BATCH_DISPATCHED
- Task events: TASK_ABANDONED
- Capacity: CLUSTER_CAPACITY_CHANGED

Usage:
    class P2POrchestrator(EventEmissionMixin, ...):
        pass

    # Then use like:
    await self._emit_host_offline(node_id, reason="timeout")
    await self._emit_leader_elected(leader_id, term=5)
"""

from __future__ import annotations

import asyncio

from app.core.async_context import safe_create_task
import logging
from typing import TYPE_CHECKING, Any, ClassVar

from scripts.p2p.p2p_mixin_base import P2PMixinBase

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def _check_event_emitters() -> bool:
    """Check if event emission infrastructure is available.

    Returns True if we should attempt event emission, False otherwise.
    This is a module-level function to avoid repeated import attempts.
    """
    try:
        # Quick check if event router is importable
        import importlib

        importlib.import_module("app.coordination.event_router")
        return True
    except ImportError:
        return False


class EventEmissionMixin(P2PMixinBase):
    """Mixin providing consolidated event emission for P2P orchestrator.

    All methods are safe - they catch exceptions and log instead of raising.
    This ensures event emission failures don't crash the orchestrator.

    Inherits from P2PMixinBase (Dec 29, 2025) to share common functionality:
    - Database helpers, state initialization, peer management
    - Logging helpers (_log_debug, _log_info, etc.)
    - Event emission circuit breaker

    Required parent class attributes (inherited from P2PMixinBase):
    - node_id: str - This node's identifier
    - verbose: bool - Verbose logging flag (optional)
    """

    # Class identifier for logging (overrides P2PMixinBase.MIXIN_TYPE)
    MIXIN_TYPE: ClassVar[str] = "event_emission"

    # Cache for event emitter availability
    _event_emitters_available: bool | None = None

    # =========================================================================
    # Generic Event Emission Helper
    # =========================================================================

    async def _emit_event_safe(
        self,
        emit_func_name: str,
        event_name: str,
        context_id: str = "",
        log_level: str = "debug",
        source_module: str = "app.coordination.event_router",
        **kwargs: Any,
    ) -> bool:
        """Generic helper for safe event emission with consistent error handling.

        Consolidates the try/except/import pattern used by all emit functions.
        Reduces boilerplate while maintaining safety guarantees.

        Args:
            emit_func_name: Name of emit function (e.g., "emit_host_offline")
            event_name: Human-readable event name for logging (e.g., "HOST_OFFLINE")
            context_id: Identifier for logging (e.g., node_id)
            log_level: Log level for success message ("debug" or "info")
            source_module: Module containing the emit function
            **kwargs: Keyword arguments to pass to the emit function

        Returns:
            True if emitted successfully, False otherwise
        """
        if not _check_event_emitters():
            return False

        try:
            import importlib

            module = importlib.import_module(source_module)
            emit_func = getattr(module, emit_func_name)
            await emit_func(**kwargs)

            msg = f"[P2P Event] Emitted {event_name}"
            if context_id:
                msg += f" for {context_id}"
            if log_level == "info":
                logger.info(msg)
            else:
                logger.debug(msg)
            return True

        except (ImportError, AttributeError, RuntimeError, TypeError) as e:
            logger.debug(f"[P2P Event] Failed to emit {event_name}: {e}")
            return False

    def _emit_event_sync(
        self,
        emit_func_name: str,
        event_name: str,
        context_id: str = "",
        **kwargs: Any,
    ) -> bool:
        """Synchronous wrapper for event emission via fire-and-forget task.

        For use in sync code paths. Creates an async task if event loop is running.

        Args:
            emit_func_name: Name of emit function
            event_name: Human-readable event name for logging
            context_id: Identifier for logging
            **kwargs: Keyword arguments to pass to the emit function

        Returns:
            True if task was created, False otherwise
        """
        if not _check_event_emitters():
            return False

        try:
            asyncio.get_running_loop()
            safe_create_task(
                self._emit_event_safe(emit_func_name, event_name, context_id, **kwargs),
                name=f"emit-{event_name}",
            )
            return True
        except RuntimeError:
            # No running event loop
            return False

    # =========================================================================
    # Host Lifecycle Events
    # =========================================================================

    async def _emit_host_offline(
        self,
        node_id: str,
        reason: str = "timeout",
        last_seen: float | None = None,
    ) -> bool:
        """Emit HOST_OFFLINE event when a peer goes offline/is retired.

        Args:
            node_id: The node that went offline
            reason: Why the node went offline (timeout, retired, error)
            last_seen: Timestamp when node was last seen
        """
        return await self._emit_event_safe(
            "emit_host_offline",
            "HOST_OFFLINE",
            node_id,
            host=node_id,
            reason=reason,
            last_seen=last_seen,
            source="p2p_orchestrator",
        )

    def _emit_host_offline_sync(
        self,
        node_id: str,
        reason: str = "timeout",
        last_seen: float | None = None,
    ) -> bool:
        """Synchronous version of _emit_host_offline."""
        return self._emit_event_sync(
            "emit_host_offline",
            "HOST_OFFLINE",
            node_id,
            host=node_id,
            reason=reason,
            last_seen=last_seen,
            source="p2p_orchestrator",
        )

    async def _emit_host_online(
        self,
        node_id: str,
        capabilities: list[str] | None = None,
    ) -> bool:
        """Emit HOST_ONLINE event when a peer comes back online.

        Args:
            node_id: The node that came back online
            capabilities: List of node capabilities (gpu, cpu, etc.)
        """
        return await self._emit_event_safe(
            "emit_host_online",
            "HOST_ONLINE",
            node_id,
            host=node_id,
            capabilities=capabilities or [],
            source="p2p_orchestrator",
        )

    def _emit_host_online_sync(
        self,
        node_id: str,
        capabilities: list[str] | None = None,
    ) -> bool:
        """Synchronous version of _emit_host_online."""
        return self._emit_event_sync(
            "emit_host_online",
            "HOST_ONLINE",
            node_id,
            host=node_id,
            capabilities=capabilities or [],
            source="p2p_orchestrator",
        )

    async def _emit_node_dead(
        self,
        node_id: str,
        reason: str = "unresponsive",
        last_seen: float | None = None,
        job_count: int = 0,
    ) -> bool:
        """Emit P2P_NODE_DEAD event when a node is declared dead.

        More severe than HOST_OFFLINE - indicates node removal from cluster.

        Args:
            node_id: The dead node
            reason: Why node is dead (unresponsive, crashed, terminated)
            last_seen: Timestamp when node was last seen
            job_count: Number of jobs that were running on the node
        """
        return await self._emit_event_safe(
            "emit_node_dead",
            "P2P_NODE_DEAD",
            node_id,
            node_id=node_id,
            reason=reason,
            last_seen=last_seen,
            job_count=job_count,
            source="p2p_orchestrator",
        )

    def _emit_node_dead_sync(
        self,
        node_id: str,
        reason: str = "unresponsive",
        last_seen: float | None = None,
        job_count: int = 0,
    ) -> bool:
        """Synchronous version of _emit_node_dead."""
        return self._emit_event_sync(
            "emit_node_dead",
            "P2P_NODE_DEAD",
            node_id,
            node_id=node_id,
            reason=reason,
            last_seen=last_seen,
            job_count=job_count,
            source="p2p_orchestrator",
        )

    async def _emit_node_suspect(
        self,
        node_id: str,
        last_seen: float | None = None,
        seconds_since_heartbeat: float = 0.0,
    ) -> bool:
        """Emit NODE_SUSPECT event when a node enters SUSPECT state.

        December 2025: SUSPECT is a grace period between ALIVE and DEAD.
        Enables health monitors to track nodes that may be experiencing
        transient issues before declaring them dead.

        Args:
            node_id: The suspect node
            last_seen: Timestamp when node was last seen
            seconds_since_heartbeat: Seconds since last heartbeat
        """
        return await self._emit_event_safe(
            "emit_node_suspect",
            "NODE_SUSPECT",
            node_id,
            log_level="info",
            node_id=node_id,
            last_seen=last_seen,
            seconds_since_heartbeat=seconds_since_heartbeat,
            source="p2p_orchestrator",
        )

    def _emit_node_suspect_sync(
        self,
        node_id: str,
        last_seen: float | None = None,
        seconds_since_heartbeat: float = 0.0,
    ) -> bool:
        """Synchronous version of _emit_node_suspect."""
        return self._emit_event_sync(
            "emit_node_suspect",
            "NODE_SUSPECT",
            node_id,
            node_id=node_id,
            last_seen=last_seen,
            seconds_since_heartbeat=seconds_since_heartbeat,
            source="p2p_orchestrator",
        )

    async def _emit_node_retired(
        self,
        node_id: str,
        reason: str = "timeout",
        last_seen: float | None = None,
        total_uptime_seconds: float = 0.0,
    ) -> bool:
        """Emit NODE_RETIRED event when a node is retired from the cluster.

        December 2025: Retired nodes are excluded from job allocation but
        may be recovered later via the PeerRecoveryLoop.

        Args:
            node_id: The retired node
            reason: Why the node was retired (timeout, manual, error)
            last_seen: Timestamp when node was last seen
            total_uptime_seconds: Total uptime before retirement
        """
        return await self._emit_event_safe(
            "emit_node_retired",
            "NODE_RETIRED",
            node_id,
            log_level="info",
            node_id=node_id,
            reason=reason,
            last_seen=last_seen,
            total_uptime_seconds=total_uptime_seconds,
            source="p2p_orchestrator",
        )

    def _emit_node_retired_sync(
        self,
        node_id: str,
        reason: str = "timeout",
        last_seen: float | None = None,
        total_uptime_seconds: float = 0.0,
    ) -> bool:
        """Synchronous version of _emit_node_retired."""
        return self._emit_event_sync(
            "emit_node_retired",
            "NODE_RETIRED",
            node_id,
            node_id=node_id,
            reason=reason,
            last_seen=last_seen,
            total_uptime_seconds=total_uptime_seconds,
            source="p2p_orchestrator",
        )

    async def _emit_node_recovered(
        self,
        node_id: str,
        recovery_type: str = "automatic",
        offline_duration_seconds: float = 0.0,
    ) -> bool:
        """Emit NODE_RECOVERED event when a node recovers to healthy state.

        December 2025: Emitted on transitions:
        - SUSPECT -> ALIVE (recovered before DEAD)
        - DEAD -> ALIVE (recovered from dead state)
        - RETIRED -> ALIVE (recovered after retirement)

        Args:
            node_id: The recovered node
            recovery_type: How the node recovered (automatic, manual, heartbeat)
            offline_duration_seconds: How long the node was offline
        """
        return await self._emit_event_safe(
            "emit_node_recovered",
            "NODE_RECOVERED",
            node_id,
            log_level="info",
            node_id=node_id,
            recovery_type=recovery_type,
            offline_duration_seconds=offline_duration_seconds,
            source="p2p_orchestrator",
        )

    def _emit_node_recovered_sync(
        self,
        node_id: str,
        recovery_type: str = "automatic",
        offline_duration_seconds: float = 0.0,
    ) -> bool:
        """Synchronous version of _emit_node_recovered."""
        return self._emit_event_sync(
            "emit_node_recovered",
            "NODE_RECOVERED",
            node_id,
            node_id=node_id,
            recovery_type=recovery_type,
            offline_duration_seconds=offline_duration_seconds,
            source="p2p_orchestrator",
        )

    # =========================================================================
    # Leader Events
    # =========================================================================

    async def _emit_leader_elected(
        self,
        leader_id: str,
        term: int = 0,
    ) -> bool:
        """Emit LEADER_ELECTED event when this node becomes leader.

        Args:
            leader_id: The new leader node ID
            term: Election term number
        """
        result = await self._emit_event_safe(
            "emit_leader_elected",
            "LEADER_ELECTED",
            leader_id,
            log_level="info",
            leader_id=leader_id,
            term=term,
            source="p2p_orchestrator",
        )

        # Jan 4, 2026: Propagate leader info via gossip for NAT-blocked nodes
        # This ensures all nodes (including those behind NAT) learn about
        # the new leader through gossip state propagation
        if leader_id == getattr(self, "node_id", None):
            if hasattr(self, "_propagate_leader_via_gossip"):
                try:
                    self._propagate_leader_via_gossip(force=True)
                except Exception as e:
                    # Don't let propagation failure affect election
                    import logging
                    logging.getLogger(__name__).debug(
                        f"Leader propagation failed (non-critical): {e}"
                    )

        return result

    async def _emit_leader_lost(
        self,
        old_leader_id: str,
        reason: str = "",
    ) -> bool:
        """Emit LEADER_LOST event when leader becomes unavailable.

        Enables LeadershipCoordinator, SyncRouter, and other coordinators
        to react to leader loss in real-time.

        Args:
            old_leader_id: The previous leader node ID
            reason: Reason for leader loss (dead, ineligible, expired)
        """
        return await self._emit_event_safe(
            "emit_leader_lost",
            "LEADER_LOST",
            old_leader_id,
            log_level="info",
            old_leader_id=old_leader_id,
            reason=reason,
            source="p2p_orchestrator",
        )

    def _emit_leader_lost_sync(
        self,
        old_leader_id: str,
        reason: str = "",
    ) -> bool:
        """Synchronous version of _emit_leader_lost."""
        return self._emit_event_sync(
            "emit_leader_lost",
            "LEADER_LOST",
            old_leader_id,
            old_leader_id=old_leader_id,
            reason=reason,
            source="p2p_orchestrator",
        )

    # =========================================================================
    # Cluster Health Events
    # =========================================================================

    async def _emit_cluster_capacity_changed(
        self,
        total_nodes: int,
        alive_nodes: int,
        gpu_nodes: int,
        training_nodes: int,
        change_type: str = "node_count",
        change_details: dict[str, Any] | None = None,
    ) -> bool:
        """Emit CLUSTER_CAPACITY_CHANGED event when cluster size changes.

        Args:
            total_nodes: Total configured nodes
            alive_nodes: Currently alive nodes
            gpu_nodes: Nodes with GPUs
            training_nodes: Nodes available for training
            change_type: Type of change (node_count, gpu_availability)
            change_details: Additional details about the change
        """
        return await self._emit_event_safe(
            "emit_cluster_capacity_changed",
            "CLUSTER_CAPACITY_CHANGED",
            f"{alive_nodes}/{total_nodes}",
            total_nodes=total_nodes,
            alive_nodes=alive_nodes,
            gpu_nodes=gpu_nodes,
            training_nodes=training_nodes,
            change_type=change_type,
            change_details=change_details or {},
            source="p2p_orchestrator",
        )

    def _emit_cluster_capacity_changed_sync(
        self,
        total_nodes: int,
        alive_nodes: int,
        gpu_nodes: int,
        training_nodes: int,
        change_type: str = "node_count",
        change_details: dict[str, Any] | None = None,
    ) -> bool:
        """Synchronous version of _emit_cluster_capacity_changed."""
        return self._emit_event_sync(
            "emit_cluster_capacity_changed",
            "CLUSTER_CAPACITY_CHANGED",
            f"{alive_nodes}/{total_nodes}",
            total_nodes=total_nodes,
            alive_nodes=alive_nodes,
            gpu_nodes=gpu_nodes,
            training_nodes=training_nodes,
            change_type=change_type,
            change_details=change_details or {},
            source="p2p_orchestrator",
        )

    async def _emit_cluster_healthy(
        self,
        alive_peers: int,
        quorum_met: bool = True,
        leader_id: str | None = None,
    ) -> bool:
        """Emit P2P_CLUSTER_HEALTHY when cluster is in healthy state.

        Args:
            alive_peers: Number of alive peers
            quorum_met: Whether voter quorum is met
            leader_id: Current leader node ID
        """
        return await self._emit_event_safe(
            "emit_cluster_healthy",
            "P2P_CLUSTER_HEALTHY",
            f"{alive_peers} peers",
            alive_peers=alive_peers,
            quorum_met=quorum_met,
            leader_id=leader_id,
            source="p2p_orchestrator",
        )

    async def _emit_cluster_unhealthy(
        self,
        alive_peers: int,
        quorum_met: bool,
        reason: str = "",
        leader_id: str | None = None,
    ) -> bool:
        """Emit P2P_CLUSTER_UNHEALTHY when cluster health degrades.

        Args:
            alive_peers: Number of alive peers
            quorum_met: Whether voter quorum is met
            reason: Reason for unhealthy state
            leader_id: Current leader node ID
        """
        return await self._emit_event_safe(
            "emit_cluster_unhealthy",
            "P2P_CLUSTER_UNHEALTHY",
            reason,
            log_level="info",
            alive_peers=alive_peers,
            quorum_met=quorum_met,
            reason=reason,
            leader_id=leader_id,
            source="p2p_orchestrator",
        )

    def _emit_cluster_health_event_sync(
        self,
        is_healthy: bool,
        alive_peers: int,
        quorum_met: bool,
        leader_id: str | None = None,
        reason: str = "",
    ) -> bool:
        """Synchronous wrapper for cluster health events."""
        if is_healthy:
            return self._emit_event_sync(
                "emit_cluster_healthy",
                "P2P_CLUSTER_HEALTHY",
                f"{alive_peers} peers",
                alive_peers=alive_peers,
                quorum_met=quorum_met,
                leader_id=leader_id,
                source="p2p_orchestrator",
            )
        else:
            return self._emit_event_sync(
                "emit_cluster_unhealthy",
                "P2P_CLUSTER_UNHEALTHY",
                reason,
                alive_peers=alive_peers,
                quorum_met=quorum_met,
                reason=reason,
                leader_id=leader_id,
                source="p2p_orchestrator",
            )

    async def _emit_split_brain_detected(
        self,
        detected_leaders: list[str],
        our_leader: str | None = None,
        resolution_action: str = "election",
    ) -> bool:
        """Emit SPLIT_BRAIN_DETECTED when multiple leaders are detected.

        Critical event that triggers immediate resolution action.

        Args:
            detected_leaders: List of node IDs claiming leadership
            our_leader: Our current leader (if any)
            resolution_action: How we're resolving (election, stepdown)
        """
        return await self._emit_event_safe(
            "emit_split_brain_detected",
            "SPLIT_BRAIN_DETECTED",
            f"{len(detected_leaders)} leaders",
            log_level="info",
            detected_leaders=detected_leaders,
            our_leader=our_leader,
            resolution_action=resolution_action,
            source="p2p_orchestrator",
        )

    def _emit_split_brain_detected_sync(
        self,
        detected_leaders: list[str],
        our_leader: str | None = None,
        resolution_action: str = "election",
    ) -> bool:
        """Synchronous version of _emit_split_brain_detected."""
        return self._emit_event_sync(
            "emit_split_brain_detected",
            "SPLIT_BRAIN_DETECTED",
            f"{len(detected_leaders)} leaders",
            detected_leaders=detected_leaders,
            our_leader=our_leader,
            resolution_action=resolution_action,
            source="p2p_orchestrator",
        )

    # =========================================================================
    # Task Events
    # =========================================================================

    async def _emit_task_abandoned(
        self,
        job_id: str,
        config_key: str,
        node_id: str,
        reason: str = "unknown",
        progress: float = 0.0,
    ) -> bool:
        """Emit TASK_ABANDONED when a task is intentionally cancelled.

        Distinct from TASK_FAILED - indicates deliberate cancellation.

        Args:
            job_id: Job identifier
            config_key: Configuration key
            node_id: Node that was running the task
            reason: Why task was abandoned
            progress: Progress when abandoned (0.0-1.0)
        """
        return await self._emit_event_safe(
            "emit_task_abandoned",
            "TASK_ABANDONED",
            job_id,
            job_id=job_id,
            config_key=config_key,
            node_id=node_id,
            reason=reason,
            progress=progress,
            source="p2p_orchestrator",
        )

    # =========================================================================
    # Data Sync Events
    # =========================================================================

    async def _emit_data_sync_started(
        self,
        sync_type: str,
        source_node: str | None = None,
        target_nodes: list[str] | None = None,
        file_count: int = 0,
        config_key: str | None = None,
    ) -> bool:
        """Emit DATA_SYNC_STARTED when sync operation begins.

        Args:
            sync_type: Type of sync (games, models, npz)
            source_node: Source node for sync
            target_nodes: Target nodes for sync
            file_count: Number of files to sync
            config_key: Configuration key if sync is config-specific
        """
        return await self._emit_event_safe(
            "emit_data_sync_started",
            "DATA_SYNC_STARTED",
            sync_type,
            sync_type=sync_type,
            source_node=source_node,
            target_nodes=target_nodes or [],
            file_count=file_count,
            config_key=config_key,
            source="p2p_orchestrator",
        )

    async def _emit_data_sync_completed(
        self,
        sync_type: str,
        duration_seconds: float,
        files_synced: int,
        bytes_transferred: int = 0,
        source_node: str | None = None,
        target_nodes: list[str] | None = None,
        config_key: str | None = None,
    ) -> bool:
        """Emit DATA_SYNC_COMPLETED when sync operation finishes successfully.

        Args:
            sync_type: Type of sync (games, models, npz)
            duration_seconds: How long sync took
            files_synced: Number of files successfully synced
            bytes_transferred: Total bytes transferred
            source_node: Source node for sync
            target_nodes: Target nodes for sync
            config_key: Configuration key if sync is config-specific
        """
        return await self._emit_event_safe(
            "emit_data_sync_completed",
            "DATA_SYNC_COMPLETED",
            f"{sync_type}/{files_synced} files",
            sync_type=sync_type,
            duration_seconds=duration_seconds,
            files_synced=files_synced,
            bytes_transferred=bytes_transferred,
            source_node=source_node,
            target_nodes=target_nodes or [],
            config_key=config_key,
            source="p2p_orchestrator",
        )

    async def _emit_data_sync_failed(
        self,
        sync_type: str,
        error: str,
        source_node: str | None = None,
        target_nodes: list[str] | None = None,
        config_key: str | None = None,
        retry_count: int = 0,
    ) -> bool:
        """Emit DATA_SYNC_FAILED when sync operation fails.

        Args:
            sync_type: Type of sync (games, models, npz)
            error: Error message
            source_node: Source node for sync
            target_nodes: Target nodes for sync
            config_key: Configuration key if sync is config-specific
            retry_count: Number of retry attempts
        """
        return await self._emit_event_safe(
            "emit_data_sync_failed",
            "DATA_SYNC_FAILED",
            f"{sync_type}: {error}",
            log_level="info",
            sync_type=sync_type,
            error=error,
            source_node=source_node,
            target_nodes=target_nodes or [],
            config_key=config_key,
            retry_count=retry_count,
            source="p2p_orchestrator",
        )

    # =========================================================================
    # Model Distribution Events
    # =========================================================================

    async def _emit_model_distribution_started(
        self,
        model_id: str,
        config_key: str,
        target_nodes: list[str],
        model_size_bytes: int = 0,
    ) -> bool:
        """Emit MODEL_DISTRIBUTION_STARTED when model distribution begins.

        Args:
            model_id: Model identifier
            config_key: Configuration key
            target_nodes: Nodes to distribute to
            model_size_bytes: Size of model file
        """
        return await self._emit_event_safe(
            "emit_model_distribution_started",
            "MODEL_DISTRIBUTION_STARTED",
            f"{model_id}/{config_key}",
            model_id=model_id,
            config_key=config_key,
            target_nodes=target_nodes,
            model_size_bytes=model_size_bytes,
            source="p2p_orchestrator",
        )

    async def _emit_model_distribution_complete(
        self,
        model_id: str,
        config_key: str,
        nodes_succeeded: list[str],
        nodes_failed: list[str] | None = None,
        duration_seconds: float = 0.0,
    ) -> bool:
        """Emit MODEL_DISTRIBUTION_COMPLETE when model distribution finishes.

        Args:
            model_id: Model identifier
            config_key: Configuration key
            nodes_succeeded: Nodes that received model
            nodes_failed: Nodes that failed to receive model
            duration_seconds: How long distribution took
        """
        return await self._emit_event_safe(
            "emit_model_distribution_complete",
            "MODEL_DISTRIBUTION_COMPLETE",
            f"{model_id}/{config_key}",
            model_id=model_id,
            config_key=config_key,
            nodes_succeeded=nodes_succeeded,
            nodes_failed=nodes_failed or [],
            duration_seconds=duration_seconds,
            source="p2p_orchestrator",
        )

    async def _emit_model_distribution_failed(
        self,
        model_id: str,
        config_key: str,
        error: str,
        target_nodes: list[str] | None = None,
    ) -> bool:
        """Emit MODEL_DISTRIBUTION_FAILED when model distribution fails.

        Args:
            model_id: Model identifier
            config_key: Configuration key
            error: Error message
            target_nodes: Nodes that were targeted
        """
        return await self._emit_event_safe(
            "emit_model_distribution_failed",
            "MODEL_DISTRIBUTION_FAILED",
            f"{model_id}/{config_key}: {error}",
            log_level="info",
            model_id=model_id,
            config_key=config_key,
            error=error,
            target_nodes=target_nodes or [],
            source="p2p_orchestrator",
        )

    async def _emit_model_promoted(
        self,
        model_id: str,
        config_key: str,
        elo: float = 0.0,
        elo_gain: float = 0.0,
    ) -> bool:
        """Emit MODEL_PROMOTED when a model is promoted to baseline.

        Enables:
        - Model distribution daemon to sync model to cluster
        - SelfplayModelSelector to hot-reload model cache
        - Temperature scheduling to adjust exploration
        - Metrics recording

        Args:
            model_id: Model identifier (e.g., filename or path)
            config_key: Configuration key (e.g., "hex8_2p")
            elo: Model's Elo rating (if known)
            elo_gain: Elo gain over previous baseline (if known)
        """
        if not _check_event_emitters():
            return False

        try:
            from app.distributed.event_helpers import emit_model_promoted_safe

            await emit_model_promoted_safe(
                model_id=model_id,
                config=config_key,
                elo=elo,
                elo_gain=elo_gain,
                source="p2p_orchestrator",
            )
            logger.info(
                f"[P2P Event] Emitted MODEL_PROMOTED: {model_id} "
                f"(config: {config_key}, elo: {elo}, gain: {elo_gain})"
            )
            return True
        except ImportError:
            return False
        except (AttributeError, RuntimeError, TypeError) as e:
            logger.debug(f"[P2P Event] Failed to emit MODEL_PROMOTED: {e}")
            return False

    # =========================================================================
    # Batch Events
    # =========================================================================

    async def _emit_batch_scheduled(
        self,
        batch_id: str,
        batch_type: str,
        config_key: str,
        job_count: int,
        target_nodes: list[str],
        reason: str = "normal",
    ) -> bool:
        """Emit BATCH_SCHEDULED when a batch of jobs is scheduled.

        Enables DataPipelineOrchestrator to track batch operations.

        Args:
            batch_id: Unique batch identifier
            batch_type: Type of batch ("selfplay", "training", "tournament")
            config_key: Configuration key (e.g., "hex8_2p")
            job_count: Number of jobs in the batch
            target_nodes: List of node IDs selected for dispatch
            reason: Why the batch was scheduled
        """
        try:
            from app.distributed.data_events import emit_batch_scheduled

            await emit_batch_scheduled(
                batch_id=batch_id,
                batch_type=batch_type,
                config_key=config_key,
                job_count=job_count,
                target_nodes=target_nodes,
                reason=reason,
                source="p2p_orchestrator",
            )
            logger.debug(
                f"[P2P Event] Emitted BATCH_SCHEDULED: {batch_type}/{config_key} "
                f"({job_count} jobs to {len(target_nodes)} nodes)"
            )
            return True
        except ImportError:
            return False
        except (AttributeError, RuntimeError, TypeError) as e:
            logger.debug(f"[P2P Event] Failed to emit BATCH_SCHEDULED: {e}")
            return False

    async def _emit_batch_dispatched(
        self,
        batch_id: str,
        batch_type: str,
        config_key: str,
        jobs_dispatched: int,
        jobs_failed: int = 0,
        target_nodes: list[str] | None = None,
    ) -> bool:
        """Emit BATCH_DISPATCHED when batch jobs are sent to nodes.

        Enables DataPipelineOrchestrator to track batch completion.

        Args:
            batch_id: Unique batch identifier (same as BATCH_SCHEDULED)
            batch_type: Type of batch ("selfplay", "training", "tournament")
            config_key: Configuration key (e.g., "hex8_2p")
            jobs_dispatched: Number of jobs successfully dispatched
            jobs_failed: Number of jobs that failed to dispatch
            target_nodes: List of node IDs that received jobs
        """
        try:
            from app.distributed.data_events import emit_batch_dispatched

            await emit_batch_dispatched(
                batch_id=batch_id,
                batch_type=batch_type,
                config_key=config_key,
                jobs_dispatched=jobs_dispatched,
                jobs_failed=jobs_failed,
                target_nodes=target_nodes,
                source="p2p_orchestrator",
            )
            logger.debug(
                f"[P2P Event] Emitted BATCH_DISPATCHED: {batch_type}/{config_key} "
                f"({jobs_dispatched} dispatched, {jobs_failed} failed)"
            )
            return True
        except ImportError:
            return False
        except (AttributeError, RuntimeError, TypeError) as e:
            logger.debug(f"[P2P Event] Failed to emit BATCH_DISPATCHED: {e}")
            return False

    # =========================================================================
    # Health Check Implementation (Dec 29, 2025)
    # =========================================================================

    def health_check(self) -> dict[str, Any]:
        """Return health status for event emission subsystem.

        Checks if event emitters are available and functional.
        Consistent with ConsensusMixin and MembershipMixin patterns.

        Returns:
            dict with keys:
                - healthy: bool - Overall health status
                - message: str - Human-readable status message
                - details: dict - Detailed health information
        """
        # Check if event emission infrastructure is available
        if EventEmissionMixin._event_emitters_available is None:
            EventEmissionMixin._event_emitters_available = _check_event_emitters()

        is_healthy = EventEmissionMixin._event_emitters_available

        message = (
            "Event emitters available"
            if is_healthy
            else "Event emitters unavailable"
        )

        details = {
            "event_emitters_available": is_healthy,
            "mixin_type": self.MIXIN_TYPE,
            "node_id": getattr(self, "node_id", "unknown"),
        }

        return self._build_health_response(is_healthy, message, details)
