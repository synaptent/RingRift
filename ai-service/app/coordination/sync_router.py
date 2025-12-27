"""Sync Router - Intelligent Data Routing Based on Node Capabilities.

Routes data (games, models, NPZ files) to appropriate nodes based on:
- Node role (training, selfplay, coordinator)
- Disk capacity and usage limits
- Storage type (persistent, ephemeral, NFS)
- Network topology (same provider, NFS sharing)
- Exclusion rules

Usage:
    from app.coordination.sync_router import SyncRouter, get_sync_router

    router = get_sync_router()

    # Get sync targets for game data
    targets = router.get_sync_targets(data_type="game", board_type="hex8")

    # Check if a node should receive specific data
    if router.should_sync_to_node("gpu-node-1", data_type="game"):
        sync_to_node(...)
"""

from __future__ import annotations

import logging
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config.cluster_config import load_cluster_config, get_host_provider
from app.distributed.cluster_manifest import (
    ClusterManifest,
    DataType,
    SyncTarget,
    get_cluster_manifest,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Data classes
    "SyncRoute",
    "NodeSyncCapability",
    # Main class
    "SyncRouter",
    # Singleton accessors
    "get_sync_router",
    "reset_sync_router",
]


@dataclass
class SyncRoute:
    """A sync route with source and target information."""
    source_node: str
    target_node: str
    data_type: DataType
    priority: int = 0
    reason: str = ""
    estimated_size_bytes: int = 0
    bandwidth_limit_mbps: int | None = None
    quality_score: float = 0.0  # Dec 2025: Quality-based priority boost


@dataclass
class NodeSyncCapability:
    """Sync capabilities for a node."""
    node_id: str
    can_receive_games: bool = True
    can_receive_models: bool = True
    can_receive_npz: bool = True
    is_training_node: bool = False
    is_priority_node: bool = False
    is_ephemeral: bool = False
    shares_nfs: bool = False
    provider: str = "unknown"
    disk_usage_percent: float = 0.0
    available_gb: float = 0.0


class SyncRouter:
    """Routes data to nodes based on capabilities and policies.

    Integrates with ClusterManifest for:
    - Exclusion rules and sync policies
    - Disk capacity checks
    - Replication tracking
    """

    # P2.3 Dec 2025: Capacity refresh interval (5 minutes)
    CAPACITY_REFRESH_INTERVAL = 300.0

    def __init__(
        self,
        config_path: Path | None = None,
        manifest: ClusterManifest | None = None,
    ):
        """Initialize the sync router.

        Args:
            config_path: Path to distributed_hosts.yaml
            manifest: ClusterManifest instance (uses singleton if None)
        """
        self.node_id = socket.gethostname()
        self._manifest = manifest or get_cluster_manifest()

        # Load host configuration
        self._hosts_config: dict[str, Any] = {}
        self._sync_routing: dict[str, Any] = {}
        self._node_capabilities: dict[str, NodeSyncCapability] = {}
        self._load_config(config_path)

        # P2.3 Dec 2025: Capacity refresh tracking
        self._last_capacity_refresh = 0.0

        logger.info(f"SyncRouter initialized: {len(self._node_capabilities)} nodes")

    def _load_config(self, config_path: Path | None = None) -> None:
        """Load configuration from distributed_hosts.yaml using cluster_config.

        Uses the consolidated cluster_config module instead of inline yaml loading.
        """
        try:
            cluster_config = load_cluster_config(config_path)

            self._hosts_config = cluster_config.hosts_raw
            self._sync_routing = cluster_config.get_raw_section("sync_routing")

            # Dec 2025: Load allowed_external_storage for coordinator backup
            self._external_storage: list[dict[str, Any]] = [
                {
                    "host": storage.host,
                    "path": storage.path,
                    "receive_games": storage.receive_games,
                    "receive_npz": storage.receive_npz,
                    "receive_models": storage.receive_models,
                    "subdirs": storage.subdirs,
                }
                for storage in cluster_config.sync_routing.allowed_external_storage
            ]

            # Build node capabilities from hosts config
            self._build_node_capabilities()

            # Dec 2025: Log external storage config
            if self._external_storage:
                for storage in self._external_storage:
                    logger.info(
                        f"[SyncRouter] External storage configured: "
                        f"{storage.get('host')} -> {storage.get('path')}"
                    )

        except Exception as e:
            logger.error(f"Failed to load config: {e}")

    def _build_node_capabilities(self) -> None:
        """Build node capability information from config."""
        priority_hosts = set(self._sync_routing.get("priority_hosts", []))

        # Dec 2025: Build external storage lookup for coordinator backup
        external_storage_hosts = {}
        for storage in getattr(self, "_external_storage", []):
            host = storage.get("host", "")
            if host:
                external_storage_hosts[host] = storage

        for host_name, host_config in self._hosts_config.items():
            role = host_config.get("role", "selfplay")
            gpu = host_config.get("gpu", "")

            # Use consolidated provider detection from cluster_config
            provider = get_host_provider(host_name)

            # Check if shares NFS (Lambda nodes with same provider)
            shares_nfs = provider == "lambda"

            # Check if ephemeral (Vast.ai)
            is_ephemeral = provider == "vast"

            # Check if training node
            is_training = "training" in role or host_name in priority_hosts

            # Get sync policy from manifest
            policy = self._manifest.get_sync_policy(host_name)

            # Dec 2025: Override with external storage config if present
            # This enables coordinator backup via external drives
            if host_name in external_storage_hosts:
                ext_config = external_storage_hosts[host_name]
                can_receive_games = ext_config.get("receive_games", False)
                can_receive_models = ext_config.get("receive_models", False)
                can_receive_npz = ext_config.get("receive_npz", False)
                is_priority = True  # External storage is priority for backup
            else:
                can_receive_games = policy.receive_games
                can_receive_models = policy.receive_models
                can_receive_npz = policy.receive_npz
                is_priority = host_name in priority_hosts

            cap = NodeSyncCapability(
                node_id=host_name,
                can_receive_games=can_receive_games,
                can_receive_models=can_receive_models,
                can_receive_npz=can_receive_npz,
                is_training_node=is_training,
                is_priority_node=is_priority,
                is_ephemeral=is_ephemeral,
                shares_nfs=shares_nfs,
                provider=provider,
            )

            self._node_capabilities[host_name] = cap

    def get_external_storage_path(self, host: str, data_type: str) -> str | None:
        """Get the external storage path for a host and data type.

        December 2025: Supports coordinator backup via external drives.

        Args:
            host: Hostname to check
            data_type: Type of data ("games", "models", "npz")

        Returns:
            Storage path if configured, None otherwise
        """
        for storage in getattr(self, "_external_storage", []):
            if storage.get("host") == host:
                base_path = storage.get("path", "")
                subdirs = storage.get("subdirs", {})
                subdir = subdirs.get(data_type, data_type)
                return f"{base_path}/{subdir}" if base_path else None
        return None

    def get_sync_targets(
        self,
        data_type: str | DataType,
        board_type: str | None = None,
        num_players: int | None = None,
        exclude_nodes: list[str] | None = None,
        max_targets: int = 10,
    ) -> list[SyncTarget]:
        """Get candidate nodes for syncing data.

        Args:
            data_type: Type of data ("game", "model", "npz")
            board_type: Optional board type filter
            num_players: Optional num_players filter
            exclude_nodes: Nodes to exclude
            max_targets: Maximum number of targets to return

        Returns:
            List of SyncTarget sorted by priority
        """
        if isinstance(data_type, str):
            data_type = DataType(data_type)

        exclude = set(exclude_nodes or [])
        exclude.add(self.node_id)  # Don't sync to self

        targets: list[SyncTarget] = []

        for node_id, cap in self._node_capabilities.items():
            if node_id in exclude:
                continue

            # Check if node can receive this data type
            if not self._can_receive_data_type(cap, data_type):
                continue

            # Check disk capacity
            if not self._check_node_capacity(node_id):
                continue

            # Skip if NFS sharing applies (data already visible)
            if self._shares_storage_with(node_id):
                continue

            # Compute priority
            priority = self._compute_target_priority(cap, data_type)
            reason = self._get_target_reason(cap)

            targets.append(SyncTarget(
                node_id=node_id,
                priority=priority,
                reason=reason,
            ))

        # Sort by priority (highest first)
        targets.sort(key=lambda t: t.priority, reverse=True)

        return targets[:max_targets]

    def _can_receive_data_type(
        self,
        cap: NodeSyncCapability,
        data_type: DataType,
    ) -> bool:
        """Check if a node can receive a specific data type."""
        if data_type == DataType.GAME:
            return cap.can_receive_games
        elif data_type == DataType.MODEL:
            return cap.can_receive_models
        elif data_type == DataType.NPZ:
            return cap.can_receive_npz
        return False

    def _check_node_capacity(self, node_id: str) -> bool:
        """Check if a node has capacity for more data.

        P2.3 Dec 2025: Added capacity refresh check to avoid stale data.
        """
        # Refresh capacity if stale
        self._maybe_refresh_capacity()
        return self._manifest.can_receive_data(node_id, DataType.GAME)

    def _maybe_refresh_capacity(self) -> None:
        """Refresh capacity data if stale.

        P2.3 Dec 2025: Prevents routing decisions based on hours-old capacity data.
        """
        import time

        now = time.time()
        if now - self._last_capacity_refresh < self.CAPACITY_REFRESH_INTERVAL:
            return

        self._last_capacity_refresh = now

        # Refresh local node capacity
        try:
            self._manifest.update_local_capacity()
            logger.debug("[SyncRouter] Refreshed local capacity")
        except Exception as e:
            logger.debug(f"[SyncRouter] Failed to refresh local capacity: {e}")

        # Emit capacity refresh event for cluster-wide updates
        try:
            from app.coordination.event_router import DataEventType, get_event_bus

            bus = get_event_bus()
            if bus:
                bus.emit(DataEventType.NODE_CAPACITY_UPDATED, {
                    "node_id": self.node_id,
                    "reason": "capacity_refresh",
                })
        except Exception as e:
            logger.debug(f"[SyncRouter] Failed to emit capacity event: {e}")

    def refresh_all_capacity(self) -> None:
        """Force refresh of all capacity data.

        P2.3 Dec 2025: Call this when capacity data is suspected to be stale.
        """
        import time

        self._last_capacity_refresh = time.time()

        # Update local capacity
        try:
            self._manifest.update_local_capacity()
            logger.info("[SyncRouter] Force refreshed local capacity")
        except Exception as e:
            logger.warning(f"[SyncRouter] Failed to refresh capacity: {e}")

    def _shares_storage_with(self, target_node: str) -> bool:
        """Check if current node shares storage with target node.

        If both nodes are Lambda nodes with NFS, they share storage
        and don't need to sync between each other.
        """
        my_cap = self._node_capabilities.get(self.node_id)
        target_cap = self._node_capabilities.get(target_node)

        if not my_cap or not target_cap:
            return False

        # Both Lambda nodes share NFS
        if my_cap.shares_nfs and target_cap.shares_nfs:
            return True

        return False

    def _compute_target_priority(
        self,
        cap: NodeSyncCapability,
        data_type: DataType,
    ) -> int:
        """Compute sync priority for a target node.

        Higher priority = sync first.
        """
        priority = 50  # Base priority

        # Training nodes get highest priority for game/NPZ data
        if cap.is_training_node and data_type in (DataType.GAME, DataType.NPZ):
            priority += 30

        # Priority nodes get bonus
        if cap.is_priority_node:
            priority += 20

        # Ephemeral nodes get lower priority for receiving data
        # (we should sync FROM them, not TO them)
        if cap.is_ephemeral:
            priority -= 15

        # Prefer nodes with more available space
        if cap.disk_usage_percent < 50:
            priority += 10
        elif cap.disk_usage_percent > 70:
            priority -= 20

        return priority

    def _get_target_reason(self, cap: NodeSyncCapability) -> str:
        """Get human-readable reason for target selection."""
        reasons = []

        if cap.is_training_node:
            reasons.append("training")
        if cap.is_priority_node:
            reasons.append("priority")
        if cap.is_ephemeral:
            reasons.append("ephemeral")

        return ", ".join(reasons) if reasons else "available"

    def should_sync_to_node(
        self,
        target_node: str,
        data_type: str | DataType,
        source_node: str | None = None,
    ) -> bool:
        """Check if data should be synced to a specific node.

        Args:
            target_node: Target node ID
            data_type: Type of data
            source_node: Source node (defaults to current node)

        Returns:
            True if sync should proceed
        """
        if isinstance(data_type, str):
            data_type = DataType(data_type)

        source = source_node or self.node_id

        # Can't sync to self
        if target_node == source:
            return False

        # Check node capabilities
        cap = self._node_capabilities.get(target_node)
        if not cap:
            # Unknown node - use manifest policy
            return self._manifest.can_receive_data(target_node, data_type)

        # Check data type permissions
        if not self._can_receive_data_type(cap, data_type):
            return False

        # Check capacity
        if not self._check_node_capacity(target_node):
            return False

        # Check NFS sharing
        if self._shares_storage_with(target_node):
            return False

        return True

    def get_optimal_source(
        self,
        game_id: str,
        target_node: str,
    ) -> str | None:
        """Find the optimal source node for syncing a game.

        Considers:
        - Network proximity (same provider)
        - Load balancing
        - Bandwidth availability

        Args:
            game_id: Game to sync
            target_node: Destination node

        Returns:
            Best source node ID or None if not found
        """
        # Find locations for this game
        locations = self._manifest.find_game(game_id)
        if not locations:
            return None

        target_cap = self._node_capabilities.get(target_node)
        target_provider = target_cap.provider if target_cap else "unknown"

        candidates: list[tuple[str, int]] = []

        for loc in locations:
            source = loc.node_id
            if source == target_node:
                continue

            source_cap = self._node_capabilities.get(source)
            score = 50  # Base score

            if source_cap:
                # Prefer same provider (lower latency)
                if source_cap.provider == target_provider:
                    score += 20

                # Avoid ephemeral nodes as sources if possible
                if source_cap.is_ephemeral:
                    score -= 10

                # Prefer nodes with low disk usage (less IO contention)
                if source_cap.disk_usage_percent < 50:
                    score += 10

            candidates.append((source, score))

        if not candidates:
            return None

        # Return highest scoring source
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def plan_replication(
        self,
        game_id: str,
        min_copies: int = 2,
        quality_score: float | None = None,
    ) -> list[SyncRoute]:
        """Plan replication routes for a game.

        December 2025: Now supports quality-based priority boost.
        High-quality games get synced first for faster training data availability.

        Args:
            game_id: Game to replicate
            min_copies: Minimum number of copies desired
            quality_score: Optional quality score (0-1). If not provided,
                          will attempt to fetch from manifest.

        Returns:
            List of SyncRoute describing the sync plan
        """
        routes: list[SyncRoute] = []

        # Get current locations
        locations = self._manifest.find_game(game_id)
        current_nodes = {loc.node_id for loc in locations}

        copies_needed = min_copies - len(current_nodes)
        if copies_needed <= 0:
            return routes

        # Get quality score if not provided (Dec 2025: Quality-based priority)
        if quality_score is None:
            quality_score = self._get_game_quality_score(game_id)

        # Compute quality priority boost (0-30 points based on quality)
        # High-quality games get significant priority boost for faster sync
        quality_priority_boost = int(quality_score * 30) if quality_score else 0

        # Get targets
        targets = self._manifest.get_replication_targets(
            game_id,
            min_copies=min_copies,
            exclude_nodes=list(current_nodes),
        )

        for target in targets:
            # Find best source
            source = self.get_optimal_source(game_id, target.node_id)
            if not source:
                source = list(current_nodes)[0] if current_nodes else self.node_id

            # Apply quality boost to priority
            adjusted_priority = target.priority + quality_priority_boost

            routes.append(SyncRoute(
                source_node=source,
                target_node=target.node_id,
                data_type=DataType.GAME,
                priority=adjusted_priority,
                reason=target.reason + (f" (quality={quality_score:.2f})" if quality_score else ""),
                quality_score=quality_score or 0.0,
            ))

        # Sort by quality-adjusted priority (highest first)
        routes.sort(key=lambda r: r.priority, reverse=True)

        return routes

    def _get_game_quality_score(self, game_id: str) -> float:
        """Get quality score for a game from manifest or compute it.

        December 2025: Integrates with unified_quality for quality-based sync.
        """
        # Try manifest first (metadata might have quality)
        try:
            metadata = self._manifest.get_game_metadata(game_id)
            if metadata and hasattr(metadata, 'quality_score'):
                return metadata.quality_score or 0.0
        except Exception as e:
            logger.debug(f"Failed to get quality score from manifest for {game_id}: {e}")

        # Try unified quality scorer
        try:
            from app.quality.unified_quality import get_quality_scorer

            # Get game details from manifest
            locations = self._manifest.find_game(game_id)
            if locations:
                loc = locations[0]
                if hasattr(loc, 'game_length') and hasattr(loc, 'winner'):
                    scorer = get_quality_scorer()
                    quality = scorer.compute_game_quality(
                        game_id=game_id,
                        game_length=getattr(loc, 'game_length', 50),
                        winner=getattr(loc, 'winner', None),
                        avg_player_elo=getattr(loc, 'avg_elo', 1200.0),
                    )
                    return quality.overall_score
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[SyncRouter] Could not get quality for {game_id}: {e}")

        return 0.5  # Default neutral quality

    def get_node_capability(self, node_id: str) -> NodeSyncCapability | None:
        """Get sync capability information for a node."""
        return self._node_capabilities.get(node_id)

    def get_status(self) -> dict[str, Any]:
        """Get router status."""
        return {
            "node_id": self.node_id,
            "total_nodes": len(self._node_capabilities),
            "training_nodes": sum(
                1 for c in self._node_capabilities.values() if c.is_training_node
            ),
            "priority_nodes": sum(
                1 for c in self._node_capabilities.values() if c.is_priority_node
            ),
            "ephemeral_nodes": sum(
                1 for c in self._node_capabilities.values() if c.is_ephemeral
            ),
            "nfs_nodes": sum(
                1 for c in self._node_capabilities.values() if c.shares_nfs
            ),
        }

    def health_check(self) -> dict[str, Any]:
        """Check health status of SyncRouter.

        December 27, 2025: Added to meet P2P manager health_check() standard.

        Returns:
            Dict with status, node counts, and configuration health.
        """
        status = "healthy"
        errors_count = 0
        last_error: str | None = None

        # Check node capabilities are loaded
        total_nodes = len(self._node_capabilities)
        if total_nodes == 0:
            status = "degraded"
            last_error = "No node capabilities loaded"

        # Check manifest availability
        manifest_healthy = False
        try:
            manifest_healthy = self._manifest is not None and hasattr(
                self._manifest, "find_game"
            )
        except Exception as e:
            status = "unhealthy"
            last_error = f"Manifest error: {e}"
            errors_count += 1

        if not manifest_healthy and status == "healthy":
            status = "degraded"
            last_error = "Cluster manifest not available"

        # Count enabled vs disabled nodes
        enabled_nodes = sum(
            1
            for c in self._node_capabilities.values()
            if c.can_receive_games or c.can_receive_models or c.can_receive_npz
        )

        # If all nodes are disabled, that's degraded
        if enabled_nodes == 0 and total_nodes > 0:
            status = "degraded"
            last_error = "All nodes have sync disabled"

        return {
            "status": status,
            "operations_count": total_nodes,  # Number of configured nodes
            "errors_count": errors_count,
            "last_error": last_error,
            "total_nodes": total_nodes,
            "enabled_nodes": enabled_nodes,
            "manifest_available": manifest_healthy,
            "training_nodes": sum(
                1 for c in self._node_capabilities.values() if c.is_training_node
            ),
            "priority_nodes": sum(
                1 for c in self._node_capabilities.values() if c.is_priority_node
            ),
        }

    # =========================================================================
    # Event Integration (December 2025)
    # =========================================================================

    def wire_to_event_router(self) -> None:
        """Wire this router to the event system.

        Subscribes to:
        - NEW_GAMES_AVAILABLE: Route new games to appropriate nodes
        - TRAINING_STARTED: Prioritize training nodes
        - HOST_ONLINE/OFFLINE: Update node capabilities
        - NODE_RECOVERED: Re-enable sync to recovered nodes (Dec 2025)
        - CLUSTER_CAPACITY_CHANGED: React to cluster membership changes
        """
        try:
            from app.coordination.event_router import DataEventType, get_router

            router = get_router()

            # Subscribe to game events
            router.subscribe(
                DataEventType.NEW_GAMES_AVAILABLE.value,
                self._on_new_games_available,
            )

            # Subscribe to training events
            router.subscribe(
                DataEventType.TRAINING_STARTED.value,
                self._on_training_started,
            )

            # Subscribe to host events
            router.subscribe(
                DataEventType.HOST_ONLINE.value,
                self._on_host_online,
            )
            router.subscribe(
                DataEventType.HOST_OFFLINE.value,
                self._on_host_offline,
            )

            # Dec 2025: Subscribe to NODE_RECOVERED to re-enable sync
            router.subscribe(
                DataEventType.NODE_RECOVERED.value,
                self._on_node_recovered,
            )

            # Subscribe to cluster capacity changes (Dec 2025 - P2P integration)
            router.subscribe(
                DataEventType.CLUSTER_CAPACITY_CHANGED.value,
                self._on_cluster_capacity_changed,
            )

            # Dec 2025: Subscribe to MODEL_SYNC_REQUESTED to trigger model re-download
            router.subscribe(
                DataEventType.MODEL_SYNC_REQUESTED.value,
                self._on_model_sync_requested,
            )

            logger.info(
                "[SyncRouter] Wired to event router "
                "(NEW_GAMES_AVAILABLE, TRAINING_STARTED, HOST_ONLINE/OFFLINE, "
                "NODE_RECOVERED, CLUSTER_CAPACITY_CHANGED, MODEL_SYNC_REQUESTED)"
            )

        except ImportError as e:
            logger.warning(f"[SyncRouter] Event router not available: {e}")
        except Exception as e:
            logger.error(f"[SyncRouter] Failed to wire to event router: {e}")

    async def _on_new_games_available(self, event: Any) -> None:
        """Handle NEW_GAMES_AVAILABLE event - route games to targets."""
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            game_count = payload.get("count", 0)
            source_node = payload.get("source", self.node_id)

            if game_count > 0:
                # Get sync targets for the new games
                targets = self.get_sync_targets(
                    data_type=DataType.GAME,
                    exclude_nodes=[source_node],
                    max_targets=5,
                )

                if targets:
                    logger.info(
                        f"[SyncRouter] New games ({game_count}) from {source_node}, "
                        f"routing to {len(targets)} targets"
                    )

                    # Emit sync routing decision
                    await self._emit_sync_routing_decision(
                        source=source_node,
                        targets=[t.node_id for t in targets],
                        data_type=DataType.GAME,
                        reason=f"new_games:{game_count}",
                    )

        except Exception as e:
            logger.error(f"[SyncRouter] Error handling new games event: {e}")

    async def _on_training_started(self, event: Any) -> None:
        """Handle TRAINING_STARTED event - mark node as training priority."""
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            node_id = payload.get("node_id") or payload.get("host")

            if node_id and node_id in self._node_capabilities:
                cap = self._node_capabilities[node_id]
                cap.is_training_node = True
                cap.is_priority_node = True
                logger.info(f"[SyncRouter] Marked {node_id} as training priority")

        except Exception as e:
            logger.error(f"[SyncRouter] Error handling training started: {e}")

    async def _on_host_online(self, event: Any) -> None:
        """Handle HOST_ONLINE event - add/update node capabilities."""
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            node_id = payload.get("host")

            if node_id and node_id not in self._node_capabilities:
                # Create default capability for new node
                cap = NodeSyncCapability(node_id=node_id)
                self._node_capabilities[node_id] = cap
                logger.info(f"[SyncRouter] Added new node: {node_id}")

        except Exception as e:
            logger.debug(f"[SyncRouter] Error handling host online: {e}")

    async def _on_host_offline(self, event: Any) -> None:
        """Handle HOST_OFFLINE event - mark node as unavailable."""
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            node_id = payload.get("host")

            if node_id and node_id in self._node_capabilities:
                # Mark node as unavailable rather than removing
                cap = self._node_capabilities[node_id]
                cap.can_receive_games = False
                cap.can_receive_models = False
                cap.can_receive_npz = False
                logger.info(f"[SyncRouter] Marked {node_id} as offline")

        except Exception as e:
            logger.debug(f"[SyncRouter] Error handling host offline: {e}")

    async def _on_node_recovered(self, event: Any) -> None:
        """Handle NODE_RECOVERED event - re-enable sync to recovered node.

        December 2025: Added to complete health event integration.
        When a node recovers after being offline, restore its sync capabilities.
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            node_id = payload.get("node_id") or payload.get("host")

            if node_id and node_id in self._node_capabilities:
                cap = self._node_capabilities[node_id]
                # Re-enable based on original policy from manifest
                policy = self._manifest.get_sync_policy(node_id)
                cap.can_receive_games = policy.receive_games
                cap.can_receive_models = policy.receive_models
                cap.can_receive_npz = policy.receive_npz
                logger.info(
                    f"[SyncRouter] Restored sync capabilities for recovered node: {node_id}"
                )
            elif node_id:
                # New node recovered - add with default capabilities
                cap = NodeSyncCapability(node_id=node_id)
                self._node_capabilities[node_id] = cap
                logger.info(f"[SyncRouter] Added recovered node: {node_id}")

        except Exception as e:
            logger.debug(f"[SyncRouter] Error handling node recovered: {e}")

    async def _on_cluster_capacity_changed(self, event: Any) -> None:
        """Handle CLUSTER_CAPACITY_CHANGED event - refresh capacity data and recalculate routes.

        December 2025: Enables real-time reaction to cluster membership changes.
        When nodes join or leave, we refresh capacity data to ensure sync
        targets are current and appropriately prioritized.

        Args:
            event: Event with payload containing change_type, node_id, total_nodes, gpu_nodes
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            change_type = payload.get("change_type", "unknown")
            node_id = payload.get("node_id", "unknown")
            total_nodes = payload.get("total_nodes", 0)
            gpu_nodes = payload.get("gpu_nodes", 0)
            reason = payload.get("reason", "")

            logger.info(
                f"[SyncRouter] Cluster capacity changed: {change_type} "
                f"node={node_id}, total={total_nodes}, gpu={gpu_nodes}, reason={reason}"
            )

            # Refresh capacity data from manifest
            if self._manifest:
                self._manifest.refresh_capacity_data()

            # Update node capability based on change
            if change_type == "node_removed":
                if node_id in self._node_capabilities:
                    cap = self._node_capabilities[node_id]
                    cap.can_receive_games = False
                    cap.can_receive_models = False
                    cap.can_receive_npz = False
                    logger.debug(f"[SyncRouter] Disabled sync to removed node: {node_id}")

            elif change_type == "node_added":
                if node_id not in self._node_capabilities:
                    # Add new node with default capabilities
                    self._node_capabilities[node_id] = NodeSyncCapability(node_id=node_id)
                else:
                    # Re-enable existing node
                    cap = self._node_capabilities[node_id]
                    cap.can_receive_games = True
                    cap.can_receive_models = True
                    cap.can_receive_npz = True
                logger.debug(f"[SyncRouter] Enabled sync to added node: {node_id}")

            # Emit capacity refresh event for downstream consumers
            await self._emit_capacity_refresh(
                change_type=change_type,
                node_id=node_id,
                total_nodes=total_nodes,
                gpu_nodes=gpu_nodes,
            )

        except Exception as e:
            logger.warning(f"[SyncRouter] Error handling cluster capacity changed: {e}")

    async def _on_model_sync_requested(self, event: Any) -> None:
        """Handle MODEL_SYNC_REQUESTED event - trigger model re-download from healthy nodes.

        December 2025: Wired to address critical gap where model sync requests were
        emitted but had no subscribers. This handler routes models from healthy nodes
        to requesting nodes.

        Args:
            event: Event with payload containing model_id, requesting_node, reason
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            model_id = payload.get("model_id", "")
            requesting_node = payload.get("node_id", "") or payload.get("requesting_node", "")
            reason = payload.get("reason", "sync_requested")

            if not model_id or not requesting_node:
                logger.debug(
                    f"[SyncRouter] MODEL_SYNC_REQUESTED missing model_id or node: {payload}"
                )
                return

            logger.info(
                f"[SyncRouter] Model sync requested: {model_id} for {requesting_node}, "
                f"reason: {reason}"
            )

            # Find nodes that have this model and can serve as sources
            sources = self.get_sync_targets(
                data_type=DataType.MODEL,
                exclude_nodes=[requesting_node],
                max_targets=3,
            )

            if not sources:
                logger.warning(
                    f"[SyncRouter] No sources found for model {model_id}"
                )
                return

            # Select the best source (first one - highest priority)
            source_node = sources[0].node_id

            logger.info(
                f"[SyncRouter] Routing model {model_id} from {source_node} "
                f"to {requesting_node}"
            )

            # Emit sync routing decision
            await self._emit_sync_routing_decision(
                source=source_node,
                targets=[requesting_node],
                data_type=DataType.MODEL,
                reason=f"model_sync:{model_id}:{reason}",
            )

            # Also emit a MODEL_SYNC_STARTED event for tracking
            try:
                from app.coordination.event_router import get_router

                router = get_router()
                await router.publish(
                    "MODEL_SYNC_STARTED",
                    {
                        "model_id": model_id,
                        "source_node": source_node,
                        "target_node": requesting_node,
                        "reason": reason,
                    },
                )
            except Exception as e:
                logger.debug(f"[SyncRouter] Could not emit MODEL_SYNC_STARTED: {e}")

        except Exception as e:
            logger.error(f"[SyncRouter] Error handling model sync request: {e}")

    async def _emit_capacity_refresh(
        self,
        change_type: str,
        node_id: str,
        total_nodes: int,
        gpu_nodes: int,
    ) -> None:
        """Emit a capacity refresh event for downstream consumers."""
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            router.publish(
                "SYNC_CAPACITY_REFRESHED",
                {
                    "change_type": change_type,
                    "node_id": node_id,
                    "total_nodes": total_nodes,
                    "gpu_nodes": gpu_nodes,
                    "router": "SyncRouter",
                },
            )
        except Exception as e:
            logger.debug(f"[SyncRouter] Could not emit capacity refresh: {e}")

    async def _emit_sync_routing_decision(
        self,
        source: str,
        targets: list[str],
        data_type: DataType,
        reason: str,
    ) -> None:
        """Emit a sync routing decision event."""
        try:
            from app.coordination.event_router import (
                DataEvent,
                DataEventType,
                get_event_bus,
            )

            # Dec 2025: Explicit null check before publish
            bus = get_event_bus()
            if bus is not None:
                await bus.publish(DataEvent(
                    event_type=DataEventType.SYNC_REQUEST.value,
                    payload={
                        "source": source,
                        "targets": targets,
                        "data_type": data_type.value,
                        "reason": reason,
                        "router": "SyncRouter",
                    },
                    source="SyncRouter",
                ))

        except Exception as e:
            logger.warning(f"[SyncRouter] Could not emit routing decision: {e}")
            # Still log the decision for debugging even if event emission failed
            logger.info(
                f"[SyncRouter] Routing decision: {len(targets)} targets for "
                f"{data_type.value} data from {source}"
            )


# Module-level singleton
_sync_router: SyncRouter | None = None


def get_sync_router() -> SyncRouter:
    """Get the singleton SyncRouter instance."""
    global _sync_router
    if _sync_router is None:
        _sync_router = SyncRouter()
    return _sync_router


def reset_sync_router() -> None:
    """Reset the singleton (for testing)."""
    global _sync_router
    _sync_router = None
