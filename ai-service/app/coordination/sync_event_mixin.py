"""Event subscription and handler mixin for AutoSyncDaemon.

December 2025: Extracted from auto_sync_daemon.py as part of mixin-based refactoring.
December 2025: Updated to inherit from SyncMixinBase for common functionality.

This mixin provides:
- Event subscription infrastructure (_subscribe_to_events)
- All _on_* event handlers for sync triggers
- Push-to-neighbors for Layer 1 replication
- Urgent sync triggering
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from app.core.async_context import fire_and_forget
from app.coordination.sync_mixin_base import SyncMixinBase
from app.coordination.event_handler_utils import extract_config_key
from app.coordination.event_utils import make_config_key

if TYPE_CHECKING:
    from app.coordination.sync_strategies import AutoSyncConfig, SyncStats
    from app.distributed.cluster_manifest import ClusterManifest

logger = logging.getLogger(__name__)

# Import resilient handler wrapper if available
try:
    from app.coordination.handler_resilience import resilient_handler
    HAS_RESILIENT_HANDLER = True
except ImportError:
    HAS_RESILIENT_HANDLER = False
    resilient_handler = None


class SyncEventMixin(SyncMixinBase):
    """Mixin providing event subscription and handler methods for AutoSyncDaemon.

    Inherits from SyncMixinBase for common error handling and logging utilities.

    Additional expected attributes from main class:
    - _subscribed: bool
    - _urgent_sync_pending: dict[str, float]
    - _cluster_manifest: ClusterManifest | None

    Expected methods from main class:
    - _sync_all() -> None
    - _sync_to_peer(node_id: str) -> bool
    """

    # Additional type hints specific to this mixin
    _subscribed: bool
    _urgent_sync_pending: dict[str, float]
    _cluster_manifest: ClusterManifest | None

    def _wrap_handler(self, handler):
        """Wrap handler with resilient_handler for fault tolerance (December 2025).

        Args:
            handler: The async event handler to wrap

        Returns:
            Wrapped handler with exception boundary and metrics, or original if unavailable
        """
        if HAS_RESILIENT_HANDLER and resilient_handler:
            return resilient_handler(handler, coordinator="AutoSyncDaemon")
        return handler

    def _subscribe_to_events(self) -> None:
        """Subscribe to events that trigger sync (Phase 9).

        December 2025: Handlers wrapped with resilient_handler for fault tolerance.

        Subscribes to:
        - DATA_STALE: Training data is stale, trigger urgent sync
        - SYNC_TRIGGERED: External sync request
        - SYNC_REQUEST: SyncRouter-triggered sync (December 28, 2025)
        """
        if self._subscribed:
            return
        try:
            from app.coordination.event_router import DataEventType, get_event_bus

            bus = get_event_bus()

            # Subscribe to DATA_STALE to trigger urgent sync
            if hasattr(DataEventType, 'DATA_STALE'):
                bus.subscribe(DataEventType.DATA_STALE, self._wrap_handler(self._on_data_stale))
                logger.info("[AutoSyncDaemon] Subscribed to DATA_STALE")

            # Subscribe to SYNC_TRIGGERED for external requests
            if hasattr(DataEventType, 'SYNC_TRIGGERED'):
                bus.subscribe(DataEventType.SYNC_TRIGGERED, self._wrap_handler(self._on_sync_triggered))
                logger.info("[AutoSyncDaemon] Subscribed to SYNC_TRIGGERED")

            # Subscribe to NEW_GAMES_AVAILABLE for push-on-generate (Dec 2025)
            # Layer 1: Immediate push to neighbors when games are generated
            if hasattr(DataEventType, 'NEW_GAMES_AVAILABLE'):
                bus.subscribe(DataEventType.NEW_GAMES_AVAILABLE, self._wrap_handler(self._on_new_games_available))
                logger.info("[AutoSyncDaemon] Subscribed to NEW_GAMES_AVAILABLE (push-on-generate)")

            # Subscribe to DATA_SYNC_STARTED for sync coordination (Dec 2025)
            if hasattr(DataEventType, 'DATA_SYNC_STARTED'):
                bus.subscribe(DataEventType.DATA_SYNC_STARTED, self._wrap_handler(self._on_data_sync_started))
                logger.info("[AutoSyncDaemon] Subscribed to DATA_SYNC_STARTED")

            # Subscribe to MODEL_DISTRIBUTION_COMPLETE for model sync tracking (Dec 2025)
            if hasattr(DataEventType, 'MODEL_DISTRIBUTION_COMPLETE'):
                bus.subscribe(DataEventType.MODEL_DISTRIBUTION_COMPLETE, self._wrap_handler(self._on_model_distribution_complete))
                logger.info("[AutoSyncDaemon] Subscribed to MODEL_DISTRIBUTION_COMPLETE")

            # Subscribe to SELFPLAY_COMPLETE for immediate sync on selfplay completion (Dec 2025)
            # Phase F: Trigger sync immediately when selfplay batch finishes
            if hasattr(DataEventType, 'SELFPLAY_COMPLETE'):
                bus.subscribe(DataEventType.SELFPLAY_COMPLETE, self._wrap_handler(self._on_selfplay_complete))
                logger.info("[AutoSyncDaemon] Subscribed to SELFPLAY_COMPLETE (immediate sync)")

            # Subscribe to TRAINING_STARTED for priority sync to training nodes (Dec 2025)
            if hasattr(DataEventType, 'TRAINING_STARTED'):
                bus.subscribe(DataEventType.TRAINING_STARTED, self._wrap_handler(self._on_training_started))
                logger.info("[AutoSyncDaemon] Subscribed to TRAINING_STARTED (priority sync)")

            # Subscribe to NODE_RECOVERED to clear exclusion state for recovered nodes (Dec 2025)
            if hasattr(DataEventType, 'NODE_RECOVERED'):
                bus.subscribe(DataEventType.NODE_RECOVERED, self._wrap_handler(self._on_node_recovered))
                logger.info("[AutoSyncDaemon] Subscribed to NODE_RECOVERED (exclusion reset)")

            # December 28, 2025: Subscribe to backpressure events to pause sync during high load
            if hasattr(DataEventType, 'BACKPRESSURE_ACTIVATED'):
                bus.subscribe(DataEventType.BACKPRESSURE_ACTIVATED, self._wrap_handler(self._on_backpressure_activated))
                logger.info("[AutoSyncDaemon] Subscribed to BACKPRESSURE_ACTIVATED (pause sync)")

            if hasattr(DataEventType, 'BACKPRESSURE_RELEASED'):
                bus.subscribe(DataEventType.BACKPRESSURE_RELEASED, self._wrap_handler(self._on_backpressure_released))
                logger.info("[AutoSyncDaemon] Subscribed to BACKPRESSURE_RELEASED (resume sync)")

            # December 28, 2025: Subscribe to SYNC_REQUEST for SyncRouter-triggered sync operations
            # This wires the previously orphaned SYNC_REQUEST event emitted by SyncRouter._emit_sync_routing_decision
            if hasattr(DataEventType, 'SYNC_REQUEST'):
                bus.subscribe(DataEventType.SYNC_REQUEST, self._wrap_handler(self._on_sync_request))
                logger.info("[AutoSyncDaemon] Subscribed to SYNC_REQUEST (SyncRouter integration)")

            # December 30, 2025: Subscribe to CONFIG_UPDATED for distributed config sync
            # When cluster config changes, reload cached config and update node list
            if hasattr(DataEventType, 'CONFIG_UPDATED'):
                bus.subscribe(DataEventType.CONFIG_UPDATED, self._wrap_handler(self._on_config_updated))
                logger.info("[AutoSyncDaemon] Subscribed to CONFIG_UPDATED (cluster config sync)")

            self._subscribed = True
            if HAS_RESILIENT_HANDLER:
                logger.info("[AutoSyncDaemon] Event handlers wrapped with resilient_handler")
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning(f"[AutoSyncDaemon] Failed to subscribe to events: {e}")

    async def _on_data_stale(self, event) -> None:
        """Handle DATA_STALE event by triggering urgent sync (Phase 9).

        When training data is detected as stale, we trigger an immediate
        sync operation to fetch fresh data from the cluster.

        December 29, 2025: Now uses trigger_sync() for event-driven wake-up.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            board_type = payload.get("board_type", "")
            num_players = payload.get("num_players", 0)
            data_age_hours = payload.get("data_age_hours", 0.0)

            config_key = make_config_key(board_type, num_players) if board_type and num_players else "unknown"

            logger.warning(
                f"[AutoSyncDaemon] DATA_STALE received for {config_key}: "
                f"age={data_age_hours:.1f}h - triggering urgent sync"
            )

            # Track the urgent sync request
            self._urgent_sync_pending[config_key] = time.time()
            self._events_processed += 1

            # December 29, 2025: Wake the sync loop immediately via event
            if hasattr(self, 'trigger_sync'):
                self.trigger_sync()
            else:
                # Fallback for backward compatibility
                fire_and_forget(self._trigger_urgent_sync(config_key))

        except (RuntimeError, OSError, ConnectionError) as e:
            self._errors_count += 1
            self._last_error = str(e)
            logger.error(f"[AutoSyncDaemon] Error handling DATA_STALE: {e}")

    async def _on_sync_triggered(self, event) -> None:
        """Handle external SYNC_TRIGGERED event (Phase 9).

        December 29, 2025: Now uses trigger_sync() for event-driven wake-up.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            reason = payload.get("reason", "unknown")
            config_key = extract_config_key(payload)

            logger.info(
                f"[AutoSyncDaemon] SYNC_TRIGGERED received: "
                f"reason={reason}, config={config_key}"
            )

            self._events_processed += 1

            # December 29, 2025: Wake the sync loop immediately via event
            if hasattr(self, 'trigger_sync'):
                self.trigger_sync()
            else:
                # Fallback for backward compatibility
                if config_key:
                    fire_and_forget(self._trigger_urgent_sync(config_key))
                else:
                    fire_and_forget(self._sync_all())

        except (RuntimeError, OSError, ConnectionError) as e:
            self._errors_count += 1
            self._last_error = str(e)
            logger.error(f"[AutoSyncDaemon] Error handling SYNC_TRIGGERED: {e}")

    async def _on_new_games_available(self, event) -> None:
        """Handle NEW_GAMES_AVAILABLE event - push-on-generate (Dec 2025).

        Layer 1 of the sync architecture: When new games are generated,
        immediately push to up to 3 neighbor nodes for rapid replication.
        This is especially important for Vast.ai ephemeral nodes.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            # Dec 27, 2025: Handle both "config_key" and "config" for compatibility
            config_key = extract_config_key(payload)
            new_games = payload.get("new_games", 0)
            total_games = payload.get("total_games", 0)

            # Only push if we have a meaningful batch (avoid spamming for 1-2 games)
            min_games = self.config.min_games_to_sync or 5
            if new_games < min_games:
                logger.debug(
                    f"[AutoSyncDaemon] Push-on-generate: skipping for {config_key} "
                    f"({new_games} < {min_games} min games)"
                )
                return

            logger.info(
                f"[AutoSyncDaemon] Push-on-generate: {config_key} "
                f"({new_games} new games, {total_games} total) - pushing to neighbors"
            )

            self._events_processed += 1

            # Trigger push to neighbors (Layer 1)
            fire_and_forget(self._push_to_neighbors(config_key, new_games))

        except (RuntimeError, OSError, ConnectionError) as e:
            self._errors_count += 1
            self._last_error = str(e)
            logger.error(f"[AutoSyncDaemon] Error handling NEW_GAMES_AVAILABLE: {e}")

    async def _on_data_sync_started(self, event) -> None:
        """Handle DATA_SYNC_STARTED - sync operation initiated.

        Tracks active sync operations to avoid concurrent syncs to the
        same target, which can cause conflicts and waste bandwidth.

        Added: December 2025
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            host = payload.get("host", "")
            sync_type = payload.get("sync_type", "incremental")

            logger.info(
                f"[AutoSyncDaemon] Sync started to {host} (type: {sync_type})"
            )

            # Track active sync to avoid concurrent operations
            if not hasattr(self, "_active_syncs"):
                self._active_syncs = {}
            self._active_syncs[host] = {
                "start_time": time.time(),
                "sync_type": sync_type,
            }

            self._events_processed += 1

        except (RuntimeError, AttributeError) as e:
            logger.warning(f"[AutoSyncDaemon] Error handling DATA_SYNC_STARTED: {e}")

    async def _on_model_distribution_complete(self, event) -> None:
        """Handle MODEL_DISTRIBUTION_COMPLETE - model synced to cluster.

        Logs model distribution completion and clears any pending model
        sync requests. This prevents redundant model distribution attempts.

        Added: December 2025
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            model_id = payload.get("model_id", "")
            board_type = payload.get("board_type", "")
            num_players = payload.get("num_players", 0)
            distributed_to = payload.get("distributed_to", [])

            config_key = make_config_key(board_type, num_players) if board_type and num_players else ""

            logger.info(
                f"[AutoSyncDaemon] Model distribution complete: {model_id} "
                f"({config_key}) -> {len(distributed_to)} nodes"
            )

            # Clear any pending model sync requests
            if hasattr(self, "_pending_model_syncs"):
                self._pending_model_syncs.pop(config_key, None)

            self._events_processed += 1

        except (RuntimeError, AttributeError) as e:
            logger.warning(f"[AutoSyncDaemon] Error handling MODEL_DISTRIBUTION_COMPLETE: {e}")

    async def _on_selfplay_complete(self, event) -> None:
        """Handle SELFPLAY_COMPLETE event - immediate sync on selfplay finish.

        Phase F (December 2025): When a selfplay batch completes, immediately
        trigger game data sync to propagate fresh training data across the cluster.
        This closes the loop from selfplay -> sync -> training for faster iteration.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            config_key = extract_config_key(payload)
            games_played = payload.get("games_played", 0)

            # Only sync if we have a meaningful batch
            min_games = self.config.min_games_to_sync or 5
            if games_played < min_games:
                logger.debug(
                    f"[AutoSyncDaemon] Selfplay sync skipped: {config_key} "
                    f"({games_played} < {min_games} min games)"
                )
                return

            logger.info(
                f"[AutoSyncDaemon] SELFPLAY_COMPLETE: {config_key} "
                f"({games_played} games) - triggering immediate cluster sync"
            )

            self._events_processed += 1

            # Track the urgent sync request
            self._urgent_sync_pending[config_key] = time.time()

            # December 29, 2025: Wake the sync loop immediately via event
            if hasattr(self, 'trigger_sync'):
                self.trigger_sync()
            else:
                # Fallback for backward compatibility
                fire_and_forget(self._trigger_urgent_sync(config_key))

            # Also push to neighbors for Layer 1 replication
            fire_and_forget(self._push_to_neighbors(config_key, games_played))

        except (RuntimeError, OSError, ConnectionError) as e:
            self._errors_count += 1
            self._last_error = str(e)
            logger.error(f"[AutoSyncDaemon] Error handling SELFPLAY_COMPLETE: {e}")

    async def _on_training_started(self, event) -> None:
        """Handle TRAINING_STARTED event - priority sync to training node.

        December 2025: When training starts on a node, we should immediately
        sync fresh data to that node to ensure it has the latest training samples.
        This reduces latency in the training feedback loop.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            config_key = extract_config_key(payload)
            node_id = payload.get("node_id", "")

            if not node_id:
                logger.debug("[AutoSyncDaemon] TRAINING_STARTED: no node_id, skipping")
                return

            logger.info(
                f"[AutoSyncDaemon] TRAINING_STARTED: {config_key} on {node_id} "
                "- triggering priority sync to training node"
            )

            self._events_processed += 1

            # Trigger priority sync to the training node
            try:
                success = await self._sync_to_peer(node_id)
                if success:
                    logger.info(
                        f"[AutoSyncDaemon] Priority sync to training node {node_id} complete"
                    )
            except (RuntimeError, OSError, ConnectionError) as sync_err:
                logger.warning(
                    f"[AutoSyncDaemon] Priority sync to {node_id} failed: {sync_err}"
                )

        except (RuntimeError, OSError, ConnectionError) as e:
            self._errors_count += 1
            self._last_error = str(e)
            logger.error(f"[AutoSyncDaemon] Error handling TRAINING_STARTED: {e}")

    async def _on_node_recovered(self, event) -> None:
        """Handle NODE_RECOVERED event - clear exclusion state for recovered nodes.

        December 2025: When a node recovers from being offline or unhealthy,
        we should clear any temporary exclusion state so it can participate
        in sync operations again.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            node_id = payload.get("node_id", "")

            if not node_id:
                logger.debug("[AutoSyncDaemon] NODE_RECOVERED: no node_id, skipping")
                return

            logger.info(
                f"[AutoSyncDaemon] NODE_RECOVERED: {node_id} "
                "- clearing exclusion state"
            )

            self._events_processed += 1

            # Clear any temporary exclusion for this node
            if hasattr(self, "_excluded_nodes") and node_id in self._excluded_nodes:
                self._excluded_nodes.discard(node_id)
                logger.info(f"[AutoSyncDaemon] Cleared exclusion for {node_id}")

            # Reset failure counter if we track per-node failures
            if hasattr(self, "_node_failure_counts") and node_id in self._node_failure_counts:
                self._node_failure_counts[node_id] = 0
                logger.debug(f"[AutoSyncDaemon] Reset failure count for {node_id}")

        except (RuntimeError, OSError, ConnectionError) as e:
            self._errors_count += 1
            self._last_error = str(e)
            logger.error(f"[AutoSyncDaemon] Error handling NODE_RECOVERED: {e}")

    async def _push_to_neighbors(self, config_key: str, new_games: int) -> None:
        """Push data to up to 3 neighbor nodes (Layer 1: push-from-generator).

        Prefers storage nodes with large disk capacity.
        Skips coordinator nodes and nodes with low disk space.
        """
        try:
            # Get available neighbors
            neighbors = await self._get_push_neighbors(max_neighbors=3)
            if not neighbors:
                logger.debug(
                    f"[AutoSyncDaemon] Push-on-generate: no eligible neighbors for {config_key}"
                )
                return

            # Push to each neighbor
            pushed_count = 0
            for neighbor_id in neighbors[:3]:
                try:
                    success = await self._sync_to_peer(neighbor_id)
                    if success:
                        pushed_count += 1
                        logger.debug(
                            f"[AutoSyncDaemon] Pushed {config_key} to {neighbor_id}"
                        )
                except (RuntimeError, OSError, ConnectionError) as e:
                    logger.warning(
                        f"[AutoSyncDaemon] Failed to push to {neighbor_id}: {e}"
                    )

            if pushed_count > 0:
                logger.info(
                    f"[AutoSyncDaemon] Push-on-generate complete: {config_key} "
                    f"pushed to {pushed_count}/{len(neighbors)} neighbors"
                )

        except (RuntimeError, OSError, ConnectionError) as e:
            logger.error(f"[AutoSyncDaemon] Push-on-generate failed for {config_key}: {e}")

    async def _get_push_neighbors(self, max_neighbors: int = 3) -> list[str]:
        """Get list of neighbor nodes for push-on-generate.

        Returns nodes sorted by priority:
        1. Storage nodes (large disk)
        2. Non-ephemeral nodes
        3. Healthy nodes with low disk usage
        """
        try:
            neighbors = []

            # Get cluster manifest if available
            if self._cluster_manifest:
                # Get all nodes with their storage capacity
                all_nodes = self._cluster_manifest.get_all_nodes()

                for node_id, node_info in all_nodes.items():
                    # Skip self
                    if node_id == self.node_id:
                        continue

                    # Skip excluded nodes (coordinators, etc.)
                    if node_id in self.config.exclude_hosts:
                        continue

                    # Skip nodes with high disk usage
                    disk_usage = node_info.get("disk_usage_percent", 0)
                    if disk_usage > self.config.max_disk_usage_percent:
                        continue

                    # Compute priority score
                    priority = 0.0
                    # Prefer storage nodes
                    if node_info.get("is_storage_node", False):
                        priority += 10.0
                    # Prefer non-ephemeral
                    if not node_info.get("is_ephemeral", False):
                        priority += 5.0
                    # Prefer nodes with more free space
                    priority += (100 - disk_usage) / 20.0

                    neighbors.append((node_id, priority))

                # Sort by priority (descending) and return top N
                neighbors.sort(key=lambda x: x[1], reverse=True)
                return [n[0] for n in neighbors[:max_neighbors]]

            # Fallback: no manifest available, return empty list
            # (we cannot determine neighbors without cluster manifest)
            logger.debug("[AutoSyncDaemon] No cluster manifest available for push neighbors")
            return neighbors

        except (RuntimeError, AttributeError, KeyError) as e:
            logger.warning(f"[AutoSyncDaemon] Error getting push neighbors: {e}")
            return []

    async def _trigger_urgent_sync(self, config_key: str) -> None:
        """Trigger an urgent sync operation for a specific config (Phase 9)."""
        try:
            logger.info(f"[AutoSyncDaemon] Urgent sync starting for {config_key}")

            # Find nodes with fresh data for this config
            if self._cluster_manifest:
                # Use manifest to find data sources
                await self._sync_all()
            else:
                # Fallback to full sync
                await self._sync_all()

            # Clear the pending flag
            self._urgent_sync_pending.pop(config_key, None)

            logger.info(f"[AutoSyncDaemon] Urgent sync completed for {config_key}")

        except (RuntimeError, OSError, ConnectionError) as e:
            logger.error(f"[AutoSyncDaemon] Urgent sync failed for {config_key}: {e}")

    # Note: _sync_all() and _sync_to_peer() are expected from main class
    # _emit_sync_failure() and _emit_sync_stalled() are inherited from SyncMixinBase

    async def _on_backpressure_activated(self, event) -> None:
        """Handle BACKPRESSURE_ACTIVATED event - pause sync operations.

        December 28, 2025: When the cluster is under high backpressure
        (training jobs consuming resources, high sync queue), we pause
        sync operations to avoid competing with training for resources.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            overall_pressure = payload.get("overall_pressure", 0.0)
            reason = payload.get("reason", "cluster_overloaded")

            logger.info(
                f"[AutoSyncDaemon] BACKPRESSURE_ACTIVATED: pressure={overall_pressure:.2f}, "
                f"reason={reason} - pausing sync operations"
            )

            self._sync_paused = True
            self._backpressure_reason = reason
            self._events_processed += 1

        except (RuntimeError, OSError, ConnectionError) as e:
            self._errors_count += 1
            self._last_error = str(e)
            logger.error(f"[AutoSyncDaemon] Error handling BACKPRESSURE_ACTIVATED: {e}")

    async def _on_backpressure_released(self, event) -> None:
        """Handle BACKPRESSURE_RELEASED event - resume sync operations.

        December 28, 2025: When backpressure is released, we resume
        sync operations. This may also trigger an immediate sync cycle
        to catch up on any missed data.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            paused_duration = payload.get("paused_duration_seconds", 0.0)

            logger.info(
                f"[AutoSyncDaemon] BACKPRESSURE_RELEASED: paused for {paused_duration:.1f}s "
                "- resuming sync operations"
            )

            self._sync_paused = False
            self._backpressure_reason = ""
            self._events_processed += 1

            # December 29, 2025: Wake the sync loop immediately via event
            # The loop will do a catch-up sync on resume
            if hasattr(self, 'trigger_sync'):
                self.trigger_sync()
                if paused_duration > 60.0:
                    logger.info(
                        "[AutoSyncDaemon] Extended backpressure period - sync loop will catch up"
                    )
            else:
                # Fallback for backward compatibility
                if paused_duration > 60.0:
                    logger.info(
                        "[AutoSyncDaemon] Extended backpressure period - triggering catch-up sync"
                    )
                    fire_and_forget(
                        self._sync_all(),
                        on_error=lambda exc: logger.warning(f"Catch-up sync failed: {exc}"),
                    )

        except (RuntimeError, OSError, ConnectionError) as e:
            self._errors_count += 1
            self._last_error = str(e)
            logger.error(f"[AutoSyncDaemon] Error handling BACKPRESSURE_RELEASED: {e}")

    async def _on_sync_request(self, event) -> None:
        """Handle SYNC_REQUEST event - execute SyncRouter-triggered sync.

        December 28, 2025: Wires the previously orphaned SYNC_REQUEST event
        emitted by SyncRouter._emit_sync_routing_decision(). This enables
        the SyncRouter to trigger targeted sync operations.

        Payload:
            source: Node ID of data source
            targets: List of target node IDs for sync
            data_type: Type of data (game, model, npz)
            reason: Human-readable sync reason
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            source = payload.get("source", "")
            targets = payload.get("targets", [])
            data_type = payload.get("data_type", "game")
            reason = payload.get("reason", "sync_request")

            if not targets:
                logger.debug("[AutoSyncDaemon] SYNC_REQUEST: no targets, skipping")
                return

            logger.info(
                f"[AutoSyncDaemon] SYNC_REQUEST received: "
                f"source={source}, targets={len(targets)}, "
                f"data_type={data_type}, reason={reason}"
            )

            self._events_processed += 1

            # Sync to each target node
            success_count = 0
            for target in targets:
                if target == self.node_id:
                    # Skip self - we're the target, not the source
                    continue
                try:
                    success = await self._sync_to_peer(target)
                    if success:
                        success_count += 1
                except (RuntimeError, OSError, ConnectionError) as sync_err:
                    logger.warning(
                        f"[AutoSyncDaemon] SYNC_REQUEST sync to {target} failed: {sync_err}"
                    )

            if success_count > 0:
                logger.info(
                    f"[AutoSyncDaemon] SYNC_REQUEST completed: "
                    f"{success_count}/{len(targets)} targets synced"
                )

        except (RuntimeError, OSError, ConnectionError) as e:
            self._errors_count += 1
            self._last_error = str(e)
            logger.error(f"[AutoSyncDaemon] Error handling SYNC_REQUEST: {e}")

    async def _on_config_updated(self, event) -> None:
        """Handle CONFIG_UPDATED event - reload cluster config.

        December 30, 2025: Part of distributed config sync infrastructure.
        When a peer has a newer config version (detected via gossip), it
        triggers a config pull and emits this event. We should reload our
        cached config and update the node list for sync operations.

        Payload:
            source_node: Node that triggered the config update
            timestamp: When the config was updated
            hash: Config content hash (optional)
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            source_node = payload.get("source_node", "unknown")
            timestamp = payload.get("timestamp", 0)

            logger.info(
                f"[AutoSyncDaemon] CONFIG_UPDATED from {source_node} at {timestamp:.0f} "
                "- reloading cluster config"
            )

            self._events_processed += 1

            # Force reload cluster config to pick up changes
            try:
                from app.config.cluster_config import get_config_cache

                cache = get_config_cache()
                new_config = cache.get_config(force_reload=True)
                logger.debug(
                    f"[AutoSyncDaemon] Reloaded cluster config: "
                    f"{len(new_config.hosts if hasattr(new_config, 'hosts') else [])} hosts"
                )
            except ImportError:
                logger.warning("[AutoSyncDaemon] Could not import config cache for reload")
            except (OSError, ValueError) as reload_err:
                logger.warning(f"[AutoSyncDaemon] Config reload failed: {reload_err}")

            # Update cluster manifest if we have one
            if self._cluster_manifest:
                try:
                    await self._cluster_manifest.refresh()
                    logger.debug("[AutoSyncDaemon] Refreshed cluster manifest after config update")
                except (RuntimeError, OSError, ConnectionError) as refresh_err:
                    logger.warning(f"[AutoSyncDaemon] Manifest refresh failed: {refresh_err}")

        except (RuntimeError, OSError, ConnectionError) as e:
            self._errors_count += 1
            self._last_error = str(e)
            logger.error(f"[AutoSyncDaemon] Error handling CONFIG_UPDATED: {e}")
