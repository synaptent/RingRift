"""Queue Populator Daemon - Automatic work queue population (December 2025).

This daemon automatically populates the work queue based on cluster state,
eliminating manual job queuing. It enables self-organizing cluster behavior
where idle resources automatically pick up useful work.

Key features:
- Monitors cluster node status for idle resources
- Detects data needs per configuration (games, samples)
- Prioritizes work by curriculum weights and data gaps
- Respects node policies and capabilities
- Emits events for work queue changes
- Integrates with ClusterManifest for data awareness

Decision Logic:
1. Scan for idle nodes with matching capabilities
2. Check data needs per config (games below threshold)
3. Weight by curriculum priorities and data gaps
4. Generate appropriate work items (selfplay, export, validation)
5. Avoid over-population (queue cap)

Usage:
    from app.coordination.queue_populator_daemon import QueuePopulatorDaemon

    daemon = QueuePopulatorDaemon()
    await daemon.start()

December 2025: Created as part of Phase 2 automation improvements.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Board configurations
BOARD_CONFIGS = [
    ("square8", 2),
    ("square8", 3),
    ("square8", 4),
    ("square19", 2),
    ("square19", 3),
    ("square19", 4),
    ("hex8", 2),
    ("hex8", 3),
    ("hex8", 4),
    ("hexagonal", 2),
    ("hexagonal", 3),
    ("hexagonal", 4),
]


@dataclass
class QueuePopulatorConfig:
    """Configuration for queue populator daemon."""

    enabled: bool = True
    # Scan interval
    scan_interval_seconds: int = 60  # 1 minute
    # Maximum pending items in queue before stopping generation
    max_pending_items: int = 50
    # Minimum idle nodes to trigger population
    min_idle_nodes_to_populate: int = 1
    # Games per selfplay batch
    games_per_selfplay_batch: int = 500
    # Target games per config for data balance
    target_games_per_config: int = 10000
    # Timeout for selfplay work (1 hour)
    selfplay_timeout_seconds: float = 3600.0
    # Timeout for export work (1 hour)
    export_timeout_seconds: float = 3600.0
    # Timeout for validation work (30 minutes)
    validation_timeout_seconds: float = 1800.0
    # Priorities
    selfplay_base_priority: int = 50
    export_priority: int = 70
    validation_priority: int = 60
    # Balance multiplier for data-starved configs
    data_gap_priority_boost: int = 20


@dataclass
class ConfigDataState:
    """Tracks data availability for a configuration."""

    config_key: str
    board_type: str
    num_players: int
    # Game counts
    total_games: int = 0
    games_since_last_export: int = 0
    # Sample counts
    total_samples: int = 0
    # Last update times
    last_game_time: float = 0.0
    last_export_time: float = 0.0
    # Pending work
    pending_selfplay_count: int = 0
    pending_export: bool = False
    # Curriculum weight (0.0-1.0)
    curriculum_weight: float = 1.0


class QueuePopulatorDaemon:
    """Daemon that automatically populates the work queue based on cluster state."""

    def __init__(self, config: QueuePopulatorConfig | None = None):
        self.config = config or QueuePopulatorConfig()
        self._running = False
        self._task: asyncio.Task | None = None
        self._data_states: dict[str, ConfigDataState] = {}
        self._event_subscriptions: list[Any] = []
        self._last_population_time: float = 0.0

        # Initialize data states for all configs
        for board_type, num_players in BOARD_CONFIGS:
            config_key = f"{board_type}_{num_players}p"
            self._data_states[config_key] = ConfigDataState(
                config_key=config_key,
                board_type=board_type,
                num_players=num_players,
            )

    async def start(self) -> None:
        """Start the queue populator daemon."""
        if self._running:
            logger.warning("[QueuePopulator] Already running")
            return

        self._running = True
        logger.info("[QueuePopulator] Starting queue populator daemon")

        # Subscribe to events
        await self._subscribe_to_events()

        # Load curriculum weights
        self._load_curriculum_weights()

        # Start background monitoring task
        self._task = asyncio.create_task(self._monitor_loop())
        self._task.add_done_callback(self._on_task_done)

    async def stop(self) -> None:
        """Stop the queue populator daemon."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Unsubscribe from events
        for unsub in self._event_subscriptions:
            try:
                if callable(unsub):
                    unsub()
            except Exception as e:
                logger.debug(f"[QueuePopulator] Error unsubscribing: {e}")

        logger.info("[QueuePopulator] Stopped")

    def _on_task_done(self, task: asyncio.Task) -> None:
        """Handle task completion or failure."""
        try:
            exc = task.exception()
            if exc:
                logger.error(f"[QueuePopulator] Task failed: {exc}")
        except asyncio.CancelledError:
            pass
        except asyncio.InvalidStateError:
            pass

    async def _subscribe_to_events(self) -> None:
        """Subscribe to relevant events."""
        # Subscribe to selfplay completion
        try:
            from app.coordination.stage_events import StageEvent, get_event_bus

            bus = get_event_bus()
            unsub = bus.subscribe(
                StageEvent.SELFPLAY_COMPLETE, self._on_selfplay_complete
            )
            self._event_subscriptions.append(unsub)
            logger.info("[QueuePopulator] Subscribed to SELFPLAY_COMPLETE events")
        except ImportError:
            logger.debug("[QueuePopulator] Stage events not available")

        # Subscribe to export completion
        try:
            from app.coordination.stage_events import StageEvent, get_event_bus

            bus = get_event_bus()
            unsub = bus.subscribe(
                StageEvent.NPZ_EXPORT_COMPLETE, self._on_export_complete
            )
            self._event_subscriptions.append(unsub)
            logger.info("[QueuePopulator] Subscribed to NPZ_EXPORT_COMPLETE events")
        except ImportError:
            pass

        # Subscribe to work events
        try:
            from app.distributed.data_events import DataEventType, get_event_bus

            bus = get_event_bus()
            unsub = bus.subscribe(DataEventType.WORK_COMPLETED, self._on_work_completed)
            self._event_subscriptions.append(unsub)
            unsub = bus.subscribe(DataEventType.WORK_FAILED, self._on_work_failed)
            self._event_subscriptions.append(unsub)
            logger.info("[QueuePopulator] Subscribed to WORK events")
        except ImportError:
            logger.debug("[QueuePopulator] Data events not available")

    async def _on_selfplay_complete(self, result: Any) -> None:
        """Handle selfplay completion - update data state."""
        try:
            board_type = getattr(result, "board_type", None)
            num_players = getattr(result, "num_players", None)
            games_generated = getattr(result, "games_generated", 0)

            if not board_type or not num_players:
                metadata = getattr(result, "metadata", {})
                board_type = board_type or metadata.get("board_type")
                num_players = num_players or metadata.get("num_players")

            if board_type and num_players:
                config_key = f"{board_type}_{num_players}p"
                if config_key in self._data_states:
                    state = self._data_states[config_key]
                    state.total_games += games_generated
                    state.games_since_last_export += games_generated
                    state.last_game_time = time.time()
                    if state.pending_selfplay_count > 0:
                        state.pending_selfplay_count -= 1

                    logger.debug(
                        f"[QueuePopulator] {config_key}: +{games_generated} games, "
                        f"total: {state.total_games}"
                    )

        except Exception as e:
            logger.error(f"[QueuePopulator] Error handling selfplay complete: {e}")

    async def _on_export_complete(self, result: Any) -> None:
        """Handle export completion - update data state."""
        try:
            metadata = getattr(result, "metadata", {})
            config_key = metadata.get("config")
            samples = metadata.get("samples", 0)

            if config_key and config_key in self._data_states:
                state = self._data_states[config_key]
                state.games_since_last_export = 0
                state.last_export_time = time.time()
                state.total_samples = samples
                state.pending_export = False

                logger.debug(
                    f"[QueuePopulator] {config_key}: export complete, "
                    f"{samples} samples"
                )

        except Exception as e:
            logger.error(f"[QueuePopulator] Error handling export complete: {e}")

    async def _on_work_completed(self, event: Any) -> None:
        """Handle work completion event."""
        try:
            payload = getattr(event, "payload", event)
            if isinstance(payload, dict):
                work_type = payload.get("work_type")
                if work_type == "selfplay":
                    board_type = payload.get("board_type")
                    num_players = payload.get("num_players")
                    if board_type and num_players:
                        config_key = f"{board_type}_{num_players}p"
                        if config_key in self._data_states:
                            state = self._data_states[config_key]
                            if state.pending_selfplay_count > 0:
                                state.pending_selfplay_count -= 1
        except Exception as e:
            logger.debug(f"[QueuePopulator] Error handling work completed: {e}")

    async def _on_work_failed(self, event: Any) -> None:
        """Handle work failure event."""
        try:
            payload = getattr(event, "payload", event)
            if isinstance(payload, dict):
                work_type = payload.get("work_type")
                if work_type == "selfplay":
                    board_type = payload.get("board_type")
                    num_players = payload.get("num_players")
                    if board_type and num_players:
                        config_key = f"{board_type}_{num_players}p"
                        if config_key in self._data_states:
                            state = self._data_states[config_key]
                            if state.pending_selfplay_count > 0:
                                state.pending_selfplay_count -= 1
        except Exception as e:
            logger.debug(f"[QueuePopulator] Error handling work failed: {e}")

    def _load_curriculum_weights(self) -> None:
        """Load curriculum weights for prioritization."""
        try:
            from app.coordination.curriculum_weights import load_curriculum_weights

            weights = load_curriculum_weights()
            for config_key, weight in weights.items():
                if config_key in self._data_states:
                    self._data_states[config_key].curriculum_weight = weight
            logger.info("[QueuePopulator] Loaded curriculum weights")
        except ImportError:
            # Use default weights
            default_weights = {
                "square8_2p": 1.0,
                "square8_3p": 0.7,
                "square8_4p": 0.5,
                "square19_2p": 0.8,
                "square19_3p": 0.5,
                "square19_4p": 0.4,
                "hex8_2p": 0.9,
                "hex8_3p": 0.6,
                "hex8_4p": 0.5,
                "hexagonal_2p": 0.7,
                "hexagonal_3p": 0.5,
                "hexagonal_4p": 0.4,
            }
            for config_key, weight in default_weights.items():
                if config_key in self._data_states:
                    self._data_states[config_key].curriculum_weight = weight
            logger.debug("[QueuePopulator] Using default curriculum weights")

    async def _monitor_loop(self) -> None:
        """Background loop to periodically populate queue."""
        while self._running:
            try:
                await self._populate_queue()
                await asyncio.sleep(self.config.scan_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[QueuePopulator] Monitor loop error: {e}")
                await asyncio.sleep(30)

    async def _populate_queue(self) -> None:
        """Main population logic - check state and add work."""
        if not self.config.enabled:
            return

        try:
            # Get current queue status
            queue_status = self._get_queue_status()
            pending_count = queue_status.get("pending", 0)

            # Check if queue is already full
            if pending_count >= self.config.max_pending_items:
                logger.debug(
                    f"[QueuePopulator] Queue full ({pending_count} pending), skipping"
                )
                return

            # Get idle node count
            idle_nodes = await self._count_idle_nodes()
            if idle_nodes < self.config.min_idle_nodes_to_populate:
                logger.debug(
                    f"[QueuePopulator] Not enough idle nodes ({idle_nodes}), skipping"
                )
                return

            # Update data states from cluster manifest
            await self._refresh_data_states()

            # Calculate how many items to add
            slots_available = min(
                self.config.max_pending_items - pending_count,
                idle_nodes * 2,  # Up to 2 items per idle node
            )

            if slots_available <= 0:
                return

            # Generate prioritized work list
            work_items = self._generate_work_items(slots_available)

            # Add items to queue
            added = 0
            for item in work_items:
                if await self._add_work_item(item):
                    added += 1

            if added > 0:
                self._last_population_time = time.time()
                logger.info(
                    f"[QueuePopulator] Added {added} work items "
                    f"({idle_nodes} idle nodes, {pending_count} pending)"
                )

        except Exception as e:
            logger.error(f"[QueuePopulator] Population error: {e}")

    def _get_queue_status(self) -> dict[str, Any]:
        """Get current work queue status."""
        try:
            from app.coordination.work_queue import get_work_queue

            queue = get_work_queue()
            return {
                "pending": queue.get_pending_count(),
                "running": queue.get_running_count(),
            }
        except ImportError:
            return {"pending": 0, "running": 0}

    async def _count_idle_nodes(self) -> int:
        """Count nodes that are idle and available for work."""
        try:
            # Try P2P status first
            from app.distributed.p2p_daemon import get_p2p_daemon

            daemon = get_p2p_daemon()
            if daemon and hasattr(daemon, "get_peer_status"):
                peers = daemon.get_peer_status()
                idle = sum(
                    1
                    for peer in peers.values()
                    if peer.get("status") == "alive"
                    and not peer.get("busy", False)
                )
                return idle

        except ImportError:
            pass

        # Fallback: check work queue for running items
        try:
            from app.coordination.work_queue import get_work_queue

            queue = get_work_queue()
            running = queue.get_running_count()
            # Assume some cluster size
            estimated_total = 20
            return max(0, estimated_total - running)

        except ImportError:
            return 5  # Default assumption

    async def _refresh_data_states(self) -> None:
        """Update data states from cluster manifest and local discovery."""
        # Try cluster manifest
        try:
            from app.distributed.cluster_manifest import get_cluster_manifest

            manifest = get_cluster_manifest()
            summary = manifest.get_data_summary()

            for config_key, info in summary.items():
                if config_key in self._data_states:
                    state = self._data_states[config_key]
                    state.total_games = info.get("total_games", 0)
                    state.total_samples = info.get("total_samples", 0)

        except ImportError:
            pass

        # Try local game discovery
        try:
            from app.utils.game_discovery import GameDiscovery

            discovery = GameDiscovery()
            for db_info in discovery.find_all_databases():
                if db_info.board_type and db_info.num_players:
                    config_key = f"{db_info.board_type}_{db_info.num_players}p"
                    if config_key in self._data_states:
                        state = self._data_states[config_key]
                        state.total_games = max(
                            state.total_games, db_info.game_count
                        )

        except ImportError:
            pass

    def _generate_work_items(self, max_items: int) -> list[dict[str, Any]]:
        """Generate prioritized list of work items to add."""
        items = []

        # Calculate priorities for each config
        priorities: list[tuple[float, str, str, dict]] = []

        for config_key, state in self._data_states.items():
            if state.curriculum_weight <= 0:
                continue

            # Skip if already has pending selfplay
            if state.pending_selfplay_count >= 2:
                continue

            # Calculate data gap priority boost
            gap_ratio = 1.0 - min(
                state.total_games / self.config.target_games_per_config, 1.0
            )
            gap_boost = int(gap_ratio * self.config.data_gap_priority_boost)

            # Calculate final priority
            priority = (
                self.config.selfplay_base_priority
                + int(state.curriculum_weight * 30)
                + gap_boost
            )

            # Selfplay item
            selfplay_item = {
                "work_type": "selfplay",
                "priority": priority,
                "config": {
                    "board_type": state.board_type,
                    "num_players": state.num_players,
                    "num_games": self.config.games_per_selfplay_batch,
                    "auto_generated": True,
                },
                "timeout_seconds": self.config.selfplay_timeout_seconds,
            }
            priorities.append((priority, config_key, "selfplay", selfplay_item))

            # Check if export needed
            if (
                state.games_since_last_export >= 500
                and not state.pending_export
            ):
                export_item = {
                    "work_type": "data_sync",  # Maps to export
                    "priority": self.config.export_priority,
                    "config": {
                        "board_type": state.board_type,
                        "num_players": state.num_players,
                        "action": "export",
                        "auto_generated": True,
                    },
                    "timeout_seconds": self.config.export_timeout_seconds,
                }
                priorities.append(
                    (self.config.export_priority, config_key, "export", export_item)
                )

        # Sort by priority (descending)
        priorities.sort(key=lambda x: -x[0])

        # Take top items
        for priority, config_key, work_type, item in priorities[:max_items]:
            items.append(item)

            # Track pending state
            state = self._data_states[config_key]
            if work_type == "selfplay":
                state.pending_selfplay_count += 1
            elif work_type == "export":
                state.pending_export = True

        return items

    async def _add_work_item(self, item_config: dict[str, Any]) -> bool:
        """Add a work item to the queue."""
        try:
            from app.coordination.work_queue import WorkItem, WorkType, get_work_queue

            queue = get_work_queue()

            # Map work type string to enum
            work_type_map = {
                "selfplay": WorkType.SELFPLAY,
                "training": WorkType.TRAINING,
                "data_sync": WorkType.DATA_SYNC,
                "data_merge": WorkType.DATA_MERGE,
                "validation": WorkType.VALIDATION,
                "gauntlet": WorkType.GAUNTLET,
            }

            work_type = work_type_map.get(
                item_config.get("work_type", "selfplay"), WorkType.SELFPLAY
            )

            item = WorkItem(
                work_type=work_type,
                priority=item_config.get("priority", 50),
                config=item_config.get("config", {}),
                timeout_seconds=item_config.get("timeout_seconds", 3600.0),
            )

            queue.add_work(item)
            return True

        except Exception as e:
            logger.error(f"[QueuePopulator] Failed to add work item: {e}")
            return False

    def get_status(self) -> dict[str, Any]:
        """Get current daemon status."""
        return {
            "running": self._running,
            "last_population": self._last_population_time,
            "config_states": {
                key: {
                    "total_games": state.total_games,
                    "games_pending_export": state.games_since_last_export,
                    "total_samples": state.total_samples,
                    "pending_selfplay": state.pending_selfplay_count,
                    "pending_export": state.pending_export,
                    "curriculum_weight": state.curriculum_weight,
                }
                for key, state in self._data_states.items()
            },
        }


# Singleton instance
_daemon: QueuePopulatorDaemon | None = None


def get_queue_populator_daemon() -> QueuePopulatorDaemon:
    """Get or create the singleton queue populator daemon."""
    global _daemon
    if _daemon is None:
        _daemon = QueuePopulatorDaemon()
    return _daemon


async def start_queue_populator_daemon() -> QueuePopulatorDaemon:
    """Start the queue populator daemon (convenience function)."""
    daemon = get_queue_populator_daemon()
    await daemon.start()
    return daemon


__all__ = [
    "QueuePopulatorConfig",
    "QueuePopulatorDaemon",
    "ConfigDataState",
    "get_queue_populator_daemon",
    "start_queue_populator_daemon",
]
