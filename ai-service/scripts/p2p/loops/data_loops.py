"""Data Management Loops for P2P Orchestrator.

December 2025: Background loops for data processing and synchronization.

Loops:
- ModelSyncLoop: Synchronizes models across cluster nodes
- DataAggregationLoop: Aggregates training data from nodes

Usage:
    from scripts.p2p.loops import ModelSyncLoop, DataAggregationLoop

    sync = ModelSyncLoop(
        get_model_versions=lambda: orchestrator.model_versions,
        sync_model=orchestrator.sync_model_to_node,
    )
    await sync.run_forever()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine

from .base import BaseLoop

logger = logging.getLogger(__name__)


@dataclass
class ModelSyncConfig:
    """Configuration for model sync loop."""

    check_interval_seconds: float = 120.0  # 2 minutes
    max_sync_operations_per_cycle: int = 5
    sync_timeout_seconds: float = 300.0  # 5 minutes
    priority_configs: list[str] = field(default_factory=lambda: ["hex8_2p", "square8_2p"])


class ModelSyncLoop(BaseLoop):
    """Background loop that synchronizes models across cluster nodes.

    Ensures all nodes have the latest models for their assigned configurations.
    Prioritizes recently promoted models and high-traffic configurations.
    """

    def __init__(
        self,
        get_model_versions: Callable[[], dict[str, dict[str, str]]],
        get_node_models: Callable[[str], Coroutine[Any, Any, dict[str, str]]],
        sync_model: Callable[[str, str, str], Coroutine[Any, Any, bool]],
        get_active_nodes: Callable[[], list[str]],
        config: ModelSyncConfig | None = None,
    ):
        """Initialize model sync loop.

        Args:
            get_model_versions: Callback returning config -> {version, path, checksum}
            get_node_models: Async callback to get node's model versions
            sync_model: Async callback to sync model to node (node_id, config, source_path)
            get_active_nodes: Callback returning list of active node IDs
            config: Sync configuration
        """
        self.config = config or ModelSyncConfig()
        super().__init__(
            name="model_sync",
            interval=self.config.check_interval_seconds,
        )
        self._get_model_versions = get_model_versions
        self._get_node_models = get_node_models
        self._sync_model = sync_model
        self._get_active_nodes = get_active_nodes
        self._sync_stats = {
            "models_synced": 0,
            "sync_failures": 0,
            "bytes_transferred": 0,
        }

    async def _run_once(self) -> None:
        """Check for and sync outdated models."""
        current_versions = self._get_model_versions()
        if not current_versions:
            return

        active_nodes = self._get_active_nodes()
        if not active_nodes:
            return

        sync_tasks: list[tuple[str, str, str]] = []  # (node_id, config, path)

        # Check each node's model versions
        for node_id in active_nodes:
            try:
                node_models = await self._get_node_models(node_id)

                for config, version_info in current_versions.items():
                    current_version = version_info.get("version", "")
                    node_version = node_models.get(config, "")

                    if node_version != current_version:
                        sync_tasks.append((
                            node_id,
                            config,
                            version_info.get("path", ""),
                        ))

            except Exception as e:
                logger.debug(f"[ModelSync] Failed to check {node_id}: {e}")

        # Sort by priority configs first
        def sync_priority(task: tuple[str, str, str]) -> int:
            _, config, _ = task
            if config in self.config.priority_configs:
                return 0
            return 1

        sync_tasks.sort(key=sync_priority)

        # Execute sync operations
        synced_count = 0
        for node_id, config, path in sync_tasks[:self.config.max_sync_operations_per_cycle]:
            try:
                success = await asyncio.wait_for(
                    self._sync_model(node_id, config, path),
                    timeout=self.config.sync_timeout_seconds,
                )
                if success:
                    synced_count += 1
                    self._sync_stats["models_synced"] += 1
                    logger.info(f"[ModelSync] Synced {config} to {node_id}")
                else:
                    self._sync_stats["sync_failures"] += 1
            except asyncio.TimeoutError:
                logger.warning(f"[ModelSync] Timeout syncing {config} to {node_id}")
                self._sync_stats["sync_failures"] += 1
            except Exception as e:
                logger.warning(f"[ModelSync] Failed to sync {config} to {node_id}: {e}")
                self._sync_stats["sync_failures"] += 1

        if synced_count > 0:
            logger.info(f"[ModelSync] Synced {synced_count} models this cycle")

    def get_sync_stats(self) -> dict[str, Any]:
        """Get sync statistics."""
        return {
            **self._sync_stats,
            **self.stats.to_dict(),
        }


@dataclass
class DataAggregationConfig:
    """Configuration for data aggregation loop."""

    check_interval_seconds: float = 300.0  # 5 minutes
    min_games_to_aggregate: int = 100
    max_nodes_per_cycle: int = 10
    aggregation_timeout_seconds: float = 600.0  # 10 minutes


class DataAggregationLoop(BaseLoop):
    """Background loop that aggregates training data from cluster nodes.

    Collects selfplay game databases from distributed nodes and consolidates
    them to central storage for training.
    """

    def __init__(
        self,
        get_node_game_counts: Callable[[], dict[str, int]],
        aggregate_from_node: Callable[[str], Coroutine[Any, Any, dict[str, Any]]],
        config: DataAggregationConfig | None = None,
    ):
        """Initialize data aggregation loop.

        Args:
            get_node_game_counts: Callback returning node_id -> game_count
            aggregate_from_node: Async callback to aggregate data from a node
            config: Aggregation configuration
        """
        self.config = config or DataAggregationConfig()
        super().__init__(
            name="data_aggregation",
            interval=self.config.check_interval_seconds,
        )
        self._get_node_game_counts = get_node_game_counts
        self._aggregate_from_node = aggregate_from_node
        self._aggregation_stats = {
            "total_games_aggregated": 0,
            "total_bytes_transferred": 0,
            "aggregation_failures": 0,
        }

    async def _run_once(self) -> None:
        """Check for nodes with data to aggregate."""
        node_counts = self._get_node_game_counts()
        if not node_counts:
            return

        # Find nodes with enough games to aggregate
        nodes_to_aggregate = [
            (node_id, count)
            for node_id, count in node_counts.items()
            if count >= self.config.min_games_to_aggregate
        ]

        # Sort by game count descending (highest priority first)
        nodes_to_aggregate.sort(key=lambda x: x[1], reverse=True)

        # Aggregate from top nodes
        aggregated_count = 0
        for node_id, game_count in nodes_to_aggregate[:self.config.max_nodes_per_cycle]:
            try:
                result = await asyncio.wait_for(
                    self._aggregate_from_node(node_id),
                    timeout=self.config.aggregation_timeout_seconds,
                )

                games = result.get("games_aggregated", 0)
                bytes_transferred = result.get("bytes_transferred", 0)

                if games > 0:
                    aggregated_count += games
                    self._aggregation_stats["total_games_aggregated"] += games
                    self._aggregation_stats["total_bytes_transferred"] += bytes_transferred
                    logger.info(
                        f"[DataAggregation] Aggregated {games} games from {node_id} "
                        f"({bytes_transferred / (1024**2):.1f} MB)"
                    )

            except asyncio.TimeoutError:
                logger.warning(f"[DataAggregation] Timeout aggregating from {node_id}")
                self._aggregation_stats["aggregation_failures"] += 1
            except Exception as e:
                logger.warning(f"[DataAggregation] Failed to aggregate from {node_id}: {e}")
                self._aggregation_stats["aggregation_failures"] += 1

        if aggregated_count > 0:
            logger.info(f"[DataAggregation] Aggregated {aggregated_count} total games this cycle")

    def get_aggregation_stats(self) -> dict[str, Any]:
        """Get aggregation statistics."""
        return {
            **self._aggregation_stats,
            **self.stats.to_dict(),
        }


__all__ = [
    "DataAggregationConfig",
    "DataAggregationLoop",
    "ModelSyncConfig",
    "ModelSyncLoop",
]
