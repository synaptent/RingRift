"""Cluster-Wide Model Inventory System for RingRift.

This module provides a unified inventory of ALL available NN and NNUE models
across the entire RingRift cluster. It integrates with:
- Local model discovery (discover_models)
- Cluster-wide model discovery (ClusterModelDiscovery)
- NNUE registry for NNUE models
- EloService for tracking existing ratings

The inventory is used by the evaluation scheduler to identify models
that need fresh Elo rankings through gauntlets and tournaments.

Usage:
    from app.coordination.cluster_model_inventory import (
        get_cluster_model_inventory,
        ClusterModelEntry,
    )

    inventory = get_cluster_model_inventory()
    await inventory.build_full_inventory()

    # Get models needing evaluation
    models = inventory.get_models_needing_evaluation()
    for model in models:
        print(f"{model.model_id}: needs eval under {model.compatible_harnesses}")
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from app.ai.harness.base_harness import HarnessType, ModelType
from app.ai.harness.harness_registry import (
    get_compatible_harnesses,
    get_harnesses_for_model_and_players,
)

logger = logging.getLogger(__name__)

# Constants
ELO_STALENESS_DAYS = 7  # Elo ratings older than this are considered stale
MIN_GAMES_FOR_VALID_ELO = 30  # Minimum games for a meaningful Elo rating


@dataclass
class ClusterModelEntry:
    """Unified model entry with cluster-wide metadata.

    Attributes:
        model_id: Unique identifier for the model (usually path stem)
        model_type: "nn" for neural network, "nnue" for NNUE
        path: Local or remote path to the model file
        node_id: ID of the node where model is located
        board_type: Board type (hex8, square8, square19, hexagonal)
        num_players: Number of players (2, 3, or 4)
        compatible_harnesses: List of harness types compatible with this model
        has_elo: Whether the model has any Elo rating
        elo_stale: Whether the Elo rating is older than ELO_STALENESS_DAYS
        elo_rating: Current Elo rating if available
        elo_games: Number of games used for Elo calculation
        last_elo_update: Timestamp of last Elo update
        is_local: Whether the model is available locally
        size_bytes: File size in bytes
        architecture_version: Architecture version if known (v2, v4, v5-heavy, etc.)
        sync_priority: Priority for syncing to local node
    """

    model_id: str
    model_type: str  # "nn" or "nnue"
    path: str
    node_id: str
    board_type: str
    num_players: int
    compatible_harnesses: list[HarnessType] = field(default_factory=list)
    has_elo: bool = False
    elo_stale: bool = False
    elo_rating: float | None = None
    elo_games: int = 0
    last_elo_update: float | None = None
    is_local: bool = False
    size_bytes: int = 0
    architecture_version: str | None = None
    sync_priority: int = 0

    def needs_evaluation(self) -> bool:
        """Check if this model needs fresh evaluation."""
        # No Elo at all
        if not self.has_elo:
            return True
        # Stale Elo
        if self.elo_stale:
            return True
        # Not enough games for confidence
        if self.elo_games < MIN_GAMES_FOR_VALID_ELO:
            return True
        return False

    def get_priority_score(self) -> float:
        """Get priority score for evaluation scheduling.

        Higher score = higher priority.
        """
        score = 0.0

        # No Elo is highest priority
        if not self.has_elo:
            score += 300.0

        # Stale Elo gets a boost
        if self.elo_stale:
            score += 150.0

        # Low game count gets a boost
        if self.elo_games < MIN_GAMES_FOR_VALID_ELO:
            score += 100.0 * (1 - self.elo_games / MIN_GAMES_FOR_VALID_ELO)

        # 4-player configs get 2x priority (least data typically)
        if self.num_players == 4:
            score *= 2.0
        elif self.num_players == 3:
            score *= 1.5

        # Local models are easier to evaluate
        if self.is_local:
            score += 20.0

        return score

    def get_config_key(self) -> str:
        """Get config key for this model."""
        return f"{self.board_type}_{self.num_players}p"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "path": self.path,
            "node_id": self.node_id,
            "board_type": self.board_type,
            "num_players": self.num_players,
            "compatible_harnesses": [h.value for h in self.compatible_harnesses],
            "has_elo": self.has_elo,
            "elo_stale": self.elo_stale,
            "elo_rating": self.elo_rating,
            "elo_games": self.elo_games,
            "last_elo_update": self.last_elo_update,
            "is_local": self.is_local,
            "size_bytes": self.size_bytes,
            "architecture_version": self.architecture_version,
            "sync_priority": self.sync_priority,
        }


class ClusterModelInventoryManager:
    """Manages a unified inventory of all models across the cluster.

    This class discovers models from:
    1. Local models directory (NN and NNUE)
    2. Remote cluster nodes via ClusterModelDiscovery
    3. NNUE registry for canonical NNUE models

    And cross-references with EloService to determine which models
    need fresh evaluation.
    """

    def __init__(
        self,
        elo_staleness_days: int = ELO_STALENESS_DAYS,
        min_games_for_valid_elo: int = MIN_GAMES_FOR_VALID_ELO,
    ):
        """Initialize the inventory manager.

        Args:
            elo_staleness_days: Days after which Elo ratings are considered stale
            min_games_for_valid_elo: Minimum games for valid Elo confidence
        """
        self.elo_staleness_days = elo_staleness_days
        self.min_games_for_valid_elo = min_games_for_valid_elo
        self.node_id = socket.gethostname()

        # Inventory state
        self._inventory: dict[str, ClusterModelEntry] = {}
        self._last_build_time: float = 0
        self._build_in_progress: bool = False

    async def build_full_inventory(
        self,
        include_remote: bool = True,
        max_remote_nodes: int = 10,
        force_refresh: bool = False,
    ) -> list[ClusterModelEntry]:
        """Build complete inventory of all models across cluster.

        Args:
            include_remote: Whether to query remote nodes
            max_remote_nodes: Maximum number of remote nodes to query
            force_refresh: Force rebuild even if cache is fresh

        Returns:
            List of all discovered model entries
        """
        # Skip if build in progress
        if self._build_in_progress:
            logger.debug("Inventory build already in progress, returning cached")
            return list(self._inventory.values())

        # Skip if recently built (< 5 minutes) unless forced
        if not force_refresh and (time.time() - self._last_build_time) < 300:
            logger.debug("Using cached inventory (built < 5 min ago)")
            return list(self._inventory.values())

        self._build_in_progress = True
        try:
            logger.info("Building cluster-wide model inventory...")

            # Clear existing inventory
            self._inventory.clear()

            # Discover local NN models
            await self._discover_local_nn_models()

            # Discover local NNUE models
            await self._discover_local_nnue_models()

            # Discover remote models if enabled
            if include_remote:
                await self._discover_remote_models(max_nodes=max_remote_nodes)

            # Cross-reference with Elo service
            await self._enrich_with_elo_data()

            # Compute compatible harnesses for each model
            self._compute_harness_compatibility()

            self._last_build_time = time.time()
            logger.info(f"Inventory complete: {len(self._inventory)} models found")

            return list(self._inventory.values())

        finally:
            self._build_in_progress = False

    async def _discover_local_nn_models(self) -> None:
        """Discover local neural network models."""
        try:
            from app.models.discovery import discover_models

            # Run discovery in thread to avoid blocking
            models = await asyncio.to_thread(
                discover_models,
                model_type="nn",
                include_unknown=False,
            )

            for model in models:
                entry = ClusterModelEntry(
                    model_id=model.name,
                    model_type="nn",
                    path=model.path,
                    node_id=self.node_id,
                    board_type=model.board_type,
                    num_players=model.num_players,
                    is_local=True,
                    size_bytes=model.size_bytes,
                    architecture_version=model.architecture_version,
                    sync_priority=100,  # Local is highest priority
                )
                self._add_entry(entry)

            logger.debug(f"Discovered {len(models)} local NN models")

        except ImportError as e:
            logger.warning(f"Could not import discovery module: {e}")
        except Exception as e:
            logger.error(f"Error discovering local NN models: {e}")

    async def _discover_local_nnue_models(self) -> None:
        """Discover local NNUE models from registry."""
        try:
            from app.ai.nnue_registry.registry import get_existing_nnue_models

            # Run discovery in thread
            nnue_models = await asyncio.to_thread(get_existing_nnue_models)

            for info in nnue_models:
                entry = ClusterModelEntry(
                    model_id=f"nnue_{info.config_key}",
                    model_type="nnue",
                    path=str(info.path),
                    node_id=self.node_id,
                    board_type=info.board_type,
                    num_players=info.num_players,
                    is_local=True,
                    size_bytes=info.file_size_bytes,
                    sync_priority=100,
                )
                self._add_entry(entry)

            logger.debug(f"Discovered {len(nnue_models)} local NNUE models")

        except ImportError as e:
            logger.warning(f"Could not import NNUE registry: {e}")
        except Exception as e:
            logger.error(f"Error discovering local NNUE models: {e}")

    async def _discover_remote_models(self, max_nodes: int = 10) -> None:
        """Discover models on remote cluster nodes."""
        try:
            from app.models.cluster_discovery import get_cluster_model_discovery

            discovery = get_cluster_model_discovery()

            # Query remote nodes for all board types
            board_types = ["hex8", "square8", "square19", "hexagonal"]
            player_counts = [2, 3, 4]

            for board_type in board_types:
                for num_players in player_counts:
                    try:
                        remote_models = await asyncio.to_thread(
                            discovery.discover_cluster_models,
                            board_type=board_type,
                            num_players=num_players,
                            include_local=False,
                            include_remote=True,
                            max_remote_nodes=max_nodes,
                        )

                        for rm in remote_models:
                            entry = ClusterModelEntry(
                                model_id=rm.model_info.name,
                                model_type=rm.model_info.model_type,
                                path=rm.remote_path,
                                node_id=rm.node_id,
                                board_type=board_type,
                                num_players=num_players,
                                is_local=False,
                                size_bytes=rm.model_info.size_bytes,
                                architecture_version=rm.model_info.architecture_version,
                                sync_priority=rm.sync_priority,
                            )
                            self._add_entry(entry)

                    except Exception as e:
                        logger.debug(
                            f"Error discovering {board_type}_{num_players}p "
                            f"on remote nodes: {e}"
                        )

            logger.debug("Remote model discovery complete")

        except ImportError as e:
            logger.warning(f"Could not import cluster discovery: {e}")
        except Exception as e:
            logger.error(f"Error discovering remote models: {e}")

    async def _enrich_with_elo_data(self) -> None:
        """Enrich inventory entries with Elo rating data."""
        try:
            from app.training.elo_service import get_elo_service

            elo_service = get_elo_service()
            staleness_threshold = time.time() - (self.elo_staleness_days * 86400)

            for entry in self._inventory.values():
                try:
                    # Try to get rating for this model
                    rating = await asyncio.to_thread(
                        elo_service.get_rating,
                        entry.model_id,
                        entry.board_type,
                        entry.num_players,
                    )

                    if rating and rating.games_played > 0:
                        entry.has_elo = True
                        entry.elo_rating = rating.rating
                        entry.elo_games = rating.games_played
                        entry.last_elo_update = rating.last_update

                        # Check staleness
                        if rating.last_update < staleness_threshold:
                            entry.elo_stale = True

                except Exception:
                    # Model not in Elo database - that's fine
                    pass

        except ImportError as e:
            logger.warning(f"Could not import EloService: {e}")
        except Exception as e:
            logger.error(f"Error enriching with Elo data: {e}")

    def _compute_harness_compatibility(self) -> None:
        """Compute compatible harnesses for each model."""
        for entry in self._inventory.values():
            # Determine model type enum
            if entry.model_type == "nn":
                model_type = ModelType.NEURAL_NET
            elif entry.model_type == "nnue":
                model_type = ModelType.NNUE
            else:
                model_type = ModelType.HEURISTIC

            # Get compatible harnesses for this model and player count
            try:
                harnesses = get_harnesses_for_model_and_players(
                    model_type=model_type,
                    num_players=entry.num_players,
                )
                entry.compatible_harnesses = harnesses
            except Exception as e:
                logger.debug(
                    f"Error computing harness compatibility for {entry.model_id}: {e}"
                )
                # Default to empty list
                entry.compatible_harnesses = []

    def _add_entry(self, entry: ClusterModelEntry) -> None:
        """Add entry to inventory, deduplicating by model_id."""
        key = f"{entry.model_id}:{entry.node_id}"

        # Prefer local entries over remote
        if key in self._inventory:
            existing = self._inventory[key]
            if entry.is_local and not existing.is_local:
                self._inventory[key] = entry
            # Keep existing if it's local and new is remote
            return

        self._inventory[key] = entry

    def get_models_needing_evaluation(
        self,
        board_type: str | None = None,
        num_players: int | None = None,
        model_type: str | None = None,
        local_only: bool = False,
    ) -> list[ClusterModelEntry]:
        """Get models that need fresh Elo evaluation.

        Args:
            board_type: Filter by board type
            num_players: Filter by number of players
            model_type: Filter by model type ("nn" or "nnue")
            local_only: Only return models available locally

        Returns:
            List of entries sorted by priority (highest first)
        """
        results = []

        for entry in self._inventory.values():
            # Apply filters
            if board_type and entry.board_type != board_type:
                continue
            if num_players and entry.num_players != num_players:
                continue
            if model_type and entry.model_type != model_type:
                continue
            if local_only and not entry.is_local:
                continue

            # Check if needs evaluation
            if entry.needs_evaluation():
                results.append(entry)

        # Sort by priority (highest first)
        results.sort(key=lambda e: e.get_priority_score(), reverse=True)

        return results

    def get_all_models(
        self,
        board_type: str | None = None,
        num_players: int | None = None,
    ) -> list[ClusterModelEntry]:
        """Get all models in inventory.

        Args:
            board_type: Optional filter by board type
            num_players: Optional filter by number of players

        Returns:
            List of all model entries matching filters
        """
        results = []

        for entry in self._inventory.values():
            if board_type and entry.board_type != board_type:
                continue
            if num_players and entry.num_players != num_players:
                continue
            results.append(entry)

        return results

    def get_model_by_id(
        self,
        model_id: str,
        node_id: str | None = None,
    ) -> ClusterModelEntry | None:
        """Get a specific model by ID.

        Args:
            model_id: Model identifier
            node_id: Optional node filter (defaults to preferring local)

        Returns:
            Model entry or None if not found
        """
        # First try exact key match
        if node_id:
            key = f"{model_id}:{node_id}"
            if key in self._inventory:
                return self._inventory[key]

        # Search for model_id across all nodes, preferring local
        candidates = [
            e for e in self._inventory.values()
            if e.model_id == model_id
        ]

        if not candidates:
            return None

        # Prefer local
        local = [e for e in candidates if e.is_local]
        if local:
            return local[0]

        return candidates[0]

    def get_inventory_stats(self) -> dict[str, Any]:
        """Get statistics about the inventory.

        Returns:
            Dictionary with inventory statistics
        """
        total = len(self._inventory)
        local_count = sum(1 for e in self._inventory.values() if e.is_local)
        remote_count = total - local_count

        nn_count = sum(1 for e in self._inventory.values() if e.model_type == "nn")
        nnue_count = sum(1 for e in self._inventory.values() if e.model_type == "nnue")

        needs_eval = len(self.get_models_needing_evaluation())
        has_elo = sum(1 for e in self._inventory.values() if e.has_elo)
        stale_elo = sum(1 for e in self._inventory.values() if e.elo_stale)

        # Count by config
        by_config: dict[str, int] = {}
        for entry in self._inventory.values():
            config_key = entry.get_config_key()
            by_config[config_key] = by_config.get(config_key, 0) + 1

        return {
            "total_models": total,
            "local_models": local_count,
            "remote_models": remote_count,
            "nn_models": nn_count,
            "nnue_models": nnue_count,
            "needs_evaluation": needs_eval,
            "has_elo": has_elo,
            "stale_elo": stale_elo,
            "by_config": by_config,
            "last_build_time": self._last_build_time,
            "node_id": self.node_id,
        }

    def clear(self) -> None:
        """Clear the inventory cache."""
        self._inventory.clear()
        self._last_build_time = 0


# Module-level singleton
_inventory_manager: ClusterModelInventoryManager | None = None


def get_cluster_model_inventory() -> ClusterModelInventoryManager:
    """Get the singleton ClusterModelInventoryManager instance."""
    global _inventory_manager
    if _inventory_manager is None:
        _inventory_manager = ClusterModelInventoryManager()
    return _inventory_manager


def reset_cluster_model_inventory() -> None:
    """Reset the singleton (for testing)."""
    global _inventory_manager
    _inventory_manager = None


# CLI for inventory inspection
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cluster Model Inventory")
    parser.add_argument(
        "--build", action="store_true", help="Build full inventory"
    )
    parser.add_argument(
        "--needs-eval", action="store_true", help="Show models needing evaluation"
    )
    parser.add_argument(
        "--stats", action="store_true", help="Show inventory statistics"
    )
    parser.add_argument(
        "--board-type", type=str, help="Filter by board type"
    )
    parser.add_argument(
        "--num-players", type=int, help="Filter by number of players"
    )
    parser.add_argument(
        "--local-only", action="store_true", help="Only show local models"
    )
    parser.add_argument(
        "--no-remote", action="store_true", help="Skip remote node discovery"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    async def main():
        inventory = get_cluster_model_inventory()

        if args.build or args.needs_eval or args.stats:
            await inventory.build_full_inventory(
                include_remote=not args.no_remote,
                force_refresh=True,
            )

        if args.stats:
            stats = inventory.get_inventory_stats()
            print("\n=== Cluster Model Inventory Statistics ===\n")
            print(f"Total models:       {stats['total_models']}")
            print(f"  Local:            {stats['local_models']}")
            print(f"  Remote:           {stats['remote_models']}")
            print(f"  NN:               {stats['nn_models']}")
            print(f"  NNUE:             {stats['nnue_models']}")
            print(f"Needs evaluation:   {stats['needs_evaluation']}")
            print(f"Has Elo:            {stats['has_elo']}")
            print(f"Stale Elo:          {stats['stale_elo']}")
            print(f"\nBy config:")
            for config_key, count in sorted(stats['by_config'].items()):
                print(f"  {config_key}: {count}")
            print()

        if args.needs_eval:
            models = inventory.get_models_needing_evaluation(
                board_type=args.board_type,
                num_players=args.num_players,
                local_only=args.local_only,
            )
            print(f"\n=== Models Needing Evaluation ({len(models)}) ===\n")
            print(f"{'Model ID':<40} {'Type':<6} {'Config':<15} {'Node':<20} {'Priority':<8}")
            print("-" * 95)
            for model in models[:50]:  # Limit output
                print(
                    f"{model.model_id[:40]:<40} "
                    f"{model.model_type:<6} "
                    f"{model.get_config_key():<15} "
                    f"{model.node_id[:20]:<20} "
                    f"{model.get_priority_score():.1f}"
                )
            if len(models) > 50:
                print(f"... and {len(models) - 50} more")
            print()

    asyncio.run(main())
