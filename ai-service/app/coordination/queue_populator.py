"""
Work queue populator to maintain minimum queue depth until Elo targets are met.

This module ensures there are always at least N work items in the queue until
all board/player configurations reach the target Elo rating (default 2000).

Work distribution:
- 60% selfplay (data generation)
- 30% training (model improvement)
- 10% tournament (Elo measurement)
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from app.coordination.work_queue import WorkQueue

logger = logging.getLogger(__name__)


class BoardType(str, Enum):
    """Board types to train."""
    SQUARE8 = "square8"
    SQUARE19 = "square19"
    HEX8 = "hex8"
    HEXAGONAL = "hexagonal"


@dataclass
class ConfigTarget:
    """Target for a specific board/player configuration."""
    board_type: str
    num_players: int
    target_elo: float = 2000.0
    current_best_elo: float = 1500.0
    best_model_id: Optional[str] = None
    games_played: int = 0
    training_runs: int = 0
    last_updated: float = field(default_factory=time.time)

    @property
    def target_met(self) -> bool:
        return self.current_best_elo >= self.target_elo

    @property
    def elo_gap(self) -> float:
        return max(0, self.target_elo - self.current_best_elo)

    @property
    def config_key(self) -> str:
        return f"{self.board_type}_{self.num_players}p"


@dataclass
class PopulatorConfig:
    """Configuration for the queue populator."""
    # Minimum queue depth to maintain
    min_queue_depth: int = 50

    # Target Elo for all configurations
    target_elo: float = 2000.0

    # Work distribution (must sum to 1.0)
    selfplay_ratio: float = 0.60
    training_ratio: float = 0.30
    tournament_ratio: float = 0.10

    # Board types to train
    board_types: List[str] = field(default_factory=lambda: [
        "square8", "square19", "hex8", "hexagonal"
    ])

    # Player counts to train
    player_counts: List[int] = field(default_factory=lambda: [2, 3, 4])

    # Selfplay settings
    selfplay_games_per_item: int = 50
    selfplay_priority: int = 50

    # Training settings
    training_priority: int = 100
    min_games_for_training: int = 300

    # Tournament settings
    tournament_games: int = 50
    tournament_priority: int = 80

    # Enabled flag
    enabled: bool = True

    # Check interval
    check_interval_seconds: int = 60


class QueuePopulator:
    """
    Maintains minimum work queue depth until Elo targets are met.

    Automatically populates the work queue with selfplay, training, and
    tournament work items to ensure continuous progress toward the
    target Elo rating for each board/player configuration.
    """

    def __init__(
        self,
        config: Optional[PopulatorConfig] = None,
        work_queue: Optional["WorkQueue"] = None,
    ):
        self.config = config or PopulatorConfig()
        self._work_queue = work_queue

        # Track configuration targets
        self._targets: Dict[str, ConfigTarget] = {}
        self._init_targets()

        # Track what we've queued
        self._queued_work_ids: Set[str] = set()
        self._last_populate_time: float = 0

    def _init_targets(self) -> None:
        """Initialize targets for all board/player configurations."""
        for board_type in self.config.board_types:
            for num_players in self.config.player_counts:
                target = ConfigTarget(
                    board_type=board_type,
                    num_players=num_players,
                    target_elo=self.config.target_elo,
                )
                self._targets[target.config_key] = target

    def set_work_queue(self, work_queue: "WorkQueue") -> None:
        """Set the work queue reference."""
        self._work_queue = work_queue

    def update_target_elo(
        self,
        board_type: str,
        num_players: int,
        elo: float,
        model_id: Optional[str] = None,
    ) -> None:
        """Update the current best Elo for a configuration."""
        key = f"{board_type}_{num_players}p"
        if key in self._targets:
            target = self._targets[key]
            if elo > target.current_best_elo:
                target.current_best_elo = elo
                target.best_model_id = model_id
                target.last_updated = time.time()
                logger.info(
                    f"Updated {key} best Elo: {elo:.1f} "
                    f"(gap: {target.elo_gap:.1f}, model: {model_id})"
                )

    def increment_games(self, board_type: str, num_players: int, count: int = 1) -> None:
        """Increment games played for a configuration."""
        key = f"{board_type}_{num_players}p"
        if key in self._targets:
            self._targets[key].games_played += count

    def increment_training(self, board_type: str, num_players: int) -> None:
        """Increment training runs for a configuration."""
        key = f"{board_type}_{num_players}p"
        if key in self._targets:
            self._targets[key].training_runs += 1

    def all_targets_met(self) -> bool:
        """Check if all configurations have reached target Elo."""
        return all(t.target_met for t in self._targets.values())

    def get_unmet_targets(self) -> List[ConfigTarget]:
        """Get configurations that haven't reached target Elo."""
        return [t for t in self._targets.values() if not t.target_met]

    def get_priority_target(self) -> Optional[ConfigTarget]:
        """Get the configuration that needs the most attention.

        Prioritizes by:
        1. Largest Elo gap (furthest from target)
        2. Fewest games played (needs more data)
        """
        unmet = self.get_unmet_targets()
        if not unmet:
            return None

        # Sort by Elo gap (largest first), then by games (fewest first)
        unmet.sort(key=lambda t: (-t.elo_gap, t.games_played))
        return unmet[0]

    def get_current_queue_depth(self) -> int:
        """Get current work queue depth."""
        if self._work_queue is None:
            return 0
        status = self._work_queue.get_queue_status()
        pending = len(status.get("pending", []))
        running = len(status.get("running", []))
        return pending + running

    def calculate_items_needed(self) -> int:
        """Calculate how many items to add to reach minimum depth."""
        current = self.get_current_queue_depth()
        needed = max(0, self.config.min_queue_depth - current)
        return needed

    def _create_selfplay_item(
        self,
        board_type: str,
        num_players: int,
    ) -> Dict[str, Any]:
        """Create a selfplay work item."""
        from app.coordination.work_queue import WorkItem, WorkType

        work_id = f"selfplay_{board_type}_{num_players}p_{int(time.time() * 1000)}"
        return WorkItem(
            work_id=work_id,
            work_type=WorkType.SELFPLAY,
            priority=self.config.selfplay_priority,
            config={
                "board_type": board_type,
                "num_players": num_players,
                "games": self.config.selfplay_games_per_item,
                "source": "queue_populator",
            },
        )

    def _create_training_item(
        self,
        board_type: str,
        num_players: int,
    ) -> Dict[str, Any]:
        """Create a training work item."""
        from app.coordination.work_queue import WorkItem, WorkType

        work_id = f"training_{board_type}_{num_players}p_{int(time.time() * 1000)}"
        return WorkItem(
            work_id=work_id,
            work_type=WorkType.TRAINING,
            priority=self.config.training_priority,
            config={
                "board_type": board_type,
                "num_players": num_players,
                "source": "queue_populator",
            },
        )

    def _create_tournament_item(
        self,
        board_type: str,
        num_players: int,
    ) -> Dict[str, Any]:
        """Create a tournament work item."""
        from app.coordination.work_queue import WorkItem, WorkType

        work_id = f"tournament_{board_type}_{num_players}p_{int(time.time() * 1000)}"
        return WorkItem(
            work_id=work_id,
            work_type=WorkType.TOURNAMENT,
            priority=self.config.tournament_priority,
            config={
                "board_type": board_type,
                "num_players": num_players,
                "games": self.config.tournament_games,
                "source": "queue_populator",
            },
        )

    def populate(self) -> int:
        """Populate the work queue to maintain minimum depth.

        Returns:
            Number of items added
        """
        if not self.config.enabled:
            return 0

        if self._work_queue is None:
            logger.warning("No work queue set, cannot populate")
            return 0

        if self.all_targets_met():
            logger.info("All Elo targets met, no population needed")
            return 0

        items_needed = self.calculate_items_needed()
        if items_needed <= 0:
            return 0

        # Calculate distribution
        selfplay_count = int(items_needed * self.config.selfplay_ratio)
        training_count = int(items_needed * self.config.training_ratio)
        tournament_count = items_needed - selfplay_count - training_count

        # Get unmet targets sorted by priority
        unmet = self.get_unmet_targets()
        if not unmet:
            return 0

        added = 0

        # Add selfplay items
        for i in range(selfplay_count):
            target = unmet[i % len(unmet)]
            try:
                item = self._create_selfplay_item(target.board_type, target.num_players)
                self._work_queue.add_work(item)
                self._queued_work_ids.add(item.work_id)
                added += 1
            except Exception as e:
                logger.error(f"Failed to add selfplay item: {e}")

        # Add training items
        for i in range(training_count):
            target = unmet[i % len(unmet)]
            try:
                item = self._create_training_item(target.board_type, target.num_players)
                self._work_queue.add_work(item)
                self._queued_work_ids.add(item.work_id)
                added += 1
            except Exception as e:
                logger.error(f"Failed to add training item: {e}")

        # Add tournament items
        for i in range(tournament_count):
            target = unmet[i % len(unmet)]
            try:
                item = self._create_tournament_item(target.board_type, target.num_players)
                self._work_queue.add_work(item)
                self._queued_work_ids.add(item.work_id)
                added += 1
            except Exception as e:
                logger.error(f"Failed to add tournament item: {e}")

        self._last_populate_time = time.time()
        logger.info(
            f"Populated queue with {added} items "
            f"(selfplay={selfplay_count}, training={training_count}, tournament={tournament_count})"
        )

        return added

    def get_status(self) -> Dict[str, Any]:
        """Get populator status for monitoring."""
        unmet = self.get_unmet_targets()
        met = [t for t in self._targets.values() if t.target_met]

        return {
            "enabled": self.config.enabled,
            "min_queue_depth": self.config.min_queue_depth,
            "current_queue_depth": self.get_current_queue_depth(),
            "target_elo": self.config.target_elo,
            "total_configs": len(self._targets),
            "configs_met": len(met),
            "configs_unmet": len(unmet),
            "all_targets_met": self.all_targets_met(),
            "unmet_configs": [
                {
                    "config": t.config_key,
                    "current_elo": t.current_best_elo,
                    "gap": t.elo_gap,
                    "games": t.games_played,
                    "training_runs": t.training_runs,
                }
                for t in unmet
            ],
            "last_populate_time": self._last_populate_time,
            "total_queued": len(self._queued_work_ids),
        }


def load_populator_config_from_yaml(yaml_config: Dict[str, Any]) -> PopulatorConfig:
    """Load PopulatorConfig from YAML configuration dict."""
    populator = yaml_config.get("queue_populator", {})

    return PopulatorConfig(
        min_queue_depth=populator.get("min_queue_depth", 50),
        target_elo=populator.get("target_elo", 2000.0),
        selfplay_ratio=populator.get("selfplay_ratio", 0.60),
        training_ratio=populator.get("training_ratio", 0.30),
        tournament_ratio=populator.get("tournament_ratio", 0.10),
        board_types=populator.get("board_types", ["square8", "square19", "hex8", "hexagonal"]),
        player_counts=populator.get("player_counts", [2, 3, 4]),
        selfplay_games_per_item=populator.get("selfplay_games_per_item", 50),
        selfplay_priority=populator.get("selfplay_priority", 50),
        training_priority=populator.get("training_priority", 100),
        min_games_for_training=populator.get("min_games_for_training", 300),
        tournament_games=populator.get("tournament_games", 50),
        tournament_priority=populator.get("tournament_priority", 80),
        enabled=populator.get("enabled", True),
        check_interval_seconds=populator.get("check_interval_seconds", 60),
    )
