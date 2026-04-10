"""State-loading helpers for ``UnifiedQueuePopulator``."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from app.coordination.event_utils import make_config_key
from app.coordination.queue_strategies.common import DEFAULT_CURRICULUM_WEIGHTS

logger = logging.getLogger(__name__)


class QueuePopulationStateMixin:
    """Extracted queue population behavior."""

    def _scale_queue_depth_to_cluster(self) -> None:
        """Scale min_queue_depth based on cluster size."""
        try:
            from app.coordination.cluster_status_monitor import ClusterMonitor

            monitor = ClusterMonitor()
            status = monitor.get_cluster_status(
                include_game_counts=False,
                include_training_status=False,
                include_disk_usage=False,
            )
            active_nodes = status.active_nodes

            if active_nodes > 0:
                scaled_depth = max(50, active_nodes * 2)
                old_depth = self.config.min_queue_depth
                self.config.min_queue_depth = scaled_depth
                logger.info(
                    f"[QueuePopulator] Scaled queue depth: {old_depth} -> {scaled_depth} "
                    f"(for {active_nodes} active nodes)"
                )

        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"[QueuePopulator] Failed to scale queue depth: {e}")
    def _init_targets(self) -> None:
        """Initialize targets for all board/player configurations."""
        from app.coordination.unified_queue_populator import ConfigTarget

        for board_type in self.config.board_types:
            for num_players in self.config.player_counts:
                target = ConfigTarget(
                    board_type=board_type,
                    num_players=num_players,
                    target_elo=self.config.target_elo,
                )
                self._targets[target.config_key] = target
    def _load_existing_elo(self) -> None:
        """Load existing Elo ratings from the database."""
        if self._elo_db_path:
            db_path = Path(self._elo_db_path)
        else:
            candidates = [
                Path(__file__).parent.parent.parent / "data" / "unified_elo.db",
                Path("/lambda/nfs/RingRift/elo/unified_elo.db"),
                Path.home() / "ringrift" / "ai-service" / "data" / "unified_elo.db",
            ]
            db_path = None
            for candidate in candidates:
                if candidate.exists():
                    db_path = candidate
                    break

        if not db_path or not db_path.exists():
            logger.info("No Elo database found, starting with default 1500 Elo")
            return

        try:
            # December 27, 2025: Use context manager to prevent connection leaks
            with sqlite3.connect(str(db_path), timeout=10.0) as conn:
                cursor = conn.cursor()

                # Exclude baseline participants (random, heuristic) and
                # heuristic-harness variants of real models which have inflated Elo
                # from playing as heuristic against weak baselines.
                # Real model participants use gumbel_mcts or similar neural harnesses.
                cursor.execute("""
                    SELECT e.board_type, e.num_players, e.rating as best_elo,
                           e.participant_id, e.games_played
                    FROM elo_ratings e
                    INNER JOIN (
                        SELECT board_type, num_players, MAX(rating) as max_rating
                        FROM elo_ratings
                        WHERE archived_at IS NULL
                          AND participant_id NOT LIKE '%:heuristic:%'
                          AND participant_id NOT IN ('random', 'heuristic')
                          AND participant_id NOT LIKE 'none:%'
                          AND participant_id NOT LIKE 'baseline_%'
                        GROUP BY board_type, num_players
                    ) m ON e.board_type = m.board_type
                       AND e.num_players = m.num_players
                       AND e.rating = m.max_rating
                    WHERE e.archived_at IS NULL
                """)

                rows = cursor.fetchall()

            for row in rows:
                board_type, num_players, best_elo, model_id, games = row
                key = make_config_key(board_type, num_players)
                if key in self._targets:
                    target = self._targets[key]
                    target.current_best_elo = best_elo
                    target.best_model_id = model_id
                    target.games_played = games or 0
                    target.record_elo(best_elo)
                    logger.info(
                        f"Loaded existing Elo for {key}: {best_elo:.1f} "
                        f"(model: {model_id}, games: {games})"
                    )

            met = sum(1 for t in self._targets.values() if t.target_met)
            logger.info(
                f"Loaded Elo data: {met}/{len(self._targets)} configs at target "
                f"({self.config.target_elo}+ Elo)"
            )

        except Exception as e:
            logger.warning(f"Failed to load existing Elo data: {e}")
    def _load_curriculum_weights(self) -> None:
        """Load curriculum weights for prioritization."""
        try:
            from app.coordination.curriculum_weights import load_curriculum_weights

            weights = load_curriculum_weights()
            for config_key, weight in weights.items():
                if config_key in self._targets:
                    self._targets[config_key].curriculum_weight = weight
            logger.info("[QueuePopulator] Loaded curriculum weights")
        except ImportError:
            for config_key, weight in DEFAULT_CURRICULUM_WEIGHTS.items():
                if config_key in self._targets:
                    self._targets[config_key].curriculum_weight = weight
            logger.debug("[QueuePopulator] Using default curriculum weights")
    def _find_canonical_model(self, board_type: str, num_players: int) -> str | None:
        """Find a canonical model for the given configuration.

        Jan 13, 2026: Added to fix the 98% heuristic-only selfplay bug.
        When no model has an Elo rating yet, this finds a canonical model
        to enable neural network modes (gumbel-mcts, policy-only, etc.).

        Args:
            board_type: Board type (e.g., "hex8", "square8")
            num_players: Number of players (2, 3, or 4)

        Returns:
            Path to canonical model if found, None otherwise.
        """
        from pathlib import Path

        # Canonical model naming patterns to try (in priority order)
        patterns = [
            f"canonical_{board_type}_{num_players}p_v2.pth",  # Latest v2 architecture
            f"canonical_{board_type}_{num_players}p.pth",     # Standard canonical
            f"canonical_{board_type}_{num_players}p_new.pth", # New training run
        ]

        models_dir = Path("models")
        if not models_dir.exists():
            # Try ai-service relative path
            models_dir = Path("ai-service/models")

        for pattern in patterns:
            model_path = models_dir / pattern
            if model_path.exists():
                return str(model_path)

        # Try glob for any matching canonical model
        try:
            matches = list(models_dir.glob(f"canonical_{board_type}_{num_players}p*.pth"))
            if matches:
                # Return most recently modified
                matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return str(matches[0])
        except (OSError, ValueError):
            pass

        return None
    def _load_game_counts(self) -> None:
        """Load actual game counts from canonical databases.

        Jan 5, 2026 (Session 17.34): Fix for starvation boost not being applied.
        Previously, games_played was loaded from elo_ratings.games_played which
        tracks evaluation games, not actual selfplay games. This caused all configs
        to show 0 games and the starvation multiplier to never be applied.

        Now loads from canonical databases via get_game_counts_summary().
        """
        try:
            from app.utils.game_discovery import get_game_counts_summary

            counts = get_game_counts_summary()
            total_games = 0
            for config_key, count in counts.items():
                if config_key in self._targets:
                    self._targets[config_key].games_played = count
                    total_games += count

            logger.info(
                f"[QueuePopulator] Loaded game counts from canonical DBs: "
                f"{total_games:,} total across {len(counts)} configs"
            )

            # Log configs with low game counts for visibility
            low_game_configs = [
                (k, v) for k, v in counts.items()
                if v < 500 and k in self._targets
            ]
            if low_game_configs:
                low_game_configs.sort(key=lambda x: x[1])
                logger.warning(
                    f"[QueuePopulator] Low game count configs (starvation candidates): "
                    f"{', '.join(f'{k}:{v}' for k, v in low_game_configs[:5])}"
                )

        except ImportError:
            logger.warning("[QueuePopulator] game_discovery not available, using elo_ratings game counts")
        except Exception as e:
            logger.warning(f"[QueuePopulator] Failed to load game counts from canonical DBs: {e}")
    def ensure_game_counts_loaded(self) -> None:
        """Load game counts if not already loaded. Safe to call multiple times.

        Mar 2026: Deferred from __init__ to avoid blocking the event loop.
        Called lazily before first populate().
        """
        if not self._game_counts_loaded:
            self._load_game_counts()
            self._game_counts_loaded = True
