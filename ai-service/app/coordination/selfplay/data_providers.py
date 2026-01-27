"""DataProviderMixin - Data fetching and game count methods for SelfplayScheduler.

January 2026 Sprint 17.9: Extracted from selfplay_scheduler.py (~500 LOC)
to reduce main file size from 3,713 LOC toward ~1,800 LOC target.

This mixin provides:
- Game count normalization (samples per game, games needed)
- Data quality fetching from QualityMonitorDaemon
- Cluster-wide game count aggregation from UnifiedDataRegistry

Usage:
    class SelfplayScheduler(DataProviderMixin, ...):
        pass

The mixin expects the following attributes on the class:
- _config_priorities: dict[str, ConfigPriority]
- _quality_cache: ConfigStateCache
- _cluster_game_counts: dict[str, int]
- _cluster_game_counts_last_update: float
- _cluster_game_counts_ttl: float
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Optional

from app.coordination.budget_calculator import parse_config_key
from app.coordination.priority_calculator import (
    ALL_CONFIGS,
    SAMPLES_PER_GAME_BY_BOARD,
)

if TYPE_CHECKING:
    from app.coordination.selfplay_priority_types import ConfigPriority
    from app.coordination.config_state_cache import ConfigStateCache

logger = logging.getLogger(__name__)


class DataProviderMixin:
    """Mixin providing data fetching and game count methods.

    This mixin extracts data provider responsibilities from SelfplayScheduler:
    - Game count normalization (samples per game estimates)
    - Training sample target management
    - Quality score fetching with TTL caching
    - Cluster-wide game count aggregation

    Attributes expected from main class:
        _config_priorities: Dict mapping config_key to ConfigPriority
        _quality_cache: ConfigStateCache for quality score caching
        _cluster_game_counts: Cached cluster-wide game counts
        _cluster_game_counts_last_update: Last cache refresh timestamp
        _cluster_game_counts_ttl: Cache TTL in seconds
    """

    # Type hints for attributes provided by SelfplayScheduler
    _config_priorities: dict[str, "ConfigPriority"]
    _quality_cache: "ConfigStateCache"
    _cluster_game_counts: dict[str, int]
    _cluster_game_counts_last_update: float
    _cluster_game_counts_ttl: float

    # =========================================================================
    # Game Count Normalization (Dec 29, 2025)
    # =========================================================================

    def _get_samples_per_game_estimate(self, config_key: str) -> float:
        """Get estimated samples per game for a config.

        Uses historical averages from SAMPLES_PER_GAME_BY_BOARD, falling back
        to a conservative default of 50 samples/game if config not found.

        Args:
            config_key: Config key like "hex8_2p", "square19_4p"

        Returns:
            Estimated samples per game
        """
        board_type, num_players = parse_config_key(config_key)
        player_key = f"{num_players}p"
        if board_type in SAMPLES_PER_GAME_BY_BOARD:
            return float(SAMPLES_PER_GAME_BY_BOARD[board_type].get(player_key, 50))
        return 50.0  # Conservative default

    def set_target_training_samples(self, config_key: str, target_samples: int) -> None:
        """Set training sample target for a configuration.

        Dec 29, 2025: Part of game count normalization feature.
        DataPipelineOrchestrator calls this before requesting export to
        communicate how many samples are needed for training.

        This updates the ConfigPriority's target_training_samples and
        samples_per_game_estimate, which are then used by games_needed
        property to determine how many more games should be generated.

        Args:
            config_key: Config key (e.g., "hex8_2p", "square19_4p")
            target_samples: Number of training samples needed

        Example:
            scheduler.set_target_training_samples("hex8_2p", 100000)
            # Now scheduler.get_games_needed("hex8_2p") returns games needed
        """
        from app.coordination.selfplay_priority_types import ConfigPriority

        if config_key not in self._config_priorities:
            logger.warning(
                f"[SelfplayScheduler] Unknown config {config_key}, creating priority entry"
            )
            self._config_priorities[config_key] = ConfigPriority(config_key=config_key)

        priority = self._config_priorities[config_key]
        # Validate target: 0 or negative keeps existing value or uses default
        if target_samples <= 0:
            if priority.target_training_samples <= 0:
                # Use default if no existing value
                priority.target_training_samples = 100000
            # Otherwise keep existing value
            logger.debug(
                f"[SelfplayScheduler] Target {target_samples} <= 0 for {config_key}, "
                f"using existing/default: {priority.target_training_samples}"
            )
        else:
            priority.target_training_samples = target_samples
        priority.samples_per_game_estimate = self._get_samples_per_game_estimate(config_key)

        games_needed = priority.games_needed
        logger.info(
            f"[SelfplayScheduler] Set training target for {config_key}: "
            f"{target_samples} samples, {priority.samples_per_game_estimate:.0f} samples/game estimate, "
            f"{games_needed} games needed"
        )

    def get_games_needed(self, config_key: str) -> int:
        """Get number of additional games needed for a configuration.

        Dec 29, 2025: Part of game count normalization feature.
        Returns how many more games are needed to meet the training sample target.

        Args:
            config_key: Config key (e.g., "hex8_2p", "square19_4p")

        Returns:
            Number of games needed (0 if target already met or config unknown)
        """
        if config_key not in self._config_priorities:
            return 0
        return self._config_priorities[config_key].games_needed

    def get_all_games_needed(self) -> dict[str, int]:
        """Get games needed for all configurations.

        Dec 29, 2025: Part of game count normalization feature.
        Returns a dict mapping config_key to games_needed for all configs.

        Returns:
            Dict[config_key, games_needed] for all tracked configurations
        """
        return {
            cfg: priority.games_needed
            for cfg, priority in self._config_priorities.items()
        }

    # =========================================================================
    # Quality Score Fetching
    # =========================================================================

    def _fetch_quality_from_daemon(self, config: str) -> Optional[float]:
        """Fetch quality score from QualityMonitorDaemon.

        Dec 30, 2025: Extracted as provider for ConfigStateCache.

        Args:
            config: Config key like "hex8_2p"

        Returns:
            Quality score 0.0-1.0, or None if unavailable
        """
        try:
            from app.coordination.quality_monitor_daemon import get_quality_daemon

            daemon = get_quality_daemon()
            if daemon:
                return daemon.get_config_quality(config)
        except ImportError:
            logger.debug("[SelfplayScheduler] quality_monitor_daemon not available")
        except (AttributeError, KeyError) as e:
            logger.debug(f"[SelfplayScheduler] Error getting quality for {config}: {e}")
        return None

    def _get_config_data_quality(self, config: str) -> float:
        """Get data quality score for a config from QualityMonitorDaemon.

        Dec 29, 2025 - Phase 1: Quality-weighted selfplay allocation.
        Higher quality score = better training data (Gumbel MCTS, passed parity).
        Lower quality = heuristic-only games, parity failures.

        Dec 30, 2025: Refactored to use ConfigStateCache for TTL caching.

        Args:
            config: Config key like "hex8_2p"

        Returns:
            Quality score 0.0-1.0 (default 0.7 if unavailable)
        """
        return self._quality_cache.get_quality_or_fetch(config)

    async def _get_all_config_qualities(self) -> dict[str, float]:
        """Get data quality scores for all configs.

        Dec 29, 2025 - Phase 1: Batch quality lookup for priority calculation.
        Dec 30, 2025: Refactored to use ConfigStateCache.

        Returns:
            Dict mapping config_key to quality score (0.0-1.0)
        """
        return self._quality_cache.get_all_qualities(list(ALL_CONFIGS))

    def invalidate_quality_cache(self, config: str | None = None) -> int:
        """Invalidate quality cache for a config or all configs.

        Dec 30, 2025: Refactored to use ConfigStateCache.
        Call this when quality data changes externally
        (e.g., after evaluation completes, after parity gate updates).

        Args:
            config: Specific config to invalidate, or None to clear all

        Returns:
            Number of entries invalidated
        """
        return self._quality_cache.invalidate(config)

    # =========================================================================
    # Cluster-Wide Game Counts
    # =========================================================================

    def _get_cluster_game_counts(self) -> dict[str, int]:
        """Get cluster-wide game counts from UnifiedDataRegistry.

        Jan 2026 Phase 2: Cluster awareness for selfplay scheduling.
        Returns total games available across the cluster (local + remote nodes)
        for each config to prevent duplicate game generation.

        Uses TTL caching to avoid repeated lookups during priority calculation.
        Logs source breakdown for visibility and emits events on fallback.

        Returns:
            Dict mapping config_key to cluster-wide total game count
        """
        now = time.time()

        # Check if cache is fresh
        if now - self._cluster_game_counts_last_update < self._cluster_game_counts_ttl:
            return self._cluster_game_counts

        # Refresh from registry
        try:
            from app.distributed.data_catalog import get_data_registry

            registry = get_data_registry()
            status = registry.get_cluster_status()

            if status and sum(c.get("total", 0) for c in status.values()) > 0:
                # Update cache with total game counts
                self._cluster_game_counts = {
                    config_key: config_data.get("total", 0)
                    for config_key, config_data in status.items()
                }
                self._cluster_game_counts_last_update = now

                # Log source breakdown for visibility (Jan 2026 - cluster-wide measurement)
                local_total = sum(c.get("local", 0) for c in status.values())
                cluster_total = sum(c.get("cluster", 0) for c in status.values())
                owc_total = sum(c.get("owc", 0) for c in status.values())
                total = sum(self._cluster_game_counts.values())

                logger.info(
                    f"[SelfplayScheduler] Using CLUSTER-WIDE counts: "
                    f"local={local_total:,}, cluster={cluster_total:,}, owc={owc_total:,}, "
                    f"total={total:,} games across {len(self._cluster_game_counts)} configs"
                )
                return self._cluster_game_counts

            # Registry returned empty - fall through to fallback
            logger.warning("[SelfplayScheduler] Cluster registry returned empty counts")

        except ImportError as e:
            logger.warning(f"[SelfplayScheduler] DataRegistry not available: {e}")
        except (RuntimeError, ConnectionError, TimeoutError, OSError) as e:
            logger.warning(f"[SelfplayScheduler] Error getting cluster counts: {e}")

        # Fallback to local-only counts with explicit warning
        logger.warning(
            "[SelfplayScheduler] FALLBACK to LOCAL-ONLY game counts! "
            "Cluster data unavailable - progress measurement may be incomplete."
        )

        # Emit event for monitoring (Jan 2026 - cluster visibility)
        try:
            from app.distributed.data_events.event_types import DataEventType
            from app.coordination.event_router import emit_event

            emit_event(
                DataEventType.CLUSTER_VISIBILITY_DEGRADED,
                {
                    "reason": "cluster_manifest_unavailable",
                    "node_id": getattr(self, "_node_id", "unknown"),
                    "cached_count": sum(self._cluster_game_counts.values()),
                },
            )
        except Exception:
            pass  # Don't fail on event emission

        return self._cluster_game_counts

    def _get_cluster_game_count(self, config_key: str) -> int:
        """Get cluster-wide game count for a specific config.

        Jan 2026 Phase 2: Helper for priority calculation.

        Args:
            config_key: Config key like "hex8_2p"

        Returns:
            Total games available cluster-wide for this config (default: 0)
        """
        counts = self._get_cluster_game_counts()
        return counts.get(config_key, 0)

    async def _get_game_counts(self) -> dict[str, int]:
        """Get game counts per config - cluster-aware with fallback to local.

        January 2026: Made cluster-aware. Coordinator nodes don't have local
        canonical databases, so we must use cluster-wide aggregation first.

        Priority order:
        1. UnifiedDataRegistry (cluster manifest + local + OWC + S3)
        2. Local GameDiscovery (last resort fallback)

        Returns:
            Dict mapping config_key to game count
        """
        result: dict[str, int] = {}

        # Try 1: Cluster-wide counts from UnifiedDataRegistry
        try:
            cluster_counts = self._get_cluster_game_counts()
            if cluster_counts and sum(cluster_counts.values()) > 0:
                result = dict(cluster_counts)
                logger.debug(
                    f"[SelfplayScheduler] Using cluster game counts: "
                    f"{sum(result.values()):,} total across {len(result)} configs"
                )
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Cluster counts unavailable: {e}")

        # Try 2: Local-only fallback (for nodes without cluster connectivity)
        if not result or sum(result.values()) == 0:
            try:
                from app.utils.game_discovery import get_game_counts_summary

                result = get_game_counts_summary()
                if sum(result.values()) > 0:
                    logger.debug(
                        f"[SelfplayScheduler] Falling back to local counts: "
                        f"{sum(result.values()):,} total across {len(result)} configs"
                    )
            except ImportError:
                logger.debug("[SelfplayScheduler] game_discovery not available")
            except Exception as e:
                logger.warning(f"[SelfplayScheduler] Error getting local game counts: {e}")

        # Ensure all configs have a count (0 if not found)
        for config_key in ALL_CONFIGS:
            if config_key not in result:
                result[config_key] = 0

        return result
