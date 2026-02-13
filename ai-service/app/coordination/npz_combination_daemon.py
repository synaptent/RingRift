"""NPZ Combination Daemon - Automated quality-weighted NPZ combination.

This daemon automatically combines NPZ training files after export,
creating a single combined file that includes quality-weighted samples
from both historical and fresh data.

Features:
- Subscribes to NPZ_EXPORT_COMPLETE events to trigger combination
- Uses quality-aware weighting from npz_combiner.py
- Emits NPZ_COMBINATION_COMPLETE event when done
- Provides health check reporting for daemon manager

Usage:
    from app.coordination.npz_combination_daemon import (
        NPZCombinationDaemon,
        get_npz_combination_daemon,
    )

    # Get singleton daemon
    daemon = get_npz_combination_daemon()
    await daemon.start()

December 2025 - Automated NPZ combination for training pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config.thresholds import NPZ_COMBINATION_MIN_QUALITY
from app.coordination.contracts import HealthCheckResult
from app.coordination.event_handler_utils import extract_config_key
from app.coordination.event_emission_helpers import safe_emit_event
from app.coordination.event_router import get_event_payload
from app.coordination.handler_base import HandlerBase, HandlerStats
from app.coordination.singleton_mixin import SingletonMixin
from app.distributed.data_events import DataEventType
from app.training.npz_combiner import (
    CombineResult,
    NPZCombinerConfig,
    discover_and_combine_for_config,
)
from app.utils.canonical_naming import parse_config_key

logger = logging.getLogger(__name__)

__all__ = [
    "NPZCombinationConfig",
    "NPZCombinationDaemon",
    "get_npz_combination_daemon",
]


@dataclass
class NPZCombinationConfig:
    """Configuration for NPZ combination daemon."""

    # Combination settings
    freshness_weight: float = 1.5  # Weight for fresh data (1.0 = no weighting)
    freshness_half_life_hours: float = 24.0  # Half-life for freshness decay
    min_quality_score: float = NPZ_COMBINATION_MIN_QUALITY  # From thresholds.py
    deduplicate: bool = True  # Deduplicate samples by game_id
    dedup_threshold: float = 0.98  # Cosine similarity threshold for deduplication

    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("data/training"))
    output_suffix: str = "_combined"  # Appended to config_key for output filename

    # Processing settings
    min_input_files: int = 1  # Minimum NPZ files required to trigger combination
    combine_on_single_file: bool = False  # Combine even with single file (for metadata)

    # Throttling
    # January 3, 2026: Reduced from 60s to 30s for faster training iteration cycles.
    # January 4, 2026 (Session 17.11): Reduced from 30s to 5s for +5-8 Elo improvement.
    # Analysis showed 30s added unnecessary latency between export and training.
    # January 5, 2026 (Session 17.24): Reduced from 5s to 3s for +1-2 Elo improvement.
    # January 5, 2026 (Session 17.26): Reduced from 3s to 1.5s for marginal improvement.
    # January 5, 2026 (Session 17.28): Reduced from 1.5s to 0.5s for +2-4 Elo improvement.
    # 0.5s is minimum safe interval that prevents thundering herd on concurrent exports.
    min_interval_seconds: float = 0.5  # Minimum seconds between combinations per config

    def to_combiner_config(self) -> NPZCombinerConfig:
        """Convert to NPZCombinerConfig."""
        return NPZCombinerConfig(
            freshness_weight=self.freshness_weight,
            freshness_half_life_hours=self.freshness_half_life_hours,
            min_quality_score=self.min_quality_score,
            deduplicate=self.deduplicate,
        )


@dataclass
class CombinationStats:
    """Statistics for NPZ combination daemon."""

    # Combination tracking
    combinations_triggered: int = 0
    combinations_succeeded: int = 0
    combinations_failed: int = 0
    combinations_skipped: int = 0  # Throttled or insufficient files

    # Sample tracking
    total_samples_combined: int = 0
    samples_deduplicated: int = 0

    # Timing
    last_combination_time: float = 0.0
    last_combination_config: str = ""

    # Per-config tracking
    last_combination_by_config: dict[str, float] = field(default_factory=dict)


class NPZCombinationDaemon(SingletonMixin, HandlerBase):
    """Daemon that automatically combines NPZ files after exports.

    Subscribes to NPZ_EXPORT_COMPLETE events and triggers quality-weighted
    combination of all available NPZ files for the config.

    January 2026: Migrated to use SingletonMixin for consistency.
    """

    def __init__(self, config: NPZCombinationConfig | None = None) -> None:
        """Initialize NPZ combination daemon.

        Args:
            config: Daemon configuration. If None, uses defaults.
        """
        super().__init__(
            name="NPZCombinationDaemon",
            cycle_interval=300.0,  # 5 minute cycle for periodic checks
        )
        self.config = config or NPZCombinationConfig()
        self.combination_stats = CombinationStats()
        self._last_combination_results: dict[str, CombineResult] = {}

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """Return event type to handler mapping."""
        return {
            DataEventType.NPZ_EXPORT_COMPLETE.value: self._on_npz_export_complete,
            DataEventType.NPZ_COMBINATION_COMPLETE.value: self._on_npz_export_complete,  # Also on manual trigger
        }

    async def _run_cycle(self) -> None:
        """Periodic check for NPZ files that may need combination.

        This runs every 5 minutes to catch any files that weren't combined
        due to missed events.
        """
        # In cycle mode, we don't proactively combine - we wait for events
        # This is just a health check / stats update cycle
        self.stats.cycles_completed += 1
        self.stats.last_activity = time.time()

    async def _on_npz_export_complete(self, event: dict[str, Any]) -> None:
        """Handle NPZ export completion event.

        Triggers combination for the config that just completed export.
        """
        # Deduplication check
        if self._is_duplicate_event(event):
            logger.debug("Skipping duplicate NPZ_EXPORT_COMPLETE event")
            return

        config_key = extract_config_key(get_event_payload(event))
        if not config_key:
            logger.warning("NPZ_EXPORT_COMPLETE event missing config_key")
            return

        # Check throttling
        last_time = self.combination_stats.last_combination_by_config.get(config_key, 0.0)
        if time.time() - last_time < self.config.min_interval_seconds:
            logger.debug(
                f"Skipping combination for {config_key} - throttled "
                f"(last: {time.time() - last_time:.1f}s ago)"
            )
            self.combination_stats.combinations_skipped += 1
            return

        # Trigger combination
        self.combination_stats.combinations_triggered += 1
        logger.info(f"Triggering NPZ combination for {config_key}")

        # Emit start event (Dec 28, 2025 - was previously orphan event)
        self._emit_combination_started(config_key)

        try:
            result = await self._combine_for_config(config_key)
            if result and result.success:
                self.combination_stats.combinations_succeeded += 1
                self.combination_stats.total_samples_combined += result.total_samples
                self._emit_combination_complete(config_key, result)
            else:
                self.combination_stats.combinations_failed += 1
                error = result.error if result else "Unknown error"
                self._emit_combination_failed(config_key, error)
        except (OSError, IOError, ValueError, MemoryError) as e:
            # OSError/IOError: file access, ValueError: data format, MemoryError: large arrays
            logger.exception(f"Error combining NPZ for {config_key}: {e}")
            self.combination_stats.combinations_failed += 1
            self._emit_combination_failed(config_key, str(e))

    async def _combine_for_config(self, config_key: str) -> CombineResult | None:
        """Combine NPZ files for a specific config.

        Args:
            config_key: Config key (e.g., 'hex8_2p')

        Returns:
            CombineResult or None if combination not performed
        """
        try:
            # Parse config key to validate
            board_type, num_players = parse_config_key(config_key)
        except ValueError as e:
            logger.error(f"Invalid config_key {config_key}: {e}")
            return CombineResult(success=False, error=str(e))

        # Determine output path
        output_path = (
            self.config.output_dir / f"{config_key}{self.config.output_suffix}.npz"
        )

        # Run combination (this is sync but may take a while)
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: discover_and_combine_for_config(
                config_key=config_key,
                output_path=output_path,
                combiner_config=self.config.to_combiner_config(),
            ),
        )

        # Update tracking
        self.combination_stats.last_combination_time = time.time()
        self.combination_stats.last_combination_config = config_key
        self.combination_stats.last_combination_by_config[config_key] = time.time()
        self._last_combination_results[config_key] = result

        if result.success:
            logger.info(
                f"Combined {result.total_samples} samples for {config_key} -> {output_path}"
            )
            # January 2026 Sprint 10: Verify quality_score array after combination
            # Expected improvement: +2-5 Elo from catching corrupt NPZ files early
            # Session 17.11: Run as fire-and-forget to reduce latency (+2-4 Elo)
            # Previously blocked event emission; now emit immediately, verify async
            self._safe_create_task(
                asyncio.to_thread(
                    self._verify_npz_quality, output_path, config_key, result
                ),
                context=f"npz_quality_verify_{config_key}",
            )
        else:
            logger.warning(f"Combination failed for {config_key}: {result.error}")

        return result

    def _emit_combination_complete(self, config_key: str, result: CombineResult) -> None:
        """Emit NPZ_COMBINATION_COMPLETE event."""
        safe_emit_event(
            "npz_combination_complete",
            {
                "config_key": config_key,
                "output_path": str(result.output_path) if result.output_path else None,
                "total_samples": result.total_samples,
                "samples_by_source": result.samples_by_source,
                "source": "NPZCombinationDaemon",
            },
            context="NPZCombination",
        )

    def _emit_combination_failed(self, config_key: str, error: str) -> None:
        """Emit NPZ_COMBINATION_FAILED event."""
        safe_emit_event(
            "npz_combination_failed",
            {
                "config_key": config_key,
                "error": error,
                "source": "NPZCombinationDaemon",
            },
            context="NPZCombination",
        )

    def _emit_combination_started(self, config_key: str) -> None:
        """Emit NPZ_COMBINATION_STARTED event.

        Dec 28, 2025 - Added to fix orphan event (was defined but never emitted).
        """
        safe_emit_event(
            "npz_combination_started",
            {
                "config_key": config_key,
                "source": "NPZCombinationDaemon",
            },
            context="NPZCombination",
        )

    def _verify_npz_quality(
        self, output_path: Path, config_key: str, result: CombineResult
    ) -> None:
        """Verify quality_score array in combined NPZ file.

        January 2026 Sprint 10: NPZ quality verification after combination.
        Expected improvement: +2-5 Elo from catching corrupt NPZ files early.

        Checks:
        1. quality_score array exists
        2. No NaN values in quality_score
        3. quality_score values are in valid range [0, 1]
        4. quality_score length matches sample count
        """
        try:
            import numpy as np

            if not output_path.exists():
                logger.warning(
                    f"[NPZCombinationDaemon] Combined NPZ not found: {output_path}"
                )
                return

            with np.load(output_path, allow_pickle=False) as data:
                # Check if quality_score exists
                if "quality_score" not in data:
                    logger.warning(
                        f"[NPZCombinationDaemon] No quality_score in {config_key} NPZ"
                    )
                    return

                quality_scores = data["quality_score"]
                total_samples = result.total_samples

                # Check for NaN values
                nan_count = np.isnan(quality_scores).sum()
                if nan_count > 0:
                    logger.error(
                        f"[NPZCombinationDaemon] {config_key} has {nan_count} NaN quality scores"
                    )

                # Check value range [0, 1]
                out_of_range = np.sum((quality_scores < 0) | (quality_scores > 1))
                if out_of_range > 0:
                    logger.error(
                        f"[NPZCombinationDaemon] {config_key} has {out_of_range} quality scores out of [0,1] range"
                    )

                # Check length matches
                if len(quality_scores) != total_samples:
                    logger.error(
                        f"[NPZCombinationDaemon] {config_key} quality_score length mismatch: "
                        f"{len(quality_scores)} vs {total_samples} samples"
                    )

                # Log quality statistics
                avg_quality = float(np.mean(quality_scores))
                min_quality = float(np.min(quality_scores))
                max_quality = float(np.max(quality_scores))
                logger.info(
                    f"[NPZCombinationDaemon] {config_key} quality stats: "
                    f"avg={avg_quality:.3f}, min={min_quality:.3f}, max={max_quality:.3f}"
                )

        except Exception as e:
            logger.warning(f"[NPZCombinationDaemon] Quality verification failed: {e}")

    def health_check(self) -> HealthCheckResult:
        """Return health status for daemon manager."""
        # Check if daemon is running and responsive
        if not self.is_running:
            return HealthCheckResult(
                healthy=False,
                status="stopped",
                message="Daemon is not running",
                details={},
            )

        # Calculate health based on error rate
        total = (
            self.combination_stats.combinations_succeeded
            + self.combination_stats.combinations_failed
        )
        error_rate = (
            self.combination_stats.combinations_failed / max(total, 1)
        )

        healthy = error_rate < 0.5  # Healthy if <50% error rate

        return HealthCheckResult(
            healthy=healthy,
            status="healthy" if healthy else "degraded",
            message=(
                f"Combinations: {self.combination_stats.combinations_succeeded} succeeded, "
                f"{self.combination_stats.combinations_failed} failed"
            ),
            details={
                "combinations_triggered": self.combination_stats.combinations_triggered,
                "combinations_succeeded": self.combination_stats.combinations_succeeded,
                "combinations_failed": self.combination_stats.combinations_failed,
                "combinations_skipped": self.combination_stats.combinations_skipped,
                "total_samples_combined": self.combination_stats.total_samples_combined,
                "last_combination_config": self.combination_stats.last_combination_config,
                "error_rate": error_rate,
                "uptime_seconds": time.time() - self.stats.started_at if self.stats.started_at else 0,
            },
        )

    def get_last_result(self, config_key: str) -> CombineResult | None:
        """Get the last combination result for a config."""
        return self._last_combination_results.get(config_key)

    def get_combination_stats(self) -> dict[str, Any]:
        """Get combination statistics."""
        return {
            "triggered": self.combination_stats.combinations_triggered,
            "succeeded": self.combination_stats.combinations_succeeded,
            "failed": self.combination_stats.combinations_failed,
            "skipped": self.combination_stats.combinations_skipped,
            "total_samples": self.combination_stats.total_samples_combined,
            "last_config": self.combination_stats.last_combination_config,
            "last_time": self.combination_stats.last_combination_time,
        }


def get_npz_combination_daemon() -> NPZCombinationDaemon:
    """Get the singleton NPZ combination daemon instance.

    January 2026: Now uses SingletonMixin.get_instance() (thread-safe, sync).
    The async version was unnecessary since instance creation is fast.

    Returns:
        NPZCombinationDaemon singleton instance
    """
    return NPZCombinationDaemon.get_instance()


# Backward compatibility alias
get_npz_combination_daemon_sync = get_npz_combination_daemon
