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

from app.coordination.contracts import HealthCheckResult
from app.coordination.handler_base import HandlerBase, HandlerStats
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
    min_quality_score: float = 0.2  # Minimum quality score threshold
    deduplicate: bool = True  # Deduplicate samples by game_id
    dedup_threshold: float = 0.98  # Cosine similarity threshold for deduplication

    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("data/training"))
    output_suffix: str = "_combined"  # Appended to config_key for output filename

    # Processing settings
    min_input_files: int = 1  # Minimum NPZ files required to trigger combination
    combine_on_single_file: bool = False  # Combine even with single file (for metadata)

    # Throttling
    min_interval_seconds: float = 60.0  # Minimum seconds between combinations per config

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


class NPZCombinationDaemon(HandlerBase):
    """Daemon that automatically combines NPZ files after exports.

    Subscribes to NPZ_EXPORT_COMPLETE events and triggers quality-weighted
    combination of all available NPZ files for the config.
    """

    _instance: NPZCombinationDaemon | None = None
    _lock: asyncio.Lock = asyncio.Lock()

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

    @classmethod
    async def get_instance(cls) -> NPZCombinationDaemon:
        """Get singleton instance (async-safe)."""
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """Return event type to handler mapping."""
        return {
            "npz_export_complete": self._on_npz_export_complete,
            "npz_combination_complete": self._on_npz_export_complete,  # Also on manual trigger
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

        config_key = event.get("config_key")
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
        except Exception as e:
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
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
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
        else:
            logger.warning(f"Combination failed for {config_key}: {result.error}")

        return result

    def _emit_combination_complete(self, config_key: str, result: CombineResult) -> None:
        """Emit NPZ_COMBINATION_COMPLETE event."""
        try:
            from app.distributed.data_events import DataEventType, emit_data_event

            emit_data_event(
                DataEventType.NPZ_COMBINATION_COMPLETE,
                config_key=config_key,
                output_path=str(result.output_path) if result.output_path else None,
                total_samples=result.total_samples,
                samples_by_source=result.samples_by_source,
                source="NPZCombinationDaemon",
            )
        except Exception as e:
            logger.warning(f"Failed to emit NPZ_COMBINATION_COMPLETE: {e}")

    def _emit_combination_failed(self, config_key: str, error: str) -> None:
        """Emit NPZ_COMBINATION_FAILED event."""
        try:
            from app.distributed.data_events import DataEventType, emit_data_event

            emit_data_event(
                DataEventType.NPZ_COMBINATION_FAILED,
                config_key=config_key,
                error=error,
                source="NPZCombinationDaemon",
            )
        except Exception as e:
            logger.warning(f"Failed to emit NPZ_COMBINATION_FAILED: {e}")

    def health_check(self) -> HealthCheckResult:
        """Return health status for daemon manager."""
        # Check if daemon is running and responsive
        if not self._running:
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


# Singleton accessor
_daemon_instance: NPZCombinationDaemon | None = None
_daemon_lock = asyncio.Lock()


async def get_npz_combination_daemon() -> NPZCombinationDaemon:
    """Get the singleton NPZ combination daemon instance.

    Returns:
        NPZCombinationDaemon singleton instance
    """
    global _daemon_instance
    async with _daemon_lock:
        if _daemon_instance is None:
            _daemon_instance = NPZCombinationDaemon()
        return _daemon_instance


def get_npz_combination_daemon_sync() -> NPZCombinationDaemon:
    """Get the singleton instance (sync version for imports).

    Note: Use get_npz_combination_daemon() in async code.
    """
    global _daemon_instance
    if _daemon_instance is None:
        _daemon_instance = NPZCombinationDaemon()
    return _daemon_instance
