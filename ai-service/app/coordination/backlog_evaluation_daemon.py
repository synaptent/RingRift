"""BacklogEvaluationDaemon - Queue OWC models for Elo evaluation.

Sprint 15 (January 3, 2026): Part of OWC Model Evaluation Automation.

This daemon periodically scans the OWC drive for model files that haven't been
evaluated for Elo ratings, and queues them for evaluation by EvaluationDaemon.

Key features:
- Discovers models on OWC via OWCModelDiscovery
- Prioritizes models by: config staleness, naming (canonical/best), recency, source
- Rate-limited queuing (configurable batch size and hourly limits)
- Respects backpressure from EvaluationDaemon
- Tracks evaluation status in unified_elo.db

The daemon emits synthetic TRAINING_COMPLETED events with source="backlog_*" to
trigger evaluation through the existing pipeline.

Environment Variables:
    RINGRIFT_BACKLOG_EVAL_ENABLED: Enable/disable daemon (default: true)
    RINGRIFT_BACKLOG_SCAN_INTERVAL: Seconds between scans (default: 900)
    RINGRIFT_BACKLOG_BATCH_SIZE: Models per cycle (default: 3)
    RINGRIFT_BACKLOG_MAX_HOURLY: Max models/hour (default: 10)
    RINGRIFT_BACKLOG_PAUSE_DEPTH: Pause when queue exceeds (default: 50)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.coordination.contracts import HealthCheckResult
from app.coordination.handler_base import HandlerBase

if TYPE_CHECKING:
    from app.models.owc_discovery import DiscoveredModel, OWCModelDiscovery
    from app.training.evaluation_status import EvaluationStatusTracker

logger = logging.getLogger(__name__)

__all__ = [
    "BacklogEvaluationDaemon",
    "BacklogEvalConfig",
    "BacklogEvalStats",
    "get_backlog_evaluation_daemon",
    "reset_backlog_evaluation_daemon",
]


# ============================================================================
# Priority Constants
# ============================================================================

# Priority scoring: lower = higher priority (0-200 range)
PRIORITY_CANONICAL = -50  # Canonical models get highest priority
PRIORITY_BEST = -30  # "best" models
PRIORITY_RECENT = -10  # Models from last 7 days
PRIORITY_LOCAL = -10  # Local models (faster access)
PRIORITY_UNDERSERVED = -50  # Underserved configs
PRIORITY_BASE = 100  # Base priority


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class BacklogEvalConfig:
    """Configuration for backlog evaluation daemon."""

    # Enable/disable
    enabled: bool = True

    # Scan interval (15 minutes default)
    scan_interval_seconds: int = 900

    # Rate limiting
    batch_size: int = 3  # Models per cycle
    max_hourly: int = 10  # Max models per hour
    pause_queue_depth: int = 50  # Pause when eval queue exceeds this

    # Priority weighting
    canonical_priority_boost: int = PRIORITY_CANONICAL
    best_priority_boost: int = PRIORITY_BEST
    recent_priority_boost: int = PRIORITY_RECENT
    local_priority_boost: int = PRIORITY_LOCAL
    underserved_priority_boost: int = PRIORITY_UNDERSERVED

    # Staleness detection
    stale_evaluation_days: int = 7  # Re-evaluate after this many days

    # Underserved config thresholds
    underserved_game_threshold: int = 5000  # Configs with fewer games are underserved

    @classmethod
    def from_env(cls) -> "BacklogEvalConfig":
        """Load configuration from environment."""
        return cls(
            enabled=os.getenv("RINGRIFT_BACKLOG_EVAL_ENABLED", "true").lower() == "true",
            scan_interval_seconds=int(os.getenv("RINGRIFT_BACKLOG_SCAN_INTERVAL", "900")),
            batch_size=int(os.getenv("RINGRIFT_BACKLOG_BATCH_SIZE", "3")),
            max_hourly=int(os.getenv("RINGRIFT_BACKLOG_MAX_HOURLY", "10")),
            pause_queue_depth=int(os.getenv("RINGRIFT_BACKLOG_PAUSE_DEPTH", "50")),
        )


@dataclass
class BacklogEvalStats:
    """Statistics for backlog evaluation operations."""

    # Discovery
    discovery_cycles: int = 0
    models_discovered: int = 0
    models_registered: int = 0

    # Queueing
    models_queued: int = 0
    models_skipped_backpressure: int = 0
    models_skipped_rate_limit: int = 0

    # Evaluation tracking
    evaluations_started: int = 0
    evaluations_completed: int = 0
    evaluations_failed: int = 0

    # Rate limiting
    hourly_queued: int = 0
    hourly_window_start: float = field(default_factory=time.time)

    # Timing
    last_discovery_time: float = 0.0
    last_queue_time: float = 0.0


# ============================================================================
# BacklogEvaluationDaemon
# ============================================================================


class BacklogEvaluationDaemon(HandlerBase):
    """Daemon that queues OWC models for Elo evaluation.

    This daemon periodically scans the OWC drive for model files that haven't
    been evaluated, prioritizes them, and emits events to trigger evaluation.

    The daemon respects backpressure from EvaluationDaemon and rate-limits
    its own queuing to avoid overwhelming the evaluation infrastructure.
    """

    _instance: "BacklogEvaluationDaemon | None" = None

    def __init__(self, config: BacklogEvalConfig | None = None):
        self._daemon_config = config or BacklogEvalConfig.from_env()

        super().__init__(
            name="backlog_evaluation",
            cycle_interval=float(self._daemon_config.scan_interval_seconds),
        )

        self._stats = BacklogEvalStats()
        self._backpressure_active = False

        # Lazy-loaded components
        self._discovery: "OWCModelDiscovery | None" = None
        self._tracker: "EvaluationStatusTracker | None" = None

        # Cache for underserved configs
        self._underserved_configs: set[str] = set()
        self._underserved_cache_time: float = 0.0

    @classmethod
    def get_instance(
        cls, config: BacklogEvalConfig | None = None
    ) -> "BacklogEvaluationDaemon":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        if cls._instance is not None:
            asyncio.create_task(cls._instance.stop())
        cls._instance = None

    @property
    def config(self) -> BacklogEvalConfig:
        """Get daemon configuration."""
        return self._daemon_config

    @property
    def stats(self) -> BacklogEvalStats:
        """Get daemon statistics."""
        return self._stats

    # =========================================================================
    # Event Subscriptions
    # =========================================================================

    def _get_event_subscriptions(self) -> dict:
        """Define event subscriptions for backpressure handling."""
        return {
            "evaluation_backpressure": self._on_backpressure,
            "evaluation_backpressure_released": self._on_backpressure_released,
            "evaluation_completed": self._on_evaluation_completed,
            "evaluation_failed": self._on_evaluation_failed,
        }

    async def _on_backpressure(self, event: dict) -> None:
        """Handle backpressure signal from EvaluationDaemon."""
        if self._is_duplicate_event(event):
            return

        self._backpressure_active = True
        queue_depth = event.get("queue_depth", 0)
        logger.info(
            f"[BacklogEval] Backpressure active (queue_depth={queue_depth}), "
            f"pausing model queuing"
        )

    async def _on_backpressure_released(self, event: dict) -> None:
        """Handle backpressure release from EvaluationDaemon."""
        if self._is_duplicate_event(event):
            return

        self._backpressure_active = False
        logger.info("[BacklogEval] Backpressure released, resuming model queuing")

    async def _on_evaluation_completed(self, event: dict) -> None:
        """Track completed evaluations from backlog."""
        source = event.get("source", "")
        if source.startswith("backlog_"):
            self._stats.evaluations_completed += 1
            logger.debug(
                f"[BacklogEval] Evaluation completed: {event.get('model_path', 'unknown')}"
            )

    async def _on_evaluation_failed(self, event: dict) -> None:
        """Track failed evaluations from backlog."""
        source = event.get("source", "")
        if source.startswith("backlog_"):
            self._stats.evaluations_failed += 1
            logger.warning(
                f"[BacklogEval] Evaluation failed: {event.get('model_path', 'unknown')}"
            )

    # =========================================================================
    # Component Access
    # =========================================================================

    def _get_discovery(self) -> "OWCModelDiscovery":
        """Lazy-load OWCModelDiscovery."""
        if self._discovery is None:
            from app.models.owc_discovery import OWCModelDiscovery

            self._discovery = OWCModelDiscovery.get_instance()
        return self._discovery

    def _get_tracker(self) -> "EvaluationStatusTracker":
        """Lazy-load EvaluationStatusTracker."""
        if self._tracker is None:
            from app.training.evaluation_status import EvaluationStatusTracker

            self._tracker = EvaluationStatusTracker.get_instance()
        return self._tracker

    # =========================================================================
    # Main Cycle
    # =========================================================================

    async def _run_cycle(self) -> None:
        """Main cycle: discover, prioritize, and queue models for evaluation."""
        if not self._daemon_config.enabled:
            return

        # Check backpressure
        if self._backpressure_active:
            logger.debug("[BacklogEval] Skipping cycle due to backpressure")
            self._stats.models_skipped_backpressure += 1
            return

        # Check hourly rate limit
        if not self._check_hourly_rate_limit():
            logger.debug("[BacklogEval] Skipping cycle due to hourly rate limit")
            self._stats.models_skipped_rate_limit += 1
            return

        cycle_start = time.time()
        self._stats.discovery_cycles += 1

        try:
            # Discover models
            discovery = self._get_discovery()

            if not await discovery.check_available():
                logger.warning("[BacklogEval] OWC drive not available, skipping cycle")
                return

            models = await discovery.discover_all_models(force_refresh=True)
            self._stats.models_discovered = len(models)
            self._stats.last_discovery_time = time.time()

            if not models:
                logger.debug("[BacklogEval] No models found on OWC")
                self._emit_discovery_completed(0, 0)
                return

            # Register with tracker
            registered = await discovery.register_with_tracker(models)
            self._stats.models_registered += registered

            # Get unevaluated models
            unevaluated = await self._get_unevaluated_models()

            if not unevaluated:
                logger.info(
                    f"[BacklogEval] All {len(models)} models are evaluated, nothing to queue"
                )
                self._emit_discovery_completed(len(models), 0)
                return

            # Prioritize models
            prioritized = self._prioritize_models(unevaluated)

            # Queue top N for evaluation (respecting batch size)
            batch = prioritized[: self._daemon_config.batch_size]
            queued = 0

            for model in batch:
                if self._check_hourly_rate_limit():
                    if await self._queue_for_evaluation(model):
                        queued += 1
                        self._stats.models_queued += 1
                        self._stats.hourly_queued += 1
                else:
                    break

            self._stats.last_queue_time = time.time()

            logger.info(
                f"[BacklogEval] Cycle complete: discovered={len(models)}, "
                f"unevaluated={len(unevaluated)}, queued={queued}"
            )

            self._emit_discovery_completed(len(models), queued)

        except Exception as e:
            logger.error(f"[BacklogEval] Cycle failed: {e}", exc_info=True)
            self._record_error(f"Cycle failed: {e}", e)

        finally:
            cycle_duration = time.time() - cycle_start
            logger.debug(f"[BacklogEval] Cycle took {cycle_duration:.1f}s")

    # =========================================================================
    # Priority Scoring
    # =========================================================================

    def _prioritize_models(
        self, models: list["DiscoveredModel"]
    ) -> list["DiscoveredModel"]:
        """Prioritize models for evaluation.

        Lower priority score = higher priority.

        Priority factors:
        1. Config staleness (-50 for underserved configs)
        2. Model naming (-50 for "canonical", -30 for "best")
        3. Recency (-10 for <7 days old)
        4. Source (-10 for local vs OWC)

        Args:
            models: List of unevaluated models

        Returns:
            Models sorted by priority (lowest score first)
        """
        self._refresh_underserved_configs()

        scored_models: list[tuple[int, "DiscoveredModel"]] = []

        for model in models:
            score = PRIORITY_BASE

            # Canonical models get highest priority
            if model.is_canonical:
                score += self._daemon_config.canonical_priority_boost

            # "best" models
            if model.is_best:
                score += self._daemon_config.best_priority_boost

            # Recent models (within 7 days)
            if model.modified_at:
                age_days = (time.time() - model.modified_at) / 86400
                if age_days < 7:
                    score += self._daemon_config.recent_priority_boost

            # Local models (faster to evaluate)
            if model.source == "local":
                score += self._daemon_config.local_priority_boost

            # Underserved configs
            if model.config_key and model.config_key in self._underserved_configs:
                score += self._daemon_config.underserved_priority_boost

            scored_models.append((score, model))

        # Sort by score (ascending = highest priority first)
        scored_models.sort(key=lambda x: x[0])

        return [model for _, model in scored_models]

    def _refresh_underserved_configs(self) -> None:
        """Refresh cache of underserved configurations.

        Caches for 1 hour to avoid repeated database queries.
        """
        if time.time() - self._underserved_cache_time < 3600:
            return

        try:
            from app.training.elo_service import get_elo_service

            elo_service = get_elo_service()
            self._underserved_configs = set()

            # Check each config's game count
            for board_type in ["hex8", "hexagonal", "square8", "square19"]:
                for num_players in [2, 3, 4]:
                    config_key = f"{board_type}_{num_players}p"
                    try:
                        stats = elo_service.get_config_stats(config_key)
                        if stats and stats.total_games < self._daemon_config.underserved_game_threshold:
                            self._underserved_configs.add(config_key)
                    except Exception:
                        # If we can't get stats, assume underserved
                        self._underserved_configs.add(config_key)

            self._underserved_cache_time = time.time()

        except ImportError:
            pass

    # =========================================================================
    # Rate Limiting
    # =========================================================================

    def _check_hourly_rate_limit(self) -> bool:
        """Check if we're within the hourly rate limit.

        Returns:
            True if we can queue more models, False if at limit
        """
        # Reset hourly counter if window expired
        if time.time() - self._stats.hourly_window_start > 3600:
            self._stats.hourly_queued = 0
            self._stats.hourly_window_start = time.time()

        return self._stats.hourly_queued < self._daemon_config.max_hourly

    # =========================================================================
    # Evaluation Queueing
    # =========================================================================

    async def _get_unevaluated_models(self) -> list["DiscoveredModel"]:
        """Get models that need evaluation.

        Returns:
            List of unevaluated DiscoveredModel
        """
        discovery = self._get_discovery()
        tracker = self._get_tracker()

        # Get models from tracker that are pending/stale/failed
        unevaluated_status = tracker.get_unevaluated_models(
            limit=100,
            source="owc",
            include_stale=True,
        )

        # Also get stale evaluations that need refresh
        stale = tracker.get_stale_evaluations(
            max_age_days=self._daemon_config.stale_evaluation_days,
            limit=50,
        )

        # Combine paths
        paths_to_evaluate = set()
        for status in unevaluated_status:
            paths_to_evaluate.add(status.model_path)
        for status in stale:
            paths_to_evaluate.add(status.model_path)

        # Match to discovered models
        return await discovery.get_unevaluated_models(limit=100)

    async def _queue_for_evaluation(self, model: "DiscoveredModel") -> bool:
        """Queue a model for evaluation by emitting TRAINING_COMPLETED event.

        This emits a synthetic TRAINING_COMPLETED event with source="backlog_owc"
        which the EvaluationDaemon will pick up and process.

        Args:
            model: Model to queue

        Returns:
            True if queued successfully, False otherwise
        """
        if not model.config_key:
            logger.warning(
                f"[BacklogEval] Cannot queue model without config_key: {model.file_name}"
            )
            return False

        try:
            # Update tracker status to "queued"
            tracker = self._get_tracker()
            if model.sha256:
                await asyncio.to_thread(
                    tracker.mark_queued,
                    model.sha256,
                    model.board_type,
                    model.num_players,
                )

            # Emit OWC_MODEL_BACKLOG_QUEUED event
            self._emit_model_queued(model)

            # Emit synthetic TRAINING_COMPLETED to trigger evaluation
            self._emit_synthetic_training_completed(model)

            self._stats.evaluations_started += 1

            logger.info(
                f"[BacklogEval] Queued for evaluation: {model.file_name} "
                f"(config={model.config_key}, priority={model.sha256[:8] if model.sha256 else 'unknown'})"
            )

            return True

        except Exception as e:
            logger.error(
                f"[BacklogEval] Failed to queue {model.file_name}: {e}",
                exc_info=True,
            )
            return False

    # =========================================================================
    # Event Emission
    # =========================================================================

    def _emit_discovery_completed(self, total_models: int, queued: int) -> None:
        """Emit BACKLOG_DISCOVERY_COMPLETED event."""
        from app.coordination.event_emission_helpers import safe_emit_event

        safe_emit_event(
            "BACKLOG_DISCOVERY_COMPLETED",
            {
                "total_models": total_models,
                "queued": queued,
                "discovery_cycles": self._stats.discovery_cycles,
                "evaluations_completed": self._stats.evaluations_completed,
                "evaluations_failed": self._stats.evaluations_failed,
                "timestamp": time.time(),
            },
            context="BacklogEvaluationDaemon",
        )

    def _emit_model_queued(self, model: "DiscoveredModel") -> None:
        """Emit OWC_MODEL_BACKLOG_QUEUED event."""
        from app.coordination.event_emission_helpers import safe_emit_event

        safe_emit_event(
            "OWC_MODEL_BACKLOG_QUEUED",
            {
                "model_path": model.path,
                "model_name": model.file_name,
                "config_key": model.config_key,
                "board_type": model.board_type,
                "num_players": model.num_players,
                "source": "owc",
                "timestamp": time.time(),
            },
            context="BacklogEvaluationDaemon",
        )

    def _emit_synthetic_training_completed(self, model: "DiscoveredModel") -> None:
        """Emit synthetic TRAINING_COMPLETED event to trigger evaluation.

        This event is picked up by EvaluationDaemon to run gauntlet evaluation.
        """
        from app.coordination.event_emission_helpers import safe_emit_event

        safe_emit_event(
            "TRAINING_COMPLETED",
            {
                "config_key": model.config_key,
                "board_type": model.board_type,
                "num_players": model.num_players,
                "model_path": model.path,  # OWC path, needs download
                "source": "backlog_owc",  # Marker for EvaluationDaemon
                "architecture": model.architecture_version or "unknown",
                "epochs": 0,  # Not applicable for backlog
                "job_id": f"backlog_{model.sha256[:16]}" if model.sha256 else None,
                "timestamp": time.time(),
            },
            context="BacklogEvaluationDaemon",
        )

    # =========================================================================
    # Health Check
    # =========================================================================

    def health_check(self) -> HealthCheckResult:
        """Return health check result for daemon manager integration."""
        if not self._daemon_config.enabled:
            return HealthCheckResult(
                healthy=True,
                status="disabled",
                message="Backlog evaluation disabled by config",
                details={"enabled": False},
            )

        # Check if OWC is available (use cached check)
        owc_available = self._stats.last_discovery_time > 0

        # Check for recent activity
        now = time.time()
        stale_threshold = self._daemon_config.scan_interval_seconds * 3
        is_stale = (now - self._stats.last_discovery_time) > stale_threshold

        # Determine health status
        if self._backpressure_active:
            status = "backpressure"
            message = "Paused due to evaluation backpressure"
            healthy = True
        elif not owc_available:
            status = "degraded"
            message = "OWC drive not yet scanned"
            healthy = True
        elif is_stale:
            status = "warning"
            message = f"No discovery in {stale_threshold}s"
            healthy = True
        else:
            status = "healthy"
            message = f"Discovered {self._stats.models_discovered} models"
            healthy = True

        return HealthCheckResult(
            healthy=healthy,
            status=status,
            message=message,
            details={
                "enabled": self._daemon_config.enabled,
                "backpressure_active": self._backpressure_active,
                "discovery_cycles": self._stats.discovery_cycles,
                "models_discovered": self._stats.models_discovered,
                "models_queued": self._stats.models_queued,
                "evaluations_completed": self._stats.evaluations_completed,
                "evaluations_failed": self._stats.evaluations_failed,
                "hourly_queued": self._stats.hourly_queued,
                "hourly_limit": self._daemon_config.max_hourly,
                "last_discovery_time": self._stats.last_discovery_time,
            },
        )


# ============================================================================
# Module-level helpers
# ============================================================================


def get_backlog_evaluation_daemon(
    config: BacklogEvalConfig | None = None,
) -> BacklogEvaluationDaemon:
    """Get singleton BacklogEvaluationDaemon instance.

    Args:
        config: Optional configuration

    Returns:
        BacklogEvaluationDaemon instance
    """
    return BacklogEvaluationDaemon.get_instance(config)


def reset_backlog_evaluation_daemon() -> None:
    """Reset singleton instance (for testing)."""
    BacklogEvaluationDaemon.reset_instance()
