"""StaleEvaluationDaemon - Re-evaluate models with old Elo ratings.

Sprint 13 Session 4 (January 3, 2026): Part of model evaluation automation.

This daemon periodically checks for models with stale Elo ratings and queues
them for re-evaluation. Stale ratings occur when:
1. The meta has shifted (other models improved)
2. The gauntlet baselines have been updated
3. Training data quality has improved

Re-evaluation ensures ratings remain accurate and comparable over time.

Environment Variables:
    RINGRIFT_STALE_EVAL_ENABLED: Enable/disable daemon (default: true)
    RINGRIFT_STALE_EVAL_INTERVAL: Check interval in seconds (default: 86400 = 24h)
    RINGRIFT_STALE_EVAL_AGE_DAYS: Rating age threshold in days (default: 30)
    RINGRIFT_STALE_EVAL_MAX_PER_CYCLE: Max models per cycle (default: 5)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.coordination.event_router import DataEventType, safe_emit_event
from app.coordination.event_utils import make_config_key
from app.coordination.evaluation_queue import (
    PersistentEvaluationQueue,
    get_evaluation_queue,
)
from app.coordination.handler_base import HandlerBase, HealthCheckResult

logger = logging.getLogger(__name__)

__all__ = [
    "StaleEvaluationConfig",
    "StaleEvaluationDaemon",
    "StaleModelInfo",
    "get_stale_evaluation_daemon",
    "reset_stale_evaluation_daemon",
]


# ============================================================================
# Configuration
# ============================================================================

# Default staleness threshold: 1 day (was 30, reduced Jan 2026 for active training)
DEFAULT_STALE_AGE_DAYS = 1

# Default check interval: 24 hours
DEFAULT_CHECK_INTERVAL_SECONDS = 86400

# Priority adjustment for stale re-evaluation (lower than new models)
STALE_PRIORITY_PENALTY = 30


@dataclass
class StaleEvaluationConfig:
    """Configuration for stale evaluation daemon."""

    # Check interval (24 hours default)
    check_interval_seconds: int = DEFAULT_CHECK_INTERVAL_SECONDS

    # Daemon control
    enabled: bool = True

    # Rating age threshold (days)
    stale_age_days: int = DEFAULT_STALE_AGE_DAYS

    # Maximum models to queue per cycle
    max_models_per_cycle: int = 5

    # Base priority for stale re-evaluation requests
    base_priority: int = 20  # Lower than new model priority (50)

    @classmethod
    def from_env(cls) -> "StaleEvaluationConfig":
        """Load configuration from environment."""
        return cls(
            enabled=os.getenv("RINGRIFT_STALE_EVAL_ENABLED", "true").lower() == "true",
            check_interval_seconds=int(
                os.getenv("RINGRIFT_STALE_EVAL_INTERVAL", str(DEFAULT_CHECK_INTERVAL_SECONDS))
            ),
            stale_age_days=int(
                os.getenv("RINGRIFT_STALE_EVAL_AGE_DAYS", str(DEFAULT_STALE_AGE_DAYS))
            ),
            max_models_per_cycle=int(
                os.getenv("RINGRIFT_STALE_EVAL_MAX_PER_CYCLE", "5")
            ),
        )


@dataclass
class StaleModelInfo:
    """Model with a stale Elo rating that needs re-evaluation."""

    participant_id: str
    model_path: str | None
    board_type: str
    num_players: int
    current_rating: float
    games_played: int
    last_update: float  # Unix timestamp
    rating_age_days: float


@dataclass
class StaleEvaluationStats:
    """Statistics for stale evaluation operations."""

    cycle_count: int = 0
    models_checked: int = 0
    stale_models_found: int = 0
    models_queued: int = 0
    models_skipped_no_path: int = 0
    last_cycle_time: float = 0.0
    last_cycle_duration: float = 0.0


# ============================================================================
# Stale Evaluation Daemon
# ============================================================================


class StaleEvaluationDaemon(HandlerBase):
    """Daemon that finds and re-evaluates models with stale Elo ratings.

    This daemon:
    1. Queries EloService for models with old ratings
    2. Prioritizes by rating age and config importance
    3. Emits EVALUATION_REQUESTED events for re-evaluation
    4. Runs at lower priority than new model evaluation
    """

    def __init__(self, config: StaleEvaluationConfig | None = None):
        self._daemon_config = config or StaleEvaluationConfig.from_env()

        super().__init__(
            name="StaleEvaluationDaemon",
            config=self._daemon_config,
            cycle_interval=float(self._daemon_config.check_interval_seconds),
        )

        self._stats = StaleEvaluationStats()
        self._eval_queue: PersistentEvaluationQueue | None = None
        self._elo_service: Any = None  # Lazy loaded

    @property
    def config(self) -> StaleEvaluationConfig:
        """Get daemon configuration."""
        return self._daemon_config

    def _get_eval_queue(self) -> PersistentEvaluationQueue:
        """Get or create the evaluation queue."""
        if self._eval_queue is None:
            self._eval_queue = get_evaluation_queue()
        return self._eval_queue

    def _get_elo_service(self) -> Any:
        """Get or create the EloService."""
        if self._elo_service is None:
            try:
                from app.training.elo_service import EloService

                self._elo_service = EloService.get_instance()
            except ImportError:
                logger.warning("[StaleEval] EloService not available")
        return self._elo_service

    # =========================================================================
    # Stale Rating Detection
    # =========================================================================

    async def _find_stale_ratings(self) -> list[StaleModelInfo]:
        """Find models with stale Elo ratings.

        Returns:
            List of models with ratings older than stale_age_days
        """
        elo_service = self._get_elo_service()
        if elo_service is None:
            return []

        now = time.time()
        stale_threshold = now - (self._daemon_config.stale_age_days * 86400)

        def _query_stale_ratings() -> list[StaleModelInfo]:
            """Query stale ratings in thread to avoid blocking event loop."""
            stale_models: list[StaleModelInfo] = []
            with elo_service._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT
                        er.participant_id,
                        er.board_type,
                        er.num_players,
                        er.rating,
                        er.games_played,
                        er.last_update,
                        p.model_path
                    FROM elo_ratings er
                    LEFT JOIN participants p ON er.participant_id = p.participant_id
                    WHERE er.last_update IS NOT NULL
                      AND er.last_update < ?
                      AND er.games_played > 0
                    ORDER BY er.last_update ASC
                    """,
                    (stale_threshold,),
                )

                for row in cursor.fetchall():
                    last_update = row["last_update"]
                    if last_update is None:
                        continue

                    rating_age_days = (now - last_update) / 86400

                    stale_models.append(
                        StaleModelInfo(
                            participant_id=row["participant_id"],
                            model_path=row["model_path"],
                            board_type=row["board_type"],
                            num_players=row["num_players"],
                            current_rating=row["rating"],
                            games_played=row["games_played"],
                            last_update=last_update,
                            rating_age_days=rating_age_days,
                        )
                    )
            return stale_models

        try:
            # Run blocking SQLite query in thread to avoid blocking event loop
            stale_models = await asyncio.to_thread(_query_stale_ratings)

            self._stats.models_checked += len(stale_models)
            self._stats.stale_models_found += len(stale_models)

            logger.info(
                f"[StaleEval] Found {len(stale_models)} models with stale ratings "
                f"(>{self._daemon_config.stale_age_days} days old)"
            )

            return stale_models

        except Exception as e:
            logger.error(f"[StaleEval] Error querying stale ratings: {e}")
            return []

    def _compute_priority(self, model: StaleModelInfo) -> int:
        """Compute re-evaluation priority.

        Older ratings get higher priority, but all stale re-evaluations
        are lower priority than new model evaluations.

        Args:
            model: Stale model info

        Returns:
            Priority value (higher = evaluated sooner)
        """
        priority = self._daemon_config.base_priority

        # Add age-based priority: older ratings get higher priority
        # +1 priority for every 10 days past the threshold
        age_bonus = int((model.rating_age_days - self._daemon_config.stale_age_days) / 10)
        priority += min(age_bonus, 30)  # Cap at +30

        # 4-player configs get slight priority boost
        if model.num_players == 4:
            priority += 10

        # Hexagonal (complex) configs get slight priority boost
        if model.board_type in ("hexagonal",):
            priority += 5

        return priority

    # =========================================================================
    # Main Loop
    # =========================================================================

    async def _run_cycle(self) -> None:
        """Run one stale evaluation check cycle."""
        cycle_start = time.time()
        self._stats.cycle_count += 1

        if not self._daemon_config.enabled:
            logger.debug("[StaleEval] Daemon disabled, skipping cycle")
            return

        # Find stale ratings
        stale_models = await self._find_stale_ratings()
        if not stale_models:
            logger.debug("[StaleEval] No stale ratings found")
            return

        # Queue up to max_models_per_cycle for re-evaluation
        queue = self._get_eval_queue()
        queued_count = 0

        for model in stale_models[: self._daemon_config.max_models_per_cycle]:
            # Skip if no model path (can't evaluate)
            if not model.model_path:
                self._stats.models_skipped_no_path += 1
                continue

            # Skip if model file doesn't exist
            model_path = Path(model.model_path)
            if not model_path.exists():
                logger.debug(f"[StaleEval] Model file not found: {model.model_path}")
                self._stats.models_skipped_no_path += 1
                continue

            priority = self._compute_priority(model)

            request_id = queue.add_request(
                model_path=model.model_path,
                board_type=model.board_type,
                num_players=model.num_players,
                priority=priority,
                source="stale_re_evaluation",
            )

            if request_id:
                queued_count += 1
                self._stats.models_queued += 1
                await self._emit_evaluation_requested(model, priority, request_id)

        cycle_duration = time.time() - cycle_start
        self._stats.last_cycle_time = time.time()
        self._stats.last_cycle_duration = cycle_duration

        if queued_count > 0:
            logger.info(
                f"[StaleEval] Queued {queued_count} stale models for re-evaluation "
                f"in {cycle_duration:.1f}s"
            )

    async def _emit_evaluation_requested(
        self,
        model: StaleModelInfo,
        priority: int,
        request_id: str,
    ) -> None:
        """Emit EVALUATION_REQUESTED event for a stale model."""
        payload = {
            "request_id": request_id,
            "model_path": model.model_path,
            "board_type": model.board_type,
            "num_players": model.num_players,
            "config_key": make_config_key(model.board_type, model.num_players),
            "priority": priority,
            "source": "stale_re_evaluation",
            "current_rating": model.current_rating,
            "rating_age_days": model.rating_age_days,
            "timestamp": time.time(),
        }

        try:
            safe_emit_event(DataEventType.EVALUATION_REQUESTED, payload)
        except Exception as e:
            logger.debug(f"[StaleEval] Failed to emit eval request: {e}")

    # =========================================================================
    # Health Check
    # =========================================================================

    def health_check(self) -> HealthCheckResult:
        """Check daemon health."""
        details = {
            "enabled": self._daemon_config.enabled,
            "stale_age_days": self._daemon_config.stale_age_days,
            "cycle_count": self._stats.cycle_count,
            "models_checked": self._stats.models_checked,
            "stale_models_found": self._stats.stale_models_found,
            "models_queued": self._stats.models_queued,
            "models_skipped_no_path": self._stats.models_skipped_no_path,
            "last_cycle_time": self._stats.last_cycle_time,
            "last_cycle_duration": self._stats.last_cycle_duration,
        }

        if not self._daemon_config.enabled:
            return HealthCheckResult(
                healthy=True,
                status="disabled",
                message="Stale evaluation daemon is disabled",
                details=details,
            )

        # Check EloService availability
        elo_service = self._get_elo_service()
        if elo_service is None:
            return HealthCheckResult(
                healthy=False,
                status="degraded",
                message="EloService not available",
                details=details,
            )

        return HealthCheckResult(
            healthy=True,
            status="healthy",
            message=f"Queued {self._stats.models_queued} stale models for re-evaluation",
            details=details,
        )


# ============================================================================
# Singleton Access
# ============================================================================

_daemon_instance: StaleEvaluationDaemon | None = None


def get_stale_evaluation_daemon(
    config: StaleEvaluationConfig | None = None,
) -> StaleEvaluationDaemon:
    """Get the singleton StaleEvaluationDaemon instance.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        The singleton daemon instance
    """
    global _daemon_instance

    if _daemon_instance is None:
        _daemon_instance = StaleEvaluationDaemon(config)
    return _daemon_instance


def reset_stale_evaluation_daemon() -> None:
    """Reset the singleton (for testing)."""
    global _daemon_instance
    _daemon_instance = None
