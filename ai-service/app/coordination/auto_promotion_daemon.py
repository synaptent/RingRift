"""Auto-Promotion Daemon - Automatically promote models that pass evaluation.

December 2025: This daemon closes the critical feedback loop gap by automatically
promoting models based on gauntlet evaluation results. Previously, promotion
required manual intervention after evaluation completed.

The daemon:
1. Subscribes to EVALUATION_COMPLETED events
2. Checks win rates against promotion thresholds
3. Auto-promotes if thresholds met
4. Emits MODEL_PROMOTED to trigger distribution

Usage:
    from app.coordination.auto_promotion_daemon import AutoPromotionDaemon

    daemon = AutoPromotionDaemon()
    await daemon.start()

Integration with DaemonManager:
    DaemonType.AUTO_PROMOTION factory creates and manages this daemon.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "AutoPromotionConfig",
    "AutoPromotionDaemon",
    "get_auto_promotion_daemon",
]


@dataclass
class AutoPromotionConfig:
    """Configuration for auto-promotion."""
    enabled: bool = True
    # Minimum games required for promotion decision
    min_games_vs_random: int = 20
    min_games_vs_heuristic: int = 20
    # Cooldown between promotion attempts (seconds)
    promotion_cooldown_seconds: float = 300.0  # 5 minutes
    # Whether to wait for both RANDOM and HEURISTIC results
    require_both_baselines: bool = True
    # Safety: require consecutive successful evaluations
    consecutive_passes_required: int = 1
    # Dry run mode - log but don't actually promote
    dry_run: bool = False


@dataclass
class PromotionCandidate:
    """Track a model's promotion eligibility."""
    config_key: str
    model_path: str
    evaluation_results: dict[str, float] = field(default_factory=dict)
    evaluation_games: dict[str, int] = field(default_factory=dict)
    consecutive_passes: int = 0
    last_evaluation_time: float = 0.0
    last_promotion_time: float = 0.0


class AutoPromotionDaemon:
    """Daemon that auto-promotes models based on evaluation results.

    Subscribes to EVALUATION_COMPLETED events and promotes models that
    meet win rate thresholds against RANDOM and HEURISTIC baselines.
    """

    def __init__(self, config: AutoPromotionConfig | None = None):
        self.config = config or AutoPromotionConfig()
        self._running = False
        self._candidates: dict[str, PromotionCandidate] = {}
        self._promotion_history: list[dict[str, Any]] = []
        self._subscribed = False

    async def start(self) -> None:
        """Start the auto-promotion daemon."""
        if self._running:
            return

        self._running = True
        await self._subscribe_to_events()
        logger.info("[AutoPromotion] Daemon started")

    async def stop(self) -> None:
        """Stop the daemon."""
        self._running = False
        logger.info("[AutoPromotion] Daemon stopped")

    async def _subscribe_to_events(self) -> None:
        """Subscribe to EVALUATION_COMPLETED events with retry logic.

        Retries up to 3 times with exponential backoff if the router
        is not immediately available.
        """
        if self._subscribed:
            return

        max_retries = 3
        for attempt in range(max_retries):
            try:
                import asyncio

                from app.coordination.event_router import DataEventType, get_router

                router = get_router()
                if not router:
                    if attempt == max_retries - 1:
                        logger.warning("[AutoPromotion] Router unavailable after retries")
                        self._subscribed = False
                        return
                    await asyncio.sleep(0.5 * (2 ** attempt))
                    continue

                await router.subscribe(
                    DataEventType.EVALUATION_COMPLETED,
                    self._on_evaluation_completed,
                )
                self._subscribed = True
                logger.info("[AutoPromotion] Subscribed to EVALUATION_COMPLETED")
                return
            except ImportError as e:
                logger.warning(f"[AutoPromotion] Event system not available: {e}")
                self._subscribed = False
                return
            except (RuntimeError, AttributeError, TypeError) as e:
                if attempt == max_retries - 1:
                    logger.warning(f"[AutoPromotion] Failed to subscribe after retries: {e}")
                    self._subscribed = False
                else:
                    logger.debug(f"[AutoPromotion] Subscription attempt {attempt+1} failed: {e}")

    async def _on_evaluation_completed(self, event: Any) -> None:
        """Handle EVALUATION_COMPLETED event.

        Args:
            event: Event with payload containing evaluation results
        """
        if not self.config.enabled:
            return

        try:
            payload = event.payload if hasattr(event, "payload") else event
            await self._process_evaluation(payload)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[AutoPromotion] Error processing evaluation: {e}")

    async def _process_evaluation(self, payload: dict[str, Any]) -> None:
        """Process evaluation results and decide on promotion.

        Args:
            payload: Evaluation event payload with:
                - config_key: e.g., "hex8_2p"
                - model_path: Path to evaluated model
                - opponent_type: "RANDOM" or "HEURISTIC"
                - win_rate: Win rate against opponent
                - games_played: Number of games
        """
        config_key = payload.get("config_key") or payload.get("config")
        model_path = payload.get("model_path")
        opponent_type = payload.get("opponent_type", "").upper()
        win_rate = payload.get("win_rate", 0.0)
        games_played = payload.get("games_played", 0)

        if not config_key or not model_path:
            logger.debug("[AutoPromotion] Missing config_key or model_path")
            return

        # Get or create candidate
        if config_key not in self._candidates:
            self._candidates[config_key] = PromotionCandidate(
                config_key=config_key,
                model_path=model_path,
            )

        candidate = self._candidates[config_key]
        candidate.model_path = model_path
        candidate.last_evaluation_time = time.time()

        # Record result
        if opponent_type in ("RANDOM", "HEURISTIC"):
            candidate.evaluation_results[opponent_type] = win_rate
            candidate.evaluation_games[opponent_type] = games_played
            logger.info(
                f"[AutoPromotion] Recorded {config_key} vs {opponent_type}: "
                f"{win_rate:.1%} ({games_played} games)"
            )

        # Check if ready for promotion decision
        await self._check_promotion(candidate)

    async def _check_promotion(self, candidate: PromotionCandidate) -> None:
        """Check if candidate meets promotion criteria.

        Args:
            candidate: PromotionCandidate to evaluate
        """
        # Get thresholds for this config
        from app.config.thresholds import get_promotion_thresholds

        thresholds = get_promotion_thresholds(candidate.config_key)
        vs_random_threshold = thresholds.get("vs_random", 0.85)
        vs_heuristic_threshold = thresholds.get("vs_heuristic", 0.60)

        # Check if we have required results
        has_random = "RANDOM" in candidate.evaluation_results
        has_heuristic = "HEURISTIC" in candidate.evaluation_results

        if self.config.require_both_baselines and not (has_random and has_heuristic):
            logger.debug(
                f"[AutoPromotion] {candidate.config_key}: Waiting for both baselines "
                f"(RANDOM={has_random}, HEURISTIC={has_heuristic})"
            )
            return

        # Check game counts
        random_games = candidate.evaluation_games.get("RANDOM", 0)
        heuristic_games = candidate.evaluation_games.get("HEURISTIC", 0)

        if random_games < self.config.min_games_vs_random:
            logger.debug(
                f"[AutoPromotion] {candidate.config_key}: "
                f"Need {self.config.min_games_vs_random} games vs RANDOM, have {random_games}"
            )
            return

        if heuristic_games < self.config.min_games_vs_heuristic:
            logger.debug(
                f"[AutoPromotion] {candidate.config_key}: "
                f"Need {self.config.min_games_vs_heuristic} games vs HEURISTIC, have {heuristic_games}"
            )
            return

        # Check win rates
        random_win_rate = candidate.evaluation_results.get("RANDOM", 0.0)
        heuristic_win_rate = candidate.evaluation_results.get("HEURISTIC", 0.0)

        passes_random = random_win_rate >= vs_random_threshold
        passes_heuristic = heuristic_win_rate >= vs_heuristic_threshold

        if passes_random and passes_heuristic:
            candidate.consecutive_passes += 1
            logger.info(
                f"[AutoPromotion] {candidate.config_key} PASSES: "
                f"vs_random={random_win_rate:.1%} (>={vs_random_threshold:.0%}), "
                f"vs_heuristic={heuristic_win_rate:.1%} (>={vs_heuristic_threshold:.0%}) "
                f"[streak={candidate.consecutive_passes}]"
            )

            # Check cooldown
            time_since_last = time.time() - candidate.last_promotion_time
            if time_since_last < self.config.promotion_cooldown_seconds:
                remaining = self.config.promotion_cooldown_seconds - time_since_last
                logger.info(
                    f"[AutoPromotion] {candidate.config_key}: "
                    f"In cooldown, {remaining:.0f}s remaining"
                )
                return

            # Check consecutive passes
            if candidate.consecutive_passes >= self.config.consecutive_passes_required:
                await self._promote_model(candidate)
        else:
            # Reset streak on failure
            candidate.consecutive_passes = 0
            logger.info(
                f"[AutoPromotion] {candidate.config_key} FAILS: "
                f"vs_random={random_win_rate:.1%} (need {vs_random_threshold:.0%}), "
                f"vs_heuristic={heuristic_win_rate:.1%} (need {vs_heuristic_threshold:.0%})"
            )

    async def _promote_model(self, candidate: PromotionCandidate) -> None:
        """Promote a model that passed evaluation.

        Args:
            candidate: PromotionCandidate to promote
        """
        config_key = candidate.config_key
        model_path = candidate.model_path

        if self.config.dry_run:
            logger.info(
                f"[AutoPromotion] DRY RUN: Would promote {config_key} "
                f"({model_path})"
            )
            return

        logger.info(f"[AutoPromotion] Promoting {config_key} ({model_path})")

        try:
            # Import promotion controller
            from app.training.promotion_controller import PromotionController

            controller = PromotionController()
            success = await controller.promote_model(
                config_key=config_key,
                model_path=model_path,
                reason="auto_promotion_passed_evaluation",
                evaluation_results={
                    "vs_random": candidate.evaluation_results.get("RANDOM", 0.0),
                    "vs_heuristic": candidate.evaluation_results.get("HEURISTIC", 0.0),
                },
            )

            if success:
                candidate.last_promotion_time = time.time()
                candidate.consecutive_passes = 0
                candidate.evaluation_results.clear()
                candidate.evaluation_games.clear()

                self._promotion_history.append({
                    "config_key": config_key,
                    "model_path": model_path,
                    "timestamp": time.time(),
                    "vs_random": candidate.evaluation_results.get("RANDOM"),
                    "vs_heuristic": candidate.evaluation_results.get("HEURISTIC"),
                })

                logger.info(f"[AutoPromotion] Successfully promoted {config_key}")

                # Emit MODEL_PROMOTED event
                await self._emit_promotion_event(candidate)
            else:
                logger.warning(f"[AutoPromotion] Promotion failed for {config_key}")

                # Emit PROMOTION_FAILED event to trigger curriculum weight increase
                await self._emit_promotion_failed(candidate, error="Promotion validation failed")

        except ImportError:
            # Fallback: just emit the event
            logger.warning(
                "[AutoPromotion] PromotionController not available, "
                "emitting event only"
            )
            await self._emit_promotion_event(candidate)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[AutoPromotion] Promotion error for {config_key}: {e}")

            # Emit PROMOTION_FAILED event on exception
            await self._emit_promotion_failed(candidate, error=str(e))

    async def _emit_promotion_event(self, candidate: PromotionCandidate) -> None:
        """Emit MODEL_PROMOTED event and CURRICULUM_ADVANCED if applicable.

        Args:
            candidate: PromotionCandidate that was promoted
        """
        try:
            from app.coordination.event_router import DataEventType, emit_curriculum_advanced, get_router

            router = get_router()
            if router:
                await router.publish(
                    event_type=DataEventType.MODEL_PROMOTED,
                    payload={
                        "config_key": candidate.config_key,
                        "model_path": candidate.model_path,
                        "reason": "auto_promotion_daemon",
                        "vs_random": candidate.evaluation_results.get("RANDOM", 0.0),
                        "vs_heuristic": candidate.evaluation_results.get("HEURISTIC", 0.0),
                        "timestamp": time.time(),
                    },
                    source="auto_promotion_daemon",
                )
                logger.info(
                    f"[AutoPromotion] Emitted MODEL_PROMOTED for {candidate.config_key}"
                )

                # P0.5 Dec 2025: Emit CURRICULUM_ADVANCED when consecutive promotions
                # indicate curriculum tier progression readiness
                if candidate.consecutive_passes >= 2:
                    # Determine tier from consecutive pass count
                    old_tier = f"TIER_{candidate.consecutive_passes - 1}"
                    new_tier = f"TIER_{candidate.consecutive_passes}"
                    await emit_curriculum_advanced(
                        config=candidate.config_key,
                        old_tier=old_tier,
                        new_tier=new_tier,
                        trigger_reason="consecutive_promotions",
                        win_rate=candidate.evaluation_results.get("HEURISTIC", 0.0),
                        games_at_tier=candidate.evaluation_games,
                        source="auto_promotion_daemon",
                    )
                    logger.info(
                        f"[AutoPromotion] Emitted CURRICULUM_ADVANCED for {candidate.config_key}: "
                        f"{old_tier} → {new_tier}"
                    )
        except Exception as e:  # noqa: BLE001
            logger.error(f"[AutoPromotion] Failed to emit promotion event: {e}")

    async def _emit_promotion_failed(
        self,
        candidate: PromotionCandidate,
        error: str,
    ) -> None:
        """Emit PROMOTION_FAILED event to trigger curriculum weight increase.

        Args:
            candidate: PromotionCandidate that failed promotion
            error: Error message or reason for failure
        """
        try:
            from app.coordination.event_router import get_router
            from app.events.types import RingRiftEventType

            router = get_router()
            if router:
                await router.publish(
                    event_type=RingRiftEventType.PROMOTION_FAILED,
                    payload={
                        "config_key": candidate.config_key,
                        "config": candidate.config_key,  # Alternate key for compatibility
                        "model_id": candidate.model_path,
                        "error": error,
                        "vs_random": candidate.evaluation_results.get("RANDOM", 0.0),
                        "vs_heuristic": candidate.evaluation_results.get("HEURISTIC", 0.0),
                        "timestamp": time.time(),
                    },
                    source="auto_promotion_daemon",
                )
                logger.info(
                    f"[AutoPromotion] Emitted PROMOTION_FAILED for {candidate.config_key}: {error}"
                )
        except Exception as e:  # noqa: BLE001
            logger.error(f"[AutoPromotion] Failed to emit PROMOTION_FAILED event: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get daemon status."""
        return {
            "running": self._running,
            "subscribed": self._subscribed,
            "enabled": self.config.enabled,
            "dry_run": self.config.dry_run,
            "candidates": {
                k: {
                    "model_path": v.model_path,
                    "results": v.evaluation_results,
                    "games": v.evaluation_games,
                    "consecutive_passes": v.consecutive_passes,
                    "last_evaluation": v.last_evaluation_time,
                    "last_promotion": v.last_promotion_time,
                }
                for k, v in self._candidates.items()
            },
            "promotion_history_count": len(self._promotion_history),
            "recent_promotions": self._promotion_history[-5:] if self._promotion_history else [],
        }

    def health_check(self) -> "HealthCheckResult":
        """Check daemon health (December 2025: CoordinatorProtocol compliance).

        Returns:
            HealthCheckResult with status and details
        """
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="AutoPromotion daemon not running",
            )

        if not self._subscribed:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message="AutoPromotion daemon not subscribed to events",
                details=self.get_status(),
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"AutoPromotion daemon running ({len(self._promotion_history)} promotions)",
            details=self.get_status(),
        )


# Module-level singleton
_auto_promotion_daemon: AutoPromotionDaemon | None = None


def get_auto_promotion_daemon() -> AutoPromotionDaemon:
    """Get the singleton AutoPromotionDaemon instance."""
    global _auto_promotion_daemon
    if _auto_promotion_daemon is None:
        _auto_promotion_daemon = AutoPromotionDaemon()
    return _auto_promotion_daemon


def reset_auto_promotion_daemon() -> None:
    """Reset the singleton (for testing)."""
    global _auto_promotion_daemon
    _auto_promotion_daemon = None
