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

    daemon = AutoPromotionDaemon.get_instance()
    await daemon.start()

Integration with DaemonManager:
    DaemonType.AUTO_PROMOTION factory creates and manages this daemon.

January 2026 (Sprint 12.2): Migrated to HandlerBase for unified lifecycle,
event subscription, health checks, and fire-and-forget helpers.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from app.config.coordination_defaults import PromotionGameDefaults
from app.config.thresholds import AUTO_PROMOTION_MIN_QUALITY
from app.coordination.event_utils import parse_config_key
from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.contracts import CoordinatorStatus
from app.utils.game_discovery import count_games_for_config
from app.utils.retry import RetryConfig

logger = logging.getLogger(__name__)

# January 26, 2026 (P4): Elo velocity gate - import Elo trend function
try:
    from app.training.elo_service import get_elo_trend_for_config
    HAS_ELO_TREND = True
except ImportError:
    HAS_ELO_TREND = False
    get_elo_trend_for_config = None

# January 3, 2026 (Sprint 16.2): Hashgraph consensus for BFT model promotion
# Requires supermajority approval before promotion can proceed
try:
    from app.coordination.hashgraph import (
        get_promotion_consensus_manager,
        PromotionConsensusConfig,
        EvaluationEvidence,
    )
    HAS_HASHGRAPH_CONSENSUS = True
except ImportError:
    HAS_HASHGRAPH_CONSENSUS = False
    get_promotion_consensus_manager = None
    PromotionConsensusConfig = None
    EvaluationEvidence = None

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
    # Dec 30: REVERTED from 20 to 50 - 20 games gives 95% CI ±10% which is insufficient
    # to distinguish quality. 50 games gives 95% CI ±6.4% which is acceptable.
    min_games_vs_random: int = 50
    min_games_vs_heuristic: int = 50
    # Cooldown between promotion attempts (seconds)
    promotion_cooldown_seconds: float = 300.0  # 5 minutes
    # Whether to wait for both RANDOM and HEURISTIC results
    require_both_baselines: bool = True
    # Safety: require consecutive successful evaluations
    # Feb 2026: Reduced from 2 to 1. The 2-pass requirement (Dec 30) predated the
    # head-to-head gate (Jan 10), gauntlet verification (Jan 5), and velocity gate (Jan 26).
    # With 6 other safety gates, consecutive passes add 4-8h delay without meaningful signal.
    consecutive_passes_required: int = 1
    # Dry run mode - log but don't actually promote
    dry_run: bool = False
    # Feb 22, 2026: Raised from 10→25. With 20-50 game evaluations, Elo estimates
    # have ~50-100 point CIs. +10 was within noise, causing random promotions.
    min_elo_improvement: float = 25.0
    # December 2025: Quality gate settings to prevent bad model promotion
    quality_gate_enabled: bool = True
    # Jan 12, 2026: Lowered from 1000 to 100 to enable early promotion for bootstrap configs
    # 100 games provides ~10% confidence interval, aligns with MIN_GAMES_FOR_EXPORT_4P
    min_training_games: int = 100
    # Dec 30: REVERTED to 0.55 - balance between quality and iteration speed
    min_quality_score: float = 0.55
    require_parity_validation: bool = True  # Require TS parity validation passed
    # December 2025: Stability gate to prevent promoting volatile models
    stability_gate_enabled: bool = True
    max_volatility_score: float = 0.6  # Block models with volatility > 0.6
    # January 5, 2026: Gauntlet verification gate - require minimum combined win rate
    # This ensures models prove their strength in head-to-head competition before promotion
    gauntlet_verification_enabled: bool = True
    min_gauntlet_win_rate: float = 0.55  # Require 55% combined win rate vs all opponents
    # January 10, 2026: Head-to-head gate vs current canonical model
    # CRITICAL: Ensures new models actually beat the current best, not just baselines
    # This prevents model regression where new models are worse than current canonical
    head_to_head_enabled: bool = True
    # Feb 22, 2026: Tightened from 0.52/50 to 0.55/100. At 52% over 50 games
    # (26/50 wins), the p-value vs 50% is ~0.44 — statistically meaningless.
    # Feb 23, 2026: Raised from 0.55 to 0.58. At n=100 games, 58% is the
    # minimum for p < 0.05 statistical significance (binomial test vs 50%).
    min_win_rate_vs_canonical: float = 0.58  # Must win 58%+ vs current canonical
    head_to_head_games: int = 100  # Games to play vs canonical for evaluation
    # January 26, 2026 (P4): Elo velocity gate - block promotion if Elo is declining
    # This prevents promoting models during regression periods, ensuring only models
    # with positive momentum (or at least stable Elo) get promoted.
    velocity_gate_enabled: bool = True
    # Feb 2026: Relaxed from 0.0 to -2.0. Zero threshold blocked on noise (-0.01 Elo/hr).
    # -2.0 still blocks severe regression (>48 Elo/day decline) while allowing normal variance.
    min_velocity_for_promotion: float = -2.0


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
    # Dec 27, 2025: Track Elo improvement for stricter promotion
    elo_improvement: float = 0.0  # Elo gain vs previous model
    estimated_elo: float = 0.0  # Estimated Elo from evaluation
    # Dec 28, 2025: Track if model beats current best (for relative promotion)
    beats_current_best: bool = False  # True if this model won head-to-head vs champion
    # Dec 29, 2025: Track consecutive failures for curriculum regression
    consecutive_failures: int = 0  # Number of consecutive failed promotions
    previous_elo: float = 0.0  # Elo before this evaluation (for calculating change)
    # Dec 31, 2025: Track training game count for graduated thresholds
    training_game_count: int = 0  # Total training games for this config


class AutoPromotionDaemon(HandlerBase):
    """Daemon that auto-promotes models based on evaluation results.

    Subscribes to EVALUATION_COMPLETED events and promotes models that
    meet win rate thresholds against RANDOM and HEURISTIC baselines.

    January 2026 (Sprint 12.2): Migrated to HandlerBase for unified:
    - Singleton management (get_instance/reset_instance)
    - Event subscription via _get_event_subscriptions()
    - Lifecycle management (start/stop)
    - Health checks
    - Fire-and-forget task helpers (_safe_create_task, _try_emit_event)
    """

    # Event source identifier for SafeEventEmitterMixin
    _event_source = "AutoPromotionDaemon"

    def __init__(self, config: AutoPromotionConfig | None = None):
        # Long cycle interval since this daemon is purely event-driven
        super().__init__(name="auto_promotion", cycle_interval=300.0)
        self.config = config or AutoPromotionConfig()
        self._candidates: dict[str, PromotionCandidate] = {}
        self._promotion_history: list[dict[str, Any]] = []
        # Background subscription retry task (for resilience when router starts late)
        self._subscription_retry_task: asyncio.Task | None = None

    def _get_event_subscriptions(self) -> dict[str, Callable[[dict[str, Any]], Any]]:
        """Return event subscriptions for HandlerBase.

        January 2026: Migrated from manual _subscribe_to_events() to use
        HandlerBase's declarative subscription system.

        Note: Event types must match exactly (case-sensitive) with publisher.
        The event_router publishes as "EVALUATION_COMPLETED" (uppercase).
        """
        return {
            "EVALUATION_COMPLETED": self._on_evaluation_completed,
        }

    async def _on_start(self) -> None:
        """Hook called when daemon starts.

        If initial subscription fails, start background retry task
        to handle case where router becomes available later.
        """
        # If HandlerBase subscription didn't succeed, start retry task
        if not self._event_subscribed:
            logger.info("[AutoPromotion] Initial subscription pending, starting retry task")
            self._subscription_retry_task = self._safe_create_task(
                self._periodic_subscription_retry(),
                context="subscription_retry",
            )

    async def _on_stop(self) -> None:
        """Hook called when daemon stops."""
        # Cancel background subscription task if running
        if self._subscription_retry_task and not self._subscription_retry_task.done():
            self._subscription_retry_task.cancel()
            try:
                await self._subscription_retry_task
            except asyncio.CancelledError:
                pass
            self._subscription_retry_task = None

    async def _run_cycle(self) -> None:
        """Main work loop - minimal for this event-driven daemon.

        The daemon is purely event-driven (handles EVALUATION_COMPLETED),
        so the run cycle just checks subscription health.
        """
        # If not subscribed yet, try again
        if not self._event_subscribed:
            self._subscribe_all_events()

    async def _periodic_subscription_retry(self) -> None:
        """Periodically retry event subscription until successful.

        Dec 29, 2025: Added to handle case where router becomes available
        after daemon starts. Retries every 60 seconds until subscribed.
        """
        retry_interval = 60.0  # seconds
        max_attempts = 10  # Give up after 10 minutes

        for attempt in range(max_attempts):
            if not self._running:
                return

            await asyncio.sleep(retry_interval)

            if self._event_subscribed:
                logger.debug("[AutoPromotion] Already subscribed, stopping retry loop")
                return

            logger.debug(f"[AutoPromotion] Subscription retry attempt {attempt + 1}/{max_attempts}")
            self._subscribe_all_events()

            if self._event_subscribed:
                logger.info("[AutoPromotion] Subscription succeeded on retry")
                return

        logger.warning("[AutoPromotion] Gave up on subscription after max retries")

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

        # Dec 31, 2025: Update training game count for graduated thresholds
        parsed = parse_config_key(config_key)
        if parsed:
            try:
                game_count = count_games_for_config(parsed.board_type, parsed.num_players)
                candidate.training_game_count = game_count
                logger.debug(f"[AutoPromotion] {config_key} has {game_count} training games")
            except Exception as e:
                logger.debug(f"[AutoPromotion] Could not get game count for {config_key}: {e}")

        # Record result - single opponent type
        if opponent_type in ("RANDOM", "HEURISTIC"):
            candidate.evaluation_results[opponent_type] = win_rate
            candidate.evaluation_games[opponent_type] = games_played
            logger.info(
                f"[AutoPromotion] Recorded {config_key} vs {opponent_type}: "
                f"{win_rate:.1%} ({games_played} games)"
            )

        # Jan 5, 2026: Extract from opponent_results dict (from evaluation_daemon)
        # The evaluation_daemon sends opponent_results with keys like "random", "heuristic"
        opponent_results = payload.get("opponent_results", {})
        for opp_key, opp_result in opponent_results.items():
            # Normalize key to uppercase for consistency
            normalized_key = opp_key.upper() if isinstance(opp_key, str) else str(opp_key).upper()
            if normalized_key in ("RANDOM", "HEURISTIC"):
                opp_win_rate = opp_result.get("win_rate", 0.0) if isinstance(opp_result, dict) else 0.0
                opp_games = opp_result.get("games_played", 0) if isinstance(opp_result, dict) else 0
                candidate.evaluation_results[normalized_key] = float(opp_win_rate)
                candidate.evaluation_games[normalized_key] = opp_games
                logger.info(
                    f"[AutoPromotion] Recorded {config_key} vs {normalized_key} from opponent_results: "
                    f"{opp_win_rate:.1%} ({opp_games} games)"
                )

        # Dec 28, 2025: Also check for direct baseline win rates from gauntlet
        # The gauntlet emits both rates in a single event
        vs_random_rate = payload.get("vs_random_rate")
        vs_heuristic_rate = payload.get("vs_heuristic_rate")
        if vs_random_rate is not None:
            candidate.evaluation_results["RANDOM"] = float(vs_random_rate)
            # Use total games if per-opponent count not available
            if "RANDOM" not in candidate.evaluation_games:
                candidate.evaluation_games["RANDOM"] = games_played // 2 or 50
            logger.info(
                f"[AutoPromotion] Recorded {config_key} vs RANDOM from gauntlet: "
                f"{vs_random_rate:.1%}"
            )
        if vs_heuristic_rate is not None:
            candidate.evaluation_results["HEURISTIC"] = float(vs_heuristic_rate)
            if "HEURISTIC" not in candidate.evaluation_games:
                candidate.evaluation_games["HEURISTIC"] = games_played // 2 or 50
            logger.info(
                f"[AutoPromotion] Recorded {config_key} vs HEURISTIC from gauntlet: "
                f"{vs_heuristic_rate:.1%}"
            )

        # Dec 27, 2025: Extract Elo information from payload
        elo_improvement = payload.get("elo_improvement", payload.get("elo_delta", 0.0))
        estimated_elo = payload.get("estimated_elo", payload.get("elo", 0.0))
        if elo_improvement:
            candidate.elo_improvement = float(elo_improvement)
        if estimated_elo:
            candidate.estimated_elo = float(estimated_elo)

        # Dec 28, 2025: Extract beats_current_best from payload
        # This flag indicates if the model won head-to-head vs the current champion
        beats_current_best = payload.get("beats_current_best", payload.get("beats_champion", False))
        if beats_current_best:
            candidate.beats_current_best = bool(beats_current_best)

        # Check if ready for promotion decision
        await self._check_promotion(candidate)

    def _get_canonical_heuristic_win_rate(self, config_key: str) -> float | None:
        """Get the current canonical model's heuristic win rate for Tier 3.5 check.

        January 3, 2026: Added to enable Tier 3.5 significant improvement promotion.
        If a new model beats the current canonical by >10%, it should promote even
        if it doesn't meet aspirational thresholds.

        Returns:
            Heuristic win rate of current canonical model, or None if not available.
        """
        try:
            from app.coordination.elo_progress_tracker import get_elo_progress_tracker

            tracker = get_elo_progress_tracker()
            snapshot = tracker.get_latest_snapshot(config_key)

            if snapshot and snapshot.vs_heuristic_win_rate is not None:
                return snapshot.vs_heuristic_win_rate
            return None
        except (ImportError, OSError, RuntimeError) as e:
            logger.debug(f"[AutoPromotion] Could not get canonical heuristic rate: {e}")
            return None

    def _get_effective_consecutive_passes_required(self, training_game_count: int) -> int:
        """Get effective consecutive passes required, accounting for bootstrap phase.

        January 6, 2026: Bootstrap configs (<5000 games) use relaxed requirements to
        break the catch-22 where models can't get games without promotion and can't
        promote without games.

        Feb 2026: Now returns 1 for all configs since consecutive_passes_required
        default changed to 1. Kept for clarity and in case the default is raised again.

        Args:
            training_game_count: Number of training games for this config.

        Returns:
            1 for bootstrap configs, otherwise the config default (now 1).
        """
        from app.config.thresholds import GAME_COUNT_BOOTSTRAP_THRESHOLD

        is_bootstrap = training_game_count < GAME_COUNT_BOOTSTRAP_THRESHOLD
        if is_bootstrap:
            return 1  # Single pass for bootstrap configs
        return self.config.consecutive_passes_required

    def _get_effective_min_elo_improvement(self, training_game_count: int) -> float:
        """Get effective minimum Elo improvement, accounting for bootstrap phase.

        January 6, 2026: Bootstrap configs (<5000 games) use relaxed requirements.
        Any positive Elo improvement (even +5) should be promoted during bootstrap
        to accelerate the training flywheel.

        Args:
            training_game_count: Number of training games for this config.

        Returns:
            5.0 for bootstrap configs, otherwise the config default (typically 10.0).
        """
        from app.config.thresholds import GAME_COUNT_BOOTSTRAP_THRESHOLD

        is_bootstrap = training_game_count < GAME_COUNT_BOOTSTRAP_THRESHOLD
        if is_bootstrap:
            return 5.0  # Lower Elo threshold for bootstrap configs
        return self.config.min_elo_improvement

    async def _check_promotion(self, candidate: PromotionCandidate) -> None:
        """Check if candidate meets promotion criteria.

        Uses the two-tier promotion system from thresholds.py:
        1. ASPIRATIONAL thresholds - strong models that definitely should promote
        2. MINIMUM floor + beats_current_best - incremental improvements that beat champion

        Args:
            candidate: PromotionCandidate to evaluate
        """
        # Dec 28, 2025: Use unified should_promote_model() for two-tier promotion
        from app.config.thresholds import should_promote_model

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
        # January 2026 (Phase 2.3): Graduate minimum games by player count
        # 4-player games have higher variance, requiring more games for statistical confidence
        random_games = candidate.evaluation_games.get("RANDOM", 0)
        heuristic_games = candidate.evaluation_games.get("HEURISTIC", 0)

        # Get player count from config_key for graduated thresholds
        parsed = parse_config_key(candidate.config_key)
        num_players = parsed.num_players if parsed else 2
        min_games = PromotionGameDefaults.get_min_games(num_players)

        if random_games < min_games:
            logger.debug(
                f"[AutoPromotion] {candidate.config_key}: "
                f"Need {min_games} games vs RANDOM (graduated for {num_players}p), have {random_games}"
            )
            return

        if heuristic_games < min_games:
            logger.debug(
                f"[AutoPromotion] {candidate.config_key}: "
                f"Need {min_games} games vs HEURISTIC (graduated for {num_players}p), have {heuristic_games}"
            )
            return

        # Get win rates
        random_win_rate = candidate.evaluation_results.get("RANDOM", 0.0)
        heuristic_win_rate = candidate.evaluation_results.get("HEURISTIC", 0.0)

        # Dec 28, 2025: Use multi-tier promotion system
        # - Elo-adaptive: Bootstrap models get easier thresholds
        # - Game-count graduated: Configs with less data get easier thresholds
        # - Aspirational: Model meets high thresholds for strong performance
        # - Relative: Model beats current best AND meets minimum floor
        # Dec 30, 2025: Pass current_best_elo to enable safety check
        # This prevents "race to the bottom" where weak models beat weaker models
        current_best_elo = candidate.previous_elo if candidate.previous_elo > 0 else None
        # Dec 30, 2025: Pass model_elo to enable Elo-adaptive thresholds
        # This allows bootstrap models (800-1200 Elo) to pass with lower thresholds
        model_elo = candidate.estimated_elo if candidate.estimated_elo > 0 else None
        # Dec 31, 2025: Pass game_count to enable graduated thresholds
        # Configs with limited training data get easier thresholds during bootstrap
        game_count = candidate.training_game_count if candidate.training_game_count > 0 else None
        # Jan 3, 2026: Pass current_vs_heuristic_rate to enable Tier 3.5 significant improvement
        # This allows models with >10% improvement over canonical to promote even if not aspirational
        current_vs_heuristic = self._get_canonical_heuristic_win_rate(candidate.config_key)
        should_promote, reason = should_promote_model(
            config_key=candidate.config_key,
            vs_random_rate=random_win_rate,
            vs_heuristic_rate=heuristic_win_rate,
            beats_current_best=candidate.beats_current_best,
            current_best_elo=current_best_elo,
            model_elo=model_elo,
            game_count=game_count,
            current_vs_heuristic_rate=current_vs_heuristic,
        )

        if should_promote:
            candidate.consecutive_passes += 1
            logger.info(
                f"[AutoPromotion] {candidate.config_key} PASSES: {reason} "
                f"[streak={candidate.consecutive_passes}, beats_best={candidate.beats_current_best}]"
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
            # Jan 6, 2026: Use bootstrap-aware thresholds for configs with limited data
            game_count = candidate.training_game_count
            effective_passes_required = self._get_effective_consecutive_passes_required(game_count)
            effective_min_elo = self._get_effective_min_elo_improvement(game_count)

            # Log if using bootstrap thresholds
            if effective_passes_required < self.config.consecutive_passes_required:
                logger.info(
                    f"[AutoPromotion] {candidate.config_key}: Using bootstrap thresholds "
                    f"(games={game_count}, passes={effective_passes_required}, min_elo={effective_min_elo:.1f})"
                )

            if candidate.consecutive_passes >= effective_passes_required:
                # Dec 27, 2025: Check Elo improvement requirement (optional)
                # Note: For relative promotion, we may want to skip this check
                # since beating champion is already strong signal
                if (
                    effective_min_elo > 0
                    and not candidate.beats_current_best  # Skip Elo check if beat champion
                    and candidate.elo_improvement < effective_min_elo
                ):
                    logger.info(
                        f"[AutoPromotion] {candidate.config_key}: "
                        f"Elo improvement {candidate.elo_improvement:+.1f} < {effective_min_elo:.1f} required"
                    )
                    self._emit_promotion_rejected(
                        candidate,
                        gate="elo_improvement",
                        reason=(
                            f"Elo improvement {candidate.elo_improvement:+.1f} "
                            f"< {effective_min_elo:.1f} required"
                        ),
                        actual=candidate.elo_improvement,
                        threshold=effective_min_elo,
                    )
                    return

                # January 26, 2026 (P4): Elo velocity gate - block promotion if declining
                # This ensures we don't promote models during regression periods
                if self.config.velocity_gate_enabled and HAS_ELO_TREND:
                    velocity_passed, velocity_reason = await self._check_velocity_gate(candidate)
                    if not velocity_passed:
                        logger.info(
                            f"[AutoPromotion] {candidate.config_key}: "
                            f"Velocity gate FAILED: {velocity_reason}"
                        )
                        self._emit_promotion_rejected(
                            candidate,
                            gate="velocity",
                            reason=f"Velocity gate failed: {velocity_reason}",
                        )
                        return

                await self._promote_model(candidate)
        else:
            # Reset streak on failure
            candidate.consecutive_passes = 0
            logger.info(
                f"[AutoPromotion] {candidate.config_key} FAILS: {reason} "
                f"(vs_random={random_win_rate:.1%}, vs_heuristic={heuristic_win_rate:.1%}, "
                f"beats_best={candidate.beats_current_best})"
            )
            self._emit_promotion_rejected(
                candidate,
                gate="threshold",
                reason=reason,
                actual=heuristic_win_rate,
            )

    async def _check_quality_gate(
        self,
        candidate: PromotionCandidate,
    ) -> tuple[bool, str]:
        """Check if candidate passes quality gate before promotion.

        December 2025: Prevents promotion of models trained on corrupted or
        insufficient data by verifying:
        1. Parity validation is complete (TS/Python match)
        2. Sufficient training game count (1000+)
        3. Data quality score is acceptable (>0.6)

        Args:
            candidate: PromotionCandidate to check

        Returns:
            Tuple of (passed, reason) where reason explains the gate result
        """
        if not self.config.quality_gate_enabled:
            return True, "quality_gate_disabled"

        config_key = candidate.config_key

        # Parse board_type and num_players using canonical utility
        parsed = parse_config_key(config_key)
        if not parsed:
            logger.warning(f"[AutoPromotion] Cannot parse config_key: {config_key}")
            return True, "config_key_unparseable"
        board_type = parsed.board_type
        num_players = parsed.num_players

        # Check parity validation status
        if self.config.require_parity_validation:
            parity_passed, parity_reason = await self._check_parity_status(
                board_type, num_players
            )
            if not parity_passed:
                # Dec 2025: If database check fails due to pending/incomplete,
                # try live parity validation on coordinator (has Node.js)
                if "pending" in parity_reason.lower() or "incomplete" in parity_reason.lower():
                    logger.info(
                        f"[AutoPromotion] Database parity check incomplete for {config_key}, "
                        f"attempting live validation"
                    )
                    live_passed, live_reason = await self._run_live_parity_validation(
                        config_key, sample_games=100
                    )
                    if not live_passed:
                        return False, f"live_parity_failed: {live_reason}"
                    # Live validation passed, continue
                    logger.info(
                        f"[AutoPromotion] Live parity validation passed for {config_key}: "
                        f"{live_reason}"
                    )
                else:
                    return False, f"parity_failed: {parity_reason}"

        # Check training data quality
        quality_passed, quality_reason = await self._check_data_quality(
            board_type, num_players
        )
        if not quality_passed:
            return False, f"quality_failed: {quality_reason}"

        return True, "quality_gate_passed"

    async def _check_velocity_gate(
        self,
        candidate: PromotionCandidate,
    ) -> tuple[bool, str]:
        """Check if config's Elo velocity allows promotion.

        January 26, 2026 (P4): Blocks promotion if Elo is declining. This prevents
        promoting models during regression periods, ensuring only models with positive
        momentum (or at least stable Elo) get promoted.

        Args:
            candidate: PromotionCandidate to check

        Returns:
            Tuple of (passed, reason) where reason explains the gate result
        """
        if not HAS_ELO_TREND or get_elo_trend_for_config is None:
            # Gracefully skip if elo_service not available
            return True, "velocity_check_unavailable"

        try:
            # Get Elo trend for this config over the last 48 hours
            # Feb 2026: Extended from 48h to 72h for more stable velocity estimates
            trend = get_elo_trend_for_config(candidate.config_key, hours=72)

            if trend is None:
                # No trend data available - allow promotion (bootstrap case)
                return True, "no_trend_data_available"

            velocity = trend.get("slope", 0.0)
            is_declining = trend.get("is_declining", False)

            # Block if velocity is below threshold (default: 0.0, meaning declining)
            if is_declining or velocity < self.config.min_velocity_for_promotion:
                return False, f"velocity={velocity:.2f}_elo_per_hour_declining"

            return True, f"velocity_ok_{velocity:.2f}_elo_per_hour"

        except Exception as e:
            # Don't block on velocity check errors - log and allow promotion
            logger.warning(f"[AutoPromotion] Velocity check error for {candidate.config_key}: {e}")
            return True, f"velocity_check_error: {e}"

    async def _check_parity_status(
        self,
        board_type: str,
        num_players: int,
    ) -> tuple[bool, str]:
        """Check if parity validation has passed for this config.

        Args:
            board_type: Board type (e.g., "hex8")
            num_players: Number of players (2, 3, or 4)

        Returns:
            Tuple of (passed, reason)
        """
        try:
            from pathlib import Path

            from app.db.game_replay import GameReplayDB

            # Find canonical database
            db_path = Path(f"data/games/canonical_{board_type}_{num_players}p.db")
            if not db_path.exists():
                return False, f"database_not_found: {db_path}"

            # Check parity_gate status in database using context manager
            with GameReplayDB(str(db_path)) as db:
                with db._get_conn() as conn:
                    # Count games by parity status
                    cursor = conn.execute(
                        """
                        SELECT parity_gate, COUNT(*) as count
                        FROM games
                        WHERE game_status = 'completed'
                        GROUP BY parity_gate
                        """
                    )
                    status_counts = {row[0]: row[1] for row in cursor.fetchall()}

            total_games = sum(status_counts.values())
            passed_games = status_counts.get("passed", 0)
            pending_games = status_counts.get("pending_gate", 0)
            failed_games = status_counts.get("failed", 0)

            # Require majority of games to have passed parity
            if total_games == 0:
                return False, "no_completed_games"

            pass_rate = passed_games / total_games if total_games > 0 else 0

            if pass_rate < 0.5:
                return False, (
                    f"low_parity_pass_rate: {pass_rate:.1%} "
                    f"(passed={passed_games}, pending={pending_games}, failed={failed_games})"
                )

            # If too many pending, validation hasn't run
            if pending_games > passed_games:
                return False, (
                    f"parity_validation_incomplete: pending={pending_games}, passed={passed_games}"
                )

            return True, f"parity_ok: {pass_rate:.1%} passed"

        except Exception as e:
            logger.warning(f"[AutoPromotion] Parity check error: {e}")
            # Don't block on parity check errors
            return True, f"parity_check_error: {e}"

    async def _run_live_parity_validation(
        self, config_key: str, sample_games: int = 100
    ) -> tuple[bool, str]:
        """Run live TS/Python parity validation on coordinator.

        December 2025: The coordinator (mac-studio) has Node.js installed, allowing
        us to run actual parity validation instead of relying on database status.
        Cluster nodes lack npx so parity gates often show "pending_gate".

        This method runs the parity check script to validate a sample of games,
        ensuring the Python rules engine matches TypeScript before promotion.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")
            sample_games: Number of games to validate (default: 100)

        Returns:
            Tuple of (passed, reason) where reason explains the result
        """
        import asyncio
        import subprocess
        from pathlib import Path

        # Only run on coordinator (has Node.js)
        try:
            from app.config.env import env
            if not env.is_coordinator:
                logger.debug(
                    f"[AutoPromotion] Skipping live parity validation for {config_key} "
                    "(not coordinator)"
                )
                return True, "skipped_not_coordinator"
        except ImportError:
            # If env module unavailable, check by hostname
            import socket
            hostname = socket.gethostname().lower()
            if "mac-studio" not in hostname and "local-mac" not in hostname:
                return True, "skipped_not_coordinator"

        # Parse board_type and num_players using canonical utility
        parsed = parse_config_key(config_key)
        if not parsed:
            logger.warning(f"[AutoPromotion] Cannot parse config_key: {config_key}")
            return True, "config_key_unparseable"
        board_type = parsed.board_type
        num_players = parsed.num_players

        # Find canonical database for this config
        db_path = Path(f"data/games/canonical_{board_type}_{num_players}p.db")
        if not db_path.exists():
            logger.debug(
                f"[AutoPromotion] No canonical DB for {config_key}, skipping live parity"
            )
            return True, "no_canonical_db"

        # Build parity check command
        script_path = Path("scripts/check_ts_python_replay_parity.py")
        if not script_path.exists():
            logger.warning("[AutoPromotion] Parity check script not found")
            return True, "script_not_found"

        cmd = [
            "python",
            str(script_path),
            "--db", str(db_path),
            "--limit", str(sample_games),
        ]

        logger.info(
            f"[AutoPromotion] Running live parity validation for {config_key} "
            f"({sample_games} games from {db_path})"
        )

        try:
            # Run parity check with timeout
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                timeout=300,  # 5 minute timeout
                cwd=str(Path.cwd()),
            )

            if result.returncode == 0:
                logger.info(
                    f"[AutoPromotion] Live parity validation PASSED for {config_key}"
                )
                return True, "parity_passed"
            else:
                # Extract error from stderr
                error_msg = result.stderr.decode("utf-8", errors="replace")[:500]
                stdout_msg = result.stdout.decode("utf-8", errors="replace")[:200]
                logger.warning(
                    f"[AutoPromotion] Live parity validation FAILED for {config_key}: "
                    f"exit={result.returncode}, stderr={error_msg}"
                )
                return False, f"parity_failed: {error_msg or stdout_msg}"

        except subprocess.TimeoutExpired:
            logger.warning(
                f"[AutoPromotion] Parity validation timed out for {config_key}"
            )
            return False, "parity_timeout"
        except FileNotFoundError as e:
            logger.warning(f"[AutoPromotion] Parity script not executable: {e}")
            return True, f"parity_script_error: {e}"
        except OSError as e:
            logger.warning(f"[AutoPromotion] Parity validation OS error: {e}")
            return True, f"parity_os_error: {e}"

    async def _check_data_quality(
        self,
        board_type: str,
        num_players: int,
    ) -> tuple[bool, str]:
        """Check if training data quality is sufficient.

        Args:
            board_type: Board type (e.g., "hex8")
            num_players: Number of players (2, 3, or 4)

        Returns:
            Tuple of (passed, reason)
        """
        try:
            from pathlib import Path

            from app.db.game_replay import GameReplayDB

            # Find canonical database
            db_path = Path(f"data/games/canonical_{board_type}_{num_players}p.db")
            if not db_path.exists():
                return False, f"database_not_found: {db_path}"

            # Check game count using context manager
            with GameReplayDB(str(db_path)) as db:
                with db._get_conn() as conn:
                    cursor = conn.execute(
                        """
                        SELECT COUNT(*)
                        FROM games
                        WHERE game_status = 'completed'
                        """
                    )
                    game_count = cursor.fetchone()[0]

            if game_count < self.config.min_training_games:
                return False, (
                    f"insufficient_games: {game_count} < {self.config.min_training_games}"
                )

            # Try to get quality score if available
            try:
                from app.training.data_quality import (
                    DatabaseQualityChecker,
                    get_database_quality_score,
                )

                quality_score = get_database_quality_score(str(db_path))
                if quality_score < self.config.min_quality_score:
                    return False, (
                        f"low_quality_score: {quality_score:.2f} < {self.config.min_quality_score}"
                    )

            except (ImportError, AttributeError):
                # Quality score not available, skip this check
                logger.debug("[AutoPromotion] Quality score check not available")

            return True, f"quality_ok: {game_count} games"

        except Exception as e:
            logger.warning(f"[AutoPromotion] Quality check error: {e}")
            # Don't block on quality check errors
            return True, f"quality_check_error: {e}"

    async def _check_stability_gate(
        self,
        candidate: PromotionCandidate,
    ) -> tuple[bool, str]:
        """Check if candidate passes stability gate before promotion.

        December 2025: Prevents promotion of volatile models by verifying:
        1. Rating volatility is within acceptable bounds
        2. Model is not in a declining trend
        3. Rating has stabilized (not oscillating)

        Args:
            candidate: PromotionCandidate to check

        Returns:
            Tuple of (passed, reason) where reason explains the gate result
        """
        if not self.config.stability_gate_enabled:
            return True, "stability_gate_disabled"

        config_key = candidate.config_key

        # Parse board_type and num_players using canonical utility
        parsed = parse_config_key(config_key)
        if not parsed:
            logger.warning(f"[AutoPromotion] Cannot parse config_key: {config_key}")
            return True, "config_key_unparseable"
        board_type = parsed.board_type
        num_players = parsed.num_players

        try:
            from app.coordination.stability_heuristic import (
                StabilityLevel,
                assess_model_stability,
            )

            # Assess stability
            assessment = assess_model_stability(
                model_id="canonical",
                board_type=board_type,
                num_players=num_players,
            )

            # Log the assessment for debugging
            logger.info(
                f"[AutoPromotion] Stability assessment for {config_key}: "
                f"level={assessment.level.value}, volatility={assessment.volatility_score:.3f}, "
                f"slope={assessment.slope:.2f}, samples={assessment.sample_count}"
            )

            # Check volatility score
            if assessment.volatility_score > self.config.max_volatility_score:
                return False, (
                    f"high_volatility: {assessment.volatility_score:.2f} > "
                    f"{self.config.max_volatility_score}"
                )

            # Block declining models
            if assessment.level == StabilityLevel.DECLINING:
                return False, (
                    f"declining_trend: slope={assessment.slope:.2f} Elo/hour"
                )

            # Block highly volatile models
            if assessment.level == StabilityLevel.VOLATILE:
                actions = ", ".join(assessment.recommended_actions[:2])
                return False, f"volatile_model: {actions}"

            # Check if promotion is explicitly unsafe
            if not assessment.promotion_safe:
                return False, f"promotion_unsafe: {', '.join(assessment.recommended_actions[:1])}"

            return True, (
                f"stability_ok: level={assessment.level.value}, "
                f"volatility={assessment.volatility_score:.2f}"
            )

        except ImportError:
            logger.debug("[AutoPromotion] Stability heuristic not available")
            return True, "stability_check_unavailable"
        except Exception as e:
            logger.warning(f"[AutoPromotion] Stability check error: {e}")
            # Don't block on stability check errors
            return True, f"stability_check_error: {e}"

    async def _check_gauntlet_gate(
        self,
        candidate: PromotionCandidate,
    ) -> tuple[bool, str]:
        """Check if candidate passes gauntlet verification gate before promotion.

        January 5, 2026 (Session 17.34): Added gauntlet verification to ensure models
        prove their strength in head-to-head competition before promotion.

        The gate calculates a combined win rate from all gauntlet opponents and
        requires it to meet a minimum threshold (default: 55%).

        Args:
            candidate: PromotionCandidate to check

        Returns:
            Tuple of (passed, reason) where reason explains the gate result
        """
        if not self.config.gauntlet_verification_enabled:
            return True, "gauntlet_verification_disabled"

        # Calculate combined win rate from all gauntlet opponents
        total_games = 0
        total_wins = 0
        opponent_rates = []

        for opponent, win_rate in candidate.evaluation_results.items():
            games = candidate.evaluation_games.get(opponent, 0)
            if games > 0:
                total_games += games
                total_wins += int(win_rate * games)
                opponent_rates.append(f"{opponent}:{win_rate:.1%}")

        if total_games == 0:
            # No evaluation data - pass through (should be caught by other checks)
            return True, "no_gauntlet_data"

        combined_win_rate = total_wins / total_games

        if combined_win_rate >= self.config.min_gauntlet_win_rate:
            return True, (
                f"gauntlet_pass: {combined_win_rate:.1%} >= {self.config.min_gauntlet_win_rate:.0%} "
                f"(games={total_games}, opponents={', '.join(opponent_rates)})"
            )
        else:
            return False, (
                f"gauntlet_fail: {combined_win_rate:.1%} < {self.config.min_gauntlet_win_rate:.0%} "
                f"(games={total_games}, opponents={', '.join(opponent_rates)})"
            )

    async def _check_head_to_head_gate(
        self,
        candidate: PromotionCandidate,
    ) -> tuple[bool, str]:
        """Check if candidate beats the current canonical model head-to-head.

        January 10, 2026: CRITICAL gate to prevent model regression.
        Runs games between candidate and current canonical model.
        Candidate must win >= min_win_rate_vs_canonical to pass.

        This ensures new models are actually BETTER than the current best,
        not just better than baselines like random/heuristic.

        Args:
            candidate: PromotionCandidate to check

        Returns:
            Tuple of (passed, reason) where reason explains the gate result
        """
        if not self.config.head_to_head_enabled:
            return True, "head_to_head_disabled"

        # Check if we already have head-to-head results cached
        cached_result = candidate.evaluation_results.get("CANONICAL")
        if cached_result is not None:
            games = candidate.evaluation_games.get("CANONICAL", 0)
            if cached_result >= self.config.min_win_rate_vs_canonical:
                return True, (
                    f"head_to_head_pass: {cached_result:.1%} >= "
                    f"{self.config.min_win_rate_vs_canonical:.0%} (games={games})"
                )
            else:
                return False, (
                    f"head_to_head_fail: {cached_result:.1%} < "
                    f"{self.config.min_win_rate_vs_canonical:.0%} (games={games})"
                )

        # Need to run head-to-head games
        config_key = candidate.config_key
        try:
            # Parse board_type and num_players from config_key
            parts = config_key.rsplit("_", 1)
            if len(parts) != 2 or not parts[1].endswith("p"):
                return True, f"invalid_config_key: {config_key}"

            board_type = parts[0]
            num_players = int(parts[1][:-1])

            # Find canonical model path
            import os
            canonical_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "models", f"canonical_{config_key}.pth"
            )

            if not os.path.exists(canonical_path):
                logger.info(f"[AutoPromotion] No canonical model for {config_key}, skipping head-to-head")
                return True, f"no_canonical_model: {config_key}"

            # Run head-to-head games
            logger.info(
                f"[AutoPromotion] Running head-to-head: {candidate.model_path} vs {canonical_path}"
            )

            from app.training.game_gauntlet import run_model_vs_model
            result = await asyncio.to_thread(
                run_model_vs_model,
                candidate.model_path,
                canonical_path,
                board_type=board_type,
                num_players=num_players,
                num_games=self.config.head_to_head_games,
            )

            win_rate = result.get("win_rate", 0.0)
            games_played = result.get("games_played", 0)

            # Cache the result
            candidate.evaluation_results["CANONICAL"] = win_rate
            candidate.evaluation_games["CANONICAL"] = games_played

            if win_rate >= self.config.min_win_rate_vs_canonical:
                return True, (
                    f"head_to_head_pass: {win_rate:.1%} >= "
                    f"{self.config.min_win_rate_vs_canonical:.0%} (games={games_played})"
                )
            else:
                return False, (
                    f"head_to_head_fail: {win_rate:.1%} < "
                    f"{self.config.min_win_rate_vs_canonical:.0%} (games={games_played})"
                )

        except ImportError as e:
            logger.error(f"[AutoPromotion] Head-to-head gate unavailable (blocking promotion): {e}")
            return False, f"head_to_head_unavailable: {e}"
        except Exception as e:
            logger.error(f"[AutoPromotion] Head-to-head gate error (blocking promotion): {e}")
            # Feb 23, 2026: Changed from pass-through to block. Previously returned True
            # on errors, silently promoting models that couldn't be evaluated. This defeats
            # the purpose of the gate - if we can't prove improvement, don't promote.
            return False, f"head_to_head_error: {e}"

    async def _promote_model(self, candidate: PromotionCandidate) -> None:
        """Promote a model that passed evaluation.

        Args:
            candidate: PromotionCandidate to promote
        """
        config_key = candidate.config_key
        model_path = candidate.model_path

        # December 2025: Check quality gate before promotion
        quality_passed, quality_reason = await self._check_quality_gate(candidate)
        if not quality_passed:
            logger.warning(
                f"[AutoPromotion] {config_key} blocked by quality gate: {quality_reason}"
            )
            self._emit_promotion_rejected(
                candidate, gate="quality", reason=quality_reason,
            )
            await self._emit_promotion_failed(candidate, error=f"quality_gate: {quality_reason}")
            return

        # December 2025: Check stability gate to prevent promoting volatile models
        stability_passed, stability_reason = await self._check_stability_gate(candidate)
        if not stability_passed:
            logger.warning(
                f"[AutoPromotion] {config_key} blocked by stability gate: {stability_reason}"
            )
            self._emit_promotion_rejected(
                candidate, gate="stability", reason=stability_reason,
            )
            await self._emit_promotion_failed(candidate, error=f"stability_gate: {stability_reason}")
            return

        # January 5, 2026: Check gauntlet verification gate
        gauntlet_passed, gauntlet_reason = await self._check_gauntlet_gate(candidate)
        if not gauntlet_passed:
            logger.warning(
                f"[AutoPromotion] {config_key} blocked by gauntlet gate: {gauntlet_reason}"
            )
            self._emit_promotion_rejected(
                candidate, gate="gauntlet_win_rate", reason=gauntlet_reason,
            )
            await self._emit_promotion_failed(candidate, error=f"gauntlet_gate: {gauntlet_reason}")
            return

        # January 10, 2026: CRITICAL - Check head-to-head vs current canonical
        # This prevents model regression where new models are worse than current best
        head_to_head_passed, head_to_head_reason = await self._check_head_to_head_gate(candidate)
        if not head_to_head_passed:
            logger.warning(
                f"[AutoPromotion] {config_key} blocked by head-to-head gate: {head_to_head_reason}"
            )
            self._emit_promotion_rejected(
                candidate, gate="head_to_head", reason=head_to_head_reason,
            )
            await self._emit_promotion_failed(candidate, error=f"head_to_head_gate: {head_to_head_reason}")
            return

        if self.config.dry_run:
            logger.info(
                f"[AutoPromotion] DRY RUN: Would promote {config_key} "
                f"({model_path})"
            )
            return

        logger.info(f"[AutoPromotion] Promoting {config_key} ({model_path})")

        try:
            # Import promotion controller and decision types
            # Jan 3, 2026: Fixed API - execute_promotion() requires PromotionDecision
            from pathlib import Path as PathLib
            from app.training.promotion_controller import (
                PromotionController,
                PromotionDecision,
                PromotionType,
            )

            controller = PromotionController()

            # Create PromotionDecision with evaluation data
            model_stem = PathLib(model_path).stem if model_path else "unknown"
            decision = PromotionDecision(
                model_id=f"{config_key}_{model_stem}",
                promotion_type=PromotionType.ELO_IMPROVEMENT,
                should_promote=True,
                reason="auto_promotion_passed_evaluation",
                win_rate=candidate.evaluation_results.get("HEURISTIC", 0.0),
                current_elo=candidate.estimated_elo,
                games_played=sum(candidate.evaluation_games.values()),
                model_path=model_path,
            )

            # Note: execute_promotion is sync, run in thread to avoid blocking
            success = await asyncio.to_thread(
                controller.execute_promotion,
                decision,
                self.config.dry_run,
            )

            if success:
                candidate.last_promotion_time = time.time()
                # Dec 29, 2025: Reset failures and update Elo tracking on success
                candidate.consecutive_failures = 0
                candidate.previous_elo = candidate.estimated_elo  # Update baseline

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

                # January 3, 2026 (Sprint 16.2): Submit to hashgraph consensus for BFT audit trail
                await self._submit_to_hashgraph_consensus(candidate)

                # Dec 29, 2025: Emit unified PROMOTION_COMPLETED for curriculum
                await self._emit_promotion_completed(candidate, success=True)

                # Jan 3, 2026: Record lineage for tracking model history
                await self._record_lineage(candidate)

                # Clear results after successful emission
                candidate.consecutive_passes = 0
                candidate.evaluation_results.clear()
                candidate.evaluation_games.clear()
            else:
                # Dec 29, 2025: Track consecutive failures for curriculum regression
                candidate.consecutive_failures += 1

                logger.warning(
                    f"[AutoPromotion] Promotion failed for {config_key} "
                    f"(consecutive_failures={candidate.consecutive_failures})"
                )

                # Emit PROMOTION_FAILED event to trigger curriculum weight increase
                await self._emit_promotion_failed(candidate, error="Promotion validation failed")

                # Dec 29, 2025: Emit unified PROMOTION_COMPLETED for curriculum
                await self._emit_promotion_completed(candidate, success=False)

        except ImportError:
            # Fallback: just emit the event
            logger.warning(
                "[AutoPromotion] PromotionController not available, "
                "emitting event only"
            )
            await self._emit_promotion_event(candidate)
            await self._emit_promotion_completed(candidate, success=True)
        except Exception as e:  # noqa: BLE001
            # Dec 29, 2025: Track failures on exception too
            candidate.consecutive_failures += 1
            logger.error(
                f"[AutoPromotion] Promotion error for {config_key}: {e} "
                f"(consecutive_failures={candidate.consecutive_failures})"
            )

            # Emit PROMOTION_FAILED event on exception
            await self._emit_promotion_failed(candidate, error=str(e))

            # Dec 29, 2025: Emit unified PROMOTION_COMPLETED for curriculum
            await self._emit_promotion_completed(candidate, success=False)

    def _emit_promotion_rejected(
        self,
        candidate: PromotionCandidate,
        gate: str,
        reason: str,
        actual: float | None = None,
        threshold: float | None = None,
    ) -> None:
        """Emit PROMOTION_REJECTED event for observability.

        Feb 23, 2026: Provides structured observability into WHY models fail
        promotion. PipelineCompletenessMonitor subscribes to these events to
        detect systematic blocking (e.g., same gate failing 5+ times).

        Args:
            candidate: PromotionCandidate that was rejected
            gate: Name of the gate that rejected (e.g., "quality", "stability",
                  "gauntlet_win_rate", "head_to_head", "elo_improvement",
                  "velocity", "threshold")
            reason: Human-readable explanation of the rejection
            actual: Actual value that failed the gate (optional)
            threshold: Threshold value the actual was compared against (optional)
        """
        from app.coordination.event_router import safe_emit_event

        payload: dict[str, Any] = {
            "config_key": candidate.config_key,
            "gate": gate,
            "reason": reason,
            "model_path": candidate.model_path,
            "estimated_elo": candidate.estimated_elo,
            "timestamp": time.time(),
        }
        if actual is not None:
            payload["actual"] = actual
        if threshold is not None:
            payload["threshold"] = threshold

        safe_emit_event(
            "PROMOTION_REJECTED",
            payload,
            source="AutoPromotionDaemon",
        )

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

    async def _submit_to_hashgraph_consensus(
        self,
        candidate: "PromotionCandidate",
    ) -> None:
        """Submit promotion decision to hashgraph consensus for BFT audit trail.

        January 3, 2026 (Sprint 16.2): Records promotion decisions in hashgraph DAG.

        This creates an immutable record of promotion decisions that:
        - Provides audit trail for all promotions
        - Enables BFT enforcement in future (require 2/3+ approval)
        - Detects fork attempts (duplicate conflicting promotions)
        - Supports rollback with cryptographic proofs

        Args:
            candidate: PromotionCandidate that was promoted
        """
        if not HAS_HASHGRAPH_CONSENSUS:
            logger.debug("[AutoPromotion] Hashgraph consensus not available")
            return

        try:
            import hashlib
            import socket

            # Compute model hash for consensus tracking
            model_hash = hashlib.sha256(candidate.model_path.encode()).hexdigest()[:16]
            node_id = socket.gethostname()

            # Get promotion consensus manager
            consensus = get_promotion_consensus_manager()
            if consensus is None:
                logger.debug("[AutoPromotion] Promotion consensus manager not initialized")
                return

            # Create evaluation evidence for the proposal
            evidence = EvaluationEvidence(
                win_rate=candidate.evaluation_results.get("HEURISTIC", 0.0),
                elo=candidate.estimated_elo,
                games_played=sum(candidate.evaluation_games.values()),
                vs_baselines={
                    "RANDOM": candidate.evaluation_results.get("RANDOM", 0.0),
                    "HEURISTIC": candidate.evaluation_results.get("HEURISTIC", 0.0),
                },
            )

            # Propose promotion to hashgraph
            proposal = await consensus.propose_promotion(
                model_hash=model_hash,
                config_key=candidate.config_key,
                proposer=node_id,
                evidence=evidence,
            )

            # Auto-approve since this is a local promotion (BFT enforcement is future work)
            await consensus.vote_on_proposal(
                proposal_id=proposal.proposal_id,
                voter=node_id,
                approve=True,
            )

            logger.info(
                f"[AutoPromotion] Submitted to hashgraph consensus: "
                f"config={candidate.config_key}, model={model_hash[:8]}, "
                f"proposal={proposal.proposal_id[:8]}"
            )

            # Emit event for monitoring (safe_emit_event handles errors internally)
            from app.coordination.event_router import safe_emit_event

            safe_emit_event(
                "PROMOTION_CONSENSUS_APPROVED",
                {
                    "config_key": candidate.config_key,
                    "model_path": candidate.model_path,
                    "model_hash": model_hash,
                    "proposer_node": node_id,
                    "proposal_id": proposal.proposal_id,
                    "win_rate": evidence.win_rate,
                    "elo": evidence.elo,
                },
                context="AutoPromotionDaemon.submit_to_hashgraph",
            )

        except Exception as e:  # noqa: BLE001
            # Don't fail promotion just because consensus recording failed
            logger.warning(f"[AutoPromotion] Failed to submit to hashgraph: {e}")

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

    async def _emit_promotion_completed(
        self,
        candidate: PromotionCandidate,
        success: bool,
    ) -> None:
        """Emit unified PROMOTION_COMPLETED event for curriculum feedback.

        Dec 29, 2025: Provides a single event for curriculum_integration to
        subscribe to, containing all information needed for curriculum
        advancement or regression decisions.

        Args:
            candidate: PromotionCandidate that was evaluated
            success: Whether promotion succeeded
        """
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            if router:
                # Calculate Elo change
                elo_change = candidate.estimated_elo - candidate.previous_elo

                await router.publish(
                    event_type="PROMOTION_COMPLETED",
                    payload={
                        "config_key": candidate.config_key,
                        "success": success,
                        "elo_change": elo_change,
                        "estimated_elo": candidate.estimated_elo,
                        "previous_elo": candidate.previous_elo,
                        "consecutive_failures": candidate.consecutive_failures,
                        "consecutive_passes": candidate.consecutive_passes,
                        "vs_random": candidate.evaluation_results.get("RANDOM", 0.0),
                        "vs_heuristic": candidate.evaluation_results.get("HEURISTIC", 0.0),
                        "timestamp": time.time(),
                    },
                    source="auto_promotion_daemon",
                )
                logger.info(
                    f"[AutoPromotion] Emitted PROMOTION_COMPLETED for {candidate.config_key}: "
                    f"success={success}, elo_change={elo_change:+.0f}"
                )
        except Exception as e:  # noqa: BLE001
            logger.error(f"[AutoPromotion] Failed to emit PROMOTION_COMPLETED event: {e}")

    async def _record_lineage(self, candidate: PromotionCandidate) -> None:
        """Record promoted model in the lineage database.

        Jan 3, 2026: Tracks model history for debugging performance regressions
        and understanding model evolution.

        Args:
            candidate: PromotionCandidate that was promoted
        """
        try:
            import asyncio
            from pathlib import Path

            from scripts.model_lineage import register_model, update_performance

            # Parse config key
            parts = candidate.config_key.rsplit("_", 1)
            if len(parts) == 2:
                board_type = parts[0]
                num_players = int(parts[1].rstrip("p"))
            else:
                board_type = candidate.config_key
                num_players = 2

            # Register model in lineage database
            model_id = await asyncio.to_thread(
                register_model,
                model_path=candidate.model_path,
                board_type=board_type,
                num_players=num_players,
                architecture="unknown",  # Could be enhanced to detect
                tags=["auto_promoted"],
            )

            # Record performance metrics
            vs_random = candidate.evaluation_results.get("RANDOM", 0.0)
            vs_heuristic = candidate.evaluation_results.get("HEURISTIC", 0.0)

            if vs_random > 0:
                await asyncio.to_thread(
                    update_performance,
                    model_id,
                    "vs_random",
                    vs_random,
                    context="auto_promotion",
                )
            if vs_heuristic > 0:
                await asyncio.to_thread(
                    update_performance,
                    model_id,
                    "vs_heuristic",
                    vs_heuristic,
                    context="auto_promotion",
                )
            if candidate.estimated_elo > 0:
                await asyncio.to_thread(
                    update_performance,
                    model_id,
                    "elo",
                    candidate.estimated_elo,
                    context="auto_promotion",
                )

            logger.info(
                f"[AutoPromotion] Recorded lineage for {candidate.config_key}: "
                f"model_id={model_id}"
            )
        except ImportError:
            logger.debug("[AutoPromotion] model_lineage module not available, skipping lineage")
        except (OSError, ValueError) as e:
            logger.warning(f"[AutoPromotion] Failed to record lineage: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get daemon status."""
        return {
            "running": self._running,
            "subscribed": self._event_subscribed,
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

    def health_check(self) -> HealthCheckResult:
        """Check daemon health.

        January 2026 (Sprint 12.2): Enhanced to use HandlerBase health details
        while preserving domain-specific checks.

        Returns:
            HealthCheckResult with status and details
        """
        # Get base health check from HandlerBase
        base_health = super().health_check()

        # Add domain-specific check: subscription status
        if self._running and not self._event_subscribed:
            return HealthCheckResult(
                healthy=True,  # Still healthy, just degraded
                status=CoordinatorStatus.DEGRADED,
                message="AutoPromotion daemon not subscribed to events (retry pending)",
                details={**base_health.details, **self.get_status()},
            )

        # Enhance message with promotion count
        if base_health.healthy:
            return HealthCheckResult(
                healthy=True,
                status=base_health.status,
                message=f"AutoPromotion daemon running ({len(self._promotion_history)} promotions)",
                details={**base_health.details, **self.get_status()},
            )

        return base_health

    def get_promotion_status(self) -> dict:
        """Return detailed promotion candidate status for debugging.

        January 7, 2026: Added for 48h autonomous operation visibility.
        Provides detailed candidate state for monitoring and debugging.

        Returns:
            Dict with:
                - candidates: Per-config candidate state
                - last_promotion_time: When last promotion occurred
                - thresholds: Current promotion thresholds
        """
        from datetime import datetime

        candidate_details = {}
        for config_key, candidate in self._candidates.items():
            candidate_details[config_key] = {
                "model_path": candidate.model_path,
                "consecutive_passes": candidate.consecutive_passes,
                "vs_random": candidate.evaluation_results.get("RANDOM", 0.0),
                "vs_heuristic": candidate.evaluation_results.get("HEURISTIC", 0.0),
                "training_game_count": candidate.training_game_count,
                "beats_current_best": candidate.beats_current_best,
                "evaluation_games": candidate.evaluation_games,
                "last_evaluation_time": candidate.last_evaluation_time,
                "last_promotion_time": candidate.last_promotion_time,
            }

        # Get the most recent promotion time across all candidates
        last_promotion = None
        if self._promotion_history:
            last_promotion = self._promotion_history[-1]

        return {
            "candidates": candidate_details,
            "last_promotion": last_promotion,
            "promotion_history_count": len(self._promotion_history),
            "thresholds": {
                # Jan 2026: Use correct config attributes (was referencing non-existent attrs)
                "min_games_vs_random": self.config.min_games_vs_random,
                "min_games_vs_heuristic": self.config.min_games_vs_heuristic,
                "min_gauntlet_win_rate": self.config.min_gauntlet_win_rate,
                "consecutive_passes_required": self.config.consecutive_passes_required,
            },
            "enabled": self.config.enabled,
            "dry_run": self.config.dry_run,
        }


# =============================================================================
# Module-Level Singleton Accessors (January 2026: Delegates to HandlerBase)
# =============================================================================


def get_auto_promotion_daemon() -> AutoPromotionDaemon:
    """Get the singleton AutoPromotionDaemon instance.

    January 2026: Now delegates to HandlerBase.get_instance() for
    thread-safe singleton management.
    """
    return AutoPromotionDaemon.get_instance()


def reset_auto_promotion_daemon() -> None:
    """Reset the singleton (for testing).

    January 2026: Now delegates to HandlerBase.reset_instance().
    """
    AutoPromotionDaemon.reset_instance()
