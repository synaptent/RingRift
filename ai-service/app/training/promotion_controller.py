"""Unified Promotion Controller for RingRift AI.

Provides a single entry point for all model promotion decisions, consolidating
logic from multiple modules:
- app/training/model_registry.py - Model lifecycle tracking
- app/training/tier_promotion_registry.py - Tier-based promotion for difficulty ladder
- app/integration/model_lifecycle.py - Full lifecycle management with evaluation

This controller does NOT replace those modules - it delegates to them based on
the promotion type. Use this controller for NEW code to ensure consistent
promotion criteria across the system.

Usage:
    from app.training.promotion_controller import PromotionController, PromotionType

    controller = PromotionController()

    # Check if a model should be promoted
    decision = controller.evaluate_promotion(
        model_id="model_v42",
        board_type="square8",
        num_players=2,
        promotion_type=PromotionType.PRODUCTION,
    )

    if decision.should_promote:
        controller.execute_promotion(decision)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# Import unified signals for regression detection
try:
    from app.training.unified_signals import (
        TrainingSignals,
        get_signal_computer,
    )
    HAS_UNIFIED_SIGNALS = True
except ImportError:
    HAS_UNIFIED_SIGNALS = False
    get_signal_computer = None

# Import centralized timeout constants (December 2025)
try:
    from app.config.thresholds import (
        URLOPEN_SHORT_TIMEOUT,
        URLOPEN_TIMEOUT,
        ELO_TARGET_ALL_CONFIGS,
        get_elo_target,
        get_elo_gap,
        is_target_met,
    )
    HAS_ELO_TARGETS = True
except ImportError:
    URLOPEN_SHORT_TIMEOUT = 5
    URLOPEN_TIMEOUT = 10
    ELO_TARGET_ALL_CONFIGS = 2000.0
    HAS_ELO_TARGETS = False

# Jan 2026: Import Wilson score for 48h autonomous operation
try:
    from app.training.significance import wilson_lower_bound
    HAS_WILSON = True
except ImportError:
    HAS_WILSON = False
    wilson_lower_bound = None
    get_elo_target = lambda k: 2000.0
    get_elo_gap = lambda k, e: max(0.0, 2000.0 - e)
    is_target_met = lambda k, e: e >= 2000.0

# Event bus integration for auto-promotion on Elo changes (December 2025)
try:
    from app.coordination.event_router import get_router
    from app.coordination.event_router import DataEventType
    HAS_EVENT_BUS = True
except ImportError:
    HAS_EVENT_BUS = False
    DataEventType = None
    get_router = None

# Distributed locking for cross-node coordination (December 2025)
try:
    from app.training.locking_integration import TrainingLocks
    _HAS_DISTRIBUTED_LOCKS = True
except ImportError:
    _HAS_DISTRIBUTED_LOCKS = False
    TrainingLocks = None

# Improvement optimizer for positive feedback acceleration (December 2025)
try:
    from app.training.improvement_optimizer import (
        get_improvement_optimizer,
        ImprovementOptimizer,
    )
    _HAS_IMPROVEMENT_OPTIMIZER = True
except ImportError:
    _HAS_IMPROVEMENT_OPTIMIZER = False
    get_improvement_optimizer = None
    ImprovementOptimizer = None

logger = logging.getLogger(__name__)


class PromotionType(Enum):
    """Types of promotion decisions."""
    STAGING = "staging"          # Development → Staging
    PRODUCTION = "production"    # Staging → Production
    TIER = "tier"                # Tier ladder promotion (D1→D2, etc.)
    CHAMPION = "champion"        # Tournament champion promotion
    ROLLBACK = "rollback"        # Rollback to previous version
    ELO_IMPROVEMENT = "elo_improvement"  # Auto-promotion after Elo gain


@dataclass
class PromotionCriteria:
    """Criteria for promotion evaluation.

    These are the canonical thresholds - sourced from unified_config.py.
    """
    min_elo_improvement: float = 25.0
    # Jan 2026: Increased from 50 to 100 for 48h autonomous operation
    # Ensures statistical significance before promotion
    min_games_played: int = 100
    min_win_rate: float = 0.52
    max_value_mse_degradation: float = 0.05
    confidence_threshold: float = 0.95

    # Tier-specific
    tier_elo_threshold: float | None = None
    tier_games_required: int = 100

    # Absolute Elo targets (December 2025)
    # If enabled, models must meet both relative AND absolute Elo criteria.
    # Default is False: most promotion decisions are relative (vs a baseline) and
    # should not be blocked by global target attainment.
    require_absolute_elo_target: bool = False
    absolute_elo_target: float = ELO_TARGET_ALL_CONFIGS  # Default: 2000.0

    # Jan 2026: Wilson CI requirement for 48h autonomous operation
    # Requires Wilson 95% CI lower bound > 50% to ensure statistical confidence
    require_wilson_ci_above_50: bool = True


@dataclass
class PromotionDecision:
    """Result of a promotion evaluation."""
    model_id: str
    promotion_type: PromotionType
    should_promote: bool
    reason: str

    # Evaluation metrics
    current_elo: float | None = None
    elo_improvement: float | None = None
    games_played: int = 0
    win_rate: float | None = None
    confidence: float | None = None

    # Model path (Feb 2026: used by ELO_IMPROVEMENT to find candidate file)
    model_path: str | None = None

    # For tier promotions
    current_tier: str | None = None
    target_tier: str | None = None

    # Absolute Elo target tracking (December 2025)
    elo_target: float | None = None
    elo_gap: float | None = None
    target_met: bool = False
    blocked_by_target: bool = False  # True if would promote but for target gate

    # Metadata
    evaluated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    criteria_used: PromotionCriteria | None = None
    # Jan 12, 2026: Added harness_type for multi-harness tracking
    harness_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "promotion_type": self.promotion_type.value,
            "should_promote": self.should_promote,
            "reason": self.reason,
            "current_elo": self.current_elo,
            "elo_improvement": self.elo_improvement,
            "games_played": self.games_played,
            "win_rate": self.win_rate,
            "confidence": self.confidence,
            "current_tier": self.current_tier,
            "target_tier": self.target_tier,
            "elo_target": self.elo_target,
            "elo_gap": self.elo_gap,
            "target_met": self.target_met,
            "blocked_by_target": self.blocked_by_target,
            "evaluated_at": self.evaluated_at,
            "harness_type": self.harness_type,
        }


class PromotionController:
    """Unified controller for all model promotion decisions.

    Consolidates promotion logic from multiple systems while preserving
    backward compatibility with existing code.
    """

    def __init__(
        self,
        criteria: PromotionCriteria | None = None,
        elo_service: Any | None = None,
        model_registry: Any | None = None,
    ):
        """Initialize the promotion controller.

        Args:
            criteria: Custom promotion criteria (uses defaults if None)
            elo_service: EloService instance (lazy-loaded if None)
            model_registry: ModelRegistry instance (lazy-loaded if None)
        """
        self.criteria = criteria or PromotionCriteria()
        self._elo_service = elo_service
        self._model_registry = model_registry
        self._lifecycle_manager = None
        self._tier_registry = None
        # Unified signal computer for regression detection
        self._signal_computer = get_signal_computer() if HAS_UNIFIED_SIGNALS else None
        # Event subscription state (December 2025)
        self._event_subscribed = False
        self._pending_promotion_checks: dict[str, float] = {}  # model_id -> elo_delta

        # Auto-wire to event bus for ELO-triggered promotion checks (December 2025)
        # This enables automatic promotion evaluation when ELO ratings change significantly
        self.setup_event_subscriptions()

    @property
    def elo_service(self):
        """Lazy-load EloService."""
        if self._elo_service is None:
            try:
                from app.training.elo_service import get_elo_service
                self._elo_service = get_elo_service()
            except ImportError:
                logger.warning("EloService not available")
        return self._elo_service

    @property
    def model_registry(self):
        """Lazy-load ModelRegistry."""
        if self._model_registry is None:
            try:
                from app.training.model_registry import ModelRegistry
                self._model_registry = ModelRegistry()
            except ImportError:
                logger.warning("ModelRegistry not available")
        return self._model_registry

    @property
    def lifecycle_manager(self):
        """Lazy-load ModelLifecycleManager."""
        if self._lifecycle_manager is None:
            try:
                from app.integration.model_lifecycle import (
                    LifecycleConfig,
                    ModelLifecycleManager,
                )
                self._lifecycle_manager = ModelLifecycleManager(LifecycleConfig())
            except ImportError:
                logger.warning("ModelLifecycleManager not available")
        return self._lifecycle_manager

    def setup_event_subscriptions(self) -> bool:
        """Subscribe to ELO events for automatic promotion checks (December 2025).

        When ELO_SIGNIFICANT_CHANGE is received, queues a promotion check
        for the improved model.

        Returns:
            True if subscription was successful, False otherwise
        """
        if not HAS_EVENT_BUS:
            logger.debug("Event bus not available for promotion controller")
            return False

        if self._event_subscribed:
            return True

        try:
            router = get_router()

            async def on_elo_significant_change(event):
                """Handle significant Elo changes for auto-promotion evaluation."""
                payload = event.payload if hasattr(event, 'payload') else {}
                model_id = payload.get("model_id", "")
                elo_delta = payload.get("elo_delta", 0.0)
                new_elo = payload.get("new_elo", 0.0)

                # Only consider positive Elo changes for promotion
                if elo_delta >= self.criteria.min_elo_improvement:
                    logger.info(
                        f"[PromotionController] Significant Elo gain detected: "
                        f"{model_id} +{elo_delta:.1f} (now {new_elo:.0f}), "
                        f"queuing promotion check"
                    )
                    self._pending_promotion_checks[model_id] = elo_delta

            router.subscribe(DataEventType.ELO_SIGNIFICANT_CHANGE.value, on_elo_significant_change)

            # Subscribe to EVALUATION_COMPLETED for automatic promotion checks (December 2025)
            async def on_evaluation_completed(event):
                """Handle evaluation completion for auto-promotion checks."""
                payload = event.payload if hasattr(event, 'payload') else {}
                model_id = payload.get("model_id", payload.get("model_path", ""))
                board_type = payload.get("board_type", "square8")
                num_players = payload.get("num_players", 2)

                # Check win rate vs heuristic (primary promotion criterion)
                win_rate_vs_heuristic = payload.get("win_rate_vs_heuristic", 0.0)

                # Also check result dict structure
                results = payload.get("results", {})
                for opponent, data in results.items():
                    if "heuristic" in opponent.lower():
                        win_rate_vs_heuristic = max(
                            win_rate_vs_heuristic,
                            data.get("win_rate", 0.0)
                        )

                # If model beats heuristic 55%+ of the time, emit promotion candidate event
                if win_rate_vs_heuristic >= 0.55:
                    logger.info(
                        f"[PromotionController] Promotion candidate detected: "
                        f"{model_id} with {win_rate_vs_heuristic:.1%} vs heuristic"
                    )
                    # Emit PROMOTION_CANDIDATE for downstream handlers
                    try:
                        from app.coordination.event_emission_helpers import safe_emit_event

                        safe_emit_event(
                            "PROMOTION_CANDIDATE",
                            {
                                "model_id": model_id,
                                "board_type": board_type,
                                "num_players": num_players,
                                "win_rate_vs_heuristic": win_rate_vs_heuristic,
                                "source": "PromotionController",
                            },
                            context="promotion_controller",
                        )
                    except Exception as e:
                        logger.debug(f"Failed to emit PROMOTION_CANDIDATE: {e}")
                    # Queue for promotion evaluation
                    self._pending_promotion_checks[model_id] = win_rate_vs_heuristic

            router.subscribe(DataEventType.EVALUATION_COMPLETED.value, on_evaluation_completed)

            self._event_subscribed = True
            logger.info(
                "[PromotionController] Subscribed to ELO_SIGNIFICANT_CHANGE and "
                "EVALUATION_COMPLETED events"
            )
            return True

        except Exception as e:
            logger.warning(f"[PromotionController] Failed to subscribe to events: {e}")
            return False

    def get_pending_promotion_checks(self) -> dict[str, float]:
        """Get models with pending promotion checks from Elo events.

        Returns:
            Dict mapping model_id to elo_delta for models that need promotion evaluation
        """
        return dict(self._pending_promotion_checks)

    def clear_pending_check(self, model_id: str) -> None:
        """Clear a pending promotion check after evaluation."""
        self._pending_promotion_checks.pop(model_id, None)

    def _check_multi_harness_gate(
        self,
        model_id: str,
        board_type: str,
        num_players: int,
        min_harnesses: int = 2,
    ) -> tuple[bool, str, list[str]]:
        """Check if model has been evaluated under multiple harnesses.

        January 2026: Require multi-harness evaluation before production promotion.
        This ensures models are robust across different evaluation methods, not just
        optimized for a single harness.

        Args:
            model_id: Model to check
            board_type: Board type for evaluation
            num_players: Number of players
            min_harnesses: Minimum number of distinct harnesses required (default: 2)

        Returns:
            (passes, reason, harnesses_tested) - Tuple with pass status, explanation,
            and list of harness types that have been tested
        """
        harnesses_tested: list[str] = []

        if self.elo_service is None:
            return False, "Elo service not available", harnesses_tested

        try:
            # Query match_history for distinct harness types for this model
            conn = self.elo_service._get_connection()
            cursor = conn.execute("""
                SELECT DISTINCT harness_type
                FROM match_history
                WHERE (player1_id = ? OR player2_id = ?)
                AND board_type = ?
                AND num_players = ?
                AND harness_type IS NOT NULL
                AND harness_type != ''
            """, (model_id, model_id, board_type, num_players))

            for row in cursor:
                if row[0]:
                    harnesses_tested.append(row[0])

        except Exception as e:
            logger.warning(f"Multi-harness gate check failed: {e}")
            return False, f"Failed to query harness data: {e}", harnesses_tested

        if len(harnesses_tested) < min_harnesses:
            return (
                False,
                f"Only {len(harnesses_tested)} harness(es) tested ({harnesses_tested}), "
                f"need >= {min_harnesses}. Run multi_harness_gauntlet first.",
                harnesses_tested,
            )

        return (
            True,
            f"Passed with {len(harnesses_tested)} harnesses: {', '.join(harnesses_tested)}",
            harnesses_tested,
        )

    def evaluate_promotion(
        self,
        model_id: str,
        board_type: str = "square8",
        num_players: int = 2,
        promotion_type: PromotionType = PromotionType.PRODUCTION,
        baseline_model_id: str | None = None,
        harness_type: str | None = None,
        **kwargs,
    ) -> PromotionDecision:
        """Evaluate whether a model should be promoted.

        Args:
            model_id: ID of the model to evaluate
            board_type: Board type for evaluation
            num_players: Number of players
            promotion_type: Type of promotion to evaluate
            baseline_model_id: Model to compare against (for PRODUCTION/CHAMPION)
            harness_type: AI harness used for evaluation (e.g., "gumbel_mcts", "minimax")
            **kwargs: Additional arguments for specific promotion types

        Returns:
            PromotionDecision with evaluation results
        """
        if promotion_type == PromotionType.TIER:
            return self._evaluate_tier_promotion(
                model_id, board_type, num_players, **kwargs
            )
        elif promotion_type == PromotionType.CHAMPION:
            return self._evaluate_champion_promotion(
                model_id, board_type, num_players, baseline_model_id, **kwargs
            )
        elif promotion_type == PromotionType.ROLLBACK:
            return self._evaluate_rollback(
                model_id, board_type, num_players, **kwargs
            )
        else:
            return self._evaluate_standard_promotion(
                model_id, board_type, num_players, promotion_type, baseline_model_id
            )

    def _evaluate_standard_promotion(
        self,
        model_id: str,
        board_type: str,
        num_players: int,
        promotion_type: PromotionType,
        baseline_model_id: str | None,
    ) -> PromotionDecision:
        """Evaluate standard staging/production promotion."""
        # Get current Elo and stats
        current_elo = None
        games_played = 0
        win_rate = None
        elo_improvement = None
        harness_type: str | None = None  # Jan 2026: Initialize for multi-harness tracking

        if self.elo_service:
            try:
                rating = self.elo_service.get_rating(model_id, board_type, num_players)
                if rating:
                    current_elo = rating.rating
                    games_played = rating.games_played
                    win_rate = rating.win_rate
            except Exception as e:
                logger.warning(f"Failed to get Elo for {model_id}: {e}")

        # Get baseline Elo for comparison
        baseline_elo = None
        if baseline_model_id and self.elo_service:
            try:
                baseline = self.elo_service.get_rating(
                    baseline_model_id, board_type, num_players
                )
                if baseline:
                    baseline_elo = baseline.rating
                    if current_elo is not None:
                        elo_improvement = current_elo - baseline_elo
            except Exception as e:
                logger.warning(f"Failed to get baseline Elo: {e}")

        # Calculate absolute Elo target status (December 2025)
        config_key = f"{board_type}_{num_players}p"
        elo_target = get_elo_target(config_key)
        elo_gap_value = get_elo_gap(config_key, current_elo or 0)
        target_met = is_target_met(config_key, current_elo or 0)

        # Check promotion criteria
        should_promote = False
        reason = ""
        blocked_by_target = False

        if games_played < self.criteria.min_games_played:
            reason = f"Insufficient games ({games_played} < {self.criteria.min_games_played})"
        elif elo_improvement is not None and elo_improvement < self.criteria.min_elo_improvement:
            reason = f"Insufficient Elo improvement ({elo_improvement:.1f} < {self.criteria.min_elo_improvement})"
        elif win_rate is not None and win_rate < self.criteria.min_win_rate:
            reason = f"Win rate too low ({win_rate:.2%} < {self.criteria.min_win_rate:.2%})"
        elif (
            self.criteria.require_wilson_ci_above_50
            and HAS_WILSON
            and wilson_lower_bound is not None
            and win_rate is not None
            and games_played > 0
        ):
            # Jan 2026: Wilson CI check for 48h autonomous operation
            # Require 95% CI lower bound > 50% for statistical confidence
            wilson_ci_lower = wilson_lower_bound(
                wins=int(win_rate * games_played),
                total=games_played,
                confidence=0.95,
            )
            if wilson_ci_lower < 0.50:
                reason = (
                    f"Wilson 95% CI lower bound too low ({wilson_ci_lower:.2%} < 50%). "
                    f"Win rate {win_rate:.2%} over {games_played} games not statistically significant."
                )
            elif self.criteria.require_absolute_elo_target and not target_met:
                # Wilson passed but absolute target not met
                blocked_by_target = True
                reason = (
                    f"Below absolute Elo target: {current_elo or 0:.0f} < {elo_target:.0f} "
                    f"(gap: {elo_gap_value:.0f} Elo). Wilson CI passed ({wilson_ci_lower:.2%})."
                )
            else:
                should_promote = True
                reason = (
                    f"Meets all criteria: Elo +{elo_improvement or 0:.1f}, {games_played} games, "
                    f"Wilson CI {wilson_ci_lower:.2%} >= 50%"
                )
        elif self.criteria.require_absolute_elo_target and not target_met:
            # Model meets relative criteria but not absolute target
            blocked_by_target = True
            reason = (
                f"Below absolute Elo target: {current_elo or 0:.0f} < {elo_target:.0f} "
                f"(gap: {elo_gap_value:.0f} Elo). "
                f"Meets relative criteria (Elo +{elo_improvement or 0:.1f}, {games_played} games)"
            )
            logger.info(
                f"Model {model_id} blocked by absolute Elo target: "
                f"{(current_elo or 0):.0f} < {elo_target:.0f} for {config_key}"
            )
        else:
            should_promote = True
            if self.criteria.require_absolute_elo_target:
                reason = (
                    f"Meets all criteria: Elo +{elo_improvement or 0:.1f}, {games_played} games, "
                    f"absolute Elo {current_elo or 0:.0f} >= {elo_target:.0f}"
                )
            else:
                # Relative-only promotion: baseline improvement + enough games + win rate.
                reason = (
                    f"Meets all criteria: Elo +{elo_improvement or 0:.1f}, {games_played} games"
                )

        # January 2026: Multi-harness gate for PRODUCTION promotions
        # Require evaluation under multiple harnesses to ensure robustness
        harnesses_tested: list[str] = []
        if should_promote and promotion_type == PromotionType.PRODUCTION:
            gate_passes, gate_reason, harnesses_tested = self._check_multi_harness_gate(
                model_id, board_type, num_players, min_harnesses=2
            )
            if not gate_passes:
                should_promote = False
                reason = f"Multi-harness gate failed: {gate_reason}"
                logger.info(
                    f"Model {model_id} blocked by multi-harness gate: {gate_reason}"
                )
            else:
                # Gate passed - include harness info in reason
                reason = f"{reason} | Multi-harness: {', '.join(harnesses_tested)}"
                harness_type = ", ".join(harnesses_tested) if harnesses_tested else None

        decision = PromotionDecision(
            model_id=model_id,
            promotion_type=promotion_type,
            should_promote=should_promote,
            reason=reason,
            current_elo=current_elo,
            elo_improvement=elo_improvement,
            games_played=games_played,
            win_rate=win_rate,
            elo_target=elo_target,
            elo_gap=elo_gap_value,
            target_met=target_met,
            blocked_by_target=blocked_by_target,
            criteria_used=self.criteria,
            harness_type=harness_type,
        )

        # Emit metrics
        self._emit_decision_metrics(decision)

        return decision

    def _emit_decision_metrics(self, decision: PromotionDecision) -> None:
        """Emit Prometheus metrics for a promotion decision."""
        try:
            from app.metrics import record_promotion_decision
            record_promotion_decision(
                promotion_type=decision.promotion_type.value,
                approved=decision.should_promote,
                elo_improvement=decision.elo_improvement,
            )
        except ImportError:
            pass

    def _evaluate_tier_promotion(
        self,
        model_id: str,
        board_type: str,
        num_players: int,
        current_tier: str | None = None,
        target_tier: str | None = None,
        **kwargs,
    ) -> PromotionDecision:
        """Evaluate tier-based promotion for difficulty ladder."""
        # Load tier registry
        try:
            from app.config.ladder_config import get_tier_threshold

            # Get current model stats
            current_elo = None
            games_played = 0

            if self.elo_service:
                rating = self.elo_service.get_rating(model_id, board_type, num_players)
                if rating:
                    current_elo = rating.rating
                    games_played = rating.games_played

            # Check against tier threshold
            if target_tier:
                threshold = get_tier_threshold(target_tier)
                if threshold and current_elo is not None:
                    if current_elo >= threshold and games_played >= self.criteria.tier_games_required:
                        return PromotionDecision(
                            model_id=model_id,
                            promotion_type=PromotionType.TIER,
                            should_promote=True,
                            reason=f"Elo {current_elo:.0f} >= tier {target_tier} threshold {threshold}",
                            current_elo=current_elo,
                            games_played=games_played,
                            current_tier=current_tier,
                            target_tier=target_tier,
                        )
                    else:
                        return PromotionDecision(
                            model_id=model_id,
                            promotion_type=PromotionType.TIER,
                            should_promote=False,
                            reason=f"Elo {current_elo or 0:.0f} < tier {target_tier} threshold {threshold}",
                            current_elo=current_elo,
                            games_played=games_played,
                            current_tier=current_tier,
                            target_tier=target_tier,
                        )
        except ImportError as e:
            logger.warning(f"Tier promotion modules not available: {e}")

        return PromotionDecision(
            model_id=model_id,
            promotion_type=PromotionType.TIER,
            should_promote=False,
            reason="Tier promotion evaluation failed - modules not available",
        )

    def _evaluate_champion_promotion(
        self,
        model_id: str,
        board_type: str,
        num_players: int,
        baseline_model_id: str | None,
        tournament_results: dict | None = None,
        **kwargs,
    ) -> PromotionDecision:
        """Evaluate champion promotion after tournament win."""
        # Use lifecycle manager if available
        if self.lifecycle_manager:
            try:
                from app.integration.model_lifecycle import EvaluationResult

                result = EvaluationResult(
                    model_id=model_id,
                    version=0,
                    games_played=tournament_results.get("games_played", 0) if tournament_results else 0,
                    win_rate=tournament_results.get("win_rate") if tournament_results else None,
                )

                # Use the lifecycle manager's promotion gate
                decision, reason = self.lifecycle_manager.promotion_gate.evaluate_for_staging(result)

                return PromotionDecision(
                    model_id=model_id,
                    promotion_type=PromotionType.CHAMPION,
                    should_promote=decision.value == "promote",
                    reason=reason,
                    games_played=result.games_played,
                    win_rate=result.win_rate,
                )
            except Exception as e:
                logger.warning(f"Lifecycle manager evaluation failed: {e}")

        # Fallback to standard evaluation
        return self._evaluate_standard_promotion(
            model_id, board_type, num_players, PromotionType.CHAMPION, baseline_model_id
        )

    def _evaluate_rollback(
        self,
        model_id: str,
        board_type: str,
        num_players: int,
        regression_threshold: float | None = None,
        current_games: int | None = None,
        **kwargs,
    ) -> PromotionDecision:
        """Evaluate whether to rollback to a previous model.

        Uses unified signals when available for faster regression detection.
        """
        threshold = regression_threshold or -30.0
        config_key = f"{board_type}_{num_players}p"

        # Try unified signals first (cached, fast)
        if self._signal_computer is not None and current_games is not None:
            try:
                # Get current Elo from elo service
                current_elo = None
                if self.elo_service:
                    rating = self.elo_service.get_rating(model_id, board_type, num_players)
                    if rating:
                        current_elo = rating.rating

                if current_elo is not None:
                    signals = self._signal_computer.compute_signals(
                        current_games=current_games,
                        current_elo=current_elo,
                        config_key=config_key,
                    )

                    if signals.elo_regression_detected and signals.elo_drop_magnitude > abs(threshold):
                        return PromotionDecision(
                            model_id=model_id,
                            promotion_type=PromotionType.ROLLBACK,
                            should_promote=True,
                            reason=f"Unified signals: Elo regression {signals.elo_trend:.1f}/hr detected",
                            current_elo=current_elo,
                            elo_improvement=-signals.elo_drop_magnitude,
                        )
            except Exception as e:
                logger.debug(f"Unified signals rollback check failed: {e}")

        # Fallback to direct Elo history check
        if self.elo_service:
            try:
                history = self.elo_service.get_rating_history(
                    model_id, board_type, num_players, limit=10
                )
                if len(history) >= 2:
                    recent_elo = history[0].get("rating", 0)
                    older_elo = history[-1].get("rating", 0)
                    elo_change = recent_elo - older_elo

                    if elo_change < threshold:
                        return PromotionDecision(
                            model_id=model_id,
                            promotion_type=PromotionType.ROLLBACK,
                            should_promote=True,
                            reason=f"Elo regression {elo_change:.1f} < threshold {threshold}",
                            current_elo=recent_elo,
                            elo_improvement=elo_change,
                        )
            except Exception as e:
                logger.warning(f"Failed to evaluate rollback: {e}")

        return PromotionDecision(
            model_id=model_id,
            promotion_type=PromotionType.ROLLBACK,
            should_promote=False,
            reason="No significant regression detected",
        )

    def get_unified_signals(
        self,
        config_key: str,
        current_games: int,
        current_elo: float,
    ) -> TrainingSignals | None:
        """Get unified training signals for a config.

        Useful for callers that want to see the full signal state.
        """
        if self._signal_computer is None:
            return None
        return self._signal_computer.compute_signals(
            current_games=current_games,
            current_elo=current_elo,
            config_key=config_key,
        )

    def execute_promotion(
        self,
        decision: PromotionDecision,
        dry_run: bool = False,
    ) -> bool:
        """Execute a promotion decision.

        Args:
            decision: The promotion decision to execute
            dry_run: If True, only log what would happen

        Returns:
            True if promotion was successful
        """
        if not decision.should_promote:
            logger.info(f"Skipping promotion for {decision.model_id}: {decision.reason}")
            return False

        if dry_run:
            logger.info(f"[DRY RUN] Would promote {decision.model_id}: {decision.reason}")
            self._emit_execution_metrics(decision, success=True, dry_run=True)
            return True

        # Use distributed lock for cross-node coordination (December 2025)
        # Extract config_key from model_id (format: "{board}_{players}p_v{N}")
        config_key = self._extract_config_key(decision.model_id)

        if _HAS_DISTRIBUTED_LOCKS and TrainingLocks is not None:
            with TrainingLocks.promotion(config_key, timeout=60) as lock:
                if not lock:
                    logger.warning(
                        f"Could not acquire promotion lock for {config_key}, "
                        f"another node may be promoting"
                    )
                    return False
                return self._execute_promotion_locked(decision)
        else:
            # Fallback to unlocked execution if distributed locks unavailable
            return self._execute_promotion_locked(decision)

    def _extract_config_key(self, model_id: str) -> str:
        """Extract config key from model ID.

        Args:
            model_id: Model identifier (e.g., "square8_2p_v42")

        Returns:
            Config key (e.g., "square8_2p")
        """
        # Model ID format: {board}_{players}p_v{N} or {board}_{players}p_{suffix}
        parts = model_id.rsplit("_", 1)
        if len(parts) == 2 and (parts[1].startswith("v") or parts[1].isdigit()):
            return parts[0]
        # Try to find the config portion before version
        if "_v" in model_id:
            return model_id.split("_v")[0]
        return model_id

    def _execute_promotion_locked(self, decision: PromotionDecision) -> bool:
        """Execute promotion while holding the distributed lock.

        Args:
            decision: The promotion decision to execute

        Returns:
            True if promotion was successful
        """
        success = False
        try:
            if decision.promotion_type == PromotionType.TIER:
                success = self._execute_tier_promotion(decision)
            elif decision.promotion_type in (PromotionType.STAGING, PromotionType.PRODUCTION):
                success = self._execute_stage_promotion(decision)
            elif decision.promotion_type == PromotionType.CHAMPION:
                success = self._execute_champion_promotion(decision)
            elif decision.promotion_type == PromotionType.ROLLBACK:
                success = self._execute_rollback(decision)
            elif decision.promotion_type == PromotionType.ELO_IMPROVEMENT:
                success = self._execute_elo_promotion(decision)
            else:
                logger.error(f"Unknown promotion type: {decision.promotion_type}")
                success = False
        except Exception as e:
            logger.error(f"Promotion failed for {decision.model_id}: {e}")
            success = False

        self._emit_execution_metrics(decision, success=success, dry_run=False)

        # Notify multiple systems on successful promotion (December 2025)
        if success:
            self._notify_promotion(decision)

        # Record promotion result with ImprovementOptimizer for feedback loop (December 2025)
        self._record_promotion_feedback(decision, success)

        return success

    def _record_promotion_feedback(
        self,
        decision: PromotionDecision,
        success: bool,
    ) -> None:
        """Record promotion result with ImprovementOptimizer.

        This closes the feedback loop between promotion decisions and
        training threshold adjustments. Successful promotions accelerate
        training; failed promotions trigger more careful evaluation.

        Args:
            decision: The promotion decision
            success: Whether promotion succeeded
        """
        if not _HAS_IMPROVEMENT_OPTIMIZER or get_improvement_optimizer is None:
            return

        try:
            optimizer = get_improvement_optimizer()
            config_key = self._extract_config_key(decision.model_id)
            elo_gain = decision.elo_improvement or 0.0

            if success:
                # Record successful promotion for positive feedback
                optimizer.record_promotion_success(
                    config_key=config_key,
                    elo_gain=elo_gain,
                    model_id=decision.model_id,
                )
                logger.debug(
                    f"[PromotionController] Recorded promotion success: "
                    f"{config_key} +{elo_gain:.1f} Elo"
                )
            else:
                # Record failed promotion for feedback adjustment
                optimizer.record_promotion_failure(
                    config_key=config_key,
                    reason=decision.reason or "Unknown failure",
                )
                logger.debug(
                    f"[PromotionController] Recorded promotion failure: "
                    f"{config_key} - {decision.reason}"
                )
        except Exception as e:
            logger.debug(f"[PromotionController] ImprovementOptimizer feedback failed: {e}")

    def _notify_promotion(self, decision: PromotionDecision) -> None:
        """Notify multiple systems about a successful promotion.

        Broadcasts promotion events to:
        1. Event bus (for cross-process coordination)
        2. P2P orchestrator (for cluster sync)
        3. Slack/webhook (for team notifications)
        4. Model sync coordinator (for model distribution)

        Args:
            decision: The executed promotion decision
        """
        payload = decision.to_dict()

        # 1. Publish to event bus for cross-process coordination
        self._notify_event_bus(payload)

        # 2. Notify P2P orchestrator for cluster-wide awareness
        self._notify_p2p_orchestrator(payload)

        # 3. Send Slack notification for visibility
        self._notify_slack(decision)

        # 4. Trigger model sync across cluster
        self._notify_model_sync(decision)

        logger.info(f"Multi-system notifications sent for {decision.model_id} promotion")

    def _notify_event_bus(self, payload: dict[str, Any]) -> None:
        """Publish promotion event to the data event bus."""
        try:
            from app.coordination.event_router import get_router
            from app.coordination.event_router import (
                DataEvent,
                DataEventType,
            )

            event = DataEvent(
                event_type=DataEventType.MODEL_PROMOTED,
                payload=payload,
                source="promotion_controller",
            )

            router = get_router()
            import asyncio
            try:
                asyncio.get_running_loop()  # Verify we're in async context
                asyncio.create_task(router.publish(event))
            except RuntimeError:
                if hasattr(router, 'publish_sync'):
                    router.publish_sync(event)

            logger.debug("Published MODEL_PROMOTED event to event bus")
        except Exception as e:
            logger.debug(f"Event bus notification failed: {e}")

        # Also emit PROMOTION_COMPLETE StageEvent (December 2025)
        self._emit_stage_event(payload)

    def _emit_stage_event(self, payload: dict[str, Any]) -> None:
        """Emit PROMOTION_COMPLETE StageEvent via centralized event_emitters.

        This connects promotion events to the stage event bus, enabling
        downstream consumers like TrainingDataCoordinator to react.

        December 2025: Migrated to use event_emitters.py for unified routing.
        """
        try:
            from app.coordination.event_emission_helpers import safe_emit_event

            safe_emit_event(
                "PROMOTION_COMPLETE",
                {
                    "model_id": payload.get("model_id"),
                    "model_path": payload.get("model_path"),
                    "board_type": payload.get("board_type"),
                    "num_players": payload.get("num_players"),
                    "old_elo": payload.get("old_elo"),
                    "new_elo": payload.get("new_elo"),
                    "promotion_type": payload.get("promotion_type"),
                },
                context="promotion_controller",
            )
        except ImportError:
            logger.debug("Event emission helpers not available")

    def _notify_p2p_orchestrator(self, payload: dict[str, Any]) -> None:
        """Notify P2P orchestrator about the promotion."""
        try:
            import json
            import os
            import urllib.request

            # Dec 2025: Use centralized P2P URL helper
            from app.config.ports import get_local_p2p_url
            p2p_url = get_local_p2p_url()
            url = f"{p2p_url}/api/model/promoted"

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            try:
                urllib.request.urlopen(req, timeout=URLOPEN_SHORT_TIMEOUT)
                logger.debug("Notified P2P orchestrator about promotion")
            except urllib.error.URLError:
                pass  # P2P might not be running
        except Exception as e:
            logger.debug(f"P2P notification failed: {e}")

    def _notify_slack(self, decision: PromotionDecision) -> None:
        """Send Slack notification for the promotion."""
        try:
            import json
            import os
            import urllib.request

            webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
            if not webhook_url:
                return

            # Build message based on promotion type
            emoji = {
                PromotionType.PRODUCTION: ":rocket:",
                PromotionType.STAGING: ":test_tube:",
                PromotionType.TIER: ":trophy:",
                PromotionType.CHAMPION: ":crown:",
                PromotionType.ROLLBACK: ":rewind:",
            }.get(decision.promotion_type, ":arrow_up:")

            color = "#36a64f" if decision.promotion_type != PromotionType.ROLLBACK else "#f2c744"

            text = (
                f"{emoji} *Model {decision.promotion_type.value.upper()}*\n"
                f"Model: `{decision.model_id}`\n"
                f"Elo: {decision.current_elo or 'N/A'}"
            )
            if decision.elo_improvement:
                text += f" (+{decision.elo_improvement:.1f})"
            text += f"\nReason: {decision.reason}"

            payload = json.dumps({
                "attachments": [{
                    "color": color,
                    "text": text,
                    "footer": "RingRift Promotion Controller",
                }]
            }).encode("utf-8")

            req = urllib.request.Request(
                webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=URLOPEN_TIMEOUT)
            logger.debug("Sent Slack notification for promotion")
        except Exception as e:
            logger.debug(f"Slack notification failed: {e}")

    def _notify_model_sync(self, decision: PromotionDecision) -> None:
        """Trigger model sync to distribute promoted model across cluster."""
        try:
            # Only sync for production/champion promotions
            if decision.promotion_type not in (PromotionType.PRODUCTION, PromotionType.CHAMPION):
                return

            from app.coordination.work_queue import WorkItem, WorkType, get_work_queue

            queue = get_work_queue()

            # Add high-priority sync work
            work = WorkItem(
                work_type=WorkType.DATA_SYNC,
                priority=90,  # High priority for production model sync
                config={
                    "model_id": decision.model_id,
                    "promotion_type": decision.promotion_type.value,
                    "sync_type": "model_distribution",
                },
                timeout_seconds=1800.0,  # 30 min for model sync
            )
            queue.add_work(work)
            logger.debug(f"Queued model sync work for {decision.model_id}")
        except Exception as e:
            logger.debug(f"Model sync notification failed: {e}")

    def _emit_execution_metrics(
        self,
        decision: PromotionDecision,
        success: bool,
        dry_run: bool,
    ) -> None:
        """Emit Prometheus metrics for a promotion execution."""
        try:
            from app.metrics import record_promotion_execution
            record_promotion_execution(
                promotion_type=decision.promotion_type.value,
                success=success,
                dry_run=dry_run,
            )
        except ImportError:
            pass

    def _execute_stage_promotion(self, decision: PromotionDecision) -> bool:
        """Execute staging or production promotion."""
        if self.model_registry:
            try:
                from app.training.model_registry import ModelStage

                target_stage = (
                    ModelStage.STAGING if decision.promotion_type == PromotionType.STAGING
                    else ModelStage.PRODUCTION
                )

                self.model_registry.promote_model(decision.model_id, target_stage)
                logger.info(f"Promoted {decision.model_id} to {target_stage.value}")

                # December 2025: Create inference symlinks for model distribution
                # This ensures ModelDistributionDaemon can find and distribute the model
                if target_stage == ModelStage.PRODUCTION:
                    self._create_inference_symlinks(decision)

                return True
            except Exception as e:
                logger.error(f"Model registry promotion failed: {e}")
        return False

    def _create_inference_symlinks(self, decision: PromotionDecision) -> None:
        """Create inference symlinks after production promotion (December 2025).

        Creates ringrift_best_{config}.pth -> canonical_{config}.pth symlinks
        which ModelDistributionDaemon uses to discover and distribute models.

        Args:
            decision: The promotion decision with model info
        """
        try:
            from pathlib import Path

            config_key = self._extract_config_key(decision.model_id)
            models_dir = Path(__file__).parent.parent.parent / "models"

            canonical_name = f"canonical_{config_key}.pth"
            symlink_name = f"ringrift_best_{config_key}.pth"

            canonical_path = models_dir / canonical_name
            symlink_path = models_dir / symlink_name

            # Only create symlink if canonical model exists
            if not canonical_path.exists():
                logger.debug(
                    f"[PromotionController] Canonical model not found: {canonical_path}, "
                    f"skipping symlink creation"
                )
                return

            # Remove existing symlink if present
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()

            # Create new symlink (relative path for portability)
            symlink_path.symlink_to(canonical_name)
            logger.info(
                f"[PromotionController] Created inference symlink: "
                f"{symlink_name} -> {canonical_name}"
            )

        except Exception as e:
            # Symlink creation failure shouldn't fail the promotion
            logger.warning(f"[PromotionController] Failed to create inference symlink: {e}")

    def _execute_elo_promotion(self, decision: PromotionDecision) -> bool:
        """Execute Elo-based auto-promotion: copy candidate model to canonical.

        Feb 2026: Training now saves to candidate_{config}.pth instead of
        canonical_{config}.pth. This method copies candidate → canonical
        after evaluation confirms the model is an improvement.
        """
        try:
            import shutil
            from pathlib import Path

            config_key = self._extract_config_key(decision.model_id)
            models_dir = Path(__file__).parent.parent.parent / "models"

            # Use explicit model_path if available, else derive from config_key
            if decision.model_path:
                candidate_path = Path(decision.model_path)
                if not candidate_path.is_absolute():
                    candidate_path = models_dir.parent / candidate_path
            else:
                candidate_path = models_dir / f"candidate_{config_key}.pth"
            canonical_path = models_dir / f"canonical_{config_key}.pth"

            if not candidate_path.exists():
                logger.warning(
                    f"[PromotionController] Candidate model not found: {candidate_path}"
                )
                return False

            # Back up current canonical before overwriting
            if canonical_path.exists():
                backup_path = models_dir / f"canonical_{config_key}.backup.pth"
                shutil.copy2(str(canonical_path), str(backup_path))
                logger.info(
                    f"[PromotionController] Backed up canonical to {backup_path.name}"
                )

            # Atomic copy: write to temp, then rename
            tmp_path = canonical_path.with_suffix(".tmp")
            shutil.copy2(str(candidate_path), str(tmp_path))
            tmp_path.rename(canonical_path)

            logger.info(
                f"[PromotionController] Promoted candidate → canonical for {config_key}"
            )

            # Create inference symlinks
            self._create_inference_symlinks(decision)

            # Update model registry if available
            if self.model_registry:
                try:
                    from app.training.model_registry import ModelStage
                    self.model_registry.promote_model(decision.model_id, ModelStage.PRODUCTION)
                except Exception as e:
                    logger.debug(f"[PromotionController] Registry update: {e}")

            return True
        except Exception as e:
            logger.error(f"[PromotionController] Elo promotion failed: {e}")
            return False

    def _execute_tier_promotion(self, decision: PromotionDecision) -> bool:
        """Execute tier promotion.

        Dec 2025: Now emits TIER_PROMOTION event for downstream coordination.
        """
        try:
            from app.training.tier_promotion_registry import (
                load_square8_two_player_registry,
                save_square8_two_player_registry,
            )

            registry = load_square8_two_player_registry()
            tier = decision.target_tier
            old_tier = None

            # Track old tier for event
            if tier and "tiers" in registry:
                # Find current tier for this config
                config_key = f"{decision.board_type or 'unknown'}_{decision.num_players or 2}p"
                for t, t_data in registry.get("tiers", {}).items():
                    if t_data.get("promoted_model"):
                        old_tier = t
                        break

            if tier and tier not in registry.get("tiers", {}):
                registry.setdefault("tiers", {})[tier] = {}

            if tier:
                registry["tiers"][tier]["promoted_model"] = decision.model_id
                registry["tiers"][tier]["promoted_at"] = decision.evaluated_at
                registry["tiers"][tier]["elo"] = decision.current_elo

                save_square8_two_player_registry(registry)
                logger.info(f"Promoted {decision.model_id} to tier {tier}")

                # Dec 2025: Emit TIER_PROMOTION event for curriculum integration
                try:
                    from app.distributed.event_helpers import emit_tier_promotion_safe
                    import asyncio

                    config_key = f"{decision.board_type or 'unknown'}_{decision.num_players or 2}p"
                    asyncio.get_event_loop().create_task(
                        emit_tier_promotion_safe(
                            config_key=config_key,
                            old_tier=old_tier or "unknown",
                            new_tier=tier,
                            model_id=decision.model_id,
                            elo=decision.current_elo or 0.0,
                            win_rate=decision.win_rate or 0.0,
                            source="promotion_controller",
                        )
                    )
                    logger.debug(f"Emitted TIER_PROMOTION event: {config_key} -> {tier}")
                except Exception as emit_err:
                    # Event emission failure shouldn't block promotion
                    logger.warning(f"Failed to emit TIER_PROMOTION event: {emit_err}")

                return True
        except Exception as e:
            logger.error(f"Tier promotion failed: {e}")
        return False

    def _execute_champion_promotion(self, decision: PromotionDecision) -> bool:
        """Execute champion promotion."""
        # Champion promotion typically goes to production
        return self._execute_stage_promotion(
            PromotionDecision(
                model_id=decision.model_id,
                promotion_type=PromotionType.PRODUCTION,
                should_promote=True,
                reason=f"Champion promotion: {decision.reason}",
                current_elo=decision.current_elo,
                games_played=decision.games_played,
                win_rate=decision.win_rate,
            )
        )

    def _execute_rollback(self, decision: PromotionDecision) -> bool:
        """Execute rollback to previous model."""
        if self.model_registry:
            try:
                # Archive current production model
                from app.training.model_registry import ModelStage

                current_production = self.model_registry.get_production_model()
                if current_production:
                    self.model_registry.promote_model(
                        current_production, ModelStage.ARCHIVED
                    )

                # Restore previous model to production
                self.model_registry.promote_model(
                    decision.model_id, ModelStage.PRODUCTION
                )
                logger.info(f"Rolled back to {decision.model_id}")
                return True
            except Exception as e:
                logger.error(f"Rollback failed: {e}")
        return False

    def health_check(self) -> "HealthCheckResult":
        """Check health of the PromotionController.

        Returns:
            HealthCheckResult with controller health status

        Dec 2025: Added for DaemonManager integration.
        """
        try:
            from app.coordination.protocols import HealthCheckResult
        except ImportError:
            # Return dict for backward compatibility
            return {
                "healthy": True,
                "message": "PromotionController operational",
                "details": {"event_subscribed": self._event_subscribed},
            }

        details = {
            "event_subscribed": self._event_subscribed,
            "pending_checks": len(self._pending_promotion_checks),
            "elo_service_available": self._elo_service is not None,
            "model_registry_available": self._model_registry is not None,
            "lifecycle_manager_available": self._lifecycle_manager is not None,
            "signal_computer_available": self._signal_computer is not None,
        }

        # Check if event subscription failed (should be subscribed)
        if HAS_EVENT_BUS and not self._event_subscribed:
            return HealthCheckResult(
                healthy=False,
                message="Event subscription failed",
                details=details,
            )

        return HealthCheckResult(
            healthy=True,
            message="PromotionController operational",
            details=details,
        )


class GraduatedResponseAction(str, Enum):
    """Graduated response actions for rollback scenarios."""
    NOTIFY = "notify"              # Just notify, allow rollback
    SLOW_DOWN = "slow_down"        # Increase cooldown, allow rollback
    INVESTIGATE = "investigate"     # Trigger investigation, allow rollback
    PAUSE_TRAINING = "pause_training"  # Pause training, allow rollback
    ESCALATE_HUMAN = "escalate_human"  # Require human intervention


def get_adaptive_regression_threshold(config_key: str, current_elo: float) -> float:
    """Get regression threshold adapted to board difficulty and model strength.

    Dec 29, 2025 - Phase 5: Adaptive promotion thresholds.
    Board difficulty varies: hex8 is easy (1200+ Elo), square19 is hard (<800 Elo).
    Static -30 Elo threshold is too strict for hard boards, too lenient for easy ones.

    Args:
        config_key: Config like "hex8_2p" or board type like "hex8"
        current_elo: Current model Elo rating

    Returns:
        Regression threshold (negative value, e.g., -50 for hex8, -20 for square19)
    """
    # Extract board type from config_key
    board = config_key.split("_")[0] if "_" in config_key else config_key

    # Board-specific base thresholds (harder boards = more lenient)
    base_thresholds = {
        "hex8": -50,      # Easy board - stricter threshold
        "square8": -40,   # Medium board
        "square19": -20,  # Hard board - more lenient
        "hexagonal": -25, # Hard board - more lenient
    }
    threshold = base_thresholds.get(board, -30)

    # Adjust for model strength
    # Weak models (<700 Elo) get more lenient thresholds
    # Strong models (>1500 Elo) get stricter thresholds
    if current_elo < 700:
        threshold = threshold * 1.5  # More lenient for weak models
    elif current_elo > 1500:
        threshold = threshold * 0.7  # Stricter for strong models

    return threshold


@dataclass
class RollbackCriteria:
    """Criteria for automatic rollback decisions."""
    # Elo regression threshold (negative = regression)
    # Dec 29, 2025: This is the default; use get_adaptive_regression_threshold() for config-specific
    elo_regression_threshold: float = -30.0
    # Minimum games required before considering rollback
    min_games_for_regression: int = 20
    # Number of consecutive regression checks before triggering rollback
    consecutive_checks_required: int = 3
    # Win rate threshold below which rollback may be considered
    min_win_rate: float = 0.40
    # Time window in seconds for regression detection
    time_window_seconds: int = 3600  # 1 hour
    # Cooldown period in seconds between rollbacks for the same config
    cooldown_seconds: int = 3600  # 1 hour default
    # Maximum rollbacks per day before requiring manual intervention
    max_rollbacks_per_day: int = 3
    # Enable graduated response based on rollback count
    graduated_response_enabled: bool = True
    # Cooldown multiplier for "slow_down" response
    slow_down_multiplier: float = 2.0


class NotificationHook:
    """Base class for rollback notification hooks.

    Extend this class to implement custom notification handlers.

    Example:
        class SlackHook(NotificationHook):
            def __init__(self, webhook_url: str):
                self.webhook_url = webhook_url

            def on_rollback_triggered(self, event):
                import requests
                requests.post(self.webhook_url, json={
                    "text": f"Rollback triggered: {event.reason}"
                })
    """

    def on_regression_detected(self, model_id: str, status: dict[str, Any]) -> None:
        """Called when regression is detected but rollback not yet triggered."""

    def on_at_risk(self, model_id: str, status: dict[str, Any]) -> None:
        """Called when model enters at-risk state."""

    def on_rollback_triggered(self, event: RollbackEvent) -> None:
        """Called when rollback is triggered (before execution)."""

    def on_rollback_completed(self, event: RollbackEvent, success: bool) -> None:
        """Called after rollback execution."""


class LoggingNotificationHook(NotificationHook):
    """Default hook that logs notifications."""

    def __init__(self, logger_name: str = "ringrift.rollback"):
        import logging
        self.logger = logging.getLogger(logger_name)

    def on_regression_detected(self, model_id: str, status: dict[str, Any]) -> None:
        self.logger.warning(
            f"Regression detected for {model_id}: "
            f"consecutive={status.get('consecutive_regressions', 0)}, "
            f"avg={status.get('avg_regression', 0):.1f}"
        )

    def on_at_risk(self, model_id: str, status: dict[str, Any]) -> None:
        self.logger.warning(
            f"MODEL AT RISK: {model_id} - "
            f"consecutive regressions: {status.get('consecutive_regressions', 0)}"
        )

    def on_rollback_triggered(self, event: RollbackEvent) -> None:
        self.logger.critical(
            f"ROLLBACK TRIGGERED: {event.current_model_id} -> {event.rollback_model_id} "
            f"(reason: {event.reason})"
        )

    def on_rollback_completed(self, event: RollbackEvent, success: bool) -> None:
        if success:
            self.logger.info(
                f"Rollback completed: {event.current_model_id} -> {event.rollback_model_id}"
            )
        else:
            self.logger.error(
                f"Rollback FAILED: {event.current_model_id} -> {event.rollback_model_id}"
            )


class WebhookNotificationHook(NotificationHook):
    """Notification hook that sends webhooks to external services.

    Supports Slack, Discord, and generic webhook endpoints.
    """

    def __init__(
        self,
        webhook_url: str,
        webhook_type: str = "generic",  # "slack", "discord", "generic"
        timeout: int = 10,
    ):
        self.webhook_url = webhook_url
        self.webhook_type = webhook_type
        self.timeout = timeout

    def _send_webhook(self, message: str, level: str = "info") -> bool:
        """Send a webhook notification."""
        try:
            import json
            import urllib.error
            import urllib.request

            if self.webhook_type == "slack":
                # Slack format
                color = {"info": "good", "warning": "warning", "critical": "danger"}.get(level, "good")
                payload = {
                    "attachments": [{
                        "color": color,
                        "text": message,
                        "footer": "RingRift AI Rollback Monitor",
                    }]
                }
            elif self.webhook_type == "discord":
                # Discord format
                color = {"info": 0x00FF00, "warning": 0xFFFF00, "critical": 0xFF0000}.get(level, 0x00FF00)
                payload = {
                    "embeds": [{
                        "description": message,
                        "color": color,
                        "footer": {"text": "RingRift AI Rollback Monitor"},
                    }]
                }
            else:
                # Generic format
                payload = {
                    "message": message,
                    "level": level,
                    "source": "ringrift_rollback_monitor",
                }

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=self.timeout)
            return True

        except Exception as e:
            print(f"[WebhookHook] Failed to send webhook: {e}")
            return False

    def on_at_risk(self, model_id: str, status: dict[str, Any]) -> None:
        self._send_webhook(
            f"⚠️ Model at risk: `{model_id}` - "
            f"{status.get('consecutive_regressions', 0)} consecutive regressions",
            level="warning",
        )

    def on_rollback_triggered(self, event: RollbackEvent) -> None:
        self._send_webhook(
            f"🔄 Rollback triggered: `{event.current_model_id}` → `{event.rollback_model_id}`\n"
            f"Reason: {event.reason}",
            level="critical",
        )

    def on_rollback_completed(self, event: RollbackEvent, success: bool) -> None:
        if success:
            self._send_webhook(
                f"✅ Rollback completed: `{event.current_model_id}` → `{event.rollback_model_id}`",
                level="info",
            )
        else:
            self._send_webhook(
                f"❌ Rollback FAILED: `{event.current_model_id}` → `{event.rollback_model_id}`",
                level="critical",
            )


@dataclass
class RollbackEvent:
    """Record of a rollback event."""
    triggered_at: str
    current_model_id: str
    rollback_model_id: str
    reason: str
    elo_regression: float | None = None
    games_played: int = 0
    win_rate: float | None = None
    auto_triggered: bool = True
    board_type: str = "square8"
    num_players: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "triggered_at": self.triggered_at,
            "current_model_id": self.current_model_id,
            "rollback_model_id": self.rollback_model_id,
            "reason": self.reason,
            "elo_regression": self.elo_regression,
            "games_played": self.games_played,
            "win_rate": self.win_rate,
            "auto_triggered": self.auto_triggered,
            "board_type": self.board_type,
            "num_players": self.num_players,
        }


class RollbackMonitor:
    """Automated rollback monitoring for promoted models.

    Monitors model performance after promotion and triggers automatic
    rollback if significant regression is detected.

    Features graduated response levels based on rollback count:
    - 1st rollback: notify only
    - 2nd rollback: 2x cooldown (slow_down)
    - 3rd rollback: trigger investigation
    - 5th rollback: pause training, alert
    - 7th+ rollback: require human intervention

    Usage:
        from app.training.promotion_controller import RollbackMonitor, RollbackCriteria

        monitor = RollbackMonitor(
            criteria=RollbackCriteria(elo_regression_threshold=-25.0)
        )

        # Check if rollback is needed
        should_rollback, event = monitor.check_for_regression(
            model_id="model_v42",
            board_type="square8",
            num_players=2,
            previous_model_id="model_v41",
        )

        if should_rollback:
            success = monitor.execute_rollback(event)
    """

    # Graduated response levels: (rollback_count_threshold, action)
    # Actions escalate as rollback count increases
    RESPONSE_LEVELS = [
        (1, GraduatedResponseAction.NOTIFY),           # 1st: just notify
        (2, GraduatedResponseAction.SLOW_DOWN),        # 2nd: 2x cooldown
        (3, GraduatedResponseAction.INVESTIGATE),      # 3rd: root cause analysis
        (5, GraduatedResponseAction.PAUSE_TRAINING),   # 5th: pause training, alert
        (7, GraduatedResponseAction.ESCALATE_HUMAN),   # 7th+: require human
    ]

    def __init__(
        self,
        criteria: RollbackCriteria | None = None,
        promotion_controller: PromotionController | None = None,
        notification_hooks: list[NotificationHook] | None = None,
    ):
        self.criteria = criteria or RollbackCriteria()
        self._controller = promotion_controller
        # Track regression history per model: model_id -> list of (timestamp, elo_diff)
        self._regression_history: dict[str, list[tuple[str, float]]] = {}
        # Track rollback events
        self._rollback_events: list[RollbackEvent] = []
        # Notification hooks
        self._hooks: list[NotificationHook] = notification_hooks or []
        # Track which models we've already notified about being at-risk (avoid spam)
        self._at_risk_notified: set = set()
        # Track last rollback time per config key (board_type, num_players)
        self._last_rollback_time: dict[str, datetime] = {}
        # Track cooldown bypass state (for manual overrides)
        self._cooldown_bypass: bool = False
        # Track whether training is paused due to rollback escalation
        self._training_paused: bool = False
        # Track investigation tasks triggered
        self._pending_investigations: list[str] = []

    # =========================================================================
    # GRADUATED RESPONSE METHODS
    # =========================================================================

    def get_response_action(self, rollback_count: int | None = None) -> GraduatedResponseAction:
        """Get the appropriate response action based on rollback count.

        Args:
            rollback_count: Override count (if None, uses daily count)

        Returns:
            The appropriate GraduatedResponseAction for this rollback count
        """
        if not self.criteria.graduated_response_enabled:
            return GraduatedResponseAction.NOTIFY

        count = rollback_count if rollback_count is not None else self.get_daily_rollback_count()

        # Find the appropriate response level
        action = GraduatedResponseAction.NOTIFY  # Default
        for threshold, response_action in self.RESPONSE_LEVELS:
            if count >= threshold:
                action = response_action
            else:
                break

        return action

    def get_effective_cooldown(self) -> int:
        """Get the effective cooldown based on current response level.

        Returns:
            Cooldown in seconds (may be multiplied for slow_down response)
        """
        action = self.get_response_action()
        base_cooldown = self.criteria.cooldown_seconds

        if action == GraduatedResponseAction.SLOW_DOWN:
            return int(base_cooldown * self.criteria.slow_down_multiplier)
        elif action in (GraduatedResponseAction.INVESTIGATE,
                       GraduatedResponseAction.PAUSE_TRAINING):
            # Even longer cooldown during investigation/pause
            return int(base_cooldown * self.criteria.slow_down_multiplier * 2)

        return base_cooldown

    def should_block_rollback(self) -> tuple[bool, str]:
        """Check if rollback should be blocked due to escalation.

        Returns:
            Tuple of (should_block, reason)
        """
        action = self.get_response_action()

        if action == GraduatedResponseAction.ESCALATE_HUMAN:
            return True, "Human intervention required - too many rollbacks"

        return False, ""

    def is_training_paused(self) -> bool:
        """Check if training is currently paused due to rollback escalation."""
        return self._training_paused

    def pause_training(self, reason: str = "Rollback escalation") -> None:
        """Pause training due to rollback escalation."""
        if self._training_paused:
            return

        self._training_paused = True
        logger.warning(f"TRAINING PAUSED: {reason}")

        # Notify all hooks about pause
        for hook in self._hooks:
            try:
                if hasattr(hook, 'on_training_paused'):
                    hook.on_training_paused(reason)
            except Exception as e:
                logger.warning(f"Notification hook error on pause: {e}")

    def resume_training(self, reason: str = "Manual resume") -> None:
        """Resume training after pause."""
        if not self._training_paused:
            return

        self._training_paused = False
        logger.info(f"TRAINING RESUMED: {reason}")

        # Notify all hooks about resume
        for hook in self._hooks:
            try:
                if hasattr(hook, 'on_training_resumed'):
                    hook.on_training_resumed(reason)
            except Exception as e:
                logger.warning(f"Notification hook error on resume: {e}")

    def trigger_investigation(self, model_id: str, event: RollbackEvent) -> str:
        """Trigger a root cause investigation for repeated rollbacks.

        Args:
            model_id: Model experiencing rollbacks
            event: The rollback event that triggered investigation

        Returns:
            Investigation ID for tracking
        """
        investigation_id = f"inv_{model_id}_{int(datetime.now().timestamp())}"

        investigation_data = {
            "id": investigation_id,
            "model_id": model_id,
            "triggered_at": datetime.now().isoformat(),
            "rollback_count": self.get_daily_rollback_count(),
            "event": event.to_dict(),
            "status": "pending",
        }

        self._pending_investigations.append(investigation_id)
        logger.warning(
            f"INVESTIGATION TRIGGERED: {investigation_id} for model {model_id} "
            f"(rollback #{self.get_daily_rollback_count()})"
        )

        # Notify hooks about investigation
        for hook in self._hooks:
            try:
                if hasattr(hook, 'on_investigation_triggered'):
                    hook.on_investigation_triggered(investigation_data)
            except Exception as e:
                logger.warning(f"Notification hook error on investigation: {e}")

        return investigation_id

    def apply_graduated_response(self, event: RollbackEvent) -> GraduatedResponseAction:
        """Apply the graduated response for the current rollback situation.

        Args:
            event: The rollback event

        Returns:
            The action that was applied
        """
        action = self.get_response_action()

        logger.info(
            f"Graduated response: {action.value} (rollback #{self.get_daily_rollback_count()})"
        )

        if action == GraduatedResponseAction.SLOW_DOWN:
            # Cooldown is automatically handled via get_effective_cooldown()
            logger.info(
                f"Applying slow-down response: cooldown increased to "
                f"{self.get_effective_cooldown()}s"
            )

        elif action == GraduatedResponseAction.INVESTIGATE:
            self.trigger_investigation(event.current_model_id, event)

        elif action == GraduatedResponseAction.PAUSE_TRAINING:
            self.pause_training(
                f"Too many rollbacks ({self.get_daily_rollback_count()}) - "
                f"pausing training for investigation"
            )
            self.trigger_investigation(event.current_model_id, event)

        elif action == GraduatedResponseAction.ESCALATE_HUMAN:
            # This should have been blocked earlier, but log just in case
            logger.critical(
                f"HUMAN ESCALATION REQUIRED: {self.get_daily_rollback_count()} rollbacks "
                f"in 24 hours - automatic rollback disabled"
            )

        return action

    def get_graduated_response_status(self) -> dict[str, Any]:
        """Get current graduated response status for monitoring.

        Returns:
            Dict with current response level info
        """
        daily_count = self.get_daily_rollback_count()
        action = self.get_response_action()

        return {
            "enabled": self.criteria.graduated_response_enabled,
            "daily_rollback_count": daily_count,
            "current_action": action.value,
            "effective_cooldown_seconds": self.get_effective_cooldown(),
            "training_paused": self._training_paused,
            "pending_investigations": len(self._pending_investigations),
            "response_levels": [
                {"threshold": t, "action": a.value}
                for t, a in self.RESPONSE_LEVELS
            ],
        }

    def add_notification_hook(self, hook: NotificationHook) -> None:
        """Add a notification hook."""
        self._hooks.append(hook)

    def _notify_regression_detected(self, model_id: str, status: dict[str, Any]) -> None:
        """Notify hooks about detected regression."""
        for hook in self._hooks:
            try:
                hook.on_regression_detected(model_id, status)
            except Exception as e:
                logger.warning(f"Notification hook error: {e}")

    def _notify_at_risk(self, model_id: str, status: dict[str, Any]) -> None:
        """Notify hooks about model entering at-risk state."""
        if model_id in self._at_risk_notified:
            return  # Already notified
        self._at_risk_notified.add(model_id)
        for hook in self._hooks:
            try:
                hook.on_at_risk(model_id, status)
            except Exception as e:
                logger.warning(f"Notification hook error: {e}")

    def _notify_rollback_triggered(self, event: RollbackEvent) -> None:
        """Notify hooks about rollback being triggered."""
        for hook in self._hooks:
            try:
                hook.on_rollback_triggered(event)
            except Exception as e:
                logger.warning(f"Notification hook error: {e}")

    def _notify_rollback_completed(self, event: RollbackEvent, success: bool) -> None:
        """Notify hooks about rollback completion."""
        # Clear at-risk notification state for this model
        if success and event.current_model_id in self._at_risk_notified:
            self._at_risk_notified.discard(event.current_model_id)
        for hook in self._hooks:
            try:
                hook.on_rollback_completed(event, success)
            except Exception as e:
                logger.warning(f"Notification hook error: {e}")

    def _config_key(self, board_type: str, num_players: int) -> str:
        """Generate a config key for tracking cooldowns."""
        return f"{board_type}_{num_players}p"

    def is_cooldown_active(self, board_type: str, num_players: int) -> tuple[bool, int | None]:
        """Check if rollback cooldown is active for this config.

        Uses effective cooldown which may be multiplied based on graduated response.

        Args:
            board_type: Board type
            num_players: Number of players

        Returns:
            Tuple of (is_active, seconds_remaining or None)
        """
        if self._cooldown_bypass:
            return False, None

        key = self._config_key(board_type, num_players)
        last_time = self._last_rollback_time.get(key)
        if not last_time:
            return False, None

        # Use effective cooldown (may be multiplied by graduated response)
        effective_cooldown = self.get_effective_cooldown()
        elapsed = (datetime.now() - last_time).total_seconds()
        if elapsed < effective_cooldown:
            remaining = int(effective_cooldown - elapsed)
            return True, remaining
        return False, None

    def get_daily_rollback_count(self) -> int:
        """Get the number of rollbacks in the last 24 hours."""
        now = datetime.now()
        cutoff = now.timestamp() - 86400  # 24 hours
        count = 0
        for event in self._rollback_events:
            try:
                event_time = datetime.fromisoformat(event.triggered_at).timestamp()
                if event_time >= cutoff:
                    count += 1
            except (ValueError, TypeError):
                pass
        return count

    def is_max_daily_rollbacks_reached(self) -> tuple[bool, int]:
        """Check if max daily rollbacks limit has been reached.

        Returns:
            Tuple of (is_reached, current_count)
        """
        count = self.get_daily_rollback_count()
        return count >= self.criteria.max_rollbacks_per_day, count

    def set_cooldown_bypass(self, bypass: bool) -> None:
        """Enable or disable cooldown bypass for manual rollbacks.

        Args:
            bypass: If True, cooldown checks are skipped
        """
        self._cooldown_bypass = bypass
        if bypass:
            logger.warning("Rollback cooldown bypass ENABLED - use with caution")

    def _record_rollback_time(self, board_type: str, num_players: int) -> None:
        """Record the time of a rollback for cooldown tracking."""
        key = self._config_key(board_type, num_players)
        self._last_rollback_time[key] = datetime.now()

    @property
    def controller(self) -> PromotionController:
        """Lazy-load PromotionController."""
        if self._controller is None:
            self._controller = PromotionController()
        return self._controller

    def check_for_regression(
        self,
        model_id: str,
        board_type: str = "square8",
        num_players: int = 2,
        previous_model_id: str | None = None,
        baseline_elo: float | None = None,
    ) -> tuple[bool, RollbackEvent | None]:
        """Check if a model has regressed and should be rolled back.

        Args:
            model_id: Current model to check
            board_type: Board type
            num_players: Number of players
            previous_model_id: Model to rollback to if needed
            baseline_elo: Expected Elo at promotion time (for comparison)

        Returns:
            Tuple of (should_rollback, RollbackEvent or None)
        """
        now = datetime.now().isoformat()

        # Check cooldown period
        cooldown_active, cooldown_remaining = self.is_cooldown_active(board_type, num_players)
        if cooldown_active:
            logger.info(
                f"Rollback cooldown active for {board_type}/{num_players}p - "
                f"{cooldown_remaining}s remaining"
            )
            return False, None

        # Check daily rollback limit
        max_reached, daily_count = self.is_max_daily_rollbacks_reached()
        if max_reached:
            logger.warning(
                f"Max daily rollbacks ({self.criteria.max_rollbacks_per_day}) reached. "
                f"Manual intervention required. Count: {daily_count}"
            )
            return False, None

        # Get current model stats
        current_elo = None
        games_played = 0
        win_rate = None

        elo_service = self.controller.elo_service
        if elo_service:
            try:
                rating = elo_service.get_rating(model_id, board_type, num_players)
                if rating:
                    current_elo = rating.rating
                    games_played = rating.games_played
                    win_rate = rating.win_rate
            except Exception as e:
                logger.warning(f"Failed to get Elo for {model_id}: {e}")
                return False, None

        # Not enough games to make a decision
        if games_played < self.criteria.min_games_for_regression:
            return False, None

        # Calculate regression from baseline
        elo_regression = None
        if baseline_elo is not None and current_elo is not None:
            elo_regression = current_elo - baseline_elo
        elif previous_model_id and elo_service:
            try:
                prev_rating = elo_service.get_rating(previous_model_id, board_type, num_players)
                if prev_rating and current_elo is not None:
                    elo_regression = current_elo - prev_rating.rating
            except (OSError, AttributeError, KeyError, TypeError, ValueError):
                pass

        # Record this check in history
        if model_id not in self._regression_history:
            self._regression_history[model_id] = []

        if elo_regression is not None:
            self._regression_history[model_id].append((now, elo_regression))
            # Keep only recent history
            self._prune_history(model_id)

        # Dec 29, 2025 - Phase 5: Use adaptive threshold instead of static value
        # Board-specific: hex8 (-50), square8 (-40), square19 (-20), hexagonal (-25)
        # Elo-adjusted: More lenient for weak models (<700), stricter for strong (>1500)
        config_key = f"{board_type}_{num_players}p"
        adaptive_threshold = get_adaptive_regression_threshold(
            config_key, current_elo or 1200.0
        )

        # Check regression criteria
        should_rollback = False
        reason = ""

        # Check for severe win rate drop
        if win_rate is not None and win_rate < self.criteria.min_win_rate:
            should_rollback = True
            reason = f"Win rate {win_rate:.2%} below threshold {self.criteria.min_win_rate:.2%}"

        # Check for consecutive regression
        elif self._has_consecutive_regression(model_id):
            should_rollback = True
            avg_regression = self._get_average_regression(model_id)
            reason = f"Consecutive Elo regression detected: avg {avg_regression:.1f}"

        # Check for immediate severe regression (using adaptive threshold)
        elif elo_regression is not None and elo_regression < adaptive_threshold * 2:
            # Severe regression triggers immediate rollback
            should_rollback = True
            reason = f"Severe Elo regression: {elo_regression:.1f} (threshold: {adaptive_threshold * 2:.1f})"

        # Get regression status for notifications
        status = self.get_regression_status(model_id)

        # Notify if regression detected but not triggering rollback (using adaptive threshold)
        if elo_regression is not None and elo_regression < adaptive_threshold:
            self._notify_regression_detected(model_id, status)

        # Notify if model is at risk
        if status.get("at_risk"):
            self._notify_at_risk(model_id, status)

        if not should_rollback:
            self._emit_check_metrics(model_id, False, elo_regression)
            return False, None

        # Create rollback event
        rollback_model = previous_model_id or self._get_previous_production_model(model_id)
        if not rollback_model:
            logger.warning(f"No rollback target found for {model_id}")
            return False, None

        event = RollbackEvent(
            triggered_at=now,
            current_model_id=model_id,
            rollback_model_id=rollback_model,
            reason=reason,
            elo_regression=elo_regression,
            games_played=games_played,
            win_rate=win_rate,
            auto_triggered=True,
            board_type=board_type,
            num_players=num_players,
        )

        # Notify hooks about rollback being triggered
        self._notify_rollback_triggered(event)

        self._emit_check_metrics(model_id, True, elo_regression)
        return True, event

    def check_against_baselines(
        self,
        model_id: str,
        board_type: str = "square8",
        num_players: int = 2,
        baseline_model_ids: list[str] | None = None,
        num_baselines: int = 3,
    ) -> dict[str, Any]:
        """Compare model against multiple baseline models.

        Args:
            model_id: Current model to check
            board_type: Board type
            num_players: Number of players
            baseline_model_ids: Specific models to compare against (if None, use recent production)
            num_baselines: Number of recent models to compare against if baseline_model_ids not provided

        Returns:
            Dict with comparison results for each baseline
        """
        elo_service = self.controller.elo_service
        if not elo_service:
            return {"error": "No Elo service available"}

        # Get current model's rating
        try:
            current_rating = elo_service.get_rating(model_id, board_type, num_players)
            if not current_rating:
                return {"error": f"No rating found for {model_id}"}
        except Exception as e:
            return {"error": f"Failed to get rating: {e}"}

        current_elo = current_rating.rating

        # Dec 29, 2025 - Phase 5: Use adaptive threshold
        config_key = f"{board_type}_{num_players}p"
        adaptive_threshold = get_adaptive_regression_threshold(config_key, current_elo)

        # Get baseline models
        if baseline_model_ids is None:
            baseline_model_ids = self._get_recent_production_models(model_id, num_baselines)

        if not baseline_model_ids:
            return {
                "model_id": model_id,
                "current_elo": current_elo,
                "baselines": [],
                "summary": "No baseline models found",
            }

        # Compare against each baseline
        comparisons = []
        for baseline_id in baseline_model_ids:
            try:
                baseline_rating = elo_service.get_rating(baseline_id, board_type, num_players)
                if baseline_rating:
                    diff = current_elo - baseline_rating.rating
                    comparisons.append({
                        "baseline_id": baseline_id,
                        "baseline_elo": baseline_rating.rating,
                        "elo_diff": diff,
                        "is_regression": diff < adaptive_threshold,
                    })
            except (OSError, AttributeError, KeyError, TypeError, ValueError):
                comparisons.append({
                    "baseline_id": baseline_id,
                    "error": "Failed to get rating",
                })

        # Calculate summary stats
        valid_diffs = [c["elo_diff"] for c in comparisons if "elo_diff" in c]
        if valid_diffs:
            avg_diff = sum(valid_diffs) / len(valid_diffs)
            min_diff = min(valid_diffs)
            max_diff = max(valid_diffs)
            regressions = sum(1 for d in valid_diffs if d < adaptive_threshold)
        else:
            avg_diff = min_diff = max_diff = 0.0
            regressions = 0

        summary = "healthy"
        if regressions == len(valid_diffs) and len(valid_diffs) > 0:
            summary = "regression_against_all"
        elif regressions > len(valid_diffs) / 2:
            summary = "regression_against_majority"
        elif regressions > 0:
            summary = "regression_against_some"

        return {
            "model_id": model_id,
            "current_elo": current_elo,
            "games_played": current_rating.games_played,
            "baselines": comparisons,
            "avg_diff": avg_diff,
            "min_diff": min_diff,
            "max_diff": max_diff,
            "regressions": regressions,
            "total_baselines": len(valid_diffs),
            "summary": summary,
        }

    def _get_recent_production_models(
        self,
        exclude_model_id: str,
        count: int = 3,
    ) -> list[str]:
        """Get list of recent production models to use as baselines."""
        registry = self.controller.model_registry
        if not registry:
            return []

        try:
            history = registry.get_model_history(limit=count + 5)
            models = []
            for entry in history:
                model_id = entry.get("model_id")
                if (model_id and model_id != exclude_model_id
                        and entry.get("stage") in ("production", "staging")):
                    models.append(model_id)
                    if len(models) >= count:
                        break
            return models
        except (AttributeError, KeyError, TypeError):
            return []

    def _prune_history(self, model_id: str) -> None:
        """Prune old entries from regression history."""
        if model_id not in self._regression_history:
            return

        cutoff_time = datetime.now().timestamp() - self.criteria.time_window_seconds
        history = self._regression_history[model_id]

        # Keep only entries within time window (compare ISO strings)
        cutoff_iso = datetime.fromtimestamp(cutoff_time).isoformat()
        self._regression_history[model_id] = [
            (ts, val) for ts, val in history if ts >= cutoff_iso
        ]

    def _has_consecutive_regression(self, model_id: str) -> bool:
        """Check if model has consecutive regression checks exceeding threshold.

        Note: Uses static elo_regression_threshold for historical analysis.
        Adaptive thresholds are applied in check_for_regression() for immediate checks.
        """
        history = self._regression_history.get(model_id, [])
        if len(history) < self.criteria.consecutive_checks_required:
            return False

        # Check last N entries
        recent = history[-self.criteria.consecutive_checks_required:]
        return all(
            val < self.criteria.elo_regression_threshold
            for _, val in recent
        )

    def _get_average_regression(self, model_id: str) -> float:
        """Get average regression from recent history."""
        history = self._regression_history.get(model_id, [])
        if not history:
            return 0.0
        return sum(val for _, val in history) / len(history)

    def _get_previous_production_model(self, current_model_id: str) -> str | None:
        """Get the previous production model to rollback to."""
        registry = self.controller.model_registry
        if not registry:
            return None

        try:
            # Get model history
            history = registry.get_model_history(limit=10)
            found_current = False
            for entry in history:
                if entry.get("model_id") == current_model_id:
                    found_current = True
                    continue
                if found_current and entry.get("stage") in ("production", "staging"):
                    return entry.get("model_id")
        except (AttributeError, KeyError, TypeError):
            pass

        return None

    def _emit_check_metrics(
        self,
        model_id: str,
        triggered: bool,
        elo_regression: float | None,
    ) -> None:
        """Emit metrics for regression check."""
        try:
            from app.metrics import record_rollback_check
            status = self.get_regression_status(model_id)
            record_rollback_check(
                model_id=model_id,
                triggered=triggered,
                elo_regression=elo_regression or 0.0,
                at_risk=status.get("at_risk", False),
            )
        except ImportError:
            pass

    def execute_rollback(
        self,
        event: RollbackEvent,
        dry_run: bool = False,
    ) -> bool:
        """Execute an automatic rollback with graduated response.

        Args:
            event: The rollback event to execute
            dry_run: If True, only log what would happen

        Returns:
            True if rollback was successful
        """
        # Check for graduated response blocking
        blocked, block_reason = self.should_block_rollback()
        if blocked and not self._cooldown_bypass:
            logger.critical(
                f"ROLLBACK BLOCKED: {block_reason} for "
                f"{event.current_model_id} -> {event.rollback_model_id}"
            )
            # Notify hooks about blocked rollback
            for hook in self._hooks:
                try:
                    if hasattr(hook, 'on_rollback_blocked'):
                        hook.on_rollback_blocked(event, block_reason)
                except Exception as e:
                    logger.warning(f"Notification hook error on block: {e}")
            return False

        logger.warning(
            f"{'[DRY RUN] ' if dry_run else ''}Executing automatic rollback: "
            f"{event.current_model_id} -> {event.rollback_model_id} "
            f"(reason: {event.reason})"
        )

        if dry_run:
            self._emit_rollback_metrics(event, success=True, dry_run=True)
            return True

        # Apply graduated response (may trigger investigation, pause training, etc.)
        response_action = self.apply_graduated_response(event)

        # Create rollback decision
        decision = PromotionDecision(
            model_id=event.rollback_model_id,
            promotion_type=PromotionType.ROLLBACK,
            should_promote=True,
            reason=f"Auto-rollback from {event.current_model_id}: {event.reason} "
                   f"[response: {response_action.value}]",
            elo_improvement=event.elo_regression,
            games_played=event.games_played,
            win_rate=event.win_rate,
        )

        # Execute via controller
        success = self.controller.execute_promotion(decision)

        if success:
            self._rollback_events.append(event)
            # Clear regression history for the model we rolled back from
            if event.current_model_id in self._regression_history:
                del self._regression_history[event.current_model_id]
            # Record rollback time for cooldown tracking (using effective cooldown)
            self._record_rollback_time(event.board_type, event.num_players)

        # Notify hooks about rollback completion
        self._notify_rollback_completed(event, success)

        self._emit_rollback_metrics(event, success=success, dry_run=False)
        return success

    def _emit_rollback_metrics(
        self,
        event: RollbackEvent,
        success: bool,
        dry_run: bool,
    ) -> None:
        """Emit metrics for rollback execution."""
        try:
            from app.metrics import record_auto_rollback
            record_auto_rollback(
                from_model=event.current_model_id,
                to_model=event.rollback_model_id,
                success=success,
                dry_run=dry_run,
                reason=event.reason,
            )
        except ImportError:
            pass

    def get_rollback_history(self) -> list[RollbackEvent]:
        """Get history of rollback events."""
        return self._rollback_events.copy()

    def get_regression_status(self, model_id: str) -> dict[str, Any]:
        """Get current regression status for a model."""
        history = self._regression_history.get(model_id, [])
        if not history:
            return {
                "model_id": model_id,
                "checks": 0,
                "avg_regression": 0.0,
                "consecutive_regressions": 0,
                "at_risk": False,
            }

        # Count consecutive regressions
        consecutive = 0
        for _, val in reversed(history):
            if val < self.criteria.elo_regression_threshold:
                consecutive += 1
            else:
                break

        return {
            "model_id": model_id,
            "checks": len(history),
            "avg_regression": self._get_average_regression(model_id),
            "consecutive_regressions": consecutive,
            "at_risk": consecutive >= self.criteria.consecutive_checks_required - 1,
        }


# =============================================================================
# Singleton with State Machine Integration
# =============================================================================

_promotion_controller_singleton: PromotionController | None = None
_promotion_controller_lock = None  # Lazy init to avoid import issues


def _get_controller_lock():
    """Get or create the singleton lock."""
    global _promotion_controller_lock
    if _promotion_controller_lock is None:
        import threading
        _promotion_controller_lock = threading.Lock()
    return _promotion_controller_lock


def get_promotion_controller(
    criteria: PromotionCriteria | None = None,
    use_singleton: bool = True,
) -> PromotionController:
    """Get a configured promotion controller instance.

    Args:
        criteria: Optional promotion criteria
        use_singleton: If True (default), return singleton with state machine wired.
                      If False, create a new instance without state machine.

    Returns:
        PromotionController instance

    Note:
        The singleton is automatically wired to the ModelLifecycleStateMachine
        for audit trail tracking of all promotion decisions.
    """
    global _promotion_controller_singleton

    if not use_singleton:
        return PromotionController(criteria=criteria)

    if _promotion_controller_singleton is None:
        with _get_controller_lock():
            if _promotion_controller_singleton is None:
                controller = PromotionController(criteria=criteria)

                # Wire state machine integration for audit trail
                try:
                    from app.training.model_state_machine import PromotionControllerIntegration
                    integration = PromotionControllerIntegration()
                    integration.wire_promotion_controller(controller)
                    logger.info("[PromotionController] State machine integration wired")
                except ImportError:
                    logger.debug("[PromotionController] State machine not available")
                except Exception as e:
                    logger.warning(f"[PromotionController] Failed to wire state machine: {e}")

                _promotion_controller_singleton = controller

    return _promotion_controller_singleton


def reset_promotion_controller() -> None:
    """Reset the promotion controller singleton (for testing)."""
    global _promotion_controller_singleton
    with _get_controller_lock():
        _promotion_controller_singleton = None


def get_rollback_monitor(
    criteria: RollbackCriteria | None = None,
) -> RollbackMonitor:
    """Get a configured rollback monitor instance."""
    return RollbackMonitor(criteria=criteria)


# =============================================================================
# A/B Testing Support
# =============================================================================


@dataclass
class ABTestConfig:
    """Configuration for an A/B test experiment.

    Attributes:
        test_id: Unique identifier for the test
        control_model_id: The baseline/control model
        treatment_model_id: The new model being tested
        board_type: Board type for the test
        num_players: Number of players
        traffic_split: Fraction of traffic to route to treatment (0.0-1.0)
        min_games_per_variant: Minimum games before drawing conclusions
        significance_threshold: Statistical significance threshold (default 95%)
        auto_promote: Whether to auto-promote winner when test concludes
    """
    test_id: str
    control_model_id: str
    treatment_model_id: str
    board_type: str = "square8"
    num_players: int = 2
    traffic_split: float = 0.5  # 50/50 split by default
    min_games_per_variant: int = 100
    significance_threshold: float = 0.95
    auto_promote: bool = False
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "control_model_id": self.control_model_id,
            "treatment_model_id": self.treatment_model_id,
            "board_type": self.board_type,
            "num_players": self.num_players,
            "traffic_split": self.traffic_split,
            "min_games_per_variant": self.min_games_per_variant,
            "significance_threshold": self.significance_threshold,
            "auto_promote": self.auto_promote,
            "started_at": self.started_at,
        }


@dataclass
class ABTestResult:
    """Results of an A/B test comparison."""
    test_id: str
    control_elo: float
    treatment_elo: float
    elo_difference: float
    control_games: int
    treatment_games: int
    control_win_rate: float
    treatment_win_rate: float
    is_significant: bool
    winner: str | None  # "control", "treatment", or None if inconclusive
    confidence: float  # Statistical confidence (0.0-1.0)
    recommendation: str
    analyzed_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "control_elo": self.control_elo,
            "treatment_elo": self.treatment_elo,
            "elo_difference": self.elo_difference,
            "control_games": self.control_games,
            "treatment_games": self.treatment_games,
            "control_win_rate": self.control_win_rate,
            "treatment_win_rate": self.treatment_win_rate,
            "is_significant": self.is_significant,
            "winner": self.winner,
            "confidence": self.confidence,
            "recommendation": self.recommendation,
            "analyzed_at": self.analyzed_at,
        }


class ABTestManager:
    """Manages A/B testing experiments for model comparison.

    Allows running multiple model versions concurrently and comparing
    their Elo performance before full promotion.

    Usage:
        from app.training.promotion_controller import ABTestManager, ABTestConfig

        manager = ABTestManager()

        # Start a test
        config = ABTestConfig(
            test_id="test_v42_vs_v41",
            control_model_id="model_v41",
            treatment_model_id="model_v42",
            traffic_split=0.5,
        )
        manager.start_test(config)

        # Get model for a game (routes based on traffic split)
        model_id = manager.get_model_for_game("test_v42_vs_v41")

        # Check results
        result = manager.analyze_test("test_v42_vs_v41")
        if result.is_significant:
            print(f"Winner: {result.winner}")
    """

    def __init__(
        self,
        elo_service: Any | None = None,
        promotion_controller: PromotionController | None = None,
    ):
        self._elo_service = elo_service
        self._controller = promotion_controller
        # Active tests: test_id -> ABTestConfig
        self._active_tests: dict[str, ABTestConfig] = {}
        # Completed tests: test_id -> ABTestResult
        self._completed_tests: dict[str, ABTestResult] = {}
        # Random state for traffic routing (for reproducibility if needed)
        import random
        self._rng = random.Random()

    @property
    def elo_service(self):
        """Lazy-load EloService."""
        if self._elo_service is None:
            if self._controller:
                return self._controller.elo_service
            try:
                from app.training.elo_service import get_elo_service
                self._elo_service = get_elo_service()
            except ImportError:
                logger.warning("EloService not available")
        return self._elo_service

    @property
    def controller(self) -> PromotionController:
        """Lazy-load PromotionController."""
        if self._controller is None:
            self._controller = PromotionController()
        return self._controller

    def start_test(self, config: ABTestConfig) -> bool:
        """Start a new A/B test.

        Args:
            config: Test configuration

        Returns:
            True if test started successfully
        """
        if config.test_id in self._active_tests:
            logger.warning(f"Test {config.test_id} already active")
            return False

        if config.traffic_split < 0 or config.traffic_split > 1:
            logger.error(f"Invalid traffic split: {config.traffic_split}")
            return False

        self._active_tests[config.test_id] = config
        logger.info(
            f"Started A/B test {config.test_id}: "
            f"{config.control_model_id} vs {config.treatment_model_id} "
            f"({config.traffic_split:.0%} treatment traffic)"
        )
        return True

    def stop_test(self, test_id: str, analyze: bool = True) -> ABTestResult | None:
        """Stop an active A/B test.

        Args:
            test_id: Test identifier
            analyze: Whether to analyze results before stopping

        Returns:
            ABTestResult if analyze=True, None otherwise
        """
        if test_id not in self._active_tests:
            logger.warning(f"Test {test_id} not found")
            return None

        result = None
        if analyze:
            result = self.analyze_test(test_id)
            if result:
                self._completed_tests[test_id] = result

        del self._active_tests[test_id]
        logger.info(f"Stopped A/B test {test_id}")
        return result

    def get_model_for_game(self, test_id: str) -> str | None:
        """Get the model to use for a game based on traffic routing.

        Args:
            test_id: Test identifier

        Returns:
            Model ID (control or treatment) based on traffic split
        """
        if test_id not in self._active_tests:
            return None

        config = self._active_tests[test_id]
        if self._rng.random() < config.traffic_split:
            return config.treatment_model_id
        return config.control_model_id

    def get_all_test_models(self, test_id: str) -> tuple[str | None, str | None]:
        """Get both models in a test.

        Args:
            test_id: Test identifier

        Returns:
            Tuple of (control_model_id, treatment_model_id)
        """
        if test_id not in self._active_tests:
            return None, None
        config = self._active_tests[test_id]
        return config.control_model_id, config.treatment_model_id

    def analyze_test(self, test_id: str) -> ABTestResult | None:
        """Analyze the current results of an A/B test.

        Args:
            test_id: Test identifier

        Returns:
            ABTestResult with analysis
        """
        if test_id not in self._active_tests:
            logger.warning(f"Test {test_id} not found")
            return None

        config = self._active_tests[test_id]

        if not self.elo_service:
            logger.error("No Elo service available for analysis")
            return None

        # Get ratings for both models
        try:
            control_rating = self.elo_service.get_rating(
                config.control_model_id, config.board_type, config.num_players
            )
            treatment_rating = self.elo_service.get_rating(
                config.treatment_model_id, config.board_type, config.num_players
            )
        except Exception as e:
            logger.error(f"Failed to get ratings for test {test_id}: {e}")
            return None

        if not control_rating or not treatment_rating:
            logger.warning(f"Missing ratings for test {test_id}")
            return ABTestResult(
                test_id=test_id,
                control_elo=0.0,
                treatment_elo=0.0,
                elo_difference=0.0,
                control_games=0,
                treatment_games=0,
                control_win_rate=0.0,
                treatment_win_rate=0.0,
                is_significant=False,
                winner=None,
                confidence=0.0,
                recommendation="Insufficient data - waiting for more games",
            )

        control_elo = control_rating.rating
        treatment_elo = treatment_rating.rating
        elo_diff = treatment_elo - control_elo

        control_games = control_rating.games_played
        treatment_games = treatment_rating.games_played

        control_win_rate = getattr(control_rating, 'win_rate', 0.5) or 0.5
        treatment_win_rate = getattr(treatment_rating, 'win_rate', 0.5) or 0.5

        # Check if we have enough games
        min_games = config.min_games_per_variant
        has_enough_games = (
            control_games >= min_games and treatment_games >= min_games
        )

        # Calculate statistical significance using Elo difference
        # Rule of thumb: ~30 Elo difference is statistically significant
        # with 100+ games each
        confidence = self._calculate_confidence(
            elo_diff, control_games, treatment_games
        )
        is_significant = confidence >= config.significance_threshold and has_enough_games

        # Determine winner
        winner = None
        if is_significant:
            if elo_diff > 25:  # Treatment is significantly better
                winner = "treatment"
            elif elo_diff < -25:  # Control is significantly better
                winner = "control"

        # Generate recommendation
        if not has_enough_games:
            recommendation = (
                f"Need more games: control={control_games}/{min_games}, "
                f"treatment={treatment_games}/{min_games}"
            )
        elif not is_significant:
            recommendation = (
                f"Results not significant (confidence={confidence:.1%}). "
                f"Elo diff: {elo_diff:+.1f}"
            )
        elif winner == "treatment":
            recommendation = (
                f"Treatment model ({config.treatment_model_id}) is significantly better. "
                f"Consider promoting. Elo: {elo_diff:+.1f}"
            )
        elif winner == "control":
            recommendation = (
                f"Control model ({config.control_model_id}) is better. "
                f"Treatment shows regression: {elo_diff:+.1f} Elo"
            )
        else:
            recommendation = (
                f"Models are statistically equivalent (diff={elo_diff:+.1f}). "
                f"Consider other factors for decision."
            )

        result = ABTestResult(
            test_id=test_id,
            control_elo=control_elo,
            treatment_elo=treatment_elo,
            elo_difference=elo_diff,
            control_games=control_games,
            treatment_games=treatment_games,
            control_win_rate=control_win_rate,
            treatment_win_rate=treatment_win_rate,
            is_significant=is_significant,
            winner=winner,
            confidence=confidence,
            recommendation=recommendation,
        )

        # Auto-promote if configured and winner is clear
        if config.auto_promote and is_significant and winner == "treatment":
            self._auto_promote_winner(config, result)

        return result

    def _calculate_confidence(
        self,
        elo_diff: float,
        control_games: int,
        treatment_games: int,
    ) -> float:
        """Calculate statistical confidence of Elo difference.

        Uses a simplified model based on expected Elo variance.
        Standard error of Elo estimate ≈ 400 / sqrt(n) for random opponents.

        Args:
            elo_diff: Difference in Elo (treatment - control)
            control_games: Number of games played by control
            treatment_games: Number of games played by treatment

        Returns:
            Confidence level (0.0 - 1.0)
        """
        import math

        if control_games == 0 or treatment_games == 0:
            return 0.0

        # Approximate standard error of the difference
        # SE ≈ sqrt(SE_control^2 + SE_treatment^2)
        se_control = 400 / math.sqrt(control_games)
        se_treatment = 400 / math.sqrt(treatment_games)
        se_diff = math.sqrt(se_control**2 + se_treatment**2)

        if se_diff == 0:
            return 1.0

        # Z-score for the difference
        z_score = abs(elo_diff) / se_diff

        # Convert Z-score to confidence using normal CDF approximation
        # P(|Z| > z) ≈ 2 * (1 - Φ(z))
        # Confidence = 1 - P(|Z| > z) = 2 * Φ(z) - 1
        def norm_cdf(x):
            """Approximation of normal CDF."""
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))

        confidence = 2 * norm_cdf(z_score) - 1
        return min(max(confidence, 0.0), 1.0)

    def _auto_promote_winner(
        self,
        config: ABTestConfig,
        result: ABTestResult,
    ) -> bool:
        """Auto-promote the winning model.

        Args:
            config: Test configuration
            result: Test result showing treatment won

        Returns:
            True if promotion was successful
        """
        logger.info(
            f"Auto-promoting winner of test {config.test_id}: "
            f"{config.treatment_model_id} (Elo: {result.elo_difference:+.1f})"
        )

        decision = self.controller.evaluate_promotion(
            model_id=config.treatment_model_id,
            board_type=config.board_type,
            num_players=config.num_players,
            promotion_type=PromotionType.PRODUCTION,
            baseline_model_id=config.control_model_id,
        )

        if decision.should_promote:
            return self.controller.execute_promotion(decision)
        else:
            logger.warning(
                f"A/B test winner {config.treatment_model_id} did not pass "
                f"standard promotion criteria: {decision.reason}"
            )
            return False

    def list_active_tests(self) -> list[ABTestConfig]:
        """Get all active tests."""
        return list(self._active_tests.values())

    def list_completed_tests(self) -> list[ABTestResult]:
        """Get all completed test results."""
        return list(self._completed_tests.values())

    def get_test_status(self, test_id: str) -> dict[str, Any] | None:
        """Get detailed status of a test.

        Args:
            test_id: Test identifier

        Returns:
            Dict with test config and current analysis
        """
        if test_id not in self._active_tests:
            # Check completed tests
            if test_id in self._completed_tests:
                return {
                    "status": "completed",
                    "result": self._completed_tests[test_id].to_dict(),
                }
            return None

        config = self._active_tests[test_id]
        result = self.analyze_test(test_id)

        return {
            "status": "active",
            "config": config.to_dict(),
            "current_results": result.to_dict() if result else None,
        }


def get_ab_test_manager(
    elo_service: Any | None = None,
) -> ABTestManager:
    """Get a configured A/B test manager instance."""
    return ABTestManager(elo_service=elo_service)
