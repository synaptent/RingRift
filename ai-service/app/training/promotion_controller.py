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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PromotionType(Enum):
    """Types of promotion decisions."""
    STAGING = "staging"          # Development → Staging
    PRODUCTION = "production"    # Staging → Production
    TIER = "tier"                # Tier ladder promotion (D1→D2, etc.)
    CHAMPION = "champion"        # Tournament champion promotion
    ROLLBACK = "rollback"        # Rollback to previous version


@dataclass
class PromotionCriteria:
    """Criteria for promotion evaluation.

    These are the canonical thresholds - sourced from unified_config.py.
    """
    min_elo_improvement: float = 25.0
    min_games_played: int = 50
    min_win_rate: float = 0.52
    max_value_mse_degradation: float = 0.05
    confidence_threshold: float = 0.95

    # Tier-specific
    tier_elo_threshold: Optional[float] = None
    tier_games_required: int = 100


@dataclass
class PromotionDecision:
    """Result of a promotion evaluation."""
    model_id: str
    promotion_type: PromotionType
    should_promote: bool
    reason: str

    # Evaluation metrics
    current_elo: Optional[float] = None
    elo_improvement: Optional[float] = None
    games_played: int = 0
    win_rate: Optional[float] = None
    confidence: Optional[float] = None

    # For tier promotions
    current_tier: Optional[str] = None
    target_tier: Optional[str] = None

    # Metadata
    evaluated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    criteria_used: Optional[PromotionCriteria] = None

    def to_dict(self) -> Dict[str, Any]:
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
            "evaluated_at": self.evaluated_at,
        }


class PromotionController:
    """Unified controller for all model promotion decisions.

    Consolidates promotion logic from multiple systems while preserving
    backward compatibility with existing code.
    """

    def __init__(
        self,
        criteria: Optional[PromotionCriteria] = None,
        elo_service: Optional[Any] = None,
        model_registry: Optional[Any] = None,
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
                    ModelLifecycleManager,
                    LifecycleConfig,
                )
                self._lifecycle_manager = ModelLifecycleManager(LifecycleConfig())
            except ImportError:
                logger.warning("ModelLifecycleManager not available")
        return self._lifecycle_manager

    def evaluate_promotion(
        self,
        model_id: str,
        board_type: str = "square8",
        num_players: int = 2,
        promotion_type: PromotionType = PromotionType.PRODUCTION,
        baseline_model_id: Optional[str] = None,
        **kwargs,
    ) -> PromotionDecision:
        """Evaluate whether a model should be promoted.

        Args:
            model_id: ID of the model to evaluate
            board_type: Board type for evaluation
            num_players: Number of players
            promotion_type: Type of promotion to evaluate
            baseline_model_id: Model to compare against (for PRODUCTION/CHAMPION)
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
        baseline_model_id: Optional[str],
    ) -> PromotionDecision:
        """Evaluate standard staging/production promotion."""
        # Get current Elo and stats
        current_elo = None
        games_played = 0
        win_rate = None
        elo_improvement = None

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

        # Check promotion criteria
        should_promote = False
        reason = ""

        if games_played < self.criteria.min_games_played:
            reason = f"Insufficient games ({games_played} < {self.criteria.min_games_played})"
        elif elo_improvement is not None and elo_improvement < self.criteria.min_elo_improvement:
            reason = f"Insufficient Elo improvement ({elo_improvement:.1f} < {self.criteria.min_elo_improvement})"
        elif win_rate is not None and win_rate < self.criteria.min_win_rate:
            reason = f"Win rate too low ({win_rate:.2%} < {self.criteria.min_win_rate:.2%})"
        else:
            should_promote = True
            reason = f"Meets all criteria: Elo +{elo_improvement or 0:.1f}, {games_played} games"

        decision = PromotionDecision(
            model_id=model_id,
            promotion_type=promotion_type,
            should_promote=should_promote,
            reason=reason,
            current_elo=current_elo,
            elo_improvement=elo_improvement,
            games_played=games_played,
            win_rate=win_rate,
            criteria_used=self.criteria,
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
        current_tier: Optional[str] = None,
        target_tier: Optional[str] = None,
        **kwargs,
    ) -> PromotionDecision:
        """Evaluate tier-based promotion for difficulty ladder."""
        # Load tier registry
        try:
            from app.training.tier_promotion_registry import (
                load_square8_two_player_registry,
                get_current_ladder_model_for_tier,
            )
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
        baseline_model_id: Optional[str],
        tournament_results: Optional[Dict] = None,
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
        regression_threshold: Optional[float] = None,
        **kwargs,
    ) -> PromotionDecision:
        """Evaluate whether to rollback to a previous model."""
        threshold = regression_threshold or -30.0

        # Get recent Elo trend
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
            else:
                logger.error(f"Unknown promotion type: {decision.promotion_type}")
                success = False
        except Exception as e:
            logger.error(f"Promotion failed for {decision.model_id}: {e}")
            success = False

        self._emit_execution_metrics(decision, success=success, dry_run=False)
        return success

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
                return True
            except Exception as e:
                logger.error(f"Model registry promotion failed: {e}")
        return False

    def _execute_tier_promotion(self, decision: PromotionDecision) -> bool:
        """Execute tier promotion."""
        try:
            from app.training.tier_promotion_registry import (
                load_square8_two_player_registry,
                save_square8_two_player_registry,
            )

            registry = load_square8_two_player_registry()
            tier = decision.target_tier

            if tier and tier not in registry.get("tiers", {}):
                registry.setdefault("tiers", {})[tier] = {}

            if tier:
                registry["tiers"][tier]["promoted_model"] = decision.model_id
                registry["tiers"][tier]["promoted_at"] = decision.evaluated_at
                registry["tiers"][tier]["elo"] = decision.current_elo

                save_square8_two_player_registry(registry)
                logger.info(f"Promoted {decision.model_id} to tier {tier}")
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


# Convenience function
def get_promotion_controller(
    criteria: Optional[PromotionCriteria] = None,
) -> PromotionController:
    """Get a configured promotion controller instance."""
    return PromotionController(criteria=criteria)
