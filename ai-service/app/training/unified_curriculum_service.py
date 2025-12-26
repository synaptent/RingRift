"""Unified Curriculum Weighting Service.

Single source of truth for curriculum weights across the training pipeline.
Consolidates weights from multiple sources:
- CurriculumFeedback: win rate, Elo trend, weak opponents, model count
- FeedbackAccelerator: momentum state (accelerating, improving, plateau, regressing)
- ImprovementOptimizer: promotion success/failure, threshold adjustments

December 2025: Created as part of Phase 4 feedback loop unification.

Usage:
    from app.training.unified_curriculum_service import (
        get_unified_curriculum_weights,
        UnifiedCurriculumService,
    )

    # Quick usage
    weights = get_unified_curriculum_weights()
    # {"hex8_2p": 1.3, "square8_2p": 0.9, ...}

    # Full service for detailed access
    service = get_unified_curriculum_service()
    weights = service.get_weights()
    breakdown = service.get_weight_breakdown("hex8_2p")
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .curriculum_feedback import CurriculumFeedback
    from .feedback_accelerator import FeedbackAccelerator
    from .improvement_optimizer import ImprovementOptimizer

logger = logging.getLogger(__name__)

# Weight bounds
WEIGHT_MIN = 0.3
WEIGHT_MAX = 3.0
WEIGHT_DEFAULT = 1.0


@dataclass
class WeightBreakdown:
    """Detailed breakdown of weight components for a config."""

    config_key: str
    final_weight: float

    # Component weights (multiplicative)
    base_weight: float = 1.0
    win_rate_factor: float = 1.0
    elo_trend_factor: float = 1.0
    model_count_factor: float = 1.0
    weak_opponent_factor: float = 1.0
    momentum_factor: float = 1.0
    promotion_factor: float = 1.0
    staleness_factor: float = 1.0

    # Source availability
    has_curriculum_feedback: bool = False
    has_feedback_accelerator: bool = False
    has_improvement_optimizer: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "config_key": self.config_key,
            "final_weight": round(self.final_weight, 3),
            "components": {
                "base": round(self.base_weight, 3),
                "win_rate": round(self.win_rate_factor, 3),
                "elo_trend": round(self.elo_trend_factor, 3),
                "model_count": round(self.model_count_factor, 3),
                "weak_opponent": round(self.weak_opponent_factor, 3),
                "momentum": round(self.momentum_factor, 3),
                "promotion": round(self.promotion_factor, 3),
                "staleness": round(self.staleness_factor, 3),
            },
            "sources": {
                "curriculum_feedback": self.has_curriculum_feedback,
                "feedback_accelerator": self.has_feedback_accelerator,
                "improvement_optimizer": self.has_improvement_optimizer,
            },
        }


@dataclass
class UnifiedCurriculumState:
    """Cached state for unified curriculum weights."""

    weights: dict[str, float] = field(default_factory=dict)
    breakdowns: dict[str, WeightBreakdown] = field(default_factory=dict)
    last_update_time: float = 0.0


class UnifiedCurriculumService:
    """
    Unified service for computing curriculum weights.

    Combines signals from multiple sources:
    1. CurriculumFeedback: Performance-based weights (win rate, Elo)
    2. FeedbackAccelerator: Momentum-based weights (improvement trends)
    3. ImprovementOptimizer: Success-based weights (promotion history)

    The final weight is computed multiplicatively:
        final = base * win_rate * elo_trend * momentum * promotion * ...

    All components are normalized to have mean ~1.0, so multiplication
    produces a balanced result.
    """

    def __init__(
        self,
        weight_min: float = WEIGHT_MIN,
        weight_max: float = WEIGHT_MAX,
        curriculum_feedback: CurriculumFeedback | None = None,
        feedback_accelerator: FeedbackAccelerator | None = None,
        improvement_optimizer: ImprovementOptimizer | None = None,
    ):
        """Initialize the unified curriculum service.

        Args:
            weight_min: Minimum allowed weight
            weight_max: Maximum allowed weight
            curriculum_feedback: Optional CurriculumFeedback instance
            feedback_accelerator: Optional FeedbackAccelerator instance
            improvement_optimizer: Optional ImprovementOptimizer instance
        """
        self.weight_min = weight_min
        self.weight_max = weight_max

        # Lazy-loaded source instances
        self._curriculum_feedback = curriculum_feedback
        self._feedback_accelerator = feedback_accelerator
        self._improvement_optimizer = improvement_optimizer

        # State
        self._state = UnifiedCurriculumState()
        self._lock = threading.Lock()

    def _get_curriculum_feedback(self) -> CurriculumFeedback | None:
        """Lazy-load CurriculumFeedback instance."""
        if self._curriculum_feedback is not None:
            return self._curriculum_feedback
        try:
            from .curriculum_feedback import get_curriculum_feedback
            self._curriculum_feedback = get_curriculum_feedback()
            return self._curriculum_feedback
        except ImportError:
            logger.debug("CurriculumFeedback not available")
            return None

    def _get_feedback_accelerator(self) -> FeedbackAccelerator | None:
        """Lazy-load FeedbackAccelerator instance."""
        if self._feedback_accelerator is not None:
            return self._feedback_accelerator
        try:
            from .feedback_accelerator import get_feedback_accelerator
            self._feedback_accelerator = get_feedback_accelerator()
            return self._feedback_accelerator
        except ImportError:
            logger.debug("FeedbackAccelerator not available")
            return None

    def _get_improvement_optimizer(self) -> ImprovementOptimizer | None:
        """Lazy-load ImprovementOptimizer instance."""
        if self._improvement_optimizer is not None:
            return self._improvement_optimizer
        try:
            from .improvement_optimizer import get_improvement_optimizer
            self._improvement_optimizer = get_improvement_optimizer()
            return self._improvement_optimizer
        except ImportError:
            logger.debug("ImprovementOptimizer not available")
            return None

    def get_weights(self) -> dict[str, float]:
        """Get unified curriculum weights for all configs.

        Returns:
            Dict mapping config_key -> weight (WEIGHT_MIN to WEIGHT_MAX)
        """
        with self._lock:
            self._compute_all_weights()
            return dict(self._state.weights)

    def get_weight(self, config_key: str) -> float:
        """Get unified curriculum weight for a single config.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")

        Returns:
            Weight value (WEIGHT_MIN to WEIGHT_MAX), or 1.0 if unknown
        """
        with self._lock:
            if config_key not in self._state.weights:
                self._compute_all_weights()
            return self._state.weights.get(config_key, WEIGHT_DEFAULT)

    def get_weight_breakdown(self, config_key: str) -> WeightBreakdown | None:
        """Get detailed breakdown of weight components for a config.

        Args:
            config_key: Configuration key

        Returns:
            WeightBreakdown with component details, or None if not found
        """
        with self._lock:
            if config_key not in self._state.breakdowns:
                self._compute_all_weights()
            return self._state.breakdowns.get(config_key)

    def _compute_all_weights(self) -> None:
        """Compute weights for all known configs."""
        # Collect all known config keys from all sources
        all_config_keys: set[str] = set()

        # Get curriculum feedback data
        cf = self._get_curriculum_feedback()
        cf_weights: dict[str, float] = {}
        cf_metrics: dict = {}
        if cf:
            try:
                cf_weights = cf.get_curriculum_weights()
                cf_metrics = cf.get_all_metrics()
                all_config_keys.update(cf_weights.keys())
            except Exception as e:
                logger.warning(f"Failed to get curriculum feedback weights: {e}")

        # Get feedback accelerator data
        fa = self._get_feedback_accelerator()
        fa_weights: dict[str, float] = {}
        if fa:
            try:
                fa_weights = fa.get_curriculum_weights()
                all_config_keys.update(fa_weights.keys())
            except Exception as e:
                logger.warning(f"Failed to get feedback accelerator weights: {e}")

        # Get improvement optimizer data
        io = self._get_improvement_optimizer()
        io_boosts: dict[str, float] = {}
        if io:
            try:
                io_boosts = io._state.config_boosts
                all_config_keys.update(io_boosts.keys())
            except Exception as e:
                logger.warning(f"Failed to get improvement optimizer boosts: {e}")

        # Compute unified weights for each config
        new_weights: dict[str, float] = {}
        new_breakdowns: dict[str, WeightBreakdown] = {}

        for config_key in all_config_keys:
            breakdown = self._compute_config_weight(
                config_key, cf, cf_weights, cf_metrics, fa, fa_weights, io, io_boosts
            )
            new_weights[config_key] = breakdown.final_weight
            new_breakdowns[config_key] = breakdown

        self._state.weights = new_weights
        self._state.breakdowns = new_breakdowns

    def _compute_config_weight(
        self,
        config_key: str,
        cf: CurriculumFeedback | None,
        cf_weights: dict[str, float],
        cf_metrics: dict,
        fa: FeedbackAccelerator | None,
        fa_weights: dict[str, float],
        io: ImprovementOptimizer | None,
        io_boosts: dict[str, float],
    ) -> WeightBreakdown:
        """Compute unified weight for a single config.

        Strategy: Start with base 1.0, apply multiplicative factors from each source.
        Each factor is normalized to have mean ~1.0 so multiplication is balanced.
        """
        breakdown = WeightBreakdown(config_key=config_key, final_weight=WEIGHT_DEFAULT)
        weight = 1.0

        # 1. CurriculumFeedback contributions
        if cf and config_key in cf_weights:
            breakdown.has_curriculum_feedback = True

            # The cf_weight already incorporates win_rate, elo_trend, etc.
            # We use it as a combined factor but normalize around 1.0
            # cf_weights are typically 0.5-2.0, so we scale to 0.7-1.4 range
            cf_weight = cf_weights[config_key]
            normalized_cf = 0.5 + (cf_weight - 0.5) * 0.5  # Maps 0.5-2.0 -> 0.5-1.25
            weight *= normalized_cf

            # Store component factors if metrics available
            if config_key in cf_metrics:
                metrics = cf_metrics[config_key]
                # Win rate factor (low win rate -> higher weight)
                target_wr = getattr(cf, 'target_win_rate', 0.55)
                wr = getattr(metrics, 'win_rate', 0.5)
                wr_diff = target_wr - wr
                breakdown.win_rate_factor = 1.0 + wr_diff * 0.5  # Scale down impact

                # Elo trend factor
                elo_trend = getattr(metrics, 'elo_trend', 0.0)
                if elo_trend < -20:
                    breakdown.elo_trend_factor = 1.1
                elif elo_trend > 30:
                    breakdown.elo_trend_factor = 0.95
                else:
                    breakdown.elo_trend_factor = 1.0

                # Model count factor
                model_count = getattr(metrics, 'model_count', 1)
                if model_count == 0:
                    breakdown.model_count_factor = 1.3
                elif model_count == 1:
                    breakdown.model_count_factor = 1.1
                else:
                    breakdown.model_count_factor = 1.0

        # 2. FeedbackAccelerator contributions (momentum)
        if fa and config_key in fa_weights:
            breakdown.has_feedback_accelerator = True

            # fa_weights are typically 1.0-2.0 based on momentum state
            fa_weight = fa_weights[config_key]
            # Normalize: momentum weights boost training for improving/stuck configs
            # Scale to 0.9-1.3 range to avoid over-amplification
            breakdown.momentum_factor = 0.8 + fa_weight * 0.2  # Maps 1.0-2.0 -> 1.0-1.2
            weight *= breakdown.momentum_factor

        # 3. ImprovementOptimizer contributions (promotion success)
        if io and config_key in io_boosts:
            breakdown.has_improvement_optimizer = True

            # io_boosts are <1.0 for successful configs (faster iteration)
            # We invert this: successful configs should still get training attention
            io_boost = io_boosts[config_key]
            # Lower boost = more successful = less urgent training need
            # But we don't want to starve successful configs, so limit impact
            breakdown.promotion_factor = 0.9 + io_boost * 0.2  # Maps 0.5-1.0 -> 1.0-1.1
            weight *= breakdown.promotion_factor

        # Apply bounds
        breakdown.final_weight = max(self.weight_min, min(self.weight_max, weight))
        breakdown.base_weight = 1.0

        return breakdown

    def refresh(self) -> None:
        """Force refresh of all weights."""
        with self._lock:
            self._state.weights.clear()
            self._state.breakdowns.clear()
            self._compute_all_weights()


# Singleton instance
_unified_service: UnifiedCurriculumService | None = None
_service_lock = threading.Lock()


def get_unified_curriculum_service() -> UnifiedCurriculumService:
    """Get the singleton UnifiedCurriculumService instance."""
    global _unified_service
    with _service_lock:
        if _unified_service is None:
            _unified_service = UnifiedCurriculumService()
        return _unified_service


def get_unified_curriculum_weights() -> dict[str, float]:
    """Get unified curriculum weights (convenience function).

    Returns:
        Dict mapping config_key -> weight
    """
    return get_unified_curriculum_service().get_weights()


def get_unified_curriculum_weight(config_key: str) -> float:
    """Get unified curriculum weight for a single config (convenience function).

    Args:
        config_key: Configuration key (e.g., "hex8_2p")

    Returns:
        Weight value, or 1.0 if unknown
    """
    return get_unified_curriculum_service().get_weight(config_key)
