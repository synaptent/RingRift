"""Improvement Optimizer for maximizing AI training throughput and positive feedback.

This module provides proactive optimization of the self-improvement loop:
1. Dynamic training thresholds based on cluster utilization and data quality
2. Positive feedback signals that accelerate successful patterns
3. Fast-path training for high-quality data
4. Adaptive evaluation frequency
5. Success acceleration - when things go well, go faster

Goals:
- Target 60-80% cluster utilization
- Maximize training iterations per day
- Reward successful promotions with faster cycles
- Detect and amplify winning strategies

Usage:
    from app.training.improvement_optimizer import (
        ImprovementOptimizer,
        get_improvement_optimizer,
        should_fast_track_training,
        get_dynamic_threshold,
    )

    optimizer = get_improvement_optimizer()

    # Check if we should fast-track training
    if optimizer.should_fast_track_training(config_key):
        threshold = optimizer.get_dynamic_threshold(config_key)
        # Use lower threshold for faster iteration
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from app.utils.paths import AI_SERVICE_ROOT

logger = logging.getLogger(__name__)
OPTIMIZER_STATE_PATH = AI_SERVICE_ROOT / "logs" / "improvement_optimizer_state.json"


class ImprovementSignal(str, Enum):
    """Signals for positive reinforcement in the improvement loop."""

    # Success signals (accelerate the pipeline)
    PROMOTION_STREAK = "promotion_streak"        # Multiple successful promotions
    ELO_BREAKTHROUGH = "elo_breakthrough"        # Large Elo gain
    QUALITY_DATA_SURGE = "quality_data_surge"    # High-quality games arriving fast
    CALIBRATION_EXCELLENT = "calibration_excellent"  # Model well-calibrated
    EFFICIENCY_OPTIMAL = "efficiency_optimal"    # Cluster running at 60-80%

    # Momentum signals (maintain current pace)
    STEADY_IMPROVEMENT = "steady_improvement"    # Consistent positive trend
    HEALTHY_PIPELINE = "healthy_pipeline"        # All stages functioning well

    # Opportunity signals (can safely push harder)
    UNDERUTILIZED_CAPACITY = "underutilized_capacity"  # Cluster has headroom
    DATA_QUALITY_HIGH = "data_quality_high"      # Parity validation passing
    LOW_QUEUE_DEPTH = "low_queue_depth"          # Not backlogged

    # Warning signals (slow down pipeline)
    REGRESSION_DETECTED = "regression_detected"  # Model performance regressing


@dataclass
class ImprovementState:
    """Persistent state for improvement optimization."""

    # Success tracking
    consecutive_promotions: int = 0
    total_promotions_24h: int = 0
    last_promotion_time: float = 0.0
    promotion_times: list[float] = field(default_factory=list)

    # Performance metrics
    avg_elo_gain_per_promotion: float = 25.0
    best_elo_gain: float = 0.0
    elo_gains: list[float] = field(default_factory=list)

    # Throughput tracking
    training_runs_24h: int = 0
    training_times: list[float] = field(default_factory=list)
    avg_training_duration_seconds: float = 3600.0

    # Quality metrics
    data_quality_score: float = 1.0
    parity_success_rate: float = 1.0
    calibration_ece: float = 0.1

    # Dynamic thresholds (adjusted by success)
    threshold_multiplier: float = 1.0  # 1.0 = baseline, <1.0 = faster training
    evaluation_frequency_multiplier: float = 1.0  # <1.0 = more frequent

    # Config-specific boosts
    config_boosts: dict[str, float] = field(default_factory=dict)

    # Last update
    updated_at: float = field(default_factory=time.time)


@dataclass
class OptimizationRecommendation:
    """A recommendation for pipeline optimization."""

    signal: ImprovementSignal
    config_key: str
    threshold_adjustment: float  # Multiplier for training threshold
    evaluation_adjustment: float  # Multiplier for evaluation interval
    reason: str
    confidence: float  # 0-1, how confident in this recommendation
    metadata: dict[str, Any] = field(default_factory=dict)


class ImprovementOptimizer:
    """Optimizer for maximizing self-improvement throughput and quality.

    This class provides:
    1. Dynamic threshold adjustment based on recent success
    2. Positive feedback amplification
    3. Fast-path detection for high-quality scenarios
    4. Success streak tracking and rewards
    """

    _instance: ImprovementOptimizer | None = None
    _lock = threading.RLock()

    # Baseline thresholds (from unified_config.py)
    BASELINE_TRAINING_THRESHOLD = 500
    BASELINE_EVALUATION_INTERVAL = 900  # 15 min shadow, 3600 full

    # Optimization bounds
    MIN_THRESHOLD_MULTIPLIER = 0.4  # Can reduce threshold to 40% (200 games)
    MAX_THRESHOLD_MULTIPLIER = 1.5  # Can increase to 150% (750 games)
    MIN_EVAL_MULTIPLIER = 0.5       # Can evaluate 2x as often
    MAX_EVAL_MULTIPLIER = 2.0       # Can slow down to half frequency

    # Success acceleration factors
    PROMOTION_STREAK_BONUS = 0.1    # 10% faster per consecutive promotion
    ELO_BREAKTHROUGH_BONUS = 0.15   # 15% faster for large Elo gains
    QUALITY_DATA_BONUS = 0.08       # 8% faster for high-quality data

    # Thresholds for positive signals
    PROMOTION_STREAK_THRESHOLD = 3  # 3+ promotions in a row
    ELO_BREAKTHROUGH_THRESHOLD = 50  # 50+ Elo gain
    QUALITY_DATA_THRESHOLD = 0.98   # 98%+ parity success rate
    CALIBRATION_EXCELLENT_THRESHOLD = 0.05  # ECE < 5%

    def __init__(self, state_path: Path | None = None):
        self._state_path = state_path or OPTIMIZER_STATE_PATH
        self._state = ImprovementState()
        self._callbacks: list[Callable[[OptimizationRecommendation], None]] = []
        self._load_state()

    @classmethod
    def get_instance(cls, state_path: Path | None = None) -> ImprovementOptimizer:
        """Get or create singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(state_path)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton for testing."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._save_state()
            cls._instance = None

    def register_callback(self, callback: Callable[[OptimizationRecommendation], None]) -> None:
        """Register callback for optimization recommendations."""
        self._callbacks.append(callback)

    # =========================================================================
    # State Management
    # =========================================================================

    def _load_state(self) -> None:
        """Load state from disk."""
        try:
            if self._state_path.exists():
                with open(self._state_path) as f:
                    data = json.load(f)
                    # Update state with loaded values
                    for key, value in data.items():
                        if hasattr(self._state, key):
                            setattr(self._state, key, value)
                logger.info(f"[ImprovementOptimizer] Loaded state from {self._state_path}")
        except Exception as e:
            logger.warning(f"[ImprovementOptimizer] Failed to load state: {e}")

    def _save_state(self) -> None:
        """Save state to disk."""
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state.updated_at = time.time()
            with open(self._state_path, "w") as f:
                json.dump(asdict(self._state), f, indent=2)
        except Exception as e:
            logger.warning(f"[ImprovementOptimizer] Failed to save state: {e}")

    def _cleanup_old_times(self) -> None:
        """Remove timestamps older than 24 hours."""
        cutoff = time.time() - 86400

        self._state.promotion_times = [t for t in self._state.promotion_times if t > cutoff]
        self._state.training_times = [t for t in self._state.training_times if t > cutoff]
        self._state.total_promotions_24h = len(self._state.promotion_times)
        self._state.training_runs_24h = len(self._state.training_times)

    # =========================================================================
    # Success Event Handlers
    # =========================================================================

    def record_promotion_success(
        self,
        config_key: str,
        elo_gain: float,
        model_id: str = "",
    ) -> OptimizationRecommendation:
        """Record a successful model promotion.

        This is a key positive feedback event that can accelerate the pipeline.
        """
        now = time.time()
        self._cleanup_old_times()

        # Update streak
        self._state.consecutive_promotions += 1
        self._state.last_promotion_time = now
        self._state.promotion_times.append(now)
        self._state.total_promotions_24h = len(self._state.promotion_times)

        # Track Elo gains
        self._state.elo_gains.append(elo_gain)
        if len(self._state.elo_gains) > 50:
            self._state.elo_gains = self._state.elo_gains[-50:]
        self._state.avg_elo_gain_per_promotion = sum(self._state.elo_gains) / len(self._state.elo_gains)

        if elo_gain > self._state.best_elo_gain:
            self._state.best_elo_gain = elo_gain

        # Calculate acceleration
        acceleration = 0.0
        signal = ImprovementSignal.STEADY_IMPROVEMENT
        reason_parts = []

        # Promotion streak bonus
        if self._state.consecutive_promotions >= self.PROMOTION_STREAK_THRESHOLD:
            streak_bonus = min(0.3, self.PROMOTION_STREAK_BONUS * self._state.consecutive_promotions)
            acceleration += streak_bonus
            signal = ImprovementSignal.PROMOTION_STREAK
            reason_parts.append(f"{self._state.consecutive_promotions} consecutive promotions")

        # Elo breakthrough bonus
        if elo_gain >= self.ELO_BREAKTHROUGH_THRESHOLD:
            acceleration += self.ELO_BREAKTHROUGH_BONUS
            signal = ImprovementSignal.ELO_BREAKTHROUGH
            reason_parts.append(f"Large Elo gain: +{elo_gain:.0f}")

        # Apply acceleration to threshold multiplier
        self._state.threshold_multiplier = max(
            self.MIN_THRESHOLD_MULTIPLIER,
            self._state.threshold_multiplier * (1.0 - acceleration)
        )

        # Boost this config specifically
        current_boost = self._state.config_boosts.get(config_key, 1.0)
        self._state.config_boosts[config_key] = max(
            self.MIN_THRESHOLD_MULTIPLIER,
            current_boost * (1.0 - acceleration * 0.5)  # Half the acceleration for config-specific
        )

        self._save_state()

        reason = f"Promotion succeeded (+{elo_gain:.0f} Elo). " + ", ".join(reason_parts) if reason_parts else f"Promotion succeeded (+{elo_gain:.0f} Elo)"

        rec = OptimizationRecommendation(
            signal=signal,
            config_key=config_key,
            threshold_adjustment=self._state.threshold_multiplier,
            evaluation_adjustment=self._state.evaluation_frequency_multiplier,
            reason=reason,
            confidence=min(1.0, 0.5 + acceleration),
            metadata={
                "elo_gain": elo_gain,
                "consecutive_promotions": self._state.consecutive_promotions,
                "model_id": model_id,
            },
        )

        self._emit_recommendation(rec)
        return rec

    def record_promotion_failure(self, config_key: str, reason: str = "") -> None:
        """Record a failed promotion attempt.

        Resets the promotion streak but doesn't dramatically slow down.
        """
        self._state.consecutive_promotions = 0

        # Slight slowdown on failure
        self._state.threshold_multiplier = min(
            self.MAX_THRESHOLD_MULTIPLIER,
            self._state.threshold_multiplier * 1.05  # 5% slower
        )

        # Reduce config-specific boost
        if config_key in self._state.config_boosts:
            self._state.config_boosts[config_key] = min(
                1.0, self._state.config_boosts[config_key] * 1.1  # 10% slower
            )

        self._save_state()

    def record_regression(
        self,
        config_key: str,
        regression_type: str = "unknown",
        severity: str = "moderate",
        elo_drop: float = 0.0,
        win_rate: float = 0.0,
    ) -> OptimizationRecommendation:
        """Record a detected model regression.

        December 2025: Added for REGRESSION_DETECTED event handling.
        Regression is a strong negative signal that slows down the pipeline
        to allow for investigation and recovery.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")
            regression_type: Type of regression (e.g., "elo_drop", "win_rate_decline")
            severity: Regression severity ("mild", "moderate", "severe")
            elo_drop: Amount of Elo lost (positive value)
            win_rate: Current win rate

        Returns:
            OptimizationRecommendation with adjusted thresholds
        """
        # Reset promotion streak on regression
        self._state.consecutive_promotions = 0

        # Calculate slowdown based on severity
        severity_multipliers = {
            "mild": 1.10,      # 10% slower
            "moderate": 1.25,  # 25% slower
            "severe": 1.50,    # 50% slower
        }
        slowdown = severity_multipliers.get(severity, 1.25)

        # Apply slowdown to threshold multiplier
        self._state.threshold_multiplier = min(
            self.MAX_THRESHOLD_MULTIPLIER,
            self._state.threshold_multiplier * slowdown
        )

        # Heavily reduce config-specific boost (regression is a strong signal)
        self._state.config_boosts[config_key] = min(
            1.0, self._state.config_boosts.get(config_key, 1.0) * slowdown
        )

        # Reduce evaluation frequency multiplier (evaluate more often to catch issues)
        self._state.evaluation_frequency_multiplier = max(
            0.5, self._state.evaluation_frequency_multiplier * 0.8
        )

        self._save_state()

        reason_parts = [
            f"Regression detected: {regression_type} ({severity})",
        ]
        if elo_drop > 0:
            reason_parts.append(f"Elo drop: {elo_drop:.1f}")
        if win_rate > 0:
            reason_parts.append(f"Win rate: {win_rate:.1%}")

        rec = OptimizationRecommendation(
            signal=ImprovementSignal.REGRESSION_DETECTED,
            config_key=config_key,
            threshold_adjustment=self._state.threshold_multiplier,
            evaluation_adjustment=self._state.evaluation_frequency_multiplier,
            reason=". ".join(reason_parts),
            confidence=0.9,  # High confidence in regression detection
            metadata={
                "regression_type": regression_type,
                "severity": severity,
                "elo_drop": elo_drop,
                "win_rate": win_rate,
                "slowdown_applied": slowdown,
            },
        )

        self._emit_recommendation(rec)
        logger.warning(
            f"[ImprovementOptimizer] Regression recorded for {config_key}: "
            f"{regression_type} ({severity}), slowdown={slowdown}x"
        )
        return rec

    def record_training_complete(
        self,
        config_key: str,
        duration_seconds: float,
        val_loss: float,
        calibration_ece: float | None = None,
    ) -> OptimizationRecommendation:
        """Record training completion.

        Good training results can accelerate future cycles.
        """
        now = time.time()
        self._cleanup_old_times()

        self._state.training_times.append(now)
        self._state.training_runs_24h = len(self._state.training_times)

        # Update average duration
        self._state.avg_training_duration_seconds = (
            self._state.avg_training_duration_seconds * 0.9 + duration_seconds * 0.1
        )

        # Track calibration if provided
        signal = ImprovementSignal.HEALTHY_PIPELINE
        acceleration = 0.0
        reason_parts = [f"Training completed in {duration_seconds/60:.1f}min"]

        if calibration_ece is not None:
            self._state.calibration_ece = calibration_ece
            if calibration_ece < self.CALIBRATION_EXCELLENT_THRESHOLD:
                signal = ImprovementSignal.CALIBRATION_EXCELLENT
                acceleration += 0.05
                reason_parts.append(f"Excellent calibration (ECE={calibration_ece:.3f})")

        # Apply any acceleration
        if acceleration > 0:
            self._state.threshold_multiplier = max(
                self.MIN_THRESHOLD_MULTIPLIER,
                self._state.threshold_multiplier * (1.0 - acceleration)
            )

        self._save_state()

        rec = OptimizationRecommendation(
            signal=signal,
            config_key=config_key,
            threshold_adjustment=self._state.threshold_multiplier,
            evaluation_adjustment=self._state.evaluation_frequency_multiplier,
            reason=". ".join(reason_parts),
            confidence=0.7,
            metadata={
                "duration_seconds": duration_seconds,
                "val_loss": val_loss,
                "calibration_ece": calibration_ece,
            },
        )

        self._emit_recommendation(rec)
        return rec

    def record_data_quality(
        self,
        parity_success_rate: float,
        data_quality_score: float,
    ) -> OptimizationRecommendation:
        """Record data quality metrics.

        High-quality data enables faster training cycles.
        """
        self._state.parity_success_rate = parity_success_rate
        self._state.data_quality_score = data_quality_score

        signal = ImprovementSignal.HEALTHY_PIPELINE
        acceleration = 0.0
        reason_parts = []

        if parity_success_rate >= self.QUALITY_DATA_THRESHOLD:
            signal = ImprovementSignal.QUALITY_DATA_SURGE
            acceleration += self.QUALITY_DATA_BONUS
            reason_parts.append(f"High parity success: {parity_success_rate:.1%}")

        if data_quality_score >= 0.95:
            acceleration += 0.05
            reason_parts.append(f"Excellent data quality: {data_quality_score:.1%}")

        if acceleration > 0:
            self._state.threshold_multiplier = max(
                self.MIN_THRESHOLD_MULTIPLIER,
                self._state.threshold_multiplier * (1.0 - acceleration)
            )

        self._save_state()

        rec = OptimizationRecommendation(
            signal=signal,
            config_key="global",
            threshold_adjustment=self._state.threshold_multiplier,
            evaluation_adjustment=self._state.evaluation_frequency_multiplier,
            reason=". ".join(reason_parts) if reason_parts else "Data quality acceptable",
            confidence=0.8 if acceleration > 0 else 0.5,
            metadata={
                "parity_success_rate": parity_success_rate,
                "data_quality_score": data_quality_score,
            },
        )

        self._emit_recommendation(rec)
        return rec

    def record_cluster_utilization(
        self,
        cpu_utilization: float,
        gpu_utilization: float,
    ) -> OptimizationRecommendation:
        """Record cluster utilization.

        Underutilization signals we can push harder.
        """
        signal = ImprovementSignal.HEALTHY_PIPELINE
        acceleration = 0.0
        reason_parts = []

        # Check if in optimal range
        cpu_optimal = 60 <= cpu_utilization <= 80
        gpu_optimal = gpu_utilization == 0 or 60 <= gpu_utilization <= 80

        if cpu_optimal and gpu_optimal:
            signal = ImprovementSignal.EFFICIENCY_OPTIMAL
            reason_parts.append(f"Optimal utilization (CPU={cpu_utilization:.0f}%, GPU={gpu_utilization:.0f}%)")
        elif cpu_utilization < 50 or (gpu_utilization > 0 and gpu_utilization < 50):
            # Underutilized - accelerate!
            signal = ImprovementSignal.UNDERUTILIZED_CAPACITY
            acceleration = 0.1
            reason_parts.append(f"Underutilized (CPU={cpu_utilization:.0f}%, GPU={gpu_utilization:.0f}%)")

            # Also speed up evaluations
            self._state.evaluation_frequency_multiplier = max(
                self.MIN_EVAL_MULTIPLIER,
                self._state.evaluation_frequency_multiplier * 0.9
            )

        if acceleration > 0:
            self._state.threshold_multiplier = max(
                self.MIN_THRESHOLD_MULTIPLIER,
                self._state.threshold_multiplier * (1.0 - acceleration)
            )

        self._save_state()

        rec = OptimizationRecommendation(
            signal=signal,
            config_key="global",
            threshold_adjustment=self._state.threshold_multiplier,
            evaluation_adjustment=self._state.evaluation_frequency_multiplier,
            reason=". ".join(reason_parts) if reason_parts else f"Utilization: CPU={cpu_utilization:.0f}%, GPU={gpu_utilization:.0f}%",
            confidence=0.9,
            metadata={
                "cpu_utilization": cpu_utilization,
                "gpu_utilization": gpu_utilization,
            },
        )

        self._emit_recommendation(rec)
        return rec

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_dynamic_threshold(self, config_key: str) -> int:
        """Get the dynamically adjusted training threshold for a config.

        Lower thresholds = faster training cycles = more iterations.
        """
        base_threshold = self.BASELINE_TRAINING_THRESHOLD

        # Apply global multiplier
        threshold = base_threshold * self._state.threshold_multiplier

        # Apply config-specific boost
        config_boost = self._state.config_boosts.get(config_key, 1.0)
        threshold *= config_boost

        # Ensure within bounds
        threshold = max(
            int(base_threshold * self.MIN_THRESHOLD_MULTIPLIER),
            min(int(base_threshold * self.MAX_THRESHOLD_MULTIPLIER), int(threshold))
        )

        return threshold

    def get_evaluation_interval(self, base_interval: int) -> int:
        """Get the dynamically adjusted evaluation interval.

        Lower intervals = more frequent evaluation = faster promotion.
        """
        interval = base_interval * self._state.evaluation_frequency_multiplier

        return max(
            int(base_interval * self.MIN_EVAL_MULTIPLIER),
            min(int(base_interval * self.MAX_EVAL_MULTIPLIER), int(interval))
        )

    def should_fast_track_training(self, config_key: str) -> bool:
        """Check if training should be fast-tracked for a config.

        Fast-tracking means using a lower threshold for faster iteration.
        """
        # Fast-track if:
        # 1. On a promotion streak
        # 2. Data quality is high
        # 3. Cluster is underutilized

        if self._state.consecutive_promotions >= 2:
            return True

        if self._state.parity_success_rate >= self.QUALITY_DATA_THRESHOLD:
            return True

        config_boost = self._state.config_boosts.get(config_key, 1.0)
        if config_boost < 0.8:  # Config has earned faster cycles
            return True

        return False

    def get_selfplay_priority_boost(self, config_key: str) -> float:
        """Get priority boost for selfplay based on improvement signals.

        When training is going well (promotion streak, high data quality),
        we should accelerate selfplay to feed more data into the successful pipeline.

        Args:
            config_key: Config identifier

        Returns:
            Priority boost value (-0.1 to +0.15):
            - Positive: Config is on a streak, deserves more selfplay attention
            - Negative: Config is underperforming, reduce selfplay focus
            - Zero: Neutral, no adjustment
        """
        boost = 0.0

        # Promotion streak bonus (+0.10 to +0.15)
        if self._state.consecutive_promotions >= 3:
            boost += 0.15
            logger.debug(f"[ImprovementOptimizer] {config_key}: +0.15 (3+ promotion streak)")
        elif self._state.consecutive_promotions >= 2:
            boost += 0.10
            logger.debug(f"[ImprovementOptimizer] {config_key}: +0.10 (2 promotion streak)")

        # Config-specific boost from past success
        config_boost = self._state.config_boosts.get(config_key, 1.0)
        if config_boost < 0.8:
            # This config has earned faster cycles, boost its selfplay too
            boost += 0.10
            logger.debug(f"[ImprovementOptimizer] {config_key}: +0.10 (earned faster cycles)")

        # High recent Elo gains (+0.05)
        if self._state.best_elo_gain > 50:
            boost += 0.05
            logger.debug(f"[ImprovementOptimizer] {config_key}: +0.05 (recent Elo breakthrough)")

        # Penalize if data quality is low (-0.10)
        if self._state.parity_success_rate < 0.8:
            boost -= 0.10
            logger.debug(f"[ImprovementOptimizer] {config_key}: -0.10 (low data quality)")

        return max(-0.10, min(0.15, boost))  # Clamp to reasonable range

    def get_improvement_metrics(self) -> dict[str, Any]:
        """Get metrics for monitoring improvement efficiency."""
        self._cleanup_old_times()

        return {
            # Success metrics
            "consecutive_promotions": self._state.consecutive_promotions,
            "promotions_24h": self._state.total_promotions_24h,
            "avg_elo_gain": self._state.avg_elo_gain_per_promotion,
            "best_elo_gain": self._state.best_elo_gain,

            # Throughput metrics
            "training_runs_24h": self._state.training_runs_24h,
            "avg_training_duration_min": self._state.avg_training_duration_seconds / 60,

            # Quality metrics
            "data_quality_score": self._state.data_quality_score,
            "parity_success_rate": self._state.parity_success_rate,
            "calibration_ece": self._state.calibration_ece,

            # Dynamic thresholds
            "threshold_multiplier": self._state.threshold_multiplier,
            "evaluation_multiplier": self._state.evaluation_frequency_multiplier,
            "effective_threshold": int(self.BASELINE_TRAINING_THRESHOLD * self._state.threshold_multiplier),

            # Config boosts
            "config_boosts": dict(self._state.config_boosts),
        }

    def get_training_adjustment(self, config_key: str = "") -> dict[str, Any]:
        """Get training hyperparameter adjustments based on optimizer state.

        This is the unified interface for training to query what adjustments
        to apply based on promotion success, evaluation results, and data quality.

        December 2025: Added for Phase 2 of self-improvement feedback loop.

        Args:
            config_key: Optional config for config-specific adjustments

        Returns:
            Dict with recommended adjustments:
            - lr_multiplier: Multiply current LR by this (1.0 = no change)
            - regularization_boost: Add this to weight decay (0.0 = no change)
            - batch_size_multiplier: Multiply batch size (1.0 = no change)
            - should_fast_track: Whether to use reduced epochs
            - reason: Human-readable explanation

        Example usage in train.py:
            adjustment = optimizer.get_training_adjustment("square8_2p")
            if adjustment["lr_multiplier"] != 1.0:
                for pg in optimizer.param_groups:
                    pg["lr"] *= adjustment["lr_multiplier"]
        """
        adjustment = {
            "lr_multiplier": 1.0,
            "regularization_boost": 0.0,
            "batch_size_multiplier": 1.0,
            "should_fast_track": False,
            "reason": "baseline",
            "threshold_multiplier": self._state.threshold_multiplier,
            "evaluation_multiplier": self._state.evaluation_frequency_multiplier,
        }

        reasons = []

        # Success acceleration: 3+ consecutive promotions → faster LR
        if self._state.consecutive_promotions >= self.PROMOTION_STREAK_THRESHOLD:
            # On a streak: slightly reduce LR to avoid overshooting
            adjustment["lr_multiplier"] *= 0.9
            adjustment["should_fast_track"] = True
            reasons.append(f"promotion_streak_{self._state.consecutive_promotions}")

        # Elo breakthrough: Recent large gain → maintain momentum
        recent_gains = [g for g in self._state.elo_gains[-5:] if g >= 50]
        if recent_gains:
            # Large gains suggest current hyperparams are good
            adjustment["lr_multiplier"] *= 0.95
            reasons.append(f"elo_breakthrough_+{max(recent_gains):.0f}")

        # Data quality issues: Low quality → increase regularization
        if self._state.data_quality_score < 0.7:
            adjustment["regularization_boost"] = 0.0005
            adjustment["lr_multiplier"] *= 1.1  # Slower learning on noisy data
            reasons.append("low_data_quality")

        # Parity failures: Increase regularization, reduce LR
        if self._state.parity_success_rate < 0.9:
            adjustment["regularization_boost"] += 0.0002
            adjustment["lr_multiplier"] *= 1.05
            reasons.append(f"parity_{self._state.parity_success_rate:.0%}")

        # Calibration issues: High ECE suggests overconfidence
        if self._state.calibration_ece > 0.15:
            adjustment["regularization_boost"] += 0.0003
            reasons.append(f"calibration_ece_{self._state.calibration_ece:.2f}")

        # Config-specific boost from curriculum feedback
        config_boost = self._state.config_boosts.get(config_key, 0.0)
        if config_boost != 0.0:
            # Positive boost = config needs more training → normal LR
            # Negative boost = config doing well → slightly reduce LR
            if config_boost < 0:
                adjustment["lr_multiplier"] *= 0.95
            reasons.append(f"config_boost_{config_boost:+.2f}")

        # Apply threshold multiplier (from promotion success)
        if self._state.threshold_multiplier < 0.8:
            # We're accelerating: use faster training settings
            adjustment["should_fast_track"] = True
            adjustment["batch_size_multiplier"] = 1.25  # Larger batches for speed
            reasons.append("accelerated_threshold")

        adjustment["reason"] = ", ".join(reasons) if reasons else "baseline"
        return adjustment

    def get_recommendations(self) -> list[OptimizationRecommendation]:
        """Get current optimization recommendations based on state."""
        recommendations = []

        # Check for promotion streak opportunity
        if self._state.consecutive_promotions >= self.PROMOTION_STREAK_THRESHOLD:
            recommendations.append(OptimizationRecommendation(
                signal=ImprovementSignal.PROMOTION_STREAK,
                config_key="global",
                threshold_adjustment=self._state.threshold_multiplier,
                evaluation_adjustment=self._state.evaluation_frequency_multiplier,
                reason=f"On {self._state.consecutive_promotions}-promotion streak, pushing harder",
                confidence=0.9,
            ))

        # Check for underutilization
        if self._state.threshold_multiplier > 0.8:  # Not already accelerated
            recommendations.append(OptimizationRecommendation(
                signal=ImprovementSignal.OPPORTUNITY_AVAILABLE if self._state.data_quality_score > 0.9 else ImprovementSignal.HEALTHY_PIPELINE,
                config_key="global",
                threshold_adjustment=0.9,  # Suggest 10% acceleration
                evaluation_adjustment=0.9,
                reason="Data quality high, consider accelerating",
                confidence=0.6,
            ))

        return recommendations

    def _emit_recommendation(self, rec: OptimizationRecommendation) -> None:
        """Emit recommendation to callbacks."""
        for callback in self._callbacks:
            try:
                callback(rec)
            except Exception as e:
                logger.warning(f"[ImprovementOptimizer] Callback error: {e}")


# Add missing signal
ImprovementSignal.OPPORTUNITY_AVAILABLE = "opportunity_available"


# =============================================================================
# Module-level convenience functions
# =============================================================================

_optimizer: ImprovementOptimizer | None = None


def get_improvement_optimizer() -> ImprovementOptimizer:
    """Get the singleton improvement optimizer."""
    return ImprovementOptimizer.get_instance()


def should_fast_track_training(config_key: str) -> bool:
    """Check if training should be fast-tracked."""
    return get_improvement_optimizer().should_fast_track_training(config_key)


def get_dynamic_threshold(config_key: str) -> int:
    """Get dynamically adjusted training threshold."""
    return get_improvement_optimizer().get_dynamic_threshold(config_key)


def get_evaluation_interval(base_interval: int) -> int:
    """Get dynamically adjusted evaluation interval."""
    return get_improvement_optimizer().get_evaluation_interval(base_interval)


def record_promotion_success(config_key: str, elo_gain: float, model_id: str = "") -> None:
    """Record a successful promotion."""
    get_improvement_optimizer().record_promotion_success(config_key, elo_gain, model_id)


def record_training_complete(
    config_key: str,
    duration_seconds: float,
    val_loss: float,
    calibration_ece: float | None = None,
) -> None:
    """Record training completion."""
    get_improvement_optimizer().record_training_complete(
        config_key, duration_seconds, val_loss, calibration_ece
    )


def get_improvement_metrics() -> dict[str, Any]:
    """Get improvement efficiency metrics."""
    return get_improvement_optimizer().get_improvement_metrics()


def get_training_adjustment(config_key: str = "") -> dict[str, Any]:
    """Get training hyperparameter adjustments based on optimizer state.

    December 2025: Added for Phase 2 self-improvement feedback loop.

    Returns:
        Dict with recommended adjustments (lr_multiplier, regularization_boost, etc.)
    """
    return get_improvement_optimizer().get_training_adjustment(config_key)


def get_selfplay_priority_boost(config_key: str) -> float:
    """Get selfplay priority boost based on improvement signals.

    Returns a value from -0.10 to +0.15 that should be added to
    the selfplay priority score for this config.
    """
    return get_improvement_optimizer().get_selfplay_priority_boost(config_key)


def reset_improvement_optimizer() -> None:
    """Reset the optimizer singleton."""
    ImprovementOptimizer.reset_instance()
